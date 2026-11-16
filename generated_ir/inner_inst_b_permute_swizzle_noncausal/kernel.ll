; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/async_task_noncausal/input.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/async_task_noncausal/input.mlir"

@shm_0 = addrspace(3) global [16384 x i8] undef, align 16

; Function Attrs: convergent willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

declare float @__ocml_exp2_f32(float)

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3) #1 !dbg !4 !reqd_work_group_size !30 {
  %kdma.tid = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !6
  %kdma.wave = and i32 %kdma.tid, 960, !dbg !6
  %kdma.lane = lshr i32 %kdma.tid, 1, !dbg !6
  %5 = and i32 %kdma.lane, 31, !dbg !6
  %kdma.owner = or disjoint i32 %kdma.wave, %5, !dbg !6
  %kdma.owner.hi = or disjoint i32 %kdma.owner, 32, !dbg !6
  %6 = shl nuw nsw i32 %kdma.tid, 3, !dbg !6
  %kdma.intra = and i32 %6, 8, !dbg !6
  %kdma.intra.hi = xor i32 %kdma.intra, 8, !dbg !6
  %7 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !6
  %8 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %7), !dbg !6
  %9 = add nuw nsw i32 %8, 64, !dbg !6
  %10 = xor i32 %8, 16, !dbg !6
  %11 = and i32 %9, 192, !dbg !6
  %12 = icmp samesign ult i32 %10, %11, !dbg !6
  %13 = select i1 %12, i32 %10, i32 %8, !dbg !6
  %14 = shl nuw nsw i32 %13, 2, !dbg !6
  %15 = xor i32 %8, 32, !dbg !6
  %16 = icmp samesign ult i32 %15, %11, !dbg !6
  %17 = select i1 %16, i32 %15, i32 %8, !dbg !6
  %18 = shl nuw nsw i32 %17, 2, !dbg !6
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
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #9, !dbg !9, !range !32, !frisk.register_lifetimes !1
  %address.local = zext nneg i32 %address.thread to i64, !dbg !9
  %address.local316 = shl nuw nsw i64 %address.local, 3, !dbg !9
  %address.local317 = and i64 %address.local316, 1536, !dbg !9
  %address.local318 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local317, !dbg !9
  %address.local319 = ptrtoint ptr addrspace(3) %address.local318 to i64, !dbg !9
  %address.local320 = trunc i64 %address.local319 to i32, !dbg !9
  %36 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local320), !dbg !9
  %address.local321 = shl nuw nsw i64 %address.local, 7, !dbg !9
  %address.local322 = and i64 %address.local321, 16256, !dbg !9
  %address.local323 = lshr i64 %address.local, 4, !dbg !9
  %address.local324 = and i64 %address.local323, 8, !dbg !9
  %address.local325 = or disjoint i64 %address.local322, %address.local324, !dbg !9
  %37 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !9
  %.idx1006 = shl i64 %34, 8, !dbg !9
  %38 = add i64 %.idx, %.idx1006, !dbg !9
  %address.local326.idx = shl nuw nsw i64 %address.local325, 1, !dbg !9
  %39 = or disjoint i64 %38, %address.local326.idx, !dbg !9
  %dma.buffer.offset1004 = trunc i64 %39 to i32, !dbg !9
  %40 = insertelement <2 x i64> poison, i64 %37, i64 0, !dbg !9
  %41 = bitcast <2 x i64> %40 to <4 x i32>, !dbg !9
  %dma.buffer.rsrc1005 = shufflevector <4 x i32> %41, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !9
  %42 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %36, i32 %dma.buffer.offset1004, <4 x i32> %dma.buffer.rsrc1005) #10, !dbg !9
  %address.local327 = or disjoint i64 %address.local, 256, !dbg !9
  %address.local328 = shl nuw nsw i64 %address.local327, 3, !dbg !9
  %address.local329 = and i64 %address.local328, 3584, !dbg !9
  %address.local330 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local329, !dbg !9
  %address.local331 = ptrtoint ptr addrspace(3) %address.local330 to i64, !dbg !9
  %address.local332 = trunc i64 %address.local331 to i32, !dbg !9
  %43 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local332), !dbg !9
  %address.local335 = lshr i64 %address.local327, 4, !dbg !9
  %address.local336 = and i64 %address.local335, 24, !dbg !9
  %address.local337 = or disjoint i64 %address.local322, %address.local336, !dbg !9
  %address.local338.idx = shl nuw nsw i64 %address.local337, 1, !dbg !9
  %44 = or disjoint i64 %38, %address.local338.idx, !dbg !9
  %dma.buffer.offset1002 = trunc i64 %44 to i32, !dbg !9
  %45 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %43, i32 %dma.buffer.offset1002, <4 x i32> %dma.buffer.rsrc1005) #10, !dbg !9
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %46 = shl nuw nsw i64 %31, 19, !dbg !9
  %47 = add i64 %46, %35, !dbg !9
  %address.local339 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local317, !dbg !9
  %address.local340 = ptrtoint ptr addrspace(3) %address.local339 to i64, !dbg !9
  %address.local341 = trunc i64 %address.local340 to i32, !dbg !9
  %48 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local341), !dbg !9
  %49 = or disjoint i64 %47, 32, !dbg !9
  %50 = or disjoint i64 %address.local325, %49, !dbg !9
  %.tr = trunc i64 %50 to i32, !dbg !9
  %dma.buffer.offset1000 = shl i32 %.tr, 1, !dbg !9
  %51 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %48, i32 %dma.buffer.offset1000, <4 x i32> %dma.buffer.rsrc1005) #10, !dbg !9
  %address.local343 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local329, !dbg !9
  %address.local344 = ptrtoint ptr addrspace(3) %address.local343 to i64, !dbg !9
  %address.local345 = trunc i64 %address.local344 to i32, !dbg !9
  %52 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local345), !dbg !9
  %53 = or disjoint i64 %address.local337, %49, !dbg !9
  %.tr1010 = trunc i64 %53 to i32, !dbg !9
  %dma.buffer.offset998 = shl i32 %.tr1010, 1, !dbg !9
  %54 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %52, i32 %dma.buffer.offset998, <4 x i32> %dma.buffer.rsrc1005) #10, !dbg !9
  %address.local347 = lshr i32 %address.thread, 2, !dbg !12
  %address.local348 = and i32 %address.local347, 4, !dbg !12
  %address.local349 = shl nuw nsw i32 %address.thread, 1, !dbg !12
  %address.local350 = and i32 %address.local349, 384, !dbg !12
  %address.local351 = or disjoint i32 %address.local348, %address.local350, !dbg !12
  %address.local352 = shl nuw nsw i32 %address.thread, 3, !dbg !12
  %address.local353 = and i32 %address.local352, 120, !dbg !12
  %address.local354 = or disjoint i32 %address.local351, %address.local353, !dbg !12
  %address.local355 = shl nuw nsw i32 %address.thread, 5, !dbg !12
  %address.local356 = and i32 %address.local355, 1024, !dbg !12
  %address.local357 = or disjoint i32 %address.local354, %address.local356, !dbg !12
  %address.local358 = zext nneg i32 %address.local357 to i64, !dbg !12
  %address.local359 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local358, !dbg !12
  %55 = load <4 x half>, ptr addrspace(3) %address.local359, align 8, !dbg !12
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local359, i64 4096, !dbg !12
  %56 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !12
  %57 = fmul <4 x half> %55, splat (half 0xH3015), !dbg !12
  %lds.const.gep827 = getelementptr i8, ptr addrspace(3) %address.local359, i64 1024, !dbg !12
  %58 = load <4 x half>, ptr addrspace(3) %lds.const.gep827, align 8, !dbg !12
  %59 = fmul <4 x half> %56, splat (half 0xH3015), !dbg !12
  %lds.const.gep828 = getelementptr i8, ptr addrspace(3) %address.local359, i64 5120, !dbg !12
  %60 = load <4 x half>, ptr addrspace(3) %lds.const.gep828, align 8, !dbg !12
  %61 = fmul <4 x half> %58, splat (half 0xH3015), !dbg !12
  %62 = fmul <4 x half> %60, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %63 = or disjoint i64 %47, 64, !dbg !9
  %64 = or disjoint i64 %address.local325, %63, !dbg !9
  %.tr1012 = trunc i64 %64 to i32, !dbg !9
  %dma.buffer.offset996 = shl i32 %.tr1012, 1, !dbg !9
  %65 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %36, i32 %dma.buffer.offset996, <4 x i32> %dma.buffer.rsrc1005) #10, !dbg !9
  %66 = or disjoint i64 %address.local337, %63, !dbg !9
  %.tr1014 = trunc i64 %66 to i32, !dbg !9
  %dma.buffer.offset994 = shl i32 %.tr1014, 1, !dbg !9
  %67 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %43, i32 %dma.buffer.offset994, <4 x i32> %dma.buffer.rsrc1005) #10, !dbg !9
  %lds.const.gep829 = getelementptr i8, ptr addrspace(3) %address.local359, i64 8192, !dbg !12
  %68 = load <4 x half>, ptr addrspace(3) %lds.const.gep829, align 8, !dbg !12
  %lds.const.gep830 = getelementptr i8, ptr addrspace(3) %address.local359, i64 12288, !dbg !12
  %69 = load <4 x half>, ptr addrspace(3) %lds.const.gep830, align 8, !dbg !12
  %lds.const.gep831 = getelementptr i8, ptr addrspace(3) %address.local359, i64 9216, !dbg !12
  %70 = load <4 x half>, ptr addrspace(3) %lds.const.gep831, align 8, !dbg !12
  %71 = fmul <4 x half> %68, splat (half 0xH3015), !dbg !12
  %lds.const.gep832 = getelementptr i8, ptr addrspace(3) %address.local359, i64 13312, !dbg !12
  %72 = load <4 x half>, ptr addrspace(3) %lds.const.gep832, align 8, !dbg !12
  %73 = fmul <4 x half> %69, splat (half 0xH3015), !dbg !12
  %74 = fmul <4 x half> %70, splat (half 0xH3015), !dbg !12
  %75 = fmul <4 x half> %72, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %76 = or disjoint i64 %47, 96, !dbg !9
  %77 = or disjoint i64 %address.local325, %76, !dbg !9
  %.tr1016 = trunc i64 %77 to i32, !dbg !9
  %dma.buffer.offset992 = shl i32 %.tr1016, 1, !dbg !9
  %78 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %48, i32 %dma.buffer.offset992, <4 x i32> %dma.buffer.rsrc1005) #10, !dbg !9
  %79 = or disjoint i64 %address.local337, %76, !dbg !9
  %.tr1018 = trunc i64 %79 to i32, !dbg !9
  %dma.buffer.offset990 = shl i32 %.tr1018, 1, !dbg !9
  %80 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %52, i32 %dma.buffer.offset990, <4 x i32> %dma.buffer.rsrc1005) #10, !dbg !9
  %81 = load <4 x half>, ptr addrspace(3) %address.local359, align 8, !dbg !12
  %82 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !12
  %83 = load <4 x half>, ptr addrspace(3) %lds.const.gep827, align 8, !dbg !12
  %84 = fmul <4 x half> %81, splat (half 0xH3015), !dbg !12
  %85 = load <4 x half>, ptr addrspace(3) %lds.const.gep828, align 8, !dbg !12
  %86 = fmul <4 x half> %82, splat (half 0xH3015), !dbg !12
  %87 = fmul <4 x half> %83, splat (half 0xH3015), !dbg !12
  %88 = fmul <4 x half> %85, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %89 = load <4 x half>, ptr addrspace(3) %lds.const.gep829, align 8, !dbg !12
  %90 = load <4 x half>, ptr addrspace(3) %lds.const.gep830, align 8, !dbg !12
  %91 = load <4 x half>, ptr addrspace(3) %lds.const.gep831, align 8, !dbg !12
  %92 = fmul <4 x half> %89, splat (half 0xH3015), !dbg !12
  %93 = load <4 x half>, ptr addrspace(3) %lds.const.gep832, align 8, !dbg !12
  %94 = fmul <4 x half> %90, splat (half 0xH3015), !dbg !12
  %95 = fmul <4 x half> %91, splat (half 0xH3015), !dbg !12
  %96 = fmul <4 x half> %93, splat (half 0xH3015), !dbg !12
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %k0.prime. = shl i64 %25, 19, !dbg !13
  %k0.prime.1567 = and i64 %k0.prime., 16252928, !dbg !13
  %k0.prime.1568 = shl nuw nsw i64 0, 14, !dbg !13
  %k0.prime.1569 = or disjoint i64 %k0.prime.1567, %k0.prime.1568, !dbg !13
  %k0.prime.kdma.source = zext nneg i32 %kdma.owner to i64, !dbg !13
  %k0.prime.kdma.source1110 = lshr i64 %k0.prime.kdma.source, 4, !dbg !13
  %k0.prime.kdma.source1111 = and i64 %k0.prime.kdma.source1110, 8, !dbg !13
  %k0.prime.kdma.source1112 = and i64 %k0.prime.kdma.source, 95, !dbg !13
  %k0.prime.kdma.source1113 = shl nuw nsw i64 %k0.prime.kdma.source, 2, !dbg !13
  %k0.prime.kdma.source1114 = and i64 %k0.prime.kdma.source1113, 32, !dbg !13
  %k0.prime.kdma.source1117 = or disjoint i64 %k0.prime.kdma.source1112, %k0.prime.kdma.source1114, !dbg !13
  %k0.prime.kdma.source1118 = shl nuw nsw i64 %k0.prime.kdma.source1117, 7, !dbg !13
  %k0.prime.kdma.source1119 = or disjoint i64 %k0.prime.kdma.source1111, %k0.prime.kdma.source1118, !dbg !13
  %k0.prime.kdma.source1120 = or disjoint i64 %k0.prime.1569, %k0.prime.kdma.source1119, !dbg !13
  %k0.prime.kdma.source1121 = trunc nuw i64 %k0.prime.kdma.source1120 to i32, !dbg !13
  %k0.prime.kdma.source1122 = shl nuw nsw i32 %k0.prime.kdma.source1121, 1, !dbg !13
  %k0.prime.kdma.offset = or disjoint i32 %k0.prime.kdma.source1122, %kdma.intra, !dbg !13
  %k0.prime.1570 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !13
  %k0.prime.1571 = insertelement <2 x i64> poison, i64 %k0.prime.1570, i64 0, !dbg !13
  %k0.prime.1572 = bitcast <2 x i64> %k0.prime.1571 to <4 x i32>, !dbg !13
  %k0.prime.dma.buffer.rsrc989 = shufflevector <4 x i32> %k0.prime.1572, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !13
  %k0.prime.1573 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %36, i32 %k0.prime.kdma.offset, <4 x i32> %k0.prime.dma.buffer.rsrc989) #10, !dbg !13, !frisk.cross_bn_k0 !1
  %k0.prime.kdma.lds1138 = add i32 %36, 512, !dbg !13
  %k0.prime.kdma.source1123 = zext nneg i32 %kdma.owner.hi to i64, !dbg !13
  %k0.prime.kdma.source1124 = lshr i64 %k0.prime.kdma.source1123, 4, !dbg !13
  %k0.prime.kdma.source1125 = and i64 %k0.prime.kdma.source1124, 8, !dbg !13
  %k0.prime.kdma.source1126 = and i64 %k0.prime.kdma.source1123, 95, !dbg !13
  %k0.prime.kdma.source1127 = shl nuw nsw i64 %k0.prime.kdma.source1123, 2, !dbg !13
  %k0.prime.kdma.source1128 = and i64 %k0.prime.kdma.source1127, 32, !dbg !13
  %k0.prime.kdma.source1129 = add nuw nsw i64 %k0.prime.kdma.source1128, %k0.prime.kdma.source1123, !dbg !13
  %k0.prime.kdma.source1130 = and i64 %k0.prime.kdma.source1129, 32, !dbg !13
  %k0.prime.kdma.source1131 = or disjoint i64 %k0.prime.kdma.source1126, %k0.prime.kdma.source1130, !dbg !13
  %k0.prime.kdma.source1132 = shl nuw nsw i64 %k0.prime.kdma.source1131, 7, !dbg !13
  %k0.prime.kdma.source1133 = or disjoint i64 %k0.prime.kdma.source1125, %k0.prime.kdma.source1132, !dbg !13
  %k0.prime.kdma.source1134 = or disjoint i64 %k0.prime.1569, %k0.prime.kdma.source1133, !dbg !13
  %k0.prime.kdma.source1135 = trunc nuw i64 %k0.prime.kdma.source1134 to i32, !dbg !13
  %k0.prime.kdma.source1136 = shl nuw nsw i32 %k0.prime.kdma.source1135, 1, !dbg !13
  %k0.prime.kdma.offset1137 = or disjoint i32 %k0.prime.kdma.source1136, %kdma.intra.hi, !dbg !13
  %k0.prime.1574 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.kdma.lds1138, i32 %k0.prime.kdma.offset1137, <4 x i32> %k0.prime.dma.buffer.rsrc989) #10, !dbg !13, !frisk.cross_bn_k0 !1
  %k0.prime.kdma.source1150 = or disjoint i64 %k0.prime.kdma.source1119, 16, !dbg !13
  %k0.prime.kdma.source1151 = or disjoint i64 %k0.prime.1569, %k0.prime.kdma.source1150, !dbg !13
  %k0.prime.kdma.source1152 = trunc nuw i64 %k0.prime.kdma.source1151 to i32, !dbg !13
  %k0.prime.kdma.source1153 = shl nuw nsw i32 %k0.prime.kdma.source1152, 1, !dbg !13
  %k0.prime.kdma.offset1154 = or disjoint i32 %k0.prime.kdma.source1153, %kdma.intra, !dbg !13
  %k0.prime.1575 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %43, i32 %k0.prime.kdma.offset1154, <4 x i32> %k0.prime.dma.buffer.rsrc989) #10, !dbg !13, !frisk.cross_bn_k0 !1
  %k0.prime.kdma.lds1172 = add i32 %43, 512, !dbg !13
  %k0.prime.kdma.source1167 = or disjoint i64 %k0.prime.kdma.source1133, 16, !dbg !13
  %k0.prime.kdma.source1168 = or disjoint i64 %k0.prime.1569, %k0.prime.kdma.source1167, !dbg !13
  %k0.prime.kdma.source1169 = trunc nuw i64 %k0.prime.kdma.source1168 to i32, !dbg !13
  %k0.prime.kdma.source1170 = shl nuw nsw i32 %k0.prime.kdma.source1169, 1, !dbg !13
  %k0.prime.kdma.offset1171 = or disjoint i32 %k0.prime.kdma.source1170, %kdma.intra.hi, !dbg !13
  %k0.prime.1576 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.kdma.lds1172, i32 %k0.prime.kdma.offset1171, <4 x i32> %k0.prime.dma.buffer.rsrc989) #10, !dbg !13, !frisk.cross_bn_k0 !1
  br label %97, !dbg !8

97:                                               ; preds = %k0.join, %4
  %98 = phi i64 [ %912, %k0.join ], [ 0, %4 ], !dbg !8
  %99 = phi [2 x <8 x float>] [ %990, %k0.join ], [ zeroinitializer, %4 ], !dbg !8
  %100 = phi [2 x <1 x float>] [ %755, %k0.join ], [ zeroinitializer, %4 ], !dbg !8
  %101 = phi [2 x <1 x float>] [ %992, %k0.join ], [ [<1 x float> splat (float 0xFFF0000000000000), <1 x float> splat (float 0xFFF0000000000000)], %4 ], !dbg !8
  %102 = phi [2 x <8 x float>] [ %996, %k0.join ], [ zeroinitializer, %4 ], !dbg !8
  %103 = phi [2 x <8 x float>] [ %1000, %k0.join ], [ zeroinitializer, %4 ], !dbg !8
  %104 = phi [2 x <8 x float>] [ %1004, %k0.join ], [ zeroinitializer, %4 ], !dbg !8
  %105 = icmp samesign ult i64 %98, 32, !dbg !8
  br i1 %105, label %106, label %1005, !dbg !8

106:                                              ; preds = %97
  %107 = shl i64 %25, 19, !dbg !13
  %108 = and i64 %107, 16252928, !dbg !13
  %109 = shl nuw nsw i64 %98, 14, !dbg !13
  %110 = or disjoint i64 %108, %109, !dbg !13
  %address.thread410 = call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #9, !dbg !13, !range !32, !frisk.register_lifetimes !1
  %address.local411 = zext nneg i32 %address.thread410 to i64, !dbg !9
  %111 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !13
  %112 = insertelement <2 x i64> poison, i64 %111, i64 0, !dbg !13
  %113 = bitcast <2 x i64> %112 to <4 x i32>, !dbg !13
  %dma.buffer.rsrc989 = shufflevector <4 x i32> %113, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !13
  %kdma.source = zext nneg i32 %kdma.owner to i64, !dbg !13
  %kdma.source1110 = lshr i64 %kdma.source, 4, !dbg !13
  %kdma.source1111 = and i64 %kdma.source1110, 8, !dbg !13
  %kdma.source1112 = and i64 %kdma.source, 95, !dbg !13
  %kdma.source1113 = shl nuw nsw i64 %kdma.source, 2, !dbg !13
  %kdma.source1114 = and i64 %kdma.source1113, 32, !dbg !13
  %kdma.source1117 = or disjoint i64 %kdma.source1112, %kdma.source1114, !dbg !13
  %kdma.source1118 = shl nuw nsw i64 %kdma.source1117, 7, !dbg !13
  %kdma.source1119 = or disjoint i64 %kdma.source1111, %kdma.source1118, !dbg !13
  %kdma.source1120 = or disjoint i64 %110, %kdma.source1119, !dbg !13
  %kdma.source1121 = trunc nuw i64 %kdma.source1120 to i32, !dbg !13
  %kdma.source1122 = shl nuw nsw i32 %kdma.source1121, 1, !dbg !13
  %kdma.offset = or disjoint i32 %kdma.source1122, %kdma.intra, !dbg !13
  %kdma.source1123 = zext nneg i32 %kdma.owner.hi to i64, !dbg !13
  %kdma.source1124 = lshr i64 %kdma.source1123, 4, !dbg !13
  %kdma.source1125 = and i64 %kdma.source1124, 8, !dbg !13
  %kdma.source1126 = and i64 %kdma.source1123, 95, !dbg !13
  %kdma.source1127 = shl nuw nsw i64 %kdma.source1123, 2, !dbg !13
  %kdma.source1128 = and i64 %kdma.source1127, 32, !dbg !13
  %kdma.source1129 = add nuw nsw i64 %kdma.source1128, %kdma.source1123, !dbg !13
  %kdma.source1130 = and i64 %kdma.source1129, 32, !dbg !13
  %kdma.source1131 = or disjoint i64 %kdma.source1126, %kdma.source1130, !dbg !13
  %kdma.source1132 = shl nuw nsw i64 %kdma.source1131, 7, !dbg !13
  %kdma.source1133 = or disjoint i64 %kdma.source1125, %kdma.source1132, !dbg !13
  %kdma.source1134 = or disjoint i64 %110, %kdma.source1133, !dbg !13
  %kdma.source1135 = trunc nuw i64 %kdma.source1134 to i32, !dbg !13
  %kdma.source1136 = shl nuw nsw i32 %kdma.source1135, 1, !dbg !13
  %kdma.offset1137 = or disjoint i32 %kdma.source1136, %kdma.intra.hi, !dbg !13
  %kdma.lds1138 = add i32 %36, 512, !dbg !13
  %kdma.source1150 = or disjoint i64 %kdma.source1119, 16, !dbg !13
  %kdma.source1151 = or disjoint i64 %110, %kdma.source1150, !dbg !13
  %kdma.source1152 = trunc nuw i64 %kdma.source1151 to i32, !dbg !13
  %kdma.source1153 = shl nuw nsw i32 %kdma.source1152, 1, !dbg !13
  %kdma.offset1154 = or disjoint i32 %kdma.source1153, %kdma.intra, !dbg !13
  %kdma.source1167 = or disjoint i64 %kdma.source1133, 16, !dbg !13
  %kdma.source1168 = or disjoint i64 %110, %kdma.source1167, !dbg !13
  %kdma.source1169 = trunc nuw i64 %kdma.source1168 to i32, !dbg !13
  %kdma.source1170 = shl nuw nsw i32 %kdma.source1169, 1, !dbg !13
  %kdma.offset1171 = or disjoint i32 %kdma.source1170, %kdma.intra.hi, !dbg !13
  %kdma.lds1172 = add i32 %43, 512, !dbg !13
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %114 = or disjoint i64 %110, 32, !dbg !13
  %kdma.source1184 = or disjoint i64 %114, %kdma.source1119, !dbg !13
  %kdma.source1185 = trunc nuw i64 %kdma.source1184 to i32, !dbg !13
  %kdma.source1186 = shl nuw nsw i32 %kdma.source1185, 1, !dbg !13
  %kdma.offset1187 = or disjoint i32 %kdma.source1186, %kdma.intra, !dbg !13
  %115 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %48, i32 %kdma.offset1187, <4 x i32> %dma.buffer.rsrc989) #10, !dbg !13
  %kdma.source1200 = or disjoint i64 %114, %kdma.source1133, !dbg !13
  %kdma.source1201 = trunc nuw i64 %kdma.source1200 to i32, !dbg !13
  %kdma.source1202 = shl nuw nsw i32 %kdma.source1201, 1, !dbg !13
  %kdma.offset1203 = or disjoint i32 %kdma.source1202, %kdma.intra.hi, !dbg !13
  %kdma.lds1204 = add i32 %48, 512, !dbg !13
  %116 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1204, i32 %kdma.offset1203, <4 x i32> %dma.buffer.rsrc989) #10, !dbg !13
  %kdma.source1217 = or disjoint i64 %114, %kdma.source1150, !dbg !13
  %kdma.source1218 = trunc nuw i64 %kdma.source1217 to i32, !dbg !13
  %kdma.source1219 = shl nuw nsw i32 %kdma.source1218, 1, !dbg !13
  %kdma.offset1220 = or disjoint i32 %kdma.source1219, %kdma.intra, !dbg !13
  %117 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %52, i32 %kdma.offset1220, <4 x i32> %dma.buffer.rsrc989) #10, !dbg !13
  %kdma.source1234 = or disjoint i64 %114, %kdma.source1167, !dbg !13
  %kdma.source1235 = trunc nuw i64 %kdma.source1234 to i32, !dbg !13
  %kdma.source1236 = shl nuw nsw i32 %kdma.source1235, 1, !dbg !13
  %kdma.offset1237 = or disjoint i32 %kdma.source1236, %kdma.intra.hi, !dbg !13
  %kdma.lds1238 = add i32 %52, 512, !dbg !13
  %118 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1238, i32 %kdma.offset1237, <4 x i32> %dma.buffer.rsrc989) #10, !dbg !13
  %address.local437 = lshr i64 %address.local411, 2, !dbg !12
  %address.local438 = and i64 %address.local437, 12, !dbg !12
  %address.local439 = shl nuw nsw i32 %address.thread410, 1, !dbg !15
  %address.local440 = and i32 %address.local439, 24, !dbg !15
  %address.local441 = shl nuw nsw i32 %address.thread410, 5, !dbg !15
  %address.local442 = and i32 %address.local441, 96, !dbg !15
  %address.local443 = or disjoint i32 %address.local440, %address.local442, !dbg !15
  %address.local444 = shl nuw nsw i32 %address.thread410, 7, !dbg !15
  %address.local445 = and i32 %address.local444, 256, !dbg !15
  %address.local446 = or disjoint i32 %address.local443, %address.local445, !dbg !15
  %address.local447 = zext nneg i32 %address.local446 to i64, !dbg !15
  %address.local449 = and i64 %address.local411, 32, !dbg !12
  %address.local450 = icmp eq i64 %address.local449, 0, !dbg !12
  %address.local451 = select i1 %address.local450, i64 0, i64 1016, !dbg !12
  %119 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local438, !dbg !15
  %120 = getelementptr [2 x i8], ptr addrspace(3) %119, i64 %address.local447, !dbg !15
  %address.local453 = getelementptr [2 x i8], ptr addrspace(3) %120, i64 %address.local451, !dbg !15
  %.idx14971499 = add nuw nsw i64 %address.local438, %address.local447, !dbg !15
  %121 = add nuw nsw i64 %.idx14971499, %address.local451, !dbg !15
  %122 = shl nuw nsw i64 %121, 1, !dbg !15
  %123 = lshr i64 %121, 5, !dbg !15
  %124 = and i64 %123, 8, !dbg !15
  %125 = xor i64 %122, %124, !dbg !15
  %126 = sub nsw i64 %125, %122, !dbg !15
  %kdma.swizzled = getelementptr i8, ptr addrspace(3) %address.local453, i64 %126, !dbg !15
  %127 = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8, !dbg !15, !frisk.k_lds_pair !35
  %kdma.gep1371 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 256, !dbg !15
  %128 = load <4 x half>, ptr addrspace(3) %kdma.gep1371, align 8, !dbg !15, !frisk.k_lds_pair !35
  %kdma.gep = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4096, !dbg !15
  %129 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8, !dbg !15, !frisk.k_lds_pair !36
  %kdma.gep1372 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4352, !dbg !15
  %130 = load <4 x half>, ptr addrspace(3) %kdma.gep1372, align 8, !dbg !15, !frisk.k_lds_pair !36
  %address.local478 = shl nuw nsw i32 %address.thread410, 8, !dbg !15
  %address.local479 = and i32 %address.local478, 512, !dbg !15
  %address.local480 = zext nneg i32 %address.local479 to i64, !dbg !15
  %address.local481 = sub nsw i64 %address.local447, %address.local480, !dbg !15
  %131 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local481, !dbg !15
  %132 = getelementptr [2 x i8], ptr addrspace(3) %131, i64 %address.local438, !dbg !15
  %133 = getelementptr [2 x i8], ptr addrspace(3) %132, i64 %address.local451, !dbg !15
  %address.local485 = getelementptr i8, ptr addrspace(3) %133, i64 512, !dbg !15
  %.idx15001503 = add nuw nsw i64 %address.local481, %address.local438, !dbg !15
  %134 = add nsw i64 %.idx15001503, %address.local451, !dbg !15
  %135 = shl nsw i64 %134, 1, !dbg !15
  %136 = add nsw i64 %135, 512, !dbg !15
  %137 = lshr i64 %136, 6, !dbg !15
  %138 = and i64 %137, 8, !dbg !15
  %139 = xor i64 %136, %138, !dbg !15
  %140 = sub i64 %139, %136, !dbg !15
  %kdma.swizzled1373 = getelementptr i8, ptr addrspace(3) %address.local485, i64 %140, !dbg !15
  %141 = load <4 x half>, ptr addrspace(3) %kdma.swizzled1373, align 8, !dbg !15, !frisk.k_lds_pair !37
  %kdma.gep1375 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1373, i64 256, !dbg !15
  %142 = load <4 x half>, ptr addrspace(3) %kdma.gep1375, align 8, !dbg !15, !frisk.k_lds_pair !37
  %kdma.gep1374 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1373, i64 4096, !dbg !15
  %143 = load <4 x half>, ptr addrspace(3) %kdma.gep1374, align 8, !dbg !15, !frisk.k_lds_pair !38
  %kdma.gep1376 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1373, i64 4352, !dbg !15
  %144 = load <4 x half>, ptr addrspace(3) %kdma.gep1376, align 8, !dbg !15, !frisk.k_lds_pair !38
  %kdma.gep1377 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1024, !dbg !15
  %145 = load <4 x half>, ptr addrspace(3) %kdma.gep1377, align 8, !dbg !15, !frisk.k_lds_pair !33
  %kdma.gep1379 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1280, !dbg !15
  %146 = load <4 x half>, ptr addrspace(3) %kdma.gep1379, align 8, !dbg !15, !frisk.k_lds_pair !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %147 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %127, <4 x float> zeroinitializer) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %kdma.gep1378 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5120, !dbg !15
  %148 = load <4 x half>, ptr addrspace(3) %kdma.gep1378, align 8, !dbg !15, !frisk.k_lds_pair !39
  %kdma.gep1380 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5376, !dbg !15
  %149 = load <4 x half>, ptr addrspace(3) %kdma.gep1380, align 8, !dbg !15, !frisk.k_lds_pair !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %150 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %129, <4 x float> %147) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %151 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %128, <4 x float> zeroinitializer) #9, !dbg !15
  %152 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %130, <4 x float> %151) #9, !dbg !15
  %kdma.gep1381 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1373, i64 1024, !dbg !15
  %153 = load <4 x half>, ptr addrspace(3) %kdma.gep1381, align 8, !dbg !15, !frisk.k_lds_pair !40
  %kdma.gep1383 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1373, i64 1280, !dbg !15
  %154 = load <4 x half>, ptr addrspace(3) %kdma.gep1383, align 8, !dbg !15, !frisk.k_lds_pair !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %155 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %141, <4 x float> zeroinitializer) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %kdma.gep1382 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1373, i64 5120, !dbg !15
  %156 = load <4 x half>, ptr addrspace(3) %kdma.gep1382, align 8, !dbg !15, !frisk.k_lds_pair !41
  %kdma.gep1384 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1373, i64 5376, !dbg !15
  %157 = load <4 x half>, ptr addrspace(3) %kdma.gep1384, align 8, !dbg !15, !frisk.k_lds_pair !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %158 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %143, <4 x float> %155) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %159 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %142, <4 x float> zeroinitializer) #9, !dbg !15
  %160 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %144, <4 x float> %159) #9, !dbg !15
  %161 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %145, <4 x float> zeroinitializer) #9, !dbg !15
  %162 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %148, <4 x float> %161) #9, !dbg !15
  %163 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %146, <4 x float> zeroinitializer) #9, !dbg !15
  %164 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %149, <4 x float> %163) #9, !dbg !15
  %165 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %153, <4 x float> zeroinitializer) #9, !dbg !15
  %166 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %156, <4 x float> %165) #9, !dbg !15
  %167 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %154, <4 x float> zeroinitializer) #9, !dbg !15
  %168 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %157, <4 x float> %167) #9, !dbg !15
  %169 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %127, <4 x float> zeroinitializer) #9, !dbg !15
  %170 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %129, <4 x float> %169) #9, !dbg !15
  %171 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %128, <4 x float> zeroinitializer) #9, !dbg !15
  %172 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %130, <4 x float> %171) #9, !dbg !15
  %173 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %141, <4 x float> zeroinitializer) #9, !dbg !15
  %174 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %143, <4 x float> %173) #9, !dbg !15
  %175 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %142, <4 x float> zeroinitializer) #9, !dbg !15
  %176 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %144, <4 x float> %175) #9, !dbg !15
  %177 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %145, <4 x float> zeroinitializer) #9, !dbg !15
  %178 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %148, <4 x float> %177) #9, !dbg !15
  %179 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %146, <4 x float> zeroinitializer) #9, !dbg !15
  %180 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %149, <4 x float> %179) #9, !dbg !15
  %181 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %153, <4 x float> zeroinitializer) #9, !dbg !15
  %182 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %156, <4 x float> %181) #9, !dbg !15
  %183 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %154, <4 x float> zeroinitializer) #9, !dbg !15
  %184 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %157, <4 x float> %183) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %185 = or disjoint i64 %110, 64, !dbg !13
  %kdma.source1250 = or disjoint i64 %185, %kdma.source1119, !dbg !13
  %kdma.source1251 = trunc nuw i64 %kdma.source1250 to i32, !dbg !13
  %kdma.source1252 = shl nuw nsw i32 %kdma.source1251, 1, !dbg !13
  %kdma.offset1253 = or disjoint i32 %kdma.source1252, %kdma.intra, !dbg !13
  %186 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %36, i32 %kdma.offset1253, <4 x i32> %dma.buffer.rsrc989) #10, !dbg !13
  %kdma.source1266 = or disjoint i64 %185, %kdma.source1133, !dbg !13
  %kdma.source1267 = trunc nuw i64 %kdma.source1266 to i32, !dbg !13
  %kdma.source1268 = shl nuw nsw i32 %kdma.source1267, 1, !dbg !13
  %kdma.offset1269 = or disjoint i32 %kdma.source1268, %kdma.intra.hi, !dbg !13
  %187 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1138, i32 %kdma.offset1269, <4 x i32> %dma.buffer.rsrc989) #10, !dbg !13
  %kdma.source1283 = or disjoint i64 %185, %kdma.source1150, !dbg !13
  %kdma.source1284 = trunc nuw i64 %kdma.source1283 to i32, !dbg !13
  %kdma.source1285 = shl nuw nsw i32 %kdma.source1284, 1, !dbg !13
  %kdma.offset1286 = or disjoint i32 %kdma.source1285, %kdma.intra, !dbg !13
  %188 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %43, i32 %kdma.offset1286, <4 x i32> %dma.buffer.rsrc989) #10, !dbg !13
  %kdma.source1300 = or disjoint i64 %185, %kdma.source1167, !dbg !13
  %kdma.source1301 = trunc nuw i64 %kdma.source1300 to i32, !dbg !13
  %kdma.source1302 = shl nuw nsw i32 %kdma.source1301, 1, !dbg !13
  %kdma.offset1303 = or disjoint i32 %kdma.source1302, %kdma.intra.hi, !dbg !13
  %189 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1172, i32 %kdma.offset1303, <4 x i32> %dma.buffer.rsrc989) #10, !dbg !13
  %kdma.gep1401 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8192, !dbg !15
  %190 = load <4 x half>, ptr addrspace(3) %kdma.gep1401, align 8, !dbg !15, !frisk.k_lds_pair !42
  %kdma.gep1403 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8448, !dbg !15
  %191 = load <4 x half>, ptr addrspace(3) %kdma.gep1403, align 8, !dbg !15, !frisk.k_lds_pair !42
  %kdma.gep1402 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12288, !dbg !15
  %192 = load <4 x half>, ptr addrspace(3) %kdma.gep1402, align 8, !dbg !15, !frisk.k_lds_pair !43
  %kdma.gep1404 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12544, !dbg !15
  %193 = load <4 x half>, ptr addrspace(3) %kdma.gep1404, align 8, !dbg !15, !frisk.k_lds_pair !43
  %kdma.gep1405 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1373, i64 8192, !dbg !15
  %194 = load <4 x half>, ptr addrspace(3) %kdma.gep1405, align 8, !dbg !15, !frisk.k_lds_pair !44
  %kdma.gep1407 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1373, i64 8448, !dbg !15
  %195 = load <4 x half>, ptr addrspace(3) %kdma.gep1407, align 8, !dbg !15, !frisk.k_lds_pair !44
  %kdma.gep1406 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1373, i64 12288, !dbg !15
  %196 = load <4 x half>, ptr addrspace(3) %kdma.gep1406, align 8, !dbg !15, !frisk.k_lds_pair !45
  %kdma.gep1408 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1373, i64 12544, !dbg !15
  %197 = load <4 x half>, ptr addrspace(3) %kdma.gep1408, align 8, !dbg !15, !frisk.k_lds_pair !45
  %kdma.gep1409 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9216, !dbg !15
  %198 = load <4 x half>, ptr addrspace(3) %kdma.gep1409, align 8, !dbg !15, !frisk.k_lds_pair !46
  %kdma.gep1411 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9472, !dbg !15
  %199 = load <4 x half>, ptr addrspace(3) %kdma.gep1411, align 8, !dbg !15, !frisk.k_lds_pair !46
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %200 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %190, <4 x float> %150) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %kdma.gep1410 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13312, !dbg !15
  %201 = load <4 x half>, ptr addrspace(3) %kdma.gep1410, align 8, !dbg !15, !frisk.k_lds_pair !47
  %kdma.gep1412 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13568, !dbg !15
  %202 = load <4 x half>, ptr addrspace(3) %kdma.gep1412, align 8, !dbg !15, !frisk.k_lds_pair !47
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %203 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %192, <4 x float> %200) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %204 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %191, <4 x float> %152) #9, !dbg !15
  %205 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %193, <4 x float> %204) #9, !dbg !15
  %kdma.gep1413 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1373, i64 9216, !dbg !15
  %206 = load <4 x half>, ptr addrspace(3) %kdma.gep1413, align 8, !dbg !15, !frisk.k_lds_pair !48
  %kdma.gep1415 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1373, i64 9472, !dbg !15
  %207 = load <4 x half>, ptr addrspace(3) %kdma.gep1415, align 8, !dbg !15, !frisk.k_lds_pair !48
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %208 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %194, <4 x float> %158) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %kdma.gep1414 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1373, i64 13312, !dbg !15
  %209 = load <4 x half>, ptr addrspace(3) %kdma.gep1414, align 8, !dbg !15, !frisk.k_lds_pair !49
  %kdma.gep1416 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1373, i64 13568, !dbg !15
  %210 = load <4 x half>, ptr addrspace(3) %kdma.gep1416, align 8, !dbg !15, !frisk.k_lds_pair !49
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %211 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %196, <4 x float> %208) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %212 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %195, <4 x float> %160) #9, !dbg !15
  %213 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %197, <4 x float> %212) #9, !dbg !15
  %214 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %198, <4 x float> %162) #9, !dbg !15
  %215 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %201, <4 x float> %214) #9, !dbg !15
  %216 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %199, <4 x float> %164) #9, !dbg !15
  %217 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %202, <4 x float> %216) #9, !dbg !15
  %218 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %206, <4 x float> %166) #9, !dbg !15
  %219 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %209, <4 x float> %218) #9, !dbg !15
  %220 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %207, <4 x float> %168) #9, !dbg !15
  %221 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %210, <4 x float> %220) #9, !dbg !15
  %222 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %190, <4 x float> %170) #9, !dbg !15
  %223 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %192, <4 x float> %222) #9, !dbg !15
  %224 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %191, <4 x float> %172) #9, !dbg !15
  %225 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %193, <4 x float> %224) #9, !dbg !15
  %226 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %194, <4 x float> %174) #9, !dbg !15
  %227 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %196, <4 x float> %226) #9, !dbg !15
  %228 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %195, <4 x float> %176) #9, !dbg !15
  %229 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %197, <4 x float> %228) #9, !dbg !15
  %230 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %198, <4 x float> %178) #9, !dbg !15
  %231 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %201, <4 x float> %230) #9, !dbg !15
  %232 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %199, <4 x float> %180) #9, !dbg !15
  %233 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %202, <4 x float> %232) #9, !dbg !15
  %234 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %206, <4 x float> %182) #9, !dbg !15
  %235 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %209, <4 x float> %234) #9, !dbg !15
  %236 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %207, <4 x float> %184) #9, !dbg !15
  %237 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %210, <4 x float> %236) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %238 = or disjoint i64 %110, 96, !dbg !13
  %kdma.source1316 = or disjoint i64 %238, %kdma.source1119, !dbg !13
  %kdma.source1317 = trunc nuw i64 %kdma.source1316 to i32, !dbg !13
  %kdma.source1318 = shl nuw nsw i32 %kdma.source1317, 1, !dbg !13
  %kdma.offset1319 = or disjoint i32 %kdma.source1318, %kdma.intra, !dbg !13
  %239 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %48, i32 %kdma.offset1319, <4 x i32> %dma.buffer.rsrc989) #10, !dbg !13
  %kdma.source1332 = or disjoint i64 %238, %kdma.source1133, !dbg !13
  %kdma.source1333 = trunc nuw i64 %kdma.source1332 to i32, !dbg !13
  %kdma.source1334 = shl nuw nsw i32 %kdma.source1333, 1, !dbg !13
  %kdma.offset1335 = or disjoint i32 %kdma.source1334, %kdma.intra.hi, !dbg !13
  %240 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1204, i32 %kdma.offset1335, <4 x i32> %dma.buffer.rsrc989) #10, !dbg !13
  %kdma.source1349 = or disjoint i64 %238, %kdma.source1150, !dbg !13
  %kdma.source1350 = trunc nuw i64 %kdma.source1349 to i32, !dbg !13
  %kdma.source1351 = shl nuw nsw i32 %kdma.source1350, 1, !dbg !13
  %kdma.offset1352 = or disjoint i32 %kdma.source1351, %kdma.intra, !dbg !13
  %241 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %52, i32 %kdma.offset1352, <4 x i32> %dma.buffer.rsrc989) #10, !dbg !13
  %kdma.source1366 = or disjoint i64 %238, %kdma.source1167, !dbg !13
  %kdma.source1367 = trunc nuw i64 %kdma.source1366 to i32, !dbg !13
  %kdma.source1368 = shl nuw nsw i32 %kdma.source1367, 1, !dbg !13
  %kdma.offset1369 = or disjoint i32 %kdma.source1368, %kdma.intra.hi, !dbg !13
  %242 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1238, i32 %kdma.offset1369, <4 x i32> %dma.buffer.rsrc989) #10, !dbg !13
  %243 = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8, !dbg !15, !frisk.k_lds_pair !50
  %244 = load <4 x half>, ptr addrspace(3) %kdma.gep1371, align 8, !dbg !15, !frisk.k_lds_pair !50
  %245 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8, !dbg !15, !frisk.k_lds_pair !51
  %246 = load <4 x half>, ptr addrspace(3) %kdma.gep1372, align 8, !dbg !15, !frisk.k_lds_pair !51
  %247 = load <4 x half>, ptr addrspace(3) %kdma.swizzled1373, align 8, !dbg !15, !frisk.k_lds_pair !52
  %248 = load <4 x half>, ptr addrspace(3) %kdma.gep1375, align 8, !dbg !15, !frisk.k_lds_pair !52
  %249 = load <4 x half>, ptr addrspace(3) %kdma.gep1374, align 8, !dbg !15, !frisk.k_lds_pair !53
  %250 = load <4 x half>, ptr addrspace(3) %kdma.gep1376, align 8, !dbg !15, !frisk.k_lds_pair !53
  %251 = load <4 x half>, ptr addrspace(3) %kdma.gep1377, align 8, !dbg !15, !frisk.k_lds_pair !54
  %252 = load <4 x half>, ptr addrspace(3) %kdma.gep1379, align 8, !dbg !15, !frisk.k_lds_pair !54
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %253 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %243, <4 x float> %203) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %254 = load <4 x half>, ptr addrspace(3) %kdma.gep1378, align 8, !dbg !15, !frisk.k_lds_pair !55
  %255 = load <4 x half>, ptr addrspace(3) %kdma.gep1380, align 8, !dbg !15, !frisk.k_lds_pair !55
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %256 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %245, <4 x float> %253) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %257 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %244, <4 x float> %205) #9, !dbg !15
  %258 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %246, <4 x float> %257) #9, !dbg !15
  %259 = load <4 x half>, ptr addrspace(3) %kdma.gep1381, align 8, !dbg !15, !frisk.k_lds_pair !56
  %260 = load <4 x half>, ptr addrspace(3) %kdma.gep1383, align 8, !dbg !15, !frisk.k_lds_pair !56
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %261 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %247, <4 x float> %211) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %262 = load <4 x half>, ptr addrspace(3) %kdma.gep1382, align 8, !dbg !15, !frisk.k_lds_pair !57
  %263 = load <4 x half>, ptr addrspace(3) %kdma.gep1384, align 8, !dbg !15, !frisk.k_lds_pair !57
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %264 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %249, <4 x float> %261) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %265 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %248, <4 x float> %213) #9, !dbg !15
  %266 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %250, <4 x float> %265) #9, !dbg !15
  %267 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %251, <4 x float> %215) #9, !dbg !15
  %268 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %254, <4 x float> %267) #9, !dbg !15
  %269 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %252, <4 x float> %217) #9, !dbg !15
  %270 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %255, <4 x float> %269) #9, !dbg !15
  %271 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %259, <4 x float> %219) #9, !dbg !15
  %272 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %262, <4 x float> %271) #9, !dbg !15
  %273 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %260, <4 x float> %221) #9, !dbg !15
  %274 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %263, <4 x float> %273) #9, !dbg !15
  %275 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %243, <4 x float> %223) #9, !dbg !15
  %276 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %245, <4 x float> %275) #9, !dbg !15
  %277 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %244, <4 x float> %225) #9, !dbg !15
  %278 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %246, <4 x float> %277) #9, !dbg !15
  %279 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %247, <4 x float> %227) #9, !dbg !15
  %280 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %249, <4 x float> %279) #9, !dbg !15
  %281 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %248, <4 x float> %229) #9, !dbg !15
  %282 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %250, <4 x float> %281) #9, !dbg !15
  %283 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %251, <4 x float> %231) #9, !dbg !15
  %284 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %254, <4 x float> %283) #9, !dbg !15
  %285 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %252, <4 x float> %233) #9, !dbg !15
  %286 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %255, <4 x float> %285) #9, !dbg !15
  %287 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %259, <4 x float> %235) #9, !dbg !15
  %288 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %262, <4 x float> %287) #9, !dbg !15
  %289 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %260, <4 x float> %237) #9, !dbg !15
  %290 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %263, <4 x float> %289) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %address.local568 = shl nuw nsw i64 %address.local411, 5, !dbg !16
  %address.local569 = and i64 %address.local568, 8064, !dbg !16
  %address.local570 = shl nuw nsw i64 %address.local411, 3, !dbg !16
  %address.local571 = and i64 %address.local570, 24, !dbg !16
  %address.local572 = or disjoint i64 %address.local569, %address.local571, !dbg !16
  %291 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !16
  %.idx10351036 = or disjoint i64 %110, %address.local572, !dbg !16
  %.idx10351036.tr = trunc nuw i64 %.idx10351036 to i32, !dbg !16
  %dma.buffer.offset972 = shl nuw nsw i32 %.idx10351036.tr, 1, !dbg !16
  %292 = insertelement <2 x i64> poison, i64 %291, i64 0, !dbg !16
  %293 = bitcast <2 x i64> %292 to <4 x i32>, !dbg !16
  %dma.buffer.rsrc973 = shufflevector <4 x i32> %293, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !16
  %294 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %36, i32 %dma.buffer.offset972, <4 x i32> %dma.buffer.rsrc973) #10, !dbg !16
  %address.local578 = or disjoint i64 %address.local572, 8192, !dbg !16
  %.idx10371038 = or disjoint i64 %110, %address.local578, !dbg !16
  %.idx10371038.tr = trunc nuw i64 %.idx10371038 to i32, !dbg !16
  %dma.buffer.offset970 = shl nuw nsw i32 %.idx10371038.tr, 1, !dbg !16
  %295 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %43, i32 %dma.buffer.offset970, <4 x i32> %dma.buffer.rsrc973) #10, !dbg !16
  %296 = load <4 x half>, ptr addrspace(3) %kdma.gep1401, align 8, !dbg !15, !frisk.k_tail_pair !35
  %297 = load <4 x half>, ptr addrspace(3) %kdma.gep1403, align 8, !dbg !15, !frisk.k_tail_pair !35
  %298 = load <4 x half>, ptr addrspace(3) %kdma.gep1402, align 8, !dbg !15, !frisk.k_tail_pair !36
  %299 = load <4 x half>, ptr addrspace(3) %kdma.gep1404, align 8, !dbg !15, !frisk.k_tail_pair !36
  %300 = load <4 x half>, ptr addrspace(3) %kdma.gep1405, align 8, !dbg !15, !frisk.k_tail_pair !37
  %301 = load <4 x half>, ptr addrspace(3) %kdma.gep1407, align 8, !dbg !15, !frisk.k_tail_pair !37
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_tail_prefetch !1
  %302 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %296, <4 x float> %256) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_tail_prefetch !1
  %303 = load <4 x half>, ptr addrspace(3) %kdma.gep1406, align 8, !dbg !15, !frisk.k_tail_pair !38
  %304 = load <4 x half>, ptr addrspace(3) %kdma.gep1408, align 8, !dbg !15, !frisk.k_tail_pair !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_tail_prefetch !1
  %305 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %298, <4 x float> %302) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_tail_prefetch !1
  %306 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %297, <4 x float> %258) #9, !dbg !15
  %307 = extractelement <4 x float> %305, i64 0, !dbg !20
  %308 = call float @llvm.maxnum.f32(float %307, float 0xFFF0000000000000), !dbg !6, !frisk.local_reduction !1
  %309 = extractelement <4 x float> %305, i64 1, !dbg !20
  %310 = extractelement <4 x float> %305, i64 2, !dbg !20
  %311 = extractelement <4 x float> %305, i64 3, !dbg !20
  %312 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %299, <4 x float> %306) #9, !dbg !15
  %313 = load <4 x half>, ptr addrspace(3) %kdma.gep1409, align 8, !dbg !15, !frisk.k_tail_pair !33
  %314 = load <4 x half>, ptr addrspace(3) %kdma.gep1411, align 8, !dbg !15, !frisk.k_tail_pair !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_tail_prefetch !1
  %315 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %300, <4 x float> %264) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !20, !frisk.k_tail_prefetch !1
  %316 = extractelement <4 x float> %312, i64 0, !dbg !20
  %local.reduce143 = call float @llvm.maxnum.f32(float %308, float %316), !dbg !6, !frisk.local_reduction !1
  %317 = extractelement <4 x float> %312, i64 1, !dbg !20
  %local.reduce144 = call float @llvm.maxnum.f32(float %309, float %317), !dbg !6, !frisk.local_reduction !1
  %318 = extractelement <4 x float> %312, i64 2, !dbg !20
  %local.reduce145 = call float @llvm.maxnum.f32(float %310, float %318), !dbg !6, !frisk.local_reduction !1
  %319 = extractelement <4 x float> %312, i64 3, !dbg !20
  %local.reduce146 = call float @llvm.maxnum.f32(float %311, float %319), !dbg !6, !frisk.local_reduction !1
  %320 = load <4 x half>, ptr addrspace(3) %kdma.gep1410, align 8, !dbg !15, !frisk.k_tail_pair !39
  %321 = load <4 x half>, ptr addrspace(3) %kdma.gep1412, align 8, !dbg !15, !frisk.k_tail_pair !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_tail_prefetch !1
  %322 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %303, <4 x float> %315) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_tail_prefetch !1
  %323 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %301, <4 x float> %266) #9, !dbg !15
  %324 = extractelement <4 x float> %322, i64 0, !dbg !20
  %local.reduce147 = call float @llvm.maxnum.f32(float %local.reduce143, float %324), !dbg !6, !frisk.local_reduction !1
  %325 = extractelement <4 x float> %322, i64 1, !dbg !20
  %local.reduce148 = call float @llvm.maxnum.f32(float %local.reduce144, float %325), !dbg !6, !frisk.local_reduction !1
  %326 = extractelement <4 x float> %322, i64 2, !dbg !20
  %local.reduce149 = call float @llvm.maxnum.f32(float %local.reduce145, float %326), !dbg !6, !frisk.local_reduction !1
  %327 = extractelement <4 x float> %322, i64 3, !dbg !20
  %local.reduce150 = call float @llvm.maxnum.f32(float %local.reduce146, float %327), !dbg !6, !frisk.local_reduction !1
  %328 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %304, <4 x float> %323) #9, !dbg !15
  %329 = load <4 x half>, ptr addrspace(3) %kdma.gep1413, align 8, !dbg !15, !frisk.k_tail_pair !40
  %330 = load <4 x half>, ptr addrspace(3) %kdma.gep1415, align 8, !dbg !15, !frisk.k_tail_pair !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_tail_prefetch !1
  %331 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %313, <4 x float> %268) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !20, !frisk.k_tail_prefetch !1
  %332 = extractelement <4 x float> %328, i64 0, !dbg !20
  %local.reduce151 = call float @llvm.maxnum.f32(float %local.reduce147, float %332), !dbg !6, !frisk.local_reduction !1
  %333 = extractelement <4 x float> %328, i64 1, !dbg !20
  %local.reduce152 = call float @llvm.maxnum.f32(float %local.reduce148, float %333), !dbg !6, !frisk.local_reduction !1
  %334 = extractelement <4 x float> %328, i64 2, !dbg !20
  %local.reduce153 = call float @llvm.maxnum.f32(float %local.reduce149, float %334), !dbg !6, !frisk.local_reduction !1
  %335 = extractelement <4 x float> %328, i64 3, !dbg !20
  %local.reduce154 = call float @llvm.maxnum.f32(float %local.reduce150, float %335), !dbg !6, !frisk.local_reduction !1
  %336 = load <4 x half>, ptr addrspace(3) %kdma.gep1414, align 8, !dbg !15, !frisk.k_tail_pair !41
  %337 = load <4 x half>, ptr addrspace(3) %kdma.gep1416, align 8, !dbg !15, !frisk.k_tail_pair !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_tail_prefetch !1
  %338 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %320, <4 x float> %331) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_tail_prefetch !1
  %339 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %314, <4 x float> %270) #9, !dbg !15
  %340 = extractelement <4 x float> %338, i64 0, !dbg !20
  %local.reduce155 = call float @llvm.maxnum.f32(float %local.reduce151, float %340), !dbg !6, !frisk.local_reduction !1
  %341 = extractelement <4 x float> %338, i64 1, !dbg !20
  %local.reduce156 = call float @llvm.maxnum.f32(float %local.reduce152, float %341), !dbg !6, !frisk.local_reduction !1
  %342 = extractelement <4 x float> %338, i64 2, !dbg !20
  %local.reduce157 = call float @llvm.maxnum.f32(float %local.reduce153, float %342), !dbg !6, !frisk.local_reduction !1
  %343 = extractelement <4 x float> %338, i64 3, !dbg !20
  %local.reduce158 = call float @llvm.maxnum.f32(float %local.reduce154, float %343), !dbg !6, !frisk.local_reduction !1
  %344 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %321, <4 x float> %339) #9, !dbg !15
  %345 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %329, <4 x float> %272) #9, !dbg !15
  %346 = extractelement <4 x float> %344, i64 0, !dbg !20
  %local.reduce159 = call float @llvm.maxnum.f32(float %local.reduce155, float %346), !dbg !6, !frisk.local_reduction !1
  %347 = extractelement <4 x float> %344, i64 1, !dbg !20
  %local.reduce160 = call float @llvm.maxnum.f32(float %local.reduce156, float %347), !dbg !6, !frisk.local_reduction !1
  %348 = extractelement <4 x float> %344, i64 2, !dbg !20
  %local.reduce161 = call float @llvm.maxnum.f32(float %local.reduce157, float %348), !dbg !6, !frisk.local_reduction !1
  %349 = extractelement <4 x float> %344, i64 3, !dbg !20
  %local.reduce162 = call float @llvm.maxnum.f32(float %local.reduce158, float %349), !dbg !6, !frisk.local_reduction !1
  %350 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %336, <4 x float> %345) #9, !dbg !15
  %351 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %330, <4 x float> %274) #9, !dbg !15
  %352 = extractelement <4 x float> %350, i64 0, !dbg !20
  %local.reduce163 = call float @llvm.maxnum.f32(float %local.reduce159, float %352), !dbg !6, !frisk.local_reduction !1
  %353 = extractelement <4 x float> %350, i64 1, !dbg !20
  %local.reduce164 = call float @llvm.maxnum.f32(float %local.reduce160, float %353), !dbg !6, !frisk.local_reduction !1
  %354 = extractelement <4 x float> %350, i64 2, !dbg !20
  %local.reduce165 = call float @llvm.maxnum.f32(float %local.reduce161, float %354), !dbg !6, !frisk.local_reduction !1
  %355 = extractelement <4 x float> %350, i64 3, !dbg !20
  %local.reduce166 = call float @llvm.maxnum.f32(float %local.reduce162, float %355), !dbg !6, !frisk.local_reduction !1
  %356 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %337, <4 x float> %351) #9, !dbg !15
  %357 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %296, <4 x float> %276) #9, !dbg !15
  %358 = extractelement <4 x float> %356, i64 0, !dbg !20
  %local.reduce167 = call float @llvm.maxnum.f32(float %local.reduce163, float %358), !dbg !6, !frisk.local_reduction !1
  %359 = extractelement <4 x float> %356, i64 1, !dbg !20
  %local.reduce168 = call float @llvm.maxnum.f32(float %local.reduce164, float %359), !dbg !6, !frisk.local_reduction !1
  %360 = extractelement <4 x float> %356, i64 2, !dbg !20
  %local.reduce169 = call float @llvm.maxnum.f32(float %local.reduce165, float %360), !dbg !6, !frisk.local_reduction !1
  %361 = extractelement <4 x float> %356, i64 3, !dbg !20
  %local.reduce170 = call float @llvm.maxnum.f32(float %local.reduce166, float %361), !dbg !6, !frisk.local_reduction !1
  %local.reduce171 = call float @llvm.maxnum.f32(float %local.reduce167, float %local.reduce168), !dbg !6, !frisk.local_reduction !1
  %local.reduce172 = call float @llvm.maxnum.f32(float %local.reduce169, float %local.reduce170), !dbg !6, !frisk.local_reduction !1
  %362 = call float @llvm.maxnum.f32(float %local.reduce171, float %local.reduce172), !dbg !6, !frisk.local_reduction !1
  %363 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %298, <4 x float> %357) #9, !dbg !15
  %364 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %297, <4 x float> %278) #9, !dbg !15
  %365 = extractelement <4 x float> %363, i64 0, !dbg !20
  %366 = extractelement <4 x float> %363, i64 1, !dbg !20
  %367 = extractelement <4 x float> %363, i64 2, !dbg !20
  %368 = extractelement <4 x float> %363, i64 3, !dbg !20
  %369 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %299, <4 x float> %364) #9, !dbg !15
  %370 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %300, <4 x float> %280) #9, !dbg !15
  %371 = extractelement <4 x float> %369, i64 0, !dbg !20
  %372 = extractelement <4 x float> %369, i64 1, !dbg !20
  %373 = extractelement <4 x float> %369, i64 2, !dbg !20
  %374 = extractelement <4 x float> %369, i64 3, !dbg !20
  %375 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %303, <4 x float> %370) #9, !dbg !15
  %376 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %301, <4 x float> %282) #9, !dbg !15
  %377 = extractelement <4 x float> %375, i64 0, !dbg !20
  %378 = extractelement <4 x float> %375, i64 1, !dbg !20
  %379 = extractelement <4 x float> %375, i64 2, !dbg !20
  %380 = extractelement <4 x float> %375, i64 3, !dbg !20
  %381 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %304, <4 x float> %376) #9, !dbg !15
  %382 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %313, <4 x float> %284) #9, !dbg !15
  %383 = extractelement <4 x float> %381, i64 0, !dbg !20
  %384 = extractelement <4 x float> %381, i64 1, !dbg !20
  %385 = extractelement <4 x float> %381, i64 2, !dbg !20
  %386 = extractelement <4 x float> %381, i64 3, !dbg !20
  %387 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %320, <4 x float> %382) #9, !dbg !15
  %388 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %314, <4 x float> %286) #9, !dbg !15
  %389 = extractelement <4 x float> %387, i64 0, !dbg !20
  %390 = extractelement <4 x float> %387, i64 1, !dbg !20
  %391 = extractelement <4 x float> %387, i64 2, !dbg !20
  %392 = extractelement <4 x float> %387, i64 3, !dbg !20
  %393 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %321, <4 x float> %388) #9, !dbg !15
  %394 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %329, <4 x float> %288) #9, !dbg !15
  %395 = extractelement <4 x float> %393, i64 0, !dbg !20
  %396 = extractelement <4 x float> %393, i64 1, !dbg !20
  %397 = extractelement <4 x float> %393, i64 2, !dbg !20
  %398 = extractelement <4 x float> %393, i64 3, !dbg !20
  %399 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %336, <4 x float> %394) #9, !dbg !15
  %400 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %330, <4 x float> %290) #9, !dbg !15
  %401 = extractelement <4 x float> %399, i64 0, !dbg !20
  %402 = extractelement <4 x float> %399, i64 1, !dbg !20
  %403 = extractelement <4 x float> %399, i64 2, !dbg !20
  %404 = extractelement <4 x float> %399, i64 3, !dbg !20
  %405 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %337, <4 x float> %400) #9, !dbg !15
  %406 = extractelement <4 x float> %405, i64 0, !dbg !20
  %407 = extractelement <4 x float> %405, i64 1, !dbg !20
  %408 = extractelement <4 x float> %405, i64 2, !dbg !20
  %409 = extractelement <4 x float> %405, i64 3, !dbg !20
  %410 = bitcast float %362 to i32, !dbg !6
  %411 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %410), !dbg !6
  %412 = bitcast i32 %411 to float, !dbg !6
  %413 = call float @llvm.maxnum.f32(float %362, float %412), !dbg !6
  %414 = bitcast float %413 to i32, !dbg !6
  %415 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %414), !dbg !6
  %416 = bitcast i32 %415 to float, !dbg !6
  %417 = call float @llvm.maxnum.f32(float %413, float %416), !dbg !6
  %418 = call float @llvm.maxnum.f32(float %365, float 0xFFF0000000000000), !dbg !6, !frisk.local_reduction !1
  %local.reduce112 = call float @llvm.maxnum.f32(float %418, float %371), !dbg !6, !frisk.local_reduction !1
  %local.reduce113 = call float @llvm.maxnum.f32(float %366, float %372), !dbg !6, !frisk.local_reduction !1
  %local.reduce114 = call float @llvm.maxnum.f32(float %367, float %373), !dbg !6, !frisk.local_reduction !1
  %local.reduce115 = call float @llvm.maxnum.f32(float %368, float %374), !dbg !6, !frisk.local_reduction !1
  %local.reduce116 = call float @llvm.maxnum.f32(float %local.reduce112, float %377), !dbg !6, !frisk.local_reduction !1
  %local.reduce117 = call float @llvm.maxnum.f32(float %local.reduce113, float %378), !dbg !6, !frisk.local_reduction !1
  %local.reduce118 = call float @llvm.maxnum.f32(float %local.reduce114, float %379), !dbg !6, !frisk.local_reduction !1
  %local.reduce119 = call float @llvm.maxnum.f32(float %local.reduce115, float %380), !dbg !6, !frisk.local_reduction !1
  %local.reduce120 = call float @llvm.maxnum.f32(float %local.reduce116, float %383), !dbg !6, !frisk.local_reduction !1
  %local.reduce121 = call float @llvm.maxnum.f32(float %local.reduce117, float %384), !dbg !6, !frisk.local_reduction !1
  %local.reduce122 = call float @llvm.maxnum.f32(float %local.reduce118, float %385), !dbg !6, !frisk.local_reduction !1
  %local.reduce123 = call float @llvm.maxnum.f32(float %local.reduce119, float %386), !dbg !6, !frisk.local_reduction !1
  %local.reduce124 = call float @llvm.maxnum.f32(float %local.reduce120, float %389), !dbg !6, !frisk.local_reduction !1
  %local.reduce125 = call float @llvm.maxnum.f32(float %local.reduce121, float %390), !dbg !6, !frisk.local_reduction !1
  %local.reduce126 = call float @llvm.maxnum.f32(float %local.reduce122, float %391), !dbg !6, !frisk.local_reduction !1
  %local.reduce127 = call float @llvm.maxnum.f32(float %local.reduce123, float %392), !dbg !6, !frisk.local_reduction !1
  %local.reduce128 = call float @llvm.maxnum.f32(float %local.reduce124, float %395), !dbg !6, !frisk.local_reduction !1
  %local.reduce129 = call float @llvm.maxnum.f32(float %local.reduce125, float %396), !dbg !6, !frisk.local_reduction !1
  %local.reduce130 = call float @llvm.maxnum.f32(float %local.reduce126, float %397), !dbg !6, !frisk.local_reduction !1
  %local.reduce131 = call float @llvm.maxnum.f32(float %local.reduce127, float %398), !dbg !6, !frisk.local_reduction !1
  %local.reduce132 = call float @llvm.maxnum.f32(float %local.reduce128, float %401), !dbg !6, !frisk.local_reduction !1
  %local.reduce133 = call float @llvm.maxnum.f32(float %local.reduce129, float %402), !dbg !6, !frisk.local_reduction !1
  %local.reduce134 = call float @llvm.maxnum.f32(float %local.reduce130, float %403), !dbg !6, !frisk.local_reduction !1
  %local.reduce135 = call float @llvm.maxnum.f32(float %local.reduce131, float %404), !dbg !6, !frisk.local_reduction !1
  %local.reduce136 = call float @llvm.maxnum.f32(float %local.reduce132, float %406), !dbg !6, !frisk.local_reduction !1
  %local.reduce137 = call float @llvm.maxnum.f32(float %local.reduce133, float %407), !dbg !6, !frisk.local_reduction !1
  %local.reduce138 = call float @llvm.maxnum.f32(float %local.reduce134, float %408), !dbg !6, !frisk.local_reduction !1
  %local.reduce139 = call float @llvm.maxnum.f32(float %local.reduce135, float %409), !dbg !6, !frisk.local_reduction !1
  %local.reduce140 = call float @llvm.maxnum.f32(float %local.reduce136, float %local.reduce137), !dbg !6, !frisk.local_reduction !1
  %local.reduce141 = call float @llvm.maxnum.f32(float %local.reduce138, float %local.reduce139), !dbg !6, !frisk.local_reduction !1
  %419 = call float @llvm.maxnum.f32(float %local.reduce140, float %local.reduce141), !dbg !6, !frisk.local_reduction !1
  %420 = bitcast float %419 to i32, !dbg !6
  %421 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %420), !dbg !6
  %422 = bitcast i32 %421 to float, !dbg !6
  %423 = call float @llvm.maxnum.f32(float %419, float %422), !dbg !6
  %424 = bitcast float %423 to i32, !dbg !6
  %425 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %424), !dbg !6
  %426 = bitcast i32 %425 to float, !dbg !6
  %427 = call float @llvm.maxnum.f32(float %423, float %426), !dbg !6
  %428 = extractvalue [2 x <1 x float>] %101, 0, !dbg !6
  %429 = insertelement <1 x float> poison, float %417, i64 0, !dbg !6
  %430 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %429, <1 x float> %428), !dbg !6
  %431 = extractvalue [2 x <1 x float>] %101, 1, !dbg !6
  %432 = insertelement <1 x float> poison, float %427, i64 0, !dbg !6
  %433 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %432, <1 x float> %431), !dbg !6
  %434 = shufflevector <1 x float> %430, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !21
  %435 = shufflevector <1 x float> %433, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !21
  %436 = fsub <1 x float> %428, %430, !dbg !22
  %437 = extractelement <1 x float> %436, i64 0, !dbg !23
  %438 = call float @llvm.amdgcn.exp2.f32(float %437), !dbg !23
  %439 = insertelement <4 x float> poison, float %438, i64 0, !dbg !28
  %440 = shufflevector <4 x float> %439, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !28
  %441 = extractvalue [2 x <1 x float>] %100, 0, !dbg !24
  %442 = extractelement <1 x float> %441, i64 0, !dbg !24
  %443 = fmul float %442, %438, !dbg !24
  %444 = fsub <1 x float> %431, %433, !dbg !22
  %445 = extractelement <1 x float> %444, i64 0, !dbg !23
  %446 = call float @llvm.amdgcn.exp2.f32(float %445), !dbg !23
  %447 = insertelement <4 x float> poison, float %446, i64 0, !dbg !28
  %448 = shufflevector <4 x float> %447, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !28
  %449 = extractvalue [2 x <1 x float>] %100, 1, !dbg !24
  %450 = extractelement <1 x float> %449, i64 0, !dbg !24
  %451 = fmul float %450, %446, !dbg !24
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %vdb.address = or disjoint i64 %114, %address.local572, !dbg !16
  %vdb.address1051 = trunc nuw i64 %vdb.address to i32, !dbg !16
  %vdb.address1052 = shl nuw nsw i32 %vdb.address1051, 1, !dbg !16
  %vdb.address1057 = or disjoint i64 %114, %address.local578, !dbg !16
  %vdb.address1058 = trunc nuw i64 %vdb.address1057 to i32, !dbg !16
  %vdb.address1059 = shl nuw nsw i32 %vdb.address1058, 1, !dbg !16
  %vdb.dma.base = add i32 %36, 8192, !dbg !14
  %452 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1052, <4 x i32> %dma.buffer.rsrc973) #10, !dbg !16
  %vdb.dma.base1064 = add i32 %43, 8192, !dbg !14
  %453 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1064, i32 %vdb.address1059, <4 x i32> %dma.buffer.rsrc973) #10, !dbg !16
  %address.local580 = shl nuw nsw i32 %address.thread410, 3, !dbg !14
  %address.local581 = and i32 %address.local580, 504, !dbg !14
  %address.local582 = zext nneg i32 %address.local581 to i64, !dbg !14
  %address.local583 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local582, !dbg !14
  %454 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local583, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %455 = bitcast <4 x i32> %454 to <8 x half>, !dbg !14
  %address.local585 = getelementptr i8, ptr addrspace(3) %address.local583, i64 1024, !dbg !14
  %456 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local585, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %457 = bitcast <4 x i32> %456 to <8 x half>, !dbg !14
  %address.local587 = getelementptr i8, ptr addrspace(3) %address.local583, i64 2048, !dbg !14
  %458 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local587, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %459 = bitcast <4 x i32> %458 to <8 x half>, !dbg !14
  %address.local589 = getelementptr i8, ptr addrspace(3) %address.local583, i64 3072, !dbg !14
  %460 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local589, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %461 = bitcast <4 x i32> %460 to <8 x half>, !dbg !14
  %address.local591 = getelementptr i8, ptr addrspace(3) %address.local583, i64 4096, !dbg !14
  %462 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local591, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %463 = bitcast <4 x i32> %462 to <8 x half>, !dbg !14
  %address.local593 = getelementptr i8, ptr addrspace(3) %address.local583, i64 5120, !dbg !14
  %464 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local593, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %465 = bitcast <4 x i32> %464 to <8 x half>, !dbg !14
  %address.local595 = getelementptr i8, ptr addrspace(3) %address.local583, i64 6144, !dbg !14
  %466 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local595, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %467 = bitcast <4 x i32> %466 to <8 x half>, !dbg !14
  %address.local597 = getelementptr i8, ptr addrspace(3) %address.local583, i64 7168, !dbg !14
  %468 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local597, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %469 = shufflevector <8 x half> %455, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %470 = extractvalue [2 x <8 x float>] %99, 0, !dbg !28
  %471 = shufflevector <8 x float> %470, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %472 = fmul <4 x float> %471, %440, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %473 = fsub <4 x float> %305, %434, !dbg !21
  %474 = extractelement <4 x float> %473, i64 0, !dbg !25
  %475 = call float @llvm.amdgcn.exp2.f32(float %474), !dbg !25
  %476 = insertelement <4 x float> poison, float %475, i64 0, !dbg !27
  %477 = extractelement <4 x float> %473, i64 1, !dbg !25
  %478 = call float @llvm.amdgcn.exp2.f32(float %477), !dbg !25
  %479 = insertelement <4 x float> %476, float %478, i64 1, !dbg !27
  %480 = extractelement <4 x float> %473, i64 2, !dbg !25
  %481 = call float @llvm.amdgcn.exp2.f32(float %480), !dbg !25
  %482 = insertelement <4 x float> %479, float %481, i64 2, !dbg !27
  %483 = extractelement <4 x float> %473, i64 3, !dbg !25
  %484 = call float @llvm.amdgcn.exp2.f32(float %483), !dbg !25
  %485 = insertelement <4 x float> %482, float %484, i64 3, !dbg !27
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %485) #9, !dbg !27, !frisk.register_lifetimes !1
  %486 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !27
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %486) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %487 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %469, <4 x float> %472) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %488 = shufflevector <8 x half> %455, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %489 = shufflevector <8 x float> %470, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %490 = fmul <4 x float> %489, %440, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %491 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %488, <4 x float> %490) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %492 = extractvalue [2 x <8 x float>] %99, 1, !dbg !28
  %493 = shufflevector <8 x float> %492, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %494 = fmul <4 x float> %493, %448, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %495 = fsub <4 x float> %363, %435, !dbg !21
  %496 = extractelement <4 x float> %495, i64 0, !dbg !25
  %497 = call float @llvm.amdgcn.exp2.f32(float %496), !dbg !25
  %498 = insertelement <4 x float> poison, float %497, i64 0, !dbg !27
  %499 = extractelement <4 x float> %495, i64 1, !dbg !25
  %500 = call float @llvm.amdgcn.exp2.f32(float %499), !dbg !25
  %501 = insertelement <4 x float> %498, float %500, i64 1, !dbg !27
  %502 = extractelement <4 x float> %495, i64 2, !dbg !25
  %503 = call float @llvm.amdgcn.exp2.f32(float %502), !dbg !25
  %504 = insertelement <4 x float> %501, float %503, i64 2, !dbg !27
  %505 = extractelement <4 x float> %495, i64 3, !dbg !25
  %506 = call float @llvm.amdgcn.exp2.f32(float %505), !dbg !25
  %507 = insertelement <4 x float> %504, float %506, i64 3, !dbg !27
  %math.result300 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %507) #9, !dbg !27, !frisk.register_lifetimes !1
  %508 = fptrunc <4 x float> %math.result300 to <4 x half>, !dbg !27
  %math.packed301 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %508) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %509 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed301, <4 x half> %469, <4 x float> %494) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %510 = shufflevector <8 x float> %492, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %511 = fmul <4 x float> %510, %448, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %512 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed301, <4 x half> %488, <4 x float> %511) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %513 = shufflevector <8 x half> %457, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %514 = fsub <4 x float> %312, %434, !dbg !21
  %515 = extractelement <4 x float> %514, i64 0, !dbg !25
  %516 = call float @llvm.amdgcn.exp2.f32(float %515), !dbg !25
  %local.reduce81 = fadd float %475, %516, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor = call float asm sideeffect "", "=v,0"(float %local.reduce81) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %517 = insertelement <4 x float> poison, float %516, i64 0, !dbg !27
  %518 = extractelement <4 x float> %514, i64 1, !dbg !25
  %519 = call float @llvm.amdgcn.exp2.f32(float %518), !dbg !25
  %local.reduce82 = fadd float %478, %519, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1504 = call float asm sideeffect "", "=v,0"(float %local.reduce82) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %520 = insertelement <4 x float> %517, float %519, i64 1, !dbg !27
  %521 = extractelement <4 x float> %514, i64 2, !dbg !25
  %522 = call float @llvm.amdgcn.exp2.f32(float %521), !dbg !25
  %local.reduce83 = fadd float %481, %522, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1505 = call float asm sideeffect "", "=v,0"(float %local.reduce83) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %523 = insertelement <4 x float> %520, float %522, i64 2, !dbg !27
  %524 = extractelement <4 x float> %514, i64 3, !dbg !25
  %525 = call float @llvm.amdgcn.exp2.f32(float %524), !dbg !25
  %local.reduce84 = fadd float %484, %525, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1506 = call float asm sideeffect "", "=v,0"(float %local.reduce84) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %526 = insertelement <4 x float> %523, float %525, i64 3, !dbg !27
  %math.result286 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %526) #9, !dbg !27, !frisk.register_lifetimes !1
  %527 = fptrunc <4 x float> %math.result286 to <4 x half>, !dbg !27
  %math.packed287 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %527) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %528 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed287, <4 x half> %513, <4 x float> %487) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %529 = shufflevector <8 x half> %457, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %530 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed287, <4 x half> %529, <4 x float> %491) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %531 = fsub <4 x float> %369, %435, !dbg !21
  %532 = extractelement <4 x float> %531, i64 0, !dbg !25
  %533 = call float @llvm.amdgcn.exp2.f32(float %532), !dbg !25
  %local.reduce = fadd float %497, %533, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1507 = call float asm sideeffect "", "=v,0"(float %local.reduce) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %534 = insertelement <4 x float> poison, float %533, i64 0, !dbg !27
  %535 = extractelement <4 x float> %531, i64 1, !dbg !25
  %536 = call float @llvm.amdgcn.exp2.f32(float %535), !dbg !25
  %local.reduce51 = fadd float %500, %536, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1508 = call float asm sideeffect "", "=v,0"(float %local.reduce51) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %537 = insertelement <4 x float> %534, float %536, i64 1, !dbg !27
  %538 = extractelement <4 x float> %531, i64 2, !dbg !25
  %539 = call float @llvm.amdgcn.exp2.f32(float %538), !dbg !25
  %local.reduce52 = fadd float %503, %539, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1509 = call float asm sideeffect "", "=v,0"(float %local.reduce52) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %540 = insertelement <4 x float> %537, float %539, i64 2, !dbg !27
  %541 = extractelement <4 x float> %531, i64 3, !dbg !25
  %542 = call float @llvm.amdgcn.exp2.f32(float %541), !dbg !25
  %local.reduce53 = fadd float %506, %542, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1510 = call float asm sideeffect "", "=v,0"(float %local.reduce53) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %543 = insertelement <4 x float> %540, float %542, i64 3, !dbg !27
  %math.result302 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %543) #9, !dbg !27, !frisk.register_lifetimes !1
  %544 = fptrunc <4 x float> %math.result302 to <4 x half>, !dbg !27
  %math.packed303 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %544) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %545 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed303, <4 x half> %513, <4 x float> %509) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %546 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed303, <4 x half> %529, <4 x float> %512) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %547 = shufflevector <8 x half> %459, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %548 = fsub <4 x float> %322, %434, !dbg !21
  %549 = extractelement <4 x float> %548, i64 0, !dbg !25
  %550 = call float @llvm.amdgcn.exp2.f32(float %549), !dbg !25
  %local.reduce85 = fadd float %k0.sum.anchor, %550, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1511 = call float asm sideeffect "", "=v,0"(float %local.reduce85) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %551 = insertelement <4 x float> poison, float %550, i64 0, !dbg !27
  %552 = extractelement <4 x float> %548, i64 1, !dbg !25
  %553 = call float @llvm.amdgcn.exp2.f32(float %552), !dbg !25
  %local.reduce86 = fadd float %k0.sum.anchor1504, %553, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1512 = call float asm sideeffect "", "=v,0"(float %local.reduce86) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %554 = insertelement <4 x float> %551, float %553, i64 1, !dbg !27
  %555 = extractelement <4 x float> %548, i64 2, !dbg !25
  %556 = call float @llvm.amdgcn.exp2.f32(float %555), !dbg !25
  %local.reduce87 = fadd float %k0.sum.anchor1505, %556, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1513 = call float asm sideeffect "", "=v,0"(float %local.reduce87) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %557 = insertelement <4 x float> %554, float %556, i64 2, !dbg !27
  %558 = extractelement <4 x float> %548, i64 3, !dbg !25
  %559 = call float @llvm.amdgcn.exp2.f32(float %558), !dbg !25
  %local.reduce88 = fadd float %k0.sum.anchor1506, %559, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1514 = call float asm sideeffect "", "=v,0"(float %local.reduce88) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %560 = insertelement <4 x float> %557, float %559, i64 3, !dbg !27
  %math.result288 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %560) #9, !dbg !27, !frisk.register_lifetimes !1
  %561 = fptrunc <4 x float> %math.result288 to <4 x half>, !dbg !27
  %math.packed289 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %561) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %562 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed289, <4 x half> %547, <4 x float> %528) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %563 = shufflevector <8 x half> %459, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %564 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed289, <4 x half> %563, <4 x float> %530) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %565 = fsub <4 x float> %375, %435, !dbg !21
  %566 = extractelement <4 x float> %565, i64 0, !dbg !25
  %567 = call float @llvm.amdgcn.exp2.f32(float %566), !dbg !25
  %local.reduce54 = fadd float %k0.sum.anchor1507, %567, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1515 = call float asm sideeffect "", "=v,0"(float %local.reduce54) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %568 = insertelement <4 x float> poison, float %567, i64 0, !dbg !27
  %569 = extractelement <4 x float> %565, i64 1, !dbg !25
  %570 = call float @llvm.amdgcn.exp2.f32(float %569), !dbg !25
  %local.reduce55 = fadd float %k0.sum.anchor1508, %570, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1516 = call float asm sideeffect "", "=v,0"(float %local.reduce55) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %571 = insertelement <4 x float> %568, float %570, i64 1, !dbg !27
  %572 = extractelement <4 x float> %565, i64 2, !dbg !25
  %573 = call float @llvm.amdgcn.exp2.f32(float %572), !dbg !25
  %local.reduce56 = fadd float %k0.sum.anchor1509, %573, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1517 = call float asm sideeffect "", "=v,0"(float %local.reduce56) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %574 = insertelement <4 x float> %571, float %573, i64 2, !dbg !27
  %575 = extractelement <4 x float> %565, i64 3, !dbg !25
  %576 = call float @llvm.amdgcn.exp2.f32(float %575), !dbg !25
  %local.reduce57 = fadd float %k0.sum.anchor1510, %576, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1518 = call float asm sideeffect "", "=v,0"(float %local.reduce57) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %577 = insertelement <4 x float> %574, float %576, i64 3, !dbg !27
  %math.result304 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %577) #9, !dbg !27, !frisk.register_lifetimes !1
  %578 = fptrunc <4 x float> %math.result304 to <4 x half>, !dbg !27
  %math.packed305 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %578) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %579 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed305, <4 x half> %547, <4 x float> %545) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %580 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed305, <4 x half> %563, <4 x float> %546) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %581 = shufflevector <8 x half> %461, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %582 = fsub <4 x float> %328, %434, !dbg !21
  %583 = extractelement <4 x float> %582, i64 0, !dbg !25
  %584 = call float @llvm.amdgcn.exp2.f32(float %583), !dbg !25
  %local.reduce89 = fadd float %k0.sum.anchor1511, %584, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1519 = call float asm sideeffect "", "=v,0"(float %local.reduce89) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %585 = insertelement <4 x float> poison, float %584, i64 0, !dbg !27
  %586 = extractelement <4 x float> %582, i64 1, !dbg !25
  %587 = call float @llvm.amdgcn.exp2.f32(float %586), !dbg !25
  %local.reduce90 = fadd float %k0.sum.anchor1512, %587, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1520 = call float asm sideeffect "", "=v,0"(float %local.reduce90) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %588 = insertelement <4 x float> %585, float %587, i64 1, !dbg !27
  %589 = extractelement <4 x float> %582, i64 2, !dbg !25
  %590 = call float @llvm.amdgcn.exp2.f32(float %589), !dbg !25
  %local.reduce91 = fadd float %k0.sum.anchor1513, %590, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1521 = call float asm sideeffect "", "=v,0"(float %local.reduce91) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %591 = insertelement <4 x float> %588, float %590, i64 2, !dbg !27
  %592 = extractelement <4 x float> %582, i64 3, !dbg !25
  %593 = call float @llvm.amdgcn.exp2.f32(float %592), !dbg !25
  %local.reduce92 = fadd float %k0.sum.anchor1514, %593, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1522 = call float asm sideeffect "", "=v,0"(float %local.reduce92) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %594 = insertelement <4 x float> %591, float %593, i64 3, !dbg !27
  %math.result290 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %594) #9, !dbg !27, !frisk.register_lifetimes !1
  %595 = fptrunc <4 x float> %math.result290 to <4 x half>, !dbg !27
  %math.packed291 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %595) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %596 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed291, <4 x half> %581, <4 x float> %562) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %597 = shufflevector <8 x half> %461, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %598 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed291, <4 x half> %597, <4 x float> %564) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %599 = fsub <4 x float> %381, %435, !dbg !21
  %600 = extractelement <4 x float> %599, i64 0, !dbg !25
  %601 = call float @llvm.amdgcn.exp2.f32(float %600), !dbg !25
  %local.reduce58 = fadd float %k0.sum.anchor1515, %601, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1523 = call float asm sideeffect "", "=v,0"(float %local.reduce58) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %602 = insertelement <4 x float> poison, float %601, i64 0, !dbg !27
  %603 = extractelement <4 x float> %599, i64 1, !dbg !25
  %604 = call float @llvm.amdgcn.exp2.f32(float %603), !dbg !25
  %local.reduce59 = fadd float %k0.sum.anchor1516, %604, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1524 = call float asm sideeffect "", "=v,0"(float %local.reduce59) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %605 = insertelement <4 x float> %602, float %604, i64 1, !dbg !27
  %606 = extractelement <4 x float> %599, i64 2, !dbg !25
  %607 = call float @llvm.amdgcn.exp2.f32(float %606), !dbg !25
  %local.reduce60 = fadd float %k0.sum.anchor1517, %607, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1525 = call float asm sideeffect "", "=v,0"(float %local.reduce60) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %608 = insertelement <4 x float> %605, float %607, i64 2, !dbg !27
  %609 = extractelement <4 x float> %599, i64 3, !dbg !25
  %610 = call float @llvm.amdgcn.exp2.f32(float %609), !dbg !25
  %local.reduce61 = fadd float %k0.sum.anchor1518, %610, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1526 = call float asm sideeffect "", "=v,0"(float %local.reduce61) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %611 = insertelement <4 x float> %608, float %610, i64 3, !dbg !27
  %math.result306 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %611) #9, !dbg !27, !frisk.register_lifetimes !1
  %612 = fptrunc <4 x float> %math.result306 to <4 x half>, !dbg !27
  %math.packed307 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %612) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %613 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed307, <4 x half> %581, <4 x float> %579) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %614 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed307, <4 x half> %597, <4 x float> %580) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %615 = shufflevector <8 x half> %463, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %616 = fsub <4 x float> %338, %434, !dbg !21
  %617 = extractelement <4 x float> %616, i64 0, !dbg !25
  %618 = call float @llvm.amdgcn.exp2.f32(float %617), !dbg !25
  %local.reduce93 = fadd float %k0.sum.anchor1519, %618, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1527 = call float asm sideeffect "", "=v,0"(float %local.reduce93) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %619 = insertelement <4 x float> poison, float %618, i64 0, !dbg !27
  %620 = extractelement <4 x float> %616, i64 1, !dbg !25
  %621 = call float @llvm.amdgcn.exp2.f32(float %620), !dbg !25
  %local.reduce94 = fadd float %k0.sum.anchor1520, %621, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1528 = call float asm sideeffect "", "=v,0"(float %local.reduce94) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %622 = insertelement <4 x float> %619, float %621, i64 1, !dbg !27
  %623 = extractelement <4 x float> %616, i64 2, !dbg !25
  %624 = call float @llvm.amdgcn.exp2.f32(float %623), !dbg !25
  %local.reduce95 = fadd float %k0.sum.anchor1521, %624, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1529 = call float asm sideeffect "", "=v,0"(float %local.reduce95) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %625 = insertelement <4 x float> %622, float %624, i64 2, !dbg !27
  %626 = extractelement <4 x float> %616, i64 3, !dbg !25
  %627 = call float @llvm.amdgcn.exp2.f32(float %626), !dbg !25
  %local.reduce96 = fadd float %k0.sum.anchor1522, %627, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1530 = call float asm sideeffect "", "=v,0"(float %local.reduce96) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %628 = insertelement <4 x float> %625, float %627, i64 3, !dbg !27
  %math.result292 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %628) #9, !dbg !27, !frisk.register_lifetimes !1
  %629 = fptrunc <4 x float> %math.result292 to <4 x half>, !dbg !27
  %math.packed293 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %629) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %630 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed293, <4 x half> %615, <4 x float> %596) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %631 = shufflevector <8 x half> %463, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %632 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed293, <4 x half> %631, <4 x float> %598) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %633 = fsub <4 x float> %387, %435, !dbg !21
  %634 = extractelement <4 x float> %633, i64 0, !dbg !25
  %635 = call float @llvm.amdgcn.exp2.f32(float %634), !dbg !25
  %local.reduce62 = fadd float %k0.sum.anchor1523, %635, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1531 = call float asm sideeffect "", "=v,0"(float %local.reduce62) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %636 = insertelement <4 x float> poison, float %635, i64 0, !dbg !27
  %637 = extractelement <4 x float> %633, i64 1, !dbg !25
  %638 = call float @llvm.amdgcn.exp2.f32(float %637), !dbg !25
  %local.reduce63 = fadd float %k0.sum.anchor1524, %638, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1532 = call float asm sideeffect "", "=v,0"(float %local.reduce63) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %639 = insertelement <4 x float> %636, float %638, i64 1, !dbg !27
  %640 = extractelement <4 x float> %633, i64 2, !dbg !25
  %641 = call float @llvm.amdgcn.exp2.f32(float %640), !dbg !25
  %local.reduce64 = fadd float %k0.sum.anchor1525, %641, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1533 = call float asm sideeffect "", "=v,0"(float %local.reduce64) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %642 = insertelement <4 x float> %639, float %641, i64 2, !dbg !27
  %643 = extractelement <4 x float> %633, i64 3, !dbg !25
  %644 = call float @llvm.amdgcn.exp2.f32(float %643), !dbg !25
  %local.reduce65 = fadd float %k0.sum.anchor1526, %644, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1534 = call float asm sideeffect "", "=v,0"(float %local.reduce65) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %645 = insertelement <4 x float> %642, float %644, i64 3, !dbg !27
  %math.result308 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %645) #9, !dbg !27, !frisk.register_lifetimes !1
  %646 = fptrunc <4 x float> %math.result308 to <4 x half>, !dbg !27
  %math.packed309 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %646) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %647 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed309, <4 x half> %615, <4 x float> %613) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %648 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed309, <4 x half> %631, <4 x float> %614) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %649 = shufflevector <8 x half> %465, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %650 = fsub <4 x float> %344, %434, !dbg !21
  %651 = extractelement <4 x float> %650, i64 0, !dbg !25
  %652 = call float @llvm.amdgcn.exp2.f32(float %651), !dbg !25
  %local.reduce97 = fadd float %k0.sum.anchor1527, %652, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1535 = call float asm sideeffect "", "=v,0"(float %local.reduce97) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %653 = insertelement <4 x float> poison, float %652, i64 0, !dbg !27
  %654 = extractelement <4 x float> %650, i64 1, !dbg !25
  %655 = call float @llvm.amdgcn.exp2.f32(float %654), !dbg !25
  %local.reduce98 = fadd float %k0.sum.anchor1528, %655, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1536 = call float asm sideeffect "", "=v,0"(float %local.reduce98) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %656 = insertelement <4 x float> %653, float %655, i64 1, !dbg !27
  %657 = extractelement <4 x float> %650, i64 2, !dbg !25
  %658 = call float @llvm.amdgcn.exp2.f32(float %657), !dbg !25
  %local.reduce99 = fadd float %k0.sum.anchor1529, %658, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1537 = call float asm sideeffect "", "=v,0"(float %local.reduce99) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %659 = insertelement <4 x float> %656, float %658, i64 2, !dbg !27
  %660 = extractelement <4 x float> %650, i64 3, !dbg !25
  %661 = call float @llvm.amdgcn.exp2.f32(float %660), !dbg !25
  %local.reduce100 = fadd float %k0.sum.anchor1530, %661, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1538 = call float asm sideeffect "", "=v,0"(float %local.reduce100) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %662 = insertelement <4 x float> %659, float %661, i64 3, !dbg !27
  %math.result294 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %662) #9, !dbg !27, !frisk.register_lifetimes !1
  %663 = fptrunc <4 x float> %math.result294 to <4 x half>, !dbg !27
  %math.packed295 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %663) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %664 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed295, <4 x half> %649, <4 x float> %630) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %665 = shufflevector <8 x half> %465, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %666 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed295, <4 x half> %665, <4 x float> %632) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %667 = fsub <4 x float> %393, %435, !dbg !21
  %668 = extractelement <4 x float> %667, i64 0, !dbg !25
  %669 = call float @llvm.amdgcn.exp2.f32(float %668), !dbg !25
  %local.reduce66 = fadd float %k0.sum.anchor1531, %669, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1539 = call float asm sideeffect "", "=v,0"(float %local.reduce66) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %670 = insertelement <4 x float> poison, float %669, i64 0, !dbg !27
  %671 = extractelement <4 x float> %667, i64 1, !dbg !25
  %672 = call float @llvm.amdgcn.exp2.f32(float %671), !dbg !25
  %local.reduce67 = fadd float %k0.sum.anchor1532, %672, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1540 = call float asm sideeffect "", "=v,0"(float %local.reduce67) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %673 = insertelement <4 x float> %670, float %672, i64 1, !dbg !27
  %674 = extractelement <4 x float> %667, i64 2, !dbg !25
  %675 = call float @llvm.amdgcn.exp2.f32(float %674), !dbg !25
  %local.reduce68 = fadd float %k0.sum.anchor1533, %675, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1541 = call float asm sideeffect "", "=v,0"(float %local.reduce68) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %676 = insertelement <4 x float> %673, float %675, i64 2, !dbg !27
  %677 = extractelement <4 x float> %667, i64 3, !dbg !25
  %678 = call float @llvm.amdgcn.exp2.f32(float %677), !dbg !25
  %local.reduce69 = fadd float %k0.sum.anchor1534, %678, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1542 = call float asm sideeffect "", "=v,0"(float %local.reduce69) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %679 = insertelement <4 x float> %676, float %678, i64 3, !dbg !27
  %math.result310 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %679) #9, !dbg !27, !frisk.register_lifetimes !1
  %680 = fptrunc <4 x float> %math.result310 to <4 x half>, !dbg !27
  %math.packed311 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %680) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %681 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed311, <4 x half> %649, <4 x float> %647) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %682 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed311, <4 x half> %665, <4 x float> %648) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %683 = shufflevector <8 x half> %467, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %684 = fsub <4 x float> %350, %434, !dbg !21
  %685 = extractelement <4 x float> %684, i64 0, !dbg !25
  %686 = call float @llvm.amdgcn.exp2.f32(float %685), !dbg !25
  %local.reduce101 = fadd float %k0.sum.anchor1535, %686, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1543 = call float asm sideeffect "", "=v,0"(float %local.reduce101) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %687 = insertelement <4 x float> poison, float %686, i64 0, !dbg !27
  %688 = extractelement <4 x float> %684, i64 1, !dbg !25
  %689 = call float @llvm.amdgcn.exp2.f32(float %688), !dbg !25
  %local.reduce102 = fadd float %k0.sum.anchor1536, %689, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1544 = call float asm sideeffect "", "=v,0"(float %local.reduce102) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %690 = insertelement <4 x float> %687, float %689, i64 1, !dbg !27
  %691 = extractelement <4 x float> %684, i64 2, !dbg !25
  %692 = call float @llvm.amdgcn.exp2.f32(float %691), !dbg !25
  %local.reduce103 = fadd float %k0.sum.anchor1537, %692, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1545 = call float asm sideeffect "", "=v,0"(float %local.reduce103) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %693 = insertelement <4 x float> %690, float %692, i64 2, !dbg !27
  %694 = extractelement <4 x float> %684, i64 3, !dbg !25
  %695 = call float @llvm.amdgcn.exp2.f32(float %694), !dbg !25
  %local.reduce104 = fadd float %k0.sum.anchor1538, %695, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1546 = call float asm sideeffect "", "=v,0"(float %local.reduce104) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %696 = insertelement <4 x float> %693, float %695, i64 3, !dbg !27
  %math.result296 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %696) #9, !dbg !27, !frisk.register_lifetimes !1
  %697 = fptrunc <4 x float> %math.result296 to <4 x half>, !dbg !27
  %math.packed297 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %697) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %698 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed297, <4 x half> %683, <4 x float> %664) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %699 = shufflevector <8 x half> %467, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %700 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed297, <4 x half> %699, <4 x float> %666) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %701 = fsub <4 x float> %399, %435, !dbg !21
  %702 = extractelement <4 x float> %701, i64 0, !dbg !25
  %703 = call float @llvm.amdgcn.exp2.f32(float %702), !dbg !25
  %local.reduce70 = fadd float %k0.sum.anchor1539, %703, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1547 = call float asm sideeffect "", "=v,0"(float %local.reduce70) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %704 = insertelement <4 x float> poison, float %703, i64 0, !dbg !27
  %705 = extractelement <4 x float> %701, i64 1, !dbg !25
  %706 = call float @llvm.amdgcn.exp2.f32(float %705), !dbg !25
  %local.reduce71 = fadd float %k0.sum.anchor1540, %706, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1548 = call float asm sideeffect "", "=v,0"(float %local.reduce71) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %707 = insertelement <4 x float> %704, float %706, i64 1, !dbg !27
  %708 = extractelement <4 x float> %701, i64 2, !dbg !25
  %709 = call float @llvm.amdgcn.exp2.f32(float %708), !dbg !25
  %local.reduce72 = fadd float %k0.sum.anchor1541, %709, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1549 = call float asm sideeffect "", "=v,0"(float %local.reduce72) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %710 = insertelement <4 x float> %707, float %709, i64 2, !dbg !27
  %711 = extractelement <4 x float> %701, i64 3, !dbg !25
  %712 = call float @llvm.amdgcn.exp2.f32(float %711), !dbg !25
  %local.reduce73 = fadd float %k0.sum.anchor1542, %712, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1550 = call float asm sideeffect "", "=v,0"(float %local.reduce73) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %713 = insertelement <4 x float> %710, float %712, i64 3, !dbg !27
  %math.result312 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %713) #9, !dbg !27, !frisk.register_lifetimes !1
  %714 = fptrunc <4 x float> %math.result312 to <4 x half>, !dbg !27
  %math.packed313 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %714) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %715 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed313, <4 x half> %683, <4 x float> %681) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %716 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed313, <4 x half> %699, <4 x float> %682) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %717 = bitcast <4 x i32> %468 to <8 x half>, !dbg !14
  %718 = shufflevector <8 x half> %717, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %719 = fsub <4 x float> %356, %434, !dbg !21
  %720 = extractelement <4 x float> %719, i64 0, !dbg !25
  %721 = call float @llvm.amdgcn.exp2.f32(float %720), !dbg !25
  %local.reduce105 = fadd float %k0.sum.anchor1543, %721, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1551 = call float asm sideeffect "", "=v,0"(float %local.reduce105) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %722 = insertelement <4 x float> poison, float %721, i64 0, !dbg !27
  %723 = extractelement <4 x float> %719, i64 1, !dbg !25
  %724 = call float @llvm.amdgcn.exp2.f32(float %723), !dbg !25
  %local.reduce106 = fadd float %k0.sum.anchor1544, %724, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1552 = call float asm sideeffect "", "=v,0"(float %local.reduce106) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %local.reduce109 = fadd float %k0.sum.anchor1551, %k0.sum.anchor1552, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1553 = call float asm sideeffect "", "=v,0"(float %local.reduce109) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %725 = insertelement <4 x float> %722, float %724, i64 1, !dbg !27
  %726 = extractelement <4 x float> %719, i64 2, !dbg !25
  %727 = call float @llvm.amdgcn.exp2.f32(float %726), !dbg !25
  %local.reduce107 = fadd float %k0.sum.anchor1545, %727, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1554 = call float asm sideeffect "", "=v,0"(float %local.reduce107) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %728 = insertelement <4 x float> %725, float %727, i64 2, !dbg !27
  %729 = extractelement <4 x float> %719, i64 3, !dbg !25
  %730 = call float @llvm.amdgcn.exp2.f32(float %729), !dbg !25
  %local.reduce108 = fadd float %k0.sum.anchor1546, %730, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1555 = call float asm sideeffect "", "=v,0"(float %local.reduce108) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %local.reduce110 = fadd float %k0.sum.anchor1554, %k0.sum.anchor1555, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1556 = call float asm sideeffect "", "=v,0"(float %local.reduce110) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %731 = fadd float %k0.sum.anchor1553, %k0.sum.anchor1556, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1557 = call float asm sideeffect "", "=v,0"(float %731) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %732 = fadd float %k0.sum.anchor1557, %443, !dbg !26
  %k0.sum.anchor1558 = call float asm sideeffect "", "=v,0"(float %732) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %733 = insertelement <1 x float> poison, float %k0.sum.anchor1558, i64 0, !dbg !8
  %734 = insertvalue [2 x <1 x float>] poison, <1 x float> %733, 0, !dbg !8
  %735 = insertelement <4 x float> %728, float %730, i64 3, !dbg !27
  %math.result298 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %735) #9, !dbg !27, !frisk.register_lifetimes !1
  %736 = fptrunc <4 x float> %math.result298 to <4 x half>, !dbg !27
  %math.packed299 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %736) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %737 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed299, <4 x half> %718, <4 x float> %698) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %738 = shufflevector <8 x half> %717, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %739 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed299, <4 x half> %738, <4 x float> %700) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %740 = fsub <4 x float> %405, %435, !dbg !21
  %741 = extractelement <4 x float> %740, i64 0, !dbg !25
  %742 = call float @llvm.amdgcn.exp2.f32(float %741), !dbg !25
  %local.reduce74 = fadd float %k0.sum.anchor1547, %742, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1559 = call float asm sideeffect "", "=v,0"(float %local.reduce74) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %743 = insertelement <4 x float> poison, float %742, i64 0, !dbg !27
  %744 = extractelement <4 x float> %740, i64 1, !dbg !25
  %745 = call float @llvm.amdgcn.exp2.f32(float %744), !dbg !25
  %local.reduce75 = fadd float %k0.sum.anchor1548, %745, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1560 = call float asm sideeffect "", "=v,0"(float %local.reduce75) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %local.reduce78 = fadd float %k0.sum.anchor1559, %k0.sum.anchor1560, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1561 = call float asm sideeffect "", "=v,0"(float %local.reduce78) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %746 = insertelement <4 x float> %743, float %745, i64 1, !dbg !27
  %747 = extractelement <4 x float> %740, i64 2, !dbg !25
  %748 = call float @llvm.amdgcn.exp2.f32(float %747), !dbg !25
  %local.reduce76 = fadd float %k0.sum.anchor1549, %748, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1562 = call float asm sideeffect "", "=v,0"(float %local.reduce76) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %749 = insertelement <4 x float> %746, float %748, i64 2, !dbg !27
  %750 = extractelement <4 x float> %740, i64 3, !dbg !25
  %751 = call float @llvm.amdgcn.exp2.f32(float %750), !dbg !25
  %local.reduce77 = fadd float %k0.sum.anchor1550, %751, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1563 = call float asm sideeffect "", "=v,0"(float %local.reduce77) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %local.reduce79 = fadd float %k0.sum.anchor1562, %k0.sum.anchor1563, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1564 = call float asm sideeffect "", "=v,0"(float %local.reduce79) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %752 = fadd float %k0.sum.anchor1561, %k0.sum.anchor1564, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1565 = call float asm sideeffect "", "=v,0"(float %752) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %753 = fadd float %k0.sum.anchor1565, %451, !dbg !26
  %k0.sum.anchor1566 = call float asm sideeffect "", "=v,0"(float %753) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %754 = insertelement <1 x float> poison, float %k0.sum.anchor1566, i64 0, !dbg !8
  %755 = insertvalue [2 x <1 x float>] %734, <1 x float> %754, 1, !dbg !8
  %756 = insertelement <4 x float> %749, float %751, i64 3, !dbg !27
  %math.result314 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %756) #9, !dbg !27, !frisk.register_lifetimes !1
  %757 = fptrunc <4 x float> %math.result314 to <4 x half>, !dbg !27
  %math.packed315 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %757) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %758 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed315, <4 x half> %718, <4 x float> %715) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %759 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed315, <4 x half> %738, <4 x float> %716) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %vdb.address1072 = or disjoint i64 %185, %address.local572, !dbg !16
  %vdb.address1073 = trunc nuw i64 %vdb.address1072 to i32, !dbg !16
  %vdb.address1074 = shl nuw nsw i32 %vdb.address1073, 1, !dbg !16
  %vdb.address1079 = or disjoint i64 %185, %address.local578, !dbg !16
  %vdb.address1080 = trunc nuw i64 %vdb.address1079 to i32, !dbg !16
  %vdb.address1081 = shl nuw nsw i32 %vdb.address1080, 1, !dbg !16
  %760 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %36, i32 %vdb.address1074, <4 x i32> %dma.buffer.rsrc973) #10, !dbg !16
  %761 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %43, i32 %vdb.address1081, <4 x i32> %dma.buffer.rsrc973) #10, !dbg !16
  %vdb.read = getelementptr i8, ptr addrspace(3) %address.local583, i64 8192, !dbg !14
  %762 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %763 = bitcast <4 x i32> %762 to <8 x half>, !dbg !14
  %vdb.read1065 = getelementptr i8, ptr addrspace(3) %address.local583, i64 9216, !dbg !14
  %764 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1065, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %765 = bitcast <4 x i32> %764 to <8 x half>, !dbg !14
  %vdb.read1066 = getelementptr i8, ptr addrspace(3) %address.local583, i64 10240, !dbg !14
  %766 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1066, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %767 = bitcast <4 x i32> %766 to <8 x half>, !dbg !14
  %vdb.read1067 = getelementptr i8, ptr addrspace(3) %address.local583, i64 11264, !dbg !14
  %768 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1067, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %769 = bitcast <4 x i32> %768 to <8 x half>, !dbg !14
  %vdb.read1068 = getelementptr i8, ptr addrspace(3) %address.local583, i64 12288, !dbg !14
  %770 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1068, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %771 = bitcast <4 x i32> %770 to <8 x half>, !dbg !14
  %vdb.read1069 = getelementptr i8, ptr addrspace(3) %address.local583, i64 13312, !dbg !14
  %772 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1069, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %773 = bitcast <4 x i32> %772 to <8 x half>, !dbg !14
  %vdb.read1070 = getelementptr i8, ptr addrspace(3) %address.local583, i64 14336, !dbg !14
  %774 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1070, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %775 = bitcast <4 x i32> %774 to <8 x half>, !dbg !14
  %vdb.read1071 = getelementptr i8, ptr addrspace(3) %address.local583, i64 15360, !dbg !14
  %776 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1071, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %777 = shufflevector <8 x half> %763, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %778 = extractvalue [2 x <8 x float>] %102, 0, !dbg !28
  %779 = shufflevector <8 x float> %778, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %780 = fmul <4 x float> %779, %440, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %781 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %777, <4 x float> %780) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %782 = shufflevector <8 x half> %763, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %783 = shufflevector <8 x float> %778, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %784 = fmul <4 x float> %783, %440, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %785 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %782, <4 x float> %784) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %786 = extractvalue [2 x <8 x float>] %102, 1, !dbg !28
  %787 = shufflevector <8 x float> %786, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %788 = fmul <4 x float> %787, %448, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %789 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed301, <4 x half> %777, <4 x float> %788) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %790 = shufflevector <8 x float> %786, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %791 = fmul <4 x float> %790, %448, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %792 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed301, <4 x half> %782, <4 x float> %791) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %793 = shufflevector <8 x half> %765, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %794 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed287, <4 x half> %793, <4 x float> %781) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %795 = shufflevector <8 x half> %765, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %796 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed287, <4 x half> %795, <4 x float> %785) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %797 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed303, <4 x half> %793, <4 x float> %789) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %798 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed303, <4 x half> %795, <4 x float> %792) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %799 = shufflevector <8 x half> %767, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %800 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed289, <4 x half> %799, <4 x float> %794) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %801 = shufflevector <8 x half> %767, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %802 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed289, <4 x half> %801, <4 x float> %796) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %803 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed305, <4 x half> %799, <4 x float> %797) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %804 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed305, <4 x half> %801, <4 x float> %798) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %805 = shufflevector <8 x half> %769, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %806 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed291, <4 x half> %805, <4 x float> %800) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %807 = shufflevector <8 x half> %769, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %808 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed291, <4 x half> %807, <4 x float> %802) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %809 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed307, <4 x half> %805, <4 x float> %803) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %810 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed307, <4 x half> %807, <4 x float> %804) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %811 = shufflevector <8 x half> %771, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %812 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed293, <4 x half> %811, <4 x float> %806) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %813 = shufflevector <8 x half> %771, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %814 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed293, <4 x half> %813, <4 x float> %808) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %815 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed309, <4 x half> %811, <4 x float> %809) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %816 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed309, <4 x half> %813, <4 x float> %810) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %817 = shufflevector <8 x half> %773, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %818 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed295, <4 x half> %817, <4 x float> %812) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %819 = shufflevector <8 x half> %773, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %820 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed295, <4 x half> %819, <4 x float> %814) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %821 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed311, <4 x half> %817, <4 x float> %815) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %822 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed311, <4 x half> %819, <4 x float> %816) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %823 = shufflevector <8 x half> %775, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %824 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed297, <4 x half> %823, <4 x float> %818) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %825 = shufflevector <8 x half> %775, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %826 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed297, <4 x half> %825, <4 x float> %820) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %827 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed313, <4 x half> %823, <4 x float> %821) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %828 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed313, <4 x half> %825, <4 x float> %822) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %829 = bitcast <4 x i32> %776 to <8 x half>, !dbg !14
  %830 = shufflevector <8 x half> %829, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %831 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed299, <4 x half> %830, <4 x float> %824) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %832 = shufflevector <8 x half> %829, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %833 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed299, <4 x half> %832, <4 x float> %826) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %834 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed315, <4 x half> %830, <4 x float> %827) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %835 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed315, <4 x half> %832, <4 x float> %828) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %vdb.address1086 = or disjoint i64 %238, %address.local572, !dbg !16
  %vdb.address1087 = trunc nuw i64 %vdb.address1086 to i32, !dbg !16
  %vdb.address1088 = shl nuw nsw i32 %vdb.address1087, 1, !dbg !16
  %vdb.address1093 = or disjoint i64 %238, %address.local578, !dbg !16
  %vdb.address1094 = trunc nuw i64 %vdb.address1093 to i32, !dbg !16
  %vdb.address1095 = shl nuw nsw i32 %vdb.address1094, 1, !dbg !16
  %836 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1088, <4 x i32> %dma.buffer.rsrc973) #10, !dbg !16
  %837 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1064, i32 %vdb.address1095, <4 x i32> %dma.buffer.rsrc973) #10, !dbg !16
  %838 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local583, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %839 = bitcast <4 x i32> %838 to <8 x half>, !dbg !14
  %840 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local585, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %841 = bitcast <4 x i32> %840 to <8 x half>, !dbg !14
  %842 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local587, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %843 = bitcast <4 x i32> %842 to <8 x half>, !dbg !14
  %844 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local589, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %845 = bitcast <4 x i32> %844 to <8 x half>, !dbg !14
  %846 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local591, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %847 = bitcast <4 x i32> %846 to <8 x half>, !dbg !14
  %848 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local593, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %849 = bitcast <4 x i32> %848 to <8 x half>, !dbg !14
  %850 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local595, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %851 = bitcast <4 x i32> %850 to <8 x half>, !dbg !14
  %852 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local597, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %853 = shufflevector <8 x half> %839, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %854 = extractvalue [2 x <8 x float>] %103, 0, !dbg !28
  %855 = shufflevector <8 x float> %854, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %856 = fmul <4 x float> %855, %440, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %857 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %853, <4 x float> %856) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %858 = shufflevector <8 x half> %839, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %859 = shufflevector <8 x float> %854, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %860 = fmul <4 x float> %859, %440, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %861 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %858, <4 x float> %860) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %862 = extractvalue [2 x <8 x float>] %103, 1, !dbg !28
  %863 = shufflevector <8 x float> %862, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %864 = fmul <4 x float> %863, %448, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %865 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed301, <4 x half> %853, <4 x float> %864) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %866 = shufflevector <8 x float> %862, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %867 = fmul <4 x float> %866, %448, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %868 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed301, <4 x half> %858, <4 x float> %867) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %869 = shufflevector <8 x half> %841, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %870 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed287, <4 x half> %869, <4 x float> %857) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %871 = shufflevector <8 x half> %841, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %872 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed287, <4 x half> %871, <4 x float> %861) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %873 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed303, <4 x half> %869, <4 x float> %865) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %874 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed303, <4 x half> %871, <4 x float> %868) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %875 = shufflevector <8 x half> %843, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %876 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed289, <4 x half> %875, <4 x float> %870) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %877 = shufflevector <8 x half> %843, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %878 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed289, <4 x half> %877, <4 x float> %872) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %879 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed305, <4 x half> %875, <4 x float> %873) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %880 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed305, <4 x half> %877, <4 x float> %874) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %881 = shufflevector <8 x half> %845, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %882 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed291, <4 x half> %881, <4 x float> %876) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %883 = shufflevector <8 x half> %845, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %884 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed291, <4 x half> %883, <4 x float> %878) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %885 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed307, <4 x half> %881, <4 x float> %879) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %886 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed307, <4 x half> %883, <4 x float> %880) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %887 = shufflevector <8 x half> %847, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %888 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed293, <4 x half> %887, <4 x float> %882) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %889 = shufflevector <8 x half> %847, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %890 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed293, <4 x half> %889, <4 x float> %884) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %891 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed309, <4 x half> %887, <4 x float> %885) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %892 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed309, <4 x half> %889, <4 x float> %886) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %893 = shufflevector <8 x half> %849, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %894 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed295, <4 x half> %893, <4 x float> %888) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %895 = shufflevector <8 x half> %849, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %896 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed295, <4 x half> %895, <4 x float> %890) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %897 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed311, <4 x half> %893, <4 x float> %891) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %898 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed311, <4 x half> %895, <4 x float> %892) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %899 = shufflevector <8 x half> %851, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %900 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed297, <4 x half> %899, <4 x float> %894) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %901 = shufflevector <8 x half> %851, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %902 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed297, <4 x half> %901, <4 x float> %896) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %903 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed313, <4 x half> %899, <4 x float> %897) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %904 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed313, <4 x half> %901, <4 x float> %898) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %905 = bitcast <4 x i32> %852 to <8 x half>, !dbg !14
  %906 = shufflevector <8 x half> %905, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %907 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed299, <4 x half> %906, <4 x float> %900) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %908 = shufflevector <8 x half> %905, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %909 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed299, <4 x half> %908, <4 x float> %902) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %910 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed315, <4 x half> %906, <4 x float> %903) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %911 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed315, <4 x half> %908, <4 x float> %904) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %912 = add nuw nsw i64 %98, 1, !dbg !8
  %k0.has.next = icmp ult i64 %912, 32, !dbg !14
  br i1 %k0.has.next, label %k0.issue, label %k0.join, !dbg !14

k0.issue:                                         ; preds = %106
  %k0.next. = shl nuw nsw i64 %912, 14, !dbg !13
  %k0.next.1577 = or disjoint i64 %108, %k0.next., !dbg !13
  %k0.next.kdma.source1120 = or disjoint i64 %k0.next.1577, %kdma.source1119, !dbg !13
  %k0.next.kdma.source1121 = trunc nuw i64 %k0.next.kdma.source1120 to i32, !dbg !13
  %k0.next.kdma.source1122 = shl nuw nsw i32 %k0.next.kdma.source1121, 1, !dbg !13
  %k0.next.kdma.offset = or disjoint i32 %k0.next.kdma.source1122, %kdma.intra, !dbg !13
  %k0.next.1578 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %36, i32 %k0.next.kdma.offset, <4 x i32> %dma.buffer.rsrc989) #10, !dbg !13, !frisk.cross_bn_k0 !1
  %k0.next.kdma.source1134 = or disjoint i64 %k0.next.1577, %kdma.source1133, !dbg !13
  %k0.next.kdma.source1135 = trunc nuw i64 %k0.next.kdma.source1134 to i32, !dbg !13
  %k0.next.kdma.source1136 = shl nuw nsw i32 %k0.next.kdma.source1135, 1, !dbg !13
  %k0.next.kdma.offset1137 = or disjoint i32 %k0.next.kdma.source1136, %kdma.intra.hi, !dbg !13
  %k0.next.1579 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1138, i32 %k0.next.kdma.offset1137, <4 x i32> %dma.buffer.rsrc989) #10, !dbg !13, !frisk.cross_bn_k0 !1
  %k0.next.kdma.source1151 = or disjoint i64 %k0.next.1577, %kdma.source1150, !dbg !13
  %k0.next.kdma.source1152 = trunc nuw i64 %k0.next.kdma.source1151 to i32, !dbg !13
  %k0.next.kdma.source1153 = shl nuw nsw i32 %k0.next.kdma.source1152, 1, !dbg !13
  %k0.next.kdma.offset1154 = or disjoint i32 %k0.next.kdma.source1153, %kdma.intra, !dbg !13
  %k0.next.1580 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %43, i32 %k0.next.kdma.offset1154, <4 x i32> %dma.buffer.rsrc989) #10, !dbg !13, !frisk.cross_bn_k0 !1
  %k0.next.kdma.source1168 = or disjoint i64 %k0.next.1577, %kdma.source1167, !dbg !13
  %k0.next.kdma.source1169 = trunc nuw i64 %k0.next.kdma.source1168 to i32, !dbg !13
  %k0.next.kdma.source1170 = shl nuw nsw i32 %k0.next.kdma.source1169, 1, !dbg !13
  %k0.next.kdma.offset1171 = or disjoint i32 %k0.next.kdma.source1170, %kdma.intra.hi, !dbg !13
  %k0.next.1581 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1172, i32 %k0.next.kdma.offset1171, <4 x i32> %dma.buffer.rsrc989) #10, !dbg !13, !frisk.cross_bn_k0 !1
  br label %k0.join

k0.join:                                          ; preds = %106, %k0.issue
  %913 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %914 = bitcast <4 x i32> %913 to <8 x half>, !dbg !14
  %915 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1065, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %916 = bitcast <4 x i32> %915 to <8 x half>, !dbg !14
  %917 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1066, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %918 = bitcast <4 x i32> %917 to <8 x half>, !dbg !14
  %919 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1067, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %920 = bitcast <4 x i32> %919 to <8 x half>, !dbg !14
  %921 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1068, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %922 = bitcast <4 x i32> %921 to <8 x half>, !dbg !14
  %923 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1069, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %924 = bitcast <4 x i32> %923 to <8 x half>, !dbg !14
  %925 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1070, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %926 = bitcast <4 x i32> %925 to <8 x half>, !dbg !14
  %927 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1071, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %928 = shufflevector <8 x half> %914, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %929 = extractvalue [2 x <8 x float>] %104, 0, !dbg !28
  %930 = shufflevector <8 x float> %929, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %931 = fmul <4 x float> %930, %440, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %932 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %928, <4 x float> %931) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %933 = shufflevector <8 x half> %914, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %934 = shufflevector <8 x float> %929, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %935 = fmul <4 x float> %934, %440, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %936 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %933, <4 x float> %935) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %937 = extractvalue [2 x <8 x float>] %104, 1, !dbg !28
  %938 = shufflevector <8 x float> %937, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %939 = fmul <4 x float> %938, %448, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %940 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed301, <4 x half> %928, <4 x float> %939) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %941 = shufflevector <8 x float> %937, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %942 = fmul <4 x float> %941, %448, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %943 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed301, <4 x half> %933, <4 x float> %942) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %944 = shufflevector <8 x half> %916, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %945 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed287, <4 x half> %944, <4 x float> %932) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %946 = shufflevector <8 x half> %916, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %947 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed287, <4 x half> %946, <4 x float> %936) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %948 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed303, <4 x half> %944, <4 x float> %940) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %949 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed303, <4 x half> %946, <4 x float> %943) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %950 = shufflevector <8 x half> %918, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %951 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed289, <4 x half> %950, <4 x float> %945) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %952 = shufflevector <8 x half> %918, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %953 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed289, <4 x half> %952, <4 x float> %947) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %954 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed305, <4 x half> %950, <4 x float> %948) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %955 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed305, <4 x half> %952, <4 x float> %949) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %956 = shufflevector <8 x half> %920, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %957 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed291, <4 x half> %956, <4 x float> %951) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %958 = shufflevector <8 x half> %920, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %959 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed291, <4 x half> %958, <4 x float> %953) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %960 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed307, <4 x half> %956, <4 x float> %954) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %961 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed307, <4 x half> %958, <4 x float> %955) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %962 = shufflevector <8 x half> %922, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %963 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed293, <4 x half> %962, <4 x float> %957) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %964 = shufflevector <8 x half> %922, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %965 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed293, <4 x half> %964, <4 x float> %959) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %966 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed309, <4 x half> %962, <4 x float> %960) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %967 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed309, <4 x half> %964, <4 x float> %961) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %968 = shufflevector <8 x half> %924, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %969 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed295, <4 x half> %968, <4 x float> %963) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %970 = shufflevector <8 x half> %924, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %971 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed295, <4 x half> %970, <4 x float> %965) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %972 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed311, <4 x half> %968, <4 x float> %966) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %973 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed311, <4 x half> %970, <4 x float> %967) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %974 = shufflevector <8 x half> %926, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %975 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed297, <4 x half> %974, <4 x float> %969) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %976 = shufflevector <8 x half> %926, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %977 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed297, <4 x half> %976, <4 x float> %971) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %978 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed313, <4 x half> %974, <4 x float> %972) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %979 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed313, <4 x half> %976, <4 x float> %973) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %980 = bitcast <4 x i32> %927 to <8 x half>, !dbg !14
  %981 = shufflevector <8 x half> %980, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %982 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed299, <4 x half> %981, <4 x float> %975) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %983 = shufflevector <8 x half> %980, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %984 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed299, <4 x half> %983, <4 x float> %977) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %985 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed315, <4 x half> %981, <4 x float> %978) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %986 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed315, <4 x half> %983, <4 x float> %979) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !8
  %987 = shufflevector <4 x float> %737, <4 x float> %739, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %988 = insertvalue [2 x <8 x float>] poison, <8 x float> %987, 0, !dbg !8
  %989 = shufflevector <4 x float> %758, <4 x float> %759, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %990 = insertvalue [2 x <8 x float>] %988, <8 x float> %989, 1, !dbg !8
  %991 = insertvalue [2 x <1 x float>] poison, <1 x float> %430, 0, !dbg !8
  %992 = insertvalue [2 x <1 x float>] %991, <1 x float> %433, 1, !dbg !8
  %993 = shufflevector <4 x float> %831, <4 x float> %833, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %994 = insertvalue [2 x <8 x float>] poison, <8 x float> %993, 0, !dbg !8
  %995 = shufflevector <4 x float> %834, <4 x float> %835, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %996 = insertvalue [2 x <8 x float>] %994, <8 x float> %995, 1, !dbg !8
  %997 = shufflevector <4 x float> %907, <4 x float> %909, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %998 = insertvalue [2 x <8 x float>] poison, <8 x float> %997, 0, !dbg !8
  %999 = shufflevector <4 x float> %910, <4 x float> %911, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %1000 = insertvalue [2 x <8 x float>] %998, <8 x float> %999, 1, !dbg !8
  %1001 = shufflevector <4 x float> %982, <4 x float> %984, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %1002 = insertvalue [2 x <8 x float>] poison, <8 x float> %1001, 0, !dbg !8
  %1003 = shufflevector <4 x float> %985, <4 x float> %986, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %1004 = insertvalue [2 x <8 x float>] %1002, <8 x float> %1003, 1, !dbg !8
  br label %97, !dbg !8

1005:                                             ; preds = %97
  %lane.sum.field176 = extractvalue [2 x <1 x float>] %100, 1, !dbg !8
  %lane.sum.local177 = extractelement <1 x float> %lane.sum.field176, i64 0, !dbg !8
  %1006 = shl nuw nsw i32 %8, 2, !dbg !8
  %1007 = xor i32 %1006, 64, !dbg !8
  %1008 = bitcast float %lane.sum.local177 to i32, !dbg !8
  %lane.sum.shuffle180 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1007, i32 %1008), !dbg !8
  %1009 = bitcast i32 %lane.sum.shuffle180 to float, !dbg !8
  %lane.sum.final181 = fadd float %lane.sum.local177, %1009, !dbg !8
  %1010 = xor i32 %1006, 128, !dbg !8
  %1011 = bitcast float %lane.sum.final181 to i32, !dbg !8
  %lane.sum.shuffle182 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1010, i32 %1011), !dbg !8
  %1012 = bitcast i32 %lane.sum.shuffle182 to float, !dbg !8
  %lane.sum.final183 = fadd float %lane.sum.final181, %1012, !dbg !8
  %1013 = insertelement <1 x float> poison, float %lane.sum.final183, i64 0, !dbg !8
  %lane.sum.field = extractvalue [2 x <1 x float>] %100, 0, !dbg !8
  %lane.sum.local = extractelement <1 x float> %lane.sum.field, i64 0, !dbg !8
  %1014 = bitcast float %lane.sum.local to i32, !dbg !8
  %lane.sum.shuffle = call i32 @llvm.amdgcn.ds.bpermute(i32 %1007, i32 %1014), !dbg !8
  %1015 = bitcast i32 %lane.sum.shuffle to float, !dbg !8
  %lane.sum.final = fadd float %lane.sum.local, %1015, !dbg !8
  %1016 = bitcast float %lane.sum.final to i32, !dbg !8
  %lane.sum.shuffle174 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1010, i32 %1016), !dbg !8
  %1017 = bitcast i32 %lane.sum.shuffle174 to float, !dbg !8
  %lane.sum.final175 = fadd float %lane.sum.final, %1017, !dbg !8
  %1018 = insertelement <1 x float> poison, float %lane.sum.final175, i64 0, !dbg !8
  %1019 = and i32 %kdma.tid, 16, !dbg !8
  %.not = icmp eq i32 %1019, 0, !dbg !8
  %1020 = and i32 %kdma.tid, 32, !dbg !8
  %.not28 = icmp eq i32 %1020, 0, !dbg !8
  %1021 = extractvalue [2 x <8 x float>] %104, 0, !dbg !8
  %1022 = extractelement <8 x float> %1021, i64 0, !dbg !8
  %1023 = extractelement <8 x float> %1021, i64 1, !dbg !8
  %1024 = extractelement <8 x float> %1021, i64 2, !dbg !8
  %1025 = extractelement <8 x float> %1021, i64 3, !dbg !8
  %1026 = select i1 %.not, float %1023, float %1022, !dbg !8
  %1027 = bitcast float %1026 to i32, !dbg !8
  %1028 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1027), !dbg !8
  %1029 = bitcast i32 %1028 to float, !dbg !8
  %1030 = select i1 %.not, float %1022, float %1029, !dbg !8
  %1031 = select i1 %.not, float %1029, float %1023, !dbg !8
  %1032 = select i1 %.not, float %1025, float %1024, !dbg !8
  %1033 = bitcast float %1032 to i32, !dbg !8
  %1034 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1033), !dbg !8
  %1035 = bitcast i32 %1034 to float, !dbg !8
  %1036 = select i1 %.not, float %1024, float %1035, !dbg !8
  %1037 = select i1 %.not, float %1035, float %1025, !dbg !8
  %1038 = select i1 %.not28, float %1036, float %1030, !dbg !8
  %1039 = bitcast float %1038 to i32, !dbg !8
  %1040 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1039), !dbg !8
  %1041 = bitcast i32 %1040 to float, !dbg !8
  %1042 = select i1 %.not28, float %1030, float %1041, !dbg !8
  %1043 = select i1 %.not28, float %1041, float %1036, !dbg !8
  %1044 = select i1 %.not28, float %1037, float %1031, !dbg !8
  %1045 = bitcast float %1044 to i32, !dbg !8
  %1046 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1045), !dbg !8
  %1047 = bitcast i32 %1046 to float, !dbg !8
  %1048 = select i1 %.not28, float %1031, float %1047, !dbg !8
  %1049 = select i1 %.not28, float %1047, float %1037, !dbg !8
  %1050 = extractelement <8 x float> %1021, i64 4, !dbg !8
  %1051 = extractelement <8 x float> %1021, i64 5, !dbg !8
  %1052 = extractelement <8 x float> %1021, i64 6, !dbg !8
  %1053 = extractelement <8 x float> %1021, i64 7, !dbg !8
  %1054 = select i1 %.not, float %1051, float %1050, !dbg !8
  %1055 = bitcast float %1054 to i32, !dbg !8
  %1056 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1055), !dbg !8
  %1057 = bitcast i32 %1056 to float, !dbg !8
  %1058 = select i1 %.not, float %1050, float %1057, !dbg !8
  %1059 = select i1 %.not, float %1057, float %1051, !dbg !8
  %1060 = select i1 %.not, float %1053, float %1052, !dbg !8
  %1061 = bitcast float %1060 to i32, !dbg !8
  %1062 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1061), !dbg !8
  %1063 = bitcast i32 %1062 to float, !dbg !8
  %1064 = select i1 %.not, float %1052, float %1063, !dbg !8
  %1065 = select i1 %.not, float %1063, float %1053, !dbg !8
  %1066 = select i1 %.not28, float %1064, float %1058, !dbg !8
  %1067 = bitcast float %1066 to i32, !dbg !8
  %1068 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1067), !dbg !8
  %1069 = bitcast i32 %1068 to float, !dbg !8
  %1070 = select i1 %.not28, float %1058, float %1069, !dbg !8
  %1071 = select i1 %.not28, float %1069, float %1064, !dbg !8
  %1072 = select i1 %.not28, float %1065, float %1059, !dbg !8
  %1073 = bitcast float %1072 to i32, !dbg !8
  %1074 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1073), !dbg !8
  %1075 = bitcast i32 %1074 to float, !dbg !8
  %1076 = select i1 %.not28, float %1059, float %1075, !dbg !8
  %1077 = select i1 %.not28, float %1075, float %1065, !dbg !8
  %1078 = extractvalue [2 x <8 x float>] %104, 1, !dbg !8
  %1079 = extractelement <8 x float> %1078, i64 0, !dbg !8
  %1080 = extractelement <8 x float> %1078, i64 1, !dbg !8
  %1081 = extractelement <8 x float> %1078, i64 2, !dbg !8
  %1082 = extractelement <8 x float> %1078, i64 3, !dbg !8
  %1083 = select i1 %.not, float %1080, float %1079, !dbg !8
  %1084 = bitcast float %1083 to i32, !dbg !8
  %1085 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1084), !dbg !8
  %1086 = bitcast i32 %1085 to float, !dbg !8
  %1087 = select i1 %.not, float %1079, float %1086, !dbg !8
  %1088 = select i1 %.not, float %1086, float %1080, !dbg !8
  %1089 = select i1 %.not, float %1082, float %1081, !dbg !8
  %1090 = bitcast float %1089 to i32, !dbg !8
  %1091 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1090), !dbg !8
  %1092 = bitcast i32 %1091 to float, !dbg !8
  %1093 = select i1 %.not, float %1081, float %1092, !dbg !8
  %1094 = select i1 %.not, float %1092, float %1082, !dbg !8
  %1095 = select i1 %.not28, float %1093, float %1087, !dbg !8
  %1096 = bitcast float %1095 to i32, !dbg !8
  %1097 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1096), !dbg !8
  %1098 = bitcast i32 %1097 to float, !dbg !8
  %1099 = select i1 %.not28, float %1087, float %1098, !dbg !8
  %1100 = select i1 %.not28, float %1098, float %1093, !dbg !8
  %1101 = select i1 %.not28, float %1094, float %1088, !dbg !8
  %1102 = bitcast float %1101 to i32, !dbg !8
  %1103 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1102), !dbg !8
  %1104 = bitcast i32 %1103 to float, !dbg !8
  %1105 = select i1 %.not28, float %1088, float %1104, !dbg !8
  %1106 = select i1 %.not28, float %1104, float %1094, !dbg !8
  %1107 = extractelement <8 x float> %1078, i64 4, !dbg !8
  %1108 = extractelement <8 x float> %1078, i64 5, !dbg !8
  %1109 = extractelement <8 x float> %1078, i64 6, !dbg !8
  %1110 = extractelement <8 x float> %1078, i64 7, !dbg !8
  %1111 = select i1 %.not, float %1108, float %1107, !dbg !8
  %1112 = bitcast float %1111 to i32, !dbg !8
  %1113 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1112), !dbg !8
  %1114 = bitcast i32 %1113 to float, !dbg !8
  %1115 = select i1 %.not, float %1107, float %1114, !dbg !8
  %1116 = select i1 %.not, float %1114, float %1108, !dbg !8
  %1117 = select i1 %.not, float %1110, float %1109, !dbg !8
  %1118 = bitcast float %1117 to i32, !dbg !8
  %1119 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1118), !dbg !8
  %1120 = bitcast i32 %1119 to float, !dbg !8
  %1121 = select i1 %.not, float %1109, float %1120, !dbg !8
  %1122 = select i1 %.not, float %1120, float %1110, !dbg !8
  %1123 = select i1 %.not28, float %1121, float %1115, !dbg !8
  %1124 = bitcast float %1123 to i32, !dbg !8
  %1125 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1124), !dbg !8
  %1126 = bitcast i32 %1125 to float, !dbg !8
  %1127 = select i1 %.not28, float %1115, float %1126, !dbg !8
  %1128 = select i1 %.not28, float %1126, float %1121, !dbg !8
  %1129 = select i1 %.not28, float %1122, float %1116, !dbg !8
  %1130 = bitcast float %1129 to i32, !dbg !8
  %1131 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1130), !dbg !8
  %1132 = bitcast i32 %1131 to float, !dbg !8
  %1133 = select i1 %.not28, float %1116, float %1132, !dbg !8
  %1134 = select i1 %.not28, float %1132, float %1122, !dbg !8
  %1135 = extractvalue [2 x <8 x float>] %103, 0, !dbg !8
  %1136 = extractelement <8 x float> %1135, i64 0, !dbg !8
  %1137 = extractelement <8 x float> %1135, i64 1, !dbg !8
  %1138 = extractelement <8 x float> %1135, i64 2, !dbg !8
  %1139 = extractelement <8 x float> %1135, i64 3, !dbg !8
  %1140 = select i1 %.not, float %1137, float %1136, !dbg !8
  %1141 = bitcast float %1140 to i32, !dbg !8
  %1142 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1141), !dbg !8
  %1143 = bitcast i32 %1142 to float, !dbg !8
  %1144 = select i1 %.not, float %1136, float %1143, !dbg !8
  %1145 = select i1 %.not, float %1143, float %1137, !dbg !8
  %1146 = select i1 %.not, float %1139, float %1138, !dbg !8
  %1147 = bitcast float %1146 to i32, !dbg !8
  %1148 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1147), !dbg !8
  %1149 = bitcast i32 %1148 to float, !dbg !8
  %1150 = select i1 %.not, float %1138, float %1149, !dbg !8
  %1151 = select i1 %.not, float %1149, float %1139, !dbg !8
  %1152 = select i1 %.not28, float %1150, float %1144, !dbg !8
  %1153 = bitcast float %1152 to i32, !dbg !8
  %1154 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1153), !dbg !8
  %1155 = bitcast i32 %1154 to float, !dbg !8
  %1156 = select i1 %.not28, float %1144, float %1155, !dbg !8
  %1157 = select i1 %.not28, float %1155, float %1150, !dbg !8
  %1158 = select i1 %.not28, float %1151, float %1145, !dbg !8
  %1159 = bitcast float %1158 to i32, !dbg !8
  %1160 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1159), !dbg !8
  %1161 = bitcast i32 %1160 to float, !dbg !8
  %1162 = select i1 %.not28, float %1145, float %1161, !dbg !8
  %1163 = select i1 %.not28, float %1161, float %1151, !dbg !8
  %1164 = extractelement <8 x float> %1135, i64 4, !dbg !8
  %1165 = extractelement <8 x float> %1135, i64 5, !dbg !8
  %1166 = extractelement <8 x float> %1135, i64 6, !dbg !8
  %1167 = extractelement <8 x float> %1135, i64 7, !dbg !8
  %1168 = select i1 %.not, float %1165, float %1164, !dbg !8
  %1169 = bitcast float %1168 to i32, !dbg !8
  %1170 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1169), !dbg !8
  %1171 = bitcast i32 %1170 to float, !dbg !8
  %1172 = select i1 %.not, float %1164, float %1171, !dbg !8
  %1173 = select i1 %.not, float %1171, float %1165, !dbg !8
  %1174 = select i1 %.not, float %1167, float %1166, !dbg !8
  %1175 = bitcast float %1174 to i32, !dbg !8
  %1176 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1175), !dbg !8
  %1177 = bitcast i32 %1176 to float, !dbg !8
  %1178 = select i1 %.not, float %1166, float %1177, !dbg !8
  %1179 = select i1 %.not, float %1177, float %1167, !dbg !8
  %1180 = select i1 %.not28, float %1178, float %1172, !dbg !8
  %1181 = bitcast float %1180 to i32, !dbg !8
  %1182 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1181), !dbg !8
  %1183 = bitcast i32 %1182 to float, !dbg !8
  %1184 = select i1 %.not28, float %1172, float %1183, !dbg !8
  %1185 = select i1 %.not28, float %1183, float %1178, !dbg !8
  %1186 = select i1 %.not28, float %1179, float %1173, !dbg !8
  %1187 = bitcast float %1186 to i32, !dbg !8
  %1188 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1187), !dbg !8
  %1189 = bitcast i32 %1188 to float, !dbg !8
  %1190 = select i1 %.not28, float %1173, float %1189, !dbg !8
  %1191 = select i1 %.not28, float %1189, float %1179, !dbg !8
  %1192 = extractvalue [2 x <8 x float>] %103, 1, !dbg !8
  %1193 = extractelement <8 x float> %1192, i64 0, !dbg !8
  %1194 = extractelement <8 x float> %1192, i64 1, !dbg !8
  %1195 = extractelement <8 x float> %1192, i64 2, !dbg !8
  %1196 = extractelement <8 x float> %1192, i64 3, !dbg !8
  %1197 = select i1 %.not, float %1194, float %1193, !dbg !8
  %1198 = bitcast float %1197 to i32, !dbg !8
  %1199 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1198), !dbg !8
  %1200 = bitcast i32 %1199 to float, !dbg !8
  %1201 = select i1 %.not, float %1193, float %1200, !dbg !8
  %1202 = select i1 %.not, float %1200, float %1194, !dbg !8
  %1203 = select i1 %.not, float %1196, float %1195, !dbg !8
  %1204 = bitcast float %1203 to i32, !dbg !8
  %1205 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1204), !dbg !8
  %1206 = bitcast i32 %1205 to float, !dbg !8
  %1207 = select i1 %.not, float %1195, float %1206, !dbg !8
  %1208 = select i1 %.not, float %1206, float %1196, !dbg !8
  %1209 = select i1 %.not28, float %1207, float %1201, !dbg !8
  %1210 = bitcast float %1209 to i32, !dbg !8
  %1211 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1210), !dbg !8
  %1212 = bitcast i32 %1211 to float, !dbg !8
  %1213 = select i1 %.not28, float %1201, float %1212, !dbg !8
  %1214 = select i1 %.not28, float %1212, float %1207, !dbg !8
  %1215 = select i1 %.not28, float %1208, float %1202, !dbg !8
  %1216 = bitcast float %1215 to i32, !dbg !8
  %1217 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1216), !dbg !8
  %1218 = bitcast i32 %1217 to float, !dbg !8
  %1219 = select i1 %.not28, float %1202, float %1218, !dbg !8
  %1220 = select i1 %.not28, float %1218, float %1208, !dbg !8
  %1221 = extractelement <8 x float> %1192, i64 4, !dbg !8
  %1222 = extractelement <8 x float> %1192, i64 5, !dbg !8
  %1223 = extractelement <8 x float> %1192, i64 6, !dbg !8
  %1224 = extractelement <8 x float> %1192, i64 7, !dbg !8
  %1225 = select i1 %.not, float %1222, float %1221, !dbg !8
  %1226 = bitcast float %1225 to i32, !dbg !8
  %1227 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1226), !dbg !8
  %1228 = bitcast i32 %1227 to float, !dbg !8
  %1229 = select i1 %.not, float %1221, float %1228, !dbg !8
  %1230 = select i1 %.not, float %1228, float %1222, !dbg !8
  %1231 = select i1 %.not, float %1224, float %1223, !dbg !8
  %1232 = bitcast float %1231 to i32, !dbg !8
  %1233 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1232), !dbg !8
  %1234 = bitcast i32 %1233 to float, !dbg !8
  %1235 = select i1 %.not, float %1223, float %1234, !dbg !8
  %1236 = select i1 %.not, float %1234, float %1224, !dbg !8
  %1237 = select i1 %.not28, float %1235, float %1229, !dbg !8
  %1238 = bitcast float %1237 to i32, !dbg !8
  %1239 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1238), !dbg !8
  %1240 = bitcast i32 %1239 to float, !dbg !8
  %1241 = select i1 %.not28, float %1229, float %1240, !dbg !8
  %1242 = select i1 %.not28, float %1240, float %1235, !dbg !8
  %1243 = select i1 %.not28, float %1236, float %1230, !dbg !8
  %1244 = bitcast float %1243 to i32, !dbg !8
  %1245 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1244), !dbg !8
  %1246 = bitcast i32 %1245 to float, !dbg !8
  %1247 = select i1 %.not28, float %1230, float %1246, !dbg !8
  %1248 = select i1 %.not28, float %1246, float %1236, !dbg !8
  %1249 = extractvalue [2 x <8 x float>] %102, 0, !dbg !8
  %1250 = extractelement <8 x float> %1249, i64 0, !dbg !8
  %1251 = extractelement <8 x float> %1249, i64 1, !dbg !8
  %1252 = extractelement <8 x float> %1249, i64 2, !dbg !8
  %1253 = extractelement <8 x float> %1249, i64 3, !dbg !8
  %1254 = select i1 %.not, float %1251, float %1250, !dbg !8
  %1255 = bitcast float %1254 to i32, !dbg !8
  %1256 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1255), !dbg !8
  %1257 = bitcast i32 %1256 to float, !dbg !8
  %1258 = select i1 %.not, float %1250, float %1257, !dbg !8
  %1259 = select i1 %.not, float %1257, float %1251, !dbg !8
  %1260 = select i1 %.not, float %1253, float %1252, !dbg !8
  %1261 = bitcast float %1260 to i32, !dbg !8
  %1262 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1261), !dbg !8
  %1263 = bitcast i32 %1262 to float, !dbg !8
  %1264 = select i1 %.not, float %1252, float %1263, !dbg !8
  %1265 = select i1 %.not, float %1263, float %1253, !dbg !8
  %1266 = select i1 %.not28, float %1264, float %1258, !dbg !8
  %1267 = bitcast float %1266 to i32, !dbg !8
  %1268 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1267), !dbg !8
  %1269 = bitcast i32 %1268 to float, !dbg !8
  %1270 = select i1 %.not28, float %1258, float %1269, !dbg !8
  %1271 = select i1 %.not28, float %1269, float %1264, !dbg !8
  %1272 = select i1 %.not28, float %1265, float %1259, !dbg !8
  %1273 = bitcast float %1272 to i32, !dbg !8
  %1274 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1273), !dbg !8
  %1275 = bitcast i32 %1274 to float, !dbg !8
  %1276 = select i1 %.not28, float %1259, float %1275, !dbg !8
  %1277 = select i1 %.not28, float %1275, float %1265, !dbg !8
  %1278 = extractelement <8 x float> %1249, i64 4, !dbg !8
  %1279 = extractelement <8 x float> %1249, i64 5, !dbg !8
  %1280 = extractelement <8 x float> %1249, i64 6, !dbg !8
  %1281 = extractelement <8 x float> %1249, i64 7, !dbg !8
  %1282 = select i1 %.not, float %1279, float %1278, !dbg !8
  %1283 = bitcast float %1282 to i32, !dbg !8
  %1284 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1283), !dbg !8
  %1285 = bitcast i32 %1284 to float, !dbg !8
  %1286 = select i1 %.not, float %1278, float %1285, !dbg !8
  %1287 = select i1 %.not, float %1285, float %1279, !dbg !8
  %1288 = select i1 %.not, float %1281, float %1280, !dbg !8
  %1289 = bitcast float %1288 to i32, !dbg !8
  %1290 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1289), !dbg !8
  %1291 = bitcast i32 %1290 to float, !dbg !8
  %1292 = select i1 %.not, float %1280, float %1291, !dbg !8
  %1293 = select i1 %.not, float %1291, float %1281, !dbg !8
  %1294 = select i1 %.not28, float %1292, float %1286, !dbg !8
  %1295 = bitcast float %1294 to i32, !dbg !8
  %1296 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1295), !dbg !8
  %1297 = bitcast i32 %1296 to float, !dbg !8
  %1298 = select i1 %.not28, float %1286, float %1297, !dbg !8
  %1299 = select i1 %.not28, float %1297, float %1292, !dbg !8
  %1300 = select i1 %.not28, float %1293, float %1287, !dbg !8
  %1301 = bitcast float %1300 to i32, !dbg !8
  %1302 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1301), !dbg !8
  %1303 = bitcast i32 %1302 to float, !dbg !8
  %1304 = select i1 %.not28, float %1287, float %1303, !dbg !8
  %1305 = select i1 %.not28, float %1303, float %1293, !dbg !8
  %1306 = extractvalue [2 x <8 x float>] %102, 1, !dbg !8
  %1307 = extractelement <8 x float> %1306, i64 0, !dbg !8
  %1308 = extractelement <8 x float> %1306, i64 1, !dbg !8
  %1309 = extractelement <8 x float> %1306, i64 2, !dbg !8
  %1310 = extractelement <8 x float> %1306, i64 3, !dbg !8
  %1311 = select i1 %.not, float %1308, float %1307, !dbg !8
  %1312 = bitcast float %1311 to i32, !dbg !8
  %1313 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1312), !dbg !8
  %1314 = bitcast i32 %1313 to float, !dbg !8
  %1315 = select i1 %.not, float %1307, float %1314, !dbg !8
  %1316 = select i1 %.not, float %1314, float %1308, !dbg !8
  %1317 = select i1 %.not, float %1310, float %1309, !dbg !8
  %1318 = bitcast float %1317 to i32, !dbg !8
  %1319 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1318), !dbg !8
  %1320 = bitcast i32 %1319 to float, !dbg !8
  %1321 = select i1 %.not, float %1309, float %1320, !dbg !8
  %1322 = select i1 %.not, float %1320, float %1310, !dbg !8
  %1323 = select i1 %.not28, float %1321, float %1315, !dbg !8
  %1324 = bitcast float %1323 to i32, !dbg !8
  %1325 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1324), !dbg !8
  %1326 = bitcast i32 %1325 to float, !dbg !8
  %1327 = select i1 %.not28, float %1315, float %1326, !dbg !8
  %1328 = select i1 %.not28, float %1326, float %1321, !dbg !8
  %1329 = select i1 %.not28, float %1322, float %1316, !dbg !8
  %1330 = bitcast float %1329 to i32, !dbg !8
  %1331 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1330), !dbg !8
  %1332 = bitcast i32 %1331 to float, !dbg !8
  %1333 = select i1 %.not28, float %1316, float %1332, !dbg !8
  %1334 = select i1 %.not28, float %1332, float %1322, !dbg !8
  %1335 = extractelement <8 x float> %1306, i64 4, !dbg !8
  %1336 = extractelement <8 x float> %1306, i64 5, !dbg !8
  %1337 = extractelement <8 x float> %1306, i64 6, !dbg !8
  %1338 = extractelement <8 x float> %1306, i64 7, !dbg !8
  %1339 = select i1 %.not, float %1336, float %1335, !dbg !8
  %1340 = bitcast float %1339 to i32, !dbg !8
  %1341 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1340), !dbg !8
  %1342 = bitcast i32 %1341 to float, !dbg !8
  %1343 = select i1 %.not, float %1335, float %1342, !dbg !8
  %1344 = select i1 %.not, float %1342, float %1336, !dbg !8
  %1345 = select i1 %.not, float %1338, float %1337, !dbg !8
  %1346 = bitcast float %1345 to i32, !dbg !8
  %1347 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1346), !dbg !8
  %1348 = bitcast i32 %1347 to float, !dbg !8
  %1349 = select i1 %.not, float %1337, float %1348, !dbg !8
  %1350 = select i1 %.not, float %1348, float %1338, !dbg !8
  %1351 = select i1 %.not28, float %1349, float %1343, !dbg !8
  %1352 = bitcast float %1351 to i32, !dbg !8
  %1353 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1352), !dbg !8
  %1354 = bitcast i32 %1353 to float, !dbg !8
  %1355 = select i1 %.not28, float %1343, float %1354, !dbg !8
  %1356 = select i1 %.not28, float %1354, float %1349, !dbg !8
  %1357 = select i1 %.not28, float %1350, float %1344, !dbg !8
  %1358 = bitcast float %1357 to i32, !dbg !8
  %1359 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1358), !dbg !8
  %1360 = bitcast i32 %1359 to float, !dbg !8
  %1361 = select i1 %.not28, float %1344, float %1360, !dbg !8
  %1362 = select i1 %.not28, float %1360, float %1350, !dbg !8
  %1363 = extractvalue [2 x <8 x float>] %99, 0, !dbg !8
  %1364 = extractelement <8 x float> %1363, i64 0, !dbg !8
  %1365 = extractelement <8 x float> %1363, i64 1, !dbg !8
  %1366 = extractelement <8 x float> %1363, i64 2, !dbg !8
  %1367 = extractelement <8 x float> %1363, i64 3, !dbg !8
  %1368 = select i1 %.not, float %1365, float %1364, !dbg !8
  %1369 = bitcast float %1368 to i32, !dbg !8
  %1370 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1369), !dbg !8
  %1371 = bitcast i32 %1370 to float, !dbg !8
  %1372 = select i1 %.not, float %1364, float %1371, !dbg !8
  %1373 = select i1 %.not, float %1371, float %1365, !dbg !8
  %1374 = select i1 %.not, float %1367, float %1366, !dbg !8
  %1375 = bitcast float %1374 to i32, !dbg !8
  %1376 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1375), !dbg !8
  %1377 = bitcast i32 %1376 to float, !dbg !8
  %1378 = select i1 %.not, float %1366, float %1377, !dbg !8
  %1379 = select i1 %.not, float %1377, float %1367, !dbg !8
  %1380 = select i1 %.not28, float %1378, float %1372, !dbg !8
  %1381 = bitcast float %1380 to i32, !dbg !8
  %1382 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1381), !dbg !8
  %1383 = select i1 %.not28, float %1379, float %1373, !dbg !8
  %1384 = bitcast float %1383 to i32, !dbg !8
  %1385 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1384), !dbg !8
  %1386 = extractelement <8 x float> %1363, i64 4, !dbg !8
  %1387 = extractelement <8 x float> %1363, i64 5, !dbg !8
  %1388 = extractelement <8 x float> %1363, i64 6, !dbg !8
  %1389 = extractelement <8 x float> %1363, i64 7, !dbg !8
  %1390 = select i1 %.not, float %1387, float %1386, !dbg !8
  %1391 = bitcast float %1390 to i32, !dbg !8
  %1392 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1391), !dbg !8
  %1393 = bitcast i32 %1392 to float, !dbg !8
  %1394 = select i1 %.not, float %1386, float %1393, !dbg !8
  %1395 = select i1 %.not, float %1393, float %1387, !dbg !8
  %1396 = select i1 %.not, float %1389, float %1388, !dbg !8
  %1397 = bitcast float %1396 to i32, !dbg !8
  %1398 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1397), !dbg !8
  %1399 = bitcast i32 %1398 to float, !dbg !8
  %1400 = select i1 %.not, float %1388, float %1399, !dbg !8
  %1401 = select i1 %.not, float %1399, float %1389, !dbg !8
  %1402 = select i1 %.not28, float %1400, float %1394, !dbg !8
  %1403 = bitcast float %1402 to i32, !dbg !8
  %1404 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1403), !dbg !8
  %1405 = select i1 %.not28, float %1401, float %1395, !dbg !8
  %1406 = bitcast float %1405 to i32, !dbg !8
  %1407 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1406), !dbg !8
  %1408 = extractvalue [2 x <8 x float>] %99, 1, !dbg !8
  %1409 = extractelement <8 x float> %1408, i64 0, !dbg !8
  %1410 = extractelement <8 x float> %1408, i64 1, !dbg !8
  %1411 = extractelement <8 x float> %1408, i64 2, !dbg !8
  %1412 = extractelement <8 x float> %1408, i64 3, !dbg !8
  %1413 = select i1 %.not, float %1410, float %1409, !dbg !8
  %1414 = bitcast float %1413 to i32, !dbg !8
  %1415 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1414), !dbg !8
  %1416 = bitcast i32 %1415 to float, !dbg !8
  %1417 = select i1 %.not, float %1409, float %1416, !dbg !8
  %1418 = select i1 %.not, float %1416, float %1410, !dbg !8
  %1419 = select i1 %.not, float %1412, float %1411, !dbg !8
  %1420 = bitcast float %1419 to i32, !dbg !8
  %1421 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1420), !dbg !8
  %1422 = bitcast i32 %1421 to float, !dbg !8
  %1423 = select i1 %.not, float %1411, float %1422, !dbg !8
  %1424 = select i1 %.not, float %1422, float %1412, !dbg !8
  %1425 = select i1 %.not28, float %1423, float %1417, !dbg !8
  %1426 = bitcast float %1425 to i32, !dbg !8
  %1427 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1426), !dbg !8
  %1428 = select i1 %.not28, float %1424, float %1418, !dbg !8
  %1429 = bitcast float %1428 to i32, !dbg !8
  %1430 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1429), !dbg !8
  %1431 = extractelement <8 x float> %1408, i64 4, !dbg !8
  %1432 = extractelement <8 x float> %1408, i64 5, !dbg !8
  %1433 = extractelement <8 x float> %1408, i64 6, !dbg !8
  %1434 = extractelement <8 x float> %1408, i64 7, !dbg !8
  %1435 = select i1 %.not, float %1432, float %1431, !dbg !8
  %1436 = bitcast float %1435 to i32, !dbg !8
  %1437 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1436), !dbg !8
  %1438 = bitcast i32 %1437 to float, !dbg !8
  %1439 = select i1 %.not, float %1431, float %1438, !dbg !8
  %1440 = select i1 %.not, float %1438, float %1432, !dbg !8
  %1441 = select i1 %.not, float %1434, float %1433, !dbg !8
  %1442 = bitcast float %1441 to i32, !dbg !8
  %1443 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1442), !dbg !8
  %1444 = bitcast i32 %1443 to float, !dbg !8
  %1445 = select i1 %.not, float %1433, float %1444, !dbg !8
  %1446 = select i1 %.not, float %1444, float %1434, !dbg !8
  %1447 = select i1 %.not28, float %1445, float %1439, !dbg !8
  %1448 = bitcast float %1447 to i32, !dbg !8
  %1449 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1448), !dbg !8
  %1450 = select i1 %.not28, float %1446, float %1440, !dbg !8
  %1451 = bitcast float %1450 to i32, !dbg !8
  %1452 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1451), !dbg !8
  %1453 = shufflevector <1 x float> %1018, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !17
  %1454 = shufflevector <1 x float> %1013, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !17
  %1455 = bitcast i32 %1449 to float, !dbg !8
  %1456 = select i1 %.not28, float %1439, float %1455, !dbg !8
  %1457 = insertelement <4 x float> poison, float %1456, i64 0, !dbg !17
  %1458 = bitcast i32 %1452 to float, !dbg !8
  %1459 = select i1 %.not28, float %1440, float %1458, !dbg !8
  %1460 = insertelement <4 x float> %1457, float %1459, i64 1, !dbg !17
  %1461 = select i1 %.not28, float %1455, float %1445, !dbg !8
  %1462 = insertelement <4 x float> %1460, float %1461, i64 2, !dbg !17
  %1463 = select i1 %.not28, float %1458, float %1446, !dbg !8
  %1464 = insertelement <4 x float> %1462, float %1463, i64 3, !dbg !17
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1454, !dbg !17
  %normalized = fmul <4 x float> %1464, %row.reciprocal, !dbg !17
  %1465 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !18
  %1466 = extractelement <4 x half> %1465, i64 3, !dbg !18
  %1467 = extractelement <4 x half> %1465, i64 2, !dbg !18
  %1468 = extractelement <4 x half> %1465, i64 1, !dbg !18
  %1469 = extractelement <4 x half> %1465, i64 0, !dbg !18
  %1470 = bitcast i32 %1427 to float, !dbg !8
  %1471 = select i1 %.not28, float %1417, float %1470, !dbg !8
  %1472 = insertelement <4 x float> poison, float %1471, i64 0, !dbg !17
  %1473 = bitcast i32 %1430 to float, !dbg !8
  %1474 = select i1 %.not28, float %1418, float %1473, !dbg !8
  %1475 = insertelement <4 x float> %1472, float %1474, i64 1, !dbg !17
  %1476 = select i1 %.not28, float %1470, float %1423, !dbg !8
  %1477 = insertelement <4 x float> %1475, float %1476, i64 2, !dbg !17
  %1478 = select i1 %.not28, float %1473, float %1424, !dbg !8
  %1479 = insertelement <4 x float> %1477, float %1478, i64 3, !dbg !17
  %normalized29 = fmul <4 x float> %1479, %row.reciprocal, !dbg !17
  %1480 = fptrunc <4 x float> %normalized29 to <4 x half>, !dbg !18
  %1481 = extractelement <4 x half> %1480, i64 3, !dbg !18
  %1482 = extractelement <4 x half> %1480, i64 2, !dbg !18
  %1483 = extractelement <4 x half> %1480, i64 1, !dbg !18
  %1484 = extractelement <4 x half> %1480, i64 0, !dbg !18
  %1485 = bitcast i32 %1404 to float, !dbg !8
  %1486 = select i1 %.not28, float %1394, float %1485, !dbg !8
  %1487 = insertelement <4 x float> poison, float %1486, i64 0, !dbg !17
  %1488 = bitcast i32 %1407 to float, !dbg !8
  %1489 = select i1 %.not28, float %1395, float %1488, !dbg !8
  %1490 = insertelement <4 x float> %1487, float %1489, i64 1, !dbg !17
  %1491 = select i1 %.not28, float %1485, float %1400, !dbg !8
  %1492 = insertelement <4 x float> %1490, float %1491, i64 2, !dbg !17
  %1493 = select i1 %.not28, float %1488, float %1401, !dbg !8
  %1494 = insertelement <4 x float> %1492, float %1493, i64 3, !dbg !17
  %row.reciprocal30 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1453, !dbg !17
  %normalized31 = fmul <4 x float> %1494, %row.reciprocal30, !dbg !17
  %1495 = fptrunc <4 x float> %normalized31 to <4 x half>, !dbg !18
  %1496 = extractelement <4 x half> %1495, i64 3, !dbg !18
  %1497 = extractelement <4 x half> %1495, i64 2, !dbg !18
  %1498 = extractelement <4 x half> %1495, i64 1, !dbg !18
  %1499 = extractelement <4 x half> %1495, i64 0, !dbg !18
  %1500 = bitcast i32 %1382 to float, !dbg !8
  %1501 = select i1 %.not28, float %1372, float %1500, !dbg !8
  %1502 = insertelement <4 x float> poison, float %1501, i64 0, !dbg !17
  %1503 = bitcast i32 %1385 to float, !dbg !8
  %1504 = select i1 %.not28, float %1373, float %1503, !dbg !8
  %1505 = insertelement <4 x float> %1502, float %1504, i64 1, !dbg !17
  %1506 = select i1 %.not28, float %1500, float %1378, !dbg !8
  %1507 = insertelement <4 x float> %1505, float %1506, i64 2, !dbg !17
  %1508 = select i1 %.not28, float %1503, float %1379, !dbg !8
  %1509 = insertelement <4 x float> %1507, float %1508, i64 3, !dbg !17
  %normalized32 = fmul <4 x float> %1509, %row.reciprocal30, !dbg !17
  %1510 = fptrunc <4 x float> %normalized32 to <4 x half>, !dbg !18
  %1511 = extractelement <4 x half> %1510, i64 3, !dbg !18
  %1512 = extractelement <4 x half> %1510, i64 2, !dbg !18
  %1513 = extractelement <4 x half> %1510, i64 1, !dbg !18
  %1514 = extractelement <4 x half> %1510, i64 0, !dbg !18
  %address.thread604 = call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #9, !dbg !19, !range !32, !frisk.register_lifetimes !1
  %address.local605 = zext nneg i32 %address.thread604 to i64, !dbg !9
  %address.local606 = add nuw i64 %32, %address.local605, !dbg !19
  %1515 = lshr i64 %address.local605, 2, !dbg !19
  %address.local608 = and i64 %1515, 48, !dbg !19
  %address.local609 = add i64 %address.local606, %address.local608, !dbg !19
  %address.local610 = and i64 %address.local605, 240, !dbg !19
  %address.local611 = sub i64 %address.local609, %address.local610, !dbg !19
  %address.local612 = add i64 %address.local611, %33, !dbg !19
  %address.local613 = shl nuw nsw i64 %address.local612, 7, !dbg !19
  %1516 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %46, !dbg !19
  %address.local615 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %1516, i64 %address.local613, !dbg !19
  %address.local617 = and i64 %1515, 12, !dbg !12
  %address.local618 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local615, i64 %address.local617, !dbg !19
  store half %1514, ptr addrspace(1) %address.local618, align 2, !dbg !19
  %address.local620 = or disjoint i64 %address.local617, 1, !dbg !19
  %address.local621 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local615, i64 %address.local620, !dbg !19
  store half %1513, ptr addrspace(1) %address.local621, align 2, !dbg !19
  %address.local623 = or disjoint i64 %address.local617, 2, !dbg !19
  %address.local624 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local615, i64 %address.local623, !dbg !19
  store half %1512, ptr addrspace(1) %address.local624, align 2, !dbg !19
  %address.local626 = or disjoint i64 %address.local617, 3, !dbg !19
  %address.local627 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local615, i64 %address.local626, !dbg !19
  store half %1511, ptr addrspace(1) %address.local627, align 2, !dbg !19
  %address.local629 = or disjoint i64 %address.local617, 16, !dbg !19
  %address.local630 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local615, i64 %address.local629, !dbg !19
  store half %1499, ptr addrspace(1) %address.local630, align 2, !dbg !19
  %address.local632 = or disjoint i64 %address.local617, 17, !dbg !19
  %address.local633 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local615, i64 %address.local632, !dbg !19
  store half %1498, ptr addrspace(1) %address.local633, align 2, !dbg !19
  %address.local635 = or disjoint i64 %address.local617, 18, !dbg !19
  %address.local636 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local615, i64 %address.local635, !dbg !19
  store half %1497, ptr addrspace(1) %address.local636, align 2, !dbg !19
  %address.local638 = or disjoint i64 %address.local617, 19, !dbg !19
  %address.local639 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local615, i64 %address.local638, !dbg !19
  store half %1496, ptr addrspace(1) %address.local639, align 2, !dbg !19
  %address.local643 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local615, i64 16384, !dbg !19
  %address.local644 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local643, i64 %address.local617, !dbg !19
  store half %1484, ptr addrspace(1) %address.local644, align 2, !dbg !19
  %address.local646 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local643, i64 %address.local620, !dbg !19
  store half %1483, ptr addrspace(1) %address.local646, align 2, !dbg !19
  %address.local648 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local643, i64 %address.local623, !dbg !19
  store half %1482, ptr addrspace(1) %address.local648, align 2, !dbg !19
  %address.local650 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local643, i64 %address.local626, !dbg !19
  store half %1481, ptr addrspace(1) %address.local650, align 2, !dbg !19
  %address.local652 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local643, i64 %address.local629, !dbg !19
  store half %1469, ptr addrspace(1) %address.local652, align 2, !dbg !19
  %address.local654 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local643, i64 %address.local632, !dbg !19
  store half %1468, ptr addrspace(1) %address.local654, align 2, !dbg !19
  %address.local656 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local643, i64 %address.local635, !dbg !19
  store half %1467, ptr addrspace(1) %address.local656, align 2, !dbg !19
  %address.local658 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local643, i64 %address.local638, !dbg !19
  store half %1466, ptr addrspace(1) %address.local658, align 2, !dbg !19
  %1517 = insertelement <4 x float> poison, float %1355, i64 0, !dbg !17
  %1518 = insertelement <4 x float> %1517, float %1361, i64 1, !dbg !17
  %1519 = insertelement <4 x float> %1518, float %1356, i64 2, !dbg !17
  %1520 = insertelement <4 x float> %1519, float %1362, i64 3, !dbg !17
  %normalized34 = fmul <4 x float> %1520, %row.reciprocal, !dbg !17
  %1521 = fptrunc <4 x float> %normalized34 to <4 x half>, !dbg !18
  %1522 = extractelement <4 x half> %1521, i64 3, !dbg !18
  %1523 = extractelement <4 x half> %1521, i64 2, !dbg !18
  %1524 = extractelement <4 x half> %1521, i64 1, !dbg !18
  %1525 = extractelement <4 x half> %1521, i64 0, !dbg !18
  %1526 = insertelement <4 x float> poison, float %1327, i64 0, !dbg !17
  %1527 = insertelement <4 x float> %1526, float %1333, i64 1, !dbg !17
  %1528 = insertelement <4 x float> %1527, float %1328, i64 2, !dbg !17
  %1529 = insertelement <4 x float> %1528, float %1334, i64 3, !dbg !17
  %normalized35 = fmul <4 x float> %1529, %row.reciprocal, !dbg !17
  %1530 = fptrunc <4 x float> %normalized35 to <4 x half>, !dbg !18
  %1531 = extractelement <4 x half> %1530, i64 3, !dbg !18
  %1532 = extractelement <4 x half> %1530, i64 2, !dbg !18
  %1533 = extractelement <4 x half> %1530, i64 1, !dbg !18
  %1534 = extractelement <4 x half> %1530, i64 0, !dbg !18
  %1535 = insertelement <4 x float> poison, float %1298, i64 0, !dbg !17
  %1536 = insertelement <4 x float> %1535, float %1304, i64 1, !dbg !17
  %1537 = insertelement <4 x float> %1536, float %1299, i64 2, !dbg !17
  %1538 = insertelement <4 x float> %1537, float %1305, i64 3, !dbg !17
  %normalized37 = fmul <4 x float> %1538, %row.reciprocal30, !dbg !17
  %1539 = fptrunc <4 x float> %normalized37 to <4 x half>, !dbg !18
  %1540 = extractelement <4 x half> %1539, i64 3, !dbg !18
  %1541 = extractelement <4 x half> %1539, i64 2, !dbg !18
  %1542 = extractelement <4 x half> %1539, i64 1, !dbg !18
  %1543 = extractelement <4 x half> %1539, i64 0, !dbg !18
  %1544 = insertelement <4 x float> poison, float %1270, i64 0, !dbg !17
  %1545 = insertelement <4 x float> %1544, float %1276, i64 1, !dbg !17
  %1546 = insertelement <4 x float> %1545, float %1271, i64 2, !dbg !17
  %1547 = insertelement <4 x float> %1546, float %1277, i64 3, !dbg !17
  %normalized38 = fmul <4 x float> %1547, %row.reciprocal30, !dbg !17
  %1548 = fptrunc <4 x float> %normalized38 to <4 x half>, !dbg !18
  %1549 = extractelement <4 x half> %1548, i64 3, !dbg !18
  %1550 = extractelement <4 x half> %1548, i64 2, !dbg !18
  %1551 = extractelement <4 x half> %1548, i64 1, !dbg !18
  %1552 = extractelement <4 x half> %1548, i64 0, !dbg !18
  %address.local673 = or disjoint i64 %address.local617, 32, !dbg !19
  %address.local674 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local615, i64 %address.local673, !dbg !19
  store half %1552, ptr addrspace(1) %address.local674, align 2, !dbg !19
  %address.local676 = or disjoint i64 %address.local617, 33, !dbg !19
  %address.local677 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local615, i64 %address.local676, !dbg !19
  store half %1551, ptr addrspace(1) %address.local677, align 2, !dbg !19
  %address.local679 = or disjoint i64 %address.local617, 34, !dbg !19
  %address.local680 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local615, i64 %address.local679, !dbg !19
  store half %1550, ptr addrspace(1) %address.local680, align 2, !dbg !19
  %address.local682 = or disjoint i64 %address.local617, 35, !dbg !19
  %address.local683 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local615, i64 %address.local682, !dbg !19
  store half %1549, ptr addrspace(1) %address.local683, align 2, !dbg !19
  %address.local685 = or disjoint i64 %address.local617, 48, !dbg !19
  %address.local686 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local615, i64 %address.local685, !dbg !19
  store half %1543, ptr addrspace(1) %address.local686, align 2, !dbg !19
  %address.local688 = or disjoint i64 %address.local617, 49, !dbg !19
  %address.local689 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local615, i64 %address.local688, !dbg !19
  store half %1542, ptr addrspace(1) %address.local689, align 2, !dbg !19
  %address.local691 = or disjoint i64 %address.local617, 50, !dbg !19
  %address.local692 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local615, i64 %address.local691, !dbg !19
  store half %1541, ptr addrspace(1) %address.local692, align 2, !dbg !19
  %address.local694 = or i64 %1515, 51, !dbg !19
  %address.local695 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local615, i64 %address.local694, !dbg !19
  store half %1540, ptr addrspace(1) %address.local695, align 2, !dbg !19
  %address.local700 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local643, i64 %address.local673, !dbg !19
  store half %1534, ptr addrspace(1) %address.local700, align 2, !dbg !19
  %address.local702 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local643, i64 %address.local676, !dbg !19
  store half %1533, ptr addrspace(1) %address.local702, align 2, !dbg !19
  %address.local704 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local643, i64 %address.local679, !dbg !19
  store half %1532, ptr addrspace(1) %address.local704, align 2, !dbg !19
  %address.local706 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local643, i64 %address.local682, !dbg !19
  store half %1531, ptr addrspace(1) %address.local706, align 2, !dbg !19
  %address.local708 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local643, i64 %address.local685, !dbg !19
  store half %1525, ptr addrspace(1) %address.local708, align 2, !dbg !19
  %address.local710 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local643, i64 %address.local688, !dbg !19
  store half %1524, ptr addrspace(1) %address.local710, align 2, !dbg !19
  %address.local712 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local643, i64 %address.local691, !dbg !19
  store half %1523, ptr addrspace(1) %address.local712, align 2, !dbg !19
  %address.local714 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local643, i64 %address.local694, !dbg !19
  store half %1522, ptr addrspace(1) %address.local714, align 2, !dbg !19
  %1553 = insertelement <4 x float> poison, float %1241, i64 0, !dbg !17
  %1554 = insertelement <4 x float> %1553, float %1247, i64 1, !dbg !17
  %1555 = insertelement <4 x float> %1554, float %1242, i64 2, !dbg !17
  %1556 = insertelement <4 x float> %1555, float %1248, i64 3, !dbg !17
  %normalized40 = fmul <4 x float> %1556, %row.reciprocal, !dbg !17
  %1557 = fptrunc <4 x float> %normalized40 to <4 x half>, !dbg !18
  %1558 = extractelement <4 x half> %1557, i64 3, !dbg !18
  %1559 = extractelement <4 x half> %1557, i64 2, !dbg !18
  %1560 = extractelement <4 x half> %1557, i64 1, !dbg !18
  %1561 = extractelement <4 x half> %1557, i64 0, !dbg !18
  %1562 = insertelement <4 x float> poison, float %1213, i64 0, !dbg !17
  %1563 = insertelement <4 x float> %1562, float %1219, i64 1, !dbg !17
  %1564 = insertelement <4 x float> %1563, float %1214, i64 2, !dbg !17
  %1565 = insertelement <4 x float> %1564, float %1220, i64 3, !dbg !17
  %normalized41 = fmul <4 x float> %1565, %row.reciprocal, !dbg !17
  %1566 = fptrunc <4 x float> %normalized41 to <4 x half>, !dbg !18
  %1567 = extractelement <4 x half> %1566, i64 3, !dbg !18
  %1568 = extractelement <4 x half> %1566, i64 2, !dbg !18
  %1569 = extractelement <4 x half> %1566, i64 1, !dbg !18
  %1570 = extractelement <4 x half> %1566, i64 0, !dbg !18
  %1571 = insertelement <4 x float> poison, float %1184, i64 0, !dbg !17
  %1572 = insertelement <4 x float> %1571, float %1190, i64 1, !dbg !17
  %1573 = insertelement <4 x float> %1572, float %1185, i64 2, !dbg !17
  %1574 = insertelement <4 x float> %1573, float %1191, i64 3, !dbg !17
  %normalized43 = fmul <4 x float> %1574, %row.reciprocal30, !dbg !17
  %1575 = fptrunc <4 x float> %normalized43 to <4 x half>, !dbg !18
  %1576 = extractelement <4 x half> %1575, i64 3, !dbg !18
  %1577 = extractelement <4 x half> %1575, i64 2, !dbg !18
  %1578 = extractelement <4 x half> %1575, i64 1, !dbg !18
  %1579 = extractelement <4 x half> %1575, i64 0, !dbg !18
  %1580 = insertelement <4 x float> poison, float %1156, i64 0, !dbg !17
  %1581 = insertelement <4 x float> %1580, float %1162, i64 1, !dbg !17
  %1582 = insertelement <4 x float> %1581, float %1157, i64 2, !dbg !17
  %1583 = insertelement <4 x float> %1582, float %1163, i64 3, !dbg !17
  %normalized44 = fmul <4 x float> %1583, %row.reciprocal30, !dbg !17
  %1584 = fptrunc <4 x float> %normalized44 to <4 x half>, !dbg !18
  %1585 = extractelement <4 x half> %1584, i64 3, !dbg !18
  %1586 = extractelement <4 x half> %1584, i64 2, !dbg !18
  %1587 = extractelement <4 x half> %1584, i64 1, !dbg !18
  %1588 = extractelement <4 x half> %1584, i64 0, !dbg !18
  %address.local729 = or disjoint i64 %address.local617, 64, !dbg !19
  %address.local730 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local615, i64 %address.local729, !dbg !19
  store half %1588, ptr addrspace(1) %address.local730, align 2, !dbg !19
  %address.local732 = or disjoint i64 %address.local617, 65, !dbg !19
  %address.local733 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local615, i64 %address.local732, !dbg !19
  store half %1587, ptr addrspace(1) %address.local733, align 2, !dbg !19
  %address.local735 = or disjoint i64 %address.local617, 66, !dbg !19
  %address.local736 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local615, i64 %address.local735, !dbg !19
  store half %1586, ptr addrspace(1) %address.local736, align 2, !dbg !19
  %address.local738 = or disjoint i64 %address.local617, 67, !dbg !19
  %address.local739 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local615, i64 %address.local738, !dbg !19
  store half %1585, ptr addrspace(1) %address.local739, align 2, !dbg !19
  %address.local741 = or disjoint i64 %address.local617, 80, !dbg !19
  %address.local742 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local615, i64 %address.local741, !dbg !19
  store half %1579, ptr addrspace(1) %address.local742, align 2, !dbg !19
  %address.local744 = or disjoint i64 %address.local617, 81, !dbg !19
  %address.local745 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local615, i64 %address.local744, !dbg !19
  store half %1578, ptr addrspace(1) %address.local745, align 2, !dbg !19
  %address.local747 = or disjoint i64 %address.local617, 82, !dbg !19
  %address.local748 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local615, i64 %address.local747, !dbg !19
  store half %1577, ptr addrspace(1) %address.local748, align 2, !dbg !19
  %address.local750 = or disjoint i64 %address.local617, 83, !dbg !19
  %address.local751 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local615, i64 %address.local750, !dbg !19
  store half %1576, ptr addrspace(1) %address.local751, align 2, !dbg !19
  %address.local756 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local643, i64 %address.local729, !dbg !19
  store half %1570, ptr addrspace(1) %address.local756, align 2, !dbg !19
  %address.local758 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local643, i64 %address.local732, !dbg !19
  store half %1569, ptr addrspace(1) %address.local758, align 2, !dbg !19
  %address.local760 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local643, i64 %address.local735, !dbg !19
  store half %1568, ptr addrspace(1) %address.local760, align 2, !dbg !19
  %address.local762 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local643, i64 %address.local738, !dbg !19
  store half %1567, ptr addrspace(1) %address.local762, align 2, !dbg !19
  %address.local764 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local643, i64 %address.local741, !dbg !19
  store half %1561, ptr addrspace(1) %address.local764, align 2, !dbg !19
  %address.local766 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local643, i64 %address.local744, !dbg !19
  store half %1560, ptr addrspace(1) %address.local766, align 2, !dbg !19
  %address.local768 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local643, i64 %address.local747, !dbg !19
  store half %1559, ptr addrspace(1) %address.local768, align 2, !dbg !19
  %address.local770 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local643, i64 %address.local750, !dbg !19
  store half %1558, ptr addrspace(1) %address.local770, align 2, !dbg !19
  %1589 = insertelement <4 x float> poison, float %1127, i64 0, !dbg !17
  %1590 = insertelement <4 x float> %1589, float %1133, i64 1, !dbg !17
  %1591 = insertelement <4 x float> %1590, float %1128, i64 2, !dbg !17
  %1592 = insertelement <4 x float> %1591, float %1134, i64 3, !dbg !17
  %normalized46 = fmul <4 x float> %1592, %row.reciprocal, !dbg !17
  %1593 = fptrunc <4 x float> %normalized46 to <4 x half>, !dbg !18
  %1594 = extractelement <4 x half> %1593, i64 3, !dbg !18
  %1595 = extractelement <4 x half> %1593, i64 2, !dbg !18
  %1596 = extractelement <4 x half> %1593, i64 1, !dbg !18
  %1597 = extractelement <4 x half> %1593, i64 0, !dbg !18
  %1598 = insertelement <4 x float> poison, float %1099, i64 0, !dbg !17
  %1599 = insertelement <4 x float> %1598, float %1105, i64 1, !dbg !17
  %1600 = insertelement <4 x float> %1599, float %1100, i64 2, !dbg !17
  %1601 = insertelement <4 x float> %1600, float %1106, i64 3, !dbg !17
  %normalized47 = fmul <4 x float> %1601, %row.reciprocal, !dbg !17
  %1602 = fptrunc <4 x float> %normalized47 to <4 x half>, !dbg !18
  %1603 = extractelement <4 x half> %1602, i64 3, !dbg !18
  %1604 = extractelement <4 x half> %1602, i64 2, !dbg !18
  %1605 = extractelement <4 x half> %1602, i64 1, !dbg !18
  %1606 = extractelement <4 x half> %1602, i64 0, !dbg !18
  %1607 = insertelement <4 x float> poison, float %1070, i64 0, !dbg !17
  %1608 = insertelement <4 x float> %1607, float %1076, i64 1, !dbg !17
  %1609 = insertelement <4 x float> %1608, float %1071, i64 2, !dbg !17
  %1610 = insertelement <4 x float> %1609, float %1077, i64 3, !dbg !17
  %normalized49 = fmul <4 x float> %1610, %row.reciprocal30, !dbg !17
  %1611 = fptrunc <4 x float> %normalized49 to <4 x half>, !dbg !18
  %1612 = extractelement <4 x half> %1611, i64 3, !dbg !18
  %1613 = extractelement <4 x half> %1611, i64 2, !dbg !18
  %1614 = extractelement <4 x half> %1611, i64 1, !dbg !18
  %1615 = extractelement <4 x half> %1611, i64 0, !dbg !18
  %1616 = insertelement <4 x float> poison, float %1042, i64 0, !dbg !17
  %1617 = insertelement <4 x float> %1616, float %1048, i64 1, !dbg !17
  %1618 = insertelement <4 x float> %1617, float %1043, i64 2, !dbg !17
  %1619 = insertelement <4 x float> %1618, float %1049, i64 3, !dbg !17
  %normalized50 = fmul <4 x float> %1619, %row.reciprocal30, !dbg !17
  %1620 = fptrunc <4 x float> %normalized50 to <4 x half>, !dbg !18
  %1621 = extractelement <4 x half> %1620, i64 3, !dbg !18
  %1622 = extractelement <4 x half> %1620, i64 2, !dbg !18
  %1623 = extractelement <4 x half> %1620, i64 1, !dbg !18
  %1624 = extractelement <4 x half> %1620, i64 0, !dbg !18
  %address.local785 = or disjoint i64 %address.local617, 96, !dbg !19
  %address.local786 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local615, i64 %address.local785, !dbg !19
  store half %1624, ptr addrspace(1) %address.local786, align 2, !dbg !19
  %address.local788 = or disjoint i64 %address.local617, 97, !dbg !19
  %address.local789 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local615, i64 %address.local788, !dbg !19
  store half %1623, ptr addrspace(1) %address.local789, align 2, !dbg !19
  %address.local791 = or disjoint i64 %address.local617, 98, !dbg !19
  %address.local792 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local615, i64 %address.local791, !dbg !19
  store half %1622, ptr addrspace(1) %address.local792, align 2, !dbg !19
  %address.local794 = or disjoint i64 %address.local617, 99, !dbg !19
  %address.local795 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local615, i64 %address.local794, !dbg !19
  store half %1621, ptr addrspace(1) %address.local795, align 2, !dbg !19
  %address.local797 = or disjoint i64 %address.local617, 112, !dbg !19
  %address.local798 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local615, i64 %address.local797, !dbg !19
  store half %1615, ptr addrspace(1) %address.local798, align 2, !dbg !19
  %address.local800 = or disjoint i64 %address.local617, 113, !dbg !19
  %address.local801 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local615, i64 %address.local800, !dbg !19
  store half %1614, ptr addrspace(1) %address.local801, align 2, !dbg !19
  %address.local803 = or disjoint i64 %address.local617, 114, !dbg !19
  %address.local804 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local615, i64 %address.local803, !dbg !19
  store half %1613, ptr addrspace(1) %address.local804, align 2, !dbg !19
  %address.local806 = or i64 %1515, 115, !dbg !19
  %address.local807 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local615, i64 %address.local806, !dbg !19
  store half %1612, ptr addrspace(1) %address.local807, align 2, !dbg !19
  %address.local812 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local643, i64 %address.local785, !dbg !19
  store half %1606, ptr addrspace(1) %address.local812, align 2, !dbg !19
  %address.local814 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local643, i64 %address.local788, !dbg !19
  store half %1605, ptr addrspace(1) %address.local814, align 2, !dbg !19
  %address.local816 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local643, i64 %address.local791, !dbg !19
  store half %1604, ptr addrspace(1) %address.local816, align 2, !dbg !19
  %address.local818 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local643, i64 %address.local794, !dbg !19
  store half %1603, ptr addrspace(1) %address.local818, align 2, !dbg !19
  %address.local820 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local643, i64 %address.local797, !dbg !19
  store half %1597, ptr addrspace(1) %address.local820, align 2, !dbg !19
  %address.local822 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local643, i64 %address.local800, !dbg !19
  store half %1596, ptr addrspace(1) %address.local822, align 2, !dbg !19
  %address.local824 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local643, i64 %address.local803, !dbg !19
  store half %1595, ptr addrspace(1) %address.local824, align 2, !dbg !19
  %address.local826 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local643, i64 %address.local806, !dbg !19
  store half %1594, ptr addrspace(1) %address.local826, align 2, !dbg !19
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

attributes #0 = { convergent willreturn memory(read) }
attributes #1 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1" "frisk.cross_bn_k0"="sum-anchor-v1" "frisk.global_prefetch"="8" "frisk.k_dma"="64-swizzle:1" "frisk.k_lds_prefetch"="8" "frisk.k_tail_prefetch"="4" "frisk.register_comparisons"="1" "frisk.register_lifetimes"="1" "frisk.softmax_fragments"="pv" "uniform-work-group-size" }
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
!llvm.module.flags = !{!31}

!0 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/async_task_noncausal")
!1 = !{}
!2 = !DISubroutineType(types: !1)
!3 = distinct !DICompileUnit(language: DW_LANG_C, file: !0, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!4 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !0, file: !0, line: 23, type: !2, scopeLine: 23, spFlags: DISPFlagDefinition, unit: !3)
!5 = !DILocation(line: 23, column: 3, scope: !4)
!6 = !DILocation(line: 74, column: 20, scope: !4)
!8 = !DILocation(line: 57, column: 24, scope: !4)
!9 = !DILocation(line: 67, column: 22, scope: !4)
!12 = !DILocation(line: 69, column: 24, scope: !4)
!13 = !DILocation(line: 68, column: 22, scope: !4)
!14 = !DILocation(line: 87, column: 23, scope: !4)
!15 = !DILocation(line: 70, column: 26, scope: !4)
!16 = !DILocation(line: 86, column: 20, scope: !4)
!17 = !DILocation(line: 92, column: 23, scope: !4)
!18 = !DILocation(line: 93, column: 19, scope: !4)
!19 = !DILocation(line: 94, column: 9, scope: !4)
!20 = !DILocation(line: 71, column: 23, scope: !4)
!21 = !DILocation(line: 75, column: 22, scope: !4)
!22 = !DILocation(line: 76, column: 23, scope: !4)
!23 = !DILocation(line: 77, column: 20, scope: !4)
!24 = !DILocation(line: 78, column: 22, scope: !4)
!25 = !DILocation(line: 79, column: 22, scope: !4)
!26 = !DILocation(line: 80, column: 20, scope: !4)
!27 = !DILocation(line: 81, column: 16, scope: !4)
!28 = !DILocation(line: 88, column: 22, scope: !4)
!30 = !{i32 256, i32 1, i32 1}
!31 = !{i32 2, !"Debug Info Version", i32 3}
!32 = !{i32 0, i32 256}
!33 = !{i32 4}
!34 = !{i64 8192}
!35 = !{i32 0}
!36 = !{i32 1}
!37 = !{i32 2}
!38 = !{i32 3}
!39 = !{i32 5}
!40 = !{i32 6}
!41 = !{i32 7}
!42 = !{i32 8}
!43 = !{i32 9}
!44 = !{i32 10}
!45 = !{i32 11}
!46 = !{i32 12}
!47 = !{i32 13}
!48 = !{i32 14}
!49 = !{i32 15}
!50 = !{i32 16}
!51 = !{i32 17}
!52 = !{i32 18}
!53 = !{i32 19}
!54 = !{i32 20}
!55 = !{i32 21}
!56 = !{i32 22}
!57 = !{i32 23}
