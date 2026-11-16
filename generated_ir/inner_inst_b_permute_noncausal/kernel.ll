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
  %address.local351 = shl nuw nsw i64 %address.local, 3, !dbg !9
  %address.local352 = and i64 %address.local351, 1536, !dbg !9
  %address.local353 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local352, !dbg !9
  %address.local354 = ptrtoint ptr addrspace(3) %address.local353 to i64, !dbg !9
  %address.local355 = trunc i64 %address.local354 to i32, !dbg !9
  %36 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local355), !dbg !9
  %address.local356 = shl nuw nsw i64 %address.local, 7, !dbg !9
  %address.local357 = and i64 %address.local356, 16256, !dbg !9
  %address.local358 = lshr i64 %address.local, 4, !dbg !9
  %address.local359 = and i64 %address.local358, 8, !dbg !9
  %address.local360 = or disjoint i64 %address.local357, %address.local359, !dbg !9
  %37 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !9
  %.idx1129 = shl i64 %34, 8, !dbg !9
  %38 = add i64 %.idx, %.idx1129, !dbg !9
  %address.local361.idx = shl nuw nsw i64 %address.local360, 1, !dbg !9
  %39 = or disjoint i64 %38, %address.local361.idx, !dbg !9
  %dma.buffer.offset1127 = trunc i64 %39 to i32, !dbg !9
  %40 = insertelement <2 x i64> poison, i64 %37, i64 0, !dbg !9
  %41 = bitcast <2 x i64> %40 to <4 x i32>, !dbg !9
  %dma.buffer.rsrc1128 = shufflevector <4 x i32> %41, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !9
  %42 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %36, i32 %dma.buffer.offset1127, <4 x i32> %dma.buffer.rsrc1128) #10, !dbg !9
  %address.local362 = or disjoint i64 %address.local, 256, !dbg !9
  %address.local363 = shl nuw nsw i64 %address.local362, 3, !dbg !9
  %address.local364 = and i64 %address.local363, 3584, !dbg !9
  %address.local365 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local364, !dbg !9
  %address.local366 = ptrtoint ptr addrspace(3) %address.local365 to i64, !dbg !9
  %address.local367 = trunc i64 %address.local366 to i32, !dbg !9
  %43 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local367), !dbg !9
  %address.local370 = lshr i64 %address.local362, 4, !dbg !9
  %address.local371 = and i64 %address.local370, 24, !dbg !9
  %address.local372 = or disjoint i64 %address.local357, %address.local371, !dbg !9
  %address.local373.idx = shl nuw nsw i64 %address.local372, 1, !dbg !9
  %44 = or disjoint i64 %38, %address.local373.idx, !dbg !9
  %dma.buffer.offset1125 = trunc i64 %44 to i32, !dbg !9
  %45 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %43, i32 %dma.buffer.offset1125, <4 x i32> %dma.buffer.rsrc1128) #10, !dbg !9
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %46 = shl nuw nsw i64 %31, 19, !dbg !9
  %47 = add i64 %46, %35, !dbg !9
  %address.local374 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local352, !dbg !9
  %address.local375 = ptrtoint ptr addrspace(3) %address.local374 to i64, !dbg !9
  %address.local376 = trunc i64 %address.local375 to i32, !dbg !9
  %48 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local376), !dbg !9
  %49 = or disjoint i64 %47, 32, !dbg !9
  %50 = or disjoint i64 %address.local360, %49, !dbg !9
  %.tr = trunc i64 %50 to i32, !dbg !9
  %dma.buffer.offset1123 = shl i32 %.tr, 1, !dbg !9
  %51 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %48, i32 %dma.buffer.offset1123, <4 x i32> %dma.buffer.rsrc1128) #10, !dbg !9
  %address.local378 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local364, !dbg !9
  %address.local379 = ptrtoint ptr addrspace(3) %address.local378 to i64, !dbg !9
  %address.local380 = trunc i64 %address.local379 to i32, !dbg !9
  %52 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local380), !dbg !9
  %53 = or disjoint i64 %address.local372, %49, !dbg !9
  %.tr1133 = trunc i64 %53 to i32, !dbg !9
  %dma.buffer.offset1121 = shl i32 %.tr1133, 1, !dbg !9
  %54 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %52, i32 %dma.buffer.offset1121, <4 x i32> %dma.buffer.rsrc1128) #10, !dbg !9
  %address.local382 = lshr i32 %address.thread, 2, !dbg !12
  %address.local383 = and i32 %address.local382, 4, !dbg !12
  %address.local384 = shl nuw nsw i32 %address.thread, 1, !dbg !12
  %address.local385 = and i32 %address.local384, 384, !dbg !12
  %address.local386 = or disjoint i32 %address.local383, %address.local385, !dbg !12
  %address.local387 = shl nuw nsw i32 %address.thread, 3, !dbg !12
  %address.local388 = and i32 %address.local387, 120, !dbg !12
  %address.local389 = or disjoint i32 %address.local386, %address.local388, !dbg !12
  %address.local390 = shl nuw nsw i32 %address.thread, 5, !dbg !12
  %address.local391 = and i32 %address.local390, 1024, !dbg !12
  %address.local392 = or disjoint i32 %address.local389, %address.local391, !dbg !12
  %address.local393 = zext nneg i32 %address.local392 to i64, !dbg !12
  %address.local394 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local393, !dbg !12
  %55 = load <4 x half>, ptr addrspace(3) %address.local394, align 8, !dbg !12
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local394, i64 4096, !dbg !12
  %56 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !12
  %57 = fmul <4 x half> %55, splat (half 0xH3015), !dbg !12
  %lds.const.gep950 = getelementptr i8, ptr addrspace(3) %address.local394, i64 1024, !dbg !12
  %58 = load <4 x half>, ptr addrspace(3) %lds.const.gep950, align 8, !dbg !12
  %59 = fmul <4 x half> %56, splat (half 0xH3015), !dbg !12
  %lds.const.gep951 = getelementptr i8, ptr addrspace(3) %address.local394, i64 5120, !dbg !12
  %60 = load <4 x half>, ptr addrspace(3) %lds.const.gep951, align 8, !dbg !12
  %61 = fmul <4 x half> %58, splat (half 0xH3015), !dbg !12
  %62 = fmul <4 x half> %60, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %63 = or disjoint i64 %47, 64, !dbg !9
  %64 = or disjoint i64 %address.local360, %63, !dbg !9
  %.tr1135 = trunc i64 %64 to i32, !dbg !9
  %dma.buffer.offset1119 = shl i32 %.tr1135, 1, !dbg !9
  %65 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %36, i32 %dma.buffer.offset1119, <4 x i32> %dma.buffer.rsrc1128) #10, !dbg !9
  %66 = or disjoint i64 %address.local372, %63, !dbg !9
  %.tr1137 = trunc i64 %66 to i32, !dbg !9
  %dma.buffer.offset1117 = shl i32 %.tr1137, 1, !dbg !9
  %67 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %43, i32 %dma.buffer.offset1117, <4 x i32> %dma.buffer.rsrc1128) #10, !dbg !9
  %lds.const.gep952 = getelementptr i8, ptr addrspace(3) %address.local394, i64 8192, !dbg !12
  %68 = load <4 x half>, ptr addrspace(3) %lds.const.gep952, align 8, !dbg !12
  %lds.const.gep953 = getelementptr i8, ptr addrspace(3) %address.local394, i64 12288, !dbg !12
  %69 = load <4 x half>, ptr addrspace(3) %lds.const.gep953, align 8, !dbg !12
  %lds.const.gep954 = getelementptr i8, ptr addrspace(3) %address.local394, i64 9216, !dbg !12
  %70 = load <4 x half>, ptr addrspace(3) %lds.const.gep954, align 8, !dbg !12
  %71 = fmul <4 x half> %68, splat (half 0xH3015), !dbg !12
  %lds.const.gep955 = getelementptr i8, ptr addrspace(3) %address.local394, i64 13312, !dbg !12
  %72 = load <4 x half>, ptr addrspace(3) %lds.const.gep955, align 8, !dbg !12
  %73 = fmul <4 x half> %69, splat (half 0xH3015), !dbg !12
  %74 = fmul <4 x half> %70, splat (half 0xH3015), !dbg !12
  %75 = fmul <4 x half> %72, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %76 = or disjoint i64 %47, 96, !dbg !9
  %77 = or disjoint i64 %address.local360, %76, !dbg !9
  %.tr1139 = trunc i64 %77 to i32, !dbg !9
  %dma.buffer.offset1115 = shl i32 %.tr1139, 1, !dbg !9
  %78 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %48, i32 %dma.buffer.offset1115, <4 x i32> %dma.buffer.rsrc1128) #10, !dbg !9
  %79 = or disjoint i64 %address.local372, %76, !dbg !9
  %.tr1141 = trunc i64 %79 to i32, !dbg !9
  %dma.buffer.offset1113 = shl i32 %.tr1141, 1, !dbg !9
  %80 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %52, i32 %dma.buffer.offset1113, <4 x i32> %dma.buffer.rsrc1128) #10, !dbg !9
  %81 = load <4 x half>, ptr addrspace(3) %address.local394, align 8, !dbg !12
  %82 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !12
  %83 = load <4 x half>, ptr addrspace(3) %lds.const.gep950, align 8, !dbg !12
  %84 = fmul <4 x half> %81, splat (half 0xH3015), !dbg !12
  %85 = load <4 x half>, ptr addrspace(3) %lds.const.gep951, align 8, !dbg !12
  %86 = fmul <4 x half> %82, splat (half 0xH3015), !dbg !12
  %87 = fmul <4 x half> %83, splat (half 0xH3015), !dbg !12
  %88 = fmul <4 x half> %85, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %89 = load <4 x half>, ptr addrspace(3) %lds.const.gep952, align 8, !dbg !12
  %90 = load <4 x half>, ptr addrspace(3) %lds.const.gep953, align 8, !dbg !12
  %91 = load <4 x half>, ptr addrspace(3) %lds.const.gep954, align 8, !dbg !12
  %92 = fmul <4 x half> %89, splat (half 0xH3015), !dbg !12
  %93 = load <4 x half>, ptr addrspace(3) %lds.const.gep955, align 8, !dbg !12
  %94 = fmul <4 x half> %90, splat (half 0xH3015), !dbg !12
  %95 = fmul <4 x half> %91, splat (half 0xH3015), !dbg !12
  %96 = fmul <4 x half> %93, splat (half 0xH3015), !dbg !12
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %k0.prime. = shl i64 %25, 19, !dbg !13
  %k0.prime.1684 = and i64 %k0.prime., 16252928, !dbg !13
  %k0.prime.1685 = shl nuw nsw i64 0, 14, !dbg !13
  %k0.prime.1686 = or disjoint i64 %k0.prime.1684, %k0.prime.1685, !dbg !13
  %k0.prime.kdma.source = zext nneg i32 %kdma.owner to i64, !dbg !13
  %k0.prime.kdma.source1233 = lshr i64 %k0.prime.kdma.source, 4, !dbg !13
  %k0.prime.kdma.source1234 = and i64 %k0.prime.kdma.source1233, 8, !dbg !13
  %k0.prime.kdma.source1235 = and i64 %k0.prime.kdma.source, 91, !dbg !13
  %k0.prime.kdma.source1236 = lshr i64 %k0.prime.kdma.source, 2, !dbg !13
  %k0.prime.kdma.source1237 = and i64 %k0.prime.kdma.source1236, 4, !dbg !13
  %k0.prime.kdma.source1238 = add nuw nsw i64 %k0.prime.kdma.source1237, %k0.prime.kdma.source, !dbg !13
  %k0.prime.kdma.source1239 = and i64 %k0.prime.kdma.source1238, 4, !dbg !13
  %k0.prime.kdma.source1240 = or disjoint i64 %k0.prime.kdma.source1235, %k0.prime.kdma.source1239, !dbg !13
  %k0.prime.kdma.source1241 = shl nuw nsw i64 %k0.prime.kdma.source1240, 7, !dbg !13
  %k0.prime.kdma.source1242 = or disjoint i64 %k0.prime.kdma.source1234, %k0.prime.kdma.source1241, !dbg !13
  %k0.prime.kdma.source1243 = or disjoint i64 %k0.prime.1686, %k0.prime.kdma.source1242, !dbg !13
  %k0.prime.kdma.source1244 = trunc nuw i64 %k0.prime.kdma.source1243 to i32, !dbg !13
  %k0.prime.kdma.source1245 = shl nuw nsw i32 %k0.prime.kdma.source1244, 1, !dbg !13
  %k0.prime.kdma.offset = or disjoint i32 %k0.prime.kdma.source1245, %kdma.intra, !dbg !13
  %k0.prime.1687 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !13
  %k0.prime.1688 = insertelement <2 x i64> poison, i64 %k0.prime.1687, i64 0, !dbg !13
  %k0.prime.1689 = bitcast <2 x i64> %k0.prime.1688 to <4 x i32>, !dbg !13
  %k0.prime.dma.buffer.rsrc1112 = shufflevector <4 x i32> %k0.prime.1689, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !13
  %k0.prime.1690 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %36, i32 %k0.prime.kdma.offset, <4 x i32> %k0.prime.dma.buffer.rsrc1112) #10, !dbg !13, !frisk.cross_bn_k0 !1
  %k0.prime.kdma.lds1261 = add i32 %36, 512, !dbg !13
  %k0.prime.kdma.source1246 = zext nneg i32 %kdma.owner.hi to i64, !dbg !13
  %k0.prime.kdma.source1247 = lshr i64 %k0.prime.kdma.source1246, 4, !dbg !13
  %k0.prime.kdma.source1248 = and i64 %k0.prime.kdma.source1247, 8, !dbg !13
  %k0.prime.kdma.source1249 = and i64 %k0.prime.kdma.source1246, 123, !dbg !13
  %k0.prime.kdma.source1250 = lshr i64 %k0.prime.kdma.source1246, 2, !dbg !13
  %k0.prime.kdma.source1251 = and i64 %k0.prime.kdma.source1250, 4, !dbg !13
  %k0.prime.kdma.source1252 = add nuw nsw i64 %k0.prime.kdma.source1251, %k0.prime.kdma.source1246, !dbg !13
  %k0.prime.kdma.source1253 = and i64 %k0.prime.kdma.source1252, 4, !dbg !13
  %k0.prime.kdma.source1254 = or disjoint i64 %k0.prime.kdma.source1249, %k0.prime.kdma.source1253, !dbg !13
  %k0.prime.kdma.source1255 = shl nuw nsw i64 %k0.prime.kdma.source1254, 7, !dbg !13
  %k0.prime.kdma.source1256 = or disjoint i64 %k0.prime.kdma.source1248, %k0.prime.kdma.source1255, !dbg !13
  %k0.prime.kdma.source1257 = or disjoint i64 %k0.prime.1686, %k0.prime.kdma.source1256, !dbg !13
  %k0.prime.kdma.source1258 = trunc nuw i64 %k0.prime.kdma.source1257 to i32, !dbg !13
  %k0.prime.kdma.source1259 = shl nuw nsw i32 %k0.prime.kdma.source1258, 1, !dbg !13
  %k0.prime.kdma.offset1260 = or disjoint i32 %k0.prime.kdma.source1259, %kdma.intra.hi, !dbg !13
  %k0.prime.1691 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.kdma.lds1261, i32 %k0.prime.kdma.offset1260, <4 x i32> %k0.prime.dma.buffer.rsrc1112) #10, !dbg !13, !frisk.cross_bn_k0 !1
  %k0.prime.kdma.source1273 = or disjoint i64 %k0.prime.kdma.source1242, 16, !dbg !13
  %k0.prime.kdma.source1274 = or disjoint i64 %k0.prime.1686, %k0.prime.kdma.source1273, !dbg !13
  %k0.prime.kdma.source1275 = trunc nuw i64 %k0.prime.kdma.source1274 to i32, !dbg !13
  %k0.prime.kdma.source1276 = shl nuw nsw i32 %k0.prime.kdma.source1275, 1, !dbg !13
  %k0.prime.kdma.offset1277 = or disjoint i32 %k0.prime.kdma.source1276, %kdma.intra, !dbg !13
  %k0.prime.1692 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %43, i32 %k0.prime.kdma.offset1277, <4 x i32> %k0.prime.dma.buffer.rsrc1112) #10, !dbg !13, !frisk.cross_bn_k0 !1
  %k0.prime.kdma.lds1295 = add i32 %43, 512, !dbg !13
  %k0.prime.kdma.source1290 = or disjoint i64 %k0.prime.kdma.source1256, 16, !dbg !13
  %k0.prime.kdma.source1291 = or disjoint i64 %k0.prime.1686, %k0.prime.kdma.source1290, !dbg !13
  %k0.prime.kdma.source1292 = trunc nuw i64 %k0.prime.kdma.source1291 to i32, !dbg !13
  %k0.prime.kdma.source1293 = shl nuw nsw i32 %k0.prime.kdma.source1292, 1, !dbg !13
  %k0.prime.kdma.offset1294 = or disjoint i32 %k0.prime.kdma.source1293, %kdma.intra.hi, !dbg !13
  %k0.prime.1693 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.kdma.lds1295, i32 %k0.prime.kdma.offset1294, <4 x i32> %k0.prime.dma.buffer.rsrc1112) #10, !dbg !13, !frisk.cross_bn_k0 !1
  br label %97, !dbg !8

97:                                               ; preds = %k0.join, %4
  %98 = phi i64 [ %906, %k0.join ], [ 0, %4 ], !dbg !8
  %99 = phi [2 x <8 x float>] [ %984, %k0.join ], [ zeroinitializer, %4 ], !dbg !8
  %100 = phi [2 x <1 x float>] [ %749, %k0.join ], [ zeroinitializer, %4 ], !dbg !8
  %101 = phi [2 x <1 x float>] [ %986, %k0.join ], [ [<1 x float> splat (float 0xFFF0000000000000), <1 x float> splat (float 0xFFF0000000000000)], %4 ], !dbg !8
  %102 = phi [2 x <8 x float>] [ %990, %k0.join ], [ zeroinitializer, %4 ], !dbg !8
  %103 = phi [2 x <8 x float>] [ %994, %k0.join ], [ zeroinitializer, %4 ], !dbg !8
  %104 = phi [2 x <8 x float>] [ %998, %k0.join ], [ zeroinitializer, %4 ], !dbg !8
  %105 = icmp samesign ult i64 %98, 32, !dbg !8
  br i1 %105, label %106, label %999, !dbg !8

106:                                              ; preds = %97
  %107 = shl i64 %25, 19, !dbg !13
  %108 = and i64 %107, 16252928, !dbg !13
  %109 = shl nuw nsw i64 %98, 14, !dbg !13
  %110 = or disjoint i64 %108, %109, !dbg !13
  %address.thread445 = call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #9, !dbg !13, !range !32, !frisk.register_lifetimes !1
  %address.local446 = zext nneg i32 %address.thread445 to i64, !dbg !9
  %111 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !13
  %112 = insertelement <2 x i64> poison, i64 %111, i64 0, !dbg !13
  %113 = bitcast <2 x i64> %112 to <4 x i32>, !dbg !13
  %dma.buffer.rsrc1112 = shufflevector <4 x i32> %113, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !13
  %kdma.source = zext nneg i32 %kdma.owner to i64, !dbg !13
  %kdma.source1233 = lshr i64 %kdma.source, 4, !dbg !13
  %kdma.source1234 = and i64 %kdma.source1233, 8, !dbg !13
  %kdma.source1235 = and i64 %kdma.source, 91, !dbg !13
  %kdma.source1236 = lshr i64 %kdma.source, 2, !dbg !13
  %kdma.source1237 = and i64 %kdma.source1236, 4, !dbg !13
  %kdma.source1238 = add nuw nsw i64 %kdma.source1237, %kdma.source, !dbg !13
  %kdma.source1239 = and i64 %kdma.source1238, 4, !dbg !13
  %kdma.source1240 = or disjoint i64 %kdma.source1235, %kdma.source1239, !dbg !13
  %kdma.source1241 = shl nuw nsw i64 %kdma.source1240, 7, !dbg !13
  %kdma.source1242 = or disjoint i64 %kdma.source1234, %kdma.source1241, !dbg !13
  %kdma.source1243 = or disjoint i64 %110, %kdma.source1242, !dbg !13
  %kdma.source1244 = trunc nuw i64 %kdma.source1243 to i32, !dbg !13
  %kdma.source1245 = shl nuw nsw i32 %kdma.source1244, 1, !dbg !13
  %kdma.offset = or disjoint i32 %kdma.source1245, %kdma.intra, !dbg !13
  %kdma.source1246 = zext nneg i32 %kdma.owner.hi to i64, !dbg !13
  %kdma.source1247 = lshr i64 %kdma.source1246, 4, !dbg !13
  %kdma.source1248 = and i64 %kdma.source1247, 8, !dbg !13
  %kdma.source1249 = and i64 %kdma.source1246, 123, !dbg !13
  %kdma.source1250 = lshr i64 %kdma.source1246, 2, !dbg !13
  %kdma.source1251 = and i64 %kdma.source1250, 4, !dbg !13
  %kdma.source1252 = add nuw nsw i64 %kdma.source1251, %kdma.source1246, !dbg !13
  %kdma.source1253 = and i64 %kdma.source1252, 4, !dbg !13
  %kdma.source1254 = or disjoint i64 %kdma.source1249, %kdma.source1253, !dbg !13
  %kdma.source1255 = shl nuw nsw i64 %kdma.source1254, 7, !dbg !13
  %kdma.source1256 = or disjoint i64 %kdma.source1248, %kdma.source1255, !dbg !13
  %kdma.source1257 = or disjoint i64 %110, %kdma.source1256, !dbg !13
  %kdma.source1258 = trunc nuw i64 %kdma.source1257 to i32, !dbg !13
  %kdma.source1259 = shl nuw nsw i32 %kdma.source1258, 1, !dbg !13
  %kdma.offset1260 = or disjoint i32 %kdma.source1259, %kdma.intra.hi, !dbg !13
  %kdma.lds1261 = add i32 %36, 512, !dbg !13
  %kdma.source1273 = or disjoint i64 %kdma.source1242, 16, !dbg !13
  %kdma.source1274 = or disjoint i64 %110, %kdma.source1273, !dbg !13
  %kdma.source1275 = trunc nuw i64 %kdma.source1274 to i32, !dbg !13
  %kdma.source1276 = shl nuw nsw i32 %kdma.source1275, 1, !dbg !13
  %kdma.offset1277 = or disjoint i32 %kdma.source1276, %kdma.intra, !dbg !13
  %kdma.source1290 = or disjoint i64 %kdma.source1256, 16, !dbg !13
  %kdma.source1291 = or disjoint i64 %110, %kdma.source1290, !dbg !13
  %kdma.source1292 = trunc nuw i64 %kdma.source1291 to i32, !dbg !13
  %kdma.source1293 = shl nuw nsw i32 %kdma.source1292, 1, !dbg !13
  %kdma.offset1294 = or disjoint i32 %kdma.source1293, %kdma.intra.hi, !dbg !13
  %kdma.lds1295 = add i32 %43, 512, !dbg !13
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %114 = or disjoint i64 %110, 32, !dbg !13
  %kdma.source1307 = or disjoint i64 %114, %kdma.source1242, !dbg !13
  %kdma.source1308 = trunc nuw i64 %kdma.source1307 to i32, !dbg !13
  %kdma.source1309 = shl nuw nsw i32 %kdma.source1308, 1, !dbg !13
  %kdma.offset1310 = or disjoint i32 %kdma.source1309, %kdma.intra, !dbg !13
  %115 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %48, i32 %kdma.offset1310, <4 x i32> %dma.buffer.rsrc1112) #10, !dbg !13
  %kdma.source1323 = or disjoint i64 %114, %kdma.source1256, !dbg !13
  %kdma.source1324 = trunc nuw i64 %kdma.source1323 to i32, !dbg !13
  %kdma.source1325 = shl nuw nsw i32 %kdma.source1324, 1, !dbg !13
  %kdma.offset1326 = or disjoint i32 %kdma.source1325, %kdma.intra.hi, !dbg !13
  %kdma.lds1327 = add i32 %48, 512, !dbg !13
  %116 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1327, i32 %kdma.offset1326, <4 x i32> %dma.buffer.rsrc1112) #10, !dbg !13
  %kdma.source1340 = or disjoint i64 %114, %kdma.source1273, !dbg !13
  %kdma.source1341 = trunc nuw i64 %kdma.source1340 to i32, !dbg !13
  %kdma.source1342 = shl nuw nsw i32 %kdma.source1341, 1, !dbg !13
  %kdma.offset1343 = or disjoint i32 %kdma.source1342, %kdma.intra, !dbg !13
  %117 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %52, i32 %kdma.offset1343, <4 x i32> %dma.buffer.rsrc1112) #10, !dbg !13
  %kdma.source1357 = or disjoint i64 %114, %kdma.source1290, !dbg !13
  %kdma.source1358 = trunc nuw i64 %kdma.source1357 to i32, !dbg !13
  %kdma.source1359 = shl nuw nsw i32 %kdma.source1358, 1, !dbg !13
  %kdma.offset1360 = or disjoint i32 %kdma.source1359, %kdma.intra.hi, !dbg !13
  %kdma.lds1361 = add i32 %52, 512, !dbg !13
  %118 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1361, i32 %kdma.offset1360, <4 x i32> %dma.buffer.rsrc1112) #10, !dbg !13
  %address.local472 = lshr i32 %address.thread445, 2, !dbg !15
  %address.local473 = and i32 %address.local472, 4, !dbg !15
  %address.local474 = shl nuw nsw i32 %address.thread445, 1, !dbg !15
  %address.local475 = and i32 %address.local474, 24, !dbg !15
  %address.local476 = or disjoint i32 %address.local473, %address.local475, !dbg !15
  %address.local477 = shl nuw nsw i32 %address.thread445, 5, !dbg !15
  %address.local478 = and i32 %address.local477, 1120, !dbg !15
  %address.local479 = or disjoint i32 %address.local476, %address.local478, !dbg !15
  %address.local480 = zext nneg i32 %address.local479 to i64, !dbg !15
  %address.local481 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local480, !dbg !15
  %119 = load <4 x half>, ptr addrspace(3) %address.local481, align 8, !dbg !15, !frisk.k_lds_pair !35
  %kdma.gep1500 = getelementptr i8, ptr addrspace(3) %address.local481, i64 1024, !dbg !15
  %120 = load <4 x half>, ptr addrspace(3) %kdma.gep1500, align 8, !dbg !15, !frisk.k_lds_pair !35
  %kdma.gep = getelementptr i8, ptr addrspace(3) %address.local481, i64 4096, !dbg !15
  %121 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8, !dbg !15, !frisk.k_lds_pair !36
  %kdma.gep1501 = getelementptr i8, ptr addrspace(3) %address.local481, i64 5120, !dbg !15
  %122 = load <4 x half>, ptr addrspace(3) %kdma.gep1501, align 8, !dbg !15, !frisk.k_lds_pair !36
  %address.local499 = and i32 %address.local477, 1088, !dbg !15
  %address.local500 = or disjoint i32 %address.local476, %address.local499, !dbg !15
  %address.local501 = or disjoint i32 %address.local500, 160, !dbg !15
  %address.local503 = and i32 %address.local477, 32, !dbg !15
  %address.local504 = sub nuw nsw i32 %address.local501, %address.local503, !dbg !15
  %address.local505 = zext nneg i32 %address.local504 to i64, !dbg !15
  %address.local506 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local505, !dbg !15
  %address.local506.idx = shl nuw nsw i64 %address.local505, 1, !dbg !15
  %123 = lshr i64 %address.local505, 5, !dbg !15
  %124 = and i64 %123, 8, !dbg !15
  %125 = xor i64 %address.local506.idx, %124, !dbg !15
  %126 = sub nsw i64 %125, %address.local506.idx, !dbg !15
  %kdma.swizzled1494 = getelementptr i8, ptr addrspace(3) %address.local506, i64 %126, !dbg !15
  %127 = load <4 x half>, ptr addrspace(3) %kdma.swizzled1494, align 8, !dbg !15, !frisk.k_lds_pair !37
  %kdma.gep1502 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1494, i64 1024, !dbg !15
  %128 = load <4 x half>, ptr addrspace(3) %kdma.gep1502, align 8, !dbg !15, !frisk.k_lds_pair !37
  %kdma.gep1495 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1494, i64 4096, !dbg !15
  %129 = load <4 x half>, ptr addrspace(3) %kdma.gep1495, align 8, !dbg !15, !frisk.k_lds_pair !38
  %kdma.gep1503 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1494, i64 5120, !dbg !15
  %130 = load <4 x half>, ptr addrspace(3) %kdma.gep1503, align 8, !dbg !15, !frisk.k_lds_pair !38
  %lds.const.gep965 = getelementptr i8, ptr addrspace(3) %address.local481, i64 512, !dbg !15
  %address.local481.idx = shl nuw nsw i64 %address.local480, 1, !dbg !15
  %131 = or disjoint i64 %address.local481.idx, 512, !dbg !15
  %132 = xor i64 %address.local481.idx, 520, !dbg !15
  %133 = sub nsw i64 %132, %131, !dbg !15
  %kdma.swizzled1496 = getelementptr i8, ptr addrspace(3) %lds.const.gep965, i64 %133, !dbg !15
  %134 = load <4 x half>, ptr addrspace(3) %kdma.swizzled1496, align 8, !dbg !15, !frisk.k_lds_pair !33
  %kdma.gep1504 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1496, i64 1024, !dbg !15
  %135 = load <4 x half>, ptr addrspace(3) %kdma.gep1504, align 8, !dbg !15, !frisk.k_lds_pair !33
  %kdma.gep1497 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1496, i64 4096, !dbg !15
  %136 = load <4 x half>, ptr addrspace(3) %kdma.gep1497, align 8, !dbg !15, !frisk.k_lds_pair !39
  %kdma.gep1505 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1496, i64 5120, !dbg !15
  %137 = load <4 x half>, ptr addrspace(3) %kdma.gep1505, align 8, !dbg !15, !frisk.k_lds_pair !39
  %lds.const.gep967 = getelementptr i8, ptr addrspace(3) %address.local506, i64 512, !dbg !15
  %138 = add nuw nsw i64 %address.local506.idx, 512, !dbg !15
  %139 = lshr i64 %138, 6, !dbg !15
  %140 = and i64 %139, 8, !dbg !15
  %141 = xor i64 %138, %140, !dbg !15
  %142 = sub nsw i64 %141, %138, !dbg !15
  %kdma.swizzled1498 = getelementptr i8, ptr addrspace(3) %lds.const.gep967, i64 %142, !dbg !15
  %143 = load <4 x half>, ptr addrspace(3) %kdma.swizzled1498, align 8, !dbg !15, !frisk.k_lds_pair !40
  %kdma.gep1506 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1498, i64 1024, !dbg !15
  %144 = load <4 x half>, ptr addrspace(3) %kdma.gep1506, align 8, !dbg !15, !frisk.k_lds_pair !40
  %kdma.gep1499 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1498, i64 4096, !dbg !15
  %145 = load <4 x half>, ptr addrspace(3) %kdma.gep1499, align 8, !dbg !15, !frisk.k_lds_pair !41
  %kdma.gep1507 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1498, i64 5120, !dbg !15
  %146 = load <4 x half>, ptr addrspace(3) %kdma.gep1507, align 8, !dbg !15, !frisk.k_lds_pair !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %147 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %119, <4 x float> zeroinitializer) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %148 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %121, <4 x float> %147) #9, !dbg !15
  %149 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %127, <4 x float> zeroinitializer) #9, !dbg !15
  %150 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %129, <4 x float> %149) #9, !dbg !15
  %151 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %134, <4 x float> zeroinitializer) #9, !dbg !15
  %152 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %136, <4 x float> %151) #9, !dbg !15
  %153 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %143, <4 x float> zeroinitializer) #9, !dbg !15
  %154 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %145, <4 x float> %153) #9, !dbg !15
  %155 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %120, <4 x float> zeroinitializer) #9, !dbg !15
  %156 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %122, <4 x float> %155) #9, !dbg !15
  %157 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %128, <4 x float> zeroinitializer) #9, !dbg !15
  %158 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %130, <4 x float> %157) #9, !dbg !15
  %159 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %135, <4 x float> zeroinitializer) #9, !dbg !15
  %160 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %137, <4 x float> %159) #9, !dbg !15
  %161 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %144, <4 x float> zeroinitializer) #9, !dbg !15
  %162 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %146, <4 x float> %161) #9, !dbg !15
  %163 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %119, <4 x float> zeroinitializer) #9, !dbg !15
  %164 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %121, <4 x float> %163) #9, !dbg !15
  %165 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %127, <4 x float> zeroinitializer) #9, !dbg !15
  %166 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %129, <4 x float> %165) #9, !dbg !15
  %167 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %134, <4 x float> zeroinitializer) #9, !dbg !15
  %168 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %136, <4 x float> %167) #9, !dbg !15
  %169 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %143, <4 x float> zeroinitializer) #9, !dbg !15
  %170 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %145, <4 x float> %169) #9, !dbg !15
  %171 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %120, <4 x float> zeroinitializer) #9, !dbg !15
  %172 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %122, <4 x float> %171) #9, !dbg !15
  %173 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %128, <4 x float> zeroinitializer) #9, !dbg !15
  %174 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %130, <4 x float> %173) #9, !dbg !15
  %175 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %135, <4 x float> zeroinitializer) #9, !dbg !15
  %176 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %137, <4 x float> %175) #9, !dbg !15
  %177 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %144, <4 x float> zeroinitializer) #9, !dbg !15
  %178 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %146, <4 x float> %177) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %179 = or disjoint i64 %110, 64, !dbg !13
  %kdma.source1373 = or disjoint i64 %179, %kdma.source1242, !dbg !13
  %kdma.source1374 = trunc nuw i64 %kdma.source1373 to i32, !dbg !13
  %kdma.source1375 = shl nuw nsw i32 %kdma.source1374, 1, !dbg !13
  %kdma.offset1376 = or disjoint i32 %kdma.source1375, %kdma.intra, !dbg !13
  %180 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %36, i32 %kdma.offset1376, <4 x i32> %dma.buffer.rsrc1112) #10, !dbg !13
  %kdma.source1389 = or disjoint i64 %179, %kdma.source1256, !dbg !13
  %kdma.source1390 = trunc nuw i64 %kdma.source1389 to i32, !dbg !13
  %kdma.source1391 = shl nuw nsw i32 %kdma.source1390, 1, !dbg !13
  %kdma.offset1392 = or disjoint i32 %kdma.source1391, %kdma.intra.hi, !dbg !13
  %181 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1261, i32 %kdma.offset1392, <4 x i32> %dma.buffer.rsrc1112) #10, !dbg !13
  %kdma.source1406 = or disjoint i64 %179, %kdma.source1273, !dbg !13
  %kdma.source1407 = trunc nuw i64 %kdma.source1406 to i32, !dbg !13
  %kdma.source1408 = shl nuw nsw i32 %kdma.source1407, 1, !dbg !13
  %kdma.offset1409 = or disjoint i32 %kdma.source1408, %kdma.intra, !dbg !13
  %182 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %43, i32 %kdma.offset1409, <4 x i32> %dma.buffer.rsrc1112) #10, !dbg !13
  %kdma.source1423 = or disjoint i64 %179, %kdma.source1290, !dbg !13
  %kdma.source1424 = trunc nuw i64 %kdma.source1423 to i32, !dbg !13
  %kdma.source1425 = shl nuw nsw i32 %kdma.source1424, 1, !dbg !13
  %kdma.offset1426 = or disjoint i32 %kdma.source1425, %kdma.intra.hi, !dbg !13
  %183 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1295, i32 %kdma.offset1426, <4 x i32> %dma.buffer.rsrc1112) #10, !dbg !13
  %kdma.gep1524 = getelementptr i8, ptr addrspace(3) %address.local481, i64 8192, !dbg !15
  %184 = load <4 x half>, ptr addrspace(3) %kdma.gep1524, align 8, !dbg !15, !frisk.k_lds_pair !42
  %kdma.gep1532 = getelementptr i8, ptr addrspace(3) %address.local481, i64 9216, !dbg !15
  %185 = load <4 x half>, ptr addrspace(3) %kdma.gep1532, align 8, !dbg !15, !frisk.k_lds_pair !42
  %kdma.gep1525 = getelementptr i8, ptr addrspace(3) %address.local481, i64 12288, !dbg !15
  %186 = load <4 x half>, ptr addrspace(3) %kdma.gep1525, align 8, !dbg !15, !frisk.k_lds_pair !43
  %kdma.gep1533 = getelementptr i8, ptr addrspace(3) %address.local481, i64 13312, !dbg !15
  %187 = load <4 x half>, ptr addrspace(3) %kdma.gep1533, align 8, !dbg !15, !frisk.k_lds_pair !43
  %kdma.gep1526 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1494, i64 8192, !dbg !15
  %188 = load <4 x half>, ptr addrspace(3) %kdma.gep1526, align 8, !dbg !15, !frisk.k_lds_pair !44
  %kdma.gep1534 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1494, i64 9216, !dbg !15
  %189 = load <4 x half>, ptr addrspace(3) %kdma.gep1534, align 8, !dbg !15, !frisk.k_lds_pair !44
  %kdma.gep1527 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1494, i64 12288, !dbg !15
  %190 = load <4 x half>, ptr addrspace(3) %kdma.gep1527, align 8, !dbg !15, !frisk.k_lds_pair !45
  %kdma.gep1535 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1494, i64 13312, !dbg !15
  %191 = load <4 x half>, ptr addrspace(3) %kdma.gep1535, align 8, !dbg !15, !frisk.k_lds_pair !45
  %kdma.gep1528 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1496, i64 8192, !dbg !15
  %192 = load <4 x half>, ptr addrspace(3) %kdma.gep1528, align 8, !dbg !15, !frisk.k_lds_pair !46
  %kdma.gep1536 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1496, i64 9216, !dbg !15
  %193 = load <4 x half>, ptr addrspace(3) %kdma.gep1536, align 8, !dbg !15, !frisk.k_lds_pair !46
  %kdma.gep1529 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1496, i64 12288, !dbg !15
  %194 = load <4 x half>, ptr addrspace(3) %kdma.gep1529, align 8, !dbg !15, !frisk.k_lds_pair !47
  %kdma.gep1537 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1496, i64 13312, !dbg !15
  %195 = load <4 x half>, ptr addrspace(3) %kdma.gep1537, align 8, !dbg !15, !frisk.k_lds_pair !47
  %kdma.gep1530 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1498, i64 8192, !dbg !15
  %196 = load <4 x half>, ptr addrspace(3) %kdma.gep1530, align 8, !dbg !15, !frisk.k_lds_pair !48
  %kdma.gep1538 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1498, i64 9216, !dbg !15
  %197 = load <4 x half>, ptr addrspace(3) %kdma.gep1538, align 8, !dbg !15, !frisk.k_lds_pair !48
  %kdma.gep1531 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1498, i64 12288, !dbg !15
  %198 = load <4 x half>, ptr addrspace(3) %kdma.gep1531, align 8, !dbg !15, !frisk.k_lds_pair !49
  %kdma.gep1539 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1498, i64 13312, !dbg !15
  %199 = load <4 x half>, ptr addrspace(3) %kdma.gep1539, align 8, !dbg !15, !frisk.k_lds_pair !49
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %200 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %184, <4 x float> %148) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %201 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %186, <4 x float> %200) #9, !dbg !15
  %202 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %188, <4 x float> %150) #9, !dbg !15
  %203 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %190, <4 x float> %202) #9, !dbg !15
  %204 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %192, <4 x float> %152) #9, !dbg !15
  %205 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %194, <4 x float> %204) #9, !dbg !15
  %206 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %196, <4 x float> %154) #9, !dbg !15
  %207 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %198, <4 x float> %206) #9, !dbg !15
  %208 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %185, <4 x float> %156) #9, !dbg !15
  %209 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %187, <4 x float> %208) #9, !dbg !15
  %210 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %189, <4 x float> %158) #9, !dbg !15
  %211 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %191, <4 x float> %210) #9, !dbg !15
  %212 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %193, <4 x float> %160) #9, !dbg !15
  %213 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %195, <4 x float> %212) #9, !dbg !15
  %214 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %197, <4 x float> %162) #9, !dbg !15
  %215 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %199, <4 x float> %214) #9, !dbg !15
  %216 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %184, <4 x float> %164) #9, !dbg !15
  %217 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %186, <4 x float> %216) #9, !dbg !15
  %218 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %188, <4 x float> %166) #9, !dbg !15
  %219 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %190, <4 x float> %218) #9, !dbg !15
  %220 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %192, <4 x float> %168) #9, !dbg !15
  %221 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %194, <4 x float> %220) #9, !dbg !15
  %222 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %196, <4 x float> %170) #9, !dbg !15
  %223 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %198, <4 x float> %222) #9, !dbg !15
  %224 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %185, <4 x float> %172) #9, !dbg !15
  %225 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %187, <4 x float> %224) #9, !dbg !15
  %226 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %189, <4 x float> %174) #9, !dbg !15
  %227 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %191, <4 x float> %226) #9, !dbg !15
  %228 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %193, <4 x float> %176) #9, !dbg !15
  %229 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %195, <4 x float> %228) #9, !dbg !15
  %230 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %197, <4 x float> %178) #9, !dbg !15
  %231 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %199, <4 x float> %230) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %232 = or disjoint i64 %110, 96, !dbg !13
  %kdma.source1439 = or disjoint i64 %232, %kdma.source1242, !dbg !13
  %kdma.source1440 = trunc nuw i64 %kdma.source1439 to i32, !dbg !13
  %kdma.source1441 = shl nuw nsw i32 %kdma.source1440, 1, !dbg !13
  %kdma.offset1442 = or disjoint i32 %kdma.source1441, %kdma.intra, !dbg !13
  %233 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %48, i32 %kdma.offset1442, <4 x i32> %dma.buffer.rsrc1112) #10, !dbg !13
  %kdma.source1455 = or disjoint i64 %232, %kdma.source1256, !dbg !13
  %kdma.source1456 = trunc nuw i64 %kdma.source1455 to i32, !dbg !13
  %kdma.source1457 = shl nuw nsw i32 %kdma.source1456, 1, !dbg !13
  %kdma.offset1458 = or disjoint i32 %kdma.source1457, %kdma.intra.hi, !dbg !13
  %234 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1327, i32 %kdma.offset1458, <4 x i32> %dma.buffer.rsrc1112) #10, !dbg !13
  %kdma.source1472 = or disjoint i64 %232, %kdma.source1273, !dbg !13
  %kdma.source1473 = trunc nuw i64 %kdma.source1472 to i32, !dbg !13
  %kdma.source1474 = shl nuw nsw i32 %kdma.source1473, 1, !dbg !13
  %kdma.offset1475 = or disjoint i32 %kdma.source1474, %kdma.intra, !dbg !13
  %235 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %52, i32 %kdma.offset1475, <4 x i32> %dma.buffer.rsrc1112) #10, !dbg !13
  %kdma.source1489 = or disjoint i64 %232, %kdma.source1290, !dbg !13
  %kdma.source1490 = trunc nuw i64 %kdma.source1489 to i32, !dbg !13
  %kdma.source1491 = shl nuw nsw i32 %kdma.source1490, 1, !dbg !13
  %kdma.offset1492 = or disjoint i32 %kdma.source1491, %kdma.intra.hi, !dbg !13
  %236 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1361, i32 %kdma.offset1492, <4 x i32> %dma.buffer.rsrc1112) #10, !dbg !13
  %237 = load <4 x half>, ptr addrspace(3) %address.local481, align 8, !dbg !15, !frisk.k_lds_pair !50
  %238 = load <4 x half>, ptr addrspace(3) %kdma.gep1500, align 8, !dbg !15, !frisk.k_lds_pair !50
  %239 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8, !dbg !15, !frisk.k_lds_pair !51
  %240 = load <4 x half>, ptr addrspace(3) %kdma.gep1501, align 8, !dbg !15, !frisk.k_lds_pair !51
  %241 = load <4 x half>, ptr addrspace(3) %kdma.swizzled1494, align 8, !dbg !15, !frisk.k_lds_pair !52
  %242 = load <4 x half>, ptr addrspace(3) %kdma.gep1502, align 8, !dbg !15, !frisk.k_lds_pair !52
  %243 = load <4 x half>, ptr addrspace(3) %kdma.gep1495, align 8, !dbg !15, !frisk.k_lds_pair !53
  %244 = load <4 x half>, ptr addrspace(3) %kdma.gep1503, align 8, !dbg !15, !frisk.k_lds_pair !53
  %245 = load <4 x half>, ptr addrspace(3) %kdma.swizzled1496, align 8, !dbg !15, !frisk.k_lds_pair !54
  %246 = load <4 x half>, ptr addrspace(3) %kdma.gep1504, align 8, !dbg !15, !frisk.k_lds_pair !54
  %247 = load <4 x half>, ptr addrspace(3) %kdma.gep1497, align 8, !dbg !15, !frisk.k_lds_pair !55
  %248 = load <4 x half>, ptr addrspace(3) %kdma.gep1505, align 8, !dbg !15, !frisk.k_lds_pair !55
  %249 = load <4 x half>, ptr addrspace(3) %kdma.swizzled1498, align 8, !dbg !15, !frisk.k_lds_pair !56
  %250 = load <4 x half>, ptr addrspace(3) %kdma.gep1506, align 8, !dbg !15, !frisk.k_lds_pair !56
  %251 = load <4 x half>, ptr addrspace(3) %kdma.gep1499, align 8, !dbg !15, !frisk.k_lds_pair !57
  %252 = load <4 x half>, ptr addrspace(3) %kdma.gep1507, align 8, !dbg !15, !frisk.k_lds_pair !57
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %253 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %237, <4 x float> %201) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %254 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %239, <4 x float> %253) #9, !dbg !15
  %255 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %241, <4 x float> %203) #9, !dbg !15
  %256 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %243, <4 x float> %255) #9, !dbg !15
  %257 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %245, <4 x float> %205) #9, !dbg !15
  %258 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %247, <4 x float> %257) #9, !dbg !15
  %259 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %249, <4 x float> %207) #9, !dbg !15
  %260 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %251, <4 x float> %259) #9, !dbg !15
  %261 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %238, <4 x float> %209) #9, !dbg !15
  %262 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %240, <4 x float> %261) #9, !dbg !15
  %263 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %242, <4 x float> %211) #9, !dbg !15
  %264 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %244, <4 x float> %263) #9, !dbg !15
  %265 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %246, <4 x float> %213) #9, !dbg !15
  %266 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %248, <4 x float> %265) #9, !dbg !15
  %267 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %250, <4 x float> %215) #9, !dbg !15
  %268 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %252, <4 x float> %267) #9, !dbg !15
  %269 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %237, <4 x float> %217) #9, !dbg !15
  %270 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %239, <4 x float> %269) #9, !dbg !15
  %271 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %241, <4 x float> %219) #9, !dbg !15
  %272 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %243, <4 x float> %271) #9, !dbg !15
  %273 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %245, <4 x float> %221) #9, !dbg !15
  %274 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %247, <4 x float> %273) #9, !dbg !15
  %275 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %249, <4 x float> %223) #9, !dbg !15
  %276 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %251, <4 x float> %275) #9, !dbg !15
  %277 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %238, <4 x float> %225) #9, !dbg !15
  %278 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %240, <4 x float> %277) #9, !dbg !15
  %279 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %242, <4 x float> %227) #9, !dbg !15
  %280 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %244, <4 x float> %279) #9, !dbg !15
  %281 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %246, <4 x float> %229) #9, !dbg !15
  %282 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %248, <4 x float> %281) #9, !dbg !15
  %283 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %250, <4 x float> %231) #9, !dbg !15
  %284 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %252, <4 x float> %283) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %address.local691 = shl nuw nsw i64 %address.local446, 5, !dbg !16
  %address.local692 = and i64 %address.local691, 8064, !dbg !16
  %address.local693 = shl nuw nsw i64 %address.local446, 3, !dbg !16
  %address.local694 = and i64 %address.local693, 24, !dbg !16
  %address.local695 = or disjoint i64 %address.local692, %address.local694, !dbg !16
  %285 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !16
  %.idx11581159 = or disjoint i64 %110, %address.local695, !dbg !16
  %.idx11581159.tr = trunc nuw i64 %.idx11581159 to i32, !dbg !16
  %dma.buffer.offset1095 = shl nuw nsw i32 %.idx11581159.tr, 1, !dbg !16
  %286 = insertelement <2 x i64> poison, i64 %285, i64 0, !dbg !16
  %287 = bitcast <2 x i64> %286 to <4 x i32>, !dbg !16
  %dma.buffer.rsrc1096 = shufflevector <4 x i32> %287, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !16
  %288 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %36, i32 %dma.buffer.offset1095, <4 x i32> %dma.buffer.rsrc1096) #10, !dbg !16
  %address.local701 = or disjoint i64 %address.local695, 8192, !dbg !16
  %.idx11601161 = or disjoint i64 %110, %address.local701, !dbg !16
  %.idx11601161.tr = trunc nuw i64 %.idx11601161 to i32, !dbg !16
  %dma.buffer.offset1093 = shl nuw nsw i32 %.idx11601161.tr, 1, !dbg !16
  %289 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %43, i32 %dma.buffer.offset1093, <4 x i32> %dma.buffer.rsrc1096) #10, !dbg !16
  %290 = load <4 x half>, ptr addrspace(3) %kdma.gep1524, align 8, !dbg !15, !frisk.k_tail_pair !35
  %291 = load <4 x half>, ptr addrspace(3) %kdma.gep1532, align 8, !dbg !15, !frisk.k_tail_pair !35
  %292 = load <4 x half>, ptr addrspace(3) %kdma.gep1525, align 8, !dbg !15, !frisk.k_tail_pair !36
  %293 = load <4 x half>, ptr addrspace(3) %kdma.gep1533, align 8, !dbg !15, !frisk.k_tail_pair !36
  %294 = load <4 x half>, ptr addrspace(3) %kdma.gep1526, align 8, !dbg !15, !frisk.k_tail_pair !37
  %295 = load <4 x half>, ptr addrspace(3) %kdma.gep1534, align 8, !dbg !15, !frisk.k_tail_pair !37
  %296 = load <4 x half>, ptr addrspace(3) %kdma.gep1527, align 8, !dbg !15, !frisk.k_tail_pair !38
  %297 = load <4 x half>, ptr addrspace(3) %kdma.gep1535, align 8, !dbg !15, !frisk.k_tail_pair !38
  %298 = load <4 x half>, ptr addrspace(3) %kdma.gep1528, align 8, !dbg !15, !frisk.k_tail_pair !33
  %299 = load <4 x half>, ptr addrspace(3) %kdma.gep1536, align 8, !dbg !15, !frisk.k_tail_pair !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_tail_prefetch !1
  %300 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %290, <4 x float> %254) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_tail_prefetch !1
  %301 = load <4 x half>, ptr addrspace(3) %kdma.gep1529, align 8, !dbg !15, !frisk.k_tail_pair !39
  %302 = load <4 x half>, ptr addrspace(3) %kdma.gep1537, align 8, !dbg !15, !frisk.k_tail_pair !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_tail_prefetch !1
  %303 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %292, <4 x float> %300) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_tail_prefetch !1
  %304 = load <4 x half>, ptr addrspace(3) %kdma.gep1530, align 8, !dbg !15, !frisk.k_tail_pair !40
  %305 = load <4 x half>, ptr addrspace(3) %kdma.gep1538, align 8, !dbg !15, !frisk.k_tail_pair !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_tail_prefetch !1
  %306 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %294, <4 x float> %256) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !20, !frisk.k_tail_prefetch !1
  %307 = extractelement <4 x float> %303, i64 0, !dbg !20
  %308 = call float @llvm.maxnum.f32(float %307, float 0xFFF0000000000000), !dbg !6, !frisk.local_reduction !1
  %309 = extractelement <4 x float> %303, i64 1, !dbg !20
  %310 = extractelement <4 x float> %303, i64 2, !dbg !20
  %311 = extractelement <4 x float> %303, i64 3, !dbg !20
  %312 = load <4 x half>, ptr addrspace(3) %kdma.gep1531, align 8, !dbg !15, !frisk.k_tail_pair !41
  %313 = load <4 x half>, ptr addrspace(3) %kdma.gep1539, align 8, !dbg !15, !frisk.k_tail_pair !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_tail_prefetch !1
  %314 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %296, <4 x float> %306) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_tail_prefetch !1
  %315 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %298, <4 x float> %258) #9, !dbg !15
  %316 = extractelement <4 x float> %314, i64 0, !dbg !20
  %local.reduce133 = call float @llvm.maxnum.f32(float %308, float %316), !dbg !6, !frisk.local_reduction !1
  %317 = extractelement <4 x float> %314, i64 1, !dbg !20
  %local.reduce134 = call float @llvm.maxnum.f32(float %309, float %317), !dbg !6, !frisk.local_reduction !1
  %318 = extractelement <4 x float> %314, i64 2, !dbg !20
  %local.reduce135 = call float @llvm.maxnum.f32(float %310, float %318), !dbg !6, !frisk.local_reduction !1
  %319 = extractelement <4 x float> %314, i64 3, !dbg !20
  %local.reduce136 = call float @llvm.maxnum.f32(float %311, float %319), !dbg !6, !frisk.local_reduction !1
  %320 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %301, <4 x float> %315) #9, !dbg !15
  %321 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %304, <4 x float> %260) #9, !dbg !15
  %322 = extractelement <4 x float> %320, i64 0, !dbg !20
  %local.reduce137 = call float @llvm.maxnum.f32(float %local.reduce133, float %322), !dbg !6, !frisk.local_reduction !1
  %323 = extractelement <4 x float> %320, i64 1, !dbg !20
  %local.reduce138 = call float @llvm.maxnum.f32(float %local.reduce134, float %323), !dbg !6, !frisk.local_reduction !1
  %324 = extractelement <4 x float> %320, i64 2, !dbg !20
  %local.reduce139 = call float @llvm.maxnum.f32(float %local.reduce135, float %324), !dbg !6, !frisk.local_reduction !1
  %325 = extractelement <4 x float> %320, i64 3, !dbg !20
  %local.reduce140 = call float @llvm.maxnum.f32(float %local.reduce136, float %325), !dbg !6, !frisk.local_reduction !1
  %326 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %312, <4 x float> %321) #9, !dbg !15
  %327 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %291, <4 x float> %262) #9, !dbg !15
  %328 = extractelement <4 x float> %326, i64 0, !dbg !20
  %local.reduce141 = call float @llvm.maxnum.f32(float %local.reduce137, float %328), !dbg !6, !frisk.local_reduction !1
  %329 = extractelement <4 x float> %326, i64 1, !dbg !20
  %local.reduce142 = call float @llvm.maxnum.f32(float %local.reduce138, float %329), !dbg !6, !frisk.local_reduction !1
  %330 = extractelement <4 x float> %326, i64 2, !dbg !20
  %local.reduce143 = call float @llvm.maxnum.f32(float %local.reduce139, float %330), !dbg !6, !frisk.local_reduction !1
  %331 = extractelement <4 x float> %326, i64 3, !dbg !20
  %local.reduce144 = call float @llvm.maxnum.f32(float %local.reduce140, float %331), !dbg !6, !frisk.local_reduction !1
  %332 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %293, <4 x float> %327) #9, !dbg !15
  %333 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %295, <4 x float> %264) #9, !dbg !15
  %334 = extractelement <4 x float> %332, i64 0, !dbg !20
  %local.reduce145 = call float @llvm.maxnum.f32(float %local.reduce141, float %334), !dbg !6, !frisk.local_reduction !1
  %335 = extractelement <4 x float> %332, i64 1, !dbg !20
  %local.reduce146 = call float @llvm.maxnum.f32(float %local.reduce142, float %335), !dbg !6, !frisk.local_reduction !1
  %336 = extractelement <4 x float> %332, i64 2, !dbg !20
  %local.reduce147 = call float @llvm.maxnum.f32(float %local.reduce143, float %336), !dbg !6, !frisk.local_reduction !1
  %337 = extractelement <4 x float> %332, i64 3, !dbg !20
  %local.reduce148 = call float @llvm.maxnum.f32(float %local.reduce144, float %337), !dbg !6, !frisk.local_reduction !1
  %338 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %297, <4 x float> %333) #9, !dbg !15
  %339 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %299, <4 x float> %266) #9, !dbg !15
  %340 = extractelement <4 x float> %338, i64 0, !dbg !20
  %local.reduce149 = call float @llvm.maxnum.f32(float %local.reduce145, float %340), !dbg !6, !frisk.local_reduction !1
  %341 = extractelement <4 x float> %338, i64 1, !dbg !20
  %local.reduce150 = call float @llvm.maxnum.f32(float %local.reduce146, float %341), !dbg !6, !frisk.local_reduction !1
  %342 = extractelement <4 x float> %338, i64 2, !dbg !20
  %local.reduce151 = call float @llvm.maxnum.f32(float %local.reduce147, float %342), !dbg !6, !frisk.local_reduction !1
  %343 = extractelement <4 x float> %338, i64 3, !dbg !20
  %local.reduce152 = call float @llvm.maxnum.f32(float %local.reduce148, float %343), !dbg !6, !frisk.local_reduction !1
  %344 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %302, <4 x float> %339) #9, !dbg !15
  %345 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %305, <4 x float> %268) #9, !dbg !15
  %346 = extractelement <4 x float> %344, i64 0, !dbg !20
  %local.reduce153 = call float @llvm.maxnum.f32(float %local.reduce149, float %346), !dbg !6, !frisk.local_reduction !1
  %347 = extractelement <4 x float> %344, i64 1, !dbg !20
  %local.reduce154 = call float @llvm.maxnum.f32(float %local.reduce150, float %347), !dbg !6, !frisk.local_reduction !1
  %348 = extractelement <4 x float> %344, i64 2, !dbg !20
  %local.reduce155 = call float @llvm.maxnum.f32(float %local.reduce151, float %348), !dbg !6, !frisk.local_reduction !1
  %349 = extractelement <4 x float> %344, i64 3, !dbg !20
  %local.reduce156 = call float @llvm.maxnum.f32(float %local.reduce152, float %349), !dbg !6, !frisk.local_reduction !1
  %350 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %313, <4 x float> %345) #9, !dbg !15
  %351 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %290, <4 x float> %270) #9, !dbg !15
  %352 = extractelement <4 x float> %350, i64 0, !dbg !20
  %local.reduce157 = call float @llvm.maxnum.f32(float %local.reduce153, float %352), !dbg !6, !frisk.local_reduction !1
  %353 = extractelement <4 x float> %350, i64 1, !dbg !20
  %local.reduce158 = call float @llvm.maxnum.f32(float %local.reduce154, float %353), !dbg !6, !frisk.local_reduction !1
  %354 = extractelement <4 x float> %350, i64 2, !dbg !20
  %local.reduce159 = call float @llvm.maxnum.f32(float %local.reduce155, float %354), !dbg !6, !frisk.local_reduction !1
  %355 = extractelement <4 x float> %350, i64 3, !dbg !20
  %local.reduce160 = call float @llvm.maxnum.f32(float %local.reduce156, float %355), !dbg !6, !frisk.local_reduction !1
  %local.reduce161 = call float @llvm.maxnum.f32(float %local.reduce157, float %local.reduce158), !dbg !6, !frisk.local_reduction !1
  %local.reduce162 = call float @llvm.maxnum.f32(float %local.reduce159, float %local.reduce160), !dbg !6, !frisk.local_reduction !1
  %356 = call float @llvm.maxnum.f32(float %local.reduce161, float %local.reduce162), !dbg !6, !frisk.local_reduction !1
  %357 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %292, <4 x float> %351) #9, !dbg !15
  %358 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %294, <4 x float> %272) #9, !dbg !15
  %359 = extractelement <4 x float> %357, i64 0, !dbg !20
  %360 = extractelement <4 x float> %357, i64 1, !dbg !20
  %361 = extractelement <4 x float> %357, i64 2, !dbg !20
  %362 = extractelement <4 x float> %357, i64 3, !dbg !20
  %363 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %296, <4 x float> %358) #9, !dbg !15
  %364 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %298, <4 x float> %274) #9, !dbg !15
  %365 = extractelement <4 x float> %363, i64 0, !dbg !20
  %366 = extractelement <4 x float> %363, i64 1, !dbg !20
  %367 = extractelement <4 x float> %363, i64 2, !dbg !20
  %368 = extractelement <4 x float> %363, i64 3, !dbg !20
  %369 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %301, <4 x float> %364) #9, !dbg !15
  %370 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %304, <4 x float> %276) #9, !dbg !15
  %371 = extractelement <4 x float> %369, i64 0, !dbg !20
  %372 = extractelement <4 x float> %369, i64 1, !dbg !20
  %373 = extractelement <4 x float> %369, i64 2, !dbg !20
  %374 = extractelement <4 x float> %369, i64 3, !dbg !20
  %375 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %312, <4 x float> %370) #9, !dbg !15
  %376 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %291, <4 x float> %278) #9, !dbg !15
  %377 = extractelement <4 x float> %375, i64 0, !dbg !20
  %378 = extractelement <4 x float> %375, i64 1, !dbg !20
  %379 = extractelement <4 x float> %375, i64 2, !dbg !20
  %380 = extractelement <4 x float> %375, i64 3, !dbg !20
  %381 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %293, <4 x float> %376) #9, !dbg !15
  %382 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %295, <4 x float> %280) #9, !dbg !15
  %383 = extractelement <4 x float> %381, i64 0, !dbg !20
  %384 = extractelement <4 x float> %381, i64 1, !dbg !20
  %385 = extractelement <4 x float> %381, i64 2, !dbg !20
  %386 = extractelement <4 x float> %381, i64 3, !dbg !20
  %387 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %297, <4 x float> %382) #9, !dbg !15
  %388 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %299, <4 x float> %282) #9, !dbg !15
  %389 = extractelement <4 x float> %387, i64 0, !dbg !20
  %390 = extractelement <4 x float> %387, i64 1, !dbg !20
  %391 = extractelement <4 x float> %387, i64 2, !dbg !20
  %392 = extractelement <4 x float> %387, i64 3, !dbg !20
  %393 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %302, <4 x float> %388) #9, !dbg !15
  %394 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %305, <4 x float> %284) #9, !dbg !15
  %395 = extractelement <4 x float> %393, i64 0, !dbg !20
  %396 = extractelement <4 x float> %393, i64 1, !dbg !20
  %397 = extractelement <4 x float> %393, i64 2, !dbg !20
  %398 = extractelement <4 x float> %393, i64 3, !dbg !20
  %399 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %313, <4 x float> %394) #9, !dbg !15
  %400 = extractelement <4 x float> %399, i64 0, !dbg !20
  %401 = extractelement <4 x float> %399, i64 1, !dbg !20
  %402 = extractelement <4 x float> %399, i64 2, !dbg !20
  %403 = extractelement <4 x float> %399, i64 3, !dbg !20
  %404 = bitcast float %356 to i32, !dbg !6
  %405 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %404), !dbg !6
  %406 = bitcast i32 %405 to float, !dbg !6
  %407 = call float @llvm.maxnum.f32(float %356, float %406), !dbg !6
  %408 = bitcast float %407 to i32, !dbg !6
  %409 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %408), !dbg !6
  %410 = bitcast i32 %409 to float, !dbg !6
  %411 = call float @llvm.maxnum.f32(float %407, float %410), !dbg !6
  %412 = call float @llvm.maxnum.f32(float %359, float 0xFFF0000000000000), !dbg !6, !frisk.local_reduction !1
  %local.reduce102 = call float @llvm.maxnum.f32(float %412, float %365), !dbg !6, !frisk.local_reduction !1
  %local.reduce103 = call float @llvm.maxnum.f32(float %360, float %366), !dbg !6, !frisk.local_reduction !1
  %local.reduce104 = call float @llvm.maxnum.f32(float %361, float %367), !dbg !6, !frisk.local_reduction !1
  %local.reduce105 = call float @llvm.maxnum.f32(float %362, float %368), !dbg !6, !frisk.local_reduction !1
  %local.reduce106 = call float @llvm.maxnum.f32(float %local.reduce102, float %371), !dbg !6, !frisk.local_reduction !1
  %local.reduce107 = call float @llvm.maxnum.f32(float %local.reduce103, float %372), !dbg !6, !frisk.local_reduction !1
  %local.reduce108 = call float @llvm.maxnum.f32(float %local.reduce104, float %373), !dbg !6, !frisk.local_reduction !1
  %local.reduce109 = call float @llvm.maxnum.f32(float %local.reduce105, float %374), !dbg !6, !frisk.local_reduction !1
  %local.reduce110 = call float @llvm.maxnum.f32(float %local.reduce106, float %377), !dbg !6, !frisk.local_reduction !1
  %local.reduce111 = call float @llvm.maxnum.f32(float %local.reduce107, float %378), !dbg !6, !frisk.local_reduction !1
  %local.reduce112 = call float @llvm.maxnum.f32(float %local.reduce108, float %379), !dbg !6, !frisk.local_reduction !1
  %local.reduce113 = call float @llvm.maxnum.f32(float %local.reduce109, float %380), !dbg !6, !frisk.local_reduction !1
  %local.reduce114 = call float @llvm.maxnum.f32(float %local.reduce110, float %383), !dbg !6, !frisk.local_reduction !1
  %local.reduce115 = call float @llvm.maxnum.f32(float %local.reduce111, float %384), !dbg !6, !frisk.local_reduction !1
  %local.reduce116 = call float @llvm.maxnum.f32(float %local.reduce112, float %385), !dbg !6, !frisk.local_reduction !1
  %local.reduce117 = call float @llvm.maxnum.f32(float %local.reduce113, float %386), !dbg !6, !frisk.local_reduction !1
  %local.reduce118 = call float @llvm.maxnum.f32(float %local.reduce114, float %389), !dbg !6, !frisk.local_reduction !1
  %local.reduce119 = call float @llvm.maxnum.f32(float %local.reduce115, float %390), !dbg !6, !frisk.local_reduction !1
  %local.reduce120 = call float @llvm.maxnum.f32(float %local.reduce116, float %391), !dbg !6, !frisk.local_reduction !1
  %local.reduce121 = call float @llvm.maxnum.f32(float %local.reduce117, float %392), !dbg !6, !frisk.local_reduction !1
  %local.reduce122 = call float @llvm.maxnum.f32(float %local.reduce118, float %395), !dbg !6, !frisk.local_reduction !1
  %local.reduce123 = call float @llvm.maxnum.f32(float %local.reduce119, float %396), !dbg !6, !frisk.local_reduction !1
  %local.reduce124 = call float @llvm.maxnum.f32(float %local.reduce120, float %397), !dbg !6, !frisk.local_reduction !1
  %local.reduce125 = call float @llvm.maxnum.f32(float %local.reduce121, float %398), !dbg !6, !frisk.local_reduction !1
  %local.reduce126 = call float @llvm.maxnum.f32(float %local.reduce122, float %400), !dbg !6, !frisk.local_reduction !1
  %local.reduce127 = call float @llvm.maxnum.f32(float %local.reduce123, float %401), !dbg !6, !frisk.local_reduction !1
  %local.reduce128 = call float @llvm.maxnum.f32(float %local.reduce124, float %402), !dbg !6, !frisk.local_reduction !1
  %local.reduce129 = call float @llvm.maxnum.f32(float %local.reduce125, float %403), !dbg !6, !frisk.local_reduction !1
  %local.reduce130 = call float @llvm.maxnum.f32(float %local.reduce126, float %local.reduce127), !dbg !6, !frisk.local_reduction !1
  %local.reduce131 = call float @llvm.maxnum.f32(float %local.reduce128, float %local.reduce129), !dbg !6, !frisk.local_reduction !1
  %413 = call float @llvm.maxnum.f32(float %local.reduce130, float %local.reduce131), !dbg !6, !frisk.local_reduction !1
  %414 = bitcast float %413 to i32, !dbg !6
  %415 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %414), !dbg !6
  %416 = bitcast i32 %415 to float, !dbg !6
  %417 = call float @llvm.maxnum.f32(float %413, float %416), !dbg !6
  %418 = bitcast float %417 to i32, !dbg !6
  %419 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %418), !dbg !6
  %420 = bitcast i32 %419 to float, !dbg !6
  %421 = call float @llvm.maxnum.f32(float %417, float %420), !dbg !6
  %422 = extractvalue [2 x <1 x float>] %101, 0, !dbg !6
  %423 = insertelement <1 x float> poison, float %411, i64 0, !dbg !6
  %424 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %423, <1 x float> %422), !dbg !6
  %425 = extractvalue [2 x <1 x float>] %101, 1, !dbg !6
  %426 = insertelement <1 x float> poison, float %421, i64 0, !dbg !6
  %427 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %426, <1 x float> %425), !dbg !6
  %428 = shufflevector <1 x float> %424, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !21
  %429 = shufflevector <1 x float> %427, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !21
  %430 = fsub <1 x float> %422, %424, !dbg !22
  %431 = extractelement <1 x float> %430, i64 0, !dbg !23
  %432 = call float @llvm.amdgcn.exp2.f32(float %431), !dbg !23
  %433 = insertelement <4 x float> poison, float %432, i64 0, !dbg !28
  %434 = shufflevector <4 x float> %433, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !28
  %435 = extractvalue [2 x <1 x float>] %100, 0, !dbg !24
  %436 = extractelement <1 x float> %435, i64 0, !dbg !24
  %437 = fmul float %436, %432, !dbg !24
  %438 = fsub <1 x float> %425, %427, !dbg !22
  %439 = extractelement <1 x float> %438, i64 0, !dbg !23
  %440 = call float @llvm.amdgcn.exp2.f32(float %439), !dbg !23
  %441 = insertelement <4 x float> poison, float %440, i64 0, !dbg !28
  %442 = shufflevector <4 x float> %441, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !28
  %443 = extractvalue [2 x <1 x float>] %100, 1, !dbg !24
  %444 = extractelement <1 x float> %443, i64 0, !dbg !24
  %445 = fmul float %444, %440, !dbg !24
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %vdb.address = or disjoint i64 %114, %address.local695, !dbg !16
  %vdb.address1174 = trunc nuw i64 %vdb.address to i32, !dbg !16
  %vdb.address1175 = shl nuw nsw i32 %vdb.address1174, 1, !dbg !16
  %vdb.address1180 = or disjoint i64 %114, %address.local701, !dbg !16
  %vdb.address1181 = trunc nuw i64 %vdb.address1180 to i32, !dbg !16
  %vdb.address1182 = shl nuw nsw i32 %vdb.address1181, 1, !dbg !16
  %vdb.dma.base = add i32 %36, 8192, !dbg !14
  %446 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1175, <4 x i32> %dma.buffer.rsrc1096) #10, !dbg !16
  %vdb.dma.base1187 = add i32 %43, 8192, !dbg !14
  %447 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1187, i32 %vdb.address1182, <4 x i32> %dma.buffer.rsrc1096) #10, !dbg !16
  %address.local703 = shl nuw nsw i32 %address.thread445, 3, !dbg !14
  %address.local704 = and i32 %address.local703, 504, !dbg !14
  %address.local705 = zext nneg i32 %address.local704 to i64, !dbg !14
  %address.local706 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local705, !dbg !14
  %448 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local706, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %449 = bitcast <4 x i32> %448 to <8 x half>, !dbg !14
  %address.local708 = getelementptr i8, ptr addrspace(3) %address.local706, i64 1024, !dbg !14
  %450 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local708, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %451 = bitcast <4 x i32> %450 to <8 x half>, !dbg !14
  %address.local710 = getelementptr i8, ptr addrspace(3) %address.local706, i64 2048, !dbg !14
  %452 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local710, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %453 = bitcast <4 x i32> %452 to <8 x half>, !dbg !14
  %address.local712 = getelementptr i8, ptr addrspace(3) %address.local706, i64 3072, !dbg !14
  %454 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local712, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %455 = bitcast <4 x i32> %454 to <8 x half>, !dbg !14
  %address.local714 = getelementptr i8, ptr addrspace(3) %address.local706, i64 4096, !dbg !14
  %456 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local714, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %457 = bitcast <4 x i32> %456 to <8 x half>, !dbg !14
  %address.local716 = getelementptr i8, ptr addrspace(3) %address.local706, i64 5120, !dbg !14
  %458 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local716, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %459 = bitcast <4 x i32> %458 to <8 x half>, !dbg !14
  %address.local718 = getelementptr i8, ptr addrspace(3) %address.local706, i64 6144, !dbg !14
  %460 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local718, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %461 = bitcast <4 x i32> %460 to <8 x half>, !dbg !14
  %address.local720 = getelementptr i8, ptr addrspace(3) %address.local706, i64 7168, !dbg !14
  %462 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local720, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %463 = shufflevector <8 x half> %449, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %464 = extractvalue [2 x <8 x float>] %99, 0, !dbg !28
  %465 = shufflevector <8 x float> %464, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %466 = fmul <4 x float> %465, %434, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %467 = fsub <4 x float> %303, %428, !dbg !21
  %468 = extractelement <4 x float> %467, i64 0, !dbg !25
  %469 = call float @llvm.amdgcn.exp2.f32(float %468), !dbg !25
  %470 = insertelement <4 x float> poison, float %469, i64 0, !dbg !27
  %471 = extractelement <4 x float> %467, i64 1, !dbg !25
  %472 = call float @llvm.amdgcn.exp2.f32(float %471), !dbg !25
  %473 = insertelement <4 x float> %470, float %472, i64 1, !dbg !27
  %474 = extractelement <4 x float> %467, i64 2, !dbg !25
  %475 = call float @llvm.amdgcn.exp2.f32(float %474), !dbg !25
  %476 = insertelement <4 x float> %473, float %475, i64 2, !dbg !27
  %477 = extractelement <4 x float> %467, i64 3, !dbg !25
  %478 = call float @llvm.amdgcn.exp2.f32(float %477), !dbg !25
  %479 = insertelement <4 x float> %476, float %478, i64 3, !dbg !27
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %479) #9, !dbg !27, !frisk.register_lifetimes !1
  %480 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !27
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %480) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %481 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %463, <4 x float> %466) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %482 = shufflevector <8 x half> %449, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %483 = shufflevector <8 x float> %464, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %484 = fmul <4 x float> %483, %434, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %485 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %482, <4 x float> %484) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %486 = extractvalue [2 x <8 x float>] %99, 1, !dbg !28
  %487 = shufflevector <8 x float> %486, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %488 = fmul <4 x float> %487, %442, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %489 = fsub <4 x float> %357, %429, !dbg !21
  %490 = extractelement <4 x float> %489, i64 0, !dbg !25
  %491 = call float @llvm.amdgcn.exp2.f32(float %490), !dbg !25
  %492 = insertelement <4 x float> poison, float %491, i64 0, !dbg !27
  %493 = extractelement <4 x float> %489, i64 1, !dbg !25
  %494 = call float @llvm.amdgcn.exp2.f32(float %493), !dbg !25
  %495 = insertelement <4 x float> %492, float %494, i64 1, !dbg !27
  %496 = extractelement <4 x float> %489, i64 2, !dbg !25
  %497 = call float @llvm.amdgcn.exp2.f32(float %496), !dbg !25
  %498 = insertelement <4 x float> %495, float %497, i64 2, !dbg !27
  %499 = extractelement <4 x float> %489, i64 3, !dbg !25
  %500 = call float @llvm.amdgcn.exp2.f32(float %499), !dbg !25
  %501 = insertelement <4 x float> %498, float %500, i64 3, !dbg !27
  %math.result335 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %501) #9, !dbg !27, !frisk.register_lifetimes !1
  %502 = fptrunc <4 x float> %math.result335 to <4 x half>, !dbg !27
  %math.packed336 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %502) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %503 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %463, <4 x float> %488) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %504 = shufflevector <8 x float> %486, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %505 = fmul <4 x float> %504, %442, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %506 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %482, <4 x float> %505) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %507 = shufflevector <8 x half> %451, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %508 = fsub <4 x float> %314, %428, !dbg !21
  %509 = extractelement <4 x float> %508, i64 0, !dbg !25
  %510 = call float @llvm.amdgcn.exp2.f32(float %509), !dbg !25
  %local.reduce71 = fadd float %469, %510, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor = call float asm sideeffect "", "=v,0"(float %local.reduce71) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %511 = insertelement <4 x float> poison, float %510, i64 0, !dbg !27
  %512 = extractelement <4 x float> %508, i64 1, !dbg !25
  %513 = call float @llvm.amdgcn.exp2.f32(float %512), !dbg !25
  %local.reduce72 = fadd float %472, %513, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1621 = call float asm sideeffect "", "=v,0"(float %local.reduce72) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %514 = insertelement <4 x float> %511, float %513, i64 1, !dbg !27
  %515 = extractelement <4 x float> %508, i64 2, !dbg !25
  %516 = call float @llvm.amdgcn.exp2.f32(float %515), !dbg !25
  %local.reduce73 = fadd float %475, %516, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1622 = call float asm sideeffect "", "=v,0"(float %local.reduce73) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %517 = insertelement <4 x float> %514, float %516, i64 2, !dbg !27
  %518 = extractelement <4 x float> %508, i64 3, !dbg !25
  %519 = call float @llvm.amdgcn.exp2.f32(float %518), !dbg !25
  %local.reduce74 = fadd float %478, %519, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1623 = call float asm sideeffect "", "=v,0"(float %local.reduce74) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %520 = insertelement <4 x float> %517, float %519, i64 3, !dbg !27
  %math.result321 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %520) #9, !dbg !27, !frisk.register_lifetimes !1
  %521 = fptrunc <4 x float> %math.result321 to <4 x half>, !dbg !27
  %math.packed322 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %521) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %522 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %507, <4 x float> %481) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %523 = shufflevector <8 x half> %451, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %524 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %523, <4 x float> %485) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %525 = fsub <4 x float> %363, %429, !dbg !21
  %526 = extractelement <4 x float> %525, i64 0, !dbg !25
  %527 = call float @llvm.amdgcn.exp2.f32(float %526), !dbg !25
  %local.reduce = fadd float %491, %527, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1624 = call float asm sideeffect "", "=v,0"(float %local.reduce) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %528 = insertelement <4 x float> poison, float %527, i64 0, !dbg !27
  %529 = extractelement <4 x float> %525, i64 1, !dbg !25
  %530 = call float @llvm.amdgcn.exp2.f32(float %529), !dbg !25
  %local.reduce41 = fadd float %494, %530, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1625 = call float asm sideeffect "", "=v,0"(float %local.reduce41) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %531 = insertelement <4 x float> %528, float %530, i64 1, !dbg !27
  %532 = extractelement <4 x float> %525, i64 2, !dbg !25
  %533 = call float @llvm.amdgcn.exp2.f32(float %532), !dbg !25
  %local.reduce42 = fadd float %497, %533, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1626 = call float asm sideeffect "", "=v,0"(float %local.reduce42) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %534 = insertelement <4 x float> %531, float %533, i64 2, !dbg !27
  %535 = extractelement <4 x float> %525, i64 3, !dbg !25
  %536 = call float @llvm.amdgcn.exp2.f32(float %535), !dbg !25
  %local.reduce43 = fadd float %500, %536, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1627 = call float asm sideeffect "", "=v,0"(float %local.reduce43) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %537 = insertelement <4 x float> %534, float %536, i64 3, !dbg !27
  %math.result337 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %537) #9, !dbg !27, !frisk.register_lifetimes !1
  %538 = fptrunc <4 x float> %math.result337 to <4 x half>, !dbg !27
  %math.packed338 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %538) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %539 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %507, <4 x float> %503) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %540 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %523, <4 x float> %506) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %541 = shufflevector <8 x half> %453, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %542 = fsub <4 x float> %320, %428, !dbg !21
  %543 = extractelement <4 x float> %542, i64 0, !dbg !25
  %544 = call float @llvm.amdgcn.exp2.f32(float %543), !dbg !25
  %local.reduce75 = fadd float %k0.sum.anchor, %544, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1628 = call float asm sideeffect "", "=v,0"(float %local.reduce75) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %545 = insertelement <4 x float> poison, float %544, i64 0, !dbg !27
  %546 = extractelement <4 x float> %542, i64 1, !dbg !25
  %547 = call float @llvm.amdgcn.exp2.f32(float %546), !dbg !25
  %local.reduce76 = fadd float %k0.sum.anchor1621, %547, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1629 = call float asm sideeffect "", "=v,0"(float %local.reduce76) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %548 = insertelement <4 x float> %545, float %547, i64 1, !dbg !27
  %549 = extractelement <4 x float> %542, i64 2, !dbg !25
  %550 = call float @llvm.amdgcn.exp2.f32(float %549), !dbg !25
  %local.reduce77 = fadd float %k0.sum.anchor1622, %550, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1630 = call float asm sideeffect "", "=v,0"(float %local.reduce77) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %551 = insertelement <4 x float> %548, float %550, i64 2, !dbg !27
  %552 = extractelement <4 x float> %542, i64 3, !dbg !25
  %553 = call float @llvm.amdgcn.exp2.f32(float %552), !dbg !25
  %local.reduce78 = fadd float %k0.sum.anchor1623, %553, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1631 = call float asm sideeffect "", "=v,0"(float %local.reduce78) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %554 = insertelement <4 x float> %551, float %553, i64 3, !dbg !27
  %math.result323 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %554) #9, !dbg !27, !frisk.register_lifetimes !1
  %555 = fptrunc <4 x float> %math.result323 to <4 x half>, !dbg !27
  %math.packed324 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %555) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %556 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %541, <4 x float> %522) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %557 = shufflevector <8 x half> %453, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %558 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %557, <4 x float> %524) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %559 = fsub <4 x float> %369, %429, !dbg !21
  %560 = extractelement <4 x float> %559, i64 0, !dbg !25
  %561 = call float @llvm.amdgcn.exp2.f32(float %560), !dbg !25
  %local.reduce44 = fadd float %k0.sum.anchor1624, %561, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1632 = call float asm sideeffect "", "=v,0"(float %local.reduce44) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %562 = insertelement <4 x float> poison, float %561, i64 0, !dbg !27
  %563 = extractelement <4 x float> %559, i64 1, !dbg !25
  %564 = call float @llvm.amdgcn.exp2.f32(float %563), !dbg !25
  %local.reduce45 = fadd float %k0.sum.anchor1625, %564, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1633 = call float asm sideeffect "", "=v,0"(float %local.reduce45) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %565 = insertelement <4 x float> %562, float %564, i64 1, !dbg !27
  %566 = extractelement <4 x float> %559, i64 2, !dbg !25
  %567 = call float @llvm.amdgcn.exp2.f32(float %566), !dbg !25
  %local.reduce46 = fadd float %k0.sum.anchor1626, %567, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1634 = call float asm sideeffect "", "=v,0"(float %local.reduce46) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %568 = insertelement <4 x float> %565, float %567, i64 2, !dbg !27
  %569 = extractelement <4 x float> %559, i64 3, !dbg !25
  %570 = call float @llvm.amdgcn.exp2.f32(float %569), !dbg !25
  %local.reduce47 = fadd float %k0.sum.anchor1627, %570, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1635 = call float asm sideeffect "", "=v,0"(float %local.reduce47) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %571 = insertelement <4 x float> %568, float %570, i64 3, !dbg !27
  %math.result339 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %571) #9, !dbg !27, !frisk.register_lifetimes !1
  %572 = fptrunc <4 x float> %math.result339 to <4 x half>, !dbg !27
  %math.packed340 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %572) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %573 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed340, <4 x half> %541, <4 x float> %539) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %574 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed340, <4 x half> %557, <4 x float> %540) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %575 = shufflevector <8 x half> %455, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %576 = fsub <4 x float> %326, %428, !dbg !21
  %577 = extractelement <4 x float> %576, i64 0, !dbg !25
  %578 = call float @llvm.amdgcn.exp2.f32(float %577), !dbg !25
  %local.reduce79 = fadd float %k0.sum.anchor1628, %578, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1636 = call float asm sideeffect "", "=v,0"(float %local.reduce79) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %579 = insertelement <4 x float> poison, float %578, i64 0, !dbg !27
  %580 = extractelement <4 x float> %576, i64 1, !dbg !25
  %581 = call float @llvm.amdgcn.exp2.f32(float %580), !dbg !25
  %local.reduce80 = fadd float %k0.sum.anchor1629, %581, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1637 = call float asm sideeffect "", "=v,0"(float %local.reduce80) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %582 = insertelement <4 x float> %579, float %581, i64 1, !dbg !27
  %583 = extractelement <4 x float> %576, i64 2, !dbg !25
  %584 = call float @llvm.amdgcn.exp2.f32(float %583), !dbg !25
  %local.reduce81 = fadd float %k0.sum.anchor1630, %584, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1638 = call float asm sideeffect "", "=v,0"(float %local.reduce81) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %585 = insertelement <4 x float> %582, float %584, i64 2, !dbg !27
  %586 = extractelement <4 x float> %576, i64 3, !dbg !25
  %587 = call float @llvm.amdgcn.exp2.f32(float %586), !dbg !25
  %local.reduce82 = fadd float %k0.sum.anchor1631, %587, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1639 = call float asm sideeffect "", "=v,0"(float %local.reduce82) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %588 = insertelement <4 x float> %585, float %587, i64 3, !dbg !27
  %math.result325 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %588) #9, !dbg !27, !frisk.register_lifetimes !1
  %589 = fptrunc <4 x float> %math.result325 to <4 x half>, !dbg !27
  %math.packed326 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %589) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %590 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %575, <4 x float> %556) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %591 = shufflevector <8 x half> %455, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %592 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %591, <4 x float> %558) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %593 = fsub <4 x float> %375, %429, !dbg !21
  %594 = extractelement <4 x float> %593, i64 0, !dbg !25
  %595 = call float @llvm.amdgcn.exp2.f32(float %594), !dbg !25
  %local.reduce48 = fadd float %k0.sum.anchor1632, %595, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1640 = call float asm sideeffect "", "=v,0"(float %local.reduce48) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %596 = insertelement <4 x float> poison, float %595, i64 0, !dbg !27
  %597 = extractelement <4 x float> %593, i64 1, !dbg !25
  %598 = call float @llvm.amdgcn.exp2.f32(float %597), !dbg !25
  %local.reduce49 = fadd float %k0.sum.anchor1633, %598, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1641 = call float asm sideeffect "", "=v,0"(float %local.reduce49) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %599 = insertelement <4 x float> %596, float %598, i64 1, !dbg !27
  %600 = extractelement <4 x float> %593, i64 2, !dbg !25
  %601 = call float @llvm.amdgcn.exp2.f32(float %600), !dbg !25
  %local.reduce50 = fadd float %k0.sum.anchor1634, %601, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1642 = call float asm sideeffect "", "=v,0"(float %local.reduce50) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %602 = insertelement <4 x float> %599, float %601, i64 2, !dbg !27
  %603 = extractelement <4 x float> %593, i64 3, !dbg !25
  %604 = call float @llvm.amdgcn.exp2.f32(float %603), !dbg !25
  %local.reduce51 = fadd float %k0.sum.anchor1635, %604, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1643 = call float asm sideeffect "", "=v,0"(float %local.reduce51) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %605 = insertelement <4 x float> %602, float %604, i64 3, !dbg !27
  %math.result341 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %605) #9, !dbg !27, !frisk.register_lifetimes !1
  %606 = fptrunc <4 x float> %math.result341 to <4 x half>, !dbg !27
  %math.packed342 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %606) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %607 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed342, <4 x half> %575, <4 x float> %573) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %608 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed342, <4 x half> %591, <4 x float> %574) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %609 = shufflevector <8 x half> %457, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %610 = fsub <4 x float> %332, %428, !dbg !21
  %611 = extractelement <4 x float> %610, i64 0, !dbg !25
  %612 = call float @llvm.amdgcn.exp2.f32(float %611), !dbg !25
  %local.reduce83 = fadd float %k0.sum.anchor1636, %612, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1644 = call float asm sideeffect "", "=v,0"(float %local.reduce83) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %613 = insertelement <4 x float> poison, float %612, i64 0, !dbg !27
  %614 = extractelement <4 x float> %610, i64 1, !dbg !25
  %615 = call float @llvm.amdgcn.exp2.f32(float %614), !dbg !25
  %local.reduce84 = fadd float %k0.sum.anchor1637, %615, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1645 = call float asm sideeffect "", "=v,0"(float %local.reduce84) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %616 = insertelement <4 x float> %613, float %615, i64 1, !dbg !27
  %617 = extractelement <4 x float> %610, i64 2, !dbg !25
  %618 = call float @llvm.amdgcn.exp2.f32(float %617), !dbg !25
  %local.reduce85 = fadd float %k0.sum.anchor1638, %618, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1646 = call float asm sideeffect "", "=v,0"(float %local.reduce85) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %619 = insertelement <4 x float> %616, float %618, i64 2, !dbg !27
  %620 = extractelement <4 x float> %610, i64 3, !dbg !25
  %621 = call float @llvm.amdgcn.exp2.f32(float %620), !dbg !25
  %local.reduce86 = fadd float %k0.sum.anchor1639, %621, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1647 = call float asm sideeffect "", "=v,0"(float %local.reduce86) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %622 = insertelement <4 x float> %619, float %621, i64 3, !dbg !27
  %math.result327 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %622) #9, !dbg !27, !frisk.register_lifetimes !1
  %623 = fptrunc <4 x float> %math.result327 to <4 x half>, !dbg !27
  %math.packed328 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %623) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %624 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %609, <4 x float> %590) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %625 = shufflevector <8 x half> %457, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %626 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %625, <4 x float> %592) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %627 = fsub <4 x float> %381, %429, !dbg !21
  %628 = extractelement <4 x float> %627, i64 0, !dbg !25
  %629 = call float @llvm.amdgcn.exp2.f32(float %628), !dbg !25
  %local.reduce52 = fadd float %k0.sum.anchor1640, %629, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1648 = call float asm sideeffect "", "=v,0"(float %local.reduce52) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %630 = insertelement <4 x float> poison, float %629, i64 0, !dbg !27
  %631 = extractelement <4 x float> %627, i64 1, !dbg !25
  %632 = call float @llvm.amdgcn.exp2.f32(float %631), !dbg !25
  %local.reduce53 = fadd float %k0.sum.anchor1641, %632, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1649 = call float asm sideeffect "", "=v,0"(float %local.reduce53) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %633 = insertelement <4 x float> %630, float %632, i64 1, !dbg !27
  %634 = extractelement <4 x float> %627, i64 2, !dbg !25
  %635 = call float @llvm.amdgcn.exp2.f32(float %634), !dbg !25
  %local.reduce54 = fadd float %k0.sum.anchor1642, %635, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1650 = call float asm sideeffect "", "=v,0"(float %local.reduce54) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %636 = insertelement <4 x float> %633, float %635, i64 2, !dbg !27
  %637 = extractelement <4 x float> %627, i64 3, !dbg !25
  %638 = call float @llvm.amdgcn.exp2.f32(float %637), !dbg !25
  %local.reduce55 = fadd float %k0.sum.anchor1643, %638, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1651 = call float asm sideeffect "", "=v,0"(float %local.reduce55) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %639 = insertelement <4 x float> %636, float %638, i64 3, !dbg !27
  %math.result343 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %639) #9, !dbg !27, !frisk.register_lifetimes !1
  %640 = fptrunc <4 x float> %math.result343 to <4 x half>, !dbg !27
  %math.packed344 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %640) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %641 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed344, <4 x half> %609, <4 x float> %607) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %642 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed344, <4 x half> %625, <4 x float> %608) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %643 = shufflevector <8 x half> %459, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %644 = fsub <4 x float> %338, %428, !dbg !21
  %645 = extractelement <4 x float> %644, i64 0, !dbg !25
  %646 = call float @llvm.amdgcn.exp2.f32(float %645), !dbg !25
  %local.reduce87 = fadd float %k0.sum.anchor1644, %646, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1652 = call float asm sideeffect "", "=v,0"(float %local.reduce87) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %647 = insertelement <4 x float> poison, float %646, i64 0, !dbg !27
  %648 = extractelement <4 x float> %644, i64 1, !dbg !25
  %649 = call float @llvm.amdgcn.exp2.f32(float %648), !dbg !25
  %local.reduce88 = fadd float %k0.sum.anchor1645, %649, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1653 = call float asm sideeffect "", "=v,0"(float %local.reduce88) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %650 = insertelement <4 x float> %647, float %649, i64 1, !dbg !27
  %651 = extractelement <4 x float> %644, i64 2, !dbg !25
  %652 = call float @llvm.amdgcn.exp2.f32(float %651), !dbg !25
  %local.reduce89 = fadd float %k0.sum.anchor1646, %652, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1654 = call float asm sideeffect "", "=v,0"(float %local.reduce89) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %653 = insertelement <4 x float> %650, float %652, i64 2, !dbg !27
  %654 = extractelement <4 x float> %644, i64 3, !dbg !25
  %655 = call float @llvm.amdgcn.exp2.f32(float %654), !dbg !25
  %local.reduce90 = fadd float %k0.sum.anchor1647, %655, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1655 = call float asm sideeffect "", "=v,0"(float %local.reduce90) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %656 = insertelement <4 x float> %653, float %655, i64 3, !dbg !27
  %math.result329 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %656) #9, !dbg !27, !frisk.register_lifetimes !1
  %657 = fptrunc <4 x float> %math.result329 to <4 x half>, !dbg !27
  %math.packed330 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %657) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %658 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %643, <4 x float> %624) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %659 = shufflevector <8 x half> %459, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %660 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %659, <4 x float> %626) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %661 = fsub <4 x float> %387, %429, !dbg !21
  %662 = extractelement <4 x float> %661, i64 0, !dbg !25
  %663 = call float @llvm.amdgcn.exp2.f32(float %662), !dbg !25
  %local.reduce56 = fadd float %k0.sum.anchor1648, %663, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1656 = call float asm sideeffect "", "=v,0"(float %local.reduce56) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %664 = insertelement <4 x float> poison, float %663, i64 0, !dbg !27
  %665 = extractelement <4 x float> %661, i64 1, !dbg !25
  %666 = call float @llvm.amdgcn.exp2.f32(float %665), !dbg !25
  %local.reduce57 = fadd float %k0.sum.anchor1649, %666, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1657 = call float asm sideeffect "", "=v,0"(float %local.reduce57) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %667 = insertelement <4 x float> %664, float %666, i64 1, !dbg !27
  %668 = extractelement <4 x float> %661, i64 2, !dbg !25
  %669 = call float @llvm.amdgcn.exp2.f32(float %668), !dbg !25
  %local.reduce58 = fadd float %k0.sum.anchor1650, %669, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1658 = call float asm sideeffect "", "=v,0"(float %local.reduce58) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %670 = insertelement <4 x float> %667, float %669, i64 2, !dbg !27
  %671 = extractelement <4 x float> %661, i64 3, !dbg !25
  %672 = call float @llvm.amdgcn.exp2.f32(float %671), !dbg !25
  %local.reduce59 = fadd float %k0.sum.anchor1651, %672, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1659 = call float asm sideeffect "", "=v,0"(float %local.reduce59) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %673 = insertelement <4 x float> %670, float %672, i64 3, !dbg !27
  %math.result345 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %673) #9, !dbg !27, !frisk.register_lifetimes !1
  %674 = fptrunc <4 x float> %math.result345 to <4 x half>, !dbg !27
  %math.packed346 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %674) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %675 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed346, <4 x half> %643, <4 x float> %641) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %676 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed346, <4 x half> %659, <4 x float> %642) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %677 = shufflevector <8 x half> %461, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %678 = fsub <4 x float> %344, %428, !dbg !21
  %679 = extractelement <4 x float> %678, i64 0, !dbg !25
  %680 = call float @llvm.amdgcn.exp2.f32(float %679), !dbg !25
  %local.reduce91 = fadd float %k0.sum.anchor1652, %680, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1660 = call float asm sideeffect "", "=v,0"(float %local.reduce91) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %681 = insertelement <4 x float> poison, float %680, i64 0, !dbg !27
  %682 = extractelement <4 x float> %678, i64 1, !dbg !25
  %683 = call float @llvm.amdgcn.exp2.f32(float %682), !dbg !25
  %local.reduce92 = fadd float %k0.sum.anchor1653, %683, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1661 = call float asm sideeffect "", "=v,0"(float %local.reduce92) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %684 = insertelement <4 x float> %681, float %683, i64 1, !dbg !27
  %685 = extractelement <4 x float> %678, i64 2, !dbg !25
  %686 = call float @llvm.amdgcn.exp2.f32(float %685), !dbg !25
  %local.reduce93 = fadd float %k0.sum.anchor1654, %686, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1662 = call float asm sideeffect "", "=v,0"(float %local.reduce93) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %687 = insertelement <4 x float> %684, float %686, i64 2, !dbg !27
  %688 = extractelement <4 x float> %678, i64 3, !dbg !25
  %689 = call float @llvm.amdgcn.exp2.f32(float %688), !dbg !25
  %local.reduce94 = fadd float %k0.sum.anchor1655, %689, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1663 = call float asm sideeffect "", "=v,0"(float %local.reduce94) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %690 = insertelement <4 x float> %687, float %689, i64 3, !dbg !27
  %math.result331 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %690) #9, !dbg !27, !frisk.register_lifetimes !1
  %691 = fptrunc <4 x float> %math.result331 to <4 x half>, !dbg !27
  %math.packed332 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %691) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %692 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %677, <4 x float> %658) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %693 = shufflevector <8 x half> %461, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %694 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %693, <4 x float> %660) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %695 = fsub <4 x float> %393, %429, !dbg !21
  %696 = extractelement <4 x float> %695, i64 0, !dbg !25
  %697 = call float @llvm.amdgcn.exp2.f32(float %696), !dbg !25
  %local.reduce60 = fadd float %k0.sum.anchor1656, %697, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1664 = call float asm sideeffect "", "=v,0"(float %local.reduce60) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %698 = insertelement <4 x float> poison, float %697, i64 0, !dbg !27
  %699 = extractelement <4 x float> %695, i64 1, !dbg !25
  %700 = call float @llvm.amdgcn.exp2.f32(float %699), !dbg !25
  %local.reduce61 = fadd float %k0.sum.anchor1657, %700, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1665 = call float asm sideeffect "", "=v,0"(float %local.reduce61) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %701 = insertelement <4 x float> %698, float %700, i64 1, !dbg !27
  %702 = extractelement <4 x float> %695, i64 2, !dbg !25
  %703 = call float @llvm.amdgcn.exp2.f32(float %702), !dbg !25
  %local.reduce62 = fadd float %k0.sum.anchor1658, %703, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1666 = call float asm sideeffect "", "=v,0"(float %local.reduce62) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %704 = insertelement <4 x float> %701, float %703, i64 2, !dbg !27
  %705 = extractelement <4 x float> %695, i64 3, !dbg !25
  %706 = call float @llvm.amdgcn.exp2.f32(float %705), !dbg !25
  %local.reduce63 = fadd float %k0.sum.anchor1659, %706, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1667 = call float asm sideeffect "", "=v,0"(float %local.reduce63) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %707 = insertelement <4 x float> %704, float %706, i64 3, !dbg !27
  %math.result347 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %707) #9, !dbg !27, !frisk.register_lifetimes !1
  %708 = fptrunc <4 x float> %math.result347 to <4 x half>, !dbg !27
  %math.packed348 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %708) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %709 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed348, <4 x half> %677, <4 x float> %675) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %710 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed348, <4 x half> %693, <4 x float> %676) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %711 = bitcast <4 x i32> %462 to <8 x half>, !dbg !14
  %712 = shufflevector <8 x half> %711, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %713 = fsub <4 x float> %350, %428, !dbg !21
  %714 = extractelement <4 x float> %713, i64 0, !dbg !25
  %715 = call float @llvm.amdgcn.exp2.f32(float %714), !dbg !25
  %local.reduce95 = fadd float %k0.sum.anchor1660, %715, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1668 = call float asm sideeffect "", "=v,0"(float %local.reduce95) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %716 = insertelement <4 x float> poison, float %715, i64 0, !dbg !27
  %717 = extractelement <4 x float> %713, i64 1, !dbg !25
  %718 = call float @llvm.amdgcn.exp2.f32(float %717), !dbg !25
  %local.reduce96 = fadd float %k0.sum.anchor1661, %718, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1669 = call float asm sideeffect "", "=v,0"(float %local.reduce96) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %local.reduce99 = fadd float %k0.sum.anchor1668, %k0.sum.anchor1669, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1670 = call float asm sideeffect "", "=v,0"(float %local.reduce99) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %719 = insertelement <4 x float> %716, float %718, i64 1, !dbg !27
  %720 = extractelement <4 x float> %713, i64 2, !dbg !25
  %721 = call float @llvm.amdgcn.exp2.f32(float %720), !dbg !25
  %local.reduce97 = fadd float %k0.sum.anchor1662, %721, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1671 = call float asm sideeffect "", "=v,0"(float %local.reduce97) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %722 = insertelement <4 x float> %719, float %721, i64 2, !dbg !27
  %723 = extractelement <4 x float> %713, i64 3, !dbg !25
  %724 = call float @llvm.amdgcn.exp2.f32(float %723), !dbg !25
  %local.reduce98 = fadd float %k0.sum.anchor1663, %724, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1672 = call float asm sideeffect "", "=v,0"(float %local.reduce98) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %local.reduce100 = fadd float %k0.sum.anchor1671, %k0.sum.anchor1672, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1673 = call float asm sideeffect "", "=v,0"(float %local.reduce100) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %725 = fadd float %k0.sum.anchor1670, %k0.sum.anchor1673, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1674 = call float asm sideeffect "", "=v,0"(float %725) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %726 = fadd float %k0.sum.anchor1674, %437, !dbg !26
  %k0.sum.anchor1675 = call float asm sideeffect "", "=v,0"(float %726) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %727 = insertelement <1 x float> poison, float %k0.sum.anchor1675, i64 0, !dbg !8
  %728 = insertvalue [2 x <1 x float>] poison, <1 x float> %727, 0, !dbg !8
  %729 = insertelement <4 x float> %722, float %724, i64 3, !dbg !27
  %math.result333 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %729) #9, !dbg !27, !frisk.register_lifetimes !1
  %730 = fptrunc <4 x float> %math.result333 to <4 x half>, !dbg !27
  %math.packed334 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %730) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %731 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %712, <4 x float> %692) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %732 = shufflevector <8 x half> %711, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %733 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %732, <4 x float> %694) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %734 = fsub <4 x float> %399, %429, !dbg !21
  %735 = extractelement <4 x float> %734, i64 0, !dbg !25
  %736 = call float @llvm.amdgcn.exp2.f32(float %735), !dbg !25
  %local.reduce64 = fadd float %k0.sum.anchor1664, %736, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1676 = call float asm sideeffect "", "=v,0"(float %local.reduce64) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %737 = insertelement <4 x float> poison, float %736, i64 0, !dbg !27
  %738 = extractelement <4 x float> %734, i64 1, !dbg !25
  %739 = call float @llvm.amdgcn.exp2.f32(float %738), !dbg !25
  %local.reduce65 = fadd float %k0.sum.anchor1665, %739, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1677 = call float asm sideeffect "", "=v,0"(float %local.reduce65) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %local.reduce68 = fadd float %k0.sum.anchor1676, %k0.sum.anchor1677, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1678 = call float asm sideeffect "", "=v,0"(float %local.reduce68) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %740 = insertelement <4 x float> %737, float %739, i64 1, !dbg !27
  %741 = extractelement <4 x float> %734, i64 2, !dbg !25
  %742 = call float @llvm.amdgcn.exp2.f32(float %741), !dbg !25
  %local.reduce66 = fadd float %k0.sum.anchor1666, %742, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1679 = call float asm sideeffect "", "=v,0"(float %local.reduce66) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %743 = insertelement <4 x float> %740, float %742, i64 2, !dbg !27
  %744 = extractelement <4 x float> %734, i64 3, !dbg !25
  %745 = call float @llvm.amdgcn.exp2.f32(float %744), !dbg !25
  %local.reduce67 = fadd float %k0.sum.anchor1667, %745, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1680 = call float asm sideeffect "", "=v,0"(float %local.reduce67) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %local.reduce69 = fadd float %k0.sum.anchor1679, %k0.sum.anchor1680, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1681 = call float asm sideeffect "", "=v,0"(float %local.reduce69) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %746 = fadd float %k0.sum.anchor1678, %k0.sum.anchor1681, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1682 = call float asm sideeffect "", "=v,0"(float %746) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %747 = fadd float %k0.sum.anchor1682, %445, !dbg !26
  %k0.sum.anchor1683 = call float asm sideeffect "", "=v,0"(float %747) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %748 = insertelement <1 x float> poison, float %k0.sum.anchor1683, i64 0, !dbg !8
  %749 = insertvalue [2 x <1 x float>] %728, <1 x float> %748, 1, !dbg !8
  %750 = insertelement <4 x float> %743, float %745, i64 3, !dbg !27
  %math.result349 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %750) #9, !dbg !27, !frisk.register_lifetimes !1
  %751 = fptrunc <4 x float> %math.result349 to <4 x half>, !dbg !27
  %math.packed350 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %751) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %752 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed350, <4 x half> %712, <4 x float> %709) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %753 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed350, <4 x half> %732, <4 x float> %710) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %vdb.address1195 = or disjoint i64 %179, %address.local695, !dbg !16
  %vdb.address1196 = trunc nuw i64 %vdb.address1195 to i32, !dbg !16
  %vdb.address1197 = shl nuw nsw i32 %vdb.address1196, 1, !dbg !16
  %vdb.address1202 = or disjoint i64 %179, %address.local701, !dbg !16
  %vdb.address1203 = trunc nuw i64 %vdb.address1202 to i32, !dbg !16
  %vdb.address1204 = shl nuw nsw i32 %vdb.address1203, 1, !dbg !16
  %754 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %36, i32 %vdb.address1197, <4 x i32> %dma.buffer.rsrc1096) #10, !dbg !16
  %755 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %43, i32 %vdb.address1204, <4 x i32> %dma.buffer.rsrc1096) #10, !dbg !16
  %vdb.read = getelementptr i8, ptr addrspace(3) %address.local706, i64 8192, !dbg !14
  %756 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %757 = bitcast <4 x i32> %756 to <8 x half>, !dbg !14
  %vdb.read1188 = getelementptr i8, ptr addrspace(3) %address.local706, i64 9216, !dbg !14
  %758 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1188, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %759 = bitcast <4 x i32> %758 to <8 x half>, !dbg !14
  %vdb.read1189 = getelementptr i8, ptr addrspace(3) %address.local706, i64 10240, !dbg !14
  %760 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1189, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %761 = bitcast <4 x i32> %760 to <8 x half>, !dbg !14
  %vdb.read1190 = getelementptr i8, ptr addrspace(3) %address.local706, i64 11264, !dbg !14
  %762 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1190, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %763 = bitcast <4 x i32> %762 to <8 x half>, !dbg !14
  %vdb.read1191 = getelementptr i8, ptr addrspace(3) %address.local706, i64 12288, !dbg !14
  %764 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1191, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %765 = bitcast <4 x i32> %764 to <8 x half>, !dbg !14
  %vdb.read1192 = getelementptr i8, ptr addrspace(3) %address.local706, i64 13312, !dbg !14
  %766 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1192, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %767 = bitcast <4 x i32> %766 to <8 x half>, !dbg !14
  %vdb.read1193 = getelementptr i8, ptr addrspace(3) %address.local706, i64 14336, !dbg !14
  %768 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1193, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %769 = bitcast <4 x i32> %768 to <8 x half>, !dbg !14
  %vdb.read1194 = getelementptr i8, ptr addrspace(3) %address.local706, i64 15360, !dbg !14
  %770 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1194, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %771 = shufflevector <8 x half> %757, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %772 = extractvalue [2 x <8 x float>] %102, 0, !dbg !28
  %773 = shufflevector <8 x float> %772, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %774 = fmul <4 x float> %773, %434, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %775 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %771, <4 x float> %774) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %776 = shufflevector <8 x half> %757, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %777 = shufflevector <8 x float> %772, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %778 = fmul <4 x float> %777, %434, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %779 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %776, <4 x float> %778) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %780 = extractvalue [2 x <8 x float>] %102, 1, !dbg !28
  %781 = shufflevector <8 x float> %780, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %782 = fmul <4 x float> %781, %442, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %783 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %771, <4 x float> %782) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %784 = shufflevector <8 x float> %780, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %785 = fmul <4 x float> %784, %442, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %786 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %776, <4 x float> %785) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %787 = shufflevector <8 x half> %759, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %788 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %787, <4 x float> %775) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %789 = shufflevector <8 x half> %759, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %790 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %789, <4 x float> %779) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %791 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %787, <4 x float> %783) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %792 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %789, <4 x float> %786) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %793 = shufflevector <8 x half> %761, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %794 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %793, <4 x float> %788) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %795 = shufflevector <8 x half> %761, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %796 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %795, <4 x float> %790) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %797 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed340, <4 x half> %793, <4 x float> %791) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %798 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed340, <4 x half> %795, <4 x float> %792) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %799 = shufflevector <8 x half> %763, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %800 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %799, <4 x float> %794) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %801 = shufflevector <8 x half> %763, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %802 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %801, <4 x float> %796) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %803 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed342, <4 x half> %799, <4 x float> %797) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %804 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed342, <4 x half> %801, <4 x float> %798) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %805 = shufflevector <8 x half> %765, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %806 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %805, <4 x float> %800) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %807 = shufflevector <8 x half> %765, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %808 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %807, <4 x float> %802) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %809 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed344, <4 x half> %805, <4 x float> %803) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %810 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed344, <4 x half> %807, <4 x float> %804) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %811 = shufflevector <8 x half> %767, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %812 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %811, <4 x float> %806) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %813 = shufflevector <8 x half> %767, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %814 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %813, <4 x float> %808) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %815 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed346, <4 x half> %811, <4 x float> %809) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %816 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed346, <4 x half> %813, <4 x float> %810) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %817 = shufflevector <8 x half> %769, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %818 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %817, <4 x float> %812) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %819 = shufflevector <8 x half> %769, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %820 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %819, <4 x float> %814) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %821 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed348, <4 x half> %817, <4 x float> %815) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %822 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed348, <4 x half> %819, <4 x float> %816) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %823 = bitcast <4 x i32> %770 to <8 x half>, !dbg !14
  %824 = shufflevector <8 x half> %823, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %825 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %824, <4 x float> %818) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %826 = shufflevector <8 x half> %823, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %827 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %826, <4 x float> %820) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %828 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed350, <4 x half> %824, <4 x float> %821) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %829 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed350, <4 x half> %826, <4 x float> %822) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %vdb.address1209 = or disjoint i64 %232, %address.local695, !dbg !16
  %vdb.address1210 = trunc nuw i64 %vdb.address1209 to i32, !dbg !16
  %vdb.address1211 = shl nuw nsw i32 %vdb.address1210, 1, !dbg !16
  %vdb.address1216 = or disjoint i64 %232, %address.local701, !dbg !16
  %vdb.address1217 = trunc nuw i64 %vdb.address1216 to i32, !dbg !16
  %vdb.address1218 = shl nuw nsw i32 %vdb.address1217, 1, !dbg !16
  %830 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1211, <4 x i32> %dma.buffer.rsrc1096) #10, !dbg !16
  %831 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1187, i32 %vdb.address1218, <4 x i32> %dma.buffer.rsrc1096) #10, !dbg !16
  %832 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local706, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %833 = bitcast <4 x i32> %832 to <8 x half>, !dbg !14
  %834 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local708, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %835 = bitcast <4 x i32> %834 to <8 x half>, !dbg !14
  %836 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local710, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %837 = bitcast <4 x i32> %836 to <8 x half>, !dbg !14
  %838 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local712, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %839 = bitcast <4 x i32> %838 to <8 x half>, !dbg !14
  %840 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local714, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %841 = bitcast <4 x i32> %840 to <8 x half>, !dbg !14
  %842 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local716, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %843 = bitcast <4 x i32> %842 to <8 x half>, !dbg !14
  %844 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local718, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %845 = bitcast <4 x i32> %844 to <8 x half>, !dbg !14
  %846 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local720, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %847 = shufflevector <8 x half> %833, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %848 = extractvalue [2 x <8 x float>] %103, 0, !dbg !28
  %849 = shufflevector <8 x float> %848, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %850 = fmul <4 x float> %849, %434, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %851 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %847, <4 x float> %850) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %852 = shufflevector <8 x half> %833, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %853 = shufflevector <8 x float> %848, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %854 = fmul <4 x float> %853, %434, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %855 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %852, <4 x float> %854) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %856 = extractvalue [2 x <8 x float>] %103, 1, !dbg !28
  %857 = shufflevector <8 x float> %856, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %858 = fmul <4 x float> %857, %442, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %859 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %847, <4 x float> %858) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %860 = shufflevector <8 x float> %856, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %861 = fmul <4 x float> %860, %442, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %862 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %852, <4 x float> %861) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %863 = shufflevector <8 x half> %835, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %864 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %863, <4 x float> %851) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %865 = shufflevector <8 x half> %835, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %866 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %865, <4 x float> %855) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %867 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %863, <4 x float> %859) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %868 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %865, <4 x float> %862) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %869 = shufflevector <8 x half> %837, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %870 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %869, <4 x float> %864) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %871 = shufflevector <8 x half> %837, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %872 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %871, <4 x float> %866) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %873 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed340, <4 x half> %869, <4 x float> %867) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %874 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed340, <4 x half> %871, <4 x float> %868) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %875 = shufflevector <8 x half> %839, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %876 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %875, <4 x float> %870) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %877 = shufflevector <8 x half> %839, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %878 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %877, <4 x float> %872) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %879 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed342, <4 x half> %875, <4 x float> %873) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %880 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed342, <4 x half> %877, <4 x float> %874) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %881 = shufflevector <8 x half> %841, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %882 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %881, <4 x float> %876) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %883 = shufflevector <8 x half> %841, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %884 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %883, <4 x float> %878) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %885 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed344, <4 x half> %881, <4 x float> %879) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %886 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed344, <4 x half> %883, <4 x float> %880) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %887 = shufflevector <8 x half> %843, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %888 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %887, <4 x float> %882) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %889 = shufflevector <8 x half> %843, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %890 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %889, <4 x float> %884) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %891 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed346, <4 x half> %887, <4 x float> %885) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %892 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed346, <4 x half> %889, <4 x float> %886) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %893 = shufflevector <8 x half> %845, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %894 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %893, <4 x float> %888) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %895 = shufflevector <8 x half> %845, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %896 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %895, <4 x float> %890) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %897 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed348, <4 x half> %893, <4 x float> %891) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %898 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed348, <4 x half> %895, <4 x float> %892) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %899 = bitcast <4 x i32> %846 to <8 x half>, !dbg !14
  %900 = shufflevector <8 x half> %899, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %901 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %900, <4 x float> %894) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %902 = shufflevector <8 x half> %899, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %903 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %902, <4 x float> %896) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %904 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed350, <4 x half> %900, <4 x float> %897) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %905 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed350, <4 x half> %902, <4 x float> %898) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %906 = add nuw nsw i64 %98, 1, !dbg !8
  %k0.has.next = icmp ult i64 %906, 32, !dbg !14
  br i1 %k0.has.next, label %k0.issue, label %k0.join, !dbg !14

k0.issue:                                         ; preds = %106
  %k0.next. = shl nuw nsw i64 %906, 14, !dbg !13
  %k0.next.1694 = or disjoint i64 %108, %k0.next., !dbg !13
  %k0.next.kdma.source1243 = or disjoint i64 %k0.next.1694, %kdma.source1242, !dbg !13
  %k0.next.kdma.source1244 = trunc nuw i64 %k0.next.kdma.source1243 to i32, !dbg !13
  %k0.next.kdma.source1245 = shl nuw nsw i32 %k0.next.kdma.source1244, 1, !dbg !13
  %k0.next.kdma.offset = or disjoint i32 %k0.next.kdma.source1245, %kdma.intra, !dbg !13
  %k0.next.1695 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %36, i32 %k0.next.kdma.offset, <4 x i32> %dma.buffer.rsrc1112) #10, !dbg !13, !frisk.cross_bn_k0 !1
  %k0.next.kdma.source1257 = or disjoint i64 %k0.next.1694, %kdma.source1256, !dbg !13
  %k0.next.kdma.source1258 = trunc nuw i64 %k0.next.kdma.source1257 to i32, !dbg !13
  %k0.next.kdma.source1259 = shl nuw nsw i32 %k0.next.kdma.source1258, 1, !dbg !13
  %k0.next.kdma.offset1260 = or disjoint i32 %k0.next.kdma.source1259, %kdma.intra.hi, !dbg !13
  %k0.next.1696 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1261, i32 %k0.next.kdma.offset1260, <4 x i32> %dma.buffer.rsrc1112) #10, !dbg !13, !frisk.cross_bn_k0 !1
  %k0.next.kdma.source1274 = or disjoint i64 %k0.next.1694, %kdma.source1273, !dbg !13
  %k0.next.kdma.source1275 = trunc nuw i64 %k0.next.kdma.source1274 to i32, !dbg !13
  %k0.next.kdma.source1276 = shl nuw nsw i32 %k0.next.kdma.source1275, 1, !dbg !13
  %k0.next.kdma.offset1277 = or disjoint i32 %k0.next.kdma.source1276, %kdma.intra, !dbg !13
  %k0.next.1697 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %43, i32 %k0.next.kdma.offset1277, <4 x i32> %dma.buffer.rsrc1112) #10, !dbg !13, !frisk.cross_bn_k0 !1
  %k0.next.kdma.source1291 = or disjoint i64 %k0.next.1694, %kdma.source1290, !dbg !13
  %k0.next.kdma.source1292 = trunc nuw i64 %k0.next.kdma.source1291 to i32, !dbg !13
  %k0.next.kdma.source1293 = shl nuw nsw i32 %k0.next.kdma.source1292, 1, !dbg !13
  %k0.next.kdma.offset1294 = or disjoint i32 %k0.next.kdma.source1293, %kdma.intra.hi, !dbg !13
  %k0.next.1698 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1295, i32 %k0.next.kdma.offset1294, <4 x i32> %dma.buffer.rsrc1112) #10, !dbg !13, !frisk.cross_bn_k0 !1
  br label %k0.join

k0.join:                                          ; preds = %106, %k0.issue
  %907 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %908 = bitcast <4 x i32> %907 to <8 x half>, !dbg !14
  %909 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1188, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %910 = bitcast <4 x i32> %909 to <8 x half>, !dbg !14
  %911 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1189, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %912 = bitcast <4 x i32> %911 to <8 x half>, !dbg !14
  %913 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1190, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %914 = bitcast <4 x i32> %913 to <8 x half>, !dbg !14
  %915 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1191, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %916 = bitcast <4 x i32> %915 to <8 x half>, !dbg !14
  %917 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1192, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %918 = bitcast <4 x i32> %917 to <8 x half>, !dbg !14
  %919 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1193, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %920 = bitcast <4 x i32> %919 to <8 x half>, !dbg !14
  %921 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1194, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %922 = shufflevector <8 x half> %908, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %923 = extractvalue [2 x <8 x float>] %104, 0, !dbg !28
  %924 = shufflevector <8 x float> %923, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %925 = fmul <4 x float> %924, %434, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %926 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %922, <4 x float> %925) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %927 = shufflevector <8 x half> %908, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %928 = shufflevector <8 x float> %923, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %929 = fmul <4 x float> %928, %434, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %930 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %927, <4 x float> %929) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %931 = extractvalue [2 x <8 x float>] %104, 1, !dbg !28
  %932 = shufflevector <8 x float> %931, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %933 = fmul <4 x float> %932, %442, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %934 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %922, <4 x float> %933) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %935 = shufflevector <8 x float> %931, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %936 = fmul <4 x float> %935, %442, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %937 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %927, <4 x float> %936) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %938 = shufflevector <8 x half> %910, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %939 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %938, <4 x float> %926) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %940 = shufflevector <8 x half> %910, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %941 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %940, <4 x float> %930) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %942 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %938, <4 x float> %934) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %943 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %940, <4 x float> %937) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %944 = shufflevector <8 x half> %912, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %945 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %944, <4 x float> %939) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %946 = shufflevector <8 x half> %912, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %947 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %946, <4 x float> %941) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %948 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed340, <4 x half> %944, <4 x float> %942) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %949 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed340, <4 x half> %946, <4 x float> %943) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %950 = shufflevector <8 x half> %914, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %951 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %950, <4 x float> %945) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %952 = shufflevector <8 x half> %914, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %953 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %952, <4 x float> %947) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %954 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed342, <4 x half> %950, <4 x float> %948) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %955 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed342, <4 x half> %952, <4 x float> %949) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %956 = shufflevector <8 x half> %916, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %957 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %956, <4 x float> %951) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %958 = shufflevector <8 x half> %916, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %959 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %958, <4 x float> %953) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %960 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed344, <4 x half> %956, <4 x float> %954) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %961 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed344, <4 x half> %958, <4 x float> %955) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %962 = shufflevector <8 x half> %918, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %963 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %962, <4 x float> %957) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %964 = shufflevector <8 x half> %918, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %965 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %964, <4 x float> %959) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %966 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed346, <4 x half> %962, <4 x float> %960) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %967 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed346, <4 x half> %964, <4 x float> %961) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %968 = shufflevector <8 x half> %920, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %969 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %968, <4 x float> %963) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %970 = shufflevector <8 x half> %920, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %971 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %970, <4 x float> %965) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %972 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed348, <4 x half> %968, <4 x float> %966) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %973 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed348, <4 x half> %970, <4 x float> %967) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %974 = bitcast <4 x i32> %921 to <8 x half>, !dbg !14
  %975 = shufflevector <8 x half> %974, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %976 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %975, <4 x float> %969) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %977 = shufflevector <8 x half> %974, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %978 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %977, <4 x float> %971) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %979 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed350, <4 x half> %975, <4 x float> %972) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %980 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed350, <4 x half> %977, <4 x float> %973) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !8
  %981 = shufflevector <4 x float> %731, <4 x float> %733, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %982 = insertvalue [2 x <8 x float>] poison, <8 x float> %981, 0, !dbg !8
  %983 = shufflevector <4 x float> %752, <4 x float> %753, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %984 = insertvalue [2 x <8 x float>] %982, <8 x float> %983, 1, !dbg !8
  %985 = insertvalue [2 x <1 x float>] poison, <1 x float> %424, 0, !dbg !8
  %986 = insertvalue [2 x <1 x float>] %985, <1 x float> %427, 1, !dbg !8
  %987 = shufflevector <4 x float> %825, <4 x float> %827, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %988 = insertvalue [2 x <8 x float>] poison, <8 x float> %987, 0, !dbg !8
  %989 = shufflevector <4 x float> %828, <4 x float> %829, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %990 = insertvalue [2 x <8 x float>] %988, <8 x float> %989, 1, !dbg !8
  %991 = shufflevector <4 x float> %901, <4 x float> %903, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %992 = insertvalue [2 x <8 x float>] poison, <8 x float> %991, 0, !dbg !8
  %993 = shufflevector <4 x float> %904, <4 x float> %905, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %994 = insertvalue [2 x <8 x float>] %992, <8 x float> %993, 1, !dbg !8
  %995 = shufflevector <4 x float> %976, <4 x float> %978, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %996 = insertvalue [2 x <8 x float>] poison, <8 x float> %995, 0, !dbg !8
  %997 = shufflevector <4 x float> %979, <4 x float> %980, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %998 = insertvalue [2 x <8 x float>] %996, <8 x float> %997, 1, !dbg !8
  br label %97, !dbg !8

999:                                              ; preds = %97
  %lane.sum.field166 = extractvalue [2 x <1 x float>] %100, 1, !dbg !8
  %lane.sum.local167 = extractelement <1 x float> %lane.sum.field166, i64 0, !dbg !8
  %1000 = shl nuw nsw i32 %8, 2, !dbg !8
  %1001 = xor i32 %1000, 64, !dbg !8
  %1002 = bitcast float %lane.sum.local167 to i32, !dbg !8
  %lane.sum.shuffle170 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1001, i32 %1002), !dbg !8
  %1003 = bitcast i32 %lane.sum.shuffle170 to float, !dbg !8
  %lane.sum.final171 = fadd float %lane.sum.local167, %1003, !dbg !8
  %1004 = xor i32 %1000, 128, !dbg !8
  %1005 = bitcast float %lane.sum.final171 to i32, !dbg !8
  %lane.sum.shuffle172 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1004, i32 %1005), !dbg !8
  %1006 = bitcast i32 %lane.sum.shuffle172 to float, !dbg !8
  %lane.sum.final173 = fadd float %lane.sum.final171, %1006, !dbg !8
  %1007 = insertelement <1 x float> poison, float %lane.sum.final173, i64 0, !dbg !8
  %lane.sum.field = extractvalue [2 x <1 x float>] %100, 0, !dbg !8
  %lane.sum.local = extractelement <1 x float> %lane.sum.field, i64 0, !dbg !8
  %1008 = bitcast float %lane.sum.local to i32, !dbg !8
  %lane.sum.shuffle = call i32 @llvm.amdgcn.ds.bpermute(i32 %1001, i32 %1008), !dbg !8
  %1009 = bitcast i32 %lane.sum.shuffle to float, !dbg !8
  %lane.sum.final = fadd float %lane.sum.local, %1009, !dbg !8
  %1010 = bitcast float %lane.sum.final to i32, !dbg !8
  %lane.sum.shuffle164 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1004, i32 %1010), !dbg !8
  %1011 = bitcast i32 %lane.sum.shuffle164 to float, !dbg !8
  %lane.sum.final165 = fadd float %lane.sum.final, %1011, !dbg !8
  %1012 = insertelement <1 x float> poison, float %lane.sum.final165, i64 0, !dbg !8
  %1013 = and i32 %kdma.tid, 16, !dbg !8
  %.not = icmp eq i32 %1013, 0, !dbg !8
  %1014 = and i32 %kdma.tid, 32, !dbg !8
  %.not18 = icmp eq i32 %1014, 0, !dbg !8
  %1015 = extractvalue [2 x <8 x float>] %104, 0, !dbg !8
  %1016 = extractelement <8 x float> %1015, i64 0, !dbg !8
  %1017 = extractelement <8 x float> %1015, i64 1, !dbg !8
  %1018 = extractelement <8 x float> %1015, i64 2, !dbg !8
  %1019 = extractelement <8 x float> %1015, i64 3, !dbg !8
  %1020 = select i1 %.not, float %1017, float %1016, !dbg !8
  %1021 = bitcast float %1020 to i32, !dbg !8
  %1022 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1021), !dbg !8
  %1023 = bitcast i32 %1022 to float, !dbg !8
  %1024 = select i1 %.not, float %1016, float %1023, !dbg !8
  %1025 = select i1 %.not, float %1023, float %1017, !dbg !8
  %1026 = select i1 %.not, float %1019, float %1018, !dbg !8
  %1027 = bitcast float %1026 to i32, !dbg !8
  %1028 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1027), !dbg !8
  %1029 = bitcast i32 %1028 to float, !dbg !8
  %1030 = select i1 %.not, float %1018, float %1029, !dbg !8
  %1031 = select i1 %.not, float %1029, float %1019, !dbg !8
  %1032 = select i1 %.not18, float %1030, float %1024, !dbg !8
  %1033 = bitcast float %1032 to i32, !dbg !8
  %1034 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1033), !dbg !8
  %1035 = bitcast i32 %1034 to float, !dbg !8
  %1036 = select i1 %.not18, float %1024, float %1035, !dbg !8
  %1037 = select i1 %.not18, float %1035, float %1030, !dbg !8
  %1038 = select i1 %.not18, float %1031, float %1025, !dbg !8
  %1039 = bitcast float %1038 to i32, !dbg !8
  %1040 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1039), !dbg !8
  %1041 = bitcast i32 %1040 to float, !dbg !8
  %1042 = select i1 %.not18, float %1025, float %1041, !dbg !8
  %1043 = select i1 %.not18, float %1041, float %1031, !dbg !8
  %1044 = extractelement <8 x float> %1015, i64 4, !dbg !8
  %1045 = extractelement <8 x float> %1015, i64 5, !dbg !8
  %1046 = extractelement <8 x float> %1015, i64 6, !dbg !8
  %1047 = extractelement <8 x float> %1015, i64 7, !dbg !8
  %1048 = select i1 %.not, float %1045, float %1044, !dbg !8
  %1049 = bitcast float %1048 to i32, !dbg !8
  %1050 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1049), !dbg !8
  %1051 = bitcast i32 %1050 to float, !dbg !8
  %1052 = select i1 %.not, float %1044, float %1051, !dbg !8
  %1053 = select i1 %.not, float %1051, float %1045, !dbg !8
  %1054 = select i1 %.not, float %1047, float %1046, !dbg !8
  %1055 = bitcast float %1054 to i32, !dbg !8
  %1056 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1055), !dbg !8
  %1057 = bitcast i32 %1056 to float, !dbg !8
  %1058 = select i1 %.not, float %1046, float %1057, !dbg !8
  %1059 = select i1 %.not, float %1057, float %1047, !dbg !8
  %1060 = select i1 %.not18, float %1058, float %1052, !dbg !8
  %1061 = bitcast float %1060 to i32, !dbg !8
  %1062 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1061), !dbg !8
  %1063 = bitcast i32 %1062 to float, !dbg !8
  %1064 = select i1 %.not18, float %1052, float %1063, !dbg !8
  %1065 = select i1 %.not18, float %1063, float %1058, !dbg !8
  %1066 = select i1 %.not18, float %1059, float %1053, !dbg !8
  %1067 = bitcast float %1066 to i32, !dbg !8
  %1068 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1067), !dbg !8
  %1069 = bitcast i32 %1068 to float, !dbg !8
  %1070 = select i1 %.not18, float %1053, float %1069, !dbg !8
  %1071 = select i1 %.not18, float %1069, float %1059, !dbg !8
  %1072 = extractvalue [2 x <8 x float>] %104, 1, !dbg !8
  %1073 = extractelement <8 x float> %1072, i64 0, !dbg !8
  %1074 = extractelement <8 x float> %1072, i64 1, !dbg !8
  %1075 = extractelement <8 x float> %1072, i64 2, !dbg !8
  %1076 = extractelement <8 x float> %1072, i64 3, !dbg !8
  %1077 = select i1 %.not, float %1074, float %1073, !dbg !8
  %1078 = bitcast float %1077 to i32, !dbg !8
  %1079 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1078), !dbg !8
  %1080 = bitcast i32 %1079 to float, !dbg !8
  %1081 = select i1 %.not, float %1073, float %1080, !dbg !8
  %1082 = select i1 %.not, float %1080, float %1074, !dbg !8
  %1083 = select i1 %.not, float %1076, float %1075, !dbg !8
  %1084 = bitcast float %1083 to i32, !dbg !8
  %1085 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1084), !dbg !8
  %1086 = bitcast i32 %1085 to float, !dbg !8
  %1087 = select i1 %.not, float %1075, float %1086, !dbg !8
  %1088 = select i1 %.not, float %1086, float %1076, !dbg !8
  %1089 = select i1 %.not18, float %1087, float %1081, !dbg !8
  %1090 = bitcast float %1089 to i32, !dbg !8
  %1091 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1090), !dbg !8
  %1092 = bitcast i32 %1091 to float, !dbg !8
  %1093 = select i1 %.not18, float %1081, float %1092, !dbg !8
  %1094 = select i1 %.not18, float %1092, float %1087, !dbg !8
  %1095 = select i1 %.not18, float %1088, float %1082, !dbg !8
  %1096 = bitcast float %1095 to i32, !dbg !8
  %1097 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1096), !dbg !8
  %1098 = bitcast i32 %1097 to float, !dbg !8
  %1099 = select i1 %.not18, float %1082, float %1098, !dbg !8
  %1100 = select i1 %.not18, float %1098, float %1088, !dbg !8
  %1101 = extractelement <8 x float> %1072, i64 4, !dbg !8
  %1102 = extractelement <8 x float> %1072, i64 5, !dbg !8
  %1103 = extractelement <8 x float> %1072, i64 6, !dbg !8
  %1104 = extractelement <8 x float> %1072, i64 7, !dbg !8
  %1105 = select i1 %.not, float %1102, float %1101, !dbg !8
  %1106 = bitcast float %1105 to i32, !dbg !8
  %1107 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1106), !dbg !8
  %1108 = bitcast i32 %1107 to float, !dbg !8
  %1109 = select i1 %.not, float %1101, float %1108, !dbg !8
  %1110 = select i1 %.not, float %1108, float %1102, !dbg !8
  %1111 = select i1 %.not, float %1104, float %1103, !dbg !8
  %1112 = bitcast float %1111 to i32, !dbg !8
  %1113 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1112), !dbg !8
  %1114 = bitcast i32 %1113 to float, !dbg !8
  %1115 = select i1 %.not, float %1103, float %1114, !dbg !8
  %1116 = select i1 %.not, float %1114, float %1104, !dbg !8
  %1117 = select i1 %.not18, float %1115, float %1109, !dbg !8
  %1118 = bitcast float %1117 to i32, !dbg !8
  %1119 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1118), !dbg !8
  %1120 = bitcast i32 %1119 to float, !dbg !8
  %1121 = select i1 %.not18, float %1109, float %1120, !dbg !8
  %1122 = select i1 %.not18, float %1120, float %1115, !dbg !8
  %1123 = select i1 %.not18, float %1116, float %1110, !dbg !8
  %1124 = bitcast float %1123 to i32, !dbg !8
  %1125 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1124), !dbg !8
  %1126 = bitcast i32 %1125 to float, !dbg !8
  %1127 = select i1 %.not18, float %1110, float %1126, !dbg !8
  %1128 = select i1 %.not18, float %1126, float %1116, !dbg !8
  %1129 = extractvalue [2 x <8 x float>] %103, 0, !dbg !8
  %1130 = extractelement <8 x float> %1129, i64 0, !dbg !8
  %1131 = extractelement <8 x float> %1129, i64 1, !dbg !8
  %1132 = extractelement <8 x float> %1129, i64 2, !dbg !8
  %1133 = extractelement <8 x float> %1129, i64 3, !dbg !8
  %1134 = select i1 %.not, float %1131, float %1130, !dbg !8
  %1135 = bitcast float %1134 to i32, !dbg !8
  %1136 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1135), !dbg !8
  %1137 = bitcast i32 %1136 to float, !dbg !8
  %1138 = select i1 %.not, float %1130, float %1137, !dbg !8
  %1139 = select i1 %.not, float %1137, float %1131, !dbg !8
  %1140 = select i1 %.not, float %1133, float %1132, !dbg !8
  %1141 = bitcast float %1140 to i32, !dbg !8
  %1142 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1141), !dbg !8
  %1143 = bitcast i32 %1142 to float, !dbg !8
  %1144 = select i1 %.not, float %1132, float %1143, !dbg !8
  %1145 = select i1 %.not, float %1143, float %1133, !dbg !8
  %1146 = select i1 %.not18, float %1144, float %1138, !dbg !8
  %1147 = bitcast float %1146 to i32, !dbg !8
  %1148 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1147), !dbg !8
  %1149 = bitcast i32 %1148 to float, !dbg !8
  %1150 = select i1 %.not18, float %1138, float %1149, !dbg !8
  %1151 = select i1 %.not18, float %1149, float %1144, !dbg !8
  %1152 = select i1 %.not18, float %1145, float %1139, !dbg !8
  %1153 = bitcast float %1152 to i32, !dbg !8
  %1154 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1153), !dbg !8
  %1155 = bitcast i32 %1154 to float, !dbg !8
  %1156 = select i1 %.not18, float %1139, float %1155, !dbg !8
  %1157 = select i1 %.not18, float %1155, float %1145, !dbg !8
  %1158 = extractelement <8 x float> %1129, i64 4, !dbg !8
  %1159 = extractelement <8 x float> %1129, i64 5, !dbg !8
  %1160 = extractelement <8 x float> %1129, i64 6, !dbg !8
  %1161 = extractelement <8 x float> %1129, i64 7, !dbg !8
  %1162 = select i1 %.not, float %1159, float %1158, !dbg !8
  %1163 = bitcast float %1162 to i32, !dbg !8
  %1164 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1163), !dbg !8
  %1165 = bitcast i32 %1164 to float, !dbg !8
  %1166 = select i1 %.not, float %1158, float %1165, !dbg !8
  %1167 = select i1 %.not, float %1165, float %1159, !dbg !8
  %1168 = select i1 %.not, float %1161, float %1160, !dbg !8
  %1169 = bitcast float %1168 to i32, !dbg !8
  %1170 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1169), !dbg !8
  %1171 = bitcast i32 %1170 to float, !dbg !8
  %1172 = select i1 %.not, float %1160, float %1171, !dbg !8
  %1173 = select i1 %.not, float %1171, float %1161, !dbg !8
  %1174 = select i1 %.not18, float %1172, float %1166, !dbg !8
  %1175 = bitcast float %1174 to i32, !dbg !8
  %1176 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1175), !dbg !8
  %1177 = bitcast i32 %1176 to float, !dbg !8
  %1178 = select i1 %.not18, float %1166, float %1177, !dbg !8
  %1179 = select i1 %.not18, float %1177, float %1172, !dbg !8
  %1180 = select i1 %.not18, float %1173, float %1167, !dbg !8
  %1181 = bitcast float %1180 to i32, !dbg !8
  %1182 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1181), !dbg !8
  %1183 = bitcast i32 %1182 to float, !dbg !8
  %1184 = select i1 %.not18, float %1167, float %1183, !dbg !8
  %1185 = select i1 %.not18, float %1183, float %1173, !dbg !8
  %1186 = extractvalue [2 x <8 x float>] %103, 1, !dbg !8
  %1187 = extractelement <8 x float> %1186, i64 0, !dbg !8
  %1188 = extractelement <8 x float> %1186, i64 1, !dbg !8
  %1189 = extractelement <8 x float> %1186, i64 2, !dbg !8
  %1190 = extractelement <8 x float> %1186, i64 3, !dbg !8
  %1191 = select i1 %.not, float %1188, float %1187, !dbg !8
  %1192 = bitcast float %1191 to i32, !dbg !8
  %1193 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1192), !dbg !8
  %1194 = bitcast i32 %1193 to float, !dbg !8
  %1195 = select i1 %.not, float %1187, float %1194, !dbg !8
  %1196 = select i1 %.not, float %1194, float %1188, !dbg !8
  %1197 = select i1 %.not, float %1190, float %1189, !dbg !8
  %1198 = bitcast float %1197 to i32, !dbg !8
  %1199 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1198), !dbg !8
  %1200 = bitcast i32 %1199 to float, !dbg !8
  %1201 = select i1 %.not, float %1189, float %1200, !dbg !8
  %1202 = select i1 %.not, float %1200, float %1190, !dbg !8
  %1203 = select i1 %.not18, float %1201, float %1195, !dbg !8
  %1204 = bitcast float %1203 to i32, !dbg !8
  %1205 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1204), !dbg !8
  %1206 = bitcast i32 %1205 to float, !dbg !8
  %1207 = select i1 %.not18, float %1195, float %1206, !dbg !8
  %1208 = select i1 %.not18, float %1206, float %1201, !dbg !8
  %1209 = select i1 %.not18, float %1202, float %1196, !dbg !8
  %1210 = bitcast float %1209 to i32, !dbg !8
  %1211 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1210), !dbg !8
  %1212 = bitcast i32 %1211 to float, !dbg !8
  %1213 = select i1 %.not18, float %1196, float %1212, !dbg !8
  %1214 = select i1 %.not18, float %1212, float %1202, !dbg !8
  %1215 = extractelement <8 x float> %1186, i64 4, !dbg !8
  %1216 = extractelement <8 x float> %1186, i64 5, !dbg !8
  %1217 = extractelement <8 x float> %1186, i64 6, !dbg !8
  %1218 = extractelement <8 x float> %1186, i64 7, !dbg !8
  %1219 = select i1 %.not, float %1216, float %1215, !dbg !8
  %1220 = bitcast float %1219 to i32, !dbg !8
  %1221 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1220), !dbg !8
  %1222 = bitcast i32 %1221 to float, !dbg !8
  %1223 = select i1 %.not, float %1215, float %1222, !dbg !8
  %1224 = select i1 %.not, float %1222, float %1216, !dbg !8
  %1225 = select i1 %.not, float %1218, float %1217, !dbg !8
  %1226 = bitcast float %1225 to i32, !dbg !8
  %1227 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1226), !dbg !8
  %1228 = bitcast i32 %1227 to float, !dbg !8
  %1229 = select i1 %.not, float %1217, float %1228, !dbg !8
  %1230 = select i1 %.not, float %1228, float %1218, !dbg !8
  %1231 = select i1 %.not18, float %1229, float %1223, !dbg !8
  %1232 = bitcast float %1231 to i32, !dbg !8
  %1233 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1232), !dbg !8
  %1234 = bitcast i32 %1233 to float, !dbg !8
  %1235 = select i1 %.not18, float %1223, float %1234, !dbg !8
  %1236 = select i1 %.not18, float %1234, float %1229, !dbg !8
  %1237 = select i1 %.not18, float %1230, float %1224, !dbg !8
  %1238 = bitcast float %1237 to i32, !dbg !8
  %1239 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1238), !dbg !8
  %1240 = bitcast i32 %1239 to float, !dbg !8
  %1241 = select i1 %.not18, float %1224, float %1240, !dbg !8
  %1242 = select i1 %.not18, float %1240, float %1230, !dbg !8
  %1243 = extractvalue [2 x <8 x float>] %102, 0, !dbg !8
  %1244 = extractelement <8 x float> %1243, i64 0, !dbg !8
  %1245 = extractelement <8 x float> %1243, i64 1, !dbg !8
  %1246 = extractelement <8 x float> %1243, i64 2, !dbg !8
  %1247 = extractelement <8 x float> %1243, i64 3, !dbg !8
  %1248 = select i1 %.not, float %1245, float %1244, !dbg !8
  %1249 = bitcast float %1248 to i32, !dbg !8
  %1250 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1249), !dbg !8
  %1251 = bitcast i32 %1250 to float, !dbg !8
  %1252 = select i1 %.not, float %1244, float %1251, !dbg !8
  %1253 = select i1 %.not, float %1251, float %1245, !dbg !8
  %1254 = select i1 %.not, float %1247, float %1246, !dbg !8
  %1255 = bitcast float %1254 to i32, !dbg !8
  %1256 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1255), !dbg !8
  %1257 = bitcast i32 %1256 to float, !dbg !8
  %1258 = select i1 %.not, float %1246, float %1257, !dbg !8
  %1259 = select i1 %.not, float %1257, float %1247, !dbg !8
  %1260 = select i1 %.not18, float %1258, float %1252, !dbg !8
  %1261 = bitcast float %1260 to i32, !dbg !8
  %1262 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1261), !dbg !8
  %1263 = bitcast i32 %1262 to float, !dbg !8
  %1264 = select i1 %.not18, float %1252, float %1263, !dbg !8
  %1265 = select i1 %.not18, float %1263, float %1258, !dbg !8
  %1266 = select i1 %.not18, float %1259, float %1253, !dbg !8
  %1267 = bitcast float %1266 to i32, !dbg !8
  %1268 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1267), !dbg !8
  %1269 = bitcast i32 %1268 to float, !dbg !8
  %1270 = select i1 %.not18, float %1253, float %1269, !dbg !8
  %1271 = select i1 %.not18, float %1269, float %1259, !dbg !8
  %1272 = extractelement <8 x float> %1243, i64 4, !dbg !8
  %1273 = extractelement <8 x float> %1243, i64 5, !dbg !8
  %1274 = extractelement <8 x float> %1243, i64 6, !dbg !8
  %1275 = extractelement <8 x float> %1243, i64 7, !dbg !8
  %1276 = select i1 %.not, float %1273, float %1272, !dbg !8
  %1277 = bitcast float %1276 to i32, !dbg !8
  %1278 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1277), !dbg !8
  %1279 = bitcast i32 %1278 to float, !dbg !8
  %1280 = select i1 %.not, float %1272, float %1279, !dbg !8
  %1281 = select i1 %.not, float %1279, float %1273, !dbg !8
  %1282 = select i1 %.not, float %1275, float %1274, !dbg !8
  %1283 = bitcast float %1282 to i32, !dbg !8
  %1284 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1283), !dbg !8
  %1285 = bitcast i32 %1284 to float, !dbg !8
  %1286 = select i1 %.not, float %1274, float %1285, !dbg !8
  %1287 = select i1 %.not, float %1285, float %1275, !dbg !8
  %1288 = select i1 %.not18, float %1286, float %1280, !dbg !8
  %1289 = bitcast float %1288 to i32, !dbg !8
  %1290 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1289), !dbg !8
  %1291 = bitcast i32 %1290 to float, !dbg !8
  %1292 = select i1 %.not18, float %1280, float %1291, !dbg !8
  %1293 = select i1 %.not18, float %1291, float %1286, !dbg !8
  %1294 = select i1 %.not18, float %1287, float %1281, !dbg !8
  %1295 = bitcast float %1294 to i32, !dbg !8
  %1296 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1295), !dbg !8
  %1297 = bitcast i32 %1296 to float, !dbg !8
  %1298 = select i1 %.not18, float %1281, float %1297, !dbg !8
  %1299 = select i1 %.not18, float %1297, float %1287, !dbg !8
  %1300 = extractvalue [2 x <8 x float>] %102, 1, !dbg !8
  %1301 = extractelement <8 x float> %1300, i64 0, !dbg !8
  %1302 = extractelement <8 x float> %1300, i64 1, !dbg !8
  %1303 = extractelement <8 x float> %1300, i64 2, !dbg !8
  %1304 = extractelement <8 x float> %1300, i64 3, !dbg !8
  %1305 = select i1 %.not, float %1302, float %1301, !dbg !8
  %1306 = bitcast float %1305 to i32, !dbg !8
  %1307 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1306), !dbg !8
  %1308 = bitcast i32 %1307 to float, !dbg !8
  %1309 = select i1 %.not, float %1301, float %1308, !dbg !8
  %1310 = select i1 %.not, float %1308, float %1302, !dbg !8
  %1311 = select i1 %.not, float %1304, float %1303, !dbg !8
  %1312 = bitcast float %1311 to i32, !dbg !8
  %1313 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1312), !dbg !8
  %1314 = bitcast i32 %1313 to float, !dbg !8
  %1315 = select i1 %.not, float %1303, float %1314, !dbg !8
  %1316 = select i1 %.not, float %1314, float %1304, !dbg !8
  %1317 = select i1 %.not18, float %1315, float %1309, !dbg !8
  %1318 = bitcast float %1317 to i32, !dbg !8
  %1319 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1318), !dbg !8
  %1320 = bitcast i32 %1319 to float, !dbg !8
  %1321 = select i1 %.not18, float %1309, float %1320, !dbg !8
  %1322 = select i1 %.not18, float %1320, float %1315, !dbg !8
  %1323 = select i1 %.not18, float %1316, float %1310, !dbg !8
  %1324 = bitcast float %1323 to i32, !dbg !8
  %1325 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1324), !dbg !8
  %1326 = bitcast i32 %1325 to float, !dbg !8
  %1327 = select i1 %.not18, float %1310, float %1326, !dbg !8
  %1328 = select i1 %.not18, float %1326, float %1316, !dbg !8
  %1329 = extractelement <8 x float> %1300, i64 4, !dbg !8
  %1330 = extractelement <8 x float> %1300, i64 5, !dbg !8
  %1331 = extractelement <8 x float> %1300, i64 6, !dbg !8
  %1332 = extractelement <8 x float> %1300, i64 7, !dbg !8
  %1333 = select i1 %.not, float %1330, float %1329, !dbg !8
  %1334 = bitcast float %1333 to i32, !dbg !8
  %1335 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1334), !dbg !8
  %1336 = bitcast i32 %1335 to float, !dbg !8
  %1337 = select i1 %.not, float %1329, float %1336, !dbg !8
  %1338 = select i1 %.not, float %1336, float %1330, !dbg !8
  %1339 = select i1 %.not, float %1332, float %1331, !dbg !8
  %1340 = bitcast float %1339 to i32, !dbg !8
  %1341 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1340), !dbg !8
  %1342 = bitcast i32 %1341 to float, !dbg !8
  %1343 = select i1 %.not, float %1331, float %1342, !dbg !8
  %1344 = select i1 %.not, float %1342, float %1332, !dbg !8
  %1345 = select i1 %.not18, float %1343, float %1337, !dbg !8
  %1346 = bitcast float %1345 to i32, !dbg !8
  %1347 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1346), !dbg !8
  %1348 = bitcast i32 %1347 to float, !dbg !8
  %1349 = select i1 %.not18, float %1337, float %1348, !dbg !8
  %1350 = select i1 %.not18, float %1348, float %1343, !dbg !8
  %1351 = select i1 %.not18, float %1344, float %1338, !dbg !8
  %1352 = bitcast float %1351 to i32, !dbg !8
  %1353 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1352), !dbg !8
  %1354 = bitcast i32 %1353 to float, !dbg !8
  %1355 = select i1 %.not18, float %1338, float %1354, !dbg !8
  %1356 = select i1 %.not18, float %1354, float %1344, !dbg !8
  %1357 = extractvalue [2 x <8 x float>] %99, 0, !dbg !8
  %1358 = extractelement <8 x float> %1357, i64 0, !dbg !8
  %1359 = extractelement <8 x float> %1357, i64 1, !dbg !8
  %1360 = extractelement <8 x float> %1357, i64 2, !dbg !8
  %1361 = extractelement <8 x float> %1357, i64 3, !dbg !8
  %1362 = select i1 %.not, float %1359, float %1358, !dbg !8
  %1363 = bitcast float %1362 to i32, !dbg !8
  %1364 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1363), !dbg !8
  %1365 = bitcast i32 %1364 to float, !dbg !8
  %1366 = select i1 %.not, float %1358, float %1365, !dbg !8
  %1367 = select i1 %.not, float %1365, float %1359, !dbg !8
  %1368 = select i1 %.not, float %1361, float %1360, !dbg !8
  %1369 = bitcast float %1368 to i32, !dbg !8
  %1370 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1369), !dbg !8
  %1371 = bitcast i32 %1370 to float, !dbg !8
  %1372 = select i1 %.not, float %1360, float %1371, !dbg !8
  %1373 = select i1 %.not, float %1371, float %1361, !dbg !8
  %1374 = select i1 %.not18, float %1372, float %1366, !dbg !8
  %1375 = bitcast float %1374 to i32, !dbg !8
  %1376 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1375), !dbg !8
  %1377 = select i1 %.not18, float %1373, float %1367, !dbg !8
  %1378 = bitcast float %1377 to i32, !dbg !8
  %1379 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1378), !dbg !8
  %1380 = extractelement <8 x float> %1357, i64 4, !dbg !8
  %1381 = extractelement <8 x float> %1357, i64 5, !dbg !8
  %1382 = extractelement <8 x float> %1357, i64 6, !dbg !8
  %1383 = extractelement <8 x float> %1357, i64 7, !dbg !8
  %1384 = select i1 %.not, float %1381, float %1380, !dbg !8
  %1385 = bitcast float %1384 to i32, !dbg !8
  %1386 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1385), !dbg !8
  %1387 = bitcast i32 %1386 to float, !dbg !8
  %1388 = select i1 %.not, float %1380, float %1387, !dbg !8
  %1389 = select i1 %.not, float %1387, float %1381, !dbg !8
  %1390 = select i1 %.not, float %1383, float %1382, !dbg !8
  %1391 = bitcast float %1390 to i32, !dbg !8
  %1392 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1391), !dbg !8
  %1393 = bitcast i32 %1392 to float, !dbg !8
  %1394 = select i1 %.not, float %1382, float %1393, !dbg !8
  %1395 = select i1 %.not, float %1393, float %1383, !dbg !8
  %1396 = select i1 %.not18, float %1394, float %1388, !dbg !8
  %1397 = bitcast float %1396 to i32, !dbg !8
  %1398 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1397), !dbg !8
  %1399 = select i1 %.not18, float %1395, float %1389, !dbg !8
  %1400 = bitcast float %1399 to i32, !dbg !8
  %1401 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1400), !dbg !8
  %1402 = extractvalue [2 x <8 x float>] %99, 1, !dbg !8
  %1403 = extractelement <8 x float> %1402, i64 0, !dbg !8
  %1404 = extractelement <8 x float> %1402, i64 1, !dbg !8
  %1405 = extractelement <8 x float> %1402, i64 2, !dbg !8
  %1406 = extractelement <8 x float> %1402, i64 3, !dbg !8
  %1407 = select i1 %.not, float %1404, float %1403, !dbg !8
  %1408 = bitcast float %1407 to i32, !dbg !8
  %1409 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1408), !dbg !8
  %1410 = bitcast i32 %1409 to float, !dbg !8
  %1411 = select i1 %.not, float %1403, float %1410, !dbg !8
  %1412 = select i1 %.not, float %1410, float %1404, !dbg !8
  %1413 = select i1 %.not, float %1406, float %1405, !dbg !8
  %1414 = bitcast float %1413 to i32, !dbg !8
  %1415 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1414), !dbg !8
  %1416 = bitcast i32 %1415 to float, !dbg !8
  %1417 = select i1 %.not, float %1405, float %1416, !dbg !8
  %1418 = select i1 %.not, float %1416, float %1406, !dbg !8
  %1419 = select i1 %.not18, float %1417, float %1411, !dbg !8
  %1420 = bitcast float %1419 to i32, !dbg !8
  %1421 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1420), !dbg !8
  %1422 = select i1 %.not18, float %1418, float %1412, !dbg !8
  %1423 = bitcast float %1422 to i32, !dbg !8
  %1424 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1423), !dbg !8
  %1425 = extractelement <8 x float> %1402, i64 4, !dbg !8
  %1426 = extractelement <8 x float> %1402, i64 5, !dbg !8
  %1427 = extractelement <8 x float> %1402, i64 6, !dbg !8
  %1428 = extractelement <8 x float> %1402, i64 7, !dbg !8
  %1429 = select i1 %.not, float %1426, float %1425, !dbg !8
  %1430 = bitcast float %1429 to i32, !dbg !8
  %1431 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1430), !dbg !8
  %1432 = bitcast i32 %1431 to float, !dbg !8
  %1433 = select i1 %.not, float %1425, float %1432, !dbg !8
  %1434 = select i1 %.not, float %1432, float %1426, !dbg !8
  %1435 = select i1 %.not, float %1428, float %1427, !dbg !8
  %1436 = bitcast float %1435 to i32, !dbg !8
  %1437 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1436), !dbg !8
  %1438 = bitcast i32 %1437 to float, !dbg !8
  %1439 = select i1 %.not, float %1427, float %1438, !dbg !8
  %1440 = select i1 %.not, float %1438, float %1428, !dbg !8
  %1441 = select i1 %.not18, float %1439, float %1433, !dbg !8
  %1442 = bitcast float %1441 to i32, !dbg !8
  %1443 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1442), !dbg !8
  %1444 = select i1 %.not18, float %1440, float %1434, !dbg !8
  %1445 = bitcast float %1444 to i32, !dbg !8
  %1446 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1445), !dbg !8
  %1447 = shufflevector <1 x float> %1012, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !17
  %1448 = shufflevector <1 x float> %1007, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !17
  %1449 = bitcast i32 %1443 to float, !dbg !8
  %1450 = select i1 %.not18, float %1433, float %1449, !dbg !8
  %1451 = insertelement <4 x float> poison, float %1450, i64 0, !dbg !17
  %1452 = bitcast i32 %1446 to float, !dbg !8
  %1453 = select i1 %.not18, float %1434, float %1452, !dbg !8
  %1454 = insertelement <4 x float> %1451, float %1453, i64 1, !dbg !17
  %1455 = select i1 %.not18, float %1449, float %1439, !dbg !8
  %1456 = insertelement <4 x float> %1454, float %1455, i64 2, !dbg !17
  %1457 = select i1 %.not18, float %1452, float %1440, !dbg !8
  %1458 = insertelement <4 x float> %1456, float %1457, i64 3, !dbg !17
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1448, !dbg !17
  %normalized = fmul <4 x float> %1458, %row.reciprocal, !dbg !17
  %1459 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !18
  %1460 = extractelement <4 x half> %1459, i64 3, !dbg !18
  %1461 = extractelement <4 x half> %1459, i64 2, !dbg !18
  %1462 = extractelement <4 x half> %1459, i64 1, !dbg !18
  %1463 = extractelement <4 x half> %1459, i64 0, !dbg !18
  %1464 = bitcast i32 %1421 to float, !dbg !8
  %1465 = select i1 %.not18, float %1411, float %1464, !dbg !8
  %1466 = insertelement <4 x float> poison, float %1465, i64 0, !dbg !17
  %1467 = bitcast i32 %1424 to float, !dbg !8
  %1468 = select i1 %.not18, float %1412, float %1467, !dbg !8
  %1469 = insertelement <4 x float> %1466, float %1468, i64 1, !dbg !17
  %1470 = select i1 %.not18, float %1464, float %1417, !dbg !8
  %1471 = insertelement <4 x float> %1469, float %1470, i64 2, !dbg !17
  %1472 = select i1 %.not18, float %1467, float %1418, !dbg !8
  %1473 = insertelement <4 x float> %1471, float %1472, i64 3, !dbg !17
  %normalized19 = fmul <4 x float> %1473, %row.reciprocal, !dbg !17
  %1474 = fptrunc <4 x float> %normalized19 to <4 x half>, !dbg !18
  %1475 = extractelement <4 x half> %1474, i64 3, !dbg !18
  %1476 = extractelement <4 x half> %1474, i64 2, !dbg !18
  %1477 = extractelement <4 x half> %1474, i64 1, !dbg !18
  %1478 = extractelement <4 x half> %1474, i64 0, !dbg !18
  %1479 = bitcast i32 %1398 to float, !dbg !8
  %1480 = select i1 %.not18, float %1388, float %1479, !dbg !8
  %1481 = insertelement <4 x float> poison, float %1480, i64 0, !dbg !17
  %1482 = bitcast i32 %1401 to float, !dbg !8
  %1483 = select i1 %.not18, float %1389, float %1482, !dbg !8
  %1484 = insertelement <4 x float> %1481, float %1483, i64 1, !dbg !17
  %1485 = select i1 %.not18, float %1479, float %1394, !dbg !8
  %1486 = insertelement <4 x float> %1484, float %1485, i64 2, !dbg !17
  %1487 = select i1 %.not18, float %1482, float %1395, !dbg !8
  %1488 = insertelement <4 x float> %1486, float %1487, i64 3, !dbg !17
  %row.reciprocal20 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1447, !dbg !17
  %normalized21 = fmul <4 x float> %1488, %row.reciprocal20, !dbg !17
  %1489 = fptrunc <4 x float> %normalized21 to <4 x half>, !dbg !18
  %1490 = extractelement <4 x half> %1489, i64 3, !dbg !18
  %1491 = extractelement <4 x half> %1489, i64 2, !dbg !18
  %1492 = extractelement <4 x half> %1489, i64 1, !dbg !18
  %1493 = extractelement <4 x half> %1489, i64 0, !dbg !18
  %1494 = bitcast i32 %1376 to float, !dbg !8
  %1495 = select i1 %.not18, float %1366, float %1494, !dbg !8
  %1496 = insertelement <4 x float> poison, float %1495, i64 0, !dbg !17
  %1497 = bitcast i32 %1379 to float, !dbg !8
  %1498 = select i1 %.not18, float %1367, float %1497, !dbg !8
  %1499 = insertelement <4 x float> %1496, float %1498, i64 1, !dbg !17
  %1500 = select i1 %.not18, float %1494, float %1372, !dbg !8
  %1501 = insertelement <4 x float> %1499, float %1500, i64 2, !dbg !17
  %1502 = select i1 %.not18, float %1497, float %1373, !dbg !8
  %1503 = insertelement <4 x float> %1501, float %1502, i64 3, !dbg !17
  %normalized22 = fmul <4 x float> %1503, %row.reciprocal20, !dbg !17
  %1504 = fptrunc <4 x float> %normalized22 to <4 x half>, !dbg !18
  %1505 = extractelement <4 x half> %1504, i64 3, !dbg !18
  %1506 = extractelement <4 x half> %1504, i64 2, !dbg !18
  %1507 = extractelement <4 x half> %1504, i64 1, !dbg !18
  %1508 = extractelement <4 x half> %1504, i64 0, !dbg !18
  %address.thread727 = call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #9, !dbg !19, !range !32, !frisk.register_lifetimes !1
  %address.local728 = zext nneg i32 %address.thread727 to i64, !dbg !9
  %address.local729 = add nuw i64 %32, %address.local728, !dbg !19
  %1509 = lshr i64 %address.local728, 2, !dbg !19
  %address.local731 = and i64 %1509, 48, !dbg !19
  %address.local732 = add i64 %address.local729, %address.local731, !dbg !19
  %address.local733 = and i64 %address.local728, 240, !dbg !19
  %address.local734 = sub i64 %address.local732, %address.local733, !dbg !19
  %address.local735 = add i64 %address.local734, %33, !dbg !19
  %address.local736 = shl nuw nsw i64 %address.local735, 7, !dbg !19
  %1510 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %46, !dbg !19
  %address.local738 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %1510, i64 %address.local736, !dbg !19
  %address.local740 = and i64 %1509, 12, !dbg !12
  %address.local741 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local738, i64 %address.local740, !dbg !19
  store half %1508, ptr addrspace(1) %address.local741, align 2, !dbg !19
  %address.local743 = or disjoint i64 %address.local740, 1, !dbg !19
  %address.local744 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local738, i64 %address.local743, !dbg !19
  store half %1507, ptr addrspace(1) %address.local744, align 2, !dbg !19
  %address.local746 = or disjoint i64 %address.local740, 2, !dbg !19
  %address.local747 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local738, i64 %address.local746, !dbg !19
  store half %1506, ptr addrspace(1) %address.local747, align 2, !dbg !19
  %address.local749 = or disjoint i64 %address.local740, 3, !dbg !19
  %address.local750 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local738, i64 %address.local749, !dbg !19
  store half %1505, ptr addrspace(1) %address.local750, align 2, !dbg !19
  %address.local752 = or disjoint i64 %address.local740, 16, !dbg !19
  %address.local753 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local738, i64 %address.local752, !dbg !19
  store half %1493, ptr addrspace(1) %address.local753, align 2, !dbg !19
  %address.local755 = or disjoint i64 %address.local740, 17, !dbg !19
  %address.local756 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local738, i64 %address.local755, !dbg !19
  store half %1492, ptr addrspace(1) %address.local756, align 2, !dbg !19
  %address.local758 = or disjoint i64 %address.local740, 18, !dbg !19
  %address.local759 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local738, i64 %address.local758, !dbg !19
  store half %1491, ptr addrspace(1) %address.local759, align 2, !dbg !19
  %address.local761 = or disjoint i64 %address.local740, 19, !dbg !19
  %address.local762 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local738, i64 %address.local761, !dbg !19
  store half %1490, ptr addrspace(1) %address.local762, align 2, !dbg !19
  %address.local766 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local738, i64 16384, !dbg !19
  %address.local767 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local766, i64 %address.local740, !dbg !19
  store half %1478, ptr addrspace(1) %address.local767, align 2, !dbg !19
  %address.local769 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local766, i64 %address.local743, !dbg !19
  store half %1477, ptr addrspace(1) %address.local769, align 2, !dbg !19
  %address.local771 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local766, i64 %address.local746, !dbg !19
  store half %1476, ptr addrspace(1) %address.local771, align 2, !dbg !19
  %address.local773 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local766, i64 %address.local749, !dbg !19
  store half %1475, ptr addrspace(1) %address.local773, align 2, !dbg !19
  %address.local775 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local766, i64 %address.local752, !dbg !19
  store half %1463, ptr addrspace(1) %address.local775, align 2, !dbg !19
  %address.local777 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local766, i64 %address.local755, !dbg !19
  store half %1462, ptr addrspace(1) %address.local777, align 2, !dbg !19
  %address.local779 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local766, i64 %address.local758, !dbg !19
  store half %1461, ptr addrspace(1) %address.local779, align 2, !dbg !19
  %address.local781 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local766, i64 %address.local761, !dbg !19
  store half %1460, ptr addrspace(1) %address.local781, align 2, !dbg !19
  %1511 = insertelement <4 x float> poison, float %1349, i64 0, !dbg !17
  %1512 = insertelement <4 x float> %1511, float %1355, i64 1, !dbg !17
  %1513 = insertelement <4 x float> %1512, float %1350, i64 2, !dbg !17
  %1514 = insertelement <4 x float> %1513, float %1356, i64 3, !dbg !17
  %normalized24 = fmul <4 x float> %1514, %row.reciprocal, !dbg !17
  %1515 = fptrunc <4 x float> %normalized24 to <4 x half>, !dbg !18
  %1516 = extractelement <4 x half> %1515, i64 3, !dbg !18
  %1517 = extractelement <4 x half> %1515, i64 2, !dbg !18
  %1518 = extractelement <4 x half> %1515, i64 1, !dbg !18
  %1519 = extractelement <4 x half> %1515, i64 0, !dbg !18
  %1520 = insertelement <4 x float> poison, float %1321, i64 0, !dbg !17
  %1521 = insertelement <4 x float> %1520, float %1327, i64 1, !dbg !17
  %1522 = insertelement <4 x float> %1521, float %1322, i64 2, !dbg !17
  %1523 = insertelement <4 x float> %1522, float %1328, i64 3, !dbg !17
  %normalized25 = fmul <4 x float> %1523, %row.reciprocal, !dbg !17
  %1524 = fptrunc <4 x float> %normalized25 to <4 x half>, !dbg !18
  %1525 = extractelement <4 x half> %1524, i64 3, !dbg !18
  %1526 = extractelement <4 x half> %1524, i64 2, !dbg !18
  %1527 = extractelement <4 x half> %1524, i64 1, !dbg !18
  %1528 = extractelement <4 x half> %1524, i64 0, !dbg !18
  %1529 = insertelement <4 x float> poison, float %1292, i64 0, !dbg !17
  %1530 = insertelement <4 x float> %1529, float %1298, i64 1, !dbg !17
  %1531 = insertelement <4 x float> %1530, float %1293, i64 2, !dbg !17
  %1532 = insertelement <4 x float> %1531, float %1299, i64 3, !dbg !17
  %normalized27 = fmul <4 x float> %1532, %row.reciprocal20, !dbg !17
  %1533 = fptrunc <4 x float> %normalized27 to <4 x half>, !dbg !18
  %1534 = extractelement <4 x half> %1533, i64 3, !dbg !18
  %1535 = extractelement <4 x half> %1533, i64 2, !dbg !18
  %1536 = extractelement <4 x half> %1533, i64 1, !dbg !18
  %1537 = extractelement <4 x half> %1533, i64 0, !dbg !18
  %1538 = insertelement <4 x float> poison, float %1264, i64 0, !dbg !17
  %1539 = insertelement <4 x float> %1538, float %1270, i64 1, !dbg !17
  %1540 = insertelement <4 x float> %1539, float %1265, i64 2, !dbg !17
  %1541 = insertelement <4 x float> %1540, float %1271, i64 3, !dbg !17
  %normalized28 = fmul <4 x float> %1541, %row.reciprocal20, !dbg !17
  %1542 = fptrunc <4 x float> %normalized28 to <4 x half>, !dbg !18
  %1543 = extractelement <4 x half> %1542, i64 3, !dbg !18
  %1544 = extractelement <4 x half> %1542, i64 2, !dbg !18
  %1545 = extractelement <4 x half> %1542, i64 1, !dbg !18
  %1546 = extractelement <4 x half> %1542, i64 0, !dbg !18
  %address.local796 = or disjoint i64 %address.local740, 32, !dbg !19
  %address.local797 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local738, i64 %address.local796, !dbg !19
  store half %1546, ptr addrspace(1) %address.local797, align 2, !dbg !19
  %address.local799 = or disjoint i64 %address.local740, 33, !dbg !19
  %address.local800 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local738, i64 %address.local799, !dbg !19
  store half %1545, ptr addrspace(1) %address.local800, align 2, !dbg !19
  %address.local802 = or disjoint i64 %address.local740, 34, !dbg !19
  %address.local803 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local738, i64 %address.local802, !dbg !19
  store half %1544, ptr addrspace(1) %address.local803, align 2, !dbg !19
  %address.local805 = or disjoint i64 %address.local740, 35, !dbg !19
  %address.local806 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local738, i64 %address.local805, !dbg !19
  store half %1543, ptr addrspace(1) %address.local806, align 2, !dbg !19
  %address.local808 = or disjoint i64 %address.local740, 48, !dbg !19
  %address.local809 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local738, i64 %address.local808, !dbg !19
  store half %1537, ptr addrspace(1) %address.local809, align 2, !dbg !19
  %address.local811 = or disjoint i64 %address.local740, 49, !dbg !19
  %address.local812 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local738, i64 %address.local811, !dbg !19
  store half %1536, ptr addrspace(1) %address.local812, align 2, !dbg !19
  %address.local814 = or disjoint i64 %address.local740, 50, !dbg !19
  %address.local815 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local738, i64 %address.local814, !dbg !19
  store half %1535, ptr addrspace(1) %address.local815, align 2, !dbg !19
  %address.local817 = or i64 %1509, 51, !dbg !19
  %address.local818 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local738, i64 %address.local817, !dbg !19
  store half %1534, ptr addrspace(1) %address.local818, align 2, !dbg !19
  %address.local823 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local766, i64 %address.local796, !dbg !19
  store half %1528, ptr addrspace(1) %address.local823, align 2, !dbg !19
  %address.local825 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local766, i64 %address.local799, !dbg !19
  store half %1527, ptr addrspace(1) %address.local825, align 2, !dbg !19
  %address.local827 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local766, i64 %address.local802, !dbg !19
  store half %1526, ptr addrspace(1) %address.local827, align 2, !dbg !19
  %address.local829 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local766, i64 %address.local805, !dbg !19
  store half %1525, ptr addrspace(1) %address.local829, align 2, !dbg !19
  %address.local831 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local766, i64 %address.local808, !dbg !19
  store half %1519, ptr addrspace(1) %address.local831, align 2, !dbg !19
  %address.local833 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local766, i64 %address.local811, !dbg !19
  store half %1518, ptr addrspace(1) %address.local833, align 2, !dbg !19
  %address.local835 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local766, i64 %address.local814, !dbg !19
  store half %1517, ptr addrspace(1) %address.local835, align 2, !dbg !19
  %address.local837 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local766, i64 %address.local817, !dbg !19
  store half %1516, ptr addrspace(1) %address.local837, align 2, !dbg !19
  %1547 = insertelement <4 x float> poison, float %1235, i64 0, !dbg !17
  %1548 = insertelement <4 x float> %1547, float %1241, i64 1, !dbg !17
  %1549 = insertelement <4 x float> %1548, float %1236, i64 2, !dbg !17
  %1550 = insertelement <4 x float> %1549, float %1242, i64 3, !dbg !17
  %normalized30 = fmul <4 x float> %1550, %row.reciprocal, !dbg !17
  %1551 = fptrunc <4 x float> %normalized30 to <4 x half>, !dbg !18
  %1552 = extractelement <4 x half> %1551, i64 3, !dbg !18
  %1553 = extractelement <4 x half> %1551, i64 2, !dbg !18
  %1554 = extractelement <4 x half> %1551, i64 1, !dbg !18
  %1555 = extractelement <4 x half> %1551, i64 0, !dbg !18
  %1556 = insertelement <4 x float> poison, float %1207, i64 0, !dbg !17
  %1557 = insertelement <4 x float> %1556, float %1213, i64 1, !dbg !17
  %1558 = insertelement <4 x float> %1557, float %1208, i64 2, !dbg !17
  %1559 = insertelement <4 x float> %1558, float %1214, i64 3, !dbg !17
  %normalized31 = fmul <4 x float> %1559, %row.reciprocal, !dbg !17
  %1560 = fptrunc <4 x float> %normalized31 to <4 x half>, !dbg !18
  %1561 = extractelement <4 x half> %1560, i64 3, !dbg !18
  %1562 = extractelement <4 x half> %1560, i64 2, !dbg !18
  %1563 = extractelement <4 x half> %1560, i64 1, !dbg !18
  %1564 = extractelement <4 x half> %1560, i64 0, !dbg !18
  %1565 = insertelement <4 x float> poison, float %1178, i64 0, !dbg !17
  %1566 = insertelement <4 x float> %1565, float %1184, i64 1, !dbg !17
  %1567 = insertelement <4 x float> %1566, float %1179, i64 2, !dbg !17
  %1568 = insertelement <4 x float> %1567, float %1185, i64 3, !dbg !17
  %normalized33 = fmul <4 x float> %1568, %row.reciprocal20, !dbg !17
  %1569 = fptrunc <4 x float> %normalized33 to <4 x half>, !dbg !18
  %1570 = extractelement <4 x half> %1569, i64 3, !dbg !18
  %1571 = extractelement <4 x half> %1569, i64 2, !dbg !18
  %1572 = extractelement <4 x half> %1569, i64 1, !dbg !18
  %1573 = extractelement <4 x half> %1569, i64 0, !dbg !18
  %1574 = insertelement <4 x float> poison, float %1150, i64 0, !dbg !17
  %1575 = insertelement <4 x float> %1574, float %1156, i64 1, !dbg !17
  %1576 = insertelement <4 x float> %1575, float %1151, i64 2, !dbg !17
  %1577 = insertelement <4 x float> %1576, float %1157, i64 3, !dbg !17
  %normalized34 = fmul <4 x float> %1577, %row.reciprocal20, !dbg !17
  %1578 = fptrunc <4 x float> %normalized34 to <4 x half>, !dbg !18
  %1579 = extractelement <4 x half> %1578, i64 3, !dbg !18
  %1580 = extractelement <4 x half> %1578, i64 2, !dbg !18
  %1581 = extractelement <4 x half> %1578, i64 1, !dbg !18
  %1582 = extractelement <4 x half> %1578, i64 0, !dbg !18
  %address.local852 = or disjoint i64 %address.local740, 64, !dbg !19
  %address.local853 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local738, i64 %address.local852, !dbg !19
  store half %1582, ptr addrspace(1) %address.local853, align 2, !dbg !19
  %address.local855 = or disjoint i64 %address.local740, 65, !dbg !19
  %address.local856 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local738, i64 %address.local855, !dbg !19
  store half %1581, ptr addrspace(1) %address.local856, align 2, !dbg !19
  %address.local858 = or disjoint i64 %address.local740, 66, !dbg !19
  %address.local859 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local738, i64 %address.local858, !dbg !19
  store half %1580, ptr addrspace(1) %address.local859, align 2, !dbg !19
  %address.local861 = or disjoint i64 %address.local740, 67, !dbg !19
  %address.local862 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local738, i64 %address.local861, !dbg !19
  store half %1579, ptr addrspace(1) %address.local862, align 2, !dbg !19
  %address.local864 = or disjoint i64 %address.local740, 80, !dbg !19
  %address.local865 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local738, i64 %address.local864, !dbg !19
  store half %1573, ptr addrspace(1) %address.local865, align 2, !dbg !19
  %address.local867 = or disjoint i64 %address.local740, 81, !dbg !19
  %address.local868 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local738, i64 %address.local867, !dbg !19
  store half %1572, ptr addrspace(1) %address.local868, align 2, !dbg !19
  %address.local870 = or disjoint i64 %address.local740, 82, !dbg !19
  %address.local871 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local738, i64 %address.local870, !dbg !19
  store half %1571, ptr addrspace(1) %address.local871, align 2, !dbg !19
  %address.local873 = or disjoint i64 %address.local740, 83, !dbg !19
  %address.local874 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local738, i64 %address.local873, !dbg !19
  store half %1570, ptr addrspace(1) %address.local874, align 2, !dbg !19
  %address.local879 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local766, i64 %address.local852, !dbg !19
  store half %1564, ptr addrspace(1) %address.local879, align 2, !dbg !19
  %address.local881 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local766, i64 %address.local855, !dbg !19
  store half %1563, ptr addrspace(1) %address.local881, align 2, !dbg !19
  %address.local883 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local766, i64 %address.local858, !dbg !19
  store half %1562, ptr addrspace(1) %address.local883, align 2, !dbg !19
  %address.local885 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local766, i64 %address.local861, !dbg !19
  store half %1561, ptr addrspace(1) %address.local885, align 2, !dbg !19
  %address.local887 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local766, i64 %address.local864, !dbg !19
  store half %1555, ptr addrspace(1) %address.local887, align 2, !dbg !19
  %address.local889 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local766, i64 %address.local867, !dbg !19
  store half %1554, ptr addrspace(1) %address.local889, align 2, !dbg !19
  %address.local891 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local766, i64 %address.local870, !dbg !19
  store half %1553, ptr addrspace(1) %address.local891, align 2, !dbg !19
  %address.local893 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local766, i64 %address.local873, !dbg !19
  store half %1552, ptr addrspace(1) %address.local893, align 2, !dbg !19
  %1583 = insertelement <4 x float> poison, float %1121, i64 0, !dbg !17
  %1584 = insertelement <4 x float> %1583, float %1127, i64 1, !dbg !17
  %1585 = insertelement <4 x float> %1584, float %1122, i64 2, !dbg !17
  %1586 = insertelement <4 x float> %1585, float %1128, i64 3, !dbg !17
  %normalized36 = fmul <4 x float> %1586, %row.reciprocal, !dbg !17
  %1587 = fptrunc <4 x float> %normalized36 to <4 x half>, !dbg !18
  %1588 = extractelement <4 x half> %1587, i64 3, !dbg !18
  %1589 = extractelement <4 x half> %1587, i64 2, !dbg !18
  %1590 = extractelement <4 x half> %1587, i64 1, !dbg !18
  %1591 = extractelement <4 x half> %1587, i64 0, !dbg !18
  %1592 = insertelement <4 x float> poison, float %1093, i64 0, !dbg !17
  %1593 = insertelement <4 x float> %1592, float %1099, i64 1, !dbg !17
  %1594 = insertelement <4 x float> %1593, float %1094, i64 2, !dbg !17
  %1595 = insertelement <4 x float> %1594, float %1100, i64 3, !dbg !17
  %normalized37 = fmul <4 x float> %1595, %row.reciprocal, !dbg !17
  %1596 = fptrunc <4 x float> %normalized37 to <4 x half>, !dbg !18
  %1597 = extractelement <4 x half> %1596, i64 3, !dbg !18
  %1598 = extractelement <4 x half> %1596, i64 2, !dbg !18
  %1599 = extractelement <4 x half> %1596, i64 1, !dbg !18
  %1600 = extractelement <4 x half> %1596, i64 0, !dbg !18
  %1601 = insertelement <4 x float> poison, float %1064, i64 0, !dbg !17
  %1602 = insertelement <4 x float> %1601, float %1070, i64 1, !dbg !17
  %1603 = insertelement <4 x float> %1602, float %1065, i64 2, !dbg !17
  %1604 = insertelement <4 x float> %1603, float %1071, i64 3, !dbg !17
  %normalized39 = fmul <4 x float> %1604, %row.reciprocal20, !dbg !17
  %1605 = fptrunc <4 x float> %normalized39 to <4 x half>, !dbg !18
  %1606 = extractelement <4 x half> %1605, i64 3, !dbg !18
  %1607 = extractelement <4 x half> %1605, i64 2, !dbg !18
  %1608 = extractelement <4 x half> %1605, i64 1, !dbg !18
  %1609 = extractelement <4 x half> %1605, i64 0, !dbg !18
  %1610 = insertelement <4 x float> poison, float %1036, i64 0, !dbg !17
  %1611 = insertelement <4 x float> %1610, float %1042, i64 1, !dbg !17
  %1612 = insertelement <4 x float> %1611, float %1037, i64 2, !dbg !17
  %1613 = insertelement <4 x float> %1612, float %1043, i64 3, !dbg !17
  %normalized40 = fmul <4 x float> %1613, %row.reciprocal20, !dbg !17
  %1614 = fptrunc <4 x float> %normalized40 to <4 x half>, !dbg !18
  %1615 = extractelement <4 x half> %1614, i64 3, !dbg !18
  %1616 = extractelement <4 x half> %1614, i64 2, !dbg !18
  %1617 = extractelement <4 x half> %1614, i64 1, !dbg !18
  %1618 = extractelement <4 x half> %1614, i64 0, !dbg !18
  %address.local908 = or disjoint i64 %address.local740, 96, !dbg !19
  %address.local909 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local738, i64 %address.local908, !dbg !19
  store half %1618, ptr addrspace(1) %address.local909, align 2, !dbg !19
  %address.local911 = or disjoint i64 %address.local740, 97, !dbg !19
  %address.local912 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local738, i64 %address.local911, !dbg !19
  store half %1617, ptr addrspace(1) %address.local912, align 2, !dbg !19
  %address.local914 = or disjoint i64 %address.local740, 98, !dbg !19
  %address.local915 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local738, i64 %address.local914, !dbg !19
  store half %1616, ptr addrspace(1) %address.local915, align 2, !dbg !19
  %address.local917 = or disjoint i64 %address.local740, 99, !dbg !19
  %address.local918 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local738, i64 %address.local917, !dbg !19
  store half %1615, ptr addrspace(1) %address.local918, align 2, !dbg !19
  %address.local920 = or disjoint i64 %address.local740, 112, !dbg !19
  %address.local921 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local738, i64 %address.local920, !dbg !19
  store half %1609, ptr addrspace(1) %address.local921, align 2, !dbg !19
  %address.local923 = or disjoint i64 %address.local740, 113, !dbg !19
  %address.local924 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local738, i64 %address.local923, !dbg !19
  store half %1608, ptr addrspace(1) %address.local924, align 2, !dbg !19
  %address.local926 = or disjoint i64 %address.local740, 114, !dbg !19
  %address.local927 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local738, i64 %address.local926, !dbg !19
  store half %1607, ptr addrspace(1) %address.local927, align 2, !dbg !19
  %address.local929 = or i64 %1509, 115, !dbg !19
  %address.local930 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local738, i64 %address.local929, !dbg !19
  store half %1606, ptr addrspace(1) %address.local930, align 2, !dbg !19
  %address.local935 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local766, i64 %address.local908, !dbg !19
  store half %1600, ptr addrspace(1) %address.local935, align 2, !dbg !19
  %address.local937 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local766, i64 %address.local911, !dbg !19
  store half %1599, ptr addrspace(1) %address.local937, align 2, !dbg !19
  %address.local939 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local766, i64 %address.local914, !dbg !19
  store half %1598, ptr addrspace(1) %address.local939, align 2, !dbg !19
  %address.local941 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local766, i64 %address.local917, !dbg !19
  store half %1597, ptr addrspace(1) %address.local941, align 2, !dbg !19
  %address.local943 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local766, i64 %address.local920, !dbg !19
  store half %1591, ptr addrspace(1) %address.local943, align 2, !dbg !19
  %address.local945 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local766, i64 %address.local923, !dbg !19
  store half %1590, ptr addrspace(1) %address.local945, align 2, !dbg !19
  %address.local947 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local766, i64 %address.local926, !dbg !19
  store half %1589, ptr addrspace(1) %address.local947, align 2, !dbg !19
  %address.local949 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local766, i64 %address.local929, !dbg !19
  store half %1588, ptr addrspace(1) %address.local949, align 2, !dbg !19
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
