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
  %address.local404 = shl nuw nsw i64 %address.local, 3, !dbg !9
  %address.local405 = and i64 %address.local404, 1536, !dbg !9
  %address.local406 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local405, !dbg !9
  %address.local407 = ptrtoint ptr addrspace(3) %address.local406 to i64, !dbg !9
  %address.local408 = trunc i64 %address.local407 to i32, !dbg !9
  %36 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local408), !dbg !9
  %address.local409 = shl nuw nsw i64 %address.local, 7, !dbg !9
  %address.local410 = and i64 %address.local409, 16256, !dbg !9
  %address.local411 = lshr i64 %address.local, 4, !dbg !9
  %address.local412 = and i64 %address.local411, 8, !dbg !9
  %address.local413 = or disjoint i64 %address.local410, %address.local412, !dbg !9
  %37 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !9
  %.idx1360 = shl i64 %34, 8, !dbg !9
  %38 = add i64 %.idx, %.idx1360, !dbg !9
  %address.local414.idx = shl nuw nsw i64 %address.local413, 1, !dbg !9
  %39 = or disjoint i64 %38, %address.local414.idx, !dbg !9
  %dma.buffer.offset1358 = trunc i64 %39 to i32, !dbg !9
  %40 = insertelement <2 x i64> poison, i64 %37, i64 0, !dbg !9
  %41 = bitcast <2 x i64> %40 to <4 x i32>, !dbg !9
  %dma.buffer.rsrc1359 = shufflevector <4 x i32> %41, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !9
  %42 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %36, i32 %dma.buffer.offset1358, <4 x i32> %dma.buffer.rsrc1359) #10, !dbg !9
  %address.local415 = or disjoint i64 %address.local, 256, !dbg !9
  %address.local416 = shl nuw nsw i64 %address.local415, 3, !dbg !9
  %address.local417 = and i64 %address.local416, 3584, !dbg !9
  %address.local418 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local417, !dbg !9
  %address.local419 = ptrtoint ptr addrspace(3) %address.local418 to i64, !dbg !9
  %address.local420 = trunc i64 %address.local419 to i32, !dbg !9
  %43 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local420), !dbg !9
  %address.local423 = lshr i64 %address.local415, 4, !dbg !9
  %address.local424 = and i64 %address.local423, 24, !dbg !9
  %address.local425 = or disjoint i64 %address.local410, %address.local424, !dbg !9
  %address.local426.idx = shl nuw nsw i64 %address.local425, 1, !dbg !9
  %44 = or disjoint i64 %38, %address.local426.idx, !dbg !9
  %dma.buffer.offset1356 = trunc i64 %44 to i32, !dbg !9
  %45 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %43, i32 %dma.buffer.offset1356, <4 x i32> %dma.buffer.rsrc1359) #10, !dbg !9
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %46 = shl nuw nsw i64 %31, 19, !dbg !9
  %47 = add i64 %46, %35, !dbg !9
  %address.local427 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local405, !dbg !9
  %address.local428 = ptrtoint ptr addrspace(3) %address.local427 to i64, !dbg !9
  %address.local429 = trunc i64 %address.local428 to i32, !dbg !9
  %48 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local429), !dbg !9
  %49 = or disjoint i64 %47, 32, !dbg !9
  %50 = or disjoint i64 %address.local413, %49, !dbg !9
  %.tr = trunc i64 %50 to i32, !dbg !9
  %dma.buffer.offset1354 = shl i32 %.tr, 1, !dbg !9
  %51 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %48, i32 %dma.buffer.offset1354, <4 x i32> %dma.buffer.rsrc1359) #10, !dbg !9
  %address.local431 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local417, !dbg !9
  %address.local432 = ptrtoint ptr addrspace(3) %address.local431 to i64, !dbg !9
  %address.local433 = trunc i64 %address.local432 to i32, !dbg !9
  %52 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local433), !dbg !9
  %53 = or disjoint i64 %address.local425, %49, !dbg !9
  %.tr1364 = trunc i64 %53 to i32, !dbg !9
  %dma.buffer.offset1352 = shl i32 %.tr1364, 1, !dbg !9
  %54 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %52, i32 %dma.buffer.offset1352, <4 x i32> %dma.buffer.rsrc1359) #10, !dbg !9
  %address.local435 = lshr i32 %address.thread, 2, !dbg !12
  %address.local436 = and i32 %address.local435, 4, !dbg !12
  %address.local437 = shl nuw nsw i32 %address.thread, 1, !dbg !12
  %address.local438 = and i32 %address.local437, 384, !dbg !12
  %address.local439 = or disjoint i32 %address.local436, %address.local438, !dbg !12
  %address.local440 = shl nuw nsw i32 %address.thread, 3, !dbg !12
  %address.local441 = and i32 %address.local440, 120, !dbg !12
  %address.local442 = or disjoint i32 %address.local439, %address.local441, !dbg !12
  %address.local443 = shl nuw nsw i32 %address.thread, 5, !dbg !12
  %address.local444 = and i32 %address.local443, 1024, !dbg !12
  %address.local445 = or disjoint i32 %address.local442, %address.local444, !dbg !12
  %address.local446 = zext nneg i32 %address.local445 to i64, !dbg !12
  %address.local447 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local446, !dbg !12
  %55 = load <4 x half>, ptr addrspace(3) %address.local447, align 8, !dbg !12
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local447, i64 4096, !dbg !12
  %56 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !12
  %57 = fmul <4 x half> %55, splat (half 0xH3015), !dbg !12
  %lds.const.gep1181 = getelementptr i8, ptr addrspace(3) %address.local447, i64 1024, !dbg !12
  %58 = load <4 x half>, ptr addrspace(3) %lds.const.gep1181, align 8, !dbg !12
  %59 = fmul <4 x half> %56, splat (half 0xH3015), !dbg !12
  %lds.const.gep1182 = getelementptr i8, ptr addrspace(3) %address.local447, i64 5120, !dbg !12
  %60 = load <4 x half>, ptr addrspace(3) %lds.const.gep1182, align 8, !dbg !12
  %61 = fmul <4 x half> %58, splat (half 0xH3015), !dbg !12
  %62 = fmul <4 x half> %60, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %63 = or disjoint i64 %47, 64, !dbg !9
  %64 = or disjoint i64 %address.local413, %63, !dbg !9
  %.tr1366 = trunc i64 %64 to i32, !dbg !9
  %dma.buffer.offset1350 = shl i32 %.tr1366, 1, !dbg !9
  %65 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %36, i32 %dma.buffer.offset1350, <4 x i32> %dma.buffer.rsrc1359) #10, !dbg !9
  %66 = or disjoint i64 %address.local425, %63, !dbg !9
  %.tr1368 = trunc i64 %66 to i32, !dbg !9
  %dma.buffer.offset1348 = shl i32 %.tr1368, 1, !dbg !9
  %67 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %43, i32 %dma.buffer.offset1348, <4 x i32> %dma.buffer.rsrc1359) #10, !dbg !9
  %lds.const.gep1183 = getelementptr i8, ptr addrspace(3) %address.local447, i64 8192, !dbg !12
  %68 = load <4 x half>, ptr addrspace(3) %lds.const.gep1183, align 8, !dbg !12
  %lds.const.gep1184 = getelementptr i8, ptr addrspace(3) %address.local447, i64 12288, !dbg !12
  %69 = load <4 x half>, ptr addrspace(3) %lds.const.gep1184, align 8, !dbg !12
  %lds.const.gep1185 = getelementptr i8, ptr addrspace(3) %address.local447, i64 9216, !dbg !12
  %70 = load <4 x half>, ptr addrspace(3) %lds.const.gep1185, align 8, !dbg !12
  %71 = fmul <4 x half> %68, splat (half 0xH3015), !dbg !12
  %lds.const.gep1186 = getelementptr i8, ptr addrspace(3) %address.local447, i64 13312, !dbg !12
  %72 = load <4 x half>, ptr addrspace(3) %lds.const.gep1186, align 8, !dbg !12
  %73 = fmul <4 x half> %69, splat (half 0xH3015), !dbg !12
  %74 = fmul <4 x half> %70, splat (half 0xH3015), !dbg !12
  %75 = fmul <4 x half> %72, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %76 = or disjoint i64 %47, 96, !dbg !9
  %77 = or disjoint i64 %address.local413, %76, !dbg !9
  %.tr1370 = trunc i64 %77 to i32, !dbg !9
  %dma.buffer.offset1346 = shl i32 %.tr1370, 1, !dbg !9
  %78 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %48, i32 %dma.buffer.offset1346, <4 x i32> %dma.buffer.rsrc1359) #10, !dbg !9
  %79 = or disjoint i64 %address.local425, %76, !dbg !9
  %.tr1372 = trunc i64 %79 to i32, !dbg !9
  %dma.buffer.offset1344 = shl i32 %.tr1372, 1, !dbg !9
  %80 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %52, i32 %dma.buffer.offset1344, <4 x i32> %dma.buffer.rsrc1359) #10, !dbg !9
  %81 = load <4 x half>, ptr addrspace(3) %address.local447, align 8, !dbg !12
  %82 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !12
  %83 = load <4 x half>, ptr addrspace(3) %lds.const.gep1181, align 8, !dbg !12
  %84 = fmul <4 x half> %81, splat (half 0xH3015), !dbg !12
  %85 = load <4 x half>, ptr addrspace(3) %lds.const.gep1182, align 8, !dbg !12
  %86 = fmul <4 x half> %82, splat (half 0xH3015), !dbg !12
  %87 = fmul <4 x half> %83, splat (half 0xH3015), !dbg !12
  %88 = fmul <4 x half> %85, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %89 = load <4 x half>, ptr addrspace(3) %lds.const.gep1183, align 8, !dbg !12
  %90 = load <4 x half>, ptr addrspace(3) %lds.const.gep1184, align 8, !dbg !12
  %91 = load <4 x half>, ptr addrspace(3) %lds.const.gep1185, align 8, !dbg !12
  %92 = fmul <4 x half> %89, splat (half 0xH3015), !dbg !12
  %93 = load <4 x half>, ptr addrspace(3) %lds.const.gep1186, align 8, !dbg !12
  %94 = fmul <4 x half> %90, splat (half 0xH3015), !dbg !12
  %95 = fmul <4 x half> %91, splat (half 0xH3015), !dbg !12
  %96 = fmul <4 x half> %93, splat (half 0xH3015), !dbg !12
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %97 = shl nuw nsw i32 %kdma.tid, 2, !dbg !15
  %98 = and i32 %97, 3084, !dbg !15
  %99 = and i32 %kdma.tid, 1, !dbg !15
  %100 = or disjoint i32 %98, %99, !dbg !15
  %narrow = add nuw nsw i32 %100, 1, !dbg !15
  %101 = lshr i32 %narrow, 1, !dbg !15
  %102 = zext nneg i32 %101 to i64, !dbg !15
  %k0.prime. = shl i64 %25, 19, !dbg !13
  %k0.prime.1923 = and i64 %k0.prime., 16252928, !dbg !13
  %k0.prime.1924 = shl nuw nsw i64 0, 14, !dbg !13
  %k0.prime.1925 = or disjoint i64 %k0.prime.1923, %k0.prime.1924, !dbg !13
  %k0.prime.kdma.source = zext nneg i32 %kdma.owner to i64, !dbg !13
  %k0.prime.kdma.source1466 = lshr i64 %k0.prime.kdma.source, 4, !dbg !13
  %k0.prime.kdma.source1467 = and i64 %k0.prime.kdma.source1466, 8, !dbg !13
  %k0.prime.kdma.source1468 = and i64 %k0.prime.kdma.source, 91, !dbg !13
  %k0.prime.kdma.source1469 = lshr i64 %k0.prime.kdma.source, 2, !dbg !13
  %k0.prime.kdma.source1470 = and i64 %k0.prime.kdma.source1469, 4, !dbg !13
  %k0.prime.kdma.source1471 = add nuw nsw i64 %k0.prime.kdma.source1470, %k0.prime.kdma.source, !dbg !13
  %k0.prime.kdma.source1472 = and i64 %k0.prime.kdma.source1471, 4, !dbg !13
  %k0.prime.kdma.source1473 = or disjoint i64 %k0.prime.kdma.source1468, %k0.prime.kdma.source1472, !dbg !13
  %k0.prime.kdma.source1474 = shl nuw nsw i64 %k0.prime.kdma.source1473, 7, !dbg !13
  %k0.prime.kdma.source1475 = or disjoint i64 %k0.prime.kdma.source1467, %k0.prime.kdma.source1474, !dbg !13
  %k0.prime.kdma.source1476 = or disjoint i64 %k0.prime.1925, %k0.prime.kdma.source1475, !dbg !13
  %k0.prime.kdma.source1477 = trunc nuw i64 %k0.prime.kdma.source1476 to i32, !dbg !13
  %k0.prime.kdma.source1478 = shl nuw nsw i32 %k0.prime.kdma.source1477, 1, !dbg !13
  %k0.prime.kdma.offset = or disjoint i32 %k0.prime.kdma.source1478, %kdma.intra, !dbg !13
  %k0.prime.1926 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !13
  %k0.prime.1927 = insertelement <2 x i64> poison, i64 %k0.prime.1926, i64 0, !dbg !13
  %k0.prime.1928 = bitcast <2 x i64> %k0.prime.1927 to <4 x i32>, !dbg !13
  %k0.prime.dma.buffer.rsrc1343 = shufflevector <4 x i32> %k0.prime.1928, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !13
  %k0.prime.1929 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %36, i32 %k0.prime.kdma.offset, <4 x i32> %k0.prime.dma.buffer.rsrc1343) #10, !dbg !13, !frisk.cross_bn_k0 !1
  %k0.prime.kdma.lds1494 = add i32 %36, 512, !dbg !13
  %k0.prime.kdma.source1479 = zext nneg i32 %kdma.owner.hi to i64, !dbg !13
  %k0.prime.kdma.source1480 = lshr i64 %k0.prime.kdma.source1479, 4, !dbg !13
  %k0.prime.kdma.source1481 = and i64 %k0.prime.kdma.source1480, 8, !dbg !13
  %k0.prime.kdma.source1482 = and i64 %k0.prime.kdma.source1479, 123, !dbg !13
  %k0.prime.kdma.source1483 = lshr i64 %k0.prime.kdma.source1479, 2, !dbg !13
  %k0.prime.kdma.source1484 = and i64 %k0.prime.kdma.source1483, 4, !dbg !13
  %k0.prime.kdma.source1485 = add nuw nsw i64 %k0.prime.kdma.source1484, %k0.prime.kdma.source1479, !dbg !13
  %k0.prime.kdma.source1486 = and i64 %k0.prime.kdma.source1485, 4, !dbg !13
  %k0.prime.kdma.source1487 = or disjoint i64 %k0.prime.kdma.source1482, %k0.prime.kdma.source1486, !dbg !13
  %k0.prime.kdma.source1488 = shl nuw nsw i64 %k0.prime.kdma.source1487, 7, !dbg !13
  %k0.prime.kdma.source1489 = or disjoint i64 %k0.prime.kdma.source1481, %k0.prime.kdma.source1488, !dbg !13
  %k0.prime.kdma.source1490 = or disjoint i64 %k0.prime.1925, %k0.prime.kdma.source1489, !dbg !13
  %k0.prime.kdma.source1491 = trunc nuw i64 %k0.prime.kdma.source1490 to i32, !dbg !13
  %k0.prime.kdma.source1492 = shl nuw nsw i32 %k0.prime.kdma.source1491, 1, !dbg !13
  %k0.prime.kdma.offset1493 = or disjoint i32 %k0.prime.kdma.source1492, %kdma.intra.hi, !dbg !13
  %k0.prime.1930 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.kdma.lds1494, i32 %k0.prime.kdma.offset1493, <4 x i32> %k0.prime.dma.buffer.rsrc1343) #10, !dbg !13, !frisk.cross_bn_k0 !1
  %k0.prime.kdma.source1506 = or disjoint i64 %k0.prime.kdma.source1475, 16, !dbg !13
  %k0.prime.kdma.source1507 = or disjoint i64 %k0.prime.1925, %k0.prime.kdma.source1506, !dbg !13
  %k0.prime.kdma.source1508 = trunc nuw i64 %k0.prime.kdma.source1507 to i32, !dbg !13
  %k0.prime.kdma.source1509 = shl nuw nsw i32 %k0.prime.kdma.source1508, 1, !dbg !13
  %k0.prime.kdma.offset1510 = or disjoint i32 %k0.prime.kdma.source1509, %kdma.intra, !dbg !13
  %k0.prime.1931 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %43, i32 %k0.prime.kdma.offset1510, <4 x i32> %k0.prime.dma.buffer.rsrc1343) #10, !dbg !13, !frisk.cross_bn_k0 !1
  %k0.prime.kdma.lds1528 = add i32 %43, 512, !dbg !13
  %k0.prime.kdma.source1523 = or disjoint i64 %k0.prime.kdma.source1489, 16, !dbg !13
  %k0.prime.kdma.source1524 = or disjoint i64 %k0.prime.1925, %k0.prime.kdma.source1523, !dbg !13
  %k0.prime.kdma.source1525 = trunc nuw i64 %k0.prime.kdma.source1524 to i32, !dbg !13
  %k0.prime.kdma.source1526 = shl nuw nsw i32 %k0.prime.kdma.source1525, 1, !dbg !13
  %k0.prime.kdma.offset1527 = or disjoint i32 %k0.prime.kdma.source1526, %kdma.intra.hi, !dbg !13
  %k0.prime.1932 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.kdma.lds1528, i32 %k0.prime.kdma.offset1527, <4 x i32> %k0.prime.dma.buffer.rsrc1343) #10, !dbg !13, !frisk.cross_bn_k0 !1
  br label %103, !dbg !8

103:                                              ; preds = %k0.join, %4
  %104 = phi i64 [ %925, %k0.join ], [ 0, %4 ], !dbg !8
  %105 = phi [2 x <8 x float>] [ %1003, %k0.join ], [ zeroinitializer, %4 ], !dbg !8
  %106 = phi [2 x <1 x float>] [ %768, %k0.join ], [ zeroinitializer, %4 ], !dbg !8
  %107 = phi [2 x <1 x float>] [ %1005, %k0.join ], [ [<1 x float> splat (float 0xFFF0000000000000), <1 x float> splat (float 0xFFF0000000000000)], %4 ], !dbg !8
  %108 = phi [2 x <8 x float>] [ %1009, %k0.join ], [ zeroinitializer, %4 ], !dbg !8
  %109 = phi [2 x <8 x float>] [ %1013, %k0.join ], [ zeroinitializer, %4 ], !dbg !8
  %110 = phi [2 x <8 x float>] [ %1017, %k0.join ], [ zeroinitializer, %4 ], !dbg !8
  %111 = icmp samesign ult i64 %104, 32, !dbg !8
  br i1 %111, label %112, label %1018, !dbg !8

112:                                              ; preds = %103
  %113 = shl i64 %25, 19, !dbg !13
  %114 = and i64 %113, 16252928, !dbg !13
  %115 = shl nuw nsw i64 %104, 14, !dbg !13
  %116 = or disjoint i64 %114, %115, !dbg !13
  %address.thread498 = call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #9, !dbg !13, !range !32, !frisk.register_lifetimes !1
  %address.local499 = zext nneg i32 %address.thread498 to i64, !dbg !9
  %117 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !13
  %118 = insertelement <2 x i64> poison, i64 %117, i64 0, !dbg !13
  %119 = bitcast <2 x i64> %118 to <4 x i32>, !dbg !13
  %dma.buffer.rsrc1343 = shufflevector <4 x i32> %119, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !13
  %kdma.source = zext nneg i32 %kdma.owner to i64, !dbg !13
  %kdma.source1466 = lshr i64 %kdma.source, 4, !dbg !13
  %kdma.source1467 = and i64 %kdma.source1466, 8, !dbg !13
  %kdma.source1468 = and i64 %kdma.source, 91, !dbg !13
  %kdma.source1469 = lshr i64 %kdma.source, 2, !dbg !13
  %kdma.source1470 = and i64 %kdma.source1469, 4, !dbg !13
  %kdma.source1471 = add nuw nsw i64 %kdma.source1470, %kdma.source, !dbg !13
  %kdma.source1472 = and i64 %kdma.source1471, 4, !dbg !13
  %kdma.source1473 = or disjoint i64 %kdma.source1468, %kdma.source1472, !dbg !13
  %kdma.source1474 = shl nuw nsw i64 %kdma.source1473, 7, !dbg !13
  %kdma.source1475 = or disjoint i64 %kdma.source1467, %kdma.source1474, !dbg !13
  %kdma.source1476 = or disjoint i64 %116, %kdma.source1475, !dbg !13
  %kdma.source1477 = trunc nuw i64 %kdma.source1476 to i32, !dbg !13
  %kdma.source1478 = shl nuw nsw i32 %kdma.source1477, 1, !dbg !13
  %kdma.offset = or disjoint i32 %kdma.source1478, %kdma.intra, !dbg !13
  %kdma.source1479 = zext nneg i32 %kdma.owner.hi to i64, !dbg !13
  %kdma.source1480 = lshr i64 %kdma.source1479, 4, !dbg !13
  %kdma.source1481 = and i64 %kdma.source1480, 8, !dbg !13
  %kdma.source1482 = and i64 %kdma.source1479, 123, !dbg !13
  %kdma.source1483 = lshr i64 %kdma.source1479, 2, !dbg !13
  %kdma.source1484 = and i64 %kdma.source1483, 4, !dbg !13
  %kdma.source1485 = add nuw nsw i64 %kdma.source1484, %kdma.source1479, !dbg !13
  %kdma.source1486 = and i64 %kdma.source1485, 4, !dbg !13
  %kdma.source1487 = or disjoint i64 %kdma.source1482, %kdma.source1486, !dbg !13
  %kdma.source1488 = shl nuw nsw i64 %kdma.source1487, 7, !dbg !13
  %kdma.source1489 = or disjoint i64 %kdma.source1481, %kdma.source1488, !dbg !13
  %kdma.source1490 = or disjoint i64 %116, %kdma.source1489, !dbg !13
  %kdma.source1491 = trunc nuw i64 %kdma.source1490 to i32, !dbg !13
  %kdma.source1492 = shl nuw nsw i32 %kdma.source1491, 1, !dbg !13
  %kdma.offset1493 = or disjoint i32 %kdma.source1492, %kdma.intra.hi, !dbg !13
  %kdma.lds1494 = add i32 %36, 512, !dbg !13
  %kdma.source1506 = or disjoint i64 %kdma.source1475, 16, !dbg !13
  %kdma.source1507 = or disjoint i64 %116, %kdma.source1506, !dbg !13
  %kdma.source1508 = trunc nuw i64 %kdma.source1507 to i32, !dbg !13
  %kdma.source1509 = shl nuw nsw i32 %kdma.source1508, 1, !dbg !13
  %kdma.offset1510 = or disjoint i32 %kdma.source1509, %kdma.intra, !dbg !13
  %kdma.source1523 = or disjoint i64 %kdma.source1489, 16, !dbg !13
  %kdma.source1524 = or disjoint i64 %116, %kdma.source1523, !dbg !13
  %kdma.source1525 = trunc nuw i64 %kdma.source1524 to i32, !dbg !13
  %kdma.source1526 = shl nuw nsw i32 %kdma.source1525, 1, !dbg !13
  %kdma.offset1527 = or disjoint i32 %kdma.source1526, %kdma.intra.hi, !dbg !13
  %kdma.lds1528 = add i32 %43, 512, !dbg !13
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %120 = or disjoint i64 %116, 32, !dbg !13
  %kdma.source1540 = or disjoint i64 %120, %kdma.source1475, !dbg !13
  %kdma.source1541 = trunc nuw i64 %kdma.source1540 to i32, !dbg !13
  %kdma.source1542 = shl nuw nsw i32 %kdma.source1541, 1, !dbg !13
  %kdma.offset1543 = or disjoint i32 %kdma.source1542, %kdma.intra, !dbg !13
  %121 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %48, i32 %kdma.offset1543, <4 x i32> %dma.buffer.rsrc1343) #10, !dbg !13
  %kdma.source1556 = or disjoint i64 %120, %kdma.source1489, !dbg !13
  %kdma.source1557 = trunc nuw i64 %kdma.source1556 to i32, !dbg !13
  %kdma.source1558 = shl nuw nsw i32 %kdma.source1557, 1, !dbg !13
  %kdma.offset1559 = or disjoint i32 %kdma.source1558, %kdma.intra.hi, !dbg !13
  %kdma.lds1560 = add i32 %48, 512, !dbg !13
  %122 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1560, i32 %kdma.offset1559, <4 x i32> %dma.buffer.rsrc1343) #10, !dbg !13
  %kdma.source1573 = or disjoint i64 %120, %kdma.source1506, !dbg !13
  %kdma.source1574 = trunc nuw i64 %kdma.source1573 to i32, !dbg !13
  %kdma.source1575 = shl nuw nsw i32 %kdma.source1574, 1, !dbg !13
  %kdma.offset1576 = or disjoint i32 %kdma.source1575, %kdma.intra, !dbg !13
  %123 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %52, i32 %kdma.offset1576, <4 x i32> %dma.buffer.rsrc1343) #10, !dbg !13
  %kdma.source1590 = or disjoint i64 %120, %kdma.source1523, !dbg !13
  %kdma.source1591 = trunc nuw i64 %kdma.source1590 to i32, !dbg !13
  %kdma.source1592 = shl nuw nsw i32 %kdma.source1591, 1, !dbg !13
  %kdma.offset1593 = or disjoint i32 %kdma.source1592, %kdma.intra.hi, !dbg !13
  %kdma.lds1594 = add i32 %52, 512, !dbg !13
  %124 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1594, i32 %kdma.offset1593, <4 x i32> %dma.buffer.rsrc1343) #10, !dbg !13
  %address.local526 = lshr i64 %address.local499, 2, !dbg !15
  %address.local529 = shl nuw nsw i64 %address.local499, 1, !dbg !15
  %address.local530 = and i64 %address.local529, 24, !dbg !15
  %address.local532 = shl i64 %address.local499, 62, !dbg !15
  %address.local533 = ashr exact i64 %address.local532, 57, !dbg !15
  %address.local534 = and i64 %address.local533, -64, !dbg !15
  %address.local537 = and i64 %address.local526, 12, !dbg !12
  %address.local539 = and i64 %address.local499, 63, !dbg !12
  %address.local540 = lshr i64 %address.local539, 5, !dbg !12
  %address.local541 = mul nuw nsw i64 %address.local540, 1016, !dbg !12
  %.idx1381 = mul nuw nsw i64 %address.local499, 320, !dbg !15
  %125 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %.idx1381, !dbg !15
  %.idx1382 = mul nsw i64 %address.local526, -1280, !dbg !15
  %126 = getelementptr i8, ptr addrspace(3) %125, i64 %.idx1382, !dbg !15
  %127 = getelementptr [2 x i8], ptr addrspace(3) %126, i64 %address.local530, !dbg !15
  %128 = getelementptr [2 x i8], ptr addrspace(3) %127, i64 %address.local534, !dbg !15
  %129 = getelementptr [2 x i8], ptr addrspace(3) %128, i64 %address.local537, !dbg !15
  %address.local543 = getelementptr [2 x i8], ptr addrspace(3) %129, i64 %address.local541, !dbg !15
  %130 = add nsw i64 %.idx1381, %.idx1382, !dbg !15
  %.idx1853 = shl nuw nsw i64 %address.local530, 1, !dbg !15
  %131 = or disjoint i64 %130, %.idx1853, !dbg !15
  %132 = or disjoint i64 %address.local534, %address.local537, !dbg !15
  %133 = shl nsw i64 %132, 1, !dbg !15
  %134 = add nsw i64 %133, %131, !dbg !15
  %address.local543.idx = mul nuw nsw i64 %address.local540, 2032, !dbg !15
  %135 = add nsw i64 %134, %address.local543.idx, !dbg !15
  %136 = lshr i64 %135, 6, !dbg !15
  %137 = and i64 %136, 8, !dbg !15
  %138 = xor i64 %135, %137, !dbg !15
  %139 = sub i64 %138, %135, !dbg !15
  %kdma.swizzled = getelementptr i8, ptr addrspace(3) %address.local543, i64 %139, !dbg !15
  %140 = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8, !dbg !15
  %141 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %140, <4 x float> zeroinitializer) #9, !dbg !15
  %kdma.gep = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4096, !dbg !15
  %142 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8, !dbg !15
  %address.local554 = shl nuw nsw i32 %address.thread498, 1, !dbg !15
  %address.local555 = and i32 %address.local554, 24, !dbg !15
  %address.local556 = shl nuw nsw i32 %address.thread498, 8, !dbg !15
  %address.local557 = and i32 %address.local556, 768, !dbg !15
  %address.local558 = or disjoint i32 %address.local555, %address.local557, !dbg !15
  %address.local559 = shl nuw nsw i32 %address.thread498, 5, !dbg !15
  %address.local560 = and i32 %address.local559, 96, !dbg !15
  %address.local561 = or disjoint i32 %address.local558, %address.local560, !dbg !15
  %address.local562 = zext nneg i32 %address.local561 to i64, !dbg !15
  %address.base = call i64 asm sideeffect "", "=v,0"(i64 %102) #9, !dbg !15, !frisk.register_lifetimes !1
  %address.local563 = lshr i32 %address.thread498, 1, !dbg !15
  %address.local564 = and i32 %address.local563, 1, !dbg !15
  %address.local565 = zext nneg i32 %address.local564 to i64, !dbg !15
  %address.local566 = add nsw i64 %address.base, %address.local565, !dbg !15
  %address.local567 = shl i64 %address.local566, 6, !dbg !15
  %address.local568 = sub i64 %address.local562, %address.local567, !dbg !15
  %143 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local568, !dbg !15
  %144 = getelementptr [2 x i8], ptr addrspace(3) %143, i64 %address.local537, !dbg !15
  %145 = getelementptr [2 x i8], ptr addrspace(3) %144, i64 %address.local541, !dbg !15
  %address.local572 = getelementptr i8, ptr addrspace(3) %145, i64 64, !dbg !15
  %.idx18561859 = add i64 %address.local568, %address.local537, !dbg !15
  %146 = shl i64 %.idx18561859, 1, !dbg !15
  %147 = add i64 %146, %address.local543.idx, !dbg !15
  %148 = add i64 %147, 64, !dbg !15
  %149 = lshr i64 %148, 6, !dbg !15
  %150 = and i64 %149, 8, !dbg !15
  %151 = xor i64 %148, %150, !dbg !15
  %152 = sub i64 %151, %148, !dbg !15
  %kdma.swizzled1727 = getelementptr i8, ptr addrspace(3) %address.local572, i64 %152, !dbg !15
  %153 = load <4 x half>, ptr addrspace(3) %kdma.swizzled1727, align 8, !dbg !15, !frisk.k_lds_pair !35
  %kdma.gep1731 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1727, i64 128, !dbg !15
  %154 = load <4 x half>, ptr addrspace(3) %kdma.gep1731, align 8, !dbg !15, !frisk.k_lds_pair !35
  %kdma.gep1728 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1727, i64 4096, !dbg !15
  %155 = load <4 x half>, ptr addrspace(3) %kdma.gep1728, align 8, !dbg !15, !frisk.k_lds_pair !36
  %kdma.gep1732 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1727, i64 4224, !dbg !15
  %156 = load <4 x half>, ptr addrspace(3) %kdma.gep1732, align 8, !dbg !15, !frisk.k_lds_pair !36
  %kdma.gep1729 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 128, !dbg !15
  %157 = load <4 x half>, ptr addrspace(3) %kdma.gep1729, align 8, !dbg !15, !frisk.k_lds_pair !37
  %kdma.gep1733 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1024, !dbg !15
  %158 = load <4 x half>, ptr addrspace(3) %kdma.gep1733, align 8, !dbg !15, !frisk.k_lds_pair !37
  %kdma.gep1730 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4224, !dbg !15
  %159 = load <4 x half>, ptr addrspace(3) %kdma.gep1730, align 8, !dbg !15, !frisk.k_lds_pair !38
  %kdma.gep1734 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5120, !dbg !15
  %160 = load <4 x half>, ptr addrspace(3) %kdma.gep1734, align 8, !dbg !15, !frisk.k_lds_pair !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %161 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %142, <4 x float> %141) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %kdma.gep1735 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1727, i64 1024, !dbg !15
  %162 = load <4 x half>, ptr addrspace(3) %kdma.gep1735, align 8, !dbg !15, !frisk.k_lds_pair !33
  %kdma.gep1739 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1727, i64 1152, !dbg !15
  %163 = load <4 x half>, ptr addrspace(3) %kdma.gep1739, align 8, !dbg !15, !frisk.k_lds_pair !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %164 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %153, <4 x float> zeroinitializer) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %kdma.gep1736 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1727, i64 5120, !dbg !15
  %165 = load <4 x half>, ptr addrspace(3) %kdma.gep1736, align 8, !dbg !15, !frisk.k_lds_pair !39
  %kdma.gep1740 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1727, i64 5248, !dbg !15
  %166 = load <4 x half>, ptr addrspace(3) %kdma.gep1740, align 8, !dbg !15, !frisk.k_lds_pair !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %167 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %155, <4 x float> %164) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %kdma.gep1737 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1152, !dbg !15
  %168 = load <4 x half>, ptr addrspace(3) %kdma.gep1737, align 8, !dbg !15, !frisk.k_lds_pair !40
  %kdma.gep1738 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5248, !dbg !15
  %169 = load <4 x half>, ptr addrspace(3) %kdma.gep1738, align 8, !dbg !15, !frisk.k_lds_pair !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %170 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %157, <4 x float> zeroinitializer) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %171 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %159, <4 x float> %170) #9, !dbg !15
  %172 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %154, <4 x float> zeroinitializer) #9, !dbg !15
  %173 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %156, <4 x float> %172) #9, !dbg !15
  %174 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %158, <4 x float> zeroinitializer) #9, !dbg !15
  %175 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %160, <4 x float> %174) #9, !dbg !15
  %176 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %162, <4 x float> zeroinitializer) #9, !dbg !15
  %177 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %165, <4 x float> %176) #9, !dbg !15
  %178 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %168, <4 x float> zeroinitializer) #9, !dbg !15
  %179 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %169, <4 x float> %178) #9, !dbg !15
  %180 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %163, <4 x float> zeroinitializer) #9, !dbg !15
  %181 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %166, <4 x float> %180) #9, !dbg !15
  %182 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %140, <4 x float> zeroinitializer) #9, !dbg !15
  %183 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %142, <4 x float> %182) #9, !dbg !15
  %184 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %153, <4 x float> zeroinitializer) #9, !dbg !15
  %185 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %155, <4 x float> %184) #9, !dbg !15
  %186 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %157, <4 x float> zeroinitializer) #9, !dbg !15
  %187 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %159, <4 x float> %186) #9, !dbg !15
  %188 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %154, <4 x float> zeroinitializer) #9, !dbg !15
  %189 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %156, <4 x float> %188) #9, !dbg !15
  %190 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %158, <4 x float> zeroinitializer) #9, !dbg !15
  %191 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %160, <4 x float> %190) #9, !dbg !15
  %192 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %162, <4 x float> zeroinitializer) #9, !dbg !15
  %193 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %165, <4 x float> %192) #9, !dbg !15
  %194 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %168, <4 x float> zeroinitializer) #9, !dbg !15
  %195 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %169, <4 x float> %194) #9, !dbg !15
  %196 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %163, <4 x float> zeroinitializer) #9, !dbg !15
  %197 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %166, <4 x float> %196) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %198 = or disjoint i64 %116, 64, !dbg !13
  %kdma.source1606 = or disjoint i64 %198, %kdma.source1475, !dbg !13
  %kdma.source1607 = trunc nuw i64 %kdma.source1606 to i32, !dbg !13
  %kdma.source1608 = shl nuw nsw i32 %kdma.source1607, 1, !dbg !13
  %kdma.offset1609 = or disjoint i32 %kdma.source1608, %kdma.intra, !dbg !13
  %199 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %36, i32 %kdma.offset1609, <4 x i32> %dma.buffer.rsrc1343) #10, !dbg !13
  %kdma.source1622 = or disjoint i64 %198, %kdma.source1489, !dbg !13
  %kdma.source1623 = trunc nuw i64 %kdma.source1622 to i32, !dbg !13
  %kdma.source1624 = shl nuw nsw i32 %kdma.source1623, 1, !dbg !13
  %kdma.offset1625 = or disjoint i32 %kdma.source1624, %kdma.intra.hi, !dbg !13
  %200 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1494, i32 %kdma.offset1625, <4 x i32> %dma.buffer.rsrc1343) #10, !dbg !13
  %kdma.source1639 = or disjoint i64 %198, %kdma.source1506, !dbg !13
  %kdma.source1640 = trunc nuw i64 %kdma.source1639 to i32, !dbg !13
  %kdma.source1641 = shl nuw nsw i32 %kdma.source1640, 1, !dbg !13
  %kdma.offset1642 = or disjoint i32 %kdma.source1641, %kdma.intra, !dbg !13
  %201 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %43, i32 %kdma.offset1642, <4 x i32> %dma.buffer.rsrc1343) #10, !dbg !13
  %kdma.source1656 = or disjoint i64 %198, %kdma.source1523, !dbg !13
  %kdma.source1657 = trunc nuw i64 %kdma.source1656 to i32, !dbg !13
  %kdma.source1658 = shl nuw nsw i32 %kdma.source1657, 1, !dbg !13
  %kdma.offset1659 = or disjoint i32 %kdma.source1658, %kdma.intra.hi, !dbg !13
  %202 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1528, i32 %kdma.offset1659, <4 x i32> %dma.buffer.rsrc1343) #10, !dbg !13
  %kdma.gep1757 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8192, !dbg !15
  %203 = load <4 x half>, ptr addrspace(3) %kdma.gep1757, align 8, !dbg !15, !frisk.k_lds_pair !41
  %kdma.gep1761 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8320, !dbg !15
  %204 = load <4 x half>, ptr addrspace(3) %kdma.gep1761, align 8, !dbg !15, !frisk.k_lds_pair !41
  %kdma.gep1758 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12288, !dbg !15
  %205 = load <4 x half>, ptr addrspace(3) %kdma.gep1758, align 8, !dbg !15, !frisk.k_lds_pair !42
  %kdma.gep1762 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12416, !dbg !15
  %206 = load <4 x half>, ptr addrspace(3) %kdma.gep1762, align 8, !dbg !15, !frisk.k_lds_pair !42
  %kdma.gep1759 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1727, i64 8192, !dbg !15
  %207 = load <4 x half>, ptr addrspace(3) %kdma.gep1759, align 8, !dbg !15, !frisk.k_lds_pair !43
  %kdma.gep1763 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1727, i64 8320, !dbg !15
  %208 = load <4 x half>, ptr addrspace(3) %kdma.gep1763, align 8, !dbg !15, !frisk.k_lds_pair !43
  %kdma.gep1760 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1727, i64 12288, !dbg !15
  %209 = load <4 x half>, ptr addrspace(3) %kdma.gep1760, align 8, !dbg !15, !frisk.k_lds_pair !44
  %kdma.gep1764 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1727, i64 12416, !dbg !15
  %210 = load <4 x half>, ptr addrspace(3) %kdma.gep1764, align 8, !dbg !15, !frisk.k_lds_pair !44
  %kdma.gep1765 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9216, !dbg !15
  %211 = load <4 x half>, ptr addrspace(3) %kdma.gep1765, align 8, !dbg !15, !frisk.k_lds_pair !45
  %kdma.gep1769 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9344, !dbg !15
  %212 = load <4 x half>, ptr addrspace(3) %kdma.gep1769, align 8, !dbg !15, !frisk.k_lds_pair !45
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %213 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %203, <4 x float> %161) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %kdma.gep1766 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13312, !dbg !15
  %214 = load <4 x half>, ptr addrspace(3) %kdma.gep1766, align 8, !dbg !15, !frisk.k_lds_pair !46
  %kdma.gep1770 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13440, !dbg !15
  %215 = load <4 x half>, ptr addrspace(3) %kdma.gep1770, align 8, !dbg !15, !frisk.k_lds_pair !46
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %216 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %205, <4 x float> %213) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %kdma.gep1767 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1727, i64 9216, !dbg !15
  %217 = load <4 x half>, ptr addrspace(3) %kdma.gep1767, align 8, !dbg !15, !frisk.k_lds_pair !47
  %kdma.gep1771 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1727, i64 9344, !dbg !15
  %218 = load <4 x half>, ptr addrspace(3) %kdma.gep1771, align 8, !dbg !15, !frisk.k_lds_pair !47
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %219 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %207, <4 x float> %167) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %kdma.gep1768 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1727, i64 13312, !dbg !15
  %220 = load <4 x half>, ptr addrspace(3) %kdma.gep1768, align 8, !dbg !15, !frisk.k_lds_pair !48
  %kdma.gep1772 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1727, i64 13440, !dbg !15
  %221 = load <4 x half>, ptr addrspace(3) %kdma.gep1772, align 8, !dbg !15, !frisk.k_lds_pair !48
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %222 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %209, <4 x float> %219) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %223 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %204, <4 x float> %171) #9, !dbg !15
  %224 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %206, <4 x float> %223) #9, !dbg !15
  %225 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %208, <4 x float> %173) #9, !dbg !15
  %226 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %210, <4 x float> %225) #9, !dbg !15
  %227 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %211, <4 x float> %175) #9, !dbg !15
  %228 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %214, <4 x float> %227) #9, !dbg !15
  %229 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %217, <4 x float> %177) #9, !dbg !15
  %230 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %220, <4 x float> %229) #9, !dbg !15
  %231 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %212, <4 x float> %179) #9, !dbg !15
  %232 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %215, <4 x float> %231) #9, !dbg !15
  %233 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %218, <4 x float> %181) #9, !dbg !15
  %234 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %221, <4 x float> %233) #9, !dbg !15
  %235 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %203, <4 x float> %183) #9, !dbg !15
  %236 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %205, <4 x float> %235) #9, !dbg !15
  %237 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %207, <4 x float> %185) #9, !dbg !15
  %238 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %209, <4 x float> %237) #9, !dbg !15
  %239 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %204, <4 x float> %187) #9, !dbg !15
  %240 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %206, <4 x float> %239) #9, !dbg !15
  %241 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %208, <4 x float> %189) #9, !dbg !15
  %242 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %210, <4 x float> %241) #9, !dbg !15
  %243 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %211, <4 x float> %191) #9, !dbg !15
  %244 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %214, <4 x float> %243) #9, !dbg !15
  %245 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %217, <4 x float> %193) #9, !dbg !15
  %246 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %220, <4 x float> %245) #9, !dbg !15
  %247 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %212, <4 x float> %195) #9, !dbg !15
  %248 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %215, <4 x float> %247) #9, !dbg !15
  %249 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %218, <4 x float> %197) #9, !dbg !15
  %250 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %221, <4 x float> %249) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %251 = or disjoint i64 %116, 96, !dbg !13
  %kdma.source1672 = or disjoint i64 %251, %kdma.source1475, !dbg !13
  %kdma.source1673 = trunc nuw i64 %kdma.source1672 to i32, !dbg !13
  %kdma.source1674 = shl nuw nsw i32 %kdma.source1673, 1, !dbg !13
  %kdma.offset1675 = or disjoint i32 %kdma.source1674, %kdma.intra, !dbg !13
  %252 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %48, i32 %kdma.offset1675, <4 x i32> %dma.buffer.rsrc1343) #10, !dbg !13
  %kdma.source1688 = or disjoint i64 %251, %kdma.source1489, !dbg !13
  %kdma.source1689 = trunc nuw i64 %kdma.source1688 to i32, !dbg !13
  %kdma.source1690 = shl nuw nsw i32 %kdma.source1689, 1, !dbg !13
  %kdma.offset1691 = or disjoint i32 %kdma.source1690, %kdma.intra.hi, !dbg !13
  %253 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1560, i32 %kdma.offset1691, <4 x i32> %dma.buffer.rsrc1343) #10, !dbg !13
  %kdma.source1705 = or disjoint i64 %251, %kdma.source1506, !dbg !13
  %kdma.source1706 = trunc nuw i64 %kdma.source1705 to i32, !dbg !13
  %kdma.source1707 = shl nuw nsw i32 %kdma.source1706, 1, !dbg !13
  %kdma.offset1708 = or disjoint i32 %kdma.source1707, %kdma.intra, !dbg !13
  %254 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %52, i32 %kdma.offset1708, <4 x i32> %dma.buffer.rsrc1343) #10, !dbg !13
  %kdma.source1722 = or disjoint i64 %251, %kdma.source1523, !dbg !13
  %kdma.source1723 = trunc nuw i64 %kdma.source1722 to i32, !dbg !13
  %kdma.source1724 = shl nuw nsw i32 %kdma.source1723, 1, !dbg !13
  %kdma.offset1725 = or disjoint i32 %kdma.source1724, %kdma.intra.hi, !dbg !13
  %255 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1594, i32 %kdma.offset1725, <4 x i32> %dma.buffer.rsrc1343) #10, !dbg !13
  %256 = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8, !dbg !15, !frisk.k_lds_pair !49
  %257 = load <4 x half>, ptr addrspace(3) %kdma.gep1729, align 8, !dbg !15, !frisk.k_lds_pair !49
  %258 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8, !dbg !15, !frisk.k_lds_pair !50
  %259 = load <4 x half>, ptr addrspace(3) %kdma.gep1730, align 8, !dbg !15, !frisk.k_lds_pair !50
  %260 = load <4 x half>, ptr addrspace(3) %kdma.swizzled1727, align 8, !dbg !15, !frisk.k_lds_pair !51
  %261 = load <4 x half>, ptr addrspace(3) %kdma.gep1731, align 8, !dbg !15, !frisk.k_lds_pair !51
  %262 = load <4 x half>, ptr addrspace(3) %kdma.gep1728, align 8, !dbg !15, !frisk.k_lds_pair !52
  %263 = load <4 x half>, ptr addrspace(3) %kdma.gep1732, align 8, !dbg !15, !frisk.k_lds_pair !52
  %264 = load <4 x half>, ptr addrspace(3) %kdma.gep1733, align 8, !dbg !15, !frisk.k_lds_pair !53
  %265 = load <4 x half>, ptr addrspace(3) %kdma.gep1737, align 8, !dbg !15, !frisk.k_lds_pair !53
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %266 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %256, <4 x float> %216) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %267 = load <4 x half>, ptr addrspace(3) %kdma.gep1734, align 8, !dbg !15, !frisk.k_lds_pair !54
  %268 = load <4 x half>, ptr addrspace(3) %kdma.gep1738, align 8, !dbg !15, !frisk.k_lds_pair !54
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %269 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %258, <4 x float> %266) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %270 = load <4 x half>, ptr addrspace(3) %kdma.gep1735, align 8, !dbg !15, !frisk.k_lds_pair !55
  %271 = load <4 x half>, ptr addrspace(3) %kdma.gep1739, align 8, !dbg !15, !frisk.k_lds_pair !55
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %272 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %260, <4 x float> %222) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %273 = load <4 x half>, ptr addrspace(3) %kdma.gep1736, align 8, !dbg !15, !frisk.k_lds_pair !56
  %274 = load <4 x half>, ptr addrspace(3) %kdma.gep1740, align 8, !dbg !15, !frisk.k_lds_pair !56
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %275 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %262, <4 x float> %272) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %276 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %257, <4 x float> %224) #9, !dbg !15
  %277 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %259, <4 x float> %276) #9, !dbg !15
  %278 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %261, <4 x float> %226) #9, !dbg !15
  %279 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %263, <4 x float> %278) #9, !dbg !15
  %280 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %264, <4 x float> %228) #9, !dbg !15
  %281 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %267, <4 x float> %280) #9, !dbg !15
  %282 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %270, <4 x float> %230) #9, !dbg !15
  %283 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %273, <4 x float> %282) #9, !dbg !15
  %284 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %265, <4 x float> %232) #9, !dbg !15
  %285 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %268, <4 x float> %284) #9, !dbg !15
  %286 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %271, <4 x float> %234) #9, !dbg !15
  %287 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %274, <4 x float> %286) #9, !dbg !15
  %288 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %256, <4 x float> %236) #9, !dbg !15
  %289 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %258, <4 x float> %288) #9, !dbg !15
  %290 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %260, <4 x float> %238) #9, !dbg !15
  %291 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %262, <4 x float> %290) #9, !dbg !15
  %292 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %257, <4 x float> %240) #9, !dbg !15
  %293 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %259, <4 x float> %292) #9, !dbg !15
  %294 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %261, <4 x float> %242) #9, !dbg !15
  %295 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %263, <4 x float> %294) #9, !dbg !15
  %296 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %264, <4 x float> %244) #9, !dbg !15
  %297 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %267, <4 x float> %296) #9, !dbg !15
  %298 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %270, <4 x float> %246) #9, !dbg !15
  %299 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %273, <4 x float> %298) #9, !dbg !15
  %300 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %265, <4 x float> %248) #9, !dbg !15
  %301 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %268, <4 x float> %300) #9, !dbg !15
  %302 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %271, <4 x float> %250) #9, !dbg !15
  %303 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %274, <4 x float> %302) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %address.local901 = shl nuw nsw i64 %address.local499, 5, !dbg !16
  %address.local902 = and i64 %address.local901, 8064, !dbg !16
  %address.local903 = shl nuw nsw i64 %address.local499, 3, !dbg !16
  %address.local904 = and i64 %address.local903, 24, !dbg !16
  %address.local905 = or disjoint i64 %address.local902, %address.local904, !dbg !16
  %304 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !16
  %.idx13911392 = or disjoint i64 %116, %address.local905, !dbg !16
  %.idx13911392.tr = trunc nuw i64 %.idx13911392 to i32, !dbg !16
  %dma.buffer.offset1326 = shl nuw nsw i32 %.idx13911392.tr, 1, !dbg !16
  %305 = insertelement <2 x i64> poison, i64 %304, i64 0, !dbg !16
  %306 = bitcast <2 x i64> %305 to <4 x i32>, !dbg !16
  %dma.buffer.rsrc1327 = shufflevector <4 x i32> %306, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !16
  %307 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %36, i32 %dma.buffer.offset1326, <4 x i32> %dma.buffer.rsrc1327) #10, !dbg !16
  %address.local911 = or disjoint i64 %address.local905, 8192, !dbg !16
  %.idx13931394 = or disjoint i64 %116, %address.local911, !dbg !16
  %.idx13931394.tr = trunc nuw i64 %.idx13931394 to i32, !dbg !16
  %dma.buffer.offset1324 = shl nuw nsw i32 %.idx13931394.tr, 1, !dbg !16
  %308 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %43, i32 %dma.buffer.offset1324, <4 x i32> %dma.buffer.rsrc1327) #10, !dbg !16
  %309 = load <4 x half>, ptr addrspace(3) %kdma.gep1757, align 8, !dbg !15, !frisk.k_tail_pair !35
  %310 = load <4 x half>, ptr addrspace(3) %kdma.gep1761, align 8, !dbg !15, !frisk.k_tail_pair !35
  %311 = load <4 x half>, ptr addrspace(3) %kdma.gep1758, align 8, !dbg !15, !frisk.k_tail_pair !36
  %312 = load <4 x half>, ptr addrspace(3) %kdma.gep1762, align 8, !dbg !15, !frisk.k_tail_pair !36
  %313 = load <4 x half>, ptr addrspace(3) %kdma.gep1759, align 8, !dbg !15, !frisk.k_tail_pair !37
  %314 = load <4 x half>, ptr addrspace(3) %kdma.gep1763, align 8, !dbg !15, !frisk.k_tail_pair !37
  %315 = load <4 x half>, ptr addrspace(3) %kdma.gep1760, align 8, !dbg !15, !frisk.k_tail_pair !38
  %316 = load <4 x half>, ptr addrspace(3) %kdma.gep1764, align 8, !dbg !15, !frisk.k_tail_pair !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_tail_prefetch !1
  %317 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %309, <4 x float> %269) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_tail_prefetch !1
  %318 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %311, <4 x float> %317) #9, !dbg !15
  %319 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %313, <4 x float> %275) #9, !dbg !15
  %320 = extractelement <4 x float> %318, i64 0, !dbg !20
  %321 = call float @llvm.maxnum.f32(float %320, float 0xFFF0000000000000), !dbg !6, !frisk.local_reduction !1
  %322 = extractelement <4 x float> %318, i64 1, !dbg !20
  %323 = extractelement <4 x float> %318, i64 2, !dbg !20
  %324 = extractelement <4 x float> %318, i64 3, !dbg !20
  %325 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %315, <4 x float> %319) #9, !dbg !15
  %326 = load <4 x half>, ptr addrspace(3) %kdma.gep1765, align 8, !dbg !15, !frisk.k_tail_pair !33
  %327 = load <4 x half>, ptr addrspace(3) %kdma.gep1769, align 8, !dbg !15, !frisk.k_tail_pair !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_tail_prefetch !1
  %328 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %310, <4 x float> %277) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !20, !frisk.k_tail_prefetch !1
  %329 = extractelement <4 x float> %325, i64 0, !dbg !20
  %local.reduce151 = call float @llvm.maxnum.f32(float %321, float %329), !dbg !6, !frisk.local_reduction !1
  %330 = extractelement <4 x float> %325, i64 1, !dbg !20
  %local.reduce152 = call float @llvm.maxnum.f32(float %322, float %330), !dbg !6, !frisk.local_reduction !1
  %331 = extractelement <4 x float> %325, i64 2, !dbg !20
  %local.reduce153 = call float @llvm.maxnum.f32(float %323, float %331), !dbg !6, !frisk.local_reduction !1
  %332 = extractelement <4 x float> %325, i64 3, !dbg !20
  %local.reduce154 = call float @llvm.maxnum.f32(float %324, float %332), !dbg !6, !frisk.local_reduction !1
  %333 = load <4 x half>, ptr addrspace(3) %kdma.gep1766, align 8, !dbg !15, !frisk.k_tail_pair !39
  %334 = load <4 x half>, ptr addrspace(3) %kdma.gep1770, align 8, !dbg !15, !frisk.k_tail_pair !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_tail_prefetch !1
  %335 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %312, <4 x float> %328) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_tail_prefetch !1
  %336 = load <4 x half>, ptr addrspace(3) %kdma.gep1767, align 8, !dbg !15, !frisk.k_tail_pair !40
  %337 = load <4 x half>, ptr addrspace(3) %kdma.gep1771, align 8, !dbg !15, !frisk.k_tail_pair !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_tail_prefetch !1
  %338 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %314, <4 x float> %279) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !20, !frisk.k_tail_prefetch !1
  %339 = extractelement <4 x float> %335, i64 0, !dbg !20
  %local.reduce155 = call float @llvm.maxnum.f32(float %local.reduce151, float %339), !dbg !6, !frisk.local_reduction !1
  %340 = extractelement <4 x float> %335, i64 1, !dbg !20
  %local.reduce156 = call float @llvm.maxnum.f32(float %local.reduce152, float %340), !dbg !6, !frisk.local_reduction !1
  %341 = extractelement <4 x float> %335, i64 2, !dbg !20
  %local.reduce157 = call float @llvm.maxnum.f32(float %local.reduce153, float %341), !dbg !6, !frisk.local_reduction !1
  %342 = extractelement <4 x float> %335, i64 3, !dbg !20
  %local.reduce158 = call float @llvm.maxnum.f32(float %local.reduce154, float %342), !dbg !6, !frisk.local_reduction !1
  %343 = load <4 x half>, ptr addrspace(3) %kdma.gep1768, align 8, !dbg !15, !frisk.k_tail_pair !41
  %344 = load <4 x half>, ptr addrspace(3) %kdma.gep1772, align 8, !dbg !15, !frisk.k_tail_pair !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_tail_prefetch !1
  %345 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %316, <4 x float> %338) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_tail_prefetch !1
  %346 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %326, <4 x float> %281) #9, !dbg !15
  %347 = extractelement <4 x float> %345, i64 0, !dbg !20
  %local.reduce159 = call float @llvm.maxnum.f32(float %local.reduce155, float %347), !dbg !6, !frisk.local_reduction !1
  %348 = extractelement <4 x float> %345, i64 1, !dbg !20
  %local.reduce160 = call float @llvm.maxnum.f32(float %local.reduce156, float %348), !dbg !6, !frisk.local_reduction !1
  %349 = extractelement <4 x float> %345, i64 2, !dbg !20
  %local.reduce161 = call float @llvm.maxnum.f32(float %local.reduce157, float %349), !dbg !6, !frisk.local_reduction !1
  %350 = extractelement <4 x float> %345, i64 3, !dbg !20
  %local.reduce162 = call float @llvm.maxnum.f32(float %local.reduce158, float %350), !dbg !6, !frisk.local_reduction !1
  %351 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %333, <4 x float> %346) #9, !dbg !15
  %352 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %336, <4 x float> %283) #9, !dbg !15
  %353 = extractelement <4 x float> %351, i64 0, !dbg !20
  %local.reduce163 = call float @llvm.maxnum.f32(float %local.reduce159, float %353), !dbg !6, !frisk.local_reduction !1
  %354 = extractelement <4 x float> %351, i64 1, !dbg !20
  %local.reduce164 = call float @llvm.maxnum.f32(float %local.reduce160, float %354), !dbg !6, !frisk.local_reduction !1
  %355 = extractelement <4 x float> %351, i64 2, !dbg !20
  %local.reduce165 = call float @llvm.maxnum.f32(float %local.reduce161, float %355), !dbg !6, !frisk.local_reduction !1
  %356 = extractelement <4 x float> %351, i64 3, !dbg !20
  %local.reduce166 = call float @llvm.maxnum.f32(float %local.reduce162, float %356), !dbg !6, !frisk.local_reduction !1
  %357 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %343, <4 x float> %352) #9, !dbg !15
  %358 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %327, <4 x float> %285) #9, !dbg !15
  %359 = extractelement <4 x float> %357, i64 0, !dbg !20
  %local.reduce167 = call float @llvm.maxnum.f32(float %local.reduce163, float %359), !dbg !6, !frisk.local_reduction !1
  %360 = extractelement <4 x float> %357, i64 1, !dbg !20
  %local.reduce168 = call float @llvm.maxnum.f32(float %local.reduce164, float %360), !dbg !6, !frisk.local_reduction !1
  %361 = extractelement <4 x float> %357, i64 2, !dbg !20
  %local.reduce169 = call float @llvm.maxnum.f32(float %local.reduce165, float %361), !dbg !6, !frisk.local_reduction !1
  %362 = extractelement <4 x float> %357, i64 3, !dbg !20
  %local.reduce170 = call float @llvm.maxnum.f32(float %local.reduce166, float %362), !dbg !6, !frisk.local_reduction !1
  %363 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %334, <4 x float> %358) #9, !dbg !15
  %364 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %337, <4 x float> %287) #9, !dbg !15
  %365 = extractelement <4 x float> %363, i64 0, !dbg !20
  %local.reduce171 = call float @llvm.maxnum.f32(float %local.reduce167, float %365), !dbg !6, !frisk.local_reduction !1
  %366 = extractelement <4 x float> %363, i64 1, !dbg !20
  %local.reduce172 = call float @llvm.maxnum.f32(float %local.reduce168, float %366), !dbg !6, !frisk.local_reduction !1
  %367 = extractelement <4 x float> %363, i64 2, !dbg !20
  %local.reduce173 = call float @llvm.maxnum.f32(float %local.reduce169, float %367), !dbg !6, !frisk.local_reduction !1
  %368 = extractelement <4 x float> %363, i64 3, !dbg !20
  %local.reduce174 = call float @llvm.maxnum.f32(float %local.reduce170, float %368), !dbg !6, !frisk.local_reduction !1
  %369 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %344, <4 x float> %364) #9, !dbg !15
  %370 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %309, <4 x float> %289) #9, !dbg !15
  %371 = extractelement <4 x float> %369, i64 0, !dbg !20
  %local.reduce175 = call float @llvm.maxnum.f32(float %local.reduce171, float %371), !dbg !6, !frisk.local_reduction !1
  %372 = extractelement <4 x float> %369, i64 1, !dbg !20
  %local.reduce176 = call float @llvm.maxnum.f32(float %local.reduce172, float %372), !dbg !6, !frisk.local_reduction !1
  %373 = extractelement <4 x float> %369, i64 2, !dbg !20
  %local.reduce177 = call float @llvm.maxnum.f32(float %local.reduce173, float %373), !dbg !6, !frisk.local_reduction !1
  %374 = extractelement <4 x float> %369, i64 3, !dbg !20
  %local.reduce178 = call float @llvm.maxnum.f32(float %local.reduce174, float %374), !dbg !6, !frisk.local_reduction !1
  %local.reduce179 = call float @llvm.maxnum.f32(float %local.reduce175, float %local.reduce176), !dbg !6, !frisk.local_reduction !1
  %local.reduce180 = call float @llvm.maxnum.f32(float %local.reduce177, float %local.reduce178), !dbg !6, !frisk.local_reduction !1
  %375 = call float @llvm.maxnum.f32(float %local.reduce179, float %local.reduce180), !dbg !6, !frisk.local_reduction !1
  %376 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %311, <4 x float> %370) #9, !dbg !15
  %377 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %313, <4 x float> %291) #9, !dbg !15
  %378 = extractelement <4 x float> %376, i64 0, !dbg !20
  %379 = extractelement <4 x float> %376, i64 1, !dbg !20
  %380 = extractelement <4 x float> %376, i64 2, !dbg !20
  %381 = extractelement <4 x float> %376, i64 3, !dbg !20
  %382 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %315, <4 x float> %377) #9, !dbg !15
  %383 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %310, <4 x float> %293) #9, !dbg !15
  %384 = extractelement <4 x float> %382, i64 0, !dbg !20
  %385 = extractelement <4 x float> %382, i64 1, !dbg !20
  %386 = extractelement <4 x float> %382, i64 2, !dbg !20
  %387 = extractelement <4 x float> %382, i64 3, !dbg !20
  %388 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %312, <4 x float> %383) #9, !dbg !15
  %389 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %314, <4 x float> %295) #9, !dbg !15
  %390 = extractelement <4 x float> %388, i64 0, !dbg !20
  %391 = extractelement <4 x float> %388, i64 1, !dbg !20
  %392 = extractelement <4 x float> %388, i64 2, !dbg !20
  %393 = extractelement <4 x float> %388, i64 3, !dbg !20
  %394 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %316, <4 x float> %389) #9, !dbg !15
  %395 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %326, <4 x float> %297) #9, !dbg !15
  %396 = extractelement <4 x float> %394, i64 0, !dbg !20
  %397 = extractelement <4 x float> %394, i64 1, !dbg !20
  %398 = extractelement <4 x float> %394, i64 2, !dbg !20
  %399 = extractelement <4 x float> %394, i64 3, !dbg !20
  %400 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %333, <4 x float> %395) #9, !dbg !15
  %401 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %336, <4 x float> %299) #9, !dbg !15
  %402 = extractelement <4 x float> %400, i64 0, !dbg !20
  %403 = extractelement <4 x float> %400, i64 1, !dbg !20
  %404 = extractelement <4 x float> %400, i64 2, !dbg !20
  %405 = extractelement <4 x float> %400, i64 3, !dbg !20
  %406 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %343, <4 x float> %401) #9, !dbg !15
  %407 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %327, <4 x float> %301) #9, !dbg !15
  %408 = extractelement <4 x float> %406, i64 0, !dbg !20
  %409 = extractelement <4 x float> %406, i64 1, !dbg !20
  %410 = extractelement <4 x float> %406, i64 2, !dbg !20
  %411 = extractelement <4 x float> %406, i64 3, !dbg !20
  %412 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %334, <4 x float> %407) #9, !dbg !15
  %413 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %337, <4 x float> %303) #9, !dbg !15
  %414 = extractelement <4 x float> %412, i64 0, !dbg !20
  %415 = extractelement <4 x float> %412, i64 1, !dbg !20
  %416 = extractelement <4 x float> %412, i64 2, !dbg !20
  %417 = extractelement <4 x float> %412, i64 3, !dbg !20
  %418 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %344, <4 x float> %413) #9, !dbg !15
  %419 = extractelement <4 x float> %418, i64 0, !dbg !20
  %420 = extractelement <4 x float> %418, i64 1, !dbg !20
  %421 = extractelement <4 x float> %418, i64 2, !dbg !20
  %422 = extractelement <4 x float> %418, i64 3, !dbg !20
  %423 = bitcast float %375 to i32, !dbg !6
  %424 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %423), !dbg !6
  %425 = bitcast i32 %424 to float, !dbg !6
  %426 = call float @llvm.maxnum.f32(float %375, float %425), !dbg !6
  %427 = bitcast float %426 to i32, !dbg !6
  %428 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %427), !dbg !6
  %429 = bitcast i32 %428 to float, !dbg !6
  %430 = call float @llvm.maxnum.f32(float %426, float %429), !dbg !6
  %431 = call float @llvm.maxnum.f32(float %378, float 0xFFF0000000000000), !dbg !6, !frisk.local_reduction !1
  %local.reduce120 = call float @llvm.maxnum.f32(float %431, float %384), !dbg !6, !frisk.local_reduction !1
  %local.reduce121 = call float @llvm.maxnum.f32(float %379, float %385), !dbg !6, !frisk.local_reduction !1
  %local.reduce122 = call float @llvm.maxnum.f32(float %380, float %386), !dbg !6, !frisk.local_reduction !1
  %local.reduce123 = call float @llvm.maxnum.f32(float %381, float %387), !dbg !6, !frisk.local_reduction !1
  %local.reduce124 = call float @llvm.maxnum.f32(float %local.reduce120, float %390), !dbg !6, !frisk.local_reduction !1
  %local.reduce125 = call float @llvm.maxnum.f32(float %local.reduce121, float %391), !dbg !6, !frisk.local_reduction !1
  %local.reduce126 = call float @llvm.maxnum.f32(float %local.reduce122, float %392), !dbg !6, !frisk.local_reduction !1
  %local.reduce127 = call float @llvm.maxnum.f32(float %local.reduce123, float %393), !dbg !6, !frisk.local_reduction !1
  %local.reduce128 = call float @llvm.maxnum.f32(float %local.reduce124, float %396), !dbg !6, !frisk.local_reduction !1
  %local.reduce129 = call float @llvm.maxnum.f32(float %local.reduce125, float %397), !dbg !6, !frisk.local_reduction !1
  %local.reduce130 = call float @llvm.maxnum.f32(float %local.reduce126, float %398), !dbg !6, !frisk.local_reduction !1
  %local.reduce131 = call float @llvm.maxnum.f32(float %local.reduce127, float %399), !dbg !6, !frisk.local_reduction !1
  %local.reduce132 = call float @llvm.maxnum.f32(float %local.reduce128, float %402), !dbg !6, !frisk.local_reduction !1
  %local.reduce133 = call float @llvm.maxnum.f32(float %local.reduce129, float %403), !dbg !6, !frisk.local_reduction !1
  %local.reduce134 = call float @llvm.maxnum.f32(float %local.reduce130, float %404), !dbg !6, !frisk.local_reduction !1
  %local.reduce135 = call float @llvm.maxnum.f32(float %local.reduce131, float %405), !dbg !6, !frisk.local_reduction !1
  %local.reduce136 = call float @llvm.maxnum.f32(float %local.reduce132, float %408), !dbg !6, !frisk.local_reduction !1
  %local.reduce137 = call float @llvm.maxnum.f32(float %local.reduce133, float %409), !dbg !6, !frisk.local_reduction !1
  %local.reduce138 = call float @llvm.maxnum.f32(float %local.reduce134, float %410), !dbg !6, !frisk.local_reduction !1
  %local.reduce139 = call float @llvm.maxnum.f32(float %local.reduce135, float %411), !dbg !6, !frisk.local_reduction !1
  %local.reduce140 = call float @llvm.maxnum.f32(float %local.reduce136, float %414), !dbg !6, !frisk.local_reduction !1
  %local.reduce141 = call float @llvm.maxnum.f32(float %local.reduce137, float %415), !dbg !6, !frisk.local_reduction !1
  %local.reduce142 = call float @llvm.maxnum.f32(float %local.reduce138, float %416), !dbg !6, !frisk.local_reduction !1
  %local.reduce143 = call float @llvm.maxnum.f32(float %local.reduce139, float %417), !dbg !6, !frisk.local_reduction !1
  %local.reduce144 = call float @llvm.maxnum.f32(float %local.reduce140, float %419), !dbg !6, !frisk.local_reduction !1
  %local.reduce145 = call float @llvm.maxnum.f32(float %local.reduce141, float %420), !dbg !6, !frisk.local_reduction !1
  %local.reduce146 = call float @llvm.maxnum.f32(float %local.reduce142, float %421), !dbg !6, !frisk.local_reduction !1
  %local.reduce147 = call float @llvm.maxnum.f32(float %local.reduce143, float %422), !dbg !6, !frisk.local_reduction !1
  %local.reduce148 = call float @llvm.maxnum.f32(float %local.reduce144, float %local.reduce145), !dbg !6, !frisk.local_reduction !1
  %local.reduce149 = call float @llvm.maxnum.f32(float %local.reduce146, float %local.reduce147), !dbg !6, !frisk.local_reduction !1
  %432 = call float @llvm.maxnum.f32(float %local.reduce148, float %local.reduce149), !dbg !6, !frisk.local_reduction !1
  %433 = bitcast float %432 to i32, !dbg !6
  %434 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %433), !dbg !6
  %435 = bitcast i32 %434 to float, !dbg !6
  %436 = call float @llvm.maxnum.f32(float %432, float %435), !dbg !6
  %437 = bitcast float %436 to i32, !dbg !6
  %438 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %437), !dbg !6
  %439 = bitcast i32 %438 to float, !dbg !6
  %440 = call float @llvm.maxnum.f32(float %436, float %439), !dbg !6
  %441 = extractvalue [2 x <1 x float>] %107, 0, !dbg !6
  %442 = insertelement <1 x float> poison, float %430, i64 0, !dbg !6
  %443 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %442, <1 x float> %441), !dbg !6
  %444 = extractvalue [2 x <1 x float>] %107, 1, !dbg !6
  %445 = insertelement <1 x float> poison, float %440, i64 0, !dbg !6
  %446 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %445, <1 x float> %444), !dbg !6
  %447 = shufflevector <1 x float> %443, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !21
  %448 = shufflevector <1 x float> %446, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !21
  %449 = fsub <1 x float> %441, %443, !dbg !22
  %450 = extractelement <1 x float> %449, i64 0, !dbg !23
  %451 = call float @llvm.amdgcn.exp2.f32(float %450), !dbg !23
  %452 = insertelement <4 x float> poison, float %451, i64 0, !dbg !28
  %453 = shufflevector <4 x float> %452, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !28
  %454 = extractvalue [2 x <1 x float>] %106, 0, !dbg !24
  %455 = extractelement <1 x float> %454, i64 0, !dbg !24
  %456 = fmul float %455, %451, !dbg !24
  %457 = fsub <1 x float> %444, %446, !dbg !22
  %458 = extractelement <1 x float> %457, i64 0, !dbg !23
  %459 = call float @llvm.amdgcn.exp2.f32(float %458), !dbg !23
  %460 = insertelement <4 x float> poison, float %459, i64 0, !dbg !28
  %461 = shufflevector <4 x float> %460, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !28
  %462 = extractvalue [2 x <1 x float>] %106, 1, !dbg !24
  %463 = extractelement <1 x float> %462, i64 0, !dbg !24
  %464 = fmul float %463, %459, !dbg !24
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %vdb.address = or disjoint i64 %120, %address.local905, !dbg !16
  %vdb.address1407 = trunc nuw i64 %vdb.address to i32, !dbg !16
  %vdb.address1408 = shl nuw nsw i32 %vdb.address1407, 1, !dbg !16
  %vdb.address1413 = or disjoint i64 %120, %address.local911, !dbg !16
  %vdb.address1414 = trunc nuw i64 %vdb.address1413 to i32, !dbg !16
  %vdb.address1415 = shl nuw nsw i32 %vdb.address1414, 1, !dbg !16
  %vdb.dma.base = add i32 %36, 8192, !dbg !14
  %465 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1408, <4 x i32> %dma.buffer.rsrc1327) #10, !dbg !16
  %vdb.dma.base1420 = add i32 %43, 8192, !dbg !14
  %466 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1420, i32 %vdb.address1415, <4 x i32> %dma.buffer.rsrc1327) #10, !dbg !16
  %address.local913 = shl nuw nsw i32 %address.thread498, 3, !dbg !14
  %address.local914 = and i32 %address.local913, 120, !dbg !14
  %address.local916 = and i32 %address.local559, 1536, !dbg !14
  %address.local917 = or disjoint i32 %address.local914, %address.local916, !dbg !14
  %address.local918 = zext nneg i32 %address.local917 to i64, !dbg !14
  %address.local919 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local918, !dbg !14
  %467 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local919, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %468 = bitcast <4 x i32> %467 to <8 x half>, !dbg !14
  %address.local920 = and i64 %address.local499, 60, !dbg !14
  %address.local921 = add nuw nsw i64 %address.local920, %address.local499, !dbg !14
  %address.local923.idx = shl nuw nsw i64 %address.local921, 4, !dbg !14
  %address.local923 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %address.local923.idx, !dbg !14
  %address.local924 = lshr i64 %address.local539, 4, !dbg !14
  %address.local926.idx = mul nuw nsw i64 %address.local924, 768, !dbg !14
  %address.local926 = getelementptr i8, ptr addrspace(3) %address.local923, i64 %address.local926.idx, !dbg !14
  %address.local928.idx = mul nsw i64 %address.local526, -64, !dbg !14
  %address.local928 = getelementptr i8, ptr addrspace(3) %address.local926, i64 %address.local928.idx, !dbg !14
  %address.local929 = getelementptr i8, ptr addrspace(3) %address.local928, i64 256, !dbg !14
  %469 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local929, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %470 = bitcast <4 x i32> %469 to <8 x half>, !dbg !14
  %address.local933 = getelementptr i8, ptr addrspace(3) %address.local928, i64 512, !dbg !14
  %471 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local933, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %472 = bitcast <4 x i32> %471 to <8 x half>, !dbg !14
  %address.local937 = getelementptr i8, ptr addrspace(3) %address.local928, i64 768, !dbg !14
  %473 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local937, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %474 = bitcast <4 x i32> %473 to <8 x half>, !dbg !14
  %address.local939 = getelementptr i8, ptr addrspace(3) %address.local919, i64 4096, !dbg !14
  %475 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local939, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %476 = bitcast <4 x i32> %475 to <8 x half>, !dbg !14
  %address.local943 = getelementptr i8, ptr addrspace(3) %address.local928, i64 4352, !dbg !14
  %477 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local943, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %478 = bitcast <4 x i32> %477 to <8 x half>, !dbg !14
  %address.local947 = getelementptr i8, ptr addrspace(3) %address.local928, i64 4608, !dbg !14
  %479 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local947, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %480 = bitcast <4 x i32> %479 to <8 x half>, !dbg !14
  %address.local951 = getelementptr i8, ptr addrspace(3) %address.local928, i64 4864, !dbg !14
  %481 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local951, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %482 = shufflevector <8 x half> %468, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %483 = extractvalue [2 x <8 x float>] %105, 0, !dbg !28
  %484 = shufflevector <8 x float> %483, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %485 = fmul <4 x float> %484, %453, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %486 = fsub <4 x float> %318, %447, !dbg !21
  %487 = extractelement <4 x float> %486, i64 0, !dbg !25
  %488 = call float @llvm.amdgcn.exp2.f32(float %487), !dbg !25
  %489 = insertelement <4 x float> poison, float %488, i64 0, !dbg !27
  %490 = extractelement <4 x float> %486, i64 1, !dbg !25
  %491 = call float @llvm.amdgcn.exp2.f32(float %490), !dbg !25
  %492 = insertelement <4 x float> %489, float %491, i64 1, !dbg !27
  %493 = extractelement <4 x float> %486, i64 2, !dbg !25
  %494 = call float @llvm.amdgcn.exp2.f32(float %493), !dbg !25
  %495 = insertelement <4 x float> %492, float %494, i64 2, !dbg !27
  %496 = extractelement <4 x float> %486, i64 3, !dbg !25
  %497 = call float @llvm.amdgcn.exp2.f32(float %496), !dbg !25
  %498 = insertelement <4 x float> %495, float %497, i64 3, !dbg !27
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %498) #9, !dbg !27, !frisk.register_lifetimes !1
  %499 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !27
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %499) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %500 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %482, <4 x float> %485) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %501 = shufflevector <8 x half> %468, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %502 = shufflevector <8 x float> %483, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %503 = fmul <4 x float> %502, %453, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %504 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %501, <4 x float> %503) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %505 = extractvalue [2 x <8 x float>] %105, 1, !dbg !28
  %506 = shufflevector <8 x float> %505, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %507 = fmul <4 x float> %506, %461, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %508 = fsub <4 x float> %376, %448, !dbg !21
  %509 = extractelement <4 x float> %508, i64 0, !dbg !25
  %510 = call float @llvm.amdgcn.exp2.f32(float %509), !dbg !25
  %511 = insertelement <4 x float> poison, float %510, i64 0, !dbg !27
  %512 = extractelement <4 x float> %508, i64 1, !dbg !25
  %513 = call float @llvm.amdgcn.exp2.f32(float %512), !dbg !25
  %514 = insertelement <4 x float> %511, float %513, i64 1, !dbg !27
  %515 = extractelement <4 x float> %508, i64 2, !dbg !25
  %516 = call float @llvm.amdgcn.exp2.f32(float %515), !dbg !25
  %517 = insertelement <4 x float> %514, float %516, i64 2, !dbg !27
  %518 = extractelement <4 x float> %508, i64 3, !dbg !25
  %519 = call float @llvm.amdgcn.exp2.f32(float %518), !dbg !25
  %520 = insertelement <4 x float> %517, float %519, i64 3, !dbg !27
  %math.result388 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %520) #9, !dbg !27, !frisk.register_lifetimes !1
  %521 = fptrunc <4 x float> %math.result388 to <4 x half>, !dbg !27
  %math.packed389 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %521) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %522 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %482, <4 x float> %507) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %523 = shufflevector <8 x float> %505, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %524 = fmul <4 x float> %523, %461, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %525 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %501, <4 x float> %524) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %526 = shufflevector <8 x half> %470, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %527 = fsub <4 x float> %325, %447, !dbg !21
  %528 = extractelement <4 x float> %527, i64 0, !dbg !25
  %529 = call float @llvm.amdgcn.exp2.f32(float %528), !dbg !25
  %local.reduce89 = fadd float %488, %529, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor = call float asm sideeffect "", "=v,0"(float %local.reduce89) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %530 = insertelement <4 x float> poison, float %529, i64 0, !dbg !27
  %531 = extractelement <4 x float> %527, i64 1, !dbg !25
  %532 = call float @llvm.amdgcn.exp2.f32(float %531), !dbg !25
  %local.reduce90 = fadd float %491, %532, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1860 = call float asm sideeffect "", "=v,0"(float %local.reduce90) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %533 = insertelement <4 x float> %530, float %532, i64 1, !dbg !27
  %534 = extractelement <4 x float> %527, i64 2, !dbg !25
  %535 = call float @llvm.amdgcn.exp2.f32(float %534), !dbg !25
  %local.reduce91 = fadd float %494, %535, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1861 = call float asm sideeffect "", "=v,0"(float %local.reduce91) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %536 = insertelement <4 x float> %533, float %535, i64 2, !dbg !27
  %537 = extractelement <4 x float> %527, i64 3, !dbg !25
  %538 = call float @llvm.amdgcn.exp2.f32(float %537), !dbg !25
  %local.reduce92 = fadd float %497, %538, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1862 = call float asm sideeffect "", "=v,0"(float %local.reduce92) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %539 = insertelement <4 x float> %536, float %538, i64 3, !dbg !27
  %math.result374 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %539) #9, !dbg !27, !frisk.register_lifetimes !1
  %540 = fptrunc <4 x float> %math.result374 to <4 x half>, !dbg !27
  %math.packed375 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %540) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %541 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %526, <4 x float> %500) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %542 = shufflevector <8 x half> %470, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %543 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %542, <4 x float> %504) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %544 = fsub <4 x float> %382, %448, !dbg !21
  %545 = extractelement <4 x float> %544, i64 0, !dbg !25
  %546 = call float @llvm.amdgcn.exp2.f32(float %545), !dbg !25
  %local.reduce = fadd float %510, %546, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1863 = call float asm sideeffect "", "=v,0"(float %local.reduce) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %547 = insertelement <4 x float> poison, float %546, i64 0, !dbg !27
  %548 = extractelement <4 x float> %544, i64 1, !dbg !25
  %549 = call float @llvm.amdgcn.exp2.f32(float %548), !dbg !25
  %local.reduce59 = fadd float %513, %549, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1864 = call float asm sideeffect "", "=v,0"(float %local.reduce59) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %550 = insertelement <4 x float> %547, float %549, i64 1, !dbg !27
  %551 = extractelement <4 x float> %544, i64 2, !dbg !25
  %552 = call float @llvm.amdgcn.exp2.f32(float %551), !dbg !25
  %local.reduce60 = fadd float %516, %552, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1865 = call float asm sideeffect "", "=v,0"(float %local.reduce60) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %553 = insertelement <4 x float> %550, float %552, i64 2, !dbg !27
  %554 = extractelement <4 x float> %544, i64 3, !dbg !25
  %555 = call float @llvm.amdgcn.exp2.f32(float %554), !dbg !25
  %local.reduce61 = fadd float %519, %555, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1866 = call float asm sideeffect "", "=v,0"(float %local.reduce61) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %556 = insertelement <4 x float> %553, float %555, i64 3, !dbg !27
  %math.result390 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %556) #9, !dbg !27, !frisk.register_lifetimes !1
  %557 = fptrunc <4 x float> %math.result390 to <4 x half>, !dbg !27
  %math.packed391 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %557) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %558 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %526, <4 x float> %522) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %559 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %542, <4 x float> %525) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %560 = shufflevector <8 x half> %472, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %561 = fsub <4 x float> %335, %447, !dbg !21
  %562 = extractelement <4 x float> %561, i64 0, !dbg !25
  %563 = call float @llvm.amdgcn.exp2.f32(float %562), !dbg !25
  %local.reduce93 = fadd float %k0.sum.anchor, %563, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1867 = call float asm sideeffect "", "=v,0"(float %local.reduce93) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %564 = insertelement <4 x float> poison, float %563, i64 0, !dbg !27
  %565 = extractelement <4 x float> %561, i64 1, !dbg !25
  %566 = call float @llvm.amdgcn.exp2.f32(float %565), !dbg !25
  %local.reduce94 = fadd float %k0.sum.anchor1860, %566, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1868 = call float asm sideeffect "", "=v,0"(float %local.reduce94) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %567 = insertelement <4 x float> %564, float %566, i64 1, !dbg !27
  %568 = extractelement <4 x float> %561, i64 2, !dbg !25
  %569 = call float @llvm.amdgcn.exp2.f32(float %568), !dbg !25
  %local.reduce95 = fadd float %k0.sum.anchor1861, %569, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1869 = call float asm sideeffect "", "=v,0"(float %local.reduce95) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %570 = insertelement <4 x float> %567, float %569, i64 2, !dbg !27
  %571 = extractelement <4 x float> %561, i64 3, !dbg !25
  %572 = call float @llvm.amdgcn.exp2.f32(float %571), !dbg !25
  %local.reduce96 = fadd float %k0.sum.anchor1862, %572, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1870 = call float asm sideeffect "", "=v,0"(float %local.reduce96) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %573 = insertelement <4 x float> %570, float %572, i64 3, !dbg !27
  %math.result376 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %573) #9, !dbg !27, !frisk.register_lifetimes !1
  %574 = fptrunc <4 x float> %math.result376 to <4 x half>, !dbg !27
  %math.packed377 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %574) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %575 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %560, <4 x float> %541) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %576 = shufflevector <8 x half> %472, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %577 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %576, <4 x float> %543) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %578 = fsub <4 x float> %388, %448, !dbg !21
  %579 = extractelement <4 x float> %578, i64 0, !dbg !25
  %580 = call float @llvm.amdgcn.exp2.f32(float %579), !dbg !25
  %local.reduce62 = fadd float %k0.sum.anchor1863, %580, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1871 = call float asm sideeffect "", "=v,0"(float %local.reduce62) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %581 = insertelement <4 x float> poison, float %580, i64 0, !dbg !27
  %582 = extractelement <4 x float> %578, i64 1, !dbg !25
  %583 = call float @llvm.amdgcn.exp2.f32(float %582), !dbg !25
  %local.reduce63 = fadd float %k0.sum.anchor1864, %583, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1872 = call float asm sideeffect "", "=v,0"(float %local.reduce63) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %584 = insertelement <4 x float> %581, float %583, i64 1, !dbg !27
  %585 = extractelement <4 x float> %578, i64 2, !dbg !25
  %586 = call float @llvm.amdgcn.exp2.f32(float %585), !dbg !25
  %local.reduce64 = fadd float %k0.sum.anchor1865, %586, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1873 = call float asm sideeffect "", "=v,0"(float %local.reduce64) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %587 = insertelement <4 x float> %584, float %586, i64 2, !dbg !27
  %588 = extractelement <4 x float> %578, i64 3, !dbg !25
  %589 = call float @llvm.amdgcn.exp2.f32(float %588), !dbg !25
  %local.reduce65 = fadd float %k0.sum.anchor1866, %589, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1874 = call float asm sideeffect "", "=v,0"(float %local.reduce65) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %590 = insertelement <4 x float> %587, float %589, i64 3, !dbg !27
  %math.result392 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %590) #9, !dbg !27, !frisk.register_lifetimes !1
  %591 = fptrunc <4 x float> %math.result392 to <4 x half>, !dbg !27
  %math.packed393 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %591) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %592 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %560, <4 x float> %558) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %593 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %576, <4 x float> %559) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %594 = shufflevector <8 x half> %474, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %595 = fsub <4 x float> %345, %447, !dbg !21
  %596 = extractelement <4 x float> %595, i64 0, !dbg !25
  %597 = call float @llvm.amdgcn.exp2.f32(float %596), !dbg !25
  %local.reduce97 = fadd float %k0.sum.anchor1867, %597, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1875 = call float asm sideeffect "", "=v,0"(float %local.reduce97) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %598 = insertelement <4 x float> poison, float %597, i64 0, !dbg !27
  %599 = extractelement <4 x float> %595, i64 1, !dbg !25
  %600 = call float @llvm.amdgcn.exp2.f32(float %599), !dbg !25
  %local.reduce98 = fadd float %k0.sum.anchor1868, %600, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1876 = call float asm sideeffect "", "=v,0"(float %local.reduce98) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %601 = insertelement <4 x float> %598, float %600, i64 1, !dbg !27
  %602 = extractelement <4 x float> %595, i64 2, !dbg !25
  %603 = call float @llvm.amdgcn.exp2.f32(float %602), !dbg !25
  %local.reduce99 = fadd float %k0.sum.anchor1869, %603, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1877 = call float asm sideeffect "", "=v,0"(float %local.reduce99) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %604 = insertelement <4 x float> %601, float %603, i64 2, !dbg !27
  %605 = extractelement <4 x float> %595, i64 3, !dbg !25
  %606 = call float @llvm.amdgcn.exp2.f32(float %605), !dbg !25
  %local.reduce100 = fadd float %k0.sum.anchor1870, %606, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1878 = call float asm sideeffect "", "=v,0"(float %local.reduce100) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %607 = insertelement <4 x float> %604, float %606, i64 3, !dbg !27
  %math.result378 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %607) #9, !dbg !27, !frisk.register_lifetimes !1
  %608 = fptrunc <4 x float> %math.result378 to <4 x half>, !dbg !27
  %math.packed379 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %608) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %609 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %594, <4 x float> %575) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %610 = shufflevector <8 x half> %474, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %611 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %610, <4 x float> %577) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %612 = fsub <4 x float> %394, %448, !dbg !21
  %613 = extractelement <4 x float> %612, i64 0, !dbg !25
  %614 = call float @llvm.amdgcn.exp2.f32(float %613), !dbg !25
  %local.reduce66 = fadd float %k0.sum.anchor1871, %614, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1879 = call float asm sideeffect "", "=v,0"(float %local.reduce66) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %615 = insertelement <4 x float> poison, float %614, i64 0, !dbg !27
  %616 = extractelement <4 x float> %612, i64 1, !dbg !25
  %617 = call float @llvm.amdgcn.exp2.f32(float %616), !dbg !25
  %local.reduce67 = fadd float %k0.sum.anchor1872, %617, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1880 = call float asm sideeffect "", "=v,0"(float %local.reduce67) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %618 = insertelement <4 x float> %615, float %617, i64 1, !dbg !27
  %619 = extractelement <4 x float> %612, i64 2, !dbg !25
  %620 = call float @llvm.amdgcn.exp2.f32(float %619), !dbg !25
  %local.reduce68 = fadd float %k0.sum.anchor1873, %620, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1881 = call float asm sideeffect "", "=v,0"(float %local.reduce68) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %621 = insertelement <4 x float> %618, float %620, i64 2, !dbg !27
  %622 = extractelement <4 x float> %612, i64 3, !dbg !25
  %623 = call float @llvm.amdgcn.exp2.f32(float %622), !dbg !25
  %local.reduce69 = fadd float %k0.sum.anchor1874, %623, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1882 = call float asm sideeffect "", "=v,0"(float %local.reduce69) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %624 = insertelement <4 x float> %621, float %623, i64 3, !dbg !27
  %math.result394 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %624) #9, !dbg !27, !frisk.register_lifetimes !1
  %625 = fptrunc <4 x float> %math.result394 to <4 x half>, !dbg !27
  %math.packed395 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %625) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %626 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %594, <4 x float> %592) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %627 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %610, <4 x float> %593) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %628 = shufflevector <8 x half> %476, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %629 = fsub <4 x float> %351, %447, !dbg !21
  %630 = extractelement <4 x float> %629, i64 0, !dbg !25
  %631 = call float @llvm.amdgcn.exp2.f32(float %630), !dbg !25
  %local.reduce101 = fadd float %k0.sum.anchor1875, %631, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1883 = call float asm sideeffect "", "=v,0"(float %local.reduce101) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %632 = insertelement <4 x float> poison, float %631, i64 0, !dbg !27
  %633 = extractelement <4 x float> %629, i64 1, !dbg !25
  %634 = call float @llvm.amdgcn.exp2.f32(float %633), !dbg !25
  %local.reduce102 = fadd float %k0.sum.anchor1876, %634, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1884 = call float asm sideeffect "", "=v,0"(float %local.reduce102) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %635 = insertelement <4 x float> %632, float %634, i64 1, !dbg !27
  %636 = extractelement <4 x float> %629, i64 2, !dbg !25
  %637 = call float @llvm.amdgcn.exp2.f32(float %636), !dbg !25
  %local.reduce103 = fadd float %k0.sum.anchor1877, %637, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1885 = call float asm sideeffect "", "=v,0"(float %local.reduce103) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %638 = insertelement <4 x float> %635, float %637, i64 2, !dbg !27
  %639 = extractelement <4 x float> %629, i64 3, !dbg !25
  %640 = call float @llvm.amdgcn.exp2.f32(float %639), !dbg !25
  %local.reduce104 = fadd float %k0.sum.anchor1878, %640, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1886 = call float asm sideeffect "", "=v,0"(float %local.reduce104) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %641 = insertelement <4 x float> %638, float %640, i64 3, !dbg !27
  %math.result380 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %641) #9, !dbg !27, !frisk.register_lifetimes !1
  %642 = fptrunc <4 x float> %math.result380 to <4 x half>, !dbg !27
  %math.packed381 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %642) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %643 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %628, <4 x float> %609) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %644 = shufflevector <8 x half> %476, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %645 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %644, <4 x float> %611) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %646 = fsub <4 x float> %400, %448, !dbg !21
  %647 = extractelement <4 x float> %646, i64 0, !dbg !25
  %648 = call float @llvm.amdgcn.exp2.f32(float %647), !dbg !25
  %local.reduce70 = fadd float %k0.sum.anchor1879, %648, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1887 = call float asm sideeffect "", "=v,0"(float %local.reduce70) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %649 = insertelement <4 x float> poison, float %648, i64 0, !dbg !27
  %650 = extractelement <4 x float> %646, i64 1, !dbg !25
  %651 = call float @llvm.amdgcn.exp2.f32(float %650), !dbg !25
  %local.reduce71 = fadd float %k0.sum.anchor1880, %651, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1888 = call float asm sideeffect "", "=v,0"(float %local.reduce71) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %652 = insertelement <4 x float> %649, float %651, i64 1, !dbg !27
  %653 = extractelement <4 x float> %646, i64 2, !dbg !25
  %654 = call float @llvm.amdgcn.exp2.f32(float %653), !dbg !25
  %local.reduce72 = fadd float %k0.sum.anchor1881, %654, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1889 = call float asm sideeffect "", "=v,0"(float %local.reduce72) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %655 = insertelement <4 x float> %652, float %654, i64 2, !dbg !27
  %656 = extractelement <4 x float> %646, i64 3, !dbg !25
  %657 = call float @llvm.amdgcn.exp2.f32(float %656), !dbg !25
  %local.reduce73 = fadd float %k0.sum.anchor1882, %657, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1890 = call float asm sideeffect "", "=v,0"(float %local.reduce73) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %658 = insertelement <4 x float> %655, float %657, i64 3, !dbg !27
  %math.result396 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %658) #9, !dbg !27, !frisk.register_lifetimes !1
  %659 = fptrunc <4 x float> %math.result396 to <4 x half>, !dbg !27
  %math.packed397 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %659) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %660 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %628, <4 x float> %626) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %661 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %644, <4 x float> %627) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %662 = shufflevector <8 x half> %478, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %663 = fsub <4 x float> %357, %447, !dbg !21
  %664 = extractelement <4 x float> %663, i64 0, !dbg !25
  %665 = call float @llvm.amdgcn.exp2.f32(float %664), !dbg !25
  %local.reduce105 = fadd float %k0.sum.anchor1883, %665, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1891 = call float asm sideeffect "", "=v,0"(float %local.reduce105) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %666 = insertelement <4 x float> poison, float %665, i64 0, !dbg !27
  %667 = extractelement <4 x float> %663, i64 1, !dbg !25
  %668 = call float @llvm.amdgcn.exp2.f32(float %667), !dbg !25
  %local.reduce106 = fadd float %k0.sum.anchor1884, %668, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1892 = call float asm sideeffect "", "=v,0"(float %local.reduce106) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %669 = insertelement <4 x float> %666, float %668, i64 1, !dbg !27
  %670 = extractelement <4 x float> %663, i64 2, !dbg !25
  %671 = call float @llvm.amdgcn.exp2.f32(float %670), !dbg !25
  %local.reduce107 = fadd float %k0.sum.anchor1885, %671, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1893 = call float asm sideeffect "", "=v,0"(float %local.reduce107) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %672 = insertelement <4 x float> %669, float %671, i64 2, !dbg !27
  %673 = extractelement <4 x float> %663, i64 3, !dbg !25
  %674 = call float @llvm.amdgcn.exp2.f32(float %673), !dbg !25
  %local.reduce108 = fadd float %k0.sum.anchor1886, %674, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1894 = call float asm sideeffect "", "=v,0"(float %local.reduce108) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %675 = insertelement <4 x float> %672, float %674, i64 3, !dbg !27
  %math.result382 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %675) #9, !dbg !27, !frisk.register_lifetimes !1
  %676 = fptrunc <4 x float> %math.result382 to <4 x half>, !dbg !27
  %math.packed383 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %676) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %677 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %662, <4 x float> %643) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %678 = shufflevector <8 x half> %478, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %679 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %678, <4 x float> %645) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %680 = fsub <4 x float> %406, %448, !dbg !21
  %681 = extractelement <4 x float> %680, i64 0, !dbg !25
  %682 = call float @llvm.amdgcn.exp2.f32(float %681), !dbg !25
  %local.reduce74 = fadd float %k0.sum.anchor1887, %682, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1895 = call float asm sideeffect "", "=v,0"(float %local.reduce74) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %683 = insertelement <4 x float> poison, float %682, i64 0, !dbg !27
  %684 = extractelement <4 x float> %680, i64 1, !dbg !25
  %685 = call float @llvm.amdgcn.exp2.f32(float %684), !dbg !25
  %local.reduce75 = fadd float %k0.sum.anchor1888, %685, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1896 = call float asm sideeffect "", "=v,0"(float %local.reduce75) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %686 = insertelement <4 x float> %683, float %685, i64 1, !dbg !27
  %687 = extractelement <4 x float> %680, i64 2, !dbg !25
  %688 = call float @llvm.amdgcn.exp2.f32(float %687), !dbg !25
  %local.reduce76 = fadd float %k0.sum.anchor1889, %688, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1897 = call float asm sideeffect "", "=v,0"(float %local.reduce76) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %689 = insertelement <4 x float> %686, float %688, i64 2, !dbg !27
  %690 = extractelement <4 x float> %680, i64 3, !dbg !25
  %691 = call float @llvm.amdgcn.exp2.f32(float %690), !dbg !25
  %local.reduce77 = fadd float %k0.sum.anchor1890, %691, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1898 = call float asm sideeffect "", "=v,0"(float %local.reduce77) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %692 = insertelement <4 x float> %689, float %691, i64 3, !dbg !27
  %math.result398 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %692) #9, !dbg !27, !frisk.register_lifetimes !1
  %693 = fptrunc <4 x float> %math.result398 to <4 x half>, !dbg !27
  %math.packed399 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %693) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %694 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %662, <4 x float> %660) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %695 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %678, <4 x float> %661) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %696 = shufflevector <8 x half> %480, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %697 = fsub <4 x float> %363, %447, !dbg !21
  %698 = extractelement <4 x float> %697, i64 0, !dbg !25
  %699 = call float @llvm.amdgcn.exp2.f32(float %698), !dbg !25
  %local.reduce109 = fadd float %k0.sum.anchor1891, %699, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1899 = call float asm sideeffect "", "=v,0"(float %local.reduce109) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %700 = insertelement <4 x float> poison, float %699, i64 0, !dbg !27
  %701 = extractelement <4 x float> %697, i64 1, !dbg !25
  %702 = call float @llvm.amdgcn.exp2.f32(float %701), !dbg !25
  %local.reduce110 = fadd float %k0.sum.anchor1892, %702, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1900 = call float asm sideeffect "", "=v,0"(float %local.reduce110) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %703 = insertelement <4 x float> %700, float %702, i64 1, !dbg !27
  %704 = extractelement <4 x float> %697, i64 2, !dbg !25
  %705 = call float @llvm.amdgcn.exp2.f32(float %704), !dbg !25
  %local.reduce111 = fadd float %k0.sum.anchor1893, %705, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1901 = call float asm sideeffect "", "=v,0"(float %local.reduce111) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %706 = insertelement <4 x float> %703, float %705, i64 2, !dbg !27
  %707 = extractelement <4 x float> %697, i64 3, !dbg !25
  %708 = call float @llvm.amdgcn.exp2.f32(float %707), !dbg !25
  %local.reduce112 = fadd float %k0.sum.anchor1894, %708, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1902 = call float asm sideeffect "", "=v,0"(float %local.reduce112) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %709 = insertelement <4 x float> %706, float %708, i64 3, !dbg !27
  %math.result384 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %709) #9, !dbg !27, !frisk.register_lifetimes !1
  %710 = fptrunc <4 x float> %math.result384 to <4 x half>, !dbg !27
  %math.packed385 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %710) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %711 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %696, <4 x float> %677) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %712 = shufflevector <8 x half> %480, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %713 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %712, <4 x float> %679) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %714 = fsub <4 x float> %412, %448, !dbg !21
  %715 = extractelement <4 x float> %714, i64 0, !dbg !25
  %716 = call float @llvm.amdgcn.exp2.f32(float %715), !dbg !25
  %local.reduce78 = fadd float %k0.sum.anchor1895, %716, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1903 = call float asm sideeffect "", "=v,0"(float %local.reduce78) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %717 = insertelement <4 x float> poison, float %716, i64 0, !dbg !27
  %718 = extractelement <4 x float> %714, i64 1, !dbg !25
  %719 = call float @llvm.amdgcn.exp2.f32(float %718), !dbg !25
  %local.reduce79 = fadd float %k0.sum.anchor1896, %719, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1904 = call float asm sideeffect "", "=v,0"(float %local.reduce79) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %720 = insertelement <4 x float> %717, float %719, i64 1, !dbg !27
  %721 = extractelement <4 x float> %714, i64 2, !dbg !25
  %722 = call float @llvm.amdgcn.exp2.f32(float %721), !dbg !25
  %local.reduce80 = fadd float %k0.sum.anchor1897, %722, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1905 = call float asm sideeffect "", "=v,0"(float %local.reduce80) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %723 = insertelement <4 x float> %720, float %722, i64 2, !dbg !27
  %724 = extractelement <4 x float> %714, i64 3, !dbg !25
  %725 = call float @llvm.amdgcn.exp2.f32(float %724), !dbg !25
  %local.reduce81 = fadd float %k0.sum.anchor1898, %725, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1906 = call float asm sideeffect "", "=v,0"(float %local.reduce81) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %726 = insertelement <4 x float> %723, float %725, i64 3, !dbg !27
  %math.result400 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %726) #9, !dbg !27, !frisk.register_lifetimes !1
  %727 = fptrunc <4 x float> %math.result400 to <4 x half>, !dbg !27
  %math.packed401 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %727) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %728 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %696, <4 x float> %694) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %729 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %712, <4 x float> %695) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %730 = bitcast <4 x i32> %481 to <8 x half>, !dbg !14
  %731 = shufflevector <8 x half> %730, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %732 = fsub <4 x float> %369, %447, !dbg !21
  %733 = extractelement <4 x float> %732, i64 0, !dbg !25
  %734 = call float @llvm.amdgcn.exp2.f32(float %733), !dbg !25
  %local.reduce113 = fadd float %k0.sum.anchor1899, %734, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1907 = call float asm sideeffect "", "=v,0"(float %local.reduce113) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %735 = insertelement <4 x float> poison, float %734, i64 0, !dbg !27
  %736 = extractelement <4 x float> %732, i64 1, !dbg !25
  %737 = call float @llvm.amdgcn.exp2.f32(float %736), !dbg !25
  %local.reduce114 = fadd float %k0.sum.anchor1900, %737, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1908 = call float asm sideeffect "", "=v,0"(float %local.reduce114) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %local.reduce117 = fadd float %k0.sum.anchor1907, %k0.sum.anchor1908, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1909 = call float asm sideeffect "", "=v,0"(float %local.reduce117) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %738 = insertelement <4 x float> %735, float %737, i64 1, !dbg !27
  %739 = extractelement <4 x float> %732, i64 2, !dbg !25
  %740 = call float @llvm.amdgcn.exp2.f32(float %739), !dbg !25
  %local.reduce115 = fadd float %k0.sum.anchor1901, %740, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1910 = call float asm sideeffect "", "=v,0"(float %local.reduce115) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %741 = insertelement <4 x float> %738, float %740, i64 2, !dbg !27
  %742 = extractelement <4 x float> %732, i64 3, !dbg !25
  %743 = call float @llvm.amdgcn.exp2.f32(float %742), !dbg !25
  %local.reduce116 = fadd float %k0.sum.anchor1902, %743, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1911 = call float asm sideeffect "", "=v,0"(float %local.reduce116) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %local.reduce118 = fadd float %k0.sum.anchor1910, %k0.sum.anchor1911, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1912 = call float asm sideeffect "", "=v,0"(float %local.reduce118) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %744 = fadd float %k0.sum.anchor1909, %k0.sum.anchor1912, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1913 = call float asm sideeffect "", "=v,0"(float %744) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %745 = fadd float %k0.sum.anchor1913, %456, !dbg !26
  %k0.sum.anchor1914 = call float asm sideeffect "", "=v,0"(float %745) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %746 = insertelement <1 x float> poison, float %k0.sum.anchor1914, i64 0, !dbg !8
  %747 = insertvalue [2 x <1 x float>] poison, <1 x float> %746, 0, !dbg !8
  %748 = insertelement <4 x float> %741, float %743, i64 3, !dbg !27
  %math.result386 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %748) #9, !dbg !27, !frisk.register_lifetimes !1
  %749 = fptrunc <4 x float> %math.result386 to <4 x half>, !dbg !27
  %math.packed387 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %749) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %750 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %731, <4 x float> %711) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %751 = shufflevector <8 x half> %730, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %752 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %751, <4 x float> %713) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %753 = fsub <4 x float> %418, %448, !dbg !21
  %754 = extractelement <4 x float> %753, i64 0, !dbg !25
  %755 = call float @llvm.amdgcn.exp2.f32(float %754), !dbg !25
  %local.reduce82 = fadd float %k0.sum.anchor1903, %755, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1915 = call float asm sideeffect "", "=v,0"(float %local.reduce82) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %756 = insertelement <4 x float> poison, float %755, i64 0, !dbg !27
  %757 = extractelement <4 x float> %753, i64 1, !dbg !25
  %758 = call float @llvm.amdgcn.exp2.f32(float %757), !dbg !25
  %local.reduce83 = fadd float %k0.sum.anchor1904, %758, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1916 = call float asm sideeffect "", "=v,0"(float %local.reduce83) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %local.reduce86 = fadd float %k0.sum.anchor1915, %k0.sum.anchor1916, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1917 = call float asm sideeffect "", "=v,0"(float %local.reduce86) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %759 = insertelement <4 x float> %756, float %758, i64 1, !dbg !27
  %760 = extractelement <4 x float> %753, i64 2, !dbg !25
  %761 = call float @llvm.amdgcn.exp2.f32(float %760), !dbg !25
  %local.reduce84 = fadd float %k0.sum.anchor1905, %761, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1918 = call float asm sideeffect "", "=v,0"(float %local.reduce84) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %762 = insertelement <4 x float> %759, float %761, i64 2, !dbg !27
  %763 = extractelement <4 x float> %753, i64 3, !dbg !25
  %764 = call float @llvm.amdgcn.exp2.f32(float %763), !dbg !25
  %local.reduce85 = fadd float %k0.sum.anchor1906, %764, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1919 = call float asm sideeffect "", "=v,0"(float %local.reduce85) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %local.reduce87 = fadd float %k0.sum.anchor1918, %k0.sum.anchor1919, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1920 = call float asm sideeffect "", "=v,0"(float %local.reduce87) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %765 = fadd float %k0.sum.anchor1917, %k0.sum.anchor1920, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1921 = call float asm sideeffect "", "=v,0"(float %765) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %766 = fadd float %k0.sum.anchor1921, %464, !dbg !26
  %k0.sum.anchor1922 = call float asm sideeffect "", "=v,0"(float %766) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %767 = insertelement <1 x float> poison, float %k0.sum.anchor1922, i64 0, !dbg !8
  %768 = insertvalue [2 x <1 x float>] %747, <1 x float> %767, 1, !dbg !8
  %769 = insertelement <4 x float> %762, float %764, i64 3, !dbg !27
  %math.result402 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %769) #9, !dbg !27, !frisk.register_lifetimes !1
  %770 = fptrunc <4 x float> %math.result402 to <4 x half>, !dbg !27
  %math.packed403 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %770) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %771 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %731, <4 x float> %728) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %772 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %751, <4 x float> %729) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %vdb.address1428 = or disjoint i64 %198, %address.local905, !dbg !16
  %vdb.address1429 = trunc nuw i64 %vdb.address1428 to i32, !dbg !16
  %vdb.address1430 = shl nuw nsw i32 %vdb.address1429, 1, !dbg !16
  %vdb.address1435 = or disjoint i64 %198, %address.local911, !dbg !16
  %vdb.address1436 = trunc nuw i64 %vdb.address1435 to i32, !dbg !16
  %vdb.address1437 = shl nuw nsw i32 %vdb.address1436, 1, !dbg !16
  %773 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %36, i32 %vdb.address1430, <4 x i32> %dma.buffer.rsrc1327) #10, !dbg !16
  %774 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %43, i32 %vdb.address1437, <4 x i32> %dma.buffer.rsrc1327) #10, !dbg !16
  %vdb.read = getelementptr i8, ptr addrspace(3) %address.local919, i64 8192, !dbg !14
  %775 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %776 = bitcast <4 x i32> %775 to <8 x half>, !dbg !14
  %vdb.read1421 = getelementptr i8, ptr addrspace(3) %address.local928, i64 8448, !dbg !14
  %777 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1421, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %778 = bitcast <4 x i32> %777 to <8 x half>, !dbg !14
  %vdb.read1422 = getelementptr i8, ptr addrspace(3) %address.local928, i64 8704, !dbg !14
  %779 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1422, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %780 = bitcast <4 x i32> %779 to <8 x half>, !dbg !14
  %vdb.read1423 = getelementptr i8, ptr addrspace(3) %address.local928, i64 8960, !dbg !14
  %781 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1423, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %782 = bitcast <4 x i32> %781 to <8 x half>, !dbg !14
  %vdb.read1424 = getelementptr i8, ptr addrspace(3) %address.local919, i64 12288, !dbg !14
  %783 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1424, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %784 = bitcast <4 x i32> %783 to <8 x half>, !dbg !14
  %vdb.read1425 = getelementptr i8, ptr addrspace(3) %address.local928, i64 12544, !dbg !14
  %785 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1425, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %786 = bitcast <4 x i32> %785 to <8 x half>, !dbg !14
  %vdb.read1426 = getelementptr i8, ptr addrspace(3) %address.local928, i64 12800, !dbg !14
  %787 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1426, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %788 = bitcast <4 x i32> %787 to <8 x half>, !dbg !14
  %vdb.read1427 = getelementptr i8, ptr addrspace(3) %address.local928, i64 13056, !dbg !14
  %789 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1427, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %790 = shufflevector <8 x half> %776, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %791 = extractvalue [2 x <8 x float>] %108, 0, !dbg !28
  %792 = shufflevector <8 x float> %791, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %793 = fmul <4 x float> %792, %453, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %794 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %790, <4 x float> %793) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %795 = shufflevector <8 x half> %776, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %796 = shufflevector <8 x float> %791, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %797 = fmul <4 x float> %796, %453, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %798 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %795, <4 x float> %797) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %799 = extractvalue [2 x <8 x float>] %108, 1, !dbg !28
  %800 = shufflevector <8 x float> %799, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %801 = fmul <4 x float> %800, %461, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %802 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %790, <4 x float> %801) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %803 = shufflevector <8 x float> %799, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %804 = fmul <4 x float> %803, %461, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %805 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %795, <4 x float> %804) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %806 = shufflevector <8 x half> %778, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %807 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %806, <4 x float> %794) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %808 = shufflevector <8 x half> %778, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %809 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %808, <4 x float> %798) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %810 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %806, <4 x float> %802) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %811 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %808, <4 x float> %805) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %812 = shufflevector <8 x half> %780, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %813 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %812, <4 x float> %807) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %814 = shufflevector <8 x half> %780, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %815 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %814, <4 x float> %809) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %816 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %812, <4 x float> %810) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %817 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %814, <4 x float> %811) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %818 = shufflevector <8 x half> %782, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %819 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %818, <4 x float> %813) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %820 = shufflevector <8 x half> %782, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %821 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %820, <4 x float> %815) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %822 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %818, <4 x float> %816) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %823 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %820, <4 x float> %817) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %824 = shufflevector <8 x half> %784, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %825 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %824, <4 x float> %819) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %826 = shufflevector <8 x half> %784, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %827 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %826, <4 x float> %821) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %828 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %824, <4 x float> %822) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %829 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %826, <4 x float> %823) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %830 = shufflevector <8 x half> %786, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %831 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %830, <4 x float> %825) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %832 = shufflevector <8 x half> %786, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %833 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %832, <4 x float> %827) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %834 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %830, <4 x float> %828) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %835 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %832, <4 x float> %829) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %836 = shufflevector <8 x half> %788, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %837 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %836, <4 x float> %831) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %838 = shufflevector <8 x half> %788, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %839 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %838, <4 x float> %833) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %840 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %836, <4 x float> %834) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %841 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %838, <4 x float> %835) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %842 = bitcast <4 x i32> %789 to <8 x half>, !dbg !14
  %843 = shufflevector <8 x half> %842, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %844 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %843, <4 x float> %837) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %845 = shufflevector <8 x half> %842, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %846 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %845, <4 x float> %839) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %847 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %843, <4 x float> %840) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %848 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %845, <4 x float> %841) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %vdb.address1442 = or disjoint i64 %251, %address.local905, !dbg !16
  %vdb.address1443 = trunc nuw i64 %vdb.address1442 to i32, !dbg !16
  %vdb.address1444 = shl nuw nsw i32 %vdb.address1443, 1, !dbg !16
  %vdb.address1449 = or disjoint i64 %251, %address.local911, !dbg !16
  %vdb.address1450 = trunc nuw i64 %vdb.address1449 to i32, !dbg !16
  %vdb.address1451 = shl nuw nsw i32 %vdb.address1450, 1, !dbg !16
  %849 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1444, <4 x i32> %dma.buffer.rsrc1327) #10, !dbg !16
  %850 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1420, i32 %vdb.address1451, <4 x i32> %dma.buffer.rsrc1327) #10, !dbg !16
  %851 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local919, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %852 = bitcast <4 x i32> %851 to <8 x half>, !dbg !14
  %853 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local929, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %854 = bitcast <4 x i32> %853 to <8 x half>, !dbg !14
  %855 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local933, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %856 = bitcast <4 x i32> %855 to <8 x half>, !dbg !14
  %857 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local937, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %858 = bitcast <4 x i32> %857 to <8 x half>, !dbg !14
  %859 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local939, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %860 = bitcast <4 x i32> %859 to <8 x half>, !dbg !14
  %861 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local943, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %862 = bitcast <4 x i32> %861 to <8 x half>, !dbg !14
  %863 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local947, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %864 = bitcast <4 x i32> %863 to <8 x half>, !dbg !14
  %865 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local951, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %866 = shufflevector <8 x half> %852, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %867 = extractvalue [2 x <8 x float>] %109, 0, !dbg !28
  %868 = shufflevector <8 x float> %867, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %869 = fmul <4 x float> %868, %453, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %870 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %866, <4 x float> %869) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %871 = shufflevector <8 x half> %852, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %872 = shufflevector <8 x float> %867, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %873 = fmul <4 x float> %872, %453, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %874 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %871, <4 x float> %873) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %875 = extractvalue [2 x <8 x float>] %109, 1, !dbg !28
  %876 = shufflevector <8 x float> %875, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %877 = fmul <4 x float> %876, %461, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %878 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %866, <4 x float> %877) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %879 = shufflevector <8 x float> %875, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %880 = fmul <4 x float> %879, %461, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %881 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %871, <4 x float> %880) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %882 = shufflevector <8 x half> %854, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %883 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %882, <4 x float> %870) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %884 = shufflevector <8 x half> %854, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %885 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %884, <4 x float> %874) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %886 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %882, <4 x float> %878) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %887 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %884, <4 x float> %881) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %888 = shufflevector <8 x half> %856, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %889 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %888, <4 x float> %883) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %890 = shufflevector <8 x half> %856, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %891 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %890, <4 x float> %885) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %892 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %888, <4 x float> %886) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %893 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %890, <4 x float> %887) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %894 = shufflevector <8 x half> %858, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %895 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %894, <4 x float> %889) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %896 = shufflevector <8 x half> %858, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %897 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %896, <4 x float> %891) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %898 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %894, <4 x float> %892) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %899 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %896, <4 x float> %893) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %900 = shufflevector <8 x half> %860, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %901 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %900, <4 x float> %895) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %902 = shufflevector <8 x half> %860, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %903 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %902, <4 x float> %897) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %904 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %900, <4 x float> %898) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %905 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %902, <4 x float> %899) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %906 = shufflevector <8 x half> %862, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %907 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %906, <4 x float> %901) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %908 = shufflevector <8 x half> %862, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %909 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %908, <4 x float> %903) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %910 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %906, <4 x float> %904) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %911 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %908, <4 x float> %905) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %912 = shufflevector <8 x half> %864, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %913 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %912, <4 x float> %907) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %914 = shufflevector <8 x half> %864, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %915 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %914, <4 x float> %909) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %916 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %912, <4 x float> %910) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %917 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %914, <4 x float> %911) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %918 = bitcast <4 x i32> %865 to <8 x half>, !dbg !14
  %919 = shufflevector <8 x half> %918, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %920 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %919, <4 x float> %913) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %921 = shufflevector <8 x half> %918, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %922 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %921, <4 x float> %915) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %923 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %919, <4 x float> %916) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %924 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %921, <4 x float> %917) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %925 = add nuw nsw i64 %104, 1, !dbg !8
  %k0.has.next = icmp ult i64 %925, 32, !dbg !14
  br i1 %k0.has.next, label %k0.issue, label %k0.join, !dbg !14

k0.issue:                                         ; preds = %112
  %k0.next. = shl nuw nsw i64 %925, 14, !dbg !13
  %k0.next.1933 = or disjoint i64 %114, %k0.next., !dbg !13
  %k0.next.kdma.source1476 = or disjoint i64 %k0.next.1933, %kdma.source1475, !dbg !13
  %k0.next.kdma.source1477 = trunc nuw i64 %k0.next.kdma.source1476 to i32, !dbg !13
  %k0.next.kdma.source1478 = shl nuw nsw i32 %k0.next.kdma.source1477, 1, !dbg !13
  %k0.next.kdma.offset = or disjoint i32 %k0.next.kdma.source1478, %kdma.intra, !dbg !13
  %k0.next.1934 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %36, i32 %k0.next.kdma.offset, <4 x i32> %dma.buffer.rsrc1343) #10, !dbg !13, !frisk.cross_bn_k0 !1
  %k0.next.kdma.source1490 = or disjoint i64 %k0.next.1933, %kdma.source1489, !dbg !13
  %k0.next.kdma.source1491 = trunc nuw i64 %k0.next.kdma.source1490 to i32, !dbg !13
  %k0.next.kdma.source1492 = shl nuw nsw i32 %k0.next.kdma.source1491, 1, !dbg !13
  %k0.next.kdma.offset1493 = or disjoint i32 %k0.next.kdma.source1492, %kdma.intra.hi, !dbg !13
  %k0.next.1935 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1494, i32 %k0.next.kdma.offset1493, <4 x i32> %dma.buffer.rsrc1343) #10, !dbg !13, !frisk.cross_bn_k0 !1
  %k0.next.kdma.source1507 = or disjoint i64 %k0.next.1933, %kdma.source1506, !dbg !13
  %k0.next.kdma.source1508 = trunc nuw i64 %k0.next.kdma.source1507 to i32, !dbg !13
  %k0.next.kdma.source1509 = shl nuw nsw i32 %k0.next.kdma.source1508, 1, !dbg !13
  %k0.next.kdma.offset1510 = or disjoint i32 %k0.next.kdma.source1509, %kdma.intra, !dbg !13
  %k0.next.1936 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %43, i32 %k0.next.kdma.offset1510, <4 x i32> %dma.buffer.rsrc1343) #10, !dbg !13, !frisk.cross_bn_k0 !1
  %k0.next.kdma.source1524 = or disjoint i64 %k0.next.1933, %kdma.source1523, !dbg !13
  %k0.next.kdma.source1525 = trunc nuw i64 %k0.next.kdma.source1524 to i32, !dbg !13
  %k0.next.kdma.source1526 = shl nuw nsw i32 %k0.next.kdma.source1525, 1, !dbg !13
  %k0.next.kdma.offset1527 = or disjoint i32 %k0.next.kdma.source1526, %kdma.intra.hi, !dbg !13
  %k0.next.1937 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1528, i32 %k0.next.kdma.offset1527, <4 x i32> %dma.buffer.rsrc1343) #10, !dbg !13, !frisk.cross_bn_k0 !1
  br label %k0.join

k0.join:                                          ; preds = %112, %k0.issue
  %926 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %927 = bitcast <4 x i32> %926 to <8 x half>, !dbg !14
  %928 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1421, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %929 = bitcast <4 x i32> %928 to <8 x half>, !dbg !14
  %930 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1422, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %931 = bitcast <4 x i32> %930 to <8 x half>, !dbg !14
  %932 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1423, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %933 = bitcast <4 x i32> %932 to <8 x half>, !dbg !14
  %934 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1424, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %935 = bitcast <4 x i32> %934 to <8 x half>, !dbg !14
  %936 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1425, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %937 = bitcast <4 x i32> %936 to <8 x half>, !dbg !14
  %938 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1426, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %939 = bitcast <4 x i32> %938 to <8 x half>, !dbg !14
  %940 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1427, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %941 = shufflevector <8 x half> %927, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %942 = extractvalue [2 x <8 x float>] %110, 0, !dbg !28
  %943 = shufflevector <8 x float> %942, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %944 = fmul <4 x float> %943, %453, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %945 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %941, <4 x float> %944) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %946 = shufflevector <8 x half> %927, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %947 = shufflevector <8 x float> %942, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %948 = fmul <4 x float> %947, %453, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %949 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %946, <4 x float> %948) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %950 = extractvalue [2 x <8 x float>] %110, 1, !dbg !28
  %951 = shufflevector <8 x float> %950, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %952 = fmul <4 x float> %951, %461, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %953 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %941, <4 x float> %952) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %954 = shufflevector <8 x float> %950, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %955 = fmul <4 x float> %954, %461, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %956 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %946, <4 x float> %955) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %957 = shufflevector <8 x half> %929, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %958 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %957, <4 x float> %945) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %959 = shufflevector <8 x half> %929, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %960 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %959, <4 x float> %949) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %961 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %957, <4 x float> %953) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %962 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %959, <4 x float> %956) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %963 = shufflevector <8 x half> %931, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %964 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %963, <4 x float> %958) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %965 = shufflevector <8 x half> %931, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %966 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %965, <4 x float> %960) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %967 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %963, <4 x float> %961) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %968 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %965, <4 x float> %962) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %969 = shufflevector <8 x half> %933, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %970 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %969, <4 x float> %964) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %971 = shufflevector <8 x half> %933, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %972 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %971, <4 x float> %966) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %973 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %969, <4 x float> %967) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %974 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %971, <4 x float> %968) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %975 = shufflevector <8 x half> %935, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %976 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %975, <4 x float> %970) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %977 = shufflevector <8 x half> %935, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %978 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %977, <4 x float> %972) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %979 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %975, <4 x float> %973) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %980 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %977, <4 x float> %974) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %981 = shufflevector <8 x half> %937, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %982 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %981, <4 x float> %976) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %983 = shufflevector <8 x half> %937, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %984 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %983, <4 x float> %978) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %985 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %981, <4 x float> %979) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %986 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %983, <4 x float> %980) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %987 = shufflevector <8 x half> %939, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %988 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %987, <4 x float> %982) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %989 = shufflevector <8 x half> %939, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %990 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %989, <4 x float> %984) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %991 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %987, <4 x float> %985) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %992 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %989, <4 x float> %986) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %993 = bitcast <4 x i32> %940 to <8 x half>, !dbg !14
  %994 = shufflevector <8 x half> %993, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %995 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %994, <4 x float> %988) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %996 = shufflevector <8 x half> %993, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %997 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %996, <4 x float> %990) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %998 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %994, <4 x float> %991) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %999 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %996, <4 x float> %992) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !8
  %1000 = shufflevector <4 x float> %750, <4 x float> %752, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %1001 = insertvalue [2 x <8 x float>] poison, <8 x float> %1000, 0, !dbg !8
  %1002 = shufflevector <4 x float> %771, <4 x float> %772, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %1003 = insertvalue [2 x <8 x float>] %1001, <8 x float> %1002, 1, !dbg !8
  %1004 = insertvalue [2 x <1 x float>] poison, <1 x float> %443, 0, !dbg !8
  %1005 = insertvalue [2 x <1 x float>] %1004, <1 x float> %446, 1, !dbg !8
  %1006 = shufflevector <4 x float> %844, <4 x float> %846, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %1007 = insertvalue [2 x <8 x float>] poison, <8 x float> %1006, 0, !dbg !8
  %1008 = shufflevector <4 x float> %847, <4 x float> %848, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %1009 = insertvalue [2 x <8 x float>] %1007, <8 x float> %1008, 1, !dbg !8
  %1010 = shufflevector <4 x float> %920, <4 x float> %922, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %1011 = insertvalue [2 x <8 x float>] poison, <8 x float> %1010, 0, !dbg !8
  %1012 = shufflevector <4 x float> %923, <4 x float> %924, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %1013 = insertvalue [2 x <8 x float>] %1011, <8 x float> %1012, 1, !dbg !8
  %1014 = shufflevector <4 x float> %995, <4 x float> %997, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %1015 = insertvalue [2 x <8 x float>] poison, <8 x float> %1014, 0, !dbg !8
  %1016 = shufflevector <4 x float> %998, <4 x float> %999, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %1017 = insertvalue [2 x <8 x float>] %1015, <8 x float> %1016, 1, !dbg !8
  br label %103, !dbg !8

1018:                                             ; preds = %103
  %lane.sum.field184 = extractvalue [2 x <1 x float>] %106, 1, !dbg !8
  %lane.sum.local185 = extractelement <1 x float> %lane.sum.field184, i64 0, !dbg !8
  %1019 = shl nuw nsw i32 %8, 2, !dbg !8
  %1020 = xor i32 %1019, 64, !dbg !8
  %1021 = bitcast float %lane.sum.local185 to i32, !dbg !8
  %lane.sum.shuffle188 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1020, i32 %1021), !dbg !8
  %1022 = bitcast i32 %lane.sum.shuffle188 to float, !dbg !8
  %lane.sum.final189 = fadd float %lane.sum.local185, %1022, !dbg !8
  %1023 = xor i32 %1019, 128, !dbg !8
  %1024 = bitcast float %lane.sum.final189 to i32, !dbg !8
  %lane.sum.shuffle190 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1023, i32 %1024), !dbg !8
  %1025 = bitcast i32 %lane.sum.shuffle190 to float, !dbg !8
  %lane.sum.final191 = fadd float %lane.sum.final189, %1025, !dbg !8
  %1026 = insertelement <1 x float> poison, float %lane.sum.final191, i64 0, !dbg !8
  %lane.sum.field = extractvalue [2 x <1 x float>] %106, 0, !dbg !8
  %lane.sum.local = extractelement <1 x float> %lane.sum.field, i64 0, !dbg !8
  %1027 = bitcast float %lane.sum.local to i32, !dbg !8
  %lane.sum.shuffle = call i32 @llvm.amdgcn.ds.bpermute(i32 %1020, i32 %1027), !dbg !8
  %1028 = bitcast i32 %lane.sum.shuffle to float, !dbg !8
  %lane.sum.final = fadd float %lane.sum.local, %1028, !dbg !8
  %1029 = bitcast float %lane.sum.final to i32, !dbg !8
  %lane.sum.shuffle182 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1023, i32 %1029), !dbg !8
  %1030 = bitcast i32 %lane.sum.shuffle182 to float, !dbg !8
  %lane.sum.final183 = fadd float %lane.sum.final, %1030, !dbg !8
  %1031 = insertelement <1 x float> poison, float %lane.sum.final183, i64 0, !dbg !8
  %1032 = and i32 %kdma.tid, 16, !dbg !8
  %.not = icmp eq i32 %1032, 0, !dbg !8
  %1033 = and i32 %kdma.tid, 32, !dbg !8
  %.not36 = icmp eq i32 %1033, 0, !dbg !8
  %1034 = extractvalue [2 x <8 x float>] %110, 0, !dbg !8
  %1035 = extractelement <8 x float> %1034, i64 0, !dbg !8
  %1036 = extractelement <8 x float> %1034, i64 1, !dbg !8
  %1037 = extractelement <8 x float> %1034, i64 2, !dbg !8
  %1038 = extractelement <8 x float> %1034, i64 3, !dbg !8
  %1039 = select i1 %.not, float %1036, float %1035, !dbg !8
  %1040 = bitcast float %1039 to i32, !dbg !8
  %1041 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1040), !dbg !8
  %1042 = bitcast i32 %1041 to float, !dbg !8
  %1043 = select i1 %.not, float %1035, float %1042, !dbg !8
  %1044 = select i1 %.not, float %1042, float %1036, !dbg !8
  %1045 = select i1 %.not, float %1038, float %1037, !dbg !8
  %1046 = bitcast float %1045 to i32, !dbg !8
  %1047 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1046), !dbg !8
  %1048 = bitcast i32 %1047 to float, !dbg !8
  %1049 = select i1 %.not, float %1037, float %1048, !dbg !8
  %1050 = select i1 %.not, float %1048, float %1038, !dbg !8
  %1051 = select i1 %.not36, float %1049, float %1043, !dbg !8
  %1052 = bitcast float %1051 to i32, !dbg !8
  %1053 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1052), !dbg !8
  %1054 = bitcast i32 %1053 to float, !dbg !8
  %1055 = select i1 %.not36, float %1043, float %1054, !dbg !8
  %1056 = select i1 %.not36, float %1054, float %1049, !dbg !8
  %1057 = select i1 %.not36, float %1050, float %1044, !dbg !8
  %1058 = bitcast float %1057 to i32, !dbg !8
  %1059 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1058), !dbg !8
  %1060 = bitcast i32 %1059 to float, !dbg !8
  %1061 = select i1 %.not36, float %1044, float %1060, !dbg !8
  %1062 = select i1 %.not36, float %1060, float %1050, !dbg !8
  %1063 = extractelement <8 x float> %1034, i64 4, !dbg !8
  %1064 = extractelement <8 x float> %1034, i64 5, !dbg !8
  %1065 = extractelement <8 x float> %1034, i64 6, !dbg !8
  %1066 = extractelement <8 x float> %1034, i64 7, !dbg !8
  %1067 = select i1 %.not, float %1064, float %1063, !dbg !8
  %1068 = bitcast float %1067 to i32, !dbg !8
  %1069 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1068), !dbg !8
  %1070 = bitcast i32 %1069 to float, !dbg !8
  %1071 = select i1 %.not, float %1063, float %1070, !dbg !8
  %1072 = select i1 %.not, float %1070, float %1064, !dbg !8
  %1073 = select i1 %.not, float %1066, float %1065, !dbg !8
  %1074 = bitcast float %1073 to i32, !dbg !8
  %1075 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1074), !dbg !8
  %1076 = bitcast i32 %1075 to float, !dbg !8
  %1077 = select i1 %.not, float %1065, float %1076, !dbg !8
  %1078 = select i1 %.not, float %1076, float %1066, !dbg !8
  %1079 = select i1 %.not36, float %1077, float %1071, !dbg !8
  %1080 = bitcast float %1079 to i32, !dbg !8
  %1081 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1080), !dbg !8
  %1082 = bitcast i32 %1081 to float, !dbg !8
  %1083 = select i1 %.not36, float %1071, float %1082, !dbg !8
  %1084 = select i1 %.not36, float %1082, float %1077, !dbg !8
  %1085 = select i1 %.not36, float %1078, float %1072, !dbg !8
  %1086 = bitcast float %1085 to i32, !dbg !8
  %1087 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1086), !dbg !8
  %1088 = bitcast i32 %1087 to float, !dbg !8
  %1089 = select i1 %.not36, float %1072, float %1088, !dbg !8
  %1090 = select i1 %.not36, float %1088, float %1078, !dbg !8
  %1091 = extractvalue [2 x <8 x float>] %110, 1, !dbg !8
  %1092 = extractelement <8 x float> %1091, i64 0, !dbg !8
  %1093 = extractelement <8 x float> %1091, i64 1, !dbg !8
  %1094 = extractelement <8 x float> %1091, i64 2, !dbg !8
  %1095 = extractelement <8 x float> %1091, i64 3, !dbg !8
  %1096 = select i1 %.not, float %1093, float %1092, !dbg !8
  %1097 = bitcast float %1096 to i32, !dbg !8
  %1098 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1097), !dbg !8
  %1099 = bitcast i32 %1098 to float, !dbg !8
  %1100 = select i1 %.not, float %1092, float %1099, !dbg !8
  %1101 = select i1 %.not, float %1099, float %1093, !dbg !8
  %1102 = select i1 %.not, float %1095, float %1094, !dbg !8
  %1103 = bitcast float %1102 to i32, !dbg !8
  %1104 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1103), !dbg !8
  %1105 = bitcast i32 %1104 to float, !dbg !8
  %1106 = select i1 %.not, float %1094, float %1105, !dbg !8
  %1107 = select i1 %.not, float %1105, float %1095, !dbg !8
  %1108 = select i1 %.not36, float %1106, float %1100, !dbg !8
  %1109 = bitcast float %1108 to i32, !dbg !8
  %1110 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1109), !dbg !8
  %1111 = bitcast i32 %1110 to float, !dbg !8
  %1112 = select i1 %.not36, float %1100, float %1111, !dbg !8
  %1113 = select i1 %.not36, float %1111, float %1106, !dbg !8
  %1114 = select i1 %.not36, float %1107, float %1101, !dbg !8
  %1115 = bitcast float %1114 to i32, !dbg !8
  %1116 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1115), !dbg !8
  %1117 = bitcast i32 %1116 to float, !dbg !8
  %1118 = select i1 %.not36, float %1101, float %1117, !dbg !8
  %1119 = select i1 %.not36, float %1117, float %1107, !dbg !8
  %1120 = extractelement <8 x float> %1091, i64 4, !dbg !8
  %1121 = extractelement <8 x float> %1091, i64 5, !dbg !8
  %1122 = extractelement <8 x float> %1091, i64 6, !dbg !8
  %1123 = extractelement <8 x float> %1091, i64 7, !dbg !8
  %1124 = select i1 %.not, float %1121, float %1120, !dbg !8
  %1125 = bitcast float %1124 to i32, !dbg !8
  %1126 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1125), !dbg !8
  %1127 = bitcast i32 %1126 to float, !dbg !8
  %1128 = select i1 %.not, float %1120, float %1127, !dbg !8
  %1129 = select i1 %.not, float %1127, float %1121, !dbg !8
  %1130 = select i1 %.not, float %1123, float %1122, !dbg !8
  %1131 = bitcast float %1130 to i32, !dbg !8
  %1132 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1131), !dbg !8
  %1133 = bitcast i32 %1132 to float, !dbg !8
  %1134 = select i1 %.not, float %1122, float %1133, !dbg !8
  %1135 = select i1 %.not, float %1133, float %1123, !dbg !8
  %1136 = select i1 %.not36, float %1134, float %1128, !dbg !8
  %1137 = bitcast float %1136 to i32, !dbg !8
  %1138 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1137), !dbg !8
  %1139 = bitcast i32 %1138 to float, !dbg !8
  %1140 = select i1 %.not36, float %1128, float %1139, !dbg !8
  %1141 = select i1 %.not36, float %1139, float %1134, !dbg !8
  %1142 = select i1 %.not36, float %1135, float %1129, !dbg !8
  %1143 = bitcast float %1142 to i32, !dbg !8
  %1144 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1143), !dbg !8
  %1145 = bitcast i32 %1144 to float, !dbg !8
  %1146 = select i1 %.not36, float %1129, float %1145, !dbg !8
  %1147 = select i1 %.not36, float %1145, float %1135, !dbg !8
  %1148 = extractvalue [2 x <8 x float>] %109, 0, !dbg !8
  %1149 = extractelement <8 x float> %1148, i64 0, !dbg !8
  %1150 = extractelement <8 x float> %1148, i64 1, !dbg !8
  %1151 = extractelement <8 x float> %1148, i64 2, !dbg !8
  %1152 = extractelement <8 x float> %1148, i64 3, !dbg !8
  %1153 = select i1 %.not, float %1150, float %1149, !dbg !8
  %1154 = bitcast float %1153 to i32, !dbg !8
  %1155 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1154), !dbg !8
  %1156 = bitcast i32 %1155 to float, !dbg !8
  %1157 = select i1 %.not, float %1149, float %1156, !dbg !8
  %1158 = select i1 %.not, float %1156, float %1150, !dbg !8
  %1159 = select i1 %.not, float %1152, float %1151, !dbg !8
  %1160 = bitcast float %1159 to i32, !dbg !8
  %1161 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1160), !dbg !8
  %1162 = bitcast i32 %1161 to float, !dbg !8
  %1163 = select i1 %.not, float %1151, float %1162, !dbg !8
  %1164 = select i1 %.not, float %1162, float %1152, !dbg !8
  %1165 = select i1 %.not36, float %1163, float %1157, !dbg !8
  %1166 = bitcast float %1165 to i32, !dbg !8
  %1167 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1166), !dbg !8
  %1168 = bitcast i32 %1167 to float, !dbg !8
  %1169 = select i1 %.not36, float %1157, float %1168, !dbg !8
  %1170 = select i1 %.not36, float %1168, float %1163, !dbg !8
  %1171 = select i1 %.not36, float %1164, float %1158, !dbg !8
  %1172 = bitcast float %1171 to i32, !dbg !8
  %1173 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1172), !dbg !8
  %1174 = bitcast i32 %1173 to float, !dbg !8
  %1175 = select i1 %.not36, float %1158, float %1174, !dbg !8
  %1176 = select i1 %.not36, float %1174, float %1164, !dbg !8
  %1177 = extractelement <8 x float> %1148, i64 4, !dbg !8
  %1178 = extractelement <8 x float> %1148, i64 5, !dbg !8
  %1179 = extractelement <8 x float> %1148, i64 6, !dbg !8
  %1180 = extractelement <8 x float> %1148, i64 7, !dbg !8
  %1181 = select i1 %.not, float %1178, float %1177, !dbg !8
  %1182 = bitcast float %1181 to i32, !dbg !8
  %1183 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1182), !dbg !8
  %1184 = bitcast i32 %1183 to float, !dbg !8
  %1185 = select i1 %.not, float %1177, float %1184, !dbg !8
  %1186 = select i1 %.not, float %1184, float %1178, !dbg !8
  %1187 = select i1 %.not, float %1180, float %1179, !dbg !8
  %1188 = bitcast float %1187 to i32, !dbg !8
  %1189 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1188), !dbg !8
  %1190 = bitcast i32 %1189 to float, !dbg !8
  %1191 = select i1 %.not, float %1179, float %1190, !dbg !8
  %1192 = select i1 %.not, float %1190, float %1180, !dbg !8
  %1193 = select i1 %.not36, float %1191, float %1185, !dbg !8
  %1194 = bitcast float %1193 to i32, !dbg !8
  %1195 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1194), !dbg !8
  %1196 = bitcast i32 %1195 to float, !dbg !8
  %1197 = select i1 %.not36, float %1185, float %1196, !dbg !8
  %1198 = select i1 %.not36, float %1196, float %1191, !dbg !8
  %1199 = select i1 %.not36, float %1192, float %1186, !dbg !8
  %1200 = bitcast float %1199 to i32, !dbg !8
  %1201 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1200), !dbg !8
  %1202 = bitcast i32 %1201 to float, !dbg !8
  %1203 = select i1 %.not36, float %1186, float %1202, !dbg !8
  %1204 = select i1 %.not36, float %1202, float %1192, !dbg !8
  %1205 = extractvalue [2 x <8 x float>] %109, 1, !dbg !8
  %1206 = extractelement <8 x float> %1205, i64 0, !dbg !8
  %1207 = extractelement <8 x float> %1205, i64 1, !dbg !8
  %1208 = extractelement <8 x float> %1205, i64 2, !dbg !8
  %1209 = extractelement <8 x float> %1205, i64 3, !dbg !8
  %1210 = select i1 %.not, float %1207, float %1206, !dbg !8
  %1211 = bitcast float %1210 to i32, !dbg !8
  %1212 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1211), !dbg !8
  %1213 = bitcast i32 %1212 to float, !dbg !8
  %1214 = select i1 %.not, float %1206, float %1213, !dbg !8
  %1215 = select i1 %.not, float %1213, float %1207, !dbg !8
  %1216 = select i1 %.not, float %1209, float %1208, !dbg !8
  %1217 = bitcast float %1216 to i32, !dbg !8
  %1218 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1217), !dbg !8
  %1219 = bitcast i32 %1218 to float, !dbg !8
  %1220 = select i1 %.not, float %1208, float %1219, !dbg !8
  %1221 = select i1 %.not, float %1219, float %1209, !dbg !8
  %1222 = select i1 %.not36, float %1220, float %1214, !dbg !8
  %1223 = bitcast float %1222 to i32, !dbg !8
  %1224 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1223), !dbg !8
  %1225 = bitcast i32 %1224 to float, !dbg !8
  %1226 = select i1 %.not36, float %1214, float %1225, !dbg !8
  %1227 = select i1 %.not36, float %1225, float %1220, !dbg !8
  %1228 = select i1 %.not36, float %1221, float %1215, !dbg !8
  %1229 = bitcast float %1228 to i32, !dbg !8
  %1230 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1229), !dbg !8
  %1231 = bitcast i32 %1230 to float, !dbg !8
  %1232 = select i1 %.not36, float %1215, float %1231, !dbg !8
  %1233 = select i1 %.not36, float %1231, float %1221, !dbg !8
  %1234 = extractelement <8 x float> %1205, i64 4, !dbg !8
  %1235 = extractelement <8 x float> %1205, i64 5, !dbg !8
  %1236 = extractelement <8 x float> %1205, i64 6, !dbg !8
  %1237 = extractelement <8 x float> %1205, i64 7, !dbg !8
  %1238 = select i1 %.not, float %1235, float %1234, !dbg !8
  %1239 = bitcast float %1238 to i32, !dbg !8
  %1240 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1239), !dbg !8
  %1241 = bitcast i32 %1240 to float, !dbg !8
  %1242 = select i1 %.not, float %1234, float %1241, !dbg !8
  %1243 = select i1 %.not, float %1241, float %1235, !dbg !8
  %1244 = select i1 %.not, float %1237, float %1236, !dbg !8
  %1245 = bitcast float %1244 to i32, !dbg !8
  %1246 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1245), !dbg !8
  %1247 = bitcast i32 %1246 to float, !dbg !8
  %1248 = select i1 %.not, float %1236, float %1247, !dbg !8
  %1249 = select i1 %.not, float %1247, float %1237, !dbg !8
  %1250 = select i1 %.not36, float %1248, float %1242, !dbg !8
  %1251 = bitcast float %1250 to i32, !dbg !8
  %1252 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1251), !dbg !8
  %1253 = bitcast i32 %1252 to float, !dbg !8
  %1254 = select i1 %.not36, float %1242, float %1253, !dbg !8
  %1255 = select i1 %.not36, float %1253, float %1248, !dbg !8
  %1256 = select i1 %.not36, float %1249, float %1243, !dbg !8
  %1257 = bitcast float %1256 to i32, !dbg !8
  %1258 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1257), !dbg !8
  %1259 = bitcast i32 %1258 to float, !dbg !8
  %1260 = select i1 %.not36, float %1243, float %1259, !dbg !8
  %1261 = select i1 %.not36, float %1259, float %1249, !dbg !8
  %1262 = extractvalue [2 x <8 x float>] %108, 0, !dbg !8
  %1263 = extractelement <8 x float> %1262, i64 0, !dbg !8
  %1264 = extractelement <8 x float> %1262, i64 1, !dbg !8
  %1265 = extractelement <8 x float> %1262, i64 2, !dbg !8
  %1266 = extractelement <8 x float> %1262, i64 3, !dbg !8
  %1267 = select i1 %.not, float %1264, float %1263, !dbg !8
  %1268 = bitcast float %1267 to i32, !dbg !8
  %1269 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1268), !dbg !8
  %1270 = bitcast i32 %1269 to float, !dbg !8
  %1271 = select i1 %.not, float %1263, float %1270, !dbg !8
  %1272 = select i1 %.not, float %1270, float %1264, !dbg !8
  %1273 = select i1 %.not, float %1266, float %1265, !dbg !8
  %1274 = bitcast float %1273 to i32, !dbg !8
  %1275 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1274), !dbg !8
  %1276 = bitcast i32 %1275 to float, !dbg !8
  %1277 = select i1 %.not, float %1265, float %1276, !dbg !8
  %1278 = select i1 %.not, float %1276, float %1266, !dbg !8
  %1279 = select i1 %.not36, float %1277, float %1271, !dbg !8
  %1280 = bitcast float %1279 to i32, !dbg !8
  %1281 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1280), !dbg !8
  %1282 = bitcast i32 %1281 to float, !dbg !8
  %1283 = select i1 %.not36, float %1271, float %1282, !dbg !8
  %1284 = select i1 %.not36, float %1282, float %1277, !dbg !8
  %1285 = select i1 %.not36, float %1278, float %1272, !dbg !8
  %1286 = bitcast float %1285 to i32, !dbg !8
  %1287 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1286), !dbg !8
  %1288 = bitcast i32 %1287 to float, !dbg !8
  %1289 = select i1 %.not36, float %1272, float %1288, !dbg !8
  %1290 = select i1 %.not36, float %1288, float %1278, !dbg !8
  %1291 = extractelement <8 x float> %1262, i64 4, !dbg !8
  %1292 = extractelement <8 x float> %1262, i64 5, !dbg !8
  %1293 = extractelement <8 x float> %1262, i64 6, !dbg !8
  %1294 = extractelement <8 x float> %1262, i64 7, !dbg !8
  %1295 = select i1 %.not, float %1292, float %1291, !dbg !8
  %1296 = bitcast float %1295 to i32, !dbg !8
  %1297 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1296), !dbg !8
  %1298 = bitcast i32 %1297 to float, !dbg !8
  %1299 = select i1 %.not, float %1291, float %1298, !dbg !8
  %1300 = select i1 %.not, float %1298, float %1292, !dbg !8
  %1301 = select i1 %.not, float %1294, float %1293, !dbg !8
  %1302 = bitcast float %1301 to i32, !dbg !8
  %1303 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1302), !dbg !8
  %1304 = bitcast i32 %1303 to float, !dbg !8
  %1305 = select i1 %.not, float %1293, float %1304, !dbg !8
  %1306 = select i1 %.not, float %1304, float %1294, !dbg !8
  %1307 = select i1 %.not36, float %1305, float %1299, !dbg !8
  %1308 = bitcast float %1307 to i32, !dbg !8
  %1309 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1308), !dbg !8
  %1310 = bitcast i32 %1309 to float, !dbg !8
  %1311 = select i1 %.not36, float %1299, float %1310, !dbg !8
  %1312 = select i1 %.not36, float %1310, float %1305, !dbg !8
  %1313 = select i1 %.not36, float %1306, float %1300, !dbg !8
  %1314 = bitcast float %1313 to i32, !dbg !8
  %1315 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1314), !dbg !8
  %1316 = bitcast i32 %1315 to float, !dbg !8
  %1317 = select i1 %.not36, float %1300, float %1316, !dbg !8
  %1318 = select i1 %.not36, float %1316, float %1306, !dbg !8
  %1319 = extractvalue [2 x <8 x float>] %108, 1, !dbg !8
  %1320 = extractelement <8 x float> %1319, i64 0, !dbg !8
  %1321 = extractelement <8 x float> %1319, i64 1, !dbg !8
  %1322 = extractelement <8 x float> %1319, i64 2, !dbg !8
  %1323 = extractelement <8 x float> %1319, i64 3, !dbg !8
  %1324 = select i1 %.not, float %1321, float %1320, !dbg !8
  %1325 = bitcast float %1324 to i32, !dbg !8
  %1326 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1325), !dbg !8
  %1327 = bitcast i32 %1326 to float, !dbg !8
  %1328 = select i1 %.not, float %1320, float %1327, !dbg !8
  %1329 = select i1 %.not, float %1327, float %1321, !dbg !8
  %1330 = select i1 %.not, float %1323, float %1322, !dbg !8
  %1331 = bitcast float %1330 to i32, !dbg !8
  %1332 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1331), !dbg !8
  %1333 = bitcast i32 %1332 to float, !dbg !8
  %1334 = select i1 %.not, float %1322, float %1333, !dbg !8
  %1335 = select i1 %.not, float %1333, float %1323, !dbg !8
  %1336 = select i1 %.not36, float %1334, float %1328, !dbg !8
  %1337 = bitcast float %1336 to i32, !dbg !8
  %1338 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1337), !dbg !8
  %1339 = bitcast i32 %1338 to float, !dbg !8
  %1340 = select i1 %.not36, float %1328, float %1339, !dbg !8
  %1341 = select i1 %.not36, float %1339, float %1334, !dbg !8
  %1342 = select i1 %.not36, float %1335, float %1329, !dbg !8
  %1343 = bitcast float %1342 to i32, !dbg !8
  %1344 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1343), !dbg !8
  %1345 = bitcast i32 %1344 to float, !dbg !8
  %1346 = select i1 %.not36, float %1329, float %1345, !dbg !8
  %1347 = select i1 %.not36, float %1345, float %1335, !dbg !8
  %1348 = extractelement <8 x float> %1319, i64 4, !dbg !8
  %1349 = extractelement <8 x float> %1319, i64 5, !dbg !8
  %1350 = extractelement <8 x float> %1319, i64 6, !dbg !8
  %1351 = extractelement <8 x float> %1319, i64 7, !dbg !8
  %1352 = select i1 %.not, float %1349, float %1348, !dbg !8
  %1353 = bitcast float %1352 to i32, !dbg !8
  %1354 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1353), !dbg !8
  %1355 = bitcast i32 %1354 to float, !dbg !8
  %1356 = select i1 %.not, float %1348, float %1355, !dbg !8
  %1357 = select i1 %.not, float %1355, float %1349, !dbg !8
  %1358 = select i1 %.not, float %1351, float %1350, !dbg !8
  %1359 = bitcast float %1358 to i32, !dbg !8
  %1360 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1359), !dbg !8
  %1361 = bitcast i32 %1360 to float, !dbg !8
  %1362 = select i1 %.not, float %1350, float %1361, !dbg !8
  %1363 = select i1 %.not, float %1361, float %1351, !dbg !8
  %1364 = select i1 %.not36, float %1362, float %1356, !dbg !8
  %1365 = bitcast float %1364 to i32, !dbg !8
  %1366 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1365), !dbg !8
  %1367 = bitcast i32 %1366 to float, !dbg !8
  %1368 = select i1 %.not36, float %1356, float %1367, !dbg !8
  %1369 = select i1 %.not36, float %1367, float %1362, !dbg !8
  %1370 = select i1 %.not36, float %1363, float %1357, !dbg !8
  %1371 = bitcast float %1370 to i32, !dbg !8
  %1372 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1371), !dbg !8
  %1373 = bitcast i32 %1372 to float, !dbg !8
  %1374 = select i1 %.not36, float %1357, float %1373, !dbg !8
  %1375 = select i1 %.not36, float %1373, float %1363, !dbg !8
  %1376 = extractvalue [2 x <8 x float>] %105, 0, !dbg !8
  %1377 = extractelement <8 x float> %1376, i64 0, !dbg !8
  %1378 = extractelement <8 x float> %1376, i64 1, !dbg !8
  %1379 = extractelement <8 x float> %1376, i64 2, !dbg !8
  %1380 = extractelement <8 x float> %1376, i64 3, !dbg !8
  %1381 = select i1 %.not, float %1378, float %1377, !dbg !8
  %1382 = bitcast float %1381 to i32, !dbg !8
  %1383 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1382), !dbg !8
  %1384 = bitcast i32 %1383 to float, !dbg !8
  %1385 = select i1 %.not, float %1377, float %1384, !dbg !8
  %1386 = select i1 %.not, float %1384, float %1378, !dbg !8
  %1387 = select i1 %.not, float %1380, float %1379, !dbg !8
  %1388 = bitcast float %1387 to i32, !dbg !8
  %1389 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1388), !dbg !8
  %1390 = bitcast i32 %1389 to float, !dbg !8
  %1391 = select i1 %.not, float %1379, float %1390, !dbg !8
  %1392 = select i1 %.not, float %1390, float %1380, !dbg !8
  %1393 = select i1 %.not36, float %1391, float %1385, !dbg !8
  %1394 = bitcast float %1393 to i32, !dbg !8
  %1395 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1394), !dbg !8
  %1396 = select i1 %.not36, float %1392, float %1386, !dbg !8
  %1397 = bitcast float %1396 to i32, !dbg !8
  %1398 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1397), !dbg !8
  %1399 = extractelement <8 x float> %1376, i64 4, !dbg !8
  %1400 = extractelement <8 x float> %1376, i64 5, !dbg !8
  %1401 = extractelement <8 x float> %1376, i64 6, !dbg !8
  %1402 = extractelement <8 x float> %1376, i64 7, !dbg !8
  %1403 = select i1 %.not, float %1400, float %1399, !dbg !8
  %1404 = bitcast float %1403 to i32, !dbg !8
  %1405 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1404), !dbg !8
  %1406 = bitcast i32 %1405 to float, !dbg !8
  %1407 = select i1 %.not, float %1399, float %1406, !dbg !8
  %1408 = select i1 %.not, float %1406, float %1400, !dbg !8
  %1409 = select i1 %.not, float %1402, float %1401, !dbg !8
  %1410 = bitcast float %1409 to i32, !dbg !8
  %1411 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1410), !dbg !8
  %1412 = bitcast i32 %1411 to float, !dbg !8
  %1413 = select i1 %.not, float %1401, float %1412, !dbg !8
  %1414 = select i1 %.not, float %1412, float %1402, !dbg !8
  %1415 = select i1 %.not36, float %1413, float %1407, !dbg !8
  %1416 = bitcast float %1415 to i32, !dbg !8
  %1417 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1416), !dbg !8
  %1418 = select i1 %.not36, float %1414, float %1408, !dbg !8
  %1419 = bitcast float %1418 to i32, !dbg !8
  %1420 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1419), !dbg !8
  %1421 = extractvalue [2 x <8 x float>] %105, 1, !dbg !8
  %1422 = extractelement <8 x float> %1421, i64 0, !dbg !8
  %1423 = extractelement <8 x float> %1421, i64 1, !dbg !8
  %1424 = extractelement <8 x float> %1421, i64 2, !dbg !8
  %1425 = extractelement <8 x float> %1421, i64 3, !dbg !8
  %1426 = select i1 %.not, float %1423, float %1422, !dbg !8
  %1427 = bitcast float %1426 to i32, !dbg !8
  %1428 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1427), !dbg !8
  %1429 = bitcast i32 %1428 to float, !dbg !8
  %1430 = select i1 %.not, float %1422, float %1429, !dbg !8
  %1431 = select i1 %.not, float %1429, float %1423, !dbg !8
  %1432 = select i1 %.not, float %1425, float %1424, !dbg !8
  %1433 = bitcast float %1432 to i32, !dbg !8
  %1434 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1433), !dbg !8
  %1435 = bitcast i32 %1434 to float, !dbg !8
  %1436 = select i1 %.not, float %1424, float %1435, !dbg !8
  %1437 = select i1 %.not, float %1435, float %1425, !dbg !8
  %1438 = select i1 %.not36, float %1436, float %1430, !dbg !8
  %1439 = bitcast float %1438 to i32, !dbg !8
  %1440 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1439), !dbg !8
  %1441 = select i1 %.not36, float %1437, float %1431, !dbg !8
  %1442 = bitcast float %1441 to i32, !dbg !8
  %1443 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1442), !dbg !8
  %1444 = extractelement <8 x float> %1421, i64 4, !dbg !8
  %1445 = extractelement <8 x float> %1421, i64 5, !dbg !8
  %1446 = extractelement <8 x float> %1421, i64 6, !dbg !8
  %1447 = extractelement <8 x float> %1421, i64 7, !dbg !8
  %1448 = select i1 %.not, float %1445, float %1444, !dbg !8
  %1449 = bitcast float %1448 to i32, !dbg !8
  %1450 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1449), !dbg !8
  %1451 = bitcast i32 %1450 to float, !dbg !8
  %1452 = select i1 %.not, float %1444, float %1451, !dbg !8
  %1453 = select i1 %.not, float %1451, float %1445, !dbg !8
  %1454 = select i1 %.not, float %1447, float %1446, !dbg !8
  %1455 = bitcast float %1454 to i32, !dbg !8
  %1456 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1455), !dbg !8
  %1457 = bitcast i32 %1456 to float, !dbg !8
  %1458 = select i1 %.not, float %1446, float %1457, !dbg !8
  %1459 = select i1 %.not, float %1457, float %1447, !dbg !8
  %1460 = select i1 %.not36, float %1458, float %1452, !dbg !8
  %1461 = bitcast float %1460 to i32, !dbg !8
  %1462 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1461), !dbg !8
  %1463 = select i1 %.not36, float %1459, float %1453, !dbg !8
  %1464 = bitcast float %1463 to i32, !dbg !8
  %1465 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1464), !dbg !8
  %1466 = shufflevector <1 x float> %1031, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !17
  %1467 = shufflevector <1 x float> %1026, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !17
  %1468 = bitcast i32 %1462 to float, !dbg !8
  %1469 = select i1 %.not36, float %1452, float %1468, !dbg !8
  %1470 = insertelement <4 x float> poison, float %1469, i64 0, !dbg !17
  %1471 = bitcast i32 %1465 to float, !dbg !8
  %1472 = select i1 %.not36, float %1453, float %1471, !dbg !8
  %1473 = insertelement <4 x float> %1470, float %1472, i64 1, !dbg !17
  %1474 = select i1 %.not36, float %1468, float %1458, !dbg !8
  %1475 = insertelement <4 x float> %1473, float %1474, i64 2, !dbg !17
  %1476 = select i1 %.not36, float %1471, float %1459, !dbg !8
  %1477 = insertelement <4 x float> %1475, float %1476, i64 3, !dbg !17
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1467, !dbg !17
  %normalized = fmul <4 x float> %1477, %row.reciprocal, !dbg !17
  %1478 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !18
  %1479 = extractelement <4 x half> %1478, i64 3, !dbg !18
  %1480 = extractelement <4 x half> %1478, i64 2, !dbg !18
  %1481 = extractelement <4 x half> %1478, i64 1, !dbg !18
  %1482 = extractelement <4 x half> %1478, i64 0, !dbg !18
  %1483 = bitcast i32 %1440 to float, !dbg !8
  %1484 = select i1 %.not36, float %1430, float %1483, !dbg !8
  %1485 = insertelement <4 x float> poison, float %1484, i64 0, !dbg !17
  %1486 = bitcast i32 %1443 to float, !dbg !8
  %1487 = select i1 %.not36, float %1431, float %1486, !dbg !8
  %1488 = insertelement <4 x float> %1485, float %1487, i64 1, !dbg !17
  %1489 = select i1 %.not36, float %1483, float %1436, !dbg !8
  %1490 = insertelement <4 x float> %1488, float %1489, i64 2, !dbg !17
  %1491 = select i1 %.not36, float %1486, float %1437, !dbg !8
  %1492 = insertelement <4 x float> %1490, float %1491, i64 3, !dbg !17
  %normalized37 = fmul <4 x float> %1492, %row.reciprocal, !dbg !17
  %1493 = fptrunc <4 x float> %normalized37 to <4 x half>, !dbg !18
  %1494 = extractelement <4 x half> %1493, i64 3, !dbg !18
  %1495 = extractelement <4 x half> %1493, i64 2, !dbg !18
  %1496 = extractelement <4 x half> %1493, i64 1, !dbg !18
  %1497 = extractelement <4 x half> %1493, i64 0, !dbg !18
  %1498 = bitcast i32 %1417 to float, !dbg !8
  %1499 = select i1 %.not36, float %1407, float %1498, !dbg !8
  %1500 = insertelement <4 x float> poison, float %1499, i64 0, !dbg !17
  %1501 = bitcast i32 %1420 to float, !dbg !8
  %1502 = select i1 %.not36, float %1408, float %1501, !dbg !8
  %1503 = insertelement <4 x float> %1500, float %1502, i64 1, !dbg !17
  %1504 = select i1 %.not36, float %1498, float %1413, !dbg !8
  %1505 = insertelement <4 x float> %1503, float %1504, i64 2, !dbg !17
  %1506 = select i1 %.not36, float %1501, float %1414, !dbg !8
  %1507 = insertelement <4 x float> %1505, float %1506, i64 3, !dbg !17
  %row.reciprocal38 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1466, !dbg !17
  %normalized39 = fmul <4 x float> %1507, %row.reciprocal38, !dbg !17
  %1508 = fptrunc <4 x float> %normalized39 to <4 x half>, !dbg !18
  %1509 = extractelement <4 x half> %1508, i64 3, !dbg !18
  %1510 = extractelement <4 x half> %1508, i64 2, !dbg !18
  %1511 = extractelement <4 x half> %1508, i64 1, !dbg !18
  %1512 = extractelement <4 x half> %1508, i64 0, !dbg !18
  %1513 = bitcast i32 %1395 to float, !dbg !8
  %1514 = select i1 %.not36, float %1385, float %1513, !dbg !8
  %1515 = insertelement <4 x float> poison, float %1514, i64 0, !dbg !17
  %1516 = bitcast i32 %1398 to float, !dbg !8
  %1517 = select i1 %.not36, float %1386, float %1516, !dbg !8
  %1518 = insertelement <4 x float> %1515, float %1517, i64 1, !dbg !17
  %1519 = select i1 %.not36, float %1513, float %1391, !dbg !8
  %1520 = insertelement <4 x float> %1518, float %1519, i64 2, !dbg !17
  %1521 = select i1 %.not36, float %1516, float %1392, !dbg !8
  %1522 = insertelement <4 x float> %1520, float %1521, i64 3, !dbg !17
  %normalized40 = fmul <4 x float> %1522, %row.reciprocal38, !dbg !17
  %1523 = fptrunc <4 x float> %normalized40 to <4 x half>, !dbg !18
  %1524 = extractelement <4 x half> %1523, i64 3, !dbg !18
  %1525 = extractelement <4 x half> %1523, i64 2, !dbg !18
  %1526 = extractelement <4 x half> %1523, i64 1, !dbg !18
  %1527 = extractelement <4 x half> %1523, i64 0, !dbg !18
  %address.thread958 = call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #9, !dbg !19, !range !32, !frisk.register_lifetimes !1
  %address.local959 = zext nneg i32 %address.thread958 to i64, !dbg !9
  %address.local960 = add nuw i64 %32, %address.local959, !dbg !19
  %1528 = lshr i64 %address.local959, 2, !dbg !19
  %address.local962 = and i64 %1528, 48, !dbg !19
  %address.local963 = add i64 %address.local960, %address.local962, !dbg !19
  %address.local964 = and i64 %address.local959, 240, !dbg !19
  %address.local965 = sub i64 %address.local963, %address.local964, !dbg !19
  %address.local966 = add i64 %address.local965, %33, !dbg !19
  %address.local967 = shl nuw nsw i64 %address.local966, 7, !dbg !19
  %1529 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %46, !dbg !19
  %address.local969 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %1529, i64 %address.local967, !dbg !19
  %address.local971 = and i64 %1528, 12, !dbg !12
  %address.local972 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local971, !dbg !19
  store half %1527, ptr addrspace(1) %address.local972, align 2, !dbg !19
  %address.local974 = or disjoint i64 %address.local971, 1, !dbg !19
  %address.local975 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local974, !dbg !19
  store half %1526, ptr addrspace(1) %address.local975, align 2, !dbg !19
  %address.local977 = or disjoint i64 %address.local971, 2, !dbg !19
  %address.local978 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local977, !dbg !19
  store half %1525, ptr addrspace(1) %address.local978, align 2, !dbg !19
  %address.local980 = or disjoint i64 %address.local971, 3, !dbg !19
  %address.local981 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local980, !dbg !19
  store half %1524, ptr addrspace(1) %address.local981, align 2, !dbg !19
  %address.local983 = or disjoint i64 %address.local971, 16, !dbg !19
  %address.local984 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local983, !dbg !19
  store half %1512, ptr addrspace(1) %address.local984, align 2, !dbg !19
  %address.local986 = or disjoint i64 %address.local971, 17, !dbg !19
  %address.local987 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local986, !dbg !19
  store half %1511, ptr addrspace(1) %address.local987, align 2, !dbg !19
  %address.local989 = or disjoint i64 %address.local971, 18, !dbg !19
  %address.local990 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local989, !dbg !19
  store half %1510, ptr addrspace(1) %address.local990, align 2, !dbg !19
  %address.local992 = or disjoint i64 %address.local971, 19, !dbg !19
  %address.local993 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local992, !dbg !19
  store half %1509, ptr addrspace(1) %address.local993, align 2, !dbg !19
  %address.local997 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local969, i64 16384, !dbg !19
  %address.local998 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local971, !dbg !19
  store half %1497, ptr addrspace(1) %address.local998, align 2, !dbg !19
  %address.local1000 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local974, !dbg !19
  store half %1496, ptr addrspace(1) %address.local1000, align 2, !dbg !19
  %address.local1002 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local977, !dbg !19
  store half %1495, ptr addrspace(1) %address.local1002, align 2, !dbg !19
  %address.local1004 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local980, !dbg !19
  store half %1494, ptr addrspace(1) %address.local1004, align 2, !dbg !19
  %address.local1006 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local983, !dbg !19
  store half %1482, ptr addrspace(1) %address.local1006, align 2, !dbg !19
  %address.local1008 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local986, !dbg !19
  store half %1481, ptr addrspace(1) %address.local1008, align 2, !dbg !19
  %address.local1010 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local989, !dbg !19
  store half %1480, ptr addrspace(1) %address.local1010, align 2, !dbg !19
  %address.local1012 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local992, !dbg !19
  store half %1479, ptr addrspace(1) %address.local1012, align 2, !dbg !19
  %1530 = insertelement <4 x float> poison, float %1368, i64 0, !dbg !17
  %1531 = insertelement <4 x float> %1530, float %1374, i64 1, !dbg !17
  %1532 = insertelement <4 x float> %1531, float %1369, i64 2, !dbg !17
  %1533 = insertelement <4 x float> %1532, float %1375, i64 3, !dbg !17
  %normalized42 = fmul <4 x float> %1533, %row.reciprocal, !dbg !17
  %1534 = fptrunc <4 x float> %normalized42 to <4 x half>, !dbg !18
  %1535 = extractelement <4 x half> %1534, i64 3, !dbg !18
  %1536 = extractelement <4 x half> %1534, i64 2, !dbg !18
  %1537 = extractelement <4 x half> %1534, i64 1, !dbg !18
  %1538 = extractelement <4 x half> %1534, i64 0, !dbg !18
  %1539 = insertelement <4 x float> poison, float %1340, i64 0, !dbg !17
  %1540 = insertelement <4 x float> %1539, float %1346, i64 1, !dbg !17
  %1541 = insertelement <4 x float> %1540, float %1341, i64 2, !dbg !17
  %1542 = insertelement <4 x float> %1541, float %1347, i64 3, !dbg !17
  %normalized43 = fmul <4 x float> %1542, %row.reciprocal, !dbg !17
  %1543 = fptrunc <4 x float> %normalized43 to <4 x half>, !dbg !18
  %1544 = extractelement <4 x half> %1543, i64 3, !dbg !18
  %1545 = extractelement <4 x half> %1543, i64 2, !dbg !18
  %1546 = extractelement <4 x half> %1543, i64 1, !dbg !18
  %1547 = extractelement <4 x half> %1543, i64 0, !dbg !18
  %1548 = insertelement <4 x float> poison, float %1311, i64 0, !dbg !17
  %1549 = insertelement <4 x float> %1548, float %1317, i64 1, !dbg !17
  %1550 = insertelement <4 x float> %1549, float %1312, i64 2, !dbg !17
  %1551 = insertelement <4 x float> %1550, float %1318, i64 3, !dbg !17
  %normalized45 = fmul <4 x float> %1551, %row.reciprocal38, !dbg !17
  %1552 = fptrunc <4 x float> %normalized45 to <4 x half>, !dbg !18
  %1553 = extractelement <4 x half> %1552, i64 3, !dbg !18
  %1554 = extractelement <4 x half> %1552, i64 2, !dbg !18
  %1555 = extractelement <4 x half> %1552, i64 1, !dbg !18
  %1556 = extractelement <4 x half> %1552, i64 0, !dbg !18
  %1557 = insertelement <4 x float> poison, float %1283, i64 0, !dbg !17
  %1558 = insertelement <4 x float> %1557, float %1289, i64 1, !dbg !17
  %1559 = insertelement <4 x float> %1558, float %1284, i64 2, !dbg !17
  %1560 = insertelement <4 x float> %1559, float %1290, i64 3, !dbg !17
  %normalized46 = fmul <4 x float> %1560, %row.reciprocal38, !dbg !17
  %1561 = fptrunc <4 x float> %normalized46 to <4 x half>, !dbg !18
  %1562 = extractelement <4 x half> %1561, i64 3, !dbg !18
  %1563 = extractelement <4 x half> %1561, i64 2, !dbg !18
  %1564 = extractelement <4 x half> %1561, i64 1, !dbg !18
  %1565 = extractelement <4 x half> %1561, i64 0, !dbg !18
  %address.local1027 = or disjoint i64 %address.local971, 32, !dbg !19
  %address.local1028 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1027, !dbg !19
  store half %1565, ptr addrspace(1) %address.local1028, align 2, !dbg !19
  %address.local1030 = or disjoint i64 %address.local971, 33, !dbg !19
  %address.local1031 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1030, !dbg !19
  store half %1564, ptr addrspace(1) %address.local1031, align 2, !dbg !19
  %address.local1033 = or disjoint i64 %address.local971, 34, !dbg !19
  %address.local1034 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1033, !dbg !19
  store half %1563, ptr addrspace(1) %address.local1034, align 2, !dbg !19
  %address.local1036 = or disjoint i64 %address.local971, 35, !dbg !19
  %address.local1037 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1036, !dbg !19
  store half %1562, ptr addrspace(1) %address.local1037, align 2, !dbg !19
  %address.local1039 = or disjoint i64 %address.local971, 48, !dbg !19
  %address.local1040 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1039, !dbg !19
  store half %1556, ptr addrspace(1) %address.local1040, align 2, !dbg !19
  %address.local1042 = or disjoint i64 %address.local971, 49, !dbg !19
  %address.local1043 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1042, !dbg !19
  store half %1555, ptr addrspace(1) %address.local1043, align 2, !dbg !19
  %address.local1045 = or disjoint i64 %address.local971, 50, !dbg !19
  %address.local1046 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1045, !dbg !19
  store half %1554, ptr addrspace(1) %address.local1046, align 2, !dbg !19
  %address.local1048 = or i64 %1528, 51, !dbg !19
  %address.local1049 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1048, !dbg !19
  store half %1553, ptr addrspace(1) %address.local1049, align 2, !dbg !19
  %address.local1054 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1027, !dbg !19
  store half %1547, ptr addrspace(1) %address.local1054, align 2, !dbg !19
  %address.local1056 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1030, !dbg !19
  store half %1546, ptr addrspace(1) %address.local1056, align 2, !dbg !19
  %address.local1058 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1033, !dbg !19
  store half %1545, ptr addrspace(1) %address.local1058, align 2, !dbg !19
  %address.local1060 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1036, !dbg !19
  store half %1544, ptr addrspace(1) %address.local1060, align 2, !dbg !19
  %address.local1062 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1039, !dbg !19
  store half %1538, ptr addrspace(1) %address.local1062, align 2, !dbg !19
  %address.local1064 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1042, !dbg !19
  store half %1537, ptr addrspace(1) %address.local1064, align 2, !dbg !19
  %address.local1066 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1045, !dbg !19
  store half %1536, ptr addrspace(1) %address.local1066, align 2, !dbg !19
  %address.local1068 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1048, !dbg !19
  store half %1535, ptr addrspace(1) %address.local1068, align 2, !dbg !19
  %1566 = insertelement <4 x float> poison, float %1254, i64 0, !dbg !17
  %1567 = insertelement <4 x float> %1566, float %1260, i64 1, !dbg !17
  %1568 = insertelement <4 x float> %1567, float %1255, i64 2, !dbg !17
  %1569 = insertelement <4 x float> %1568, float %1261, i64 3, !dbg !17
  %normalized48 = fmul <4 x float> %1569, %row.reciprocal, !dbg !17
  %1570 = fptrunc <4 x float> %normalized48 to <4 x half>, !dbg !18
  %1571 = extractelement <4 x half> %1570, i64 3, !dbg !18
  %1572 = extractelement <4 x half> %1570, i64 2, !dbg !18
  %1573 = extractelement <4 x half> %1570, i64 1, !dbg !18
  %1574 = extractelement <4 x half> %1570, i64 0, !dbg !18
  %1575 = insertelement <4 x float> poison, float %1226, i64 0, !dbg !17
  %1576 = insertelement <4 x float> %1575, float %1232, i64 1, !dbg !17
  %1577 = insertelement <4 x float> %1576, float %1227, i64 2, !dbg !17
  %1578 = insertelement <4 x float> %1577, float %1233, i64 3, !dbg !17
  %normalized49 = fmul <4 x float> %1578, %row.reciprocal, !dbg !17
  %1579 = fptrunc <4 x float> %normalized49 to <4 x half>, !dbg !18
  %1580 = extractelement <4 x half> %1579, i64 3, !dbg !18
  %1581 = extractelement <4 x half> %1579, i64 2, !dbg !18
  %1582 = extractelement <4 x half> %1579, i64 1, !dbg !18
  %1583 = extractelement <4 x half> %1579, i64 0, !dbg !18
  %1584 = insertelement <4 x float> poison, float %1197, i64 0, !dbg !17
  %1585 = insertelement <4 x float> %1584, float %1203, i64 1, !dbg !17
  %1586 = insertelement <4 x float> %1585, float %1198, i64 2, !dbg !17
  %1587 = insertelement <4 x float> %1586, float %1204, i64 3, !dbg !17
  %normalized51 = fmul <4 x float> %1587, %row.reciprocal38, !dbg !17
  %1588 = fptrunc <4 x float> %normalized51 to <4 x half>, !dbg !18
  %1589 = extractelement <4 x half> %1588, i64 3, !dbg !18
  %1590 = extractelement <4 x half> %1588, i64 2, !dbg !18
  %1591 = extractelement <4 x half> %1588, i64 1, !dbg !18
  %1592 = extractelement <4 x half> %1588, i64 0, !dbg !18
  %1593 = insertelement <4 x float> poison, float %1169, i64 0, !dbg !17
  %1594 = insertelement <4 x float> %1593, float %1175, i64 1, !dbg !17
  %1595 = insertelement <4 x float> %1594, float %1170, i64 2, !dbg !17
  %1596 = insertelement <4 x float> %1595, float %1176, i64 3, !dbg !17
  %normalized52 = fmul <4 x float> %1596, %row.reciprocal38, !dbg !17
  %1597 = fptrunc <4 x float> %normalized52 to <4 x half>, !dbg !18
  %1598 = extractelement <4 x half> %1597, i64 3, !dbg !18
  %1599 = extractelement <4 x half> %1597, i64 2, !dbg !18
  %1600 = extractelement <4 x half> %1597, i64 1, !dbg !18
  %1601 = extractelement <4 x half> %1597, i64 0, !dbg !18
  %address.local1083 = or disjoint i64 %address.local971, 64, !dbg !19
  %address.local1084 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1083, !dbg !19
  store half %1601, ptr addrspace(1) %address.local1084, align 2, !dbg !19
  %address.local1086 = or disjoint i64 %address.local971, 65, !dbg !19
  %address.local1087 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1086, !dbg !19
  store half %1600, ptr addrspace(1) %address.local1087, align 2, !dbg !19
  %address.local1089 = or disjoint i64 %address.local971, 66, !dbg !19
  %address.local1090 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1089, !dbg !19
  store half %1599, ptr addrspace(1) %address.local1090, align 2, !dbg !19
  %address.local1092 = or disjoint i64 %address.local971, 67, !dbg !19
  %address.local1093 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1092, !dbg !19
  store half %1598, ptr addrspace(1) %address.local1093, align 2, !dbg !19
  %address.local1095 = or disjoint i64 %address.local971, 80, !dbg !19
  %address.local1096 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1095, !dbg !19
  store half %1592, ptr addrspace(1) %address.local1096, align 2, !dbg !19
  %address.local1098 = or disjoint i64 %address.local971, 81, !dbg !19
  %address.local1099 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1098, !dbg !19
  store half %1591, ptr addrspace(1) %address.local1099, align 2, !dbg !19
  %address.local1101 = or disjoint i64 %address.local971, 82, !dbg !19
  %address.local1102 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1101, !dbg !19
  store half %1590, ptr addrspace(1) %address.local1102, align 2, !dbg !19
  %address.local1104 = or disjoint i64 %address.local971, 83, !dbg !19
  %address.local1105 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1104, !dbg !19
  store half %1589, ptr addrspace(1) %address.local1105, align 2, !dbg !19
  %address.local1110 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1083, !dbg !19
  store half %1583, ptr addrspace(1) %address.local1110, align 2, !dbg !19
  %address.local1112 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1086, !dbg !19
  store half %1582, ptr addrspace(1) %address.local1112, align 2, !dbg !19
  %address.local1114 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1089, !dbg !19
  store half %1581, ptr addrspace(1) %address.local1114, align 2, !dbg !19
  %address.local1116 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1092, !dbg !19
  store half %1580, ptr addrspace(1) %address.local1116, align 2, !dbg !19
  %address.local1118 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1095, !dbg !19
  store half %1574, ptr addrspace(1) %address.local1118, align 2, !dbg !19
  %address.local1120 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1098, !dbg !19
  store half %1573, ptr addrspace(1) %address.local1120, align 2, !dbg !19
  %address.local1122 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1101, !dbg !19
  store half %1572, ptr addrspace(1) %address.local1122, align 2, !dbg !19
  %address.local1124 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1104, !dbg !19
  store half %1571, ptr addrspace(1) %address.local1124, align 2, !dbg !19
  %1602 = insertelement <4 x float> poison, float %1140, i64 0, !dbg !17
  %1603 = insertelement <4 x float> %1602, float %1146, i64 1, !dbg !17
  %1604 = insertelement <4 x float> %1603, float %1141, i64 2, !dbg !17
  %1605 = insertelement <4 x float> %1604, float %1147, i64 3, !dbg !17
  %normalized54 = fmul <4 x float> %1605, %row.reciprocal, !dbg !17
  %1606 = fptrunc <4 x float> %normalized54 to <4 x half>, !dbg !18
  %1607 = extractelement <4 x half> %1606, i64 3, !dbg !18
  %1608 = extractelement <4 x half> %1606, i64 2, !dbg !18
  %1609 = extractelement <4 x half> %1606, i64 1, !dbg !18
  %1610 = extractelement <4 x half> %1606, i64 0, !dbg !18
  %1611 = insertelement <4 x float> poison, float %1112, i64 0, !dbg !17
  %1612 = insertelement <4 x float> %1611, float %1118, i64 1, !dbg !17
  %1613 = insertelement <4 x float> %1612, float %1113, i64 2, !dbg !17
  %1614 = insertelement <4 x float> %1613, float %1119, i64 3, !dbg !17
  %normalized55 = fmul <4 x float> %1614, %row.reciprocal, !dbg !17
  %1615 = fptrunc <4 x float> %normalized55 to <4 x half>, !dbg !18
  %1616 = extractelement <4 x half> %1615, i64 3, !dbg !18
  %1617 = extractelement <4 x half> %1615, i64 2, !dbg !18
  %1618 = extractelement <4 x half> %1615, i64 1, !dbg !18
  %1619 = extractelement <4 x half> %1615, i64 0, !dbg !18
  %1620 = insertelement <4 x float> poison, float %1083, i64 0, !dbg !17
  %1621 = insertelement <4 x float> %1620, float %1089, i64 1, !dbg !17
  %1622 = insertelement <4 x float> %1621, float %1084, i64 2, !dbg !17
  %1623 = insertelement <4 x float> %1622, float %1090, i64 3, !dbg !17
  %normalized57 = fmul <4 x float> %1623, %row.reciprocal38, !dbg !17
  %1624 = fptrunc <4 x float> %normalized57 to <4 x half>, !dbg !18
  %1625 = extractelement <4 x half> %1624, i64 3, !dbg !18
  %1626 = extractelement <4 x half> %1624, i64 2, !dbg !18
  %1627 = extractelement <4 x half> %1624, i64 1, !dbg !18
  %1628 = extractelement <4 x half> %1624, i64 0, !dbg !18
  %1629 = insertelement <4 x float> poison, float %1055, i64 0, !dbg !17
  %1630 = insertelement <4 x float> %1629, float %1061, i64 1, !dbg !17
  %1631 = insertelement <4 x float> %1630, float %1056, i64 2, !dbg !17
  %1632 = insertelement <4 x float> %1631, float %1062, i64 3, !dbg !17
  %normalized58 = fmul <4 x float> %1632, %row.reciprocal38, !dbg !17
  %1633 = fptrunc <4 x float> %normalized58 to <4 x half>, !dbg !18
  %1634 = extractelement <4 x half> %1633, i64 3, !dbg !18
  %1635 = extractelement <4 x half> %1633, i64 2, !dbg !18
  %1636 = extractelement <4 x half> %1633, i64 1, !dbg !18
  %1637 = extractelement <4 x half> %1633, i64 0, !dbg !18
  %address.local1139 = or disjoint i64 %address.local971, 96, !dbg !19
  %address.local1140 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1139, !dbg !19
  store half %1637, ptr addrspace(1) %address.local1140, align 2, !dbg !19
  %address.local1142 = or disjoint i64 %address.local971, 97, !dbg !19
  %address.local1143 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1142, !dbg !19
  store half %1636, ptr addrspace(1) %address.local1143, align 2, !dbg !19
  %address.local1145 = or disjoint i64 %address.local971, 98, !dbg !19
  %address.local1146 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1145, !dbg !19
  store half %1635, ptr addrspace(1) %address.local1146, align 2, !dbg !19
  %address.local1148 = or disjoint i64 %address.local971, 99, !dbg !19
  %address.local1149 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1148, !dbg !19
  store half %1634, ptr addrspace(1) %address.local1149, align 2, !dbg !19
  %address.local1151 = or disjoint i64 %address.local971, 112, !dbg !19
  %address.local1152 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1151, !dbg !19
  store half %1628, ptr addrspace(1) %address.local1152, align 2, !dbg !19
  %address.local1154 = or disjoint i64 %address.local971, 113, !dbg !19
  %address.local1155 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1154, !dbg !19
  store half %1627, ptr addrspace(1) %address.local1155, align 2, !dbg !19
  %address.local1157 = or disjoint i64 %address.local971, 114, !dbg !19
  %address.local1158 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1157, !dbg !19
  store half %1626, ptr addrspace(1) %address.local1158, align 2, !dbg !19
  %address.local1160 = or i64 %1528, 115, !dbg !19
  %address.local1161 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1160, !dbg !19
  store half %1625, ptr addrspace(1) %address.local1161, align 2, !dbg !19
  %address.local1166 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1139, !dbg !19
  store half %1619, ptr addrspace(1) %address.local1166, align 2, !dbg !19
  %address.local1168 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1142, !dbg !19
  store half %1618, ptr addrspace(1) %address.local1168, align 2, !dbg !19
  %address.local1170 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1145, !dbg !19
  store half %1617, ptr addrspace(1) %address.local1170, align 2, !dbg !19
  %address.local1172 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1148, !dbg !19
  store half %1616, ptr addrspace(1) %address.local1172, align 2, !dbg !19
  %address.local1174 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1151, !dbg !19
  store half %1610, ptr addrspace(1) %address.local1174, align 2, !dbg !19
  %address.local1176 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1154, !dbg !19
  store half %1609, ptr addrspace(1) %address.local1176, align 2, !dbg !19
  %address.local1178 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1157, !dbg !19
  store half %1608, ptr addrspace(1) %address.local1178, align 2, !dbg !19
  %address.local1180 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1160, !dbg !19
  store half %1607, ptr addrspace(1) %address.local1180, align 2, !dbg !19
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
