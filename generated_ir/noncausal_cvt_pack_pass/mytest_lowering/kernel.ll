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
  %104 = phi i64 [ %877, %k0.join ], [ 0, %4 ], !dbg !8
  %105 = phi [2 x <8 x float>] [ %951, %k0.join ], [ zeroinitializer, %4 ], !dbg !8
  %106 = phi [2 x <1 x float>] [ %728, %k0.join ], [ zeroinitializer, %4 ], !dbg !8
  %107 = phi [2 x <1 x float>] [ %lazy.base.join, %k0.join ], [ zeroinitializer, %4 ], !dbg !8
  %108 = phi [2 x <8 x float>] [ %955, %k0.join ], [ zeroinitializer, %4 ], !dbg !8
  %109 = phi [2 x <8 x float>] [ %959, %k0.join ], [ zeroinitializer, %4 ], !dbg !8
  %110 = phi [2 x <8 x float>] [ %963, %k0.join ], [ zeroinitializer, %4 ], !dbg !8
  %111 = icmp samesign ult i64 %104, 32, !dbg !8
  br i1 %111, label %112, label %964, !dbg !8

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
  %lazy.base = extractvalue [2 x <1 x float>] %107, 0, !dbg !15
  %lazy.seed = shufflevector <1 x float> %lazy.base, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !15
  %lazy.base1938 = extractvalue [2 x <1 x float>] %107, 1, !dbg !15
  %lazy.seed1939 = shufflevector <1 x float> %lazy.base1938, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !15
  %141 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %140, <4 x float> %lazy.seed) #9, !dbg !15
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
  %164 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %153, <4 x float> %lazy.seed) #9, !dbg !15
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
  %170 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %157, <4 x float> %lazy.seed) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %171 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %159, <4 x float> %170) #9, !dbg !15
  %172 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %154, <4 x float> %lazy.seed) #9, !dbg !15
  %173 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %156, <4 x float> %172) #9, !dbg !15
  %174 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %158, <4 x float> %lazy.seed) #9, !dbg !15
  %175 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %160, <4 x float> %174) #9, !dbg !15
  %176 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %162, <4 x float> %lazy.seed) #9, !dbg !15
  %177 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %165, <4 x float> %176) #9, !dbg !15
  %178 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %168, <4 x float> %lazy.seed) #9, !dbg !15
  %179 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %169, <4 x float> %178) #9, !dbg !15
  %180 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %163, <4 x float> %lazy.seed) #9, !dbg !15
  %181 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %166, <4 x float> %180) #9, !dbg !15
  %182 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %140, <4 x float> %lazy.seed1939) #9, !dbg !15
  %183 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %142, <4 x float> %182) #9, !dbg !15
  %184 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %153, <4 x float> %lazy.seed1939) #9, !dbg !15
  %185 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %155, <4 x float> %184) #9, !dbg !15
  %186 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %157, <4 x float> %lazy.seed1939) #9, !dbg !15
  %187 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %159, <4 x float> %186) #9, !dbg !15
  %188 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %154, <4 x float> %lazy.seed1939) #9, !dbg !15
  %189 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %156, <4 x float> %188) #9, !dbg !15
  %190 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %158, <4 x float> %lazy.seed1939) #9, !dbg !15
  %191 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %160, <4 x float> %190) #9, !dbg !15
  %192 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %162, <4 x float> %lazy.seed1939) #9, !dbg !15
  %193 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %165, <4 x float> %192) #9, !dbg !15
  %194 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %168, <4 x float> %lazy.seed1939) #9, !dbg !15
  %195 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %169, <4 x float> %194) #9, !dbg !15
  %196 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %163, <4 x float> %lazy.seed1939) #9, !dbg !15
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
  %lazy.peak = call float @llvm.maxnum.f32(float %430, float %440), !dbg !24
  %lazy.over = fcmp ogt float %lazy.peak, 0x4020000000000000, !dbg !24
  %lazy.mask = call i64 @llvm.amdgcn.ballot.i64(i1 %lazy.over), !dbg !24
  %lazy.any = icmp ne i64 %lazy.mask, 0, !dbg !24
  %lazy.first = icmp eq i64 %104, 0, !dbg !24
  %lazy.take = or i1 %lazy.first, %lazy.any, !dbg !24
  br i1 %lazy.take, label %lazy.slow, label %lazy.join, !dbg !24

lazy.slow:                                        ; preds = %112
  %lazy.positive = call float @llvm.maxnum.f32(float %430, float 0x0000000000000000), !dbg !24
  %lazy.delta = select i1 %lazy.first, float %430, float %lazy.positive, !dbg !24
  %lazy.negated = fsub float 0x0000000000000000, %lazy.delta, !dbg !24
  %lazy.exp = call float @llvm.amdgcn.exp2.f32(float %lazy.negated), !dbg !24
  %lazy.alpha = select i1 %lazy.first, float 0x3FF0000000000000, float %lazy.exp, !dbg !24
  %lazy.positive1940 = call float @llvm.maxnum.f32(float %440, float 0x0000000000000000), !dbg !24
  %lazy.delta1941 = select i1 %lazy.first, float %440, float %lazy.positive1940, !dbg !24
  %lazy.negated1942 = fsub float 0x0000000000000000, %lazy.delta1941, !dbg !24
  %lazy.exp1943 = call float @llvm.amdgcn.exp2.f32(float %lazy.negated1942), !dbg !24
  %lazy.alpha1944 = select i1 %lazy.first, float 0x3FF0000000000000, float %lazy.exp1943, !dbg !24
  %lazy.delta.v.lane = insertelement <1 x float> poison, float %lazy.delta, i64 0, !dbg !24
  %lazy.delta.v = shufflevector <1 x float> %lazy.delta.v.lane, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !24
  %lazy.shifted = fsub <4 x float> %318, %lazy.delta.v, !dbg !24
  %lazy.delta.v.lane1945 = insertelement <1 x float> poison, float %lazy.delta, i64 0, !dbg !24
  %lazy.delta.v1946 = shufflevector <1 x float> %lazy.delta.v.lane1945, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !24
  %lazy.shifted1947 = fsub <4 x float> %325, %lazy.delta.v1946, !dbg !24
  %lazy.delta.v.lane1948 = insertelement <1 x float> poison, float %lazy.delta, i64 0, !dbg !24
  %lazy.delta.v1949 = shufflevector <1 x float> %lazy.delta.v.lane1948, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !24
  %lazy.shifted1950 = fsub <4 x float> %335, %lazy.delta.v1949, !dbg !24
  %lazy.delta.v.lane1951 = insertelement <1 x float> poison, float %lazy.delta, i64 0, !dbg !24
  %lazy.delta.v1952 = shufflevector <1 x float> %lazy.delta.v.lane1951, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !24
  %lazy.shifted1953 = fsub <4 x float> %345, %lazy.delta.v1952, !dbg !24
  %lazy.delta.v.lane1954 = insertelement <1 x float> poison, float %lazy.delta, i64 0, !dbg !24
  %lazy.delta.v1955 = shufflevector <1 x float> %lazy.delta.v.lane1954, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !24
  %lazy.shifted1956 = fsub <4 x float> %351, %lazy.delta.v1955, !dbg !24
  %lazy.delta.v.lane1957 = insertelement <1 x float> poison, float %lazy.delta, i64 0, !dbg !24
  %lazy.delta.v1958 = shufflevector <1 x float> %lazy.delta.v.lane1957, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !24
  %lazy.shifted1959 = fsub <4 x float> %357, %lazy.delta.v1958, !dbg !24
  %lazy.delta.v.lane1960 = insertelement <1 x float> poison, float %lazy.delta, i64 0, !dbg !24
  %lazy.delta.v1961 = shufflevector <1 x float> %lazy.delta.v.lane1960, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !24
  %lazy.shifted1962 = fsub <4 x float> %363, %lazy.delta.v1961, !dbg !24
  %lazy.delta.v.lane1963 = insertelement <1 x float> poison, float %lazy.delta, i64 0, !dbg !24
  %lazy.delta.v1964 = shufflevector <1 x float> %lazy.delta.v.lane1963, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !24
  %lazy.shifted1965 = fsub <4 x float> %369, %lazy.delta.v1964, !dbg !24
  %lazy.delta.v.lane1966 = insertelement <1 x float> poison, float %lazy.delta1941, i64 0, !dbg !24
  %lazy.delta.v1967 = shufflevector <1 x float> %lazy.delta.v.lane1966, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !24
  %lazy.shifted1968 = fsub <4 x float> %376, %lazy.delta.v1967, !dbg !24
  %lazy.delta.v.lane1969 = insertelement <1 x float> poison, float %lazy.delta1941, i64 0, !dbg !24
  %lazy.delta.v1970 = shufflevector <1 x float> %lazy.delta.v.lane1969, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !24
  %lazy.shifted1971 = fsub <4 x float> %382, %lazy.delta.v1970, !dbg !24
  %lazy.delta.v.lane1972 = insertelement <1 x float> poison, float %lazy.delta1941, i64 0, !dbg !24
  %lazy.delta.v1973 = shufflevector <1 x float> %lazy.delta.v.lane1972, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !24
  %lazy.shifted1974 = fsub <4 x float> %388, %lazy.delta.v1973, !dbg !24
  %lazy.delta.v.lane1975 = insertelement <1 x float> poison, float %lazy.delta1941, i64 0, !dbg !24
  %lazy.delta.v1976 = shufflevector <1 x float> %lazy.delta.v.lane1975, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !24
  %lazy.shifted1977 = fsub <4 x float> %394, %lazy.delta.v1976, !dbg !24
  %lazy.delta.v.lane1978 = insertelement <1 x float> poison, float %lazy.delta1941, i64 0, !dbg !24
  %lazy.delta.v1979 = shufflevector <1 x float> %lazy.delta.v.lane1978, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !24
  %lazy.shifted1980 = fsub <4 x float> %400, %lazy.delta.v1979, !dbg !24
  %lazy.delta.v.lane1981 = insertelement <1 x float> poison, float %lazy.delta1941, i64 0, !dbg !24
  %lazy.delta.v1982 = shufflevector <1 x float> %lazy.delta.v.lane1981, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !24
  %lazy.shifted1983 = fsub <4 x float> %406, %lazy.delta.v1982, !dbg !24
  %lazy.delta.v.lane1984 = insertelement <1 x float> poison, float %lazy.delta1941, i64 0, !dbg !24
  %lazy.delta.v1985 = shufflevector <1 x float> %lazy.delta.v.lane1984, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !24
  %lazy.shifted1986 = fsub <4 x float> %412, %lazy.delta.v1985, !dbg !24
  %lazy.delta.v.lane1987 = insertelement <1 x float> poison, float %lazy.delta1941, i64 0, !dbg !24
  %lazy.delta.v1988 = shufflevector <1 x float> %lazy.delta.v.lane1987, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !24
  %lazy.shifted1989 = fsub <4 x float> %418, %lazy.delta.v1988, !dbg !24
  %lazy.state = extractvalue [2 x <1 x float>] %106, 0, !dbg !24
  %lazy.alpha.v.lane = insertelement <1 x float> poison, float %lazy.alpha, i64 0, !dbg !24
  %lazy.scaled = fmul <1 x float> %lazy.state, %lazy.alpha.v.lane, !dbg !24
  %lazy.state.next = insertvalue [2 x <1 x float>] poison, <1 x float> %lazy.scaled, 0, !dbg !24
  %lazy.state1990 = extractvalue [2 x <1 x float>] %106, 1, !dbg !24
  %lazy.alpha.v.lane1991 = insertelement <1 x float> poison, float %lazy.alpha1944, i64 0, !dbg !24
  %lazy.scaled1992 = fmul <1 x float> %lazy.state1990, %lazy.alpha.v.lane1991, !dbg !24
  %lazy.state.next1993 = insertvalue [2 x <1 x float>] %lazy.state.next, <1 x float> %lazy.scaled1992, 1, !dbg !24
  %lazy.state1994 = extractvalue [2 x <8 x float>] %110, 0, !dbg !24
  %lazy.alpha.v.lane1995 = insertelement <1 x float> poison, float %lazy.alpha, i64 0, !dbg !24
  %lazy.alpha.v = shufflevector <1 x float> %lazy.alpha.v.lane1995, <1 x float> poison, <8 x i32> zeroinitializer, !dbg !24
  %lazy.scaled1996 = fmul <8 x float> %lazy.state1994, %lazy.alpha.v, !dbg !24
  %lazy.state.next1997 = insertvalue [2 x <8 x float>] poison, <8 x float> %lazy.scaled1996, 0, !dbg !24
  %lazy.state1998 = extractvalue [2 x <8 x float>] %110, 1, !dbg !24
  %lazy.alpha.v.lane1999 = insertelement <1 x float> poison, float %lazy.alpha1944, i64 0, !dbg !24
  %lazy.alpha.v2000 = shufflevector <1 x float> %lazy.alpha.v.lane1999, <1 x float> poison, <8 x i32> zeroinitializer, !dbg !24
  %lazy.scaled2001 = fmul <8 x float> %lazy.state1998, %lazy.alpha.v2000, !dbg !24
  %lazy.state.next2002 = insertvalue [2 x <8 x float>] %lazy.state.next1997, <8 x float> %lazy.scaled2001, 1, !dbg !24
  %lazy.state2003 = extractvalue [2 x <8 x float>] %109, 0, !dbg !24
  %lazy.alpha.v.lane2004 = insertelement <1 x float> poison, float %lazy.alpha, i64 0, !dbg !24
  %lazy.alpha.v2005 = shufflevector <1 x float> %lazy.alpha.v.lane2004, <1 x float> poison, <8 x i32> zeroinitializer, !dbg !24
  %lazy.scaled2006 = fmul <8 x float> %lazy.state2003, %lazy.alpha.v2005, !dbg !24
  %lazy.state.next2007 = insertvalue [2 x <8 x float>] poison, <8 x float> %lazy.scaled2006, 0, !dbg !24
  %lazy.state2008 = extractvalue [2 x <8 x float>] %109, 1, !dbg !24
  %lazy.alpha.v.lane2009 = insertelement <1 x float> poison, float %lazy.alpha1944, i64 0, !dbg !24
  %lazy.alpha.v2010 = shufflevector <1 x float> %lazy.alpha.v.lane2009, <1 x float> poison, <8 x i32> zeroinitializer, !dbg !24
  %lazy.scaled2011 = fmul <8 x float> %lazy.state2008, %lazy.alpha.v2010, !dbg !24
  %lazy.state.next2012 = insertvalue [2 x <8 x float>] %lazy.state.next2007, <8 x float> %lazy.scaled2011, 1, !dbg !24
  %lazy.state2013 = extractvalue [2 x <8 x float>] %108, 0, !dbg !24
  %lazy.alpha.v.lane2014 = insertelement <1 x float> poison, float %lazy.alpha, i64 0, !dbg !24
  %lazy.alpha.v2015 = shufflevector <1 x float> %lazy.alpha.v.lane2014, <1 x float> poison, <8 x i32> zeroinitializer, !dbg !24
  %lazy.scaled2016 = fmul <8 x float> %lazy.state2013, %lazy.alpha.v2015, !dbg !24
  %lazy.state.next2017 = insertvalue [2 x <8 x float>] poison, <8 x float> %lazy.scaled2016, 0, !dbg !24
  %lazy.state2018 = extractvalue [2 x <8 x float>] %108, 1, !dbg !24
  %lazy.alpha.v.lane2019 = insertelement <1 x float> poison, float %lazy.alpha1944, i64 0, !dbg !24
  %lazy.alpha.v2020 = shufflevector <1 x float> %lazy.alpha.v.lane2019, <1 x float> poison, <8 x i32> zeroinitializer, !dbg !24
  %lazy.scaled2021 = fmul <8 x float> %lazy.state2018, %lazy.alpha.v2020, !dbg !24
  %lazy.state.next2022 = insertvalue [2 x <8 x float>] %lazy.state.next2017, <8 x float> %lazy.scaled2021, 1, !dbg !24
  %lazy.state2023 = extractvalue [2 x <8 x float>] %105, 0, !dbg !24
  %lazy.alpha.v.lane2024 = insertelement <1 x float> poison, float %lazy.alpha, i64 0, !dbg !24
  %lazy.alpha.v2025 = shufflevector <1 x float> %lazy.alpha.v.lane2024, <1 x float> poison, <8 x i32> zeroinitializer, !dbg !24
  %lazy.scaled2026 = fmul <8 x float> %lazy.state2023, %lazy.alpha.v2025, !dbg !24
  %lazy.state.next2027 = insertvalue [2 x <8 x float>] poison, <8 x float> %lazy.scaled2026, 0, !dbg !24
  %lazy.state2028 = extractvalue [2 x <8 x float>] %105, 1, !dbg !24
  %lazy.alpha.v.lane2029 = insertelement <1 x float> poison, float %lazy.alpha1944, i64 0, !dbg !24
  %lazy.alpha.v2030 = shufflevector <1 x float> %lazy.alpha.v.lane2029, <1 x float> poison, <8 x i32> zeroinitializer, !dbg !24
  %lazy.scaled2031 = fmul <8 x float> %lazy.state2028, %lazy.alpha.v2030, !dbg !24
  %lazy.state.next2032 = insertvalue [2 x <8 x float>] %lazy.state.next2027, <8 x float> %lazy.scaled2031, 1, !dbg !24
  %lazy.base.old = extractvalue [2 x <1 x float>] %107, 0, !dbg !24
  %lazy.delta.b.lane = insertelement <1 x float> poison, float %lazy.delta, i64 0, !dbg !24
  %lazy.base.next = fsub <1 x float> %lazy.base.old, %lazy.delta.b.lane, !dbg !24
  %lazy.base.state = insertvalue [2 x <1 x float>] poison, <1 x float> %lazy.base.next, 0, !dbg !24
  %lazy.base.old2033 = extractvalue [2 x <1 x float>] %107, 1, !dbg !24
  %lazy.delta.b.lane2034 = insertelement <1 x float> poison, float %lazy.delta1941, i64 0, !dbg !24
  %lazy.base.next2035 = fsub <1 x float> %lazy.base.old2033, %lazy.delta.b.lane2034, !dbg !24
  %lazy.base.state2036 = insertvalue [2 x <1 x float>] %lazy.base.state, <1 x float> %lazy.base.next2035, 1, !dbg !24
  br label %lazy.join, !dbg !24

lazy.join:                                        ; preds = %lazy.slow, %112
  %lazy.score = phi <4 x float> [ %318, %112 ], [ %lazy.shifted, %lazy.slow ], !dbg !24
  %lazy.score2037 = phi <4 x float> [ %325, %112 ], [ %lazy.shifted1947, %lazy.slow ], !dbg !24
  %lazy.score2038 = phi <4 x float> [ %335, %112 ], [ %lazy.shifted1950, %lazy.slow ], !dbg !24
  %lazy.score2039 = phi <4 x float> [ %345, %112 ], [ %lazy.shifted1953, %lazy.slow ], !dbg !24
  %lazy.score2040 = phi <4 x float> [ %351, %112 ], [ %lazy.shifted1956, %lazy.slow ], !dbg !24
  %lazy.score2041 = phi <4 x float> [ %357, %112 ], [ %lazy.shifted1959, %lazy.slow ], !dbg !24
  %lazy.score2042 = phi <4 x float> [ %363, %112 ], [ %lazy.shifted1962, %lazy.slow ], !dbg !24
  %lazy.score2043 = phi <4 x float> [ %369, %112 ], [ %lazy.shifted1965, %lazy.slow ], !dbg !24
  %lazy.score2044 = phi <4 x float> [ %376, %112 ], [ %lazy.shifted1968, %lazy.slow ], !dbg !24
  %lazy.score2045 = phi <4 x float> [ %382, %112 ], [ %lazy.shifted1971, %lazy.slow ], !dbg !24
  %lazy.score2046 = phi <4 x float> [ %388, %112 ], [ %lazy.shifted1974, %lazy.slow ], !dbg !24
  %lazy.score2047 = phi <4 x float> [ %394, %112 ], [ %lazy.shifted1977, %lazy.slow ], !dbg !24
  %lazy.score2048 = phi <4 x float> [ %400, %112 ], [ %lazy.shifted1980, %lazy.slow ], !dbg !24
  %lazy.score2049 = phi <4 x float> [ %406, %112 ], [ %lazy.shifted1983, %lazy.slow ], !dbg !24
  %lazy.score2050 = phi <4 x float> [ %412, %112 ], [ %lazy.shifted1986, %lazy.slow ], !dbg !24
  %lazy.score2051 = phi <4 x float> [ %418, %112 ], [ %lazy.shifted1989, %lazy.slow ], !dbg !24
  %lazy.state2052 = phi [2 x <1 x float>] [ %106, %112 ], [ %lazy.state.next1993, %lazy.slow ], !dbg !24
  %lazy.state2053 = phi [2 x <8 x float>] [ %110, %112 ], [ %lazy.state.next2002, %lazy.slow ], !dbg !24
  %lazy.state2054 = phi [2 x <8 x float>] [ %109, %112 ], [ %lazy.state.next2012, %lazy.slow ], !dbg !24
  %lazy.state2055 = phi [2 x <8 x float>] [ %108, %112 ], [ %lazy.state.next2022, %lazy.slow ], !dbg !24
  %lazy.state2056 = phi [2 x <8 x float>] [ %105, %112 ], [ %lazy.state.next2032, %lazy.slow ], !dbg !24
  %lazy.base.join = phi [2 x <1 x float>] [ %107, %112 ], [ %lazy.base.state2036, %lazy.slow ], !dbg !24
  %441 = extractvalue [2 x <1 x float>] %lazy.state2052, 0, !dbg !24
  %442 = extractelement <1 x float> %441, i64 0, !dbg !24
  %443 = extractvalue [2 x <1 x float>] %lazy.state2052, 1, !dbg !24
  %444 = extractelement <1 x float> %443, i64 0, !dbg !24
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
  %445 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1408, <4 x i32> %dma.buffer.rsrc1327) #10, !dbg !16
  %vdb.dma.base1420 = add i32 %43, 8192, !dbg !14
  %446 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1420, i32 %vdb.address1415, <4 x i32> %dma.buffer.rsrc1327) #10, !dbg !16
  %address.local913 = shl nuw nsw i32 %address.thread498, 3, !dbg !14
  %address.local914 = and i32 %address.local913, 120, !dbg !14
  %address.local916 = and i32 %address.local559, 1536, !dbg !14
  %address.local917 = or disjoint i32 %address.local914, %address.local916, !dbg !14
  %address.local918 = zext nneg i32 %address.local917 to i64, !dbg !14
  %address.local919 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local918, !dbg !14
  %447 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local919, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %448 = bitcast <4 x i32> %447 to <8 x half>, !dbg !14
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
  %449 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local929, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %450 = bitcast <4 x i32> %449 to <8 x half>, !dbg !14
  %address.local933 = getelementptr i8, ptr addrspace(3) %address.local928, i64 512, !dbg !14
  %451 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local933, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %452 = bitcast <4 x i32> %451 to <8 x half>, !dbg !14
  %address.local937 = getelementptr i8, ptr addrspace(3) %address.local928, i64 768, !dbg !14
  %453 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local937, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %454 = bitcast <4 x i32> %453 to <8 x half>, !dbg !14
  %address.local939 = getelementptr i8, ptr addrspace(3) %address.local919, i64 4096, !dbg !14
  %455 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local939, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %456 = bitcast <4 x i32> %455 to <8 x half>, !dbg !14
  %address.local943 = getelementptr i8, ptr addrspace(3) %address.local928, i64 4352, !dbg !14
  %457 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local943, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %458 = bitcast <4 x i32> %457 to <8 x half>, !dbg !14
  %address.local947 = getelementptr i8, ptr addrspace(3) %address.local928, i64 4608, !dbg !14
  %459 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local947, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %460 = bitcast <4 x i32> %459 to <8 x half>, !dbg !14
  %address.local951 = getelementptr i8, ptr addrspace(3) %address.local928, i64 4864, !dbg !14
  %461 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local951, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %462 = shufflevector <8 x half> %448, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %463 = extractvalue [2 x <8 x float>] %lazy.state2056, 0, !dbg !28
  %464 = shufflevector <8 x float> %463, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %465 = extractelement <4 x float> %lazy.score, i64 0, !dbg !25
  %466 = call float @llvm.amdgcn.exp2.f32(float %465), !dbg !25
  %467 = insertelement <4 x float> poison, float %466, i64 0, !dbg !27
  %468 = extractelement <4 x float> %lazy.score, i64 1, !dbg !25
  %469 = call float @llvm.amdgcn.exp2.f32(float %468), !dbg !25
  %470 = insertelement <4 x float> %467, float %469, i64 1, !dbg !27
  %471 = extractelement <4 x float> %lazy.score, i64 2, !dbg !25
  %472 = call float @llvm.amdgcn.exp2.f32(float %471), !dbg !25
  %473 = insertelement <4 x float> %470, float %472, i64 2, !dbg !27
  %474 = extractelement <4 x float> %lazy.score, i64 3, !dbg !25
  %475 = call float @llvm.amdgcn.exp2.f32(float %474), !dbg !25
  %476 = insertelement <4 x float> %473, float %475, i64 3, !dbg !27
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %476) #9, !dbg !27, !frisk.register_lifetimes !1
  %.cvtpack.f0 = extractelement <4 x float> %math.result, i32 0, !dbg !27
  %.cvtpack.f1 = extractelement <4 x float> %math.result, i32 1, !dbg !27
  %.cvtpack.f2 = extractelement <4 x float> %math.result, i32 2, !dbg !27
  %.cvtpack.f3 = extractelement <4 x float> %math.result, i32 3, !dbg !27
  %.cvtpack.p0 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f0, float %.cvtpack.f1), !dbg !27
  %.cvtpack.p1 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f2, float %.cvtpack.f3), !dbg !27
  %.cvtpack.v0 = insertelement <2 x i32> poison, i32 %.cvtpack.p0, i32 0, !dbg !27
  %.cvtpack.v1 = insertelement <2 x i32> %.cvtpack.v0, i32 %.cvtpack.p1, i32 1, !dbg !27
  %477 = bitcast <2 x i32> %.cvtpack.v1 to <4 x half>, !dbg !27
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %477) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %478 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %462, <4 x float> %464) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %479 = shufflevector <8 x half> %448, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %480 = shufflevector <8 x float> %463, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %481 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %479, <4 x float> %480) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %482 = extractvalue [2 x <8 x float>] %lazy.state2056, 1, !dbg !28
  %483 = shufflevector <8 x float> %482, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %484 = extractelement <4 x float> %lazy.score2044, i64 0, !dbg !25
  %485 = call float @llvm.amdgcn.exp2.f32(float %484), !dbg !25
  %486 = insertelement <4 x float> poison, float %485, i64 0, !dbg !27
  %487 = extractelement <4 x float> %lazy.score2044, i64 1, !dbg !25
  %488 = call float @llvm.amdgcn.exp2.f32(float %487), !dbg !25
  %489 = insertelement <4 x float> %486, float %488, i64 1, !dbg !27
  %490 = extractelement <4 x float> %lazy.score2044, i64 2, !dbg !25
  %491 = call float @llvm.amdgcn.exp2.f32(float %490), !dbg !25
  %492 = insertelement <4 x float> %489, float %491, i64 2, !dbg !27
  %493 = extractelement <4 x float> %lazy.score2044, i64 3, !dbg !25
  %494 = call float @llvm.amdgcn.exp2.f32(float %493), !dbg !25
  %495 = insertelement <4 x float> %492, float %494, i64 3, !dbg !27
  %math.result388 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %495) #9, !dbg !27, !frisk.register_lifetimes !1
  %.cvtpack.f02057 = extractelement <4 x float> %math.result388, i32 0, !dbg !27
  %.cvtpack.f12058 = extractelement <4 x float> %math.result388, i32 1, !dbg !27
  %.cvtpack.f22059 = extractelement <4 x float> %math.result388, i32 2, !dbg !27
  %.cvtpack.f32060 = extractelement <4 x float> %math.result388, i32 3, !dbg !27
  %.cvtpack.p02061 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02057, float %.cvtpack.f12058), !dbg !27
  %.cvtpack.p12062 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22059, float %.cvtpack.f32060), !dbg !27
  %.cvtpack.v02063 = insertelement <2 x i32> poison, i32 %.cvtpack.p02061, i32 0, !dbg !27
  %.cvtpack.v12064 = insertelement <2 x i32> %.cvtpack.v02063, i32 %.cvtpack.p12062, i32 1, !dbg !27
  %496 = bitcast <2 x i32> %.cvtpack.v12064 to <4 x half>, !dbg !27
  %math.packed389 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %496) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %497 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %462, <4 x float> %483) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %498 = shufflevector <8 x float> %482, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %499 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %479, <4 x float> %498) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %500 = shufflevector <8 x half> %450, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %501 = extractelement <4 x float> %lazy.score2037, i64 0, !dbg !25
  %502 = call float @llvm.amdgcn.exp2.f32(float %501), !dbg !25
  %local.reduce89 = fadd float %466, %502, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor = call float asm sideeffect "", "=v,0"(float %local.reduce89) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %503 = insertelement <4 x float> poison, float %502, i64 0, !dbg !27
  %504 = extractelement <4 x float> %lazy.score2037, i64 1, !dbg !25
  %505 = call float @llvm.amdgcn.exp2.f32(float %504), !dbg !25
  %local.reduce90 = fadd float %469, %505, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1860 = call float asm sideeffect "", "=v,0"(float %local.reduce90) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %506 = insertelement <4 x float> %503, float %505, i64 1, !dbg !27
  %507 = extractelement <4 x float> %lazy.score2037, i64 2, !dbg !25
  %508 = call float @llvm.amdgcn.exp2.f32(float %507), !dbg !25
  %local.reduce91 = fadd float %472, %508, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1861 = call float asm sideeffect "", "=v,0"(float %local.reduce91) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %509 = insertelement <4 x float> %506, float %508, i64 2, !dbg !27
  %510 = extractelement <4 x float> %lazy.score2037, i64 3, !dbg !25
  %511 = call float @llvm.amdgcn.exp2.f32(float %510), !dbg !25
  %local.reduce92 = fadd float %475, %511, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1862 = call float asm sideeffect "", "=v,0"(float %local.reduce92) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %512 = insertelement <4 x float> %509, float %511, i64 3, !dbg !27
  %math.result374 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %512) #9, !dbg !27, !frisk.register_lifetimes !1
  %.cvtpack.f02065 = extractelement <4 x float> %math.result374, i32 0, !dbg !27
  %.cvtpack.f12066 = extractelement <4 x float> %math.result374, i32 1, !dbg !27
  %.cvtpack.f22067 = extractelement <4 x float> %math.result374, i32 2, !dbg !27
  %.cvtpack.f32068 = extractelement <4 x float> %math.result374, i32 3, !dbg !27
  %.cvtpack.p02069 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02065, float %.cvtpack.f12066), !dbg !27
  %.cvtpack.p12070 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22067, float %.cvtpack.f32068), !dbg !27
  %.cvtpack.v02071 = insertelement <2 x i32> poison, i32 %.cvtpack.p02069, i32 0, !dbg !27
  %.cvtpack.v12072 = insertelement <2 x i32> %.cvtpack.v02071, i32 %.cvtpack.p12070, i32 1, !dbg !27
  %513 = bitcast <2 x i32> %.cvtpack.v12072 to <4 x half>, !dbg !27
  %math.packed375 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %513) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %514 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %500, <4 x float> %478) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %515 = shufflevector <8 x half> %450, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %516 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %515, <4 x float> %481) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %517 = extractelement <4 x float> %lazy.score2045, i64 0, !dbg !25
  %518 = call float @llvm.amdgcn.exp2.f32(float %517), !dbg !25
  %local.reduce = fadd float %485, %518, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1863 = call float asm sideeffect "", "=v,0"(float %local.reduce) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %519 = insertelement <4 x float> poison, float %518, i64 0, !dbg !27
  %520 = extractelement <4 x float> %lazy.score2045, i64 1, !dbg !25
  %521 = call float @llvm.amdgcn.exp2.f32(float %520), !dbg !25
  %local.reduce59 = fadd float %488, %521, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1864 = call float asm sideeffect "", "=v,0"(float %local.reduce59) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %522 = insertelement <4 x float> %519, float %521, i64 1, !dbg !27
  %523 = extractelement <4 x float> %lazy.score2045, i64 2, !dbg !25
  %524 = call float @llvm.amdgcn.exp2.f32(float %523), !dbg !25
  %local.reduce60 = fadd float %491, %524, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1865 = call float asm sideeffect "", "=v,0"(float %local.reduce60) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %525 = insertelement <4 x float> %522, float %524, i64 2, !dbg !27
  %526 = extractelement <4 x float> %lazy.score2045, i64 3, !dbg !25
  %527 = call float @llvm.amdgcn.exp2.f32(float %526), !dbg !25
  %local.reduce61 = fadd float %494, %527, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1866 = call float asm sideeffect "", "=v,0"(float %local.reduce61) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %528 = insertelement <4 x float> %525, float %527, i64 3, !dbg !27
  %math.result390 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %528) #9, !dbg !27, !frisk.register_lifetimes !1
  %.cvtpack.f02073 = extractelement <4 x float> %math.result390, i32 0, !dbg !27
  %.cvtpack.f12074 = extractelement <4 x float> %math.result390, i32 1, !dbg !27
  %.cvtpack.f22075 = extractelement <4 x float> %math.result390, i32 2, !dbg !27
  %.cvtpack.f32076 = extractelement <4 x float> %math.result390, i32 3, !dbg !27
  %.cvtpack.p02077 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02073, float %.cvtpack.f12074), !dbg !27
  %.cvtpack.p12078 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22075, float %.cvtpack.f32076), !dbg !27
  %.cvtpack.v02079 = insertelement <2 x i32> poison, i32 %.cvtpack.p02077, i32 0, !dbg !27
  %.cvtpack.v12080 = insertelement <2 x i32> %.cvtpack.v02079, i32 %.cvtpack.p12078, i32 1, !dbg !27
  %529 = bitcast <2 x i32> %.cvtpack.v12080 to <4 x half>, !dbg !27
  %math.packed391 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %529) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %530 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %500, <4 x float> %497) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %531 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %515, <4 x float> %499) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %532 = shufflevector <8 x half> %452, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %533 = extractelement <4 x float> %lazy.score2038, i64 0, !dbg !25
  %534 = call float @llvm.amdgcn.exp2.f32(float %533), !dbg !25
  %local.reduce93 = fadd float %k0.sum.anchor, %534, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1867 = call float asm sideeffect "", "=v,0"(float %local.reduce93) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %535 = insertelement <4 x float> poison, float %534, i64 0, !dbg !27
  %536 = extractelement <4 x float> %lazy.score2038, i64 1, !dbg !25
  %537 = call float @llvm.amdgcn.exp2.f32(float %536), !dbg !25
  %local.reduce94 = fadd float %k0.sum.anchor1860, %537, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1868 = call float asm sideeffect "", "=v,0"(float %local.reduce94) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %538 = insertelement <4 x float> %535, float %537, i64 1, !dbg !27
  %539 = extractelement <4 x float> %lazy.score2038, i64 2, !dbg !25
  %540 = call float @llvm.amdgcn.exp2.f32(float %539), !dbg !25
  %local.reduce95 = fadd float %k0.sum.anchor1861, %540, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1869 = call float asm sideeffect "", "=v,0"(float %local.reduce95) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %541 = insertelement <4 x float> %538, float %540, i64 2, !dbg !27
  %542 = extractelement <4 x float> %lazy.score2038, i64 3, !dbg !25
  %543 = call float @llvm.amdgcn.exp2.f32(float %542), !dbg !25
  %local.reduce96 = fadd float %k0.sum.anchor1862, %543, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1870 = call float asm sideeffect "", "=v,0"(float %local.reduce96) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %544 = insertelement <4 x float> %541, float %543, i64 3, !dbg !27
  %math.result376 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %544) #9, !dbg !27, !frisk.register_lifetimes !1
  %.cvtpack.f02081 = extractelement <4 x float> %math.result376, i32 0, !dbg !27
  %.cvtpack.f12082 = extractelement <4 x float> %math.result376, i32 1, !dbg !27
  %.cvtpack.f22083 = extractelement <4 x float> %math.result376, i32 2, !dbg !27
  %.cvtpack.f32084 = extractelement <4 x float> %math.result376, i32 3, !dbg !27
  %.cvtpack.p02085 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02081, float %.cvtpack.f12082), !dbg !27
  %.cvtpack.p12086 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22083, float %.cvtpack.f32084), !dbg !27
  %.cvtpack.v02087 = insertelement <2 x i32> poison, i32 %.cvtpack.p02085, i32 0, !dbg !27
  %.cvtpack.v12088 = insertelement <2 x i32> %.cvtpack.v02087, i32 %.cvtpack.p12086, i32 1, !dbg !27
  %545 = bitcast <2 x i32> %.cvtpack.v12088 to <4 x half>, !dbg !27
  %math.packed377 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %545) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %546 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %532, <4 x float> %514) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %547 = shufflevector <8 x half> %452, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %548 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %547, <4 x float> %516) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %549 = extractelement <4 x float> %lazy.score2046, i64 0, !dbg !25
  %550 = call float @llvm.amdgcn.exp2.f32(float %549), !dbg !25
  %local.reduce62 = fadd float %k0.sum.anchor1863, %550, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1871 = call float asm sideeffect "", "=v,0"(float %local.reduce62) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %551 = insertelement <4 x float> poison, float %550, i64 0, !dbg !27
  %552 = extractelement <4 x float> %lazy.score2046, i64 1, !dbg !25
  %553 = call float @llvm.amdgcn.exp2.f32(float %552), !dbg !25
  %local.reduce63 = fadd float %k0.sum.anchor1864, %553, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1872 = call float asm sideeffect "", "=v,0"(float %local.reduce63) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %554 = insertelement <4 x float> %551, float %553, i64 1, !dbg !27
  %555 = extractelement <4 x float> %lazy.score2046, i64 2, !dbg !25
  %556 = call float @llvm.amdgcn.exp2.f32(float %555), !dbg !25
  %local.reduce64 = fadd float %k0.sum.anchor1865, %556, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1873 = call float asm sideeffect "", "=v,0"(float %local.reduce64) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %557 = insertelement <4 x float> %554, float %556, i64 2, !dbg !27
  %558 = extractelement <4 x float> %lazy.score2046, i64 3, !dbg !25
  %559 = call float @llvm.amdgcn.exp2.f32(float %558), !dbg !25
  %local.reduce65 = fadd float %k0.sum.anchor1866, %559, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1874 = call float asm sideeffect "", "=v,0"(float %local.reduce65) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %560 = insertelement <4 x float> %557, float %559, i64 3, !dbg !27
  %math.result392 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %560) #9, !dbg !27, !frisk.register_lifetimes !1
  %.cvtpack.f02089 = extractelement <4 x float> %math.result392, i32 0, !dbg !27
  %.cvtpack.f12090 = extractelement <4 x float> %math.result392, i32 1, !dbg !27
  %.cvtpack.f22091 = extractelement <4 x float> %math.result392, i32 2, !dbg !27
  %.cvtpack.f32092 = extractelement <4 x float> %math.result392, i32 3, !dbg !27
  %.cvtpack.p02093 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02089, float %.cvtpack.f12090), !dbg !27
  %.cvtpack.p12094 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22091, float %.cvtpack.f32092), !dbg !27
  %.cvtpack.v02095 = insertelement <2 x i32> poison, i32 %.cvtpack.p02093, i32 0, !dbg !27
  %.cvtpack.v12096 = insertelement <2 x i32> %.cvtpack.v02095, i32 %.cvtpack.p12094, i32 1, !dbg !27
  %561 = bitcast <2 x i32> %.cvtpack.v12096 to <4 x half>, !dbg !27
  %math.packed393 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %561) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %562 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %532, <4 x float> %530) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %563 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %547, <4 x float> %531) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %564 = shufflevector <8 x half> %454, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %565 = extractelement <4 x float> %lazy.score2039, i64 0, !dbg !25
  %566 = call float @llvm.amdgcn.exp2.f32(float %565), !dbg !25
  %local.reduce97 = fadd float %k0.sum.anchor1867, %566, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1875 = call float asm sideeffect "", "=v,0"(float %local.reduce97) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %567 = insertelement <4 x float> poison, float %566, i64 0, !dbg !27
  %568 = extractelement <4 x float> %lazy.score2039, i64 1, !dbg !25
  %569 = call float @llvm.amdgcn.exp2.f32(float %568), !dbg !25
  %local.reduce98 = fadd float %k0.sum.anchor1868, %569, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1876 = call float asm sideeffect "", "=v,0"(float %local.reduce98) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %570 = insertelement <4 x float> %567, float %569, i64 1, !dbg !27
  %571 = extractelement <4 x float> %lazy.score2039, i64 2, !dbg !25
  %572 = call float @llvm.amdgcn.exp2.f32(float %571), !dbg !25
  %local.reduce99 = fadd float %k0.sum.anchor1869, %572, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1877 = call float asm sideeffect "", "=v,0"(float %local.reduce99) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %573 = insertelement <4 x float> %570, float %572, i64 2, !dbg !27
  %574 = extractelement <4 x float> %lazy.score2039, i64 3, !dbg !25
  %575 = call float @llvm.amdgcn.exp2.f32(float %574), !dbg !25
  %local.reduce100 = fadd float %k0.sum.anchor1870, %575, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1878 = call float asm sideeffect "", "=v,0"(float %local.reduce100) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %576 = insertelement <4 x float> %573, float %575, i64 3, !dbg !27
  %math.result378 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %576) #9, !dbg !27, !frisk.register_lifetimes !1
  %.cvtpack.f02097 = extractelement <4 x float> %math.result378, i32 0, !dbg !27
  %.cvtpack.f12098 = extractelement <4 x float> %math.result378, i32 1, !dbg !27
  %.cvtpack.f22099 = extractelement <4 x float> %math.result378, i32 2, !dbg !27
  %.cvtpack.f32100 = extractelement <4 x float> %math.result378, i32 3, !dbg !27
  %.cvtpack.p02101 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02097, float %.cvtpack.f12098), !dbg !27
  %.cvtpack.p12102 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22099, float %.cvtpack.f32100), !dbg !27
  %.cvtpack.v02103 = insertelement <2 x i32> poison, i32 %.cvtpack.p02101, i32 0, !dbg !27
  %.cvtpack.v12104 = insertelement <2 x i32> %.cvtpack.v02103, i32 %.cvtpack.p12102, i32 1, !dbg !27
  %577 = bitcast <2 x i32> %.cvtpack.v12104 to <4 x half>, !dbg !27
  %math.packed379 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %577) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %578 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %564, <4 x float> %546) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %579 = shufflevector <8 x half> %454, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %580 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %579, <4 x float> %548) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %581 = extractelement <4 x float> %lazy.score2047, i64 0, !dbg !25
  %582 = call float @llvm.amdgcn.exp2.f32(float %581), !dbg !25
  %local.reduce66 = fadd float %k0.sum.anchor1871, %582, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1879 = call float asm sideeffect "", "=v,0"(float %local.reduce66) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %583 = insertelement <4 x float> poison, float %582, i64 0, !dbg !27
  %584 = extractelement <4 x float> %lazy.score2047, i64 1, !dbg !25
  %585 = call float @llvm.amdgcn.exp2.f32(float %584), !dbg !25
  %local.reduce67 = fadd float %k0.sum.anchor1872, %585, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1880 = call float asm sideeffect "", "=v,0"(float %local.reduce67) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %586 = insertelement <4 x float> %583, float %585, i64 1, !dbg !27
  %587 = extractelement <4 x float> %lazy.score2047, i64 2, !dbg !25
  %588 = call float @llvm.amdgcn.exp2.f32(float %587), !dbg !25
  %local.reduce68 = fadd float %k0.sum.anchor1873, %588, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1881 = call float asm sideeffect "", "=v,0"(float %local.reduce68) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %589 = insertelement <4 x float> %586, float %588, i64 2, !dbg !27
  %590 = extractelement <4 x float> %lazy.score2047, i64 3, !dbg !25
  %591 = call float @llvm.amdgcn.exp2.f32(float %590), !dbg !25
  %local.reduce69 = fadd float %k0.sum.anchor1874, %591, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1882 = call float asm sideeffect "", "=v,0"(float %local.reduce69) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %592 = insertelement <4 x float> %589, float %591, i64 3, !dbg !27
  %math.result394 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %592) #9, !dbg !27, !frisk.register_lifetimes !1
  %.cvtpack.f02105 = extractelement <4 x float> %math.result394, i32 0, !dbg !27
  %.cvtpack.f12106 = extractelement <4 x float> %math.result394, i32 1, !dbg !27
  %.cvtpack.f22107 = extractelement <4 x float> %math.result394, i32 2, !dbg !27
  %.cvtpack.f32108 = extractelement <4 x float> %math.result394, i32 3, !dbg !27
  %.cvtpack.p02109 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02105, float %.cvtpack.f12106), !dbg !27
  %.cvtpack.p12110 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22107, float %.cvtpack.f32108), !dbg !27
  %.cvtpack.v02111 = insertelement <2 x i32> poison, i32 %.cvtpack.p02109, i32 0, !dbg !27
  %.cvtpack.v12112 = insertelement <2 x i32> %.cvtpack.v02111, i32 %.cvtpack.p12110, i32 1, !dbg !27
  %593 = bitcast <2 x i32> %.cvtpack.v12112 to <4 x half>, !dbg !27
  %math.packed395 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %593) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %594 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %564, <4 x float> %562) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %595 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %579, <4 x float> %563) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %596 = shufflevector <8 x half> %456, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %597 = extractelement <4 x float> %lazy.score2040, i64 0, !dbg !25
  %598 = call float @llvm.amdgcn.exp2.f32(float %597), !dbg !25
  %local.reduce101 = fadd float %k0.sum.anchor1875, %598, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1883 = call float asm sideeffect "", "=v,0"(float %local.reduce101) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %599 = insertelement <4 x float> poison, float %598, i64 0, !dbg !27
  %600 = extractelement <4 x float> %lazy.score2040, i64 1, !dbg !25
  %601 = call float @llvm.amdgcn.exp2.f32(float %600), !dbg !25
  %local.reduce102 = fadd float %k0.sum.anchor1876, %601, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1884 = call float asm sideeffect "", "=v,0"(float %local.reduce102) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %602 = insertelement <4 x float> %599, float %601, i64 1, !dbg !27
  %603 = extractelement <4 x float> %lazy.score2040, i64 2, !dbg !25
  %604 = call float @llvm.amdgcn.exp2.f32(float %603), !dbg !25
  %local.reduce103 = fadd float %k0.sum.anchor1877, %604, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1885 = call float asm sideeffect "", "=v,0"(float %local.reduce103) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %605 = insertelement <4 x float> %602, float %604, i64 2, !dbg !27
  %606 = extractelement <4 x float> %lazy.score2040, i64 3, !dbg !25
  %607 = call float @llvm.amdgcn.exp2.f32(float %606), !dbg !25
  %local.reduce104 = fadd float %k0.sum.anchor1878, %607, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1886 = call float asm sideeffect "", "=v,0"(float %local.reduce104) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %608 = insertelement <4 x float> %605, float %607, i64 3, !dbg !27
  %math.result380 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %608) #9, !dbg !27, !frisk.register_lifetimes !1
  %.cvtpack.f02113 = extractelement <4 x float> %math.result380, i32 0, !dbg !27
  %.cvtpack.f12114 = extractelement <4 x float> %math.result380, i32 1, !dbg !27
  %.cvtpack.f22115 = extractelement <4 x float> %math.result380, i32 2, !dbg !27
  %.cvtpack.f32116 = extractelement <4 x float> %math.result380, i32 3, !dbg !27
  %.cvtpack.p02117 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02113, float %.cvtpack.f12114), !dbg !27
  %.cvtpack.p12118 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22115, float %.cvtpack.f32116), !dbg !27
  %.cvtpack.v02119 = insertelement <2 x i32> poison, i32 %.cvtpack.p02117, i32 0, !dbg !27
  %.cvtpack.v12120 = insertelement <2 x i32> %.cvtpack.v02119, i32 %.cvtpack.p12118, i32 1, !dbg !27
  %609 = bitcast <2 x i32> %.cvtpack.v12120 to <4 x half>, !dbg !27
  %math.packed381 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %609) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %610 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %596, <4 x float> %578) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %611 = shufflevector <8 x half> %456, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %612 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %611, <4 x float> %580) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %613 = extractelement <4 x float> %lazy.score2048, i64 0, !dbg !25
  %614 = call float @llvm.amdgcn.exp2.f32(float %613), !dbg !25
  %local.reduce70 = fadd float %k0.sum.anchor1879, %614, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1887 = call float asm sideeffect "", "=v,0"(float %local.reduce70) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %615 = insertelement <4 x float> poison, float %614, i64 0, !dbg !27
  %616 = extractelement <4 x float> %lazy.score2048, i64 1, !dbg !25
  %617 = call float @llvm.amdgcn.exp2.f32(float %616), !dbg !25
  %local.reduce71 = fadd float %k0.sum.anchor1880, %617, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1888 = call float asm sideeffect "", "=v,0"(float %local.reduce71) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %618 = insertelement <4 x float> %615, float %617, i64 1, !dbg !27
  %619 = extractelement <4 x float> %lazy.score2048, i64 2, !dbg !25
  %620 = call float @llvm.amdgcn.exp2.f32(float %619), !dbg !25
  %local.reduce72 = fadd float %k0.sum.anchor1881, %620, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1889 = call float asm sideeffect "", "=v,0"(float %local.reduce72) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %621 = insertelement <4 x float> %618, float %620, i64 2, !dbg !27
  %622 = extractelement <4 x float> %lazy.score2048, i64 3, !dbg !25
  %623 = call float @llvm.amdgcn.exp2.f32(float %622), !dbg !25
  %local.reduce73 = fadd float %k0.sum.anchor1882, %623, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1890 = call float asm sideeffect "", "=v,0"(float %local.reduce73) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %624 = insertelement <4 x float> %621, float %623, i64 3, !dbg !27
  %math.result396 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %624) #9, !dbg !27, !frisk.register_lifetimes !1
  %.cvtpack.f02121 = extractelement <4 x float> %math.result396, i32 0, !dbg !27
  %.cvtpack.f12122 = extractelement <4 x float> %math.result396, i32 1, !dbg !27
  %.cvtpack.f22123 = extractelement <4 x float> %math.result396, i32 2, !dbg !27
  %.cvtpack.f32124 = extractelement <4 x float> %math.result396, i32 3, !dbg !27
  %.cvtpack.p02125 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02121, float %.cvtpack.f12122), !dbg !27
  %.cvtpack.p12126 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22123, float %.cvtpack.f32124), !dbg !27
  %.cvtpack.v02127 = insertelement <2 x i32> poison, i32 %.cvtpack.p02125, i32 0, !dbg !27
  %.cvtpack.v12128 = insertelement <2 x i32> %.cvtpack.v02127, i32 %.cvtpack.p12126, i32 1, !dbg !27
  %625 = bitcast <2 x i32> %.cvtpack.v12128 to <4 x half>, !dbg !27
  %math.packed397 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %625) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %626 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %596, <4 x float> %594) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %627 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %611, <4 x float> %595) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %628 = shufflevector <8 x half> %458, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %629 = extractelement <4 x float> %lazy.score2041, i64 0, !dbg !25
  %630 = call float @llvm.amdgcn.exp2.f32(float %629), !dbg !25
  %local.reduce105 = fadd float %k0.sum.anchor1883, %630, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1891 = call float asm sideeffect "", "=v,0"(float %local.reduce105) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %631 = insertelement <4 x float> poison, float %630, i64 0, !dbg !27
  %632 = extractelement <4 x float> %lazy.score2041, i64 1, !dbg !25
  %633 = call float @llvm.amdgcn.exp2.f32(float %632), !dbg !25
  %local.reduce106 = fadd float %k0.sum.anchor1884, %633, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1892 = call float asm sideeffect "", "=v,0"(float %local.reduce106) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %634 = insertelement <4 x float> %631, float %633, i64 1, !dbg !27
  %635 = extractelement <4 x float> %lazy.score2041, i64 2, !dbg !25
  %636 = call float @llvm.amdgcn.exp2.f32(float %635), !dbg !25
  %local.reduce107 = fadd float %k0.sum.anchor1885, %636, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1893 = call float asm sideeffect "", "=v,0"(float %local.reduce107) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %637 = insertelement <4 x float> %634, float %636, i64 2, !dbg !27
  %638 = extractelement <4 x float> %lazy.score2041, i64 3, !dbg !25
  %639 = call float @llvm.amdgcn.exp2.f32(float %638), !dbg !25
  %local.reduce108 = fadd float %k0.sum.anchor1886, %639, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1894 = call float asm sideeffect "", "=v,0"(float %local.reduce108) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %640 = insertelement <4 x float> %637, float %639, i64 3, !dbg !27
  %math.result382 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %640) #9, !dbg !27, !frisk.register_lifetimes !1
  %.cvtpack.f02129 = extractelement <4 x float> %math.result382, i32 0, !dbg !27
  %.cvtpack.f12130 = extractelement <4 x float> %math.result382, i32 1, !dbg !27
  %.cvtpack.f22131 = extractelement <4 x float> %math.result382, i32 2, !dbg !27
  %.cvtpack.f32132 = extractelement <4 x float> %math.result382, i32 3, !dbg !27
  %.cvtpack.p02133 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02129, float %.cvtpack.f12130), !dbg !27
  %.cvtpack.p12134 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22131, float %.cvtpack.f32132), !dbg !27
  %.cvtpack.v02135 = insertelement <2 x i32> poison, i32 %.cvtpack.p02133, i32 0, !dbg !27
  %.cvtpack.v12136 = insertelement <2 x i32> %.cvtpack.v02135, i32 %.cvtpack.p12134, i32 1, !dbg !27
  %641 = bitcast <2 x i32> %.cvtpack.v12136 to <4 x half>, !dbg !27
  %math.packed383 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %641) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %642 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %628, <4 x float> %610) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %643 = shufflevector <8 x half> %458, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %644 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %643, <4 x float> %612) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %645 = extractelement <4 x float> %lazy.score2049, i64 0, !dbg !25
  %646 = call float @llvm.amdgcn.exp2.f32(float %645), !dbg !25
  %local.reduce74 = fadd float %k0.sum.anchor1887, %646, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1895 = call float asm sideeffect "", "=v,0"(float %local.reduce74) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %647 = insertelement <4 x float> poison, float %646, i64 0, !dbg !27
  %648 = extractelement <4 x float> %lazy.score2049, i64 1, !dbg !25
  %649 = call float @llvm.amdgcn.exp2.f32(float %648), !dbg !25
  %local.reduce75 = fadd float %k0.sum.anchor1888, %649, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1896 = call float asm sideeffect "", "=v,0"(float %local.reduce75) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %650 = insertelement <4 x float> %647, float %649, i64 1, !dbg !27
  %651 = extractelement <4 x float> %lazy.score2049, i64 2, !dbg !25
  %652 = call float @llvm.amdgcn.exp2.f32(float %651), !dbg !25
  %local.reduce76 = fadd float %k0.sum.anchor1889, %652, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1897 = call float asm sideeffect "", "=v,0"(float %local.reduce76) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %653 = insertelement <4 x float> %650, float %652, i64 2, !dbg !27
  %654 = extractelement <4 x float> %lazy.score2049, i64 3, !dbg !25
  %655 = call float @llvm.amdgcn.exp2.f32(float %654), !dbg !25
  %local.reduce77 = fadd float %k0.sum.anchor1890, %655, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1898 = call float asm sideeffect "", "=v,0"(float %local.reduce77) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %656 = insertelement <4 x float> %653, float %655, i64 3, !dbg !27
  %math.result398 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %656) #9, !dbg !27, !frisk.register_lifetimes !1
  %.cvtpack.f02137 = extractelement <4 x float> %math.result398, i32 0, !dbg !27
  %.cvtpack.f12138 = extractelement <4 x float> %math.result398, i32 1, !dbg !27
  %.cvtpack.f22139 = extractelement <4 x float> %math.result398, i32 2, !dbg !27
  %.cvtpack.f32140 = extractelement <4 x float> %math.result398, i32 3, !dbg !27
  %.cvtpack.p02141 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02137, float %.cvtpack.f12138), !dbg !27
  %.cvtpack.p12142 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22139, float %.cvtpack.f32140), !dbg !27
  %.cvtpack.v02143 = insertelement <2 x i32> poison, i32 %.cvtpack.p02141, i32 0, !dbg !27
  %.cvtpack.v12144 = insertelement <2 x i32> %.cvtpack.v02143, i32 %.cvtpack.p12142, i32 1, !dbg !27
  %657 = bitcast <2 x i32> %.cvtpack.v12144 to <4 x half>, !dbg !27
  %math.packed399 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %657) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %658 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %628, <4 x float> %626) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %659 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %643, <4 x float> %627) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %660 = shufflevector <8 x half> %460, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %661 = extractelement <4 x float> %lazy.score2042, i64 0, !dbg !25
  %662 = call float @llvm.amdgcn.exp2.f32(float %661), !dbg !25
  %local.reduce109 = fadd float %k0.sum.anchor1891, %662, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1899 = call float asm sideeffect "", "=v,0"(float %local.reduce109) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %663 = insertelement <4 x float> poison, float %662, i64 0, !dbg !27
  %664 = extractelement <4 x float> %lazy.score2042, i64 1, !dbg !25
  %665 = call float @llvm.amdgcn.exp2.f32(float %664), !dbg !25
  %local.reduce110 = fadd float %k0.sum.anchor1892, %665, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1900 = call float asm sideeffect "", "=v,0"(float %local.reduce110) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %666 = insertelement <4 x float> %663, float %665, i64 1, !dbg !27
  %667 = extractelement <4 x float> %lazy.score2042, i64 2, !dbg !25
  %668 = call float @llvm.amdgcn.exp2.f32(float %667), !dbg !25
  %local.reduce111 = fadd float %k0.sum.anchor1893, %668, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1901 = call float asm sideeffect "", "=v,0"(float %local.reduce111) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %669 = insertelement <4 x float> %666, float %668, i64 2, !dbg !27
  %670 = extractelement <4 x float> %lazy.score2042, i64 3, !dbg !25
  %671 = call float @llvm.amdgcn.exp2.f32(float %670), !dbg !25
  %local.reduce112 = fadd float %k0.sum.anchor1894, %671, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1902 = call float asm sideeffect "", "=v,0"(float %local.reduce112) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %672 = insertelement <4 x float> %669, float %671, i64 3, !dbg !27
  %math.result384 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %672) #9, !dbg !27, !frisk.register_lifetimes !1
  %.cvtpack.f02145 = extractelement <4 x float> %math.result384, i32 0, !dbg !27
  %.cvtpack.f12146 = extractelement <4 x float> %math.result384, i32 1, !dbg !27
  %.cvtpack.f22147 = extractelement <4 x float> %math.result384, i32 2, !dbg !27
  %.cvtpack.f32148 = extractelement <4 x float> %math.result384, i32 3, !dbg !27
  %.cvtpack.p02149 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02145, float %.cvtpack.f12146), !dbg !27
  %.cvtpack.p12150 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22147, float %.cvtpack.f32148), !dbg !27
  %.cvtpack.v02151 = insertelement <2 x i32> poison, i32 %.cvtpack.p02149, i32 0, !dbg !27
  %.cvtpack.v12152 = insertelement <2 x i32> %.cvtpack.v02151, i32 %.cvtpack.p12150, i32 1, !dbg !27
  %673 = bitcast <2 x i32> %.cvtpack.v12152 to <4 x half>, !dbg !27
  %math.packed385 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %673) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %674 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %660, <4 x float> %642) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %675 = shufflevector <8 x half> %460, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %676 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %675, <4 x float> %644) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %677 = extractelement <4 x float> %lazy.score2050, i64 0, !dbg !25
  %678 = call float @llvm.amdgcn.exp2.f32(float %677), !dbg !25
  %local.reduce78 = fadd float %k0.sum.anchor1895, %678, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1903 = call float asm sideeffect "", "=v,0"(float %local.reduce78) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %679 = insertelement <4 x float> poison, float %678, i64 0, !dbg !27
  %680 = extractelement <4 x float> %lazy.score2050, i64 1, !dbg !25
  %681 = call float @llvm.amdgcn.exp2.f32(float %680), !dbg !25
  %local.reduce79 = fadd float %k0.sum.anchor1896, %681, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1904 = call float asm sideeffect "", "=v,0"(float %local.reduce79) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %682 = insertelement <4 x float> %679, float %681, i64 1, !dbg !27
  %683 = extractelement <4 x float> %lazy.score2050, i64 2, !dbg !25
  %684 = call float @llvm.amdgcn.exp2.f32(float %683), !dbg !25
  %local.reduce80 = fadd float %k0.sum.anchor1897, %684, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1905 = call float asm sideeffect "", "=v,0"(float %local.reduce80) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %685 = insertelement <4 x float> %682, float %684, i64 2, !dbg !27
  %686 = extractelement <4 x float> %lazy.score2050, i64 3, !dbg !25
  %687 = call float @llvm.amdgcn.exp2.f32(float %686), !dbg !25
  %local.reduce81 = fadd float %k0.sum.anchor1898, %687, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1906 = call float asm sideeffect "", "=v,0"(float %local.reduce81) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %688 = insertelement <4 x float> %685, float %687, i64 3, !dbg !27
  %math.result400 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %688) #9, !dbg !27, !frisk.register_lifetimes !1
  %.cvtpack.f02153 = extractelement <4 x float> %math.result400, i32 0, !dbg !27
  %.cvtpack.f12154 = extractelement <4 x float> %math.result400, i32 1, !dbg !27
  %.cvtpack.f22155 = extractelement <4 x float> %math.result400, i32 2, !dbg !27
  %.cvtpack.f32156 = extractelement <4 x float> %math.result400, i32 3, !dbg !27
  %.cvtpack.p02157 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02153, float %.cvtpack.f12154), !dbg !27
  %.cvtpack.p12158 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22155, float %.cvtpack.f32156), !dbg !27
  %.cvtpack.v02159 = insertelement <2 x i32> poison, i32 %.cvtpack.p02157, i32 0, !dbg !27
  %.cvtpack.v12160 = insertelement <2 x i32> %.cvtpack.v02159, i32 %.cvtpack.p12158, i32 1, !dbg !27
  %689 = bitcast <2 x i32> %.cvtpack.v12160 to <4 x half>, !dbg !27
  %math.packed401 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %689) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %690 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %660, <4 x float> %658) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %691 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %675, <4 x float> %659) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %692 = bitcast <4 x i32> %461 to <8 x half>, !dbg !14
  %693 = shufflevector <8 x half> %692, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %694 = extractelement <4 x float> %lazy.score2043, i64 0, !dbg !25
  %695 = call float @llvm.amdgcn.exp2.f32(float %694), !dbg !25
  %local.reduce113 = fadd float %k0.sum.anchor1899, %695, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1907 = call float asm sideeffect "", "=v,0"(float %local.reduce113) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %696 = insertelement <4 x float> poison, float %695, i64 0, !dbg !27
  %697 = extractelement <4 x float> %lazy.score2043, i64 1, !dbg !25
  %698 = call float @llvm.amdgcn.exp2.f32(float %697), !dbg !25
  %local.reduce114 = fadd float %k0.sum.anchor1900, %698, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1908 = call float asm sideeffect "", "=v,0"(float %local.reduce114) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %local.reduce117 = fadd float %k0.sum.anchor1907, %k0.sum.anchor1908, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1909 = call float asm sideeffect "", "=v,0"(float %local.reduce117) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %699 = insertelement <4 x float> %696, float %698, i64 1, !dbg !27
  %700 = extractelement <4 x float> %lazy.score2043, i64 2, !dbg !25
  %701 = call float @llvm.amdgcn.exp2.f32(float %700), !dbg !25
  %local.reduce115 = fadd float %k0.sum.anchor1901, %701, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1910 = call float asm sideeffect "", "=v,0"(float %local.reduce115) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %702 = insertelement <4 x float> %699, float %701, i64 2, !dbg !27
  %703 = extractelement <4 x float> %lazy.score2043, i64 3, !dbg !25
  %704 = call float @llvm.amdgcn.exp2.f32(float %703), !dbg !25
  %local.reduce116 = fadd float %k0.sum.anchor1902, %704, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1911 = call float asm sideeffect "", "=v,0"(float %local.reduce116) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %local.reduce118 = fadd float %k0.sum.anchor1910, %k0.sum.anchor1911, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1912 = call float asm sideeffect "", "=v,0"(float %local.reduce118) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %705 = fadd float %k0.sum.anchor1909, %k0.sum.anchor1912, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1913 = call float asm sideeffect "", "=v,0"(float %705) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %706 = fadd float %k0.sum.anchor1913, %442, !dbg !26
  %k0.sum.anchor1914 = call float asm sideeffect "", "=v,0"(float %706) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %707 = insertelement <1 x float> poison, float %k0.sum.anchor1914, i64 0, !dbg !8
  %708 = insertvalue [2 x <1 x float>] poison, <1 x float> %707, 0, !dbg !8
  %709 = insertelement <4 x float> %702, float %704, i64 3, !dbg !27
  %math.result386 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %709) #9, !dbg !27, !frisk.register_lifetimes !1
  %.cvtpack.f02161 = extractelement <4 x float> %math.result386, i32 0, !dbg !27
  %.cvtpack.f12162 = extractelement <4 x float> %math.result386, i32 1, !dbg !27
  %.cvtpack.f22163 = extractelement <4 x float> %math.result386, i32 2, !dbg !27
  %.cvtpack.f32164 = extractelement <4 x float> %math.result386, i32 3, !dbg !27
  %.cvtpack.p02165 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02161, float %.cvtpack.f12162), !dbg !27
  %.cvtpack.p12166 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22163, float %.cvtpack.f32164), !dbg !27
  %.cvtpack.v02167 = insertelement <2 x i32> poison, i32 %.cvtpack.p02165, i32 0, !dbg !27
  %.cvtpack.v12168 = insertelement <2 x i32> %.cvtpack.v02167, i32 %.cvtpack.p12166, i32 1, !dbg !27
  %710 = bitcast <2 x i32> %.cvtpack.v12168 to <4 x half>, !dbg !27
  %math.packed387 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %710) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %711 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %693, <4 x float> %674) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %712 = shufflevector <8 x half> %692, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %713 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %712, <4 x float> %676) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %714 = extractelement <4 x float> %lazy.score2051, i64 0, !dbg !25
  %715 = call float @llvm.amdgcn.exp2.f32(float %714), !dbg !25
  %local.reduce82 = fadd float %k0.sum.anchor1903, %715, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1915 = call float asm sideeffect "", "=v,0"(float %local.reduce82) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %716 = insertelement <4 x float> poison, float %715, i64 0, !dbg !27
  %717 = extractelement <4 x float> %lazy.score2051, i64 1, !dbg !25
  %718 = call float @llvm.amdgcn.exp2.f32(float %717), !dbg !25
  %local.reduce83 = fadd float %k0.sum.anchor1904, %718, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1916 = call float asm sideeffect "", "=v,0"(float %local.reduce83) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %local.reduce86 = fadd float %k0.sum.anchor1915, %k0.sum.anchor1916, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1917 = call float asm sideeffect "", "=v,0"(float %local.reduce86) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %719 = insertelement <4 x float> %716, float %718, i64 1, !dbg !27
  %720 = extractelement <4 x float> %lazy.score2051, i64 2, !dbg !25
  %721 = call float @llvm.amdgcn.exp2.f32(float %720), !dbg !25
  %local.reduce84 = fadd float %k0.sum.anchor1905, %721, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1918 = call float asm sideeffect "", "=v,0"(float %local.reduce84) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %722 = insertelement <4 x float> %719, float %721, i64 2, !dbg !27
  %723 = extractelement <4 x float> %lazy.score2051, i64 3, !dbg !25
  %724 = call float @llvm.amdgcn.exp2.f32(float %723), !dbg !25
  %local.reduce85 = fadd float %k0.sum.anchor1906, %724, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1919 = call float asm sideeffect "", "=v,0"(float %local.reduce85) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %local.reduce87 = fadd float %k0.sum.anchor1918, %k0.sum.anchor1919, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1920 = call float asm sideeffect "", "=v,0"(float %local.reduce87) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %725 = fadd float %k0.sum.anchor1917, %k0.sum.anchor1920, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1921 = call float asm sideeffect "", "=v,0"(float %725) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %726 = fadd float %k0.sum.anchor1921, %444, !dbg !26
  %k0.sum.anchor1922 = call float asm sideeffect "", "=v,0"(float %726) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %727 = insertelement <1 x float> poison, float %k0.sum.anchor1922, i64 0, !dbg !8
  %728 = insertvalue [2 x <1 x float>] %708, <1 x float> %727, 1, !dbg !8
  %729 = insertelement <4 x float> %722, float %724, i64 3, !dbg !27
  %math.result402 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %729) #9, !dbg !27, !frisk.register_lifetimes !1
  %.cvtpack.f02169 = extractelement <4 x float> %math.result402, i32 0, !dbg !27
  %.cvtpack.f12170 = extractelement <4 x float> %math.result402, i32 1, !dbg !27
  %.cvtpack.f22171 = extractelement <4 x float> %math.result402, i32 2, !dbg !27
  %.cvtpack.f32172 = extractelement <4 x float> %math.result402, i32 3, !dbg !27
  %.cvtpack.p02173 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02169, float %.cvtpack.f12170), !dbg !27
  %.cvtpack.p12174 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22171, float %.cvtpack.f32172), !dbg !27
  %.cvtpack.v02175 = insertelement <2 x i32> poison, i32 %.cvtpack.p02173, i32 0, !dbg !27
  %.cvtpack.v12176 = insertelement <2 x i32> %.cvtpack.v02175, i32 %.cvtpack.p12174, i32 1, !dbg !27
  %730 = bitcast <2 x i32> %.cvtpack.v12176 to <4 x half>, !dbg !27
  %math.packed403 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %730) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %731 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %693, <4 x float> %690) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %732 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %712, <4 x float> %691) #9, !dbg !14, !frisk.pv_interleave !33
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
  %733 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %36, i32 %vdb.address1430, <4 x i32> %dma.buffer.rsrc1327) #10, !dbg !16
  %734 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %43, i32 %vdb.address1437, <4 x i32> %dma.buffer.rsrc1327) #10, !dbg !16
  %vdb.read = getelementptr i8, ptr addrspace(3) %address.local919, i64 8192, !dbg !14
  %735 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %736 = bitcast <4 x i32> %735 to <8 x half>, !dbg !14
  %vdb.read1421 = getelementptr i8, ptr addrspace(3) %address.local928, i64 8448, !dbg !14
  %737 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1421, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %738 = bitcast <4 x i32> %737 to <8 x half>, !dbg !14
  %vdb.read1422 = getelementptr i8, ptr addrspace(3) %address.local928, i64 8704, !dbg !14
  %739 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1422, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %740 = bitcast <4 x i32> %739 to <8 x half>, !dbg !14
  %vdb.read1423 = getelementptr i8, ptr addrspace(3) %address.local928, i64 8960, !dbg !14
  %741 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1423, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %742 = bitcast <4 x i32> %741 to <8 x half>, !dbg !14
  %vdb.read1424 = getelementptr i8, ptr addrspace(3) %address.local919, i64 12288, !dbg !14
  %743 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1424, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %744 = bitcast <4 x i32> %743 to <8 x half>, !dbg !14
  %vdb.read1425 = getelementptr i8, ptr addrspace(3) %address.local928, i64 12544, !dbg !14
  %745 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1425, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %746 = bitcast <4 x i32> %745 to <8 x half>, !dbg !14
  %vdb.read1426 = getelementptr i8, ptr addrspace(3) %address.local928, i64 12800, !dbg !14
  %747 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1426, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %748 = bitcast <4 x i32> %747 to <8 x half>, !dbg !14
  %vdb.read1427 = getelementptr i8, ptr addrspace(3) %address.local928, i64 13056, !dbg !14
  %749 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1427, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %750 = shufflevector <8 x half> %736, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %751 = extractvalue [2 x <8 x float>] %lazy.state2055, 0, !dbg !28
  %752 = shufflevector <8 x float> %751, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %753 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %750, <4 x float> %752) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %754 = shufflevector <8 x half> %736, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %755 = shufflevector <8 x float> %751, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %756 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %754, <4 x float> %755) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %757 = extractvalue [2 x <8 x float>] %lazy.state2055, 1, !dbg !28
  %758 = shufflevector <8 x float> %757, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %759 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %750, <4 x float> %758) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %760 = shufflevector <8 x float> %757, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %761 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %754, <4 x float> %760) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %762 = shufflevector <8 x half> %738, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %763 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %762, <4 x float> %753) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %764 = shufflevector <8 x half> %738, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %765 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %764, <4 x float> %756) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %766 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %762, <4 x float> %759) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %767 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %764, <4 x float> %761) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %768 = shufflevector <8 x half> %740, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %769 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %768, <4 x float> %763) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %770 = shufflevector <8 x half> %740, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %771 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %770, <4 x float> %765) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %772 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %768, <4 x float> %766) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %773 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %770, <4 x float> %767) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %774 = shufflevector <8 x half> %742, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %775 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %774, <4 x float> %769) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %776 = shufflevector <8 x half> %742, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %777 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %776, <4 x float> %771) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %778 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %774, <4 x float> %772) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %779 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %776, <4 x float> %773) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %780 = shufflevector <8 x half> %744, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %781 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %780, <4 x float> %775) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %782 = shufflevector <8 x half> %744, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %783 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %782, <4 x float> %777) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %784 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %780, <4 x float> %778) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %785 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %782, <4 x float> %779) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %786 = shufflevector <8 x half> %746, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %787 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %786, <4 x float> %781) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %788 = shufflevector <8 x half> %746, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %789 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %788, <4 x float> %783) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %790 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %786, <4 x float> %784) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %791 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %788, <4 x float> %785) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %792 = shufflevector <8 x half> %748, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %793 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %792, <4 x float> %787) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %794 = shufflevector <8 x half> %748, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %795 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %794, <4 x float> %789) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %796 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %792, <4 x float> %790) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %797 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %794, <4 x float> %791) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %798 = bitcast <4 x i32> %749 to <8 x half>, !dbg !14
  %799 = shufflevector <8 x half> %798, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %800 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %799, <4 x float> %793) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %801 = shufflevector <8 x half> %798, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %802 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %801, <4 x float> %795) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %803 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %799, <4 x float> %796) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %804 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %801, <4 x float> %797) #9, !dbg !14, !frisk.pv_interleave !33
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
  %805 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1444, <4 x i32> %dma.buffer.rsrc1327) #10, !dbg !16
  %806 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1420, i32 %vdb.address1451, <4 x i32> %dma.buffer.rsrc1327) #10, !dbg !16
  %807 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local919, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %808 = bitcast <4 x i32> %807 to <8 x half>, !dbg !14
  %809 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local929, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %810 = bitcast <4 x i32> %809 to <8 x half>, !dbg !14
  %811 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local933, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %812 = bitcast <4 x i32> %811 to <8 x half>, !dbg !14
  %813 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local937, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %814 = bitcast <4 x i32> %813 to <8 x half>, !dbg !14
  %815 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local939, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %816 = bitcast <4 x i32> %815 to <8 x half>, !dbg !14
  %817 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local943, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %818 = bitcast <4 x i32> %817 to <8 x half>, !dbg !14
  %819 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local947, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %820 = bitcast <4 x i32> %819 to <8 x half>, !dbg !14
  %821 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local951, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %822 = shufflevector <8 x half> %808, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %823 = extractvalue [2 x <8 x float>] %lazy.state2054, 0, !dbg !28
  %824 = shufflevector <8 x float> %823, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %825 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %822, <4 x float> %824) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %826 = shufflevector <8 x half> %808, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %827 = shufflevector <8 x float> %823, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %828 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %826, <4 x float> %827) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %829 = extractvalue [2 x <8 x float>] %lazy.state2054, 1, !dbg !28
  %830 = shufflevector <8 x float> %829, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %831 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %822, <4 x float> %830) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %832 = shufflevector <8 x float> %829, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %833 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %826, <4 x float> %832) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %834 = shufflevector <8 x half> %810, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %835 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %834, <4 x float> %825) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %836 = shufflevector <8 x half> %810, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %837 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %836, <4 x float> %828) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %838 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %834, <4 x float> %831) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %839 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %836, <4 x float> %833) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %840 = shufflevector <8 x half> %812, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %841 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %840, <4 x float> %835) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %842 = shufflevector <8 x half> %812, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %843 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %842, <4 x float> %837) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %844 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %840, <4 x float> %838) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %845 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %842, <4 x float> %839) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %846 = shufflevector <8 x half> %814, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %847 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %846, <4 x float> %841) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %848 = shufflevector <8 x half> %814, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %849 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %848, <4 x float> %843) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %850 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %846, <4 x float> %844) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %851 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %848, <4 x float> %845) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %852 = shufflevector <8 x half> %816, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %853 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %852, <4 x float> %847) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %854 = shufflevector <8 x half> %816, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %855 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %854, <4 x float> %849) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %856 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %852, <4 x float> %850) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %857 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %854, <4 x float> %851) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %858 = shufflevector <8 x half> %818, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %859 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %858, <4 x float> %853) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %860 = shufflevector <8 x half> %818, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %861 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %860, <4 x float> %855) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %862 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %858, <4 x float> %856) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %863 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %860, <4 x float> %857) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %864 = shufflevector <8 x half> %820, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %865 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %864, <4 x float> %859) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %866 = shufflevector <8 x half> %820, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %867 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %866, <4 x float> %861) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %868 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %864, <4 x float> %862) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %869 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %866, <4 x float> %863) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %870 = bitcast <4 x i32> %821 to <8 x half>, !dbg !14
  %871 = shufflevector <8 x half> %870, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %872 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %871, <4 x float> %865) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %873 = shufflevector <8 x half> %870, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %874 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %873, <4 x float> %867) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %875 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %871, <4 x float> %868) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %876 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %873, <4 x float> %869) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %877 = add nuw nsw i64 %104, 1, !dbg !8
  %k0.has.next = icmp ult i64 %877, 32, !dbg !14
  br i1 %k0.has.next, label %k0.issue, label %k0.join, !dbg !14

k0.issue:                                         ; preds = %lazy.join
  %k0.next. = shl nuw nsw i64 %877, 14, !dbg !13
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

k0.join:                                          ; preds = %lazy.join, %k0.issue
  %878 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %879 = bitcast <4 x i32> %878 to <8 x half>, !dbg !14
  %880 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1421, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %881 = bitcast <4 x i32> %880 to <8 x half>, !dbg !14
  %882 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1422, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %883 = bitcast <4 x i32> %882 to <8 x half>, !dbg !14
  %884 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1423, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %885 = bitcast <4 x i32> %884 to <8 x half>, !dbg !14
  %886 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1424, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %887 = bitcast <4 x i32> %886 to <8 x half>, !dbg !14
  %888 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1425, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %889 = bitcast <4 x i32> %888 to <8 x half>, !dbg !14
  %890 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1426, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %891 = bitcast <4 x i32> %890 to <8 x half>, !dbg !14
  %892 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1427, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %893 = shufflevector <8 x half> %879, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %894 = extractvalue [2 x <8 x float>] %lazy.state2053, 0, !dbg !28
  %895 = shufflevector <8 x float> %894, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %896 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %893, <4 x float> %895) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %897 = shufflevector <8 x half> %879, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %898 = shufflevector <8 x float> %894, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %899 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %897, <4 x float> %898) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %900 = extractvalue [2 x <8 x float>] %lazy.state2053, 1, !dbg !28
  %901 = shufflevector <8 x float> %900, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %902 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %893, <4 x float> %901) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %903 = shufflevector <8 x float> %900, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %904 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %897, <4 x float> %903) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %905 = shufflevector <8 x half> %881, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %906 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %905, <4 x float> %896) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %907 = shufflevector <8 x half> %881, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %908 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %907, <4 x float> %899) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %909 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %905, <4 x float> %902) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %910 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %907, <4 x float> %904) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %911 = shufflevector <8 x half> %883, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %912 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %911, <4 x float> %906) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %913 = shufflevector <8 x half> %883, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %914 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %913, <4 x float> %908) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %915 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %911, <4 x float> %909) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %916 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %913, <4 x float> %910) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %917 = shufflevector <8 x half> %885, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %918 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %917, <4 x float> %912) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %919 = shufflevector <8 x half> %885, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %920 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %919, <4 x float> %914) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %921 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %917, <4 x float> %915) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %922 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %919, <4 x float> %916) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %923 = shufflevector <8 x half> %887, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %924 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %923, <4 x float> %918) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %925 = shufflevector <8 x half> %887, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %926 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %925, <4 x float> %920) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %927 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %923, <4 x float> %921) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %928 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %925, <4 x float> %922) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %929 = shufflevector <8 x half> %889, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %930 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %929, <4 x float> %924) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %931 = shufflevector <8 x half> %889, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %932 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %931, <4 x float> %926) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %933 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %929, <4 x float> %927) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %934 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %931, <4 x float> %928) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %935 = shufflevector <8 x half> %891, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %936 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %935, <4 x float> %930) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %937 = shufflevector <8 x half> %891, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %938 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %937, <4 x float> %932) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %939 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %935, <4 x float> %933) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %940 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %937, <4 x float> %934) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %941 = bitcast <4 x i32> %892 to <8 x half>, !dbg !14
  %942 = shufflevector <8 x half> %941, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %943 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %942, <4 x float> %936) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %944 = shufflevector <8 x half> %941, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %945 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %944, <4 x float> %938) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %946 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %942, <4 x float> %939) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %947 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %944, <4 x float> %940) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !8
  %948 = shufflevector <4 x float> %711, <4 x float> %713, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %949 = insertvalue [2 x <8 x float>] poison, <8 x float> %948, 0, !dbg !8
  %950 = shufflevector <4 x float> %731, <4 x float> %732, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %951 = insertvalue [2 x <8 x float>] %949, <8 x float> %950, 1, !dbg !8
  %952 = shufflevector <4 x float> %800, <4 x float> %802, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %953 = insertvalue [2 x <8 x float>] poison, <8 x float> %952, 0, !dbg !8
  %954 = shufflevector <4 x float> %803, <4 x float> %804, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %955 = insertvalue [2 x <8 x float>] %953, <8 x float> %954, 1, !dbg !8
  %956 = shufflevector <4 x float> %872, <4 x float> %874, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %957 = insertvalue [2 x <8 x float>] poison, <8 x float> %956, 0, !dbg !8
  %958 = shufflevector <4 x float> %875, <4 x float> %876, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %959 = insertvalue [2 x <8 x float>] %957, <8 x float> %958, 1, !dbg !8
  %960 = shufflevector <4 x float> %943, <4 x float> %945, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %961 = insertvalue [2 x <8 x float>] poison, <8 x float> %960, 0, !dbg !8
  %962 = shufflevector <4 x float> %946, <4 x float> %947, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %963 = insertvalue [2 x <8 x float>] %961, <8 x float> %962, 1, !dbg !8
  br label %103, !dbg !8

964:                                              ; preds = %103
  %lane.sum.field184 = extractvalue [2 x <1 x float>] %106, 1, !dbg !8
  %lane.sum.local185 = extractelement <1 x float> %lane.sum.field184, i64 0, !dbg !8
  %965 = shl nuw nsw i32 %8, 2, !dbg !8
  %966 = xor i32 %965, 64, !dbg !8
  %967 = bitcast float %lane.sum.local185 to i32, !dbg !8
  %lane.sum.shuffle188 = call i32 @llvm.amdgcn.ds.bpermute(i32 %966, i32 %967), !dbg !8
  %968 = bitcast i32 %lane.sum.shuffle188 to float, !dbg !8
  %lane.sum.final189 = fadd float %lane.sum.local185, %968, !dbg !8
  %969 = xor i32 %965, 128, !dbg !8
  %970 = bitcast float %lane.sum.final189 to i32, !dbg !8
  %lane.sum.shuffle190 = call i32 @llvm.amdgcn.ds.bpermute(i32 %969, i32 %970), !dbg !8
  %971 = bitcast i32 %lane.sum.shuffle190 to float, !dbg !8
  %lane.sum.final191 = fadd float %lane.sum.final189, %971, !dbg !8
  %972 = insertelement <1 x float> poison, float %lane.sum.final191, i64 0, !dbg !8
  %lane.sum.field = extractvalue [2 x <1 x float>] %106, 0, !dbg !8
  %lane.sum.local = extractelement <1 x float> %lane.sum.field, i64 0, !dbg !8
  %973 = bitcast float %lane.sum.local to i32, !dbg !8
  %lane.sum.shuffle = call i32 @llvm.amdgcn.ds.bpermute(i32 %966, i32 %973), !dbg !8
  %974 = bitcast i32 %lane.sum.shuffle to float, !dbg !8
  %lane.sum.final = fadd float %lane.sum.local, %974, !dbg !8
  %975 = bitcast float %lane.sum.final to i32, !dbg !8
  %lane.sum.shuffle182 = call i32 @llvm.amdgcn.ds.bpermute(i32 %969, i32 %975), !dbg !8
  %976 = bitcast i32 %lane.sum.shuffle182 to float, !dbg !8
  %lane.sum.final183 = fadd float %lane.sum.final, %976, !dbg !8
  %977 = insertelement <1 x float> poison, float %lane.sum.final183, i64 0, !dbg !8
  %978 = and i32 %kdma.tid, 16, !dbg !8
  %.not = icmp eq i32 %978, 0, !dbg !8
  %979 = and i32 %kdma.tid, 32, !dbg !8
  %.not36 = icmp eq i32 %979, 0, !dbg !8
  %980 = extractvalue [2 x <8 x float>] %110, 0, !dbg !8
  %981 = extractelement <8 x float> %980, i64 0, !dbg !8
  %982 = extractelement <8 x float> %980, i64 1, !dbg !8
  %983 = extractelement <8 x float> %980, i64 2, !dbg !8
  %984 = extractelement <8 x float> %980, i64 3, !dbg !8
  %985 = select i1 %.not, float %982, float %981, !dbg !8
  %986 = bitcast float %985 to i32, !dbg !8
  %987 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %986), !dbg !8
  %988 = bitcast i32 %987 to float, !dbg !8
  %989 = select i1 %.not, float %981, float %988, !dbg !8
  %990 = select i1 %.not, float %988, float %982, !dbg !8
  %991 = select i1 %.not, float %984, float %983, !dbg !8
  %992 = bitcast float %991 to i32, !dbg !8
  %993 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %992), !dbg !8
  %994 = bitcast i32 %993 to float, !dbg !8
  %995 = select i1 %.not, float %983, float %994, !dbg !8
  %996 = select i1 %.not, float %994, float %984, !dbg !8
  %997 = select i1 %.not36, float %995, float %989, !dbg !8
  %998 = bitcast float %997 to i32, !dbg !8
  %999 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %998), !dbg !8
  %1000 = bitcast i32 %999 to float, !dbg !8
  %1001 = select i1 %.not36, float %989, float %1000, !dbg !8
  %1002 = select i1 %.not36, float %1000, float %995, !dbg !8
  %1003 = select i1 %.not36, float %996, float %990, !dbg !8
  %1004 = bitcast float %1003 to i32, !dbg !8
  %1005 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1004), !dbg !8
  %1006 = bitcast i32 %1005 to float, !dbg !8
  %1007 = select i1 %.not36, float %990, float %1006, !dbg !8
  %1008 = select i1 %.not36, float %1006, float %996, !dbg !8
  %1009 = extractelement <8 x float> %980, i64 4, !dbg !8
  %1010 = extractelement <8 x float> %980, i64 5, !dbg !8
  %1011 = extractelement <8 x float> %980, i64 6, !dbg !8
  %1012 = extractelement <8 x float> %980, i64 7, !dbg !8
  %1013 = select i1 %.not, float %1010, float %1009, !dbg !8
  %1014 = bitcast float %1013 to i32, !dbg !8
  %1015 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1014), !dbg !8
  %1016 = bitcast i32 %1015 to float, !dbg !8
  %1017 = select i1 %.not, float %1009, float %1016, !dbg !8
  %1018 = select i1 %.not, float %1016, float %1010, !dbg !8
  %1019 = select i1 %.not, float %1012, float %1011, !dbg !8
  %1020 = bitcast float %1019 to i32, !dbg !8
  %1021 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1020), !dbg !8
  %1022 = bitcast i32 %1021 to float, !dbg !8
  %1023 = select i1 %.not, float %1011, float %1022, !dbg !8
  %1024 = select i1 %.not, float %1022, float %1012, !dbg !8
  %1025 = select i1 %.not36, float %1023, float %1017, !dbg !8
  %1026 = bitcast float %1025 to i32, !dbg !8
  %1027 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1026), !dbg !8
  %1028 = bitcast i32 %1027 to float, !dbg !8
  %1029 = select i1 %.not36, float %1017, float %1028, !dbg !8
  %1030 = select i1 %.not36, float %1028, float %1023, !dbg !8
  %1031 = select i1 %.not36, float %1024, float %1018, !dbg !8
  %1032 = bitcast float %1031 to i32, !dbg !8
  %1033 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1032), !dbg !8
  %1034 = bitcast i32 %1033 to float, !dbg !8
  %1035 = select i1 %.not36, float %1018, float %1034, !dbg !8
  %1036 = select i1 %.not36, float %1034, float %1024, !dbg !8
  %1037 = extractvalue [2 x <8 x float>] %110, 1, !dbg !8
  %1038 = extractelement <8 x float> %1037, i64 0, !dbg !8
  %1039 = extractelement <8 x float> %1037, i64 1, !dbg !8
  %1040 = extractelement <8 x float> %1037, i64 2, !dbg !8
  %1041 = extractelement <8 x float> %1037, i64 3, !dbg !8
  %1042 = select i1 %.not, float %1039, float %1038, !dbg !8
  %1043 = bitcast float %1042 to i32, !dbg !8
  %1044 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1043), !dbg !8
  %1045 = bitcast i32 %1044 to float, !dbg !8
  %1046 = select i1 %.not, float %1038, float %1045, !dbg !8
  %1047 = select i1 %.not, float %1045, float %1039, !dbg !8
  %1048 = select i1 %.not, float %1041, float %1040, !dbg !8
  %1049 = bitcast float %1048 to i32, !dbg !8
  %1050 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1049), !dbg !8
  %1051 = bitcast i32 %1050 to float, !dbg !8
  %1052 = select i1 %.not, float %1040, float %1051, !dbg !8
  %1053 = select i1 %.not, float %1051, float %1041, !dbg !8
  %1054 = select i1 %.not36, float %1052, float %1046, !dbg !8
  %1055 = bitcast float %1054 to i32, !dbg !8
  %1056 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1055), !dbg !8
  %1057 = bitcast i32 %1056 to float, !dbg !8
  %1058 = select i1 %.not36, float %1046, float %1057, !dbg !8
  %1059 = select i1 %.not36, float %1057, float %1052, !dbg !8
  %1060 = select i1 %.not36, float %1053, float %1047, !dbg !8
  %1061 = bitcast float %1060 to i32, !dbg !8
  %1062 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1061), !dbg !8
  %1063 = bitcast i32 %1062 to float, !dbg !8
  %1064 = select i1 %.not36, float %1047, float %1063, !dbg !8
  %1065 = select i1 %.not36, float %1063, float %1053, !dbg !8
  %1066 = extractelement <8 x float> %1037, i64 4, !dbg !8
  %1067 = extractelement <8 x float> %1037, i64 5, !dbg !8
  %1068 = extractelement <8 x float> %1037, i64 6, !dbg !8
  %1069 = extractelement <8 x float> %1037, i64 7, !dbg !8
  %1070 = select i1 %.not, float %1067, float %1066, !dbg !8
  %1071 = bitcast float %1070 to i32, !dbg !8
  %1072 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1071), !dbg !8
  %1073 = bitcast i32 %1072 to float, !dbg !8
  %1074 = select i1 %.not, float %1066, float %1073, !dbg !8
  %1075 = select i1 %.not, float %1073, float %1067, !dbg !8
  %1076 = select i1 %.not, float %1069, float %1068, !dbg !8
  %1077 = bitcast float %1076 to i32, !dbg !8
  %1078 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1077), !dbg !8
  %1079 = bitcast i32 %1078 to float, !dbg !8
  %1080 = select i1 %.not, float %1068, float %1079, !dbg !8
  %1081 = select i1 %.not, float %1079, float %1069, !dbg !8
  %1082 = select i1 %.not36, float %1080, float %1074, !dbg !8
  %1083 = bitcast float %1082 to i32, !dbg !8
  %1084 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1083), !dbg !8
  %1085 = bitcast i32 %1084 to float, !dbg !8
  %1086 = select i1 %.not36, float %1074, float %1085, !dbg !8
  %1087 = select i1 %.not36, float %1085, float %1080, !dbg !8
  %1088 = select i1 %.not36, float %1081, float %1075, !dbg !8
  %1089 = bitcast float %1088 to i32, !dbg !8
  %1090 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1089), !dbg !8
  %1091 = bitcast i32 %1090 to float, !dbg !8
  %1092 = select i1 %.not36, float %1075, float %1091, !dbg !8
  %1093 = select i1 %.not36, float %1091, float %1081, !dbg !8
  %1094 = extractvalue [2 x <8 x float>] %109, 0, !dbg !8
  %1095 = extractelement <8 x float> %1094, i64 0, !dbg !8
  %1096 = extractelement <8 x float> %1094, i64 1, !dbg !8
  %1097 = extractelement <8 x float> %1094, i64 2, !dbg !8
  %1098 = extractelement <8 x float> %1094, i64 3, !dbg !8
  %1099 = select i1 %.not, float %1096, float %1095, !dbg !8
  %1100 = bitcast float %1099 to i32, !dbg !8
  %1101 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1100), !dbg !8
  %1102 = bitcast i32 %1101 to float, !dbg !8
  %1103 = select i1 %.not, float %1095, float %1102, !dbg !8
  %1104 = select i1 %.not, float %1102, float %1096, !dbg !8
  %1105 = select i1 %.not, float %1098, float %1097, !dbg !8
  %1106 = bitcast float %1105 to i32, !dbg !8
  %1107 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1106), !dbg !8
  %1108 = bitcast i32 %1107 to float, !dbg !8
  %1109 = select i1 %.not, float %1097, float %1108, !dbg !8
  %1110 = select i1 %.not, float %1108, float %1098, !dbg !8
  %1111 = select i1 %.not36, float %1109, float %1103, !dbg !8
  %1112 = bitcast float %1111 to i32, !dbg !8
  %1113 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1112), !dbg !8
  %1114 = bitcast i32 %1113 to float, !dbg !8
  %1115 = select i1 %.not36, float %1103, float %1114, !dbg !8
  %1116 = select i1 %.not36, float %1114, float %1109, !dbg !8
  %1117 = select i1 %.not36, float %1110, float %1104, !dbg !8
  %1118 = bitcast float %1117 to i32, !dbg !8
  %1119 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1118), !dbg !8
  %1120 = bitcast i32 %1119 to float, !dbg !8
  %1121 = select i1 %.not36, float %1104, float %1120, !dbg !8
  %1122 = select i1 %.not36, float %1120, float %1110, !dbg !8
  %1123 = extractelement <8 x float> %1094, i64 4, !dbg !8
  %1124 = extractelement <8 x float> %1094, i64 5, !dbg !8
  %1125 = extractelement <8 x float> %1094, i64 6, !dbg !8
  %1126 = extractelement <8 x float> %1094, i64 7, !dbg !8
  %1127 = select i1 %.not, float %1124, float %1123, !dbg !8
  %1128 = bitcast float %1127 to i32, !dbg !8
  %1129 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1128), !dbg !8
  %1130 = bitcast i32 %1129 to float, !dbg !8
  %1131 = select i1 %.not, float %1123, float %1130, !dbg !8
  %1132 = select i1 %.not, float %1130, float %1124, !dbg !8
  %1133 = select i1 %.not, float %1126, float %1125, !dbg !8
  %1134 = bitcast float %1133 to i32, !dbg !8
  %1135 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1134), !dbg !8
  %1136 = bitcast i32 %1135 to float, !dbg !8
  %1137 = select i1 %.not, float %1125, float %1136, !dbg !8
  %1138 = select i1 %.not, float %1136, float %1126, !dbg !8
  %1139 = select i1 %.not36, float %1137, float %1131, !dbg !8
  %1140 = bitcast float %1139 to i32, !dbg !8
  %1141 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1140), !dbg !8
  %1142 = bitcast i32 %1141 to float, !dbg !8
  %1143 = select i1 %.not36, float %1131, float %1142, !dbg !8
  %1144 = select i1 %.not36, float %1142, float %1137, !dbg !8
  %1145 = select i1 %.not36, float %1138, float %1132, !dbg !8
  %1146 = bitcast float %1145 to i32, !dbg !8
  %1147 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1146), !dbg !8
  %1148 = bitcast i32 %1147 to float, !dbg !8
  %1149 = select i1 %.not36, float %1132, float %1148, !dbg !8
  %1150 = select i1 %.not36, float %1148, float %1138, !dbg !8
  %1151 = extractvalue [2 x <8 x float>] %109, 1, !dbg !8
  %1152 = extractelement <8 x float> %1151, i64 0, !dbg !8
  %1153 = extractelement <8 x float> %1151, i64 1, !dbg !8
  %1154 = extractelement <8 x float> %1151, i64 2, !dbg !8
  %1155 = extractelement <8 x float> %1151, i64 3, !dbg !8
  %1156 = select i1 %.not, float %1153, float %1152, !dbg !8
  %1157 = bitcast float %1156 to i32, !dbg !8
  %1158 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1157), !dbg !8
  %1159 = bitcast i32 %1158 to float, !dbg !8
  %1160 = select i1 %.not, float %1152, float %1159, !dbg !8
  %1161 = select i1 %.not, float %1159, float %1153, !dbg !8
  %1162 = select i1 %.not, float %1155, float %1154, !dbg !8
  %1163 = bitcast float %1162 to i32, !dbg !8
  %1164 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1163), !dbg !8
  %1165 = bitcast i32 %1164 to float, !dbg !8
  %1166 = select i1 %.not, float %1154, float %1165, !dbg !8
  %1167 = select i1 %.not, float %1165, float %1155, !dbg !8
  %1168 = select i1 %.not36, float %1166, float %1160, !dbg !8
  %1169 = bitcast float %1168 to i32, !dbg !8
  %1170 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1169), !dbg !8
  %1171 = bitcast i32 %1170 to float, !dbg !8
  %1172 = select i1 %.not36, float %1160, float %1171, !dbg !8
  %1173 = select i1 %.not36, float %1171, float %1166, !dbg !8
  %1174 = select i1 %.not36, float %1167, float %1161, !dbg !8
  %1175 = bitcast float %1174 to i32, !dbg !8
  %1176 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1175), !dbg !8
  %1177 = bitcast i32 %1176 to float, !dbg !8
  %1178 = select i1 %.not36, float %1161, float %1177, !dbg !8
  %1179 = select i1 %.not36, float %1177, float %1167, !dbg !8
  %1180 = extractelement <8 x float> %1151, i64 4, !dbg !8
  %1181 = extractelement <8 x float> %1151, i64 5, !dbg !8
  %1182 = extractelement <8 x float> %1151, i64 6, !dbg !8
  %1183 = extractelement <8 x float> %1151, i64 7, !dbg !8
  %1184 = select i1 %.not, float %1181, float %1180, !dbg !8
  %1185 = bitcast float %1184 to i32, !dbg !8
  %1186 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1185), !dbg !8
  %1187 = bitcast i32 %1186 to float, !dbg !8
  %1188 = select i1 %.not, float %1180, float %1187, !dbg !8
  %1189 = select i1 %.not, float %1187, float %1181, !dbg !8
  %1190 = select i1 %.not, float %1183, float %1182, !dbg !8
  %1191 = bitcast float %1190 to i32, !dbg !8
  %1192 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1191), !dbg !8
  %1193 = bitcast i32 %1192 to float, !dbg !8
  %1194 = select i1 %.not, float %1182, float %1193, !dbg !8
  %1195 = select i1 %.not, float %1193, float %1183, !dbg !8
  %1196 = select i1 %.not36, float %1194, float %1188, !dbg !8
  %1197 = bitcast float %1196 to i32, !dbg !8
  %1198 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1197), !dbg !8
  %1199 = bitcast i32 %1198 to float, !dbg !8
  %1200 = select i1 %.not36, float %1188, float %1199, !dbg !8
  %1201 = select i1 %.not36, float %1199, float %1194, !dbg !8
  %1202 = select i1 %.not36, float %1195, float %1189, !dbg !8
  %1203 = bitcast float %1202 to i32, !dbg !8
  %1204 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1203), !dbg !8
  %1205 = bitcast i32 %1204 to float, !dbg !8
  %1206 = select i1 %.not36, float %1189, float %1205, !dbg !8
  %1207 = select i1 %.not36, float %1205, float %1195, !dbg !8
  %1208 = extractvalue [2 x <8 x float>] %108, 0, !dbg !8
  %1209 = extractelement <8 x float> %1208, i64 0, !dbg !8
  %1210 = extractelement <8 x float> %1208, i64 1, !dbg !8
  %1211 = extractelement <8 x float> %1208, i64 2, !dbg !8
  %1212 = extractelement <8 x float> %1208, i64 3, !dbg !8
  %1213 = select i1 %.not, float %1210, float %1209, !dbg !8
  %1214 = bitcast float %1213 to i32, !dbg !8
  %1215 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1214), !dbg !8
  %1216 = bitcast i32 %1215 to float, !dbg !8
  %1217 = select i1 %.not, float %1209, float %1216, !dbg !8
  %1218 = select i1 %.not, float %1216, float %1210, !dbg !8
  %1219 = select i1 %.not, float %1212, float %1211, !dbg !8
  %1220 = bitcast float %1219 to i32, !dbg !8
  %1221 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1220), !dbg !8
  %1222 = bitcast i32 %1221 to float, !dbg !8
  %1223 = select i1 %.not, float %1211, float %1222, !dbg !8
  %1224 = select i1 %.not, float %1222, float %1212, !dbg !8
  %1225 = select i1 %.not36, float %1223, float %1217, !dbg !8
  %1226 = bitcast float %1225 to i32, !dbg !8
  %1227 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1226), !dbg !8
  %1228 = bitcast i32 %1227 to float, !dbg !8
  %1229 = select i1 %.not36, float %1217, float %1228, !dbg !8
  %1230 = select i1 %.not36, float %1228, float %1223, !dbg !8
  %1231 = select i1 %.not36, float %1224, float %1218, !dbg !8
  %1232 = bitcast float %1231 to i32, !dbg !8
  %1233 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1232), !dbg !8
  %1234 = bitcast i32 %1233 to float, !dbg !8
  %1235 = select i1 %.not36, float %1218, float %1234, !dbg !8
  %1236 = select i1 %.not36, float %1234, float %1224, !dbg !8
  %1237 = extractelement <8 x float> %1208, i64 4, !dbg !8
  %1238 = extractelement <8 x float> %1208, i64 5, !dbg !8
  %1239 = extractelement <8 x float> %1208, i64 6, !dbg !8
  %1240 = extractelement <8 x float> %1208, i64 7, !dbg !8
  %1241 = select i1 %.not, float %1238, float %1237, !dbg !8
  %1242 = bitcast float %1241 to i32, !dbg !8
  %1243 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1242), !dbg !8
  %1244 = bitcast i32 %1243 to float, !dbg !8
  %1245 = select i1 %.not, float %1237, float %1244, !dbg !8
  %1246 = select i1 %.not, float %1244, float %1238, !dbg !8
  %1247 = select i1 %.not, float %1240, float %1239, !dbg !8
  %1248 = bitcast float %1247 to i32, !dbg !8
  %1249 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1248), !dbg !8
  %1250 = bitcast i32 %1249 to float, !dbg !8
  %1251 = select i1 %.not, float %1239, float %1250, !dbg !8
  %1252 = select i1 %.not, float %1250, float %1240, !dbg !8
  %1253 = select i1 %.not36, float %1251, float %1245, !dbg !8
  %1254 = bitcast float %1253 to i32, !dbg !8
  %1255 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1254), !dbg !8
  %1256 = bitcast i32 %1255 to float, !dbg !8
  %1257 = select i1 %.not36, float %1245, float %1256, !dbg !8
  %1258 = select i1 %.not36, float %1256, float %1251, !dbg !8
  %1259 = select i1 %.not36, float %1252, float %1246, !dbg !8
  %1260 = bitcast float %1259 to i32, !dbg !8
  %1261 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1260), !dbg !8
  %1262 = bitcast i32 %1261 to float, !dbg !8
  %1263 = select i1 %.not36, float %1246, float %1262, !dbg !8
  %1264 = select i1 %.not36, float %1262, float %1252, !dbg !8
  %1265 = extractvalue [2 x <8 x float>] %108, 1, !dbg !8
  %1266 = extractelement <8 x float> %1265, i64 0, !dbg !8
  %1267 = extractelement <8 x float> %1265, i64 1, !dbg !8
  %1268 = extractelement <8 x float> %1265, i64 2, !dbg !8
  %1269 = extractelement <8 x float> %1265, i64 3, !dbg !8
  %1270 = select i1 %.not, float %1267, float %1266, !dbg !8
  %1271 = bitcast float %1270 to i32, !dbg !8
  %1272 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1271), !dbg !8
  %1273 = bitcast i32 %1272 to float, !dbg !8
  %1274 = select i1 %.not, float %1266, float %1273, !dbg !8
  %1275 = select i1 %.not, float %1273, float %1267, !dbg !8
  %1276 = select i1 %.not, float %1269, float %1268, !dbg !8
  %1277 = bitcast float %1276 to i32, !dbg !8
  %1278 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1277), !dbg !8
  %1279 = bitcast i32 %1278 to float, !dbg !8
  %1280 = select i1 %.not, float %1268, float %1279, !dbg !8
  %1281 = select i1 %.not, float %1279, float %1269, !dbg !8
  %1282 = select i1 %.not36, float %1280, float %1274, !dbg !8
  %1283 = bitcast float %1282 to i32, !dbg !8
  %1284 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1283), !dbg !8
  %1285 = bitcast i32 %1284 to float, !dbg !8
  %1286 = select i1 %.not36, float %1274, float %1285, !dbg !8
  %1287 = select i1 %.not36, float %1285, float %1280, !dbg !8
  %1288 = select i1 %.not36, float %1281, float %1275, !dbg !8
  %1289 = bitcast float %1288 to i32, !dbg !8
  %1290 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1289), !dbg !8
  %1291 = bitcast i32 %1290 to float, !dbg !8
  %1292 = select i1 %.not36, float %1275, float %1291, !dbg !8
  %1293 = select i1 %.not36, float %1291, float %1281, !dbg !8
  %1294 = extractelement <8 x float> %1265, i64 4, !dbg !8
  %1295 = extractelement <8 x float> %1265, i64 5, !dbg !8
  %1296 = extractelement <8 x float> %1265, i64 6, !dbg !8
  %1297 = extractelement <8 x float> %1265, i64 7, !dbg !8
  %1298 = select i1 %.not, float %1295, float %1294, !dbg !8
  %1299 = bitcast float %1298 to i32, !dbg !8
  %1300 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1299), !dbg !8
  %1301 = bitcast i32 %1300 to float, !dbg !8
  %1302 = select i1 %.not, float %1294, float %1301, !dbg !8
  %1303 = select i1 %.not, float %1301, float %1295, !dbg !8
  %1304 = select i1 %.not, float %1297, float %1296, !dbg !8
  %1305 = bitcast float %1304 to i32, !dbg !8
  %1306 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1305), !dbg !8
  %1307 = bitcast i32 %1306 to float, !dbg !8
  %1308 = select i1 %.not, float %1296, float %1307, !dbg !8
  %1309 = select i1 %.not, float %1307, float %1297, !dbg !8
  %1310 = select i1 %.not36, float %1308, float %1302, !dbg !8
  %1311 = bitcast float %1310 to i32, !dbg !8
  %1312 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1311), !dbg !8
  %1313 = bitcast i32 %1312 to float, !dbg !8
  %1314 = select i1 %.not36, float %1302, float %1313, !dbg !8
  %1315 = select i1 %.not36, float %1313, float %1308, !dbg !8
  %1316 = select i1 %.not36, float %1309, float %1303, !dbg !8
  %1317 = bitcast float %1316 to i32, !dbg !8
  %1318 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1317), !dbg !8
  %1319 = bitcast i32 %1318 to float, !dbg !8
  %1320 = select i1 %.not36, float %1303, float %1319, !dbg !8
  %1321 = select i1 %.not36, float %1319, float %1309, !dbg !8
  %1322 = extractvalue [2 x <8 x float>] %105, 0, !dbg !8
  %1323 = extractelement <8 x float> %1322, i64 0, !dbg !8
  %1324 = extractelement <8 x float> %1322, i64 1, !dbg !8
  %1325 = extractelement <8 x float> %1322, i64 2, !dbg !8
  %1326 = extractelement <8 x float> %1322, i64 3, !dbg !8
  %1327 = select i1 %.not, float %1324, float %1323, !dbg !8
  %1328 = bitcast float %1327 to i32, !dbg !8
  %1329 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1328), !dbg !8
  %1330 = bitcast i32 %1329 to float, !dbg !8
  %1331 = select i1 %.not, float %1323, float %1330, !dbg !8
  %1332 = select i1 %.not, float %1330, float %1324, !dbg !8
  %1333 = select i1 %.not, float %1326, float %1325, !dbg !8
  %1334 = bitcast float %1333 to i32, !dbg !8
  %1335 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1334), !dbg !8
  %1336 = bitcast i32 %1335 to float, !dbg !8
  %1337 = select i1 %.not, float %1325, float %1336, !dbg !8
  %1338 = select i1 %.not, float %1336, float %1326, !dbg !8
  %1339 = select i1 %.not36, float %1337, float %1331, !dbg !8
  %1340 = bitcast float %1339 to i32, !dbg !8
  %1341 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1340), !dbg !8
  %1342 = select i1 %.not36, float %1338, float %1332, !dbg !8
  %1343 = bitcast float %1342 to i32, !dbg !8
  %1344 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1343), !dbg !8
  %1345 = extractelement <8 x float> %1322, i64 4, !dbg !8
  %1346 = extractelement <8 x float> %1322, i64 5, !dbg !8
  %1347 = extractelement <8 x float> %1322, i64 6, !dbg !8
  %1348 = extractelement <8 x float> %1322, i64 7, !dbg !8
  %1349 = select i1 %.not, float %1346, float %1345, !dbg !8
  %1350 = bitcast float %1349 to i32, !dbg !8
  %1351 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1350), !dbg !8
  %1352 = bitcast i32 %1351 to float, !dbg !8
  %1353 = select i1 %.not, float %1345, float %1352, !dbg !8
  %1354 = select i1 %.not, float %1352, float %1346, !dbg !8
  %1355 = select i1 %.not, float %1348, float %1347, !dbg !8
  %1356 = bitcast float %1355 to i32, !dbg !8
  %1357 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1356), !dbg !8
  %1358 = bitcast i32 %1357 to float, !dbg !8
  %1359 = select i1 %.not, float %1347, float %1358, !dbg !8
  %1360 = select i1 %.not, float %1358, float %1348, !dbg !8
  %1361 = select i1 %.not36, float %1359, float %1353, !dbg !8
  %1362 = bitcast float %1361 to i32, !dbg !8
  %1363 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1362), !dbg !8
  %1364 = select i1 %.not36, float %1360, float %1354, !dbg !8
  %1365 = bitcast float %1364 to i32, !dbg !8
  %1366 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1365), !dbg !8
  %1367 = extractvalue [2 x <8 x float>] %105, 1, !dbg !8
  %1368 = extractelement <8 x float> %1367, i64 0, !dbg !8
  %1369 = extractelement <8 x float> %1367, i64 1, !dbg !8
  %1370 = extractelement <8 x float> %1367, i64 2, !dbg !8
  %1371 = extractelement <8 x float> %1367, i64 3, !dbg !8
  %1372 = select i1 %.not, float %1369, float %1368, !dbg !8
  %1373 = bitcast float %1372 to i32, !dbg !8
  %1374 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1373), !dbg !8
  %1375 = bitcast i32 %1374 to float, !dbg !8
  %1376 = select i1 %.not, float %1368, float %1375, !dbg !8
  %1377 = select i1 %.not, float %1375, float %1369, !dbg !8
  %1378 = select i1 %.not, float %1371, float %1370, !dbg !8
  %1379 = bitcast float %1378 to i32, !dbg !8
  %1380 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1379), !dbg !8
  %1381 = bitcast i32 %1380 to float, !dbg !8
  %1382 = select i1 %.not, float %1370, float %1381, !dbg !8
  %1383 = select i1 %.not, float %1381, float %1371, !dbg !8
  %1384 = select i1 %.not36, float %1382, float %1376, !dbg !8
  %1385 = bitcast float %1384 to i32, !dbg !8
  %1386 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1385), !dbg !8
  %1387 = select i1 %.not36, float %1383, float %1377, !dbg !8
  %1388 = bitcast float %1387 to i32, !dbg !8
  %1389 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1388), !dbg !8
  %1390 = extractelement <8 x float> %1367, i64 4, !dbg !8
  %1391 = extractelement <8 x float> %1367, i64 5, !dbg !8
  %1392 = extractelement <8 x float> %1367, i64 6, !dbg !8
  %1393 = extractelement <8 x float> %1367, i64 7, !dbg !8
  %1394 = select i1 %.not, float %1391, float %1390, !dbg !8
  %1395 = bitcast float %1394 to i32, !dbg !8
  %1396 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1395), !dbg !8
  %1397 = bitcast i32 %1396 to float, !dbg !8
  %1398 = select i1 %.not, float %1390, float %1397, !dbg !8
  %1399 = select i1 %.not, float %1397, float %1391, !dbg !8
  %1400 = select i1 %.not, float %1393, float %1392, !dbg !8
  %1401 = bitcast float %1400 to i32, !dbg !8
  %1402 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1401), !dbg !8
  %1403 = bitcast i32 %1402 to float, !dbg !8
  %1404 = select i1 %.not, float %1392, float %1403, !dbg !8
  %1405 = select i1 %.not, float %1403, float %1393, !dbg !8
  %1406 = select i1 %.not36, float %1404, float %1398, !dbg !8
  %1407 = bitcast float %1406 to i32, !dbg !8
  %1408 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1407), !dbg !8
  %1409 = select i1 %.not36, float %1405, float %1399, !dbg !8
  %1410 = bitcast float %1409 to i32, !dbg !8
  %1411 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1410), !dbg !8
  %1412 = shufflevector <1 x float> %977, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !17
  %1413 = shufflevector <1 x float> %972, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !17
  %1414 = bitcast i32 %1408 to float, !dbg !8
  %1415 = select i1 %.not36, float %1398, float %1414, !dbg !8
  %1416 = insertelement <4 x float> poison, float %1415, i64 0, !dbg !17
  %1417 = bitcast i32 %1411 to float, !dbg !8
  %1418 = select i1 %.not36, float %1399, float %1417, !dbg !8
  %1419 = insertelement <4 x float> %1416, float %1418, i64 1, !dbg !17
  %1420 = select i1 %.not36, float %1414, float %1404, !dbg !8
  %1421 = insertelement <4 x float> %1419, float %1420, i64 2, !dbg !17
  %1422 = select i1 %.not36, float %1417, float %1405, !dbg !8
  %1423 = insertelement <4 x float> %1421, float %1422, i64 3, !dbg !17
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1413, !dbg !17
  %normalized = fmul <4 x float> %1423, %row.reciprocal, !dbg !17
  %1424 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !18
  %1425 = extractelement <4 x half> %1424, i64 3, !dbg !18
  %1426 = extractelement <4 x half> %1424, i64 2, !dbg !18
  %1427 = extractelement <4 x half> %1424, i64 1, !dbg !18
  %1428 = extractelement <4 x half> %1424, i64 0, !dbg !18
  %1429 = bitcast i32 %1386 to float, !dbg !8
  %1430 = select i1 %.not36, float %1376, float %1429, !dbg !8
  %1431 = insertelement <4 x float> poison, float %1430, i64 0, !dbg !17
  %1432 = bitcast i32 %1389 to float, !dbg !8
  %1433 = select i1 %.not36, float %1377, float %1432, !dbg !8
  %1434 = insertelement <4 x float> %1431, float %1433, i64 1, !dbg !17
  %1435 = select i1 %.not36, float %1429, float %1382, !dbg !8
  %1436 = insertelement <4 x float> %1434, float %1435, i64 2, !dbg !17
  %1437 = select i1 %.not36, float %1432, float %1383, !dbg !8
  %1438 = insertelement <4 x float> %1436, float %1437, i64 3, !dbg !17
  %normalized37 = fmul <4 x float> %1438, %row.reciprocal, !dbg !17
  %1439 = fptrunc <4 x float> %normalized37 to <4 x half>, !dbg !18
  %1440 = extractelement <4 x half> %1439, i64 3, !dbg !18
  %1441 = extractelement <4 x half> %1439, i64 2, !dbg !18
  %1442 = extractelement <4 x half> %1439, i64 1, !dbg !18
  %1443 = extractelement <4 x half> %1439, i64 0, !dbg !18
  %1444 = bitcast i32 %1363 to float, !dbg !8
  %1445 = select i1 %.not36, float %1353, float %1444, !dbg !8
  %1446 = insertelement <4 x float> poison, float %1445, i64 0, !dbg !17
  %1447 = bitcast i32 %1366 to float, !dbg !8
  %1448 = select i1 %.not36, float %1354, float %1447, !dbg !8
  %1449 = insertelement <4 x float> %1446, float %1448, i64 1, !dbg !17
  %1450 = select i1 %.not36, float %1444, float %1359, !dbg !8
  %1451 = insertelement <4 x float> %1449, float %1450, i64 2, !dbg !17
  %1452 = select i1 %.not36, float %1447, float %1360, !dbg !8
  %1453 = insertelement <4 x float> %1451, float %1452, i64 3, !dbg !17
  %row.reciprocal38 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1412, !dbg !17
  %normalized39 = fmul <4 x float> %1453, %row.reciprocal38, !dbg !17
  %1454 = fptrunc <4 x float> %normalized39 to <4 x half>, !dbg !18
  %1455 = extractelement <4 x half> %1454, i64 3, !dbg !18
  %1456 = extractelement <4 x half> %1454, i64 2, !dbg !18
  %1457 = extractelement <4 x half> %1454, i64 1, !dbg !18
  %1458 = extractelement <4 x half> %1454, i64 0, !dbg !18
  %1459 = bitcast i32 %1341 to float, !dbg !8
  %1460 = select i1 %.not36, float %1331, float %1459, !dbg !8
  %1461 = insertelement <4 x float> poison, float %1460, i64 0, !dbg !17
  %1462 = bitcast i32 %1344 to float, !dbg !8
  %1463 = select i1 %.not36, float %1332, float %1462, !dbg !8
  %1464 = insertelement <4 x float> %1461, float %1463, i64 1, !dbg !17
  %1465 = select i1 %.not36, float %1459, float %1337, !dbg !8
  %1466 = insertelement <4 x float> %1464, float %1465, i64 2, !dbg !17
  %1467 = select i1 %.not36, float %1462, float %1338, !dbg !8
  %1468 = insertelement <4 x float> %1466, float %1467, i64 3, !dbg !17
  %normalized40 = fmul <4 x float> %1468, %row.reciprocal38, !dbg !17
  %1469 = fptrunc <4 x float> %normalized40 to <4 x half>, !dbg !18
  %1470 = extractelement <4 x half> %1469, i64 3, !dbg !18
  %1471 = extractelement <4 x half> %1469, i64 2, !dbg !18
  %1472 = extractelement <4 x half> %1469, i64 1, !dbg !18
  %1473 = extractelement <4 x half> %1469, i64 0, !dbg !18
  %address.thread958 = call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #9, !dbg !19, !range !32, !frisk.register_lifetimes !1
  %address.local959 = zext nneg i32 %address.thread958 to i64, !dbg !9
  %address.local960 = add nuw i64 %32, %address.local959, !dbg !19
  %1474 = lshr i64 %address.local959, 2, !dbg !19
  %address.local962 = and i64 %1474, 48, !dbg !19
  %address.local963 = add i64 %address.local960, %address.local962, !dbg !19
  %address.local964 = and i64 %address.local959, 240, !dbg !19
  %address.local965 = sub i64 %address.local963, %address.local964, !dbg !19
  %address.local966 = add i64 %address.local965, %33, !dbg !19
  %address.local967 = shl nuw nsw i64 %address.local966, 7, !dbg !19
  %1475 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %46, !dbg !19
  %address.local969 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %1475, i64 %address.local967, !dbg !19
  %address.local971 = and i64 %1474, 12, !dbg !12
  %address.local972 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local971, !dbg !19
  store half %1473, ptr addrspace(1) %address.local972, align 2, !dbg !19
  %address.local974 = or disjoint i64 %address.local971, 1, !dbg !19
  %address.local975 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local974, !dbg !19
  store half %1472, ptr addrspace(1) %address.local975, align 2, !dbg !19
  %address.local977 = or disjoint i64 %address.local971, 2, !dbg !19
  %address.local978 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local977, !dbg !19
  store half %1471, ptr addrspace(1) %address.local978, align 2, !dbg !19
  %address.local980 = or disjoint i64 %address.local971, 3, !dbg !19
  %address.local981 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local980, !dbg !19
  store half %1470, ptr addrspace(1) %address.local981, align 2, !dbg !19
  %address.local983 = or disjoint i64 %address.local971, 16, !dbg !19
  %address.local984 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local983, !dbg !19
  store half %1458, ptr addrspace(1) %address.local984, align 2, !dbg !19
  %address.local986 = or disjoint i64 %address.local971, 17, !dbg !19
  %address.local987 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local986, !dbg !19
  store half %1457, ptr addrspace(1) %address.local987, align 2, !dbg !19
  %address.local989 = or disjoint i64 %address.local971, 18, !dbg !19
  %address.local990 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local989, !dbg !19
  store half %1456, ptr addrspace(1) %address.local990, align 2, !dbg !19
  %address.local992 = or disjoint i64 %address.local971, 19, !dbg !19
  %address.local993 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local992, !dbg !19
  store half %1455, ptr addrspace(1) %address.local993, align 2, !dbg !19
  %address.local997 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local969, i64 16384, !dbg !19
  %address.local998 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local971, !dbg !19
  store half %1443, ptr addrspace(1) %address.local998, align 2, !dbg !19
  %address.local1000 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local974, !dbg !19
  store half %1442, ptr addrspace(1) %address.local1000, align 2, !dbg !19
  %address.local1002 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local977, !dbg !19
  store half %1441, ptr addrspace(1) %address.local1002, align 2, !dbg !19
  %address.local1004 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local980, !dbg !19
  store half %1440, ptr addrspace(1) %address.local1004, align 2, !dbg !19
  %address.local1006 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local983, !dbg !19
  store half %1428, ptr addrspace(1) %address.local1006, align 2, !dbg !19
  %address.local1008 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local986, !dbg !19
  store half %1427, ptr addrspace(1) %address.local1008, align 2, !dbg !19
  %address.local1010 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local989, !dbg !19
  store half %1426, ptr addrspace(1) %address.local1010, align 2, !dbg !19
  %address.local1012 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local992, !dbg !19
  store half %1425, ptr addrspace(1) %address.local1012, align 2, !dbg !19
  %1476 = insertelement <4 x float> poison, float %1314, i64 0, !dbg !17
  %1477 = insertelement <4 x float> %1476, float %1320, i64 1, !dbg !17
  %1478 = insertelement <4 x float> %1477, float %1315, i64 2, !dbg !17
  %1479 = insertelement <4 x float> %1478, float %1321, i64 3, !dbg !17
  %normalized42 = fmul <4 x float> %1479, %row.reciprocal, !dbg !17
  %1480 = fptrunc <4 x float> %normalized42 to <4 x half>, !dbg !18
  %1481 = extractelement <4 x half> %1480, i64 3, !dbg !18
  %1482 = extractelement <4 x half> %1480, i64 2, !dbg !18
  %1483 = extractelement <4 x half> %1480, i64 1, !dbg !18
  %1484 = extractelement <4 x half> %1480, i64 0, !dbg !18
  %1485 = insertelement <4 x float> poison, float %1286, i64 0, !dbg !17
  %1486 = insertelement <4 x float> %1485, float %1292, i64 1, !dbg !17
  %1487 = insertelement <4 x float> %1486, float %1287, i64 2, !dbg !17
  %1488 = insertelement <4 x float> %1487, float %1293, i64 3, !dbg !17
  %normalized43 = fmul <4 x float> %1488, %row.reciprocal, !dbg !17
  %1489 = fptrunc <4 x float> %normalized43 to <4 x half>, !dbg !18
  %1490 = extractelement <4 x half> %1489, i64 3, !dbg !18
  %1491 = extractelement <4 x half> %1489, i64 2, !dbg !18
  %1492 = extractelement <4 x half> %1489, i64 1, !dbg !18
  %1493 = extractelement <4 x half> %1489, i64 0, !dbg !18
  %1494 = insertelement <4 x float> poison, float %1257, i64 0, !dbg !17
  %1495 = insertelement <4 x float> %1494, float %1263, i64 1, !dbg !17
  %1496 = insertelement <4 x float> %1495, float %1258, i64 2, !dbg !17
  %1497 = insertelement <4 x float> %1496, float %1264, i64 3, !dbg !17
  %normalized45 = fmul <4 x float> %1497, %row.reciprocal38, !dbg !17
  %1498 = fptrunc <4 x float> %normalized45 to <4 x half>, !dbg !18
  %1499 = extractelement <4 x half> %1498, i64 3, !dbg !18
  %1500 = extractelement <4 x half> %1498, i64 2, !dbg !18
  %1501 = extractelement <4 x half> %1498, i64 1, !dbg !18
  %1502 = extractelement <4 x half> %1498, i64 0, !dbg !18
  %1503 = insertelement <4 x float> poison, float %1229, i64 0, !dbg !17
  %1504 = insertelement <4 x float> %1503, float %1235, i64 1, !dbg !17
  %1505 = insertelement <4 x float> %1504, float %1230, i64 2, !dbg !17
  %1506 = insertelement <4 x float> %1505, float %1236, i64 3, !dbg !17
  %normalized46 = fmul <4 x float> %1506, %row.reciprocal38, !dbg !17
  %1507 = fptrunc <4 x float> %normalized46 to <4 x half>, !dbg !18
  %1508 = extractelement <4 x half> %1507, i64 3, !dbg !18
  %1509 = extractelement <4 x half> %1507, i64 2, !dbg !18
  %1510 = extractelement <4 x half> %1507, i64 1, !dbg !18
  %1511 = extractelement <4 x half> %1507, i64 0, !dbg !18
  %address.local1027 = or disjoint i64 %address.local971, 32, !dbg !19
  %address.local1028 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1027, !dbg !19
  store half %1511, ptr addrspace(1) %address.local1028, align 2, !dbg !19
  %address.local1030 = or disjoint i64 %address.local971, 33, !dbg !19
  %address.local1031 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1030, !dbg !19
  store half %1510, ptr addrspace(1) %address.local1031, align 2, !dbg !19
  %address.local1033 = or disjoint i64 %address.local971, 34, !dbg !19
  %address.local1034 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1033, !dbg !19
  store half %1509, ptr addrspace(1) %address.local1034, align 2, !dbg !19
  %address.local1036 = or disjoint i64 %address.local971, 35, !dbg !19
  %address.local1037 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1036, !dbg !19
  store half %1508, ptr addrspace(1) %address.local1037, align 2, !dbg !19
  %address.local1039 = or disjoint i64 %address.local971, 48, !dbg !19
  %address.local1040 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1039, !dbg !19
  store half %1502, ptr addrspace(1) %address.local1040, align 2, !dbg !19
  %address.local1042 = or disjoint i64 %address.local971, 49, !dbg !19
  %address.local1043 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1042, !dbg !19
  store half %1501, ptr addrspace(1) %address.local1043, align 2, !dbg !19
  %address.local1045 = or disjoint i64 %address.local971, 50, !dbg !19
  %address.local1046 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1045, !dbg !19
  store half %1500, ptr addrspace(1) %address.local1046, align 2, !dbg !19
  %address.local1048 = or i64 %1474, 51, !dbg !19
  %address.local1049 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1048, !dbg !19
  store half %1499, ptr addrspace(1) %address.local1049, align 2, !dbg !19
  %address.local1054 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1027, !dbg !19
  store half %1493, ptr addrspace(1) %address.local1054, align 2, !dbg !19
  %address.local1056 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1030, !dbg !19
  store half %1492, ptr addrspace(1) %address.local1056, align 2, !dbg !19
  %address.local1058 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1033, !dbg !19
  store half %1491, ptr addrspace(1) %address.local1058, align 2, !dbg !19
  %address.local1060 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1036, !dbg !19
  store half %1490, ptr addrspace(1) %address.local1060, align 2, !dbg !19
  %address.local1062 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1039, !dbg !19
  store half %1484, ptr addrspace(1) %address.local1062, align 2, !dbg !19
  %address.local1064 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1042, !dbg !19
  store half %1483, ptr addrspace(1) %address.local1064, align 2, !dbg !19
  %address.local1066 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1045, !dbg !19
  store half %1482, ptr addrspace(1) %address.local1066, align 2, !dbg !19
  %address.local1068 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1048, !dbg !19
  store half %1481, ptr addrspace(1) %address.local1068, align 2, !dbg !19
  %1512 = insertelement <4 x float> poison, float %1200, i64 0, !dbg !17
  %1513 = insertelement <4 x float> %1512, float %1206, i64 1, !dbg !17
  %1514 = insertelement <4 x float> %1513, float %1201, i64 2, !dbg !17
  %1515 = insertelement <4 x float> %1514, float %1207, i64 3, !dbg !17
  %normalized48 = fmul <4 x float> %1515, %row.reciprocal, !dbg !17
  %1516 = fptrunc <4 x float> %normalized48 to <4 x half>, !dbg !18
  %1517 = extractelement <4 x half> %1516, i64 3, !dbg !18
  %1518 = extractelement <4 x half> %1516, i64 2, !dbg !18
  %1519 = extractelement <4 x half> %1516, i64 1, !dbg !18
  %1520 = extractelement <4 x half> %1516, i64 0, !dbg !18
  %1521 = insertelement <4 x float> poison, float %1172, i64 0, !dbg !17
  %1522 = insertelement <4 x float> %1521, float %1178, i64 1, !dbg !17
  %1523 = insertelement <4 x float> %1522, float %1173, i64 2, !dbg !17
  %1524 = insertelement <4 x float> %1523, float %1179, i64 3, !dbg !17
  %normalized49 = fmul <4 x float> %1524, %row.reciprocal, !dbg !17
  %1525 = fptrunc <4 x float> %normalized49 to <4 x half>, !dbg !18
  %1526 = extractelement <4 x half> %1525, i64 3, !dbg !18
  %1527 = extractelement <4 x half> %1525, i64 2, !dbg !18
  %1528 = extractelement <4 x half> %1525, i64 1, !dbg !18
  %1529 = extractelement <4 x half> %1525, i64 0, !dbg !18
  %1530 = insertelement <4 x float> poison, float %1143, i64 0, !dbg !17
  %1531 = insertelement <4 x float> %1530, float %1149, i64 1, !dbg !17
  %1532 = insertelement <4 x float> %1531, float %1144, i64 2, !dbg !17
  %1533 = insertelement <4 x float> %1532, float %1150, i64 3, !dbg !17
  %normalized51 = fmul <4 x float> %1533, %row.reciprocal38, !dbg !17
  %1534 = fptrunc <4 x float> %normalized51 to <4 x half>, !dbg !18
  %1535 = extractelement <4 x half> %1534, i64 3, !dbg !18
  %1536 = extractelement <4 x half> %1534, i64 2, !dbg !18
  %1537 = extractelement <4 x half> %1534, i64 1, !dbg !18
  %1538 = extractelement <4 x half> %1534, i64 0, !dbg !18
  %1539 = insertelement <4 x float> poison, float %1115, i64 0, !dbg !17
  %1540 = insertelement <4 x float> %1539, float %1121, i64 1, !dbg !17
  %1541 = insertelement <4 x float> %1540, float %1116, i64 2, !dbg !17
  %1542 = insertelement <4 x float> %1541, float %1122, i64 3, !dbg !17
  %normalized52 = fmul <4 x float> %1542, %row.reciprocal38, !dbg !17
  %1543 = fptrunc <4 x float> %normalized52 to <4 x half>, !dbg !18
  %1544 = extractelement <4 x half> %1543, i64 3, !dbg !18
  %1545 = extractelement <4 x half> %1543, i64 2, !dbg !18
  %1546 = extractelement <4 x half> %1543, i64 1, !dbg !18
  %1547 = extractelement <4 x half> %1543, i64 0, !dbg !18
  %address.local1083 = or disjoint i64 %address.local971, 64, !dbg !19
  %address.local1084 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1083, !dbg !19
  store half %1547, ptr addrspace(1) %address.local1084, align 2, !dbg !19
  %address.local1086 = or disjoint i64 %address.local971, 65, !dbg !19
  %address.local1087 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1086, !dbg !19
  store half %1546, ptr addrspace(1) %address.local1087, align 2, !dbg !19
  %address.local1089 = or disjoint i64 %address.local971, 66, !dbg !19
  %address.local1090 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1089, !dbg !19
  store half %1545, ptr addrspace(1) %address.local1090, align 2, !dbg !19
  %address.local1092 = or disjoint i64 %address.local971, 67, !dbg !19
  %address.local1093 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1092, !dbg !19
  store half %1544, ptr addrspace(1) %address.local1093, align 2, !dbg !19
  %address.local1095 = or disjoint i64 %address.local971, 80, !dbg !19
  %address.local1096 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1095, !dbg !19
  store half %1538, ptr addrspace(1) %address.local1096, align 2, !dbg !19
  %address.local1098 = or disjoint i64 %address.local971, 81, !dbg !19
  %address.local1099 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1098, !dbg !19
  store half %1537, ptr addrspace(1) %address.local1099, align 2, !dbg !19
  %address.local1101 = or disjoint i64 %address.local971, 82, !dbg !19
  %address.local1102 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1101, !dbg !19
  store half %1536, ptr addrspace(1) %address.local1102, align 2, !dbg !19
  %address.local1104 = or disjoint i64 %address.local971, 83, !dbg !19
  %address.local1105 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1104, !dbg !19
  store half %1535, ptr addrspace(1) %address.local1105, align 2, !dbg !19
  %address.local1110 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1083, !dbg !19
  store half %1529, ptr addrspace(1) %address.local1110, align 2, !dbg !19
  %address.local1112 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1086, !dbg !19
  store half %1528, ptr addrspace(1) %address.local1112, align 2, !dbg !19
  %address.local1114 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1089, !dbg !19
  store half %1527, ptr addrspace(1) %address.local1114, align 2, !dbg !19
  %address.local1116 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1092, !dbg !19
  store half %1526, ptr addrspace(1) %address.local1116, align 2, !dbg !19
  %address.local1118 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1095, !dbg !19
  store half %1520, ptr addrspace(1) %address.local1118, align 2, !dbg !19
  %address.local1120 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1098, !dbg !19
  store half %1519, ptr addrspace(1) %address.local1120, align 2, !dbg !19
  %address.local1122 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1101, !dbg !19
  store half %1518, ptr addrspace(1) %address.local1122, align 2, !dbg !19
  %address.local1124 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1104, !dbg !19
  store half %1517, ptr addrspace(1) %address.local1124, align 2, !dbg !19
  %1548 = insertelement <4 x float> poison, float %1086, i64 0, !dbg !17
  %1549 = insertelement <4 x float> %1548, float %1092, i64 1, !dbg !17
  %1550 = insertelement <4 x float> %1549, float %1087, i64 2, !dbg !17
  %1551 = insertelement <4 x float> %1550, float %1093, i64 3, !dbg !17
  %normalized54 = fmul <4 x float> %1551, %row.reciprocal, !dbg !17
  %1552 = fptrunc <4 x float> %normalized54 to <4 x half>, !dbg !18
  %1553 = extractelement <4 x half> %1552, i64 3, !dbg !18
  %1554 = extractelement <4 x half> %1552, i64 2, !dbg !18
  %1555 = extractelement <4 x half> %1552, i64 1, !dbg !18
  %1556 = extractelement <4 x half> %1552, i64 0, !dbg !18
  %1557 = insertelement <4 x float> poison, float %1058, i64 0, !dbg !17
  %1558 = insertelement <4 x float> %1557, float %1064, i64 1, !dbg !17
  %1559 = insertelement <4 x float> %1558, float %1059, i64 2, !dbg !17
  %1560 = insertelement <4 x float> %1559, float %1065, i64 3, !dbg !17
  %normalized55 = fmul <4 x float> %1560, %row.reciprocal, !dbg !17
  %1561 = fptrunc <4 x float> %normalized55 to <4 x half>, !dbg !18
  %1562 = extractelement <4 x half> %1561, i64 3, !dbg !18
  %1563 = extractelement <4 x half> %1561, i64 2, !dbg !18
  %1564 = extractelement <4 x half> %1561, i64 1, !dbg !18
  %1565 = extractelement <4 x half> %1561, i64 0, !dbg !18
  %1566 = insertelement <4 x float> poison, float %1029, i64 0, !dbg !17
  %1567 = insertelement <4 x float> %1566, float %1035, i64 1, !dbg !17
  %1568 = insertelement <4 x float> %1567, float %1030, i64 2, !dbg !17
  %1569 = insertelement <4 x float> %1568, float %1036, i64 3, !dbg !17
  %normalized57 = fmul <4 x float> %1569, %row.reciprocal38, !dbg !17
  %1570 = fptrunc <4 x float> %normalized57 to <4 x half>, !dbg !18
  %1571 = extractelement <4 x half> %1570, i64 3, !dbg !18
  %1572 = extractelement <4 x half> %1570, i64 2, !dbg !18
  %1573 = extractelement <4 x half> %1570, i64 1, !dbg !18
  %1574 = extractelement <4 x half> %1570, i64 0, !dbg !18
  %1575 = insertelement <4 x float> poison, float %1001, i64 0, !dbg !17
  %1576 = insertelement <4 x float> %1575, float %1007, i64 1, !dbg !17
  %1577 = insertelement <4 x float> %1576, float %1002, i64 2, !dbg !17
  %1578 = insertelement <4 x float> %1577, float %1008, i64 3, !dbg !17
  %normalized58 = fmul <4 x float> %1578, %row.reciprocal38, !dbg !17
  %1579 = fptrunc <4 x float> %normalized58 to <4 x half>, !dbg !18
  %1580 = extractelement <4 x half> %1579, i64 3, !dbg !18
  %1581 = extractelement <4 x half> %1579, i64 2, !dbg !18
  %1582 = extractelement <4 x half> %1579, i64 1, !dbg !18
  %1583 = extractelement <4 x half> %1579, i64 0, !dbg !18
  %address.local1139 = or disjoint i64 %address.local971, 96, !dbg !19
  %address.local1140 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1139, !dbg !19
  store half %1583, ptr addrspace(1) %address.local1140, align 2, !dbg !19
  %address.local1142 = or disjoint i64 %address.local971, 97, !dbg !19
  %address.local1143 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1142, !dbg !19
  store half %1582, ptr addrspace(1) %address.local1143, align 2, !dbg !19
  %address.local1145 = or disjoint i64 %address.local971, 98, !dbg !19
  %address.local1146 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1145, !dbg !19
  store half %1581, ptr addrspace(1) %address.local1146, align 2, !dbg !19
  %address.local1148 = or disjoint i64 %address.local971, 99, !dbg !19
  %address.local1149 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1148, !dbg !19
  store half %1580, ptr addrspace(1) %address.local1149, align 2, !dbg !19
  %address.local1151 = or disjoint i64 %address.local971, 112, !dbg !19
  %address.local1152 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1151, !dbg !19
  store half %1574, ptr addrspace(1) %address.local1152, align 2, !dbg !19
  %address.local1154 = or disjoint i64 %address.local971, 113, !dbg !19
  %address.local1155 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1154, !dbg !19
  store half %1573, ptr addrspace(1) %address.local1155, align 2, !dbg !19
  %address.local1157 = or disjoint i64 %address.local971, 114, !dbg !19
  %address.local1158 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1157, !dbg !19
  store half %1572, ptr addrspace(1) %address.local1158, align 2, !dbg !19
  %address.local1160 = or i64 %1474, 115, !dbg !19
  %address.local1161 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1160, !dbg !19
  store half %1571, ptr addrspace(1) %address.local1161, align 2, !dbg !19
  %address.local1166 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1139, !dbg !19
  store half %1565, ptr addrspace(1) %address.local1166, align 2, !dbg !19
  %address.local1168 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1142, !dbg !19
  store half %1564, ptr addrspace(1) %address.local1168, align 2, !dbg !19
  %address.local1170 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1145, !dbg !19
  store half %1563, ptr addrspace(1) %address.local1170, align 2, !dbg !19
  %address.local1172 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1148, !dbg !19
  store half %1562, ptr addrspace(1) %address.local1172, align 2, !dbg !19
  %address.local1174 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1151, !dbg !19
  store half %1556, ptr addrspace(1) %address.local1174, align 2, !dbg !19
  %address.local1176 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1154, !dbg !19
  store half %1555, ptr addrspace(1) %address.local1176, align 2, !dbg !19
  %address.local1178 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1157, !dbg !19
  store half %1554, ptr addrspace(1) %address.local1178, align 2, !dbg !19
  %address.local1180 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1160, !dbg !19
  store half %1553, ptr addrspace(1) %address.local1180, align 2, !dbg !19
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
