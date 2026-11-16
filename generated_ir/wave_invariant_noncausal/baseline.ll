; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/async_task_noncausal/input.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/async_task_noncausal/input.mlir"

@shm_0 = addrspace(3) global [16384 x i8] undef, align 16

; Function Attrs: convergent willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

declare float @__ocml_exp2_f32(float)

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3) #1 !dbg !4 !reqd_work_group_size !30 {
  %kdma.tid = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !5
  %kdma.wave = and i32 %kdma.tid, 960, !dbg !5
  %kdma.lane = lshr i32 %kdma.tid, 1, !dbg !5
  %5 = and i32 %kdma.lane, 31, !dbg !5
  %kdma.owner = or disjoint i32 %kdma.wave, %5, !dbg !5
  %kdma.owner.hi = or disjoint i32 %kdma.owner, 32, !dbg !5
  %6 = shl nuw nsw i32 %kdma.tid, 3, !dbg !5
  %kdma.intra = and i32 %6, 8, !dbg !5
  %kdma.intra.hi = xor i32 %kdma.intra, 8, !dbg !5
  %7 = call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !5
  %8 = sext i32 %7 to i64, !dbg !5
  %9 = call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !5
  %10 = sext i32 %9 to i64, !dbg !5
  %11 = shl nsw i64 %10, 5, !dbg !5
  %12 = add nsw i64 %11, %8, !dbg !5
  %13 = lshr i64 %12, 5, !dbg !5
  %14 = and i64 %13, 576460752303423456, !dbg !5
  %15 = and i64 %8, 31, !dbg !5
  %16 = or disjoint i64 %14, %15, !dbg !5
  %17 = and i64 %13, 31, !dbg !5
  %18 = call i32 @llvm.amdgcn.workgroup.id.z(), !dbg !5
  %19 = sext i32 %18 to i64, !dbg !5
  %20 = lshr i64 %12, 10, !dbg !8
  %21 = add nuw nsw i64 %13, %20, !dbg !8
  %22 = and i64 %21, 31, !dbg !8
  %23 = shl nsw i64 %16, 7, !dbg !8
  %24 = mul nsw i64 %20, -4096, !dbg !8
  %25 = add i64 %23, %24, !dbg !8
  %26 = shl nsw i64 %25, 7, !dbg !8
  %.idx = shl nuw nsw i64 %22, 20, !dbg !8
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #9, !dbg !8, !range !32, !frisk.register_lifetimes !1
  %address.local = zext nneg i32 %address.thread to i64, !dbg !8
  %address.local407 = shl nuw nsw i64 %address.local, 3, !dbg !8
  %address.local408 = and i64 %address.local407, 1536, !dbg !8
  %address.local409 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local408, !dbg !8
  %address.local410 = ptrtoint ptr addrspace(3) %address.local409 to i64, !dbg !8
  %address.local411 = trunc i64 %address.local410 to i32, !dbg !8
  %27 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local411), !dbg !8
  %address.local412 = shl nuw nsw i64 %address.local, 7, !dbg !8
  %address.local413 = and i64 %address.local412, 16256, !dbg !8
  %address.local414 = lshr i64 %address.local, 4, !dbg !8
  %address.local415 = and i64 %address.local414, 8, !dbg !8
  %address.local416 = or disjoint i64 %address.local413, %address.local415, !dbg !8
  %28 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %.idx1402 = shl i64 %25, 8, !dbg !8
  %29 = add i64 %.idx, %.idx1402, !dbg !8
  %address.local417.idx = shl nuw nsw i64 %address.local416, 1, !dbg !8
  %30 = or disjoint i64 %29, %address.local417.idx, !dbg !8
  %dma.buffer.offset1399 = trunc i64 %30 to i32, !dbg !8
  %31 = insertelement <2 x i64> poison, i64 %28, i64 0, !dbg !8
  %32 = bitcast <2 x i64> %31 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc1400 = shufflevector <4 x i32> %32, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
  %dma.buffer.lds1401 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %27), !dbg !8
  %33 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1401, i32 %dma.buffer.offset1399, <4 x i32> %dma.buffer.rsrc1400) #10, !dbg !8
  %address.local418 = or disjoint i64 %address.local, 256, !dbg !8
  %address.local419 = shl nuw nsw i64 %address.local418, 3, !dbg !8
  %address.local420 = and i64 %address.local419, 3584, !dbg !8
  %address.local421 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local420, !dbg !8
  %address.local422 = ptrtoint ptr addrspace(3) %address.local421 to i64, !dbg !8
  %address.local423 = trunc i64 %address.local422 to i32, !dbg !8
  %34 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local423), !dbg !8
  %address.local426 = lshr i64 %address.local418, 4, !dbg !8
  %address.local427 = and i64 %address.local426, 24, !dbg !8
  %address.local428 = or disjoint i64 %address.local413, %address.local427, !dbg !8
  %address.local429.idx = shl nuw nsw i64 %address.local428, 1, !dbg !8
  %35 = or disjoint i64 %29, %address.local429.idx, !dbg !8
  %dma.buffer.offset1396 = trunc i64 %35 to i32, !dbg !8
  %dma.buffer.lds1398 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %34), !dbg !8
  %36 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1398, i32 %dma.buffer.offset1396, <4 x i32> %dma.buffer.rsrc1400) #10, !dbg !8
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %37 = shl nuw nsw i64 %22, 19, !dbg !8
  %38 = add i64 %37, %26, !dbg !8
  %address.local430 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local408, !dbg !8
  %address.local431 = ptrtoint ptr addrspace(3) %address.local430 to i64, !dbg !8
  %address.local432 = trunc i64 %address.local431 to i32, !dbg !8
  %39 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local432), !dbg !8
  %40 = or disjoint i64 %38, 32, !dbg !8
  %41 = or disjoint i64 %address.local416, %40, !dbg !8
  %.tr = trunc i64 %41 to i32, !dbg !8
  %dma.buffer.offset1393 = shl i32 %.tr, 1, !dbg !8
  %dma.buffer.lds1395 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %39), !dbg !8
  %42 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1395, i32 %dma.buffer.offset1393, <4 x i32> %dma.buffer.rsrc1400) #10, !dbg !8
  %address.local434 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local420, !dbg !8
  %address.local435 = ptrtoint ptr addrspace(3) %address.local434 to i64, !dbg !8
  %address.local436 = trunc i64 %address.local435 to i32, !dbg !8
  %43 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local436), !dbg !8
  %44 = or disjoint i64 %address.local428, %40, !dbg !8
  %.tr1406 = trunc i64 %44 to i32, !dbg !8
  %dma.buffer.offset1390 = shl i32 %.tr1406, 1, !dbg !8
  %dma.buffer.lds1392 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %43), !dbg !8
  %45 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1392, i32 %dma.buffer.offset1390, <4 x i32> %dma.buffer.rsrc1400) #10, !dbg !8
  %address.local438 = lshr i32 %address.thread, 2, !dbg !11
  %address.local439 = and i32 %address.local438, 4, !dbg !11
  %address.local440 = shl nuw nsw i32 %address.thread, 1, !dbg !11
  %address.local441 = and i32 %address.local440, 384, !dbg !11
  %address.local442 = or disjoint i32 %address.local439, %address.local441, !dbg !11
  %address.local443 = shl nuw nsw i32 %address.thread, 3, !dbg !11
  %address.local444 = and i32 %address.local443, 120, !dbg !11
  %address.local445 = or disjoint i32 %address.local442, %address.local444, !dbg !11
  %address.local446 = shl nuw nsw i32 %address.thread, 5, !dbg !11
  %address.local447 = and i32 %address.local446, 1024, !dbg !11
  %address.local448 = or disjoint i32 %address.local445, %address.local447, !dbg !11
  %address.local449 = zext nneg i32 %address.local448 to i64, !dbg !11
  %address.local450 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local449, !dbg !11
  %46 = load <4 x half>, ptr addrspace(3) %address.local450, align 8, !dbg !11
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local450, i64 4096, !dbg !11
  %47 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !11
  %48 = fmul <4 x half> %46, splat (half 0xH3015), !dbg !11
  %lds.const.gep1200 = getelementptr i8, ptr addrspace(3) %address.local450, i64 1024, !dbg !11
  %49 = load <4 x half>, ptr addrspace(3) %lds.const.gep1200, align 8, !dbg !11
  %50 = fmul <4 x half> %47, splat (half 0xH3015), !dbg !11
  %lds.const.gep1201 = getelementptr i8, ptr addrspace(3) %address.local450, i64 5120, !dbg !11
  %51 = load <4 x half>, ptr addrspace(3) %lds.const.gep1201, align 8, !dbg !11
  %52 = fmul <4 x half> %49, splat (half 0xH3015), !dbg !11
  %53 = fmul <4 x half> %51, splat (half 0xH3015), !dbg !11
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %54 = or disjoint i64 %38, 64, !dbg !8
  %55 = or disjoint i64 %address.local416, %54, !dbg !8
  %.tr1408 = trunc i64 %55 to i32, !dbg !8
  %dma.buffer.offset1387 = shl i32 %.tr1408, 1, !dbg !8
  %56 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1401, i32 %dma.buffer.offset1387, <4 x i32> %dma.buffer.rsrc1400) #10, !dbg !8
  %57 = or disjoint i64 %address.local428, %54, !dbg !8
  %.tr1410 = trunc i64 %57 to i32, !dbg !8
  %dma.buffer.offset1384 = shl i32 %.tr1410, 1, !dbg !8
  %58 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1398, i32 %dma.buffer.offset1384, <4 x i32> %dma.buffer.rsrc1400) #10, !dbg !8
  %lds.const.gep1202 = getelementptr i8, ptr addrspace(3) %address.local450, i64 8192, !dbg !11
  %59 = load <4 x half>, ptr addrspace(3) %lds.const.gep1202, align 8, !dbg !11
  %lds.const.gep1203 = getelementptr i8, ptr addrspace(3) %address.local450, i64 12288, !dbg !11
  %60 = load <4 x half>, ptr addrspace(3) %lds.const.gep1203, align 8, !dbg !11
  %lds.const.gep1204 = getelementptr i8, ptr addrspace(3) %address.local450, i64 9216, !dbg !11
  %61 = load <4 x half>, ptr addrspace(3) %lds.const.gep1204, align 8, !dbg !11
  %62 = fmul <4 x half> %59, splat (half 0xH3015), !dbg !11
  %lds.const.gep1205 = getelementptr i8, ptr addrspace(3) %address.local450, i64 13312, !dbg !11
  %63 = load <4 x half>, ptr addrspace(3) %lds.const.gep1205, align 8, !dbg !11
  %64 = fmul <4 x half> %60, splat (half 0xH3015), !dbg !11
  %65 = fmul <4 x half> %61, splat (half 0xH3015), !dbg !11
  %66 = fmul <4 x half> %63, splat (half 0xH3015), !dbg !11
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %67 = or disjoint i64 %38, 96, !dbg !8
  %68 = or disjoint i64 %address.local416, %67, !dbg !8
  %.tr1412 = trunc i64 %68 to i32, !dbg !8
  %dma.buffer.offset1381 = shl i32 %.tr1412, 1, !dbg !8
  %69 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1395, i32 %dma.buffer.offset1381, <4 x i32> %dma.buffer.rsrc1400) #10, !dbg !8
  %70 = or disjoint i64 %address.local428, %67, !dbg !8
  %.tr1414 = trunc i64 %70 to i32, !dbg !8
  %dma.buffer.offset1378 = shl i32 %.tr1414, 1, !dbg !8
  %71 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1392, i32 %dma.buffer.offset1378, <4 x i32> %dma.buffer.rsrc1400) #10, !dbg !8
  %72 = load <4 x half>, ptr addrspace(3) %address.local450, align 8, !dbg !11
  %73 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !11
  %74 = load <4 x half>, ptr addrspace(3) %lds.const.gep1200, align 8, !dbg !11
  %75 = fmul <4 x half> %72, splat (half 0xH3015), !dbg !11
  %76 = load <4 x half>, ptr addrspace(3) %lds.const.gep1201, align 8, !dbg !11
  %77 = fmul <4 x half> %73, splat (half 0xH3015), !dbg !11
  %78 = fmul <4 x half> %74, splat (half 0xH3015), !dbg !11
  %79 = fmul <4 x half> %76, splat (half 0xH3015), !dbg !11
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %80 = load <4 x half>, ptr addrspace(3) %lds.const.gep1202, align 8, !dbg !11
  %81 = load <4 x half>, ptr addrspace(3) %lds.const.gep1203, align 8, !dbg !11
  %82 = load <4 x half>, ptr addrspace(3) %lds.const.gep1204, align 8, !dbg !11
  %83 = fmul <4 x half> %80, splat (half 0xH3015), !dbg !11
  %84 = load <4 x half>, ptr addrspace(3) %lds.const.gep1205, align 8, !dbg !11
  %85 = fmul <4 x half> %81, splat (half 0xH3015), !dbg !11
  %86 = fmul <4 x half> %82, splat (half 0xH3015), !dbg !11
  %87 = fmul <4 x half> %84, splat (half 0xH3015), !dbg !11
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %88 = shl nuw nsw i64 %17, 12, !dbg !12
  %89 = shl nsw i64 %19, 17, !dbg !12
  %90 = shl nuw nsw i32 %kdma.tid, 2, !dbg !14
  %91 = and i32 %90, 3084, !dbg !14
  %92 = and i32 %kdma.tid, 1, !dbg !14
  %93 = or disjoint i32 %91, %92, !dbg !14
  %narrow = add nuw nsw i32 %93, 1, !dbg !14
  %94 = lshr i32 %narrow, 1, !dbg !14
  %95 = zext nneg i32 %94 to i64, !dbg !14
  %k0.prime.address.thread501 = call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #9, !dbg !12, !range !32, !frisk.register_lifetimes !1
  %k0.prime.address.local502 = zext nneg i32 %k0.prime.address.thread501 to i64, !dbg !8
  %k0.prime.address.local503 = shl nuw nsw i64 %k0.prime.address.local502, 3, !dbg !8
  %k0.prime.address.local504 = and i64 %k0.prime.address.local503, 1536, !dbg !8
  %k0.prime.address.local505 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %k0.prime.address.local504, !dbg !8
  %k0.prime.address.local506 = ptrtoint ptr addrspace(3) %k0.prime.address.local505 to i64, !dbg !8
  %k0.prime.address.local507 = trunc i64 %k0.prime.address.local506 to i32, !dbg !8
  %k0.prime. = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %k0.prime.address.local507), !dbg !12
  %k0.prime.dma.buffer.lds1377 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %k0.prime.), !dbg !12
  %k0.prime.2007 = shl nuw nsw i64 %17, 19, !dbg !12
  %k0.prime.2008 = shl nuw nsw i64 0, 14, !dbg !12
  %k0.prime.2009 = or disjoint i64 %k0.prime.2007, %k0.prime.2008, !dbg !12
  %k0.prime.kdma.source = zext nneg i32 %kdma.owner to i64, !dbg !12
  %k0.prime.kdma.source1526 = lshr i64 %k0.prime.kdma.source, 4, !dbg !12
  %k0.prime.kdma.source1527 = and i64 %k0.prime.kdma.source1526, 8, !dbg !12
  %k0.prime.kdma.source1528 = and i64 %k0.prime.kdma.source, 91, !dbg !12
  %k0.prime.kdma.source1529 = lshr i64 %k0.prime.kdma.source, 2, !dbg !12
  %k0.prime.kdma.source1530 = and i64 %k0.prime.kdma.source1529, 4, !dbg !12
  %k0.prime.kdma.source1531 = add nuw nsw i64 %k0.prime.kdma.source1530, %k0.prime.kdma.source, !dbg !12
  %k0.prime.kdma.source1532 = and i64 %k0.prime.kdma.source1531, 4, !dbg !12
  %k0.prime.kdma.source1533 = or disjoint i64 %k0.prime.kdma.source1528, %k0.prime.kdma.source1532, !dbg !12
  %k0.prime.kdma.source1534 = shl nuw nsw i64 %k0.prime.kdma.source1533, 7, !dbg !12
  %k0.prime.kdma.source1535 = or disjoint i64 %k0.prime.kdma.source1527, %k0.prime.kdma.source1534, !dbg !12
  %k0.prime.kdma.source1536 = or disjoint i64 %k0.prime.2009, %k0.prime.kdma.source1535, !dbg !12
  %k0.prime.kdma.source1537 = trunc nuw i64 %k0.prime.kdma.source1536 to i32, !dbg !12
  %k0.prime.kdma.source1538 = shl nuw nsw i32 %k0.prime.kdma.source1537, 1, !dbg !12
  %k0.prime.kdma.offset = or disjoint i32 %k0.prime.kdma.source1538, %kdma.intra, !dbg !12
  %k0.prime.2010 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !12
  %k0.prime.2011 = insertelement <2 x i64> poison, i64 %k0.prime.2010, i64 0, !dbg !12
  %k0.prime.2012 = bitcast <2 x i64> %k0.prime.2011 to <4 x i32>, !dbg !12
  %k0.prime.dma.buffer.rsrc1376 = shufflevector <4 x i32> %k0.prime.2012, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !12
  %k0.prime.2013 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.dma.buffer.lds1377, i32 %k0.prime.kdma.offset, <4 x i32> %k0.prime.dma.buffer.rsrc1376) #10, !dbg !12, !frisk.cross_bn_k0 !1
  %k0.prime.kdma.lds1554 = add i32 %k0.prime.dma.buffer.lds1377, 512, !dbg !12
  %k0.prime.kdma.source1539 = zext nneg i32 %kdma.owner.hi to i64, !dbg !12
  %k0.prime.kdma.source1540 = lshr i64 %k0.prime.kdma.source1539, 4, !dbg !12
  %k0.prime.kdma.source1541 = and i64 %k0.prime.kdma.source1540, 8, !dbg !12
  %k0.prime.kdma.source1542 = and i64 %k0.prime.kdma.source1539, 123, !dbg !12
  %k0.prime.kdma.source1543 = lshr i64 %k0.prime.kdma.source1539, 2, !dbg !12
  %k0.prime.kdma.source1544 = and i64 %k0.prime.kdma.source1543, 4, !dbg !12
  %k0.prime.kdma.source1545 = add nuw nsw i64 %k0.prime.kdma.source1544, %k0.prime.kdma.source1539, !dbg !12
  %k0.prime.kdma.source1546 = and i64 %k0.prime.kdma.source1545, 4, !dbg !12
  %k0.prime.kdma.source1547 = or disjoint i64 %k0.prime.kdma.source1542, %k0.prime.kdma.source1546, !dbg !12
  %k0.prime.kdma.source1548 = shl nuw nsw i64 %k0.prime.kdma.source1547, 7, !dbg !12
  %k0.prime.kdma.source1549 = or disjoint i64 %k0.prime.kdma.source1541, %k0.prime.kdma.source1548, !dbg !12
  %k0.prime.kdma.source1550 = or disjoint i64 %k0.prime.2009, %k0.prime.kdma.source1549, !dbg !12
  %k0.prime.kdma.source1551 = trunc nuw i64 %k0.prime.kdma.source1550 to i32, !dbg !12
  %k0.prime.kdma.source1552 = shl nuw nsw i32 %k0.prime.kdma.source1551, 1, !dbg !12
  %k0.prime.kdma.offset1553 = or disjoint i32 %k0.prime.kdma.source1552, %kdma.intra.hi, !dbg !12
  %k0.prime.2014 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.kdma.lds1554, i32 %k0.prime.kdma.offset1553, <4 x i32> %k0.prime.dma.buffer.rsrc1376) #10, !dbg !12, !frisk.cross_bn_k0 !1
  %k0.prime.address.local519 = or disjoint i64 %k0.prime.address.local502, 256, !dbg !8
  %k0.prime.address.local520 = shl nuw nsw i64 %k0.prime.address.local519, 3, !dbg !8
  %k0.prime.address.local521 = and i64 %k0.prime.address.local520, 3584, !dbg !8
  %k0.prime.address.local522 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %k0.prime.address.local521, !dbg !8
  %k0.prime.address.local523 = ptrtoint ptr addrspace(3) %k0.prime.address.local522 to i64, !dbg !8
  %k0.prime.address.local524 = trunc i64 %k0.prime.address.local523 to i32, !dbg !8
  %k0.prime.2015 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %k0.prime.address.local524), !dbg !12
  %k0.prime.dma.buffer.lds1374 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %k0.prime.2015), !dbg !12
  %k0.prime.kdma.source1566 = or disjoint i64 %k0.prime.kdma.source1535, 16, !dbg !12
  %k0.prime.kdma.source1567 = or disjoint i64 %k0.prime.2009, %k0.prime.kdma.source1566, !dbg !12
  %k0.prime.kdma.source1568 = trunc nuw i64 %k0.prime.kdma.source1567 to i32, !dbg !12
  %k0.prime.kdma.source1569 = shl nuw nsw i32 %k0.prime.kdma.source1568, 1, !dbg !12
  %k0.prime.kdma.offset1570 = or disjoint i32 %k0.prime.kdma.source1569, %kdma.intra, !dbg !12
  %k0.prime.2016 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.dma.buffer.lds1374, i32 %k0.prime.kdma.offset1570, <4 x i32> %k0.prime.dma.buffer.rsrc1376) #10, !dbg !12, !frisk.cross_bn_k0 !1
  %k0.prime.kdma.lds1588 = add i32 %k0.prime.dma.buffer.lds1374, 512, !dbg !12
  %k0.prime.kdma.source1583 = or disjoint i64 %k0.prime.kdma.source1549, 16, !dbg !12
  %k0.prime.kdma.source1584 = or disjoint i64 %k0.prime.2009, %k0.prime.kdma.source1583, !dbg !12
  %k0.prime.kdma.source1585 = trunc nuw i64 %k0.prime.kdma.source1584 to i32, !dbg !12
  %k0.prime.kdma.source1586 = shl nuw nsw i32 %k0.prime.kdma.source1585, 1, !dbg !12
  %k0.prime.kdma.offset1587 = or disjoint i32 %k0.prime.kdma.source1586, %kdma.intra.hi, !dbg !12
  %k0.prime.2017 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.kdma.lds1588, i32 %k0.prime.kdma.offset1587, <4 x i32> %k0.prime.dma.buffer.rsrc1376) #10, !dbg !12, !frisk.cross_bn_k0 !1
  br label %96, !dbg !7

96:                                               ; preds = %k0.join, %4
  %97 = phi i64 [ %955, %k0.join ], [ 0, %4 ], !dbg !7
  %98 = phi [2 x <8 x float>] [ %1033, %k0.join ], [ zeroinitializer, %4 ], !dbg !7
  %99 = phi [2 x <1 x float>] [ %798, %k0.join ], [ zeroinitializer, %4 ], !dbg !7
  %100 = phi [2 x <1 x float>] [ %1035, %k0.join ], [ [<1 x float> splat (float 0xFFF0000000000000), <1 x float> splat (float 0xFFF0000000000000)], %4 ], !dbg !7
  %101 = phi [2 x <8 x float>] [ %1039, %k0.join ], [ zeroinitializer, %4 ], !dbg !7
  %102 = phi [2 x <8 x float>] [ %1043, %k0.join ], [ zeroinitializer, %4 ], !dbg !7
  %103 = phi [2 x <8 x float>] [ %1047, %k0.join ], [ zeroinitializer, %4 ], !dbg !7
  %104 = icmp samesign ult i64 %97, 32, !dbg !7
  br i1 %104, label %105, label %1048, !dbg !7

105:                                              ; preds = %96
  %106 = shl nuw nsw i64 %17, 19, !dbg !12
  %107 = shl nuw nsw i64 %97, 14, !dbg !12
  %108 = or disjoint i64 %106, %107, !dbg !12
  %address.thread501 = call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #9, !dbg !12, !range !32, !frisk.register_lifetimes !1
  %address.local502 = zext nneg i32 %address.thread501 to i64, !dbg !8
  %address.local503 = shl nuw nsw i64 %address.local502, 3, !dbg !8
  %address.local504 = and i64 %address.local503, 1536, !dbg !8
  %address.local505 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local504, !dbg !8
  %address.local506 = ptrtoint ptr addrspace(3) %address.local505 to i64, !dbg !8
  %address.local507 = trunc i64 %address.local506 to i32, !dbg !8
  %109 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local507), !dbg !12
  %110 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !12
  %111 = insertelement <2 x i64> poison, i64 %110, i64 0, !dbg !12
  %112 = bitcast <2 x i64> %111 to <4 x i32>, !dbg !12
  %dma.buffer.rsrc1376 = shufflevector <4 x i32> %112, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !12
  %dma.buffer.lds1377 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %109), !dbg !12
  %kdma.source = zext nneg i32 %kdma.owner to i64, !dbg !12
  %kdma.source1526 = lshr i64 %kdma.source, 4, !dbg !12
  %kdma.source1527 = and i64 %kdma.source1526, 8, !dbg !12
  %kdma.source1528 = and i64 %kdma.source, 91, !dbg !12
  %kdma.source1529 = lshr i64 %kdma.source, 2, !dbg !12
  %kdma.source1530 = and i64 %kdma.source1529, 4, !dbg !12
  %kdma.source1531 = add nuw nsw i64 %kdma.source1530, %kdma.source, !dbg !12
  %kdma.source1532 = and i64 %kdma.source1531, 4, !dbg !12
  %kdma.source1533 = or disjoint i64 %kdma.source1528, %kdma.source1532, !dbg !12
  %kdma.source1534 = shl nuw nsw i64 %kdma.source1533, 7, !dbg !12
  %kdma.source1535 = or disjoint i64 %kdma.source1527, %kdma.source1534, !dbg !12
  %kdma.source1536 = or disjoint i64 %108, %kdma.source1535, !dbg !12
  %kdma.source1537 = trunc nuw i64 %kdma.source1536 to i32, !dbg !12
  %kdma.source1538 = shl nuw nsw i32 %kdma.source1537, 1, !dbg !12
  %kdma.offset = or disjoint i32 %kdma.source1538, %kdma.intra, !dbg !12
  %kdma.source1539 = zext nneg i32 %kdma.owner.hi to i64, !dbg !12
  %kdma.source1540 = lshr i64 %kdma.source1539, 4, !dbg !12
  %kdma.source1541 = and i64 %kdma.source1540, 8, !dbg !12
  %kdma.source1542 = and i64 %kdma.source1539, 123, !dbg !12
  %kdma.source1543 = lshr i64 %kdma.source1539, 2, !dbg !12
  %kdma.source1544 = and i64 %kdma.source1543, 4, !dbg !12
  %kdma.source1545 = add nuw nsw i64 %kdma.source1544, %kdma.source1539, !dbg !12
  %kdma.source1546 = and i64 %kdma.source1545, 4, !dbg !12
  %kdma.source1547 = or disjoint i64 %kdma.source1542, %kdma.source1546, !dbg !12
  %kdma.source1548 = shl nuw nsw i64 %kdma.source1547, 7, !dbg !12
  %kdma.source1549 = or disjoint i64 %kdma.source1541, %kdma.source1548, !dbg !12
  %kdma.source1550 = or disjoint i64 %108, %kdma.source1549, !dbg !12
  %kdma.source1551 = trunc nuw i64 %kdma.source1550 to i32, !dbg !12
  %kdma.source1552 = shl nuw nsw i32 %kdma.source1551, 1, !dbg !12
  %kdma.offset1553 = or disjoint i32 %kdma.source1552, %kdma.intra.hi, !dbg !12
  %kdma.lds1554 = add i32 %dma.buffer.lds1377, 512, !dbg !12
  %address.local519 = or disjoint i64 %address.local502, 256, !dbg !8
  %address.local520 = shl nuw nsw i64 %address.local519, 3, !dbg !8
  %address.local521 = and i64 %address.local520, 3584, !dbg !8
  %address.local522 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local521, !dbg !8
  %address.local523 = ptrtoint ptr addrspace(3) %address.local522 to i64, !dbg !8
  %address.local524 = trunc i64 %address.local523 to i32, !dbg !8
  %113 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local524), !dbg !12
  %dma.buffer.lds1374 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %113), !dbg !12
  %kdma.source1566 = or disjoint i64 %kdma.source1535, 16, !dbg !12
  %kdma.source1567 = or disjoint i64 %108, %kdma.source1566, !dbg !12
  %kdma.source1568 = trunc nuw i64 %kdma.source1567 to i32, !dbg !12
  %kdma.source1569 = shl nuw nsw i32 %kdma.source1568, 1, !dbg !12
  %kdma.offset1570 = or disjoint i32 %kdma.source1569, %kdma.intra, !dbg !12
  %kdma.source1583 = or disjoint i64 %kdma.source1549, 16, !dbg !12
  %kdma.source1584 = or disjoint i64 %108, %kdma.source1583, !dbg !12
  %kdma.source1585 = trunc nuw i64 %kdma.source1584 to i32, !dbg !12
  %kdma.source1586 = shl nuw nsw i32 %kdma.source1585, 1, !dbg !12
  %kdma.offset1587 = or disjoint i32 %kdma.source1586, %kdma.intra.hi, !dbg !12
  %kdma.lds1588 = add i32 %dma.buffer.lds1374, 512, !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %114 = or disjoint i64 %88, %89, !dbg !12
  %115 = or disjoint i64 %114, %97, !dbg !12
  %116 = or disjoint i64 %115, 1024, !dbg !12
  %117 = icmp slt i32 %18, 0, !dbg !12
  %118 = sub nuw nsw i64 -1025, %115, !dbg !12
  %119 = select i1 %117, i64 %118, i64 %116, !dbg !12
  %120 = sdiv i64 %119, 4096, !dbg !12
  %.lobit37 = ashr i32 %18, 31, !dbg !12
  %121 = zext i32 %.lobit37 to i64, !dbg !12
  %122 = xor i64 %120, %121, !dbg !12
  %123 = shl i64 %122, 19, !dbg !12
  %124 = and i64 %123, 16252928, !dbg !12
  %address.local536 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local504, !dbg !8
  %address.local537 = ptrtoint ptr addrspace(3) %address.local536 to i64, !dbg !8
  %address.local538 = trunc i64 %address.local537 to i32, !dbg !8
  %125 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local538), !dbg !12
  %.idx1420 = shl nuw nsw i64 %97, 15, !dbg !12
  %dma.buffer.lds1371 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %125), !dbg !12
  %kdma.source1600 = or disjoint i64 %124, %kdma.source1535, !dbg !12
  %kdma.source1601 = shl nuw nsw i64 %kdma.source1600, 1, !dbg !12
  %kdma.source1602 = or i64 %kdma.source1601, %.idx1420, !dbg !12
  %kdma.source1603 = trunc i64 %kdma.source1602 to i32, !dbg !12
  %kdma.source1604 = or i32 %kdma.source1603, 64, !dbg !12
  %kdma.offset1605 = add i32 %kdma.source1604, %kdma.intra, !dbg !12
  %126 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1371, i32 %kdma.offset1605, <4 x i32> %dma.buffer.rsrc1376) #10, !dbg !12
  %kdma.source1618 = or disjoint i64 %124, %kdma.source1549, !dbg !12
  %kdma.source1619 = shl nuw nsw i64 %kdma.source1618, 1, !dbg !12
  %kdma.source1620 = or i64 %kdma.source1619, %.idx1420, !dbg !12
  %kdma.source1621 = trunc i64 %kdma.source1620 to i32, !dbg !12
  %kdma.source1622 = or i32 %kdma.source1621, 64, !dbg !12
  %kdma.offset1623 = add i32 %kdma.source1622, %kdma.intra.hi, !dbg !12
  %kdma.lds1624 = add i32 %dma.buffer.lds1371, 512, !dbg !12
  %127 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1624, i32 %kdma.offset1623, <4 x i32> %dma.buffer.rsrc1376) #10, !dbg !12
  %address.local540 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local521, !dbg !8
  %address.local541 = ptrtoint ptr addrspace(3) %address.local540 to i64, !dbg !8
  %address.local542 = trunc i64 %address.local541 to i32, !dbg !8
  %128 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local542), !dbg !12
  %dma.buffer.lds1368 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %128), !dbg !12
  %kdma.source1637 = or disjoint i64 %124, %kdma.source1566, !dbg !12
  %kdma.source1638 = shl nuw nsw i64 %kdma.source1637, 1, !dbg !12
  %kdma.source1639 = or i64 %kdma.source1638, %.idx1420, !dbg !12
  %kdma.source1640 = trunc i64 %kdma.source1639 to i32, !dbg !12
  %kdma.source1641 = or i32 %kdma.source1640, 64, !dbg !12
  %kdma.offset1642 = add i32 %kdma.source1641, %kdma.intra, !dbg !12
  %129 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1368, i32 %kdma.offset1642, <4 x i32> %dma.buffer.rsrc1376) #10, !dbg !12
  %kdma.source1656 = or disjoint i64 %124, %kdma.source1583, !dbg !12
  %kdma.source1657 = shl nuw nsw i64 %kdma.source1656, 1, !dbg !12
  %kdma.source1658 = or i64 %kdma.source1657, %.idx1420, !dbg !12
  %kdma.source1659 = trunc i64 %kdma.source1658 to i32, !dbg !12
  %kdma.source1660 = or i32 %kdma.source1659, 64, !dbg !12
  %kdma.offset1661 = add i32 %kdma.source1660, %kdma.intra.hi, !dbg !12
  %kdma.lds1662 = add i32 %dma.buffer.lds1368, 512, !dbg !12
  %130 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1662, i32 %kdma.offset1661, <4 x i32> %dma.buffer.rsrc1376) #10, !dbg !12
  %address.local545 = lshr i64 %address.local502, 2, !dbg !14
  %address.local548 = shl nuw nsw i64 %address.local502, 1, !dbg !14
  %address.local549 = and i64 %address.local548, 24, !dbg !14
  %address.local551 = shl i64 %address.local502, 62, !dbg !14
  %address.local552 = ashr exact i64 %address.local551, 57, !dbg !14
  %address.local553 = and i64 %address.local552, -64, !dbg !14
  %address.local556 = and i64 %address.local545, 12, !dbg !11
  %address.local558 = and i64 %address.local502, 63, !dbg !11
  %address.local559 = lshr i64 %address.local558, 5, !dbg !11
  %address.local560 = mul nuw nsw i64 %address.local559, 1016, !dbg !11
  %.idx1423 = mul nuw nsw i64 %address.local502, 320, !dbg !14
  %131 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %.idx1423, !dbg !14
  %.idx1424 = mul nsw i64 %address.local545, -1280, !dbg !14
  %132 = getelementptr i8, ptr addrspace(3) %131, i64 %.idx1424, !dbg !14
  %133 = getelementptr [2 x i8], ptr addrspace(3) %132, i64 %address.local549, !dbg !14
  %134 = getelementptr [2 x i8], ptr addrspace(3) %133, i64 %address.local553, !dbg !14
  %135 = getelementptr [2 x i8], ptr addrspace(3) %134, i64 %address.local556, !dbg !14
  %address.local562 = getelementptr [2 x i8], ptr addrspace(3) %135, i64 %address.local560, !dbg !14
  %136 = add nsw i64 %.idx1423, %.idx1424, !dbg !14
  %.idx1937 = shl nuw nsw i64 %address.local549, 1, !dbg !14
  %137 = or disjoint i64 %136, %.idx1937, !dbg !14
  %138 = or disjoint i64 %address.local553, %address.local556, !dbg !14
  %139 = shl nsw i64 %138, 1, !dbg !14
  %140 = add nsw i64 %139, %137, !dbg !14
  %address.local562.idx = mul nuw nsw i64 %address.local559, 2032, !dbg !14
  %141 = add nsw i64 %140, %address.local562.idx, !dbg !14
  %142 = lshr i64 %141, 6, !dbg !14
  %143 = and i64 %142, 8, !dbg !14
  %144 = xor i64 %141, %143, !dbg !14
  %145 = sub i64 %144, %141, !dbg !14
  %kdma.swizzled = getelementptr i8, ptr addrspace(3) %address.local562, i64 %145, !dbg !14
  %146 = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8, !dbg !14
  %147 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %48, <4 x half> %146, <4 x float> zeroinitializer) #9, !dbg !14
  %kdma.gep = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4096, !dbg !14
  %148 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8, !dbg !14
  %address.local573 = shl nuw nsw i32 %address.thread501, 1, !dbg !14
  %address.local574 = and i32 %address.local573, 24, !dbg !14
  %address.local575 = shl nuw nsw i32 %address.thread501, 8, !dbg !14
  %address.local576 = and i32 %address.local575, 768, !dbg !14
  %address.local577 = or disjoint i32 %address.local574, %address.local576, !dbg !14
  %address.local578 = shl nuw nsw i32 %address.thread501, 5, !dbg !14
  %address.local579 = and i32 %address.local578, 96, !dbg !14
  %address.local580 = or disjoint i32 %address.local577, %address.local579, !dbg !14
  %address.local581 = zext nneg i32 %address.local580 to i64, !dbg !14
  %address.base = call i64 asm sideeffect "", "=v,0"(i64 %95) #9, !dbg !14, !frisk.register_lifetimes !1
  %address.local582 = lshr i32 %address.thread501, 1, !dbg !14
  %address.local583 = and i32 %address.local582, 1, !dbg !14
  %address.local584 = zext nneg i32 %address.local583 to i64, !dbg !14
  %address.local585 = add nsw i64 %address.base, %address.local584, !dbg !14
  %address.local586 = shl i64 %address.local585, 6, !dbg !14
  %address.local587 = sub i64 %address.local581, %address.local586, !dbg !14
  %149 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local587, !dbg !14
  %150 = getelementptr [2 x i8], ptr addrspace(3) %149, i64 %address.local556, !dbg !14
  %151 = getelementptr [2 x i8], ptr addrspace(3) %150, i64 %address.local560, !dbg !14
  %address.local591 = getelementptr i8, ptr addrspace(3) %151, i64 64, !dbg !14
  %.idx19401943 = add i64 %address.local587, %address.local556, !dbg !14
  %152 = shl i64 %.idx19401943, 1, !dbg !14
  %153 = add i64 %152, %address.local562.idx, !dbg !14
  %154 = add i64 %153, 64, !dbg !14
  %155 = lshr i64 %154, 6, !dbg !14
  %156 = and i64 %155, 8, !dbg !14
  %157 = xor i64 %154, %156, !dbg !14
  %158 = sub i64 %157, %154, !dbg !14
  %kdma.swizzled1811 = getelementptr i8, ptr addrspace(3) %address.local591, i64 %158, !dbg !14
  %159 = load <4 x half>, ptr addrspace(3) %kdma.swizzled1811, align 8, !dbg !14, !frisk.k_lds_pair !35
  %kdma.gep1815 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1811, i64 128, !dbg !14
  %160 = load <4 x half>, ptr addrspace(3) %kdma.gep1815, align 8, !dbg !14, !frisk.k_lds_pair !35
  %kdma.gep1812 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1811, i64 4096, !dbg !14
  %161 = load <4 x half>, ptr addrspace(3) %kdma.gep1812, align 8, !dbg !14, !frisk.k_lds_pair !36
  %kdma.gep1816 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1811, i64 4224, !dbg !14
  %162 = load <4 x half>, ptr addrspace(3) %kdma.gep1816, align 8, !dbg !14, !frisk.k_lds_pair !36
  %kdma.gep1813 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 128, !dbg !14
  %163 = load <4 x half>, ptr addrspace(3) %kdma.gep1813, align 8, !dbg !14, !frisk.k_lds_pair !37
  %kdma.gep1817 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1024, !dbg !14
  %164 = load <4 x half>, ptr addrspace(3) %kdma.gep1817, align 8, !dbg !14, !frisk.k_lds_pair !37
  %kdma.gep1814 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4224, !dbg !14
  %165 = load <4 x half>, ptr addrspace(3) %kdma.gep1814, align 8, !dbg !14, !frisk.k_lds_pair !38
  %kdma.gep1818 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5120, !dbg !14
  %166 = load <4 x half>, ptr addrspace(3) %kdma.gep1818, align 8, !dbg !14, !frisk.k_lds_pair !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.k_lds_prefetch !1
  %167 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %50, <4 x half> %148, <4 x float> %147) #9, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.k_lds_prefetch !1
  %kdma.gep1819 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1811, i64 1024, !dbg !14
  %168 = load <4 x half>, ptr addrspace(3) %kdma.gep1819, align 8, !dbg !14, !frisk.k_lds_pair !33
  %kdma.gep1823 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1811, i64 1152, !dbg !14
  %169 = load <4 x half>, ptr addrspace(3) %kdma.gep1823, align 8, !dbg !14, !frisk.k_lds_pair !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.k_lds_prefetch !1
  %170 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %48, <4 x half> %159, <4 x float> zeroinitializer) #9, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.k_lds_prefetch !1
  %kdma.gep1820 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1811, i64 5120, !dbg !14
  %171 = load <4 x half>, ptr addrspace(3) %kdma.gep1820, align 8, !dbg !14, !frisk.k_lds_pair !39
  %kdma.gep1824 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1811, i64 5248, !dbg !14
  %172 = load <4 x half>, ptr addrspace(3) %kdma.gep1824, align 8, !dbg !14, !frisk.k_lds_pair !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.k_lds_prefetch !1
  %173 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %50, <4 x half> %161, <4 x float> %170) #9, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.k_lds_prefetch !1
  %kdma.gep1821 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1152, !dbg !14
  %174 = load <4 x half>, ptr addrspace(3) %kdma.gep1821, align 8, !dbg !14, !frisk.k_lds_pair !40
  %kdma.gep1822 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5248, !dbg !14
  %175 = load <4 x half>, ptr addrspace(3) %kdma.gep1822, align 8, !dbg !14, !frisk.k_lds_pair !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.k_lds_prefetch !1
  %176 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %48, <4 x half> %163, <4 x float> zeroinitializer) #9, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.k_lds_prefetch !1
  %177 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %50, <4 x half> %165, <4 x float> %176) #9, !dbg !14
  %178 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %48, <4 x half> %160, <4 x float> zeroinitializer) #9, !dbg !14
  %179 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %50, <4 x half> %162, <4 x float> %178) #9, !dbg !14
  %180 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %48, <4 x half> %164, <4 x float> zeroinitializer) #9, !dbg !14
  %181 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %50, <4 x half> %166, <4 x float> %180) #9, !dbg !14
  %182 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %48, <4 x half> %168, <4 x float> zeroinitializer) #9, !dbg !14
  %183 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %50, <4 x half> %171, <4 x float> %182) #9, !dbg !14
  %184 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %48, <4 x half> %174, <4 x float> zeroinitializer) #9, !dbg !14
  %185 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %50, <4 x half> %175, <4 x float> %184) #9, !dbg !14
  %186 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %48, <4 x half> %169, <4 x float> zeroinitializer) #9, !dbg !14
  %187 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %50, <4 x half> %172, <4 x float> %186) #9, !dbg !14
  %188 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %52, <4 x half> %146, <4 x float> zeroinitializer) #9, !dbg !14
  %189 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %53, <4 x half> %148, <4 x float> %188) #9, !dbg !14
  %190 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %52, <4 x half> %159, <4 x float> zeroinitializer) #9, !dbg !14
  %191 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %53, <4 x half> %161, <4 x float> %190) #9, !dbg !14
  %192 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %52, <4 x half> %163, <4 x float> zeroinitializer) #9, !dbg !14
  %193 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %53, <4 x half> %165, <4 x float> %192) #9, !dbg !14
  %194 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %52, <4 x half> %160, <4 x float> zeroinitializer) #9, !dbg !14
  %195 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %53, <4 x half> %162, <4 x float> %194) #9, !dbg !14
  %196 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %52, <4 x half> %164, <4 x float> zeroinitializer) #9, !dbg !14
  %197 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %53, <4 x half> %166, <4 x float> %196) #9, !dbg !14
  %198 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %52, <4 x half> %168, <4 x float> zeroinitializer) #9, !dbg !14
  %199 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %53, <4 x half> %171, <4 x float> %198) #9, !dbg !14
  %200 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %52, <4 x half> %174, <4 x float> zeroinitializer) #9, !dbg !14
  %201 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %53, <4 x half> %175, <4 x float> %200) #9, !dbg !14
  %202 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %52, <4 x half> %169, <4 x float> zeroinitializer) #9, !dbg !14
  %203 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %53, <4 x half> %172, <4 x float> %202) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %204 = or disjoint i64 %115, 2048, !dbg !12
  %205 = sub nuw nsw i64 -2049, %115, !dbg !12
  %206 = select i1 %117, i64 %205, i64 %204, !dbg !12
  %207 = sdiv i64 %206, 4096, !dbg !12
  %208 = xor i64 %207, %121, !dbg !12
  %209 = shl i64 %208, 19, !dbg !12
  %210 = and i64 %209, 16252928, !dbg !12
  %kdma.source1674 = or disjoint i64 %210, %kdma.source1535, !dbg !12
  %kdma.source1675 = shl nuw nsw i64 %kdma.source1674, 1, !dbg !12
  %kdma.source1676 = or i64 %kdma.source1675, %.idx1420, !dbg !12
  %kdma.source1677 = trunc i64 %kdma.source1676 to i32, !dbg !12
  %kdma.source1678 = or i32 %kdma.source1677, 128, !dbg !12
  %kdma.offset1679 = add i32 %kdma.source1678, %kdma.intra, !dbg !12
  %211 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1377, i32 %kdma.offset1679, <4 x i32> %dma.buffer.rsrc1376) #10, !dbg !12
  %kdma.source1692 = or disjoint i64 %210, %kdma.source1549, !dbg !12
  %kdma.source1693 = shl nuw nsw i64 %kdma.source1692, 1, !dbg !12
  %kdma.source1694 = or i64 %kdma.source1693, %.idx1420, !dbg !12
  %kdma.source1695 = trunc i64 %kdma.source1694 to i32, !dbg !12
  %kdma.source1696 = or i32 %kdma.source1695, 128, !dbg !12
  %kdma.offset1697 = add i32 %kdma.source1696, %kdma.intra.hi, !dbg !12
  %212 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1554, i32 %kdma.offset1697, <4 x i32> %dma.buffer.rsrc1376) #10, !dbg !12
  %kdma.source1711 = or disjoint i64 %210, %kdma.source1566, !dbg !12
  %kdma.source1712 = shl nuw nsw i64 %kdma.source1711, 1, !dbg !12
  %kdma.source1713 = or i64 %kdma.source1712, %.idx1420, !dbg !12
  %kdma.source1714 = trunc i64 %kdma.source1713 to i32, !dbg !12
  %kdma.source1715 = or i32 %kdma.source1714, 128, !dbg !12
  %kdma.offset1716 = add i32 %kdma.source1715, %kdma.intra, !dbg !12
  %213 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1374, i32 %kdma.offset1716, <4 x i32> %dma.buffer.rsrc1376) #10, !dbg !12
  %kdma.source1730 = or disjoint i64 %210, %kdma.source1583, !dbg !12
  %kdma.source1731 = shl nuw nsw i64 %kdma.source1730, 1, !dbg !12
  %kdma.source1732 = or i64 %kdma.source1731, %.idx1420, !dbg !12
  %kdma.source1733 = trunc i64 %kdma.source1732 to i32, !dbg !12
  %kdma.source1734 = or i32 %kdma.source1733, 128, !dbg !12
  %kdma.offset1735 = add i32 %kdma.source1734, %kdma.intra.hi, !dbg !12
  %214 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1588, i32 %kdma.offset1735, <4 x i32> %dma.buffer.rsrc1376) #10, !dbg !12
  %kdma.gep1841 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8192, !dbg !14
  %215 = load <4 x half>, ptr addrspace(3) %kdma.gep1841, align 8, !dbg !14, !frisk.k_lds_pair !41
  %kdma.gep1845 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8320, !dbg !14
  %216 = load <4 x half>, ptr addrspace(3) %kdma.gep1845, align 8, !dbg !14, !frisk.k_lds_pair !41
  %kdma.gep1842 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12288, !dbg !14
  %217 = load <4 x half>, ptr addrspace(3) %kdma.gep1842, align 8, !dbg !14, !frisk.k_lds_pair !42
  %kdma.gep1846 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12416, !dbg !14
  %218 = load <4 x half>, ptr addrspace(3) %kdma.gep1846, align 8, !dbg !14, !frisk.k_lds_pair !42
  %kdma.gep1843 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1811, i64 8192, !dbg !14
  %219 = load <4 x half>, ptr addrspace(3) %kdma.gep1843, align 8, !dbg !14, !frisk.k_lds_pair !43
  %kdma.gep1847 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1811, i64 8320, !dbg !14
  %220 = load <4 x half>, ptr addrspace(3) %kdma.gep1847, align 8, !dbg !14, !frisk.k_lds_pair !43
  %kdma.gep1844 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1811, i64 12288, !dbg !14
  %221 = load <4 x half>, ptr addrspace(3) %kdma.gep1844, align 8, !dbg !14, !frisk.k_lds_pair !44
  %kdma.gep1848 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1811, i64 12416, !dbg !14
  %222 = load <4 x half>, ptr addrspace(3) %kdma.gep1848, align 8, !dbg !14, !frisk.k_lds_pair !44
  %kdma.gep1849 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9216, !dbg !14
  %223 = load <4 x half>, ptr addrspace(3) %kdma.gep1849, align 8, !dbg !14, !frisk.k_lds_pair !45
  %kdma.gep1853 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9344, !dbg !14
  %224 = load <4 x half>, ptr addrspace(3) %kdma.gep1853, align 8, !dbg !14, !frisk.k_lds_pair !45
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.k_lds_prefetch !1
  %225 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %215, <4 x float> %167) #9, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.k_lds_prefetch !1
  %kdma.gep1850 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13312, !dbg !14
  %226 = load <4 x half>, ptr addrspace(3) %kdma.gep1850, align 8, !dbg !14, !frisk.k_lds_pair !46
  %kdma.gep1854 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13440, !dbg !14
  %227 = load <4 x half>, ptr addrspace(3) %kdma.gep1854, align 8, !dbg !14, !frisk.k_lds_pair !46
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.k_lds_prefetch !1
  %228 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %217, <4 x float> %225) #9, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.k_lds_prefetch !1
  %kdma.gep1851 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1811, i64 9216, !dbg !14
  %229 = load <4 x half>, ptr addrspace(3) %kdma.gep1851, align 8, !dbg !14, !frisk.k_lds_pair !47
  %kdma.gep1855 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1811, i64 9344, !dbg !14
  %230 = load <4 x half>, ptr addrspace(3) %kdma.gep1855, align 8, !dbg !14, !frisk.k_lds_pair !47
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.k_lds_prefetch !1
  %231 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %219, <4 x float> %173) #9, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.k_lds_prefetch !1
  %kdma.gep1852 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1811, i64 13312, !dbg !14
  %232 = load <4 x half>, ptr addrspace(3) %kdma.gep1852, align 8, !dbg !14, !frisk.k_lds_pair !48
  %kdma.gep1856 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1811, i64 13440, !dbg !14
  %233 = load <4 x half>, ptr addrspace(3) %kdma.gep1856, align 8, !dbg !14, !frisk.k_lds_pair !48
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.k_lds_prefetch !1
  %234 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %221, <4 x float> %231) #9, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.k_lds_prefetch !1
  %235 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %216, <4 x float> %177) #9, !dbg !14
  %236 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %218, <4 x float> %235) #9, !dbg !14
  %237 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %220, <4 x float> %179) #9, !dbg !14
  %238 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %222, <4 x float> %237) #9, !dbg !14
  %239 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %223, <4 x float> %181) #9, !dbg !14
  %240 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %226, <4 x float> %239) #9, !dbg !14
  %241 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %229, <4 x float> %183) #9, !dbg !14
  %242 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %232, <4 x float> %241) #9, !dbg !14
  %243 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %224, <4 x float> %185) #9, !dbg !14
  %244 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %227, <4 x float> %243) #9, !dbg !14
  %245 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %230, <4 x float> %187) #9, !dbg !14
  %246 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %233, <4 x float> %245) #9, !dbg !14
  %247 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %215, <4 x float> %189) #9, !dbg !14
  %248 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %217, <4 x float> %247) #9, !dbg !14
  %249 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %219, <4 x float> %191) #9, !dbg !14
  %250 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %221, <4 x float> %249) #9, !dbg !14
  %251 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %216, <4 x float> %193) #9, !dbg !14
  %252 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %218, <4 x float> %251) #9, !dbg !14
  %253 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %220, <4 x float> %195) #9, !dbg !14
  %254 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %222, <4 x float> %253) #9, !dbg !14
  %255 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %223, <4 x float> %197) #9, !dbg !14
  %256 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %226, <4 x float> %255) #9, !dbg !14
  %257 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %229, <4 x float> %199) #9, !dbg !14
  %258 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %232, <4 x float> %257) #9, !dbg !14
  %259 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %224, <4 x float> %201) #9, !dbg !14
  %260 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %227, <4 x float> %259) #9, !dbg !14
  %261 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %230, <4 x float> %203) #9, !dbg !14
  %262 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %233, <4 x float> %261) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %263 = or disjoint i64 %115, 3072, !dbg !12
  %264 = sub nuw nsw i64 -3073, %115, !dbg !12
  %265 = select i1 %117, i64 %264, i64 %263, !dbg !12
  %266 = sdiv i64 %265, 4096, !dbg !12
  %267 = xor i64 %266, %121, !dbg !12
  %268 = shl i64 %267, 19, !dbg !12
  %269 = and i64 %268, 16252928, !dbg !12
  %kdma.source1748 = or disjoint i64 %269, %kdma.source1535, !dbg !12
  %kdma.source1749 = shl nuw nsw i64 %kdma.source1748, 1, !dbg !12
  %kdma.source1750 = or i64 %kdma.source1749, %.idx1420, !dbg !12
  %kdma.source1751 = trunc i64 %kdma.source1750 to i32, !dbg !12
  %kdma.source1752 = or i32 %kdma.source1751, 192, !dbg !12
  %kdma.offset1753 = add i32 %kdma.source1752, %kdma.intra, !dbg !12
  %270 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1371, i32 %kdma.offset1753, <4 x i32> %dma.buffer.rsrc1376) #10, !dbg !12
  %kdma.source1766 = or disjoint i64 %269, %kdma.source1549, !dbg !12
  %kdma.source1767 = shl nuw nsw i64 %kdma.source1766, 1, !dbg !12
  %kdma.source1768 = or i64 %kdma.source1767, %.idx1420, !dbg !12
  %kdma.source1769 = trunc i64 %kdma.source1768 to i32, !dbg !12
  %kdma.source1770 = or i32 %kdma.source1769, 192, !dbg !12
  %kdma.offset1771 = add i32 %kdma.source1770, %kdma.intra.hi, !dbg !12
  %271 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1624, i32 %kdma.offset1771, <4 x i32> %dma.buffer.rsrc1376) #10, !dbg !12
  %kdma.source1785 = or disjoint i64 %269, %kdma.source1566, !dbg !12
  %kdma.source1786 = shl nuw nsw i64 %kdma.source1785, 1, !dbg !12
  %kdma.source1787 = or i64 %kdma.source1786, %.idx1420, !dbg !12
  %kdma.source1788 = trunc i64 %kdma.source1787 to i32, !dbg !12
  %kdma.source1789 = or i32 %kdma.source1788, 192, !dbg !12
  %kdma.offset1790 = add i32 %kdma.source1789, %kdma.intra, !dbg !12
  %272 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1368, i32 %kdma.offset1790, <4 x i32> %dma.buffer.rsrc1376) #10, !dbg !12
  %kdma.source1804 = or disjoint i64 %269, %kdma.source1583, !dbg !12
  %kdma.source1805 = shl nuw nsw i64 %kdma.source1804, 1, !dbg !12
  %kdma.source1806 = or i64 %kdma.source1805, %.idx1420, !dbg !12
  %kdma.source1807 = trunc i64 %kdma.source1806 to i32, !dbg !12
  %kdma.source1808 = or i32 %kdma.source1807, 192, !dbg !12
  %kdma.offset1809 = add i32 %kdma.source1808, %kdma.intra.hi, !dbg !12
  %273 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1662, i32 %kdma.offset1809, <4 x i32> %dma.buffer.rsrc1376) #10, !dbg !12
  %274 = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8, !dbg !14, !frisk.k_lds_pair !49
  %275 = load <4 x half>, ptr addrspace(3) %kdma.gep1813, align 8, !dbg !14, !frisk.k_lds_pair !49
  %276 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8, !dbg !14, !frisk.k_lds_pair !50
  %277 = load <4 x half>, ptr addrspace(3) %kdma.gep1814, align 8, !dbg !14, !frisk.k_lds_pair !50
  %278 = load <4 x half>, ptr addrspace(3) %kdma.swizzled1811, align 8, !dbg !14, !frisk.k_lds_pair !51
  %279 = load <4 x half>, ptr addrspace(3) %kdma.gep1815, align 8, !dbg !14, !frisk.k_lds_pair !51
  %280 = load <4 x half>, ptr addrspace(3) %kdma.gep1812, align 8, !dbg !14, !frisk.k_lds_pair !52
  %281 = load <4 x half>, ptr addrspace(3) %kdma.gep1816, align 8, !dbg !14, !frisk.k_lds_pair !52
  %282 = load <4 x half>, ptr addrspace(3) %kdma.gep1817, align 8, !dbg !14, !frisk.k_lds_pair !53
  %283 = load <4 x half>, ptr addrspace(3) %kdma.gep1821, align 8, !dbg !14, !frisk.k_lds_pair !53
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.k_lds_prefetch !1
  %284 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %274, <4 x float> %228) #9, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.k_lds_prefetch !1
  %285 = load <4 x half>, ptr addrspace(3) %kdma.gep1818, align 8, !dbg !14, !frisk.k_lds_pair !54
  %286 = load <4 x half>, ptr addrspace(3) %kdma.gep1822, align 8, !dbg !14, !frisk.k_lds_pair !54
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.k_lds_prefetch !1
  %287 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %276, <4 x float> %284) #9, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.k_lds_prefetch !1
  %288 = load <4 x half>, ptr addrspace(3) %kdma.gep1819, align 8, !dbg !14, !frisk.k_lds_pair !55
  %289 = load <4 x half>, ptr addrspace(3) %kdma.gep1823, align 8, !dbg !14, !frisk.k_lds_pair !55
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.k_lds_prefetch !1
  %290 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %278, <4 x float> %234) #9, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.k_lds_prefetch !1
  %291 = load <4 x half>, ptr addrspace(3) %kdma.gep1820, align 8, !dbg !14, !frisk.k_lds_pair !56
  %292 = load <4 x half>, ptr addrspace(3) %kdma.gep1824, align 8, !dbg !14, !frisk.k_lds_pair !56
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.k_lds_prefetch !1
  %293 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %280, <4 x float> %290) #9, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.k_lds_prefetch !1
  %294 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %275, <4 x float> %236) #9, !dbg !14
  %295 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %277, <4 x float> %294) #9, !dbg !14
  %296 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %279, <4 x float> %238) #9, !dbg !14
  %297 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %281, <4 x float> %296) #9, !dbg !14
  %298 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %282, <4 x float> %240) #9, !dbg !14
  %299 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %285, <4 x float> %298) #9, !dbg !14
  %300 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %288, <4 x float> %242) #9, !dbg !14
  %301 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %291, <4 x float> %300) #9, !dbg !14
  %302 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %283, <4 x float> %244) #9, !dbg !14
  %303 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %286, <4 x float> %302) #9, !dbg !14
  %304 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %289, <4 x float> %246) #9, !dbg !14
  %305 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %292, <4 x float> %304) #9, !dbg !14
  %306 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %274, <4 x float> %248) #9, !dbg !14
  %307 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %276, <4 x float> %306) #9, !dbg !14
  %308 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %278, <4 x float> %250) #9, !dbg !14
  %309 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %280, <4 x float> %308) #9, !dbg !14
  %310 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %275, <4 x float> %252) #9, !dbg !14
  %311 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %277, <4 x float> %310) #9, !dbg !14
  %312 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %279, <4 x float> %254) #9, !dbg !14
  %313 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %281, <4 x float> %312) #9, !dbg !14
  %314 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %282, <4 x float> %256) #9, !dbg !14
  %315 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %285, <4 x float> %314) #9, !dbg !14
  %316 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %288, <4 x float> %258) #9, !dbg !14
  %317 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %291, <4 x float> %316) #9, !dbg !14
  %318 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %283, <4 x float> %260) #9, !dbg !14
  %319 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %286, <4 x float> %318) #9, !dbg !14
  %320 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %289, <4 x float> %262) #9, !dbg !14
  %321 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %292, <4 x float> %320) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %address.local920 = shl nuw nsw i64 %address.local502, 5, !dbg !15
  %address.local921 = and i64 %address.local920, 8064, !dbg !15
  %address.local923 = and i64 %address.local503, 24, !dbg !15
  %address.local924 = or disjoint i64 %address.local921, %address.local923, !dbg !15
  %322 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %.idx14331434 = or disjoint i64 %108, %address.local924, !dbg !15
  %.idx14331434.tr = trunc nuw i64 %.idx14331434 to i32, !dbg !15
  %dma.buffer.offset1351 = shl nuw nsw i32 %.idx14331434.tr, 1, !dbg !15
  %323 = insertelement <2 x i64> poison, i64 %322, i64 0, !dbg !15
  %324 = bitcast <2 x i64> %323 to <4 x i32>, !dbg !15
  %dma.buffer.rsrc1352 = shufflevector <4 x i32> %324, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %325 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1377, i32 %dma.buffer.offset1351, <4 x i32> %dma.buffer.rsrc1352) #10, !dbg !15
  %address.local926 = shl nuw nsw i64 %address.local519, 5, !dbg !15
  %address.local927 = and i64 %address.local926, 16256, !dbg !15
  %address.local930 = or disjoint i64 %address.local927, %address.local923, !dbg !15
  %.idx14351436 = or disjoint i64 %108, %address.local930, !dbg !15
  %.idx14351436.tr = trunc nuw i64 %.idx14351436 to i32, !dbg !15
  %dma.buffer.offset1348 = shl nuw nsw i32 %.idx14351436.tr, 1, !dbg !15
  %326 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1374, i32 %dma.buffer.offset1348, <4 x i32> %dma.buffer.rsrc1352) #10, !dbg !15
  %327 = load <4 x half>, ptr addrspace(3) %kdma.gep1841, align 8, !dbg !14, !frisk.k_tail_pair !35
  %328 = load <4 x half>, ptr addrspace(3) %kdma.gep1845, align 8, !dbg !14, !frisk.k_tail_pair !35
  %329 = load <4 x half>, ptr addrspace(3) %kdma.gep1842, align 8, !dbg !14, !frisk.k_tail_pair !36
  %330 = load <4 x half>, ptr addrspace(3) %kdma.gep1846, align 8, !dbg !14, !frisk.k_tail_pair !36
  %331 = load <4 x half>, ptr addrspace(3) %kdma.gep1843, align 8, !dbg !14, !frisk.k_tail_pair !37
  %332 = load <4 x half>, ptr addrspace(3) %kdma.gep1847, align 8, !dbg !14, !frisk.k_tail_pair !37
  %333 = load <4 x half>, ptr addrspace(3) %kdma.gep1844, align 8, !dbg !14, !frisk.k_tail_pair !38
  %334 = load <4 x half>, ptr addrspace(3) %kdma.gep1848, align 8, !dbg !14, !frisk.k_tail_pair !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.k_tail_prefetch !1
  %335 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %327, <4 x float> %287) #9, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.k_tail_prefetch !1
  %336 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %329, <4 x float> %335) #9, !dbg !14
  %337 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %331, <4 x float> %293) #9, !dbg !14
  %338 = extractelement <4 x float> %336, i64 0, !dbg !19
  %339 = call float @llvm.maxnum.f32(float %338, float 0xFFF0000000000000), !dbg !20, !frisk.local_reduction !1
  %340 = extractelement <4 x float> %336, i64 1, !dbg !19
  %341 = extractelement <4 x float> %336, i64 2, !dbg !19
  %342 = extractelement <4 x float> %336, i64 3, !dbg !19
  %343 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %333, <4 x float> %337) #9, !dbg !14
  %344 = load <4 x half>, ptr addrspace(3) %kdma.gep1849, align 8, !dbg !14, !frisk.k_tail_pair !33
  %345 = load <4 x half>, ptr addrspace(3) %kdma.gep1853, align 8, !dbg !14, !frisk.k_tail_pair !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.k_tail_prefetch !1
  %346 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %328, <4 x float> %295) #9, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !19, !frisk.k_tail_prefetch !1
  %347 = extractelement <4 x float> %343, i64 0, !dbg !19
  %local.reduce154 = call float @llvm.maxnum.f32(float %339, float %347), !dbg !20, !frisk.local_reduction !1
  %348 = extractelement <4 x float> %343, i64 1, !dbg !19
  %local.reduce155 = call float @llvm.maxnum.f32(float %340, float %348), !dbg !20, !frisk.local_reduction !1
  %349 = extractelement <4 x float> %343, i64 2, !dbg !19
  %local.reduce156 = call float @llvm.maxnum.f32(float %341, float %349), !dbg !20, !frisk.local_reduction !1
  %350 = extractelement <4 x float> %343, i64 3, !dbg !19
  %local.reduce157 = call float @llvm.maxnum.f32(float %342, float %350), !dbg !20, !frisk.local_reduction !1
  %351 = load <4 x half>, ptr addrspace(3) %kdma.gep1850, align 8, !dbg !14, !frisk.k_tail_pair !39
  %352 = load <4 x half>, ptr addrspace(3) %kdma.gep1854, align 8, !dbg !14, !frisk.k_tail_pair !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.k_tail_prefetch !1
  %353 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %330, <4 x float> %346) #9, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.k_tail_prefetch !1
  %354 = load <4 x half>, ptr addrspace(3) %kdma.gep1851, align 8, !dbg !14, !frisk.k_tail_pair !40
  %355 = load <4 x half>, ptr addrspace(3) %kdma.gep1855, align 8, !dbg !14, !frisk.k_tail_pair !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.k_tail_prefetch !1
  %356 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %332, <4 x float> %297) #9, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !19, !frisk.k_tail_prefetch !1
  %357 = extractelement <4 x float> %353, i64 0, !dbg !19
  %local.reduce158 = call float @llvm.maxnum.f32(float %local.reduce154, float %357), !dbg !20, !frisk.local_reduction !1
  %358 = extractelement <4 x float> %353, i64 1, !dbg !19
  %local.reduce159 = call float @llvm.maxnum.f32(float %local.reduce155, float %358), !dbg !20, !frisk.local_reduction !1
  %359 = extractelement <4 x float> %353, i64 2, !dbg !19
  %local.reduce160 = call float @llvm.maxnum.f32(float %local.reduce156, float %359), !dbg !20, !frisk.local_reduction !1
  %360 = extractelement <4 x float> %353, i64 3, !dbg !19
  %local.reduce161 = call float @llvm.maxnum.f32(float %local.reduce157, float %360), !dbg !20, !frisk.local_reduction !1
  %361 = load <4 x half>, ptr addrspace(3) %kdma.gep1852, align 8, !dbg !14, !frisk.k_tail_pair !41
  %362 = load <4 x half>, ptr addrspace(3) %kdma.gep1856, align 8, !dbg !14, !frisk.k_tail_pair !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.k_tail_prefetch !1
  %363 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %334, <4 x float> %356) #9, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.k_tail_prefetch !1
  %364 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %344, <4 x float> %299) #9, !dbg !14
  %365 = extractelement <4 x float> %363, i64 0, !dbg !19
  %local.reduce162 = call float @llvm.maxnum.f32(float %local.reduce158, float %365), !dbg !20, !frisk.local_reduction !1
  %366 = extractelement <4 x float> %363, i64 1, !dbg !19
  %local.reduce163 = call float @llvm.maxnum.f32(float %local.reduce159, float %366), !dbg !20, !frisk.local_reduction !1
  %367 = extractelement <4 x float> %363, i64 2, !dbg !19
  %local.reduce164 = call float @llvm.maxnum.f32(float %local.reduce160, float %367), !dbg !20, !frisk.local_reduction !1
  %368 = extractelement <4 x float> %363, i64 3, !dbg !19
  %local.reduce165 = call float @llvm.maxnum.f32(float %local.reduce161, float %368), !dbg !20, !frisk.local_reduction !1
  %369 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %351, <4 x float> %364) #9, !dbg !14
  %370 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %354, <4 x float> %301) #9, !dbg !14
  %371 = extractelement <4 x float> %369, i64 0, !dbg !19
  %local.reduce166 = call float @llvm.maxnum.f32(float %local.reduce162, float %371), !dbg !20, !frisk.local_reduction !1
  %372 = extractelement <4 x float> %369, i64 1, !dbg !19
  %local.reduce167 = call float @llvm.maxnum.f32(float %local.reduce163, float %372), !dbg !20, !frisk.local_reduction !1
  %373 = extractelement <4 x float> %369, i64 2, !dbg !19
  %local.reduce168 = call float @llvm.maxnum.f32(float %local.reduce164, float %373), !dbg !20, !frisk.local_reduction !1
  %374 = extractelement <4 x float> %369, i64 3, !dbg !19
  %local.reduce169 = call float @llvm.maxnum.f32(float %local.reduce165, float %374), !dbg !20, !frisk.local_reduction !1
  %375 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %361, <4 x float> %370) #9, !dbg !14
  %376 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %345, <4 x float> %303) #9, !dbg !14
  %377 = extractelement <4 x float> %375, i64 0, !dbg !19
  %local.reduce170 = call float @llvm.maxnum.f32(float %local.reduce166, float %377), !dbg !20, !frisk.local_reduction !1
  %378 = extractelement <4 x float> %375, i64 1, !dbg !19
  %local.reduce171 = call float @llvm.maxnum.f32(float %local.reduce167, float %378), !dbg !20, !frisk.local_reduction !1
  %379 = extractelement <4 x float> %375, i64 2, !dbg !19
  %local.reduce172 = call float @llvm.maxnum.f32(float %local.reduce168, float %379), !dbg !20, !frisk.local_reduction !1
  %380 = extractelement <4 x float> %375, i64 3, !dbg !19
  %local.reduce173 = call float @llvm.maxnum.f32(float %local.reduce169, float %380), !dbg !20, !frisk.local_reduction !1
  %381 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %352, <4 x float> %376) #9, !dbg !14
  %382 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %355, <4 x float> %305) #9, !dbg !14
  %383 = extractelement <4 x float> %381, i64 0, !dbg !19
  %local.reduce174 = call float @llvm.maxnum.f32(float %local.reduce170, float %383), !dbg !20, !frisk.local_reduction !1
  %384 = extractelement <4 x float> %381, i64 1, !dbg !19
  %local.reduce175 = call float @llvm.maxnum.f32(float %local.reduce171, float %384), !dbg !20, !frisk.local_reduction !1
  %385 = extractelement <4 x float> %381, i64 2, !dbg !19
  %local.reduce176 = call float @llvm.maxnum.f32(float %local.reduce172, float %385), !dbg !20, !frisk.local_reduction !1
  %386 = extractelement <4 x float> %381, i64 3, !dbg !19
  %local.reduce177 = call float @llvm.maxnum.f32(float %local.reduce173, float %386), !dbg !20, !frisk.local_reduction !1
  %387 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %362, <4 x float> %382) #9, !dbg !14
  %388 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %327, <4 x float> %307) #9, !dbg !14
  %389 = extractelement <4 x float> %387, i64 0, !dbg !19
  %local.reduce178 = call float @llvm.maxnum.f32(float %local.reduce174, float %389), !dbg !20, !frisk.local_reduction !1
  %390 = extractelement <4 x float> %387, i64 1, !dbg !19
  %local.reduce179 = call float @llvm.maxnum.f32(float %local.reduce175, float %390), !dbg !20, !frisk.local_reduction !1
  %391 = extractelement <4 x float> %387, i64 2, !dbg !19
  %local.reduce180 = call float @llvm.maxnum.f32(float %local.reduce176, float %391), !dbg !20, !frisk.local_reduction !1
  %392 = extractelement <4 x float> %387, i64 3, !dbg !19
  %local.reduce181 = call float @llvm.maxnum.f32(float %local.reduce177, float %392), !dbg !20, !frisk.local_reduction !1
  %local.reduce182 = call float @llvm.maxnum.f32(float %local.reduce178, float %local.reduce179), !dbg !20, !frisk.local_reduction !1
  %local.reduce183 = call float @llvm.maxnum.f32(float %local.reduce180, float %local.reduce181), !dbg !20, !frisk.local_reduction !1
  %393 = call float @llvm.maxnum.f32(float %local.reduce182, float %local.reduce183), !dbg !20, !frisk.local_reduction !1
  %394 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %329, <4 x float> %388) #9, !dbg !14
  %395 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %331, <4 x float> %309) #9, !dbg !14
  %396 = extractelement <4 x float> %394, i64 0, !dbg !19
  %397 = extractelement <4 x float> %394, i64 1, !dbg !19
  %398 = extractelement <4 x float> %394, i64 2, !dbg !19
  %399 = extractelement <4 x float> %394, i64 3, !dbg !19
  %400 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %333, <4 x float> %395) #9, !dbg !14
  %401 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %328, <4 x float> %311) #9, !dbg !14
  %402 = extractelement <4 x float> %400, i64 0, !dbg !19
  %403 = extractelement <4 x float> %400, i64 1, !dbg !19
  %404 = extractelement <4 x float> %400, i64 2, !dbg !19
  %405 = extractelement <4 x float> %400, i64 3, !dbg !19
  %406 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %330, <4 x float> %401) #9, !dbg !14
  %407 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %332, <4 x float> %313) #9, !dbg !14
  %408 = extractelement <4 x float> %406, i64 0, !dbg !19
  %409 = extractelement <4 x float> %406, i64 1, !dbg !19
  %410 = extractelement <4 x float> %406, i64 2, !dbg !19
  %411 = extractelement <4 x float> %406, i64 3, !dbg !19
  %412 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %334, <4 x float> %407) #9, !dbg !14
  %413 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %344, <4 x float> %315) #9, !dbg !14
  %414 = extractelement <4 x float> %412, i64 0, !dbg !19
  %415 = extractelement <4 x float> %412, i64 1, !dbg !19
  %416 = extractelement <4 x float> %412, i64 2, !dbg !19
  %417 = extractelement <4 x float> %412, i64 3, !dbg !19
  %418 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %351, <4 x float> %413) #9, !dbg !14
  %419 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %354, <4 x float> %317) #9, !dbg !14
  %420 = extractelement <4 x float> %418, i64 0, !dbg !19
  %421 = extractelement <4 x float> %418, i64 1, !dbg !19
  %422 = extractelement <4 x float> %418, i64 2, !dbg !19
  %423 = extractelement <4 x float> %418, i64 3, !dbg !19
  %424 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %361, <4 x float> %419) #9, !dbg !14
  %425 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %345, <4 x float> %319) #9, !dbg !14
  %426 = extractelement <4 x float> %424, i64 0, !dbg !19
  %427 = extractelement <4 x float> %424, i64 1, !dbg !19
  %428 = extractelement <4 x float> %424, i64 2, !dbg !19
  %429 = extractelement <4 x float> %424, i64 3, !dbg !19
  %430 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %352, <4 x float> %425) #9, !dbg !14
  %431 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %355, <4 x float> %321) #9, !dbg !14
  %432 = extractelement <4 x float> %430, i64 0, !dbg !19
  %433 = extractelement <4 x float> %430, i64 1, !dbg !19
  %434 = extractelement <4 x float> %430, i64 2, !dbg !19
  %435 = extractelement <4 x float> %430, i64 3, !dbg !19
  %436 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %362, <4 x float> %431) #9, !dbg !14
  %437 = extractelement <4 x float> %436, i64 0, !dbg !19
  %438 = extractelement <4 x float> %436, i64 1, !dbg !19
  %439 = extractelement <4 x float> %436, i64 2, !dbg !19
  %440 = extractelement <4 x float> %436, i64 3, !dbg !19
  %441 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !20
  %442 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %441), !dbg !20
  %443 = add nuw nsw i32 %442, 64, !dbg !20
  %444 = xor i32 %442, 16, !dbg !20
  %445 = and i32 %443, 192, !dbg !20
  %446 = icmp samesign ult i32 %444, %445, !dbg !20
  %447 = select i1 %446, i32 %444, i32 %442, !dbg !20
  %448 = shl nuw nsw i32 %447, 2, !dbg !20
  %449 = bitcast float %393 to i32, !dbg !20
  %450 = call i32 @llvm.amdgcn.ds.bpermute(i32 %448, i32 %449), !dbg !20
  %451 = bitcast i32 %450 to float, !dbg !20
  %452 = call float @llvm.maxnum.f32(float %393, float %451), !dbg !20
  %453 = xor i32 %442, 32, !dbg !20
  %454 = icmp samesign ult i32 %453, %445, !dbg !20
  %455 = select i1 %454, i32 %453, i32 %442, !dbg !20
  %456 = shl nuw nsw i32 %455, 2, !dbg !20
  %457 = bitcast float %452 to i32, !dbg !20
  %458 = call i32 @llvm.amdgcn.ds.bpermute(i32 %456, i32 %457), !dbg !20
  %459 = bitcast i32 %458 to float, !dbg !20
  %460 = call float @llvm.maxnum.f32(float %452, float %459), !dbg !20
  %461 = call float @llvm.maxnum.f32(float %396, float 0xFFF0000000000000), !dbg !20, !frisk.local_reduction !1
  %local.reduce123 = call float @llvm.maxnum.f32(float %461, float %402), !dbg !20, !frisk.local_reduction !1
  %local.reduce124 = call float @llvm.maxnum.f32(float %397, float %403), !dbg !20, !frisk.local_reduction !1
  %local.reduce125 = call float @llvm.maxnum.f32(float %398, float %404), !dbg !20, !frisk.local_reduction !1
  %local.reduce126 = call float @llvm.maxnum.f32(float %399, float %405), !dbg !20, !frisk.local_reduction !1
  %local.reduce127 = call float @llvm.maxnum.f32(float %local.reduce123, float %408), !dbg !20, !frisk.local_reduction !1
  %local.reduce128 = call float @llvm.maxnum.f32(float %local.reduce124, float %409), !dbg !20, !frisk.local_reduction !1
  %local.reduce129 = call float @llvm.maxnum.f32(float %local.reduce125, float %410), !dbg !20, !frisk.local_reduction !1
  %local.reduce130 = call float @llvm.maxnum.f32(float %local.reduce126, float %411), !dbg !20, !frisk.local_reduction !1
  %local.reduce131 = call float @llvm.maxnum.f32(float %local.reduce127, float %414), !dbg !20, !frisk.local_reduction !1
  %local.reduce132 = call float @llvm.maxnum.f32(float %local.reduce128, float %415), !dbg !20, !frisk.local_reduction !1
  %local.reduce133 = call float @llvm.maxnum.f32(float %local.reduce129, float %416), !dbg !20, !frisk.local_reduction !1
  %local.reduce134 = call float @llvm.maxnum.f32(float %local.reduce130, float %417), !dbg !20, !frisk.local_reduction !1
  %local.reduce135 = call float @llvm.maxnum.f32(float %local.reduce131, float %420), !dbg !20, !frisk.local_reduction !1
  %local.reduce136 = call float @llvm.maxnum.f32(float %local.reduce132, float %421), !dbg !20, !frisk.local_reduction !1
  %local.reduce137 = call float @llvm.maxnum.f32(float %local.reduce133, float %422), !dbg !20, !frisk.local_reduction !1
  %local.reduce138 = call float @llvm.maxnum.f32(float %local.reduce134, float %423), !dbg !20, !frisk.local_reduction !1
  %local.reduce139 = call float @llvm.maxnum.f32(float %local.reduce135, float %426), !dbg !20, !frisk.local_reduction !1
  %local.reduce140 = call float @llvm.maxnum.f32(float %local.reduce136, float %427), !dbg !20, !frisk.local_reduction !1
  %local.reduce141 = call float @llvm.maxnum.f32(float %local.reduce137, float %428), !dbg !20, !frisk.local_reduction !1
  %local.reduce142 = call float @llvm.maxnum.f32(float %local.reduce138, float %429), !dbg !20, !frisk.local_reduction !1
  %local.reduce143 = call float @llvm.maxnum.f32(float %local.reduce139, float %432), !dbg !20, !frisk.local_reduction !1
  %local.reduce144 = call float @llvm.maxnum.f32(float %local.reduce140, float %433), !dbg !20, !frisk.local_reduction !1
  %local.reduce145 = call float @llvm.maxnum.f32(float %local.reduce141, float %434), !dbg !20, !frisk.local_reduction !1
  %local.reduce146 = call float @llvm.maxnum.f32(float %local.reduce142, float %435), !dbg !20, !frisk.local_reduction !1
  %local.reduce147 = call float @llvm.maxnum.f32(float %local.reduce143, float %437), !dbg !20, !frisk.local_reduction !1
  %local.reduce148 = call float @llvm.maxnum.f32(float %local.reduce144, float %438), !dbg !20, !frisk.local_reduction !1
  %local.reduce149 = call float @llvm.maxnum.f32(float %local.reduce145, float %439), !dbg !20, !frisk.local_reduction !1
  %local.reduce150 = call float @llvm.maxnum.f32(float %local.reduce146, float %440), !dbg !20, !frisk.local_reduction !1
  %local.reduce151 = call float @llvm.maxnum.f32(float %local.reduce147, float %local.reduce148), !dbg !20, !frisk.local_reduction !1
  %local.reduce152 = call float @llvm.maxnum.f32(float %local.reduce149, float %local.reduce150), !dbg !20, !frisk.local_reduction !1
  %462 = call float @llvm.maxnum.f32(float %local.reduce151, float %local.reduce152), !dbg !20, !frisk.local_reduction !1
  %463 = bitcast float %462 to i32, !dbg !20
  %464 = call i32 @llvm.amdgcn.ds.bpermute(i32 %448, i32 %463), !dbg !20
  %465 = bitcast i32 %464 to float, !dbg !20
  %466 = call float @llvm.maxnum.f32(float %462, float %465), !dbg !20
  %467 = bitcast float %466 to i32, !dbg !20
  %468 = call i32 @llvm.amdgcn.ds.bpermute(i32 %456, i32 %467), !dbg !20
  %469 = bitcast i32 %468 to float, !dbg !20
  %470 = call float @llvm.maxnum.f32(float %466, float %469), !dbg !20
  %471 = extractvalue [2 x <1 x float>] %100, 0, !dbg !20
  %472 = insertelement <1 x float> poison, float %460, i64 0, !dbg !20
  %473 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %472, <1 x float> %471), !dbg !20
  %474 = extractvalue [2 x <1 x float>] %100, 1, !dbg !20
  %475 = insertelement <1 x float> poison, float %470, i64 0, !dbg !20
  %476 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %475, <1 x float> %474), !dbg !20
  %477 = shufflevector <1 x float> %473, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !21
  %478 = shufflevector <1 x float> %476, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !21
  %479 = fsub <1 x float> %471, %473, !dbg !22
  %480 = extractelement <1 x float> %479, i64 0, !dbg !23
  %481 = call float @llvm.amdgcn.exp2.f32(float %480), !dbg !23
  %482 = insertelement <4 x float> poison, float %481, i64 0, !dbg !28
  %483 = shufflevector <4 x float> %482, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !28
  %484 = extractvalue [2 x <1 x float>] %99, 0, !dbg !24
  %485 = extractelement <1 x float> %484, i64 0, !dbg !24
  %486 = fmul float %485, %481, !dbg !24
  %487 = fsub <1 x float> %474, %476, !dbg !22
  %488 = extractelement <1 x float> %487, i64 0, !dbg !23
  %489 = call float @llvm.amdgcn.exp2.f32(float %488), !dbg !23
  %490 = insertelement <4 x float> poison, float %489, i64 0, !dbg !28
  %491 = shufflevector <4 x float> %490, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !28
  %492 = extractvalue [2 x <1 x float>] %99, 1, !dbg !24
  %493 = extractelement <1 x float> %492, i64 0, !dbg !24
  %494 = fmul float %493, %489, !dbg !24
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %vdb.address1450 = or disjoint i64 %108, 32, !dbg !15
  %vdb.address1451 = or disjoint i64 %address.local924, %vdb.address1450, !dbg !15
  %vdb.address1452 = trunc nuw i64 %vdb.address1451 to i32, !dbg !15
  %vdb.address1453 = shl nuw nsw i32 %vdb.address1452, 1, !dbg !15
  %vdb.address1461 = or disjoint i64 %address.local930, %vdb.address1450, !dbg !15
  %vdb.address1462 = trunc nuw i64 %vdb.address1461 to i32, !dbg !15
  %vdb.address1463 = shl nuw nsw i32 %vdb.address1462, 1, !dbg !15
  %vdb.dma.base = add i32 %dma.buffer.lds1377, 8192, !dbg !13
  %495 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1453, <4 x i32> %dma.buffer.rsrc1352) #10, !dbg !15
  %vdb.dma.base1468 = add i32 %dma.buffer.lds1374, 8192, !dbg !13
  %496 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1468, i32 %vdb.address1463, <4 x i32> %dma.buffer.rsrc1352) #10, !dbg !15
  %address.local932 = shl nuw nsw i32 %address.thread501, 3, !dbg !13
  %address.local933 = and i32 %address.local932, 120, !dbg !13
  %address.local935 = and i32 %address.local578, 1536, !dbg !13
  %address.local936 = or disjoint i32 %address.local933, %address.local935, !dbg !13
  %address.local937 = zext nneg i32 %address.local936 to i64, !dbg !13
  %address.local938 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local937, !dbg !13
  %497 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local938, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %498 = bitcast <4 x i32> %497 to <8 x half>, !dbg !13
  %address.local939 = and i64 %address.local502, 60, !dbg !13
  %address.local940 = add nuw nsw i64 %address.local939, %address.local502, !dbg !13
  %address.local942.idx = shl nuw nsw i64 %address.local940, 4, !dbg !13
  %address.local942 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %address.local942.idx, !dbg !13
  %address.local943 = lshr i64 %address.local558, 4, !dbg !13
  %address.local945.idx = mul nuw nsw i64 %address.local943, 768, !dbg !13
  %address.local945 = getelementptr i8, ptr addrspace(3) %address.local942, i64 %address.local945.idx, !dbg !13
  %address.local947.idx = mul nsw i64 %address.local545, -64, !dbg !13
  %address.local947 = getelementptr i8, ptr addrspace(3) %address.local945, i64 %address.local947.idx, !dbg !13
  %address.local948 = getelementptr i8, ptr addrspace(3) %address.local947, i64 256, !dbg !13
  %499 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local948, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %500 = bitcast <4 x i32> %499 to <8 x half>, !dbg !13
  %address.local952 = getelementptr i8, ptr addrspace(3) %address.local947, i64 512, !dbg !13
  %501 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local952, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %502 = bitcast <4 x i32> %501 to <8 x half>, !dbg !13
  %address.local956 = getelementptr i8, ptr addrspace(3) %address.local947, i64 768, !dbg !13
  %503 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local956, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %504 = bitcast <4 x i32> %503 to <8 x half>, !dbg !13
  %address.local958 = getelementptr i8, ptr addrspace(3) %address.local938, i64 4096, !dbg !13
  %505 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local958, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %506 = bitcast <4 x i32> %505 to <8 x half>, !dbg !13
  %address.local962 = getelementptr i8, ptr addrspace(3) %address.local947, i64 4352, !dbg !13
  %507 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local962, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %508 = bitcast <4 x i32> %507 to <8 x half>, !dbg !13
  %address.local966 = getelementptr i8, ptr addrspace(3) %address.local947, i64 4608, !dbg !13
  %509 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local966, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %510 = bitcast <4 x i32> %509 to <8 x half>, !dbg !13
  %address.local970 = getelementptr i8, ptr addrspace(3) %address.local947, i64 4864, !dbg !13
  %511 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local970, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %512 = shufflevector <8 x half> %498, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %513 = extractvalue [2 x <8 x float>] %98, 0, !dbg !28
  %514 = shufflevector <8 x float> %513, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %515 = fmul <4 x float> %514, %483, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %516 = fsub <4 x float> %336, %477, !dbg !21
  %517 = extractelement <4 x float> %516, i64 0, !dbg !25
  %518 = call float @llvm.amdgcn.exp2.f32(float %517), !dbg !25
  %519 = insertelement <4 x float> poison, float %518, i64 0, !dbg !27
  %520 = extractelement <4 x float> %516, i64 1, !dbg !25
  %521 = call float @llvm.amdgcn.exp2.f32(float %520), !dbg !25
  %522 = insertelement <4 x float> %519, float %521, i64 1, !dbg !27
  %523 = extractelement <4 x float> %516, i64 2, !dbg !25
  %524 = call float @llvm.amdgcn.exp2.f32(float %523), !dbg !25
  %525 = insertelement <4 x float> %522, float %524, i64 2, !dbg !27
  %526 = extractelement <4 x float> %516, i64 3, !dbg !25
  %527 = call float @llvm.amdgcn.exp2.f32(float %526), !dbg !25
  %528 = insertelement <4 x float> %525, float %527, i64 3, !dbg !27
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %528) #9, !dbg !27, !frisk.register_lifetimes !1
  %529 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !27
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %529) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %530 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %512, <4 x float> %515) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %531 = shufflevector <8 x half> %498, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %532 = shufflevector <8 x float> %513, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %533 = fmul <4 x float> %532, %483, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %534 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %531, <4 x float> %533) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %535 = extractvalue [2 x <8 x float>] %98, 1, !dbg !28
  %536 = shufflevector <8 x float> %535, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %537 = fmul <4 x float> %536, %491, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %538 = fsub <4 x float> %394, %478, !dbg !21
  %539 = extractelement <4 x float> %538, i64 0, !dbg !25
  %540 = call float @llvm.amdgcn.exp2.f32(float %539), !dbg !25
  %541 = insertelement <4 x float> poison, float %540, i64 0, !dbg !27
  %542 = extractelement <4 x float> %538, i64 1, !dbg !25
  %543 = call float @llvm.amdgcn.exp2.f32(float %542), !dbg !25
  %544 = insertelement <4 x float> %541, float %543, i64 1, !dbg !27
  %545 = extractelement <4 x float> %538, i64 2, !dbg !25
  %546 = call float @llvm.amdgcn.exp2.f32(float %545), !dbg !25
  %547 = insertelement <4 x float> %544, float %546, i64 2, !dbg !27
  %548 = extractelement <4 x float> %538, i64 3, !dbg !25
  %549 = call float @llvm.amdgcn.exp2.f32(float %548), !dbg !25
  %550 = insertelement <4 x float> %547, float %549, i64 3, !dbg !27
  %math.result391 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %550) #9, !dbg !27, !frisk.register_lifetimes !1
  %551 = fptrunc <4 x float> %math.result391 to <4 x half>, !dbg !27
  %math.packed392 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %551) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %552 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed392, <4 x half> %512, <4 x float> %537) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %553 = shufflevector <8 x float> %535, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %554 = fmul <4 x float> %553, %491, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %555 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed392, <4 x half> %531, <4 x float> %554) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %556 = shufflevector <8 x half> %500, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %557 = fsub <4 x float> %343, %477, !dbg !21
  %558 = extractelement <4 x float> %557, i64 0, !dbg !25
  %559 = call float @llvm.amdgcn.exp2.f32(float %558), !dbg !25
  %local.reduce92 = fadd float %518, %559, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor = call float asm sideeffect "", "=v,0"(float %local.reduce92) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %560 = insertelement <4 x float> poison, float %559, i64 0, !dbg !27
  %561 = extractelement <4 x float> %557, i64 1, !dbg !25
  %562 = call float @llvm.amdgcn.exp2.f32(float %561), !dbg !25
  %local.reduce93 = fadd float %521, %562, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1944 = call float asm sideeffect "", "=v,0"(float %local.reduce93) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %563 = insertelement <4 x float> %560, float %562, i64 1, !dbg !27
  %564 = extractelement <4 x float> %557, i64 2, !dbg !25
  %565 = call float @llvm.amdgcn.exp2.f32(float %564), !dbg !25
  %local.reduce94 = fadd float %524, %565, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1945 = call float asm sideeffect "", "=v,0"(float %local.reduce94) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %566 = insertelement <4 x float> %563, float %565, i64 2, !dbg !27
  %567 = extractelement <4 x float> %557, i64 3, !dbg !25
  %568 = call float @llvm.amdgcn.exp2.f32(float %567), !dbg !25
  %local.reduce95 = fadd float %527, %568, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1946 = call float asm sideeffect "", "=v,0"(float %local.reduce95) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %569 = insertelement <4 x float> %566, float %568, i64 3, !dbg !27
  %math.result377 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %569) #9, !dbg !27, !frisk.register_lifetimes !1
  %570 = fptrunc <4 x float> %math.result377 to <4 x half>, !dbg !27
  %math.packed378 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %570) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %571 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed378, <4 x half> %556, <4 x float> %530) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %572 = shufflevector <8 x half> %500, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %573 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed378, <4 x half> %572, <4 x float> %534) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %574 = fsub <4 x float> %400, %478, !dbg !21
  %575 = extractelement <4 x float> %574, i64 0, !dbg !25
  %576 = call float @llvm.amdgcn.exp2.f32(float %575), !dbg !25
  %local.reduce = fadd float %540, %576, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1947 = call float asm sideeffect "", "=v,0"(float %local.reduce) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %577 = insertelement <4 x float> poison, float %576, i64 0, !dbg !27
  %578 = extractelement <4 x float> %574, i64 1, !dbg !25
  %579 = call float @llvm.amdgcn.exp2.f32(float %578), !dbg !25
  %local.reduce62 = fadd float %543, %579, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1948 = call float asm sideeffect "", "=v,0"(float %local.reduce62) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %580 = insertelement <4 x float> %577, float %579, i64 1, !dbg !27
  %581 = extractelement <4 x float> %574, i64 2, !dbg !25
  %582 = call float @llvm.amdgcn.exp2.f32(float %581), !dbg !25
  %local.reduce63 = fadd float %546, %582, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1949 = call float asm sideeffect "", "=v,0"(float %local.reduce63) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %583 = insertelement <4 x float> %580, float %582, i64 2, !dbg !27
  %584 = extractelement <4 x float> %574, i64 3, !dbg !25
  %585 = call float @llvm.amdgcn.exp2.f32(float %584), !dbg !25
  %local.reduce64 = fadd float %549, %585, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1950 = call float asm sideeffect "", "=v,0"(float %local.reduce64) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %586 = insertelement <4 x float> %583, float %585, i64 3, !dbg !27
  %math.result393 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %586) #9, !dbg !27, !frisk.register_lifetimes !1
  %587 = fptrunc <4 x float> %math.result393 to <4 x half>, !dbg !27
  %math.packed394 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %587) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %588 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed394, <4 x half> %556, <4 x float> %552) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %589 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed394, <4 x half> %572, <4 x float> %555) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %590 = shufflevector <8 x half> %502, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %591 = fsub <4 x float> %353, %477, !dbg !21
  %592 = extractelement <4 x float> %591, i64 0, !dbg !25
  %593 = call float @llvm.amdgcn.exp2.f32(float %592), !dbg !25
  %local.reduce96 = fadd float %k0.sum.anchor, %593, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1951 = call float asm sideeffect "", "=v,0"(float %local.reduce96) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %594 = insertelement <4 x float> poison, float %593, i64 0, !dbg !27
  %595 = extractelement <4 x float> %591, i64 1, !dbg !25
  %596 = call float @llvm.amdgcn.exp2.f32(float %595), !dbg !25
  %local.reduce97 = fadd float %k0.sum.anchor1944, %596, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1952 = call float asm sideeffect "", "=v,0"(float %local.reduce97) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %597 = insertelement <4 x float> %594, float %596, i64 1, !dbg !27
  %598 = extractelement <4 x float> %591, i64 2, !dbg !25
  %599 = call float @llvm.amdgcn.exp2.f32(float %598), !dbg !25
  %local.reduce98 = fadd float %k0.sum.anchor1945, %599, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1953 = call float asm sideeffect "", "=v,0"(float %local.reduce98) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %600 = insertelement <4 x float> %597, float %599, i64 2, !dbg !27
  %601 = extractelement <4 x float> %591, i64 3, !dbg !25
  %602 = call float @llvm.amdgcn.exp2.f32(float %601), !dbg !25
  %local.reduce99 = fadd float %k0.sum.anchor1946, %602, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1954 = call float asm sideeffect "", "=v,0"(float %local.reduce99) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %603 = insertelement <4 x float> %600, float %602, i64 3, !dbg !27
  %math.result379 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %603) #9, !dbg !27, !frisk.register_lifetimes !1
  %604 = fptrunc <4 x float> %math.result379 to <4 x half>, !dbg !27
  %math.packed380 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %604) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %605 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed380, <4 x half> %590, <4 x float> %571) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %606 = shufflevector <8 x half> %502, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %607 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed380, <4 x half> %606, <4 x float> %573) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %608 = fsub <4 x float> %406, %478, !dbg !21
  %609 = extractelement <4 x float> %608, i64 0, !dbg !25
  %610 = call float @llvm.amdgcn.exp2.f32(float %609), !dbg !25
  %local.reduce65 = fadd float %k0.sum.anchor1947, %610, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1955 = call float asm sideeffect "", "=v,0"(float %local.reduce65) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %611 = insertelement <4 x float> poison, float %610, i64 0, !dbg !27
  %612 = extractelement <4 x float> %608, i64 1, !dbg !25
  %613 = call float @llvm.amdgcn.exp2.f32(float %612), !dbg !25
  %local.reduce66 = fadd float %k0.sum.anchor1948, %613, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1956 = call float asm sideeffect "", "=v,0"(float %local.reduce66) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %614 = insertelement <4 x float> %611, float %613, i64 1, !dbg !27
  %615 = extractelement <4 x float> %608, i64 2, !dbg !25
  %616 = call float @llvm.amdgcn.exp2.f32(float %615), !dbg !25
  %local.reduce67 = fadd float %k0.sum.anchor1949, %616, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1957 = call float asm sideeffect "", "=v,0"(float %local.reduce67) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %617 = insertelement <4 x float> %614, float %616, i64 2, !dbg !27
  %618 = extractelement <4 x float> %608, i64 3, !dbg !25
  %619 = call float @llvm.amdgcn.exp2.f32(float %618), !dbg !25
  %local.reduce68 = fadd float %k0.sum.anchor1950, %619, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1958 = call float asm sideeffect "", "=v,0"(float %local.reduce68) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %620 = insertelement <4 x float> %617, float %619, i64 3, !dbg !27
  %math.result395 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %620) #9, !dbg !27, !frisk.register_lifetimes !1
  %621 = fptrunc <4 x float> %math.result395 to <4 x half>, !dbg !27
  %math.packed396 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %621) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %622 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed396, <4 x half> %590, <4 x float> %588) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %623 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed396, <4 x half> %606, <4 x float> %589) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %624 = shufflevector <8 x half> %504, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %625 = fsub <4 x float> %363, %477, !dbg !21
  %626 = extractelement <4 x float> %625, i64 0, !dbg !25
  %627 = call float @llvm.amdgcn.exp2.f32(float %626), !dbg !25
  %local.reduce100 = fadd float %k0.sum.anchor1951, %627, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1959 = call float asm sideeffect "", "=v,0"(float %local.reduce100) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %628 = insertelement <4 x float> poison, float %627, i64 0, !dbg !27
  %629 = extractelement <4 x float> %625, i64 1, !dbg !25
  %630 = call float @llvm.amdgcn.exp2.f32(float %629), !dbg !25
  %local.reduce101 = fadd float %k0.sum.anchor1952, %630, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1960 = call float asm sideeffect "", "=v,0"(float %local.reduce101) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %631 = insertelement <4 x float> %628, float %630, i64 1, !dbg !27
  %632 = extractelement <4 x float> %625, i64 2, !dbg !25
  %633 = call float @llvm.amdgcn.exp2.f32(float %632), !dbg !25
  %local.reduce102 = fadd float %k0.sum.anchor1953, %633, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1961 = call float asm sideeffect "", "=v,0"(float %local.reduce102) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %634 = insertelement <4 x float> %631, float %633, i64 2, !dbg !27
  %635 = extractelement <4 x float> %625, i64 3, !dbg !25
  %636 = call float @llvm.amdgcn.exp2.f32(float %635), !dbg !25
  %local.reduce103 = fadd float %k0.sum.anchor1954, %636, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1962 = call float asm sideeffect "", "=v,0"(float %local.reduce103) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %637 = insertelement <4 x float> %634, float %636, i64 3, !dbg !27
  %math.result381 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %637) #9, !dbg !27, !frisk.register_lifetimes !1
  %638 = fptrunc <4 x float> %math.result381 to <4 x half>, !dbg !27
  %math.packed382 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %638) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %639 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed382, <4 x half> %624, <4 x float> %605) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %640 = shufflevector <8 x half> %504, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %641 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed382, <4 x half> %640, <4 x float> %607) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %642 = fsub <4 x float> %412, %478, !dbg !21
  %643 = extractelement <4 x float> %642, i64 0, !dbg !25
  %644 = call float @llvm.amdgcn.exp2.f32(float %643), !dbg !25
  %local.reduce69 = fadd float %k0.sum.anchor1955, %644, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1963 = call float asm sideeffect "", "=v,0"(float %local.reduce69) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %645 = insertelement <4 x float> poison, float %644, i64 0, !dbg !27
  %646 = extractelement <4 x float> %642, i64 1, !dbg !25
  %647 = call float @llvm.amdgcn.exp2.f32(float %646), !dbg !25
  %local.reduce70 = fadd float %k0.sum.anchor1956, %647, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1964 = call float asm sideeffect "", "=v,0"(float %local.reduce70) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %648 = insertelement <4 x float> %645, float %647, i64 1, !dbg !27
  %649 = extractelement <4 x float> %642, i64 2, !dbg !25
  %650 = call float @llvm.amdgcn.exp2.f32(float %649), !dbg !25
  %local.reduce71 = fadd float %k0.sum.anchor1957, %650, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1965 = call float asm sideeffect "", "=v,0"(float %local.reduce71) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %651 = insertelement <4 x float> %648, float %650, i64 2, !dbg !27
  %652 = extractelement <4 x float> %642, i64 3, !dbg !25
  %653 = call float @llvm.amdgcn.exp2.f32(float %652), !dbg !25
  %local.reduce72 = fadd float %k0.sum.anchor1958, %653, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1966 = call float asm sideeffect "", "=v,0"(float %local.reduce72) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %654 = insertelement <4 x float> %651, float %653, i64 3, !dbg !27
  %math.result397 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %654) #9, !dbg !27, !frisk.register_lifetimes !1
  %655 = fptrunc <4 x float> %math.result397 to <4 x half>, !dbg !27
  %math.packed398 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %655) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %656 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed398, <4 x half> %624, <4 x float> %622) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %657 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed398, <4 x half> %640, <4 x float> %623) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %658 = shufflevector <8 x half> %506, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %659 = fsub <4 x float> %369, %477, !dbg !21
  %660 = extractelement <4 x float> %659, i64 0, !dbg !25
  %661 = call float @llvm.amdgcn.exp2.f32(float %660), !dbg !25
  %local.reduce104 = fadd float %k0.sum.anchor1959, %661, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1967 = call float asm sideeffect "", "=v,0"(float %local.reduce104) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %662 = insertelement <4 x float> poison, float %661, i64 0, !dbg !27
  %663 = extractelement <4 x float> %659, i64 1, !dbg !25
  %664 = call float @llvm.amdgcn.exp2.f32(float %663), !dbg !25
  %local.reduce105 = fadd float %k0.sum.anchor1960, %664, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1968 = call float asm sideeffect "", "=v,0"(float %local.reduce105) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %665 = insertelement <4 x float> %662, float %664, i64 1, !dbg !27
  %666 = extractelement <4 x float> %659, i64 2, !dbg !25
  %667 = call float @llvm.amdgcn.exp2.f32(float %666), !dbg !25
  %local.reduce106 = fadd float %k0.sum.anchor1961, %667, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1969 = call float asm sideeffect "", "=v,0"(float %local.reduce106) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %668 = insertelement <4 x float> %665, float %667, i64 2, !dbg !27
  %669 = extractelement <4 x float> %659, i64 3, !dbg !25
  %670 = call float @llvm.amdgcn.exp2.f32(float %669), !dbg !25
  %local.reduce107 = fadd float %k0.sum.anchor1962, %670, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1970 = call float asm sideeffect "", "=v,0"(float %local.reduce107) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %671 = insertelement <4 x float> %668, float %670, i64 3, !dbg !27
  %math.result383 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %671) #9, !dbg !27, !frisk.register_lifetimes !1
  %672 = fptrunc <4 x float> %math.result383 to <4 x half>, !dbg !27
  %math.packed384 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %672) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %673 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed384, <4 x half> %658, <4 x float> %639) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %674 = shufflevector <8 x half> %506, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %675 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed384, <4 x half> %674, <4 x float> %641) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %676 = fsub <4 x float> %418, %478, !dbg !21
  %677 = extractelement <4 x float> %676, i64 0, !dbg !25
  %678 = call float @llvm.amdgcn.exp2.f32(float %677), !dbg !25
  %local.reduce73 = fadd float %k0.sum.anchor1963, %678, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1971 = call float asm sideeffect "", "=v,0"(float %local.reduce73) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %679 = insertelement <4 x float> poison, float %678, i64 0, !dbg !27
  %680 = extractelement <4 x float> %676, i64 1, !dbg !25
  %681 = call float @llvm.amdgcn.exp2.f32(float %680), !dbg !25
  %local.reduce74 = fadd float %k0.sum.anchor1964, %681, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1972 = call float asm sideeffect "", "=v,0"(float %local.reduce74) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %682 = insertelement <4 x float> %679, float %681, i64 1, !dbg !27
  %683 = extractelement <4 x float> %676, i64 2, !dbg !25
  %684 = call float @llvm.amdgcn.exp2.f32(float %683), !dbg !25
  %local.reduce75 = fadd float %k0.sum.anchor1965, %684, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1973 = call float asm sideeffect "", "=v,0"(float %local.reduce75) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %685 = insertelement <4 x float> %682, float %684, i64 2, !dbg !27
  %686 = extractelement <4 x float> %676, i64 3, !dbg !25
  %687 = call float @llvm.amdgcn.exp2.f32(float %686), !dbg !25
  %local.reduce76 = fadd float %k0.sum.anchor1966, %687, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1974 = call float asm sideeffect "", "=v,0"(float %local.reduce76) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %688 = insertelement <4 x float> %685, float %687, i64 3, !dbg !27
  %math.result399 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %688) #9, !dbg !27, !frisk.register_lifetimes !1
  %689 = fptrunc <4 x float> %math.result399 to <4 x half>, !dbg !27
  %math.packed400 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %689) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %690 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed400, <4 x half> %658, <4 x float> %656) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %691 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed400, <4 x half> %674, <4 x float> %657) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %692 = shufflevector <8 x half> %508, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %693 = fsub <4 x float> %375, %477, !dbg !21
  %694 = extractelement <4 x float> %693, i64 0, !dbg !25
  %695 = call float @llvm.amdgcn.exp2.f32(float %694), !dbg !25
  %local.reduce108 = fadd float %k0.sum.anchor1967, %695, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1975 = call float asm sideeffect "", "=v,0"(float %local.reduce108) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %696 = insertelement <4 x float> poison, float %695, i64 0, !dbg !27
  %697 = extractelement <4 x float> %693, i64 1, !dbg !25
  %698 = call float @llvm.amdgcn.exp2.f32(float %697), !dbg !25
  %local.reduce109 = fadd float %k0.sum.anchor1968, %698, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1976 = call float asm sideeffect "", "=v,0"(float %local.reduce109) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %699 = insertelement <4 x float> %696, float %698, i64 1, !dbg !27
  %700 = extractelement <4 x float> %693, i64 2, !dbg !25
  %701 = call float @llvm.amdgcn.exp2.f32(float %700), !dbg !25
  %local.reduce110 = fadd float %k0.sum.anchor1969, %701, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1977 = call float asm sideeffect "", "=v,0"(float %local.reduce110) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %702 = insertelement <4 x float> %699, float %701, i64 2, !dbg !27
  %703 = extractelement <4 x float> %693, i64 3, !dbg !25
  %704 = call float @llvm.amdgcn.exp2.f32(float %703), !dbg !25
  %local.reduce111 = fadd float %k0.sum.anchor1970, %704, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1978 = call float asm sideeffect "", "=v,0"(float %local.reduce111) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %705 = insertelement <4 x float> %702, float %704, i64 3, !dbg !27
  %math.result385 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %705) #9, !dbg !27, !frisk.register_lifetimes !1
  %706 = fptrunc <4 x float> %math.result385 to <4 x half>, !dbg !27
  %math.packed386 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %706) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %707 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed386, <4 x half> %692, <4 x float> %673) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %708 = shufflevector <8 x half> %508, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %709 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed386, <4 x half> %708, <4 x float> %675) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %710 = fsub <4 x float> %424, %478, !dbg !21
  %711 = extractelement <4 x float> %710, i64 0, !dbg !25
  %712 = call float @llvm.amdgcn.exp2.f32(float %711), !dbg !25
  %local.reduce77 = fadd float %k0.sum.anchor1971, %712, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1979 = call float asm sideeffect "", "=v,0"(float %local.reduce77) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %713 = insertelement <4 x float> poison, float %712, i64 0, !dbg !27
  %714 = extractelement <4 x float> %710, i64 1, !dbg !25
  %715 = call float @llvm.amdgcn.exp2.f32(float %714), !dbg !25
  %local.reduce78 = fadd float %k0.sum.anchor1972, %715, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1980 = call float asm sideeffect "", "=v,0"(float %local.reduce78) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %716 = insertelement <4 x float> %713, float %715, i64 1, !dbg !27
  %717 = extractelement <4 x float> %710, i64 2, !dbg !25
  %718 = call float @llvm.amdgcn.exp2.f32(float %717), !dbg !25
  %local.reduce79 = fadd float %k0.sum.anchor1973, %718, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1981 = call float asm sideeffect "", "=v,0"(float %local.reduce79) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %719 = insertelement <4 x float> %716, float %718, i64 2, !dbg !27
  %720 = extractelement <4 x float> %710, i64 3, !dbg !25
  %721 = call float @llvm.amdgcn.exp2.f32(float %720), !dbg !25
  %local.reduce80 = fadd float %k0.sum.anchor1974, %721, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1982 = call float asm sideeffect "", "=v,0"(float %local.reduce80) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %722 = insertelement <4 x float> %719, float %721, i64 3, !dbg !27
  %math.result401 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %722) #9, !dbg !27, !frisk.register_lifetimes !1
  %723 = fptrunc <4 x float> %math.result401 to <4 x half>, !dbg !27
  %math.packed402 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %723) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %724 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed402, <4 x half> %692, <4 x float> %690) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %725 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed402, <4 x half> %708, <4 x float> %691) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %726 = shufflevector <8 x half> %510, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %727 = fsub <4 x float> %381, %477, !dbg !21
  %728 = extractelement <4 x float> %727, i64 0, !dbg !25
  %729 = call float @llvm.amdgcn.exp2.f32(float %728), !dbg !25
  %local.reduce112 = fadd float %k0.sum.anchor1975, %729, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1983 = call float asm sideeffect "", "=v,0"(float %local.reduce112) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %730 = insertelement <4 x float> poison, float %729, i64 0, !dbg !27
  %731 = extractelement <4 x float> %727, i64 1, !dbg !25
  %732 = call float @llvm.amdgcn.exp2.f32(float %731), !dbg !25
  %local.reduce113 = fadd float %k0.sum.anchor1976, %732, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1984 = call float asm sideeffect "", "=v,0"(float %local.reduce113) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %733 = insertelement <4 x float> %730, float %732, i64 1, !dbg !27
  %734 = extractelement <4 x float> %727, i64 2, !dbg !25
  %735 = call float @llvm.amdgcn.exp2.f32(float %734), !dbg !25
  %local.reduce114 = fadd float %k0.sum.anchor1977, %735, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1985 = call float asm sideeffect "", "=v,0"(float %local.reduce114) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %736 = insertelement <4 x float> %733, float %735, i64 2, !dbg !27
  %737 = extractelement <4 x float> %727, i64 3, !dbg !25
  %738 = call float @llvm.amdgcn.exp2.f32(float %737), !dbg !25
  %local.reduce115 = fadd float %k0.sum.anchor1978, %738, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1986 = call float asm sideeffect "", "=v,0"(float %local.reduce115) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %739 = insertelement <4 x float> %736, float %738, i64 3, !dbg !27
  %math.result387 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %739) #9, !dbg !27, !frisk.register_lifetimes !1
  %740 = fptrunc <4 x float> %math.result387 to <4 x half>, !dbg !27
  %math.packed388 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %740) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %741 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed388, <4 x half> %726, <4 x float> %707) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %742 = shufflevector <8 x half> %510, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %743 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed388, <4 x half> %742, <4 x float> %709) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %744 = fsub <4 x float> %430, %478, !dbg !21
  %745 = extractelement <4 x float> %744, i64 0, !dbg !25
  %746 = call float @llvm.amdgcn.exp2.f32(float %745), !dbg !25
  %local.reduce81 = fadd float %k0.sum.anchor1979, %746, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1987 = call float asm sideeffect "", "=v,0"(float %local.reduce81) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %747 = insertelement <4 x float> poison, float %746, i64 0, !dbg !27
  %748 = extractelement <4 x float> %744, i64 1, !dbg !25
  %749 = call float @llvm.amdgcn.exp2.f32(float %748), !dbg !25
  %local.reduce82 = fadd float %k0.sum.anchor1980, %749, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1988 = call float asm sideeffect "", "=v,0"(float %local.reduce82) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %750 = insertelement <4 x float> %747, float %749, i64 1, !dbg !27
  %751 = extractelement <4 x float> %744, i64 2, !dbg !25
  %752 = call float @llvm.amdgcn.exp2.f32(float %751), !dbg !25
  %local.reduce83 = fadd float %k0.sum.anchor1981, %752, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1989 = call float asm sideeffect "", "=v,0"(float %local.reduce83) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %753 = insertelement <4 x float> %750, float %752, i64 2, !dbg !27
  %754 = extractelement <4 x float> %744, i64 3, !dbg !25
  %755 = call float @llvm.amdgcn.exp2.f32(float %754), !dbg !25
  %local.reduce84 = fadd float %k0.sum.anchor1982, %755, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1990 = call float asm sideeffect "", "=v,0"(float %local.reduce84) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %756 = insertelement <4 x float> %753, float %755, i64 3, !dbg !27
  %math.result403 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %756) #9, !dbg !27, !frisk.register_lifetimes !1
  %757 = fptrunc <4 x float> %math.result403 to <4 x half>, !dbg !27
  %math.packed404 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %757) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %758 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed404, <4 x half> %726, <4 x float> %724) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %759 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed404, <4 x half> %742, <4 x float> %725) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %760 = bitcast <4 x i32> %511 to <8 x half>, !dbg !13
  %761 = shufflevector <8 x half> %760, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %762 = fsub <4 x float> %387, %477, !dbg !21
  %763 = extractelement <4 x float> %762, i64 0, !dbg !25
  %764 = call float @llvm.amdgcn.exp2.f32(float %763), !dbg !25
  %local.reduce116 = fadd float %k0.sum.anchor1983, %764, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1991 = call float asm sideeffect "", "=v,0"(float %local.reduce116) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %765 = insertelement <4 x float> poison, float %764, i64 0, !dbg !27
  %766 = extractelement <4 x float> %762, i64 1, !dbg !25
  %767 = call float @llvm.amdgcn.exp2.f32(float %766), !dbg !25
  %local.reduce117 = fadd float %k0.sum.anchor1984, %767, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1992 = call float asm sideeffect "", "=v,0"(float %local.reduce117) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %local.reduce120 = fadd float %k0.sum.anchor1991, %k0.sum.anchor1992, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1993 = call float asm sideeffect "", "=v,0"(float %local.reduce120) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %768 = insertelement <4 x float> %765, float %767, i64 1, !dbg !27
  %769 = extractelement <4 x float> %762, i64 2, !dbg !25
  %770 = call float @llvm.amdgcn.exp2.f32(float %769), !dbg !25
  %local.reduce118 = fadd float %k0.sum.anchor1985, %770, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1994 = call float asm sideeffect "", "=v,0"(float %local.reduce118) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %771 = insertelement <4 x float> %768, float %770, i64 2, !dbg !27
  %772 = extractelement <4 x float> %762, i64 3, !dbg !25
  %773 = call float @llvm.amdgcn.exp2.f32(float %772), !dbg !25
  %local.reduce119 = fadd float %k0.sum.anchor1986, %773, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1995 = call float asm sideeffect "", "=v,0"(float %local.reduce119) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %local.reduce121 = fadd float %k0.sum.anchor1994, %k0.sum.anchor1995, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1996 = call float asm sideeffect "", "=v,0"(float %local.reduce121) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %774 = fadd float %k0.sum.anchor1993, %k0.sum.anchor1996, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1997 = call float asm sideeffect "", "=v,0"(float %774) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %775 = fadd float %k0.sum.anchor1997, %486, !dbg !26
  %k0.sum.anchor1998 = call float asm sideeffect "", "=v,0"(float %775) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %776 = insertelement <1 x float> poison, float %k0.sum.anchor1998, i64 0, !dbg !7
  %777 = insertvalue [2 x <1 x float>] poison, <1 x float> %776, 0, !dbg !7
  %778 = insertelement <4 x float> %771, float %773, i64 3, !dbg !27
  %math.result389 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %778) #9, !dbg !27, !frisk.register_lifetimes !1
  %779 = fptrunc <4 x float> %math.result389 to <4 x half>, !dbg !27
  %math.packed390 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %779) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %780 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed390, <4 x half> %761, <4 x float> %741) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %781 = shufflevector <8 x half> %760, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %782 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed390, <4 x half> %781, <4 x float> %743) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %783 = fsub <4 x float> %436, %478, !dbg !21
  %784 = extractelement <4 x float> %783, i64 0, !dbg !25
  %785 = call float @llvm.amdgcn.exp2.f32(float %784), !dbg !25
  %local.reduce85 = fadd float %k0.sum.anchor1987, %785, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1999 = call float asm sideeffect "", "=v,0"(float %local.reduce85) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %786 = insertelement <4 x float> poison, float %785, i64 0, !dbg !27
  %787 = extractelement <4 x float> %783, i64 1, !dbg !25
  %788 = call float @llvm.amdgcn.exp2.f32(float %787), !dbg !25
  %local.reduce86 = fadd float %k0.sum.anchor1988, %788, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor2000 = call float asm sideeffect "", "=v,0"(float %local.reduce86) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %local.reduce89 = fadd float %k0.sum.anchor1999, %k0.sum.anchor2000, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor2001 = call float asm sideeffect "", "=v,0"(float %local.reduce89) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %789 = insertelement <4 x float> %786, float %788, i64 1, !dbg !27
  %790 = extractelement <4 x float> %783, i64 2, !dbg !25
  %791 = call float @llvm.amdgcn.exp2.f32(float %790), !dbg !25
  %local.reduce87 = fadd float %k0.sum.anchor1989, %791, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor2002 = call float asm sideeffect "", "=v,0"(float %local.reduce87) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %792 = insertelement <4 x float> %789, float %791, i64 2, !dbg !27
  %793 = extractelement <4 x float> %783, i64 3, !dbg !25
  %794 = call float @llvm.amdgcn.exp2.f32(float %793), !dbg !25
  %local.reduce88 = fadd float %k0.sum.anchor1990, %794, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor2003 = call float asm sideeffect "", "=v,0"(float %local.reduce88) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %local.reduce90 = fadd float %k0.sum.anchor2002, %k0.sum.anchor2003, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor2004 = call float asm sideeffect "", "=v,0"(float %local.reduce90) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %795 = fadd float %k0.sum.anchor2001, %k0.sum.anchor2004, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor2005 = call float asm sideeffect "", "=v,0"(float %795) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %796 = fadd float %k0.sum.anchor2005, %494, !dbg !26
  %k0.sum.anchor2006 = call float asm sideeffect "", "=v,0"(float %796) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %797 = insertelement <1 x float> poison, float %k0.sum.anchor2006, i64 0, !dbg !7
  %798 = insertvalue [2 x <1 x float>] %777, <1 x float> %797, 1, !dbg !7
  %799 = insertelement <4 x float> %792, float %794, i64 3, !dbg !27
  %math.result405 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %799) #9, !dbg !27, !frisk.register_lifetimes !1
  %800 = fptrunc <4 x float> %math.result405 to <4 x half>, !dbg !27
  %math.packed406 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %800) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %801 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed406, <4 x half> %761, <4 x float> %758) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %802 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed406, <4 x half> %781, <4 x float> %759) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %vdb.address1478 = or disjoint i64 %108, 64, !dbg !15
  %vdb.address1479 = or disjoint i64 %address.local924, %vdb.address1478, !dbg !15
  %vdb.address1480 = trunc nuw i64 %vdb.address1479 to i32, !dbg !15
  %vdb.address1481 = shl nuw nsw i32 %vdb.address1480, 1, !dbg !15
  %vdb.address1489 = or disjoint i64 %address.local930, %vdb.address1478, !dbg !15
  %vdb.address1490 = trunc nuw i64 %vdb.address1489 to i32, !dbg !15
  %vdb.address1491 = shl nuw nsw i32 %vdb.address1490, 1, !dbg !15
  %803 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1377, i32 %vdb.address1481, <4 x i32> %dma.buffer.rsrc1352) #10, !dbg !15
  %804 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1374, i32 %vdb.address1491, <4 x i32> %dma.buffer.rsrc1352) #10, !dbg !15
  %vdb.read = getelementptr i8, ptr addrspace(3) %address.local938, i64 8192, !dbg !13
  %805 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %806 = bitcast <4 x i32> %805 to <8 x half>, !dbg !13
  %vdb.read1469 = getelementptr i8, ptr addrspace(3) %address.local947, i64 8448, !dbg !13
  %807 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1469, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %808 = bitcast <4 x i32> %807 to <8 x half>, !dbg !13
  %vdb.read1470 = getelementptr i8, ptr addrspace(3) %address.local947, i64 8704, !dbg !13
  %809 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1470, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %810 = bitcast <4 x i32> %809 to <8 x half>, !dbg !13
  %vdb.read1471 = getelementptr i8, ptr addrspace(3) %address.local947, i64 8960, !dbg !13
  %811 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1471, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %812 = bitcast <4 x i32> %811 to <8 x half>, !dbg !13
  %vdb.read1472 = getelementptr i8, ptr addrspace(3) %address.local938, i64 12288, !dbg !13
  %813 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1472, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %814 = bitcast <4 x i32> %813 to <8 x half>, !dbg !13
  %vdb.read1473 = getelementptr i8, ptr addrspace(3) %address.local947, i64 12544, !dbg !13
  %815 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1473, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %816 = bitcast <4 x i32> %815 to <8 x half>, !dbg !13
  %vdb.read1474 = getelementptr i8, ptr addrspace(3) %address.local947, i64 12800, !dbg !13
  %817 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1474, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %818 = bitcast <4 x i32> %817 to <8 x half>, !dbg !13
  %vdb.read1475 = getelementptr i8, ptr addrspace(3) %address.local947, i64 13056, !dbg !13
  %819 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1475, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %820 = shufflevector <8 x half> %806, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %821 = extractvalue [2 x <8 x float>] %101, 0, !dbg !28
  %822 = shufflevector <8 x float> %821, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %823 = fmul <4 x float> %822, %483, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %824 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %820, <4 x float> %823) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %825 = shufflevector <8 x half> %806, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %826 = shufflevector <8 x float> %821, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %827 = fmul <4 x float> %826, %483, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %828 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %825, <4 x float> %827) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %829 = extractvalue [2 x <8 x float>] %101, 1, !dbg !28
  %830 = shufflevector <8 x float> %829, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %831 = fmul <4 x float> %830, %491, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %832 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed392, <4 x half> %820, <4 x float> %831) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %833 = shufflevector <8 x float> %829, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %834 = fmul <4 x float> %833, %491, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %835 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed392, <4 x half> %825, <4 x float> %834) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %836 = shufflevector <8 x half> %808, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %837 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed378, <4 x half> %836, <4 x float> %824) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %838 = shufflevector <8 x half> %808, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %839 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed378, <4 x half> %838, <4 x float> %828) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %840 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed394, <4 x half> %836, <4 x float> %832) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %841 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed394, <4 x half> %838, <4 x float> %835) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %842 = shufflevector <8 x half> %810, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %843 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed380, <4 x half> %842, <4 x float> %837) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %844 = shufflevector <8 x half> %810, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %845 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed380, <4 x half> %844, <4 x float> %839) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %846 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed396, <4 x half> %842, <4 x float> %840) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %847 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed396, <4 x half> %844, <4 x float> %841) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %848 = shufflevector <8 x half> %812, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %849 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed382, <4 x half> %848, <4 x float> %843) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %850 = shufflevector <8 x half> %812, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %851 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed382, <4 x half> %850, <4 x float> %845) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %852 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed398, <4 x half> %848, <4 x float> %846) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %853 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed398, <4 x half> %850, <4 x float> %847) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %854 = shufflevector <8 x half> %814, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %855 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed384, <4 x half> %854, <4 x float> %849) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %856 = shufflevector <8 x half> %814, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %857 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed384, <4 x half> %856, <4 x float> %851) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %858 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed400, <4 x half> %854, <4 x float> %852) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %859 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed400, <4 x half> %856, <4 x float> %853) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %860 = shufflevector <8 x half> %816, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %861 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed386, <4 x half> %860, <4 x float> %855) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %862 = shufflevector <8 x half> %816, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %863 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed386, <4 x half> %862, <4 x float> %857) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %864 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed402, <4 x half> %860, <4 x float> %858) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %865 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed402, <4 x half> %862, <4 x float> %859) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %866 = shufflevector <8 x half> %818, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %867 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed388, <4 x half> %866, <4 x float> %861) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %868 = shufflevector <8 x half> %818, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %869 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed388, <4 x half> %868, <4 x float> %863) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %870 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed404, <4 x half> %866, <4 x float> %864) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %871 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed404, <4 x half> %868, <4 x float> %865) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %872 = bitcast <4 x i32> %819 to <8 x half>, !dbg !13
  %873 = shufflevector <8 x half> %872, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %874 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed390, <4 x half> %873, <4 x float> %867) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %875 = shufflevector <8 x half> %872, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %876 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed390, <4 x half> %875, <4 x float> %869) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %877 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed406, <4 x half> %873, <4 x float> %870) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %878 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed406, <4 x half> %875, <4 x float> %871) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %vdb.address1498 = or disjoint i64 %108, 96, !dbg !15
  %vdb.address1499 = or disjoint i64 %address.local924, %vdb.address1498, !dbg !15
  %vdb.address1500 = trunc nuw i64 %vdb.address1499 to i32, !dbg !15
  %vdb.address1501 = shl nuw nsw i32 %vdb.address1500, 1, !dbg !15
  %vdb.address1509 = or disjoint i64 %address.local930, %vdb.address1498, !dbg !15
  %vdb.address1510 = trunc nuw i64 %vdb.address1509 to i32, !dbg !15
  %vdb.address1511 = shl nuw nsw i32 %vdb.address1510, 1, !dbg !15
  %879 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1501, <4 x i32> %dma.buffer.rsrc1352) #10, !dbg !15
  %880 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1468, i32 %vdb.address1511, <4 x i32> %dma.buffer.rsrc1352) #10, !dbg !15
  %881 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local938, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %882 = bitcast <4 x i32> %881 to <8 x half>, !dbg !13
  %883 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local948, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %884 = bitcast <4 x i32> %883 to <8 x half>, !dbg !13
  %885 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local952, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %886 = bitcast <4 x i32> %885 to <8 x half>, !dbg !13
  %887 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local956, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %888 = bitcast <4 x i32> %887 to <8 x half>, !dbg !13
  %889 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local958, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %890 = bitcast <4 x i32> %889 to <8 x half>, !dbg !13
  %891 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local962, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %892 = bitcast <4 x i32> %891 to <8 x half>, !dbg !13
  %893 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local966, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %894 = bitcast <4 x i32> %893 to <8 x half>, !dbg !13
  %895 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local970, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %896 = shufflevector <8 x half> %882, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %897 = extractvalue [2 x <8 x float>] %102, 0, !dbg !28
  %898 = shufflevector <8 x float> %897, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %899 = fmul <4 x float> %898, %483, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %900 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %896, <4 x float> %899) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %901 = shufflevector <8 x half> %882, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %902 = shufflevector <8 x float> %897, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %903 = fmul <4 x float> %902, %483, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %904 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %901, <4 x float> %903) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %905 = extractvalue [2 x <8 x float>] %102, 1, !dbg !28
  %906 = shufflevector <8 x float> %905, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %907 = fmul <4 x float> %906, %491, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %908 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed392, <4 x half> %896, <4 x float> %907) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %909 = shufflevector <8 x float> %905, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %910 = fmul <4 x float> %909, %491, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %911 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed392, <4 x half> %901, <4 x float> %910) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %912 = shufflevector <8 x half> %884, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %913 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed378, <4 x half> %912, <4 x float> %900) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %914 = shufflevector <8 x half> %884, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %915 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed378, <4 x half> %914, <4 x float> %904) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %916 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed394, <4 x half> %912, <4 x float> %908) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %917 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed394, <4 x half> %914, <4 x float> %911) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %918 = shufflevector <8 x half> %886, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %919 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed380, <4 x half> %918, <4 x float> %913) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %920 = shufflevector <8 x half> %886, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %921 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed380, <4 x half> %920, <4 x float> %915) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %922 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed396, <4 x half> %918, <4 x float> %916) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %923 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed396, <4 x half> %920, <4 x float> %917) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %924 = shufflevector <8 x half> %888, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %925 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed382, <4 x half> %924, <4 x float> %919) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %926 = shufflevector <8 x half> %888, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %927 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed382, <4 x half> %926, <4 x float> %921) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %928 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed398, <4 x half> %924, <4 x float> %922) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %929 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed398, <4 x half> %926, <4 x float> %923) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %930 = shufflevector <8 x half> %890, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %931 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed384, <4 x half> %930, <4 x float> %925) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %932 = shufflevector <8 x half> %890, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %933 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed384, <4 x half> %932, <4 x float> %927) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %934 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed400, <4 x half> %930, <4 x float> %928) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %935 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed400, <4 x half> %932, <4 x float> %929) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %936 = shufflevector <8 x half> %892, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %937 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed386, <4 x half> %936, <4 x float> %931) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %938 = shufflevector <8 x half> %892, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %939 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed386, <4 x half> %938, <4 x float> %933) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %940 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed402, <4 x half> %936, <4 x float> %934) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %941 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed402, <4 x half> %938, <4 x float> %935) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %942 = shufflevector <8 x half> %894, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %943 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed388, <4 x half> %942, <4 x float> %937) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %944 = shufflevector <8 x half> %894, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %945 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed388, <4 x half> %944, <4 x float> %939) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %946 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed404, <4 x half> %942, <4 x float> %940) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %947 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed404, <4 x half> %944, <4 x float> %941) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %948 = bitcast <4 x i32> %895 to <8 x half>, !dbg !13
  %949 = shufflevector <8 x half> %948, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %950 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed390, <4 x half> %949, <4 x float> %943) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %951 = shufflevector <8 x half> %948, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %952 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed390, <4 x half> %951, <4 x float> %945) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %953 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed406, <4 x half> %949, <4 x float> %946) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %954 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed406, <4 x half> %951, <4 x float> %947) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %955 = add nuw nsw i64 %97, 1, !dbg !7
  %k0.has.next = icmp ult i64 %955, 32, !dbg !13
  br i1 %k0.has.next, label %k0.issue, label %k0.join, !dbg !13

k0.issue:                                         ; preds = %105
  %k0.next. = shl nuw nsw i64 %955, 14, !dbg !12
  %k0.next.2018 = or disjoint i64 %106, %k0.next., !dbg !12
  %k0.next.kdma.source1536 = or disjoint i64 %k0.next.2018, %kdma.source1535, !dbg !12
  %k0.next.kdma.source1537 = trunc nuw i64 %k0.next.kdma.source1536 to i32, !dbg !12
  %k0.next.kdma.source1538 = shl nuw nsw i32 %k0.next.kdma.source1537, 1, !dbg !12
  %k0.next.kdma.offset = or disjoint i32 %k0.next.kdma.source1538, %kdma.intra, !dbg !12
  %k0.next.2019 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1377, i32 %k0.next.kdma.offset, <4 x i32> %dma.buffer.rsrc1376) #10, !dbg !12, !frisk.cross_bn_k0 !1
  %k0.next.kdma.source1550 = or disjoint i64 %k0.next.2018, %kdma.source1549, !dbg !12
  %k0.next.kdma.source1551 = trunc nuw i64 %k0.next.kdma.source1550 to i32, !dbg !12
  %k0.next.kdma.source1552 = shl nuw nsw i32 %k0.next.kdma.source1551, 1, !dbg !12
  %k0.next.kdma.offset1553 = or disjoint i32 %k0.next.kdma.source1552, %kdma.intra.hi, !dbg !12
  %k0.next.2020 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1554, i32 %k0.next.kdma.offset1553, <4 x i32> %dma.buffer.rsrc1376) #10, !dbg !12, !frisk.cross_bn_k0 !1
  %k0.next.kdma.source1567 = or disjoint i64 %k0.next.2018, %kdma.source1566, !dbg !12
  %k0.next.kdma.source1568 = trunc nuw i64 %k0.next.kdma.source1567 to i32, !dbg !12
  %k0.next.kdma.source1569 = shl nuw nsw i32 %k0.next.kdma.source1568, 1, !dbg !12
  %k0.next.kdma.offset1570 = or disjoint i32 %k0.next.kdma.source1569, %kdma.intra, !dbg !12
  %k0.next.2021 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1374, i32 %k0.next.kdma.offset1570, <4 x i32> %dma.buffer.rsrc1376) #10, !dbg !12, !frisk.cross_bn_k0 !1
  %k0.next.kdma.source1584 = or disjoint i64 %k0.next.2018, %kdma.source1583, !dbg !12
  %k0.next.kdma.source1585 = trunc nuw i64 %k0.next.kdma.source1584 to i32, !dbg !12
  %k0.next.kdma.source1586 = shl nuw nsw i32 %k0.next.kdma.source1585, 1, !dbg !12
  %k0.next.kdma.offset1587 = or disjoint i32 %k0.next.kdma.source1586, %kdma.intra.hi, !dbg !12
  %k0.next.2022 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1588, i32 %k0.next.kdma.offset1587, <4 x i32> %dma.buffer.rsrc1376) #10, !dbg !12, !frisk.cross_bn_k0 !1
  br label %k0.join

k0.join:                                          ; preds = %105, %k0.issue
  %956 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %957 = bitcast <4 x i32> %956 to <8 x half>, !dbg !13
  %958 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1469, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %959 = bitcast <4 x i32> %958 to <8 x half>, !dbg !13
  %960 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1470, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %961 = bitcast <4 x i32> %960 to <8 x half>, !dbg !13
  %962 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1471, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %963 = bitcast <4 x i32> %962 to <8 x half>, !dbg !13
  %964 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1472, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %965 = bitcast <4 x i32> %964 to <8 x half>, !dbg !13
  %966 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1473, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %967 = bitcast <4 x i32> %966 to <8 x half>, !dbg !13
  %968 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1474, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %969 = bitcast <4 x i32> %968 to <8 x half>, !dbg !13
  %970 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1475, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %971 = shufflevector <8 x half> %957, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %972 = extractvalue [2 x <8 x float>] %103, 0, !dbg !28
  %973 = shufflevector <8 x float> %972, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %974 = fmul <4 x float> %973, %483, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %975 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %971, <4 x float> %974) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %976 = shufflevector <8 x half> %957, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %977 = shufflevector <8 x float> %972, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %978 = fmul <4 x float> %977, %483, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %979 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %976, <4 x float> %978) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %980 = extractvalue [2 x <8 x float>] %103, 1, !dbg !28
  %981 = shufflevector <8 x float> %980, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %982 = fmul <4 x float> %981, %491, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %983 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed392, <4 x half> %971, <4 x float> %982) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %984 = shufflevector <8 x float> %980, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %985 = fmul <4 x float> %984, %491, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %986 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed392, <4 x half> %976, <4 x float> %985) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %987 = shufflevector <8 x half> %959, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %988 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed378, <4 x half> %987, <4 x float> %975) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %989 = shufflevector <8 x half> %959, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %990 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed378, <4 x half> %989, <4 x float> %979) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %991 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed394, <4 x half> %987, <4 x float> %983) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %992 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed394, <4 x half> %989, <4 x float> %986) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %993 = shufflevector <8 x half> %961, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %994 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed380, <4 x half> %993, <4 x float> %988) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %995 = shufflevector <8 x half> %961, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %996 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed380, <4 x half> %995, <4 x float> %990) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %997 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed396, <4 x half> %993, <4 x float> %991) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %998 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed396, <4 x half> %995, <4 x float> %992) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %999 = shufflevector <8 x half> %963, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1000 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed382, <4 x half> %999, <4 x float> %994) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1001 = shufflevector <8 x half> %963, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1002 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed382, <4 x half> %1001, <4 x float> %996) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1003 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed398, <4 x half> %999, <4 x float> %997) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1004 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed398, <4 x half> %1001, <4 x float> %998) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1005 = shufflevector <8 x half> %965, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1006 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed384, <4 x half> %1005, <4 x float> %1000) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1007 = shufflevector <8 x half> %965, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1008 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed384, <4 x half> %1007, <4 x float> %1002) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1009 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed400, <4 x half> %1005, <4 x float> %1003) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1010 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed400, <4 x half> %1007, <4 x float> %1004) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1011 = shufflevector <8 x half> %967, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1012 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed386, <4 x half> %1011, <4 x float> %1006) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1013 = shufflevector <8 x half> %967, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1014 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed386, <4 x half> %1013, <4 x float> %1008) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1015 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed402, <4 x half> %1011, <4 x float> %1009) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1016 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed402, <4 x half> %1013, <4 x float> %1010) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1017 = shufflevector <8 x half> %969, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1018 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed388, <4 x half> %1017, <4 x float> %1012) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1019 = shufflevector <8 x half> %969, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1020 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed388, <4 x half> %1019, <4 x float> %1014) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1021 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed404, <4 x half> %1017, <4 x float> %1015) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1022 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed404, <4 x half> %1019, <4 x float> %1016) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1023 = bitcast <4 x i32> %970 to <8 x half>, !dbg !13
  %1024 = shufflevector <8 x half> %1023, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1025 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed390, <4 x half> %1024, <4 x float> %1018) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1026 = shufflevector <8 x half> %1023, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1027 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed390, <4 x half> %1026, <4 x float> %1020) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1028 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed406, <4 x half> %1024, <4 x float> %1021) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1029 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed406, <4 x half> %1026, <4 x float> %1022) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !7
  %1030 = shufflevector <4 x float> %780, <4 x float> %782, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1031 = insertvalue [2 x <8 x float>] poison, <8 x float> %1030, 0, !dbg !7
  %1032 = shufflevector <4 x float> %801, <4 x float> %802, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1033 = insertvalue [2 x <8 x float>] %1031, <8 x float> %1032, 1, !dbg !7
  %1034 = insertvalue [2 x <1 x float>] poison, <1 x float> %473, 0, !dbg !7
  %1035 = insertvalue [2 x <1 x float>] %1034, <1 x float> %476, 1, !dbg !7
  %1036 = shufflevector <4 x float> %874, <4 x float> %876, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1037 = insertvalue [2 x <8 x float>] poison, <8 x float> %1036, 0, !dbg !7
  %1038 = shufflevector <4 x float> %877, <4 x float> %878, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1039 = insertvalue [2 x <8 x float>] %1037, <8 x float> %1038, 1, !dbg !7
  %1040 = shufflevector <4 x float> %950, <4 x float> %952, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1041 = insertvalue [2 x <8 x float>] poison, <8 x float> %1040, 0, !dbg !7
  %1042 = shufflevector <4 x float> %953, <4 x float> %954, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1043 = insertvalue [2 x <8 x float>] %1041, <8 x float> %1042, 1, !dbg !7
  %1044 = shufflevector <4 x float> %1025, <4 x float> %1027, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1045 = insertvalue [2 x <8 x float>] poison, <8 x float> %1044, 0, !dbg !7
  %1046 = shufflevector <4 x float> %1028, <4 x float> %1029, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1047 = insertvalue [2 x <8 x float>] %1045, <8 x float> %1046, 1, !dbg !7
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  br label %96, !dbg !7

1048:                                             ; preds = %96
  %lane.sum.field187 = extractvalue [2 x <1 x float>] %99, 1, !dbg !7
  %lane.sum.local188 = extractelement <1 x float> %lane.sum.field187, i64 0, !dbg !7
  %lane.sum.lane.lo189 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !7
  %lane.sum.lane190 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %lane.sum.lane.lo189), !dbg !7
  %1049 = shl nuw nsw i32 %lane.sum.lane190, 2, !dbg !7
  %1050 = xor i32 %1049, 64, !dbg !7
  %1051 = bitcast float %lane.sum.local188 to i32, !dbg !7
  %lane.sum.shuffle191 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1050, i32 %1051), !dbg !7
  %1052 = bitcast i32 %lane.sum.shuffle191 to float, !dbg !7
  %lane.sum.final192 = fadd float %lane.sum.local188, %1052, !dbg !7
  %1053 = xor i32 %1049, 128, !dbg !7
  %1054 = bitcast float %lane.sum.final192 to i32, !dbg !7
  %lane.sum.shuffle193 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1053, i32 %1054), !dbg !7
  %1055 = bitcast i32 %lane.sum.shuffle193 to float, !dbg !7
  %lane.sum.final194 = fadd float %lane.sum.final192, %1055, !dbg !7
  %1056 = insertelement <1 x float> poison, float %lane.sum.final194, i64 0, !dbg !7
  %lane.sum.field = extractvalue [2 x <1 x float>] %99, 0, !dbg !7
  %lane.sum.local = extractelement <1 x float> %lane.sum.field, i64 0, !dbg !7
  %1057 = bitcast float %lane.sum.local to i32, !dbg !7
  %lane.sum.shuffle = call i32 @llvm.amdgcn.ds.bpermute(i32 %1050, i32 %1057), !dbg !7
  %1058 = bitcast i32 %lane.sum.shuffle to float, !dbg !7
  %lane.sum.final = fadd float %lane.sum.local, %1058, !dbg !7
  %1059 = bitcast float %lane.sum.final to i32, !dbg !7
  %lane.sum.shuffle185 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1053, i32 %1059), !dbg !7
  %1060 = bitcast i32 %lane.sum.shuffle185 to float, !dbg !7
  %lane.sum.final186 = fadd float %lane.sum.final, %1060, !dbg !7
  %1061 = insertelement <1 x float> poison, float %lane.sum.final186, i64 0, !dbg !7
  %1062 = and i32 %kdma.tid, 16, !dbg !7
  %.not = icmp eq i32 %1062, 0, !dbg !7
  %1063 = and i32 %kdma.tid, 32, !dbg !7
  %.not36 = icmp eq i32 %1063, 0, !dbg !7
  %1064 = extractvalue [2 x <8 x float>] %103, 0, !dbg !7
  %1065 = extractelement <8 x float> %1064, i64 0, !dbg !7
  %1066 = extractelement <8 x float> %1064, i64 1, !dbg !7
  %1067 = extractelement <8 x float> %1064, i64 2, !dbg !7
  %1068 = extractelement <8 x float> %1064, i64 3, !dbg !7
  %1069 = select i1 %.not, float %1066, float %1065, !dbg !7
  %1070 = add nuw nsw i32 %lane.sum.lane190, 64, !dbg !7
  %1071 = xor i32 %lane.sum.lane190, 16, !dbg !7
  %1072 = and i32 %1070, 192, !dbg !7
  %1073 = icmp samesign ult i32 %1071, %1072, !dbg !7
  %1074 = select i1 %1073, i32 %1071, i32 %lane.sum.lane190, !dbg !7
  %1075 = shl nuw nsw i32 %1074, 2, !dbg !7
  %1076 = bitcast float %1069 to i32, !dbg !7
  %1077 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1075, i32 %1076), !dbg !7
  %1078 = bitcast i32 %1077 to float, !dbg !7
  %1079 = select i1 %.not, float %1065, float %1078, !dbg !7
  %1080 = select i1 %.not, float %1078, float %1066, !dbg !7
  %1081 = select i1 %.not, float %1068, float %1067, !dbg !7
  %1082 = bitcast float %1081 to i32, !dbg !7
  %1083 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1075, i32 %1082), !dbg !7
  %1084 = bitcast i32 %1083 to float, !dbg !7
  %1085 = select i1 %.not, float %1067, float %1084, !dbg !7
  %1086 = select i1 %.not, float %1084, float %1068, !dbg !7
  %1087 = select i1 %.not36, float %1085, float %1079, !dbg !7
  %1088 = xor i32 %lane.sum.lane190, 32, !dbg !7
  %1089 = icmp samesign ult i32 %1088, %1072, !dbg !7
  %1090 = select i1 %1089, i32 %1088, i32 %lane.sum.lane190, !dbg !7
  %1091 = shl nuw nsw i32 %1090, 2, !dbg !7
  %1092 = bitcast float %1087 to i32, !dbg !7
  %1093 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1091, i32 %1092), !dbg !7
  %1094 = bitcast i32 %1093 to float, !dbg !7
  %1095 = select i1 %.not36, float %1079, float %1094, !dbg !7
  %1096 = select i1 %.not36, float %1094, float %1085, !dbg !7
  %1097 = select i1 %.not36, float %1086, float %1080, !dbg !7
  %1098 = bitcast float %1097 to i32, !dbg !7
  %1099 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1091, i32 %1098), !dbg !7
  %1100 = bitcast i32 %1099 to float, !dbg !7
  %1101 = select i1 %.not36, float %1080, float %1100, !dbg !7
  %1102 = select i1 %.not36, float %1100, float %1086, !dbg !7
  %1103 = extractelement <8 x float> %1064, i64 4, !dbg !7
  %1104 = extractelement <8 x float> %1064, i64 5, !dbg !7
  %1105 = extractelement <8 x float> %1064, i64 6, !dbg !7
  %1106 = extractelement <8 x float> %1064, i64 7, !dbg !7
  %1107 = select i1 %.not, float %1104, float %1103, !dbg !7
  %1108 = bitcast float %1107 to i32, !dbg !7
  %1109 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1075, i32 %1108), !dbg !7
  %1110 = bitcast i32 %1109 to float, !dbg !7
  %1111 = select i1 %.not, float %1103, float %1110, !dbg !7
  %1112 = select i1 %.not, float %1110, float %1104, !dbg !7
  %1113 = select i1 %.not, float %1106, float %1105, !dbg !7
  %1114 = bitcast float %1113 to i32, !dbg !7
  %1115 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1075, i32 %1114), !dbg !7
  %1116 = bitcast i32 %1115 to float, !dbg !7
  %1117 = select i1 %.not, float %1105, float %1116, !dbg !7
  %1118 = select i1 %.not, float %1116, float %1106, !dbg !7
  %1119 = select i1 %.not36, float %1117, float %1111, !dbg !7
  %1120 = bitcast float %1119 to i32, !dbg !7
  %1121 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1091, i32 %1120), !dbg !7
  %1122 = bitcast i32 %1121 to float, !dbg !7
  %1123 = select i1 %.not36, float %1111, float %1122, !dbg !7
  %1124 = select i1 %.not36, float %1122, float %1117, !dbg !7
  %1125 = select i1 %.not36, float %1118, float %1112, !dbg !7
  %1126 = bitcast float %1125 to i32, !dbg !7
  %1127 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1091, i32 %1126), !dbg !7
  %1128 = bitcast i32 %1127 to float, !dbg !7
  %1129 = select i1 %.not36, float %1112, float %1128, !dbg !7
  %1130 = select i1 %.not36, float %1128, float %1118, !dbg !7
  %1131 = extractvalue [2 x <8 x float>] %103, 1, !dbg !7
  %1132 = extractelement <8 x float> %1131, i64 0, !dbg !7
  %1133 = extractelement <8 x float> %1131, i64 1, !dbg !7
  %1134 = extractelement <8 x float> %1131, i64 2, !dbg !7
  %1135 = extractelement <8 x float> %1131, i64 3, !dbg !7
  %1136 = select i1 %.not, float %1133, float %1132, !dbg !7
  %1137 = bitcast float %1136 to i32, !dbg !7
  %1138 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1075, i32 %1137), !dbg !7
  %1139 = bitcast i32 %1138 to float, !dbg !7
  %1140 = select i1 %.not, float %1132, float %1139, !dbg !7
  %1141 = select i1 %.not, float %1139, float %1133, !dbg !7
  %1142 = select i1 %.not, float %1135, float %1134, !dbg !7
  %1143 = bitcast float %1142 to i32, !dbg !7
  %1144 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1075, i32 %1143), !dbg !7
  %1145 = bitcast i32 %1144 to float, !dbg !7
  %1146 = select i1 %.not, float %1134, float %1145, !dbg !7
  %1147 = select i1 %.not, float %1145, float %1135, !dbg !7
  %1148 = select i1 %.not36, float %1146, float %1140, !dbg !7
  %1149 = bitcast float %1148 to i32, !dbg !7
  %1150 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1091, i32 %1149), !dbg !7
  %1151 = bitcast i32 %1150 to float, !dbg !7
  %1152 = select i1 %.not36, float %1140, float %1151, !dbg !7
  %1153 = select i1 %.not36, float %1151, float %1146, !dbg !7
  %1154 = select i1 %.not36, float %1147, float %1141, !dbg !7
  %1155 = bitcast float %1154 to i32, !dbg !7
  %1156 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1091, i32 %1155), !dbg !7
  %1157 = bitcast i32 %1156 to float, !dbg !7
  %1158 = select i1 %.not36, float %1141, float %1157, !dbg !7
  %1159 = select i1 %.not36, float %1157, float %1147, !dbg !7
  %1160 = extractelement <8 x float> %1131, i64 4, !dbg !7
  %1161 = extractelement <8 x float> %1131, i64 5, !dbg !7
  %1162 = extractelement <8 x float> %1131, i64 6, !dbg !7
  %1163 = extractelement <8 x float> %1131, i64 7, !dbg !7
  %1164 = select i1 %.not, float %1161, float %1160, !dbg !7
  %1165 = bitcast float %1164 to i32, !dbg !7
  %1166 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1075, i32 %1165), !dbg !7
  %1167 = bitcast i32 %1166 to float, !dbg !7
  %1168 = select i1 %.not, float %1160, float %1167, !dbg !7
  %1169 = select i1 %.not, float %1167, float %1161, !dbg !7
  %1170 = select i1 %.not, float %1163, float %1162, !dbg !7
  %1171 = bitcast float %1170 to i32, !dbg !7
  %1172 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1075, i32 %1171), !dbg !7
  %1173 = bitcast i32 %1172 to float, !dbg !7
  %1174 = select i1 %.not, float %1162, float %1173, !dbg !7
  %1175 = select i1 %.not, float %1173, float %1163, !dbg !7
  %1176 = select i1 %.not36, float %1174, float %1168, !dbg !7
  %1177 = bitcast float %1176 to i32, !dbg !7
  %1178 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1091, i32 %1177), !dbg !7
  %1179 = bitcast i32 %1178 to float, !dbg !7
  %1180 = select i1 %.not36, float %1168, float %1179, !dbg !7
  %1181 = select i1 %.not36, float %1179, float %1174, !dbg !7
  %1182 = select i1 %.not36, float %1175, float %1169, !dbg !7
  %1183 = bitcast float %1182 to i32, !dbg !7
  %1184 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1091, i32 %1183), !dbg !7
  %1185 = bitcast i32 %1184 to float, !dbg !7
  %1186 = select i1 %.not36, float %1169, float %1185, !dbg !7
  %1187 = select i1 %.not36, float %1185, float %1175, !dbg !7
  %1188 = extractvalue [2 x <8 x float>] %102, 0, !dbg !7
  %1189 = extractelement <8 x float> %1188, i64 0, !dbg !7
  %1190 = extractelement <8 x float> %1188, i64 1, !dbg !7
  %1191 = extractelement <8 x float> %1188, i64 2, !dbg !7
  %1192 = extractelement <8 x float> %1188, i64 3, !dbg !7
  %1193 = select i1 %.not, float %1190, float %1189, !dbg !7
  %1194 = bitcast float %1193 to i32, !dbg !7
  %1195 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1075, i32 %1194), !dbg !7
  %1196 = bitcast i32 %1195 to float, !dbg !7
  %1197 = select i1 %.not, float %1189, float %1196, !dbg !7
  %1198 = select i1 %.not, float %1196, float %1190, !dbg !7
  %1199 = select i1 %.not, float %1192, float %1191, !dbg !7
  %1200 = bitcast float %1199 to i32, !dbg !7
  %1201 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1075, i32 %1200), !dbg !7
  %1202 = bitcast i32 %1201 to float, !dbg !7
  %1203 = select i1 %.not, float %1191, float %1202, !dbg !7
  %1204 = select i1 %.not, float %1202, float %1192, !dbg !7
  %1205 = select i1 %.not36, float %1203, float %1197, !dbg !7
  %1206 = bitcast float %1205 to i32, !dbg !7
  %1207 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1091, i32 %1206), !dbg !7
  %1208 = bitcast i32 %1207 to float, !dbg !7
  %1209 = select i1 %.not36, float %1197, float %1208, !dbg !7
  %1210 = select i1 %.not36, float %1208, float %1203, !dbg !7
  %1211 = select i1 %.not36, float %1204, float %1198, !dbg !7
  %1212 = bitcast float %1211 to i32, !dbg !7
  %1213 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1091, i32 %1212), !dbg !7
  %1214 = bitcast i32 %1213 to float, !dbg !7
  %1215 = select i1 %.not36, float %1198, float %1214, !dbg !7
  %1216 = select i1 %.not36, float %1214, float %1204, !dbg !7
  %1217 = extractelement <8 x float> %1188, i64 4, !dbg !7
  %1218 = extractelement <8 x float> %1188, i64 5, !dbg !7
  %1219 = extractelement <8 x float> %1188, i64 6, !dbg !7
  %1220 = extractelement <8 x float> %1188, i64 7, !dbg !7
  %1221 = select i1 %.not, float %1218, float %1217, !dbg !7
  %1222 = bitcast float %1221 to i32, !dbg !7
  %1223 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1075, i32 %1222), !dbg !7
  %1224 = bitcast i32 %1223 to float, !dbg !7
  %1225 = select i1 %.not, float %1217, float %1224, !dbg !7
  %1226 = select i1 %.not, float %1224, float %1218, !dbg !7
  %1227 = select i1 %.not, float %1220, float %1219, !dbg !7
  %1228 = bitcast float %1227 to i32, !dbg !7
  %1229 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1075, i32 %1228), !dbg !7
  %1230 = bitcast i32 %1229 to float, !dbg !7
  %1231 = select i1 %.not, float %1219, float %1230, !dbg !7
  %1232 = select i1 %.not, float %1230, float %1220, !dbg !7
  %1233 = select i1 %.not36, float %1231, float %1225, !dbg !7
  %1234 = bitcast float %1233 to i32, !dbg !7
  %1235 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1091, i32 %1234), !dbg !7
  %1236 = bitcast i32 %1235 to float, !dbg !7
  %1237 = select i1 %.not36, float %1225, float %1236, !dbg !7
  %1238 = select i1 %.not36, float %1236, float %1231, !dbg !7
  %1239 = select i1 %.not36, float %1232, float %1226, !dbg !7
  %1240 = bitcast float %1239 to i32, !dbg !7
  %1241 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1091, i32 %1240), !dbg !7
  %1242 = bitcast i32 %1241 to float, !dbg !7
  %1243 = select i1 %.not36, float %1226, float %1242, !dbg !7
  %1244 = select i1 %.not36, float %1242, float %1232, !dbg !7
  %1245 = extractvalue [2 x <8 x float>] %102, 1, !dbg !7
  %1246 = extractelement <8 x float> %1245, i64 0, !dbg !7
  %1247 = extractelement <8 x float> %1245, i64 1, !dbg !7
  %1248 = extractelement <8 x float> %1245, i64 2, !dbg !7
  %1249 = extractelement <8 x float> %1245, i64 3, !dbg !7
  %1250 = select i1 %.not, float %1247, float %1246, !dbg !7
  %1251 = bitcast float %1250 to i32, !dbg !7
  %1252 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1075, i32 %1251), !dbg !7
  %1253 = bitcast i32 %1252 to float, !dbg !7
  %1254 = select i1 %.not, float %1246, float %1253, !dbg !7
  %1255 = select i1 %.not, float %1253, float %1247, !dbg !7
  %1256 = select i1 %.not, float %1249, float %1248, !dbg !7
  %1257 = bitcast float %1256 to i32, !dbg !7
  %1258 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1075, i32 %1257), !dbg !7
  %1259 = bitcast i32 %1258 to float, !dbg !7
  %1260 = select i1 %.not, float %1248, float %1259, !dbg !7
  %1261 = select i1 %.not, float %1259, float %1249, !dbg !7
  %1262 = select i1 %.not36, float %1260, float %1254, !dbg !7
  %1263 = bitcast float %1262 to i32, !dbg !7
  %1264 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1091, i32 %1263), !dbg !7
  %1265 = bitcast i32 %1264 to float, !dbg !7
  %1266 = select i1 %.not36, float %1254, float %1265, !dbg !7
  %1267 = select i1 %.not36, float %1265, float %1260, !dbg !7
  %1268 = select i1 %.not36, float %1261, float %1255, !dbg !7
  %1269 = bitcast float %1268 to i32, !dbg !7
  %1270 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1091, i32 %1269), !dbg !7
  %1271 = bitcast i32 %1270 to float, !dbg !7
  %1272 = select i1 %.not36, float %1255, float %1271, !dbg !7
  %1273 = select i1 %.not36, float %1271, float %1261, !dbg !7
  %1274 = extractelement <8 x float> %1245, i64 4, !dbg !7
  %1275 = extractelement <8 x float> %1245, i64 5, !dbg !7
  %1276 = extractelement <8 x float> %1245, i64 6, !dbg !7
  %1277 = extractelement <8 x float> %1245, i64 7, !dbg !7
  %1278 = select i1 %.not, float %1275, float %1274, !dbg !7
  %1279 = bitcast float %1278 to i32, !dbg !7
  %1280 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1075, i32 %1279), !dbg !7
  %1281 = bitcast i32 %1280 to float, !dbg !7
  %1282 = select i1 %.not, float %1274, float %1281, !dbg !7
  %1283 = select i1 %.not, float %1281, float %1275, !dbg !7
  %1284 = select i1 %.not, float %1277, float %1276, !dbg !7
  %1285 = bitcast float %1284 to i32, !dbg !7
  %1286 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1075, i32 %1285), !dbg !7
  %1287 = bitcast i32 %1286 to float, !dbg !7
  %1288 = select i1 %.not, float %1276, float %1287, !dbg !7
  %1289 = select i1 %.not, float %1287, float %1277, !dbg !7
  %1290 = select i1 %.not36, float %1288, float %1282, !dbg !7
  %1291 = bitcast float %1290 to i32, !dbg !7
  %1292 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1091, i32 %1291), !dbg !7
  %1293 = bitcast i32 %1292 to float, !dbg !7
  %1294 = select i1 %.not36, float %1282, float %1293, !dbg !7
  %1295 = select i1 %.not36, float %1293, float %1288, !dbg !7
  %1296 = select i1 %.not36, float %1289, float %1283, !dbg !7
  %1297 = bitcast float %1296 to i32, !dbg !7
  %1298 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1091, i32 %1297), !dbg !7
  %1299 = bitcast i32 %1298 to float, !dbg !7
  %1300 = select i1 %.not36, float %1283, float %1299, !dbg !7
  %1301 = select i1 %.not36, float %1299, float %1289, !dbg !7
  %1302 = extractvalue [2 x <8 x float>] %101, 0, !dbg !7
  %1303 = extractelement <8 x float> %1302, i64 0, !dbg !7
  %1304 = extractelement <8 x float> %1302, i64 1, !dbg !7
  %1305 = extractelement <8 x float> %1302, i64 2, !dbg !7
  %1306 = extractelement <8 x float> %1302, i64 3, !dbg !7
  %1307 = select i1 %.not, float %1304, float %1303, !dbg !7
  %1308 = bitcast float %1307 to i32, !dbg !7
  %1309 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1075, i32 %1308), !dbg !7
  %1310 = bitcast i32 %1309 to float, !dbg !7
  %1311 = select i1 %.not, float %1303, float %1310, !dbg !7
  %1312 = select i1 %.not, float %1310, float %1304, !dbg !7
  %1313 = select i1 %.not, float %1306, float %1305, !dbg !7
  %1314 = bitcast float %1313 to i32, !dbg !7
  %1315 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1075, i32 %1314), !dbg !7
  %1316 = bitcast i32 %1315 to float, !dbg !7
  %1317 = select i1 %.not, float %1305, float %1316, !dbg !7
  %1318 = select i1 %.not, float %1316, float %1306, !dbg !7
  %1319 = select i1 %.not36, float %1317, float %1311, !dbg !7
  %1320 = bitcast float %1319 to i32, !dbg !7
  %1321 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1091, i32 %1320), !dbg !7
  %1322 = bitcast i32 %1321 to float, !dbg !7
  %1323 = select i1 %.not36, float %1311, float %1322, !dbg !7
  %1324 = select i1 %.not36, float %1322, float %1317, !dbg !7
  %1325 = select i1 %.not36, float %1318, float %1312, !dbg !7
  %1326 = bitcast float %1325 to i32, !dbg !7
  %1327 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1091, i32 %1326), !dbg !7
  %1328 = bitcast i32 %1327 to float, !dbg !7
  %1329 = select i1 %.not36, float %1312, float %1328, !dbg !7
  %1330 = select i1 %.not36, float %1328, float %1318, !dbg !7
  %1331 = extractelement <8 x float> %1302, i64 4, !dbg !7
  %1332 = extractelement <8 x float> %1302, i64 5, !dbg !7
  %1333 = extractelement <8 x float> %1302, i64 6, !dbg !7
  %1334 = extractelement <8 x float> %1302, i64 7, !dbg !7
  %1335 = select i1 %.not, float %1332, float %1331, !dbg !7
  %1336 = bitcast float %1335 to i32, !dbg !7
  %1337 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1075, i32 %1336), !dbg !7
  %1338 = bitcast i32 %1337 to float, !dbg !7
  %1339 = select i1 %.not, float %1331, float %1338, !dbg !7
  %1340 = select i1 %.not, float %1338, float %1332, !dbg !7
  %1341 = select i1 %.not, float %1334, float %1333, !dbg !7
  %1342 = bitcast float %1341 to i32, !dbg !7
  %1343 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1075, i32 %1342), !dbg !7
  %1344 = bitcast i32 %1343 to float, !dbg !7
  %1345 = select i1 %.not, float %1333, float %1344, !dbg !7
  %1346 = select i1 %.not, float %1344, float %1334, !dbg !7
  %1347 = select i1 %.not36, float %1345, float %1339, !dbg !7
  %1348 = bitcast float %1347 to i32, !dbg !7
  %1349 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1091, i32 %1348), !dbg !7
  %1350 = bitcast i32 %1349 to float, !dbg !7
  %1351 = select i1 %.not36, float %1339, float %1350, !dbg !7
  %1352 = select i1 %.not36, float %1350, float %1345, !dbg !7
  %1353 = select i1 %.not36, float %1346, float %1340, !dbg !7
  %1354 = bitcast float %1353 to i32, !dbg !7
  %1355 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1091, i32 %1354), !dbg !7
  %1356 = bitcast i32 %1355 to float, !dbg !7
  %1357 = select i1 %.not36, float %1340, float %1356, !dbg !7
  %1358 = select i1 %.not36, float %1356, float %1346, !dbg !7
  %1359 = extractvalue [2 x <8 x float>] %101, 1, !dbg !7
  %1360 = extractelement <8 x float> %1359, i64 0, !dbg !7
  %1361 = extractelement <8 x float> %1359, i64 1, !dbg !7
  %1362 = extractelement <8 x float> %1359, i64 2, !dbg !7
  %1363 = extractelement <8 x float> %1359, i64 3, !dbg !7
  %1364 = select i1 %.not, float %1361, float %1360, !dbg !7
  %1365 = bitcast float %1364 to i32, !dbg !7
  %1366 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1075, i32 %1365), !dbg !7
  %1367 = bitcast i32 %1366 to float, !dbg !7
  %1368 = select i1 %.not, float %1360, float %1367, !dbg !7
  %1369 = select i1 %.not, float %1367, float %1361, !dbg !7
  %1370 = select i1 %.not, float %1363, float %1362, !dbg !7
  %1371 = bitcast float %1370 to i32, !dbg !7
  %1372 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1075, i32 %1371), !dbg !7
  %1373 = bitcast i32 %1372 to float, !dbg !7
  %1374 = select i1 %.not, float %1362, float %1373, !dbg !7
  %1375 = select i1 %.not, float %1373, float %1363, !dbg !7
  %1376 = select i1 %.not36, float %1374, float %1368, !dbg !7
  %1377 = bitcast float %1376 to i32, !dbg !7
  %1378 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1091, i32 %1377), !dbg !7
  %1379 = bitcast i32 %1378 to float, !dbg !7
  %1380 = select i1 %.not36, float %1368, float %1379, !dbg !7
  %1381 = select i1 %.not36, float %1379, float %1374, !dbg !7
  %1382 = select i1 %.not36, float %1375, float %1369, !dbg !7
  %1383 = bitcast float %1382 to i32, !dbg !7
  %1384 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1091, i32 %1383), !dbg !7
  %1385 = bitcast i32 %1384 to float, !dbg !7
  %1386 = select i1 %.not36, float %1369, float %1385, !dbg !7
  %1387 = select i1 %.not36, float %1385, float %1375, !dbg !7
  %1388 = extractelement <8 x float> %1359, i64 4, !dbg !7
  %1389 = extractelement <8 x float> %1359, i64 5, !dbg !7
  %1390 = extractelement <8 x float> %1359, i64 6, !dbg !7
  %1391 = extractelement <8 x float> %1359, i64 7, !dbg !7
  %1392 = select i1 %.not, float %1389, float %1388, !dbg !7
  %1393 = bitcast float %1392 to i32, !dbg !7
  %1394 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1075, i32 %1393), !dbg !7
  %1395 = bitcast i32 %1394 to float, !dbg !7
  %1396 = select i1 %.not, float %1388, float %1395, !dbg !7
  %1397 = select i1 %.not, float %1395, float %1389, !dbg !7
  %1398 = select i1 %.not, float %1391, float %1390, !dbg !7
  %1399 = bitcast float %1398 to i32, !dbg !7
  %1400 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1075, i32 %1399), !dbg !7
  %1401 = bitcast i32 %1400 to float, !dbg !7
  %1402 = select i1 %.not, float %1390, float %1401, !dbg !7
  %1403 = select i1 %.not, float %1401, float %1391, !dbg !7
  %1404 = select i1 %.not36, float %1402, float %1396, !dbg !7
  %1405 = bitcast float %1404 to i32, !dbg !7
  %1406 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1091, i32 %1405), !dbg !7
  %1407 = bitcast i32 %1406 to float, !dbg !7
  %1408 = select i1 %.not36, float %1396, float %1407, !dbg !7
  %1409 = select i1 %.not36, float %1407, float %1402, !dbg !7
  %1410 = select i1 %.not36, float %1403, float %1397, !dbg !7
  %1411 = bitcast float %1410 to i32, !dbg !7
  %1412 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1091, i32 %1411), !dbg !7
  %1413 = bitcast i32 %1412 to float, !dbg !7
  %1414 = select i1 %.not36, float %1397, float %1413, !dbg !7
  %1415 = select i1 %.not36, float %1413, float %1403, !dbg !7
  %1416 = extractvalue [2 x <8 x float>] %98, 0, !dbg !7
  %1417 = extractelement <8 x float> %1416, i64 0, !dbg !7
  %1418 = extractelement <8 x float> %1416, i64 1, !dbg !7
  %1419 = extractelement <8 x float> %1416, i64 2, !dbg !7
  %1420 = extractelement <8 x float> %1416, i64 3, !dbg !7
  %1421 = select i1 %.not, float %1418, float %1417, !dbg !7
  %1422 = bitcast float %1421 to i32, !dbg !7
  %1423 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1075, i32 %1422), !dbg !7
  %1424 = bitcast i32 %1423 to float, !dbg !7
  %1425 = select i1 %.not, float %1417, float %1424, !dbg !7
  %1426 = select i1 %.not, float %1424, float %1418, !dbg !7
  %1427 = select i1 %.not, float %1420, float %1419, !dbg !7
  %1428 = bitcast float %1427 to i32, !dbg !7
  %1429 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1075, i32 %1428), !dbg !7
  %1430 = bitcast i32 %1429 to float, !dbg !7
  %1431 = select i1 %.not, float %1419, float %1430, !dbg !7
  %1432 = select i1 %.not, float %1430, float %1420, !dbg !7
  %1433 = select i1 %.not36, float %1431, float %1425, !dbg !7
  %1434 = bitcast float %1433 to i32, !dbg !7
  %1435 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1091, i32 %1434), !dbg !7
  %1436 = select i1 %.not36, float %1432, float %1426, !dbg !7
  %1437 = bitcast float %1436 to i32, !dbg !7
  %1438 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1091, i32 %1437), !dbg !7
  %1439 = extractelement <8 x float> %1416, i64 4, !dbg !7
  %1440 = extractelement <8 x float> %1416, i64 5, !dbg !7
  %1441 = extractelement <8 x float> %1416, i64 6, !dbg !7
  %1442 = extractelement <8 x float> %1416, i64 7, !dbg !7
  %1443 = select i1 %.not, float %1440, float %1439, !dbg !7
  %1444 = bitcast float %1443 to i32, !dbg !7
  %1445 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1075, i32 %1444), !dbg !7
  %1446 = bitcast i32 %1445 to float, !dbg !7
  %1447 = select i1 %.not, float %1439, float %1446, !dbg !7
  %1448 = select i1 %.not, float %1446, float %1440, !dbg !7
  %1449 = select i1 %.not, float %1442, float %1441, !dbg !7
  %1450 = bitcast float %1449 to i32, !dbg !7
  %1451 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1075, i32 %1450), !dbg !7
  %1452 = bitcast i32 %1451 to float, !dbg !7
  %1453 = select i1 %.not, float %1441, float %1452, !dbg !7
  %1454 = select i1 %.not, float %1452, float %1442, !dbg !7
  %1455 = select i1 %.not36, float %1453, float %1447, !dbg !7
  %1456 = bitcast float %1455 to i32, !dbg !7
  %1457 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1091, i32 %1456), !dbg !7
  %1458 = select i1 %.not36, float %1454, float %1448, !dbg !7
  %1459 = bitcast float %1458 to i32, !dbg !7
  %1460 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1091, i32 %1459), !dbg !7
  %1461 = extractvalue [2 x <8 x float>] %98, 1, !dbg !7
  %1462 = extractelement <8 x float> %1461, i64 0, !dbg !7
  %1463 = extractelement <8 x float> %1461, i64 1, !dbg !7
  %1464 = extractelement <8 x float> %1461, i64 2, !dbg !7
  %1465 = extractelement <8 x float> %1461, i64 3, !dbg !7
  %1466 = select i1 %.not, float %1463, float %1462, !dbg !7
  %1467 = bitcast float %1466 to i32, !dbg !7
  %1468 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1075, i32 %1467), !dbg !7
  %1469 = bitcast i32 %1468 to float, !dbg !7
  %1470 = select i1 %.not, float %1462, float %1469, !dbg !7
  %1471 = select i1 %.not, float %1469, float %1463, !dbg !7
  %1472 = select i1 %.not, float %1465, float %1464, !dbg !7
  %1473 = bitcast float %1472 to i32, !dbg !7
  %1474 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1075, i32 %1473), !dbg !7
  %1475 = bitcast i32 %1474 to float, !dbg !7
  %1476 = select i1 %.not, float %1464, float %1475, !dbg !7
  %1477 = select i1 %.not, float %1475, float %1465, !dbg !7
  %1478 = select i1 %.not36, float %1476, float %1470, !dbg !7
  %1479 = bitcast float %1478 to i32, !dbg !7
  %1480 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1091, i32 %1479), !dbg !7
  %1481 = select i1 %.not36, float %1477, float %1471, !dbg !7
  %1482 = bitcast float %1481 to i32, !dbg !7
  %1483 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1091, i32 %1482), !dbg !7
  %1484 = extractelement <8 x float> %1461, i64 4, !dbg !7
  %1485 = extractelement <8 x float> %1461, i64 5, !dbg !7
  %1486 = extractelement <8 x float> %1461, i64 6, !dbg !7
  %1487 = extractelement <8 x float> %1461, i64 7, !dbg !7
  %1488 = select i1 %.not, float %1485, float %1484, !dbg !7
  %1489 = bitcast float %1488 to i32, !dbg !7
  %1490 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1075, i32 %1489), !dbg !7
  %1491 = bitcast i32 %1490 to float, !dbg !7
  %1492 = select i1 %.not, float %1484, float %1491, !dbg !7
  %1493 = select i1 %.not, float %1491, float %1485, !dbg !7
  %1494 = select i1 %.not, float %1487, float %1486, !dbg !7
  %1495 = bitcast float %1494 to i32, !dbg !7
  %1496 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1075, i32 %1495), !dbg !7
  %1497 = bitcast i32 %1496 to float, !dbg !7
  %1498 = select i1 %.not, float %1486, float %1497, !dbg !7
  %1499 = select i1 %.not, float %1497, float %1487, !dbg !7
  %1500 = select i1 %.not36, float %1498, float %1492, !dbg !7
  %1501 = bitcast float %1500 to i32, !dbg !7
  %1502 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1091, i32 %1501), !dbg !7
  %1503 = select i1 %.not36, float %1499, float %1493, !dbg !7
  %1504 = bitcast float %1503 to i32, !dbg !7
  %1505 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1091, i32 %1504), !dbg !7
  %1506 = shufflevector <1 x float> %1061, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !16
  %1507 = shufflevector <1 x float> %1056, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !16
  %1508 = bitcast i32 %1502 to float, !dbg !7
  %1509 = select i1 %.not36, float %1492, float %1508, !dbg !7
  %1510 = insertelement <4 x float> poison, float %1509, i64 0, !dbg !16
  %1511 = bitcast i32 %1505 to float, !dbg !7
  %1512 = select i1 %.not36, float %1493, float %1511, !dbg !7
  %1513 = insertelement <4 x float> %1510, float %1512, i64 1, !dbg !16
  %1514 = select i1 %.not36, float %1508, float %1498, !dbg !7
  %1515 = insertelement <4 x float> %1513, float %1514, i64 2, !dbg !16
  %1516 = select i1 %.not36, float %1511, float %1499, !dbg !7
  %1517 = insertelement <4 x float> %1515, float %1516, i64 3, !dbg !16
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1507, !dbg !16
  %normalized = fmul <4 x float> %1517, %row.reciprocal, !dbg !16
  %1518 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !17
  %1519 = extractelement <4 x half> %1518, i64 3, !dbg !17
  %1520 = extractelement <4 x half> %1518, i64 2, !dbg !17
  %1521 = extractelement <4 x half> %1518, i64 1, !dbg !17
  %1522 = extractelement <4 x half> %1518, i64 0, !dbg !17
  %1523 = bitcast i32 %1480 to float, !dbg !7
  %1524 = select i1 %.not36, float %1470, float %1523, !dbg !7
  %1525 = insertelement <4 x float> poison, float %1524, i64 0, !dbg !16
  %1526 = bitcast i32 %1483 to float, !dbg !7
  %1527 = select i1 %.not36, float %1471, float %1526, !dbg !7
  %1528 = insertelement <4 x float> %1525, float %1527, i64 1, !dbg !16
  %1529 = select i1 %.not36, float %1523, float %1476, !dbg !7
  %1530 = insertelement <4 x float> %1528, float %1529, i64 2, !dbg !16
  %1531 = select i1 %.not36, float %1526, float %1477, !dbg !7
  %1532 = insertelement <4 x float> %1530, float %1531, i64 3, !dbg !16
  %normalized40 = fmul <4 x float> %1532, %row.reciprocal, !dbg !16
  %1533 = fptrunc <4 x float> %normalized40 to <4 x half>, !dbg !17
  %1534 = extractelement <4 x half> %1533, i64 3, !dbg !17
  %1535 = extractelement <4 x half> %1533, i64 2, !dbg !17
  %1536 = extractelement <4 x half> %1533, i64 1, !dbg !17
  %1537 = extractelement <4 x half> %1533, i64 0, !dbg !17
  %1538 = bitcast i32 %1457 to float, !dbg !7
  %1539 = select i1 %.not36, float %1447, float %1538, !dbg !7
  %1540 = insertelement <4 x float> poison, float %1539, i64 0, !dbg !16
  %1541 = bitcast i32 %1460 to float, !dbg !7
  %1542 = select i1 %.not36, float %1448, float %1541, !dbg !7
  %1543 = insertelement <4 x float> %1540, float %1542, i64 1, !dbg !16
  %1544 = select i1 %.not36, float %1538, float %1453, !dbg !7
  %1545 = insertelement <4 x float> %1543, float %1544, i64 2, !dbg !16
  %1546 = select i1 %.not36, float %1541, float %1454, !dbg !7
  %1547 = insertelement <4 x float> %1545, float %1546, i64 3, !dbg !16
  %row.reciprocal41 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1506, !dbg !16
  %normalized42 = fmul <4 x float> %1547, %row.reciprocal41, !dbg !16
  %1548 = fptrunc <4 x float> %normalized42 to <4 x half>, !dbg !17
  %1549 = extractelement <4 x half> %1548, i64 3, !dbg !17
  %1550 = extractelement <4 x half> %1548, i64 2, !dbg !17
  %1551 = extractelement <4 x half> %1548, i64 1, !dbg !17
  %1552 = extractelement <4 x half> %1548, i64 0, !dbg !17
  %1553 = bitcast i32 %1435 to float, !dbg !7
  %1554 = select i1 %.not36, float %1425, float %1553, !dbg !7
  %1555 = insertelement <4 x float> poison, float %1554, i64 0, !dbg !16
  %1556 = bitcast i32 %1438 to float, !dbg !7
  %1557 = select i1 %.not36, float %1426, float %1556, !dbg !7
  %1558 = insertelement <4 x float> %1555, float %1557, i64 1, !dbg !16
  %1559 = select i1 %.not36, float %1553, float %1431, !dbg !7
  %1560 = insertelement <4 x float> %1558, float %1559, i64 2, !dbg !16
  %1561 = select i1 %.not36, float %1556, float %1432, !dbg !7
  %1562 = insertelement <4 x float> %1560, float %1561, i64 3, !dbg !16
  %normalized43 = fmul <4 x float> %1562, %row.reciprocal41, !dbg !16
  %1563 = fptrunc <4 x float> %normalized43 to <4 x half>, !dbg !17
  %1564 = extractelement <4 x half> %1563, i64 3, !dbg !17
  %1565 = extractelement <4 x half> %1563, i64 2, !dbg !17
  %1566 = extractelement <4 x half> %1563, i64 1, !dbg !17
  %1567 = extractelement <4 x half> %1563, i64 0, !dbg !17
  %address.thread977 = call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #9, !dbg !18, !range !32, !frisk.register_lifetimes !1
  %address.local978 = zext nneg i32 %address.thread977 to i64, !dbg !8
  %address.local979 = add nuw i64 %23, %address.local978, !dbg !18
  %1568 = lshr i64 %address.local978, 2, !dbg !18
  %address.local981 = and i64 %1568, 48, !dbg !18
  %address.local982 = add i64 %address.local979, %address.local981, !dbg !18
  %address.local983 = and i64 %address.local978, 240, !dbg !18
  %address.local984 = sub i64 %address.local982, %address.local983, !dbg !18
  %address.local985 = add i64 %address.local984, %24, !dbg !18
  %address.local986 = shl nuw nsw i64 %address.local985, 7, !dbg !18
  %1569 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %37, !dbg !18
  %address.local988 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %1569, i64 %address.local986, !dbg !18
  %address.local990 = and i64 %1568, 12, !dbg !11
  %address.local991 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local988, i64 %address.local990, !dbg !18
  store half %1567, ptr addrspace(1) %address.local991, align 2, !dbg !18
  %address.local993 = or disjoint i64 %address.local990, 1, !dbg !18
  %address.local994 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local988, i64 %address.local993, !dbg !18
  store half %1566, ptr addrspace(1) %address.local994, align 2, !dbg !18
  %address.local996 = or disjoint i64 %address.local990, 2, !dbg !18
  %address.local997 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local988, i64 %address.local996, !dbg !18
  store half %1565, ptr addrspace(1) %address.local997, align 2, !dbg !18
  %address.local999 = or disjoint i64 %address.local990, 3, !dbg !18
  %address.local1000 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local988, i64 %address.local999, !dbg !18
  store half %1564, ptr addrspace(1) %address.local1000, align 2, !dbg !18
  %address.local1002 = or disjoint i64 %address.local990, 16, !dbg !18
  %address.local1003 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local988, i64 %address.local1002, !dbg !18
  store half %1552, ptr addrspace(1) %address.local1003, align 2, !dbg !18
  %address.local1005 = or disjoint i64 %address.local990, 17, !dbg !18
  %address.local1006 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local988, i64 %address.local1005, !dbg !18
  store half %1551, ptr addrspace(1) %address.local1006, align 2, !dbg !18
  %address.local1008 = or disjoint i64 %address.local990, 18, !dbg !18
  %address.local1009 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local988, i64 %address.local1008, !dbg !18
  store half %1550, ptr addrspace(1) %address.local1009, align 2, !dbg !18
  %address.local1011 = or disjoint i64 %address.local990, 19, !dbg !18
  %address.local1012 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local988, i64 %address.local1011, !dbg !18
  store half %1549, ptr addrspace(1) %address.local1012, align 2, !dbg !18
  %address.local1016 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local988, i64 16384, !dbg !18
  %address.local1017 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1016, i64 %address.local990, !dbg !18
  store half %1537, ptr addrspace(1) %address.local1017, align 2, !dbg !18
  %address.local1019 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1016, i64 %address.local993, !dbg !18
  store half %1536, ptr addrspace(1) %address.local1019, align 2, !dbg !18
  %address.local1021 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1016, i64 %address.local996, !dbg !18
  store half %1535, ptr addrspace(1) %address.local1021, align 2, !dbg !18
  %address.local1023 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1016, i64 %address.local999, !dbg !18
  store half %1534, ptr addrspace(1) %address.local1023, align 2, !dbg !18
  %address.local1025 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1016, i64 %address.local1002, !dbg !18
  store half %1522, ptr addrspace(1) %address.local1025, align 2, !dbg !18
  %address.local1027 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1016, i64 %address.local1005, !dbg !18
  store half %1521, ptr addrspace(1) %address.local1027, align 2, !dbg !18
  %address.local1029 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1016, i64 %address.local1008, !dbg !18
  store half %1520, ptr addrspace(1) %address.local1029, align 2, !dbg !18
  %address.local1031 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1016, i64 %address.local1011, !dbg !18
  store half %1519, ptr addrspace(1) %address.local1031, align 2, !dbg !18
  %1570 = insertelement <4 x float> poison, float %1408, i64 0, !dbg !16
  %1571 = insertelement <4 x float> %1570, float %1414, i64 1, !dbg !16
  %1572 = insertelement <4 x float> %1571, float %1409, i64 2, !dbg !16
  %1573 = insertelement <4 x float> %1572, float %1415, i64 3, !dbg !16
  %normalized45 = fmul <4 x float> %1573, %row.reciprocal, !dbg !16
  %1574 = fptrunc <4 x float> %normalized45 to <4 x half>, !dbg !17
  %1575 = extractelement <4 x half> %1574, i64 3, !dbg !17
  %1576 = extractelement <4 x half> %1574, i64 2, !dbg !17
  %1577 = extractelement <4 x half> %1574, i64 1, !dbg !17
  %1578 = extractelement <4 x half> %1574, i64 0, !dbg !17
  %1579 = insertelement <4 x float> poison, float %1380, i64 0, !dbg !16
  %1580 = insertelement <4 x float> %1579, float %1386, i64 1, !dbg !16
  %1581 = insertelement <4 x float> %1580, float %1381, i64 2, !dbg !16
  %1582 = insertelement <4 x float> %1581, float %1387, i64 3, !dbg !16
  %normalized46 = fmul <4 x float> %1582, %row.reciprocal, !dbg !16
  %1583 = fptrunc <4 x float> %normalized46 to <4 x half>, !dbg !17
  %1584 = extractelement <4 x half> %1583, i64 3, !dbg !17
  %1585 = extractelement <4 x half> %1583, i64 2, !dbg !17
  %1586 = extractelement <4 x half> %1583, i64 1, !dbg !17
  %1587 = extractelement <4 x half> %1583, i64 0, !dbg !17
  %1588 = insertelement <4 x float> poison, float %1351, i64 0, !dbg !16
  %1589 = insertelement <4 x float> %1588, float %1357, i64 1, !dbg !16
  %1590 = insertelement <4 x float> %1589, float %1352, i64 2, !dbg !16
  %1591 = insertelement <4 x float> %1590, float %1358, i64 3, !dbg !16
  %normalized48 = fmul <4 x float> %1591, %row.reciprocal41, !dbg !16
  %1592 = fptrunc <4 x float> %normalized48 to <4 x half>, !dbg !17
  %1593 = extractelement <4 x half> %1592, i64 3, !dbg !17
  %1594 = extractelement <4 x half> %1592, i64 2, !dbg !17
  %1595 = extractelement <4 x half> %1592, i64 1, !dbg !17
  %1596 = extractelement <4 x half> %1592, i64 0, !dbg !17
  %1597 = insertelement <4 x float> poison, float %1323, i64 0, !dbg !16
  %1598 = insertelement <4 x float> %1597, float %1329, i64 1, !dbg !16
  %1599 = insertelement <4 x float> %1598, float %1324, i64 2, !dbg !16
  %1600 = insertelement <4 x float> %1599, float %1330, i64 3, !dbg !16
  %normalized49 = fmul <4 x float> %1600, %row.reciprocal41, !dbg !16
  %1601 = fptrunc <4 x float> %normalized49 to <4 x half>, !dbg !17
  %1602 = extractelement <4 x half> %1601, i64 3, !dbg !17
  %1603 = extractelement <4 x half> %1601, i64 2, !dbg !17
  %1604 = extractelement <4 x half> %1601, i64 1, !dbg !17
  %1605 = extractelement <4 x half> %1601, i64 0, !dbg !17
  %address.local1046 = or disjoint i64 %address.local990, 32, !dbg !18
  %address.local1047 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local988, i64 %address.local1046, !dbg !18
  store half %1605, ptr addrspace(1) %address.local1047, align 2, !dbg !18
  %address.local1049 = or disjoint i64 %address.local990, 33, !dbg !18
  %address.local1050 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local988, i64 %address.local1049, !dbg !18
  store half %1604, ptr addrspace(1) %address.local1050, align 2, !dbg !18
  %address.local1052 = or disjoint i64 %address.local990, 34, !dbg !18
  %address.local1053 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local988, i64 %address.local1052, !dbg !18
  store half %1603, ptr addrspace(1) %address.local1053, align 2, !dbg !18
  %address.local1055 = or disjoint i64 %address.local990, 35, !dbg !18
  %address.local1056 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local988, i64 %address.local1055, !dbg !18
  store half %1602, ptr addrspace(1) %address.local1056, align 2, !dbg !18
  %address.local1058 = or disjoint i64 %address.local990, 48, !dbg !18
  %address.local1059 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local988, i64 %address.local1058, !dbg !18
  store half %1596, ptr addrspace(1) %address.local1059, align 2, !dbg !18
  %address.local1061 = or disjoint i64 %address.local990, 49, !dbg !18
  %address.local1062 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local988, i64 %address.local1061, !dbg !18
  store half %1595, ptr addrspace(1) %address.local1062, align 2, !dbg !18
  %address.local1064 = or disjoint i64 %address.local990, 50, !dbg !18
  %address.local1065 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local988, i64 %address.local1064, !dbg !18
  store half %1594, ptr addrspace(1) %address.local1065, align 2, !dbg !18
  %address.local1067 = or i64 %1568, 51, !dbg !18
  %address.local1068 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local988, i64 %address.local1067, !dbg !18
  store half %1593, ptr addrspace(1) %address.local1068, align 2, !dbg !18
  %address.local1073 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1016, i64 %address.local1046, !dbg !18
  store half %1587, ptr addrspace(1) %address.local1073, align 2, !dbg !18
  %address.local1075 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1016, i64 %address.local1049, !dbg !18
  store half %1586, ptr addrspace(1) %address.local1075, align 2, !dbg !18
  %address.local1077 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1016, i64 %address.local1052, !dbg !18
  store half %1585, ptr addrspace(1) %address.local1077, align 2, !dbg !18
  %address.local1079 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1016, i64 %address.local1055, !dbg !18
  store half %1584, ptr addrspace(1) %address.local1079, align 2, !dbg !18
  %address.local1081 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1016, i64 %address.local1058, !dbg !18
  store half %1578, ptr addrspace(1) %address.local1081, align 2, !dbg !18
  %address.local1083 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1016, i64 %address.local1061, !dbg !18
  store half %1577, ptr addrspace(1) %address.local1083, align 2, !dbg !18
  %address.local1085 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1016, i64 %address.local1064, !dbg !18
  store half %1576, ptr addrspace(1) %address.local1085, align 2, !dbg !18
  %address.local1087 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1016, i64 %address.local1067, !dbg !18
  store half %1575, ptr addrspace(1) %address.local1087, align 2, !dbg !18
  %1606 = insertelement <4 x float> poison, float %1294, i64 0, !dbg !16
  %1607 = insertelement <4 x float> %1606, float %1300, i64 1, !dbg !16
  %1608 = insertelement <4 x float> %1607, float %1295, i64 2, !dbg !16
  %1609 = insertelement <4 x float> %1608, float %1301, i64 3, !dbg !16
  %normalized51 = fmul <4 x float> %1609, %row.reciprocal, !dbg !16
  %1610 = fptrunc <4 x float> %normalized51 to <4 x half>, !dbg !17
  %1611 = extractelement <4 x half> %1610, i64 3, !dbg !17
  %1612 = extractelement <4 x half> %1610, i64 2, !dbg !17
  %1613 = extractelement <4 x half> %1610, i64 1, !dbg !17
  %1614 = extractelement <4 x half> %1610, i64 0, !dbg !17
  %1615 = insertelement <4 x float> poison, float %1266, i64 0, !dbg !16
  %1616 = insertelement <4 x float> %1615, float %1272, i64 1, !dbg !16
  %1617 = insertelement <4 x float> %1616, float %1267, i64 2, !dbg !16
  %1618 = insertelement <4 x float> %1617, float %1273, i64 3, !dbg !16
  %normalized52 = fmul <4 x float> %1618, %row.reciprocal, !dbg !16
  %1619 = fptrunc <4 x float> %normalized52 to <4 x half>, !dbg !17
  %1620 = extractelement <4 x half> %1619, i64 3, !dbg !17
  %1621 = extractelement <4 x half> %1619, i64 2, !dbg !17
  %1622 = extractelement <4 x half> %1619, i64 1, !dbg !17
  %1623 = extractelement <4 x half> %1619, i64 0, !dbg !17
  %1624 = insertelement <4 x float> poison, float %1237, i64 0, !dbg !16
  %1625 = insertelement <4 x float> %1624, float %1243, i64 1, !dbg !16
  %1626 = insertelement <4 x float> %1625, float %1238, i64 2, !dbg !16
  %1627 = insertelement <4 x float> %1626, float %1244, i64 3, !dbg !16
  %normalized54 = fmul <4 x float> %1627, %row.reciprocal41, !dbg !16
  %1628 = fptrunc <4 x float> %normalized54 to <4 x half>, !dbg !17
  %1629 = extractelement <4 x half> %1628, i64 3, !dbg !17
  %1630 = extractelement <4 x half> %1628, i64 2, !dbg !17
  %1631 = extractelement <4 x half> %1628, i64 1, !dbg !17
  %1632 = extractelement <4 x half> %1628, i64 0, !dbg !17
  %1633 = insertelement <4 x float> poison, float %1209, i64 0, !dbg !16
  %1634 = insertelement <4 x float> %1633, float %1215, i64 1, !dbg !16
  %1635 = insertelement <4 x float> %1634, float %1210, i64 2, !dbg !16
  %1636 = insertelement <4 x float> %1635, float %1216, i64 3, !dbg !16
  %normalized55 = fmul <4 x float> %1636, %row.reciprocal41, !dbg !16
  %1637 = fptrunc <4 x float> %normalized55 to <4 x half>, !dbg !17
  %1638 = extractelement <4 x half> %1637, i64 3, !dbg !17
  %1639 = extractelement <4 x half> %1637, i64 2, !dbg !17
  %1640 = extractelement <4 x half> %1637, i64 1, !dbg !17
  %1641 = extractelement <4 x half> %1637, i64 0, !dbg !17
  %address.local1102 = or disjoint i64 %address.local990, 64, !dbg !18
  %address.local1103 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local988, i64 %address.local1102, !dbg !18
  store half %1641, ptr addrspace(1) %address.local1103, align 2, !dbg !18
  %address.local1105 = or disjoint i64 %address.local990, 65, !dbg !18
  %address.local1106 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local988, i64 %address.local1105, !dbg !18
  store half %1640, ptr addrspace(1) %address.local1106, align 2, !dbg !18
  %address.local1108 = or disjoint i64 %address.local990, 66, !dbg !18
  %address.local1109 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local988, i64 %address.local1108, !dbg !18
  store half %1639, ptr addrspace(1) %address.local1109, align 2, !dbg !18
  %address.local1111 = or disjoint i64 %address.local990, 67, !dbg !18
  %address.local1112 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local988, i64 %address.local1111, !dbg !18
  store half %1638, ptr addrspace(1) %address.local1112, align 2, !dbg !18
  %address.local1114 = or disjoint i64 %address.local990, 80, !dbg !18
  %address.local1115 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local988, i64 %address.local1114, !dbg !18
  store half %1632, ptr addrspace(1) %address.local1115, align 2, !dbg !18
  %address.local1117 = or disjoint i64 %address.local990, 81, !dbg !18
  %address.local1118 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local988, i64 %address.local1117, !dbg !18
  store half %1631, ptr addrspace(1) %address.local1118, align 2, !dbg !18
  %address.local1120 = or disjoint i64 %address.local990, 82, !dbg !18
  %address.local1121 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local988, i64 %address.local1120, !dbg !18
  store half %1630, ptr addrspace(1) %address.local1121, align 2, !dbg !18
  %address.local1123 = or disjoint i64 %address.local990, 83, !dbg !18
  %address.local1124 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local988, i64 %address.local1123, !dbg !18
  store half %1629, ptr addrspace(1) %address.local1124, align 2, !dbg !18
  %address.local1129 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1016, i64 %address.local1102, !dbg !18
  store half %1623, ptr addrspace(1) %address.local1129, align 2, !dbg !18
  %address.local1131 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1016, i64 %address.local1105, !dbg !18
  store half %1622, ptr addrspace(1) %address.local1131, align 2, !dbg !18
  %address.local1133 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1016, i64 %address.local1108, !dbg !18
  store half %1621, ptr addrspace(1) %address.local1133, align 2, !dbg !18
  %address.local1135 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1016, i64 %address.local1111, !dbg !18
  store half %1620, ptr addrspace(1) %address.local1135, align 2, !dbg !18
  %address.local1137 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1016, i64 %address.local1114, !dbg !18
  store half %1614, ptr addrspace(1) %address.local1137, align 2, !dbg !18
  %address.local1139 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1016, i64 %address.local1117, !dbg !18
  store half %1613, ptr addrspace(1) %address.local1139, align 2, !dbg !18
  %address.local1141 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1016, i64 %address.local1120, !dbg !18
  store half %1612, ptr addrspace(1) %address.local1141, align 2, !dbg !18
  %address.local1143 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1016, i64 %address.local1123, !dbg !18
  store half %1611, ptr addrspace(1) %address.local1143, align 2, !dbg !18
  %1642 = insertelement <4 x float> poison, float %1180, i64 0, !dbg !16
  %1643 = insertelement <4 x float> %1642, float %1186, i64 1, !dbg !16
  %1644 = insertelement <4 x float> %1643, float %1181, i64 2, !dbg !16
  %1645 = insertelement <4 x float> %1644, float %1187, i64 3, !dbg !16
  %normalized57 = fmul <4 x float> %1645, %row.reciprocal, !dbg !16
  %1646 = fptrunc <4 x float> %normalized57 to <4 x half>, !dbg !17
  %1647 = extractelement <4 x half> %1646, i64 3, !dbg !17
  %1648 = extractelement <4 x half> %1646, i64 2, !dbg !17
  %1649 = extractelement <4 x half> %1646, i64 1, !dbg !17
  %1650 = extractelement <4 x half> %1646, i64 0, !dbg !17
  %1651 = insertelement <4 x float> poison, float %1152, i64 0, !dbg !16
  %1652 = insertelement <4 x float> %1651, float %1158, i64 1, !dbg !16
  %1653 = insertelement <4 x float> %1652, float %1153, i64 2, !dbg !16
  %1654 = insertelement <4 x float> %1653, float %1159, i64 3, !dbg !16
  %normalized58 = fmul <4 x float> %1654, %row.reciprocal, !dbg !16
  %1655 = fptrunc <4 x float> %normalized58 to <4 x half>, !dbg !17
  %1656 = extractelement <4 x half> %1655, i64 3, !dbg !17
  %1657 = extractelement <4 x half> %1655, i64 2, !dbg !17
  %1658 = extractelement <4 x half> %1655, i64 1, !dbg !17
  %1659 = extractelement <4 x half> %1655, i64 0, !dbg !17
  %1660 = insertelement <4 x float> poison, float %1123, i64 0, !dbg !16
  %1661 = insertelement <4 x float> %1660, float %1129, i64 1, !dbg !16
  %1662 = insertelement <4 x float> %1661, float %1124, i64 2, !dbg !16
  %1663 = insertelement <4 x float> %1662, float %1130, i64 3, !dbg !16
  %normalized60 = fmul <4 x float> %1663, %row.reciprocal41, !dbg !16
  %1664 = fptrunc <4 x float> %normalized60 to <4 x half>, !dbg !17
  %1665 = extractelement <4 x half> %1664, i64 3, !dbg !17
  %1666 = extractelement <4 x half> %1664, i64 2, !dbg !17
  %1667 = extractelement <4 x half> %1664, i64 1, !dbg !17
  %1668 = extractelement <4 x half> %1664, i64 0, !dbg !17
  %1669 = insertelement <4 x float> poison, float %1095, i64 0, !dbg !16
  %1670 = insertelement <4 x float> %1669, float %1101, i64 1, !dbg !16
  %1671 = insertelement <4 x float> %1670, float %1096, i64 2, !dbg !16
  %1672 = insertelement <4 x float> %1671, float %1102, i64 3, !dbg !16
  %normalized61 = fmul <4 x float> %1672, %row.reciprocal41, !dbg !16
  %1673 = fptrunc <4 x float> %normalized61 to <4 x half>, !dbg !17
  %1674 = extractelement <4 x half> %1673, i64 3, !dbg !17
  %1675 = extractelement <4 x half> %1673, i64 2, !dbg !17
  %1676 = extractelement <4 x half> %1673, i64 1, !dbg !17
  %1677 = extractelement <4 x half> %1673, i64 0, !dbg !17
  %address.local1158 = or disjoint i64 %address.local990, 96, !dbg !18
  %address.local1159 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local988, i64 %address.local1158, !dbg !18
  store half %1677, ptr addrspace(1) %address.local1159, align 2, !dbg !18
  %address.local1161 = or disjoint i64 %address.local990, 97, !dbg !18
  %address.local1162 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local988, i64 %address.local1161, !dbg !18
  store half %1676, ptr addrspace(1) %address.local1162, align 2, !dbg !18
  %address.local1164 = or disjoint i64 %address.local990, 98, !dbg !18
  %address.local1165 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local988, i64 %address.local1164, !dbg !18
  store half %1675, ptr addrspace(1) %address.local1165, align 2, !dbg !18
  %address.local1167 = or disjoint i64 %address.local990, 99, !dbg !18
  %address.local1168 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local988, i64 %address.local1167, !dbg !18
  store half %1674, ptr addrspace(1) %address.local1168, align 2, !dbg !18
  %address.local1170 = or disjoint i64 %address.local990, 112, !dbg !18
  %address.local1171 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local988, i64 %address.local1170, !dbg !18
  store half %1668, ptr addrspace(1) %address.local1171, align 2, !dbg !18
  %address.local1173 = or disjoint i64 %address.local990, 113, !dbg !18
  %address.local1174 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local988, i64 %address.local1173, !dbg !18
  store half %1667, ptr addrspace(1) %address.local1174, align 2, !dbg !18
  %address.local1176 = or disjoint i64 %address.local990, 114, !dbg !18
  %address.local1177 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local988, i64 %address.local1176, !dbg !18
  store half %1666, ptr addrspace(1) %address.local1177, align 2, !dbg !18
  %address.local1179 = or i64 %1568, 115, !dbg !18
  %address.local1180 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local988, i64 %address.local1179, !dbg !18
  store half %1665, ptr addrspace(1) %address.local1180, align 2, !dbg !18
  %address.local1185 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1016, i64 %address.local1158, !dbg !18
  store half %1659, ptr addrspace(1) %address.local1185, align 2, !dbg !18
  %address.local1187 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1016, i64 %address.local1161, !dbg !18
  store half %1658, ptr addrspace(1) %address.local1187, align 2, !dbg !18
  %address.local1189 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1016, i64 %address.local1164, !dbg !18
  store half %1657, ptr addrspace(1) %address.local1189, align 2, !dbg !18
  %address.local1191 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1016, i64 %address.local1167, !dbg !18
  store half %1656, ptr addrspace(1) %address.local1191, align 2, !dbg !18
  %address.local1193 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1016, i64 %address.local1170, !dbg !18
  store half %1650, ptr addrspace(1) %address.local1193, align 2, !dbg !18
  %address.local1195 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1016, i64 %address.local1173, !dbg !18
  store half %1649, ptr addrspace(1) %address.local1195, align 2, !dbg !18
  %address.local1197 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1016, i64 %address.local1176, !dbg !18
  store half %1648, ptr addrspace(1) %address.local1197, align 2, !dbg !18
  %address.local1199 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1016, i64 %address.local1179, !dbg !18
  store half %1647, ptr addrspace(1) %address.local1199, align 2, !dbg !18
  ret void, !dbg !5
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.y() #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.z() #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #2

; Function Attrs: convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.readfirstlane.i32(i32) #3

; Function Attrs: nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.asyncmark() #4

; Function Attrs: nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.wait.asyncmark(i16 immarg) #4

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier() #5

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32) #6

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.hi(i32, i32) #6

; Function Attrs: convergent nocallback nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.ds.bpermute(i32, i32) #7

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maxnum.f32(float, float) #8

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare <1 x float> @llvm.maxnum.v1f32(<1 x float>, <1 x float>) #8

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.amdgcn.exp2.f32(float) #8

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.sched.barrier(i32 immarg) #5

attributes #0 = { convergent willreturn memory(read) }
attributes #1 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1" "frisk.cross_bn_k0"="sum-anchor-v1" "frisk.global_prefetch"="8" "frisk.k_dma"="64-swizzle:1" "frisk.k_lds_prefetch"="8" "frisk.k_tail_prefetch"="4" "frisk.register_comparisons"="1" "frisk.register_lifetimes"="1" "frisk.softmax_fragments"="pv" "uniform-work-group-size" }
attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none) }
attributes #4 = { nocallback nofree nounwind willreturn }
attributes #5 = { convergent nocallback nofree nounwind willreturn }
attributes #6 = { nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }
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
!7 = !DILocation(line: 57, column: 24, scope: !4)
!8 = !DILocation(line: 67, column: 22, scope: !4)
!11 = !DILocation(line: 69, column: 24, scope: !4)
!12 = !DILocation(line: 68, column: 22, scope: !4)
!13 = !DILocation(line: 87, column: 23, scope: !4)
!14 = !DILocation(line: 70, column: 26, scope: !4)
!15 = !DILocation(line: 86, column: 20, scope: !4)
!16 = !DILocation(line: 92, column: 23, scope: !4)
!17 = !DILocation(line: 93, column: 19, scope: !4)
!18 = !DILocation(line: 94, column: 9, scope: !4)
!19 = !DILocation(line: 71, column: 23, scope: !4)
!20 = !DILocation(line: 74, column: 20, scope: !4)
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
