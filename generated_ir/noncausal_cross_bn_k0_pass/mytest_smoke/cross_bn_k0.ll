; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/noncausal_softmax_fragments/baseline.input.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/noncausal_softmax_fragments/baseline.input.mlir"

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
  %17 = lshr i64 %12, 10, !dbg !8
  %18 = add nuw nsw i64 %13, %17, !dbg !8
  %19 = shl nsw i64 %16, 7, !dbg !8
  %20 = mul nsw i64 %17, -4096, !dbg !8
  %21 = add i64 %19, %20, !dbg !8
  %22 = shl i64 %18, 19, !dbg !8
  %23 = and i64 %22, 16252928, !dbg !8
  %24 = shl nsw i64 %21, 7, !dbg !8
  %25 = add i64 %23, %24, !dbg !8
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #9, !dbg !8, !range !32, !frisk.register_lifetimes !1
  %address.local = zext nneg i32 %address.thread to i64, !dbg !8
  %address.local404 = shl nuw nsw i64 %address.local, 3, !dbg !8
  %address.local405 = and i64 %address.local404, 1536, !dbg !8
  %address.local406 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local405, !dbg !8
  %address.local407 = ptrtoint ptr addrspace(3) %address.local406 to i64, !dbg !8
  %address.local408 = trunc i64 %address.local407 to i32, !dbg !8
  %26 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local408), !dbg !8
  %address.local409 = shl nuw nsw i64 %address.local, 7, !dbg !8
  %address.local410 = and i64 %address.local409, 16256, !dbg !8
  %address.local411 = lshr i64 %address.local, 4, !dbg !8
  %address.local412 = and i64 %address.local411, 8, !dbg !8
  %address.local413 = or disjoint i64 %address.local410, %address.local412, !dbg !8
  %27 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %.idx1399 = or disjoint i64 %25, %address.local413, !dbg !8
  %.idx1399.tr = trunc i64 %.idx1399 to i32, !dbg !8
  %dma.buffer.offset1396 = shl i32 %.idx1399.tr, 1, !dbg !8
  %28 = insertelement <2 x i64> poison, i64 %27, i64 0, !dbg !8
  %29 = bitcast <2 x i64> %28 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc1397 = shufflevector <4 x i32> %29, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
  %dma.buffer.lds1398 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %26), !dbg !8
  %30 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1398, i32 %dma.buffer.offset1396, <4 x i32> %dma.buffer.rsrc1397) #10, !dbg !8
  %address.local415 = or disjoint i64 %address.local, 256, !dbg !8
  %address.local416 = shl nuw nsw i64 %address.local415, 3, !dbg !8
  %address.local417 = and i64 %address.local416, 3584, !dbg !8
  %address.local418 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local417, !dbg !8
  %address.local419 = ptrtoint ptr addrspace(3) %address.local418 to i64, !dbg !8
  %address.local420 = trunc i64 %address.local419 to i32, !dbg !8
  %31 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local420), !dbg !8
  %address.local423 = lshr i64 %address.local415, 4, !dbg !8
  %address.local424 = and i64 %address.local423, 24, !dbg !8
  %address.local425 = or disjoint i64 %address.local410, %address.local424, !dbg !8
  %.idx1400 = or disjoint i64 %25, %address.local425, !dbg !8
  %.idx1400.tr = trunc i64 %.idx1400 to i32, !dbg !8
  %dma.buffer.offset1393 = shl i32 %.idx1400.tr, 1, !dbg !8
  %dma.buffer.lds1395 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %31), !dbg !8
  %32 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1395, i32 %dma.buffer.offset1393, <4 x i32> %dma.buffer.rsrc1397) #10, !dbg !8
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %address.local427 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local405, !dbg !8
  %address.local428 = ptrtoint ptr addrspace(3) %address.local427 to i64, !dbg !8
  %address.local429 = trunc i64 %address.local428 to i32, !dbg !8
  %33 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local429), !dbg !8
  %34 = or disjoint i64 %25, 32, !dbg !8
  %35 = or disjoint i64 %address.local413, %34, !dbg !8
  %.tr = trunc i64 %35 to i32, !dbg !8
  %dma.buffer.offset1390 = shl i32 %.tr, 1, !dbg !8
  %dma.buffer.lds1392 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %33), !dbg !8
  %36 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1392, i32 %dma.buffer.offset1390, <4 x i32> %dma.buffer.rsrc1397) #10, !dbg !8
  %address.local431 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local417, !dbg !8
  %address.local432 = ptrtoint ptr addrspace(3) %address.local431 to i64, !dbg !8
  %address.local433 = trunc i64 %address.local432 to i32, !dbg !8
  %37 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local433), !dbg !8
  %38 = or disjoint i64 %address.local425, %34, !dbg !8
  %.tr1401 = trunc i64 %38 to i32, !dbg !8
  %dma.buffer.offset1387 = shl i32 %.tr1401, 1, !dbg !8
  %dma.buffer.lds1389 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %37), !dbg !8
  %39 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1389, i32 %dma.buffer.offset1387, <4 x i32> %dma.buffer.rsrc1397) #10, !dbg !8
  %address.local435 = lshr i32 %address.thread, 2, !dbg !10
  %address.local436 = and i32 %address.local435, 4, !dbg !10
  %address.local437 = shl nuw nsw i32 %address.thread, 1, !dbg !10
  %address.local438 = and i32 %address.local437, 384, !dbg !10
  %address.local439 = or disjoint i32 %address.local436, %address.local438, !dbg !10
  %address.local440 = shl nuw nsw i32 %address.thread, 3, !dbg !10
  %address.local441 = and i32 %address.local440, 120, !dbg !10
  %address.local442 = or disjoint i32 %address.local439, %address.local441, !dbg !10
  %address.local443 = shl nuw nsw i32 %address.thread, 5, !dbg !10
  %address.local444 = and i32 %address.local443, 1024, !dbg !10
  %address.local445 = or disjoint i32 %address.local442, %address.local444, !dbg !10
  %address.local446 = zext nneg i32 %address.local445 to i64, !dbg !10
  %address.local447 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local446, !dbg !10
  %40 = load <4 x half>, ptr addrspace(3) %address.local447, align 8, !dbg !10
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local447, i64 4096, !dbg !10
  %41 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !10
  %42 = fmul <4 x half> %40, splat (half 0xH3015), !dbg !10
  %lds.const.gep1197 = getelementptr i8, ptr addrspace(3) %address.local447, i64 1024, !dbg !10
  %43 = load <4 x half>, ptr addrspace(3) %lds.const.gep1197, align 8, !dbg !10
  %44 = fmul <4 x half> %41, splat (half 0xH3015), !dbg !10
  %lds.const.gep1198 = getelementptr i8, ptr addrspace(3) %address.local447, i64 5120, !dbg !10
  %45 = load <4 x half>, ptr addrspace(3) %lds.const.gep1198, align 8, !dbg !10
  %46 = fmul <4 x half> %43, splat (half 0xH3015), !dbg !10
  %47 = fmul <4 x half> %45, splat (half 0xH3015), !dbg !10
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %48 = or disjoint i64 %25, 64, !dbg !8
  %49 = or disjoint i64 %address.local413, %48, !dbg !8
  %.tr1402 = trunc i64 %49 to i32, !dbg !8
  %dma.buffer.offset1384 = shl i32 %.tr1402, 1, !dbg !8
  %50 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1398, i32 %dma.buffer.offset1384, <4 x i32> %dma.buffer.rsrc1397) #10, !dbg !8
  %51 = or disjoint i64 %address.local425, %48, !dbg !8
  %.tr1403 = trunc i64 %51 to i32, !dbg !8
  %dma.buffer.offset1381 = shl i32 %.tr1403, 1, !dbg !8
  %52 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1395, i32 %dma.buffer.offset1381, <4 x i32> %dma.buffer.rsrc1397) #10, !dbg !8
  %lds.const.gep1199 = getelementptr i8, ptr addrspace(3) %address.local447, i64 8192, !dbg !10
  %53 = load <4 x half>, ptr addrspace(3) %lds.const.gep1199, align 8, !dbg !10
  %lds.const.gep1200 = getelementptr i8, ptr addrspace(3) %address.local447, i64 12288, !dbg !10
  %54 = load <4 x half>, ptr addrspace(3) %lds.const.gep1200, align 8, !dbg !10
  %lds.const.gep1201 = getelementptr i8, ptr addrspace(3) %address.local447, i64 9216, !dbg !10
  %55 = load <4 x half>, ptr addrspace(3) %lds.const.gep1201, align 8, !dbg !10
  %56 = fmul <4 x half> %53, splat (half 0xH3015), !dbg !10
  %lds.const.gep1202 = getelementptr i8, ptr addrspace(3) %address.local447, i64 13312, !dbg !10
  %57 = load <4 x half>, ptr addrspace(3) %lds.const.gep1202, align 8, !dbg !10
  %58 = fmul <4 x half> %54, splat (half 0xH3015), !dbg !10
  %59 = fmul <4 x half> %55, splat (half 0xH3015), !dbg !10
  %60 = fmul <4 x half> %57, splat (half 0xH3015), !dbg !10
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %61 = or disjoint i64 %25, 96, !dbg !8
  %62 = or disjoint i64 %address.local413, %61, !dbg !8
  %.tr1404 = trunc i64 %62 to i32, !dbg !8
  %dma.buffer.offset1378 = shl i32 %.tr1404, 1, !dbg !8
  %63 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1392, i32 %dma.buffer.offset1378, <4 x i32> %dma.buffer.rsrc1397) #10, !dbg !8
  %64 = or disjoint i64 %address.local425, %61, !dbg !8
  %.tr1405 = trunc i64 %64 to i32, !dbg !8
  %dma.buffer.offset1375 = shl i32 %.tr1405, 1, !dbg !8
  %65 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1389, i32 %dma.buffer.offset1375, <4 x i32> %dma.buffer.rsrc1397) #10, !dbg !8
  %66 = load <4 x half>, ptr addrspace(3) %address.local447, align 8, !dbg !10
  %67 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !10
  %68 = load <4 x half>, ptr addrspace(3) %lds.const.gep1197, align 8, !dbg !10
  %69 = fmul <4 x half> %66, splat (half 0xH3015), !dbg !10
  %70 = load <4 x half>, ptr addrspace(3) %lds.const.gep1198, align 8, !dbg !10
  %71 = fmul <4 x half> %67, splat (half 0xH3015), !dbg !10
  %72 = fmul <4 x half> %68, splat (half 0xH3015), !dbg !10
  %73 = fmul <4 x half> %70, splat (half 0xH3015), !dbg !10
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %74 = load <4 x half>, ptr addrspace(3) %lds.const.gep1199, align 8, !dbg !10
  %75 = load <4 x half>, ptr addrspace(3) %lds.const.gep1200, align 8, !dbg !10
  %76 = load <4 x half>, ptr addrspace(3) %lds.const.gep1201, align 8, !dbg !10
  %77 = fmul <4 x half> %74, splat (half 0xH3015), !dbg !10
  %78 = load <4 x half>, ptr addrspace(3) %lds.const.gep1202, align 8, !dbg !10
  %79 = fmul <4 x half> %75, splat (half 0xH3015), !dbg !10
  %80 = fmul <4 x half> %76, splat (half 0xH3015), !dbg !10
  %81 = fmul <4 x half> %78, splat (half 0xH3015), !dbg !10
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %82 = shl nuw nsw i32 %kdma.tid, 2, !dbg !14
  %83 = and i32 %82, 3084, !dbg !14
  %84 = and i32 %kdma.tid, 1, !dbg !14
  %85 = or disjoint i32 %83, %84, !dbg !14
  %narrow = add nuw nsw i32 %85, 1, !dbg !14
  %86 = lshr i32 %narrow, 1, !dbg !14
  %87 = zext nneg i32 %86 to i64, !dbg !14
  %k0.prime.address.thread498 = call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #9, !dbg !12, !range !32, !frisk.register_lifetimes !1
  %k0.prime.address.local499 = zext nneg i32 %k0.prime.address.thread498 to i64, !dbg !8
  %k0.prime.address.local500 = shl nuw nsw i64 %k0.prime.address.local499, 3, !dbg !8
  %k0.prime.address.local501 = and i64 %k0.prime.address.local500, 1536, !dbg !8
  %k0.prime.address.local502 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %k0.prime.address.local501, !dbg !8
  %k0.prime.address.local503 = ptrtoint ptr addrspace(3) %k0.prime.address.local502 to i64, !dbg !8
  %k0.prime.address.local504 = trunc i64 %k0.prime.address.local503 to i32, !dbg !8
  %k0.prime. = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %k0.prime.address.local504), !dbg !12
  %k0.prime.dma.buffer.lds1374 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %k0.prime.), !dbg !12
  %k0.prime.1963 = shl i64 %13, 19, !dbg !12
  %k0.prime.1964 = and i64 %k0.prime.1963, 16252928, !dbg !12
  %k0.prime.1965 = shl nuw nsw i64 0, 14, !dbg !12
  %k0.prime.1966 = or disjoint i64 %k0.prime.1964, %k0.prime.1965, !dbg !12
  %k0.prime.kdma.source = zext nneg i32 %kdma.owner to i64, !dbg !12
  %k0.prime.kdma.source1506 = lshr i64 %k0.prime.kdma.source, 4, !dbg !12
  %k0.prime.kdma.source1507 = and i64 %k0.prime.kdma.source1506, 8, !dbg !12
  %k0.prime.kdma.source1508 = and i64 %k0.prime.kdma.source, 91, !dbg !12
  %k0.prime.kdma.source1509 = lshr i64 %k0.prime.kdma.source, 2, !dbg !12
  %k0.prime.kdma.source1510 = and i64 %k0.prime.kdma.source1509, 4, !dbg !12
  %k0.prime.kdma.source1511 = add nuw nsw i64 %k0.prime.kdma.source1510, %k0.prime.kdma.source, !dbg !12
  %k0.prime.kdma.source1512 = and i64 %k0.prime.kdma.source1511, 4, !dbg !12
  %k0.prime.kdma.source1513 = or disjoint i64 %k0.prime.kdma.source1508, %k0.prime.kdma.source1512, !dbg !12
  %k0.prime.kdma.source1514 = shl nuw nsw i64 %k0.prime.kdma.source1513, 7, !dbg !12
  %k0.prime.kdma.source1515 = or disjoint i64 %k0.prime.kdma.source1507, %k0.prime.kdma.source1514, !dbg !12
  %k0.prime.kdma.source1516 = or disjoint i64 %k0.prime.1966, %k0.prime.kdma.source1515, !dbg !12
  %k0.prime.kdma.source1517 = trunc nuw i64 %k0.prime.kdma.source1516 to i32, !dbg !12
  %k0.prime.kdma.source1518 = shl nuw nsw i32 %k0.prime.kdma.source1517, 1, !dbg !12
  %k0.prime.kdma.offset = or disjoint i32 %k0.prime.kdma.source1518, %kdma.intra, !dbg !12
  %k0.prime.1967 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !12
  %k0.prime.1968 = insertelement <2 x i64> poison, i64 %k0.prime.1967, i64 0, !dbg !12
  %k0.prime.1969 = bitcast <2 x i64> %k0.prime.1968 to <4 x i32>, !dbg !12
  %k0.prime.dma.buffer.rsrc1373 = shufflevector <4 x i32> %k0.prime.1969, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !12
  %k0.prime.1970 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.dma.buffer.lds1374, i32 %k0.prime.kdma.offset, <4 x i32> %k0.prime.dma.buffer.rsrc1373) #10, !dbg !12, !frisk.cross_bn_k0 !1
  %k0.prime.kdma.lds1534 = add i32 %k0.prime.dma.buffer.lds1374, 512, !dbg !12
  %k0.prime.kdma.source1519 = zext nneg i32 %kdma.owner.hi to i64, !dbg !12
  %k0.prime.kdma.source1520 = lshr i64 %k0.prime.kdma.source1519, 4, !dbg !12
  %k0.prime.kdma.source1521 = and i64 %k0.prime.kdma.source1520, 8, !dbg !12
  %k0.prime.kdma.source1522 = and i64 %k0.prime.kdma.source1519, 123, !dbg !12
  %k0.prime.kdma.source1523 = lshr i64 %k0.prime.kdma.source1519, 2, !dbg !12
  %k0.prime.kdma.source1524 = and i64 %k0.prime.kdma.source1523, 4, !dbg !12
  %k0.prime.kdma.source1525 = add nuw nsw i64 %k0.prime.kdma.source1524, %k0.prime.kdma.source1519, !dbg !12
  %k0.prime.kdma.source1526 = and i64 %k0.prime.kdma.source1525, 4, !dbg !12
  %k0.prime.kdma.source1527 = or disjoint i64 %k0.prime.kdma.source1522, %k0.prime.kdma.source1526, !dbg !12
  %k0.prime.kdma.source1528 = shl nuw nsw i64 %k0.prime.kdma.source1527, 7, !dbg !12
  %k0.prime.kdma.source1529 = or disjoint i64 %k0.prime.kdma.source1521, %k0.prime.kdma.source1528, !dbg !12
  %k0.prime.kdma.source1530 = or disjoint i64 %k0.prime.1966, %k0.prime.kdma.source1529, !dbg !12
  %k0.prime.kdma.source1531 = trunc nuw i64 %k0.prime.kdma.source1530 to i32, !dbg !12
  %k0.prime.kdma.source1532 = shl nuw nsw i32 %k0.prime.kdma.source1531, 1, !dbg !12
  %k0.prime.kdma.offset1533 = or disjoint i32 %k0.prime.kdma.source1532, %kdma.intra.hi, !dbg !12
  %k0.prime.1971 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.kdma.lds1534, i32 %k0.prime.kdma.offset1533, <4 x i32> %k0.prime.dma.buffer.rsrc1373) #10, !dbg !12, !frisk.cross_bn_k0 !1
  %k0.prime.address.local516 = or disjoint i64 %k0.prime.address.local499, 256, !dbg !8
  %k0.prime.address.local517 = shl nuw nsw i64 %k0.prime.address.local516, 3, !dbg !8
  %k0.prime.address.local518 = and i64 %k0.prime.address.local517, 3584, !dbg !8
  %k0.prime.address.local519 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %k0.prime.address.local518, !dbg !8
  %k0.prime.address.local520 = ptrtoint ptr addrspace(3) %k0.prime.address.local519 to i64, !dbg !8
  %k0.prime.address.local521 = trunc i64 %k0.prime.address.local520 to i32, !dbg !8
  %k0.prime.1972 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %k0.prime.address.local521), !dbg !12
  %k0.prime.dma.buffer.lds1371 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %k0.prime.1972), !dbg !12
  %k0.prime.kdma.source1546 = or disjoint i64 %k0.prime.kdma.source1515, 16, !dbg !12
  %k0.prime.kdma.source1547 = or disjoint i64 %k0.prime.1966, %k0.prime.kdma.source1546, !dbg !12
  %k0.prime.kdma.source1548 = trunc nuw i64 %k0.prime.kdma.source1547 to i32, !dbg !12
  %k0.prime.kdma.source1549 = shl nuw nsw i32 %k0.prime.kdma.source1548, 1, !dbg !12
  %k0.prime.kdma.offset1550 = or disjoint i32 %k0.prime.kdma.source1549, %kdma.intra, !dbg !12
  %k0.prime.1973 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.dma.buffer.lds1371, i32 %k0.prime.kdma.offset1550, <4 x i32> %k0.prime.dma.buffer.rsrc1373) #10, !dbg !12, !frisk.cross_bn_k0 !1
  %k0.prime.kdma.lds1568 = add i32 %k0.prime.dma.buffer.lds1371, 512, !dbg !12
  %k0.prime.kdma.source1563 = or disjoint i64 %k0.prime.kdma.source1529, 16, !dbg !12
  %k0.prime.kdma.source1564 = or disjoint i64 %k0.prime.1966, %k0.prime.kdma.source1563, !dbg !12
  %k0.prime.kdma.source1565 = trunc nuw i64 %k0.prime.kdma.source1564 to i32, !dbg !12
  %k0.prime.kdma.source1566 = shl nuw nsw i32 %k0.prime.kdma.source1565, 1, !dbg !12
  %k0.prime.kdma.offset1567 = or disjoint i32 %k0.prime.kdma.source1566, %kdma.intra.hi, !dbg !12
  %k0.prime.1974 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.kdma.lds1568, i32 %k0.prime.kdma.offset1567, <4 x i32> %k0.prime.dma.buffer.rsrc1373) #10, !dbg !12, !frisk.cross_bn_k0 !1
  br label %88, !dbg !7

88:                                               ; preds = %k0.join, %4
  %89 = phi i64 [ %926, %k0.join ], [ 0, %4 ], !dbg !7
  %90 = phi [2 x <8 x float>] [ %1004, %k0.join ], [ zeroinitializer, %4 ], !dbg !7
  %91 = phi [2 x <1 x float>] [ %769, %k0.join ], [ zeroinitializer, %4 ], !dbg !7
  %92 = phi [2 x <1 x float>] [ %1006, %k0.join ], [ [<1 x float> splat (float 0xFFF0000000000000), <1 x float> splat (float 0xFFF0000000000000)], %4 ], !dbg !7
  %93 = phi [2 x <8 x float>] [ %1010, %k0.join ], [ zeroinitializer, %4 ], !dbg !7
  %94 = phi [2 x <8 x float>] [ %1014, %k0.join ], [ zeroinitializer, %4 ], !dbg !7
  %95 = phi [2 x <8 x float>] [ %1018, %k0.join ], [ zeroinitializer, %4 ], !dbg !7
  %96 = icmp samesign ult i64 %89, 32, !dbg !7
  br i1 %96, label %97, label %1019, !dbg !7

97:                                               ; preds = %88
  %98 = shl i64 %13, 19, !dbg !12
  %99 = and i64 %98, 16252928, !dbg !12
  %100 = shl nuw nsw i64 %89, 14, !dbg !12
  %101 = or disjoint i64 %99, %100, !dbg !12
  %address.thread498 = call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #9, !dbg !12, !range !32, !frisk.register_lifetimes !1
  %address.local499 = zext nneg i32 %address.thread498 to i64, !dbg !8
  %address.local500 = shl nuw nsw i64 %address.local499, 3, !dbg !8
  %address.local501 = and i64 %address.local500, 1536, !dbg !8
  %address.local502 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local501, !dbg !8
  %address.local503 = ptrtoint ptr addrspace(3) %address.local502 to i64, !dbg !8
  %address.local504 = trunc i64 %address.local503 to i32, !dbg !8
  %102 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local504), !dbg !12
  %103 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !12
  %104 = insertelement <2 x i64> poison, i64 %103, i64 0, !dbg !12
  %105 = bitcast <2 x i64> %104 to <4 x i32>, !dbg !12
  %dma.buffer.rsrc1373 = shufflevector <4 x i32> %105, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !12
  %dma.buffer.lds1374 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %102), !dbg !12
  %kdma.source = zext nneg i32 %kdma.owner to i64, !dbg !12
  %kdma.source1506 = lshr i64 %kdma.source, 4, !dbg !12
  %kdma.source1507 = and i64 %kdma.source1506, 8, !dbg !12
  %kdma.source1508 = and i64 %kdma.source, 91, !dbg !12
  %kdma.source1509 = lshr i64 %kdma.source, 2, !dbg !12
  %kdma.source1510 = and i64 %kdma.source1509, 4, !dbg !12
  %kdma.source1511 = add nuw nsw i64 %kdma.source1510, %kdma.source, !dbg !12
  %kdma.source1512 = and i64 %kdma.source1511, 4, !dbg !12
  %kdma.source1513 = or disjoint i64 %kdma.source1508, %kdma.source1512, !dbg !12
  %kdma.source1514 = shl nuw nsw i64 %kdma.source1513, 7, !dbg !12
  %kdma.source1515 = or disjoint i64 %kdma.source1507, %kdma.source1514, !dbg !12
  %kdma.source1516 = or disjoint i64 %101, %kdma.source1515, !dbg !12
  %kdma.source1517 = trunc nuw i64 %kdma.source1516 to i32, !dbg !12
  %kdma.source1518 = shl nuw nsw i32 %kdma.source1517, 1, !dbg !12
  %kdma.offset = or disjoint i32 %kdma.source1518, %kdma.intra, !dbg !12
  %kdma.source1519 = zext nneg i32 %kdma.owner.hi to i64, !dbg !12
  %kdma.source1520 = lshr i64 %kdma.source1519, 4, !dbg !12
  %kdma.source1521 = and i64 %kdma.source1520, 8, !dbg !12
  %kdma.source1522 = and i64 %kdma.source1519, 123, !dbg !12
  %kdma.source1523 = lshr i64 %kdma.source1519, 2, !dbg !12
  %kdma.source1524 = and i64 %kdma.source1523, 4, !dbg !12
  %kdma.source1525 = add nuw nsw i64 %kdma.source1524, %kdma.source1519, !dbg !12
  %kdma.source1526 = and i64 %kdma.source1525, 4, !dbg !12
  %kdma.source1527 = or disjoint i64 %kdma.source1522, %kdma.source1526, !dbg !12
  %kdma.source1528 = shl nuw nsw i64 %kdma.source1527, 7, !dbg !12
  %kdma.source1529 = or disjoint i64 %kdma.source1521, %kdma.source1528, !dbg !12
  %kdma.source1530 = or disjoint i64 %101, %kdma.source1529, !dbg !12
  %kdma.source1531 = trunc nuw i64 %kdma.source1530 to i32, !dbg !12
  %kdma.source1532 = shl nuw nsw i32 %kdma.source1531, 1, !dbg !12
  %kdma.offset1533 = or disjoint i32 %kdma.source1532, %kdma.intra.hi, !dbg !12
  %kdma.lds1534 = add i32 %dma.buffer.lds1374, 512, !dbg !12
  %address.local516 = or disjoint i64 %address.local499, 256, !dbg !8
  %address.local517 = shl nuw nsw i64 %address.local516, 3, !dbg !8
  %address.local518 = and i64 %address.local517, 3584, !dbg !8
  %address.local519 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local518, !dbg !8
  %address.local520 = ptrtoint ptr addrspace(3) %address.local519 to i64, !dbg !8
  %address.local521 = trunc i64 %address.local520 to i32, !dbg !8
  %106 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local521), !dbg !12
  %dma.buffer.lds1371 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %106), !dbg !12
  %kdma.source1546 = or disjoint i64 %kdma.source1515, 16, !dbg !12
  %kdma.source1547 = or disjoint i64 %101, %kdma.source1546, !dbg !12
  %kdma.source1548 = trunc nuw i64 %kdma.source1547 to i32, !dbg !12
  %kdma.source1549 = shl nuw nsw i32 %kdma.source1548, 1, !dbg !12
  %kdma.offset1550 = or disjoint i32 %kdma.source1549, %kdma.intra, !dbg !12
  %kdma.source1563 = or disjoint i64 %kdma.source1529, 16, !dbg !12
  %kdma.source1564 = or disjoint i64 %101, %kdma.source1563, !dbg !12
  %kdma.source1565 = trunc nuw i64 %kdma.source1564 to i32, !dbg !12
  %kdma.source1566 = shl nuw nsw i32 %kdma.source1565, 1, !dbg !12
  %kdma.offset1567 = or disjoint i32 %kdma.source1566, %kdma.intra.hi, !dbg !12
  %kdma.lds1568 = add i32 %dma.buffer.lds1371, 512, !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %107 = or disjoint i64 %101, 32, !dbg !12
  %address.local533 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local501, !dbg !8
  %address.local534 = ptrtoint ptr addrspace(3) %address.local533 to i64, !dbg !8
  %address.local535 = trunc i64 %address.local534 to i32, !dbg !8
  %108 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local535), !dbg !12
  %dma.buffer.lds1368 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %108), !dbg !12
  %kdma.source1580 = or disjoint i64 %107, %kdma.source1515, !dbg !12
  %kdma.source1581 = trunc nuw i64 %kdma.source1580 to i32, !dbg !12
  %kdma.source1582 = shl nuw nsw i32 %kdma.source1581, 1, !dbg !12
  %kdma.offset1583 = or disjoint i32 %kdma.source1582, %kdma.intra, !dbg !12
  %109 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1368, i32 %kdma.offset1583, <4 x i32> %dma.buffer.rsrc1373) #10, !dbg !12
  %kdma.source1596 = or disjoint i64 %107, %kdma.source1529, !dbg !12
  %kdma.source1597 = trunc nuw i64 %kdma.source1596 to i32, !dbg !12
  %kdma.source1598 = shl nuw nsw i32 %kdma.source1597, 1, !dbg !12
  %kdma.offset1599 = or disjoint i32 %kdma.source1598, %kdma.intra.hi, !dbg !12
  %kdma.lds1600 = add i32 %dma.buffer.lds1368, 512, !dbg !12
  %110 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1600, i32 %kdma.offset1599, <4 x i32> %dma.buffer.rsrc1373) #10, !dbg !12
  %address.local537 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local518, !dbg !8
  %address.local538 = ptrtoint ptr addrspace(3) %address.local537 to i64, !dbg !8
  %address.local539 = trunc i64 %address.local538 to i32, !dbg !8
  %111 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local539), !dbg !12
  %dma.buffer.lds1365 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %111), !dbg !12
  %kdma.source1613 = or disjoint i64 %107, %kdma.source1546, !dbg !12
  %kdma.source1614 = trunc nuw i64 %kdma.source1613 to i32, !dbg !12
  %kdma.source1615 = shl nuw nsw i32 %kdma.source1614, 1, !dbg !12
  %kdma.offset1616 = or disjoint i32 %kdma.source1615, %kdma.intra, !dbg !12
  %112 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1365, i32 %kdma.offset1616, <4 x i32> %dma.buffer.rsrc1373) #10, !dbg !12
  %kdma.source1630 = or disjoint i64 %107, %kdma.source1563, !dbg !12
  %kdma.source1631 = trunc nuw i64 %kdma.source1630 to i32, !dbg !12
  %kdma.source1632 = shl nuw nsw i32 %kdma.source1631, 1, !dbg !12
  %kdma.offset1633 = or disjoint i32 %kdma.source1632, %kdma.intra.hi, !dbg !12
  %kdma.lds1634 = add i32 %dma.buffer.lds1365, 512, !dbg !12
  %113 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1634, i32 %kdma.offset1633, <4 x i32> %dma.buffer.rsrc1373) #10, !dbg !12
  %address.local542 = lshr i64 %address.local499, 2, !dbg !14
  %address.local545 = shl nuw nsw i64 %address.local499, 1, !dbg !14
  %address.local546 = and i64 %address.local545, 24, !dbg !14
  %address.local548 = shl i64 %address.local499, 62, !dbg !14
  %address.local549 = ashr exact i64 %address.local548, 57, !dbg !14
  %address.local550 = and i64 %address.local549, -64, !dbg !14
  %address.local553 = and i64 %address.local542, 12, !dbg !10
  %address.local555 = and i64 %address.local499, 63, !dbg !10
  %address.local556 = lshr i64 %address.local555, 5, !dbg !10
  %address.local557 = mul nuw nsw i64 %address.local556, 1016, !dbg !10
  %.idx = mul nuw nsw i64 %address.local499, 320, !dbg !14
  %114 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %.idx, !dbg !14
  %.idx1410 = mul nsw i64 %address.local542, -1280, !dbg !14
  %115 = getelementptr i8, ptr addrspace(3) %114, i64 %.idx1410, !dbg !14
  %116 = getelementptr [2 x i8], ptr addrspace(3) %115, i64 %address.local546, !dbg !14
  %117 = getelementptr [2 x i8], ptr addrspace(3) %116, i64 %address.local550, !dbg !14
  %118 = getelementptr [2 x i8], ptr addrspace(3) %117, i64 %address.local553, !dbg !14
  %address.local559 = getelementptr [2 x i8], ptr addrspace(3) %118, i64 %address.local557, !dbg !14
  %119 = add nsw i64 %.idx, %.idx1410, !dbg !14
  %.idx1893 = shl nuw nsw i64 %address.local546, 1, !dbg !14
  %120 = or disjoint i64 %119, %.idx1893, !dbg !14
  %121 = or disjoint i64 %address.local550, %address.local553, !dbg !14
  %122 = shl nsw i64 %121, 1, !dbg !14
  %123 = add nsw i64 %122, %120, !dbg !14
  %address.local559.idx = mul nuw nsw i64 %address.local556, 2032, !dbg !14
  %124 = add nsw i64 %123, %address.local559.idx, !dbg !14
  %125 = lshr i64 %124, 6, !dbg !14
  %126 = and i64 %125, 8, !dbg !14
  %127 = xor i64 %124, %126, !dbg !14
  %128 = sub i64 %127, %124, !dbg !14
  %kdma.swizzled = getelementptr i8, ptr addrspace(3) %address.local559, i64 %128, !dbg !14
  %129 = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8, !dbg !14
  %130 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %42, <4 x half> %129, <4 x float> zeroinitializer) #9, !dbg !14
  %kdma.gep = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4096, !dbg !14
  %131 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8, !dbg !14
  %address.local570 = shl nuw nsw i32 %address.thread498, 1, !dbg !14
  %address.local571 = and i32 %address.local570, 24, !dbg !14
  %address.local572 = shl nuw nsw i32 %address.thread498, 8, !dbg !14
  %address.local573 = and i32 %address.local572, 768, !dbg !14
  %address.local574 = or disjoint i32 %address.local571, %address.local573, !dbg !14
  %address.local575 = shl nuw nsw i32 %address.thread498, 5, !dbg !14
  %address.local576 = and i32 %address.local575, 96, !dbg !14
  %address.local577 = or disjoint i32 %address.local574, %address.local576, !dbg !14
  %address.local578 = zext nneg i32 %address.local577 to i64, !dbg !14
  %address.base = call i64 asm sideeffect "", "=v,0"(i64 %87) #9, !dbg !14, !frisk.register_lifetimes !1
  %address.local579 = lshr i32 %address.thread498, 1, !dbg !14
  %address.local580 = and i32 %address.local579, 1, !dbg !14
  %address.local581 = zext nneg i32 %address.local580 to i64, !dbg !14
  %address.local582 = add nsw i64 %address.base, %address.local581, !dbg !14
  %address.local583 = shl i64 %address.local582, 6, !dbg !14
  %address.local584 = sub i64 %address.local578, %address.local583, !dbg !14
  %132 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local584, !dbg !14
  %133 = getelementptr [2 x i8], ptr addrspace(3) %132, i64 %address.local553, !dbg !14
  %134 = getelementptr [2 x i8], ptr addrspace(3) %133, i64 %address.local557, !dbg !14
  %address.local588 = getelementptr i8, ptr addrspace(3) %134, i64 64, !dbg !14
  %.idx18961899 = add i64 %address.local584, %address.local553, !dbg !14
  %135 = shl i64 %.idx18961899, 1, !dbg !14
  %136 = add i64 %135, %address.local559.idx, !dbg !14
  %137 = add i64 %136, 64, !dbg !14
  %138 = lshr i64 %137, 6, !dbg !14
  %139 = and i64 %138, 8, !dbg !14
  %140 = xor i64 %137, %139, !dbg !14
  %141 = sub i64 %140, %137, !dbg !14
  %kdma.swizzled1767 = getelementptr i8, ptr addrspace(3) %address.local588, i64 %141, !dbg !14
  %142 = load <4 x half>, ptr addrspace(3) %kdma.swizzled1767, align 8, !dbg !14, !frisk.k_lds_pair !35
  %kdma.gep1771 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 128, !dbg !14
  %143 = load <4 x half>, ptr addrspace(3) %kdma.gep1771, align 8, !dbg !14, !frisk.k_lds_pair !35
  %kdma.gep1768 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 4096, !dbg !14
  %144 = load <4 x half>, ptr addrspace(3) %kdma.gep1768, align 8, !dbg !14, !frisk.k_lds_pair !36
  %kdma.gep1772 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 4224, !dbg !14
  %145 = load <4 x half>, ptr addrspace(3) %kdma.gep1772, align 8, !dbg !14, !frisk.k_lds_pair !36
  %kdma.gep1769 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 128, !dbg !14
  %146 = load <4 x half>, ptr addrspace(3) %kdma.gep1769, align 8, !dbg !14, !frisk.k_lds_pair !37
  %kdma.gep1773 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1024, !dbg !14
  %147 = load <4 x half>, ptr addrspace(3) %kdma.gep1773, align 8, !dbg !14, !frisk.k_lds_pair !37
  %kdma.gep1770 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4224, !dbg !14
  %148 = load <4 x half>, ptr addrspace(3) %kdma.gep1770, align 8, !dbg !14, !frisk.k_lds_pair !38
  %kdma.gep1774 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5120, !dbg !14
  %149 = load <4 x half>, ptr addrspace(3) %kdma.gep1774, align 8, !dbg !14, !frisk.k_lds_pair !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.k_lds_prefetch !1
  %150 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %44, <4 x half> %131, <4 x float> %130) #9, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.k_lds_prefetch !1
  %kdma.gep1775 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 1024, !dbg !14
  %151 = load <4 x half>, ptr addrspace(3) %kdma.gep1775, align 8, !dbg !14, !frisk.k_lds_pair !33
  %kdma.gep1779 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 1152, !dbg !14
  %152 = load <4 x half>, ptr addrspace(3) %kdma.gep1779, align 8, !dbg !14, !frisk.k_lds_pair !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.k_lds_prefetch !1
  %153 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %42, <4 x half> %142, <4 x float> zeroinitializer) #9, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.k_lds_prefetch !1
  %kdma.gep1776 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 5120, !dbg !14
  %154 = load <4 x half>, ptr addrspace(3) %kdma.gep1776, align 8, !dbg !14, !frisk.k_lds_pair !39
  %kdma.gep1780 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 5248, !dbg !14
  %155 = load <4 x half>, ptr addrspace(3) %kdma.gep1780, align 8, !dbg !14, !frisk.k_lds_pair !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.k_lds_prefetch !1
  %156 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %44, <4 x half> %144, <4 x float> %153) #9, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.k_lds_prefetch !1
  %kdma.gep1777 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1152, !dbg !14
  %157 = load <4 x half>, ptr addrspace(3) %kdma.gep1777, align 8, !dbg !14, !frisk.k_lds_pair !40
  %kdma.gep1778 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5248, !dbg !14
  %158 = load <4 x half>, ptr addrspace(3) %kdma.gep1778, align 8, !dbg !14, !frisk.k_lds_pair !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.k_lds_prefetch !1
  %159 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %42, <4 x half> %146, <4 x float> zeroinitializer) #9, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.k_lds_prefetch !1
  %160 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %44, <4 x half> %148, <4 x float> %159) #9, !dbg !14
  %161 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %42, <4 x half> %143, <4 x float> zeroinitializer) #9, !dbg !14
  %162 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %44, <4 x half> %145, <4 x float> %161) #9, !dbg !14
  %163 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %42, <4 x half> %147, <4 x float> zeroinitializer) #9, !dbg !14
  %164 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %44, <4 x half> %149, <4 x float> %163) #9, !dbg !14
  %165 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %42, <4 x half> %151, <4 x float> zeroinitializer) #9, !dbg !14
  %166 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %44, <4 x half> %154, <4 x float> %165) #9, !dbg !14
  %167 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %42, <4 x half> %157, <4 x float> zeroinitializer) #9, !dbg !14
  %168 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %44, <4 x half> %158, <4 x float> %167) #9, !dbg !14
  %169 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %42, <4 x half> %152, <4 x float> zeroinitializer) #9, !dbg !14
  %170 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %44, <4 x half> %155, <4 x float> %169) #9, !dbg !14
  %171 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %46, <4 x half> %129, <4 x float> zeroinitializer) #9, !dbg !14
  %172 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %47, <4 x half> %131, <4 x float> %171) #9, !dbg !14
  %173 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %46, <4 x half> %142, <4 x float> zeroinitializer) #9, !dbg !14
  %174 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %47, <4 x half> %144, <4 x float> %173) #9, !dbg !14
  %175 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %46, <4 x half> %146, <4 x float> zeroinitializer) #9, !dbg !14
  %176 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %47, <4 x half> %148, <4 x float> %175) #9, !dbg !14
  %177 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %46, <4 x half> %143, <4 x float> zeroinitializer) #9, !dbg !14
  %178 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %47, <4 x half> %145, <4 x float> %177) #9, !dbg !14
  %179 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %46, <4 x half> %147, <4 x float> zeroinitializer) #9, !dbg !14
  %180 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %47, <4 x half> %149, <4 x float> %179) #9, !dbg !14
  %181 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %46, <4 x half> %151, <4 x float> zeroinitializer) #9, !dbg !14
  %182 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %47, <4 x half> %154, <4 x float> %181) #9, !dbg !14
  %183 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %46, <4 x half> %157, <4 x float> zeroinitializer) #9, !dbg !14
  %184 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %47, <4 x half> %158, <4 x float> %183) #9, !dbg !14
  %185 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %46, <4 x half> %152, <4 x float> zeroinitializer) #9, !dbg !14
  %186 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %47, <4 x half> %155, <4 x float> %185) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %187 = or disjoint i64 %101, 64, !dbg !12
  %kdma.source1646 = or disjoint i64 %187, %kdma.source1515, !dbg !12
  %kdma.source1647 = trunc nuw i64 %kdma.source1646 to i32, !dbg !12
  %kdma.source1648 = shl nuw nsw i32 %kdma.source1647, 1, !dbg !12
  %kdma.offset1649 = or disjoint i32 %kdma.source1648, %kdma.intra, !dbg !12
  %188 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1374, i32 %kdma.offset1649, <4 x i32> %dma.buffer.rsrc1373) #10, !dbg !12
  %kdma.source1662 = or disjoint i64 %187, %kdma.source1529, !dbg !12
  %kdma.source1663 = trunc nuw i64 %kdma.source1662 to i32, !dbg !12
  %kdma.source1664 = shl nuw nsw i32 %kdma.source1663, 1, !dbg !12
  %kdma.offset1665 = or disjoint i32 %kdma.source1664, %kdma.intra.hi, !dbg !12
  %189 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1534, i32 %kdma.offset1665, <4 x i32> %dma.buffer.rsrc1373) #10, !dbg !12
  %kdma.source1679 = or disjoint i64 %187, %kdma.source1546, !dbg !12
  %kdma.source1680 = trunc nuw i64 %kdma.source1679 to i32, !dbg !12
  %kdma.source1681 = shl nuw nsw i32 %kdma.source1680, 1, !dbg !12
  %kdma.offset1682 = or disjoint i32 %kdma.source1681, %kdma.intra, !dbg !12
  %190 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1371, i32 %kdma.offset1682, <4 x i32> %dma.buffer.rsrc1373) #10, !dbg !12
  %kdma.source1696 = or disjoint i64 %187, %kdma.source1563, !dbg !12
  %kdma.source1697 = trunc nuw i64 %kdma.source1696 to i32, !dbg !12
  %kdma.source1698 = shl nuw nsw i32 %kdma.source1697, 1, !dbg !12
  %kdma.offset1699 = or disjoint i32 %kdma.source1698, %kdma.intra.hi, !dbg !12
  %191 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1568, i32 %kdma.offset1699, <4 x i32> %dma.buffer.rsrc1373) #10, !dbg !12
  %kdma.gep1797 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8192, !dbg !14
  %192 = load <4 x half>, ptr addrspace(3) %kdma.gep1797, align 8, !dbg !14, !frisk.k_lds_pair !41
  %kdma.gep1801 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8320, !dbg !14
  %193 = load <4 x half>, ptr addrspace(3) %kdma.gep1801, align 8, !dbg !14, !frisk.k_lds_pair !41
  %kdma.gep1798 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12288, !dbg !14
  %194 = load <4 x half>, ptr addrspace(3) %kdma.gep1798, align 8, !dbg !14, !frisk.k_lds_pair !42
  %kdma.gep1802 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12416, !dbg !14
  %195 = load <4 x half>, ptr addrspace(3) %kdma.gep1802, align 8, !dbg !14, !frisk.k_lds_pair !42
  %kdma.gep1799 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 8192, !dbg !14
  %196 = load <4 x half>, ptr addrspace(3) %kdma.gep1799, align 8, !dbg !14, !frisk.k_lds_pair !43
  %kdma.gep1803 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 8320, !dbg !14
  %197 = load <4 x half>, ptr addrspace(3) %kdma.gep1803, align 8, !dbg !14, !frisk.k_lds_pair !43
  %kdma.gep1800 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 12288, !dbg !14
  %198 = load <4 x half>, ptr addrspace(3) %kdma.gep1800, align 8, !dbg !14, !frisk.k_lds_pair !44
  %kdma.gep1804 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 12416, !dbg !14
  %199 = load <4 x half>, ptr addrspace(3) %kdma.gep1804, align 8, !dbg !14, !frisk.k_lds_pair !44
  %kdma.gep1805 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9216, !dbg !14
  %200 = load <4 x half>, ptr addrspace(3) %kdma.gep1805, align 8, !dbg !14, !frisk.k_lds_pair !45
  %kdma.gep1809 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9344, !dbg !14
  %201 = load <4 x half>, ptr addrspace(3) %kdma.gep1809, align 8, !dbg !14, !frisk.k_lds_pair !45
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.k_lds_prefetch !1
  %202 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %56, <4 x half> %192, <4 x float> %150) #9, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.k_lds_prefetch !1
  %kdma.gep1806 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13312, !dbg !14
  %203 = load <4 x half>, ptr addrspace(3) %kdma.gep1806, align 8, !dbg !14, !frisk.k_lds_pair !46
  %kdma.gep1810 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13440, !dbg !14
  %204 = load <4 x half>, ptr addrspace(3) %kdma.gep1810, align 8, !dbg !14, !frisk.k_lds_pair !46
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.k_lds_prefetch !1
  %205 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %58, <4 x half> %194, <4 x float> %202) #9, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.k_lds_prefetch !1
  %kdma.gep1807 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 9216, !dbg !14
  %206 = load <4 x half>, ptr addrspace(3) %kdma.gep1807, align 8, !dbg !14, !frisk.k_lds_pair !47
  %kdma.gep1811 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 9344, !dbg !14
  %207 = load <4 x half>, ptr addrspace(3) %kdma.gep1811, align 8, !dbg !14, !frisk.k_lds_pair !47
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.k_lds_prefetch !1
  %208 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %56, <4 x half> %196, <4 x float> %156) #9, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.k_lds_prefetch !1
  %kdma.gep1808 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 13312, !dbg !14
  %209 = load <4 x half>, ptr addrspace(3) %kdma.gep1808, align 8, !dbg !14, !frisk.k_lds_pair !48
  %kdma.gep1812 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 13440, !dbg !14
  %210 = load <4 x half>, ptr addrspace(3) %kdma.gep1812, align 8, !dbg !14, !frisk.k_lds_pair !48
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.k_lds_prefetch !1
  %211 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %58, <4 x half> %198, <4 x float> %208) #9, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.k_lds_prefetch !1
  %212 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %56, <4 x half> %193, <4 x float> %160) #9, !dbg !14
  %213 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %58, <4 x half> %195, <4 x float> %212) #9, !dbg !14
  %214 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %56, <4 x half> %197, <4 x float> %162) #9, !dbg !14
  %215 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %58, <4 x half> %199, <4 x float> %214) #9, !dbg !14
  %216 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %56, <4 x half> %200, <4 x float> %164) #9, !dbg !14
  %217 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %58, <4 x half> %203, <4 x float> %216) #9, !dbg !14
  %218 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %56, <4 x half> %206, <4 x float> %166) #9, !dbg !14
  %219 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %58, <4 x half> %209, <4 x float> %218) #9, !dbg !14
  %220 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %56, <4 x half> %201, <4 x float> %168) #9, !dbg !14
  %221 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %58, <4 x half> %204, <4 x float> %220) #9, !dbg !14
  %222 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %56, <4 x half> %207, <4 x float> %170) #9, !dbg !14
  %223 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %58, <4 x half> %210, <4 x float> %222) #9, !dbg !14
  %224 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %192, <4 x float> %172) #9, !dbg !14
  %225 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %194, <4 x float> %224) #9, !dbg !14
  %226 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %196, <4 x float> %174) #9, !dbg !14
  %227 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %198, <4 x float> %226) #9, !dbg !14
  %228 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %193, <4 x float> %176) #9, !dbg !14
  %229 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %195, <4 x float> %228) #9, !dbg !14
  %230 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %197, <4 x float> %178) #9, !dbg !14
  %231 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %199, <4 x float> %230) #9, !dbg !14
  %232 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %200, <4 x float> %180) #9, !dbg !14
  %233 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %203, <4 x float> %232) #9, !dbg !14
  %234 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %206, <4 x float> %182) #9, !dbg !14
  %235 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %209, <4 x float> %234) #9, !dbg !14
  %236 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %201, <4 x float> %184) #9, !dbg !14
  %237 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %204, <4 x float> %236) #9, !dbg !14
  %238 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %207, <4 x float> %186) #9, !dbg !14
  %239 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %210, <4 x float> %238) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %240 = or disjoint i64 %101, 96, !dbg !12
  %kdma.source1712 = or disjoint i64 %240, %kdma.source1515, !dbg !12
  %kdma.source1713 = trunc nuw i64 %kdma.source1712 to i32, !dbg !12
  %kdma.source1714 = shl nuw nsw i32 %kdma.source1713, 1, !dbg !12
  %kdma.offset1715 = or disjoint i32 %kdma.source1714, %kdma.intra, !dbg !12
  %241 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1368, i32 %kdma.offset1715, <4 x i32> %dma.buffer.rsrc1373) #10, !dbg !12
  %kdma.source1728 = or disjoint i64 %240, %kdma.source1529, !dbg !12
  %kdma.source1729 = trunc nuw i64 %kdma.source1728 to i32, !dbg !12
  %kdma.source1730 = shl nuw nsw i32 %kdma.source1729, 1, !dbg !12
  %kdma.offset1731 = or disjoint i32 %kdma.source1730, %kdma.intra.hi, !dbg !12
  %242 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1600, i32 %kdma.offset1731, <4 x i32> %dma.buffer.rsrc1373) #10, !dbg !12
  %kdma.source1745 = or disjoint i64 %240, %kdma.source1546, !dbg !12
  %kdma.source1746 = trunc nuw i64 %kdma.source1745 to i32, !dbg !12
  %kdma.source1747 = shl nuw nsw i32 %kdma.source1746, 1, !dbg !12
  %kdma.offset1748 = or disjoint i32 %kdma.source1747, %kdma.intra, !dbg !12
  %243 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1365, i32 %kdma.offset1748, <4 x i32> %dma.buffer.rsrc1373) #10, !dbg !12
  %kdma.source1762 = or disjoint i64 %240, %kdma.source1563, !dbg !12
  %kdma.source1763 = trunc nuw i64 %kdma.source1762 to i32, !dbg !12
  %kdma.source1764 = shl nuw nsw i32 %kdma.source1763, 1, !dbg !12
  %kdma.offset1765 = or disjoint i32 %kdma.source1764, %kdma.intra.hi, !dbg !12
  %244 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1634, i32 %kdma.offset1765, <4 x i32> %dma.buffer.rsrc1373) #10, !dbg !12
  %245 = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8, !dbg !14, !frisk.k_lds_pair !49
  %246 = load <4 x half>, ptr addrspace(3) %kdma.gep1769, align 8, !dbg !14, !frisk.k_lds_pair !49
  %247 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8, !dbg !14, !frisk.k_lds_pair !50
  %248 = load <4 x half>, ptr addrspace(3) %kdma.gep1770, align 8, !dbg !14, !frisk.k_lds_pair !50
  %249 = load <4 x half>, ptr addrspace(3) %kdma.swizzled1767, align 8, !dbg !14, !frisk.k_lds_pair !51
  %250 = load <4 x half>, ptr addrspace(3) %kdma.gep1771, align 8, !dbg !14, !frisk.k_lds_pair !51
  %251 = load <4 x half>, ptr addrspace(3) %kdma.gep1768, align 8, !dbg !14, !frisk.k_lds_pair !52
  %252 = load <4 x half>, ptr addrspace(3) %kdma.gep1772, align 8, !dbg !14, !frisk.k_lds_pair !52
  %253 = load <4 x half>, ptr addrspace(3) %kdma.gep1773, align 8, !dbg !14, !frisk.k_lds_pair !53
  %254 = load <4 x half>, ptr addrspace(3) %kdma.gep1777, align 8, !dbg !14, !frisk.k_lds_pair !53
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.k_lds_prefetch !1
  %255 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %245, <4 x float> %205) #9, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.k_lds_prefetch !1
  %256 = load <4 x half>, ptr addrspace(3) %kdma.gep1774, align 8, !dbg !14, !frisk.k_lds_pair !54
  %257 = load <4 x half>, ptr addrspace(3) %kdma.gep1778, align 8, !dbg !14, !frisk.k_lds_pair !54
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.k_lds_prefetch !1
  %258 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %247, <4 x float> %255) #9, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.k_lds_prefetch !1
  %259 = load <4 x half>, ptr addrspace(3) %kdma.gep1775, align 8, !dbg !14, !frisk.k_lds_pair !55
  %260 = load <4 x half>, ptr addrspace(3) %kdma.gep1779, align 8, !dbg !14, !frisk.k_lds_pair !55
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.k_lds_prefetch !1
  %261 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %249, <4 x float> %211) #9, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.k_lds_prefetch !1
  %262 = load <4 x half>, ptr addrspace(3) %kdma.gep1776, align 8, !dbg !14, !frisk.k_lds_pair !56
  %263 = load <4 x half>, ptr addrspace(3) %kdma.gep1780, align 8, !dbg !14, !frisk.k_lds_pair !56
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.k_lds_prefetch !1
  %264 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %251, <4 x float> %261) #9, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.k_lds_prefetch !1
  %265 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %246, <4 x float> %213) #9, !dbg !14
  %266 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %248, <4 x float> %265) #9, !dbg !14
  %267 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %250, <4 x float> %215) #9, !dbg !14
  %268 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %252, <4 x float> %267) #9, !dbg !14
  %269 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %253, <4 x float> %217) #9, !dbg !14
  %270 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %256, <4 x float> %269) #9, !dbg !14
  %271 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %259, <4 x float> %219) #9, !dbg !14
  %272 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %262, <4 x float> %271) #9, !dbg !14
  %273 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %254, <4 x float> %221) #9, !dbg !14
  %274 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %257, <4 x float> %273) #9, !dbg !14
  %275 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %260, <4 x float> %223) #9, !dbg !14
  %276 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %263, <4 x float> %275) #9, !dbg !14
  %277 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %245, <4 x float> %225) #9, !dbg !14
  %278 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %247, <4 x float> %277) #9, !dbg !14
  %279 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %249, <4 x float> %227) #9, !dbg !14
  %280 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %251, <4 x float> %279) #9, !dbg !14
  %281 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %246, <4 x float> %229) #9, !dbg !14
  %282 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %248, <4 x float> %281) #9, !dbg !14
  %283 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %250, <4 x float> %231) #9, !dbg !14
  %284 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %252, <4 x float> %283) #9, !dbg !14
  %285 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %253, <4 x float> %233) #9, !dbg !14
  %286 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %256, <4 x float> %285) #9, !dbg !14
  %287 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %259, <4 x float> %235) #9, !dbg !14
  %288 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %262, <4 x float> %287) #9, !dbg !14
  %289 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %254, <4 x float> %237) #9, !dbg !14
  %290 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %257, <4 x float> %289) #9, !dbg !14
  %291 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %260, <4 x float> %239) #9, !dbg !14
  %292 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %263, <4 x float> %291) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %address.local917 = shl nuw nsw i64 %address.local499, 5, !dbg !15
  %address.local918 = and i64 %address.local917, 8064, !dbg !15
  %address.local920 = and i64 %address.local500, 24, !dbg !15
  %address.local921 = or disjoint i64 %address.local918, %address.local920, !dbg !15
  %293 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %.idx14191420 = or disjoint i64 %101, %address.local921, !dbg !15
  %.idx14191420.tr = trunc nuw i64 %.idx14191420 to i32, !dbg !15
  %dma.buffer.offset1348 = shl nuw nsw i32 %.idx14191420.tr, 1, !dbg !15
  %294 = insertelement <2 x i64> poison, i64 %293, i64 0, !dbg !15
  %295 = bitcast <2 x i64> %294 to <4 x i32>, !dbg !15
  %dma.buffer.rsrc1349 = shufflevector <4 x i32> %295, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %296 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1374, i32 %dma.buffer.offset1348, <4 x i32> %dma.buffer.rsrc1349) #10, !dbg !15
  %address.local923 = shl nuw nsw i64 %address.local516, 5, !dbg !15
  %address.local924 = and i64 %address.local923, 16256, !dbg !15
  %address.local927 = or disjoint i64 %address.local924, %address.local920, !dbg !15
  %.idx14211422 = or disjoint i64 %101, %address.local927, !dbg !15
  %.idx14211422.tr = trunc nuw i64 %.idx14211422 to i32, !dbg !15
  %dma.buffer.offset1345 = shl nuw nsw i32 %.idx14211422.tr, 1, !dbg !15
  %297 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1371, i32 %dma.buffer.offset1345, <4 x i32> %dma.buffer.rsrc1349) #10, !dbg !15
  %298 = load <4 x half>, ptr addrspace(3) %kdma.gep1797, align 8, !dbg !14, !frisk.k_tail_pair !35
  %299 = load <4 x half>, ptr addrspace(3) %kdma.gep1801, align 8, !dbg !14, !frisk.k_tail_pair !35
  %300 = load <4 x half>, ptr addrspace(3) %kdma.gep1798, align 8, !dbg !14, !frisk.k_tail_pair !36
  %301 = load <4 x half>, ptr addrspace(3) %kdma.gep1802, align 8, !dbg !14, !frisk.k_tail_pair !36
  %302 = load <4 x half>, ptr addrspace(3) %kdma.gep1799, align 8, !dbg !14, !frisk.k_tail_pair !37
  %303 = load <4 x half>, ptr addrspace(3) %kdma.gep1803, align 8, !dbg !14, !frisk.k_tail_pair !37
  %304 = load <4 x half>, ptr addrspace(3) %kdma.gep1800, align 8, !dbg !14, !frisk.k_tail_pair !38
  %305 = load <4 x half>, ptr addrspace(3) %kdma.gep1804, align 8, !dbg !14, !frisk.k_tail_pair !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.k_tail_prefetch !1
  %306 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %298, <4 x float> %258) #9, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.k_tail_prefetch !1
  %307 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %300, <4 x float> %306) #9, !dbg !14
  %308 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %302, <4 x float> %264) #9, !dbg !14
  %309 = extractelement <4 x float> %307, i64 0, !dbg !19
  %310 = call float @llvm.maxnum.f32(float %309, float 0xFFF0000000000000), !dbg !20, !frisk.local_reduction !1
  %311 = extractelement <4 x float> %307, i64 1, !dbg !19
  %312 = extractelement <4 x float> %307, i64 2, !dbg !19
  %313 = extractelement <4 x float> %307, i64 3, !dbg !19
  %314 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %304, <4 x float> %308) #9, !dbg !14
  %315 = load <4 x half>, ptr addrspace(3) %kdma.gep1805, align 8, !dbg !14, !frisk.k_tail_pair !33
  %316 = load <4 x half>, ptr addrspace(3) %kdma.gep1809, align 8, !dbg !14, !frisk.k_tail_pair !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.k_tail_prefetch !1
  %317 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %299, <4 x float> %266) #9, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !19, !frisk.k_tail_prefetch !1
  %318 = extractelement <4 x float> %314, i64 0, !dbg !19
  %local.reduce151 = call float @llvm.maxnum.f32(float %310, float %318), !dbg !20, !frisk.local_reduction !1
  %319 = extractelement <4 x float> %314, i64 1, !dbg !19
  %local.reduce152 = call float @llvm.maxnum.f32(float %311, float %319), !dbg !20, !frisk.local_reduction !1
  %320 = extractelement <4 x float> %314, i64 2, !dbg !19
  %local.reduce153 = call float @llvm.maxnum.f32(float %312, float %320), !dbg !20, !frisk.local_reduction !1
  %321 = extractelement <4 x float> %314, i64 3, !dbg !19
  %local.reduce154 = call float @llvm.maxnum.f32(float %313, float %321), !dbg !20, !frisk.local_reduction !1
  %322 = load <4 x half>, ptr addrspace(3) %kdma.gep1806, align 8, !dbg !14, !frisk.k_tail_pair !39
  %323 = load <4 x half>, ptr addrspace(3) %kdma.gep1810, align 8, !dbg !14, !frisk.k_tail_pair !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.k_tail_prefetch !1
  %324 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %301, <4 x float> %317) #9, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.k_tail_prefetch !1
  %325 = load <4 x half>, ptr addrspace(3) %kdma.gep1807, align 8, !dbg !14, !frisk.k_tail_pair !40
  %326 = load <4 x half>, ptr addrspace(3) %kdma.gep1811, align 8, !dbg !14, !frisk.k_tail_pair !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.k_tail_prefetch !1
  %327 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %303, <4 x float> %268) #9, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !19, !frisk.k_tail_prefetch !1
  %328 = extractelement <4 x float> %324, i64 0, !dbg !19
  %local.reduce155 = call float @llvm.maxnum.f32(float %local.reduce151, float %328), !dbg !20, !frisk.local_reduction !1
  %329 = extractelement <4 x float> %324, i64 1, !dbg !19
  %local.reduce156 = call float @llvm.maxnum.f32(float %local.reduce152, float %329), !dbg !20, !frisk.local_reduction !1
  %330 = extractelement <4 x float> %324, i64 2, !dbg !19
  %local.reduce157 = call float @llvm.maxnum.f32(float %local.reduce153, float %330), !dbg !20, !frisk.local_reduction !1
  %331 = extractelement <4 x float> %324, i64 3, !dbg !19
  %local.reduce158 = call float @llvm.maxnum.f32(float %local.reduce154, float %331), !dbg !20, !frisk.local_reduction !1
  %332 = load <4 x half>, ptr addrspace(3) %kdma.gep1808, align 8, !dbg !14, !frisk.k_tail_pair !41
  %333 = load <4 x half>, ptr addrspace(3) %kdma.gep1812, align 8, !dbg !14, !frisk.k_tail_pair !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.k_tail_prefetch !1
  %334 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %305, <4 x float> %327) #9, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.k_tail_prefetch !1
  %335 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %315, <4 x float> %270) #9, !dbg !14
  %336 = extractelement <4 x float> %334, i64 0, !dbg !19
  %local.reduce159 = call float @llvm.maxnum.f32(float %local.reduce155, float %336), !dbg !20, !frisk.local_reduction !1
  %337 = extractelement <4 x float> %334, i64 1, !dbg !19
  %local.reduce160 = call float @llvm.maxnum.f32(float %local.reduce156, float %337), !dbg !20, !frisk.local_reduction !1
  %338 = extractelement <4 x float> %334, i64 2, !dbg !19
  %local.reduce161 = call float @llvm.maxnum.f32(float %local.reduce157, float %338), !dbg !20, !frisk.local_reduction !1
  %339 = extractelement <4 x float> %334, i64 3, !dbg !19
  %local.reduce162 = call float @llvm.maxnum.f32(float %local.reduce158, float %339), !dbg !20, !frisk.local_reduction !1
  %340 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %322, <4 x float> %335) #9, !dbg !14
  %341 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %325, <4 x float> %272) #9, !dbg !14
  %342 = extractelement <4 x float> %340, i64 0, !dbg !19
  %local.reduce163 = call float @llvm.maxnum.f32(float %local.reduce159, float %342), !dbg !20, !frisk.local_reduction !1
  %343 = extractelement <4 x float> %340, i64 1, !dbg !19
  %local.reduce164 = call float @llvm.maxnum.f32(float %local.reduce160, float %343), !dbg !20, !frisk.local_reduction !1
  %344 = extractelement <4 x float> %340, i64 2, !dbg !19
  %local.reduce165 = call float @llvm.maxnum.f32(float %local.reduce161, float %344), !dbg !20, !frisk.local_reduction !1
  %345 = extractelement <4 x float> %340, i64 3, !dbg !19
  %local.reduce166 = call float @llvm.maxnum.f32(float %local.reduce162, float %345), !dbg !20, !frisk.local_reduction !1
  %346 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %332, <4 x float> %341) #9, !dbg !14
  %347 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %316, <4 x float> %274) #9, !dbg !14
  %348 = extractelement <4 x float> %346, i64 0, !dbg !19
  %local.reduce167 = call float @llvm.maxnum.f32(float %local.reduce163, float %348), !dbg !20, !frisk.local_reduction !1
  %349 = extractelement <4 x float> %346, i64 1, !dbg !19
  %local.reduce168 = call float @llvm.maxnum.f32(float %local.reduce164, float %349), !dbg !20, !frisk.local_reduction !1
  %350 = extractelement <4 x float> %346, i64 2, !dbg !19
  %local.reduce169 = call float @llvm.maxnum.f32(float %local.reduce165, float %350), !dbg !20, !frisk.local_reduction !1
  %351 = extractelement <4 x float> %346, i64 3, !dbg !19
  %local.reduce170 = call float @llvm.maxnum.f32(float %local.reduce166, float %351), !dbg !20, !frisk.local_reduction !1
  %352 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %323, <4 x float> %347) #9, !dbg !14
  %353 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %326, <4 x float> %276) #9, !dbg !14
  %354 = extractelement <4 x float> %352, i64 0, !dbg !19
  %local.reduce171 = call float @llvm.maxnum.f32(float %local.reduce167, float %354), !dbg !20, !frisk.local_reduction !1
  %355 = extractelement <4 x float> %352, i64 1, !dbg !19
  %local.reduce172 = call float @llvm.maxnum.f32(float %local.reduce168, float %355), !dbg !20, !frisk.local_reduction !1
  %356 = extractelement <4 x float> %352, i64 2, !dbg !19
  %local.reduce173 = call float @llvm.maxnum.f32(float %local.reduce169, float %356), !dbg !20, !frisk.local_reduction !1
  %357 = extractelement <4 x float> %352, i64 3, !dbg !19
  %local.reduce174 = call float @llvm.maxnum.f32(float %local.reduce170, float %357), !dbg !20, !frisk.local_reduction !1
  %358 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %333, <4 x float> %353) #9, !dbg !14
  %359 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %298, <4 x float> %278) #9, !dbg !14
  %360 = extractelement <4 x float> %358, i64 0, !dbg !19
  %local.reduce175 = call float @llvm.maxnum.f32(float %local.reduce171, float %360), !dbg !20, !frisk.local_reduction !1
  %361 = extractelement <4 x float> %358, i64 1, !dbg !19
  %local.reduce176 = call float @llvm.maxnum.f32(float %local.reduce172, float %361), !dbg !20, !frisk.local_reduction !1
  %362 = extractelement <4 x float> %358, i64 2, !dbg !19
  %local.reduce177 = call float @llvm.maxnum.f32(float %local.reduce173, float %362), !dbg !20, !frisk.local_reduction !1
  %363 = extractelement <4 x float> %358, i64 3, !dbg !19
  %local.reduce178 = call float @llvm.maxnum.f32(float %local.reduce174, float %363), !dbg !20, !frisk.local_reduction !1
  %local.reduce179 = call float @llvm.maxnum.f32(float %local.reduce175, float %local.reduce176), !dbg !20, !frisk.local_reduction !1
  %local.reduce180 = call float @llvm.maxnum.f32(float %local.reduce177, float %local.reduce178), !dbg !20, !frisk.local_reduction !1
  %364 = call float @llvm.maxnum.f32(float %local.reduce179, float %local.reduce180), !dbg !20, !frisk.local_reduction !1
  %365 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %300, <4 x float> %359) #9, !dbg !14
  %366 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %302, <4 x float> %280) #9, !dbg !14
  %367 = extractelement <4 x float> %365, i64 0, !dbg !19
  %368 = extractelement <4 x float> %365, i64 1, !dbg !19
  %369 = extractelement <4 x float> %365, i64 2, !dbg !19
  %370 = extractelement <4 x float> %365, i64 3, !dbg !19
  %371 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %304, <4 x float> %366) #9, !dbg !14
  %372 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %299, <4 x float> %282) #9, !dbg !14
  %373 = extractelement <4 x float> %371, i64 0, !dbg !19
  %374 = extractelement <4 x float> %371, i64 1, !dbg !19
  %375 = extractelement <4 x float> %371, i64 2, !dbg !19
  %376 = extractelement <4 x float> %371, i64 3, !dbg !19
  %377 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %301, <4 x float> %372) #9, !dbg !14
  %378 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %303, <4 x float> %284) #9, !dbg !14
  %379 = extractelement <4 x float> %377, i64 0, !dbg !19
  %380 = extractelement <4 x float> %377, i64 1, !dbg !19
  %381 = extractelement <4 x float> %377, i64 2, !dbg !19
  %382 = extractelement <4 x float> %377, i64 3, !dbg !19
  %383 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %305, <4 x float> %378) #9, !dbg !14
  %384 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %315, <4 x float> %286) #9, !dbg !14
  %385 = extractelement <4 x float> %383, i64 0, !dbg !19
  %386 = extractelement <4 x float> %383, i64 1, !dbg !19
  %387 = extractelement <4 x float> %383, i64 2, !dbg !19
  %388 = extractelement <4 x float> %383, i64 3, !dbg !19
  %389 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %322, <4 x float> %384) #9, !dbg !14
  %390 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %325, <4 x float> %288) #9, !dbg !14
  %391 = extractelement <4 x float> %389, i64 0, !dbg !19
  %392 = extractelement <4 x float> %389, i64 1, !dbg !19
  %393 = extractelement <4 x float> %389, i64 2, !dbg !19
  %394 = extractelement <4 x float> %389, i64 3, !dbg !19
  %395 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %332, <4 x float> %390) #9, !dbg !14
  %396 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %316, <4 x float> %290) #9, !dbg !14
  %397 = extractelement <4 x float> %395, i64 0, !dbg !19
  %398 = extractelement <4 x float> %395, i64 1, !dbg !19
  %399 = extractelement <4 x float> %395, i64 2, !dbg !19
  %400 = extractelement <4 x float> %395, i64 3, !dbg !19
  %401 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %323, <4 x float> %396) #9, !dbg !14
  %402 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %326, <4 x float> %292) #9, !dbg !14
  %403 = extractelement <4 x float> %401, i64 0, !dbg !19
  %404 = extractelement <4 x float> %401, i64 1, !dbg !19
  %405 = extractelement <4 x float> %401, i64 2, !dbg !19
  %406 = extractelement <4 x float> %401, i64 3, !dbg !19
  %407 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %333, <4 x float> %402) #9, !dbg !14
  %408 = extractelement <4 x float> %407, i64 0, !dbg !19
  %409 = extractelement <4 x float> %407, i64 1, !dbg !19
  %410 = extractelement <4 x float> %407, i64 2, !dbg !19
  %411 = extractelement <4 x float> %407, i64 3, !dbg !19
  %412 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !20
  %413 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %412), !dbg !20
  %414 = add nuw nsw i32 %413, 64, !dbg !20
  %415 = xor i32 %413, 16, !dbg !20
  %416 = and i32 %414, 192, !dbg !20
  %417 = icmp samesign ult i32 %415, %416, !dbg !20
  %418 = select i1 %417, i32 %415, i32 %413, !dbg !20
  %419 = shl nuw nsw i32 %418, 2, !dbg !20
  %420 = bitcast float %364 to i32, !dbg !20
  %421 = call i32 @llvm.amdgcn.ds.bpermute(i32 %419, i32 %420), !dbg !20
  %422 = bitcast i32 %421 to float, !dbg !20
  %423 = call float @llvm.maxnum.f32(float %364, float %422), !dbg !20
  %424 = xor i32 %413, 32, !dbg !20
  %425 = icmp samesign ult i32 %424, %416, !dbg !20
  %426 = select i1 %425, i32 %424, i32 %413, !dbg !20
  %427 = shl nuw nsw i32 %426, 2, !dbg !20
  %428 = bitcast float %423 to i32, !dbg !20
  %429 = call i32 @llvm.amdgcn.ds.bpermute(i32 %427, i32 %428), !dbg !20
  %430 = bitcast i32 %429 to float, !dbg !20
  %431 = call float @llvm.maxnum.f32(float %423, float %430), !dbg !20
  %432 = extractvalue [2 x <1 x float>] %92, 0, !dbg !20
  %433 = insertelement <1 x float> poison, float %431, i64 0, !dbg !20
  %434 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %433, <1 x float> %432), !dbg !20
  %435 = call float @llvm.maxnum.f32(float %367, float 0xFFF0000000000000), !dbg !20, !frisk.local_reduction !1
  %local.reduce120 = call float @llvm.maxnum.f32(float %435, float %373), !dbg !20, !frisk.local_reduction !1
  %local.reduce121 = call float @llvm.maxnum.f32(float %368, float %374), !dbg !20, !frisk.local_reduction !1
  %local.reduce122 = call float @llvm.maxnum.f32(float %369, float %375), !dbg !20, !frisk.local_reduction !1
  %local.reduce123 = call float @llvm.maxnum.f32(float %370, float %376), !dbg !20, !frisk.local_reduction !1
  %local.reduce124 = call float @llvm.maxnum.f32(float %local.reduce120, float %379), !dbg !20, !frisk.local_reduction !1
  %local.reduce125 = call float @llvm.maxnum.f32(float %local.reduce121, float %380), !dbg !20, !frisk.local_reduction !1
  %local.reduce126 = call float @llvm.maxnum.f32(float %local.reduce122, float %381), !dbg !20, !frisk.local_reduction !1
  %local.reduce127 = call float @llvm.maxnum.f32(float %local.reduce123, float %382), !dbg !20, !frisk.local_reduction !1
  %local.reduce128 = call float @llvm.maxnum.f32(float %local.reduce124, float %385), !dbg !20, !frisk.local_reduction !1
  %local.reduce129 = call float @llvm.maxnum.f32(float %local.reduce125, float %386), !dbg !20, !frisk.local_reduction !1
  %local.reduce130 = call float @llvm.maxnum.f32(float %local.reduce126, float %387), !dbg !20, !frisk.local_reduction !1
  %local.reduce131 = call float @llvm.maxnum.f32(float %local.reduce127, float %388), !dbg !20, !frisk.local_reduction !1
  %local.reduce132 = call float @llvm.maxnum.f32(float %local.reduce128, float %391), !dbg !20, !frisk.local_reduction !1
  %local.reduce133 = call float @llvm.maxnum.f32(float %local.reduce129, float %392), !dbg !20, !frisk.local_reduction !1
  %local.reduce134 = call float @llvm.maxnum.f32(float %local.reduce130, float %393), !dbg !20, !frisk.local_reduction !1
  %local.reduce135 = call float @llvm.maxnum.f32(float %local.reduce131, float %394), !dbg !20, !frisk.local_reduction !1
  %local.reduce136 = call float @llvm.maxnum.f32(float %local.reduce132, float %397), !dbg !20, !frisk.local_reduction !1
  %local.reduce137 = call float @llvm.maxnum.f32(float %local.reduce133, float %398), !dbg !20, !frisk.local_reduction !1
  %local.reduce138 = call float @llvm.maxnum.f32(float %local.reduce134, float %399), !dbg !20, !frisk.local_reduction !1
  %local.reduce139 = call float @llvm.maxnum.f32(float %local.reduce135, float %400), !dbg !20, !frisk.local_reduction !1
  %local.reduce140 = call float @llvm.maxnum.f32(float %local.reduce136, float %403), !dbg !20, !frisk.local_reduction !1
  %local.reduce141 = call float @llvm.maxnum.f32(float %local.reduce137, float %404), !dbg !20, !frisk.local_reduction !1
  %local.reduce142 = call float @llvm.maxnum.f32(float %local.reduce138, float %405), !dbg !20, !frisk.local_reduction !1
  %local.reduce143 = call float @llvm.maxnum.f32(float %local.reduce139, float %406), !dbg !20, !frisk.local_reduction !1
  %local.reduce144 = call float @llvm.maxnum.f32(float %local.reduce140, float %408), !dbg !20, !frisk.local_reduction !1
  %local.reduce145 = call float @llvm.maxnum.f32(float %local.reduce141, float %409), !dbg !20, !frisk.local_reduction !1
  %local.reduce146 = call float @llvm.maxnum.f32(float %local.reduce142, float %410), !dbg !20, !frisk.local_reduction !1
  %local.reduce147 = call float @llvm.maxnum.f32(float %local.reduce143, float %411), !dbg !20, !frisk.local_reduction !1
  %local.reduce148 = call float @llvm.maxnum.f32(float %local.reduce144, float %local.reduce145), !dbg !20, !frisk.local_reduction !1
  %local.reduce149 = call float @llvm.maxnum.f32(float %local.reduce146, float %local.reduce147), !dbg !20, !frisk.local_reduction !1
  %436 = call float @llvm.maxnum.f32(float %local.reduce148, float %local.reduce149), !dbg !20, !frisk.local_reduction !1
  %437 = bitcast float %436 to i32, !dbg !20
  %438 = call i32 @llvm.amdgcn.ds.bpermute(i32 %419, i32 %437), !dbg !20
  %439 = bitcast i32 %438 to float, !dbg !20
  %440 = call float @llvm.maxnum.f32(float %436, float %439), !dbg !20
  %441 = bitcast float %440 to i32, !dbg !20
  %442 = call i32 @llvm.amdgcn.ds.bpermute(i32 %427, i32 %441), !dbg !20
  %443 = bitcast i32 %442 to float, !dbg !20
  %444 = call float @llvm.maxnum.f32(float %440, float %443), !dbg !20
  %445 = extractvalue [2 x <1 x float>] %92, 1, !dbg !20
  %446 = insertelement <1 x float> poison, float %444, i64 0, !dbg !20
  %447 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %446, <1 x float> %445), !dbg !20
  %448 = shufflevector <1 x float> %434, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !21
  %449 = shufflevector <1 x float> %447, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !21
  %450 = fsub <1 x float> %432, %434, !dbg !22
  %451 = extractelement <1 x float> %450, i64 0, !dbg !23
  %452 = call float @llvm.amdgcn.exp2.f32(float %451), !dbg !23
  %453 = insertelement <4 x float> poison, float %452, i64 0, !dbg !28
  %454 = shufflevector <4 x float> %453, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !28
  %455 = extractvalue [2 x <1 x float>] %91, 0, !dbg !24
  %456 = extractelement <1 x float> %455, i64 0, !dbg !24
  %457 = fmul float %456, %452, !dbg !24
  %458 = fsub <1 x float> %445, %447, !dbg !22
  %459 = extractelement <1 x float> %458, i64 0, !dbg !23
  %460 = call float @llvm.amdgcn.exp2.f32(float %459), !dbg !23
  %461 = insertelement <4 x float> poison, float %460, i64 0, !dbg !28
  %462 = shufflevector <4 x float> %461, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !28
  %463 = extractvalue [2 x <1 x float>] %91, 1, !dbg !24
  %464 = extractelement <1 x float> %463, i64 0, !dbg !24
  %465 = fmul float %464, %460, !dbg !24
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %vdb.address1436 = or disjoint i64 %107, %address.local921, !dbg !15
  %vdb.address1437 = trunc nuw i64 %vdb.address1436 to i32, !dbg !15
  %vdb.address1438 = shl nuw nsw i32 %vdb.address1437, 1, !dbg !15
  %vdb.address1445 = or disjoint i64 %107, %address.local927, !dbg !15
  %vdb.address1446 = trunc nuw i64 %vdb.address1445 to i32, !dbg !15
  %vdb.address1447 = shl nuw nsw i32 %vdb.address1446, 1, !dbg !15
  %vdb.dma.base = add i32 %dma.buffer.lds1374, 8192, !dbg !13
  %466 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1438, <4 x i32> %dma.buffer.rsrc1349) #10, !dbg !15
  %vdb.dma.base1452 = add i32 %dma.buffer.lds1371, 8192, !dbg !13
  %467 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1452, i32 %vdb.address1447, <4 x i32> %dma.buffer.rsrc1349) #10, !dbg !15
  %address.local929 = shl nuw nsw i32 %address.thread498, 3, !dbg !13
  %address.local930 = and i32 %address.local929, 120, !dbg !13
  %address.local932 = and i32 %address.local575, 1536, !dbg !13
  %address.local933 = or disjoint i32 %address.local930, %address.local932, !dbg !13
  %address.local934 = zext nneg i32 %address.local933 to i64, !dbg !13
  %address.local935 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local934, !dbg !13
  %468 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local935, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %469 = bitcast <4 x i32> %468 to <8 x half>, !dbg !13
  %address.local936 = and i64 %address.local499, 60, !dbg !13
  %address.local937 = add nuw nsw i64 %address.local936, %address.local499, !dbg !13
  %address.local939.idx = shl nuw nsw i64 %address.local937, 4, !dbg !13
  %address.local939 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %address.local939.idx, !dbg !13
  %address.local940 = lshr i64 %address.local555, 4, !dbg !13
  %address.local942.idx = mul nuw nsw i64 %address.local940, 768, !dbg !13
  %address.local942 = getelementptr i8, ptr addrspace(3) %address.local939, i64 %address.local942.idx, !dbg !13
  %address.local944.idx = mul nsw i64 %address.local542, -64, !dbg !13
  %address.local944 = getelementptr i8, ptr addrspace(3) %address.local942, i64 %address.local944.idx, !dbg !13
  %address.local945 = getelementptr i8, ptr addrspace(3) %address.local944, i64 256, !dbg !13
  %470 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local945, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %471 = bitcast <4 x i32> %470 to <8 x half>, !dbg !13
  %address.local949 = getelementptr i8, ptr addrspace(3) %address.local944, i64 512, !dbg !13
  %472 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local949, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %473 = bitcast <4 x i32> %472 to <8 x half>, !dbg !13
  %address.local953 = getelementptr i8, ptr addrspace(3) %address.local944, i64 768, !dbg !13
  %474 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local953, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %475 = bitcast <4 x i32> %474 to <8 x half>, !dbg !13
  %address.local955 = getelementptr i8, ptr addrspace(3) %address.local935, i64 4096, !dbg !13
  %476 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local955, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %477 = bitcast <4 x i32> %476 to <8 x half>, !dbg !13
  %address.local959 = getelementptr i8, ptr addrspace(3) %address.local944, i64 4352, !dbg !13
  %478 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local959, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %479 = bitcast <4 x i32> %478 to <8 x half>, !dbg !13
  %address.local963 = getelementptr i8, ptr addrspace(3) %address.local944, i64 4608, !dbg !13
  %480 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local963, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %481 = bitcast <4 x i32> %480 to <8 x half>, !dbg !13
  %address.local967 = getelementptr i8, ptr addrspace(3) %address.local944, i64 4864, !dbg !13
  %482 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local967, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %483 = shufflevector <8 x half> %469, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %484 = extractvalue [2 x <8 x float>] %90, 0, !dbg !28
  %485 = shufflevector <8 x float> %484, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %486 = fmul <4 x float> %485, %454, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %487 = fsub <4 x float> %307, %448, !dbg !21
  %488 = extractelement <4 x float> %487, i64 0, !dbg !25
  %489 = call float @llvm.amdgcn.exp2.f32(float %488), !dbg !25
  %490 = insertelement <4 x float> poison, float %489, i64 0, !dbg !27
  %491 = extractelement <4 x float> %487, i64 1, !dbg !25
  %492 = call float @llvm.amdgcn.exp2.f32(float %491), !dbg !25
  %493 = insertelement <4 x float> %490, float %492, i64 1, !dbg !27
  %494 = extractelement <4 x float> %487, i64 2, !dbg !25
  %495 = call float @llvm.amdgcn.exp2.f32(float %494), !dbg !25
  %496 = insertelement <4 x float> %493, float %495, i64 2, !dbg !27
  %497 = extractelement <4 x float> %487, i64 3, !dbg !25
  %498 = call float @llvm.amdgcn.exp2.f32(float %497), !dbg !25
  %499 = insertelement <4 x float> %496, float %498, i64 3, !dbg !27
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %499) #9, !dbg !27, !frisk.register_lifetimes !1
  %500 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !27
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %500) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %501 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %483, <4 x float> %486) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %502 = shufflevector <8 x half> %469, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %503 = shufflevector <8 x float> %484, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %504 = fmul <4 x float> %503, %454, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %505 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %502, <4 x float> %504) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %506 = extractvalue [2 x <8 x float>] %90, 1, !dbg !28
  %507 = shufflevector <8 x float> %506, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %508 = fmul <4 x float> %507, %462, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %509 = fsub <4 x float> %365, %449, !dbg !21
  %510 = extractelement <4 x float> %509, i64 0, !dbg !25
  %511 = call float @llvm.amdgcn.exp2.f32(float %510), !dbg !25
  %512 = insertelement <4 x float> poison, float %511, i64 0, !dbg !27
  %513 = extractelement <4 x float> %509, i64 1, !dbg !25
  %514 = call float @llvm.amdgcn.exp2.f32(float %513), !dbg !25
  %515 = insertelement <4 x float> %512, float %514, i64 1, !dbg !27
  %516 = extractelement <4 x float> %509, i64 2, !dbg !25
  %517 = call float @llvm.amdgcn.exp2.f32(float %516), !dbg !25
  %518 = insertelement <4 x float> %515, float %517, i64 2, !dbg !27
  %519 = extractelement <4 x float> %509, i64 3, !dbg !25
  %520 = call float @llvm.amdgcn.exp2.f32(float %519), !dbg !25
  %521 = insertelement <4 x float> %518, float %520, i64 3, !dbg !27
  %math.result388 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %521) #9, !dbg !27, !frisk.register_lifetimes !1
  %522 = fptrunc <4 x float> %math.result388 to <4 x half>, !dbg !27
  %math.packed389 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %522) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %523 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %483, <4 x float> %508) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %524 = shufflevector <8 x float> %506, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %525 = fmul <4 x float> %524, %462, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %526 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %502, <4 x float> %525) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %527 = shufflevector <8 x half> %471, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %528 = fsub <4 x float> %314, %448, !dbg !21
  %529 = extractelement <4 x float> %528, i64 0, !dbg !25
  %530 = call float @llvm.amdgcn.exp2.f32(float %529), !dbg !25
  %local.reduce89 = fadd float %489, %530, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor = call float asm sideeffect "", "=v,0"(float %local.reduce89) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %531 = insertelement <4 x float> poison, float %530, i64 0, !dbg !27
  %532 = extractelement <4 x float> %528, i64 1, !dbg !25
  %533 = call float @llvm.amdgcn.exp2.f32(float %532), !dbg !25
  %local.reduce90 = fadd float %492, %533, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1900 = call float asm sideeffect "", "=v,0"(float %local.reduce90) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %534 = insertelement <4 x float> %531, float %533, i64 1, !dbg !27
  %535 = extractelement <4 x float> %528, i64 2, !dbg !25
  %536 = call float @llvm.amdgcn.exp2.f32(float %535), !dbg !25
  %local.reduce91 = fadd float %495, %536, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1901 = call float asm sideeffect "", "=v,0"(float %local.reduce91) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %537 = insertelement <4 x float> %534, float %536, i64 2, !dbg !27
  %538 = extractelement <4 x float> %528, i64 3, !dbg !25
  %539 = call float @llvm.amdgcn.exp2.f32(float %538), !dbg !25
  %local.reduce92 = fadd float %498, %539, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1902 = call float asm sideeffect "", "=v,0"(float %local.reduce92) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %540 = insertelement <4 x float> %537, float %539, i64 3, !dbg !27
  %math.result374 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %540) #9, !dbg !27, !frisk.register_lifetimes !1
  %541 = fptrunc <4 x float> %math.result374 to <4 x half>, !dbg !27
  %math.packed375 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %541) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %542 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %527, <4 x float> %501) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %543 = shufflevector <8 x half> %471, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %544 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %543, <4 x float> %505) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %545 = fsub <4 x float> %371, %449, !dbg !21
  %546 = extractelement <4 x float> %545, i64 0, !dbg !25
  %547 = call float @llvm.amdgcn.exp2.f32(float %546), !dbg !25
  %local.reduce = fadd float %511, %547, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1903 = call float asm sideeffect "", "=v,0"(float %local.reduce) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %548 = insertelement <4 x float> poison, float %547, i64 0, !dbg !27
  %549 = extractelement <4 x float> %545, i64 1, !dbg !25
  %550 = call float @llvm.amdgcn.exp2.f32(float %549), !dbg !25
  %local.reduce59 = fadd float %514, %550, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1904 = call float asm sideeffect "", "=v,0"(float %local.reduce59) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %551 = insertelement <4 x float> %548, float %550, i64 1, !dbg !27
  %552 = extractelement <4 x float> %545, i64 2, !dbg !25
  %553 = call float @llvm.amdgcn.exp2.f32(float %552), !dbg !25
  %local.reduce60 = fadd float %517, %553, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1905 = call float asm sideeffect "", "=v,0"(float %local.reduce60) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %554 = insertelement <4 x float> %551, float %553, i64 2, !dbg !27
  %555 = extractelement <4 x float> %545, i64 3, !dbg !25
  %556 = call float @llvm.amdgcn.exp2.f32(float %555), !dbg !25
  %local.reduce61 = fadd float %520, %556, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1906 = call float asm sideeffect "", "=v,0"(float %local.reduce61) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %557 = insertelement <4 x float> %554, float %556, i64 3, !dbg !27
  %math.result390 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %557) #9, !dbg !27, !frisk.register_lifetimes !1
  %558 = fptrunc <4 x float> %math.result390 to <4 x half>, !dbg !27
  %math.packed391 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %558) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %559 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %527, <4 x float> %523) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %560 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %543, <4 x float> %526) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %561 = shufflevector <8 x half> %473, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %562 = fsub <4 x float> %324, %448, !dbg !21
  %563 = extractelement <4 x float> %562, i64 0, !dbg !25
  %564 = call float @llvm.amdgcn.exp2.f32(float %563), !dbg !25
  %local.reduce93 = fadd float %k0.sum.anchor, %564, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1907 = call float asm sideeffect "", "=v,0"(float %local.reduce93) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %565 = insertelement <4 x float> poison, float %564, i64 0, !dbg !27
  %566 = extractelement <4 x float> %562, i64 1, !dbg !25
  %567 = call float @llvm.amdgcn.exp2.f32(float %566), !dbg !25
  %local.reduce94 = fadd float %k0.sum.anchor1900, %567, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1908 = call float asm sideeffect "", "=v,0"(float %local.reduce94) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %568 = insertelement <4 x float> %565, float %567, i64 1, !dbg !27
  %569 = extractelement <4 x float> %562, i64 2, !dbg !25
  %570 = call float @llvm.amdgcn.exp2.f32(float %569), !dbg !25
  %local.reduce95 = fadd float %k0.sum.anchor1901, %570, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1909 = call float asm sideeffect "", "=v,0"(float %local.reduce95) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %571 = insertelement <4 x float> %568, float %570, i64 2, !dbg !27
  %572 = extractelement <4 x float> %562, i64 3, !dbg !25
  %573 = call float @llvm.amdgcn.exp2.f32(float %572), !dbg !25
  %local.reduce96 = fadd float %k0.sum.anchor1902, %573, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1910 = call float asm sideeffect "", "=v,0"(float %local.reduce96) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %574 = insertelement <4 x float> %571, float %573, i64 3, !dbg !27
  %math.result376 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %574) #9, !dbg !27, !frisk.register_lifetimes !1
  %575 = fptrunc <4 x float> %math.result376 to <4 x half>, !dbg !27
  %math.packed377 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %575) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %576 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %561, <4 x float> %542) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %577 = shufflevector <8 x half> %473, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %578 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %577, <4 x float> %544) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %579 = fsub <4 x float> %377, %449, !dbg !21
  %580 = extractelement <4 x float> %579, i64 0, !dbg !25
  %581 = call float @llvm.amdgcn.exp2.f32(float %580), !dbg !25
  %local.reduce62 = fadd float %k0.sum.anchor1903, %581, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1911 = call float asm sideeffect "", "=v,0"(float %local.reduce62) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %582 = insertelement <4 x float> poison, float %581, i64 0, !dbg !27
  %583 = extractelement <4 x float> %579, i64 1, !dbg !25
  %584 = call float @llvm.amdgcn.exp2.f32(float %583), !dbg !25
  %local.reduce63 = fadd float %k0.sum.anchor1904, %584, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1912 = call float asm sideeffect "", "=v,0"(float %local.reduce63) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %585 = insertelement <4 x float> %582, float %584, i64 1, !dbg !27
  %586 = extractelement <4 x float> %579, i64 2, !dbg !25
  %587 = call float @llvm.amdgcn.exp2.f32(float %586), !dbg !25
  %local.reduce64 = fadd float %k0.sum.anchor1905, %587, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1913 = call float asm sideeffect "", "=v,0"(float %local.reduce64) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %588 = insertelement <4 x float> %585, float %587, i64 2, !dbg !27
  %589 = extractelement <4 x float> %579, i64 3, !dbg !25
  %590 = call float @llvm.amdgcn.exp2.f32(float %589), !dbg !25
  %local.reduce65 = fadd float %k0.sum.anchor1906, %590, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1914 = call float asm sideeffect "", "=v,0"(float %local.reduce65) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %591 = insertelement <4 x float> %588, float %590, i64 3, !dbg !27
  %math.result392 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %591) #9, !dbg !27, !frisk.register_lifetimes !1
  %592 = fptrunc <4 x float> %math.result392 to <4 x half>, !dbg !27
  %math.packed393 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %592) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %593 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %561, <4 x float> %559) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %594 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %577, <4 x float> %560) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %595 = shufflevector <8 x half> %475, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %596 = fsub <4 x float> %334, %448, !dbg !21
  %597 = extractelement <4 x float> %596, i64 0, !dbg !25
  %598 = call float @llvm.amdgcn.exp2.f32(float %597), !dbg !25
  %local.reduce97 = fadd float %k0.sum.anchor1907, %598, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1915 = call float asm sideeffect "", "=v,0"(float %local.reduce97) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %599 = insertelement <4 x float> poison, float %598, i64 0, !dbg !27
  %600 = extractelement <4 x float> %596, i64 1, !dbg !25
  %601 = call float @llvm.amdgcn.exp2.f32(float %600), !dbg !25
  %local.reduce98 = fadd float %k0.sum.anchor1908, %601, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1916 = call float asm sideeffect "", "=v,0"(float %local.reduce98) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %602 = insertelement <4 x float> %599, float %601, i64 1, !dbg !27
  %603 = extractelement <4 x float> %596, i64 2, !dbg !25
  %604 = call float @llvm.amdgcn.exp2.f32(float %603), !dbg !25
  %local.reduce99 = fadd float %k0.sum.anchor1909, %604, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1917 = call float asm sideeffect "", "=v,0"(float %local.reduce99) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %605 = insertelement <4 x float> %602, float %604, i64 2, !dbg !27
  %606 = extractelement <4 x float> %596, i64 3, !dbg !25
  %607 = call float @llvm.amdgcn.exp2.f32(float %606), !dbg !25
  %local.reduce100 = fadd float %k0.sum.anchor1910, %607, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1918 = call float asm sideeffect "", "=v,0"(float %local.reduce100) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %608 = insertelement <4 x float> %605, float %607, i64 3, !dbg !27
  %math.result378 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %608) #9, !dbg !27, !frisk.register_lifetimes !1
  %609 = fptrunc <4 x float> %math.result378 to <4 x half>, !dbg !27
  %math.packed379 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %609) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %610 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %595, <4 x float> %576) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %611 = shufflevector <8 x half> %475, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %612 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %611, <4 x float> %578) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %613 = fsub <4 x float> %383, %449, !dbg !21
  %614 = extractelement <4 x float> %613, i64 0, !dbg !25
  %615 = call float @llvm.amdgcn.exp2.f32(float %614), !dbg !25
  %local.reduce66 = fadd float %k0.sum.anchor1911, %615, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1919 = call float asm sideeffect "", "=v,0"(float %local.reduce66) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %616 = insertelement <4 x float> poison, float %615, i64 0, !dbg !27
  %617 = extractelement <4 x float> %613, i64 1, !dbg !25
  %618 = call float @llvm.amdgcn.exp2.f32(float %617), !dbg !25
  %local.reduce67 = fadd float %k0.sum.anchor1912, %618, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1920 = call float asm sideeffect "", "=v,0"(float %local.reduce67) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %619 = insertelement <4 x float> %616, float %618, i64 1, !dbg !27
  %620 = extractelement <4 x float> %613, i64 2, !dbg !25
  %621 = call float @llvm.amdgcn.exp2.f32(float %620), !dbg !25
  %local.reduce68 = fadd float %k0.sum.anchor1913, %621, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1921 = call float asm sideeffect "", "=v,0"(float %local.reduce68) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %622 = insertelement <4 x float> %619, float %621, i64 2, !dbg !27
  %623 = extractelement <4 x float> %613, i64 3, !dbg !25
  %624 = call float @llvm.amdgcn.exp2.f32(float %623), !dbg !25
  %local.reduce69 = fadd float %k0.sum.anchor1914, %624, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1922 = call float asm sideeffect "", "=v,0"(float %local.reduce69) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %625 = insertelement <4 x float> %622, float %624, i64 3, !dbg !27
  %math.result394 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %625) #9, !dbg !27, !frisk.register_lifetimes !1
  %626 = fptrunc <4 x float> %math.result394 to <4 x half>, !dbg !27
  %math.packed395 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %626) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %627 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %595, <4 x float> %593) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %628 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %611, <4 x float> %594) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %629 = shufflevector <8 x half> %477, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %630 = fsub <4 x float> %340, %448, !dbg !21
  %631 = extractelement <4 x float> %630, i64 0, !dbg !25
  %632 = call float @llvm.amdgcn.exp2.f32(float %631), !dbg !25
  %local.reduce101 = fadd float %k0.sum.anchor1915, %632, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1923 = call float asm sideeffect "", "=v,0"(float %local.reduce101) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %633 = insertelement <4 x float> poison, float %632, i64 0, !dbg !27
  %634 = extractelement <4 x float> %630, i64 1, !dbg !25
  %635 = call float @llvm.amdgcn.exp2.f32(float %634), !dbg !25
  %local.reduce102 = fadd float %k0.sum.anchor1916, %635, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1924 = call float asm sideeffect "", "=v,0"(float %local.reduce102) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %636 = insertelement <4 x float> %633, float %635, i64 1, !dbg !27
  %637 = extractelement <4 x float> %630, i64 2, !dbg !25
  %638 = call float @llvm.amdgcn.exp2.f32(float %637), !dbg !25
  %local.reduce103 = fadd float %k0.sum.anchor1917, %638, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1925 = call float asm sideeffect "", "=v,0"(float %local.reduce103) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %639 = insertelement <4 x float> %636, float %638, i64 2, !dbg !27
  %640 = extractelement <4 x float> %630, i64 3, !dbg !25
  %641 = call float @llvm.amdgcn.exp2.f32(float %640), !dbg !25
  %local.reduce104 = fadd float %k0.sum.anchor1918, %641, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1926 = call float asm sideeffect "", "=v,0"(float %local.reduce104) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %642 = insertelement <4 x float> %639, float %641, i64 3, !dbg !27
  %math.result380 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %642) #9, !dbg !27, !frisk.register_lifetimes !1
  %643 = fptrunc <4 x float> %math.result380 to <4 x half>, !dbg !27
  %math.packed381 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %643) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %644 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %629, <4 x float> %610) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %645 = shufflevector <8 x half> %477, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %646 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %645, <4 x float> %612) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %647 = fsub <4 x float> %389, %449, !dbg !21
  %648 = extractelement <4 x float> %647, i64 0, !dbg !25
  %649 = call float @llvm.amdgcn.exp2.f32(float %648), !dbg !25
  %local.reduce70 = fadd float %k0.sum.anchor1919, %649, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1927 = call float asm sideeffect "", "=v,0"(float %local.reduce70) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %650 = insertelement <4 x float> poison, float %649, i64 0, !dbg !27
  %651 = extractelement <4 x float> %647, i64 1, !dbg !25
  %652 = call float @llvm.amdgcn.exp2.f32(float %651), !dbg !25
  %local.reduce71 = fadd float %k0.sum.anchor1920, %652, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1928 = call float asm sideeffect "", "=v,0"(float %local.reduce71) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %653 = insertelement <4 x float> %650, float %652, i64 1, !dbg !27
  %654 = extractelement <4 x float> %647, i64 2, !dbg !25
  %655 = call float @llvm.amdgcn.exp2.f32(float %654), !dbg !25
  %local.reduce72 = fadd float %k0.sum.anchor1921, %655, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1929 = call float asm sideeffect "", "=v,0"(float %local.reduce72) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %656 = insertelement <4 x float> %653, float %655, i64 2, !dbg !27
  %657 = extractelement <4 x float> %647, i64 3, !dbg !25
  %658 = call float @llvm.amdgcn.exp2.f32(float %657), !dbg !25
  %local.reduce73 = fadd float %k0.sum.anchor1922, %658, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1930 = call float asm sideeffect "", "=v,0"(float %local.reduce73) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %659 = insertelement <4 x float> %656, float %658, i64 3, !dbg !27
  %math.result396 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %659) #9, !dbg !27, !frisk.register_lifetimes !1
  %660 = fptrunc <4 x float> %math.result396 to <4 x half>, !dbg !27
  %math.packed397 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %660) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %661 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %629, <4 x float> %627) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %662 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %645, <4 x float> %628) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %663 = shufflevector <8 x half> %479, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %664 = fsub <4 x float> %346, %448, !dbg !21
  %665 = extractelement <4 x float> %664, i64 0, !dbg !25
  %666 = call float @llvm.amdgcn.exp2.f32(float %665), !dbg !25
  %local.reduce105 = fadd float %k0.sum.anchor1923, %666, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1931 = call float asm sideeffect "", "=v,0"(float %local.reduce105) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %667 = insertelement <4 x float> poison, float %666, i64 0, !dbg !27
  %668 = extractelement <4 x float> %664, i64 1, !dbg !25
  %669 = call float @llvm.amdgcn.exp2.f32(float %668), !dbg !25
  %local.reduce106 = fadd float %k0.sum.anchor1924, %669, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1932 = call float asm sideeffect "", "=v,0"(float %local.reduce106) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %670 = insertelement <4 x float> %667, float %669, i64 1, !dbg !27
  %671 = extractelement <4 x float> %664, i64 2, !dbg !25
  %672 = call float @llvm.amdgcn.exp2.f32(float %671), !dbg !25
  %local.reduce107 = fadd float %k0.sum.anchor1925, %672, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1933 = call float asm sideeffect "", "=v,0"(float %local.reduce107) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %673 = insertelement <4 x float> %670, float %672, i64 2, !dbg !27
  %674 = extractelement <4 x float> %664, i64 3, !dbg !25
  %675 = call float @llvm.amdgcn.exp2.f32(float %674), !dbg !25
  %local.reduce108 = fadd float %k0.sum.anchor1926, %675, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1934 = call float asm sideeffect "", "=v,0"(float %local.reduce108) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %676 = insertelement <4 x float> %673, float %675, i64 3, !dbg !27
  %math.result382 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %676) #9, !dbg !27, !frisk.register_lifetimes !1
  %677 = fptrunc <4 x float> %math.result382 to <4 x half>, !dbg !27
  %math.packed383 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %677) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %678 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %663, <4 x float> %644) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %679 = shufflevector <8 x half> %479, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %680 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %679, <4 x float> %646) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %681 = fsub <4 x float> %395, %449, !dbg !21
  %682 = extractelement <4 x float> %681, i64 0, !dbg !25
  %683 = call float @llvm.amdgcn.exp2.f32(float %682), !dbg !25
  %local.reduce74 = fadd float %k0.sum.anchor1927, %683, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1935 = call float asm sideeffect "", "=v,0"(float %local.reduce74) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %684 = insertelement <4 x float> poison, float %683, i64 0, !dbg !27
  %685 = extractelement <4 x float> %681, i64 1, !dbg !25
  %686 = call float @llvm.amdgcn.exp2.f32(float %685), !dbg !25
  %local.reduce75 = fadd float %k0.sum.anchor1928, %686, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1936 = call float asm sideeffect "", "=v,0"(float %local.reduce75) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %687 = insertelement <4 x float> %684, float %686, i64 1, !dbg !27
  %688 = extractelement <4 x float> %681, i64 2, !dbg !25
  %689 = call float @llvm.amdgcn.exp2.f32(float %688), !dbg !25
  %local.reduce76 = fadd float %k0.sum.anchor1929, %689, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1937 = call float asm sideeffect "", "=v,0"(float %local.reduce76) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %690 = insertelement <4 x float> %687, float %689, i64 2, !dbg !27
  %691 = extractelement <4 x float> %681, i64 3, !dbg !25
  %692 = call float @llvm.amdgcn.exp2.f32(float %691), !dbg !25
  %local.reduce77 = fadd float %k0.sum.anchor1930, %692, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1938 = call float asm sideeffect "", "=v,0"(float %local.reduce77) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %693 = insertelement <4 x float> %690, float %692, i64 3, !dbg !27
  %math.result398 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %693) #9, !dbg !27, !frisk.register_lifetimes !1
  %694 = fptrunc <4 x float> %math.result398 to <4 x half>, !dbg !27
  %math.packed399 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %694) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %695 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %663, <4 x float> %661) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %696 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %679, <4 x float> %662) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %697 = shufflevector <8 x half> %481, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %698 = fsub <4 x float> %352, %448, !dbg !21
  %699 = extractelement <4 x float> %698, i64 0, !dbg !25
  %700 = call float @llvm.amdgcn.exp2.f32(float %699), !dbg !25
  %local.reduce109 = fadd float %k0.sum.anchor1931, %700, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1939 = call float asm sideeffect "", "=v,0"(float %local.reduce109) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %701 = insertelement <4 x float> poison, float %700, i64 0, !dbg !27
  %702 = extractelement <4 x float> %698, i64 1, !dbg !25
  %703 = call float @llvm.amdgcn.exp2.f32(float %702), !dbg !25
  %local.reduce110 = fadd float %k0.sum.anchor1932, %703, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1940 = call float asm sideeffect "", "=v,0"(float %local.reduce110) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %704 = insertelement <4 x float> %701, float %703, i64 1, !dbg !27
  %705 = extractelement <4 x float> %698, i64 2, !dbg !25
  %706 = call float @llvm.amdgcn.exp2.f32(float %705), !dbg !25
  %local.reduce111 = fadd float %k0.sum.anchor1933, %706, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1941 = call float asm sideeffect "", "=v,0"(float %local.reduce111) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %707 = insertelement <4 x float> %704, float %706, i64 2, !dbg !27
  %708 = extractelement <4 x float> %698, i64 3, !dbg !25
  %709 = call float @llvm.amdgcn.exp2.f32(float %708), !dbg !25
  %local.reduce112 = fadd float %k0.sum.anchor1934, %709, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1942 = call float asm sideeffect "", "=v,0"(float %local.reduce112) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %710 = insertelement <4 x float> %707, float %709, i64 3, !dbg !27
  %math.result384 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %710) #9, !dbg !27, !frisk.register_lifetimes !1
  %711 = fptrunc <4 x float> %math.result384 to <4 x half>, !dbg !27
  %math.packed385 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %711) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %712 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %697, <4 x float> %678) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %713 = shufflevector <8 x half> %481, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %714 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %713, <4 x float> %680) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %715 = fsub <4 x float> %401, %449, !dbg !21
  %716 = extractelement <4 x float> %715, i64 0, !dbg !25
  %717 = call float @llvm.amdgcn.exp2.f32(float %716), !dbg !25
  %local.reduce78 = fadd float %k0.sum.anchor1935, %717, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1943 = call float asm sideeffect "", "=v,0"(float %local.reduce78) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %718 = insertelement <4 x float> poison, float %717, i64 0, !dbg !27
  %719 = extractelement <4 x float> %715, i64 1, !dbg !25
  %720 = call float @llvm.amdgcn.exp2.f32(float %719), !dbg !25
  %local.reduce79 = fadd float %k0.sum.anchor1936, %720, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1944 = call float asm sideeffect "", "=v,0"(float %local.reduce79) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %721 = insertelement <4 x float> %718, float %720, i64 1, !dbg !27
  %722 = extractelement <4 x float> %715, i64 2, !dbg !25
  %723 = call float @llvm.amdgcn.exp2.f32(float %722), !dbg !25
  %local.reduce80 = fadd float %k0.sum.anchor1937, %723, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1945 = call float asm sideeffect "", "=v,0"(float %local.reduce80) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %724 = insertelement <4 x float> %721, float %723, i64 2, !dbg !27
  %725 = extractelement <4 x float> %715, i64 3, !dbg !25
  %726 = call float @llvm.amdgcn.exp2.f32(float %725), !dbg !25
  %local.reduce81 = fadd float %k0.sum.anchor1938, %726, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1946 = call float asm sideeffect "", "=v,0"(float %local.reduce81) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %727 = insertelement <4 x float> %724, float %726, i64 3, !dbg !27
  %math.result400 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %727) #9, !dbg !27, !frisk.register_lifetimes !1
  %728 = fptrunc <4 x float> %math.result400 to <4 x half>, !dbg !27
  %math.packed401 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %728) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %729 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %697, <4 x float> %695) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %730 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %713, <4 x float> %696) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %731 = bitcast <4 x i32> %482 to <8 x half>, !dbg !13
  %732 = shufflevector <8 x half> %731, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %733 = fsub <4 x float> %358, %448, !dbg !21
  %734 = extractelement <4 x float> %733, i64 0, !dbg !25
  %735 = call float @llvm.amdgcn.exp2.f32(float %734), !dbg !25
  %local.reduce113 = fadd float %k0.sum.anchor1939, %735, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1947 = call float asm sideeffect "", "=v,0"(float %local.reduce113) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %736 = insertelement <4 x float> poison, float %735, i64 0, !dbg !27
  %737 = extractelement <4 x float> %733, i64 1, !dbg !25
  %738 = call float @llvm.amdgcn.exp2.f32(float %737), !dbg !25
  %local.reduce114 = fadd float %k0.sum.anchor1940, %738, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1948 = call float asm sideeffect "", "=v,0"(float %local.reduce114) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %local.reduce117 = fadd float %k0.sum.anchor1947, %k0.sum.anchor1948, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1949 = call float asm sideeffect "", "=v,0"(float %local.reduce117) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %739 = insertelement <4 x float> %736, float %738, i64 1, !dbg !27
  %740 = extractelement <4 x float> %733, i64 2, !dbg !25
  %741 = call float @llvm.amdgcn.exp2.f32(float %740), !dbg !25
  %local.reduce115 = fadd float %k0.sum.anchor1941, %741, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1950 = call float asm sideeffect "", "=v,0"(float %local.reduce115) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %742 = insertelement <4 x float> %739, float %741, i64 2, !dbg !27
  %743 = extractelement <4 x float> %733, i64 3, !dbg !25
  %744 = call float @llvm.amdgcn.exp2.f32(float %743), !dbg !25
  %local.reduce116 = fadd float %k0.sum.anchor1942, %744, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1951 = call float asm sideeffect "", "=v,0"(float %local.reduce116) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %local.reduce118 = fadd float %k0.sum.anchor1950, %k0.sum.anchor1951, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1952 = call float asm sideeffect "", "=v,0"(float %local.reduce118) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %745 = fadd float %k0.sum.anchor1949, %k0.sum.anchor1952, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1953 = call float asm sideeffect "", "=v,0"(float %745) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %746 = fadd float %k0.sum.anchor1953, %457, !dbg !26
  %k0.sum.anchor1954 = call float asm sideeffect "", "=v,0"(float %746) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %747 = insertelement <1 x float> poison, float %k0.sum.anchor1954, i64 0, !dbg !7
  %748 = insertvalue [2 x <1 x float>] poison, <1 x float> %747, 0, !dbg !7
  %749 = insertelement <4 x float> %742, float %744, i64 3, !dbg !27
  %math.result386 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %749) #9, !dbg !27, !frisk.register_lifetimes !1
  %750 = fptrunc <4 x float> %math.result386 to <4 x half>, !dbg !27
  %math.packed387 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %750) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %751 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %732, <4 x float> %712) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %752 = shufflevector <8 x half> %731, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %753 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %752, <4 x float> %714) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %754 = fsub <4 x float> %407, %449, !dbg !21
  %755 = extractelement <4 x float> %754, i64 0, !dbg !25
  %756 = call float @llvm.amdgcn.exp2.f32(float %755), !dbg !25
  %local.reduce82 = fadd float %k0.sum.anchor1943, %756, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1955 = call float asm sideeffect "", "=v,0"(float %local.reduce82) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %757 = insertelement <4 x float> poison, float %756, i64 0, !dbg !27
  %758 = extractelement <4 x float> %754, i64 1, !dbg !25
  %759 = call float @llvm.amdgcn.exp2.f32(float %758), !dbg !25
  %local.reduce83 = fadd float %k0.sum.anchor1944, %759, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1956 = call float asm sideeffect "", "=v,0"(float %local.reduce83) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %local.reduce86 = fadd float %k0.sum.anchor1955, %k0.sum.anchor1956, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1957 = call float asm sideeffect "", "=v,0"(float %local.reduce86) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %760 = insertelement <4 x float> %757, float %759, i64 1, !dbg !27
  %761 = extractelement <4 x float> %754, i64 2, !dbg !25
  %762 = call float @llvm.amdgcn.exp2.f32(float %761), !dbg !25
  %local.reduce84 = fadd float %k0.sum.anchor1945, %762, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1958 = call float asm sideeffect "", "=v,0"(float %local.reduce84) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %763 = insertelement <4 x float> %760, float %762, i64 2, !dbg !27
  %764 = extractelement <4 x float> %754, i64 3, !dbg !25
  %765 = call float @llvm.amdgcn.exp2.f32(float %764), !dbg !25
  %local.reduce85 = fadd float %k0.sum.anchor1946, %765, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1959 = call float asm sideeffect "", "=v,0"(float %local.reduce85) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %local.reduce87 = fadd float %k0.sum.anchor1958, %k0.sum.anchor1959, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1960 = call float asm sideeffect "", "=v,0"(float %local.reduce87) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %766 = fadd float %k0.sum.anchor1957, %k0.sum.anchor1960, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1961 = call float asm sideeffect "", "=v,0"(float %766) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %767 = fadd float %k0.sum.anchor1961, %465, !dbg !26
  %k0.sum.anchor1962 = call float asm sideeffect "", "=v,0"(float %767) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %768 = insertelement <1 x float> poison, float %k0.sum.anchor1962, i64 0, !dbg !7
  %769 = insertvalue [2 x <1 x float>] %748, <1 x float> %768, 1, !dbg !7
  %770 = insertelement <4 x float> %763, float %765, i64 3, !dbg !27
  %math.result402 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %770) #9, !dbg !27, !frisk.register_lifetimes !1
  %771 = fptrunc <4 x float> %math.result402 to <4 x half>, !dbg !27
  %math.packed403 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %771) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %772 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %732, <4 x float> %729) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %773 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %752, <4 x float> %730) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %vdb.address1462 = or disjoint i64 %187, %address.local921, !dbg !15
  %vdb.address1463 = trunc nuw i64 %vdb.address1462 to i32, !dbg !15
  %vdb.address1464 = shl nuw nsw i32 %vdb.address1463, 1, !dbg !15
  %vdb.address1471 = or disjoint i64 %187, %address.local927, !dbg !15
  %vdb.address1472 = trunc nuw i64 %vdb.address1471 to i32, !dbg !15
  %vdb.address1473 = shl nuw nsw i32 %vdb.address1472, 1, !dbg !15
  %774 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1374, i32 %vdb.address1464, <4 x i32> %dma.buffer.rsrc1349) #10, !dbg !15
  %775 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1371, i32 %vdb.address1473, <4 x i32> %dma.buffer.rsrc1349) #10, !dbg !15
  %vdb.read = getelementptr i8, ptr addrspace(3) %address.local935, i64 8192, !dbg !13
  %776 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %777 = bitcast <4 x i32> %776 to <8 x half>, !dbg !13
  %vdb.read1453 = getelementptr i8, ptr addrspace(3) %address.local944, i64 8448, !dbg !13
  %778 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1453, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %779 = bitcast <4 x i32> %778 to <8 x half>, !dbg !13
  %vdb.read1454 = getelementptr i8, ptr addrspace(3) %address.local944, i64 8704, !dbg !13
  %780 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1454, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %781 = bitcast <4 x i32> %780 to <8 x half>, !dbg !13
  %vdb.read1455 = getelementptr i8, ptr addrspace(3) %address.local944, i64 8960, !dbg !13
  %782 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1455, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %783 = bitcast <4 x i32> %782 to <8 x half>, !dbg !13
  %vdb.read1456 = getelementptr i8, ptr addrspace(3) %address.local935, i64 12288, !dbg !13
  %784 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1456, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %785 = bitcast <4 x i32> %784 to <8 x half>, !dbg !13
  %vdb.read1457 = getelementptr i8, ptr addrspace(3) %address.local944, i64 12544, !dbg !13
  %786 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1457, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %787 = bitcast <4 x i32> %786 to <8 x half>, !dbg !13
  %vdb.read1458 = getelementptr i8, ptr addrspace(3) %address.local944, i64 12800, !dbg !13
  %788 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1458, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %789 = bitcast <4 x i32> %788 to <8 x half>, !dbg !13
  %vdb.read1459 = getelementptr i8, ptr addrspace(3) %address.local944, i64 13056, !dbg !13
  %790 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1459, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %791 = shufflevector <8 x half> %777, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %792 = extractvalue [2 x <8 x float>] %93, 0, !dbg !28
  %793 = shufflevector <8 x float> %792, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %794 = fmul <4 x float> %793, %454, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %795 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %791, <4 x float> %794) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %796 = shufflevector <8 x half> %777, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %797 = shufflevector <8 x float> %792, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %798 = fmul <4 x float> %797, %454, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %799 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %796, <4 x float> %798) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %800 = extractvalue [2 x <8 x float>] %93, 1, !dbg !28
  %801 = shufflevector <8 x float> %800, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %802 = fmul <4 x float> %801, %462, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %803 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %791, <4 x float> %802) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %804 = shufflevector <8 x float> %800, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %805 = fmul <4 x float> %804, %462, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %806 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %796, <4 x float> %805) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %807 = shufflevector <8 x half> %779, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %808 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %807, <4 x float> %795) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %809 = shufflevector <8 x half> %779, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %810 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %809, <4 x float> %799) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %811 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %807, <4 x float> %803) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %812 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %809, <4 x float> %806) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %813 = shufflevector <8 x half> %781, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %814 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %813, <4 x float> %808) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %815 = shufflevector <8 x half> %781, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %816 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %815, <4 x float> %810) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %817 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %813, <4 x float> %811) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %818 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %815, <4 x float> %812) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %819 = shufflevector <8 x half> %783, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %820 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %819, <4 x float> %814) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %821 = shufflevector <8 x half> %783, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %822 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %821, <4 x float> %816) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %823 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %819, <4 x float> %817) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %824 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %821, <4 x float> %818) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %825 = shufflevector <8 x half> %785, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %826 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %825, <4 x float> %820) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %827 = shufflevector <8 x half> %785, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %828 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %827, <4 x float> %822) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %829 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %825, <4 x float> %823) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %830 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %827, <4 x float> %824) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %831 = shufflevector <8 x half> %787, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %832 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %831, <4 x float> %826) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %833 = shufflevector <8 x half> %787, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %834 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %833, <4 x float> %828) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %835 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %831, <4 x float> %829) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %836 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %833, <4 x float> %830) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %837 = shufflevector <8 x half> %789, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %838 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %837, <4 x float> %832) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %839 = shufflevector <8 x half> %789, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %840 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %839, <4 x float> %834) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %841 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %837, <4 x float> %835) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %842 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %839, <4 x float> %836) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %843 = bitcast <4 x i32> %790 to <8 x half>, !dbg !13
  %844 = shufflevector <8 x half> %843, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %845 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %844, <4 x float> %838) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %846 = shufflevector <8 x half> %843, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %847 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %846, <4 x float> %840) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %848 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %844, <4 x float> %841) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %849 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %846, <4 x float> %842) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %vdb.address1480 = or disjoint i64 %240, %address.local921, !dbg !15
  %vdb.address1481 = trunc nuw i64 %vdb.address1480 to i32, !dbg !15
  %vdb.address1482 = shl nuw nsw i32 %vdb.address1481, 1, !dbg !15
  %vdb.address1489 = or disjoint i64 %240, %address.local927, !dbg !15
  %vdb.address1490 = trunc nuw i64 %vdb.address1489 to i32, !dbg !15
  %vdb.address1491 = shl nuw nsw i32 %vdb.address1490, 1, !dbg !15
  %850 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1482, <4 x i32> %dma.buffer.rsrc1349) #10, !dbg !15
  %851 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1452, i32 %vdb.address1491, <4 x i32> %dma.buffer.rsrc1349) #10, !dbg !15
  %852 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local935, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %853 = bitcast <4 x i32> %852 to <8 x half>, !dbg !13
  %854 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local945, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %855 = bitcast <4 x i32> %854 to <8 x half>, !dbg !13
  %856 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local949, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %857 = bitcast <4 x i32> %856 to <8 x half>, !dbg !13
  %858 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local953, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %859 = bitcast <4 x i32> %858 to <8 x half>, !dbg !13
  %860 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local955, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %861 = bitcast <4 x i32> %860 to <8 x half>, !dbg !13
  %862 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local959, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %863 = bitcast <4 x i32> %862 to <8 x half>, !dbg !13
  %864 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local963, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %865 = bitcast <4 x i32> %864 to <8 x half>, !dbg !13
  %866 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local967, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %867 = shufflevector <8 x half> %853, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %868 = extractvalue [2 x <8 x float>] %94, 0, !dbg !28
  %869 = shufflevector <8 x float> %868, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %870 = fmul <4 x float> %869, %454, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %871 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %867, <4 x float> %870) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %872 = shufflevector <8 x half> %853, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %873 = shufflevector <8 x float> %868, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %874 = fmul <4 x float> %873, %454, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %875 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %872, <4 x float> %874) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %876 = extractvalue [2 x <8 x float>] %94, 1, !dbg !28
  %877 = shufflevector <8 x float> %876, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %878 = fmul <4 x float> %877, %462, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %879 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %867, <4 x float> %878) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %880 = shufflevector <8 x float> %876, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %881 = fmul <4 x float> %880, %462, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %882 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %872, <4 x float> %881) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %883 = shufflevector <8 x half> %855, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %884 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %883, <4 x float> %871) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %885 = shufflevector <8 x half> %855, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %886 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %885, <4 x float> %875) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %887 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %883, <4 x float> %879) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %888 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %885, <4 x float> %882) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %889 = shufflevector <8 x half> %857, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %890 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %889, <4 x float> %884) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %891 = shufflevector <8 x half> %857, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %892 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %891, <4 x float> %886) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %893 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %889, <4 x float> %887) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %894 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %891, <4 x float> %888) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %895 = shufflevector <8 x half> %859, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %896 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %895, <4 x float> %890) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %897 = shufflevector <8 x half> %859, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %898 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %897, <4 x float> %892) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %899 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %895, <4 x float> %893) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %900 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %897, <4 x float> %894) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %901 = shufflevector <8 x half> %861, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %902 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %901, <4 x float> %896) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %903 = shufflevector <8 x half> %861, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %904 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %903, <4 x float> %898) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %905 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %901, <4 x float> %899) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %906 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %903, <4 x float> %900) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %907 = shufflevector <8 x half> %863, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %908 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %907, <4 x float> %902) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %909 = shufflevector <8 x half> %863, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %910 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %909, <4 x float> %904) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %911 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %907, <4 x float> %905) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %912 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %909, <4 x float> %906) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %913 = shufflevector <8 x half> %865, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %914 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %913, <4 x float> %908) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %915 = shufflevector <8 x half> %865, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %916 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %915, <4 x float> %910) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %917 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %913, <4 x float> %911) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %918 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %915, <4 x float> %912) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %919 = bitcast <4 x i32> %866 to <8 x half>, !dbg !13
  %920 = shufflevector <8 x half> %919, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %921 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %920, <4 x float> %914) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %922 = shufflevector <8 x half> %919, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %923 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %922, <4 x float> %916) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %924 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %920, <4 x float> %917) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %925 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %922, <4 x float> %918) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %926 = add nuw nsw i64 %89, 1, !dbg !7
  %k0.has.next = icmp ult i64 %926, 32, !dbg !13
  br i1 %k0.has.next, label %k0.issue, label %k0.join, !dbg !13

k0.issue:                                         ; preds = %97
  %k0.next. = shl nuw nsw i64 %926, 14, !dbg !12
  %k0.next.1975 = or disjoint i64 %99, %k0.next., !dbg !12
  %k0.next.kdma.source1516 = or disjoint i64 %k0.next.1975, %kdma.source1515, !dbg !12
  %k0.next.kdma.source1517 = trunc nuw i64 %k0.next.kdma.source1516 to i32, !dbg !12
  %k0.next.kdma.source1518 = shl nuw nsw i32 %k0.next.kdma.source1517, 1, !dbg !12
  %k0.next.kdma.offset = or disjoint i32 %k0.next.kdma.source1518, %kdma.intra, !dbg !12
  %k0.next.1976 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1374, i32 %k0.next.kdma.offset, <4 x i32> %dma.buffer.rsrc1373) #10, !dbg !12, !frisk.cross_bn_k0 !1
  %k0.next.kdma.source1530 = or disjoint i64 %k0.next.1975, %kdma.source1529, !dbg !12
  %k0.next.kdma.source1531 = trunc nuw i64 %k0.next.kdma.source1530 to i32, !dbg !12
  %k0.next.kdma.source1532 = shl nuw nsw i32 %k0.next.kdma.source1531, 1, !dbg !12
  %k0.next.kdma.offset1533 = or disjoint i32 %k0.next.kdma.source1532, %kdma.intra.hi, !dbg !12
  %k0.next.1977 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1534, i32 %k0.next.kdma.offset1533, <4 x i32> %dma.buffer.rsrc1373) #10, !dbg !12, !frisk.cross_bn_k0 !1
  %k0.next.kdma.source1547 = or disjoint i64 %k0.next.1975, %kdma.source1546, !dbg !12
  %k0.next.kdma.source1548 = trunc nuw i64 %k0.next.kdma.source1547 to i32, !dbg !12
  %k0.next.kdma.source1549 = shl nuw nsw i32 %k0.next.kdma.source1548, 1, !dbg !12
  %k0.next.kdma.offset1550 = or disjoint i32 %k0.next.kdma.source1549, %kdma.intra, !dbg !12
  %k0.next.1978 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1371, i32 %k0.next.kdma.offset1550, <4 x i32> %dma.buffer.rsrc1373) #10, !dbg !12, !frisk.cross_bn_k0 !1
  %k0.next.kdma.source1564 = or disjoint i64 %k0.next.1975, %kdma.source1563, !dbg !12
  %k0.next.kdma.source1565 = trunc nuw i64 %k0.next.kdma.source1564 to i32, !dbg !12
  %k0.next.kdma.source1566 = shl nuw nsw i32 %k0.next.kdma.source1565, 1, !dbg !12
  %k0.next.kdma.offset1567 = or disjoint i32 %k0.next.kdma.source1566, %kdma.intra.hi, !dbg !12
  %k0.next.1979 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1568, i32 %k0.next.kdma.offset1567, <4 x i32> %dma.buffer.rsrc1373) #10, !dbg !12, !frisk.cross_bn_k0 !1
  br label %k0.join

k0.join:                                          ; preds = %97, %k0.issue
  %927 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %928 = bitcast <4 x i32> %927 to <8 x half>, !dbg !13
  %929 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1453, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %930 = bitcast <4 x i32> %929 to <8 x half>, !dbg !13
  %931 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1454, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %932 = bitcast <4 x i32> %931 to <8 x half>, !dbg !13
  %933 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1455, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %934 = bitcast <4 x i32> %933 to <8 x half>, !dbg !13
  %935 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1456, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %936 = bitcast <4 x i32> %935 to <8 x half>, !dbg !13
  %937 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1457, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %938 = bitcast <4 x i32> %937 to <8 x half>, !dbg !13
  %939 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1458, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %940 = bitcast <4 x i32> %939 to <8 x half>, !dbg !13
  %941 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1459, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %942 = shufflevector <8 x half> %928, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %943 = extractvalue [2 x <8 x float>] %95, 0, !dbg !28
  %944 = shufflevector <8 x float> %943, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %945 = fmul <4 x float> %944, %454, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %946 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %942, <4 x float> %945) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %947 = shufflevector <8 x half> %928, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %948 = shufflevector <8 x float> %943, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %949 = fmul <4 x float> %948, %454, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %950 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %947, <4 x float> %949) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %951 = extractvalue [2 x <8 x float>] %95, 1, !dbg !28
  %952 = shufflevector <8 x float> %951, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %953 = fmul <4 x float> %952, %462, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %954 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %942, <4 x float> %953) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %955 = shufflevector <8 x float> %951, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %956 = fmul <4 x float> %955, %462, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %957 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %947, <4 x float> %956) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %958 = shufflevector <8 x half> %930, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %959 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %958, <4 x float> %946) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %960 = shufflevector <8 x half> %930, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %961 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %960, <4 x float> %950) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %962 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %958, <4 x float> %954) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %963 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %960, <4 x float> %957) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %964 = shufflevector <8 x half> %932, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %965 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %964, <4 x float> %959) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %966 = shufflevector <8 x half> %932, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %967 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %966, <4 x float> %961) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %968 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %964, <4 x float> %962) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %969 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %966, <4 x float> %963) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %970 = shufflevector <8 x half> %934, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %971 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %970, <4 x float> %965) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %972 = shufflevector <8 x half> %934, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %973 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %972, <4 x float> %967) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %974 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %970, <4 x float> %968) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %975 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %972, <4 x float> %969) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %976 = shufflevector <8 x half> %936, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %977 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %976, <4 x float> %971) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %978 = shufflevector <8 x half> %936, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %979 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %978, <4 x float> %973) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %980 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %976, <4 x float> %974) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %981 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %978, <4 x float> %975) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %982 = shufflevector <8 x half> %938, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %983 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %982, <4 x float> %977) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %984 = shufflevector <8 x half> %938, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %985 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %984, <4 x float> %979) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %986 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %982, <4 x float> %980) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %987 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %984, <4 x float> %981) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %988 = shufflevector <8 x half> %940, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %989 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %988, <4 x float> %983) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %990 = shufflevector <8 x half> %940, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %991 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %990, <4 x float> %985) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %992 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %988, <4 x float> %986) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %993 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %990, <4 x float> %987) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %994 = bitcast <4 x i32> %941 to <8 x half>, !dbg !13
  %995 = shufflevector <8 x half> %994, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %996 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %995, <4 x float> %989) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %997 = shufflevector <8 x half> %994, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %998 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %997, <4 x float> %991) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %999 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %995, <4 x float> %992) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1000 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %997, <4 x float> %993) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !7
  %1001 = shufflevector <4 x float> %751, <4 x float> %753, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1002 = insertvalue [2 x <8 x float>] poison, <8 x float> %1001, 0, !dbg !7
  %1003 = shufflevector <4 x float> %772, <4 x float> %773, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1004 = insertvalue [2 x <8 x float>] %1002, <8 x float> %1003, 1, !dbg !7
  %1005 = insertvalue [2 x <1 x float>] poison, <1 x float> %434, 0, !dbg !7
  %1006 = insertvalue [2 x <1 x float>] %1005, <1 x float> %447, 1, !dbg !7
  %1007 = shufflevector <4 x float> %845, <4 x float> %847, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1008 = insertvalue [2 x <8 x float>] poison, <8 x float> %1007, 0, !dbg !7
  %1009 = shufflevector <4 x float> %848, <4 x float> %849, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1010 = insertvalue [2 x <8 x float>] %1008, <8 x float> %1009, 1, !dbg !7
  %1011 = shufflevector <4 x float> %921, <4 x float> %923, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1012 = insertvalue [2 x <8 x float>] poison, <8 x float> %1011, 0, !dbg !7
  %1013 = shufflevector <4 x float> %924, <4 x float> %925, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1014 = insertvalue [2 x <8 x float>] %1012, <8 x float> %1013, 1, !dbg !7
  %1015 = shufflevector <4 x float> %996, <4 x float> %998, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1016 = insertvalue [2 x <8 x float>] poison, <8 x float> %1015, 0, !dbg !7
  %1017 = shufflevector <4 x float> %999, <4 x float> %1000, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1018 = insertvalue [2 x <8 x float>] %1016, <8 x float> %1017, 1, !dbg !7
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  br label %88, !dbg !7

1019:                                             ; preds = %88
  %lane.sum.field184 = extractvalue [2 x <1 x float>] %91, 1, !dbg !7
  %lane.sum.local185 = extractelement <1 x float> %lane.sum.field184, i64 0, !dbg !7
  %lane.sum.lane.lo186 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !7
  %lane.sum.lane187 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %lane.sum.lane.lo186), !dbg !7
  %1020 = shl nuw nsw i32 %lane.sum.lane187, 2, !dbg !7
  %1021 = xor i32 %1020, 64, !dbg !7
  %1022 = bitcast float %lane.sum.local185 to i32, !dbg !7
  %lane.sum.shuffle188 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1021, i32 %1022), !dbg !7
  %1023 = bitcast i32 %lane.sum.shuffle188 to float, !dbg !7
  %lane.sum.final189 = fadd float %lane.sum.local185, %1023, !dbg !7
  %1024 = xor i32 %1020, 128, !dbg !7
  %1025 = bitcast float %lane.sum.final189 to i32, !dbg !7
  %lane.sum.shuffle190 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1024, i32 %1025), !dbg !7
  %1026 = bitcast i32 %lane.sum.shuffle190 to float, !dbg !7
  %lane.sum.final191 = fadd float %lane.sum.final189, %1026, !dbg !7
  %1027 = insertelement <1 x float> poison, float %lane.sum.final191, i64 0, !dbg !7
  %lane.sum.field = extractvalue [2 x <1 x float>] %91, 0, !dbg !7
  %lane.sum.local = extractelement <1 x float> %lane.sum.field, i64 0, !dbg !7
  %1028 = bitcast float %lane.sum.local to i32, !dbg !7
  %lane.sum.shuffle = call i32 @llvm.amdgcn.ds.bpermute(i32 %1021, i32 %1028), !dbg !7
  %1029 = bitcast i32 %lane.sum.shuffle to float, !dbg !7
  %lane.sum.final = fadd float %lane.sum.local, %1029, !dbg !7
  %1030 = bitcast float %lane.sum.final to i32, !dbg !7
  %lane.sum.shuffle182 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1024, i32 %1030), !dbg !7
  %1031 = bitcast i32 %lane.sum.shuffle182 to float, !dbg !7
  %lane.sum.final183 = fadd float %lane.sum.final, %1031, !dbg !7
  %1032 = insertelement <1 x float> poison, float %lane.sum.final183, i64 0, !dbg !7
  %1033 = and i32 %kdma.tid, 16, !dbg !7
  %.not = icmp eq i32 %1033, 0, !dbg !7
  %1034 = and i32 %kdma.tid, 32, !dbg !7
  %.not36 = icmp eq i32 %1034, 0, !dbg !7
  %1035 = extractvalue [2 x <8 x float>] %95, 0, !dbg !7
  %1036 = extractelement <8 x float> %1035, i64 0, !dbg !7
  %1037 = extractelement <8 x float> %1035, i64 1, !dbg !7
  %1038 = extractelement <8 x float> %1035, i64 2, !dbg !7
  %1039 = extractelement <8 x float> %1035, i64 3, !dbg !7
  %1040 = select i1 %.not, float %1037, float %1036, !dbg !7
  %1041 = add nuw nsw i32 %lane.sum.lane187, 64, !dbg !7
  %1042 = xor i32 %lane.sum.lane187, 16, !dbg !7
  %1043 = and i32 %1041, 192, !dbg !7
  %1044 = icmp samesign ult i32 %1042, %1043, !dbg !7
  %1045 = select i1 %1044, i32 %1042, i32 %lane.sum.lane187, !dbg !7
  %1046 = shl nuw nsw i32 %1045, 2, !dbg !7
  %1047 = bitcast float %1040 to i32, !dbg !7
  %1048 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1046, i32 %1047), !dbg !7
  %1049 = bitcast i32 %1048 to float, !dbg !7
  %1050 = select i1 %.not, float %1036, float %1049, !dbg !7
  %1051 = select i1 %.not, float %1049, float %1037, !dbg !7
  %1052 = select i1 %.not, float %1039, float %1038, !dbg !7
  %1053 = bitcast float %1052 to i32, !dbg !7
  %1054 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1046, i32 %1053), !dbg !7
  %1055 = bitcast i32 %1054 to float, !dbg !7
  %1056 = select i1 %.not, float %1038, float %1055, !dbg !7
  %1057 = select i1 %.not, float %1055, float %1039, !dbg !7
  %1058 = select i1 %.not36, float %1056, float %1050, !dbg !7
  %1059 = xor i32 %lane.sum.lane187, 32, !dbg !7
  %1060 = icmp samesign ult i32 %1059, %1043, !dbg !7
  %1061 = select i1 %1060, i32 %1059, i32 %lane.sum.lane187, !dbg !7
  %1062 = shl nuw nsw i32 %1061, 2, !dbg !7
  %1063 = bitcast float %1058 to i32, !dbg !7
  %1064 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1062, i32 %1063), !dbg !7
  %1065 = bitcast i32 %1064 to float, !dbg !7
  %1066 = select i1 %.not36, float %1050, float %1065, !dbg !7
  %1067 = select i1 %.not36, float %1065, float %1056, !dbg !7
  %1068 = select i1 %.not36, float %1057, float %1051, !dbg !7
  %1069 = bitcast float %1068 to i32, !dbg !7
  %1070 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1062, i32 %1069), !dbg !7
  %1071 = bitcast i32 %1070 to float, !dbg !7
  %1072 = select i1 %.not36, float %1051, float %1071, !dbg !7
  %1073 = select i1 %.not36, float %1071, float %1057, !dbg !7
  %1074 = extractvalue [2 x <8 x float>] %94, 0, !dbg !7
  %1075 = extractelement <8 x float> %1074, i64 0, !dbg !7
  %1076 = extractelement <8 x float> %1074, i64 1, !dbg !7
  %1077 = extractelement <8 x float> %1074, i64 2, !dbg !7
  %1078 = extractelement <8 x float> %1074, i64 3, !dbg !7
  %1079 = select i1 %.not, float %1076, float %1075, !dbg !7
  %1080 = bitcast float %1079 to i32, !dbg !7
  %1081 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1046, i32 %1080), !dbg !7
  %1082 = bitcast i32 %1081 to float, !dbg !7
  %1083 = select i1 %.not, float %1075, float %1082, !dbg !7
  %1084 = select i1 %.not, float %1082, float %1076, !dbg !7
  %1085 = select i1 %.not, float %1078, float %1077, !dbg !7
  %1086 = bitcast float %1085 to i32, !dbg !7
  %1087 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1046, i32 %1086), !dbg !7
  %1088 = bitcast i32 %1087 to float, !dbg !7
  %1089 = select i1 %.not, float %1077, float %1088, !dbg !7
  %1090 = select i1 %.not, float %1088, float %1078, !dbg !7
  %1091 = select i1 %.not36, float %1089, float %1083, !dbg !7
  %1092 = bitcast float %1091 to i32, !dbg !7
  %1093 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1062, i32 %1092), !dbg !7
  %1094 = bitcast i32 %1093 to float, !dbg !7
  %1095 = select i1 %.not36, float %1083, float %1094, !dbg !7
  %1096 = select i1 %.not36, float %1094, float %1089, !dbg !7
  %1097 = select i1 %.not36, float %1090, float %1084, !dbg !7
  %1098 = bitcast float %1097 to i32, !dbg !7
  %1099 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1062, i32 %1098), !dbg !7
  %1100 = bitcast i32 %1099 to float, !dbg !7
  %1101 = select i1 %.not36, float %1084, float %1100, !dbg !7
  %1102 = select i1 %.not36, float %1100, float %1090, !dbg !7
  %1103 = extractvalue [2 x <8 x float>] %93, 0, !dbg !7
  %1104 = extractelement <8 x float> %1103, i64 0, !dbg !7
  %1105 = extractelement <8 x float> %1103, i64 1, !dbg !7
  %1106 = extractelement <8 x float> %1103, i64 2, !dbg !7
  %1107 = extractelement <8 x float> %1103, i64 3, !dbg !7
  %1108 = select i1 %.not, float %1105, float %1104, !dbg !7
  %1109 = bitcast float %1108 to i32, !dbg !7
  %1110 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1046, i32 %1109), !dbg !7
  %1111 = bitcast i32 %1110 to float, !dbg !7
  %1112 = select i1 %.not, float %1104, float %1111, !dbg !7
  %1113 = select i1 %.not, float %1111, float %1105, !dbg !7
  %1114 = select i1 %.not, float %1107, float %1106, !dbg !7
  %1115 = bitcast float %1114 to i32, !dbg !7
  %1116 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1046, i32 %1115), !dbg !7
  %1117 = bitcast i32 %1116 to float, !dbg !7
  %1118 = select i1 %.not, float %1106, float %1117, !dbg !7
  %1119 = select i1 %.not, float %1117, float %1107, !dbg !7
  %1120 = select i1 %.not36, float %1118, float %1112, !dbg !7
  %1121 = bitcast float %1120 to i32, !dbg !7
  %1122 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1062, i32 %1121), !dbg !7
  %1123 = bitcast i32 %1122 to float, !dbg !7
  %1124 = select i1 %.not36, float %1112, float %1123, !dbg !7
  %1125 = select i1 %.not36, float %1123, float %1118, !dbg !7
  %1126 = select i1 %.not36, float %1119, float %1113, !dbg !7
  %1127 = bitcast float %1126 to i32, !dbg !7
  %1128 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1062, i32 %1127), !dbg !7
  %1129 = bitcast i32 %1128 to float, !dbg !7
  %1130 = select i1 %.not36, float %1113, float %1129, !dbg !7
  %1131 = select i1 %.not36, float %1129, float %1119, !dbg !7
  %1132 = extractvalue [2 x <8 x float>] %90, 0, !dbg !7
  %1133 = extractelement <8 x float> %1132, i64 0, !dbg !7
  %1134 = extractelement <8 x float> %1132, i64 1, !dbg !7
  %1135 = extractelement <8 x float> %1132, i64 2, !dbg !7
  %1136 = extractelement <8 x float> %1132, i64 3, !dbg !7
  %1137 = select i1 %.not, float %1134, float %1133, !dbg !7
  %1138 = bitcast float %1137 to i32, !dbg !7
  %1139 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1046, i32 %1138), !dbg !7
  %1140 = bitcast i32 %1139 to float, !dbg !7
  %1141 = select i1 %.not, float %1133, float %1140, !dbg !7
  %1142 = select i1 %.not, float %1140, float %1134, !dbg !7
  %1143 = select i1 %.not, float %1136, float %1135, !dbg !7
  %1144 = bitcast float %1143 to i32, !dbg !7
  %1145 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1046, i32 %1144), !dbg !7
  %1146 = bitcast i32 %1145 to float, !dbg !7
  %1147 = select i1 %.not, float %1135, float %1146, !dbg !7
  %1148 = select i1 %.not, float %1146, float %1136, !dbg !7
  %1149 = select i1 %.not36, float %1147, float %1141, !dbg !7
  %1150 = bitcast float %1149 to i32, !dbg !7
  %1151 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1062, i32 %1150), !dbg !7
  %1152 = select i1 %.not36, float %1148, float %1142, !dbg !7
  %1153 = bitcast float %1152 to i32, !dbg !7
  %1154 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1062, i32 %1153), !dbg !7
  %1155 = shufflevector <1 x float> %1032, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !16
  %1156 = extractelement <8 x float> %1035, i64 4, !dbg !7
  %1157 = extractelement <8 x float> %1035, i64 5, !dbg !7
  %1158 = extractelement <8 x float> %1035, i64 6, !dbg !7
  %1159 = extractelement <8 x float> %1035, i64 7, !dbg !7
  %1160 = select i1 %.not, float %1157, float %1156, !dbg !7
  %1161 = bitcast float %1160 to i32, !dbg !7
  %1162 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1046, i32 %1161), !dbg !7
  %1163 = bitcast i32 %1162 to float, !dbg !7
  %1164 = select i1 %.not, float %1156, float %1163, !dbg !7
  %1165 = select i1 %.not, float %1163, float %1157, !dbg !7
  %1166 = select i1 %.not, float %1159, float %1158, !dbg !7
  %1167 = bitcast float %1166 to i32, !dbg !7
  %1168 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1046, i32 %1167), !dbg !7
  %1169 = bitcast i32 %1168 to float, !dbg !7
  %1170 = select i1 %.not, float %1158, float %1169, !dbg !7
  %1171 = select i1 %.not, float %1169, float %1159, !dbg !7
  %1172 = select i1 %.not36, float %1170, float %1164, !dbg !7
  %1173 = bitcast float %1172 to i32, !dbg !7
  %1174 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1062, i32 %1173), !dbg !7
  %1175 = bitcast i32 %1174 to float, !dbg !7
  %1176 = select i1 %.not36, float %1164, float %1175, !dbg !7
  %1177 = select i1 %.not36, float %1175, float %1170, !dbg !7
  %1178 = select i1 %.not36, float %1171, float %1165, !dbg !7
  %1179 = bitcast float %1178 to i32, !dbg !7
  %1180 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1062, i32 %1179), !dbg !7
  %1181 = bitcast i32 %1180 to float, !dbg !7
  %1182 = select i1 %.not36, float %1165, float %1181, !dbg !7
  %1183 = select i1 %.not36, float %1181, float %1171, !dbg !7
  %1184 = extractelement <8 x float> %1074, i64 4, !dbg !7
  %1185 = extractelement <8 x float> %1074, i64 5, !dbg !7
  %1186 = extractelement <8 x float> %1074, i64 6, !dbg !7
  %1187 = extractelement <8 x float> %1074, i64 7, !dbg !7
  %1188 = select i1 %.not, float %1185, float %1184, !dbg !7
  %1189 = bitcast float %1188 to i32, !dbg !7
  %1190 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1046, i32 %1189), !dbg !7
  %1191 = bitcast i32 %1190 to float, !dbg !7
  %1192 = select i1 %.not, float %1184, float %1191, !dbg !7
  %1193 = select i1 %.not, float %1191, float %1185, !dbg !7
  %1194 = select i1 %.not, float %1187, float %1186, !dbg !7
  %1195 = bitcast float %1194 to i32, !dbg !7
  %1196 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1046, i32 %1195), !dbg !7
  %1197 = bitcast i32 %1196 to float, !dbg !7
  %1198 = select i1 %.not, float %1186, float %1197, !dbg !7
  %1199 = select i1 %.not, float %1197, float %1187, !dbg !7
  %1200 = select i1 %.not36, float %1198, float %1192, !dbg !7
  %1201 = bitcast float %1200 to i32, !dbg !7
  %1202 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1062, i32 %1201), !dbg !7
  %1203 = bitcast i32 %1202 to float, !dbg !7
  %1204 = select i1 %.not36, float %1192, float %1203, !dbg !7
  %1205 = select i1 %.not36, float %1203, float %1198, !dbg !7
  %1206 = select i1 %.not36, float %1199, float %1193, !dbg !7
  %1207 = bitcast float %1206 to i32, !dbg !7
  %1208 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1062, i32 %1207), !dbg !7
  %1209 = bitcast i32 %1208 to float, !dbg !7
  %1210 = select i1 %.not36, float %1193, float %1209, !dbg !7
  %1211 = select i1 %.not36, float %1209, float %1199, !dbg !7
  %1212 = extractelement <8 x float> %1103, i64 4, !dbg !7
  %1213 = extractelement <8 x float> %1103, i64 5, !dbg !7
  %1214 = extractelement <8 x float> %1103, i64 6, !dbg !7
  %1215 = extractelement <8 x float> %1103, i64 7, !dbg !7
  %1216 = select i1 %.not, float %1213, float %1212, !dbg !7
  %1217 = bitcast float %1216 to i32, !dbg !7
  %1218 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1046, i32 %1217), !dbg !7
  %1219 = bitcast i32 %1218 to float, !dbg !7
  %1220 = select i1 %.not, float %1212, float %1219, !dbg !7
  %1221 = select i1 %.not, float %1219, float %1213, !dbg !7
  %1222 = select i1 %.not, float %1215, float %1214, !dbg !7
  %1223 = bitcast float %1222 to i32, !dbg !7
  %1224 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1046, i32 %1223), !dbg !7
  %1225 = bitcast i32 %1224 to float, !dbg !7
  %1226 = select i1 %.not, float %1214, float %1225, !dbg !7
  %1227 = select i1 %.not, float %1225, float %1215, !dbg !7
  %1228 = select i1 %.not36, float %1226, float %1220, !dbg !7
  %1229 = bitcast float %1228 to i32, !dbg !7
  %1230 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1062, i32 %1229), !dbg !7
  %1231 = bitcast i32 %1230 to float, !dbg !7
  %1232 = select i1 %.not36, float %1220, float %1231, !dbg !7
  %1233 = select i1 %.not36, float %1231, float %1226, !dbg !7
  %1234 = select i1 %.not36, float %1227, float %1221, !dbg !7
  %1235 = bitcast float %1234 to i32, !dbg !7
  %1236 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1062, i32 %1235), !dbg !7
  %1237 = bitcast i32 %1236 to float, !dbg !7
  %1238 = select i1 %.not36, float %1221, float %1237, !dbg !7
  %1239 = select i1 %.not36, float %1237, float %1227, !dbg !7
  %1240 = extractelement <8 x float> %1132, i64 4, !dbg !7
  %1241 = extractelement <8 x float> %1132, i64 5, !dbg !7
  %1242 = extractelement <8 x float> %1132, i64 6, !dbg !7
  %1243 = extractelement <8 x float> %1132, i64 7, !dbg !7
  %1244 = select i1 %.not, float %1241, float %1240, !dbg !7
  %1245 = bitcast float %1244 to i32, !dbg !7
  %1246 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1046, i32 %1245), !dbg !7
  %1247 = bitcast i32 %1246 to float, !dbg !7
  %1248 = select i1 %.not, float %1240, float %1247, !dbg !7
  %1249 = select i1 %.not, float %1247, float %1241, !dbg !7
  %1250 = select i1 %.not, float %1243, float %1242, !dbg !7
  %1251 = bitcast float %1250 to i32, !dbg !7
  %1252 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1046, i32 %1251), !dbg !7
  %1253 = bitcast i32 %1252 to float, !dbg !7
  %1254 = select i1 %.not, float %1242, float %1253, !dbg !7
  %1255 = select i1 %.not, float %1253, float %1243, !dbg !7
  %1256 = select i1 %.not36, float %1254, float %1248, !dbg !7
  %1257 = bitcast float %1256 to i32, !dbg !7
  %1258 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1062, i32 %1257), !dbg !7
  %1259 = select i1 %.not36, float %1255, float %1249, !dbg !7
  %1260 = bitcast float %1259 to i32, !dbg !7
  %1261 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1062, i32 %1260), !dbg !7
  %1262 = extractvalue [2 x <8 x float>] %95, 1, !dbg !7
  %1263 = extractelement <8 x float> %1262, i64 0, !dbg !7
  %1264 = extractelement <8 x float> %1262, i64 1, !dbg !7
  %1265 = extractelement <8 x float> %1262, i64 2, !dbg !7
  %1266 = extractelement <8 x float> %1262, i64 3, !dbg !7
  %1267 = select i1 %.not, float %1264, float %1263, !dbg !7
  %1268 = bitcast float %1267 to i32, !dbg !7
  %1269 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1046, i32 %1268), !dbg !7
  %1270 = bitcast i32 %1269 to float, !dbg !7
  %1271 = select i1 %.not, float %1263, float %1270, !dbg !7
  %1272 = select i1 %.not, float %1270, float %1264, !dbg !7
  %1273 = select i1 %.not, float %1266, float %1265, !dbg !7
  %1274 = bitcast float %1273 to i32, !dbg !7
  %1275 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1046, i32 %1274), !dbg !7
  %1276 = bitcast i32 %1275 to float, !dbg !7
  %1277 = select i1 %.not, float %1265, float %1276, !dbg !7
  %1278 = select i1 %.not, float %1276, float %1266, !dbg !7
  %1279 = select i1 %.not36, float %1277, float %1271, !dbg !7
  %1280 = bitcast float %1279 to i32, !dbg !7
  %1281 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1062, i32 %1280), !dbg !7
  %1282 = bitcast i32 %1281 to float, !dbg !7
  %1283 = select i1 %.not36, float %1271, float %1282, !dbg !7
  %1284 = select i1 %.not36, float %1282, float %1277, !dbg !7
  %1285 = select i1 %.not36, float %1278, float %1272, !dbg !7
  %1286 = bitcast float %1285 to i32, !dbg !7
  %1287 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1062, i32 %1286), !dbg !7
  %1288 = bitcast i32 %1287 to float, !dbg !7
  %1289 = select i1 %.not36, float %1272, float %1288, !dbg !7
  %1290 = select i1 %.not36, float %1288, float %1278, !dbg !7
  %1291 = extractvalue [2 x <8 x float>] %94, 1, !dbg !7
  %1292 = extractelement <8 x float> %1291, i64 0, !dbg !7
  %1293 = extractelement <8 x float> %1291, i64 1, !dbg !7
  %1294 = extractelement <8 x float> %1291, i64 2, !dbg !7
  %1295 = extractelement <8 x float> %1291, i64 3, !dbg !7
  %1296 = select i1 %.not, float %1293, float %1292, !dbg !7
  %1297 = bitcast float %1296 to i32, !dbg !7
  %1298 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1046, i32 %1297), !dbg !7
  %1299 = bitcast i32 %1298 to float, !dbg !7
  %1300 = select i1 %.not, float %1292, float %1299, !dbg !7
  %1301 = select i1 %.not, float %1299, float %1293, !dbg !7
  %1302 = select i1 %.not, float %1295, float %1294, !dbg !7
  %1303 = bitcast float %1302 to i32, !dbg !7
  %1304 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1046, i32 %1303), !dbg !7
  %1305 = bitcast i32 %1304 to float, !dbg !7
  %1306 = select i1 %.not, float %1294, float %1305, !dbg !7
  %1307 = select i1 %.not, float %1305, float %1295, !dbg !7
  %1308 = select i1 %.not36, float %1306, float %1300, !dbg !7
  %1309 = bitcast float %1308 to i32, !dbg !7
  %1310 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1062, i32 %1309), !dbg !7
  %1311 = bitcast i32 %1310 to float, !dbg !7
  %1312 = select i1 %.not36, float %1300, float %1311, !dbg !7
  %1313 = select i1 %.not36, float %1311, float %1306, !dbg !7
  %1314 = select i1 %.not36, float %1307, float %1301, !dbg !7
  %1315 = bitcast float %1314 to i32, !dbg !7
  %1316 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1062, i32 %1315), !dbg !7
  %1317 = bitcast i32 %1316 to float, !dbg !7
  %1318 = select i1 %.not36, float %1301, float %1317, !dbg !7
  %1319 = select i1 %.not36, float %1317, float %1307, !dbg !7
  %1320 = extractvalue [2 x <8 x float>] %93, 1, !dbg !7
  %1321 = extractelement <8 x float> %1320, i64 0, !dbg !7
  %1322 = extractelement <8 x float> %1320, i64 1, !dbg !7
  %1323 = extractelement <8 x float> %1320, i64 2, !dbg !7
  %1324 = extractelement <8 x float> %1320, i64 3, !dbg !7
  %1325 = select i1 %.not, float %1322, float %1321, !dbg !7
  %1326 = bitcast float %1325 to i32, !dbg !7
  %1327 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1046, i32 %1326), !dbg !7
  %1328 = bitcast i32 %1327 to float, !dbg !7
  %1329 = select i1 %.not, float %1321, float %1328, !dbg !7
  %1330 = select i1 %.not, float %1328, float %1322, !dbg !7
  %1331 = select i1 %.not, float %1324, float %1323, !dbg !7
  %1332 = bitcast float %1331 to i32, !dbg !7
  %1333 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1046, i32 %1332), !dbg !7
  %1334 = bitcast i32 %1333 to float, !dbg !7
  %1335 = select i1 %.not, float %1323, float %1334, !dbg !7
  %1336 = select i1 %.not, float %1334, float %1324, !dbg !7
  %1337 = select i1 %.not36, float %1335, float %1329, !dbg !7
  %1338 = bitcast float %1337 to i32, !dbg !7
  %1339 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1062, i32 %1338), !dbg !7
  %1340 = bitcast i32 %1339 to float, !dbg !7
  %1341 = select i1 %.not36, float %1329, float %1340, !dbg !7
  %1342 = select i1 %.not36, float %1340, float %1335, !dbg !7
  %1343 = select i1 %.not36, float %1336, float %1330, !dbg !7
  %1344 = bitcast float %1343 to i32, !dbg !7
  %1345 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1062, i32 %1344), !dbg !7
  %1346 = bitcast i32 %1345 to float, !dbg !7
  %1347 = select i1 %.not36, float %1330, float %1346, !dbg !7
  %1348 = select i1 %.not36, float %1346, float %1336, !dbg !7
  %1349 = extractvalue [2 x <8 x float>] %90, 1, !dbg !7
  %1350 = extractelement <8 x float> %1349, i64 0, !dbg !7
  %1351 = extractelement <8 x float> %1349, i64 1, !dbg !7
  %1352 = extractelement <8 x float> %1349, i64 2, !dbg !7
  %1353 = extractelement <8 x float> %1349, i64 3, !dbg !7
  %1354 = select i1 %.not, float %1351, float %1350, !dbg !7
  %1355 = bitcast float %1354 to i32, !dbg !7
  %1356 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1046, i32 %1355), !dbg !7
  %1357 = bitcast i32 %1356 to float, !dbg !7
  %1358 = select i1 %.not, float %1350, float %1357, !dbg !7
  %1359 = select i1 %.not, float %1357, float %1351, !dbg !7
  %1360 = select i1 %.not, float %1353, float %1352, !dbg !7
  %1361 = bitcast float %1360 to i32, !dbg !7
  %1362 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1046, i32 %1361), !dbg !7
  %1363 = bitcast i32 %1362 to float, !dbg !7
  %1364 = select i1 %.not, float %1352, float %1363, !dbg !7
  %1365 = select i1 %.not, float %1363, float %1353, !dbg !7
  %1366 = select i1 %.not36, float %1364, float %1358, !dbg !7
  %1367 = bitcast float %1366 to i32, !dbg !7
  %1368 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1062, i32 %1367), !dbg !7
  %1369 = select i1 %.not36, float %1365, float %1359, !dbg !7
  %1370 = bitcast float %1369 to i32, !dbg !7
  %1371 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1062, i32 %1370), !dbg !7
  %1372 = shufflevector <1 x float> %1027, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !16
  %1373 = extractelement <8 x float> %1262, i64 4, !dbg !7
  %1374 = extractelement <8 x float> %1262, i64 5, !dbg !7
  %1375 = extractelement <8 x float> %1262, i64 6, !dbg !7
  %1376 = extractelement <8 x float> %1262, i64 7, !dbg !7
  %1377 = select i1 %.not, float %1374, float %1373, !dbg !7
  %1378 = bitcast float %1377 to i32, !dbg !7
  %1379 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1046, i32 %1378), !dbg !7
  %1380 = bitcast i32 %1379 to float, !dbg !7
  %1381 = select i1 %.not, float %1373, float %1380, !dbg !7
  %1382 = select i1 %.not, float %1380, float %1374, !dbg !7
  %1383 = select i1 %.not, float %1376, float %1375, !dbg !7
  %1384 = bitcast float %1383 to i32, !dbg !7
  %1385 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1046, i32 %1384), !dbg !7
  %1386 = bitcast i32 %1385 to float, !dbg !7
  %1387 = select i1 %.not, float %1375, float %1386, !dbg !7
  %1388 = select i1 %.not, float %1386, float %1376, !dbg !7
  %1389 = select i1 %.not36, float %1387, float %1381, !dbg !7
  %1390 = bitcast float %1389 to i32, !dbg !7
  %1391 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1062, i32 %1390), !dbg !7
  %1392 = bitcast i32 %1391 to float, !dbg !7
  %1393 = select i1 %.not36, float %1381, float %1392, !dbg !7
  %1394 = select i1 %.not36, float %1392, float %1387, !dbg !7
  %1395 = select i1 %.not36, float %1388, float %1382, !dbg !7
  %1396 = bitcast float %1395 to i32, !dbg !7
  %1397 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1062, i32 %1396), !dbg !7
  %1398 = bitcast i32 %1397 to float, !dbg !7
  %1399 = select i1 %.not36, float %1382, float %1398, !dbg !7
  %1400 = select i1 %.not36, float %1398, float %1388, !dbg !7
  %1401 = extractelement <8 x float> %1291, i64 4, !dbg !7
  %1402 = extractelement <8 x float> %1291, i64 5, !dbg !7
  %1403 = extractelement <8 x float> %1291, i64 6, !dbg !7
  %1404 = extractelement <8 x float> %1291, i64 7, !dbg !7
  %1405 = select i1 %.not, float %1402, float %1401, !dbg !7
  %1406 = bitcast float %1405 to i32, !dbg !7
  %1407 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1046, i32 %1406), !dbg !7
  %1408 = bitcast i32 %1407 to float, !dbg !7
  %1409 = select i1 %.not, float %1401, float %1408, !dbg !7
  %1410 = select i1 %.not, float %1408, float %1402, !dbg !7
  %1411 = select i1 %.not, float %1404, float %1403, !dbg !7
  %1412 = bitcast float %1411 to i32, !dbg !7
  %1413 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1046, i32 %1412), !dbg !7
  %1414 = bitcast i32 %1413 to float, !dbg !7
  %1415 = select i1 %.not, float %1403, float %1414, !dbg !7
  %1416 = select i1 %.not, float %1414, float %1404, !dbg !7
  %1417 = select i1 %.not36, float %1415, float %1409, !dbg !7
  %1418 = bitcast float %1417 to i32, !dbg !7
  %1419 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1062, i32 %1418), !dbg !7
  %1420 = bitcast i32 %1419 to float, !dbg !7
  %1421 = select i1 %.not36, float %1409, float %1420, !dbg !7
  %1422 = select i1 %.not36, float %1420, float %1415, !dbg !7
  %1423 = select i1 %.not36, float %1416, float %1410, !dbg !7
  %1424 = bitcast float %1423 to i32, !dbg !7
  %1425 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1062, i32 %1424), !dbg !7
  %1426 = bitcast i32 %1425 to float, !dbg !7
  %1427 = select i1 %.not36, float %1410, float %1426, !dbg !7
  %1428 = select i1 %.not36, float %1426, float %1416, !dbg !7
  %1429 = extractelement <8 x float> %1320, i64 4, !dbg !7
  %1430 = extractelement <8 x float> %1320, i64 5, !dbg !7
  %1431 = extractelement <8 x float> %1320, i64 6, !dbg !7
  %1432 = extractelement <8 x float> %1320, i64 7, !dbg !7
  %1433 = select i1 %.not, float %1430, float %1429, !dbg !7
  %1434 = bitcast float %1433 to i32, !dbg !7
  %1435 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1046, i32 %1434), !dbg !7
  %1436 = bitcast i32 %1435 to float, !dbg !7
  %1437 = select i1 %.not, float %1429, float %1436, !dbg !7
  %1438 = select i1 %.not, float %1436, float %1430, !dbg !7
  %1439 = select i1 %.not, float %1432, float %1431, !dbg !7
  %1440 = bitcast float %1439 to i32, !dbg !7
  %1441 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1046, i32 %1440), !dbg !7
  %1442 = bitcast i32 %1441 to float, !dbg !7
  %1443 = select i1 %.not, float %1431, float %1442, !dbg !7
  %1444 = select i1 %.not, float %1442, float %1432, !dbg !7
  %1445 = select i1 %.not36, float %1443, float %1437, !dbg !7
  %1446 = bitcast float %1445 to i32, !dbg !7
  %1447 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1062, i32 %1446), !dbg !7
  %1448 = bitcast i32 %1447 to float, !dbg !7
  %1449 = select i1 %.not36, float %1437, float %1448, !dbg !7
  %1450 = select i1 %.not36, float %1448, float %1443, !dbg !7
  %1451 = select i1 %.not36, float %1444, float %1438, !dbg !7
  %1452 = bitcast float %1451 to i32, !dbg !7
  %1453 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1062, i32 %1452), !dbg !7
  %1454 = bitcast i32 %1453 to float, !dbg !7
  %1455 = select i1 %.not36, float %1438, float %1454, !dbg !7
  %1456 = select i1 %.not36, float %1454, float %1444, !dbg !7
  %1457 = extractelement <8 x float> %1349, i64 4, !dbg !7
  %1458 = extractelement <8 x float> %1349, i64 5, !dbg !7
  %1459 = extractelement <8 x float> %1349, i64 6, !dbg !7
  %1460 = extractelement <8 x float> %1349, i64 7, !dbg !7
  %1461 = select i1 %.not, float %1458, float %1457, !dbg !7
  %1462 = bitcast float %1461 to i32, !dbg !7
  %1463 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1046, i32 %1462), !dbg !7
  %1464 = bitcast i32 %1463 to float, !dbg !7
  %1465 = select i1 %.not, float %1457, float %1464, !dbg !7
  %1466 = select i1 %.not, float %1464, float %1458, !dbg !7
  %1467 = select i1 %.not, float %1460, float %1459, !dbg !7
  %1468 = bitcast float %1467 to i32, !dbg !7
  %1469 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1046, i32 %1468), !dbg !7
  %1470 = bitcast i32 %1469 to float, !dbg !7
  %1471 = select i1 %.not, float %1459, float %1470, !dbg !7
  %1472 = select i1 %.not, float %1470, float %1460, !dbg !7
  %1473 = select i1 %.not36, float %1471, float %1465, !dbg !7
  %1474 = bitcast float %1473 to i32, !dbg !7
  %1475 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1062, i32 %1474), !dbg !7
  %1476 = select i1 %.not36, float %1472, float %1466, !dbg !7
  %1477 = bitcast float %1476 to i32, !dbg !7
  %1478 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1062, i32 %1477), !dbg !7
  %1479 = bitcast i32 %1475 to float, !dbg !7
  %1480 = select i1 %.not36, float %1465, float %1479, !dbg !7
  %1481 = insertelement <4 x float> poison, float %1480, i64 0, !dbg !16
  %1482 = bitcast i32 %1478 to float, !dbg !7
  %1483 = select i1 %.not36, float %1466, float %1482, !dbg !7
  %1484 = insertelement <4 x float> %1481, float %1483, i64 1, !dbg !16
  %1485 = select i1 %.not36, float %1479, float %1471, !dbg !7
  %1486 = insertelement <4 x float> %1484, float %1485, i64 2, !dbg !16
  %1487 = select i1 %.not36, float %1482, float %1472, !dbg !7
  %1488 = insertelement <4 x float> %1486, float %1487, i64 3, !dbg !16
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1372, !dbg !16
  %normalized = fmul <4 x float> %1488, %row.reciprocal, !dbg !16
  %1489 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !17
  %1490 = extractelement <4 x half> %1489, i64 3, !dbg !17
  %1491 = extractelement <4 x half> %1489, i64 2, !dbg !17
  %1492 = extractelement <4 x half> %1489, i64 1, !dbg !17
  %1493 = extractelement <4 x half> %1489, i64 0, !dbg !17
  %1494 = bitcast i32 %1368 to float, !dbg !7
  %1495 = select i1 %.not36, float %1358, float %1494, !dbg !7
  %1496 = insertelement <4 x float> poison, float %1495, i64 0, !dbg !16
  %1497 = bitcast i32 %1371 to float, !dbg !7
  %1498 = select i1 %.not36, float %1359, float %1497, !dbg !7
  %1499 = insertelement <4 x float> %1496, float %1498, i64 1, !dbg !16
  %1500 = select i1 %.not36, float %1494, float %1364, !dbg !7
  %1501 = insertelement <4 x float> %1499, float %1500, i64 2, !dbg !16
  %1502 = select i1 %.not36, float %1497, float %1365, !dbg !7
  %1503 = insertelement <4 x float> %1501, float %1502, i64 3, !dbg !16
  %normalized37 = fmul <4 x float> %1503, %row.reciprocal, !dbg !16
  %1504 = fptrunc <4 x float> %normalized37 to <4 x half>, !dbg !17
  %1505 = extractelement <4 x half> %1504, i64 3, !dbg !17
  %1506 = extractelement <4 x half> %1504, i64 2, !dbg !17
  %1507 = extractelement <4 x half> %1504, i64 1, !dbg !17
  %1508 = extractelement <4 x half> %1504, i64 0, !dbg !17
  %1509 = bitcast i32 %1258 to float, !dbg !7
  %1510 = select i1 %.not36, float %1248, float %1509, !dbg !7
  %1511 = insertelement <4 x float> poison, float %1510, i64 0, !dbg !16
  %1512 = bitcast i32 %1261 to float, !dbg !7
  %1513 = select i1 %.not36, float %1249, float %1512, !dbg !7
  %1514 = insertelement <4 x float> %1511, float %1513, i64 1, !dbg !16
  %1515 = select i1 %.not36, float %1509, float %1254, !dbg !7
  %1516 = insertelement <4 x float> %1514, float %1515, i64 2, !dbg !16
  %1517 = select i1 %.not36, float %1512, float %1255, !dbg !7
  %1518 = insertelement <4 x float> %1516, float %1517, i64 3, !dbg !16
  %row.reciprocal38 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1155, !dbg !16
  %normalized39 = fmul <4 x float> %1518, %row.reciprocal38, !dbg !16
  %1519 = fptrunc <4 x float> %normalized39 to <4 x half>, !dbg !17
  %1520 = extractelement <4 x half> %1519, i64 3, !dbg !17
  %1521 = extractelement <4 x half> %1519, i64 2, !dbg !17
  %1522 = extractelement <4 x half> %1519, i64 1, !dbg !17
  %1523 = extractelement <4 x half> %1519, i64 0, !dbg !17
  %1524 = bitcast i32 %1151 to float, !dbg !7
  %1525 = select i1 %.not36, float %1141, float %1524, !dbg !7
  %1526 = insertelement <4 x float> poison, float %1525, i64 0, !dbg !16
  %1527 = bitcast i32 %1154 to float, !dbg !7
  %1528 = select i1 %.not36, float %1142, float %1527, !dbg !7
  %1529 = insertelement <4 x float> %1526, float %1528, i64 1, !dbg !16
  %1530 = select i1 %.not36, float %1524, float %1147, !dbg !7
  %1531 = insertelement <4 x float> %1529, float %1530, i64 2, !dbg !16
  %1532 = select i1 %.not36, float %1527, float %1148, !dbg !7
  %1533 = insertelement <4 x float> %1531, float %1532, i64 3, !dbg !16
  %normalized40 = fmul <4 x float> %1533, %row.reciprocal38, !dbg !16
  %1534 = fptrunc <4 x float> %normalized40 to <4 x half>, !dbg !17
  %1535 = extractelement <4 x half> %1534, i64 3, !dbg !17
  %1536 = extractelement <4 x half> %1534, i64 2, !dbg !17
  %1537 = extractelement <4 x half> %1534, i64 1, !dbg !17
  %1538 = extractelement <4 x half> %1534, i64 0, !dbg !17
  %address.thread974 = call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #9, !dbg !18, !range !32, !frisk.register_lifetimes !1
  %address.local975 = zext nneg i32 %address.thread974 to i64, !dbg !8
  %address.local976 = add nuw i64 %19, %address.local975, !dbg !18
  %1539 = lshr i64 %address.local975, 2, !dbg !18
  %address.local978 = and i64 %1539, 48, !dbg !18
  %address.local979 = add i64 %address.local976, %address.local978, !dbg !18
  %address.local980 = and i64 %address.local975, 240, !dbg !18
  %address.local981 = sub i64 %address.local979, %address.local980, !dbg !18
  %address.local982 = add i64 %address.local981, %20, !dbg !18
  %address.local983 = shl nuw nsw i64 %address.local982, 7, !dbg !18
  %1540 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %23, !dbg !18
  %address.local985 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %1540, i64 %address.local983, !dbg !18
  %address.local987 = and i64 %1539, 12, !dbg !10
  %address.local988 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local987, !dbg !18
  store half %1538, ptr addrspace(1) %address.local988, align 2, !dbg !18
  %address.local990 = or disjoint i64 %address.local987, 1, !dbg !18
  %address.local991 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local990, !dbg !18
  store half %1537, ptr addrspace(1) %address.local991, align 2, !dbg !18
  %address.local993 = or disjoint i64 %address.local987, 2, !dbg !18
  %address.local994 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local993, !dbg !18
  store half %1536, ptr addrspace(1) %address.local994, align 2, !dbg !18
  %address.local996 = or disjoint i64 %address.local987, 3, !dbg !18
  %address.local997 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local996, !dbg !18
  store half %1535, ptr addrspace(1) %address.local997, align 2, !dbg !18
  %address.local999 = or disjoint i64 %address.local987, 16, !dbg !18
  %address.local1000 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local999, !dbg !18
  store half %1523, ptr addrspace(1) %address.local1000, align 2, !dbg !18
  %address.local1002 = or disjoint i64 %address.local987, 17, !dbg !18
  %address.local1003 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1002, !dbg !18
  store half %1522, ptr addrspace(1) %address.local1003, align 2, !dbg !18
  %address.local1005 = or disjoint i64 %address.local987, 18, !dbg !18
  %address.local1006 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1005, !dbg !18
  store half %1521, ptr addrspace(1) %address.local1006, align 2, !dbg !18
  %address.local1008 = or disjoint i64 %address.local987, 19, !dbg !18
  %address.local1009 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1008, !dbg !18
  store half %1520, ptr addrspace(1) %address.local1009, align 2, !dbg !18
  %address.local1011 = add nuw i64 %address.local983, 8192, !dbg !18
  %address.local1013 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %1540, i64 %address.local1011, !dbg !18
  %address.local1014 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local987, !dbg !18
  store half %1508, ptr addrspace(1) %address.local1014, align 2, !dbg !18
  %address.local1016 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local990, !dbg !18
  store half %1507, ptr addrspace(1) %address.local1016, align 2, !dbg !18
  %address.local1018 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local993, !dbg !18
  store half %1506, ptr addrspace(1) %address.local1018, align 2, !dbg !18
  %address.local1020 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local996, !dbg !18
  store half %1505, ptr addrspace(1) %address.local1020, align 2, !dbg !18
  %address.local1022 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local999, !dbg !18
  store half %1493, ptr addrspace(1) %address.local1022, align 2, !dbg !18
  %address.local1024 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1002, !dbg !18
  store half %1492, ptr addrspace(1) %address.local1024, align 2, !dbg !18
  %address.local1026 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1005, !dbg !18
  store half %1491, ptr addrspace(1) %address.local1026, align 2, !dbg !18
  %address.local1028 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1008, !dbg !18
  store half %1490, ptr addrspace(1) %address.local1028, align 2, !dbg !18
  %1541 = insertelement <4 x float> poison, float %1449, i64 0, !dbg !16
  %1542 = insertelement <4 x float> %1541, float %1455, i64 1, !dbg !16
  %1543 = insertelement <4 x float> %1542, float %1450, i64 2, !dbg !16
  %1544 = insertelement <4 x float> %1543, float %1456, i64 3, !dbg !16
  %normalized42 = fmul <4 x float> %1544, %row.reciprocal, !dbg !16
  %1545 = fptrunc <4 x float> %normalized42 to <4 x half>, !dbg !17
  %1546 = extractelement <4 x half> %1545, i64 3, !dbg !17
  %1547 = extractelement <4 x half> %1545, i64 2, !dbg !17
  %1548 = extractelement <4 x half> %1545, i64 1, !dbg !17
  %1549 = extractelement <4 x half> %1545, i64 0, !dbg !17
  %1550 = insertelement <4 x float> poison, float %1341, i64 0, !dbg !16
  %1551 = insertelement <4 x float> %1550, float %1347, i64 1, !dbg !16
  %1552 = insertelement <4 x float> %1551, float %1342, i64 2, !dbg !16
  %1553 = insertelement <4 x float> %1552, float %1348, i64 3, !dbg !16
  %normalized43 = fmul <4 x float> %1553, %row.reciprocal, !dbg !16
  %1554 = fptrunc <4 x float> %normalized43 to <4 x half>, !dbg !17
  %1555 = extractelement <4 x half> %1554, i64 3, !dbg !17
  %1556 = extractelement <4 x half> %1554, i64 2, !dbg !17
  %1557 = extractelement <4 x half> %1554, i64 1, !dbg !17
  %1558 = extractelement <4 x half> %1554, i64 0, !dbg !17
  %1559 = insertelement <4 x float> poison, float %1232, i64 0, !dbg !16
  %1560 = insertelement <4 x float> %1559, float %1238, i64 1, !dbg !16
  %1561 = insertelement <4 x float> %1560, float %1233, i64 2, !dbg !16
  %1562 = insertelement <4 x float> %1561, float %1239, i64 3, !dbg !16
  %normalized45 = fmul <4 x float> %1562, %row.reciprocal38, !dbg !16
  %1563 = fptrunc <4 x float> %normalized45 to <4 x half>, !dbg !17
  %1564 = extractelement <4 x half> %1563, i64 3, !dbg !17
  %1565 = extractelement <4 x half> %1563, i64 2, !dbg !17
  %1566 = extractelement <4 x half> %1563, i64 1, !dbg !17
  %1567 = extractelement <4 x half> %1563, i64 0, !dbg !17
  %1568 = insertelement <4 x float> poison, float %1124, i64 0, !dbg !16
  %1569 = insertelement <4 x float> %1568, float %1130, i64 1, !dbg !16
  %1570 = insertelement <4 x float> %1569, float %1125, i64 2, !dbg !16
  %1571 = insertelement <4 x float> %1570, float %1131, i64 3, !dbg !16
  %normalized46 = fmul <4 x float> %1571, %row.reciprocal38, !dbg !16
  %1572 = fptrunc <4 x float> %normalized46 to <4 x half>, !dbg !17
  %1573 = extractelement <4 x half> %1572, i64 3, !dbg !17
  %1574 = extractelement <4 x half> %1572, i64 2, !dbg !17
  %1575 = extractelement <4 x half> %1572, i64 1, !dbg !17
  %1576 = extractelement <4 x half> %1572, i64 0, !dbg !17
  %address.local1043 = or disjoint i64 %address.local987, 32, !dbg !18
  %address.local1044 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1043, !dbg !18
  store half %1576, ptr addrspace(1) %address.local1044, align 2, !dbg !18
  %address.local1046 = or disjoint i64 %address.local987, 33, !dbg !18
  %address.local1047 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1046, !dbg !18
  store half %1575, ptr addrspace(1) %address.local1047, align 2, !dbg !18
  %address.local1049 = or disjoint i64 %address.local987, 34, !dbg !18
  %address.local1050 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1049, !dbg !18
  store half %1574, ptr addrspace(1) %address.local1050, align 2, !dbg !18
  %address.local1052 = or disjoint i64 %address.local987, 35, !dbg !18
  %address.local1053 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1052, !dbg !18
  store half %1573, ptr addrspace(1) %address.local1053, align 2, !dbg !18
  %address.local1055 = or disjoint i64 %address.local987, 48, !dbg !18
  %address.local1056 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1055, !dbg !18
  store half %1567, ptr addrspace(1) %address.local1056, align 2, !dbg !18
  %address.local1058 = or disjoint i64 %address.local987, 49, !dbg !18
  %address.local1059 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1058, !dbg !18
  store half %1566, ptr addrspace(1) %address.local1059, align 2, !dbg !18
  %address.local1061 = or disjoint i64 %address.local987, 50, !dbg !18
  %address.local1062 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1061, !dbg !18
  store half %1565, ptr addrspace(1) %address.local1062, align 2, !dbg !18
  %address.local1064 = or i64 %1539, 51, !dbg !18
  %address.local1065 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1064, !dbg !18
  store half %1564, ptr addrspace(1) %address.local1065, align 2, !dbg !18
  %address.local1070 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1043, !dbg !18
  store half %1558, ptr addrspace(1) %address.local1070, align 2, !dbg !18
  %address.local1072 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1046, !dbg !18
  store half %1557, ptr addrspace(1) %address.local1072, align 2, !dbg !18
  %address.local1074 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1049, !dbg !18
  store half %1556, ptr addrspace(1) %address.local1074, align 2, !dbg !18
  %address.local1076 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1052, !dbg !18
  store half %1555, ptr addrspace(1) %address.local1076, align 2, !dbg !18
  %address.local1078 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1055, !dbg !18
  store half %1549, ptr addrspace(1) %address.local1078, align 2, !dbg !18
  %address.local1080 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1058, !dbg !18
  store half %1548, ptr addrspace(1) %address.local1080, align 2, !dbg !18
  %address.local1082 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1061, !dbg !18
  store half %1547, ptr addrspace(1) %address.local1082, align 2, !dbg !18
  %address.local1084 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1064, !dbg !18
  store half %1546, ptr addrspace(1) %address.local1084, align 2, !dbg !18
  %1577 = insertelement <4 x float> poison, float %1421, i64 0, !dbg !16
  %1578 = insertelement <4 x float> %1577, float %1427, i64 1, !dbg !16
  %1579 = insertelement <4 x float> %1578, float %1422, i64 2, !dbg !16
  %1580 = insertelement <4 x float> %1579, float %1428, i64 3, !dbg !16
  %normalized48 = fmul <4 x float> %1580, %row.reciprocal, !dbg !16
  %1581 = fptrunc <4 x float> %normalized48 to <4 x half>, !dbg !17
  %1582 = extractelement <4 x half> %1581, i64 3, !dbg !17
  %1583 = extractelement <4 x half> %1581, i64 2, !dbg !17
  %1584 = extractelement <4 x half> %1581, i64 1, !dbg !17
  %1585 = extractelement <4 x half> %1581, i64 0, !dbg !17
  %1586 = insertelement <4 x float> poison, float %1312, i64 0, !dbg !16
  %1587 = insertelement <4 x float> %1586, float %1318, i64 1, !dbg !16
  %1588 = insertelement <4 x float> %1587, float %1313, i64 2, !dbg !16
  %1589 = insertelement <4 x float> %1588, float %1319, i64 3, !dbg !16
  %normalized49 = fmul <4 x float> %1589, %row.reciprocal, !dbg !16
  %1590 = fptrunc <4 x float> %normalized49 to <4 x half>, !dbg !17
  %1591 = extractelement <4 x half> %1590, i64 3, !dbg !17
  %1592 = extractelement <4 x half> %1590, i64 2, !dbg !17
  %1593 = extractelement <4 x half> %1590, i64 1, !dbg !17
  %1594 = extractelement <4 x half> %1590, i64 0, !dbg !17
  %1595 = insertelement <4 x float> poison, float %1204, i64 0, !dbg !16
  %1596 = insertelement <4 x float> %1595, float %1210, i64 1, !dbg !16
  %1597 = insertelement <4 x float> %1596, float %1205, i64 2, !dbg !16
  %1598 = insertelement <4 x float> %1597, float %1211, i64 3, !dbg !16
  %normalized51 = fmul <4 x float> %1598, %row.reciprocal38, !dbg !16
  %1599 = fptrunc <4 x float> %normalized51 to <4 x half>, !dbg !17
  %1600 = extractelement <4 x half> %1599, i64 3, !dbg !17
  %1601 = extractelement <4 x half> %1599, i64 2, !dbg !17
  %1602 = extractelement <4 x half> %1599, i64 1, !dbg !17
  %1603 = extractelement <4 x half> %1599, i64 0, !dbg !17
  %1604 = insertelement <4 x float> poison, float %1095, i64 0, !dbg !16
  %1605 = insertelement <4 x float> %1604, float %1101, i64 1, !dbg !16
  %1606 = insertelement <4 x float> %1605, float %1096, i64 2, !dbg !16
  %1607 = insertelement <4 x float> %1606, float %1102, i64 3, !dbg !16
  %normalized52 = fmul <4 x float> %1607, %row.reciprocal38, !dbg !16
  %1608 = fptrunc <4 x float> %normalized52 to <4 x half>, !dbg !17
  %1609 = extractelement <4 x half> %1608, i64 3, !dbg !17
  %1610 = extractelement <4 x half> %1608, i64 2, !dbg !17
  %1611 = extractelement <4 x half> %1608, i64 1, !dbg !17
  %1612 = extractelement <4 x half> %1608, i64 0, !dbg !17
  %address.local1099 = or disjoint i64 %address.local987, 64, !dbg !18
  %address.local1100 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1099, !dbg !18
  store half %1612, ptr addrspace(1) %address.local1100, align 2, !dbg !18
  %address.local1102 = or disjoint i64 %address.local987, 65, !dbg !18
  %address.local1103 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1102, !dbg !18
  store half %1611, ptr addrspace(1) %address.local1103, align 2, !dbg !18
  %address.local1105 = or disjoint i64 %address.local987, 66, !dbg !18
  %address.local1106 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1105, !dbg !18
  store half %1610, ptr addrspace(1) %address.local1106, align 2, !dbg !18
  %address.local1108 = or disjoint i64 %address.local987, 67, !dbg !18
  %address.local1109 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1108, !dbg !18
  store half %1609, ptr addrspace(1) %address.local1109, align 2, !dbg !18
  %address.local1111 = or disjoint i64 %address.local987, 80, !dbg !18
  %address.local1112 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1111, !dbg !18
  store half %1603, ptr addrspace(1) %address.local1112, align 2, !dbg !18
  %address.local1114 = or disjoint i64 %address.local987, 81, !dbg !18
  %address.local1115 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1114, !dbg !18
  store half %1602, ptr addrspace(1) %address.local1115, align 2, !dbg !18
  %address.local1117 = or disjoint i64 %address.local987, 82, !dbg !18
  %address.local1118 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1117, !dbg !18
  store half %1601, ptr addrspace(1) %address.local1118, align 2, !dbg !18
  %address.local1120 = or disjoint i64 %address.local987, 83, !dbg !18
  %address.local1121 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1120, !dbg !18
  store half %1600, ptr addrspace(1) %address.local1121, align 2, !dbg !18
  %address.local1126 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1099, !dbg !18
  store half %1594, ptr addrspace(1) %address.local1126, align 2, !dbg !18
  %address.local1128 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1102, !dbg !18
  store half %1593, ptr addrspace(1) %address.local1128, align 2, !dbg !18
  %address.local1130 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1105, !dbg !18
  store half %1592, ptr addrspace(1) %address.local1130, align 2, !dbg !18
  %address.local1132 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1108, !dbg !18
  store half %1591, ptr addrspace(1) %address.local1132, align 2, !dbg !18
  %address.local1134 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1111, !dbg !18
  store half %1585, ptr addrspace(1) %address.local1134, align 2, !dbg !18
  %address.local1136 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1114, !dbg !18
  store half %1584, ptr addrspace(1) %address.local1136, align 2, !dbg !18
  %address.local1138 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1117, !dbg !18
  store half %1583, ptr addrspace(1) %address.local1138, align 2, !dbg !18
  %address.local1140 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1120, !dbg !18
  store half %1582, ptr addrspace(1) %address.local1140, align 2, !dbg !18
  %1613 = insertelement <4 x float> poison, float %1393, i64 0, !dbg !16
  %1614 = insertelement <4 x float> %1613, float %1399, i64 1, !dbg !16
  %1615 = insertelement <4 x float> %1614, float %1394, i64 2, !dbg !16
  %1616 = insertelement <4 x float> %1615, float %1400, i64 3, !dbg !16
  %normalized54 = fmul <4 x float> %1616, %row.reciprocal, !dbg !16
  %1617 = fptrunc <4 x float> %normalized54 to <4 x half>, !dbg !17
  %1618 = extractelement <4 x half> %1617, i64 3, !dbg !17
  %1619 = extractelement <4 x half> %1617, i64 2, !dbg !17
  %1620 = extractelement <4 x half> %1617, i64 1, !dbg !17
  %1621 = extractelement <4 x half> %1617, i64 0, !dbg !17
  %1622 = insertelement <4 x float> poison, float %1283, i64 0, !dbg !16
  %1623 = insertelement <4 x float> %1622, float %1289, i64 1, !dbg !16
  %1624 = insertelement <4 x float> %1623, float %1284, i64 2, !dbg !16
  %1625 = insertelement <4 x float> %1624, float %1290, i64 3, !dbg !16
  %normalized55 = fmul <4 x float> %1625, %row.reciprocal, !dbg !16
  %1626 = fptrunc <4 x float> %normalized55 to <4 x half>, !dbg !17
  %1627 = extractelement <4 x half> %1626, i64 3, !dbg !17
  %1628 = extractelement <4 x half> %1626, i64 2, !dbg !17
  %1629 = extractelement <4 x half> %1626, i64 1, !dbg !17
  %1630 = extractelement <4 x half> %1626, i64 0, !dbg !17
  %1631 = insertelement <4 x float> poison, float %1176, i64 0, !dbg !16
  %1632 = insertelement <4 x float> %1631, float %1182, i64 1, !dbg !16
  %1633 = insertelement <4 x float> %1632, float %1177, i64 2, !dbg !16
  %1634 = insertelement <4 x float> %1633, float %1183, i64 3, !dbg !16
  %normalized57 = fmul <4 x float> %1634, %row.reciprocal38, !dbg !16
  %1635 = fptrunc <4 x float> %normalized57 to <4 x half>, !dbg !17
  %1636 = extractelement <4 x half> %1635, i64 3, !dbg !17
  %1637 = extractelement <4 x half> %1635, i64 2, !dbg !17
  %1638 = extractelement <4 x half> %1635, i64 1, !dbg !17
  %1639 = extractelement <4 x half> %1635, i64 0, !dbg !17
  %1640 = insertelement <4 x float> poison, float %1066, i64 0, !dbg !16
  %1641 = insertelement <4 x float> %1640, float %1072, i64 1, !dbg !16
  %1642 = insertelement <4 x float> %1641, float %1067, i64 2, !dbg !16
  %1643 = insertelement <4 x float> %1642, float %1073, i64 3, !dbg !16
  %normalized58 = fmul <4 x float> %1643, %row.reciprocal38, !dbg !16
  %1644 = fptrunc <4 x float> %normalized58 to <4 x half>, !dbg !17
  %1645 = extractelement <4 x half> %1644, i64 3, !dbg !17
  %1646 = extractelement <4 x half> %1644, i64 2, !dbg !17
  %1647 = extractelement <4 x half> %1644, i64 1, !dbg !17
  %1648 = extractelement <4 x half> %1644, i64 0, !dbg !17
  %address.local1155 = or disjoint i64 %address.local987, 96, !dbg !18
  %address.local1156 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1155, !dbg !18
  store half %1648, ptr addrspace(1) %address.local1156, align 2, !dbg !18
  %address.local1158 = or disjoint i64 %address.local987, 97, !dbg !18
  %address.local1159 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1158, !dbg !18
  store half %1647, ptr addrspace(1) %address.local1159, align 2, !dbg !18
  %address.local1161 = or disjoint i64 %address.local987, 98, !dbg !18
  %address.local1162 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1161, !dbg !18
  store half %1646, ptr addrspace(1) %address.local1162, align 2, !dbg !18
  %address.local1164 = or disjoint i64 %address.local987, 99, !dbg !18
  %address.local1165 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1164, !dbg !18
  store half %1645, ptr addrspace(1) %address.local1165, align 2, !dbg !18
  %address.local1167 = or disjoint i64 %address.local987, 112, !dbg !18
  %address.local1168 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1167, !dbg !18
  store half %1639, ptr addrspace(1) %address.local1168, align 2, !dbg !18
  %address.local1170 = or disjoint i64 %address.local987, 113, !dbg !18
  %address.local1171 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1170, !dbg !18
  store half %1638, ptr addrspace(1) %address.local1171, align 2, !dbg !18
  %address.local1173 = or disjoint i64 %address.local987, 114, !dbg !18
  %address.local1174 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1173, !dbg !18
  store half %1637, ptr addrspace(1) %address.local1174, align 2, !dbg !18
  %address.local1176 = or i64 %1539, 115, !dbg !18
  %address.local1177 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1176, !dbg !18
  store half %1636, ptr addrspace(1) %address.local1177, align 2, !dbg !18
  %address.local1182 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1155, !dbg !18
  store half %1630, ptr addrspace(1) %address.local1182, align 2, !dbg !18
  %address.local1184 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1158, !dbg !18
  store half %1629, ptr addrspace(1) %address.local1184, align 2, !dbg !18
  %address.local1186 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1161, !dbg !18
  store half %1628, ptr addrspace(1) %address.local1186, align 2, !dbg !18
  %address.local1188 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1164, !dbg !18
  store half %1627, ptr addrspace(1) %address.local1188, align 2, !dbg !18
  %address.local1190 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1167, !dbg !18
  store half %1621, ptr addrspace(1) %address.local1190, align 2, !dbg !18
  %address.local1192 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1170, !dbg !18
  store half %1620, ptr addrspace(1) %address.local1192, align 2, !dbg !18
  %address.local1194 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1173, !dbg !18
  store half %1619, ptr addrspace(1) %address.local1194, align 2, !dbg !18
  %address.local1196 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1176, !dbg !18
  store half %1618, ptr addrspace(1) %address.local1196, align 2, !dbg !18
  ret void, !dbg !5
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.y() #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #2

; Function Attrs: convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.readfirstlane.i32(i32) #3

; Function Attrs: nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.asyncmark() #4

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.z() #2

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

!0 = !DIFile(filename: "baseline.input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/noncausal_softmax_fragments")
!1 = !{}
!2 = !DISubroutineType(types: !1)
!3 = distinct !DICompileUnit(language: DW_LANG_C, file: !0, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!4 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !0, file: !0, line: 23, type: !2, scopeLine: 23, spFlags: DISPFlagDefinition, unit: !3)
!5 = !DILocation(line: 23, column: 3, scope: !4)
!7 = !DILocation(line: 57, column: 24, scope: !4)
!8 = !DILocation(line: 67, column: 22, scope: !4)
!10 = !DILocation(line: 69, column: 24, scope: !4)
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
