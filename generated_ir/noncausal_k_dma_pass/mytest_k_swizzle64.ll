; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/noncausal_k_dma_pass/mytest/run-20261030-163658-kcakvj_l/grid-000003/input.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/noncausal_k_dma_pass/mytest/run-20261030-163658-kcakvj_l/grid-000003/input.mlir"

@shm_0 = addrspace(3) global [16384 x i8] undef, align 16

; Function Attrs: convergent willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

declare float @__ocml_exp2_f32(float)

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3) #1 !dbg !4 !reqd_work_group_size !30 {
  %kdma.tid = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !5
  %kdma.lane = and i32 %kdma.tid, 63, !dbg !5
  %kdma.wave = and i32 %kdma.tid, -64, !dbg !5
  %5 = lshr i32 %kdma.lane, 1, !dbg !5
  %kdma.owner = or i32 %kdma.wave, %5, !dbg !5
  %kdma.owner.hi = add i32 %kdma.owner, 32, !dbg !5
  %6 = and i32 %kdma.lane, 1, !dbg !5
  %kdma.intra = shl i32 %6, 3, !dbg !5
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
  %17 = lshr i64 %12, 5, !dbg !5
  %18 = lshr i64 %12, 10, !dbg !8
  %19 = add nuw nsw i64 %17, %18, !dbg !8
  %20 = and i64 %19, 31, !dbg !8
  %21 = shl nsw i64 %16, 7, !dbg !8
  %22 = mul nsw i64 %18, -4096, !dbg !8
  %23 = add i64 %21, %22, !dbg !8
  %24 = shl nuw nsw i64 %20, 19, !dbg !8
  %25 = shl nsw i64 %23, 7, !dbg !8
  %26 = add i64 %24, %25, !dbg !8
  %27 = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !8, !range !32
  %28 = zext nneg i32 %27 to i64, !dbg !8
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %27) #9, !dbg !8, !range !32, !frisk.register_lifetimes !1
  %address.local = zext nneg i32 %address.thread to i64, !dbg !8
  %address.local404 = shl nuw nsw i64 %address.local, 3, !dbg !8
  %address.local405 = and i64 %address.local404, 1536, !dbg !8
  %address.local406 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local405, !dbg !8
  %address.local407 = ptrtoint ptr addrspace(3) %address.local406 to i64, !dbg !8
  %address.local408 = trunc i64 %address.local407 to i32, !dbg !8
  %29 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local408), !dbg !8
  %address.local409 = shl nuw nsw i64 %address.local, 7, !dbg !8
  %address.local410 = and i64 %address.local409, 16256, !dbg !8
  %address.local411 = lshr i64 %address.local, 4, !dbg !8
  %address.local412 = and i64 %address.local411, 8, !dbg !8
  %address.local413 = or disjoint i64 %address.local410, %address.local412, !dbg !8
  %30 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %.idx1399 = or disjoint i64 %26, %address.local413, !dbg !8
  %.idx1399.tr = trunc i64 %.idx1399 to i32, !dbg !8
  %dma.buffer.offset1396 = shl i32 %.idx1399.tr, 1, !dbg !8
  %31 = insertelement <2 x i64> poison, i64 %30, i64 0, !dbg !8
  %32 = bitcast <2 x i64> %31 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc1397 = shufflevector <4 x i32> %32, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
  %dma.buffer.lds1398 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %29), !dbg !8
  %33 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1398, i32 %dma.buffer.offset1396, <4 x i32> %dma.buffer.rsrc1397) #10, !dbg !8
  %address.local415 = or disjoint i64 %address.local, 256, !dbg !8
  %address.local416 = shl nuw nsw i64 %address.local415, 3, !dbg !8
  %address.local417 = and i64 %address.local416, 3584, !dbg !8
  %address.local418 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local417, !dbg !8
  %address.local419 = ptrtoint ptr addrspace(3) %address.local418 to i64, !dbg !8
  %address.local420 = trunc i64 %address.local419 to i32, !dbg !8
  %34 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local420), !dbg !8
  %address.local421 = shl nuw nsw i64 %address.local, 7, !dbg !8
  %address.local422 = and i64 %address.local421, 16256, !dbg !8
  %address.local423 = lshr i64 %address.local415, 4, !dbg !8
  %address.local424 = and i64 %address.local423, 24, !dbg !8
  %address.local425 = or disjoint i64 %address.local422, %address.local424, !dbg !8
  %35 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %.idx1400 = or disjoint i64 %26, %address.local425, !dbg !8
  %.idx1400.tr = trunc i64 %.idx1400 to i32, !dbg !8
  %dma.buffer.offset1393 = shl i32 %.idx1400.tr, 1, !dbg !8
  %36 = insertelement <2 x i64> poison, i64 %35, i64 0, !dbg !8
  %37 = bitcast <2 x i64> %36 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc1394 = shufflevector <4 x i32> %37, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
  %dma.buffer.lds1395 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %34), !dbg !8
  %38 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1395, i32 %dma.buffer.offset1393, <4 x i32> %dma.buffer.rsrc1394) #10, !dbg !8
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %address.local427 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local405, !dbg !8
  %address.local428 = ptrtoint ptr addrspace(3) %address.local427 to i64, !dbg !8
  %address.local429 = trunc i64 %address.local428 to i32, !dbg !8
  %39 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local429), !dbg !8
  %40 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %41 = or disjoint i64 %26, 32, !dbg !8
  %42 = or disjoint i64 %address.local413, %41, !dbg !8
  %.tr = trunc i64 %42 to i32, !dbg !8
  %dma.buffer.offset1390 = shl i32 %.tr, 1, !dbg !8
  %43 = insertelement <2 x i64> poison, i64 %40, i64 0, !dbg !8
  %44 = bitcast <2 x i64> %43 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc1391 = shufflevector <4 x i32> %44, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
  %dma.buffer.lds1392 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %39), !dbg !8
  %45 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1392, i32 %dma.buffer.offset1390, <4 x i32> %dma.buffer.rsrc1391) #10, !dbg !8
  %address.local431 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local417, !dbg !8
  %address.local432 = ptrtoint ptr addrspace(3) %address.local431 to i64, !dbg !8
  %address.local433 = trunc i64 %address.local432 to i32, !dbg !8
  %46 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local433), !dbg !8
  %47 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %48 = or disjoint i64 %26, 32, !dbg !8
  %49 = or disjoint i64 %address.local425, %48, !dbg !8
  %.tr1401 = trunc i64 %49 to i32, !dbg !8
  %dma.buffer.offset1387 = shl i32 %.tr1401, 1, !dbg !8
  %50 = insertelement <2 x i64> poison, i64 %47, i64 0, !dbg !8
  %51 = bitcast <2 x i64> %50 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc1388 = shufflevector <4 x i32> %51, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
  %dma.buffer.lds1389 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %46), !dbg !8
  %52 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1389, i32 %dma.buffer.offset1387, <4 x i32> %dma.buffer.rsrc1388) #10, !dbg !8
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
  %53 = load <4 x half>, ptr addrspace(3) %address.local447, align 8, !dbg !10
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local447, i64 4096, !dbg !10
  %54 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !10
  %55 = fmul <4 x half> %53, splat (half 0xH3015), !dbg !10
  %lds.const.gep1197 = getelementptr i8, ptr addrspace(3) %address.local447, i64 1024, !dbg !10
  %56 = load <4 x half>, ptr addrspace(3) %lds.const.gep1197, align 8, !dbg !10
  %57 = fmul <4 x half> %54, splat (half 0xH3015), !dbg !10
  %lds.const.gep1198 = getelementptr i8, ptr addrspace(3) %address.local447, i64 5120, !dbg !10
  %58 = load <4 x half>, ptr addrspace(3) %lds.const.gep1198, align 8, !dbg !10
  %59 = fmul <4 x half> %56, splat (half 0xH3015), !dbg !10
  %60 = fmul <4 x half> %58, splat (half 0xH3015), !dbg !10
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %61 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local408), !dbg !8
  %62 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %63 = or disjoint i64 %26, 64, !dbg !8
  %64 = or disjoint i64 %address.local413, %63, !dbg !8
  %.tr1402 = trunc i64 %64 to i32, !dbg !8
  %dma.buffer.offset1384 = shl i32 %.tr1402, 1, !dbg !8
  %65 = insertelement <2 x i64> poison, i64 %62, i64 0, !dbg !8
  %66 = bitcast <2 x i64> %65 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc1385 = shufflevector <4 x i32> %66, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
  %dma.buffer.lds1386 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %61), !dbg !8
  %67 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1386, i32 %dma.buffer.offset1384, <4 x i32> %dma.buffer.rsrc1385) #10, !dbg !8
  %68 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local420), !dbg !8
  %69 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %70 = or disjoint i64 %26, 64, !dbg !8
  %71 = or disjoint i64 %address.local425, %70, !dbg !8
  %.tr1403 = trunc i64 %71 to i32, !dbg !8
  %dma.buffer.offset1381 = shl i32 %.tr1403, 1, !dbg !8
  %72 = insertelement <2 x i64> poison, i64 %69, i64 0, !dbg !8
  %73 = bitcast <2 x i64> %72 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc1382 = shufflevector <4 x i32> %73, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
  %dma.buffer.lds1383 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %68), !dbg !8
  %74 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1383, i32 %dma.buffer.offset1381, <4 x i32> %dma.buffer.rsrc1382) #10, !dbg !8
  %lds.const.gep1199 = getelementptr i8, ptr addrspace(3) %address.local447, i64 8192, !dbg !10
  %75 = load <4 x half>, ptr addrspace(3) %lds.const.gep1199, align 8, !dbg !10
  %lds.const.gep1200 = getelementptr i8, ptr addrspace(3) %address.local447, i64 12288, !dbg !10
  %76 = load <4 x half>, ptr addrspace(3) %lds.const.gep1200, align 8, !dbg !10
  %lds.const.gep1201 = getelementptr i8, ptr addrspace(3) %address.local447, i64 9216, !dbg !10
  %77 = load <4 x half>, ptr addrspace(3) %lds.const.gep1201, align 8, !dbg !10
  %78 = fmul <4 x half> %75, splat (half 0xH3015), !dbg !10
  %lds.const.gep1202 = getelementptr i8, ptr addrspace(3) %address.local447, i64 13312, !dbg !10
  %79 = load <4 x half>, ptr addrspace(3) %lds.const.gep1202, align 8, !dbg !10
  %80 = fmul <4 x half> %76, splat (half 0xH3015), !dbg !10
  %81 = fmul <4 x half> %77, splat (half 0xH3015), !dbg !10
  %82 = fmul <4 x half> %79, splat (half 0xH3015), !dbg !10
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %83 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local429), !dbg !8
  %84 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %85 = or disjoint i64 %26, 96, !dbg !8
  %86 = or disjoint i64 %address.local413, %85, !dbg !8
  %.tr1404 = trunc i64 %86 to i32, !dbg !8
  %dma.buffer.offset1378 = shl i32 %.tr1404, 1, !dbg !8
  %87 = insertelement <2 x i64> poison, i64 %84, i64 0, !dbg !8
  %88 = bitcast <2 x i64> %87 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc1379 = shufflevector <4 x i32> %88, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
  %dma.buffer.lds1380 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %83), !dbg !8
  %89 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1380, i32 %dma.buffer.offset1378, <4 x i32> %dma.buffer.rsrc1379) #10, !dbg !8
  %90 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local433), !dbg !8
  %91 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %92 = or disjoint i64 %26, 96, !dbg !8
  %93 = or disjoint i64 %address.local425, %92, !dbg !8
  %.tr1405 = trunc i64 %93 to i32, !dbg !8
  %dma.buffer.offset1375 = shl i32 %.tr1405, 1, !dbg !8
  %94 = insertelement <2 x i64> poison, i64 %91, i64 0, !dbg !8
  %95 = bitcast <2 x i64> %94 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc1376 = shufflevector <4 x i32> %95, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
  %dma.buffer.lds1377 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %90), !dbg !8
  %96 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1377, i32 %dma.buffer.offset1375, <4 x i32> %dma.buffer.rsrc1376) #10, !dbg !8
  %97 = load <4 x half>, ptr addrspace(3) %address.local447, align 8, !dbg !10
  %lds.const.gep1203 = getelementptr i8, ptr addrspace(3) %address.local447, i64 4096, !dbg !10
  %98 = load <4 x half>, ptr addrspace(3) %lds.const.gep1203, align 8, !dbg !10
  %lds.const.gep1204 = getelementptr i8, ptr addrspace(3) %address.local447, i64 1024, !dbg !10
  %99 = load <4 x half>, ptr addrspace(3) %lds.const.gep1204, align 8, !dbg !10
  %100 = fmul <4 x half> %97, splat (half 0xH3015), !dbg !10
  %lds.const.gep1205 = getelementptr i8, ptr addrspace(3) %address.local447, i64 5120, !dbg !10
  %101 = load <4 x half>, ptr addrspace(3) %lds.const.gep1205, align 8, !dbg !10
  %102 = fmul <4 x half> %98, splat (half 0xH3015), !dbg !10
  %103 = fmul <4 x half> %99, splat (half 0xH3015), !dbg !10
  %104 = fmul <4 x half> %101, splat (half 0xH3015), !dbg !10
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %lds.const.gep1206 = getelementptr i8, ptr addrspace(3) %address.local447, i64 8192, !dbg !10
  %105 = load <4 x half>, ptr addrspace(3) %lds.const.gep1206, align 8, !dbg !10
  %lds.const.gep1207 = getelementptr i8, ptr addrspace(3) %address.local447, i64 12288, !dbg !10
  %106 = load <4 x half>, ptr addrspace(3) %lds.const.gep1207, align 8, !dbg !10
  %lds.const.gep1208 = getelementptr i8, ptr addrspace(3) %address.local447, i64 9216, !dbg !10
  %107 = load <4 x half>, ptr addrspace(3) %lds.const.gep1208, align 8, !dbg !10
  %108 = fmul <4 x half> %105, splat (half 0xH3015), !dbg !10
  %lds.const.gep1209 = getelementptr i8, ptr addrspace(3) %address.local447, i64 13312, !dbg !10
  %109 = load <4 x half>, ptr addrspace(3) %lds.const.gep1209, align 8, !dbg !10
  %110 = fmul <4 x half> %106, splat (half 0xH3015), !dbg !10
  %111 = fmul <4 x half> %107, splat (half 0xH3015), !dbg !10
  %112 = fmul <4 x half> %109, splat (half 0xH3015), !dbg !10
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %113 = shl nuw nsw i64 %28, 2, !dbg !14
  %114 = shl nuw nsw i64 %28, 2, !dbg !14
  %115 = and i64 %114, 1008, !dbg !14
  %116 = sub nsw i64 %113, %115, !dbg !14
  %117 = and i32 %27, 1, !dbg !14
  %index.compact344 = zext nneg i32 %117 to i64, !dbg !14
  %118 = or disjoint i64 %116, %index.compact344, !dbg !14
  %119 = add nuw nsw i64 %118, 1, !dbg !14
  %120 = sdiv i64 %119, 2, !dbg !14
  br label %121, !dbg !7

121:                                              ; preds = %130, %4
  %122 = phi i64 [ %1144, %130 ], [ 0, %4 ], !dbg !7
  %123 = phi [2 x <8 x float>] [ %1129, %130 ], [ zeroinitializer, %4 ], !dbg !7
  %124 = phi [2 x <1 x float>] [ %823, %130 ], [ zeroinitializer, %4 ], !dbg !7
  %125 = phi [2 x <1 x float>] [ %1131, %130 ], [ [<1 x float> splat (float 0xFFF0000000000000), <1 x float> splat (float 0xFFF0000000000000)], %4 ], !dbg !7
  %126 = phi [2 x <8 x float>] [ %1135, %130 ], [ zeroinitializer, %4 ], !dbg !7
  %127 = phi [2 x <8 x float>] [ %1139, %130 ], [ zeroinitializer, %4 ], !dbg !7
  %128 = phi [2 x <8 x float>] [ %1143, %130 ], [ zeroinitializer, %4 ], !dbg !7
  %129 = icmp samesign ult i64 %122, 32, !dbg !7
  br i1 %129, label %130, label %1145, !dbg !7

130:                                              ; preds = %121
  %131 = shl i64 %17, 19, !dbg !12
  %132 = and i64 %131, 16252928, !dbg !12
  %133 = shl nuw nsw i64 %122, 14, !dbg !12
  %134 = or disjoint i64 %132, %133, !dbg !12
  %address.thread498 = call i32 asm sideeffect "", "=v,0"(i32 %27) #9, !dbg !12, !range !32, !frisk.register_lifetimes !1
  %address.local499 = zext nneg i32 %address.thread498 to i64, !dbg !8
  %address.local500 = shl nuw nsw i64 %address.local499, 3, !dbg !8
  %address.local501 = and i64 %address.local500, 1536, !dbg !8
  %address.local502 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local501, !dbg !8
  %address.local503 = ptrtoint ptr addrspace(3) %address.local502 to i64, !dbg !8
  %address.local504 = trunc i64 %address.local503 to i32, !dbg !8
  %135 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local504), !dbg !12
  %136 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !12
  %137 = insertelement <2 x i64> poison, i64 %136, i64 0, !dbg !12
  %138 = bitcast <2 x i64> %137 to <4 x i32>, !dbg !12
  %dma.buffer.rsrc1373 = shufflevector <4 x i32> %138, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !12
  %dma.buffer.lds1374 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %135), !dbg !12
  %kdma.source = zext nneg i32 %kdma.owner to i64, !dbg !12
  %kdma.source1506 = lshr i64 %kdma.source, 4, !dbg !12
  %kdma.source1507 = and i64 %kdma.source1506, 8, !dbg !12
  %kdma.source1508 = and i64 %kdma.source, 123, !dbg !12
  %kdma.source1509 = lshr i64 %kdma.source, 2, !dbg !12
  %kdma.source1510 = and i64 %kdma.source1509, 4, !dbg !12
  %kdma.source1511 = add nuw nsw i64 %kdma.source1510, %kdma.source, !dbg !12
  %kdma.source1512 = and i64 %kdma.source1511, 4, !dbg !12
  %kdma.source1513 = or disjoint i64 %kdma.source1508, %kdma.source1512, !dbg !12
  %kdma.source1514 = shl nuw nsw i64 %kdma.source1513, 7, !dbg !12
  %kdma.source1515 = or disjoint i64 %kdma.source1507, %kdma.source1514, !dbg !12
  %kdma.source1516 = or disjoint i64 %134, %kdma.source1515, !dbg !12
  %kdma.source1517 = trunc nuw i64 %kdma.source1516 to i32, !dbg !12
  %kdma.source1518 = shl nuw nsw i32 %kdma.source1517, 1, !dbg !12
  %kdma.offset = add i32 %kdma.source1518, %kdma.intra, !dbg !12
  %kdma.lds = add i32 %dma.buffer.lds1374, 0, !dbg !12
  %139 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds, i32 %kdma.offset, <4 x i32> %dma.buffer.rsrc1373) #10, !dbg !12
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
  %kdma.source1530 = or disjoint i64 %134, %kdma.source1529, !dbg !12
  %kdma.source1531 = trunc nuw i64 %kdma.source1530 to i32, !dbg !12
  %kdma.source1532 = shl nuw nsw i32 %kdma.source1531, 1, !dbg !12
  %kdma.offset1533 = add i32 %kdma.source1532, %kdma.intra.hi, !dbg !12
  %kdma.lds1534 = add i32 %dma.buffer.lds1374, 512, !dbg !12
  %140 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1534, i32 %kdma.offset1533, <4 x i32> %dma.buffer.rsrc1373) #10, !dbg !12
  %address.local516 = or disjoint i64 %address.local499, 256, !dbg !8
  %address.local517 = shl nuw nsw i64 %address.local516, 3, !dbg !8
  %address.local518 = and i64 %address.local517, 3584, !dbg !8
  %address.local519 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local518, !dbg !8
  %address.local520 = ptrtoint ptr addrspace(3) %address.local519 to i64, !dbg !8
  %address.local521 = trunc i64 %address.local520 to i32, !dbg !8
  %141 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local521), !dbg !12
  %142 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !12
  %143 = insertelement <2 x i64> poison, i64 %142, i64 0, !dbg !12
  %144 = bitcast <2 x i64> %143 to <4 x i32>, !dbg !12
  %dma.buffer.rsrc1370 = shufflevector <4 x i32> %144, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !12
  %dma.buffer.lds1371 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %141), !dbg !12
  %kdma.source1535 = zext nneg i32 %kdma.owner to i64, !dbg !12
  %kdma.source1536 = or disjoint i64 %kdma.source1535, 256, !dbg !12
  %kdma.source1537 = lshr i64 %kdma.source1536, 4, !dbg !12
  %kdma.source1538 = and i64 %kdma.source1537, 24, !dbg !12
  %kdma.source1539 = and i64 %kdma.source1535, 123, !dbg !12
  %kdma.source1540 = lshr i64 %kdma.source1535, 2, !dbg !12
  %kdma.source1541 = and i64 %kdma.source1540, 4, !dbg !12
  %kdma.source1542 = add nuw nsw i64 %kdma.source1541, %kdma.source1535, !dbg !12
  %kdma.source1543 = and i64 %kdma.source1542, 4, !dbg !12
  %kdma.source1544 = or disjoint i64 %kdma.source1539, %kdma.source1543, !dbg !12
  %kdma.source1545 = shl nuw nsw i64 %kdma.source1544, 7, !dbg !12
  %kdma.source1546 = or disjoint i64 %kdma.source1538, %kdma.source1545, !dbg !12
  %kdma.source1547 = or disjoint i64 %134, %kdma.source1546, !dbg !12
  %kdma.source1548 = trunc nuw i64 %kdma.source1547 to i32, !dbg !12
  %kdma.source1549 = shl nuw nsw i32 %kdma.source1548, 1, !dbg !12
  %kdma.offset1550 = add i32 %kdma.source1549, %kdma.intra, !dbg !12
  %kdma.lds1551 = add i32 %dma.buffer.lds1371, 0, !dbg !12
  %145 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1551, i32 %kdma.offset1550, <4 x i32> %dma.buffer.rsrc1370) #10, !dbg !12
  %kdma.source1552 = zext nneg i32 %kdma.owner.hi to i64, !dbg !12
  %kdma.source1553 = or disjoint i64 %kdma.source1552, 256, !dbg !12
  %kdma.source1554 = lshr i64 %kdma.source1553, 4, !dbg !12
  %kdma.source1555 = and i64 %kdma.source1554, 24, !dbg !12
  %kdma.source1556 = and i64 %kdma.source1552, 123, !dbg !12
  %kdma.source1557 = lshr i64 %kdma.source1552, 2, !dbg !12
  %kdma.source1558 = and i64 %kdma.source1557, 4, !dbg !12
  %kdma.source1559 = add nuw nsw i64 %kdma.source1558, %kdma.source1552, !dbg !12
  %kdma.source1560 = and i64 %kdma.source1559, 4, !dbg !12
  %kdma.source1561 = or disjoint i64 %kdma.source1556, %kdma.source1560, !dbg !12
  %kdma.source1562 = shl nuw nsw i64 %kdma.source1561, 7, !dbg !12
  %kdma.source1563 = or disjoint i64 %kdma.source1555, %kdma.source1562, !dbg !12
  %kdma.source1564 = or disjoint i64 %134, %kdma.source1563, !dbg !12
  %kdma.source1565 = trunc nuw i64 %kdma.source1564 to i32, !dbg !12
  %kdma.source1566 = shl nuw nsw i32 %kdma.source1565, 1, !dbg !12
  %kdma.offset1567 = add i32 %kdma.source1566, %kdma.intra.hi, !dbg !12
  %kdma.lds1568 = add i32 %dma.buffer.lds1371, 512, !dbg !12
  %146 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1568, i32 %kdma.offset1567, <4 x i32> %dma.buffer.rsrc1370) #10, !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %147 = or disjoint i64 %134, 32, !dbg !12
  %address.local533 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local501, !dbg !8
  %address.local534 = ptrtoint ptr addrspace(3) %address.local533 to i64, !dbg !8
  %address.local535 = trunc i64 %address.local534 to i32, !dbg !8
  %148 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local535), !dbg !12
  %149 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !12
  %150 = insertelement <2 x i64> poison, i64 %149, i64 0, !dbg !12
  %151 = bitcast <2 x i64> %150 to <4 x i32>, !dbg !12
  %dma.buffer.rsrc1367 = shufflevector <4 x i32> %151, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !12
  %dma.buffer.lds1368 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %148), !dbg !12
  %kdma.source1569 = zext nneg i32 %kdma.owner to i64, !dbg !12
  %kdma.source1570 = lshr i64 %kdma.source1569, 4, !dbg !12
  %kdma.source1571 = and i64 %kdma.source1570, 8, !dbg !12
  %kdma.source1572 = and i64 %kdma.source1569, 123, !dbg !12
  %kdma.source1573 = lshr i64 %kdma.source1569, 2, !dbg !12
  %kdma.source1574 = and i64 %kdma.source1573, 4, !dbg !12
  %kdma.source1575 = add nuw nsw i64 %kdma.source1574, %kdma.source1569, !dbg !12
  %kdma.source1576 = and i64 %kdma.source1575, 4, !dbg !12
  %kdma.source1577 = or disjoint i64 %kdma.source1572, %kdma.source1576, !dbg !12
  %kdma.source1578 = shl nuw nsw i64 %kdma.source1577, 7, !dbg !12
  %kdma.source1579 = or disjoint i64 %kdma.source1571, %kdma.source1578, !dbg !12
  %kdma.source1580 = or disjoint i64 %147, %kdma.source1579, !dbg !12
  %kdma.source1581 = trunc nuw i64 %kdma.source1580 to i32, !dbg !12
  %kdma.source1582 = shl nuw nsw i32 %kdma.source1581, 1, !dbg !12
  %kdma.offset1583 = add i32 %kdma.source1582, %kdma.intra, !dbg !12
  %kdma.lds1584 = add i32 %dma.buffer.lds1368, 0, !dbg !12
  %152 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1584, i32 %kdma.offset1583, <4 x i32> %dma.buffer.rsrc1367) #10, !dbg !12
  %kdma.source1585 = zext nneg i32 %kdma.owner.hi to i64, !dbg !12
  %kdma.source1586 = lshr i64 %kdma.source1585, 4, !dbg !12
  %kdma.source1587 = and i64 %kdma.source1586, 8, !dbg !12
  %kdma.source1588 = and i64 %kdma.source1585, 123, !dbg !12
  %kdma.source1589 = lshr i64 %kdma.source1585, 2, !dbg !12
  %kdma.source1590 = and i64 %kdma.source1589, 4, !dbg !12
  %kdma.source1591 = add nuw nsw i64 %kdma.source1590, %kdma.source1585, !dbg !12
  %kdma.source1592 = and i64 %kdma.source1591, 4, !dbg !12
  %kdma.source1593 = or disjoint i64 %kdma.source1588, %kdma.source1592, !dbg !12
  %kdma.source1594 = shl nuw nsw i64 %kdma.source1593, 7, !dbg !12
  %kdma.source1595 = or disjoint i64 %kdma.source1587, %kdma.source1594, !dbg !12
  %kdma.source1596 = or disjoint i64 %147, %kdma.source1595, !dbg !12
  %kdma.source1597 = trunc nuw i64 %kdma.source1596 to i32, !dbg !12
  %kdma.source1598 = shl nuw nsw i32 %kdma.source1597, 1, !dbg !12
  %kdma.offset1599 = add i32 %kdma.source1598, %kdma.intra.hi, !dbg !12
  %kdma.lds1600 = add i32 %dma.buffer.lds1368, 512, !dbg !12
  %153 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1600, i32 %kdma.offset1599, <4 x i32> %dma.buffer.rsrc1367) #10, !dbg !12
  %address.local537 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local518, !dbg !8
  %address.local538 = ptrtoint ptr addrspace(3) %address.local537 to i64, !dbg !8
  %address.local539 = trunc i64 %address.local538 to i32, !dbg !8
  %154 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local539), !dbg !12
  %155 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !12
  %156 = insertelement <2 x i64> poison, i64 %155, i64 0, !dbg !12
  %157 = bitcast <2 x i64> %156 to <4 x i32>, !dbg !12
  %dma.buffer.rsrc1364 = shufflevector <4 x i32> %157, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !12
  %dma.buffer.lds1365 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %154), !dbg !12
  %kdma.source1601 = zext nneg i32 %kdma.owner to i64, !dbg !12
  %kdma.source1602 = or disjoint i64 %kdma.source1601, 256, !dbg !12
  %kdma.source1603 = lshr i64 %kdma.source1602, 4, !dbg !12
  %kdma.source1604 = and i64 %kdma.source1603, 24, !dbg !12
  %kdma.source1605 = and i64 %kdma.source1601, 123, !dbg !12
  %kdma.source1606 = lshr i64 %kdma.source1601, 2, !dbg !12
  %kdma.source1607 = and i64 %kdma.source1606, 4, !dbg !12
  %kdma.source1608 = add nuw nsw i64 %kdma.source1607, %kdma.source1601, !dbg !12
  %kdma.source1609 = and i64 %kdma.source1608, 4, !dbg !12
  %kdma.source1610 = or disjoint i64 %kdma.source1605, %kdma.source1609, !dbg !12
  %kdma.source1611 = shl nuw nsw i64 %kdma.source1610, 7, !dbg !12
  %kdma.source1612 = or disjoint i64 %kdma.source1604, %kdma.source1611, !dbg !12
  %kdma.source1613 = or disjoint i64 %147, %kdma.source1612, !dbg !12
  %kdma.source1614 = trunc nuw i64 %kdma.source1613 to i32, !dbg !12
  %kdma.source1615 = shl nuw nsw i32 %kdma.source1614, 1, !dbg !12
  %kdma.offset1616 = add i32 %kdma.source1615, %kdma.intra, !dbg !12
  %kdma.lds1617 = add i32 %dma.buffer.lds1365, 0, !dbg !12
  %158 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1617, i32 %kdma.offset1616, <4 x i32> %dma.buffer.rsrc1364) #10, !dbg !12
  %kdma.source1618 = zext nneg i32 %kdma.owner.hi to i64, !dbg !12
  %kdma.source1619 = or disjoint i64 %kdma.source1618, 256, !dbg !12
  %kdma.source1620 = lshr i64 %kdma.source1619, 4, !dbg !12
  %kdma.source1621 = and i64 %kdma.source1620, 24, !dbg !12
  %kdma.source1622 = and i64 %kdma.source1618, 123, !dbg !12
  %kdma.source1623 = lshr i64 %kdma.source1618, 2, !dbg !12
  %kdma.source1624 = and i64 %kdma.source1623, 4, !dbg !12
  %kdma.source1625 = add nuw nsw i64 %kdma.source1624, %kdma.source1618, !dbg !12
  %kdma.source1626 = and i64 %kdma.source1625, 4, !dbg !12
  %kdma.source1627 = or disjoint i64 %kdma.source1622, %kdma.source1626, !dbg !12
  %kdma.source1628 = shl nuw nsw i64 %kdma.source1627, 7, !dbg !12
  %kdma.source1629 = or disjoint i64 %kdma.source1621, %kdma.source1628, !dbg !12
  %kdma.source1630 = or disjoint i64 %147, %kdma.source1629, !dbg !12
  %kdma.source1631 = trunc nuw i64 %kdma.source1630 to i32, !dbg !12
  %kdma.source1632 = shl nuw nsw i32 %kdma.source1631, 1, !dbg !12
  %kdma.offset1633 = add i32 %kdma.source1632, %kdma.intra.hi, !dbg !12
  %kdma.lds1634 = add i32 %dma.buffer.lds1365, 512, !dbg !12
  %159 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1634, i32 %kdma.offset1633, <4 x i32> %dma.buffer.rsrc1364) #10, !dbg !12
  %address.local542 = lshr i64 %address.local499, 2, !dbg !14
  %address.local545 = shl nuw nsw i64 %address.local499, 1, !dbg !14
  %address.local546 = and i64 %address.local545, 24, !dbg !14
  %address.local548 = shl i64 %address.local499, 62, !dbg !14
  %address.local549 = ashr exact i64 %address.local548, 57, !dbg !14
  %address.local550 = and i64 %address.local549, -64, !dbg !14
  %address.local552 = lshr i64 %address.local499, 2, !dbg !10
  %address.local553 = and i64 %address.local552, 12, !dbg !10
  %address.local555 = and i64 %address.local499, 63, !dbg !10
  %address.local556 = lshr i64 %address.local555, 5, !dbg !10
  %address.local557 = mul nuw nsw i64 %address.local556, 1016, !dbg !10
  %.idx = mul nuw nsw i64 %address.local499, 320, !dbg !14
  %160 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %.idx, !dbg !14
  %.idx1410 = mul nsw i64 %address.local542, -1280, !dbg !14
  %161 = getelementptr i8, ptr addrspace(3) %160, i64 %.idx1410, !dbg !14
  %162 = getelementptr [2 x i8], ptr addrspace(3) %161, i64 %address.local546, !dbg !14
  %163 = getelementptr [2 x i8], ptr addrspace(3) %162, i64 %address.local550, !dbg !14
  %164 = getelementptr [2 x i8], ptr addrspace(3) %163, i64 %address.local553, !dbg !14
  %address.local559 = getelementptr [2 x i8], ptr addrspace(3) %164, i64 %address.local557, !dbg !14
  %165 = ptrtoint ptr addrspace(3) %address.local559 to i64, !dbg !14
  %166 = sub i64 %165, ptrtoint (ptr addrspace(3) @shm_0 to i64), !dbg !14
  %167 = lshr i64 %166, 6, !dbg !14
  %168 = and i64 %167, 8, !dbg !14
  %169 = xor i64 %166, %168, !dbg !14
  %170 = sub i64 %169, %166, !dbg !14
  %kdma.swizzled = getelementptr i8, ptr addrspace(3) %address.local559, i64 %170, !dbg !14
  %171 = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8, !dbg !14
  %172 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %55, <4 x half> %171, <4 x float> zeroinitializer) #9, !dbg !14
  %kdma.gep = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4096, !dbg !14
  %173 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8, !dbg !14
  %address.local570 = shl nuw nsw i32 %address.thread498, 1, !dbg !14
  %address.local571 = and i32 %address.local570, 24, !dbg !14
  %address.local572 = shl nuw nsw i32 %address.thread498, 8, !dbg !14
  %address.local573 = and i32 %address.local572, 768, !dbg !14
  %address.local574 = or disjoint i32 %address.local571, %address.local573, !dbg !14
  %address.local575 = shl nuw nsw i32 %address.thread498, 5, !dbg !14
  %address.local576 = and i32 %address.local575, 96, !dbg !14
  %address.local577 = or disjoint i32 %address.local574, %address.local576, !dbg !14
  %address.local578 = zext nneg i32 %address.local577 to i64, !dbg !14
  %address.base = call i64 asm sideeffect "", "=v,0"(i64 %120) #9, !dbg !14, !frisk.register_lifetimes !1
  %address.local579 = lshr i32 %address.thread498, 1, !dbg !14
  %address.local580 = and i32 %address.local579, 1, !dbg !14
  %address.local581 = zext nneg i32 %address.local580 to i64, !dbg !14
  %address.local582 = add nsw i64 %address.base, %address.local581, !dbg !14
  %address.local583 = shl i64 %address.local582, 6, !dbg !14
  %address.local584 = sub i64 %address.local578, %address.local583, !dbg !14
  %174 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local584, !dbg !14
  %175 = getelementptr [2 x i8], ptr addrspace(3) %174, i64 %address.local553, !dbg !14
  %176 = getelementptr [2 x i8], ptr addrspace(3) %175, i64 %address.local557, !dbg !14
  %address.local588 = getelementptr i8, ptr addrspace(3) %176, i64 64, !dbg !14
  %177 = ptrtoint ptr addrspace(3) %address.local588 to i64, !dbg !14
  %178 = sub i64 %177, ptrtoint (ptr addrspace(3) @shm_0 to i64), !dbg !14
  %179 = lshr i64 %178, 6, !dbg !14
  %180 = and i64 %179, 8, !dbg !14
  %181 = xor i64 %178, %180, !dbg !14
  %182 = sub i64 %181, %178, !dbg !14
  %kdma.swizzled1767 = getelementptr i8, ptr addrspace(3) %address.local588, i64 %182, !dbg !14
  %183 = load <4 x half>, ptr addrspace(3) %kdma.swizzled1767, align 8, !dbg !14
  %184 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %173, <4 x float> %172) #9, !dbg !14
  %185 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %55, <4 x half> %183, <4 x float> zeroinitializer) #9, !dbg !14
  %kdma.gep1768 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 4096, !dbg !14
  %186 = load <4 x half>, ptr addrspace(3) %kdma.gep1768, align 8, !dbg !14
  %kdma.gep1769 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 128, !dbg !14
  %187 = load <4 x half>, ptr addrspace(3) %kdma.gep1769, align 8, !dbg !14
  %188 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %186, <4 x float> %185) #9, !dbg !14
  %189 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %55, <4 x half> %187, <4 x float> zeroinitializer) #9, !dbg !14
  %kdma.gep1770 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4224, !dbg !14
  %190 = load <4 x half>, ptr addrspace(3) %kdma.gep1770, align 8, !dbg !14
  %kdma.gep1771 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 128, !dbg !14
  %191 = load <4 x half>, ptr addrspace(3) %kdma.gep1771, align 8, !dbg !14
  %192 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %190, <4 x float> %189) #9, !dbg !14
  %193 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %55, <4 x half> %191, <4 x float> zeroinitializer) #9, !dbg !14
  %kdma.gep1772 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 4224, !dbg !14
  %194 = load <4 x half>, ptr addrspace(3) %kdma.gep1772, align 8, !dbg !14
  %kdma.gep1773 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1024, !dbg !14
  %195 = load <4 x half>, ptr addrspace(3) %kdma.gep1773, align 8, !dbg !14
  %196 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %194, <4 x float> %193) #9, !dbg !14
  %197 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %55, <4 x half> %195, <4 x float> zeroinitializer) #9, !dbg !14
  %kdma.gep1774 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5120, !dbg !14
  %198 = load <4 x half>, ptr addrspace(3) %kdma.gep1774, align 8, !dbg !14
  %kdma.gep1775 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 1024, !dbg !14
  %199 = load <4 x half>, ptr addrspace(3) %kdma.gep1775, align 8, !dbg !14
  %200 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %198, <4 x float> %197) #9, !dbg !14
  %201 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %55, <4 x half> %199, <4 x float> zeroinitializer) #9, !dbg !14
  %kdma.gep1776 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 5120, !dbg !14
  %202 = load <4 x half>, ptr addrspace(3) %kdma.gep1776, align 8, !dbg !14
  %kdma.gep1777 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1152, !dbg !14
  %203 = load <4 x half>, ptr addrspace(3) %kdma.gep1777, align 8, !dbg !14
  %204 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %202, <4 x float> %201) #9, !dbg !14
  %205 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %55, <4 x half> %203, <4 x float> zeroinitializer) #9, !dbg !14
  %kdma.gep1778 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5248, !dbg !14
  %206 = load <4 x half>, ptr addrspace(3) %kdma.gep1778, align 8, !dbg !14
  %kdma.gep1779 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 1152, !dbg !14
  %207 = load <4 x half>, ptr addrspace(3) %kdma.gep1779, align 8, !dbg !14
  %208 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %206, <4 x float> %205) #9, !dbg !14
  %209 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %55, <4 x half> %207, <4 x float> zeroinitializer) #9, !dbg !14
  %kdma.gep1780 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 5248, !dbg !14
  %210 = load <4 x half>, ptr addrspace(3) %kdma.gep1780, align 8, !dbg !14
  %kdma.gep1781 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 0, !dbg !14
  %211 = load <4 x half>, ptr addrspace(3) %kdma.gep1781, align 8, !dbg !14
  %212 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %210, <4 x float> %209) #9, !dbg !14
  %213 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %211, <4 x float> zeroinitializer) #9, !dbg !14
  %kdma.gep1782 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4096, !dbg !14
  %214 = load <4 x half>, ptr addrspace(3) %kdma.gep1782, align 8, !dbg !14
  %kdma.gep1783 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 0, !dbg !14
  %215 = load <4 x half>, ptr addrspace(3) %kdma.gep1783, align 8, !dbg !14
  %216 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %214, <4 x float> %213) #9, !dbg !14
  %217 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %215, <4 x float> zeroinitializer) #9, !dbg !14
  %kdma.gep1784 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 4096, !dbg !14
  %218 = load <4 x half>, ptr addrspace(3) %kdma.gep1784, align 8, !dbg !14
  %kdma.gep1785 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 128, !dbg !14
  %219 = load <4 x half>, ptr addrspace(3) %kdma.gep1785, align 8, !dbg !14
  %220 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %218, <4 x float> %217) #9, !dbg !14
  %221 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %219, <4 x float> zeroinitializer) #9, !dbg !14
  %kdma.gep1786 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4224, !dbg !14
  %222 = load <4 x half>, ptr addrspace(3) %kdma.gep1786, align 8, !dbg !14
  %kdma.gep1787 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 128, !dbg !14
  %223 = load <4 x half>, ptr addrspace(3) %kdma.gep1787, align 8, !dbg !14
  %224 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %222, <4 x float> %221) #9, !dbg !14
  %225 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %223, <4 x float> zeroinitializer) #9, !dbg !14
  %kdma.gep1788 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 4224, !dbg !14
  %226 = load <4 x half>, ptr addrspace(3) %kdma.gep1788, align 8, !dbg !14
  %kdma.gep1789 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1024, !dbg !14
  %227 = load <4 x half>, ptr addrspace(3) %kdma.gep1789, align 8, !dbg !14
  %228 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %226, <4 x float> %225) #9, !dbg !14
  %229 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %227, <4 x float> zeroinitializer) #9, !dbg !14
  %kdma.gep1790 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5120, !dbg !14
  %230 = load <4 x half>, ptr addrspace(3) %kdma.gep1790, align 8, !dbg !14
  %kdma.gep1791 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 1024, !dbg !14
  %231 = load <4 x half>, ptr addrspace(3) %kdma.gep1791, align 8, !dbg !14
  %232 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %230, <4 x float> %229) #9, !dbg !14
  %233 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %231, <4 x float> zeroinitializer) #9, !dbg !14
  %kdma.gep1792 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 5120, !dbg !14
  %234 = load <4 x half>, ptr addrspace(3) %kdma.gep1792, align 8, !dbg !14
  %kdma.gep1793 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1152, !dbg !14
  %235 = load <4 x half>, ptr addrspace(3) %kdma.gep1793, align 8, !dbg !14
  %236 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %234, <4 x float> %233) #9, !dbg !14
  %237 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %235, <4 x float> zeroinitializer) #9, !dbg !14
  %kdma.gep1794 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5248, !dbg !14
  %238 = load <4 x half>, ptr addrspace(3) %kdma.gep1794, align 8, !dbg !14
  %kdma.gep1795 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 1152, !dbg !14
  %239 = load <4 x half>, ptr addrspace(3) %kdma.gep1795, align 8, !dbg !14
  %240 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %238, <4 x float> %237) #9, !dbg !14
  %241 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %239, <4 x float> zeroinitializer) #9, !dbg !14
  %kdma.gep1796 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 5248, !dbg !14
  %242 = load <4 x half>, ptr addrspace(3) %kdma.gep1796, align 8, !dbg !14
  %243 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %242, <4 x float> %241) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %244 = or disjoint i64 %134, 64, !dbg !12
  %245 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local504), !dbg !12
  %246 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !12
  %247 = insertelement <2 x i64> poison, i64 %246, i64 0, !dbg !12
  %248 = bitcast <2 x i64> %247 to <4 x i32>, !dbg !12
  %dma.buffer.rsrc1361 = shufflevector <4 x i32> %248, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !12
  %dma.buffer.lds1362 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %245), !dbg !12
  %kdma.source1635 = zext nneg i32 %kdma.owner to i64, !dbg !12
  %kdma.source1636 = lshr i64 %kdma.source1635, 4, !dbg !12
  %kdma.source1637 = and i64 %kdma.source1636, 8, !dbg !12
  %kdma.source1638 = and i64 %kdma.source1635, 123, !dbg !12
  %kdma.source1639 = lshr i64 %kdma.source1635, 2, !dbg !12
  %kdma.source1640 = and i64 %kdma.source1639, 4, !dbg !12
  %kdma.source1641 = add nuw nsw i64 %kdma.source1640, %kdma.source1635, !dbg !12
  %kdma.source1642 = and i64 %kdma.source1641, 4, !dbg !12
  %kdma.source1643 = or disjoint i64 %kdma.source1638, %kdma.source1642, !dbg !12
  %kdma.source1644 = shl nuw nsw i64 %kdma.source1643, 7, !dbg !12
  %kdma.source1645 = or disjoint i64 %kdma.source1637, %kdma.source1644, !dbg !12
  %kdma.source1646 = or disjoint i64 %244, %kdma.source1645, !dbg !12
  %kdma.source1647 = trunc nuw i64 %kdma.source1646 to i32, !dbg !12
  %kdma.source1648 = shl nuw nsw i32 %kdma.source1647, 1, !dbg !12
  %kdma.offset1649 = add i32 %kdma.source1648, %kdma.intra, !dbg !12
  %kdma.lds1650 = add i32 %dma.buffer.lds1362, 0, !dbg !12
  %249 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1650, i32 %kdma.offset1649, <4 x i32> %dma.buffer.rsrc1361) #10, !dbg !12
  %kdma.source1651 = zext nneg i32 %kdma.owner.hi to i64, !dbg !12
  %kdma.source1652 = lshr i64 %kdma.source1651, 4, !dbg !12
  %kdma.source1653 = and i64 %kdma.source1652, 8, !dbg !12
  %kdma.source1654 = and i64 %kdma.source1651, 123, !dbg !12
  %kdma.source1655 = lshr i64 %kdma.source1651, 2, !dbg !12
  %kdma.source1656 = and i64 %kdma.source1655, 4, !dbg !12
  %kdma.source1657 = add nuw nsw i64 %kdma.source1656, %kdma.source1651, !dbg !12
  %kdma.source1658 = and i64 %kdma.source1657, 4, !dbg !12
  %kdma.source1659 = or disjoint i64 %kdma.source1654, %kdma.source1658, !dbg !12
  %kdma.source1660 = shl nuw nsw i64 %kdma.source1659, 7, !dbg !12
  %kdma.source1661 = or disjoint i64 %kdma.source1653, %kdma.source1660, !dbg !12
  %kdma.source1662 = or disjoint i64 %244, %kdma.source1661, !dbg !12
  %kdma.source1663 = trunc nuw i64 %kdma.source1662 to i32, !dbg !12
  %kdma.source1664 = shl nuw nsw i32 %kdma.source1663, 1, !dbg !12
  %kdma.offset1665 = add i32 %kdma.source1664, %kdma.intra.hi, !dbg !12
  %kdma.lds1666 = add i32 %dma.buffer.lds1362, 512, !dbg !12
  %250 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1666, i32 %kdma.offset1665, <4 x i32> %dma.buffer.rsrc1361) #10, !dbg !12
  %251 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local521), !dbg !12
  %252 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !12
  %253 = insertelement <2 x i64> poison, i64 %252, i64 0, !dbg !12
  %254 = bitcast <2 x i64> %253 to <4 x i32>, !dbg !12
  %dma.buffer.rsrc1358 = shufflevector <4 x i32> %254, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !12
  %dma.buffer.lds1359 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %251), !dbg !12
  %kdma.source1667 = zext nneg i32 %kdma.owner to i64, !dbg !12
  %kdma.source1668 = or disjoint i64 %kdma.source1667, 256, !dbg !12
  %kdma.source1669 = lshr i64 %kdma.source1668, 4, !dbg !12
  %kdma.source1670 = and i64 %kdma.source1669, 24, !dbg !12
  %kdma.source1671 = and i64 %kdma.source1667, 123, !dbg !12
  %kdma.source1672 = lshr i64 %kdma.source1667, 2, !dbg !12
  %kdma.source1673 = and i64 %kdma.source1672, 4, !dbg !12
  %kdma.source1674 = add nuw nsw i64 %kdma.source1673, %kdma.source1667, !dbg !12
  %kdma.source1675 = and i64 %kdma.source1674, 4, !dbg !12
  %kdma.source1676 = or disjoint i64 %kdma.source1671, %kdma.source1675, !dbg !12
  %kdma.source1677 = shl nuw nsw i64 %kdma.source1676, 7, !dbg !12
  %kdma.source1678 = or disjoint i64 %kdma.source1670, %kdma.source1677, !dbg !12
  %kdma.source1679 = or disjoint i64 %244, %kdma.source1678, !dbg !12
  %kdma.source1680 = trunc nuw i64 %kdma.source1679 to i32, !dbg !12
  %kdma.source1681 = shl nuw nsw i32 %kdma.source1680, 1, !dbg !12
  %kdma.offset1682 = add i32 %kdma.source1681, %kdma.intra, !dbg !12
  %kdma.lds1683 = add i32 %dma.buffer.lds1359, 0, !dbg !12
  %255 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1683, i32 %kdma.offset1682, <4 x i32> %dma.buffer.rsrc1358) #10, !dbg !12
  %kdma.source1684 = zext nneg i32 %kdma.owner.hi to i64, !dbg !12
  %kdma.source1685 = or disjoint i64 %kdma.source1684, 256, !dbg !12
  %kdma.source1686 = lshr i64 %kdma.source1685, 4, !dbg !12
  %kdma.source1687 = and i64 %kdma.source1686, 24, !dbg !12
  %kdma.source1688 = and i64 %kdma.source1684, 123, !dbg !12
  %kdma.source1689 = lshr i64 %kdma.source1684, 2, !dbg !12
  %kdma.source1690 = and i64 %kdma.source1689, 4, !dbg !12
  %kdma.source1691 = add nuw nsw i64 %kdma.source1690, %kdma.source1684, !dbg !12
  %kdma.source1692 = and i64 %kdma.source1691, 4, !dbg !12
  %kdma.source1693 = or disjoint i64 %kdma.source1688, %kdma.source1692, !dbg !12
  %kdma.source1694 = shl nuw nsw i64 %kdma.source1693, 7, !dbg !12
  %kdma.source1695 = or disjoint i64 %kdma.source1687, %kdma.source1694, !dbg !12
  %kdma.source1696 = or disjoint i64 %244, %kdma.source1695, !dbg !12
  %kdma.source1697 = trunc nuw i64 %kdma.source1696 to i32, !dbg !12
  %kdma.source1698 = shl nuw nsw i32 %kdma.source1697, 1, !dbg !12
  %kdma.offset1699 = add i32 %kdma.source1698, %kdma.intra.hi, !dbg !12
  %kdma.lds1700 = add i32 %dma.buffer.lds1359, 512, !dbg !12
  %256 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1700, i32 %kdma.offset1699, <4 x i32> %dma.buffer.rsrc1358) #10, !dbg !12
  %kdma.gep1797 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8192, !dbg !14
  %257 = load <4 x half>, ptr addrspace(3) %kdma.gep1797, align 8, !dbg !14
  %258 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %257, <4 x float> %184) #9, !dbg !14
  %kdma.gep1798 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12288, !dbg !14
  %259 = load <4 x half>, ptr addrspace(3) %kdma.gep1798, align 8, !dbg !14
  %kdma.gep1799 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 8192, !dbg !14
  %260 = load <4 x half>, ptr addrspace(3) %kdma.gep1799, align 8, !dbg !14
  %261 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %259, <4 x float> %258) #9, !dbg !14
  %262 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %260, <4 x float> %188) #9, !dbg !14
  %kdma.gep1800 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 12288, !dbg !14
  %263 = load <4 x half>, ptr addrspace(3) %kdma.gep1800, align 8, !dbg !14
  %kdma.gep1801 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8320, !dbg !14
  %264 = load <4 x half>, ptr addrspace(3) %kdma.gep1801, align 8, !dbg !14
  %265 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %263, <4 x float> %262) #9, !dbg !14
  %266 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %264, <4 x float> %192) #9, !dbg !14
  %kdma.gep1802 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12416, !dbg !14
  %267 = load <4 x half>, ptr addrspace(3) %kdma.gep1802, align 8, !dbg !14
  %kdma.gep1803 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 8320, !dbg !14
  %268 = load <4 x half>, ptr addrspace(3) %kdma.gep1803, align 8, !dbg !14
  %269 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %267, <4 x float> %266) #9, !dbg !14
  %270 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %268, <4 x float> %196) #9, !dbg !14
  %kdma.gep1804 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 12416, !dbg !14
  %271 = load <4 x half>, ptr addrspace(3) %kdma.gep1804, align 8, !dbg !14
  %kdma.gep1805 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9216, !dbg !14
  %272 = load <4 x half>, ptr addrspace(3) %kdma.gep1805, align 8, !dbg !14
  %273 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %271, <4 x float> %270) #9, !dbg !14
  %274 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %272, <4 x float> %200) #9, !dbg !14
  %kdma.gep1806 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13312, !dbg !14
  %275 = load <4 x half>, ptr addrspace(3) %kdma.gep1806, align 8, !dbg !14
  %kdma.gep1807 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 9216, !dbg !14
  %276 = load <4 x half>, ptr addrspace(3) %kdma.gep1807, align 8, !dbg !14
  %277 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %275, <4 x float> %274) #9, !dbg !14
  %278 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %276, <4 x float> %204) #9, !dbg !14
  %kdma.gep1808 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 13312, !dbg !14
  %279 = load <4 x half>, ptr addrspace(3) %kdma.gep1808, align 8, !dbg !14
  %kdma.gep1809 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9344, !dbg !14
  %280 = load <4 x half>, ptr addrspace(3) %kdma.gep1809, align 8, !dbg !14
  %281 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %279, <4 x float> %278) #9, !dbg !14
  %282 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %280, <4 x float> %208) #9, !dbg !14
  %kdma.gep1810 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13440, !dbg !14
  %283 = load <4 x half>, ptr addrspace(3) %kdma.gep1810, align 8, !dbg !14
  %kdma.gep1811 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 9344, !dbg !14
  %284 = load <4 x half>, ptr addrspace(3) %kdma.gep1811, align 8, !dbg !14
  %285 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %283, <4 x float> %282) #9, !dbg !14
  %286 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %284, <4 x float> %212) #9, !dbg !14
  %kdma.gep1812 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 13440, !dbg !14
  %287 = load <4 x half>, ptr addrspace(3) %kdma.gep1812, align 8, !dbg !14
  %kdma.gep1813 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8192, !dbg !14
  %288 = load <4 x half>, ptr addrspace(3) %kdma.gep1813, align 8, !dbg !14
  %289 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %287, <4 x float> %286) #9, !dbg !14
  %290 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %288, <4 x float> %216) #9, !dbg !14
  %kdma.gep1814 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12288, !dbg !14
  %291 = load <4 x half>, ptr addrspace(3) %kdma.gep1814, align 8, !dbg !14
  %kdma.gep1815 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 8192, !dbg !14
  %292 = load <4 x half>, ptr addrspace(3) %kdma.gep1815, align 8, !dbg !14
  %293 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %291, <4 x float> %290) #9, !dbg !14
  %294 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %292, <4 x float> %220) #9, !dbg !14
  %kdma.gep1816 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 12288, !dbg !14
  %295 = load <4 x half>, ptr addrspace(3) %kdma.gep1816, align 8, !dbg !14
  %kdma.gep1817 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8320, !dbg !14
  %296 = load <4 x half>, ptr addrspace(3) %kdma.gep1817, align 8, !dbg !14
  %297 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %295, <4 x float> %294) #9, !dbg !14
  %298 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %296, <4 x float> %224) #9, !dbg !14
  %kdma.gep1818 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12416, !dbg !14
  %299 = load <4 x half>, ptr addrspace(3) %kdma.gep1818, align 8, !dbg !14
  %kdma.gep1819 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 8320, !dbg !14
  %300 = load <4 x half>, ptr addrspace(3) %kdma.gep1819, align 8, !dbg !14
  %301 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %299, <4 x float> %298) #9, !dbg !14
  %302 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %300, <4 x float> %228) #9, !dbg !14
  %kdma.gep1820 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 12416, !dbg !14
  %303 = load <4 x half>, ptr addrspace(3) %kdma.gep1820, align 8, !dbg !14
  %kdma.gep1821 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9216, !dbg !14
  %304 = load <4 x half>, ptr addrspace(3) %kdma.gep1821, align 8, !dbg !14
  %305 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %303, <4 x float> %302) #9, !dbg !14
  %306 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %304, <4 x float> %232) #9, !dbg !14
  %kdma.gep1822 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13312, !dbg !14
  %307 = load <4 x half>, ptr addrspace(3) %kdma.gep1822, align 8, !dbg !14
  %kdma.gep1823 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 9216, !dbg !14
  %308 = load <4 x half>, ptr addrspace(3) %kdma.gep1823, align 8, !dbg !14
  %309 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %307, <4 x float> %306) #9, !dbg !14
  %310 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %308, <4 x float> %236) #9, !dbg !14
  %kdma.gep1824 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 13312, !dbg !14
  %311 = load <4 x half>, ptr addrspace(3) %kdma.gep1824, align 8, !dbg !14
  %kdma.gep1825 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9344, !dbg !14
  %312 = load <4 x half>, ptr addrspace(3) %kdma.gep1825, align 8, !dbg !14
  %313 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %311, <4 x float> %310) #9, !dbg !14
  %314 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %312, <4 x float> %240) #9, !dbg !14
  %kdma.gep1826 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13440, !dbg !14
  %315 = load <4 x half>, ptr addrspace(3) %kdma.gep1826, align 8, !dbg !14
  %kdma.gep1827 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 9344, !dbg !14
  %316 = load <4 x half>, ptr addrspace(3) %kdma.gep1827, align 8, !dbg !14
  %317 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %315, <4 x float> %314) #9, !dbg !14
  %318 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %316, <4 x float> %243) #9, !dbg !14
  %kdma.gep1828 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 13440, !dbg !14
  %319 = load <4 x half>, ptr addrspace(3) %kdma.gep1828, align 8, !dbg !14
  %320 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %319, <4 x float> %318) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %321 = or disjoint i64 %134, 96, !dbg !12
  %322 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local535), !dbg !12
  %323 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !12
  %324 = insertelement <2 x i64> poison, i64 %323, i64 0, !dbg !12
  %325 = bitcast <2 x i64> %324 to <4 x i32>, !dbg !12
  %dma.buffer.rsrc1355 = shufflevector <4 x i32> %325, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !12
  %dma.buffer.lds1356 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %322), !dbg !12
  %kdma.source1701 = zext nneg i32 %kdma.owner to i64, !dbg !12
  %kdma.source1702 = lshr i64 %kdma.source1701, 4, !dbg !12
  %kdma.source1703 = and i64 %kdma.source1702, 8, !dbg !12
  %kdma.source1704 = and i64 %kdma.source1701, 123, !dbg !12
  %kdma.source1705 = lshr i64 %kdma.source1701, 2, !dbg !12
  %kdma.source1706 = and i64 %kdma.source1705, 4, !dbg !12
  %kdma.source1707 = add nuw nsw i64 %kdma.source1706, %kdma.source1701, !dbg !12
  %kdma.source1708 = and i64 %kdma.source1707, 4, !dbg !12
  %kdma.source1709 = or disjoint i64 %kdma.source1704, %kdma.source1708, !dbg !12
  %kdma.source1710 = shl nuw nsw i64 %kdma.source1709, 7, !dbg !12
  %kdma.source1711 = or disjoint i64 %kdma.source1703, %kdma.source1710, !dbg !12
  %kdma.source1712 = or disjoint i64 %321, %kdma.source1711, !dbg !12
  %kdma.source1713 = trunc nuw i64 %kdma.source1712 to i32, !dbg !12
  %kdma.source1714 = shl nuw nsw i32 %kdma.source1713, 1, !dbg !12
  %kdma.offset1715 = add i32 %kdma.source1714, %kdma.intra, !dbg !12
  %kdma.lds1716 = add i32 %dma.buffer.lds1356, 0, !dbg !12
  %326 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1716, i32 %kdma.offset1715, <4 x i32> %dma.buffer.rsrc1355) #10, !dbg !12
  %kdma.source1717 = zext nneg i32 %kdma.owner.hi to i64, !dbg !12
  %kdma.source1718 = lshr i64 %kdma.source1717, 4, !dbg !12
  %kdma.source1719 = and i64 %kdma.source1718, 8, !dbg !12
  %kdma.source1720 = and i64 %kdma.source1717, 123, !dbg !12
  %kdma.source1721 = lshr i64 %kdma.source1717, 2, !dbg !12
  %kdma.source1722 = and i64 %kdma.source1721, 4, !dbg !12
  %kdma.source1723 = add nuw nsw i64 %kdma.source1722, %kdma.source1717, !dbg !12
  %kdma.source1724 = and i64 %kdma.source1723, 4, !dbg !12
  %kdma.source1725 = or disjoint i64 %kdma.source1720, %kdma.source1724, !dbg !12
  %kdma.source1726 = shl nuw nsw i64 %kdma.source1725, 7, !dbg !12
  %kdma.source1727 = or disjoint i64 %kdma.source1719, %kdma.source1726, !dbg !12
  %kdma.source1728 = or disjoint i64 %321, %kdma.source1727, !dbg !12
  %kdma.source1729 = trunc nuw i64 %kdma.source1728 to i32, !dbg !12
  %kdma.source1730 = shl nuw nsw i32 %kdma.source1729, 1, !dbg !12
  %kdma.offset1731 = add i32 %kdma.source1730, %kdma.intra.hi, !dbg !12
  %kdma.lds1732 = add i32 %dma.buffer.lds1356, 512, !dbg !12
  %327 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1732, i32 %kdma.offset1731, <4 x i32> %dma.buffer.rsrc1355) #10, !dbg !12
  %328 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local539), !dbg !12
  %329 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !12
  %330 = insertelement <2 x i64> poison, i64 %329, i64 0, !dbg !12
  %331 = bitcast <2 x i64> %330 to <4 x i32>, !dbg !12
  %dma.buffer.rsrc1352 = shufflevector <4 x i32> %331, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !12
  %dma.buffer.lds1353 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %328), !dbg !12
  %kdma.source1733 = zext nneg i32 %kdma.owner to i64, !dbg !12
  %kdma.source1734 = or disjoint i64 %kdma.source1733, 256, !dbg !12
  %kdma.source1735 = lshr i64 %kdma.source1734, 4, !dbg !12
  %kdma.source1736 = and i64 %kdma.source1735, 24, !dbg !12
  %kdma.source1737 = and i64 %kdma.source1733, 123, !dbg !12
  %kdma.source1738 = lshr i64 %kdma.source1733, 2, !dbg !12
  %kdma.source1739 = and i64 %kdma.source1738, 4, !dbg !12
  %kdma.source1740 = add nuw nsw i64 %kdma.source1739, %kdma.source1733, !dbg !12
  %kdma.source1741 = and i64 %kdma.source1740, 4, !dbg !12
  %kdma.source1742 = or disjoint i64 %kdma.source1737, %kdma.source1741, !dbg !12
  %kdma.source1743 = shl nuw nsw i64 %kdma.source1742, 7, !dbg !12
  %kdma.source1744 = or disjoint i64 %kdma.source1736, %kdma.source1743, !dbg !12
  %kdma.source1745 = or disjoint i64 %321, %kdma.source1744, !dbg !12
  %kdma.source1746 = trunc nuw i64 %kdma.source1745 to i32, !dbg !12
  %kdma.source1747 = shl nuw nsw i32 %kdma.source1746, 1, !dbg !12
  %kdma.offset1748 = add i32 %kdma.source1747, %kdma.intra, !dbg !12
  %kdma.lds1749 = add i32 %dma.buffer.lds1353, 0, !dbg !12
  %332 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1749, i32 %kdma.offset1748, <4 x i32> %dma.buffer.rsrc1352) #10, !dbg !12
  %kdma.source1750 = zext nneg i32 %kdma.owner.hi to i64, !dbg !12
  %kdma.source1751 = or disjoint i64 %kdma.source1750, 256, !dbg !12
  %kdma.source1752 = lshr i64 %kdma.source1751, 4, !dbg !12
  %kdma.source1753 = and i64 %kdma.source1752, 24, !dbg !12
  %kdma.source1754 = and i64 %kdma.source1750, 123, !dbg !12
  %kdma.source1755 = lshr i64 %kdma.source1750, 2, !dbg !12
  %kdma.source1756 = and i64 %kdma.source1755, 4, !dbg !12
  %kdma.source1757 = add nuw nsw i64 %kdma.source1756, %kdma.source1750, !dbg !12
  %kdma.source1758 = and i64 %kdma.source1757, 4, !dbg !12
  %kdma.source1759 = or disjoint i64 %kdma.source1754, %kdma.source1758, !dbg !12
  %kdma.source1760 = shl nuw nsw i64 %kdma.source1759, 7, !dbg !12
  %kdma.source1761 = or disjoint i64 %kdma.source1753, %kdma.source1760, !dbg !12
  %kdma.source1762 = or disjoint i64 %321, %kdma.source1761, !dbg !12
  %kdma.source1763 = trunc nuw i64 %kdma.source1762 to i32, !dbg !12
  %kdma.source1764 = shl nuw nsw i32 %kdma.source1763, 1, !dbg !12
  %kdma.offset1765 = add i32 %kdma.source1764, %kdma.intra.hi, !dbg !12
  %kdma.lds1766 = add i32 %dma.buffer.lds1353, 512, !dbg !12
  %333 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1766, i32 %kdma.offset1765, <4 x i32> %dma.buffer.rsrc1352) #10, !dbg !12
  %kdma.gep1829 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 0, !dbg !14
  %334 = load <4 x half>, ptr addrspace(3) %kdma.gep1829, align 8, !dbg !14
  %335 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %334, <4 x float> %261) #9, !dbg !14
  %kdma.gep1830 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4096, !dbg !14
  %336 = load <4 x half>, ptr addrspace(3) %kdma.gep1830, align 8, !dbg !14
  %kdma.gep1831 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 0, !dbg !14
  %337 = load <4 x half>, ptr addrspace(3) %kdma.gep1831, align 8, !dbg !14
  %338 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %336, <4 x float> %335) #9, !dbg !14
  %339 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %337, <4 x float> %265) #9, !dbg !14
  %kdma.gep1832 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 4096, !dbg !14
  %340 = load <4 x half>, ptr addrspace(3) %kdma.gep1832, align 8, !dbg !14
  %kdma.gep1833 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 128, !dbg !14
  %341 = load <4 x half>, ptr addrspace(3) %kdma.gep1833, align 8, !dbg !14
  %342 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %340, <4 x float> %339) #9, !dbg !14
  %343 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %341, <4 x float> %269) #9, !dbg !14
  %kdma.gep1834 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4224, !dbg !14
  %344 = load <4 x half>, ptr addrspace(3) %kdma.gep1834, align 8, !dbg !14
  %kdma.gep1835 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 128, !dbg !14
  %345 = load <4 x half>, ptr addrspace(3) %kdma.gep1835, align 8, !dbg !14
  %346 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %344, <4 x float> %343) #9, !dbg !14
  %347 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %345, <4 x float> %273) #9, !dbg !14
  %kdma.gep1836 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 4224, !dbg !14
  %348 = load <4 x half>, ptr addrspace(3) %kdma.gep1836, align 8, !dbg !14
  %kdma.gep1837 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1024, !dbg !14
  %349 = load <4 x half>, ptr addrspace(3) %kdma.gep1837, align 8, !dbg !14
  %350 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %348, <4 x float> %347) #9, !dbg !14
  %351 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %349, <4 x float> %277) #9, !dbg !14
  %kdma.gep1838 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5120, !dbg !14
  %352 = load <4 x half>, ptr addrspace(3) %kdma.gep1838, align 8, !dbg !14
  %kdma.gep1839 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 1024, !dbg !14
  %353 = load <4 x half>, ptr addrspace(3) %kdma.gep1839, align 8, !dbg !14
  %354 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %352, <4 x float> %351) #9, !dbg !14
  %355 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %353, <4 x float> %281) #9, !dbg !14
  %kdma.gep1840 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 5120, !dbg !14
  %356 = load <4 x half>, ptr addrspace(3) %kdma.gep1840, align 8, !dbg !14
  %kdma.gep1841 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1152, !dbg !14
  %357 = load <4 x half>, ptr addrspace(3) %kdma.gep1841, align 8, !dbg !14
  %358 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %356, <4 x float> %355) #9, !dbg !14
  %359 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %357, <4 x float> %285) #9, !dbg !14
  %kdma.gep1842 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5248, !dbg !14
  %360 = load <4 x half>, ptr addrspace(3) %kdma.gep1842, align 8, !dbg !14
  %kdma.gep1843 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 1152, !dbg !14
  %361 = load <4 x half>, ptr addrspace(3) %kdma.gep1843, align 8, !dbg !14
  %362 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %360, <4 x float> %359) #9, !dbg !14
  %363 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %361, <4 x float> %289) #9, !dbg !14
  %kdma.gep1844 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 5248, !dbg !14
  %364 = load <4 x half>, ptr addrspace(3) %kdma.gep1844, align 8, !dbg !14
  %kdma.gep1845 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 0, !dbg !14
  %365 = load <4 x half>, ptr addrspace(3) %kdma.gep1845, align 8, !dbg !14
  %366 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %364, <4 x float> %363) #9, !dbg !14
  %367 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %365, <4 x float> %293) #9, !dbg !14
  %kdma.gep1846 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4096, !dbg !14
  %368 = load <4 x half>, ptr addrspace(3) %kdma.gep1846, align 8, !dbg !14
  %kdma.gep1847 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 0, !dbg !14
  %369 = load <4 x half>, ptr addrspace(3) %kdma.gep1847, align 8, !dbg !14
  %370 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %368, <4 x float> %367) #9, !dbg !14
  %371 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %369, <4 x float> %297) #9, !dbg !14
  %kdma.gep1848 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 4096, !dbg !14
  %372 = load <4 x half>, ptr addrspace(3) %kdma.gep1848, align 8, !dbg !14
  %kdma.gep1849 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 128, !dbg !14
  %373 = load <4 x half>, ptr addrspace(3) %kdma.gep1849, align 8, !dbg !14
  %374 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %372, <4 x float> %371) #9, !dbg !14
  %375 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %373, <4 x float> %301) #9, !dbg !14
  %kdma.gep1850 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4224, !dbg !14
  %376 = load <4 x half>, ptr addrspace(3) %kdma.gep1850, align 8, !dbg !14
  %kdma.gep1851 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 128, !dbg !14
  %377 = load <4 x half>, ptr addrspace(3) %kdma.gep1851, align 8, !dbg !14
  %378 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %376, <4 x float> %375) #9, !dbg !14
  %379 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %377, <4 x float> %305) #9, !dbg !14
  %kdma.gep1852 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 4224, !dbg !14
  %380 = load <4 x half>, ptr addrspace(3) %kdma.gep1852, align 8, !dbg !14
  %kdma.gep1853 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1024, !dbg !14
  %381 = load <4 x half>, ptr addrspace(3) %kdma.gep1853, align 8, !dbg !14
  %382 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %380, <4 x float> %379) #9, !dbg !14
  %383 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %381, <4 x float> %309) #9, !dbg !14
  %kdma.gep1854 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5120, !dbg !14
  %384 = load <4 x half>, ptr addrspace(3) %kdma.gep1854, align 8, !dbg !14
  %kdma.gep1855 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 1024, !dbg !14
  %385 = load <4 x half>, ptr addrspace(3) %kdma.gep1855, align 8, !dbg !14
  %386 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %384, <4 x float> %383) #9, !dbg !14
  %387 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %385, <4 x float> %313) #9, !dbg !14
  %kdma.gep1856 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 5120, !dbg !14
  %388 = load <4 x half>, ptr addrspace(3) %kdma.gep1856, align 8, !dbg !14
  %kdma.gep1857 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1152, !dbg !14
  %389 = load <4 x half>, ptr addrspace(3) %kdma.gep1857, align 8, !dbg !14
  %390 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %388, <4 x float> %387) #9, !dbg !14
  %391 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %389, <4 x float> %317) #9, !dbg !14
  %kdma.gep1858 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5248, !dbg !14
  %392 = load <4 x half>, ptr addrspace(3) %kdma.gep1858, align 8, !dbg !14
  %kdma.gep1859 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 1152, !dbg !14
  %393 = load <4 x half>, ptr addrspace(3) %kdma.gep1859, align 8, !dbg !14
  %394 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %392, <4 x float> %391) #9, !dbg !14
  %395 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %393, <4 x float> %320) #9, !dbg !14
  %kdma.gep1860 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 5248, !dbg !14
  %396 = load <4 x half>, ptr addrspace(3) %kdma.gep1860, align 8, !dbg !14
  %397 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %396, <4 x float> %395) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %398 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local504), !dbg !15
  %address.local917 = shl nuw nsw i64 %address.local499, 5, !dbg !15
  %address.local918 = and i64 %address.local917, 8064, !dbg !15
  %address.local919 = shl nuw nsw i64 %address.local499, 3, !dbg !15
  %address.local920 = and i64 %address.local919, 24, !dbg !15
  %address.local921 = or disjoint i64 %address.local918, %address.local920, !dbg !15
  %399 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %.idx14191420 = or disjoint i64 %134, %address.local921, !dbg !15
  %.idx14191420.tr = trunc nuw i64 %.idx14191420 to i32, !dbg !15
  %dma.buffer.offset1348 = shl nuw nsw i32 %.idx14191420.tr, 1, !dbg !15
  %400 = insertelement <2 x i64> poison, i64 %399, i64 0, !dbg !15
  %401 = bitcast <2 x i64> %400 to <4 x i32>, !dbg !15
  %dma.buffer.rsrc1349 = shufflevector <4 x i32> %401, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %dma.buffer.lds1350 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %398), !dbg !15
  %402 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1350, i32 %dma.buffer.offset1348, <4 x i32> %dma.buffer.rsrc1349) #10, !dbg !15
  %403 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local521), !dbg !15
  %address.local923 = shl nuw nsw i64 %address.local516, 5, !dbg !15
  %address.local924 = and i64 %address.local923, 16256, !dbg !15
  %address.local925 = shl nuw nsw i64 %address.local499, 3, !dbg !15
  %address.local926 = and i64 %address.local925, 24, !dbg !15
  %address.local927 = or disjoint i64 %address.local924, %address.local926, !dbg !15
  %404 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %.idx14211422 = or disjoint i64 %134, %address.local927, !dbg !15
  %.idx14211422.tr = trunc nuw i64 %.idx14211422 to i32, !dbg !15
  %dma.buffer.offset1345 = shl nuw nsw i32 %.idx14211422.tr, 1, !dbg !15
  %405 = insertelement <2 x i64> poison, i64 %404, i64 0, !dbg !15
  %406 = bitcast <2 x i64> %405 to <4 x i32>, !dbg !15
  %dma.buffer.rsrc1346 = shufflevector <4 x i32> %406, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %dma.buffer.lds1347 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %403), !dbg !15
  %407 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1347, i32 %dma.buffer.offset1345, <4 x i32> %dma.buffer.rsrc1346) #10, !dbg !15
  %kdma.gep1861 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8192, !dbg !14
  %408 = load <4 x half>, ptr addrspace(3) %kdma.gep1861, align 8, !dbg !14
  %409 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %108, <4 x half> %408, <4 x float> %338) #9, !dbg !14
  %kdma.gep1862 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12288, !dbg !14
  %410 = load <4 x half>, ptr addrspace(3) %kdma.gep1862, align 8, !dbg !14
  %kdma.gep1863 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 8192, !dbg !14
  %411 = load <4 x half>, ptr addrspace(3) %kdma.gep1863, align 8, !dbg !14
  %412 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %110, <4 x half> %410, <4 x float> %409) #9, !dbg !14
  %413 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %108, <4 x half> %411, <4 x float> %342) #9, !dbg !14
  %414 = extractelement <4 x float> %412, i64 0, !dbg !19
  %415 = call float @llvm.maxnum.f32(float %414, float 0xFFF0000000000000), !dbg !20, !frisk.local_reduction !1
  %416 = extractelement <4 x float> %412, i64 1, !dbg !19
  %417 = extractelement <4 x float> %412, i64 2, !dbg !19
  %418 = extractelement <4 x float> %412, i64 3, !dbg !19
  %kdma.gep1864 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 12288, !dbg !14
  %419 = load <4 x half>, ptr addrspace(3) %kdma.gep1864, align 8, !dbg !14
  %kdma.gep1865 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8320, !dbg !14
  %420 = load <4 x half>, ptr addrspace(3) %kdma.gep1865, align 8, !dbg !14
  %421 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %110, <4 x half> %419, <4 x float> %413) #9, !dbg !14
  %422 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %108, <4 x half> %420, <4 x float> %346) #9, !dbg !14
  %423 = extractelement <4 x float> %421, i64 0, !dbg !19
  %local.reduce151 = call float @llvm.maxnum.f32(float %415, float %423), !dbg !20, !frisk.local_reduction !1
  %424 = extractelement <4 x float> %421, i64 1, !dbg !19
  %local.reduce152 = call float @llvm.maxnum.f32(float %416, float %424), !dbg !20, !frisk.local_reduction !1
  %425 = extractelement <4 x float> %421, i64 2, !dbg !19
  %local.reduce153 = call float @llvm.maxnum.f32(float %417, float %425), !dbg !20, !frisk.local_reduction !1
  %426 = extractelement <4 x float> %421, i64 3, !dbg !19
  %local.reduce154 = call float @llvm.maxnum.f32(float %418, float %426), !dbg !20, !frisk.local_reduction !1
  %kdma.gep1866 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12416, !dbg !14
  %427 = load <4 x half>, ptr addrspace(3) %kdma.gep1866, align 8, !dbg !14
  %kdma.gep1867 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 8320, !dbg !14
  %428 = load <4 x half>, ptr addrspace(3) %kdma.gep1867, align 8, !dbg !14
  %429 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %110, <4 x half> %427, <4 x float> %422) #9, !dbg !14
  %430 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %108, <4 x half> %428, <4 x float> %350) #9, !dbg !14
  %431 = extractelement <4 x float> %429, i64 0, !dbg !19
  %local.reduce155 = call float @llvm.maxnum.f32(float %local.reduce151, float %431), !dbg !20, !frisk.local_reduction !1
  %432 = extractelement <4 x float> %429, i64 1, !dbg !19
  %local.reduce156 = call float @llvm.maxnum.f32(float %local.reduce152, float %432), !dbg !20, !frisk.local_reduction !1
  %433 = extractelement <4 x float> %429, i64 2, !dbg !19
  %local.reduce157 = call float @llvm.maxnum.f32(float %local.reduce153, float %433), !dbg !20, !frisk.local_reduction !1
  %434 = extractelement <4 x float> %429, i64 3, !dbg !19
  %local.reduce158 = call float @llvm.maxnum.f32(float %local.reduce154, float %434), !dbg !20, !frisk.local_reduction !1
  %kdma.gep1868 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 12416, !dbg !14
  %435 = load <4 x half>, ptr addrspace(3) %kdma.gep1868, align 8, !dbg !14
  %kdma.gep1869 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9216, !dbg !14
  %436 = load <4 x half>, ptr addrspace(3) %kdma.gep1869, align 8, !dbg !14
  %437 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %110, <4 x half> %435, <4 x float> %430) #9, !dbg !14
  %438 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %108, <4 x half> %436, <4 x float> %354) #9, !dbg !14
  %439 = extractelement <4 x float> %437, i64 0, !dbg !19
  %local.reduce159 = call float @llvm.maxnum.f32(float %local.reduce155, float %439), !dbg !20, !frisk.local_reduction !1
  %440 = extractelement <4 x float> %437, i64 1, !dbg !19
  %local.reduce160 = call float @llvm.maxnum.f32(float %local.reduce156, float %440), !dbg !20, !frisk.local_reduction !1
  %441 = extractelement <4 x float> %437, i64 2, !dbg !19
  %local.reduce161 = call float @llvm.maxnum.f32(float %local.reduce157, float %441), !dbg !20, !frisk.local_reduction !1
  %442 = extractelement <4 x float> %437, i64 3, !dbg !19
  %local.reduce162 = call float @llvm.maxnum.f32(float %local.reduce158, float %442), !dbg !20, !frisk.local_reduction !1
  %kdma.gep1870 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13312, !dbg !14
  %443 = load <4 x half>, ptr addrspace(3) %kdma.gep1870, align 8, !dbg !14
  %kdma.gep1871 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 9216, !dbg !14
  %444 = load <4 x half>, ptr addrspace(3) %kdma.gep1871, align 8, !dbg !14
  %445 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %110, <4 x half> %443, <4 x float> %438) #9, !dbg !14
  %446 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %108, <4 x half> %444, <4 x float> %358) #9, !dbg !14
  %447 = extractelement <4 x float> %445, i64 0, !dbg !19
  %local.reduce163 = call float @llvm.maxnum.f32(float %local.reduce159, float %447), !dbg !20, !frisk.local_reduction !1
  %448 = extractelement <4 x float> %445, i64 1, !dbg !19
  %local.reduce164 = call float @llvm.maxnum.f32(float %local.reduce160, float %448), !dbg !20, !frisk.local_reduction !1
  %449 = extractelement <4 x float> %445, i64 2, !dbg !19
  %local.reduce165 = call float @llvm.maxnum.f32(float %local.reduce161, float %449), !dbg !20, !frisk.local_reduction !1
  %450 = extractelement <4 x float> %445, i64 3, !dbg !19
  %local.reduce166 = call float @llvm.maxnum.f32(float %local.reduce162, float %450), !dbg !20, !frisk.local_reduction !1
  %kdma.gep1872 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 13312, !dbg !14
  %451 = load <4 x half>, ptr addrspace(3) %kdma.gep1872, align 8, !dbg !14
  %kdma.gep1873 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9344, !dbg !14
  %452 = load <4 x half>, ptr addrspace(3) %kdma.gep1873, align 8, !dbg !14
  %453 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %110, <4 x half> %451, <4 x float> %446) #9, !dbg !14
  %454 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %108, <4 x half> %452, <4 x float> %362) #9, !dbg !14
  %455 = extractelement <4 x float> %453, i64 0, !dbg !19
  %local.reduce167 = call float @llvm.maxnum.f32(float %local.reduce163, float %455), !dbg !20, !frisk.local_reduction !1
  %456 = extractelement <4 x float> %453, i64 1, !dbg !19
  %local.reduce168 = call float @llvm.maxnum.f32(float %local.reduce164, float %456), !dbg !20, !frisk.local_reduction !1
  %457 = extractelement <4 x float> %453, i64 2, !dbg !19
  %local.reduce169 = call float @llvm.maxnum.f32(float %local.reduce165, float %457), !dbg !20, !frisk.local_reduction !1
  %458 = extractelement <4 x float> %453, i64 3, !dbg !19
  %local.reduce170 = call float @llvm.maxnum.f32(float %local.reduce166, float %458), !dbg !20, !frisk.local_reduction !1
  %kdma.gep1874 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13440, !dbg !14
  %459 = load <4 x half>, ptr addrspace(3) %kdma.gep1874, align 8, !dbg !14
  %kdma.gep1875 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 9344, !dbg !14
  %460 = load <4 x half>, ptr addrspace(3) %kdma.gep1875, align 8, !dbg !14
  %461 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %110, <4 x half> %459, <4 x float> %454) #9, !dbg !14
  %462 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %108, <4 x half> %460, <4 x float> %366) #9, !dbg !14
  %463 = extractelement <4 x float> %461, i64 0, !dbg !19
  %local.reduce171 = call float @llvm.maxnum.f32(float %local.reduce167, float %463), !dbg !20, !frisk.local_reduction !1
  %464 = extractelement <4 x float> %461, i64 1, !dbg !19
  %local.reduce172 = call float @llvm.maxnum.f32(float %local.reduce168, float %464), !dbg !20, !frisk.local_reduction !1
  %465 = extractelement <4 x float> %461, i64 2, !dbg !19
  %local.reduce173 = call float @llvm.maxnum.f32(float %local.reduce169, float %465), !dbg !20, !frisk.local_reduction !1
  %466 = extractelement <4 x float> %461, i64 3, !dbg !19
  %local.reduce174 = call float @llvm.maxnum.f32(float %local.reduce170, float %466), !dbg !20, !frisk.local_reduction !1
  %kdma.gep1876 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 13440, !dbg !14
  %467 = load <4 x half>, ptr addrspace(3) %kdma.gep1876, align 8, !dbg !14
  %kdma.gep1877 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8192, !dbg !14
  %468 = load <4 x half>, ptr addrspace(3) %kdma.gep1877, align 8, !dbg !14
  %469 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %110, <4 x half> %467, <4 x float> %462) #9, !dbg !14
  %470 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %111, <4 x half> %468, <4 x float> %370) #9, !dbg !14
  %471 = extractelement <4 x float> %469, i64 0, !dbg !19
  %local.reduce175 = call float @llvm.maxnum.f32(float %local.reduce171, float %471), !dbg !20, !frisk.local_reduction !1
  %472 = extractelement <4 x float> %469, i64 1, !dbg !19
  %local.reduce176 = call float @llvm.maxnum.f32(float %local.reduce172, float %472), !dbg !20, !frisk.local_reduction !1
  %473 = extractelement <4 x float> %469, i64 2, !dbg !19
  %local.reduce177 = call float @llvm.maxnum.f32(float %local.reduce173, float %473), !dbg !20, !frisk.local_reduction !1
  %474 = extractelement <4 x float> %469, i64 3, !dbg !19
  %local.reduce178 = call float @llvm.maxnum.f32(float %local.reduce174, float %474), !dbg !20, !frisk.local_reduction !1
  %local.reduce179 = call float @llvm.maxnum.f32(float %local.reduce175, float %local.reduce176), !dbg !20, !frisk.local_reduction !1
  %local.reduce180 = call float @llvm.maxnum.f32(float %local.reduce177, float %local.reduce178), !dbg !20, !frisk.local_reduction !1
  %475 = call float @llvm.maxnum.f32(float %local.reduce179, float %local.reduce180), !dbg !20, !frisk.local_reduction !1
  %kdma.gep1878 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12288, !dbg !14
  %476 = load <4 x half>, ptr addrspace(3) %kdma.gep1878, align 8, !dbg !14
  %kdma.gep1879 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 8192, !dbg !14
  %477 = load <4 x half>, ptr addrspace(3) %kdma.gep1879, align 8, !dbg !14
  %478 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %112, <4 x half> %476, <4 x float> %470) #9, !dbg !14
  %479 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %111, <4 x half> %477, <4 x float> %374) #9, !dbg !14
  %480 = extractelement <4 x float> %478, i64 0, !dbg !19
  %481 = extractelement <4 x float> %478, i64 1, !dbg !19
  %482 = extractelement <4 x float> %478, i64 2, !dbg !19
  %483 = extractelement <4 x float> %478, i64 3, !dbg !19
  %kdma.gep1880 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 12288, !dbg !14
  %484 = load <4 x half>, ptr addrspace(3) %kdma.gep1880, align 8, !dbg !14
  %kdma.gep1881 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8320, !dbg !14
  %485 = load <4 x half>, ptr addrspace(3) %kdma.gep1881, align 8, !dbg !14
  %486 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %112, <4 x half> %484, <4 x float> %479) #9, !dbg !14
  %487 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %111, <4 x half> %485, <4 x float> %378) #9, !dbg !14
  %488 = extractelement <4 x float> %486, i64 0, !dbg !19
  %489 = extractelement <4 x float> %486, i64 1, !dbg !19
  %490 = extractelement <4 x float> %486, i64 2, !dbg !19
  %491 = extractelement <4 x float> %486, i64 3, !dbg !19
  %kdma.gep1882 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12416, !dbg !14
  %492 = load <4 x half>, ptr addrspace(3) %kdma.gep1882, align 8, !dbg !14
  %kdma.gep1883 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 8320, !dbg !14
  %493 = load <4 x half>, ptr addrspace(3) %kdma.gep1883, align 8, !dbg !14
  %494 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %112, <4 x half> %492, <4 x float> %487) #9, !dbg !14
  %495 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %111, <4 x half> %493, <4 x float> %382) #9, !dbg !14
  %496 = extractelement <4 x float> %494, i64 0, !dbg !19
  %497 = extractelement <4 x float> %494, i64 1, !dbg !19
  %498 = extractelement <4 x float> %494, i64 2, !dbg !19
  %499 = extractelement <4 x float> %494, i64 3, !dbg !19
  %kdma.gep1884 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 12416, !dbg !14
  %500 = load <4 x half>, ptr addrspace(3) %kdma.gep1884, align 8, !dbg !14
  %kdma.gep1885 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9216, !dbg !14
  %501 = load <4 x half>, ptr addrspace(3) %kdma.gep1885, align 8, !dbg !14
  %502 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %112, <4 x half> %500, <4 x float> %495) #9, !dbg !14
  %503 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %111, <4 x half> %501, <4 x float> %386) #9, !dbg !14
  %504 = extractelement <4 x float> %502, i64 0, !dbg !19
  %505 = extractelement <4 x float> %502, i64 1, !dbg !19
  %506 = extractelement <4 x float> %502, i64 2, !dbg !19
  %507 = extractelement <4 x float> %502, i64 3, !dbg !19
  %kdma.gep1886 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13312, !dbg !14
  %508 = load <4 x half>, ptr addrspace(3) %kdma.gep1886, align 8, !dbg !14
  %kdma.gep1887 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 9216, !dbg !14
  %509 = load <4 x half>, ptr addrspace(3) %kdma.gep1887, align 8, !dbg !14
  %510 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %112, <4 x half> %508, <4 x float> %503) #9, !dbg !14
  %511 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %111, <4 x half> %509, <4 x float> %390) #9, !dbg !14
  %512 = extractelement <4 x float> %510, i64 0, !dbg !19
  %513 = extractelement <4 x float> %510, i64 1, !dbg !19
  %514 = extractelement <4 x float> %510, i64 2, !dbg !19
  %515 = extractelement <4 x float> %510, i64 3, !dbg !19
  %kdma.gep1888 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 13312, !dbg !14
  %516 = load <4 x half>, ptr addrspace(3) %kdma.gep1888, align 8, !dbg !14
  %kdma.gep1889 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9344, !dbg !14
  %517 = load <4 x half>, ptr addrspace(3) %kdma.gep1889, align 8, !dbg !14
  %518 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %112, <4 x half> %516, <4 x float> %511) #9, !dbg !14
  %519 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %111, <4 x half> %517, <4 x float> %394) #9, !dbg !14
  %520 = extractelement <4 x float> %518, i64 0, !dbg !19
  %521 = extractelement <4 x float> %518, i64 1, !dbg !19
  %522 = extractelement <4 x float> %518, i64 2, !dbg !19
  %523 = extractelement <4 x float> %518, i64 3, !dbg !19
  %kdma.gep1890 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13440, !dbg !14
  %524 = load <4 x half>, ptr addrspace(3) %kdma.gep1890, align 8, !dbg !14
  %kdma.gep1891 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 9344, !dbg !14
  %525 = load <4 x half>, ptr addrspace(3) %kdma.gep1891, align 8, !dbg !14
  %526 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %112, <4 x half> %524, <4 x float> %519) #9, !dbg !14
  %527 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %111, <4 x half> %525, <4 x float> %397) #9, !dbg !14
  %528 = extractelement <4 x float> %526, i64 0, !dbg !19
  %529 = extractelement <4 x float> %526, i64 1, !dbg !19
  %530 = extractelement <4 x float> %526, i64 2, !dbg !19
  %531 = extractelement <4 x float> %526, i64 3, !dbg !19
  %kdma.gep1892 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 13440, !dbg !14
  %532 = load <4 x half>, ptr addrspace(3) %kdma.gep1892, align 8, !dbg !14
  %533 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %112, <4 x half> %532, <4 x float> %527) #9, !dbg !14
  %534 = extractelement <4 x float> %533, i64 0, !dbg !19
  %535 = extractelement <4 x float> %533, i64 1, !dbg !19
  %536 = extractelement <4 x float> %533, i64 2, !dbg !19
  %537 = extractelement <4 x float> %533, i64 3, !dbg !19
  %538 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !20
  %539 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %538), !dbg !20
  %540 = add nuw nsw i32 %539, 64, !dbg !20
  %541 = xor i32 %539, 16, !dbg !20
  %542 = and i32 %540, 192, !dbg !20
  %543 = icmp samesign ult i32 %541, %542, !dbg !20
  %544 = select i1 %543, i32 %541, i32 %539, !dbg !20
  %545 = shl nuw nsw i32 %544, 2, !dbg !20
  %546 = bitcast float %475 to i32, !dbg !20
  %547 = call i32 @llvm.amdgcn.ds.bpermute(i32 %545, i32 %546), !dbg !20
  %548 = bitcast i32 %547 to float, !dbg !20
  %549 = call float @llvm.maxnum.f32(float %475, float %548), !dbg !20
  %550 = xor i32 %539, 32, !dbg !20
  %551 = icmp samesign ult i32 %550, %542, !dbg !20
  %552 = select i1 %551, i32 %550, i32 %539, !dbg !20
  %553 = shl nuw nsw i32 %552, 2, !dbg !20
  %554 = bitcast float %549 to i32, !dbg !20
  %555 = call i32 @llvm.amdgcn.ds.bpermute(i32 %553, i32 %554), !dbg !20
  %556 = bitcast i32 %555 to float, !dbg !20
  %557 = call float @llvm.maxnum.f32(float %549, float %556), !dbg !20
  %558 = extractvalue [2 x <1 x float>] %125, 0, !dbg !20
  %559 = insertelement <1 x float> poison, float %557, i64 0, !dbg !20
  %560 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %559, <1 x float> %558), !dbg !20
  %561 = call float @llvm.maxnum.f32(float %480, float 0xFFF0000000000000), !dbg !20, !frisk.local_reduction !1
  %local.reduce120 = call float @llvm.maxnum.f32(float %561, float %488), !dbg !20, !frisk.local_reduction !1
  %local.reduce121 = call float @llvm.maxnum.f32(float %481, float %489), !dbg !20, !frisk.local_reduction !1
  %local.reduce122 = call float @llvm.maxnum.f32(float %482, float %490), !dbg !20, !frisk.local_reduction !1
  %local.reduce123 = call float @llvm.maxnum.f32(float %483, float %491), !dbg !20, !frisk.local_reduction !1
  %local.reduce124 = call float @llvm.maxnum.f32(float %local.reduce120, float %496), !dbg !20, !frisk.local_reduction !1
  %local.reduce125 = call float @llvm.maxnum.f32(float %local.reduce121, float %497), !dbg !20, !frisk.local_reduction !1
  %local.reduce126 = call float @llvm.maxnum.f32(float %local.reduce122, float %498), !dbg !20, !frisk.local_reduction !1
  %local.reduce127 = call float @llvm.maxnum.f32(float %local.reduce123, float %499), !dbg !20, !frisk.local_reduction !1
  %local.reduce128 = call float @llvm.maxnum.f32(float %local.reduce124, float %504), !dbg !20, !frisk.local_reduction !1
  %local.reduce129 = call float @llvm.maxnum.f32(float %local.reduce125, float %505), !dbg !20, !frisk.local_reduction !1
  %local.reduce130 = call float @llvm.maxnum.f32(float %local.reduce126, float %506), !dbg !20, !frisk.local_reduction !1
  %local.reduce131 = call float @llvm.maxnum.f32(float %local.reduce127, float %507), !dbg !20, !frisk.local_reduction !1
  %local.reduce132 = call float @llvm.maxnum.f32(float %local.reduce128, float %512), !dbg !20, !frisk.local_reduction !1
  %local.reduce133 = call float @llvm.maxnum.f32(float %local.reduce129, float %513), !dbg !20, !frisk.local_reduction !1
  %local.reduce134 = call float @llvm.maxnum.f32(float %local.reduce130, float %514), !dbg !20, !frisk.local_reduction !1
  %local.reduce135 = call float @llvm.maxnum.f32(float %local.reduce131, float %515), !dbg !20, !frisk.local_reduction !1
  %local.reduce136 = call float @llvm.maxnum.f32(float %local.reduce132, float %520), !dbg !20, !frisk.local_reduction !1
  %local.reduce137 = call float @llvm.maxnum.f32(float %local.reduce133, float %521), !dbg !20, !frisk.local_reduction !1
  %local.reduce138 = call float @llvm.maxnum.f32(float %local.reduce134, float %522), !dbg !20, !frisk.local_reduction !1
  %local.reduce139 = call float @llvm.maxnum.f32(float %local.reduce135, float %523), !dbg !20, !frisk.local_reduction !1
  %local.reduce140 = call float @llvm.maxnum.f32(float %local.reduce136, float %528), !dbg !20, !frisk.local_reduction !1
  %local.reduce141 = call float @llvm.maxnum.f32(float %local.reduce137, float %529), !dbg !20, !frisk.local_reduction !1
  %local.reduce142 = call float @llvm.maxnum.f32(float %local.reduce138, float %530), !dbg !20, !frisk.local_reduction !1
  %local.reduce143 = call float @llvm.maxnum.f32(float %local.reduce139, float %531), !dbg !20, !frisk.local_reduction !1
  %local.reduce144 = call float @llvm.maxnum.f32(float %local.reduce140, float %534), !dbg !20, !frisk.local_reduction !1
  %local.reduce145 = call float @llvm.maxnum.f32(float %local.reduce141, float %535), !dbg !20, !frisk.local_reduction !1
  %local.reduce146 = call float @llvm.maxnum.f32(float %local.reduce142, float %536), !dbg !20, !frisk.local_reduction !1
  %local.reduce147 = call float @llvm.maxnum.f32(float %local.reduce143, float %537), !dbg !20, !frisk.local_reduction !1
  %local.reduce148 = call float @llvm.maxnum.f32(float %local.reduce144, float %local.reduce145), !dbg !20, !frisk.local_reduction !1
  %local.reduce149 = call float @llvm.maxnum.f32(float %local.reduce146, float %local.reduce147), !dbg !20, !frisk.local_reduction !1
  %562 = call float @llvm.maxnum.f32(float %local.reduce148, float %local.reduce149), !dbg !20, !frisk.local_reduction !1
  %563 = bitcast float %562 to i32, !dbg !20
  %564 = call i32 @llvm.amdgcn.ds.bpermute(i32 %545, i32 %563), !dbg !20
  %565 = bitcast i32 %564 to float, !dbg !20
  %566 = call float @llvm.maxnum.f32(float %562, float %565), !dbg !20
  %567 = bitcast float %566 to i32, !dbg !20
  %568 = call i32 @llvm.amdgcn.ds.bpermute(i32 %553, i32 %567), !dbg !20
  %569 = bitcast i32 %568 to float, !dbg !20
  %570 = call float @llvm.maxnum.f32(float %566, float %569), !dbg !20
  %571 = extractvalue [2 x <1 x float>] %125, 1, !dbg !20
  %572 = insertelement <1 x float> poison, float %570, i64 0, !dbg !20
  %573 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %572, <1 x float> %571), !dbg !20
  %574 = shufflevector <1 x float> %560, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !21
  %575 = fsub <4 x float> %412, %574, !dbg !21
  %576 = fsub <4 x float> %421, %574, !dbg !21
  %577 = fsub <4 x float> %429, %574, !dbg !21
  %578 = fsub <4 x float> %437, %574, !dbg !21
  %579 = fsub <4 x float> %445, %574, !dbg !21
  %580 = fsub <4 x float> %453, %574, !dbg !21
  %581 = fsub <4 x float> %461, %574, !dbg !21
  %582 = fsub <4 x float> %469, %574, !dbg !21
  %583 = shufflevector <1 x float> %573, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !21
  %584 = fsub <4 x float> %478, %583, !dbg !21
  %585 = fsub <4 x float> %486, %583, !dbg !21
  %586 = fsub <4 x float> %494, %583, !dbg !21
  %587 = fsub <4 x float> %502, %583, !dbg !21
  %588 = fsub <4 x float> %510, %583, !dbg !21
  %589 = fsub <4 x float> %518, %583, !dbg !21
  %590 = fsub <4 x float> %526, %583, !dbg !21
  %591 = fsub <4 x float> %533, %583, !dbg !21
  %592 = fsub <1 x float> %558, %560, !dbg !22
  %593 = extractelement <1 x float> %592, i64 0, !dbg !23
  %594 = call float @llvm.amdgcn.exp2.f32(float %593), !dbg !23
  %595 = insertelement <4 x float> poison, float %594, i64 0, !dbg !28
  %596 = shufflevector <4 x float> %595, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !28
  %597 = extractvalue [2 x <1 x float>] %124, 0, !dbg !24
  %598 = extractelement <1 x float> %597, i64 0, !dbg !24
  %599 = fmul float %598, %594, !dbg !24
  %600 = fsub <1 x float> %571, %573, !dbg !22
  %601 = extractelement <1 x float> %600, i64 0, !dbg !23
  %602 = call float @llvm.amdgcn.exp2.f32(float %601), !dbg !23
  %603 = insertelement <4 x float> poison, float %602, i64 0, !dbg !28
  %604 = shufflevector <4 x float> %603, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !28
  %605 = extractvalue [2 x <1 x float>] %124, 1, !dbg !24
  %606 = extractelement <1 x float> %605, i64 0, !dbg !24
  %607 = fmul float %606, %602, !dbg !24
  %608 = extractelement <4 x float> %575, i64 0, !dbg !25
  %609 = call float @llvm.amdgcn.exp2.f32(float %608), !dbg !25
  %610 = insertelement <4 x float> poison, float %609, i64 0, !dbg !27
  %611 = extractelement <4 x float> %575, i64 1, !dbg !25
  %612 = call float @llvm.amdgcn.exp2.f32(float %611), !dbg !25
  %613 = insertelement <4 x float> %610, float %612, i64 1, !dbg !27
  %614 = extractelement <4 x float> %575, i64 2, !dbg !25
  %615 = call float @llvm.amdgcn.exp2.f32(float %614), !dbg !25
  %616 = insertelement <4 x float> %613, float %615, i64 2, !dbg !27
  %617 = extractelement <4 x float> %575, i64 3, !dbg !25
  %618 = call float @llvm.amdgcn.exp2.f32(float %617), !dbg !25
  %619 = insertelement <4 x float> %616, float %618, i64 3, !dbg !27
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %619) #9, !dbg !27, !frisk.register_lifetimes !1
  %620 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !27
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %620) #9, !dbg !25, !frisk.register_lifetimes !1
  %621 = extractelement <4 x float> %576, i64 0, !dbg !25
  %622 = call float @llvm.amdgcn.exp2.f32(float %621), !dbg !25
  %623 = insertelement <4 x float> poison, float %622, i64 0, !dbg !27
  %local.reduce89 = fadd float %609, %622, !dbg !26, !frisk.local_reduction !1
  %624 = extractelement <4 x float> %576, i64 1, !dbg !25
  %625 = call float @llvm.amdgcn.exp2.f32(float %624), !dbg !25
  %626 = insertelement <4 x float> %623, float %625, i64 1, !dbg !27
  %local.reduce90 = fadd float %612, %625, !dbg !26, !frisk.local_reduction !1
  %627 = extractelement <4 x float> %576, i64 2, !dbg !25
  %628 = call float @llvm.amdgcn.exp2.f32(float %627), !dbg !25
  %629 = insertelement <4 x float> %626, float %628, i64 2, !dbg !27
  %local.reduce91 = fadd float %615, %628, !dbg !26, !frisk.local_reduction !1
  %630 = extractelement <4 x float> %576, i64 3, !dbg !25
  %631 = call float @llvm.amdgcn.exp2.f32(float %630), !dbg !25
  %632 = insertelement <4 x float> %629, float %631, i64 3, !dbg !27
  %math.result374 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %632) #9, !dbg !27, !frisk.register_lifetimes !1
  %633 = fptrunc <4 x float> %math.result374 to <4 x half>, !dbg !27
  %math.packed375 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %633) #9, !dbg !26, !frisk.register_lifetimes !1
  %local.reduce92 = fadd float %618, %631, !dbg !26, !frisk.local_reduction !1
  %634 = extractelement <4 x float> %577, i64 0, !dbg !25
  %635 = call float @llvm.amdgcn.exp2.f32(float %634), !dbg !25
  %636 = insertelement <4 x float> poison, float %635, i64 0, !dbg !27
  %local.reduce93 = fadd float %local.reduce89, %635, !dbg !26, !frisk.local_reduction !1
  %637 = extractelement <4 x float> %577, i64 1, !dbg !25
  %638 = call float @llvm.amdgcn.exp2.f32(float %637), !dbg !25
  %639 = insertelement <4 x float> %636, float %638, i64 1, !dbg !27
  %local.reduce94 = fadd float %local.reduce90, %638, !dbg !26, !frisk.local_reduction !1
  %640 = extractelement <4 x float> %577, i64 2, !dbg !25
  %641 = call float @llvm.amdgcn.exp2.f32(float %640), !dbg !25
  %642 = insertelement <4 x float> %639, float %641, i64 2, !dbg !27
  %local.reduce95 = fadd float %local.reduce91, %641, !dbg !26, !frisk.local_reduction !1
  %643 = extractelement <4 x float> %577, i64 3, !dbg !25
  %644 = call float @llvm.amdgcn.exp2.f32(float %643), !dbg !25
  %645 = insertelement <4 x float> %642, float %644, i64 3, !dbg !27
  %math.result376 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %645) #9, !dbg !27, !frisk.register_lifetimes !1
  %646 = fptrunc <4 x float> %math.result376 to <4 x half>, !dbg !27
  %math.packed377 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %646) #9, !dbg !26, !frisk.register_lifetimes !1
  %local.reduce96 = fadd float %local.reduce92, %644, !dbg !26, !frisk.local_reduction !1
  %647 = extractelement <4 x float> %578, i64 0, !dbg !25
  %648 = call float @llvm.amdgcn.exp2.f32(float %647), !dbg !25
  %649 = insertelement <4 x float> poison, float %648, i64 0, !dbg !27
  %local.reduce97 = fadd float %local.reduce93, %648, !dbg !26, !frisk.local_reduction !1
  %650 = extractelement <4 x float> %578, i64 1, !dbg !25
  %651 = call float @llvm.amdgcn.exp2.f32(float %650), !dbg !25
  %652 = insertelement <4 x float> %649, float %651, i64 1, !dbg !27
  %local.reduce98 = fadd float %local.reduce94, %651, !dbg !26, !frisk.local_reduction !1
  %653 = extractelement <4 x float> %578, i64 2, !dbg !25
  %654 = call float @llvm.amdgcn.exp2.f32(float %653), !dbg !25
  %655 = insertelement <4 x float> %652, float %654, i64 2, !dbg !27
  %local.reduce99 = fadd float %local.reduce95, %654, !dbg !26, !frisk.local_reduction !1
  %656 = extractelement <4 x float> %578, i64 3, !dbg !25
  %657 = call float @llvm.amdgcn.exp2.f32(float %656), !dbg !25
  %658 = insertelement <4 x float> %655, float %657, i64 3, !dbg !27
  %math.result378 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %658) #9, !dbg !27, !frisk.register_lifetimes !1
  %659 = fptrunc <4 x float> %math.result378 to <4 x half>, !dbg !27
  %math.packed379 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %659) #9, !dbg !26, !frisk.register_lifetimes !1
  %local.reduce100 = fadd float %local.reduce96, %657, !dbg !26, !frisk.local_reduction !1
  %660 = extractelement <4 x float> %579, i64 0, !dbg !25
  %661 = call float @llvm.amdgcn.exp2.f32(float %660), !dbg !25
  %662 = insertelement <4 x float> poison, float %661, i64 0, !dbg !27
  %local.reduce101 = fadd float %local.reduce97, %661, !dbg !26, !frisk.local_reduction !1
  %663 = extractelement <4 x float> %579, i64 1, !dbg !25
  %664 = call float @llvm.amdgcn.exp2.f32(float %663), !dbg !25
  %665 = insertelement <4 x float> %662, float %664, i64 1, !dbg !27
  %local.reduce102 = fadd float %local.reduce98, %664, !dbg !26, !frisk.local_reduction !1
  %666 = extractelement <4 x float> %579, i64 2, !dbg !25
  %667 = call float @llvm.amdgcn.exp2.f32(float %666), !dbg !25
  %668 = insertelement <4 x float> %665, float %667, i64 2, !dbg !27
  %local.reduce103 = fadd float %local.reduce99, %667, !dbg !26, !frisk.local_reduction !1
  %669 = extractelement <4 x float> %579, i64 3, !dbg !25
  %670 = call float @llvm.amdgcn.exp2.f32(float %669), !dbg !25
  %671 = insertelement <4 x float> %668, float %670, i64 3, !dbg !27
  %math.result380 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %671) #9, !dbg !27, !frisk.register_lifetimes !1
  %672 = fptrunc <4 x float> %math.result380 to <4 x half>, !dbg !27
  %math.packed381 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %672) #9, !dbg !26, !frisk.register_lifetimes !1
  %local.reduce104 = fadd float %local.reduce100, %670, !dbg !26, !frisk.local_reduction !1
  %673 = extractelement <4 x float> %580, i64 0, !dbg !25
  %674 = call float @llvm.amdgcn.exp2.f32(float %673), !dbg !25
  %675 = insertelement <4 x float> poison, float %674, i64 0, !dbg !27
  %local.reduce105 = fadd float %local.reduce101, %674, !dbg !26, !frisk.local_reduction !1
  %676 = extractelement <4 x float> %580, i64 1, !dbg !25
  %677 = call float @llvm.amdgcn.exp2.f32(float %676), !dbg !25
  %678 = insertelement <4 x float> %675, float %677, i64 1, !dbg !27
  %local.reduce106 = fadd float %local.reduce102, %677, !dbg !26, !frisk.local_reduction !1
  %679 = extractelement <4 x float> %580, i64 2, !dbg !25
  %680 = call float @llvm.amdgcn.exp2.f32(float %679), !dbg !25
  %681 = insertelement <4 x float> %678, float %680, i64 2, !dbg !27
  %local.reduce107 = fadd float %local.reduce103, %680, !dbg !26, !frisk.local_reduction !1
  %682 = extractelement <4 x float> %580, i64 3, !dbg !25
  %683 = call float @llvm.amdgcn.exp2.f32(float %682), !dbg !25
  %684 = insertelement <4 x float> %681, float %683, i64 3, !dbg !27
  %math.result382 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %684) #9, !dbg !27, !frisk.register_lifetimes !1
  %685 = fptrunc <4 x float> %math.result382 to <4 x half>, !dbg !27
  %math.packed383 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %685) #9, !dbg !26, !frisk.register_lifetimes !1
  %local.reduce108 = fadd float %local.reduce104, %683, !dbg !26, !frisk.local_reduction !1
  %686 = extractelement <4 x float> %581, i64 0, !dbg !25
  %687 = call float @llvm.amdgcn.exp2.f32(float %686), !dbg !25
  %688 = insertelement <4 x float> poison, float %687, i64 0, !dbg !27
  %local.reduce109 = fadd float %local.reduce105, %687, !dbg !26, !frisk.local_reduction !1
  %689 = extractelement <4 x float> %581, i64 1, !dbg !25
  %690 = call float @llvm.amdgcn.exp2.f32(float %689), !dbg !25
  %691 = insertelement <4 x float> %688, float %690, i64 1, !dbg !27
  %local.reduce110 = fadd float %local.reduce106, %690, !dbg !26, !frisk.local_reduction !1
  %692 = extractelement <4 x float> %581, i64 2, !dbg !25
  %693 = call float @llvm.amdgcn.exp2.f32(float %692), !dbg !25
  %694 = insertelement <4 x float> %691, float %693, i64 2, !dbg !27
  %local.reduce111 = fadd float %local.reduce107, %693, !dbg !26, !frisk.local_reduction !1
  %695 = extractelement <4 x float> %581, i64 3, !dbg !25
  %696 = call float @llvm.amdgcn.exp2.f32(float %695), !dbg !25
  %697 = insertelement <4 x float> %694, float %696, i64 3, !dbg !27
  %math.result384 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %697) #9, !dbg !27, !frisk.register_lifetimes !1
  %698 = fptrunc <4 x float> %math.result384 to <4 x half>, !dbg !27
  %math.packed385 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %698) #9, !dbg !26, !frisk.register_lifetimes !1
  %local.reduce112 = fadd float %local.reduce108, %696, !dbg !26, !frisk.local_reduction !1
  %699 = extractelement <4 x float> %582, i64 0, !dbg !25
  %700 = call float @llvm.amdgcn.exp2.f32(float %699), !dbg !25
  %701 = insertelement <4 x float> poison, float %700, i64 0, !dbg !27
  %local.reduce113 = fadd float %local.reduce109, %700, !dbg !26, !frisk.local_reduction !1
  %702 = extractelement <4 x float> %582, i64 1, !dbg !25
  %703 = call float @llvm.amdgcn.exp2.f32(float %702), !dbg !25
  %704 = insertelement <4 x float> %701, float %703, i64 1, !dbg !27
  %local.reduce114 = fadd float %local.reduce110, %703, !dbg !26, !frisk.local_reduction !1
  %local.reduce117 = fadd float %local.reduce113, %local.reduce114, !dbg !26, !frisk.local_reduction !1
  %705 = extractelement <4 x float> %582, i64 2, !dbg !25
  %706 = call float @llvm.amdgcn.exp2.f32(float %705), !dbg !25
  %707 = insertelement <4 x float> %704, float %706, i64 2, !dbg !27
  %local.reduce115 = fadd float %local.reduce111, %706, !dbg !26, !frisk.local_reduction !1
  %708 = extractelement <4 x float> %582, i64 3, !dbg !25
  %709 = call float @llvm.amdgcn.exp2.f32(float %708), !dbg !25
  %710 = insertelement <4 x float> %707, float %709, i64 3, !dbg !27
  %math.result386 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %710) #9, !dbg !27, !frisk.register_lifetimes !1
  %711 = fptrunc <4 x float> %math.result386 to <4 x half>, !dbg !27
  %math.packed387 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %711) #9, !dbg !26, !frisk.register_lifetimes !1
  %local.reduce116 = fadd float %local.reduce112, %709, !dbg !26, !frisk.local_reduction !1
  %local.reduce118 = fadd float %local.reduce115, %local.reduce116, !dbg !26, !frisk.local_reduction !1
  %712 = fadd float %local.reduce117, %local.reduce118, !dbg !26, !frisk.local_reduction !1
  %713 = fadd float %712, %599, !dbg !26
  %714 = insertelement <1 x float> poison, float %713, i64 0, !dbg !7
  %715 = insertvalue [2 x <1 x float>] poison, <1 x float> %714, 0, !dbg !7
  %716 = extractelement <4 x float> %584, i64 0, !dbg !25
  %717 = call float @llvm.amdgcn.exp2.f32(float %716), !dbg !25
  %718 = insertelement <4 x float> poison, float %717, i64 0, !dbg !27
  %719 = extractelement <4 x float> %584, i64 1, !dbg !25
  %720 = call float @llvm.amdgcn.exp2.f32(float %719), !dbg !25
  %721 = insertelement <4 x float> %718, float %720, i64 1, !dbg !27
  %722 = extractelement <4 x float> %584, i64 2, !dbg !25
  %723 = call float @llvm.amdgcn.exp2.f32(float %722), !dbg !25
  %724 = insertelement <4 x float> %721, float %723, i64 2, !dbg !27
  %725 = extractelement <4 x float> %584, i64 3, !dbg !25
  %726 = call float @llvm.amdgcn.exp2.f32(float %725), !dbg !25
  %727 = insertelement <4 x float> %724, float %726, i64 3, !dbg !27
  %math.result388 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %727) #9, !dbg !27, !frisk.register_lifetimes !1
  %728 = fptrunc <4 x float> %math.result388 to <4 x half>, !dbg !27
  %math.packed389 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %728) #9, !dbg !25, !frisk.register_lifetimes !1
  %729 = extractelement <4 x float> %585, i64 0, !dbg !25
  %730 = call float @llvm.amdgcn.exp2.f32(float %729), !dbg !25
  %731 = insertelement <4 x float> poison, float %730, i64 0, !dbg !27
  %local.reduce = fadd float %717, %730, !dbg !26, !frisk.local_reduction !1
  %732 = extractelement <4 x float> %585, i64 1, !dbg !25
  %733 = call float @llvm.amdgcn.exp2.f32(float %732), !dbg !25
  %734 = insertelement <4 x float> %731, float %733, i64 1, !dbg !27
  %local.reduce59 = fadd float %720, %733, !dbg !26, !frisk.local_reduction !1
  %735 = extractelement <4 x float> %585, i64 2, !dbg !25
  %736 = call float @llvm.amdgcn.exp2.f32(float %735), !dbg !25
  %737 = insertelement <4 x float> %734, float %736, i64 2, !dbg !27
  %local.reduce60 = fadd float %723, %736, !dbg !26, !frisk.local_reduction !1
  %738 = extractelement <4 x float> %585, i64 3, !dbg !25
  %739 = call float @llvm.amdgcn.exp2.f32(float %738), !dbg !25
  %740 = insertelement <4 x float> %737, float %739, i64 3, !dbg !27
  %math.result390 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %740) #9, !dbg !27, !frisk.register_lifetimes !1
  %741 = fptrunc <4 x float> %math.result390 to <4 x half>, !dbg !27
  %math.packed391 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %741) #9, !dbg !26, !frisk.register_lifetimes !1
  %local.reduce61 = fadd float %726, %739, !dbg !26, !frisk.local_reduction !1
  %742 = extractelement <4 x float> %586, i64 0, !dbg !25
  %743 = call float @llvm.amdgcn.exp2.f32(float %742), !dbg !25
  %744 = insertelement <4 x float> poison, float %743, i64 0, !dbg !27
  %local.reduce62 = fadd float %local.reduce, %743, !dbg !26, !frisk.local_reduction !1
  %745 = extractelement <4 x float> %586, i64 1, !dbg !25
  %746 = call float @llvm.amdgcn.exp2.f32(float %745), !dbg !25
  %747 = insertelement <4 x float> %744, float %746, i64 1, !dbg !27
  %local.reduce63 = fadd float %local.reduce59, %746, !dbg !26, !frisk.local_reduction !1
  %748 = extractelement <4 x float> %586, i64 2, !dbg !25
  %749 = call float @llvm.amdgcn.exp2.f32(float %748), !dbg !25
  %750 = insertelement <4 x float> %747, float %749, i64 2, !dbg !27
  %local.reduce64 = fadd float %local.reduce60, %749, !dbg !26, !frisk.local_reduction !1
  %751 = extractelement <4 x float> %586, i64 3, !dbg !25
  %752 = call float @llvm.amdgcn.exp2.f32(float %751), !dbg !25
  %753 = insertelement <4 x float> %750, float %752, i64 3, !dbg !27
  %math.result392 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %753) #9, !dbg !27, !frisk.register_lifetimes !1
  %754 = fptrunc <4 x float> %math.result392 to <4 x half>, !dbg !27
  %math.packed393 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %754) #9, !dbg !26, !frisk.register_lifetimes !1
  %local.reduce65 = fadd float %local.reduce61, %752, !dbg !26, !frisk.local_reduction !1
  %755 = extractelement <4 x float> %587, i64 0, !dbg !25
  %756 = call float @llvm.amdgcn.exp2.f32(float %755), !dbg !25
  %757 = insertelement <4 x float> poison, float %756, i64 0, !dbg !27
  %local.reduce66 = fadd float %local.reduce62, %756, !dbg !26, !frisk.local_reduction !1
  %758 = extractelement <4 x float> %587, i64 1, !dbg !25
  %759 = call float @llvm.amdgcn.exp2.f32(float %758), !dbg !25
  %760 = insertelement <4 x float> %757, float %759, i64 1, !dbg !27
  %local.reduce67 = fadd float %local.reduce63, %759, !dbg !26, !frisk.local_reduction !1
  %761 = extractelement <4 x float> %587, i64 2, !dbg !25
  %762 = call float @llvm.amdgcn.exp2.f32(float %761), !dbg !25
  %763 = insertelement <4 x float> %760, float %762, i64 2, !dbg !27
  %local.reduce68 = fadd float %local.reduce64, %762, !dbg !26, !frisk.local_reduction !1
  %764 = extractelement <4 x float> %587, i64 3, !dbg !25
  %765 = call float @llvm.amdgcn.exp2.f32(float %764), !dbg !25
  %766 = insertelement <4 x float> %763, float %765, i64 3, !dbg !27
  %math.result394 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %766) #9, !dbg !27, !frisk.register_lifetimes !1
  %767 = fptrunc <4 x float> %math.result394 to <4 x half>, !dbg !27
  %math.packed395 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %767) #9, !dbg !26, !frisk.register_lifetimes !1
  %local.reduce69 = fadd float %local.reduce65, %765, !dbg !26, !frisk.local_reduction !1
  %768 = extractelement <4 x float> %588, i64 0, !dbg !25
  %769 = call float @llvm.amdgcn.exp2.f32(float %768), !dbg !25
  %770 = insertelement <4 x float> poison, float %769, i64 0, !dbg !27
  %local.reduce70 = fadd float %local.reduce66, %769, !dbg !26, !frisk.local_reduction !1
  %771 = extractelement <4 x float> %588, i64 1, !dbg !25
  %772 = call float @llvm.amdgcn.exp2.f32(float %771), !dbg !25
  %773 = insertelement <4 x float> %770, float %772, i64 1, !dbg !27
  %local.reduce71 = fadd float %local.reduce67, %772, !dbg !26, !frisk.local_reduction !1
  %774 = extractelement <4 x float> %588, i64 2, !dbg !25
  %775 = call float @llvm.amdgcn.exp2.f32(float %774), !dbg !25
  %776 = insertelement <4 x float> %773, float %775, i64 2, !dbg !27
  %local.reduce72 = fadd float %local.reduce68, %775, !dbg !26, !frisk.local_reduction !1
  %777 = extractelement <4 x float> %588, i64 3, !dbg !25
  %778 = call float @llvm.amdgcn.exp2.f32(float %777), !dbg !25
  %779 = insertelement <4 x float> %776, float %778, i64 3, !dbg !27
  %math.result396 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %779) #9, !dbg !27, !frisk.register_lifetimes !1
  %780 = fptrunc <4 x float> %math.result396 to <4 x half>, !dbg !27
  %math.packed397 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %780) #9, !dbg !26, !frisk.register_lifetimes !1
  %local.reduce73 = fadd float %local.reduce69, %778, !dbg !26, !frisk.local_reduction !1
  %781 = extractelement <4 x float> %589, i64 0, !dbg !25
  %782 = call float @llvm.amdgcn.exp2.f32(float %781), !dbg !25
  %783 = insertelement <4 x float> poison, float %782, i64 0, !dbg !27
  %local.reduce74 = fadd float %local.reduce70, %782, !dbg !26, !frisk.local_reduction !1
  %784 = extractelement <4 x float> %589, i64 1, !dbg !25
  %785 = call float @llvm.amdgcn.exp2.f32(float %784), !dbg !25
  %786 = insertelement <4 x float> %783, float %785, i64 1, !dbg !27
  %local.reduce75 = fadd float %local.reduce71, %785, !dbg !26, !frisk.local_reduction !1
  %787 = extractelement <4 x float> %589, i64 2, !dbg !25
  %788 = call float @llvm.amdgcn.exp2.f32(float %787), !dbg !25
  %789 = insertelement <4 x float> %786, float %788, i64 2, !dbg !27
  %local.reduce76 = fadd float %local.reduce72, %788, !dbg !26, !frisk.local_reduction !1
  %790 = extractelement <4 x float> %589, i64 3, !dbg !25
  %791 = call float @llvm.amdgcn.exp2.f32(float %790), !dbg !25
  %792 = insertelement <4 x float> %789, float %791, i64 3, !dbg !27
  %math.result398 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %792) #9, !dbg !27, !frisk.register_lifetimes !1
  %793 = fptrunc <4 x float> %math.result398 to <4 x half>, !dbg !27
  %math.packed399 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %793) #9, !dbg !26, !frisk.register_lifetimes !1
  %local.reduce77 = fadd float %local.reduce73, %791, !dbg !26, !frisk.local_reduction !1
  %794 = extractelement <4 x float> %590, i64 0, !dbg !25
  %795 = call float @llvm.amdgcn.exp2.f32(float %794), !dbg !25
  %796 = insertelement <4 x float> poison, float %795, i64 0, !dbg !27
  %local.reduce78 = fadd float %local.reduce74, %795, !dbg !26, !frisk.local_reduction !1
  %797 = extractelement <4 x float> %590, i64 1, !dbg !25
  %798 = call float @llvm.amdgcn.exp2.f32(float %797), !dbg !25
  %799 = insertelement <4 x float> %796, float %798, i64 1, !dbg !27
  %local.reduce79 = fadd float %local.reduce75, %798, !dbg !26, !frisk.local_reduction !1
  %800 = extractelement <4 x float> %590, i64 2, !dbg !25
  %801 = call float @llvm.amdgcn.exp2.f32(float %800), !dbg !25
  %802 = insertelement <4 x float> %799, float %801, i64 2, !dbg !27
  %local.reduce80 = fadd float %local.reduce76, %801, !dbg !26, !frisk.local_reduction !1
  %803 = extractelement <4 x float> %590, i64 3, !dbg !25
  %804 = call float @llvm.amdgcn.exp2.f32(float %803), !dbg !25
  %805 = insertelement <4 x float> %802, float %804, i64 3, !dbg !27
  %math.result400 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %805) #9, !dbg !27, !frisk.register_lifetimes !1
  %806 = fptrunc <4 x float> %math.result400 to <4 x half>, !dbg !27
  %math.packed401 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %806) #9, !dbg !26, !frisk.register_lifetimes !1
  %local.reduce81 = fadd float %local.reduce77, %804, !dbg !26, !frisk.local_reduction !1
  %807 = extractelement <4 x float> %591, i64 0, !dbg !25
  %808 = call float @llvm.amdgcn.exp2.f32(float %807), !dbg !25
  %809 = insertelement <4 x float> poison, float %808, i64 0, !dbg !27
  %local.reduce82 = fadd float %local.reduce78, %808, !dbg !26, !frisk.local_reduction !1
  %810 = extractelement <4 x float> %591, i64 1, !dbg !25
  %811 = call float @llvm.amdgcn.exp2.f32(float %810), !dbg !25
  %812 = insertelement <4 x float> %809, float %811, i64 1, !dbg !27
  %local.reduce83 = fadd float %local.reduce79, %811, !dbg !26, !frisk.local_reduction !1
  %local.reduce86 = fadd float %local.reduce82, %local.reduce83, !dbg !26, !frisk.local_reduction !1
  %813 = extractelement <4 x float> %591, i64 2, !dbg !25
  %814 = call float @llvm.amdgcn.exp2.f32(float %813), !dbg !25
  %815 = insertelement <4 x float> %812, float %814, i64 2, !dbg !27
  %local.reduce84 = fadd float %local.reduce80, %814, !dbg !26, !frisk.local_reduction !1
  %816 = extractelement <4 x float> %591, i64 3, !dbg !25
  %817 = call float @llvm.amdgcn.exp2.f32(float %816), !dbg !25
  %818 = insertelement <4 x float> %815, float %817, i64 3, !dbg !27
  %math.result402 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %818) #9, !dbg !27, !frisk.register_lifetimes !1
  %819 = fptrunc <4 x float> %math.result402 to <4 x half>, !dbg !27
  %math.packed403 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %819) #9, !dbg !26, !frisk.register_lifetimes !1
  %local.reduce85 = fadd float %local.reduce81, %817, !dbg !26, !frisk.local_reduction !1
  %local.reduce87 = fadd float %local.reduce84, %local.reduce85, !dbg !26, !frisk.local_reduction !1
  %820 = fadd float %local.reduce86, %local.reduce87, !dbg !26, !frisk.local_reduction !1
  %821 = fadd float %820, %607, !dbg !26
  %822 = insertelement <1 x float> poison, float %821, i64 0, !dbg !7
  %823 = insertvalue [2 x <1 x float>] %715, <1 x float> %822, 1, !dbg !7
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %vdb.address = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local504), !dbg !15
  %vdb.address1435 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %vdb.address), !dbg !15
  %vdb.address1436 = or disjoint i64 %147, %address.local921, !dbg !15
  %vdb.address1437 = trunc nuw i64 %vdb.address1436 to i32, !dbg !15
  %vdb.address1438 = shl nuw nsw i32 %vdb.address1437, 1, !dbg !15
  %vdb.address1439 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %vdb.address1440 = insertelement <2 x i64> poison, i64 %vdb.address1439, i64 0, !dbg !15
  %vdb.address1441 = bitcast <2 x i64> %vdb.address1440 to <4 x i32>, !dbg !15
  %vdb.address1442 = shufflevector <4 x i32> %vdb.address1441, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %vdb.address1443 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local521), !dbg !15
  %vdb.address1444 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %vdb.address1443), !dbg !15
  %vdb.address1445 = or disjoint i64 %147, %address.local927, !dbg !15
  %vdb.address1446 = trunc nuw i64 %vdb.address1445 to i32, !dbg !15
  %vdb.address1447 = shl nuw nsw i32 %vdb.address1446, 1, !dbg !15
  %vdb.address1448 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %vdb.address1449 = insertelement <2 x i64> poison, i64 %vdb.address1448, i64 0, !dbg !15
  %vdb.address1450 = bitcast <2 x i64> %vdb.address1449 to <4 x i32>, !dbg !15
  %vdb.address1451 = shufflevector <4 x i32> %vdb.address1450, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %vdb.dma.base = add i32 %vdb.address1435, 8192, !dbg !13
  %824 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1438, <4 x i32> %vdb.address1442) #10, !dbg !15
  %vdb.dma.base1452 = add i32 %vdb.address1444, 8192, !dbg !13
  %825 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1452, i32 %vdb.address1447, <4 x i32> %vdb.address1451) #10, !dbg !15
  %address.local929 = shl nuw nsw i32 %address.thread498, 3, !dbg !13
  %address.local930 = and i32 %address.local929, 120, !dbg !13
  %address.local931 = shl nuw nsw i32 %address.thread498, 5, !dbg !13
  %address.local932 = and i32 %address.local931, 1536, !dbg !13
  %address.local933 = or disjoint i32 %address.local930, %address.local932, !dbg !13
  %address.local934 = zext nneg i32 %address.local933 to i64, !dbg !13
  %address.local935 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local934, !dbg !13
  %826 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local935, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %827 = bitcast <4 x i32> %826 to <8 x half>, !dbg !13
  %address.local936 = and i64 %address.local499, 60, !dbg !13
  %address.local937 = add nuw nsw i64 %address.local936, %address.local499, !dbg !13
  %address.local938 = shl nuw nsw i64 %address.local937, 3, !dbg !13
  %address.local939 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local938, !dbg !13
  %address.local940 = lshr i64 %address.local555, 4, !dbg !13
  %address.local941 = mul nuw nsw i64 %address.local940, 384, !dbg !13
  %address.local942 = getelementptr [2 x i8], ptr addrspace(3) %address.local939, i64 %address.local941, !dbg !13
  %address.local943 = mul nsw i64 %address.local542, -32, !dbg !13
  %address.local944 = getelementptr [2 x i8], ptr addrspace(3) %address.local942, i64 %address.local943, !dbg !13
  %address.local945 = getelementptr i8, ptr addrspace(3) %address.local944, i64 256, !dbg !13
  %828 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local945, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %829 = bitcast <4 x i32> %828 to <8 x half>, !dbg !13
  %address.local946 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local938, !dbg !13
  %address.local947 = getelementptr [2 x i8], ptr addrspace(3) %address.local946, i64 %address.local941, !dbg !13
  %address.local948 = getelementptr [2 x i8], ptr addrspace(3) %address.local947, i64 %address.local943, !dbg !13
  %address.local949 = getelementptr i8, ptr addrspace(3) %address.local948, i64 512, !dbg !13
  %830 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local949, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %831 = bitcast <4 x i32> %830 to <8 x half>, !dbg !13
  %address.local950 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local938, !dbg !13
  %address.local951 = getelementptr [2 x i8], ptr addrspace(3) %address.local950, i64 %address.local941, !dbg !13
  %address.local952 = getelementptr [2 x i8], ptr addrspace(3) %address.local951, i64 %address.local943, !dbg !13
  %address.local953 = getelementptr i8, ptr addrspace(3) %address.local952, i64 768, !dbg !13
  %832 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local953, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %833 = bitcast <4 x i32> %832 to <8 x half>, !dbg !13
  %address.local954 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local934, !dbg !13
  %address.local955 = getelementptr i8, ptr addrspace(3) %address.local954, i64 4096, !dbg !13
  %834 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local955, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %835 = bitcast <4 x i32> %834 to <8 x half>, !dbg !13
  %address.local956 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local938, !dbg !13
  %address.local957 = getelementptr [2 x i8], ptr addrspace(3) %address.local956, i64 %address.local941, !dbg !13
  %address.local958 = getelementptr [2 x i8], ptr addrspace(3) %address.local957, i64 %address.local943, !dbg !13
  %address.local959 = getelementptr i8, ptr addrspace(3) %address.local958, i64 4352, !dbg !13
  %836 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local959, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %837 = bitcast <4 x i32> %836 to <8 x half>, !dbg !13
  %address.local960 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local938, !dbg !13
  %address.local961 = getelementptr [2 x i8], ptr addrspace(3) %address.local960, i64 %address.local941, !dbg !13
  %address.local962 = getelementptr [2 x i8], ptr addrspace(3) %address.local961, i64 %address.local943, !dbg !13
  %address.local963 = getelementptr i8, ptr addrspace(3) %address.local962, i64 4608, !dbg !13
  %838 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local963, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %839 = bitcast <4 x i32> %838 to <8 x half>, !dbg !13
  %address.local964 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local938, !dbg !13
  %address.local965 = getelementptr [2 x i8], ptr addrspace(3) %address.local964, i64 %address.local941, !dbg !13
  %address.local966 = getelementptr [2 x i8], ptr addrspace(3) %address.local965, i64 %address.local943, !dbg !13
  %address.local967 = getelementptr i8, ptr addrspace(3) %address.local966, i64 4864, !dbg !13
  %840 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local967, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %841 = shufflevector <8 x half> %827, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %842 = extractvalue [2 x <8 x float>] %123, 0, !dbg !28
  %843 = shufflevector <8 x float> %842, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %844 = fmul <4 x float> %843, %596, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %845 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %841, <4 x float> %844) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %846 = shufflevector <8 x half> %827, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %847 = shufflevector <8 x float> %842, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %848 = fmul <4 x float> %847, %596, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %849 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %846, <4 x float> %848) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %850 = extractvalue [2 x <8 x float>] %123, 1, !dbg !28
  %851 = shufflevector <8 x float> %850, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %852 = fmul <4 x float> %851, %604, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %853 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %841, <4 x float> %852) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %854 = shufflevector <8 x float> %850, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %855 = fmul <4 x float> %854, %604, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %856 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %846, <4 x float> %855) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %857 = shufflevector <8 x half> %829, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %858 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %857, <4 x float> %845) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %859 = shufflevector <8 x half> %829, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %860 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %859, <4 x float> %849) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %861 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %857, <4 x float> %853) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %862 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %859, <4 x float> %856) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %863 = shufflevector <8 x half> %831, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %864 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %863, <4 x float> %858) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %865 = shufflevector <8 x half> %831, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %866 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %865, <4 x float> %860) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %867 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %863, <4 x float> %861) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %868 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %865, <4 x float> %862) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %869 = shufflevector <8 x half> %833, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %870 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %869, <4 x float> %864) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %871 = shufflevector <8 x half> %833, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %872 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %871, <4 x float> %866) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %873 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %869, <4 x float> %867) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %874 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %871, <4 x float> %868) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %875 = shufflevector <8 x half> %835, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %876 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %875, <4 x float> %870) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %877 = shufflevector <8 x half> %835, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %878 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %877, <4 x float> %872) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %879 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %875, <4 x float> %873) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %880 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %877, <4 x float> %874) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %881 = shufflevector <8 x half> %837, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %882 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %881, <4 x float> %876) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %883 = shufflevector <8 x half> %837, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %884 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %883, <4 x float> %878) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %885 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %881, <4 x float> %879) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %886 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %883, <4 x float> %880) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %887 = shufflevector <8 x half> %839, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %888 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %887, <4 x float> %882) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %889 = shufflevector <8 x half> %839, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %890 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %889, <4 x float> %884) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %891 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %887, <4 x float> %885) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %892 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %889, <4 x float> %886) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %893 = bitcast <4 x i32> %840 to <8 x half>, !dbg !13
  %894 = shufflevector <8 x half> %893, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %895 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %894, <4 x float> %888) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %896 = shufflevector <8 x half> %893, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %897 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %896, <4 x float> %890) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %898 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %894, <4 x float> %891) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %899 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %896, <4 x float> %892) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %vdb.address1460 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local504), !dbg !15
  %vdb.address1461 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %vdb.address1460), !dbg !15
  %vdb.address1462 = or disjoint i64 %244, %address.local921, !dbg !15
  %vdb.address1463 = trunc nuw i64 %vdb.address1462 to i32, !dbg !15
  %vdb.address1464 = shl nuw nsw i32 %vdb.address1463, 1, !dbg !15
  %vdb.address1465 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %vdb.address1466 = insertelement <2 x i64> poison, i64 %vdb.address1465, i64 0, !dbg !15
  %vdb.address1467 = bitcast <2 x i64> %vdb.address1466 to <4 x i32>, !dbg !15
  %vdb.address1468 = shufflevector <4 x i32> %vdb.address1467, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %vdb.address1469 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local521), !dbg !15
  %vdb.address1470 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %vdb.address1469), !dbg !15
  %vdb.address1471 = or disjoint i64 %244, %address.local927, !dbg !15
  %vdb.address1472 = trunc nuw i64 %vdb.address1471 to i32, !dbg !15
  %vdb.address1473 = shl nuw nsw i32 %vdb.address1472, 1, !dbg !15
  %vdb.address1474 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %vdb.address1475 = insertelement <2 x i64> poison, i64 %vdb.address1474, i64 0, !dbg !15
  %vdb.address1476 = bitcast <2 x i64> %vdb.address1475 to <4 x i32>, !dbg !15
  %vdb.address1477 = shufflevector <4 x i32> %vdb.address1476, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %900 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.address1461, i32 %vdb.address1464, <4 x i32> %vdb.address1468) #10, !dbg !15
  %901 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.address1470, i32 %vdb.address1473, <4 x i32> %vdb.address1477) #10, !dbg !15
  %vdb.read = getelementptr i8, ptr addrspace(3) %address.local935, i64 8192, !dbg !13
  %902 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %903 = bitcast <4 x i32> %902 to <8 x half>, !dbg !13
  %vdb.read1453 = getelementptr i8, ptr addrspace(3) %address.local945, i64 8192, !dbg !13
  %904 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1453, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %905 = bitcast <4 x i32> %904 to <8 x half>, !dbg !13
  %vdb.read1454 = getelementptr i8, ptr addrspace(3) %address.local949, i64 8192, !dbg !13
  %906 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1454, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %907 = bitcast <4 x i32> %906 to <8 x half>, !dbg !13
  %vdb.read1455 = getelementptr i8, ptr addrspace(3) %address.local953, i64 8192, !dbg !13
  %908 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1455, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %909 = bitcast <4 x i32> %908 to <8 x half>, !dbg !13
  %vdb.read1456 = getelementptr i8, ptr addrspace(3) %address.local955, i64 8192, !dbg !13
  %910 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1456, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %911 = bitcast <4 x i32> %910 to <8 x half>, !dbg !13
  %vdb.read1457 = getelementptr i8, ptr addrspace(3) %address.local959, i64 8192, !dbg !13
  %912 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1457, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %913 = bitcast <4 x i32> %912 to <8 x half>, !dbg !13
  %vdb.read1458 = getelementptr i8, ptr addrspace(3) %address.local963, i64 8192, !dbg !13
  %914 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1458, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %915 = bitcast <4 x i32> %914 to <8 x half>, !dbg !13
  %vdb.read1459 = getelementptr i8, ptr addrspace(3) %address.local967, i64 8192, !dbg !13
  %916 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1459, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %917 = shufflevector <8 x half> %903, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %918 = extractvalue [2 x <8 x float>] %126, 0, !dbg !28
  %919 = shufflevector <8 x float> %918, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %920 = fmul <4 x float> %919, %596, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %921 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %917, <4 x float> %920) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %922 = shufflevector <8 x half> %903, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %923 = shufflevector <8 x float> %918, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %924 = fmul <4 x float> %923, %596, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %925 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %922, <4 x float> %924) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %926 = extractvalue [2 x <8 x float>] %126, 1, !dbg !28
  %927 = shufflevector <8 x float> %926, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %928 = fmul <4 x float> %927, %604, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %929 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %917, <4 x float> %928) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %930 = shufflevector <8 x float> %926, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %931 = fmul <4 x float> %930, %604, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %932 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %922, <4 x float> %931) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %933 = shufflevector <8 x half> %905, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %934 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %933, <4 x float> %921) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %935 = shufflevector <8 x half> %905, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %936 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %935, <4 x float> %925) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %937 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %933, <4 x float> %929) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %938 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %935, <4 x float> %932) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %939 = shufflevector <8 x half> %907, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %940 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %939, <4 x float> %934) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %941 = shufflevector <8 x half> %907, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %942 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %941, <4 x float> %936) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %943 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %939, <4 x float> %937) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %944 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %941, <4 x float> %938) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %945 = shufflevector <8 x half> %909, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %946 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %945, <4 x float> %940) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %947 = shufflevector <8 x half> %909, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %948 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %947, <4 x float> %942) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %949 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %945, <4 x float> %943) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %950 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %947, <4 x float> %944) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %951 = shufflevector <8 x half> %911, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %952 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %951, <4 x float> %946) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %953 = shufflevector <8 x half> %911, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %954 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %953, <4 x float> %948) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %955 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %951, <4 x float> %949) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %956 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %953, <4 x float> %950) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %957 = shufflevector <8 x half> %913, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %958 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %957, <4 x float> %952) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %959 = shufflevector <8 x half> %913, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %960 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %959, <4 x float> %954) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %961 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %957, <4 x float> %955) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %962 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %959, <4 x float> %956) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %963 = shufflevector <8 x half> %915, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %964 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %963, <4 x float> %958) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %965 = shufflevector <8 x half> %915, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %966 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %965, <4 x float> %960) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %967 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %963, <4 x float> %961) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %968 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %965, <4 x float> %962) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %969 = bitcast <4 x i32> %916 to <8 x half>, !dbg !13
  %970 = shufflevector <8 x half> %969, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %971 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %970, <4 x float> %964) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %972 = shufflevector <8 x half> %969, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %973 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %972, <4 x float> %966) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %974 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %970, <4 x float> %967) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %975 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %972, <4 x float> %968) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %vdb.address1478 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local504), !dbg !15
  %vdb.address1479 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %vdb.address1478), !dbg !15
  %vdb.address1480 = or disjoint i64 %321, %address.local921, !dbg !15
  %vdb.address1481 = trunc nuw i64 %vdb.address1480 to i32, !dbg !15
  %vdb.address1482 = shl nuw nsw i32 %vdb.address1481, 1, !dbg !15
  %vdb.address1483 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %vdb.address1484 = insertelement <2 x i64> poison, i64 %vdb.address1483, i64 0, !dbg !15
  %vdb.address1485 = bitcast <2 x i64> %vdb.address1484 to <4 x i32>, !dbg !15
  %vdb.address1486 = shufflevector <4 x i32> %vdb.address1485, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %vdb.address1487 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local521), !dbg !15
  %vdb.address1488 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %vdb.address1487), !dbg !15
  %vdb.address1489 = or disjoint i64 %321, %address.local927, !dbg !15
  %vdb.address1490 = trunc nuw i64 %vdb.address1489 to i32, !dbg !15
  %vdb.address1491 = shl nuw nsw i32 %vdb.address1490, 1, !dbg !15
  %vdb.address1492 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %vdb.address1493 = insertelement <2 x i64> poison, i64 %vdb.address1492, i64 0, !dbg !15
  %vdb.address1494 = bitcast <2 x i64> %vdb.address1493 to <4 x i32>, !dbg !15
  %vdb.address1495 = shufflevector <4 x i32> %vdb.address1494, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %vdb.dma.base1496 = add i32 %vdb.address1479, 8192, !dbg !13
  %976 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1496, i32 %vdb.address1482, <4 x i32> %vdb.address1486) #10, !dbg !15
  %vdb.dma.base1497 = add i32 %vdb.address1488, 8192, !dbg !13
  %977 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1497, i32 %vdb.address1491, <4 x i32> %vdb.address1495) #10, !dbg !15
  %978 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local935, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %979 = bitcast <4 x i32> %978 to <8 x half>, !dbg !13
  %980 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local945, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %981 = bitcast <4 x i32> %980 to <8 x half>, !dbg !13
  %982 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local949, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %983 = bitcast <4 x i32> %982 to <8 x half>, !dbg !13
  %984 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local953, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %985 = bitcast <4 x i32> %984 to <8 x half>, !dbg !13
  %986 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local955, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %987 = bitcast <4 x i32> %986 to <8 x half>, !dbg !13
  %988 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local959, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %989 = bitcast <4 x i32> %988 to <8 x half>, !dbg !13
  %990 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local963, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %991 = bitcast <4 x i32> %990 to <8 x half>, !dbg !13
  %992 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local967, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %993 = shufflevector <8 x half> %979, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %994 = extractvalue [2 x <8 x float>] %127, 0, !dbg !28
  %995 = shufflevector <8 x float> %994, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %996 = fmul <4 x float> %995, %596, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %997 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %993, <4 x float> %996) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %998 = shufflevector <8 x half> %979, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %999 = shufflevector <8 x float> %994, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %1000 = fmul <4 x float> %999, %596, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1001 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %998, <4 x float> %1000) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %1002 = extractvalue [2 x <8 x float>] %127, 1, !dbg !28
  %1003 = shufflevector <8 x float> %1002, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %1004 = fmul <4 x float> %1003, %604, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1005 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %993, <4 x float> %1004) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %1006 = shufflevector <8 x float> %1002, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %1007 = fmul <4 x float> %1006, %604, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1008 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %998, <4 x float> %1007) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1009 = shufflevector <8 x half> %981, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1010 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %1009, <4 x float> %997) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1011 = shufflevector <8 x half> %981, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1012 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %1011, <4 x float> %1001) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1013 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %1009, <4 x float> %1005) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1014 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %1011, <4 x float> %1008) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1015 = shufflevector <8 x half> %983, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1016 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %1015, <4 x float> %1010) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1017 = shufflevector <8 x half> %983, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1018 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %1017, <4 x float> %1012) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1019 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %1015, <4 x float> %1013) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1020 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %1017, <4 x float> %1014) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1021 = shufflevector <8 x half> %985, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1022 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %1021, <4 x float> %1016) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1023 = shufflevector <8 x half> %985, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1024 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %1023, <4 x float> %1018) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1025 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %1021, <4 x float> %1019) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1026 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %1023, <4 x float> %1020) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1027 = shufflevector <8 x half> %987, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1028 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %1027, <4 x float> %1022) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1029 = shufflevector <8 x half> %987, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1030 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %1029, <4 x float> %1024) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1031 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %1027, <4 x float> %1025) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1032 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %1029, <4 x float> %1026) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1033 = shufflevector <8 x half> %989, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1034 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %1033, <4 x float> %1028) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1035 = shufflevector <8 x half> %989, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1036 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %1035, <4 x float> %1030) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1037 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %1033, <4 x float> %1031) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1038 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %1035, <4 x float> %1032) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1039 = shufflevector <8 x half> %991, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1040 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %1039, <4 x float> %1034) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1041 = shufflevector <8 x half> %991, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1042 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %1041, <4 x float> %1036) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1043 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %1039, <4 x float> %1037) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1044 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %1041, <4 x float> %1038) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1045 = bitcast <4 x i32> %992 to <8 x half>, !dbg !13
  %1046 = shufflevector <8 x half> %1045, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1047 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %1046, <4 x float> %1040) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1048 = shufflevector <8 x half> %1045, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1049 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %1048, <4 x float> %1042) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1050 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %1046, <4 x float> %1043) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1051 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %1048, <4 x float> %1044) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %vdb.read1498 = getelementptr i8, ptr addrspace(3) %address.local935, i64 8192, !dbg !13
  %1052 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1498, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1053 = bitcast <4 x i32> %1052 to <8 x half>, !dbg !13
  %vdb.read1499 = getelementptr i8, ptr addrspace(3) %address.local945, i64 8192, !dbg !13
  %1054 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1499, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1055 = bitcast <4 x i32> %1054 to <8 x half>, !dbg !13
  %vdb.read1500 = getelementptr i8, ptr addrspace(3) %address.local949, i64 8192, !dbg !13
  %1056 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1500, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1057 = bitcast <4 x i32> %1056 to <8 x half>, !dbg !13
  %vdb.read1501 = getelementptr i8, ptr addrspace(3) %address.local953, i64 8192, !dbg !13
  %1058 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1501, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1059 = bitcast <4 x i32> %1058 to <8 x half>, !dbg !13
  %vdb.read1502 = getelementptr i8, ptr addrspace(3) %address.local955, i64 8192, !dbg !13
  %1060 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1502, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1061 = bitcast <4 x i32> %1060 to <8 x half>, !dbg !13
  %vdb.read1503 = getelementptr i8, ptr addrspace(3) %address.local959, i64 8192, !dbg !13
  %1062 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1503, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1063 = bitcast <4 x i32> %1062 to <8 x half>, !dbg !13
  %vdb.read1504 = getelementptr i8, ptr addrspace(3) %address.local963, i64 8192, !dbg !13
  %1064 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1504, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1065 = bitcast <4 x i32> %1064 to <8 x half>, !dbg !13
  %vdb.read1505 = getelementptr i8, ptr addrspace(3) %address.local967, i64 8192, !dbg !13
  %1066 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1505, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1067 = shufflevector <8 x half> %1053, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1068 = extractvalue [2 x <8 x float>] %128, 0, !dbg !28
  %1069 = shufflevector <8 x float> %1068, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %1070 = fmul <4 x float> %1069, %596, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1071 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1067, <4 x float> %1070) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1072 = shufflevector <8 x half> %1053, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1073 = shufflevector <8 x float> %1068, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %1074 = fmul <4 x float> %1073, %596, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1075 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1072, <4 x float> %1074) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %1076 = extractvalue [2 x <8 x float>] %128, 1, !dbg !28
  %1077 = shufflevector <8 x float> %1076, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %1078 = fmul <4 x float> %1077, %604, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1079 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %1067, <4 x float> %1078) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %1080 = shufflevector <8 x float> %1076, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %1081 = fmul <4 x float> %1080, %604, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1082 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %1072, <4 x float> %1081) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1083 = shufflevector <8 x half> %1055, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1084 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %1083, <4 x float> %1071) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1085 = shufflevector <8 x half> %1055, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1086 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %1085, <4 x float> %1075) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1087 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %1083, <4 x float> %1079) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1088 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %1085, <4 x float> %1082) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1089 = shufflevector <8 x half> %1057, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1090 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %1089, <4 x float> %1084) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1091 = shufflevector <8 x half> %1057, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1092 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %1091, <4 x float> %1086) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1093 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %1089, <4 x float> %1087) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1094 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %1091, <4 x float> %1088) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1095 = shufflevector <8 x half> %1059, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1096 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %1095, <4 x float> %1090) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1097 = shufflevector <8 x half> %1059, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1098 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %1097, <4 x float> %1092) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1099 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %1095, <4 x float> %1093) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1100 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %1097, <4 x float> %1094) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1101 = shufflevector <8 x half> %1061, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1102 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %1101, <4 x float> %1096) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1103 = shufflevector <8 x half> %1061, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1104 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %1103, <4 x float> %1098) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1105 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %1101, <4 x float> %1099) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1106 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %1103, <4 x float> %1100) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1107 = shufflevector <8 x half> %1063, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1108 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %1107, <4 x float> %1102) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1109 = shufflevector <8 x half> %1063, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1110 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %1109, <4 x float> %1104) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1111 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %1107, <4 x float> %1105) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1112 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %1109, <4 x float> %1106) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1113 = shufflevector <8 x half> %1065, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1114 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %1113, <4 x float> %1108) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1115 = shufflevector <8 x half> %1065, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1116 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %1115, <4 x float> %1110) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1117 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %1113, <4 x float> %1111) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1118 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %1115, <4 x float> %1112) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1119 = bitcast <4 x i32> %1066 to <8 x half>, !dbg !13
  %1120 = shufflevector <8 x half> %1119, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1121 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %1120, <4 x float> %1114) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1122 = shufflevector <8 x half> %1119, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1123 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %1122, <4 x float> %1116) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1124 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %1120, <4 x float> %1117) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1125 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %1122, <4 x float> %1118) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !7
  %1126 = shufflevector <4 x float> %895, <4 x float> %897, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1127 = insertvalue [2 x <8 x float>] poison, <8 x float> %1126, 0, !dbg !7
  %1128 = shufflevector <4 x float> %898, <4 x float> %899, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1129 = insertvalue [2 x <8 x float>] %1127, <8 x float> %1128, 1, !dbg !7
  %1130 = insertvalue [2 x <1 x float>] poison, <1 x float> %560, 0, !dbg !7
  %1131 = insertvalue [2 x <1 x float>] %1130, <1 x float> %573, 1, !dbg !7
  %1132 = shufflevector <4 x float> %971, <4 x float> %973, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1133 = insertvalue [2 x <8 x float>] poison, <8 x float> %1132, 0, !dbg !7
  %1134 = shufflevector <4 x float> %974, <4 x float> %975, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1135 = insertvalue [2 x <8 x float>] %1133, <8 x float> %1134, 1, !dbg !7
  %1136 = shufflevector <4 x float> %1047, <4 x float> %1049, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1137 = insertvalue [2 x <8 x float>] poison, <8 x float> %1136, 0, !dbg !7
  %1138 = shufflevector <4 x float> %1050, <4 x float> %1051, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1139 = insertvalue [2 x <8 x float>] %1137, <8 x float> %1138, 1, !dbg !7
  %1140 = shufflevector <4 x float> %1121, <4 x float> %1123, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1141 = insertvalue [2 x <8 x float>] poison, <8 x float> %1140, 0, !dbg !7
  %1142 = shufflevector <4 x float> %1124, <4 x float> %1125, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1143 = insertvalue [2 x <8 x float>] %1141, <8 x float> %1142, 1, !dbg !7
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %1144 = add nuw nsw i64 %122, 1, !dbg !7
  br label %121, !dbg !7

1145:                                             ; preds = %121
  %lane.sum.field184 = extractvalue [2 x <1 x float>] %124, 1, !dbg !7
  %lane.sum.local185 = extractelement <1 x float> %lane.sum.field184, i64 0, !dbg !7
  %lane.sum.lane.lo186 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !7
  %lane.sum.lane187 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %lane.sum.lane.lo186), !dbg !7
  %1146 = shl nuw nsw i32 %lane.sum.lane187, 2, !dbg !7
  %1147 = xor i32 %1146, 64, !dbg !7
  %1148 = bitcast float %lane.sum.local185 to i32, !dbg !7
  %lane.sum.shuffle188 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1147, i32 %1148), !dbg !7
  %1149 = bitcast i32 %lane.sum.shuffle188 to float, !dbg !7
  %lane.sum.final189 = fadd float %lane.sum.local185, %1149, !dbg !7
  %1150 = shl nuw nsw i32 %lane.sum.lane187, 2, !dbg !7
  %1151 = xor i32 %1150, 128, !dbg !7
  %1152 = bitcast float %lane.sum.final189 to i32, !dbg !7
  %lane.sum.shuffle190 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1151, i32 %1152), !dbg !7
  %1153 = bitcast i32 %lane.sum.shuffle190 to float, !dbg !7
  %lane.sum.final191 = fadd float %lane.sum.final189, %1153, !dbg !7
  %1154 = insertelement <1 x float> poison, float %lane.sum.final191, i64 0, !dbg !7
  %lane.sum.field = extractvalue [2 x <1 x float>] %124, 0, !dbg !7
  %lane.sum.local = extractelement <1 x float> %lane.sum.field, i64 0, !dbg !7
  %lane.sum.lane.lo = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !7
  %lane.sum.lane = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %lane.sum.lane.lo), !dbg !7
  %1155 = shl nuw nsw i32 %lane.sum.lane, 2, !dbg !7
  %1156 = xor i32 %1155, 64, !dbg !7
  %1157 = bitcast float %lane.sum.local to i32, !dbg !7
  %lane.sum.shuffle = call i32 @llvm.amdgcn.ds.bpermute(i32 %1156, i32 %1157), !dbg !7
  %1158 = bitcast i32 %lane.sum.shuffle to float, !dbg !7
  %lane.sum.final = fadd float %lane.sum.local, %1158, !dbg !7
  %1159 = shl nuw nsw i32 %lane.sum.lane, 2, !dbg !7
  %1160 = xor i32 %1159, 128, !dbg !7
  %1161 = bitcast float %lane.sum.final to i32, !dbg !7
  %lane.sum.shuffle182 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1160, i32 %1161), !dbg !7
  %1162 = bitcast i32 %lane.sum.shuffle182 to float, !dbg !7
  %lane.sum.final183 = fadd float %lane.sum.final, %1162, !dbg !7
  %1163 = insertelement <1 x float> poison, float %lane.sum.final183, i64 0, !dbg !7
  %1164 = and i32 %27, 16, !dbg !7
  %.not = icmp eq i32 %1164, 0, !dbg !7
  %1165 = and i32 %27, 32, !dbg !7
  %.not36 = icmp eq i32 %1165, 0, !dbg !7
  %1166 = extractvalue [2 x <8 x float>] %128, 0, !dbg !7
  %1167 = extractelement <8 x float> %1166, i64 0, !dbg !7
  %1168 = extractelement <8 x float> %1166, i64 1, !dbg !7
  %1169 = extractelement <8 x float> %1166, i64 2, !dbg !7
  %1170 = extractelement <8 x float> %1166, i64 3, !dbg !7
  %1171 = select i1 %.not, float %1168, float %1167, !dbg !7
  %1172 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !7
  %1173 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1172), !dbg !7
  %1174 = add nuw nsw i32 %1173, 64, !dbg !7
  %1175 = xor i32 %1173, 16, !dbg !7
  %1176 = and i32 %1174, 192, !dbg !7
  %1177 = icmp samesign ult i32 %1175, %1176, !dbg !7
  %1178 = select i1 %1177, i32 %1175, i32 %1173, !dbg !7
  %1179 = shl nuw nsw i32 %1178, 2, !dbg !7
  %1180 = bitcast float %1171 to i32, !dbg !7
  %1181 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1180), !dbg !7
  %1182 = bitcast i32 %1181 to float, !dbg !7
  %1183 = select i1 %.not, float %1167, float %1182, !dbg !7
  %1184 = select i1 %.not, float %1182, float %1168, !dbg !7
  %1185 = select i1 %.not, float %1170, float %1169, !dbg !7
  %1186 = bitcast float %1185 to i32, !dbg !7
  %1187 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1186), !dbg !7
  %1188 = bitcast i32 %1187 to float, !dbg !7
  %1189 = select i1 %.not, float %1169, float %1188, !dbg !7
  %1190 = select i1 %.not, float %1188, float %1170, !dbg !7
  %1191 = select i1 %.not36, float %1189, float %1183, !dbg !7
  %1192 = xor i32 %1173, 32, !dbg !7
  %1193 = icmp samesign ult i32 %1192, %1176, !dbg !7
  %1194 = select i1 %1193, i32 %1192, i32 %1173, !dbg !7
  %1195 = shl nuw nsw i32 %1194, 2, !dbg !7
  %1196 = bitcast float %1191 to i32, !dbg !7
  %1197 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1195, i32 %1196), !dbg !7
  %1198 = bitcast i32 %1197 to float, !dbg !7
  %1199 = select i1 %.not36, float %1183, float %1198, !dbg !7
  %1200 = select i1 %.not36, float %1198, float %1189, !dbg !7
  %1201 = select i1 %.not36, float %1190, float %1184, !dbg !7
  %1202 = bitcast float %1201 to i32, !dbg !7
  %1203 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1195, i32 %1202), !dbg !7
  %1204 = bitcast i32 %1203 to float, !dbg !7
  %1205 = select i1 %.not36, float %1184, float %1204, !dbg !7
  %1206 = select i1 %.not36, float %1204, float %1190, !dbg !7
  %1207 = extractvalue [2 x <8 x float>] %127, 0, !dbg !7
  %1208 = extractelement <8 x float> %1207, i64 0, !dbg !7
  %1209 = extractelement <8 x float> %1207, i64 1, !dbg !7
  %1210 = extractelement <8 x float> %1207, i64 2, !dbg !7
  %1211 = extractelement <8 x float> %1207, i64 3, !dbg !7
  %1212 = select i1 %.not, float %1209, float %1208, !dbg !7
  %1213 = bitcast float %1212 to i32, !dbg !7
  %1214 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1213), !dbg !7
  %1215 = bitcast i32 %1214 to float, !dbg !7
  %1216 = select i1 %.not, float %1208, float %1215, !dbg !7
  %1217 = select i1 %.not, float %1215, float %1209, !dbg !7
  %1218 = select i1 %.not, float %1211, float %1210, !dbg !7
  %1219 = bitcast float %1218 to i32, !dbg !7
  %1220 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1219), !dbg !7
  %1221 = bitcast i32 %1220 to float, !dbg !7
  %1222 = select i1 %.not, float %1210, float %1221, !dbg !7
  %1223 = select i1 %.not, float %1221, float %1211, !dbg !7
  %1224 = select i1 %.not36, float %1222, float %1216, !dbg !7
  %1225 = bitcast float %1224 to i32, !dbg !7
  %1226 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1195, i32 %1225), !dbg !7
  %1227 = bitcast i32 %1226 to float, !dbg !7
  %1228 = select i1 %.not36, float %1216, float %1227, !dbg !7
  %1229 = select i1 %.not36, float %1227, float %1222, !dbg !7
  %1230 = select i1 %.not36, float %1223, float %1217, !dbg !7
  %1231 = bitcast float %1230 to i32, !dbg !7
  %1232 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1195, i32 %1231), !dbg !7
  %1233 = bitcast i32 %1232 to float, !dbg !7
  %1234 = select i1 %.not36, float %1217, float %1233, !dbg !7
  %1235 = select i1 %.not36, float %1233, float %1223, !dbg !7
  %1236 = extractvalue [2 x <8 x float>] %126, 0, !dbg !7
  %1237 = extractelement <8 x float> %1236, i64 0, !dbg !7
  %1238 = extractelement <8 x float> %1236, i64 1, !dbg !7
  %1239 = extractelement <8 x float> %1236, i64 2, !dbg !7
  %1240 = extractelement <8 x float> %1236, i64 3, !dbg !7
  %1241 = select i1 %.not, float %1238, float %1237, !dbg !7
  %1242 = bitcast float %1241 to i32, !dbg !7
  %1243 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1242), !dbg !7
  %1244 = bitcast i32 %1243 to float, !dbg !7
  %1245 = select i1 %.not, float %1237, float %1244, !dbg !7
  %1246 = select i1 %.not, float %1244, float %1238, !dbg !7
  %1247 = select i1 %.not, float %1240, float %1239, !dbg !7
  %1248 = bitcast float %1247 to i32, !dbg !7
  %1249 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1248), !dbg !7
  %1250 = bitcast i32 %1249 to float, !dbg !7
  %1251 = select i1 %.not, float %1239, float %1250, !dbg !7
  %1252 = select i1 %.not, float %1250, float %1240, !dbg !7
  %1253 = select i1 %.not36, float %1251, float %1245, !dbg !7
  %1254 = bitcast float %1253 to i32, !dbg !7
  %1255 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1195, i32 %1254), !dbg !7
  %1256 = bitcast i32 %1255 to float, !dbg !7
  %1257 = select i1 %.not36, float %1245, float %1256, !dbg !7
  %1258 = select i1 %.not36, float %1256, float %1251, !dbg !7
  %1259 = select i1 %.not36, float %1252, float %1246, !dbg !7
  %1260 = bitcast float %1259 to i32, !dbg !7
  %1261 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1195, i32 %1260), !dbg !7
  %1262 = bitcast i32 %1261 to float, !dbg !7
  %1263 = select i1 %.not36, float %1246, float %1262, !dbg !7
  %1264 = select i1 %.not36, float %1262, float %1252, !dbg !7
  %1265 = extractvalue [2 x <8 x float>] %123, 0, !dbg !7
  %1266 = extractelement <8 x float> %1265, i64 0, !dbg !7
  %1267 = extractelement <8 x float> %1265, i64 1, !dbg !7
  %1268 = extractelement <8 x float> %1265, i64 2, !dbg !7
  %1269 = extractelement <8 x float> %1265, i64 3, !dbg !7
  %1270 = select i1 %.not, float %1267, float %1266, !dbg !7
  %1271 = bitcast float %1270 to i32, !dbg !7
  %1272 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1271), !dbg !7
  %1273 = bitcast i32 %1272 to float, !dbg !7
  %1274 = select i1 %.not, float %1266, float %1273, !dbg !7
  %1275 = select i1 %.not, float %1273, float %1267, !dbg !7
  %1276 = select i1 %.not, float %1269, float %1268, !dbg !7
  %1277 = bitcast float %1276 to i32, !dbg !7
  %1278 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1277), !dbg !7
  %1279 = bitcast i32 %1278 to float, !dbg !7
  %1280 = select i1 %.not, float %1268, float %1279, !dbg !7
  %1281 = select i1 %.not, float %1279, float %1269, !dbg !7
  %1282 = select i1 %.not36, float %1280, float %1274, !dbg !7
  %1283 = bitcast float %1282 to i32, !dbg !7
  %1284 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1195, i32 %1283), !dbg !7
  %1285 = select i1 %.not36, float %1281, float %1275, !dbg !7
  %1286 = bitcast float %1285 to i32, !dbg !7
  %1287 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1195, i32 %1286), !dbg !7
  %1288 = shufflevector <1 x float> %1163, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !16
  %1289 = extractelement <8 x float> %1166, i64 4, !dbg !7
  %1290 = extractelement <8 x float> %1166, i64 5, !dbg !7
  %1291 = extractelement <8 x float> %1166, i64 6, !dbg !7
  %1292 = extractelement <8 x float> %1166, i64 7, !dbg !7
  %1293 = select i1 %.not, float %1290, float %1289, !dbg !7
  %1294 = bitcast float %1293 to i32, !dbg !7
  %1295 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1294), !dbg !7
  %1296 = bitcast i32 %1295 to float, !dbg !7
  %1297 = select i1 %.not, float %1289, float %1296, !dbg !7
  %1298 = select i1 %.not, float %1296, float %1290, !dbg !7
  %1299 = select i1 %.not, float %1292, float %1291, !dbg !7
  %1300 = bitcast float %1299 to i32, !dbg !7
  %1301 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1300), !dbg !7
  %1302 = bitcast i32 %1301 to float, !dbg !7
  %1303 = select i1 %.not, float %1291, float %1302, !dbg !7
  %1304 = select i1 %.not, float %1302, float %1292, !dbg !7
  %1305 = select i1 %.not36, float %1303, float %1297, !dbg !7
  %1306 = bitcast float %1305 to i32, !dbg !7
  %1307 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1195, i32 %1306), !dbg !7
  %1308 = bitcast i32 %1307 to float, !dbg !7
  %1309 = select i1 %.not36, float %1297, float %1308, !dbg !7
  %1310 = select i1 %.not36, float %1308, float %1303, !dbg !7
  %1311 = select i1 %.not36, float %1304, float %1298, !dbg !7
  %1312 = bitcast float %1311 to i32, !dbg !7
  %1313 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1195, i32 %1312), !dbg !7
  %1314 = bitcast i32 %1313 to float, !dbg !7
  %1315 = select i1 %.not36, float %1298, float %1314, !dbg !7
  %1316 = select i1 %.not36, float %1314, float %1304, !dbg !7
  %1317 = extractelement <8 x float> %1207, i64 4, !dbg !7
  %1318 = extractelement <8 x float> %1207, i64 5, !dbg !7
  %1319 = extractelement <8 x float> %1207, i64 6, !dbg !7
  %1320 = extractelement <8 x float> %1207, i64 7, !dbg !7
  %1321 = select i1 %.not, float %1318, float %1317, !dbg !7
  %1322 = bitcast float %1321 to i32, !dbg !7
  %1323 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1322), !dbg !7
  %1324 = bitcast i32 %1323 to float, !dbg !7
  %1325 = select i1 %.not, float %1317, float %1324, !dbg !7
  %1326 = select i1 %.not, float %1324, float %1318, !dbg !7
  %1327 = select i1 %.not, float %1320, float %1319, !dbg !7
  %1328 = bitcast float %1327 to i32, !dbg !7
  %1329 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1328), !dbg !7
  %1330 = bitcast i32 %1329 to float, !dbg !7
  %1331 = select i1 %.not, float %1319, float %1330, !dbg !7
  %1332 = select i1 %.not, float %1330, float %1320, !dbg !7
  %1333 = select i1 %.not36, float %1331, float %1325, !dbg !7
  %1334 = bitcast float %1333 to i32, !dbg !7
  %1335 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1195, i32 %1334), !dbg !7
  %1336 = bitcast i32 %1335 to float, !dbg !7
  %1337 = select i1 %.not36, float %1325, float %1336, !dbg !7
  %1338 = select i1 %.not36, float %1336, float %1331, !dbg !7
  %1339 = select i1 %.not36, float %1332, float %1326, !dbg !7
  %1340 = bitcast float %1339 to i32, !dbg !7
  %1341 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1195, i32 %1340), !dbg !7
  %1342 = bitcast i32 %1341 to float, !dbg !7
  %1343 = select i1 %.not36, float %1326, float %1342, !dbg !7
  %1344 = select i1 %.not36, float %1342, float %1332, !dbg !7
  %1345 = extractelement <8 x float> %1236, i64 4, !dbg !7
  %1346 = extractelement <8 x float> %1236, i64 5, !dbg !7
  %1347 = extractelement <8 x float> %1236, i64 6, !dbg !7
  %1348 = extractelement <8 x float> %1236, i64 7, !dbg !7
  %1349 = select i1 %.not, float %1346, float %1345, !dbg !7
  %1350 = bitcast float %1349 to i32, !dbg !7
  %1351 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1350), !dbg !7
  %1352 = bitcast i32 %1351 to float, !dbg !7
  %1353 = select i1 %.not, float %1345, float %1352, !dbg !7
  %1354 = select i1 %.not, float %1352, float %1346, !dbg !7
  %1355 = select i1 %.not, float %1348, float %1347, !dbg !7
  %1356 = bitcast float %1355 to i32, !dbg !7
  %1357 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1356), !dbg !7
  %1358 = bitcast i32 %1357 to float, !dbg !7
  %1359 = select i1 %.not, float %1347, float %1358, !dbg !7
  %1360 = select i1 %.not, float %1358, float %1348, !dbg !7
  %1361 = select i1 %.not36, float %1359, float %1353, !dbg !7
  %1362 = bitcast float %1361 to i32, !dbg !7
  %1363 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1195, i32 %1362), !dbg !7
  %1364 = bitcast i32 %1363 to float, !dbg !7
  %1365 = select i1 %.not36, float %1353, float %1364, !dbg !7
  %1366 = select i1 %.not36, float %1364, float %1359, !dbg !7
  %1367 = select i1 %.not36, float %1360, float %1354, !dbg !7
  %1368 = bitcast float %1367 to i32, !dbg !7
  %1369 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1195, i32 %1368), !dbg !7
  %1370 = bitcast i32 %1369 to float, !dbg !7
  %1371 = select i1 %.not36, float %1354, float %1370, !dbg !7
  %1372 = select i1 %.not36, float %1370, float %1360, !dbg !7
  %1373 = extractelement <8 x float> %1265, i64 4, !dbg !7
  %1374 = extractelement <8 x float> %1265, i64 5, !dbg !7
  %1375 = extractelement <8 x float> %1265, i64 6, !dbg !7
  %1376 = extractelement <8 x float> %1265, i64 7, !dbg !7
  %1377 = select i1 %.not, float %1374, float %1373, !dbg !7
  %1378 = bitcast float %1377 to i32, !dbg !7
  %1379 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1378), !dbg !7
  %1380 = bitcast i32 %1379 to float, !dbg !7
  %1381 = select i1 %.not, float %1373, float %1380, !dbg !7
  %1382 = select i1 %.not, float %1380, float %1374, !dbg !7
  %1383 = select i1 %.not, float %1376, float %1375, !dbg !7
  %1384 = bitcast float %1383 to i32, !dbg !7
  %1385 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1384), !dbg !7
  %1386 = bitcast i32 %1385 to float, !dbg !7
  %1387 = select i1 %.not, float %1375, float %1386, !dbg !7
  %1388 = select i1 %.not, float %1386, float %1376, !dbg !7
  %1389 = select i1 %.not36, float %1387, float %1381, !dbg !7
  %1390 = bitcast float %1389 to i32, !dbg !7
  %1391 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1195, i32 %1390), !dbg !7
  %1392 = select i1 %.not36, float %1388, float %1382, !dbg !7
  %1393 = bitcast float %1392 to i32, !dbg !7
  %1394 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1195, i32 %1393), !dbg !7
  %1395 = extractvalue [2 x <8 x float>] %128, 1, !dbg !7
  %1396 = extractelement <8 x float> %1395, i64 0, !dbg !7
  %1397 = extractelement <8 x float> %1395, i64 1, !dbg !7
  %1398 = extractelement <8 x float> %1395, i64 2, !dbg !7
  %1399 = extractelement <8 x float> %1395, i64 3, !dbg !7
  %1400 = select i1 %.not, float %1397, float %1396, !dbg !7
  %1401 = bitcast float %1400 to i32, !dbg !7
  %1402 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1401), !dbg !7
  %1403 = bitcast i32 %1402 to float, !dbg !7
  %1404 = select i1 %.not, float %1396, float %1403, !dbg !7
  %1405 = select i1 %.not, float %1403, float %1397, !dbg !7
  %1406 = select i1 %.not, float %1399, float %1398, !dbg !7
  %1407 = bitcast float %1406 to i32, !dbg !7
  %1408 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1407), !dbg !7
  %1409 = bitcast i32 %1408 to float, !dbg !7
  %1410 = select i1 %.not, float %1398, float %1409, !dbg !7
  %1411 = select i1 %.not, float %1409, float %1399, !dbg !7
  %1412 = select i1 %.not36, float %1410, float %1404, !dbg !7
  %1413 = bitcast float %1412 to i32, !dbg !7
  %1414 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1195, i32 %1413), !dbg !7
  %1415 = bitcast i32 %1414 to float, !dbg !7
  %1416 = select i1 %.not36, float %1404, float %1415, !dbg !7
  %1417 = select i1 %.not36, float %1415, float %1410, !dbg !7
  %1418 = select i1 %.not36, float %1411, float %1405, !dbg !7
  %1419 = bitcast float %1418 to i32, !dbg !7
  %1420 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1195, i32 %1419), !dbg !7
  %1421 = bitcast i32 %1420 to float, !dbg !7
  %1422 = select i1 %.not36, float %1405, float %1421, !dbg !7
  %1423 = select i1 %.not36, float %1421, float %1411, !dbg !7
  %1424 = extractvalue [2 x <8 x float>] %127, 1, !dbg !7
  %1425 = extractelement <8 x float> %1424, i64 0, !dbg !7
  %1426 = extractelement <8 x float> %1424, i64 1, !dbg !7
  %1427 = extractelement <8 x float> %1424, i64 2, !dbg !7
  %1428 = extractelement <8 x float> %1424, i64 3, !dbg !7
  %1429 = select i1 %.not, float %1426, float %1425, !dbg !7
  %1430 = bitcast float %1429 to i32, !dbg !7
  %1431 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1430), !dbg !7
  %1432 = bitcast i32 %1431 to float, !dbg !7
  %1433 = select i1 %.not, float %1425, float %1432, !dbg !7
  %1434 = select i1 %.not, float %1432, float %1426, !dbg !7
  %1435 = select i1 %.not, float %1428, float %1427, !dbg !7
  %1436 = bitcast float %1435 to i32, !dbg !7
  %1437 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1436), !dbg !7
  %1438 = bitcast i32 %1437 to float, !dbg !7
  %1439 = select i1 %.not, float %1427, float %1438, !dbg !7
  %1440 = select i1 %.not, float %1438, float %1428, !dbg !7
  %1441 = select i1 %.not36, float %1439, float %1433, !dbg !7
  %1442 = bitcast float %1441 to i32, !dbg !7
  %1443 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1195, i32 %1442), !dbg !7
  %1444 = bitcast i32 %1443 to float, !dbg !7
  %1445 = select i1 %.not36, float %1433, float %1444, !dbg !7
  %1446 = select i1 %.not36, float %1444, float %1439, !dbg !7
  %1447 = select i1 %.not36, float %1440, float %1434, !dbg !7
  %1448 = bitcast float %1447 to i32, !dbg !7
  %1449 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1195, i32 %1448), !dbg !7
  %1450 = bitcast i32 %1449 to float, !dbg !7
  %1451 = select i1 %.not36, float %1434, float %1450, !dbg !7
  %1452 = select i1 %.not36, float %1450, float %1440, !dbg !7
  %1453 = extractvalue [2 x <8 x float>] %126, 1, !dbg !7
  %1454 = extractelement <8 x float> %1453, i64 0, !dbg !7
  %1455 = extractelement <8 x float> %1453, i64 1, !dbg !7
  %1456 = extractelement <8 x float> %1453, i64 2, !dbg !7
  %1457 = extractelement <8 x float> %1453, i64 3, !dbg !7
  %1458 = select i1 %.not, float %1455, float %1454, !dbg !7
  %1459 = bitcast float %1458 to i32, !dbg !7
  %1460 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1459), !dbg !7
  %1461 = bitcast i32 %1460 to float, !dbg !7
  %1462 = select i1 %.not, float %1454, float %1461, !dbg !7
  %1463 = select i1 %.not, float %1461, float %1455, !dbg !7
  %1464 = select i1 %.not, float %1457, float %1456, !dbg !7
  %1465 = bitcast float %1464 to i32, !dbg !7
  %1466 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1465), !dbg !7
  %1467 = bitcast i32 %1466 to float, !dbg !7
  %1468 = select i1 %.not, float %1456, float %1467, !dbg !7
  %1469 = select i1 %.not, float %1467, float %1457, !dbg !7
  %1470 = select i1 %.not36, float %1468, float %1462, !dbg !7
  %1471 = bitcast float %1470 to i32, !dbg !7
  %1472 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1195, i32 %1471), !dbg !7
  %1473 = bitcast i32 %1472 to float, !dbg !7
  %1474 = select i1 %.not36, float %1462, float %1473, !dbg !7
  %1475 = select i1 %.not36, float %1473, float %1468, !dbg !7
  %1476 = select i1 %.not36, float %1469, float %1463, !dbg !7
  %1477 = bitcast float %1476 to i32, !dbg !7
  %1478 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1195, i32 %1477), !dbg !7
  %1479 = bitcast i32 %1478 to float, !dbg !7
  %1480 = select i1 %.not36, float %1463, float %1479, !dbg !7
  %1481 = select i1 %.not36, float %1479, float %1469, !dbg !7
  %1482 = extractvalue [2 x <8 x float>] %123, 1, !dbg !7
  %1483 = extractelement <8 x float> %1482, i64 0, !dbg !7
  %1484 = extractelement <8 x float> %1482, i64 1, !dbg !7
  %1485 = extractelement <8 x float> %1482, i64 2, !dbg !7
  %1486 = extractelement <8 x float> %1482, i64 3, !dbg !7
  %1487 = select i1 %.not, float %1484, float %1483, !dbg !7
  %1488 = bitcast float %1487 to i32, !dbg !7
  %1489 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1488), !dbg !7
  %1490 = bitcast i32 %1489 to float, !dbg !7
  %1491 = select i1 %.not, float %1483, float %1490, !dbg !7
  %1492 = select i1 %.not, float %1490, float %1484, !dbg !7
  %1493 = select i1 %.not, float %1486, float %1485, !dbg !7
  %1494 = bitcast float %1493 to i32, !dbg !7
  %1495 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1494), !dbg !7
  %1496 = bitcast i32 %1495 to float, !dbg !7
  %1497 = select i1 %.not, float %1485, float %1496, !dbg !7
  %1498 = select i1 %.not, float %1496, float %1486, !dbg !7
  %1499 = select i1 %.not36, float %1497, float %1491, !dbg !7
  %1500 = bitcast float %1499 to i32, !dbg !7
  %1501 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1195, i32 %1500), !dbg !7
  %1502 = select i1 %.not36, float %1498, float %1492, !dbg !7
  %1503 = bitcast float %1502 to i32, !dbg !7
  %1504 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1195, i32 %1503), !dbg !7
  %1505 = shufflevector <1 x float> %1154, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !16
  %1506 = extractelement <8 x float> %1395, i64 4, !dbg !7
  %1507 = extractelement <8 x float> %1395, i64 5, !dbg !7
  %1508 = extractelement <8 x float> %1395, i64 6, !dbg !7
  %1509 = extractelement <8 x float> %1395, i64 7, !dbg !7
  %1510 = select i1 %.not, float %1507, float %1506, !dbg !7
  %1511 = bitcast float %1510 to i32, !dbg !7
  %1512 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1511), !dbg !7
  %1513 = bitcast i32 %1512 to float, !dbg !7
  %1514 = select i1 %.not, float %1506, float %1513, !dbg !7
  %1515 = select i1 %.not, float %1513, float %1507, !dbg !7
  %1516 = select i1 %.not, float %1509, float %1508, !dbg !7
  %1517 = bitcast float %1516 to i32, !dbg !7
  %1518 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1517), !dbg !7
  %1519 = bitcast i32 %1518 to float, !dbg !7
  %1520 = select i1 %.not, float %1508, float %1519, !dbg !7
  %1521 = select i1 %.not, float %1519, float %1509, !dbg !7
  %1522 = select i1 %.not36, float %1520, float %1514, !dbg !7
  %1523 = bitcast float %1522 to i32, !dbg !7
  %1524 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1195, i32 %1523), !dbg !7
  %1525 = bitcast i32 %1524 to float, !dbg !7
  %1526 = select i1 %.not36, float %1514, float %1525, !dbg !7
  %1527 = select i1 %.not36, float %1525, float %1520, !dbg !7
  %1528 = select i1 %.not36, float %1521, float %1515, !dbg !7
  %1529 = bitcast float %1528 to i32, !dbg !7
  %1530 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1195, i32 %1529), !dbg !7
  %1531 = bitcast i32 %1530 to float, !dbg !7
  %1532 = select i1 %.not36, float %1515, float %1531, !dbg !7
  %1533 = select i1 %.not36, float %1531, float %1521, !dbg !7
  %1534 = extractelement <8 x float> %1424, i64 4, !dbg !7
  %1535 = extractelement <8 x float> %1424, i64 5, !dbg !7
  %1536 = extractelement <8 x float> %1424, i64 6, !dbg !7
  %1537 = extractelement <8 x float> %1424, i64 7, !dbg !7
  %1538 = select i1 %.not, float %1535, float %1534, !dbg !7
  %1539 = bitcast float %1538 to i32, !dbg !7
  %1540 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1539), !dbg !7
  %1541 = bitcast i32 %1540 to float, !dbg !7
  %1542 = select i1 %.not, float %1534, float %1541, !dbg !7
  %1543 = select i1 %.not, float %1541, float %1535, !dbg !7
  %1544 = select i1 %.not, float %1537, float %1536, !dbg !7
  %1545 = bitcast float %1544 to i32, !dbg !7
  %1546 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1545), !dbg !7
  %1547 = bitcast i32 %1546 to float, !dbg !7
  %1548 = select i1 %.not, float %1536, float %1547, !dbg !7
  %1549 = select i1 %.not, float %1547, float %1537, !dbg !7
  %1550 = select i1 %.not36, float %1548, float %1542, !dbg !7
  %1551 = bitcast float %1550 to i32, !dbg !7
  %1552 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1195, i32 %1551), !dbg !7
  %1553 = bitcast i32 %1552 to float, !dbg !7
  %1554 = select i1 %.not36, float %1542, float %1553, !dbg !7
  %1555 = select i1 %.not36, float %1553, float %1548, !dbg !7
  %1556 = select i1 %.not36, float %1549, float %1543, !dbg !7
  %1557 = bitcast float %1556 to i32, !dbg !7
  %1558 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1195, i32 %1557), !dbg !7
  %1559 = bitcast i32 %1558 to float, !dbg !7
  %1560 = select i1 %.not36, float %1543, float %1559, !dbg !7
  %1561 = select i1 %.not36, float %1559, float %1549, !dbg !7
  %1562 = extractelement <8 x float> %1453, i64 4, !dbg !7
  %1563 = extractelement <8 x float> %1453, i64 5, !dbg !7
  %1564 = extractelement <8 x float> %1453, i64 6, !dbg !7
  %1565 = extractelement <8 x float> %1453, i64 7, !dbg !7
  %1566 = select i1 %.not, float %1563, float %1562, !dbg !7
  %1567 = bitcast float %1566 to i32, !dbg !7
  %1568 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1567), !dbg !7
  %1569 = bitcast i32 %1568 to float, !dbg !7
  %1570 = select i1 %.not, float %1562, float %1569, !dbg !7
  %1571 = select i1 %.not, float %1569, float %1563, !dbg !7
  %1572 = select i1 %.not, float %1565, float %1564, !dbg !7
  %1573 = bitcast float %1572 to i32, !dbg !7
  %1574 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1573), !dbg !7
  %1575 = bitcast i32 %1574 to float, !dbg !7
  %1576 = select i1 %.not, float %1564, float %1575, !dbg !7
  %1577 = select i1 %.not, float %1575, float %1565, !dbg !7
  %1578 = select i1 %.not36, float %1576, float %1570, !dbg !7
  %1579 = bitcast float %1578 to i32, !dbg !7
  %1580 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1195, i32 %1579), !dbg !7
  %1581 = bitcast i32 %1580 to float, !dbg !7
  %1582 = select i1 %.not36, float %1570, float %1581, !dbg !7
  %1583 = select i1 %.not36, float %1581, float %1576, !dbg !7
  %1584 = select i1 %.not36, float %1577, float %1571, !dbg !7
  %1585 = bitcast float %1584 to i32, !dbg !7
  %1586 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1195, i32 %1585), !dbg !7
  %1587 = bitcast i32 %1586 to float, !dbg !7
  %1588 = select i1 %.not36, float %1571, float %1587, !dbg !7
  %1589 = select i1 %.not36, float %1587, float %1577, !dbg !7
  %1590 = extractelement <8 x float> %1482, i64 4, !dbg !7
  %1591 = extractelement <8 x float> %1482, i64 5, !dbg !7
  %1592 = extractelement <8 x float> %1482, i64 6, !dbg !7
  %1593 = extractelement <8 x float> %1482, i64 7, !dbg !7
  %1594 = select i1 %.not, float %1591, float %1590, !dbg !7
  %1595 = bitcast float %1594 to i32, !dbg !7
  %1596 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1595), !dbg !7
  %1597 = bitcast i32 %1596 to float, !dbg !7
  %1598 = select i1 %.not, float %1590, float %1597, !dbg !7
  %1599 = select i1 %.not, float %1597, float %1591, !dbg !7
  %1600 = select i1 %.not, float %1593, float %1592, !dbg !7
  %1601 = bitcast float %1600 to i32, !dbg !7
  %1602 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1601), !dbg !7
  %1603 = bitcast i32 %1602 to float, !dbg !7
  %1604 = select i1 %.not, float %1592, float %1603, !dbg !7
  %1605 = select i1 %.not, float %1603, float %1593, !dbg !7
  %1606 = select i1 %.not36, float %1604, float %1598, !dbg !7
  %1607 = bitcast float %1606 to i32, !dbg !7
  %1608 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1195, i32 %1607), !dbg !7
  %1609 = select i1 %.not36, float %1605, float %1599, !dbg !7
  %1610 = bitcast float %1609 to i32, !dbg !7
  %1611 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1195, i32 %1610), !dbg !7
  %1612 = bitcast i32 %1608 to float, !dbg !7
  %1613 = select i1 %.not36, float %1598, float %1612, !dbg !7
  %1614 = insertelement <4 x float> poison, float %1613, i64 0, !dbg !16
  %1615 = bitcast i32 %1611 to float, !dbg !7
  %1616 = select i1 %.not36, float %1599, float %1615, !dbg !7
  %1617 = insertelement <4 x float> %1614, float %1616, i64 1, !dbg !16
  %1618 = select i1 %.not36, float %1612, float %1604, !dbg !7
  %1619 = insertelement <4 x float> %1617, float %1618, i64 2, !dbg !16
  %1620 = select i1 %.not36, float %1615, float %1605, !dbg !7
  %1621 = insertelement <4 x float> %1619, float %1620, i64 3, !dbg !16
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1505, !dbg !16
  %normalized = fmul <4 x float> %1621, %row.reciprocal, !dbg !16
  %1622 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !17
  %1623 = extractelement <4 x half> %1622, i64 3, !dbg !17
  %1624 = extractelement <4 x half> %1622, i64 2, !dbg !17
  %1625 = extractelement <4 x half> %1622, i64 1, !dbg !17
  %1626 = extractelement <4 x half> %1622, i64 0, !dbg !17
  %1627 = bitcast i32 %1501 to float, !dbg !7
  %1628 = select i1 %.not36, float %1491, float %1627, !dbg !7
  %1629 = insertelement <4 x float> poison, float %1628, i64 0, !dbg !16
  %1630 = bitcast i32 %1504 to float, !dbg !7
  %1631 = select i1 %.not36, float %1492, float %1630, !dbg !7
  %1632 = insertelement <4 x float> %1629, float %1631, i64 1, !dbg !16
  %1633 = select i1 %.not36, float %1627, float %1497, !dbg !7
  %1634 = insertelement <4 x float> %1632, float %1633, i64 2, !dbg !16
  %1635 = select i1 %.not36, float %1630, float %1498, !dbg !7
  %1636 = insertelement <4 x float> %1634, float %1635, i64 3, !dbg !16
  %normalized37 = fmul <4 x float> %1636, %row.reciprocal, !dbg !16
  %1637 = fptrunc <4 x float> %normalized37 to <4 x half>, !dbg !17
  %1638 = extractelement <4 x half> %1637, i64 3, !dbg !17
  %1639 = extractelement <4 x half> %1637, i64 2, !dbg !17
  %1640 = extractelement <4 x half> %1637, i64 1, !dbg !17
  %1641 = extractelement <4 x half> %1637, i64 0, !dbg !17
  %1642 = bitcast i32 %1391 to float, !dbg !7
  %1643 = select i1 %.not36, float %1381, float %1642, !dbg !7
  %1644 = insertelement <4 x float> poison, float %1643, i64 0, !dbg !16
  %1645 = bitcast i32 %1394 to float, !dbg !7
  %1646 = select i1 %.not36, float %1382, float %1645, !dbg !7
  %1647 = insertelement <4 x float> %1644, float %1646, i64 1, !dbg !16
  %1648 = select i1 %.not36, float %1642, float %1387, !dbg !7
  %1649 = insertelement <4 x float> %1647, float %1648, i64 2, !dbg !16
  %1650 = select i1 %.not36, float %1645, float %1388, !dbg !7
  %1651 = insertelement <4 x float> %1649, float %1650, i64 3, !dbg !16
  %row.reciprocal38 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1288, !dbg !16
  %normalized39 = fmul <4 x float> %1651, %row.reciprocal38, !dbg !16
  %1652 = fptrunc <4 x float> %normalized39 to <4 x half>, !dbg !17
  %1653 = extractelement <4 x half> %1652, i64 3, !dbg !17
  %1654 = extractelement <4 x half> %1652, i64 2, !dbg !17
  %1655 = extractelement <4 x half> %1652, i64 1, !dbg !17
  %1656 = extractelement <4 x half> %1652, i64 0, !dbg !17
  %1657 = bitcast i32 %1284 to float, !dbg !7
  %1658 = select i1 %.not36, float %1274, float %1657, !dbg !7
  %1659 = insertelement <4 x float> poison, float %1658, i64 0, !dbg !16
  %1660 = bitcast i32 %1287 to float, !dbg !7
  %1661 = select i1 %.not36, float %1275, float %1660, !dbg !7
  %1662 = insertelement <4 x float> %1659, float %1661, i64 1, !dbg !16
  %1663 = select i1 %.not36, float %1657, float %1280, !dbg !7
  %1664 = insertelement <4 x float> %1662, float %1663, i64 2, !dbg !16
  %1665 = select i1 %.not36, float %1660, float %1281, !dbg !7
  %1666 = insertelement <4 x float> %1664, float %1665, i64 3, !dbg !16
  %normalized40 = fmul <4 x float> %1666, %row.reciprocal38, !dbg !16
  %1667 = fptrunc <4 x float> %normalized40 to <4 x half>, !dbg !17
  %1668 = extractelement <4 x half> %1667, i64 3, !dbg !17
  %1669 = extractelement <4 x half> %1667, i64 2, !dbg !17
  %1670 = extractelement <4 x half> %1667, i64 1, !dbg !17
  %1671 = extractelement <4 x half> %1667, i64 0, !dbg !17
  %1672 = add nuw nsw i64 %17, %18, !dbg !18
  %1673 = shl i64 %1672, 19, !dbg !18
  %1674 = and i64 %1673, 16252928, !dbg !18
  %address.thread974 = call i32 asm sideeffect "", "=v,0"(i32 %27) #9, !dbg !18, !range !32, !frisk.register_lifetimes !1
  %address.local975 = zext nneg i32 %address.thread974 to i64, !dbg !8
  %address.local976 = add nuw i64 %21, %address.local975, !dbg !18
  %1675 = lshr i64 %address.local975, 2, !dbg !18
  %address.local978 = and i64 %1675, 48, !dbg !18
  %address.local979 = add i64 %address.local976, %address.local978, !dbg !18
  %address.local980 = and i64 %address.local975, 240, !dbg !18
  %address.local981 = sub i64 %address.local979, %address.local980, !dbg !18
  %address.local982 = add i64 %address.local981, %22, !dbg !18
  %address.local983 = shl nuw nsw i64 %address.local982, 7, !dbg !18
  %address.local984 = add nuw nsw i64 %1674, %address.local983, !dbg !18
  %address.local985 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local984, !dbg !18
  %address.local986 = lshr i64 %address.local975, 2, !dbg !10
  %address.local987 = and i64 %address.local986, 12, !dbg !10
  %address.local988 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local987, !dbg !18
  store half %1671, ptr addrspace(1) %address.local988, align 2, !dbg !18
  %address.local989 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local984, !dbg !18
  %address.local990 = or disjoint i64 %address.local987, 1, !dbg !18
  %address.local991 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local989, i64 %address.local990, !dbg !18
  store half %1670, ptr addrspace(1) %address.local991, align 2, !dbg !18
  %address.local992 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local984, !dbg !18
  %address.local993 = or disjoint i64 %address.local987, 2, !dbg !18
  %address.local994 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local992, i64 %address.local993, !dbg !18
  store half %1669, ptr addrspace(1) %address.local994, align 2, !dbg !18
  %address.local995 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local984, !dbg !18
  %address.local996 = or disjoint i64 %address.local987, 3, !dbg !18
  %address.local997 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local995, i64 %address.local996, !dbg !18
  store half %1668, ptr addrspace(1) %address.local997, align 2, !dbg !18
  %address.local998 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local984, !dbg !18
  %address.local999 = or disjoint i64 %address.local987, 16, !dbg !18
  %address.local1000 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local998, i64 %address.local999, !dbg !18
  store half %1656, ptr addrspace(1) %address.local1000, align 2, !dbg !18
  %address.local1001 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local984, !dbg !18
  %address.local1002 = or disjoint i64 %address.local987, 17, !dbg !18
  %address.local1003 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1001, i64 %address.local1002, !dbg !18
  store half %1655, ptr addrspace(1) %address.local1003, align 2, !dbg !18
  %address.local1004 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local984, !dbg !18
  %address.local1005 = or disjoint i64 %address.local987, 18, !dbg !18
  %address.local1006 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1004, i64 %address.local1005, !dbg !18
  store half %1654, ptr addrspace(1) %address.local1006, align 2, !dbg !18
  %address.local1007 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local984, !dbg !18
  %address.local1008 = or disjoint i64 %address.local987, 19, !dbg !18
  %address.local1009 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1007, i64 %address.local1008, !dbg !18
  store half %1653, ptr addrspace(1) %address.local1009, align 2, !dbg !18
  %address.local1010 = shl i64 %address.local982, 7, !dbg !18
  %address.local1011 = add i64 %address.local1010, 8192, !dbg !18
  %address.local1012 = add nuw nsw i64 %1674, %address.local1011, !dbg !18
  %address.local1013 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1012, !dbg !18
  %address.local1014 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local987, !dbg !18
  store half %1641, ptr addrspace(1) %address.local1014, align 2, !dbg !18
  %address.local1015 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1012, !dbg !18
  %address.local1016 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1015, i64 %address.local990, !dbg !18
  store half %1640, ptr addrspace(1) %address.local1016, align 2, !dbg !18
  %address.local1017 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1012, !dbg !18
  %address.local1018 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1017, i64 %address.local993, !dbg !18
  store half %1639, ptr addrspace(1) %address.local1018, align 2, !dbg !18
  %address.local1019 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1012, !dbg !18
  %address.local1020 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1019, i64 %address.local996, !dbg !18
  store half %1638, ptr addrspace(1) %address.local1020, align 2, !dbg !18
  %address.local1021 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1012, !dbg !18
  %address.local1022 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1021, i64 %address.local999, !dbg !18
  store half %1626, ptr addrspace(1) %address.local1022, align 2, !dbg !18
  %address.local1023 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1012, !dbg !18
  %address.local1024 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1023, i64 %address.local1002, !dbg !18
  store half %1625, ptr addrspace(1) %address.local1024, align 2, !dbg !18
  %address.local1025 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1012, !dbg !18
  %address.local1026 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1025, i64 %address.local1005, !dbg !18
  store half %1624, ptr addrspace(1) %address.local1026, align 2, !dbg !18
  %address.local1027 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1012, !dbg !18
  %address.local1028 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1027, i64 %address.local1008, !dbg !18
  store half %1623, ptr addrspace(1) %address.local1028, align 2, !dbg !18
  %1676 = insertelement <4 x float> poison, float %1582, i64 0, !dbg !16
  %1677 = insertelement <4 x float> %1676, float %1588, i64 1, !dbg !16
  %1678 = insertelement <4 x float> %1677, float %1583, i64 2, !dbg !16
  %1679 = insertelement <4 x float> %1678, float %1589, i64 3, !dbg !16
  %row.reciprocal41 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1505, !dbg !16
  %normalized42 = fmul <4 x float> %1679, %row.reciprocal41, !dbg !16
  %1680 = fptrunc <4 x float> %normalized42 to <4 x half>, !dbg !17
  %1681 = extractelement <4 x half> %1680, i64 3, !dbg !17
  %1682 = extractelement <4 x half> %1680, i64 2, !dbg !17
  %1683 = extractelement <4 x half> %1680, i64 1, !dbg !17
  %1684 = extractelement <4 x half> %1680, i64 0, !dbg !17
  %1685 = insertelement <4 x float> poison, float %1474, i64 0, !dbg !16
  %1686 = insertelement <4 x float> %1685, float %1480, i64 1, !dbg !16
  %1687 = insertelement <4 x float> %1686, float %1475, i64 2, !dbg !16
  %1688 = insertelement <4 x float> %1687, float %1481, i64 3, !dbg !16
  %normalized43 = fmul <4 x float> %1688, %row.reciprocal41, !dbg !16
  %1689 = fptrunc <4 x float> %normalized43 to <4 x half>, !dbg !17
  %1690 = extractelement <4 x half> %1689, i64 3, !dbg !17
  %1691 = extractelement <4 x half> %1689, i64 2, !dbg !17
  %1692 = extractelement <4 x half> %1689, i64 1, !dbg !17
  %1693 = extractelement <4 x half> %1689, i64 0, !dbg !17
  %1694 = insertelement <4 x float> poison, float %1365, i64 0, !dbg !16
  %1695 = insertelement <4 x float> %1694, float %1371, i64 1, !dbg !16
  %1696 = insertelement <4 x float> %1695, float %1366, i64 2, !dbg !16
  %1697 = insertelement <4 x float> %1696, float %1372, i64 3, !dbg !16
  %row.reciprocal44 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1288, !dbg !16
  %normalized45 = fmul <4 x float> %1697, %row.reciprocal44, !dbg !16
  %1698 = fptrunc <4 x float> %normalized45 to <4 x half>, !dbg !17
  %1699 = extractelement <4 x half> %1698, i64 3, !dbg !17
  %1700 = extractelement <4 x half> %1698, i64 2, !dbg !17
  %1701 = extractelement <4 x half> %1698, i64 1, !dbg !17
  %1702 = extractelement <4 x half> %1698, i64 0, !dbg !17
  %1703 = insertelement <4 x float> poison, float %1257, i64 0, !dbg !16
  %1704 = insertelement <4 x float> %1703, float %1263, i64 1, !dbg !16
  %1705 = insertelement <4 x float> %1704, float %1258, i64 2, !dbg !16
  %1706 = insertelement <4 x float> %1705, float %1264, i64 3, !dbg !16
  %normalized46 = fmul <4 x float> %1706, %row.reciprocal44, !dbg !16
  %1707 = fptrunc <4 x float> %normalized46 to <4 x half>, !dbg !17
  %1708 = extractelement <4 x half> %1707, i64 3, !dbg !17
  %1709 = extractelement <4 x half> %1707, i64 2, !dbg !17
  %1710 = extractelement <4 x half> %1707, i64 1, !dbg !17
  %1711 = extractelement <4 x half> %1707, i64 0, !dbg !17
  %1712 = shl nuw nsw i64 %20, 19, !dbg !18
  %address.thread1029 = call i32 asm sideeffect "", "=v,0"(i32 %27) #9, !dbg !18, !range !32, !frisk.register_lifetimes !1
  %address.local1030 = zext nneg i32 %address.thread1029 to i64, !dbg !8
  %address.local1031 = add nuw i64 %21, %address.local1030, !dbg !18
  %1713 = lshr i64 %address.local1030, 2, !dbg !18
  %address.local1033 = and i64 %1713, 48, !dbg !18
  %address.local1034 = add i64 %address.local1031, %address.local1033, !dbg !18
  %address.local1035 = and i64 %address.local1030, 240, !dbg !18
  %address.local1036 = sub i64 %address.local1034, %address.local1035, !dbg !18
  %address.local1037 = add i64 %address.local1036, %22, !dbg !18
  %address.local1038 = shl nuw nsw i64 %address.local1037, 7, !dbg !18
  %address.local1039 = add nuw nsw i64 %1712, %address.local1038, !dbg !18
  %address.local1040 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1039, !dbg !18
  %address.local1041 = lshr i64 %address.local1030, 2, !dbg !10
  %address.local1042 = and i64 %address.local1041, 12, !dbg !10
  %address.local1043 = or disjoint i64 %address.local1042, 32, !dbg !18
  %address.local1044 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1040, i64 %address.local1043, !dbg !18
  store half %1711, ptr addrspace(1) %address.local1044, align 2, !dbg !18
  %address.local1045 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1039, !dbg !18
  %address.local1046 = or disjoint i64 %address.local1042, 33, !dbg !18
  %address.local1047 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1045, i64 %address.local1046, !dbg !18
  store half %1710, ptr addrspace(1) %address.local1047, align 2, !dbg !18
  %address.local1048 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1039, !dbg !18
  %address.local1049 = or disjoint i64 %address.local1042, 34, !dbg !18
  %address.local1050 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1048, i64 %address.local1049, !dbg !18
  store half %1709, ptr addrspace(1) %address.local1050, align 2, !dbg !18
  %address.local1051 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1039, !dbg !18
  %address.local1052 = or disjoint i64 %address.local1042, 35, !dbg !18
  %address.local1053 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1051, i64 %address.local1052, !dbg !18
  store half %1708, ptr addrspace(1) %address.local1053, align 2, !dbg !18
  %address.local1054 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1039, !dbg !18
  %address.local1055 = or disjoint i64 %address.local1042, 48, !dbg !18
  %address.local1056 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1055, !dbg !18
  store half %1702, ptr addrspace(1) %address.local1056, align 2, !dbg !18
  %address.local1057 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1039, !dbg !18
  %address.local1058 = or disjoint i64 %address.local1042, 49, !dbg !18
  %address.local1059 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1057, i64 %address.local1058, !dbg !18
  store half %1701, ptr addrspace(1) %address.local1059, align 2, !dbg !18
  %address.local1060 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1039, !dbg !18
  %address.local1061 = or disjoint i64 %address.local1042, 50, !dbg !18
  %address.local1062 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1060, i64 %address.local1061, !dbg !18
  store half %1700, ptr addrspace(1) %address.local1062, align 2, !dbg !18
  %address.local1063 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1039, !dbg !18
  %address.local1064 = or i64 %address.local1041, 51, !dbg !18
  %address.local1065 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1063, i64 %address.local1064, !dbg !18
  store half %1699, ptr addrspace(1) %address.local1065, align 2, !dbg !18
  %address.local1066 = shl i64 %address.local1037, 7, !dbg !18
  %address.local1067 = add i64 %address.local1066, 8192, !dbg !18
  %address.local1068 = add nuw nsw i64 %1712, %address.local1067, !dbg !18
  %address.local1069 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1068, !dbg !18
  %address.local1070 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1069, i64 %address.local1043, !dbg !18
  store half %1693, ptr addrspace(1) %address.local1070, align 2, !dbg !18
  %address.local1071 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1068, !dbg !18
  %address.local1072 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1071, i64 %address.local1046, !dbg !18
  store half %1692, ptr addrspace(1) %address.local1072, align 2, !dbg !18
  %address.local1073 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1068, !dbg !18
  %address.local1074 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1073, i64 %address.local1049, !dbg !18
  store half %1691, ptr addrspace(1) %address.local1074, align 2, !dbg !18
  %address.local1075 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1068, !dbg !18
  %address.local1076 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1075, i64 %address.local1052, !dbg !18
  store half %1690, ptr addrspace(1) %address.local1076, align 2, !dbg !18
  %address.local1077 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1068, !dbg !18
  %address.local1078 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1077, i64 %address.local1055, !dbg !18
  store half %1684, ptr addrspace(1) %address.local1078, align 2, !dbg !18
  %address.local1079 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1068, !dbg !18
  %address.local1080 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1079, i64 %address.local1058, !dbg !18
  store half %1683, ptr addrspace(1) %address.local1080, align 2, !dbg !18
  %address.local1081 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1068, !dbg !18
  %address.local1082 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1061, !dbg !18
  store half %1682, ptr addrspace(1) %address.local1082, align 2, !dbg !18
  %address.local1083 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1068, !dbg !18
  %address.local1084 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1083, i64 %address.local1064, !dbg !18
  store half %1681, ptr addrspace(1) %address.local1084, align 2, !dbg !18
  %1714 = insertelement <4 x float> poison, float %1554, i64 0, !dbg !16
  %1715 = insertelement <4 x float> %1714, float %1560, i64 1, !dbg !16
  %1716 = insertelement <4 x float> %1715, float %1555, i64 2, !dbg !16
  %1717 = insertelement <4 x float> %1716, float %1561, i64 3, !dbg !16
  %row.reciprocal47 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1505, !dbg !16
  %normalized48 = fmul <4 x float> %1717, %row.reciprocal47, !dbg !16
  %1718 = fptrunc <4 x float> %normalized48 to <4 x half>, !dbg !17
  %1719 = extractelement <4 x half> %1718, i64 3, !dbg !17
  %1720 = extractelement <4 x half> %1718, i64 2, !dbg !17
  %1721 = extractelement <4 x half> %1718, i64 1, !dbg !17
  %1722 = extractelement <4 x half> %1718, i64 0, !dbg !17
  %1723 = insertelement <4 x float> poison, float %1445, i64 0, !dbg !16
  %1724 = insertelement <4 x float> %1723, float %1451, i64 1, !dbg !16
  %1725 = insertelement <4 x float> %1724, float %1446, i64 2, !dbg !16
  %1726 = insertelement <4 x float> %1725, float %1452, i64 3, !dbg !16
  %normalized49 = fmul <4 x float> %1726, %row.reciprocal47, !dbg !16
  %1727 = fptrunc <4 x float> %normalized49 to <4 x half>, !dbg !17
  %1728 = extractelement <4 x half> %1727, i64 3, !dbg !17
  %1729 = extractelement <4 x half> %1727, i64 2, !dbg !17
  %1730 = extractelement <4 x half> %1727, i64 1, !dbg !17
  %1731 = extractelement <4 x half> %1727, i64 0, !dbg !17
  %1732 = insertelement <4 x float> poison, float %1337, i64 0, !dbg !16
  %1733 = insertelement <4 x float> %1732, float %1343, i64 1, !dbg !16
  %1734 = insertelement <4 x float> %1733, float %1338, i64 2, !dbg !16
  %1735 = insertelement <4 x float> %1734, float %1344, i64 3, !dbg !16
  %row.reciprocal50 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1288, !dbg !16
  %normalized51 = fmul <4 x float> %1735, %row.reciprocal50, !dbg !16
  %1736 = fptrunc <4 x float> %normalized51 to <4 x half>, !dbg !17
  %1737 = extractelement <4 x half> %1736, i64 3, !dbg !17
  %1738 = extractelement <4 x half> %1736, i64 2, !dbg !17
  %1739 = extractelement <4 x half> %1736, i64 1, !dbg !17
  %1740 = extractelement <4 x half> %1736, i64 0, !dbg !17
  %1741 = insertelement <4 x float> poison, float %1228, i64 0, !dbg !16
  %1742 = insertelement <4 x float> %1741, float %1234, i64 1, !dbg !16
  %1743 = insertelement <4 x float> %1742, float %1229, i64 2, !dbg !16
  %1744 = insertelement <4 x float> %1743, float %1235, i64 3, !dbg !16
  %normalized52 = fmul <4 x float> %1744, %row.reciprocal50, !dbg !16
  %1745 = fptrunc <4 x float> %normalized52 to <4 x half>, !dbg !17
  %1746 = extractelement <4 x half> %1745, i64 3, !dbg !17
  %1747 = extractelement <4 x half> %1745, i64 2, !dbg !17
  %1748 = extractelement <4 x half> %1745, i64 1, !dbg !17
  %1749 = extractelement <4 x half> %1745, i64 0, !dbg !17
  %1750 = shl nuw nsw i64 %20, 19, !dbg !18
  %address.thread1085 = call i32 asm sideeffect "", "=v,0"(i32 %27) #9, !dbg !18, !range !32, !frisk.register_lifetimes !1
  %address.local1086 = zext nneg i32 %address.thread1085 to i64, !dbg !8
  %address.local1087 = add nuw i64 %21, %address.local1086, !dbg !18
  %1751 = lshr i64 %address.local1086, 2, !dbg !18
  %address.local1089 = and i64 %1751, 48, !dbg !18
  %address.local1090 = add i64 %address.local1087, %address.local1089, !dbg !18
  %address.local1091 = and i64 %address.local1086, 240, !dbg !18
  %address.local1092 = sub i64 %address.local1090, %address.local1091, !dbg !18
  %address.local1093 = add i64 %address.local1092, %22, !dbg !18
  %address.local1094 = shl nuw nsw i64 %address.local1093, 7, !dbg !18
  %address.local1095 = add nuw nsw i64 %1750, %address.local1094, !dbg !18
  %address.local1096 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1095, !dbg !18
  %address.local1097 = lshr i64 %address.local1086, 2, !dbg !10
  %address.local1098 = and i64 %address.local1097, 12, !dbg !10
  %address.local1099 = or disjoint i64 %address.local1098, 64, !dbg !18
  %address.local1100 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1096, i64 %address.local1099, !dbg !18
  store half %1749, ptr addrspace(1) %address.local1100, align 2, !dbg !18
  %address.local1101 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1095, !dbg !18
  %address.local1102 = or disjoint i64 %address.local1098, 65, !dbg !18
  %address.local1103 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1101, i64 %address.local1102, !dbg !18
  store half %1748, ptr addrspace(1) %address.local1103, align 2, !dbg !18
  %address.local1104 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1095, !dbg !18
  %address.local1105 = or disjoint i64 %address.local1098, 66, !dbg !18
  %address.local1106 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1104, i64 %address.local1105, !dbg !18
  store half %1747, ptr addrspace(1) %address.local1106, align 2, !dbg !18
  %address.local1107 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1095, !dbg !18
  %address.local1108 = or disjoint i64 %address.local1098, 67, !dbg !18
  %address.local1109 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1107, i64 %address.local1108, !dbg !18
  store half %1746, ptr addrspace(1) %address.local1109, align 2, !dbg !18
  %address.local1110 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1095, !dbg !18
  %address.local1111 = or disjoint i64 %address.local1098, 80, !dbg !18
  %address.local1112 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1110, i64 %address.local1111, !dbg !18
  store half %1740, ptr addrspace(1) %address.local1112, align 2, !dbg !18
  %address.local1113 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1095, !dbg !18
  %address.local1114 = or disjoint i64 %address.local1098, 81, !dbg !18
  %address.local1115 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1113, i64 %address.local1114, !dbg !18
  store half %1739, ptr addrspace(1) %address.local1115, align 2, !dbg !18
  %address.local1116 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1095, !dbg !18
  %address.local1117 = or disjoint i64 %address.local1098, 82, !dbg !18
  %address.local1118 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1116, i64 %address.local1117, !dbg !18
  store half %1738, ptr addrspace(1) %address.local1118, align 2, !dbg !18
  %address.local1119 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1095, !dbg !18
  %address.local1120 = or disjoint i64 %address.local1098, 83, !dbg !18
  %address.local1121 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1119, i64 %address.local1120, !dbg !18
  store half %1737, ptr addrspace(1) %address.local1121, align 2, !dbg !18
  %address.local1122 = shl i64 %address.local1093, 7, !dbg !18
  %address.local1123 = add i64 %address.local1122, 8192, !dbg !18
  %address.local1124 = add nuw nsw i64 %1750, %address.local1123, !dbg !18
  %address.local1125 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1124, !dbg !18
  %address.local1126 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1125, i64 %address.local1099, !dbg !18
  store half %1731, ptr addrspace(1) %address.local1126, align 2, !dbg !18
  %address.local1127 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1124, !dbg !18
  %address.local1128 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1127, i64 %address.local1102, !dbg !18
  store half %1730, ptr addrspace(1) %address.local1128, align 2, !dbg !18
  %address.local1129 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1124, !dbg !18
  %address.local1130 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1129, i64 %address.local1105, !dbg !18
  store half %1729, ptr addrspace(1) %address.local1130, align 2, !dbg !18
  %address.local1131 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1124, !dbg !18
  %address.local1132 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1131, i64 %address.local1108, !dbg !18
  store half %1728, ptr addrspace(1) %address.local1132, align 2, !dbg !18
  %address.local1133 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1124, !dbg !18
  %address.local1134 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1133, i64 %address.local1111, !dbg !18
  store half %1722, ptr addrspace(1) %address.local1134, align 2, !dbg !18
  %address.local1135 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1124, !dbg !18
  %address.local1136 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1135, i64 %address.local1114, !dbg !18
  store half %1721, ptr addrspace(1) %address.local1136, align 2, !dbg !18
  %address.local1137 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1124, !dbg !18
  %address.local1138 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1137, i64 %address.local1117, !dbg !18
  store half %1720, ptr addrspace(1) %address.local1138, align 2, !dbg !18
  %address.local1139 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1124, !dbg !18
  %address.local1140 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1139, i64 %address.local1120, !dbg !18
  store half %1719, ptr addrspace(1) %address.local1140, align 2, !dbg !18
  %1752 = insertelement <4 x float> poison, float %1526, i64 0, !dbg !16
  %1753 = insertelement <4 x float> %1752, float %1532, i64 1, !dbg !16
  %1754 = insertelement <4 x float> %1753, float %1527, i64 2, !dbg !16
  %1755 = insertelement <4 x float> %1754, float %1533, i64 3, !dbg !16
  %row.reciprocal53 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1505, !dbg !16
  %normalized54 = fmul <4 x float> %1755, %row.reciprocal53, !dbg !16
  %1756 = fptrunc <4 x float> %normalized54 to <4 x half>, !dbg !17
  %1757 = extractelement <4 x half> %1756, i64 3, !dbg !17
  %1758 = extractelement <4 x half> %1756, i64 2, !dbg !17
  %1759 = extractelement <4 x half> %1756, i64 1, !dbg !17
  %1760 = extractelement <4 x half> %1756, i64 0, !dbg !17
  %1761 = insertelement <4 x float> poison, float %1416, i64 0, !dbg !16
  %1762 = insertelement <4 x float> %1761, float %1422, i64 1, !dbg !16
  %1763 = insertelement <4 x float> %1762, float %1417, i64 2, !dbg !16
  %1764 = insertelement <4 x float> %1763, float %1423, i64 3, !dbg !16
  %normalized55 = fmul <4 x float> %1764, %row.reciprocal53, !dbg !16
  %1765 = fptrunc <4 x float> %normalized55 to <4 x half>, !dbg !17
  %1766 = extractelement <4 x half> %1765, i64 3, !dbg !17
  %1767 = extractelement <4 x half> %1765, i64 2, !dbg !17
  %1768 = extractelement <4 x half> %1765, i64 1, !dbg !17
  %1769 = extractelement <4 x half> %1765, i64 0, !dbg !17
  %1770 = insertelement <4 x float> poison, float %1309, i64 0, !dbg !16
  %1771 = insertelement <4 x float> %1770, float %1315, i64 1, !dbg !16
  %1772 = insertelement <4 x float> %1771, float %1310, i64 2, !dbg !16
  %1773 = insertelement <4 x float> %1772, float %1316, i64 3, !dbg !16
  %row.reciprocal56 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1288, !dbg !16
  %normalized57 = fmul <4 x float> %1773, %row.reciprocal56, !dbg !16
  %1774 = fptrunc <4 x float> %normalized57 to <4 x half>, !dbg !17
  %1775 = extractelement <4 x half> %1774, i64 3, !dbg !17
  %1776 = extractelement <4 x half> %1774, i64 2, !dbg !17
  %1777 = extractelement <4 x half> %1774, i64 1, !dbg !17
  %1778 = extractelement <4 x half> %1774, i64 0, !dbg !17
  %1779 = insertelement <4 x float> poison, float %1199, i64 0, !dbg !16
  %1780 = insertelement <4 x float> %1779, float %1205, i64 1, !dbg !16
  %1781 = insertelement <4 x float> %1780, float %1200, i64 2, !dbg !16
  %1782 = insertelement <4 x float> %1781, float %1206, i64 3, !dbg !16
  %normalized58 = fmul <4 x float> %1782, %row.reciprocal56, !dbg !16
  %1783 = fptrunc <4 x float> %normalized58 to <4 x half>, !dbg !17
  %1784 = extractelement <4 x half> %1783, i64 3, !dbg !17
  %1785 = extractelement <4 x half> %1783, i64 2, !dbg !17
  %1786 = extractelement <4 x half> %1783, i64 1, !dbg !17
  %1787 = extractelement <4 x half> %1783, i64 0, !dbg !17
  %1788 = shl nuw nsw i64 %20, 19, !dbg !18
  %address.thread1141 = call i32 asm sideeffect "", "=v,0"(i32 %27) #9, !dbg !18, !range !32, !frisk.register_lifetimes !1
  %address.local1142 = zext nneg i32 %address.thread1141 to i64, !dbg !8
  %address.local1143 = add nuw i64 %21, %address.local1142, !dbg !18
  %1789 = lshr i64 %address.local1142, 2, !dbg !18
  %address.local1145 = and i64 %1789, 48, !dbg !18
  %address.local1146 = add i64 %address.local1143, %address.local1145, !dbg !18
  %address.local1147 = and i64 %address.local1142, 240, !dbg !18
  %address.local1148 = sub i64 %address.local1146, %address.local1147, !dbg !18
  %address.local1149 = add i64 %address.local1148, %22, !dbg !18
  %address.local1150 = shl nuw nsw i64 %address.local1149, 7, !dbg !18
  %address.local1151 = add nuw nsw i64 %1788, %address.local1150, !dbg !18
  %address.local1152 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1151, !dbg !18
  %address.local1153 = lshr i64 %address.local1142, 2, !dbg !10
  %address.local1154 = and i64 %address.local1153, 12, !dbg !10
  %address.local1155 = or disjoint i64 %address.local1154, 96, !dbg !18
  %address.local1156 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1152, i64 %address.local1155, !dbg !18
  store half %1787, ptr addrspace(1) %address.local1156, align 2, !dbg !18
  %address.local1157 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1151, !dbg !18
  %address.local1158 = or disjoint i64 %address.local1154, 97, !dbg !18
  %address.local1159 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1157, i64 %address.local1158, !dbg !18
  store half %1786, ptr addrspace(1) %address.local1159, align 2, !dbg !18
  %address.local1160 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1151, !dbg !18
  %address.local1161 = or disjoint i64 %address.local1154, 98, !dbg !18
  %address.local1162 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1160, i64 %address.local1161, !dbg !18
  store half %1785, ptr addrspace(1) %address.local1162, align 2, !dbg !18
  %address.local1163 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1151, !dbg !18
  %address.local1164 = or disjoint i64 %address.local1154, 99, !dbg !18
  %address.local1165 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1163, i64 %address.local1164, !dbg !18
  store half %1784, ptr addrspace(1) %address.local1165, align 2, !dbg !18
  %address.local1166 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1151, !dbg !18
  %address.local1167 = or disjoint i64 %address.local1154, 112, !dbg !18
  %address.local1168 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1166, i64 %address.local1167, !dbg !18
  store half %1778, ptr addrspace(1) %address.local1168, align 2, !dbg !18
  %address.local1169 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1151, !dbg !18
  %address.local1170 = or disjoint i64 %address.local1154, 113, !dbg !18
  %address.local1171 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1169, i64 %address.local1170, !dbg !18
  store half %1777, ptr addrspace(1) %address.local1171, align 2, !dbg !18
  %address.local1172 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1151, !dbg !18
  %address.local1173 = or disjoint i64 %address.local1154, 114, !dbg !18
  %address.local1174 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1172, i64 %address.local1173, !dbg !18
  store half %1776, ptr addrspace(1) %address.local1174, align 2, !dbg !18
  %address.local1175 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1151, !dbg !18
  %address.local1176 = or i64 %address.local1153, 115, !dbg !18
  %address.local1177 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1175, i64 %address.local1176, !dbg !18
  store half %1775, ptr addrspace(1) %address.local1177, align 2, !dbg !18
  %address.local1178 = shl i64 %address.local1149, 7, !dbg !18
  %address.local1179 = add i64 %address.local1178, 8192, !dbg !18
  %address.local1180 = add nuw nsw i64 %1788, %address.local1179, !dbg !18
  %address.local1181 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1180, !dbg !18
  %address.local1182 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1181, i64 %address.local1155, !dbg !18
  store half %1769, ptr addrspace(1) %address.local1182, align 2, !dbg !18
  %address.local1183 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1180, !dbg !18
  %address.local1184 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1183, i64 %address.local1158, !dbg !18
  store half %1768, ptr addrspace(1) %address.local1184, align 2, !dbg !18
  %address.local1185 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1180, !dbg !18
  %address.local1186 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1185, i64 %address.local1161, !dbg !18
  store half %1767, ptr addrspace(1) %address.local1186, align 2, !dbg !18
  %address.local1187 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1180, !dbg !18
  %address.local1188 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1187, i64 %address.local1164, !dbg !18
  store half %1766, ptr addrspace(1) %address.local1188, align 2, !dbg !18
  %address.local1189 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1180, !dbg !18
  %address.local1190 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1189, i64 %address.local1167, !dbg !18
  store half %1760, ptr addrspace(1) %address.local1190, align 2, !dbg !18
  %address.local1191 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1180, !dbg !18
  %address.local1192 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1191, i64 %address.local1170, !dbg !18
  store half %1759, ptr addrspace(1) %address.local1192, align 2, !dbg !18
  %address.local1193 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1180, !dbg !18
  %address.local1194 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1193, i64 %address.local1173, !dbg !18
  store half %1758, ptr addrspace(1) %address.local1194, align 2, !dbg !18
  %address.local1195 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1180, !dbg !18
  %address.local1196 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1195, i64 %address.local1176, !dbg !18
  store half %1757, ptr addrspace(1) %address.local1196, align 2, !dbg !18
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
attributes #1 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1" "frisk.global_prefetch"="8" "frisk.k_dma"="64-swizzle:1" "frisk.register_comparisons"="1" "frisk.register_lifetimes"="1" "uniform-work-group-size" }
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

!0 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/noncausal_k_dma_pass/mytest/run-20261030-163658-kcakvj_l/grid-000003")
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
