; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/noncausal_k_dma_pass/mytest/run-20261030-163658-kcakvj_l/grid-000002/input.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/noncausal_k_dma_pass/mytest/run-20261030-163658-kcakvj_l/grid-000002/input.mlir"

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
  %122 = phi i64 [ %1132, %130 ], [ 0, %4 ], !dbg !7
  %123 = phi [2 x <8 x float>] [ %1117, %130 ], [ zeroinitializer, %4 ], !dbg !7
  %124 = phi [2 x <1 x float>] [ %811, %130 ], [ zeroinitializer, %4 ], !dbg !7
  %125 = phi [2 x <1 x float>] [ %1119, %130 ], [ [<1 x float> splat (float 0xFFF0000000000000), <1 x float> splat (float 0xFFF0000000000000)], %4 ], !dbg !7
  %126 = phi [2 x <8 x float>] [ %1123, %130 ], [ zeroinitializer, %4 ], !dbg !7
  %127 = phi [2 x <8 x float>] [ %1127, %130 ], [ zeroinitializer, %4 ], !dbg !7
  %128 = phi [2 x <8 x float>] [ %1131, %130 ], [ zeroinitializer, %4 ], !dbg !7
  %129 = icmp samesign ult i64 %122, 32, !dbg !7
  br i1 %129, label %130, label %1133, !dbg !7

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
  %kdma.offset1533 = add i32 %kdma.source1532, %kdma.intra, !dbg !12
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
  %kdma.offset1567 = add i32 %kdma.source1566, %kdma.intra, !dbg !12
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
  %kdma.offset1599 = add i32 %kdma.source1598, %kdma.intra, !dbg !12
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
  %kdma.offset1633 = add i32 %kdma.source1632, %kdma.intra, !dbg !12
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
  %165 = load <4 x half>, ptr addrspace(3) %address.local559, align 8, !dbg !14
  %166 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %55, <4 x half> %165, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1210 = getelementptr i8, ptr addrspace(3) %address.local559, i64 4096, !dbg !14
  %167 = load <4 x half>, ptr addrspace(3) %lds.const.gep1210, align 8, !dbg !14
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
  %168 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local584, !dbg !14
  %169 = getelementptr [2 x i8], ptr addrspace(3) %168, i64 %address.local553, !dbg !14
  %170 = getelementptr [2 x i8], ptr addrspace(3) %169, i64 %address.local557, !dbg !14
  %address.local588 = getelementptr i8, ptr addrspace(3) %170, i64 64, !dbg !14
  %171 = load <4 x half>, ptr addrspace(3) %address.local588, align 8, !dbg !14
  %172 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %167, <4 x float> %166) #9, !dbg !14
  %173 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %55, <4 x half> %171, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1211 = getelementptr i8, ptr addrspace(3) %170, i64 4160, !dbg !14
  %174 = load <4 x half>, ptr addrspace(3) %lds.const.gep1211, align 8, !dbg !14
  %lds.const.gep1212 = getelementptr i8, ptr addrspace(3) %address.local559, i64 128, !dbg !14
  %175 = load <4 x half>, ptr addrspace(3) %lds.const.gep1212, align 8, !dbg !14
  %176 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %174, <4 x float> %173) #9, !dbg !14
  %177 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %55, <4 x half> %175, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1213 = getelementptr i8, ptr addrspace(3) %address.local559, i64 4224, !dbg !14
  %178 = load <4 x half>, ptr addrspace(3) %lds.const.gep1213, align 8, !dbg !14
  %lds.const.gep1214 = getelementptr i8, ptr addrspace(3) %170, i64 192, !dbg !14
  %179 = load <4 x half>, ptr addrspace(3) %lds.const.gep1214, align 8, !dbg !14
  %180 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %178, <4 x float> %177) #9, !dbg !14
  %181 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %55, <4 x half> %179, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1215 = getelementptr i8, ptr addrspace(3) %170, i64 4288, !dbg !14
  %182 = load <4 x half>, ptr addrspace(3) %lds.const.gep1215, align 8, !dbg !14
  %lds.const.gep1216 = getelementptr i8, ptr addrspace(3) %address.local559, i64 1024, !dbg !14
  %183 = load <4 x half>, ptr addrspace(3) %lds.const.gep1216, align 8, !dbg !14
  %184 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %182, <4 x float> %181) #9, !dbg !14
  %185 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %55, <4 x half> %183, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1217 = getelementptr i8, ptr addrspace(3) %address.local559, i64 5120, !dbg !14
  %186 = load <4 x half>, ptr addrspace(3) %lds.const.gep1217, align 8, !dbg !14
  %lds.const.gep1218 = getelementptr i8, ptr addrspace(3) %170, i64 1088, !dbg !14
  %187 = load <4 x half>, ptr addrspace(3) %lds.const.gep1218, align 8, !dbg !14
  %188 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %186, <4 x float> %185) #9, !dbg !14
  %189 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %55, <4 x half> %187, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1219 = getelementptr i8, ptr addrspace(3) %170, i64 5184, !dbg !14
  %190 = load <4 x half>, ptr addrspace(3) %lds.const.gep1219, align 8, !dbg !14
  %lds.const.gep1220 = getelementptr i8, ptr addrspace(3) %address.local559, i64 1152, !dbg !14
  %191 = load <4 x half>, ptr addrspace(3) %lds.const.gep1220, align 8, !dbg !14
  %192 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %190, <4 x float> %189) #9, !dbg !14
  %193 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %55, <4 x half> %191, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1221 = getelementptr i8, ptr addrspace(3) %address.local559, i64 5248, !dbg !14
  %194 = load <4 x half>, ptr addrspace(3) %lds.const.gep1221, align 8, !dbg !14
  %lds.const.gep1222 = getelementptr i8, ptr addrspace(3) %170, i64 1216, !dbg !14
  %195 = load <4 x half>, ptr addrspace(3) %lds.const.gep1222, align 8, !dbg !14
  %196 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %194, <4 x float> %193) #9, !dbg !14
  %197 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %55, <4 x half> %195, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1223 = getelementptr i8, ptr addrspace(3) %170, i64 5312, !dbg !14
  %198 = load <4 x half>, ptr addrspace(3) %lds.const.gep1223, align 8, !dbg !14
  %199 = load <4 x half>, ptr addrspace(3) %address.local559, align 8, !dbg !14
  %200 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %198, <4 x float> %197) #9, !dbg !14
  %201 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %199, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1224 = getelementptr i8, ptr addrspace(3) %address.local559, i64 4096, !dbg !14
  %202 = load <4 x half>, ptr addrspace(3) %lds.const.gep1224, align 8, !dbg !14
  %203 = load <4 x half>, ptr addrspace(3) %address.local588, align 8, !dbg !14
  %204 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %202, <4 x float> %201) #9, !dbg !14
  %205 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %203, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1225 = getelementptr i8, ptr addrspace(3) %170, i64 4160, !dbg !14
  %206 = load <4 x half>, ptr addrspace(3) %lds.const.gep1225, align 8, !dbg !14
  %lds.const.gep1226 = getelementptr i8, ptr addrspace(3) %address.local559, i64 128, !dbg !14
  %207 = load <4 x half>, ptr addrspace(3) %lds.const.gep1226, align 8, !dbg !14
  %208 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %206, <4 x float> %205) #9, !dbg !14
  %209 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %207, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1227 = getelementptr i8, ptr addrspace(3) %address.local559, i64 4224, !dbg !14
  %210 = load <4 x half>, ptr addrspace(3) %lds.const.gep1227, align 8, !dbg !14
  %lds.const.gep1228 = getelementptr i8, ptr addrspace(3) %170, i64 192, !dbg !14
  %211 = load <4 x half>, ptr addrspace(3) %lds.const.gep1228, align 8, !dbg !14
  %212 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %210, <4 x float> %209) #9, !dbg !14
  %213 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %211, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1229 = getelementptr i8, ptr addrspace(3) %170, i64 4288, !dbg !14
  %214 = load <4 x half>, ptr addrspace(3) %lds.const.gep1229, align 8, !dbg !14
  %lds.const.gep1230 = getelementptr i8, ptr addrspace(3) %address.local559, i64 1024, !dbg !14
  %215 = load <4 x half>, ptr addrspace(3) %lds.const.gep1230, align 8, !dbg !14
  %216 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %214, <4 x float> %213) #9, !dbg !14
  %217 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %215, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1231 = getelementptr i8, ptr addrspace(3) %address.local559, i64 5120, !dbg !14
  %218 = load <4 x half>, ptr addrspace(3) %lds.const.gep1231, align 8, !dbg !14
  %lds.const.gep1232 = getelementptr i8, ptr addrspace(3) %170, i64 1088, !dbg !14
  %219 = load <4 x half>, ptr addrspace(3) %lds.const.gep1232, align 8, !dbg !14
  %220 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %218, <4 x float> %217) #9, !dbg !14
  %221 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %219, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1233 = getelementptr i8, ptr addrspace(3) %170, i64 5184, !dbg !14
  %222 = load <4 x half>, ptr addrspace(3) %lds.const.gep1233, align 8, !dbg !14
  %lds.const.gep1234 = getelementptr i8, ptr addrspace(3) %address.local559, i64 1152, !dbg !14
  %223 = load <4 x half>, ptr addrspace(3) %lds.const.gep1234, align 8, !dbg !14
  %224 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %222, <4 x float> %221) #9, !dbg !14
  %225 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %223, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1235 = getelementptr i8, ptr addrspace(3) %address.local559, i64 5248, !dbg !14
  %226 = load <4 x half>, ptr addrspace(3) %lds.const.gep1235, align 8, !dbg !14
  %lds.const.gep1236 = getelementptr i8, ptr addrspace(3) %170, i64 1216, !dbg !14
  %227 = load <4 x half>, ptr addrspace(3) %lds.const.gep1236, align 8, !dbg !14
  %228 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %226, <4 x float> %225) #9, !dbg !14
  %229 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %227, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1237 = getelementptr i8, ptr addrspace(3) %170, i64 5312, !dbg !14
  %230 = load <4 x half>, ptr addrspace(3) %lds.const.gep1237, align 8, !dbg !14
  %231 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %230, <4 x float> %229) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %232 = or disjoint i64 %134, 64, !dbg !12
  %233 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local504), !dbg !12
  %234 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !12
  %235 = insertelement <2 x i64> poison, i64 %234, i64 0, !dbg !12
  %236 = bitcast <2 x i64> %235 to <4 x i32>, !dbg !12
  %dma.buffer.rsrc1361 = shufflevector <4 x i32> %236, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !12
  %dma.buffer.lds1362 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %233), !dbg !12
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
  %kdma.source1646 = or disjoint i64 %232, %kdma.source1645, !dbg !12
  %kdma.source1647 = trunc nuw i64 %kdma.source1646 to i32, !dbg !12
  %kdma.source1648 = shl nuw nsw i32 %kdma.source1647, 1, !dbg !12
  %kdma.offset1649 = add i32 %kdma.source1648, %kdma.intra, !dbg !12
  %kdma.lds1650 = add i32 %dma.buffer.lds1362, 0, !dbg !12
  %237 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1650, i32 %kdma.offset1649, <4 x i32> %dma.buffer.rsrc1361) #10, !dbg !12
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
  %kdma.source1662 = or disjoint i64 %232, %kdma.source1661, !dbg !12
  %kdma.source1663 = trunc nuw i64 %kdma.source1662 to i32, !dbg !12
  %kdma.source1664 = shl nuw nsw i32 %kdma.source1663, 1, !dbg !12
  %kdma.offset1665 = add i32 %kdma.source1664, %kdma.intra, !dbg !12
  %kdma.lds1666 = add i32 %dma.buffer.lds1362, 512, !dbg !12
  %238 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1666, i32 %kdma.offset1665, <4 x i32> %dma.buffer.rsrc1361) #10, !dbg !12
  %239 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local521), !dbg !12
  %240 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !12
  %241 = insertelement <2 x i64> poison, i64 %240, i64 0, !dbg !12
  %242 = bitcast <2 x i64> %241 to <4 x i32>, !dbg !12
  %dma.buffer.rsrc1358 = shufflevector <4 x i32> %242, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !12
  %dma.buffer.lds1359 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %239), !dbg !12
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
  %kdma.source1679 = or disjoint i64 %232, %kdma.source1678, !dbg !12
  %kdma.source1680 = trunc nuw i64 %kdma.source1679 to i32, !dbg !12
  %kdma.source1681 = shl nuw nsw i32 %kdma.source1680, 1, !dbg !12
  %kdma.offset1682 = add i32 %kdma.source1681, %kdma.intra, !dbg !12
  %kdma.lds1683 = add i32 %dma.buffer.lds1359, 0, !dbg !12
  %243 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1683, i32 %kdma.offset1682, <4 x i32> %dma.buffer.rsrc1358) #10, !dbg !12
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
  %kdma.source1696 = or disjoint i64 %232, %kdma.source1695, !dbg !12
  %kdma.source1697 = trunc nuw i64 %kdma.source1696 to i32, !dbg !12
  %kdma.source1698 = shl nuw nsw i32 %kdma.source1697, 1, !dbg !12
  %kdma.offset1699 = add i32 %kdma.source1698, %kdma.intra, !dbg !12
  %kdma.lds1700 = add i32 %dma.buffer.lds1359, 512, !dbg !12
  %244 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1700, i32 %kdma.offset1699, <4 x i32> %dma.buffer.rsrc1358) #10, !dbg !12
  %lds.const.gep1238 = getelementptr i8, ptr addrspace(3) %address.local559, i64 8192, !dbg !14
  %245 = load <4 x half>, ptr addrspace(3) %lds.const.gep1238, align 8, !dbg !14
  %246 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %245, <4 x float> %172) #9, !dbg !14
  %lds.const.gep1239 = getelementptr i8, ptr addrspace(3) %address.local559, i64 12288, !dbg !14
  %247 = load <4 x half>, ptr addrspace(3) %lds.const.gep1239, align 8, !dbg !14
  %lds.const.gep1240 = getelementptr i8, ptr addrspace(3) %170, i64 8256, !dbg !14
  %248 = load <4 x half>, ptr addrspace(3) %lds.const.gep1240, align 8, !dbg !14
  %249 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %247, <4 x float> %246) #9, !dbg !14
  %250 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %248, <4 x float> %176) #9, !dbg !14
  %lds.const.gep1241 = getelementptr i8, ptr addrspace(3) %170, i64 12352, !dbg !14
  %251 = load <4 x half>, ptr addrspace(3) %lds.const.gep1241, align 8, !dbg !14
  %lds.const.gep1242 = getelementptr i8, ptr addrspace(3) %address.local559, i64 8320, !dbg !14
  %252 = load <4 x half>, ptr addrspace(3) %lds.const.gep1242, align 8, !dbg !14
  %253 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %251, <4 x float> %250) #9, !dbg !14
  %254 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %252, <4 x float> %180) #9, !dbg !14
  %lds.const.gep1243 = getelementptr i8, ptr addrspace(3) %address.local559, i64 12416, !dbg !14
  %255 = load <4 x half>, ptr addrspace(3) %lds.const.gep1243, align 8, !dbg !14
  %lds.const.gep1244 = getelementptr i8, ptr addrspace(3) %170, i64 8384, !dbg !14
  %256 = load <4 x half>, ptr addrspace(3) %lds.const.gep1244, align 8, !dbg !14
  %257 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %255, <4 x float> %254) #9, !dbg !14
  %258 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %256, <4 x float> %184) #9, !dbg !14
  %lds.const.gep1245 = getelementptr i8, ptr addrspace(3) %170, i64 12480, !dbg !14
  %259 = load <4 x half>, ptr addrspace(3) %lds.const.gep1245, align 8, !dbg !14
  %lds.const.gep1246 = getelementptr i8, ptr addrspace(3) %address.local559, i64 9216, !dbg !14
  %260 = load <4 x half>, ptr addrspace(3) %lds.const.gep1246, align 8, !dbg !14
  %261 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %259, <4 x float> %258) #9, !dbg !14
  %262 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %260, <4 x float> %188) #9, !dbg !14
  %lds.const.gep1247 = getelementptr i8, ptr addrspace(3) %address.local559, i64 13312, !dbg !14
  %263 = load <4 x half>, ptr addrspace(3) %lds.const.gep1247, align 8, !dbg !14
  %lds.const.gep1248 = getelementptr i8, ptr addrspace(3) %170, i64 9280, !dbg !14
  %264 = load <4 x half>, ptr addrspace(3) %lds.const.gep1248, align 8, !dbg !14
  %265 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %263, <4 x float> %262) #9, !dbg !14
  %266 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %264, <4 x float> %192) #9, !dbg !14
  %lds.const.gep1249 = getelementptr i8, ptr addrspace(3) %170, i64 13376, !dbg !14
  %267 = load <4 x half>, ptr addrspace(3) %lds.const.gep1249, align 8, !dbg !14
  %lds.const.gep1250 = getelementptr i8, ptr addrspace(3) %address.local559, i64 9344, !dbg !14
  %268 = load <4 x half>, ptr addrspace(3) %lds.const.gep1250, align 8, !dbg !14
  %269 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %267, <4 x float> %266) #9, !dbg !14
  %270 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %268, <4 x float> %196) #9, !dbg !14
  %lds.const.gep1251 = getelementptr i8, ptr addrspace(3) %address.local559, i64 13440, !dbg !14
  %271 = load <4 x half>, ptr addrspace(3) %lds.const.gep1251, align 8, !dbg !14
  %lds.const.gep1252 = getelementptr i8, ptr addrspace(3) %170, i64 9408, !dbg !14
  %272 = load <4 x half>, ptr addrspace(3) %lds.const.gep1252, align 8, !dbg !14
  %273 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %271, <4 x float> %270) #9, !dbg !14
  %274 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %272, <4 x float> %200) #9, !dbg !14
  %lds.const.gep1253 = getelementptr i8, ptr addrspace(3) %170, i64 13504, !dbg !14
  %275 = load <4 x half>, ptr addrspace(3) %lds.const.gep1253, align 8, !dbg !14
  %lds.const.gep1254 = getelementptr i8, ptr addrspace(3) %address.local559, i64 8192, !dbg !14
  %276 = load <4 x half>, ptr addrspace(3) %lds.const.gep1254, align 8, !dbg !14
  %277 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %275, <4 x float> %274) #9, !dbg !14
  %278 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %276, <4 x float> %204) #9, !dbg !14
  %lds.const.gep1255 = getelementptr i8, ptr addrspace(3) %address.local559, i64 12288, !dbg !14
  %279 = load <4 x half>, ptr addrspace(3) %lds.const.gep1255, align 8, !dbg !14
  %lds.const.gep1256 = getelementptr i8, ptr addrspace(3) %170, i64 8256, !dbg !14
  %280 = load <4 x half>, ptr addrspace(3) %lds.const.gep1256, align 8, !dbg !14
  %281 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %279, <4 x float> %278) #9, !dbg !14
  %282 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %280, <4 x float> %208) #9, !dbg !14
  %lds.const.gep1257 = getelementptr i8, ptr addrspace(3) %170, i64 12352, !dbg !14
  %283 = load <4 x half>, ptr addrspace(3) %lds.const.gep1257, align 8, !dbg !14
  %lds.const.gep1258 = getelementptr i8, ptr addrspace(3) %address.local559, i64 8320, !dbg !14
  %284 = load <4 x half>, ptr addrspace(3) %lds.const.gep1258, align 8, !dbg !14
  %285 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %283, <4 x float> %282) #9, !dbg !14
  %286 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %284, <4 x float> %212) #9, !dbg !14
  %lds.const.gep1259 = getelementptr i8, ptr addrspace(3) %address.local559, i64 12416, !dbg !14
  %287 = load <4 x half>, ptr addrspace(3) %lds.const.gep1259, align 8, !dbg !14
  %lds.const.gep1260 = getelementptr i8, ptr addrspace(3) %170, i64 8384, !dbg !14
  %288 = load <4 x half>, ptr addrspace(3) %lds.const.gep1260, align 8, !dbg !14
  %289 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %287, <4 x float> %286) #9, !dbg !14
  %290 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %288, <4 x float> %216) #9, !dbg !14
  %lds.const.gep1261 = getelementptr i8, ptr addrspace(3) %170, i64 12480, !dbg !14
  %291 = load <4 x half>, ptr addrspace(3) %lds.const.gep1261, align 8, !dbg !14
  %lds.const.gep1262 = getelementptr i8, ptr addrspace(3) %address.local559, i64 9216, !dbg !14
  %292 = load <4 x half>, ptr addrspace(3) %lds.const.gep1262, align 8, !dbg !14
  %293 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %291, <4 x float> %290) #9, !dbg !14
  %294 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %292, <4 x float> %220) #9, !dbg !14
  %lds.const.gep1263 = getelementptr i8, ptr addrspace(3) %address.local559, i64 13312, !dbg !14
  %295 = load <4 x half>, ptr addrspace(3) %lds.const.gep1263, align 8, !dbg !14
  %lds.const.gep1264 = getelementptr i8, ptr addrspace(3) %170, i64 9280, !dbg !14
  %296 = load <4 x half>, ptr addrspace(3) %lds.const.gep1264, align 8, !dbg !14
  %297 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %295, <4 x float> %294) #9, !dbg !14
  %298 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %296, <4 x float> %224) #9, !dbg !14
  %lds.const.gep1265 = getelementptr i8, ptr addrspace(3) %170, i64 13376, !dbg !14
  %299 = load <4 x half>, ptr addrspace(3) %lds.const.gep1265, align 8, !dbg !14
  %lds.const.gep1266 = getelementptr i8, ptr addrspace(3) %address.local559, i64 9344, !dbg !14
  %300 = load <4 x half>, ptr addrspace(3) %lds.const.gep1266, align 8, !dbg !14
  %301 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %299, <4 x float> %298) #9, !dbg !14
  %302 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %300, <4 x float> %228) #9, !dbg !14
  %lds.const.gep1267 = getelementptr i8, ptr addrspace(3) %address.local559, i64 13440, !dbg !14
  %303 = load <4 x half>, ptr addrspace(3) %lds.const.gep1267, align 8, !dbg !14
  %lds.const.gep1268 = getelementptr i8, ptr addrspace(3) %170, i64 9408, !dbg !14
  %304 = load <4 x half>, ptr addrspace(3) %lds.const.gep1268, align 8, !dbg !14
  %305 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %303, <4 x float> %302) #9, !dbg !14
  %306 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %304, <4 x float> %231) #9, !dbg !14
  %lds.const.gep1269 = getelementptr i8, ptr addrspace(3) %170, i64 13504, !dbg !14
  %307 = load <4 x half>, ptr addrspace(3) %lds.const.gep1269, align 8, !dbg !14
  %308 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %307, <4 x float> %306) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %309 = or disjoint i64 %134, 96, !dbg !12
  %310 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local535), !dbg !12
  %311 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !12
  %312 = insertelement <2 x i64> poison, i64 %311, i64 0, !dbg !12
  %313 = bitcast <2 x i64> %312 to <4 x i32>, !dbg !12
  %dma.buffer.rsrc1355 = shufflevector <4 x i32> %313, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !12
  %dma.buffer.lds1356 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %310), !dbg !12
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
  %kdma.source1712 = or disjoint i64 %309, %kdma.source1711, !dbg !12
  %kdma.source1713 = trunc nuw i64 %kdma.source1712 to i32, !dbg !12
  %kdma.source1714 = shl nuw nsw i32 %kdma.source1713, 1, !dbg !12
  %kdma.offset1715 = add i32 %kdma.source1714, %kdma.intra, !dbg !12
  %kdma.lds1716 = add i32 %dma.buffer.lds1356, 0, !dbg !12
  %314 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1716, i32 %kdma.offset1715, <4 x i32> %dma.buffer.rsrc1355) #10, !dbg !12
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
  %kdma.source1728 = or disjoint i64 %309, %kdma.source1727, !dbg !12
  %kdma.source1729 = trunc nuw i64 %kdma.source1728 to i32, !dbg !12
  %kdma.source1730 = shl nuw nsw i32 %kdma.source1729, 1, !dbg !12
  %kdma.offset1731 = add i32 %kdma.source1730, %kdma.intra, !dbg !12
  %kdma.lds1732 = add i32 %dma.buffer.lds1356, 512, !dbg !12
  %315 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1732, i32 %kdma.offset1731, <4 x i32> %dma.buffer.rsrc1355) #10, !dbg !12
  %316 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local539), !dbg !12
  %317 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !12
  %318 = insertelement <2 x i64> poison, i64 %317, i64 0, !dbg !12
  %319 = bitcast <2 x i64> %318 to <4 x i32>, !dbg !12
  %dma.buffer.rsrc1352 = shufflevector <4 x i32> %319, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !12
  %dma.buffer.lds1353 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %316), !dbg !12
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
  %kdma.source1745 = or disjoint i64 %309, %kdma.source1744, !dbg !12
  %kdma.source1746 = trunc nuw i64 %kdma.source1745 to i32, !dbg !12
  %kdma.source1747 = shl nuw nsw i32 %kdma.source1746, 1, !dbg !12
  %kdma.offset1748 = add i32 %kdma.source1747, %kdma.intra, !dbg !12
  %kdma.lds1749 = add i32 %dma.buffer.lds1353, 0, !dbg !12
  %320 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1749, i32 %kdma.offset1748, <4 x i32> %dma.buffer.rsrc1352) #10, !dbg !12
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
  %kdma.source1762 = or disjoint i64 %309, %kdma.source1761, !dbg !12
  %kdma.source1763 = trunc nuw i64 %kdma.source1762 to i32, !dbg !12
  %kdma.source1764 = shl nuw nsw i32 %kdma.source1763, 1, !dbg !12
  %kdma.offset1765 = add i32 %kdma.source1764, %kdma.intra, !dbg !12
  %kdma.lds1766 = add i32 %dma.buffer.lds1353, 512, !dbg !12
  %321 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1766, i32 %kdma.offset1765, <4 x i32> %dma.buffer.rsrc1352) #10, !dbg !12
  %322 = load <4 x half>, ptr addrspace(3) %address.local559, align 8, !dbg !14
  %323 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %322, <4 x float> %249) #9, !dbg !14
  %lds.const.gep1270 = getelementptr i8, ptr addrspace(3) %address.local559, i64 4096, !dbg !14
  %324 = load <4 x half>, ptr addrspace(3) %lds.const.gep1270, align 8, !dbg !14
  %325 = load <4 x half>, ptr addrspace(3) %address.local588, align 8, !dbg !14
  %326 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %324, <4 x float> %323) #9, !dbg !14
  %327 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %325, <4 x float> %253) #9, !dbg !14
  %lds.const.gep1271 = getelementptr i8, ptr addrspace(3) %170, i64 4160, !dbg !14
  %328 = load <4 x half>, ptr addrspace(3) %lds.const.gep1271, align 8, !dbg !14
  %lds.const.gep1272 = getelementptr i8, ptr addrspace(3) %address.local559, i64 128, !dbg !14
  %329 = load <4 x half>, ptr addrspace(3) %lds.const.gep1272, align 8, !dbg !14
  %330 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %328, <4 x float> %327) #9, !dbg !14
  %331 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %329, <4 x float> %257) #9, !dbg !14
  %lds.const.gep1273 = getelementptr i8, ptr addrspace(3) %address.local559, i64 4224, !dbg !14
  %332 = load <4 x half>, ptr addrspace(3) %lds.const.gep1273, align 8, !dbg !14
  %lds.const.gep1274 = getelementptr i8, ptr addrspace(3) %170, i64 192, !dbg !14
  %333 = load <4 x half>, ptr addrspace(3) %lds.const.gep1274, align 8, !dbg !14
  %334 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %332, <4 x float> %331) #9, !dbg !14
  %335 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %333, <4 x float> %261) #9, !dbg !14
  %lds.const.gep1275 = getelementptr i8, ptr addrspace(3) %170, i64 4288, !dbg !14
  %336 = load <4 x half>, ptr addrspace(3) %lds.const.gep1275, align 8, !dbg !14
  %lds.const.gep1276 = getelementptr i8, ptr addrspace(3) %address.local559, i64 1024, !dbg !14
  %337 = load <4 x half>, ptr addrspace(3) %lds.const.gep1276, align 8, !dbg !14
  %338 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %336, <4 x float> %335) #9, !dbg !14
  %339 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %337, <4 x float> %265) #9, !dbg !14
  %lds.const.gep1277 = getelementptr i8, ptr addrspace(3) %address.local559, i64 5120, !dbg !14
  %340 = load <4 x half>, ptr addrspace(3) %lds.const.gep1277, align 8, !dbg !14
  %lds.const.gep1278 = getelementptr i8, ptr addrspace(3) %170, i64 1088, !dbg !14
  %341 = load <4 x half>, ptr addrspace(3) %lds.const.gep1278, align 8, !dbg !14
  %342 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %340, <4 x float> %339) #9, !dbg !14
  %343 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %341, <4 x float> %269) #9, !dbg !14
  %lds.const.gep1279 = getelementptr i8, ptr addrspace(3) %170, i64 5184, !dbg !14
  %344 = load <4 x half>, ptr addrspace(3) %lds.const.gep1279, align 8, !dbg !14
  %lds.const.gep1280 = getelementptr i8, ptr addrspace(3) %address.local559, i64 1152, !dbg !14
  %345 = load <4 x half>, ptr addrspace(3) %lds.const.gep1280, align 8, !dbg !14
  %346 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %344, <4 x float> %343) #9, !dbg !14
  %347 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %345, <4 x float> %273) #9, !dbg !14
  %lds.const.gep1281 = getelementptr i8, ptr addrspace(3) %address.local559, i64 5248, !dbg !14
  %348 = load <4 x half>, ptr addrspace(3) %lds.const.gep1281, align 8, !dbg !14
  %lds.const.gep1282 = getelementptr i8, ptr addrspace(3) %170, i64 1216, !dbg !14
  %349 = load <4 x half>, ptr addrspace(3) %lds.const.gep1282, align 8, !dbg !14
  %350 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %348, <4 x float> %347) #9, !dbg !14
  %351 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %349, <4 x float> %277) #9, !dbg !14
  %lds.const.gep1283 = getelementptr i8, ptr addrspace(3) %170, i64 5312, !dbg !14
  %352 = load <4 x half>, ptr addrspace(3) %lds.const.gep1283, align 8, !dbg !14
  %353 = load <4 x half>, ptr addrspace(3) %address.local559, align 8, !dbg !14
  %354 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %352, <4 x float> %351) #9, !dbg !14
  %355 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %353, <4 x float> %281) #9, !dbg !14
  %lds.const.gep1284 = getelementptr i8, ptr addrspace(3) %address.local559, i64 4096, !dbg !14
  %356 = load <4 x half>, ptr addrspace(3) %lds.const.gep1284, align 8, !dbg !14
  %357 = load <4 x half>, ptr addrspace(3) %address.local588, align 8, !dbg !14
  %358 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %356, <4 x float> %355) #9, !dbg !14
  %359 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %357, <4 x float> %285) #9, !dbg !14
  %lds.const.gep1285 = getelementptr i8, ptr addrspace(3) %170, i64 4160, !dbg !14
  %360 = load <4 x half>, ptr addrspace(3) %lds.const.gep1285, align 8, !dbg !14
  %lds.const.gep1286 = getelementptr i8, ptr addrspace(3) %address.local559, i64 128, !dbg !14
  %361 = load <4 x half>, ptr addrspace(3) %lds.const.gep1286, align 8, !dbg !14
  %362 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %360, <4 x float> %359) #9, !dbg !14
  %363 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %361, <4 x float> %289) #9, !dbg !14
  %lds.const.gep1287 = getelementptr i8, ptr addrspace(3) %address.local559, i64 4224, !dbg !14
  %364 = load <4 x half>, ptr addrspace(3) %lds.const.gep1287, align 8, !dbg !14
  %lds.const.gep1288 = getelementptr i8, ptr addrspace(3) %170, i64 192, !dbg !14
  %365 = load <4 x half>, ptr addrspace(3) %lds.const.gep1288, align 8, !dbg !14
  %366 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %364, <4 x float> %363) #9, !dbg !14
  %367 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %365, <4 x float> %293) #9, !dbg !14
  %lds.const.gep1289 = getelementptr i8, ptr addrspace(3) %170, i64 4288, !dbg !14
  %368 = load <4 x half>, ptr addrspace(3) %lds.const.gep1289, align 8, !dbg !14
  %lds.const.gep1290 = getelementptr i8, ptr addrspace(3) %address.local559, i64 1024, !dbg !14
  %369 = load <4 x half>, ptr addrspace(3) %lds.const.gep1290, align 8, !dbg !14
  %370 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %368, <4 x float> %367) #9, !dbg !14
  %371 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %369, <4 x float> %297) #9, !dbg !14
  %lds.const.gep1291 = getelementptr i8, ptr addrspace(3) %address.local559, i64 5120, !dbg !14
  %372 = load <4 x half>, ptr addrspace(3) %lds.const.gep1291, align 8, !dbg !14
  %lds.const.gep1292 = getelementptr i8, ptr addrspace(3) %170, i64 1088, !dbg !14
  %373 = load <4 x half>, ptr addrspace(3) %lds.const.gep1292, align 8, !dbg !14
  %374 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %372, <4 x float> %371) #9, !dbg !14
  %375 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %373, <4 x float> %301) #9, !dbg !14
  %lds.const.gep1293 = getelementptr i8, ptr addrspace(3) %170, i64 5184, !dbg !14
  %376 = load <4 x half>, ptr addrspace(3) %lds.const.gep1293, align 8, !dbg !14
  %lds.const.gep1294 = getelementptr i8, ptr addrspace(3) %address.local559, i64 1152, !dbg !14
  %377 = load <4 x half>, ptr addrspace(3) %lds.const.gep1294, align 8, !dbg !14
  %378 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %376, <4 x float> %375) #9, !dbg !14
  %379 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %377, <4 x float> %305) #9, !dbg !14
  %lds.const.gep1295 = getelementptr i8, ptr addrspace(3) %address.local559, i64 5248, !dbg !14
  %380 = load <4 x half>, ptr addrspace(3) %lds.const.gep1295, align 8, !dbg !14
  %lds.const.gep1296 = getelementptr i8, ptr addrspace(3) %170, i64 1216, !dbg !14
  %381 = load <4 x half>, ptr addrspace(3) %lds.const.gep1296, align 8, !dbg !14
  %382 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %380, <4 x float> %379) #9, !dbg !14
  %383 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %381, <4 x float> %308) #9, !dbg !14
  %lds.const.gep1297 = getelementptr i8, ptr addrspace(3) %170, i64 5312, !dbg !14
  %384 = load <4 x half>, ptr addrspace(3) %lds.const.gep1297, align 8, !dbg !14
  %385 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %384, <4 x float> %383) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %386 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local504), !dbg !15
  %address.local917 = shl nuw nsw i64 %address.local499, 5, !dbg !15
  %address.local918 = and i64 %address.local917, 8064, !dbg !15
  %address.local919 = shl nuw nsw i64 %address.local499, 3, !dbg !15
  %address.local920 = and i64 %address.local919, 24, !dbg !15
  %address.local921 = or disjoint i64 %address.local918, %address.local920, !dbg !15
  %387 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %.idx14191420 = or disjoint i64 %134, %address.local921, !dbg !15
  %.idx14191420.tr = trunc nuw i64 %.idx14191420 to i32, !dbg !15
  %dma.buffer.offset1348 = shl nuw nsw i32 %.idx14191420.tr, 1, !dbg !15
  %388 = insertelement <2 x i64> poison, i64 %387, i64 0, !dbg !15
  %389 = bitcast <2 x i64> %388 to <4 x i32>, !dbg !15
  %dma.buffer.rsrc1349 = shufflevector <4 x i32> %389, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %dma.buffer.lds1350 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %386), !dbg !15
  %390 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1350, i32 %dma.buffer.offset1348, <4 x i32> %dma.buffer.rsrc1349) #10, !dbg !15
  %391 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local521), !dbg !15
  %address.local923 = shl nuw nsw i64 %address.local516, 5, !dbg !15
  %address.local924 = and i64 %address.local923, 16256, !dbg !15
  %address.local925 = shl nuw nsw i64 %address.local499, 3, !dbg !15
  %address.local926 = and i64 %address.local925, 24, !dbg !15
  %address.local927 = or disjoint i64 %address.local924, %address.local926, !dbg !15
  %392 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %.idx14211422 = or disjoint i64 %134, %address.local927, !dbg !15
  %.idx14211422.tr = trunc nuw i64 %.idx14211422 to i32, !dbg !15
  %dma.buffer.offset1345 = shl nuw nsw i32 %.idx14211422.tr, 1, !dbg !15
  %393 = insertelement <2 x i64> poison, i64 %392, i64 0, !dbg !15
  %394 = bitcast <2 x i64> %393 to <4 x i32>, !dbg !15
  %dma.buffer.rsrc1346 = shufflevector <4 x i32> %394, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %dma.buffer.lds1347 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %391), !dbg !15
  %395 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1347, i32 %dma.buffer.offset1345, <4 x i32> %dma.buffer.rsrc1346) #10, !dbg !15
  %lds.const.gep1298 = getelementptr i8, ptr addrspace(3) %address.local559, i64 8192, !dbg !14
  %396 = load <4 x half>, ptr addrspace(3) %lds.const.gep1298, align 8, !dbg !14
  %397 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %108, <4 x half> %396, <4 x float> %326) #9, !dbg !14
  %lds.const.gep1299 = getelementptr i8, ptr addrspace(3) %address.local559, i64 12288, !dbg !14
  %398 = load <4 x half>, ptr addrspace(3) %lds.const.gep1299, align 8, !dbg !14
  %lds.const.gep1300 = getelementptr i8, ptr addrspace(3) %170, i64 8256, !dbg !14
  %399 = load <4 x half>, ptr addrspace(3) %lds.const.gep1300, align 8, !dbg !14
  %400 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %110, <4 x half> %398, <4 x float> %397) #9, !dbg !14
  %401 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %108, <4 x half> %399, <4 x float> %330) #9, !dbg !14
  %402 = extractelement <4 x float> %400, i64 0, !dbg !19
  %403 = call float @llvm.maxnum.f32(float %402, float 0xFFF0000000000000), !dbg !20, !frisk.local_reduction !1
  %404 = extractelement <4 x float> %400, i64 1, !dbg !19
  %405 = extractelement <4 x float> %400, i64 2, !dbg !19
  %406 = extractelement <4 x float> %400, i64 3, !dbg !19
  %lds.const.gep1301 = getelementptr i8, ptr addrspace(3) %170, i64 12352, !dbg !14
  %407 = load <4 x half>, ptr addrspace(3) %lds.const.gep1301, align 8, !dbg !14
  %lds.const.gep1302 = getelementptr i8, ptr addrspace(3) %address.local559, i64 8320, !dbg !14
  %408 = load <4 x half>, ptr addrspace(3) %lds.const.gep1302, align 8, !dbg !14
  %409 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %110, <4 x half> %407, <4 x float> %401) #9, !dbg !14
  %410 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %108, <4 x half> %408, <4 x float> %334) #9, !dbg !14
  %411 = extractelement <4 x float> %409, i64 0, !dbg !19
  %local.reduce151 = call float @llvm.maxnum.f32(float %403, float %411), !dbg !20, !frisk.local_reduction !1
  %412 = extractelement <4 x float> %409, i64 1, !dbg !19
  %local.reduce152 = call float @llvm.maxnum.f32(float %404, float %412), !dbg !20, !frisk.local_reduction !1
  %413 = extractelement <4 x float> %409, i64 2, !dbg !19
  %local.reduce153 = call float @llvm.maxnum.f32(float %405, float %413), !dbg !20, !frisk.local_reduction !1
  %414 = extractelement <4 x float> %409, i64 3, !dbg !19
  %local.reduce154 = call float @llvm.maxnum.f32(float %406, float %414), !dbg !20, !frisk.local_reduction !1
  %lds.const.gep1303 = getelementptr i8, ptr addrspace(3) %address.local559, i64 12416, !dbg !14
  %415 = load <4 x half>, ptr addrspace(3) %lds.const.gep1303, align 8, !dbg !14
  %lds.const.gep1304 = getelementptr i8, ptr addrspace(3) %170, i64 8384, !dbg !14
  %416 = load <4 x half>, ptr addrspace(3) %lds.const.gep1304, align 8, !dbg !14
  %417 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %110, <4 x half> %415, <4 x float> %410) #9, !dbg !14
  %418 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %108, <4 x half> %416, <4 x float> %338) #9, !dbg !14
  %419 = extractelement <4 x float> %417, i64 0, !dbg !19
  %local.reduce155 = call float @llvm.maxnum.f32(float %local.reduce151, float %419), !dbg !20, !frisk.local_reduction !1
  %420 = extractelement <4 x float> %417, i64 1, !dbg !19
  %local.reduce156 = call float @llvm.maxnum.f32(float %local.reduce152, float %420), !dbg !20, !frisk.local_reduction !1
  %421 = extractelement <4 x float> %417, i64 2, !dbg !19
  %local.reduce157 = call float @llvm.maxnum.f32(float %local.reduce153, float %421), !dbg !20, !frisk.local_reduction !1
  %422 = extractelement <4 x float> %417, i64 3, !dbg !19
  %local.reduce158 = call float @llvm.maxnum.f32(float %local.reduce154, float %422), !dbg !20, !frisk.local_reduction !1
  %lds.const.gep1305 = getelementptr i8, ptr addrspace(3) %170, i64 12480, !dbg !14
  %423 = load <4 x half>, ptr addrspace(3) %lds.const.gep1305, align 8, !dbg !14
  %lds.const.gep1306 = getelementptr i8, ptr addrspace(3) %address.local559, i64 9216, !dbg !14
  %424 = load <4 x half>, ptr addrspace(3) %lds.const.gep1306, align 8, !dbg !14
  %425 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %110, <4 x half> %423, <4 x float> %418) #9, !dbg !14
  %426 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %108, <4 x half> %424, <4 x float> %342) #9, !dbg !14
  %427 = extractelement <4 x float> %425, i64 0, !dbg !19
  %local.reduce159 = call float @llvm.maxnum.f32(float %local.reduce155, float %427), !dbg !20, !frisk.local_reduction !1
  %428 = extractelement <4 x float> %425, i64 1, !dbg !19
  %local.reduce160 = call float @llvm.maxnum.f32(float %local.reduce156, float %428), !dbg !20, !frisk.local_reduction !1
  %429 = extractelement <4 x float> %425, i64 2, !dbg !19
  %local.reduce161 = call float @llvm.maxnum.f32(float %local.reduce157, float %429), !dbg !20, !frisk.local_reduction !1
  %430 = extractelement <4 x float> %425, i64 3, !dbg !19
  %local.reduce162 = call float @llvm.maxnum.f32(float %local.reduce158, float %430), !dbg !20, !frisk.local_reduction !1
  %lds.const.gep1307 = getelementptr i8, ptr addrspace(3) %address.local559, i64 13312, !dbg !14
  %431 = load <4 x half>, ptr addrspace(3) %lds.const.gep1307, align 8, !dbg !14
  %lds.const.gep1308 = getelementptr i8, ptr addrspace(3) %170, i64 9280, !dbg !14
  %432 = load <4 x half>, ptr addrspace(3) %lds.const.gep1308, align 8, !dbg !14
  %433 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %110, <4 x half> %431, <4 x float> %426) #9, !dbg !14
  %434 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %108, <4 x half> %432, <4 x float> %346) #9, !dbg !14
  %435 = extractelement <4 x float> %433, i64 0, !dbg !19
  %local.reduce163 = call float @llvm.maxnum.f32(float %local.reduce159, float %435), !dbg !20, !frisk.local_reduction !1
  %436 = extractelement <4 x float> %433, i64 1, !dbg !19
  %local.reduce164 = call float @llvm.maxnum.f32(float %local.reduce160, float %436), !dbg !20, !frisk.local_reduction !1
  %437 = extractelement <4 x float> %433, i64 2, !dbg !19
  %local.reduce165 = call float @llvm.maxnum.f32(float %local.reduce161, float %437), !dbg !20, !frisk.local_reduction !1
  %438 = extractelement <4 x float> %433, i64 3, !dbg !19
  %local.reduce166 = call float @llvm.maxnum.f32(float %local.reduce162, float %438), !dbg !20, !frisk.local_reduction !1
  %lds.const.gep1309 = getelementptr i8, ptr addrspace(3) %170, i64 13376, !dbg !14
  %439 = load <4 x half>, ptr addrspace(3) %lds.const.gep1309, align 8, !dbg !14
  %lds.const.gep1310 = getelementptr i8, ptr addrspace(3) %address.local559, i64 9344, !dbg !14
  %440 = load <4 x half>, ptr addrspace(3) %lds.const.gep1310, align 8, !dbg !14
  %441 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %110, <4 x half> %439, <4 x float> %434) #9, !dbg !14
  %442 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %108, <4 x half> %440, <4 x float> %350) #9, !dbg !14
  %443 = extractelement <4 x float> %441, i64 0, !dbg !19
  %local.reduce167 = call float @llvm.maxnum.f32(float %local.reduce163, float %443), !dbg !20, !frisk.local_reduction !1
  %444 = extractelement <4 x float> %441, i64 1, !dbg !19
  %local.reduce168 = call float @llvm.maxnum.f32(float %local.reduce164, float %444), !dbg !20, !frisk.local_reduction !1
  %445 = extractelement <4 x float> %441, i64 2, !dbg !19
  %local.reduce169 = call float @llvm.maxnum.f32(float %local.reduce165, float %445), !dbg !20, !frisk.local_reduction !1
  %446 = extractelement <4 x float> %441, i64 3, !dbg !19
  %local.reduce170 = call float @llvm.maxnum.f32(float %local.reduce166, float %446), !dbg !20, !frisk.local_reduction !1
  %lds.const.gep1311 = getelementptr i8, ptr addrspace(3) %address.local559, i64 13440, !dbg !14
  %447 = load <4 x half>, ptr addrspace(3) %lds.const.gep1311, align 8, !dbg !14
  %lds.const.gep1312 = getelementptr i8, ptr addrspace(3) %170, i64 9408, !dbg !14
  %448 = load <4 x half>, ptr addrspace(3) %lds.const.gep1312, align 8, !dbg !14
  %449 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %110, <4 x half> %447, <4 x float> %442) #9, !dbg !14
  %450 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %108, <4 x half> %448, <4 x float> %354) #9, !dbg !14
  %451 = extractelement <4 x float> %449, i64 0, !dbg !19
  %local.reduce171 = call float @llvm.maxnum.f32(float %local.reduce167, float %451), !dbg !20, !frisk.local_reduction !1
  %452 = extractelement <4 x float> %449, i64 1, !dbg !19
  %local.reduce172 = call float @llvm.maxnum.f32(float %local.reduce168, float %452), !dbg !20, !frisk.local_reduction !1
  %453 = extractelement <4 x float> %449, i64 2, !dbg !19
  %local.reduce173 = call float @llvm.maxnum.f32(float %local.reduce169, float %453), !dbg !20, !frisk.local_reduction !1
  %454 = extractelement <4 x float> %449, i64 3, !dbg !19
  %local.reduce174 = call float @llvm.maxnum.f32(float %local.reduce170, float %454), !dbg !20, !frisk.local_reduction !1
  %lds.const.gep1313 = getelementptr i8, ptr addrspace(3) %170, i64 13504, !dbg !14
  %455 = load <4 x half>, ptr addrspace(3) %lds.const.gep1313, align 8, !dbg !14
  %lds.const.gep1314 = getelementptr i8, ptr addrspace(3) %address.local559, i64 8192, !dbg !14
  %456 = load <4 x half>, ptr addrspace(3) %lds.const.gep1314, align 8, !dbg !14
  %457 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %110, <4 x half> %455, <4 x float> %450) #9, !dbg !14
  %458 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %111, <4 x half> %456, <4 x float> %358) #9, !dbg !14
  %459 = extractelement <4 x float> %457, i64 0, !dbg !19
  %local.reduce175 = call float @llvm.maxnum.f32(float %local.reduce171, float %459), !dbg !20, !frisk.local_reduction !1
  %460 = extractelement <4 x float> %457, i64 1, !dbg !19
  %local.reduce176 = call float @llvm.maxnum.f32(float %local.reduce172, float %460), !dbg !20, !frisk.local_reduction !1
  %461 = extractelement <4 x float> %457, i64 2, !dbg !19
  %local.reduce177 = call float @llvm.maxnum.f32(float %local.reduce173, float %461), !dbg !20, !frisk.local_reduction !1
  %462 = extractelement <4 x float> %457, i64 3, !dbg !19
  %local.reduce178 = call float @llvm.maxnum.f32(float %local.reduce174, float %462), !dbg !20, !frisk.local_reduction !1
  %local.reduce179 = call float @llvm.maxnum.f32(float %local.reduce175, float %local.reduce176), !dbg !20, !frisk.local_reduction !1
  %local.reduce180 = call float @llvm.maxnum.f32(float %local.reduce177, float %local.reduce178), !dbg !20, !frisk.local_reduction !1
  %463 = call float @llvm.maxnum.f32(float %local.reduce179, float %local.reduce180), !dbg !20, !frisk.local_reduction !1
  %lds.const.gep1315 = getelementptr i8, ptr addrspace(3) %address.local559, i64 12288, !dbg !14
  %464 = load <4 x half>, ptr addrspace(3) %lds.const.gep1315, align 8, !dbg !14
  %lds.const.gep1316 = getelementptr i8, ptr addrspace(3) %170, i64 8256, !dbg !14
  %465 = load <4 x half>, ptr addrspace(3) %lds.const.gep1316, align 8, !dbg !14
  %466 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %112, <4 x half> %464, <4 x float> %458) #9, !dbg !14
  %467 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %111, <4 x half> %465, <4 x float> %362) #9, !dbg !14
  %468 = extractelement <4 x float> %466, i64 0, !dbg !19
  %469 = extractelement <4 x float> %466, i64 1, !dbg !19
  %470 = extractelement <4 x float> %466, i64 2, !dbg !19
  %471 = extractelement <4 x float> %466, i64 3, !dbg !19
  %lds.const.gep1317 = getelementptr i8, ptr addrspace(3) %170, i64 12352, !dbg !14
  %472 = load <4 x half>, ptr addrspace(3) %lds.const.gep1317, align 8, !dbg !14
  %lds.const.gep1318 = getelementptr i8, ptr addrspace(3) %address.local559, i64 8320, !dbg !14
  %473 = load <4 x half>, ptr addrspace(3) %lds.const.gep1318, align 8, !dbg !14
  %474 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %112, <4 x half> %472, <4 x float> %467) #9, !dbg !14
  %475 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %111, <4 x half> %473, <4 x float> %366) #9, !dbg !14
  %476 = extractelement <4 x float> %474, i64 0, !dbg !19
  %477 = extractelement <4 x float> %474, i64 1, !dbg !19
  %478 = extractelement <4 x float> %474, i64 2, !dbg !19
  %479 = extractelement <4 x float> %474, i64 3, !dbg !19
  %lds.const.gep1319 = getelementptr i8, ptr addrspace(3) %address.local559, i64 12416, !dbg !14
  %480 = load <4 x half>, ptr addrspace(3) %lds.const.gep1319, align 8, !dbg !14
  %lds.const.gep1320 = getelementptr i8, ptr addrspace(3) %170, i64 8384, !dbg !14
  %481 = load <4 x half>, ptr addrspace(3) %lds.const.gep1320, align 8, !dbg !14
  %482 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %112, <4 x half> %480, <4 x float> %475) #9, !dbg !14
  %483 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %111, <4 x half> %481, <4 x float> %370) #9, !dbg !14
  %484 = extractelement <4 x float> %482, i64 0, !dbg !19
  %485 = extractelement <4 x float> %482, i64 1, !dbg !19
  %486 = extractelement <4 x float> %482, i64 2, !dbg !19
  %487 = extractelement <4 x float> %482, i64 3, !dbg !19
  %lds.const.gep1321 = getelementptr i8, ptr addrspace(3) %170, i64 12480, !dbg !14
  %488 = load <4 x half>, ptr addrspace(3) %lds.const.gep1321, align 8, !dbg !14
  %lds.const.gep1322 = getelementptr i8, ptr addrspace(3) %address.local559, i64 9216, !dbg !14
  %489 = load <4 x half>, ptr addrspace(3) %lds.const.gep1322, align 8, !dbg !14
  %490 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %112, <4 x half> %488, <4 x float> %483) #9, !dbg !14
  %491 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %111, <4 x half> %489, <4 x float> %374) #9, !dbg !14
  %492 = extractelement <4 x float> %490, i64 0, !dbg !19
  %493 = extractelement <4 x float> %490, i64 1, !dbg !19
  %494 = extractelement <4 x float> %490, i64 2, !dbg !19
  %495 = extractelement <4 x float> %490, i64 3, !dbg !19
  %lds.const.gep1323 = getelementptr i8, ptr addrspace(3) %address.local559, i64 13312, !dbg !14
  %496 = load <4 x half>, ptr addrspace(3) %lds.const.gep1323, align 8, !dbg !14
  %lds.const.gep1324 = getelementptr i8, ptr addrspace(3) %170, i64 9280, !dbg !14
  %497 = load <4 x half>, ptr addrspace(3) %lds.const.gep1324, align 8, !dbg !14
  %498 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %112, <4 x half> %496, <4 x float> %491) #9, !dbg !14
  %499 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %111, <4 x half> %497, <4 x float> %378) #9, !dbg !14
  %500 = extractelement <4 x float> %498, i64 0, !dbg !19
  %501 = extractelement <4 x float> %498, i64 1, !dbg !19
  %502 = extractelement <4 x float> %498, i64 2, !dbg !19
  %503 = extractelement <4 x float> %498, i64 3, !dbg !19
  %lds.const.gep1325 = getelementptr i8, ptr addrspace(3) %170, i64 13376, !dbg !14
  %504 = load <4 x half>, ptr addrspace(3) %lds.const.gep1325, align 8, !dbg !14
  %lds.const.gep1326 = getelementptr i8, ptr addrspace(3) %address.local559, i64 9344, !dbg !14
  %505 = load <4 x half>, ptr addrspace(3) %lds.const.gep1326, align 8, !dbg !14
  %506 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %112, <4 x half> %504, <4 x float> %499) #9, !dbg !14
  %507 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %111, <4 x half> %505, <4 x float> %382) #9, !dbg !14
  %508 = extractelement <4 x float> %506, i64 0, !dbg !19
  %509 = extractelement <4 x float> %506, i64 1, !dbg !19
  %510 = extractelement <4 x float> %506, i64 2, !dbg !19
  %511 = extractelement <4 x float> %506, i64 3, !dbg !19
  %lds.const.gep1327 = getelementptr i8, ptr addrspace(3) %address.local559, i64 13440, !dbg !14
  %512 = load <4 x half>, ptr addrspace(3) %lds.const.gep1327, align 8, !dbg !14
  %lds.const.gep1328 = getelementptr i8, ptr addrspace(3) %170, i64 9408, !dbg !14
  %513 = load <4 x half>, ptr addrspace(3) %lds.const.gep1328, align 8, !dbg !14
  %514 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %112, <4 x half> %512, <4 x float> %507) #9, !dbg !14
  %515 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %111, <4 x half> %513, <4 x float> %385) #9, !dbg !14
  %516 = extractelement <4 x float> %514, i64 0, !dbg !19
  %517 = extractelement <4 x float> %514, i64 1, !dbg !19
  %518 = extractelement <4 x float> %514, i64 2, !dbg !19
  %519 = extractelement <4 x float> %514, i64 3, !dbg !19
  %lds.const.gep1329 = getelementptr i8, ptr addrspace(3) %170, i64 13504, !dbg !14
  %520 = load <4 x half>, ptr addrspace(3) %lds.const.gep1329, align 8, !dbg !14
  %521 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %112, <4 x half> %520, <4 x float> %515) #9, !dbg !14
  %522 = extractelement <4 x float> %521, i64 0, !dbg !19
  %523 = extractelement <4 x float> %521, i64 1, !dbg !19
  %524 = extractelement <4 x float> %521, i64 2, !dbg !19
  %525 = extractelement <4 x float> %521, i64 3, !dbg !19
  %526 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !20
  %527 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %526), !dbg !20
  %528 = add nuw nsw i32 %527, 64, !dbg !20
  %529 = xor i32 %527, 16, !dbg !20
  %530 = and i32 %528, 192, !dbg !20
  %531 = icmp samesign ult i32 %529, %530, !dbg !20
  %532 = select i1 %531, i32 %529, i32 %527, !dbg !20
  %533 = shl nuw nsw i32 %532, 2, !dbg !20
  %534 = bitcast float %463 to i32, !dbg !20
  %535 = call i32 @llvm.amdgcn.ds.bpermute(i32 %533, i32 %534), !dbg !20
  %536 = bitcast i32 %535 to float, !dbg !20
  %537 = call float @llvm.maxnum.f32(float %463, float %536), !dbg !20
  %538 = xor i32 %527, 32, !dbg !20
  %539 = icmp samesign ult i32 %538, %530, !dbg !20
  %540 = select i1 %539, i32 %538, i32 %527, !dbg !20
  %541 = shl nuw nsw i32 %540, 2, !dbg !20
  %542 = bitcast float %537 to i32, !dbg !20
  %543 = call i32 @llvm.amdgcn.ds.bpermute(i32 %541, i32 %542), !dbg !20
  %544 = bitcast i32 %543 to float, !dbg !20
  %545 = call float @llvm.maxnum.f32(float %537, float %544), !dbg !20
  %546 = extractvalue [2 x <1 x float>] %125, 0, !dbg !20
  %547 = insertelement <1 x float> poison, float %545, i64 0, !dbg !20
  %548 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %547, <1 x float> %546), !dbg !20
  %549 = call float @llvm.maxnum.f32(float %468, float 0xFFF0000000000000), !dbg !20, !frisk.local_reduction !1
  %local.reduce120 = call float @llvm.maxnum.f32(float %549, float %476), !dbg !20, !frisk.local_reduction !1
  %local.reduce121 = call float @llvm.maxnum.f32(float %469, float %477), !dbg !20, !frisk.local_reduction !1
  %local.reduce122 = call float @llvm.maxnum.f32(float %470, float %478), !dbg !20, !frisk.local_reduction !1
  %local.reduce123 = call float @llvm.maxnum.f32(float %471, float %479), !dbg !20, !frisk.local_reduction !1
  %local.reduce124 = call float @llvm.maxnum.f32(float %local.reduce120, float %484), !dbg !20, !frisk.local_reduction !1
  %local.reduce125 = call float @llvm.maxnum.f32(float %local.reduce121, float %485), !dbg !20, !frisk.local_reduction !1
  %local.reduce126 = call float @llvm.maxnum.f32(float %local.reduce122, float %486), !dbg !20, !frisk.local_reduction !1
  %local.reduce127 = call float @llvm.maxnum.f32(float %local.reduce123, float %487), !dbg !20, !frisk.local_reduction !1
  %local.reduce128 = call float @llvm.maxnum.f32(float %local.reduce124, float %492), !dbg !20, !frisk.local_reduction !1
  %local.reduce129 = call float @llvm.maxnum.f32(float %local.reduce125, float %493), !dbg !20, !frisk.local_reduction !1
  %local.reduce130 = call float @llvm.maxnum.f32(float %local.reduce126, float %494), !dbg !20, !frisk.local_reduction !1
  %local.reduce131 = call float @llvm.maxnum.f32(float %local.reduce127, float %495), !dbg !20, !frisk.local_reduction !1
  %local.reduce132 = call float @llvm.maxnum.f32(float %local.reduce128, float %500), !dbg !20, !frisk.local_reduction !1
  %local.reduce133 = call float @llvm.maxnum.f32(float %local.reduce129, float %501), !dbg !20, !frisk.local_reduction !1
  %local.reduce134 = call float @llvm.maxnum.f32(float %local.reduce130, float %502), !dbg !20, !frisk.local_reduction !1
  %local.reduce135 = call float @llvm.maxnum.f32(float %local.reduce131, float %503), !dbg !20, !frisk.local_reduction !1
  %local.reduce136 = call float @llvm.maxnum.f32(float %local.reduce132, float %508), !dbg !20, !frisk.local_reduction !1
  %local.reduce137 = call float @llvm.maxnum.f32(float %local.reduce133, float %509), !dbg !20, !frisk.local_reduction !1
  %local.reduce138 = call float @llvm.maxnum.f32(float %local.reduce134, float %510), !dbg !20, !frisk.local_reduction !1
  %local.reduce139 = call float @llvm.maxnum.f32(float %local.reduce135, float %511), !dbg !20, !frisk.local_reduction !1
  %local.reduce140 = call float @llvm.maxnum.f32(float %local.reduce136, float %516), !dbg !20, !frisk.local_reduction !1
  %local.reduce141 = call float @llvm.maxnum.f32(float %local.reduce137, float %517), !dbg !20, !frisk.local_reduction !1
  %local.reduce142 = call float @llvm.maxnum.f32(float %local.reduce138, float %518), !dbg !20, !frisk.local_reduction !1
  %local.reduce143 = call float @llvm.maxnum.f32(float %local.reduce139, float %519), !dbg !20, !frisk.local_reduction !1
  %local.reduce144 = call float @llvm.maxnum.f32(float %local.reduce140, float %522), !dbg !20, !frisk.local_reduction !1
  %local.reduce145 = call float @llvm.maxnum.f32(float %local.reduce141, float %523), !dbg !20, !frisk.local_reduction !1
  %local.reduce146 = call float @llvm.maxnum.f32(float %local.reduce142, float %524), !dbg !20, !frisk.local_reduction !1
  %local.reduce147 = call float @llvm.maxnum.f32(float %local.reduce143, float %525), !dbg !20, !frisk.local_reduction !1
  %local.reduce148 = call float @llvm.maxnum.f32(float %local.reduce144, float %local.reduce145), !dbg !20, !frisk.local_reduction !1
  %local.reduce149 = call float @llvm.maxnum.f32(float %local.reduce146, float %local.reduce147), !dbg !20, !frisk.local_reduction !1
  %550 = call float @llvm.maxnum.f32(float %local.reduce148, float %local.reduce149), !dbg !20, !frisk.local_reduction !1
  %551 = bitcast float %550 to i32, !dbg !20
  %552 = call i32 @llvm.amdgcn.ds.bpermute(i32 %533, i32 %551), !dbg !20
  %553 = bitcast i32 %552 to float, !dbg !20
  %554 = call float @llvm.maxnum.f32(float %550, float %553), !dbg !20
  %555 = bitcast float %554 to i32, !dbg !20
  %556 = call i32 @llvm.amdgcn.ds.bpermute(i32 %541, i32 %555), !dbg !20
  %557 = bitcast i32 %556 to float, !dbg !20
  %558 = call float @llvm.maxnum.f32(float %554, float %557), !dbg !20
  %559 = extractvalue [2 x <1 x float>] %125, 1, !dbg !20
  %560 = insertelement <1 x float> poison, float %558, i64 0, !dbg !20
  %561 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %560, <1 x float> %559), !dbg !20
  %562 = shufflevector <1 x float> %548, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !21
  %563 = fsub <4 x float> %400, %562, !dbg !21
  %564 = fsub <4 x float> %409, %562, !dbg !21
  %565 = fsub <4 x float> %417, %562, !dbg !21
  %566 = fsub <4 x float> %425, %562, !dbg !21
  %567 = fsub <4 x float> %433, %562, !dbg !21
  %568 = fsub <4 x float> %441, %562, !dbg !21
  %569 = fsub <4 x float> %449, %562, !dbg !21
  %570 = fsub <4 x float> %457, %562, !dbg !21
  %571 = shufflevector <1 x float> %561, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !21
  %572 = fsub <4 x float> %466, %571, !dbg !21
  %573 = fsub <4 x float> %474, %571, !dbg !21
  %574 = fsub <4 x float> %482, %571, !dbg !21
  %575 = fsub <4 x float> %490, %571, !dbg !21
  %576 = fsub <4 x float> %498, %571, !dbg !21
  %577 = fsub <4 x float> %506, %571, !dbg !21
  %578 = fsub <4 x float> %514, %571, !dbg !21
  %579 = fsub <4 x float> %521, %571, !dbg !21
  %580 = fsub <1 x float> %546, %548, !dbg !22
  %581 = extractelement <1 x float> %580, i64 0, !dbg !23
  %582 = call float @llvm.amdgcn.exp2.f32(float %581), !dbg !23
  %583 = insertelement <4 x float> poison, float %582, i64 0, !dbg !28
  %584 = shufflevector <4 x float> %583, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !28
  %585 = extractvalue [2 x <1 x float>] %124, 0, !dbg !24
  %586 = extractelement <1 x float> %585, i64 0, !dbg !24
  %587 = fmul float %586, %582, !dbg !24
  %588 = fsub <1 x float> %559, %561, !dbg !22
  %589 = extractelement <1 x float> %588, i64 0, !dbg !23
  %590 = call float @llvm.amdgcn.exp2.f32(float %589), !dbg !23
  %591 = insertelement <4 x float> poison, float %590, i64 0, !dbg !28
  %592 = shufflevector <4 x float> %591, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !28
  %593 = extractvalue [2 x <1 x float>] %124, 1, !dbg !24
  %594 = extractelement <1 x float> %593, i64 0, !dbg !24
  %595 = fmul float %594, %590, !dbg !24
  %596 = extractelement <4 x float> %563, i64 0, !dbg !25
  %597 = call float @llvm.amdgcn.exp2.f32(float %596), !dbg !25
  %598 = insertelement <4 x float> poison, float %597, i64 0, !dbg !27
  %599 = extractelement <4 x float> %563, i64 1, !dbg !25
  %600 = call float @llvm.amdgcn.exp2.f32(float %599), !dbg !25
  %601 = insertelement <4 x float> %598, float %600, i64 1, !dbg !27
  %602 = extractelement <4 x float> %563, i64 2, !dbg !25
  %603 = call float @llvm.amdgcn.exp2.f32(float %602), !dbg !25
  %604 = insertelement <4 x float> %601, float %603, i64 2, !dbg !27
  %605 = extractelement <4 x float> %563, i64 3, !dbg !25
  %606 = call float @llvm.amdgcn.exp2.f32(float %605), !dbg !25
  %607 = insertelement <4 x float> %604, float %606, i64 3, !dbg !27
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %607) #9, !dbg !27, !frisk.register_lifetimes !1
  %608 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !27
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %608) #9, !dbg !25, !frisk.register_lifetimes !1
  %609 = extractelement <4 x float> %564, i64 0, !dbg !25
  %610 = call float @llvm.amdgcn.exp2.f32(float %609), !dbg !25
  %611 = insertelement <4 x float> poison, float %610, i64 0, !dbg !27
  %local.reduce89 = fadd float %597, %610, !dbg !26, !frisk.local_reduction !1
  %612 = extractelement <4 x float> %564, i64 1, !dbg !25
  %613 = call float @llvm.amdgcn.exp2.f32(float %612), !dbg !25
  %614 = insertelement <4 x float> %611, float %613, i64 1, !dbg !27
  %local.reduce90 = fadd float %600, %613, !dbg !26, !frisk.local_reduction !1
  %615 = extractelement <4 x float> %564, i64 2, !dbg !25
  %616 = call float @llvm.amdgcn.exp2.f32(float %615), !dbg !25
  %617 = insertelement <4 x float> %614, float %616, i64 2, !dbg !27
  %local.reduce91 = fadd float %603, %616, !dbg !26, !frisk.local_reduction !1
  %618 = extractelement <4 x float> %564, i64 3, !dbg !25
  %619 = call float @llvm.amdgcn.exp2.f32(float %618), !dbg !25
  %620 = insertelement <4 x float> %617, float %619, i64 3, !dbg !27
  %math.result374 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %620) #9, !dbg !27, !frisk.register_lifetimes !1
  %621 = fptrunc <4 x float> %math.result374 to <4 x half>, !dbg !27
  %math.packed375 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %621) #9, !dbg !26, !frisk.register_lifetimes !1
  %local.reduce92 = fadd float %606, %619, !dbg !26, !frisk.local_reduction !1
  %622 = extractelement <4 x float> %565, i64 0, !dbg !25
  %623 = call float @llvm.amdgcn.exp2.f32(float %622), !dbg !25
  %624 = insertelement <4 x float> poison, float %623, i64 0, !dbg !27
  %local.reduce93 = fadd float %local.reduce89, %623, !dbg !26, !frisk.local_reduction !1
  %625 = extractelement <4 x float> %565, i64 1, !dbg !25
  %626 = call float @llvm.amdgcn.exp2.f32(float %625), !dbg !25
  %627 = insertelement <4 x float> %624, float %626, i64 1, !dbg !27
  %local.reduce94 = fadd float %local.reduce90, %626, !dbg !26, !frisk.local_reduction !1
  %628 = extractelement <4 x float> %565, i64 2, !dbg !25
  %629 = call float @llvm.amdgcn.exp2.f32(float %628), !dbg !25
  %630 = insertelement <4 x float> %627, float %629, i64 2, !dbg !27
  %local.reduce95 = fadd float %local.reduce91, %629, !dbg !26, !frisk.local_reduction !1
  %631 = extractelement <4 x float> %565, i64 3, !dbg !25
  %632 = call float @llvm.amdgcn.exp2.f32(float %631), !dbg !25
  %633 = insertelement <4 x float> %630, float %632, i64 3, !dbg !27
  %math.result376 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %633) #9, !dbg !27, !frisk.register_lifetimes !1
  %634 = fptrunc <4 x float> %math.result376 to <4 x half>, !dbg !27
  %math.packed377 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %634) #9, !dbg !26, !frisk.register_lifetimes !1
  %local.reduce96 = fadd float %local.reduce92, %632, !dbg !26, !frisk.local_reduction !1
  %635 = extractelement <4 x float> %566, i64 0, !dbg !25
  %636 = call float @llvm.amdgcn.exp2.f32(float %635), !dbg !25
  %637 = insertelement <4 x float> poison, float %636, i64 0, !dbg !27
  %local.reduce97 = fadd float %local.reduce93, %636, !dbg !26, !frisk.local_reduction !1
  %638 = extractelement <4 x float> %566, i64 1, !dbg !25
  %639 = call float @llvm.amdgcn.exp2.f32(float %638), !dbg !25
  %640 = insertelement <4 x float> %637, float %639, i64 1, !dbg !27
  %local.reduce98 = fadd float %local.reduce94, %639, !dbg !26, !frisk.local_reduction !1
  %641 = extractelement <4 x float> %566, i64 2, !dbg !25
  %642 = call float @llvm.amdgcn.exp2.f32(float %641), !dbg !25
  %643 = insertelement <4 x float> %640, float %642, i64 2, !dbg !27
  %local.reduce99 = fadd float %local.reduce95, %642, !dbg !26, !frisk.local_reduction !1
  %644 = extractelement <4 x float> %566, i64 3, !dbg !25
  %645 = call float @llvm.amdgcn.exp2.f32(float %644), !dbg !25
  %646 = insertelement <4 x float> %643, float %645, i64 3, !dbg !27
  %math.result378 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %646) #9, !dbg !27, !frisk.register_lifetimes !1
  %647 = fptrunc <4 x float> %math.result378 to <4 x half>, !dbg !27
  %math.packed379 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %647) #9, !dbg !26, !frisk.register_lifetimes !1
  %local.reduce100 = fadd float %local.reduce96, %645, !dbg !26, !frisk.local_reduction !1
  %648 = extractelement <4 x float> %567, i64 0, !dbg !25
  %649 = call float @llvm.amdgcn.exp2.f32(float %648), !dbg !25
  %650 = insertelement <4 x float> poison, float %649, i64 0, !dbg !27
  %local.reduce101 = fadd float %local.reduce97, %649, !dbg !26, !frisk.local_reduction !1
  %651 = extractelement <4 x float> %567, i64 1, !dbg !25
  %652 = call float @llvm.amdgcn.exp2.f32(float %651), !dbg !25
  %653 = insertelement <4 x float> %650, float %652, i64 1, !dbg !27
  %local.reduce102 = fadd float %local.reduce98, %652, !dbg !26, !frisk.local_reduction !1
  %654 = extractelement <4 x float> %567, i64 2, !dbg !25
  %655 = call float @llvm.amdgcn.exp2.f32(float %654), !dbg !25
  %656 = insertelement <4 x float> %653, float %655, i64 2, !dbg !27
  %local.reduce103 = fadd float %local.reduce99, %655, !dbg !26, !frisk.local_reduction !1
  %657 = extractelement <4 x float> %567, i64 3, !dbg !25
  %658 = call float @llvm.amdgcn.exp2.f32(float %657), !dbg !25
  %659 = insertelement <4 x float> %656, float %658, i64 3, !dbg !27
  %math.result380 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %659) #9, !dbg !27, !frisk.register_lifetimes !1
  %660 = fptrunc <4 x float> %math.result380 to <4 x half>, !dbg !27
  %math.packed381 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %660) #9, !dbg !26, !frisk.register_lifetimes !1
  %local.reduce104 = fadd float %local.reduce100, %658, !dbg !26, !frisk.local_reduction !1
  %661 = extractelement <4 x float> %568, i64 0, !dbg !25
  %662 = call float @llvm.amdgcn.exp2.f32(float %661), !dbg !25
  %663 = insertelement <4 x float> poison, float %662, i64 0, !dbg !27
  %local.reduce105 = fadd float %local.reduce101, %662, !dbg !26, !frisk.local_reduction !1
  %664 = extractelement <4 x float> %568, i64 1, !dbg !25
  %665 = call float @llvm.amdgcn.exp2.f32(float %664), !dbg !25
  %666 = insertelement <4 x float> %663, float %665, i64 1, !dbg !27
  %local.reduce106 = fadd float %local.reduce102, %665, !dbg !26, !frisk.local_reduction !1
  %667 = extractelement <4 x float> %568, i64 2, !dbg !25
  %668 = call float @llvm.amdgcn.exp2.f32(float %667), !dbg !25
  %669 = insertelement <4 x float> %666, float %668, i64 2, !dbg !27
  %local.reduce107 = fadd float %local.reduce103, %668, !dbg !26, !frisk.local_reduction !1
  %670 = extractelement <4 x float> %568, i64 3, !dbg !25
  %671 = call float @llvm.amdgcn.exp2.f32(float %670), !dbg !25
  %672 = insertelement <4 x float> %669, float %671, i64 3, !dbg !27
  %math.result382 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %672) #9, !dbg !27, !frisk.register_lifetimes !1
  %673 = fptrunc <4 x float> %math.result382 to <4 x half>, !dbg !27
  %math.packed383 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %673) #9, !dbg !26, !frisk.register_lifetimes !1
  %local.reduce108 = fadd float %local.reduce104, %671, !dbg !26, !frisk.local_reduction !1
  %674 = extractelement <4 x float> %569, i64 0, !dbg !25
  %675 = call float @llvm.amdgcn.exp2.f32(float %674), !dbg !25
  %676 = insertelement <4 x float> poison, float %675, i64 0, !dbg !27
  %local.reduce109 = fadd float %local.reduce105, %675, !dbg !26, !frisk.local_reduction !1
  %677 = extractelement <4 x float> %569, i64 1, !dbg !25
  %678 = call float @llvm.amdgcn.exp2.f32(float %677), !dbg !25
  %679 = insertelement <4 x float> %676, float %678, i64 1, !dbg !27
  %local.reduce110 = fadd float %local.reduce106, %678, !dbg !26, !frisk.local_reduction !1
  %680 = extractelement <4 x float> %569, i64 2, !dbg !25
  %681 = call float @llvm.amdgcn.exp2.f32(float %680), !dbg !25
  %682 = insertelement <4 x float> %679, float %681, i64 2, !dbg !27
  %local.reduce111 = fadd float %local.reduce107, %681, !dbg !26, !frisk.local_reduction !1
  %683 = extractelement <4 x float> %569, i64 3, !dbg !25
  %684 = call float @llvm.amdgcn.exp2.f32(float %683), !dbg !25
  %685 = insertelement <4 x float> %682, float %684, i64 3, !dbg !27
  %math.result384 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %685) #9, !dbg !27, !frisk.register_lifetimes !1
  %686 = fptrunc <4 x float> %math.result384 to <4 x half>, !dbg !27
  %math.packed385 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %686) #9, !dbg !26, !frisk.register_lifetimes !1
  %local.reduce112 = fadd float %local.reduce108, %684, !dbg !26, !frisk.local_reduction !1
  %687 = extractelement <4 x float> %570, i64 0, !dbg !25
  %688 = call float @llvm.amdgcn.exp2.f32(float %687), !dbg !25
  %689 = insertelement <4 x float> poison, float %688, i64 0, !dbg !27
  %local.reduce113 = fadd float %local.reduce109, %688, !dbg !26, !frisk.local_reduction !1
  %690 = extractelement <4 x float> %570, i64 1, !dbg !25
  %691 = call float @llvm.amdgcn.exp2.f32(float %690), !dbg !25
  %692 = insertelement <4 x float> %689, float %691, i64 1, !dbg !27
  %local.reduce114 = fadd float %local.reduce110, %691, !dbg !26, !frisk.local_reduction !1
  %local.reduce117 = fadd float %local.reduce113, %local.reduce114, !dbg !26, !frisk.local_reduction !1
  %693 = extractelement <4 x float> %570, i64 2, !dbg !25
  %694 = call float @llvm.amdgcn.exp2.f32(float %693), !dbg !25
  %695 = insertelement <4 x float> %692, float %694, i64 2, !dbg !27
  %local.reduce115 = fadd float %local.reduce111, %694, !dbg !26, !frisk.local_reduction !1
  %696 = extractelement <4 x float> %570, i64 3, !dbg !25
  %697 = call float @llvm.amdgcn.exp2.f32(float %696), !dbg !25
  %698 = insertelement <4 x float> %695, float %697, i64 3, !dbg !27
  %math.result386 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %698) #9, !dbg !27, !frisk.register_lifetimes !1
  %699 = fptrunc <4 x float> %math.result386 to <4 x half>, !dbg !27
  %math.packed387 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %699) #9, !dbg !26, !frisk.register_lifetimes !1
  %local.reduce116 = fadd float %local.reduce112, %697, !dbg !26, !frisk.local_reduction !1
  %local.reduce118 = fadd float %local.reduce115, %local.reduce116, !dbg !26, !frisk.local_reduction !1
  %700 = fadd float %local.reduce117, %local.reduce118, !dbg !26, !frisk.local_reduction !1
  %701 = fadd float %700, %587, !dbg !26
  %702 = insertelement <1 x float> poison, float %701, i64 0, !dbg !7
  %703 = insertvalue [2 x <1 x float>] poison, <1 x float> %702, 0, !dbg !7
  %704 = extractelement <4 x float> %572, i64 0, !dbg !25
  %705 = call float @llvm.amdgcn.exp2.f32(float %704), !dbg !25
  %706 = insertelement <4 x float> poison, float %705, i64 0, !dbg !27
  %707 = extractelement <4 x float> %572, i64 1, !dbg !25
  %708 = call float @llvm.amdgcn.exp2.f32(float %707), !dbg !25
  %709 = insertelement <4 x float> %706, float %708, i64 1, !dbg !27
  %710 = extractelement <4 x float> %572, i64 2, !dbg !25
  %711 = call float @llvm.amdgcn.exp2.f32(float %710), !dbg !25
  %712 = insertelement <4 x float> %709, float %711, i64 2, !dbg !27
  %713 = extractelement <4 x float> %572, i64 3, !dbg !25
  %714 = call float @llvm.amdgcn.exp2.f32(float %713), !dbg !25
  %715 = insertelement <4 x float> %712, float %714, i64 3, !dbg !27
  %math.result388 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %715) #9, !dbg !27, !frisk.register_lifetimes !1
  %716 = fptrunc <4 x float> %math.result388 to <4 x half>, !dbg !27
  %math.packed389 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %716) #9, !dbg !25, !frisk.register_lifetimes !1
  %717 = extractelement <4 x float> %573, i64 0, !dbg !25
  %718 = call float @llvm.amdgcn.exp2.f32(float %717), !dbg !25
  %719 = insertelement <4 x float> poison, float %718, i64 0, !dbg !27
  %local.reduce = fadd float %705, %718, !dbg !26, !frisk.local_reduction !1
  %720 = extractelement <4 x float> %573, i64 1, !dbg !25
  %721 = call float @llvm.amdgcn.exp2.f32(float %720), !dbg !25
  %722 = insertelement <4 x float> %719, float %721, i64 1, !dbg !27
  %local.reduce59 = fadd float %708, %721, !dbg !26, !frisk.local_reduction !1
  %723 = extractelement <4 x float> %573, i64 2, !dbg !25
  %724 = call float @llvm.amdgcn.exp2.f32(float %723), !dbg !25
  %725 = insertelement <4 x float> %722, float %724, i64 2, !dbg !27
  %local.reduce60 = fadd float %711, %724, !dbg !26, !frisk.local_reduction !1
  %726 = extractelement <4 x float> %573, i64 3, !dbg !25
  %727 = call float @llvm.amdgcn.exp2.f32(float %726), !dbg !25
  %728 = insertelement <4 x float> %725, float %727, i64 3, !dbg !27
  %math.result390 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %728) #9, !dbg !27, !frisk.register_lifetimes !1
  %729 = fptrunc <4 x float> %math.result390 to <4 x half>, !dbg !27
  %math.packed391 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %729) #9, !dbg !26, !frisk.register_lifetimes !1
  %local.reduce61 = fadd float %714, %727, !dbg !26, !frisk.local_reduction !1
  %730 = extractelement <4 x float> %574, i64 0, !dbg !25
  %731 = call float @llvm.amdgcn.exp2.f32(float %730), !dbg !25
  %732 = insertelement <4 x float> poison, float %731, i64 0, !dbg !27
  %local.reduce62 = fadd float %local.reduce, %731, !dbg !26, !frisk.local_reduction !1
  %733 = extractelement <4 x float> %574, i64 1, !dbg !25
  %734 = call float @llvm.amdgcn.exp2.f32(float %733), !dbg !25
  %735 = insertelement <4 x float> %732, float %734, i64 1, !dbg !27
  %local.reduce63 = fadd float %local.reduce59, %734, !dbg !26, !frisk.local_reduction !1
  %736 = extractelement <4 x float> %574, i64 2, !dbg !25
  %737 = call float @llvm.amdgcn.exp2.f32(float %736), !dbg !25
  %738 = insertelement <4 x float> %735, float %737, i64 2, !dbg !27
  %local.reduce64 = fadd float %local.reduce60, %737, !dbg !26, !frisk.local_reduction !1
  %739 = extractelement <4 x float> %574, i64 3, !dbg !25
  %740 = call float @llvm.amdgcn.exp2.f32(float %739), !dbg !25
  %741 = insertelement <4 x float> %738, float %740, i64 3, !dbg !27
  %math.result392 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %741) #9, !dbg !27, !frisk.register_lifetimes !1
  %742 = fptrunc <4 x float> %math.result392 to <4 x half>, !dbg !27
  %math.packed393 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %742) #9, !dbg !26, !frisk.register_lifetimes !1
  %local.reduce65 = fadd float %local.reduce61, %740, !dbg !26, !frisk.local_reduction !1
  %743 = extractelement <4 x float> %575, i64 0, !dbg !25
  %744 = call float @llvm.amdgcn.exp2.f32(float %743), !dbg !25
  %745 = insertelement <4 x float> poison, float %744, i64 0, !dbg !27
  %local.reduce66 = fadd float %local.reduce62, %744, !dbg !26, !frisk.local_reduction !1
  %746 = extractelement <4 x float> %575, i64 1, !dbg !25
  %747 = call float @llvm.amdgcn.exp2.f32(float %746), !dbg !25
  %748 = insertelement <4 x float> %745, float %747, i64 1, !dbg !27
  %local.reduce67 = fadd float %local.reduce63, %747, !dbg !26, !frisk.local_reduction !1
  %749 = extractelement <4 x float> %575, i64 2, !dbg !25
  %750 = call float @llvm.amdgcn.exp2.f32(float %749), !dbg !25
  %751 = insertelement <4 x float> %748, float %750, i64 2, !dbg !27
  %local.reduce68 = fadd float %local.reduce64, %750, !dbg !26, !frisk.local_reduction !1
  %752 = extractelement <4 x float> %575, i64 3, !dbg !25
  %753 = call float @llvm.amdgcn.exp2.f32(float %752), !dbg !25
  %754 = insertelement <4 x float> %751, float %753, i64 3, !dbg !27
  %math.result394 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %754) #9, !dbg !27, !frisk.register_lifetimes !1
  %755 = fptrunc <4 x float> %math.result394 to <4 x half>, !dbg !27
  %math.packed395 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %755) #9, !dbg !26, !frisk.register_lifetimes !1
  %local.reduce69 = fadd float %local.reduce65, %753, !dbg !26, !frisk.local_reduction !1
  %756 = extractelement <4 x float> %576, i64 0, !dbg !25
  %757 = call float @llvm.amdgcn.exp2.f32(float %756), !dbg !25
  %758 = insertelement <4 x float> poison, float %757, i64 0, !dbg !27
  %local.reduce70 = fadd float %local.reduce66, %757, !dbg !26, !frisk.local_reduction !1
  %759 = extractelement <4 x float> %576, i64 1, !dbg !25
  %760 = call float @llvm.amdgcn.exp2.f32(float %759), !dbg !25
  %761 = insertelement <4 x float> %758, float %760, i64 1, !dbg !27
  %local.reduce71 = fadd float %local.reduce67, %760, !dbg !26, !frisk.local_reduction !1
  %762 = extractelement <4 x float> %576, i64 2, !dbg !25
  %763 = call float @llvm.amdgcn.exp2.f32(float %762), !dbg !25
  %764 = insertelement <4 x float> %761, float %763, i64 2, !dbg !27
  %local.reduce72 = fadd float %local.reduce68, %763, !dbg !26, !frisk.local_reduction !1
  %765 = extractelement <4 x float> %576, i64 3, !dbg !25
  %766 = call float @llvm.amdgcn.exp2.f32(float %765), !dbg !25
  %767 = insertelement <4 x float> %764, float %766, i64 3, !dbg !27
  %math.result396 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %767) #9, !dbg !27, !frisk.register_lifetimes !1
  %768 = fptrunc <4 x float> %math.result396 to <4 x half>, !dbg !27
  %math.packed397 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %768) #9, !dbg !26, !frisk.register_lifetimes !1
  %local.reduce73 = fadd float %local.reduce69, %766, !dbg !26, !frisk.local_reduction !1
  %769 = extractelement <4 x float> %577, i64 0, !dbg !25
  %770 = call float @llvm.amdgcn.exp2.f32(float %769), !dbg !25
  %771 = insertelement <4 x float> poison, float %770, i64 0, !dbg !27
  %local.reduce74 = fadd float %local.reduce70, %770, !dbg !26, !frisk.local_reduction !1
  %772 = extractelement <4 x float> %577, i64 1, !dbg !25
  %773 = call float @llvm.amdgcn.exp2.f32(float %772), !dbg !25
  %774 = insertelement <4 x float> %771, float %773, i64 1, !dbg !27
  %local.reduce75 = fadd float %local.reduce71, %773, !dbg !26, !frisk.local_reduction !1
  %775 = extractelement <4 x float> %577, i64 2, !dbg !25
  %776 = call float @llvm.amdgcn.exp2.f32(float %775), !dbg !25
  %777 = insertelement <4 x float> %774, float %776, i64 2, !dbg !27
  %local.reduce76 = fadd float %local.reduce72, %776, !dbg !26, !frisk.local_reduction !1
  %778 = extractelement <4 x float> %577, i64 3, !dbg !25
  %779 = call float @llvm.amdgcn.exp2.f32(float %778), !dbg !25
  %780 = insertelement <4 x float> %777, float %779, i64 3, !dbg !27
  %math.result398 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %780) #9, !dbg !27, !frisk.register_lifetimes !1
  %781 = fptrunc <4 x float> %math.result398 to <4 x half>, !dbg !27
  %math.packed399 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %781) #9, !dbg !26, !frisk.register_lifetimes !1
  %local.reduce77 = fadd float %local.reduce73, %779, !dbg !26, !frisk.local_reduction !1
  %782 = extractelement <4 x float> %578, i64 0, !dbg !25
  %783 = call float @llvm.amdgcn.exp2.f32(float %782), !dbg !25
  %784 = insertelement <4 x float> poison, float %783, i64 0, !dbg !27
  %local.reduce78 = fadd float %local.reduce74, %783, !dbg !26, !frisk.local_reduction !1
  %785 = extractelement <4 x float> %578, i64 1, !dbg !25
  %786 = call float @llvm.amdgcn.exp2.f32(float %785), !dbg !25
  %787 = insertelement <4 x float> %784, float %786, i64 1, !dbg !27
  %local.reduce79 = fadd float %local.reduce75, %786, !dbg !26, !frisk.local_reduction !1
  %788 = extractelement <4 x float> %578, i64 2, !dbg !25
  %789 = call float @llvm.amdgcn.exp2.f32(float %788), !dbg !25
  %790 = insertelement <4 x float> %787, float %789, i64 2, !dbg !27
  %local.reduce80 = fadd float %local.reduce76, %789, !dbg !26, !frisk.local_reduction !1
  %791 = extractelement <4 x float> %578, i64 3, !dbg !25
  %792 = call float @llvm.amdgcn.exp2.f32(float %791), !dbg !25
  %793 = insertelement <4 x float> %790, float %792, i64 3, !dbg !27
  %math.result400 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %793) #9, !dbg !27, !frisk.register_lifetimes !1
  %794 = fptrunc <4 x float> %math.result400 to <4 x half>, !dbg !27
  %math.packed401 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %794) #9, !dbg !26, !frisk.register_lifetimes !1
  %local.reduce81 = fadd float %local.reduce77, %792, !dbg !26, !frisk.local_reduction !1
  %795 = extractelement <4 x float> %579, i64 0, !dbg !25
  %796 = call float @llvm.amdgcn.exp2.f32(float %795), !dbg !25
  %797 = insertelement <4 x float> poison, float %796, i64 0, !dbg !27
  %local.reduce82 = fadd float %local.reduce78, %796, !dbg !26, !frisk.local_reduction !1
  %798 = extractelement <4 x float> %579, i64 1, !dbg !25
  %799 = call float @llvm.amdgcn.exp2.f32(float %798), !dbg !25
  %800 = insertelement <4 x float> %797, float %799, i64 1, !dbg !27
  %local.reduce83 = fadd float %local.reduce79, %799, !dbg !26, !frisk.local_reduction !1
  %local.reduce86 = fadd float %local.reduce82, %local.reduce83, !dbg !26, !frisk.local_reduction !1
  %801 = extractelement <4 x float> %579, i64 2, !dbg !25
  %802 = call float @llvm.amdgcn.exp2.f32(float %801), !dbg !25
  %803 = insertelement <4 x float> %800, float %802, i64 2, !dbg !27
  %local.reduce84 = fadd float %local.reduce80, %802, !dbg !26, !frisk.local_reduction !1
  %804 = extractelement <4 x float> %579, i64 3, !dbg !25
  %805 = call float @llvm.amdgcn.exp2.f32(float %804), !dbg !25
  %806 = insertelement <4 x float> %803, float %805, i64 3, !dbg !27
  %math.result402 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %806) #9, !dbg !27, !frisk.register_lifetimes !1
  %807 = fptrunc <4 x float> %math.result402 to <4 x half>, !dbg !27
  %math.packed403 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %807) #9, !dbg !26, !frisk.register_lifetimes !1
  %local.reduce85 = fadd float %local.reduce81, %805, !dbg !26, !frisk.local_reduction !1
  %local.reduce87 = fadd float %local.reduce84, %local.reduce85, !dbg !26, !frisk.local_reduction !1
  %808 = fadd float %local.reduce86, %local.reduce87, !dbg !26, !frisk.local_reduction !1
  %809 = fadd float %808, %595, !dbg !26
  %810 = insertelement <1 x float> poison, float %809, i64 0, !dbg !7
  %811 = insertvalue [2 x <1 x float>] %703, <1 x float> %810, 1, !dbg !7
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
  %812 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1438, <4 x i32> %vdb.address1442) #10, !dbg !15
  %vdb.dma.base1452 = add i32 %vdb.address1444, 8192, !dbg !13
  %813 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1452, i32 %vdb.address1447, <4 x i32> %vdb.address1451) #10, !dbg !15
  %address.local929 = shl nuw nsw i32 %address.thread498, 3, !dbg !13
  %address.local930 = and i32 %address.local929, 120, !dbg !13
  %address.local931 = shl nuw nsw i32 %address.thread498, 5, !dbg !13
  %address.local932 = and i32 %address.local931, 1536, !dbg !13
  %address.local933 = or disjoint i32 %address.local930, %address.local932, !dbg !13
  %address.local934 = zext nneg i32 %address.local933 to i64, !dbg !13
  %address.local935 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local934, !dbg !13
  %814 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local935, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %815 = bitcast <4 x i32> %814 to <8 x half>, !dbg !13
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
  %816 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local945, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %817 = bitcast <4 x i32> %816 to <8 x half>, !dbg !13
  %address.local946 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local938, !dbg !13
  %address.local947 = getelementptr [2 x i8], ptr addrspace(3) %address.local946, i64 %address.local941, !dbg !13
  %address.local948 = getelementptr [2 x i8], ptr addrspace(3) %address.local947, i64 %address.local943, !dbg !13
  %address.local949 = getelementptr i8, ptr addrspace(3) %address.local948, i64 512, !dbg !13
  %818 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local949, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %819 = bitcast <4 x i32> %818 to <8 x half>, !dbg !13
  %address.local950 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local938, !dbg !13
  %address.local951 = getelementptr [2 x i8], ptr addrspace(3) %address.local950, i64 %address.local941, !dbg !13
  %address.local952 = getelementptr [2 x i8], ptr addrspace(3) %address.local951, i64 %address.local943, !dbg !13
  %address.local953 = getelementptr i8, ptr addrspace(3) %address.local952, i64 768, !dbg !13
  %820 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local953, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %821 = bitcast <4 x i32> %820 to <8 x half>, !dbg !13
  %address.local954 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local934, !dbg !13
  %address.local955 = getelementptr i8, ptr addrspace(3) %address.local954, i64 4096, !dbg !13
  %822 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local955, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %823 = bitcast <4 x i32> %822 to <8 x half>, !dbg !13
  %address.local956 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local938, !dbg !13
  %address.local957 = getelementptr [2 x i8], ptr addrspace(3) %address.local956, i64 %address.local941, !dbg !13
  %address.local958 = getelementptr [2 x i8], ptr addrspace(3) %address.local957, i64 %address.local943, !dbg !13
  %address.local959 = getelementptr i8, ptr addrspace(3) %address.local958, i64 4352, !dbg !13
  %824 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local959, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %825 = bitcast <4 x i32> %824 to <8 x half>, !dbg !13
  %address.local960 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local938, !dbg !13
  %address.local961 = getelementptr [2 x i8], ptr addrspace(3) %address.local960, i64 %address.local941, !dbg !13
  %address.local962 = getelementptr [2 x i8], ptr addrspace(3) %address.local961, i64 %address.local943, !dbg !13
  %address.local963 = getelementptr i8, ptr addrspace(3) %address.local962, i64 4608, !dbg !13
  %826 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local963, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %827 = bitcast <4 x i32> %826 to <8 x half>, !dbg !13
  %address.local964 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local938, !dbg !13
  %address.local965 = getelementptr [2 x i8], ptr addrspace(3) %address.local964, i64 %address.local941, !dbg !13
  %address.local966 = getelementptr [2 x i8], ptr addrspace(3) %address.local965, i64 %address.local943, !dbg !13
  %address.local967 = getelementptr i8, ptr addrspace(3) %address.local966, i64 4864, !dbg !13
  %828 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local967, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %829 = shufflevector <8 x half> %815, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %830 = extractvalue [2 x <8 x float>] %123, 0, !dbg !28
  %831 = shufflevector <8 x float> %830, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %832 = fmul <4 x float> %831, %584, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %833 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %829, <4 x float> %832) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %834 = shufflevector <8 x half> %815, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %835 = shufflevector <8 x float> %830, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %836 = fmul <4 x float> %835, %584, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %837 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %834, <4 x float> %836) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %838 = extractvalue [2 x <8 x float>] %123, 1, !dbg !28
  %839 = shufflevector <8 x float> %838, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %840 = fmul <4 x float> %839, %592, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %841 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %829, <4 x float> %840) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %842 = shufflevector <8 x float> %838, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %843 = fmul <4 x float> %842, %592, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %844 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %834, <4 x float> %843) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %845 = shufflevector <8 x half> %817, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %846 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %845, <4 x float> %833) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %847 = shufflevector <8 x half> %817, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %848 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %847, <4 x float> %837) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %849 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %845, <4 x float> %841) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %850 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %847, <4 x float> %844) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %851 = shufflevector <8 x half> %819, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %852 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %851, <4 x float> %846) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %853 = shufflevector <8 x half> %819, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %854 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %853, <4 x float> %848) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %855 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %851, <4 x float> %849) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %856 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %853, <4 x float> %850) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %857 = shufflevector <8 x half> %821, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %858 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %857, <4 x float> %852) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %859 = shufflevector <8 x half> %821, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %860 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %859, <4 x float> %854) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %861 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %857, <4 x float> %855) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %862 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %859, <4 x float> %856) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %863 = shufflevector <8 x half> %823, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %864 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %863, <4 x float> %858) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %865 = shufflevector <8 x half> %823, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %866 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %865, <4 x float> %860) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %867 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %863, <4 x float> %861) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %868 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %865, <4 x float> %862) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %869 = shufflevector <8 x half> %825, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %870 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %869, <4 x float> %864) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %871 = shufflevector <8 x half> %825, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %872 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %871, <4 x float> %866) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %873 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %869, <4 x float> %867) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %874 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %871, <4 x float> %868) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %875 = shufflevector <8 x half> %827, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %876 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %875, <4 x float> %870) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %877 = shufflevector <8 x half> %827, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %878 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %877, <4 x float> %872) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %879 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %875, <4 x float> %873) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %880 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %877, <4 x float> %874) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %881 = bitcast <4 x i32> %828 to <8 x half>, !dbg !13
  %882 = shufflevector <8 x half> %881, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %883 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %882, <4 x float> %876) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %884 = shufflevector <8 x half> %881, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %885 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %884, <4 x float> %878) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %886 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %882, <4 x float> %879) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %887 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %884, <4 x float> %880) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %vdb.address1460 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local504), !dbg !15
  %vdb.address1461 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %vdb.address1460), !dbg !15
  %vdb.address1462 = or disjoint i64 %232, %address.local921, !dbg !15
  %vdb.address1463 = trunc nuw i64 %vdb.address1462 to i32, !dbg !15
  %vdb.address1464 = shl nuw nsw i32 %vdb.address1463, 1, !dbg !15
  %vdb.address1465 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %vdb.address1466 = insertelement <2 x i64> poison, i64 %vdb.address1465, i64 0, !dbg !15
  %vdb.address1467 = bitcast <2 x i64> %vdb.address1466 to <4 x i32>, !dbg !15
  %vdb.address1468 = shufflevector <4 x i32> %vdb.address1467, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %vdb.address1469 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local521), !dbg !15
  %vdb.address1470 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %vdb.address1469), !dbg !15
  %vdb.address1471 = or disjoint i64 %232, %address.local927, !dbg !15
  %vdb.address1472 = trunc nuw i64 %vdb.address1471 to i32, !dbg !15
  %vdb.address1473 = shl nuw nsw i32 %vdb.address1472, 1, !dbg !15
  %vdb.address1474 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %vdb.address1475 = insertelement <2 x i64> poison, i64 %vdb.address1474, i64 0, !dbg !15
  %vdb.address1476 = bitcast <2 x i64> %vdb.address1475 to <4 x i32>, !dbg !15
  %vdb.address1477 = shufflevector <4 x i32> %vdb.address1476, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %888 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.address1461, i32 %vdb.address1464, <4 x i32> %vdb.address1468) #10, !dbg !15
  %889 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.address1470, i32 %vdb.address1473, <4 x i32> %vdb.address1477) #10, !dbg !15
  %vdb.read = getelementptr i8, ptr addrspace(3) %address.local935, i64 8192, !dbg !13
  %890 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %891 = bitcast <4 x i32> %890 to <8 x half>, !dbg !13
  %vdb.read1453 = getelementptr i8, ptr addrspace(3) %address.local945, i64 8192, !dbg !13
  %892 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1453, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %893 = bitcast <4 x i32> %892 to <8 x half>, !dbg !13
  %vdb.read1454 = getelementptr i8, ptr addrspace(3) %address.local949, i64 8192, !dbg !13
  %894 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1454, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %895 = bitcast <4 x i32> %894 to <8 x half>, !dbg !13
  %vdb.read1455 = getelementptr i8, ptr addrspace(3) %address.local953, i64 8192, !dbg !13
  %896 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1455, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %897 = bitcast <4 x i32> %896 to <8 x half>, !dbg !13
  %vdb.read1456 = getelementptr i8, ptr addrspace(3) %address.local955, i64 8192, !dbg !13
  %898 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1456, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %899 = bitcast <4 x i32> %898 to <8 x half>, !dbg !13
  %vdb.read1457 = getelementptr i8, ptr addrspace(3) %address.local959, i64 8192, !dbg !13
  %900 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1457, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %901 = bitcast <4 x i32> %900 to <8 x half>, !dbg !13
  %vdb.read1458 = getelementptr i8, ptr addrspace(3) %address.local963, i64 8192, !dbg !13
  %902 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1458, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %903 = bitcast <4 x i32> %902 to <8 x half>, !dbg !13
  %vdb.read1459 = getelementptr i8, ptr addrspace(3) %address.local967, i64 8192, !dbg !13
  %904 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1459, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %905 = shufflevector <8 x half> %891, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %906 = extractvalue [2 x <8 x float>] %126, 0, !dbg !28
  %907 = shufflevector <8 x float> %906, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %908 = fmul <4 x float> %907, %584, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %909 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %905, <4 x float> %908) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %910 = shufflevector <8 x half> %891, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %911 = shufflevector <8 x float> %906, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %912 = fmul <4 x float> %911, %584, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %913 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %910, <4 x float> %912) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %914 = extractvalue [2 x <8 x float>] %126, 1, !dbg !28
  %915 = shufflevector <8 x float> %914, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %916 = fmul <4 x float> %915, %592, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %917 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %905, <4 x float> %916) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %918 = shufflevector <8 x float> %914, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %919 = fmul <4 x float> %918, %592, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %920 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %910, <4 x float> %919) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %921 = shufflevector <8 x half> %893, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %922 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %921, <4 x float> %909) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %923 = shufflevector <8 x half> %893, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %924 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %923, <4 x float> %913) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %925 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %921, <4 x float> %917) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %926 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %923, <4 x float> %920) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %927 = shufflevector <8 x half> %895, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %928 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %927, <4 x float> %922) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %929 = shufflevector <8 x half> %895, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %930 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %929, <4 x float> %924) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %931 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %927, <4 x float> %925) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %932 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %929, <4 x float> %926) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %933 = shufflevector <8 x half> %897, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %934 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %933, <4 x float> %928) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %935 = shufflevector <8 x half> %897, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %936 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %935, <4 x float> %930) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %937 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %933, <4 x float> %931) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %938 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %935, <4 x float> %932) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %939 = shufflevector <8 x half> %899, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %940 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %939, <4 x float> %934) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %941 = shufflevector <8 x half> %899, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %942 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %941, <4 x float> %936) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %943 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %939, <4 x float> %937) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %944 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %941, <4 x float> %938) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %945 = shufflevector <8 x half> %901, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %946 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %945, <4 x float> %940) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %947 = shufflevector <8 x half> %901, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %948 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %947, <4 x float> %942) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %949 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %945, <4 x float> %943) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %950 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %947, <4 x float> %944) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %951 = shufflevector <8 x half> %903, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %952 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %951, <4 x float> %946) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %953 = shufflevector <8 x half> %903, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %954 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %953, <4 x float> %948) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %955 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %951, <4 x float> %949) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %956 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %953, <4 x float> %950) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %957 = bitcast <4 x i32> %904 to <8 x half>, !dbg !13
  %958 = shufflevector <8 x half> %957, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %959 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %958, <4 x float> %952) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %960 = shufflevector <8 x half> %957, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %961 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %960, <4 x float> %954) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %962 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %958, <4 x float> %955) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %963 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %960, <4 x float> %956) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %vdb.address1478 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local504), !dbg !15
  %vdb.address1479 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %vdb.address1478), !dbg !15
  %vdb.address1480 = or disjoint i64 %309, %address.local921, !dbg !15
  %vdb.address1481 = trunc nuw i64 %vdb.address1480 to i32, !dbg !15
  %vdb.address1482 = shl nuw nsw i32 %vdb.address1481, 1, !dbg !15
  %vdb.address1483 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %vdb.address1484 = insertelement <2 x i64> poison, i64 %vdb.address1483, i64 0, !dbg !15
  %vdb.address1485 = bitcast <2 x i64> %vdb.address1484 to <4 x i32>, !dbg !15
  %vdb.address1486 = shufflevector <4 x i32> %vdb.address1485, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %vdb.address1487 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local521), !dbg !15
  %vdb.address1488 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %vdb.address1487), !dbg !15
  %vdb.address1489 = or disjoint i64 %309, %address.local927, !dbg !15
  %vdb.address1490 = trunc nuw i64 %vdb.address1489 to i32, !dbg !15
  %vdb.address1491 = shl nuw nsw i32 %vdb.address1490, 1, !dbg !15
  %vdb.address1492 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %vdb.address1493 = insertelement <2 x i64> poison, i64 %vdb.address1492, i64 0, !dbg !15
  %vdb.address1494 = bitcast <2 x i64> %vdb.address1493 to <4 x i32>, !dbg !15
  %vdb.address1495 = shufflevector <4 x i32> %vdb.address1494, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %vdb.dma.base1496 = add i32 %vdb.address1479, 8192, !dbg !13
  %964 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1496, i32 %vdb.address1482, <4 x i32> %vdb.address1486) #10, !dbg !15
  %vdb.dma.base1497 = add i32 %vdb.address1488, 8192, !dbg !13
  %965 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1497, i32 %vdb.address1491, <4 x i32> %vdb.address1495) #10, !dbg !15
  %966 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local935, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %967 = bitcast <4 x i32> %966 to <8 x half>, !dbg !13
  %968 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local945, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %969 = bitcast <4 x i32> %968 to <8 x half>, !dbg !13
  %970 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local949, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %971 = bitcast <4 x i32> %970 to <8 x half>, !dbg !13
  %972 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local953, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %973 = bitcast <4 x i32> %972 to <8 x half>, !dbg !13
  %974 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local955, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %975 = bitcast <4 x i32> %974 to <8 x half>, !dbg !13
  %976 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local959, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %977 = bitcast <4 x i32> %976 to <8 x half>, !dbg !13
  %978 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local963, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %979 = bitcast <4 x i32> %978 to <8 x half>, !dbg !13
  %980 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local967, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %981 = shufflevector <8 x half> %967, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %982 = extractvalue [2 x <8 x float>] %127, 0, !dbg !28
  %983 = shufflevector <8 x float> %982, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %984 = fmul <4 x float> %983, %584, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %985 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %981, <4 x float> %984) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %986 = shufflevector <8 x half> %967, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %987 = shufflevector <8 x float> %982, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %988 = fmul <4 x float> %987, %584, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %989 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %986, <4 x float> %988) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %990 = extractvalue [2 x <8 x float>] %127, 1, !dbg !28
  %991 = shufflevector <8 x float> %990, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %992 = fmul <4 x float> %991, %592, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %993 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %981, <4 x float> %992) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %994 = shufflevector <8 x float> %990, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %995 = fmul <4 x float> %994, %592, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %996 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %986, <4 x float> %995) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %997 = shufflevector <8 x half> %969, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %998 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %997, <4 x float> %985) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %999 = shufflevector <8 x half> %969, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1000 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %999, <4 x float> %989) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1001 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %997, <4 x float> %993) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1002 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %999, <4 x float> %996) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1003 = shufflevector <8 x half> %971, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1004 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %1003, <4 x float> %998) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1005 = shufflevector <8 x half> %971, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1006 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %1005, <4 x float> %1000) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1007 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %1003, <4 x float> %1001) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1008 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %1005, <4 x float> %1002) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1009 = shufflevector <8 x half> %973, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1010 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %1009, <4 x float> %1004) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1011 = shufflevector <8 x half> %973, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1012 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %1011, <4 x float> %1006) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1013 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %1009, <4 x float> %1007) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1014 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %1011, <4 x float> %1008) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1015 = shufflevector <8 x half> %975, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1016 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %1015, <4 x float> %1010) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1017 = shufflevector <8 x half> %975, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1018 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %1017, <4 x float> %1012) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1019 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %1015, <4 x float> %1013) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1020 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %1017, <4 x float> %1014) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1021 = shufflevector <8 x half> %977, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1022 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %1021, <4 x float> %1016) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1023 = shufflevector <8 x half> %977, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1024 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %1023, <4 x float> %1018) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1025 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %1021, <4 x float> %1019) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1026 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %1023, <4 x float> %1020) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1027 = shufflevector <8 x half> %979, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1028 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %1027, <4 x float> %1022) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1029 = shufflevector <8 x half> %979, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1030 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %1029, <4 x float> %1024) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1031 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %1027, <4 x float> %1025) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1032 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %1029, <4 x float> %1026) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1033 = bitcast <4 x i32> %980 to <8 x half>, !dbg !13
  %1034 = shufflevector <8 x half> %1033, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1035 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %1034, <4 x float> %1028) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1036 = shufflevector <8 x half> %1033, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1037 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %1036, <4 x float> %1030) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1038 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %1034, <4 x float> %1031) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1039 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %1036, <4 x float> %1032) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %vdb.read1498 = getelementptr i8, ptr addrspace(3) %address.local935, i64 8192, !dbg !13
  %1040 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1498, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1041 = bitcast <4 x i32> %1040 to <8 x half>, !dbg !13
  %vdb.read1499 = getelementptr i8, ptr addrspace(3) %address.local945, i64 8192, !dbg !13
  %1042 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1499, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1043 = bitcast <4 x i32> %1042 to <8 x half>, !dbg !13
  %vdb.read1500 = getelementptr i8, ptr addrspace(3) %address.local949, i64 8192, !dbg !13
  %1044 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1500, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1045 = bitcast <4 x i32> %1044 to <8 x half>, !dbg !13
  %vdb.read1501 = getelementptr i8, ptr addrspace(3) %address.local953, i64 8192, !dbg !13
  %1046 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1501, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1047 = bitcast <4 x i32> %1046 to <8 x half>, !dbg !13
  %vdb.read1502 = getelementptr i8, ptr addrspace(3) %address.local955, i64 8192, !dbg !13
  %1048 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1502, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1049 = bitcast <4 x i32> %1048 to <8 x half>, !dbg !13
  %vdb.read1503 = getelementptr i8, ptr addrspace(3) %address.local959, i64 8192, !dbg !13
  %1050 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1503, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1051 = bitcast <4 x i32> %1050 to <8 x half>, !dbg !13
  %vdb.read1504 = getelementptr i8, ptr addrspace(3) %address.local963, i64 8192, !dbg !13
  %1052 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1504, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1053 = bitcast <4 x i32> %1052 to <8 x half>, !dbg !13
  %vdb.read1505 = getelementptr i8, ptr addrspace(3) %address.local967, i64 8192, !dbg !13
  %1054 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1505, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1055 = shufflevector <8 x half> %1041, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1056 = extractvalue [2 x <8 x float>] %128, 0, !dbg !28
  %1057 = shufflevector <8 x float> %1056, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %1058 = fmul <4 x float> %1057, %584, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1059 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1055, <4 x float> %1058) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1060 = shufflevector <8 x half> %1041, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1061 = shufflevector <8 x float> %1056, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %1062 = fmul <4 x float> %1061, %584, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1063 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1060, <4 x float> %1062) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %1064 = extractvalue [2 x <8 x float>] %128, 1, !dbg !28
  %1065 = shufflevector <8 x float> %1064, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %1066 = fmul <4 x float> %1065, %592, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1067 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %1055, <4 x float> %1066) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %1068 = shufflevector <8 x float> %1064, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %1069 = fmul <4 x float> %1068, %592, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1070 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %1060, <4 x float> %1069) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1071 = shufflevector <8 x half> %1043, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1072 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %1071, <4 x float> %1059) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1073 = shufflevector <8 x half> %1043, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1074 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %1073, <4 x float> %1063) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1075 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %1071, <4 x float> %1067) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1076 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %1073, <4 x float> %1070) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1077 = shufflevector <8 x half> %1045, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1078 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %1077, <4 x float> %1072) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1079 = shufflevector <8 x half> %1045, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1080 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %1079, <4 x float> %1074) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1081 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %1077, <4 x float> %1075) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1082 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %1079, <4 x float> %1076) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1083 = shufflevector <8 x half> %1047, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1084 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %1083, <4 x float> %1078) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1085 = shufflevector <8 x half> %1047, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1086 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %1085, <4 x float> %1080) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1087 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %1083, <4 x float> %1081) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1088 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %1085, <4 x float> %1082) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1089 = shufflevector <8 x half> %1049, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1090 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %1089, <4 x float> %1084) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1091 = shufflevector <8 x half> %1049, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1092 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %1091, <4 x float> %1086) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1093 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %1089, <4 x float> %1087) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1094 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %1091, <4 x float> %1088) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1095 = shufflevector <8 x half> %1051, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1096 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %1095, <4 x float> %1090) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1097 = shufflevector <8 x half> %1051, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1098 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %1097, <4 x float> %1092) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1099 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %1095, <4 x float> %1093) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1100 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %1097, <4 x float> %1094) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1101 = shufflevector <8 x half> %1053, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1102 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %1101, <4 x float> %1096) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1103 = shufflevector <8 x half> %1053, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1104 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %1103, <4 x float> %1098) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1105 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %1101, <4 x float> %1099) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1106 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %1103, <4 x float> %1100) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1107 = bitcast <4 x i32> %1054 to <8 x half>, !dbg !13
  %1108 = shufflevector <8 x half> %1107, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1109 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %1108, <4 x float> %1102) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1110 = shufflevector <8 x half> %1107, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1111 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %1110, <4 x float> %1104) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1112 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %1108, <4 x float> %1105) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1113 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %1110, <4 x float> %1106) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !7
  %1114 = shufflevector <4 x float> %883, <4 x float> %885, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1115 = insertvalue [2 x <8 x float>] poison, <8 x float> %1114, 0, !dbg !7
  %1116 = shufflevector <4 x float> %886, <4 x float> %887, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1117 = insertvalue [2 x <8 x float>] %1115, <8 x float> %1116, 1, !dbg !7
  %1118 = insertvalue [2 x <1 x float>] poison, <1 x float> %548, 0, !dbg !7
  %1119 = insertvalue [2 x <1 x float>] %1118, <1 x float> %561, 1, !dbg !7
  %1120 = shufflevector <4 x float> %959, <4 x float> %961, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1121 = insertvalue [2 x <8 x float>] poison, <8 x float> %1120, 0, !dbg !7
  %1122 = shufflevector <4 x float> %962, <4 x float> %963, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1123 = insertvalue [2 x <8 x float>] %1121, <8 x float> %1122, 1, !dbg !7
  %1124 = shufflevector <4 x float> %1035, <4 x float> %1037, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1125 = insertvalue [2 x <8 x float>] poison, <8 x float> %1124, 0, !dbg !7
  %1126 = shufflevector <4 x float> %1038, <4 x float> %1039, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1127 = insertvalue [2 x <8 x float>] %1125, <8 x float> %1126, 1, !dbg !7
  %1128 = shufflevector <4 x float> %1109, <4 x float> %1111, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1129 = insertvalue [2 x <8 x float>] poison, <8 x float> %1128, 0, !dbg !7
  %1130 = shufflevector <4 x float> %1112, <4 x float> %1113, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1131 = insertvalue [2 x <8 x float>] %1129, <8 x float> %1130, 1, !dbg !7
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %1132 = add nuw nsw i64 %122, 1, !dbg !7
  br label %121, !dbg !7

1133:                                             ; preds = %121
  %lane.sum.field184 = extractvalue [2 x <1 x float>] %124, 1, !dbg !7
  %lane.sum.local185 = extractelement <1 x float> %lane.sum.field184, i64 0, !dbg !7
  %lane.sum.lane.lo186 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !7
  %lane.sum.lane187 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %lane.sum.lane.lo186), !dbg !7
  %1134 = shl nuw nsw i32 %lane.sum.lane187, 2, !dbg !7
  %1135 = xor i32 %1134, 64, !dbg !7
  %1136 = bitcast float %lane.sum.local185 to i32, !dbg !7
  %lane.sum.shuffle188 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1135, i32 %1136), !dbg !7
  %1137 = bitcast i32 %lane.sum.shuffle188 to float, !dbg !7
  %lane.sum.final189 = fadd float %lane.sum.local185, %1137, !dbg !7
  %1138 = shl nuw nsw i32 %lane.sum.lane187, 2, !dbg !7
  %1139 = xor i32 %1138, 128, !dbg !7
  %1140 = bitcast float %lane.sum.final189 to i32, !dbg !7
  %lane.sum.shuffle190 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1139, i32 %1140), !dbg !7
  %1141 = bitcast i32 %lane.sum.shuffle190 to float, !dbg !7
  %lane.sum.final191 = fadd float %lane.sum.final189, %1141, !dbg !7
  %1142 = insertelement <1 x float> poison, float %lane.sum.final191, i64 0, !dbg !7
  %lane.sum.field = extractvalue [2 x <1 x float>] %124, 0, !dbg !7
  %lane.sum.local = extractelement <1 x float> %lane.sum.field, i64 0, !dbg !7
  %lane.sum.lane.lo = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !7
  %lane.sum.lane = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %lane.sum.lane.lo), !dbg !7
  %1143 = shl nuw nsw i32 %lane.sum.lane, 2, !dbg !7
  %1144 = xor i32 %1143, 64, !dbg !7
  %1145 = bitcast float %lane.sum.local to i32, !dbg !7
  %lane.sum.shuffle = call i32 @llvm.amdgcn.ds.bpermute(i32 %1144, i32 %1145), !dbg !7
  %1146 = bitcast i32 %lane.sum.shuffle to float, !dbg !7
  %lane.sum.final = fadd float %lane.sum.local, %1146, !dbg !7
  %1147 = shl nuw nsw i32 %lane.sum.lane, 2, !dbg !7
  %1148 = xor i32 %1147, 128, !dbg !7
  %1149 = bitcast float %lane.sum.final to i32, !dbg !7
  %lane.sum.shuffle182 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1148, i32 %1149), !dbg !7
  %1150 = bitcast i32 %lane.sum.shuffle182 to float, !dbg !7
  %lane.sum.final183 = fadd float %lane.sum.final, %1150, !dbg !7
  %1151 = insertelement <1 x float> poison, float %lane.sum.final183, i64 0, !dbg !7
  %1152 = and i32 %27, 16, !dbg !7
  %.not = icmp eq i32 %1152, 0, !dbg !7
  %1153 = and i32 %27, 32, !dbg !7
  %.not36 = icmp eq i32 %1153, 0, !dbg !7
  %1154 = extractvalue [2 x <8 x float>] %128, 0, !dbg !7
  %1155 = extractelement <8 x float> %1154, i64 0, !dbg !7
  %1156 = extractelement <8 x float> %1154, i64 1, !dbg !7
  %1157 = extractelement <8 x float> %1154, i64 2, !dbg !7
  %1158 = extractelement <8 x float> %1154, i64 3, !dbg !7
  %1159 = select i1 %.not, float %1156, float %1155, !dbg !7
  %1160 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !7
  %1161 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1160), !dbg !7
  %1162 = add nuw nsw i32 %1161, 64, !dbg !7
  %1163 = xor i32 %1161, 16, !dbg !7
  %1164 = and i32 %1162, 192, !dbg !7
  %1165 = icmp samesign ult i32 %1163, %1164, !dbg !7
  %1166 = select i1 %1165, i32 %1163, i32 %1161, !dbg !7
  %1167 = shl nuw nsw i32 %1166, 2, !dbg !7
  %1168 = bitcast float %1159 to i32, !dbg !7
  %1169 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1167, i32 %1168), !dbg !7
  %1170 = bitcast i32 %1169 to float, !dbg !7
  %1171 = select i1 %.not, float %1155, float %1170, !dbg !7
  %1172 = select i1 %.not, float %1170, float %1156, !dbg !7
  %1173 = select i1 %.not, float %1158, float %1157, !dbg !7
  %1174 = bitcast float %1173 to i32, !dbg !7
  %1175 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1167, i32 %1174), !dbg !7
  %1176 = bitcast i32 %1175 to float, !dbg !7
  %1177 = select i1 %.not, float %1157, float %1176, !dbg !7
  %1178 = select i1 %.not, float %1176, float %1158, !dbg !7
  %1179 = select i1 %.not36, float %1177, float %1171, !dbg !7
  %1180 = xor i32 %1161, 32, !dbg !7
  %1181 = icmp samesign ult i32 %1180, %1164, !dbg !7
  %1182 = select i1 %1181, i32 %1180, i32 %1161, !dbg !7
  %1183 = shl nuw nsw i32 %1182, 2, !dbg !7
  %1184 = bitcast float %1179 to i32, !dbg !7
  %1185 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1183, i32 %1184), !dbg !7
  %1186 = bitcast i32 %1185 to float, !dbg !7
  %1187 = select i1 %.not36, float %1171, float %1186, !dbg !7
  %1188 = select i1 %.not36, float %1186, float %1177, !dbg !7
  %1189 = select i1 %.not36, float %1178, float %1172, !dbg !7
  %1190 = bitcast float %1189 to i32, !dbg !7
  %1191 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1183, i32 %1190), !dbg !7
  %1192 = bitcast i32 %1191 to float, !dbg !7
  %1193 = select i1 %.not36, float %1172, float %1192, !dbg !7
  %1194 = select i1 %.not36, float %1192, float %1178, !dbg !7
  %1195 = extractvalue [2 x <8 x float>] %127, 0, !dbg !7
  %1196 = extractelement <8 x float> %1195, i64 0, !dbg !7
  %1197 = extractelement <8 x float> %1195, i64 1, !dbg !7
  %1198 = extractelement <8 x float> %1195, i64 2, !dbg !7
  %1199 = extractelement <8 x float> %1195, i64 3, !dbg !7
  %1200 = select i1 %.not, float %1197, float %1196, !dbg !7
  %1201 = bitcast float %1200 to i32, !dbg !7
  %1202 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1167, i32 %1201), !dbg !7
  %1203 = bitcast i32 %1202 to float, !dbg !7
  %1204 = select i1 %.not, float %1196, float %1203, !dbg !7
  %1205 = select i1 %.not, float %1203, float %1197, !dbg !7
  %1206 = select i1 %.not, float %1199, float %1198, !dbg !7
  %1207 = bitcast float %1206 to i32, !dbg !7
  %1208 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1167, i32 %1207), !dbg !7
  %1209 = bitcast i32 %1208 to float, !dbg !7
  %1210 = select i1 %.not, float %1198, float %1209, !dbg !7
  %1211 = select i1 %.not, float %1209, float %1199, !dbg !7
  %1212 = select i1 %.not36, float %1210, float %1204, !dbg !7
  %1213 = bitcast float %1212 to i32, !dbg !7
  %1214 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1183, i32 %1213), !dbg !7
  %1215 = bitcast i32 %1214 to float, !dbg !7
  %1216 = select i1 %.not36, float %1204, float %1215, !dbg !7
  %1217 = select i1 %.not36, float %1215, float %1210, !dbg !7
  %1218 = select i1 %.not36, float %1211, float %1205, !dbg !7
  %1219 = bitcast float %1218 to i32, !dbg !7
  %1220 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1183, i32 %1219), !dbg !7
  %1221 = bitcast i32 %1220 to float, !dbg !7
  %1222 = select i1 %.not36, float %1205, float %1221, !dbg !7
  %1223 = select i1 %.not36, float %1221, float %1211, !dbg !7
  %1224 = extractvalue [2 x <8 x float>] %126, 0, !dbg !7
  %1225 = extractelement <8 x float> %1224, i64 0, !dbg !7
  %1226 = extractelement <8 x float> %1224, i64 1, !dbg !7
  %1227 = extractelement <8 x float> %1224, i64 2, !dbg !7
  %1228 = extractelement <8 x float> %1224, i64 3, !dbg !7
  %1229 = select i1 %.not, float %1226, float %1225, !dbg !7
  %1230 = bitcast float %1229 to i32, !dbg !7
  %1231 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1167, i32 %1230), !dbg !7
  %1232 = bitcast i32 %1231 to float, !dbg !7
  %1233 = select i1 %.not, float %1225, float %1232, !dbg !7
  %1234 = select i1 %.not, float %1232, float %1226, !dbg !7
  %1235 = select i1 %.not, float %1228, float %1227, !dbg !7
  %1236 = bitcast float %1235 to i32, !dbg !7
  %1237 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1167, i32 %1236), !dbg !7
  %1238 = bitcast i32 %1237 to float, !dbg !7
  %1239 = select i1 %.not, float %1227, float %1238, !dbg !7
  %1240 = select i1 %.not, float %1238, float %1228, !dbg !7
  %1241 = select i1 %.not36, float %1239, float %1233, !dbg !7
  %1242 = bitcast float %1241 to i32, !dbg !7
  %1243 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1183, i32 %1242), !dbg !7
  %1244 = bitcast i32 %1243 to float, !dbg !7
  %1245 = select i1 %.not36, float %1233, float %1244, !dbg !7
  %1246 = select i1 %.not36, float %1244, float %1239, !dbg !7
  %1247 = select i1 %.not36, float %1240, float %1234, !dbg !7
  %1248 = bitcast float %1247 to i32, !dbg !7
  %1249 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1183, i32 %1248), !dbg !7
  %1250 = bitcast i32 %1249 to float, !dbg !7
  %1251 = select i1 %.not36, float %1234, float %1250, !dbg !7
  %1252 = select i1 %.not36, float %1250, float %1240, !dbg !7
  %1253 = extractvalue [2 x <8 x float>] %123, 0, !dbg !7
  %1254 = extractelement <8 x float> %1253, i64 0, !dbg !7
  %1255 = extractelement <8 x float> %1253, i64 1, !dbg !7
  %1256 = extractelement <8 x float> %1253, i64 2, !dbg !7
  %1257 = extractelement <8 x float> %1253, i64 3, !dbg !7
  %1258 = select i1 %.not, float %1255, float %1254, !dbg !7
  %1259 = bitcast float %1258 to i32, !dbg !7
  %1260 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1167, i32 %1259), !dbg !7
  %1261 = bitcast i32 %1260 to float, !dbg !7
  %1262 = select i1 %.not, float %1254, float %1261, !dbg !7
  %1263 = select i1 %.not, float %1261, float %1255, !dbg !7
  %1264 = select i1 %.not, float %1257, float %1256, !dbg !7
  %1265 = bitcast float %1264 to i32, !dbg !7
  %1266 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1167, i32 %1265), !dbg !7
  %1267 = bitcast i32 %1266 to float, !dbg !7
  %1268 = select i1 %.not, float %1256, float %1267, !dbg !7
  %1269 = select i1 %.not, float %1267, float %1257, !dbg !7
  %1270 = select i1 %.not36, float %1268, float %1262, !dbg !7
  %1271 = bitcast float %1270 to i32, !dbg !7
  %1272 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1183, i32 %1271), !dbg !7
  %1273 = select i1 %.not36, float %1269, float %1263, !dbg !7
  %1274 = bitcast float %1273 to i32, !dbg !7
  %1275 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1183, i32 %1274), !dbg !7
  %1276 = shufflevector <1 x float> %1151, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !16
  %1277 = extractelement <8 x float> %1154, i64 4, !dbg !7
  %1278 = extractelement <8 x float> %1154, i64 5, !dbg !7
  %1279 = extractelement <8 x float> %1154, i64 6, !dbg !7
  %1280 = extractelement <8 x float> %1154, i64 7, !dbg !7
  %1281 = select i1 %.not, float %1278, float %1277, !dbg !7
  %1282 = bitcast float %1281 to i32, !dbg !7
  %1283 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1167, i32 %1282), !dbg !7
  %1284 = bitcast i32 %1283 to float, !dbg !7
  %1285 = select i1 %.not, float %1277, float %1284, !dbg !7
  %1286 = select i1 %.not, float %1284, float %1278, !dbg !7
  %1287 = select i1 %.not, float %1280, float %1279, !dbg !7
  %1288 = bitcast float %1287 to i32, !dbg !7
  %1289 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1167, i32 %1288), !dbg !7
  %1290 = bitcast i32 %1289 to float, !dbg !7
  %1291 = select i1 %.not, float %1279, float %1290, !dbg !7
  %1292 = select i1 %.not, float %1290, float %1280, !dbg !7
  %1293 = select i1 %.not36, float %1291, float %1285, !dbg !7
  %1294 = bitcast float %1293 to i32, !dbg !7
  %1295 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1183, i32 %1294), !dbg !7
  %1296 = bitcast i32 %1295 to float, !dbg !7
  %1297 = select i1 %.not36, float %1285, float %1296, !dbg !7
  %1298 = select i1 %.not36, float %1296, float %1291, !dbg !7
  %1299 = select i1 %.not36, float %1292, float %1286, !dbg !7
  %1300 = bitcast float %1299 to i32, !dbg !7
  %1301 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1183, i32 %1300), !dbg !7
  %1302 = bitcast i32 %1301 to float, !dbg !7
  %1303 = select i1 %.not36, float %1286, float %1302, !dbg !7
  %1304 = select i1 %.not36, float %1302, float %1292, !dbg !7
  %1305 = extractelement <8 x float> %1195, i64 4, !dbg !7
  %1306 = extractelement <8 x float> %1195, i64 5, !dbg !7
  %1307 = extractelement <8 x float> %1195, i64 6, !dbg !7
  %1308 = extractelement <8 x float> %1195, i64 7, !dbg !7
  %1309 = select i1 %.not, float %1306, float %1305, !dbg !7
  %1310 = bitcast float %1309 to i32, !dbg !7
  %1311 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1167, i32 %1310), !dbg !7
  %1312 = bitcast i32 %1311 to float, !dbg !7
  %1313 = select i1 %.not, float %1305, float %1312, !dbg !7
  %1314 = select i1 %.not, float %1312, float %1306, !dbg !7
  %1315 = select i1 %.not, float %1308, float %1307, !dbg !7
  %1316 = bitcast float %1315 to i32, !dbg !7
  %1317 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1167, i32 %1316), !dbg !7
  %1318 = bitcast i32 %1317 to float, !dbg !7
  %1319 = select i1 %.not, float %1307, float %1318, !dbg !7
  %1320 = select i1 %.not, float %1318, float %1308, !dbg !7
  %1321 = select i1 %.not36, float %1319, float %1313, !dbg !7
  %1322 = bitcast float %1321 to i32, !dbg !7
  %1323 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1183, i32 %1322), !dbg !7
  %1324 = bitcast i32 %1323 to float, !dbg !7
  %1325 = select i1 %.not36, float %1313, float %1324, !dbg !7
  %1326 = select i1 %.not36, float %1324, float %1319, !dbg !7
  %1327 = select i1 %.not36, float %1320, float %1314, !dbg !7
  %1328 = bitcast float %1327 to i32, !dbg !7
  %1329 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1183, i32 %1328), !dbg !7
  %1330 = bitcast i32 %1329 to float, !dbg !7
  %1331 = select i1 %.not36, float %1314, float %1330, !dbg !7
  %1332 = select i1 %.not36, float %1330, float %1320, !dbg !7
  %1333 = extractelement <8 x float> %1224, i64 4, !dbg !7
  %1334 = extractelement <8 x float> %1224, i64 5, !dbg !7
  %1335 = extractelement <8 x float> %1224, i64 6, !dbg !7
  %1336 = extractelement <8 x float> %1224, i64 7, !dbg !7
  %1337 = select i1 %.not, float %1334, float %1333, !dbg !7
  %1338 = bitcast float %1337 to i32, !dbg !7
  %1339 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1167, i32 %1338), !dbg !7
  %1340 = bitcast i32 %1339 to float, !dbg !7
  %1341 = select i1 %.not, float %1333, float %1340, !dbg !7
  %1342 = select i1 %.not, float %1340, float %1334, !dbg !7
  %1343 = select i1 %.not, float %1336, float %1335, !dbg !7
  %1344 = bitcast float %1343 to i32, !dbg !7
  %1345 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1167, i32 %1344), !dbg !7
  %1346 = bitcast i32 %1345 to float, !dbg !7
  %1347 = select i1 %.not, float %1335, float %1346, !dbg !7
  %1348 = select i1 %.not, float %1346, float %1336, !dbg !7
  %1349 = select i1 %.not36, float %1347, float %1341, !dbg !7
  %1350 = bitcast float %1349 to i32, !dbg !7
  %1351 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1183, i32 %1350), !dbg !7
  %1352 = bitcast i32 %1351 to float, !dbg !7
  %1353 = select i1 %.not36, float %1341, float %1352, !dbg !7
  %1354 = select i1 %.not36, float %1352, float %1347, !dbg !7
  %1355 = select i1 %.not36, float %1348, float %1342, !dbg !7
  %1356 = bitcast float %1355 to i32, !dbg !7
  %1357 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1183, i32 %1356), !dbg !7
  %1358 = bitcast i32 %1357 to float, !dbg !7
  %1359 = select i1 %.not36, float %1342, float %1358, !dbg !7
  %1360 = select i1 %.not36, float %1358, float %1348, !dbg !7
  %1361 = extractelement <8 x float> %1253, i64 4, !dbg !7
  %1362 = extractelement <8 x float> %1253, i64 5, !dbg !7
  %1363 = extractelement <8 x float> %1253, i64 6, !dbg !7
  %1364 = extractelement <8 x float> %1253, i64 7, !dbg !7
  %1365 = select i1 %.not, float %1362, float %1361, !dbg !7
  %1366 = bitcast float %1365 to i32, !dbg !7
  %1367 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1167, i32 %1366), !dbg !7
  %1368 = bitcast i32 %1367 to float, !dbg !7
  %1369 = select i1 %.not, float %1361, float %1368, !dbg !7
  %1370 = select i1 %.not, float %1368, float %1362, !dbg !7
  %1371 = select i1 %.not, float %1364, float %1363, !dbg !7
  %1372 = bitcast float %1371 to i32, !dbg !7
  %1373 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1167, i32 %1372), !dbg !7
  %1374 = bitcast i32 %1373 to float, !dbg !7
  %1375 = select i1 %.not, float %1363, float %1374, !dbg !7
  %1376 = select i1 %.not, float %1374, float %1364, !dbg !7
  %1377 = select i1 %.not36, float %1375, float %1369, !dbg !7
  %1378 = bitcast float %1377 to i32, !dbg !7
  %1379 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1183, i32 %1378), !dbg !7
  %1380 = select i1 %.not36, float %1376, float %1370, !dbg !7
  %1381 = bitcast float %1380 to i32, !dbg !7
  %1382 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1183, i32 %1381), !dbg !7
  %1383 = extractvalue [2 x <8 x float>] %128, 1, !dbg !7
  %1384 = extractelement <8 x float> %1383, i64 0, !dbg !7
  %1385 = extractelement <8 x float> %1383, i64 1, !dbg !7
  %1386 = extractelement <8 x float> %1383, i64 2, !dbg !7
  %1387 = extractelement <8 x float> %1383, i64 3, !dbg !7
  %1388 = select i1 %.not, float %1385, float %1384, !dbg !7
  %1389 = bitcast float %1388 to i32, !dbg !7
  %1390 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1167, i32 %1389), !dbg !7
  %1391 = bitcast i32 %1390 to float, !dbg !7
  %1392 = select i1 %.not, float %1384, float %1391, !dbg !7
  %1393 = select i1 %.not, float %1391, float %1385, !dbg !7
  %1394 = select i1 %.not, float %1387, float %1386, !dbg !7
  %1395 = bitcast float %1394 to i32, !dbg !7
  %1396 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1167, i32 %1395), !dbg !7
  %1397 = bitcast i32 %1396 to float, !dbg !7
  %1398 = select i1 %.not, float %1386, float %1397, !dbg !7
  %1399 = select i1 %.not, float %1397, float %1387, !dbg !7
  %1400 = select i1 %.not36, float %1398, float %1392, !dbg !7
  %1401 = bitcast float %1400 to i32, !dbg !7
  %1402 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1183, i32 %1401), !dbg !7
  %1403 = bitcast i32 %1402 to float, !dbg !7
  %1404 = select i1 %.not36, float %1392, float %1403, !dbg !7
  %1405 = select i1 %.not36, float %1403, float %1398, !dbg !7
  %1406 = select i1 %.not36, float %1399, float %1393, !dbg !7
  %1407 = bitcast float %1406 to i32, !dbg !7
  %1408 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1183, i32 %1407), !dbg !7
  %1409 = bitcast i32 %1408 to float, !dbg !7
  %1410 = select i1 %.not36, float %1393, float %1409, !dbg !7
  %1411 = select i1 %.not36, float %1409, float %1399, !dbg !7
  %1412 = extractvalue [2 x <8 x float>] %127, 1, !dbg !7
  %1413 = extractelement <8 x float> %1412, i64 0, !dbg !7
  %1414 = extractelement <8 x float> %1412, i64 1, !dbg !7
  %1415 = extractelement <8 x float> %1412, i64 2, !dbg !7
  %1416 = extractelement <8 x float> %1412, i64 3, !dbg !7
  %1417 = select i1 %.not, float %1414, float %1413, !dbg !7
  %1418 = bitcast float %1417 to i32, !dbg !7
  %1419 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1167, i32 %1418), !dbg !7
  %1420 = bitcast i32 %1419 to float, !dbg !7
  %1421 = select i1 %.not, float %1413, float %1420, !dbg !7
  %1422 = select i1 %.not, float %1420, float %1414, !dbg !7
  %1423 = select i1 %.not, float %1416, float %1415, !dbg !7
  %1424 = bitcast float %1423 to i32, !dbg !7
  %1425 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1167, i32 %1424), !dbg !7
  %1426 = bitcast i32 %1425 to float, !dbg !7
  %1427 = select i1 %.not, float %1415, float %1426, !dbg !7
  %1428 = select i1 %.not, float %1426, float %1416, !dbg !7
  %1429 = select i1 %.not36, float %1427, float %1421, !dbg !7
  %1430 = bitcast float %1429 to i32, !dbg !7
  %1431 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1183, i32 %1430), !dbg !7
  %1432 = bitcast i32 %1431 to float, !dbg !7
  %1433 = select i1 %.not36, float %1421, float %1432, !dbg !7
  %1434 = select i1 %.not36, float %1432, float %1427, !dbg !7
  %1435 = select i1 %.not36, float %1428, float %1422, !dbg !7
  %1436 = bitcast float %1435 to i32, !dbg !7
  %1437 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1183, i32 %1436), !dbg !7
  %1438 = bitcast i32 %1437 to float, !dbg !7
  %1439 = select i1 %.not36, float %1422, float %1438, !dbg !7
  %1440 = select i1 %.not36, float %1438, float %1428, !dbg !7
  %1441 = extractvalue [2 x <8 x float>] %126, 1, !dbg !7
  %1442 = extractelement <8 x float> %1441, i64 0, !dbg !7
  %1443 = extractelement <8 x float> %1441, i64 1, !dbg !7
  %1444 = extractelement <8 x float> %1441, i64 2, !dbg !7
  %1445 = extractelement <8 x float> %1441, i64 3, !dbg !7
  %1446 = select i1 %.not, float %1443, float %1442, !dbg !7
  %1447 = bitcast float %1446 to i32, !dbg !7
  %1448 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1167, i32 %1447), !dbg !7
  %1449 = bitcast i32 %1448 to float, !dbg !7
  %1450 = select i1 %.not, float %1442, float %1449, !dbg !7
  %1451 = select i1 %.not, float %1449, float %1443, !dbg !7
  %1452 = select i1 %.not, float %1445, float %1444, !dbg !7
  %1453 = bitcast float %1452 to i32, !dbg !7
  %1454 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1167, i32 %1453), !dbg !7
  %1455 = bitcast i32 %1454 to float, !dbg !7
  %1456 = select i1 %.not, float %1444, float %1455, !dbg !7
  %1457 = select i1 %.not, float %1455, float %1445, !dbg !7
  %1458 = select i1 %.not36, float %1456, float %1450, !dbg !7
  %1459 = bitcast float %1458 to i32, !dbg !7
  %1460 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1183, i32 %1459), !dbg !7
  %1461 = bitcast i32 %1460 to float, !dbg !7
  %1462 = select i1 %.not36, float %1450, float %1461, !dbg !7
  %1463 = select i1 %.not36, float %1461, float %1456, !dbg !7
  %1464 = select i1 %.not36, float %1457, float %1451, !dbg !7
  %1465 = bitcast float %1464 to i32, !dbg !7
  %1466 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1183, i32 %1465), !dbg !7
  %1467 = bitcast i32 %1466 to float, !dbg !7
  %1468 = select i1 %.not36, float %1451, float %1467, !dbg !7
  %1469 = select i1 %.not36, float %1467, float %1457, !dbg !7
  %1470 = extractvalue [2 x <8 x float>] %123, 1, !dbg !7
  %1471 = extractelement <8 x float> %1470, i64 0, !dbg !7
  %1472 = extractelement <8 x float> %1470, i64 1, !dbg !7
  %1473 = extractelement <8 x float> %1470, i64 2, !dbg !7
  %1474 = extractelement <8 x float> %1470, i64 3, !dbg !7
  %1475 = select i1 %.not, float %1472, float %1471, !dbg !7
  %1476 = bitcast float %1475 to i32, !dbg !7
  %1477 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1167, i32 %1476), !dbg !7
  %1478 = bitcast i32 %1477 to float, !dbg !7
  %1479 = select i1 %.not, float %1471, float %1478, !dbg !7
  %1480 = select i1 %.not, float %1478, float %1472, !dbg !7
  %1481 = select i1 %.not, float %1474, float %1473, !dbg !7
  %1482 = bitcast float %1481 to i32, !dbg !7
  %1483 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1167, i32 %1482), !dbg !7
  %1484 = bitcast i32 %1483 to float, !dbg !7
  %1485 = select i1 %.not, float %1473, float %1484, !dbg !7
  %1486 = select i1 %.not, float %1484, float %1474, !dbg !7
  %1487 = select i1 %.not36, float %1485, float %1479, !dbg !7
  %1488 = bitcast float %1487 to i32, !dbg !7
  %1489 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1183, i32 %1488), !dbg !7
  %1490 = select i1 %.not36, float %1486, float %1480, !dbg !7
  %1491 = bitcast float %1490 to i32, !dbg !7
  %1492 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1183, i32 %1491), !dbg !7
  %1493 = shufflevector <1 x float> %1142, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !16
  %1494 = extractelement <8 x float> %1383, i64 4, !dbg !7
  %1495 = extractelement <8 x float> %1383, i64 5, !dbg !7
  %1496 = extractelement <8 x float> %1383, i64 6, !dbg !7
  %1497 = extractelement <8 x float> %1383, i64 7, !dbg !7
  %1498 = select i1 %.not, float %1495, float %1494, !dbg !7
  %1499 = bitcast float %1498 to i32, !dbg !7
  %1500 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1167, i32 %1499), !dbg !7
  %1501 = bitcast i32 %1500 to float, !dbg !7
  %1502 = select i1 %.not, float %1494, float %1501, !dbg !7
  %1503 = select i1 %.not, float %1501, float %1495, !dbg !7
  %1504 = select i1 %.not, float %1497, float %1496, !dbg !7
  %1505 = bitcast float %1504 to i32, !dbg !7
  %1506 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1167, i32 %1505), !dbg !7
  %1507 = bitcast i32 %1506 to float, !dbg !7
  %1508 = select i1 %.not, float %1496, float %1507, !dbg !7
  %1509 = select i1 %.not, float %1507, float %1497, !dbg !7
  %1510 = select i1 %.not36, float %1508, float %1502, !dbg !7
  %1511 = bitcast float %1510 to i32, !dbg !7
  %1512 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1183, i32 %1511), !dbg !7
  %1513 = bitcast i32 %1512 to float, !dbg !7
  %1514 = select i1 %.not36, float %1502, float %1513, !dbg !7
  %1515 = select i1 %.not36, float %1513, float %1508, !dbg !7
  %1516 = select i1 %.not36, float %1509, float %1503, !dbg !7
  %1517 = bitcast float %1516 to i32, !dbg !7
  %1518 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1183, i32 %1517), !dbg !7
  %1519 = bitcast i32 %1518 to float, !dbg !7
  %1520 = select i1 %.not36, float %1503, float %1519, !dbg !7
  %1521 = select i1 %.not36, float %1519, float %1509, !dbg !7
  %1522 = extractelement <8 x float> %1412, i64 4, !dbg !7
  %1523 = extractelement <8 x float> %1412, i64 5, !dbg !7
  %1524 = extractelement <8 x float> %1412, i64 6, !dbg !7
  %1525 = extractelement <8 x float> %1412, i64 7, !dbg !7
  %1526 = select i1 %.not, float %1523, float %1522, !dbg !7
  %1527 = bitcast float %1526 to i32, !dbg !7
  %1528 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1167, i32 %1527), !dbg !7
  %1529 = bitcast i32 %1528 to float, !dbg !7
  %1530 = select i1 %.not, float %1522, float %1529, !dbg !7
  %1531 = select i1 %.not, float %1529, float %1523, !dbg !7
  %1532 = select i1 %.not, float %1525, float %1524, !dbg !7
  %1533 = bitcast float %1532 to i32, !dbg !7
  %1534 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1167, i32 %1533), !dbg !7
  %1535 = bitcast i32 %1534 to float, !dbg !7
  %1536 = select i1 %.not, float %1524, float %1535, !dbg !7
  %1537 = select i1 %.not, float %1535, float %1525, !dbg !7
  %1538 = select i1 %.not36, float %1536, float %1530, !dbg !7
  %1539 = bitcast float %1538 to i32, !dbg !7
  %1540 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1183, i32 %1539), !dbg !7
  %1541 = bitcast i32 %1540 to float, !dbg !7
  %1542 = select i1 %.not36, float %1530, float %1541, !dbg !7
  %1543 = select i1 %.not36, float %1541, float %1536, !dbg !7
  %1544 = select i1 %.not36, float %1537, float %1531, !dbg !7
  %1545 = bitcast float %1544 to i32, !dbg !7
  %1546 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1183, i32 %1545), !dbg !7
  %1547 = bitcast i32 %1546 to float, !dbg !7
  %1548 = select i1 %.not36, float %1531, float %1547, !dbg !7
  %1549 = select i1 %.not36, float %1547, float %1537, !dbg !7
  %1550 = extractelement <8 x float> %1441, i64 4, !dbg !7
  %1551 = extractelement <8 x float> %1441, i64 5, !dbg !7
  %1552 = extractelement <8 x float> %1441, i64 6, !dbg !7
  %1553 = extractelement <8 x float> %1441, i64 7, !dbg !7
  %1554 = select i1 %.not, float %1551, float %1550, !dbg !7
  %1555 = bitcast float %1554 to i32, !dbg !7
  %1556 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1167, i32 %1555), !dbg !7
  %1557 = bitcast i32 %1556 to float, !dbg !7
  %1558 = select i1 %.not, float %1550, float %1557, !dbg !7
  %1559 = select i1 %.not, float %1557, float %1551, !dbg !7
  %1560 = select i1 %.not, float %1553, float %1552, !dbg !7
  %1561 = bitcast float %1560 to i32, !dbg !7
  %1562 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1167, i32 %1561), !dbg !7
  %1563 = bitcast i32 %1562 to float, !dbg !7
  %1564 = select i1 %.not, float %1552, float %1563, !dbg !7
  %1565 = select i1 %.not, float %1563, float %1553, !dbg !7
  %1566 = select i1 %.not36, float %1564, float %1558, !dbg !7
  %1567 = bitcast float %1566 to i32, !dbg !7
  %1568 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1183, i32 %1567), !dbg !7
  %1569 = bitcast i32 %1568 to float, !dbg !7
  %1570 = select i1 %.not36, float %1558, float %1569, !dbg !7
  %1571 = select i1 %.not36, float %1569, float %1564, !dbg !7
  %1572 = select i1 %.not36, float %1565, float %1559, !dbg !7
  %1573 = bitcast float %1572 to i32, !dbg !7
  %1574 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1183, i32 %1573), !dbg !7
  %1575 = bitcast i32 %1574 to float, !dbg !7
  %1576 = select i1 %.not36, float %1559, float %1575, !dbg !7
  %1577 = select i1 %.not36, float %1575, float %1565, !dbg !7
  %1578 = extractelement <8 x float> %1470, i64 4, !dbg !7
  %1579 = extractelement <8 x float> %1470, i64 5, !dbg !7
  %1580 = extractelement <8 x float> %1470, i64 6, !dbg !7
  %1581 = extractelement <8 x float> %1470, i64 7, !dbg !7
  %1582 = select i1 %.not, float %1579, float %1578, !dbg !7
  %1583 = bitcast float %1582 to i32, !dbg !7
  %1584 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1167, i32 %1583), !dbg !7
  %1585 = bitcast i32 %1584 to float, !dbg !7
  %1586 = select i1 %.not, float %1578, float %1585, !dbg !7
  %1587 = select i1 %.not, float %1585, float %1579, !dbg !7
  %1588 = select i1 %.not, float %1581, float %1580, !dbg !7
  %1589 = bitcast float %1588 to i32, !dbg !7
  %1590 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1167, i32 %1589), !dbg !7
  %1591 = bitcast i32 %1590 to float, !dbg !7
  %1592 = select i1 %.not, float %1580, float %1591, !dbg !7
  %1593 = select i1 %.not, float %1591, float %1581, !dbg !7
  %1594 = select i1 %.not36, float %1592, float %1586, !dbg !7
  %1595 = bitcast float %1594 to i32, !dbg !7
  %1596 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1183, i32 %1595), !dbg !7
  %1597 = select i1 %.not36, float %1593, float %1587, !dbg !7
  %1598 = bitcast float %1597 to i32, !dbg !7
  %1599 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1183, i32 %1598), !dbg !7
  %1600 = bitcast i32 %1596 to float, !dbg !7
  %1601 = select i1 %.not36, float %1586, float %1600, !dbg !7
  %1602 = insertelement <4 x float> poison, float %1601, i64 0, !dbg !16
  %1603 = bitcast i32 %1599 to float, !dbg !7
  %1604 = select i1 %.not36, float %1587, float %1603, !dbg !7
  %1605 = insertelement <4 x float> %1602, float %1604, i64 1, !dbg !16
  %1606 = select i1 %.not36, float %1600, float %1592, !dbg !7
  %1607 = insertelement <4 x float> %1605, float %1606, i64 2, !dbg !16
  %1608 = select i1 %.not36, float %1603, float %1593, !dbg !7
  %1609 = insertelement <4 x float> %1607, float %1608, i64 3, !dbg !16
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1493, !dbg !16
  %normalized = fmul <4 x float> %1609, %row.reciprocal, !dbg !16
  %1610 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !17
  %1611 = extractelement <4 x half> %1610, i64 3, !dbg !17
  %1612 = extractelement <4 x half> %1610, i64 2, !dbg !17
  %1613 = extractelement <4 x half> %1610, i64 1, !dbg !17
  %1614 = extractelement <4 x half> %1610, i64 0, !dbg !17
  %1615 = bitcast i32 %1489 to float, !dbg !7
  %1616 = select i1 %.not36, float %1479, float %1615, !dbg !7
  %1617 = insertelement <4 x float> poison, float %1616, i64 0, !dbg !16
  %1618 = bitcast i32 %1492 to float, !dbg !7
  %1619 = select i1 %.not36, float %1480, float %1618, !dbg !7
  %1620 = insertelement <4 x float> %1617, float %1619, i64 1, !dbg !16
  %1621 = select i1 %.not36, float %1615, float %1485, !dbg !7
  %1622 = insertelement <4 x float> %1620, float %1621, i64 2, !dbg !16
  %1623 = select i1 %.not36, float %1618, float %1486, !dbg !7
  %1624 = insertelement <4 x float> %1622, float %1623, i64 3, !dbg !16
  %normalized37 = fmul <4 x float> %1624, %row.reciprocal, !dbg !16
  %1625 = fptrunc <4 x float> %normalized37 to <4 x half>, !dbg !17
  %1626 = extractelement <4 x half> %1625, i64 3, !dbg !17
  %1627 = extractelement <4 x half> %1625, i64 2, !dbg !17
  %1628 = extractelement <4 x half> %1625, i64 1, !dbg !17
  %1629 = extractelement <4 x half> %1625, i64 0, !dbg !17
  %1630 = bitcast i32 %1379 to float, !dbg !7
  %1631 = select i1 %.not36, float %1369, float %1630, !dbg !7
  %1632 = insertelement <4 x float> poison, float %1631, i64 0, !dbg !16
  %1633 = bitcast i32 %1382 to float, !dbg !7
  %1634 = select i1 %.not36, float %1370, float %1633, !dbg !7
  %1635 = insertelement <4 x float> %1632, float %1634, i64 1, !dbg !16
  %1636 = select i1 %.not36, float %1630, float %1375, !dbg !7
  %1637 = insertelement <4 x float> %1635, float %1636, i64 2, !dbg !16
  %1638 = select i1 %.not36, float %1633, float %1376, !dbg !7
  %1639 = insertelement <4 x float> %1637, float %1638, i64 3, !dbg !16
  %row.reciprocal38 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1276, !dbg !16
  %normalized39 = fmul <4 x float> %1639, %row.reciprocal38, !dbg !16
  %1640 = fptrunc <4 x float> %normalized39 to <4 x half>, !dbg !17
  %1641 = extractelement <4 x half> %1640, i64 3, !dbg !17
  %1642 = extractelement <4 x half> %1640, i64 2, !dbg !17
  %1643 = extractelement <4 x half> %1640, i64 1, !dbg !17
  %1644 = extractelement <4 x half> %1640, i64 0, !dbg !17
  %1645 = bitcast i32 %1272 to float, !dbg !7
  %1646 = select i1 %.not36, float %1262, float %1645, !dbg !7
  %1647 = insertelement <4 x float> poison, float %1646, i64 0, !dbg !16
  %1648 = bitcast i32 %1275 to float, !dbg !7
  %1649 = select i1 %.not36, float %1263, float %1648, !dbg !7
  %1650 = insertelement <4 x float> %1647, float %1649, i64 1, !dbg !16
  %1651 = select i1 %.not36, float %1645, float %1268, !dbg !7
  %1652 = insertelement <4 x float> %1650, float %1651, i64 2, !dbg !16
  %1653 = select i1 %.not36, float %1648, float %1269, !dbg !7
  %1654 = insertelement <4 x float> %1652, float %1653, i64 3, !dbg !16
  %normalized40 = fmul <4 x float> %1654, %row.reciprocal38, !dbg !16
  %1655 = fptrunc <4 x float> %normalized40 to <4 x half>, !dbg !17
  %1656 = extractelement <4 x half> %1655, i64 3, !dbg !17
  %1657 = extractelement <4 x half> %1655, i64 2, !dbg !17
  %1658 = extractelement <4 x half> %1655, i64 1, !dbg !17
  %1659 = extractelement <4 x half> %1655, i64 0, !dbg !17
  %1660 = add nuw nsw i64 %17, %18, !dbg !18
  %1661 = shl i64 %1660, 19, !dbg !18
  %1662 = and i64 %1661, 16252928, !dbg !18
  %address.thread974 = call i32 asm sideeffect "", "=v,0"(i32 %27) #9, !dbg !18, !range !32, !frisk.register_lifetimes !1
  %address.local975 = zext nneg i32 %address.thread974 to i64, !dbg !8
  %address.local976 = add nuw i64 %21, %address.local975, !dbg !18
  %1663 = lshr i64 %address.local975, 2, !dbg !18
  %address.local978 = and i64 %1663, 48, !dbg !18
  %address.local979 = add i64 %address.local976, %address.local978, !dbg !18
  %address.local980 = and i64 %address.local975, 240, !dbg !18
  %address.local981 = sub i64 %address.local979, %address.local980, !dbg !18
  %address.local982 = add i64 %address.local981, %22, !dbg !18
  %address.local983 = shl nuw nsw i64 %address.local982, 7, !dbg !18
  %address.local984 = add nuw nsw i64 %1662, %address.local983, !dbg !18
  %address.local985 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local984, !dbg !18
  %address.local986 = lshr i64 %address.local975, 2, !dbg !10
  %address.local987 = and i64 %address.local986, 12, !dbg !10
  %address.local988 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local987, !dbg !18
  store half %1659, ptr addrspace(1) %address.local988, align 2, !dbg !18
  %address.local989 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local984, !dbg !18
  %address.local990 = or disjoint i64 %address.local987, 1, !dbg !18
  %address.local991 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local989, i64 %address.local990, !dbg !18
  store half %1658, ptr addrspace(1) %address.local991, align 2, !dbg !18
  %address.local992 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local984, !dbg !18
  %address.local993 = or disjoint i64 %address.local987, 2, !dbg !18
  %address.local994 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local992, i64 %address.local993, !dbg !18
  store half %1657, ptr addrspace(1) %address.local994, align 2, !dbg !18
  %address.local995 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local984, !dbg !18
  %address.local996 = or disjoint i64 %address.local987, 3, !dbg !18
  %address.local997 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local995, i64 %address.local996, !dbg !18
  store half %1656, ptr addrspace(1) %address.local997, align 2, !dbg !18
  %address.local998 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local984, !dbg !18
  %address.local999 = or disjoint i64 %address.local987, 16, !dbg !18
  %address.local1000 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local998, i64 %address.local999, !dbg !18
  store half %1644, ptr addrspace(1) %address.local1000, align 2, !dbg !18
  %address.local1001 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local984, !dbg !18
  %address.local1002 = or disjoint i64 %address.local987, 17, !dbg !18
  %address.local1003 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1001, i64 %address.local1002, !dbg !18
  store half %1643, ptr addrspace(1) %address.local1003, align 2, !dbg !18
  %address.local1004 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local984, !dbg !18
  %address.local1005 = or disjoint i64 %address.local987, 18, !dbg !18
  %address.local1006 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1004, i64 %address.local1005, !dbg !18
  store half %1642, ptr addrspace(1) %address.local1006, align 2, !dbg !18
  %address.local1007 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local984, !dbg !18
  %address.local1008 = or disjoint i64 %address.local987, 19, !dbg !18
  %address.local1009 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1007, i64 %address.local1008, !dbg !18
  store half %1641, ptr addrspace(1) %address.local1009, align 2, !dbg !18
  %address.local1010 = shl i64 %address.local982, 7, !dbg !18
  %address.local1011 = add i64 %address.local1010, 8192, !dbg !18
  %address.local1012 = add nuw nsw i64 %1662, %address.local1011, !dbg !18
  %address.local1013 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1012, !dbg !18
  %address.local1014 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local987, !dbg !18
  store half %1629, ptr addrspace(1) %address.local1014, align 2, !dbg !18
  %address.local1015 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1012, !dbg !18
  %address.local1016 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1015, i64 %address.local990, !dbg !18
  store half %1628, ptr addrspace(1) %address.local1016, align 2, !dbg !18
  %address.local1017 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1012, !dbg !18
  %address.local1018 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1017, i64 %address.local993, !dbg !18
  store half %1627, ptr addrspace(1) %address.local1018, align 2, !dbg !18
  %address.local1019 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1012, !dbg !18
  %address.local1020 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1019, i64 %address.local996, !dbg !18
  store half %1626, ptr addrspace(1) %address.local1020, align 2, !dbg !18
  %address.local1021 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1012, !dbg !18
  %address.local1022 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1021, i64 %address.local999, !dbg !18
  store half %1614, ptr addrspace(1) %address.local1022, align 2, !dbg !18
  %address.local1023 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1012, !dbg !18
  %address.local1024 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1023, i64 %address.local1002, !dbg !18
  store half %1613, ptr addrspace(1) %address.local1024, align 2, !dbg !18
  %address.local1025 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1012, !dbg !18
  %address.local1026 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1025, i64 %address.local1005, !dbg !18
  store half %1612, ptr addrspace(1) %address.local1026, align 2, !dbg !18
  %address.local1027 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1012, !dbg !18
  %address.local1028 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1027, i64 %address.local1008, !dbg !18
  store half %1611, ptr addrspace(1) %address.local1028, align 2, !dbg !18
  %1664 = insertelement <4 x float> poison, float %1570, i64 0, !dbg !16
  %1665 = insertelement <4 x float> %1664, float %1576, i64 1, !dbg !16
  %1666 = insertelement <4 x float> %1665, float %1571, i64 2, !dbg !16
  %1667 = insertelement <4 x float> %1666, float %1577, i64 3, !dbg !16
  %row.reciprocal41 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1493, !dbg !16
  %normalized42 = fmul <4 x float> %1667, %row.reciprocal41, !dbg !16
  %1668 = fptrunc <4 x float> %normalized42 to <4 x half>, !dbg !17
  %1669 = extractelement <4 x half> %1668, i64 3, !dbg !17
  %1670 = extractelement <4 x half> %1668, i64 2, !dbg !17
  %1671 = extractelement <4 x half> %1668, i64 1, !dbg !17
  %1672 = extractelement <4 x half> %1668, i64 0, !dbg !17
  %1673 = insertelement <4 x float> poison, float %1462, i64 0, !dbg !16
  %1674 = insertelement <4 x float> %1673, float %1468, i64 1, !dbg !16
  %1675 = insertelement <4 x float> %1674, float %1463, i64 2, !dbg !16
  %1676 = insertelement <4 x float> %1675, float %1469, i64 3, !dbg !16
  %normalized43 = fmul <4 x float> %1676, %row.reciprocal41, !dbg !16
  %1677 = fptrunc <4 x float> %normalized43 to <4 x half>, !dbg !17
  %1678 = extractelement <4 x half> %1677, i64 3, !dbg !17
  %1679 = extractelement <4 x half> %1677, i64 2, !dbg !17
  %1680 = extractelement <4 x half> %1677, i64 1, !dbg !17
  %1681 = extractelement <4 x half> %1677, i64 0, !dbg !17
  %1682 = insertelement <4 x float> poison, float %1353, i64 0, !dbg !16
  %1683 = insertelement <4 x float> %1682, float %1359, i64 1, !dbg !16
  %1684 = insertelement <4 x float> %1683, float %1354, i64 2, !dbg !16
  %1685 = insertelement <4 x float> %1684, float %1360, i64 3, !dbg !16
  %row.reciprocal44 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1276, !dbg !16
  %normalized45 = fmul <4 x float> %1685, %row.reciprocal44, !dbg !16
  %1686 = fptrunc <4 x float> %normalized45 to <4 x half>, !dbg !17
  %1687 = extractelement <4 x half> %1686, i64 3, !dbg !17
  %1688 = extractelement <4 x half> %1686, i64 2, !dbg !17
  %1689 = extractelement <4 x half> %1686, i64 1, !dbg !17
  %1690 = extractelement <4 x half> %1686, i64 0, !dbg !17
  %1691 = insertelement <4 x float> poison, float %1245, i64 0, !dbg !16
  %1692 = insertelement <4 x float> %1691, float %1251, i64 1, !dbg !16
  %1693 = insertelement <4 x float> %1692, float %1246, i64 2, !dbg !16
  %1694 = insertelement <4 x float> %1693, float %1252, i64 3, !dbg !16
  %normalized46 = fmul <4 x float> %1694, %row.reciprocal44, !dbg !16
  %1695 = fptrunc <4 x float> %normalized46 to <4 x half>, !dbg !17
  %1696 = extractelement <4 x half> %1695, i64 3, !dbg !17
  %1697 = extractelement <4 x half> %1695, i64 2, !dbg !17
  %1698 = extractelement <4 x half> %1695, i64 1, !dbg !17
  %1699 = extractelement <4 x half> %1695, i64 0, !dbg !17
  %1700 = shl nuw nsw i64 %20, 19, !dbg !18
  %address.thread1029 = call i32 asm sideeffect "", "=v,0"(i32 %27) #9, !dbg !18, !range !32, !frisk.register_lifetimes !1
  %address.local1030 = zext nneg i32 %address.thread1029 to i64, !dbg !8
  %address.local1031 = add nuw i64 %21, %address.local1030, !dbg !18
  %1701 = lshr i64 %address.local1030, 2, !dbg !18
  %address.local1033 = and i64 %1701, 48, !dbg !18
  %address.local1034 = add i64 %address.local1031, %address.local1033, !dbg !18
  %address.local1035 = and i64 %address.local1030, 240, !dbg !18
  %address.local1036 = sub i64 %address.local1034, %address.local1035, !dbg !18
  %address.local1037 = add i64 %address.local1036, %22, !dbg !18
  %address.local1038 = shl nuw nsw i64 %address.local1037, 7, !dbg !18
  %address.local1039 = add nuw nsw i64 %1700, %address.local1038, !dbg !18
  %address.local1040 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1039, !dbg !18
  %address.local1041 = lshr i64 %address.local1030, 2, !dbg !10
  %address.local1042 = and i64 %address.local1041, 12, !dbg !10
  %address.local1043 = or disjoint i64 %address.local1042, 32, !dbg !18
  %address.local1044 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1040, i64 %address.local1043, !dbg !18
  store half %1699, ptr addrspace(1) %address.local1044, align 2, !dbg !18
  %address.local1045 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1039, !dbg !18
  %address.local1046 = or disjoint i64 %address.local1042, 33, !dbg !18
  %address.local1047 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1045, i64 %address.local1046, !dbg !18
  store half %1698, ptr addrspace(1) %address.local1047, align 2, !dbg !18
  %address.local1048 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1039, !dbg !18
  %address.local1049 = or disjoint i64 %address.local1042, 34, !dbg !18
  %address.local1050 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1048, i64 %address.local1049, !dbg !18
  store half %1697, ptr addrspace(1) %address.local1050, align 2, !dbg !18
  %address.local1051 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1039, !dbg !18
  %address.local1052 = or disjoint i64 %address.local1042, 35, !dbg !18
  %address.local1053 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1051, i64 %address.local1052, !dbg !18
  store half %1696, ptr addrspace(1) %address.local1053, align 2, !dbg !18
  %address.local1054 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1039, !dbg !18
  %address.local1055 = or disjoint i64 %address.local1042, 48, !dbg !18
  %address.local1056 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1055, !dbg !18
  store half %1690, ptr addrspace(1) %address.local1056, align 2, !dbg !18
  %address.local1057 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1039, !dbg !18
  %address.local1058 = or disjoint i64 %address.local1042, 49, !dbg !18
  %address.local1059 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1057, i64 %address.local1058, !dbg !18
  store half %1689, ptr addrspace(1) %address.local1059, align 2, !dbg !18
  %address.local1060 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1039, !dbg !18
  %address.local1061 = or disjoint i64 %address.local1042, 50, !dbg !18
  %address.local1062 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1060, i64 %address.local1061, !dbg !18
  store half %1688, ptr addrspace(1) %address.local1062, align 2, !dbg !18
  %address.local1063 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1039, !dbg !18
  %address.local1064 = or i64 %address.local1041, 51, !dbg !18
  %address.local1065 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1063, i64 %address.local1064, !dbg !18
  store half %1687, ptr addrspace(1) %address.local1065, align 2, !dbg !18
  %address.local1066 = shl i64 %address.local1037, 7, !dbg !18
  %address.local1067 = add i64 %address.local1066, 8192, !dbg !18
  %address.local1068 = add nuw nsw i64 %1700, %address.local1067, !dbg !18
  %address.local1069 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1068, !dbg !18
  %address.local1070 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1069, i64 %address.local1043, !dbg !18
  store half %1681, ptr addrspace(1) %address.local1070, align 2, !dbg !18
  %address.local1071 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1068, !dbg !18
  %address.local1072 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1071, i64 %address.local1046, !dbg !18
  store half %1680, ptr addrspace(1) %address.local1072, align 2, !dbg !18
  %address.local1073 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1068, !dbg !18
  %address.local1074 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1073, i64 %address.local1049, !dbg !18
  store half %1679, ptr addrspace(1) %address.local1074, align 2, !dbg !18
  %address.local1075 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1068, !dbg !18
  %address.local1076 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1075, i64 %address.local1052, !dbg !18
  store half %1678, ptr addrspace(1) %address.local1076, align 2, !dbg !18
  %address.local1077 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1068, !dbg !18
  %address.local1078 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1077, i64 %address.local1055, !dbg !18
  store half %1672, ptr addrspace(1) %address.local1078, align 2, !dbg !18
  %address.local1079 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1068, !dbg !18
  %address.local1080 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1079, i64 %address.local1058, !dbg !18
  store half %1671, ptr addrspace(1) %address.local1080, align 2, !dbg !18
  %address.local1081 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1068, !dbg !18
  %address.local1082 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1061, !dbg !18
  store half %1670, ptr addrspace(1) %address.local1082, align 2, !dbg !18
  %address.local1083 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1068, !dbg !18
  %address.local1084 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1083, i64 %address.local1064, !dbg !18
  store half %1669, ptr addrspace(1) %address.local1084, align 2, !dbg !18
  %1702 = insertelement <4 x float> poison, float %1542, i64 0, !dbg !16
  %1703 = insertelement <4 x float> %1702, float %1548, i64 1, !dbg !16
  %1704 = insertelement <4 x float> %1703, float %1543, i64 2, !dbg !16
  %1705 = insertelement <4 x float> %1704, float %1549, i64 3, !dbg !16
  %row.reciprocal47 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1493, !dbg !16
  %normalized48 = fmul <4 x float> %1705, %row.reciprocal47, !dbg !16
  %1706 = fptrunc <4 x float> %normalized48 to <4 x half>, !dbg !17
  %1707 = extractelement <4 x half> %1706, i64 3, !dbg !17
  %1708 = extractelement <4 x half> %1706, i64 2, !dbg !17
  %1709 = extractelement <4 x half> %1706, i64 1, !dbg !17
  %1710 = extractelement <4 x half> %1706, i64 0, !dbg !17
  %1711 = insertelement <4 x float> poison, float %1433, i64 0, !dbg !16
  %1712 = insertelement <4 x float> %1711, float %1439, i64 1, !dbg !16
  %1713 = insertelement <4 x float> %1712, float %1434, i64 2, !dbg !16
  %1714 = insertelement <4 x float> %1713, float %1440, i64 3, !dbg !16
  %normalized49 = fmul <4 x float> %1714, %row.reciprocal47, !dbg !16
  %1715 = fptrunc <4 x float> %normalized49 to <4 x half>, !dbg !17
  %1716 = extractelement <4 x half> %1715, i64 3, !dbg !17
  %1717 = extractelement <4 x half> %1715, i64 2, !dbg !17
  %1718 = extractelement <4 x half> %1715, i64 1, !dbg !17
  %1719 = extractelement <4 x half> %1715, i64 0, !dbg !17
  %1720 = insertelement <4 x float> poison, float %1325, i64 0, !dbg !16
  %1721 = insertelement <4 x float> %1720, float %1331, i64 1, !dbg !16
  %1722 = insertelement <4 x float> %1721, float %1326, i64 2, !dbg !16
  %1723 = insertelement <4 x float> %1722, float %1332, i64 3, !dbg !16
  %row.reciprocal50 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1276, !dbg !16
  %normalized51 = fmul <4 x float> %1723, %row.reciprocal50, !dbg !16
  %1724 = fptrunc <4 x float> %normalized51 to <4 x half>, !dbg !17
  %1725 = extractelement <4 x half> %1724, i64 3, !dbg !17
  %1726 = extractelement <4 x half> %1724, i64 2, !dbg !17
  %1727 = extractelement <4 x half> %1724, i64 1, !dbg !17
  %1728 = extractelement <4 x half> %1724, i64 0, !dbg !17
  %1729 = insertelement <4 x float> poison, float %1216, i64 0, !dbg !16
  %1730 = insertelement <4 x float> %1729, float %1222, i64 1, !dbg !16
  %1731 = insertelement <4 x float> %1730, float %1217, i64 2, !dbg !16
  %1732 = insertelement <4 x float> %1731, float %1223, i64 3, !dbg !16
  %normalized52 = fmul <4 x float> %1732, %row.reciprocal50, !dbg !16
  %1733 = fptrunc <4 x float> %normalized52 to <4 x half>, !dbg !17
  %1734 = extractelement <4 x half> %1733, i64 3, !dbg !17
  %1735 = extractelement <4 x half> %1733, i64 2, !dbg !17
  %1736 = extractelement <4 x half> %1733, i64 1, !dbg !17
  %1737 = extractelement <4 x half> %1733, i64 0, !dbg !17
  %1738 = shl nuw nsw i64 %20, 19, !dbg !18
  %address.thread1085 = call i32 asm sideeffect "", "=v,0"(i32 %27) #9, !dbg !18, !range !32, !frisk.register_lifetimes !1
  %address.local1086 = zext nneg i32 %address.thread1085 to i64, !dbg !8
  %address.local1087 = add nuw i64 %21, %address.local1086, !dbg !18
  %1739 = lshr i64 %address.local1086, 2, !dbg !18
  %address.local1089 = and i64 %1739, 48, !dbg !18
  %address.local1090 = add i64 %address.local1087, %address.local1089, !dbg !18
  %address.local1091 = and i64 %address.local1086, 240, !dbg !18
  %address.local1092 = sub i64 %address.local1090, %address.local1091, !dbg !18
  %address.local1093 = add i64 %address.local1092, %22, !dbg !18
  %address.local1094 = shl nuw nsw i64 %address.local1093, 7, !dbg !18
  %address.local1095 = add nuw nsw i64 %1738, %address.local1094, !dbg !18
  %address.local1096 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1095, !dbg !18
  %address.local1097 = lshr i64 %address.local1086, 2, !dbg !10
  %address.local1098 = and i64 %address.local1097, 12, !dbg !10
  %address.local1099 = or disjoint i64 %address.local1098, 64, !dbg !18
  %address.local1100 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1096, i64 %address.local1099, !dbg !18
  store half %1737, ptr addrspace(1) %address.local1100, align 2, !dbg !18
  %address.local1101 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1095, !dbg !18
  %address.local1102 = or disjoint i64 %address.local1098, 65, !dbg !18
  %address.local1103 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1101, i64 %address.local1102, !dbg !18
  store half %1736, ptr addrspace(1) %address.local1103, align 2, !dbg !18
  %address.local1104 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1095, !dbg !18
  %address.local1105 = or disjoint i64 %address.local1098, 66, !dbg !18
  %address.local1106 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1104, i64 %address.local1105, !dbg !18
  store half %1735, ptr addrspace(1) %address.local1106, align 2, !dbg !18
  %address.local1107 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1095, !dbg !18
  %address.local1108 = or disjoint i64 %address.local1098, 67, !dbg !18
  %address.local1109 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1107, i64 %address.local1108, !dbg !18
  store half %1734, ptr addrspace(1) %address.local1109, align 2, !dbg !18
  %address.local1110 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1095, !dbg !18
  %address.local1111 = or disjoint i64 %address.local1098, 80, !dbg !18
  %address.local1112 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1110, i64 %address.local1111, !dbg !18
  store half %1728, ptr addrspace(1) %address.local1112, align 2, !dbg !18
  %address.local1113 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1095, !dbg !18
  %address.local1114 = or disjoint i64 %address.local1098, 81, !dbg !18
  %address.local1115 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1113, i64 %address.local1114, !dbg !18
  store half %1727, ptr addrspace(1) %address.local1115, align 2, !dbg !18
  %address.local1116 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1095, !dbg !18
  %address.local1117 = or disjoint i64 %address.local1098, 82, !dbg !18
  %address.local1118 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1116, i64 %address.local1117, !dbg !18
  store half %1726, ptr addrspace(1) %address.local1118, align 2, !dbg !18
  %address.local1119 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1095, !dbg !18
  %address.local1120 = or disjoint i64 %address.local1098, 83, !dbg !18
  %address.local1121 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1119, i64 %address.local1120, !dbg !18
  store half %1725, ptr addrspace(1) %address.local1121, align 2, !dbg !18
  %address.local1122 = shl i64 %address.local1093, 7, !dbg !18
  %address.local1123 = add i64 %address.local1122, 8192, !dbg !18
  %address.local1124 = add nuw nsw i64 %1738, %address.local1123, !dbg !18
  %address.local1125 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1124, !dbg !18
  %address.local1126 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1125, i64 %address.local1099, !dbg !18
  store half %1719, ptr addrspace(1) %address.local1126, align 2, !dbg !18
  %address.local1127 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1124, !dbg !18
  %address.local1128 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1127, i64 %address.local1102, !dbg !18
  store half %1718, ptr addrspace(1) %address.local1128, align 2, !dbg !18
  %address.local1129 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1124, !dbg !18
  %address.local1130 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1129, i64 %address.local1105, !dbg !18
  store half %1717, ptr addrspace(1) %address.local1130, align 2, !dbg !18
  %address.local1131 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1124, !dbg !18
  %address.local1132 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1131, i64 %address.local1108, !dbg !18
  store half %1716, ptr addrspace(1) %address.local1132, align 2, !dbg !18
  %address.local1133 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1124, !dbg !18
  %address.local1134 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1133, i64 %address.local1111, !dbg !18
  store half %1710, ptr addrspace(1) %address.local1134, align 2, !dbg !18
  %address.local1135 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1124, !dbg !18
  %address.local1136 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1135, i64 %address.local1114, !dbg !18
  store half %1709, ptr addrspace(1) %address.local1136, align 2, !dbg !18
  %address.local1137 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1124, !dbg !18
  %address.local1138 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1137, i64 %address.local1117, !dbg !18
  store half %1708, ptr addrspace(1) %address.local1138, align 2, !dbg !18
  %address.local1139 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1124, !dbg !18
  %address.local1140 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1139, i64 %address.local1120, !dbg !18
  store half %1707, ptr addrspace(1) %address.local1140, align 2, !dbg !18
  %1740 = insertelement <4 x float> poison, float %1514, i64 0, !dbg !16
  %1741 = insertelement <4 x float> %1740, float %1520, i64 1, !dbg !16
  %1742 = insertelement <4 x float> %1741, float %1515, i64 2, !dbg !16
  %1743 = insertelement <4 x float> %1742, float %1521, i64 3, !dbg !16
  %row.reciprocal53 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1493, !dbg !16
  %normalized54 = fmul <4 x float> %1743, %row.reciprocal53, !dbg !16
  %1744 = fptrunc <4 x float> %normalized54 to <4 x half>, !dbg !17
  %1745 = extractelement <4 x half> %1744, i64 3, !dbg !17
  %1746 = extractelement <4 x half> %1744, i64 2, !dbg !17
  %1747 = extractelement <4 x half> %1744, i64 1, !dbg !17
  %1748 = extractelement <4 x half> %1744, i64 0, !dbg !17
  %1749 = insertelement <4 x float> poison, float %1404, i64 0, !dbg !16
  %1750 = insertelement <4 x float> %1749, float %1410, i64 1, !dbg !16
  %1751 = insertelement <4 x float> %1750, float %1405, i64 2, !dbg !16
  %1752 = insertelement <4 x float> %1751, float %1411, i64 3, !dbg !16
  %normalized55 = fmul <4 x float> %1752, %row.reciprocal53, !dbg !16
  %1753 = fptrunc <4 x float> %normalized55 to <4 x half>, !dbg !17
  %1754 = extractelement <4 x half> %1753, i64 3, !dbg !17
  %1755 = extractelement <4 x half> %1753, i64 2, !dbg !17
  %1756 = extractelement <4 x half> %1753, i64 1, !dbg !17
  %1757 = extractelement <4 x half> %1753, i64 0, !dbg !17
  %1758 = insertelement <4 x float> poison, float %1297, i64 0, !dbg !16
  %1759 = insertelement <4 x float> %1758, float %1303, i64 1, !dbg !16
  %1760 = insertelement <4 x float> %1759, float %1298, i64 2, !dbg !16
  %1761 = insertelement <4 x float> %1760, float %1304, i64 3, !dbg !16
  %row.reciprocal56 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1276, !dbg !16
  %normalized57 = fmul <4 x float> %1761, %row.reciprocal56, !dbg !16
  %1762 = fptrunc <4 x float> %normalized57 to <4 x half>, !dbg !17
  %1763 = extractelement <4 x half> %1762, i64 3, !dbg !17
  %1764 = extractelement <4 x half> %1762, i64 2, !dbg !17
  %1765 = extractelement <4 x half> %1762, i64 1, !dbg !17
  %1766 = extractelement <4 x half> %1762, i64 0, !dbg !17
  %1767 = insertelement <4 x float> poison, float %1187, i64 0, !dbg !16
  %1768 = insertelement <4 x float> %1767, float %1193, i64 1, !dbg !16
  %1769 = insertelement <4 x float> %1768, float %1188, i64 2, !dbg !16
  %1770 = insertelement <4 x float> %1769, float %1194, i64 3, !dbg !16
  %normalized58 = fmul <4 x float> %1770, %row.reciprocal56, !dbg !16
  %1771 = fptrunc <4 x float> %normalized58 to <4 x half>, !dbg !17
  %1772 = extractelement <4 x half> %1771, i64 3, !dbg !17
  %1773 = extractelement <4 x half> %1771, i64 2, !dbg !17
  %1774 = extractelement <4 x half> %1771, i64 1, !dbg !17
  %1775 = extractelement <4 x half> %1771, i64 0, !dbg !17
  %1776 = shl nuw nsw i64 %20, 19, !dbg !18
  %address.thread1141 = call i32 asm sideeffect "", "=v,0"(i32 %27) #9, !dbg !18, !range !32, !frisk.register_lifetimes !1
  %address.local1142 = zext nneg i32 %address.thread1141 to i64, !dbg !8
  %address.local1143 = add nuw i64 %21, %address.local1142, !dbg !18
  %1777 = lshr i64 %address.local1142, 2, !dbg !18
  %address.local1145 = and i64 %1777, 48, !dbg !18
  %address.local1146 = add i64 %address.local1143, %address.local1145, !dbg !18
  %address.local1147 = and i64 %address.local1142, 240, !dbg !18
  %address.local1148 = sub i64 %address.local1146, %address.local1147, !dbg !18
  %address.local1149 = add i64 %address.local1148, %22, !dbg !18
  %address.local1150 = shl nuw nsw i64 %address.local1149, 7, !dbg !18
  %address.local1151 = add nuw nsw i64 %1776, %address.local1150, !dbg !18
  %address.local1152 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1151, !dbg !18
  %address.local1153 = lshr i64 %address.local1142, 2, !dbg !10
  %address.local1154 = and i64 %address.local1153, 12, !dbg !10
  %address.local1155 = or disjoint i64 %address.local1154, 96, !dbg !18
  %address.local1156 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1152, i64 %address.local1155, !dbg !18
  store half %1775, ptr addrspace(1) %address.local1156, align 2, !dbg !18
  %address.local1157 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1151, !dbg !18
  %address.local1158 = or disjoint i64 %address.local1154, 97, !dbg !18
  %address.local1159 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1157, i64 %address.local1158, !dbg !18
  store half %1774, ptr addrspace(1) %address.local1159, align 2, !dbg !18
  %address.local1160 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1151, !dbg !18
  %address.local1161 = or disjoint i64 %address.local1154, 98, !dbg !18
  %address.local1162 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1160, i64 %address.local1161, !dbg !18
  store half %1773, ptr addrspace(1) %address.local1162, align 2, !dbg !18
  %address.local1163 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1151, !dbg !18
  %address.local1164 = or disjoint i64 %address.local1154, 99, !dbg !18
  %address.local1165 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1163, i64 %address.local1164, !dbg !18
  store half %1772, ptr addrspace(1) %address.local1165, align 2, !dbg !18
  %address.local1166 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1151, !dbg !18
  %address.local1167 = or disjoint i64 %address.local1154, 112, !dbg !18
  %address.local1168 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1166, i64 %address.local1167, !dbg !18
  store half %1766, ptr addrspace(1) %address.local1168, align 2, !dbg !18
  %address.local1169 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1151, !dbg !18
  %address.local1170 = or disjoint i64 %address.local1154, 113, !dbg !18
  %address.local1171 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1169, i64 %address.local1170, !dbg !18
  store half %1765, ptr addrspace(1) %address.local1171, align 2, !dbg !18
  %address.local1172 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1151, !dbg !18
  %address.local1173 = or disjoint i64 %address.local1154, 114, !dbg !18
  %address.local1174 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1172, i64 %address.local1173, !dbg !18
  store half %1764, ptr addrspace(1) %address.local1174, align 2, !dbg !18
  %address.local1175 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1151, !dbg !18
  %address.local1176 = or i64 %address.local1153, 115, !dbg !18
  %address.local1177 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1175, i64 %address.local1176, !dbg !18
  store half %1763, ptr addrspace(1) %address.local1177, align 2, !dbg !18
  %address.local1178 = shl i64 %address.local1149, 7, !dbg !18
  %address.local1179 = add i64 %address.local1178, 8192, !dbg !18
  %address.local1180 = add nuw nsw i64 %1776, %address.local1179, !dbg !18
  %address.local1181 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1180, !dbg !18
  %address.local1182 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1181, i64 %address.local1155, !dbg !18
  store half %1757, ptr addrspace(1) %address.local1182, align 2, !dbg !18
  %address.local1183 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1180, !dbg !18
  %address.local1184 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1183, i64 %address.local1158, !dbg !18
  store half %1756, ptr addrspace(1) %address.local1184, align 2, !dbg !18
  %address.local1185 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1180, !dbg !18
  %address.local1186 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1185, i64 %address.local1161, !dbg !18
  store half %1755, ptr addrspace(1) %address.local1186, align 2, !dbg !18
  %address.local1187 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1180, !dbg !18
  %address.local1188 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1187, i64 %address.local1164, !dbg !18
  store half %1754, ptr addrspace(1) %address.local1188, align 2, !dbg !18
  %address.local1189 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1180, !dbg !18
  %address.local1190 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1189, i64 %address.local1167, !dbg !18
  store half %1748, ptr addrspace(1) %address.local1190, align 2, !dbg !18
  %address.local1191 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1180, !dbg !18
  %address.local1192 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1191, i64 %address.local1170, !dbg !18
  store half %1747, ptr addrspace(1) %address.local1192, align 2, !dbg !18
  %address.local1193 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1180, !dbg !18
  %address.local1194 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1193, i64 %address.local1173, !dbg !18
  store half %1746, ptr addrspace(1) %address.local1194, align 2, !dbg !18
  %address.local1195 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1180, !dbg !18
  %address.local1196 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1195, i64 %address.local1176, !dbg !18
  store half %1745, ptr addrspace(1) %address.local1196, align 2, !dbg !18
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
attributes #1 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1" "frisk.global_prefetch"="8" "frisk.k_dma"="64:1" "frisk.register_comparisons"="1" "frisk.register_lifetimes"="1" "uniform-work-group-size" }
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

!0 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/noncausal_k_dma_pass/mytest/run-20261030-163658-kcakvj_l/grid-000002")
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
