; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/noncausal_k_dma_pass/mytest_k_swizzle64.ll'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/noncausal_k_dma_pass/mytest/run-20261030-163658-kcakvj_l/grid-000003/input.mlir"

@shm_0 = addrspace(3) global [16384 x i8] undef, align 16

; Function Attrs: convergent willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

declare float @__ocml_exp2_f32(float)

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3) #1 !dbg !33 !reqd_work_group_size !55 {
  %kdma.tid = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !34
  %kdma.lane = and i32 %kdma.tid, 63, !dbg !34
  %kdma.wave = and i32 %kdma.tid, -64, !dbg !34
  %5 = lshr i32 %kdma.lane, 1, !dbg !34
  %kdma.owner = or i32 %kdma.wave, %5, !dbg !34
  %kdma.owner.hi = add i32 %kdma.owner, 32, !dbg !34
  %6 = and i32 %kdma.lane, 1, !dbg !34
  %kdma.intra = shl i32 %6, 3, !dbg !34
  %kdma.intra.hi = xor i32 %kdma.intra, 8, !dbg !34
  %7 = call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !34
  %8 = sext i32 %7 to i64, !dbg !34
  %9 = call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !34
  %10 = sext i32 %9 to i64, !dbg !34
  %11 = shl nsw i64 %10, 5, !dbg !34
  %12 = add nsw i64 %11, %8, !dbg !34
  %13 = lshr i64 %12, 5, !dbg !34
  %14 = and i64 %13, 576460752303423456, !dbg !34
  %15 = and i64 %8, 31, !dbg !34
  %16 = or disjoint i64 %14, %15, !dbg !34
  %17 = lshr i64 %12, 5, !dbg !34
  %18 = lshr i64 %12, 10, !dbg !36
  %19 = add nuw nsw i64 %17, %18, !dbg !36
  %20 = and i64 %19, 31, !dbg !36
  %21 = shl nsw i64 %16, 7, !dbg !36
  %22 = mul nsw i64 %18, -4096, !dbg !36
  %23 = add i64 %21, %22, !dbg !36
  %24 = shl nuw nsw i64 %20, 19, !dbg !36
  %25 = shl nsw i64 %23, 7, !dbg !36
  %26 = add i64 %24, %25, !dbg !36
  %27 = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !36, !range !57
  %28 = zext nneg i32 %27 to i64, !dbg !36
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %27) #9, !dbg !36, !range !57, !frisk.register_lifetimes !30
  %address.local = zext nneg i32 %address.thread to i64, !dbg !36
  %address.local404 = shl nuw nsw i64 %address.local, 3, !dbg !36
  %address.local405 = and i64 %address.local404, 1536, !dbg !36
  %address.local406 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local405, !dbg !36
  %address.local407 = ptrtoint ptr addrspace(3) %address.local406 to i64, !dbg !36
  %address.local408 = trunc i64 %address.local407 to i32, !dbg !36
  %29 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local408), !dbg !36
  %address.local409 = shl nuw nsw i64 %address.local, 7, !dbg !36
  %address.local410 = and i64 %address.local409, 16256, !dbg !36
  %address.local411 = lshr i64 %address.local, 4, !dbg !36
  %address.local412 = and i64 %address.local411, 8, !dbg !36
  %address.local413 = or disjoint i64 %address.local410, %address.local412, !dbg !36
  %30 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !36
  %.idx1399 = or disjoint i64 %26, %address.local413, !dbg !36
  %.idx1399.tr = trunc i64 %.idx1399 to i32, !dbg !36
  %dma.buffer.offset1396 = shl i32 %.idx1399.tr, 1, !dbg !36
  %31 = insertelement <2 x i64> poison, i64 %30, i64 0, !dbg !36
  %32 = bitcast <2 x i64> %31 to <4 x i32>, !dbg !36
  %dma.buffer.rsrc1397 = shufflevector <4 x i32> %32, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !36
  %dma.buffer.lds1398 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %29), !dbg !36
  %33 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1398, i32 %dma.buffer.offset1396, <4 x i32> %dma.buffer.rsrc1397) #10, !dbg !36
  %address.local415 = or disjoint i64 %address.local, 256, !dbg !36
  %address.local416 = shl nuw nsw i64 %address.local415, 3, !dbg !36
  %address.local417 = and i64 %address.local416, 3584, !dbg !36
  %address.local418 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local417, !dbg !36
  %address.local419 = ptrtoint ptr addrspace(3) %address.local418 to i64, !dbg !36
  %address.local420 = trunc i64 %address.local419 to i32, !dbg !36
  %34 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local420), !dbg !36
  %address.local421 = shl nuw nsw i64 %address.local, 7, !dbg !36
  %address.local422 = and i64 %address.local421, 16256, !dbg !36
  %address.local423 = lshr i64 %address.local415, 4, !dbg !36
  %address.local424 = and i64 %address.local423, 24, !dbg !36
  %address.local425 = or disjoint i64 %address.local422, %address.local424, !dbg !36
  %35 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !36
  %.idx1400 = or disjoint i64 %26, %address.local425, !dbg !36
  %.idx1400.tr = trunc i64 %.idx1400 to i32, !dbg !36
  %dma.buffer.offset1393 = shl i32 %.idx1400.tr, 1, !dbg !36
  %36 = insertelement <2 x i64> poison, i64 %35, i64 0, !dbg !36
  %37 = bitcast <2 x i64> %36 to <4 x i32>, !dbg !36
  %dma.buffer.rsrc1394 = shufflevector <4 x i32> %37, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !36
  %dma.buffer.lds1395 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %34), !dbg !36
  %38 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1395, i32 %dma.buffer.offset1393, <4 x i32> %dma.buffer.rsrc1394) #10, !dbg !36
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !36
  fence syncscope("workgroup") release, !dbg !36
  call void @llvm.amdgcn.s.barrier(), !dbg !36
  fence syncscope("workgroup") acquire, !dbg !36
  %address.local427 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local405, !dbg !36
  %address.local428 = ptrtoint ptr addrspace(3) %address.local427 to i64, !dbg !36
  %address.local429 = trunc i64 %address.local428 to i32, !dbg !36
  %39 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local429), !dbg !36
  %40 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !36
  %41 = or disjoint i64 %26, 32, !dbg !36
  %42 = or disjoint i64 %address.local413, %41, !dbg !36
  %.tr = trunc i64 %42 to i32, !dbg !36
  %dma.buffer.offset1390 = shl i32 %.tr, 1, !dbg !36
  %43 = insertelement <2 x i64> poison, i64 %40, i64 0, !dbg !36
  %44 = bitcast <2 x i64> %43 to <4 x i32>, !dbg !36
  %dma.buffer.rsrc1391 = shufflevector <4 x i32> %44, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !36
  %dma.buffer.lds1392 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %39), !dbg !36
  %45 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1392, i32 %dma.buffer.offset1390, <4 x i32> %dma.buffer.rsrc1391) #10, !dbg !36
  %address.local431 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local417, !dbg !36
  %address.local432 = ptrtoint ptr addrspace(3) %address.local431 to i64, !dbg !36
  %address.local433 = trunc i64 %address.local432 to i32, !dbg !36
  %46 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local433), !dbg !36
  %47 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !36
  %48 = or disjoint i64 %26, 32, !dbg !36
  %49 = or disjoint i64 %address.local425, %48, !dbg !36
  %.tr1401 = trunc i64 %49 to i32, !dbg !36
  %dma.buffer.offset1387 = shl i32 %.tr1401, 1, !dbg !36
  %50 = insertelement <2 x i64> poison, i64 %47, i64 0, !dbg !36
  %51 = bitcast <2 x i64> %50 to <4 x i32>, !dbg !36
  %dma.buffer.rsrc1388 = shufflevector <4 x i32> %51, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !36
  %dma.buffer.lds1389 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %46), !dbg !36
  %52 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1389, i32 %dma.buffer.offset1387, <4 x i32> %dma.buffer.rsrc1388) #10, !dbg !36
  %address.local435 = lshr i32 %address.thread, 2, !dbg !37
  %address.local436 = and i32 %address.local435, 4, !dbg !37
  %address.local437 = shl nuw nsw i32 %address.thread, 1, !dbg !37
  %address.local438 = and i32 %address.local437, 384, !dbg !37
  %address.local439 = or disjoint i32 %address.local436, %address.local438, !dbg !37
  %address.local440 = shl nuw nsw i32 %address.thread, 3, !dbg !37
  %address.local441 = and i32 %address.local440, 120, !dbg !37
  %address.local442 = or disjoint i32 %address.local439, %address.local441, !dbg !37
  %address.local443 = shl nuw nsw i32 %address.thread, 5, !dbg !37
  %address.local444 = and i32 %address.local443, 1024, !dbg !37
  %address.local445 = or disjoint i32 %address.local442, %address.local444, !dbg !37
  %address.local446 = zext nneg i32 %address.local445 to i64, !dbg !37
  %address.local447 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local446, !dbg !37
  %53 = load <4 x half>, ptr addrspace(3) %address.local447, align 8, !dbg !37
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local447, i64 4096, !dbg !37
  %54 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !37
  %55 = fmul <4 x half> %53, splat (half 1.275630e-01), !dbg !37
  %lds.const.gep1197 = getelementptr i8, ptr addrspace(3) %address.local447, i64 1024, !dbg !37
  %56 = load <4 x half>, ptr addrspace(3) %lds.const.gep1197, align 8, !dbg !37
  %57 = fmul <4 x half> %54, splat (half 1.275630e-01), !dbg !37
  %lds.const.gep1198 = getelementptr i8, ptr addrspace(3) %address.local447, i64 5120, !dbg !37
  %58 = load <4 x half>, ptr addrspace(3) %lds.const.gep1198, align 8, !dbg !37
  %59 = fmul <4 x half> %56, splat (half 1.275630e-01), !dbg !37
  %60 = fmul <4 x half> %58, splat (half 1.275630e-01), !dbg !37
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !36
  fence syncscope("workgroup") release, !dbg !36
  call void @llvm.amdgcn.s.barrier(), !dbg !36
  fence syncscope("workgroup") acquire, !dbg !36
  %61 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local408), !dbg !36
  %62 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !36
  %63 = or disjoint i64 %26, 64, !dbg !36
  %64 = or disjoint i64 %address.local413, %63, !dbg !36
  %.tr1402 = trunc i64 %64 to i32, !dbg !36
  %dma.buffer.offset1384 = shl i32 %.tr1402, 1, !dbg !36
  %65 = insertelement <2 x i64> poison, i64 %62, i64 0, !dbg !36
  %66 = bitcast <2 x i64> %65 to <4 x i32>, !dbg !36
  %dma.buffer.rsrc1385 = shufflevector <4 x i32> %66, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !36
  %dma.buffer.lds1386 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %61), !dbg !36
  %67 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1386, i32 %dma.buffer.offset1384, <4 x i32> %dma.buffer.rsrc1385) #10, !dbg !36
  %68 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local420), !dbg !36
  %69 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !36
  %70 = or disjoint i64 %26, 64, !dbg !36
  %71 = or disjoint i64 %address.local425, %70, !dbg !36
  %.tr1403 = trunc i64 %71 to i32, !dbg !36
  %dma.buffer.offset1381 = shl i32 %.tr1403, 1, !dbg !36
  %72 = insertelement <2 x i64> poison, i64 %69, i64 0, !dbg !36
  %73 = bitcast <2 x i64> %72 to <4 x i32>, !dbg !36
  %dma.buffer.rsrc1382 = shufflevector <4 x i32> %73, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !36
  %dma.buffer.lds1383 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %68), !dbg !36
  %74 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1383, i32 %dma.buffer.offset1381, <4 x i32> %dma.buffer.rsrc1382) #10, !dbg !36
  %lds.const.gep1199 = getelementptr i8, ptr addrspace(3) %address.local447, i64 8192, !dbg !37
  %75 = load <4 x half>, ptr addrspace(3) %lds.const.gep1199, align 8, !dbg !37
  %lds.const.gep1200 = getelementptr i8, ptr addrspace(3) %address.local447, i64 12288, !dbg !37
  %76 = load <4 x half>, ptr addrspace(3) %lds.const.gep1200, align 8, !dbg !37
  %lds.const.gep1201 = getelementptr i8, ptr addrspace(3) %address.local447, i64 9216, !dbg !37
  %77 = load <4 x half>, ptr addrspace(3) %lds.const.gep1201, align 8, !dbg !37
  %78 = fmul <4 x half> %75, splat (half 1.275630e-01), !dbg !37
  %lds.const.gep1202 = getelementptr i8, ptr addrspace(3) %address.local447, i64 13312, !dbg !37
  %79 = load <4 x half>, ptr addrspace(3) %lds.const.gep1202, align 8, !dbg !37
  %80 = fmul <4 x half> %76, splat (half 1.275630e-01), !dbg !37
  %81 = fmul <4 x half> %77, splat (half 1.275630e-01), !dbg !37
  %82 = fmul <4 x half> %79, splat (half 1.275630e-01), !dbg !37
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !36
  fence syncscope("workgroup") release, !dbg !36
  call void @llvm.amdgcn.s.barrier(), !dbg !36
  fence syncscope("workgroup") acquire, !dbg !36
  %83 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local429), !dbg !36
  %84 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !36
  %85 = or disjoint i64 %26, 96, !dbg !36
  %86 = or disjoint i64 %address.local413, %85, !dbg !36
  %.tr1404 = trunc i64 %86 to i32, !dbg !36
  %dma.buffer.offset1378 = shl i32 %.tr1404, 1, !dbg !36
  %87 = insertelement <2 x i64> poison, i64 %84, i64 0, !dbg !36
  %88 = bitcast <2 x i64> %87 to <4 x i32>, !dbg !36
  %dma.buffer.rsrc1379 = shufflevector <4 x i32> %88, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !36
  %dma.buffer.lds1380 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %83), !dbg !36
  %89 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1380, i32 %dma.buffer.offset1378, <4 x i32> %dma.buffer.rsrc1379) #10, !dbg !36
  %90 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local433), !dbg !36
  %91 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !36
  %92 = or disjoint i64 %26, 96, !dbg !36
  %93 = or disjoint i64 %address.local425, %92, !dbg !36
  %.tr1405 = trunc i64 %93 to i32, !dbg !36
  %dma.buffer.offset1375 = shl i32 %.tr1405, 1, !dbg !36
  %94 = insertelement <2 x i64> poison, i64 %91, i64 0, !dbg !36
  %95 = bitcast <2 x i64> %94 to <4 x i32>, !dbg !36
  %dma.buffer.rsrc1376 = shufflevector <4 x i32> %95, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !36
  %dma.buffer.lds1377 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %90), !dbg !36
  %96 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1377, i32 %dma.buffer.offset1375, <4 x i32> %dma.buffer.rsrc1376) #10, !dbg !36
  %97 = load <4 x half>, ptr addrspace(3) %address.local447, align 8, !dbg !37
  %lds.const.gep1203 = getelementptr i8, ptr addrspace(3) %address.local447, i64 4096, !dbg !37
  %98 = load <4 x half>, ptr addrspace(3) %lds.const.gep1203, align 8, !dbg !37
  %lds.const.gep1204 = getelementptr i8, ptr addrspace(3) %address.local447, i64 1024, !dbg !37
  %99 = load <4 x half>, ptr addrspace(3) %lds.const.gep1204, align 8, !dbg !37
  %100 = fmul <4 x half> %97, splat (half 1.275630e-01), !dbg !37
  %lds.const.gep1205 = getelementptr i8, ptr addrspace(3) %address.local447, i64 5120, !dbg !37
  %101 = load <4 x half>, ptr addrspace(3) %lds.const.gep1205, align 8, !dbg !37
  %102 = fmul <4 x half> %98, splat (half 1.275630e-01), !dbg !37
  %103 = fmul <4 x half> %99, splat (half 1.275630e-01), !dbg !37
  %104 = fmul <4 x half> %101, splat (half 1.275630e-01), !dbg !37
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !36
  fence syncscope("workgroup") release, !dbg !36
  call void @llvm.amdgcn.s.barrier(), !dbg !36
  fence syncscope("workgroup") acquire, !dbg !36
  %lds.const.gep1206 = getelementptr i8, ptr addrspace(3) %address.local447, i64 8192, !dbg !37
  %105 = load <4 x half>, ptr addrspace(3) %lds.const.gep1206, align 8, !dbg !37
  %lds.const.gep1207 = getelementptr i8, ptr addrspace(3) %address.local447, i64 12288, !dbg !37
  %106 = load <4 x half>, ptr addrspace(3) %lds.const.gep1207, align 8, !dbg !37
  %lds.const.gep1208 = getelementptr i8, ptr addrspace(3) %address.local447, i64 9216, !dbg !37
  %107 = load <4 x half>, ptr addrspace(3) %lds.const.gep1208, align 8, !dbg !37
  %108 = fmul <4 x half> %105, splat (half 1.275630e-01), !dbg !37
  %lds.const.gep1209 = getelementptr i8, ptr addrspace(3) %address.local447, i64 13312, !dbg !37
  %109 = load <4 x half>, ptr addrspace(3) %lds.const.gep1209, align 8, !dbg !37
  %110 = fmul <4 x half> %106, splat (half 1.275630e-01), !dbg !37
  %111 = fmul <4 x half> %107, splat (half 1.275630e-01), !dbg !37
  %112 = fmul <4 x half> %109, splat (half 1.275630e-01), !dbg !37
  fence syncscope("workgroup") release, !dbg !35
  call void @llvm.amdgcn.s.barrier(), !dbg !35
  fence syncscope("workgroup") acquire, !dbg !35
  %113 = shl nuw nsw i64 %28, 2, !dbg !40
  %114 = shl nuw nsw i64 %28, 2, !dbg !40
  %115 = and i64 %114, 1008, !dbg !40
  %116 = sub nsw i64 %113, %115, !dbg !40
  %117 = and i32 %27, 1, !dbg !40
  %index.compact344 = zext nneg i32 %117 to i64, !dbg !40
  %118 = or disjoint i64 %116, %index.compact344, !dbg !40
  %119 = add nuw nsw i64 %118, 1, !dbg !40
  %120 = sdiv i64 %119, 2, !dbg !40
  br label %121, !dbg !35

121:                                              ; preds = %130, %4
  %122 = phi i64 [ %1144, %130 ], [ 0, %4 ], !dbg !35
  %123 = phi [2 x <8 x float>] [ %1129, %130 ], [ zeroinitializer, %4 ], !dbg !35
  %124 = phi [2 x <1 x float>] [ %823, %130 ], [ zeroinitializer, %4 ], !dbg !35
  %125 = phi [2 x <1 x float>] [ %1131, %130 ], [ [<1 x float> splat (float -inf), <1 x float> splat (float -inf)], %4 ], !dbg !35
  %126 = phi [2 x <8 x float>] [ %1135, %130 ], [ zeroinitializer, %4 ], !dbg !35
  %127 = phi [2 x <8 x float>] [ %1139, %130 ], [ zeroinitializer, %4 ], !dbg !35
  %128 = phi [2 x <8 x float>] [ %1143, %130 ], [ zeroinitializer, %4 ], !dbg !35
  %129 = icmp samesign ult i64 %122, 32, !dbg !35
  br i1 %129, label %130, label %1145, !dbg !35

130:                                              ; preds = %121
  %131 = shl i64 %17, 19, !dbg !38
  %132 = and i64 %131, 16252928, !dbg !38
  %133 = shl nuw nsw i64 %122, 14, !dbg !38
  %134 = or disjoint i64 %132, %133, !dbg !38
  %address.thread498 = call i32 asm sideeffect "", "=v,0"(i32 %27) #9, !dbg !38, !range !57, !frisk.register_lifetimes !30
  %address.local499 = zext nneg i32 %address.thread498 to i64, !dbg !36
  %address.local500 = shl nuw nsw i64 %address.local499, 3, !dbg !36
  %address.local501 = and i64 %address.local500, 1536, !dbg !36
  %address.local502 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local501, !dbg !36
  %address.local503 = ptrtoint ptr addrspace(3) %address.local502 to i64, !dbg !36
  %address.local504 = trunc i64 %address.local503 to i32, !dbg !36
  %135 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local504), !dbg !38
  %136 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !38
  %137 = insertelement <2 x i64> poison, i64 %136, i64 0, !dbg !38
  %138 = bitcast <2 x i64> %137 to <4 x i32>, !dbg !38
  %dma.buffer.rsrc1373 = shufflevector <4 x i32> %138, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !38
  %dma.buffer.lds1374 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %135), !dbg !38
  %kdma.source = zext nneg i32 %kdma.owner to i64, !dbg !38
  %kdma.source1506 = lshr i64 %kdma.source, 4, !dbg !38
  %kdma.source1507 = and i64 %kdma.source1506, 8, !dbg !38
  %kdma.source1508 = and i64 %kdma.source, 123, !dbg !38
  %kdma.source1509 = lshr i64 %kdma.source, 2, !dbg !38
  %kdma.source1510 = and i64 %kdma.source1509, 4, !dbg !38
  %kdma.source1511 = add nuw nsw i64 %kdma.source1510, %kdma.source, !dbg !38
  %kdma.source1512 = and i64 %kdma.source1511, 4, !dbg !38
  %kdma.source1513 = or disjoint i64 %kdma.source1508, %kdma.source1512, !dbg !38
  %kdma.source1514 = shl nuw nsw i64 %kdma.source1513, 7, !dbg !38
  %kdma.source1515 = or disjoint i64 %kdma.source1507, %kdma.source1514, !dbg !38
  %kdma.source1516 = or disjoint i64 %134, %kdma.source1515, !dbg !38
  %kdma.source1517 = trunc nuw i64 %kdma.source1516 to i32, !dbg !38
  %kdma.source1518 = shl nuw nsw i32 %kdma.source1517, 1, !dbg !38
  %kdma.offset = add i32 %kdma.source1518, %kdma.intra, !dbg !38
  %kdma.lds = add i32 %dma.buffer.lds1374, 0, !dbg !38
  %139 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds, i32 %kdma.offset, <4 x i32> %dma.buffer.rsrc1373) #10, !dbg !38
  %kdma.source1519 = zext nneg i32 %kdma.owner.hi to i64, !dbg !38
  %kdma.source1520 = lshr i64 %kdma.source1519, 4, !dbg !38
  %kdma.source1521 = and i64 %kdma.source1520, 8, !dbg !38
  %kdma.source1522 = and i64 %kdma.source1519, 123, !dbg !38
  %kdma.source1523 = lshr i64 %kdma.source1519, 2, !dbg !38
  %kdma.source1524 = and i64 %kdma.source1523, 4, !dbg !38
  %kdma.source1525 = add nuw nsw i64 %kdma.source1524, %kdma.source1519, !dbg !38
  %kdma.source1526 = and i64 %kdma.source1525, 4, !dbg !38
  %kdma.source1527 = or disjoint i64 %kdma.source1522, %kdma.source1526, !dbg !38
  %kdma.source1528 = shl nuw nsw i64 %kdma.source1527, 7, !dbg !38
  %kdma.source1529 = or disjoint i64 %kdma.source1521, %kdma.source1528, !dbg !38
  %kdma.source1530 = or disjoint i64 %134, %kdma.source1529, !dbg !38
  %kdma.source1531 = trunc nuw i64 %kdma.source1530 to i32, !dbg !38
  %kdma.source1532 = shl nuw nsw i32 %kdma.source1531, 1, !dbg !38
  %kdma.offset1533 = add i32 %kdma.source1532, %kdma.intra.hi, !dbg !38
  %kdma.lds1534 = add i32 %dma.buffer.lds1374, 512, !dbg !38
  %140 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1534, i32 %kdma.offset1533, <4 x i32> %dma.buffer.rsrc1373) #10, !dbg !38
  %address.local516 = or disjoint i64 %address.local499, 256, !dbg !36
  %address.local517 = shl nuw nsw i64 %address.local516, 3, !dbg !36
  %address.local518 = and i64 %address.local517, 3584, !dbg !36
  %address.local519 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local518, !dbg !36
  %address.local520 = ptrtoint ptr addrspace(3) %address.local519 to i64, !dbg !36
  %address.local521 = trunc i64 %address.local520 to i32, !dbg !36
  %141 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local521), !dbg !38
  %142 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !38
  %143 = insertelement <2 x i64> poison, i64 %142, i64 0, !dbg !38
  %144 = bitcast <2 x i64> %143 to <4 x i32>, !dbg !38
  %dma.buffer.rsrc1370 = shufflevector <4 x i32> %144, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !38
  %dma.buffer.lds1371 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %141), !dbg !38
  %kdma.source1535 = zext nneg i32 %kdma.owner to i64, !dbg !38
  %kdma.source1536 = or disjoint i64 %kdma.source1535, 256, !dbg !38
  %kdma.source1537 = lshr i64 %kdma.source1536, 4, !dbg !38
  %kdma.source1538 = and i64 %kdma.source1537, 24, !dbg !38
  %kdma.source1539 = and i64 %kdma.source1535, 123, !dbg !38
  %kdma.source1540 = lshr i64 %kdma.source1535, 2, !dbg !38
  %kdma.source1541 = and i64 %kdma.source1540, 4, !dbg !38
  %kdma.source1542 = add nuw nsw i64 %kdma.source1541, %kdma.source1535, !dbg !38
  %kdma.source1543 = and i64 %kdma.source1542, 4, !dbg !38
  %kdma.source1544 = or disjoint i64 %kdma.source1539, %kdma.source1543, !dbg !38
  %kdma.source1545 = shl nuw nsw i64 %kdma.source1544, 7, !dbg !38
  %kdma.source1546 = or disjoint i64 %kdma.source1538, %kdma.source1545, !dbg !38
  %kdma.source1547 = or disjoint i64 %134, %kdma.source1546, !dbg !38
  %kdma.source1548 = trunc nuw i64 %kdma.source1547 to i32, !dbg !38
  %kdma.source1549 = shl nuw nsw i32 %kdma.source1548, 1, !dbg !38
  %kdma.offset1550 = add i32 %kdma.source1549, %kdma.intra, !dbg !38
  %kdma.lds1551 = add i32 %dma.buffer.lds1371, 0, !dbg !38
  %145 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1551, i32 %kdma.offset1550, <4 x i32> %dma.buffer.rsrc1370) #10, !dbg !38
  %kdma.source1552 = zext nneg i32 %kdma.owner.hi to i64, !dbg !38
  %kdma.source1553 = or disjoint i64 %kdma.source1552, 256, !dbg !38
  %kdma.source1554 = lshr i64 %kdma.source1553, 4, !dbg !38
  %kdma.source1555 = and i64 %kdma.source1554, 24, !dbg !38
  %kdma.source1556 = and i64 %kdma.source1552, 123, !dbg !38
  %kdma.source1557 = lshr i64 %kdma.source1552, 2, !dbg !38
  %kdma.source1558 = and i64 %kdma.source1557, 4, !dbg !38
  %kdma.source1559 = add nuw nsw i64 %kdma.source1558, %kdma.source1552, !dbg !38
  %kdma.source1560 = and i64 %kdma.source1559, 4, !dbg !38
  %kdma.source1561 = or disjoint i64 %kdma.source1556, %kdma.source1560, !dbg !38
  %kdma.source1562 = shl nuw nsw i64 %kdma.source1561, 7, !dbg !38
  %kdma.source1563 = or disjoint i64 %kdma.source1555, %kdma.source1562, !dbg !38
  %kdma.source1564 = or disjoint i64 %134, %kdma.source1563, !dbg !38
  %kdma.source1565 = trunc nuw i64 %kdma.source1564 to i32, !dbg !38
  %kdma.source1566 = shl nuw nsw i32 %kdma.source1565, 1, !dbg !38
  %kdma.offset1567 = add i32 %kdma.source1566, %kdma.intra.hi, !dbg !38
  %kdma.lds1568 = add i32 %dma.buffer.lds1371, 512, !dbg !38
  %146 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1568, i32 %kdma.offset1567, <4 x i32> %dma.buffer.rsrc1370) #10, !dbg !38
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !38
  fence syncscope("workgroup") release, !dbg !38
  call void @llvm.amdgcn.s.barrier(), !dbg !38
  fence syncscope("workgroup") acquire, !dbg !38
  %147 = or disjoint i64 %134, 32, !dbg !38
  %address.local533 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local501, !dbg !36
  %address.local534 = ptrtoint ptr addrspace(3) %address.local533 to i64, !dbg !36
  %address.local535 = trunc i64 %address.local534 to i32, !dbg !36
  %148 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local535), !dbg !38
  %149 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !38
  %150 = insertelement <2 x i64> poison, i64 %149, i64 0, !dbg !38
  %151 = bitcast <2 x i64> %150 to <4 x i32>, !dbg !38
  %dma.buffer.rsrc1367 = shufflevector <4 x i32> %151, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !38
  %dma.buffer.lds1368 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %148), !dbg !38
  %kdma.source1569 = zext nneg i32 %kdma.owner to i64, !dbg !38
  %kdma.source1570 = lshr i64 %kdma.source1569, 4, !dbg !38
  %kdma.source1571 = and i64 %kdma.source1570, 8, !dbg !38
  %kdma.source1572 = and i64 %kdma.source1569, 123, !dbg !38
  %kdma.source1573 = lshr i64 %kdma.source1569, 2, !dbg !38
  %kdma.source1574 = and i64 %kdma.source1573, 4, !dbg !38
  %kdma.source1575 = add nuw nsw i64 %kdma.source1574, %kdma.source1569, !dbg !38
  %kdma.source1576 = and i64 %kdma.source1575, 4, !dbg !38
  %kdma.source1577 = or disjoint i64 %kdma.source1572, %kdma.source1576, !dbg !38
  %kdma.source1578 = shl nuw nsw i64 %kdma.source1577, 7, !dbg !38
  %kdma.source1579 = or disjoint i64 %kdma.source1571, %kdma.source1578, !dbg !38
  %kdma.source1580 = or disjoint i64 %147, %kdma.source1579, !dbg !38
  %kdma.source1581 = trunc nuw i64 %kdma.source1580 to i32, !dbg !38
  %kdma.source1582 = shl nuw nsw i32 %kdma.source1581, 1, !dbg !38
  %kdma.offset1583 = add i32 %kdma.source1582, %kdma.intra, !dbg !38
  %kdma.lds1584 = add i32 %dma.buffer.lds1368, 0, !dbg !38
  %152 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1584, i32 %kdma.offset1583, <4 x i32> %dma.buffer.rsrc1367) #10, !dbg !38
  %kdma.source1585 = zext nneg i32 %kdma.owner.hi to i64, !dbg !38
  %kdma.source1586 = lshr i64 %kdma.source1585, 4, !dbg !38
  %kdma.source1587 = and i64 %kdma.source1586, 8, !dbg !38
  %kdma.source1588 = and i64 %kdma.source1585, 123, !dbg !38
  %kdma.source1589 = lshr i64 %kdma.source1585, 2, !dbg !38
  %kdma.source1590 = and i64 %kdma.source1589, 4, !dbg !38
  %kdma.source1591 = add nuw nsw i64 %kdma.source1590, %kdma.source1585, !dbg !38
  %kdma.source1592 = and i64 %kdma.source1591, 4, !dbg !38
  %kdma.source1593 = or disjoint i64 %kdma.source1588, %kdma.source1592, !dbg !38
  %kdma.source1594 = shl nuw nsw i64 %kdma.source1593, 7, !dbg !38
  %kdma.source1595 = or disjoint i64 %kdma.source1587, %kdma.source1594, !dbg !38
  %kdma.source1596 = or disjoint i64 %147, %kdma.source1595, !dbg !38
  %kdma.source1597 = trunc nuw i64 %kdma.source1596 to i32, !dbg !38
  %kdma.source1598 = shl nuw nsw i32 %kdma.source1597, 1, !dbg !38
  %kdma.offset1599 = add i32 %kdma.source1598, %kdma.intra.hi, !dbg !38
  %kdma.lds1600 = add i32 %dma.buffer.lds1368, 512, !dbg !38
  %153 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1600, i32 %kdma.offset1599, <4 x i32> %dma.buffer.rsrc1367) #10, !dbg !38
  %address.local537 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local518, !dbg !36
  %address.local538 = ptrtoint ptr addrspace(3) %address.local537 to i64, !dbg !36
  %address.local539 = trunc i64 %address.local538 to i32, !dbg !36
  %154 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local539), !dbg !38
  %155 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !38
  %156 = insertelement <2 x i64> poison, i64 %155, i64 0, !dbg !38
  %157 = bitcast <2 x i64> %156 to <4 x i32>, !dbg !38
  %dma.buffer.rsrc1364 = shufflevector <4 x i32> %157, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !38
  %dma.buffer.lds1365 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %154), !dbg !38
  %kdma.source1601 = zext nneg i32 %kdma.owner to i64, !dbg !38
  %kdma.source1602 = or disjoint i64 %kdma.source1601, 256, !dbg !38
  %kdma.source1603 = lshr i64 %kdma.source1602, 4, !dbg !38
  %kdma.source1604 = and i64 %kdma.source1603, 24, !dbg !38
  %kdma.source1605 = and i64 %kdma.source1601, 123, !dbg !38
  %kdma.source1606 = lshr i64 %kdma.source1601, 2, !dbg !38
  %kdma.source1607 = and i64 %kdma.source1606, 4, !dbg !38
  %kdma.source1608 = add nuw nsw i64 %kdma.source1607, %kdma.source1601, !dbg !38
  %kdma.source1609 = and i64 %kdma.source1608, 4, !dbg !38
  %kdma.source1610 = or disjoint i64 %kdma.source1605, %kdma.source1609, !dbg !38
  %kdma.source1611 = shl nuw nsw i64 %kdma.source1610, 7, !dbg !38
  %kdma.source1612 = or disjoint i64 %kdma.source1604, %kdma.source1611, !dbg !38
  %kdma.source1613 = or disjoint i64 %147, %kdma.source1612, !dbg !38
  %kdma.source1614 = trunc nuw i64 %kdma.source1613 to i32, !dbg !38
  %kdma.source1615 = shl nuw nsw i32 %kdma.source1614, 1, !dbg !38
  %kdma.offset1616 = add i32 %kdma.source1615, %kdma.intra, !dbg !38
  %kdma.lds1617 = add i32 %dma.buffer.lds1365, 0, !dbg !38
  %158 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1617, i32 %kdma.offset1616, <4 x i32> %dma.buffer.rsrc1364) #10, !dbg !38
  %kdma.source1618 = zext nneg i32 %kdma.owner.hi to i64, !dbg !38
  %kdma.source1619 = or disjoint i64 %kdma.source1618, 256, !dbg !38
  %kdma.source1620 = lshr i64 %kdma.source1619, 4, !dbg !38
  %kdma.source1621 = and i64 %kdma.source1620, 24, !dbg !38
  %kdma.source1622 = and i64 %kdma.source1618, 123, !dbg !38
  %kdma.source1623 = lshr i64 %kdma.source1618, 2, !dbg !38
  %kdma.source1624 = and i64 %kdma.source1623, 4, !dbg !38
  %kdma.source1625 = add nuw nsw i64 %kdma.source1624, %kdma.source1618, !dbg !38
  %kdma.source1626 = and i64 %kdma.source1625, 4, !dbg !38
  %kdma.source1627 = or disjoint i64 %kdma.source1622, %kdma.source1626, !dbg !38
  %kdma.source1628 = shl nuw nsw i64 %kdma.source1627, 7, !dbg !38
  %kdma.source1629 = or disjoint i64 %kdma.source1621, %kdma.source1628, !dbg !38
  %kdma.source1630 = or disjoint i64 %147, %kdma.source1629, !dbg !38
  %kdma.source1631 = trunc nuw i64 %kdma.source1630 to i32, !dbg !38
  %kdma.source1632 = shl nuw nsw i32 %kdma.source1631, 1, !dbg !38
  %kdma.offset1633 = add i32 %kdma.source1632, %kdma.intra.hi, !dbg !38
  %kdma.lds1634 = add i32 %dma.buffer.lds1365, 512, !dbg !38
  %159 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1634, i32 %kdma.offset1633, <4 x i32> %dma.buffer.rsrc1364) #10, !dbg !38
  %address.local542 = lshr i64 %address.local499, 2, !dbg !40
  %address.local545 = shl nuw nsw i64 %address.local499, 1, !dbg !40
  %address.local546 = and i64 %address.local545, 24, !dbg !40
  %address.local548 = shl i64 %address.local499, 62, !dbg !40
  %address.local549 = ashr exact i64 %address.local548, 57, !dbg !40
  %address.local550 = and i64 %address.local549, -64, !dbg !40
  %address.local552 = lshr i64 %address.local499, 2, !dbg !37
  %address.local553 = and i64 %address.local552, 12, !dbg !37
  %address.local555 = and i64 %address.local499, 63, !dbg !37
  %address.local556 = lshr i64 %address.local555, 5, !dbg !37
  %address.local557 = mul nuw nsw i64 %address.local556, 1016, !dbg !37
  %.idx = mul nuw nsw i64 %address.local499, 320, !dbg !40
  %160 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %.idx, !dbg !40
  %.idx1410 = mul nsw i64 %address.local542, -1280, !dbg !40
  %161 = getelementptr i8, ptr addrspace(3) %160, i64 %.idx1410, !dbg !40
  %162 = getelementptr [2 x i8], ptr addrspace(3) %161, i64 %address.local546, !dbg !40
  %163 = getelementptr [2 x i8], ptr addrspace(3) %162, i64 %address.local550, !dbg !40
  %164 = getelementptr [2 x i8], ptr addrspace(3) %163, i64 %address.local553, !dbg !40
  %address.local559 = getelementptr [2 x i8], ptr addrspace(3) %164, i64 %address.local557, !dbg !40
  %165 = ptrtoint ptr addrspace(3) %address.local559 to i64, !dbg !40
  %166 = sub i64 %165, ptrtoint (ptr addrspace(3) @shm_0 to i64), !dbg !40
  %167 = lshr i64 %166, 6, !dbg !40
  %168 = and i64 %167, 8, !dbg !40
  %169 = xor i64 %166, %168, !dbg !40
  %170 = sub i64 %169, %166, !dbg !40
  %kdma.swizzled = getelementptr i8, ptr addrspace(3) %address.local559, i64 %170, !dbg !40
  %171 = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8, !dbg !40
  %172 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %55, <4 x half> %171, <4 x float> zeroinitializer) #9, !dbg !40
  %kdma.gep = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4096, !dbg !40
  %173 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8, !dbg !40
  %address.local570 = shl nuw nsw i32 %address.thread498, 1, !dbg !40
  %address.local571 = and i32 %address.local570, 24, !dbg !40
  %address.local572 = shl nuw nsw i32 %address.thread498, 8, !dbg !40
  %address.local573 = and i32 %address.local572, 768, !dbg !40
  %address.local574 = or disjoint i32 %address.local571, %address.local573, !dbg !40
  %address.local575 = shl nuw nsw i32 %address.thread498, 5, !dbg !40
  %address.local576 = and i32 %address.local575, 96, !dbg !40
  %address.local577 = or disjoint i32 %address.local574, %address.local576, !dbg !40
  %address.local578 = zext nneg i32 %address.local577 to i64, !dbg !40
  %address.base = call i64 asm sideeffect "", "=v,0"(i64 %120) #9, !dbg !40, !frisk.register_lifetimes !30
  %address.local579 = lshr i32 %address.thread498, 1, !dbg !40
  %address.local580 = and i32 %address.local579, 1, !dbg !40
  %address.local581 = zext nneg i32 %address.local580 to i64, !dbg !40
  %address.local582 = add nsw i64 %address.base, %address.local581, !dbg !40
  %address.local583 = shl i64 %address.local582, 6, !dbg !40
  %address.local584 = sub i64 %address.local578, %address.local583, !dbg !40
  %174 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local584, !dbg !40
  %175 = getelementptr [2 x i8], ptr addrspace(3) %174, i64 %address.local553, !dbg !40
  %176 = getelementptr [2 x i8], ptr addrspace(3) %175, i64 %address.local557, !dbg !40
  %address.local588 = getelementptr i8, ptr addrspace(3) %176, i64 64, !dbg !40
  %177 = ptrtoint ptr addrspace(3) %address.local588 to i64, !dbg !40
  %178 = sub i64 %177, ptrtoint (ptr addrspace(3) @shm_0 to i64), !dbg !40
  %179 = lshr i64 %178, 6, !dbg !40
  %180 = and i64 %179, 8, !dbg !40
  %181 = xor i64 %178, %180, !dbg !40
  %182 = sub i64 %181, %178, !dbg !40
  %kdma.swizzled1767 = getelementptr i8, ptr addrspace(3) %address.local588, i64 %182, !dbg !40
  %183 = load <4 x half>, ptr addrspace(3) %kdma.swizzled1767, align 8, !dbg !40
  %184 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %173, <4 x float> %172) #9, !dbg !40
  %185 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %55, <4 x half> %183, <4 x float> zeroinitializer) #9, !dbg !40
  %kdma.gep1768 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 4096, !dbg !40
  %186 = load <4 x half>, ptr addrspace(3) %kdma.gep1768, align 8, !dbg !40
  %kdma.gep1769 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 128, !dbg !40
  %187 = load <4 x half>, ptr addrspace(3) %kdma.gep1769, align 8, !dbg !40
  %188 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %186, <4 x float> %185) #9, !dbg !40
  %189 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %55, <4 x half> %187, <4 x float> zeroinitializer) #9, !dbg !40
  %kdma.gep1770 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4224, !dbg !40
  %190 = load <4 x half>, ptr addrspace(3) %kdma.gep1770, align 8, !dbg !40
  %kdma.gep1771 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 128, !dbg !40
  %191 = load <4 x half>, ptr addrspace(3) %kdma.gep1771, align 8, !dbg !40
  %192 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %190, <4 x float> %189) #9, !dbg !40
  %193 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %55, <4 x half> %191, <4 x float> zeroinitializer) #9, !dbg !40
  %kdma.gep1772 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 4224, !dbg !40
  %194 = load <4 x half>, ptr addrspace(3) %kdma.gep1772, align 8, !dbg !40
  %kdma.gep1773 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1024, !dbg !40
  %195 = load <4 x half>, ptr addrspace(3) %kdma.gep1773, align 8, !dbg !40
  %196 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %194, <4 x float> %193) #9, !dbg !40
  %197 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %55, <4 x half> %195, <4 x float> zeroinitializer) #9, !dbg !40
  %kdma.gep1774 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5120, !dbg !40
  %198 = load <4 x half>, ptr addrspace(3) %kdma.gep1774, align 8, !dbg !40
  %kdma.gep1775 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 1024, !dbg !40
  %199 = load <4 x half>, ptr addrspace(3) %kdma.gep1775, align 8, !dbg !40
  %200 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %198, <4 x float> %197) #9, !dbg !40
  %201 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %55, <4 x half> %199, <4 x float> zeroinitializer) #9, !dbg !40
  %kdma.gep1776 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 5120, !dbg !40
  %202 = load <4 x half>, ptr addrspace(3) %kdma.gep1776, align 8, !dbg !40
  %kdma.gep1777 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1152, !dbg !40
  %203 = load <4 x half>, ptr addrspace(3) %kdma.gep1777, align 8, !dbg !40
  %204 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %202, <4 x float> %201) #9, !dbg !40
  %205 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %55, <4 x half> %203, <4 x float> zeroinitializer) #9, !dbg !40
  %kdma.gep1778 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5248, !dbg !40
  %206 = load <4 x half>, ptr addrspace(3) %kdma.gep1778, align 8, !dbg !40
  %kdma.gep1779 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 1152, !dbg !40
  %207 = load <4 x half>, ptr addrspace(3) %kdma.gep1779, align 8, !dbg !40
  %208 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %206, <4 x float> %205) #9, !dbg !40
  %209 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %55, <4 x half> %207, <4 x float> zeroinitializer) #9, !dbg !40
  %kdma.gep1780 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 5248, !dbg !40
  %210 = load <4 x half>, ptr addrspace(3) %kdma.gep1780, align 8, !dbg !40
  %kdma.gep1781 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 0, !dbg !40
  %211 = load <4 x half>, ptr addrspace(3) %kdma.gep1781, align 8, !dbg !40
  %212 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %210, <4 x float> %209) #9, !dbg !40
  %213 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %211, <4 x float> zeroinitializer) #9, !dbg !40
  %kdma.gep1782 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4096, !dbg !40
  %214 = load <4 x half>, ptr addrspace(3) %kdma.gep1782, align 8, !dbg !40
  %kdma.gep1783 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 0, !dbg !40
  %215 = load <4 x half>, ptr addrspace(3) %kdma.gep1783, align 8, !dbg !40
  %216 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %214, <4 x float> %213) #9, !dbg !40
  %217 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %215, <4 x float> zeroinitializer) #9, !dbg !40
  %kdma.gep1784 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 4096, !dbg !40
  %218 = load <4 x half>, ptr addrspace(3) %kdma.gep1784, align 8, !dbg !40
  %kdma.gep1785 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 128, !dbg !40
  %219 = load <4 x half>, ptr addrspace(3) %kdma.gep1785, align 8, !dbg !40
  %220 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %218, <4 x float> %217) #9, !dbg !40
  %221 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %219, <4 x float> zeroinitializer) #9, !dbg !40
  %kdma.gep1786 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4224, !dbg !40
  %222 = load <4 x half>, ptr addrspace(3) %kdma.gep1786, align 8, !dbg !40
  %kdma.gep1787 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 128, !dbg !40
  %223 = load <4 x half>, ptr addrspace(3) %kdma.gep1787, align 8, !dbg !40
  %224 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %222, <4 x float> %221) #9, !dbg !40
  %225 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %223, <4 x float> zeroinitializer) #9, !dbg !40
  %kdma.gep1788 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 4224, !dbg !40
  %226 = load <4 x half>, ptr addrspace(3) %kdma.gep1788, align 8, !dbg !40
  %kdma.gep1789 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1024, !dbg !40
  %227 = load <4 x half>, ptr addrspace(3) %kdma.gep1789, align 8, !dbg !40
  %228 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %226, <4 x float> %225) #9, !dbg !40
  %229 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %227, <4 x float> zeroinitializer) #9, !dbg !40
  %kdma.gep1790 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5120, !dbg !40
  %230 = load <4 x half>, ptr addrspace(3) %kdma.gep1790, align 8, !dbg !40
  %kdma.gep1791 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 1024, !dbg !40
  %231 = load <4 x half>, ptr addrspace(3) %kdma.gep1791, align 8, !dbg !40
  %232 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %230, <4 x float> %229) #9, !dbg !40
  %233 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %231, <4 x float> zeroinitializer) #9, !dbg !40
  %kdma.gep1792 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 5120, !dbg !40
  %234 = load <4 x half>, ptr addrspace(3) %kdma.gep1792, align 8, !dbg !40
  %kdma.gep1793 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1152, !dbg !40
  %235 = load <4 x half>, ptr addrspace(3) %kdma.gep1793, align 8, !dbg !40
  %236 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %234, <4 x float> %233) #9, !dbg !40
  %237 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %235, <4 x float> zeroinitializer) #9, !dbg !40
  %kdma.gep1794 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5248, !dbg !40
  %238 = load <4 x half>, ptr addrspace(3) %kdma.gep1794, align 8, !dbg !40
  %kdma.gep1795 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 1152, !dbg !40
  %239 = load <4 x half>, ptr addrspace(3) %kdma.gep1795, align 8, !dbg !40
  %240 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %238, <4 x float> %237) #9, !dbg !40
  %241 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %239, <4 x float> zeroinitializer) #9, !dbg !40
  %kdma.gep1796 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 5248, !dbg !40
  %242 = load <4 x half>, ptr addrspace(3) %kdma.gep1796, align 8, !dbg !40
  %243 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %242, <4 x float> %241) #9, !dbg !40
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !38
  fence syncscope("workgroup") release, !dbg !38
  call void @llvm.amdgcn.s.barrier(), !dbg !38
  fence syncscope("workgroup") acquire, !dbg !38
  %244 = or disjoint i64 %134, 64, !dbg !38
  %245 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local504), !dbg !38
  %246 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !38
  %247 = insertelement <2 x i64> poison, i64 %246, i64 0, !dbg !38
  %248 = bitcast <2 x i64> %247 to <4 x i32>, !dbg !38
  %dma.buffer.rsrc1361 = shufflevector <4 x i32> %248, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !38
  %dma.buffer.lds1362 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %245), !dbg !38
  %kdma.source1635 = zext nneg i32 %kdma.owner to i64, !dbg !38
  %kdma.source1636 = lshr i64 %kdma.source1635, 4, !dbg !38
  %kdma.source1637 = and i64 %kdma.source1636, 8, !dbg !38
  %kdma.source1638 = and i64 %kdma.source1635, 123, !dbg !38
  %kdma.source1639 = lshr i64 %kdma.source1635, 2, !dbg !38
  %kdma.source1640 = and i64 %kdma.source1639, 4, !dbg !38
  %kdma.source1641 = add nuw nsw i64 %kdma.source1640, %kdma.source1635, !dbg !38
  %kdma.source1642 = and i64 %kdma.source1641, 4, !dbg !38
  %kdma.source1643 = or disjoint i64 %kdma.source1638, %kdma.source1642, !dbg !38
  %kdma.source1644 = shl nuw nsw i64 %kdma.source1643, 7, !dbg !38
  %kdma.source1645 = or disjoint i64 %kdma.source1637, %kdma.source1644, !dbg !38
  %kdma.source1646 = or disjoint i64 %244, %kdma.source1645, !dbg !38
  %kdma.source1647 = trunc nuw i64 %kdma.source1646 to i32, !dbg !38
  %kdma.source1648 = shl nuw nsw i32 %kdma.source1647, 1, !dbg !38
  %kdma.offset1649 = add i32 %kdma.source1648, %kdma.intra, !dbg !38
  %kdma.lds1650 = add i32 %dma.buffer.lds1362, 0, !dbg !38
  %249 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1650, i32 %kdma.offset1649, <4 x i32> %dma.buffer.rsrc1361) #10, !dbg !38
  %kdma.source1651 = zext nneg i32 %kdma.owner.hi to i64, !dbg !38
  %kdma.source1652 = lshr i64 %kdma.source1651, 4, !dbg !38
  %kdma.source1653 = and i64 %kdma.source1652, 8, !dbg !38
  %kdma.source1654 = and i64 %kdma.source1651, 123, !dbg !38
  %kdma.source1655 = lshr i64 %kdma.source1651, 2, !dbg !38
  %kdma.source1656 = and i64 %kdma.source1655, 4, !dbg !38
  %kdma.source1657 = add nuw nsw i64 %kdma.source1656, %kdma.source1651, !dbg !38
  %kdma.source1658 = and i64 %kdma.source1657, 4, !dbg !38
  %kdma.source1659 = or disjoint i64 %kdma.source1654, %kdma.source1658, !dbg !38
  %kdma.source1660 = shl nuw nsw i64 %kdma.source1659, 7, !dbg !38
  %kdma.source1661 = or disjoint i64 %kdma.source1653, %kdma.source1660, !dbg !38
  %kdma.source1662 = or disjoint i64 %244, %kdma.source1661, !dbg !38
  %kdma.source1663 = trunc nuw i64 %kdma.source1662 to i32, !dbg !38
  %kdma.source1664 = shl nuw nsw i32 %kdma.source1663, 1, !dbg !38
  %kdma.offset1665 = add i32 %kdma.source1664, %kdma.intra.hi, !dbg !38
  %kdma.lds1666 = add i32 %dma.buffer.lds1362, 512, !dbg !38
  %250 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1666, i32 %kdma.offset1665, <4 x i32> %dma.buffer.rsrc1361) #10, !dbg !38
  %251 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local521), !dbg !38
  %252 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !38
  %253 = insertelement <2 x i64> poison, i64 %252, i64 0, !dbg !38
  %254 = bitcast <2 x i64> %253 to <4 x i32>, !dbg !38
  %dma.buffer.rsrc1358 = shufflevector <4 x i32> %254, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !38
  %dma.buffer.lds1359 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %251), !dbg !38
  %kdma.source1667 = zext nneg i32 %kdma.owner to i64, !dbg !38
  %kdma.source1668 = or disjoint i64 %kdma.source1667, 256, !dbg !38
  %kdma.source1669 = lshr i64 %kdma.source1668, 4, !dbg !38
  %kdma.source1670 = and i64 %kdma.source1669, 24, !dbg !38
  %kdma.source1671 = and i64 %kdma.source1667, 123, !dbg !38
  %kdma.source1672 = lshr i64 %kdma.source1667, 2, !dbg !38
  %kdma.source1673 = and i64 %kdma.source1672, 4, !dbg !38
  %kdma.source1674 = add nuw nsw i64 %kdma.source1673, %kdma.source1667, !dbg !38
  %kdma.source1675 = and i64 %kdma.source1674, 4, !dbg !38
  %kdma.source1676 = or disjoint i64 %kdma.source1671, %kdma.source1675, !dbg !38
  %kdma.source1677 = shl nuw nsw i64 %kdma.source1676, 7, !dbg !38
  %kdma.source1678 = or disjoint i64 %kdma.source1670, %kdma.source1677, !dbg !38
  %kdma.source1679 = or disjoint i64 %244, %kdma.source1678, !dbg !38
  %kdma.source1680 = trunc nuw i64 %kdma.source1679 to i32, !dbg !38
  %kdma.source1681 = shl nuw nsw i32 %kdma.source1680, 1, !dbg !38
  %kdma.offset1682 = add i32 %kdma.source1681, %kdma.intra, !dbg !38
  %kdma.lds1683 = add i32 %dma.buffer.lds1359, 0, !dbg !38
  %255 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1683, i32 %kdma.offset1682, <4 x i32> %dma.buffer.rsrc1358) #10, !dbg !38
  %kdma.source1684 = zext nneg i32 %kdma.owner.hi to i64, !dbg !38
  %kdma.source1685 = or disjoint i64 %kdma.source1684, 256, !dbg !38
  %kdma.source1686 = lshr i64 %kdma.source1685, 4, !dbg !38
  %kdma.source1687 = and i64 %kdma.source1686, 24, !dbg !38
  %kdma.source1688 = and i64 %kdma.source1684, 123, !dbg !38
  %kdma.source1689 = lshr i64 %kdma.source1684, 2, !dbg !38
  %kdma.source1690 = and i64 %kdma.source1689, 4, !dbg !38
  %kdma.source1691 = add nuw nsw i64 %kdma.source1690, %kdma.source1684, !dbg !38
  %kdma.source1692 = and i64 %kdma.source1691, 4, !dbg !38
  %kdma.source1693 = or disjoint i64 %kdma.source1688, %kdma.source1692, !dbg !38
  %kdma.source1694 = shl nuw nsw i64 %kdma.source1693, 7, !dbg !38
  %kdma.source1695 = or disjoint i64 %kdma.source1687, %kdma.source1694, !dbg !38
  %kdma.source1696 = or disjoint i64 %244, %kdma.source1695, !dbg !38
  %kdma.source1697 = trunc nuw i64 %kdma.source1696 to i32, !dbg !38
  %kdma.source1698 = shl nuw nsw i32 %kdma.source1697, 1, !dbg !38
  %kdma.offset1699 = add i32 %kdma.source1698, %kdma.intra.hi, !dbg !38
  %kdma.lds1700 = add i32 %dma.buffer.lds1359, 512, !dbg !38
  %256 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1700, i32 %kdma.offset1699, <4 x i32> %dma.buffer.rsrc1358) #10, !dbg !38
  %kdma.gep1797 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8192, !dbg !40
  %257 = load <4 x half>, ptr addrspace(3) %kdma.gep1797, align 8, !dbg !40
  %258 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %257, <4 x float> %184) #9, !dbg !40
  %kdma.gep1798 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12288, !dbg !40
  %259 = load <4 x half>, ptr addrspace(3) %kdma.gep1798, align 8, !dbg !40
  %kdma.gep1799 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 8192, !dbg !40
  %260 = load <4 x half>, ptr addrspace(3) %kdma.gep1799, align 8, !dbg !40
  %261 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %259, <4 x float> %258) #9, !dbg !40
  %262 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %260, <4 x float> %188) #9, !dbg !40
  %kdma.gep1800 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 12288, !dbg !40
  %263 = load <4 x half>, ptr addrspace(3) %kdma.gep1800, align 8, !dbg !40
  %kdma.gep1801 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8320, !dbg !40
  %264 = load <4 x half>, ptr addrspace(3) %kdma.gep1801, align 8, !dbg !40
  %265 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %263, <4 x float> %262) #9, !dbg !40
  %266 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %264, <4 x float> %192) #9, !dbg !40
  %kdma.gep1802 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12416, !dbg !40
  %267 = load <4 x half>, ptr addrspace(3) %kdma.gep1802, align 8, !dbg !40
  %kdma.gep1803 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 8320, !dbg !40
  %268 = load <4 x half>, ptr addrspace(3) %kdma.gep1803, align 8, !dbg !40
  %269 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %267, <4 x float> %266) #9, !dbg !40
  %270 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %268, <4 x float> %196) #9, !dbg !40
  %kdma.gep1804 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 12416, !dbg !40
  %271 = load <4 x half>, ptr addrspace(3) %kdma.gep1804, align 8, !dbg !40
  %kdma.gep1805 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9216, !dbg !40
  %272 = load <4 x half>, ptr addrspace(3) %kdma.gep1805, align 8, !dbg !40
  %273 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %271, <4 x float> %270) #9, !dbg !40
  %274 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %272, <4 x float> %200) #9, !dbg !40
  %kdma.gep1806 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13312, !dbg !40
  %275 = load <4 x half>, ptr addrspace(3) %kdma.gep1806, align 8, !dbg !40
  %kdma.gep1807 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 9216, !dbg !40
  %276 = load <4 x half>, ptr addrspace(3) %kdma.gep1807, align 8, !dbg !40
  %277 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %275, <4 x float> %274) #9, !dbg !40
  %278 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %276, <4 x float> %204) #9, !dbg !40
  %kdma.gep1808 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 13312, !dbg !40
  %279 = load <4 x half>, ptr addrspace(3) %kdma.gep1808, align 8, !dbg !40
  %kdma.gep1809 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9344, !dbg !40
  %280 = load <4 x half>, ptr addrspace(3) %kdma.gep1809, align 8, !dbg !40
  %281 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %279, <4 x float> %278) #9, !dbg !40
  %282 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %280, <4 x float> %208) #9, !dbg !40
  %kdma.gep1810 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13440, !dbg !40
  %283 = load <4 x half>, ptr addrspace(3) %kdma.gep1810, align 8, !dbg !40
  %kdma.gep1811 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 9344, !dbg !40
  %284 = load <4 x half>, ptr addrspace(3) %kdma.gep1811, align 8, !dbg !40
  %285 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %283, <4 x float> %282) #9, !dbg !40
  %286 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %284, <4 x float> %212) #9, !dbg !40
  %kdma.gep1812 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 13440, !dbg !40
  %287 = load <4 x half>, ptr addrspace(3) %kdma.gep1812, align 8, !dbg !40
  %kdma.gep1813 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8192, !dbg !40
  %288 = load <4 x half>, ptr addrspace(3) %kdma.gep1813, align 8, !dbg !40
  %289 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %287, <4 x float> %286) #9, !dbg !40
  %290 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %288, <4 x float> %216) #9, !dbg !40
  %kdma.gep1814 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12288, !dbg !40
  %291 = load <4 x half>, ptr addrspace(3) %kdma.gep1814, align 8, !dbg !40
  %kdma.gep1815 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 8192, !dbg !40
  %292 = load <4 x half>, ptr addrspace(3) %kdma.gep1815, align 8, !dbg !40
  %293 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %291, <4 x float> %290) #9, !dbg !40
  %294 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %292, <4 x float> %220) #9, !dbg !40
  %kdma.gep1816 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 12288, !dbg !40
  %295 = load <4 x half>, ptr addrspace(3) %kdma.gep1816, align 8, !dbg !40
  %kdma.gep1817 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8320, !dbg !40
  %296 = load <4 x half>, ptr addrspace(3) %kdma.gep1817, align 8, !dbg !40
  %297 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %295, <4 x float> %294) #9, !dbg !40
  %298 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %296, <4 x float> %224) #9, !dbg !40
  %kdma.gep1818 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12416, !dbg !40
  %299 = load <4 x half>, ptr addrspace(3) %kdma.gep1818, align 8, !dbg !40
  %kdma.gep1819 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 8320, !dbg !40
  %300 = load <4 x half>, ptr addrspace(3) %kdma.gep1819, align 8, !dbg !40
  %301 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %299, <4 x float> %298) #9, !dbg !40
  %302 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %300, <4 x float> %228) #9, !dbg !40
  %kdma.gep1820 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 12416, !dbg !40
  %303 = load <4 x half>, ptr addrspace(3) %kdma.gep1820, align 8, !dbg !40
  %kdma.gep1821 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9216, !dbg !40
  %304 = load <4 x half>, ptr addrspace(3) %kdma.gep1821, align 8, !dbg !40
  %305 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %303, <4 x float> %302) #9, !dbg !40
  %306 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %304, <4 x float> %232) #9, !dbg !40
  %kdma.gep1822 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13312, !dbg !40
  %307 = load <4 x half>, ptr addrspace(3) %kdma.gep1822, align 8, !dbg !40
  %kdma.gep1823 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 9216, !dbg !40
  %308 = load <4 x half>, ptr addrspace(3) %kdma.gep1823, align 8, !dbg !40
  %309 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %307, <4 x float> %306) #9, !dbg !40
  %310 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %308, <4 x float> %236) #9, !dbg !40
  %kdma.gep1824 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 13312, !dbg !40
  %311 = load <4 x half>, ptr addrspace(3) %kdma.gep1824, align 8, !dbg !40
  %kdma.gep1825 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9344, !dbg !40
  %312 = load <4 x half>, ptr addrspace(3) %kdma.gep1825, align 8, !dbg !40
  %313 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %311, <4 x float> %310) #9, !dbg !40
  %314 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %312, <4 x float> %240) #9, !dbg !40
  %kdma.gep1826 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13440, !dbg !40
  %315 = load <4 x half>, ptr addrspace(3) %kdma.gep1826, align 8, !dbg !40
  %kdma.gep1827 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 9344, !dbg !40
  %316 = load <4 x half>, ptr addrspace(3) %kdma.gep1827, align 8, !dbg !40
  %317 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %315, <4 x float> %314) #9, !dbg !40
  %318 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %316, <4 x float> %243) #9, !dbg !40
  %kdma.gep1828 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 13440, !dbg !40
  %319 = load <4 x half>, ptr addrspace(3) %kdma.gep1828, align 8, !dbg !40
  %320 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %319, <4 x float> %318) #9, !dbg !40
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !38
  fence syncscope("workgroup") release, !dbg !38
  call void @llvm.amdgcn.s.barrier(), !dbg !38
  fence syncscope("workgroup") acquire, !dbg !38
  %321 = or disjoint i64 %134, 96, !dbg !38
  %322 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local535), !dbg !38
  %323 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !38
  %324 = insertelement <2 x i64> poison, i64 %323, i64 0, !dbg !38
  %325 = bitcast <2 x i64> %324 to <4 x i32>, !dbg !38
  %dma.buffer.rsrc1355 = shufflevector <4 x i32> %325, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !38
  %dma.buffer.lds1356 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %322), !dbg !38
  %kdma.source1701 = zext nneg i32 %kdma.owner to i64, !dbg !38
  %kdma.source1702 = lshr i64 %kdma.source1701, 4, !dbg !38
  %kdma.source1703 = and i64 %kdma.source1702, 8, !dbg !38
  %kdma.source1704 = and i64 %kdma.source1701, 123, !dbg !38
  %kdma.source1705 = lshr i64 %kdma.source1701, 2, !dbg !38
  %kdma.source1706 = and i64 %kdma.source1705, 4, !dbg !38
  %kdma.source1707 = add nuw nsw i64 %kdma.source1706, %kdma.source1701, !dbg !38
  %kdma.source1708 = and i64 %kdma.source1707, 4, !dbg !38
  %kdma.source1709 = or disjoint i64 %kdma.source1704, %kdma.source1708, !dbg !38
  %kdma.source1710 = shl nuw nsw i64 %kdma.source1709, 7, !dbg !38
  %kdma.source1711 = or disjoint i64 %kdma.source1703, %kdma.source1710, !dbg !38
  %kdma.source1712 = or disjoint i64 %321, %kdma.source1711, !dbg !38
  %kdma.source1713 = trunc nuw i64 %kdma.source1712 to i32, !dbg !38
  %kdma.source1714 = shl nuw nsw i32 %kdma.source1713, 1, !dbg !38
  %kdma.offset1715 = add i32 %kdma.source1714, %kdma.intra, !dbg !38
  %kdma.lds1716 = add i32 %dma.buffer.lds1356, 0, !dbg !38
  %326 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1716, i32 %kdma.offset1715, <4 x i32> %dma.buffer.rsrc1355) #10, !dbg !38
  %kdma.source1717 = zext nneg i32 %kdma.owner.hi to i64, !dbg !38
  %kdma.source1718 = lshr i64 %kdma.source1717, 4, !dbg !38
  %kdma.source1719 = and i64 %kdma.source1718, 8, !dbg !38
  %kdma.source1720 = and i64 %kdma.source1717, 123, !dbg !38
  %kdma.source1721 = lshr i64 %kdma.source1717, 2, !dbg !38
  %kdma.source1722 = and i64 %kdma.source1721, 4, !dbg !38
  %kdma.source1723 = add nuw nsw i64 %kdma.source1722, %kdma.source1717, !dbg !38
  %kdma.source1724 = and i64 %kdma.source1723, 4, !dbg !38
  %kdma.source1725 = or disjoint i64 %kdma.source1720, %kdma.source1724, !dbg !38
  %kdma.source1726 = shl nuw nsw i64 %kdma.source1725, 7, !dbg !38
  %kdma.source1727 = or disjoint i64 %kdma.source1719, %kdma.source1726, !dbg !38
  %kdma.source1728 = or disjoint i64 %321, %kdma.source1727, !dbg !38
  %kdma.source1729 = trunc nuw i64 %kdma.source1728 to i32, !dbg !38
  %kdma.source1730 = shl nuw nsw i32 %kdma.source1729, 1, !dbg !38
  %kdma.offset1731 = add i32 %kdma.source1730, %kdma.intra.hi, !dbg !38
  %kdma.lds1732 = add i32 %dma.buffer.lds1356, 512, !dbg !38
  %327 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1732, i32 %kdma.offset1731, <4 x i32> %dma.buffer.rsrc1355) #10, !dbg !38
  %328 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local539), !dbg !38
  %329 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !38
  %330 = insertelement <2 x i64> poison, i64 %329, i64 0, !dbg !38
  %331 = bitcast <2 x i64> %330 to <4 x i32>, !dbg !38
  %dma.buffer.rsrc1352 = shufflevector <4 x i32> %331, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !38
  %dma.buffer.lds1353 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %328), !dbg !38
  %kdma.source1733 = zext nneg i32 %kdma.owner to i64, !dbg !38
  %kdma.source1734 = or disjoint i64 %kdma.source1733, 256, !dbg !38
  %kdma.source1735 = lshr i64 %kdma.source1734, 4, !dbg !38
  %kdma.source1736 = and i64 %kdma.source1735, 24, !dbg !38
  %kdma.source1737 = and i64 %kdma.source1733, 123, !dbg !38
  %kdma.source1738 = lshr i64 %kdma.source1733, 2, !dbg !38
  %kdma.source1739 = and i64 %kdma.source1738, 4, !dbg !38
  %kdma.source1740 = add nuw nsw i64 %kdma.source1739, %kdma.source1733, !dbg !38
  %kdma.source1741 = and i64 %kdma.source1740, 4, !dbg !38
  %kdma.source1742 = or disjoint i64 %kdma.source1737, %kdma.source1741, !dbg !38
  %kdma.source1743 = shl nuw nsw i64 %kdma.source1742, 7, !dbg !38
  %kdma.source1744 = or disjoint i64 %kdma.source1736, %kdma.source1743, !dbg !38
  %kdma.source1745 = or disjoint i64 %321, %kdma.source1744, !dbg !38
  %kdma.source1746 = trunc nuw i64 %kdma.source1745 to i32, !dbg !38
  %kdma.source1747 = shl nuw nsw i32 %kdma.source1746, 1, !dbg !38
  %kdma.offset1748 = add i32 %kdma.source1747, %kdma.intra, !dbg !38
  %kdma.lds1749 = add i32 %dma.buffer.lds1353, 0, !dbg !38
  %332 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1749, i32 %kdma.offset1748, <4 x i32> %dma.buffer.rsrc1352) #10, !dbg !38
  %kdma.source1750 = zext nneg i32 %kdma.owner.hi to i64, !dbg !38
  %kdma.source1751 = or disjoint i64 %kdma.source1750, 256, !dbg !38
  %kdma.source1752 = lshr i64 %kdma.source1751, 4, !dbg !38
  %kdma.source1753 = and i64 %kdma.source1752, 24, !dbg !38
  %kdma.source1754 = and i64 %kdma.source1750, 123, !dbg !38
  %kdma.source1755 = lshr i64 %kdma.source1750, 2, !dbg !38
  %kdma.source1756 = and i64 %kdma.source1755, 4, !dbg !38
  %kdma.source1757 = add nuw nsw i64 %kdma.source1756, %kdma.source1750, !dbg !38
  %kdma.source1758 = and i64 %kdma.source1757, 4, !dbg !38
  %kdma.source1759 = or disjoint i64 %kdma.source1754, %kdma.source1758, !dbg !38
  %kdma.source1760 = shl nuw nsw i64 %kdma.source1759, 7, !dbg !38
  %kdma.source1761 = or disjoint i64 %kdma.source1753, %kdma.source1760, !dbg !38
  %kdma.source1762 = or disjoint i64 %321, %kdma.source1761, !dbg !38
  %kdma.source1763 = trunc nuw i64 %kdma.source1762 to i32, !dbg !38
  %kdma.source1764 = shl nuw nsw i32 %kdma.source1763, 1, !dbg !38
  %kdma.offset1765 = add i32 %kdma.source1764, %kdma.intra.hi, !dbg !38
  %kdma.lds1766 = add i32 %dma.buffer.lds1353, 512, !dbg !38
  %333 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1766, i32 %kdma.offset1765, <4 x i32> %dma.buffer.rsrc1352) #10, !dbg !38
  %kdma.gep1829 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 0, !dbg !40
  %334 = load <4 x half>, ptr addrspace(3) %kdma.gep1829, align 8, !dbg !40
  %335 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %334, <4 x float> %261) #9, !dbg !40
  %kdma.gep1830 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4096, !dbg !40
  %336 = load <4 x half>, ptr addrspace(3) %kdma.gep1830, align 8, !dbg !40
  %kdma.gep1831 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 0, !dbg !40
  %337 = load <4 x half>, ptr addrspace(3) %kdma.gep1831, align 8, !dbg !40
  %338 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %336, <4 x float> %335) #9, !dbg !40
  %339 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %337, <4 x float> %265) #9, !dbg !40
  %kdma.gep1832 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 4096, !dbg !40
  %340 = load <4 x half>, ptr addrspace(3) %kdma.gep1832, align 8, !dbg !40
  %kdma.gep1833 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 128, !dbg !40
  %341 = load <4 x half>, ptr addrspace(3) %kdma.gep1833, align 8, !dbg !40
  %342 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %340, <4 x float> %339) #9, !dbg !40
  %343 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %341, <4 x float> %269) #9, !dbg !40
  %kdma.gep1834 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4224, !dbg !40
  %344 = load <4 x half>, ptr addrspace(3) %kdma.gep1834, align 8, !dbg !40
  %kdma.gep1835 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 128, !dbg !40
  %345 = load <4 x half>, ptr addrspace(3) %kdma.gep1835, align 8, !dbg !40
  %346 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %344, <4 x float> %343) #9, !dbg !40
  %347 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %345, <4 x float> %273) #9, !dbg !40
  %kdma.gep1836 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 4224, !dbg !40
  %348 = load <4 x half>, ptr addrspace(3) %kdma.gep1836, align 8, !dbg !40
  %kdma.gep1837 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1024, !dbg !40
  %349 = load <4 x half>, ptr addrspace(3) %kdma.gep1837, align 8, !dbg !40
  %350 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %348, <4 x float> %347) #9, !dbg !40
  %351 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %349, <4 x float> %277) #9, !dbg !40
  %kdma.gep1838 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5120, !dbg !40
  %352 = load <4 x half>, ptr addrspace(3) %kdma.gep1838, align 8, !dbg !40
  %kdma.gep1839 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 1024, !dbg !40
  %353 = load <4 x half>, ptr addrspace(3) %kdma.gep1839, align 8, !dbg !40
  %354 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %352, <4 x float> %351) #9, !dbg !40
  %355 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %353, <4 x float> %281) #9, !dbg !40
  %kdma.gep1840 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 5120, !dbg !40
  %356 = load <4 x half>, ptr addrspace(3) %kdma.gep1840, align 8, !dbg !40
  %kdma.gep1841 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1152, !dbg !40
  %357 = load <4 x half>, ptr addrspace(3) %kdma.gep1841, align 8, !dbg !40
  %358 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %356, <4 x float> %355) #9, !dbg !40
  %359 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %357, <4 x float> %285) #9, !dbg !40
  %kdma.gep1842 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5248, !dbg !40
  %360 = load <4 x half>, ptr addrspace(3) %kdma.gep1842, align 8, !dbg !40
  %kdma.gep1843 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 1152, !dbg !40
  %361 = load <4 x half>, ptr addrspace(3) %kdma.gep1843, align 8, !dbg !40
  %362 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %360, <4 x float> %359) #9, !dbg !40
  %363 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %361, <4 x float> %289) #9, !dbg !40
  %kdma.gep1844 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 5248, !dbg !40
  %364 = load <4 x half>, ptr addrspace(3) %kdma.gep1844, align 8, !dbg !40
  %kdma.gep1845 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 0, !dbg !40
  %365 = load <4 x half>, ptr addrspace(3) %kdma.gep1845, align 8, !dbg !40
  %366 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %364, <4 x float> %363) #9, !dbg !40
  %367 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %365, <4 x float> %293) #9, !dbg !40
  %kdma.gep1846 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4096, !dbg !40
  %368 = load <4 x half>, ptr addrspace(3) %kdma.gep1846, align 8, !dbg !40
  %kdma.gep1847 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 0, !dbg !40
  %369 = load <4 x half>, ptr addrspace(3) %kdma.gep1847, align 8, !dbg !40
  %370 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %368, <4 x float> %367) #9, !dbg !40
  %371 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %369, <4 x float> %297) #9, !dbg !40
  %kdma.gep1848 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 4096, !dbg !40
  %372 = load <4 x half>, ptr addrspace(3) %kdma.gep1848, align 8, !dbg !40
  %kdma.gep1849 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 128, !dbg !40
  %373 = load <4 x half>, ptr addrspace(3) %kdma.gep1849, align 8, !dbg !40
  %374 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %372, <4 x float> %371) #9, !dbg !40
  %375 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %373, <4 x float> %301) #9, !dbg !40
  %kdma.gep1850 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4224, !dbg !40
  %376 = load <4 x half>, ptr addrspace(3) %kdma.gep1850, align 8, !dbg !40
  %kdma.gep1851 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 128, !dbg !40
  %377 = load <4 x half>, ptr addrspace(3) %kdma.gep1851, align 8, !dbg !40
  %378 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %376, <4 x float> %375) #9, !dbg !40
  %379 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %377, <4 x float> %305) #9, !dbg !40
  %kdma.gep1852 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 4224, !dbg !40
  %380 = load <4 x half>, ptr addrspace(3) %kdma.gep1852, align 8, !dbg !40
  %kdma.gep1853 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1024, !dbg !40
  %381 = load <4 x half>, ptr addrspace(3) %kdma.gep1853, align 8, !dbg !40
  %382 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %380, <4 x float> %379) #9, !dbg !40
  %383 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %381, <4 x float> %309) #9, !dbg !40
  %kdma.gep1854 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5120, !dbg !40
  %384 = load <4 x half>, ptr addrspace(3) %kdma.gep1854, align 8, !dbg !40
  %kdma.gep1855 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 1024, !dbg !40
  %385 = load <4 x half>, ptr addrspace(3) %kdma.gep1855, align 8, !dbg !40
  %386 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %384, <4 x float> %383) #9, !dbg !40
  %387 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %385, <4 x float> %313) #9, !dbg !40
  %kdma.gep1856 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 5120, !dbg !40
  %388 = load <4 x half>, ptr addrspace(3) %kdma.gep1856, align 8, !dbg !40
  %kdma.gep1857 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1152, !dbg !40
  %389 = load <4 x half>, ptr addrspace(3) %kdma.gep1857, align 8, !dbg !40
  %390 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %388, <4 x float> %387) #9, !dbg !40
  %391 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %389, <4 x float> %317) #9, !dbg !40
  %kdma.gep1858 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5248, !dbg !40
  %392 = load <4 x half>, ptr addrspace(3) %kdma.gep1858, align 8, !dbg !40
  %kdma.gep1859 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 1152, !dbg !40
  %393 = load <4 x half>, ptr addrspace(3) %kdma.gep1859, align 8, !dbg !40
  %394 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %392, <4 x float> %391) #9, !dbg !40
  %395 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %393, <4 x float> %320) #9, !dbg !40
  %kdma.gep1860 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 5248, !dbg !40
  %396 = load <4 x half>, ptr addrspace(3) %kdma.gep1860, align 8, !dbg !40
  %397 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %396, <4 x float> %395) #9, !dbg !40
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !38
  fence syncscope("workgroup") release, !dbg !38
  call void @llvm.amdgcn.s.barrier(), !dbg !38
  fence syncscope("workgroup") acquire, !dbg !38
  %398 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local504), !dbg !41
  %address.local917 = shl nuw nsw i64 %address.local499, 5, !dbg !41
  %address.local918 = and i64 %address.local917, 8064, !dbg !41
  %address.local919 = shl nuw nsw i64 %address.local499, 3, !dbg !41
  %address.local920 = and i64 %address.local919, 24, !dbg !41
  %address.local921 = or disjoint i64 %address.local918, %address.local920, !dbg !41
  %399 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !41
  %.idx14191420 = or disjoint i64 %134, %address.local921, !dbg !41
  %.idx14191420.tr = trunc nuw i64 %.idx14191420 to i32, !dbg !41
  %dma.buffer.offset1348 = shl nuw nsw i32 %.idx14191420.tr, 1, !dbg !41
  %400 = insertelement <2 x i64> poison, i64 %399, i64 0, !dbg !41
  %401 = bitcast <2 x i64> %400 to <4 x i32>, !dbg !41
  %dma.buffer.rsrc1349 = shufflevector <4 x i32> %401, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !41
  %dma.buffer.lds1350 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %398), !dbg !41
  %402 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1350, i32 %dma.buffer.offset1348, <4 x i32> %dma.buffer.rsrc1349) #10, !dbg !41
  %403 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local521), !dbg !41
  %address.local923 = shl nuw nsw i64 %address.local516, 5, !dbg !41
  %address.local924 = and i64 %address.local923, 16256, !dbg !41
  %address.local925 = shl nuw nsw i64 %address.local499, 3, !dbg !41
  %address.local926 = and i64 %address.local925, 24, !dbg !41
  %address.local927 = or disjoint i64 %address.local924, %address.local926, !dbg !41
  %404 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !41
  %.idx14211422 = or disjoint i64 %134, %address.local927, !dbg !41
  %.idx14211422.tr = trunc nuw i64 %.idx14211422 to i32, !dbg !41
  %dma.buffer.offset1345 = shl nuw nsw i32 %.idx14211422.tr, 1, !dbg !41
  %405 = insertelement <2 x i64> poison, i64 %404, i64 0, !dbg !41
  %406 = bitcast <2 x i64> %405 to <4 x i32>, !dbg !41
  %dma.buffer.rsrc1346 = shufflevector <4 x i32> %406, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !41
  %dma.buffer.lds1347 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %403), !dbg !41
  %407 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1347, i32 %dma.buffer.offset1345, <4 x i32> %dma.buffer.rsrc1346) #10, !dbg !41
  %kdma.gep1861 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8192, !dbg !40
  %408 = load <4 x half>, ptr addrspace(3) %kdma.gep1861, align 8, !dbg !40
  %409 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %108, <4 x half> %408, <4 x float> %338) #9, !dbg !40
  %kdma.gep1862 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12288, !dbg !40
  %410 = load <4 x half>, ptr addrspace(3) %kdma.gep1862, align 8, !dbg !40
  %kdma.gep1863 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 8192, !dbg !40
  %411 = load <4 x half>, ptr addrspace(3) %kdma.gep1863, align 8, !dbg !40
  %412 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %110, <4 x half> %410, <4 x float> %409) #9, !dbg !40
  %413 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %108, <4 x half> %411, <4 x float> %342) #9, !dbg !40
  %414 = extractelement <4 x float> %412, i64 0, !dbg !45
  %415 = call float @llvm.maxnum.f32(float %414, float -inf), !dbg !46, !frisk.local_reduction !30
  %416 = extractelement <4 x float> %412, i64 1, !dbg !45
  %417 = extractelement <4 x float> %412, i64 2, !dbg !45
  %418 = extractelement <4 x float> %412, i64 3, !dbg !45
  %kdma.gep1864 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 12288, !dbg !40
  %419 = load <4 x half>, ptr addrspace(3) %kdma.gep1864, align 8, !dbg !40
  %kdma.gep1865 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8320, !dbg !40
  %420 = load <4 x half>, ptr addrspace(3) %kdma.gep1865, align 8, !dbg !40
  %421 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %110, <4 x half> %419, <4 x float> %413) #9, !dbg !40
  %422 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %108, <4 x half> %420, <4 x float> %346) #9, !dbg !40
  %423 = extractelement <4 x float> %421, i64 0, !dbg !45
  %local.reduce151 = call float @llvm.maxnum.f32(float %415, float %423), !dbg !46, !frisk.local_reduction !30
  %424 = extractelement <4 x float> %421, i64 1, !dbg !45
  %local.reduce152 = call float @llvm.maxnum.f32(float %416, float %424), !dbg !46, !frisk.local_reduction !30
  %425 = extractelement <4 x float> %421, i64 2, !dbg !45
  %local.reduce153 = call float @llvm.maxnum.f32(float %417, float %425), !dbg !46, !frisk.local_reduction !30
  %426 = extractelement <4 x float> %421, i64 3, !dbg !45
  %local.reduce154 = call float @llvm.maxnum.f32(float %418, float %426), !dbg !46, !frisk.local_reduction !30
  %kdma.gep1866 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12416, !dbg !40
  %427 = load <4 x half>, ptr addrspace(3) %kdma.gep1866, align 8, !dbg !40
  %kdma.gep1867 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 8320, !dbg !40
  %428 = load <4 x half>, ptr addrspace(3) %kdma.gep1867, align 8, !dbg !40
  %429 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %110, <4 x half> %427, <4 x float> %422) #9, !dbg !40
  %430 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %108, <4 x half> %428, <4 x float> %350) #9, !dbg !40
  %431 = extractelement <4 x float> %429, i64 0, !dbg !45
  %local.reduce155 = call float @llvm.maxnum.f32(float %local.reduce151, float %431), !dbg !46, !frisk.local_reduction !30
  %432 = extractelement <4 x float> %429, i64 1, !dbg !45
  %local.reduce156 = call float @llvm.maxnum.f32(float %local.reduce152, float %432), !dbg !46, !frisk.local_reduction !30
  %433 = extractelement <4 x float> %429, i64 2, !dbg !45
  %local.reduce157 = call float @llvm.maxnum.f32(float %local.reduce153, float %433), !dbg !46, !frisk.local_reduction !30
  %434 = extractelement <4 x float> %429, i64 3, !dbg !45
  %local.reduce158 = call float @llvm.maxnum.f32(float %local.reduce154, float %434), !dbg !46, !frisk.local_reduction !30
  %kdma.gep1868 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 12416, !dbg !40
  %435 = load <4 x half>, ptr addrspace(3) %kdma.gep1868, align 8, !dbg !40
  %kdma.gep1869 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9216, !dbg !40
  %436 = load <4 x half>, ptr addrspace(3) %kdma.gep1869, align 8, !dbg !40
  %437 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %110, <4 x half> %435, <4 x float> %430) #9, !dbg !40
  %438 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %108, <4 x half> %436, <4 x float> %354) #9, !dbg !40
  %439 = extractelement <4 x float> %437, i64 0, !dbg !45
  %local.reduce159 = call float @llvm.maxnum.f32(float %local.reduce155, float %439), !dbg !46, !frisk.local_reduction !30
  %440 = extractelement <4 x float> %437, i64 1, !dbg !45
  %local.reduce160 = call float @llvm.maxnum.f32(float %local.reduce156, float %440), !dbg !46, !frisk.local_reduction !30
  %441 = extractelement <4 x float> %437, i64 2, !dbg !45
  %local.reduce161 = call float @llvm.maxnum.f32(float %local.reduce157, float %441), !dbg !46, !frisk.local_reduction !30
  %442 = extractelement <4 x float> %437, i64 3, !dbg !45
  %local.reduce162 = call float @llvm.maxnum.f32(float %local.reduce158, float %442), !dbg !46, !frisk.local_reduction !30
  %kdma.gep1870 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13312, !dbg !40
  %443 = load <4 x half>, ptr addrspace(3) %kdma.gep1870, align 8, !dbg !40
  %kdma.gep1871 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 9216, !dbg !40
  %444 = load <4 x half>, ptr addrspace(3) %kdma.gep1871, align 8, !dbg !40
  %445 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %110, <4 x half> %443, <4 x float> %438) #9, !dbg !40
  %446 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %108, <4 x half> %444, <4 x float> %358) #9, !dbg !40
  %447 = extractelement <4 x float> %445, i64 0, !dbg !45
  %local.reduce163 = call float @llvm.maxnum.f32(float %local.reduce159, float %447), !dbg !46, !frisk.local_reduction !30
  %448 = extractelement <4 x float> %445, i64 1, !dbg !45
  %local.reduce164 = call float @llvm.maxnum.f32(float %local.reduce160, float %448), !dbg !46, !frisk.local_reduction !30
  %449 = extractelement <4 x float> %445, i64 2, !dbg !45
  %local.reduce165 = call float @llvm.maxnum.f32(float %local.reduce161, float %449), !dbg !46, !frisk.local_reduction !30
  %450 = extractelement <4 x float> %445, i64 3, !dbg !45
  %local.reduce166 = call float @llvm.maxnum.f32(float %local.reduce162, float %450), !dbg !46, !frisk.local_reduction !30
  %kdma.gep1872 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 13312, !dbg !40
  %451 = load <4 x half>, ptr addrspace(3) %kdma.gep1872, align 8, !dbg !40
  %kdma.gep1873 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9344, !dbg !40
  %452 = load <4 x half>, ptr addrspace(3) %kdma.gep1873, align 8, !dbg !40
  %453 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %110, <4 x half> %451, <4 x float> %446) #9, !dbg !40
  %454 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %108, <4 x half> %452, <4 x float> %362) #9, !dbg !40
  %455 = extractelement <4 x float> %453, i64 0, !dbg !45
  %local.reduce167 = call float @llvm.maxnum.f32(float %local.reduce163, float %455), !dbg !46, !frisk.local_reduction !30
  %456 = extractelement <4 x float> %453, i64 1, !dbg !45
  %local.reduce168 = call float @llvm.maxnum.f32(float %local.reduce164, float %456), !dbg !46, !frisk.local_reduction !30
  %457 = extractelement <4 x float> %453, i64 2, !dbg !45
  %local.reduce169 = call float @llvm.maxnum.f32(float %local.reduce165, float %457), !dbg !46, !frisk.local_reduction !30
  %458 = extractelement <4 x float> %453, i64 3, !dbg !45
  %local.reduce170 = call float @llvm.maxnum.f32(float %local.reduce166, float %458), !dbg !46, !frisk.local_reduction !30
  %kdma.gep1874 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13440, !dbg !40
  %459 = load <4 x half>, ptr addrspace(3) %kdma.gep1874, align 8, !dbg !40
  %kdma.gep1875 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 9344, !dbg !40
  %460 = load <4 x half>, ptr addrspace(3) %kdma.gep1875, align 8, !dbg !40
  %461 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %110, <4 x half> %459, <4 x float> %454) #9, !dbg !40
  %462 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %108, <4 x half> %460, <4 x float> %366) #9, !dbg !40
  %463 = extractelement <4 x float> %461, i64 0, !dbg !45
  %local.reduce171 = call float @llvm.maxnum.f32(float %local.reduce167, float %463), !dbg !46, !frisk.local_reduction !30
  %464 = extractelement <4 x float> %461, i64 1, !dbg !45
  %local.reduce172 = call float @llvm.maxnum.f32(float %local.reduce168, float %464), !dbg !46, !frisk.local_reduction !30
  %465 = extractelement <4 x float> %461, i64 2, !dbg !45
  %local.reduce173 = call float @llvm.maxnum.f32(float %local.reduce169, float %465), !dbg !46, !frisk.local_reduction !30
  %466 = extractelement <4 x float> %461, i64 3, !dbg !45
  %local.reduce174 = call float @llvm.maxnum.f32(float %local.reduce170, float %466), !dbg !46, !frisk.local_reduction !30
  %kdma.gep1876 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 13440, !dbg !40
  %467 = load <4 x half>, ptr addrspace(3) %kdma.gep1876, align 8, !dbg !40
  %kdma.gep1877 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8192, !dbg !40
  %468 = load <4 x half>, ptr addrspace(3) %kdma.gep1877, align 8, !dbg !40
  %469 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %110, <4 x half> %467, <4 x float> %462) #9, !dbg !40
  %470 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %111, <4 x half> %468, <4 x float> %370) #9, !dbg !40
  %471 = extractelement <4 x float> %469, i64 0, !dbg !45
  %local.reduce175 = call float @llvm.maxnum.f32(float %local.reduce171, float %471), !dbg !46, !frisk.local_reduction !30
  %472 = extractelement <4 x float> %469, i64 1, !dbg !45
  %local.reduce176 = call float @llvm.maxnum.f32(float %local.reduce172, float %472), !dbg !46, !frisk.local_reduction !30
  %473 = extractelement <4 x float> %469, i64 2, !dbg !45
  %local.reduce177 = call float @llvm.maxnum.f32(float %local.reduce173, float %473), !dbg !46, !frisk.local_reduction !30
  %474 = extractelement <4 x float> %469, i64 3, !dbg !45
  %local.reduce178 = call float @llvm.maxnum.f32(float %local.reduce174, float %474), !dbg !46, !frisk.local_reduction !30
  %local.reduce179 = call float @llvm.maxnum.f32(float %local.reduce175, float %local.reduce176), !dbg !46, !frisk.local_reduction !30
  %local.reduce180 = call float @llvm.maxnum.f32(float %local.reduce177, float %local.reduce178), !dbg !46, !frisk.local_reduction !30
  %475 = call float @llvm.maxnum.f32(float %local.reduce179, float %local.reduce180), !dbg !46, !frisk.local_reduction !30
  %kdma.gep1878 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12288, !dbg !40
  %476 = load <4 x half>, ptr addrspace(3) %kdma.gep1878, align 8, !dbg !40
  %kdma.gep1879 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 8192, !dbg !40
  %477 = load <4 x half>, ptr addrspace(3) %kdma.gep1879, align 8, !dbg !40
  %478 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %112, <4 x half> %476, <4 x float> %470) #9, !dbg !40
  %479 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %111, <4 x half> %477, <4 x float> %374) #9, !dbg !40
  %480 = extractelement <4 x float> %478, i64 0, !dbg !45
  %481 = extractelement <4 x float> %478, i64 1, !dbg !45
  %482 = extractelement <4 x float> %478, i64 2, !dbg !45
  %483 = extractelement <4 x float> %478, i64 3, !dbg !45
  %kdma.gep1880 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 12288, !dbg !40
  %484 = load <4 x half>, ptr addrspace(3) %kdma.gep1880, align 8, !dbg !40
  %kdma.gep1881 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8320, !dbg !40
  %485 = load <4 x half>, ptr addrspace(3) %kdma.gep1881, align 8, !dbg !40
  %486 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %112, <4 x half> %484, <4 x float> %479) #9, !dbg !40
  %487 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %111, <4 x half> %485, <4 x float> %378) #9, !dbg !40
  %488 = extractelement <4 x float> %486, i64 0, !dbg !45
  %489 = extractelement <4 x float> %486, i64 1, !dbg !45
  %490 = extractelement <4 x float> %486, i64 2, !dbg !45
  %491 = extractelement <4 x float> %486, i64 3, !dbg !45
  %kdma.gep1882 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12416, !dbg !40
  %492 = load <4 x half>, ptr addrspace(3) %kdma.gep1882, align 8, !dbg !40
  %kdma.gep1883 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 8320, !dbg !40
  %493 = load <4 x half>, ptr addrspace(3) %kdma.gep1883, align 8, !dbg !40
  %494 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %112, <4 x half> %492, <4 x float> %487) #9, !dbg !40
  %495 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %111, <4 x half> %493, <4 x float> %382) #9, !dbg !40
  %496 = extractelement <4 x float> %494, i64 0, !dbg !45
  %497 = extractelement <4 x float> %494, i64 1, !dbg !45
  %498 = extractelement <4 x float> %494, i64 2, !dbg !45
  %499 = extractelement <4 x float> %494, i64 3, !dbg !45
  %kdma.gep1884 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 12416, !dbg !40
  %500 = load <4 x half>, ptr addrspace(3) %kdma.gep1884, align 8, !dbg !40
  %kdma.gep1885 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9216, !dbg !40
  %501 = load <4 x half>, ptr addrspace(3) %kdma.gep1885, align 8, !dbg !40
  %502 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %112, <4 x half> %500, <4 x float> %495) #9, !dbg !40
  %503 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %111, <4 x half> %501, <4 x float> %386) #9, !dbg !40
  %504 = extractelement <4 x float> %502, i64 0, !dbg !45
  %505 = extractelement <4 x float> %502, i64 1, !dbg !45
  %506 = extractelement <4 x float> %502, i64 2, !dbg !45
  %507 = extractelement <4 x float> %502, i64 3, !dbg !45
  %kdma.gep1886 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13312, !dbg !40
  %508 = load <4 x half>, ptr addrspace(3) %kdma.gep1886, align 8, !dbg !40
  %kdma.gep1887 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 9216, !dbg !40
  %509 = load <4 x half>, ptr addrspace(3) %kdma.gep1887, align 8, !dbg !40
  %510 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %112, <4 x half> %508, <4 x float> %503) #9, !dbg !40
  %511 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %111, <4 x half> %509, <4 x float> %390) #9, !dbg !40
  %512 = extractelement <4 x float> %510, i64 0, !dbg !45
  %513 = extractelement <4 x float> %510, i64 1, !dbg !45
  %514 = extractelement <4 x float> %510, i64 2, !dbg !45
  %515 = extractelement <4 x float> %510, i64 3, !dbg !45
  %kdma.gep1888 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 13312, !dbg !40
  %516 = load <4 x half>, ptr addrspace(3) %kdma.gep1888, align 8, !dbg !40
  %kdma.gep1889 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9344, !dbg !40
  %517 = load <4 x half>, ptr addrspace(3) %kdma.gep1889, align 8, !dbg !40
  %518 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %112, <4 x half> %516, <4 x float> %511) #9, !dbg !40
  %519 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %111, <4 x half> %517, <4 x float> %394) #9, !dbg !40
  %520 = extractelement <4 x float> %518, i64 0, !dbg !45
  %521 = extractelement <4 x float> %518, i64 1, !dbg !45
  %522 = extractelement <4 x float> %518, i64 2, !dbg !45
  %523 = extractelement <4 x float> %518, i64 3, !dbg !45
  %kdma.gep1890 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13440, !dbg !40
  %524 = load <4 x half>, ptr addrspace(3) %kdma.gep1890, align 8, !dbg !40
  %kdma.gep1891 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 9344, !dbg !40
  %525 = load <4 x half>, ptr addrspace(3) %kdma.gep1891, align 8, !dbg !40
  %526 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %112, <4 x half> %524, <4 x float> %519) #9, !dbg !40
  %527 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %111, <4 x half> %525, <4 x float> %397) #9, !dbg !40
  %528 = extractelement <4 x float> %526, i64 0, !dbg !45
  %529 = extractelement <4 x float> %526, i64 1, !dbg !45
  %530 = extractelement <4 x float> %526, i64 2, !dbg !45
  %531 = extractelement <4 x float> %526, i64 3, !dbg !45
  %kdma.gep1892 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 13440, !dbg !40
  %532 = load <4 x half>, ptr addrspace(3) %kdma.gep1892, align 8, !dbg !40
  %533 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %112, <4 x half> %532, <4 x float> %527) #9, !dbg !40
  %534 = extractelement <4 x float> %533, i64 0, !dbg !45
  %535 = extractelement <4 x float> %533, i64 1, !dbg !45
  %536 = extractelement <4 x float> %533, i64 2, !dbg !45
  %537 = extractelement <4 x float> %533, i64 3, !dbg !45
  %538 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !46
  %539 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %538), !dbg !46
  %540 = add nuw nsw i32 %539, 64, !dbg !46
  %541 = xor i32 %539, 16, !dbg !46
  %542 = and i32 %540, 192, !dbg !46
  %543 = icmp samesign ult i32 %541, %542, !dbg !46
  %544 = select i1 %543, i32 %541, i32 %539, !dbg !46
  %545 = shl nuw nsw i32 %544, 2, !dbg !46
  %546 = bitcast float %475 to i32, !dbg !46
  %547 = call i32 @llvm.amdgcn.ds.bpermute(i32 %545, i32 %546), !dbg !46
  %548 = bitcast i32 %547 to float, !dbg !46
  %549 = call float @llvm.maxnum.f32(float %475, float %548), !dbg !46
  %550 = xor i32 %539, 32, !dbg !46
  %551 = icmp samesign ult i32 %550, %542, !dbg !46
  %552 = select i1 %551, i32 %550, i32 %539, !dbg !46
  %553 = shl nuw nsw i32 %552, 2, !dbg !46
  %554 = bitcast float %549 to i32, !dbg !46
  %555 = call i32 @llvm.amdgcn.ds.bpermute(i32 %553, i32 %554), !dbg !46
  %556 = bitcast i32 %555 to float, !dbg !46
  %557 = call float @llvm.maxnum.f32(float %549, float %556), !dbg !46
  %558 = extractvalue [2 x <1 x float>] %125, 0, !dbg !46
  %559 = insertelement <1 x float> poison, float %557, i64 0, !dbg !46
  %560 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %559, <1 x float> %558), !dbg !46
  %561 = call float @llvm.maxnum.f32(float %480, float -inf), !dbg !46, !frisk.local_reduction !30
  %local.reduce120 = call float @llvm.maxnum.f32(float %561, float %488), !dbg !46, !frisk.local_reduction !30
  %local.reduce121 = call float @llvm.maxnum.f32(float %481, float %489), !dbg !46, !frisk.local_reduction !30
  %local.reduce122 = call float @llvm.maxnum.f32(float %482, float %490), !dbg !46, !frisk.local_reduction !30
  %local.reduce123 = call float @llvm.maxnum.f32(float %483, float %491), !dbg !46, !frisk.local_reduction !30
  %local.reduce124 = call float @llvm.maxnum.f32(float %local.reduce120, float %496), !dbg !46, !frisk.local_reduction !30
  %local.reduce125 = call float @llvm.maxnum.f32(float %local.reduce121, float %497), !dbg !46, !frisk.local_reduction !30
  %local.reduce126 = call float @llvm.maxnum.f32(float %local.reduce122, float %498), !dbg !46, !frisk.local_reduction !30
  %local.reduce127 = call float @llvm.maxnum.f32(float %local.reduce123, float %499), !dbg !46, !frisk.local_reduction !30
  %local.reduce128 = call float @llvm.maxnum.f32(float %local.reduce124, float %504), !dbg !46, !frisk.local_reduction !30
  %local.reduce129 = call float @llvm.maxnum.f32(float %local.reduce125, float %505), !dbg !46, !frisk.local_reduction !30
  %local.reduce130 = call float @llvm.maxnum.f32(float %local.reduce126, float %506), !dbg !46, !frisk.local_reduction !30
  %local.reduce131 = call float @llvm.maxnum.f32(float %local.reduce127, float %507), !dbg !46, !frisk.local_reduction !30
  %local.reduce132 = call float @llvm.maxnum.f32(float %local.reduce128, float %512), !dbg !46, !frisk.local_reduction !30
  %local.reduce133 = call float @llvm.maxnum.f32(float %local.reduce129, float %513), !dbg !46, !frisk.local_reduction !30
  %local.reduce134 = call float @llvm.maxnum.f32(float %local.reduce130, float %514), !dbg !46, !frisk.local_reduction !30
  %local.reduce135 = call float @llvm.maxnum.f32(float %local.reduce131, float %515), !dbg !46, !frisk.local_reduction !30
  %local.reduce136 = call float @llvm.maxnum.f32(float %local.reduce132, float %520), !dbg !46, !frisk.local_reduction !30
  %local.reduce137 = call float @llvm.maxnum.f32(float %local.reduce133, float %521), !dbg !46, !frisk.local_reduction !30
  %local.reduce138 = call float @llvm.maxnum.f32(float %local.reduce134, float %522), !dbg !46, !frisk.local_reduction !30
  %local.reduce139 = call float @llvm.maxnum.f32(float %local.reduce135, float %523), !dbg !46, !frisk.local_reduction !30
  %local.reduce140 = call float @llvm.maxnum.f32(float %local.reduce136, float %528), !dbg !46, !frisk.local_reduction !30
  %local.reduce141 = call float @llvm.maxnum.f32(float %local.reduce137, float %529), !dbg !46, !frisk.local_reduction !30
  %local.reduce142 = call float @llvm.maxnum.f32(float %local.reduce138, float %530), !dbg !46, !frisk.local_reduction !30
  %local.reduce143 = call float @llvm.maxnum.f32(float %local.reduce139, float %531), !dbg !46, !frisk.local_reduction !30
  %local.reduce144 = call float @llvm.maxnum.f32(float %local.reduce140, float %534), !dbg !46, !frisk.local_reduction !30
  %local.reduce145 = call float @llvm.maxnum.f32(float %local.reduce141, float %535), !dbg !46, !frisk.local_reduction !30
  %local.reduce146 = call float @llvm.maxnum.f32(float %local.reduce142, float %536), !dbg !46, !frisk.local_reduction !30
  %local.reduce147 = call float @llvm.maxnum.f32(float %local.reduce143, float %537), !dbg !46, !frisk.local_reduction !30
  %local.reduce148 = call float @llvm.maxnum.f32(float %local.reduce144, float %local.reduce145), !dbg !46, !frisk.local_reduction !30
  %local.reduce149 = call float @llvm.maxnum.f32(float %local.reduce146, float %local.reduce147), !dbg !46, !frisk.local_reduction !30
  %562 = call float @llvm.maxnum.f32(float %local.reduce148, float %local.reduce149), !dbg !46, !frisk.local_reduction !30
  %563 = bitcast float %562 to i32, !dbg !46
  %564 = call i32 @llvm.amdgcn.ds.bpermute(i32 %545, i32 %563), !dbg !46
  %565 = bitcast i32 %564 to float, !dbg !46
  %566 = call float @llvm.maxnum.f32(float %562, float %565), !dbg !46
  %567 = bitcast float %566 to i32, !dbg !46
  %568 = call i32 @llvm.amdgcn.ds.bpermute(i32 %553, i32 %567), !dbg !46
  %569 = bitcast i32 %568 to float, !dbg !46
  %570 = call float @llvm.maxnum.f32(float %566, float %569), !dbg !46
  %571 = extractvalue [2 x <1 x float>] %125, 1, !dbg !46
  %572 = insertelement <1 x float> poison, float %570, i64 0, !dbg !46
  %573 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %572, <1 x float> %571), !dbg !46
  %574 = shufflevector <1 x float> %560, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !47
  %575 = fsub <4 x float> %412, %574, !dbg !47
  %576 = fsub <4 x float> %421, %574, !dbg !47
  %577 = fsub <4 x float> %429, %574, !dbg !47
  %578 = fsub <4 x float> %437, %574, !dbg !47
  %579 = fsub <4 x float> %445, %574, !dbg !47
  %580 = fsub <4 x float> %453, %574, !dbg !47
  %581 = fsub <4 x float> %461, %574, !dbg !47
  %582 = fsub <4 x float> %469, %574, !dbg !47
  %583 = shufflevector <1 x float> %573, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !47
  %584 = fsub <4 x float> %478, %583, !dbg !47
  %585 = fsub <4 x float> %486, %583, !dbg !47
  %586 = fsub <4 x float> %494, %583, !dbg !47
  %587 = fsub <4 x float> %502, %583, !dbg !47
  %588 = fsub <4 x float> %510, %583, !dbg !47
  %589 = fsub <4 x float> %518, %583, !dbg !47
  %590 = fsub <4 x float> %526, %583, !dbg !47
  %591 = fsub <4 x float> %533, %583, !dbg !47
  %592 = fsub <1 x float> %558, %560, !dbg !48
  %593 = extractelement <1 x float> %592, i64 0, !dbg !49
  %594 = call float @llvm.amdgcn.exp2.f32(float %593), !dbg !49
  %595 = insertelement <4 x float> poison, float %594, i64 0, !dbg !54
  %596 = shufflevector <4 x float> %595, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !54
  %597 = extractvalue [2 x <1 x float>] %124, 0, !dbg !50
  %598 = extractelement <1 x float> %597, i64 0, !dbg !50
  %599 = fmul float %598, %594, !dbg !50
  %600 = fsub <1 x float> %571, %573, !dbg !48
  %601 = extractelement <1 x float> %600, i64 0, !dbg !49
  %602 = call float @llvm.amdgcn.exp2.f32(float %601), !dbg !49
  %603 = insertelement <4 x float> poison, float %602, i64 0, !dbg !54
  %604 = shufflevector <4 x float> %603, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !54
  %605 = extractvalue [2 x <1 x float>] %124, 1, !dbg !50
  %606 = extractelement <1 x float> %605, i64 0, !dbg !50
  %607 = fmul float %606, %602, !dbg !50
  %608 = extractelement <4 x float> %575, i64 0, !dbg !51
  %609 = call float @llvm.amdgcn.exp2.f32(float %608), !dbg !51
  %610 = insertelement <4 x float> poison, float %609, i64 0, !dbg !53
  %611 = extractelement <4 x float> %575, i64 1, !dbg !51
  %612 = call float @llvm.amdgcn.exp2.f32(float %611), !dbg !51
  %613 = insertelement <4 x float> %610, float %612, i64 1, !dbg !53
  %614 = extractelement <4 x float> %575, i64 2, !dbg !51
  %615 = call float @llvm.amdgcn.exp2.f32(float %614), !dbg !51
  %616 = insertelement <4 x float> %613, float %615, i64 2, !dbg !53
  %617 = extractelement <4 x float> %575, i64 3, !dbg !51
  %618 = call float @llvm.amdgcn.exp2.f32(float %617), !dbg !51
  %619 = insertelement <4 x float> %616, float %618, i64 3, !dbg !53
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %619) #9, !dbg !53, !frisk.register_lifetimes !30
  %620 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !53
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %620) #9, !dbg !51, !frisk.register_lifetimes !30
  %621 = extractelement <4 x float> %576, i64 0, !dbg !51
  %622 = call float @llvm.amdgcn.exp2.f32(float %621), !dbg !51
  %623 = insertelement <4 x float> poison, float %622, i64 0, !dbg !53
  %local.reduce89 = fadd float %609, %622, !dbg !52, !frisk.local_reduction !30
  %624 = extractelement <4 x float> %576, i64 1, !dbg !51
  %625 = call float @llvm.amdgcn.exp2.f32(float %624), !dbg !51
  %626 = insertelement <4 x float> %623, float %625, i64 1, !dbg !53
  %local.reduce90 = fadd float %612, %625, !dbg !52, !frisk.local_reduction !30
  %627 = extractelement <4 x float> %576, i64 2, !dbg !51
  %628 = call float @llvm.amdgcn.exp2.f32(float %627), !dbg !51
  %629 = insertelement <4 x float> %626, float %628, i64 2, !dbg !53
  %local.reduce91 = fadd float %615, %628, !dbg !52, !frisk.local_reduction !30
  %630 = extractelement <4 x float> %576, i64 3, !dbg !51
  %631 = call float @llvm.amdgcn.exp2.f32(float %630), !dbg !51
  %632 = insertelement <4 x float> %629, float %631, i64 3, !dbg !53
  %math.result374 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %632) #9, !dbg !53, !frisk.register_lifetimes !30
  %633 = fptrunc <4 x float> %math.result374 to <4 x half>, !dbg !53
  %math.packed375 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %633) #9, !dbg !52, !frisk.register_lifetimes !30
  %local.reduce92 = fadd float %618, %631, !dbg !52, !frisk.local_reduction !30
  %634 = extractelement <4 x float> %577, i64 0, !dbg !51
  %635 = call float @llvm.amdgcn.exp2.f32(float %634), !dbg !51
  %636 = insertelement <4 x float> poison, float %635, i64 0, !dbg !53
  %local.reduce93 = fadd float %local.reduce89, %635, !dbg !52, !frisk.local_reduction !30
  %637 = extractelement <4 x float> %577, i64 1, !dbg !51
  %638 = call float @llvm.amdgcn.exp2.f32(float %637), !dbg !51
  %639 = insertelement <4 x float> %636, float %638, i64 1, !dbg !53
  %local.reduce94 = fadd float %local.reduce90, %638, !dbg !52, !frisk.local_reduction !30
  %640 = extractelement <4 x float> %577, i64 2, !dbg !51
  %641 = call float @llvm.amdgcn.exp2.f32(float %640), !dbg !51
  %642 = insertelement <4 x float> %639, float %641, i64 2, !dbg !53
  %local.reduce95 = fadd float %local.reduce91, %641, !dbg !52, !frisk.local_reduction !30
  %643 = extractelement <4 x float> %577, i64 3, !dbg !51
  %644 = call float @llvm.amdgcn.exp2.f32(float %643), !dbg !51
  %645 = insertelement <4 x float> %642, float %644, i64 3, !dbg !53
  %math.result376 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %645) #9, !dbg !53, !frisk.register_lifetimes !30
  %646 = fptrunc <4 x float> %math.result376 to <4 x half>, !dbg !53
  %math.packed377 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %646) #9, !dbg !52, !frisk.register_lifetimes !30
  %local.reduce96 = fadd float %local.reduce92, %644, !dbg !52, !frisk.local_reduction !30
  %647 = extractelement <4 x float> %578, i64 0, !dbg !51
  %648 = call float @llvm.amdgcn.exp2.f32(float %647), !dbg !51
  %649 = insertelement <4 x float> poison, float %648, i64 0, !dbg !53
  %local.reduce97 = fadd float %local.reduce93, %648, !dbg !52, !frisk.local_reduction !30
  %650 = extractelement <4 x float> %578, i64 1, !dbg !51
  %651 = call float @llvm.amdgcn.exp2.f32(float %650), !dbg !51
  %652 = insertelement <4 x float> %649, float %651, i64 1, !dbg !53
  %local.reduce98 = fadd float %local.reduce94, %651, !dbg !52, !frisk.local_reduction !30
  %653 = extractelement <4 x float> %578, i64 2, !dbg !51
  %654 = call float @llvm.amdgcn.exp2.f32(float %653), !dbg !51
  %655 = insertelement <4 x float> %652, float %654, i64 2, !dbg !53
  %local.reduce99 = fadd float %local.reduce95, %654, !dbg !52, !frisk.local_reduction !30
  %656 = extractelement <4 x float> %578, i64 3, !dbg !51
  %657 = call float @llvm.amdgcn.exp2.f32(float %656), !dbg !51
  %658 = insertelement <4 x float> %655, float %657, i64 3, !dbg !53
  %math.result378 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %658) #9, !dbg !53, !frisk.register_lifetimes !30
  %659 = fptrunc <4 x float> %math.result378 to <4 x half>, !dbg !53
  %math.packed379 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %659) #9, !dbg !52, !frisk.register_lifetimes !30
  %local.reduce100 = fadd float %local.reduce96, %657, !dbg !52, !frisk.local_reduction !30
  %660 = extractelement <4 x float> %579, i64 0, !dbg !51
  %661 = call float @llvm.amdgcn.exp2.f32(float %660), !dbg !51
  %662 = insertelement <4 x float> poison, float %661, i64 0, !dbg !53
  %local.reduce101 = fadd float %local.reduce97, %661, !dbg !52, !frisk.local_reduction !30
  %663 = extractelement <4 x float> %579, i64 1, !dbg !51
  %664 = call float @llvm.amdgcn.exp2.f32(float %663), !dbg !51
  %665 = insertelement <4 x float> %662, float %664, i64 1, !dbg !53
  %local.reduce102 = fadd float %local.reduce98, %664, !dbg !52, !frisk.local_reduction !30
  %666 = extractelement <4 x float> %579, i64 2, !dbg !51
  %667 = call float @llvm.amdgcn.exp2.f32(float %666), !dbg !51
  %668 = insertelement <4 x float> %665, float %667, i64 2, !dbg !53
  %local.reduce103 = fadd float %local.reduce99, %667, !dbg !52, !frisk.local_reduction !30
  %669 = extractelement <4 x float> %579, i64 3, !dbg !51
  %670 = call float @llvm.amdgcn.exp2.f32(float %669), !dbg !51
  %671 = insertelement <4 x float> %668, float %670, i64 3, !dbg !53
  %math.result380 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %671) #9, !dbg !53, !frisk.register_lifetimes !30
  %672 = fptrunc <4 x float> %math.result380 to <4 x half>, !dbg !53
  %math.packed381 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %672) #9, !dbg !52, !frisk.register_lifetimes !30
  %local.reduce104 = fadd float %local.reduce100, %670, !dbg !52, !frisk.local_reduction !30
  %673 = extractelement <4 x float> %580, i64 0, !dbg !51
  %674 = call float @llvm.amdgcn.exp2.f32(float %673), !dbg !51
  %675 = insertelement <4 x float> poison, float %674, i64 0, !dbg !53
  %local.reduce105 = fadd float %local.reduce101, %674, !dbg !52, !frisk.local_reduction !30
  %676 = extractelement <4 x float> %580, i64 1, !dbg !51
  %677 = call float @llvm.amdgcn.exp2.f32(float %676), !dbg !51
  %678 = insertelement <4 x float> %675, float %677, i64 1, !dbg !53
  %local.reduce106 = fadd float %local.reduce102, %677, !dbg !52, !frisk.local_reduction !30
  %679 = extractelement <4 x float> %580, i64 2, !dbg !51
  %680 = call float @llvm.amdgcn.exp2.f32(float %679), !dbg !51
  %681 = insertelement <4 x float> %678, float %680, i64 2, !dbg !53
  %local.reduce107 = fadd float %local.reduce103, %680, !dbg !52, !frisk.local_reduction !30
  %682 = extractelement <4 x float> %580, i64 3, !dbg !51
  %683 = call float @llvm.amdgcn.exp2.f32(float %682), !dbg !51
  %684 = insertelement <4 x float> %681, float %683, i64 3, !dbg !53
  %math.result382 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %684) #9, !dbg !53, !frisk.register_lifetimes !30
  %685 = fptrunc <4 x float> %math.result382 to <4 x half>, !dbg !53
  %math.packed383 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %685) #9, !dbg !52, !frisk.register_lifetimes !30
  %local.reduce108 = fadd float %local.reduce104, %683, !dbg !52, !frisk.local_reduction !30
  %686 = extractelement <4 x float> %581, i64 0, !dbg !51
  %687 = call float @llvm.amdgcn.exp2.f32(float %686), !dbg !51
  %688 = insertelement <4 x float> poison, float %687, i64 0, !dbg !53
  %local.reduce109 = fadd float %local.reduce105, %687, !dbg !52, !frisk.local_reduction !30
  %689 = extractelement <4 x float> %581, i64 1, !dbg !51
  %690 = call float @llvm.amdgcn.exp2.f32(float %689), !dbg !51
  %691 = insertelement <4 x float> %688, float %690, i64 1, !dbg !53
  %local.reduce110 = fadd float %local.reduce106, %690, !dbg !52, !frisk.local_reduction !30
  %692 = extractelement <4 x float> %581, i64 2, !dbg !51
  %693 = call float @llvm.amdgcn.exp2.f32(float %692), !dbg !51
  %694 = insertelement <4 x float> %691, float %693, i64 2, !dbg !53
  %local.reduce111 = fadd float %local.reduce107, %693, !dbg !52, !frisk.local_reduction !30
  %695 = extractelement <4 x float> %581, i64 3, !dbg !51
  %696 = call float @llvm.amdgcn.exp2.f32(float %695), !dbg !51
  %697 = insertelement <4 x float> %694, float %696, i64 3, !dbg !53
  %math.result384 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %697) #9, !dbg !53, !frisk.register_lifetimes !30
  %698 = fptrunc <4 x float> %math.result384 to <4 x half>, !dbg !53
  %math.packed385 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %698) #9, !dbg !52, !frisk.register_lifetimes !30
  %local.reduce112 = fadd float %local.reduce108, %696, !dbg !52, !frisk.local_reduction !30
  %699 = extractelement <4 x float> %582, i64 0, !dbg !51
  %700 = call float @llvm.amdgcn.exp2.f32(float %699), !dbg !51
  %701 = insertelement <4 x float> poison, float %700, i64 0, !dbg !53
  %local.reduce113 = fadd float %local.reduce109, %700, !dbg !52, !frisk.local_reduction !30
  %702 = extractelement <4 x float> %582, i64 1, !dbg !51
  %703 = call float @llvm.amdgcn.exp2.f32(float %702), !dbg !51
  %704 = insertelement <4 x float> %701, float %703, i64 1, !dbg !53
  %local.reduce114 = fadd float %local.reduce110, %703, !dbg !52, !frisk.local_reduction !30
  %local.reduce117 = fadd float %local.reduce113, %local.reduce114, !dbg !52, !frisk.local_reduction !30
  %705 = extractelement <4 x float> %582, i64 2, !dbg !51
  %706 = call float @llvm.amdgcn.exp2.f32(float %705), !dbg !51
  %707 = insertelement <4 x float> %704, float %706, i64 2, !dbg !53
  %local.reduce115 = fadd float %local.reduce111, %706, !dbg !52, !frisk.local_reduction !30
  %708 = extractelement <4 x float> %582, i64 3, !dbg !51
  %709 = call float @llvm.amdgcn.exp2.f32(float %708), !dbg !51
  %710 = insertelement <4 x float> %707, float %709, i64 3, !dbg !53
  %math.result386 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %710) #9, !dbg !53, !frisk.register_lifetimes !30
  %711 = fptrunc <4 x float> %math.result386 to <4 x half>, !dbg !53
  %math.packed387 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %711) #9, !dbg !52, !frisk.register_lifetimes !30
  %local.reduce116 = fadd float %local.reduce112, %709, !dbg !52, !frisk.local_reduction !30
  %local.reduce118 = fadd float %local.reduce115, %local.reduce116, !dbg !52, !frisk.local_reduction !30
  %712 = fadd float %local.reduce117, %local.reduce118, !dbg !52, !frisk.local_reduction !30
  %713 = fadd float %712, %599, !dbg !52
  %714 = insertelement <1 x float> poison, float %713, i64 0, !dbg !35
  %715 = insertvalue [2 x <1 x float>] poison, <1 x float> %714, 0, !dbg !35
  %716 = extractelement <4 x float> %584, i64 0, !dbg !51
  %717 = call float @llvm.amdgcn.exp2.f32(float %716), !dbg !51
  %718 = insertelement <4 x float> poison, float %717, i64 0, !dbg !53
  %719 = extractelement <4 x float> %584, i64 1, !dbg !51
  %720 = call float @llvm.amdgcn.exp2.f32(float %719), !dbg !51
  %721 = insertelement <4 x float> %718, float %720, i64 1, !dbg !53
  %722 = extractelement <4 x float> %584, i64 2, !dbg !51
  %723 = call float @llvm.amdgcn.exp2.f32(float %722), !dbg !51
  %724 = insertelement <4 x float> %721, float %723, i64 2, !dbg !53
  %725 = extractelement <4 x float> %584, i64 3, !dbg !51
  %726 = call float @llvm.amdgcn.exp2.f32(float %725), !dbg !51
  %727 = insertelement <4 x float> %724, float %726, i64 3, !dbg !53
  %math.result388 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %727) #9, !dbg !53, !frisk.register_lifetimes !30
  %728 = fptrunc <4 x float> %math.result388 to <4 x half>, !dbg !53
  %math.packed389 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %728) #9, !dbg !51, !frisk.register_lifetimes !30
  %729 = extractelement <4 x float> %585, i64 0, !dbg !51
  %730 = call float @llvm.amdgcn.exp2.f32(float %729), !dbg !51
  %731 = insertelement <4 x float> poison, float %730, i64 0, !dbg !53
  %local.reduce = fadd float %717, %730, !dbg !52, !frisk.local_reduction !30
  %732 = extractelement <4 x float> %585, i64 1, !dbg !51
  %733 = call float @llvm.amdgcn.exp2.f32(float %732), !dbg !51
  %734 = insertelement <4 x float> %731, float %733, i64 1, !dbg !53
  %local.reduce59 = fadd float %720, %733, !dbg !52, !frisk.local_reduction !30
  %735 = extractelement <4 x float> %585, i64 2, !dbg !51
  %736 = call float @llvm.amdgcn.exp2.f32(float %735), !dbg !51
  %737 = insertelement <4 x float> %734, float %736, i64 2, !dbg !53
  %local.reduce60 = fadd float %723, %736, !dbg !52, !frisk.local_reduction !30
  %738 = extractelement <4 x float> %585, i64 3, !dbg !51
  %739 = call float @llvm.amdgcn.exp2.f32(float %738), !dbg !51
  %740 = insertelement <4 x float> %737, float %739, i64 3, !dbg !53
  %math.result390 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %740) #9, !dbg !53, !frisk.register_lifetimes !30
  %741 = fptrunc <4 x float> %math.result390 to <4 x half>, !dbg !53
  %math.packed391 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %741) #9, !dbg !52, !frisk.register_lifetimes !30
  %local.reduce61 = fadd float %726, %739, !dbg !52, !frisk.local_reduction !30
  %742 = extractelement <4 x float> %586, i64 0, !dbg !51
  %743 = call float @llvm.amdgcn.exp2.f32(float %742), !dbg !51
  %744 = insertelement <4 x float> poison, float %743, i64 0, !dbg !53
  %local.reduce62 = fadd float %local.reduce, %743, !dbg !52, !frisk.local_reduction !30
  %745 = extractelement <4 x float> %586, i64 1, !dbg !51
  %746 = call float @llvm.amdgcn.exp2.f32(float %745), !dbg !51
  %747 = insertelement <4 x float> %744, float %746, i64 1, !dbg !53
  %local.reduce63 = fadd float %local.reduce59, %746, !dbg !52, !frisk.local_reduction !30
  %748 = extractelement <4 x float> %586, i64 2, !dbg !51
  %749 = call float @llvm.amdgcn.exp2.f32(float %748), !dbg !51
  %750 = insertelement <4 x float> %747, float %749, i64 2, !dbg !53
  %local.reduce64 = fadd float %local.reduce60, %749, !dbg !52, !frisk.local_reduction !30
  %751 = extractelement <4 x float> %586, i64 3, !dbg !51
  %752 = call float @llvm.amdgcn.exp2.f32(float %751), !dbg !51
  %753 = insertelement <4 x float> %750, float %752, i64 3, !dbg !53
  %math.result392 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %753) #9, !dbg !53, !frisk.register_lifetimes !30
  %754 = fptrunc <4 x float> %math.result392 to <4 x half>, !dbg !53
  %math.packed393 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %754) #9, !dbg !52, !frisk.register_lifetimes !30
  %local.reduce65 = fadd float %local.reduce61, %752, !dbg !52, !frisk.local_reduction !30
  %755 = extractelement <4 x float> %587, i64 0, !dbg !51
  %756 = call float @llvm.amdgcn.exp2.f32(float %755), !dbg !51
  %757 = insertelement <4 x float> poison, float %756, i64 0, !dbg !53
  %local.reduce66 = fadd float %local.reduce62, %756, !dbg !52, !frisk.local_reduction !30
  %758 = extractelement <4 x float> %587, i64 1, !dbg !51
  %759 = call float @llvm.amdgcn.exp2.f32(float %758), !dbg !51
  %760 = insertelement <4 x float> %757, float %759, i64 1, !dbg !53
  %local.reduce67 = fadd float %local.reduce63, %759, !dbg !52, !frisk.local_reduction !30
  %761 = extractelement <4 x float> %587, i64 2, !dbg !51
  %762 = call float @llvm.amdgcn.exp2.f32(float %761), !dbg !51
  %763 = insertelement <4 x float> %760, float %762, i64 2, !dbg !53
  %local.reduce68 = fadd float %local.reduce64, %762, !dbg !52, !frisk.local_reduction !30
  %764 = extractelement <4 x float> %587, i64 3, !dbg !51
  %765 = call float @llvm.amdgcn.exp2.f32(float %764), !dbg !51
  %766 = insertelement <4 x float> %763, float %765, i64 3, !dbg !53
  %math.result394 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %766) #9, !dbg !53, !frisk.register_lifetimes !30
  %767 = fptrunc <4 x float> %math.result394 to <4 x half>, !dbg !53
  %math.packed395 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %767) #9, !dbg !52, !frisk.register_lifetimes !30
  %local.reduce69 = fadd float %local.reduce65, %765, !dbg !52, !frisk.local_reduction !30
  %768 = extractelement <4 x float> %588, i64 0, !dbg !51
  %769 = call float @llvm.amdgcn.exp2.f32(float %768), !dbg !51
  %770 = insertelement <4 x float> poison, float %769, i64 0, !dbg !53
  %local.reduce70 = fadd float %local.reduce66, %769, !dbg !52, !frisk.local_reduction !30
  %771 = extractelement <4 x float> %588, i64 1, !dbg !51
  %772 = call float @llvm.amdgcn.exp2.f32(float %771), !dbg !51
  %773 = insertelement <4 x float> %770, float %772, i64 1, !dbg !53
  %local.reduce71 = fadd float %local.reduce67, %772, !dbg !52, !frisk.local_reduction !30
  %774 = extractelement <4 x float> %588, i64 2, !dbg !51
  %775 = call float @llvm.amdgcn.exp2.f32(float %774), !dbg !51
  %776 = insertelement <4 x float> %773, float %775, i64 2, !dbg !53
  %local.reduce72 = fadd float %local.reduce68, %775, !dbg !52, !frisk.local_reduction !30
  %777 = extractelement <4 x float> %588, i64 3, !dbg !51
  %778 = call float @llvm.amdgcn.exp2.f32(float %777), !dbg !51
  %779 = insertelement <4 x float> %776, float %778, i64 3, !dbg !53
  %math.result396 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %779) #9, !dbg !53, !frisk.register_lifetimes !30
  %780 = fptrunc <4 x float> %math.result396 to <4 x half>, !dbg !53
  %math.packed397 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %780) #9, !dbg !52, !frisk.register_lifetimes !30
  %local.reduce73 = fadd float %local.reduce69, %778, !dbg !52, !frisk.local_reduction !30
  %781 = extractelement <4 x float> %589, i64 0, !dbg !51
  %782 = call float @llvm.amdgcn.exp2.f32(float %781), !dbg !51
  %783 = insertelement <4 x float> poison, float %782, i64 0, !dbg !53
  %local.reduce74 = fadd float %local.reduce70, %782, !dbg !52, !frisk.local_reduction !30
  %784 = extractelement <4 x float> %589, i64 1, !dbg !51
  %785 = call float @llvm.amdgcn.exp2.f32(float %784), !dbg !51
  %786 = insertelement <4 x float> %783, float %785, i64 1, !dbg !53
  %local.reduce75 = fadd float %local.reduce71, %785, !dbg !52, !frisk.local_reduction !30
  %787 = extractelement <4 x float> %589, i64 2, !dbg !51
  %788 = call float @llvm.amdgcn.exp2.f32(float %787), !dbg !51
  %789 = insertelement <4 x float> %786, float %788, i64 2, !dbg !53
  %local.reduce76 = fadd float %local.reduce72, %788, !dbg !52, !frisk.local_reduction !30
  %790 = extractelement <4 x float> %589, i64 3, !dbg !51
  %791 = call float @llvm.amdgcn.exp2.f32(float %790), !dbg !51
  %792 = insertelement <4 x float> %789, float %791, i64 3, !dbg !53
  %math.result398 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %792) #9, !dbg !53, !frisk.register_lifetimes !30
  %793 = fptrunc <4 x float> %math.result398 to <4 x half>, !dbg !53
  %math.packed399 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %793) #9, !dbg !52, !frisk.register_lifetimes !30
  %local.reduce77 = fadd float %local.reduce73, %791, !dbg !52, !frisk.local_reduction !30
  %794 = extractelement <4 x float> %590, i64 0, !dbg !51
  %795 = call float @llvm.amdgcn.exp2.f32(float %794), !dbg !51
  %796 = insertelement <4 x float> poison, float %795, i64 0, !dbg !53
  %local.reduce78 = fadd float %local.reduce74, %795, !dbg !52, !frisk.local_reduction !30
  %797 = extractelement <4 x float> %590, i64 1, !dbg !51
  %798 = call float @llvm.amdgcn.exp2.f32(float %797), !dbg !51
  %799 = insertelement <4 x float> %796, float %798, i64 1, !dbg !53
  %local.reduce79 = fadd float %local.reduce75, %798, !dbg !52, !frisk.local_reduction !30
  %800 = extractelement <4 x float> %590, i64 2, !dbg !51
  %801 = call float @llvm.amdgcn.exp2.f32(float %800), !dbg !51
  %802 = insertelement <4 x float> %799, float %801, i64 2, !dbg !53
  %local.reduce80 = fadd float %local.reduce76, %801, !dbg !52, !frisk.local_reduction !30
  %803 = extractelement <4 x float> %590, i64 3, !dbg !51
  %804 = call float @llvm.amdgcn.exp2.f32(float %803), !dbg !51
  %805 = insertelement <4 x float> %802, float %804, i64 3, !dbg !53
  %math.result400 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %805) #9, !dbg !53, !frisk.register_lifetimes !30
  %806 = fptrunc <4 x float> %math.result400 to <4 x half>, !dbg !53
  %math.packed401 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %806) #9, !dbg !52, !frisk.register_lifetimes !30
  %local.reduce81 = fadd float %local.reduce77, %804, !dbg !52, !frisk.local_reduction !30
  %807 = extractelement <4 x float> %591, i64 0, !dbg !51
  %808 = call float @llvm.amdgcn.exp2.f32(float %807), !dbg !51
  %809 = insertelement <4 x float> poison, float %808, i64 0, !dbg !53
  %local.reduce82 = fadd float %local.reduce78, %808, !dbg !52, !frisk.local_reduction !30
  %810 = extractelement <4 x float> %591, i64 1, !dbg !51
  %811 = call float @llvm.amdgcn.exp2.f32(float %810), !dbg !51
  %812 = insertelement <4 x float> %809, float %811, i64 1, !dbg !53
  %local.reduce83 = fadd float %local.reduce79, %811, !dbg !52, !frisk.local_reduction !30
  %local.reduce86 = fadd float %local.reduce82, %local.reduce83, !dbg !52, !frisk.local_reduction !30
  %813 = extractelement <4 x float> %591, i64 2, !dbg !51
  %814 = call float @llvm.amdgcn.exp2.f32(float %813), !dbg !51
  %815 = insertelement <4 x float> %812, float %814, i64 2, !dbg !53
  %local.reduce84 = fadd float %local.reduce80, %814, !dbg !52, !frisk.local_reduction !30
  %816 = extractelement <4 x float> %591, i64 3, !dbg !51
  %817 = call float @llvm.amdgcn.exp2.f32(float %816), !dbg !51
  %818 = insertelement <4 x float> %815, float %817, i64 3, !dbg !53
  %math.result402 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %818) #9, !dbg !53, !frisk.register_lifetimes !30
  %819 = fptrunc <4 x float> %math.result402 to <4 x half>, !dbg !53
  %math.packed403 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %819) #9, !dbg !52, !frisk.register_lifetimes !30
  %local.reduce85 = fadd float %local.reduce81, %817, !dbg !52, !frisk.local_reduction !30
  %local.reduce87 = fadd float %local.reduce84, %local.reduce85, !dbg !52, !frisk.local_reduction !30
  %820 = fadd float %local.reduce86, %local.reduce87, !dbg !52, !frisk.local_reduction !30
  %821 = fadd float %820, %607, !dbg !52
  %822 = insertelement <1 x float> poison, float %821, i64 0, !dbg !35
  %823 = insertvalue [2 x <1 x float>] %715, <1 x float> %822, 1, !dbg !35
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !41
  fence syncscope("workgroup") release, !dbg !41
  call void @llvm.amdgcn.s.barrier(), !dbg !41
  fence syncscope("workgroup") acquire, !dbg !41
  %vdb.address = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local504), !dbg !41
  %vdb.address1435 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %vdb.address), !dbg !41
  %vdb.address1436 = or disjoint i64 %147, %address.local921, !dbg !41
  %vdb.address1437 = trunc nuw i64 %vdb.address1436 to i32, !dbg !41
  %vdb.address1438 = shl nuw nsw i32 %vdb.address1437, 1, !dbg !41
  %vdb.address1439 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !41
  %vdb.address1440 = insertelement <2 x i64> poison, i64 %vdb.address1439, i64 0, !dbg !41
  %vdb.address1441 = bitcast <2 x i64> %vdb.address1440 to <4 x i32>, !dbg !41
  %vdb.address1442 = shufflevector <4 x i32> %vdb.address1441, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !41
  %vdb.address1443 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local521), !dbg !41
  %vdb.address1444 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %vdb.address1443), !dbg !41
  %vdb.address1445 = or disjoint i64 %147, %address.local927, !dbg !41
  %vdb.address1446 = trunc nuw i64 %vdb.address1445 to i32, !dbg !41
  %vdb.address1447 = shl nuw nsw i32 %vdb.address1446, 1, !dbg !41
  %vdb.address1448 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !41
  %vdb.address1449 = insertelement <2 x i64> poison, i64 %vdb.address1448, i64 0, !dbg !41
  %vdb.address1450 = bitcast <2 x i64> %vdb.address1449 to <4 x i32>, !dbg !41
  %vdb.address1451 = shufflevector <4 x i32> %vdb.address1450, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !41
  %vdb.dma.base = add i32 %vdb.address1435, 8192, !dbg !39
  %824 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1438, <4 x i32> %vdb.address1442) #10, !dbg !41
  %vdb.dma.base1452 = add i32 %vdb.address1444, 8192, !dbg !39
  %825 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1452, i32 %vdb.address1447, <4 x i32> %vdb.address1451) #10, !dbg !41
  %address.local929 = shl nuw nsw i32 %address.thread498, 3, !dbg !39
  %address.local930 = and i32 %address.local929, 120, !dbg !39
  %address.local931 = shl nuw nsw i32 %address.thread498, 5, !dbg !39
  %address.local932 = and i32 %address.local931, 1536, !dbg !39
  %address.local933 = or disjoint i32 %address.local930, %address.local932, !dbg !39
  %address.local934 = zext nneg i32 %address.local933 to i64, !dbg !39
  %address.local935 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local934, !dbg !39
  %826 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local935, i32 0) #0, !dbg !39, !frisk.v_double_buffer !59
  %827 = bitcast <4 x i32> %826 to <8 x half>, !dbg !39
  %address.local936 = and i64 %address.local499, 60, !dbg !39
  %address.local937 = add nuw nsw i64 %address.local936, %address.local499, !dbg !39
  %address.local938 = shl nuw nsw i64 %address.local937, 3, !dbg !39
  %address.local939 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local938, !dbg !39
  %address.local940 = lshr i64 %address.local555, 4, !dbg !39
  %address.local941 = mul nuw nsw i64 %address.local940, 384, !dbg !39
  %address.local942 = getelementptr [2 x i8], ptr addrspace(3) %address.local939, i64 %address.local941, !dbg !39
  %address.local943 = mul nsw i64 %address.local542, -32, !dbg !39
  %address.local944 = getelementptr [2 x i8], ptr addrspace(3) %address.local942, i64 %address.local943, !dbg !39
  %address.local945 = getelementptr i8, ptr addrspace(3) %address.local944, i64 256, !dbg !39
  %828 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local945, i32 0) #0, !dbg !39, !frisk.v_double_buffer !59
  %829 = bitcast <4 x i32> %828 to <8 x half>, !dbg !39
  %address.local946 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local938, !dbg !39
  %address.local947 = getelementptr [2 x i8], ptr addrspace(3) %address.local946, i64 %address.local941, !dbg !39
  %address.local948 = getelementptr [2 x i8], ptr addrspace(3) %address.local947, i64 %address.local943, !dbg !39
  %address.local949 = getelementptr i8, ptr addrspace(3) %address.local948, i64 512, !dbg !39
  %830 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local949, i32 0) #0, !dbg !39, !frisk.v_double_buffer !59
  %831 = bitcast <4 x i32> %830 to <8 x half>, !dbg !39
  %address.local950 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local938, !dbg !39
  %address.local951 = getelementptr [2 x i8], ptr addrspace(3) %address.local950, i64 %address.local941, !dbg !39
  %address.local952 = getelementptr [2 x i8], ptr addrspace(3) %address.local951, i64 %address.local943, !dbg !39
  %address.local953 = getelementptr i8, ptr addrspace(3) %address.local952, i64 768, !dbg !39
  %832 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local953, i32 0) #0, !dbg !39, !frisk.v_double_buffer !59
  %833 = bitcast <4 x i32> %832 to <8 x half>, !dbg !39
  %address.local954 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local934, !dbg !39
  %address.local955 = getelementptr i8, ptr addrspace(3) %address.local954, i64 4096, !dbg !39
  %834 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local955, i32 0) #0, !dbg !39, !frisk.v_double_buffer !59
  %835 = bitcast <4 x i32> %834 to <8 x half>, !dbg !39
  %address.local956 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local938, !dbg !39
  %address.local957 = getelementptr [2 x i8], ptr addrspace(3) %address.local956, i64 %address.local941, !dbg !39
  %address.local958 = getelementptr [2 x i8], ptr addrspace(3) %address.local957, i64 %address.local943, !dbg !39
  %address.local959 = getelementptr i8, ptr addrspace(3) %address.local958, i64 4352, !dbg !39
  %836 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local959, i32 0) #0, !dbg !39, !frisk.v_double_buffer !59
  %837 = bitcast <4 x i32> %836 to <8 x half>, !dbg !39
  %address.local960 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local938, !dbg !39
  %address.local961 = getelementptr [2 x i8], ptr addrspace(3) %address.local960, i64 %address.local941, !dbg !39
  %address.local962 = getelementptr [2 x i8], ptr addrspace(3) %address.local961, i64 %address.local943, !dbg !39
  %address.local963 = getelementptr i8, ptr addrspace(3) %address.local962, i64 4608, !dbg !39
  %838 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local963, i32 0) #0, !dbg !39, !frisk.v_double_buffer !59
  %839 = bitcast <4 x i32> %838 to <8 x half>, !dbg !39
  %address.local964 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local938, !dbg !39
  %address.local965 = getelementptr [2 x i8], ptr addrspace(3) %address.local964, i64 %address.local941, !dbg !39
  %address.local966 = getelementptr [2 x i8], ptr addrspace(3) %address.local965, i64 %address.local943, !dbg !39
  %address.local967 = getelementptr i8, ptr addrspace(3) %address.local966, i64 4864, !dbg !39
  %840 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local967, i32 0) #0, !dbg !39, !frisk.v_double_buffer !59
  %841 = shufflevector <8 x half> %827, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  %842 = extractvalue [2 x <8 x float>] %123, 0, !dbg !54
  %843 = shufflevector <8 x float> %842, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !54
  %844 = fmul <4 x float> %843, %596, !dbg !54
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %845 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %841, <4 x float> %844) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %846 = shufflevector <8 x half> %827, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  %847 = shufflevector <8 x float> %842, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !54
  %848 = fmul <4 x float> %847, %596, !dbg !54
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %849 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %846, <4 x float> %848) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !54
  %850 = extractvalue [2 x <8 x float>] %123, 1, !dbg !54
  %851 = shufflevector <8 x float> %850, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !54
  %852 = fmul <4 x float> %851, %604, !dbg !54
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %853 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %841, <4 x float> %852) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !54
  %854 = shufflevector <8 x float> %850, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !54
  %855 = fmul <4 x float> %854, %604, !dbg !54
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %856 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %846, <4 x float> %855) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %857 = shufflevector <8 x half> %829, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %858 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %857, <4 x float> %845) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %859 = shufflevector <8 x half> %829, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %860 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %859, <4 x float> %849) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %861 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %857, <4 x float> %853) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %862 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %859, <4 x float> %856) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %863 = shufflevector <8 x half> %831, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %864 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %863, <4 x float> %858) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %865 = shufflevector <8 x half> %831, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %866 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %865, <4 x float> %860) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %867 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %863, <4 x float> %861) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %868 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %865, <4 x float> %862) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %869 = shufflevector <8 x half> %833, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %870 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %869, <4 x float> %864) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %871 = shufflevector <8 x half> %833, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %872 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %871, <4 x float> %866) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %873 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %869, <4 x float> %867) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %874 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %871, <4 x float> %868) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %875 = shufflevector <8 x half> %835, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %876 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %875, <4 x float> %870) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %877 = shufflevector <8 x half> %835, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %878 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %877, <4 x float> %872) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %879 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %875, <4 x float> %873) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %880 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %877, <4 x float> %874) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %881 = shufflevector <8 x half> %837, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %882 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %881, <4 x float> %876) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %883 = shufflevector <8 x half> %837, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %884 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %883, <4 x float> %878) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %885 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %881, <4 x float> %879) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %886 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %883, <4 x float> %880) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %887 = shufflevector <8 x half> %839, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %888 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %887, <4 x float> %882) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %889 = shufflevector <8 x half> %839, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %890 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %889, <4 x float> %884) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %891 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %887, <4 x float> %885) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %892 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %889, <4 x float> %886) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %893 = bitcast <4 x i32> %840 to <8 x half>, !dbg !39
  %894 = shufflevector <8 x half> %893, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %895 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %894, <4 x float> %888) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %896 = shufflevector <8 x half> %893, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %897 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %896, <4 x float> %890) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %898 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %894, <4 x float> %891) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %899 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %896, <4 x float> %892) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !41
  fence syncscope("workgroup") release, !dbg !41
  call void @llvm.amdgcn.s.barrier(), !dbg !41
  fence syncscope("workgroup") acquire, !dbg !41
  %vdb.address1460 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local504), !dbg !41
  %vdb.address1461 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %vdb.address1460), !dbg !41
  %vdb.address1462 = or disjoint i64 %244, %address.local921, !dbg !41
  %vdb.address1463 = trunc nuw i64 %vdb.address1462 to i32, !dbg !41
  %vdb.address1464 = shl nuw nsw i32 %vdb.address1463, 1, !dbg !41
  %vdb.address1465 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !41
  %vdb.address1466 = insertelement <2 x i64> poison, i64 %vdb.address1465, i64 0, !dbg !41
  %vdb.address1467 = bitcast <2 x i64> %vdb.address1466 to <4 x i32>, !dbg !41
  %vdb.address1468 = shufflevector <4 x i32> %vdb.address1467, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !41
  %vdb.address1469 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local521), !dbg !41
  %vdb.address1470 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %vdb.address1469), !dbg !41
  %vdb.address1471 = or disjoint i64 %244, %address.local927, !dbg !41
  %vdb.address1472 = trunc nuw i64 %vdb.address1471 to i32, !dbg !41
  %vdb.address1473 = shl nuw nsw i32 %vdb.address1472, 1, !dbg !41
  %vdb.address1474 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !41
  %vdb.address1475 = insertelement <2 x i64> poison, i64 %vdb.address1474, i64 0, !dbg !41
  %vdb.address1476 = bitcast <2 x i64> %vdb.address1475 to <4 x i32>, !dbg !41
  %vdb.address1477 = shufflevector <4 x i32> %vdb.address1476, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !41
  %900 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.address1461, i32 %vdb.address1464, <4 x i32> %vdb.address1468) #10, !dbg !41
  %901 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.address1470, i32 %vdb.address1473, <4 x i32> %vdb.address1477) #10, !dbg !41
  %vdb.read = getelementptr i8, ptr addrspace(3) %address.local935, i64 8192, !dbg !39
  %902 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #0, !dbg !39, !frisk.v_double_buffer !59
  %903 = bitcast <4 x i32> %902 to <8 x half>, !dbg !39
  %vdb.read1453 = getelementptr i8, ptr addrspace(3) %address.local945, i64 8192, !dbg !39
  %904 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1453, i32 0) #0, !dbg !39, !frisk.v_double_buffer !59
  %905 = bitcast <4 x i32> %904 to <8 x half>, !dbg !39
  %vdb.read1454 = getelementptr i8, ptr addrspace(3) %address.local949, i64 8192, !dbg !39
  %906 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1454, i32 0) #0, !dbg !39, !frisk.v_double_buffer !59
  %907 = bitcast <4 x i32> %906 to <8 x half>, !dbg !39
  %vdb.read1455 = getelementptr i8, ptr addrspace(3) %address.local953, i64 8192, !dbg !39
  %908 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1455, i32 0) #0, !dbg !39, !frisk.v_double_buffer !59
  %909 = bitcast <4 x i32> %908 to <8 x half>, !dbg !39
  %vdb.read1456 = getelementptr i8, ptr addrspace(3) %address.local955, i64 8192, !dbg !39
  %910 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1456, i32 0) #0, !dbg !39, !frisk.v_double_buffer !59
  %911 = bitcast <4 x i32> %910 to <8 x half>, !dbg !39
  %vdb.read1457 = getelementptr i8, ptr addrspace(3) %address.local959, i64 8192, !dbg !39
  %912 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1457, i32 0) #0, !dbg !39, !frisk.v_double_buffer !59
  %913 = bitcast <4 x i32> %912 to <8 x half>, !dbg !39
  %vdb.read1458 = getelementptr i8, ptr addrspace(3) %address.local963, i64 8192, !dbg !39
  %914 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1458, i32 0) #0, !dbg !39, !frisk.v_double_buffer !59
  %915 = bitcast <4 x i32> %914 to <8 x half>, !dbg !39
  %vdb.read1459 = getelementptr i8, ptr addrspace(3) %address.local967, i64 8192, !dbg !39
  %916 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1459, i32 0) #0, !dbg !39, !frisk.v_double_buffer !59
  %917 = shufflevector <8 x half> %903, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  %918 = extractvalue [2 x <8 x float>] %126, 0, !dbg !54
  %919 = shufflevector <8 x float> %918, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !54
  %920 = fmul <4 x float> %919, %596, !dbg !54
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %921 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %917, <4 x float> %920) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %922 = shufflevector <8 x half> %903, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  %923 = shufflevector <8 x float> %918, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !54
  %924 = fmul <4 x float> %923, %596, !dbg !54
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %925 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %922, <4 x float> %924) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !54
  %926 = extractvalue [2 x <8 x float>] %126, 1, !dbg !54
  %927 = shufflevector <8 x float> %926, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !54
  %928 = fmul <4 x float> %927, %604, !dbg !54
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %929 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %917, <4 x float> %928) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !54
  %930 = shufflevector <8 x float> %926, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !54
  %931 = fmul <4 x float> %930, %604, !dbg !54
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %932 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %922, <4 x float> %931) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %933 = shufflevector <8 x half> %905, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %934 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %933, <4 x float> %921) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %935 = shufflevector <8 x half> %905, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %936 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %935, <4 x float> %925) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %937 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %933, <4 x float> %929) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %938 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %935, <4 x float> %932) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %939 = shufflevector <8 x half> %907, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %940 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %939, <4 x float> %934) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %941 = shufflevector <8 x half> %907, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %942 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %941, <4 x float> %936) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %943 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %939, <4 x float> %937) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %944 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %941, <4 x float> %938) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %945 = shufflevector <8 x half> %909, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %946 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %945, <4 x float> %940) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %947 = shufflevector <8 x half> %909, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %948 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %947, <4 x float> %942) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %949 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %945, <4 x float> %943) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %950 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %947, <4 x float> %944) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %951 = shufflevector <8 x half> %911, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %952 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %951, <4 x float> %946) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %953 = shufflevector <8 x half> %911, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %954 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %953, <4 x float> %948) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %955 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %951, <4 x float> %949) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %956 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %953, <4 x float> %950) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %957 = shufflevector <8 x half> %913, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %958 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %957, <4 x float> %952) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %959 = shufflevector <8 x half> %913, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %960 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %959, <4 x float> %954) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %961 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %957, <4 x float> %955) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %962 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %959, <4 x float> %956) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %963 = shufflevector <8 x half> %915, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %964 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %963, <4 x float> %958) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %965 = shufflevector <8 x half> %915, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %966 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %965, <4 x float> %960) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %967 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %963, <4 x float> %961) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %968 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %965, <4 x float> %962) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %969 = bitcast <4 x i32> %916 to <8 x half>, !dbg !39
  %970 = shufflevector <8 x half> %969, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %971 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %970, <4 x float> %964) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %972 = shufflevector <8 x half> %969, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %973 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %972, <4 x float> %966) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %974 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %970, <4 x float> %967) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %975 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %972, <4 x float> %968) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !41
  fence syncscope("workgroup") release, !dbg !41
  call void @llvm.amdgcn.s.barrier(), !dbg !41
  fence syncscope("workgroup") acquire, !dbg !41
  %vdb.address1478 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local504), !dbg !41
  %vdb.address1479 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %vdb.address1478), !dbg !41
  %vdb.address1480 = or disjoint i64 %321, %address.local921, !dbg !41
  %vdb.address1481 = trunc nuw i64 %vdb.address1480 to i32, !dbg !41
  %vdb.address1482 = shl nuw nsw i32 %vdb.address1481, 1, !dbg !41
  %vdb.address1483 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !41
  %vdb.address1484 = insertelement <2 x i64> poison, i64 %vdb.address1483, i64 0, !dbg !41
  %vdb.address1485 = bitcast <2 x i64> %vdb.address1484 to <4 x i32>, !dbg !41
  %vdb.address1486 = shufflevector <4 x i32> %vdb.address1485, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !41
  %vdb.address1487 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local521), !dbg !41
  %vdb.address1488 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %vdb.address1487), !dbg !41
  %vdb.address1489 = or disjoint i64 %321, %address.local927, !dbg !41
  %vdb.address1490 = trunc nuw i64 %vdb.address1489 to i32, !dbg !41
  %vdb.address1491 = shl nuw nsw i32 %vdb.address1490, 1, !dbg !41
  %vdb.address1492 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !41
  %vdb.address1493 = insertelement <2 x i64> poison, i64 %vdb.address1492, i64 0, !dbg !41
  %vdb.address1494 = bitcast <2 x i64> %vdb.address1493 to <4 x i32>, !dbg !41
  %vdb.address1495 = shufflevector <4 x i32> %vdb.address1494, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !41
  %vdb.dma.base1496 = add i32 %vdb.address1479, 8192, !dbg !39
  %976 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1496, i32 %vdb.address1482, <4 x i32> %vdb.address1486) #10, !dbg !41
  %vdb.dma.base1497 = add i32 %vdb.address1488, 8192, !dbg !39
  %977 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1497, i32 %vdb.address1491, <4 x i32> %vdb.address1495) #10, !dbg !41
  %978 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local935, i32 0) #0, !dbg !39, !frisk.v_double_buffer !59
  %979 = bitcast <4 x i32> %978 to <8 x half>, !dbg !39
  %980 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local945, i32 0) #0, !dbg !39, !frisk.v_double_buffer !59
  %981 = bitcast <4 x i32> %980 to <8 x half>, !dbg !39
  %982 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local949, i32 0) #0, !dbg !39, !frisk.v_double_buffer !59
  %983 = bitcast <4 x i32> %982 to <8 x half>, !dbg !39
  %984 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local953, i32 0) #0, !dbg !39, !frisk.v_double_buffer !59
  %985 = bitcast <4 x i32> %984 to <8 x half>, !dbg !39
  %986 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local955, i32 0) #0, !dbg !39, !frisk.v_double_buffer !59
  %987 = bitcast <4 x i32> %986 to <8 x half>, !dbg !39
  %988 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local959, i32 0) #0, !dbg !39, !frisk.v_double_buffer !59
  %989 = bitcast <4 x i32> %988 to <8 x half>, !dbg !39
  %990 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local963, i32 0) #0, !dbg !39, !frisk.v_double_buffer !59
  %991 = bitcast <4 x i32> %990 to <8 x half>, !dbg !39
  %992 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local967, i32 0) #0, !dbg !39, !frisk.v_double_buffer !59
  %993 = shufflevector <8 x half> %979, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  %994 = extractvalue [2 x <8 x float>] %127, 0, !dbg !54
  %995 = shufflevector <8 x float> %994, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !54
  %996 = fmul <4 x float> %995, %596, !dbg !54
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %997 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %993, <4 x float> %996) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %998 = shufflevector <8 x half> %979, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  %999 = shufflevector <8 x float> %994, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !54
  %1000 = fmul <4 x float> %999, %596, !dbg !54
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1001 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %998, <4 x float> %1000) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !54
  %1002 = extractvalue [2 x <8 x float>] %127, 1, !dbg !54
  %1003 = shufflevector <8 x float> %1002, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !54
  %1004 = fmul <4 x float> %1003, %604, !dbg !54
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1005 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %993, <4 x float> %1004) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !54
  %1006 = shufflevector <8 x float> %1002, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !54
  %1007 = fmul <4 x float> %1006, %604, !dbg !54
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1008 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %998, <4 x float> %1007) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1009 = shufflevector <8 x half> %981, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1010 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %1009, <4 x float> %997) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1011 = shufflevector <8 x half> %981, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1012 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %1011, <4 x float> %1001) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1013 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %1009, <4 x float> %1005) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1014 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %1011, <4 x float> %1008) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1015 = shufflevector <8 x half> %983, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1016 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %1015, <4 x float> %1010) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1017 = shufflevector <8 x half> %983, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1018 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %1017, <4 x float> %1012) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1019 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %1015, <4 x float> %1013) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1020 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %1017, <4 x float> %1014) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1021 = shufflevector <8 x half> %985, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1022 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %1021, <4 x float> %1016) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1023 = shufflevector <8 x half> %985, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1024 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %1023, <4 x float> %1018) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1025 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %1021, <4 x float> %1019) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1026 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %1023, <4 x float> %1020) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1027 = shufflevector <8 x half> %987, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1028 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %1027, <4 x float> %1022) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1029 = shufflevector <8 x half> %987, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1030 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %1029, <4 x float> %1024) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1031 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %1027, <4 x float> %1025) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1032 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %1029, <4 x float> %1026) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1033 = shufflevector <8 x half> %989, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1034 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %1033, <4 x float> %1028) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1035 = shufflevector <8 x half> %989, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1036 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %1035, <4 x float> %1030) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1037 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %1033, <4 x float> %1031) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1038 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %1035, <4 x float> %1032) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1039 = shufflevector <8 x half> %991, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1040 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %1039, <4 x float> %1034) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1041 = shufflevector <8 x half> %991, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1042 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %1041, <4 x float> %1036) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1043 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %1039, <4 x float> %1037) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1044 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %1041, <4 x float> %1038) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1045 = bitcast <4 x i32> %992 to <8 x half>, !dbg !39
  %1046 = shufflevector <8 x half> %1045, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1047 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %1046, <4 x float> %1040) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1048 = shufflevector <8 x half> %1045, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1049 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %1048, <4 x float> %1042) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1050 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %1046, <4 x float> %1043) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1051 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %1048, <4 x float> %1044) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !41
  fence syncscope("workgroup") release, !dbg !41
  call void @llvm.amdgcn.s.barrier(), !dbg !41
  fence syncscope("workgroup") acquire, !dbg !41
  %vdb.read1498 = getelementptr i8, ptr addrspace(3) %address.local935, i64 8192, !dbg !39
  %1052 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1498, i32 0) #0, !dbg !39, !frisk.v_double_buffer !59
  %1053 = bitcast <4 x i32> %1052 to <8 x half>, !dbg !39
  %vdb.read1499 = getelementptr i8, ptr addrspace(3) %address.local945, i64 8192, !dbg !39
  %1054 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1499, i32 0) #0, !dbg !39, !frisk.v_double_buffer !59
  %1055 = bitcast <4 x i32> %1054 to <8 x half>, !dbg !39
  %vdb.read1500 = getelementptr i8, ptr addrspace(3) %address.local949, i64 8192, !dbg !39
  %1056 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1500, i32 0) #0, !dbg !39, !frisk.v_double_buffer !59
  %1057 = bitcast <4 x i32> %1056 to <8 x half>, !dbg !39
  %vdb.read1501 = getelementptr i8, ptr addrspace(3) %address.local953, i64 8192, !dbg !39
  %1058 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1501, i32 0) #0, !dbg !39, !frisk.v_double_buffer !59
  %1059 = bitcast <4 x i32> %1058 to <8 x half>, !dbg !39
  %vdb.read1502 = getelementptr i8, ptr addrspace(3) %address.local955, i64 8192, !dbg !39
  %1060 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1502, i32 0) #0, !dbg !39, !frisk.v_double_buffer !59
  %1061 = bitcast <4 x i32> %1060 to <8 x half>, !dbg !39
  %vdb.read1503 = getelementptr i8, ptr addrspace(3) %address.local959, i64 8192, !dbg !39
  %1062 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1503, i32 0) #0, !dbg !39, !frisk.v_double_buffer !59
  %1063 = bitcast <4 x i32> %1062 to <8 x half>, !dbg !39
  %vdb.read1504 = getelementptr i8, ptr addrspace(3) %address.local963, i64 8192, !dbg !39
  %1064 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1504, i32 0) #0, !dbg !39, !frisk.v_double_buffer !59
  %1065 = bitcast <4 x i32> %1064 to <8 x half>, !dbg !39
  %vdb.read1505 = getelementptr i8, ptr addrspace(3) %address.local967, i64 8192, !dbg !39
  %1066 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1505, i32 0) #0, !dbg !39, !frisk.v_double_buffer !59
  %1067 = shufflevector <8 x half> %1053, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  %1068 = extractvalue [2 x <8 x float>] %128, 0, !dbg !54
  %1069 = shufflevector <8 x float> %1068, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !54
  %1070 = fmul <4 x float> %1069, %596, !dbg !54
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1071 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1067, <4 x float> %1070) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1072 = shufflevector <8 x half> %1053, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  %1073 = shufflevector <8 x float> %1068, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !54
  %1074 = fmul <4 x float> %1073, %596, !dbg !54
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1075 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1072, <4 x float> %1074) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !54
  %1076 = extractvalue [2 x <8 x float>] %128, 1, !dbg !54
  %1077 = shufflevector <8 x float> %1076, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !54
  %1078 = fmul <4 x float> %1077, %604, !dbg !54
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1079 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %1067, <4 x float> %1078) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !54
  %1080 = shufflevector <8 x float> %1076, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !54
  %1081 = fmul <4 x float> %1080, %604, !dbg !54
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1082 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %1072, <4 x float> %1081) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1083 = shufflevector <8 x half> %1055, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1084 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %1083, <4 x float> %1071) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1085 = shufflevector <8 x half> %1055, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1086 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %1085, <4 x float> %1075) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1087 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %1083, <4 x float> %1079) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1088 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %1085, <4 x float> %1082) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1089 = shufflevector <8 x half> %1057, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1090 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %1089, <4 x float> %1084) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1091 = shufflevector <8 x half> %1057, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1092 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %1091, <4 x float> %1086) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1093 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %1089, <4 x float> %1087) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1094 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %1091, <4 x float> %1088) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1095 = shufflevector <8 x half> %1059, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1096 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %1095, <4 x float> %1090) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1097 = shufflevector <8 x half> %1059, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1098 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %1097, <4 x float> %1092) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1099 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %1095, <4 x float> %1093) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1100 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %1097, <4 x float> %1094) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1101 = shufflevector <8 x half> %1061, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1102 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %1101, <4 x float> %1096) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1103 = shufflevector <8 x half> %1061, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1104 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %1103, <4 x float> %1098) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1105 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %1101, <4 x float> %1099) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1106 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %1103, <4 x float> %1100) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1107 = shufflevector <8 x half> %1063, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1108 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %1107, <4 x float> %1102) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1109 = shufflevector <8 x half> %1063, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1110 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %1109, <4 x float> %1104) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1111 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %1107, <4 x float> %1105) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1112 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %1109, <4 x float> %1106) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1113 = shufflevector <8 x half> %1065, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1114 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %1113, <4 x float> %1108) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1115 = shufflevector <8 x half> %1065, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1116 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %1115, <4 x float> %1110) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1117 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %1113, <4 x float> %1111) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1118 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %1115, <4 x float> %1112) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1119 = bitcast <4 x i32> %1066 to <8 x half>, !dbg !39
  %1120 = shufflevector <8 x half> %1119, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1121 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %1120, <4 x float> %1114) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1122 = shufflevector <8 x half> %1119, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1123 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %1122, <4 x float> %1116) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1124 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %1120, <4 x float> %1117) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1125 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %1122, <4 x float> %1118) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !35
  %1126 = shufflevector <4 x float> %895, <4 x float> %897, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1127 = insertvalue [2 x <8 x float>] poison, <8 x float> %1126, 0, !dbg !35
  %1128 = shufflevector <4 x float> %898, <4 x float> %899, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1129 = insertvalue [2 x <8 x float>] %1127, <8 x float> %1128, 1, !dbg !35
  %1130 = insertvalue [2 x <1 x float>] poison, <1 x float> %560, 0, !dbg !35
  %1131 = insertvalue [2 x <1 x float>] %1130, <1 x float> %573, 1, !dbg !35
  %1132 = shufflevector <4 x float> %971, <4 x float> %973, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1133 = insertvalue [2 x <8 x float>] poison, <8 x float> %1132, 0, !dbg !35
  %1134 = shufflevector <4 x float> %974, <4 x float> %975, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1135 = insertvalue [2 x <8 x float>] %1133, <8 x float> %1134, 1, !dbg !35
  %1136 = shufflevector <4 x float> %1047, <4 x float> %1049, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1137 = insertvalue [2 x <8 x float>] poison, <8 x float> %1136, 0, !dbg !35
  %1138 = shufflevector <4 x float> %1050, <4 x float> %1051, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1139 = insertvalue [2 x <8 x float>] %1137, <8 x float> %1138, 1, !dbg !35
  %1140 = shufflevector <4 x float> %1121, <4 x float> %1123, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1141 = insertvalue [2 x <8 x float>] poison, <8 x float> %1140, 0, !dbg !35
  %1142 = shufflevector <4 x float> %1124, <4 x float> %1125, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1143 = insertvalue [2 x <8 x float>] %1141, <8 x float> %1142, 1, !dbg !35
  fence syncscope("workgroup") release, !dbg !35
  call void @llvm.amdgcn.s.barrier(), !dbg !35
  fence syncscope("workgroup") acquire, !dbg !35
  %1144 = add nuw nsw i64 %122, 1, !dbg !35
  br label %121, !dbg !35

1145:                                             ; preds = %121
  %lane.sum.field184 = extractvalue [2 x <1 x float>] %124, 1, !dbg !35
  %lane.sum.local185 = extractelement <1 x float> %lane.sum.field184, i64 0, !dbg !35
  %lane.sum.lane.lo186 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !35
  %lane.sum.lane187 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %lane.sum.lane.lo186), !dbg !35
  %1146 = shl nuw nsw i32 %lane.sum.lane187, 2, !dbg !35
  %1147 = xor i32 %1146, 64, !dbg !35
  %1148 = bitcast float %lane.sum.local185 to i32, !dbg !35
  %lane.sum.shuffle188 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1147, i32 %1148), !dbg !35
  %1149 = bitcast i32 %lane.sum.shuffle188 to float, !dbg !35
  %lane.sum.final189 = fadd float %lane.sum.local185, %1149, !dbg !35
  %1150 = shl nuw nsw i32 %lane.sum.lane187, 2, !dbg !35
  %1151 = xor i32 %1150, 128, !dbg !35
  %1152 = bitcast float %lane.sum.final189 to i32, !dbg !35
  %lane.sum.shuffle190 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1151, i32 %1152), !dbg !35
  %1153 = bitcast i32 %lane.sum.shuffle190 to float, !dbg !35
  %lane.sum.final191 = fadd float %lane.sum.final189, %1153, !dbg !35
  %1154 = insertelement <1 x float> poison, float %lane.sum.final191, i64 0, !dbg !35
  %lane.sum.field = extractvalue [2 x <1 x float>] %124, 0, !dbg !35
  %lane.sum.local = extractelement <1 x float> %lane.sum.field, i64 0, !dbg !35
  %lane.sum.lane.lo = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !35
  %lane.sum.lane = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %lane.sum.lane.lo), !dbg !35
  %1155 = shl nuw nsw i32 %lane.sum.lane, 2, !dbg !35
  %1156 = xor i32 %1155, 64, !dbg !35
  %1157 = bitcast float %lane.sum.local to i32, !dbg !35
  %lane.sum.shuffle = call i32 @llvm.amdgcn.ds.bpermute(i32 %1156, i32 %1157), !dbg !35
  %1158 = bitcast i32 %lane.sum.shuffle to float, !dbg !35
  %lane.sum.final = fadd float %lane.sum.local, %1158, !dbg !35
  %1159 = shl nuw nsw i32 %lane.sum.lane, 2, !dbg !35
  %1160 = xor i32 %1159, 128, !dbg !35
  %1161 = bitcast float %lane.sum.final to i32, !dbg !35
  %lane.sum.shuffle182 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1160, i32 %1161), !dbg !35
  %1162 = bitcast i32 %lane.sum.shuffle182 to float, !dbg !35
  %lane.sum.final183 = fadd float %lane.sum.final, %1162, !dbg !35
  %1163 = insertelement <1 x float> poison, float %lane.sum.final183, i64 0, !dbg !35
  %1164 = and i32 %27, 16, !dbg !35
  %.not = icmp eq i32 %1164, 0, !dbg !35
  %1165 = and i32 %27, 32, !dbg !35
  %.not36 = icmp eq i32 %1165, 0, !dbg !35
  %1166 = extractvalue [2 x <8 x float>] %128, 0, !dbg !35
  %1167 = extractelement <8 x float> %1166, i64 0, !dbg !35
  %1168 = extractelement <8 x float> %1166, i64 1, !dbg !35
  %1169 = extractelement <8 x float> %1166, i64 2, !dbg !35
  %1170 = extractelement <8 x float> %1166, i64 3, !dbg !35
  %1171 = select i1 %.not, float %1168, float %1167, !dbg !35
  %1172 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !35
  %1173 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1172), !dbg !35
  %1174 = add nuw nsw i32 %1173, 64, !dbg !35
  %1175 = xor i32 %1173, 16, !dbg !35
  %1176 = and i32 %1174, 192, !dbg !35
  %1177 = icmp samesign ult i32 %1175, %1176, !dbg !35
  %1178 = select i1 %1177, i32 %1175, i32 %1173, !dbg !35
  %1179 = shl nuw nsw i32 %1178, 2, !dbg !35
  %1180 = bitcast float %1171 to i32, !dbg !35
  %1181 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1180), !dbg !35
  %1182 = bitcast i32 %1181 to float, !dbg !35
  %1183 = select i1 %.not, float %1167, float %1182, !dbg !35
  %1184 = select i1 %.not, float %1182, float %1168, !dbg !35
  %1185 = select i1 %.not, float %1170, float %1169, !dbg !35
  %1186 = bitcast float %1185 to i32, !dbg !35
  %1187 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1186), !dbg !35
  %1188 = bitcast i32 %1187 to float, !dbg !35
  %1189 = select i1 %.not, float %1169, float %1188, !dbg !35
  %1190 = select i1 %.not, float %1188, float %1170, !dbg !35
  %1191 = select i1 %.not36, float %1189, float %1183, !dbg !35
  %1192 = xor i32 %1173, 32, !dbg !35
  %1193 = icmp samesign ult i32 %1192, %1176, !dbg !35
  %1194 = select i1 %1193, i32 %1192, i32 %1173, !dbg !35
  %1195 = shl nuw nsw i32 %1194, 2, !dbg !35
  %1196 = bitcast float %1191 to i32, !dbg !35
  %1197 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1195, i32 %1196), !dbg !35
  %1198 = bitcast i32 %1197 to float, !dbg !35
  %1199 = select i1 %.not36, float %1183, float %1198, !dbg !35
  %1200 = select i1 %.not36, float %1198, float %1189, !dbg !35
  %1201 = select i1 %.not36, float %1190, float %1184, !dbg !35
  %1202 = bitcast float %1201 to i32, !dbg !35
  %1203 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1195, i32 %1202), !dbg !35
  %1204 = bitcast i32 %1203 to float, !dbg !35
  %1205 = select i1 %.not36, float %1184, float %1204, !dbg !35
  %1206 = select i1 %.not36, float %1204, float %1190, !dbg !35
  %1207 = extractvalue [2 x <8 x float>] %127, 0, !dbg !35
  %1208 = extractelement <8 x float> %1207, i64 0, !dbg !35
  %1209 = extractelement <8 x float> %1207, i64 1, !dbg !35
  %1210 = extractelement <8 x float> %1207, i64 2, !dbg !35
  %1211 = extractelement <8 x float> %1207, i64 3, !dbg !35
  %1212 = select i1 %.not, float %1209, float %1208, !dbg !35
  %1213 = bitcast float %1212 to i32, !dbg !35
  %1214 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1213), !dbg !35
  %1215 = bitcast i32 %1214 to float, !dbg !35
  %1216 = select i1 %.not, float %1208, float %1215, !dbg !35
  %1217 = select i1 %.not, float %1215, float %1209, !dbg !35
  %1218 = select i1 %.not, float %1211, float %1210, !dbg !35
  %1219 = bitcast float %1218 to i32, !dbg !35
  %1220 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1219), !dbg !35
  %1221 = bitcast i32 %1220 to float, !dbg !35
  %1222 = select i1 %.not, float %1210, float %1221, !dbg !35
  %1223 = select i1 %.not, float %1221, float %1211, !dbg !35
  %1224 = select i1 %.not36, float %1222, float %1216, !dbg !35
  %1225 = bitcast float %1224 to i32, !dbg !35
  %1226 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1195, i32 %1225), !dbg !35
  %1227 = bitcast i32 %1226 to float, !dbg !35
  %1228 = select i1 %.not36, float %1216, float %1227, !dbg !35
  %1229 = select i1 %.not36, float %1227, float %1222, !dbg !35
  %1230 = select i1 %.not36, float %1223, float %1217, !dbg !35
  %1231 = bitcast float %1230 to i32, !dbg !35
  %1232 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1195, i32 %1231), !dbg !35
  %1233 = bitcast i32 %1232 to float, !dbg !35
  %1234 = select i1 %.not36, float %1217, float %1233, !dbg !35
  %1235 = select i1 %.not36, float %1233, float %1223, !dbg !35
  %1236 = extractvalue [2 x <8 x float>] %126, 0, !dbg !35
  %1237 = extractelement <8 x float> %1236, i64 0, !dbg !35
  %1238 = extractelement <8 x float> %1236, i64 1, !dbg !35
  %1239 = extractelement <8 x float> %1236, i64 2, !dbg !35
  %1240 = extractelement <8 x float> %1236, i64 3, !dbg !35
  %1241 = select i1 %.not, float %1238, float %1237, !dbg !35
  %1242 = bitcast float %1241 to i32, !dbg !35
  %1243 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1242), !dbg !35
  %1244 = bitcast i32 %1243 to float, !dbg !35
  %1245 = select i1 %.not, float %1237, float %1244, !dbg !35
  %1246 = select i1 %.not, float %1244, float %1238, !dbg !35
  %1247 = select i1 %.not, float %1240, float %1239, !dbg !35
  %1248 = bitcast float %1247 to i32, !dbg !35
  %1249 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1248), !dbg !35
  %1250 = bitcast i32 %1249 to float, !dbg !35
  %1251 = select i1 %.not, float %1239, float %1250, !dbg !35
  %1252 = select i1 %.not, float %1250, float %1240, !dbg !35
  %1253 = select i1 %.not36, float %1251, float %1245, !dbg !35
  %1254 = bitcast float %1253 to i32, !dbg !35
  %1255 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1195, i32 %1254), !dbg !35
  %1256 = bitcast i32 %1255 to float, !dbg !35
  %1257 = select i1 %.not36, float %1245, float %1256, !dbg !35
  %1258 = select i1 %.not36, float %1256, float %1251, !dbg !35
  %1259 = select i1 %.not36, float %1252, float %1246, !dbg !35
  %1260 = bitcast float %1259 to i32, !dbg !35
  %1261 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1195, i32 %1260), !dbg !35
  %1262 = bitcast i32 %1261 to float, !dbg !35
  %1263 = select i1 %.not36, float %1246, float %1262, !dbg !35
  %1264 = select i1 %.not36, float %1262, float %1252, !dbg !35
  %1265 = extractvalue [2 x <8 x float>] %123, 0, !dbg !35
  %1266 = extractelement <8 x float> %1265, i64 0, !dbg !35
  %1267 = extractelement <8 x float> %1265, i64 1, !dbg !35
  %1268 = extractelement <8 x float> %1265, i64 2, !dbg !35
  %1269 = extractelement <8 x float> %1265, i64 3, !dbg !35
  %1270 = select i1 %.not, float %1267, float %1266, !dbg !35
  %1271 = bitcast float %1270 to i32, !dbg !35
  %1272 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1271), !dbg !35
  %1273 = bitcast i32 %1272 to float, !dbg !35
  %1274 = select i1 %.not, float %1266, float %1273, !dbg !35
  %1275 = select i1 %.not, float %1273, float %1267, !dbg !35
  %1276 = select i1 %.not, float %1269, float %1268, !dbg !35
  %1277 = bitcast float %1276 to i32, !dbg !35
  %1278 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1277), !dbg !35
  %1279 = bitcast i32 %1278 to float, !dbg !35
  %1280 = select i1 %.not, float %1268, float %1279, !dbg !35
  %1281 = select i1 %.not, float %1279, float %1269, !dbg !35
  %1282 = select i1 %.not36, float %1280, float %1274, !dbg !35
  %1283 = bitcast float %1282 to i32, !dbg !35
  %1284 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1195, i32 %1283), !dbg !35
  %1285 = select i1 %.not36, float %1281, float %1275, !dbg !35
  %1286 = bitcast float %1285 to i32, !dbg !35
  %1287 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1195, i32 %1286), !dbg !35
  %1288 = shufflevector <1 x float> %1163, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !42
  %1289 = extractelement <8 x float> %1166, i64 4, !dbg !35
  %1290 = extractelement <8 x float> %1166, i64 5, !dbg !35
  %1291 = extractelement <8 x float> %1166, i64 6, !dbg !35
  %1292 = extractelement <8 x float> %1166, i64 7, !dbg !35
  %1293 = select i1 %.not, float %1290, float %1289, !dbg !35
  %1294 = bitcast float %1293 to i32, !dbg !35
  %1295 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1294), !dbg !35
  %1296 = bitcast i32 %1295 to float, !dbg !35
  %1297 = select i1 %.not, float %1289, float %1296, !dbg !35
  %1298 = select i1 %.not, float %1296, float %1290, !dbg !35
  %1299 = select i1 %.not, float %1292, float %1291, !dbg !35
  %1300 = bitcast float %1299 to i32, !dbg !35
  %1301 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1300), !dbg !35
  %1302 = bitcast i32 %1301 to float, !dbg !35
  %1303 = select i1 %.not, float %1291, float %1302, !dbg !35
  %1304 = select i1 %.not, float %1302, float %1292, !dbg !35
  %1305 = select i1 %.not36, float %1303, float %1297, !dbg !35
  %1306 = bitcast float %1305 to i32, !dbg !35
  %1307 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1195, i32 %1306), !dbg !35
  %1308 = bitcast i32 %1307 to float, !dbg !35
  %1309 = select i1 %.not36, float %1297, float %1308, !dbg !35
  %1310 = select i1 %.not36, float %1308, float %1303, !dbg !35
  %1311 = select i1 %.not36, float %1304, float %1298, !dbg !35
  %1312 = bitcast float %1311 to i32, !dbg !35
  %1313 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1195, i32 %1312), !dbg !35
  %1314 = bitcast i32 %1313 to float, !dbg !35
  %1315 = select i1 %.not36, float %1298, float %1314, !dbg !35
  %1316 = select i1 %.not36, float %1314, float %1304, !dbg !35
  %1317 = extractelement <8 x float> %1207, i64 4, !dbg !35
  %1318 = extractelement <8 x float> %1207, i64 5, !dbg !35
  %1319 = extractelement <8 x float> %1207, i64 6, !dbg !35
  %1320 = extractelement <8 x float> %1207, i64 7, !dbg !35
  %1321 = select i1 %.not, float %1318, float %1317, !dbg !35
  %1322 = bitcast float %1321 to i32, !dbg !35
  %1323 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1322), !dbg !35
  %1324 = bitcast i32 %1323 to float, !dbg !35
  %1325 = select i1 %.not, float %1317, float %1324, !dbg !35
  %1326 = select i1 %.not, float %1324, float %1318, !dbg !35
  %1327 = select i1 %.not, float %1320, float %1319, !dbg !35
  %1328 = bitcast float %1327 to i32, !dbg !35
  %1329 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1328), !dbg !35
  %1330 = bitcast i32 %1329 to float, !dbg !35
  %1331 = select i1 %.not, float %1319, float %1330, !dbg !35
  %1332 = select i1 %.not, float %1330, float %1320, !dbg !35
  %1333 = select i1 %.not36, float %1331, float %1325, !dbg !35
  %1334 = bitcast float %1333 to i32, !dbg !35
  %1335 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1195, i32 %1334), !dbg !35
  %1336 = bitcast i32 %1335 to float, !dbg !35
  %1337 = select i1 %.not36, float %1325, float %1336, !dbg !35
  %1338 = select i1 %.not36, float %1336, float %1331, !dbg !35
  %1339 = select i1 %.not36, float %1332, float %1326, !dbg !35
  %1340 = bitcast float %1339 to i32, !dbg !35
  %1341 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1195, i32 %1340), !dbg !35
  %1342 = bitcast i32 %1341 to float, !dbg !35
  %1343 = select i1 %.not36, float %1326, float %1342, !dbg !35
  %1344 = select i1 %.not36, float %1342, float %1332, !dbg !35
  %1345 = extractelement <8 x float> %1236, i64 4, !dbg !35
  %1346 = extractelement <8 x float> %1236, i64 5, !dbg !35
  %1347 = extractelement <8 x float> %1236, i64 6, !dbg !35
  %1348 = extractelement <8 x float> %1236, i64 7, !dbg !35
  %1349 = select i1 %.not, float %1346, float %1345, !dbg !35
  %1350 = bitcast float %1349 to i32, !dbg !35
  %1351 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1350), !dbg !35
  %1352 = bitcast i32 %1351 to float, !dbg !35
  %1353 = select i1 %.not, float %1345, float %1352, !dbg !35
  %1354 = select i1 %.not, float %1352, float %1346, !dbg !35
  %1355 = select i1 %.not, float %1348, float %1347, !dbg !35
  %1356 = bitcast float %1355 to i32, !dbg !35
  %1357 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1356), !dbg !35
  %1358 = bitcast i32 %1357 to float, !dbg !35
  %1359 = select i1 %.not, float %1347, float %1358, !dbg !35
  %1360 = select i1 %.not, float %1358, float %1348, !dbg !35
  %1361 = select i1 %.not36, float %1359, float %1353, !dbg !35
  %1362 = bitcast float %1361 to i32, !dbg !35
  %1363 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1195, i32 %1362), !dbg !35
  %1364 = bitcast i32 %1363 to float, !dbg !35
  %1365 = select i1 %.not36, float %1353, float %1364, !dbg !35
  %1366 = select i1 %.not36, float %1364, float %1359, !dbg !35
  %1367 = select i1 %.not36, float %1360, float %1354, !dbg !35
  %1368 = bitcast float %1367 to i32, !dbg !35
  %1369 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1195, i32 %1368), !dbg !35
  %1370 = bitcast i32 %1369 to float, !dbg !35
  %1371 = select i1 %.not36, float %1354, float %1370, !dbg !35
  %1372 = select i1 %.not36, float %1370, float %1360, !dbg !35
  %1373 = extractelement <8 x float> %1265, i64 4, !dbg !35
  %1374 = extractelement <8 x float> %1265, i64 5, !dbg !35
  %1375 = extractelement <8 x float> %1265, i64 6, !dbg !35
  %1376 = extractelement <8 x float> %1265, i64 7, !dbg !35
  %1377 = select i1 %.not, float %1374, float %1373, !dbg !35
  %1378 = bitcast float %1377 to i32, !dbg !35
  %1379 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1378), !dbg !35
  %1380 = bitcast i32 %1379 to float, !dbg !35
  %1381 = select i1 %.not, float %1373, float %1380, !dbg !35
  %1382 = select i1 %.not, float %1380, float %1374, !dbg !35
  %1383 = select i1 %.not, float %1376, float %1375, !dbg !35
  %1384 = bitcast float %1383 to i32, !dbg !35
  %1385 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1384), !dbg !35
  %1386 = bitcast i32 %1385 to float, !dbg !35
  %1387 = select i1 %.not, float %1375, float %1386, !dbg !35
  %1388 = select i1 %.not, float %1386, float %1376, !dbg !35
  %1389 = select i1 %.not36, float %1387, float %1381, !dbg !35
  %1390 = bitcast float %1389 to i32, !dbg !35
  %1391 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1195, i32 %1390), !dbg !35
  %1392 = select i1 %.not36, float %1388, float %1382, !dbg !35
  %1393 = bitcast float %1392 to i32, !dbg !35
  %1394 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1195, i32 %1393), !dbg !35
  %1395 = extractvalue [2 x <8 x float>] %128, 1, !dbg !35
  %1396 = extractelement <8 x float> %1395, i64 0, !dbg !35
  %1397 = extractelement <8 x float> %1395, i64 1, !dbg !35
  %1398 = extractelement <8 x float> %1395, i64 2, !dbg !35
  %1399 = extractelement <8 x float> %1395, i64 3, !dbg !35
  %1400 = select i1 %.not, float %1397, float %1396, !dbg !35
  %1401 = bitcast float %1400 to i32, !dbg !35
  %1402 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1401), !dbg !35
  %1403 = bitcast i32 %1402 to float, !dbg !35
  %1404 = select i1 %.not, float %1396, float %1403, !dbg !35
  %1405 = select i1 %.not, float %1403, float %1397, !dbg !35
  %1406 = select i1 %.not, float %1399, float %1398, !dbg !35
  %1407 = bitcast float %1406 to i32, !dbg !35
  %1408 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1407), !dbg !35
  %1409 = bitcast i32 %1408 to float, !dbg !35
  %1410 = select i1 %.not, float %1398, float %1409, !dbg !35
  %1411 = select i1 %.not, float %1409, float %1399, !dbg !35
  %1412 = select i1 %.not36, float %1410, float %1404, !dbg !35
  %1413 = bitcast float %1412 to i32, !dbg !35
  %1414 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1195, i32 %1413), !dbg !35
  %1415 = bitcast i32 %1414 to float, !dbg !35
  %1416 = select i1 %.not36, float %1404, float %1415, !dbg !35
  %1417 = select i1 %.not36, float %1415, float %1410, !dbg !35
  %1418 = select i1 %.not36, float %1411, float %1405, !dbg !35
  %1419 = bitcast float %1418 to i32, !dbg !35
  %1420 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1195, i32 %1419), !dbg !35
  %1421 = bitcast i32 %1420 to float, !dbg !35
  %1422 = select i1 %.not36, float %1405, float %1421, !dbg !35
  %1423 = select i1 %.not36, float %1421, float %1411, !dbg !35
  %1424 = extractvalue [2 x <8 x float>] %127, 1, !dbg !35
  %1425 = extractelement <8 x float> %1424, i64 0, !dbg !35
  %1426 = extractelement <8 x float> %1424, i64 1, !dbg !35
  %1427 = extractelement <8 x float> %1424, i64 2, !dbg !35
  %1428 = extractelement <8 x float> %1424, i64 3, !dbg !35
  %1429 = select i1 %.not, float %1426, float %1425, !dbg !35
  %1430 = bitcast float %1429 to i32, !dbg !35
  %1431 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1430), !dbg !35
  %1432 = bitcast i32 %1431 to float, !dbg !35
  %1433 = select i1 %.not, float %1425, float %1432, !dbg !35
  %1434 = select i1 %.not, float %1432, float %1426, !dbg !35
  %1435 = select i1 %.not, float %1428, float %1427, !dbg !35
  %1436 = bitcast float %1435 to i32, !dbg !35
  %1437 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1436), !dbg !35
  %1438 = bitcast i32 %1437 to float, !dbg !35
  %1439 = select i1 %.not, float %1427, float %1438, !dbg !35
  %1440 = select i1 %.not, float %1438, float %1428, !dbg !35
  %1441 = select i1 %.not36, float %1439, float %1433, !dbg !35
  %1442 = bitcast float %1441 to i32, !dbg !35
  %1443 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1195, i32 %1442), !dbg !35
  %1444 = bitcast i32 %1443 to float, !dbg !35
  %1445 = select i1 %.not36, float %1433, float %1444, !dbg !35
  %1446 = select i1 %.not36, float %1444, float %1439, !dbg !35
  %1447 = select i1 %.not36, float %1440, float %1434, !dbg !35
  %1448 = bitcast float %1447 to i32, !dbg !35
  %1449 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1195, i32 %1448), !dbg !35
  %1450 = bitcast i32 %1449 to float, !dbg !35
  %1451 = select i1 %.not36, float %1434, float %1450, !dbg !35
  %1452 = select i1 %.not36, float %1450, float %1440, !dbg !35
  %1453 = extractvalue [2 x <8 x float>] %126, 1, !dbg !35
  %1454 = extractelement <8 x float> %1453, i64 0, !dbg !35
  %1455 = extractelement <8 x float> %1453, i64 1, !dbg !35
  %1456 = extractelement <8 x float> %1453, i64 2, !dbg !35
  %1457 = extractelement <8 x float> %1453, i64 3, !dbg !35
  %1458 = select i1 %.not, float %1455, float %1454, !dbg !35
  %1459 = bitcast float %1458 to i32, !dbg !35
  %1460 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1459), !dbg !35
  %1461 = bitcast i32 %1460 to float, !dbg !35
  %1462 = select i1 %.not, float %1454, float %1461, !dbg !35
  %1463 = select i1 %.not, float %1461, float %1455, !dbg !35
  %1464 = select i1 %.not, float %1457, float %1456, !dbg !35
  %1465 = bitcast float %1464 to i32, !dbg !35
  %1466 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1465), !dbg !35
  %1467 = bitcast i32 %1466 to float, !dbg !35
  %1468 = select i1 %.not, float %1456, float %1467, !dbg !35
  %1469 = select i1 %.not, float %1467, float %1457, !dbg !35
  %1470 = select i1 %.not36, float %1468, float %1462, !dbg !35
  %1471 = bitcast float %1470 to i32, !dbg !35
  %1472 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1195, i32 %1471), !dbg !35
  %1473 = bitcast i32 %1472 to float, !dbg !35
  %1474 = select i1 %.not36, float %1462, float %1473, !dbg !35
  %1475 = select i1 %.not36, float %1473, float %1468, !dbg !35
  %1476 = select i1 %.not36, float %1469, float %1463, !dbg !35
  %1477 = bitcast float %1476 to i32, !dbg !35
  %1478 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1195, i32 %1477), !dbg !35
  %1479 = bitcast i32 %1478 to float, !dbg !35
  %1480 = select i1 %.not36, float %1463, float %1479, !dbg !35
  %1481 = select i1 %.not36, float %1479, float %1469, !dbg !35
  %1482 = extractvalue [2 x <8 x float>] %123, 1, !dbg !35
  %1483 = extractelement <8 x float> %1482, i64 0, !dbg !35
  %1484 = extractelement <8 x float> %1482, i64 1, !dbg !35
  %1485 = extractelement <8 x float> %1482, i64 2, !dbg !35
  %1486 = extractelement <8 x float> %1482, i64 3, !dbg !35
  %1487 = select i1 %.not, float %1484, float %1483, !dbg !35
  %1488 = bitcast float %1487 to i32, !dbg !35
  %1489 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1488), !dbg !35
  %1490 = bitcast i32 %1489 to float, !dbg !35
  %1491 = select i1 %.not, float %1483, float %1490, !dbg !35
  %1492 = select i1 %.not, float %1490, float %1484, !dbg !35
  %1493 = select i1 %.not, float %1486, float %1485, !dbg !35
  %1494 = bitcast float %1493 to i32, !dbg !35
  %1495 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1494), !dbg !35
  %1496 = bitcast i32 %1495 to float, !dbg !35
  %1497 = select i1 %.not, float %1485, float %1496, !dbg !35
  %1498 = select i1 %.not, float %1496, float %1486, !dbg !35
  %1499 = select i1 %.not36, float %1497, float %1491, !dbg !35
  %1500 = bitcast float %1499 to i32, !dbg !35
  %1501 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1195, i32 %1500), !dbg !35
  %1502 = select i1 %.not36, float %1498, float %1492, !dbg !35
  %1503 = bitcast float %1502 to i32, !dbg !35
  %1504 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1195, i32 %1503), !dbg !35
  %1505 = shufflevector <1 x float> %1154, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !42
  %1506 = extractelement <8 x float> %1395, i64 4, !dbg !35
  %1507 = extractelement <8 x float> %1395, i64 5, !dbg !35
  %1508 = extractelement <8 x float> %1395, i64 6, !dbg !35
  %1509 = extractelement <8 x float> %1395, i64 7, !dbg !35
  %1510 = select i1 %.not, float %1507, float %1506, !dbg !35
  %1511 = bitcast float %1510 to i32, !dbg !35
  %1512 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1511), !dbg !35
  %1513 = bitcast i32 %1512 to float, !dbg !35
  %1514 = select i1 %.not, float %1506, float %1513, !dbg !35
  %1515 = select i1 %.not, float %1513, float %1507, !dbg !35
  %1516 = select i1 %.not, float %1509, float %1508, !dbg !35
  %1517 = bitcast float %1516 to i32, !dbg !35
  %1518 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1517), !dbg !35
  %1519 = bitcast i32 %1518 to float, !dbg !35
  %1520 = select i1 %.not, float %1508, float %1519, !dbg !35
  %1521 = select i1 %.not, float %1519, float %1509, !dbg !35
  %1522 = select i1 %.not36, float %1520, float %1514, !dbg !35
  %1523 = bitcast float %1522 to i32, !dbg !35
  %1524 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1195, i32 %1523), !dbg !35
  %1525 = bitcast i32 %1524 to float, !dbg !35
  %1526 = select i1 %.not36, float %1514, float %1525, !dbg !35
  %1527 = select i1 %.not36, float %1525, float %1520, !dbg !35
  %1528 = select i1 %.not36, float %1521, float %1515, !dbg !35
  %1529 = bitcast float %1528 to i32, !dbg !35
  %1530 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1195, i32 %1529), !dbg !35
  %1531 = bitcast i32 %1530 to float, !dbg !35
  %1532 = select i1 %.not36, float %1515, float %1531, !dbg !35
  %1533 = select i1 %.not36, float %1531, float %1521, !dbg !35
  %1534 = extractelement <8 x float> %1424, i64 4, !dbg !35
  %1535 = extractelement <8 x float> %1424, i64 5, !dbg !35
  %1536 = extractelement <8 x float> %1424, i64 6, !dbg !35
  %1537 = extractelement <8 x float> %1424, i64 7, !dbg !35
  %1538 = select i1 %.not, float %1535, float %1534, !dbg !35
  %1539 = bitcast float %1538 to i32, !dbg !35
  %1540 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1539), !dbg !35
  %1541 = bitcast i32 %1540 to float, !dbg !35
  %1542 = select i1 %.not, float %1534, float %1541, !dbg !35
  %1543 = select i1 %.not, float %1541, float %1535, !dbg !35
  %1544 = select i1 %.not, float %1537, float %1536, !dbg !35
  %1545 = bitcast float %1544 to i32, !dbg !35
  %1546 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1545), !dbg !35
  %1547 = bitcast i32 %1546 to float, !dbg !35
  %1548 = select i1 %.not, float %1536, float %1547, !dbg !35
  %1549 = select i1 %.not, float %1547, float %1537, !dbg !35
  %1550 = select i1 %.not36, float %1548, float %1542, !dbg !35
  %1551 = bitcast float %1550 to i32, !dbg !35
  %1552 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1195, i32 %1551), !dbg !35
  %1553 = bitcast i32 %1552 to float, !dbg !35
  %1554 = select i1 %.not36, float %1542, float %1553, !dbg !35
  %1555 = select i1 %.not36, float %1553, float %1548, !dbg !35
  %1556 = select i1 %.not36, float %1549, float %1543, !dbg !35
  %1557 = bitcast float %1556 to i32, !dbg !35
  %1558 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1195, i32 %1557), !dbg !35
  %1559 = bitcast i32 %1558 to float, !dbg !35
  %1560 = select i1 %.not36, float %1543, float %1559, !dbg !35
  %1561 = select i1 %.not36, float %1559, float %1549, !dbg !35
  %1562 = extractelement <8 x float> %1453, i64 4, !dbg !35
  %1563 = extractelement <8 x float> %1453, i64 5, !dbg !35
  %1564 = extractelement <8 x float> %1453, i64 6, !dbg !35
  %1565 = extractelement <8 x float> %1453, i64 7, !dbg !35
  %1566 = select i1 %.not, float %1563, float %1562, !dbg !35
  %1567 = bitcast float %1566 to i32, !dbg !35
  %1568 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1567), !dbg !35
  %1569 = bitcast i32 %1568 to float, !dbg !35
  %1570 = select i1 %.not, float %1562, float %1569, !dbg !35
  %1571 = select i1 %.not, float %1569, float %1563, !dbg !35
  %1572 = select i1 %.not, float %1565, float %1564, !dbg !35
  %1573 = bitcast float %1572 to i32, !dbg !35
  %1574 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1573), !dbg !35
  %1575 = bitcast i32 %1574 to float, !dbg !35
  %1576 = select i1 %.not, float %1564, float %1575, !dbg !35
  %1577 = select i1 %.not, float %1575, float %1565, !dbg !35
  %1578 = select i1 %.not36, float %1576, float %1570, !dbg !35
  %1579 = bitcast float %1578 to i32, !dbg !35
  %1580 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1195, i32 %1579), !dbg !35
  %1581 = bitcast i32 %1580 to float, !dbg !35
  %1582 = select i1 %.not36, float %1570, float %1581, !dbg !35
  %1583 = select i1 %.not36, float %1581, float %1576, !dbg !35
  %1584 = select i1 %.not36, float %1577, float %1571, !dbg !35
  %1585 = bitcast float %1584 to i32, !dbg !35
  %1586 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1195, i32 %1585), !dbg !35
  %1587 = bitcast i32 %1586 to float, !dbg !35
  %1588 = select i1 %.not36, float %1571, float %1587, !dbg !35
  %1589 = select i1 %.not36, float %1587, float %1577, !dbg !35
  %1590 = extractelement <8 x float> %1482, i64 4, !dbg !35
  %1591 = extractelement <8 x float> %1482, i64 5, !dbg !35
  %1592 = extractelement <8 x float> %1482, i64 6, !dbg !35
  %1593 = extractelement <8 x float> %1482, i64 7, !dbg !35
  %1594 = select i1 %.not, float %1591, float %1590, !dbg !35
  %1595 = bitcast float %1594 to i32, !dbg !35
  %1596 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1595), !dbg !35
  %1597 = bitcast i32 %1596 to float, !dbg !35
  %1598 = select i1 %.not, float %1590, float %1597, !dbg !35
  %1599 = select i1 %.not, float %1597, float %1591, !dbg !35
  %1600 = select i1 %.not, float %1593, float %1592, !dbg !35
  %1601 = bitcast float %1600 to i32, !dbg !35
  %1602 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1179, i32 %1601), !dbg !35
  %1603 = bitcast i32 %1602 to float, !dbg !35
  %1604 = select i1 %.not, float %1592, float %1603, !dbg !35
  %1605 = select i1 %.not, float %1603, float %1593, !dbg !35
  %1606 = select i1 %.not36, float %1604, float %1598, !dbg !35
  %1607 = bitcast float %1606 to i32, !dbg !35
  %1608 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1195, i32 %1607), !dbg !35
  %1609 = select i1 %.not36, float %1605, float %1599, !dbg !35
  %1610 = bitcast float %1609 to i32, !dbg !35
  %1611 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1195, i32 %1610), !dbg !35
  %1612 = bitcast i32 %1608 to float, !dbg !35
  %1613 = select i1 %.not36, float %1598, float %1612, !dbg !35
  %1614 = insertelement <4 x float> poison, float %1613, i64 0, !dbg !42
  %1615 = bitcast i32 %1611 to float, !dbg !35
  %1616 = select i1 %.not36, float %1599, float %1615, !dbg !35
  %1617 = insertelement <4 x float> %1614, float %1616, i64 1, !dbg !42
  %1618 = select i1 %.not36, float %1612, float %1604, !dbg !35
  %1619 = insertelement <4 x float> %1617, float %1618, i64 2, !dbg !42
  %1620 = select i1 %.not36, float %1615, float %1605, !dbg !35
  %1621 = insertelement <4 x float> %1619, float %1620, i64 3, !dbg !42
  %row.reciprocal = fdiv arcp <4 x float> splat (float 1.000000e+00), %1505, !dbg !42
  %normalized = fmul <4 x float> %1621, %row.reciprocal, !dbg !42
  %1622 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !43
  %1623 = extractelement <4 x half> %1622, i64 3, !dbg !43
  %1624 = extractelement <4 x half> %1622, i64 2, !dbg !43
  %1625 = extractelement <4 x half> %1622, i64 1, !dbg !43
  %1626 = extractelement <4 x half> %1622, i64 0, !dbg !43
  %1627 = bitcast i32 %1501 to float, !dbg !35
  %1628 = select i1 %.not36, float %1491, float %1627, !dbg !35
  %1629 = insertelement <4 x float> poison, float %1628, i64 0, !dbg !42
  %1630 = bitcast i32 %1504 to float, !dbg !35
  %1631 = select i1 %.not36, float %1492, float %1630, !dbg !35
  %1632 = insertelement <4 x float> %1629, float %1631, i64 1, !dbg !42
  %1633 = select i1 %.not36, float %1627, float %1497, !dbg !35
  %1634 = insertelement <4 x float> %1632, float %1633, i64 2, !dbg !42
  %1635 = select i1 %.not36, float %1630, float %1498, !dbg !35
  %1636 = insertelement <4 x float> %1634, float %1635, i64 3, !dbg !42
  %normalized37 = fmul <4 x float> %1636, %row.reciprocal, !dbg !42
  %1637 = fptrunc <4 x float> %normalized37 to <4 x half>, !dbg !43
  %1638 = extractelement <4 x half> %1637, i64 3, !dbg !43
  %1639 = extractelement <4 x half> %1637, i64 2, !dbg !43
  %1640 = extractelement <4 x half> %1637, i64 1, !dbg !43
  %1641 = extractelement <4 x half> %1637, i64 0, !dbg !43
  %1642 = bitcast i32 %1391 to float, !dbg !35
  %1643 = select i1 %.not36, float %1381, float %1642, !dbg !35
  %1644 = insertelement <4 x float> poison, float %1643, i64 0, !dbg !42
  %1645 = bitcast i32 %1394 to float, !dbg !35
  %1646 = select i1 %.not36, float %1382, float %1645, !dbg !35
  %1647 = insertelement <4 x float> %1644, float %1646, i64 1, !dbg !42
  %1648 = select i1 %.not36, float %1642, float %1387, !dbg !35
  %1649 = insertelement <4 x float> %1647, float %1648, i64 2, !dbg !42
  %1650 = select i1 %.not36, float %1645, float %1388, !dbg !35
  %1651 = insertelement <4 x float> %1649, float %1650, i64 3, !dbg !42
  %row.reciprocal38 = fdiv arcp <4 x float> splat (float 1.000000e+00), %1288, !dbg !42
  %normalized39 = fmul <4 x float> %1651, %row.reciprocal38, !dbg !42
  %1652 = fptrunc <4 x float> %normalized39 to <4 x half>, !dbg !43
  %1653 = extractelement <4 x half> %1652, i64 3, !dbg !43
  %1654 = extractelement <4 x half> %1652, i64 2, !dbg !43
  %1655 = extractelement <4 x half> %1652, i64 1, !dbg !43
  %1656 = extractelement <4 x half> %1652, i64 0, !dbg !43
  %1657 = bitcast i32 %1284 to float, !dbg !35
  %1658 = select i1 %.not36, float %1274, float %1657, !dbg !35
  %1659 = insertelement <4 x float> poison, float %1658, i64 0, !dbg !42
  %1660 = bitcast i32 %1287 to float, !dbg !35
  %1661 = select i1 %.not36, float %1275, float %1660, !dbg !35
  %1662 = insertelement <4 x float> %1659, float %1661, i64 1, !dbg !42
  %1663 = select i1 %.not36, float %1657, float %1280, !dbg !35
  %1664 = insertelement <4 x float> %1662, float %1663, i64 2, !dbg !42
  %1665 = select i1 %.not36, float %1660, float %1281, !dbg !35
  %1666 = insertelement <4 x float> %1664, float %1665, i64 3, !dbg !42
  %normalized40 = fmul <4 x float> %1666, %row.reciprocal38, !dbg !42
  %1667 = fptrunc <4 x float> %normalized40 to <4 x half>, !dbg !43
  %1668 = extractelement <4 x half> %1667, i64 3, !dbg !43
  %1669 = extractelement <4 x half> %1667, i64 2, !dbg !43
  %1670 = extractelement <4 x half> %1667, i64 1, !dbg !43
  %1671 = extractelement <4 x half> %1667, i64 0, !dbg !43
  %1672 = add nuw nsw i64 %17, %18, !dbg !44
  %1673 = shl i64 %1672, 19, !dbg !44
  %1674 = and i64 %1673, 16252928, !dbg !44
  %address.thread974 = call i32 asm sideeffect "", "=v,0"(i32 %27) #9, !dbg !44, !range !57, !frisk.register_lifetimes !30
  %address.local975 = zext nneg i32 %address.thread974 to i64, !dbg !36
  %address.local976 = add nuw i64 %21, %address.local975, !dbg !44
  %1675 = lshr i64 %address.local975, 2, !dbg !44
  %address.local978 = and i64 %1675, 48, !dbg !44
  %address.local979 = add i64 %address.local976, %address.local978, !dbg !44
  %address.local980 = and i64 %address.local975, 240, !dbg !44
  %address.local981 = sub i64 %address.local979, %address.local980, !dbg !44
  %address.local982 = add i64 %address.local981, %22, !dbg !44
  %address.local983 = shl nuw nsw i64 %address.local982, 7, !dbg !44
  %address.local984 = add nuw nsw i64 %1674, %address.local983, !dbg !44
  %address.local985 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local984, !dbg !44
  %address.local986 = lshr i64 %address.local975, 2, !dbg !37
  %address.local987 = and i64 %address.local986, 12, !dbg !37
  %address.local988 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local987, !dbg !44
  store half %1671, ptr addrspace(1) %address.local988, align 2, !dbg !44
  %address.local989 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local984, !dbg !44
  %address.local990 = or disjoint i64 %address.local987, 1, !dbg !44
  %address.local991 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local989, i64 %address.local990, !dbg !44
  store half %1670, ptr addrspace(1) %address.local991, align 2, !dbg !44
  %address.local992 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local984, !dbg !44
  %address.local993 = or disjoint i64 %address.local987, 2, !dbg !44
  %address.local994 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local992, i64 %address.local993, !dbg !44
  store half %1669, ptr addrspace(1) %address.local994, align 2, !dbg !44
  %address.local995 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local984, !dbg !44
  %address.local996 = or disjoint i64 %address.local987, 3, !dbg !44
  %address.local997 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local995, i64 %address.local996, !dbg !44
  store half %1668, ptr addrspace(1) %address.local997, align 2, !dbg !44
  %address.local998 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local984, !dbg !44
  %address.local999 = or disjoint i64 %address.local987, 16, !dbg !44
  %address.local1000 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local998, i64 %address.local999, !dbg !44
  store half %1656, ptr addrspace(1) %address.local1000, align 2, !dbg !44
  %address.local1001 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local984, !dbg !44
  %address.local1002 = or disjoint i64 %address.local987, 17, !dbg !44
  %address.local1003 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1001, i64 %address.local1002, !dbg !44
  store half %1655, ptr addrspace(1) %address.local1003, align 2, !dbg !44
  %address.local1004 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local984, !dbg !44
  %address.local1005 = or disjoint i64 %address.local987, 18, !dbg !44
  %address.local1006 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1004, i64 %address.local1005, !dbg !44
  store half %1654, ptr addrspace(1) %address.local1006, align 2, !dbg !44
  %address.local1007 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local984, !dbg !44
  %address.local1008 = or disjoint i64 %address.local987, 19, !dbg !44
  %address.local1009 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1007, i64 %address.local1008, !dbg !44
  store half %1653, ptr addrspace(1) %address.local1009, align 2, !dbg !44
  %address.local1010 = shl i64 %address.local982, 7, !dbg !44
  %address.local1011 = add i64 %address.local1010, 8192, !dbg !44
  %address.local1012 = add nuw nsw i64 %1674, %address.local1011, !dbg !44
  %address.local1013 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1012, !dbg !44
  %address.local1014 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local987, !dbg !44
  store half %1641, ptr addrspace(1) %address.local1014, align 2, !dbg !44
  %address.local1015 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1012, !dbg !44
  %address.local1016 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1015, i64 %address.local990, !dbg !44
  store half %1640, ptr addrspace(1) %address.local1016, align 2, !dbg !44
  %address.local1017 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1012, !dbg !44
  %address.local1018 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1017, i64 %address.local993, !dbg !44
  store half %1639, ptr addrspace(1) %address.local1018, align 2, !dbg !44
  %address.local1019 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1012, !dbg !44
  %address.local1020 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1019, i64 %address.local996, !dbg !44
  store half %1638, ptr addrspace(1) %address.local1020, align 2, !dbg !44
  %address.local1021 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1012, !dbg !44
  %address.local1022 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1021, i64 %address.local999, !dbg !44
  store half %1626, ptr addrspace(1) %address.local1022, align 2, !dbg !44
  %address.local1023 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1012, !dbg !44
  %address.local1024 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1023, i64 %address.local1002, !dbg !44
  store half %1625, ptr addrspace(1) %address.local1024, align 2, !dbg !44
  %address.local1025 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1012, !dbg !44
  %address.local1026 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1025, i64 %address.local1005, !dbg !44
  store half %1624, ptr addrspace(1) %address.local1026, align 2, !dbg !44
  %address.local1027 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1012, !dbg !44
  %address.local1028 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1027, i64 %address.local1008, !dbg !44
  store half %1623, ptr addrspace(1) %address.local1028, align 2, !dbg !44
  %1676 = insertelement <4 x float> poison, float %1582, i64 0, !dbg !42
  %1677 = insertelement <4 x float> %1676, float %1588, i64 1, !dbg !42
  %1678 = insertelement <4 x float> %1677, float %1583, i64 2, !dbg !42
  %1679 = insertelement <4 x float> %1678, float %1589, i64 3, !dbg !42
  %row.reciprocal41 = fdiv arcp <4 x float> splat (float 1.000000e+00), %1505, !dbg !42
  %normalized42 = fmul <4 x float> %1679, %row.reciprocal41, !dbg !42
  %1680 = fptrunc <4 x float> %normalized42 to <4 x half>, !dbg !43
  %1681 = extractelement <4 x half> %1680, i64 3, !dbg !43
  %1682 = extractelement <4 x half> %1680, i64 2, !dbg !43
  %1683 = extractelement <4 x half> %1680, i64 1, !dbg !43
  %1684 = extractelement <4 x half> %1680, i64 0, !dbg !43
  %1685 = insertelement <4 x float> poison, float %1474, i64 0, !dbg !42
  %1686 = insertelement <4 x float> %1685, float %1480, i64 1, !dbg !42
  %1687 = insertelement <4 x float> %1686, float %1475, i64 2, !dbg !42
  %1688 = insertelement <4 x float> %1687, float %1481, i64 3, !dbg !42
  %normalized43 = fmul <4 x float> %1688, %row.reciprocal41, !dbg !42
  %1689 = fptrunc <4 x float> %normalized43 to <4 x half>, !dbg !43
  %1690 = extractelement <4 x half> %1689, i64 3, !dbg !43
  %1691 = extractelement <4 x half> %1689, i64 2, !dbg !43
  %1692 = extractelement <4 x half> %1689, i64 1, !dbg !43
  %1693 = extractelement <4 x half> %1689, i64 0, !dbg !43
  %1694 = insertelement <4 x float> poison, float %1365, i64 0, !dbg !42
  %1695 = insertelement <4 x float> %1694, float %1371, i64 1, !dbg !42
  %1696 = insertelement <4 x float> %1695, float %1366, i64 2, !dbg !42
  %1697 = insertelement <4 x float> %1696, float %1372, i64 3, !dbg !42
  %row.reciprocal44 = fdiv arcp <4 x float> splat (float 1.000000e+00), %1288, !dbg !42
  %normalized45 = fmul <4 x float> %1697, %row.reciprocal44, !dbg !42
  %1698 = fptrunc <4 x float> %normalized45 to <4 x half>, !dbg !43
  %1699 = extractelement <4 x half> %1698, i64 3, !dbg !43
  %1700 = extractelement <4 x half> %1698, i64 2, !dbg !43
  %1701 = extractelement <4 x half> %1698, i64 1, !dbg !43
  %1702 = extractelement <4 x half> %1698, i64 0, !dbg !43
  %1703 = insertelement <4 x float> poison, float %1257, i64 0, !dbg !42
  %1704 = insertelement <4 x float> %1703, float %1263, i64 1, !dbg !42
  %1705 = insertelement <4 x float> %1704, float %1258, i64 2, !dbg !42
  %1706 = insertelement <4 x float> %1705, float %1264, i64 3, !dbg !42
  %normalized46 = fmul <4 x float> %1706, %row.reciprocal44, !dbg !42
  %1707 = fptrunc <4 x float> %normalized46 to <4 x half>, !dbg !43
  %1708 = extractelement <4 x half> %1707, i64 3, !dbg !43
  %1709 = extractelement <4 x half> %1707, i64 2, !dbg !43
  %1710 = extractelement <4 x half> %1707, i64 1, !dbg !43
  %1711 = extractelement <4 x half> %1707, i64 0, !dbg !43
  %1712 = shl nuw nsw i64 %20, 19, !dbg !44
  %address.thread1029 = call i32 asm sideeffect "", "=v,0"(i32 %27) #9, !dbg !44, !range !57, !frisk.register_lifetimes !30
  %address.local1030 = zext nneg i32 %address.thread1029 to i64, !dbg !36
  %address.local1031 = add nuw i64 %21, %address.local1030, !dbg !44
  %1713 = lshr i64 %address.local1030, 2, !dbg !44
  %address.local1033 = and i64 %1713, 48, !dbg !44
  %address.local1034 = add i64 %address.local1031, %address.local1033, !dbg !44
  %address.local1035 = and i64 %address.local1030, 240, !dbg !44
  %address.local1036 = sub i64 %address.local1034, %address.local1035, !dbg !44
  %address.local1037 = add i64 %address.local1036, %22, !dbg !44
  %address.local1038 = shl nuw nsw i64 %address.local1037, 7, !dbg !44
  %address.local1039 = add nuw nsw i64 %1712, %address.local1038, !dbg !44
  %address.local1040 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1039, !dbg !44
  %address.local1041 = lshr i64 %address.local1030, 2, !dbg !37
  %address.local1042 = and i64 %address.local1041, 12, !dbg !37
  %address.local1043 = or disjoint i64 %address.local1042, 32, !dbg !44
  %address.local1044 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1040, i64 %address.local1043, !dbg !44
  store half %1711, ptr addrspace(1) %address.local1044, align 2, !dbg !44
  %address.local1045 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1039, !dbg !44
  %address.local1046 = or disjoint i64 %address.local1042, 33, !dbg !44
  %address.local1047 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1045, i64 %address.local1046, !dbg !44
  store half %1710, ptr addrspace(1) %address.local1047, align 2, !dbg !44
  %address.local1048 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1039, !dbg !44
  %address.local1049 = or disjoint i64 %address.local1042, 34, !dbg !44
  %address.local1050 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1048, i64 %address.local1049, !dbg !44
  store half %1709, ptr addrspace(1) %address.local1050, align 2, !dbg !44
  %address.local1051 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1039, !dbg !44
  %address.local1052 = or disjoint i64 %address.local1042, 35, !dbg !44
  %address.local1053 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1051, i64 %address.local1052, !dbg !44
  store half %1708, ptr addrspace(1) %address.local1053, align 2, !dbg !44
  %address.local1054 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1039, !dbg !44
  %address.local1055 = or disjoint i64 %address.local1042, 48, !dbg !44
  %address.local1056 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1055, !dbg !44
  store half %1702, ptr addrspace(1) %address.local1056, align 2, !dbg !44
  %address.local1057 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1039, !dbg !44
  %address.local1058 = or disjoint i64 %address.local1042, 49, !dbg !44
  %address.local1059 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1057, i64 %address.local1058, !dbg !44
  store half %1701, ptr addrspace(1) %address.local1059, align 2, !dbg !44
  %address.local1060 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1039, !dbg !44
  %address.local1061 = or disjoint i64 %address.local1042, 50, !dbg !44
  %address.local1062 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1060, i64 %address.local1061, !dbg !44
  store half %1700, ptr addrspace(1) %address.local1062, align 2, !dbg !44
  %address.local1063 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1039, !dbg !44
  %address.local1064 = or i64 %address.local1041, 51, !dbg !44
  %address.local1065 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1063, i64 %address.local1064, !dbg !44
  store half %1699, ptr addrspace(1) %address.local1065, align 2, !dbg !44
  %address.local1066 = shl i64 %address.local1037, 7, !dbg !44
  %address.local1067 = add i64 %address.local1066, 8192, !dbg !44
  %address.local1068 = add nuw nsw i64 %1712, %address.local1067, !dbg !44
  %address.local1069 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1068, !dbg !44
  %address.local1070 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1069, i64 %address.local1043, !dbg !44
  store half %1693, ptr addrspace(1) %address.local1070, align 2, !dbg !44
  %address.local1071 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1068, !dbg !44
  %address.local1072 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1071, i64 %address.local1046, !dbg !44
  store half %1692, ptr addrspace(1) %address.local1072, align 2, !dbg !44
  %address.local1073 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1068, !dbg !44
  %address.local1074 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1073, i64 %address.local1049, !dbg !44
  store half %1691, ptr addrspace(1) %address.local1074, align 2, !dbg !44
  %address.local1075 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1068, !dbg !44
  %address.local1076 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1075, i64 %address.local1052, !dbg !44
  store half %1690, ptr addrspace(1) %address.local1076, align 2, !dbg !44
  %address.local1077 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1068, !dbg !44
  %address.local1078 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1077, i64 %address.local1055, !dbg !44
  store half %1684, ptr addrspace(1) %address.local1078, align 2, !dbg !44
  %address.local1079 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1068, !dbg !44
  %address.local1080 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1079, i64 %address.local1058, !dbg !44
  store half %1683, ptr addrspace(1) %address.local1080, align 2, !dbg !44
  %address.local1081 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1068, !dbg !44
  %address.local1082 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1061, !dbg !44
  store half %1682, ptr addrspace(1) %address.local1082, align 2, !dbg !44
  %address.local1083 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1068, !dbg !44
  %address.local1084 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1083, i64 %address.local1064, !dbg !44
  store half %1681, ptr addrspace(1) %address.local1084, align 2, !dbg !44
  %1714 = insertelement <4 x float> poison, float %1554, i64 0, !dbg !42
  %1715 = insertelement <4 x float> %1714, float %1560, i64 1, !dbg !42
  %1716 = insertelement <4 x float> %1715, float %1555, i64 2, !dbg !42
  %1717 = insertelement <4 x float> %1716, float %1561, i64 3, !dbg !42
  %row.reciprocal47 = fdiv arcp <4 x float> splat (float 1.000000e+00), %1505, !dbg !42
  %normalized48 = fmul <4 x float> %1717, %row.reciprocal47, !dbg !42
  %1718 = fptrunc <4 x float> %normalized48 to <4 x half>, !dbg !43
  %1719 = extractelement <4 x half> %1718, i64 3, !dbg !43
  %1720 = extractelement <4 x half> %1718, i64 2, !dbg !43
  %1721 = extractelement <4 x half> %1718, i64 1, !dbg !43
  %1722 = extractelement <4 x half> %1718, i64 0, !dbg !43
  %1723 = insertelement <4 x float> poison, float %1445, i64 0, !dbg !42
  %1724 = insertelement <4 x float> %1723, float %1451, i64 1, !dbg !42
  %1725 = insertelement <4 x float> %1724, float %1446, i64 2, !dbg !42
  %1726 = insertelement <4 x float> %1725, float %1452, i64 3, !dbg !42
  %normalized49 = fmul <4 x float> %1726, %row.reciprocal47, !dbg !42
  %1727 = fptrunc <4 x float> %normalized49 to <4 x half>, !dbg !43
  %1728 = extractelement <4 x half> %1727, i64 3, !dbg !43
  %1729 = extractelement <4 x half> %1727, i64 2, !dbg !43
  %1730 = extractelement <4 x half> %1727, i64 1, !dbg !43
  %1731 = extractelement <4 x half> %1727, i64 0, !dbg !43
  %1732 = insertelement <4 x float> poison, float %1337, i64 0, !dbg !42
  %1733 = insertelement <4 x float> %1732, float %1343, i64 1, !dbg !42
  %1734 = insertelement <4 x float> %1733, float %1338, i64 2, !dbg !42
  %1735 = insertelement <4 x float> %1734, float %1344, i64 3, !dbg !42
  %row.reciprocal50 = fdiv arcp <4 x float> splat (float 1.000000e+00), %1288, !dbg !42
  %normalized51 = fmul <4 x float> %1735, %row.reciprocal50, !dbg !42
  %1736 = fptrunc <4 x float> %normalized51 to <4 x half>, !dbg !43
  %1737 = extractelement <4 x half> %1736, i64 3, !dbg !43
  %1738 = extractelement <4 x half> %1736, i64 2, !dbg !43
  %1739 = extractelement <4 x half> %1736, i64 1, !dbg !43
  %1740 = extractelement <4 x half> %1736, i64 0, !dbg !43
  %1741 = insertelement <4 x float> poison, float %1228, i64 0, !dbg !42
  %1742 = insertelement <4 x float> %1741, float %1234, i64 1, !dbg !42
  %1743 = insertelement <4 x float> %1742, float %1229, i64 2, !dbg !42
  %1744 = insertelement <4 x float> %1743, float %1235, i64 3, !dbg !42
  %normalized52 = fmul <4 x float> %1744, %row.reciprocal50, !dbg !42
  %1745 = fptrunc <4 x float> %normalized52 to <4 x half>, !dbg !43
  %1746 = extractelement <4 x half> %1745, i64 3, !dbg !43
  %1747 = extractelement <4 x half> %1745, i64 2, !dbg !43
  %1748 = extractelement <4 x half> %1745, i64 1, !dbg !43
  %1749 = extractelement <4 x half> %1745, i64 0, !dbg !43
  %1750 = shl nuw nsw i64 %20, 19, !dbg !44
  %address.thread1085 = call i32 asm sideeffect "", "=v,0"(i32 %27) #9, !dbg !44, !range !57, !frisk.register_lifetimes !30
  %address.local1086 = zext nneg i32 %address.thread1085 to i64, !dbg !36
  %address.local1087 = add nuw i64 %21, %address.local1086, !dbg !44
  %1751 = lshr i64 %address.local1086, 2, !dbg !44
  %address.local1089 = and i64 %1751, 48, !dbg !44
  %address.local1090 = add i64 %address.local1087, %address.local1089, !dbg !44
  %address.local1091 = and i64 %address.local1086, 240, !dbg !44
  %address.local1092 = sub i64 %address.local1090, %address.local1091, !dbg !44
  %address.local1093 = add i64 %address.local1092, %22, !dbg !44
  %address.local1094 = shl nuw nsw i64 %address.local1093, 7, !dbg !44
  %address.local1095 = add nuw nsw i64 %1750, %address.local1094, !dbg !44
  %address.local1096 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1095, !dbg !44
  %address.local1097 = lshr i64 %address.local1086, 2, !dbg !37
  %address.local1098 = and i64 %address.local1097, 12, !dbg !37
  %address.local1099 = or disjoint i64 %address.local1098, 64, !dbg !44
  %address.local1100 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1096, i64 %address.local1099, !dbg !44
  store half %1749, ptr addrspace(1) %address.local1100, align 2, !dbg !44
  %address.local1101 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1095, !dbg !44
  %address.local1102 = or disjoint i64 %address.local1098, 65, !dbg !44
  %address.local1103 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1101, i64 %address.local1102, !dbg !44
  store half %1748, ptr addrspace(1) %address.local1103, align 2, !dbg !44
  %address.local1104 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1095, !dbg !44
  %address.local1105 = or disjoint i64 %address.local1098, 66, !dbg !44
  %address.local1106 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1104, i64 %address.local1105, !dbg !44
  store half %1747, ptr addrspace(1) %address.local1106, align 2, !dbg !44
  %address.local1107 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1095, !dbg !44
  %address.local1108 = or disjoint i64 %address.local1098, 67, !dbg !44
  %address.local1109 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1107, i64 %address.local1108, !dbg !44
  store half %1746, ptr addrspace(1) %address.local1109, align 2, !dbg !44
  %address.local1110 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1095, !dbg !44
  %address.local1111 = or disjoint i64 %address.local1098, 80, !dbg !44
  %address.local1112 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1110, i64 %address.local1111, !dbg !44
  store half %1740, ptr addrspace(1) %address.local1112, align 2, !dbg !44
  %address.local1113 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1095, !dbg !44
  %address.local1114 = or disjoint i64 %address.local1098, 81, !dbg !44
  %address.local1115 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1113, i64 %address.local1114, !dbg !44
  store half %1739, ptr addrspace(1) %address.local1115, align 2, !dbg !44
  %address.local1116 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1095, !dbg !44
  %address.local1117 = or disjoint i64 %address.local1098, 82, !dbg !44
  %address.local1118 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1116, i64 %address.local1117, !dbg !44
  store half %1738, ptr addrspace(1) %address.local1118, align 2, !dbg !44
  %address.local1119 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1095, !dbg !44
  %address.local1120 = or disjoint i64 %address.local1098, 83, !dbg !44
  %address.local1121 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1119, i64 %address.local1120, !dbg !44
  store half %1737, ptr addrspace(1) %address.local1121, align 2, !dbg !44
  %address.local1122 = shl i64 %address.local1093, 7, !dbg !44
  %address.local1123 = add i64 %address.local1122, 8192, !dbg !44
  %address.local1124 = add nuw nsw i64 %1750, %address.local1123, !dbg !44
  %address.local1125 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1124, !dbg !44
  %address.local1126 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1125, i64 %address.local1099, !dbg !44
  store half %1731, ptr addrspace(1) %address.local1126, align 2, !dbg !44
  %address.local1127 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1124, !dbg !44
  %address.local1128 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1127, i64 %address.local1102, !dbg !44
  store half %1730, ptr addrspace(1) %address.local1128, align 2, !dbg !44
  %address.local1129 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1124, !dbg !44
  %address.local1130 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1129, i64 %address.local1105, !dbg !44
  store half %1729, ptr addrspace(1) %address.local1130, align 2, !dbg !44
  %address.local1131 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1124, !dbg !44
  %address.local1132 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1131, i64 %address.local1108, !dbg !44
  store half %1728, ptr addrspace(1) %address.local1132, align 2, !dbg !44
  %address.local1133 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1124, !dbg !44
  %address.local1134 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1133, i64 %address.local1111, !dbg !44
  store half %1722, ptr addrspace(1) %address.local1134, align 2, !dbg !44
  %address.local1135 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1124, !dbg !44
  %address.local1136 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1135, i64 %address.local1114, !dbg !44
  store half %1721, ptr addrspace(1) %address.local1136, align 2, !dbg !44
  %address.local1137 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1124, !dbg !44
  %address.local1138 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1137, i64 %address.local1117, !dbg !44
  store half %1720, ptr addrspace(1) %address.local1138, align 2, !dbg !44
  %address.local1139 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1124, !dbg !44
  %address.local1140 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1139, i64 %address.local1120, !dbg !44
  store half %1719, ptr addrspace(1) %address.local1140, align 2, !dbg !44
  %1752 = insertelement <4 x float> poison, float %1526, i64 0, !dbg !42
  %1753 = insertelement <4 x float> %1752, float %1532, i64 1, !dbg !42
  %1754 = insertelement <4 x float> %1753, float %1527, i64 2, !dbg !42
  %1755 = insertelement <4 x float> %1754, float %1533, i64 3, !dbg !42
  %row.reciprocal53 = fdiv arcp <4 x float> splat (float 1.000000e+00), %1505, !dbg !42
  %normalized54 = fmul <4 x float> %1755, %row.reciprocal53, !dbg !42
  %1756 = fptrunc <4 x float> %normalized54 to <4 x half>, !dbg !43
  %1757 = extractelement <4 x half> %1756, i64 3, !dbg !43
  %1758 = extractelement <4 x half> %1756, i64 2, !dbg !43
  %1759 = extractelement <4 x half> %1756, i64 1, !dbg !43
  %1760 = extractelement <4 x half> %1756, i64 0, !dbg !43
  %1761 = insertelement <4 x float> poison, float %1416, i64 0, !dbg !42
  %1762 = insertelement <4 x float> %1761, float %1422, i64 1, !dbg !42
  %1763 = insertelement <4 x float> %1762, float %1417, i64 2, !dbg !42
  %1764 = insertelement <4 x float> %1763, float %1423, i64 3, !dbg !42
  %normalized55 = fmul <4 x float> %1764, %row.reciprocal53, !dbg !42
  %1765 = fptrunc <4 x float> %normalized55 to <4 x half>, !dbg !43
  %1766 = extractelement <4 x half> %1765, i64 3, !dbg !43
  %1767 = extractelement <4 x half> %1765, i64 2, !dbg !43
  %1768 = extractelement <4 x half> %1765, i64 1, !dbg !43
  %1769 = extractelement <4 x half> %1765, i64 0, !dbg !43
  %1770 = insertelement <4 x float> poison, float %1309, i64 0, !dbg !42
  %1771 = insertelement <4 x float> %1770, float %1315, i64 1, !dbg !42
  %1772 = insertelement <4 x float> %1771, float %1310, i64 2, !dbg !42
  %1773 = insertelement <4 x float> %1772, float %1316, i64 3, !dbg !42
  %row.reciprocal56 = fdiv arcp <4 x float> splat (float 1.000000e+00), %1288, !dbg !42
  %normalized57 = fmul <4 x float> %1773, %row.reciprocal56, !dbg !42
  %1774 = fptrunc <4 x float> %normalized57 to <4 x half>, !dbg !43
  %1775 = extractelement <4 x half> %1774, i64 3, !dbg !43
  %1776 = extractelement <4 x half> %1774, i64 2, !dbg !43
  %1777 = extractelement <4 x half> %1774, i64 1, !dbg !43
  %1778 = extractelement <4 x half> %1774, i64 0, !dbg !43
  %1779 = insertelement <4 x float> poison, float %1199, i64 0, !dbg !42
  %1780 = insertelement <4 x float> %1779, float %1205, i64 1, !dbg !42
  %1781 = insertelement <4 x float> %1780, float %1200, i64 2, !dbg !42
  %1782 = insertelement <4 x float> %1781, float %1206, i64 3, !dbg !42
  %normalized58 = fmul <4 x float> %1782, %row.reciprocal56, !dbg !42
  %1783 = fptrunc <4 x float> %normalized58 to <4 x half>, !dbg !43
  %1784 = extractelement <4 x half> %1783, i64 3, !dbg !43
  %1785 = extractelement <4 x half> %1783, i64 2, !dbg !43
  %1786 = extractelement <4 x half> %1783, i64 1, !dbg !43
  %1787 = extractelement <4 x half> %1783, i64 0, !dbg !43
  %1788 = shl nuw nsw i64 %20, 19, !dbg !44
  %address.thread1141 = call i32 asm sideeffect "", "=v,0"(i32 %27) #9, !dbg !44, !range !57, !frisk.register_lifetimes !30
  %address.local1142 = zext nneg i32 %address.thread1141 to i64, !dbg !36
  %address.local1143 = add nuw i64 %21, %address.local1142, !dbg !44
  %1789 = lshr i64 %address.local1142, 2, !dbg !44
  %address.local1145 = and i64 %1789, 48, !dbg !44
  %address.local1146 = add i64 %address.local1143, %address.local1145, !dbg !44
  %address.local1147 = and i64 %address.local1142, 240, !dbg !44
  %address.local1148 = sub i64 %address.local1146, %address.local1147, !dbg !44
  %address.local1149 = add i64 %address.local1148, %22, !dbg !44
  %address.local1150 = shl nuw nsw i64 %address.local1149, 7, !dbg !44
  %address.local1151 = add nuw nsw i64 %1788, %address.local1150, !dbg !44
  %address.local1152 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1151, !dbg !44
  %address.local1153 = lshr i64 %address.local1142, 2, !dbg !37
  %address.local1154 = and i64 %address.local1153, 12, !dbg !37
  %address.local1155 = or disjoint i64 %address.local1154, 96, !dbg !44
  %address.local1156 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1152, i64 %address.local1155, !dbg !44
  store half %1787, ptr addrspace(1) %address.local1156, align 2, !dbg !44
  %address.local1157 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1151, !dbg !44
  %address.local1158 = or disjoint i64 %address.local1154, 97, !dbg !44
  %address.local1159 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1157, i64 %address.local1158, !dbg !44
  store half %1786, ptr addrspace(1) %address.local1159, align 2, !dbg !44
  %address.local1160 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1151, !dbg !44
  %address.local1161 = or disjoint i64 %address.local1154, 98, !dbg !44
  %address.local1162 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1160, i64 %address.local1161, !dbg !44
  store half %1785, ptr addrspace(1) %address.local1162, align 2, !dbg !44
  %address.local1163 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1151, !dbg !44
  %address.local1164 = or disjoint i64 %address.local1154, 99, !dbg !44
  %address.local1165 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1163, i64 %address.local1164, !dbg !44
  store half %1784, ptr addrspace(1) %address.local1165, align 2, !dbg !44
  %address.local1166 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1151, !dbg !44
  %address.local1167 = or disjoint i64 %address.local1154, 112, !dbg !44
  %address.local1168 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1166, i64 %address.local1167, !dbg !44
  store half %1778, ptr addrspace(1) %address.local1168, align 2, !dbg !44
  %address.local1169 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1151, !dbg !44
  %address.local1170 = or disjoint i64 %address.local1154, 113, !dbg !44
  %address.local1171 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1169, i64 %address.local1170, !dbg !44
  store half %1777, ptr addrspace(1) %address.local1171, align 2, !dbg !44
  %address.local1172 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1151, !dbg !44
  %address.local1173 = or disjoint i64 %address.local1154, 114, !dbg !44
  %address.local1174 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1172, i64 %address.local1173, !dbg !44
  store half %1776, ptr addrspace(1) %address.local1174, align 2, !dbg !44
  %address.local1175 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1151, !dbg !44
  %address.local1176 = or i64 %address.local1153, 115, !dbg !44
  %address.local1177 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1175, i64 %address.local1176, !dbg !44
  store half %1775, ptr addrspace(1) %address.local1177, align 2, !dbg !44
  %address.local1178 = shl i64 %address.local1149, 7, !dbg !44
  %address.local1179 = add i64 %address.local1178, 8192, !dbg !44
  %address.local1180 = add nuw nsw i64 %1788, %address.local1179, !dbg !44
  %address.local1181 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1180, !dbg !44
  %address.local1182 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1181, i64 %address.local1155, !dbg !44
  store half %1769, ptr addrspace(1) %address.local1182, align 2, !dbg !44
  %address.local1183 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1180, !dbg !44
  %address.local1184 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1183, i64 %address.local1158, !dbg !44
  store half %1768, ptr addrspace(1) %address.local1184, align 2, !dbg !44
  %address.local1185 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1180, !dbg !44
  %address.local1186 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1185, i64 %address.local1161, !dbg !44
  store half %1767, ptr addrspace(1) %address.local1186, align 2, !dbg !44
  %address.local1187 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1180, !dbg !44
  %address.local1188 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1187, i64 %address.local1164, !dbg !44
  store half %1766, ptr addrspace(1) %address.local1188, align 2, !dbg !44
  %address.local1189 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1180, !dbg !44
  %address.local1190 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1189, i64 %address.local1167, !dbg !44
  store half %1760, ptr addrspace(1) %address.local1190, align 2, !dbg !44
  %address.local1191 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1180, !dbg !44
  %address.local1192 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1191, i64 %address.local1170, !dbg !44
  store half %1759, ptr addrspace(1) %address.local1192, align 2, !dbg !44
  %address.local1193 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1180, !dbg !44
  %address.local1194 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1193, i64 %address.local1173, !dbg !44
  store half %1758, ptr addrspace(1) %address.local1194, align 2, !dbg !44
  %address.local1195 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1180, !dbg !44
  %address.local1196 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1195, i64 %address.local1176, !dbg !44
  store half %1757, ptr addrspace(1) %address.local1196, align 2, !dbg !44
  ret void, !dbg !34
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

!llvm.dbg.cu = !{!32}
!llvm.module.flags = !{!56}

!29 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/noncausal_k_dma_pass/mytest/run-20261030-163658-kcakvj_l/grid-000003")
!30 = !{}
!31 = !DISubroutineType(types: !30)
!32 = distinct !DICompileUnit(language: DW_LANG_C, file: !29, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!33 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !29, file: !29, line: 23, type: !31, scopeLine: 23, spFlags: DISPFlagDefinition, unit: !32)
!34 = !DILocation(line: 23, column: 3, scope: !33)
!35 = !DILocation(line: 57, column: 24, scope: !33)
!36 = !DILocation(line: 67, column: 22, scope: !33)
!37 = !DILocation(line: 69, column: 24, scope: !33)
!38 = !DILocation(line: 68, column: 22, scope: !33)
!39 = !DILocation(line: 87, column: 23, scope: !33)
!40 = !DILocation(line: 70, column: 26, scope: !33)
!41 = !DILocation(line: 86, column: 20, scope: !33)
!42 = !DILocation(line: 92, column: 23, scope: !33)
!43 = !DILocation(line: 93, column: 19, scope: !33)
!44 = !DILocation(line: 94, column: 9, scope: !33)
!45 = !DILocation(line: 71, column: 23, scope: !33)
!46 = !DILocation(line: 74, column: 20, scope: !33)
!47 = !DILocation(line: 75, column: 22, scope: !33)
!48 = !DILocation(line: 76, column: 23, scope: !33)
!49 = !DILocation(line: 77, column: 20, scope: !33)
!50 = !DILocation(line: 78, column: 22, scope: !33)
!51 = !DILocation(line: 79, column: 22, scope: !33)
!52 = !DILocation(line: 80, column: 20, scope: !33)
!53 = !DILocation(line: 81, column: 16, scope: !33)
!54 = !DILocation(line: 88, column: 22, scope: !33)
!55 = !{i32 256, i32 1, i32 1}
!56 = !{i32 2, !"Debug Info Version", i32 3}
!57 = !{i32 0, i32 256}
!58 = !{i32 4}
!59 = !{i64 8192}
