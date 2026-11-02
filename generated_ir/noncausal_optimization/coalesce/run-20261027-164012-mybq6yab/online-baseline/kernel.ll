; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/noncausal_optimization/coalesce/run-20261027-164012-mybq6yab/online-baseline/input.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/noncausal_optimization/coalesce/run-20261027-164012-mybq6yab/online-baseline/input.mlir"

@shm_0 = addrspace(3) global [16384 x i8] undef, align 16

; Function Attrs: convergent willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

declare float @__ocml_exp2_f32(float)

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3) #1 !dbg !4 !reqd_work_group_size !30 {
  %5 = call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !5
  %6 = sext i32 %5 to i64, !dbg !5
  %7 = call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !5
  %8 = sext i32 %7 to i64, !dbg !5
  %9 = shl nsw i64 %8, 5, !dbg !5
  %10 = add nsw i64 %9, %6, !dbg !5
  %11 = lshr i64 %10, 5, !dbg !5
  %12 = and i64 %11, 576460752303423484, !dbg !5
  %13 = and i64 %6, 3, !dbg !5
  %14 = or disjoint i64 %12, %13, !dbg !5
  %15 = sub nsw i64 31, %14, !dbg !5
  %16 = lshr i64 %10, 2, !dbg !5
  %17 = icmp ugt i64 %10, 1023, !dbg !8
  %18 = add nsw i64 %14, -32, !dbg !8
  %19 = select i1 %17, i64 %18, i64 %15, !dbg !8
  %20 = sdiv i64 %19, 32, !dbg !8
  %21 = sext i1 %17 to i64, !dbg !8
  %22 = xor i64 %20, %21, !dbg !8
  %23 = add nsw i64 %16, %22, !dbg !8
  %24 = and i64 %23, 31, !dbg !8
  %25 = shl nsw i64 %15, 7, !dbg !8
  %26 = mul nsw i64 %22, -4096, !dbg !8
  %27 = add i64 %25, %26, !dbg !8
  %28 = shl nuw nsw i64 %24, 19, !dbg !8
  %29 = shl nsw i64 %27, 7, !dbg !8
  %30 = add i64 %28, %29, !dbg !8
  %31 = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !8, !range !32
  %32 = zext nneg i32 %31 to i64, !dbg !8
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !8, !range !32, !frisk.register_lifetimes !1
  %address.local = zext nneg i32 %address.thread to i64, !dbg !8
  %address.local404 = shl nuw nsw i64 %address.local, 3, !dbg !8
  %address.local405 = and i64 %address.local404, 1536, !dbg !8
  %address.local406 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local405, !dbg !8
  %address.local407 = ptrtoint ptr addrspace(3) %address.local406 to i64, !dbg !8
  %address.local408 = trunc i64 %address.local407 to i32, !dbg !8
  %33 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local408), !dbg !8
  %address.local409 = shl nuw nsw i64 %address.local, 7, !dbg !8
  %address.local410 = and i64 %address.local409, 16256, !dbg !8
  %address.local411 = lshr i64 %address.local, 4, !dbg !8
  %address.local412 = and i64 %address.local411, 8, !dbg !8
  %address.local413 = or disjoint i64 %address.local410, %address.local412, !dbg !8
  %34 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %.idx1399 = or disjoint i64 %30, %address.local413, !dbg !8
  %.idx1399.tr = trunc i64 %.idx1399 to i32, !dbg !8
  %dma.buffer.offset1396 = shl i32 %.idx1399.tr, 1, !dbg !8
  %35 = insertelement <2 x i64> poison, i64 %34, i64 0, !dbg !8
  %36 = bitcast <2 x i64> %35 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc1397 = shufflevector <4 x i32> %36, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
  %dma.buffer.lds1398 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %33), !dbg !8
  %37 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1398, i32 %dma.buffer.offset1396, <4 x i32> %dma.buffer.rsrc1397) #10, !dbg !8
  %address.local415 = or disjoint i64 %address.local, 256, !dbg !8
  %address.local416 = shl nuw nsw i64 %address.local415, 3, !dbg !8
  %address.local417 = and i64 %address.local416, 3584, !dbg !8
  %address.local418 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local417, !dbg !8
  %address.local419 = ptrtoint ptr addrspace(3) %address.local418 to i64, !dbg !8
  %address.local420 = trunc i64 %address.local419 to i32, !dbg !8
  %38 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local420), !dbg !8
  %address.local421 = shl nuw nsw i64 %address.local, 7, !dbg !8
  %address.local422 = and i64 %address.local421, 16256, !dbg !8
  %address.local423 = lshr i64 %address.local415, 4, !dbg !8
  %address.local424 = and i64 %address.local423, 24, !dbg !8
  %address.local425 = or disjoint i64 %address.local422, %address.local424, !dbg !8
  %39 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %.idx1400 = or disjoint i64 %30, %address.local425, !dbg !8
  %.idx1400.tr = trunc i64 %.idx1400 to i32, !dbg !8
  %dma.buffer.offset1393 = shl i32 %.idx1400.tr, 1, !dbg !8
  %40 = insertelement <2 x i64> poison, i64 %39, i64 0, !dbg !8
  %41 = bitcast <2 x i64> %40 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc1394 = shufflevector <4 x i32> %41, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
  %dma.buffer.lds1395 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %38), !dbg !8
  %42 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1395, i32 %dma.buffer.offset1393, <4 x i32> %dma.buffer.rsrc1394) #10, !dbg !8
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %address.local427 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local405, !dbg !8
  %address.local428 = ptrtoint ptr addrspace(3) %address.local427 to i64, !dbg !8
  %address.local429 = trunc i64 %address.local428 to i32, !dbg !8
  %43 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local429), !dbg !8
  %44 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %45 = or disjoint i64 %30, 32, !dbg !8
  %46 = or disjoint i64 %address.local413, %45, !dbg !8
  %.tr = trunc i64 %46 to i32, !dbg !8
  %dma.buffer.offset1390 = shl i32 %.tr, 1, !dbg !8
  %47 = insertelement <2 x i64> poison, i64 %44, i64 0, !dbg !8
  %48 = bitcast <2 x i64> %47 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc1391 = shufflevector <4 x i32> %48, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
  %dma.buffer.lds1392 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %43), !dbg !8
  %49 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1392, i32 %dma.buffer.offset1390, <4 x i32> %dma.buffer.rsrc1391) #10, !dbg !8
  %address.local431 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local417, !dbg !8
  %address.local432 = ptrtoint ptr addrspace(3) %address.local431 to i64, !dbg !8
  %address.local433 = trunc i64 %address.local432 to i32, !dbg !8
  %50 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local433), !dbg !8
  %51 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %52 = or disjoint i64 %30, 32, !dbg !8
  %53 = or disjoint i64 %address.local425, %52, !dbg !8
  %.tr1401 = trunc i64 %53 to i32, !dbg !8
  %dma.buffer.offset1387 = shl i32 %.tr1401, 1, !dbg !8
  %54 = insertelement <2 x i64> poison, i64 %51, i64 0, !dbg !8
  %55 = bitcast <2 x i64> %54 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc1388 = shufflevector <4 x i32> %55, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
  %dma.buffer.lds1389 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %50), !dbg !8
  %56 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1389, i32 %dma.buffer.offset1387, <4 x i32> %dma.buffer.rsrc1388) #10, !dbg !8
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
  %57 = load <4 x half>, ptr addrspace(3) %address.local447, align 8, !dbg !10
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local447, i64 4096, !dbg !10
  %58 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !10
  %59 = fmul <4 x half> %57, splat (half 0xH3015), !dbg !10
  %lds.const.gep1197 = getelementptr i8, ptr addrspace(3) %address.local447, i64 1024, !dbg !10
  %60 = load <4 x half>, ptr addrspace(3) %lds.const.gep1197, align 8, !dbg !10
  %61 = fmul <4 x half> %58, splat (half 0xH3015), !dbg !10
  %lds.const.gep1198 = getelementptr i8, ptr addrspace(3) %address.local447, i64 5120, !dbg !10
  %62 = load <4 x half>, ptr addrspace(3) %lds.const.gep1198, align 8, !dbg !10
  %63 = fmul <4 x half> %60, splat (half 0xH3015), !dbg !10
  %64 = fmul <4 x half> %62, splat (half 0xH3015), !dbg !10
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %65 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local408), !dbg !8
  %66 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %67 = or disjoint i64 %30, 64, !dbg !8
  %68 = or disjoint i64 %address.local413, %67, !dbg !8
  %.tr1402 = trunc i64 %68 to i32, !dbg !8
  %dma.buffer.offset1384 = shl i32 %.tr1402, 1, !dbg !8
  %69 = insertelement <2 x i64> poison, i64 %66, i64 0, !dbg !8
  %70 = bitcast <2 x i64> %69 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc1385 = shufflevector <4 x i32> %70, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
  %dma.buffer.lds1386 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %65), !dbg !8
  %71 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1386, i32 %dma.buffer.offset1384, <4 x i32> %dma.buffer.rsrc1385) #10, !dbg !8
  %72 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local420), !dbg !8
  %73 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %74 = or disjoint i64 %30, 64, !dbg !8
  %75 = or disjoint i64 %address.local425, %74, !dbg !8
  %.tr1403 = trunc i64 %75 to i32, !dbg !8
  %dma.buffer.offset1381 = shl i32 %.tr1403, 1, !dbg !8
  %76 = insertelement <2 x i64> poison, i64 %73, i64 0, !dbg !8
  %77 = bitcast <2 x i64> %76 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc1382 = shufflevector <4 x i32> %77, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
  %dma.buffer.lds1383 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %72), !dbg !8
  %78 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1383, i32 %dma.buffer.offset1381, <4 x i32> %dma.buffer.rsrc1382) #10, !dbg !8
  %lds.const.gep1199 = getelementptr i8, ptr addrspace(3) %address.local447, i64 8192, !dbg !10
  %79 = load <4 x half>, ptr addrspace(3) %lds.const.gep1199, align 8, !dbg !10
  %lds.const.gep1200 = getelementptr i8, ptr addrspace(3) %address.local447, i64 12288, !dbg !10
  %80 = load <4 x half>, ptr addrspace(3) %lds.const.gep1200, align 8, !dbg !10
  %lds.const.gep1201 = getelementptr i8, ptr addrspace(3) %address.local447, i64 9216, !dbg !10
  %81 = load <4 x half>, ptr addrspace(3) %lds.const.gep1201, align 8, !dbg !10
  %82 = fmul <4 x half> %79, splat (half 0xH3015), !dbg !10
  %lds.const.gep1202 = getelementptr i8, ptr addrspace(3) %address.local447, i64 13312, !dbg !10
  %83 = load <4 x half>, ptr addrspace(3) %lds.const.gep1202, align 8, !dbg !10
  %84 = fmul <4 x half> %80, splat (half 0xH3015), !dbg !10
  %85 = fmul <4 x half> %81, splat (half 0xH3015), !dbg !10
  %86 = fmul <4 x half> %83, splat (half 0xH3015), !dbg !10
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %87 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local429), !dbg !8
  %88 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %89 = or disjoint i64 %30, 96, !dbg !8
  %90 = or disjoint i64 %address.local413, %89, !dbg !8
  %.tr1404 = trunc i64 %90 to i32, !dbg !8
  %dma.buffer.offset1378 = shl i32 %.tr1404, 1, !dbg !8
  %91 = insertelement <2 x i64> poison, i64 %88, i64 0, !dbg !8
  %92 = bitcast <2 x i64> %91 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc1379 = shufflevector <4 x i32> %92, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
  %dma.buffer.lds1380 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %87), !dbg !8
  %93 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1380, i32 %dma.buffer.offset1378, <4 x i32> %dma.buffer.rsrc1379) #10, !dbg !8
  %94 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local433), !dbg !8
  %95 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %96 = or disjoint i64 %30, 96, !dbg !8
  %97 = or disjoint i64 %address.local425, %96, !dbg !8
  %.tr1405 = trunc i64 %97 to i32, !dbg !8
  %dma.buffer.offset1375 = shl i32 %.tr1405, 1, !dbg !8
  %98 = insertelement <2 x i64> poison, i64 %95, i64 0, !dbg !8
  %99 = bitcast <2 x i64> %98 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc1376 = shufflevector <4 x i32> %99, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
  %dma.buffer.lds1377 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %94), !dbg !8
  %100 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1377, i32 %dma.buffer.offset1375, <4 x i32> %dma.buffer.rsrc1376) #10, !dbg !8
  %101 = load <4 x half>, ptr addrspace(3) %address.local447, align 8, !dbg !10
  %lds.const.gep1203 = getelementptr i8, ptr addrspace(3) %address.local447, i64 4096, !dbg !10
  %102 = load <4 x half>, ptr addrspace(3) %lds.const.gep1203, align 8, !dbg !10
  %lds.const.gep1204 = getelementptr i8, ptr addrspace(3) %address.local447, i64 1024, !dbg !10
  %103 = load <4 x half>, ptr addrspace(3) %lds.const.gep1204, align 8, !dbg !10
  %104 = fmul <4 x half> %101, splat (half 0xH3015), !dbg !10
  %lds.const.gep1205 = getelementptr i8, ptr addrspace(3) %address.local447, i64 5120, !dbg !10
  %105 = load <4 x half>, ptr addrspace(3) %lds.const.gep1205, align 8, !dbg !10
  %106 = fmul <4 x half> %102, splat (half 0xH3015), !dbg !10
  %107 = fmul <4 x half> %103, splat (half 0xH3015), !dbg !10
  %108 = fmul <4 x half> %105, splat (half 0xH3015), !dbg !10
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %lds.const.gep1206 = getelementptr i8, ptr addrspace(3) %address.local447, i64 8192, !dbg !10
  %109 = load <4 x half>, ptr addrspace(3) %lds.const.gep1206, align 8, !dbg !10
  %lds.const.gep1207 = getelementptr i8, ptr addrspace(3) %address.local447, i64 12288, !dbg !10
  %110 = load <4 x half>, ptr addrspace(3) %lds.const.gep1207, align 8, !dbg !10
  %lds.const.gep1208 = getelementptr i8, ptr addrspace(3) %address.local447, i64 9216, !dbg !10
  %111 = load <4 x half>, ptr addrspace(3) %lds.const.gep1208, align 8, !dbg !10
  %112 = fmul <4 x half> %109, splat (half 0xH3015), !dbg !10
  %lds.const.gep1209 = getelementptr i8, ptr addrspace(3) %address.local447, i64 13312, !dbg !10
  %113 = load <4 x half>, ptr addrspace(3) %lds.const.gep1209, align 8, !dbg !10
  %114 = fmul <4 x half> %110, splat (half 0xH3015), !dbg !10
  %115 = fmul <4 x half> %111, splat (half 0xH3015), !dbg !10
  %116 = fmul <4 x half> %113, splat (half 0xH3015), !dbg !10
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %117 = shl nuw nsw i64 %32, 2, !dbg !14
  %118 = shl nuw nsw i64 %32, 2, !dbg !14
  %119 = and i64 %118, 1008, !dbg !14
  %120 = sub nsw i64 %117, %119, !dbg !14
  %121 = and i32 %31, 1, !dbg !14
  %index.compact344 = zext nneg i32 %121 to i64, !dbg !14
  %122 = or disjoint i64 %120, %index.compact344, !dbg !14
  %123 = add nuw nsw i64 %122, 1, !dbg !14
  %124 = sdiv i64 %123, 2, !dbg !14
  br label %125, !dbg !7

125:                                              ; preds = %134, %4
  %126 = phi i64 [ %1152, %134 ], [ 0, %4 ], !dbg !7
  %127 = phi [2 x <8 x float>] [ %1137, %134 ], [ zeroinitializer, %4 ], !dbg !7
  %128 = phi [2 x <1 x float>] [ %807, %134 ], [ zeroinitializer, %4 ], !dbg !7
  %129 = phi [2 x <1 x float>] [ %1139, %134 ], [ [<1 x float> splat (float 0xFFF0000000000000), <1 x float> splat (float 0xFFF0000000000000)], %4 ], !dbg !7
  %130 = phi [2 x <8 x float>] [ %1143, %134 ], [ zeroinitializer, %4 ], !dbg !7
  %131 = phi [2 x <8 x float>] [ %1147, %134 ], [ zeroinitializer, %4 ], !dbg !7
  %132 = phi [2 x <8 x float>] [ %1151, %134 ], [ zeroinitializer, %4 ], !dbg !7
  %133 = icmp samesign ult i64 %126, 32, !dbg !7
  br i1 %133, label %134, label %1153, !dbg !7

134:                                              ; preds = %125
  %135 = shl i64 %16, 19, !dbg !12
  %136 = and i64 %135, 16252928, !dbg !12
  %137 = shl nuw nsw i64 %126, 14, !dbg !12
  %138 = or disjoint i64 %136, %137, !dbg !12
  %address.thread498 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !12, !range !32, !frisk.register_lifetimes !1
  %address.local499 = zext nneg i32 %address.thread498 to i64, !dbg !8
  %address.local500 = shl nuw nsw i64 %address.local499, 3, !dbg !8
  %address.local501 = and i64 %address.local500, 1536, !dbg !8
  %address.local502 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local501, !dbg !8
  %address.local503 = ptrtoint ptr addrspace(3) %address.local502 to i64, !dbg !8
  %address.local504 = trunc i64 %address.local503 to i32, !dbg !8
  %139 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local504), !dbg !12
  %address.local505 = lshr i64 %address.local499, 4, !dbg !12
  %address.local506 = and i64 %address.local505, 8, !dbg !12
  %address.local507 = and i64 %address.local499, 123, !dbg !12
  %address.local508 = lshr i64 %address.local499, 2, !dbg !12
  %address.local509 = and i64 %address.local508, 4, !dbg !12
  %address.local510 = add nuw nsw i64 %address.local509, %address.local499, !dbg !12
  %address.local511 = and i64 %address.local510, 4, !dbg !12
  %address.local512 = or disjoint i64 %address.local507, %address.local511, !dbg !12
  %address.local513 = shl nuw nsw i64 %address.local512, 7, !dbg !12
  %address.local514 = or disjoint i64 %address.local506, %address.local513, !dbg !12
  %140 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !12
  %.idx1406 = or disjoint i64 %138, %address.local514, !dbg !12
  %.idx1406.tr = trunc nuw i64 %.idx1406 to i32, !dbg !12
  %dma.buffer.offset1372 = shl nuw nsw i32 %.idx1406.tr, 1, !dbg !12
  %141 = insertelement <2 x i64> poison, i64 %140, i64 0, !dbg !12
  %142 = bitcast <2 x i64> %141 to <4 x i32>, !dbg !12
  %dma.buffer.rsrc1373 = shufflevector <4 x i32> %142, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !12
  %dma.buffer.lds1374 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %139), !dbg !12
  %143 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1374, i32 %dma.buffer.offset1372, <4 x i32> %dma.buffer.rsrc1373) #10, !dbg !12
  %address.local516 = or disjoint i64 %address.local499, 256, !dbg !8
  %address.local517 = shl nuw nsw i64 %address.local516, 3, !dbg !8
  %address.local518 = and i64 %address.local517, 3584, !dbg !8
  %address.local519 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local518, !dbg !8
  %address.local520 = ptrtoint ptr addrspace(3) %address.local519 to i64, !dbg !8
  %address.local521 = trunc i64 %address.local520 to i32, !dbg !8
  %144 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local521), !dbg !12
  %address.local522 = lshr i64 %address.local516, 4, !dbg !12
  %address.local523 = and i64 %address.local522, 24, !dbg !12
  %address.local524 = and i64 %address.local499, 123, !dbg !12
  %address.local525 = lshr i64 %address.local499, 2, !dbg !12
  %address.local526 = and i64 %address.local525, 4, !dbg !12
  %address.local527 = add nuw nsw i64 %address.local526, %address.local499, !dbg !12
  %address.local528 = and i64 %address.local527, 4, !dbg !12
  %address.local529 = or disjoint i64 %address.local524, %address.local528, !dbg !12
  %address.local530 = shl nuw nsw i64 %address.local529, 7, !dbg !12
  %address.local531 = or disjoint i64 %address.local523, %address.local530, !dbg !12
  %145 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !12
  %.idx1407 = or disjoint i64 %138, %address.local531, !dbg !12
  %.idx1407.tr = trunc nuw i64 %.idx1407 to i32, !dbg !12
  %dma.buffer.offset1369 = shl nuw nsw i32 %.idx1407.tr, 1, !dbg !12
  %146 = insertelement <2 x i64> poison, i64 %145, i64 0, !dbg !12
  %147 = bitcast <2 x i64> %146 to <4 x i32>, !dbg !12
  %dma.buffer.rsrc1370 = shufflevector <4 x i32> %147, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !12
  %dma.buffer.lds1371 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %144), !dbg !12
  %148 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1371, i32 %dma.buffer.offset1369, <4 x i32> %dma.buffer.rsrc1370) #10, !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %149 = or disjoint i64 %138, 32, !dbg !12
  %address.local533 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local501, !dbg !8
  %address.local534 = ptrtoint ptr addrspace(3) %address.local533 to i64, !dbg !8
  %address.local535 = trunc i64 %address.local534 to i32, !dbg !8
  %150 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local535), !dbg !12
  %151 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !12
  %.idx1408 = or disjoint i64 %149, %address.local514, !dbg !12
  %.idx1408.tr = trunc nuw i64 %.idx1408 to i32, !dbg !12
  %dma.buffer.offset1366 = shl nuw nsw i32 %.idx1408.tr, 1, !dbg !12
  %152 = insertelement <2 x i64> poison, i64 %151, i64 0, !dbg !12
  %153 = bitcast <2 x i64> %152 to <4 x i32>, !dbg !12
  %dma.buffer.rsrc1367 = shufflevector <4 x i32> %153, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !12
  %dma.buffer.lds1368 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %150), !dbg !12
  %154 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1368, i32 %dma.buffer.offset1366, <4 x i32> %dma.buffer.rsrc1367) #10, !dbg !12
  %address.local537 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local518, !dbg !8
  %address.local538 = ptrtoint ptr addrspace(3) %address.local537 to i64, !dbg !8
  %address.local539 = trunc i64 %address.local538 to i32, !dbg !8
  %155 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local539), !dbg !12
  %156 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !12
  %.idx1409 = or disjoint i64 %149, %address.local531, !dbg !12
  %.idx1409.tr = trunc nuw i64 %.idx1409 to i32, !dbg !12
  %dma.buffer.offset1363 = shl nuw nsw i32 %.idx1409.tr, 1, !dbg !12
  %157 = insertelement <2 x i64> poison, i64 %156, i64 0, !dbg !12
  %158 = bitcast <2 x i64> %157 to <4 x i32>, !dbg !12
  %dma.buffer.rsrc1364 = shufflevector <4 x i32> %158, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !12
  %dma.buffer.lds1365 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %155), !dbg !12
  %159 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1365, i32 %dma.buffer.offset1363, <4 x i32> %dma.buffer.rsrc1364) #10, !dbg !12
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
  %166 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %165, <4 x float> zeroinitializer) #9, !dbg !14
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
  %address.base = call i64 asm sideeffect "", "=v,0"(i64 %124) #9, !dbg !14, !frisk.register_lifetimes !1
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
  %172 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %167, <4 x float> %166) #9, !dbg !14
  %173 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %171, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1211 = getelementptr i8, ptr addrspace(3) %170, i64 4160, !dbg !14
  %174 = load <4 x half>, ptr addrspace(3) %lds.const.gep1211, align 8, !dbg !14
  %lds.const.gep1212 = getelementptr i8, ptr addrspace(3) %address.local559, i64 128, !dbg !14
  %175 = load <4 x half>, ptr addrspace(3) %lds.const.gep1212, align 8, !dbg !14
  %176 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %174, <4 x float> %173) #9, !dbg !14
  %177 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %175, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1213 = getelementptr i8, ptr addrspace(3) %address.local559, i64 4224, !dbg !14
  %178 = load <4 x half>, ptr addrspace(3) %lds.const.gep1213, align 8, !dbg !14
  %lds.const.gep1214 = getelementptr i8, ptr addrspace(3) %170, i64 192, !dbg !14
  %179 = load <4 x half>, ptr addrspace(3) %lds.const.gep1214, align 8, !dbg !14
  %180 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %178, <4 x float> %177) #9, !dbg !14
  %181 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %179, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1215 = getelementptr i8, ptr addrspace(3) %170, i64 4288, !dbg !14
  %182 = load <4 x half>, ptr addrspace(3) %lds.const.gep1215, align 8, !dbg !14
  %lds.const.gep1216 = getelementptr i8, ptr addrspace(3) %address.local559, i64 1024, !dbg !14
  %183 = load <4 x half>, ptr addrspace(3) %lds.const.gep1216, align 8, !dbg !14
  %184 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %182, <4 x float> %181) #9, !dbg !14
  %185 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %183, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1217 = getelementptr i8, ptr addrspace(3) %address.local559, i64 5120, !dbg !14
  %186 = load <4 x half>, ptr addrspace(3) %lds.const.gep1217, align 8, !dbg !14
  %lds.const.gep1218 = getelementptr i8, ptr addrspace(3) %170, i64 1088, !dbg !14
  %187 = load <4 x half>, ptr addrspace(3) %lds.const.gep1218, align 8, !dbg !14
  %188 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %186, <4 x float> %185) #9, !dbg !14
  %189 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %187, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1219 = getelementptr i8, ptr addrspace(3) %170, i64 5184, !dbg !14
  %190 = load <4 x half>, ptr addrspace(3) %lds.const.gep1219, align 8, !dbg !14
  %lds.const.gep1220 = getelementptr i8, ptr addrspace(3) %address.local559, i64 1152, !dbg !14
  %191 = load <4 x half>, ptr addrspace(3) %lds.const.gep1220, align 8, !dbg !14
  %192 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %190, <4 x float> %189) #9, !dbg !14
  %193 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %191, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1221 = getelementptr i8, ptr addrspace(3) %address.local559, i64 5248, !dbg !14
  %194 = load <4 x half>, ptr addrspace(3) %lds.const.gep1221, align 8, !dbg !14
  %lds.const.gep1222 = getelementptr i8, ptr addrspace(3) %170, i64 1216, !dbg !14
  %195 = load <4 x half>, ptr addrspace(3) %lds.const.gep1222, align 8, !dbg !14
  %196 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %194, <4 x float> %193) #9, !dbg !14
  %197 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %195, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1223 = getelementptr i8, ptr addrspace(3) %170, i64 5312, !dbg !14
  %198 = load <4 x half>, ptr addrspace(3) %lds.const.gep1223, align 8, !dbg !14
  %199 = load <4 x half>, ptr addrspace(3) %address.local559, align 8, !dbg !14
  %200 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %198, <4 x float> %197) #9, !dbg !14
  %201 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %63, <4 x half> %199, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1224 = getelementptr i8, ptr addrspace(3) %address.local559, i64 4096, !dbg !14
  %202 = load <4 x half>, ptr addrspace(3) %lds.const.gep1224, align 8, !dbg !14
  %203 = load <4 x half>, ptr addrspace(3) %address.local588, align 8, !dbg !14
  %204 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %202, <4 x float> %201) #9, !dbg !14
  %205 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %63, <4 x half> %203, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1225 = getelementptr i8, ptr addrspace(3) %170, i64 4160, !dbg !14
  %206 = load <4 x half>, ptr addrspace(3) %lds.const.gep1225, align 8, !dbg !14
  %lds.const.gep1226 = getelementptr i8, ptr addrspace(3) %address.local559, i64 128, !dbg !14
  %207 = load <4 x half>, ptr addrspace(3) %lds.const.gep1226, align 8, !dbg !14
  %208 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %206, <4 x float> %205) #9, !dbg !14
  %209 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %63, <4 x half> %207, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1227 = getelementptr i8, ptr addrspace(3) %address.local559, i64 4224, !dbg !14
  %210 = load <4 x half>, ptr addrspace(3) %lds.const.gep1227, align 8, !dbg !14
  %lds.const.gep1228 = getelementptr i8, ptr addrspace(3) %170, i64 192, !dbg !14
  %211 = load <4 x half>, ptr addrspace(3) %lds.const.gep1228, align 8, !dbg !14
  %212 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %210, <4 x float> %209) #9, !dbg !14
  %213 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %63, <4 x half> %211, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1229 = getelementptr i8, ptr addrspace(3) %170, i64 4288, !dbg !14
  %214 = load <4 x half>, ptr addrspace(3) %lds.const.gep1229, align 8, !dbg !14
  %lds.const.gep1230 = getelementptr i8, ptr addrspace(3) %address.local559, i64 1024, !dbg !14
  %215 = load <4 x half>, ptr addrspace(3) %lds.const.gep1230, align 8, !dbg !14
  %216 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %214, <4 x float> %213) #9, !dbg !14
  %217 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %63, <4 x half> %215, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1231 = getelementptr i8, ptr addrspace(3) %address.local559, i64 5120, !dbg !14
  %218 = load <4 x half>, ptr addrspace(3) %lds.const.gep1231, align 8, !dbg !14
  %lds.const.gep1232 = getelementptr i8, ptr addrspace(3) %170, i64 1088, !dbg !14
  %219 = load <4 x half>, ptr addrspace(3) %lds.const.gep1232, align 8, !dbg !14
  %220 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %218, <4 x float> %217) #9, !dbg !14
  %221 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %63, <4 x half> %219, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1233 = getelementptr i8, ptr addrspace(3) %170, i64 5184, !dbg !14
  %222 = load <4 x half>, ptr addrspace(3) %lds.const.gep1233, align 8, !dbg !14
  %lds.const.gep1234 = getelementptr i8, ptr addrspace(3) %address.local559, i64 1152, !dbg !14
  %223 = load <4 x half>, ptr addrspace(3) %lds.const.gep1234, align 8, !dbg !14
  %224 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %222, <4 x float> %221) #9, !dbg !14
  %225 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %63, <4 x half> %223, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1235 = getelementptr i8, ptr addrspace(3) %address.local559, i64 5248, !dbg !14
  %226 = load <4 x half>, ptr addrspace(3) %lds.const.gep1235, align 8, !dbg !14
  %lds.const.gep1236 = getelementptr i8, ptr addrspace(3) %170, i64 1216, !dbg !14
  %227 = load <4 x half>, ptr addrspace(3) %lds.const.gep1236, align 8, !dbg !14
  %228 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %226, <4 x float> %225) #9, !dbg !14
  %229 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %63, <4 x half> %227, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1237 = getelementptr i8, ptr addrspace(3) %170, i64 5312, !dbg !14
  %230 = load <4 x half>, ptr addrspace(3) %lds.const.gep1237, align 8, !dbg !14
  %231 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %230, <4 x float> %229) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %232 = or disjoint i64 %138, 64, !dbg !12
  %233 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local504), !dbg !12
  %234 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !12
  %.idx14111412 = or disjoint i64 %232, %address.local514, !dbg !12
  %.idx14111412.tr = trunc nuw i64 %.idx14111412 to i32, !dbg !12
  %dma.buffer.offset1360 = shl nuw nsw i32 %.idx14111412.tr, 1, !dbg !12
  %235 = insertelement <2 x i64> poison, i64 %234, i64 0, !dbg !12
  %236 = bitcast <2 x i64> %235 to <4 x i32>, !dbg !12
  %dma.buffer.rsrc1361 = shufflevector <4 x i32> %236, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !12
  %dma.buffer.lds1362 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %233), !dbg !12
  %237 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1362, i32 %dma.buffer.offset1360, <4 x i32> %dma.buffer.rsrc1361) #10, !dbg !12
  %238 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local521), !dbg !12
  %239 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !12
  %.idx14131414 = or disjoint i64 %232, %address.local531, !dbg !12
  %.idx14131414.tr = trunc nuw i64 %.idx14131414 to i32, !dbg !12
  %dma.buffer.offset1357 = shl nuw nsw i32 %.idx14131414.tr, 1, !dbg !12
  %240 = insertelement <2 x i64> poison, i64 %239, i64 0, !dbg !12
  %241 = bitcast <2 x i64> %240 to <4 x i32>, !dbg !12
  %dma.buffer.rsrc1358 = shufflevector <4 x i32> %241, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !12
  %dma.buffer.lds1359 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %238), !dbg !12
  %242 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1359, i32 %dma.buffer.offset1357, <4 x i32> %dma.buffer.rsrc1358) #10, !dbg !12
  %lds.const.gep1238 = getelementptr i8, ptr addrspace(3) %address.local559, i64 8192, !dbg !14
  %243 = load <4 x half>, ptr addrspace(3) %lds.const.gep1238, align 8, !dbg !14
  %244 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %243, <4 x float> %172) #9, !dbg !14
  %lds.const.gep1239 = getelementptr i8, ptr addrspace(3) %address.local559, i64 12288, !dbg !14
  %245 = load <4 x half>, ptr addrspace(3) %lds.const.gep1239, align 8, !dbg !14
  %lds.const.gep1240 = getelementptr i8, ptr addrspace(3) %170, i64 8256, !dbg !14
  %246 = load <4 x half>, ptr addrspace(3) %lds.const.gep1240, align 8, !dbg !14
  %247 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %245, <4 x float> %244) #9, !dbg !14
  %248 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %246, <4 x float> %176) #9, !dbg !14
  %lds.const.gep1241 = getelementptr i8, ptr addrspace(3) %170, i64 12352, !dbg !14
  %249 = load <4 x half>, ptr addrspace(3) %lds.const.gep1241, align 8, !dbg !14
  %lds.const.gep1242 = getelementptr i8, ptr addrspace(3) %address.local559, i64 8320, !dbg !14
  %250 = load <4 x half>, ptr addrspace(3) %lds.const.gep1242, align 8, !dbg !14
  %251 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %249, <4 x float> %248) #9, !dbg !14
  %252 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %250, <4 x float> %180) #9, !dbg !14
  %lds.const.gep1243 = getelementptr i8, ptr addrspace(3) %address.local559, i64 12416, !dbg !14
  %253 = load <4 x half>, ptr addrspace(3) %lds.const.gep1243, align 8, !dbg !14
  %lds.const.gep1244 = getelementptr i8, ptr addrspace(3) %170, i64 8384, !dbg !14
  %254 = load <4 x half>, ptr addrspace(3) %lds.const.gep1244, align 8, !dbg !14
  %255 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %253, <4 x float> %252) #9, !dbg !14
  %256 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %254, <4 x float> %184) #9, !dbg !14
  %lds.const.gep1245 = getelementptr i8, ptr addrspace(3) %170, i64 12480, !dbg !14
  %257 = load <4 x half>, ptr addrspace(3) %lds.const.gep1245, align 8, !dbg !14
  %lds.const.gep1246 = getelementptr i8, ptr addrspace(3) %address.local559, i64 9216, !dbg !14
  %258 = load <4 x half>, ptr addrspace(3) %lds.const.gep1246, align 8, !dbg !14
  %259 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %257, <4 x float> %256) #9, !dbg !14
  %260 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %258, <4 x float> %188) #9, !dbg !14
  %lds.const.gep1247 = getelementptr i8, ptr addrspace(3) %address.local559, i64 13312, !dbg !14
  %261 = load <4 x half>, ptr addrspace(3) %lds.const.gep1247, align 8, !dbg !14
  %lds.const.gep1248 = getelementptr i8, ptr addrspace(3) %170, i64 9280, !dbg !14
  %262 = load <4 x half>, ptr addrspace(3) %lds.const.gep1248, align 8, !dbg !14
  %263 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %261, <4 x float> %260) #9, !dbg !14
  %264 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %262, <4 x float> %192) #9, !dbg !14
  %lds.const.gep1249 = getelementptr i8, ptr addrspace(3) %170, i64 13376, !dbg !14
  %265 = load <4 x half>, ptr addrspace(3) %lds.const.gep1249, align 8, !dbg !14
  %lds.const.gep1250 = getelementptr i8, ptr addrspace(3) %address.local559, i64 9344, !dbg !14
  %266 = load <4 x half>, ptr addrspace(3) %lds.const.gep1250, align 8, !dbg !14
  %267 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %265, <4 x float> %264) #9, !dbg !14
  %268 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %266, <4 x float> %196) #9, !dbg !14
  %lds.const.gep1251 = getelementptr i8, ptr addrspace(3) %address.local559, i64 13440, !dbg !14
  %269 = load <4 x half>, ptr addrspace(3) %lds.const.gep1251, align 8, !dbg !14
  %lds.const.gep1252 = getelementptr i8, ptr addrspace(3) %170, i64 9408, !dbg !14
  %270 = load <4 x half>, ptr addrspace(3) %lds.const.gep1252, align 8, !dbg !14
  %271 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %269, <4 x float> %268) #9, !dbg !14
  %272 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %270, <4 x float> %200) #9, !dbg !14
  %lds.const.gep1253 = getelementptr i8, ptr addrspace(3) %170, i64 13504, !dbg !14
  %273 = load <4 x half>, ptr addrspace(3) %lds.const.gep1253, align 8, !dbg !14
  %lds.const.gep1254 = getelementptr i8, ptr addrspace(3) %address.local559, i64 8192, !dbg !14
  %274 = load <4 x half>, ptr addrspace(3) %lds.const.gep1254, align 8, !dbg !14
  %275 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %273, <4 x float> %272) #9, !dbg !14
  %276 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %274, <4 x float> %204) #9, !dbg !14
  %lds.const.gep1255 = getelementptr i8, ptr addrspace(3) %address.local559, i64 12288, !dbg !14
  %277 = load <4 x half>, ptr addrspace(3) %lds.const.gep1255, align 8, !dbg !14
  %lds.const.gep1256 = getelementptr i8, ptr addrspace(3) %170, i64 8256, !dbg !14
  %278 = load <4 x half>, ptr addrspace(3) %lds.const.gep1256, align 8, !dbg !14
  %279 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %277, <4 x float> %276) #9, !dbg !14
  %280 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %278, <4 x float> %208) #9, !dbg !14
  %lds.const.gep1257 = getelementptr i8, ptr addrspace(3) %170, i64 12352, !dbg !14
  %281 = load <4 x half>, ptr addrspace(3) %lds.const.gep1257, align 8, !dbg !14
  %lds.const.gep1258 = getelementptr i8, ptr addrspace(3) %address.local559, i64 8320, !dbg !14
  %282 = load <4 x half>, ptr addrspace(3) %lds.const.gep1258, align 8, !dbg !14
  %283 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %281, <4 x float> %280) #9, !dbg !14
  %284 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %282, <4 x float> %212) #9, !dbg !14
  %lds.const.gep1259 = getelementptr i8, ptr addrspace(3) %address.local559, i64 12416, !dbg !14
  %285 = load <4 x half>, ptr addrspace(3) %lds.const.gep1259, align 8, !dbg !14
  %lds.const.gep1260 = getelementptr i8, ptr addrspace(3) %170, i64 8384, !dbg !14
  %286 = load <4 x half>, ptr addrspace(3) %lds.const.gep1260, align 8, !dbg !14
  %287 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %285, <4 x float> %284) #9, !dbg !14
  %288 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %286, <4 x float> %216) #9, !dbg !14
  %lds.const.gep1261 = getelementptr i8, ptr addrspace(3) %170, i64 12480, !dbg !14
  %289 = load <4 x half>, ptr addrspace(3) %lds.const.gep1261, align 8, !dbg !14
  %lds.const.gep1262 = getelementptr i8, ptr addrspace(3) %address.local559, i64 9216, !dbg !14
  %290 = load <4 x half>, ptr addrspace(3) %lds.const.gep1262, align 8, !dbg !14
  %291 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %289, <4 x float> %288) #9, !dbg !14
  %292 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %290, <4 x float> %220) #9, !dbg !14
  %lds.const.gep1263 = getelementptr i8, ptr addrspace(3) %address.local559, i64 13312, !dbg !14
  %293 = load <4 x half>, ptr addrspace(3) %lds.const.gep1263, align 8, !dbg !14
  %lds.const.gep1264 = getelementptr i8, ptr addrspace(3) %170, i64 9280, !dbg !14
  %294 = load <4 x half>, ptr addrspace(3) %lds.const.gep1264, align 8, !dbg !14
  %295 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %293, <4 x float> %292) #9, !dbg !14
  %296 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %294, <4 x float> %224) #9, !dbg !14
  %lds.const.gep1265 = getelementptr i8, ptr addrspace(3) %170, i64 13376, !dbg !14
  %297 = load <4 x half>, ptr addrspace(3) %lds.const.gep1265, align 8, !dbg !14
  %lds.const.gep1266 = getelementptr i8, ptr addrspace(3) %address.local559, i64 9344, !dbg !14
  %298 = load <4 x half>, ptr addrspace(3) %lds.const.gep1266, align 8, !dbg !14
  %299 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %297, <4 x float> %296) #9, !dbg !14
  %300 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %298, <4 x float> %228) #9, !dbg !14
  %lds.const.gep1267 = getelementptr i8, ptr addrspace(3) %address.local559, i64 13440, !dbg !14
  %301 = load <4 x half>, ptr addrspace(3) %lds.const.gep1267, align 8, !dbg !14
  %lds.const.gep1268 = getelementptr i8, ptr addrspace(3) %170, i64 9408, !dbg !14
  %302 = load <4 x half>, ptr addrspace(3) %lds.const.gep1268, align 8, !dbg !14
  %303 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %301, <4 x float> %300) #9, !dbg !14
  %304 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %302, <4 x float> %231) #9, !dbg !14
  %lds.const.gep1269 = getelementptr i8, ptr addrspace(3) %170, i64 13504, !dbg !14
  %305 = load <4 x half>, ptr addrspace(3) %lds.const.gep1269, align 8, !dbg !14
  %306 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %305, <4 x float> %304) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %307 = or disjoint i64 %138, 96, !dbg !12
  %308 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local535), !dbg !12
  %309 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !12
  %.idx14151416 = or disjoint i64 %307, %address.local514, !dbg !12
  %.idx14151416.tr = trunc nuw i64 %.idx14151416 to i32, !dbg !12
  %dma.buffer.offset1354 = shl nuw nsw i32 %.idx14151416.tr, 1, !dbg !12
  %310 = insertelement <2 x i64> poison, i64 %309, i64 0, !dbg !12
  %311 = bitcast <2 x i64> %310 to <4 x i32>, !dbg !12
  %dma.buffer.rsrc1355 = shufflevector <4 x i32> %311, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !12
  %dma.buffer.lds1356 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %308), !dbg !12
  %312 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1356, i32 %dma.buffer.offset1354, <4 x i32> %dma.buffer.rsrc1355) #10, !dbg !12
  %313 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local539), !dbg !12
  %314 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !12
  %.idx14171418 = or disjoint i64 %307, %address.local531, !dbg !12
  %.idx14171418.tr = trunc nuw i64 %.idx14171418 to i32, !dbg !12
  %dma.buffer.offset1351 = shl nuw nsw i32 %.idx14171418.tr, 1, !dbg !12
  %315 = insertelement <2 x i64> poison, i64 %314, i64 0, !dbg !12
  %316 = bitcast <2 x i64> %315 to <4 x i32>, !dbg !12
  %dma.buffer.rsrc1352 = shufflevector <4 x i32> %316, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !12
  %dma.buffer.lds1353 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %313), !dbg !12
  %317 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1353, i32 %dma.buffer.offset1351, <4 x i32> %dma.buffer.rsrc1352) #10, !dbg !12
  %318 = load <4 x half>, ptr addrspace(3) %address.local559, align 8, !dbg !14
  %319 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %318, <4 x float> %247) #9, !dbg !14
  %lds.const.gep1270 = getelementptr i8, ptr addrspace(3) %address.local559, i64 4096, !dbg !14
  %320 = load <4 x half>, ptr addrspace(3) %lds.const.gep1270, align 8, !dbg !14
  %321 = load <4 x half>, ptr addrspace(3) %address.local588, align 8, !dbg !14
  %322 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %106, <4 x half> %320, <4 x float> %319) #9, !dbg !14
  %323 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %321, <4 x float> %251) #9, !dbg !14
  %lds.const.gep1271 = getelementptr i8, ptr addrspace(3) %170, i64 4160, !dbg !14
  %324 = load <4 x half>, ptr addrspace(3) %lds.const.gep1271, align 8, !dbg !14
  %lds.const.gep1272 = getelementptr i8, ptr addrspace(3) %address.local559, i64 128, !dbg !14
  %325 = load <4 x half>, ptr addrspace(3) %lds.const.gep1272, align 8, !dbg !14
  %326 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %106, <4 x half> %324, <4 x float> %323) #9, !dbg !14
  %327 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %325, <4 x float> %255) #9, !dbg !14
  %lds.const.gep1273 = getelementptr i8, ptr addrspace(3) %address.local559, i64 4224, !dbg !14
  %328 = load <4 x half>, ptr addrspace(3) %lds.const.gep1273, align 8, !dbg !14
  %lds.const.gep1274 = getelementptr i8, ptr addrspace(3) %170, i64 192, !dbg !14
  %329 = load <4 x half>, ptr addrspace(3) %lds.const.gep1274, align 8, !dbg !14
  %330 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %106, <4 x half> %328, <4 x float> %327) #9, !dbg !14
  %331 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %329, <4 x float> %259) #9, !dbg !14
  %lds.const.gep1275 = getelementptr i8, ptr addrspace(3) %170, i64 4288, !dbg !14
  %332 = load <4 x half>, ptr addrspace(3) %lds.const.gep1275, align 8, !dbg !14
  %lds.const.gep1276 = getelementptr i8, ptr addrspace(3) %address.local559, i64 1024, !dbg !14
  %333 = load <4 x half>, ptr addrspace(3) %lds.const.gep1276, align 8, !dbg !14
  %334 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %106, <4 x half> %332, <4 x float> %331) #9, !dbg !14
  %335 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %333, <4 x float> %263) #9, !dbg !14
  %lds.const.gep1277 = getelementptr i8, ptr addrspace(3) %address.local559, i64 5120, !dbg !14
  %336 = load <4 x half>, ptr addrspace(3) %lds.const.gep1277, align 8, !dbg !14
  %lds.const.gep1278 = getelementptr i8, ptr addrspace(3) %170, i64 1088, !dbg !14
  %337 = load <4 x half>, ptr addrspace(3) %lds.const.gep1278, align 8, !dbg !14
  %338 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %106, <4 x half> %336, <4 x float> %335) #9, !dbg !14
  %339 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %337, <4 x float> %267) #9, !dbg !14
  %lds.const.gep1279 = getelementptr i8, ptr addrspace(3) %170, i64 5184, !dbg !14
  %340 = load <4 x half>, ptr addrspace(3) %lds.const.gep1279, align 8, !dbg !14
  %lds.const.gep1280 = getelementptr i8, ptr addrspace(3) %address.local559, i64 1152, !dbg !14
  %341 = load <4 x half>, ptr addrspace(3) %lds.const.gep1280, align 8, !dbg !14
  %342 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %106, <4 x half> %340, <4 x float> %339) #9, !dbg !14
  %343 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %341, <4 x float> %271) #9, !dbg !14
  %lds.const.gep1281 = getelementptr i8, ptr addrspace(3) %address.local559, i64 5248, !dbg !14
  %344 = load <4 x half>, ptr addrspace(3) %lds.const.gep1281, align 8, !dbg !14
  %lds.const.gep1282 = getelementptr i8, ptr addrspace(3) %170, i64 1216, !dbg !14
  %345 = load <4 x half>, ptr addrspace(3) %lds.const.gep1282, align 8, !dbg !14
  %346 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %106, <4 x half> %344, <4 x float> %343) #9, !dbg !14
  %347 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %345, <4 x float> %275) #9, !dbg !14
  %lds.const.gep1283 = getelementptr i8, ptr addrspace(3) %170, i64 5312, !dbg !14
  %348 = load <4 x half>, ptr addrspace(3) %lds.const.gep1283, align 8, !dbg !14
  %349 = load <4 x half>, ptr addrspace(3) %address.local559, align 8, !dbg !14
  %350 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %106, <4 x half> %348, <4 x float> %347) #9, !dbg !14
  %351 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %107, <4 x half> %349, <4 x float> %279) #9, !dbg !14
  %lds.const.gep1284 = getelementptr i8, ptr addrspace(3) %address.local559, i64 4096, !dbg !14
  %352 = load <4 x half>, ptr addrspace(3) %lds.const.gep1284, align 8, !dbg !14
  %353 = load <4 x half>, ptr addrspace(3) %address.local588, align 8, !dbg !14
  %354 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %108, <4 x half> %352, <4 x float> %351) #9, !dbg !14
  %355 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %107, <4 x half> %353, <4 x float> %283) #9, !dbg !14
  %lds.const.gep1285 = getelementptr i8, ptr addrspace(3) %170, i64 4160, !dbg !14
  %356 = load <4 x half>, ptr addrspace(3) %lds.const.gep1285, align 8, !dbg !14
  %lds.const.gep1286 = getelementptr i8, ptr addrspace(3) %address.local559, i64 128, !dbg !14
  %357 = load <4 x half>, ptr addrspace(3) %lds.const.gep1286, align 8, !dbg !14
  %358 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %108, <4 x half> %356, <4 x float> %355) #9, !dbg !14
  %359 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %107, <4 x half> %357, <4 x float> %287) #9, !dbg !14
  %lds.const.gep1287 = getelementptr i8, ptr addrspace(3) %address.local559, i64 4224, !dbg !14
  %360 = load <4 x half>, ptr addrspace(3) %lds.const.gep1287, align 8, !dbg !14
  %lds.const.gep1288 = getelementptr i8, ptr addrspace(3) %170, i64 192, !dbg !14
  %361 = load <4 x half>, ptr addrspace(3) %lds.const.gep1288, align 8, !dbg !14
  %362 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %108, <4 x half> %360, <4 x float> %359) #9, !dbg !14
  %363 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %107, <4 x half> %361, <4 x float> %291) #9, !dbg !14
  %lds.const.gep1289 = getelementptr i8, ptr addrspace(3) %170, i64 4288, !dbg !14
  %364 = load <4 x half>, ptr addrspace(3) %lds.const.gep1289, align 8, !dbg !14
  %lds.const.gep1290 = getelementptr i8, ptr addrspace(3) %address.local559, i64 1024, !dbg !14
  %365 = load <4 x half>, ptr addrspace(3) %lds.const.gep1290, align 8, !dbg !14
  %366 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %108, <4 x half> %364, <4 x float> %363) #9, !dbg !14
  %367 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %107, <4 x half> %365, <4 x float> %295) #9, !dbg !14
  %lds.const.gep1291 = getelementptr i8, ptr addrspace(3) %address.local559, i64 5120, !dbg !14
  %368 = load <4 x half>, ptr addrspace(3) %lds.const.gep1291, align 8, !dbg !14
  %lds.const.gep1292 = getelementptr i8, ptr addrspace(3) %170, i64 1088, !dbg !14
  %369 = load <4 x half>, ptr addrspace(3) %lds.const.gep1292, align 8, !dbg !14
  %370 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %108, <4 x half> %368, <4 x float> %367) #9, !dbg !14
  %371 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %107, <4 x half> %369, <4 x float> %299) #9, !dbg !14
  %lds.const.gep1293 = getelementptr i8, ptr addrspace(3) %170, i64 5184, !dbg !14
  %372 = load <4 x half>, ptr addrspace(3) %lds.const.gep1293, align 8, !dbg !14
  %lds.const.gep1294 = getelementptr i8, ptr addrspace(3) %address.local559, i64 1152, !dbg !14
  %373 = load <4 x half>, ptr addrspace(3) %lds.const.gep1294, align 8, !dbg !14
  %374 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %108, <4 x half> %372, <4 x float> %371) #9, !dbg !14
  %375 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %107, <4 x half> %373, <4 x float> %303) #9, !dbg !14
  %lds.const.gep1295 = getelementptr i8, ptr addrspace(3) %address.local559, i64 5248, !dbg !14
  %376 = load <4 x half>, ptr addrspace(3) %lds.const.gep1295, align 8, !dbg !14
  %lds.const.gep1296 = getelementptr i8, ptr addrspace(3) %170, i64 1216, !dbg !14
  %377 = load <4 x half>, ptr addrspace(3) %lds.const.gep1296, align 8, !dbg !14
  %378 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %108, <4 x half> %376, <4 x float> %375) #9, !dbg !14
  %379 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %107, <4 x half> %377, <4 x float> %306) #9, !dbg !14
  %lds.const.gep1297 = getelementptr i8, ptr addrspace(3) %170, i64 5312, !dbg !14
  %380 = load <4 x half>, ptr addrspace(3) %lds.const.gep1297, align 8, !dbg !14
  %381 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %108, <4 x half> %380, <4 x float> %379) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %382 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local504), !dbg !15
  %address.local917 = shl nuw nsw i64 %address.local499, 5, !dbg !15
  %address.local918 = and i64 %address.local917, 8064, !dbg !15
  %address.local919 = shl nuw nsw i64 %address.local499, 3, !dbg !15
  %address.local920 = and i64 %address.local919, 24, !dbg !15
  %address.local921 = or disjoint i64 %address.local918, %address.local920, !dbg !15
  %383 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %.idx14191420 = or disjoint i64 %138, %address.local921, !dbg !15
  %.idx14191420.tr = trunc nuw i64 %.idx14191420 to i32, !dbg !15
  %dma.buffer.offset1348 = shl nuw nsw i32 %.idx14191420.tr, 1, !dbg !15
  %384 = insertelement <2 x i64> poison, i64 %383, i64 0, !dbg !15
  %385 = bitcast <2 x i64> %384 to <4 x i32>, !dbg !15
  %dma.buffer.rsrc1349 = shufflevector <4 x i32> %385, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %dma.buffer.lds1350 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %382), !dbg !15
  %386 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1350, i32 %dma.buffer.offset1348, <4 x i32> %dma.buffer.rsrc1349) #10, !dbg !15
  %387 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local521), !dbg !15
  %address.local923 = shl nuw nsw i64 %address.local516, 5, !dbg !15
  %address.local924 = and i64 %address.local923, 16256, !dbg !15
  %address.local925 = shl nuw nsw i64 %address.local499, 3, !dbg !15
  %address.local926 = and i64 %address.local925, 24, !dbg !15
  %address.local927 = or disjoint i64 %address.local924, %address.local926, !dbg !15
  %388 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %.idx14211422 = or disjoint i64 %138, %address.local927, !dbg !15
  %.idx14211422.tr = trunc nuw i64 %.idx14211422 to i32, !dbg !15
  %dma.buffer.offset1345 = shl nuw nsw i32 %.idx14211422.tr, 1, !dbg !15
  %389 = insertelement <2 x i64> poison, i64 %388, i64 0, !dbg !15
  %390 = bitcast <2 x i64> %389 to <4 x i32>, !dbg !15
  %dma.buffer.rsrc1346 = shufflevector <4 x i32> %390, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %dma.buffer.lds1347 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %387), !dbg !15
  %391 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1347, i32 %dma.buffer.offset1345, <4 x i32> %dma.buffer.rsrc1346) #10, !dbg !15
  %lds.const.gep1298 = getelementptr i8, ptr addrspace(3) %address.local559, i64 8192, !dbg !14
  %392 = load <4 x half>, ptr addrspace(3) %lds.const.gep1298, align 8, !dbg !14
  %393 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %112, <4 x half> %392, <4 x float> %322) #9, !dbg !14
  %lds.const.gep1299 = getelementptr i8, ptr addrspace(3) %address.local559, i64 12288, !dbg !14
  %394 = load <4 x half>, ptr addrspace(3) %lds.const.gep1299, align 8, !dbg !14
  %lds.const.gep1300 = getelementptr i8, ptr addrspace(3) %170, i64 8256, !dbg !14
  %395 = load <4 x half>, ptr addrspace(3) %lds.const.gep1300, align 8, !dbg !14
  %396 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %114, <4 x half> %394, <4 x float> %393) #9, !dbg !14
  %397 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %112, <4 x half> %395, <4 x float> %326) #9, !dbg !14
  %398 = extractelement <4 x float> %396, i64 0, !dbg !19
  %399 = call float @llvm.maxnum.f32(float %398, float 0xFFF0000000000000), !dbg !20, !frisk.local_reduction !1
  %400 = extractelement <4 x float> %396, i64 1, !dbg !19
  %401 = extractelement <4 x float> %396, i64 2, !dbg !19
  %402 = extractelement <4 x float> %396, i64 3, !dbg !19
  %lds.const.gep1301 = getelementptr i8, ptr addrspace(3) %170, i64 12352, !dbg !14
  %403 = load <4 x half>, ptr addrspace(3) %lds.const.gep1301, align 8, !dbg !14
  %lds.const.gep1302 = getelementptr i8, ptr addrspace(3) %address.local559, i64 8320, !dbg !14
  %404 = load <4 x half>, ptr addrspace(3) %lds.const.gep1302, align 8, !dbg !14
  %405 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %114, <4 x half> %403, <4 x float> %397) #9, !dbg !14
  %406 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %112, <4 x half> %404, <4 x float> %330) #9, !dbg !14
  %407 = extractelement <4 x float> %405, i64 0, !dbg !19
  %local.reduce151 = call float @llvm.maxnum.f32(float %399, float %407), !dbg !20, !frisk.local_reduction !1
  %408 = extractelement <4 x float> %405, i64 1, !dbg !19
  %local.reduce152 = call float @llvm.maxnum.f32(float %400, float %408), !dbg !20, !frisk.local_reduction !1
  %409 = extractelement <4 x float> %405, i64 2, !dbg !19
  %local.reduce153 = call float @llvm.maxnum.f32(float %401, float %409), !dbg !20, !frisk.local_reduction !1
  %410 = extractelement <4 x float> %405, i64 3, !dbg !19
  %local.reduce154 = call float @llvm.maxnum.f32(float %402, float %410), !dbg !20, !frisk.local_reduction !1
  %lds.const.gep1303 = getelementptr i8, ptr addrspace(3) %address.local559, i64 12416, !dbg !14
  %411 = load <4 x half>, ptr addrspace(3) %lds.const.gep1303, align 8, !dbg !14
  %lds.const.gep1304 = getelementptr i8, ptr addrspace(3) %170, i64 8384, !dbg !14
  %412 = load <4 x half>, ptr addrspace(3) %lds.const.gep1304, align 8, !dbg !14
  %413 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %114, <4 x half> %411, <4 x float> %406) #9, !dbg !14
  %414 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %112, <4 x half> %412, <4 x float> %334) #9, !dbg !14
  %415 = extractelement <4 x float> %413, i64 0, !dbg !19
  %local.reduce155 = call float @llvm.maxnum.f32(float %local.reduce151, float %415), !dbg !20, !frisk.local_reduction !1
  %416 = extractelement <4 x float> %413, i64 1, !dbg !19
  %local.reduce156 = call float @llvm.maxnum.f32(float %local.reduce152, float %416), !dbg !20, !frisk.local_reduction !1
  %417 = extractelement <4 x float> %413, i64 2, !dbg !19
  %local.reduce157 = call float @llvm.maxnum.f32(float %local.reduce153, float %417), !dbg !20, !frisk.local_reduction !1
  %418 = extractelement <4 x float> %413, i64 3, !dbg !19
  %local.reduce158 = call float @llvm.maxnum.f32(float %local.reduce154, float %418), !dbg !20, !frisk.local_reduction !1
  %lds.const.gep1305 = getelementptr i8, ptr addrspace(3) %170, i64 12480, !dbg !14
  %419 = load <4 x half>, ptr addrspace(3) %lds.const.gep1305, align 8, !dbg !14
  %lds.const.gep1306 = getelementptr i8, ptr addrspace(3) %address.local559, i64 9216, !dbg !14
  %420 = load <4 x half>, ptr addrspace(3) %lds.const.gep1306, align 8, !dbg !14
  %421 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %114, <4 x half> %419, <4 x float> %414) #9, !dbg !14
  %422 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %112, <4 x half> %420, <4 x float> %338) #9, !dbg !14
  %423 = extractelement <4 x float> %421, i64 0, !dbg !19
  %local.reduce159 = call float @llvm.maxnum.f32(float %local.reduce155, float %423), !dbg !20, !frisk.local_reduction !1
  %424 = extractelement <4 x float> %421, i64 1, !dbg !19
  %local.reduce160 = call float @llvm.maxnum.f32(float %local.reduce156, float %424), !dbg !20, !frisk.local_reduction !1
  %425 = extractelement <4 x float> %421, i64 2, !dbg !19
  %local.reduce161 = call float @llvm.maxnum.f32(float %local.reduce157, float %425), !dbg !20, !frisk.local_reduction !1
  %426 = extractelement <4 x float> %421, i64 3, !dbg !19
  %local.reduce162 = call float @llvm.maxnum.f32(float %local.reduce158, float %426), !dbg !20, !frisk.local_reduction !1
  %lds.const.gep1307 = getelementptr i8, ptr addrspace(3) %address.local559, i64 13312, !dbg !14
  %427 = load <4 x half>, ptr addrspace(3) %lds.const.gep1307, align 8, !dbg !14
  %lds.const.gep1308 = getelementptr i8, ptr addrspace(3) %170, i64 9280, !dbg !14
  %428 = load <4 x half>, ptr addrspace(3) %lds.const.gep1308, align 8, !dbg !14
  %429 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %114, <4 x half> %427, <4 x float> %422) #9, !dbg !14
  %430 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %112, <4 x half> %428, <4 x float> %342) #9, !dbg !14
  %431 = extractelement <4 x float> %429, i64 0, !dbg !19
  %local.reduce163 = call float @llvm.maxnum.f32(float %local.reduce159, float %431), !dbg !20, !frisk.local_reduction !1
  %432 = extractelement <4 x float> %429, i64 1, !dbg !19
  %local.reduce164 = call float @llvm.maxnum.f32(float %local.reduce160, float %432), !dbg !20, !frisk.local_reduction !1
  %433 = extractelement <4 x float> %429, i64 2, !dbg !19
  %local.reduce165 = call float @llvm.maxnum.f32(float %local.reduce161, float %433), !dbg !20, !frisk.local_reduction !1
  %434 = extractelement <4 x float> %429, i64 3, !dbg !19
  %local.reduce166 = call float @llvm.maxnum.f32(float %local.reduce162, float %434), !dbg !20, !frisk.local_reduction !1
  %lds.const.gep1309 = getelementptr i8, ptr addrspace(3) %170, i64 13376, !dbg !14
  %435 = load <4 x half>, ptr addrspace(3) %lds.const.gep1309, align 8, !dbg !14
  %lds.const.gep1310 = getelementptr i8, ptr addrspace(3) %address.local559, i64 9344, !dbg !14
  %436 = load <4 x half>, ptr addrspace(3) %lds.const.gep1310, align 8, !dbg !14
  %437 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %114, <4 x half> %435, <4 x float> %430) #9, !dbg !14
  %438 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %112, <4 x half> %436, <4 x float> %346) #9, !dbg !14
  %439 = extractelement <4 x float> %437, i64 0, !dbg !19
  %local.reduce167 = call float @llvm.maxnum.f32(float %local.reduce163, float %439), !dbg !20, !frisk.local_reduction !1
  %440 = extractelement <4 x float> %437, i64 1, !dbg !19
  %local.reduce168 = call float @llvm.maxnum.f32(float %local.reduce164, float %440), !dbg !20, !frisk.local_reduction !1
  %441 = extractelement <4 x float> %437, i64 2, !dbg !19
  %local.reduce169 = call float @llvm.maxnum.f32(float %local.reduce165, float %441), !dbg !20, !frisk.local_reduction !1
  %442 = extractelement <4 x float> %437, i64 3, !dbg !19
  %local.reduce170 = call float @llvm.maxnum.f32(float %local.reduce166, float %442), !dbg !20, !frisk.local_reduction !1
  %lds.const.gep1311 = getelementptr i8, ptr addrspace(3) %address.local559, i64 13440, !dbg !14
  %443 = load <4 x half>, ptr addrspace(3) %lds.const.gep1311, align 8, !dbg !14
  %lds.const.gep1312 = getelementptr i8, ptr addrspace(3) %170, i64 9408, !dbg !14
  %444 = load <4 x half>, ptr addrspace(3) %lds.const.gep1312, align 8, !dbg !14
  %445 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %114, <4 x half> %443, <4 x float> %438) #9, !dbg !14
  %446 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %112, <4 x half> %444, <4 x float> %350) #9, !dbg !14
  %447 = extractelement <4 x float> %445, i64 0, !dbg !19
  %local.reduce171 = call float @llvm.maxnum.f32(float %local.reduce167, float %447), !dbg !20, !frisk.local_reduction !1
  %448 = extractelement <4 x float> %445, i64 1, !dbg !19
  %local.reduce172 = call float @llvm.maxnum.f32(float %local.reduce168, float %448), !dbg !20, !frisk.local_reduction !1
  %449 = extractelement <4 x float> %445, i64 2, !dbg !19
  %local.reduce173 = call float @llvm.maxnum.f32(float %local.reduce169, float %449), !dbg !20, !frisk.local_reduction !1
  %450 = extractelement <4 x float> %445, i64 3, !dbg !19
  %local.reduce174 = call float @llvm.maxnum.f32(float %local.reduce170, float %450), !dbg !20, !frisk.local_reduction !1
  %lds.const.gep1313 = getelementptr i8, ptr addrspace(3) %170, i64 13504, !dbg !14
  %451 = load <4 x half>, ptr addrspace(3) %lds.const.gep1313, align 8, !dbg !14
  %lds.const.gep1314 = getelementptr i8, ptr addrspace(3) %address.local559, i64 8192, !dbg !14
  %452 = load <4 x half>, ptr addrspace(3) %lds.const.gep1314, align 8, !dbg !14
  %453 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %114, <4 x half> %451, <4 x float> %446) #9, !dbg !14
  %454 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %115, <4 x half> %452, <4 x float> %354) #9, !dbg !14
  %455 = extractelement <4 x float> %453, i64 0, !dbg !19
  %local.reduce175 = call float @llvm.maxnum.f32(float %local.reduce171, float %455), !dbg !20, !frisk.local_reduction !1
  %456 = extractelement <4 x float> %453, i64 1, !dbg !19
  %local.reduce176 = call float @llvm.maxnum.f32(float %local.reduce172, float %456), !dbg !20, !frisk.local_reduction !1
  %457 = extractelement <4 x float> %453, i64 2, !dbg !19
  %local.reduce177 = call float @llvm.maxnum.f32(float %local.reduce173, float %457), !dbg !20, !frisk.local_reduction !1
  %458 = extractelement <4 x float> %453, i64 3, !dbg !19
  %local.reduce178 = call float @llvm.maxnum.f32(float %local.reduce174, float %458), !dbg !20, !frisk.local_reduction !1
  %local.reduce179 = call float @llvm.maxnum.f32(float %local.reduce175, float %local.reduce176), !dbg !20, !frisk.local_reduction !1
  %local.reduce180 = call float @llvm.maxnum.f32(float %local.reduce177, float %local.reduce178), !dbg !20, !frisk.local_reduction !1
  %459 = call float @llvm.maxnum.f32(float %local.reduce179, float %local.reduce180), !dbg !20, !frisk.local_reduction !1
  %lds.const.gep1315 = getelementptr i8, ptr addrspace(3) %address.local559, i64 12288, !dbg !14
  %460 = load <4 x half>, ptr addrspace(3) %lds.const.gep1315, align 8, !dbg !14
  %lds.const.gep1316 = getelementptr i8, ptr addrspace(3) %170, i64 8256, !dbg !14
  %461 = load <4 x half>, ptr addrspace(3) %lds.const.gep1316, align 8, !dbg !14
  %462 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %116, <4 x half> %460, <4 x float> %454) #9, !dbg !14
  %463 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %115, <4 x half> %461, <4 x float> %358) #9, !dbg !14
  %464 = extractelement <4 x float> %462, i64 0, !dbg !19
  %465 = extractelement <4 x float> %462, i64 1, !dbg !19
  %466 = extractelement <4 x float> %462, i64 2, !dbg !19
  %467 = extractelement <4 x float> %462, i64 3, !dbg !19
  %lds.const.gep1317 = getelementptr i8, ptr addrspace(3) %170, i64 12352, !dbg !14
  %468 = load <4 x half>, ptr addrspace(3) %lds.const.gep1317, align 8, !dbg !14
  %lds.const.gep1318 = getelementptr i8, ptr addrspace(3) %address.local559, i64 8320, !dbg !14
  %469 = load <4 x half>, ptr addrspace(3) %lds.const.gep1318, align 8, !dbg !14
  %470 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %116, <4 x half> %468, <4 x float> %463) #9, !dbg !14
  %471 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %115, <4 x half> %469, <4 x float> %362) #9, !dbg !14
  %472 = extractelement <4 x float> %470, i64 0, !dbg !19
  %473 = extractelement <4 x float> %470, i64 1, !dbg !19
  %474 = extractelement <4 x float> %470, i64 2, !dbg !19
  %475 = extractelement <4 x float> %470, i64 3, !dbg !19
  %lds.const.gep1319 = getelementptr i8, ptr addrspace(3) %address.local559, i64 12416, !dbg !14
  %476 = load <4 x half>, ptr addrspace(3) %lds.const.gep1319, align 8, !dbg !14
  %lds.const.gep1320 = getelementptr i8, ptr addrspace(3) %170, i64 8384, !dbg !14
  %477 = load <4 x half>, ptr addrspace(3) %lds.const.gep1320, align 8, !dbg !14
  %478 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %116, <4 x half> %476, <4 x float> %471) #9, !dbg !14
  %479 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %115, <4 x half> %477, <4 x float> %366) #9, !dbg !14
  %480 = extractelement <4 x float> %478, i64 0, !dbg !19
  %481 = extractelement <4 x float> %478, i64 1, !dbg !19
  %482 = extractelement <4 x float> %478, i64 2, !dbg !19
  %483 = extractelement <4 x float> %478, i64 3, !dbg !19
  %lds.const.gep1321 = getelementptr i8, ptr addrspace(3) %170, i64 12480, !dbg !14
  %484 = load <4 x half>, ptr addrspace(3) %lds.const.gep1321, align 8, !dbg !14
  %lds.const.gep1322 = getelementptr i8, ptr addrspace(3) %address.local559, i64 9216, !dbg !14
  %485 = load <4 x half>, ptr addrspace(3) %lds.const.gep1322, align 8, !dbg !14
  %486 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %116, <4 x half> %484, <4 x float> %479) #9, !dbg !14
  %487 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %115, <4 x half> %485, <4 x float> %370) #9, !dbg !14
  %488 = extractelement <4 x float> %486, i64 0, !dbg !19
  %489 = extractelement <4 x float> %486, i64 1, !dbg !19
  %490 = extractelement <4 x float> %486, i64 2, !dbg !19
  %491 = extractelement <4 x float> %486, i64 3, !dbg !19
  %lds.const.gep1323 = getelementptr i8, ptr addrspace(3) %address.local559, i64 13312, !dbg !14
  %492 = load <4 x half>, ptr addrspace(3) %lds.const.gep1323, align 8, !dbg !14
  %lds.const.gep1324 = getelementptr i8, ptr addrspace(3) %170, i64 9280, !dbg !14
  %493 = load <4 x half>, ptr addrspace(3) %lds.const.gep1324, align 8, !dbg !14
  %494 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %116, <4 x half> %492, <4 x float> %487) #9, !dbg !14
  %495 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %115, <4 x half> %493, <4 x float> %374) #9, !dbg !14
  %496 = extractelement <4 x float> %494, i64 0, !dbg !19
  %497 = extractelement <4 x float> %494, i64 1, !dbg !19
  %498 = extractelement <4 x float> %494, i64 2, !dbg !19
  %499 = extractelement <4 x float> %494, i64 3, !dbg !19
  %lds.const.gep1325 = getelementptr i8, ptr addrspace(3) %170, i64 13376, !dbg !14
  %500 = load <4 x half>, ptr addrspace(3) %lds.const.gep1325, align 8, !dbg !14
  %lds.const.gep1326 = getelementptr i8, ptr addrspace(3) %address.local559, i64 9344, !dbg !14
  %501 = load <4 x half>, ptr addrspace(3) %lds.const.gep1326, align 8, !dbg !14
  %502 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %116, <4 x half> %500, <4 x float> %495) #9, !dbg !14
  %503 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %115, <4 x half> %501, <4 x float> %378) #9, !dbg !14
  %504 = extractelement <4 x float> %502, i64 0, !dbg !19
  %505 = extractelement <4 x float> %502, i64 1, !dbg !19
  %506 = extractelement <4 x float> %502, i64 2, !dbg !19
  %507 = extractelement <4 x float> %502, i64 3, !dbg !19
  %lds.const.gep1327 = getelementptr i8, ptr addrspace(3) %address.local559, i64 13440, !dbg !14
  %508 = load <4 x half>, ptr addrspace(3) %lds.const.gep1327, align 8, !dbg !14
  %lds.const.gep1328 = getelementptr i8, ptr addrspace(3) %170, i64 9408, !dbg !14
  %509 = load <4 x half>, ptr addrspace(3) %lds.const.gep1328, align 8, !dbg !14
  %510 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %116, <4 x half> %508, <4 x float> %503) #9, !dbg !14
  %511 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %115, <4 x half> %509, <4 x float> %381) #9, !dbg !14
  %512 = extractelement <4 x float> %510, i64 0, !dbg !19
  %513 = extractelement <4 x float> %510, i64 1, !dbg !19
  %514 = extractelement <4 x float> %510, i64 2, !dbg !19
  %515 = extractelement <4 x float> %510, i64 3, !dbg !19
  %lds.const.gep1329 = getelementptr i8, ptr addrspace(3) %170, i64 13504, !dbg !14
  %516 = load <4 x half>, ptr addrspace(3) %lds.const.gep1329, align 8, !dbg !14
  %517 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %116, <4 x half> %516, <4 x float> %511) #9, !dbg !14
  %518 = extractelement <4 x float> %517, i64 0, !dbg !19
  %519 = extractelement <4 x float> %517, i64 1, !dbg !19
  %520 = extractelement <4 x float> %517, i64 2, !dbg !19
  %521 = extractelement <4 x float> %517, i64 3, !dbg !19
  %522 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !20
  %523 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %522), !dbg !20
  %524 = add nuw nsw i32 %523, 64, !dbg !20
  %525 = xor i32 %523, 16, !dbg !20
  %526 = and i32 %524, 192, !dbg !20
  %527 = icmp samesign ult i32 %525, %526, !dbg !20
  %528 = select i1 %527, i32 %525, i32 %523, !dbg !20
  %529 = shl nuw nsw i32 %528, 2, !dbg !20
  %530 = bitcast float %459 to i32, !dbg !20
  %531 = call i32 @llvm.amdgcn.ds.bpermute(i32 %529, i32 %530), !dbg !20
  %532 = bitcast i32 %531 to float, !dbg !20
  %533 = call float @llvm.maxnum.f32(float %459, float %532), !dbg !20
  %534 = xor i32 %523, 32, !dbg !20
  %535 = icmp samesign ult i32 %534, %526, !dbg !20
  %536 = select i1 %535, i32 %534, i32 %523, !dbg !20
  %537 = shl nuw nsw i32 %536, 2, !dbg !20
  %538 = bitcast float %533 to i32, !dbg !20
  %539 = call i32 @llvm.amdgcn.ds.bpermute(i32 %537, i32 %538), !dbg !20
  %540 = bitcast i32 %539 to float, !dbg !20
  %541 = call float @llvm.maxnum.f32(float %533, float %540), !dbg !20
  %542 = extractvalue [2 x <1 x float>] %129, 0, !dbg !20
  %543 = insertelement <1 x float> poison, float %541, i64 0, !dbg !20
  %544 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %543, <1 x float> %542), !dbg !20
  %545 = call float @llvm.maxnum.f32(float %464, float 0xFFF0000000000000), !dbg !20, !frisk.local_reduction !1
  %local.reduce120 = call float @llvm.maxnum.f32(float %545, float %472), !dbg !20, !frisk.local_reduction !1
  %local.reduce121 = call float @llvm.maxnum.f32(float %465, float %473), !dbg !20, !frisk.local_reduction !1
  %local.reduce122 = call float @llvm.maxnum.f32(float %466, float %474), !dbg !20, !frisk.local_reduction !1
  %local.reduce123 = call float @llvm.maxnum.f32(float %467, float %475), !dbg !20, !frisk.local_reduction !1
  %local.reduce124 = call float @llvm.maxnum.f32(float %local.reduce120, float %480), !dbg !20, !frisk.local_reduction !1
  %local.reduce125 = call float @llvm.maxnum.f32(float %local.reduce121, float %481), !dbg !20, !frisk.local_reduction !1
  %local.reduce126 = call float @llvm.maxnum.f32(float %local.reduce122, float %482), !dbg !20, !frisk.local_reduction !1
  %local.reduce127 = call float @llvm.maxnum.f32(float %local.reduce123, float %483), !dbg !20, !frisk.local_reduction !1
  %local.reduce128 = call float @llvm.maxnum.f32(float %local.reduce124, float %488), !dbg !20, !frisk.local_reduction !1
  %local.reduce129 = call float @llvm.maxnum.f32(float %local.reduce125, float %489), !dbg !20, !frisk.local_reduction !1
  %local.reduce130 = call float @llvm.maxnum.f32(float %local.reduce126, float %490), !dbg !20, !frisk.local_reduction !1
  %local.reduce131 = call float @llvm.maxnum.f32(float %local.reduce127, float %491), !dbg !20, !frisk.local_reduction !1
  %local.reduce132 = call float @llvm.maxnum.f32(float %local.reduce128, float %496), !dbg !20, !frisk.local_reduction !1
  %local.reduce133 = call float @llvm.maxnum.f32(float %local.reduce129, float %497), !dbg !20, !frisk.local_reduction !1
  %local.reduce134 = call float @llvm.maxnum.f32(float %local.reduce130, float %498), !dbg !20, !frisk.local_reduction !1
  %local.reduce135 = call float @llvm.maxnum.f32(float %local.reduce131, float %499), !dbg !20, !frisk.local_reduction !1
  %local.reduce136 = call float @llvm.maxnum.f32(float %local.reduce132, float %504), !dbg !20, !frisk.local_reduction !1
  %local.reduce137 = call float @llvm.maxnum.f32(float %local.reduce133, float %505), !dbg !20, !frisk.local_reduction !1
  %local.reduce138 = call float @llvm.maxnum.f32(float %local.reduce134, float %506), !dbg !20, !frisk.local_reduction !1
  %local.reduce139 = call float @llvm.maxnum.f32(float %local.reduce135, float %507), !dbg !20, !frisk.local_reduction !1
  %local.reduce140 = call float @llvm.maxnum.f32(float %local.reduce136, float %512), !dbg !20, !frisk.local_reduction !1
  %local.reduce141 = call float @llvm.maxnum.f32(float %local.reduce137, float %513), !dbg !20, !frisk.local_reduction !1
  %local.reduce142 = call float @llvm.maxnum.f32(float %local.reduce138, float %514), !dbg !20, !frisk.local_reduction !1
  %local.reduce143 = call float @llvm.maxnum.f32(float %local.reduce139, float %515), !dbg !20, !frisk.local_reduction !1
  %local.reduce144 = call float @llvm.maxnum.f32(float %local.reduce140, float %518), !dbg !20, !frisk.local_reduction !1
  %local.reduce145 = call float @llvm.maxnum.f32(float %local.reduce141, float %519), !dbg !20, !frisk.local_reduction !1
  %local.reduce146 = call float @llvm.maxnum.f32(float %local.reduce142, float %520), !dbg !20, !frisk.local_reduction !1
  %local.reduce147 = call float @llvm.maxnum.f32(float %local.reduce143, float %521), !dbg !20, !frisk.local_reduction !1
  %local.reduce148 = call float @llvm.maxnum.f32(float %local.reduce144, float %local.reduce145), !dbg !20, !frisk.local_reduction !1
  %local.reduce149 = call float @llvm.maxnum.f32(float %local.reduce146, float %local.reduce147), !dbg !20, !frisk.local_reduction !1
  %546 = call float @llvm.maxnum.f32(float %local.reduce148, float %local.reduce149), !dbg !20, !frisk.local_reduction !1
  %547 = bitcast float %546 to i32, !dbg !20
  %548 = call i32 @llvm.amdgcn.ds.bpermute(i32 %529, i32 %547), !dbg !20
  %549 = bitcast i32 %548 to float, !dbg !20
  %550 = call float @llvm.maxnum.f32(float %546, float %549), !dbg !20
  %551 = bitcast float %550 to i32, !dbg !20
  %552 = call i32 @llvm.amdgcn.ds.bpermute(i32 %537, i32 %551), !dbg !20
  %553 = bitcast i32 %552 to float, !dbg !20
  %554 = call float @llvm.maxnum.f32(float %550, float %553), !dbg !20
  %555 = extractvalue [2 x <1 x float>] %129, 1, !dbg !20
  %556 = insertelement <1 x float> poison, float %554, i64 0, !dbg !20
  %557 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %556, <1 x float> %555), !dbg !20
  %558 = shufflevector <1 x float> %544, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !21
  %559 = fsub <4 x float> %396, %558, !dbg !21
  %560 = fsub <4 x float> %405, %558, !dbg !21
  %561 = fsub <4 x float> %413, %558, !dbg !21
  %562 = fsub <4 x float> %421, %558, !dbg !21
  %563 = fsub <4 x float> %429, %558, !dbg !21
  %564 = fsub <4 x float> %437, %558, !dbg !21
  %565 = fsub <4 x float> %445, %558, !dbg !21
  %566 = fsub <4 x float> %453, %558, !dbg !21
  %567 = shufflevector <1 x float> %557, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !21
  %568 = fsub <4 x float> %462, %567, !dbg !21
  %569 = fsub <4 x float> %470, %567, !dbg !21
  %570 = fsub <4 x float> %478, %567, !dbg !21
  %571 = fsub <4 x float> %486, %567, !dbg !21
  %572 = fsub <4 x float> %494, %567, !dbg !21
  %573 = fsub <4 x float> %502, %567, !dbg !21
  %574 = fsub <4 x float> %510, %567, !dbg !21
  %575 = fsub <4 x float> %517, %567, !dbg !21
  %576 = fsub <1 x float> %542, %544, !dbg !22
  %577 = extractelement <1 x float> %576, i64 0, !dbg !23
  %578 = call float @llvm.amdgcn.exp2.f32(float %577), !dbg !23
  %579 = insertelement <4 x float> poison, float %578, i64 0, !dbg !28
  %580 = shufflevector <4 x float> %579, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !28
  %581 = extractvalue [2 x <1 x float>] %128, 0, !dbg !24
  %582 = extractelement <1 x float> %581, i64 0, !dbg !24
  %583 = fmul float %582, %578, !dbg !24
  %584 = fsub <1 x float> %555, %557, !dbg !22
  %585 = extractelement <1 x float> %584, i64 0, !dbg !23
  %586 = call float @llvm.amdgcn.exp2.f32(float %585), !dbg !23
  %587 = insertelement <4 x float> poison, float %586, i64 0, !dbg !28
  %588 = shufflevector <4 x float> %587, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !28
  %589 = extractvalue [2 x <1 x float>] %128, 1, !dbg !24
  %590 = extractelement <1 x float> %589, i64 0, !dbg !24
  %591 = fmul float %590, %586, !dbg !24
  %592 = extractelement <4 x float> %559, i64 0, !dbg !25
  %593 = call float @llvm.amdgcn.exp2.f32(float %592), !dbg !25
  %594 = insertelement <4 x float> poison, float %593, i64 0, !dbg !27
  %595 = extractelement <4 x float> %559, i64 1, !dbg !25
  %596 = call float @llvm.amdgcn.exp2.f32(float %595), !dbg !25
  %597 = insertelement <4 x float> %594, float %596, i64 1, !dbg !27
  %598 = extractelement <4 x float> %559, i64 2, !dbg !25
  %599 = call float @llvm.amdgcn.exp2.f32(float %598), !dbg !25
  %600 = insertelement <4 x float> %597, float %599, i64 2, !dbg !27
  %601 = extractelement <4 x float> %559, i64 3, !dbg !25
  %602 = call float @llvm.amdgcn.exp2.f32(float %601), !dbg !25
  %603 = insertelement <4 x float> %600, float %602, i64 3, !dbg !27
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %603) #9, !dbg !27, !frisk.register_lifetimes !1
  %604 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !27
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %604) #9, !dbg !25, !frisk.register_lifetimes !1
  %605 = extractelement <4 x float> %560, i64 0, !dbg !25
  %606 = call float @llvm.amdgcn.exp2.f32(float %605), !dbg !25
  %607 = insertelement <4 x float> poison, float %606, i64 0, !dbg !27
  %local.reduce89 = fadd float %593, %606, !dbg !26, !frisk.local_reduction !1
  %608 = extractelement <4 x float> %560, i64 1, !dbg !25
  %609 = call float @llvm.amdgcn.exp2.f32(float %608), !dbg !25
  %610 = insertelement <4 x float> %607, float %609, i64 1, !dbg !27
  %local.reduce90 = fadd float %596, %609, !dbg !26, !frisk.local_reduction !1
  %611 = extractelement <4 x float> %560, i64 2, !dbg !25
  %612 = call float @llvm.amdgcn.exp2.f32(float %611), !dbg !25
  %613 = insertelement <4 x float> %610, float %612, i64 2, !dbg !27
  %local.reduce91 = fadd float %599, %612, !dbg !26, !frisk.local_reduction !1
  %614 = extractelement <4 x float> %560, i64 3, !dbg !25
  %615 = call float @llvm.amdgcn.exp2.f32(float %614), !dbg !25
  %616 = insertelement <4 x float> %613, float %615, i64 3, !dbg !27
  %math.result374 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %616) #9, !dbg !27, !frisk.register_lifetimes !1
  %617 = fptrunc <4 x float> %math.result374 to <4 x half>, !dbg !27
  %math.packed375 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %617) #9, !dbg !26, !frisk.register_lifetimes !1
  %local.reduce92 = fadd float %602, %615, !dbg !26, !frisk.local_reduction !1
  %618 = extractelement <4 x float> %561, i64 0, !dbg !25
  %619 = call float @llvm.amdgcn.exp2.f32(float %618), !dbg !25
  %620 = insertelement <4 x float> poison, float %619, i64 0, !dbg !27
  %local.reduce93 = fadd float %local.reduce89, %619, !dbg !26, !frisk.local_reduction !1
  %621 = extractelement <4 x float> %561, i64 1, !dbg !25
  %622 = call float @llvm.amdgcn.exp2.f32(float %621), !dbg !25
  %623 = insertelement <4 x float> %620, float %622, i64 1, !dbg !27
  %local.reduce94 = fadd float %local.reduce90, %622, !dbg !26, !frisk.local_reduction !1
  %624 = extractelement <4 x float> %561, i64 2, !dbg !25
  %625 = call float @llvm.amdgcn.exp2.f32(float %624), !dbg !25
  %626 = insertelement <4 x float> %623, float %625, i64 2, !dbg !27
  %local.reduce95 = fadd float %local.reduce91, %625, !dbg !26, !frisk.local_reduction !1
  %627 = extractelement <4 x float> %561, i64 3, !dbg !25
  %628 = call float @llvm.amdgcn.exp2.f32(float %627), !dbg !25
  %629 = insertelement <4 x float> %626, float %628, i64 3, !dbg !27
  %math.result376 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %629) #9, !dbg !27, !frisk.register_lifetimes !1
  %630 = fptrunc <4 x float> %math.result376 to <4 x half>, !dbg !27
  %math.packed377 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %630) #9, !dbg !26, !frisk.register_lifetimes !1
  %local.reduce96 = fadd float %local.reduce92, %628, !dbg !26, !frisk.local_reduction !1
  %631 = extractelement <4 x float> %562, i64 0, !dbg !25
  %632 = call float @llvm.amdgcn.exp2.f32(float %631), !dbg !25
  %633 = insertelement <4 x float> poison, float %632, i64 0, !dbg !27
  %local.reduce97 = fadd float %local.reduce93, %632, !dbg !26, !frisk.local_reduction !1
  %634 = extractelement <4 x float> %562, i64 1, !dbg !25
  %635 = call float @llvm.amdgcn.exp2.f32(float %634), !dbg !25
  %636 = insertelement <4 x float> %633, float %635, i64 1, !dbg !27
  %local.reduce98 = fadd float %local.reduce94, %635, !dbg !26, !frisk.local_reduction !1
  %637 = extractelement <4 x float> %562, i64 2, !dbg !25
  %638 = call float @llvm.amdgcn.exp2.f32(float %637), !dbg !25
  %639 = insertelement <4 x float> %636, float %638, i64 2, !dbg !27
  %local.reduce99 = fadd float %local.reduce95, %638, !dbg !26, !frisk.local_reduction !1
  %640 = extractelement <4 x float> %562, i64 3, !dbg !25
  %641 = call float @llvm.amdgcn.exp2.f32(float %640), !dbg !25
  %642 = insertelement <4 x float> %639, float %641, i64 3, !dbg !27
  %math.result378 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %642) #9, !dbg !27, !frisk.register_lifetimes !1
  %643 = fptrunc <4 x float> %math.result378 to <4 x half>, !dbg !27
  %math.packed379 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %643) #9, !dbg !26, !frisk.register_lifetimes !1
  %local.reduce100 = fadd float %local.reduce96, %641, !dbg !26, !frisk.local_reduction !1
  %644 = extractelement <4 x float> %563, i64 0, !dbg !25
  %645 = call float @llvm.amdgcn.exp2.f32(float %644), !dbg !25
  %646 = insertelement <4 x float> poison, float %645, i64 0, !dbg !27
  %local.reduce101 = fadd float %local.reduce97, %645, !dbg !26, !frisk.local_reduction !1
  %647 = extractelement <4 x float> %563, i64 1, !dbg !25
  %648 = call float @llvm.amdgcn.exp2.f32(float %647), !dbg !25
  %649 = insertelement <4 x float> %646, float %648, i64 1, !dbg !27
  %local.reduce102 = fadd float %local.reduce98, %648, !dbg !26, !frisk.local_reduction !1
  %650 = extractelement <4 x float> %563, i64 2, !dbg !25
  %651 = call float @llvm.amdgcn.exp2.f32(float %650), !dbg !25
  %652 = insertelement <4 x float> %649, float %651, i64 2, !dbg !27
  %local.reduce103 = fadd float %local.reduce99, %651, !dbg !26, !frisk.local_reduction !1
  %653 = extractelement <4 x float> %563, i64 3, !dbg !25
  %654 = call float @llvm.amdgcn.exp2.f32(float %653), !dbg !25
  %655 = insertelement <4 x float> %652, float %654, i64 3, !dbg !27
  %math.result380 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %655) #9, !dbg !27, !frisk.register_lifetimes !1
  %656 = fptrunc <4 x float> %math.result380 to <4 x half>, !dbg !27
  %math.packed381 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %656) #9, !dbg !26, !frisk.register_lifetimes !1
  %local.reduce104 = fadd float %local.reduce100, %654, !dbg !26, !frisk.local_reduction !1
  %657 = extractelement <4 x float> %564, i64 0, !dbg !25
  %658 = call float @llvm.amdgcn.exp2.f32(float %657), !dbg !25
  %659 = insertelement <4 x float> poison, float %658, i64 0, !dbg !27
  %local.reduce105 = fadd float %local.reduce101, %658, !dbg !26, !frisk.local_reduction !1
  %660 = extractelement <4 x float> %564, i64 1, !dbg !25
  %661 = call float @llvm.amdgcn.exp2.f32(float %660), !dbg !25
  %662 = insertelement <4 x float> %659, float %661, i64 1, !dbg !27
  %local.reduce106 = fadd float %local.reduce102, %661, !dbg !26, !frisk.local_reduction !1
  %663 = extractelement <4 x float> %564, i64 2, !dbg !25
  %664 = call float @llvm.amdgcn.exp2.f32(float %663), !dbg !25
  %665 = insertelement <4 x float> %662, float %664, i64 2, !dbg !27
  %local.reduce107 = fadd float %local.reduce103, %664, !dbg !26, !frisk.local_reduction !1
  %666 = extractelement <4 x float> %564, i64 3, !dbg !25
  %667 = call float @llvm.amdgcn.exp2.f32(float %666), !dbg !25
  %668 = insertelement <4 x float> %665, float %667, i64 3, !dbg !27
  %math.result382 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %668) #9, !dbg !27, !frisk.register_lifetimes !1
  %669 = fptrunc <4 x float> %math.result382 to <4 x half>, !dbg !27
  %math.packed383 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %669) #9, !dbg !26, !frisk.register_lifetimes !1
  %local.reduce108 = fadd float %local.reduce104, %667, !dbg !26, !frisk.local_reduction !1
  %670 = extractelement <4 x float> %565, i64 0, !dbg !25
  %671 = call float @llvm.amdgcn.exp2.f32(float %670), !dbg !25
  %672 = insertelement <4 x float> poison, float %671, i64 0, !dbg !27
  %local.reduce109 = fadd float %local.reduce105, %671, !dbg !26, !frisk.local_reduction !1
  %673 = extractelement <4 x float> %565, i64 1, !dbg !25
  %674 = call float @llvm.amdgcn.exp2.f32(float %673), !dbg !25
  %675 = insertelement <4 x float> %672, float %674, i64 1, !dbg !27
  %local.reduce110 = fadd float %local.reduce106, %674, !dbg !26, !frisk.local_reduction !1
  %676 = extractelement <4 x float> %565, i64 2, !dbg !25
  %677 = call float @llvm.amdgcn.exp2.f32(float %676), !dbg !25
  %678 = insertelement <4 x float> %675, float %677, i64 2, !dbg !27
  %local.reduce111 = fadd float %local.reduce107, %677, !dbg !26, !frisk.local_reduction !1
  %679 = extractelement <4 x float> %565, i64 3, !dbg !25
  %680 = call float @llvm.amdgcn.exp2.f32(float %679), !dbg !25
  %681 = insertelement <4 x float> %678, float %680, i64 3, !dbg !27
  %math.result384 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %681) #9, !dbg !27, !frisk.register_lifetimes !1
  %682 = fptrunc <4 x float> %math.result384 to <4 x half>, !dbg !27
  %math.packed385 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %682) #9, !dbg !26, !frisk.register_lifetimes !1
  %local.reduce112 = fadd float %local.reduce108, %680, !dbg !26, !frisk.local_reduction !1
  %683 = extractelement <4 x float> %566, i64 0, !dbg !25
  %684 = call float @llvm.amdgcn.exp2.f32(float %683), !dbg !25
  %685 = insertelement <4 x float> poison, float %684, i64 0, !dbg !27
  %local.reduce113 = fadd float %local.reduce109, %684, !dbg !26, !frisk.local_reduction !1
  %686 = extractelement <4 x float> %566, i64 1, !dbg !25
  %687 = call float @llvm.amdgcn.exp2.f32(float %686), !dbg !25
  %688 = insertelement <4 x float> %685, float %687, i64 1, !dbg !27
  %local.reduce114 = fadd float %local.reduce110, %687, !dbg !26, !frisk.local_reduction !1
  %local.reduce117 = fadd float %local.reduce113, %local.reduce114, !dbg !26, !frisk.local_reduction !1
  %689 = extractelement <4 x float> %566, i64 2, !dbg !25
  %690 = call float @llvm.amdgcn.exp2.f32(float %689), !dbg !25
  %691 = insertelement <4 x float> %688, float %690, i64 2, !dbg !27
  %local.reduce115 = fadd float %local.reduce111, %690, !dbg !26, !frisk.local_reduction !1
  %692 = extractelement <4 x float> %566, i64 3, !dbg !25
  %693 = call float @llvm.amdgcn.exp2.f32(float %692), !dbg !25
  %694 = insertelement <4 x float> %691, float %693, i64 3, !dbg !27
  %math.result386 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %694) #9, !dbg !27, !frisk.register_lifetimes !1
  %695 = fptrunc <4 x float> %math.result386 to <4 x half>, !dbg !27
  %math.packed387 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %695) #9, !dbg !26, !frisk.register_lifetimes !1
  %local.reduce116 = fadd float %local.reduce112, %693, !dbg !26, !frisk.local_reduction !1
  %local.reduce118 = fadd float %local.reduce115, %local.reduce116, !dbg !26, !frisk.local_reduction !1
  %696 = fadd float %local.reduce117, %local.reduce118, !dbg !26, !frisk.local_reduction !1
  %697 = fadd float %696, %583, !dbg !26
  %698 = insertelement <1 x float> poison, float %697, i64 0, !dbg !7
  %699 = insertvalue [2 x <1 x float>] poison, <1 x float> %698, 0, !dbg !7
  %700 = extractelement <4 x float> %568, i64 0, !dbg !25
  %701 = call float @llvm.amdgcn.exp2.f32(float %700), !dbg !25
  %702 = insertelement <4 x float> poison, float %701, i64 0, !dbg !27
  %703 = extractelement <4 x float> %568, i64 1, !dbg !25
  %704 = call float @llvm.amdgcn.exp2.f32(float %703), !dbg !25
  %705 = insertelement <4 x float> %702, float %704, i64 1, !dbg !27
  %706 = extractelement <4 x float> %568, i64 2, !dbg !25
  %707 = call float @llvm.amdgcn.exp2.f32(float %706), !dbg !25
  %708 = insertelement <4 x float> %705, float %707, i64 2, !dbg !27
  %709 = extractelement <4 x float> %568, i64 3, !dbg !25
  %710 = call float @llvm.amdgcn.exp2.f32(float %709), !dbg !25
  %711 = insertelement <4 x float> %708, float %710, i64 3, !dbg !27
  %math.result388 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %711) #9, !dbg !27, !frisk.register_lifetimes !1
  %712 = fptrunc <4 x float> %math.result388 to <4 x half>, !dbg !27
  %math.packed389 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %712) #9, !dbg !25, !frisk.register_lifetimes !1
  %713 = extractelement <4 x float> %569, i64 0, !dbg !25
  %714 = call float @llvm.amdgcn.exp2.f32(float %713), !dbg !25
  %715 = insertelement <4 x float> poison, float %714, i64 0, !dbg !27
  %local.reduce = fadd float %701, %714, !dbg !26, !frisk.local_reduction !1
  %716 = extractelement <4 x float> %569, i64 1, !dbg !25
  %717 = call float @llvm.amdgcn.exp2.f32(float %716), !dbg !25
  %718 = insertelement <4 x float> %715, float %717, i64 1, !dbg !27
  %local.reduce59 = fadd float %704, %717, !dbg !26, !frisk.local_reduction !1
  %719 = extractelement <4 x float> %569, i64 2, !dbg !25
  %720 = call float @llvm.amdgcn.exp2.f32(float %719), !dbg !25
  %721 = insertelement <4 x float> %718, float %720, i64 2, !dbg !27
  %local.reduce60 = fadd float %707, %720, !dbg !26, !frisk.local_reduction !1
  %722 = extractelement <4 x float> %569, i64 3, !dbg !25
  %723 = call float @llvm.amdgcn.exp2.f32(float %722), !dbg !25
  %724 = insertelement <4 x float> %721, float %723, i64 3, !dbg !27
  %math.result390 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %724) #9, !dbg !27, !frisk.register_lifetimes !1
  %725 = fptrunc <4 x float> %math.result390 to <4 x half>, !dbg !27
  %math.packed391 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %725) #9, !dbg !26, !frisk.register_lifetimes !1
  %local.reduce61 = fadd float %710, %723, !dbg !26, !frisk.local_reduction !1
  %726 = extractelement <4 x float> %570, i64 0, !dbg !25
  %727 = call float @llvm.amdgcn.exp2.f32(float %726), !dbg !25
  %728 = insertelement <4 x float> poison, float %727, i64 0, !dbg !27
  %local.reduce62 = fadd float %local.reduce, %727, !dbg !26, !frisk.local_reduction !1
  %729 = extractelement <4 x float> %570, i64 1, !dbg !25
  %730 = call float @llvm.amdgcn.exp2.f32(float %729), !dbg !25
  %731 = insertelement <4 x float> %728, float %730, i64 1, !dbg !27
  %local.reduce63 = fadd float %local.reduce59, %730, !dbg !26, !frisk.local_reduction !1
  %732 = extractelement <4 x float> %570, i64 2, !dbg !25
  %733 = call float @llvm.amdgcn.exp2.f32(float %732), !dbg !25
  %734 = insertelement <4 x float> %731, float %733, i64 2, !dbg !27
  %local.reduce64 = fadd float %local.reduce60, %733, !dbg !26, !frisk.local_reduction !1
  %735 = extractelement <4 x float> %570, i64 3, !dbg !25
  %736 = call float @llvm.amdgcn.exp2.f32(float %735), !dbg !25
  %737 = insertelement <4 x float> %734, float %736, i64 3, !dbg !27
  %math.result392 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %737) #9, !dbg !27, !frisk.register_lifetimes !1
  %738 = fptrunc <4 x float> %math.result392 to <4 x half>, !dbg !27
  %math.packed393 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %738) #9, !dbg !26, !frisk.register_lifetimes !1
  %local.reduce65 = fadd float %local.reduce61, %736, !dbg !26, !frisk.local_reduction !1
  %739 = extractelement <4 x float> %571, i64 0, !dbg !25
  %740 = call float @llvm.amdgcn.exp2.f32(float %739), !dbg !25
  %741 = insertelement <4 x float> poison, float %740, i64 0, !dbg !27
  %local.reduce66 = fadd float %local.reduce62, %740, !dbg !26, !frisk.local_reduction !1
  %742 = extractelement <4 x float> %571, i64 1, !dbg !25
  %743 = call float @llvm.amdgcn.exp2.f32(float %742), !dbg !25
  %744 = insertelement <4 x float> %741, float %743, i64 1, !dbg !27
  %local.reduce67 = fadd float %local.reduce63, %743, !dbg !26, !frisk.local_reduction !1
  %745 = extractelement <4 x float> %571, i64 2, !dbg !25
  %746 = call float @llvm.amdgcn.exp2.f32(float %745), !dbg !25
  %747 = insertelement <4 x float> %744, float %746, i64 2, !dbg !27
  %local.reduce68 = fadd float %local.reduce64, %746, !dbg !26, !frisk.local_reduction !1
  %748 = extractelement <4 x float> %571, i64 3, !dbg !25
  %749 = call float @llvm.amdgcn.exp2.f32(float %748), !dbg !25
  %750 = insertelement <4 x float> %747, float %749, i64 3, !dbg !27
  %math.result394 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %750) #9, !dbg !27, !frisk.register_lifetimes !1
  %751 = fptrunc <4 x float> %math.result394 to <4 x half>, !dbg !27
  %math.packed395 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %751) #9, !dbg !26, !frisk.register_lifetimes !1
  %local.reduce69 = fadd float %local.reduce65, %749, !dbg !26, !frisk.local_reduction !1
  %752 = extractelement <4 x float> %572, i64 0, !dbg !25
  %753 = call float @llvm.amdgcn.exp2.f32(float %752), !dbg !25
  %754 = insertelement <4 x float> poison, float %753, i64 0, !dbg !27
  %local.reduce70 = fadd float %local.reduce66, %753, !dbg !26, !frisk.local_reduction !1
  %755 = extractelement <4 x float> %572, i64 1, !dbg !25
  %756 = call float @llvm.amdgcn.exp2.f32(float %755), !dbg !25
  %757 = insertelement <4 x float> %754, float %756, i64 1, !dbg !27
  %local.reduce71 = fadd float %local.reduce67, %756, !dbg !26, !frisk.local_reduction !1
  %758 = extractelement <4 x float> %572, i64 2, !dbg !25
  %759 = call float @llvm.amdgcn.exp2.f32(float %758), !dbg !25
  %760 = insertelement <4 x float> %757, float %759, i64 2, !dbg !27
  %local.reduce72 = fadd float %local.reduce68, %759, !dbg !26, !frisk.local_reduction !1
  %761 = extractelement <4 x float> %572, i64 3, !dbg !25
  %762 = call float @llvm.amdgcn.exp2.f32(float %761), !dbg !25
  %763 = insertelement <4 x float> %760, float %762, i64 3, !dbg !27
  %math.result396 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %763) #9, !dbg !27, !frisk.register_lifetimes !1
  %764 = fptrunc <4 x float> %math.result396 to <4 x half>, !dbg !27
  %math.packed397 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %764) #9, !dbg !26, !frisk.register_lifetimes !1
  %local.reduce73 = fadd float %local.reduce69, %762, !dbg !26, !frisk.local_reduction !1
  %765 = extractelement <4 x float> %573, i64 0, !dbg !25
  %766 = call float @llvm.amdgcn.exp2.f32(float %765), !dbg !25
  %767 = insertelement <4 x float> poison, float %766, i64 0, !dbg !27
  %local.reduce74 = fadd float %local.reduce70, %766, !dbg !26, !frisk.local_reduction !1
  %768 = extractelement <4 x float> %573, i64 1, !dbg !25
  %769 = call float @llvm.amdgcn.exp2.f32(float %768), !dbg !25
  %770 = insertelement <4 x float> %767, float %769, i64 1, !dbg !27
  %local.reduce75 = fadd float %local.reduce71, %769, !dbg !26, !frisk.local_reduction !1
  %771 = extractelement <4 x float> %573, i64 2, !dbg !25
  %772 = call float @llvm.amdgcn.exp2.f32(float %771), !dbg !25
  %773 = insertelement <4 x float> %770, float %772, i64 2, !dbg !27
  %local.reduce76 = fadd float %local.reduce72, %772, !dbg !26, !frisk.local_reduction !1
  %774 = extractelement <4 x float> %573, i64 3, !dbg !25
  %775 = call float @llvm.amdgcn.exp2.f32(float %774), !dbg !25
  %776 = insertelement <4 x float> %773, float %775, i64 3, !dbg !27
  %math.result398 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %776) #9, !dbg !27, !frisk.register_lifetimes !1
  %777 = fptrunc <4 x float> %math.result398 to <4 x half>, !dbg !27
  %math.packed399 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %777) #9, !dbg !26, !frisk.register_lifetimes !1
  %local.reduce77 = fadd float %local.reduce73, %775, !dbg !26, !frisk.local_reduction !1
  %778 = extractelement <4 x float> %574, i64 0, !dbg !25
  %779 = call float @llvm.amdgcn.exp2.f32(float %778), !dbg !25
  %780 = insertelement <4 x float> poison, float %779, i64 0, !dbg !27
  %local.reduce78 = fadd float %local.reduce74, %779, !dbg !26, !frisk.local_reduction !1
  %781 = extractelement <4 x float> %574, i64 1, !dbg !25
  %782 = call float @llvm.amdgcn.exp2.f32(float %781), !dbg !25
  %783 = insertelement <4 x float> %780, float %782, i64 1, !dbg !27
  %local.reduce79 = fadd float %local.reduce75, %782, !dbg !26, !frisk.local_reduction !1
  %784 = extractelement <4 x float> %574, i64 2, !dbg !25
  %785 = call float @llvm.amdgcn.exp2.f32(float %784), !dbg !25
  %786 = insertelement <4 x float> %783, float %785, i64 2, !dbg !27
  %local.reduce80 = fadd float %local.reduce76, %785, !dbg !26, !frisk.local_reduction !1
  %787 = extractelement <4 x float> %574, i64 3, !dbg !25
  %788 = call float @llvm.amdgcn.exp2.f32(float %787), !dbg !25
  %789 = insertelement <4 x float> %786, float %788, i64 3, !dbg !27
  %math.result400 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %789) #9, !dbg !27, !frisk.register_lifetimes !1
  %790 = fptrunc <4 x float> %math.result400 to <4 x half>, !dbg !27
  %math.packed401 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %790) #9, !dbg !26, !frisk.register_lifetimes !1
  %local.reduce81 = fadd float %local.reduce77, %788, !dbg !26, !frisk.local_reduction !1
  %791 = extractelement <4 x float> %575, i64 0, !dbg !25
  %792 = call float @llvm.amdgcn.exp2.f32(float %791), !dbg !25
  %793 = insertelement <4 x float> poison, float %792, i64 0, !dbg !27
  %local.reduce82 = fadd float %local.reduce78, %792, !dbg !26, !frisk.local_reduction !1
  %794 = extractelement <4 x float> %575, i64 1, !dbg !25
  %795 = call float @llvm.amdgcn.exp2.f32(float %794), !dbg !25
  %796 = insertelement <4 x float> %793, float %795, i64 1, !dbg !27
  %local.reduce83 = fadd float %local.reduce79, %795, !dbg !26, !frisk.local_reduction !1
  %local.reduce86 = fadd float %local.reduce82, %local.reduce83, !dbg !26, !frisk.local_reduction !1
  %797 = extractelement <4 x float> %575, i64 2, !dbg !25
  %798 = call float @llvm.amdgcn.exp2.f32(float %797), !dbg !25
  %799 = insertelement <4 x float> %796, float %798, i64 2, !dbg !27
  %local.reduce84 = fadd float %local.reduce80, %798, !dbg !26, !frisk.local_reduction !1
  %800 = extractelement <4 x float> %575, i64 3, !dbg !25
  %801 = call float @llvm.amdgcn.exp2.f32(float %800), !dbg !25
  %802 = insertelement <4 x float> %799, float %801, i64 3, !dbg !27
  %math.result402 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %802) #9, !dbg !27, !frisk.register_lifetimes !1
  %803 = fptrunc <4 x float> %math.result402 to <4 x half>, !dbg !27
  %math.packed403 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %803) #9, !dbg !26, !frisk.register_lifetimes !1
  %local.reduce85 = fadd float %local.reduce81, %801, !dbg !26, !frisk.local_reduction !1
  %local.reduce87 = fadd float %local.reduce84, %local.reduce85, !dbg !26, !frisk.local_reduction !1
  %804 = fadd float %local.reduce86, %local.reduce87, !dbg !26, !frisk.local_reduction !1
  %805 = fadd float %804, %591, !dbg !26
  %806 = insertelement <1 x float> poison, float %805, i64 0, !dbg !7
  %807 = insertvalue [2 x <1 x float>] %699, <1 x float> %806, 1, !dbg !7
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %address.local929 = shl nuw nsw i32 %address.thread498, 3, !dbg !13
  %address.local930 = and i32 %address.local929, 120, !dbg !13
  %address.local931 = shl nuw nsw i32 %address.thread498, 5, !dbg !13
  %address.local932 = and i32 %address.local931, 1536, !dbg !13
  %address.local933 = or disjoint i32 %address.local930, %address.local932, !dbg !13
  %address.local934 = zext nneg i32 %address.local933 to i64, !dbg !13
  %address.local935 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local934, !dbg !13
  %808 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local935, i32 0) #0, !dbg !13
  %809 = bitcast <4 x i32> %808 to <8 x half>, !dbg !13
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
  %810 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local945, i32 0) #0, !dbg !13
  %811 = bitcast <4 x i32> %810 to <8 x half>, !dbg !13
  %address.local946 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local938, !dbg !13
  %address.local947 = getelementptr [2 x i8], ptr addrspace(3) %address.local946, i64 %address.local941, !dbg !13
  %address.local948 = getelementptr [2 x i8], ptr addrspace(3) %address.local947, i64 %address.local943, !dbg !13
  %address.local949 = getelementptr i8, ptr addrspace(3) %address.local948, i64 512, !dbg !13
  %812 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local949, i32 0) #0, !dbg !13
  %813 = bitcast <4 x i32> %812 to <8 x half>, !dbg !13
  %address.local950 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local938, !dbg !13
  %address.local951 = getelementptr [2 x i8], ptr addrspace(3) %address.local950, i64 %address.local941, !dbg !13
  %address.local952 = getelementptr [2 x i8], ptr addrspace(3) %address.local951, i64 %address.local943, !dbg !13
  %address.local953 = getelementptr i8, ptr addrspace(3) %address.local952, i64 768, !dbg !13
  %814 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local953, i32 0) #0, !dbg !13
  %815 = bitcast <4 x i32> %814 to <8 x half>, !dbg !13
  %address.local954 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local934, !dbg !13
  %address.local955 = getelementptr i8, ptr addrspace(3) %address.local954, i64 4096, !dbg !13
  %816 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local955, i32 0) #0, !dbg !13
  %817 = bitcast <4 x i32> %816 to <8 x half>, !dbg !13
  %address.local956 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local938, !dbg !13
  %address.local957 = getelementptr [2 x i8], ptr addrspace(3) %address.local956, i64 %address.local941, !dbg !13
  %address.local958 = getelementptr [2 x i8], ptr addrspace(3) %address.local957, i64 %address.local943, !dbg !13
  %address.local959 = getelementptr i8, ptr addrspace(3) %address.local958, i64 4352, !dbg !13
  %818 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local959, i32 0) #0, !dbg !13
  %819 = bitcast <4 x i32> %818 to <8 x half>, !dbg !13
  %address.local960 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local938, !dbg !13
  %address.local961 = getelementptr [2 x i8], ptr addrspace(3) %address.local960, i64 %address.local941, !dbg !13
  %address.local962 = getelementptr [2 x i8], ptr addrspace(3) %address.local961, i64 %address.local943, !dbg !13
  %address.local963 = getelementptr i8, ptr addrspace(3) %address.local962, i64 4608, !dbg !13
  %820 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local963, i32 0) #0, !dbg !13
  %821 = bitcast <4 x i32> %820 to <8 x half>, !dbg !13
  %address.local964 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local938, !dbg !13
  %address.local965 = getelementptr [2 x i8], ptr addrspace(3) %address.local964, i64 %address.local941, !dbg !13
  %address.local966 = getelementptr [2 x i8], ptr addrspace(3) %address.local965, i64 %address.local943, !dbg !13
  %address.local967 = getelementptr i8, ptr addrspace(3) %address.local966, i64 4864, !dbg !13
  %822 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local967, i32 0) #0, !dbg !13
  %823 = bitcast <4 x i32> %822 to <8 x half>, !dbg !13
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %824 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local504), !dbg !15
  %825 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %.idx14231424 = or disjoint i64 %149, %address.local921, !dbg !15
  %.idx14231424.tr = trunc nuw i64 %.idx14231424 to i32, !dbg !15
  %dma.buffer.offset1342 = shl nuw nsw i32 %.idx14231424.tr, 1, !dbg !15
  %826 = insertelement <2 x i64> poison, i64 %825, i64 0, !dbg !15
  %827 = bitcast <2 x i64> %826 to <4 x i32>, !dbg !15
  %dma.buffer.rsrc1343 = shufflevector <4 x i32> %827, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %dma.buffer.lds1344 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %824), !dbg !15
  %828 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1344, i32 %dma.buffer.offset1342, <4 x i32> %dma.buffer.rsrc1343) #10, !dbg !15
  %829 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local521), !dbg !15
  %830 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %.idx14251426 = or disjoint i64 %149, %address.local927, !dbg !15
  %.idx14251426.tr = trunc nuw i64 %.idx14251426 to i32, !dbg !15
  %dma.buffer.offset1339 = shl nuw nsw i32 %.idx14251426.tr, 1, !dbg !15
  %831 = insertelement <2 x i64> poison, i64 %830, i64 0, !dbg !15
  %832 = bitcast <2 x i64> %831 to <4 x i32>, !dbg !15
  %dma.buffer.rsrc1340 = shufflevector <4 x i32> %832, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %dma.buffer.lds1341 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %829), !dbg !15
  %833 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1341, i32 %dma.buffer.offset1339, <4 x i32> %dma.buffer.rsrc1340) #10, !dbg !15
  %834 = shufflevector <8 x half> %809, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %835 = extractvalue [2 x <8 x float>] %127, 0, !dbg !28
  %836 = shufflevector <8 x float> %835, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %837 = fmul <4 x float> %836, %580, !dbg !28
  %838 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %834, <4 x float> %837) #9, !dbg !13
  %839 = shufflevector <8 x half> %811, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %840 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %839, <4 x float> %838) #9, !dbg !13
  %841 = shufflevector <8 x half> %813, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %842 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %841, <4 x float> %840) #9, !dbg !13
  %843 = shufflevector <8 x half> %815, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %844 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %843, <4 x float> %842) #9, !dbg !13
  %845 = shufflevector <8 x half> %817, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %846 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %845, <4 x float> %844) #9, !dbg !13
  %847 = shufflevector <8 x half> %819, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %848 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %847, <4 x float> %846) #9, !dbg !13
  %849 = shufflevector <8 x half> %821, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %850 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %849, <4 x float> %848) #9, !dbg !13
  %851 = shufflevector <8 x half> %823, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %852 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %851, <4 x float> %850) #9, !dbg !13
  %853 = shufflevector <8 x half> %809, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %854 = shufflevector <8 x float> %835, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %855 = fmul <4 x float> %854, %580, !dbg !28
  %856 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %853, <4 x float> %855) #9, !dbg !13
  %857 = shufflevector <8 x half> %811, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %858 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %857, <4 x float> %856) #9, !dbg !13
  %859 = shufflevector <8 x half> %813, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %860 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %859, <4 x float> %858) #9, !dbg !13
  %861 = shufflevector <8 x half> %815, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %862 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %861, <4 x float> %860) #9, !dbg !13
  %863 = shufflevector <8 x half> %817, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %864 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %863, <4 x float> %862) #9, !dbg !13
  %865 = shufflevector <8 x half> %819, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %866 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %865, <4 x float> %864) #9, !dbg !13
  %867 = shufflevector <8 x half> %821, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %868 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %867, <4 x float> %866) #9, !dbg !13
  %869 = shufflevector <8 x half> %823, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %870 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %869, <4 x float> %868) #9, !dbg !13
  %871 = extractvalue [2 x <8 x float>] %127, 1, !dbg !28
  %872 = shufflevector <8 x float> %871, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %873 = fmul <4 x float> %872, %588, !dbg !28
  %874 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %834, <4 x float> %873) #9, !dbg !13
  %875 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %839, <4 x float> %874) #9, !dbg !13
  %876 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %841, <4 x float> %875) #9, !dbg !13
  %877 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %843, <4 x float> %876) #9, !dbg !13
  %878 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %845, <4 x float> %877) #9, !dbg !13
  %879 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %847, <4 x float> %878) #9, !dbg !13
  %880 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %849, <4 x float> %879) #9, !dbg !13
  %881 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %851, <4 x float> %880) #9, !dbg !13
  %882 = shufflevector <8 x float> %871, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %883 = fmul <4 x float> %882, %588, !dbg !28
  %884 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %853, <4 x float> %883) #9, !dbg !13
  %885 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %857, <4 x float> %884) #9, !dbg !13
  %886 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %859, <4 x float> %885) #9, !dbg !13
  %887 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %861, <4 x float> %886) #9, !dbg !13
  %888 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %863, <4 x float> %887) #9, !dbg !13
  %889 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %865, <4 x float> %888) #9, !dbg !13
  %890 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %867, <4 x float> %889) #9, !dbg !13
  %891 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %869, <4 x float> %890) #9, !dbg !13
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %892 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local935, i32 0) #0, !dbg !13
  %893 = bitcast <4 x i32> %892 to <8 x half>, !dbg !13
  %894 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local945, i32 0) #0, !dbg !13
  %895 = bitcast <4 x i32> %894 to <8 x half>, !dbg !13
  %896 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local949, i32 0) #0, !dbg !13
  %897 = bitcast <4 x i32> %896 to <8 x half>, !dbg !13
  %898 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local953, i32 0) #0, !dbg !13
  %899 = bitcast <4 x i32> %898 to <8 x half>, !dbg !13
  %900 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local955, i32 0) #0, !dbg !13
  %901 = bitcast <4 x i32> %900 to <8 x half>, !dbg !13
  %902 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local959, i32 0) #0, !dbg !13
  %903 = bitcast <4 x i32> %902 to <8 x half>, !dbg !13
  %904 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local963, i32 0) #0, !dbg !13
  %905 = bitcast <4 x i32> %904 to <8 x half>, !dbg !13
  %906 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local967, i32 0) #0, !dbg !13
  %907 = bitcast <4 x i32> %906 to <8 x half>, !dbg !13
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %908 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local504), !dbg !15
  %909 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %.idx14271428 = or disjoint i64 %232, %address.local921, !dbg !15
  %.idx14271428.tr = trunc nuw i64 %.idx14271428 to i32, !dbg !15
  %dma.buffer.offset1336 = shl nuw nsw i32 %.idx14271428.tr, 1, !dbg !15
  %910 = insertelement <2 x i64> poison, i64 %909, i64 0, !dbg !15
  %911 = bitcast <2 x i64> %910 to <4 x i32>, !dbg !15
  %dma.buffer.rsrc1337 = shufflevector <4 x i32> %911, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %dma.buffer.lds1338 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %908), !dbg !15
  %912 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1338, i32 %dma.buffer.offset1336, <4 x i32> %dma.buffer.rsrc1337) #10, !dbg !15
  %913 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local521), !dbg !15
  %914 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %.idx14291430 = or disjoint i64 %232, %address.local927, !dbg !15
  %.idx14291430.tr = trunc nuw i64 %.idx14291430 to i32, !dbg !15
  %dma.buffer.offset1333 = shl nuw nsw i32 %.idx14291430.tr, 1, !dbg !15
  %915 = insertelement <2 x i64> poison, i64 %914, i64 0, !dbg !15
  %916 = bitcast <2 x i64> %915 to <4 x i32>, !dbg !15
  %dma.buffer.rsrc1334 = shufflevector <4 x i32> %916, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %dma.buffer.lds1335 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %913), !dbg !15
  %917 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1335, i32 %dma.buffer.offset1333, <4 x i32> %dma.buffer.rsrc1334) #10, !dbg !15
  %918 = shufflevector <8 x half> %893, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %919 = extractvalue [2 x <8 x float>] %130, 0, !dbg !28
  %920 = shufflevector <8 x float> %919, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %921 = fmul <4 x float> %920, %580, !dbg !28
  %922 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %918, <4 x float> %921) #9, !dbg !13
  %923 = shufflevector <8 x half> %895, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %924 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %923, <4 x float> %922) #9, !dbg !13
  %925 = shufflevector <8 x half> %897, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %926 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %925, <4 x float> %924) #9, !dbg !13
  %927 = shufflevector <8 x half> %899, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %928 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %927, <4 x float> %926) #9, !dbg !13
  %929 = shufflevector <8 x half> %901, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %930 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %929, <4 x float> %928) #9, !dbg !13
  %931 = shufflevector <8 x half> %903, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %932 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %931, <4 x float> %930) #9, !dbg !13
  %933 = shufflevector <8 x half> %905, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %934 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %933, <4 x float> %932) #9, !dbg !13
  %935 = shufflevector <8 x half> %907, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %936 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %935, <4 x float> %934) #9, !dbg !13
  %937 = shufflevector <8 x half> %893, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %938 = shufflevector <8 x float> %919, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %939 = fmul <4 x float> %938, %580, !dbg !28
  %940 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %937, <4 x float> %939) #9, !dbg !13
  %941 = shufflevector <8 x half> %895, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %942 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %941, <4 x float> %940) #9, !dbg !13
  %943 = shufflevector <8 x half> %897, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %944 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %943, <4 x float> %942) #9, !dbg !13
  %945 = shufflevector <8 x half> %899, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %946 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %945, <4 x float> %944) #9, !dbg !13
  %947 = shufflevector <8 x half> %901, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %948 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %947, <4 x float> %946) #9, !dbg !13
  %949 = shufflevector <8 x half> %903, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %950 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %949, <4 x float> %948) #9, !dbg !13
  %951 = shufflevector <8 x half> %905, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %952 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %951, <4 x float> %950) #9, !dbg !13
  %953 = shufflevector <8 x half> %907, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %954 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %953, <4 x float> %952) #9, !dbg !13
  %955 = extractvalue [2 x <8 x float>] %130, 1, !dbg !28
  %956 = shufflevector <8 x float> %955, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %957 = fmul <4 x float> %956, %588, !dbg !28
  %958 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %918, <4 x float> %957) #9, !dbg !13
  %959 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %923, <4 x float> %958) #9, !dbg !13
  %960 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %925, <4 x float> %959) #9, !dbg !13
  %961 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %927, <4 x float> %960) #9, !dbg !13
  %962 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %929, <4 x float> %961) #9, !dbg !13
  %963 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %931, <4 x float> %962) #9, !dbg !13
  %964 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %933, <4 x float> %963) #9, !dbg !13
  %965 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %935, <4 x float> %964) #9, !dbg !13
  %966 = shufflevector <8 x float> %955, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %967 = fmul <4 x float> %966, %588, !dbg !28
  %968 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %937, <4 x float> %967) #9, !dbg !13
  %969 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %941, <4 x float> %968) #9, !dbg !13
  %970 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %943, <4 x float> %969) #9, !dbg !13
  %971 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %945, <4 x float> %970) #9, !dbg !13
  %972 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %947, <4 x float> %971) #9, !dbg !13
  %973 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %949, <4 x float> %972) #9, !dbg !13
  %974 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %951, <4 x float> %973) #9, !dbg !13
  %975 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %953, <4 x float> %974) #9, !dbg !13
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %976 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local935, i32 0) #0, !dbg !13
  %977 = bitcast <4 x i32> %976 to <8 x half>, !dbg !13
  %978 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local945, i32 0) #0, !dbg !13
  %979 = bitcast <4 x i32> %978 to <8 x half>, !dbg !13
  %980 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local949, i32 0) #0, !dbg !13
  %981 = bitcast <4 x i32> %980 to <8 x half>, !dbg !13
  %982 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local953, i32 0) #0, !dbg !13
  %983 = bitcast <4 x i32> %982 to <8 x half>, !dbg !13
  %984 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local955, i32 0) #0, !dbg !13
  %985 = bitcast <4 x i32> %984 to <8 x half>, !dbg !13
  %986 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local959, i32 0) #0, !dbg !13
  %987 = bitcast <4 x i32> %986 to <8 x half>, !dbg !13
  %988 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local963, i32 0) #0, !dbg !13
  %989 = bitcast <4 x i32> %988 to <8 x half>, !dbg !13
  %990 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local967, i32 0) #0, !dbg !13
  %991 = bitcast <4 x i32> %990 to <8 x half>, !dbg !13
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %992 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local504), !dbg !15
  %993 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %.idx14311432 = or disjoint i64 %307, %address.local921, !dbg !15
  %.idx14311432.tr = trunc nuw i64 %.idx14311432 to i32, !dbg !15
  %dma.buffer.offset1330 = shl nuw nsw i32 %.idx14311432.tr, 1, !dbg !15
  %994 = insertelement <2 x i64> poison, i64 %993, i64 0, !dbg !15
  %995 = bitcast <2 x i64> %994 to <4 x i32>, !dbg !15
  %dma.buffer.rsrc1331 = shufflevector <4 x i32> %995, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %dma.buffer.lds1332 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %992), !dbg !15
  %996 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1332, i32 %dma.buffer.offset1330, <4 x i32> %dma.buffer.rsrc1331) #10, !dbg !15
  %997 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local521), !dbg !15
  %998 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %.idx14331434 = or disjoint i64 %307, %address.local927, !dbg !15
  %.idx14331434.tr = trunc nuw i64 %.idx14331434 to i32, !dbg !15
  %dma.buffer.offset = shl nuw nsw i32 %.idx14331434.tr, 1, !dbg !15
  %999 = insertelement <2 x i64> poison, i64 %998, i64 0, !dbg !15
  %1000 = bitcast <2 x i64> %999 to <4 x i32>, !dbg !15
  %dma.buffer.rsrc = shufflevector <4 x i32> %1000, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %dma.buffer.lds = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %997), !dbg !15
  %1001 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc) #10, !dbg !15
  %1002 = shufflevector <8 x half> %977, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1003 = extractvalue [2 x <8 x float>] %131, 0, !dbg !28
  %1004 = shufflevector <8 x float> %1003, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %1005 = fmul <4 x float> %1004, %580, !dbg !28
  %1006 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1002, <4 x float> %1005) #9, !dbg !13
  %1007 = shufflevector <8 x half> %979, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1008 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %1007, <4 x float> %1006) #9, !dbg !13
  %1009 = shufflevector <8 x half> %981, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1010 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %1009, <4 x float> %1008) #9, !dbg !13
  %1011 = shufflevector <8 x half> %983, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1012 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %1011, <4 x float> %1010) #9, !dbg !13
  %1013 = shufflevector <8 x half> %985, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1014 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %1013, <4 x float> %1012) #9, !dbg !13
  %1015 = shufflevector <8 x half> %987, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1016 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %1015, <4 x float> %1014) #9, !dbg !13
  %1017 = shufflevector <8 x half> %989, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1018 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %1017, <4 x float> %1016) #9, !dbg !13
  %1019 = shufflevector <8 x half> %991, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1020 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %1019, <4 x float> %1018) #9, !dbg !13
  %1021 = shufflevector <8 x half> %977, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1022 = shufflevector <8 x float> %1003, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %1023 = fmul <4 x float> %1022, %580, !dbg !28
  %1024 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1021, <4 x float> %1023) #9, !dbg !13
  %1025 = shufflevector <8 x half> %979, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1026 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %1025, <4 x float> %1024) #9, !dbg !13
  %1027 = shufflevector <8 x half> %981, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1028 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %1027, <4 x float> %1026) #9, !dbg !13
  %1029 = shufflevector <8 x half> %983, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1030 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %1029, <4 x float> %1028) #9, !dbg !13
  %1031 = shufflevector <8 x half> %985, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1032 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %1031, <4 x float> %1030) #9, !dbg !13
  %1033 = shufflevector <8 x half> %987, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1034 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %1033, <4 x float> %1032) #9, !dbg !13
  %1035 = shufflevector <8 x half> %989, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1036 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %1035, <4 x float> %1034) #9, !dbg !13
  %1037 = shufflevector <8 x half> %991, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1038 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %1037, <4 x float> %1036) #9, !dbg !13
  %1039 = extractvalue [2 x <8 x float>] %131, 1, !dbg !28
  %1040 = shufflevector <8 x float> %1039, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %1041 = fmul <4 x float> %1040, %588, !dbg !28
  %1042 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %1002, <4 x float> %1041) #9, !dbg !13
  %1043 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %1007, <4 x float> %1042) #9, !dbg !13
  %1044 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %1009, <4 x float> %1043) #9, !dbg !13
  %1045 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %1011, <4 x float> %1044) #9, !dbg !13
  %1046 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %1013, <4 x float> %1045) #9, !dbg !13
  %1047 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %1015, <4 x float> %1046) #9, !dbg !13
  %1048 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %1017, <4 x float> %1047) #9, !dbg !13
  %1049 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %1019, <4 x float> %1048) #9, !dbg !13
  %1050 = shufflevector <8 x float> %1039, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %1051 = fmul <4 x float> %1050, %588, !dbg !28
  %1052 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %1021, <4 x float> %1051) #9, !dbg !13
  %1053 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %1025, <4 x float> %1052) #9, !dbg !13
  %1054 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %1027, <4 x float> %1053) #9, !dbg !13
  %1055 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %1029, <4 x float> %1054) #9, !dbg !13
  %1056 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %1031, <4 x float> %1055) #9, !dbg !13
  %1057 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %1033, <4 x float> %1056) #9, !dbg !13
  %1058 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %1035, <4 x float> %1057) #9, !dbg !13
  %1059 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %1037, <4 x float> %1058) #9, !dbg !13
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %1060 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local935, i32 0) #0, !dbg !13
  %1061 = bitcast <4 x i32> %1060 to <8 x half>, !dbg !13
  %1062 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local945, i32 0) #0, !dbg !13
  %1063 = bitcast <4 x i32> %1062 to <8 x half>, !dbg !13
  %1064 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local949, i32 0) #0, !dbg !13
  %1065 = bitcast <4 x i32> %1064 to <8 x half>, !dbg !13
  %1066 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local953, i32 0) #0, !dbg !13
  %1067 = bitcast <4 x i32> %1066 to <8 x half>, !dbg !13
  %1068 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local955, i32 0) #0, !dbg !13
  %1069 = bitcast <4 x i32> %1068 to <8 x half>, !dbg !13
  %1070 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local959, i32 0) #0, !dbg !13
  %1071 = bitcast <4 x i32> %1070 to <8 x half>, !dbg !13
  %1072 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local963, i32 0) #0, !dbg !13
  %1073 = bitcast <4 x i32> %1072 to <8 x half>, !dbg !13
  %1074 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local967, i32 0) #0, !dbg !13
  %1075 = bitcast <4 x i32> %1074 to <8 x half>, !dbg !13
  %1076 = shufflevector <8 x half> %1061, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1077 = extractvalue [2 x <8 x float>] %132, 0, !dbg !28
  %1078 = shufflevector <8 x float> %1077, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %1079 = fmul <4 x float> %1078, %580, !dbg !28
  %1080 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1076, <4 x float> %1079) #9, !dbg !13
  %1081 = shufflevector <8 x half> %1063, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1082 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %1081, <4 x float> %1080) #9, !dbg !13
  %1083 = shufflevector <8 x half> %1065, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1084 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %1083, <4 x float> %1082) #9, !dbg !13
  %1085 = shufflevector <8 x half> %1067, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1086 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %1085, <4 x float> %1084) #9, !dbg !13
  %1087 = shufflevector <8 x half> %1069, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1088 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %1087, <4 x float> %1086) #9, !dbg !13
  %1089 = shufflevector <8 x half> %1071, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1090 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %1089, <4 x float> %1088) #9, !dbg !13
  %1091 = shufflevector <8 x half> %1073, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1092 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %1091, <4 x float> %1090) #9, !dbg !13
  %1093 = shufflevector <8 x half> %1075, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1094 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %1093, <4 x float> %1092) #9, !dbg !13
  %1095 = shufflevector <8 x half> %1061, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1096 = shufflevector <8 x float> %1077, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %1097 = fmul <4 x float> %1096, %580, !dbg !28
  %1098 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1095, <4 x float> %1097) #9, !dbg !13
  %1099 = shufflevector <8 x half> %1063, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1100 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %1099, <4 x float> %1098) #9, !dbg !13
  %1101 = shufflevector <8 x half> %1065, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1102 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %1101, <4 x float> %1100) #9, !dbg !13
  %1103 = shufflevector <8 x half> %1067, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1104 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %1103, <4 x float> %1102) #9, !dbg !13
  %1105 = shufflevector <8 x half> %1069, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1106 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %1105, <4 x float> %1104) #9, !dbg !13
  %1107 = shufflevector <8 x half> %1071, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1108 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %1107, <4 x float> %1106) #9, !dbg !13
  %1109 = shufflevector <8 x half> %1073, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1110 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %1109, <4 x float> %1108) #9, !dbg !13
  %1111 = shufflevector <8 x half> %1075, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1112 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %1111, <4 x float> %1110) #9, !dbg !13
  %1113 = extractvalue [2 x <8 x float>] %132, 1, !dbg !28
  %1114 = shufflevector <8 x float> %1113, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %1115 = fmul <4 x float> %1114, %588, !dbg !28
  %1116 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %1076, <4 x float> %1115) #9, !dbg !13
  %1117 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %1081, <4 x float> %1116) #9, !dbg !13
  %1118 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %1083, <4 x float> %1117) #9, !dbg !13
  %1119 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %1085, <4 x float> %1118) #9, !dbg !13
  %1120 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %1087, <4 x float> %1119) #9, !dbg !13
  %1121 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %1089, <4 x float> %1120) #9, !dbg !13
  %1122 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %1091, <4 x float> %1121) #9, !dbg !13
  %1123 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %1093, <4 x float> %1122) #9, !dbg !13
  %1124 = shufflevector <8 x float> %1113, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %1125 = fmul <4 x float> %1124, %588, !dbg !28
  %1126 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %1095, <4 x float> %1125) #9, !dbg !13
  %1127 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %1099, <4 x float> %1126) #9, !dbg !13
  %1128 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %1101, <4 x float> %1127) #9, !dbg !13
  %1129 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %1103, <4 x float> %1128) #9, !dbg !13
  %1130 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %1105, <4 x float> %1129) #9, !dbg !13
  %1131 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %1107, <4 x float> %1130) #9, !dbg !13
  %1132 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %1109, <4 x float> %1131) #9, !dbg !13
  %1133 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %1111, <4 x float> %1132) #9, !dbg !13
  %1134 = shufflevector <4 x float> %852, <4 x float> %870, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1135 = insertvalue [2 x <8 x float>] poison, <8 x float> %1134, 0, !dbg !7
  %1136 = shufflevector <4 x float> %881, <4 x float> %891, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1137 = insertvalue [2 x <8 x float>] %1135, <8 x float> %1136, 1, !dbg !7
  %1138 = insertvalue [2 x <1 x float>] poison, <1 x float> %544, 0, !dbg !7
  %1139 = insertvalue [2 x <1 x float>] %1138, <1 x float> %557, 1, !dbg !7
  %1140 = shufflevector <4 x float> %936, <4 x float> %954, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1141 = insertvalue [2 x <8 x float>] poison, <8 x float> %1140, 0, !dbg !7
  %1142 = shufflevector <4 x float> %965, <4 x float> %975, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1143 = insertvalue [2 x <8 x float>] %1141, <8 x float> %1142, 1, !dbg !7
  %1144 = shufflevector <4 x float> %1020, <4 x float> %1038, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1145 = insertvalue [2 x <8 x float>] poison, <8 x float> %1144, 0, !dbg !7
  %1146 = shufflevector <4 x float> %1049, <4 x float> %1059, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1147 = insertvalue [2 x <8 x float>] %1145, <8 x float> %1146, 1, !dbg !7
  %1148 = shufflevector <4 x float> %1094, <4 x float> %1112, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1149 = insertvalue [2 x <8 x float>] poison, <8 x float> %1148, 0, !dbg !7
  %1150 = shufflevector <4 x float> %1123, <4 x float> %1133, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1151 = insertvalue [2 x <8 x float>] %1149, <8 x float> %1150, 1, !dbg !7
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %1152 = add nuw nsw i64 %126, 1, !dbg !7
  br label %125, !dbg !7

1153:                                             ; preds = %125
  %lane.sum.field184 = extractvalue [2 x <1 x float>] %128, 1, !dbg !7
  %lane.sum.local185 = extractelement <1 x float> %lane.sum.field184, i64 0, !dbg !7
  %lane.sum.lane.lo186 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !7
  %lane.sum.lane187 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %lane.sum.lane.lo186), !dbg !7
  %1154 = shl nuw nsw i32 %lane.sum.lane187, 2, !dbg !7
  %1155 = xor i32 %1154, 64, !dbg !7
  %1156 = bitcast float %lane.sum.local185 to i32, !dbg !7
  %lane.sum.shuffle188 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1155, i32 %1156), !dbg !7
  %1157 = bitcast i32 %lane.sum.shuffle188 to float, !dbg !7
  %lane.sum.final189 = fadd float %lane.sum.local185, %1157, !dbg !7
  %1158 = shl nuw nsw i32 %lane.sum.lane187, 2, !dbg !7
  %1159 = xor i32 %1158, 128, !dbg !7
  %1160 = bitcast float %lane.sum.final189 to i32, !dbg !7
  %lane.sum.shuffle190 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1159, i32 %1160), !dbg !7
  %1161 = bitcast i32 %lane.sum.shuffle190 to float, !dbg !7
  %lane.sum.final191 = fadd float %lane.sum.final189, %1161, !dbg !7
  %1162 = insertelement <1 x float> poison, float %lane.sum.final191, i64 0, !dbg !7
  %lane.sum.field = extractvalue [2 x <1 x float>] %128, 0, !dbg !7
  %lane.sum.local = extractelement <1 x float> %lane.sum.field, i64 0, !dbg !7
  %lane.sum.lane.lo = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !7
  %lane.sum.lane = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %lane.sum.lane.lo), !dbg !7
  %1163 = shl nuw nsw i32 %lane.sum.lane, 2, !dbg !7
  %1164 = xor i32 %1163, 64, !dbg !7
  %1165 = bitcast float %lane.sum.local to i32, !dbg !7
  %lane.sum.shuffle = call i32 @llvm.amdgcn.ds.bpermute(i32 %1164, i32 %1165), !dbg !7
  %1166 = bitcast i32 %lane.sum.shuffle to float, !dbg !7
  %lane.sum.final = fadd float %lane.sum.local, %1166, !dbg !7
  %1167 = shl nuw nsw i32 %lane.sum.lane, 2, !dbg !7
  %1168 = xor i32 %1167, 128, !dbg !7
  %1169 = bitcast float %lane.sum.final to i32, !dbg !7
  %lane.sum.shuffle182 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1168, i32 %1169), !dbg !7
  %1170 = bitcast i32 %lane.sum.shuffle182 to float, !dbg !7
  %lane.sum.final183 = fadd float %lane.sum.final, %1170, !dbg !7
  %1171 = insertelement <1 x float> poison, float %lane.sum.final183, i64 0, !dbg !7
  %1172 = and i32 %31, 16, !dbg !7
  %.not = icmp eq i32 %1172, 0, !dbg !7
  %1173 = and i32 %31, 32, !dbg !7
  %.not36 = icmp eq i32 %1173, 0, !dbg !7
  %1174 = extractvalue [2 x <8 x float>] %132, 0, !dbg !7
  %1175 = extractelement <8 x float> %1174, i64 0, !dbg !7
  %1176 = extractelement <8 x float> %1174, i64 1, !dbg !7
  %1177 = extractelement <8 x float> %1174, i64 2, !dbg !7
  %1178 = extractelement <8 x float> %1174, i64 3, !dbg !7
  %1179 = select i1 %.not, float %1176, float %1175, !dbg !7
  %1180 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !7
  %1181 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1180), !dbg !7
  %1182 = add nuw nsw i32 %1181, 64, !dbg !7
  %1183 = xor i32 %1181, 16, !dbg !7
  %1184 = and i32 %1182, 192, !dbg !7
  %1185 = icmp samesign ult i32 %1183, %1184, !dbg !7
  %1186 = select i1 %1185, i32 %1183, i32 %1181, !dbg !7
  %1187 = shl nuw nsw i32 %1186, 2, !dbg !7
  %1188 = bitcast float %1179 to i32, !dbg !7
  %1189 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1188), !dbg !7
  %1190 = bitcast i32 %1189 to float, !dbg !7
  %1191 = select i1 %.not, float %1175, float %1190, !dbg !7
  %1192 = select i1 %.not, float %1190, float %1176, !dbg !7
  %1193 = select i1 %.not, float %1178, float %1177, !dbg !7
  %1194 = bitcast float %1193 to i32, !dbg !7
  %1195 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1194), !dbg !7
  %1196 = bitcast i32 %1195 to float, !dbg !7
  %1197 = select i1 %.not, float %1177, float %1196, !dbg !7
  %1198 = select i1 %.not, float %1196, float %1178, !dbg !7
  %1199 = select i1 %.not36, float %1197, float %1191, !dbg !7
  %1200 = xor i32 %1181, 32, !dbg !7
  %1201 = icmp samesign ult i32 %1200, %1184, !dbg !7
  %1202 = select i1 %1201, i32 %1200, i32 %1181, !dbg !7
  %1203 = shl nuw nsw i32 %1202, 2, !dbg !7
  %1204 = bitcast float %1199 to i32, !dbg !7
  %1205 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1204), !dbg !7
  %1206 = bitcast i32 %1205 to float, !dbg !7
  %1207 = select i1 %.not36, float %1191, float %1206, !dbg !7
  %1208 = select i1 %.not36, float %1206, float %1197, !dbg !7
  %1209 = select i1 %.not36, float %1198, float %1192, !dbg !7
  %1210 = bitcast float %1209 to i32, !dbg !7
  %1211 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1210), !dbg !7
  %1212 = bitcast i32 %1211 to float, !dbg !7
  %1213 = select i1 %.not36, float %1192, float %1212, !dbg !7
  %1214 = select i1 %.not36, float %1212, float %1198, !dbg !7
  %1215 = extractvalue [2 x <8 x float>] %131, 0, !dbg !7
  %1216 = extractelement <8 x float> %1215, i64 0, !dbg !7
  %1217 = extractelement <8 x float> %1215, i64 1, !dbg !7
  %1218 = extractelement <8 x float> %1215, i64 2, !dbg !7
  %1219 = extractelement <8 x float> %1215, i64 3, !dbg !7
  %1220 = select i1 %.not, float %1217, float %1216, !dbg !7
  %1221 = bitcast float %1220 to i32, !dbg !7
  %1222 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1221), !dbg !7
  %1223 = bitcast i32 %1222 to float, !dbg !7
  %1224 = select i1 %.not, float %1216, float %1223, !dbg !7
  %1225 = select i1 %.not, float %1223, float %1217, !dbg !7
  %1226 = select i1 %.not, float %1219, float %1218, !dbg !7
  %1227 = bitcast float %1226 to i32, !dbg !7
  %1228 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1227), !dbg !7
  %1229 = bitcast i32 %1228 to float, !dbg !7
  %1230 = select i1 %.not, float %1218, float %1229, !dbg !7
  %1231 = select i1 %.not, float %1229, float %1219, !dbg !7
  %1232 = select i1 %.not36, float %1230, float %1224, !dbg !7
  %1233 = bitcast float %1232 to i32, !dbg !7
  %1234 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1233), !dbg !7
  %1235 = bitcast i32 %1234 to float, !dbg !7
  %1236 = select i1 %.not36, float %1224, float %1235, !dbg !7
  %1237 = select i1 %.not36, float %1235, float %1230, !dbg !7
  %1238 = select i1 %.not36, float %1231, float %1225, !dbg !7
  %1239 = bitcast float %1238 to i32, !dbg !7
  %1240 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1239), !dbg !7
  %1241 = bitcast i32 %1240 to float, !dbg !7
  %1242 = select i1 %.not36, float %1225, float %1241, !dbg !7
  %1243 = select i1 %.not36, float %1241, float %1231, !dbg !7
  %1244 = extractvalue [2 x <8 x float>] %130, 0, !dbg !7
  %1245 = extractelement <8 x float> %1244, i64 0, !dbg !7
  %1246 = extractelement <8 x float> %1244, i64 1, !dbg !7
  %1247 = extractelement <8 x float> %1244, i64 2, !dbg !7
  %1248 = extractelement <8 x float> %1244, i64 3, !dbg !7
  %1249 = select i1 %.not, float %1246, float %1245, !dbg !7
  %1250 = bitcast float %1249 to i32, !dbg !7
  %1251 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1250), !dbg !7
  %1252 = bitcast i32 %1251 to float, !dbg !7
  %1253 = select i1 %.not, float %1245, float %1252, !dbg !7
  %1254 = select i1 %.not, float %1252, float %1246, !dbg !7
  %1255 = select i1 %.not, float %1248, float %1247, !dbg !7
  %1256 = bitcast float %1255 to i32, !dbg !7
  %1257 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1256), !dbg !7
  %1258 = bitcast i32 %1257 to float, !dbg !7
  %1259 = select i1 %.not, float %1247, float %1258, !dbg !7
  %1260 = select i1 %.not, float %1258, float %1248, !dbg !7
  %1261 = select i1 %.not36, float %1259, float %1253, !dbg !7
  %1262 = bitcast float %1261 to i32, !dbg !7
  %1263 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1262), !dbg !7
  %1264 = bitcast i32 %1263 to float, !dbg !7
  %1265 = select i1 %.not36, float %1253, float %1264, !dbg !7
  %1266 = select i1 %.not36, float %1264, float %1259, !dbg !7
  %1267 = select i1 %.not36, float %1260, float %1254, !dbg !7
  %1268 = bitcast float %1267 to i32, !dbg !7
  %1269 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1268), !dbg !7
  %1270 = bitcast i32 %1269 to float, !dbg !7
  %1271 = select i1 %.not36, float %1254, float %1270, !dbg !7
  %1272 = select i1 %.not36, float %1270, float %1260, !dbg !7
  %1273 = extractvalue [2 x <8 x float>] %127, 0, !dbg !7
  %1274 = extractelement <8 x float> %1273, i64 0, !dbg !7
  %1275 = extractelement <8 x float> %1273, i64 1, !dbg !7
  %1276 = extractelement <8 x float> %1273, i64 2, !dbg !7
  %1277 = extractelement <8 x float> %1273, i64 3, !dbg !7
  %1278 = select i1 %.not, float %1275, float %1274, !dbg !7
  %1279 = bitcast float %1278 to i32, !dbg !7
  %1280 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1279), !dbg !7
  %1281 = bitcast i32 %1280 to float, !dbg !7
  %1282 = select i1 %.not, float %1274, float %1281, !dbg !7
  %1283 = select i1 %.not, float %1281, float %1275, !dbg !7
  %1284 = select i1 %.not, float %1277, float %1276, !dbg !7
  %1285 = bitcast float %1284 to i32, !dbg !7
  %1286 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1285), !dbg !7
  %1287 = bitcast i32 %1286 to float, !dbg !7
  %1288 = select i1 %.not, float %1276, float %1287, !dbg !7
  %1289 = select i1 %.not, float %1287, float %1277, !dbg !7
  %1290 = select i1 %.not36, float %1288, float %1282, !dbg !7
  %1291 = bitcast float %1290 to i32, !dbg !7
  %1292 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1291), !dbg !7
  %1293 = select i1 %.not36, float %1289, float %1283, !dbg !7
  %1294 = bitcast float %1293 to i32, !dbg !7
  %1295 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1294), !dbg !7
  %1296 = shufflevector <1 x float> %1171, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !16
  %1297 = extractelement <8 x float> %1174, i64 4, !dbg !7
  %1298 = extractelement <8 x float> %1174, i64 5, !dbg !7
  %1299 = extractelement <8 x float> %1174, i64 6, !dbg !7
  %1300 = extractelement <8 x float> %1174, i64 7, !dbg !7
  %1301 = select i1 %.not, float %1298, float %1297, !dbg !7
  %1302 = bitcast float %1301 to i32, !dbg !7
  %1303 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1302), !dbg !7
  %1304 = bitcast i32 %1303 to float, !dbg !7
  %1305 = select i1 %.not, float %1297, float %1304, !dbg !7
  %1306 = select i1 %.not, float %1304, float %1298, !dbg !7
  %1307 = select i1 %.not, float %1300, float %1299, !dbg !7
  %1308 = bitcast float %1307 to i32, !dbg !7
  %1309 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1308), !dbg !7
  %1310 = bitcast i32 %1309 to float, !dbg !7
  %1311 = select i1 %.not, float %1299, float %1310, !dbg !7
  %1312 = select i1 %.not, float %1310, float %1300, !dbg !7
  %1313 = select i1 %.not36, float %1311, float %1305, !dbg !7
  %1314 = bitcast float %1313 to i32, !dbg !7
  %1315 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1314), !dbg !7
  %1316 = bitcast i32 %1315 to float, !dbg !7
  %1317 = select i1 %.not36, float %1305, float %1316, !dbg !7
  %1318 = select i1 %.not36, float %1316, float %1311, !dbg !7
  %1319 = select i1 %.not36, float %1312, float %1306, !dbg !7
  %1320 = bitcast float %1319 to i32, !dbg !7
  %1321 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1320), !dbg !7
  %1322 = bitcast i32 %1321 to float, !dbg !7
  %1323 = select i1 %.not36, float %1306, float %1322, !dbg !7
  %1324 = select i1 %.not36, float %1322, float %1312, !dbg !7
  %1325 = extractelement <8 x float> %1215, i64 4, !dbg !7
  %1326 = extractelement <8 x float> %1215, i64 5, !dbg !7
  %1327 = extractelement <8 x float> %1215, i64 6, !dbg !7
  %1328 = extractelement <8 x float> %1215, i64 7, !dbg !7
  %1329 = select i1 %.not, float %1326, float %1325, !dbg !7
  %1330 = bitcast float %1329 to i32, !dbg !7
  %1331 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1330), !dbg !7
  %1332 = bitcast i32 %1331 to float, !dbg !7
  %1333 = select i1 %.not, float %1325, float %1332, !dbg !7
  %1334 = select i1 %.not, float %1332, float %1326, !dbg !7
  %1335 = select i1 %.not, float %1328, float %1327, !dbg !7
  %1336 = bitcast float %1335 to i32, !dbg !7
  %1337 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1336), !dbg !7
  %1338 = bitcast i32 %1337 to float, !dbg !7
  %1339 = select i1 %.not, float %1327, float %1338, !dbg !7
  %1340 = select i1 %.not, float %1338, float %1328, !dbg !7
  %1341 = select i1 %.not36, float %1339, float %1333, !dbg !7
  %1342 = bitcast float %1341 to i32, !dbg !7
  %1343 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1342), !dbg !7
  %1344 = bitcast i32 %1343 to float, !dbg !7
  %1345 = select i1 %.not36, float %1333, float %1344, !dbg !7
  %1346 = select i1 %.not36, float %1344, float %1339, !dbg !7
  %1347 = select i1 %.not36, float %1340, float %1334, !dbg !7
  %1348 = bitcast float %1347 to i32, !dbg !7
  %1349 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1348), !dbg !7
  %1350 = bitcast i32 %1349 to float, !dbg !7
  %1351 = select i1 %.not36, float %1334, float %1350, !dbg !7
  %1352 = select i1 %.not36, float %1350, float %1340, !dbg !7
  %1353 = extractelement <8 x float> %1244, i64 4, !dbg !7
  %1354 = extractelement <8 x float> %1244, i64 5, !dbg !7
  %1355 = extractelement <8 x float> %1244, i64 6, !dbg !7
  %1356 = extractelement <8 x float> %1244, i64 7, !dbg !7
  %1357 = select i1 %.not, float %1354, float %1353, !dbg !7
  %1358 = bitcast float %1357 to i32, !dbg !7
  %1359 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1358), !dbg !7
  %1360 = bitcast i32 %1359 to float, !dbg !7
  %1361 = select i1 %.not, float %1353, float %1360, !dbg !7
  %1362 = select i1 %.not, float %1360, float %1354, !dbg !7
  %1363 = select i1 %.not, float %1356, float %1355, !dbg !7
  %1364 = bitcast float %1363 to i32, !dbg !7
  %1365 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1364), !dbg !7
  %1366 = bitcast i32 %1365 to float, !dbg !7
  %1367 = select i1 %.not, float %1355, float %1366, !dbg !7
  %1368 = select i1 %.not, float %1366, float %1356, !dbg !7
  %1369 = select i1 %.not36, float %1367, float %1361, !dbg !7
  %1370 = bitcast float %1369 to i32, !dbg !7
  %1371 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1370), !dbg !7
  %1372 = bitcast i32 %1371 to float, !dbg !7
  %1373 = select i1 %.not36, float %1361, float %1372, !dbg !7
  %1374 = select i1 %.not36, float %1372, float %1367, !dbg !7
  %1375 = select i1 %.not36, float %1368, float %1362, !dbg !7
  %1376 = bitcast float %1375 to i32, !dbg !7
  %1377 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1376), !dbg !7
  %1378 = bitcast i32 %1377 to float, !dbg !7
  %1379 = select i1 %.not36, float %1362, float %1378, !dbg !7
  %1380 = select i1 %.not36, float %1378, float %1368, !dbg !7
  %1381 = extractelement <8 x float> %1273, i64 4, !dbg !7
  %1382 = extractelement <8 x float> %1273, i64 5, !dbg !7
  %1383 = extractelement <8 x float> %1273, i64 6, !dbg !7
  %1384 = extractelement <8 x float> %1273, i64 7, !dbg !7
  %1385 = select i1 %.not, float %1382, float %1381, !dbg !7
  %1386 = bitcast float %1385 to i32, !dbg !7
  %1387 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1386), !dbg !7
  %1388 = bitcast i32 %1387 to float, !dbg !7
  %1389 = select i1 %.not, float %1381, float %1388, !dbg !7
  %1390 = select i1 %.not, float %1388, float %1382, !dbg !7
  %1391 = select i1 %.not, float %1384, float %1383, !dbg !7
  %1392 = bitcast float %1391 to i32, !dbg !7
  %1393 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1392), !dbg !7
  %1394 = bitcast i32 %1393 to float, !dbg !7
  %1395 = select i1 %.not, float %1383, float %1394, !dbg !7
  %1396 = select i1 %.not, float %1394, float %1384, !dbg !7
  %1397 = select i1 %.not36, float %1395, float %1389, !dbg !7
  %1398 = bitcast float %1397 to i32, !dbg !7
  %1399 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1398), !dbg !7
  %1400 = select i1 %.not36, float %1396, float %1390, !dbg !7
  %1401 = bitcast float %1400 to i32, !dbg !7
  %1402 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1401), !dbg !7
  %1403 = extractvalue [2 x <8 x float>] %132, 1, !dbg !7
  %1404 = extractelement <8 x float> %1403, i64 0, !dbg !7
  %1405 = extractelement <8 x float> %1403, i64 1, !dbg !7
  %1406 = extractelement <8 x float> %1403, i64 2, !dbg !7
  %1407 = extractelement <8 x float> %1403, i64 3, !dbg !7
  %1408 = select i1 %.not, float %1405, float %1404, !dbg !7
  %1409 = bitcast float %1408 to i32, !dbg !7
  %1410 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1409), !dbg !7
  %1411 = bitcast i32 %1410 to float, !dbg !7
  %1412 = select i1 %.not, float %1404, float %1411, !dbg !7
  %1413 = select i1 %.not, float %1411, float %1405, !dbg !7
  %1414 = select i1 %.not, float %1407, float %1406, !dbg !7
  %1415 = bitcast float %1414 to i32, !dbg !7
  %1416 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1415), !dbg !7
  %1417 = bitcast i32 %1416 to float, !dbg !7
  %1418 = select i1 %.not, float %1406, float %1417, !dbg !7
  %1419 = select i1 %.not, float %1417, float %1407, !dbg !7
  %1420 = select i1 %.not36, float %1418, float %1412, !dbg !7
  %1421 = bitcast float %1420 to i32, !dbg !7
  %1422 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1421), !dbg !7
  %1423 = bitcast i32 %1422 to float, !dbg !7
  %1424 = select i1 %.not36, float %1412, float %1423, !dbg !7
  %1425 = select i1 %.not36, float %1423, float %1418, !dbg !7
  %1426 = select i1 %.not36, float %1419, float %1413, !dbg !7
  %1427 = bitcast float %1426 to i32, !dbg !7
  %1428 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1427), !dbg !7
  %1429 = bitcast i32 %1428 to float, !dbg !7
  %1430 = select i1 %.not36, float %1413, float %1429, !dbg !7
  %1431 = select i1 %.not36, float %1429, float %1419, !dbg !7
  %1432 = extractvalue [2 x <8 x float>] %131, 1, !dbg !7
  %1433 = extractelement <8 x float> %1432, i64 0, !dbg !7
  %1434 = extractelement <8 x float> %1432, i64 1, !dbg !7
  %1435 = extractelement <8 x float> %1432, i64 2, !dbg !7
  %1436 = extractelement <8 x float> %1432, i64 3, !dbg !7
  %1437 = select i1 %.not, float %1434, float %1433, !dbg !7
  %1438 = bitcast float %1437 to i32, !dbg !7
  %1439 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1438), !dbg !7
  %1440 = bitcast i32 %1439 to float, !dbg !7
  %1441 = select i1 %.not, float %1433, float %1440, !dbg !7
  %1442 = select i1 %.not, float %1440, float %1434, !dbg !7
  %1443 = select i1 %.not, float %1436, float %1435, !dbg !7
  %1444 = bitcast float %1443 to i32, !dbg !7
  %1445 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1444), !dbg !7
  %1446 = bitcast i32 %1445 to float, !dbg !7
  %1447 = select i1 %.not, float %1435, float %1446, !dbg !7
  %1448 = select i1 %.not, float %1446, float %1436, !dbg !7
  %1449 = select i1 %.not36, float %1447, float %1441, !dbg !7
  %1450 = bitcast float %1449 to i32, !dbg !7
  %1451 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1450), !dbg !7
  %1452 = bitcast i32 %1451 to float, !dbg !7
  %1453 = select i1 %.not36, float %1441, float %1452, !dbg !7
  %1454 = select i1 %.not36, float %1452, float %1447, !dbg !7
  %1455 = select i1 %.not36, float %1448, float %1442, !dbg !7
  %1456 = bitcast float %1455 to i32, !dbg !7
  %1457 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1456), !dbg !7
  %1458 = bitcast i32 %1457 to float, !dbg !7
  %1459 = select i1 %.not36, float %1442, float %1458, !dbg !7
  %1460 = select i1 %.not36, float %1458, float %1448, !dbg !7
  %1461 = extractvalue [2 x <8 x float>] %130, 1, !dbg !7
  %1462 = extractelement <8 x float> %1461, i64 0, !dbg !7
  %1463 = extractelement <8 x float> %1461, i64 1, !dbg !7
  %1464 = extractelement <8 x float> %1461, i64 2, !dbg !7
  %1465 = extractelement <8 x float> %1461, i64 3, !dbg !7
  %1466 = select i1 %.not, float %1463, float %1462, !dbg !7
  %1467 = bitcast float %1466 to i32, !dbg !7
  %1468 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1467), !dbg !7
  %1469 = bitcast i32 %1468 to float, !dbg !7
  %1470 = select i1 %.not, float %1462, float %1469, !dbg !7
  %1471 = select i1 %.not, float %1469, float %1463, !dbg !7
  %1472 = select i1 %.not, float %1465, float %1464, !dbg !7
  %1473 = bitcast float %1472 to i32, !dbg !7
  %1474 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1473), !dbg !7
  %1475 = bitcast i32 %1474 to float, !dbg !7
  %1476 = select i1 %.not, float %1464, float %1475, !dbg !7
  %1477 = select i1 %.not, float %1475, float %1465, !dbg !7
  %1478 = select i1 %.not36, float %1476, float %1470, !dbg !7
  %1479 = bitcast float %1478 to i32, !dbg !7
  %1480 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1479), !dbg !7
  %1481 = bitcast i32 %1480 to float, !dbg !7
  %1482 = select i1 %.not36, float %1470, float %1481, !dbg !7
  %1483 = select i1 %.not36, float %1481, float %1476, !dbg !7
  %1484 = select i1 %.not36, float %1477, float %1471, !dbg !7
  %1485 = bitcast float %1484 to i32, !dbg !7
  %1486 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1485), !dbg !7
  %1487 = bitcast i32 %1486 to float, !dbg !7
  %1488 = select i1 %.not36, float %1471, float %1487, !dbg !7
  %1489 = select i1 %.not36, float %1487, float %1477, !dbg !7
  %1490 = extractvalue [2 x <8 x float>] %127, 1, !dbg !7
  %1491 = extractelement <8 x float> %1490, i64 0, !dbg !7
  %1492 = extractelement <8 x float> %1490, i64 1, !dbg !7
  %1493 = extractelement <8 x float> %1490, i64 2, !dbg !7
  %1494 = extractelement <8 x float> %1490, i64 3, !dbg !7
  %1495 = select i1 %.not, float %1492, float %1491, !dbg !7
  %1496 = bitcast float %1495 to i32, !dbg !7
  %1497 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1496), !dbg !7
  %1498 = bitcast i32 %1497 to float, !dbg !7
  %1499 = select i1 %.not, float %1491, float %1498, !dbg !7
  %1500 = select i1 %.not, float %1498, float %1492, !dbg !7
  %1501 = select i1 %.not, float %1494, float %1493, !dbg !7
  %1502 = bitcast float %1501 to i32, !dbg !7
  %1503 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1502), !dbg !7
  %1504 = bitcast i32 %1503 to float, !dbg !7
  %1505 = select i1 %.not, float %1493, float %1504, !dbg !7
  %1506 = select i1 %.not, float %1504, float %1494, !dbg !7
  %1507 = select i1 %.not36, float %1505, float %1499, !dbg !7
  %1508 = bitcast float %1507 to i32, !dbg !7
  %1509 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1508), !dbg !7
  %1510 = select i1 %.not36, float %1506, float %1500, !dbg !7
  %1511 = bitcast float %1510 to i32, !dbg !7
  %1512 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1511), !dbg !7
  %1513 = shufflevector <1 x float> %1162, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !16
  %1514 = extractelement <8 x float> %1403, i64 4, !dbg !7
  %1515 = extractelement <8 x float> %1403, i64 5, !dbg !7
  %1516 = extractelement <8 x float> %1403, i64 6, !dbg !7
  %1517 = extractelement <8 x float> %1403, i64 7, !dbg !7
  %1518 = select i1 %.not, float %1515, float %1514, !dbg !7
  %1519 = bitcast float %1518 to i32, !dbg !7
  %1520 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1519), !dbg !7
  %1521 = bitcast i32 %1520 to float, !dbg !7
  %1522 = select i1 %.not, float %1514, float %1521, !dbg !7
  %1523 = select i1 %.not, float %1521, float %1515, !dbg !7
  %1524 = select i1 %.not, float %1517, float %1516, !dbg !7
  %1525 = bitcast float %1524 to i32, !dbg !7
  %1526 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1525), !dbg !7
  %1527 = bitcast i32 %1526 to float, !dbg !7
  %1528 = select i1 %.not, float %1516, float %1527, !dbg !7
  %1529 = select i1 %.not, float %1527, float %1517, !dbg !7
  %1530 = select i1 %.not36, float %1528, float %1522, !dbg !7
  %1531 = bitcast float %1530 to i32, !dbg !7
  %1532 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1531), !dbg !7
  %1533 = bitcast i32 %1532 to float, !dbg !7
  %1534 = select i1 %.not36, float %1522, float %1533, !dbg !7
  %1535 = select i1 %.not36, float %1533, float %1528, !dbg !7
  %1536 = select i1 %.not36, float %1529, float %1523, !dbg !7
  %1537 = bitcast float %1536 to i32, !dbg !7
  %1538 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1537), !dbg !7
  %1539 = bitcast i32 %1538 to float, !dbg !7
  %1540 = select i1 %.not36, float %1523, float %1539, !dbg !7
  %1541 = select i1 %.not36, float %1539, float %1529, !dbg !7
  %1542 = extractelement <8 x float> %1432, i64 4, !dbg !7
  %1543 = extractelement <8 x float> %1432, i64 5, !dbg !7
  %1544 = extractelement <8 x float> %1432, i64 6, !dbg !7
  %1545 = extractelement <8 x float> %1432, i64 7, !dbg !7
  %1546 = select i1 %.not, float %1543, float %1542, !dbg !7
  %1547 = bitcast float %1546 to i32, !dbg !7
  %1548 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1547), !dbg !7
  %1549 = bitcast i32 %1548 to float, !dbg !7
  %1550 = select i1 %.not, float %1542, float %1549, !dbg !7
  %1551 = select i1 %.not, float %1549, float %1543, !dbg !7
  %1552 = select i1 %.not, float %1545, float %1544, !dbg !7
  %1553 = bitcast float %1552 to i32, !dbg !7
  %1554 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1553), !dbg !7
  %1555 = bitcast i32 %1554 to float, !dbg !7
  %1556 = select i1 %.not, float %1544, float %1555, !dbg !7
  %1557 = select i1 %.not, float %1555, float %1545, !dbg !7
  %1558 = select i1 %.not36, float %1556, float %1550, !dbg !7
  %1559 = bitcast float %1558 to i32, !dbg !7
  %1560 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1559), !dbg !7
  %1561 = bitcast i32 %1560 to float, !dbg !7
  %1562 = select i1 %.not36, float %1550, float %1561, !dbg !7
  %1563 = select i1 %.not36, float %1561, float %1556, !dbg !7
  %1564 = select i1 %.not36, float %1557, float %1551, !dbg !7
  %1565 = bitcast float %1564 to i32, !dbg !7
  %1566 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1565), !dbg !7
  %1567 = bitcast i32 %1566 to float, !dbg !7
  %1568 = select i1 %.not36, float %1551, float %1567, !dbg !7
  %1569 = select i1 %.not36, float %1567, float %1557, !dbg !7
  %1570 = extractelement <8 x float> %1461, i64 4, !dbg !7
  %1571 = extractelement <8 x float> %1461, i64 5, !dbg !7
  %1572 = extractelement <8 x float> %1461, i64 6, !dbg !7
  %1573 = extractelement <8 x float> %1461, i64 7, !dbg !7
  %1574 = select i1 %.not, float %1571, float %1570, !dbg !7
  %1575 = bitcast float %1574 to i32, !dbg !7
  %1576 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1575), !dbg !7
  %1577 = bitcast i32 %1576 to float, !dbg !7
  %1578 = select i1 %.not, float %1570, float %1577, !dbg !7
  %1579 = select i1 %.not, float %1577, float %1571, !dbg !7
  %1580 = select i1 %.not, float %1573, float %1572, !dbg !7
  %1581 = bitcast float %1580 to i32, !dbg !7
  %1582 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1581), !dbg !7
  %1583 = bitcast i32 %1582 to float, !dbg !7
  %1584 = select i1 %.not, float %1572, float %1583, !dbg !7
  %1585 = select i1 %.not, float %1583, float %1573, !dbg !7
  %1586 = select i1 %.not36, float %1584, float %1578, !dbg !7
  %1587 = bitcast float %1586 to i32, !dbg !7
  %1588 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1587), !dbg !7
  %1589 = bitcast i32 %1588 to float, !dbg !7
  %1590 = select i1 %.not36, float %1578, float %1589, !dbg !7
  %1591 = select i1 %.not36, float %1589, float %1584, !dbg !7
  %1592 = select i1 %.not36, float %1585, float %1579, !dbg !7
  %1593 = bitcast float %1592 to i32, !dbg !7
  %1594 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1593), !dbg !7
  %1595 = bitcast i32 %1594 to float, !dbg !7
  %1596 = select i1 %.not36, float %1579, float %1595, !dbg !7
  %1597 = select i1 %.not36, float %1595, float %1585, !dbg !7
  %1598 = extractelement <8 x float> %1490, i64 4, !dbg !7
  %1599 = extractelement <8 x float> %1490, i64 5, !dbg !7
  %1600 = extractelement <8 x float> %1490, i64 6, !dbg !7
  %1601 = extractelement <8 x float> %1490, i64 7, !dbg !7
  %1602 = select i1 %.not, float %1599, float %1598, !dbg !7
  %1603 = bitcast float %1602 to i32, !dbg !7
  %1604 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1603), !dbg !7
  %1605 = bitcast i32 %1604 to float, !dbg !7
  %1606 = select i1 %.not, float %1598, float %1605, !dbg !7
  %1607 = select i1 %.not, float %1605, float %1599, !dbg !7
  %1608 = select i1 %.not, float %1601, float %1600, !dbg !7
  %1609 = bitcast float %1608 to i32, !dbg !7
  %1610 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1609), !dbg !7
  %1611 = bitcast i32 %1610 to float, !dbg !7
  %1612 = select i1 %.not, float %1600, float %1611, !dbg !7
  %1613 = select i1 %.not, float %1611, float %1601, !dbg !7
  %1614 = select i1 %.not36, float %1612, float %1606, !dbg !7
  %1615 = bitcast float %1614 to i32, !dbg !7
  %1616 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1615), !dbg !7
  %1617 = select i1 %.not36, float %1613, float %1607, !dbg !7
  %1618 = bitcast float %1617 to i32, !dbg !7
  %1619 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1618), !dbg !7
  %1620 = bitcast i32 %1616 to float, !dbg !7
  %1621 = select i1 %.not36, float %1606, float %1620, !dbg !7
  %1622 = insertelement <4 x float> poison, float %1621, i64 0, !dbg !16
  %1623 = bitcast i32 %1619 to float, !dbg !7
  %1624 = select i1 %.not36, float %1607, float %1623, !dbg !7
  %1625 = insertelement <4 x float> %1622, float %1624, i64 1, !dbg !16
  %1626 = select i1 %.not36, float %1620, float %1612, !dbg !7
  %1627 = insertelement <4 x float> %1625, float %1626, i64 2, !dbg !16
  %1628 = select i1 %.not36, float %1623, float %1613, !dbg !7
  %1629 = insertelement <4 x float> %1627, float %1628, i64 3, !dbg !16
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1513, !dbg !16
  %normalized = fmul <4 x float> %1629, %row.reciprocal, !dbg !16
  %1630 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !17
  %1631 = extractelement <4 x half> %1630, i64 3, !dbg !17
  %1632 = extractelement <4 x half> %1630, i64 2, !dbg !17
  %1633 = extractelement <4 x half> %1630, i64 1, !dbg !17
  %1634 = extractelement <4 x half> %1630, i64 0, !dbg !17
  %1635 = bitcast i32 %1509 to float, !dbg !7
  %1636 = select i1 %.not36, float %1499, float %1635, !dbg !7
  %1637 = insertelement <4 x float> poison, float %1636, i64 0, !dbg !16
  %1638 = bitcast i32 %1512 to float, !dbg !7
  %1639 = select i1 %.not36, float %1500, float %1638, !dbg !7
  %1640 = insertelement <4 x float> %1637, float %1639, i64 1, !dbg !16
  %1641 = select i1 %.not36, float %1635, float %1505, !dbg !7
  %1642 = insertelement <4 x float> %1640, float %1641, i64 2, !dbg !16
  %1643 = select i1 %.not36, float %1638, float %1506, !dbg !7
  %1644 = insertelement <4 x float> %1642, float %1643, i64 3, !dbg !16
  %normalized37 = fmul <4 x float> %1644, %row.reciprocal, !dbg !16
  %1645 = fptrunc <4 x float> %normalized37 to <4 x half>, !dbg !17
  %1646 = extractelement <4 x half> %1645, i64 3, !dbg !17
  %1647 = extractelement <4 x half> %1645, i64 2, !dbg !17
  %1648 = extractelement <4 x half> %1645, i64 1, !dbg !17
  %1649 = extractelement <4 x half> %1645, i64 0, !dbg !17
  %1650 = bitcast i32 %1399 to float, !dbg !7
  %1651 = select i1 %.not36, float %1389, float %1650, !dbg !7
  %1652 = insertelement <4 x float> poison, float %1651, i64 0, !dbg !16
  %1653 = bitcast i32 %1402 to float, !dbg !7
  %1654 = select i1 %.not36, float %1390, float %1653, !dbg !7
  %1655 = insertelement <4 x float> %1652, float %1654, i64 1, !dbg !16
  %1656 = select i1 %.not36, float %1650, float %1395, !dbg !7
  %1657 = insertelement <4 x float> %1655, float %1656, i64 2, !dbg !16
  %1658 = select i1 %.not36, float %1653, float %1396, !dbg !7
  %1659 = insertelement <4 x float> %1657, float %1658, i64 3, !dbg !16
  %row.reciprocal38 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1296, !dbg !16
  %normalized39 = fmul <4 x float> %1659, %row.reciprocal38, !dbg !16
  %1660 = fptrunc <4 x float> %normalized39 to <4 x half>, !dbg !17
  %1661 = extractelement <4 x half> %1660, i64 3, !dbg !17
  %1662 = extractelement <4 x half> %1660, i64 2, !dbg !17
  %1663 = extractelement <4 x half> %1660, i64 1, !dbg !17
  %1664 = extractelement <4 x half> %1660, i64 0, !dbg !17
  %1665 = bitcast i32 %1292 to float, !dbg !7
  %1666 = select i1 %.not36, float %1282, float %1665, !dbg !7
  %1667 = insertelement <4 x float> poison, float %1666, i64 0, !dbg !16
  %1668 = bitcast i32 %1295 to float, !dbg !7
  %1669 = select i1 %.not36, float %1283, float %1668, !dbg !7
  %1670 = insertelement <4 x float> %1667, float %1669, i64 1, !dbg !16
  %1671 = select i1 %.not36, float %1665, float %1288, !dbg !7
  %1672 = insertelement <4 x float> %1670, float %1671, i64 2, !dbg !16
  %1673 = select i1 %.not36, float %1668, float %1289, !dbg !7
  %1674 = insertelement <4 x float> %1672, float %1673, i64 3, !dbg !16
  %normalized40 = fmul <4 x float> %1674, %row.reciprocal38, !dbg !16
  %1675 = fptrunc <4 x float> %normalized40 to <4 x half>, !dbg !17
  %1676 = extractelement <4 x half> %1675, i64 3, !dbg !17
  %1677 = extractelement <4 x half> %1675, i64 2, !dbg !17
  %1678 = extractelement <4 x half> %1675, i64 1, !dbg !17
  %1679 = extractelement <4 x half> %1675, i64 0, !dbg !17
  %1680 = add nsw i64 %16, %22, !dbg !18
  %1681 = shl i64 %1680, 19, !dbg !18
  %1682 = and i64 %1681, 16252928, !dbg !18
  %address.thread974 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !18, !range !32, !frisk.register_lifetimes !1
  %address.local975 = zext nneg i32 %address.thread974 to i64, !dbg !8
  %address.local976 = add i64 %25, %address.local975, !dbg !18
  %1683 = lshr i64 %address.local975, 2, !dbg !18
  %address.local978 = and i64 %1683, 48, !dbg !18
  %address.local979 = add i64 %address.local976, %address.local978, !dbg !18
  %address.local980 = and i64 %address.local975, 240, !dbg !18
  %address.local981 = sub i64 %address.local979, %address.local980, !dbg !18
  %address.local982 = add i64 %address.local981, %26, !dbg !18
  %address.local983 = shl nuw nsw i64 %address.local982, 7, !dbg !18
  %address.local984 = add nuw nsw i64 %1682, %address.local983, !dbg !18
  %address.local985 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local984, !dbg !18
  %address.local986 = lshr i64 %address.local975, 2, !dbg !10
  %address.local987 = and i64 %address.local986, 12, !dbg !10
  %address.local988 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local987, !dbg !18
  store half %1679, ptr addrspace(1) %address.local988, align 2, !dbg !18
  %address.local989 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local984, !dbg !18
  %address.local990 = or disjoint i64 %address.local987, 1, !dbg !18
  %address.local991 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local989, i64 %address.local990, !dbg !18
  store half %1678, ptr addrspace(1) %address.local991, align 2, !dbg !18
  %address.local992 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local984, !dbg !18
  %address.local993 = or disjoint i64 %address.local987, 2, !dbg !18
  %address.local994 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local992, i64 %address.local993, !dbg !18
  store half %1677, ptr addrspace(1) %address.local994, align 2, !dbg !18
  %address.local995 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local984, !dbg !18
  %address.local996 = or disjoint i64 %address.local987, 3, !dbg !18
  %address.local997 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local995, i64 %address.local996, !dbg !18
  store half %1676, ptr addrspace(1) %address.local997, align 2, !dbg !18
  %address.local998 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local984, !dbg !18
  %address.local999 = or disjoint i64 %address.local987, 16, !dbg !18
  %address.local1000 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local998, i64 %address.local999, !dbg !18
  store half %1664, ptr addrspace(1) %address.local1000, align 2, !dbg !18
  %address.local1001 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local984, !dbg !18
  %address.local1002 = or disjoint i64 %address.local987, 17, !dbg !18
  %address.local1003 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1001, i64 %address.local1002, !dbg !18
  store half %1663, ptr addrspace(1) %address.local1003, align 2, !dbg !18
  %address.local1004 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local984, !dbg !18
  %address.local1005 = or disjoint i64 %address.local987, 18, !dbg !18
  %address.local1006 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1004, i64 %address.local1005, !dbg !18
  store half %1662, ptr addrspace(1) %address.local1006, align 2, !dbg !18
  %address.local1007 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local984, !dbg !18
  %address.local1008 = or disjoint i64 %address.local987, 19, !dbg !18
  %address.local1009 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1007, i64 %address.local1008, !dbg !18
  store half %1661, ptr addrspace(1) %address.local1009, align 2, !dbg !18
  %address.local1010 = shl i64 %address.local982, 7, !dbg !18
  %address.local1011 = add i64 %address.local1010, 8192, !dbg !18
  %address.local1012 = add nuw nsw i64 %1682, %address.local1011, !dbg !18
  %address.local1013 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1012, !dbg !18
  %address.local1014 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local987, !dbg !18
  store half %1649, ptr addrspace(1) %address.local1014, align 2, !dbg !18
  %address.local1015 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1012, !dbg !18
  %address.local1016 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1015, i64 %address.local990, !dbg !18
  store half %1648, ptr addrspace(1) %address.local1016, align 2, !dbg !18
  %address.local1017 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1012, !dbg !18
  %address.local1018 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1017, i64 %address.local993, !dbg !18
  store half %1647, ptr addrspace(1) %address.local1018, align 2, !dbg !18
  %address.local1019 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1012, !dbg !18
  %address.local1020 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1019, i64 %address.local996, !dbg !18
  store half %1646, ptr addrspace(1) %address.local1020, align 2, !dbg !18
  %address.local1021 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1012, !dbg !18
  %address.local1022 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1021, i64 %address.local999, !dbg !18
  store half %1634, ptr addrspace(1) %address.local1022, align 2, !dbg !18
  %address.local1023 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1012, !dbg !18
  %address.local1024 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1023, i64 %address.local1002, !dbg !18
  store half %1633, ptr addrspace(1) %address.local1024, align 2, !dbg !18
  %address.local1025 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1012, !dbg !18
  %address.local1026 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1025, i64 %address.local1005, !dbg !18
  store half %1632, ptr addrspace(1) %address.local1026, align 2, !dbg !18
  %address.local1027 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1012, !dbg !18
  %address.local1028 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1027, i64 %address.local1008, !dbg !18
  store half %1631, ptr addrspace(1) %address.local1028, align 2, !dbg !18
  %1684 = insertelement <4 x float> poison, float %1590, i64 0, !dbg !16
  %1685 = insertelement <4 x float> %1684, float %1596, i64 1, !dbg !16
  %1686 = insertelement <4 x float> %1685, float %1591, i64 2, !dbg !16
  %1687 = insertelement <4 x float> %1686, float %1597, i64 3, !dbg !16
  %row.reciprocal41 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1513, !dbg !16
  %normalized42 = fmul <4 x float> %1687, %row.reciprocal41, !dbg !16
  %1688 = fptrunc <4 x float> %normalized42 to <4 x half>, !dbg !17
  %1689 = extractelement <4 x half> %1688, i64 3, !dbg !17
  %1690 = extractelement <4 x half> %1688, i64 2, !dbg !17
  %1691 = extractelement <4 x half> %1688, i64 1, !dbg !17
  %1692 = extractelement <4 x half> %1688, i64 0, !dbg !17
  %1693 = insertelement <4 x float> poison, float %1482, i64 0, !dbg !16
  %1694 = insertelement <4 x float> %1693, float %1488, i64 1, !dbg !16
  %1695 = insertelement <4 x float> %1694, float %1483, i64 2, !dbg !16
  %1696 = insertelement <4 x float> %1695, float %1489, i64 3, !dbg !16
  %normalized43 = fmul <4 x float> %1696, %row.reciprocal41, !dbg !16
  %1697 = fptrunc <4 x float> %normalized43 to <4 x half>, !dbg !17
  %1698 = extractelement <4 x half> %1697, i64 3, !dbg !17
  %1699 = extractelement <4 x half> %1697, i64 2, !dbg !17
  %1700 = extractelement <4 x half> %1697, i64 1, !dbg !17
  %1701 = extractelement <4 x half> %1697, i64 0, !dbg !17
  %1702 = insertelement <4 x float> poison, float %1373, i64 0, !dbg !16
  %1703 = insertelement <4 x float> %1702, float %1379, i64 1, !dbg !16
  %1704 = insertelement <4 x float> %1703, float %1374, i64 2, !dbg !16
  %1705 = insertelement <4 x float> %1704, float %1380, i64 3, !dbg !16
  %row.reciprocal44 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1296, !dbg !16
  %normalized45 = fmul <4 x float> %1705, %row.reciprocal44, !dbg !16
  %1706 = fptrunc <4 x float> %normalized45 to <4 x half>, !dbg !17
  %1707 = extractelement <4 x half> %1706, i64 3, !dbg !17
  %1708 = extractelement <4 x half> %1706, i64 2, !dbg !17
  %1709 = extractelement <4 x half> %1706, i64 1, !dbg !17
  %1710 = extractelement <4 x half> %1706, i64 0, !dbg !17
  %1711 = insertelement <4 x float> poison, float %1265, i64 0, !dbg !16
  %1712 = insertelement <4 x float> %1711, float %1271, i64 1, !dbg !16
  %1713 = insertelement <4 x float> %1712, float %1266, i64 2, !dbg !16
  %1714 = insertelement <4 x float> %1713, float %1272, i64 3, !dbg !16
  %normalized46 = fmul <4 x float> %1714, %row.reciprocal44, !dbg !16
  %1715 = fptrunc <4 x float> %normalized46 to <4 x half>, !dbg !17
  %1716 = extractelement <4 x half> %1715, i64 3, !dbg !17
  %1717 = extractelement <4 x half> %1715, i64 2, !dbg !17
  %1718 = extractelement <4 x half> %1715, i64 1, !dbg !17
  %1719 = extractelement <4 x half> %1715, i64 0, !dbg !17
  %1720 = shl nuw nsw i64 %24, 19, !dbg !18
  %address.thread1029 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !18, !range !32, !frisk.register_lifetimes !1
  %address.local1030 = zext nneg i32 %address.thread1029 to i64, !dbg !8
  %address.local1031 = add i64 %25, %address.local1030, !dbg !18
  %1721 = lshr i64 %address.local1030, 2, !dbg !18
  %address.local1033 = and i64 %1721, 48, !dbg !18
  %address.local1034 = add i64 %address.local1031, %address.local1033, !dbg !18
  %address.local1035 = and i64 %address.local1030, 240, !dbg !18
  %address.local1036 = sub i64 %address.local1034, %address.local1035, !dbg !18
  %address.local1037 = add i64 %address.local1036, %26, !dbg !18
  %address.local1038 = shl nuw nsw i64 %address.local1037, 7, !dbg !18
  %address.local1039 = add nuw nsw i64 %1720, %address.local1038, !dbg !18
  %address.local1040 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1039, !dbg !18
  %address.local1041 = lshr i64 %address.local1030, 2, !dbg !10
  %address.local1042 = and i64 %address.local1041, 12, !dbg !10
  %address.local1043 = or disjoint i64 %address.local1042, 32, !dbg !18
  %address.local1044 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1040, i64 %address.local1043, !dbg !18
  store half %1719, ptr addrspace(1) %address.local1044, align 2, !dbg !18
  %address.local1045 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1039, !dbg !18
  %address.local1046 = or disjoint i64 %address.local1042, 33, !dbg !18
  %address.local1047 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1045, i64 %address.local1046, !dbg !18
  store half %1718, ptr addrspace(1) %address.local1047, align 2, !dbg !18
  %address.local1048 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1039, !dbg !18
  %address.local1049 = or disjoint i64 %address.local1042, 34, !dbg !18
  %address.local1050 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1048, i64 %address.local1049, !dbg !18
  store half %1717, ptr addrspace(1) %address.local1050, align 2, !dbg !18
  %address.local1051 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1039, !dbg !18
  %address.local1052 = or disjoint i64 %address.local1042, 35, !dbg !18
  %address.local1053 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1051, i64 %address.local1052, !dbg !18
  store half %1716, ptr addrspace(1) %address.local1053, align 2, !dbg !18
  %address.local1054 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1039, !dbg !18
  %address.local1055 = or disjoint i64 %address.local1042, 48, !dbg !18
  %address.local1056 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1055, !dbg !18
  store half %1710, ptr addrspace(1) %address.local1056, align 2, !dbg !18
  %address.local1057 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1039, !dbg !18
  %address.local1058 = or disjoint i64 %address.local1042, 49, !dbg !18
  %address.local1059 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1057, i64 %address.local1058, !dbg !18
  store half %1709, ptr addrspace(1) %address.local1059, align 2, !dbg !18
  %address.local1060 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1039, !dbg !18
  %address.local1061 = or disjoint i64 %address.local1042, 50, !dbg !18
  %address.local1062 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1060, i64 %address.local1061, !dbg !18
  store half %1708, ptr addrspace(1) %address.local1062, align 2, !dbg !18
  %address.local1063 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1039, !dbg !18
  %address.local1064 = or i64 %address.local1041, 51, !dbg !18
  %address.local1065 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1063, i64 %address.local1064, !dbg !18
  store half %1707, ptr addrspace(1) %address.local1065, align 2, !dbg !18
  %address.local1066 = shl i64 %address.local1037, 7, !dbg !18
  %address.local1067 = add i64 %address.local1066, 8192, !dbg !18
  %address.local1068 = add nuw nsw i64 %1720, %address.local1067, !dbg !18
  %address.local1069 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1068, !dbg !18
  %address.local1070 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1069, i64 %address.local1043, !dbg !18
  store half %1701, ptr addrspace(1) %address.local1070, align 2, !dbg !18
  %address.local1071 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1068, !dbg !18
  %address.local1072 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1071, i64 %address.local1046, !dbg !18
  store half %1700, ptr addrspace(1) %address.local1072, align 2, !dbg !18
  %address.local1073 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1068, !dbg !18
  %address.local1074 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1073, i64 %address.local1049, !dbg !18
  store half %1699, ptr addrspace(1) %address.local1074, align 2, !dbg !18
  %address.local1075 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1068, !dbg !18
  %address.local1076 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1075, i64 %address.local1052, !dbg !18
  store half %1698, ptr addrspace(1) %address.local1076, align 2, !dbg !18
  %address.local1077 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1068, !dbg !18
  %address.local1078 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1077, i64 %address.local1055, !dbg !18
  store half %1692, ptr addrspace(1) %address.local1078, align 2, !dbg !18
  %address.local1079 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1068, !dbg !18
  %address.local1080 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1079, i64 %address.local1058, !dbg !18
  store half %1691, ptr addrspace(1) %address.local1080, align 2, !dbg !18
  %address.local1081 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1068, !dbg !18
  %address.local1082 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1061, !dbg !18
  store half %1690, ptr addrspace(1) %address.local1082, align 2, !dbg !18
  %address.local1083 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1068, !dbg !18
  %address.local1084 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1083, i64 %address.local1064, !dbg !18
  store half %1689, ptr addrspace(1) %address.local1084, align 2, !dbg !18
  %1722 = insertelement <4 x float> poison, float %1562, i64 0, !dbg !16
  %1723 = insertelement <4 x float> %1722, float %1568, i64 1, !dbg !16
  %1724 = insertelement <4 x float> %1723, float %1563, i64 2, !dbg !16
  %1725 = insertelement <4 x float> %1724, float %1569, i64 3, !dbg !16
  %row.reciprocal47 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1513, !dbg !16
  %normalized48 = fmul <4 x float> %1725, %row.reciprocal47, !dbg !16
  %1726 = fptrunc <4 x float> %normalized48 to <4 x half>, !dbg !17
  %1727 = extractelement <4 x half> %1726, i64 3, !dbg !17
  %1728 = extractelement <4 x half> %1726, i64 2, !dbg !17
  %1729 = extractelement <4 x half> %1726, i64 1, !dbg !17
  %1730 = extractelement <4 x half> %1726, i64 0, !dbg !17
  %1731 = insertelement <4 x float> poison, float %1453, i64 0, !dbg !16
  %1732 = insertelement <4 x float> %1731, float %1459, i64 1, !dbg !16
  %1733 = insertelement <4 x float> %1732, float %1454, i64 2, !dbg !16
  %1734 = insertelement <4 x float> %1733, float %1460, i64 3, !dbg !16
  %normalized49 = fmul <4 x float> %1734, %row.reciprocal47, !dbg !16
  %1735 = fptrunc <4 x float> %normalized49 to <4 x half>, !dbg !17
  %1736 = extractelement <4 x half> %1735, i64 3, !dbg !17
  %1737 = extractelement <4 x half> %1735, i64 2, !dbg !17
  %1738 = extractelement <4 x half> %1735, i64 1, !dbg !17
  %1739 = extractelement <4 x half> %1735, i64 0, !dbg !17
  %1740 = insertelement <4 x float> poison, float %1345, i64 0, !dbg !16
  %1741 = insertelement <4 x float> %1740, float %1351, i64 1, !dbg !16
  %1742 = insertelement <4 x float> %1741, float %1346, i64 2, !dbg !16
  %1743 = insertelement <4 x float> %1742, float %1352, i64 3, !dbg !16
  %row.reciprocal50 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1296, !dbg !16
  %normalized51 = fmul <4 x float> %1743, %row.reciprocal50, !dbg !16
  %1744 = fptrunc <4 x float> %normalized51 to <4 x half>, !dbg !17
  %1745 = extractelement <4 x half> %1744, i64 3, !dbg !17
  %1746 = extractelement <4 x half> %1744, i64 2, !dbg !17
  %1747 = extractelement <4 x half> %1744, i64 1, !dbg !17
  %1748 = extractelement <4 x half> %1744, i64 0, !dbg !17
  %1749 = insertelement <4 x float> poison, float %1236, i64 0, !dbg !16
  %1750 = insertelement <4 x float> %1749, float %1242, i64 1, !dbg !16
  %1751 = insertelement <4 x float> %1750, float %1237, i64 2, !dbg !16
  %1752 = insertelement <4 x float> %1751, float %1243, i64 3, !dbg !16
  %normalized52 = fmul <4 x float> %1752, %row.reciprocal50, !dbg !16
  %1753 = fptrunc <4 x float> %normalized52 to <4 x half>, !dbg !17
  %1754 = extractelement <4 x half> %1753, i64 3, !dbg !17
  %1755 = extractelement <4 x half> %1753, i64 2, !dbg !17
  %1756 = extractelement <4 x half> %1753, i64 1, !dbg !17
  %1757 = extractelement <4 x half> %1753, i64 0, !dbg !17
  %1758 = shl nuw nsw i64 %24, 19, !dbg !18
  %address.thread1085 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !18, !range !32, !frisk.register_lifetimes !1
  %address.local1086 = zext nneg i32 %address.thread1085 to i64, !dbg !8
  %address.local1087 = add i64 %25, %address.local1086, !dbg !18
  %1759 = lshr i64 %address.local1086, 2, !dbg !18
  %address.local1089 = and i64 %1759, 48, !dbg !18
  %address.local1090 = add i64 %address.local1087, %address.local1089, !dbg !18
  %address.local1091 = and i64 %address.local1086, 240, !dbg !18
  %address.local1092 = sub i64 %address.local1090, %address.local1091, !dbg !18
  %address.local1093 = add i64 %address.local1092, %26, !dbg !18
  %address.local1094 = shl nuw nsw i64 %address.local1093, 7, !dbg !18
  %address.local1095 = add nuw nsw i64 %1758, %address.local1094, !dbg !18
  %address.local1096 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1095, !dbg !18
  %address.local1097 = lshr i64 %address.local1086, 2, !dbg !10
  %address.local1098 = and i64 %address.local1097, 12, !dbg !10
  %address.local1099 = or disjoint i64 %address.local1098, 64, !dbg !18
  %address.local1100 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1096, i64 %address.local1099, !dbg !18
  store half %1757, ptr addrspace(1) %address.local1100, align 2, !dbg !18
  %address.local1101 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1095, !dbg !18
  %address.local1102 = or disjoint i64 %address.local1098, 65, !dbg !18
  %address.local1103 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1101, i64 %address.local1102, !dbg !18
  store half %1756, ptr addrspace(1) %address.local1103, align 2, !dbg !18
  %address.local1104 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1095, !dbg !18
  %address.local1105 = or disjoint i64 %address.local1098, 66, !dbg !18
  %address.local1106 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1104, i64 %address.local1105, !dbg !18
  store half %1755, ptr addrspace(1) %address.local1106, align 2, !dbg !18
  %address.local1107 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1095, !dbg !18
  %address.local1108 = or disjoint i64 %address.local1098, 67, !dbg !18
  %address.local1109 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1107, i64 %address.local1108, !dbg !18
  store half %1754, ptr addrspace(1) %address.local1109, align 2, !dbg !18
  %address.local1110 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1095, !dbg !18
  %address.local1111 = or disjoint i64 %address.local1098, 80, !dbg !18
  %address.local1112 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1110, i64 %address.local1111, !dbg !18
  store half %1748, ptr addrspace(1) %address.local1112, align 2, !dbg !18
  %address.local1113 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1095, !dbg !18
  %address.local1114 = or disjoint i64 %address.local1098, 81, !dbg !18
  %address.local1115 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1113, i64 %address.local1114, !dbg !18
  store half %1747, ptr addrspace(1) %address.local1115, align 2, !dbg !18
  %address.local1116 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1095, !dbg !18
  %address.local1117 = or disjoint i64 %address.local1098, 82, !dbg !18
  %address.local1118 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1116, i64 %address.local1117, !dbg !18
  store half %1746, ptr addrspace(1) %address.local1118, align 2, !dbg !18
  %address.local1119 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1095, !dbg !18
  %address.local1120 = or disjoint i64 %address.local1098, 83, !dbg !18
  %address.local1121 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1119, i64 %address.local1120, !dbg !18
  store half %1745, ptr addrspace(1) %address.local1121, align 2, !dbg !18
  %address.local1122 = shl i64 %address.local1093, 7, !dbg !18
  %address.local1123 = add i64 %address.local1122, 8192, !dbg !18
  %address.local1124 = add nuw nsw i64 %1758, %address.local1123, !dbg !18
  %address.local1125 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1124, !dbg !18
  %address.local1126 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1125, i64 %address.local1099, !dbg !18
  store half %1739, ptr addrspace(1) %address.local1126, align 2, !dbg !18
  %address.local1127 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1124, !dbg !18
  %address.local1128 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1127, i64 %address.local1102, !dbg !18
  store half %1738, ptr addrspace(1) %address.local1128, align 2, !dbg !18
  %address.local1129 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1124, !dbg !18
  %address.local1130 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1129, i64 %address.local1105, !dbg !18
  store half %1737, ptr addrspace(1) %address.local1130, align 2, !dbg !18
  %address.local1131 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1124, !dbg !18
  %address.local1132 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1131, i64 %address.local1108, !dbg !18
  store half %1736, ptr addrspace(1) %address.local1132, align 2, !dbg !18
  %address.local1133 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1124, !dbg !18
  %address.local1134 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1133, i64 %address.local1111, !dbg !18
  store half %1730, ptr addrspace(1) %address.local1134, align 2, !dbg !18
  %address.local1135 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1124, !dbg !18
  %address.local1136 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1135, i64 %address.local1114, !dbg !18
  store half %1729, ptr addrspace(1) %address.local1136, align 2, !dbg !18
  %address.local1137 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1124, !dbg !18
  %address.local1138 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1137, i64 %address.local1117, !dbg !18
  store half %1728, ptr addrspace(1) %address.local1138, align 2, !dbg !18
  %address.local1139 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1124, !dbg !18
  %address.local1140 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1139, i64 %address.local1120, !dbg !18
  store half %1727, ptr addrspace(1) %address.local1140, align 2, !dbg !18
  %1760 = insertelement <4 x float> poison, float %1534, i64 0, !dbg !16
  %1761 = insertelement <4 x float> %1760, float %1540, i64 1, !dbg !16
  %1762 = insertelement <4 x float> %1761, float %1535, i64 2, !dbg !16
  %1763 = insertelement <4 x float> %1762, float %1541, i64 3, !dbg !16
  %row.reciprocal53 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1513, !dbg !16
  %normalized54 = fmul <4 x float> %1763, %row.reciprocal53, !dbg !16
  %1764 = fptrunc <4 x float> %normalized54 to <4 x half>, !dbg !17
  %1765 = extractelement <4 x half> %1764, i64 3, !dbg !17
  %1766 = extractelement <4 x half> %1764, i64 2, !dbg !17
  %1767 = extractelement <4 x half> %1764, i64 1, !dbg !17
  %1768 = extractelement <4 x half> %1764, i64 0, !dbg !17
  %1769 = insertelement <4 x float> poison, float %1424, i64 0, !dbg !16
  %1770 = insertelement <4 x float> %1769, float %1430, i64 1, !dbg !16
  %1771 = insertelement <4 x float> %1770, float %1425, i64 2, !dbg !16
  %1772 = insertelement <4 x float> %1771, float %1431, i64 3, !dbg !16
  %normalized55 = fmul <4 x float> %1772, %row.reciprocal53, !dbg !16
  %1773 = fptrunc <4 x float> %normalized55 to <4 x half>, !dbg !17
  %1774 = extractelement <4 x half> %1773, i64 3, !dbg !17
  %1775 = extractelement <4 x half> %1773, i64 2, !dbg !17
  %1776 = extractelement <4 x half> %1773, i64 1, !dbg !17
  %1777 = extractelement <4 x half> %1773, i64 0, !dbg !17
  %1778 = insertelement <4 x float> poison, float %1317, i64 0, !dbg !16
  %1779 = insertelement <4 x float> %1778, float %1323, i64 1, !dbg !16
  %1780 = insertelement <4 x float> %1779, float %1318, i64 2, !dbg !16
  %1781 = insertelement <4 x float> %1780, float %1324, i64 3, !dbg !16
  %row.reciprocal56 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1296, !dbg !16
  %normalized57 = fmul <4 x float> %1781, %row.reciprocal56, !dbg !16
  %1782 = fptrunc <4 x float> %normalized57 to <4 x half>, !dbg !17
  %1783 = extractelement <4 x half> %1782, i64 3, !dbg !17
  %1784 = extractelement <4 x half> %1782, i64 2, !dbg !17
  %1785 = extractelement <4 x half> %1782, i64 1, !dbg !17
  %1786 = extractelement <4 x half> %1782, i64 0, !dbg !17
  %1787 = insertelement <4 x float> poison, float %1207, i64 0, !dbg !16
  %1788 = insertelement <4 x float> %1787, float %1213, i64 1, !dbg !16
  %1789 = insertelement <4 x float> %1788, float %1208, i64 2, !dbg !16
  %1790 = insertelement <4 x float> %1789, float %1214, i64 3, !dbg !16
  %normalized58 = fmul <4 x float> %1790, %row.reciprocal56, !dbg !16
  %1791 = fptrunc <4 x float> %normalized58 to <4 x half>, !dbg !17
  %1792 = extractelement <4 x half> %1791, i64 3, !dbg !17
  %1793 = extractelement <4 x half> %1791, i64 2, !dbg !17
  %1794 = extractelement <4 x half> %1791, i64 1, !dbg !17
  %1795 = extractelement <4 x half> %1791, i64 0, !dbg !17
  %1796 = shl nuw nsw i64 %24, 19, !dbg !18
  %address.thread1141 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !18, !range !32, !frisk.register_lifetimes !1
  %address.local1142 = zext nneg i32 %address.thread1141 to i64, !dbg !8
  %address.local1143 = add i64 %25, %address.local1142, !dbg !18
  %1797 = lshr i64 %address.local1142, 2, !dbg !18
  %address.local1145 = and i64 %1797, 48, !dbg !18
  %address.local1146 = add i64 %address.local1143, %address.local1145, !dbg !18
  %address.local1147 = and i64 %address.local1142, 240, !dbg !18
  %address.local1148 = sub i64 %address.local1146, %address.local1147, !dbg !18
  %address.local1149 = add i64 %address.local1148, %26, !dbg !18
  %address.local1150 = shl nuw nsw i64 %address.local1149, 7, !dbg !18
  %address.local1151 = add nuw nsw i64 %1796, %address.local1150, !dbg !18
  %address.local1152 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1151, !dbg !18
  %address.local1153 = lshr i64 %address.local1142, 2, !dbg !10
  %address.local1154 = and i64 %address.local1153, 12, !dbg !10
  %address.local1155 = or disjoint i64 %address.local1154, 96, !dbg !18
  %address.local1156 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1152, i64 %address.local1155, !dbg !18
  store half %1795, ptr addrspace(1) %address.local1156, align 2, !dbg !18
  %address.local1157 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1151, !dbg !18
  %address.local1158 = or disjoint i64 %address.local1154, 97, !dbg !18
  %address.local1159 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1157, i64 %address.local1158, !dbg !18
  store half %1794, ptr addrspace(1) %address.local1159, align 2, !dbg !18
  %address.local1160 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1151, !dbg !18
  %address.local1161 = or disjoint i64 %address.local1154, 98, !dbg !18
  %address.local1162 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1160, i64 %address.local1161, !dbg !18
  store half %1793, ptr addrspace(1) %address.local1162, align 2, !dbg !18
  %address.local1163 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1151, !dbg !18
  %address.local1164 = or disjoint i64 %address.local1154, 99, !dbg !18
  %address.local1165 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1163, i64 %address.local1164, !dbg !18
  store half %1792, ptr addrspace(1) %address.local1165, align 2, !dbg !18
  %address.local1166 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1151, !dbg !18
  %address.local1167 = or disjoint i64 %address.local1154, 112, !dbg !18
  %address.local1168 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1166, i64 %address.local1167, !dbg !18
  store half %1786, ptr addrspace(1) %address.local1168, align 2, !dbg !18
  %address.local1169 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1151, !dbg !18
  %address.local1170 = or disjoint i64 %address.local1154, 113, !dbg !18
  %address.local1171 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1169, i64 %address.local1170, !dbg !18
  store half %1785, ptr addrspace(1) %address.local1171, align 2, !dbg !18
  %address.local1172 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1151, !dbg !18
  %address.local1173 = or disjoint i64 %address.local1154, 114, !dbg !18
  %address.local1174 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1172, i64 %address.local1173, !dbg !18
  store half %1784, ptr addrspace(1) %address.local1174, align 2, !dbg !18
  %address.local1175 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1151, !dbg !18
  %address.local1176 = or i64 %address.local1153, 115, !dbg !18
  %address.local1177 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1175, i64 %address.local1176, !dbg !18
  store half %1783, ptr addrspace(1) %address.local1177, align 2, !dbg !18
  %address.local1178 = shl i64 %address.local1149, 7, !dbg !18
  %address.local1179 = add i64 %address.local1178, 8192, !dbg !18
  %address.local1180 = add nuw nsw i64 %1796, %address.local1179, !dbg !18
  %address.local1181 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1180, !dbg !18
  %address.local1182 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1181, i64 %address.local1155, !dbg !18
  store half %1777, ptr addrspace(1) %address.local1182, align 2, !dbg !18
  %address.local1183 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1180, !dbg !18
  %address.local1184 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1183, i64 %address.local1158, !dbg !18
  store half %1776, ptr addrspace(1) %address.local1184, align 2, !dbg !18
  %address.local1185 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1180, !dbg !18
  %address.local1186 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1185, i64 %address.local1161, !dbg !18
  store half %1775, ptr addrspace(1) %address.local1186, align 2, !dbg !18
  %address.local1187 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1180, !dbg !18
  %address.local1188 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1187, i64 %address.local1164, !dbg !18
  store half %1774, ptr addrspace(1) %address.local1188, align 2, !dbg !18
  %address.local1189 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1180, !dbg !18
  %address.local1190 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1189, i64 %address.local1167, !dbg !18
  store half %1768, ptr addrspace(1) %address.local1190, align 2, !dbg !18
  %address.local1191 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1180, !dbg !18
  %address.local1192 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1191, i64 %address.local1170, !dbg !18
  store half %1767, ptr addrspace(1) %address.local1192, align 2, !dbg !18
  %address.local1193 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1180, !dbg !18
  %address.local1194 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1193, i64 %address.local1173, !dbg !18
  store half %1766, ptr addrspace(1) %address.local1194, align 2, !dbg !18
  %address.local1195 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1180, !dbg !18
  %address.local1196 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1195, i64 %address.local1176, !dbg !18
  store half %1765, ptr addrspace(1) %address.local1196, align 2, !dbg !18
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

attributes #0 = { convergent willreturn memory(read) }
attributes #1 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1" "frisk.global_prefetch"="8" "frisk.register_comparisons"="1" "frisk.register_lifetimes"="1" "uniform-work-group-size" }
attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none) }
attributes #4 = { nocallback nofree nounwind willreturn }
attributes #5 = { convergent nocallback nofree nounwind willreturn }
attributes #6 = { nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }
attributes #7 = { convergent nocallback nofree nounwind willreturn memory(none) }
attributes #8 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #9 = { convergent nounwind willreturn memory(none) }
attributes #10 = { convergent nounwind }

!llvm.dbg.cu = !{!3}
!llvm.module.flags = !{!31}

!0 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/noncausal_optimization/coalesce/run-20261027-164012-mybq6yab/online-baseline")
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
