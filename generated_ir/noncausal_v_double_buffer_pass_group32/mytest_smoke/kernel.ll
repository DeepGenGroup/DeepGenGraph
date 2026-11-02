; ModuleID = '/data2/xsl/DeepGenGraph/3rd/deepgengraph/test/test_input_spidD_noncasual.mlir'
source_filename = "/data2/xsl/DeepGenGraph/3rd/deepgengraph/test/test_input_spidD_noncasual.mlir"

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
  %12 = and i64 %11, 576460752303423456, !dbg !5
  %13 = and i64 %6, 31, !dbg !5
  %14 = or disjoint i64 %12, %13, !dbg !5
  %15 = lshr i64 %10, 5, !dbg !5
  %16 = lshr i64 %10, 10, !dbg !8
  %17 = add nuw nsw i64 %15, %16, !dbg !8
  %18 = and i64 %17, 31, !dbg !8
  %19 = shl nsw i64 %14, 7, !dbg !8
  %20 = mul nsw i64 %16, -4096, !dbg !8
  %21 = add i64 %19, %20, !dbg !8
  %22 = shl nuw nsw i64 %18, 19, !dbg !8
  %23 = shl nsw i64 %21, 7, !dbg !8
  %24 = add i64 %22, %23, !dbg !8
  %25 = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !8, !range !32
  %26 = zext nneg i32 %25 to i64, !dbg !8
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %25) #9, !dbg !8, !range !32, !frisk.register_lifetimes !1
  %address.local = zext nneg i32 %address.thread to i64, !dbg !8
  %address.local404 = shl nuw nsw i64 %address.local, 3, !dbg !8
  %address.local405 = and i64 %address.local404, 1536, !dbg !8
  %address.local406 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local405, !dbg !8
  %address.local407 = ptrtoint ptr addrspace(3) %address.local406 to i64, !dbg !8
  %address.local408 = trunc i64 %address.local407 to i32, !dbg !8
  %27 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local408), !dbg !8
  %address.local409 = shl nuw nsw i64 %address.local, 7, !dbg !8
  %address.local410 = and i64 %address.local409, 16256, !dbg !8
  %address.local411 = lshr i64 %address.local, 4, !dbg !8
  %address.local412 = and i64 %address.local411, 8, !dbg !8
  %address.local413 = or disjoint i64 %address.local410, %address.local412, !dbg !8
  %28 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %.idx1399 = or disjoint i64 %24, %address.local413, !dbg !8
  %.idx1399.tr = trunc i64 %.idx1399 to i32, !dbg !8
  %dma.buffer.offset1396 = shl i32 %.idx1399.tr, 1, !dbg !8
  %29 = insertelement <2 x i64> poison, i64 %28, i64 0, !dbg !8
  %30 = bitcast <2 x i64> %29 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc1397 = shufflevector <4 x i32> %30, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
  %dma.buffer.lds1398 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %27), !dbg !8
  %31 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1398, i32 %dma.buffer.offset1396, <4 x i32> %dma.buffer.rsrc1397) #10, !dbg !8
  %address.local415 = or disjoint i64 %address.local, 256, !dbg !8
  %address.local416 = shl nuw nsw i64 %address.local415, 3, !dbg !8
  %address.local417 = and i64 %address.local416, 3584, !dbg !8
  %address.local418 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local417, !dbg !8
  %address.local419 = ptrtoint ptr addrspace(3) %address.local418 to i64, !dbg !8
  %address.local420 = trunc i64 %address.local419 to i32, !dbg !8
  %32 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local420), !dbg !8
  %address.local421 = shl nuw nsw i64 %address.local, 7, !dbg !8
  %address.local422 = and i64 %address.local421, 16256, !dbg !8
  %address.local423 = lshr i64 %address.local415, 4, !dbg !8
  %address.local424 = and i64 %address.local423, 24, !dbg !8
  %address.local425 = or disjoint i64 %address.local422, %address.local424, !dbg !8
  %33 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %.idx1400 = or disjoint i64 %24, %address.local425, !dbg !8
  %.idx1400.tr = trunc i64 %.idx1400 to i32, !dbg !8
  %dma.buffer.offset1393 = shl i32 %.idx1400.tr, 1, !dbg !8
  %34 = insertelement <2 x i64> poison, i64 %33, i64 0, !dbg !8
  %35 = bitcast <2 x i64> %34 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc1394 = shufflevector <4 x i32> %35, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
  %dma.buffer.lds1395 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %32), !dbg !8
  %36 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1395, i32 %dma.buffer.offset1393, <4 x i32> %dma.buffer.rsrc1394) #10, !dbg !8
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %address.local427 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local405, !dbg !8
  %address.local428 = ptrtoint ptr addrspace(3) %address.local427 to i64, !dbg !8
  %address.local429 = trunc i64 %address.local428 to i32, !dbg !8
  %37 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local429), !dbg !8
  %38 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %39 = or disjoint i64 %24, 32, !dbg !8
  %40 = or disjoint i64 %address.local413, %39, !dbg !8
  %.tr = trunc i64 %40 to i32, !dbg !8
  %dma.buffer.offset1390 = shl i32 %.tr, 1, !dbg !8
  %41 = insertelement <2 x i64> poison, i64 %38, i64 0, !dbg !8
  %42 = bitcast <2 x i64> %41 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc1391 = shufflevector <4 x i32> %42, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
  %dma.buffer.lds1392 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %37), !dbg !8
  %43 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1392, i32 %dma.buffer.offset1390, <4 x i32> %dma.buffer.rsrc1391) #10, !dbg !8
  %address.local431 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local417, !dbg !8
  %address.local432 = ptrtoint ptr addrspace(3) %address.local431 to i64, !dbg !8
  %address.local433 = trunc i64 %address.local432 to i32, !dbg !8
  %44 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local433), !dbg !8
  %45 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %46 = or disjoint i64 %24, 32, !dbg !8
  %47 = or disjoint i64 %address.local425, %46, !dbg !8
  %.tr1401 = trunc i64 %47 to i32, !dbg !8
  %dma.buffer.offset1387 = shl i32 %.tr1401, 1, !dbg !8
  %48 = insertelement <2 x i64> poison, i64 %45, i64 0, !dbg !8
  %49 = bitcast <2 x i64> %48 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc1388 = shufflevector <4 x i32> %49, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
  %dma.buffer.lds1389 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %44), !dbg !8
  %50 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1389, i32 %dma.buffer.offset1387, <4 x i32> %dma.buffer.rsrc1388) #10, !dbg !8
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
  %51 = load <4 x half>, ptr addrspace(3) %address.local447, align 8, !dbg !10
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local447, i64 4096, !dbg !10
  %52 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !10
  %53 = fmul <4 x half> %51, splat (half 0xH3015), !dbg !10
  %lds.const.gep1197 = getelementptr i8, ptr addrspace(3) %address.local447, i64 1024, !dbg !10
  %54 = load <4 x half>, ptr addrspace(3) %lds.const.gep1197, align 8, !dbg !10
  %55 = fmul <4 x half> %52, splat (half 0xH3015), !dbg !10
  %lds.const.gep1198 = getelementptr i8, ptr addrspace(3) %address.local447, i64 5120, !dbg !10
  %56 = load <4 x half>, ptr addrspace(3) %lds.const.gep1198, align 8, !dbg !10
  %57 = fmul <4 x half> %54, splat (half 0xH3015), !dbg !10
  %58 = fmul <4 x half> %56, splat (half 0xH3015), !dbg !10
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %59 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local408), !dbg !8
  %60 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %61 = or disjoint i64 %24, 64, !dbg !8
  %62 = or disjoint i64 %address.local413, %61, !dbg !8
  %.tr1402 = trunc i64 %62 to i32, !dbg !8
  %dma.buffer.offset1384 = shl i32 %.tr1402, 1, !dbg !8
  %63 = insertelement <2 x i64> poison, i64 %60, i64 0, !dbg !8
  %64 = bitcast <2 x i64> %63 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc1385 = shufflevector <4 x i32> %64, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
  %dma.buffer.lds1386 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %59), !dbg !8
  %65 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1386, i32 %dma.buffer.offset1384, <4 x i32> %dma.buffer.rsrc1385) #10, !dbg !8
  %66 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local420), !dbg !8
  %67 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %68 = or disjoint i64 %24, 64, !dbg !8
  %69 = or disjoint i64 %address.local425, %68, !dbg !8
  %.tr1403 = trunc i64 %69 to i32, !dbg !8
  %dma.buffer.offset1381 = shl i32 %.tr1403, 1, !dbg !8
  %70 = insertelement <2 x i64> poison, i64 %67, i64 0, !dbg !8
  %71 = bitcast <2 x i64> %70 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc1382 = shufflevector <4 x i32> %71, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
  %dma.buffer.lds1383 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %66), !dbg !8
  %72 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1383, i32 %dma.buffer.offset1381, <4 x i32> %dma.buffer.rsrc1382) #10, !dbg !8
  %lds.const.gep1199 = getelementptr i8, ptr addrspace(3) %address.local447, i64 8192, !dbg !10
  %73 = load <4 x half>, ptr addrspace(3) %lds.const.gep1199, align 8, !dbg !10
  %lds.const.gep1200 = getelementptr i8, ptr addrspace(3) %address.local447, i64 12288, !dbg !10
  %74 = load <4 x half>, ptr addrspace(3) %lds.const.gep1200, align 8, !dbg !10
  %lds.const.gep1201 = getelementptr i8, ptr addrspace(3) %address.local447, i64 9216, !dbg !10
  %75 = load <4 x half>, ptr addrspace(3) %lds.const.gep1201, align 8, !dbg !10
  %76 = fmul <4 x half> %73, splat (half 0xH3015), !dbg !10
  %lds.const.gep1202 = getelementptr i8, ptr addrspace(3) %address.local447, i64 13312, !dbg !10
  %77 = load <4 x half>, ptr addrspace(3) %lds.const.gep1202, align 8, !dbg !10
  %78 = fmul <4 x half> %74, splat (half 0xH3015), !dbg !10
  %79 = fmul <4 x half> %75, splat (half 0xH3015), !dbg !10
  %80 = fmul <4 x half> %77, splat (half 0xH3015), !dbg !10
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %81 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local429), !dbg !8
  %82 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %83 = or disjoint i64 %24, 96, !dbg !8
  %84 = or disjoint i64 %address.local413, %83, !dbg !8
  %.tr1404 = trunc i64 %84 to i32, !dbg !8
  %dma.buffer.offset1378 = shl i32 %.tr1404, 1, !dbg !8
  %85 = insertelement <2 x i64> poison, i64 %82, i64 0, !dbg !8
  %86 = bitcast <2 x i64> %85 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc1379 = shufflevector <4 x i32> %86, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
  %dma.buffer.lds1380 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %81), !dbg !8
  %87 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1380, i32 %dma.buffer.offset1378, <4 x i32> %dma.buffer.rsrc1379) #10, !dbg !8
  %88 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local433), !dbg !8
  %89 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %90 = or disjoint i64 %24, 96, !dbg !8
  %91 = or disjoint i64 %address.local425, %90, !dbg !8
  %.tr1405 = trunc i64 %91 to i32, !dbg !8
  %dma.buffer.offset1375 = shl i32 %.tr1405, 1, !dbg !8
  %92 = insertelement <2 x i64> poison, i64 %89, i64 0, !dbg !8
  %93 = bitcast <2 x i64> %92 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc1376 = shufflevector <4 x i32> %93, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
  %dma.buffer.lds1377 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %88), !dbg !8
  %94 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1377, i32 %dma.buffer.offset1375, <4 x i32> %dma.buffer.rsrc1376) #10, !dbg !8
  %95 = load <4 x half>, ptr addrspace(3) %address.local447, align 8, !dbg !10
  %lds.const.gep1203 = getelementptr i8, ptr addrspace(3) %address.local447, i64 4096, !dbg !10
  %96 = load <4 x half>, ptr addrspace(3) %lds.const.gep1203, align 8, !dbg !10
  %lds.const.gep1204 = getelementptr i8, ptr addrspace(3) %address.local447, i64 1024, !dbg !10
  %97 = load <4 x half>, ptr addrspace(3) %lds.const.gep1204, align 8, !dbg !10
  %98 = fmul <4 x half> %95, splat (half 0xH3015), !dbg !10
  %lds.const.gep1205 = getelementptr i8, ptr addrspace(3) %address.local447, i64 5120, !dbg !10
  %99 = load <4 x half>, ptr addrspace(3) %lds.const.gep1205, align 8, !dbg !10
  %100 = fmul <4 x half> %96, splat (half 0xH3015), !dbg !10
  %101 = fmul <4 x half> %97, splat (half 0xH3015), !dbg !10
  %102 = fmul <4 x half> %99, splat (half 0xH3015), !dbg !10
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %lds.const.gep1206 = getelementptr i8, ptr addrspace(3) %address.local447, i64 8192, !dbg !10
  %103 = load <4 x half>, ptr addrspace(3) %lds.const.gep1206, align 8, !dbg !10
  %lds.const.gep1207 = getelementptr i8, ptr addrspace(3) %address.local447, i64 12288, !dbg !10
  %104 = load <4 x half>, ptr addrspace(3) %lds.const.gep1207, align 8, !dbg !10
  %lds.const.gep1208 = getelementptr i8, ptr addrspace(3) %address.local447, i64 9216, !dbg !10
  %105 = load <4 x half>, ptr addrspace(3) %lds.const.gep1208, align 8, !dbg !10
  %106 = fmul <4 x half> %103, splat (half 0xH3015), !dbg !10
  %lds.const.gep1209 = getelementptr i8, ptr addrspace(3) %address.local447, i64 13312, !dbg !10
  %107 = load <4 x half>, ptr addrspace(3) %lds.const.gep1209, align 8, !dbg !10
  %108 = fmul <4 x half> %104, splat (half 0xH3015), !dbg !10
  %109 = fmul <4 x half> %105, splat (half 0xH3015), !dbg !10
  %110 = fmul <4 x half> %107, splat (half 0xH3015), !dbg !10
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %111 = shl nuw nsw i64 %26, 2, !dbg !14
  %112 = shl nuw nsw i64 %26, 2, !dbg !14
  %113 = and i64 %112, 1008, !dbg !14
  %114 = sub nsw i64 %111, %113, !dbg !14
  %115 = and i32 %25, 1, !dbg !14
  %index.compact344 = zext nneg i32 %115 to i64, !dbg !14
  %116 = or disjoint i64 %114, %index.compact344, !dbg !14
  %117 = add nuw nsw i64 %116, 1, !dbg !14
  %118 = sdiv i64 %117, 2, !dbg !14
  br label %119, !dbg !7

119:                                              ; preds = %128, %4
  %120 = phi i64 [ %1146, %128 ], [ 0, %4 ], !dbg !7
  %121 = phi [2 x <8 x float>] [ %1131, %128 ], [ zeroinitializer, %4 ], !dbg !7
  %122 = phi [2 x <1 x float>] [ %801, %128 ], [ zeroinitializer, %4 ], !dbg !7
  %123 = phi [2 x <1 x float>] [ %1133, %128 ], [ [<1 x float> splat (float 0xFFF0000000000000), <1 x float> splat (float 0xFFF0000000000000)], %4 ], !dbg !7
  %124 = phi [2 x <8 x float>] [ %1137, %128 ], [ zeroinitializer, %4 ], !dbg !7
  %125 = phi [2 x <8 x float>] [ %1141, %128 ], [ zeroinitializer, %4 ], !dbg !7
  %126 = phi [2 x <8 x float>] [ %1145, %128 ], [ zeroinitializer, %4 ], !dbg !7
  %127 = icmp samesign ult i64 %120, 32, !dbg !7
  br i1 %127, label %128, label %1147, !dbg !7

128:                                              ; preds = %119
  %129 = shl i64 %15, 19, !dbg !12
  %130 = and i64 %129, 16252928, !dbg !12
  %131 = shl nuw nsw i64 %120, 14, !dbg !12
  %132 = or disjoint i64 %130, %131, !dbg !12
  %address.thread498 = call i32 asm sideeffect "", "=v,0"(i32 %25) #9, !dbg !12, !range !32, !frisk.register_lifetimes !1
  %address.local499 = zext nneg i32 %address.thread498 to i64, !dbg !8
  %address.local500 = shl nuw nsw i64 %address.local499, 3, !dbg !8
  %address.local501 = and i64 %address.local500, 1536, !dbg !8
  %address.local502 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local501, !dbg !8
  %address.local503 = ptrtoint ptr addrspace(3) %address.local502 to i64, !dbg !8
  %address.local504 = trunc i64 %address.local503 to i32, !dbg !8
  %133 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local504), !dbg !12
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
  %134 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !12
  %.idx1406 = or disjoint i64 %132, %address.local514, !dbg !12
  %.idx1406.tr = trunc nuw i64 %.idx1406 to i32, !dbg !12
  %dma.buffer.offset1372 = shl nuw nsw i32 %.idx1406.tr, 1, !dbg !12
  %135 = insertelement <2 x i64> poison, i64 %134, i64 0, !dbg !12
  %136 = bitcast <2 x i64> %135 to <4 x i32>, !dbg !12
  %dma.buffer.rsrc1373 = shufflevector <4 x i32> %136, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !12
  %dma.buffer.lds1374 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %133), !dbg !12
  %137 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1374, i32 %dma.buffer.offset1372, <4 x i32> %dma.buffer.rsrc1373) #10, !dbg !12
  %address.local516 = or disjoint i64 %address.local499, 256, !dbg !8
  %address.local517 = shl nuw nsw i64 %address.local516, 3, !dbg !8
  %address.local518 = and i64 %address.local517, 3584, !dbg !8
  %address.local519 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local518, !dbg !8
  %address.local520 = ptrtoint ptr addrspace(3) %address.local519 to i64, !dbg !8
  %address.local521 = trunc i64 %address.local520 to i32, !dbg !8
  %138 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local521), !dbg !12
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
  %139 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !12
  %.idx1407 = or disjoint i64 %132, %address.local531, !dbg !12
  %.idx1407.tr = trunc nuw i64 %.idx1407 to i32, !dbg !12
  %dma.buffer.offset1369 = shl nuw nsw i32 %.idx1407.tr, 1, !dbg !12
  %140 = insertelement <2 x i64> poison, i64 %139, i64 0, !dbg !12
  %141 = bitcast <2 x i64> %140 to <4 x i32>, !dbg !12
  %dma.buffer.rsrc1370 = shufflevector <4 x i32> %141, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !12
  %dma.buffer.lds1371 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %138), !dbg !12
  %142 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1371, i32 %dma.buffer.offset1369, <4 x i32> %dma.buffer.rsrc1370) #10, !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %143 = or disjoint i64 %132, 32, !dbg !12
  %address.local533 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local501, !dbg !8
  %address.local534 = ptrtoint ptr addrspace(3) %address.local533 to i64, !dbg !8
  %address.local535 = trunc i64 %address.local534 to i32, !dbg !8
  %144 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local535), !dbg !12
  %145 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !12
  %.idx1408 = or disjoint i64 %143, %address.local514, !dbg !12
  %.idx1408.tr = trunc nuw i64 %.idx1408 to i32, !dbg !12
  %dma.buffer.offset1366 = shl nuw nsw i32 %.idx1408.tr, 1, !dbg !12
  %146 = insertelement <2 x i64> poison, i64 %145, i64 0, !dbg !12
  %147 = bitcast <2 x i64> %146 to <4 x i32>, !dbg !12
  %dma.buffer.rsrc1367 = shufflevector <4 x i32> %147, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !12
  %dma.buffer.lds1368 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %144), !dbg !12
  %148 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1368, i32 %dma.buffer.offset1366, <4 x i32> %dma.buffer.rsrc1367) #10, !dbg !12
  %address.local537 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local518, !dbg !8
  %address.local538 = ptrtoint ptr addrspace(3) %address.local537 to i64, !dbg !8
  %address.local539 = trunc i64 %address.local538 to i32, !dbg !8
  %149 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local539), !dbg !12
  %150 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !12
  %.idx1409 = or disjoint i64 %143, %address.local531, !dbg !12
  %.idx1409.tr = trunc nuw i64 %.idx1409 to i32, !dbg !12
  %dma.buffer.offset1363 = shl nuw nsw i32 %.idx1409.tr, 1, !dbg !12
  %151 = insertelement <2 x i64> poison, i64 %150, i64 0, !dbg !12
  %152 = bitcast <2 x i64> %151 to <4 x i32>, !dbg !12
  %dma.buffer.rsrc1364 = shufflevector <4 x i32> %152, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !12
  %dma.buffer.lds1365 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %149), !dbg !12
  %153 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1365, i32 %dma.buffer.offset1363, <4 x i32> %dma.buffer.rsrc1364) #10, !dbg !12
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
  %154 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %.idx, !dbg !14
  %.idx1410 = mul nsw i64 %address.local542, -1280, !dbg !14
  %155 = getelementptr i8, ptr addrspace(3) %154, i64 %.idx1410, !dbg !14
  %156 = getelementptr [2 x i8], ptr addrspace(3) %155, i64 %address.local546, !dbg !14
  %157 = getelementptr [2 x i8], ptr addrspace(3) %156, i64 %address.local550, !dbg !14
  %158 = getelementptr [2 x i8], ptr addrspace(3) %157, i64 %address.local553, !dbg !14
  %address.local559 = getelementptr [2 x i8], ptr addrspace(3) %158, i64 %address.local557, !dbg !14
  %159 = load <4 x half>, ptr addrspace(3) %address.local559, align 8, !dbg !14
  %160 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %53, <4 x half> %159, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1210 = getelementptr i8, ptr addrspace(3) %address.local559, i64 4096, !dbg !14
  %161 = load <4 x half>, ptr addrspace(3) %lds.const.gep1210, align 8, !dbg !14
  %address.local570 = shl nuw nsw i32 %address.thread498, 1, !dbg !14
  %address.local571 = and i32 %address.local570, 24, !dbg !14
  %address.local572 = shl nuw nsw i32 %address.thread498, 8, !dbg !14
  %address.local573 = and i32 %address.local572, 768, !dbg !14
  %address.local574 = or disjoint i32 %address.local571, %address.local573, !dbg !14
  %address.local575 = shl nuw nsw i32 %address.thread498, 5, !dbg !14
  %address.local576 = and i32 %address.local575, 96, !dbg !14
  %address.local577 = or disjoint i32 %address.local574, %address.local576, !dbg !14
  %address.local578 = zext nneg i32 %address.local577 to i64, !dbg !14
  %address.base = call i64 asm sideeffect "", "=v,0"(i64 %118) #9, !dbg !14, !frisk.register_lifetimes !1
  %address.local579 = lshr i32 %address.thread498, 1, !dbg !14
  %address.local580 = and i32 %address.local579, 1, !dbg !14
  %address.local581 = zext nneg i32 %address.local580 to i64, !dbg !14
  %address.local582 = add nsw i64 %address.base, %address.local581, !dbg !14
  %address.local583 = shl i64 %address.local582, 6, !dbg !14
  %address.local584 = sub i64 %address.local578, %address.local583, !dbg !14
  %162 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local584, !dbg !14
  %163 = getelementptr [2 x i8], ptr addrspace(3) %162, i64 %address.local553, !dbg !14
  %164 = getelementptr [2 x i8], ptr addrspace(3) %163, i64 %address.local557, !dbg !14
  %address.local588 = getelementptr i8, ptr addrspace(3) %164, i64 64, !dbg !14
  %165 = load <4 x half>, ptr addrspace(3) %address.local588, align 8, !dbg !14
  %166 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %55, <4 x half> %161, <4 x float> %160) #9, !dbg !14
  %167 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %53, <4 x half> %165, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1211 = getelementptr i8, ptr addrspace(3) %164, i64 4160, !dbg !14
  %168 = load <4 x half>, ptr addrspace(3) %lds.const.gep1211, align 8, !dbg !14
  %lds.const.gep1212 = getelementptr i8, ptr addrspace(3) %address.local559, i64 128, !dbg !14
  %169 = load <4 x half>, ptr addrspace(3) %lds.const.gep1212, align 8, !dbg !14
  %170 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %55, <4 x half> %168, <4 x float> %167) #9, !dbg !14
  %171 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %53, <4 x half> %169, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1213 = getelementptr i8, ptr addrspace(3) %address.local559, i64 4224, !dbg !14
  %172 = load <4 x half>, ptr addrspace(3) %lds.const.gep1213, align 8, !dbg !14
  %lds.const.gep1214 = getelementptr i8, ptr addrspace(3) %164, i64 192, !dbg !14
  %173 = load <4 x half>, ptr addrspace(3) %lds.const.gep1214, align 8, !dbg !14
  %174 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %55, <4 x half> %172, <4 x float> %171) #9, !dbg !14
  %175 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %53, <4 x half> %173, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1215 = getelementptr i8, ptr addrspace(3) %164, i64 4288, !dbg !14
  %176 = load <4 x half>, ptr addrspace(3) %lds.const.gep1215, align 8, !dbg !14
  %lds.const.gep1216 = getelementptr i8, ptr addrspace(3) %address.local559, i64 1024, !dbg !14
  %177 = load <4 x half>, ptr addrspace(3) %lds.const.gep1216, align 8, !dbg !14
  %178 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %55, <4 x half> %176, <4 x float> %175) #9, !dbg !14
  %179 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %53, <4 x half> %177, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1217 = getelementptr i8, ptr addrspace(3) %address.local559, i64 5120, !dbg !14
  %180 = load <4 x half>, ptr addrspace(3) %lds.const.gep1217, align 8, !dbg !14
  %lds.const.gep1218 = getelementptr i8, ptr addrspace(3) %164, i64 1088, !dbg !14
  %181 = load <4 x half>, ptr addrspace(3) %lds.const.gep1218, align 8, !dbg !14
  %182 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %55, <4 x half> %180, <4 x float> %179) #9, !dbg !14
  %183 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %53, <4 x half> %181, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1219 = getelementptr i8, ptr addrspace(3) %164, i64 5184, !dbg !14
  %184 = load <4 x half>, ptr addrspace(3) %lds.const.gep1219, align 8, !dbg !14
  %lds.const.gep1220 = getelementptr i8, ptr addrspace(3) %address.local559, i64 1152, !dbg !14
  %185 = load <4 x half>, ptr addrspace(3) %lds.const.gep1220, align 8, !dbg !14
  %186 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %55, <4 x half> %184, <4 x float> %183) #9, !dbg !14
  %187 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %53, <4 x half> %185, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1221 = getelementptr i8, ptr addrspace(3) %address.local559, i64 5248, !dbg !14
  %188 = load <4 x half>, ptr addrspace(3) %lds.const.gep1221, align 8, !dbg !14
  %lds.const.gep1222 = getelementptr i8, ptr addrspace(3) %164, i64 1216, !dbg !14
  %189 = load <4 x half>, ptr addrspace(3) %lds.const.gep1222, align 8, !dbg !14
  %190 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %55, <4 x half> %188, <4 x float> %187) #9, !dbg !14
  %191 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %53, <4 x half> %189, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1223 = getelementptr i8, ptr addrspace(3) %164, i64 5312, !dbg !14
  %192 = load <4 x half>, ptr addrspace(3) %lds.const.gep1223, align 8, !dbg !14
  %193 = load <4 x half>, ptr addrspace(3) %address.local559, align 8, !dbg !14
  %194 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %55, <4 x half> %192, <4 x float> %191) #9, !dbg !14
  %195 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %193, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1224 = getelementptr i8, ptr addrspace(3) %address.local559, i64 4096, !dbg !14
  %196 = load <4 x half>, ptr addrspace(3) %lds.const.gep1224, align 8, !dbg !14
  %197 = load <4 x half>, ptr addrspace(3) %address.local588, align 8, !dbg !14
  %198 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %58, <4 x half> %196, <4 x float> %195) #9, !dbg !14
  %199 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %197, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1225 = getelementptr i8, ptr addrspace(3) %164, i64 4160, !dbg !14
  %200 = load <4 x half>, ptr addrspace(3) %lds.const.gep1225, align 8, !dbg !14
  %lds.const.gep1226 = getelementptr i8, ptr addrspace(3) %address.local559, i64 128, !dbg !14
  %201 = load <4 x half>, ptr addrspace(3) %lds.const.gep1226, align 8, !dbg !14
  %202 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %58, <4 x half> %200, <4 x float> %199) #9, !dbg !14
  %203 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %201, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1227 = getelementptr i8, ptr addrspace(3) %address.local559, i64 4224, !dbg !14
  %204 = load <4 x half>, ptr addrspace(3) %lds.const.gep1227, align 8, !dbg !14
  %lds.const.gep1228 = getelementptr i8, ptr addrspace(3) %164, i64 192, !dbg !14
  %205 = load <4 x half>, ptr addrspace(3) %lds.const.gep1228, align 8, !dbg !14
  %206 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %58, <4 x half> %204, <4 x float> %203) #9, !dbg !14
  %207 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %205, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1229 = getelementptr i8, ptr addrspace(3) %164, i64 4288, !dbg !14
  %208 = load <4 x half>, ptr addrspace(3) %lds.const.gep1229, align 8, !dbg !14
  %lds.const.gep1230 = getelementptr i8, ptr addrspace(3) %address.local559, i64 1024, !dbg !14
  %209 = load <4 x half>, ptr addrspace(3) %lds.const.gep1230, align 8, !dbg !14
  %210 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %58, <4 x half> %208, <4 x float> %207) #9, !dbg !14
  %211 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %209, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1231 = getelementptr i8, ptr addrspace(3) %address.local559, i64 5120, !dbg !14
  %212 = load <4 x half>, ptr addrspace(3) %lds.const.gep1231, align 8, !dbg !14
  %lds.const.gep1232 = getelementptr i8, ptr addrspace(3) %164, i64 1088, !dbg !14
  %213 = load <4 x half>, ptr addrspace(3) %lds.const.gep1232, align 8, !dbg !14
  %214 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %58, <4 x half> %212, <4 x float> %211) #9, !dbg !14
  %215 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %213, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1233 = getelementptr i8, ptr addrspace(3) %164, i64 5184, !dbg !14
  %216 = load <4 x half>, ptr addrspace(3) %lds.const.gep1233, align 8, !dbg !14
  %lds.const.gep1234 = getelementptr i8, ptr addrspace(3) %address.local559, i64 1152, !dbg !14
  %217 = load <4 x half>, ptr addrspace(3) %lds.const.gep1234, align 8, !dbg !14
  %218 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %58, <4 x half> %216, <4 x float> %215) #9, !dbg !14
  %219 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %217, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1235 = getelementptr i8, ptr addrspace(3) %address.local559, i64 5248, !dbg !14
  %220 = load <4 x half>, ptr addrspace(3) %lds.const.gep1235, align 8, !dbg !14
  %lds.const.gep1236 = getelementptr i8, ptr addrspace(3) %164, i64 1216, !dbg !14
  %221 = load <4 x half>, ptr addrspace(3) %lds.const.gep1236, align 8, !dbg !14
  %222 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %58, <4 x half> %220, <4 x float> %219) #9, !dbg !14
  %223 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %221, <4 x float> zeroinitializer) #9, !dbg !14
  %lds.const.gep1237 = getelementptr i8, ptr addrspace(3) %164, i64 5312, !dbg !14
  %224 = load <4 x half>, ptr addrspace(3) %lds.const.gep1237, align 8, !dbg !14
  %225 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %58, <4 x half> %224, <4 x float> %223) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %226 = or disjoint i64 %132, 64, !dbg !12
  %227 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local504), !dbg !12
  %228 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !12
  %.idx14111412 = or disjoint i64 %226, %address.local514, !dbg !12
  %.idx14111412.tr = trunc nuw i64 %.idx14111412 to i32, !dbg !12
  %dma.buffer.offset1360 = shl nuw nsw i32 %.idx14111412.tr, 1, !dbg !12
  %229 = insertelement <2 x i64> poison, i64 %228, i64 0, !dbg !12
  %230 = bitcast <2 x i64> %229 to <4 x i32>, !dbg !12
  %dma.buffer.rsrc1361 = shufflevector <4 x i32> %230, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !12
  %dma.buffer.lds1362 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %227), !dbg !12
  %231 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1362, i32 %dma.buffer.offset1360, <4 x i32> %dma.buffer.rsrc1361) #10, !dbg !12
  %232 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local521), !dbg !12
  %233 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !12
  %.idx14131414 = or disjoint i64 %226, %address.local531, !dbg !12
  %.idx14131414.tr = trunc nuw i64 %.idx14131414 to i32, !dbg !12
  %dma.buffer.offset1357 = shl nuw nsw i32 %.idx14131414.tr, 1, !dbg !12
  %234 = insertelement <2 x i64> poison, i64 %233, i64 0, !dbg !12
  %235 = bitcast <2 x i64> %234 to <4 x i32>, !dbg !12
  %dma.buffer.rsrc1358 = shufflevector <4 x i32> %235, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !12
  %dma.buffer.lds1359 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %232), !dbg !12
  %236 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1359, i32 %dma.buffer.offset1357, <4 x i32> %dma.buffer.rsrc1358) #10, !dbg !12
  %lds.const.gep1238 = getelementptr i8, ptr addrspace(3) %address.local559, i64 8192, !dbg !14
  %237 = load <4 x half>, ptr addrspace(3) %lds.const.gep1238, align 8, !dbg !14
  %238 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %76, <4 x half> %237, <4 x float> %166) #9, !dbg !14
  %lds.const.gep1239 = getelementptr i8, ptr addrspace(3) %address.local559, i64 12288, !dbg !14
  %239 = load <4 x half>, ptr addrspace(3) %lds.const.gep1239, align 8, !dbg !14
  %lds.const.gep1240 = getelementptr i8, ptr addrspace(3) %164, i64 8256, !dbg !14
  %240 = load <4 x half>, ptr addrspace(3) %lds.const.gep1240, align 8, !dbg !14
  %241 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %239, <4 x float> %238) #9, !dbg !14
  %242 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %76, <4 x half> %240, <4 x float> %170) #9, !dbg !14
  %lds.const.gep1241 = getelementptr i8, ptr addrspace(3) %164, i64 12352, !dbg !14
  %243 = load <4 x half>, ptr addrspace(3) %lds.const.gep1241, align 8, !dbg !14
  %lds.const.gep1242 = getelementptr i8, ptr addrspace(3) %address.local559, i64 8320, !dbg !14
  %244 = load <4 x half>, ptr addrspace(3) %lds.const.gep1242, align 8, !dbg !14
  %245 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %243, <4 x float> %242) #9, !dbg !14
  %246 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %76, <4 x half> %244, <4 x float> %174) #9, !dbg !14
  %lds.const.gep1243 = getelementptr i8, ptr addrspace(3) %address.local559, i64 12416, !dbg !14
  %247 = load <4 x half>, ptr addrspace(3) %lds.const.gep1243, align 8, !dbg !14
  %lds.const.gep1244 = getelementptr i8, ptr addrspace(3) %164, i64 8384, !dbg !14
  %248 = load <4 x half>, ptr addrspace(3) %lds.const.gep1244, align 8, !dbg !14
  %249 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %247, <4 x float> %246) #9, !dbg !14
  %250 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %76, <4 x half> %248, <4 x float> %178) #9, !dbg !14
  %lds.const.gep1245 = getelementptr i8, ptr addrspace(3) %164, i64 12480, !dbg !14
  %251 = load <4 x half>, ptr addrspace(3) %lds.const.gep1245, align 8, !dbg !14
  %lds.const.gep1246 = getelementptr i8, ptr addrspace(3) %address.local559, i64 9216, !dbg !14
  %252 = load <4 x half>, ptr addrspace(3) %lds.const.gep1246, align 8, !dbg !14
  %253 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %251, <4 x float> %250) #9, !dbg !14
  %254 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %76, <4 x half> %252, <4 x float> %182) #9, !dbg !14
  %lds.const.gep1247 = getelementptr i8, ptr addrspace(3) %address.local559, i64 13312, !dbg !14
  %255 = load <4 x half>, ptr addrspace(3) %lds.const.gep1247, align 8, !dbg !14
  %lds.const.gep1248 = getelementptr i8, ptr addrspace(3) %164, i64 9280, !dbg !14
  %256 = load <4 x half>, ptr addrspace(3) %lds.const.gep1248, align 8, !dbg !14
  %257 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %255, <4 x float> %254) #9, !dbg !14
  %258 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %76, <4 x half> %256, <4 x float> %186) #9, !dbg !14
  %lds.const.gep1249 = getelementptr i8, ptr addrspace(3) %164, i64 13376, !dbg !14
  %259 = load <4 x half>, ptr addrspace(3) %lds.const.gep1249, align 8, !dbg !14
  %lds.const.gep1250 = getelementptr i8, ptr addrspace(3) %address.local559, i64 9344, !dbg !14
  %260 = load <4 x half>, ptr addrspace(3) %lds.const.gep1250, align 8, !dbg !14
  %261 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %259, <4 x float> %258) #9, !dbg !14
  %262 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %76, <4 x half> %260, <4 x float> %190) #9, !dbg !14
  %lds.const.gep1251 = getelementptr i8, ptr addrspace(3) %address.local559, i64 13440, !dbg !14
  %263 = load <4 x half>, ptr addrspace(3) %lds.const.gep1251, align 8, !dbg !14
  %lds.const.gep1252 = getelementptr i8, ptr addrspace(3) %164, i64 9408, !dbg !14
  %264 = load <4 x half>, ptr addrspace(3) %lds.const.gep1252, align 8, !dbg !14
  %265 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %263, <4 x float> %262) #9, !dbg !14
  %266 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %76, <4 x half> %264, <4 x float> %194) #9, !dbg !14
  %lds.const.gep1253 = getelementptr i8, ptr addrspace(3) %164, i64 13504, !dbg !14
  %267 = load <4 x half>, ptr addrspace(3) %lds.const.gep1253, align 8, !dbg !14
  %lds.const.gep1254 = getelementptr i8, ptr addrspace(3) %address.local559, i64 8192, !dbg !14
  %268 = load <4 x half>, ptr addrspace(3) %lds.const.gep1254, align 8, !dbg !14
  %269 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %267, <4 x float> %266) #9, !dbg !14
  %270 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %268, <4 x float> %198) #9, !dbg !14
  %lds.const.gep1255 = getelementptr i8, ptr addrspace(3) %address.local559, i64 12288, !dbg !14
  %271 = load <4 x half>, ptr addrspace(3) %lds.const.gep1255, align 8, !dbg !14
  %lds.const.gep1256 = getelementptr i8, ptr addrspace(3) %164, i64 8256, !dbg !14
  %272 = load <4 x half>, ptr addrspace(3) %lds.const.gep1256, align 8, !dbg !14
  %273 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %271, <4 x float> %270) #9, !dbg !14
  %274 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %272, <4 x float> %202) #9, !dbg !14
  %lds.const.gep1257 = getelementptr i8, ptr addrspace(3) %164, i64 12352, !dbg !14
  %275 = load <4 x half>, ptr addrspace(3) %lds.const.gep1257, align 8, !dbg !14
  %lds.const.gep1258 = getelementptr i8, ptr addrspace(3) %address.local559, i64 8320, !dbg !14
  %276 = load <4 x half>, ptr addrspace(3) %lds.const.gep1258, align 8, !dbg !14
  %277 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %275, <4 x float> %274) #9, !dbg !14
  %278 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %276, <4 x float> %206) #9, !dbg !14
  %lds.const.gep1259 = getelementptr i8, ptr addrspace(3) %address.local559, i64 12416, !dbg !14
  %279 = load <4 x half>, ptr addrspace(3) %lds.const.gep1259, align 8, !dbg !14
  %lds.const.gep1260 = getelementptr i8, ptr addrspace(3) %164, i64 8384, !dbg !14
  %280 = load <4 x half>, ptr addrspace(3) %lds.const.gep1260, align 8, !dbg !14
  %281 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %279, <4 x float> %278) #9, !dbg !14
  %282 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %280, <4 x float> %210) #9, !dbg !14
  %lds.const.gep1261 = getelementptr i8, ptr addrspace(3) %164, i64 12480, !dbg !14
  %283 = load <4 x half>, ptr addrspace(3) %lds.const.gep1261, align 8, !dbg !14
  %lds.const.gep1262 = getelementptr i8, ptr addrspace(3) %address.local559, i64 9216, !dbg !14
  %284 = load <4 x half>, ptr addrspace(3) %lds.const.gep1262, align 8, !dbg !14
  %285 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %283, <4 x float> %282) #9, !dbg !14
  %286 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %284, <4 x float> %214) #9, !dbg !14
  %lds.const.gep1263 = getelementptr i8, ptr addrspace(3) %address.local559, i64 13312, !dbg !14
  %287 = load <4 x half>, ptr addrspace(3) %lds.const.gep1263, align 8, !dbg !14
  %lds.const.gep1264 = getelementptr i8, ptr addrspace(3) %164, i64 9280, !dbg !14
  %288 = load <4 x half>, ptr addrspace(3) %lds.const.gep1264, align 8, !dbg !14
  %289 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %287, <4 x float> %286) #9, !dbg !14
  %290 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %288, <4 x float> %218) #9, !dbg !14
  %lds.const.gep1265 = getelementptr i8, ptr addrspace(3) %164, i64 13376, !dbg !14
  %291 = load <4 x half>, ptr addrspace(3) %lds.const.gep1265, align 8, !dbg !14
  %lds.const.gep1266 = getelementptr i8, ptr addrspace(3) %address.local559, i64 9344, !dbg !14
  %292 = load <4 x half>, ptr addrspace(3) %lds.const.gep1266, align 8, !dbg !14
  %293 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %291, <4 x float> %290) #9, !dbg !14
  %294 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %292, <4 x float> %222) #9, !dbg !14
  %lds.const.gep1267 = getelementptr i8, ptr addrspace(3) %address.local559, i64 13440, !dbg !14
  %295 = load <4 x half>, ptr addrspace(3) %lds.const.gep1267, align 8, !dbg !14
  %lds.const.gep1268 = getelementptr i8, ptr addrspace(3) %164, i64 9408, !dbg !14
  %296 = load <4 x half>, ptr addrspace(3) %lds.const.gep1268, align 8, !dbg !14
  %297 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %295, <4 x float> %294) #9, !dbg !14
  %298 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %296, <4 x float> %225) #9, !dbg !14
  %lds.const.gep1269 = getelementptr i8, ptr addrspace(3) %164, i64 13504, !dbg !14
  %299 = load <4 x half>, ptr addrspace(3) %lds.const.gep1269, align 8, !dbg !14
  %300 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %299, <4 x float> %298) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %301 = or disjoint i64 %132, 96, !dbg !12
  %302 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local535), !dbg !12
  %303 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !12
  %.idx14151416 = or disjoint i64 %301, %address.local514, !dbg !12
  %.idx14151416.tr = trunc nuw i64 %.idx14151416 to i32, !dbg !12
  %dma.buffer.offset1354 = shl nuw nsw i32 %.idx14151416.tr, 1, !dbg !12
  %304 = insertelement <2 x i64> poison, i64 %303, i64 0, !dbg !12
  %305 = bitcast <2 x i64> %304 to <4 x i32>, !dbg !12
  %dma.buffer.rsrc1355 = shufflevector <4 x i32> %305, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !12
  %dma.buffer.lds1356 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %302), !dbg !12
  %306 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1356, i32 %dma.buffer.offset1354, <4 x i32> %dma.buffer.rsrc1355) #10, !dbg !12
  %307 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local539), !dbg !12
  %308 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !12
  %.idx14171418 = or disjoint i64 %301, %address.local531, !dbg !12
  %.idx14171418.tr = trunc nuw i64 %.idx14171418 to i32, !dbg !12
  %dma.buffer.offset1351 = shl nuw nsw i32 %.idx14171418.tr, 1, !dbg !12
  %309 = insertelement <2 x i64> poison, i64 %308, i64 0, !dbg !12
  %310 = bitcast <2 x i64> %309 to <4 x i32>, !dbg !12
  %dma.buffer.rsrc1352 = shufflevector <4 x i32> %310, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !12
  %dma.buffer.lds1353 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %307), !dbg !12
  %311 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1353, i32 %dma.buffer.offset1351, <4 x i32> %dma.buffer.rsrc1352) #10, !dbg !12
  %312 = load <4 x half>, ptr addrspace(3) %address.local559, align 8, !dbg !14
  %313 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %312, <4 x float> %241) #9, !dbg !14
  %lds.const.gep1270 = getelementptr i8, ptr addrspace(3) %address.local559, i64 4096, !dbg !14
  %314 = load <4 x half>, ptr addrspace(3) %lds.const.gep1270, align 8, !dbg !14
  %315 = load <4 x half>, ptr addrspace(3) %address.local588, align 8, !dbg !14
  %316 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %314, <4 x float> %313) #9, !dbg !14
  %317 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %315, <4 x float> %245) #9, !dbg !14
  %lds.const.gep1271 = getelementptr i8, ptr addrspace(3) %164, i64 4160, !dbg !14
  %318 = load <4 x half>, ptr addrspace(3) %lds.const.gep1271, align 8, !dbg !14
  %lds.const.gep1272 = getelementptr i8, ptr addrspace(3) %address.local559, i64 128, !dbg !14
  %319 = load <4 x half>, ptr addrspace(3) %lds.const.gep1272, align 8, !dbg !14
  %320 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %318, <4 x float> %317) #9, !dbg !14
  %321 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %319, <4 x float> %249) #9, !dbg !14
  %lds.const.gep1273 = getelementptr i8, ptr addrspace(3) %address.local559, i64 4224, !dbg !14
  %322 = load <4 x half>, ptr addrspace(3) %lds.const.gep1273, align 8, !dbg !14
  %lds.const.gep1274 = getelementptr i8, ptr addrspace(3) %164, i64 192, !dbg !14
  %323 = load <4 x half>, ptr addrspace(3) %lds.const.gep1274, align 8, !dbg !14
  %324 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %322, <4 x float> %321) #9, !dbg !14
  %325 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %323, <4 x float> %253) #9, !dbg !14
  %lds.const.gep1275 = getelementptr i8, ptr addrspace(3) %164, i64 4288, !dbg !14
  %326 = load <4 x half>, ptr addrspace(3) %lds.const.gep1275, align 8, !dbg !14
  %lds.const.gep1276 = getelementptr i8, ptr addrspace(3) %address.local559, i64 1024, !dbg !14
  %327 = load <4 x half>, ptr addrspace(3) %lds.const.gep1276, align 8, !dbg !14
  %328 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %326, <4 x float> %325) #9, !dbg !14
  %329 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %327, <4 x float> %257) #9, !dbg !14
  %lds.const.gep1277 = getelementptr i8, ptr addrspace(3) %address.local559, i64 5120, !dbg !14
  %330 = load <4 x half>, ptr addrspace(3) %lds.const.gep1277, align 8, !dbg !14
  %lds.const.gep1278 = getelementptr i8, ptr addrspace(3) %164, i64 1088, !dbg !14
  %331 = load <4 x half>, ptr addrspace(3) %lds.const.gep1278, align 8, !dbg !14
  %332 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %330, <4 x float> %329) #9, !dbg !14
  %333 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %331, <4 x float> %261) #9, !dbg !14
  %lds.const.gep1279 = getelementptr i8, ptr addrspace(3) %164, i64 5184, !dbg !14
  %334 = load <4 x half>, ptr addrspace(3) %lds.const.gep1279, align 8, !dbg !14
  %lds.const.gep1280 = getelementptr i8, ptr addrspace(3) %address.local559, i64 1152, !dbg !14
  %335 = load <4 x half>, ptr addrspace(3) %lds.const.gep1280, align 8, !dbg !14
  %336 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %334, <4 x float> %333) #9, !dbg !14
  %337 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %335, <4 x float> %265) #9, !dbg !14
  %lds.const.gep1281 = getelementptr i8, ptr addrspace(3) %address.local559, i64 5248, !dbg !14
  %338 = load <4 x half>, ptr addrspace(3) %lds.const.gep1281, align 8, !dbg !14
  %lds.const.gep1282 = getelementptr i8, ptr addrspace(3) %164, i64 1216, !dbg !14
  %339 = load <4 x half>, ptr addrspace(3) %lds.const.gep1282, align 8, !dbg !14
  %340 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %338, <4 x float> %337) #9, !dbg !14
  %341 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %339, <4 x float> %269) #9, !dbg !14
  %lds.const.gep1283 = getelementptr i8, ptr addrspace(3) %164, i64 5312, !dbg !14
  %342 = load <4 x half>, ptr addrspace(3) %lds.const.gep1283, align 8, !dbg !14
  %343 = load <4 x half>, ptr addrspace(3) %address.local559, align 8, !dbg !14
  %344 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %342, <4 x float> %341) #9, !dbg !14
  %345 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %343, <4 x float> %273) #9, !dbg !14
  %lds.const.gep1284 = getelementptr i8, ptr addrspace(3) %address.local559, i64 4096, !dbg !14
  %346 = load <4 x half>, ptr addrspace(3) %lds.const.gep1284, align 8, !dbg !14
  %347 = load <4 x half>, ptr addrspace(3) %address.local588, align 8, !dbg !14
  %348 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %346, <4 x float> %345) #9, !dbg !14
  %349 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %347, <4 x float> %277) #9, !dbg !14
  %lds.const.gep1285 = getelementptr i8, ptr addrspace(3) %164, i64 4160, !dbg !14
  %350 = load <4 x half>, ptr addrspace(3) %lds.const.gep1285, align 8, !dbg !14
  %lds.const.gep1286 = getelementptr i8, ptr addrspace(3) %address.local559, i64 128, !dbg !14
  %351 = load <4 x half>, ptr addrspace(3) %lds.const.gep1286, align 8, !dbg !14
  %352 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %350, <4 x float> %349) #9, !dbg !14
  %353 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %351, <4 x float> %281) #9, !dbg !14
  %lds.const.gep1287 = getelementptr i8, ptr addrspace(3) %address.local559, i64 4224, !dbg !14
  %354 = load <4 x half>, ptr addrspace(3) %lds.const.gep1287, align 8, !dbg !14
  %lds.const.gep1288 = getelementptr i8, ptr addrspace(3) %164, i64 192, !dbg !14
  %355 = load <4 x half>, ptr addrspace(3) %lds.const.gep1288, align 8, !dbg !14
  %356 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %354, <4 x float> %353) #9, !dbg !14
  %357 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %355, <4 x float> %285) #9, !dbg !14
  %lds.const.gep1289 = getelementptr i8, ptr addrspace(3) %164, i64 4288, !dbg !14
  %358 = load <4 x half>, ptr addrspace(3) %lds.const.gep1289, align 8, !dbg !14
  %lds.const.gep1290 = getelementptr i8, ptr addrspace(3) %address.local559, i64 1024, !dbg !14
  %359 = load <4 x half>, ptr addrspace(3) %lds.const.gep1290, align 8, !dbg !14
  %360 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %358, <4 x float> %357) #9, !dbg !14
  %361 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %359, <4 x float> %289) #9, !dbg !14
  %lds.const.gep1291 = getelementptr i8, ptr addrspace(3) %address.local559, i64 5120, !dbg !14
  %362 = load <4 x half>, ptr addrspace(3) %lds.const.gep1291, align 8, !dbg !14
  %lds.const.gep1292 = getelementptr i8, ptr addrspace(3) %164, i64 1088, !dbg !14
  %363 = load <4 x half>, ptr addrspace(3) %lds.const.gep1292, align 8, !dbg !14
  %364 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %362, <4 x float> %361) #9, !dbg !14
  %365 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %363, <4 x float> %293) #9, !dbg !14
  %lds.const.gep1293 = getelementptr i8, ptr addrspace(3) %164, i64 5184, !dbg !14
  %366 = load <4 x half>, ptr addrspace(3) %lds.const.gep1293, align 8, !dbg !14
  %lds.const.gep1294 = getelementptr i8, ptr addrspace(3) %address.local559, i64 1152, !dbg !14
  %367 = load <4 x half>, ptr addrspace(3) %lds.const.gep1294, align 8, !dbg !14
  %368 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %366, <4 x float> %365) #9, !dbg !14
  %369 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %367, <4 x float> %297) #9, !dbg !14
  %lds.const.gep1295 = getelementptr i8, ptr addrspace(3) %address.local559, i64 5248, !dbg !14
  %370 = load <4 x half>, ptr addrspace(3) %lds.const.gep1295, align 8, !dbg !14
  %lds.const.gep1296 = getelementptr i8, ptr addrspace(3) %164, i64 1216, !dbg !14
  %371 = load <4 x half>, ptr addrspace(3) %lds.const.gep1296, align 8, !dbg !14
  %372 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %370, <4 x float> %369) #9, !dbg !14
  %373 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %371, <4 x float> %300) #9, !dbg !14
  %lds.const.gep1297 = getelementptr i8, ptr addrspace(3) %164, i64 5312, !dbg !14
  %374 = load <4 x half>, ptr addrspace(3) %lds.const.gep1297, align 8, !dbg !14
  %375 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %374, <4 x float> %373) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %376 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local504), !dbg !15
  %address.local917 = shl nuw nsw i64 %address.local499, 5, !dbg !15
  %address.local918 = and i64 %address.local917, 8064, !dbg !15
  %address.local919 = shl nuw nsw i64 %address.local499, 3, !dbg !15
  %address.local920 = and i64 %address.local919, 24, !dbg !15
  %address.local921 = or disjoint i64 %address.local918, %address.local920, !dbg !15
  %377 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %.idx14191420 = or disjoint i64 %132, %address.local921, !dbg !15
  %.idx14191420.tr = trunc nuw i64 %.idx14191420 to i32, !dbg !15
  %dma.buffer.offset1348 = shl nuw nsw i32 %.idx14191420.tr, 1, !dbg !15
  %378 = insertelement <2 x i64> poison, i64 %377, i64 0, !dbg !15
  %379 = bitcast <2 x i64> %378 to <4 x i32>, !dbg !15
  %dma.buffer.rsrc1349 = shufflevector <4 x i32> %379, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %dma.buffer.lds1350 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %376), !dbg !15
  %380 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1350, i32 %dma.buffer.offset1348, <4 x i32> %dma.buffer.rsrc1349) #10, !dbg !15
  %381 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local521), !dbg !15
  %address.local923 = shl nuw nsw i64 %address.local516, 5, !dbg !15
  %address.local924 = and i64 %address.local923, 16256, !dbg !15
  %address.local925 = shl nuw nsw i64 %address.local499, 3, !dbg !15
  %address.local926 = and i64 %address.local925, 24, !dbg !15
  %address.local927 = or disjoint i64 %address.local924, %address.local926, !dbg !15
  %382 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %.idx14211422 = or disjoint i64 %132, %address.local927, !dbg !15
  %.idx14211422.tr = trunc nuw i64 %.idx14211422 to i32, !dbg !15
  %dma.buffer.offset1345 = shl nuw nsw i32 %.idx14211422.tr, 1, !dbg !15
  %383 = insertelement <2 x i64> poison, i64 %382, i64 0, !dbg !15
  %384 = bitcast <2 x i64> %383 to <4 x i32>, !dbg !15
  %dma.buffer.rsrc1346 = shufflevector <4 x i32> %384, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %dma.buffer.lds1347 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %381), !dbg !15
  %385 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1347, i32 %dma.buffer.offset1345, <4 x i32> %dma.buffer.rsrc1346) #10, !dbg !15
  %lds.const.gep1298 = getelementptr i8, ptr addrspace(3) %address.local559, i64 8192, !dbg !14
  %386 = load <4 x half>, ptr addrspace(3) %lds.const.gep1298, align 8, !dbg !14
  %387 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %106, <4 x half> %386, <4 x float> %316) #9, !dbg !14
  %lds.const.gep1299 = getelementptr i8, ptr addrspace(3) %address.local559, i64 12288, !dbg !14
  %388 = load <4 x half>, ptr addrspace(3) %lds.const.gep1299, align 8, !dbg !14
  %lds.const.gep1300 = getelementptr i8, ptr addrspace(3) %164, i64 8256, !dbg !14
  %389 = load <4 x half>, ptr addrspace(3) %lds.const.gep1300, align 8, !dbg !14
  %390 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %108, <4 x half> %388, <4 x float> %387) #9, !dbg !14
  %391 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %106, <4 x half> %389, <4 x float> %320) #9, !dbg !14
  %392 = extractelement <4 x float> %390, i64 0, !dbg !19
  %393 = call float @llvm.maxnum.f32(float %392, float 0xFFF0000000000000), !dbg !20, !frisk.local_reduction !1
  %394 = extractelement <4 x float> %390, i64 1, !dbg !19
  %395 = extractelement <4 x float> %390, i64 2, !dbg !19
  %396 = extractelement <4 x float> %390, i64 3, !dbg !19
  %lds.const.gep1301 = getelementptr i8, ptr addrspace(3) %164, i64 12352, !dbg !14
  %397 = load <4 x half>, ptr addrspace(3) %lds.const.gep1301, align 8, !dbg !14
  %lds.const.gep1302 = getelementptr i8, ptr addrspace(3) %address.local559, i64 8320, !dbg !14
  %398 = load <4 x half>, ptr addrspace(3) %lds.const.gep1302, align 8, !dbg !14
  %399 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %108, <4 x half> %397, <4 x float> %391) #9, !dbg !14
  %400 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %106, <4 x half> %398, <4 x float> %324) #9, !dbg !14
  %401 = extractelement <4 x float> %399, i64 0, !dbg !19
  %local.reduce151 = call float @llvm.maxnum.f32(float %393, float %401), !dbg !20, !frisk.local_reduction !1
  %402 = extractelement <4 x float> %399, i64 1, !dbg !19
  %local.reduce152 = call float @llvm.maxnum.f32(float %394, float %402), !dbg !20, !frisk.local_reduction !1
  %403 = extractelement <4 x float> %399, i64 2, !dbg !19
  %local.reduce153 = call float @llvm.maxnum.f32(float %395, float %403), !dbg !20, !frisk.local_reduction !1
  %404 = extractelement <4 x float> %399, i64 3, !dbg !19
  %local.reduce154 = call float @llvm.maxnum.f32(float %396, float %404), !dbg !20, !frisk.local_reduction !1
  %lds.const.gep1303 = getelementptr i8, ptr addrspace(3) %address.local559, i64 12416, !dbg !14
  %405 = load <4 x half>, ptr addrspace(3) %lds.const.gep1303, align 8, !dbg !14
  %lds.const.gep1304 = getelementptr i8, ptr addrspace(3) %164, i64 8384, !dbg !14
  %406 = load <4 x half>, ptr addrspace(3) %lds.const.gep1304, align 8, !dbg !14
  %407 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %108, <4 x half> %405, <4 x float> %400) #9, !dbg !14
  %408 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %106, <4 x half> %406, <4 x float> %328) #9, !dbg !14
  %409 = extractelement <4 x float> %407, i64 0, !dbg !19
  %local.reduce155 = call float @llvm.maxnum.f32(float %local.reduce151, float %409), !dbg !20, !frisk.local_reduction !1
  %410 = extractelement <4 x float> %407, i64 1, !dbg !19
  %local.reduce156 = call float @llvm.maxnum.f32(float %local.reduce152, float %410), !dbg !20, !frisk.local_reduction !1
  %411 = extractelement <4 x float> %407, i64 2, !dbg !19
  %local.reduce157 = call float @llvm.maxnum.f32(float %local.reduce153, float %411), !dbg !20, !frisk.local_reduction !1
  %412 = extractelement <4 x float> %407, i64 3, !dbg !19
  %local.reduce158 = call float @llvm.maxnum.f32(float %local.reduce154, float %412), !dbg !20, !frisk.local_reduction !1
  %lds.const.gep1305 = getelementptr i8, ptr addrspace(3) %164, i64 12480, !dbg !14
  %413 = load <4 x half>, ptr addrspace(3) %lds.const.gep1305, align 8, !dbg !14
  %lds.const.gep1306 = getelementptr i8, ptr addrspace(3) %address.local559, i64 9216, !dbg !14
  %414 = load <4 x half>, ptr addrspace(3) %lds.const.gep1306, align 8, !dbg !14
  %415 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %108, <4 x half> %413, <4 x float> %408) #9, !dbg !14
  %416 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %106, <4 x half> %414, <4 x float> %332) #9, !dbg !14
  %417 = extractelement <4 x float> %415, i64 0, !dbg !19
  %local.reduce159 = call float @llvm.maxnum.f32(float %local.reduce155, float %417), !dbg !20, !frisk.local_reduction !1
  %418 = extractelement <4 x float> %415, i64 1, !dbg !19
  %local.reduce160 = call float @llvm.maxnum.f32(float %local.reduce156, float %418), !dbg !20, !frisk.local_reduction !1
  %419 = extractelement <4 x float> %415, i64 2, !dbg !19
  %local.reduce161 = call float @llvm.maxnum.f32(float %local.reduce157, float %419), !dbg !20, !frisk.local_reduction !1
  %420 = extractelement <4 x float> %415, i64 3, !dbg !19
  %local.reduce162 = call float @llvm.maxnum.f32(float %local.reduce158, float %420), !dbg !20, !frisk.local_reduction !1
  %lds.const.gep1307 = getelementptr i8, ptr addrspace(3) %address.local559, i64 13312, !dbg !14
  %421 = load <4 x half>, ptr addrspace(3) %lds.const.gep1307, align 8, !dbg !14
  %lds.const.gep1308 = getelementptr i8, ptr addrspace(3) %164, i64 9280, !dbg !14
  %422 = load <4 x half>, ptr addrspace(3) %lds.const.gep1308, align 8, !dbg !14
  %423 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %108, <4 x half> %421, <4 x float> %416) #9, !dbg !14
  %424 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %106, <4 x half> %422, <4 x float> %336) #9, !dbg !14
  %425 = extractelement <4 x float> %423, i64 0, !dbg !19
  %local.reduce163 = call float @llvm.maxnum.f32(float %local.reduce159, float %425), !dbg !20, !frisk.local_reduction !1
  %426 = extractelement <4 x float> %423, i64 1, !dbg !19
  %local.reduce164 = call float @llvm.maxnum.f32(float %local.reduce160, float %426), !dbg !20, !frisk.local_reduction !1
  %427 = extractelement <4 x float> %423, i64 2, !dbg !19
  %local.reduce165 = call float @llvm.maxnum.f32(float %local.reduce161, float %427), !dbg !20, !frisk.local_reduction !1
  %428 = extractelement <4 x float> %423, i64 3, !dbg !19
  %local.reduce166 = call float @llvm.maxnum.f32(float %local.reduce162, float %428), !dbg !20, !frisk.local_reduction !1
  %lds.const.gep1309 = getelementptr i8, ptr addrspace(3) %164, i64 13376, !dbg !14
  %429 = load <4 x half>, ptr addrspace(3) %lds.const.gep1309, align 8, !dbg !14
  %lds.const.gep1310 = getelementptr i8, ptr addrspace(3) %address.local559, i64 9344, !dbg !14
  %430 = load <4 x half>, ptr addrspace(3) %lds.const.gep1310, align 8, !dbg !14
  %431 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %108, <4 x half> %429, <4 x float> %424) #9, !dbg !14
  %432 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %106, <4 x half> %430, <4 x float> %340) #9, !dbg !14
  %433 = extractelement <4 x float> %431, i64 0, !dbg !19
  %local.reduce167 = call float @llvm.maxnum.f32(float %local.reduce163, float %433), !dbg !20, !frisk.local_reduction !1
  %434 = extractelement <4 x float> %431, i64 1, !dbg !19
  %local.reduce168 = call float @llvm.maxnum.f32(float %local.reduce164, float %434), !dbg !20, !frisk.local_reduction !1
  %435 = extractelement <4 x float> %431, i64 2, !dbg !19
  %local.reduce169 = call float @llvm.maxnum.f32(float %local.reduce165, float %435), !dbg !20, !frisk.local_reduction !1
  %436 = extractelement <4 x float> %431, i64 3, !dbg !19
  %local.reduce170 = call float @llvm.maxnum.f32(float %local.reduce166, float %436), !dbg !20, !frisk.local_reduction !1
  %lds.const.gep1311 = getelementptr i8, ptr addrspace(3) %address.local559, i64 13440, !dbg !14
  %437 = load <4 x half>, ptr addrspace(3) %lds.const.gep1311, align 8, !dbg !14
  %lds.const.gep1312 = getelementptr i8, ptr addrspace(3) %164, i64 9408, !dbg !14
  %438 = load <4 x half>, ptr addrspace(3) %lds.const.gep1312, align 8, !dbg !14
  %439 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %108, <4 x half> %437, <4 x float> %432) #9, !dbg !14
  %440 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %106, <4 x half> %438, <4 x float> %344) #9, !dbg !14
  %441 = extractelement <4 x float> %439, i64 0, !dbg !19
  %local.reduce171 = call float @llvm.maxnum.f32(float %local.reduce167, float %441), !dbg !20, !frisk.local_reduction !1
  %442 = extractelement <4 x float> %439, i64 1, !dbg !19
  %local.reduce172 = call float @llvm.maxnum.f32(float %local.reduce168, float %442), !dbg !20, !frisk.local_reduction !1
  %443 = extractelement <4 x float> %439, i64 2, !dbg !19
  %local.reduce173 = call float @llvm.maxnum.f32(float %local.reduce169, float %443), !dbg !20, !frisk.local_reduction !1
  %444 = extractelement <4 x float> %439, i64 3, !dbg !19
  %local.reduce174 = call float @llvm.maxnum.f32(float %local.reduce170, float %444), !dbg !20, !frisk.local_reduction !1
  %lds.const.gep1313 = getelementptr i8, ptr addrspace(3) %164, i64 13504, !dbg !14
  %445 = load <4 x half>, ptr addrspace(3) %lds.const.gep1313, align 8, !dbg !14
  %lds.const.gep1314 = getelementptr i8, ptr addrspace(3) %address.local559, i64 8192, !dbg !14
  %446 = load <4 x half>, ptr addrspace(3) %lds.const.gep1314, align 8, !dbg !14
  %447 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %108, <4 x half> %445, <4 x float> %440) #9, !dbg !14
  %448 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %109, <4 x half> %446, <4 x float> %348) #9, !dbg !14
  %449 = extractelement <4 x float> %447, i64 0, !dbg !19
  %local.reduce175 = call float @llvm.maxnum.f32(float %local.reduce171, float %449), !dbg !20, !frisk.local_reduction !1
  %450 = extractelement <4 x float> %447, i64 1, !dbg !19
  %local.reduce176 = call float @llvm.maxnum.f32(float %local.reduce172, float %450), !dbg !20, !frisk.local_reduction !1
  %451 = extractelement <4 x float> %447, i64 2, !dbg !19
  %local.reduce177 = call float @llvm.maxnum.f32(float %local.reduce173, float %451), !dbg !20, !frisk.local_reduction !1
  %452 = extractelement <4 x float> %447, i64 3, !dbg !19
  %local.reduce178 = call float @llvm.maxnum.f32(float %local.reduce174, float %452), !dbg !20, !frisk.local_reduction !1
  %local.reduce179 = call float @llvm.maxnum.f32(float %local.reduce175, float %local.reduce176), !dbg !20, !frisk.local_reduction !1
  %local.reduce180 = call float @llvm.maxnum.f32(float %local.reduce177, float %local.reduce178), !dbg !20, !frisk.local_reduction !1
  %453 = call float @llvm.maxnum.f32(float %local.reduce179, float %local.reduce180), !dbg !20, !frisk.local_reduction !1
  %lds.const.gep1315 = getelementptr i8, ptr addrspace(3) %address.local559, i64 12288, !dbg !14
  %454 = load <4 x half>, ptr addrspace(3) %lds.const.gep1315, align 8, !dbg !14
  %lds.const.gep1316 = getelementptr i8, ptr addrspace(3) %164, i64 8256, !dbg !14
  %455 = load <4 x half>, ptr addrspace(3) %lds.const.gep1316, align 8, !dbg !14
  %456 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %110, <4 x half> %454, <4 x float> %448) #9, !dbg !14
  %457 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %109, <4 x half> %455, <4 x float> %352) #9, !dbg !14
  %458 = extractelement <4 x float> %456, i64 0, !dbg !19
  %459 = extractelement <4 x float> %456, i64 1, !dbg !19
  %460 = extractelement <4 x float> %456, i64 2, !dbg !19
  %461 = extractelement <4 x float> %456, i64 3, !dbg !19
  %lds.const.gep1317 = getelementptr i8, ptr addrspace(3) %164, i64 12352, !dbg !14
  %462 = load <4 x half>, ptr addrspace(3) %lds.const.gep1317, align 8, !dbg !14
  %lds.const.gep1318 = getelementptr i8, ptr addrspace(3) %address.local559, i64 8320, !dbg !14
  %463 = load <4 x half>, ptr addrspace(3) %lds.const.gep1318, align 8, !dbg !14
  %464 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %110, <4 x half> %462, <4 x float> %457) #9, !dbg !14
  %465 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %109, <4 x half> %463, <4 x float> %356) #9, !dbg !14
  %466 = extractelement <4 x float> %464, i64 0, !dbg !19
  %467 = extractelement <4 x float> %464, i64 1, !dbg !19
  %468 = extractelement <4 x float> %464, i64 2, !dbg !19
  %469 = extractelement <4 x float> %464, i64 3, !dbg !19
  %lds.const.gep1319 = getelementptr i8, ptr addrspace(3) %address.local559, i64 12416, !dbg !14
  %470 = load <4 x half>, ptr addrspace(3) %lds.const.gep1319, align 8, !dbg !14
  %lds.const.gep1320 = getelementptr i8, ptr addrspace(3) %164, i64 8384, !dbg !14
  %471 = load <4 x half>, ptr addrspace(3) %lds.const.gep1320, align 8, !dbg !14
  %472 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %110, <4 x half> %470, <4 x float> %465) #9, !dbg !14
  %473 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %109, <4 x half> %471, <4 x float> %360) #9, !dbg !14
  %474 = extractelement <4 x float> %472, i64 0, !dbg !19
  %475 = extractelement <4 x float> %472, i64 1, !dbg !19
  %476 = extractelement <4 x float> %472, i64 2, !dbg !19
  %477 = extractelement <4 x float> %472, i64 3, !dbg !19
  %lds.const.gep1321 = getelementptr i8, ptr addrspace(3) %164, i64 12480, !dbg !14
  %478 = load <4 x half>, ptr addrspace(3) %lds.const.gep1321, align 8, !dbg !14
  %lds.const.gep1322 = getelementptr i8, ptr addrspace(3) %address.local559, i64 9216, !dbg !14
  %479 = load <4 x half>, ptr addrspace(3) %lds.const.gep1322, align 8, !dbg !14
  %480 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %110, <4 x half> %478, <4 x float> %473) #9, !dbg !14
  %481 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %109, <4 x half> %479, <4 x float> %364) #9, !dbg !14
  %482 = extractelement <4 x float> %480, i64 0, !dbg !19
  %483 = extractelement <4 x float> %480, i64 1, !dbg !19
  %484 = extractelement <4 x float> %480, i64 2, !dbg !19
  %485 = extractelement <4 x float> %480, i64 3, !dbg !19
  %lds.const.gep1323 = getelementptr i8, ptr addrspace(3) %address.local559, i64 13312, !dbg !14
  %486 = load <4 x half>, ptr addrspace(3) %lds.const.gep1323, align 8, !dbg !14
  %lds.const.gep1324 = getelementptr i8, ptr addrspace(3) %164, i64 9280, !dbg !14
  %487 = load <4 x half>, ptr addrspace(3) %lds.const.gep1324, align 8, !dbg !14
  %488 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %110, <4 x half> %486, <4 x float> %481) #9, !dbg !14
  %489 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %109, <4 x half> %487, <4 x float> %368) #9, !dbg !14
  %490 = extractelement <4 x float> %488, i64 0, !dbg !19
  %491 = extractelement <4 x float> %488, i64 1, !dbg !19
  %492 = extractelement <4 x float> %488, i64 2, !dbg !19
  %493 = extractelement <4 x float> %488, i64 3, !dbg !19
  %lds.const.gep1325 = getelementptr i8, ptr addrspace(3) %164, i64 13376, !dbg !14
  %494 = load <4 x half>, ptr addrspace(3) %lds.const.gep1325, align 8, !dbg !14
  %lds.const.gep1326 = getelementptr i8, ptr addrspace(3) %address.local559, i64 9344, !dbg !14
  %495 = load <4 x half>, ptr addrspace(3) %lds.const.gep1326, align 8, !dbg !14
  %496 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %110, <4 x half> %494, <4 x float> %489) #9, !dbg !14
  %497 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %109, <4 x half> %495, <4 x float> %372) #9, !dbg !14
  %498 = extractelement <4 x float> %496, i64 0, !dbg !19
  %499 = extractelement <4 x float> %496, i64 1, !dbg !19
  %500 = extractelement <4 x float> %496, i64 2, !dbg !19
  %501 = extractelement <4 x float> %496, i64 3, !dbg !19
  %lds.const.gep1327 = getelementptr i8, ptr addrspace(3) %address.local559, i64 13440, !dbg !14
  %502 = load <4 x half>, ptr addrspace(3) %lds.const.gep1327, align 8, !dbg !14
  %lds.const.gep1328 = getelementptr i8, ptr addrspace(3) %164, i64 9408, !dbg !14
  %503 = load <4 x half>, ptr addrspace(3) %lds.const.gep1328, align 8, !dbg !14
  %504 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %110, <4 x half> %502, <4 x float> %497) #9, !dbg !14
  %505 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %109, <4 x half> %503, <4 x float> %375) #9, !dbg !14
  %506 = extractelement <4 x float> %504, i64 0, !dbg !19
  %507 = extractelement <4 x float> %504, i64 1, !dbg !19
  %508 = extractelement <4 x float> %504, i64 2, !dbg !19
  %509 = extractelement <4 x float> %504, i64 3, !dbg !19
  %lds.const.gep1329 = getelementptr i8, ptr addrspace(3) %164, i64 13504, !dbg !14
  %510 = load <4 x half>, ptr addrspace(3) %lds.const.gep1329, align 8, !dbg !14
  %511 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %110, <4 x half> %510, <4 x float> %505) #9, !dbg !14
  %512 = extractelement <4 x float> %511, i64 0, !dbg !19
  %513 = extractelement <4 x float> %511, i64 1, !dbg !19
  %514 = extractelement <4 x float> %511, i64 2, !dbg !19
  %515 = extractelement <4 x float> %511, i64 3, !dbg !19
  %516 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !20
  %517 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %516), !dbg !20
  %518 = add nuw nsw i32 %517, 64, !dbg !20
  %519 = xor i32 %517, 16, !dbg !20
  %520 = and i32 %518, 192, !dbg !20
  %521 = icmp samesign ult i32 %519, %520, !dbg !20
  %522 = select i1 %521, i32 %519, i32 %517, !dbg !20
  %523 = shl nuw nsw i32 %522, 2, !dbg !20
  %524 = bitcast float %453 to i32, !dbg !20
  %525 = call i32 @llvm.amdgcn.ds.bpermute(i32 %523, i32 %524), !dbg !20
  %526 = bitcast i32 %525 to float, !dbg !20
  %527 = call float @llvm.maxnum.f32(float %453, float %526), !dbg !20
  %528 = xor i32 %517, 32, !dbg !20
  %529 = icmp samesign ult i32 %528, %520, !dbg !20
  %530 = select i1 %529, i32 %528, i32 %517, !dbg !20
  %531 = shl nuw nsw i32 %530, 2, !dbg !20
  %532 = bitcast float %527 to i32, !dbg !20
  %533 = call i32 @llvm.amdgcn.ds.bpermute(i32 %531, i32 %532), !dbg !20
  %534 = bitcast i32 %533 to float, !dbg !20
  %535 = call float @llvm.maxnum.f32(float %527, float %534), !dbg !20
  %536 = extractvalue [2 x <1 x float>] %123, 0, !dbg !20
  %537 = insertelement <1 x float> poison, float %535, i64 0, !dbg !20
  %538 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %537, <1 x float> %536), !dbg !20
  %539 = call float @llvm.maxnum.f32(float %458, float 0xFFF0000000000000), !dbg !20, !frisk.local_reduction !1
  %local.reduce120 = call float @llvm.maxnum.f32(float %539, float %466), !dbg !20, !frisk.local_reduction !1
  %local.reduce121 = call float @llvm.maxnum.f32(float %459, float %467), !dbg !20, !frisk.local_reduction !1
  %local.reduce122 = call float @llvm.maxnum.f32(float %460, float %468), !dbg !20, !frisk.local_reduction !1
  %local.reduce123 = call float @llvm.maxnum.f32(float %461, float %469), !dbg !20, !frisk.local_reduction !1
  %local.reduce124 = call float @llvm.maxnum.f32(float %local.reduce120, float %474), !dbg !20, !frisk.local_reduction !1
  %local.reduce125 = call float @llvm.maxnum.f32(float %local.reduce121, float %475), !dbg !20, !frisk.local_reduction !1
  %local.reduce126 = call float @llvm.maxnum.f32(float %local.reduce122, float %476), !dbg !20, !frisk.local_reduction !1
  %local.reduce127 = call float @llvm.maxnum.f32(float %local.reduce123, float %477), !dbg !20, !frisk.local_reduction !1
  %local.reduce128 = call float @llvm.maxnum.f32(float %local.reduce124, float %482), !dbg !20, !frisk.local_reduction !1
  %local.reduce129 = call float @llvm.maxnum.f32(float %local.reduce125, float %483), !dbg !20, !frisk.local_reduction !1
  %local.reduce130 = call float @llvm.maxnum.f32(float %local.reduce126, float %484), !dbg !20, !frisk.local_reduction !1
  %local.reduce131 = call float @llvm.maxnum.f32(float %local.reduce127, float %485), !dbg !20, !frisk.local_reduction !1
  %local.reduce132 = call float @llvm.maxnum.f32(float %local.reduce128, float %490), !dbg !20, !frisk.local_reduction !1
  %local.reduce133 = call float @llvm.maxnum.f32(float %local.reduce129, float %491), !dbg !20, !frisk.local_reduction !1
  %local.reduce134 = call float @llvm.maxnum.f32(float %local.reduce130, float %492), !dbg !20, !frisk.local_reduction !1
  %local.reduce135 = call float @llvm.maxnum.f32(float %local.reduce131, float %493), !dbg !20, !frisk.local_reduction !1
  %local.reduce136 = call float @llvm.maxnum.f32(float %local.reduce132, float %498), !dbg !20, !frisk.local_reduction !1
  %local.reduce137 = call float @llvm.maxnum.f32(float %local.reduce133, float %499), !dbg !20, !frisk.local_reduction !1
  %local.reduce138 = call float @llvm.maxnum.f32(float %local.reduce134, float %500), !dbg !20, !frisk.local_reduction !1
  %local.reduce139 = call float @llvm.maxnum.f32(float %local.reduce135, float %501), !dbg !20, !frisk.local_reduction !1
  %local.reduce140 = call float @llvm.maxnum.f32(float %local.reduce136, float %506), !dbg !20, !frisk.local_reduction !1
  %local.reduce141 = call float @llvm.maxnum.f32(float %local.reduce137, float %507), !dbg !20, !frisk.local_reduction !1
  %local.reduce142 = call float @llvm.maxnum.f32(float %local.reduce138, float %508), !dbg !20, !frisk.local_reduction !1
  %local.reduce143 = call float @llvm.maxnum.f32(float %local.reduce139, float %509), !dbg !20, !frisk.local_reduction !1
  %local.reduce144 = call float @llvm.maxnum.f32(float %local.reduce140, float %512), !dbg !20, !frisk.local_reduction !1
  %local.reduce145 = call float @llvm.maxnum.f32(float %local.reduce141, float %513), !dbg !20, !frisk.local_reduction !1
  %local.reduce146 = call float @llvm.maxnum.f32(float %local.reduce142, float %514), !dbg !20, !frisk.local_reduction !1
  %local.reduce147 = call float @llvm.maxnum.f32(float %local.reduce143, float %515), !dbg !20, !frisk.local_reduction !1
  %local.reduce148 = call float @llvm.maxnum.f32(float %local.reduce144, float %local.reduce145), !dbg !20, !frisk.local_reduction !1
  %local.reduce149 = call float @llvm.maxnum.f32(float %local.reduce146, float %local.reduce147), !dbg !20, !frisk.local_reduction !1
  %540 = call float @llvm.maxnum.f32(float %local.reduce148, float %local.reduce149), !dbg !20, !frisk.local_reduction !1
  %541 = bitcast float %540 to i32, !dbg !20
  %542 = call i32 @llvm.amdgcn.ds.bpermute(i32 %523, i32 %541), !dbg !20
  %543 = bitcast i32 %542 to float, !dbg !20
  %544 = call float @llvm.maxnum.f32(float %540, float %543), !dbg !20
  %545 = bitcast float %544 to i32, !dbg !20
  %546 = call i32 @llvm.amdgcn.ds.bpermute(i32 %531, i32 %545), !dbg !20
  %547 = bitcast i32 %546 to float, !dbg !20
  %548 = call float @llvm.maxnum.f32(float %544, float %547), !dbg !20
  %549 = extractvalue [2 x <1 x float>] %123, 1, !dbg !20
  %550 = insertelement <1 x float> poison, float %548, i64 0, !dbg !20
  %551 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %550, <1 x float> %549), !dbg !20
  %552 = shufflevector <1 x float> %538, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !21
  %553 = fsub <4 x float> %390, %552, !dbg !21
  %554 = fsub <4 x float> %399, %552, !dbg !21
  %555 = fsub <4 x float> %407, %552, !dbg !21
  %556 = fsub <4 x float> %415, %552, !dbg !21
  %557 = fsub <4 x float> %423, %552, !dbg !21
  %558 = fsub <4 x float> %431, %552, !dbg !21
  %559 = fsub <4 x float> %439, %552, !dbg !21
  %560 = fsub <4 x float> %447, %552, !dbg !21
  %561 = shufflevector <1 x float> %551, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !21
  %562 = fsub <4 x float> %456, %561, !dbg !21
  %563 = fsub <4 x float> %464, %561, !dbg !21
  %564 = fsub <4 x float> %472, %561, !dbg !21
  %565 = fsub <4 x float> %480, %561, !dbg !21
  %566 = fsub <4 x float> %488, %561, !dbg !21
  %567 = fsub <4 x float> %496, %561, !dbg !21
  %568 = fsub <4 x float> %504, %561, !dbg !21
  %569 = fsub <4 x float> %511, %561, !dbg !21
  %570 = fsub <1 x float> %536, %538, !dbg !22
  %571 = extractelement <1 x float> %570, i64 0, !dbg !23
  %572 = call float @llvm.amdgcn.exp2.f32(float %571), !dbg !23
  %573 = insertelement <4 x float> poison, float %572, i64 0, !dbg !28
  %574 = shufflevector <4 x float> %573, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !28
  %575 = extractvalue [2 x <1 x float>] %122, 0, !dbg !24
  %576 = extractelement <1 x float> %575, i64 0, !dbg !24
  %577 = fmul float %576, %572, !dbg !24
  %578 = fsub <1 x float> %549, %551, !dbg !22
  %579 = extractelement <1 x float> %578, i64 0, !dbg !23
  %580 = call float @llvm.amdgcn.exp2.f32(float %579), !dbg !23
  %581 = insertelement <4 x float> poison, float %580, i64 0, !dbg !28
  %582 = shufflevector <4 x float> %581, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !28
  %583 = extractvalue [2 x <1 x float>] %122, 1, !dbg !24
  %584 = extractelement <1 x float> %583, i64 0, !dbg !24
  %585 = fmul float %584, %580, !dbg !24
  %586 = extractelement <4 x float> %553, i64 0, !dbg !25
  %587 = call float @llvm.amdgcn.exp2.f32(float %586), !dbg !25
  %588 = insertelement <4 x float> poison, float %587, i64 0, !dbg !27
  %589 = extractelement <4 x float> %553, i64 1, !dbg !25
  %590 = call float @llvm.amdgcn.exp2.f32(float %589), !dbg !25
  %591 = insertelement <4 x float> %588, float %590, i64 1, !dbg !27
  %592 = extractelement <4 x float> %553, i64 2, !dbg !25
  %593 = call float @llvm.amdgcn.exp2.f32(float %592), !dbg !25
  %594 = insertelement <4 x float> %591, float %593, i64 2, !dbg !27
  %595 = extractelement <4 x float> %553, i64 3, !dbg !25
  %596 = call float @llvm.amdgcn.exp2.f32(float %595), !dbg !25
  %597 = insertelement <4 x float> %594, float %596, i64 3, !dbg !27
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %597) #9, !dbg !27, !frisk.register_lifetimes !1
  %598 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !27
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %598) #9, !dbg !25, !frisk.register_lifetimes !1
  %599 = extractelement <4 x float> %554, i64 0, !dbg !25
  %600 = call float @llvm.amdgcn.exp2.f32(float %599), !dbg !25
  %601 = insertelement <4 x float> poison, float %600, i64 0, !dbg !27
  %local.reduce89 = fadd float %587, %600, !dbg !26, !frisk.local_reduction !1
  %602 = extractelement <4 x float> %554, i64 1, !dbg !25
  %603 = call float @llvm.amdgcn.exp2.f32(float %602), !dbg !25
  %604 = insertelement <4 x float> %601, float %603, i64 1, !dbg !27
  %local.reduce90 = fadd float %590, %603, !dbg !26, !frisk.local_reduction !1
  %605 = extractelement <4 x float> %554, i64 2, !dbg !25
  %606 = call float @llvm.amdgcn.exp2.f32(float %605), !dbg !25
  %607 = insertelement <4 x float> %604, float %606, i64 2, !dbg !27
  %local.reduce91 = fadd float %593, %606, !dbg !26, !frisk.local_reduction !1
  %608 = extractelement <4 x float> %554, i64 3, !dbg !25
  %609 = call float @llvm.amdgcn.exp2.f32(float %608), !dbg !25
  %610 = insertelement <4 x float> %607, float %609, i64 3, !dbg !27
  %math.result374 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %610) #9, !dbg !27, !frisk.register_lifetimes !1
  %611 = fptrunc <4 x float> %math.result374 to <4 x half>, !dbg !27
  %math.packed375 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %611) #9, !dbg !26, !frisk.register_lifetimes !1
  %local.reduce92 = fadd float %596, %609, !dbg !26, !frisk.local_reduction !1
  %612 = extractelement <4 x float> %555, i64 0, !dbg !25
  %613 = call float @llvm.amdgcn.exp2.f32(float %612), !dbg !25
  %614 = insertelement <4 x float> poison, float %613, i64 0, !dbg !27
  %local.reduce93 = fadd float %local.reduce89, %613, !dbg !26, !frisk.local_reduction !1
  %615 = extractelement <4 x float> %555, i64 1, !dbg !25
  %616 = call float @llvm.amdgcn.exp2.f32(float %615), !dbg !25
  %617 = insertelement <4 x float> %614, float %616, i64 1, !dbg !27
  %local.reduce94 = fadd float %local.reduce90, %616, !dbg !26, !frisk.local_reduction !1
  %618 = extractelement <4 x float> %555, i64 2, !dbg !25
  %619 = call float @llvm.amdgcn.exp2.f32(float %618), !dbg !25
  %620 = insertelement <4 x float> %617, float %619, i64 2, !dbg !27
  %local.reduce95 = fadd float %local.reduce91, %619, !dbg !26, !frisk.local_reduction !1
  %621 = extractelement <4 x float> %555, i64 3, !dbg !25
  %622 = call float @llvm.amdgcn.exp2.f32(float %621), !dbg !25
  %623 = insertelement <4 x float> %620, float %622, i64 3, !dbg !27
  %math.result376 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %623) #9, !dbg !27, !frisk.register_lifetimes !1
  %624 = fptrunc <4 x float> %math.result376 to <4 x half>, !dbg !27
  %math.packed377 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %624) #9, !dbg !26, !frisk.register_lifetimes !1
  %local.reduce96 = fadd float %local.reduce92, %622, !dbg !26, !frisk.local_reduction !1
  %625 = extractelement <4 x float> %556, i64 0, !dbg !25
  %626 = call float @llvm.amdgcn.exp2.f32(float %625), !dbg !25
  %627 = insertelement <4 x float> poison, float %626, i64 0, !dbg !27
  %local.reduce97 = fadd float %local.reduce93, %626, !dbg !26, !frisk.local_reduction !1
  %628 = extractelement <4 x float> %556, i64 1, !dbg !25
  %629 = call float @llvm.amdgcn.exp2.f32(float %628), !dbg !25
  %630 = insertelement <4 x float> %627, float %629, i64 1, !dbg !27
  %local.reduce98 = fadd float %local.reduce94, %629, !dbg !26, !frisk.local_reduction !1
  %631 = extractelement <4 x float> %556, i64 2, !dbg !25
  %632 = call float @llvm.amdgcn.exp2.f32(float %631), !dbg !25
  %633 = insertelement <4 x float> %630, float %632, i64 2, !dbg !27
  %local.reduce99 = fadd float %local.reduce95, %632, !dbg !26, !frisk.local_reduction !1
  %634 = extractelement <4 x float> %556, i64 3, !dbg !25
  %635 = call float @llvm.amdgcn.exp2.f32(float %634), !dbg !25
  %636 = insertelement <4 x float> %633, float %635, i64 3, !dbg !27
  %math.result378 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %636) #9, !dbg !27, !frisk.register_lifetimes !1
  %637 = fptrunc <4 x float> %math.result378 to <4 x half>, !dbg !27
  %math.packed379 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %637) #9, !dbg !26, !frisk.register_lifetimes !1
  %local.reduce100 = fadd float %local.reduce96, %635, !dbg !26, !frisk.local_reduction !1
  %638 = extractelement <4 x float> %557, i64 0, !dbg !25
  %639 = call float @llvm.amdgcn.exp2.f32(float %638), !dbg !25
  %640 = insertelement <4 x float> poison, float %639, i64 0, !dbg !27
  %local.reduce101 = fadd float %local.reduce97, %639, !dbg !26, !frisk.local_reduction !1
  %641 = extractelement <4 x float> %557, i64 1, !dbg !25
  %642 = call float @llvm.amdgcn.exp2.f32(float %641), !dbg !25
  %643 = insertelement <4 x float> %640, float %642, i64 1, !dbg !27
  %local.reduce102 = fadd float %local.reduce98, %642, !dbg !26, !frisk.local_reduction !1
  %644 = extractelement <4 x float> %557, i64 2, !dbg !25
  %645 = call float @llvm.amdgcn.exp2.f32(float %644), !dbg !25
  %646 = insertelement <4 x float> %643, float %645, i64 2, !dbg !27
  %local.reduce103 = fadd float %local.reduce99, %645, !dbg !26, !frisk.local_reduction !1
  %647 = extractelement <4 x float> %557, i64 3, !dbg !25
  %648 = call float @llvm.amdgcn.exp2.f32(float %647), !dbg !25
  %649 = insertelement <4 x float> %646, float %648, i64 3, !dbg !27
  %math.result380 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %649) #9, !dbg !27, !frisk.register_lifetimes !1
  %650 = fptrunc <4 x float> %math.result380 to <4 x half>, !dbg !27
  %math.packed381 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %650) #9, !dbg !26, !frisk.register_lifetimes !1
  %local.reduce104 = fadd float %local.reduce100, %648, !dbg !26, !frisk.local_reduction !1
  %651 = extractelement <4 x float> %558, i64 0, !dbg !25
  %652 = call float @llvm.amdgcn.exp2.f32(float %651), !dbg !25
  %653 = insertelement <4 x float> poison, float %652, i64 0, !dbg !27
  %local.reduce105 = fadd float %local.reduce101, %652, !dbg !26, !frisk.local_reduction !1
  %654 = extractelement <4 x float> %558, i64 1, !dbg !25
  %655 = call float @llvm.amdgcn.exp2.f32(float %654), !dbg !25
  %656 = insertelement <4 x float> %653, float %655, i64 1, !dbg !27
  %local.reduce106 = fadd float %local.reduce102, %655, !dbg !26, !frisk.local_reduction !1
  %657 = extractelement <4 x float> %558, i64 2, !dbg !25
  %658 = call float @llvm.amdgcn.exp2.f32(float %657), !dbg !25
  %659 = insertelement <4 x float> %656, float %658, i64 2, !dbg !27
  %local.reduce107 = fadd float %local.reduce103, %658, !dbg !26, !frisk.local_reduction !1
  %660 = extractelement <4 x float> %558, i64 3, !dbg !25
  %661 = call float @llvm.amdgcn.exp2.f32(float %660), !dbg !25
  %662 = insertelement <4 x float> %659, float %661, i64 3, !dbg !27
  %math.result382 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %662) #9, !dbg !27, !frisk.register_lifetimes !1
  %663 = fptrunc <4 x float> %math.result382 to <4 x half>, !dbg !27
  %math.packed383 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %663) #9, !dbg !26, !frisk.register_lifetimes !1
  %local.reduce108 = fadd float %local.reduce104, %661, !dbg !26, !frisk.local_reduction !1
  %664 = extractelement <4 x float> %559, i64 0, !dbg !25
  %665 = call float @llvm.amdgcn.exp2.f32(float %664), !dbg !25
  %666 = insertelement <4 x float> poison, float %665, i64 0, !dbg !27
  %local.reduce109 = fadd float %local.reduce105, %665, !dbg !26, !frisk.local_reduction !1
  %667 = extractelement <4 x float> %559, i64 1, !dbg !25
  %668 = call float @llvm.amdgcn.exp2.f32(float %667), !dbg !25
  %669 = insertelement <4 x float> %666, float %668, i64 1, !dbg !27
  %local.reduce110 = fadd float %local.reduce106, %668, !dbg !26, !frisk.local_reduction !1
  %670 = extractelement <4 x float> %559, i64 2, !dbg !25
  %671 = call float @llvm.amdgcn.exp2.f32(float %670), !dbg !25
  %672 = insertelement <4 x float> %669, float %671, i64 2, !dbg !27
  %local.reduce111 = fadd float %local.reduce107, %671, !dbg !26, !frisk.local_reduction !1
  %673 = extractelement <4 x float> %559, i64 3, !dbg !25
  %674 = call float @llvm.amdgcn.exp2.f32(float %673), !dbg !25
  %675 = insertelement <4 x float> %672, float %674, i64 3, !dbg !27
  %math.result384 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %675) #9, !dbg !27, !frisk.register_lifetimes !1
  %676 = fptrunc <4 x float> %math.result384 to <4 x half>, !dbg !27
  %math.packed385 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %676) #9, !dbg !26, !frisk.register_lifetimes !1
  %local.reduce112 = fadd float %local.reduce108, %674, !dbg !26, !frisk.local_reduction !1
  %677 = extractelement <4 x float> %560, i64 0, !dbg !25
  %678 = call float @llvm.amdgcn.exp2.f32(float %677), !dbg !25
  %679 = insertelement <4 x float> poison, float %678, i64 0, !dbg !27
  %local.reduce113 = fadd float %local.reduce109, %678, !dbg !26, !frisk.local_reduction !1
  %680 = extractelement <4 x float> %560, i64 1, !dbg !25
  %681 = call float @llvm.amdgcn.exp2.f32(float %680), !dbg !25
  %682 = insertelement <4 x float> %679, float %681, i64 1, !dbg !27
  %local.reduce114 = fadd float %local.reduce110, %681, !dbg !26, !frisk.local_reduction !1
  %local.reduce117 = fadd float %local.reduce113, %local.reduce114, !dbg !26, !frisk.local_reduction !1
  %683 = extractelement <4 x float> %560, i64 2, !dbg !25
  %684 = call float @llvm.amdgcn.exp2.f32(float %683), !dbg !25
  %685 = insertelement <4 x float> %682, float %684, i64 2, !dbg !27
  %local.reduce115 = fadd float %local.reduce111, %684, !dbg !26, !frisk.local_reduction !1
  %686 = extractelement <4 x float> %560, i64 3, !dbg !25
  %687 = call float @llvm.amdgcn.exp2.f32(float %686), !dbg !25
  %688 = insertelement <4 x float> %685, float %687, i64 3, !dbg !27
  %math.result386 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %688) #9, !dbg !27, !frisk.register_lifetimes !1
  %689 = fptrunc <4 x float> %math.result386 to <4 x half>, !dbg !27
  %math.packed387 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %689) #9, !dbg !26, !frisk.register_lifetimes !1
  %local.reduce116 = fadd float %local.reduce112, %687, !dbg !26, !frisk.local_reduction !1
  %local.reduce118 = fadd float %local.reduce115, %local.reduce116, !dbg !26, !frisk.local_reduction !1
  %690 = fadd float %local.reduce117, %local.reduce118, !dbg !26, !frisk.local_reduction !1
  %691 = fadd float %690, %577, !dbg !26
  %692 = insertelement <1 x float> poison, float %691, i64 0, !dbg !7
  %693 = insertvalue [2 x <1 x float>] poison, <1 x float> %692, 0, !dbg !7
  %694 = extractelement <4 x float> %562, i64 0, !dbg !25
  %695 = call float @llvm.amdgcn.exp2.f32(float %694), !dbg !25
  %696 = insertelement <4 x float> poison, float %695, i64 0, !dbg !27
  %697 = extractelement <4 x float> %562, i64 1, !dbg !25
  %698 = call float @llvm.amdgcn.exp2.f32(float %697), !dbg !25
  %699 = insertelement <4 x float> %696, float %698, i64 1, !dbg !27
  %700 = extractelement <4 x float> %562, i64 2, !dbg !25
  %701 = call float @llvm.amdgcn.exp2.f32(float %700), !dbg !25
  %702 = insertelement <4 x float> %699, float %701, i64 2, !dbg !27
  %703 = extractelement <4 x float> %562, i64 3, !dbg !25
  %704 = call float @llvm.amdgcn.exp2.f32(float %703), !dbg !25
  %705 = insertelement <4 x float> %702, float %704, i64 3, !dbg !27
  %math.result388 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %705) #9, !dbg !27, !frisk.register_lifetimes !1
  %706 = fptrunc <4 x float> %math.result388 to <4 x half>, !dbg !27
  %math.packed389 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %706) #9, !dbg !25, !frisk.register_lifetimes !1
  %707 = extractelement <4 x float> %563, i64 0, !dbg !25
  %708 = call float @llvm.amdgcn.exp2.f32(float %707), !dbg !25
  %709 = insertelement <4 x float> poison, float %708, i64 0, !dbg !27
  %local.reduce = fadd float %695, %708, !dbg !26, !frisk.local_reduction !1
  %710 = extractelement <4 x float> %563, i64 1, !dbg !25
  %711 = call float @llvm.amdgcn.exp2.f32(float %710), !dbg !25
  %712 = insertelement <4 x float> %709, float %711, i64 1, !dbg !27
  %local.reduce59 = fadd float %698, %711, !dbg !26, !frisk.local_reduction !1
  %713 = extractelement <4 x float> %563, i64 2, !dbg !25
  %714 = call float @llvm.amdgcn.exp2.f32(float %713), !dbg !25
  %715 = insertelement <4 x float> %712, float %714, i64 2, !dbg !27
  %local.reduce60 = fadd float %701, %714, !dbg !26, !frisk.local_reduction !1
  %716 = extractelement <4 x float> %563, i64 3, !dbg !25
  %717 = call float @llvm.amdgcn.exp2.f32(float %716), !dbg !25
  %718 = insertelement <4 x float> %715, float %717, i64 3, !dbg !27
  %math.result390 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %718) #9, !dbg !27, !frisk.register_lifetimes !1
  %719 = fptrunc <4 x float> %math.result390 to <4 x half>, !dbg !27
  %math.packed391 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %719) #9, !dbg !26, !frisk.register_lifetimes !1
  %local.reduce61 = fadd float %704, %717, !dbg !26, !frisk.local_reduction !1
  %720 = extractelement <4 x float> %564, i64 0, !dbg !25
  %721 = call float @llvm.amdgcn.exp2.f32(float %720), !dbg !25
  %722 = insertelement <4 x float> poison, float %721, i64 0, !dbg !27
  %local.reduce62 = fadd float %local.reduce, %721, !dbg !26, !frisk.local_reduction !1
  %723 = extractelement <4 x float> %564, i64 1, !dbg !25
  %724 = call float @llvm.amdgcn.exp2.f32(float %723), !dbg !25
  %725 = insertelement <4 x float> %722, float %724, i64 1, !dbg !27
  %local.reduce63 = fadd float %local.reduce59, %724, !dbg !26, !frisk.local_reduction !1
  %726 = extractelement <4 x float> %564, i64 2, !dbg !25
  %727 = call float @llvm.amdgcn.exp2.f32(float %726), !dbg !25
  %728 = insertelement <4 x float> %725, float %727, i64 2, !dbg !27
  %local.reduce64 = fadd float %local.reduce60, %727, !dbg !26, !frisk.local_reduction !1
  %729 = extractelement <4 x float> %564, i64 3, !dbg !25
  %730 = call float @llvm.amdgcn.exp2.f32(float %729), !dbg !25
  %731 = insertelement <4 x float> %728, float %730, i64 3, !dbg !27
  %math.result392 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %731) #9, !dbg !27, !frisk.register_lifetimes !1
  %732 = fptrunc <4 x float> %math.result392 to <4 x half>, !dbg !27
  %math.packed393 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %732) #9, !dbg !26, !frisk.register_lifetimes !1
  %local.reduce65 = fadd float %local.reduce61, %730, !dbg !26, !frisk.local_reduction !1
  %733 = extractelement <4 x float> %565, i64 0, !dbg !25
  %734 = call float @llvm.amdgcn.exp2.f32(float %733), !dbg !25
  %735 = insertelement <4 x float> poison, float %734, i64 0, !dbg !27
  %local.reduce66 = fadd float %local.reduce62, %734, !dbg !26, !frisk.local_reduction !1
  %736 = extractelement <4 x float> %565, i64 1, !dbg !25
  %737 = call float @llvm.amdgcn.exp2.f32(float %736), !dbg !25
  %738 = insertelement <4 x float> %735, float %737, i64 1, !dbg !27
  %local.reduce67 = fadd float %local.reduce63, %737, !dbg !26, !frisk.local_reduction !1
  %739 = extractelement <4 x float> %565, i64 2, !dbg !25
  %740 = call float @llvm.amdgcn.exp2.f32(float %739), !dbg !25
  %741 = insertelement <4 x float> %738, float %740, i64 2, !dbg !27
  %local.reduce68 = fadd float %local.reduce64, %740, !dbg !26, !frisk.local_reduction !1
  %742 = extractelement <4 x float> %565, i64 3, !dbg !25
  %743 = call float @llvm.amdgcn.exp2.f32(float %742), !dbg !25
  %744 = insertelement <4 x float> %741, float %743, i64 3, !dbg !27
  %math.result394 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %744) #9, !dbg !27, !frisk.register_lifetimes !1
  %745 = fptrunc <4 x float> %math.result394 to <4 x half>, !dbg !27
  %math.packed395 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %745) #9, !dbg !26, !frisk.register_lifetimes !1
  %local.reduce69 = fadd float %local.reduce65, %743, !dbg !26, !frisk.local_reduction !1
  %746 = extractelement <4 x float> %566, i64 0, !dbg !25
  %747 = call float @llvm.amdgcn.exp2.f32(float %746), !dbg !25
  %748 = insertelement <4 x float> poison, float %747, i64 0, !dbg !27
  %local.reduce70 = fadd float %local.reduce66, %747, !dbg !26, !frisk.local_reduction !1
  %749 = extractelement <4 x float> %566, i64 1, !dbg !25
  %750 = call float @llvm.amdgcn.exp2.f32(float %749), !dbg !25
  %751 = insertelement <4 x float> %748, float %750, i64 1, !dbg !27
  %local.reduce71 = fadd float %local.reduce67, %750, !dbg !26, !frisk.local_reduction !1
  %752 = extractelement <4 x float> %566, i64 2, !dbg !25
  %753 = call float @llvm.amdgcn.exp2.f32(float %752), !dbg !25
  %754 = insertelement <4 x float> %751, float %753, i64 2, !dbg !27
  %local.reduce72 = fadd float %local.reduce68, %753, !dbg !26, !frisk.local_reduction !1
  %755 = extractelement <4 x float> %566, i64 3, !dbg !25
  %756 = call float @llvm.amdgcn.exp2.f32(float %755), !dbg !25
  %757 = insertelement <4 x float> %754, float %756, i64 3, !dbg !27
  %math.result396 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %757) #9, !dbg !27, !frisk.register_lifetimes !1
  %758 = fptrunc <4 x float> %math.result396 to <4 x half>, !dbg !27
  %math.packed397 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %758) #9, !dbg !26, !frisk.register_lifetimes !1
  %local.reduce73 = fadd float %local.reduce69, %756, !dbg !26, !frisk.local_reduction !1
  %759 = extractelement <4 x float> %567, i64 0, !dbg !25
  %760 = call float @llvm.amdgcn.exp2.f32(float %759), !dbg !25
  %761 = insertelement <4 x float> poison, float %760, i64 0, !dbg !27
  %local.reduce74 = fadd float %local.reduce70, %760, !dbg !26, !frisk.local_reduction !1
  %762 = extractelement <4 x float> %567, i64 1, !dbg !25
  %763 = call float @llvm.amdgcn.exp2.f32(float %762), !dbg !25
  %764 = insertelement <4 x float> %761, float %763, i64 1, !dbg !27
  %local.reduce75 = fadd float %local.reduce71, %763, !dbg !26, !frisk.local_reduction !1
  %765 = extractelement <4 x float> %567, i64 2, !dbg !25
  %766 = call float @llvm.amdgcn.exp2.f32(float %765), !dbg !25
  %767 = insertelement <4 x float> %764, float %766, i64 2, !dbg !27
  %local.reduce76 = fadd float %local.reduce72, %766, !dbg !26, !frisk.local_reduction !1
  %768 = extractelement <4 x float> %567, i64 3, !dbg !25
  %769 = call float @llvm.amdgcn.exp2.f32(float %768), !dbg !25
  %770 = insertelement <4 x float> %767, float %769, i64 3, !dbg !27
  %math.result398 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %770) #9, !dbg !27, !frisk.register_lifetimes !1
  %771 = fptrunc <4 x float> %math.result398 to <4 x half>, !dbg !27
  %math.packed399 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %771) #9, !dbg !26, !frisk.register_lifetimes !1
  %local.reduce77 = fadd float %local.reduce73, %769, !dbg !26, !frisk.local_reduction !1
  %772 = extractelement <4 x float> %568, i64 0, !dbg !25
  %773 = call float @llvm.amdgcn.exp2.f32(float %772), !dbg !25
  %774 = insertelement <4 x float> poison, float %773, i64 0, !dbg !27
  %local.reduce78 = fadd float %local.reduce74, %773, !dbg !26, !frisk.local_reduction !1
  %775 = extractelement <4 x float> %568, i64 1, !dbg !25
  %776 = call float @llvm.amdgcn.exp2.f32(float %775), !dbg !25
  %777 = insertelement <4 x float> %774, float %776, i64 1, !dbg !27
  %local.reduce79 = fadd float %local.reduce75, %776, !dbg !26, !frisk.local_reduction !1
  %778 = extractelement <4 x float> %568, i64 2, !dbg !25
  %779 = call float @llvm.amdgcn.exp2.f32(float %778), !dbg !25
  %780 = insertelement <4 x float> %777, float %779, i64 2, !dbg !27
  %local.reduce80 = fadd float %local.reduce76, %779, !dbg !26, !frisk.local_reduction !1
  %781 = extractelement <4 x float> %568, i64 3, !dbg !25
  %782 = call float @llvm.amdgcn.exp2.f32(float %781), !dbg !25
  %783 = insertelement <4 x float> %780, float %782, i64 3, !dbg !27
  %math.result400 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %783) #9, !dbg !27, !frisk.register_lifetimes !1
  %784 = fptrunc <4 x float> %math.result400 to <4 x half>, !dbg !27
  %math.packed401 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %784) #9, !dbg !26, !frisk.register_lifetimes !1
  %local.reduce81 = fadd float %local.reduce77, %782, !dbg !26, !frisk.local_reduction !1
  %785 = extractelement <4 x float> %569, i64 0, !dbg !25
  %786 = call float @llvm.amdgcn.exp2.f32(float %785), !dbg !25
  %787 = insertelement <4 x float> poison, float %786, i64 0, !dbg !27
  %local.reduce82 = fadd float %local.reduce78, %786, !dbg !26, !frisk.local_reduction !1
  %788 = extractelement <4 x float> %569, i64 1, !dbg !25
  %789 = call float @llvm.amdgcn.exp2.f32(float %788), !dbg !25
  %790 = insertelement <4 x float> %787, float %789, i64 1, !dbg !27
  %local.reduce83 = fadd float %local.reduce79, %789, !dbg !26, !frisk.local_reduction !1
  %local.reduce86 = fadd float %local.reduce82, %local.reduce83, !dbg !26, !frisk.local_reduction !1
  %791 = extractelement <4 x float> %569, i64 2, !dbg !25
  %792 = call float @llvm.amdgcn.exp2.f32(float %791), !dbg !25
  %793 = insertelement <4 x float> %790, float %792, i64 2, !dbg !27
  %local.reduce84 = fadd float %local.reduce80, %792, !dbg !26, !frisk.local_reduction !1
  %794 = extractelement <4 x float> %569, i64 3, !dbg !25
  %795 = call float @llvm.amdgcn.exp2.f32(float %794), !dbg !25
  %796 = insertelement <4 x float> %793, float %795, i64 3, !dbg !27
  %math.result402 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %796) #9, !dbg !27, !frisk.register_lifetimes !1
  %797 = fptrunc <4 x float> %math.result402 to <4 x half>, !dbg !27
  %math.packed403 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %797) #9, !dbg !26, !frisk.register_lifetimes !1
  %local.reduce85 = fadd float %local.reduce81, %795, !dbg !26, !frisk.local_reduction !1
  %local.reduce87 = fadd float %local.reduce84, %local.reduce85, !dbg !26, !frisk.local_reduction !1
  %798 = fadd float %local.reduce86, %local.reduce87, !dbg !26, !frisk.local_reduction !1
  %799 = fadd float %798, %585, !dbg !26
  %800 = insertelement <1 x float> poison, float %799, i64 0, !dbg !7
  %801 = insertvalue [2 x <1 x float>] %693, <1 x float> %800, 1, !dbg !7
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %vdb.address = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local504), !dbg !15
  %vdb.address1435 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %vdb.address), !dbg !15
  %vdb.address1436 = or disjoint i64 %143, %address.local921, !dbg !15
  %vdb.address1437 = trunc nuw i64 %vdb.address1436 to i32, !dbg !15
  %vdb.address1438 = shl nuw nsw i32 %vdb.address1437, 1, !dbg !15
  %vdb.address1439 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %vdb.address1440 = insertelement <2 x i64> poison, i64 %vdb.address1439, i64 0, !dbg !15
  %vdb.address1441 = bitcast <2 x i64> %vdb.address1440 to <4 x i32>, !dbg !15
  %vdb.address1442 = shufflevector <4 x i32> %vdb.address1441, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %vdb.address1443 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local521), !dbg !15
  %vdb.address1444 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %vdb.address1443), !dbg !15
  %vdb.address1445 = or disjoint i64 %143, %address.local927, !dbg !15
  %vdb.address1446 = trunc nuw i64 %vdb.address1445 to i32, !dbg !15
  %vdb.address1447 = shl nuw nsw i32 %vdb.address1446, 1, !dbg !15
  %vdb.address1448 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %vdb.address1449 = insertelement <2 x i64> poison, i64 %vdb.address1448, i64 0, !dbg !15
  %vdb.address1450 = bitcast <2 x i64> %vdb.address1449 to <4 x i32>, !dbg !15
  %vdb.address1451 = shufflevector <4 x i32> %vdb.address1450, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %vdb.dma.base = add i32 %vdb.address1435, 8192, !dbg !13
  %802 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1438, <4 x i32> %vdb.address1442) #10, !dbg !15
  %vdb.dma.base1452 = add i32 %vdb.address1444, 8192, !dbg !13
  %803 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1452, i32 %vdb.address1447, <4 x i32> %vdb.address1451) #10, !dbg !15
  %address.local929 = shl nuw nsw i32 %address.thread498, 3, !dbg !13
  %address.local930 = and i32 %address.local929, 120, !dbg !13
  %address.local931 = shl nuw nsw i32 %address.thread498, 5, !dbg !13
  %address.local932 = and i32 %address.local931, 1536, !dbg !13
  %address.local933 = or disjoint i32 %address.local930, %address.local932, !dbg !13
  %address.local934 = zext nneg i32 %address.local933 to i64, !dbg !13
  %address.local935 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local934, !dbg !13
  %804 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local935, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %805 = bitcast <4 x i32> %804 to <8 x half>, !dbg !13
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
  %806 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local945, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %807 = bitcast <4 x i32> %806 to <8 x half>, !dbg !13
  %address.local946 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local938, !dbg !13
  %address.local947 = getelementptr [2 x i8], ptr addrspace(3) %address.local946, i64 %address.local941, !dbg !13
  %address.local948 = getelementptr [2 x i8], ptr addrspace(3) %address.local947, i64 %address.local943, !dbg !13
  %address.local949 = getelementptr i8, ptr addrspace(3) %address.local948, i64 512, !dbg !13
  %808 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local949, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %809 = bitcast <4 x i32> %808 to <8 x half>, !dbg !13
  %address.local950 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local938, !dbg !13
  %address.local951 = getelementptr [2 x i8], ptr addrspace(3) %address.local950, i64 %address.local941, !dbg !13
  %address.local952 = getelementptr [2 x i8], ptr addrspace(3) %address.local951, i64 %address.local943, !dbg !13
  %address.local953 = getelementptr i8, ptr addrspace(3) %address.local952, i64 768, !dbg !13
  %810 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local953, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %811 = bitcast <4 x i32> %810 to <8 x half>, !dbg !13
  %address.local954 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local934, !dbg !13
  %address.local955 = getelementptr i8, ptr addrspace(3) %address.local954, i64 4096, !dbg !13
  %812 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local955, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %813 = bitcast <4 x i32> %812 to <8 x half>, !dbg !13
  %address.local956 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local938, !dbg !13
  %address.local957 = getelementptr [2 x i8], ptr addrspace(3) %address.local956, i64 %address.local941, !dbg !13
  %address.local958 = getelementptr [2 x i8], ptr addrspace(3) %address.local957, i64 %address.local943, !dbg !13
  %address.local959 = getelementptr i8, ptr addrspace(3) %address.local958, i64 4352, !dbg !13
  %814 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local959, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %815 = bitcast <4 x i32> %814 to <8 x half>, !dbg !13
  %address.local960 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local938, !dbg !13
  %address.local961 = getelementptr [2 x i8], ptr addrspace(3) %address.local960, i64 %address.local941, !dbg !13
  %address.local962 = getelementptr [2 x i8], ptr addrspace(3) %address.local961, i64 %address.local943, !dbg !13
  %address.local963 = getelementptr i8, ptr addrspace(3) %address.local962, i64 4608, !dbg !13
  %816 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local963, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %817 = bitcast <4 x i32> %816 to <8 x half>, !dbg !13
  %address.local964 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local938, !dbg !13
  %address.local965 = getelementptr [2 x i8], ptr addrspace(3) %address.local964, i64 %address.local941, !dbg !13
  %address.local966 = getelementptr [2 x i8], ptr addrspace(3) %address.local965, i64 %address.local943, !dbg !13
  %address.local967 = getelementptr i8, ptr addrspace(3) %address.local966, i64 4864, !dbg !13
  %818 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local967, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %819 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local504), !dbg !15
  %dma.buffer.lds1344 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %819), !dbg !15
  %.idx14231424 = or disjoint i64 %143, %address.local921, !dbg !15
  %.idx14231424.tr = trunc nuw i64 %.idx14231424 to i32, !dbg !15
  %dma.buffer.offset1342 = shl nuw nsw i32 %.idx14231424.tr, 1, !dbg !15
  %820 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %821 = insertelement <2 x i64> poison, i64 %820, i64 0, !dbg !15
  %822 = bitcast <2 x i64> %821 to <4 x i32>, !dbg !15
  %dma.buffer.rsrc1343 = shufflevector <4 x i32> %822, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %823 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local521), !dbg !15
  %dma.buffer.lds1341 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %823), !dbg !15
  %.idx14251426 = or disjoint i64 %143, %address.local927, !dbg !15
  %.idx14251426.tr = trunc nuw i64 %.idx14251426 to i32, !dbg !15
  %dma.buffer.offset1339 = shl nuw nsw i32 %.idx14251426.tr, 1, !dbg !15
  %824 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %825 = insertelement <2 x i64> poison, i64 %824, i64 0, !dbg !15
  %826 = bitcast <2 x i64> %825 to <4 x i32>, !dbg !15
  %dma.buffer.rsrc1340 = shufflevector <4 x i32> %826, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %827 = shufflevector <8 x half> %805, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %828 = extractvalue [2 x <8 x float>] %121, 0, !dbg !28
  %829 = shufflevector <8 x float> %828, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %830 = fmul <4 x float> %829, %574, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %831 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %827, <4 x float> %830) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %832 = shufflevector <8 x half> %805, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %833 = shufflevector <8 x float> %828, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %834 = fmul <4 x float> %833, %574, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %835 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %832, <4 x float> %834) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %836 = extractvalue [2 x <8 x float>] %121, 1, !dbg !28
  %837 = shufflevector <8 x float> %836, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %838 = fmul <4 x float> %837, %582, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %839 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %827, <4 x float> %838) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %840 = shufflevector <8 x float> %836, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %841 = fmul <4 x float> %840, %582, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %842 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %832, <4 x float> %841) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %843 = shufflevector <8 x half> %807, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %844 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %843, <4 x float> %831) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %845 = shufflevector <8 x half> %807, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %846 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %845, <4 x float> %835) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %847 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %843, <4 x float> %839) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %848 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %845, <4 x float> %842) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %849 = shufflevector <8 x half> %809, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %850 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %849, <4 x float> %844) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %851 = shufflevector <8 x half> %809, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %852 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %851, <4 x float> %846) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %853 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %849, <4 x float> %847) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %854 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %851, <4 x float> %848) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %855 = shufflevector <8 x half> %811, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %856 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %855, <4 x float> %850) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %857 = shufflevector <8 x half> %811, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %858 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %857, <4 x float> %852) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %859 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %855, <4 x float> %853) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %860 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %857, <4 x float> %854) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %861 = shufflevector <8 x half> %813, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %862 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %861, <4 x float> %856) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %863 = shufflevector <8 x half> %813, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %864 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %863, <4 x float> %858) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %865 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %861, <4 x float> %859) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %866 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %863, <4 x float> %860) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %867 = shufflevector <8 x half> %815, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %868 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %867, <4 x float> %862) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %869 = shufflevector <8 x half> %815, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %870 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %869, <4 x float> %864) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %871 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %867, <4 x float> %865) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %872 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %869, <4 x float> %866) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %873 = shufflevector <8 x half> %817, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %874 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %873, <4 x float> %868) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %875 = shufflevector <8 x half> %817, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %876 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %875, <4 x float> %870) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %877 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %873, <4 x float> %871) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %878 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %875, <4 x float> %872) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %879 = bitcast <4 x i32> %818 to <8 x half>, !dbg !13
  %880 = shufflevector <8 x half> %879, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %881 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %880, <4 x float> %874) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %882 = shufflevector <8 x half> %879, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %883 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %882, <4 x float> %876) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %884 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %880, <4 x float> %877) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %885 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %882, <4 x float> %878) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %vdb.address1460 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local504), !dbg !15
  %vdb.address1461 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %vdb.address1460), !dbg !15
  %vdb.address1462 = or disjoint i64 %226, %address.local921, !dbg !15
  %vdb.address1463 = trunc nuw i64 %vdb.address1462 to i32, !dbg !15
  %vdb.address1464 = shl nuw nsw i32 %vdb.address1463, 1, !dbg !15
  %vdb.address1465 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %vdb.address1466 = insertelement <2 x i64> poison, i64 %vdb.address1465, i64 0, !dbg !15
  %vdb.address1467 = bitcast <2 x i64> %vdb.address1466 to <4 x i32>, !dbg !15
  %vdb.address1468 = shufflevector <4 x i32> %vdb.address1467, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %vdb.address1469 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local521), !dbg !15
  %vdb.address1470 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %vdb.address1469), !dbg !15
  %vdb.address1471 = or disjoint i64 %226, %address.local927, !dbg !15
  %vdb.address1472 = trunc nuw i64 %vdb.address1471 to i32, !dbg !15
  %vdb.address1473 = shl nuw nsw i32 %vdb.address1472, 1, !dbg !15
  %vdb.address1474 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %vdb.address1475 = insertelement <2 x i64> poison, i64 %vdb.address1474, i64 0, !dbg !15
  %vdb.address1476 = bitcast <2 x i64> %vdb.address1475 to <4 x i32>, !dbg !15
  %vdb.address1477 = shufflevector <4 x i32> %vdb.address1476, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %886 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.address1461, i32 %vdb.address1464, <4 x i32> %vdb.address1468) #10, !dbg !15
  %887 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.address1470, i32 %vdb.address1473, <4 x i32> %vdb.address1477) #10, !dbg !15
  %vdb.read = getelementptr i8, ptr addrspace(3) %address.local935, i64 8192, !dbg !13
  %888 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %889 = bitcast <4 x i32> %888 to <8 x half>, !dbg !13
  %vdb.read1453 = getelementptr i8, ptr addrspace(3) %address.local945, i64 8192, !dbg !13
  %890 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1453, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %891 = bitcast <4 x i32> %890 to <8 x half>, !dbg !13
  %vdb.read1454 = getelementptr i8, ptr addrspace(3) %address.local949, i64 8192, !dbg !13
  %892 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1454, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %893 = bitcast <4 x i32> %892 to <8 x half>, !dbg !13
  %vdb.read1455 = getelementptr i8, ptr addrspace(3) %address.local953, i64 8192, !dbg !13
  %894 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1455, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %895 = bitcast <4 x i32> %894 to <8 x half>, !dbg !13
  %vdb.read1456 = getelementptr i8, ptr addrspace(3) %address.local955, i64 8192, !dbg !13
  %896 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1456, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %897 = bitcast <4 x i32> %896 to <8 x half>, !dbg !13
  %vdb.read1457 = getelementptr i8, ptr addrspace(3) %address.local959, i64 8192, !dbg !13
  %898 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1457, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %899 = bitcast <4 x i32> %898 to <8 x half>, !dbg !13
  %vdb.read1458 = getelementptr i8, ptr addrspace(3) %address.local963, i64 8192, !dbg !13
  %900 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1458, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %901 = bitcast <4 x i32> %900 to <8 x half>, !dbg !13
  %vdb.read1459 = getelementptr i8, ptr addrspace(3) %address.local967, i64 8192, !dbg !13
  %902 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1459, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %903 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local504), !dbg !15
  %dma.buffer.lds1338 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %903), !dbg !15
  %.idx14271428 = or disjoint i64 %226, %address.local921, !dbg !15
  %.idx14271428.tr = trunc nuw i64 %.idx14271428 to i32, !dbg !15
  %dma.buffer.offset1336 = shl nuw nsw i32 %.idx14271428.tr, 1, !dbg !15
  %904 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %905 = insertelement <2 x i64> poison, i64 %904, i64 0, !dbg !15
  %906 = bitcast <2 x i64> %905 to <4 x i32>, !dbg !15
  %dma.buffer.rsrc1337 = shufflevector <4 x i32> %906, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %907 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local521), !dbg !15
  %dma.buffer.lds1335 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %907), !dbg !15
  %.idx14291430 = or disjoint i64 %226, %address.local927, !dbg !15
  %.idx14291430.tr = trunc nuw i64 %.idx14291430 to i32, !dbg !15
  %dma.buffer.offset1333 = shl nuw nsw i32 %.idx14291430.tr, 1, !dbg !15
  %908 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %909 = insertelement <2 x i64> poison, i64 %908, i64 0, !dbg !15
  %910 = bitcast <2 x i64> %909 to <4 x i32>, !dbg !15
  %dma.buffer.rsrc1334 = shufflevector <4 x i32> %910, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %911 = shufflevector <8 x half> %889, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %912 = extractvalue [2 x <8 x float>] %124, 0, !dbg !28
  %913 = shufflevector <8 x float> %912, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %914 = fmul <4 x float> %913, %574, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %915 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %911, <4 x float> %914) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %916 = shufflevector <8 x half> %889, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %917 = shufflevector <8 x float> %912, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %918 = fmul <4 x float> %917, %574, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %919 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %916, <4 x float> %918) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %920 = extractvalue [2 x <8 x float>] %124, 1, !dbg !28
  %921 = shufflevector <8 x float> %920, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %922 = fmul <4 x float> %921, %582, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %923 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %911, <4 x float> %922) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %924 = shufflevector <8 x float> %920, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %925 = fmul <4 x float> %924, %582, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %926 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %916, <4 x float> %925) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %927 = shufflevector <8 x half> %891, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %928 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %927, <4 x float> %915) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %929 = shufflevector <8 x half> %891, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %930 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %929, <4 x float> %919) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %931 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %927, <4 x float> %923) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %932 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %929, <4 x float> %926) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %933 = shufflevector <8 x half> %893, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %934 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %933, <4 x float> %928) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %935 = shufflevector <8 x half> %893, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %936 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %935, <4 x float> %930) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %937 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %933, <4 x float> %931) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %938 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %935, <4 x float> %932) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %939 = shufflevector <8 x half> %895, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %940 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %939, <4 x float> %934) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %941 = shufflevector <8 x half> %895, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %942 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %941, <4 x float> %936) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %943 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %939, <4 x float> %937) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %944 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %941, <4 x float> %938) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %945 = shufflevector <8 x half> %897, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %946 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %945, <4 x float> %940) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %947 = shufflevector <8 x half> %897, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %948 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %947, <4 x float> %942) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %949 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %945, <4 x float> %943) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %950 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %947, <4 x float> %944) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %951 = shufflevector <8 x half> %899, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %952 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %951, <4 x float> %946) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %953 = shufflevector <8 x half> %899, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %954 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %953, <4 x float> %948) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %955 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %951, <4 x float> %949) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %956 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %953, <4 x float> %950) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %957 = shufflevector <8 x half> %901, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %958 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %957, <4 x float> %952) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %959 = shufflevector <8 x half> %901, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %960 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %959, <4 x float> %954) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %961 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %957, <4 x float> %955) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %962 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %959, <4 x float> %956) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %963 = bitcast <4 x i32> %902 to <8 x half>, !dbg !13
  %964 = shufflevector <8 x half> %963, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %965 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %964, <4 x float> %958) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %966 = shufflevector <8 x half> %963, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %967 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %966, <4 x float> %960) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %968 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %964, <4 x float> %961) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %969 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %966, <4 x float> %962) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %vdb.address1478 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local504), !dbg !15
  %vdb.address1479 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %vdb.address1478), !dbg !15
  %vdb.address1480 = or disjoint i64 %301, %address.local921, !dbg !15
  %vdb.address1481 = trunc nuw i64 %vdb.address1480 to i32, !dbg !15
  %vdb.address1482 = shl nuw nsw i32 %vdb.address1481, 1, !dbg !15
  %vdb.address1483 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %vdb.address1484 = insertelement <2 x i64> poison, i64 %vdb.address1483, i64 0, !dbg !15
  %vdb.address1485 = bitcast <2 x i64> %vdb.address1484 to <4 x i32>, !dbg !15
  %vdb.address1486 = shufflevector <4 x i32> %vdb.address1485, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %vdb.address1487 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local521), !dbg !15
  %vdb.address1488 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %vdb.address1487), !dbg !15
  %vdb.address1489 = or disjoint i64 %301, %address.local927, !dbg !15
  %vdb.address1490 = trunc nuw i64 %vdb.address1489 to i32, !dbg !15
  %vdb.address1491 = shl nuw nsw i32 %vdb.address1490, 1, !dbg !15
  %vdb.address1492 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %vdb.address1493 = insertelement <2 x i64> poison, i64 %vdb.address1492, i64 0, !dbg !15
  %vdb.address1494 = bitcast <2 x i64> %vdb.address1493 to <4 x i32>, !dbg !15
  %vdb.address1495 = shufflevector <4 x i32> %vdb.address1494, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %vdb.dma.base1496 = add i32 %vdb.address1479, 8192, !dbg !13
  %970 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1496, i32 %vdb.address1482, <4 x i32> %vdb.address1486) #10, !dbg !15
  %vdb.dma.base1497 = add i32 %vdb.address1488, 8192, !dbg !13
  %971 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1497, i32 %vdb.address1491, <4 x i32> %vdb.address1495) #10, !dbg !15
  %972 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local935, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %973 = bitcast <4 x i32> %972 to <8 x half>, !dbg !13
  %974 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local945, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %975 = bitcast <4 x i32> %974 to <8 x half>, !dbg !13
  %976 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local949, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %977 = bitcast <4 x i32> %976 to <8 x half>, !dbg !13
  %978 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local953, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %979 = bitcast <4 x i32> %978 to <8 x half>, !dbg !13
  %980 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local955, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %981 = bitcast <4 x i32> %980 to <8 x half>, !dbg !13
  %982 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local959, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %983 = bitcast <4 x i32> %982 to <8 x half>, !dbg !13
  %984 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local963, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %985 = bitcast <4 x i32> %984 to <8 x half>, !dbg !13
  %986 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local967, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %987 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local504), !dbg !15
  %dma.buffer.lds1332 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %987), !dbg !15
  %.idx14311432 = or disjoint i64 %301, %address.local921, !dbg !15
  %.idx14311432.tr = trunc nuw i64 %.idx14311432 to i32, !dbg !15
  %dma.buffer.offset1330 = shl nuw nsw i32 %.idx14311432.tr, 1, !dbg !15
  %988 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %989 = insertelement <2 x i64> poison, i64 %988, i64 0, !dbg !15
  %990 = bitcast <2 x i64> %989 to <4 x i32>, !dbg !15
  %dma.buffer.rsrc1331 = shufflevector <4 x i32> %990, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %991 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local521), !dbg !15
  %dma.buffer.lds = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %991), !dbg !15
  %.idx14331434 = or disjoint i64 %301, %address.local927, !dbg !15
  %.idx14331434.tr = trunc nuw i64 %.idx14331434 to i32, !dbg !15
  %dma.buffer.offset = shl nuw nsw i32 %.idx14331434.tr, 1, !dbg !15
  %992 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %993 = insertelement <2 x i64> poison, i64 %992, i64 0, !dbg !15
  %994 = bitcast <2 x i64> %993 to <4 x i32>, !dbg !15
  %dma.buffer.rsrc = shufflevector <4 x i32> %994, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %995 = shufflevector <8 x half> %973, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %996 = extractvalue [2 x <8 x float>] %125, 0, !dbg !28
  %997 = shufflevector <8 x float> %996, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %998 = fmul <4 x float> %997, %574, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %999 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %995, <4 x float> %998) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1000 = shufflevector <8 x half> %973, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1001 = shufflevector <8 x float> %996, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %1002 = fmul <4 x float> %1001, %574, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1003 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1000, <4 x float> %1002) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %1004 = extractvalue [2 x <8 x float>] %125, 1, !dbg !28
  %1005 = shufflevector <8 x float> %1004, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %1006 = fmul <4 x float> %1005, %582, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1007 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %995, <4 x float> %1006) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %1008 = shufflevector <8 x float> %1004, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %1009 = fmul <4 x float> %1008, %582, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1010 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %1000, <4 x float> %1009) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1011 = shufflevector <8 x half> %975, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1012 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %1011, <4 x float> %999) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1013 = shufflevector <8 x half> %975, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1014 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %1013, <4 x float> %1003) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1015 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %1011, <4 x float> %1007) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1016 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %1013, <4 x float> %1010) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1017 = shufflevector <8 x half> %977, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1018 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %1017, <4 x float> %1012) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1019 = shufflevector <8 x half> %977, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1020 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %1019, <4 x float> %1014) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1021 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %1017, <4 x float> %1015) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1022 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %1019, <4 x float> %1016) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1023 = shufflevector <8 x half> %979, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1024 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %1023, <4 x float> %1018) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1025 = shufflevector <8 x half> %979, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1026 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %1025, <4 x float> %1020) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1027 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %1023, <4 x float> %1021) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1028 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %1025, <4 x float> %1022) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1029 = shufflevector <8 x half> %981, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1030 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %1029, <4 x float> %1024) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1031 = shufflevector <8 x half> %981, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1032 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %1031, <4 x float> %1026) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1033 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %1029, <4 x float> %1027) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1034 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %1031, <4 x float> %1028) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1035 = shufflevector <8 x half> %983, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1036 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %1035, <4 x float> %1030) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1037 = shufflevector <8 x half> %983, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1038 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %1037, <4 x float> %1032) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1039 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %1035, <4 x float> %1033) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1040 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %1037, <4 x float> %1034) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1041 = shufflevector <8 x half> %985, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1042 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %1041, <4 x float> %1036) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1043 = shufflevector <8 x half> %985, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1044 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %1043, <4 x float> %1038) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1045 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %1041, <4 x float> %1039) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1046 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %1043, <4 x float> %1040) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1047 = bitcast <4 x i32> %986 to <8 x half>, !dbg !13
  %1048 = shufflevector <8 x half> %1047, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1049 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %1048, <4 x float> %1042) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1050 = shufflevector <8 x half> %1047, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1051 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %1050, <4 x float> %1044) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1052 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %1048, <4 x float> %1045) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1053 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %1050, <4 x float> %1046) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %vdb.read1498 = getelementptr i8, ptr addrspace(3) %address.local935, i64 8192, !dbg !13
  %1054 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1498, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1055 = bitcast <4 x i32> %1054 to <8 x half>, !dbg !13
  %vdb.read1499 = getelementptr i8, ptr addrspace(3) %address.local945, i64 8192, !dbg !13
  %1056 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1499, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1057 = bitcast <4 x i32> %1056 to <8 x half>, !dbg !13
  %vdb.read1500 = getelementptr i8, ptr addrspace(3) %address.local949, i64 8192, !dbg !13
  %1058 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1500, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1059 = bitcast <4 x i32> %1058 to <8 x half>, !dbg !13
  %vdb.read1501 = getelementptr i8, ptr addrspace(3) %address.local953, i64 8192, !dbg !13
  %1060 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1501, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1061 = bitcast <4 x i32> %1060 to <8 x half>, !dbg !13
  %vdb.read1502 = getelementptr i8, ptr addrspace(3) %address.local955, i64 8192, !dbg !13
  %1062 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1502, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1063 = bitcast <4 x i32> %1062 to <8 x half>, !dbg !13
  %vdb.read1503 = getelementptr i8, ptr addrspace(3) %address.local959, i64 8192, !dbg !13
  %1064 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1503, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1065 = bitcast <4 x i32> %1064 to <8 x half>, !dbg !13
  %vdb.read1504 = getelementptr i8, ptr addrspace(3) %address.local963, i64 8192, !dbg !13
  %1066 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1504, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1067 = bitcast <4 x i32> %1066 to <8 x half>, !dbg !13
  %vdb.read1505 = getelementptr i8, ptr addrspace(3) %address.local967, i64 8192, !dbg !13
  %1068 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1505, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %1069 = shufflevector <8 x half> %1055, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1070 = extractvalue [2 x <8 x float>] %126, 0, !dbg !28
  %1071 = shufflevector <8 x float> %1070, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %1072 = fmul <4 x float> %1071, %574, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1073 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1069, <4 x float> %1072) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1074 = shufflevector <8 x half> %1055, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1075 = shufflevector <8 x float> %1070, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %1076 = fmul <4 x float> %1075, %574, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1077 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1074, <4 x float> %1076) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %1078 = extractvalue [2 x <8 x float>] %126, 1, !dbg !28
  %1079 = shufflevector <8 x float> %1078, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %1080 = fmul <4 x float> %1079, %582, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1081 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %1069, <4 x float> %1080) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %1082 = shufflevector <8 x float> %1078, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %1083 = fmul <4 x float> %1082, %582, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1084 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %1074, <4 x float> %1083) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1085 = shufflevector <8 x half> %1057, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1086 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %1085, <4 x float> %1073) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1087 = shufflevector <8 x half> %1057, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1088 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %1087, <4 x float> %1077) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1089 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %1085, <4 x float> %1081) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1090 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %1087, <4 x float> %1084) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1091 = shufflevector <8 x half> %1059, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1092 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %1091, <4 x float> %1086) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1093 = shufflevector <8 x half> %1059, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1094 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %1093, <4 x float> %1088) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1095 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %1091, <4 x float> %1089) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1096 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %1093, <4 x float> %1090) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1097 = shufflevector <8 x half> %1061, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1098 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %1097, <4 x float> %1092) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1099 = shufflevector <8 x half> %1061, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1100 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %1099, <4 x float> %1094) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1101 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %1097, <4 x float> %1095) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1102 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %1099, <4 x float> %1096) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1103 = shufflevector <8 x half> %1063, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1104 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %1103, <4 x float> %1098) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1105 = shufflevector <8 x half> %1063, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1106 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %1105, <4 x float> %1100) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1107 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %1103, <4 x float> %1101) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1108 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %1105, <4 x float> %1102) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1109 = shufflevector <8 x half> %1065, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1110 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %1109, <4 x float> %1104) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1111 = shufflevector <8 x half> %1065, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1112 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %1111, <4 x float> %1106) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1113 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %1109, <4 x float> %1107) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1114 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %1111, <4 x float> %1108) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1115 = shufflevector <8 x half> %1067, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1116 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %1115, <4 x float> %1110) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1117 = shufflevector <8 x half> %1067, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1118 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %1117, <4 x float> %1112) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1119 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %1115, <4 x float> %1113) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1120 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %1117, <4 x float> %1114) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1121 = bitcast <4 x i32> %1068 to <8 x half>, !dbg !13
  %1122 = shufflevector <8 x half> %1121, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1123 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %1122, <4 x float> %1116) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1124 = shufflevector <8 x half> %1121, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1125 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %1124, <4 x float> %1118) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1126 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %1122, <4 x float> %1119) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1127 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %1124, <4 x float> %1120) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !7
  %1128 = shufflevector <4 x float> %881, <4 x float> %883, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1129 = insertvalue [2 x <8 x float>] poison, <8 x float> %1128, 0, !dbg !7
  %1130 = shufflevector <4 x float> %884, <4 x float> %885, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1131 = insertvalue [2 x <8 x float>] %1129, <8 x float> %1130, 1, !dbg !7
  %1132 = insertvalue [2 x <1 x float>] poison, <1 x float> %538, 0, !dbg !7
  %1133 = insertvalue [2 x <1 x float>] %1132, <1 x float> %551, 1, !dbg !7
  %1134 = shufflevector <4 x float> %965, <4 x float> %967, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1135 = insertvalue [2 x <8 x float>] poison, <8 x float> %1134, 0, !dbg !7
  %1136 = shufflevector <4 x float> %968, <4 x float> %969, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1137 = insertvalue [2 x <8 x float>] %1135, <8 x float> %1136, 1, !dbg !7
  %1138 = shufflevector <4 x float> %1049, <4 x float> %1051, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1139 = insertvalue [2 x <8 x float>] poison, <8 x float> %1138, 0, !dbg !7
  %1140 = shufflevector <4 x float> %1052, <4 x float> %1053, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1141 = insertvalue [2 x <8 x float>] %1139, <8 x float> %1140, 1, !dbg !7
  %1142 = shufflevector <4 x float> %1123, <4 x float> %1125, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1143 = insertvalue [2 x <8 x float>] poison, <8 x float> %1142, 0, !dbg !7
  %1144 = shufflevector <4 x float> %1126, <4 x float> %1127, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1145 = insertvalue [2 x <8 x float>] %1143, <8 x float> %1144, 1, !dbg !7
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %1146 = add nuw nsw i64 %120, 1, !dbg !7
  br label %119, !dbg !7

1147:                                             ; preds = %119
  %lane.sum.field184 = extractvalue [2 x <1 x float>] %122, 1, !dbg !7
  %lane.sum.local185 = extractelement <1 x float> %lane.sum.field184, i64 0, !dbg !7
  %lane.sum.lane.lo186 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !7
  %lane.sum.lane187 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %lane.sum.lane.lo186), !dbg !7
  %1148 = shl nuw nsw i32 %lane.sum.lane187, 2, !dbg !7
  %1149 = xor i32 %1148, 64, !dbg !7
  %1150 = bitcast float %lane.sum.local185 to i32, !dbg !7
  %lane.sum.shuffle188 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1149, i32 %1150), !dbg !7
  %1151 = bitcast i32 %lane.sum.shuffle188 to float, !dbg !7
  %lane.sum.final189 = fadd float %lane.sum.local185, %1151, !dbg !7
  %1152 = shl nuw nsw i32 %lane.sum.lane187, 2, !dbg !7
  %1153 = xor i32 %1152, 128, !dbg !7
  %1154 = bitcast float %lane.sum.final189 to i32, !dbg !7
  %lane.sum.shuffle190 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1153, i32 %1154), !dbg !7
  %1155 = bitcast i32 %lane.sum.shuffle190 to float, !dbg !7
  %lane.sum.final191 = fadd float %lane.sum.final189, %1155, !dbg !7
  %1156 = insertelement <1 x float> poison, float %lane.sum.final191, i64 0, !dbg !7
  %lane.sum.field = extractvalue [2 x <1 x float>] %122, 0, !dbg !7
  %lane.sum.local = extractelement <1 x float> %lane.sum.field, i64 0, !dbg !7
  %lane.sum.lane.lo = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !7
  %lane.sum.lane = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %lane.sum.lane.lo), !dbg !7
  %1157 = shl nuw nsw i32 %lane.sum.lane, 2, !dbg !7
  %1158 = xor i32 %1157, 64, !dbg !7
  %1159 = bitcast float %lane.sum.local to i32, !dbg !7
  %lane.sum.shuffle = call i32 @llvm.amdgcn.ds.bpermute(i32 %1158, i32 %1159), !dbg !7
  %1160 = bitcast i32 %lane.sum.shuffle to float, !dbg !7
  %lane.sum.final = fadd float %lane.sum.local, %1160, !dbg !7
  %1161 = shl nuw nsw i32 %lane.sum.lane, 2, !dbg !7
  %1162 = xor i32 %1161, 128, !dbg !7
  %1163 = bitcast float %lane.sum.final to i32, !dbg !7
  %lane.sum.shuffle182 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1162, i32 %1163), !dbg !7
  %1164 = bitcast i32 %lane.sum.shuffle182 to float, !dbg !7
  %lane.sum.final183 = fadd float %lane.sum.final, %1164, !dbg !7
  %1165 = insertelement <1 x float> poison, float %lane.sum.final183, i64 0, !dbg !7
  %1166 = and i32 %25, 16, !dbg !7
  %.not = icmp eq i32 %1166, 0, !dbg !7
  %1167 = and i32 %25, 32, !dbg !7
  %.not36 = icmp eq i32 %1167, 0, !dbg !7
  %1168 = extractvalue [2 x <8 x float>] %126, 0, !dbg !7
  %1169 = extractelement <8 x float> %1168, i64 0, !dbg !7
  %1170 = extractelement <8 x float> %1168, i64 1, !dbg !7
  %1171 = extractelement <8 x float> %1168, i64 2, !dbg !7
  %1172 = extractelement <8 x float> %1168, i64 3, !dbg !7
  %1173 = select i1 %.not, float %1170, float %1169, !dbg !7
  %1174 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !7
  %1175 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1174), !dbg !7
  %1176 = add nuw nsw i32 %1175, 64, !dbg !7
  %1177 = xor i32 %1175, 16, !dbg !7
  %1178 = and i32 %1176, 192, !dbg !7
  %1179 = icmp samesign ult i32 %1177, %1178, !dbg !7
  %1180 = select i1 %1179, i32 %1177, i32 %1175, !dbg !7
  %1181 = shl nuw nsw i32 %1180, 2, !dbg !7
  %1182 = bitcast float %1173 to i32, !dbg !7
  %1183 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1181, i32 %1182), !dbg !7
  %1184 = bitcast i32 %1183 to float, !dbg !7
  %1185 = select i1 %.not, float %1169, float %1184, !dbg !7
  %1186 = select i1 %.not, float %1184, float %1170, !dbg !7
  %1187 = select i1 %.not, float %1172, float %1171, !dbg !7
  %1188 = bitcast float %1187 to i32, !dbg !7
  %1189 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1181, i32 %1188), !dbg !7
  %1190 = bitcast i32 %1189 to float, !dbg !7
  %1191 = select i1 %.not, float %1171, float %1190, !dbg !7
  %1192 = select i1 %.not, float %1190, float %1172, !dbg !7
  %1193 = select i1 %.not36, float %1191, float %1185, !dbg !7
  %1194 = xor i32 %1175, 32, !dbg !7
  %1195 = icmp samesign ult i32 %1194, %1178, !dbg !7
  %1196 = select i1 %1195, i32 %1194, i32 %1175, !dbg !7
  %1197 = shl nuw nsw i32 %1196, 2, !dbg !7
  %1198 = bitcast float %1193 to i32, !dbg !7
  %1199 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1197, i32 %1198), !dbg !7
  %1200 = bitcast i32 %1199 to float, !dbg !7
  %1201 = select i1 %.not36, float %1185, float %1200, !dbg !7
  %1202 = select i1 %.not36, float %1200, float %1191, !dbg !7
  %1203 = select i1 %.not36, float %1192, float %1186, !dbg !7
  %1204 = bitcast float %1203 to i32, !dbg !7
  %1205 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1197, i32 %1204), !dbg !7
  %1206 = bitcast i32 %1205 to float, !dbg !7
  %1207 = select i1 %.not36, float %1186, float %1206, !dbg !7
  %1208 = select i1 %.not36, float %1206, float %1192, !dbg !7
  %1209 = extractvalue [2 x <8 x float>] %125, 0, !dbg !7
  %1210 = extractelement <8 x float> %1209, i64 0, !dbg !7
  %1211 = extractelement <8 x float> %1209, i64 1, !dbg !7
  %1212 = extractelement <8 x float> %1209, i64 2, !dbg !7
  %1213 = extractelement <8 x float> %1209, i64 3, !dbg !7
  %1214 = select i1 %.not, float %1211, float %1210, !dbg !7
  %1215 = bitcast float %1214 to i32, !dbg !7
  %1216 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1181, i32 %1215), !dbg !7
  %1217 = bitcast i32 %1216 to float, !dbg !7
  %1218 = select i1 %.not, float %1210, float %1217, !dbg !7
  %1219 = select i1 %.not, float %1217, float %1211, !dbg !7
  %1220 = select i1 %.not, float %1213, float %1212, !dbg !7
  %1221 = bitcast float %1220 to i32, !dbg !7
  %1222 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1181, i32 %1221), !dbg !7
  %1223 = bitcast i32 %1222 to float, !dbg !7
  %1224 = select i1 %.not, float %1212, float %1223, !dbg !7
  %1225 = select i1 %.not, float %1223, float %1213, !dbg !7
  %1226 = select i1 %.not36, float %1224, float %1218, !dbg !7
  %1227 = bitcast float %1226 to i32, !dbg !7
  %1228 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1197, i32 %1227), !dbg !7
  %1229 = bitcast i32 %1228 to float, !dbg !7
  %1230 = select i1 %.not36, float %1218, float %1229, !dbg !7
  %1231 = select i1 %.not36, float %1229, float %1224, !dbg !7
  %1232 = select i1 %.not36, float %1225, float %1219, !dbg !7
  %1233 = bitcast float %1232 to i32, !dbg !7
  %1234 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1197, i32 %1233), !dbg !7
  %1235 = bitcast i32 %1234 to float, !dbg !7
  %1236 = select i1 %.not36, float %1219, float %1235, !dbg !7
  %1237 = select i1 %.not36, float %1235, float %1225, !dbg !7
  %1238 = extractvalue [2 x <8 x float>] %124, 0, !dbg !7
  %1239 = extractelement <8 x float> %1238, i64 0, !dbg !7
  %1240 = extractelement <8 x float> %1238, i64 1, !dbg !7
  %1241 = extractelement <8 x float> %1238, i64 2, !dbg !7
  %1242 = extractelement <8 x float> %1238, i64 3, !dbg !7
  %1243 = select i1 %.not, float %1240, float %1239, !dbg !7
  %1244 = bitcast float %1243 to i32, !dbg !7
  %1245 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1181, i32 %1244), !dbg !7
  %1246 = bitcast i32 %1245 to float, !dbg !7
  %1247 = select i1 %.not, float %1239, float %1246, !dbg !7
  %1248 = select i1 %.not, float %1246, float %1240, !dbg !7
  %1249 = select i1 %.not, float %1242, float %1241, !dbg !7
  %1250 = bitcast float %1249 to i32, !dbg !7
  %1251 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1181, i32 %1250), !dbg !7
  %1252 = bitcast i32 %1251 to float, !dbg !7
  %1253 = select i1 %.not, float %1241, float %1252, !dbg !7
  %1254 = select i1 %.not, float %1252, float %1242, !dbg !7
  %1255 = select i1 %.not36, float %1253, float %1247, !dbg !7
  %1256 = bitcast float %1255 to i32, !dbg !7
  %1257 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1197, i32 %1256), !dbg !7
  %1258 = bitcast i32 %1257 to float, !dbg !7
  %1259 = select i1 %.not36, float %1247, float %1258, !dbg !7
  %1260 = select i1 %.not36, float %1258, float %1253, !dbg !7
  %1261 = select i1 %.not36, float %1254, float %1248, !dbg !7
  %1262 = bitcast float %1261 to i32, !dbg !7
  %1263 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1197, i32 %1262), !dbg !7
  %1264 = bitcast i32 %1263 to float, !dbg !7
  %1265 = select i1 %.not36, float %1248, float %1264, !dbg !7
  %1266 = select i1 %.not36, float %1264, float %1254, !dbg !7
  %1267 = extractvalue [2 x <8 x float>] %121, 0, !dbg !7
  %1268 = extractelement <8 x float> %1267, i64 0, !dbg !7
  %1269 = extractelement <8 x float> %1267, i64 1, !dbg !7
  %1270 = extractelement <8 x float> %1267, i64 2, !dbg !7
  %1271 = extractelement <8 x float> %1267, i64 3, !dbg !7
  %1272 = select i1 %.not, float %1269, float %1268, !dbg !7
  %1273 = bitcast float %1272 to i32, !dbg !7
  %1274 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1181, i32 %1273), !dbg !7
  %1275 = bitcast i32 %1274 to float, !dbg !7
  %1276 = select i1 %.not, float %1268, float %1275, !dbg !7
  %1277 = select i1 %.not, float %1275, float %1269, !dbg !7
  %1278 = select i1 %.not, float %1271, float %1270, !dbg !7
  %1279 = bitcast float %1278 to i32, !dbg !7
  %1280 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1181, i32 %1279), !dbg !7
  %1281 = bitcast i32 %1280 to float, !dbg !7
  %1282 = select i1 %.not, float %1270, float %1281, !dbg !7
  %1283 = select i1 %.not, float %1281, float %1271, !dbg !7
  %1284 = select i1 %.not36, float %1282, float %1276, !dbg !7
  %1285 = bitcast float %1284 to i32, !dbg !7
  %1286 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1197, i32 %1285), !dbg !7
  %1287 = select i1 %.not36, float %1283, float %1277, !dbg !7
  %1288 = bitcast float %1287 to i32, !dbg !7
  %1289 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1197, i32 %1288), !dbg !7
  %1290 = shufflevector <1 x float> %1165, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !16
  %1291 = extractelement <8 x float> %1168, i64 4, !dbg !7
  %1292 = extractelement <8 x float> %1168, i64 5, !dbg !7
  %1293 = extractelement <8 x float> %1168, i64 6, !dbg !7
  %1294 = extractelement <8 x float> %1168, i64 7, !dbg !7
  %1295 = select i1 %.not, float %1292, float %1291, !dbg !7
  %1296 = bitcast float %1295 to i32, !dbg !7
  %1297 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1181, i32 %1296), !dbg !7
  %1298 = bitcast i32 %1297 to float, !dbg !7
  %1299 = select i1 %.not, float %1291, float %1298, !dbg !7
  %1300 = select i1 %.not, float %1298, float %1292, !dbg !7
  %1301 = select i1 %.not, float %1294, float %1293, !dbg !7
  %1302 = bitcast float %1301 to i32, !dbg !7
  %1303 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1181, i32 %1302), !dbg !7
  %1304 = bitcast i32 %1303 to float, !dbg !7
  %1305 = select i1 %.not, float %1293, float %1304, !dbg !7
  %1306 = select i1 %.not, float %1304, float %1294, !dbg !7
  %1307 = select i1 %.not36, float %1305, float %1299, !dbg !7
  %1308 = bitcast float %1307 to i32, !dbg !7
  %1309 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1197, i32 %1308), !dbg !7
  %1310 = bitcast i32 %1309 to float, !dbg !7
  %1311 = select i1 %.not36, float %1299, float %1310, !dbg !7
  %1312 = select i1 %.not36, float %1310, float %1305, !dbg !7
  %1313 = select i1 %.not36, float %1306, float %1300, !dbg !7
  %1314 = bitcast float %1313 to i32, !dbg !7
  %1315 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1197, i32 %1314), !dbg !7
  %1316 = bitcast i32 %1315 to float, !dbg !7
  %1317 = select i1 %.not36, float %1300, float %1316, !dbg !7
  %1318 = select i1 %.not36, float %1316, float %1306, !dbg !7
  %1319 = extractelement <8 x float> %1209, i64 4, !dbg !7
  %1320 = extractelement <8 x float> %1209, i64 5, !dbg !7
  %1321 = extractelement <8 x float> %1209, i64 6, !dbg !7
  %1322 = extractelement <8 x float> %1209, i64 7, !dbg !7
  %1323 = select i1 %.not, float %1320, float %1319, !dbg !7
  %1324 = bitcast float %1323 to i32, !dbg !7
  %1325 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1181, i32 %1324), !dbg !7
  %1326 = bitcast i32 %1325 to float, !dbg !7
  %1327 = select i1 %.not, float %1319, float %1326, !dbg !7
  %1328 = select i1 %.not, float %1326, float %1320, !dbg !7
  %1329 = select i1 %.not, float %1322, float %1321, !dbg !7
  %1330 = bitcast float %1329 to i32, !dbg !7
  %1331 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1181, i32 %1330), !dbg !7
  %1332 = bitcast i32 %1331 to float, !dbg !7
  %1333 = select i1 %.not, float %1321, float %1332, !dbg !7
  %1334 = select i1 %.not, float %1332, float %1322, !dbg !7
  %1335 = select i1 %.not36, float %1333, float %1327, !dbg !7
  %1336 = bitcast float %1335 to i32, !dbg !7
  %1337 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1197, i32 %1336), !dbg !7
  %1338 = bitcast i32 %1337 to float, !dbg !7
  %1339 = select i1 %.not36, float %1327, float %1338, !dbg !7
  %1340 = select i1 %.not36, float %1338, float %1333, !dbg !7
  %1341 = select i1 %.not36, float %1334, float %1328, !dbg !7
  %1342 = bitcast float %1341 to i32, !dbg !7
  %1343 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1197, i32 %1342), !dbg !7
  %1344 = bitcast i32 %1343 to float, !dbg !7
  %1345 = select i1 %.not36, float %1328, float %1344, !dbg !7
  %1346 = select i1 %.not36, float %1344, float %1334, !dbg !7
  %1347 = extractelement <8 x float> %1238, i64 4, !dbg !7
  %1348 = extractelement <8 x float> %1238, i64 5, !dbg !7
  %1349 = extractelement <8 x float> %1238, i64 6, !dbg !7
  %1350 = extractelement <8 x float> %1238, i64 7, !dbg !7
  %1351 = select i1 %.not, float %1348, float %1347, !dbg !7
  %1352 = bitcast float %1351 to i32, !dbg !7
  %1353 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1181, i32 %1352), !dbg !7
  %1354 = bitcast i32 %1353 to float, !dbg !7
  %1355 = select i1 %.not, float %1347, float %1354, !dbg !7
  %1356 = select i1 %.not, float %1354, float %1348, !dbg !7
  %1357 = select i1 %.not, float %1350, float %1349, !dbg !7
  %1358 = bitcast float %1357 to i32, !dbg !7
  %1359 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1181, i32 %1358), !dbg !7
  %1360 = bitcast i32 %1359 to float, !dbg !7
  %1361 = select i1 %.not, float %1349, float %1360, !dbg !7
  %1362 = select i1 %.not, float %1360, float %1350, !dbg !7
  %1363 = select i1 %.not36, float %1361, float %1355, !dbg !7
  %1364 = bitcast float %1363 to i32, !dbg !7
  %1365 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1197, i32 %1364), !dbg !7
  %1366 = bitcast i32 %1365 to float, !dbg !7
  %1367 = select i1 %.not36, float %1355, float %1366, !dbg !7
  %1368 = select i1 %.not36, float %1366, float %1361, !dbg !7
  %1369 = select i1 %.not36, float %1362, float %1356, !dbg !7
  %1370 = bitcast float %1369 to i32, !dbg !7
  %1371 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1197, i32 %1370), !dbg !7
  %1372 = bitcast i32 %1371 to float, !dbg !7
  %1373 = select i1 %.not36, float %1356, float %1372, !dbg !7
  %1374 = select i1 %.not36, float %1372, float %1362, !dbg !7
  %1375 = extractelement <8 x float> %1267, i64 4, !dbg !7
  %1376 = extractelement <8 x float> %1267, i64 5, !dbg !7
  %1377 = extractelement <8 x float> %1267, i64 6, !dbg !7
  %1378 = extractelement <8 x float> %1267, i64 7, !dbg !7
  %1379 = select i1 %.not, float %1376, float %1375, !dbg !7
  %1380 = bitcast float %1379 to i32, !dbg !7
  %1381 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1181, i32 %1380), !dbg !7
  %1382 = bitcast i32 %1381 to float, !dbg !7
  %1383 = select i1 %.not, float %1375, float %1382, !dbg !7
  %1384 = select i1 %.not, float %1382, float %1376, !dbg !7
  %1385 = select i1 %.not, float %1378, float %1377, !dbg !7
  %1386 = bitcast float %1385 to i32, !dbg !7
  %1387 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1181, i32 %1386), !dbg !7
  %1388 = bitcast i32 %1387 to float, !dbg !7
  %1389 = select i1 %.not, float %1377, float %1388, !dbg !7
  %1390 = select i1 %.not, float %1388, float %1378, !dbg !7
  %1391 = select i1 %.not36, float %1389, float %1383, !dbg !7
  %1392 = bitcast float %1391 to i32, !dbg !7
  %1393 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1197, i32 %1392), !dbg !7
  %1394 = select i1 %.not36, float %1390, float %1384, !dbg !7
  %1395 = bitcast float %1394 to i32, !dbg !7
  %1396 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1197, i32 %1395), !dbg !7
  %1397 = extractvalue [2 x <8 x float>] %126, 1, !dbg !7
  %1398 = extractelement <8 x float> %1397, i64 0, !dbg !7
  %1399 = extractelement <8 x float> %1397, i64 1, !dbg !7
  %1400 = extractelement <8 x float> %1397, i64 2, !dbg !7
  %1401 = extractelement <8 x float> %1397, i64 3, !dbg !7
  %1402 = select i1 %.not, float %1399, float %1398, !dbg !7
  %1403 = bitcast float %1402 to i32, !dbg !7
  %1404 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1181, i32 %1403), !dbg !7
  %1405 = bitcast i32 %1404 to float, !dbg !7
  %1406 = select i1 %.not, float %1398, float %1405, !dbg !7
  %1407 = select i1 %.not, float %1405, float %1399, !dbg !7
  %1408 = select i1 %.not, float %1401, float %1400, !dbg !7
  %1409 = bitcast float %1408 to i32, !dbg !7
  %1410 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1181, i32 %1409), !dbg !7
  %1411 = bitcast i32 %1410 to float, !dbg !7
  %1412 = select i1 %.not, float %1400, float %1411, !dbg !7
  %1413 = select i1 %.not, float %1411, float %1401, !dbg !7
  %1414 = select i1 %.not36, float %1412, float %1406, !dbg !7
  %1415 = bitcast float %1414 to i32, !dbg !7
  %1416 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1197, i32 %1415), !dbg !7
  %1417 = bitcast i32 %1416 to float, !dbg !7
  %1418 = select i1 %.not36, float %1406, float %1417, !dbg !7
  %1419 = select i1 %.not36, float %1417, float %1412, !dbg !7
  %1420 = select i1 %.not36, float %1413, float %1407, !dbg !7
  %1421 = bitcast float %1420 to i32, !dbg !7
  %1422 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1197, i32 %1421), !dbg !7
  %1423 = bitcast i32 %1422 to float, !dbg !7
  %1424 = select i1 %.not36, float %1407, float %1423, !dbg !7
  %1425 = select i1 %.not36, float %1423, float %1413, !dbg !7
  %1426 = extractvalue [2 x <8 x float>] %125, 1, !dbg !7
  %1427 = extractelement <8 x float> %1426, i64 0, !dbg !7
  %1428 = extractelement <8 x float> %1426, i64 1, !dbg !7
  %1429 = extractelement <8 x float> %1426, i64 2, !dbg !7
  %1430 = extractelement <8 x float> %1426, i64 3, !dbg !7
  %1431 = select i1 %.not, float %1428, float %1427, !dbg !7
  %1432 = bitcast float %1431 to i32, !dbg !7
  %1433 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1181, i32 %1432), !dbg !7
  %1434 = bitcast i32 %1433 to float, !dbg !7
  %1435 = select i1 %.not, float %1427, float %1434, !dbg !7
  %1436 = select i1 %.not, float %1434, float %1428, !dbg !7
  %1437 = select i1 %.not, float %1430, float %1429, !dbg !7
  %1438 = bitcast float %1437 to i32, !dbg !7
  %1439 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1181, i32 %1438), !dbg !7
  %1440 = bitcast i32 %1439 to float, !dbg !7
  %1441 = select i1 %.not, float %1429, float %1440, !dbg !7
  %1442 = select i1 %.not, float %1440, float %1430, !dbg !7
  %1443 = select i1 %.not36, float %1441, float %1435, !dbg !7
  %1444 = bitcast float %1443 to i32, !dbg !7
  %1445 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1197, i32 %1444), !dbg !7
  %1446 = bitcast i32 %1445 to float, !dbg !7
  %1447 = select i1 %.not36, float %1435, float %1446, !dbg !7
  %1448 = select i1 %.not36, float %1446, float %1441, !dbg !7
  %1449 = select i1 %.not36, float %1442, float %1436, !dbg !7
  %1450 = bitcast float %1449 to i32, !dbg !7
  %1451 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1197, i32 %1450), !dbg !7
  %1452 = bitcast i32 %1451 to float, !dbg !7
  %1453 = select i1 %.not36, float %1436, float %1452, !dbg !7
  %1454 = select i1 %.not36, float %1452, float %1442, !dbg !7
  %1455 = extractvalue [2 x <8 x float>] %124, 1, !dbg !7
  %1456 = extractelement <8 x float> %1455, i64 0, !dbg !7
  %1457 = extractelement <8 x float> %1455, i64 1, !dbg !7
  %1458 = extractelement <8 x float> %1455, i64 2, !dbg !7
  %1459 = extractelement <8 x float> %1455, i64 3, !dbg !7
  %1460 = select i1 %.not, float %1457, float %1456, !dbg !7
  %1461 = bitcast float %1460 to i32, !dbg !7
  %1462 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1181, i32 %1461), !dbg !7
  %1463 = bitcast i32 %1462 to float, !dbg !7
  %1464 = select i1 %.not, float %1456, float %1463, !dbg !7
  %1465 = select i1 %.not, float %1463, float %1457, !dbg !7
  %1466 = select i1 %.not, float %1459, float %1458, !dbg !7
  %1467 = bitcast float %1466 to i32, !dbg !7
  %1468 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1181, i32 %1467), !dbg !7
  %1469 = bitcast i32 %1468 to float, !dbg !7
  %1470 = select i1 %.not, float %1458, float %1469, !dbg !7
  %1471 = select i1 %.not, float %1469, float %1459, !dbg !7
  %1472 = select i1 %.not36, float %1470, float %1464, !dbg !7
  %1473 = bitcast float %1472 to i32, !dbg !7
  %1474 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1197, i32 %1473), !dbg !7
  %1475 = bitcast i32 %1474 to float, !dbg !7
  %1476 = select i1 %.not36, float %1464, float %1475, !dbg !7
  %1477 = select i1 %.not36, float %1475, float %1470, !dbg !7
  %1478 = select i1 %.not36, float %1471, float %1465, !dbg !7
  %1479 = bitcast float %1478 to i32, !dbg !7
  %1480 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1197, i32 %1479), !dbg !7
  %1481 = bitcast i32 %1480 to float, !dbg !7
  %1482 = select i1 %.not36, float %1465, float %1481, !dbg !7
  %1483 = select i1 %.not36, float %1481, float %1471, !dbg !7
  %1484 = extractvalue [2 x <8 x float>] %121, 1, !dbg !7
  %1485 = extractelement <8 x float> %1484, i64 0, !dbg !7
  %1486 = extractelement <8 x float> %1484, i64 1, !dbg !7
  %1487 = extractelement <8 x float> %1484, i64 2, !dbg !7
  %1488 = extractelement <8 x float> %1484, i64 3, !dbg !7
  %1489 = select i1 %.not, float %1486, float %1485, !dbg !7
  %1490 = bitcast float %1489 to i32, !dbg !7
  %1491 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1181, i32 %1490), !dbg !7
  %1492 = bitcast i32 %1491 to float, !dbg !7
  %1493 = select i1 %.not, float %1485, float %1492, !dbg !7
  %1494 = select i1 %.not, float %1492, float %1486, !dbg !7
  %1495 = select i1 %.not, float %1488, float %1487, !dbg !7
  %1496 = bitcast float %1495 to i32, !dbg !7
  %1497 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1181, i32 %1496), !dbg !7
  %1498 = bitcast i32 %1497 to float, !dbg !7
  %1499 = select i1 %.not, float %1487, float %1498, !dbg !7
  %1500 = select i1 %.not, float %1498, float %1488, !dbg !7
  %1501 = select i1 %.not36, float %1499, float %1493, !dbg !7
  %1502 = bitcast float %1501 to i32, !dbg !7
  %1503 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1197, i32 %1502), !dbg !7
  %1504 = select i1 %.not36, float %1500, float %1494, !dbg !7
  %1505 = bitcast float %1504 to i32, !dbg !7
  %1506 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1197, i32 %1505), !dbg !7
  %1507 = shufflevector <1 x float> %1156, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !16
  %1508 = extractelement <8 x float> %1397, i64 4, !dbg !7
  %1509 = extractelement <8 x float> %1397, i64 5, !dbg !7
  %1510 = extractelement <8 x float> %1397, i64 6, !dbg !7
  %1511 = extractelement <8 x float> %1397, i64 7, !dbg !7
  %1512 = select i1 %.not, float %1509, float %1508, !dbg !7
  %1513 = bitcast float %1512 to i32, !dbg !7
  %1514 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1181, i32 %1513), !dbg !7
  %1515 = bitcast i32 %1514 to float, !dbg !7
  %1516 = select i1 %.not, float %1508, float %1515, !dbg !7
  %1517 = select i1 %.not, float %1515, float %1509, !dbg !7
  %1518 = select i1 %.not, float %1511, float %1510, !dbg !7
  %1519 = bitcast float %1518 to i32, !dbg !7
  %1520 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1181, i32 %1519), !dbg !7
  %1521 = bitcast i32 %1520 to float, !dbg !7
  %1522 = select i1 %.not, float %1510, float %1521, !dbg !7
  %1523 = select i1 %.not, float %1521, float %1511, !dbg !7
  %1524 = select i1 %.not36, float %1522, float %1516, !dbg !7
  %1525 = bitcast float %1524 to i32, !dbg !7
  %1526 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1197, i32 %1525), !dbg !7
  %1527 = bitcast i32 %1526 to float, !dbg !7
  %1528 = select i1 %.not36, float %1516, float %1527, !dbg !7
  %1529 = select i1 %.not36, float %1527, float %1522, !dbg !7
  %1530 = select i1 %.not36, float %1523, float %1517, !dbg !7
  %1531 = bitcast float %1530 to i32, !dbg !7
  %1532 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1197, i32 %1531), !dbg !7
  %1533 = bitcast i32 %1532 to float, !dbg !7
  %1534 = select i1 %.not36, float %1517, float %1533, !dbg !7
  %1535 = select i1 %.not36, float %1533, float %1523, !dbg !7
  %1536 = extractelement <8 x float> %1426, i64 4, !dbg !7
  %1537 = extractelement <8 x float> %1426, i64 5, !dbg !7
  %1538 = extractelement <8 x float> %1426, i64 6, !dbg !7
  %1539 = extractelement <8 x float> %1426, i64 7, !dbg !7
  %1540 = select i1 %.not, float %1537, float %1536, !dbg !7
  %1541 = bitcast float %1540 to i32, !dbg !7
  %1542 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1181, i32 %1541), !dbg !7
  %1543 = bitcast i32 %1542 to float, !dbg !7
  %1544 = select i1 %.not, float %1536, float %1543, !dbg !7
  %1545 = select i1 %.not, float %1543, float %1537, !dbg !7
  %1546 = select i1 %.not, float %1539, float %1538, !dbg !7
  %1547 = bitcast float %1546 to i32, !dbg !7
  %1548 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1181, i32 %1547), !dbg !7
  %1549 = bitcast i32 %1548 to float, !dbg !7
  %1550 = select i1 %.not, float %1538, float %1549, !dbg !7
  %1551 = select i1 %.not, float %1549, float %1539, !dbg !7
  %1552 = select i1 %.not36, float %1550, float %1544, !dbg !7
  %1553 = bitcast float %1552 to i32, !dbg !7
  %1554 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1197, i32 %1553), !dbg !7
  %1555 = bitcast i32 %1554 to float, !dbg !7
  %1556 = select i1 %.not36, float %1544, float %1555, !dbg !7
  %1557 = select i1 %.not36, float %1555, float %1550, !dbg !7
  %1558 = select i1 %.not36, float %1551, float %1545, !dbg !7
  %1559 = bitcast float %1558 to i32, !dbg !7
  %1560 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1197, i32 %1559), !dbg !7
  %1561 = bitcast i32 %1560 to float, !dbg !7
  %1562 = select i1 %.not36, float %1545, float %1561, !dbg !7
  %1563 = select i1 %.not36, float %1561, float %1551, !dbg !7
  %1564 = extractelement <8 x float> %1455, i64 4, !dbg !7
  %1565 = extractelement <8 x float> %1455, i64 5, !dbg !7
  %1566 = extractelement <8 x float> %1455, i64 6, !dbg !7
  %1567 = extractelement <8 x float> %1455, i64 7, !dbg !7
  %1568 = select i1 %.not, float %1565, float %1564, !dbg !7
  %1569 = bitcast float %1568 to i32, !dbg !7
  %1570 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1181, i32 %1569), !dbg !7
  %1571 = bitcast i32 %1570 to float, !dbg !7
  %1572 = select i1 %.not, float %1564, float %1571, !dbg !7
  %1573 = select i1 %.not, float %1571, float %1565, !dbg !7
  %1574 = select i1 %.not, float %1567, float %1566, !dbg !7
  %1575 = bitcast float %1574 to i32, !dbg !7
  %1576 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1181, i32 %1575), !dbg !7
  %1577 = bitcast i32 %1576 to float, !dbg !7
  %1578 = select i1 %.not, float %1566, float %1577, !dbg !7
  %1579 = select i1 %.not, float %1577, float %1567, !dbg !7
  %1580 = select i1 %.not36, float %1578, float %1572, !dbg !7
  %1581 = bitcast float %1580 to i32, !dbg !7
  %1582 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1197, i32 %1581), !dbg !7
  %1583 = bitcast i32 %1582 to float, !dbg !7
  %1584 = select i1 %.not36, float %1572, float %1583, !dbg !7
  %1585 = select i1 %.not36, float %1583, float %1578, !dbg !7
  %1586 = select i1 %.not36, float %1579, float %1573, !dbg !7
  %1587 = bitcast float %1586 to i32, !dbg !7
  %1588 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1197, i32 %1587), !dbg !7
  %1589 = bitcast i32 %1588 to float, !dbg !7
  %1590 = select i1 %.not36, float %1573, float %1589, !dbg !7
  %1591 = select i1 %.not36, float %1589, float %1579, !dbg !7
  %1592 = extractelement <8 x float> %1484, i64 4, !dbg !7
  %1593 = extractelement <8 x float> %1484, i64 5, !dbg !7
  %1594 = extractelement <8 x float> %1484, i64 6, !dbg !7
  %1595 = extractelement <8 x float> %1484, i64 7, !dbg !7
  %1596 = select i1 %.not, float %1593, float %1592, !dbg !7
  %1597 = bitcast float %1596 to i32, !dbg !7
  %1598 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1181, i32 %1597), !dbg !7
  %1599 = bitcast i32 %1598 to float, !dbg !7
  %1600 = select i1 %.not, float %1592, float %1599, !dbg !7
  %1601 = select i1 %.not, float %1599, float %1593, !dbg !7
  %1602 = select i1 %.not, float %1595, float %1594, !dbg !7
  %1603 = bitcast float %1602 to i32, !dbg !7
  %1604 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1181, i32 %1603), !dbg !7
  %1605 = bitcast i32 %1604 to float, !dbg !7
  %1606 = select i1 %.not, float %1594, float %1605, !dbg !7
  %1607 = select i1 %.not, float %1605, float %1595, !dbg !7
  %1608 = select i1 %.not36, float %1606, float %1600, !dbg !7
  %1609 = bitcast float %1608 to i32, !dbg !7
  %1610 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1197, i32 %1609), !dbg !7
  %1611 = select i1 %.not36, float %1607, float %1601, !dbg !7
  %1612 = bitcast float %1611 to i32, !dbg !7
  %1613 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1197, i32 %1612), !dbg !7
  %1614 = bitcast i32 %1610 to float, !dbg !7
  %1615 = select i1 %.not36, float %1600, float %1614, !dbg !7
  %1616 = insertelement <4 x float> poison, float %1615, i64 0, !dbg !16
  %1617 = bitcast i32 %1613 to float, !dbg !7
  %1618 = select i1 %.not36, float %1601, float %1617, !dbg !7
  %1619 = insertelement <4 x float> %1616, float %1618, i64 1, !dbg !16
  %1620 = select i1 %.not36, float %1614, float %1606, !dbg !7
  %1621 = insertelement <4 x float> %1619, float %1620, i64 2, !dbg !16
  %1622 = select i1 %.not36, float %1617, float %1607, !dbg !7
  %1623 = insertelement <4 x float> %1621, float %1622, i64 3, !dbg !16
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1507, !dbg !16
  %normalized = fmul <4 x float> %1623, %row.reciprocal, !dbg !16
  %1624 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !17
  %1625 = extractelement <4 x half> %1624, i64 3, !dbg !17
  %1626 = extractelement <4 x half> %1624, i64 2, !dbg !17
  %1627 = extractelement <4 x half> %1624, i64 1, !dbg !17
  %1628 = extractelement <4 x half> %1624, i64 0, !dbg !17
  %1629 = bitcast i32 %1503 to float, !dbg !7
  %1630 = select i1 %.not36, float %1493, float %1629, !dbg !7
  %1631 = insertelement <4 x float> poison, float %1630, i64 0, !dbg !16
  %1632 = bitcast i32 %1506 to float, !dbg !7
  %1633 = select i1 %.not36, float %1494, float %1632, !dbg !7
  %1634 = insertelement <4 x float> %1631, float %1633, i64 1, !dbg !16
  %1635 = select i1 %.not36, float %1629, float %1499, !dbg !7
  %1636 = insertelement <4 x float> %1634, float %1635, i64 2, !dbg !16
  %1637 = select i1 %.not36, float %1632, float %1500, !dbg !7
  %1638 = insertelement <4 x float> %1636, float %1637, i64 3, !dbg !16
  %normalized37 = fmul <4 x float> %1638, %row.reciprocal, !dbg !16
  %1639 = fptrunc <4 x float> %normalized37 to <4 x half>, !dbg !17
  %1640 = extractelement <4 x half> %1639, i64 3, !dbg !17
  %1641 = extractelement <4 x half> %1639, i64 2, !dbg !17
  %1642 = extractelement <4 x half> %1639, i64 1, !dbg !17
  %1643 = extractelement <4 x half> %1639, i64 0, !dbg !17
  %1644 = bitcast i32 %1393 to float, !dbg !7
  %1645 = select i1 %.not36, float %1383, float %1644, !dbg !7
  %1646 = insertelement <4 x float> poison, float %1645, i64 0, !dbg !16
  %1647 = bitcast i32 %1396 to float, !dbg !7
  %1648 = select i1 %.not36, float %1384, float %1647, !dbg !7
  %1649 = insertelement <4 x float> %1646, float %1648, i64 1, !dbg !16
  %1650 = select i1 %.not36, float %1644, float %1389, !dbg !7
  %1651 = insertelement <4 x float> %1649, float %1650, i64 2, !dbg !16
  %1652 = select i1 %.not36, float %1647, float %1390, !dbg !7
  %1653 = insertelement <4 x float> %1651, float %1652, i64 3, !dbg !16
  %row.reciprocal38 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1290, !dbg !16
  %normalized39 = fmul <4 x float> %1653, %row.reciprocal38, !dbg !16
  %1654 = fptrunc <4 x float> %normalized39 to <4 x half>, !dbg !17
  %1655 = extractelement <4 x half> %1654, i64 3, !dbg !17
  %1656 = extractelement <4 x half> %1654, i64 2, !dbg !17
  %1657 = extractelement <4 x half> %1654, i64 1, !dbg !17
  %1658 = extractelement <4 x half> %1654, i64 0, !dbg !17
  %1659 = bitcast i32 %1286 to float, !dbg !7
  %1660 = select i1 %.not36, float %1276, float %1659, !dbg !7
  %1661 = insertelement <4 x float> poison, float %1660, i64 0, !dbg !16
  %1662 = bitcast i32 %1289 to float, !dbg !7
  %1663 = select i1 %.not36, float %1277, float %1662, !dbg !7
  %1664 = insertelement <4 x float> %1661, float %1663, i64 1, !dbg !16
  %1665 = select i1 %.not36, float %1659, float %1282, !dbg !7
  %1666 = insertelement <4 x float> %1664, float %1665, i64 2, !dbg !16
  %1667 = select i1 %.not36, float %1662, float %1283, !dbg !7
  %1668 = insertelement <4 x float> %1666, float %1667, i64 3, !dbg !16
  %normalized40 = fmul <4 x float> %1668, %row.reciprocal38, !dbg !16
  %1669 = fptrunc <4 x float> %normalized40 to <4 x half>, !dbg !17
  %1670 = extractelement <4 x half> %1669, i64 3, !dbg !17
  %1671 = extractelement <4 x half> %1669, i64 2, !dbg !17
  %1672 = extractelement <4 x half> %1669, i64 1, !dbg !17
  %1673 = extractelement <4 x half> %1669, i64 0, !dbg !17
  %1674 = add nuw nsw i64 %15, %16, !dbg !18
  %1675 = shl i64 %1674, 19, !dbg !18
  %1676 = and i64 %1675, 16252928, !dbg !18
  %address.thread974 = call i32 asm sideeffect "", "=v,0"(i32 %25) #9, !dbg !18, !range !32, !frisk.register_lifetimes !1
  %address.local975 = zext nneg i32 %address.thread974 to i64, !dbg !8
  %address.local976 = add nuw i64 %19, %address.local975, !dbg !18
  %1677 = lshr i64 %address.local975, 2, !dbg !18
  %address.local978 = and i64 %1677, 48, !dbg !18
  %address.local979 = add i64 %address.local976, %address.local978, !dbg !18
  %address.local980 = and i64 %address.local975, 240, !dbg !18
  %address.local981 = sub i64 %address.local979, %address.local980, !dbg !18
  %address.local982 = add i64 %address.local981, %20, !dbg !18
  %address.local983 = shl nuw nsw i64 %address.local982, 7, !dbg !18
  %address.local984 = add nuw nsw i64 %1676, %address.local983, !dbg !18
  %address.local985 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local984, !dbg !18
  %address.local986 = lshr i64 %address.local975, 2, !dbg !10
  %address.local987 = and i64 %address.local986, 12, !dbg !10
  %address.local988 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local987, !dbg !18
  store half %1673, ptr addrspace(1) %address.local988, align 2, !dbg !18
  %address.local989 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local984, !dbg !18
  %address.local990 = or disjoint i64 %address.local987, 1, !dbg !18
  %address.local991 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local989, i64 %address.local990, !dbg !18
  store half %1672, ptr addrspace(1) %address.local991, align 2, !dbg !18
  %address.local992 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local984, !dbg !18
  %address.local993 = or disjoint i64 %address.local987, 2, !dbg !18
  %address.local994 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local992, i64 %address.local993, !dbg !18
  store half %1671, ptr addrspace(1) %address.local994, align 2, !dbg !18
  %address.local995 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local984, !dbg !18
  %address.local996 = or disjoint i64 %address.local987, 3, !dbg !18
  %address.local997 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local995, i64 %address.local996, !dbg !18
  store half %1670, ptr addrspace(1) %address.local997, align 2, !dbg !18
  %address.local998 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local984, !dbg !18
  %address.local999 = or disjoint i64 %address.local987, 16, !dbg !18
  %address.local1000 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local998, i64 %address.local999, !dbg !18
  store half %1658, ptr addrspace(1) %address.local1000, align 2, !dbg !18
  %address.local1001 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local984, !dbg !18
  %address.local1002 = or disjoint i64 %address.local987, 17, !dbg !18
  %address.local1003 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1001, i64 %address.local1002, !dbg !18
  store half %1657, ptr addrspace(1) %address.local1003, align 2, !dbg !18
  %address.local1004 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local984, !dbg !18
  %address.local1005 = or disjoint i64 %address.local987, 18, !dbg !18
  %address.local1006 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1004, i64 %address.local1005, !dbg !18
  store half %1656, ptr addrspace(1) %address.local1006, align 2, !dbg !18
  %address.local1007 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local984, !dbg !18
  %address.local1008 = or disjoint i64 %address.local987, 19, !dbg !18
  %address.local1009 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1007, i64 %address.local1008, !dbg !18
  store half %1655, ptr addrspace(1) %address.local1009, align 2, !dbg !18
  %address.local1010 = shl i64 %address.local982, 7, !dbg !18
  %address.local1011 = add i64 %address.local1010, 8192, !dbg !18
  %address.local1012 = add nuw nsw i64 %1676, %address.local1011, !dbg !18
  %address.local1013 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1012, !dbg !18
  %address.local1014 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local987, !dbg !18
  store half %1643, ptr addrspace(1) %address.local1014, align 2, !dbg !18
  %address.local1015 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1012, !dbg !18
  %address.local1016 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1015, i64 %address.local990, !dbg !18
  store half %1642, ptr addrspace(1) %address.local1016, align 2, !dbg !18
  %address.local1017 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1012, !dbg !18
  %address.local1018 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1017, i64 %address.local993, !dbg !18
  store half %1641, ptr addrspace(1) %address.local1018, align 2, !dbg !18
  %address.local1019 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1012, !dbg !18
  %address.local1020 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1019, i64 %address.local996, !dbg !18
  store half %1640, ptr addrspace(1) %address.local1020, align 2, !dbg !18
  %address.local1021 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1012, !dbg !18
  %address.local1022 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1021, i64 %address.local999, !dbg !18
  store half %1628, ptr addrspace(1) %address.local1022, align 2, !dbg !18
  %address.local1023 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1012, !dbg !18
  %address.local1024 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1023, i64 %address.local1002, !dbg !18
  store half %1627, ptr addrspace(1) %address.local1024, align 2, !dbg !18
  %address.local1025 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1012, !dbg !18
  %address.local1026 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1025, i64 %address.local1005, !dbg !18
  store half %1626, ptr addrspace(1) %address.local1026, align 2, !dbg !18
  %address.local1027 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1012, !dbg !18
  %address.local1028 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1027, i64 %address.local1008, !dbg !18
  store half %1625, ptr addrspace(1) %address.local1028, align 2, !dbg !18
  %1678 = insertelement <4 x float> poison, float %1584, i64 0, !dbg !16
  %1679 = insertelement <4 x float> %1678, float %1590, i64 1, !dbg !16
  %1680 = insertelement <4 x float> %1679, float %1585, i64 2, !dbg !16
  %1681 = insertelement <4 x float> %1680, float %1591, i64 3, !dbg !16
  %row.reciprocal41 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1507, !dbg !16
  %normalized42 = fmul <4 x float> %1681, %row.reciprocal41, !dbg !16
  %1682 = fptrunc <4 x float> %normalized42 to <4 x half>, !dbg !17
  %1683 = extractelement <4 x half> %1682, i64 3, !dbg !17
  %1684 = extractelement <4 x half> %1682, i64 2, !dbg !17
  %1685 = extractelement <4 x half> %1682, i64 1, !dbg !17
  %1686 = extractelement <4 x half> %1682, i64 0, !dbg !17
  %1687 = insertelement <4 x float> poison, float %1476, i64 0, !dbg !16
  %1688 = insertelement <4 x float> %1687, float %1482, i64 1, !dbg !16
  %1689 = insertelement <4 x float> %1688, float %1477, i64 2, !dbg !16
  %1690 = insertelement <4 x float> %1689, float %1483, i64 3, !dbg !16
  %normalized43 = fmul <4 x float> %1690, %row.reciprocal41, !dbg !16
  %1691 = fptrunc <4 x float> %normalized43 to <4 x half>, !dbg !17
  %1692 = extractelement <4 x half> %1691, i64 3, !dbg !17
  %1693 = extractelement <4 x half> %1691, i64 2, !dbg !17
  %1694 = extractelement <4 x half> %1691, i64 1, !dbg !17
  %1695 = extractelement <4 x half> %1691, i64 0, !dbg !17
  %1696 = insertelement <4 x float> poison, float %1367, i64 0, !dbg !16
  %1697 = insertelement <4 x float> %1696, float %1373, i64 1, !dbg !16
  %1698 = insertelement <4 x float> %1697, float %1368, i64 2, !dbg !16
  %1699 = insertelement <4 x float> %1698, float %1374, i64 3, !dbg !16
  %row.reciprocal44 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1290, !dbg !16
  %normalized45 = fmul <4 x float> %1699, %row.reciprocal44, !dbg !16
  %1700 = fptrunc <4 x float> %normalized45 to <4 x half>, !dbg !17
  %1701 = extractelement <4 x half> %1700, i64 3, !dbg !17
  %1702 = extractelement <4 x half> %1700, i64 2, !dbg !17
  %1703 = extractelement <4 x half> %1700, i64 1, !dbg !17
  %1704 = extractelement <4 x half> %1700, i64 0, !dbg !17
  %1705 = insertelement <4 x float> poison, float %1259, i64 0, !dbg !16
  %1706 = insertelement <4 x float> %1705, float %1265, i64 1, !dbg !16
  %1707 = insertelement <4 x float> %1706, float %1260, i64 2, !dbg !16
  %1708 = insertelement <4 x float> %1707, float %1266, i64 3, !dbg !16
  %normalized46 = fmul <4 x float> %1708, %row.reciprocal44, !dbg !16
  %1709 = fptrunc <4 x float> %normalized46 to <4 x half>, !dbg !17
  %1710 = extractelement <4 x half> %1709, i64 3, !dbg !17
  %1711 = extractelement <4 x half> %1709, i64 2, !dbg !17
  %1712 = extractelement <4 x half> %1709, i64 1, !dbg !17
  %1713 = extractelement <4 x half> %1709, i64 0, !dbg !17
  %1714 = shl nuw nsw i64 %18, 19, !dbg !18
  %address.thread1029 = call i32 asm sideeffect "", "=v,0"(i32 %25) #9, !dbg !18, !range !32, !frisk.register_lifetimes !1
  %address.local1030 = zext nneg i32 %address.thread1029 to i64, !dbg !8
  %address.local1031 = add nuw i64 %19, %address.local1030, !dbg !18
  %1715 = lshr i64 %address.local1030, 2, !dbg !18
  %address.local1033 = and i64 %1715, 48, !dbg !18
  %address.local1034 = add i64 %address.local1031, %address.local1033, !dbg !18
  %address.local1035 = and i64 %address.local1030, 240, !dbg !18
  %address.local1036 = sub i64 %address.local1034, %address.local1035, !dbg !18
  %address.local1037 = add i64 %address.local1036, %20, !dbg !18
  %address.local1038 = shl nuw nsw i64 %address.local1037, 7, !dbg !18
  %address.local1039 = add nuw nsw i64 %1714, %address.local1038, !dbg !18
  %address.local1040 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1039, !dbg !18
  %address.local1041 = lshr i64 %address.local1030, 2, !dbg !10
  %address.local1042 = and i64 %address.local1041, 12, !dbg !10
  %address.local1043 = or disjoint i64 %address.local1042, 32, !dbg !18
  %address.local1044 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1040, i64 %address.local1043, !dbg !18
  store half %1713, ptr addrspace(1) %address.local1044, align 2, !dbg !18
  %address.local1045 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1039, !dbg !18
  %address.local1046 = or disjoint i64 %address.local1042, 33, !dbg !18
  %address.local1047 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1045, i64 %address.local1046, !dbg !18
  store half %1712, ptr addrspace(1) %address.local1047, align 2, !dbg !18
  %address.local1048 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1039, !dbg !18
  %address.local1049 = or disjoint i64 %address.local1042, 34, !dbg !18
  %address.local1050 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1048, i64 %address.local1049, !dbg !18
  store half %1711, ptr addrspace(1) %address.local1050, align 2, !dbg !18
  %address.local1051 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1039, !dbg !18
  %address.local1052 = or disjoint i64 %address.local1042, 35, !dbg !18
  %address.local1053 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1051, i64 %address.local1052, !dbg !18
  store half %1710, ptr addrspace(1) %address.local1053, align 2, !dbg !18
  %address.local1054 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1039, !dbg !18
  %address.local1055 = or disjoint i64 %address.local1042, 48, !dbg !18
  %address.local1056 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1055, !dbg !18
  store half %1704, ptr addrspace(1) %address.local1056, align 2, !dbg !18
  %address.local1057 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1039, !dbg !18
  %address.local1058 = or disjoint i64 %address.local1042, 49, !dbg !18
  %address.local1059 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1057, i64 %address.local1058, !dbg !18
  store half %1703, ptr addrspace(1) %address.local1059, align 2, !dbg !18
  %address.local1060 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1039, !dbg !18
  %address.local1061 = or disjoint i64 %address.local1042, 50, !dbg !18
  %address.local1062 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1060, i64 %address.local1061, !dbg !18
  store half %1702, ptr addrspace(1) %address.local1062, align 2, !dbg !18
  %address.local1063 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1039, !dbg !18
  %address.local1064 = or i64 %address.local1041, 51, !dbg !18
  %address.local1065 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1063, i64 %address.local1064, !dbg !18
  store half %1701, ptr addrspace(1) %address.local1065, align 2, !dbg !18
  %address.local1066 = shl i64 %address.local1037, 7, !dbg !18
  %address.local1067 = add i64 %address.local1066, 8192, !dbg !18
  %address.local1068 = add nuw nsw i64 %1714, %address.local1067, !dbg !18
  %address.local1069 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1068, !dbg !18
  %address.local1070 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1069, i64 %address.local1043, !dbg !18
  store half %1695, ptr addrspace(1) %address.local1070, align 2, !dbg !18
  %address.local1071 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1068, !dbg !18
  %address.local1072 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1071, i64 %address.local1046, !dbg !18
  store half %1694, ptr addrspace(1) %address.local1072, align 2, !dbg !18
  %address.local1073 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1068, !dbg !18
  %address.local1074 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1073, i64 %address.local1049, !dbg !18
  store half %1693, ptr addrspace(1) %address.local1074, align 2, !dbg !18
  %address.local1075 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1068, !dbg !18
  %address.local1076 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1075, i64 %address.local1052, !dbg !18
  store half %1692, ptr addrspace(1) %address.local1076, align 2, !dbg !18
  %address.local1077 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1068, !dbg !18
  %address.local1078 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1077, i64 %address.local1055, !dbg !18
  store half %1686, ptr addrspace(1) %address.local1078, align 2, !dbg !18
  %address.local1079 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1068, !dbg !18
  %address.local1080 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1079, i64 %address.local1058, !dbg !18
  store half %1685, ptr addrspace(1) %address.local1080, align 2, !dbg !18
  %address.local1081 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1068, !dbg !18
  %address.local1082 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1061, !dbg !18
  store half %1684, ptr addrspace(1) %address.local1082, align 2, !dbg !18
  %address.local1083 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1068, !dbg !18
  %address.local1084 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1083, i64 %address.local1064, !dbg !18
  store half %1683, ptr addrspace(1) %address.local1084, align 2, !dbg !18
  %1716 = insertelement <4 x float> poison, float %1556, i64 0, !dbg !16
  %1717 = insertelement <4 x float> %1716, float %1562, i64 1, !dbg !16
  %1718 = insertelement <4 x float> %1717, float %1557, i64 2, !dbg !16
  %1719 = insertelement <4 x float> %1718, float %1563, i64 3, !dbg !16
  %row.reciprocal47 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1507, !dbg !16
  %normalized48 = fmul <4 x float> %1719, %row.reciprocal47, !dbg !16
  %1720 = fptrunc <4 x float> %normalized48 to <4 x half>, !dbg !17
  %1721 = extractelement <4 x half> %1720, i64 3, !dbg !17
  %1722 = extractelement <4 x half> %1720, i64 2, !dbg !17
  %1723 = extractelement <4 x half> %1720, i64 1, !dbg !17
  %1724 = extractelement <4 x half> %1720, i64 0, !dbg !17
  %1725 = insertelement <4 x float> poison, float %1447, i64 0, !dbg !16
  %1726 = insertelement <4 x float> %1725, float %1453, i64 1, !dbg !16
  %1727 = insertelement <4 x float> %1726, float %1448, i64 2, !dbg !16
  %1728 = insertelement <4 x float> %1727, float %1454, i64 3, !dbg !16
  %normalized49 = fmul <4 x float> %1728, %row.reciprocal47, !dbg !16
  %1729 = fptrunc <4 x float> %normalized49 to <4 x half>, !dbg !17
  %1730 = extractelement <4 x half> %1729, i64 3, !dbg !17
  %1731 = extractelement <4 x half> %1729, i64 2, !dbg !17
  %1732 = extractelement <4 x half> %1729, i64 1, !dbg !17
  %1733 = extractelement <4 x half> %1729, i64 0, !dbg !17
  %1734 = insertelement <4 x float> poison, float %1339, i64 0, !dbg !16
  %1735 = insertelement <4 x float> %1734, float %1345, i64 1, !dbg !16
  %1736 = insertelement <4 x float> %1735, float %1340, i64 2, !dbg !16
  %1737 = insertelement <4 x float> %1736, float %1346, i64 3, !dbg !16
  %row.reciprocal50 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1290, !dbg !16
  %normalized51 = fmul <4 x float> %1737, %row.reciprocal50, !dbg !16
  %1738 = fptrunc <4 x float> %normalized51 to <4 x half>, !dbg !17
  %1739 = extractelement <4 x half> %1738, i64 3, !dbg !17
  %1740 = extractelement <4 x half> %1738, i64 2, !dbg !17
  %1741 = extractelement <4 x half> %1738, i64 1, !dbg !17
  %1742 = extractelement <4 x half> %1738, i64 0, !dbg !17
  %1743 = insertelement <4 x float> poison, float %1230, i64 0, !dbg !16
  %1744 = insertelement <4 x float> %1743, float %1236, i64 1, !dbg !16
  %1745 = insertelement <4 x float> %1744, float %1231, i64 2, !dbg !16
  %1746 = insertelement <4 x float> %1745, float %1237, i64 3, !dbg !16
  %normalized52 = fmul <4 x float> %1746, %row.reciprocal50, !dbg !16
  %1747 = fptrunc <4 x float> %normalized52 to <4 x half>, !dbg !17
  %1748 = extractelement <4 x half> %1747, i64 3, !dbg !17
  %1749 = extractelement <4 x half> %1747, i64 2, !dbg !17
  %1750 = extractelement <4 x half> %1747, i64 1, !dbg !17
  %1751 = extractelement <4 x half> %1747, i64 0, !dbg !17
  %1752 = shl nuw nsw i64 %18, 19, !dbg !18
  %address.thread1085 = call i32 asm sideeffect "", "=v,0"(i32 %25) #9, !dbg !18, !range !32, !frisk.register_lifetimes !1
  %address.local1086 = zext nneg i32 %address.thread1085 to i64, !dbg !8
  %address.local1087 = add nuw i64 %19, %address.local1086, !dbg !18
  %1753 = lshr i64 %address.local1086, 2, !dbg !18
  %address.local1089 = and i64 %1753, 48, !dbg !18
  %address.local1090 = add i64 %address.local1087, %address.local1089, !dbg !18
  %address.local1091 = and i64 %address.local1086, 240, !dbg !18
  %address.local1092 = sub i64 %address.local1090, %address.local1091, !dbg !18
  %address.local1093 = add i64 %address.local1092, %20, !dbg !18
  %address.local1094 = shl nuw nsw i64 %address.local1093, 7, !dbg !18
  %address.local1095 = add nuw nsw i64 %1752, %address.local1094, !dbg !18
  %address.local1096 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1095, !dbg !18
  %address.local1097 = lshr i64 %address.local1086, 2, !dbg !10
  %address.local1098 = and i64 %address.local1097, 12, !dbg !10
  %address.local1099 = or disjoint i64 %address.local1098, 64, !dbg !18
  %address.local1100 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1096, i64 %address.local1099, !dbg !18
  store half %1751, ptr addrspace(1) %address.local1100, align 2, !dbg !18
  %address.local1101 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1095, !dbg !18
  %address.local1102 = or disjoint i64 %address.local1098, 65, !dbg !18
  %address.local1103 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1101, i64 %address.local1102, !dbg !18
  store half %1750, ptr addrspace(1) %address.local1103, align 2, !dbg !18
  %address.local1104 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1095, !dbg !18
  %address.local1105 = or disjoint i64 %address.local1098, 66, !dbg !18
  %address.local1106 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1104, i64 %address.local1105, !dbg !18
  store half %1749, ptr addrspace(1) %address.local1106, align 2, !dbg !18
  %address.local1107 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1095, !dbg !18
  %address.local1108 = or disjoint i64 %address.local1098, 67, !dbg !18
  %address.local1109 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1107, i64 %address.local1108, !dbg !18
  store half %1748, ptr addrspace(1) %address.local1109, align 2, !dbg !18
  %address.local1110 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1095, !dbg !18
  %address.local1111 = or disjoint i64 %address.local1098, 80, !dbg !18
  %address.local1112 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1110, i64 %address.local1111, !dbg !18
  store half %1742, ptr addrspace(1) %address.local1112, align 2, !dbg !18
  %address.local1113 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1095, !dbg !18
  %address.local1114 = or disjoint i64 %address.local1098, 81, !dbg !18
  %address.local1115 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1113, i64 %address.local1114, !dbg !18
  store half %1741, ptr addrspace(1) %address.local1115, align 2, !dbg !18
  %address.local1116 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1095, !dbg !18
  %address.local1117 = or disjoint i64 %address.local1098, 82, !dbg !18
  %address.local1118 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1116, i64 %address.local1117, !dbg !18
  store half %1740, ptr addrspace(1) %address.local1118, align 2, !dbg !18
  %address.local1119 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1095, !dbg !18
  %address.local1120 = or disjoint i64 %address.local1098, 83, !dbg !18
  %address.local1121 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1119, i64 %address.local1120, !dbg !18
  store half %1739, ptr addrspace(1) %address.local1121, align 2, !dbg !18
  %address.local1122 = shl i64 %address.local1093, 7, !dbg !18
  %address.local1123 = add i64 %address.local1122, 8192, !dbg !18
  %address.local1124 = add nuw nsw i64 %1752, %address.local1123, !dbg !18
  %address.local1125 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1124, !dbg !18
  %address.local1126 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1125, i64 %address.local1099, !dbg !18
  store half %1733, ptr addrspace(1) %address.local1126, align 2, !dbg !18
  %address.local1127 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1124, !dbg !18
  %address.local1128 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1127, i64 %address.local1102, !dbg !18
  store half %1732, ptr addrspace(1) %address.local1128, align 2, !dbg !18
  %address.local1129 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1124, !dbg !18
  %address.local1130 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1129, i64 %address.local1105, !dbg !18
  store half %1731, ptr addrspace(1) %address.local1130, align 2, !dbg !18
  %address.local1131 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1124, !dbg !18
  %address.local1132 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1131, i64 %address.local1108, !dbg !18
  store half %1730, ptr addrspace(1) %address.local1132, align 2, !dbg !18
  %address.local1133 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1124, !dbg !18
  %address.local1134 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1133, i64 %address.local1111, !dbg !18
  store half %1724, ptr addrspace(1) %address.local1134, align 2, !dbg !18
  %address.local1135 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1124, !dbg !18
  %address.local1136 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1135, i64 %address.local1114, !dbg !18
  store half %1723, ptr addrspace(1) %address.local1136, align 2, !dbg !18
  %address.local1137 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1124, !dbg !18
  %address.local1138 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1137, i64 %address.local1117, !dbg !18
  store half %1722, ptr addrspace(1) %address.local1138, align 2, !dbg !18
  %address.local1139 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1124, !dbg !18
  %address.local1140 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1139, i64 %address.local1120, !dbg !18
  store half %1721, ptr addrspace(1) %address.local1140, align 2, !dbg !18
  %1754 = insertelement <4 x float> poison, float %1528, i64 0, !dbg !16
  %1755 = insertelement <4 x float> %1754, float %1534, i64 1, !dbg !16
  %1756 = insertelement <4 x float> %1755, float %1529, i64 2, !dbg !16
  %1757 = insertelement <4 x float> %1756, float %1535, i64 3, !dbg !16
  %row.reciprocal53 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1507, !dbg !16
  %normalized54 = fmul <4 x float> %1757, %row.reciprocal53, !dbg !16
  %1758 = fptrunc <4 x float> %normalized54 to <4 x half>, !dbg !17
  %1759 = extractelement <4 x half> %1758, i64 3, !dbg !17
  %1760 = extractelement <4 x half> %1758, i64 2, !dbg !17
  %1761 = extractelement <4 x half> %1758, i64 1, !dbg !17
  %1762 = extractelement <4 x half> %1758, i64 0, !dbg !17
  %1763 = insertelement <4 x float> poison, float %1418, i64 0, !dbg !16
  %1764 = insertelement <4 x float> %1763, float %1424, i64 1, !dbg !16
  %1765 = insertelement <4 x float> %1764, float %1419, i64 2, !dbg !16
  %1766 = insertelement <4 x float> %1765, float %1425, i64 3, !dbg !16
  %normalized55 = fmul <4 x float> %1766, %row.reciprocal53, !dbg !16
  %1767 = fptrunc <4 x float> %normalized55 to <4 x half>, !dbg !17
  %1768 = extractelement <4 x half> %1767, i64 3, !dbg !17
  %1769 = extractelement <4 x half> %1767, i64 2, !dbg !17
  %1770 = extractelement <4 x half> %1767, i64 1, !dbg !17
  %1771 = extractelement <4 x half> %1767, i64 0, !dbg !17
  %1772 = insertelement <4 x float> poison, float %1311, i64 0, !dbg !16
  %1773 = insertelement <4 x float> %1772, float %1317, i64 1, !dbg !16
  %1774 = insertelement <4 x float> %1773, float %1312, i64 2, !dbg !16
  %1775 = insertelement <4 x float> %1774, float %1318, i64 3, !dbg !16
  %row.reciprocal56 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1290, !dbg !16
  %normalized57 = fmul <4 x float> %1775, %row.reciprocal56, !dbg !16
  %1776 = fptrunc <4 x float> %normalized57 to <4 x half>, !dbg !17
  %1777 = extractelement <4 x half> %1776, i64 3, !dbg !17
  %1778 = extractelement <4 x half> %1776, i64 2, !dbg !17
  %1779 = extractelement <4 x half> %1776, i64 1, !dbg !17
  %1780 = extractelement <4 x half> %1776, i64 0, !dbg !17
  %1781 = insertelement <4 x float> poison, float %1201, i64 0, !dbg !16
  %1782 = insertelement <4 x float> %1781, float %1207, i64 1, !dbg !16
  %1783 = insertelement <4 x float> %1782, float %1202, i64 2, !dbg !16
  %1784 = insertelement <4 x float> %1783, float %1208, i64 3, !dbg !16
  %normalized58 = fmul <4 x float> %1784, %row.reciprocal56, !dbg !16
  %1785 = fptrunc <4 x float> %normalized58 to <4 x half>, !dbg !17
  %1786 = extractelement <4 x half> %1785, i64 3, !dbg !17
  %1787 = extractelement <4 x half> %1785, i64 2, !dbg !17
  %1788 = extractelement <4 x half> %1785, i64 1, !dbg !17
  %1789 = extractelement <4 x half> %1785, i64 0, !dbg !17
  %1790 = shl nuw nsw i64 %18, 19, !dbg !18
  %address.thread1141 = call i32 asm sideeffect "", "=v,0"(i32 %25) #9, !dbg !18, !range !32, !frisk.register_lifetimes !1
  %address.local1142 = zext nneg i32 %address.thread1141 to i64, !dbg !8
  %address.local1143 = add nuw i64 %19, %address.local1142, !dbg !18
  %1791 = lshr i64 %address.local1142, 2, !dbg !18
  %address.local1145 = and i64 %1791, 48, !dbg !18
  %address.local1146 = add i64 %address.local1143, %address.local1145, !dbg !18
  %address.local1147 = and i64 %address.local1142, 240, !dbg !18
  %address.local1148 = sub i64 %address.local1146, %address.local1147, !dbg !18
  %address.local1149 = add i64 %address.local1148, %20, !dbg !18
  %address.local1150 = shl nuw nsw i64 %address.local1149, 7, !dbg !18
  %address.local1151 = add nuw nsw i64 %1790, %address.local1150, !dbg !18
  %address.local1152 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1151, !dbg !18
  %address.local1153 = lshr i64 %address.local1142, 2, !dbg !10
  %address.local1154 = and i64 %address.local1153, 12, !dbg !10
  %address.local1155 = or disjoint i64 %address.local1154, 96, !dbg !18
  %address.local1156 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1152, i64 %address.local1155, !dbg !18
  store half %1789, ptr addrspace(1) %address.local1156, align 2, !dbg !18
  %address.local1157 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1151, !dbg !18
  %address.local1158 = or disjoint i64 %address.local1154, 97, !dbg !18
  %address.local1159 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1157, i64 %address.local1158, !dbg !18
  store half %1788, ptr addrspace(1) %address.local1159, align 2, !dbg !18
  %address.local1160 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1151, !dbg !18
  %address.local1161 = or disjoint i64 %address.local1154, 98, !dbg !18
  %address.local1162 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1160, i64 %address.local1161, !dbg !18
  store half %1787, ptr addrspace(1) %address.local1162, align 2, !dbg !18
  %address.local1163 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1151, !dbg !18
  %address.local1164 = or disjoint i64 %address.local1154, 99, !dbg !18
  %address.local1165 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1163, i64 %address.local1164, !dbg !18
  store half %1786, ptr addrspace(1) %address.local1165, align 2, !dbg !18
  %address.local1166 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1151, !dbg !18
  %address.local1167 = or disjoint i64 %address.local1154, 112, !dbg !18
  %address.local1168 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1166, i64 %address.local1167, !dbg !18
  store half %1780, ptr addrspace(1) %address.local1168, align 2, !dbg !18
  %address.local1169 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1151, !dbg !18
  %address.local1170 = or disjoint i64 %address.local1154, 113, !dbg !18
  %address.local1171 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1169, i64 %address.local1170, !dbg !18
  store half %1779, ptr addrspace(1) %address.local1171, align 2, !dbg !18
  %address.local1172 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1151, !dbg !18
  %address.local1173 = or disjoint i64 %address.local1154, 114, !dbg !18
  %address.local1174 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1172, i64 %address.local1173, !dbg !18
  store half %1778, ptr addrspace(1) %address.local1174, align 2, !dbg !18
  %address.local1175 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1151, !dbg !18
  %address.local1176 = or i64 %address.local1153, 115, !dbg !18
  %address.local1177 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1175, i64 %address.local1176, !dbg !18
  store half %1777, ptr addrspace(1) %address.local1177, align 2, !dbg !18
  %address.local1178 = shl i64 %address.local1149, 7, !dbg !18
  %address.local1179 = add i64 %address.local1178, 8192, !dbg !18
  %address.local1180 = add nuw nsw i64 %1790, %address.local1179, !dbg !18
  %address.local1181 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1180, !dbg !18
  %address.local1182 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1181, i64 %address.local1155, !dbg !18
  store half %1771, ptr addrspace(1) %address.local1182, align 2, !dbg !18
  %address.local1183 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1180, !dbg !18
  %address.local1184 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1183, i64 %address.local1158, !dbg !18
  store half %1770, ptr addrspace(1) %address.local1184, align 2, !dbg !18
  %address.local1185 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1180, !dbg !18
  %address.local1186 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1185, i64 %address.local1161, !dbg !18
  store half %1769, ptr addrspace(1) %address.local1186, align 2, !dbg !18
  %address.local1187 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1180, !dbg !18
  %address.local1188 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1187, i64 %address.local1164, !dbg !18
  store half %1768, ptr addrspace(1) %address.local1188, align 2, !dbg !18
  %address.local1189 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1180, !dbg !18
  %address.local1190 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1189, i64 %address.local1167, !dbg !18
  store half %1762, ptr addrspace(1) %address.local1190, align 2, !dbg !18
  %address.local1191 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1180, !dbg !18
  %address.local1192 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1191, i64 %address.local1170, !dbg !18
  store half %1761, ptr addrspace(1) %address.local1192, align 2, !dbg !18
  %address.local1193 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1180, !dbg !18
  %address.local1194 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1193, i64 %address.local1173, !dbg !18
  store half %1760, ptr addrspace(1) %address.local1194, align 2, !dbg !18
  %address.local1195 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1180, !dbg !18
  %address.local1196 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1195, i64 %address.local1176, !dbg !18
  store half %1759, ptr addrspace(1) %address.local1196, align 2, !dbg !18
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
attributes #11 = { memory(inaccessiblemem: readwrite) }

!llvm.dbg.cu = !{!3}
!llvm.module.flags = !{!31}

!0 = !DIFile(filename: "test_input_spidD_noncasual.mlir", directory: "/data2/xsl/DeepGenGraph/3rd/deepgengraph/test")
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
