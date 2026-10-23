; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/online_softmax_native_acc/export/run-20261023-182033-aeo9_47b/input-0001-test_input_1x32x4096x128-128-128-32-online/retained32-baseline/input.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/online_softmax_native_acc/export/run-20261023-182033-aeo9_47b/input-0001-test_input_1x32x4096x128-128-128-32-online/retained32-baseline/input.mlir"

@shm_0 = addrspace(3) global [16384 x i8] undef, align 16

; Function Attrs: convergent willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

declare float @__ocml_exp2_f32(float)

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3) #1 !dbg !4 !reqd_work_group_size !37 {
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
  %31 = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !8, !range !39
  %32 = zext nneg i32 %31 to i64, !dbg !8
  %33 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %30, !dbg !8
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !8, !range !39, !frisk.register_lifetimes !1
  %address.local = zext nneg i32 %address.thread to i64, !dbg !8
  %address.local339 = shl nuw nsw i64 %address.local, 3, !dbg !8
  %address.local340 = and i64 %address.local339, 1536, !dbg !8
  %address.local341 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local340, !dbg !8
  %address.local342 = ptrtoint ptr addrspace(3) %address.local341 to i64, !dbg !8
  %address.local343 = trunc i64 %address.local342 to i32, !dbg !8
  %34 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local343), !dbg !8
  %35 = zext i32 %34 to i64, !dbg !8
  %36 = inttoptr i64 %35 to ptr addrspace(3), !dbg !8
  %address.local344 = shl nuw nsw i64 %address.local, 7, !dbg !8
  %address.local345 = and i64 %address.local344, 16256, !dbg !8
  %address.local346 = lshr i64 %address.local, 4, !dbg !8
  %address.local347 = and i64 %address.local346, 8, !dbg !8
  %address.local348 = or disjoint i64 %address.local345, %address.local347, !dbg !8
  %address.local349 = getelementptr [2 x i8], ptr addrspace(1) %33, i64 %address.local348, !dbg !8
  %37 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %38 = ptrtoint ptr addrspace(1) %address.local349 to i64, !dbg !8
  %39 = sub i64 %38, %37, !dbg !8
  %dma.buffer.offset1612 = trunc i64 %39 to i32, !dbg !8
  %40 = trunc i64 %37 to i32, !dbg !8
  %41 = insertelement <4 x i32> poison, i32 %40, i64 0, !dbg !8
  %42 = lshr i64 %37, 32, !dbg !8
  %43 = trunc i64 %42 to i32, !dbg !8
  %44 = insertelement <4 x i32> %41, i32 %43, i64 1, !dbg !8
  %45 = insertelement <4 x i32> %44, i32 33554432, i64 2, !dbg !8
  %dma.buffer.rsrc1613 = insertelement <4 x i32> %45, i32 131072, i64 3, !dbg !8
  %46 = ptrtoint ptr addrspace(3) %36 to i32, !dbg !8
  %dma.buffer.lds1614 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %46), !dbg !8
  %47 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1614, i32 %dma.buffer.offset1612, <4 x i32> %dma.buffer.rsrc1613) #10, !dbg !8
  %address.local350 = or disjoint i64 %address.local, 256, !dbg !8
  %address.local351 = shl nuw nsw i64 %address.local350, 3, !dbg !8
  %address.local352 = and i64 %address.local351, 3584, !dbg !8
  %address.local353 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local352, !dbg !8
  %address.local354 = ptrtoint ptr addrspace(3) %address.local353 to i64, !dbg !8
  %address.local355 = trunc i64 %address.local354 to i32, !dbg !8
  %48 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local355), !dbg !8
  %49 = zext i32 %48 to i64, !dbg !8
  %50 = inttoptr i64 %49 to ptr addrspace(3), !dbg !8
  %address.local356 = shl nuw nsw i64 %address.local, 7, !dbg !8
  %address.local357 = and i64 %address.local356, 16256, !dbg !8
  %address.local358 = lshr i64 %address.local350, 4, !dbg !8
  %address.local359 = and i64 %address.local358, 24, !dbg !8
  %address.local360 = or disjoint i64 %address.local357, %address.local359, !dbg !8
  %address.local361 = getelementptr [2 x i8], ptr addrspace(1) %33, i64 %address.local360, !dbg !8
  %51 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %52 = ptrtoint ptr addrspace(1) %address.local361 to i64, !dbg !8
  %53 = sub i64 %52, %51, !dbg !8
  %dma.buffer.offset1609 = trunc i64 %53 to i32, !dbg !8
  %54 = trunc i64 %51 to i32, !dbg !8
  %55 = insertelement <4 x i32> poison, i32 %54, i64 0, !dbg !8
  %56 = lshr i64 %51, 32, !dbg !8
  %57 = trunc i64 %56 to i32, !dbg !8
  %58 = insertelement <4 x i32> %55, i32 %57, i64 1, !dbg !8
  %59 = insertelement <4 x i32> %58, i32 33554432, i64 2, !dbg !8
  %dma.buffer.rsrc1610 = insertelement <4 x i32> %59, i32 131072, i64 3, !dbg !8
  %60 = ptrtoint ptr addrspace(3) %50 to i32, !dbg !8
  %dma.buffer.lds1611 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %60), !dbg !8
  %61 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1611, i32 %dma.buffer.offset1609, <4 x i32> %dma.buffer.rsrc1610) #10, !dbg !8
  %62 = call i32 @llvm.amdgcn.workgroup.id.z(), !dbg !5
  %63 = sext i32 %62 to i64, !dbg !5
  %64 = shl i64 %15, 7, !dbg !10
  %65 = lshr i64 %32, 6, !dbg !11
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %66 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %30, !dbg !8
  %67 = getelementptr i8, ptr addrspace(1) %66, i64 64, !dbg !8
  %address.local362 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local340, !dbg !8
  %address.local363 = ptrtoint ptr addrspace(3) %address.local362 to i64, !dbg !8
  %address.local364 = trunc i64 %address.local363 to i32, !dbg !8
  %68 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local364), !dbg !8
  %69 = zext i32 %68 to i64, !dbg !8
  %70 = inttoptr i64 %69 to ptr addrspace(3), !dbg !8
  %address.local365 = getelementptr [2 x i8], ptr addrspace(1) %67, i64 %address.local348, !dbg !8
  %71 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %72 = ptrtoint ptr addrspace(1) %address.local365 to i64, !dbg !8
  %73 = sub i64 %72, %71, !dbg !8
  %dma.buffer.offset1606 = trunc i64 %73 to i32, !dbg !8
  %74 = trunc i64 %71 to i32, !dbg !8
  %75 = insertelement <4 x i32> poison, i32 %74, i64 0, !dbg !8
  %76 = lshr i64 %71, 32, !dbg !8
  %77 = trunc i64 %76 to i32, !dbg !8
  %78 = insertelement <4 x i32> %75, i32 %77, i64 1, !dbg !8
  %79 = insertelement <4 x i32> %78, i32 33554432, i64 2, !dbg !8
  %dma.buffer.rsrc1607 = insertelement <4 x i32> %79, i32 131072, i64 3, !dbg !8
  %80 = ptrtoint ptr addrspace(3) %70 to i32, !dbg !8
  %dma.buffer.lds1608 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %80), !dbg !8
  %81 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1608, i32 %dma.buffer.offset1606, <4 x i32> %dma.buffer.rsrc1607) #10, !dbg !8
  %address.local366 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local352, !dbg !8
  %address.local367 = ptrtoint ptr addrspace(3) %address.local366 to i64, !dbg !8
  %address.local368 = trunc i64 %address.local367 to i32, !dbg !8
  %82 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local368), !dbg !8
  %83 = zext i32 %82 to i64, !dbg !8
  %84 = inttoptr i64 %83 to ptr addrspace(3), !dbg !8
  %address.local369 = getelementptr [2 x i8], ptr addrspace(1) %67, i64 %address.local360, !dbg !8
  %85 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %86 = ptrtoint ptr addrspace(1) %address.local369 to i64, !dbg !8
  %87 = sub i64 %86, %85, !dbg !8
  %dma.buffer.offset1603 = trunc i64 %87 to i32, !dbg !8
  %88 = trunc i64 %85 to i32, !dbg !8
  %89 = insertelement <4 x i32> poison, i32 %88, i64 0, !dbg !8
  %90 = lshr i64 %85, 32, !dbg !8
  %91 = trunc i64 %90 to i32, !dbg !8
  %92 = insertelement <4 x i32> %89, i32 %91, i64 1, !dbg !8
  %93 = insertelement <4 x i32> %92, i32 33554432, i64 2, !dbg !8
  %dma.buffer.rsrc1604 = insertelement <4 x i32> %93, i32 131072, i64 3, !dbg !8
  %94 = ptrtoint ptr addrspace(3) %84 to i32, !dbg !8
  %dma.buffer.lds1605 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %94), !dbg !8
  %95 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1605, i32 %dma.buffer.offset1603, <4 x i32> %dma.buffer.rsrc1604) #10, !dbg !8
  %address.local370 = lshr i32 %address.thread, 2, !dbg !11
  %address.local371 = and i32 %address.local370, 4, !dbg !11
  %address.local372 = shl nuw nsw i32 %address.thread, 1, !dbg !11
  %address.local373 = and i32 %address.local372, 384, !dbg !11
  %address.local374 = or disjoint i32 %address.local371, %address.local373, !dbg !11
  %address.local375 = shl nuw nsw i32 %address.thread, 3, !dbg !11
  %address.local376 = and i32 %address.local375, 120, !dbg !11
  %address.local377 = or disjoint i32 %address.local374, %address.local376, !dbg !11
  %address.local378 = shl nuw nsw i32 %address.thread, 5, !dbg !11
  %address.local379 = and i32 %address.local378, 1024, !dbg !11
  %address.local380 = or disjoint i32 %address.local377, %address.local379, !dbg !11
  %address.local381 = zext nneg i32 %address.local380 to i64, !dbg !11
  %address.local382 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local381, !dbg !11
  %96 = load <4 x half>, ptr addrspace(3) %address.local382, align 8, !dbg !11
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local382, i64 4096, !dbg !11
  %97 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !11
  %98 = fmul <4 x half> %96, splat (half 0xH3015), !dbg !11
  %lds.const.gep1413 = getelementptr i8, ptr addrspace(3) %address.local382, i64 1024, !dbg !11
  %99 = load <4 x half>, ptr addrspace(3) %lds.const.gep1413, align 8, !dbg !11
  %100 = fmul <4 x half> %97, splat (half 0xH3015), !dbg !11
  %lds.const.gep1414 = getelementptr i8, ptr addrspace(3) %address.local382, i64 5120, !dbg !11
  %101 = load <4 x half>, ptr addrspace(3) %lds.const.gep1414, align 8, !dbg !11
  %102 = fmul <4 x half> %99, splat (half 0xH3015), !dbg !11
  %103 = fmul <4 x half> %101, splat (half 0xH3015), !dbg !11
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %104 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %30, !dbg !8
  %105 = getelementptr i8, ptr addrspace(1) %104, i64 128, !dbg !8
  %106 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local343), !dbg !8
  %107 = zext i32 %106 to i64, !dbg !8
  %108 = inttoptr i64 %107 to ptr addrspace(3), !dbg !8
  %address.local425 = getelementptr [2 x i8], ptr addrspace(1) %105, i64 %address.local348, !dbg !8
  %109 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %110 = ptrtoint ptr addrspace(1) %address.local425 to i64, !dbg !8
  %111 = sub i64 %110, %109, !dbg !8
  %dma.buffer.offset1600 = trunc i64 %111 to i32, !dbg !8
  %112 = trunc i64 %109 to i32, !dbg !8
  %113 = insertelement <4 x i32> poison, i32 %112, i64 0, !dbg !8
  %114 = lshr i64 %109, 32, !dbg !8
  %115 = trunc i64 %114 to i32, !dbg !8
  %116 = insertelement <4 x i32> %113, i32 %115, i64 1, !dbg !8
  %117 = insertelement <4 x i32> %116, i32 33554432, i64 2, !dbg !8
  %dma.buffer.rsrc1601 = insertelement <4 x i32> %117, i32 131072, i64 3, !dbg !8
  %118 = ptrtoint ptr addrspace(3) %108 to i32, !dbg !8
  %dma.buffer.lds1602 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %118), !dbg !8
  %119 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1602, i32 %dma.buffer.offset1600, <4 x i32> %dma.buffer.rsrc1601) #10, !dbg !8
  %120 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local355), !dbg !8
  %121 = zext i32 %120 to i64, !dbg !8
  %122 = inttoptr i64 %121 to ptr addrspace(3), !dbg !8
  %address.local426 = getelementptr [2 x i8], ptr addrspace(1) %105, i64 %address.local360, !dbg !8
  %123 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %124 = ptrtoint ptr addrspace(1) %address.local426 to i64, !dbg !8
  %125 = sub i64 %124, %123, !dbg !8
  %dma.buffer.offset1597 = trunc i64 %125 to i32, !dbg !8
  %126 = trunc i64 %123 to i32, !dbg !8
  %127 = insertelement <4 x i32> poison, i32 %126, i64 0, !dbg !8
  %128 = lshr i64 %123, 32, !dbg !8
  %129 = trunc i64 %128 to i32, !dbg !8
  %130 = insertelement <4 x i32> %127, i32 %129, i64 1, !dbg !8
  %131 = insertelement <4 x i32> %130, i32 33554432, i64 2, !dbg !8
  %dma.buffer.rsrc1598 = insertelement <4 x i32> %131, i32 131072, i64 3, !dbg !8
  %132 = ptrtoint ptr addrspace(3) %122 to i32, !dbg !8
  %dma.buffer.lds1599 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %132), !dbg !8
  %133 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1599, i32 %dma.buffer.offset1597, <4 x i32> %dma.buffer.rsrc1598) #10, !dbg !8
  %lds.const.gep1415 = getelementptr i8, ptr addrspace(3) %address.local382, i64 8192, !dbg !11
  %134 = load <4 x half>, ptr addrspace(3) %lds.const.gep1415, align 8, !dbg !11
  %lds.const.gep1416 = getelementptr i8, ptr addrspace(3) %address.local382, i64 12288, !dbg !11
  %135 = load <4 x half>, ptr addrspace(3) %lds.const.gep1416, align 8, !dbg !11
  %lds.const.gep1417 = getelementptr i8, ptr addrspace(3) %address.local382, i64 9216, !dbg !11
  %136 = load <4 x half>, ptr addrspace(3) %lds.const.gep1417, align 8, !dbg !11
  %137 = fmul <4 x half> %134, splat (half 0xH3015), !dbg !11
  %lds.const.gep1418 = getelementptr i8, ptr addrspace(3) %address.local382, i64 13312, !dbg !11
  %138 = load <4 x half>, ptr addrspace(3) %lds.const.gep1418, align 8, !dbg !11
  %139 = fmul <4 x half> %135, splat (half 0xH3015), !dbg !11
  %140 = fmul <4 x half> %136, splat (half 0xH3015), !dbg !11
  %141 = fmul <4 x half> %138, splat (half 0xH3015), !dbg !11
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %142 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %30, !dbg !8
  %143 = getelementptr i8, ptr addrspace(1) %142, i64 192, !dbg !8
  %144 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local364), !dbg !8
  %145 = zext i32 %144 to i64, !dbg !8
  %146 = inttoptr i64 %145 to ptr addrspace(3), !dbg !8
  %address.local431 = getelementptr [2 x i8], ptr addrspace(1) %143, i64 %address.local348, !dbg !8
  %147 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %148 = ptrtoint ptr addrspace(1) %address.local431 to i64, !dbg !8
  %149 = sub i64 %148, %147, !dbg !8
  %dma.buffer.offset1594 = trunc i64 %149 to i32, !dbg !8
  %150 = trunc i64 %147 to i32, !dbg !8
  %151 = insertelement <4 x i32> poison, i32 %150, i64 0, !dbg !8
  %152 = lshr i64 %147, 32, !dbg !8
  %153 = trunc i64 %152 to i32, !dbg !8
  %154 = insertelement <4 x i32> %151, i32 %153, i64 1, !dbg !8
  %155 = insertelement <4 x i32> %154, i32 33554432, i64 2, !dbg !8
  %dma.buffer.rsrc1595 = insertelement <4 x i32> %155, i32 131072, i64 3, !dbg !8
  %156 = ptrtoint ptr addrspace(3) %146 to i32, !dbg !8
  %dma.buffer.lds1596 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %156), !dbg !8
  %157 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1596, i32 %dma.buffer.offset1594, <4 x i32> %dma.buffer.rsrc1595) #10, !dbg !8
  %158 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local368), !dbg !8
  %159 = zext i32 %158 to i64, !dbg !8
  %160 = inttoptr i64 %159 to ptr addrspace(3), !dbg !8
  %address.local432 = getelementptr [2 x i8], ptr addrspace(1) %143, i64 %address.local360, !dbg !8
  %161 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %162 = ptrtoint ptr addrspace(1) %address.local432 to i64, !dbg !8
  %163 = sub i64 %162, %161, !dbg !8
  %dma.buffer.offset1591 = trunc i64 %163 to i32, !dbg !8
  %164 = trunc i64 %161 to i32, !dbg !8
  %165 = insertelement <4 x i32> poison, i32 %164, i64 0, !dbg !8
  %166 = lshr i64 %161, 32, !dbg !8
  %167 = trunc i64 %166 to i32, !dbg !8
  %168 = insertelement <4 x i32> %165, i32 %167, i64 1, !dbg !8
  %169 = insertelement <4 x i32> %168, i32 33554432, i64 2, !dbg !8
  %dma.buffer.rsrc1592 = insertelement <4 x i32> %169, i32 131072, i64 3, !dbg !8
  %170 = ptrtoint ptr addrspace(3) %160 to i32, !dbg !8
  %dma.buffer.lds1593 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %170), !dbg !8
  %171 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1593, i32 %dma.buffer.offset1591, <4 x i32> %dma.buffer.rsrc1592) #10, !dbg !8
  %172 = load <4 x half>, ptr addrspace(3) %address.local382, align 8, !dbg !11
  %lds.const.gep1419 = getelementptr i8, ptr addrspace(3) %address.local382, i64 4096, !dbg !11
  %173 = load <4 x half>, ptr addrspace(3) %lds.const.gep1419, align 8, !dbg !11
  %lds.const.gep1420 = getelementptr i8, ptr addrspace(3) %address.local382, i64 1024, !dbg !11
  %174 = load <4 x half>, ptr addrspace(3) %lds.const.gep1420, align 8, !dbg !11
  %175 = fmul <4 x half> %172, splat (half 0xH3015), !dbg !11
  %lds.const.gep1421 = getelementptr i8, ptr addrspace(3) %address.local382, i64 5120, !dbg !11
  %176 = load <4 x half>, ptr addrspace(3) %lds.const.gep1421, align 8, !dbg !11
  %177 = fmul <4 x half> %173, splat (half 0xH3015), !dbg !11
  %178 = fmul <4 x half> %174, splat (half 0xH3015), !dbg !11
  %179 = fmul <4 x half> %176, splat (half 0xH3015), !dbg !11
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %lds.const.gep1422 = getelementptr i8, ptr addrspace(3) %address.local382, i64 8192, !dbg !11
  %180 = load <4 x half>, ptr addrspace(3) %lds.const.gep1422, align 8, !dbg !11
  %lds.const.gep1423 = getelementptr i8, ptr addrspace(3) %address.local382, i64 12288, !dbg !11
  %181 = load <4 x half>, ptr addrspace(3) %lds.const.gep1423, align 8, !dbg !11
  %lds.const.gep1424 = getelementptr i8, ptr addrspace(3) %address.local382, i64 9216, !dbg !11
  %182 = load <4 x half>, ptr addrspace(3) %lds.const.gep1424, align 8, !dbg !11
  %183 = fmul <4 x half> %180, splat (half 0xH3015), !dbg !11
  %lds.const.gep1425 = getelementptr i8, ptr addrspace(3) %address.local382, i64 13312, !dbg !11
  %184 = load <4 x half>, ptr addrspace(3) %lds.const.gep1425, align 8, !dbg !11
  %185 = fmul <4 x half> %181, splat (half 0xH3015), !dbg !11
  %186 = fmul <4 x half> %182, splat (half 0xH3015), !dbg !11
  %187 = fmul <4 x half> %184, splat (half 0xH3015), !dbg !11
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %188 = or disjoint i64 %64, 127, !dbg !13
  %189 = add i64 %64, 128, !dbg !13
  %190 = icmp uge i64 %189, %188, !dbg !13
  %191 = sub nsw i64 32, %14, !dbg !15
  %192 = shl i64 %16, 7, !dbg !16
  %193 = and i64 %192, 3968, !dbg !16
  %194 = shl nsw i64 %63, 12, !dbg !16
  %195 = or disjoint i64 %193, %194, !dbg !16
  %196 = lshr i64 %32, 2, !dbg !17
  %197 = shl nuw nsw i64 %32, 2, !dbg !17
  %198 = mul nsw i64 %196, -16, !dbg !17
  %199 = add nsw i64 %197, %198, !dbg !17
  %200 = and i32 %31, 1, !dbg !17
  %index.compact279 = zext nneg i32 %200 to i64, !dbg !17
  %201 = or disjoint i64 %199, %index.compact279, !dbg !17
  %202 = add nuw nsw i64 %201, 1, !dbg !17
  %203 = sdiv i64 %202, 2, !dbg !17
  br label %204, !dbg !7

204:                                              ; preds = %848, %4
  %205 = phi i64 [ %1781, %848 ], [ 0, %4 ], !dbg !7
  %206 = phi [2 x <8 x float>] [ %1484, %848 ], [ zeroinitializer, %4 ], !dbg !7
  %207 = phi [2 x <1 x float>] [ %1370, %848 ], [ zeroinitializer, %4 ], !dbg !7
  %208 = phi [2 x <1 x float>] [ %1768, %848 ], [ [<1 x float> splat (float 0xFFF0000000000000), <1 x float> splat (float 0xFFF0000000000000)], %4 ], !dbg !7
  %209 = phi [2 x <8 x float>] [ %1772, %848 ], [ zeroinitializer, %4 ], !dbg !7
  %210 = phi [2 x <8 x float>] [ %1776, %848 ], [ zeroinitializer, %4 ], !dbg !7
  %211 = phi [2 x <8 x float>] [ %1780, %848 ], [ zeroinitializer, %4 ], !dbg !7
  %212 = icmp slt i64 %205, %191, !dbg !7
  br i1 %212, label %213, label %1782, !dbg !7

213:                                              ; preds = %204
  %214 = shl nsw i64 %205, 7, !dbg !7
  %215 = lshr i64 %205, 12, !dbg !16
  %216 = add nuw nsw i64 %215, %16, !dbg !16
  %217 = lshr i64 %205, 5, !dbg !16
  %218 = and i64 %217, 127, !dbg !16
  %219 = shl i64 %216, 19, !dbg !16
  %220 = and i64 %219, 16252928, !dbg !16
  %221 = shl i64 %217, 19, !dbg !16
  %222 = shl i64 %205, 14, !dbg !16
  %223 = sub i64 %222, %221, !dbg !16
  %224 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %220, !dbg !16
  %225 = getelementptr [2 x i8], ptr addrspace(1) %224, i64 %218, !dbg !16
  %226 = getelementptr [2 x i8], ptr addrspace(1) %225, i64 %223, !dbg !16
  %address.thread433 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !16, !range !39, !frisk.register_lifetimes !1
  %address.local434 = zext nneg i32 %address.thread433 to i64, !dbg !8
  %address.local435 = shl nuw nsw i64 %address.local434, 3, !dbg !8
  %address.local436 = and i64 %address.local435, 1536, !dbg !8
  %address.local437 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local436, !dbg !8
  %address.local438 = ptrtoint ptr addrspace(3) %address.local437 to i64, !dbg !8
  %address.local439 = trunc i64 %address.local438 to i32, !dbg !8
  %227 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local439), !dbg !16
  %228 = zext i32 %227 to i64, !dbg !16
  %229 = inttoptr i64 %228 to ptr addrspace(3), !dbg !16
  %address.local440 = lshr i64 %address.local434, 4, !dbg !16
  %address.local441 = and i64 %address.local440, 8, !dbg !16
  %address.local442 = and i64 %address.local434, 123, !dbg !16
  %address.local443 = lshr i64 %address.local434, 2, !dbg !16
  %address.local444 = and i64 %address.local443, 4, !dbg !16
  %address.local445 = add nuw nsw i64 %address.local444, %address.local434, !dbg !16
  %address.local446 = and i64 %address.local445, 4, !dbg !16
  %address.local447 = or disjoint i64 %address.local442, %address.local446, !dbg !16
  %address.local448 = shl nuw nsw i64 %address.local447, 7, !dbg !16
  %address.local449 = or disjoint i64 %address.local441, %address.local448, !dbg !16
  %address.local450 = getelementptr [2 x i8], ptr addrspace(1) %226, i64 %address.local449, !dbg !16
  %230 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %231 = ptrtoint ptr addrspace(1) %address.local450 to i64, !dbg !16
  %232 = sub i64 %231, %230, !dbg !16
  %dma.buffer.offset1588 = trunc i64 %232 to i32, !dbg !16
  %233 = trunc i64 %230 to i32, !dbg !16
  %234 = insertelement <4 x i32> poison, i32 %233, i64 0, !dbg !16
  %235 = lshr i64 %230, 32, !dbg !16
  %236 = trunc i64 %235 to i32, !dbg !16
  %237 = insertelement <4 x i32> %234, i32 %236, i64 1, !dbg !16
  %238 = insertelement <4 x i32> %237, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1589 = insertelement <4 x i32> %238, i32 131072, i64 3, !dbg !16
  %239 = ptrtoint ptr addrspace(3) %229 to i32, !dbg !16
  %dma.buffer.lds1590 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %239), !dbg !16
  %240 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1590, i32 %dma.buffer.offset1588, <4 x i32> %dma.buffer.rsrc1589) #10, !dbg !16
  %address.local451 = or disjoint i64 %address.local434, 256, !dbg !8
  %address.local452 = shl nuw nsw i64 %address.local451, 3, !dbg !8
  %address.local453 = and i64 %address.local452, 3584, !dbg !8
  %address.local454 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local453, !dbg !8
  %address.local455 = ptrtoint ptr addrspace(3) %address.local454 to i64, !dbg !8
  %address.local456 = trunc i64 %address.local455 to i32, !dbg !8
  %241 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local456), !dbg !16
  %242 = zext i32 %241 to i64, !dbg !16
  %243 = inttoptr i64 %242 to ptr addrspace(3), !dbg !16
  %address.local457 = lshr i64 %address.local451, 4, !dbg !16
  %address.local458 = and i64 %address.local457, 24, !dbg !16
  %address.local459 = and i64 %address.local434, 123, !dbg !16
  %address.local460 = lshr i64 %address.local434, 2, !dbg !16
  %address.local461 = and i64 %address.local460, 4, !dbg !16
  %address.local462 = add nuw nsw i64 %address.local461, %address.local434, !dbg !16
  %address.local463 = and i64 %address.local462, 4, !dbg !16
  %address.local464 = or disjoint i64 %address.local459, %address.local463, !dbg !16
  %address.local465 = shl nuw nsw i64 %address.local464, 7, !dbg !16
  %address.local466 = or disjoint i64 %address.local458, %address.local465, !dbg !16
  %address.local467 = getelementptr [2 x i8], ptr addrspace(1) %226, i64 %address.local466, !dbg !16
  %244 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %245 = ptrtoint ptr addrspace(1) %address.local467 to i64, !dbg !16
  %246 = sub i64 %245, %244, !dbg !16
  %dma.buffer.offset1585 = trunc i64 %246 to i32, !dbg !16
  %247 = trunc i64 %244 to i32, !dbg !16
  %248 = insertelement <4 x i32> poison, i32 %247, i64 0, !dbg !16
  %249 = lshr i64 %244, 32, !dbg !16
  %250 = trunc i64 %249 to i32, !dbg !16
  %251 = insertelement <4 x i32> %248, i32 %250, i64 1, !dbg !16
  %252 = insertelement <4 x i32> %251, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1586 = insertelement <4 x i32> %252, i32 131072, i64 3, !dbg !16
  %253 = ptrtoint ptr addrspace(3) %243 to i32, !dbg !16
  %dma.buffer.lds1587 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %253), !dbg !16
  %254 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1587, i32 %dma.buffer.offset1585, <4 x i32> %dma.buffer.rsrc1586) #10, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %255 = add nuw i64 %205, 1024, !dbg !16
  %256 = icmp slt i64 %255, 0, !dbg !16
  %257 = sub nuw i64 -1025, %205, !dbg !16
  %258 = select i1 %256, i64 %257, i64 %255, !dbg !16
  %259 = sdiv i64 %258, 4096, !dbg !16
  %.lobit37 = ashr i64 %255, 63, !dbg !16
  %260 = xor i64 %259, %.lobit37, !dbg !16
  %261 = add nsw i64 %260, %16, !dbg !16
  %262 = add nsw i64 %217, %195, !dbg !16
  %263 = add nsw i64 %262, 32, !dbg !16
  %264 = icmp slt i64 %262, -32, !dbg !16
  %265 = sub nsw i64 -33, %262, !dbg !16
  %266 = select i1 %264, i64 %265, i64 %263, !dbg !16
  %267 = sdiv i64 %266, 128, !dbg !16
  %268 = sext i1 %264 to i64, !dbg !16
  %269 = xor i64 %267, %268, !dbg !16
  %270 = shl nsw i64 %269, 7, !dbg !16
  %271 = sub i64 %195, %270, !dbg !16
  %272 = shl i64 %261, 19, !dbg !16
  %273 = and i64 %272, 16252928, !dbg !16
  %274 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %273, !dbg !16
  %275 = getelementptr [2 x i8], ptr addrspace(1) %274, i64 %217, !dbg !16
  %276 = getelementptr [2 x i8], ptr addrspace(1) %275, i64 %271, !dbg !16
  %277 = getelementptr i8, ptr addrspace(1) %276, i64 64, !dbg !16
  %278 = getelementptr [2 x i8], ptr addrspace(1) %277, i64 %223, !dbg !16
  %address.local468 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local436, !dbg !8
  %address.local469 = ptrtoint ptr addrspace(3) %address.local468 to i64, !dbg !8
  %address.local470 = trunc i64 %address.local469 to i32, !dbg !8
  %279 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local470), !dbg !16
  %280 = zext i32 %279 to i64, !dbg !16
  %281 = inttoptr i64 %280 to ptr addrspace(3), !dbg !16
  %address.local471 = getelementptr [2 x i8], ptr addrspace(1) %278, i64 %address.local449, !dbg !16
  %282 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %283 = ptrtoint ptr addrspace(1) %address.local471 to i64, !dbg !16
  %284 = sub i64 %283, %282, !dbg !16
  %dma.buffer.offset1582 = trunc i64 %284 to i32, !dbg !16
  %285 = trunc i64 %282 to i32, !dbg !16
  %286 = insertelement <4 x i32> poison, i32 %285, i64 0, !dbg !16
  %287 = lshr i64 %282, 32, !dbg !16
  %288 = trunc i64 %287 to i32, !dbg !16
  %289 = insertelement <4 x i32> %286, i32 %288, i64 1, !dbg !16
  %290 = insertelement <4 x i32> %289, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1583 = insertelement <4 x i32> %290, i32 131072, i64 3, !dbg !16
  %291 = ptrtoint ptr addrspace(3) %281 to i32, !dbg !16
  %dma.buffer.lds1584 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %291), !dbg !16
  %292 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1584, i32 %dma.buffer.offset1582, <4 x i32> %dma.buffer.rsrc1583) #10, !dbg !16
  %address.local472 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local453, !dbg !8
  %address.local473 = ptrtoint ptr addrspace(3) %address.local472 to i64, !dbg !8
  %address.local474 = trunc i64 %address.local473 to i32, !dbg !8
  %293 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local474), !dbg !16
  %294 = zext i32 %293 to i64, !dbg !16
  %295 = inttoptr i64 %294 to ptr addrspace(3), !dbg !16
  %address.local475 = getelementptr [2 x i8], ptr addrspace(1) %278, i64 %address.local466, !dbg !16
  %296 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %297 = ptrtoint ptr addrspace(1) %address.local475 to i64, !dbg !16
  %298 = sub i64 %297, %296, !dbg !16
  %dma.buffer.offset1579 = trunc i64 %298 to i32, !dbg !16
  %299 = trunc i64 %296 to i32, !dbg !16
  %300 = insertelement <4 x i32> poison, i32 %299, i64 0, !dbg !16
  %301 = lshr i64 %296, 32, !dbg !16
  %302 = trunc i64 %301 to i32, !dbg !16
  %303 = insertelement <4 x i32> %300, i32 %302, i64 1, !dbg !16
  %304 = insertelement <4 x i32> %303, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1580 = insertelement <4 x i32> %304, i32 131072, i64 3, !dbg !16
  %305 = ptrtoint ptr addrspace(3) %295 to i32, !dbg !16
  %dma.buffer.lds1581 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %305), !dbg !16
  %306 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1581, i32 %dma.buffer.offset1579, <4 x i32> %dma.buffer.rsrc1580) #10, !dbg !16
  %address.local476 = mul nuw nsw i64 %address.local434, 160, !dbg !17
  %address.local477 = lshr i64 %address.local434, 2, !dbg !17
  %address.local478 = mul nsw i64 %address.local477, -640, !dbg !17
  %address.local479 = add nsw i64 %address.local476, %address.local478, !dbg !17
  %address.local480 = shl nuw nsw i64 %address.local434, 1, !dbg !17
  %address.local481 = and i64 %address.local480, 24, !dbg !17
  %address.local482 = or disjoint i64 %address.local479, %address.local481, !dbg !17
  %address.local483 = shl i64 %address.local434, 62, !dbg !17
  %address.local484 = ashr exact i64 %address.local483, 57, !dbg !17
  %address.local485 = and i64 %address.local484, -64, !dbg !17
  %address.local486 = add nsw i64 %address.local482, %address.local485, !dbg !17
  %address.local487 = lshr i64 %address.local434, 2, !dbg !11
  %address.local488 = and i64 %address.local487, 12, !dbg !11
  %address.local489 = add nsw i64 %address.local486, %address.local488, !dbg !17
  %address.local490 = and i64 %address.local434, 63, !dbg !11
  %address.local491 = lshr i64 %address.local490, 5, !dbg !11
  %address.local492 = mul nuw nsw i64 %address.local491, 1016, !dbg !11
  %address.local493 = add i64 %address.local489, %address.local492, !dbg !17
  %address.local494 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local493, !dbg !17
  %307 = load <4 x half>, ptr addrspace(3) %address.local494, align 8, !dbg !17
  %308 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %307, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1426 = getelementptr i8, ptr addrspace(3) %address.local494, i64 4096, !dbg !17
  %309 = load <4 x half>, ptr addrspace(3) %lds.const.gep1426, align 8, !dbg !17
  %address.local505 = shl nuw nsw i32 %address.thread433, 1, !dbg !17
  %address.local506 = and i32 %address.local505, 24, !dbg !17
  %address.local507 = shl nuw nsw i32 %address.thread433, 8, !dbg !17
  %address.local508 = and i32 %address.local507, 768, !dbg !17
  %address.local509 = or disjoint i32 %address.local506, %address.local508, !dbg !17
  %address.local510 = shl nuw nsw i32 %address.thread433, 5, !dbg !17
  %address.local511 = and i32 %address.local510, 96, !dbg !17
  %address.local512 = or disjoint i32 %address.local509, %address.local511, !dbg !17
  %address.local513 = zext nneg i32 %address.local512 to i64, !dbg !17
  %address.base = call i64 asm sideeffect "", "=v,0"(i64 %203) #9, !dbg !17, !frisk.register_lifetimes !1
  %address.local514 = lshr i32 %address.thread433, 1, !dbg !17
  %address.local515 = and i32 %address.local514, 1, !dbg !17
  %address.local516 = zext nneg i32 %address.local515 to i64, !dbg !17
  %address.local517 = add nsw i64 %address.base, %address.local516, !dbg !17
  %address.local518 = shl i64 %address.local517, 6, !dbg !17
  %address.local519 = sub i64 %address.local513, %address.local518, !dbg !17
  %address.local520 = add i64 %address.local519, %address.local488, !dbg !17
  %address.local521 = add i64 %address.local520, %address.local492, !dbg !17
  %address.local522 = add i64 %address.local521, 32, !dbg !17
  %address.local523 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local522, !dbg !17
  %310 = load <4 x half>, ptr addrspace(3) %address.local523, align 8, !dbg !17
  %311 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %309, <4 x float> %308) #9, !dbg !17
  %312 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %310, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1427 = getelementptr i8, ptr addrspace(3) %address.local523, i64 4096, !dbg !17
  %313 = load <4 x half>, ptr addrspace(3) %lds.const.gep1427, align 8, !dbg !17
  %lds.const.gep1428 = getelementptr i8, ptr addrspace(3) %address.local494, i64 128, !dbg !17
  %314 = load <4 x half>, ptr addrspace(3) %lds.const.gep1428, align 8, !dbg !17
  %315 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %313, <4 x float> %312) #9, !dbg !17
  %316 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %314, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1429 = getelementptr i8, ptr addrspace(3) %address.local494, i64 4224, !dbg !17
  %317 = load <4 x half>, ptr addrspace(3) %lds.const.gep1429, align 8, !dbg !17
  %lds.const.gep1430 = getelementptr i8, ptr addrspace(3) %address.local523, i64 128, !dbg !17
  %318 = load <4 x half>, ptr addrspace(3) %lds.const.gep1430, align 8, !dbg !17
  %319 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %317, <4 x float> %316) #9, !dbg !17
  %320 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %318, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1431 = getelementptr i8, ptr addrspace(3) %address.local523, i64 4224, !dbg !17
  %321 = load <4 x half>, ptr addrspace(3) %lds.const.gep1431, align 8, !dbg !17
  %lds.const.gep1432 = getelementptr i8, ptr addrspace(3) %address.local494, i64 1024, !dbg !17
  %322 = load <4 x half>, ptr addrspace(3) %lds.const.gep1432, align 8, !dbg !17
  %323 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %321, <4 x float> %320) #9, !dbg !17
  %324 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %322, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1433 = getelementptr i8, ptr addrspace(3) %address.local494, i64 5120, !dbg !17
  %325 = load <4 x half>, ptr addrspace(3) %lds.const.gep1433, align 8, !dbg !17
  %lds.const.gep1434 = getelementptr i8, ptr addrspace(3) %address.local523, i64 1024, !dbg !17
  %326 = load <4 x half>, ptr addrspace(3) %lds.const.gep1434, align 8, !dbg !17
  %327 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %325, <4 x float> %324) #9, !dbg !17
  %328 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %326, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1435 = getelementptr i8, ptr addrspace(3) %address.local523, i64 5120, !dbg !17
  %329 = load <4 x half>, ptr addrspace(3) %lds.const.gep1435, align 8, !dbg !17
  %lds.const.gep1436 = getelementptr i8, ptr addrspace(3) %address.local494, i64 1152, !dbg !17
  %330 = load <4 x half>, ptr addrspace(3) %lds.const.gep1436, align 8, !dbg !17
  %331 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %329, <4 x float> %328) #9, !dbg !17
  %332 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %330, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1437 = getelementptr i8, ptr addrspace(3) %address.local494, i64 5248, !dbg !17
  %333 = load <4 x half>, ptr addrspace(3) %lds.const.gep1437, align 8, !dbg !17
  %lds.const.gep1438 = getelementptr i8, ptr addrspace(3) %address.local523, i64 1152, !dbg !17
  %334 = load <4 x half>, ptr addrspace(3) %lds.const.gep1438, align 8, !dbg !17
  %335 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %333, <4 x float> %332) #9, !dbg !17
  %336 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %334, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1439 = getelementptr i8, ptr addrspace(3) %address.local523, i64 5248, !dbg !17
  %337 = load <4 x half>, ptr addrspace(3) %lds.const.gep1439, align 8, !dbg !17
  %338 = load <4 x half>, ptr addrspace(3) %address.local494, align 8, !dbg !17
  %339 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %337, <4 x float> %336) #9, !dbg !17
  %340 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %338, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1440 = getelementptr i8, ptr addrspace(3) %address.local494, i64 4096, !dbg !17
  %341 = load <4 x half>, ptr addrspace(3) %lds.const.gep1440, align 8, !dbg !17
  %342 = load <4 x half>, ptr addrspace(3) %address.local523, align 8, !dbg !17
  %343 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %341, <4 x float> %340) #9, !dbg !17
  %344 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %342, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1441 = getelementptr i8, ptr addrspace(3) %address.local523, i64 4096, !dbg !17
  %345 = load <4 x half>, ptr addrspace(3) %lds.const.gep1441, align 8, !dbg !17
  %lds.const.gep1442 = getelementptr i8, ptr addrspace(3) %address.local494, i64 128, !dbg !17
  %346 = load <4 x half>, ptr addrspace(3) %lds.const.gep1442, align 8, !dbg !17
  %347 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %345, <4 x float> %344) #9, !dbg !17
  %348 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %346, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1443 = getelementptr i8, ptr addrspace(3) %address.local494, i64 4224, !dbg !17
  %349 = load <4 x half>, ptr addrspace(3) %lds.const.gep1443, align 8, !dbg !17
  %lds.const.gep1444 = getelementptr i8, ptr addrspace(3) %address.local523, i64 128, !dbg !17
  %350 = load <4 x half>, ptr addrspace(3) %lds.const.gep1444, align 8, !dbg !17
  %351 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %349, <4 x float> %348) #9, !dbg !17
  %352 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %350, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1445 = getelementptr i8, ptr addrspace(3) %address.local523, i64 4224, !dbg !17
  %353 = load <4 x half>, ptr addrspace(3) %lds.const.gep1445, align 8, !dbg !17
  %lds.const.gep1446 = getelementptr i8, ptr addrspace(3) %address.local494, i64 1024, !dbg !17
  %354 = load <4 x half>, ptr addrspace(3) %lds.const.gep1446, align 8, !dbg !17
  %355 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %353, <4 x float> %352) #9, !dbg !17
  %356 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %354, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1447 = getelementptr i8, ptr addrspace(3) %address.local494, i64 5120, !dbg !17
  %357 = load <4 x half>, ptr addrspace(3) %lds.const.gep1447, align 8, !dbg !17
  %lds.const.gep1448 = getelementptr i8, ptr addrspace(3) %address.local523, i64 1024, !dbg !17
  %358 = load <4 x half>, ptr addrspace(3) %lds.const.gep1448, align 8, !dbg !17
  %359 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %357, <4 x float> %356) #9, !dbg !17
  %360 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %358, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1449 = getelementptr i8, ptr addrspace(3) %address.local523, i64 5120, !dbg !17
  %361 = load <4 x half>, ptr addrspace(3) %lds.const.gep1449, align 8, !dbg !17
  %lds.const.gep1450 = getelementptr i8, ptr addrspace(3) %address.local494, i64 1152, !dbg !17
  %362 = load <4 x half>, ptr addrspace(3) %lds.const.gep1450, align 8, !dbg !17
  %363 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %361, <4 x float> %360) #9, !dbg !17
  %364 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %362, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1451 = getelementptr i8, ptr addrspace(3) %address.local494, i64 5248, !dbg !17
  %365 = load <4 x half>, ptr addrspace(3) %lds.const.gep1451, align 8, !dbg !17
  %lds.const.gep1452 = getelementptr i8, ptr addrspace(3) %address.local523, i64 1152, !dbg !17
  %366 = load <4 x half>, ptr addrspace(3) %lds.const.gep1452, align 8, !dbg !17
  %367 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %365, <4 x float> %364) #9, !dbg !17
  %368 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %366, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1453 = getelementptr i8, ptr addrspace(3) %address.local523, i64 5248, !dbg !17
  %369 = load <4 x half>, ptr addrspace(3) %lds.const.gep1453, align 8, !dbg !17
  %370 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %369, <4 x float> %368) #9, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %371 = add nuw i64 %205, 2048, !dbg !16
  %372 = icmp slt i64 %371, 0, !dbg !16
  %373 = sub nuw i64 -2049, %205, !dbg !16
  %374 = select i1 %372, i64 %373, i64 %371, !dbg !16
  %375 = sdiv i64 %374, 4096, !dbg !16
  %.lobit38 = ashr i64 %371, 63, !dbg !16
  %376 = xor i64 %375, %.lobit38, !dbg !16
  %377 = add nsw i64 %376, %16, !dbg !16
  %378 = add nsw i64 %262, 64, !dbg !16
  %379 = icmp slt i64 %262, -64, !dbg !16
  %380 = sub nsw i64 -65, %262, !dbg !16
  %381 = select i1 %379, i64 %380, i64 %378, !dbg !16
  %382 = sdiv i64 %381, 128, !dbg !16
  %383 = sext i1 %379 to i64, !dbg !16
  %384 = xor i64 %382, %383, !dbg !16
  %385 = shl nsw i64 %384, 7, !dbg !16
  %386 = sub i64 %195, %385, !dbg !16
  %387 = shl i64 %377, 19, !dbg !16
  %388 = and i64 %387, 16252928, !dbg !16
  %389 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %388, !dbg !16
  %390 = getelementptr [2 x i8], ptr addrspace(1) %389, i64 %217, !dbg !16
  %391 = getelementptr [2 x i8], ptr addrspace(1) %390, i64 %386, !dbg !16
  %392 = getelementptr i8, ptr addrspace(1) %391, i64 128, !dbg !16
  %393 = getelementptr [2 x i8], ptr addrspace(1) %392, i64 %223, !dbg !16
  %394 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local439), !dbg !16
  %395 = zext i32 %394 to i64, !dbg !16
  %396 = inttoptr i64 %395 to ptr addrspace(3), !dbg !16
  %address.local832 = getelementptr [2 x i8], ptr addrspace(1) %393, i64 %address.local449, !dbg !16
  %397 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %398 = ptrtoint ptr addrspace(1) %address.local832 to i64, !dbg !16
  %399 = sub i64 %398, %397, !dbg !16
  %dma.buffer.offset1576 = trunc i64 %399 to i32, !dbg !16
  %400 = trunc i64 %397 to i32, !dbg !16
  %401 = insertelement <4 x i32> poison, i32 %400, i64 0, !dbg !16
  %402 = lshr i64 %397, 32, !dbg !16
  %403 = trunc i64 %402 to i32, !dbg !16
  %404 = insertelement <4 x i32> %401, i32 %403, i64 1, !dbg !16
  %405 = insertelement <4 x i32> %404, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1577 = insertelement <4 x i32> %405, i32 131072, i64 3, !dbg !16
  %406 = ptrtoint ptr addrspace(3) %396 to i32, !dbg !16
  %dma.buffer.lds1578 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %406), !dbg !16
  %407 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1578, i32 %dma.buffer.offset1576, <4 x i32> %dma.buffer.rsrc1577) #10, !dbg !16
  %408 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local456), !dbg !16
  %409 = zext i32 %408 to i64, !dbg !16
  %410 = inttoptr i64 %409 to ptr addrspace(3), !dbg !16
  %address.local833 = getelementptr [2 x i8], ptr addrspace(1) %393, i64 %address.local466, !dbg !16
  %411 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %412 = ptrtoint ptr addrspace(1) %address.local833 to i64, !dbg !16
  %413 = sub i64 %412, %411, !dbg !16
  %dma.buffer.offset1573 = trunc i64 %413 to i32, !dbg !16
  %414 = trunc i64 %411 to i32, !dbg !16
  %415 = insertelement <4 x i32> poison, i32 %414, i64 0, !dbg !16
  %416 = lshr i64 %411, 32, !dbg !16
  %417 = trunc i64 %416 to i32, !dbg !16
  %418 = insertelement <4 x i32> %415, i32 %417, i64 1, !dbg !16
  %419 = insertelement <4 x i32> %418, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1574 = insertelement <4 x i32> %419, i32 131072, i64 3, !dbg !16
  %420 = ptrtoint ptr addrspace(3) %410 to i32, !dbg !16
  %dma.buffer.lds1575 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %420), !dbg !16
  %421 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1575, i32 %dma.buffer.offset1573, <4 x i32> %dma.buffer.rsrc1574) #10, !dbg !16
  %lds.const.gep1454 = getelementptr i8, ptr addrspace(3) %address.local494, i64 8192, !dbg !17
  %422 = load <4 x half>, ptr addrspace(3) %lds.const.gep1454, align 8, !dbg !17
  %423 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %422, <4 x float> %311) #9, !dbg !17
  %lds.const.gep1455 = getelementptr i8, ptr addrspace(3) %address.local494, i64 12288, !dbg !17
  %424 = load <4 x half>, ptr addrspace(3) %lds.const.gep1455, align 8, !dbg !17
  %lds.const.gep1456 = getelementptr i8, ptr addrspace(3) %address.local523, i64 8192, !dbg !17
  %425 = load <4 x half>, ptr addrspace(3) %lds.const.gep1456, align 8, !dbg !17
  %426 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %424, <4 x float> %423) #9, !dbg !17
  %427 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %425, <4 x float> %315) #9, !dbg !17
  %lds.const.gep1457 = getelementptr i8, ptr addrspace(3) %address.local523, i64 12288, !dbg !17
  %428 = load <4 x half>, ptr addrspace(3) %lds.const.gep1457, align 8, !dbg !17
  %lds.const.gep1458 = getelementptr i8, ptr addrspace(3) %address.local494, i64 8320, !dbg !17
  %429 = load <4 x half>, ptr addrspace(3) %lds.const.gep1458, align 8, !dbg !17
  %430 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %428, <4 x float> %427) #9, !dbg !17
  %431 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %429, <4 x float> %319) #9, !dbg !17
  %lds.const.gep1459 = getelementptr i8, ptr addrspace(3) %address.local494, i64 12416, !dbg !17
  %432 = load <4 x half>, ptr addrspace(3) %lds.const.gep1459, align 8, !dbg !17
  %lds.const.gep1460 = getelementptr i8, ptr addrspace(3) %address.local523, i64 8320, !dbg !17
  %433 = load <4 x half>, ptr addrspace(3) %lds.const.gep1460, align 8, !dbg !17
  %434 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %432, <4 x float> %431) #9, !dbg !17
  %435 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %433, <4 x float> %323) #9, !dbg !17
  %lds.const.gep1461 = getelementptr i8, ptr addrspace(3) %address.local523, i64 12416, !dbg !17
  %436 = load <4 x half>, ptr addrspace(3) %lds.const.gep1461, align 8, !dbg !17
  %lds.const.gep1462 = getelementptr i8, ptr addrspace(3) %address.local494, i64 9216, !dbg !17
  %437 = load <4 x half>, ptr addrspace(3) %lds.const.gep1462, align 8, !dbg !17
  %438 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %436, <4 x float> %435) #9, !dbg !17
  %439 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %437, <4 x float> %327) #9, !dbg !17
  %lds.const.gep1463 = getelementptr i8, ptr addrspace(3) %address.local494, i64 13312, !dbg !17
  %440 = load <4 x half>, ptr addrspace(3) %lds.const.gep1463, align 8, !dbg !17
  %lds.const.gep1464 = getelementptr i8, ptr addrspace(3) %address.local523, i64 9216, !dbg !17
  %441 = load <4 x half>, ptr addrspace(3) %lds.const.gep1464, align 8, !dbg !17
  %442 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %440, <4 x float> %439) #9, !dbg !17
  %443 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %441, <4 x float> %331) #9, !dbg !17
  %lds.const.gep1465 = getelementptr i8, ptr addrspace(3) %address.local523, i64 13312, !dbg !17
  %444 = load <4 x half>, ptr addrspace(3) %lds.const.gep1465, align 8, !dbg !17
  %lds.const.gep1466 = getelementptr i8, ptr addrspace(3) %address.local494, i64 9344, !dbg !17
  %445 = load <4 x half>, ptr addrspace(3) %lds.const.gep1466, align 8, !dbg !17
  %446 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %444, <4 x float> %443) #9, !dbg !17
  %447 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %445, <4 x float> %335) #9, !dbg !17
  %lds.const.gep1467 = getelementptr i8, ptr addrspace(3) %address.local494, i64 13440, !dbg !17
  %448 = load <4 x half>, ptr addrspace(3) %lds.const.gep1467, align 8, !dbg !17
  %lds.const.gep1468 = getelementptr i8, ptr addrspace(3) %address.local523, i64 9344, !dbg !17
  %449 = load <4 x half>, ptr addrspace(3) %lds.const.gep1468, align 8, !dbg !17
  %450 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %448, <4 x float> %447) #9, !dbg !17
  %451 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %449, <4 x float> %339) #9, !dbg !17
  %lds.const.gep1469 = getelementptr i8, ptr addrspace(3) %address.local523, i64 13440, !dbg !17
  %452 = load <4 x half>, ptr addrspace(3) %lds.const.gep1469, align 8, !dbg !17
  %lds.const.gep1470 = getelementptr i8, ptr addrspace(3) %address.local494, i64 8192, !dbg !17
  %453 = load <4 x half>, ptr addrspace(3) %lds.const.gep1470, align 8, !dbg !17
  %454 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %452, <4 x float> %451) #9, !dbg !17
  %455 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %453, <4 x float> %343) #9, !dbg !17
  %lds.const.gep1471 = getelementptr i8, ptr addrspace(3) %address.local494, i64 12288, !dbg !17
  %456 = load <4 x half>, ptr addrspace(3) %lds.const.gep1471, align 8, !dbg !17
  %lds.const.gep1472 = getelementptr i8, ptr addrspace(3) %address.local523, i64 8192, !dbg !17
  %457 = load <4 x half>, ptr addrspace(3) %lds.const.gep1472, align 8, !dbg !17
  %458 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %456, <4 x float> %455) #9, !dbg !17
  %459 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %457, <4 x float> %347) #9, !dbg !17
  %lds.const.gep1473 = getelementptr i8, ptr addrspace(3) %address.local523, i64 12288, !dbg !17
  %460 = load <4 x half>, ptr addrspace(3) %lds.const.gep1473, align 8, !dbg !17
  %lds.const.gep1474 = getelementptr i8, ptr addrspace(3) %address.local494, i64 8320, !dbg !17
  %461 = load <4 x half>, ptr addrspace(3) %lds.const.gep1474, align 8, !dbg !17
  %462 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %460, <4 x float> %459) #9, !dbg !17
  %463 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %461, <4 x float> %351) #9, !dbg !17
  %lds.const.gep1475 = getelementptr i8, ptr addrspace(3) %address.local494, i64 12416, !dbg !17
  %464 = load <4 x half>, ptr addrspace(3) %lds.const.gep1475, align 8, !dbg !17
  %lds.const.gep1476 = getelementptr i8, ptr addrspace(3) %address.local523, i64 8320, !dbg !17
  %465 = load <4 x half>, ptr addrspace(3) %lds.const.gep1476, align 8, !dbg !17
  %466 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %464, <4 x float> %463) #9, !dbg !17
  %467 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %465, <4 x float> %355) #9, !dbg !17
  %lds.const.gep1477 = getelementptr i8, ptr addrspace(3) %address.local523, i64 12416, !dbg !17
  %468 = load <4 x half>, ptr addrspace(3) %lds.const.gep1477, align 8, !dbg !17
  %lds.const.gep1478 = getelementptr i8, ptr addrspace(3) %address.local494, i64 9216, !dbg !17
  %469 = load <4 x half>, ptr addrspace(3) %lds.const.gep1478, align 8, !dbg !17
  %470 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %468, <4 x float> %467) #9, !dbg !17
  %471 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %469, <4 x float> %359) #9, !dbg !17
  %lds.const.gep1479 = getelementptr i8, ptr addrspace(3) %address.local494, i64 13312, !dbg !17
  %472 = load <4 x half>, ptr addrspace(3) %lds.const.gep1479, align 8, !dbg !17
  %lds.const.gep1480 = getelementptr i8, ptr addrspace(3) %address.local523, i64 9216, !dbg !17
  %473 = load <4 x half>, ptr addrspace(3) %lds.const.gep1480, align 8, !dbg !17
  %474 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %472, <4 x float> %471) #9, !dbg !17
  %475 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %473, <4 x float> %363) #9, !dbg !17
  %lds.const.gep1481 = getelementptr i8, ptr addrspace(3) %address.local523, i64 13312, !dbg !17
  %476 = load <4 x half>, ptr addrspace(3) %lds.const.gep1481, align 8, !dbg !17
  %lds.const.gep1482 = getelementptr i8, ptr addrspace(3) %address.local494, i64 9344, !dbg !17
  %477 = load <4 x half>, ptr addrspace(3) %lds.const.gep1482, align 8, !dbg !17
  %478 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %476, <4 x float> %475) #9, !dbg !17
  %479 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %477, <4 x float> %367) #9, !dbg !17
  %lds.const.gep1483 = getelementptr i8, ptr addrspace(3) %address.local494, i64 13440, !dbg !17
  %480 = load <4 x half>, ptr addrspace(3) %lds.const.gep1483, align 8, !dbg !17
  %lds.const.gep1484 = getelementptr i8, ptr addrspace(3) %address.local523, i64 9344, !dbg !17
  %481 = load <4 x half>, ptr addrspace(3) %lds.const.gep1484, align 8, !dbg !17
  %482 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %480, <4 x float> %479) #9, !dbg !17
  %483 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %481, <4 x float> %370) #9, !dbg !17
  %lds.const.gep1485 = getelementptr i8, ptr addrspace(3) %address.local523, i64 13440, !dbg !17
  %484 = load <4 x half>, ptr addrspace(3) %lds.const.gep1485, align 8, !dbg !17
  %485 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %484, <4 x float> %483) #9, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %486 = add nuw i64 %205, 3072, !dbg !16
  %487 = icmp slt i64 %486, 0, !dbg !16
  %488 = sub nuw i64 -3073, %205, !dbg !16
  %489 = select i1 %487, i64 %488, i64 %486, !dbg !16
  %490 = sdiv i64 %489, 4096, !dbg !16
  %.lobit39 = ashr i64 %486, 63, !dbg !16
  %491 = xor i64 %490, %.lobit39, !dbg !16
  %492 = add nsw i64 %491, %16, !dbg !16
  %493 = add nsw i64 %262, 96, !dbg !16
  %494 = icmp slt i64 %262, -96, !dbg !16
  %495 = sub nsw i64 -97, %262, !dbg !16
  %496 = select i1 %494, i64 %495, i64 %493, !dbg !16
  %497 = sdiv i64 %496, 128, !dbg !16
  %498 = sext i1 %494 to i64, !dbg !16
  %499 = xor i64 %497, %498, !dbg !16
  %500 = shl nsw i64 %499, 7, !dbg !16
  %501 = sub i64 %195, %500, !dbg !16
  %502 = shl i64 %492, 19, !dbg !16
  %503 = and i64 %502, 16252928, !dbg !16
  %504 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %503, !dbg !16
  %505 = getelementptr [2 x i8], ptr addrspace(1) %504, i64 %217, !dbg !16
  %506 = getelementptr [2 x i8], ptr addrspace(1) %505, i64 %501, !dbg !16
  %507 = getelementptr i8, ptr addrspace(1) %506, i64 192, !dbg !16
  %508 = getelementptr [2 x i8], ptr addrspace(1) %507, i64 %223, !dbg !16
  %509 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local470), !dbg !16
  %510 = zext i32 %509 to i64, !dbg !16
  %511 = inttoptr i64 %510 to ptr addrspace(3), !dbg !16
  %address.local850 = getelementptr [2 x i8], ptr addrspace(1) %508, i64 %address.local449, !dbg !16
  %512 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %513 = ptrtoint ptr addrspace(1) %address.local850 to i64, !dbg !16
  %514 = sub i64 %513, %512, !dbg !16
  %dma.buffer.offset1570 = trunc i64 %514 to i32, !dbg !16
  %515 = trunc i64 %512 to i32, !dbg !16
  %516 = insertelement <4 x i32> poison, i32 %515, i64 0, !dbg !16
  %517 = lshr i64 %512, 32, !dbg !16
  %518 = trunc i64 %517 to i32, !dbg !16
  %519 = insertelement <4 x i32> %516, i32 %518, i64 1, !dbg !16
  %520 = insertelement <4 x i32> %519, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1571 = insertelement <4 x i32> %520, i32 131072, i64 3, !dbg !16
  %521 = ptrtoint ptr addrspace(3) %511 to i32, !dbg !16
  %dma.buffer.lds1572 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %521), !dbg !16
  %522 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1572, i32 %dma.buffer.offset1570, <4 x i32> %dma.buffer.rsrc1571) #10, !dbg !16
  %523 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local474), !dbg !16
  %524 = zext i32 %523 to i64, !dbg !16
  %525 = inttoptr i64 %524 to ptr addrspace(3), !dbg !16
  %address.local851 = getelementptr [2 x i8], ptr addrspace(1) %508, i64 %address.local466, !dbg !16
  %526 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %527 = ptrtoint ptr addrspace(1) %address.local851 to i64, !dbg !16
  %528 = sub i64 %527, %526, !dbg !16
  %dma.buffer.offset1567 = trunc i64 %528 to i32, !dbg !16
  %529 = trunc i64 %526 to i32, !dbg !16
  %530 = insertelement <4 x i32> poison, i32 %529, i64 0, !dbg !16
  %531 = lshr i64 %526, 32, !dbg !16
  %532 = trunc i64 %531 to i32, !dbg !16
  %533 = insertelement <4 x i32> %530, i32 %532, i64 1, !dbg !16
  %534 = insertelement <4 x i32> %533, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1568 = insertelement <4 x i32> %534, i32 131072, i64 3, !dbg !16
  %535 = ptrtoint ptr addrspace(3) %525 to i32, !dbg !16
  %dma.buffer.lds1569 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %535), !dbg !16
  %536 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1569, i32 %dma.buffer.offset1567, <4 x i32> %dma.buffer.rsrc1568) #10, !dbg !16
  %537 = load <4 x half>, ptr addrspace(3) %address.local494, align 8, !dbg !17
  %538 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %175, <4 x half> %537, <4 x float> %426) #9, !dbg !17
  %lds.const.gep1486 = getelementptr i8, ptr addrspace(3) %address.local494, i64 4096, !dbg !17
  %539 = load <4 x half>, ptr addrspace(3) %lds.const.gep1486, align 8, !dbg !17
  %540 = load <4 x half>, ptr addrspace(3) %address.local523, align 8, !dbg !17
  %541 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %539, <4 x float> %538) #9, !dbg !17
  %542 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %175, <4 x half> %540, <4 x float> %430) #9, !dbg !17
  %lds.const.gep1487 = getelementptr i8, ptr addrspace(3) %address.local523, i64 4096, !dbg !17
  %543 = load <4 x half>, ptr addrspace(3) %lds.const.gep1487, align 8, !dbg !17
  %lds.const.gep1488 = getelementptr i8, ptr addrspace(3) %address.local494, i64 128, !dbg !17
  %544 = load <4 x half>, ptr addrspace(3) %lds.const.gep1488, align 8, !dbg !17
  %545 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %543, <4 x float> %542) #9, !dbg !17
  %546 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %175, <4 x half> %544, <4 x float> %434) #9, !dbg !17
  %lds.const.gep1489 = getelementptr i8, ptr addrspace(3) %address.local494, i64 4224, !dbg !17
  %547 = load <4 x half>, ptr addrspace(3) %lds.const.gep1489, align 8, !dbg !17
  %lds.const.gep1490 = getelementptr i8, ptr addrspace(3) %address.local523, i64 128, !dbg !17
  %548 = load <4 x half>, ptr addrspace(3) %lds.const.gep1490, align 8, !dbg !17
  %549 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %547, <4 x float> %546) #9, !dbg !17
  %550 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %175, <4 x half> %548, <4 x float> %438) #9, !dbg !17
  %lds.const.gep1491 = getelementptr i8, ptr addrspace(3) %address.local523, i64 4224, !dbg !17
  %551 = load <4 x half>, ptr addrspace(3) %lds.const.gep1491, align 8, !dbg !17
  %lds.const.gep1492 = getelementptr i8, ptr addrspace(3) %address.local494, i64 1024, !dbg !17
  %552 = load <4 x half>, ptr addrspace(3) %lds.const.gep1492, align 8, !dbg !17
  %553 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %551, <4 x float> %550) #9, !dbg !17
  %554 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %175, <4 x half> %552, <4 x float> %442) #9, !dbg !17
  %lds.const.gep1493 = getelementptr i8, ptr addrspace(3) %address.local494, i64 5120, !dbg !17
  %555 = load <4 x half>, ptr addrspace(3) %lds.const.gep1493, align 8, !dbg !17
  %lds.const.gep1494 = getelementptr i8, ptr addrspace(3) %address.local523, i64 1024, !dbg !17
  %556 = load <4 x half>, ptr addrspace(3) %lds.const.gep1494, align 8, !dbg !17
  %557 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %555, <4 x float> %554) #9, !dbg !17
  %558 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %175, <4 x half> %556, <4 x float> %446) #9, !dbg !17
  %lds.const.gep1495 = getelementptr i8, ptr addrspace(3) %address.local523, i64 5120, !dbg !17
  %559 = load <4 x half>, ptr addrspace(3) %lds.const.gep1495, align 8, !dbg !17
  %lds.const.gep1496 = getelementptr i8, ptr addrspace(3) %address.local494, i64 1152, !dbg !17
  %560 = load <4 x half>, ptr addrspace(3) %lds.const.gep1496, align 8, !dbg !17
  %561 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %559, <4 x float> %558) #9, !dbg !17
  %562 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %175, <4 x half> %560, <4 x float> %450) #9, !dbg !17
  %lds.const.gep1497 = getelementptr i8, ptr addrspace(3) %address.local494, i64 5248, !dbg !17
  %563 = load <4 x half>, ptr addrspace(3) %lds.const.gep1497, align 8, !dbg !17
  %lds.const.gep1498 = getelementptr i8, ptr addrspace(3) %address.local523, i64 1152, !dbg !17
  %564 = load <4 x half>, ptr addrspace(3) %lds.const.gep1498, align 8, !dbg !17
  %565 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %563, <4 x float> %562) #9, !dbg !17
  %566 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %175, <4 x half> %564, <4 x float> %454) #9, !dbg !17
  %lds.const.gep1499 = getelementptr i8, ptr addrspace(3) %address.local523, i64 5248, !dbg !17
  %567 = load <4 x half>, ptr addrspace(3) %lds.const.gep1499, align 8, !dbg !17
  %568 = load <4 x half>, ptr addrspace(3) %address.local494, align 8, !dbg !17
  %569 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %567, <4 x float> %566) #9, !dbg !17
  %570 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %568, <4 x float> %458) #9, !dbg !17
  %lds.const.gep1500 = getelementptr i8, ptr addrspace(3) %address.local494, i64 4096, !dbg !17
  %571 = load <4 x half>, ptr addrspace(3) %lds.const.gep1500, align 8, !dbg !17
  %572 = load <4 x half>, ptr addrspace(3) %address.local523, align 8, !dbg !17
  %573 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %179, <4 x half> %571, <4 x float> %570) #9, !dbg !17
  %574 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %572, <4 x float> %462) #9, !dbg !17
  %lds.const.gep1501 = getelementptr i8, ptr addrspace(3) %address.local523, i64 4096, !dbg !17
  %575 = load <4 x half>, ptr addrspace(3) %lds.const.gep1501, align 8, !dbg !17
  %lds.const.gep1502 = getelementptr i8, ptr addrspace(3) %address.local494, i64 128, !dbg !17
  %576 = load <4 x half>, ptr addrspace(3) %lds.const.gep1502, align 8, !dbg !17
  %577 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %179, <4 x half> %575, <4 x float> %574) #9, !dbg !17
  %578 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %576, <4 x float> %466) #9, !dbg !17
  %lds.const.gep1503 = getelementptr i8, ptr addrspace(3) %address.local494, i64 4224, !dbg !17
  %579 = load <4 x half>, ptr addrspace(3) %lds.const.gep1503, align 8, !dbg !17
  %lds.const.gep1504 = getelementptr i8, ptr addrspace(3) %address.local523, i64 128, !dbg !17
  %580 = load <4 x half>, ptr addrspace(3) %lds.const.gep1504, align 8, !dbg !17
  %581 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %179, <4 x half> %579, <4 x float> %578) #9, !dbg !17
  %582 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %580, <4 x float> %470) #9, !dbg !17
  %lds.const.gep1505 = getelementptr i8, ptr addrspace(3) %address.local523, i64 4224, !dbg !17
  %583 = load <4 x half>, ptr addrspace(3) %lds.const.gep1505, align 8, !dbg !17
  %lds.const.gep1506 = getelementptr i8, ptr addrspace(3) %address.local494, i64 1024, !dbg !17
  %584 = load <4 x half>, ptr addrspace(3) %lds.const.gep1506, align 8, !dbg !17
  %585 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %179, <4 x half> %583, <4 x float> %582) #9, !dbg !17
  %586 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %584, <4 x float> %474) #9, !dbg !17
  %lds.const.gep1507 = getelementptr i8, ptr addrspace(3) %address.local494, i64 5120, !dbg !17
  %587 = load <4 x half>, ptr addrspace(3) %lds.const.gep1507, align 8, !dbg !17
  %lds.const.gep1508 = getelementptr i8, ptr addrspace(3) %address.local523, i64 1024, !dbg !17
  %588 = load <4 x half>, ptr addrspace(3) %lds.const.gep1508, align 8, !dbg !17
  %589 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %179, <4 x half> %587, <4 x float> %586) #9, !dbg !17
  %590 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %588, <4 x float> %478) #9, !dbg !17
  %lds.const.gep1509 = getelementptr i8, ptr addrspace(3) %address.local523, i64 5120, !dbg !17
  %591 = load <4 x half>, ptr addrspace(3) %lds.const.gep1509, align 8, !dbg !17
  %lds.const.gep1510 = getelementptr i8, ptr addrspace(3) %address.local494, i64 1152, !dbg !17
  %592 = load <4 x half>, ptr addrspace(3) %lds.const.gep1510, align 8, !dbg !17
  %593 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %179, <4 x half> %591, <4 x float> %590) #9, !dbg !17
  %594 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %592, <4 x float> %482) #9, !dbg !17
  %lds.const.gep1511 = getelementptr i8, ptr addrspace(3) %address.local494, i64 5248, !dbg !17
  %595 = load <4 x half>, ptr addrspace(3) %lds.const.gep1511, align 8, !dbg !17
  %lds.const.gep1512 = getelementptr i8, ptr addrspace(3) %address.local523, i64 1152, !dbg !17
  %596 = load <4 x half>, ptr addrspace(3) %lds.const.gep1512, align 8, !dbg !17
  %597 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %179, <4 x half> %595, <4 x float> %594) #9, !dbg !17
  %598 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %596, <4 x float> %485) #9, !dbg !17
  %lds.const.gep1513 = getelementptr i8, ptr addrspace(3) %address.local523, i64 5248, !dbg !17
  %599 = load <4 x half>, ptr addrspace(3) %lds.const.gep1513, align 8, !dbg !17
  %600 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %179, <4 x half> %599, <4 x float> %598) #9, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %601 = add nuw nsw i64 %217, %16, !dbg !18
  %602 = shl i64 %601, 19, !dbg !18
  %603 = and i64 %602, 16252928, !dbg !18
  %604 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %603, !dbg !18
  %605 = getelementptr [2 x i8], ptr addrspace(1) %604, i64 %223, !dbg !18
  %606 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local439), !dbg !18
  %607 = zext i32 %606 to i64, !dbg !18
  %608 = inttoptr i64 %607 to ptr addrspace(3), !dbg !18
  %address.local852 = shl nuw nsw i64 %address.local434, 5, !dbg !18
  %address.local853 = and i64 %address.local852, 8064, !dbg !18
  %address.local854 = shl nuw nsw i64 %address.local434, 3, !dbg !18
  %address.local855 = and i64 %address.local854, 24, !dbg !18
  %address.local856 = or disjoint i64 %address.local853, %address.local855, !dbg !18
  %address.local857 = getelementptr [2 x i8], ptr addrspace(1) %605, i64 %address.local856, !dbg !18
  %609 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %610 = ptrtoint ptr addrspace(1) %address.local857 to i64, !dbg !18
  %611 = sub i64 %610, %609, !dbg !18
  %dma.buffer.offset1564 = trunc i64 %611 to i32, !dbg !18
  %612 = trunc i64 %609 to i32, !dbg !18
  %613 = insertelement <4 x i32> poison, i32 %612, i64 0, !dbg !18
  %614 = lshr i64 %609, 32, !dbg !18
  %615 = trunc i64 %614 to i32, !dbg !18
  %616 = insertelement <4 x i32> %613, i32 %615, i64 1, !dbg !18
  %617 = insertelement <4 x i32> %616, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1565 = insertelement <4 x i32> %617, i32 131072, i64 3, !dbg !18
  %618 = ptrtoint ptr addrspace(3) %608 to i32, !dbg !18
  %dma.buffer.lds1566 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %618), !dbg !18
  %619 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1566, i32 %dma.buffer.offset1564, <4 x i32> %dma.buffer.rsrc1565) #10, !dbg !18
  %620 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local456), !dbg !18
  %621 = zext i32 %620 to i64, !dbg !18
  %622 = inttoptr i64 %621 to ptr addrspace(3), !dbg !18
  %address.local858 = shl nuw nsw i64 %address.local451, 5, !dbg !18
  %address.local859 = and i64 %address.local858, 16256, !dbg !18
  %address.local860 = shl nuw nsw i64 %address.local434, 3, !dbg !18
  %address.local861 = and i64 %address.local860, 24, !dbg !18
  %address.local862 = or disjoint i64 %address.local859, %address.local861, !dbg !18
  %address.local863 = getelementptr [2 x i8], ptr addrspace(1) %605, i64 %address.local862, !dbg !18
  %623 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %624 = ptrtoint ptr addrspace(1) %address.local863 to i64, !dbg !18
  %625 = sub i64 %624, %623, !dbg !18
  %dma.buffer.offset1561 = trunc i64 %625 to i32, !dbg !18
  %626 = trunc i64 %623 to i32, !dbg !18
  %627 = insertelement <4 x i32> poison, i32 %626, i64 0, !dbg !18
  %628 = lshr i64 %623, 32, !dbg !18
  %629 = trunc i64 %628 to i32, !dbg !18
  %630 = insertelement <4 x i32> %627, i32 %629, i64 1, !dbg !18
  %631 = insertelement <4 x i32> %630, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1562 = insertelement <4 x i32> %631, i32 131072, i64 3, !dbg !18
  %632 = ptrtoint ptr addrspace(3) %622 to i32, !dbg !18
  %dma.buffer.lds1563 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %632), !dbg !18
  %633 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1563, i32 %dma.buffer.offset1561, <4 x i32> %dma.buffer.rsrc1562) #10, !dbg !18
  %lds.const.gep1514 = getelementptr i8, ptr addrspace(3) %address.local494, i64 8192, !dbg !17
  %634 = load <4 x half>, ptr addrspace(3) %lds.const.gep1514, align 8, !dbg !17
  %635 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %634, <4 x float> %541) #9, !dbg !17
  %lds.const.gep1515 = getelementptr i8, ptr addrspace(3) %address.local494, i64 12288, !dbg !17
  %636 = load <4 x half>, ptr addrspace(3) %lds.const.gep1515, align 8, !dbg !17
  %lds.const.gep1516 = getelementptr i8, ptr addrspace(3) %address.local523, i64 8192, !dbg !17
  %637 = load <4 x half>, ptr addrspace(3) %lds.const.gep1516, align 8, !dbg !17
  %638 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %636, <4 x float> %635) #9, !dbg !17
  %639 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %637, <4 x float> %545) #9, !dbg !17
  %lds.const.gep1517 = getelementptr i8, ptr addrspace(3) %address.local523, i64 12288, !dbg !17
  %640 = load <4 x half>, ptr addrspace(3) %lds.const.gep1517, align 8, !dbg !17
  %lds.const.gep1518 = getelementptr i8, ptr addrspace(3) %address.local494, i64 8320, !dbg !17
  %641 = load <4 x half>, ptr addrspace(3) %lds.const.gep1518, align 8, !dbg !17
  %642 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %640, <4 x float> %639) #9, !dbg !17
  %643 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %641, <4 x float> %549) #9, !dbg !17
  %lds.const.gep1519 = getelementptr i8, ptr addrspace(3) %address.local494, i64 12416, !dbg !17
  %644 = load <4 x half>, ptr addrspace(3) %lds.const.gep1519, align 8, !dbg !17
  %lds.const.gep1520 = getelementptr i8, ptr addrspace(3) %address.local523, i64 8320, !dbg !17
  %645 = load <4 x half>, ptr addrspace(3) %lds.const.gep1520, align 8, !dbg !17
  %646 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %644, <4 x float> %643) #9, !dbg !17
  %647 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %645, <4 x float> %553) #9, !dbg !17
  %lds.const.gep1521 = getelementptr i8, ptr addrspace(3) %address.local523, i64 12416, !dbg !17
  %648 = load <4 x half>, ptr addrspace(3) %lds.const.gep1521, align 8, !dbg !17
  %lds.const.gep1522 = getelementptr i8, ptr addrspace(3) %address.local494, i64 9216, !dbg !17
  %649 = load <4 x half>, ptr addrspace(3) %lds.const.gep1522, align 8, !dbg !17
  %650 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %648, <4 x float> %647) #9, !dbg !17
  %651 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %649, <4 x float> %557) #9, !dbg !17
  %lds.const.gep1523 = getelementptr i8, ptr addrspace(3) %address.local494, i64 13312, !dbg !17
  %652 = load <4 x half>, ptr addrspace(3) %lds.const.gep1523, align 8, !dbg !17
  %lds.const.gep1524 = getelementptr i8, ptr addrspace(3) %address.local523, i64 9216, !dbg !17
  %653 = load <4 x half>, ptr addrspace(3) %lds.const.gep1524, align 8, !dbg !17
  %654 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %652, <4 x float> %651) #9, !dbg !17
  %655 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %653, <4 x float> %561) #9, !dbg !17
  %lds.const.gep1525 = getelementptr i8, ptr addrspace(3) %address.local523, i64 13312, !dbg !17
  %656 = load <4 x half>, ptr addrspace(3) %lds.const.gep1525, align 8, !dbg !17
  %lds.const.gep1526 = getelementptr i8, ptr addrspace(3) %address.local494, i64 9344, !dbg !17
  %657 = load <4 x half>, ptr addrspace(3) %lds.const.gep1526, align 8, !dbg !17
  %658 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %656, <4 x float> %655) #9, !dbg !17
  %659 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %657, <4 x float> %565) #9, !dbg !17
  %lds.const.gep1527 = getelementptr i8, ptr addrspace(3) %address.local494, i64 13440, !dbg !17
  %660 = load <4 x half>, ptr addrspace(3) %lds.const.gep1527, align 8, !dbg !17
  %lds.const.gep1528 = getelementptr i8, ptr addrspace(3) %address.local523, i64 9344, !dbg !17
  %661 = load <4 x half>, ptr addrspace(3) %lds.const.gep1528, align 8, !dbg !17
  %662 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %660, <4 x float> %659) #9, !dbg !17
  %663 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %661, <4 x float> %569) #9, !dbg !17
  %lds.const.gep1529 = getelementptr i8, ptr addrspace(3) %address.local523, i64 13440, !dbg !17
  %664 = load <4 x half>, ptr addrspace(3) %lds.const.gep1529, align 8, !dbg !17
  %lds.const.gep1530 = getelementptr i8, ptr addrspace(3) %address.local494, i64 8192, !dbg !17
  %665 = load <4 x half>, ptr addrspace(3) %lds.const.gep1530, align 8, !dbg !17
  %666 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %664, <4 x float> %663) #9, !dbg !17
  %667 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %665, <4 x float> %573) #9, !dbg !17
  %lds.const.gep1531 = getelementptr i8, ptr addrspace(3) %address.local494, i64 12288, !dbg !17
  %668 = load <4 x half>, ptr addrspace(3) %lds.const.gep1531, align 8, !dbg !17
  %lds.const.gep1532 = getelementptr i8, ptr addrspace(3) %address.local523, i64 8192, !dbg !17
  %669 = load <4 x half>, ptr addrspace(3) %lds.const.gep1532, align 8, !dbg !17
  %670 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %187, <4 x half> %668, <4 x float> %667) #9, !dbg !17
  %671 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %669, <4 x float> %577) #9, !dbg !17
  %lds.const.gep1533 = getelementptr i8, ptr addrspace(3) %address.local523, i64 12288, !dbg !17
  %672 = load <4 x half>, ptr addrspace(3) %lds.const.gep1533, align 8, !dbg !17
  %lds.const.gep1534 = getelementptr i8, ptr addrspace(3) %address.local494, i64 8320, !dbg !17
  %673 = load <4 x half>, ptr addrspace(3) %lds.const.gep1534, align 8, !dbg !17
  %674 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %187, <4 x half> %672, <4 x float> %671) #9, !dbg !17
  %675 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %673, <4 x float> %581) #9, !dbg !17
  %lds.const.gep1535 = getelementptr i8, ptr addrspace(3) %address.local494, i64 12416, !dbg !17
  %676 = load <4 x half>, ptr addrspace(3) %lds.const.gep1535, align 8, !dbg !17
  %lds.const.gep1536 = getelementptr i8, ptr addrspace(3) %address.local523, i64 8320, !dbg !17
  %677 = load <4 x half>, ptr addrspace(3) %lds.const.gep1536, align 8, !dbg !17
  %678 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %187, <4 x half> %676, <4 x float> %675) #9, !dbg !17
  %679 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %677, <4 x float> %585) #9, !dbg !17
  %lds.const.gep1537 = getelementptr i8, ptr addrspace(3) %address.local523, i64 12416, !dbg !17
  %680 = load <4 x half>, ptr addrspace(3) %lds.const.gep1537, align 8, !dbg !17
  %lds.const.gep1538 = getelementptr i8, ptr addrspace(3) %address.local494, i64 9216, !dbg !17
  %681 = load <4 x half>, ptr addrspace(3) %lds.const.gep1538, align 8, !dbg !17
  %682 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %187, <4 x half> %680, <4 x float> %679) #9, !dbg !17
  %683 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %681, <4 x float> %589) #9, !dbg !17
  %lds.const.gep1539 = getelementptr i8, ptr addrspace(3) %address.local494, i64 13312, !dbg !17
  %684 = load <4 x half>, ptr addrspace(3) %lds.const.gep1539, align 8, !dbg !17
  %lds.const.gep1540 = getelementptr i8, ptr addrspace(3) %address.local523, i64 9216, !dbg !17
  %685 = load <4 x half>, ptr addrspace(3) %lds.const.gep1540, align 8, !dbg !17
  %686 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %187, <4 x half> %684, <4 x float> %683) #9, !dbg !17
  %687 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %685, <4 x float> %593) #9, !dbg !17
  %lds.const.gep1541 = getelementptr i8, ptr addrspace(3) %address.local523, i64 13312, !dbg !17
  %688 = load <4 x half>, ptr addrspace(3) %lds.const.gep1541, align 8, !dbg !17
  %lds.const.gep1542 = getelementptr i8, ptr addrspace(3) %address.local494, i64 9344, !dbg !17
  %689 = load <4 x half>, ptr addrspace(3) %lds.const.gep1542, align 8, !dbg !17
  %690 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %187, <4 x half> %688, <4 x float> %687) #9, !dbg !17
  %691 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %689, <4 x float> %597) #9, !dbg !17
  %lds.const.gep1543 = getelementptr i8, ptr addrspace(3) %address.local494, i64 13440, !dbg !17
  %692 = load <4 x half>, ptr addrspace(3) %lds.const.gep1543, align 8, !dbg !17
  %lds.const.gep1544 = getelementptr i8, ptr addrspace(3) %address.local523, i64 9344, !dbg !17
  %693 = load <4 x half>, ptr addrspace(3) %lds.const.gep1544, align 8, !dbg !17
  %694 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %187, <4 x half> %692, <4 x float> %691) #9, !dbg !17
  %695 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %693, <4 x float> %600) #9, !dbg !17
  %lds.const.gep1545 = getelementptr i8, ptr addrspace(3) %address.local523, i64 13440, !dbg !17
  %696 = load <4 x half>, ptr addrspace(3) %lds.const.gep1545, align 8, !dbg !17
  %697 = or disjoint i64 %214, 127, !dbg !13
  %698 = icmp uge i64 %64, %697, !dbg !13
  %699 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %187, <4 x half> %696, <4 x float> %695) #9, !dbg !17
  %700 = and i1 %190, %698, !dbg !13
  br i1 %700, label %701, label %702, !dbg !13

701:                                              ; preds = %213
  br label %846, !dbg !13

702:                                              ; preds = %213
  %703 = add i64 %64, %32, !dbg !13
  %704 = shl nuw nsw i64 %65, 4, !dbg !13
  %705 = add i64 %703, %704, !dbg !13
  %706 = and i32 %31, 240, !dbg !13
  %index.compact = zext nneg i32 %706 to i64, !dbg !13
  %707 = sub i64 %705, %index.compact, !dbg !13
  %708 = and i64 %32, 48, !dbg !13
  %709 = or disjoint i64 %214, %708, !dbg !13
  %710 = add i64 %707, 1, !dbg !23
  %711 = lshr i64 %710, 32, !dbg !24
  %compare.high = trunc i64 %711 to i32, !dbg !24
  %712 = lshr i64 %709, 32, !dbg !24
  %compare.high1157 = trunc i64 %712 to i32, !dbg !24
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high1157, !dbg !24
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high1157, !dbg !24
  %compare.low.base = trunc i64 %710 to i32, !dbg !24
  %compare.low.base1158 = trunc i64 %709 to i32, !dbg !24
  %compare.low = icmp ugt i32 %compare.low.base, %compare.low.base1158, !dbg !24
  %compare.result = select i1 %compare.high.equal, i1 %compare.low, i1 %compare.high.order, !dbg !24
  %713 = select i1 %compare.result, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 1) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local = or i32 %compare.local.base, 1, !dbg !24
  %compare.low1159 = icmp ugt i32 %compare.low.base, %compare.local, !dbg !24
  %compare.result1160 = select i1 %compare.high.equal, i1 %compare.low1159, i1 %compare.high.order, !dbg !24
  %714 = select i1 %compare.result1160, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1161 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 2) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1162 = or i32 %compare.local.base1161, 2, !dbg !24
  %compare.low1163 = icmp ugt i32 %compare.low.base, %compare.local1162, !dbg !24
  %compare.result1164 = select i1 %compare.high.equal, i1 %compare.low1163, i1 %compare.high.order, !dbg !24
  %715 = select i1 %compare.result1164, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1165 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 3) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1166 = or i32 %compare.local.base1165, 3, !dbg !24
  %compare.low1167 = icmp ugt i32 %compare.low.base, %compare.local1166, !dbg !24
  %compare.result1168 = select i1 %compare.high.equal, i1 %compare.low1167, i1 %compare.high.order, !dbg !24
  %716 = select i1 %compare.result1168, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1169 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 4) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1170 = or i32 %compare.local.base1169, 4, !dbg !24
  %compare.low1171 = icmp ugt i32 %compare.low.base, %compare.local1170, !dbg !24
  %compare.result1172 = select i1 %compare.high.equal, i1 %compare.low1171, i1 %compare.high.order, !dbg !24
  %717 = select i1 %compare.result1172, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1173 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 5) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1174 = or i32 %compare.local.base1173, 5, !dbg !24
  %compare.low1175 = icmp ugt i32 %compare.low.base, %compare.local1174, !dbg !24
  %compare.result1176 = select i1 %compare.high.equal, i1 %compare.low1175, i1 %compare.high.order, !dbg !24
  %718 = select i1 %compare.result1176, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1177 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 6) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1178 = or i32 %compare.local.base1177, 6, !dbg !24
  %compare.low1179 = icmp ugt i32 %compare.low.base, %compare.local1178, !dbg !24
  %compare.result1180 = select i1 %compare.high.equal, i1 %compare.low1179, i1 %compare.high.order, !dbg !24
  %719 = select i1 %compare.result1180, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1181 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 7) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1182 = or i32 %compare.local.base1181, 7, !dbg !24
  %compare.low1183 = icmp ugt i32 %compare.low.base, %compare.local1182, !dbg !24
  %compare.result1184 = select i1 %compare.high.equal, i1 %compare.low1183, i1 %compare.high.order, !dbg !24
  %720 = select i1 %compare.result1184, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1185 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 8) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1186 = or i32 %compare.local.base1185, 8, !dbg !24
  %compare.low1187 = icmp ugt i32 %compare.low.base, %compare.local1186, !dbg !24
  %compare.result1188 = select i1 %compare.high.equal, i1 %compare.low1187, i1 %compare.high.order, !dbg !24
  %721 = select i1 %compare.result1188, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1189 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 9) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1190 = or i32 %compare.local.base1189, 9, !dbg !24
  %compare.low1191 = icmp ugt i32 %compare.low.base, %compare.local1190, !dbg !24
  %compare.result1192 = select i1 %compare.high.equal, i1 %compare.low1191, i1 %compare.high.order, !dbg !24
  %722 = select i1 %compare.result1192, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1193 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 10) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1194 = or i32 %compare.local.base1193, 10, !dbg !24
  %compare.low1195 = icmp ugt i32 %compare.low.base, %compare.local1194, !dbg !24
  %compare.result1196 = select i1 %compare.high.equal, i1 %compare.low1195, i1 %compare.high.order, !dbg !24
  %723 = select i1 %compare.result1196, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1197 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 11) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1198 = or i32 %compare.local.base1197, 11, !dbg !24
  %compare.low1199 = icmp ugt i32 %compare.low.base, %compare.local1198, !dbg !24
  %compare.result1200 = select i1 %compare.high.equal, i1 %compare.low1199, i1 %compare.high.order, !dbg !24
  %724 = select i1 %compare.result1200, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1201 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 12) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1202 = or i32 %compare.local.base1201, 12, !dbg !24
  %compare.low1203 = icmp ugt i32 %compare.low.base, %compare.local1202, !dbg !24
  %compare.result1204 = select i1 %compare.high.equal, i1 %compare.low1203, i1 %compare.high.order, !dbg !24
  %725 = select i1 %compare.result1204, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1205 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 13) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1206 = or i32 %compare.local.base1205, 13, !dbg !24
  %compare.low1207 = icmp ugt i32 %compare.low.base, %compare.local1206, !dbg !24
  %compare.result1208 = select i1 %compare.high.equal, i1 %compare.low1207, i1 %compare.high.order, !dbg !24
  %726 = select i1 %compare.result1208, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1209 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 14) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1210 = or i32 %compare.local.base1209, 14, !dbg !24
  %compare.low1211 = icmp ugt i32 %compare.low.base, %compare.local1210, !dbg !24
  %compare.result1212 = select i1 %compare.high.equal, i1 %compare.low1211, i1 %compare.high.order, !dbg !24
  %727 = select i1 %compare.result1212, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1213 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 15) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1214 = or i32 %compare.local.base1213, 15, !dbg !24
  %compare.low1215 = icmp ugt i32 %compare.low.base, %compare.local1214, !dbg !24
  %compare.result1216 = select i1 %compare.high.equal, i1 %compare.low1215, i1 %compare.high.order, !dbg !24
  %728 = select i1 %compare.result1216, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1217 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 16) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1218 = or i32 %compare.local.base1217, 64, !dbg !24
  %compare.low1219 = icmp ugt i32 %compare.low.base, %compare.local1218, !dbg !24
  %compare.result1220 = select i1 %compare.high.equal, i1 %compare.low1219, i1 %compare.high.order, !dbg !24
  %729 = select i1 %compare.result1220, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1221 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 17) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1222 = or i32 %compare.local.base1221, 65, !dbg !24
  %compare.low1223 = icmp ugt i32 %compare.low.base, %compare.local1222, !dbg !24
  %compare.result1224 = select i1 %compare.high.equal, i1 %compare.low1223, i1 %compare.high.order, !dbg !24
  %730 = select i1 %compare.result1224, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1225 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 18) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1226 = or i32 %compare.local.base1225, 66, !dbg !24
  %compare.low1227 = icmp ugt i32 %compare.low.base, %compare.local1226, !dbg !24
  %compare.result1228 = select i1 %compare.high.equal, i1 %compare.low1227, i1 %compare.high.order, !dbg !24
  %731 = select i1 %compare.result1228, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1229 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 19) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1230 = or i32 %compare.local.base1229, 67, !dbg !24
  %compare.low1231 = icmp ugt i32 %compare.low.base, %compare.local1230, !dbg !24
  %compare.result1232 = select i1 %compare.high.equal, i1 %compare.low1231, i1 %compare.high.order, !dbg !24
  %732 = select i1 %compare.result1232, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1233 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 20) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1234 = or i32 %compare.local.base1233, 68, !dbg !24
  %compare.low1235 = icmp ugt i32 %compare.low.base, %compare.local1234, !dbg !24
  %compare.result1236 = select i1 %compare.high.equal, i1 %compare.low1235, i1 %compare.high.order, !dbg !24
  %733 = select i1 %compare.result1236, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1237 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 21) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1238 = or i32 %compare.local.base1237, 69, !dbg !24
  %compare.low1239 = icmp ugt i32 %compare.low.base, %compare.local1238, !dbg !24
  %compare.result1240 = select i1 %compare.high.equal, i1 %compare.low1239, i1 %compare.high.order, !dbg !24
  %734 = select i1 %compare.result1240, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1241 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 22) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1242 = or i32 %compare.local.base1241, 70, !dbg !24
  %compare.low1243 = icmp ugt i32 %compare.low.base, %compare.local1242, !dbg !24
  %compare.result1244 = select i1 %compare.high.equal, i1 %compare.low1243, i1 %compare.high.order, !dbg !24
  %735 = select i1 %compare.result1244, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1245 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 23) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1246 = or i32 %compare.local.base1245, 71, !dbg !24
  %compare.low1247 = icmp ugt i32 %compare.low.base, %compare.local1246, !dbg !24
  %compare.result1248 = select i1 %compare.high.equal, i1 %compare.low1247, i1 %compare.high.order, !dbg !24
  %736 = select i1 %compare.result1248, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1249 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 24) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1250 = or i32 %compare.local.base1249, 72, !dbg !24
  %compare.low1251 = icmp ugt i32 %compare.low.base, %compare.local1250, !dbg !24
  %compare.result1252 = select i1 %compare.high.equal, i1 %compare.low1251, i1 %compare.high.order, !dbg !24
  %737 = select i1 %compare.result1252, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1253 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 25) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1254 = or i32 %compare.local.base1253, 73, !dbg !24
  %compare.low1255 = icmp ugt i32 %compare.low.base, %compare.local1254, !dbg !24
  %compare.result1256 = select i1 %compare.high.equal, i1 %compare.low1255, i1 %compare.high.order, !dbg !24
  %738 = select i1 %compare.result1256, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1257 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 26) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1258 = or i32 %compare.local.base1257, 74, !dbg !24
  %compare.low1259 = icmp ugt i32 %compare.low.base, %compare.local1258, !dbg !24
  %compare.result1260 = select i1 %compare.high.equal, i1 %compare.low1259, i1 %compare.high.order, !dbg !24
  %739 = select i1 %compare.result1260, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1261 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 27) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1262 = or i32 %compare.local.base1261, 75, !dbg !24
  %compare.low1263 = icmp ugt i32 %compare.low.base, %compare.local1262, !dbg !24
  %compare.result1264 = select i1 %compare.high.equal, i1 %compare.low1263, i1 %compare.high.order, !dbg !24
  %740 = select i1 %compare.result1264, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1265 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 28) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1266 = or i32 %compare.local.base1265, 76, !dbg !24
  %compare.low1267 = icmp ugt i32 %compare.low.base, %compare.local1266, !dbg !24
  %compare.result1268 = select i1 %compare.high.equal, i1 %compare.low1267, i1 %compare.high.order, !dbg !24
  %741 = select i1 %compare.result1268, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1269 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 29) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1270 = or i32 %compare.local.base1269, 77, !dbg !24
  %compare.low1271 = icmp ugt i32 %compare.low.base, %compare.local1270, !dbg !24
  %compare.result1272 = select i1 %compare.high.equal, i1 %compare.low1271, i1 %compare.high.order, !dbg !24
  %742 = select i1 %compare.result1272, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1273 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 30) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1274 = or i32 %compare.local.base1273, 78, !dbg !24
  %compare.low1275 = icmp ugt i32 %compare.low.base, %compare.local1274, !dbg !24
  %compare.result1276 = select i1 %compare.high.equal, i1 %compare.low1275, i1 %compare.high.order, !dbg !24
  %743 = select i1 %compare.result1276, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1277 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 31) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1278 = or i32 %compare.local.base1277, 79, !dbg !24
  %compare.low1279 = icmp ugt i32 %compare.low.base, %compare.local1278, !dbg !24
  %compare.result1280 = select i1 %compare.high.equal, i1 %compare.low1279, i1 %compare.high.order, !dbg !24
  %744 = select i1 %compare.result1280, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %745 = add i64 %707, 65, !dbg !23
  %746 = lshr i64 %745, 32, !dbg !24
  %compare.high1281 = trunc i64 %746 to i32, !dbg !24
  %747 = lshr i64 %709, 32, !dbg !24
  %compare.high1282 = trunc i64 %747 to i32, !dbg !24
  %compare.high.equal1283 = icmp eq i32 %compare.high1281, %compare.high1282, !dbg !24
  %compare.high.order1284 = icmp ugt i32 %compare.high1281, %compare.high1282, !dbg !24
  %compare.low.base1285 = trunc i64 %745 to i32, !dbg !24
  %compare.low.base1286 = trunc i64 %709 to i32, !dbg !24
  %compare.low1287 = icmp ugt i32 %compare.low.base1285, %compare.low.base1286, !dbg !24
  %compare.result1288 = select i1 %compare.high.equal1283, i1 %compare.low1287, i1 %compare.high.order1284, !dbg !24
  %748 = select i1 %compare.result1288, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1289 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 32) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1290 = or i32 %compare.local.base1289, 1, !dbg !24
  %compare.low1291 = icmp ugt i32 %compare.low.base1285, %compare.local1290, !dbg !24
  %compare.result1292 = select i1 %compare.high.equal1283, i1 %compare.low1291, i1 %compare.high.order1284, !dbg !24
  %749 = select i1 %compare.result1292, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1293 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 33) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1294 = or i32 %compare.local.base1293, 2, !dbg !24
  %compare.low1295 = icmp ugt i32 %compare.low.base1285, %compare.local1294, !dbg !24
  %compare.result1296 = select i1 %compare.high.equal1283, i1 %compare.low1295, i1 %compare.high.order1284, !dbg !24
  %750 = select i1 %compare.result1296, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1297 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 34) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1298 = or i32 %compare.local.base1297, 3, !dbg !24
  %compare.low1299 = icmp ugt i32 %compare.low.base1285, %compare.local1298, !dbg !24
  %compare.result1300 = select i1 %compare.high.equal1283, i1 %compare.low1299, i1 %compare.high.order1284, !dbg !24
  %751 = select i1 %compare.result1300, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1301 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 35) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1302 = or i32 %compare.local.base1301, 4, !dbg !24
  %compare.low1303 = icmp ugt i32 %compare.low.base1285, %compare.local1302, !dbg !24
  %compare.result1304 = select i1 %compare.high.equal1283, i1 %compare.low1303, i1 %compare.high.order1284, !dbg !24
  %752 = select i1 %compare.result1304, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1305 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 36) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1306 = or i32 %compare.local.base1305, 5, !dbg !24
  %compare.low1307 = icmp ugt i32 %compare.low.base1285, %compare.local1306, !dbg !24
  %compare.result1308 = select i1 %compare.high.equal1283, i1 %compare.low1307, i1 %compare.high.order1284, !dbg !24
  %753 = select i1 %compare.result1308, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1309 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 37) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1310 = or i32 %compare.local.base1309, 6, !dbg !24
  %compare.low1311 = icmp ugt i32 %compare.low.base1285, %compare.local1310, !dbg !24
  %compare.result1312 = select i1 %compare.high.equal1283, i1 %compare.low1311, i1 %compare.high.order1284, !dbg !24
  %754 = select i1 %compare.result1312, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1313 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 38) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1314 = or i32 %compare.local.base1313, 7, !dbg !24
  %compare.low1315 = icmp ugt i32 %compare.low.base1285, %compare.local1314, !dbg !24
  %compare.result1316 = select i1 %compare.high.equal1283, i1 %compare.low1315, i1 %compare.high.order1284, !dbg !24
  %755 = select i1 %compare.result1316, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1317 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 39) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1318 = or i32 %compare.local.base1317, 8, !dbg !24
  %compare.low1319 = icmp ugt i32 %compare.low.base1285, %compare.local1318, !dbg !24
  %compare.result1320 = select i1 %compare.high.equal1283, i1 %compare.low1319, i1 %compare.high.order1284, !dbg !24
  %756 = select i1 %compare.result1320, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1321 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 40) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1322 = or i32 %compare.local.base1321, 9, !dbg !24
  %compare.low1323 = icmp ugt i32 %compare.low.base1285, %compare.local1322, !dbg !24
  %compare.result1324 = select i1 %compare.high.equal1283, i1 %compare.low1323, i1 %compare.high.order1284, !dbg !24
  %757 = select i1 %compare.result1324, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1325 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 41) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1326 = or i32 %compare.local.base1325, 10, !dbg !24
  %compare.low1327 = icmp ugt i32 %compare.low.base1285, %compare.local1326, !dbg !24
  %compare.result1328 = select i1 %compare.high.equal1283, i1 %compare.low1327, i1 %compare.high.order1284, !dbg !24
  %758 = select i1 %compare.result1328, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1329 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 42) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1330 = or i32 %compare.local.base1329, 11, !dbg !24
  %compare.low1331 = icmp ugt i32 %compare.low.base1285, %compare.local1330, !dbg !24
  %compare.result1332 = select i1 %compare.high.equal1283, i1 %compare.low1331, i1 %compare.high.order1284, !dbg !24
  %759 = select i1 %compare.result1332, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1333 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 43) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1334 = or i32 %compare.local.base1333, 12, !dbg !24
  %compare.low1335 = icmp ugt i32 %compare.low.base1285, %compare.local1334, !dbg !24
  %compare.result1336 = select i1 %compare.high.equal1283, i1 %compare.low1335, i1 %compare.high.order1284, !dbg !24
  %760 = select i1 %compare.result1336, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1337 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 44) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1338 = or i32 %compare.local.base1337, 13, !dbg !24
  %compare.low1339 = icmp ugt i32 %compare.low.base1285, %compare.local1338, !dbg !24
  %compare.result1340 = select i1 %compare.high.equal1283, i1 %compare.low1339, i1 %compare.high.order1284, !dbg !24
  %761 = select i1 %compare.result1340, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1341 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 45) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1342 = or i32 %compare.local.base1341, 14, !dbg !24
  %compare.low1343 = icmp ugt i32 %compare.low.base1285, %compare.local1342, !dbg !24
  %compare.result1344 = select i1 %compare.high.equal1283, i1 %compare.low1343, i1 %compare.high.order1284, !dbg !24
  %762 = select i1 %compare.result1344, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1345 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 46) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1346 = or i32 %compare.local.base1345, 15, !dbg !24
  %compare.low1347 = icmp ugt i32 %compare.low.base1285, %compare.local1346, !dbg !24
  %compare.result1348 = select i1 %compare.high.equal1283, i1 %compare.low1347, i1 %compare.high.order1284, !dbg !24
  %763 = select i1 %compare.result1348, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1349 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 47) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1350 = or i32 %compare.local.base1349, 64, !dbg !24
  %compare.low1351 = icmp ugt i32 %compare.low.base1285, %compare.local1350, !dbg !24
  %compare.result1352 = select i1 %compare.high.equal1283, i1 %compare.low1351, i1 %compare.high.order1284, !dbg !24
  %764 = select i1 %compare.result1352, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1353 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 48) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1354 = or i32 %compare.local.base1353, 65, !dbg !24
  %compare.low1355 = icmp ugt i32 %compare.low.base1285, %compare.local1354, !dbg !24
  %compare.result1356 = select i1 %compare.high.equal1283, i1 %compare.low1355, i1 %compare.high.order1284, !dbg !24
  %765 = select i1 %compare.result1356, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1357 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 49) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1358 = or i32 %compare.local.base1357, 66, !dbg !24
  %compare.low1359 = icmp ugt i32 %compare.low.base1285, %compare.local1358, !dbg !24
  %compare.result1360 = select i1 %compare.high.equal1283, i1 %compare.low1359, i1 %compare.high.order1284, !dbg !24
  %766 = select i1 %compare.result1360, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1361 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 50) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1362 = or i32 %compare.local.base1361, 67, !dbg !24
  %compare.low1363 = icmp ugt i32 %compare.low.base1285, %compare.local1362, !dbg !24
  %compare.result1364 = select i1 %compare.high.equal1283, i1 %compare.low1363, i1 %compare.high.order1284, !dbg !24
  %767 = select i1 %compare.result1364, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1365 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 51) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1366 = or i32 %compare.local.base1365, 68, !dbg !24
  %compare.low1367 = icmp ugt i32 %compare.low.base1285, %compare.local1366, !dbg !24
  %compare.result1368 = select i1 %compare.high.equal1283, i1 %compare.low1367, i1 %compare.high.order1284, !dbg !24
  %768 = select i1 %compare.result1368, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1369 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 52) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1370 = or i32 %compare.local.base1369, 69, !dbg !24
  %compare.low1371 = icmp ugt i32 %compare.low.base1285, %compare.local1370, !dbg !24
  %compare.result1372 = select i1 %compare.high.equal1283, i1 %compare.low1371, i1 %compare.high.order1284, !dbg !24
  %769 = select i1 %compare.result1372, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1373 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 53) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1374 = or i32 %compare.local.base1373, 70, !dbg !24
  %compare.low1375 = icmp ugt i32 %compare.low.base1285, %compare.local1374, !dbg !24
  %compare.result1376 = select i1 %compare.high.equal1283, i1 %compare.low1375, i1 %compare.high.order1284, !dbg !24
  %770 = select i1 %compare.result1376, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1377 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 54) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1378 = or i32 %compare.local.base1377, 71, !dbg !24
  %compare.low1379 = icmp ugt i32 %compare.low.base1285, %compare.local1378, !dbg !24
  %compare.result1380 = select i1 %compare.high.equal1283, i1 %compare.low1379, i1 %compare.high.order1284, !dbg !24
  %771 = select i1 %compare.result1380, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1381 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 55) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1382 = or i32 %compare.local.base1381, 72, !dbg !24
  %compare.low1383 = icmp ugt i32 %compare.low.base1285, %compare.local1382, !dbg !24
  %compare.result1384 = select i1 %compare.high.equal1283, i1 %compare.low1383, i1 %compare.high.order1284, !dbg !24
  %772 = select i1 %compare.result1384, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1385 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 56) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1386 = or i32 %compare.local.base1385, 73, !dbg !24
  %compare.low1387 = icmp ugt i32 %compare.low.base1285, %compare.local1386, !dbg !24
  %compare.result1388 = select i1 %compare.high.equal1283, i1 %compare.low1387, i1 %compare.high.order1284, !dbg !24
  %773 = select i1 %compare.result1388, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1389 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 57) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1390 = or i32 %compare.local.base1389, 74, !dbg !24
  %compare.low1391 = icmp ugt i32 %compare.low.base1285, %compare.local1390, !dbg !24
  %compare.result1392 = select i1 %compare.high.equal1283, i1 %compare.low1391, i1 %compare.high.order1284, !dbg !24
  %774 = select i1 %compare.result1392, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1393 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 58) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1394 = or i32 %compare.local.base1393, 75, !dbg !24
  %compare.low1395 = icmp ugt i32 %compare.low.base1285, %compare.local1394, !dbg !24
  %compare.result1396 = select i1 %compare.high.equal1283, i1 %compare.low1395, i1 %compare.high.order1284, !dbg !24
  %775 = select i1 %compare.result1396, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1397 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 59) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1398 = or i32 %compare.local.base1397, 76, !dbg !24
  %compare.low1399 = icmp ugt i32 %compare.low.base1285, %compare.local1398, !dbg !24
  %compare.result1400 = select i1 %compare.high.equal1283, i1 %compare.low1399, i1 %compare.high.order1284, !dbg !24
  %776 = select i1 %compare.result1400, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1401 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 60) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1402 = or i32 %compare.local.base1401, 77, !dbg !24
  %compare.low1403 = icmp ugt i32 %compare.low.base1285, %compare.local1402, !dbg !24
  %compare.result1404 = select i1 %compare.high.equal1283, i1 %compare.low1403, i1 %compare.high.order1284, !dbg !24
  %777 = select i1 %compare.result1404, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1405 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 61) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1406 = or i32 %compare.local.base1405, 78, !dbg !24
  %compare.low1407 = icmp ugt i32 %compare.low.base1285, %compare.local1406, !dbg !24
  %compare.result1408 = select i1 %compare.high.equal1283, i1 %compare.low1407, i1 %compare.high.order1284, !dbg !24
  %778 = select i1 %compare.result1408, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1409 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 62) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1410 = or i32 %compare.local.base1409, 79, !dbg !24
  %compare.low1411 = icmp ugt i32 %compare.low.base1285, %compare.local1410, !dbg !24
  %compare.result1412 = select i1 %compare.high.equal1283, i1 %compare.low1411, i1 %compare.high.order1284, !dbg !24
  %779 = select i1 %compare.result1412, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %780 = insertelement <32 x float> poison, float %713, i64 0, !dbg !13
  %781 = insertelement <32 x float> %780, float %714, i64 1, !dbg !13
  %782 = insertelement <32 x float> %781, float %715, i64 2, !dbg !13
  %783 = insertelement <32 x float> %782, float %716, i64 3, !dbg !13
  %784 = insertelement <32 x float> %783, float %717, i64 4, !dbg !13
  %785 = insertelement <32 x float> %784, float %718, i64 5, !dbg !13
  %786 = insertelement <32 x float> %785, float %719, i64 6, !dbg !13
  %787 = insertelement <32 x float> %786, float %720, i64 7, !dbg !13
  %788 = insertelement <32 x float> %787, float %721, i64 8, !dbg !13
  %789 = insertelement <32 x float> %788, float %722, i64 9, !dbg !13
  %790 = insertelement <32 x float> %789, float %723, i64 10, !dbg !13
  %791 = insertelement <32 x float> %790, float %724, i64 11, !dbg !13
  %792 = insertelement <32 x float> %791, float %725, i64 12, !dbg !13
  %793 = insertelement <32 x float> %792, float %726, i64 13, !dbg !13
  %794 = insertelement <32 x float> %793, float %727, i64 14, !dbg !13
  %795 = insertelement <32 x float> %794, float %728, i64 15, !dbg !13
  %796 = insertelement <32 x float> %795, float %729, i64 16, !dbg !13
  %797 = insertelement <32 x float> %796, float %730, i64 17, !dbg !13
  %798 = insertelement <32 x float> %797, float %731, i64 18, !dbg !13
  %799 = insertelement <32 x float> %798, float %732, i64 19, !dbg !13
  %800 = insertelement <32 x float> %799, float %733, i64 20, !dbg !13
  %801 = insertelement <32 x float> %800, float %734, i64 21, !dbg !13
  %802 = insertelement <32 x float> %801, float %735, i64 22, !dbg !13
  %803 = insertelement <32 x float> %802, float %736, i64 23, !dbg !13
  %804 = insertelement <32 x float> %803, float %737, i64 24, !dbg !13
  %805 = insertelement <32 x float> %804, float %738, i64 25, !dbg !13
  %806 = insertelement <32 x float> %805, float %739, i64 26, !dbg !13
  %807 = insertelement <32 x float> %806, float %740, i64 27, !dbg !13
  %808 = insertelement <32 x float> %807, float %741, i64 28, !dbg !13
  %809 = insertelement <32 x float> %808, float %742, i64 29, !dbg !13
  %810 = insertelement <32 x float> %809, float %743, i64 30, !dbg !13
  %811 = insertelement <32 x float> %810, float %744, i64 31, !dbg !13
  %812 = insertvalue [2 x <32 x float>] poison, <32 x float> %811, 0, !dbg !13
  %813 = insertelement <32 x float> poison, float %748, i64 0, !dbg !13
  %814 = insertelement <32 x float> %813, float %749, i64 1, !dbg !13
  %815 = insertelement <32 x float> %814, float %750, i64 2, !dbg !13
  %816 = insertelement <32 x float> %815, float %751, i64 3, !dbg !13
  %817 = insertelement <32 x float> %816, float %752, i64 4, !dbg !13
  %818 = insertelement <32 x float> %817, float %753, i64 5, !dbg !13
  %819 = insertelement <32 x float> %818, float %754, i64 6, !dbg !13
  %820 = insertelement <32 x float> %819, float %755, i64 7, !dbg !13
  %821 = insertelement <32 x float> %820, float %756, i64 8, !dbg !13
  %822 = insertelement <32 x float> %821, float %757, i64 9, !dbg !13
  %823 = insertelement <32 x float> %822, float %758, i64 10, !dbg !13
  %824 = insertelement <32 x float> %823, float %759, i64 11, !dbg !13
  %825 = insertelement <32 x float> %824, float %760, i64 12, !dbg !13
  %826 = insertelement <32 x float> %825, float %761, i64 13, !dbg !13
  %827 = insertelement <32 x float> %826, float %762, i64 14, !dbg !13
  %828 = insertelement <32 x float> %827, float %763, i64 15, !dbg !13
  %829 = insertelement <32 x float> %828, float %764, i64 16, !dbg !13
  %830 = insertelement <32 x float> %829, float %765, i64 17, !dbg !13
  %831 = insertelement <32 x float> %830, float %766, i64 18, !dbg !13
  %832 = insertelement <32 x float> %831, float %767, i64 19, !dbg !13
  %833 = insertelement <32 x float> %832, float %768, i64 20, !dbg !13
  %834 = insertelement <32 x float> %833, float %769, i64 21, !dbg !13
  %835 = insertelement <32 x float> %834, float %770, i64 22, !dbg !13
  %836 = insertelement <32 x float> %835, float %771, i64 23, !dbg !13
  %837 = insertelement <32 x float> %836, float %772, i64 24, !dbg !13
  %838 = insertelement <32 x float> %837, float %773, i64 25, !dbg !13
  %839 = insertelement <32 x float> %838, float %774, i64 26, !dbg !13
  %840 = insertelement <32 x float> %839, float %775, i64 27, !dbg !13
  %841 = insertelement <32 x float> %840, float %776, i64 28, !dbg !13
  %842 = insertelement <32 x float> %841, float %777, i64 29, !dbg !13
  %843 = insertelement <32 x float> %842, float %778, i64 30, !dbg !13
  %844 = insertelement <32 x float> %843, float %779, i64 31, !dbg !13
  %845 = insertvalue [2 x <32 x float>] %812, <32 x float> %844, 1, !dbg !13
  br label %846, !dbg !13

846:                                              ; preds = %701, %702
  %847 = phi [2 x <32 x float>] [ %845, %702 ], [ zeroinitializer, %701 ], !dbg !13
  br label %848, !dbg !13

848:                                              ; preds = %846
  %849 = extractvalue [2 x <32 x float>] %847, 0, !dbg !26
  %850 = shufflevector <32 x float> %849, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %851 = fadd <4 x float> %638, %850, !dbg !26
  %852 = extractelement <4 x float> %851, i64 0, !dbg !26
  %853 = extractelement <4 x float> %851, i64 1, !dbg !26
  %854 = extractelement <4 x float> %851, i64 2, !dbg !26
  %855 = extractelement <4 x float> %851, i64 3, !dbg !26
  %856 = shufflevector <32 x float> %849, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %857 = fadd <4 x float> %642, %856, !dbg !26
  %858 = extractelement <4 x float> %857, i64 0, !dbg !26
  %859 = extractelement <4 x float> %857, i64 1, !dbg !26
  %860 = extractelement <4 x float> %857, i64 2, !dbg !26
  %861 = extractelement <4 x float> %857, i64 3, !dbg !26
  %862 = shufflevector <32 x float> %849, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !26
  %863 = fadd <4 x float> %646, %862, !dbg !26
  %864 = extractelement <4 x float> %863, i64 0, !dbg !26
  %865 = extractelement <4 x float> %863, i64 1, !dbg !26
  %866 = extractelement <4 x float> %863, i64 2, !dbg !26
  %867 = extractelement <4 x float> %863, i64 3, !dbg !26
  %868 = shufflevector <32 x float> %849, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !26
  %869 = fadd <4 x float> %650, %868, !dbg !26
  %870 = extractelement <4 x float> %869, i64 0, !dbg !26
  %871 = extractelement <4 x float> %869, i64 1, !dbg !26
  %872 = extractelement <4 x float> %869, i64 2, !dbg !26
  %873 = extractelement <4 x float> %869, i64 3, !dbg !26
  %874 = shufflevector <32 x float> %849, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !26
  %875 = fadd <4 x float> %654, %874, !dbg !26
  %876 = extractelement <4 x float> %875, i64 0, !dbg !26
  %877 = extractelement <4 x float> %875, i64 1, !dbg !26
  %878 = extractelement <4 x float> %875, i64 2, !dbg !26
  %879 = extractelement <4 x float> %875, i64 3, !dbg !26
  %880 = shufflevector <32 x float> %849, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !26
  %881 = fadd <4 x float> %658, %880, !dbg !26
  %882 = extractelement <4 x float> %881, i64 0, !dbg !26
  %883 = extractelement <4 x float> %881, i64 1, !dbg !26
  %884 = extractelement <4 x float> %881, i64 2, !dbg !26
  %885 = extractelement <4 x float> %881, i64 3, !dbg !26
  %886 = shufflevector <32 x float> %849, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !26
  %887 = fadd <4 x float> %662, %886, !dbg !26
  %888 = extractelement <4 x float> %887, i64 0, !dbg !26
  %889 = extractelement <4 x float> %887, i64 1, !dbg !26
  %890 = extractelement <4 x float> %887, i64 2, !dbg !26
  %891 = extractelement <4 x float> %887, i64 3, !dbg !26
  %892 = shufflevector <32 x float> %849, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !26
  %893 = fadd <4 x float> %666, %892, !dbg !26
  %894 = extractelement <4 x float> %893, i64 0, !dbg !26
  %895 = extractelement <4 x float> %893, i64 1, !dbg !26
  %896 = extractelement <4 x float> %893, i64 2, !dbg !26
  %897 = extractelement <4 x float> %893, i64 3, !dbg !26
  %898 = extractvalue [2 x <32 x float>] %847, 1, !dbg !26
  %899 = shufflevector <32 x float> %898, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %900 = fadd <4 x float> %670, %899, !dbg !26
  %901 = extractelement <4 x float> %900, i64 0, !dbg !26
  %902 = extractelement <4 x float> %900, i64 1, !dbg !26
  %903 = extractelement <4 x float> %900, i64 2, !dbg !26
  %904 = extractelement <4 x float> %900, i64 3, !dbg !26
  %905 = shufflevector <32 x float> %898, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %906 = fadd <4 x float> %674, %905, !dbg !26
  %907 = extractelement <4 x float> %906, i64 0, !dbg !26
  %908 = extractelement <4 x float> %906, i64 1, !dbg !26
  %909 = extractelement <4 x float> %906, i64 2, !dbg !26
  %910 = extractelement <4 x float> %906, i64 3, !dbg !26
  %911 = shufflevector <32 x float> %898, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !26
  %912 = fadd <4 x float> %678, %911, !dbg !26
  %913 = extractelement <4 x float> %912, i64 0, !dbg !26
  %914 = extractelement <4 x float> %912, i64 1, !dbg !26
  %915 = extractelement <4 x float> %912, i64 2, !dbg !26
  %916 = extractelement <4 x float> %912, i64 3, !dbg !26
  %917 = shufflevector <32 x float> %898, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !26
  %918 = fadd <4 x float> %682, %917, !dbg !26
  %919 = extractelement <4 x float> %918, i64 0, !dbg !26
  %920 = extractelement <4 x float> %918, i64 1, !dbg !26
  %921 = extractelement <4 x float> %918, i64 2, !dbg !26
  %922 = extractelement <4 x float> %918, i64 3, !dbg !26
  %923 = shufflevector <32 x float> %898, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !26
  %924 = fadd <4 x float> %686, %923, !dbg !26
  %925 = extractelement <4 x float> %924, i64 0, !dbg !26
  %926 = extractelement <4 x float> %924, i64 1, !dbg !26
  %927 = extractelement <4 x float> %924, i64 2, !dbg !26
  %928 = extractelement <4 x float> %924, i64 3, !dbg !26
  %929 = shufflevector <32 x float> %898, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !26
  %930 = fadd <4 x float> %690, %929, !dbg !26
  %931 = extractelement <4 x float> %930, i64 0, !dbg !26
  %932 = extractelement <4 x float> %930, i64 1, !dbg !26
  %933 = extractelement <4 x float> %930, i64 2, !dbg !26
  %934 = extractelement <4 x float> %930, i64 3, !dbg !26
  %935 = shufflevector <32 x float> %898, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !26
  %936 = fadd <4 x float> %694, %935, !dbg !26
  %937 = extractelement <4 x float> %936, i64 0, !dbg !26
  %938 = extractelement <4 x float> %936, i64 1, !dbg !26
  %939 = extractelement <4 x float> %936, i64 2, !dbg !26
  %940 = extractelement <4 x float> %936, i64 3, !dbg !26
  %941 = shufflevector <32 x float> %898, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !26
  %942 = fadd <4 x float> %699, %941, !dbg !26
  %943 = extractelement <4 x float> %942, i64 0, !dbg !26
  %944 = extractelement <4 x float> %942, i64 1, !dbg !26
  %945 = extractelement <4 x float> %942, i64 2, !dbg !26
  %946 = extractelement <4 x float> %942, i64 3, !dbg !26
  %947 = call float @llvm.maxnum.f32(float %852, float 0xFFF0000000000000), !dbg !27
  %948 = call float @llvm.maxnum.f32(float %947, float %853), !dbg !27
  %949 = call float @llvm.maxnum.f32(float %948, float %854), !dbg !27
  %950 = call float @llvm.maxnum.f32(float %949, float %855), !dbg !27
  %951 = call float @llvm.maxnum.f32(float %950, float %858), !dbg !27
  %952 = call float @llvm.maxnum.f32(float %951, float %859), !dbg !27
  %953 = call float @llvm.maxnum.f32(float %952, float %860), !dbg !27
  %954 = call float @llvm.maxnum.f32(float %953, float %861), !dbg !27
  %955 = call float @llvm.maxnum.f32(float %954, float %864), !dbg !27
  %956 = call float @llvm.maxnum.f32(float %955, float %865), !dbg !27
  %957 = call float @llvm.maxnum.f32(float %956, float %866), !dbg !27
  %958 = call float @llvm.maxnum.f32(float %957, float %867), !dbg !27
  %959 = call float @llvm.maxnum.f32(float %958, float %870), !dbg !27
  %960 = call float @llvm.maxnum.f32(float %959, float %871), !dbg !27
  %961 = call float @llvm.maxnum.f32(float %960, float %872), !dbg !27
  %962 = call float @llvm.maxnum.f32(float %961, float %873), !dbg !27
  %963 = call float @llvm.maxnum.f32(float %962, float %876), !dbg !27
  %964 = call float @llvm.maxnum.f32(float %963, float %877), !dbg !27
  %965 = call float @llvm.maxnum.f32(float %964, float %878), !dbg !27
  %966 = call float @llvm.maxnum.f32(float %965, float %879), !dbg !27
  %967 = call float @llvm.maxnum.f32(float %966, float %882), !dbg !27
  %968 = call float @llvm.maxnum.f32(float %967, float %883), !dbg !27
  %969 = call float @llvm.maxnum.f32(float %968, float %884), !dbg !27
  %970 = call float @llvm.maxnum.f32(float %969, float %885), !dbg !27
  %971 = call float @llvm.maxnum.f32(float %970, float %888), !dbg !27
  %972 = call float @llvm.maxnum.f32(float %971, float %889), !dbg !27
  %973 = call float @llvm.maxnum.f32(float %972, float %890), !dbg !27
  %974 = call float @llvm.maxnum.f32(float %973, float %891), !dbg !27
  %975 = call float @llvm.maxnum.f32(float %974, float %894), !dbg !27
  %976 = call float @llvm.maxnum.f32(float %975, float %895), !dbg !27
  %977 = call float @llvm.maxnum.f32(float %976, float %896), !dbg !27
  %978 = call float @llvm.maxnum.f32(float %977, float %897), !dbg !27
  %979 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !27
  %980 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %979), !dbg !27
  %981 = add nuw nsw i32 %980, 64, !dbg !27
  %982 = xor i32 %980, 16, !dbg !27
  %983 = and i32 %981, 192, !dbg !27
  %984 = icmp samesign ult i32 %982, %983, !dbg !27
  %985 = select i1 %984, i32 %982, i32 %980, !dbg !27
  %986 = shl nuw nsw i32 %985, 2, !dbg !27
  %987 = bitcast float %978 to i32, !dbg !27
  %988 = call i32 @llvm.amdgcn.ds.bpermute(i32 %986, i32 %987), !dbg !27
  %989 = bitcast i32 %988 to float, !dbg !27
  %990 = call float @llvm.maxnum.f32(float %978, float %989), !dbg !27
  %991 = xor i32 %980, 32, !dbg !27
  %992 = icmp samesign ult i32 %991, %983, !dbg !27
  %993 = select i1 %992, i32 %991, i32 %980, !dbg !27
  %994 = shl nuw nsw i32 %993, 2, !dbg !27
  %995 = bitcast float %990 to i32, !dbg !27
  %996 = call i32 @llvm.amdgcn.ds.bpermute(i32 %994, i32 %995), !dbg !27
  %997 = bitcast i32 %996 to float, !dbg !27
  %998 = call float @llvm.maxnum.f32(float %990, float %997), !dbg !27
  %999 = extractvalue [2 x <1 x float>] %208, 0, !dbg !27
  %1000 = insertelement <1 x float> poison, float %998, i64 0, !dbg !27
  %1001 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %1000, <1 x float> %999), !dbg !27
  %1002 = call float @llvm.maxnum.f32(float %901, float 0xFFF0000000000000), !dbg !27
  %1003 = call float @llvm.maxnum.f32(float %1002, float %902), !dbg !27
  %1004 = call float @llvm.maxnum.f32(float %1003, float %903), !dbg !27
  %1005 = call float @llvm.maxnum.f32(float %1004, float %904), !dbg !27
  %1006 = call float @llvm.maxnum.f32(float %1005, float %907), !dbg !27
  %1007 = call float @llvm.maxnum.f32(float %1006, float %908), !dbg !27
  %1008 = call float @llvm.maxnum.f32(float %1007, float %909), !dbg !27
  %1009 = call float @llvm.maxnum.f32(float %1008, float %910), !dbg !27
  %1010 = call float @llvm.maxnum.f32(float %1009, float %913), !dbg !27
  %1011 = call float @llvm.maxnum.f32(float %1010, float %914), !dbg !27
  %1012 = call float @llvm.maxnum.f32(float %1011, float %915), !dbg !27
  %1013 = call float @llvm.maxnum.f32(float %1012, float %916), !dbg !27
  %1014 = call float @llvm.maxnum.f32(float %1013, float %919), !dbg !27
  %1015 = call float @llvm.maxnum.f32(float %1014, float %920), !dbg !27
  %1016 = call float @llvm.maxnum.f32(float %1015, float %921), !dbg !27
  %1017 = call float @llvm.maxnum.f32(float %1016, float %922), !dbg !27
  %1018 = call float @llvm.maxnum.f32(float %1017, float %925), !dbg !27
  %1019 = call float @llvm.maxnum.f32(float %1018, float %926), !dbg !27
  %1020 = call float @llvm.maxnum.f32(float %1019, float %927), !dbg !27
  %1021 = call float @llvm.maxnum.f32(float %1020, float %928), !dbg !27
  %1022 = call float @llvm.maxnum.f32(float %1021, float %931), !dbg !27
  %1023 = call float @llvm.maxnum.f32(float %1022, float %932), !dbg !27
  %1024 = call float @llvm.maxnum.f32(float %1023, float %933), !dbg !27
  %1025 = call float @llvm.maxnum.f32(float %1024, float %934), !dbg !27
  %1026 = call float @llvm.maxnum.f32(float %1025, float %937), !dbg !27
  %1027 = call float @llvm.maxnum.f32(float %1026, float %938), !dbg !27
  %1028 = call float @llvm.maxnum.f32(float %1027, float %939), !dbg !27
  %1029 = call float @llvm.maxnum.f32(float %1028, float %940), !dbg !27
  %1030 = call float @llvm.maxnum.f32(float %1029, float %943), !dbg !27
  %1031 = call float @llvm.maxnum.f32(float %1030, float %944), !dbg !27
  %1032 = call float @llvm.maxnum.f32(float %1031, float %945), !dbg !27
  %1033 = call float @llvm.maxnum.f32(float %1032, float %946), !dbg !27
  %1034 = bitcast float %1033 to i32, !dbg !27
  %1035 = call i32 @llvm.amdgcn.ds.bpermute(i32 %986, i32 %1034), !dbg !27
  %1036 = bitcast i32 %1035 to float, !dbg !27
  %1037 = call float @llvm.maxnum.f32(float %1033, float %1036), !dbg !27
  %1038 = bitcast float %1037 to i32, !dbg !27
  %1039 = call i32 @llvm.amdgcn.ds.bpermute(i32 %994, i32 %1038), !dbg !27
  %1040 = bitcast i32 %1039 to float, !dbg !27
  %1041 = call float @llvm.maxnum.f32(float %1037, float %1040), !dbg !27
  %1042 = extractvalue [2 x <1 x float>] %208, 1, !dbg !27
  %1043 = insertelement <1 x float> poison, float %1041, i64 0, !dbg !27
  %1044 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %1043, <1 x float> %1042), !dbg !27
  %1045 = shufflevector <1 x float> %1001, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !28
  %1046 = fsub <4 x float> %851, %1045, !dbg !28
  %1047 = fsub <4 x float> %857, %1045, !dbg !28
  %1048 = fsub <4 x float> %863, %1045, !dbg !28
  %1049 = fsub <4 x float> %869, %1045, !dbg !28
  %1050 = fsub <4 x float> %875, %1045, !dbg !28
  %1051 = fsub <4 x float> %881, %1045, !dbg !28
  %1052 = fsub <4 x float> %887, %1045, !dbg !28
  %1053 = fsub <4 x float> %893, %1045, !dbg !28
  %1054 = shufflevector <1 x float> %1044, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !28
  %1055 = fsub <4 x float> %900, %1054, !dbg !28
  %1056 = fsub <4 x float> %906, %1054, !dbg !28
  %1057 = fsub <4 x float> %912, %1054, !dbg !28
  %1058 = fsub <4 x float> %918, %1054, !dbg !28
  %1059 = fsub <4 x float> %924, %1054, !dbg !28
  %1060 = fsub <4 x float> %930, %1054, !dbg !28
  %1061 = fsub <4 x float> %936, %1054, !dbg !28
  %1062 = fsub <4 x float> %942, %1054, !dbg !28
  %1063 = fsub <1 x float> %999, %1001, !dbg !29
  %1064 = extractelement <1 x float> %1063, i64 0, !dbg !30
  %1065 = call float @llvm.amdgcn.exp2.f32(float %1064), !dbg !30
  %1066 = insertelement <4 x float> poison, float %1065, i64 0, !dbg !35
  %1067 = shufflevector <4 x float> %1066, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !35
  %1068 = extractvalue [2 x <1 x float>] %207, 0, !dbg !31
  %1069 = extractelement <1 x float> %1068, i64 0, !dbg !31
  %1070 = fmul float %1069, %1065, !dbg !31
  %1071 = fsub <1 x float> %1042, %1044, !dbg !29
  %1072 = extractelement <1 x float> %1071, i64 0, !dbg !30
  %1073 = call float @llvm.amdgcn.exp2.f32(float %1072), !dbg !30
  %1074 = insertelement <4 x float> poison, float %1073, i64 0, !dbg !35
  %1075 = shufflevector <4 x float> %1074, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !35
  %1076 = extractvalue [2 x <1 x float>] %207, 1, !dbg !31
  %1077 = extractelement <1 x float> %1076, i64 0, !dbg !31
  %1078 = fmul float %1077, %1073, !dbg !31
  %1079 = extractelement <4 x float> %1046, i64 0, !dbg !32
  %1080 = call float @llvm.amdgcn.exp2.f32(float %1079), !dbg !32
  %1081 = insertelement <4 x float> poison, float %1080, i64 0, !dbg !34
  %1082 = extractelement <4 x float> %1046, i64 1, !dbg !32
  %1083 = call float @llvm.amdgcn.exp2.f32(float %1082), !dbg !32
  %1084 = insertelement <4 x float> %1081, float %1083, i64 1, !dbg !34
  %1085 = fadd float %1080, %1083, !dbg !33
  %1086 = extractelement <4 x float> %1046, i64 2, !dbg !32
  %1087 = call float @llvm.amdgcn.exp2.f32(float %1086), !dbg !32
  %1088 = insertelement <4 x float> %1084, float %1087, i64 2, !dbg !34
  %1089 = fadd float %1085, %1087, !dbg !33
  %1090 = extractelement <4 x float> %1046, i64 3, !dbg !32
  %1091 = call float @llvm.amdgcn.exp2.f32(float %1090), !dbg !32
  %1092 = insertelement <4 x float> %1088, float %1091, i64 3, !dbg !34
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1092) #9, !dbg !34, !frisk.register_lifetimes !1
  %1093 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !34
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1093) #9, !dbg !33, !frisk.register_lifetimes !1
  %1094 = fadd float %1089, %1091, !dbg !33
  %1095 = extractelement <4 x float> %1047, i64 0, !dbg !32
  %1096 = call float @llvm.amdgcn.exp2.f32(float %1095), !dbg !32
  %1097 = insertelement <4 x float> poison, float %1096, i64 0, !dbg !34
  %1098 = fadd float %1094, %1096, !dbg !33
  %1099 = extractelement <4 x float> %1047, i64 1, !dbg !32
  %1100 = call float @llvm.amdgcn.exp2.f32(float %1099), !dbg !32
  %1101 = insertelement <4 x float> %1097, float %1100, i64 1, !dbg !34
  %1102 = fadd float %1098, %1100, !dbg !33
  %1103 = extractelement <4 x float> %1047, i64 2, !dbg !32
  %1104 = call float @llvm.amdgcn.exp2.f32(float %1103), !dbg !32
  %1105 = insertelement <4 x float> %1101, float %1104, i64 2, !dbg !34
  %1106 = fadd float %1102, %1104, !dbg !33
  %1107 = extractelement <4 x float> %1047, i64 3, !dbg !32
  %1108 = call float @llvm.amdgcn.exp2.f32(float %1107), !dbg !32
  %1109 = insertelement <4 x float> %1105, float %1108, i64 3, !dbg !34
  %math.result309 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1109) #9, !dbg !34, !frisk.register_lifetimes !1
  %1110 = fptrunc <4 x float> %math.result309 to <4 x half>, !dbg !34
  %math.packed310 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1110) #9, !dbg !33, !frisk.register_lifetimes !1
  %1111 = fadd float %1106, %1108, !dbg !33
  %1112 = extractelement <4 x float> %1048, i64 0, !dbg !32
  %1113 = call float @llvm.amdgcn.exp2.f32(float %1112), !dbg !32
  %1114 = insertelement <4 x float> poison, float %1113, i64 0, !dbg !34
  %1115 = fadd float %1111, %1113, !dbg !33
  %1116 = extractelement <4 x float> %1048, i64 1, !dbg !32
  %1117 = call float @llvm.amdgcn.exp2.f32(float %1116), !dbg !32
  %1118 = insertelement <4 x float> %1114, float %1117, i64 1, !dbg !34
  %1119 = fadd float %1115, %1117, !dbg !33
  %1120 = extractelement <4 x float> %1048, i64 2, !dbg !32
  %1121 = call float @llvm.amdgcn.exp2.f32(float %1120), !dbg !32
  %1122 = insertelement <4 x float> %1118, float %1121, i64 2, !dbg !34
  %1123 = fadd float %1119, %1121, !dbg !33
  %1124 = extractelement <4 x float> %1048, i64 3, !dbg !32
  %1125 = call float @llvm.amdgcn.exp2.f32(float %1124), !dbg !32
  %1126 = insertelement <4 x float> %1122, float %1125, i64 3, !dbg !34
  %math.result311 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1126) #9, !dbg !34, !frisk.register_lifetimes !1
  %1127 = fptrunc <4 x float> %math.result311 to <4 x half>, !dbg !34
  %math.packed312 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1127) #9, !dbg !33, !frisk.register_lifetimes !1
  %1128 = fadd float %1123, %1125, !dbg !33
  %1129 = extractelement <4 x float> %1049, i64 0, !dbg !32
  %1130 = call float @llvm.amdgcn.exp2.f32(float %1129), !dbg !32
  %1131 = insertelement <4 x float> poison, float %1130, i64 0, !dbg !34
  %1132 = fadd float %1128, %1130, !dbg !33
  %1133 = extractelement <4 x float> %1049, i64 1, !dbg !32
  %1134 = call float @llvm.amdgcn.exp2.f32(float %1133), !dbg !32
  %1135 = insertelement <4 x float> %1131, float %1134, i64 1, !dbg !34
  %1136 = fadd float %1132, %1134, !dbg !33
  %1137 = extractelement <4 x float> %1049, i64 2, !dbg !32
  %1138 = call float @llvm.amdgcn.exp2.f32(float %1137), !dbg !32
  %1139 = insertelement <4 x float> %1135, float %1138, i64 2, !dbg !34
  %1140 = fadd float %1136, %1138, !dbg !33
  %1141 = extractelement <4 x float> %1049, i64 3, !dbg !32
  %1142 = call float @llvm.amdgcn.exp2.f32(float %1141), !dbg !32
  %1143 = insertelement <4 x float> %1139, float %1142, i64 3, !dbg !34
  %math.result313 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1143) #9, !dbg !34, !frisk.register_lifetimes !1
  %1144 = fptrunc <4 x float> %math.result313 to <4 x half>, !dbg !34
  %math.packed314 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1144) #9, !dbg !33, !frisk.register_lifetimes !1
  %1145 = fadd float %1140, %1142, !dbg !33
  %1146 = extractelement <4 x float> %1050, i64 0, !dbg !32
  %1147 = call float @llvm.amdgcn.exp2.f32(float %1146), !dbg !32
  %1148 = insertelement <4 x float> poison, float %1147, i64 0, !dbg !34
  %1149 = fadd float %1145, %1147, !dbg !33
  %1150 = extractelement <4 x float> %1050, i64 1, !dbg !32
  %1151 = call float @llvm.amdgcn.exp2.f32(float %1150), !dbg !32
  %1152 = insertelement <4 x float> %1148, float %1151, i64 1, !dbg !34
  %1153 = fadd float %1149, %1151, !dbg !33
  %1154 = extractelement <4 x float> %1050, i64 2, !dbg !32
  %1155 = call float @llvm.amdgcn.exp2.f32(float %1154), !dbg !32
  %1156 = insertelement <4 x float> %1152, float %1155, i64 2, !dbg !34
  %1157 = fadd float %1153, %1155, !dbg !33
  %1158 = extractelement <4 x float> %1050, i64 3, !dbg !32
  %1159 = call float @llvm.amdgcn.exp2.f32(float %1158), !dbg !32
  %1160 = insertelement <4 x float> %1156, float %1159, i64 3, !dbg !34
  %math.result315 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1160) #9, !dbg !34, !frisk.register_lifetimes !1
  %1161 = fptrunc <4 x float> %math.result315 to <4 x half>, !dbg !34
  %math.packed316 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1161) #9, !dbg !33, !frisk.register_lifetimes !1
  %1162 = fadd float %1157, %1159, !dbg !33
  %1163 = extractelement <4 x float> %1051, i64 0, !dbg !32
  %1164 = call float @llvm.amdgcn.exp2.f32(float %1163), !dbg !32
  %1165 = insertelement <4 x float> poison, float %1164, i64 0, !dbg !34
  %1166 = fadd float %1162, %1164, !dbg !33
  %1167 = extractelement <4 x float> %1051, i64 1, !dbg !32
  %1168 = call float @llvm.amdgcn.exp2.f32(float %1167), !dbg !32
  %1169 = insertelement <4 x float> %1165, float %1168, i64 1, !dbg !34
  %1170 = fadd float %1166, %1168, !dbg !33
  %1171 = extractelement <4 x float> %1051, i64 2, !dbg !32
  %1172 = call float @llvm.amdgcn.exp2.f32(float %1171), !dbg !32
  %1173 = insertelement <4 x float> %1169, float %1172, i64 2, !dbg !34
  %1174 = fadd float %1170, %1172, !dbg !33
  %1175 = extractelement <4 x float> %1051, i64 3, !dbg !32
  %1176 = call float @llvm.amdgcn.exp2.f32(float %1175), !dbg !32
  %1177 = insertelement <4 x float> %1173, float %1176, i64 3, !dbg !34
  %math.result317 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1177) #9, !dbg !34, !frisk.register_lifetimes !1
  %1178 = fptrunc <4 x float> %math.result317 to <4 x half>, !dbg !34
  %math.packed318 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1178) #9, !dbg !33, !frisk.register_lifetimes !1
  %1179 = fadd float %1174, %1176, !dbg !33
  %1180 = extractelement <4 x float> %1052, i64 0, !dbg !32
  %1181 = call float @llvm.amdgcn.exp2.f32(float %1180), !dbg !32
  %1182 = insertelement <4 x float> poison, float %1181, i64 0, !dbg !34
  %1183 = fadd float %1179, %1181, !dbg !33
  %1184 = extractelement <4 x float> %1052, i64 1, !dbg !32
  %1185 = call float @llvm.amdgcn.exp2.f32(float %1184), !dbg !32
  %1186 = insertelement <4 x float> %1182, float %1185, i64 1, !dbg !34
  %1187 = fadd float %1183, %1185, !dbg !33
  %1188 = extractelement <4 x float> %1052, i64 2, !dbg !32
  %1189 = call float @llvm.amdgcn.exp2.f32(float %1188), !dbg !32
  %1190 = insertelement <4 x float> %1186, float %1189, i64 2, !dbg !34
  %1191 = fadd float %1187, %1189, !dbg !33
  %1192 = extractelement <4 x float> %1052, i64 3, !dbg !32
  %1193 = call float @llvm.amdgcn.exp2.f32(float %1192), !dbg !32
  %1194 = insertelement <4 x float> %1190, float %1193, i64 3, !dbg !34
  %math.result319 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1194) #9, !dbg !34, !frisk.register_lifetimes !1
  %1195 = fptrunc <4 x float> %math.result319 to <4 x half>, !dbg !34
  %math.packed320 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1195) #9, !dbg !33, !frisk.register_lifetimes !1
  %1196 = fadd float %1191, %1193, !dbg !33
  %1197 = extractelement <4 x float> %1053, i64 0, !dbg !32
  %1198 = call float @llvm.amdgcn.exp2.f32(float %1197), !dbg !32
  %1199 = insertelement <4 x float> poison, float %1198, i64 0, !dbg !34
  %1200 = fadd float %1196, %1198, !dbg !33
  %1201 = extractelement <4 x float> %1053, i64 1, !dbg !32
  %1202 = call float @llvm.amdgcn.exp2.f32(float %1201), !dbg !32
  %1203 = insertelement <4 x float> %1199, float %1202, i64 1, !dbg !34
  %1204 = fadd float %1200, %1202, !dbg !33
  %1205 = extractelement <4 x float> %1053, i64 2, !dbg !32
  %1206 = call float @llvm.amdgcn.exp2.f32(float %1205), !dbg !32
  %1207 = insertelement <4 x float> %1203, float %1206, i64 2, !dbg !34
  %1208 = fadd float %1204, %1206, !dbg !33
  %1209 = extractelement <4 x float> %1053, i64 3, !dbg !32
  %1210 = call float @llvm.amdgcn.exp2.f32(float %1209), !dbg !32
  %1211 = insertelement <4 x float> %1207, float %1210, i64 3, !dbg !34
  %math.result321 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1211) #9, !dbg !34, !frisk.register_lifetimes !1
  %1212 = fptrunc <4 x float> %math.result321 to <4 x half>, !dbg !34
  %math.packed322 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1212) #9, !dbg !33, !frisk.register_lifetimes !1
  %1213 = fadd float %1208, %1210, !dbg !33
  %1214 = extractelement <4 x float> %1055, i64 0, !dbg !32
  %1215 = call float @llvm.amdgcn.exp2.f32(float %1214), !dbg !32
  %1216 = insertelement <4 x float> poison, float %1215, i64 0, !dbg !34
  %1217 = extractelement <4 x float> %1055, i64 1, !dbg !32
  %1218 = call float @llvm.amdgcn.exp2.f32(float %1217), !dbg !32
  %1219 = insertelement <4 x float> %1216, float %1218, i64 1, !dbg !34
  %1220 = fadd float %1215, %1218, !dbg !33
  %1221 = extractelement <4 x float> %1055, i64 2, !dbg !32
  %1222 = call float @llvm.amdgcn.exp2.f32(float %1221), !dbg !32
  %1223 = insertelement <4 x float> %1219, float %1222, i64 2, !dbg !34
  %1224 = fadd float %1220, %1222, !dbg !33
  %1225 = extractelement <4 x float> %1055, i64 3, !dbg !32
  %1226 = call float @llvm.amdgcn.exp2.f32(float %1225), !dbg !32
  %1227 = insertelement <4 x float> %1223, float %1226, i64 3, !dbg !34
  %math.result323 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1227) #9, !dbg !34, !frisk.register_lifetimes !1
  %1228 = fptrunc <4 x float> %math.result323 to <4 x half>, !dbg !34
  %math.packed324 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1228) #9, !dbg !33, !frisk.register_lifetimes !1
  %1229 = fadd float %1224, %1226, !dbg !33
  %1230 = extractelement <4 x float> %1056, i64 0, !dbg !32
  %1231 = call float @llvm.amdgcn.exp2.f32(float %1230), !dbg !32
  %1232 = insertelement <4 x float> poison, float %1231, i64 0, !dbg !34
  %1233 = fadd float %1229, %1231, !dbg !33
  %1234 = extractelement <4 x float> %1056, i64 1, !dbg !32
  %1235 = call float @llvm.amdgcn.exp2.f32(float %1234), !dbg !32
  %1236 = insertelement <4 x float> %1232, float %1235, i64 1, !dbg !34
  %1237 = fadd float %1233, %1235, !dbg !33
  %1238 = extractelement <4 x float> %1056, i64 2, !dbg !32
  %1239 = call float @llvm.amdgcn.exp2.f32(float %1238), !dbg !32
  %1240 = insertelement <4 x float> %1236, float %1239, i64 2, !dbg !34
  %1241 = fadd float %1237, %1239, !dbg !33
  %1242 = extractelement <4 x float> %1056, i64 3, !dbg !32
  %1243 = call float @llvm.amdgcn.exp2.f32(float %1242), !dbg !32
  %1244 = insertelement <4 x float> %1240, float %1243, i64 3, !dbg !34
  %math.result325 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1244) #9, !dbg !34, !frisk.register_lifetimes !1
  %1245 = fptrunc <4 x float> %math.result325 to <4 x half>, !dbg !34
  %math.packed326 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1245) #9, !dbg !33, !frisk.register_lifetimes !1
  %1246 = fadd float %1241, %1243, !dbg !33
  %1247 = extractelement <4 x float> %1057, i64 0, !dbg !32
  %1248 = call float @llvm.amdgcn.exp2.f32(float %1247), !dbg !32
  %1249 = insertelement <4 x float> poison, float %1248, i64 0, !dbg !34
  %1250 = fadd float %1246, %1248, !dbg !33
  %1251 = extractelement <4 x float> %1057, i64 1, !dbg !32
  %1252 = call float @llvm.amdgcn.exp2.f32(float %1251), !dbg !32
  %1253 = insertelement <4 x float> %1249, float %1252, i64 1, !dbg !34
  %1254 = fadd float %1250, %1252, !dbg !33
  %1255 = extractelement <4 x float> %1057, i64 2, !dbg !32
  %1256 = call float @llvm.amdgcn.exp2.f32(float %1255), !dbg !32
  %1257 = insertelement <4 x float> %1253, float %1256, i64 2, !dbg !34
  %1258 = fadd float %1254, %1256, !dbg !33
  %1259 = extractelement <4 x float> %1057, i64 3, !dbg !32
  %1260 = call float @llvm.amdgcn.exp2.f32(float %1259), !dbg !32
  %1261 = insertelement <4 x float> %1257, float %1260, i64 3, !dbg !34
  %math.result327 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1261) #9, !dbg !34, !frisk.register_lifetimes !1
  %1262 = fptrunc <4 x float> %math.result327 to <4 x half>, !dbg !34
  %math.packed328 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1262) #9, !dbg !33, !frisk.register_lifetimes !1
  %1263 = fadd float %1258, %1260, !dbg !33
  %1264 = extractelement <4 x float> %1058, i64 0, !dbg !32
  %1265 = call float @llvm.amdgcn.exp2.f32(float %1264), !dbg !32
  %1266 = insertelement <4 x float> poison, float %1265, i64 0, !dbg !34
  %1267 = fadd float %1263, %1265, !dbg !33
  %1268 = extractelement <4 x float> %1058, i64 1, !dbg !32
  %1269 = call float @llvm.amdgcn.exp2.f32(float %1268), !dbg !32
  %1270 = insertelement <4 x float> %1266, float %1269, i64 1, !dbg !34
  %1271 = fadd float %1267, %1269, !dbg !33
  %1272 = extractelement <4 x float> %1058, i64 2, !dbg !32
  %1273 = call float @llvm.amdgcn.exp2.f32(float %1272), !dbg !32
  %1274 = insertelement <4 x float> %1270, float %1273, i64 2, !dbg !34
  %1275 = fadd float %1271, %1273, !dbg !33
  %1276 = extractelement <4 x float> %1058, i64 3, !dbg !32
  %1277 = call float @llvm.amdgcn.exp2.f32(float %1276), !dbg !32
  %1278 = insertelement <4 x float> %1274, float %1277, i64 3, !dbg !34
  %math.result329 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1278) #9, !dbg !34, !frisk.register_lifetimes !1
  %1279 = fptrunc <4 x float> %math.result329 to <4 x half>, !dbg !34
  %math.packed330 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1279) #9, !dbg !33, !frisk.register_lifetimes !1
  %1280 = fadd float %1275, %1277, !dbg !33
  %1281 = extractelement <4 x float> %1059, i64 0, !dbg !32
  %1282 = call float @llvm.amdgcn.exp2.f32(float %1281), !dbg !32
  %1283 = insertelement <4 x float> poison, float %1282, i64 0, !dbg !34
  %1284 = fadd float %1280, %1282, !dbg !33
  %1285 = extractelement <4 x float> %1059, i64 1, !dbg !32
  %1286 = call float @llvm.amdgcn.exp2.f32(float %1285), !dbg !32
  %1287 = insertelement <4 x float> %1283, float %1286, i64 1, !dbg !34
  %1288 = fadd float %1284, %1286, !dbg !33
  %1289 = extractelement <4 x float> %1059, i64 2, !dbg !32
  %1290 = call float @llvm.amdgcn.exp2.f32(float %1289), !dbg !32
  %1291 = insertelement <4 x float> %1287, float %1290, i64 2, !dbg !34
  %1292 = fadd float %1288, %1290, !dbg !33
  %1293 = extractelement <4 x float> %1059, i64 3, !dbg !32
  %1294 = call float @llvm.amdgcn.exp2.f32(float %1293), !dbg !32
  %1295 = insertelement <4 x float> %1291, float %1294, i64 3, !dbg !34
  %math.result331 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1295) #9, !dbg !34, !frisk.register_lifetimes !1
  %1296 = fptrunc <4 x float> %math.result331 to <4 x half>, !dbg !34
  %math.packed332 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1296) #9, !dbg !33, !frisk.register_lifetimes !1
  %1297 = fadd float %1292, %1294, !dbg !33
  %1298 = extractelement <4 x float> %1060, i64 0, !dbg !32
  %1299 = call float @llvm.amdgcn.exp2.f32(float %1298), !dbg !32
  %1300 = insertelement <4 x float> poison, float %1299, i64 0, !dbg !34
  %1301 = fadd float %1297, %1299, !dbg !33
  %1302 = extractelement <4 x float> %1060, i64 1, !dbg !32
  %1303 = call float @llvm.amdgcn.exp2.f32(float %1302), !dbg !32
  %1304 = insertelement <4 x float> %1300, float %1303, i64 1, !dbg !34
  %1305 = fadd float %1301, %1303, !dbg !33
  %1306 = extractelement <4 x float> %1060, i64 2, !dbg !32
  %1307 = call float @llvm.amdgcn.exp2.f32(float %1306), !dbg !32
  %1308 = insertelement <4 x float> %1304, float %1307, i64 2, !dbg !34
  %1309 = fadd float %1305, %1307, !dbg !33
  %1310 = extractelement <4 x float> %1060, i64 3, !dbg !32
  %1311 = call float @llvm.amdgcn.exp2.f32(float %1310), !dbg !32
  %1312 = insertelement <4 x float> %1308, float %1311, i64 3, !dbg !34
  %math.result333 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1312) #9, !dbg !34, !frisk.register_lifetimes !1
  %1313 = fptrunc <4 x float> %math.result333 to <4 x half>, !dbg !34
  %math.packed334 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1313) #9, !dbg !33, !frisk.register_lifetimes !1
  %1314 = fadd float %1309, %1311, !dbg !33
  %1315 = extractelement <4 x float> %1061, i64 0, !dbg !32
  %1316 = call float @llvm.amdgcn.exp2.f32(float %1315), !dbg !32
  %1317 = insertelement <4 x float> poison, float %1316, i64 0, !dbg !34
  %1318 = fadd float %1314, %1316, !dbg !33
  %1319 = extractelement <4 x float> %1061, i64 1, !dbg !32
  %1320 = call float @llvm.amdgcn.exp2.f32(float %1319), !dbg !32
  %1321 = insertelement <4 x float> %1317, float %1320, i64 1, !dbg !34
  %1322 = fadd float %1318, %1320, !dbg !33
  %1323 = extractelement <4 x float> %1061, i64 2, !dbg !32
  %1324 = call float @llvm.amdgcn.exp2.f32(float %1323), !dbg !32
  %1325 = insertelement <4 x float> %1321, float %1324, i64 2, !dbg !34
  %1326 = fadd float %1322, %1324, !dbg !33
  %1327 = extractelement <4 x float> %1061, i64 3, !dbg !32
  %1328 = call float @llvm.amdgcn.exp2.f32(float %1327), !dbg !32
  %1329 = insertelement <4 x float> %1325, float %1328, i64 3, !dbg !34
  %math.result335 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1329) #9, !dbg !34, !frisk.register_lifetimes !1
  %1330 = fptrunc <4 x float> %math.result335 to <4 x half>, !dbg !34
  %math.packed336 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1330) #9, !dbg !33, !frisk.register_lifetimes !1
  %1331 = fadd float %1326, %1328, !dbg !33
  %1332 = extractelement <4 x float> %1062, i64 0, !dbg !32
  %1333 = call float @llvm.amdgcn.exp2.f32(float %1332), !dbg !32
  %1334 = insertelement <4 x float> poison, float %1333, i64 0, !dbg !34
  %1335 = fadd float %1331, %1333, !dbg !33
  %1336 = extractelement <4 x float> %1062, i64 1, !dbg !32
  %1337 = call float @llvm.amdgcn.exp2.f32(float %1336), !dbg !32
  %1338 = insertelement <4 x float> %1334, float %1337, i64 1, !dbg !34
  %1339 = fadd float %1335, %1337, !dbg !33
  %1340 = extractelement <4 x float> %1062, i64 2, !dbg !32
  %1341 = call float @llvm.amdgcn.exp2.f32(float %1340), !dbg !32
  %1342 = insertelement <4 x float> %1338, float %1341, i64 2, !dbg !34
  %1343 = fadd float %1339, %1341, !dbg !33
  %1344 = extractelement <4 x float> %1062, i64 3, !dbg !32
  %1345 = call float @llvm.amdgcn.exp2.f32(float %1344), !dbg !32
  %1346 = insertelement <4 x float> %1342, float %1345, i64 3, !dbg !34
  %math.result337 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1346) #9, !dbg !34, !frisk.register_lifetimes !1
  %1347 = fptrunc <4 x float> %math.result337 to <4 x half>, !dbg !34
  %math.packed338 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1347) #9, !dbg !33, !frisk.register_lifetimes !1
  %1348 = fadd float %1343, %1345, !dbg !33
  %1349 = bitcast float %1213 to i32, !dbg !33
  %1350 = call i32 @llvm.amdgcn.ds.bpermute(i32 %986, i32 %1349), !dbg !33
  %1351 = bitcast i32 %1350 to float, !dbg !33
  %1352 = fadd float %1213, %1351, !dbg !33
  %1353 = bitcast float %1352 to i32, !dbg !33
  %1354 = call i32 @llvm.amdgcn.ds.bpermute(i32 %994, i32 %1353), !dbg !33
  %1355 = bitcast i32 %1354 to float, !dbg !33
  %1356 = fadd float %1352, %1355, !dbg !33
  %1357 = fadd float %1356, %1070, !dbg !33
  %1358 = insertelement <1 x float> poison, float %1357, i64 0, !dbg !7
  %1359 = insertvalue [2 x <1 x float>] poison, <1 x float> %1358, 0, !dbg !7
  %1360 = bitcast float %1348 to i32, !dbg !33
  %1361 = call i32 @llvm.amdgcn.ds.bpermute(i32 %986, i32 %1360), !dbg !33
  %1362 = bitcast i32 %1361 to float, !dbg !33
  %1363 = fadd float %1348, %1362, !dbg !33
  %1364 = bitcast float %1363 to i32, !dbg !33
  %1365 = call i32 @llvm.amdgcn.ds.bpermute(i32 %994, i32 %1364), !dbg !33
  %1366 = bitcast i32 %1365 to float, !dbg !33
  %1367 = fadd float %1363, %1366, !dbg !33
  %1368 = fadd float %1367, %1078, !dbg !33
  %1369 = insertelement <1 x float> poison, float %1368, i64 0, !dbg !7
  %1370 = insertvalue [2 x <1 x float>] %1359, <1 x float> %1369, 1, !dbg !7
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %address.thread864 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !14, !range !39, !frisk.register_lifetimes !1
  %address.local865 = shl nuw nsw i32 %address.thread864, 3, !dbg !14
  %address.local866 = and i32 %address.local865, 120, !dbg !14
  %address.local867 = shl nuw nsw i32 %address.thread864, 5, !dbg !14
  %address.local868 = and i32 %address.local867, 1536, !dbg !14
  %address.local869 = or disjoint i32 %address.local866, %address.local868, !dbg !14
  %address.local870 = zext nneg i32 %address.local869 to i64, !dbg !14
  %address.local871 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local870, !dbg !14
  %1371 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local871, i32 0) #0, !dbg !14
  %1372 = bitcast <4 x i32> %1371 to <8 x half>, !dbg !14
  %address.local872 = zext nneg i32 %address.thread864 to i64, !dbg !8
  %address.local873 = and i64 %address.local872, 60, !dbg !14
  %address.local874 = add nuw nsw i64 %address.local873, %address.local872, !dbg !14
  %address.local875 = shl nuw nsw i64 %address.local874, 3, !dbg !14
  %address.local876 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local875, !dbg !14
  %address.local877 = and i64 %address.local872, 63, !dbg !11
  %address.local878 = lshr i64 %address.local877, 4, !dbg !14
  %address.local879 = mul nuw nsw i64 %address.local878, 384, !dbg !14
  %address.local880 = getelementptr [2 x i8], ptr addrspace(3) %address.local876, i64 %address.local879, !dbg !14
  %address.local881 = lshr i64 %address.local872, 2, !dbg !17
  %address.local882 = mul nsw i64 %address.local881, -32, !dbg !14
  %address.local883 = getelementptr [2 x i8], ptr addrspace(3) %address.local880, i64 %address.local882, !dbg !14
  %address.local884 = getelementptr i8, ptr addrspace(3) %address.local883, i64 256, !dbg !14
  %1373 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local884, i32 0) #0, !dbg !14
  %1374 = bitcast <4 x i32> %1373 to <8 x half>, !dbg !14
  %address.local885 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local875, !dbg !14
  %address.local886 = getelementptr [2 x i8], ptr addrspace(3) %address.local885, i64 %address.local879, !dbg !14
  %address.local887 = getelementptr [2 x i8], ptr addrspace(3) %address.local886, i64 %address.local882, !dbg !14
  %address.local888 = getelementptr i8, ptr addrspace(3) %address.local887, i64 512, !dbg !14
  %1375 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local888, i32 0) #0, !dbg !14
  %1376 = bitcast <4 x i32> %1375 to <8 x half>, !dbg !14
  %address.local889 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local875, !dbg !14
  %address.local890 = getelementptr [2 x i8], ptr addrspace(3) %address.local889, i64 %address.local879, !dbg !14
  %address.local891 = getelementptr [2 x i8], ptr addrspace(3) %address.local890, i64 %address.local882, !dbg !14
  %address.local892 = getelementptr i8, ptr addrspace(3) %address.local891, i64 768, !dbg !14
  %1377 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local892, i32 0) #0, !dbg !14
  %1378 = bitcast <4 x i32> %1377 to <8 x half>, !dbg !14
  %address.local893 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local870, !dbg !14
  %address.local894 = getelementptr i8, ptr addrspace(3) %address.local893, i64 4096, !dbg !14
  %1379 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local894, i32 0) #0, !dbg !14
  %1380 = bitcast <4 x i32> %1379 to <8 x half>, !dbg !14
  %address.local895 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local875, !dbg !14
  %address.local896 = getelementptr [2 x i8], ptr addrspace(3) %address.local895, i64 %address.local879, !dbg !14
  %address.local897 = getelementptr [2 x i8], ptr addrspace(3) %address.local896, i64 %address.local882, !dbg !14
  %address.local898 = getelementptr i8, ptr addrspace(3) %address.local897, i64 4352, !dbg !14
  %1381 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local898, i32 0) #0, !dbg !14
  %1382 = bitcast <4 x i32> %1381 to <8 x half>, !dbg !14
  %address.local899 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local875, !dbg !14
  %address.local900 = getelementptr [2 x i8], ptr addrspace(3) %address.local899, i64 %address.local879, !dbg !14
  %address.local901 = getelementptr [2 x i8], ptr addrspace(3) %address.local900, i64 %address.local882, !dbg !14
  %address.local902 = getelementptr i8, ptr addrspace(3) %address.local901, i64 4608, !dbg !14
  %1383 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local902, i32 0) #0, !dbg !14
  %1384 = bitcast <4 x i32> %1383 to <8 x half>, !dbg !14
  %address.local903 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local875, !dbg !14
  %address.local904 = getelementptr [2 x i8], ptr addrspace(3) %address.local903, i64 %address.local879, !dbg !14
  %address.local905 = getelementptr [2 x i8], ptr addrspace(3) %address.local904, i64 %address.local882, !dbg !14
  %address.local906 = getelementptr i8, ptr addrspace(3) %address.local905, i64 4864, !dbg !14
  %1385 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local906, i32 0) #0, !dbg !14
  %1386 = bitcast <4 x i32> %1385 to <8 x half>, !dbg !14
  %1387 = add nuw nsw i64 %217, %16, !dbg !18
  %1388 = shl i64 %1387, 19, !dbg !18
  %1389 = and i64 %1388, 16252928, !dbg !18
  %1390 = add i64 %1389, %223, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1391 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %1390, !dbg !18
  %1392 = getelementptr i8, ptr addrspace(1) %1391, i64 64, !dbg !18
  %address.local907 = shl nuw nsw i64 %address.local872, 3, !dbg !8
  %address.local908 = and i64 %address.local907, 1536, !dbg !8
  %address.local909 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local908, !dbg !8
  %address.local910 = ptrtoint ptr addrspace(3) %address.local909 to i64, !dbg !8
  %address.local911 = trunc i64 %address.local910 to i32, !dbg !8
  %1393 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local911), !dbg !18
  %1394 = zext i32 %1393 to i64, !dbg !18
  %1395 = inttoptr i64 %1394 to ptr addrspace(3), !dbg !18
  %address.local912 = shl nuw nsw i64 %address.local872, 5, !dbg !18
  %address.local913 = and i64 %address.local912, 8064, !dbg !18
  %address.local914 = shl nuw nsw i64 %address.local872, 3, !dbg !18
  %address.local915 = and i64 %address.local914, 24, !dbg !18
  %address.local916 = or disjoint i64 %address.local913, %address.local915, !dbg !18
  %address.local917 = getelementptr [2 x i8], ptr addrspace(1) %1392, i64 %address.local916, !dbg !18
  %1396 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1397 = ptrtoint ptr addrspace(1) %address.local917 to i64, !dbg !18
  %1398 = sub i64 %1397, %1396, !dbg !18
  %dma.buffer.offset1558 = trunc i64 %1398 to i32, !dbg !18
  %1399 = trunc i64 %1396 to i32, !dbg !18
  %1400 = insertelement <4 x i32> poison, i32 %1399, i64 0, !dbg !18
  %1401 = lshr i64 %1396, 32, !dbg !18
  %1402 = trunc i64 %1401 to i32, !dbg !18
  %1403 = insertelement <4 x i32> %1400, i32 %1402, i64 1, !dbg !18
  %1404 = insertelement <4 x i32> %1403, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1559 = insertelement <4 x i32> %1404, i32 131072, i64 3, !dbg !18
  %1405 = ptrtoint ptr addrspace(3) %1395 to i32, !dbg !18
  %dma.buffer.lds1560 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1405), !dbg !18
  %1406 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1560, i32 %dma.buffer.offset1558, <4 x i32> %dma.buffer.rsrc1559) #10, !dbg !18
  %address.local918 = or disjoint i64 %address.local872, 256, !dbg !8
  %address.local919 = shl nuw nsw i64 %address.local918, 3, !dbg !8
  %address.local920 = and i64 %address.local919, 3584, !dbg !8
  %address.local921 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local920, !dbg !8
  %address.local922 = ptrtoint ptr addrspace(3) %address.local921 to i64, !dbg !8
  %address.local923 = trunc i64 %address.local922 to i32, !dbg !8
  %1407 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local923), !dbg !18
  %1408 = zext i32 %1407 to i64, !dbg !18
  %1409 = inttoptr i64 %1408 to ptr addrspace(3), !dbg !18
  %address.local924 = shl nuw nsw i64 %address.local918, 5, !dbg !18
  %address.local925 = and i64 %address.local924, 16256, !dbg !18
  %address.local926 = shl nuw nsw i64 %address.local872, 3, !dbg !18
  %address.local927 = and i64 %address.local926, 24, !dbg !18
  %address.local928 = or disjoint i64 %address.local925, %address.local927, !dbg !18
  %address.local929 = getelementptr [2 x i8], ptr addrspace(1) %1392, i64 %address.local928, !dbg !18
  %1410 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1411 = ptrtoint ptr addrspace(1) %address.local929 to i64, !dbg !18
  %1412 = sub i64 %1411, %1410, !dbg !18
  %dma.buffer.offset1555 = trunc i64 %1412 to i32, !dbg !18
  %1413 = trunc i64 %1410 to i32, !dbg !18
  %1414 = insertelement <4 x i32> poison, i32 %1413, i64 0, !dbg !18
  %1415 = lshr i64 %1410, 32, !dbg !18
  %1416 = trunc i64 %1415 to i32, !dbg !18
  %1417 = insertelement <4 x i32> %1414, i32 %1416, i64 1, !dbg !18
  %1418 = insertelement <4 x i32> %1417, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1556 = insertelement <4 x i32> %1418, i32 131072, i64 3, !dbg !18
  %1419 = ptrtoint ptr addrspace(3) %1409 to i32, !dbg !18
  %dma.buffer.lds1557 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1419), !dbg !18
  %1420 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1557, i32 %dma.buffer.offset1555, <4 x i32> %dma.buffer.rsrc1556) #10, !dbg !18
  %1421 = shufflevector <8 x half> %1372, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1422 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1421, <4 x float> zeroinitializer) #9, !dbg !14
  %1423 = shufflevector <8 x half> %1374, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1424 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %1423, <4 x float> %1422) #9, !dbg !14
  %1425 = shufflevector <8 x half> %1376, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1426 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %1425, <4 x float> %1424) #9, !dbg !14
  %1427 = shufflevector <8 x half> %1378, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1428 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %1427, <4 x float> %1426) #9, !dbg !14
  %1429 = shufflevector <8 x half> %1380, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1430 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %1429, <4 x float> %1428) #9, !dbg !14
  %1431 = shufflevector <8 x half> %1382, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1432 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %1431, <4 x float> %1430) #9, !dbg !14
  %1433 = shufflevector <8 x half> %1384, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1434 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %1433, <4 x float> %1432) #9, !dbg !14
  %1435 = shufflevector <8 x half> %1386, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1436 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %1435, <4 x float> %1434) #9, !dbg !14
  %1437 = extractvalue [2 x <8 x float>] %206, 0, !dbg !35
  %1438 = shufflevector <8 x float> %1437, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1439 = fmul <4 x float> %1438, %1067, !dbg !35
  %1440 = fadd <4 x float> %1439, %1436, !dbg !36
  %1441 = shufflevector <8 x half> %1372, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1442 = shufflevector <8 x float> %1437, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1443 = fmul <4 x float> %1442, %1067, !dbg !35
  %1444 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1441, <4 x float> %1443) #9, !dbg !14
  %1445 = shufflevector <8 x half> %1374, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1446 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %1445, <4 x float> %1444) #9, !dbg !14
  %1447 = shufflevector <8 x half> %1376, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1448 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %1447, <4 x float> %1446) #9, !dbg !14
  %1449 = shufflevector <8 x half> %1378, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1450 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %1449, <4 x float> %1448) #9, !dbg !14
  %1451 = shufflevector <8 x half> %1380, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1452 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %1451, <4 x float> %1450) #9, !dbg !14
  %1453 = shufflevector <8 x half> %1382, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1454 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %1453, <4 x float> %1452) #9, !dbg !14
  %1455 = shufflevector <8 x half> %1384, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1456 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %1455, <4 x float> %1454) #9, !dbg !14
  %1457 = shufflevector <8 x half> %1386, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1458 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %1457, <4 x float> %1456) #9, !dbg !14
  %1459 = shufflevector <4 x float> %1440, <4 x float> %1458, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1460 = insertvalue [2 x <8 x float>] poison, <8 x float> %1459, 0, !dbg !7
  %1461 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1421, <4 x float> zeroinitializer) #9, !dbg !14
  %1462 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1423, <4 x float> %1461) #9, !dbg !14
  %1463 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1425, <4 x float> %1462) #9, !dbg !14
  %1464 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1427, <4 x float> %1463) #9, !dbg !14
  %1465 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %1429, <4 x float> %1464) #9, !dbg !14
  %1466 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %1431, <4 x float> %1465) #9, !dbg !14
  %1467 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %1433, <4 x float> %1466) #9, !dbg !14
  %1468 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %1435, <4 x float> %1467) #9, !dbg !14
  %1469 = extractvalue [2 x <8 x float>] %206, 1, !dbg !35
  %1470 = shufflevector <8 x float> %1469, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1471 = fmul <4 x float> %1470, %1075, !dbg !35
  %1472 = fadd <4 x float> %1471, %1468, !dbg !36
  %1473 = shufflevector <8 x float> %1469, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1474 = fmul <4 x float> %1473, %1075, !dbg !35
  %1475 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1441, <4 x float> %1474) #9, !dbg !14
  %1476 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1445, <4 x float> %1475) #9, !dbg !14
  %1477 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1447, <4 x float> %1476) #9, !dbg !14
  %1478 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1449, <4 x float> %1477) #9, !dbg !14
  %1479 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %1451, <4 x float> %1478) #9, !dbg !14
  %1480 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %1453, <4 x float> %1479) #9, !dbg !14
  %1481 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %1455, <4 x float> %1480) #9, !dbg !14
  %1482 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %1457, <4 x float> %1481) #9, !dbg !14
  %1483 = shufflevector <4 x float> %1472, <4 x float> %1482, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1484 = insertvalue [2 x <8 x float>] %1460, <8 x float> %1483, 1, !dbg !7
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1485 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local871, i32 0) #0, !dbg !14
  %1486 = bitcast <4 x i32> %1485 to <8 x half>, !dbg !14
  %1487 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local884, i32 0) #0, !dbg !14
  %1488 = bitcast <4 x i32> %1487 to <8 x half>, !dbg !14
  %1489 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local888, i32 0) #0, !dbg !14
  %1490 = bitcast <4 x i32> %1489 to <8 x half>, !dbg !14
  %1491 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local892, i32 0) #0, !dbg !14
  %1492 = bitcast <4 x i32> %1491 to <8 x half>, !dbg !14
  %1493 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local894, i32 0) #0, !dbg !14
  %1494 = bitcast <4 x i32> %1493 to <8 x half>, !dbg !14
  %1495 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local898, i32 0) #0, !dbg !14
  %1496 = bitcast <4 x i32> %1495 to <8 x half>, !dbg !14
  %1497 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local902, i32 0) #0, !dbg !14
  %1498 = bitcast <4 x i32> %1497 to <8 x half>, !dbg !14
  %1499 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local906, i32 0) #0, !dbg !14
  %1500 = bitcast <4 x i32> %1499 to <8 x half>, !dbg !14
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1501 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %1390, !dbg !18
  %1502 = getelementptr i8, ptr addrspace(1) %1501, i64 128, !dbg !18
  %1503 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local911), !dbg !18
  %1504 = zext i32 %1503 to i64, !dbg !18
  %1505 = inttoptr i64 %1504 to ptr addrspace(3), !dbg !18
  %address.local930 = getelementptr [2 x i8], ptr addrspace(1) %1502, i64 %address.local916, !dbg !18
  %1506 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1507 = ptrtoint ptr addrspace(1) %address.local930 to i64, !dbg !18
  %1508 = sub i64 %1507, %1506, !dbg !18
  %dma.buffer.offset1552 = trunc i64 %1508 to i32, !dbg !18
  %1509 = trunc i64 %1506 to i32, !dbg !18
  %1510 = insertelement <4 x i32> poison, i32 %1509, i64 0, !dbg !18
  %1511 = lshr i64 %1506, 32, !dbg !18
  %1512 = trunc i64 %1511 to i32, !dbg !18
  %1513 = insertelement <4 x i32> %1510, i32 %1512, i64 1, !dbg !18
  %1514 = insertelement <4 x i32> %1513, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1553 = insertelement <4 x i32> %1514, i32 131072, i64 3, !dbg !18
  %1515 = ptrtoint ptr addrspace(3) %1505 to i32, !dbg !18
  %dma.buffer.lds1554 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1515), !dbg !18
  %1516 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1554, i32 %dma.buffer.offset1552, <4 x i32> %dma.buffer.rsrc1553) #10, !dbg !18
  %1517 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local923), !dbg !18
  %1518 = zext i32 %1517 to i64, !dbg !18
  %1519 = inttoptr i64 %1518 to ptr addrspace(3), !dbg !18
  %address.local931 = getelementptr [2 x i8], ptr addrspace(1) %1502, i64 %address.local928, !dbg !18
  %1520 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1521 = ptrtoint ptr addrspace(1) %address.local931 to i64, !dbg !18
  %1522 = sub i64 %1521, %1520, !dbg !18
  %dma.buffer.offset1549 = trunc i64 %1522 to i32, !dbg !18
  %1523 = trunc i64 %1520 to i32, !dbg !18
  %1524 = insertelement <4 x i32> poison, i32 %1523, i64 0, !dbg !18
  %1525 = lshr i64 %1520, 32, !dbg !18
  %1526 = trunc i64 %1525 to i32, !dbg !18
  %1527 = insertelement <4 x i32> %1524, i32 %1526, i64 1, !dbg !18
  %1528 = insertelement <4 x i32> %1527, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1550 = insertelement <4 x i32> %1528, i32 131072, i64 3, !dbg !18
  %1529 = ptrtoint ptr addrspace(3) %1519 to i32, !dbg !18
  %dma.buffer.lds1551 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1529), !dbg !18
  %1530 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1551, i32 %dma.buffer.offset1549, <4 x i32> %dma.buffer.rsrc1550) #10, !dbg !18
  %1531 = shufflevector <8 x half> %1486, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1532 = extractvalue [2 x <8 x float>] %209, 0, !dbg !35
  %1533 = shufflevector <8 x float> %1532, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1534 = fmul <4 x float> %1533, %1067, !dbg !35
  %1535 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1531, <4 x float> %1534) #9, !dbg !14
  %1536 = shufflevector <8 x half> %1488, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1537 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %1536, <4 x float> %1535) #9, !dbg !14
  %1538 = shufflevector <8 x half> %1490, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1539 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %1538, <4 x float> %1537) #9, !dbg !14
  %1540 = shufflevector <8 x half> %1492, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1541 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %1540, <4 x float> %1539) #9, !dbg !14
  %1542 = shufflevector <8 x half> %1494, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1543 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %1542, <4 x float> %1541) #9, !dbg !14
  %1544 = shufflevector <8 x half> %1496, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1545 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %1544, <4 x float> %1543) #9, !dbg !14
  %1546 = shufflevector <8 x half> %1498, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1547 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %1546, <4 x float> %1545) #9, !dbg !14
  %1548 = shufflevector <8 x half> %1500, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1549 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %1548, <4 x float> %1547) #9, !dbg !14
  %1550 = shufflevector <8 x half> %1486, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1551 = shufflevector <8 x float> %1532, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1552 = fmul <4 x float> %1551, %1067, !dbg !35
  %1553 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1550, <4 x float> %1552) #9, !dbg !14
  %1554 = shufflevector <8 x half> %1488, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1555 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %1554, <4 x float> %1553) #9, !dbg !14
  %1556 = shufflevector <8 x half> %1490, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1557 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %1556, <4 x float> %1555) #9, !dbg !14
  %1558 = shufflevector <8 x half> %1492, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1559 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %1558, <4 x float> %1557) #9, !dbg !14
  %1560 = shufflevector <8 x half> %1494, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1561 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %1560, <4 x float> %1559) #9, !dbg !14
  %1562 = shufflevector <8 x half> %1496, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1563 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %1562, <4 x float> %1561) #9, !dbg !14
  %1564 = shufflevector <8 x half> %1498, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1565 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %1564, <4 x float> %1563) #9, !dbg !14
  %1566 = shufflevector <8 x half> %1500, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1567 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %1566, <4 x float> %1565) #9, !dbg !14
  %1568 = extractvalue [2 x <8 x float>] %209, 1, !dbg !35
  %1569 = shufflevector <8 x float> %1568, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1570 = fmul <4 x float> %1569, %1075, !dbg !35
  %1571 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1531, <4 x float> %1570) #9, !dbg !14
  %1572 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1536, <4 x float> %1571) #9, !dbg !14
  %1573 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1538, <4 x float> %1572) #9, !dbg !14
  %1574 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1540, <4 x float> %1573) #9, !dbg !14
  %1575 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %1542, <4 x float> %1574) #9, !dbg !14
  %1576 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %1544, <4 x float> %1575) #9, !dbg !14
  %1577 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %1546, <4 x float> %1576) #9, !dbg !14
  %1578 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %1548, <4 x float> %1577) #9, !dbg !14
  %1579 = shufflevector <8 x float> %1568, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1580 = fmul <4 x float> %1579, %1075, !dbg !35
  %1581 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1550, <4 x float> %1580) #9, !dbg !14
  %1582 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1554, <4 x float> %1581) #9, !dbg !14
  %1583 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1556, <4 x float> %1582) #9, !dbg !14
  %1584 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1558, <4 x float> %1583) #9, !dbg !14
  %1585 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %1560, <4 x float> %1584) #9, !dbg !14
  %1586 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %1562, <4 x float> %1585) #9, !dbg !14
  %1587 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %1564, <4 x float> %1586) #9, !dbg !14
  %1588 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %1566, <4 x float> %1587) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1589 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local871, i32 0) #0, !dbg !14
  %1590 = bitcast <4 x i32> %1589 to <8 x half>, !dbg !14
  %1591 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local884, i32 0) #0, !dbg !14
  %1592 = bitcast <4 x i32> %1591 to <8 x half>, !dbg !14
  %1593 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local888, i32 0) #0, !dbg !14
  %1594 = bitcast <4 x i32> %1593 to <8 x half>, !dbg !14
  %1595 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local892, i32 0) #0, !dbg !14
  %1596 = bitcast <4 x i32> %1595 to <8 x half>, !dbg !14
  %1597 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local894, i32 0) #0, !dbg !14
  %1598 = bitcast <4 x i32> %1597 to <8 x half>, !dbg !14
  %1599 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local898, i32 0) #0, !dbg !14
  %1600 = bitcast <4 x i32> %1599 to <8 x half>, !dbg !14
  %1601 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local902, i32 0) #0, !dbg !14
  %1602 = bitcast <4 x i32> %1601 to <8 x half>, !dbg !14
  %1603 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local906, i32 0) #0, !dbg !14
  %1604 = bitcast <4 x i32> %1603 to <8 x half>, !dbg !14
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1605 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %1390, !dbg !18
  %1606 = getelementptr i8, ptr addrspace(1) %1605, i64 192, !dbg !18
  %1607 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local911), !dbg !18
  %1608 = zext i32 %1607 to i64, !dbg !18
  %1609 = inttoptr i64 %1608 to ptr addrspace(3), !dbg !18
  %address.local932 = getelementptr [2 x i8], ptr addrspace(1) %1606, i64 %address.local916, !dbg !18
  %1610 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1611 = ptrtoint ptr addrspace(1) %address.local932 to i64, !dbg !18
  %1612 = sub i64 %1611, %1610, !dbg !18
  %dma.buffer.offset1546 = trunc i64 %1612 to i32, !dbg !18
  %1613 = trunc i64 %1610 to i32, !dbg !18
  %1614 = insertelement <4 x i32> poison, i32 %1613, i64 0, !dbg !18
  %1615 = lshr i64 %1610, 32, !dbg !18
  %1616 = trunc i64 %1615 to i32, !dbg !18
  %1617 = insertelement <4 x i32> %1614, i32 %1616, i64 1, !dbg !18
  %1618 = insertelement <4 x i32> %1617, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1547 = insertelement <4 x i32> %1618, i32 131072, i64 3, !dbg !18
  %1619 = ptrtoint ptr addrspace(3) %1609 to i32, !dbg !18
  %dma.buffer.lds1548 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1619), !dbg !18
  %1620 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1548, i32 %dma.buffer.offset1546, <4 x i32> %dma.buffer.rsrc1547) #10, !dbg !18
  %1621 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local923), !dbg !18
  %1622 = zext i32 %1621 to i64, !dbg !18
  %1623 = inttoptr i64 %1622 to ptr addrspace(3), !dbg !18
  %address.local933 = getelementptr [2 x i8], ptr addrspace(1) %1606, i64 %address.local928, !dbg !18
  %1624 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1625 = ptrtoint ptr addrspace(1) %address.local933 to i64, !dbg !18
  %1626 = sub i64 %1625, %1624, !dbg !18
  %dma.buffer.offset = trunc i64 %1626 to i32, !dbg !18
  %1627 = trunc i64 %1624 to i32, !dbg !18
  %1628 = insertelement <4 x i32> poison, i32 %1627, i64 0, !dbg !18
  %1629 = lshr i64 %1624, 32, !dbg !18
  %1630 = trunc i64 %1629 to i32, !dbg !18
  %1631 = insertelement <4 x i32> %1628, i32 %1630, i64 1, !dbg !18
  %1632 = insertelement <4 x i32> %1631, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc = insertelement <4 x i32> %1632, i32 131072, i64 3, !dbg !18
  %1633 = ptrtoint ptr addrspace(3) %1623 to i32, !dbg !18
  %dma.buffer.lds = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1633), !dbg !18
  %1634 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc) #10, !dbg !18
  %1635 = shufflevector <8 x half> %1590, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1636 = extractvalue [2 x <8 x float>] %210, 0, !dbg !35
  %1637 = shufflevector <8 x float> %1636, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1638 = fmul <4 x float> %1637, %1067, !dbg !35
  %1639 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1635, <4 x float> %1638) #9, !dbg !14
  %1640 = shufflevector <8 x half> %1592, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1641 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %1640, <4 x float> %1639) #9, !dbg !14
  %1642 = shufflevector <8 x half> %1594, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1643 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %1642, <4 x float> %1641) #9, !dbg !14
  %1644 = shufflevector <8 x half> %1596, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1645 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %1644, <4 x float> %1643) #9, !dbg !14
  %1646 = shufflevector <8 x half> %1598, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1647 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %1646, <4 x float> %1645) #9, !dbg !14
  %1648 = shufflevector <8 x half> %1600, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1649 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %1648, <4 x float> %1647) #9, !dbg !14
  %1650 = shufflevector <8 x half> %1602, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1651 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %1650, <4 x float> %1649) #9, !dbg !14
  %1652 = shufflevector <8 x half> %1604, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1653 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %1652, <4 x float> %1651) #9, !dbg !14
  %1654 = shufflevector <8 x half> %1590, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1655 = shufflevector <8 x float> %1636, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1656 = fmul <4 x float> %1655, %1067, !dbg !35
  %1657 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1654, <4 x float> %1656) #9, !dbg !14
  %1658 = shufflevector <8 x half> %1592, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1659 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %1658, <4 x float> %1657) #9, !dbg !14
  %1660 = shufflevector <8 x half> %1594, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1661 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %1660, <4 x float> %1659) #9, !dbg !14
  %1662 = shufflevector <8 x half> %1596, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1663 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %1662, <4 x float> %1661) #9, !dbg !14
  %1664 = shufflevector <8 x half> %1598, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1665 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %1664, <4 x float> %1663) #9, !dbg !14
  %1666 = shufflevector <8 x half> %1600, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1667 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %1666, <4 x float> %1665) #9, !dbg !14
  %1668 = shufflevector <8 x half> %1602, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1669 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %1668, <4 x float> %1667) #9, !dbg !14
  %1670 = shufflevector <8 x half> %1604, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1671 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %1670, <4 x float> %1669) #9, !dbg !14
  %1672 = extractvalue [2 x <8 x float>] %210, 1, !dbg !35
  %1673 = shufflevector <8 x float> %1672, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1674 = fmul <4 x float> %1673, %1075, !dbg !35
  %1675 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1635, <4 x float> %1674) #9, !dbg !14
  %1676 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1640, <4 x float> %1675) #9, !dbg !14
  %1677 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1642, <4 x float> %1676) #9, !dbg !14
  %1678 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1644, <4 x float> %1677) #9, !dbg !14
  %1679 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %1646, <4 x float> %1678) #9, !dbg !14
  %1680 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %1648, <4 x float> %1679) #9, !dbg !14
  %1681 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %1650, <4 x float> %1680) #9, !dbg !14
  %1682 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %1652, <4 x float> %1681) #9, !dbg !14
  %1683 = shufflevector <8 x float> %1672, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1684 = fmul <4 x float> %1683, %1075, !dbg !35
  %1685 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1654, <4 x float> %1684) #9, !dbg !14
  %1686 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1658, <4 x float> %1685) #9, !dbg !14
  %1687 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1660, <4 x float> %1686) #9, !dbg !14
  %1688 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1662, <4 x float> %1687) #9, !dbg !14
  %1689 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %1664, <4 x float> %1688) #9, !dbg !14
  %1690 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %1666, <4 x float> %1689) #9, !dbg !14
  %1691 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %1668, <4 x float> %1690) #9, !dbg !14
  %1692 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %1670, <4 x float> %1691) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1693 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local871, i32 0) #0, !dbg !14
  %1694 = bitcast <4 x i32> %1693 to <8 x half>, !dbg !14
  %1695 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local884, i32 0) #0, !dbg !14
  %1696 = bitcast <4 x i32> %1695 to <8 x half>, !dbg !14
  %1697 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local888, i32 0) #0, !dbg !14
  %1698 = bitcast <4 x i32> %1697 to <8 x half>, !dbg !14
  %1699 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local892, i32 0) #0, !dbg !14
  %1700 = bitcast <4 x i32> %1699 to <8 x half>, !dbg !14
  %1701 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local894, i32 0) #0, !dbg !14
  %1702 = bitcast <4 x i32> %1701 to <8 x half>, !dbg !14
  %1703 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local898, i32 0) #0, !dbg !14
  %1704 = bitcast <4 x i32> %1703 to <8 x half>, !dbg !14
  %1705 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local902, i32 0) #0, !dbg !14
  %1706 = bitcast <4 x i32> %1705 to <8 x half>, !dbg !14
  %1707 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local906, i32 0) #0, !dbg !14
  %1708 = bitcast <4 x i32> %1707 to <8 x half>, !dbg !14
  %1709 = shufflevector <8 x half> %1694, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1710 = extractvalue [2 x <8 x float>] %211, 0, !dbg !35
  %1711 = shufflevector <8 x float> %1710, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1712 = fmul <4 x float> %1711, %1067, !dbg !35
  %1713 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1709, <4 x float> %1712) #9, !dbg !14
  %1714 = shufflevector <8 x half> %1696, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1715 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %1714, <4 x float> %1713) #9, !dbg !14
  %1716 = shufflevector <8 x half> %1698, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1717 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %1716, <4 x float> %1715) #9, !dbg !14
  %1718 = shufflevector <8 x half> %1700, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1719 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %1718, <4 x float> %1717) #9, !dbg !14
  %1720 = shufflevector <8 x half> %1702, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1721 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %1720, <4 x float> %1719) #9, !dbg !14
  %1722 = shufflevector <8 x half> %1704, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1723 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %1722, <4 x float> %1721) #9, !dbg !14
  %1724 = shufflevector <8 x half> %1706, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1725 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %1724, <4 x float> %1723) #9, !dbg !14
  %1726 = shufflevector <8 x half> %1708, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1727 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %1726, <4 x float> %1725) #9, !dbg !14
  %1728 = shufflevector <8 x half> %1694, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1729 = shufflevector <8 x float> %1710, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1730 = fmul <4 x float> %1729, %1067, !dbg !35
  %1731 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1728, <4 x float> %1730) #9, !dbg !14
  %1732 = shufflevector <8 x half> %1696, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1733 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %1732, <4 x float> %1731) #9, !dbg !14
  %1734 = shufflevector <8 x half> %1698, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1735 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %1734, <4 x float> %1733) #9, !dbg !14
  %1736 = shufflevector <8 x half> %1700, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1737 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %1736, <4 x float> %1735) #9, !dbg !14
  %1738 = shufflevector <8 x half> %1702, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1739 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %1738, <4 x float> %1737) #9, !dbg !14
  %1740 = shufflevector <8 x half> %1704, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1741 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %1740, <4 x float> %1739) #9, !dbg !14
  %1742 = shufflevector <8 x half> %1706, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1743 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %1742, <4 x float> %1741) #9, !dbg !14
  %1744 = shufflevector <8 x half> %1708, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1745 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %1744, <4 x float> %1743) #9, !dbg !14
  %1746 = extractvalue [2 x <8 x float>] %211, 1, !dbg !35
  %1747 = shufflevector <8 x float> %1746, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1748 = fmul <4 x float> %1747, %1075, !dbg !35
  %1749 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1709, <4 x float> %1748) #9, !dbg !14
  %1750 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1714, <4 x float> %1749) #9, !dbg !14
  %1751 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1716, <4 x float> %1750) #9, !dbg !14
  %1752 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1718, <4 x float> %1751) #9, !dbg !14
  %1753 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %1720, <4 x float> %1752) #9, !dbg !14
  %1754 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %1722, <4 x float> %1753) #9, !dbg !14
  %1755 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %1724, <4 x float> %1754) #9, !dbg !14
  %1756 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %1726, <4 x float> %1755) #9, !dbg !14
  %1757 = shufflevector <8 x float> %1746, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1758 = fmul <4 x float> %1757, %1075, !dbg !35
  %1759 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1728, <4 x float> %1758) #9, !dbg !14
  %1760 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1732, <4 x float> %1759) #9, !dbg !14
  %1761 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1734, <4 x float> %1760) #9, !dbg !14
  %1762 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1736, <4 x float> %1761) #9, !dbg !14
  %1763 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %1738, <4 x float> %1762) #9, !dbg !14
  %1764 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %1740, <4 x float> %1763) #9, !dbg !14
  %1765 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %1742, <4 x float> %1764) #9, !dbg !14
  %1766 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %1744, <4 x float> %1765) #9, !dbg !14
  %1767 = insertvalue [2 x <1 x float>] poison, <1 x float> %1001, 0, !dbg !7
  %1768 = insertvalue [2 x <1 x float>] %1767, <1 x float> %1044, 1, !dbg !7
  %1769 = shufflevector <4 x float> %1549, <4 x float> %1567, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1770 = insertvalue [2 x <8 x float>] poison, <8 x float> %1769, 0, !dbg !7
  %1771 = shufflevector <4 x float> %1578, <4 x float> %1588, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1772 = insertvalue [2 x <8 x float>] %1770, <8 x float> %1771, 1, !dbg !7
  %1773 = shufflevector <4 x float> %1653, <4 x float> %1671, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1774 = insertvalue [2 x <8 x float>] poison, <8 x float> %1773, 0, !dbg !7
  %1775 = shufflevector <4 x float> %1682, <4 x float> %1692, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1776 = insertvalue [2 x <8 x float>] %1774, <8 x float> %1775, 1, !dbg !7
  %1777 = shufflevector <4 x float> %1727, <4 x float> %1745, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1778 = insertvalue [2 x <8 x float>] poison, <8 x float> %1777, 0, !dbg !7
  %1779 = shufflevector <4 x float> %1756, <4 x float> %1766, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1780 = insertvalue [2 x <8 x float>] %1778, <8 x float> %1779, 1, !dbg !7
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %1781 = add nuw nsw i64 %205, 1, !dbg !7
  br label %204, !dbg !7

1782:                                             ; preds = %204
  %1783 = and i32 %31, 16, !dbg !7
  %.not = icmp eq i32 %1783, 0, !dbg !7
  %1784 = and i32 %31, 32, !dbg !7
  %.not36 = icmp eq i32 %1784, 0, !dbg !7
  %1785 = extractvalue [2 x <8 x float>] %211, 0, !dbg !7
  %1786 = extractelement <8 x float> %1785, i64 0, !dbg !7
  %1787 = extractelement <8 x float> %1785, i64 1, !dbg !7
  %1788 = extractelement <8 x float> %1785, i64 2, !dbg !7
  %1789 = extractelement <8 x float> %1785, i64 3, !dbg !7
  %1790 = select i1 %.not, float %1787, float %1786, !dbg !7
  %1791 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !7
  %1792 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1791), !dbg !7
  %1793 = add nuw nsw i32 %1792, 64, !dbg !7
  %1794 = xor i32 %1792, 16, !dbg !7
  %1795 = and i32 %1793, 192, !dbg !7
  %1796 = icmp samesign ult i32 %1794, %1795, !dbg !7
  %1797 = select i1 %1796, i32 %1794, i32 %1792, !dbg !7
  %1798 = shl nuw nsw i32 %1797, 2, !dbg !7
  %1799 = bitcast float %1790 to i32, !dbg !7
  %1800 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1798, i32 %1799), !dbg !7
  %1801 = bitcast i32 %1800 to float, !dbg !7
  %1802 = select i1 %.not, float %1786, float %1801, !dbg !7
  %1803 = select i1 %.not, float %1801, float %1787, !dbg !7
  %1804 = select i1 %.not, float %1789, float %1788, !dbg !7
  %1805 = bitcast float %1804 to i32, !dbg !7
  %1806 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1798, i32 %1805), !dbg !7
  %1807 = bitcast i32 %1806 to float, !dbg !7
  %1808 = select i1 %.not, float %1788, float %1807, !dbg !7
  %1809 = select i1 %.not, float %1807, float %1789, !dbg !7
  %1810 = select i1 %.not36, float %1808, float %1802, !dbg !7
  %1811 = xor i32 %1792, 32, !dbg !7
  %1812 = icmp samesign ult i32 %1811, %1795, !dbg !7
  %1813 = select i1 %1812, i32 %1811, i32 %1792, !dbg !7
  %1814 = shl nuw nsw i32 %1813, 2, !dbg !7
  %1815 = bitcast float %1810 to i32, !dbg !7
  %1816 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1814, i32 %1815), !dbg !7
  %1817 = bitcast i32 %1816 to float, !dbg !7
  %1818 = select i1 %.not36, float %1802, float %1817, !dbg !7
  %1819 = select i1 %.not36, float %1817, float %1808, !dbg !7
  %1820 = select i1 %.not36, float %1809, float %1803, !dbg !7
  %1821 = bitcast float %1820 to i32, !dbg !7
  %1822 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1814, i32 %1821), !dbg !7
  %1823 = bitcast i32 %1822 to float, !dbg !7
  %1824 = select i1 %.not36, float %1803, float %1823, !dbg !7
  %1825 = select i1 %.not36, float %1823, float %1809, !dbg !7
  %1826 = extractvalue [2 x <8 x float>] %210, 0, !dbg !7
  %1827 = extractelement <8 x float> %1826, i64 0, !dbg !7
  %1828 = extractelement <8 x float> %1826, i64 1, !dbg !7
  %1829 = extractelement <8 x float> %1826, i64 2, !dbg !7
  %1830 = extractelement <8 x float> %1826, i64 3, !dbg !7
  %1831 = select i1 %.not, float %1828, float %1827, !dbg !7
  %1832 = bitcast float %1831 to i32, !dbg !7
  %1833 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1798, i32 %1832), !dbg !7
  %1834 = bitcast i32 %1833 to float, !dbg !7
  %1835 = select i1 %.not, float %1827, float %1834, !dbg !7
  %1836 = select i1 %.not, float %1834, float %1828, !dbg !7
  %1837 = select i1 %.not, float %1830, float %1829, !dbg !7
  %1838 = bitcast float %1837 to i32, !dbg !7
  %1839 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1798, i32 %1838), !dbg !7
  %1840 = bitcast i32 %1839 to float, !dbg !7
  %1841 = select i1 %.not, float %1829, float %1840, !dbg !7
  %1842 = select i1 %.not, float %1840, float %1830, !dbg !7
  %1843 = select i1 %.not36, float %1841, float %1835, !dbg !7
  %1844 = bitcast float %1843 to i32, !dbg !7
  %1845 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1814, i32 %1844), !dbg !7
  %1846 = bitcast i32 %1845 to float, !dbg !7
  %1847 = select i1 %.not36, float %1835, float %1846, !dbg !7
  %1848 = select i1 %.not36, float %1846, float %1841, !dbg !7
  %1849 = select i1 %.not36, float %1842, float %1836, !dbg !7
  %1850 = bitcast float %1849 to i32, !dbg !7
  %1851 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1814, i32 %1850), !dbg !7
  %1852 = bitcast i32 %1851 to float, !dbg !7
  %1853 = select i1 %.not36, float %1836, float %1852, !dbg !7
  %1854 = select i1 %.not36, float %1852, float %1842, !dbg !7
  %1855 = extractvalue [2 x <8 x float>] %209, 0, !dbg !7
  %1856 = extractelement <8 x float> %1855, i64 0, !dbg !7
  %1857 = extractelement <8 x float> %1855, i64 1, !dbg !7
  %1858 = extractelement <8 x float> %1855, i64 2, !dbg !7
  %1859 = extractelement <8 x float> %1855, i64 3, !dbg !7
  %1860 = select i1 %.not, float %1857, float %1856, !dbg !7
  %1861 = bitcast float %1860 to i32, !dbg !7
  %1862 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1798, i32 %1861), !dbg !7
  %1863 = bitcast i32 %1862 to float, !dbg !7
  %1864 = select i1 %.not, float %1856, float %1863, !dbg !7
  %1865 = select i1 %.not, float %1863, float %1857, !dbg !7
  %1866 = select i1 %.not, float %1859, float %1858, !dbg !7
  %1867 = bitcast float %1866 to i32, !dbg !7
  %1868 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1798, i32 %1867), !dbg !7
  %1869 = bitcast i32 %1868 to float, !dbg !7
  %1870 = select i1 %.not, float %1858, float %1869, !dbg !7
  %1871 = select i1 %.not, float %1869, float %1859, !dbg !7
  %1872 = select i1 %.not36, float %1870, float %1864, !dbg !7
  %1873 = bitcast float %1872 to i32, !dbg !7
  %1874 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1814, i32 %1873), !dbg !7
  %1875 = bitcast i32 %1874 to float, !dbg !7
  %1876 = select i1 %.not36, float %1864, float %1875, !dbg !7
  %1877 = select i1 %.not36, float %1875, float %1870, !dbg !7
  %1878 = select i1 %.not36, float %1871, float %1865, !dbg !7
  %1879 = bitcast float %1878 to i32, !dbg !7
  %1880 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1814, i32 %1879), !dbg !7
  %1881 = bitcast i32 %1880 to float, !dbg !7
  %1882 = select i1 %.not36, float %1865, float %1881, !dbg !7
  %1883 = select i1 %.not36, float %1881, float %1871, !dbg !7
  %1884 = extractvalue [2 x <8 x float>] %206, 0, !dbg !7
  %1885 = extractelement <8 x float> %1884, i64 0, !dbg !7
  %1886 = extractelement <8 x float> %1884, i64 1, !dbg !7
  %1887 = extractelement <8 x float> %1884, i64 2, !dbg !7
  %1888 = extractelement <8 x float> %1884, i64 3, !dbg !7
  %1889 = select i1 %.not, float %1886, float %1885, !dbg !7
  %1890 = bitcast float %1889 to i32, !dbg !7
  %1891 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1798, i32 %1890), !dbg !7
  %1892 = bitcast i32 %1891 to float, !dbg !7
  %1893 = select i1 %.not, float %1885, float %1892, !dbg !7
  %1894 = select i1 %.not, float %1892, float %1886, !dbg !7
  %1895 = select i1 %.not, float %1888, float %1887, !dbg !7
  %1896 = bitcast float %1895 to i32, !dbg !7
  %1897 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1798, i32 %1896), !dbg !7
  %1898 = bitcast i32 %1897 to float, !dbg !7
  %1899 = select i1 %.not, float %1887, float %1898, !dbg !7
  %1900 = select i1 %.not, float %1898, float %1888, !dbg !7
  %1901 = select i1 %.not36, float %1899, float %1893, !dbg !7
  %1902 = bitcast float %1901 to i32, !dbg !7
  %1903 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1814, i32 %1902), !dbg !7
  %1904 = select i1 %.not36, float %1900, float %1894, !dbg !7
  %1905 = bitcast float %1904 to i32, !dbg !7
  %1906 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1814, i32 %1905), !dbg !7
  %1907 = extractvalue [2 x <1 x float>] %207, 0, !dbg !19
  %1908 = shufflevector <1 x float> %1907, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %1909 = extractelement <8 x float> %1785, i64 4, !dbg !7
  %1910 = extractelement <8 x float> %1785, i64 5, !dbg !7
  %1911 = extractelement <8 x float> %1785, i64 6, !dbg !7
  %1912 = extractelement <8 x float> %1785, i64 7, !dbg !7
  %1913 = select i1 %.not, float %1910, float %1909, !dbg !7
  %1914 = bitcast float %1913 to i32, !dbg !7
  %1915 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1798, i32 %1914), !dbg !7
  %1916 = bitcast i32 %1915 to float, !dbg !7
  %1917 = select i1 %.not, float %1909, float %1916, !dbg !7
  %1918 = select i1 %.not, float %1916, float %1910, !dbg !7
  %1919 = select i1 %.not, float %1912, float %1911, !dbg !7
  %1920 = bitcast float %1919 to i32, !dbg !7
  %1921 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1798, i32 %1920), !dbg !7
  %1922 = bitcast i32 %1921 to float, !dbg !7
  %1923 = select i1 %.not, float %1911, float %1922, !dbg !7
  %1924 = select i1 %.not, float %1922, float %1912, !dbg !7
  %1925 = select i1 %.not36, float %1923, float %1917, !dbg !7
  %1926 = bitcast float %1925 to i32, !dbg !7
  %1927 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1814, i32 %1926), !dbg !7
  %1928 = bitcast i32 %1927 to float, !dbg !7
  %1929 = select i1 %.not36, float %1917, float %1928, !dbg !7
  %1930 = select i1 %.not36, float %1928, float %1923, !dbg !7
  %1931 = select i1 %.not36, float %1924, float %1918, !dbg !7
  %1932 = bitcast float %1931 to i32, !dbg !7
  %1933 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1814, i32 %1932), !dbg !7
  %1934 = bitcast i32 %1933 to float, !dbg !7
  %1935 = select i1 %.not36, float %1918, float %1934, !dbg !7
  %1936 = select i1 %.not36, float %1934, float %1924, !dbg !7
  %1937 = extractelement <8 x float> %1826, i64 4, !dbg !7
  %1938 = extractelement <8 x float> %1826, i64 5, !dbg !7
  %1939 = extractelement <8 x float> %1826, i64 6, !dbg !7
  %1940 = extractelement <8 x float> %1826, i64 7, !dbg !7
  %1941 = select i1 %.not, float %1938, float %1937, !dbg !7
  %1942 = bitcast float %1941 to i32, !dbg !7
  %1943 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1798, i32 %1942), !dbg !7
  %1944 = bitcast i32 %1943 to float, !dbg !7
  %1945 = select i1 %.not, float %1937, float %1944, !dbg !7
  %1946 = select i1 %.not, float %1944, float %1938, !dbg !7
  %1947 = select i1 %.not, float %1940, float %1939, !dbg !7
  %1948 = bitcast float %1947 to i32, !dbg !7
  %1949 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1798, i32 %1948), !dbg !7
  %1950 = bitcast i32 %1949 to float, !dbg !7
  %1951 = select i1 %.not, float %1939, float %1950, !dbg !7
  %1952 = select i1 %.not, float %1950, float %1940, !dbg !7
  %1953 = select i1 %.not36, float %1951, float %1945, !dbg !7
  %1954 = bitcast float %1953 to i32, !dbg !7
  %1955 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1814, i32 %1954), !dbg !7
  %1956 = bitcast i32 %1955 to float, !dbg !7
  %1957 = select i1 %.not36, float %1945, float %1956, !dbg !7
  %1958 = select i1 %.not36, float %1956, float %1951, !dbg !7
  %1959 = select i1 %.not36, float %1952, float %1946, !dbg !7
  %1960 = bitcast float %1959 to i32, !dbg !7
  %1961 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1814, i32 %1960), !dbg !7
  %1962 = bitcast i32 %1961 to float, !dbg !7
  %1963 = select i1 %.not36, float %1946, float %1962, !dbg !7
  %1964 = select i1 %.not36, float %1962, float %1952, !dbg !7
  %1965 = extractelement <8 x float> %1855, i64 4, !dbg !7
  %1966 = extractelement <8 x float> %1855, i64 5, !dbg !7
  %1967 = extractelement <8 x float> %1855, i64 6, !dbg !7
  %1968 = extractelement <8 x float> %1855, i64 7, !dbg !7
  %1969 = select i1 %.not, float %1966, float %1965, !dbg !7
  %1970 = bitcast float %1969 to i32, !dbg !7
  %1971 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1798, i32 %1970), !dbg !7
  %1972 = bitcast i32 %1971 to float, !dbg !7
  %1973 = select i1 %.not, float %1965, float %1972, !dbg !7
  %1974 = select i1 %.not, float %1972, float %1966, !dbg !7
  %1975 = select i1 %.not, float %1968, float %1967, !dbg !7
  %1976 = bitcast float %1975 to i32, !dbg !7
  %1977 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1798, i32 %1976), !dbg !7
  %1978 = bitcast i32 %1977 to float, !dbg !7
  %1979 = select i1 %.not, float %1967, float %1978, !dbg !7
  %1980 = select i1 %.not, float %1978, float %1968, !dbg !7
  %1981 = select i1 %.not36, float %1979, float %1973, !dbg !7
  %1982 = bitcast float %1981 to i32, !dbg !7
  %1983 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1814, i32 %1982), !dbg !7
  %1984 = bitcast i32 %1983 to float, !dbg !7
  %1985 = select i1 %.not36, float %1973, float %1984, !dbg !7
  %1986 = select i1 %.not36, float %1984, float %1979, !dbg !7
  %1987 = select i1 %.not36, float %1980, float %1974, !dbg !7
  %1988 = bitcast float %1987 to i32, !dbg !7
  %1989 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1814, i32 %1988), !dbg !7
  %1990 = bitcast i32 %1989 to float, !dbg !7
  %1991 = select i1 %.not36, float %1974, float %1990, !dbg !7
  %1992 = select i1 %.not36, float %1990, float %1980, !dbg !7
  %1993 = extractelement <8 x float> %1884, i64 4, !dbg !7
  %1994 = extractelement <8 x float> %1884, i64 5, !dbg !7
  %1995 = extractelement <8 x float> %1884, i64 6, !dbg !7
  %1996 = extractelement <8 x float> %1884, i64 7, !dbg !7
  %1997 = select i1 %.not, float %1994, float %1993, !dbg !7
  %1998 = bitcast float %1997 to i32, !dbg !7
  %1999 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1798, i32 %1998), !dbg !7
  %2000 = bitcast i32 %1999 to float, !dbg !7
  %2001 = select i1 %.not, float %1993, float %2000, !dbg !7
  %2002 = select i1 %.not, float %2000, float %1994, !dbg !7
  %2003 = select i1 %.not, float %1996, float %1995, !dbg !7
  %2004 = bitcast float %2003 to i32, !dbg !7
  %2005 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1798, i32 %2004), !dbg !7
  %2006 = bitcast i32 %2005 to float, !dbg !7
  %2007 = select i1 %.not, float %1995, float %2006, !dbg !7
  %2008 = select i1 %.not, float %2006, float %1996, !dbg !7
  %2009 = select i1 %.not36, float %2007, float %2001, !dbg !7
  %2010 = bitcast float %2009 to i32, !dbg !7
  %2011 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1814, i32 %2010), !dbg !7
  %2012 = select i1 %.not36, float %2008, float %2002, !dbg !7
  %2013 = bitcast float %2012 to i32, !dbg !7
  %2014 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1814, i32 %2013), !dbg !7
  %2015 = extractvalue [2 x <8 x float>] %211, 1, !dbg !7
  %2016 = extractelement <8 x float> %2015, i64 0, !dbg !7
  %2017 = extractelement <8 x float> %2015, i64 1, !dbg !7
  %2018 = extractelement <8 x float> %2015, i64 2, !dbg !7
  %2019 = extractelement <8 x float> %2015, i64 3, !dbg !7
  %2020 = select i1 %.not, float %2017, float %2016, !dbg !7
  %2021 = bitcast float %2020 to i32, !dbg !7
  %2022 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1798, i32 %2021), !dbg !7
  %2023 = bitcast i32 %2022 to float, !dbg !7
  %2024 = select i1 %.not, float %2016, float %2023, !dbg !7
  %2025 = select i1 %.not, float %2023, float %2017, !dbg !7
  %2026 = select i1 %.not, float %2019, float %2018, !dbg !7
  %2027 = bitcast float %2026 to i32, !dbg !7
  %2028 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1798, i32 %2027), !dbg !7
  %2029 = bitcast i32 %2028 to float, !dbg !7
  %2030 = select i1 %.not, float %2018, float %2029, !dbg !7
  %2031 = select i1 %.not, float %2029, float %2019, !dbg !7
  %2032 = select i1 %.not36, float %2030, float %2024, !dbg !7
  %2033 = bitcast float %2032 to i32, !dbg !7
  %2034 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1814, i32 %2033), !dbg !7
  %2035 = bitcast i32 %2034 to float, !dbg !7
  %2036 = select i1 %.not36, float %2024, float %2035, !dbg !7
  %2037 = select i1 %.not36, float %2035, float %2030, !dbg !7
  %2038 = select i1 %.not36, float %2031, float %2025, !dbg !7
  %2039 = bitcast float %2038 to i32, !dbg !7
  %2040 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1814, i32 %2039), !dbg !7
  %2041 = bitcast i32 %2040 to float, !dbg !7
  %2042 = select i1 %.not36, float %2025, float %2041, !dbg !7
  %2043 = select i1 %.not36, float %2041, float %2031, !dbg !7
  %2044 = extractvalue [2 x <8 x float>] %210, 1, !dbg !7
  %2045 = extractelement <8 x float> %2044, i64 0, !dbg !7
  %2046 = extractelement <8 x float> %2044, i64 1, !dbg !7
  %2047 = extractelement <8 x float> %2044, i64 2, !dbg !7
  %2048 = extractelement <8 x float> %2044, i64 3, !dbg !7
  %2049 = select i1 %.not, float %2046, float %2045, !dbg !7
  %2050 = bitcast float %2049 to i32, !dbg !7
  %2051 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1798, i32 %2050), !dbg !7
  %2052 = bitcast i32 %2051 to float, !dbg !7
  %2053 = select i1 %.not, float %2045, float %2052, !dbg !7
  %2054 = select i1 %.not, float %2052, float %2046, !dbg !7
  %2055 = select i1 %.not, float %2048, float %2047, !dbg !7
  %2056 = bitcast float %2055 to i32, !dbg !7
  %2057 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1798, i32 %2056), !dbg !7
  %2058 = bitcast i32 %2057 to float, !dbg !7
  %2059 = select i1 %.not, float %2047, float %2058, !dbg !7
  %2060 = select i1 %.not, float %2058, float %2048, !dbg !7
  %2061 = select i1 %.not36, float %2059, float %2053, !dbg !7
  %2062 = bitcast float %2061 to i32, !dbg !7
  %2063 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1814, i32 %2062), !dbg !7
  %2064 = bitcast i32 %2063 to float, !dbg !7
  %2065 = select i1 %.not36, float %2053, float %2064, !dbg !7
  %2066 = select i1 %.not36, float %2064, float %2059, !dbg !7
  %2067 = select i1 %.not36, float %2060, float %2054, !dbg !7
  %2068 = bitcast float %2067 to i32, !dbg !7
  %2069 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1814, i32 %2068), !dbg !7
  %2070 = bitcast i32 %2069 to float, !dbg !7
  %2071 = select i1 %.not36, float %2054, float %2070, !dbg !7
  %2072 = select i1 %.not36, float %2070, float %2060, !dbg !7
  %2073 = extractvalue [2 x <8 x float>] %209, 1, !dbg !7
  %2074 = extractelement <8 x float> %2073, i64 0, !dbg !7
  %2075 = extractelement <8 x float> %2073, i64 1, !dbg !7
  %2076 = extractelement <8 x float> %2073, i64 2, !dbg !7
  %2077 = extractelement <8 x float> %2073, i64 3, !dbg !7
  %2078 = select i1 %.not, float %2075, float %2074, !dbg !7
  %2079 = bitcast float %2078 to i32, !dbg !7
  %2080 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1798, i32 %2079), !dbg !7
  %2081 = bitcast i32 %2080 to float, !dbg !7
  %2082 = select i1 %.not, float %2074, float %2081, !dbg !7
  %2083 = select i1 %.not, float %2081, float %2075, !dbg !7
  %2084 = select i1 %.not, float %2077, float %2076, !dbg !7
  %2085 = bitcast float %2084 to i32, !dbg !7
  %2086 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1798, i32 %2085), !dbg !7
  %2087 = bitcast i32 %2086 to float, !dbg !7
  %2088 = select i1 %.not, float %2076, float %2087, !dbg !7
  %2089 = select i1 %.not, float %2087, float %2077, !dbg !7
  %2090 = select i1 %.not36, float %2088, float %2082, !dbg !7
  %2091 = bitcast float %2090 to i32, !dbg !7
  %2092 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1814, i32 %2091), !dbg !7
  %2093 = bitcast i32 %2092 to float, !dbg !7
  %2094 = select i1 %.not36, float %2082, float %2093, !dbg !7
  %2095 = select i1 %.not36, float %2093, float %2088, !dbg !7
  %2096 = select i1 %.not36, float %2089, float %2083, !dbg !7
  %2097 = bitcast float %2096 to i32, !dbg !7
  %2098 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1814, i32 %2097), !dbg !7
  %2099 = bitcast i32 %2098 to float, !dbg !7
  %2100 = select i1 %.not36, float %2083, float %2099, !dbg !7
  %2101 = select i1 %.not36, float %2099, float %2089, !dbg !7
  %2102 = extractvalue [2 x <8 x float>] %206, 1, !dbg !7
  %2103 = extractelement <8 x float> %2102, i64 0, !dbg !7
  %2104 = extractelement <8 x float> %2102, i64 1, !dbg !7
  %2105 = extractelement <8 x float> %2102, i64 2, !dbg !7
  %2106 = extractelement <8 x float> %2102, i64 3, !dbg !7
  %2107 = select i1 %.not, float %2104, float %2103, !dbg !7
  %2108 = bitcast float %2107 to i32, !dbg !7
  %2109 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1798, i32 %2108), !dbg !7
  %2110 = bitcast i32 %2109 to float, !dbg !7
  %2111 = select i1 %.not, float %2103, float %2110, !dbg !7
  %2112 = select i1 %.not, float %2110, float %2104, !dbg !7
  %2113 = select i1 %.not, float %2106, float %2105, !dbg !7
  %2114 = bitcast float %2113 to i32, !dbg !7
  %2115 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1798, i32 %2114), !dbg !7
  %2116 = bitcast i32 %2115 to float, !dbg !7
  %2117 = select i1 %.not, float %2105, float %2116, !dbg !7
  %2118 = select i1 %.not, float %2116, float %2106, !dbg !7
  %2119 = select i1 %.not36, float %2117, float %2111, !dbg !7
  %2120 = bitcast float %2119 to i32, !dbg !7
  %2121 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1814, i32 %2120), !dbg !7
  %2122 = select i1 %.not36, float %2118, float %2112, !dbg !7
  %2123 = bitcast float %2122 to i32, !dbg !7
  %2124 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1814, i32 %2123), !dbg !7
  %2125 = extractvalue [2 x <1 x float>] %207, 1, !dbg !19
  %2126 = shufflevector <1 x float> %2125, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %2127 = extractelement <8 x float> %2015, i64 4, !dbg !7
  %2128 = extractelement <8 x float> %2015, i64 5, !dbg !7
  %2129 = extractelement <8 x float> %2015, i64 6, !dbg !7
  %2130 = extractelement <8 x float> %2015, i64 7, !dbg !7
  %2131 = select i1 %.not, float %2128, float %2127, !dbg !7
  %2132 = bitcast float %2131 to i32, !dbg !7
  %2133 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1798, i32 %2132), !dbg !7
  %2134 = bitcast i32 %2133 to float, !dbg !7
  %2135 = select i1 %.not, float %2127, float %2134, !dbg !7
  %2136 = select i1 %.not, float %2134, float %2128, !dbg !7
  %2137 = select i1 %.not, float %2130, float %2129, !dbg !7
  %2138 = bitcast float %2137 to i32, !dbg !7
  %2139 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1798, i32 %2138), !dbg !7
  %2140 = bitcast i32 %2139 to float, !dbg !7
  %2141 = select i1 %.not, float %2129, float %2140, !dbg !7
  %2142 = select i1 %.not, float %2140, float %2130, !dbg !7
  %2143 = select i1 %.not36, float %2141, float %2135, !dbg !7
  %2144 = bitcast float %2143 to i32, !dbg !7
  %2145 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1814, i32 %2144), !dbg !7
  %2146 = bitcast i32 %2145 to float, !dbg !7
  %2147 = select i1 %.not36, float %2135, float %2146, !dbg !7
  %2148 = select i1 %.not36, float %2146, float %2141, !dbg !7
  %2149 = select i1 %.not36, float %2142, float %2136, !dbg !7
  %2150 = bitcast float %2149 to i32, !dbg !7
  %2151 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1814, i32 %2150), !dbg !7
  %2152 = bitcast i32 %2151 to float, !dbg !7
  %2153 = select i1 %.not36, float %2136, float %2152, !dbg !7
  %2154 = select i1 %.not36, float %2152, float %2142, !dbg !7
  %2155 = extractelement <8 x float> %2044, i64 4, !dbg !7
  %2156 = extractelement <8 x float> %2044, i64 5, !dbg !7
  %2157 = extractelement <8 x float> %2044, i64 6, !dbg !7
  %2158 = extractelement <8 x float> %2044, i64 7, !dbg !7
  %2159 = select i1 %.not, float %2156, float %2155, !dbg !7
  %2160 = bitcast float %2159 to i32, !dbg !7
  %2161 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1798, i32 %2160), !dbg !7
  %2162 = bitcast i32 %2161 to float, !dbg !7
  %2163 = select i1 %.not, float %2155, float %2162, !dbg !7
  %2164 = select i1 %.not, float %2162, float %2156, !dbg !7
  %2165 = select i1 %.not, float %2158, float %2157, !dbg !7
  %2166 = bitcast float %2165 to i32, !dbg !7
  %2167 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1798, i32 %2166), !dbg !7
  %2168 = bitcast i32 %2167 to float, !dbg !7
  %2169 = select i1 %.not, float %2157, float %2168, !dbg !7
  %2170 = select i1 %.not, float %2168, float %2158, !dbg !7
  %2171 = select i1 %.not36, float %2169, float %2163, !dbg !7
  %2172 = bitcast float %2171 to i32, !dbg !7
  %2173 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1814, i32 %2172), !dbg !7
  %2174 = bitcast i32 %2173 to float, !dbg !7
  %2175 = select i1 %.not36, float %2163, float %2174, !dbg !7
  %2176 = select i1 %.not36, float %2174, float %2169, !dbg !7
  %2177 = select i1 %.not36, float %2170, float %2164, !dbg !7
  %2178 = bitcast float %2177 to i32, !dbg !7
  %2179 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1814, i32 %2178), !dbg !7
  %2180 = bitcast i32 %2179 to float, !dbg !7
  %2181 = select i1 %.not36, float %2164, float %2180, !dbg !7
  %2182 = select i1 %.not36, float %2180, float %2170, !dbg !7
  %2183 = extractelement <8 x float> %2073, i64 4, !dbg !7
  %2184 = extractelement <8 x float> %2073, i64 5, !dbg !7
  %2185 = extractelement <8 x float> %2073, i64 6, !dbg !7
  %2186 = extractelement <8 x float> %2073, i64 7, !dbg !7
  %2187 = select i1 %.not, float %2184, float %2183, !dbg !7
  %2188 = bitcast float %2187 to i32, !dbg !7
  %2189 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1798, i32 %2188), !dbg !7
  %2190 = bitcast i32 %2189 to float, !dbg !7
  %2191 = select i1 %.not, float %2183, float %2190, !dbg !7
  %2192 = select i1 %.not, float %2190, float %2184, !dbg !7
  %2193 = select i1 %.not, float %2186, float %2185, !dbg !7
  %2194 = bitcast float %2193 to i32, !dbg !7
  %2195 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1798, i32 %2194), !dbg !7
  %2196 = bitcast i32 %2195 to float, !dbg !7
  %2197 = select i1 %.not, float %2185, float %2196, !dbg !7
  %2198 = select i1 %.not, float %2196, float %2186, !dbg !7
  %2199 = select i1 %.not36, float %2197, float %2191, !dbg !7
  %2200 = bitcast float %2199 to i32, !dbg !7
  %2201 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1814, i32 %2200), !dbg !7
  %2202 = bitcast i32 %2201 to float, !dbg !7
  %2203 = select i1 %.not36, float %2191, float %2202, !dbg !7
  %2204 = select i1 %.not36, float %2202, float %2197, !dbg !7
  %2205 = select i1 %.not36, float %2198, float %2192, !dbg !7
  %2206 = bitcast float %2205 to i32, !dbg !7
  %2207 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1814, i32 %2206), !dbg !7
  %2208 = bitcast i32 %2207 to float, !dbg !7
  %2209 = select i1 %.not36, float %2192, float %2208, !dbg !7
  %2210 = select i1 %.not36, float %2208, float %2198, !dbg !7
  %2211 = extractelement <8 x float> %2102, i64 4, !dbg !7
  %2212 = extractelement <8 x float> %2102, i64 5, !dbg !7
  %2213 = extractelement <8 x float> %2102, i64 6, !dbg !7
  %2214 = extractelement <8 x float> %2102, i64 7, !dbg !7
  %2215 = select i1 %.not, float %2212, float %2211, !dbg !7
  %2216 = bitcast float %2215 to i32, !dbg !7
  %2217 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1798, i32 %2216), !dbg !7
  %2218 = bitcast i32 %2217 to float, !dbg !7
  %2219 = select i1 %.not, float %2211, float %2218, !dbg !7
  %2220 = select i1 %.not, float %2218, float %2212, !dbg !7
  %2221 = select i1 %.not, float %2214, float %2213, !dbg !7
  %2222 = bitcast float %2221 to i32, !dbg !7
  %2223 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1798, i32 %2222), !dbg !7
  %2224 = bitcast i32 %2223 to float, !dbg !7
  %2225 = select i1 %.not, float %2213, float %2224, !dbg !7
  %2226 = select i1 %.not, float %2224, float %2214, !dbg !7
  %2227 = select i1 %.not36, float %2225, float %2219, !dbg !7
  %2228 = bitcast float %2227 to i32, !dbg !7
  %2229 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1814, i32 %2228), !dbg !7
  %2230 = select i1 %.not36, float %2226, float %2220, !dbg !7
  %2231 = bitcast float %2230 to i32, !dbg !7
  %2232 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1814, i32 %2231), !dbg !7
  br i1 true, label %2233, label %2297, !dbg !21

2233:                                             ; preds = %1782
  %2234 = bitcast i32 %2229 to float, !dbg !7
  %2235 = select i1 %.not36, float %2219, float %2234, !dbg !7
  %2236 = insertelement <4 x float> poison, float %2235, i64 0, !dbg !19
  %2237 = bitcast i32 %2232 to float, !dbg !7
  %2238 = select i1 %.not36, float %2220, float %2237, !dbg !7
  %2239 = insertelement <4 x float> %2236, float %2238, i64 1, !dbg !19
  %2240 = select i1 %.not36, float %2234, float %2225, !dbg !7
  %2241 = insertelement <4 x float> %2239, float %2240, i64 2, !dbg !19
  %2242 = select i1 %.not36, float %2237, float %2226, !dbg !7
  %2243 = insertelement <4 x float> %2241, float %2242, i64 3, !dbg !19
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %2126, !dbg !19
  %normalized = fmul <4 x float> %2243, %row.reciprocal, !dbg !19
  %2244 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !20
  %2245 = extractelement <4 x half> %2244, i64 3, !dbg !20
  %2246 = extractelement <4 x half> %2244, i64 2, !dbg !20
  %2247 = extractelement <4 x half> %2244, i64 1, !dbg !20
  %2248 = extractelement <4 x half> %2244, i64 0, !dbg !20
  %2249 = bitcast i32 %2121 to float, !dbg !7
  %2250 = select i1 %.not36, float %2111, float %2249, !dbg !7
  %2251 = insertelement <4 x float> poison, float %2250, i64 0, !dbg !19
  %2252 = bitcast i32 %2124 to float, !dbg !7
  %2253 = select i1 %.not36, float %2112, float %2252, !dbg !7
  %2254 = insertelement <4 x float> %2251, float %2253, i64 1, !dbg !19
  %2255 = select i1 %.not36, float %2249, float %2117, !dbg !7
  %2256 = insertelement <4 x float> %2254, float %2255, i64 2, !dbg !19
  %2257 = select i1 %.not36, float %2252, float %2118, !dbg !7
  %2258 = insertelement <4 x float> %2256, float %2257, i64 3, !dbg !19
  %normalized104 = fmul <4 x float> %2258, %row.reciprocal, !dbg !19
  %2259 = fptrunc <4 x float> %normalized104 to <4 x half>, !dbg !20
  %2260 = extractelement <4 x half> %2259, i64 3, !dbg !20
  %2261 = extractelement <4 x half> %2259, i64 2, !dbg !20
  %2262 = extractelement <4 x half> %2259, i64 1, !dbg !20
  %2263 = extractelement <4 x half> %2259, i64 0, !dbg !20
  %2264 = bitcast i32 %2011 to float, !dbg !7
  %2265 = select i1 %.not36, float %2001, float %2264, !dbg !7
  %2266 = insertelement <4 x float> poison, float %2265, i64 0, !dbg !19
  %2267 = bitcast i32 %2014 to float, !dbg !7
  %2268 = select i1 %.not36, float %2002, float %2267, !dbg !7
  %2269 = insertelement <4 x float> %2266, float %2268, i64 1, !dbg !19
  %2270 = select i1 %.not36, float %2264, float %2007, !dbg !7
  %2271 = insertelement <4 x float> %2269, float %2270, i64 2, !dbg !19
  %2272 = select i1 %.not36, float %2267, float %2008, !dbg !7
  %2273 = insertelement <4 x float> %2271, float %2272, i64 3, !dbg !19
  %row.reciprocal105 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1908, !dbg !19
  %normalized106 = fmul <4 x float> %2273, %row.reciprocal105, !dbg !19
  %2274 = fptrunc <4 x float> %normalized106 to <4 x half>, !dbg !20
  %2275 = extractelement <4 x half> %2274, i64 3, !dbg !20
  %2276 = extractelement <4 x half> %2274, i64 2, !dbg !20
  %2277 = extractelement <4 x half> %2274, i64 1, !dbg !20
  %2278 = extractelement <4 x half> %2274, i64 0, !dbg !20
  %2279 = bitcast i32 %1903 to float, !dbg !7
  %2280 = select i1 %.not36, float %1893, float %2279, !dbg !7
  %2281 = insertelement <4 x float> poison, float %2280, i64 0, !dbg !19
  %2282 = bitcast i32 %1906 to float, !dbg !7
  %2283 = select i1 %.not36, float %1894, float %2282, !dbg !7
  %2284 = insertelement <4 x float> %2281, float %2283, i64 1, !dbg !19
  %2285 = select i1 %.not36, float %2279, float %1899, !dbg !7
  %2286 = insertelement <4 x float> %2284, float %2285, i64 2, !dbg !19
  %2287 = select i1 %.not36, float %2282, float %1900, !dbg !7
  %2288 = insertelement <4 x float> %2286, float %2287, i64 3, !dbg !19
  %normalized107 = fmul <4 x float> %2288, %row.reciprocal105, !dbg !19
  %2289 = fptrunc <4 x float> %normalized107 to <4 x half>, !dbg !20
  %2290 = extractelement <4 x half> %2289, i64 3, !dbg !20
  %2291 = extractelement <4 x half> %2289, i64 2, !dbg !20
  %2292 = extractelement <4 x half> %2289, i64 1, !dbg !20
  %2293 = extractelement <4 x half> %2289, i64 0, !dbg !20
  %2294 = add nsw i64 %16, %22, !dbg !21
  %2295 = shl i64 %2294, 19, !dbg !21
  %2296 = and i64 %2295, 16252928, !dbg !21
  %address.thread934 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local935 = zext nneg i32 %address.thread934 to i64, !dbg !8
  %address.local936 = add i64 %25, %address.local935, !dbg !21
  %address.local937 = lshr i64 %address.local935, 6, !dbg !11
  %address.local938 = shl nuw nsw i64 %address.local937, 4, !dbg !21
  %address.local939 = add i64 %address.local936, %address.local938, !dbg !21
  %address.local940 = and i64 %address.local935, 240, !dbg !21
  %address.local941 = sub i64 %address.local939, %address.local940, !dbg !21
  %address.local942 = add i64 %address.local941, %26, !dbg !21
  %address.local943 = shl nuw nsw i64 %address.local942, 7, !dbg !21
  %address.local944 = add nuw nsw i64 %2296, %address.local943, !dbg !21
  %address.local945 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local944, !dbg !21
  %address.local946 = lshr i64 %address.local935, 2, !dbg !11
  %address.local947 = and i64 %address.local946, 12, !dbg !11
  %address.local948 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local947, !dbg !21
  store half %2293, ptr addrspace(1) %address.local948, align 2, !dbg !21
  %address.local949 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local944, !dbg !21
  %address.local950 = or disjoint i64 %address.local947, 1, !dbg !21
  %address.local951 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local949, i64 %address.local950, !dbg !21
  store half %2292, ptr addrspace(1) %address.local951, align 2, !dbg !21
  %address.local952 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local944, !dbg !21
  %address.local953 = or disjoint i64 %address.local947, 2, !dbg !21
  %address.local954 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local952, i64 %address.local953, !dbg !21
  store half %2291, ptr addrspace(1) %address.local954, align 2, !dbg !21
  %address.local955 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local944, !dbg !21
  %address.local956 = or disjoint i64 %address.local947, 3, !dbg !21
  %address.local957 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local955, i64 %address.local956, !dbg !21
  store half %2290, ptr addrspace(1) %address.local957, align 2, !dbg !21
  %address.local958 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local944, !dbg !21
  %address.local959 = or disjoint i64 %address.local947, 16, !dbg !21
  %address.local960 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local958, i64 %address.local959, !dbg !21
  store half %2278, ptr addrspace(1) %address.local960, align 2, !dbg !21
  %address.local961 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local944, !dbg !21
  %address.local962 = or disjoint i64 %address.local947, 17, !dbg !21
  %address.local963 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local961, i64 %address.local962, !dbg !21
  store half %2277, ptr addrspace(1) %address.local963, align 2, !dbg !21
  %address.local964 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local944, !dbg !21
  %address.local965 = or disjoint i64 %address.local947, 18, !dbg !21
  %address.local966 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local964, i64 %address.local965, !dbg !21
  store half %2276, ptr addrspace(1) %address.local966, align 2, !dbg !21
  %address.local967 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local944, !dbg !21
  %address.local968 = or disjoint i64 %address.local947, 19, !dbg !21
  %address.local969 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local967, i64 %address.local968, !dbg !21
  store half %2275, ptr addrspace(1) %address.local969, align 2, !dbg !21
  %address.local970 = shl i64 %address.local942, 7, !dbg !21
  %address.local971 = add i64 %address.local970, 8192, !dbg !21
  %address.local972 = add nuw nsw i64 %2296, %address.local971, !dbg !21
  %address.local973 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local972, !dbg !21
  %address.local974 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local947, !dbg !21
  store half %2263, ptr addrspace(1) %address.local974, align 2, !dbg !21
  %address.local975 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local972, !dbg !21
  %address.local976 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local975, i64 %address.local950, !dbg !21
  store half %2262, ptr addrspace(1) %address.local976, align 2, !dbg !21
  %address.local977 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local972, !dbg !21
  %address.local978 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local977, i64 %address.local953, !dbg !21
  store half %2261, ptr addrspace(1) %address.local978, align 2, !dbg !21
  %address.local979 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local972, !dbg !21
  %address.local980 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local979, i64 %address.local956, !dbg !21
  store half %2260, ptr addrspace(1) %address.local980, align 2, !dbg !21
  %address.local981 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local972, !dbg !21
  %address.local982 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local981, i64 %address.local959, !dbg !21
  store half %2248, ptr addrspace(1) %address.local982, align 2, !dbg !21
  %address.local983 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local972, !dbg !21
  %address.local984 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local983, i64 %address.local962, !dbg !21
  store half %2247, ptr addrspace(1) %address.local984, align 2, !dbg !21
  %address.local985 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local972, !dbg !21
  %address.local986 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local965, !dbg !21
  store half %2246, ptr addrspace(1) %address.local986, align 2, !dbg !21
  %address.local987 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local972, !dbg !21
  %address.local988 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local987, i64 %address.local968, !dbg !21
  store half %2245, ptr addrspace(1) %address.local988, align 2, !dbg !21
  br label %2297, !dbg !21

2297:                                             ; preds = %2233, %1782
  br i1 true, label %2298, label %2336, !dbg !21

2298:                                             ; preds = %2297
  %2299 = insertelement <4 x float> poison, float %2203, i64 0, !dbg !19
  %2300 = insertelement <4 x float> %2299, float %2209, i64 1, !dbg !19
  %2301 = insertelement <4 x float> %2300, float %2204, i64 2, !dbg !19
  %2302 = insertelement <4 x float> %2301, float %2210, i64 3, !dbg !19
  %row.reciprocal108 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %2126, !dbg !19
  %normalized109 = fmul <4 x float> %2302, %row.reciprocal108, !dbg !19
  %2303 = fptrunc <4 x float> %normalized109 to <4 x half>, !dbg !20
  %2304 = extractelement <4 x half> %2303, i64 3, !dbg !20
  %2305 = extractelement <4 x half> %2303, i64 2, !dbg !20
  %2306 = extractelement <4 x half> %2303, i64 1, !dbg !20
  %2307 = extractelement <4 x half> %2303, i64 0, !dbg !20
  %2308 = insertelement <4 x float> poison, float %2094, i64 0, !dbg !19
  %2309 = insertelement <4 x float> %2308, float %2100, i64 1, !dbg !19
  %2310 = insertelement <4 x float> %2309, float %2095, i64 2, !dbg !19
  %2311 = insertelement <4 x float> %2310, float %2101, i64 3, !dbg !19
  %normalized110 = fmul <4 x float> %2311, %row.reciprocal108, !dbg !19
  %2312 = fptrunc <4 x float> %normalized110 to <4 x half>, !dbg !20
  %2313 = extractelement <4 x half> %2312, i64 3, !dbg !20
  %2314 = extractelement <4 x half> %2312, i64 2, !dbg !20
  %2315 = extractelement <4 x half> %2312, i64 1, !dbg !20
  %2316 = extractelement <4 x half> %2312, i64 0, !dbg !20
  %2317 = insertelement <4 x float> poison, float %1985, i64 0, !dbg !19
  %2318 = insertelement <4 x float> %2317, float %1991, i64 1, !dbg !19
  %2319 = insertelement <4 x float> %2318, float %1986, i64 2, !dbg !19
  %2320 = insertelement <4 x float> %2319, float %1992, i64 3, !dbg !19
  %row.reciprocal111 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1908, !dbg !19
  %normalized112 = fmul <4 x float> %2320, %row.reciprocal111, !dbg !19
  %2321 = fptrunc <4 x float> %normalized112 to <4 x half>, !dbg !20
  %2322 = extractelement <4 x half> %2321, i64 3, !dbg !20
  %2323 = extractelement <4 x half> %2321, i64 2, !dbg !20
  %2324 = extractelement <4 x half> %2321, i64 1, !dbg !20
  %2325 = extractelement <4 x half> %2321, i64 0, !dbg !20
  %2326 = insertelement <4 x float> poison, float %1876, i64 0, !dbg !19
  %2327 = insertelement <4 x float> %2326, float %1882, i64 1, !dbg !19
  %2328 = insertelement <4 x float> %2327, float %1877, i64 2, !dbg !19
  %2329 = insertelement <4 x float> %2328, float %1883, i64 3, !dbg !19
  %normalized113 = fmul <4 x float> %2329, %row.reciprocal111, !dbg !19
  %2330 = fptrunc <4 x float> %normalized113 to <4 x half>, !dbg !20
  %2331 = extractelement <4 x half> %2330, i64 3, !dbg !20
  %2332 = extractelement <4 x half> %2330, i64 2, !dbg !20
  %2333 = extractelement <4 x half> %2330, i64 1, !dbg !20
  %2334 = extractelement <4 x half> %2330, i64 0, !dbg !20
  %2335 = shl nuw nsw i64 %24, 19, !dbg !21
  %address.thread989 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local990 = zext nneg i32 %address.thread989 to i64, !dbg !8
  %address.local991 = add i64 %25, %address.local990, !dbg !21
  %address.local992 = lshr i64 %address.local990, 6, !dbg !11
  %address.local993 = shl nuw nsw i64 %address.local992, 4, !dbg !21
  %address.local994 = add i64 %address.local991, %address.local993, !dbg !21
  %address.local995 = and i64 %address.local990, 240, !dbg !21
  %address.local996 = sub i64 %address.local994, %address.local995, !dbg !21
  %address.local997 = add i64 %address.local996, %26, !dbg !21
  %address.local998 = shl nuw nsw i64 %address.local997, 7, !dbg !21
  %address.local999 = add nuw nsw i64 %2335, %address.local998, !dbg !21
  %address.local1000 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local999, !dbg !21
  %address.local1001 = lshr i64 %address.local990, 2, !dbg !11
  %address.local1002 = and i64 %address.local1001, 12, !dbg !11
  %address.local1003 = or disjoint i64 %address.local1002, 32, !dbg !21
  %address.local1004 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1000, i64 %address.local1003, !dbg !21
  store half %2334, ptr addrspace(1) %address.local1004, align 2, !dbg !21
  %address.local1005 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local999, !dbg !21
  %address.local1006 = or disjoint i64 %address.local1002, 33, !dbg !21
  %address.local1007 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1005, i64 %address.local1006, !dbg !21
  store half %2333, ptr addrspace(1) %address.local1007, align 2, !dbg !21
  %address.local1008 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local999, !dbg !21
  %address.local1009 = or disjoint i64 %address.local1002, 34, !dbg !21
  %address.local1010 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1008, i64 %address.local1009, !dbg !21
  store half %2332, ptr addrspace(1) %address.local1010, align 2, !dbg !21
  %address.local1011 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local999, !dbg !21
  %address.local1012 = or disjoint i64 %address.local1002, 35, !dbg !21
  %address.local1013 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1011, i64 %address.local1012, !dbg !21
  store half %2331, ptr addrspace(1) %address.local1013, align 2, !dbg !21
  %address.local1014 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local999, !dbg !21
  %address.local1015 = or disjoint i64 %address.local1002, 48, !dbg !21
  %address.local1016 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1014, i64 %address.local1015, !dbg !21
  store half %2325, ptr addrspace(1) %address.local1016, align 2, !dbg !21
  %address.local1017 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local999, !dbg !21
  %address.local1018 = or disjoint i64 %address.local1002, 49, !dbg !21
  %address.local1019 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1017, i64 %address.local1018, !dbg !21
  store half %2324, ptr addrspace(1) %address.local1019, align 2, !dbg !21
  %address.local1020 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local999, !dbg !21
  %address.local1021 = or disjoint i64 %address.local1002, 50, !dbg !21
  %address.local1022 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1020, i64 %address.local1021, !dbg !21
  store half %2323, ptr addrspace(1) %address.local1022, align 2, !dbg !21
  %address.local1023 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local999, !dbg !21
  %address.local1024 = or i64 %address.local1001, 51, !dbg !21
  %address.local1025 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1023, i64 %address.local1024, !dbg !21
  store half %2322, ptr addrspace(1) %address.local1025, align 2, !dbg !21
  %address.local1026 = shl i64 %address.local997, 7, !dbg !21
  %address.local1027 = add i64 %address.local1026, 8192, !dbg !21
  %address.local1028 = add nuw nsw i64 %2335, %address.local1027, !dbg !21
  %address.local1029 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1028, !dbg !21
  %address.local1030 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1029, i64 %address.local1003, !dbg !21
  store half %2316, ptr addrspace(1) %address.local1030, align 2, !dbg !21
  %address.local1031 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1028, !dbg !21
  %address.local1032 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1031, i64 %address.local1006, !dbg !21
  store half %2315, ptr addrspace(1) %address.local1032, align 2, !dbg !21
  %address.local1033 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1028, !dbg !21
  %address.local1034 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1033, i64 %address.local1009, !dbg !21
  store half %2314, ptr addrspace(1) %address.local1034, align 2, !dbg !21
  %address.local1035 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1028, !dbg !21
  %address.local1036 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1035, i64 %address.local1012, !dbg !21
  store half %2313, ptr addrspace(1) %address.local1036, align 2, !dbg !21
  %address.local1037 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1028, !dbg !21
  %address.local1038 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1037, i64 %address.local1015, !dbg !21
  store half %2307, ptr addrspace(1) %address.local1038, align 2, !dbg !21
  %address.local1039 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1028, !dbg !21
  %address.local1040 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1039, i64 %address.local1018, !dbg !21
  store half %2306, ptr addrspace(1) %address.local1040, align 2, !dbg !21
  %address.local1041 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1028, !dbg !21
  %address.local1042 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1041, i64 %address.local1021, !dbg !21
  store half %2305, ptr addrspace(1) %address.local1042, align 2, !dbg !21
  %address.local1043 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1028, !dbg !21
  %address.local1044 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1043, i64 %address.local1024, !dbg !21
  store half %2304, ptr addrspace(1) %address.local1044, align 2, !dbg !21
  br label %2336, !dbg !21

2336:                                             ; preds = %2298, %2297
  br i1 true, label %2337, label %2375, !dbg !21

2337:                                             ; preds = %2336
  %2338 = insertelement <4 x float> poison, float %2175, i64 0, !dbg !19
  %2339 = insertelement <4 x float> %2338, float %2181, i64 1, !dbg !19
  %2340 = insertelement <4 x float> %2339, float %2176, i64 2, !dbg !19
  %2341 = insertelement <4 x float> %2340, float %2182, i64 3, !dbg !19
  %row.reciprocal114 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %2126, !dbg !19
  %normalized115 = fmul <4 x float> %2341, %row.reciprocal114, !dbg !19
  %2342 = fptrunc <4 x float> %normalized115 to <4 x half>, !dbg !20
  %2343 = extractelement <4 x half> %2342, i64 3, !dbg !20
  %2344 = extractelement <4 x half> %2342, i64 2, !dbg !20
  %2345 = extractelement <4 x half> %2342, i64 1, !dbg !20
  %2346 = extractelement <4 x half> %2342, i64 0, !dbg !20
  %2347 = insertelement <4 x float> poison, float %2065, i64 0, !dbg !19
  %2348 = insertelement <4 x float> %2347, float %2071, i64 1, !dbg !19
  %2349 = insertelement <4 x float> %2348, float %2066, i64 2, !dbg !19
  %2350 = insertelement <4 x float> %2349, float %2072, i64 3, !dbg !19
  %normalized116 = fmul <4 x float> %2350, %row.reciprocal114, !dbg !19
  %2351 = fptrunc <4 x float> %normalized116 to <4 x half>, !dbg !20
  %2352 = extractelement <4 x half> %2351, i64 3, !dbg !20
  %2353 = extractelement <4 x half> %2351, i64 2, !dbg !20
  %2354 = extractelement <4 x half> %2351, i64 1, !dbg !20
  %2355 = extractelement <4 x half> %2351, i64 0, !dbg !20
  %2356 = insertelement <4 x float> poison, float %1957, i64 0, !dbg !19
  %2357 = insertelement <4 x float> %2356, float %1963, i64 1, !dbg !19
  %2358 = insertelement <4 x float> %2357, float %1958, i64 2, !dbg !19
  %2359 = insertelement <4 x float> %2358, float %1964, i64 3, !dbg !19
  %row.reciprocal117 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1908, !dbg !19
  %normalized118 = fmul <4 x float> %2359, %row.reciprocal117, !dbg !19
  %2360 = fptrunc <4 x float> %normalized118 to <4 x half>, !dbg !20
  %2361 = extractelement <4 x half> %2360, i64 3, !dbg !20
  %2362 = extractelement <4 x half> %2360, i64 2, !dbg !20
  %2363 = extractelement <4 x half> %2360, i64 1, !dbg !20
  %2364 = extractelement <4 x half> %2360, i64 0, !dbg !20
  %2365 = insertelement <4 x float> poison, float %1847, i64 0, !dbg !19
  %2366 = insertelement <4 x float> %2365, float %1853, i64 1, !dbg !19
  %2367 = insertelement <4 x float> %2366, float %1848, i64 2, !dbg !19
  %2368 = insertelement <4 x float> %2367, float %1854, i64 3, !dbg !19
  %normalized119 = fmul <4 x float> %2368, %row.reciprocal117, !dbg !19
  %2369 = fptrunc <4 x float> %normalized119 to <4 x half>, !dbg !20
  %2370 = extractelement <4 x half> %2369, i64 3, !dbg !20
  %2371 = extractelement <4 x half> %2369, i64 2, !dbg !20
  %2372 = extractelement <4 x half> %2369, i64 1, !dbg !20
  %2373 = extractelement <4 x half> %2369, i64 0, !dbg !20
  %2374 = shl nuw nsw i64 %24, 19, !dbg !21
  %address.thread1045 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local1046 = zext nneg i32 %address.thread1045 to i64, !dbg !8
  %address.local1047 = add i64 %25, %address.local1046, !dbg !21
  %address.local1048 = lshr i64 %address.local1046, 6, !dbg !11
  %address.local1049 = shl nuw nsw i64 %address.local1048, 4, !dbg !21
  %address.local1050 = add i64 %address.local1047, %address.local1049, !dbg !21
  %address.local1051 = and i64 %address.local1046, 240, !dbg !21
  %address.local1052 = sub i64 %address.local1050, %address.local1051, !dbg !21
  %address.local1053 = add i64 %address.local1052, %26, !dbg !21
  %address.local1054 = shl nuw nsw i64 %address.local1053, 7, !dbg !21
  %address.local1055 = add nuw nsw i64 %2374, %address.local1054, !dbg !21
  %address.local1056 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1055, !dbg !21
  %address.local1057 = lshr i64 %address.local1046, 2, !dbg !11
  %address.local1058 = and i64 %address.local1057, 12, !dbg !11
  %address.local1059 = or disjoint i64 %address.local1058, 64, !dbg !21
  %address.local1060 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1056, i64 %address.local1059, !dbg !21
  store half %2373, ptr addrspace(1) %address.local1060, align 2, !dbg !21
  %address.local1061 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1055, !dbg !21
  %address.local1062 = or disjoint i64 %address.local1058, 65, !dbg !21
  %address.local1063 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1061, i64 %address.local1062, !dbg !21
  store half %2372, ptr addrspace(1) %address.local1063, align 2, !dbg !21
  %address.local1064 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1055, !dbg !21
  %address.local1065 = or disjoint i64 %address.local1058, 66, !dbg !21
  %address.local1066 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1064, i64 %address.local1065, !dbg !21
  store half %2371, ptr addrspace(1) %address.local1066, align 2, !dbg !21
  %address.local1067 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1055, !dbg !21
  %address.local1068 = or disjoint i64 %address.local1058, 67, !dbg !21
  %address.local1069 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1067, i64 %address.local1068, !dbg !21
  store half %2370, ptr addrspace(1) %address.local1069, align 2, !dbg !21
  %address.local1070 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1055, !dbg !21
  %address.local1071 = or disjoint i64 %address.local1058, 80, !dbg !21
  %address.local1072 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1070, i64 %address.local1071, !dbg !21
  store half %2364, ptr addrspace(1) %address.local1072, align 2, !dbg !21
  %address.local1073 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1055, !dbg !21
  %address.local1074 = or disjoint i64 %address.local1058, 81, !dbg !21
  %address.local1075 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1073, i64 %address.local1074, !dbg !21
  store half %2363, ptr addrspace(1) %address.local1075, align 2, !dbg !21
  %address.local1076 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1055, !dbg !21
  %address.local1077 = or disjoint i64 %address.local1058, 82, !dbg !21
  %address.local1078 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1076, i64 %address.local1077, !dbg !21
  store half %2362, ptr addrspace(1) %address.local1078, align 2, !dbg !21
  %address.local1079 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1055, !dbg !21
  %address.local1080 = or disjoint i64 %address.local1058, 83, !dbg !21
  %address.local1081 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1079, i64 %address.local1080, !dbg !21
  store half %2361, ptr addrspace(1) %address.local1081, align 2, !dbg !21
  %address.local1082 = shl i64 %address.local1053, 7, !dbg !21
  %address.local1083 = add i64 %address.local1082, 8192, !dbg !21
  %address.local1084 = add nuw nsw i64 %2374, %address.local1083, !dbg !21
  %address.local1085 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1084, !dbg !21
  %address.local1086 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1085, i64 %address.local1059, !dbg !21
  store half %2355, ptr addrspace(1) %address.local1086, align 2, !dbg !21
  %address.local1087 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1084, !dbg !21
  %address.local1088 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1087, i64 %address.local1062, !dbg !21
  store half %2354, ptr addrspace(1) %address.local1088, align 2, !dbg !21
  %address.local1089 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1084, !dbg !21
  %address.local1090 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1089, i64 %address.local1065, !dbg !21
  store half %2353, ptr addrspace(1) %address.local1090, align 2, !dbg !21
  %address.local1091 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1084, !dbg !21
  %address.local1092 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1091, i64 %address.local1068, !dbg !21
  store half %2352, ptr addrspace(1) %address.local1092, align 2, !dbg !21
  %address.local1093 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1084, !dbg !21
  %address.local1094 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1093, i64 %address.local1071, !dbg !21
  store half %2346, ptr addrspace(1) %address.local1094, align 2, !dbg !21
  %address.local1095 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1084, !dbg !21
  %address.local1096 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1095, i64 %address.local1074, !dbg !21
  store half %2345, ptr addrspace(1) %address.local1096, align 2, !dbg !21
  %address.local1097 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1084, !dbg !21
  %address.local1098 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1097, i64 %address.local1077, !dbg !21
  store half %2344, ptr addrspace(1) %address.local1098, align 2, !dbg !21
  %address.local1099 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1084, !dbg !21
  %address.local1100 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1099, i64 %address.local1080, !dbg !21
  store half %2343, ptr addrspace(1) %address.local1100, align 2, !dbg !21
  br label %2375, !dbg !21

2375:                                             ; preds = %2337, %2336
  br i1 true, label %2376, label %2414, !dbg !21

2376:                                             ; preds = %2375
  %2377 = insertelement <4 x float> poison, float %2147, i64 0, !dbg !19
  %2378 = insertelement <4 x float> %2377, float %2153, i64 1, !dbg !19
  %2379 = insertelement <4 x float> %2378, float %2148, i64 2, !dbg !19
  %2380 = insertelement <4 x float> %2379, float %2154, i64 3, !dbg !19
  %row.reciprocal120 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %2126, !dbg !19
  %normalized121 = fmul <4 x float> %2380, %row.reciprocal120, !dbg !19
  %2381 = fptrunc <4 x float> %normalized121 to <4 x half>, !dbg !20
  %2382 = extractelement <4 x half> %2381, i64 3, !dbg !20
  %2383 = extractelement <4 x half> %2381, i64 2, !dbg !20
  %2384 = extractelement <4 x half> %2381, i64 1, !dbg !20
  %2385 = extractelement <4 x half> %2381, i64 0, !dbg !20
  %2386 = insertelement <4 x float> poison, float %2036, i64 0, !dbg !19
  %2387 = insertelement <4 x float> %2386, float %2042, i64 1, !dbg !19
  %2388 = insertelement <4 x float> %2387, float %2037, i64 2, !dbg !19
  %2389 = insertelement <4 x float> %2388, float %2043, i64 3, !dbg !19
  %normalized122 = fmul <4 x float> %2389, %row.reciprocal120, !dbg !19
  %2390 = fptrunc <4 x float> %normalized122 to <4 x half>, !dbg !20
  %2391 = extractelement <4 x half> %2390, i64 3, !dbg !20
  %2392 = extractelement <4 x half> %2390, i64 2, !dbg !20
  %2393 = extractelement <4 x half> %2390, i64 1, !dbg !20
  %2394 = extractelement <4 x half> %2390, i64 0, !dbg !20
  %2395 = insertelement <4 x float> poison, float %1929, i64 0, !dbg !19
  %2396 = insertelement <4 x float> %2395, float %1935, i64 1, !dbg !19
  %2397 = insertelement <4 x float> %2396, float %1930, i64 2, !dbg !19
  %2398 = insertelement <4 x float> %2397, float %1936, i64 3, !dbg !19
  %row.reciprocal123 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1908, !dbg !19
  %normalized124 = fmul <4 x float> %2398, %row.reciprocal123, !dbg !19
  %2399 = fptrunc <4 x float> %normalized124 to <4 x half>, !dbg !20
  %2400 = extractelement <4 x half> %2399, i64 3, !dbg !20
  %2401 = extractelement <4 x half> %2399, i64 2, !dbg !20
  %2402 = extractelement <4 x half> %2399, i64 1, !dbg !20
  %2403 = extractelement <4 x half> %2399, i64 0, !dbg !20
  %2404 = insertelement <4 x float> poison, float %1818, i64 0, !dbg !19
  %2405 = insertelement <4 x float> %2404, float %1824, i64 1, !dbg !19
  %2406 = insertelement <4 x float> %2405, float %1819, i64 2, !dbg !19
  %2407 = insertelement <4 x float> %2406, float %1825, i64 3, !dbg !19
  %normalized125 = fmul <4 x float> %2407, %row.reciprocal123, !dbg !19
  %2408 = fptrunc <4 x float> %normalized125 to <4 x half>, !dbg !20
  %2409 = extractelement <4 x half> %2408, i64 3, !dbg !20
  %2410 = extractelement <4 x half> %2408, i64 2, !dbg !20
  %2411 = extractelement <4 x half> %2408, i64 1, !dbg !20
  %2412 = extractelement <4 x half> %2408, i64 0, !dbg !20
  %2413 = shl nuw nsw i64 %24, 19, !dbg !21
  %address.thread1101 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local1102 = zext nneg i32 %address.thread1101 to i64, !dbg !8
  %address.local1103 = add i64 %25, %address.local1102, !dbg !21
  %address.local1104 = lshr i64 %address.local1102, 6, !dbg !11
  %address.local1105 = shl nuw nsw i64 %address.local1104, 4, !dbg !21
  %address.local1106 = add i64 %address.local1103, %address.local1105, !dbg !21
  %address.local1107 = and i64 %address.local1102, 240, !dbg !21
  %address.local1108 = sub i64 %address.local1106, %address.local1107, !dbg !21
  %address.local1109 = add i64 %address.local1108, %26, !dbg !21
  %address.local1110 = shl nuw nsw i64 %address.local1109, 7, !dbg !21
  %address.local1111 = add nuw nsw i64 %2413, %address.local1110, !dbg !21
  %address.local1112 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1111, !dbg !21
  %address.local1113 = lshr i64 %address.local1102, 2, !dbg !11
  %address.local1114 = and i64 %address.local1113, 12, !dbg !11
  %address.local1115 = or disjoint i64 %address.local1114, 96, !dbg !21
  %address.local1116 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1112, i64 %address.local1115, !dbg !21
  store half %2412, ptr addrspace(1) %address.local1116, align 2, !dbg !21
  %address.local1117 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1111, !dbg !21
  %address.local1118 = or disjoint i64 %address.local1114, 97, !dbg !21
  %address.local1119 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1117, i64 %address.local1118, !dbg !21
  store half %2411, ptr addrspace(1) %address.local1119, align 2, !dbg !21
  %address.local1120 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1111, !dbg !21
  %address.local1121 = or disjoint i64 %address.local1114, 98, !dbg !21
  %address.local1122 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1120, i64 %address.local1121, !dbg !21
  store half %2410, ptr addrspace(1) %address.local1122, align 2, !dbg !21
  %address.local1123 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1111, !dbg !21
  %address.local1124 = or disjoint i64 %address.local1114, 99, !dbg !21
  %address.local1125 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1123, i64 %address.local1124, !dbg !21
  store half %2409, ptr addrspace(1) %address.local1125, align 2, !dbg !21
  %address.local1126 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1111, !dbg !21
  %address.local1127 = or disjoint i64 %address.local1114, 112, !dbg !21
  %address.local1128 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1126, i64 %address.local1127, !dbg !21
  store half %2403, ptr addrspace(1) %address.local1128, align 2, !dbg !21
  %address.local1129 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1111, !dbg !21
  %address.local1130 = or disjoint i64 %address.local1114, 113, !dbg !21
  %address.local1131 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1129, i64 %address.local1130, !dbg !21
  store half %2402, ptr addrspace(1) %address.local1131, align 2, !dbg !21
  %address.local1132 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1111, !dbg !21
  %address.local1133 = or disjoint i64 %address.local1114, 114, !dbg !21
  %address.local1134 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1132, i64 %address.local1133, !dbg !21
  store half %2401, ptr addrspace(1) %address.local1134, align 2, !dbg !21
  %address.local1135 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1111, !dbg !21
  %address.local1136 = or i64 %address.local1113, 115, !dbg !21
  %address.local1137 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1135, i64 %address.local1136, !dbg !21
  store half %2400, ptr addrspace(1) %address.local1137, align 2, !dbg !21
  %address.local1138 = shl i64 %address.local1109, 7, !dbg !21
  %address.local1139 = add i64 %address.local1138, 8192, !dbg !21
  %address.local1140 = add nuw nsw i64 %2413, %address.local1139, !dbg !21
  %address.local1141 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !21
  %address.local1142 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1141, i64 %address.local1115, !dbg !21
  store half %2394, ptr addrspace(1) %address.local1142, align 2, !dbg !21
  %address.local1143 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !21
  %address.local1144 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1143, i64 %address.local1118, !dbg !21
  store half %2393, ptr addrspace(1) %address.local1144, align 2, !dbg !21
  %address.local1145 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !21
  %address.local1146 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1145, i64 %address.local1121, !dbg !21
  store half %2392, ptr addrspace(1) %address.local1146, align 2, !dbg !21
  %address.local1147 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !21
  %address.local1148 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1147, i64 %address.local1124, !dbg !21
  store half %2391, ptr addrspace(1) %address.local1148, align 2, !dbg !21
  %address.local1149 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !21
  %address.local1150 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1149, i64 %address.local1127, !dbg !21
  store half %2385, ptr addrspace(1) %address.local1150, align 2, !dbg !21
  %address.local1151 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !21
  %address.local1152 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1151, i64 %address.local1130, !dbg !21
  store half %2384, ptr addrspace(1) %address.local1152, align 2, !dbg !21
  %address.local1153 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !21
  %address.local1154 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1153, i64 %address.local1133, !dbg !21
  store half %2383, ptr addrspace(1) %address.local1154, align 2, !dbg !21
  %address.local1155 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !21
  %address.local1156 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1155, i64 %address.local1136, !dbg !21
  store half %2382, ptr addrspace(1) %address.local1156, align 2, !dbg !21
  br label %2414, !dbg !21

2414:                                             ; preds = %2376, %2375
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
!llvm.module.flags = !{!38}

!0 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/online_softmax_native_acc/export/run-20261023-182033-aeo9_47b/input-0001-test_input_1x32x4096x128-128-128-32-online/retained32-baseline")
!1 = !{}
!2 = !DISubroutineType(types: !1)
!3 = distinct !DICompileUnit(language: DW_LANG_C, file: !0, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!4 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !0, file: !0, line: 23, type: !2, scopeLine: 23, spFlags: DISPFlagDefinition, unit: !3)
!5 = !DILocation(line: 23, column: 3, scope: !4)
!7 = !DILocation(line: 60, column: 24, scope: !4)
!8 = !DILocation(line: 70, column: 22, scope: !4)
!10 = !DILocation(line: 48, column: 17, scope: !4)
!11 = !DILocation(line: 72, column: 24, scope: !4)
!13 = !DILocation(line: 77, column: 19, scope: !4)
!14 = !DILocation(line: 102, column: 23, scope: !4)
!15 = !DILocation(line: 0, scope: !4)
!16 = !DILocation(line: 71, column: 22, scope: !4)
!17 = !DILocation(line: 73, column: 26, scope: !4)
!18 = !DILocation(line: 101, column: 20, scope: !4)
!19 = !DILocation(line: 107, column: 23, scope: !4)
!20 = !DILocation(line: 108, column: 19, scope: !4)
!21 = !DILocation(line: 109, column: 9, scope: !4)
!23 = !DILocation(line: 79, column: 24, scope: !4)
!24 = !DILocation(line: 80, column: 23, scope: !4)
!25 = !DILocation(line: 81, column: 26, scope: !4)
!26 = !DILocation(line: 88, column: 21, scope: !4)
!27 = !DILocation(line: 89, column: 20, scope: !4)
!28 = !DILocation(line: 90, column: 22, scope: !4)
!29 = !DILocation(line: 91, column: 23, scope: !4)
!30 = !DILocation(line: 92, column: 20, scope: !4)
!31 = !DILocation(line: 93, column: 22, scope: !4)
!32 = !DILocation(line: 94, column: 22, scope: !4)
!33 = !DILocation(line: 95, column: 20, scope: !4)
!34 = !DILocation(line: 96, column: 16, scope: !4)
!35 = !DILocation(line: 103, column: 22, scope: !4)
!36 = !DILocation(line: 104, column: 20, scope: !4)
!37 = !{i32 256, i32 1, i32 1}
!38 = !{i32 2, !"Debug Info Version", i32 3}
!39 = !{i32 0, i32 256}
