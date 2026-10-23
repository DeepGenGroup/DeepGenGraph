; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/tune_retention_integration/run-20261023-160942-dxpgegoa/retained32-baseline/input.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/tune_retention_integration/run-20261023-160942-dxpgegoa/retained32-baseline/input.mlir"

@shm_0 = addrspace(3) global [16384 x i8] undef, align 16

; Function Attrs: convergent willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

declare float @__ocml_exp2_f32(float)

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3) #1 !dbg !4 !reqd_work_group_size !32 {
  %5 = call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !5
  %6 = sext i32 %5 to i64, !dbg !5
  %7 = call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !5
  %8 = sext i32 %7 to i64, !dbg !5
  %9 = shl nsw i64 %8, 5, !dbg !5
  %10 = add nsw i64 %9, %6, !dbg !5
  %11 = lshr i64 %10, 5, !dbg !5
  %12 = and i64 %11, 576460752303423480, !dbg !5
  %13 = and i64 %6, 7, !dbg !5
  %14 = or disjoint i64 %12, %13, !dbg !5
  %15 = sub nsw i64 31, %14, !dbg !5
  %16 = lshr i64 %10, 3, !dbg !5
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
  %31 = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !8, !range !34
  %32 = zext nneg i32 %31 to i64, !dbg !8
  %33 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %30, !dbg !8
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !8, !range !34, !frisk.register_lifetimes !1
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
  fence syncscope("workgroup") release, !dbg !13
  call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %188 = or disjoint i64 %64, 127, !dbg !14
  %189 = add i64 %64, 128, !dbg !14
  %190 = icmp uge i64 %189, %188, !dbg !14
  %191 = sub nsw i64 32, %14, !dbg !16
  %192 = shl i64 %16, 7, !dbg !17
  %193 = and i64 %192, 3968, !dbg !17
  %194 = shl nsw i64 %63, 12, !dbg !17
  %195 = or disjoint i64 %193, %194, !dbg !17
  %196 = lshr i64 %32, 2, !dbg !18
  %197 = shl nuw nsw i64 %32, 2, !dbg !18
  %198 = mul nsw i64 %196, -16, !dbg !18
  %199 = add nsw i64 %197, %198, !dbg !18
  %200 = and i32 %31, 1, !dbg !18
  %index.compact279 = zext nneg i32 %200 to i64, !dbg !18
  %201 = or disjoint i64 %199, %index.compact279, !dbg !18
  %202 = add nuw nsw i64 %201, 1, !dbg !18
  %203 = sdiv i64 %202, 2, !dbg !18
  br label %204, !dbg !13

204:                                              ; preds = %847, %4
  %205 = phi i64 [ %1580, %847 ], [ 0, %4 ], !dbg !13
  %206 = phi [2 x <8 x float>] [ %1567, %847 ], [ zeroinitializer, %4 ], !dbg !13
  %207 = phi [2 x <1 x float>] [ %1189, %847 ], [ zeroinitializer, %4 ], !dbg !13
  %208 = phi [2 x <8 x float>] [ %1571, %847 ], [ zeroinitializer, %4 ], !dbg !13
  %209 = phi [2 x <8 x float>] [ %1575, %847 ], [ zeroinitializer, %4 ], !dbg !13
  %210 = phi [2 x <8 x float>] [ %1579, %847 ], [ zeroinitializer, %4 ], !dbg !13
  %211 = icmp slt i64 %205, %191, !dbg !13
  br i1 %211, label %212, label %1581, !dbg !13

212:                                              ; preds = %204
  %213 = shl nsw i64 %205, 7, !dbg !13
  %214 = lshr i64 %205, 12, !dbg !17
  %215 = add nuw nsw i64 %214, %16, !dbg !17
  %216 = lshr i64 %205, 5, !dbg !17
  %217 = and i64 %216, 127, !dbg !17
  %218 = shl i64 %215, 19, !dbg !17
  %219 = and i64 %218, 16252928, !dbg !17
  %220 = shl i64 %216, 19, !dbg !17
  %221 = shl i64 %205, 14, !dbg !17
  %222 = sub i64 %221, %220, !dbg !17
  %223 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %219, !dbg !17
  %224 = getelementptr [2 x i8], ptr addrspace(1) %223, i64 %217, !dbg !17
  %225 = getelementptr [2 x i8], ptr addrspace(1) %224, i64 %222, !dbg !17
  %address.thread433 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !17, !range !34, !frisk.register_lifetimes !1
  %address.local434 = zext nneg i32 %address.thread433 to i64, !dbg !8
  %address.local435 = shl nuw nsw i64 %address.local434, 3, !dbg !8
  %address.local436 = and i64 %address.local435, 1536, !dbg !8
  %address.local437 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local436, !dbg !8
  %address.local438 = ptrtoint ptr addrspace(3) %address.local437 to i64, !dbg !8
  %address.local439 = trunc i64 %address.local438 to i32, !dbg !8
  %226 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local439), !dbg !17
  %227 = zext i32 %226 to i64, !dbg !17
  %228 = inttoptr i64 %227 to ptr addrspace(3), !dbg !17
  %address.local440 = lshr i64 %address.local434, 4, !dbg !17
  %address.local441 = and i64 %address.local440, 8, !dbg !17
  %address.local442 = and i64 %address.local434, 123, !dbg !17
  %address.local443 = lshr i64 %address.local434, 2, !dbg !17
  %address.local444 = and i64 %address.local443, 4, !dbg !17
  %address.local445 = add nuw nsw i64 %address.local444, %address.local434, !dbg !17
  %address.local446 = and i64 %address.local445, 4, !dbg !17
  %address.local447 = or disjoint i64 %address.local442, %address.local446, !dbg !17
  %address.local448 = shl nuw nsw i64 %address.local447, 7, !dbg !17
  %address.local449 = or disjoint i64 %address.local441, %address.local448, !dbg !17
  %address.local450 = getelementptr [2 x i8], ptr addrspace(1) %225, i64 %address.local449, !dbg !17
  %229 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %230 = ptrtoint ptr addrspace(1) %address.local450 to i64, !dbg !17
  %231 = sub i64 %230, %229, !dbg !17
  %dma.buffer.offset1588 = trunc i64 %231 to i32, !dbg !17
  %232 = trunc i64 %229 to i32, !dbg !17
  %233 = insertelement <4 x i32> poison, i32 %232, i64 0, !dbg !17
  %234 = lshr i64 %229, 32, !dbg !17
  %235 = trunc i64 %234 to i32, !dbg !17
  %236 = insertelement <4 x i32> %233, i32 %235, i64 1, !dbg !17
  %237 = insertelement <4 x i32> %236, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc1589 = insertelement <4 x i32> %237, i32 131072, i64 3, !dbg !17
  %238 = ptrtoint ptr addrspace(3) %228 to i32, !dbg !17
  %dma.buffer.lds1590 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %238), !dbg !17
  %239 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1590, i32 %dma.buffer.offset1588, <4 x i32> %dma.buffer.rsrc1589) #10, !dbg !17
  %address.local451 = or disjoint i64 %address.local434, 256, !dbg !8
  %address.local452 = shl nuw nsw i64 %address.local451, 3, !dbg !8
  %address.local453 = and i64 %address.local452, 3584, !dbg !8
  %address.local454 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local453, !dbg !8
  %address.local455 = ptrtoint ptr addrspace(3) %address.local454 to i64, !dbg !8
  %address.local456 = trunc i64 %address.local455 to i32, !dbg !8
  %240 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local456), !dbg !17
  %241 = zext i32 %240 to i64, !dbg !17
  %242 = inttoptr i64 %241 to ptr addrspace(3), !dbg !17
  %address.local457 = lshr i64 %address.local451, 4, !dbg !17
  %address.local458 = and i64 %address.local457, 24, !dbg !17
  %address.local459 = and i64 %address.local434, 123, !dbg !17
  %address.local460 = lshr i64 %address.local434, 2, !dbg !17
  %address.local461 = and i64 %address.local460, 4, !dbg !17
  %address.local462 = add nuw nsw i64 %address.local461, %address.local434, !dbg !17
  %address.local463 = and i64 %address.local462, 4, !dbg !17
  %address.local464 = or disjoint i64 %address.local459, %address.local463, !dbg !17
  %address.local465 = shl nuw nsw i64 %address.local464, 7, !dbg !17
  %address.local466 = or disjoint i64 %address.local458, %address.local465, !dbg !17
  %address.local467 = getelementptr [2 x i8], ptr addrspace(1) %225, i64 %address.local466, !dbg !17
  %243 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %244 = ptrtoint ptr addrspace(1) %address.local467 to i64, !dbg !17
  %245 = sub i64 %244, %243, !dbg !17
  %dma.buffer.offset1585 = trunc i64 %245 to i32, !dbg !17
  %246 = trunc i64 %243 to i32, !dbg !17
  %247 = insertelement <4 x i32> poison, i32 %246, i64 0, !dbg !17
  %248 = lshr i64 %243, 32, !dbg !17
  %249 = trunc i64 %248 to i32, !dbg !17
  %250 = insertelement <4 x i32> %247, i32 %249, i64 1, !dbg !17
  %251 = insertelement <4 x i32> %250, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc1586 = insertelement <4 x i32> %251, i32 131072, i64 3, !dbg !17
  %252 = ptrtoint ptr addrspace(3) %242 to i32, !dbg !17
  %dma.buffer.lds1587 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %252), !dbg !17
  %253 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1587, i32 %dma.buffer.offset1585, <4 x i32> %dma.buffer.rsrc1586) #10, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %254 = add nuw i64 %205, 1024, !dbg !17
  %255 = icmp slt i64 %254, 0, !dbg !17
  %256 = sub nuw i64 -1025, %205, !dbg !17
  %257 = select i1 %255, i64 %256, i64 %254, !dbg !17
  %258 = sdiv i64 %257, 4096, !dbg !17
  %.lobit37 = ashr i64 %254, 63, !dbg !17
  %259 = xor i64 %258, %.lobit37, !dbg !17
  %260 = add nsw i64 %259, %16, !dbg !17
  %261 = add nsw i64 %216, %195, !dbg !17
  %262 = add nsw i64 %261, 32, !dbg !17
  %263 = icmp slt i64 %261, -32, !dbg !17
  %264 = sub nsw i64 -33, %261, !dbg !17
  %265 = select i1 %263, i64 %264, i64 %262, !dbg !17
  %266 = sdiv i64 %265, 128, !dbg !17
  %267 = sext i1 %263 to i64, !dbg !17
  %268 = xor i64 %266, %267, !dbg !17
  %269 = shl nsw i64 %268, 7, !dbg !17
  %270 = sub i64 %195, %269, !dbg !17
  %271 = shl i64 %260, 19, !dbg !17
  %272 = and i64 %271, 16252928, !dbg !17
  %273 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %272, !dbg !17
  %274 = getelementptr [2 x i8], ptr addrspace(1) %273, i64 %216, !dbg !17
  %275 = getelementptr [2 x i8], ptr addrspace(1) %274, i64 %270, !dbg !17
  %276 = getelementptr i8, ptr addrspace(1) %275, i64 64, !dbg !17
  %277 = getelementptr [2 x i8], ptr addrspace(1) %276, i64 %222, !dbg !17
  %address.local468 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local436, !dbg !8
  %address.local469 = ptrtoint ptr addrspace(3) %address.local468 to i64, !dbg !8
  %address.local470 = trunc i64 %address.local469 to i32, !dbg !8
  %278 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local470), !dbg !17
  %279 = zext i32 %278 to i64, !dbg !17
  %280 = inttoptr i64 %279 to ptr addrspace(3), !dbg !17
  %address.local471 = getelementptr [2 x i8], ptr addrspace(1) %277, i64 %address.local449, !dbg !17
  %281 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %282 = ptrtoint ptr addrspace(1) %address.local471 to i64, !dbg !17
  %283 = sub i64 %282, %281, !dbg !17
  %dma.buffer.offset1582 = trunc i64 %283 to i32, !dbg !17
  %284 = trunc i64 %281 to i32, !dbg !17
  %285 = insertelement <4 x i32> poison, i32 %284, i64 0, !dbg !17
  %286 = lshr i64 %281, 32, !dbg !17
  %287 = trunc i64 %286 to i32, !dbg !17
  %288 = insertelement <4 x i32> %285, i32 %287, i64 1, !dbg !17
  %289 = insertelement <4 x i32> %288, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc1583 = insertelement <4 x i32> %289, i32 131072, i64 3, !dbg !17
  %290 = ptrtoint ptr addrspace(3) %280 to i32, !dbg !17
  %dma.buffer.lds1584 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %290), !dbg !17
  %291 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1584, i32 %dma.buffer.offset1582, <4 x i32> %dma.buffer.rsrc1583) #10, !dbg !17
  %address.local472 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local453, !dbg !8
  %address.local473 = ptrtoint ptr addrspace(3) %address.local472 to i64, !dbg !8
  %address.local474 = trunc i64 %address.local473 to i32, !dbg !8
  %292 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local474), !dbg !17
  %293 = zext i32 %292 to i64, !dbg !17
  %294 = inttoptr i64 %293 to ptr addrspace(3), !dbg !17
  %address.local475 = getelementptr [2 x i8], ptr addrspace(1) %277, i64 %address.local466, !dbg !17
  %295 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %296 = ptrtoint ptr addrspace(1) %address.local475 to i64, !dbg !17
  %297 = sub i64 %296, %295, !dbg !17
  %dma.buffer.offset1579 = trunc i64 %297 to i32, !dbg !17
  %298 = trunc i64 %295 to i32, !dbg !17
  %299 = insertelement <4 x i32> poison, i32 %298, i64 0, !dbg !17
  %300 = lshr i64 %295, 32, !dbg !17
  %301 = trunc i64 %300 to i32, !dbg !17
  %302 = insertelement <4 x i32> %299, i32 %301, i64 1, !dbg !17
  %303 = insertelement <4 x i32> %302, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc1580 = insertelement <4 x i32> %303, i32 131072, i64 3, !dbg !17
  %304 = ptrtoint ptr addrspace(3) %294 to i32, !dbg !17
  %dma.buffer.lds1581 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %304), !dbg !17
  %305 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1581, i32 %dma.buffer.offset1579, <4 x i32> %dma.buffer.rsrc1580) #10, !dbg !17
  %address.local476 = mul nuw nsw i64 %address.local434, 160, !dbg !18
  %address.local477 = lshr i64 %address.local434, 2, !dbg !18
  %address.local478 = mul nsw i64 %address.local477, -640, !dbg !18
  %address.local479 = add nsw i64 %address.local476, %address.local478, !dbg !18
  %address.local480 = shl nuw nsw i64 %address.local434, 1, !dbg !18
  %address.local481 = and i64 %address.local480, 24, !dbg !18
  %address.local482 = or disjoint i64 %address.local479, %address.local481, !dbg !18
  %address.local483 = shl i64 %address.local434, 62, !dbg !18
  %address.local484 = ashr exact i64 %address.local483, 57, !dbg !18
  %address.local485 = and i64 %address.local484, -64, !dbg !18
  %address.local486 = add nsw i64 %address.local482, %address.local485, !dbg !18
  %address.local487 = lshr i64 %address.local434, 2, !dbg !11
  %address.local488 = and i64 %address.local487, 12, !dbg !11
  %address.local489 = add nsw i64 %address.local486, %address.local488, !dbg !18
  %address.local490 = and i64 %address.local434, 63, !dbg !11
  %address.local491 = lshr i64 %address.local490, 5, !dbg !11
  %address.local492 = mul nuw nsw i64 %address.local491, 1016, !dbg !11
  %address.local493 = add i64 %address.local489, %address.local492, !dbg !18
  %address.local494 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local493, !dbg !18
  %306 = load <4 x half>, ptr addrspace(3) %address.local494, align 8, !dbg !18
  %307 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %306, <4 x float> zeroinitializer) #9, !dbg !18
  %lds.const.gep1426 = getelementptr i8, ptr addrspace(3) %address.local494, i64 4096, !dbg !18
  %308 = load <4 x half>, ptr addrspace(3) %lds.const.gep1426, align 8, !dbg !18
  %address.local505 = shl nuw nsw i32 %address.thread433, 1, !dbg !18
  %address.local506 = and i32 %address.local505, 24, !dbg !18
  %address.local507 = shl nuw nsw i32 %address.thread433, 8, !dbg !18
  %address.local508 = and i32 %address.local507, 768, !dbg !18
  %address.local509 = or disjoint i32 %address.local506, %address.local508, !dbg !18
  %address.local510 = shl nuw nsw i32 %address.thread433, 5, !dbg !18
  %address.local511 = and i32 %address.local510, 96, !dbg !18
  %address.local512 = or disjoint i32 %address.local509, %address.local511, !dbg !18
  %address.local513 = zext nneg i32 %address.local512 to i64, !dbg !18
  %address.base = call i64 asm sideeffect "", "=v,0"(i64 %203) #9, !dbg !18, !frisk.register_lifetimes !1
  %address.local514 = lshr i32 %address.thread433, 1, !dbg !18
  %address.local515 = and i32 %address.local514, 1, !dbg !18
  %address.local516 = zext nneg i32 %address.local515 to i64, !dbg !18
  %address.local517 = add nsw i64 %address.base, %address.local516, !dbg !18
  %address.local518 = shl i64 %address.local517, 6, !dbg !18
  %address.local519 = sub i64 %address.local513, %address.local518, !dbg !18
  %address.local520 = add i64 %address.local519, %address.local488, !dbg !18
  %address.local521 = add i64 %address.local520, %address.local492, !dbg !18
  %address.local522 = add i64 %address.local521, 32, !dbg !18
  %address.local523 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local522, !dbg !18
  %309 = load <4 x half>, ptr addrspace(3) %address.local523, align 8, !dbg !18
  %310 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %308, <4 x float> %307) #9, !dbg !18
  %311 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %309, <4 x float> zeroinitializer) #9, !dbg !18
  %lds.const.gep1427 = getelementptr i8, ptr addrspace(3) %address.local523, i64 4096, !dbg !18
  %312 = load <4 x half>, ptr addrspace(3) %lds.const.gep1427, align 8, !dbg !18
  %lds.const.gep1428 = getelementptr i8, ptr addrspace(3) %address.local494, i64 128, !dbg !18
  %313 = load <4 x half>, ptr addrspace(3) %lds.const.gep1428, align 8, !dbg !18
  %314 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %312, <4 x float> %311) #9, !dbg !18
  %315 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %313, <4 x float> zeroinitializer) #9, !dbg !18
  %lds.const.gep1429 = getelementptr i8, ptr addrspace(3) %address.local494, i64 4224, !dbg !18
  %316 = load <4 x half>, ptr addrspace(3) %lds.const.gep1429, align 8, !dbg !18
  %lds.const.gep1430 = getelementptr i8, ptr addrspace(3) %address.local523, i64 128, !dbg !18
  %317 = load <4 x half>, ptr addrspace(3) %lds.const.gep1430, align 8, !dbg !18
  %318 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %316, <4 x float> %315) #9, !dbg !18
  %319 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %317, <4 x float> zeroinitializer) #9, !dbg !18
  %lds.const.gep1431 = getelementptr i8, ptr addrspace(3) %address.local523, i64 4224, !dbg !18
  %320 = load <4 x half>, ptr addrspace(3) %lds.const.gep1431, align 8, !dbg !18
  %lds.const.gep1432 = getelementptr i8, ptr addrspace(3) %address.local494, i64 1024, !dbg !18
  %321 = load <4 x half>, ptr addrspace(3) %lds.const.gep1432, align 8, !dbg !18
  %322 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %320, <4 x float> %319) #9, !dbg !18
  %323 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %321, <4 x float> zeroinitializer) #9, !dbg !18
  %lds.const.gep1433 = getelementptr i8, ptr addrspace(3) %address.local494, i64 5120, !dbg !18
  %324 = load <4 x half>, ptr addrspace(3) %lds.const.gep1433, align 8, !dbg !18
  %lds.const.gep1434 = getelementptr i8, ptr addrspace(3) %address.local523, i64 1024, !dbg !18
  %325 = load <4 x half>, ptr addrspace(3) %lds.const.gep1434, align 8, !dbg !18
  %326 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %324, <4 x float> %323) #9, !dbg !18
  %327 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %325, <4 x float> zeroinitializer) #9, !dbg !18
  %lds.const.gep1435 = getelementptr i8, ptr addrspace(3) %address.local523, i64 5120, !dbg !18
  %328 = load <4 x half>, ptr addrspace(3) %lds.const.gep1435, align 8, !dbg !18
  %lds.const.gep1436 = getelementptr i8, ptr addrspace(3) %address.local494, i64 1152, !dbg !18
  %329 = load <4 x half>, ptr addrspace(3) %lds.const.gep1436, align 8, !dbg !18
  %330 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %328, <4 x float> %327) #9, !dbg !18
  %331 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %329, <4 x float> zeroinitializer) #9, !dbg !18
  %lds.const.gep1437 = getelementptr i8, ptr addrspace(3) %address.local494, i64 5248, !dbg !18
  %332 = load <4 x half>, ptr addrspace(3) %lds.const.gep1437, align 8, !dbg !18
  %lds.const.gep1438 = getelementptr i8, ptr addrspace(3) %address.local523, i64 1152, !dbg !18
  %333 = load <4 x half>, ptr addrspace(3) %lds.const.gep1438, align 8, !dbg !18
  %334 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %332, <4 x float> %331) #9, !dbg !18
  %335 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %333, <4 x float> zeroinitializer) #9, !dbg !18
  %lds.const.gep1439 = getelementptr i8, ptr addrspace(3) %address.local523, i64 5248, !dbg !18
  %336 = load <4 x half>, ptr addrspace(3) %lds.const.gep1439, align 8, !dbg !18
  %337 = load <4 x half>, ptr addrspace(3) %address.local494, align 8, !dbg !18
  %338 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %336, <4 x float> %335) #9, !dbg !18
  %339 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %337, <4 x float> zeroinitializer) #9, !dbg !18
  %lds.const.gep1440 = getelementptr i8, ptr addrspace(3) %address.local494, i64 4096, !dbg !18
  %340 = load <4 x half>, ptr addrspace(3) %lds.const.gep1440, align 8, !dbg !18
  %341 = load <4 x half>, ptr addrspace(3) %address.local523, align 8, !dbg !18
  %342 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %340, <4 x float> %339) #9, !dbg !18
  %343 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %341, <4 x float> zeroinitializer) #9, !dbg !18
  %lds.const.gep1441 = getelementptr i8, ptr addrspace(3) %address.local523, i64 4096, !dbg !18
  %344 = load <4 x half>, ptr addrspace(3) %lds.const.gep1441, align 8, !dbg !18
  %lds.const.gep1442 = getelementptr i8, ptr addrspace(3) %address.local494, i64 128, !dbg !18
  %345 = load <4 x half>, ptr addrspace(3) %lds.const.gep1442, align 8, !dbg !18
  %346 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %344, <4 x float> %343) #9, !dbg !18
  %347 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %345, <4 x float> zeroinitializer) #9, !dbg !18
  %lds.const.gep1443 = getelementptr i8, ptr addrspace(3) %address.local494, i64 4224, !dbg !18
  %348 = load <4 x half>, ptr addrspace(3) %lds.const.gep1443, align 8, !dbg !18
  %lds.const.gep1444 = getelementptr i8, ptr addrspace(3) %address.local523, i64 128, !dbg !18
  %349 = load <4 x half>, ptr addrspace(3) %lds.const.gep1444, align 8, !dbg !18
  %350 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %348, <4 x float> %347) #9, !dbg !18
  %351 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %349, <4 x float> zeroinitializer) #9, !dbg !18
  %lds.const.gep1445 = getelementptr i8, ptr addrspace(3) %address.local523, i64 4224, !dbg !18
  %352 = load <4 x half>, ptr addrspace(3) %lds.const.gep1445, align 8, !dbg !18
  %lds.const.gep1446 = getelementptr i8, ptr addrspace(3) %address.local494, i64 1024, !dbg !18
  %353 = load <4 x half>, ptr addrspace(3) %lds.const.gep1446, align 8, !dbg !18
  %354 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %352, <4 x float> %351) #9, !dbg !18
  %355 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %353, <4 x float> zeroinitializer) #9, !dbg !18
  %lds.const.gep1447 = getelementptr i8, ptr addrspace(3) %address.local494, i64 5120, !dbg !18
  %356 = load <4 x half>, ptr addrspace(3) %lds.const.gep1447, align 8, !dbg !18
  %lds.const.gep1448 = getelementptr i8, ptr addrspace(3) %address.local523, i64 1024, !dbg !18
  %357 = load <4 x half>, ptr addrspace(3) %lds.const.gep1448, align 8, !dbg !18
  %358 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %356, <4 x float> %355) #9, !dbg !18
  %359 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %357, <4 x float> zeroinitializer) #9, !dbg !18
  %lds.const.gep1449 = getelementptr i8, ptr addrspace(3) %address.local523, i64 5120, !dbg !18
  %360 = load <4 x half>, ptr addrspace(3) %lds.const.gep1449, align 8, !dbg !18
  %lds.const.gep1450 = getelementptr i8, ptr addrspace(3) %address.local494, i64 1152, !dbg !18
  %361 = load <4 x half>, ptr addrspace(3) %lds.const.gep1450, align 8, !dbg !18
  %362 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %360, <4 x float> %359) #9, !dbg !18
  %363 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %361, <4 x float> zeroinitializer) #9, !dbg !18
  %lds.const.gep1451 = getelementptr i8, ptr addrspace(3) %address.local494, i64 5248, !dbg !18
  %364 = load <4 x half>, ptr addrspace(3) %lds.const.gep1451, align 8, !dbg !18
  %lds.const.gep1452 = getelementptr i8, ptr addrspace(3) %address.local523, i64 1152, !dbg !18
  %365 = load <4 x half>, ptr addrspace(3) %lds.const.gep1452, align 8, !dbg !18
  %366 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %364, <4 x float> %363) #9, !dbg !18
  %lds.const.gep1453 = getelementptr i8, ptr addrspace(3) %address.local523, i64 5248, !dbg !18
  %367 = load <4 x half>, ptr addrspace(3) %lds.const.gep1453, align 8, !dbg !18
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !18
  %368 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %365, <4 x float> zeroinitializer) #9, !dbg !18
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !18
  %369 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %367, <4 x float> %368) #9, !dbg !18
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %370 = add nuw i64 %205, 2048, !dbg !17
  %371 = icmp slt i64 %370, 0, !dbg !17
  %372 = sub nuw i64 -2049, %205, !dbg !17
  %373 = select i1 %371, i64 %372, i64 %370, !dbg !17
  %374 = sdiv i64 %373, 4096, !dbg !17
  %.lobit38 = ashr i64 %370, 63, !dbg !17
  %375 = xor i64 %374, %.lobit38, !dbg !17
  %376 = add nsw i64 %375, %16, !dbg !17
  %377 = add nsw i64 %261, 64, !dbg !17
  %378 = icmp slt i64 %261, -64, !dbg !17
  %379 = sub nsw i64 -65, %261, !dbg !17
  %380 = select i1 %378, i64 %379, i64 %377, !dbg !17
  %381 = sdiv i64 %380, 128, !dbg !17
  %382 = sext i1 %378 to i64, !dbg !17
  %383 = xor i64 %381, %382, !dbg !17
  %384 = shl nsw i64 %383, 7, !dbg !17
  %385 = sub i64 %195, %384, !dbg !17
  %386 = shl i64 %376, 19, !dbg !17
  %387 = and i64 %386, 16252928, !dbg !17
  %388 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %387, !dbg !17
  %389 = getelementptr [2 x i8], ptr addrspace(1) %388, i64 %216, !dbg !17
  %390 = getelementptr [2 x i8], ptr addrspace(1) %389, i64 %385, !dbg !17
  %391 = getelementptr i8, ptr addrspace(1) %390, i64 128, !dbg !17
  %392 = getelementptr [2 x i8], ptr addrspace(1) %391, i64 %222, !dbg !17
  %393 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local439), !dbg !17
  %394 = zext i32 %393 to i64, !dbg !17
  %395 = inttoptr i64 %394 to ptr addrspace(3), !dbg !17
  %address.local832 = getelementptr [2 x i8], ptr addrspace(1) %392, i64 %address.local449, !dbg !17
  %396 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %397 = ptrtoint ptr addrspace(1) %address.local832 to i64, !dbg !17
  %398 = sub i64 %397, %396, !dbg !17
  %dma.buffer.offset1576 = trunc i64 %398 to i32, !dbg !17
  %399 = trunc i64 %396 to i32, !dbg !17
  %400 = insertelement <4 x i32> poison, i32 %399, i64 0, !dbg !17
  %401 = lshr i64 %396, 32, !dbg !17
  %402 = trunc i64 %401 to i32, !dbg !17
  %403 = insertelement <4 x i32> %400, i32 %402, i64 1, !dbg !17
  %404 = insertelement <4 x i32> %403, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc1577 = insertelement <4 x i32> %404, i32 131072, i64 3, !dbg !17
  %405 = ptrtoint ptr addrspace(3) %395 to i32, !dbg !17
  %dma.buffer.lds1578 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %405), !dbg !17
  %406 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1578, i32 %dma.buffer.offset1576, <4 x i32> %dma.buffer.rsrc1577) #10, !dbg !17
  %407 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local456), !dbg !17
  %408 = zext i32 %407 to i64, !dbg !17
  %409 = inttoptr i64 %408 to ptr addrspace(3), !dbg !17
  %address.local833 = getelementptr [2 x i8], ptr addrspace(1) %392, i64 %address.local466, !dbg !17
  %410 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %411 = ptrtoint ptr addrspace(1) %address.local833 to i64, !dbg !17
  %412 = sub i64 %411, %410, !dbg !17
  %dma.buffer.offset1573 = trunc i64 %412 to i32, !dbg !17
  %413 = trunc i64 %410 to i32, !dbg !17
  %414 = insertelement <4 x i32> poison, i32 %413, i64 0, !dbg !17
  %415 = lshr i64 %410, 32, !dbg !17
  %416 = trunc i64 %415 to i32, !dbg !17
  %417 = insertelement <4 x i32> %414, i32 %416, i64 1, !dbg !17
  %418 = insertelement <4 x i32> %417, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc1574 = insertelement <4 x i32> %418, i32 131072, i64 3, !dbg !17
  %419 = ptrtoint ptr addrspace(3) %409 to i32, !dbg !17
  %dma.buffer.lds1575 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %419), !dbg !17
  %420 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1575, i32 %dma.buffer.offset1573, <4 x i32> %dma.buffer.rsrc1574) #10, !dbg !17
  %lds.const.gep1454 = getelementptr i8, ptr addrspace(3) %address.local494, i64 8192, !dbg !18
  %421 = load <4 x half>, ptr addrspace(3) %lds.const.gep1454, align 8, !dbg !18
  %422 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %421, <4 x float> %310) #9, !dbg !18
  %lds.const.gep1455 = getelementptr i8, ptr addrspace(3) %address.local494, i64 12288, !dbg !18
  %423 = load <4 x half>, ptr addrspace(3) %lds.const.gep1455, align 8, !dbg !18
  %lds.const.gep1456 = getelementptr i8, ptr addrspace(3) %address.local523, i64 8192, !dbg !18
  %424 = load <4 x half>, ptr addrspace(3) %lds.const.gep1456, align 8, !dbg !18
  %425 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %423, <4 x float> %422) #9, !dbg !18
  %426 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %424, <4 x float> %314) #9, !dbg !18
  %lds.const.gep1457 = getelementptr i8, ptr addrspace(3) %address.local523, i64 12288, !dbg !18
  %427 = load <4 x half>, ptr addrspace(3) %lds.const.gep1457, align 8, !dbg !18
  %lds.const.gep1458 = getelementptr i8, ptr addrspace(3) %address.local494, i64 8320, !dbg !18
  %428 = load <4 x half>, ptr addrspace(3) %lds.const.gep1458, align 8, !dbg !18
  %429 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %427, <4 x float> %426) #9, !dbg !18
  %430 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %428, <4 x float> %318) #9, !dbg !18
  %lds.const.gep1459 = getelementptr i8, ptr addrspace(3) %address.local494, i64 12416, !dbg !18
  %431 = load <4 x half>, ptr addrspace(3) %lds.const.gep1459, align 8, !dbg !18
  %lds.const.gep1460 = getelementptr i8, ptr addrspace(3) %address.local523, i64 8320, !dbg !18
  %432 = load <4 x half>, ptr addrspace(3) %lds.const.gep1460, align 8, !dbg !18
  %433 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %431, <4 x float> %430) #9, !dbg !18
  %434 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %432, <4 x float> %322) #9, !dbg !18
  %lds.const.gep1461 = getelementptr i8, ptr addrspace(3) %address.local523, i64 12416, !dbg !18
  %435 = load <4 x half>, ptr addrspace(3) %lds.const.gep1461, align 8, !dbg !18
  %lds.const.gep1462 = getelementptr i8, ptr addrspace(3) %address.local494, i64 9216, !dbg !18
  %436 = load <4 x half>, ptr addrspace(3) %lds.const.gep1462, align 8, !dbg !18
  %437 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %435, <4 x float> %434) #9, !dbg !18
  %438 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %436, <4 x float> %326) #9, !dbg !18
  %lds.const.gep1463 = getelementptr i8, ptr addrspace(3) %address.local494, i64 13312, !dbg !18
  %439 = load <4 x half>, ptr addrspace(3) %lds.const.gep1463, align 8, !dbg !18
  %lds.const.gep1464 = getelementptr i8, ptr addrspace(3) %address.local523, i64 9216, !dbg !18
  %440 = load <4 x half>, ptr addrspace(3) %lds.const.gep1464, align 8, !dbg !18
  %441 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %439, <4 x float> %438) #9, !dbg !18
  %442 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %440, <4 x float> %330) #9, !dbg !18
  %lds.const.gep1465 = getelementptr i8, ptr addrspace(3) %address.local523, i64 13312, !dbg !18
  %443 = load <4 x half>, ptr addrspace(3) %lds.const.gep1465, align 8, !dbg !18
  %lds.const.gep1466 = getelementptr i8, ptr addrspace(3) %address.local494, i64 9344, !dbg !18
  %444 = load <4 x half>, ptr addrspace(3) %lds.const.gep1466, align 8, !dbg !18
  %445 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %443, <4 x float> %442) #9, !dbg !18
  %446 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %444, <4 x float> %334) #9, !dbg !18
  %lds.const.gep1467 = getelementptr i8, ptr addrspace(3) %address.local494, i64 13440, !dbg !18
  %447 = load <4 x half>, ptr addrspace(3) %lds.const.gep1467, align 8, !dbg !18
  %lds.const.gep1468 = getelementptr i8, ptr addrspace(3) %address.local523, i64 9344, !dbg !18
  %448 = load <4 x half>, ptr addrspace(3) %lds.const.gep1468, align 8, !dbg !18
  %449 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %447, <4 x float> %446) #9, !dbg !18
  %450 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %448, <4 x float> %338) #9, !dbg !18
  %lds.const.gep1469 = getelementptr i8, ptr addrspace(3) %address.local523, i64 13440, !dbg !18
  %451 = load <4 x half>, ptr addrspace(3) %lds.const.gep1469, align 8, !dbg !18
  %lds.const.gep1470 = getelementptr i8, ptr addrspace(3) %address.local494, i64 8192, !dbg !18
  %452 = load <4 x half>, ptr addrspace(3) %lds.const.gep1470, align 8, !dbg !18
  %453 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %451, <4 x float> %450) #9, !dbg !18
  %454 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %452, <4 x float> %342) #9, !dbg !18
  %lds.const.gep1471 = getelementptr i8, ptr addrspace(3) %address.local494, i64 12288, !dbg !18
  %455 = load <4 x half>, ptr addrspace(3) %lds.const.gep1471, align 8, !dbg !18
  %lds.const.gep1472 = getelementptr i8, ptr addrspace(3) %address.local523, i64 8192, !dbg !18
  %456 = load <4 x half>, ptr addrspace(3) %lds.const.gep1472, align 8, !dbg !18
  %457 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %455, <4 x float> %454) #9, !dbg !18
  %458 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %456, <4 x float> %346) #9, !dbg !18
  %lds.const.gep1473 = getelementptr i8, ptr addrspace(3) %address.local523, i64 12288, !dbg !18
  %459 = load <4 x half>, ptr addrspace(3) %lds.const.gep1473, align 8, !dbg !18
  %lds.const.gep1474 = getelementptr i8, ptr addrspace(3) %address.local494, i64 8320, !dbg !18
  %460 = load <4 x half>, ptr addrspace(3) %lds.const.gep1474, align 8, !dbg !18
  %461 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %459, <4 x float> %458) #9, !dbg !18
  %462 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %460, <4 x float> %350) #9, !dbg !18
  %lds.const.gep1475 = getelementptr i8, ptr addrspace(3) %address.local494, i64 12416, !dbg !18
  %463 = load <4 x half>, ptr addrspace(3) %lds.const.gep1475, align 8, !dbg !18
  %lds.const.gep1476 = getelementptr i8, ptr addrspace(3) %address.local523, i64 8320, !dbg !18
  %464 = load <4 x half>, ptr addrspace(3) %lds.const.gep1476, align 8, !dbg !18
  %465 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %463, <4 x float> %462) #9, !dbg !18
  %466 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %464, <4 x float> %354) #9, !dbg !18
  %lds.const.gep1477 = getelementptr i8, ptr addrspace(3) %address.local523, i64 12416, !dbg !18
  %467 = load <4 x half>, ptr addrspace(3) %lds.const.gep1477, align 8, !dbg !18
  %lds.const.gep1478 = getelementptr i8, ptr addrspace(3) %address.local494, i64 9216, !dbg !18
  %468 = load <4 x half>, ptr addrspace(3) %lds.const.gep1478, align 8, !dbg !18
  %469 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %467, <4 x float> %466) #9, !dbg !18
  %470 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %468, <4 x float> %358) #9, !dbg !18
  %lds.const.gep1479 = getelementptr i8, ptr addrspace(3) %address.local494, i64 13312, !dbg !18
  %471 = load <4 x half>, ptr addrspace(3) %lds.const.gep1479, align 8, !dbg !18
  %lds.const.gep1480 = getelementptr i8, ptr addrspace(3) %address.local523, i64 9216, !dbg !18
  %472 = load <4 x half>, ptr addrspace(3) %lds.const.gep1480, align 8, !dbg !18
  %473 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %471, <4 x float> %470) #9, !dbg !18
  %474 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %472, <4 x float> %362) #9, !dbg !18
  %lds.const.gep1481 = getelementptr i8, ptr addrspace(3) %address.local523, i64 13312, !dbg !18
  %475 = load <4 x half>, ptr addrspace(3) %lds.const.gep1481, align 8, !dbg !18
  %lds.const.gep1482 = getelementptr i8, ptr addrspace(3) %address.local494, i64 9344, !dbg !18
  %476 = load <4 x half>, ptr addrspace(3) %lds.const.gep1482, align 8, !dbg !18
  %477 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %475, <4 x float> %474) #9, !dbg !18
  %478 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %476, <4 x float> %366) #9, !dbg !18
  %lds.const.gep1483 = getelementptr i8, ptr addrspace(3) %address.local494, i64 13440, !dbg !18
  %479 = load <4 x half>, ptr addrspace(3) %lds.const.gep1483, align 8, !dbg !18
  %lds.const.gep1484 = getelementptr i8, ptr addrspace(3) %address.local523, i64 9344, !dbg !18
  %480 = load <4 x half>, ptr addrspace(3) %lds.const.gep1484, align 8, !dbg !18
  %481 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %479, <4 x float> %478) #9, !dbg !18
  %lds.const.gep1485 = getelementptr i8, ptr addrspace(3) %address.local523, i64 13440, !dbg !18
  %482 = load <4 x half>, ptr addrspace(3) %lds.const.gep1485, align 8, !dbg !18
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !18
  %483 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %480, <4 x float> %369) #9, !dbg !18
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !18
  %484 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %482, <4 x float> %483) #9, !dbg !18
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %485 = add nuw i64 %205, 3072, !dbg !17
  %486 = icmp slt i64 %485, 0, !dbg !17
  %487 = sub nuw i64 -3073, %205, !dbg !17
  %488 = select i1 %486, i64 %487, i64 %485, !dbg !17
  %489 = sdiv i64 %488, 4096, !dbg !17
  %.lobit39 = ashr i64 %485, 63, !dbg !17
  %490 = xor i64 %489, %.lobit39, !dbg !17
  %491 = add nsw i64 %490, %16, !dbg !17
  %492 = add nsw i64 %261, 96, !dbg !17
  %493 = icmp slt i64 %261, -96, !dbg !17
  %494 = sub nsw i64 -97, %261, !dbg !17
  %495 = select i1 %493, i64 %494, i64 %492, !dbg !17
  %496 = sdiv i64 %495, 128, !dbg !17
  %497 = sext i1 %493 to i64, !dbg !17
  %498 = xor i64 %496, %497, !dbg !17
  %499 = shl nsw i64 %498, 7, !dbg !17
  %500 = sub i64 %195, %499, !dbg !17
  %501 = shl i64 %491, 19, !dbg !17
  %502 = and i64 %501, 16252928, !dbg !17
  %503 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %502, !dbg !17
  %504 = getelementptr [2 x i8], ptr addrspace(1) %503, i64 %216, !dbg !17
  %505 = getelementptr [2 x i8], ptr addrspace(1) %504, i64 %500, !dbg !17
  %506 = getelementptr i8, ptr addrspace(1) %505, i64 192, !dbg !17
  %507 = getelementptr [2 x i8], ptr addrspace(1) %506, i64 %222, !dbg !17
  %508 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local470), !dbg !17
  %509 = zext i32 %508 to i64, !dbg !17
  %510 = inttoptr i64 %509 to ptr addrspace(3), !dbg !17
  %address.local850 = getelementptr [2 x i8], ptr addrspace(1) %507, i64 %address.local449, !dbg !17
  %511 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %512 = ptrtoint ptr addrspace(1) %address.local850 to i64, !dbg !17
  %513 = sub i64 %512, %511, !dbg !17
  %dma.buffer.offset1570 = trunc i64 %513 to i32, !dbg !17
  %514 = trunc i64 %511 to i32, !dbg !17
  %515 = insertelement <4 x i32> poison, i32 %514, i64 0, !dbg !17
  %516 = lshr i64 %511, 32, !dbg !17
  %517 = trunc i64 %516 to i32, !dbg !17
  %518 = insertelement <4 x i32> %515, i32 %517, i64 1, !dbg !17
  %519 = insertelement <4 x i32> %518, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc1571 = insertelement <4 x i32> %519, i32 131072, i64 3, !dbg !17
  %520 = ptrtoint ptr addrspace(3) %510 to i32, !dbg !17
  %dma.buffer.lds1572 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %520), !dbg !17
  %521 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1572, i32 %dma.buffer.offset1570, <4 x i32> %dma.buffer.rsrc1571) #10, !dbg !17
  %522 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local474), !dbg !17
  %523 = zext i32 %522 to i64, !dbg !17
  %524 = inttoptr i64 %523 to ptr addrspace(3), !dbg !17
  %address.local851 = getelementptr [2 x i8], ptr addrspace(1) %507, i64 %address.local466, !dbg !17
  %525 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %526 = ptrtoint ptr addrspace(1) %address.local851 to i64, !dbg !17
  %527 = sub i64 %526, %525, !dbg !17
  %dma.buffer.offset1567 = trunc i64 %527 to i32, !dbg !17
  %528 = trunc i64 %525 to i32, !dbg !17
  %529 = insertelement <4 x i32> poison, i32 %528, i64 0, !dbg !17
  %530 = lshr i64 %525, 32, !dbg !17
  %531 = trunc i64 %530 to i32, !dbg !17
  %532 = insertelement <4 x i32> %529, i32 %531, i64 1, !dbg !17
  %533 = insertelement <4 x i32> %532, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc1568 = insertelement <4 x i32> %533, i32 131072, i64 3, !dbg !17
  %534 = ptrtoint ptr addrspace(3) %524 to i32, !dbg !17
  %dma.buffer.lds1569 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %534), !dbg !17
  %535 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1569, i32 %dma.buffer.offset1567, <4 x i32> %dma.buffer.rsrc1568) #10, !dbg !17
  %536 = load <4 x half>, ptr addrspace(3) %address.local494, align 8, !dbg !18
  %537 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %175, <4 x half> %536, <4 x float> %425) #9, !dbg !18
  %lds.const.gep1486 = getelementptr i8, ptr addrspace(3) %address.local494, i64 4096, !dbg !18
  %538 = load <4 x half>, ptr addrspace(3) %lds.const.gep1486, align 8, !dbg !18
  %539 = load <4 x half>, ptr addrspace(3) %address.local523, align 8, !dbg !18
  %540 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %538, <4 x float> %537) #9, !dbg !18
  %541 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %175, <4 x half> %539, <4 x float> %429) #9, !dbg !18
  %lds.const.gep1487 = getelementptr i8, ptr addrspace(3) %address.local523, i64 4096, !dbg !18
  %542 = load <4 x half>, ptr addrspace(3) %lds.const.gep1487, align 8, !dbg !18
  %lds.const.gep1488 = getelementptr i8, ptr addrspace(3) %address.local494, i64 128, !dbg !18
  %543 = load <4 x half>, ptr addrspace(3) %lds.const.gep1488, align 8, !dbg !18
  %544 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %542, <4 x float> %541) #9, !dbg !18
  %545 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %175, <4 x half> %543, <4 x float> %433) #9, !dbg !18
  %lds.const.gep1489 = getelementptr i8, ptr addrspace(3) %address.local494, i64 4224, !dbg !18
  %546 = load <4 x half>, ptr addrspace(3) %lds.const.gep1489, align 8, !dbg !18
  %lds.const.gep1490 = getelementptr i8, ptr addrspace(3) %address.local523, i64 128, !dbg !18
  %547 = load <4 x half>, ptr addrspace(3) %lds.const.gep1490, align 8, !dbg !18
  %548 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %546, <4 x float> %545) #9, !dbg !18
  %549 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %175, <4 x half> %547, <4 x float> %437) #9, !dbg !18
  %lds.const.gep1491 = getelementptr i8, ptr addrspace(3) %address.local523, i64 4224, !dbg !18
  %550 = load <4 x half>, ptr addrspace(3) %lds.const.gep1491, align 8, !dbg !18
  %lds.const.gep1492 = getelementptr i8, ptr addrspace(3) %address.local494, i64 1024, !dbg !18
  %551 = load <4 x half>, ptr addrspace(3) %lds.const.gep1492, align 8, !dbg !18
  %552 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %550, <4 x float> %549) #9, !dbg !18
  %553 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %175, <4 x half> %551, <4 x float> %441) #9, !dbg !18
  %lds.const.gep1493 = getelementptr i8, ptr addrspace(3) %address.local494, i64 5120, !dbg !18
  %554 = load <4 x half>, ptr addrspace(3) %lds.const.gep1493, align 8, !dbg !18
  %lds.const.gep1494 = getelementptr i8, ptr addrspace(3) %address.local523, i64 1024, !dbg !18
  %555 = load <4 x half>, ptr addrspace(3) %lds.const.gep1494, align 8, !dbg !18
  %556 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %554, <4 x float> %553) #9, !dbg !18
  %557 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %175, <4 x half> %555, <4 x float> %445) #9, !dbg !18
  %lds.const.gep1495 = getelementptr i8, ptr addrspace(3) %address.local523, i64 5120, !dbg !18
  %558 = load <4 x half>, ptr addrspace(3) %lds.const.gep1495, align 8, !dbg !18
  %lds.const.gep1496 = getelementptr i8, ptr addrspace(3) %address.local494, i64 1152, !dbg !18
  %559 = load <4 x half>, ptr addrspace(3) %lds.const.gep1496, align 8, !dbg !18
  %560 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %558, <4 x float> %557) #9, !dbg !18
  %561 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %175, <4 x half> %559, <4 x float> %449) #9, !dbg !18
  %lds.const.gep1497 = getelementptr i8, ptr addrspace(3) %address.local494, i64 5248, !dbg !18
  %562 = load <4 x half>, ptr addrspace(3) %lds.const.gep1497, align 8, !dbg !18
  %lds.const.gep1498 = getelementptr i8, ptr addrspace(3) %address.local523, i64 1152, !dbg !18
  %563 = load <4 x half>, ptr addrspace(3) %lds.const.gep1498, align 8, !dbg !18
  %564 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %562, <4 x float> %561) #9, !dbg !18
  %565 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %175, <4 x half> %563, <4 x float> %453) #9, !dbg !18
  %lds.const.gep1499 = getelementptr i8, ptr addrspace(3) %address.local523, i64 5248, !dbg !18
  %566 = load <4 x half>, ptr addrspace(3) %lds.const.gep1499, align 8, !dbg !18
  %567 = load <4 x half>, ptr addrspace(3) %address.local494, align 8, !dbg !18
  %568 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %566, <4 x float> %565) #9, !dbg !18
  %569 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %567, <4 x float> %457) #9, !dbg !18
  %lds.const.gep1500 = getelementptr i8, ptr addrspace(3) %address.local494, i64 4096, !dbg !18
  %570 = load <4 x half>, ptr addrspace(3) %lds.const.gep1500, align 8, !dbg !18
  %571 = load <4 x half>, ptr addrspace(3) %address.local523, align 8, !dbg !18
  %572 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %179, <4 x half> %570, <4 x float> %569) #9, !dbg !18
  %573 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %571, <4 x float> %461) #9, !dbg !18
  %lds.const.gep1501 = getelementptr i8, ptr addrspace(3) %address.local523, i64 4096, !dbg !18
  %574 = load <4 x half>, ptr addrspace(3) %lds.const.gep1501, align 8, !dbg !18
  %lds.const.gep1502 = getelementptr i8, ptr addrspace(3) %address.local494, i64 128, !dbg !18
  %575 = load <4 x half>, ptr addrspace(3) %lds.const.gep1502, align 8, !dbg !18
  %576 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %179, <4 x half> %574, <4 x float> %573) #9, !dbg !18
  %577 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %575, <4 x float> %465) #9, !dbg !18
  %lds.const.gep1503 = getelementptr i8, ptr addrspace(3) %address.local494, i64 4224, !dbg !18
  %578 = load <4 x half>, ptr addrspace(3) %lds.const.gep1503, align 8, !dbg !18
  %lds.const.gep1504 = getelementptr i8, ptr addrspace(3) %address.local523, i64 128, !dbg !18
  %579 = load <4 x half>, ptr addrspace(3) %lds.const.gep1504, align 8, !dbg !18
  %580 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %179, <4 x half> %578, <4 x float> %577) #9, !dbg !18
  %581 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %579, <4 x float> %469) #9, !dbg !18
  %lds.const.gep1505 = getelementptr i8, ptr addrspace(3) %address.local523, i64 4224, !dbg !18
  %582 = load <4 x half>, ptr addrspace(3) %lds.const.gep1505, align 8, !dbg !18
  %lds.const.gep1506 = getelementptr i8, ptr addrspace(3) %address.local494, i64 1024, !dbg !18
  %583 = load <4 x half>, ptr addrspace(3) %lds.const.gep1506, align 8, !dbg !18
  %584 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %179, <4 x half> %582, <4 x float> %581) #9, !dbg !18
  %585 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %583, <4 x float> %473) #9, !dbg !18
  %lds.const.gep1507 = getelementptr i8, ptr addrspace(3) %address.local494, i64 5120, !dbg !18
  %586 = load <4 x half>, ptr addrspace(3) %lds.const.gep1507, align 8, !dbg !18
  %lds.const.gep1508 = getelementptr i8, ptr addrspace(3) %address.local523, i64 1024, !dbg !18
  %587 = load <4 x half>, ptr addrspace(3) %lds.const.gep1508, align 8, !dbg !18
  %588 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %179, <4 x half> %586, <4 x float> %585) #9, !dbg !18
  %589 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %587, <4 x float> %477) #9, !dbg !18
  %lds.const.gep1509 = getelementptr i8, ptr addrspace(3) %address.local523, i64 5120, !dbg !18
  %590 = load <4 x half>, ptr addrspace(3) %lds.const.gep1509, align 8, !dbg !18
  %lds.const.gep1510 = getelementptr i8, ptr addrspace(3) %address.local494, i64 1152, !dbg !18
  %591 = load <4 x half>, ptr addrspace(3) %lds.const.gep1510, align 8, !dbg !18
  %592 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %179, <4 x half> %590, <4 x float> %589) #9, !dbg !18
  %593 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %591, <4 x float> %481) #9, !dbg !18
  %lds.const.gep1511 = getelementptr i8, ptr addrspace(3) %address.local494, i64 5248, !dbg !18
  %594 = load <4 x half>, ptr addrspace(3) %lds.const.gep1511, align 8, !dbg !18
  %lds.const.gep1512 = getelementptr i8, ptr addrspace(3) %address.local523, i64 1152, !dbg !18
  %595 = load <4 x half>, ptr addrspace(3) %lds.const.gep1512, align 8, !dbg !18
  %596 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %179, <4 x half> %594, <4 x float> %593) #9, !dbg !18
  %lds.const.gep1513 = getelementptr i8, ptr addrspace(3) %address.local523, i64 5248, !dbg !18
  %597 = load <4 x half>, ptr addrspace(3) %lds.const.gep1513, align 8, !dbg !18
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !18
  %598 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %595, <4 x float> %484) #9, !dbg !18
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !18
  %599 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %179, <4 x half> %597, <4 x float> %598) #9, !dbg !18
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %600 = add nuw nsw i64 %216, %16, !dbg !19
  %601 = shl i64 %600, 19, !dbg !19
  %602 = and i64 %601, 16252928, !dbg !19
  %603 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %602, !dbg !19
  %604 = getelementptr [2 x i8], ptr addrspace(1) %603, i64 %222, !dbg !19
  %605 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local439), !dbg !19
  %606 = zext i32 %605 to i64, !dbg !19
  %607 = inttoptr i64 %606 to ptr addrspace(3), !dbg !19
  %address.local852 = shl nuw nsw i64 %address.local434, 5, !dbg !19
  %address.local853 = and i64 %address.local852, 8064, !dbg !19
  %address.local854 = shl nuw nsw i64 %address.local434, 3, !dbg !19
  %address.local855 = and i64 %address.local854, 24, !dbg !19
  %address.local856 = or disjoint i64 %address.local853, %address.local855, !dbg !19
  %address.local857 = getelementptr [2 x i8], ptr addrspace(1) %604, i64 %address.local856, !dbg !19
  %608 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !19
  %609 = ptrtoint ptr addrspace(1) %address.local857 to i64, !dbg !19
  %610 = sub i64 %609, %608, !dbg !19
  %dma.buffer.offset1564 = trunc i64 %610 to i32, !dbg !19
  %611 = trunc i64 %608 to i32, !dbg !19
  %612 = insertelement <4 x i32> poison, i32 %611, i64 0, !dbg !19
  %613 = lshr i64 %608, 32, !dbg !19
  %614 = trunc i64 %613 to i32, !dbg !19
  %615 = insertelement <4 x i32> %612, i32 %614, i64 1, !dbg !19
  %616 = insertelement <4 x i32> %615, i32 33554432, i64 2, !dbg !19
  %dma.buffer.rsrc1565 = insertelement <4 x i32> %616, i32 131072, i64 3, !dbg !19
  %617 = ptrtoint ptr addrspace(3) %607 to i32, !dbg !19
  %dma.buffer.lds1566 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %617), !dbg !19
  %618 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1566, i32 %dma.buffer.offset1564, <4 x i32> %dma.buffer.rsrc1565) #10, !dbg !19
  %619 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local456), !dbg !19
  %620 = zext i32 %619 to i64, !dbg !19
  %621 = inttoptr i64 %620 to ptr addrspace(3), !dbg !19
  %address.local858 = shl nuw nsw i64 %address.local451, 5, !dbg !19
  %address.local859 = and i64 %address.local858, 16256, !dbg !19
  %address.local860 = shl nuw nsw i64 %address.local434, 3, !dbg !19
  %address.local861 = and i64 %address.local860, 24, !dbg !19
  %address.local862 = or disjoint i64 %address.local859, %address.local861, !dbg !19
  %address.local863 = getelementptr [2 x i8], ptr addrspace(1) %604, i64 %address.local862, !dbg !19
  %622 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !19
  %623 = ptrtoint ptr addrspace(1) %address.local863 to i64, !dbg !19
  %624 = sub i64 %623, %622, !dbg !19
  %dma.buffer.offset1561 = trunc i64 %624 to i32, !dbg !19
  %625 = trunc i64 %622 to i32, !dbg !19
  %626 = insertelement <4 x i32> poison, i32 %625, i64 0, !dbg !19
  %627 = lshr i64 %622, 32, !dbg !19
  %628 = trunc i64 %627 to i32, !dbg !19
  %629 = insertelement <4 x i32> %626, i32 %628, i64 1, !dbg !19
  %630 = insertelement <4 x i32> %629, i32 33554432, i64 2, !dbg !19
  %dma.buffer.rsrc1562 = insertelement <4 x i32> %630, i32 131072, i64 3, !dbg !19
  %631 = ptrtoint ptr addrspace(3) %621 to i32, !dbg !19
  %dma.buffer.lds1563 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %631), !dbg !19
  %632 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1563, i32 %dma.buffer.offset1561, <4 x i32> %dma.buffer.rsrc1562) #10, !dbg !19
  %lds.const.gep1514 = getelementptr i8, ptr addrspace(3) %address.local494, i64 8192, !dbg !18
  %633 = load <4 x half>, ptr addrspace(3) %lds.const.gep1514, align 8, !dbg !18
  %634 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %633, <4 x float> %540) #9, !dbg !18
  %lds.const.gep1515 = getelementptr i8, ptr addrspace(3) %address.local494, i64 12288, !dbg !18
  %635 = load <4 x half>, ptr addrspace(3) %lds.const.gep1515, align 8, !dbg !18
  %lds.const.gep1516 = getelementptr i8, ptr addrspace(3) %address.local523, i64 8192, !dbg !18
  %636 = load <4 x half>, ptr addrspace(3) %lds.const.gep1516, align 8, !dbg !18
  %637 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %635, <4 x float> %634) #9, !dbg !18
  %638 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %636, <4 x float> %544) #9, !dbg !18
  %lds.const.gep1517 = getelementptr i8, ptr addrspace(3) %address.local523, i64 12288, !dbg !18
  %639 = load <4 x half>, ptr addrspace(3) %lds.const.gep1517, align 8, !dbg !18
  %lds.const.gep1518 = getelementptr i8, ptr addrspace(3) %address.local494, i64 8320, !dbg !18
  %640 = load <4 x half>, ptr addrspace(3) %lds.const.gep1518, align 8, !dbg !18
  %641 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %639, <4 x float> %638) #9, !dbg !18
  %642 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %640, <4 x float> %548) #9, !dbg !18
  %lds.const.gep1519 = getelementptr i8, ptr addrspace(3) %address.local494, i64 12416, !dbg !18
  %643 = load <4 x half>, ptr addrspace(3) %lds.const.gep1519, align 8, !dbg !18
  %lds.const.gep1520 = getelementptr i8, ptr addrspace(3) %address.local523, i64 8320, !dbg !18
  %644 = load <4 x half>, ptr addrspace(3) %lds.const.gep1520, align 8, !dbg !18
  %645 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %643, <4 x float> %642) #9, !dbg !18
  %646 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %644, <4 x float> %552) #9, !dbg !18
  %lds.const.gep1521 = getelementptr i8, ptr addrspace(3) %address.local523, i64 12416, !dbg !18
  %647 = load <4 x half>, ptr addrspace(3) %lds.const.gep1521, align 8, !dbg !18
  %lds.const.gep1522 = getelementptr i8, ptr addrspace(3) %address.local494, i64 9216, !dbg !18
  %648 = load <4 x half>, ptr addrspace(3) %lds.const.gep1522, align 8, !dbg !18
  %649 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %647, <4 x float> %646) #9, !dbg !18
  %650 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %648, <4 x float> %556) #9, !dbg !18
  %lds.const.gep1523 = getelementptr i8, ptr addrspace(3) %address.local494, i64 13312, !dbg !18
  %651 = load <4 x half>, ptr addrspace(3) %lds.const.gep1523, align 8, !dbg !18
  %lds.const.gep1524 = getelementptr i8, ptr addrspace(3) %address.local523, i64 9216, !dbg !18
  %652 = load <4 x half>, ptr addrspace(3) %lds.const.gep1524, align 8, !dbg !18
  %653 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %651, <4 x float> %650) #9, !dbg !18
  %654 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %652, <4 x float> %560) #9, !dbg !18
  %lds.const.gep1525 = getelementptr i8, ptr addrspace(3) %address.local523, i64 13312, !dbg !18
  %655 = load <4 x half>, ptr addrspace(3) %lds.const.gep1525, align 8, !dbg !18
  %lds.const.gep1526 = getelementptr i8, ptr addrspace(3) %address.local494, i64 9344, !dbg !18
  %656 = load <4 x half>, ptr addrspace(3) %lds.const.gep1526, align 8, !dbg !18
  %657 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %655, <4 x float> %654) #9, !dbg !18
  %658 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %656, <4 x float> %564) #9, !dbg !18
  %lds.const.gep1527 = getelementptr i8, ptr addrspace(3) %address.local494, i64 13440, !dbg !18
  %659 = load <4 x half>, ptr addrspace(3) %lds.const.gep1527, align 8, !dbg !18
  %lds.const.gep1528 = getelementptr i8, ptr addrspace(3) %address.local523, i64 9344, !dbg !18
  %660 = load <4 x half>, ptr addrspace(3) %lds.const.gep1528, align 8, !dbg !18
  %661 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %659, <4 x float> %658) #9, !dbg !18
  %662 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %660, <4 x float> %568) #9, !dbg !18
  %lds.const.gep1529 = getelementptr i8, ptr addrspace(3) %address.local523, i64 13440, !dbg !18
  %663 = load <4 x half>, ptr addrspace(3) %lds.const.gep1529, align 8, !dbg !18
  %lds.const.gep1530 = getelementptr i8, ptr addrspace(3) %address.local494, i64 8192, !dbg !18
  %664 = load <4 x half>, ptr addrspace(3) %lds.const.gep1530, align 8, !dbg !18
  %665 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %663, <4 x float> %662) #9, !dbg !18
  %666 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %664, <4 x float> %572) #9, !dbg !18
  %lds.const.gep1531 = getelementptr i8, ptr addrspace(3) %address.local494, i64 12288, !dbg !18
  %667 = load <4 x half>, ptr addrspace(3) %lds.const.gep1531, align 8, !dbg !18
  %lds.const.gep1532 = getelementptr i8, ptr addrspace(3) %address.local523, i64 8192, !dbg !18
  %668 = load <4 x half>, ptr addrspace(3) %lds.const.gep1532, align 8, !dbg !18
  %669 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %187, <4 x half> %667, <4 x float> %666) #9, !dbg !18
  %670 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %668, <4 x float> %576) #9, !dbg !18
  %lds.const.gep1533 = getelementptr i8, ptr addrspace(3) %address.local523, i64 12288, !dbg !18
  %671 = load <4 x half>, ptr addrspace(3) %lds.const.gep1533, align 8, !dbg !18
  %lds.const.gep1534 = getelementptr i8, ptr addrspace(3) %address.local494, i64 8320, !dbg !18
  %672 = load <4 x half>, ptr addrspace(3) %lds.const.gep1534, align 8, !dbg !18
  %673 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %187, <4 x half> %671, <4 x float> %670) #9, !dbg !18
  %674 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %672, <4 x float> %580) #9, !dbg !18
  %lds.const.gep1535 = getelementptr i8, ptr addrspace(3) %address.local494, i64 12416, !dbg !18
  %675 = load <4 x half>, ptr addrspace(3) %lds.const.gep1535, align 8, !dbg !18
  %lds.const.gep1536 = getelementptr i8, ptr addrspace(3) %address.local523, i64 8320, !dbg !18
  %676 = load <4 x half>, ptr addrspace(3) %lds.const.gep1536, align 8, !dbg !18
  %677 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %187, <4 x half> %675, <4 x float> %674) #9, !dbg !18
  %678 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %676, <4 x float> %584) #9, !dbg !18
  %lds.const.gep1537 = getelementptr i8, ptr addrspace(3) %address.local523, i64 12416, !dbg !18
  %679 = load <4 x half>, ptr addrspace(3) %lds.const.gep1537, align 8, !dbg !18
  %lds.const.gep1538 = getelementptr i8, ptr addrspace(3) %address.local494, i64 9216, !dbg !18
  %680 = load <4 x half>, ptr addrspace(3) %lds.const.gep1538, align 8, !dbg !18
  %681 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %187, <4 x half> %679, <4 x float> %678) #9, !dbg !18
  %682 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %680, <4 x float> %588) #9, !dbg !18
  %lds.const.gep1539 = getelementptr i8, ptr addrspace(3) %address.local494, i64 13312, !dbg !18
  %683 = load <4 x half>, ptr addrspace(3) %lds.const.gep1539, align 8, !dbg !18
  %lds.const.gep1540 = getelementptr i8, ptr addrspace(3) %address.local523, i64 9216, !dbg !18
  %684 = load <4 x half>, ptr addrspace(3) %lds.const.gep1540, align 8, !dbg !18
  %685 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %187, <4 x half> %683, <4 x float> %682) #9, !dbg !18
  %686 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %684, <4 x float> %592) #9, !dbg !18
  %lds.const.gep1541 = getelementptr i8, ptr addrspace(3) %address.local523, i64 13312, !dbg !18
  %687 = load <4 x half>, ptr addrspace(3) %lds.const.gep1541, align 8, !dbg !18
  %lds.const.gep1542 = getelementptr i8, ptr addrspace(3) %address.local494, i64 9344, !dbg !18
  %688 = load <4 x half>, ptr addrspace(3) %lds.const.gep1542, align 8, !dbg !18
  %689 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %187, <4 x half> %687, <4 x float> %686) #9, !dbg !18
  %690 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %688, <4 x float> %596) #9, !dbg !18
  %lds.const.gep1543 = getelementptr i8, ptr addrspace(3) %address.local494, i64 13440, !dbg !18
  %691 = load <4 x half>, ptr addrspace(3) %lds.const.gep1543, align 8, !dbg !18
  %lds.const.gep1544 = getelementptr i8, ptr addrspace(3) %address.local523, i64 9344, !dbg !18
  %692 = load <4 x half>, ptr addrspace(3) %lds.const.gep1544, align 8, !dbg !18
  %693 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %187, <4 x half> %691, <4 x float> %690) #9, !dbg !18
  %lds.const.gep1545 = getelementptr i8, ptr addrspace(3) %address.local523, i64 13440, !dbg !18
  %694 = load <4 x half>, ptr addrspace(3) %lds.const.gep1545, align 8, !dbg !18
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !18
  %695 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %692, <4 x float> %599) #9, !dbg !18
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %696 = or disjoint i64 %213, 127, !dbg !14
  %697 = icmp uge i64 %64, %696, !dbg !14
  %698 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %187, <4 x half> %694, <4 x float> %695) #9, !dbg !18
  %699 = and i1 %190, %697, !dbg !14
  br i1 %699, label %700, label %701, !dbg !14

700:                                              ; preds = %212
  br label %845, !dbg !14

701:                                              ; preds = %212
  %702 = add i64 %64, %32, !dbg !14
  %703 = shl nuw nsw i64 %65, 4, !dbg !14
  %704 = add i64 %702, %703, !dbg !14
  %705 = and i32 %31, 240, !dbg !14
  %index.compact = zext nneg i32 %705 to i64, !dbg !14
  %706 = sub i64 %704, %index.compact, !dbg !14
  %707 = and i64 %32, 48, !dbg !14
  %708 = or disjoint i64 %213, %707, !dbg !14
  %709 = add i64 %706, 1, !dbg !24
  %710 = lshr i64 %709, 32, !dbg !25
  %compare.high = trunc i64 %710 to i32, !dbg !25
  %711 = lshr i64 %708, 32, !dbg !25
  %compare.high1157 = trunc i64 %711 to i32, !dbg !25
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high1157, !dbg !25
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high1157, !dbg !25
  %compare.low.base = trunc i64 %709 to i32, !dbg !25
  %compare.low.base1158 = trunc i64 %708 to i32, !dbg !25
  %compare.low = icmp ugt i32 %compare.low.base, %compare.low.base1158, !dbg !25
  %compare.result = select i1 %compare.high.equal, i1 %compare.low, i1 %compare.high.order, !dbg !25
  %712 = select i1 %compare.result, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 1) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local = or i32 %compare.local.base, 1, !dbg !25
  %compare.low1159 = icmp ugt i32 %compare.low.base, %compare.local, !dbg !25
  %compare.result1160 = select i1 %compare.high.equal, i1 %compare.low1159, i1 %compare.high.order, !dbg !25
  %713 = select i1 %compare.result1160, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1161 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 2) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1162 = or i32 %compare.local.base1161, 2, !dbg !25
  %compare.low1163 = icmp ugt i32 %compare.low.base, %compare.local1162, !dbg !25
  %compare.result1164 = select i1 %compare.high.equal, i1 %compare.low1163, i1 %compare.high.order, !dbg !25
  %714 = select i1 %compare.result1164, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1165 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 3) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1166 = or i32 %compare.local.base1165, 3, !dbg !25
  %compare.low1167 = icmp ugt i32 %compare.low.base, %compare.local1166, !dbg !25
  %compare.result1168 = select i1 %compare.high.equal, i1 %compare.low1167, i1 %compare.high.order, !dbg !25
  %715 = select i1 %compare.result1168, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1169 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 4) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1170 = or i32 %compare.local.base1169, 4, !dbg !25
  %compare.low1171 = icmp ugt i32 %compare.low.base, %compare.local1170, !dbg !25
  %compare.result1172 = select i1 %compare.high.equal, i1 %compare.low1171, i1 %compare.high.order, !dbg !25
  %716 = select i1 %compare.result1172, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1173 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 5) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1174 = or i32 %compare.local.base1173, 5, !dbg !25
  %compare.low1175 = icmp ugt i32 %compare.low.base, %compare.local1174, !dbg !25
  %compare.result1176 = select i1 %compare.high.equal, i1 %compare.low1175, i1 %compare.high.order, !dbg !25
  %717 = select i1 %compare.result1176, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1177 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 6) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1178 = or i32 %compare.local.base1177, 6, !dbg !25
  %compare.low1179 = icmp ugt i32 %compare.low.base, %compare.local1178, !dbg !25
  %compare.result1180 = select i1 %compare.high.equal, i1 %compare.low1179, i1 %compare.high.order, !dbg !25
  %718 = select i1 %compare.result1180, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1181 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 7) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1182 = or i32 %compare.local.base1181, 7, !dbg !25
  %compare.low1183 = icmp ugt i32 %compare.low.base, %compare.local1182, !dbg !25
  %compare.result1184 = select i1 %compare.high.equal, i1 %compare.low1183, i1 %compare.high.order, !dbg !25
  %719 = select i1 %compare.result1184, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1185 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 8) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1186 = or i32 %compare.local.base1185, 8, !dbg !25
  %compare.low1187 = icmp ugt i32 %compare.low.base, %compare.local1186, !dbg !25
  %compare.result1188 = select i1 %compare.high.equal, i1 %compare.low1187, i1 %compare.high.order, !dbg !25
  %720 = select i1 %compare.result1188, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1189 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 9) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1190 = or i32 %compare.local.base1189, 9, !dbg !25
  %compare.low1191 = icmp ugt i32 %compare.low.base, %compare.local1190, !dbg !25
  %compare.result1192 = select i1 %compare.high.equal, i1 %compare.low1191, i1 %compare.high.order, !dbg !25
  %721 = select i1 %compare.result1192, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1193 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 10) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1194 = or i32 %compare.local.base1193, 10, !dbg !25
  %compare.low1195 = icmp ugt i32 %compare.low.base, %compare.local1194, !dbg !25
  %compare.result1196 = select i1 %compare.high.equal, i1 %compare.low1195, i1 %compare.high.order, !dbg !25
  %722 = select i1 %compare.result1196, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1197 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 11) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1198 = or i32 %compare.local.base1197, 11, !dbg !25
  %compare.low1199 = icmp ugt i32 %compare.low.base, %compare.local1198, !dbg !25
  %compare.result1200 = select i1 %compare.high.equal, i1 %compare.low1199, i1 %compare.high.order, !dbg !25
  %723 = select i1 %compare.result1200, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1201 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 12) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1202 = or i32 %compare.local.base1201, 12, !dbg !25
  %compare.low1203 = icmp ugt i32 %compare.low.base, %compare.local1202, !dbg !25
  %compare.result1204 = select i1 %compare.high.equal, i1 %compare.low1203, i1 %compare.high.order, !dbg !25
  %724 = select i1 %compare.result1204, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1205 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 13) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1206 = or i32 %compare.local.base1205, 13, !dbg !25
  %compare.low1207 = icmp ugt i32 %compare.low.base, %compare.local1206, !dbg !25
  %compare.result1208 = select i1 %compare.high.equal, i1 %compare.low1207, i1 %compare.high.order, !dbg !25
  %725 = select i1 %compare.result1208, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1209 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 14) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1210 = or i32 %compare.local.base1209, 14, !dbg !25
  %compare.low1211 = icmp ugt i32 %compare.low.base, %compare.local1210, !dbg !25
  %compare.result1212 = select i1 %compare.high.equal, i1 %compare.low1211, i1 %compare.high.order, !dbg !25
  %726 = select i1 %compare.result1212, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1213 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 15) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1214 = or i32 %compare.local.base1213, 15, !dbg !25
  %compare.low1215 = icmp ugt i32 %compare.low.base, %compare.local1214, !dbg !25
  %compare.result1216 = select i1 %compare.high.equal, i1 %compare.low1215, i1 %compare.high.order, !dbg !25
  %727 = select i1 %compare.result1216, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1217 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 16) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1218 = or i32 %compare.local.base1217, 64, !dbg !25
  %compare.low1219 = icmp ugt i32 %compare.low.base, %compare.local1218, !dbg !25
  %compare.result1220 = select i1 %compare.high.equal, i1 %compare.low1219, i1 %compare.high.order, !dbg !25
  %728 = select i1 %compare.result1220, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1221 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 17) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1222 = or i32 %compare.local.base1221, 65, !dbg !25
  %compare.low1223 = icmp ugt i32 %compare.low.base, %compare.local1222, !dbg !25
  %compare.result1224 = select i1 %compare.high.equal, i1 %compare.low1223, i1 %compare.high.order, !dbg !25
  %729 = select i1 %compare.result1224, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1225 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 18) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1226 = or i32 %compare.local.base1225, 66, !dbg !25
  %compare.low1227 = icmp ugt i32 %compare.low.base, %compare.local1226, !dbg !25
  %compare.result1228 = select i1 %compare.high.equal, i1 %compare.low1227, i1 %compare.high.order, !dbg !25
  %730 = select i1 %compare.result1228, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1229 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 19) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1230 = or i32 %compare.local.base1229, 67, !dbg !25
  %compare.low1231 = icmp ugt i32 %compare.low.base, %compare.local1230, !dbg !25
  %compare.result1232 = select i1 %compare.high.equal, i1 %compare.low1231, i1 %compare.high.order, !dbg !25
  %731 = select i1 %compare.result1232, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1233 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 20) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1234 = or i32 %compare.local.base1233, 68, !dbg !25
  %compare.low1235 = icmp ugt i32 %compare.low.base, %compare.local1234, !dbg !25
  %compare.result1236 = select i1 %compare.high.equal, i1 %compare.low1235, i1 %compare.high.order, !dbg !25
  %732 = select i1 %compare.result1236, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1237 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 21) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1238 = or i32 %compare.local.base1237, 69, !dbg !25
  %compare.low1239 = icmp ugt i32 %compare.low.base, %compare.local1238, !dbg !25
  %compare.result1240 = select i1 %compare.high.equal, i1 %compare.low1239, i1 %compare.high.order, !dbg !25
  %733 = select i1 %compare.result1240, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1241 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 22) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1242 = or i32 %compare.local.base1241, 70, !dbg !25
  %compare.low1243 = icmp ugt i32 %compare.low.base, %compare.local1242, !dbg !25
  %compare.result1244 = select i1 %compare.high.equal, i1 %compare.low1243, i1 %compare.high.order, !dbg !25
  %734 = select i1 %compare.result1244, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1245 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 23) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1246 = or i32 %compare.local.base1245, 71, !dbg !25
  %compare.low1247 = icmp ugt i32 %compare.low.base, %compare.local1246, !dbg !25
  %compare.result1248 = select i1 %compare.high.equal, i1 %compare.low1247, i1 %compare.high.order, !dbg !25
  %735 = select i1 %compare.result1248, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1249 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 24) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1250 = or i32 %compare.local.base1249, 72, !dbg !25
  %compare.low1251 = icmp ugt i32 %compare.low.base, %compare.local1250, !dbg !25
  %compare.result1252 = select i1 %compare.high.equal, i1 %compare.low1251, i1 %compare.high.order, !dbg !25
  %736 = select i1 %compare.result1252, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1253 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 25) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1254 = or i32 %compare.local.base1253, 73, !dbg !25
  %compare.low1255 = icmp ugt i32 %compare.low.base, %compare.local1254, !dbg !25
  %compare.result1256 = select i1 %compare.high.equal, i1 %compare.low1255, i1 %compare.high.order, !dbg !25
  %737 = select i1 %compare.result1256, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1257 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 26) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1258 = or i32 %compare.local.base1257, 74, !dbg !25
  %compare.low1259 = icmp ugt i32 %compare.low.base, %compare.local1258, !dbg !25
  %compare.result1260 = select i1 %compare.high.equal, i1 %compare.low1259, i1 %compare.high.order, !dbg !25
  %738 = select i1 %compare.result1260, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1261 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 27) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1262 = or i32 %compare.local.base1261, 75, !dbg !25
  %compare.low1263 = icmp ugt i32 %compare.low.base, %compare.local1262, !dbg !25
  %compare.result1264 = select i1 %compare.high.equal, i1 %compare.low1263, i1 %compare.high.order, !dbg !25
  %739 = select i1 %compare.result1264, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1265 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 28) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1266 = or i32 %compare.local.base1265, 76, !dbg !25
  %compare.low1267 = icmp ugt i32 %compare.low.base, %compare.local1266, !dbg !25
  %compare.result1268 = select i1 %compare.high.equal, i1 %compare.low1267, i1 %compare.high.order, !dbg !25
  %740 = select i1 %compare.result1268, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1269 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 29) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1270 = or i32 %compare.local.base1269, 77, !dbg !25
  %compare.low1271 = icmp ugt i32 %compare.low.base, %compare.local1270, !dbg !25
  %compare.result1272 = select i1 %compare.high.equal, i1 %compare.low1271, i1 %compare.high.order, !dbg !25
  %741 = select i1 %compare.result1272, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1273 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 30) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1274 = or i32 %compare.local.base1273, 78, !dbg !25
  %compare.low1275 = icmp ugt i32 %compare.low.base, %compare.local1274, !dbg !25
  %compare.result1276 = select i1 %compare.high.equal, i1 %compare.low1275, i1 %compare.high.order, !dbg !25
  %742 = select i1 %compare.result1276, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1277 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 31) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1278 = or i32 %compare.local.base1277, 79, !dbg !25
  %compare.low1279 = icmp ugt i32 %compare.low.base, %compare.local1278, !dbg !25
  %compare.result1280 = select i1 %compare.high.equal, i1 %compare.low1279, i1 %compare.high.order, !dbg !25
  %743 = select i1 %compare.result1280, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %744 = add i64 %706, 65, !dbg !24
  %745 = lshr i64 %744, 32, !dbg !25
  %compare.high1281 = trunc i64 %745 to i32, !dbg !25
  %746 = lshr i64 %708, 32, !dbg !25
  %compare.high1282 = trunc i64 %746 to i32, !dbg !25
  %compare.high.equal1283 = icmp eq i32 %compare.high1281, %compare.high1282, !dbg !25
  %compare.high.order1284 = icmp ugt i32 %compare.high1281, %compare.high1282, !dbg !25
  %compare.low.base1285 = trunc i64 %744 to i32, !dbg !25
  %compare.low.base1286 = trunc i64 %708 to i32, !dbg !25
  %compare.low1287 = icmp ugt i32 %compare.low.base1285, %compare.low.base1286, !dbg !25
  %compare.result1288 = select i1 %compare.high.equal1283, i1 %compare.low1287, i1 %compare.high.order1284, !dbg !25
  %747 = select i1 %compare.result1288, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1289 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 32) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1290 = or i32 %compare.local.base1289, 1, !dbg !25
  %compare.low1291 = icmp ugt i32 %compare.low.base1285, %compare.local1290, !dbg !25
  %compare.result1292 = select i1 %compare.high.equal1283, i1 %compare.low1291, i1 %compare.high.order1284, !dbg !25
  %748 = select i1 %compare.result1292, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1293 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 33) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1294 = or i32 %compare.local.base1293, 2, !dbg !25
  %compare.low1295 = icmp ugt i32 %compare.low.base1285, %compare.local1294, !dbg !25
  %compare.result1296 = select i1 %compare.high.equal1283, i1 %compare.low1295, i1 %compare.high.order1284, !dbg !25
  %749 = select i1 %compare.result1296, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1297 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 34) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1298 = or i32 %compare.local.base1297, 3, !dbg !25
  %compare.low1299 = icmp ugt i32 %compare.low.base1285, %compare.local1298, !dbg !25
  %compare.result1300 = select i1 %compare.high.equal1283, i1 %compare.low1299, i1 %compare.high.order1284, !dbg !25
  %750 = select i1 %compare.result1300, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1301 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 35) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1302 = or i32 %compare.local.base1301, 4, !dbg !25
  %compare.low1303 = icmp ugt i32 %compare.low.base1285, %compare.local1302, !dbg !25
  %compare.result1304 = select i1 %compare.high.equal1283, i1 %compare.low1303, i1 %compare.high.order1284, !dbg !25
  %751 = select i1 %compare.result1304, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1305 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 36) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1306 = or i32 %compare.local.base1305, 5, !dbg !25
  %compare.low1307 = icmp ugt i32 %compare.low.base1285, %compare.local1306, !dbg !25
  %compare.result1308 = select i1 %compare.high.equal1283, i1 %compare.low1307, i1 %compare.high.order1284, !dbg !25
  %752 = select i1 %compare.result1308, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1309 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 37) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1310 = or i32 %compare.local.base1309, 6, !dbg !25
  %compare.low1311 = icmp ugt i32 %compare.low.base1285, %compare.local1310, !dbg !25
  %compare.result1312 = select i1 %compare.high.equal1283, i1 %compare.low1311, i1 %compare.high.order1284, !dbg !25
  %753 = select i1 %compare.result1312, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1313 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 38) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1314 = or i32 %compare.local.base1313, 7, !dbg !25
  %compare.low1315 = icmp ugt i32 %compare.low.base1285, %compare.local1314, !dbg !25
  %compare.result1316 = select i1 %compare.high.equal1283, i1 %compare.low1315, i1 %compare.high.order1284, !dbg !25
  %754 = select i1 %compare.result1316, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1317 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 39) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1318 = or i32 %compare.local.base1317, 8, !dbg !25
  %compare.low1319 = icmp ugt i32 %compare.low.base1285, %compare.local1318, !dbg !25
  %compare.result1320 = select i1 %compare.high.equal1283, i1 %compare.low1319, i1 %compare.high.order1284, !dbg !25
  %755 = select i1 %compare.result1320, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1321 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 40) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1322 = or i32 %compare.local.base1321, 9, !dbg !25
  %compare.low1323 = icmp ugt i32 %compare.low.base1285, %compare.local1322, !dbg !25
  %compare.result1324 = select i1 %compare.high.equal1283, i1 %compare.low1323, i1 %compare.high.order1284, !dbg !25
  %756 = select i1 %compare.result1324, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1325 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 41) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1326 = or i32 %compare.local.base1325, 10, !dbg !25
  %compare.low1327 = icmp ugt i32 %compare.low.base1285, %compare.local1326, !dbg !25
  %compare.result1328 = select i1 %compare.high.equal1283, i1 %compare.low1327, i1 %compare.high.order1284, !dbg !25
  %757 = select i1 %compare.result1328, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1329 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 42) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1330 = or i32 %compare.local.base1329, 11, !dbg !25
  %compare.low1331 = icmp ugt i32 %compare.low.base1285, %compare.local1330, !dbg !25
  %compare.result1332 = select i1 %compare.high.equal1283, i1 %compare.low1331, i1 %compare.high.order1284, !dbg !25
  %758 = select i1 %compare.result1332, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1333 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 43) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1334 = or i32 %compare.local.base1333, 12, !dbg !25
  %compare.low1335 = icmp ugt i32 %compare.low.base1285, %compare.local1334, !dbg !25
  %compare.result1336 = select i1 %compare.high.equal1283, i1 %compare.low1335, i1 %compare.high.order1284, !dbg !25
  %759 = select i1 %compare.result1336, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1337 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 44) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1338 = or i32 %compare.local.base1337, 13, !dbg !25
  %compare.low1339 = icmp ugt i32 %compare.low.base1285, %compare.local1338, !dbg !25
  %compare.result1340 = select i1 %compare.high.equal1283, i1 %compare.low1339, i1 %compare.high.order1284, !dbg !25
  %760 = select i1 %compare.result1340, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1341 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 45) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1342 = or i32 %compare.local.base1341, 14, !dbg !25
  %compare.low1343 = icmp ugt i32 %compare.low.base1285, %compare.local1342, !dbg !25
  %compare.result1344 = select i1 %compare.high.equal1283, i1 %compare.low1343, i1 %compare.high.order1284, !dbg !25
  %761 = select i1 %compare.result1344, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1345 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 46) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1346 = or i32 %compare.local.base1345, 15, !dbg !25
  %compare.low1347 = icmp ugt i32 %compare.low.base1285, %compare.local1346, !dbg !25
  %compare.result1348 = select i1 %compare.high.equal1283, i1 %compare.low1347, i1 %compare.high.order1284, !dbg !25
  %762 = select i1 %compare.result1348, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1349 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 47) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1350 = or i32 %compare.local.base1349, 64, !dbg !25
  %compare.low1351 = icmp ugt i32 %compare.low.base1285, %compare.local1350, !dbg !25
  %compare.result1352 = select i1 %compare.high.equal1283, i1 %compare.low1351, i1 %compare.high.order1284, !dbg !25
  %763 = select i1 %compare.result1352, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1353 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 48) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1354 = or i32 %compare.local.base1353, 65, !dbg !25
  %compare.low1355 = icmp ugt i32 %compare.low.base1285, %compare.local1354, !dbg !25
  %compare.result1356 = select i1 %compare.high.equal1283, i1 %compare.low1355, i1 %compare.high.order1284, !dbg !25
  %764 = select i1 %compare.result1356, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1357 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 49) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1358 = or i32 %compare.local.base1357, 66, !dbg !25
  %compare.low1359 = icmp ugt i32 %compare.low.base1285, %compare.local1358, !dbg !25
  %compare.result1360 = select i1 %compare.high.equal1283, i1 %compare.low1359, i1 %compare.high.order1284, !dbg !25
  %765 = select i1 %compare.result1360, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1361 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 50) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1362 = or i32 %compare.local.base1361, 67, !dbg !25
  %compare.low1363 = icmp ugt i32 %compare.low.base1285, %compare.local1362, !dbg !25
  %compare.result1364 = select i1 %compare.high.equal1283, i1 %compare.low1363, i1 %compare.high.order1284, !dbg !25
  %766 = select i1 %compare.result1364, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1365 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 51) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1366 = or i32 %compare.local.base1365, 68, !dbg !25
  %compare.low1367 = icmp ugt i32 %compare.low.base1285, %compare.local1366, !dbg !25
  %compare.result1368 = select i1 %compare.high.equal1283, i1 %compare.low1367, i1 %compare.high.order1284, !dbg !25
  %767 = select i1 %compare.result1368, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1369 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 52) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1370 = or i32 %compare.local.base1369, 69, !dbg !25
  %compare.low1371 = icmp ugt i32 %compare.low.base1285, %compare.local1370, !dbg !25
  %compare.result1372 = select i1 %compare.high.equal1283, i1 %compare.low1371, i1 %compare.high.order1284, !dbg !25
  %768 = select i1 %compare.result1372, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1373 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 53) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1374 = or i32 %compare.local.base1373, 70, !dbg !25
  %compare.low1375 = icmp ugt i32 %compare.low.base1285, %compare.local1374, !dbg !25
  %compare.result1376 = select i1 %compare.high.equal1283, i1 %compare.low1375, i1 %compare.high.order1284, !dbg !25
  %769 = select i1 %compare.result1376, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1377 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 54) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1378 = or i32 %compare.local.base1377, 71, !dbg !25
  %compare.low1379 = icmp ugt i32 %compare.low.base1285, %compare.local1378, !dbg !25
  %compare.result1380 = select i1 %compare.high.equal1283, i1 %compare.low1379, i1 %compare.high.order1284, !dbg !25
  %770 = select i1 %compare.result1380, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1381 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 55) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1382 = or i32 %compare.local.base1381, 72, !dbg !25
  %compare.low1383 = icmp ugt i32 %compare.low.base1285, %compare.local1382, !dbg !25
  %compare.result1384 = select i1 %compare.high.equal1283, i1 %compare.low1383, i1 %compare.high.order1284, !dbg !25
  %771 = select i1 %compare.result1384, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1385 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 56) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1386 = or i32 %compare.local.base1385, 73, !dbg !25
  %compare.low1387 = icmp ugt i32 %compare.low.base1285, %compare.local1386, !dbg !25
  %compare.result1388 = select i1 %compare.high.equal1283, i1 %compare.low1387, i1 %compare.high.order1284, !dbg !25
  %772 = select i1 %compare.result1388, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1389 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 57) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1390 = or i32 %compare.local.base1389, 74, !dbg !25
  %compare.low1391 = icmp ugt i32 %compare.low.base1285, %compare.local1390, !dbg !25
  %compare.result1392 = select i1 %compare.high.equal1283, i1 %compare.low1391, i1 %compare.high.order1284, !dbg !25
  %773 = select i1 %compare.result1392, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1393 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 58) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1394 = or i32 %compare.local.base1393, 75, !dbg !25
  %compare.low1395 = icmp ugt i32 %compare.low.base1285, %compare.local1394, !dbg !25
  %compare.result1396 = select i1 %compare.high.equal1283, i1 %compare.low1395, i1 %compare.high.order1284, !dbg !25
  %774 = select i1 %compare.result1396, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1397 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 59) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1398 = or i32 %compare.local.base1397, 76, !dbg !25
  %compare.low1399 = icmp ugt i32 %compare.low.base1285, %compare.local1398, !dbg !25
  %compare.result1400 = select i1 %compare.high.equal1283, i1 %compare.low1399, i1 %compare.high.order1284, !dbg !25
  %775 = select i1 %compare.result1400, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1401 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 60) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1402 = or i32 %compare.local.base1401, 77, !dbg !25
  %compare.low1403 = icmp ugt i32 %compare.low.base1285, %compare.local1402, !dbg !25
  %compare.result1404 = select i1 %compare.high.equal1283, i1 %compare.low1403, i1 %compare.high.order1284, !dbg !25
  %776 = select i1 %compare.result1404, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1405 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 61) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1406 = or i32 %compare.local.base1405, 78, !dbg !25
  %compare.low1407 = icmp ugt i32 %compare.low.base1285, %compare.local1406, !dbg !25
  %compare.result1408 = select i1 %compare.high.equal1283, i1 %compare.low1407, i1 %compare.high.order1284, !dbg !25
  %777 = select i1 %compare.result1408, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1409 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 62) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1410 = or i32 %compare.local.base1409, 79, !dbg !25
  %compare.low1411 = icmp ugt i32 %compare.low.base1285, %compare.local1410, !dbg !25
  %compare.result1412 = select i1 %compare.high.equal1283, i1 %compare.low1411, i1 %compare.high.order1284, !dbg !25
  %778 = select i1 %compare.result1412, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %779 = insertelement <32 x float> poison, float %712, i64 0, !dbg !14
  %780 = insertelement <32 x float> %779, float %713, i64 1, !dbg !14
  %781 = insertelement <32 x float> %780, float %714, i64 2, !dbg !14
  %782 = insertelement <32 x float> %781, float %715, i64 3, !dbg !14
  %783 = insertelement <32 x float> %782, float %716, i64 4, !dbg !14
  %784 = insertelement <32 x float> %783, float %717, i64 5, !dbg !14
  %785 = insertelement <32 x float> %784, float %718, i64 6, !dbg !14
  %786 = insertelement <32 x float> %785, float %719, i64 7, !dbg !14
  %787 = insertelement <32 x float> %786, float %720, i64 8, !dbg !14
  %788 = insertelement <32 x float> %787, float %721, i64 9, !dbg !14
  %789 = insertelement <32 x float> %788, float %722, i64 10, !dbg !14
  %790 = insertelement <32 x float> %789, float %723, i64 11, !dbg !14
  %791 = insertelement <32 x float> %790, float %724, i64 12, !dbg !14
  %792 = insertelement <32 x float> %791, float %725, i64 13, !dbg !14
  %793 = insertelement <32 x float> %792, float %726, i64 14, !dbg !14
  %794 = insertelement <32 x float> %793, float %727, i64 15, !dbg !14
  %795 = insertelement <32 x float> %794, float %728, i64 16, !dbg !14
  %796 = insertelement <32 x float> %795, float %729, i64 17, !dbg !14
  %797 = insertelement <32 x float> %796, float %730, i64 18, !dbg !14
  %798 = insertelement <32 x float> %797, float %731, i64 19, !dbg !14
  %799 = insertelement <32 x float> %798, float %732, i64 20, !dbg !14
  %800 = insertelement <32 x float> %799, float %733, i64 21, !dbg !14
  %801 = insertelement <32 x float> %800, float %734, i64 22, !dbg !14
  %802 = insertelement <32 x float> %801, float %735, i64 23, !dbg !14
  %803 = insertelement <32 x float> %802, float %736, i64 24, !dbg !14
  %804 = insertelement <32 x float> %803, float %737, i64 25, !dbg !14
  %805 = insertelement <32 x float> %804, float %738, i64 26, !dbg !14
  %806 = insertelement <32 x float> %805, float %739, i64 27, !dbg !14
  %807 = insertelement <32 x float> %806, float %740, i64 28, !dbg !14
  %808 = insertelement <32 x float> %807, float %741, i64 29, !dbg !14
  %809 = insertelement <32 x float> %808, float %742, i64 30, !dbg !14
  %810 = insertelement <32 x float> %809, float %743, i64 31, !dbg !14
  %811 = insertvalue [2 x <32 x float>] poison, <32 x float> %810, 0, !dbg !14
  %812 = insertelement <32 x float> poison, float %747, i64 0, !dbg !14
  %813 = insertelement <32 x float> %812, float %748, i64 1, !dbg !14
  %814 = insertelement <32 x float> %813, float %749, i64 2, !dbg !14
  %815 = insertelement <32 x float> %814, float %750, i64 3, !dbg !14
  %816 = insertelement <32 x float> %815, float %751, i64 4, !dbg !14
  %817 = insertelement <32 x float> %816, float %752, i64 5, !dbg !14
  %818 = insertelement <32 x float> %817, float %753, i64 6, !dbg !14
  %819 = insertelement <32 x float> %818, float %754, i64 7, !dbg !14
  %820 = insertelement <32 x float> %819, float %755, i64 8, !dbg !14
  %821 = insertelement <32 x float> %820, float %756, i64 9, !dbg !14
  %822 = insertelement <32 x float> %821, float %757, i64 10, !dbg !14
  %823 = insertelement <32 x float> %822, float %758, i64 11, !dbg !14
  %824 = insertelement <32 x float> %823, float %759, i64 12, !dbg !14
  %825 = insertelement <32 x float> %824, float %760, i64 13, !dbg !14
  %826 = insertelement <32 x float> %825, float %761, i64 14, !dbg !14
  %827 = insertelement <32 x float> %826, float %762, i64 15, !dbg !14
  %828 = insertelement <32 x float> %827, float %763, i64 16, !dbg !14
  %829 = insertelement <32 x float> %828, float %764, i64 17, !dbg !14
  %830 = insertelement <32 x float> %829, float %765, i64 18, !dbg !14
  %831 = insertelement <32 x float> %830, float %766, i64 19, !dbg !14
  %832 = insertelement <32 x float> %831, float %767, i64 20, !dbg !14
  %833 = insertelement <32 x float> %832, float %768, i64 21, !dbg !14
  %834 = insertelement <32 x float> %833, float %769, i64 22, !dbg !14
  %835 = insertelement <32 x float> %834, float %770, i64 23, !dbg !14
  %836 = insertelement <32 x float> %835, float %771, i64 24, !dbg !14
  %837 = insertelement <32 x float> %836, float %772, i64 25, !dbg !14
  %838 = insertelement <32 x float> %837, float %773, i64 26, !dbg !14
  %839 = insertelement <32 x float> %838, float %774, i64 27, !dbg !14
  %840 = insertelement <32 x float> %839, float %775, i64 28, !dbg !14
  %841 = insertelement <32 x float> %840, float %776, i64 29, !dbg !14
  %842 = insertelement <32 x float> %841, float %777, i64 30, !dbg !14
  %843 = insertelement <32 x float> %842, float %778, i64 31, !dbg !14
  %844 = insertvalue [2 x <32 x float>] %811, <32 x float> %843, 1, !dbg !14
  br label %845, !dbg !14

845:                                              ; preds = %700, %701
  %846 = phi [2 x <32 x float>] [ %844, %701 ], [ zeroinitializer, %700 ], !dbg !14
  br label %847, !dbg !14

847:                                              ; preds = %845
  %848 = extractvalue [2 x <32 x float>] %846, 0, !dbg !27
  %849 = shufflevector <32 x float> %848, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !27
  %850 = fadd <4 x float> %637, %849, !dbg !27
  %851 = extractelement <4 x float> %850, i64 0, !dbg !28
  %852 = call float @llvm.amdgcn.exp2.f32(float %851), !dbg !28
  %853 = insertelement <4 x float> poison, float %852, i64 0, !dbg !30
  %854 = extractelement <4 x float> %850, i64 1, !dbg !28
  %855 = call float @llvm.amdgcn.exp2.f32(float %854), !dbg !28
  %856 = insertelement <4 x float> %853, float %855, i64 1, !dbg !30
  %857 = fadd float %852, %855, !dbg !29
  %858 = extractelement <4 x float> %850, i64 2, !dbg !28
  %859 = call float @llvm.amdgcn.exp2.f32(float %858), !dbg !28
  %860 = insertelement <4 x float> %856, float %859, i64 2, !dbg !30
  %861 = fadd float %857, %859, !dbg !29
  %862 = extractelement <4 x float> %850, i64 3, !dbg !28
  %863 = call float @llvm.amdgcn.exp2.f32(float %862), !dbg !28
  %864 = insertelement <4 x float> %860, float %863, i64 3, !dbg !30
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %864) #9, !dbg !30, !frisk.register_lifetimes !1
  %865 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !30
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %865) #9, !dbg !29, !frisk.register_lifetimes !1
  %866 = fadd float %861, %863, !dbg !29
  %867 = shufflevector <32 x float> %848, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !27
  %868 = fadd <4 x float> %641, %867, !dbg !27
  %869 = extractelement <4 x float> %868, i64 0, !dbg !28
  %870 = call float @llvm.amdgcn.exp2.f32(float %869), !dbg !28
  %871 = insertelement <4 x float> poison, float %870, i64 0, !dbg !30
  %872 = fadd float %866, %870, !dbg !29
  %873 = extractelement <4 x float> %868, i64 1, !dbg !28
  %874 = call float @llvm.amdgcn.exp2.f32(float %873), !dbg !28
  %875 = insertelement <4 x float> %871, float %874, i64 1, !dbg !30
  %876 = fadd float %872, %874, !dbg !29
  %877 = extractelement <4 x float> %868, i64 2, !dbg !28
  %878 = call float @llvm.amdgcn.exp2.f32(float %877), !dbg !28
  %879 = insertelement <4 x float> %875, float %878, i64 2, !dbg !30
  %880 = fadd float %876, %878, !dbg !29
  %881 = extractelement <4 x float> %868, i64 3, !dbg !28
  %882 = call float @llvm.amdgcn.exp2.f32(float %881), !dbg !28
  %883 = insertelement <4 x float> %879, float %882, i64 3, !dbg !30
  %math.result309 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %883) #9, !dbg !30, !frisk.register_lifetimes !1
  %884 = fptrunc <4 x float> %math.result309 to <4 x half>, !dbg !30
  %math.packed310 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %884) #9, !dbg !29, !frisk.register_lifetimes !1
  %885 = fadd float %880, %882, !dbg !29
  %886 = shufflevector <32 x float> %848, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !27
  %887 = fadd <4 x float> %645, %886, !dbg !27
  %888 = extractelement <4 x float> %887, i64 0, !dbg !28
  %889 = call float @llvm.amdgcn.exp2.f32(float %888), !dbg !28
  %890 = insertelement <4 x float> poison, float %889, i64 0, !dbg !30
  %891 = fadd float %885, %889, !dbg !29
  %892 = extractelement <4 x float> %887, i64 1, !dbg !28
  %893 = call float @llvm.amdgcn.exp2.f32(float %892), !dbg !28
  %894 = insertelement <4 x float> %890, float %893, i64 1, !dbg !30
  %895 = fadd float %891, %893, !dbg !29
  %896 = extractelement <4 x float> %887, i64 2, !dbg !28
  %897 = call float @llvm.amdgcn.exp2.f32(float %896), !dbg !28
  %898 = insertelement <4 x float> %894, float %897, i64 2, !dbg !30
  %899 = fadd float %895, %897, !dbg !29
  %900 = extractelement <4 x float> %887, i64 3, !dbg !28
  %901 = call float @llvm.amdgcn.exp2.f32(float %900), !dbg !28
  %902 = insertelement <4 x float> %898, float %901, i64 3, !dbg !30
  %math.result311 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %902) #9, !dbg !30, !frisk.register_lifetimes !1
  %903 = fptrunc <4 x float> %math.result311 to <4 x half>, !dbg !30
  %math.packed312 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %903) #9, !dbg !29, !frisk.register_lifetimes !1
  %904 = fadd float %899, %901, !dbg !29
  %905 = shufflevector <32 x float> %848, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !27
  %906 = fadd <4 x float> %649, %905, !dbg !27
  %907 = extractelement <4 x float> %906, i64 0, !dbg !28
  %908 = call float @llvm.amdgcn.exp2.f32(float %907), !dbg !28
  %909 = insertelement <4 x float> poison, float %908, i64 0, !dbg !30
  %910 = fadd float %904, %908, !dbg !29
  %911 = extractelement <4 x float> %906, i64 1, !dbg !28
  %912 = call float @llvm.amdgcn.exp2.f32(float %911), !dbg !28
  %913 = insertelement <4 x float> %909, float %912, i64 1, !dbg !30
  %914 = fadd float %910, %912, !dbg !29
  %915 = extractelement <4 x float> %906, i64 2, !dbg !28
  %916 = call float @llvm.amdgcn.exp2.f32(float %915), !dbg !28
  %917 = insertelement <4 x float> %913, float %916, i64 2, !dbg !30
  %918 = fadd float %914, %916, !dbg !29
  %919 = extractelement <4 x float> %906, i64 3, !dbg !28
  %920 = call float @llvm.amdgcn.exp2.f32(float %919), !dbg !28
  %921 = insertelement <4 x float> %917, float %920, i64 3, !dbg !30
  %math.result313 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %921) #9, !dbg !30, !frisk.register_lifetimes !1
  %922 = fptrunc <4 x float> %math.result313 to <4 x half>, !dbg !30
  %math.packed314 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %922) #9, !dbg !29, !frisk.register_lifetimes !1
  %923 = fadd float %918, %920, !dbg !29
  %924 = shufflevector <32 x float> %848, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !27
  %925 = fadd <4 x float> %653, %924, !dbg !27
  %926 = extractelement <4 x float> %925, i64 0, !dbg !28
  %927 = call float @llvm.amdgcn.exp2.f32(float %926), !dbg !28
  %928 = insertelement <4 x float> poison, float %927, i64 0, !dbg !30
  %929 = fadd float %923, %927, !dbg !29
  %930 = extractelement <4 x float> %925, i64 1, !dbg !28
  %931 = call float @llvm.amdgcn.exp2.f32(float %930), !dbg !28
  %932 = insertelement <4 x float> %928, float %931, i64 1, !dbg !30
  %933 = fadd float %929, %931, !dbg !29
  %934 = extractelement <4 x float> %925, i64 2, !dbg !28
  %935 = call float @llvm.amdgcn.exp2.f32(float %934), !dbg !28
  %936 = insertelement <4 x float> %932, float %935, i64 2, !dbg !30
  %937 = fadd float %933, %935, !dbg !29
  %938 = extractelement <4 x float> %925, i64 3, !dbg !28
  %939 = call float @llvm.amdgcn.exp2.f32(float %938), !dbg !28
  %940 = insertelement <4 x float> %936, float %939, i64 3, !dbg !30
  %math.result315 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %940) #9, !dbg !30, !frisk.register_lifetimes !1
  %941 = fptrunc <4 x float> %math.result315 to <4 x half>, !dbg !30
  %math.packed316 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %941) #9, !dbg !29, !frisk.register_lifetimes !1
  %942 = fadd float %937, %939, !dbg !29
  %943 = shufflevector <32 x float> %848, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !27
  %944 = fadd <4 x float> %657, %943, !dbg !27
  %945 = extractelement <4 x float> %944, i64 0, !dbg !28
  %946 = call float @llvm.amdgcn.exp2.f32(float %945), !dbg !28
  %947 = insertelement <4 x float> poison, float %946, i64 0, !dbg !30
  %948 = fadd float %942, %946, !dbg !29
  %949 = extractelement <4 x float> %944, i64 1, !dbg !28
  %950 = call float @llvm.amdgcn.exp2.f32(float %949), !dbg !28
  %951 = insertelement <4 x float> %947, float %950, i64 1, !dbg !30
  %952 = fadd float %948, %950, !dbg !29
  %953 = extractelement <4 x float> %944, i64 2, !dbg !28
  %954 = call float @llvm.amdgcn.exp2.f32(float %953), !dbg !28
  %955 = insertelement <4 x float> %951, float %954, i64 2, !dbg !30
  %956 = fadd float %952, %954, !dbg !29
  %957 = extractelement <4 x float> %944, i64 3, !dbg !28
  %958 = call float @llvm.amdgcn.exp2.f32(float %957), !dbg !28
  %959 = insertelement <4 x float> %955, float %958, i64 3, !dbg !30
  %math.result317 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %959) #9, !dbg !30, !frisk.register_lifetimes !1
  %960 = fptrunc <4 x float> %math.result317 to <4 x half>, !dbg !30
  %math.packed318 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %960) #9, !dbg !29, !frisk.register_lifetimes !1
  %961 = fadd float %956, %958, !dbg !29
  %962 = shufflevector <32 x float> %848, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !27
  %963 = fadd <4 x float> %661, %962, !dbg !27
  %964 = extractelement <4 x float> %963, i64 0, !dbg !28
  %965 = call float @llvm.amdgcn.exp2.f32(float %964), !dbg !28
  %966 = insertelement <4 x float> poison, float %965, i64 0, !dbg !30
  %967 = fadd float %961, %965, !dbg !29
  %968 = extractelement <4 x float> %963, i64 1, !dbg !28
  %969 = call float @llvm.amdgcn.exp2.f32(float %968), !dbg !28
  %970 = insertelement <4 x float> %966, float %969, i64 1, !dbg !30
  %971 = fadd float %967, %969, !dbg !29
  %972 = extractelement <4 x float> %963, i64 2, !dbg !28
  %973 = call float @llvm.amdgcn.exp2.f32(float %972), !dbg !28
  %974 = insertelement <4 x float> %970, float %973, i64 2, !dbg !30
  %975 = fadd float %971, %973, !dbg !29
  %976 = extractelement <4 x float> %963, i64 3, !dbg !28
  %977 = call float @llvm.amdgcn.exp2.f32(float %976), !dbg !28
  %978 = insertelement <4 x float> %974, float %977, i64 3, !dbg !30
  %math.result319 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %978) #9, !dbg !30, !frisk.register_lifetimes !1
  %979 = fptrunc <4 x float> %math.result319 to <4 x half>, !dbg !30
  %math.packed320 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %979) #9, !dbg !29, !frisk.register_lifetimes !1
  %980 = fadd float %975, %977, !dbg !29
  %981 = shufflevector <32 x float> %848, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !27
  %982 = fadd <4 x float> %665, %981, !dbg !27
  %983 = extractelement <4 x float> %982, i64 0, !dbg !28
  %984 = call float @llvm.amdgcn.exp2.f32(float %983), !dbg !28
  %985 = insertelement <4 x float> poison, float %984, i64 0, !dbg !30
  %986 = fadd float %980, %984, !dbg !29
  %987 = extractelement <4 x float> %982, i64 1, !dbg !28
  %988 = call float @llvm.amdgcn.exp2.f32(float %987), !dbg !28
  %989 = insertelement <4 x float> %985, float %988, i64 1, !dbg !30
  %990 = fadd float %986, %988, !dbg !29
  %991 = extractelement <4 x float> %982, i64 2, !dbg !28
  %992 = call float @llvm.amdgcn.exp2.f32(float %991), !dbg !28
  %993 = insertelement <4 x float> %989, float %992, i64 2, !dbg !30
  %994 = fadd float %990, %992, !dbg !29
  %995 = extractelement <4 x float> %982, i64 3, !dbg !28
  %996 = call float @llvm.amdgcn.exp2.f32(float %995), !dbg !28
  %997 = insertelement <4 x float> %993, float %996, i64 3, !dbg !30
  %math.result321 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %997) #9, !dbg !30, !frisk.register_lifetimes !1
  %998 = fptrunc <4 x float> %math.result321 to <4 x half>, !dbg !30
  %math.packed322 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %998) #9, !dbg !29, !frisk.register_lifetimes !1
  %999 = fadd float %994, %996, !dbg !29
  %1000 = extractvalue [2 x <32 x float>] %846, 1, !dbg !27
  %1001 = shufflevector <32 x float> %1000, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !27
  %1002 = fadd <4 x float> %669, %1001, !dbg !27
  %1003 = extractelement <4 x float> %1002, i64 0, !dbg !28
  %1004 = call float @llvm.amdgcn.exp2.f32(float %1003), !dbg !28
  %1005 = insertelement <4 x float> poison, float %1004, i64 0, !dbg !30
  %1006 = extractelement <4 x float> %1002, i64 1, !dbg !28
  %1007 = call float @llvm.amdgcn.exp2.f32(float %1006), !dbg !28
  %1008 = insertelement <4 x float> %1005, float %1007, i64 1, !dbg !30
  %1009 = fadd float %1004, %1007, !dbg !29
  %1010 = extractelement <4 x float> %1002, i64 2, !dbg !28
  %1011 = call float @llvm.amdgcn.exp2.f32(float %1010), !dbg !28
  %1012 = insertelement <4 x float> %1008, float %1011, i64 2, !dbg !30
  %1013 = fadd float %1009, %1011, !dbg !29
  %1014 = extractelement <4 x float> %1002, i64 3, !dbg !28
  %1015 = call float @llvm.amdgcn.exp2.f32(float %1014), !dbg !28
  %1016 = insertelement <4 x float> %1012, float %1015, i64 3, !dbg !30
  %math.result323 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1016) #9, !dbg !30, !frisk.register_lifetimes !1
  %1017 = fptrunc <4 x float> %math.result323 to <4 x half>, !dbg !30
  %math.packed324 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1017) #9, !dbg !29, !frisk.register_lifetimes !1
  %1018 = fadd float %1013, %1015, !dbg !29
  %1019 = shufflevector <32 x float> %1000, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !27
  %1020 = fadd <4 x float> %673, %1019, !dbg !27
  %1021 = extractelement <4 x float> %1020, i64 0, !dbg !28
  %1022 = call float @llvm.amdgcn.exp2.f32(float %1021), !dbg !28
  %1023 = insertelement <4 x float> poison, float %1022, i64 0, !dbg !30
  %1024 = fadd float %1018, %1022, !dbg !29
  %1025 = extractelement <4 x float> %1020, i64 1, !dbg !28
  %1026 = call float @llvm.amdgcn.exp2.f32(float %1025), !dbg !28
  %1027 = insertelement <4 x float> %1023, float %1026, i64 1, !dbg !30
  %1028 = fadd float %1024, %1026, !dbg !29
  %1029 = extractelement <4 x float> %1020, i64 2, !dbg !28
  %1030 = call float @llvm.amdgcn.exp2.f32(float %1029), !dbg !28
  %1031 = insertelement <4 x float> %1027, float %1030, i64 2, !dbg !30
  %1032 = fadd float %1028, %1030, !dbg !29
  %1033 = extractelement <4 x float> %1020, i64 3, !dbg !28
  %1034 = call float @llvm.amdgcn.exp2.f32(float %1033), !dbg !28
  %1035 = insertelement <4 x float> %1031, float %1034, i64 3, !dbg !30
  %math.result325 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1035) #9, !dbg !30, !frisk.register_lifetimes !1
  %1036 = fptrunc <4 x float> %math.result325 to <4 x half>, !dbg !30
  %math.packed326 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1036) #9, !dbg !29, !frisk.register_lifetimes !1
  %1037 = fadd float %1032, %1034, !dbg !29
  %1038 = shufflevector <32 x float> %1000, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !27
  %1039 = fadd <4 x float> %677, %1038, !dbg !27
  %1040 = extractelement <4 x float> %1039, i64 0, !dbg !28
  %1041 = call float @llvm.amdgcn.exp2.f32(float %1040), !dbg !28
  %1042 = insertelement <4 x float> poison, float %1041, i64 0, !dbg !30
  %1043 = fadd float %1037, %1041, !dbg !29
  %1044 = extractelement <4 x float> %1039, i64 1, !dbg !28
  %1045 = call float @llvm.amdgcn.exp2.f32(float %1044), !dbg !28
  %1046 = insertelement <4 x float> %1042, float %1045, i64 1, !dbg !30
  %1047 = fadd float %1043, %1045, !dbg !29
  %1048 = extractelement <4 x float> %1039, i64 2, !dbg !28
  %1049 = call float @llvm.amdgcn.exp2.f32(float %1048), !dbg !28
  %1050 = insertelement <4 x float> %1046, float %1049, i64 2, !dbg !30
  %1051 = fadd float %1047, %1049, !dbg !29
  %1052 = extractelement <4 x float> %1039, i64 3, !dbg !28
  %1053 = call float @llvm.amdgcn.exp2.f32(float %1052), !dbg !28
  %1054 = insertelement <4 x float> %1050, float %1053, i64 3, !dbg !30
  %math.result327 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1054) #9, !dbg !30, !frisk.register_lifetimes !1
  %1055 = fptrunc <4 x float> %math.result327 to <4 x half>, !dbg !30
  %math.packed328 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1055) #9, !dbg !29, !frisk.register_lifetimes !1
  %1056 = fadd float %1051, %1053, !dbg !29
  %1057 = shufflevector <32 x float> %1000, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !27
  %1058 = fadd <4 x float> %681, %1057, !dbg !27
  %1059 = extractelement <4 x float> %1058, i64 0, !dbg !28
  %1060 = call float @llvm.amdgcn.exp2.f32(float %1059), !dbg !28
  %1061 = insertelement <4 x float> poison, float %1060, i64 0, !dbg !30
  %1062 = fadd float %1056, %1060, !dbg !29
  %1063 = extractelement <4 x float> %1058, i64 1, !dbg !28
  %1064 = call float @llvm.amdgcn.exp2.f32(float %1063), !dbg !28
  %1065 = insertelement <4 x float> %1061, float %1064, i64 1, !dbg !30
  %1066 = fadd float %1062, %1064, !dbg !29
  %1067 = extractelement <4 x float> %1058, i64 2, !dbg !28
  %1068 = call float @llvm.amdgcn.exp2.f32(float %1067), !dbg !28
  %1069 = insertelement <4 x float> %1065, float %1068, i64 2, !dbg !30
  %1070 = fadd float %1066, %1068, !dbg !29
  %1071 = extractelement <4 x float> %1058, i64 3, !dbg !28
  %1072 = call float @llvm.amdgcn.exp2.f32(float %1071), !dbg !28
  %1073 = insertelement <4 x float> %1069, float %1072, i64 3, !dbg !30
  %math.result329 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1073) #9, !dbg !30, !frisk.register_lifetimes !1
  %1074 = fptrunc <4 x float> %math.result329 to <4 x half>, !dbg !30
  %math.packed330 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1074) #9, !dbg !29, !frisk.register_lifetimes !1
  %1075 = fadd float %1070, %1072, !dbg !29
  %1076 = shufflevector <32 x float> %1000, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !27
  %1077 = fadd <4 x float> %685, %1076, !dbg !27
  %1078 = extractelement <4 x float> %1077, i64 0, !dbg !28
  %1079 = call float @llvm.amdgcn.exp2.f32(float %1078), !dbg !28
  %1080 = insertelement <4 x float> poison, float %1079, i64 0, !dbg !30
  %1081 = fadd float %1075, %1079, !dbg !29
  %1082 = extractelement <4 x float> %1077, i64 1, !dbg !28
  %1083 = call float @llvm.amdgcn.exp2.f32(float %1082), !dbg !28
  %1084 = insertelement <4 x float> %1080, float %1083, i64 1, !dbg !30
  %1085 = fadd float %1081, %1083, !dbg !29
  %1086 = extractelement <4 x float> %1077, i64 2, !dbg !28
  %1087 = call float @llvm.amdgcn.exp2.f32(float %1086), !dbg !28
  %1088 = insertelement <4 x float> %1084, float %1087, i64 2, !dbg !30
  %1089 = fadd float %1085, %1087, !dbg !29
  %1090 = extractelement <4 x float> %1077, i64 3, !dbg !28
  %1091 = call float @llvm.amdgcn.exp2.f32(float %1090), !dbg !28
  %1092 = insertelement <4 x float> %1088, float %1091, i64 3, !dbg !30
  %math.result331 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1092) #9, !dbg !30, !frisk.register_lifetimes !1
  %1093 = fptrunc <4 x float> %math.result331 to <4 x half>, !dbg !30
  %math.packed332 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1093) #9, !dbg !29, !frisk.register_lifetimes !1
  %1094 = fadd float %1089, %1091, !dbg !29
  %1095 = shufflevector <32 x float> %1000, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !27
  %1096 = fadd <4 x float> %689, %1095, !dbg !27
  %1097 = extractelement <4 x float> %1096, i64 0, !dbg !28
  %1098 = call float @llvm.amdgcn.exp2.f32(float %1097), !dbg !28
  %1099 = insertelement <4 x float> poison, float %1098, i64 0, !dbg !30
  %1100 = fadd float %1094, %1098, !dbg !29
  %1101 = extractelement <4 x float> %1096, i64 1, !dbg !28
  %1102 = call float @llvm.amdgcn.exp2.f32(float %1101), !dbg !28
  %1103 = insertelement <4 x float> %1099, float %1102, i64 1, !dbg !30
  %1104 = fadd float %1100, %1102, !dbg !29
  %1105 = extractelement <4 x float> %1096, i64 2, !dbg !28
  %1106 = call float @llvm.amdgcn.exp2.f32(float %1105), !dbg !28
  %1107 = insertelement <4 x float> %1103, float %1106, i64 2, !dbg !30
  %1108 = fadd float %1104, %1106, !dbg !29
  %1109 = extractelement <4 x float> %1096, i64 3, !dbg !28
  %1110 = call float @llvm.amdgcn.exp2.f32(float %1109), !dbg !28
  %1111 = insertelement <4 x float> %1107, float %1110, i64 3, !dbg !30
  %math.result333 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1111) #9, !dbg !30, !frisk.register_lifetimes !1
  %1112 = fptrunc <4 x float> %math.result333 to <4 x half>, !dbg !30
  %math.packed334 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1112) #9, !dbg !29, !frisk.register_lifetimes !1
  %1113 = fadd float %1108, %1110, !dbg !29
  %1114 = shufflevector <32 x float> %1000, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !27
  %1115 = fadd <4 x float> %693, %1114, !dbg !27
  %1116 = extractelement <4 x float> %1115, i64 0, !dbg !28
  %1117 = call float @llvm.amdgcn.exp2.f32(float %1116), !dbg !28
  %1118 = insertelement <4 x float> poison, float %1117, i64 0, !dbg !30
  %1119 = fadd float %1113, %1117, !dbg !29
  %1120 = extractelement <4 x float> %1115, i64 1, !dbg !28
  %1121 = call float @llvm.amdgcn.exp2.f32(float %1120), !dbg !28
  %1122 = insertelement <4 x float> %1118, float %1121, i64 1, !dbg !30
  %1123 = fadd float %1119, %1121, !dbg !29
  %1124 = extractelement <4 x float> %1115, i64 2, !dbg !28
  %1125 = call float @llvm.amdgcn.exp2.f32(float %1124), !dbg !28
  %1126 = insertelement <4 x float> %1122, float %1125, i64 2, !dbg !30
  %1127 = fadd float %1123, %1125, !dbg !29
  %1128 = extractelement <4 x float> %1115, i64 3, !dbg !28
  %1129 = call float @llvm.amdgcn.exp2.f32(float %1128), !dbg !28
  %1130 = insertelement <4 x float> %1126, float %1129, i64 3, !dbg !30
  %math.result335 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1130) #9, !dbg !30, !frisk.register_lifetimes !1
  %1131 = fptrunc <4 x float> %math.result335 to <4 x half>, !dbg !30
  %math.packed336 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1131) #9, !dbg !29, !frisk.register_lifetimes !1
  %1132 = fadd float %1127, %1129, !dbg !29
  %1133 = shufflevector <32 x float> %1000, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !27
  %1134 = fadd <4 x float> %698, %1133, !dbg !27
  %1135 = extractelement <4 x float> %1134, i64 0, !dbg !28
  %1136 = call float @llvm.amdgcn.exp2.f32(float %1135), !dbg !28
  %1137 = insertelement <4 x float> poison, float %1136, i64 0, !dbg !30
  %1138 = fadd float %1132, %1136, !dbg !29
  %1139 = extractelement <4 x float> %1134, i64 1, !dbg !28
  %1140 = call float @llvm.amdgcn.exp2.f32(float %1139), !dbg !28
  %1141 = insertelement <4 x float> %1137, float %1140, i64 1, !dbg !30
  %1142 = fadd float %1138, %1140, !dbg !29
  %1143 = extractelement <4 x float> %1134, i64 2, !dbg !28
  %1144 = call float @llvm.amdgcn.exp2.f32(float %1143), !dbg !28
  %1145 = insertelement <4 x float> %1141, float %1144, i64 2, !dbg !30
  %1146 = fadd float %1142, %1144, !dbg !29
  %1147 = extractelement <4 x float> %1134, i64 3, !dbg !28
  %1148 = call float @llvm.amdgcn.exp2.f32(float %1147), !dbg !28
  %1149 = insertelement <4 x float> %1145, float %1148, i64 3, !dbg !30
  %math.result337 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1149) #9, !dbg !30, !frisk.register_lifetimes !1
  %1150 = fptrunc <4 x float> %math.result337 to <4 x half>, !dbg !30
  %math.packed338 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1150) #9, !dbg !29, !frisk.register_lifetimes !1
  %1151 = fadd float %1146, %1148, !dbg !29
  %1152 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !29
  %1153 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1152), !dbg !29
  %1154 = add nuw nsw i32 %1153, 64, !dbg !29
  %1155 = xor i32 %1153, 16, !dbg !29
  %1156 = and i32 %1154, 192, !dbg !29
  %1157 = icmp samesign ult i32 %1155, %1156, !dbg !29
  %1158 = select i1 %1157, i32 %1155, i32 %1153, !dbg !29
  %1159 = shl nuw nsw i32 %1158, 2, !dbg !29
  %1160 = bitcast float %999 to i32, !dbg !29
  %1161 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1159, i32 %1160), !dbg !29
  %1162 = bitcast i32 %1161 to float, !dbg !29
  %1163 = fadd float %999, %1162, !dbg !29
  %1164 = xor i32 %1153, 32, !dbg !29
  %1165 = icmp samesign ult i32 %1164, %1156, !dbg !29
  %1166 = select i1 %1165, i32 %1164, i32 %1153, !dbg !29
  %1167 = shl nuw nsw i32 %1166, 2, !dbg !29
  %1168 = bitcast float %1163 to i32, !dbg !29
  %1169 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1167, i32 %1168), !dbg !29
  %1170 = bitcast i32 %1169 to float, !dbg !29
  %1171 = fadd float %1163, %1170, !dbg !29
  %1172 = extractvalue [2 x <1 x float>] %207, 0, !dbg !29
  %1173 = extractelement <1 x float> %1172, i64 0, !dbg !29
  %1174 = fadd float %1171, %1173, !dbg !29
  %1175 = insertelement <1 x float> poison, float %1174, i64 0, !dbg !13
  %1176 = insertvalue [2 x <1 x float>] poison, <1 x float> %1175, 0, !dbg !13
  %1177 = bitcast float %1151 to i32, !dbg !29
  %1178 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1159, i32 %1177), !dbg !29
  %1179 = bitcast i32 %1178 to float, !dbg !29
  %1180 = fadd float %1151, %1179, !dbg !29
  %1181 = bitcast float %1180 to i32, !dbg !29
  %1182 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1167, i32 %1181), !dbg !29
  %1183 = bitcast i32 %1182 to float, !dbg !29
  %1184 = fadd float %1180, %1183, !dbg !29
  %1185 = extractvalue [2 x <1 x float>] %207, 1, !dbg !29
  %1186 = extractelement <1 x float> %1185, i64 0, !dbg !29
  %1187 = fadd float %1184, %1186, !dbg !29
  %1188 = insertelement <1 x float> poison, float %1187, i64 0, !dbg !13
  %1189 = insertvalue [2 x <1 x float>] %1176, <1 x float> %1188, 1, !dbg !13
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !19
  fence syncscope("workgroup") release, !dbg !19
  call void @llvm.amdgcn.s.barrier(), !dbg !19
  fence syncscope("workgroup") acquire, !dbg !19
  %address.thread864 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !15, !range !34, !frisk.register_lifetimes !1
  %address.local865 = shl nuw nsw i32 %address.thread864, 3, !dbg !15
  %address.local866 = and i32 %address.local865, 120, !dbg !15
  %address.local867 = shl nuw nsw i32 %address.thread864, 5, !dbg !15
  %address.local868 = and i32 %address.local867, 1536, !dbg !15
  %address.local869 = or disjoint i32 %address.local866, %address.local868, !dbg !15
  %address.local870 = zext nneg i32 %address.local869 to i64, !dbg !15
  %address.local871 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local870, !dbg !15
  %1190 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local871, i32 0) #0, !dbg !15
  %1191 = bitcast <4 x i32> %1190 to <8 x half>, !dbg !15
  %address.local872 = zext nneg i32 %address.thread864 to i64, !dbg !8
  %address.local873 = and i64 %address.local872, 60, !dbg !15
  %address.local874 = add nuw nsw i64 %address.local873, %address.local872, !dbg !15
  %address.local875 = shl nuw nsw i64 %address.local874, 3, !dbg !15
  %address.local876 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local875, !dbg !15
  %address.local877 = and i64 %address.local872, 63, !dbg !11
  %address.local878 = lshr i64 %address.local877, 4, !dbg !15
  %address.local879 = mul nuw nsw i64 %address.local878, 384, !dbg !15
  %address.local880 = getelementptr [2 x i8], ptr addrspace(3) %address.local876, i64 %address.local879, !dbg !15
  %address.local881 = lshr i64 %address.local872, 2, !dbg !18
  %address.local882 = mul nsw i64 %address.local881, -32, !dbg !15
  %address.local883 = getelementptr [2 x i8], ptr addrspace(3) %address.local880, i64 %address.local882, !dbg !15
  %address.local884 = getelementptr i8, ptr addrspace(3) %address.local883, i64 256, !dbg !15
  %1192 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local884, i32 0) #0, !dbg !15
  %1193 = bitcast <4 x i32> %1192 to <8 x half>, !dbg !15
  %address.local885 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local875, !dbg !15
  %address.local886 = getelementptr [2 x i8], ptr addrspace(3) %address.local885, i64 %address.local879, !dbg !15
  %address.local887 = getelementptr [2 x i8], ptr addrspace(3) %address.local886, i64 %address.local882, !dbg !15
  %address.local888 = getelementptr i8, ptr addrspace(3) %address.local887, i64 512, !dbg !15
  %1194 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local888, i32 0) #0, !dbg !15
  %1195 = bitcast <4 x i32> %1194 to <8 x half>, !dbg !15
  %address.local889 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local875, !dbg !15
  %address.local890 = getelementptr [2 x i8], ptr addrspace(3) %address.local889, i64 %address.local879, !dbg !15
  %address.local891 = getelementptr [2 x i8], ptr addrspace(3) %address.local890, i64 %address.local882, !dbg !15
  %address.local892 = getelementptr i8, ptr addrspace(3) %address.local891, i64 768, !dbg !15
  %1196 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local892, i32 0) #0, !dbg !15
  %1197 = bitcast <4 x i32> %1196 to <8 x half>, !dbg !15
  %address.local893 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local870, !dbg !15
  %address.local894 = getelementptr i8, ptr addrspace(3) %address.local893, i64 4096, !dbg !15
  %1198 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local894, i32 0) #0, !dbg !15
  %1199 = bitcast <4 x i32> %1198 to <8 x half>, !dbg !15
  %address.local895 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local875, !dbg !15
  %address.local896 = getelementptr [2 x i8], ptr addrspace(3) %address.local895, i64 %address.local879, !dbg !15
  %address.local897 = getelementptr [2 x i8], ptr addrspace(3) %address.local896, i64 %address.local882, !dbg !15
  %address.local898 = getelementptr i8, ptr addrspace(3) %address.local897, i64 4352, !dbg !15
  %1200 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local898, i32 0) #0, !dbg !15
  %1201 = bitcast <4 x i32> %1200 to <8 x half>, !dbg !15
  %address.local899 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local875, !dbg !15
  %address.local900 = getelementptr [2 x i8], ptr addrspace(3) %address.local899, i64 %address.local879, !dbg !15
  %address.local901 = getelementptr [2 x i8], ptr addrspace(3) %address.local900, i64 %address.local882, !dbg !15
  %address.local902 = getelementptr i8, ptr addrspace(3) %address.local901, i64 4608, !dbg !15
  %1202 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local902, i32 0) #0, !dbg !15
  %1203 = bitcast <4 x i32> %1202 to <8 x half>, !dbg !15
  %address.local903 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local875, !dbg !15
  %address.local904 = getelementptr [2 x i8], ptr addrspace(3) %address.local903, i64 %address.local879, !dbg !15
  %address.local905 = getelementptr [2 x i8], ptr addrspace(3) %address.local904, i64 %address.local882, !dbg !15
  %address.local906 = getelementptr i8, ptr addrspace(3) %address.local905, i64 4864, !dbg !15
  %1204 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local906, i32 0) #0, !dbg !15
  %1205 = bitcast <4 x i32> %1204 to <8 x half>, !dbg !15
  %1206 = add nuw nsw i64 %216, %16, !dbg !19
  %1207 = shl i64 %1206, 19, !dbg !19
  %1208 = and i64 %1207, 16252928, !dbg !19
  %1209 = add i64 %1208, %222, !dbg !19
  fence syncscope("workgroup") release, !dbg !19
  call void @llvm.amdgcn.s.barrier(), !dbg !19
  fence syncscope("workgroup") acquire, !dbg !19
  %1210 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %1209, !dbg !19
  %1211 = getelementptr i8, ptr addrspace(1) %1210, i64 64, !dbg !19
  %address.local907 = shl nuw nsw i64 %address.local872, 3, !dbg !8
  %address.local908 = and i64 %address.local907, 1536, !dbg !8
  %address.local909 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local908, !dbg !8
  %address.local910 = ptrtoint ptr addrspace(3) %address.local909 to i64, !dbg !8
  %address.local911 = trunc i64 %address.local910 to i32, !dbg !8
  %1212 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local911), !dbg !19
  %1213 = zext i32 %1212 to i64, !dbg !19
  %1214 = inttoptr i64 %1213 to ptr addrspace(3), !dbg !19
  %address.local912 = shl nuw nsw i64 %address.local872, 5, !dbg !19
  %address.local913 = and i64 %address.local912, 8064, !dbg !19
  %address.local914 = shl nuw nsw i64 %address.local872, 3, !dbg !19
  %address.local915 = and i64 %address.local914, 24, !dbg !19
  %address.local916 = or disjoint i64 %address.local913, %address.local915, !dbg !19
  %address.local917 = getelementptr [2 x i8], ptr addrspace(1) %1211, i64 %address.local916, !dbg !19
  %1215 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !19
  %1216 = ptrtoint ptr addrspace(1) %address.local917 to i64, !dbg !19
  %1217 = sub i64 %1216, %1215, !dbg !19
  %dma.buffer.offset1558 = trunc i64 %1217 to i32, !dbg !19
  %1218 = trunc i64 %1215 to i32, !dbg !19
  %1219 = insertelement <4 x i32> poison, i32 %1218, i64 0, !dbg !19
  %1220 = lshr i64 %1215, 32, !dbg !19
  %1221 = trunc i64 %1220 to i32, !dbg !19
  %1222 = insertelement <4 x i32> %1219, i32 %1221, i64 1, !dbg !19
  %1223 = insertelement <4 x i32> %1222, i32 33554432, i64 2, !dbg !19
  %dma.buffer.rsrc1559 = insertelement <4 x i32> %1223, i32 131072, i64 3, !dbg !19
  %1224 = ptrtoint ptr addrspace(3) %1214 to i32, !dbg !19
  %dma.buffer.lds1560 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1224), !dbg !19
  %1225 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1560, i32 %dma.buffer.offset1558, <4 x i32> %dma.buffer.rsrc1559) #10, !dbg !19
  %address.local918 = or disjoint i64 %address.local872, 256, !dbg !8
  %address.local919 = shl nuw nsw i64 %address.local918, 3, !dbg !8
  %address.local920 = and i64 %address.local919, 3584, !dbg !8
  %address.local921 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local920, !dbg !8
  %address.local922 = ptrtoint ptr addrspace(3) %address.local921 to i64, !dbg !8
  %address.local923 = trunc i64 %address.local922 to i32, !dbg !8
  %1226 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local923), !dbg !19
  %1227 = zext i32 %1226 to i64, !dbg !19
  %1228 = inttoptr i64 %1227 to ptr addrspace(3), !dbg !19
  %address.local924 = shl nuw nsw i64 %address.local918, 5, !dbg !19
  %address.local925 = and i64 %address.local924, 16256, !dbg !19
  %address.local926 = shl nuw nsw i64 %address.local872, 3, !dbg !19
  %address.local927 = and i64 %address.local926, 24, !dbg !19
  %address.local928 = or disjoint i64 %address.local925, %address.local927, !dbg !19
  %address.local929 = getelementptr [2 x i8], ptr addrspace(1) %1211, i64 %address.local928, !dbg !19
  %1229 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !19
  %1230 = ptrtoint ptr addrspace(1) %address.local929 to i64, !dbg !19
  %1231 = sub i64 %1230, %1229, !dbg !19
  %dma.buffer.offset1555 = trunc i64 %1231 to i32, !dbg !19
  %1232 = trunc i64 %1229 to i32, !dbg !19
  %1233 = insertelement <4 x i32> poison, i32 %1232, i64 0, !dbg !19
  %1234 = lshr i64 %1229, 32, !dbg !19
  %1235 = trunc i64 %1234 to i32, !dbg !19
  %1236 = insertelement <4 x i32> %1233, i32 %1235, i64 1, !dbg !19
  %1237 = insertelement <4 x i32> %1236, i32 33554432, i64 2, !dbg !19
  %dma.buffer.rsrc1556 = insertelement <4 x i32> %1237, i32 131072, i64 3, !dbg !19
  %1238 = ptrtoint ptr addrspace(3) %1228 to i32, !dbg !19
  %dma.buffer.lds1557 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1238), !dbg !19
  %1239 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1557, i32 %dma.buffer.offset1555, <4 x i32> %dma.buffer.rsrc1556) #10, !dbg !19
  %1240 = shufflevector <8 x half> %1191, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1241 = extractvalue [2 x <8 x float>] %206, 0, !dbg !31
  %1242 = shufflevector <8 x float> %1241, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %1243 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1240, <4 x float> %1242) #9, !dbg !15
  %1244 = shufflevector <8 x half> %1193, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1245 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %1244, <4 x float> %1243) #9, !dbg !15
  %1246 = shufflevector <8 x half> %1195, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1247 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %1246, <4 x float> %1245) #9, !dbg !15
  %1248 = shufflevector <8 x half> %1197, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1249 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %1248, <4 x float> %1247) #9, !dbg !15
  %1250 = shufflevector <8 x half> %1199, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1251 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %1250, <4 x float> %1249) #9, !dbg !15
  %1252 = shufflevector <8 x half> %1201, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1253 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %1252, <4 x float> %1251) #9, !dbg !15
  %1254 = shufflevector <8 x half> %1203, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1255 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %1254, <4 x float> %1253) #9, !dbg !15
  %1256 = shufflevector <8 x half> %1205, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1257 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %1256, <4 x float> %1255) #9, !dbg !15
  %1258 = shufflevector <8 x half> %1191, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1259 = shufflevector <8 x float> %1241, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %1260 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1258, <4 x float> %1259) #9, !dbg !15
  %1261 = shufflevector <8 x half> %1193, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1262 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %1261, <4 x float> %1260) #9, !dbg !15
  %1263 = shufflevector <8 x half> %1195, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1264 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %1263, <4 x float> %1262) #9, !dbg !15
  %1265 = shufflevector <8 x half> %1197, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1266 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %1265, <4 x float> %1264) #9, !dbg !15
  %1267 = shufflevector <8 x half> %1199, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1268 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %1267, <4 x float> %1266) #9, !dbg !15
  %1269 = shufflevector <8 x half> %1201, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1270 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %1269, <4 x float> %1268) #9, !dbg !15
  %1271 = shufflevector <8 x half> %1203, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1272 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %1271, <4 x float> %1270) #9, !dbg !15
  %1273 = shufflevector <8 x half> %1205, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1274 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %1273, <4 x float> %1272) #9, !dbg !15
  %1275 = extractvalue [2 x <8 x float>] %206, 1, !dbg !31
  %1276 = shufflevector <8 x float> %1275, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %1277 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1240, <4 x float> %1276) #9, !dbg !15
  %1278 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1244, <4 x float> %1277) #9, !dbg !15
  %1279 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1246, <4 x float> %1278) #9, !dbg !15
  %1280 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1248, <4 x float> %1279) #9, !dbg !15
  %1281 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %1250, <4 x float> %1280) #9, !dbg !15
  %1282 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %1252, <4 x float> %1281) #9, !dbg !15
  %1283 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %1254, <4 x float> %1282) #9, !dbg !15
  %1284 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %1256, <4 x float> %1283) #9, !dbg !15
  %1285 = shufflevector <8 x float> %1275, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %1286 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1258, <4 x float> %1285) #9, !dbg !15
  %1287 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1261, <4 x float> %1286) #9, !dbg !15
  %1288 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1263, <4 x float> %1287) #9, !dbg !15
  %1289 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1265, <4 x float> %1288) #9, !dbg !15
  %1290 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %1267, <4 x float> %1289) #9, !dbg !15
  %1291 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %1269, <4 x float> %1290) #9, !dbg !15
  %1292 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %1271, <4 x float> %1291) #9, !dbg !15
  %1293 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %1273, <4 x float> %1292) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !19
  fence syncscope("workgroup") release, !dbg !19
  call void @llvm.amdgcn.s.barrier(), !dbg !19
  fence syncscope("workgroup") acquire, !dbg !19
  %1294 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local871, i32 0) #0, !dbg !15
  %1295 = bitcast <4 x i32> %1294 to <8 x half>, !dbg !15
  %1296 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local884, i32 0) #0, !dbg !15
  %1297 = bitcast <4 x i32> %1296 to <8 x half>, !dbg !15
  %1298 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local888, i32 0) #0, !dbg !15
  %1299 = bitcast <4 x i32> %1298 to <8 x half>, !dbg !15
  %1300 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local892, i32 0) #0, !dbg !15
  %1301 = bitcast <4 x i32> %1300 to <8 x half>, !dbg !15
  %1302 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local894, i32 0) #0, !dbg !15
  %1303 = bitcast <4 x i32> %1302 to <8 x half>, !dbg !15
  %1304 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local898, i32 0) #0, !dbg !15
  %1305 = bitcast <4 x i32> %1304 to <8 x half>, !dbg !15
  %1306 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local902, i32 0) #0, !dbg !15
  %1307 = bitcast <4 x i32> %1306 to <8 x half>, !dbg !15
  %1308 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local906, i32 0) #0, !dbg !15
  %1309 = bitcast <4 x i32> %1308 to <8 x half>, !dbg !15
  fence syncscope("workgroup") release, !dbg !19
  call void @llvm.amdgcn.s.barrier(), !dbg !19
  fence syncscope("workgroup") acquire, !dbg !19
  %1310 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %1209, !dbg !19
  %1311 = getelementptr i8, ptr addrspace(1) %1310, i64 128, !dbg !19
  %1312 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local911), !dbg !19
  %1313 = zext i32 %1312 to i64, !dbg !19
  %1314 = inttoptr i64 %1313 to ptr addrspace(3), !dbg !19
  %address.local930 = getelementptr [2 x i8], ptr addrspace(1) %1311, i64 %address.local916, !dbg !19
  %1315 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !19
  %1316 = ptrtoint ptr addrspace(1) %address.local930 to i64, !dbg !19
  %1317 = sub i64 %1316, %1315, !dbg !19
  %dma.buffer.offset1552 = trunc i64 %1317 to i32, !dbg !19
  %1318 = trunc i64 %1315 to i32, !dbg !19
  %1319 = insertelement <4 x i32> poison, i32 %1318, i64 0, !dbg !19
  %1320 = lshr i64 %1315, 32, !dbg !19
  %1321 = trunc i64 %1320 to i32, !dbg !19
  %1322 = insertelement <4 x i32> %1319, i32 %1321, i64 1, !dbg !19
  %1323 = insertelement <4 x i32> %1322, i32 33554432, i64 2, !dbg !19
  %dma.buffer.rsrc1553 = insertelement <4 x i32> %1323, i32 131072, i64 3, !dbg !19
  %1324 = ptrtoint ptr addrspace(3) %1314 to i32, !dbg !19
  %dma.buffer.lds1554 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1324), !dbg !19
  %1325 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1554, i32 %dma.buffer.offset1552, <4 x i32> %dma.buffer.rsrc1553) #10, !dbg !19
  %1326 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local923), !dbg !19
  %1327 = zext i32 %1326 to i64, !dbg !19
  %1328 = inttoptr i64 %1327 to ptr addrspace(3), !dbg !19
  %address.local931 = getelementptr [2 x i8], ptr addrspace(1) %1311, i64 %address.local928, !dbg !19
  %1329 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !19
  %1330 = ptrtoint ptr addrspace(1) %address.local931 to i64, !dbg !19
  %1331 = sub i64 %1330, %1329, !dbg !19
  %dma.buffer.offset1549 = trunc i64 %1331 to i32, !dbg !19
  %1332 = trunc i64 %1329 to i32, !dbg !19
  %1333 = insertelement <4 x i32> poison, i32 %1332, i64 0, !dbg !19
  %1334 = lshr i64 %1329, 32, !dbg !19
  %1335 = trunc i64 %1334 to i32, !dbg !19
  %1336 = insertelement <4 x i32> %1333, i32 %1335, i64 1, !dbg !19
  %1337 = insertelement <4 x i32> %1336, i32 33554432, i64 2, !dbg !19
  %dma.buffer.rsrc1550 = insertelement <4 x i32> %1337, i32 131072, i64 3, !dbg !19
  %1338 = ptrtoint ptr addrspace(3) %1328 to i32, !dbg !19
  %dma.buffer.lds1551 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1338), !dbg !19
  %1339 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1551, i32 %dma.buffer.offset1549, <4 x i32> %dma.buffer.rsrc1550) #10, !dbg !19
  %1340 = shufflevector <8 x half> %1295, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1341 = extractvalue [2 x <8 x float>] %208, 0, !dbg !31
  %1342 = shufflevector <8 x float> %1341, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %1343 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1340, <4 x float> %1342) #9, !dbg !15
  %1344 = shufflevector <8 x half> %1297, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1345 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %1344, <4 x float> %1343) #9, !dbg !15
  %1346 = shufflevector <8 x half> %1299, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1347 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %1346, <4 x float> %1345) #9, !dbg !15
  %1348 = shufflevector <8 x half> %1301, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1349 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %1348, <4 x float> %1347) #9, !dbg !15
  %1350 = shufflevector <8 x half> %1303, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1351 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %1350, <4 x float> %1349) #9, !dbg !15
  %1352 = shufflevector <8 x half> %1305, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1353 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %1352, <4 x float> %1351) #9, !dbg !15
  %1354 = shufflevector <8 x half> %1307, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1355 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %1354, <4 x float> %1353) #9, !dbg !15
  %1356 = shufflevector <8 x half> %1309, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1357 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %1356, <4 x float> %1355) #9, !dbg !15
  %1358 = shufflevector <8 x half> %1295, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1359 = shufflevector <8 x float> %1341, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %1360 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1358, <4 x float> %1359) #9, !dbg !15
  %1361 = shufflevector <8 x half> %1297, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1362 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %1361, <4 x float> %1360) #9, !dbg !15
  %1363 = shufflevector <8 x half> %1299, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1364 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %1363, <4 x float> %1362) #9, !dbg !15
  %1365 = shufflevector <8 x half> %1301, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1366 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %1365, <4 x float> %1364) #9, !dbg !15
  %1367 = shufflevector <8 x half> %1303, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1368 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %1367, <4 x float> %1366) #9, !dbg !15
  %1369 = shufflevector <8 x half> %1305, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1370 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %1369, <4 x float> %1368) #9, !dbg !15
  %1371 = shufflevector <8 x half> %1307, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1372 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %1371, <4 x float> %1370) #9, !dbg !15
  %1373 = shufflevector <8 x half> %1309, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1374 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %1373, <4 x float> %1372) #9, !dbg !15
  %1375 = extractvalue [2 x <8 x float>] %208, 1, !dbg !31
  %1376 = shufflevector <8 x float> %1375, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %1377 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1340, <4 x float> %1376) #9, !dbg !15
  %1378 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1344, <4 x float> %1377) #9, !dbg !15
  %1379 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1346, <4 x float> %1378) #9, !dbg !15
  %1380 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1348, <4 x float> %1379) #9, !dbg !15
  %1381 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %1350, <4 x float> %1380) #9, !dbg !15
  %1382 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %1352, <4 x float> %1381) #9, !dbg !15
  %1383 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %1354, <4 x float> %1382) #9, !dbg !15
  %1384 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %1356, <4 x float> %1383) #9, !dbg !15
  %1385 = shufflevector <8 x float> %1375, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %1386 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1358, <4 x float> %1385) #9, !dbg !15
  %1387 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1361, <4 x float> %1386) #9, !dbg !15
  %1388 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1363, <4 x float> %1387) #9, !dbg !15
  %1389 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1365, <4 x float> %1388) #9, !dbg !15
  %1390 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %1367, <4 x float> %1389) #9, !dbg !15
  %1391 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %1369, <4 x float> %1390) #9, !dbg !15
  %1392 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %1371, <4 x float> %1391) #9, !dbg !15
  %1393 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %1373, <4 x float> %1392) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !19
  fence syncscope("workgroup") release, !dbg !19
  call void @llvm.amdgcn.s.barrier(), !dbg !19
  fence syncscope("workgroup") acquire, !dbg !19
  %1394 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local871, i32 0) #0, !dbg !15
  %1395 = bitcast <4 x i32> %1394 to <8 x half>, !dbg !15
  %1396 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local884, i32 0) #0, !dbg !15
  %1397 = bitcast <4 x i32> %1396 to <8 x half>, !dbg !15
  %1398 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local888, i32 0) #0, !dbg !15
  %1399 = bitcast <4 x i32> %1398 to <8 x half>, !dbg !15
  %1400 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local892, i32 0) #0, !dbg !15
  %1401 = bitcast <4 x i32> %1400 to <8 x half>, !dbg !15
  %1402 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local894, i32 0) #0, !dbg !15
  %1403 = bitcast <4 x i32> %1402 to <8 x half>, !dbg !15
  %1404 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local898, i32 0) #0, !dbg !15
  %1405 = bitcast <4 x i32> %1404 to <8 x half>, !dbg !15
  %1406 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local902, i32 0) #0, !dbg !15
  %1407 = bitcast <4 x i32> %1406 to <8 x half>, !dbg !15
  %1408 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local906, i32 0) #0, !dbg !15
  %1409 = bitcast <4 x i32> %1408 to <8 x half>, !dbg !15
  fence syncscope("workgroup") release, !dbg !19
  call void @llvm.amdgcn.s.barrier(), !dbg !19
  fence syncscope("workgroup") acquire, !dbg !19
  %1410 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %1209, !dbg !19
  %1411 = getelementptr i8, ptr addrspace(1) %1410, i64 192, !dbg !19
  %1412 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local911), !dbg !19
  %1413 = zext i32 %1412 to i64, !dbg !19
  %1414 = inttoptr i64 %1413 to ptr addrspace(3), !dbg !19
  %address.local932 = getelementptr [2 x i8], ptr addrspace(1) %1411, i64 %address.local916, !dbg !19
  %1415 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !19
  %1416 = ptrtoint ptr addrspace(1) %address.local932 to i64, !dbg !19
  %1417 = sub i64 %1416, %1415, !dbg !19
  %dma.buffer.offset1546 = trunc i64 %1417 to i32, !dbg !19
  %1418 = trunc i64 %1415 to i32, !dbg !19
  %1419 = insertelement <4 x i32> poison, i32 %1418, i64 0, !dbg !19
  %1420 = lshr i64 %1415, 32, !dbg !19
  %1421 = trunc i64 %1420 to i32, !dbg !19
  %1422 = insertelement <4 x i32> %1419, i32 %1421, i64 1, !dbg !19
  %1423 = insertelement <4 x i32> %1422, i32 33554432, i64 2, !dbg !19
  %dma.buffer.rsrc1547 = insertelement <4 x i32> %1423, i32 131072, i64 3, !dbg !19
  %1424 = ptrtoint ptr addrspace(3) %1414 to i32, !dbg !19
  %dma.buffer.lds1548 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1424), !dbg !19
  %1425 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1548, i32 %dma.buffer.offset1546, <4 x i32> %dma.buffer.rsrc1547) #10, !dbg !19
  %1426 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local923), !dbg !19
  %1427 = zext i32 %1426 to i64, !dbg !19
  %1428 = inttoptr i64 %1427 to ptr addrspace(3), !dbg !19
  %address.local933 = getelementptr [2 x i8], ptr addrspace(1) %1411, i64 %address.local928, !dbg !19
  %1429 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !19
  %1430 = ptrtoint ptr addrspace(1) %address.local933 to i64, !dbg !19
  %1431 = sub i64 %1430, %1429, !dbg !19
  %dma.buffer.offset = trunc i64 %1431 to i32, !dbg !19
  %1432 = trunc i64 %1429 to i32, !dbg !19
  %1433 = insertelement <4 x i32> poison, i32 %1432, i64 0, !dbg !19
  %1434 = lshr i64 %1429, 32, !dbg !19
  %1435 = trunc i64 %1434 to i32, !dbg !19
  %1436 = insertelement <4 x i32> %1433, i32 %1435, i64 1, !dbg !19
  %1437 = insertelement <4 x i32> %1436, i32 33554432, i64 2, !dbg !19
  %dma.buffer.rsrc = insertelement <4 x i32> %1437, i32 131072, i64 3, !dbg !19
  %1438 = ptrtoint ptr addrspace(3) %1428 to i32, !dbg !19
  %dma.buffer.lds = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1438), !dbg !19
  %1439 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc) #10, !dbg !19
  %1440 = shufflevector <8 x half> %1395, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1441 = extractvalue [2 x <8 x float>] %209, 0, !dbg !31
  %1442 = shufflevector <8 x float> %1441, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %1443 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1440, <4 x float> %1442) #9, !dbg !15
  %1444 = shufflevector <8 x half> %1397, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1445 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %1444, <4 x float> %1443) #9, !dbg !15
  %1446 = shufflevector <8 x half> %1399, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1447 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %1446, <4 x float> %1445) #9, !dbg !15
  %1448 = shufflevector <8 x half> %1401, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1449 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %1448, <4 x float> %1447) #9, !dbg !15
  %1450 = shufflevector <8 x half> %1403, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1451 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %1450, <4 x float> %1449) #9, !dbg !15
  %1452 = shufflevector <8 x half> %1405, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1453 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %1452, <4 x float> %1451) #9, !dbg !15
  %1454 = shufflevector <8 x half> %1407, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1455 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %1454, <4 x float> %1453) #9, !dbg !15
  %1456 = shufflevector <8 x half> %1409, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1457 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %1456, <4 x float> %1455) #9, !dbg !15
  %1458 = shufflevector <8 x half> %1395, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1459 = shufflevector <8 x float> %1441, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %1460 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1458, <4 x float> %1459) #9, !dbg !15
  %1461 = shufflevector <8 x half> %1397, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1462 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %1461, <4 x float> %1460) #9, !dbg !15
  %1463 = shufflevector <8 x half> %1399, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1464 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %1463, <4 x float> %1462) #9, !dbg !15
  %1465 = shufflevector <8 x half> %1401, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1466 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %1465, <4 x float> %1464) #9, !dbg !15
  %1467 = shufflevector <8 x half> %1403, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1468 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %1467, <4 x float> %1466) #9, !dbg !15
  %1469 = shufflevector <8 x half> %1405, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1470 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %1469, <4 x float> %1468) #9, !dbg !15
  %1471 = shufflevector <8 x half> %1407, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1472 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %1471, <4 x float> %1470) #9, !dbg !15
  %1473 = shufflevector <8 x half> %1409, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1474 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %1473, <4 x float> %1472) #9, !dbg !15
  %1475 = extractvalue [2 x <8 x float>] %209, 1, !dbg !31
  %1476 = shufflevector <8 x float> %1475, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %1477 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1440, <4 x float> %1476) #9, !dbg !15
  %1478 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1444, <4 x float> %1477) #9, !dbg !15
  %1479 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1446, <4 x float> %1478) #9, !dbg !15
  %1480 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1448, <4 x float> %1479) #9, !dbg !15
  %1481 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %1450, <4 x float> %1480) #9, !dbg !15
  %1482 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %1452, <4 x float> %1481) #9, !dbg !15
  %1483 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %1454, <4 x float> %1482) #9, !dbg !15
  %1484 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %1456, <4 x float> %1483) #9, !dbg !15
  %1485 = shufflevector <8 x float> %1475, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %1486 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1458, <4 x float> %1485) #9, !dbg !15
  %1487 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1461, <4 x float> %1486) #9, !dbg !15
  %1488 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1463, <4 x float> %1487) #9, !dbg !15
  %1489 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1465, <4 x float> %1488) #9, !dbg !15
  %1490 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %1467, <4 x float> %1489) #9, !dbg !15
  %1491 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %1469, <4 x float> %1490) #9, !dbg !15
  %1492 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %1471, <4 x float> %1491) #9, !dbg !15
  %1493 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %1473, <4 x float> %1492) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !19
  fence syncscope("workgroup") release, !dbg !19
  call void @llvm.amdgcn.s.barrier(), !dbg !19
  fence syncscope("workgroup") acquire, !dbg !19
  %1494 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local871, i32 0) #0, !dbg !15
  %1495 = bitcast <4 x i32> %1494 to <8 x half>, !dbg !15
  %1496 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local884, i32 0) #0, !dbg !15
  %1497 = bitcast <4 x i32> %1496 to <8 x half>, !dbg !15
  %1498 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local888, i32 0) #0, !dbg !15
  %1499 = bitcast <4 x i32> %1498 to <8 x half>, !dbg !15
  %1500 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local892, i32 0) #0, !dbg !15
  %1501 = bitcast <4 x i32> %1500 to <8 x half>, !dbg !15
  %1502 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local894, i32 0) #0, !dbg !15
  %1503 = bitcast <4 x i32> %1502 to <8 x half>, !dbg !15
  %1504 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local898, i32 0) #0, !dbg !15
  %1505 = bitcast <4 x i32> %1504 to <8 x half>, !dbg !15
  %1506 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local902, i32 0) #0, !dbg !15
  %1507 = bitcast <4 x i32> %1506 to <8 x half>, !dbg !15
  %1508 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local906, i32 0) #0, !dbg !15
  %1509 = bitcast <4 x i32> %1508 to <8 x half>, !dbg !15
  %1510 = shufflevector <8 x half> %1495, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1511 = extractvalue [2 x <8 x float>] %210, 0, !dbg !31
  %1512 = shufflevector <8 x float> %1511, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %1513 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1510, <4 x float> %1512) #9, !dbg !15
  %1514 = shufflevector <8 x half> %1497, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1515 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %1514, <4 x float> %1513) #9, !dbg !15
  %1516 = shufflevector <8 x half> %1499, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1517 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %1516, <4 x float> %1515) #9, !dbg !15
  %1518 = shufflevector <8 x half> %1501, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1519 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %1518, <4 x float> %1517) #9, !dbg !15
  %1520 = shufflevector <8 x half> %1503, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1521 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %1520, <4 x float> %1519) #9, !dbg !15
  %1522 = shufflevector <8 x half> %1505, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1523 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %1522, <4 x float> %1521) #9, !dbg !15
  %1524 = shufflevector <8 x half> %1507, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1525 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %1524, <4 x float> %1523) #9, !dbg !15
  %1526 = shufflevector <8 x half> %1509, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1527 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %1526, <4 x float> %1525) #9, !dbg !15
  %1528 = shufflevector <8 x half> %1495, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1529 = shufflevector <8 x float> %1511, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %1530 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1528, <4 x float> %1529) #9, !dbg !15
  %1531 = shufflevector <8 x half> %1497, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1532 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %1531, <4 x float> %1530) #9, !dbg !15
  %1533 = shufflevector <8 x half> %1499, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1534 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %1533, <4 x float> %1532) #9, !dbg !15
  %1535 = shufflevector <8 x half> %1501, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1536 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %1535, <4 x float> %1534) #9, !dbg !15
  %1537 = shufflevector <8 x half> %1503, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1538 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %1537, <4 x float> %1536) #9, !dbg !15
  %1539 = shufflevector <8 x half> %1505, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1540 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %1539, <4 x float> %1538) #9, !dbg !15
  %1541 = shufflevector <8 x half> %1507, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1542 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %1541, <4 x float> %1540) #9, !dbg !15
  %1543 = shufflevector <8 x half> %1509, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1544 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %1543, <4 x float> %1542) #9, !dbg !15
  %1545 = extractvalue [2 x <8 x float>] %210, 1, !dbg !31
  %1546 = shufflevector <8 x float> %1545, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %1547 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1510, <4 x float> %1546) #9, !dbg !15
  %1548 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1514, <4 x float> %1547) #9, !dbg !15
  %1549 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1516, <4 x float> %1548) #9, !dbg !15
  %1550 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1518, <4 x float> %1549) #9, !dbg !15
  %1551 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %1520, <4 x float> %1550) #9, !dbg !15
  %1552 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %1522, <4 x float> %1551) #9, !dbg !15
  %1553 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %1524, <4 x float> %1552) #9, !dbg !15
  %1554 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %1526, <4 x float> %1553) #9, !dbg !15
  %1555 = shufflevector <8 x float> %1545, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %1556 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1528, <4 x float> %1555) #9, !dbg !15
  %1557 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1531, <4 x float> %1556) #9, !dbg !15
  %1558 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1533, <4 x float> %1557) #9, !dbg !15
  %1559 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1535, <4 x float> %1558) #9, !dbg !15
  %1560 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %1537, <4 x float> %1559) #9, !dbg !15
  %1561 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %1539, <4 x float> %1560) #9, !dbg !15
  %1562 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %1541, <4 x float> %1561) #9, !dbg !15
  %1563 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %1543, <4 x float> %1562) #9, !dbg !15
  %1564 = shufflevector <4 x float> %1257, <4 x float> %1274, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1565 = insertvalue [2 x <8 x float>] poison, <8 x float> %1564, 0, !dbg !13
  %1566 = shufflevector <4 x float> %1284, <4 x float> %1293, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1567 = insertvalue [2 x <8 x float>] %1565, <8 x float> %1566, 1, !dbg !13
  %1568 = shufflevector <4 x float> %1357, <4 x float> %1374, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1569 = insertvalue [2 x <8 x float>] poison, <8 x float> %1568, 0, !dbg !13
  %1570 = shufflevector <4 x float> %1384, <4 x float> %1393, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1571 = insertvalue [2 x <8 x float>] %1569, <8 x float> %1570, 1, !dbg !13
  %1572 = shufflevector <4 x float> %1457, <4 x float> %1474, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1573 = insertvalue [2 x <8 x float>] poison, <8 x float> %1572, 0, !dbg !13
  %1574 = shufflevector <4 x float> %1484, <4 x float> %1493, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1575 = insertvalue [2 x <8 x float>] %1573, <8 x float> %1574, 1, !dbg !13
  %1576 = shufflevector <4 x float> %1527, <4 x float> %1544, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1577 = insertvalue [2 x <8 x float>] poison, <8 x float> %1576, 0, !dbg !13
  %1578 = shufflevector <4 x float> %1554, <4 x float> %1563, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1579 = insertvalue [2 x <8 x float>] %1577, <8 x float> %1578, 1, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %1580 = add nuw nsw i64 %205, 1, !dbg !13
  br label %204, !dbg !13

1581:                                             ; preds = %204
  %1582 = and i32 %31, 16, !dbg !13
  %.not = icmp eq i32 %1582, 0, !dbg !13
  %1583 = and i32 %31, 32, !dbg !13
  %.not36 = icmp eq i32 %1583, 0, !dbg !13
  %1584 = extractvalue [2 x <8 x float>] %210, 0, !dbg !13
  %1585 = extractelement <8 x float> %1584, i64 0, !dbg !13
  %1586 = extractelement <8 x float> %1584, i64 1, !dbg !13
  %1587 = extractelement <8 x float> %1584, i64 2, !dbg !13
  %1588 = extractelement <8 x float> %1584, i64 3, !dbg !13
  %1589 = select i1 %.not, float %1586, float %1585, !dbg !13
  %1590 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !13
  %1591 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1590), !dbg !13
  %1592 = add nuw nsw i32 %1591, 64, !dbg !13
  %1593 = xor i32 %1591, 16, !dbg !13
  %1594 = and i32 %1592, 192, !dbg !13
  %1595 = icmp samesign ult i32 %1593, %1594, !dbg !13
  %1596 = select i1 %1595, i32 %1593, i32 %1591, !dbg !13
  %1597 = shl nuw nsw i32 %1596, 2, !dbg !13
  %1598 = bitcast float %1589 to i32, !dbg !13
  %1599 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1597, i32 %1598), !dbg !13
  %1600 = bitcast i32 %1599 to float, !dbg !13
  %1601 = select i1 %.not, float %1585, float %1600, !dbg !13
  %1602 = select i1 %.not, float %1600, float %1586, !dbg !13
  %1603 = select i1 %.not, float %1588, float %1587, !dbg !13
  %1604 = bitcast float %1603 to i32, !dbg !13
  %1605 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1597, i32 %1604), !dbg !13
  %1606 = bitcast i32 %1605 to float, !dbg !13
  %1607 = select i1 %.not, float %1587, float %1606, !dbg !13
  %1608 = select i1 %.not, float %1606, float %1588, !dbg !13
  %1609 = select i1 %.not36, float %1607, float %1601, !dbg !13
  %1610 = xor i32 %1591, 32, !dbg !13
  %1611 = icmp samesign ult i32 %1610, %1594, !dbg !13
  %1612 = select i1 %1611, i32 %1610, i32 %1591, !dbg !13
  %1613 = shl nuw nsw i32 %1612, 2, !dbg !13
  %1614 = bitcast float %1609 to i32, !dbg !13
  %1615 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1613, i32 %1614), !dbg !13
  %1616 = bitcast i32 %1615 to float, !dbg !13
  %1617 = select i1 %.not36, float %1601, float %1616, !dbg !13
  %1618 = select i1 %.not36, float %1616, float %1607, !dbg !13
  %1619 = select i1 %.not36, float %1608, float %1602, !dbg !13
  %1620 = bitcast float %1619 to i32, !dbg !13
  %1621 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1613, i32 %1620), !dbg !13
  %1622 = bitcast i32 %1621 to float, !dbg !13
  %1623 = select i1 %.not36, float %1602, float %1622, !dbg !13
  %1624 = select i1 %.not36, float %1622, float %1608, !dbg !13
  %1625 = extractvalue [2 x <8 x float>] %209, 0, !dbg !13
  %1626 = extractelement <8 x float> %1625, i64 0, !dbg !13
  %1627 = extractelement <8 x float> %1625, i64 1, !dbg !13
  %1628 = extractelement <8 x float> %1625, i64 2, !dbg !13
  %1629 = extractelement <8 x float> %1625, i64 3, !dbg !13
  %1630 = select i1 %.not, float %1627, float %1626, !dbg !13
  %1631 = bitcast float %1630 to i32, !dbg !13
  %1632 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1597, i32 %1631), !dbg !13
  %1633 = bitcast i32 %1632 to float, !dbg !13
  %1634 = select i1 %.not, float %1626, float %1633, !dbg !13
  %1635 = select i1 %.not, float %1633, float %1627, !dbg !13
  %1636 = select i1 %.not, float %1629, float %1628, !dbg !13
  %1637 = bitcast float %1636 to i32, !dbg !13
  %1638 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1597, i32 %1637), !dbg !13
  %1639 = bitcast i32 %1638 to float, !dbg !13
  %1640 = select i1 %.not, float %1628, float %1639, !dbg !13
  %1641 = select i1 %.not, float %1639, float %1629, !dbg !13
  %1642 = select i1 %.not36, float %1640, float %1634, !dbg !13
  %1643 = bitcast float %1642 to i32, !dbg !13
  %1644 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1613, i32 %1643), !dbg !13
  %1645 = bitcast i32 %1644 to float, !dbg !13
  %1646 = select i1 %.not36, float %1634, float %1645, !dbg !13
  %1647 = select i1 %.not36, float %1645, float %1640, !dbg !13
  %1648 = select i1 %.not36, float %1641, float %1635, !dbg !13
  %1649 = bitcast float %1648 to i32, !dbg !13
  %1650 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1613, i32 %1649), !dbg !13
  %1651 = bitcast i32 %1650 to float, !dbg !13
  %1652 = select i1 %.not36, float %1635, float %1651, !dbg !13
  %1653 = select i1 %.not36, float %1651, float %1641, !dbg !13
  %1654 = extractvalue [2 x <8 x float>] %208, 0, !dbg !13
  %1655 = extractelement <8 x float> %1654, i64 0, !dbg !13
  %1656 = extractelement <8 x float> %1654, i64 1, !dbg !13
  %1657 = extractelement <8 x float> %1654, i64 2, !dbg !13
  %1658 = extractelement <8 x float> %1654, i64 3, !dbg !13
  %1659 = select i1 %.not, float %1656, float %1655, !dbg !13
  %1660 = bitcast float %1659 to i32, !dbg !13
  %1661 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1597, i32 %1660), !dbg !13
  %1662 = bitcast i32 %1661 to float, !dbg !13
  %1663 = select i1 %.not, float %1655, float %1662, !dbg !13
  %1664 = select i1 %.not, float %1662, float %1656, !dbg !13
  %1665 = select i1 %.not, float %1658, float %1657, !dbg !13
  %1666 = bitcast float %1665 to i32, !dbg !13
  %1667 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1597, i32 %1666), !dbg !13
  %1668 = bitcast i32 %1667 to float, !dbg !13
  %1669 = select i1 %.not, float %1657, float %1668, !dbg !13
  %1670 = select i1 %.not, float %1668, float %1658, !dbg !13
  %1671 = select i1 %.not36, float %1669, float %1663, !dbg !13
  %1672 = bitcast float %1671 to i32, !dbg !13
  %1673 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1613, i32 %1672), !dbg !13
  %1674 = bitcast i32 %1673 to float, !dbg !13
  %1675 = select i1 %.not36, float %1663, float %1674, !dbg !13
  %1676 = select i1 %.not36, float %1674, float %1669, !dbg !13
  %1677 = select i1 %.not36, float %1670, float %1664, !dbg !13
  %1678 = bitcast float %1677 to i32, !dbg !13
  %1679 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1613, i32 %1678), !dbg !13
  %1680 = bitcast i32 %1679 to float, !dbg !13
  %1681 = select i1 %.not36, float %1664, float %1680, !dbg !13
  %1682 = select i1 %.not36, float %1680, float %1670, !dbg !13
  %1683 = extractvalue [2 x <8 x float>] %206, 0, !dbg !13
  %1684 = extractelement <8 x float> %1683, i64 0, !dbg !13
  %1685 = extractelement <8 x float> %1683, i64 1, !dbg !13
  %1686 = extractelement <8 x float> %1683, i64 2, !dbg !13
  %1687 = extractelement <8 x float> %1683, i64 3, !dbg !13
  %1688 = select i1 %.not, float %1685, float %1684, !dbg !13
  %1689 = bitcast float %1688 to i32, !dbg !13
  %1690 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1597, i32 %1689), !dbg !13
  %1691 = bitcast i32 %1690 to float, !dbg !13
  %1692 = select i1 %.not, float %1684, float %1691, !dbg !13
  %1693 = select i1 %.not, float %1691, float %1685, !dbg !13
  %1694 = select i1 %.not, float %1687, float %1686, !dbg !13
  %1695 = bitcast float %1694 to i32, !dbg !13
  %1696 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1597, i32 %1695), !dbg !13
  %1697 = bitcast i32 %1696 to float, !dbg !13
  %1698 = select i1 %.not, float %1686, float %1697, !dbg !13
  %1699 = select i1 %.not, float %1697, float %1687, !dbg !13
  %1700 = select i1 %.not36, float %1698, float %1692, !dbg !13
  %1701 = bitcast float %1700 to i32, !dbg !13
  %1702 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1613, i32 %1701), !dbg !13
  %1703 = select i1 %.not36, float %1699, float %1693, !dbg !13
  %1704 = bitcast float %1703 to i32, !dbg !13
  %1705 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1613, i32 %1704), !dbg !13
  %1706 = extractvalue [2 x <1 x float>] %207, 0, !dbg !20
  %1707 = shufflevector <1 x float> %1706, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !20
  %1708 = extractelement <8 x float> %1584, i64 4, !dbg !13
  %1709 = extractelement <8 x float> %1584, i64 5, !dbg !13
  %1710 = extractelement <8 x float> %1584, i64 6, !dbg !13
  %1711 = extractelement <8 x float> %1584, i64 7, !dbg !13
  %1712 = select i1 %.not, float %1709, float %1708, !dbg !13
  %1713 = bitcast float %1712 to i32, !dbg !13
  %1714 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1597, i32 %1713), !dbg !13
  %1715 = bitcast i32 %1714 to float, !dbg !13
  %1716 = select i1 %.not, float %1708, float %1715, !dbg !13
  %1717 = select i1 %.not, float %1715, float %1709, !dbg !13
  %1718 = select i1 %.not, float %1711, float %1710, !dbg !13
  %1719 = bitcast float %1718 to i32, !dbg !13
  %1720 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1597, i32 %1719), !dbg !13
  %1721 = bitcast i32 %1720 to float, !dbg !13
  %1722 = select i1 %.not, float %1710, float %1721, !dbg !13
  %1723 = select i1 %.not, float %1721, float %1711, !dbg !13
  %1724 = select i1 %.not36, float %1722, float %1716, !dbg !13
  %1725 = bitcast float %1724 to i32, !dbg !13
  %1726 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1613, i32 %1725), !dbg !13
  %1727 = bitcast i32 %1726 to float, !dbg !13
  %1728 = select i1 %.not36, float %1716, float %1727, !dbg !13
  %1729 = select i1 %.not36, float %1727, float %1722, !dbg !13
  %1730 = select i1 %.not36, float %1723, float %1717, !dbg !13
  %1731 = bitcast float %1730 to i32, !dbg !13
  %1732 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1613, i32 %1731), !dbg !13
  %1733 = bitcast i32 %1732 to float, !dbg !13
  %1734 = select i1 %.not36, float %1717, float %1733, !dbg !13
  %1735 = select i1 %.not36, float %1733, float %1723, !dbg !13
  %1736 = extractelement <8 x float> %1625, i64 4, !dbg !13
  %1737 = extractelement <8 x float> %1625, i64 5, !dbg !13
  %1738 = extractelement <8 x float> %1625, i64 6, !dbg !13
  %1739 = extractelement <8 x float> %1625, i64 7, !dbg !13
  %1740 = select i1 %.not, float %1737, float %1736, !dbg !13
  %1741 = bitcast float %1740 to i32, !dbg !13
  %1742 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1597, i32 %1741), !dbg !13
  %1743 = bitcast i32 %1742 to float, !dbg !13
  %1744 = select i1 %.not, float %1736, float %1743, !dbg !13
  %1745 = select i1 %.not, float %1743, float %1737, !dbg !13
  %1746 = select i1 %.not, float %1739, float %1738, !dbg !13
  %1747 = bitcast float %1746 to i32, !dbg !13
  %1748 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1597, i32 %1747), !dbg !13
  %1749 = bitcast i32 %1748 to float, !dbg !13
  %1750 = select i1 %.not, float %1738, float %1749, !dbg !13
  %1751 = select i1 %.not, float %1749, float %1739, !dbg !13
  %1752 = select i1 %.not36, float %1750, float %1744, !dbg !13
  %1753 = bitcast float %1752 to i32, !dbg !13
  %1754 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1613, i32 %1753), !dbg !13
  %1755 = bitcast i32 %1754 to float, !dbg !13
  %1756 = select i1 %.not36, float %1744, float %1755, !dbg !13
  %1757 = select i1 %.not36, float %1755, float %1750, !dbg !13
  %1758 = select i1 %.not36, float %1751, float %1745, !dbg !13
  %1759 = bitcast float %1758 to i32, !dbg !13
  %1760 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1613, i32 %1759), !dbg !13
  %1761 = bitcast i32 %1760 to float, !dbg !13
  %1762 = select i1 %.not36, float %1745, float %1761, !dbg !13
  %1763 = select i1 %.not36, float %1761, float %1751, !dbg !13
  %1764 = extractelement <8 x float> %1654, i64 4, !dbg !13
  %1765 = extractelement <8 x float> %1654, i64 5, !dbg !13
  %1766 = extractelement <8 x float> %1654, i64 6, !dbg !13
  %1767 = extractelement <8 x float> %1654, i64 7, !dbg !13
  %1768 = select i1 %.not, float %1765, float %1764, !dbg !13
  %1769 = bitcast float %1768 to i32, !dbg !13
  %1770 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1597, i32 %1769), !dbg !13
  %1771 = bitcast i32 %1770 to float, !dbg !13
  %1772 = select i1 %.not, float %1764, float %1771, !dbg !13
  %1773 = select i1 %.not, float %1771, float %1765, !dbg !13
  %1774 = select i1 %.not, float %1767, float %1766, !dbg !13
  %1775 = bitcast float %1774 to i32, !dbg !13
  %1776 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1597, i32 %1775), !dbg !13
  %1777 = bitcast i32 %1776 to float, !dbg !13
  %1778 = select i1 %.not, float %1766, float %1777, !dbg !13
  %1779 = select i1 %.not, float %1777, float %1767, !dbg !13
  %1780 = select i1 %.not36, float %1778, float %1772, !dbg !13
  %1781 = bitcast float %1780 to i32, !dbg !13
  %1782 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1613, i32 %1781), !dbg !13
  %1783 = bitcast i32 %1782 to float, !dbg !13
  %1784 = select i1 %.not36, float %1772, float %1783, !dbg !13
  %1785 = select i1 %.not36, float %1783, float %1778, !dbg !13
  %1786 = select i1 %.not36, float %1779, float %1773, !dbg !13
  %1787 = bitcast float %1786 to i32, !dbg !13
  %1788 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1613, i32 %1787), !dbg !13
  %1789 = bitcast i32 %1788 to float, !dbg !13
  %1790 = select i1 %.not36, float %1773, float %1789, !dbg !13
  %1791 = select i1 %.not36, float %1789, float %1779, !dbg !13
  %1792 = extractelement <8 x float> %1683, i64 4, !dbg !13
  %1793 = extractelement <8 x float> %1683, i64 5, !dbg !13
  %1794 = extractelement <8 x float> %1683, i64 6, !dbg !13
  %1795 = extractelement <8 x float> %1683, i64 7, !dbg !13
  %1796 = select i1 %.not, float %1793, float %1792, !dbg !13
  %1797 = bitcast float %1796 to i32, !dbg !13
  %1798 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1597, i32 %1797), !dbg !13
  %1799 = bitcast i32 %1798 to float, !dbg !13
  %1800 = select i1 %.not, float %1792, float %1799, !dbg !13
  %1801 = select i1 %.not, float %1799, float %1793, !dbg !13
  %1802 = select i1 %.not, float %1795, float %1794, !dbg !13
  %1803 = bitcast float %1802 to i32, !dbg !13
  %1804 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1597, i32 %1803), !dbg !13
  %1805 = bitcast i32 %1804 to float, !dbg !13
  %1806 = select i1 %.not, float %1794, float %1805, !dbg !13
  %1807 = select i1 %.not, float %1805, float %1795, !dbg !13
  %1808 = select i1 %.not36, float %1806, float %1800, !dbg !13
  %1809 = bitcast float %1808 to i32, !dbg !13
  %1810 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1613, i32 %1809), !dbg !13
  %1811 = select i1 %.not36, float %1807, float %1801, !dbg !13
  %1812 = bitcast float %1811 to i32, !dbg !13
  %1813 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1613, i32 %1812), !dbg !13
  %1814 = extractvalue [2 x <8 x float>] %210, 1, !dbg !13
  %1815 = extractelement <8 x float> %1814, i64 0, !dbg !13
  %1816 = extractelement <8 x float> %1814, i64 1, !dbg !13
  %1817 = extractelement <8 x float> %1814, i64 2, !dbg !13
  %1818 = extractelement <8 x float> %1814, i64 3, !dbg !13
  %1819 = select i1 %.not, float %1816, float %1815, !dbg !13
  %1820 = bitcast float %1819 to i32, !dbg !13
  %1821 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1597, i32 %1820), !dbg !13
  %1822 = bitcast i32 %1821 to float, !dbg !13
  %1823 = select i1 %.not, float %1815, float %1822, !dbg !13
  %1824 = select i1 %.not, float %1822, float %1816, !dbg !13
  %1825 = select i1 %.not, float %1818, float %1817, !dbg !13
  %1826 = bitcast float %1825 to i32, !dbg !13
  %1827 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1597, i32 %1826), !dbg !13
  %1828 = bitcast i32 %1827 to float, !dbg !13
  %1829 = select i1 %.not, float %1817, float %1828, !dbg !13
  %1830 = select i1 %.not, float %1828, float %1818, !dbg !13
  %1831 = select i1 %.not36, float %1829, float %1823, !dbg !13
  %1832 = bitcast float %1831 to i32, !dbg !13
  %1833 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1613, i32 %1832), !dbg !13
  %1834 = bitcast i32 %1833 to float, !dbg !13
  %1835 = select i1 %.not36, float %1823, float %1834, !dbg !13
  %1836 = select i1 %.not36, float %1834, float %1829, !dbg !13
  %1837 = select i1 %.not36, float %1830, float %1824, !dbg !13
  %1838 = bitcast float %1837 to i32, !dbg !13
  %1839 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1613, i32 %1838), !dbg !13
  %1840 = bitcast i32 %1839 to float, !dbg !13
  %1841 = select i1 %.not36, float %1824, float %1840, !dbg !13
  %1842 = select i1 %.not36, float %1840, float %1830, !dbg !13
  %1843 = extractvalue [2 x <8 x float>] %209, 1, !dbg !13
  %1844 = extractelement <8 x float> %1843, i64 0, !dbg !13
  %1845 = extractelement <8 x float> %1843, i64 1, !dbg !13
  %1846 = extractelement <8 x float> %1843, i64 2, !dbg !13
  %1847 = extractelement <8 x float> %1843, i64 3, !dbg !13
  %1848 = select i1 %.not, float %1845, float %1844, !dbg !13
  %1849 = bitcast float %1848 to i32, !dbg !13
  %1850 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1597, i32 %1849), !dbg !13
  %1851 = bitcast i32 %1850 to float, !dbg !13
  %1852 = select i1 %.not, float %1844, float %1851, !dbg !13
  %1853 = select i1 %.not, float %1851, float %1845, !dbg !13
  %1854 = select i1 %.not, float %1847, float %1846, !dbg !13
  %1855 = bitcast float %1854 to i32, !dbg !13
  %1856 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1597, i32 %1855), !dbg !13
  %1857 = bitcast i32 %1856 to float, !dbg !13
  %1858 = select i1 %.not, float %1846, float %1857, !dbg !13
  %1859 = select i1 %.not, float %1857, float %1847, !dbg !13
  %1860 = select i1 %.not36, float %1858, float %1852, !dbg !13
  %1861 = bitcast float %1860 to i32, !dbg !13
  %1862 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1613, i32 %1861), !dbg !13
  %1863 = bitcast i32 %1862 to float, !dbg !13
  %1864 = select i1 %.not36, float %1852, float %1863, !dbg !13
  %1865 = select i1 %.not36, float %1863, float %1858, !dbg !13
  %1866 = select i1 %.not36, float %1859, float %1853, !dbg !13
  %1867 = bitcast float %1866 to i32, !dbg !13
  %1868 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1613, i32 %1867), !dbg !13
  %1869 = bitcast i32 %1868 to float, !dbg !13
  %1870 = select i1 %.not36, float %1853, float %1869, !dbg !13
  %1871 = select i1 %.not36, float %1869, float %1859, !dbg !13
  %1872 = extractvalue [2 x <8 x float>] %208, 1, !dbg !13
  %1873 = extractelement <8 x float> %1872, i64 0, !dbg !13
  %1874 = extractelement <8 x float> %1872, i64 1, !dbg !13
  %1875 = extractelement <8 x float> %1872, i64 2, !dbg !13
  %1876 = extractelement <8 x float> %1872, i64 3, !dbg !13
  %1877 = select i1 %.not, float %1874, float %1873, !dbg !13
  %1878 = bitcast float %1877 to i32, !dbg !13
  %1879 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1597, i32 %1878), !dbg !13
  %1880 = bitcast i32 %1879 to float, !dbg !13
  %1881 = select i1 %.not, float %1873, float %1880, !dbg !13
  %1882 = select i1 %.not, float %1880, float %1874, !dbg !13
  %1883 = select i1 %.not, float %1876, float %1875, !dbg !13
  %1884 = bitcast float %1883 to i32, !dbg !13
  %1885 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1597, i32 %1884), !dbg !13
  %1886 = bitcast i32 %1885 to float, !dbg !13
  %1887 = select i1 %.not, float %1875, float %1886, !dbg !13
  %1888 = select i1 %.not, float %1886, float %1876, !dbg !13
  %1889 = select i1 %.not36, float %1887, float %1881, !dbg !13
  %1890 = bitcast float %1889 to i32, !dbg !13
  %1891 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1613, i32 %1890), !dbg !13
  %1892 = bitcast i32 %1891 to float, !dbg !13
  %1893 = select i1 %.not36, float %1881, float %1892, !dbg !13
  %1894 = select i1 %.not36, float %1892, float %1887, !dbg !13
  %1895 = select i1 %.not36, float %1888, float %1882, !dbg !13
  %1896 = bitcast float %1895 to i32, !dbg !13
  %1897 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1613, i32 %1896), !dbg !13
  %1898 = bitcast i32 %1897 to float, !dbg !13
  %1899 = select i1 %.not36, float %1882, float %1898, !dbg !13
  %1900 = select i1 %.not36, float %1898, float %1888, !dbg !13
  %1901 = extractvalue [2 x <8 x float>] %206, 1, !dbg !13
  %1902 = extractelement <8 x float> %1901, i64 0, !dbg !13
  %1903 = extractelement <8 x float> %1901, i64 1, !dbg !13
  %1904 = extractelement <8 x float> %1901, i64 2, !dbg !13
  %1905 = extractelement <8 x float> %1901, i64 3, !dbg !13
  %1906 = select i1 %.not, float %1903, float %1902, !dbg !13
  %1907 = bitcast float %1906 to i32, !dbg !13
  %1908 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1597, i32 %1907), !dbg !13
  %1909 = bitcast i32 %1908 to float, !dbg !13
  %1910 = select i1 %.not, float %1902, float %1909, !dbg !13
  %1911 = select i1 %.not, float %1909, float %1903, !dbg !13
  %1912 = select i1 %.not, float %1905, float %1904, !dbg !13
  %1913 = bitcast float %1912 to i32, !dbg !13
  %1914 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1597, i32 %1913), !dbg !13
  %1915 = bitcast i32 %1914 to float, !dbg !13
  %1916 = select i1 %.not, float %1904, float %1915, !dbg !13
  %1917 = select i1 %.not, float %1915, float %1905, !dbg !13
  %1918 = select i1 %.not36, float %1916, float %1910, !dbg !13
  %1919 = bitcast float %1918 to i32, !dbg !13
  %1920 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1613, i32 %1919), !dbg !13
  %1921 = select i1 %.not36, float %1917, float %1911, !dbg !13
  %1922 = bitcast float %1921 to i32, !dbg !13
  %1923 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1613, i32 %1922), !dbg !13
  %1924 = extractvalue [2 x <1 x float>] %207, 1, !dbg !20
  %1925 = shufflevector <1 x float> %1924, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !20
  %1926 = extractelement <8 x float> %1814, i64 4, !dbg !13
  %1927 = extractelement <8 x float> %1814, i64 5, !dbg !13
  %1928 = extractelement <8 x float> %1814, i64 6, !dbg !13
  %1929 = extractelement <8 x float> %1814, i64 7, !dbg !13
  %1930 = select i1 %.not, float %1927, float %1926, !dbg !13
  %1931 = bitcast float %1930 to i32, !dbg !13
  %1932 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1597, i32 %1931), !dbg !13
  %1933 = bitcast i32 %1932 to float, !dbg !13
  %1934 = select i1 %.not, float %1926, float %1933, !dbg !13
  %1935 = select i1 %.not, float %1933, float %1927, !dbg !13
  %1936 = select i1 %.not, float %1929, float %1928, !dbg !13
  %1937 = bitcast float %1936 to i32, !dbg !13
  %1938 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1597, i32 %1937), !dbg !13
  %1939 = bitcast i32 %1938 to float, !dbg !13
  %1940 = select i1 %.not, float %1928, float %1939, !dbg !13
  %1941 = select i1 %.not, float %1939, float %1929, !dbg !13
  %1942 = select i1 %.not36, float %1940, float %1934, !dbg !13
  %1943 = bitcast float %1942 to i32, !dbg !13
  %1944 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1613, i32 %1943), !dbg !13
  %1945 = bitcast i32 %1944 to float, !dbg !13
  %1946 = select i1 %.not36, float %1934, float %1945, !dbg !13
  %1947 = select i1 %.not36, float %1945, float %1940, !dbg !13
  %1948 = select i1 %.not36, float %1941, float %1935, !dbg !13
  %1949 = bitcast float %1948 to i32, !dbg !13
  %1950 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1613, i32 %1949), !dbg !13
  %1951 = bitcast i32 %1950 to float, !dbg !13
  %1952 = select i1 %.not36, float %1935, float %1951, !dbg !13
  %1953 = select i1 %.not36, float %1951, float %1941, !dbg !13
  %1954 = extractelement <8 x float> %1843, i64 4, !dbg !13
  %1955 = extractelement <8 x float> %1843, i64 5, !dbg !13
  %1956 = extractelement <8 x float> %1843, i64 6, !dbg !13
  %1957 = extractelement <8 x float> %1843, i64 7, !dbg !13
  %1958 = select i1 %.not, float %1955, float %1954, !dbg !13
  %1959 = bitcast float %1958 to i32, !dbg !13
  %1960 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1597, i32 %1959), !dbg !13
  %1961 = bitcast i32 %1960 to float, !dbg !13
  %1962 = select i1 %.not, float %1954, float %1961, !dbg !13
  %1963 = select i1 %.not, float %1961, float %1955, !dbg !13
  %1964 = select i1 %.not, float %1957, float %1956, !dbg !13
  %1965 = bitcast float %1964 to i32, !dbg !13
  %1966 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1597, i32 %1965), !dbg !13
  %1967 = bitcast i32 %1966 to float, !dbg !13
  %1968 = select i1 %.not, float %1956, float %1967, !dbg !13
  %1969 = select i1 %.not, float %1967, float %1957, !dbg !13
  %1970 = select i1 %.not36, float %1968, float %1962, !dbg !13
  %1971 = bitcast float %1970 to i32, !dbg !13
  %1972 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1613, i32 %1971), !dbg !13
  %1973 = bitcast i32 %1972 to float, !dbg !13
  %1974 = select i1 %.not36, float %1962, float %1973, !dbg !13
  %1975 = select i1 %.not36, float %1973, float %1968, !dbg !13
  %1976 = select i1 %.not36, float %1969, float %1963, !dbg !13
  %1977 = bitcast float %1976 to i32, !dbg !13
  %1978 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1613, i32 %1977), !dbg !13
  %1979 = bitcast i32 %1978 to float, !dbg !13
  %1980 = select i1 %.not36, float %1963, float %1979, !dbg !13
  %1981 = select i1 %.not36, float %1979, float %1969, !dbg !13
  %1982 = extractelement <8 x float> %1872, i64 4, !dbg !13
  %1983 = extractelement <8 x float> %1872, i64 5, !dbg !13
  %1984 = extractelement <8 x float> %1872, i64 6, !dbg !13
  %1985 = extractelement <8 x float> %1872, i64 7, !dbg !13
  %1986 = select i1 %.not, float %1983, float %1982, !dbg !13
  %1987 = bitcast float %1986 to i32, !dbg !13
  %1988 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1597, i32 %1987), !dbg !13
  %1989 = bitcast i32 %1988 to float, !dbg !13
  %1990 = select i1 %.not, float %1982, float %1989, !dbg !13
  %1991 = select i1 %.not, float %1989, float %1983, !dbg !13
  %1992 = select i1 %.not, float %1985, float %1984, !dbg !13
  %1993 = bitcast float %1992 to i32, !dbg !13
  %1994 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1597, i32 %1993), !dbg !13
  %1995 = bitcast i32 %1994 to float, !dbg !13
  %1996 = select i1 %.not, float %1984, float %1995, !dbg !13
  %1997 = select i1 %.not, float %1995, float %1985, !dbg !13
  %1998 = select i1 %.not36, float %1996, float %1990, !dbg !13
  %1999 = bitcast float %1998 to i32, !dbg !13
  %2000 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1613, i32 %1999), !dbg !13
  %2001 = bitcast i32 %2000 to float, !dbg !13
  %2002 = select i1 %.not36, float %1990, float %2001, !dbg !13
  %2003 = select i1 %.not36, float %2001, float %1996, !dbg !13
  %2004 = select i1 %.not36, float %1997, float %1991, !dbg !13
  %2005 = bitcast float %2004 to i32, !dbg !13
  %2006 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1613, i32 %2005), !dbg !13
  %2007 = bitcast i32 %2006 to float, !dbg !13
  %2008 = select i1 %.not36, float %1991, float %2007, !dbg !13
  %2009 = select i1 %.not36, float %2007, float %1997, !dbg !13
  %2010 = extractelement <8 x float> %1901, i64 4, !dbg !13
  %2011 = extractelement <8 x float> %1901, i64 5, !dbg !13
  %2012 = extractelement <8 x float> %1901, i64 6, !dbg !13
  %2013 = extractelement <8 x float> %1901, i64 7, !dbg !13
  %2014 = select i1 %.not, float %2011, float %2010, !dbg !13
  %2015 = bitcast float %2014 to i32, !dbg !13
  %2016 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1597, i32 %2015), !dbg !13
  %2017 = bitcast i32 %2016 to float, !dbg !13
  %2018 = select i1 %.not, float %2010, float %2017, !dbg !13
  %2019 = select i1 %.not, float %2017, float %2011, !dbg !13
  %2020 = select i1 %.not, float %2013, float %2012, !dbg !13
  %2021 = bitcast float %2020 to i32, !dbg !13
  %2022 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1597, i32 %2021), !dbg !13
  %2023 = bitcast i32 %2022 to float, !dbg !13
  %2024 = select i1 %.not, float %2012, float %2023, !dbg !13
  %2025 = select i1 %.not, float %2023, float %2013, !dbg !13
  %2026 = select i1 %.not36, float %2024, float %2018, !dbg !13
  %2027 = bitcast float %2026 to i32, !dbg !13
  %2028 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1613, i32 %2027), !dbg !13
  %2029 = select i1 %.not36, float %2025, float %2019, !dbg !13
  %2030 = bitcast float %2029 to i32, !dbg !13
  %2031 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1613, i32 %2030), !dbg !13
  br i1 true, label %2032, label %2096, !dbg !22

2032:                                             ; preds = %1581
  %2033 = bitcast i32 %2028 to float, !dbg !13
  %2034 = select i1 %.not36, float %2018, float %2033, !dbg !13
  %2035 = insertelement <4 x float> poison, float %2034, i64 0, !dbg !20
  %2036 = bitcast i32 %2031 to float, !dbg !13
  %2037 = select i1 %.not36, float %2019, float %2036, !dbg !13
  %2038 = insertelement <4 x float> %2035, float %2037, i64 1, !dbg !20
  %2039 = select i1 %.not36, float %2033, float %2024, !dbg !13
  %2040 = insertelement <4 x float> %2038, float %2039, i64 2, !dbg !20
  %2041 = select i1 %.not36, float %2036, float %2025, !dbg !13
  %2042 = insertelement <4 x float> %2040, float %2041, i64 3, !dbg !20
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1925, !dbg !20
  %normalized = fmul <4 x float> %2042, %row.reciprocal, !dbg !20
  %2043 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !21
  %2044 = extractelement <4 x half> %2043, i64 3, !dbg !21
  %2045 = extractelement <4 x half> %2043, i64 2, !dbg !21
  %2046 = extractelement <4 x half> %2043, i64 1, !dbg !21
  %2047 = extractelement <4 x half> %2043, i64 0, !dbg !21
  %2048 = bitcast i32 %1920 to float, !dbg !13
  %2049 = select i1 %.not36, float %1910, float %2048, !dbg !13
  %2050 = insertelement <4 x float> poison, float %2049, i64 0, !dbg !20
  %2051 = bitcast i32 %1923 to float, !dbg !13
  %2052 = select i1 %.not36, float %1911, float %2051, !dbg !13
  %2053 = insertelement <4 x float> %2050, float %2052, i64 1, !dbg !20
  %2054 = select i1 %.not36, float %2048, float %1916, !dbg !13
  %2055 = insertelement <4 x float> %2053, float %2054, i64 2, !dbg !20
  %2056 = select i1 %.not36, float %2051, float %1917, !dbg !13
  %2057 = insertelement <4 x float> %2055, float %2056, i64 3, !dbg !20
  %normalized104 = fmul <4 x float> %2057, %row.reciprocal, !dbg !20
  %2058 = fptrunc <4 x float> %normalized104 to <4 x half>, !dbg !21
  %2059 = extractelement <4 x half> %2058, i64 3, !dbg !21
  %2060 = extractelement <4 x half> %2058, i64 2, !dbg !21
  %2061 = extractelement <4 x half> %2058, i64 1, !dbg !21
  %2062 = extractelement <4 x half> %2058, i64 0, !dbg !21
  %2063 = bitcast i32 %1810 to float, !dbg !13
  %2064 = select i1 %.not36, float %1800, float %2063, !dbg !13
  %2065 = insertelement <4 x float> poison, float %2064, i64 0, !dbg !20
  %2066 = bitcast i32 %1813 to float, !dbg !13
  %2067 = select i1 %.not36, float %1801, float %2066, !dbg !13
  %2068 = insertelement <4 x float> %2065, float %2067, i64 1, !dbg !20
  %2069 = select i1 %.not36, float %2063, float %1806, !dbg !13
  %2070 = insertelement <4 x float> %2068, float %2069, i64 2, !dbg !20
  %2071 = select i1 %.not36, float %2066, float %1807, !dbg !13
  %2072 = insertelement <4 x float> %2070, float %2071, i64 3, !dbg !20
  %row.reciprocal105 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1707, !dbg !20
  %normalized106 = fmul <4 x float> %2072, %row.reciprocal105, !dbg !20
  %2073 = fptrunc <4 x float> %normalized106 to <4 x half>, !dbg !21
  %2074 = extractelement <4 x half> %2073, i64 3, !dbg !21
  %2075 = extractelement <4 x half> %2073, i64 2, !dbg !21
  %2076 = extractelement <4 x half> %2073, i64 1, !dbg !21
  %2077 = extractelement <4 x half> %2073, i64 0, !dbg !21
  %2078 = bitcast i32 %1702 to float, !dbg !13
  %2079 = select i1 %.not36, float %1692, float %2078, !dbg !13
  %2080 = insertelement <4 x float> poison, float %2079, i64 0, !dbg !20
  %2081 = bitcast i32 %1705 to float, !dbg !13
  %2082 = select i1 %.not36, float %1693, float %2081, !dbg !13
  %2083 = insertelement <4 x float> %2080, float %2082, i64 1, !dbg !20
  %2084 = select i1 %.not36, float %2078, float %1698, !dbg !13
  %2085 = insertelement <4 x float> %2083, float %2084, i64 2, !dbg !20
  %2086 = select i1 %.not36, float %2081, float %1699, !dbg !13
  %2087 = insertelement <4 x float> %2085, float %2086, i64 3, !dbg !20
  %normalized107 = fmul <4 x float> %2087, %row.reciprocal105, !dbg !20
  %2088 = fptrunc <4 x float> %normalized107 to <4 x half>, !dbg !21
  %2089 = extractelement <4 x half> %2088, i64 3, !dbg !21
  %2090 = extractelement <4 x half> %2088, i64 2, !dbg !21
  %2091 = extractelement <4 x half> %2088, i64 1, !dbg !21
  %2092 = extractelement <4 x half> %2088, i64 0, !dbg !21
  %2093 = add nsw i64 %16, %22, !dbg !22
  %2094 = shl i64 %2093, 19, !dbg !22
  %2095 = and i64 %2094, 16252928, !dbg !22
  %address.thread934 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !22, !range !34, !frisk.register_lifetimes !1
  %address.local935 = zext nneg i32 %address.thread934 to i64, !dbg !8
  %address.local936 = add i64 %25, %address.local935, !dbg !22
  %address.local937 = lshr i64 %address.local935, 6, !dbg !11
  %address.local938 = shl nuw nsw i64 %address.local937, 4, !dbg !22
  %address.local939 = add i64 %address.local936, %address.local938, !dbg !22
  %address.local940 = and i64 %address.local935, 240, !dbg !22
  %address.local941 = sub i64 %address.local939, %address.local940, !dbg !22
  %address.local942 = add i64 %address.local941, %26, !dbg !22
  %address.local943 = shl nuw nsw i64 %address.local942, 7, !dbg !22
  %address.local944 = add nuw nsw i64 %2095, %address.local943, !dbg !22
  %address.local945 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local944, !dbg !22
  %address.local946 = lshr i64 %address.local935, 2, !dbg !11
  %address.local947 = and i64 %address.local946, 12, !dbg !11
  %address.local948 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local947, !dbg !22
  store half %2092, ptr addrspace(1) %address.local948, align 2, !dbg !22
  %address.local949 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local944, !dbg !22
  %address.local950 = or disjoint i64 %address.local947, 1, !dbg !22
  %address.local951 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local949, i64 %address.local950, !dbg !22
  store half %2091, ptr addrspace(1) %address.local951, align 2, !dbg !22
  %address.local952 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local944, !dbg !22
  %address.local953 = or disjoint i64 %address.local947, 2, !dbg !22
  %address.local954 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local952, i64 %address.local953, !dbg !22
  store half %2090, ptr addrspace(1) %address.local954, align 2, !dbg !22
  %address.local955 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local944, !dbg !22
  %address.local956 = or disjoint i64 %address.local947, 3, !dbg !22
  %address.local957 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local955, i64 %address.local956, !dbg !22
  store half %2089, ptr addrspace(1) %address.local957, align 2, !dbg !22
  %address.local958 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local944, !dbg !22
  %address.local959 = or disjoint i64 %address.local947, 16, !dbg !22
  %address.local960 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local958, i64 %address.local959, !dbg !22
  store half %2077, ptr addrspace(1) %address.local960, align 2, !dbg !22
  %address.local961 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local944, !dbg !22
  %address.local962 = or disjoint i64 %address.local947, 17, !dbg !22
  %address.local963 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local961, i64 %address.local962, !dbg !22
  store half %2076, ptr addrspace(1) %address.local963, align 2, !dbg !22
  %address.local964 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local944, !dbg !22
  %address.local965 = or disjoint i64 %address.local947, 18, !dbg !22
  %address.local966 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local964, i64 %address.local965, !dbg !22
  store half %2075, ptr addrspace(1) %address.local966, align 2, !dbg !22
  %address.local967 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local944, !dbg !22
  %address.local968 = or disjoint i64 %address.local947, 19, !dbg !22
  %address.local969 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local967, i64 %address.local968, !dbg !22
  store half %2074, ptr addrspace(1) %address.local969, align 2, !dbg !22
  %address.local970 = shl i64 %address.local942, 7, !dbg !22
  %address.local971 = add i64 %address.local970, 8192, !dbg !22
  %address.local972 = add nuw nsw i64 %2095, %address.local971, !dbg !22
  %address.local973 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local972, !dbg !22
  %address.local974 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local947, !dbg !22
  store half %2062, ptr addrspace(1) %address.local974, align 2, !dbg !22
  %address.local975 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local972, !dbg !22
  %address.local976 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local975, i64 %address.local950, !dbg !22
  store half %2061, ptr addrspace(1) %address.local976, align 2, !dbg !22
  %address.local977 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local972, !dbg !22
  %address.local978 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local977, i64 %address.local953, !dbg !22
  store half %2060, ptr addrspace(1) %address.local978, align 2, !dbg !22
  %address.local979 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local972, !dbg !22
  %address.local980 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local979, i64 %address.local956, !dbg !22
  store half %2059, ptr addrspace(1) %address.local980, align 2, !dbg !22
  %address.local981 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local972, !dbg !22
  %address.local982 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local981, i64 %address.local959, !dbg !22
  store half %2047, ptr addrspace(1) %address.local982, align 2, !dbg !22
  %address.local983 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local972, !dbg !22
  %address.local984 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local983, i64 %address.local962, !dbg !22
  store half %2046, ptr addrspace(1) %address.local984, align 2, !dbg !22
  %address.local985 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local972, !dbg !22
  %address.local986 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local965, !dbg !22
  store half %2045, ptr addrspace(1) %address.local986, align 2, !dbg !22
  %address.local987 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local972, !dbg !22
  %address.local988 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local987, i64 %address.local968, !dbg !22
  store half %2044, ptr addrspace(1) %address.local988, align 2, !dbg !22
  br label %2096, !dbg !22

2096:                                             ; preds = %2032, %1581
  br i1 true, label %2097, label %2135, !dbg !22

2097:                                             ; preds = %2096
  %2098 = insertelement <4 x float> poison, float %2002, i64 0, !dbg !20
  %2099 = insertelement <4 x float> %2098, float %2008, i64 1, !dbg !20
  %2100 = insertelement <4 x float> %2099, float %2003, i64 2, !dbg !20
  %2101 = insertelement <4 x float> %2100, float %2009, i64 3, !dbg !20
  %row.reciprocal108 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1925, !dbg !20
  %normalized109 = fmul <4 x float> %2101, %row.reciprocal108, !dbg !20
  %2102 = fptrunc <4 x float> %normalized109 to <4 x half>, !dbg !21
  %2103 = extractelement <4 x half> %2102, i64 3, !dbg !21
  %2104 = extractelement <4 x half> %2102, i64 2, !dbg !21
  %2105 = extractelement <4 x half> %2102, i64 1, !dbg !21
  %2106 = extractelement <4 x half> %2102, i64 0, !dbg !21
  %2107 = insertelement <4 x float> poison, float %1893, i64 0, !dbg !20
  %2108 = insertelement <4 x float> %2107, float %1899, i64 1, !dbg !20
  %2109 = insertelement <4 x float> %2108, float %1894, i64 2, !dbg !20
  %2110 = insertelement <4 x float> %2109, float %1900, i64 3, !dbg !20
  %normalized110 = fmul <4 x float> %2110, %row.reciprocal108, !dbg !20
  %2111 = fptrunc <4 x float> %normalized110 to <4 x half>, !dbg !21
  %2112 = extractelement <4 x half> %2111, i64 3, !dbg !21
  %2113 = extractelement <4 x half> %2111, i64 2, !dbg !21
  %2114 = extractelement <4 x half> %2111, i64 1, !dbg !21
  %2115 = extractelement <4 x half> %2111, i64 0, !dbg !21
  %2116 = insertelement <4 x float> poison, float %1784, i64 0, !dbg !20
  %2117 = insertelement <4 x float> %2116, float %1790, i64 1, !dbg !20
  %2118 = insertelement <4 x float> %2117, float %1785, i64 2, !dbg !20
  %2119 = insertelement <4 x float> %2118, float %1791, i64 3, !dbg !20
  %row.reciprocal111 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1707, !dbg !20
  %normalized112 = fmul <4 x float> %2119, %row.reciprocal111, !dbg !20
  %2120 = fptrunc <4 x float> %normalized112 to <4 x half>, !dbg !21
  %2121 = extractelement <4 x half> %2120, i64 3, !dbg !21
  %2122 = extractelement <4 x half> %2120, i64 2, !dbg !21
  %2123 = extractelement <4 x half> %2120, i64 1, !dbg !21
  %2124 = extractelement <4 x half> %2120, i64 0, !dbg !21
  %2125 = insertelement <4 x float> poison, float %1675, i64 0, !dbg !20
  %2126 = insertelement <4 x float> %2125, float %1681, i64 1, !dbg !20
  %2127 = insertelement <4 x float> %2126, float %1676, i64 2, !dbg !20
  %2128 = insertelement <4 x float> %2127, float %1682, i64 3, !dbg !20
  %normalized113 = fmul <4 x float> %2128, %row.reciprocal111, !dbg !20
  %2129 = fptrunc <4 x float> %normalized113 to <4 x half>, !dbg !21
  %2130 = extractelement <4 x half> %2129, i64 3, !dbg !21
  %2131 = extractelement <4 x half> %2129, i64 2, !dbg !21
  %2132 = extractelement <4 x half> %2129, i64 1, !dbg !21
  %2133 = extractelement <4 x half> %2129, i64 0, !dbg !21
  %2134 = shl nuw nsw i64 %24, 19, !dbg !22
  %address.thread989 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !22, !range !34, !frisk.register_lifetimes !1
  %address.local990 = zext nneg i32 %address.thread989 to i64, !dbg !8
  %address.local991 = add i64 %25, %address.local990, !dbg !22
  %address.local992 = lshr i64 %address.local990, 6, !dbg !11
  %address.local993 = shl nuw nsw i64 %address.local992, 4, !dbg !22
  %address.local994 = add i64 %address.local991, %address.local993, !dbg !22
  %address.local995 = and i64 %address.local990, 240, !dbg !22
  %address.local996 = sub i64 %address.local994, %address.local995, !dbg !22
  %address.local997 = add i64 %address.local996, %26, !dbg !22
  %address.local998 = shl nuw nsw i64 %address.local997, 7, !dbg !22
  %address.local999 = add nuw nsw i64 %2134, %address.local998, !dbg !22
  %address.local1000 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local999, !dbg !22
  %address.local1001 = lshr i64 %address.local990, 2, !dbg !11
  %address.local1002 = and i64 %address.local1001, 12, !dbg !11
  %address.local1003 = or disjoint i64 %address.local1002, 32, !dbg !22
  %address.local1004 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1000, i64 %address.local1003, !dbg !22
  store half %2133, ptr addrspace(1) %address.local1004, align 2, !dbg !22
  %address.local1005 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local999, !dbg !22
  %address.local1006 = or disjoint i64 %address.local1002, 33, !dbg !22
  %address.local1007 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1005, i64 %address.local1006, !dbg !22
  store half %2132, ptr addrspace(1) %address.local1007, align 2, !dbg !22
  %address.local1008 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local999, !dbg !22
  %address.local1009 = or disjoint i64 %address.local1002, 34, !dbg !22
  %address.local1010 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1008, i64 %address.local1009, !dbg !22
  store half %2131, ptr addrspace(1) %address.local1010, align 2, !dbg !22
  %address.local1011 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local999, !dbg !22
  %address.local1012 = or disjoint i64 %address.local1002, 35, !dbg !22
  %address.local1013 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1011, i64 %address.local1012, !dbg !22
  store half %2130, ptr addrspace(1) %address.local1013, align 2, !dbg !22
  %address.local1014 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local999, !dbg !22
  %address.local1015 = or disjoint i64 %address.local1002, 48, !dbg !22
  %address.local1016 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1014, i64 %address.local1015, !dbg !22
  store half %2124, ptr addrspace(1) %address.local1016, align 2, !dbg !22
  %address.local1017 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local999, !dbg !22
  %address.local1018 = or disjoint i64 %address.local1002, 49, !dbg !22
  %address.local1019 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1017, i64 %address.local1018, !dbg !22
  store half %2123, ptr addrspace(1) %address.local1019, align 2, !dbg !22
  %address.local1020 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local999, !dbg !22
  %address.local1021 = or disjoint i64 %address.local1002, 50, !dbg !22
  %address.local1022 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1020, i64 %address.local1021, !dbg !22
  store half %2122, ptr addrspace(1) %address.local1022, align 2, !dbg !22
  %address.local1023 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local999, !dbg !22
  %address.local1024 = or i64 %address.local1001, 51, !dbg !22
  %address.local1025 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1023, i64 %address.local1024, !dbg !22
  store half %2121, ptr addrspace(1) %address.local1025, align 2, !dbg !22
  %address.local1026 = shl i64 %address.local997, 7, !dbg !22
  %address.local1027 = add i64 %address.local1026, 8192, !dbg !22
  %address.local1028 = add nuw nsw i64 %2134, %address.local1027, !dbg !22
  %address.local1029 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1028, !dbg !22
  %address.local1030 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1029, i64 %address.local1003, !dbg !22
  store half %2115, ptr addrspace(1) %address.local1030, align 2, !dbg !22
  %address.local1031 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1028, !dbg !22
  %address.local1032 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1031, i64 %address.local1006, !dbg !22
  store half %2114, ptr addrspace(1) %address.local1032, align 2, !dbg !22
  %address.local1033 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1028, !dbg !22
  %address.local1034 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1033, i64 %address.local1009, !dbg !22
  store half %2113, ptr addrspace(1) %address.local1034, align 2, !dbg !22
  %address.local1035 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1028, !dbg !22
  %address.local1036 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1035, i64 %address.local1012, !dbg !22
  store half %2112, ptr addrspace(1) %address.local1036, align 2, !dbg !22
  %address.local1037 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1028, !dbg !22
  %address.local1038 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1037, i64 %address.local1015, !dbg !22
  store half %2106, ptr addrspace(1) %address.local1038, align 2, !dbg !22
  %address.local1039 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1028, !dbg !22
  %address.local1040 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1039, i64 %address.local1018, !dbg !22
  store half %2105, ptr addrspace(1) %address.local1040, align 2, !dbg !22
  %address.local1041 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1028, !dbg !22
  %address.local1042 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1041, i64 %address.local1021, !dbg !22
  store half %2104, ptr addrspace(1) %address.local1042, align 2, !dbg !22
  %address.local1043 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1028, !dbg !22
  %address.local1044 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1043, i64 %address.local1024, !dbg !22
  store half %2103, ptr addrspace(1) %address.local1044, align 2, !dbg !22
  br label %2135, !dbg !22

2135:                                             ; preds = %2097, %2096
  br i1 true, label %2136, label %2174, !dbg !22

2136:                                             ; preds = %2135
  %2137 = insertelement <4 x float> poison, float %1974, i64 0, !dbg !20
  %2138 = insertelement <4 x float> %2137, float %1980, i64 1, !dbg !20
  %2139 = insertelement <4 x float> %2138, float %1975, i64 2, !dbg !20
  %2140 = insertelement <4 x float> %2139, float %1981, i64 3, !dbg !20
  %row.reciprocal114 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1925, !dbg !20
  %normalized115 = fmul <4 x float> %2140, %row.reciprocal114, !dbg !20
  %2141 = fptrunc <4 x float> %normalized115 to <4 x half>, !dbg !21
  %2142 = extractelement <4 x half> %2141, i64 3, !dbg !21
  %2143 = extractelement <4 x half> %2141, i64 2, !dbg !21
  %2144 = extractelement <4 x half> %2141, i64 1, !dbg !21
  %2145 = extractelement <4 x half> %2141, i64 0, !dbg !21
  %2146 = insertelement <4 x float> poison, float %1864, i64 0, !dbg !20
  %2147 = insertelement <4 x float> %2146, float %1870, i64 1, !dbg !20
  %2148 = insertelement <4 x float> %2147, float %1865, i64 2, !dbg !20
  %2149 = insertelement <4 x float> %2148, float %1871, i64 3, !dbg !20
  %normalized116 = fmul <4 x float> %2149, %row.reciprocal114, !dbg !20
  %2150 = fptrunc <4 x float> %normalized116 to <4 x half>, !dbg !21
  %2151 = extractelement <4 x half> %2150, i64 3, !dbg !21
  %2152 = extractelement <4 x half> %2150, i64 2, !dbg !21
  %2153 = extractelement <4 x half> %2150, i64 1, !dbg !21
  %2154 = extractelement <4 x half> %2150, i64 0, !dbg !21
  %2155 = insertelement <4 x float> poison, float %1756, i64 0, !dbg !20
  %2156 = insertelement <4 x float> %2155, float %1762, i64 1, !dbg !20
  %2157 = insertelement <4 x float> %2156, float %1757, i64 2, !dbg !20
  %2158 = insertelement <4 x float> %2157, float %1763, i64 3, !dbg !20
  %row.reciprocal117 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1707, !dbg !20
  %normalized118 = fmul <4 x float> %2158, %row.reciprocal117, !dbg !20
  %2159 = fptrunc <4 x float> %normalized118 to <4 x half>, !dbg !21
  %2160 = extractelement <4 x half> %2159, i64 3, !dbg !21
  %2161 = extractelement <4 x half> %2159, i64 2, !dbg !21
  %2162 = extractelement <4 x half> %2159, i64 1, !dbg !21
  %2163 = extractelement <4 x half> %2159, i64 0, !dbg !21
  %2164 = insertelement <4 x float> poison, float %1646, i64 0, !dbg !20
  %2165 = insertelement <4 x float> %2164, float %1652, i64 1, !dbg !20
  %2166 = insertelement <4 x float> %2165, float %1647, i64 2, !dbg !20
  %2167 = insertelement <4 x float> %2166, float %1653, i64 3, !dbg !20
  %normalized119 = fmul <4 x float> %2167, %row.reciprocal117, !dbg !20
  %2168 = fptrunc <4 x float> %normalized119 to <4 x half>, !dbg !21
  %2169 = extractelement <4 x half> %2168, i64 3, !dbg !21
  %2170 = extractelement <4 x half> %2168, i64 2, !dbg !21
  %2171 = extractelement <4 x half> %2168, i64 1, !dbg !21
  %2172 = extractelement <4 x half> %2168, i64 0, !dbg !21
  %2173 = shl nuw nsw i64 %24, 19, !dbg !22
  %address.thread1045 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !22, !range !34, !frisk.register_lifetimes !1
  %address.local1046 = zext nneg i32 %address.thread1045 to i64, !dbg !8
  %address.local1047 = add i64 %25, %address.local1046, !dbg !22
  %address.local1048 = lshr i64 %address.local1046, 6, !dbg !11
  %address.local1049 = shl nuw nsw i64 %address.local1048, 4, !dbg !22
  %address.local1050 = add i64 %address.local1047, %address.local1049, !dbg !22
  %address.local1051 = and i64 %address.local1046, 240, !dbg !22
  %address.local1052 = sub i64 %address.local1050, %address.local1051, !dbg !22
  %address.local1053 = add i64 %address.local1052, %26, !dbg !22
  %address.local1054 = shl nuw nsw i64 %address.local1053, 7, !dbg !22
  %address.local1055 = add nuw nsw i64 %2173, %address.local1054, !dbg !22
  %address.local1056 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1055, !dbg !22
  %address.local1057 = lshr i64 %address.local1046, 2, !dbg !11
  %address.local1058 = and i64 %address.local1057, 12, !dbg !11
  %address.local1059 = or disjoint i64 %address.local1058, 64, !dbg !22
  %address.local1060 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1056, i64 %address.local1059, !dbg !22
  store half %2172, ptr addrspace(1) %address.local1060, align 2, !dbg !22
  %address.local1061 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1055, !dbg !22
  %address.local1062 = or disjoint i64 %address.local1058, 65, !dbg !22
  %address.local1063 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1061, i64 %address.local1062, !dbg !22
  store half %2171, ptr addrspace(1) %address.local1063, align 2, !dbg !22
  %address.local1064 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1055, !dbg !22
  %address.local1065 = or disjoint i64 %address.local1058, 66, !dbg !22
  %address.local1066 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1064, i64 %address.local1065, !dbg !22
  store half %2170, ptr addrspace(1) %address.local1066, align 2, !dbg !22
  %address.local1067 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1055, !dbg !22
  %address.local1068 = or disjoint i64 %address.local1058, 67, !dbg !22
  %address.local1069 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1067, i64 %address.local1068, !dbg !22
  store half %2169, ptr addrspace(1) %address.local1069, align 2, !dbg !22
  %address.local1070 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1055, !dbg !22
  %address.local1071 = or disjoint i64 %address.local1058, 80, !dbg !22
  %address.local1072 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1070, i64 %address.local1071, !dbg !22
  store half %2163, ptr addrspace(1) %address.local1072, align 2, !dbg !22
  %address.local1073 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1055, !dbg !22
  %address.local1074 = or disjoint i64 %address.local1058, 81, !dbg !22
  %address.local1075 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1073, i64 %address.local1074, !dbg !22
  store half %2162, ptr addrspace(1) %address.local1075, align 2, !dbg !22
  %address.local1076 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1055, !dbg !22
  %address.local1077 = or disjoint i64 %address.local1058, 82, !dbg !22
  %address.local1078 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1076, i64 %address.local1077, !dbg !22
  store half %2161, ptr addrspace(1) %address.local1078, align 2, !dbg !22
  %address.local1079 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1055, !dbg !22
  %address.local1080 = or disjoint i64 %address.local1058, 83, !dbg !22
  %address.local1081 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1079, i64 %address.local1080, !dbg !22
  store half %2160, ptr addrspace(1) %address.local1081, align 2, !dbg !22
  %address.local1082 = shl i64 %address.local1053, 7, !dbg !22
  %address.local1083 = add i64 %address.local1082, 8192, !dbg !22
  %address.local1084 = add nuw nsw i64 %2173, %address.local1083, !dbg !22
  %address.local1085 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1084, !dbg !22
  %address.local1086 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1085, i64 %address.local1059, !dbg !22
  store half %2154, ptr addrspace(1) %address.local1086, align 2, !dbg !22
  %address.local1087 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1084, !dbg !22
  %address.local1088 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1087, i64 %address.local1062, !dbg !22
  store half %2153, ptr addrspace(1) %address.local1088, align 2, !dbg !22
  %address.local1089 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1084, !dbg !22
  %address.local1090 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1089, i64 %address.local1065, !dbg !22
  store half %2152, ptr addrspace(1) %address.local1090, align 2, !dbg !22
  %address.local1091 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1084, !dbg !22
  %address.local1092 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1091, i64 %address.local1068, !dbg !22
  store half %2151, ptr addrspace(1) %address.local1092, align 2, !dbg !22
  %address.local1093 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1084, !dbg !22
  %address.local1094 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1093, i64 %address.local1071, !dbg !22
  store half %2145, ptr addrspace(1) %address.local1094, align 2, !dbg !22
  %address.local1095 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1084, !dbg !22
  %address.local1096 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1095, i64 %address.local1074, !dbg !22
  store half %2144, ptr addrspace(1) %address.local1096, align 2, !dbg !22
  %address.local1097 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1084, !dbg !22
  %address.local1098 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1097, i64 %address.local1077, !dbg !22
  store half %2143, ptr addrspace(1) %address.local1098, align 2, !dbg !22
  %address.local1099 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1084, !dbg !22
  %address.local1100 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1099, i64 %address.local1080, !dbg !22
  store half %2142, ptr addrspace(1) %address.local1100, align 2, !dbg !22
  br label %2174, !dbg !22

2174:                                             ; preds = %2136, %2135
  br i1 true, label %2175, label %2213, !dbg !22

2175:                                             ; preds = %2174
  %2176 = insertelement <4 x float> poison, float %1946, i64 0, !dbg !20
  %2177 = insertelement <4 x float> %2176, float %1952, i64 1, !dbg !20
  %2178 = insertelement <4 x float> %2177, float %1947, i64 2, !dbg !20
  %2179 = insertelement <4 x float> %2178, float %1953, i64 3, !dbg !20
  %row.reciprocal120 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1925, !dbg !20
  %normalized121 = fmul <4 x float> %2179, %row.reciprocal120, !dbg !20
  %2180 = fptrunc <4 x float> %normalized121 to <4 x half>, !dbg !21
  %2181 = extractelement <4 x half> %2180, i64 3, !dbg !21
  %2182 = extractelement <4 x half> %2180, i64 2, !dbg !21
  %2183 = extractelement <4 x half> %2180, i64 1, !dbg !21
  %2184 = extractelement <4 x half> %2180, i64 0, !dbg !21
  %2185 = insertelement <4 x float> poison, float %1835, i64 0, !dbg !20
  %2186 = insertelement <4 x float> %2185, float %1841, i64 1, !dbg !20
  %2187 = insertelement <4 x float> %2186, float %1836, i64 2, !dbg !20
  %2188 = insertelement <4 x float> %2187, float %1842, i64 3, !dbg !20
  %normalized122 = fmul <4 x float> %2188, %row.reciprocal120, !dbg !20
  %2189 = fptrunc <4 x float> %normalized122 to <4 x half>, !dbg !21
  %2190 = extractelement <4 x half> %2189, i64 3, !dbg !21
  %2191 = extractelement <4 x half> %2189, i64 2, !dbg !21
  %2192 = extractelement <4 x half> %2189, i64 1, !dbg !21
  %2193 = extractelement <4 x half> %2189, i64 0, !dbg !21
  %2194 = insertelement <4 x float> poison, float %1728, i64 0, !dbg !20
  %2195 = insertelement <4 x float> %2194, float %1734, i64 1, !dbg !20
  %2196 = insertelement <4 x float> %2195, float %1729, i64 2, !dbg !20
  %2197 = insertelement <4 x float> %2196, float %1735, i64 3, !dbg !20
  %row.reciprocal123 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1707, !dbg !20
  %normalized124 = fmul <4 x float> %2197, %row.reciprocal123, !dbg !20
  %2198 = fptrunc <4 x float> %normalized124 to <4 x half>, !dbg !21
  %2199 = extractelement <4 x half> %2198, i64 3, !dbg !21
  %2200 = extractelement <4 x half> %2198, i64 2, !dbg !21
  %2201 = extractelement <4 x half> %2198, i64 1, !dbg !21
  %2202 = extractelement <4 x half> %2198, i64 0, !dbg !21
  %2203 = insertelement <4 x float> poison, float %1617, i64 0, !dbg !20
  %2204 = insertelement <4 x float> %2203, float %1623, i64 1, !dbg !20
  %2205 = insertelement <4 x float> %2204, float %1618, i64 2, !dbg !20
  %2206 = insertelement <4 x float> %2205, float %1624, i64 3, !dbg !20
  %normalized125 = fmul <4 x float> %2206, %row.reciprocal123, !dbg !20
  %2207 = fptrunc <4 x float> %normalized125 to <4 x half>, !dbg !21
  %2208 = extractelement <4 x half> %2207, i64 3, !dbg !21
  %2209 = extractelement <4 x half> %2207, i64 2, !dbg !21
  %2210 = extractelement <4 x half> %2207, i64 1, !dbg !21
  %2211 = extractelement <4 x half> %2207, i64 0, !dbg !21
  %2212 = shl nuw nsw i64 %24, 19, !dbg !22
  %address.thread1101 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !22, !range !34, !frisk.register_lifetimes !1
  %address.local1102 = zext nneg i32 %address.thread1101 to i64, !dbg !8
  %address.local1103 = add i64 %25, %address.local1102, !dbg !22
  %address.local1104 = lshr i64 %address.local1102, 6, !dbg !11
  %address.local1105 = shl nuw nsw i64 %address.local1104, 4, !dbg !22
  %address.local1106 = add i64 %address.local1103, %address.local1105, !dbg !22
  %address.local1107 = and i64 %address.local1102, 240, !dbg !22
  %address.local1108 = sub i64 %address.local1106, %address.local1107, !dbg !22
  %address.local1109 = add i64 %address.local1108, %26, !dbg !22
  %address.local1110 = shl nuw nsw i64 %address.local1109, 7, !dbg !22
  %address.local1111 = add nuw nsw i64 %2212, %address.local1110, !dbg !22
  %address.local1112 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1111, !dbg !22
  %address.local1113 = lshr i64 %address.local1102, 2, !dbg !11
  %address.local1114 = and i64 %address.local1113, 12, !dbg !11
  %address.local1115 = or disjoint i64 %address.local1114, 96, !dbg !22
  %address.local1116 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1112, i64 %address.local1115, !dbg !22
  store half %2211, ptr addrspace(1) %address.local1116, align 2, !dbg !22
  %address.local1117 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1111, !dbg !22
  %address.local1118 = or disjoint i64 %address.local1114, 97, !dbg !22
  %address.local1119 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1117, i64 %address.local1118, !dbg !22
  store half %2210, ptr addrspace(1) %address.local1119, align 2, !dbg !22
  %address.local1120 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1111, !dbg !22
  %address.local1121 = or disjoint i64 %address.local1114, 98, !dbg !22
  %address.local1122 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1120, i64 %address.local1121, !dbg !22
  store half %2209, ptr addrspace(1) %address.local1122, align 2, !dbg !22
  %address.local1123 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1111, !dbg !22
  %address.local1124 = or disjoint i64 %address.local1114, 99, !dbg !22
  %address.local1125 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1123, i64 %address.local1124, !dbg !22
  store half %2208, ptr addrspace(1) %address.local1125, align 2, !dbg !22
  %address.local1126 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1111, !dbg !22
  %address.local1127 = or disjoint i64 %address.local1114, 112, !dbg !22
  %address.local1128 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1126, i64 %address.local1127, !dbg !22
  store half %2202, ptr addrspace(1) %address.local1128, align 2, !dbg !22
  %address.local1129 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1111, !dbg !22
  %address.local1130 = or disjoint i64 %address.local1114, 113, !dbg !22
  %address.local1131 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1129, i64 %address.local1130, !dbg !22
  store half %2201, ptr addrspace(1) %address.local1131, align 2, !dbg !22
  %address.local1132 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1111, !dbg !22
  %address.local1133 = or disjoint i64 %address.local1114, 114, !dbg !22
  %address.local1134 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1132, i64 %address.local1133, !dbg !22
  store half %2200, ptr addrspace(1) %address.local1134, align 2, !dbg !22
  %address.local1135 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1111, !dbg !22
  %address.local1136 = or i64 %address.local1113, 115, !dbg !22
  %address.local1137 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1135, i64 %address.local1136, !dbg !22
  store half %2199, ptr addrspace(1) %address.local1137, align 2, !dbg !22
  %address.local1138 = shl i64 %address.local1109, 7, !dbg !22
  %address.local1139 = add i64 %address.local1138, 8192, !dbg !22
  %address.local1140 = add nuw nsw i64 %2212, %address.local1139, !dbg !22
  %address.local1141 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !22
  %address.local1142 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1141, i64 %address.local1115, !dbg !22
  store half %2193, ptr addrspace(1) %address.local1142, align 2, !dbg !22
  %address.local1143 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !22
  %address.local1144 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1143, i64 %address.local1118, !dbg !22
  store half %2192, ptr addrspace(1) %address.local1144, align 2, !dbg !22
  %address.local1145 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !22
  %address.local1146 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1145, i64 %address.local1121, !dbg !22
  store half %2191, ptr addrspace(1) %address.local1146, align 2, !dbg !22
  %address.local1147 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !22
  %address.local1148 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1147, i64 %address.local1124, !dbg !22
  store half %2190, ptr addrspace(1) %address.local1148, align 2, !dbg !22
  %address.local1149 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !22
  %address.local1150 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1149, i64 %address.local1127, !dbg !22
  store half %2184, ptr addrspace(1) %address.local1150, align 2, !dbg !22
  %address.local1151 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !22
  %address.local1152 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1151, i64 %address.local1130, !dbg !22
  store half %2183, ptr addrspace(1) %address.local1152, align 2, !dbg !22
  %address.local1153 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !22
  %address.local1154 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1153, i64 %address.local1133, !dbg !22
  store half %2182, ptr addrspace(1) %address.local1154, align 2, !dbg !22
  %address.local1155 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !22
  %address.local1156 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1155, i64 %address.local1136, !dbg !22
  store half %2181, ptr addrspace(1) %address.local1156, align 2, !dbg !22
  br label %2213, !dbg !22

2213:                                             ; preds = %2175, %2174
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
declare float @llvm.amdgcn.exp2.f32(float) #8

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.sched.barrier(i32 immarg) #5

attributes #0 = { convergent willreturn memory(read) }
attributes #1 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1" "frisk.frag_prefetch"="4" "frisk.global_prefetch"="8" "frisk.register_comparisons"="1" "frisk.register_lifetimes"="1" "uniform-work-group-size" }
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
!llvm.module.flags = !{!33}

!0 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/tune_retention_integration/run-20261023-160942-dxpgegoa/retained32-baseline")
!1 = !{}
!2 = !DISubroutineType(types: !1)
!3 = distinct !DICompileUnit(language: DW_LANG_C, file: !0, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!4 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !0, file: !0, line: 22, type: !2, scopeLine: 22, spFlags: DISPFlagDefinition, unit: !3)
!5 = !DILocation(line: 22, column: 3, scope: !4)
!8 = !DILocation(line: 66, column: 22, scope: !4)
!10 = !DILocation(line: 46, column: 17, scope: !4)
!11 = !DILocation(line: 68, column: 24, scope: !4)
!13 = !DILocation(line: 56, column: 24, scope: !4)
!14 = !DILocation(line: 73, column: 19, scope: !4)
!15 = !DILocation(line: 93, column: 23, scope: !4)
!16 = !DILocation(line: 0, scope: !4)
!17 = !DILocation(line: 67, column: 22, scope: !4)
!18 = !DILocation(line: 69, column: 26, scope: !4)
!19 = !DILocation(line: 92, column: 20, scope: !4)
!20 = !DILocation(line: 97, column: 23, scope: !4)
!21 = !DILocation(line: 98, column: 19, scope: !4)
!22 = !DILocation(line: 99, column: 9, scope: !4)
!24 = !DILocation(line: 75, column: 24, scope: !4)
!25 = !DILocation(line: 76, column: 23, scope: !4)
!26 = !DILocation(line: 77, column: 26, scope: !4)
!27 = !DILocation(line: 84, column: 21, scope: !4)
!28 = !DILocation(line: 85, column: 22, scope: !4)
!29 = !DILocation(line: 86, column: 20, scope: !4)
!30 = !DILocation(line: 87, column: 16, scope: !4)
!31 = !DILocation(line: 94, column: 20, scope: !4)
!32 = !{i32 256, i32 1, i32 1}
!33 = !{i32 2, !"Debug Info Version", i32 3}
!34 = !{i32 0, i32 256}
