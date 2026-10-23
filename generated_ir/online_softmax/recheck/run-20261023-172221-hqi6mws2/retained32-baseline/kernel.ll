; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/online_softmax/recheck/run-20261023-172221-hqi6mws2/retained32-baseline/input.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/online_softmax/recheck/run-20261023-172221-hqi6mws2/retained32-baseline/input.mlir"

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
  %12 = and i64 %11, 576460752303423480, !dbg !5
  %13 = and i64 %6, 7, !dbg !5
  %14 = or disjoint i64 %12, %13, !dbg !5
  %15 = sub nsw i64 31, %14, !dbg !5
  %16 = lshr i64 %10, 3, !dbg !5
  %17 = icmp ugt i64 %10, 1023, !dbg !7
  %18 = add nsw i64 %14, -32, !dbg !7
  %19 = select i1 %17, i64 %18, i64 %15, !dbg !7
  %20 = sdiv i64 %19, 32, !dbg !7
  %21 = sext i1 %17 to i64, !dbg !7
  %22 = xor i64 %20, %21, !dbg !7
  %23 = add nsw i64 %16, %22, !dbg !7
  %24 = and i64 %23, 31, !dbg !7
  %25 = shl nsw i64 %15, 7, !dbg !7
  %26 = mul nsw i64 %22, -4096, !dbg !7
  %27 = add i64 %25, %26, !dbg !7
  %28 = shl nuw nsw i64 %24, 19, !dbg !7
  %29 = shl nsw i64 %27, 7, !dbg !7
  %30 = add i64 %28, %29, !dbg !7
  %31 = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !7, !range !39
  %32 = zext nneg i32 %31 to i64, !dbg !7
  %33 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %30, !dbg !7
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !7, !range !39, !frisk.register_lifetimes !1
  %address.local = zext nneg i32 %address.thread to i64, !dbg !7
  %address.local339 = shl nuw nsw i64 %address.local, 3, !dbg !7
  %address.local340 = and i64 %address.local339, 1536, !dbg !7
  %address.local341 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local340, !dbg !7
  %address.local342 = ptrtoint ptr addrspace(3) %address.local341 to i64, !dbg !7
  %address.local343 = trunc i64 %address.local342 to i32, !dbg !7
  %34 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local343), !dbg !7
  %35 = zext i32 %34 to i64, !dbg !7
  %36 = inttoptr i64 %35 to ptr addrspace(3), !dbg !7
  %address.local344 = shl nuw nsw i64 %address.local, 7, !dbg !7
  %address.local345 = and i64 %address.local344, 16256, !dbg !7
  %address.local346 = lshr i64 %address.local, 4, !dbg !7
  %address.local347 = and i64 %address.local346, 8, !dbg !7
  %address.local348 = or disjoint i64 %address.local345, %address.local347, !dbg !7
  %address.local349 = getelementptr [2 x i8], ptr addrspace(1) %33, i64 %address.local348, !dbg !7
  %37 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !7
  %38 = ptrtoint ptr addrspace(1) %address.local349 to i64, !dbg !7
  %39 = sub i64 %38, %37, !dbg !7
  %dma.buffer.offset1612 = trunc i64 %39 to i32, !dbg !7
  %40 = trunc i64 %37 to i32, !dbg !7
  %41 = insertelement <4 x i32> poison, i32 %40, i64 0, !dbg !7
  %42 = lshr i64 %37, 32, !dbg !7
  %43 = trunc i64 %42 to i32, !dbg !7
  %44 = insertelement <4 x i32> %41, i32 %43, i64 1, !dbg !7
  %45 = insertelement <4 x i32> %44, i32 33554432, i64 2, !dbg !7
  %dma.buffer.rsrc1613 = insertelement <4 x i32> %45, i32 131072, i64 3, !dbg !7
  %46 = ptrtoint ptr addrspace(3) %36 to i32, !dbg !7
  %dma.buffer.lds1614 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %46), !dbg !7
  %47 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1614, i32 %dma.buffer.offset1612, <4 x i32> %dma.buffer.rsrc1613) #10, !dbg !7
  %address.local350 = or disjoint i64 %address.local, 256, !dbg !7
  %address.local351 = shl nuw nsw i64 %address.local350, 3, !dbg !7
  %address.local352 = and i64 %address.local351, 3584, !dbg !7
  %address.local353 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local352, !dbg !7
  %address.local354 = ptrtoint ptr addrspace(3) %address.local353 to i64, !dbg !7
  %address.local355 = trunc i64 %address.local354 to i32, !dbg !7
  %48 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local355), !dbg !7
  %49 = zext i32 %48 to i64, !dbg !7
  %50 = inttoptr i64 %49 to ptr addrspace(3), !dbg !7
  %address.local356 = shl nuw nsw i64 %address.local, 7, !dbg !7
  %address.local357 = and i64 %address.local356, 16256, !dbg !7
  %address.local358 = lshr i64 %address.local350, 4, !dbg !7
  %address.local359 = and i64 %address.local358, 24, !dbg !7
  %address.local360 = or disjoint i64 %address.local357, %address.local359, !dbg !7
  %address.local361 = getelementptr [2 x i8], ptr addrspace(1) %33, i64 %address.local360, !dbg !7
  %51 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !7
  %52 = ptrtoint ptr addrspace(1) %address.local361 to i64, !dbg !7
  %53 = sub i64 %52, %51, !dbg !7
  %dma.buffer.offset1609 = trunc i64 %53 to i32, !dbg !7
  %54 = trunc i64 %51 to i32, !dbg !7
  %55 = insertelement <4 x i32> poison, i32 %54, i64 0, !dbg !7
  %56 = lshr i64 %51, 32, !dbg !7
  %57 = trunc i64 %56 to i32, !dbg !7
  %58 = insertelement <4 x i32> %55, i32 %57, i64 1, !dbg !7
  %59 = insertelement <4 x i32> %58, i32 33554432, i64 2, !dbg !7
  %dma.buffer.rsrc1610 = insertelement <4 x i32> %59, i32 131072, i64 3, !dbg !7
  %60 = ptrtoint ptr addrspace(3) %50 to i32, !dbg !7
  %dma.buffer.lds1611 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %60), !dbg !7
  %61 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1611, i32 %dma.buffer.offset1609, <4 x i32> %dma.buffer.rsrc1610) #10, !dbg !7
  %62 = call i32 @llvm.amdgcn.workgroup.id.z(), !dbg !5
  %63 = sext i32 %62 to i64, !dbg !5
  %64 = shl i64 %15, 7, !dbg !9
  %65 = lshr i64 %32, 6, !dbg !10
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !7
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %66 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %30, !dbg !7
  %67 = getelementptr i8, ptr addrspace(1) %66, i64 64, !dbg !7
  %address.local362 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local340, !dbg !7
  %address.local363 = ptrtoint ptr addrspace(3) %address.local362 to i64, !dbg !7
  %address.local364 = trunc i64 %address.local363 to i32, !dbg !7
  %68 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local364), !dbg !7
  %69 = zext i32 %68 to i64, !dbg !7
  %70 = inttoptr i64 %69 to ptr addrspace(3), !dbg !7
  %address.local365 = getelementptr [2 x i8], ptr addrspace(1) %67, i64 %address.local348, !dbg !7
  %71 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !7
  %72 = ptrtoint ptr addrspace(1) %address.local365 to i64, !dbg !7
  %73 = sub i64 %72, %71, !dbg !7
  %dma.buffer.offset1606 = trunc i64 %73 to i32, !dbg !7
  %74 = trunc i64 %71 to i32, !dbg !7
  %75 = insertelement <4 x i32> poison, i32 %74, i64 0, !dbg !7
  %76 = lshr i64 %71, 32, !dbg !7
  %77 = trunc i64 %76 to i32, !dbg !7
  %78 = insertelement <4 x i32> %75, i32 %77, i64 1, !dbg !7
  %79 = insertelement <4 x i32> %78, i32 33554432, i64 2, !dbg !7
  %dma.buffer.rsrc1607 = insertelement <4 x i32> %79, i32 131072, i64 3, !dbg !7
  %80 = ptrtoint ptr addrspace(3) %70 to i32, !dbg !7
  %dma.buffer.lds1608 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %80), !dbg !7
  %81 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1608, i32 %dma.buffer.offset1606, <4 x i32> %dma.buffer.rsrc1607) #10, !dbg !7
  %address.local366 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local352, !dbg !7
  %address.local367 = ptrtoint ptr addrspace(3) %address.local366 to i64, !dbg !7
  %address.local368 = trunc i64 %address.local367 to i32, !dbg !7
  %82 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local368), !dbg !7
  %83 = zext i32 %82 to i64, !dbg !7
  %84 = inttoptr i64 %83 to ptr addrspace(3), !dbg !7
  %address.local369 = getelementptr [2 x i8], ptr addrspace(1) %67, i64 %address.local360, !dbg !7
  %85 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !7
  %86 = ptrtoint ptr addrspace(1) %address.local369 to i64, !dbg !7
  %87 = sub i64 %86, %85, !dbg !7
  %dma.buffer.offset1603 = trunc i64 %87 to i32, !dbg !7
  %88 = trunc i64 %85 to i32, !dbg !7
  %89 = insertelement <4 x i32> poison, i32 %88, i64 0, !dbg !7
  %90 = lshr i64 %85, 32, !dbg !7
  %91 = trunc i64 %90 to i32, !dbg !7
  %92 = insertelement <4 x i32> %89, i32 %91, i64 1, !dbg !7
  %93 = insertelement <4 x i32> %92, i32 33554432, i64 2, !dbg !7
  %dma.buffer.rsrc1604 = insertelement <4 x i32> %93, i32 131072, i64 3, !dbg !7
  %94 = ptrtoint ptr addrspace(3) %84 to i32, !dbg !7
  %dma.buffer.lds1605 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %94), !dbg !7
  %95 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1605, i32 %dma.buffer.offset1603, <4 x i32> %dma.buffer.rsrc1604) #10, !dbg !7
  %address.local370 = lshr i32 %address.thread, 2, !dbg !10
  %address.local371 = and i32 %address.local370, 4, !dbg !10
  %address.local372 = shl nuw nsw i32 %address.thread, 1, !dbg !10
  %address.local373 = and i32 %address.local372, 384, !dbg !10
  %address.local374 = or disjoint i32 %address.local371, %address.local373, !dbg !10
  %address.local375 = shl nuw nsw i32 %address.thread, 3, !dbg !10
  %address.local376 = and i32 %address.local375, 120, !dbg !10
  %address.local377 = or disjoint i32 %address.local374, %address.local376, !dbg !10
  %address.local378 = shl nuw nsw i32 %address.thread, 5, !dbg !10
  %address.local379 = and i32 %address.local378, 1024, !dbg !10
  %address.local380 = or disjoint i32 %address.local377, %address.local379, !dbg !10
  %address.local381 = zext nneg i32 %address.local380 to i64, !dbg !10
  %address.local382 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local381, !dbg !10
  %96 = load <4 x half>, ptr addrspace(3) %address.local382, align 8, !dbg !10
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local382, i64 4096, !dbg !10
  %97 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !10
  %98 = fmul <4 x half> %96, splat (half 0xH3015), !dbg !10
  %lds.const.gep1413 = getelementptr i8, ptr addrspace(3) %address.local382, i64 1024, !dbg !10
  %99 = load <4 x half>, ptr addrspace(3) %lds.const.gep1413, align 8, !dbg !10
  %100 = fmul <4 x half> %97, splat (half 0xH3015), !dbg !10
  %lds.const.gep1414 = getelementptr i8, ptr addrspace(3) %address.local382, i64 5120, !dbg !10
  %101 = load <4 x half>, ptr addrspace(3) %lds.const.gep1414, align 8, !dbg !10
  %102 = fmul <4 x half> %99, splat (half 0xH3015), !dbg !10
  %103 = fmul <4 x half> %101, splat (half 0xH3015), !dbg !10
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !7
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %104 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %30, !dbg !7
  %105 = getelementptr i8, ptr addrspace(1) %104, i64 128, !dbg !7
  %106 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local343), !dbg !7
  %107 = zext i32 %106 to i64, !dbg !7
  %108 = inttoptr i64 %107 to ptr addrspace(3), !dbg !7
  %address.local425 = getelementptr [2 x i8], ptr addrspace(1) %105, i64 %address.local348, !dbg !7
  %109 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !7
  %110 = ptrtoint ptr addrspace(1) %address.local425 to i64, !dbg !7
  %111 = sub i64 %110, %109, !dbg !7
  %dma.buffer.offset1600 = trunc i64 %111 to i32, !dbg !7
  %112 = trunc i64 %109 to i32, !dbg !7
  %113 = insertelement <4 x i32> poison, i32 %112, i64 0, !dbg !7
  %114 = lshr i64 %109, 32, !dbg !7
  %115 = trunc i64 %114 to i32, !dbg !7
  %116 = insertelement <4 x i32> %113, i32 %115, i64 1, !dbg !7
  %117 = insertelement <4 x i32> %116, i32 33554432, i64 2, !dbg !7
  %dma.buffer.rsrc1601 = insertelement <4 x i32> %117, i32 131072, i64 3, !dbg !7
  %118 = ptrtoint ptr addrspace(3) %108 to i32, !dbg !7
  %dma.buffer.lds1602 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %118), !dbg !7
  %119 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1602, i32 %dma.buffer.offset1600, <4 x i32> %dma.buffer.rsrc1601) #10, !dbg !7
  %120 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local355), !dbg !7
  %121 = zext i32 %120 to i64, !dbg !7
  %122 = inttoptr i64 %121 to ptr addrspace(3), !dbg !7
  %address.local426 = getelementptr [2 x i8], ptr addrspace(1) %105, i64 %address.local360, !dbg !7
  %123 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !7
  %124 = ptrtoint ptr addrspace(1) %address.local426 to i64, !dbg !7
  %125 = sub i64 %124, %123, !dbg !7
  %dma.buffer.offset1597 = trunc i64 %125 to i32, !dbg !7
  %126 = trunc i64 %123 to i32, !dbg !7
  %127 = insertelement <4 x i32> poison, i32 %126, i64 0, !dbg !7
  %128 = lshr i64 %123, 32, !dbg !7
  %129 = trunc i64 %128 to i32, !dbg !7
  %130 = insertelement <4 x i32> %127, i32 %129, i64 1, !dbg !7
  %131 = insertelement <4 x i32> %130, i32 33554432, i64 2, !dbg !7
  %dma.buffer.rsrc1598 = insertelement <4 x i32> %131, i32 131072, i64 3, !dbg !7
  %132 = ptrtoint ptr addrspace(3) %122 to i32, !dbg !7
  %dma.buffer.lds1599 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %132), !dbg !7
  %133 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1599, i32 %dma.buffer.offset1597, <4 x i32> %dma.buffer.rsrc1598) #10, !dbg !7
  %lds.const.gep1415 = getelementptr i8, ptr addrspace(3) %address.local382, i64 8192, !dbg !10
  %134 = load <4 x half>, ptr addrspace(3) %lds.const.gep1415, align 8, !dbg !10
  %lds.const.gep1416 = getelementptr i8, ptr addrspace(3) %address.local382, i64 12288, !dbg !10
  %135 = load <4 x half>, ptr addrspace(3) %lds.const.gep1416, align 8, !dbg !10
  %lds.const.gep1417 = getelementptr i8, ptr addrspace(3) %address.local382, i64 9216, !dbg !10
  %136 = load <4 x half>, ptr addrspace(3) %lds.const.gep1417, align 8, !dbg !10
  %137 = fmul <4 x half> %134, splat (half 0xH3015), !dbg !10
  %lds.const.gep1418 = getelementptr i8, ptr addrspace(3) %address.local382, i64 13312, !dbg !10
  %138 = load <4 x half>, ptr addrspace(3) %lds.const.gep1418, align 8, !dbg !10
  %139 = fmul <4 x half> %135, splat (half 0xH3015), !dbg !10
  %140 = fmul <4 x half> %136, splat (half 0xH3015), !dbg !10
  %141 = fmul <4 x half> %138, splat (half 0xH3015), !dbg !10
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !7
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %142 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %30, !dbg !7
  %143 = getelementptr i8, ptr addrspace(1) %142, i64 192, !dbg !7
  %144 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local364), !dbg !7
  %145 = zext i32 %144 to i64, !dbg !7
  %146 = inttoptr i64 %145 to ptr addrspace(3), !dbg !7
  %address.local431 = getelementptr [2 x i8], ptr addrspace(1) %143, i64 %address.local348, !dbg !7
  %147 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !7
  %148 = ptrtoint ptr addrspace(1) %address.local431 to i64, !dbg !7
  %149 = sub i64 %148, %147, !dbg !7
  %dma.buffer.offset1594 = trunc i64 %149 to i32, !dbg !7
  %150 = trunc i64 %147 to i32, !dbg !7
  %151 = insertelement <4 x i32> poison, i32 %150, i64 0, !dbg !7
  %152 = lshr i64 %147, 32, !dbg !7
  %153 = trunc i64 %152 to i32, !dbg !7
  %154 = insertelement <4 x i32> %151, i32 %153, i64 1, !dbg !7
  %155 = insertelement <4 x i32> %154, i32 33554432, i64 2, !dbg !7
  %dma.buffer.rsrc1595 = insertelement <4 x i32> %155, i32 131072, i64 3, !dbg !7
  %156 = ptrtoint ptr addrspace(3) %146 to i32, !dbg !7
  %dma.buffer.lds1596 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %156), !dbg !7
  %157 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1596, i32 %dma.buffer.offset1594, <4 x i32> %dma.buffer.rsrc1595) #10, !dbg !7
  %158 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local368), !dbg !7
  %159 = zext i32 %158 to i64, !dbg !7
  %160 = inttoptr i64 %159 to ptr addrspace(3), !dbg !7
  %address.local432 = getelementptr [2 x i8], ptr addrspace(1) %143, i64 %address.local360, !dbg !7
  %161 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !7
  %162 = ptrtoint ptr addrspace(1) %address.local432 to i64, !dbg !7
  %163 = sub i64 %162, %161, !dbg !7
  %dma.buffer.offset1591 = trunc i64 %163 to i32, !dbg !7
  %164 = trunc i64 %161 to i32, !dbg !7
  %165 = insertelement <4 x i32> poison, i32 %164, i64 0, !dbg !7
  %166 = lshr i64 %161, 32, !dbg !7
  %167 = trunc i64 %166 to i32, !dbg !7
  %168 = insertelement <4 x i32> %165, i32 %167, i64 1, !dbg !7
  %169 = insertelement <4 x i32> %168, i32 33554432, i64 2, !dbg !7
  %dma.buffer.rsrc1592 = insertelement <4 x i32> %169, i32 131072, i64 3, !dbg !7
  %170 = ptrtoint ptr addrspace(3) %160 to i32, !dbg !7
  %dma.buffer.lds1593 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %170), !dbg !7
  %171 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1593, i32 %dma.buffer.offset1591, <4 x i32> %dma.buffer.rsrc1592) #10, !dbg !7
  %172 = load <4 x half>, ptr addrspace(3) %address.local382, align 8, !dbg !10
  %lds.const.gep1419 = getelementptr i8, ptr addrspace(3) %address.local382, i64 4096, !dbg !10
  %173 = load <4 x half>, ptr addrspace(3) %lds.const.gep1419, align 8, !dbg !10
  %lds.const.gep1420 = getelementptr i8, ptr addrspace(3) %address.local382, i64 1024, !dbg !10
  %174 = load <4 x half>, ptr addrspace(3) %lds.const.gep1420, align 8, !dbg !10
  %175 = fmul <4 x half> %172, splat (half 0xH3015), !dbg !10
  %lds.const.gep1421 = getelementptr i8, ptr addrspace(3) %address.local382, i64 5120, !dbg !10
  %176 = load <4 x half>, ptr addrspace(3) %lds.const.gep1421, align 8, !dbg !10
  %177 = fmul <4 x half> %173, splat (half 0xH3015), !dbg !10
  %178 = fmul <4 x half> %174, splat (half 0xH3015), !dbg !10
  %179 = fmul <4 x half> %176, splat (half 0xH3015), !dbg !10
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !7
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %lds.const.gep1422 = getelementptr i8, ptr addrspace(3) %address.local382, i64 8192, !dbg !10
  %180 = load <4 x half>, ptr addrspace(3) %lds.const.gep1422, align 8, !dbg !10
  %lds.const.gep1423 = getelementptr i8, ptr addrspace(3) %address.local382, i64 12288, !dbg !10
  %181 = load <4 x half>, ptr addrspace(3) %lds.const.gep1423, align 8, !dbg !10
  %lds.const.gep1424 = getelementptr i8, ptr addrspace(3) %address.local382, i64 9216, !dbg !10
  %182 = load <4 x half>, ptr addrspace(3) %lds.const.gep1424, align 8, !dbg !10
  %183 = fmul <4 x half> %180, splat (half 0xH3015), !dbg !10
  %lds.const.gep1425 = getelementptr i8, ptr addrspace(3) %address.local382, i64 13312, !dbg !10
  %184 = load <4 x half>, ptr addrspace(3) %lds.const.gep1425, align 8, !dbg !10
  %185 = fmul <4 x half> %181, splat (half 0xH3015), !dbg !10
  %186 = fmul <4 x half> %182, splat (half 0xH3015), !dbg !10
  %187 = fmul <4 x half> %184, splat (half 0xH3015), !dbg !10
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %188 = or disjoint i64 %64, 127, !dbg !13
  %189 = add i64 %64, 128, !dbg !13
  %190 = icmp uge i64 %189, %188, !dbg !13
  %191 = and i32 %31, 16, !dbg !14
  %.not = icmp eq i32 %191, 0, !dbg !14
  %192 = and i32 %31, 32, !dbg !14
  %.not1 = icmp eq i32 %192, 0, !dbg !14
  %193 = sub nsw i64 32, %14, !dbg !15
  %194 = shl i64 %16, 7, !dbg !16
  %195 = and i64 %194, 3968, !dbg !16
  %196 = shl nsw i64 %63, 12, !dbg !16
  %197 = or disjoint i64 %195, %196, !dbg !16
  %198 = lshr i64 %32, 2, !dbg !17
  %199 = shl nuw nsw i64 %32, 2, !dbg !17
  %200 = mul nsw i64 %198, -16, !dbg !17
  %201 = add nsw i64 %199, %200, !dbg !17
  %202 = and i32 %31, 1, !dbg !17
  %index.compact279 = zext nneg i32 %202 to i64, !dbg !17
  %203 = or disjoint i64 %201, %index.compact279, !dbg !17
  %204 = add nuw nsw i64 %203, 1, !dbg !17
  %205 = sdiv i64 %204, 2, !dbg !17
  br label %206, !dbg !12

206:                                              ; preds = %849, %4
  %207 = phi i64 [ %2304, %849 ], [ 0, %4 ], !dbg !12
  %208 = phi [2 x <8 x float>] [ %1613, %849 ], [ zeroinitializer, %4 ], !dbg !12
  %209 = phi [2 x <1 x float>] [ %1369, %849 ], [ zeroinitializer, %4 ], !dbg !12
  %210 = phi [2 x <8 x float>] [ %1853, %849 ], [ zeroinitializer, %4 ], !dbg !12
  %211 = phi [2 x <8 x float>] [ %2093, %849 ], [ zeroinitializer, %4 ], !dbg !12
  %212 = phi [2 x <8 x float>] [ %2303, %849 ], [ zeroinitializer, %4 ], !dbg !12
  %213 = icmp slt i64 %207, %193, !dbg !12
  br i1 %213, label %214, label %2305, !dbg !12

214:                                              ; preds = %206
  %215 = shl nsw i64 %207, 7, !dbg !12
  %216 = lshr i64 %207, 12, !dbg !16
  %217 = add nuw nsw i64 %216, %16, !dbg !16
  %218 = lshr i64 %207, 5, !dbg !16
  %219 = and i64 %218, 127, !dbg !16
  %220 = shl i64 %217, 19, !dbg !16
  %221 = and i64 %220, 16252928, !dbg !16
  %222 = shl i64 %218, 19, !dbg !16
  %223 = shl i64 %207, 14, !dbg !16
  %224 = sub i64 %223, %222, !dbg !16
  %225 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %221, !dbg !16
  %226 = getelementptr [2 x i8], ptr addrspace(1) %225, i64 %219, !dbg !16
  %227 = getelementptr [2 x i8], ptr addrspace(1) %226, i64 %224, !dbg !16
  %address.thread433 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !16, !range !39, !frisk.register_lifetimes !1
  %address.local434 = zext nneg i32 %address.thread433 to i64, !dbg !7
  %address.local435 = shl nuw nsw i64 %address.local434, 3, !dbg !7
  %address.local436 = and i64 %address.local435, 1536, !dbg !7
  %address.local437 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local436, !dbg !7
  %address.local438 = ptrtoint ptr addrspace(3) %address.local437 to i64, !dbg !7
  %address.local439 = trunc i64 %address.local438 to i32, !dbg !7
  %228 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local439), !dbg !16
  %229 = zext i32 %228 to i64, !dbg !16
  %230 = inttoptr i64 %229 to ptr addrspace(3), !dbg !16
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
  %address.local450 = getelementptr [2 x i8], ptr addrspace(1) %227, i64 %address.local449, !dbg !16
  %231 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %232 = ptrtoint ptr addrspace(1) %address.local450 to i64, !dbg !16
  %233 = sub i64 %232, %231, !dbg !16
  %dma.buffer.offset1588 = trunc i64 %233 to i32, !dbg !16
  %234 = trunc i64 %231 to i32, !dbg !16
  %235 = insertelement <4 x i32> poison, i32 %234, i64 0, !dbg !16
  %236 = lshr i64 %231, 32, !dbg !16
  %237 = trunc i64 %236 to i32, !dbg !16
  %238 = insertelement <4 x i32> %235, i32 %237, i64 1, !dbg !16
  %239 = insertelement <4 x i32> %238, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1589 = insertelement <4 x i32> %239, i32 131072, i64 3, !dbg !16
  %240 = ptrtoint ptr addrspace(3) %230 to i32, !dbg !16
  %dma.buffer.lds1590 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %240), !dbg !16
  %241 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1590, i32 %dma.buffer.offset1588, <4 x i32> %dma.buffer.rsrc1589) #10, !dbg !16
  %address.local451 = or disjoint i64 %address.local434, 256, !dbg !7
  %address.local452 = shl nuw nsw i64 %address.local451, 3, !dbg !7
  %address.local453 = and i64 %address.local452, 3584, !dbg !7
  %address.local454 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local453, !dbg !7
  %address.local455 = ptrtoint ptr addrspace(3) %address.local454 to i64, !dbg !7
  %address.local456 = trunc i64 %address.local455 to i32, !dbg !7
  %242 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local456), !dbg !16
  %243 = zext i32 %242 to i64, !dbg !16
  %244 = inttoptr i64 %243 to ptr addrspace(3), !dbg !16
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
  %address.local467 = getelementptr [2 x i8], ptr addrspace(1) %227, i64 %address.local466, !dbg !16
  %245 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %246 = ptrtoint ptr addrspace(1) %address.local467 to i64, !dbg !16
  %247 = sub i64 %246, %245, !dbg !16
  %dma.buffer.offset1585 = trunc i64 %247 to i32, !dbg !16
  %248 = trunc i64 %245 to i32, !dbg !16
  %249 = insertelement <4 x i32> poison, i32 %248, i64 0, !dbg !16
  %250 = lshr i64 %245, 32, !dbg !16
  %251 = trunc i64 %250 to i32, !dbg !16
  %252 = insertelement <4 x i32> %249, i32 %251, i64 1, !dbg !16
  %253 = insertelement <4 x i32> %252, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1586 = insertelement <4 x i32> %253, i32 131072, i64 3, !dbg !16
  %254 = ptrtoint ptr addrspace(3) %244 to i32, !dbg !16
  %dma.buffer.lds1587 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %254), !dbg !16
  %255 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1587, i32 %dma.buffer.offset1585, <4 x i32> %dma.buffer.rsrc1586) #10, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %256 = add nuw i64 %207, 1024, !dbg !16
  %257 = icmp slt i64 %256, 0, !dbg !16
  %258 = sub nuw i64 -1025, %207, !dbg !16
  %259 = select i1 %257, i64 %258, i64 %256, !dbg !16
  %260 = sdiv i64 %259, 4096, !dbg !16
  %.lobit37 = ashr i64 %256, 63, !dbg !16
  %261 = xor i64 %260, %.lobit37, !dbg !16
  %262 = add nsw i64 %261, %16, !dbg !16
  %263 = add nsw i64 %218, %197, !dbg !16
  %264 = add nsw i64 %263, 32, !dbg !16
  %265 = icmp slt i64 %263, -32, !dbg !16
  %266 = sub nsw i64 -33, %263, !dbg !16
  %267 = select i1 %265, i64 %266, i64 %264, !dbg !16
  %268 = sdiv i64 %267, 128, !dbg !16
  %269 = sext i1 %265 to i64, !dbg !16
  %270 = xor i64 %268, %269, !dbg !16
  %271 = shl nsw i64 %270, 7, !dbg !16
  %272 = sub i64 %197, %271, !dbg !16
  %273 = shl i64 %262, 19, !dbg !16
  %274 = and i64 %273, 16252928, !dbg !16
  %275 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %274, !dbg !16
  %276 = getelementptr [2 x i8], ptr addrspace(1) %275, i64 %218, !dbg !16
  %277 = getelementptr [2 x i8], ptr addrspace(1) %276, i64 %272, !dbg !16
  %278 = getelementptr i8, ptr addrspace(1) %277, i64 64, !dbg !16
  %279 = getelementptr [2 x i8], ptr addrspace(1) %278, i64 %224, !dbg !16
  %address.local468 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local436, !dbg !7
  %address.local469 = ptrtoint ptr addrspace(3) %address.local468 to i64, !dbg !7
  %address.local470 = trunc i64 %address.local469 to i32, !dbg !7
  %280 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local470), !dbg !16
  %281 = zext i32 %280 to i64, !dbg !16
  %282 = inttoptr i64 %281 to ptr addrspace(3), !dbg !16
  %address.local471 = getelementptr [2 x i8], ptr addrspace(1) %279, i64 %address.local449, !dbg !16
  %283 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %284 = ptrtoint ptr addrspace(1) %address.local471 to i64, !dbg !16
  %285 = sub i64 %284, %283, !dbg !16
  %dma.buffer.offset1582 = trunc i64 %285 to i32, !dbg !16
  %286 = trunc i64 %283 to i32, !dbg !16
  %287 = insertelement <4 x i32> poison, i32 %286, i64 0, !dbg !16
  %288 = lshr i64 %283, 32, !dbg !16
  %289 = trunc i64 %288 to i32, !dbg !16
  %290 = insertelement <4 x i32> %287, i32 %289, i64 1, !dbg !16
  %291 = insertelement <4 x i32> %290, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1583 = insertelement <4 x i32> %291, i32 131072, i64 3, !dbg !16
  %292 = ptrtoint ptr addrspace(3) %282 to i32, !dbg !16
  %dma.buffer.lds1584 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %292), !dbg !16
  %293 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1584, i32 %dma.buffer.offset1582, <4 x i32> %dma.buffer.rsrc1583) #10, !dbg !16
  %address.local472 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local453, !dbg !7
  %address.local473 = ptrtoint ptr addrspace(3) %address.local472 to i64, !dbg !7
  %address.local474 = trunc i64 %address.local473 to i32, !dbg !7
  %294 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local474), !dbg !16
  %295 = zext i32 %294 to i64, !dbg !16
  %296 = inttoptr i64 %295 to ptr addrspace(3), !dbg !16
  %address.local475 = getelementptr [2 x i8], ptr addrspace(1) %279, i64 %address.local466, !dbg !16
  %297 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %298 = ptrtoint ptr addrspace(1) %address.local475 to i64, !dbg !16
  %299 = sub i64 %298, %297, !dbg !16
  %dma.buffer.offset1579 = trunc i64 %299 to i32, !dbg !16
  %300 = trunc i64 %297 to i32, !dbg !16
  %301 = insertelement <4 x i32> poison, i32 %300, i64 0, !dbg !16
  %302 = lshr i64 %297, 32, !dbg !16
  %303 = trunc i64 %302 to i32, !dbg !16
  %304 = insertelement <4 x i32> %301, i32 %303, i64 1, !dbg !16
  %305 = insertelement <4 x i32> %304, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1580 = insertelement <4 x i32> %305, i32 131072, i64 3, !dbg !16
  %306 = ptrtoint ptr addrspace(3) %296 to i32, !dbg !16
  %dma.buffer.lds1581 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %306), !dbg !16
  %307 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1581, i32 %dma.buffer.offset1579, <4 x i32> %dma.buffer.rsrc1580) #10, !dbg !16
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
  %address.local487 = lshr i64 %address.local434, 2, !dbg !10
  %address.local488 = and i64 %address.local487, 12, !dbg !10
  %address.local489 = add nsw i64 %address.local486, %address.local488, !dbg !17
  %address.local490 = and i64 %address.local434, 63, !dbg !10
  %address.local491 = lshr i64 %address.local490, 5, !dbg !10
  %address.local492 = mul nuw nsw i64 %address.local491, 1016, !dbg !10
  %address.local493 = add i64 %address.local489, %address.local492, !dbg !17
  %address.local494 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local493, !dbg !17
  %308 = load <4 x half>, ptr addrspace(3) %address.local494, align 8, !dbg !17
  %309 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %308, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1426 = getelementptr i8, ptr addrspace(3) %address.local494, i64 4096, !dbg !17
  %310 = load <4 x half>, ptr addrspace(3) %lds.const.gep1426, align 8, !dbg !17
  %address.local505 = shl nuw nsw i32 %address.thread433, 1, !dbg !17
  %address.local506 = and i32 %address.local505, 24, !dbg !17
  %address.local507 = shl nuw nsw i32 %address.thread433, 8, !dbg !17
  %address.local508 = and i32 %address.local507, 768, !dbg !17
  %address.local509 = or disjoint i32 %address.local506, %address.local508, !dbg !17
  %address.local510 = shl nuw nsw i32 %address.thread433, 5, !dbg !17
  %address.local511 = and i32 %address.local510, 96, !dbg !17
  %address.local512 = or disjoint i32 %address.local509, %address.local511, !dbg !17
  %address.local513 = zext nneg i32 %address.local512 to i64, !dbg !17
  %address.base = call i64 asm sideeffect "", "=v,0"(i64 %205) #9, !dbg !17, !frisk.register_lifetimes !1
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
  %311 = load <4 x half>, ptr addrspace(3) %address.local523, align 8, !dbg !17
  %312 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %310, <4 x float> %309) #9, !dbg !17
  %313 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %311, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1427 = getelementptr i8, ptr addrspace(3) %address.local523, i64 4096, !dbg !17
  %314 = load <4 x half>, ptr addrspace(3) %lds.const.gep1427, align 8, !dbg !17
  %lds.const.gep1428 = getelementptr i8, ptr addrspace(3) %address.local494, i64 128, !dbg !17
  %315 = load <4 x half>, ptr addrspace(3) %lds.const.gep1428, align 8, !dbg !17
  %316 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %314, <4 x float> %313) #9, !dbg !17
  %317 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %315, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1429 = getelementptr i8, ptr addrspace(3) %address.local494, i64 4224, !dbg !17
  %318 = load <4 x half>, ptr addrspace(3) %lds.const.gep1429, align 8, !dbg !17
  %lds.const.gep1430 = getelementptr i8, ptr addrspace(3) %address.local523, i64 128, !dbg !17
  %319 = load <4 x half>, ptr addrspace(3) %lds.const.gep1430, align 8, !dbg !17
  %320 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %318, <4 x float> %317) #9, !dbg !17
  %321 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %319, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1431 = getelementptr i8, ptr addrspace(3) %address.local523, i64 4224, !dbg !17
  %322 = load <4 x half>, ptr addrspace(3) %lds.const.gep1431, align 8, !dbg !17
  %lds.const.gep1432 = getelementptr i8, ptr addrspace(3) %address.local494, i64 1024, !dbg !17
  %323 = load <4 x half>, ptr addrspace(3) %lds.const.gep1432, align 8, !dbg !17
  %324 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %322, <4 x float> %321) #9, !dbg !17
  %325 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %323, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1433 = getelementptr i8, ptr addrspace(3) %address.local494, i64 5120, !dbg !17
  %326 = load <4 x half>, ptr addrspace(3) %lds.const.gep1433, align 8, !dbg !17
  %lds.const.gep1434 = getelementptr i8, ptr addrspace(3) %address.local523, i64 1024, !dbg !17
  %327 = load <4 x half>, ptr addrspace(3) %lds.const.gep1434, align 8, !dbg !17
  %328 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %326, <4 x float> %325) #9, !dbg !17
  %329 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %327, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1435 = getelementptr i8, ptr addrspace(3) %address.local523, i64 5120, !dbg !17
  %330 = load <4 x half>, ptr addrspace(3) %lds.const.gep1435, align 8, !dbg !17
  %lds.const.gep1436 = getelementptr i8, ptr addrspace(3) %address.local494, i64 1152, !dbg !17
  %331 = load <4 x half>, ptr addrspace(3) %lds.const.gep1436, align 8, !dbg !17
  %332 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %330, <4 x float> %329) #9, !dbg !17
  %333 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %331, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1437 = getelementptr i8, ptr addrspace(3) %address.local494, i64 5248, !dbg !17
  %334 = load <4 x half>, ptr addrspace(3) %lds.const.gep1437, align 8, !dbg !17
  %lds.const.gep1438 = getelementptr i8, ptr addrspace(3) %address.local523, i64 1152, !dbg !17
  %335 = load <4 x half>, ptr addrspace(3) %lds.const.gep1438, align 8, !dbg !17
  %336 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %334, <4 x float> %333) #9, !dbg !17
  %337 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %335, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1439 = getelementptr i8, ptr addrspace(3) %address.local523, i64 5248, !dbg !17
  %338 = load <4 x half>, ptr addrspace(3) %lds.const.gep1439, align 8, !dbg !17
  %339 = load <4 x half>, ptr addrspace(3) %address.local494, align 8, !dbg !17
  %340 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %338, <4 x float> %337) #9, !dbg !17
  %341 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %339, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1440 = getelementptr i8, ptr addrspace(3) %address.local494, i64 4096, !dbg !17
  %342 = load <4 x half>, ptr addrspace(3) %lds.const.gep1440, align 8, !dbg !17
  %343 = load <4 x half>, ptr addrspace(3) %address.local523, align 8, !dbg !17
  %344 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %342, <4 x float> %341) #9, !dbg !17
  %345 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %343, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1441 = getelementptr i8, ptr addrspace(3) %address.local523, i64 4096, !dbg !17
  %346 = load <4 x half>, ptr addrspace(3) %lds.const.gep1441, align 8, !dbg !17
  %lds.const.gep1442 = getelementptr i8, ptr addrspace(3) %address.local494, i64 128, !dbg !17
  %347 = load <4 x half>, ptr addrspace(3) %lds.const.gep1442, align 8, !dbg !17
  %348 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %346, <4 x float> %345) #9, !dbg !17
  %349 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %347, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1443 = getelementptr i8, ptr addrspace(3) %address.local494, i64 4224, !dbg !17
  %350 = load <4 x half>, ptr addrspace(3) %lds.const.gep1443, align 8, !dbg !17
  %lds.const.gep1444 = getelementptr i8, ptr addrspace(3) %address.local523, i64 128, !dbg !17
  %351 = load <4 x half>, ptr addrspace(3) %lds.const.gep1444, align 8, !dbg !17
  %352 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %350, <4 x float> %349) #9, !dbg !17
  %353 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %351, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1445 = getelementptr i8, ptr addrspace(3) %address.local523, i64 4224, !dbg !17
  %354 = load <4 x half>, ptr addrspace(3) %lds.const.gep1445, align 8, !dbg !17
  %lds.const.gep1446 = getelementptr i8, ptr addrspace(3) %address.local494, i64 1024, !dbg !17
  %355 = load <4 x half>, ptr addrspace(3) %lds.const.gep1446, align 8, !dbg !17
  %356 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %354, <4 x float> %353) #9, !dbg !17
  %357 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %355, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1447 = getelementptr i8, ptr addrspace(3) %address.local494, i64 5120, !dbg !17
  %358 = load <4 x half>, ptr addrspace(3) %lds.const.gep1447, align 8, !dbg !17
  %lds.const.gep1448 = getelementptr i8, ptr addrspace(3) %address.local523, i64 1024, !dbg !17
  %359 = load <4 x half>, ptr addrspace(3) %lds.const.gep1448, align 8, !dbg !17
  %360 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %358, <4 x float> %357) #9, !dbg !17
  %361 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %359, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1449 = getelementptr i8, ptr addrspace(3) %address.local523, i64 5120, !dbg !17
  %362 = load <4 x half>, ptr addrspace(3) %lds.const.gep1449, align 8, !dbg !17
  %lds.const.gep1450 = getelementptr i8, ptr addrspace(3) %address.local494, i64 1152, !dbg !17
  %363 = load <4 x half>, ptr addrspace(3) %lds.const.gep1450, align 8, !dbg !17
  %364 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %362, <4 x float> %361) #9, !dbg !17
  %365 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %363, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1451 = getelementptr i8, ptr addrspace(3) %address.local494, i64 5248, !dbg !17
  %366 = load <4 x half>, ptr addrspace(3) %lds.const.gep1451, align 8, !dbg !17
  %lds.const.gep1452 = getelementptr i8, ptr addrspace(3) %address.local523, i64 1152, !dbg !17
  %367 = load <4 x half>, ptr addrspace(3) %lds.const.gep1452, align 8, !dbg !17
  %368 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %366, <4 x float> %365) #9, !dbg !17
  %lds.const.gep1453 = getelementptr i8, ptr addrspace(3) %address.local523, i64 5248, !dbg !17
  %369 = load <4 x half>, ptr addrspace(3) %lds.const.gep1453, align 8, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %370 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %367, <4 x float> zeroinitializer) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %371 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %369, <4 x float> %370) #9, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %372 = add nuw i64 %207, 2048, !dbg !16
  %373 = icmp slt i64 %372, 0, !dbg !16
  %374 = sub nuw i64 -2049, %207, !dbg !16
  %375 = select i1 %373, i64 %374, i64 %372, !dbg !16
  %376 = sdiv i64 %375, 4096, !dbg !16
  %.lobit38 = ashr i64 %372, 63, !dbg !16
  %377 = xor i64 %376, %.lobit38, !dbg !16
  %378 = add nsw i64 %377, %16, !dbg !16
  %379 = add nsw i64 %263, 64, !dbg !16
  %380 = icmp slt i64 %263, -64, !dbg !16
  %381 = sub nsw i64 -65, %263, !dbg !16
  %382 = select i1 %380, i64 %381, i64 %379, !dbg !16
  %383 = sdiv i64 %382, 128, !dbg !16
  %384 = sext i1 %380 to i64, !dbg !16
  %385 = xor i64 %383, %384, !dbg !16
  %386 = shl nsw i64 %385, 7, !dbg !16
  %387 = sub i64 %197, %386, !dbg !16
  %388 = shl i64 %378, 19, !dbg !16
  %389 = and i64 %388, 16252928, !dbg !16
  %390 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %389, !dbg !16
  %391 = getelementptr [2 x i8], ptr addrspace(1) %390, i64 %218, !dbg !16
  %392 = getelementptr [2 x i8], ptr addrspace(1) %391, i64 %387, !dbg !16
  %393 = getelementptr i8, ptr addrspace(1) %392, i64 128, !dbg !16
  %394 = getelementptr [2 x i8], ptr addrspace(1) %393, i64 %224, !dbg !16
  %395 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local439), !dbg !16
  %396 = zext i32 %395 to i64, !dbg !16
  %397 = inttoptr i64 %396 to ptr addrspace(3), !dbg !16
  %address.local832 = getelementptr [2 x i8], ptr addrspace(1) %394, i64 %address.local449, !dbg !16
  %398 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %399 = ptrtoint ptr addrspace(1) %address.local832 to i64, !dbg !16
  %400 = sub i64 %399, %398, !dbg !16
  %dma.buffer.offset1576 = trunc i64 %400 to i32, !dbg !16
  %401 = trunc i64 %398 to i32, !dbg !16
  %402 = insertelement <4 x i32> poison, i32 %401, i64 0, !dbg !16
  %403 = lshr i64 %398, 32, !dbg !16
  %404 = trunc i64 %403 to i32, !dbg !16
  %405 = insertelement <4 x i32> %402, i32 %404, i64 1, !dbg !16
  %406 = insertelement <4 x i32> %405, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1577 = insertelement <4 x i32> %406, i32 131072, i64 3, !dbg !16
  %407 = ptrtoint ptr addrspace(3) %397 to i32, !dbg !16
  %dma.buffer.lds1578 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %407), !dbg !16
  %408 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1578, i32 %dma.buffer.offset1576, <4 x i32> %dma.buffer.rsrc1577) #10, !dbg !16
  %409 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local456), !dbg !16
  %410 = zext i32 %409 to i64, !dbg !16
  %411 = inttoptr i64 %410 to ptr addrspace(3), !dbg !16
  %address.local833 = getelementptr [2 x i8], ptr addrspace(1) %394, i64 %address.local466, !dbg !16
  %412 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %413 = ptrtoint ptr addrspace(1) %address.local833 to i64, !dbg !16
  %414 = sub i64 %413, %412, !dbg !16
  %dma.buffer.offset1573 = trunc i64 %414 to i32, !dbg !16
  %415 = trunc i64 %412 to i32, !dbg !16
  %416 = insertelement <4 x i32> poison, i32 %415, i64 0, !dbg !16
  %417 = lshr i64 %412, 32, !dbg !16
  %418 = trunc i64 %417 to i32, !dbg !16
  %419 = insertelement <4 x i32> %416, i32 %418, i64 1, !dbg !16
  %420 = insertelement <4 x i32> %419, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1574 = insertelement <4 x i32> %420, i32 131072, i64 3, !dbg !16
  %421 = ptrtoint ptr addrspace(3) %411 to i32, !dbg !16
  %dma.buffer.lds1575 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %421), !dbg !16
  %422 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1575, i32 %dma.buffer.offset1573, <4 x i32> %dma.buffer.rsrc1574) #10, !dbg !16
  %lds.const.gep1454 = getelementptr i8, ptr addrspace(3) %address.local494, i64 8192, !dbg !17
  %423 = load <4 x half>, ptr addrspace(3) %lds.const.gep1454, align 8, !dbg !17
  %424 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %423, <4 x float> %312) #9, !dbg !17
  %lds.const.gep1455 = getelementptr i8, ptr addrspace(3) %address.local494, i64 12288, !dbg !17
  %425 = load <4 x half>, ptr addrspace(3) %lds.const.gep1455, align 8, !dbg !17
  %lds.const.gep1456 = getelementptr i8, ptr addrspace(3) %address.local523, i64 8192, !dbg !17
  %426 = load <4 x half>, ptr addrspace(3) %lds.const.gep1456, align 8, !dbg !17
  %427 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %425, <4 x float> %424) #9, !dbg !17
  %428 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %426, <4 x float> %316) #9, !dbg !17
  %lds.const.gep1457 = getelementptr i8, ptr addrspace(3) %address.local523, i64 12288, !dbg !17
  %429 = load <4 x half>, ptr addrspace(3) %lds.const.gep1457, align 8, !dbg !17
  %lds.const.gep1458 = getelementptr i8, ptr addrspace(3) %address.local494, i64 8320, !dbg !17
  %430 = load <4 x half>, ptr addrspace(3) %lds.const.gep1458, align 8, !dbg !17
  %431 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %429, <4 x float> %428) #9, !dbg !17
  %432 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %430, <4 x float> %320) #9, !dbg !17
  %lds.const.gep1459 = getelementptr i8, ptr addrspace(3) %address.local494, i64 12416, !dbg !17
  %433 = load <4 x half>, ptr addrspace(3) %lds.const.gep1459, align 8, !dbg !17
  %lds.const.gep1460 = getelementptr i8, ptr addrspace(3) %address.local523, i64 8320, !dbg !17
  %434 = load <4 x half>, ptr addrspace(3) %lds.const.gep1460, align 8, !dbg !17
  %435 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %433, <4 x float> %432) #9, !dbg !17
  %436 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %434, <4 x float> %324) #9, !dbg !17
  %lds.const.gep1461 = getelementptr i8, ptr addrspace(3) %address.local523, i64 12416, !dbg !17
  %437 = load <4 x half>, ptr addrspace(3) %lds.const.gep1461, align 8, !dbg !17
  %lds.const.gep1462 = getelementptr i8, ptr addrspace(3) %address.local494, i64 9216, !dbg !17
  %438 = load <4 x half>, ptr addrspace(3) %lds.const.gep1462, align 8, !dbg !17
  %439 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %437, <4 x float> %436) #9, !dbg !17
  %440 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %438, <4 x float> %328) #9, !dbg !17
  %lds.const.gep1463 = getelementptr i8, ptr addrspace(3) %address.local494, i64 13312, !dbg !17
  %441 = load <4 x half>, ptr addrspace(3) %lds.const.gep1463, align 8, !dbg !17
  %lds.const.gep1464 = getelementptr i8, ptr addrspace(3) %address.local523, i64 9216, !dbg !17
  %442 = load <4 x half>, ptr addrspace(3) %lds.const.gep1464, align 8, !dbg !17
  %443 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %441, <4 x float> %440) #9, !dbg !17
  %444 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %442, <4 x float> %332) #9, !dbg !17
  %lds.const.gep1465 = getelementptr i8, ptr addrspace(3) %address.local523, i64 13312, !dbg !17
  %445 = load <4 x half>, ptr addrspace(3) %lds.const.gep1465, align 8, !dbg !17
  %lds.const.gep1466 = getelementptr i8, ptr addrspace(3) %address.local494, i64 9344, !dbg !17
  %446 = load <4 x half>, ptr addrspace(3) %lds.const.gep1466, align 8, !dbg !17
  %447 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %445, <4 x float> %444) #9, !dbg !17
  %448 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %446, <4 x float> %336) #9, !dbg !17
  %lds.const.gep1467 = getelementptr i8, ptr addrspace(3) %address.local494, i64 13440, !dbg !17
  %449 = load <4 x half>, ptr addrspace(3) %lds.const.gep1467, align 8, !dbg !17
  %lds.const.gep1468 = getelementptr i8, ptr addrspace(3) %address.local523, i64 9344, !dbg !17
  %450 = load <4 x half>, ptr addrspace(3) %lds.const.gep1468, align 8, !dbg !17
  %451 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %449, <4 x float> %448) #9, !dbg !17
  %452 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %450, <4 x float> %340) #9, !dbg !17
  %lds.const.gep1469 = getelementptr i8, ptr addrspace(3) %address.local523, i64 13440, !dbg !17
  %453 = load <4 x half>, ptr addrspace(3) %lds.const.gep1469, align 8, !dbg !17
  %lds.const.gep1470 = getelementptr i8, ptr addrspace(3) %address.local494, i64 8192, !dbg !17
  %454 = load <4 x half>, ptr addrspace(3) %lds.const.gep1470, align 8, !dbg !17
  %455 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %453, <4 x float> %452) #9, !dbg !17
  %456 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %454, <4 x float> %344) #9, !dbg !17
  %lds.const.gep1471 = getelementptr i8, ptr addrspace(3) %address.local494, i64 12288, !dbg !17
  %457 = load <4 x half>, ptr addrspace(3) %lds.const.gep1471, align 8, !dbg !17
  %lds.const.gep1472 = getelementptr i8, ptr addrspace(3) %address.local523, i64 8192, !dbg !17
  %458 = load <4 x half>, ptr addrspace(3) %lds.const.gep1472, align 8, !dbg !17
  %459 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %457, <4 x float> %456) #9, !dbg !17
  %460 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %458, <4 x float> %348) #9, !dbg !17
  %lds.const.gep1473 = getelementptr i8, ptr addrspace(3) %address.local523, i64 12288, !dbg !17
  %461 = load <4 x half>, ptr addrspace(3) %lds.const.gep1473, align 8, !dbg !17
  %lds.const.gep1474 = getelementptr i8, ptr addrspace(3) %address.local494, i64 8320, !dbg !17
  %462 = load <4 x half>, ptr addrspace(3) %lds.const.gep1474, align 8, !dbg !17
  %463 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %461, <4 x float> %460) #9, !dbg !17
  %464 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %462, <4 x float> %352) #9, !dbg !17
  %lds.const.gep1475 = getelementptr i8, ptr addrspace(3) %address.local494, i64 12416, !dbg !17
  %465 = load <4 x half>, ptr addrspace(3) %lds.const.gep1475, align 8, !dbg !17
  %lds.const.gep1476 = getelementptr i8, ptr addrspace(3) %address.local523, i64 8320, !dbg !17
  %466 = load <4 x half>, ptr addrspace(3) %lds.const.gep1476, align 8, !dbg !17
  %467 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %465, <4 x float> %464) #9, !dbg !17
  %468 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %466, <4 x float> %356) #9, !dbg !17
  %lds.const.gep1477 = getelementptr i8, ptr addrspace(3) %address.local523, i64 12416, !dbg !17
  %469 = load <4 x half>, ptr addrspace(3) %lds.const.gep1477, align 8, !dbg !17
  %lds.const.gep1478 = getelementptr i8, ptr addrspace(3) %address.local494, i64 9216, !dbg !17
  %470 = load <4 x half>, ptr addrspace(3) %lds.const.gep1478, align 8, !dbg !17
  %471 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %469, <4 x float> %468) #9, !dbg !17
  %472 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %470, <4 x float> %360) #9, !dbg !17
  %lds.const.gep1479 = getelementptr i8, ptr addrspace(3) %address.local494, i64 13312, !dbg !17
  %473 = load <4 x half>, ptr addrspace(3) %lds.const.gep1479, align 8, !dbg !17
  %lds.const.gep1480 = getelementptr i8, ptr addrspace(3) %address.local523, i64 9216, !dbg !17
  %474 = load <4 x half>, ptr addrspace(3) %lds.const.gep1480, align 8, !dbg !17
  %475 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %473, <4 x float> %472) #9, !dbg !17
  %476 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %474, <4 x float> %364) #9, !dbg !17
  %lds.const.gep1481 = getelementptr i8, ptr addrspace(3) %address.local523, i64 13312, !dbg !17
  %477 = load <4 x half>, ptr addrspace(3) %lds.const.gep1481, align 8, !dbg !17
  %lds.const.gep1482 = getelementptr i8, ptr addrspace(3) %address.local494, i64 9344, !dbg !17
  %478 = load <4 x half>, ptr addrspace(3) %lds.const.gep1482, align 8, !dbg !17
  %479 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %477, <4 x float> %476) #9, !dbg !17
  %480 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %478, <4 x float> %368) #9, !dbg !17
  %lds.const.gep1483 = getelementptr i8, ptr addrspace(3) %address.local494, i64 13440, !dbg !17
  %481 = load <4 x half>, ptr addrspace(3) %lds.const.gep1483, align 8, !dbg !17
  %lds.const.gep1484 = getelementptr i8, ptr addrspace(3) %address.local523, i64 9344, !dbg !17
  %482 = load <4 x half>, ptr addrspace(3) %lds.const.gep1484, align 8, !dbg !17
  %483 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %481, <4 x float> %480) #9, !dbg !17
  %lds.const.gep1485 = getelementptr i8, ptr addrspace(3) %address.local523, i64 13440, !dbg !17
  %484 = load <4 x half>, ptr addrspace(3) %lds.const.gep1485, align 8, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %485 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %482, <4 x float> %371) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %486 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %484, <4 x float> %485) #9, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %487 = add nuw i64 %207, 3072, !dbg !16
  %488 = icmp slt i64 %487, 0, !dbg !16
  %489 = sub nuw i64 -3073, %207, !dbg !16
  %490 = select i1 %488, i64 %489, i64 %487, !dbg !16
  %491 = sdiv i64 %490, 4096, !dbg !16
  %.lobit39 = ashr i64 %487, 63, !dbg !16
  %492 = xor i64 %491, %.lobit39, !dbg !16
  %493 = add nsw i64 %492, %16, !dbg !16
  %494 = add nsw i64 %263, 96, !dbg !16
  %495 = icmp slt i64 %263, -96, !dbg !16
  %496 = sub nsw i64 -97, %263, !dbg !16
  %497 = select i1 %495, i64 %496, i64 %494, !dbg !16
  %498 = sdiv i64 %497, 128, !dbg !16
  %499 = sext i1 %495 to i64, !dbg !16
  %500 = xor i64 %498, %499, !dbg !16
  %501 = shl nsw i64 %500, 7, !dbg !16
  %502 = sub i64 %197, %501, !dbg !16
  %503 = shl i64 %493, 19, !dbg !16
  %504 = and i64 %503, 16252928, !dbg !16
  %505 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %504, !dbg !16
  %506 = getelementptr [2 x i8], ptr addrspace(1) %505, i64 %218, !dbg !16
  %507 = getelementptr [2 x i8], ptr addrspace(1) %506, i64 %502, !dbg !16
  %508 = getelementptr i8, ptr addrspace(1) %507, i64 192, !dbg !16
  %509 = getelementptr [2 x i8], ptr addrspace(1) %508, i64 %224, !dbg !16
  %510 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local470), !dbg !16
  %511 = zext i32 %510 to i64, !dbg !16
  %512 = inttoptr i64 %511 to ptr addrspace(3), !dbg !16
  %address.local850 = getelementptr [2 x i8], ptr addrspace(1) %509, i64 %address.local449, !dbg !16
  %513 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %514 = ptrtoint ptr addrspace(1) %address.local850 to i64, !dbg !16
  %515 = sub i64 %514, %513, !dbg !16
  %dma.buffer.offset1570 = trunc i64 %515 to i32, !dbg !16
  %516 = trunc i64 %513 to i32, !dbg !16
  %517 = insertelement <4 x i32> poison, i32 %516, i64 0, !dbg !16
  %518 = lshr i64 %513, 32, !dbg !16
  %519 = trunc i64 %518 to i32, !dbg !16
  %520 = insertelement <4 x i32> %517, i32 %519, i64 1, !dbg !16
  %521 = insertelement <4 x i32> %520, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1571 = insertelement <4 x i32> %521, i32 131072, i64 3, !dbg !16
  %522 = ptrtoint ptr addrspace(3) %512 to i32, !dbg !16
  %dma.buffer.lds1572 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %522), !dbg !16
  %523 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1572, i32 %dma.buffer.offset1570, <4 x i32> %dma.buffer.rsrc1571) #10, !dbg !16
  %524 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local474), !dbg !16
  %525 = zext i32 %524 to i64, !dbg !16
  %526 = inttoptr i64 %525 to ptr addrspace(3), !dbg !16
  %address.local851 = getelementptr [2 x i8], ptr addrspace(1) %509, i64 %address.local466, !dbg !16
  %527 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %528 = ptrtoint ptr addrspace(1) %address.local851 to i64, !dbg !16
  %529 = sub i64 %528, %527, !dbg !16
  %dma.buffer.offset1567 = trunc i64 %529 to i32, !dbg !16
  %530 = trunc i64 %527 to i32, !dbg !16
  %531 = insertelement <4 x i32> poison, i32 %530, i64 0, !dbg !16
  %532 = lshr i64 %527, 32, !dbg !16
  %533 = trunc i64 %532 to i32, !dbg !16
  %534 = insertelement <4 x i32> %531, i32 %533, i64 1, !dbg !16
  %535 = insertelement <4 x i32> %534, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1568 = insertelement <4 x i32> %535, i32 131072, i64 3, !dbg !16
  %536 = ptrtoint ptr addrspace(3) %526 to i32, !dbg !16
  %dma.buffer.lds1569 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %536), !dbg !16
  %537 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1569, i32 %dma.buffer.offset1567, <4 x i32> %dma.buffer.rsrc1568) #10, !dbg !16
  %538 = load <4 x half>, ptr addrspace(3) %address.local494, align 8, !dbg !17
  %539 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %175, <4 x half> %538, <4 x float> %427) #9, !dbg !17
  %lds.const.gep1486 = getelementptr i8, ptr addrspace(3) %address.local494, i64 4096, !dbg !17
  %540 = load <4 x half>, ptr addrspace(3) %lds.const.gep1486, align 8, !dbg !17
  %541 = load <4 x half>, ptr addrspace(3) %address.local523, align 8, !dbg !17
  %542 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %540, <4 x float> %539) #9, !dbg !17
  %543 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %175, <4 x half> %541, <4 x float> %431) #9, !dbg !17
  %lds.const.gep1487 = getelementptr i8, ptr addrspace(3) %address.local523, i64 4096, !dbg !17
  %544 = load <4 x half>, ptr addrspace(3) %lds.const.gep1487, align 8, !dbg !17
  %lds.const.gep1488 = getelementptr i8, ptr addrspace(3) %address.local494, i64 128, !dbg !17
  %545 = load <4 x half>, ptr addrspace(3) %lds.const.gep1488, align 8, !dbg !17
  %546 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %544, <4 x float> %543) #9, !dbg !17
  %547 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %175, <4 x half> %545, <4 x float> %435) #9, !dbg !17
  %lds.const.gep1489 = getelementptr i8, ptr addrspace(3) %address.local494, i64 4224, !dbg !17
  %548 = load <4 x half>, ptr addrspace(3) %lds.const.gep1489, align 8, !dbg !17
  %lds.const.gep1490 = getelementptr i8, ptr addrspace(3) %address.local523, i64 128, !dbg !17
  %549 = load <4 x half>, ptr addrspace(3) %lds.const.gep1490, align 8, !dbg !17
  %550 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %548, <4 x float> %547) #9, !dbg !17
  %551 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %175, <4 x half> %549, <4 x float> %439) #9, !dbg !17
  %lds.const.gep1491 = getelementptr i8, ptr addrspace(3) %address.local523, i64 4224, !dbg !17
  %552 = load <4 x half>, ptr addrspace(3) %lds.const.gep1491, align 8, !dbg !17
  %lds.const.gep1492 = getelementptr i8, ptr addrspace(3) %address.local494, i64 1024, !dbg !17
  %553 = load <4 x half>, ptr addrspace(3) %lds.const.gep1492, align 8, !dbg !17
  %554 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %552, <4 x float> %551) #9, !dbg !17
  %555 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %175, <4 x half> %553, <4 x float> %443) #9, !dbg !17
  %lds.const.gep1493 = getelementptr i8, ptr addrspace(3) %address.local494, i64 5120, !dbg !17
  %556 = load <4 x half>, ptr addrspace(3) %lds.const.gep1493, align 8, !dbg !17
  %lds.const.gep1494 = getelementptr i8, ptr addrspace(3) %address.local523, i64 1024, !dbg !17
  %557 = load <4 x half>, ptr addrspace(3) %lds.const.gep1494, align 8, !dbg !17
  %558 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %556, <4 x float> %555) #9, !dbg !17
  %559 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %175, <4 x half> %557, <4 x float> %447) #9, !dbg !17
  %lds.const.gep1495 = getelementptr i8, ptr addrspace(3) %address.local523, i64 5120, !dbg !17
  %560 = load <4 x half>, ptr addrspace(3) %lds.const.gep1495, align 8, !dbg !17
  %lds.const.gep1496 = getelementptr i8, ptr addrspace(3) %address.local494, i64 1152, !dbg !17
  %561 = load <4 x half>, ptr addrspace(3) %lds.const.gep1496, align 8, !dbg !17
  %562 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %560, <4 x float> %559) #9, !dbg !17
  %563 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %175, <4 x half> %561, <4 x float> %451) #9, !dbg !17
  %lds.const.gep1497 = getelementptr i8, ptr addrspace(3) %address.local494, i64 5248, !dbg !17
  %564 = load <4 x half>, ptr addrspace(3) %lds.const.gep1497, align 8, !dbg !17
  %lds.const.gep1498 = getelementptr i8, ptr addrspace(3) %address.local523, i64 1152, !dbg !17
  %565 = load <4 x half>, ptr addrspace(3) %lds.const.gep1498, align 8, !dbg !17
  %566 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %564, <4 x float> %563) #9, !dbg !17
  %567 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %175, <4 x half> %565, <4 x float> %455) #9, !dbg !17
  %lds.const.gep1499 = getelementptr i8, ptr addrspace(3) %address.local523, i64 5248, !dbg !17
  %568 = load <4 x half>, ptr addrspace(3) %lds.const.gep1499, align 8, !dbg !17
  %569 = load <4 x half>, ptr addrspace(3) %address.local494, align 8, !dbg !17
  %570 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %568, <4 x float> %567) #9, !dbg !17
  %571 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %569, <4 x float> %459) #9, !dbg !17
  %lds.const.gep1500 = getelementptr i8, ptr addrspace(3) %address.local494, i64 4096, !dbg !17
  %572 = load <4 x half>, ptr addrspace(3) %lds.const.gep1500, align 8, !dbg !17
  %573 = load <4 x half>, ptr addrspace(3) %address.local523, align 8, !dbg !17
  %574 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %179, <4 x half> %572, <4 x float> %571) #9, !dbg !17
  %575 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %573, <4 x float> %463) #9, !dbg !17
  %lds.const.gep1501 = getelementptr i8, ptr addrspace(3) %address.local523, i64 4096, !dbg !17
  %576 = load <4 x half>, ptr addrspace(3) %lds.const.gep1501, align 8, !dbg !17
  %lds.const.gep1502 = getelementptr i8, ptr addrspace(3) %address.local494, i64 128, !dbg !17
  %577 = load <4 x half>, ptr addrspace(3) %lds.const.gep1502, align 8, !dbg !17
  %578 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %179, <4 x half> %576, <4 x float> %575) #9, !dbg !17
  %579 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %577, <4 x float> %467) #9, !dbg !17
  %lds.const.gep1503 = getelementptr i8, ptr addrspace(3) %address.local494, i64 4224, !dbg !17
  %580 = load <4 x half>, ptr addrspace(3) %lds.const.gep1503, align 8, !dbg !17
  %lds.const.gep1504 = getelementptr i8, ptr addrspace(3) %address.local523, i64 128, !dbg !17
  %581 = load <4 x half>, ptr addrspace(3) %lds.const.gep1504, align 8, !dbg !17
  %582 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %179, <4 x half> %580, <4 x float> %579) #9, !dbg !17
  %583 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %581, <4 x float> %471) #9, !dbg !17
  %lds.const.gep1505 = getelementptr i8, ptr addrspace(3) %address.local523, i64 4224, !dbg !17
  %584 = load <4 x half>, ptr addrspace(3) %lds.const.gep1505, align 8, !dbg !17
  %lds.const.gep1506 = getelementptr i8, ptr addrspace(3) %address.local494, i64 1024, !dbg !17
  %585 = load <4 x half>, ptr addrspace(3) %lds.const.gep1506, align 8, !dbg !17
  %586 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %179, <4 x half> %584, <4 x float> %583) #9, !dbg !17
  %587 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %585, <4 x float> %475) #9, !dbg !17
  %lds.const.gep1507 = getelementptr i8, ptr addrspace(3) %address.local494, i64 5120, !dbg !17
  %588 = load <4 x half>, ptr addrspace(3) %lds.const.gep1507, align 8, !dbg !17
  %lds.const.gep1508 = getelementptr i8, ptr addrspace(3) %address.local523, i64 1024, !dbg !17
  %589 = load <4 x half>, ptr addrspace(3) %lds.const.gep1508, align 8, !dbg !17
  %590 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %179, <4 x half> %588, <4 x float> %587) #9, !dbg !17
  %591 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %589, <4 x float> %479) #9, !dbg !17
  %lds.const.gep1509 = getelementptr i8, ptr addrspace(3) %address.local523, i64 5120, !dbg !17
  %592 = load <4 x half>, ptr addrspace(3) %lds.const.gep1509, align 8, !dbg !17
  %lds.const.gep1510 = getelementptr i8, ptr addrspace(3) %address.local494, i64 1152, !dbg !17
  %593 = load <4 x half>, ptr addrspace(3) %lds.const.gep1510, align 8, !dbg !17
  %594 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %179, <4 x half> %592, <4 x float> %591) #9, !dbg !17
  %595 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %593, <4 x float> %483) #9, !dbg !17
  %lds.const.gep1511 = getelementptr i8, ptr addrspace(3) %address.local494, i64 5248, !dbg !17
  %596 = load <4 x half>, ptr addrspace(3) %lds.const.gep1511, align 8, !dbg !17
  %lds.const.gep1512 = getelementptr i8, ptr addrspace(3) %address.local523, i64 1152, !dbg !17
  %597 = load <4 x half>, ptr addrspace(3) %lds.const.gep1512, align 8, !dbg !17
  %598 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %179, <4 x half> %596, <4 x float> %595) #9, !dbg !17
  %lds.const.gep1513 = getelementptr i8, ptr addrspace(3) %address.local523, i64 5248, !dbg !17
  %599 = load <4 x half>, ptr addrspace(3) %lds.const.gep1513, align 8, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %600 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %597, <4 x float> %486) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %601 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %179, <4 x half> %599, <4 x float> %600) #9, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %602 = add nuw nsw i64 %218, %16, !dbg !18
  %603 = shl i64 %602, 19, !dbg !18
  %604 = and i64 %603, 16252928, !dbg !18
  %605 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %604, !dbg !18
  %606 = getelementptr [2 x i8], ptr addrspace(1) %605, i64 %224, !dbg !18
  %607 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local439), !dbg !18
  %608 = zext i32 %607 to i64, !dbg !18
  %609 = inttoptr i64 %608 to ptr addrspace(3), !dbg !18
  %address.local852 = shl nuw nsw i64 %address.local434, 5, !dbg !18
  %address.local853 = and i64 %address.local852, 8064, !dbg !18
  %address.local854 = shl nuw nsw i64 %address.local434, 3, !dbg !18
  %address.local855 = and i64 %address.local854, 24, !dbg !18
  %address.local856 = or disjoint i64 %address.local853, %address.local855, !dbg !18
  %address.local857 = getelementptr [2 x i8], ptr addrspace(1) %606, i64 %address.local856, !dbg !18
  %610 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %611 = ptrtoint ptr addrspace(1) %address.local857 to i64, !dbg !18
  %612 = sub i64 %611, %610, !dbg !18
  %dma.buffer.offset1564 = trunc i64 %612 to i32, !dbg !18
  %613 = trunc i64 %610 to i32, !dbg !18
  %614 = insertelement <4 x i32> poison, i32 %613, i64 0, !dbg !18
  %615 = lshr i64 %610, 32, !dbg !18
  %616 = trunc i64 %615 to i32, !dbg !18
  %617 = insertelement <4 x i32> %614, i32 %616, i64 1, !dbg !18
  %618 = insertelement <4 x i32> %617, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1565 = insertelement <4 x i32> %618, i32 131072, i64 3, !dbg !18
  %619 = ptrtoint ptr addrspace(3) %609 to i32, !dbg !18
  %dma.buffer.lds1566 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %619), !dbg !18
  %620 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1566, i32 %dma.buffer.offset1564, <4 x i32> %dma.buffer.rsrc1565) #10, !dbg !18
  %621 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local456), !dbg !18
  %622 = zext i32 %621 to i64, !dbg !18
  %623 = inttoptr i64 %622 to ptr addrspace(3), !dbg !18
  %address.local858 = shl nuw nsw i64 %address.local451, 5, !dbg !18
  %address.local859 = and i64 %address.local858, 16256, !dbg !18
  %address.local860 = shl nuw nsw i64 %address.local434, 3, !dbg !18
  %address.local861 = and i64 %address.local860, 24, !dbg !18
  %address.local862 = or disjoint i64 %address.local859, %address.local861, !dbg !18
  %address.local863 = getelementptr [2 x i8], ptr addrspace(1) %606, i64 %address.local862, !dbg !18
  %624 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %625 = ptrtoint ptr addrspace(1) %address.local863 to i64, !dbg !18
  %626 = sub i64 %625, %624, !dbg !18
  %dma.buffer.offset1561 = trunc i64 %626 to i32, !dbg !18
  %627 = trunc i64 %624 to i32, !dbg !18
  %628 = insertelement <4 x i32> poison, i32 %627, i64 0, !dbg !18
  %629 = lshr i64 %624, 32, !dbg !18
  %630 = trunc i64 %629 to i32, !dbg !18
  %631 = insertelement <4 x i32> %628, i32 %630, i64 1, !dbg !18
  %632 = insertelement <4 x i32> %631, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1562 = insertelement <4 x i32> %632, i32 131072, i64 3, !dbg !18
  %633 = ptrtoint ptr addrspace(3) %623 to i32, !dbg !18
  %dma.buffer.lds1563 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %633), !dbg !18
  %634 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1563, i32 %dma.buffer.offset1561, <4 x i32> %dma.buffer.rsrc1562) #10, !dbg !18
  %lds.const.gep1514 = getelementptr i8, ptr addrspace(3) %address.local494, i64 8192, !dbg !17
  %635 = load <4 x half>, ptr addrspace(3) %lds.const.gep1514, align 8, !dbg !17
  %636 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %635, <4 x float> %542) #9, !dbg !17
  %lds.const.gep1515 = getelementptr i8, ptr addrspace(3) %address.local494, i64 12288, !dbg !17
  %637 = load <4 x half>, ptr addrspace(3) %lds.const.gep1515, align 8, !dbg !17
  %lds.const.gep1516 = getelementptr i8, ptr addrspace(3) %address.local523, i64 8192, !dbg !17
  %638 = load <4 x half>, ptr addrspace(3) %lds.const.gep1516, align 8, !dbg !17
  %639 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %637, <4 x float> %636) #9, !dbg !17
  %640 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %638, <4 x float> %546) #9, !dbg !17
  %lds.const.gep1517 = getelementptr i8, ptr addrspace(3) %address.local523, i64 12288, !dbg !17
  %641 = load <4 x half>, ptr addrspace(3) %lds.const.gep1517, align 8, !dbg !17
  %lds.const.gep1518 = getelementptr i8, ptr addrspace(3) %address.local494, i64 8320, !dbg !17
  %642 = load <4 x half>, ptr addrspace(3) %lds.const.gep1518, align 8, !dbg !17
  %643 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %641, <4 x float> %640) #9, !dbg !17
  %644 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %642, <4 x float> %550) #9, !dbg !17
  %lds.const.gep1519 = getelementptr i8, ptr addrspace(3) %address.local494, i64 12416, !dbg !17
  %645 = load <4 x half>, ptr addrspace(3) %lds.const.gep1519, align 8, !dbg !17
  %lds.const.gep1520 = getelementptr i8, ptr addrspace(3) %address.local523, i64 8320, !dbg !17
  %646 = load <4 x half>, ptr addrspace(3) %lds.const.gep1520, align 8, !dbg !17
  %647 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %645, <4 x float> %644) #9, !dbg !17
  %648 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %646, <4 x float> %554) #9, !dbg !17
  %lds.const.gep1521 = getelementptr i8, ptr addrspace(3) %address.local523, i64 12416, !dbg !17
  %649 = load <4 x half>, ptr addrspace(3) %lds.const.gep1521, align 8, !dbg !17
  %lds.const.gep1522 = getelementptr i8, ptr addrspace(3) %address.local494, i64 9216, !dbg !17
  %650 = load <4 x half>, ptr addrspace(3) %lds.const.gep1522, align 8, !dbg !17
  %651 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %649, <4 x float> %648) #9, !dbg !17
  %652 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %650, <4 x float> %558) #9, !dbg !17
  %lds.const.gep1523 = getelementptr i8, ptr addrspace(3) %address.local494, i64 13312, !dbg !17
  %653 = load <4 x half>, ptr addrspace(3) %lds.const.gep1523, align 8, !dbg !17
  %lds.const.gep1524 = getelementptr i8, ptr addrspace(3) %address.local523, i64 9216, !dbg !17
  %654 = load <4 x half>, ptr addrspace(3) %lds.const.gep1524, align 8, !dbg !17
  %655 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %653, <4 x float> %652) #9, !dbg !17
  %656 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %654, <4 x float> %562) #9, !dbg !17
  %lds.const.gep1525 = getelementptr i8, ptr addrspace(3) %address.local523, i64 13312, !dbg !17
  %657 = load <4 x half>, ptr addrspace(3) %lds.const.gep1525, align 8, !dbg !17
  %lds.const.gep1526 = getelementptr i8, ptr addrspace(3) %address.local494, i64 9344, !dbg !17
  %658 = load <4 x half>, ptr addrspace(3) %lds.const.gep1526, align 8, !dbg !17
  %659 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %657, <4 x float> %656) #9, !dbg !17
  %660 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %658, <4 x float> %566) #9, !dbg !17
  %lds.const.gep1527 = getelementptr i8, ptr addrspace(3) %address.local494, i64 13440, !dbg !17
  %661 = load <4 x half>, ptr addrspace(3) %lds.const.gep1527, align 8, !dbg !17
  %lds.const.gep1528 = getelementptr i8, ptr addrspace(3) %address.local523, i64 9344, !dbg !17
  %662 = load <4 x half>, ptr addrspace(3) %lds.const.gep1528, align 8, !dbg !17
  %663 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %661, <4 x float> %660) #9, !dbg !17
  %664 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %662, <4 x float> %570) #9, !dbg !17
  %lds.const.gep1529 = getelementptr i8, ptr addrspace(3) %address.local523, i64 13440, !dbg !17
  %665 = load <4 x half>, ptr addrspace(3) %lds.const.gep1529, align 8, !dbg !17
  %lds.const.gep1530 = getelementptr i8, ptr addrspace(3) %address.local494, i64 8192, !dbg !17
  %666 = load <4 x half>, ptr addrspace(3) %lds.const.gep1530, align 8, !dbg !17
  %667 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %665, <4 x float> %664) #9, !dbg !17
  %668 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %666, <4 x float> %574) #9, !dbg !17
  %lds.const.gep1531 = getelementptr i8, ptr addrspace(3) %address.local494, i64 12288, !dbg !17
  %669 = load <4 x half>, ptr addrspace(3) %lds.const.gep1531, align 8, !dbg !17
  %lds.const.gep1532 = getelementptr i8, ptr addrspace(3) %address.local523, i64 8192, !dbg !17
  %670 = load <4 x half>, ptr addrspace(3) %lds.const.gep1532, align 8, !dbg !17
  %671 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %187, <4 x half> %669, <4 x float> %668) #9, !dbg !17
  %672 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %670, <4 x float> %578) #9, !dbg !17
  %lds.const.gep1533 = getelementptr i8, ptr addrspace(3) %address.local523, i64 12288, !dbg !17
  %673 = load <4 x half>, ptr addrspace(3) %lds.const.gep1533, align 8, !dbg !17
  %lds.const.gep1534 = getelementptr i8, ptr addrspace(3) %address.local494, i64 8320, !dbg !17
  %674 = load <4 x half>, ptr addrspace(3) %lds.const.gep1534, align 8, !dbg !17
  %675 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %187, <4 x half> %673, <4 x float> %672) #9, !dbg !17
  %676 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %674, <4 x float> %582) #9, !dbg !17
  %lds.const.gep1535 = getelementptr i8, ptr addrspace(3) %address.local494, i64 12416, !dbg !17
  %677 = load <4 x half>, ptr addrspace(3) %lds.const.gep1535, align 8, !dbg !17
  %lds.const.gep1536 = getelementptr i8, ptr addrspace(3) %address.local523, i64 8320, !dbg !17
  %678 = load <4 x half>, ptr addrspace(3) %lds.const.gep1536, align 8, !dbg !17
  %679 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %187, <4 x half> %677, <4 x float> %676) #9, !dbg !17
  %680 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %678, <4 x float> %586) #9, !dbg !17
  %lds.const.gep1537 = getelementptr i8, ptr addrspace(3) %address.local523, i64 12416, !dbg !17
  %681 = load <4 x half>, ptr addrspace(3) %lds.const.gep1537, align 8, !dbg !17
  %lds.const.gep1538 = getelementptr i8, ptr addrspace(3) %address.local494, i64 9216, !dbg !17
  %682 = load <4 x half>, ptr addrspace(3) %lds.const.gep1538, align 8, !dbg !17
  %683 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %187, <4 x half> %681, <4 x float> %680) #9, !dbg !17
  %684 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %682, <4 x float> %590) #9, !dbg !17
  %lds.const.gep1539 = getelementptr i8, ptr addrspace(3) %address.local494, i64 13312, !dbg !17
  %685 = load <4 x half>, ptr addrspace(3) %lds.const.gep1539, align 8, !dbg !17
  %lds.const.gep1540 = getelementptr i8, ptr addrspace(3) %address.local523, i64 9216, !dbg !17
  %686 = load <4 x half>, ptr addrspace(3) %lds.const.gep1540, align 8, !dbg !17
  %687 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %187, <4 x half> %685, <4 x float> %684) #9, !dbg !17
  %688 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %686, <4 x float> %594) #9, !dbg !17
  %lds.const.gep1541 = getelementptr i8, ptr addrspace(3) %address.local523, i64 13312, !dbg !17
  %689 = load <4 x half>, ptr addrspace(3) %lds.const.gep1541, align 8, !dbg !17
  %lds.const.gep1542 = getelementptr i8, ptr addrspace(3) %address.local494, i64 9344, !dbg !17
  %690 = load <4 x half>, ptr addrspace(3) %lds.const.gep1542, align 8, !dbg !17
  %691 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %187, <4 x half> %689, <4 x float> %688) #9, !dbg !17
  %692 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %690, <4 x float> %598) #9, !dbg !17
  %lds.const.gep1543 = getelementptr i8, ptr addrspace(3) %address.local494, i64 13440, !dbg !17
  %693 = load <4 x half>, ptr addrspace(3) %lds.const.gep1543, align 8, !dbg !17
  %lds.const.gep1544 = getelementptr i8, ptr addrspace(3) %address.local523, i64 9344, !dbg !17
  %694 = load <4 x half>, ptr addrspace(3) %lds.const.gep1544, align 8, !dbg !17
  %695 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %187, <4 x half> %693, <4 x float> %692) #9, !dbg !17
  %lds.const.gep1545 = getelementptr i8, ptr addrspace(3) %address.local523, i64 13440, !dbg !17
  %696 = load <4 x half>, ptr addrspace(3) %lds.const.gep1545, align 8, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %697 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %694, <4 x float> %601) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %698 = or disjoint i64 %215, 127, !dbg !13
  %699 = icmp uge i64 %64, %698, !dbg !13
  %700 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %187, <4 x half> %696, <4 x float> %697) #9, !dbg !17
  %701 = and i1 %190, %699, !dbg !13
  br i1 %701, label %702, label %703, !dbg !13

702:                                              ; preds = %214
  br label %847, !dbg !13

703:                                              ; preds = %214
  %704 = add i64 %64, %32, !dbg !13
  %705 = shl nuw nsw i64 %65, 4, !dbg !13
  %706 = add i64 %704, %705, !dbg !13
  %707 = and i32 %31, 240, !dbg !13
  %index.compact = zext nneg i32 %707 to i64, !dbg !13
  %708 = sub i64 %706, %index.compact, !dbg !13
  %709 = and i64 %32, 48, !dbg !13
  %710 = or disjoint i64 %215, %709, !dbg !13
  %711 = add i64 %708, 1, !dbg !23
  %712 = lshr i64 %711, 32, !dbg !24
  %compare.high = trunc i64 %712 to i32, !dbg !24
  %713 = lshr i64 %710, 32, !dbg !24
  %compare.high1157 = trunc i64 %713 to i32, !dbg !24
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high1157, !dbg !24
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high1157, !dbg !24
  %compare.low.base = trunc i64 %711 to i32, !dbg !24
  %compare.low.base1158 = trunc i64 %710 to i32, !dbg !24
  %compare.low = icmp ugt i32 %compare.low.base, %compare.low.base1158, !dbg !24
  %compare.result = select i1 %compare.high.equal, i1 %compare.low, i1 %compare.high.order, !dbg !24
  %714 = select i1 %compare.result, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 1) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local = or i32 %compare.local.base, 1, !dbg !24
  %compare.low1159 = icmp ugt i32 %compare.low.base, %compare.local, !dbg !24
  %compare.result1160 = select i1 %compare.high.equal, i1 %compare.low1159, i1 %compare.high.order, !dbg !24
  %715 = select i1 %compare.result1160, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1161 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 2) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1162 = or i32 %compare.local.base1161, 2, !dbg !24
  %compare.low1163 = icmp ugt i32 %compare.low.base, %compare.local1162, !dbg !24
  %compare.result1164 = select i1 %compare.high.equal, i1 %compare.low1163, i1 %compare.high.order, !dbg !24
  %716 = select i1 %compare.result1164, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1165 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 3) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1166 = or i32 %compare.local.base1165, 3, !dbg !24
  %compare.low1167 = icmp ugt i32 %compare.low.base, %compare.local1166, !dbg !24
  %compare.result1168 = select i1 %compare.high.equal, i1 %compare.low1167, i1 %compare.high.order, !dbg !24
  %717 = select i1 %compare.result1168, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1169 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 4) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1170 = or i32 %compare.local.base1169, 4, !dbg !24
  %compare.low1171 = icmp ugt i32 %compare.low.base, %compare.local1170, !dbg !24
  %compare.result1172 = select i1 %compare.high.equal, i1 %compare.low1171, i1 %compare.high.order, !dbg !24
  %718 = select i1 %compare.result1172, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1173 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 5) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1174 = or i32 %compare.local.base1173, 5, !dbg !24
  %compare.low1175 = icmp ugt i32 %compare.low.base, %compare.local1174, !dbg !24
  %compare.result1176 = select i1 %compare.high.equal, i1 %compare.low1175, i1 %compare.high.order, !dbg !24
  %719 = select i1 %compare.result1176, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1177 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 6) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1178 = or i32 %compare.local.base1177, 6, !dbg !24
  %compare.low1179 = icmp ugt i32 %compare.low.base, %compare.local1178, !dbg !24
  %compare.result1180 = select i1 %compare.high.equal, i1 %compare.low1179, i1 %compare.high.order, !dbg !24
  %720 = select i1 %compare.result1180, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1181 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 7) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1182 = or i32 %compare.local.base1181, 7, !dbg !24
  %compare.low1183 = icmp ugt i32 %compare.low.base, %compare.local1182, !dbg !24
  %compare.result1184 = select i1 %compare.high.equal, i1 %compare.low1183, i1 %compare.high.order, !dbg !24
  %721 = select i1 %compare.result1184, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1185 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 8) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1186 = or i32 %compare.local.base1185, 8, !dbg !24
  %compare.low1187 = icmp ugt i32 %compare.low.base, %compare.local1186, !dbg !24
  %compare.result1188 = select i1 %compare.high.equal, i1 %compare.low1187, i1 %compare.high.order, !dbg !24
  %722 = select i1 %compare.result1188, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1189 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 9) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1190 = or i32 %compare.local.base1189, 9, !dbg !24
  %compare.low1191 = icmp ugt i32 %compare.low.base, %compare.local1190, !dbg !24
  %compare.result1192 = select i1 %compare.high.equal, i1 %compare.low1191, i1 %compare.high.order, !dbg !24
  %723 = select i1 %compare.result1192, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1193 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 10) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1194 = or i32 %compare.local.base1193, 10, !dbg !24
  %compare.low1195 = icmp ugt i32 %compare.low.base, %compare.local1194, !dbg !24
  %compare.result1196 = select i1 %compare.high.equal, i1 %compare.low1195, i1 %compare.high.order, !dbg !24
  %724 = select i1 %compare.result1196, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1197 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 11) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1198 = or i32 %compare.local.base1197, 11, !dbg !24
  %compare.low1199 = icmp ugt i32 %compare.low.base, %compare.local1198, !dbg !24
  %compare.result1200 = select i1 %compare.high.equal, i1 %compare.low1199, i1 %compare.high.order, !dbg !24
  %725 = select i1 %compare.result1200, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1201 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 12) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1202 = or i32 %compare.local.base1201, 12, !dbg !24
  %compare.low1203 = icmp ugt i32 %compare.low.base, %compare.local1202, !dbg !24
  %compare.result1204 = select i1 %compare.high.equal, i1 %compare.low1203, i1 %compare.high.order, !dbg !24
  %726 = select i1 %compare.result1204, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1205 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 13) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1206 = or i32 %compare.local.base1205, 13, !dbg !24
  %compare.low1207 = icmp ugt i32 %compare.low.base, %compare.local1206, !dbg !24
  %compare.result1208 = select i1 %compare.high.equal, i1 %compare.low1207, i1 %compare.high.order, !dbg !24
  %727 = select i1 %compare.result1208, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1209 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 14) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1210 = or i32 %compare.local.base1209, 14, !dbg !24
  %compare.low1211 = icmp ugt i32 %compare.low.base, %compare.local1210, !dbg !24
  %compare.result1212 = select i1 %compare.high.equal, i1 %compare.low1211, i1 %compare.high.order, !dbg !24
  %728 = select i1 %compare.result1212, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1213 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 15) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1214 = or i32 %compare.local.base1213, 15, !dbg !24
  %compare.low1215 = icmp ugt i32 %compare.low.base, %compare.local1214, !dbg !24
  %compare.result1216 = select i1 %compare.high.equal, i1 %compare.low1215, i1 %compare.high.order, !dbg !24
  %729 = select i1 %compare.result1216, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1217 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 16) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1218 = or i32 %compare.local.base1217, 64, !dbg !24
  %compare.low1219 = icmp ugt i32 %compare.low.base, %compare.local1218, !dbg !24
  %compare.result1220 = select i1 %compare.high.equal, i1 %compare.low1219, i1 %compare.high.order, !dbg !24
  %730 = select i1 %compare.result1220, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1221 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 17) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1222 = or i32 %compare.local.base1221, 65, !dbg !24
  %compare.low1223 = icmp ugt i32 %compare.low.base, %compare.local1222, !dbg !24
  %compare.result1224 = select i1 %compare.high.equal, i1 %compare.low1223, i1 %compare.high.order, !dbg !24
  %731 = select i1 %compare.result1224, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1225 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 18) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1226 = or i32 %compare.local.base1225, 66, !dbg !24
  %compare.low1227 = icmp ugt i32 %compare.low.base, %compare.local1226, !dbg !24
  %compare.result1228 = select i1 %compare.high.equal, i1 %compare.low1227, i1 %compare.high.order, !dbg !24
  %732 = select i1 %compare.result1228, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1229 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 19) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1230 = or i32 %compare.local.base1229, 67, !dbg !24
  %compare.low1231 = icmp ugt i32 %compare.low.base, %compare.local1230, !dbg !24
  %compare.result1232 = select i1 %compare.high.equal, i1 %compare.low1231, i1 %compare.high.order, !dbg !24
  %733 = select i1 %compare.result1232, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1233 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 20) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1234 = or i32 %compare.local.base1233, 68, !dbg !24
  %compare.low1235 = icmp ugt i32 %compare.low.base, %compare.local1234, !dbg !24
  %compare.result1236 = select i1 %compare.high.equal, i1 %compare.low1235, i1 %compare.high.order, !dbg !24
  %734 = select i1 %compare.result1236, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1237 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 21) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1238 = or i32 %compare.local.base1237, 69, !dbg !24
  %compare.low1239 = icmp ugt i32 %compare.low.base, %compare.local1238, !dbg !24
  %compare.result1240 = select i1 %compare.high.equal, i1 %compare.low1239, i1 %compare.high.order, !dbg !24
  %735 = select i1 %compare.result1240, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1241 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 22) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1242 = or i32 %compare.local.base1241, 70, !dbg !24
  %compare.low1243 = icmp ugt i32 %compare.low.base, %compare.local1242, !dbg !24
  %compare.result1244 = select i1 %compare.high.equal, i1 %compare.low1243, i1 %compare.high.order, !dbg !24
  %736 = select i1 %compare.result1244, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1245 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 23) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1246 = or i32 %compare.local.base1245, 71, !dbg !24
  %compare.low1247 = icmp ugt i32 %compare.low.base, %compare.local1246, !dbg !24
  %compare.result1248 = select i1 %compare.high.equal, i1 %compare.low1247, i1 %compare.high.order, !dbg !24
  %737 = select i1 %compare.result1248, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1249 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 24) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1250 = or i32 %compare.local.base1249, 72, !dbg !24
  %compare.low1251 = icmp ugt i32 %compare.low.base, %compare.local1250, !dbg !24
  %compare.result1252 = select i1 %compare.high.equal, i1 %compare.low1251, i1 %compare.high.order, !dbg !24
  %738 = select i1 %compare.result1252, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1253 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 25) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1254 = or i32 %compare.local.base1253, 73, !dbg !24
  %compare.low1255 = icmp ugt i32 %compare.low.base, %compare.local1254, !dbg !24
  %compare.result1256 = select i1 %compare.high.equal, i1 %compare.low1255, i1 %compare.high.order, !dbg !24
  %739 = select i1 %compare.result1256, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1257 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 26) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1258 = or i32 %compare.local.base1257, 74, !dbg !24
  %compare.low1259 = icmp ugt i32 %compare.low.base, %compare.local1258, !dbg !24
  %compare.result1260 = select i1 %compare.high.equal, i1 %compare.low1259, i1 %compare.high.order, !dbg !24
  %740 = select i1 %compare.result1260, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1261 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 27) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1262 = or i32 %compare.local.base1261, 75, !dbg !24
  %compare.low1263 = icmp ugt i32 %compare.low.base, %compare.local1262, !dbg !24
  %compare.result1264 = select i1 %compare.high.equal, i1 %compare.low1263, i1 %compare.high.order, !dbg !24
  %741 = select i1 %compare.result1264, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1265 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 28) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1266 = or i32 %compare.local.base1265, 76, !dbg !24
  %compare.low1267 = icmp ugt i32 %compare.low.base, %compare.local1266, !dbg !24
  %compare.result1268 = select i1 %compare.high.equal, i1 %compare.low1267, i1 %compare.high.order, !dbg !24
  %742 = select i1 %compare.result1268, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1269 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 29) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1270 = or i32 %compare.local.base1269, 77, !dbg !24
  %compare.low1271 = icmp ugt i32 %compare.low.base, %compare.local1270, !dbg !24
  %compare.result1272 = select i1 %compare.high.equal, i1 %compare.low1271, i1 %compare.high.order, !dbg !24
  %743 = select i1 %compare.result1272, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1273 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 30) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1274 = or i32 %compare.local.base1273, 78, !dbg !24
  %compare.low1275 = icmp ugt i32 %compare.low.base, %compare.local1274, !dbg !24
  %compare.result1276 = select i1 %compare.high.equal, i1 %compare.low1275, i1 %compare.high.order, !dbg !24
  %744 = select i1 %compare.result1276, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1277 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 31) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1278 = or i32 %compare.local.base1277, 79, !dbg !24
  %compare.low1279 = icmp ugt i32 %compare.low.base, %compare.local1278, !dbg !24
  %compare.result1280 = select i1 %compare.high.equal, i1 %compare.low1279, i1 %compare.high.order, !dbg !24
  %745 = select i1 %compare.result1280, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %746 = add i64 %708, 65, !dbg !23
  %747 = lshr i64 %746, 32, !dbg !24
  %compare.high1281 = trunc i64 %747 to i32, !dbg !24
  %748 = lshr i64 %710, 32, !dbg !24
  %compare.high1282 = trunc i64 %748 to i32, !dbg !24
  %compare.high.equal1283 = icmp eq i32 %compare.high1281, %compare.high1282, !dbg !24
  %compare.high.order1284 = icmp ugt i32 %compare.high1281, %compare.high1282, !dbg !24
  %compare.low.base1285 = trunc i64 %746 to i32, !dbg !24
  %compare.low.base1286 = trunc i64 %710 to i32, !dbg !24
  %compare.low1287 = icmp ugt i32 %compare.low.base1285, %compare.low.base1286, !dbg !24
  %compare.result1288 = select i1 %compare.high.equal1283, i1 %compare.low1287, i1 %compare.high.order1284, !dbg !24
  %749 = select i1 %compare.result1288, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1289 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 32) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1290 = or i32 %compare.local.base1289, 1, !dbg !24
  %compare.low1291 = icmp ugt i32 %compare.low.base1285, %compare.local1290, !dbg !24
  %compare.result1292 = select i1 %compare.high.equal1283, i1 %compare.low1291, i1 %compare.high.order1284, !dbg !24
  %750 = select i1 %compare.result1292, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1293 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 33) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1294 = or i32 %compare.local.base1293, 2, !dbg !24
  %compare.low1295 = icmp ugt i32 %compare.low.base1285, %compare.local1294, !dbg !24
  %compare.result1296 = select i1 %compare.high.equal1283, i1 %compare.low1295, i1 %compare.high.order1284, !dbg !24
  %751 = select i1 %compare.result1296, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1297 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 34) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1298 = or i32 %compare.local.base1297, 3, !dbg !24
  %compare.low1299 = icmp ugt i32 %compare.low.base1285, %compare.local1298, !dbg !24
  %compare.result1300 = select i1 %compare.high.equal1283, i1 %compare.low1299, i1 %compare.high.order1284, !dbg !24
  %752 = select i1 %compare.result1300, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1301 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 35) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1302 = or i32 %compare.local.base1301, 4, !dbg !24
  %compare.low1303 = icmp ugt i32 %compare.low.base1285, %compare.local1302, !dbg !24
  %compare.result1304 = select i1 %compare.high.equal1283, i1 %compare.low1303, i1 %compare.high.order1284, !dbg !24
  %753 = select i1 %compare.result1304, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1305 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 36) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1306 = or i32 %compare.local.base1305, 5, !dbg !24
  %compare.low1307 = icmp ugt i32 %compare.low.base1285, %compare.local1306, !dbg !24
  %compare.result1308 = select i1 %compare.high.equal1283, i1 %compare.low1307, i1 %compare.high.order1284, !dbg !24
  %754 = select i1 %compare.result1308, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1309 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 37) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1310 = or i32 %compare.local.base1309, 6, !dbg !24
  %compare.low1311 = icmp ugt i32 %compare.low.base1285, %compare.local1310, !dbg !24
  %compare.result1312 = select i1 %compare.high.equal1283, i1 %compare.low1311, i1 %compare.high.order1284, !dbg !24
  %755 = select i1 %compare.result1312, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1313 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 38) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1314 = or i32 %compare.local.base1313, 7, !dbg !24
  %compare.low1315 = icmp ugt i32 %compare.low.base1285, %compare.local1314, !dbg !24
  %compare.result1316 = select i1 %compare.high.equal1283, i1 %compare.low1315, i1 %compare.high.order1284, !dbg !24
  %756 = select i1 %compare.result1316, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1317 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 39) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1318 = or i32 %compare.local.base1317, 8, !dbg !24
  %compare.low1319 = icmp ugt i32 %compare.low.base1285, %compare.local1318, !dbg !24
  %compare.result1320 = select i1 %compare.high.equal1283, i1 %compare.low1319, i1 %compare.high.order1284, !dbg !24
  %757 = select i1 %compare.result1320, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1321 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 40) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1322 = or i32 %compare.local.base1321, 9, !dbg !24
  %compare.low1323 = icmp ugt i32 %compare.low.base1285, %compare.local1322, !dbg !24
  %compare.result1324 = select i1 %compare.high.equal1283, i1 %compare.low1323, i1 %compare.high.order1284, !dbg !24
  %758 = select i1 %compare.result1324, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1325 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 41) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1326 = or i32 %compare.local.base1325, 10, !dbg !24
  %compare.low1327 = icmp ugt i32 %compare.low.base1285, %compare.local1326, !dbg !24
  %compare.result1328 = select i1 %compare.high.equal1283, i1 %compare.low1327, i1 %compare.high.order1284, !dbg !24
  %759 = select i1 %compare.result1328, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1329 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 42) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1330 = or i32 %compare.local.base1329, 11, !dbg !24
  %compare.low1331 = icmp ugt i32 %compare.low.base1285, %compare.local1330, !dbg !24
  %compare.result1332 = select i1 %compare.high.equal1283, i1 %compare.low1331, i1 %compare.high.order1284, !dbg !24
  %760 = select i1 %compare.result1332, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1333 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 43) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1334 = or i32 %compare.local.base1333, 12, !dbg !24
  %compare.low1335 = icmp ugt i32 %compare.low.base1285, %compare.local1334, !dbg !24
  %compare.result1336 = select i1 %compare.high.equal1283, i1 %compare.low1335, i1 %compare.high.order1284, !dbg !24
  %761 = select i1 %compare.result1336, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1337 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 44) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1338 = or i32 %compare.local.base1337, 13, !dbg !24
  %compare.low1339 = icmp ugt i32 %compare.low.base1285, %compare.local1338, !dbg !24
  %compare.result1340 = select i1 %compare.high.equal1283, i1 %compare.low1339, i1 %compare.high.order1284, !dbg !24
  %762 = select i1 %compare.result1340, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1341 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 45) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1342 = or i32 %compare.local.base1341, 14, !dbg !24
  %compare.low1343 = icmp ugt i32 %compare.low.base1285, %compare.local1342, !dbg !24
  %compare.result1344 = select i1 %compare.high.equal1283, i1 %compare.low1343, i1 %compare.high.order1284, !dbg !24
  %763 = select i1 %compare.result1344, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1345 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 46) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1346 = or i32 %compare.local.base1345, 15, !dbg !24
  %compare.low1347 = icmp ugt i32 %compare.low.base1285, %compare.local1346, !dbg !24
  %compare.result1348 = select i1 %compare.high.equal1283, i1 %compare.low1347, i1 %compare.high.order1284, !dbg !24
  %764 = select i1 %compare.result1348, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1349 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 47) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1350 = or i32 %compare.local.base1349, 64, !dbg !24
  %compare.low1351 = icmp ugt i32 %compare.low.base1285, %compare.local1350, !dbg !24
  %compare.result1352 = select i1 %compare.high.equal1283, i1 %compare.low1351, i1 %compare.high.order1284, !dbg !24
  %765 = select i1 %compare.result1352, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1353 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 48) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1354 = or i32 %compare.local.base1353, 65, !dbg !24
  %compare.low1355 = icmp ugt i32 %compare.low.base1285, %compare.local1354, !dbg !24
  %compare.result1356 = select i1 %compare.high.equal1283, i1 %compare.low1355, i1 %compare.high.order1284, !dbg !24
  %766 = select i1 %compare.result1356, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1357 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 49) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1358 = or i32 %compare.local.base1357, 66, !dbg !24
  %compare.low1359 = icmp ugt i32 %compare.low.base1285, %compare.local1358, !dbg !24
  %compare.result1360 = select i1 %compare.high.equal1283, i1 %compare.low1359, i1 %compare.high.order1284, !dbg !24
  %767 = select i1 %compare.result1360, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1361 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 50) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1362 = or i32 %compare.local.base1361, 67, !dbg !24
  %compare.low1363 = icmp ugt i32 %compare.low.base1285, %compare.local1362, !dbg !24
  %compare.result1364 = select i1 %compare.high.equal1283, i1 %compare.low1363, i1 %compare.high.order1284, !dbg !24
  %768 = select i1 %compare.result1364, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1365 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 51) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1366 = or i32 %compare.local.base1365, 68, !dbg !24
  %compare.low1367 = icmp ugt i32 %compare.low.base1285, %compare.local1366, !dbg !24
  %compare.result1368 = select i1 %compare.high.equal1283, i1 %compare.low1367, i1 %compare.high.order1284, !dbg !24
  %769 = select i1 %compare.result1368, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1369 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 52) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1370 = or i32 %compare.local.base1369, 69, !dbg !24
  %compare.low1371 = icmp ugt i32 %compare.low.base1285, %compare.local1370, !dbg !24
  %compare.result1372 = select i1 %compare.high.equal1283, i1 %compare.low1371, i1 %compare.high.order1284, !dbg !24
  %770 = select i1 %compare.result1372, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1373 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 53) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1374 = or i32 %compare.local.base1373, 70, !dbg !24
  %compare.low1375 = icmp ugt i32 %compare.low.base1285, %compare.local1374, !dbg !24
  %compare.result1376 = select i1 %compare.high.equal1283, i1 %compare.low1375, i1 %compare.high.order1284, !dbg !24
  %771 = select i1 %compare.result1376, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1377 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 54) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1378 = or i32 %compare.local.base1377, 71, !dbg !24
  %compare.low1379 = icmp ugt i32 %compare.low.base1285, %compare.local1378, !dbg !24
  %compare.result1380 = select i1 %compare.high.equal1283, i1 %compare.low1379, i1 %compare.high.order1284, !dbg !24
  %772 = select i1 %compare.result1380, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1381 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 55) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1382 = or i32 %compare.local.base1381, 72, !dbg !24
  %compare.low1383 = icmp ugt i32 %compare.low.base1285, %compare.local1382, !dbg !24
  %compare.result1384 = select i1 %compare.high.equal1283, i1 %compare.low1383, i1 %compare.high.order1284, !dbg !24
  %773 = select i1 %compare.result1384, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1385 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 56) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1386 = or i32 %compare.local.base1385, 73, !dbg !24
  %compare.low1387 = icmp ugt i32 %compare.low.base1285, %compare.local1386, !dbg !24
  %compare.result1388 = select i1 %compare.high.equal1283, i1 %compare.low1387, i1 %compare.high.order1284, !dbg !24
  %774 = select i1 %compare.result1388, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1389 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 57) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1390 = or i32 %compare.local.base1389, 74, !dbg !24
  %compare.low1391 = icmp ugt i32 %compare.low.base1285, %compare.local1390, !dbg !24
  %compare.result1392 = select i1 %compare.high.equal1283, i1 %compare.low1391, i1 %compare.high.order1284, !dbg !24
  %775 = select i1 %compare.result1392, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1393 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 58) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1394 = or i32 %compare.local.base1393, 75, !dbg !24
  %compare.low1395 = icmp ugt i32 %compare.low.base1285, %compare.local1394, !dbg !24
  %compare.result1396 = select i1 %compare.high.equal1283, i1 %compare.low1395, i1 %compare.high.order1284, !dbg !24
  %776 = select i1 %compare.result1396, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1397 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 59) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1398 = or i32 %compare.local.base1397, 76, !dbg !24
  %compare.low1399 = icmp ugt i32 %compare.low.base1285, %compare.local1398, !dbg !24
  %compare.result1400 = select i1 %compare.high.equal1283, i1 %compare.low1399, i1 %compare.high.order1284, !dbg !24
  %777 = select i1 %compare.result1400, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1401 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 60) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1402 = or i32 %compare.local.base1401, 77, !dbg !24
  %compare.low1403 = icmp ugt i32 %compare.low.base1285, %compare.local1402, !dbg !24
  %compare.result1404 = select i1 %compare.high.equal1283, i1 %compare.low1403, i1 %compare.high.order1284, !dbg !24
  %778 = select i1 %compare.result1404, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1405 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 61) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1406 = or i32 %compare.local.base1405, 78, !dbg !24
  %compare.low1407 = icmp ugt i32 %compare.low.base1285, %compare.local1406, !dbg !24
  %compare.result1408 = select i1 %compare.high.equal1283, i1 %compare.low1407, i1 %compare.high.order1284, !dbg !24
  %779 = select i1 %compare.result1408, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1409 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 62) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1410 = or i32 %compare.local.base1409, 79, !dbg !24
  %compare.low1411 = icmp ugt i32 %compare.low.base1285, %compare.local1410, !dbg !24
  %compare.result1412 = select i1 %compare.high.equal1283, i1 %compare.low1411, i1 %compare.high.order1284, !dbg !24
  %780 = select i1 %compare.result1412, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %781 = insertelement <32 x float> poison, float %714, i64 0, !dbg !13
  %782 = insertelement <32 x float> %781, float %715, i64 1, !dbg !13
  %783 = insertelement <32 x float> %782, float %716, i64 2, !dbg !13
  %784 = insertelement <32 x float> %783, float %717, i64 3, !dbg !13
  %785 = insertelement <32 x float> %784, float %718, i64 4, !dbg !13
  %786 = insertelement <32 x float> %785, float %719, i64 5, !dbg !13
  %787 = insertelement <32 x float> %786, float %720, i64 6, !dbg !13
  %788 = insertelement <32 x float> %787, float %721, i64 7, !dbg !13
  %789 = insertelement <32 x float> %788, float %722, i64 8, !dbg !13
  %790 = insertelement <32 x float> %789, float %723, i64 9, !dbg !13
  %791 = insertelement <32 x float> %790, float %724, i64 10, !dbg !13
  %792 = insertelement <32 x float> %791, float %725, i64 11, !dbg !13
  %793 = insertelement <32 x float> %792, float %726, i64 12, !dbg !13
  %794 = insertelement <32 x float> %793, float %727, i64 13, !dbg !13
  %795 = insertelement <32 x float> %794, float %728, i64 14, !dbg !13
  %796 = insertelement <32 x float> %795, float %729, i64 15, !dbg !13
  %797 = insertelement <32 x float> %796, float %730, i64 16, !dbg !13
  %798 = insertelement <32 x float> %797, float %731, i64 17, !dbg !13
  %799 = insertelement <32 x float> %798, float %732, i64 18, !dbg !13
  %800 = insertelement <32 x float> %799, float %733, i64 19, !dbg !13
  %801 = insertelement <32 x float> %800, float %734, i64 20, !dbg !13
  %802 = insertelement <32 x float> %801, float %735, i64 21, !dbg !13
  %803 = insertelement <32 x float> %802, float %736, i64 22, !dbg !13
  %804 = insertelement <32 x float> %803, float %737, i64 23, !dbg !13
  %805 = insertelement <32 x float> %804, float %738, i64 24, !dbg !13
  %806 = insertelement <32 x float> %805, float %739, i64 25, !dbg !13
  %807 = insertelement <32 x float> %806, float %740, i64 26, !dbg !13
  %808 = insertelement <32 x float> %807, float %741, i64 27, !dbg !13
  %809 = insertelement <32 x float> %808, float %742, i64 28, !dbg !13
  %810 = insertelement <32 x float> %809, float %743, i64 29, !dbg !13
  %811 = insertelement <32 x float> %810, float %744, i64 30, !dbg !13
  %812 = insertelement <32 x float> %811, float %745, i64 31, !dbg !13
  %813 = insertvalue [2 x <32 x float>] poison, <32 x float> %812, 0, !dbg !13
  %814 = insertelement <32 x float> poison, float %749, i64 0, !dbg !13
  %815 = insertelement <32 x float> %814, float %750, i64 1, !dbg !13
  %816 = insertelement <32 x float> %815, float %751, i64 2, !dbg !13
  %817 = insertelement <32 x float> %816, float %752, i64 3, !dbg !13
  %818 = insertelement <32 x float> %817, float %753, i64 4, !dbg !13
  %819 = insertelement <32 x float> %818, float %754, i64 5, !dbg !13
  %820 = insertelement <32 x float> %819, float %755, i64 6, !dbg !13
  %821 = insertelement <32 x float> %820, float %756, i64 7, !dbg !13
  %822 = insertelement <32 x float> %821, float %757, i64 8, !dbg !13
  %823 = insertelement <32 x float> %822, float %758, i64 9, !dbg !13
  %824 = insertelement <32 x float> %823, float %759, i64 10, !dbg !13
  %825 = insertelement <32 x float> %824, float %760, i64 11, !dbg !13
  %826 = insertelement <32 x float> %825, float %761, i64 12, !dbg !13
  %827 = insertelement <32 x float> %826, float %762, i64 13, !dbg !13
  %828 = insertelement <32 x float> %827, float %763, i64 14, !dbg !13
  %829 = insertelement <32 x float> %828, float %764, i64 15, !dbg !13
  %830 = insertelement <32 x float> %829, float %765, i64 16, !dbg !13
  %831 = insertelement <32 x float> %830, float %766, i64 17, !dbg !13
  %832 = insertelement <32 x float> %831, float %767, i64 18, !dbg !13
  %833 = insertelement <32 x float> %832, float %768, i64 19, !dbg !13
  %834 = insertelement <32 x float> %833, float %769, i64 20, !dbg !13
  %835 = insertelement <32 x float> %834, float %770, i64 21, !dbg !13
  %836 = insertelement <32 x float> %835, float %771, i64 22, !dbg !13
  %837 = insertelement <32 x float> %836, float %772, i64 23, !dbg !13
  %838 = insertelement <32 x float> %837, float %773, i64 24, !dbg !13
  %839 = insertelement <32 x float> %838, float %774, i64 25, !dbg !13
  %840 = insertelement <32 x float> %839, float %775, i64 26, !dbg !13
  %841 = insertelement <32 x float> %840, float %776, i64 27, !dbg !13
  %842 = insertelement <32 x float> %841, float %777, i64 28, !dbg !13
  %843 = insertelement <32 x float> %842, float %778, i64 29, !dbg !13
  %844 = insertelement <32 x float> %843, float %779, i64 30, !dbg !13
  %845 = insertelement <32 x float> %844, float %780, i64 31, !dbg !13
  %846 = insertvalue [2 x <32 x float>] %813, <32 x float> %845, 1, !dbg !13
  br label %847, !dbg !13

847:                                              ; preds = %702, %703
  %848 = phi [2 x <32 x float>] [ %846, %703 ], [ zeroinitializer, %702 ], !dbg !13
  br label %849, !dbg !13

849:                                              ; preds = %847
  %850 = extractvalue [2 x <32 x float>] %848, 0, !dbg !26
  %851 = shufflevector <32 x float> %850, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %852 = fadd <4 x float> %639, %851, !dbg !26
  %853 = extractelement <4 x float> %852, i64 0, !dbg !26
  %854 = extractelement <4 x float> %852, i64 1, !dbg !26
  %855 = extractelement <4 x float> %852, i64 2, !dbg !26
  %856 = extractelement <4 x float> %852, i64 3, !dbg !26
  %857 = shufflevector <32 x float> %850, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %858 = fadd <4 x float> %643, %857, !dbg !26
  %859 = extractelement <4 x float> %858, i64 0, !dbg !26
  %860 = extractelement <4 x float> %858, i64 1, !dbg !26
  %861 = extractelement <4 x float> %858, i64 2, !dbg !26
  %862 = extractelement <4 x float> %858, i64 3, !dbg !26
  %863 = shufflevector <32 x float> %850, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !26
  %864 = fadd <4 x float> %647, %863, !dbg !26
  %865 = extractelement <4 x float> %864, i64 0, !dbg !26
  %866 = extractelement <4 x float> %864, i64 1, !dbg !26
  %867 = extractelement <4 x float> %864, i64 2, !dbg !26
  %868 = extractelement <4 x float> %864, i64 3, !dbg !26
  %869 = shufflevector <32 x float> %850, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !26
  %870 = fadd <4 x float> %651, %869, !dbg !26
  %871 = extractelement <4 x float> %870, i64 0, !dbg !26
  %872 = extractelement <4 x float> %870, i64 1, !dbg !26
  %873 = extractelement <4 x float> %870, i64 2, !dbg !26
  %874 = extractelement <4 x float> %870, i64 3, !dbg !26
  %875 = shufflevector <32 x float> %850, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !26
  %876 = fadd <4 x float> %655, %875, !dbg !26
  %877 = extractelement <4 x float> %876, i64 0, !dbg !26
  %878 = extractelement <4 x float> %876, i64 1, !dbg !26
  %879 = extractelement <4 x float> %876, i64 2, !dbg !26
  %880 = extractelement <4 x float> %876, i64 3, !dbg !26
  %881 = shufflevector <32 x float> %850, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !26
  %882 = fadd <4 x float> %659, %881, !dbg !26
  %883 = extractelement <4 x float> %882, i64 0, !dbg !26
  %884 = extractelement <4 x float> %882, i64 1, !dbg !26
  %885 = extractelement <4 x float> %882, i64 2, !dbg !26
  %886 = extractelement <4 x float> %882, i64 3, !dbg !26
  %887 = shufflevector <32 x float> %850, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !26
  %888 = fadd <4 x float> %663, %887, !dbg !26
  %889 = extractelement <4 x float> %888, i64 0, !dbg !26
  %890 = extractelement <4 x float> %888, i64 1, !dbg !26
  %891 = extractelement <4 x float> %888, i64 2, !dbg !26
  %892 = extractelement <4 x float> %888, i64 3, !dbg !26
  %893 = shufflevector <32 x float> %850, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !26
  %894 = fadd <4 x float> %667, %893, !dbg !26
  %895 = extractelement <4 x float> %894, i64 0, !dbg !26
  %896 = extractelement <4 x float> %894, i64 1, !dbg !26
  %897 = extractelement <4 x float> %894, i64 2, !dbg !26
  %898 = extractelement <4 x float> %894, i64 3, !dbg !26
  %899 = extractvalue [2 x <32 x float>] %848, 1, !dbg !26
  %900 = shufflevector <32 x float> %899, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %901 = fadd <4 x float> %671, %900, !dbg !26
  %902 = extractelement <4 x float> %901, i64 0, !dbg !26
  %903 = extractelement <4 x float> %901, i64 1, !dbg !26
  %904 = extractelement <4 x float> %901, i64 2, !dbg !26
  %905 = extractelement <4 x float> %901, i64 3, !dbg !26
  %906 = shufflevector <32 x float> %899, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %907 = fadd <4 x float> %675, %906, !dbg !26
  %908 = extractelement <4 x float> %907, i64 0, !dbg !26
  %909 = extractelement <4 x float> %907, i64 1, !dbg !26
  %910 = extractelement <4 x float> %907, i64 2, !dbg !26
  %911 = extractelement <4 x float> %907, i64 3, !dbg !26
  %912 = shufflevector <32 x float> %899, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !26
  %913 = fadd <4 x float> %679, %912, !dbg !26
  %914 = extractelement <4 x float> %913, i64 0, !dbg !26
  %915 = extractelement <4 x float> %913, i64 1, !dbg !26
  %916 = extractelement <4 x float> %913, i64 2, !dbg !26
  %917 = extractelement <4 x float> %913, i64 3, !dbg !26
  %918 = shufflevector <32 x float> %899, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !26
  %919 = fadd <4 x float> %683, %918, !dbg !26
  %920 = extractelement <4 x float> %919, i64 0, !dbg !26
  %921 = extractelement <4 x float> %919, i64 1, !dbg !26
  %922 = extractelement <4 x float> %919, i64 2, !dbg !26
  %923 = extractelement <4 x float> %919, i64 3, !dbg !26
  %924 = shufflevector <32 x float> %899, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !26
  %925 = fadd <4 x float> %687, %924, !dbg !26
  %926 = extractelement <4 x float> %925, i64 0, !dbg !26
  %927 = extractelement <4 x float> %925, i64 1, !dbg !26
  %928 = extractelement <4 x float> %925, i64 2, !dbg !26
  %929 = extractelement <4 x float> %925, i64 3, !dbg !26
  %930 = shufflevector <32 x float> %899, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !26
  %931 = fadd <4 x float> %691, %930, !dbg !26
  %932 = extractelement <4 x float> %931, i64 0, !dbg !26
  %933 = extractelement <4 x float> %931, i64 1, !dbg !26
  %934 = extractelement <4 x float> %931, i64 2, !dbg !26
  %935 = extractelement <4 x float> %931, i64 3, !dbg !26
  %936 = shufflevector <32 x float> %899, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !26
  %937 = fadd <4 x float> %695, %936, !dbg !26
  %938 = extractelement <4 x float> %937, i64 0, !dbg !26
  %939 = extractelement <4 x float> %937, i64 1, !dbg !26
  %940 = extractelement <4 x float> %937, i64 2, !dbg !26
  %941 = extractelement <4 x float> %937, i64 3, !dbg !26
  %942 = shufflevector <32 x float> %899, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !26
  %943 = fadd <4 x float> %700, %942, !dbg !26
  %944 = extractelement <4 x float> %943, i64 0, !dbg !26
  %945 = extractelement <4 x float> %943, i64 1, !dbg !26
  %946 = extractelement <4 x float> %943, i64 2, !dbg !26
  %947 = extractelement <4 x float> %943, i64 3, !dbg !26
  %948 = call float @llvm.maxnum.f32(float %853, float 0xFFF0000000000000), !dbg !27
  %949 = call float @llvm.maxnum.f32(float %948, float %854), !dbg !27
  %950 = call float @llvm.maxnum.f32(float %949, float %855), !dbg !27
  %951 = call float @llvm.maxnum.f32(float %950, float %856), !dbg !27
  %952 = call float @llvm.maxnum.f32(float %951, float %859), !dbg !27
  %953 = call float @llvm.maxnum.f32(float %952, float %860), !dbg !27
  %954 = call float @llvm.maxnum.f32(float %953, float %861), !dbg !27
  %955 = call float @llvm.maxnum.f32(float %954, float %862), !dbg !27
  %956 = call float @llvm.maxnum.f32(float %955, float %865), !dbg !27
  %957 = call float @llvm.maxnum.f32(float %956, float %866), !dbg !27
  %958 = call float @llvm.maxnum.f32(float %957, float %867), !dbg !27
  %959 = call float @llvm.maxnum.f32(float %958, float %868), !dbg !27
  %960 = call float @llvm.maxnum.f32(float %959, float %871), !dbg !27
  %961 = call float @llvm.maxnum.f32(float %960, float %872), !dbg !27
  %962 = call float @llvm.maxnum.f32(float %961, float %873), !dbg !27
  %963 = call float @llvm.maxnum.f32(float %962, float %874), !dbg !27
  %964 = call float @llvm.maxnum.f32(float %963, float %877), !dbg !27
  %965 = call float @llvm.maxnum.f32(float %964, float %878), !dbg !27
  %966 = call float @llvm.maxnum.f32(float %965, float %879), !dbg !27
  %967 = call float @llvm.maxnum.f32(float %966, float %880), !dbg !27
  %968 = call float @llvm.maxnum.f32(float %967, float %883), !dbg !27
  %969 = call float @llvm.maxnum.f32(float %968, float %884), !dbg !27
  %970 = call float @llvm.maxnum.f32(float %969, float %885), !dbg !27
  %971 = call float @llvm.maxnum.f32(float %970, float %886), !dbg !27
  %972 = call float @llvm.maxnum.f32(float %971, float %889), !dbg !27
  %973 = call float @llvm.maxnum.f32(float %972, float %890), !dbg !27
  %974 = call float @llvm.maxnum.f32(float %973, float %891), !dbg !27
  %975 = call float @llvm.maxnum.f32(float %974, float %892), !dbg !27
  %976 = call float @llvm.maxnum.f32(float %975, float %895), !dbg !27
  %977 = call float @llvm.maxnum.f32(float %976, float %896), !dbg !27
  %978 = call float @llvm.maxnum.f32(float %977, float %897), !dbg !27
  %979 = call float @llvm.maxnum.f32(float %978, float %898), !dbg !27
  %980 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !27
  %981 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %980), !dbg !27
  %982 = add nuw nsw i32 %981, 64, !dbg !27
  %983 = xor i32 %981, 16, !dbg !27
  %984 = and i32 %982, 192, !dbg !27
  %985 = icmp samesign ult i32 %983, %984, !dbg !27
  %986 = select i1 %985, i32 %983, i32 %981, !dbg !27
  %987 = shl nuw nsw i32 %986, 2, !dbg !27
  %988 = bitcast float %979 to i32, !dbg !27
  %989 = call i32 @llvm.amdgcn.ds.bpermute(i32 %987, i32 %988), !dbg !27
  %990 = bitcast i32 %989 to float, !dbg !27
  %991 = call float @llvm.maxnum.f32(float %979, float %990), !dbg !27
  %992 = xor i32 %981, 32, !dbg !27
  %993 = icmp samesign ult i32 %992, %984, !dbg !27
  %994 = select i1 %993, i32 %992, i32 %981, !dbg !27
  %995 = shl nuw nsw i32 %994, 2, !dbg !27
  %996 = bitcast float %991 to i32, !dbg !27
  %997 = call i32 @llvm.amdgcn.ds.bpermute(i32 %995, i32 %996), !dbg !27
  %998 = bitcast i32 %997 to float, !dbg !27
  %999 = call float @llvm.maxnum.f32(float %991, float %998), !dbg !27
  %1000 = insertelement <1 x float> poison, float %999, i64 0, !dbg !27
  %1001 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %1000, <1 x float> splat (float 0xFFF0000000000000)), !dbg !27
  %1002 = call float @llvm.maxnum.f32(float %902, float 0xFFF0000000000000), !dbg !27
  %1003 = call float @llvm.maxnum.f32(float %1002, float %903), !dbg !27
  %1004 = call float @llvm.maxnum.f32(float %1003, float %904), !dbg !27
  %1005 = call float @llvm.maxnum.f32(float %1004, float %905), !dbg !27
  %1006 = call float @llvm.maxnum.f32(float %1005, float %908), !dbg !27
  %1007 = call float @llvm.maxnum.f32(float %1006, float %909), !dbg !27
  %1008 = call float @llvm.maxnum.f32(float %1007, float %910), !dbg !27
  %1009 = call float @llvm.maxnum.f32(float %1008, float %911), !dbg !27
  %1010 = call float @llvm.maxnum.f32(float %1009, float %914), !dbg !27
  %1011 = call float @llvm.maxnum.f32(float %1010, float %915), !dbg !27
  %1012 = call float @llvm.maxnum.f32(float %1011, float %916), !dbg !27
  %1013 = call float @llvm.maxnum.f32(float %1012, float %917), !dbg !27
  %1014 = call float @llvm.maxnum.f32(float %1013, float %920), !dbg !27
  %1015 = call float @llvm.maxnum.f32(float %1014, float %921), !dbg !27
  %1016 = call float @llvm.maxnum.f32(float %1015, float %922), !dbg !27
  %1017 = call float @llvm.maxnum.f32(float %1016, float %923), !dbg !27
  %1018 = call float @llvm.maxnum.f32(float %1017, float %926), !dbg !27
  %1019 = call float @llvm.maxnum.f32(float %1018, float %927), !dbg !27
  %1020 = call float @llvm.maxnum.f32(float %1019, float %928), !dbg !27
  %1021 = call float @llvm.maxnum.f32(float %1020, float %929), !dbg !27
  %1022 = call float @llvm.maxnum.f32(float %1021, float %932), !dbg !27
  %1023 = call float @llvm.maxnum.f32(float %1022, float %933), !dbg !27
  %1024 = call float @llvm.maxnum.f32(float %1023, float %934), !dbg !27
  %1025 = call float @llvm.maxnum.f32(float %1024, float %935), !dbg !27
  %1026 = call float @llvm.maxnum.f32(float %1025, float %938), !dbg !27
  %1027 = call float @llvm.maxnum.f32(float %1026, float %939), !dbg !27
  %1028 = call float @llvm.maxnum.f32(float %1027, float %940), !dbg !27
  %1029 = call float @llvm.maxnum.f32(float %1028, float %941), !dbg !27
  %1030 = call float @llvm.maxnum.f32(float %1029, float %944), !dbg !27
  %1031 = call float @llvm.maxnum.f32(float %1030, float %945), !dbg !27
  %1032 = call float @llvm.maxnum.f32(float %1031, float %946), !dbg !27
  %1033 = call float @llvm.maxnum.f32(float %1032, float %947), !dbg !27
  %1034 = bitcast float %1033 to i32, !dbg !27
  %1035 = call i32 @llvm.amdgcn.ds.bpermute(i32 %987, i32 %1034), !dbg !27
  %1036 = bitcast i32 %1035 to float, !dbg !27
  %1037 = call float @llvm.maxnum.f32(float %1033, float %1036), !dbg !27
  %1038 = bitcast float %1037 to i32, !dbg !27
  %1039 = call i32 @llvm.amdgcn.ds.bpermute(i32 %995, i32 %1038), !dbg !27
  %1040 = bitcast i32 %1039 to float, !dbg !27
  %1041 = call float @llvm.maxnum.f32(float %1037, float %1040), !dbg !27
  %1042 = insertelement <1 x float> poison, float %1041, i64 0, !dbg !27
  %1043 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %1042, <1 x float> splat (float 0xFFF0000000000000)), !dbg !27
  %1044 = shufflevector <1 x float> %1001, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !28
  %1045 = fsub <4 x float> %852, %1044, !dbg !28
  %1046 = fsub <4 x float> %858, %1044, !dbg !28
  %1047 = fsub <4 x float> %864, %1044, !dbg !28
  %1048 = fsub <4 x float> %870, %1044, !dbg !28
  %1049 = fsub <4 x float> %876, %1044, !dbg !28
  %1050 = fsub <4 x float> %882, %1044, !dbg !28
  %1051 = fsub <4 x float> %888, %1044, !dbg !28
  %1052 = fsub <4 x float> %894, %1044, !dbg !28
  %1053 = shufflevector <1 x float> %1043, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !28
  %1054 = fsub <4 x float> %901, %1053, !dbg !28
  %1055 = fsub <4 x float> %907, %1053, !dbg !28
  %1056 = fsub <4 x float> %913, %1053, !dbg !28
  %1057 = fsub <4 x float> %919, %1053, !dbg !28
  %1058 = fsub <4 x float> %925, %1053, !dbg !28
  %1059 = fsub <4 x float> %931, %1053, !dbg !28
  %1060 = fsub <4 x float> %937, %1053, !dbg !28
  %1061 = fsub <4 x float> %943, %1053, !dbg !28
  %1062 = extractelement <1 x float> %1001, i64 0, !dbg !29
  %1063 = fsub float 0xFFF0000000000000, %1062, !dbg !29
  %1064 = call float @llvm.amdgcn.exp2.f32(float %1063), !dbg !30
  %1065 = insertelement <4 x float> poison, float %1064, i64 0, !dbg !35
  %1066 = shufflevector <4 x float> %1065, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !35
  %1067 = extractvalue [2 x <1 x float>] %209, 0, !dbg !31
  %1068 = extractelement <1 x float> %1067, i64 0, !dbg !31
  %1069 = fmul float %1068, %1064, !dbg !31
  %1070 = extractelement <1 x float> %1043, i64 0, !dbg !29
  %1071 = fsub float 0xFFF0000000000000, %1070, !dbg !29
  %1072 = call float @llvm.amdgcn.exp2.f32(float %1071), !dbg !30
  %1073 = insertelement <4 x float> poison, float %1072, i64 0, !dbg !35
  %1074 = shufflevector <4 x float> %1073, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !35
  %1075 = extractvalue [2 x <1 x float>] %209, 1, !dbg !31
  %1076 = extractelement <1 x float> %1075, i64 0, !dbg !31
  %1077 = fmul float %1076, %1072, !dbg !31
  %1078 = extractelement <4 x float> %1045, i64 0, !dbg !32
  %1079 = call float @llvm.amdgcn.exp2.f32(float %1078), !dbg !32
  %1080 = insertelement <4 x float> poison, float %1079, i64 0, !dbg !34
  %1081 = extractelement <4 x float> %1045, i64 1, !dbg !32
  %1082 = call float @llvm.amdgcn.exp2.f32(float %1081), !dbg !32
  %1083 = insertelement <4 x float> %1080, float %1082, i64 1, !dbg !34
  %1084 = fadd float %1079, %1082, !dbg !33
  %1085 = extractelement <4 x float> %1045, i64 2, !dbg !32
  %1086 = call float @llvm.amdgcn.exp2.f32(float %1085), !dbg !32
  %1087 = insertelement <4 x float> %1083, float %1086, i64 2, !dbg !34
  %1088 = fadd float %1084, %1086, !dbg !33
  %1089 = extractelement <4 x float> %1045, i64 3, !dbg !32
  %1090 = call float @llvm.amdgcn.exp2.f32(float %1089), !dbg !32
  %1091 = insertelement <4 x float> %1087, float %1090, i64 3, !dbg !34
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1091) #9, !dbg !34, !frisk.register_lifetimes !1
  %1092 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !34
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1092) #9, !dbg !33, !frisk.register_lifetimes !1
  %1093 = fadd float %1088, %1090, !dbg !33
  %1094 = extractelement <4 x float> %1046, i64 0, !dbg !32
  %1095 = call float @llvm.amdgcn.exp2.f32(float %1094), !dbg !32
  %1096 = insertelement <4 x float> poison, float %1095, i64 0, !dbg !34
  %1097 = fadd float %1093, %1095, !dbg !33
  %1098 = extractelement <4 x float> %1046, i64 1, !dbg !32
  %1099 = call float @llvm.amdgcn.exp2.f32(float %1098), !dbg !32
  %1100 = insertelement <4 x float> %1096, float %1099, i64 1, !dbg !34
  %1101 = fadd float %1097, %1099, !dbg !33
  %1102 = extractelement <4 x float> %1046, i64 2, !dbg !32
  %1103 = call float @llvm.amdgcn.exp2.f32(float %1102), !dbg !32
  %1104 = insertelement <4 x float> %1100, float %1103, i64 2, !dbg !34
  %1105 = fadd float %1101, %1103, !dbg !33
  %1106 = extractelement <4 x float> %1046, i64 3, !dbg !32
  %1107 = call float @llvm.amdgcn.exp2.f32(float %1106), !dbg !32
  %1108 = insertelement <4 x float> %1104, float %1107, i64 3, !dbg !34
  %math.result309 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1108) #9, !dbg !34, !frisk.register_lifetimes !1
  %1109 = fptrunc <4 x float> %math.result309 to <4 x half>, !dbg !34
  %math.packed310 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1109) #9, !dbg !33, !frisk.register_lifetimes !1
  %1110 = fadd float %1105, %1107, !dbg !33
  %1111 = extractelement <4 x float> %1047, i64 0, !dbg !32
  %1112 = call float @llvm.amdgcn.exp2.f32(float %1111), !dbg !32
  %1113 = insertelement <4 x float> poison, float %1112, i64 0, !dbg !34
  %1114 = fadd float %1110, %1112, !dbg !33
  %1115 = extractelement <4 x float> %1047, i64 1, !dbg !32
  %1116 = call float @llvm.amdgcn.exp2.f32(float %1115), !dbg !32
  %1117 = insertelement <4 x float> %1113, float %1116, i64 1, !dbg !34
  %1118 = fadd float %1114, %1116, !dbg !33
  %1119 = extractelement <4 x float> %1047, i64 2, !dbg !32
  %1120 = call float @llvm.amdgcn.exp2.f32(float %1119), !dbg !32
  %1121 = insertelement <4 x float> %1117, float %1120, i64 2, !dbg !34
  %1122 = fadd float %1118, %1120, !dbg !33
  %1123 = extractelement <4 x float> %1047, i64 3, !dbg !32
  %1124 = call float @llvm.amdgcn.exp2.f32(float %1123), !dbg !32
  %1125 = insertelement <4 x float> %1121, float %1124, i64 3, !dbg !34
  %math.result311 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1125) #9, !dbg !34, !frisk.register_lifetimes !1
  %1126 = fptrunc <4 x float> %math.result311 to <4 x half>, !dbg !34
  %math.packed312 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1126) #9, !dbg !33, !frisk.register_lifetimes !1
  %1127 = fadd float %1122, %1124, !dbg !33
  %1128 = extractelement <4 x float> %1048, i64 0, !dbg !32
  %1129 = call float @llvm.amdgcn.exp2.f32(float %1128), !dbg !32
  %1130 = insertelement <4 x float> poison, float %1129, i64 0, !dbg !34
  %1131 = fadd float %1127, %1129, !dbg !33
  %1132 = extractelement <4 x float> %1048, i64 1, !dbg !32
  %1133 = call float @llvm.amdgcn.exp2.f32(float %1132), !dbg !32
  %1134 = insertelement <4 x float> %1130, float %1133, i64 1, !dbg !34
  %1135 = fadd float %1131, %1133, !dbg !33
  %1136 = extractelement <4 x float> %1048, i64 2, !dbg !32
  %1137 = call float @llvm.amdgcn.exp2.f32(float %1136), !dbg !32
  %1138 = insertelement <4 x float> %1134, float %1137, i64 2, !dbg !34
  %1139 = fadd float %1135, %1137, !dbg !33
  %1140 = extractelement <4 x float> %1048, i64 3, !dbg !32
  %1141 = call float @llvm.amdgcn.exp2.f32(float %1140), !dbg !32
  %1142 = insertelement <4 x float> %1138, float %1141, i64 3, !dbg !34
  %math.result313 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1142) #9, !dbg !34, !frisk.register_lifetimes !1
  %1143 = fptrunc <4 x float> %math.result313 to <4 x half>, !dbg !34
  %math.packed314 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1143) #9, !dbg !33, !frisk.register_lifetimes !1
  %1144 = fadd float %1139, %1141, !dbg !33
  %1145 = extractelement <4 x float> %1049, i64 0, !dbg !32
  %1146 = call float @llvm.amdgcn.exp2.f32(float %1145), !dbg !32
  %1147 = insertelement <4 x float> poison, float %1146, i64 0, !dbg !34
  %1148 = fadd float %1144, %1146, !dbg !33
  %1149 = extractelement <4 x float> %1049, i64 1, !dbg !32
  %1150 = call float @llvm.amdgcn.exp2.f32(float %1149), !dbg !32
  %1151 = insertelement <4 x float> %1147, float %1150, i64 1, !dbg !34
  %1152 = fadd float %1148, %1150, !dbg !33
  %1153 = extractelement <4 x float> %1049, i64 2, !dbg !32
  %1154 = call float @llvm.amdgcn.exp2.f32(float %1153), !dbg !32
  %1155 = insertelement <4 x float> %1151, float %1154, i64 2, !dbg !34
  %1156 = fadd float %1152, %1154, !dbg !33
  %1157 = extractelement <4 x float> %1049, i64 3, !dbg !32
  %1158 = call float @llvm.amdgcn.exp2.f32(float %1157), !dbg !32
  %1159 = insertelement <4 x float> %1155, float %1158, i64 3, !dbg !34
  %math.result315 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1159) #9, !dbg !34, !frisk.register_lifetimes !1
  %1160 = fptrunc <4 x float> %math.result315 to <4 x half>, !dbg !34
  %math.packed316 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1160) #9, !dbg !33, !frisk.register_lifetimes !1
  %1161 = fadd float %1156, %1158, !dbg !33
  %1162 = extractelement <4 x float> %1050, i64 0, !dbg !32
  %1163 = call float @llvm.amdgcn.exp2.f32(float %1162), !dbg !32
  %1164 = insertelement <4 x float> poison, float %1163, i64 0, !dbg !34
  %1165 = fadd float %1161, %1163, !dbg !33
  %1166 = extractelement <4 x float> %1050, i64 1, !dbg !32
  %1167 = call float @llvm.amdgcn.exp2.f32(float %1166), !dbg !32
  %1168 = insertelement <4 x float> %1164, float %1167, i64 1, !dbg !34
  %1169 = fadd float %1165, %1167, !dbg !33
  %1170 = extractelement <4 x float> %1050, i64 2, !dbg !32
  %1171 = call float @llvm.amdgcn.exp2.f32(float %1170), !dbg !32
  %1172 = insertelement <4 x float> %1168, float %1171, i64 2, !dbg !34
  %1173 = fadd float %1169, %1171, !dbg !33
  %1174 = extractelement <4 x float> %1050, i64 3, !dbg !32
  %1175 = call float @llvm.amdgcn.exp2.f32(float %1174), !dbg !32
  %1176 = insertelement <4 x float> %1172, float %1175, i64 3, !dbg !34
  %math.result317 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1176) #9, !dbg !34, !frisk.register_lifetimes !1
  %1177 = fptrunc <4 x float> %math.result317 to <4 x half>, !dbg !34
  %math.packed318 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1177) #9, !dbg !33, !frisk.register_lifetimes !1
  %1178 = fadd float %1173, %1175, !dbg !33
  %1179 = extractelement <4 x float> %1051, i64 0, !dbg !32
  %1180 = call float @llvm.amdgcn.exp2.f32(float %1179), !dbg !32
  %1181 = insertelement <4 x float> poison, float %1180, i64 0, !dbg !34
  %1182 = fadd float %1178, %1180, !dbg !33
  %1183 = extractelement <4 x float> %1051, i64 1, !dbg !32
  %1184 = call float @llvm.amdgcn.exp2.f32(float %1183), !dbg !32
  %1185 = insertelement <4 x float> %1181, float %1184, i64 1, !dbg !34
  %1186 = fadd float %1182, %1184, !dbg !33
  %1187 = extractelement <4 x float> %1051, i64 2, !dbg !32
  %1188 = call float @llvm.amdgcn.exp2.f32(float %1187), !dbg !32
  %1189 = insertelement <4 x float> %1185, float %1188, i64 2, !dbg !34
  %1190 = fadd float %1186, %1188, !dbg !33
  %1191 = extractelement <4 x float> %1051, i64 3, !dbg !32
  %1192 = call float @llvm.amdgcn.exp2.f32(float %1191), !dbg !32
  %1193 = insertelement <4 x float> %1189, float %1192, i64 3, !dbg !34
  %math.result319 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1193) #9, !dbg !34, !frisk.register_lifetimes !1
  %1194 = fptrunc <4 x float> %math.result319 to <4 x half>, !dbg !34
  %math.packed320 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1194) #9, !dbg !33, !frisk.register_lifetimes !1
  %1195 = fadd float %1190, %1192, !dbg !33
  %1196 = extractelement <4 x float> %1052, i64 0, !dbg !32
  %1197 = call float @llvm.amdgcn.exp2.f32(float %1196), !dbg !32
  %1198 = insertelement <4 x float> poison, float %1197, i64 0, !dbg !34
  %1199 = fadd float %1195, %1197, !dbg !33
  %1200 = extractelement <4 x float> %1052, i64 1, !dbg !32
  %1201 = call float @llvm.amdgcn.exp2.f32(float %1200), !dbg !32
  %1202 = insertelement <4 x float> %1198, float %1201, i64 1, !dbg !34
  %1203 = fadd float %1199, %1201, !dbg !33
  %1204 = extractelement <4 x float> %1052, i64 2, !dbg !32
  %1205 = call float @llvm.amdgcn.exp2.f32(float %1204), !dbg !32
  %1206 = insertelement <4 x float> %1202, float %1205, i64 2, !dbg !34
  %1207 = fadd float %1203, %1205, !dbg !33
  %1208 = extractelement <4 x float> %1052, i64 3, !dbg !32
  %1209 = call float @llvm.amdgcn.exp2.f32(float %1208), !dbg !32
  %1210 = insertelement <4 x float> %1206, float %1209, i64 3, !dbg !34
  %math.result321 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1210) #9, !dbg !34, !frisk.register_lifetimes !1
  %1211 = fptrunc <4 x float> %math.result321 to <4 x half>, !dbg !34
  %math.packed322 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1211) #9, !dbg !33, !frisk.register_lifetimes !1
  %1212 = fadd float %1207, %1209, !dbg !33
  %1213 = extractelement <4 x float> %1054, i64 0, !dbg !32
  %1214 = call float @llvm.amdgcn.exp2.f32(float %1213), !dbg !32
  %1215 = insertelement <4 x float> poison, float %1214, i64 0, !dbg !34
  %1216 = extractelement <4 x float> %1054, i64 1, !dbg !32
  %1217 = call float @llvm.amdgcn.exp2.f32(float %1216), !dbg !32
  %1218 = insertelement <4 x float> %1215, float %1217, i64 1, !dbg !34
  %1219 = fadd float %1214, %1217, !dbg !33
  %1220 = extractelement <4 x float> %1054, i64 2, !dbg !32
  %1221 = call float @llvm.amdgcn.exp2.f32(float %1220), !dbg !32
  %1222 = insertelement <4 x float> %1218, float %1221, i64 2, !dbg !34
  %1223 = fadd float %1219, %1221, !dbg !33
  %1224 = extractelement <4 x float> %1054, i64 3, !dbg !32
  %1225 = call float @llvm.amdgcn.exp2.f32(float %1224), !dbg !32
  %1226 = insertelement <4 x float> %1222, float %1225, i64 3, !dbg !34
  %math.result323 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1226) #9, !dbg !34, !frisk.register_lifetimes !1
  %1227 = fptrunc <4 x float> %math.result323 to <4 x half>, !dbg !34
  %math.packed324 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1227) #9, !dbg !33, !frisk.register_lifetimes !1
  %1228 = fadd float %1223, %1225, !dbg !33
  %1229 = extractelement <4 x float> %1055, i64 0, !dbg !32
  %1230 = call float @llvm.amdgcn.exp2.f32(float %1229), !dbg !32
  %1231 = insertelement <4 x float> poison, float %1230, i64 0, !dbg !34
  %1232 = fadd float %1228, %1230, !dbg !33
  %1233 = extractelement <4 x float> %1055, i64 1, !dbg !32
  %1234 = call float @llvm.amdgcn.exp2.f32(float %1233), !dbg !32
  %1235 = insertelement <4 x float> %1231, float %1234, i64 1, !dbg !34
  %1236 = fadd float %1232, %1234, !dbg !33
  %1237 = extractelement <4 x float> %1055, i64 2, !dbg !32
  %1238 = call float @llvm.amdgcn.exp2.f32(float %1237), !dbg !32
  %1239 = insertelement <4 x float> %1235, float %1238, i64 2, !dbg !34
  %1240 = fadd float %1236, %1238, !dbg !33
  %1241 = extractelement <4 x float> %1055, i64 3, !dbg !32
  %1242 = call float @llvm.amdgcn.exp2.f32(float %1241), !dbg !32
  %1243 = insertelement <4 x float> %1239, float %1242, i64 3, !dbg !34
  %math.result325 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1243) #9, !dbg !34, !frisk.register_lifetimes !1
  %1244 = fptrunc <4 x float> %math.result325 to <4 x half>, !dbg !34
  %math.packed326 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1244) #9, !dbg !33, !frisk.register_lifetimes !1
  %1245 = fadd float %1240, %1242, !dbg !33
  %1246 = extractelement <4 x float> %1056, i64 0, !dbg !32
  %1247 = call float @llvm.amdgcn.exp2.f32(float %1246), !dbg !32
  %1248 = insertelement <4 x float> poison, float %1247, i64 0, !dbg !34
  %1249 = fadd float %1245, %1247, !dbg !33
  %1250 = extractelement <4 x float> %1056, i64 1, !dbg !32
  %1251 = call float @llvm.amdgcn.exp2.f32(float %1250), !dbg !32
  %1252 = insertelement <4 x float> %1248, float %1251, i64 1, !dbg !34
  %1253 = fadd float %1249, %1251, !dbg !33
  %1254 = extractelement <4 x float> %1056, i64 2, !dbg !32
  %1255 = call float @llvm.amdgcn.exp2.f32(float %1254), !dbg !32
  %1256 = insertelement <4 x float> %1252, float %1255, i64 2, !dbg !34
  %1257 = fadd float %1253, %1255, !dbg !33
  %1258 = extractelement <4 x float> %1056, i64 3, !dbg !32
  %1259 = call float @llvm.amdgcn.exp2.f32(float %1258), !dbg !32
  %1260 = insertelement <4 x float> %1256, float %1259, i64 3, !dbg !34
  %math.result327 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1260) #9, !dbg !34, !frisk.register_lifetimes !1
  %1261 = fptrunc <4 x float> %math.result327 to <4 x half>, !dbg !34
  %math.packed328 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1261) #9, !dbg !33, !frisk.register_lifetimes !1
  %1262 = fadd float %1257, %1259, !dbg !33
  %1263 = extractelement <4 x float> %1057, i64 0, !dbg !32
  %1264 = call float @llvm.amdgcn.exp2.f32(float %1263), !dbg !32
  %1265 = insertelement <4 x float> poison, float %1264, i64 0, !dbg !34
  %1266 = fadd float %1262, %1264, !dbg !33
  %1267 = extractelement <4 x float> %1057, i64 1, !dbg !32
  %1268 = call float @llvm.amdgcn.exp2.f32(float %1267), !dbg !32
  %1269 = insertelement <4 x float> %1265, float %1268, i64 1, !dbg !34
  %1270 = fadd float %1266, %1268, !dbg !33
  %1271 = extractelement <4 x float> %1057, i64 2, !dbg !32
  %1272 = call float @llvm.amdgcn.exp2.f32(float %1271), !dbg !32
  %1273 = insertelement <4 x float> %1269, float %1272, i64 2, !dbg !34
  %1274 = fadd float %1270, %1272, !dbg !33
  %1275 = extractelement <4 x float> %1057, i64 3, !dbg !32
  %1276 = call float @llvm.amdgcn.exp2.f32(float %1275), !dbg !32
  %1277 = insertelement <4 x float> %1273, float %1276, i64 3, !dbg !34
  %math.result329 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1277) #9, !dbg !34, !frisk.register_lifetimes !1
  %1278 = fptrunc <4 x float> %math.result329 to <4 x half>, !dbg !34
  %math.packed330 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1278) #9, !dbg !33, !frisk.register_lifetimes !1
  %1279 = fadd float %1274, %1276, !dbg !33
  %1280 = extractelement <4 x float> %1058, i64 0, !dbg !32
  %1281 = call float @llvm.amdgcn.exp2.f32(float %1280), !dbg !32
  %1282 = insertelement <4 x float> poison, float %1281, i64 0, !dbg !34
  %1283 = fadd float %1279, %1281, !dbg !33
  %1284 = extractelement <4 x float> %1058, i64 1, !dbg !32
  %1285 = call float @llvm.amdgcn.exp2.f32(float %1284), !dbg !32
  %1286 = insertelement <4 x float> %1282, float %1285, i64 1, !dbg !34
  %1287 = fadd float %1283, %1285, !dbg !33
  %1288 = extractelement <4 x float> %1058, i64 2, !dbg !32
  %1289 = call float @llvm.amdgcn.exp2.f32(float %1288), !dbg !32
  %1290 = insertelement <4 x float> %1286, float %1289, i64 2, !dbg !34
  %1291 = fadd float %1287, %1289, !dbg !33
  %1292 = extractelement <4 x float> %1058, i64 3, !dbg !32
  %1293 = call float @llvm.amdgcn.exp2.f32(float %1292), !dbg !32
  %1294 = insertelement <4 x float> %1290, float %1293, i64 3, !dbg !34
  %math.result331 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1294) #9, !dbg !34, !frisk.register_lifetimes !1
  %1295 = fptrunc <4 x float> %math.result331 to <4 x half>, !dbg !34
  %math.packed332 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1295) #9, !dbg !33, !frisk.register_lifetimes !1
  %1296 = fadd float %1291, %1293, !dbg !33
  %1297 = extractelement <4 x float> %1059, i64 0, !dbg !32
  %1298 = call float @llvm.amdgcn.exp2.f32(float %1297), !dbg !32
  %1299 = insertelement <4 x float> poison, float %1298, i64 0, !dbg !34
  %1300 = fadd float %1296, %1298, !dbg !33
  %1301 = extractelement <4 x float> %1059, i64 1, !dbg !32
  %1302 = call float @llvm.amdgcn.exp2.f32(float %1301), !dbg !32
  %1303 = insertelement <4 x float> %1299, float %1302, i64 1, !dbg !34
  %1304 = fadd float %1300, %1302, !dbg !33
  %1305 = extractelement <4 x float> %1059, i64 2, !dbg !32
  %1306 = call float @llvm.amdgcn.exp2.f32(float %1305), !dbg !32
  %1307 = insertelement <4 x float> %1303, float %1306, i64 2, !dbg !34
  %1308 = fadd float %1304, %1306, !dbg !33
  %1309 = extractelement <4 x float> %1059, i64 3, !dbg !32
  %1310 = call float @llvm.amdgcn.exp2.f32(float %1309), !dbg !32
  %1311 = insertelement <4 x float> %1307, float %1310, i64 3, !dbg !34
  %math.result333 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1311) #9, !dbg !34, !frisk.register_lifetimes !1
  %1312 = fptrunc <4 x float> %math.result333 to <4 x half>, !dbg !34
  %math.packed334 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1312) #9, !dbg !33, !frisk.register_lifetimes !1
  %1313 = fadd float %1308, %1310, !dbg !33
  %1314 = extractelement <4 x float> %1060, i64 0, !dbg !32
  %1315 = call float @llvm.amdgcn.exp2.f32(float %1314), !dbg !32
  %1316 = insertelement <4 x float> poison, float %1315, i64 0, !dbg !34
  %1317 = fadd float %1313, %1315, !dbg !33
  %1318 = extractelement <4 x float> %1060, i64 1, !dbg !32
  %1319 = call float @llvm.amdgcn.exp2.f32(float %1318), !dbg !32
  %1320 = insertelement <4 x float> %1316, float %1319, i64 1, !dbg !34
  %1321 = fadd float %1317, %1319, !dbg !33
  %1322 = extractelement <4 x float> %1060, i64 2, !dbg !32
  %1323 = call float @llvm.amdgcn.exp2.f32(float %1322), !dbg !32
  %1324 = insertelement <4 x float> %1320, float %1323, i64 2, !dbg !34
  %1325 = fadd float %1321, %1323, !dbg !33
  %1326 = extractelement <4 x float> %1060, i64 3, !dbg !32
  %1327 = call float @llvm.amdgcn.exp2.f32(float %1326), !dbg !32
  %1328 = insertelement <4 x float> %1324, float %1327, i64 3, !dbg !34
  %math.result335 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1328) #9, !dbg !34, !frisk.register_lifetimes !1
  %1329 = fptrunc <4 x float> %math.result335 to <4 x half>, !dbg !34
  %math.packed336 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1329) #9, !dbg !33, !frisk.register_lifetimes !1
  %1330 = fadd float %1325, %1327, !dbg !33
  %1331 = extractelement <4 x float> %1061, i64 0, !dbg !32
  %1332 = call float @llvm.amdgcn.exp2.f32(float %1331), !dbg !32
  %1333 = insertelement <4 x float> poison, float %1332, i64 0, !dbg !34
  %1334 = fadd float %1330, %1332, !dbg !33
  %1335 = extractelement <4 x float> %1061, i64 1, !dbg !32
  %1336 = call float @llvm.amdgcn.exp2.f32(float %1335), !dbg !32
  %1337 = insertelement <4 x float> %1333, float %1336, i64 1, !dbg !34
  %1338 = fadd float %1334, %1336, !dbg !33
  %1339 = extractelement <4 x float> %1061, i64 2, !dbg !32
  %1340 = call float @llvm.amdgcn.exp2.f32(float %1339), !dbg !32
  %1341 = insertelement <4 x float> %1337, float %1340, i64 2, !dbg !34
  %1342 = fadd float %1338, %1340, !dbg !33
  %1343 = extractelement <4 x float> %1061, i64 3, !dbg !32
  %1344 = call float @llvm.amdgcn.exp2.f32(float %1343), !dbg !32
  %1345 = insertelement <4 x float> %1341, float %1344, i64 3, !dbg !34
  %math.result337 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1345) #9, !dbg !34, !frisk.register_lifetimes !1
  %1346 = fptrunc <4 x float> %math.result337 to <4 x half>, !dbg !34
  %math.packed338 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1346) #9, !dbg !33, !frisk.register_lifetimes !1
  %1347 = fadd float %1342, %1344, !dbg !33
  %1348 = bitcast float %1212 to i32, !dbg !33
  %1349 = call i32 @llvm.amdgcn.ds.bpermute(i32 %987, i32 %1348), !dbg !33
  %1350 = bitcast i32 %1349 to float, !dbg !33
  %1351 = fadd float %1212, %1350, !dbg !33
  %1352 = bitcast float %1351 to i32, !dbg !33
  %1353 = call i32 @llvm.amdgcn.ds.bpermute(i32 %995, i32 %1352), !dbg !33
  %1354 = bitcast i32 %1353 to float, !dbg !33
  %1355 = fadd float %1351, %1354, !dbg !33
  %1356 = fadd float %1355, %1069, !dbg !33
  %1357 = insertelement <1 x float> poison, float %1356, i64 0, !dbg !12
  %1358 = insertvalue [2 x <1 x float>] poison, <1 x float> %1357, 0, !dbg !12
  %1359 = bitcast float %1347 to i32, !dbg !33
  %1360 = call i32 @llvm.amdgcn.ds.bpermute(i32 %987, i32 %1359), !dbg !33
  %1361 = bitcast i32 %1360 to float, !dbg !33
  %1362 = fadd float %1347, %1361, !dbg !33
  %1363 = bitcast float %1362 to i32, !dbg !33
  %1364 = call i32 @llvm.amdgcn.ds.bpermute(i32 %995, i32 %1363), !dbg !33
  %1365 = bitcast i32 %1364 to float, !dbg !33
  %1366 = fadd float %1362, %1365, !dbg !33
  %1367 = fadd float %1366, %1077, !dbg !33
  %1368 = insertelement <1 x float> poison, float %1367, i64 0, !dbg !12
  %1369 = insertvalue [2 x <1 x float>] %1358, <1 x float> %1368, 1, !dbg !12
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
  %1370 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local871, i32 0) #0, !dbg !14
  %1371 = bitcast <4 x i32> %1370 to <8 x half>, !dbg !14
  %address.local872 = zext nneg i32 %address.thread864 to i64, !dbg !7
  %address.local873 = and i64 %address.local872, 60, !dbg !14
  %address.local874 = add nuw nsw i64 %address.local873, %address.local872, !dbg !14
  %address.local875 = shl nuw nsw i64 %address.local874, 3, !dbg !14
  %address.local876 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local875, !dbg !14
  %address.local877 = and i64 %address.local872, 63, !dbg !10
  %address.local878 = lshr i64 %address.local877, 4, !dbg !14
  %address.local879 = mul nuw nsw i64 %address.local878, 384, !dbg !14
  %address.local880 = getelementptr [2 x i8], ptr addrspace(3) %address.local876, i64 %address.local879, !dbg !14
  %address.local881 = lshr i64 %address.local872, 2, !dbg !17
  %address.local882 = mul nsw i64 %address.local881, -32, !dbg !14
  %address.local883 = getelementptr [2 x i8], ptr addrspace(3) %address.local880, i64 %address.local882, !dbg !14
  %address.local884 = getelementptr i8, ptr addrspace(3) %address.local883, i64 256, !dbg !14
  %1372 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local884, i32 0) #0, !dbg !14
  %1373 = bitcast <4 x i32> %1372 to <8 x half>, !dbg !14
  %address.local885 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local875, !dbg !14
  %address.local886 = getelementptr [2 x i8], ptr addrspace(3) %address.local885, i64 %address.local879, !dbg !14
  %address.local887 = getelementptr [2 x i8], ptr addrspace(3) %address.local886, i64 %address.local882, !dbg !14
  %address.local888 = getelementptr i8, ptr addrspace(3) %address.local887, i64 512, !dbg !14
  %1374 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local888, i32 0) #0, !dbg !14
  %1375 = bitcast <4 x i32> %1374 to <8 x half>, !dbg !14
  %address.local889 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local875, !dbg !14
  %address.local890 = getelementptr [2 x i8], ptr addrspace(3) %address.local889, i64 %address.local879, !dbg !14
  %address.local891 = getelementptr [2 x i8], ptr addrspace(3) %address.local890, i64 %address.local882, !dbg !14
  %address.local892 = getelementptr i8, ptr addrspace(3) %address.local891, i64 768, !dbg !14
  %1376 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local892, i32 0) #0, !dbg !14
  %1377 = bitcast <4 x i32> %1376 to <8 x half>, !dbg !14
  %address.local893 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local870, !dbg !14
  %address.local894 = getelementptr i8, ptr addrspace(3) %address.local893, i64 4096, !dbg !14
  %1378 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local894, i32 0) #0, !dbg !14
  %1379 = bitcast <4 x i32> %1378 to <8 x half>, !dbg !14
  %address.local895 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local875, !dbg !14
  %address.local896 = getelementptr [2 x i8], ptr addrspace(3) %address.local895, i64 %address.local879, !dbg !14
  %address.local897 = getelementptr [2 x i8], ptr addrspace(3) %address.local896, i64 %address.local882, !dbg !14
  %address.local898 = getelementptr i8, ptr addrspace(3) %address.local897, i64 4352, !dbg !14
  %1380 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local898, i32 0) #0, !dbg !14
  %1381 = bitcast <4 x i32> %1380 to <8 x half>, !dbg !14
  %address.local899 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local875, !dbg !14
  %address.local900 = getelementptr [2 x i8], ptr addrspace(3) %address.local899, i64 %address.local879, !dbg !14
  %address.local901 = getelementptr [2 x i8], ptr addrspace(3) %address.local900, i64 %address.local882, !dbg !14
  %address.local902 = getelementptr i8, ptr addrspace(3) %address.local901, i64 4608, !dbg !14
  %1382 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local902, i32 0) #0, !dbg !14
  %1383 = bitcast <4 x i32> %1382 to <8 x half>, !dbg !14
  %address.local903 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local875, !dbg !14
  %address.local904 = getelementptr [2 x i8], ptr addrspace(3) %address.local903, i64 %address.local879, !dbg !14
  %address.local905 = getelementptr [2 x i8], ptr addrspace(3) %address.local904, i64 %address.local882, !dbg !14
  %address.local906 = getelementptr i8, ptr addrspace(3) %address.local905, i64 4864, !dbg !14
  %1384 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local906, i32 0) #0, !dbg !14
  %1385 = bitcast <4 x i32> %1384 to <8 x half>, !dbg !14
  %1386 = add nuw nsw i64 %218, %16, !dbg !18
  %1387 = shl i64 %1386, 19, !dbg !18
  %1388 = and i64 %1387, 16252928, !dbg !18
  %1389 = add i64 %1388, %224, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1390 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %1389, !dbg !18
  %1391 = getelementptr i8, ptr addrspace(1) %1390, i64 64, !dbg !18
  %address.local907 = shl nuw nsw i64 %address.local872, 3, !dbg !7
  %address.local908 = and i64 %address.local907, 1536, !dbg !7
  %address.local909 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local908, !dbg !7
  %address.local910 = ptrtoint ptr addrspace(3) %address.local909 to i64, !dbg !7
  %address.local911 = trunc i64 %address.local910 to i32, !dbg !7
  %1392 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local911), !dbg !18
  %1393 = zext i32 %1392 to i64, !dbg !18
  %1394 = inttoptr i64 %1393 to ptr addrspace(3), !dbg !18
  %address.local912 = shl nuw nsw i64 %address.local872, 5, !dbg !18
  %address.local913 = and i64 %address.local912, 8064, !dbg !18
  %address.local914 = shl nuw nsw i64 %address.local872, 3, !dbg !18
  %address.local915 = and i64 %address.local914, 24, !dbg !18
  %address.local916 = or disjoint i64 %address.local913, %address.local915, !dbg !18
  %address.local917 = getelementptr [2 x i8], ptr addrspace(1) %1391, i64 %address.local916, !dbg !18
  %1395 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1396 = ptrtoint ptr addrspace(1) %address.local917 to i64, !dbg !18
  %1397 = sub i64 %1396, %1395, !dbg !18
  %dma.buffer.offset1558 = trunc i64 %1397 to i32, !dbg !18
  %1398 = trunc i64 %1395 to i32, !dbg !18
  %1399 = insertelement <4 x i32> poison, i32 %1398, i64 0, !dbg !18
  %1400 = lshr i64 %1395, 32, !dbg !18
  %1401 = trunc i64 %1400 to i32, !dbg !18
  %1402 = insertelement <4 x i32> %1399, i32 %1401, i64 1, !dbg !18
  %1403 = insertelement <4 x i32> %1402, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1559 = insertelement <4 x i32> %1403, i32 131072, i64 3, !dbg !18
  %1404 = ptrtoint ptr addrspace(3) %1394 to i32, !dbg !18
  %dma.buffer.lds1560 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1404), !dbg !18
  %1405 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1560, i32 %dma.buffer.offset1558, <4 x i32> %dma.buffer.rsrc1559) #10, !dbg !18
  %address.local918 = or disjoint i64 %address.local872, 256, !dbg !7
  %address.local919 = shl nuw nsw i64 %address.local918, 3, !dbg !7
  %address.local920 = and i64 %address.local919, 3584, !dbg !7
  %address.local921 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local920, !dbg !7
  %address.local922 = ptrtoint ptr addrspace(3) %address.local921 to i64, !dbg !7
  %address.local923 = trunc i64 %address.local922 to i32, !dbg !7
  %1406 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local923), !dbg !18
  %1407 = zext i32 %1406 to i64, !dbg !18
  %1408 = inttoptr i64 %1407 to ptr addrspace(3), !dbg !18
  %address.local924 = shl nuw nsw i64 %address.local918, 5, !dbg !18
  %address.local925 = and i64 %address.local924, 16256, !dbg !18
  %address.local926 = shl nuw nsw i64 %address.local872, 3, !dbg !18
  %address.local927 = and i64 %address.local926, 24, !dbg !18
  %address.local928 = or disjoint i64 %address.local925, %address.local927, !dbg !18
  %address.local929 = getelementptr [2 x i8], ptr addrspace(1) %1391, i64 %address.local928, !dbg !18
  %1409 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1410 = ptrtoint ptr addrspace(1) %address.local929 to i64, !dbg !18
  %1411 = sub i64 %1410, %1409, !dbg !18
  %dma.buffer.offset1555 = trunc i64 %1411 to i32, !dbg !18
  %1412 = trunc i64 %1409 to i32, !dbg !18
  %1413 = insertelement <4 x i32> poison, i32 %1412, i64 0, !dbg !18
  %1414 = lshr i64 %1409, 32, !dbg !18
  %1415 = trunc i64 %1414 to i32, !dbg !18
  %1416 = insertelement <4 x i32> %1413, i32 %1415, i64 1, !dbg !18
  %1417 = insertelement <4 x i32> %1416, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1556 = insertelement <4 x i32> %1417, i32 131072, i64 3, !dbg !18
  %1418 = ptrtoint ptr addrspace(3) %1408 to i32, !dbg !18
  %dma.buffer.lds1557 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1418), !dbg !18
  %1419 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1557, i32 %dma.buffer.offset1555, <4 x i32> %dma.buffer.rsrc1556) #10, !dbg !18
  %1420 = shufflevector <8 x half> %1371, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1421 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1420, <4 x float> zeroinitializer) #9, !dbg !14
  %1422 = shufflevector <8 x half> %1373, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1423 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %1422, <4 x float> %1421) #9, !dbg !14
  %1424 = shufflevector <8 x half> %1375, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1425 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %1424, <4 x float> %1423) #9, !dbg !14
  %1426 = shufflevector <8 x half> %1377, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1427 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %1426, <4 x float> %1425) #9, !dbg !14
  %1428 = shufflevector <8 x half> %1379, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1429 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %1428, <4 x float> %1427) #9, !dbg !14
  %1430 = shufflevector <8 x half> %1381, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1431 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %1430, <4 x float> %1429) #9, !dbg !14
  %1432 = shufflevector <8 x half> %1383, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1433 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %1432, <4 x float> %1431) #9, !dbg !14
  %1434 = shufflevector <8 x half> %1385, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1435 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %1434, <4 x float> %1433) #9, !dbg !14
  %1436 = extractelement <4 x float> %1435, i64 0, !dbg !14
  %1437 = extractelement <4 x float> %1435, i64 1, !dbg !14
  %1438 = extractelement <4 x float> %1435, i64 2, !dbg !14
  %1439 = extractelement <4 x float> %1435, i64 3, !dbg !14
  %1440 = select i1 %.not, float %1437, float %1436, !dbg !14
  %1441 = bitcast float %1440 to i32, !dbg !14
  %1442 = call i32 @llvm.amdgcn.ds.bpermute(i32 %987, i32 %1441), !dbg !14
  %1443 = bitcast i32 %1442 to float, !dbg !14
  %1444 = select i1 %.not, float %1436, float %1443, !dbg !14
  %1445 = select i1 %.not, float %1443, float %1437, !dbg !14
  %1446 = select i1 %.not, float %1439, float %1438, !dbg !14
  %1447 = bitcast float %1446 to i32, !dbg !14
  %1448 = call i32 @llvm.amdgcn.ds.bpermute(i32 %987, i32 %1447), !dbg !14
  %1449 = bitcast i32 %1448 to float, !dbg !14
  %1450 = select i1 %.not, float %1438, float %1449, !dbg !14
  %1451 = select i1 %.not, float %1449, float %1439, !dbg !14
  %1452 = select i1 %.not1, float %1450, float %1444, !dbg !14
  %1453 = bitcast float %1452 to i32, !dbg !14
  %1454 = call i32 @llvm.amdgcn.ds.bpermute(i32 %995, i32 %1453), !dbg !14
  %1455 = bitcast i32 %1454 to float, !dbg !14
  %1456 = select i1 %.not1, float %1444, float %1455, !dbg !14
  %1457 = select i1 %.not1, float %1455, float %1450, !dbg !14
  %1458 = select i1 %.not1, float %1451, float %1445, !dbg !14
  %1459 = bitcast float %1458 to i32, !dbg !14
  %1460 = call i32 @llvm.amdgcn.ds.bpermute(i32 %995, i32 %1459), !dbg !14
  %1461 = bitcast i32 %1460 to float, !dbg !14
  %1462 = select i1 %.not1, float %1445, float %1461, !dbg !14
  %1463 = select i1 %.not1, float %1461, float %1451, !dbg !14
  %1464 = extractvalue [2 x <8 x float>] %208, 0, !dbg !35
  %1465 = shufflevector <8 x float> %1464, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1466 = fmul <4 x float> %1465, %1066, !dbg !35
  %1467 = insertelement <4 x float> poison, float %1456, i64 0, !dbg !36
  %1468 = insertelement <4 x float> %1467, float %1462, i64 1, !dbg !36
  %1469 = insertelement <4 x float> %1468, float %1457, i64 2, !dbg !36
  %1470 = insertelement <4 x float> %1469, float %1463, i64 3, !dbg !36
  %1471 = fadd <4 x float> %1466, %1470, !dbg !36
  %1472 = shufflevector <8 x half> %1371, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1473 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1472, <4 x float> zeroinitializer) #9, !dbg !14
  %1474 = shufflevector <8 x half> %1373, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1475 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %1474, <4 x float> %1473) #9, !dbg !14
  %1476 = shufflevector <8 x half> %1375, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1477 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %1476, <4 x float> %1475) #9, !dbg !14
  %1478 = shufflevector <8 x half> %1377, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1479 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %1478, <4 x float> %1477) #9, !dbg !14
  %1480 = shufflevector <8 x half> %1379, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1481 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %1480, <4 x float> %1479) #9, !dbg !14
  %1482 = shufflevector <8 x half> %1381, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1483 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %1482, <4 x float> %1481) #9, !dbg !14
  %1484 = shufflevector <8 x half> %1383, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1485 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %1484, <4 x float> %1483) #9, !dbg !14
  %1486 = shufflevector <8 x half> %1385, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1487 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %1486, <4 x float> %1485) #9, !dbg !14
  %1488 = extractelement <4 x float> %1487, i64 0, !dbg !14
  %1489 = extractelement <4 x float> %1487, i64 1, !dbg !14
  %1490 = extractelement <4 x float> %1487, i64 2, !dbg !14
  %1491 = extractelement <4 x float> %1487, i64 3, !dbg !14
  %1492 = select i1 %.not, float %1489, float %1488, !dbg !14
  %1493 = bitcast float %1492 to i32, !dbg !14
  %1494 = call i32 @llvm.amdgcn.ds.bpermute(i32 %987, i32 %1493), !dbg !14
  %1495 = bitcast i32 %1494 to float, !dbg !14
  %1496 = select i1 %.not, float %1488, float %1495, !dbg !14
  %1497 = select i1 %.not, float %1495, float %1489, !dbg !14
  %1498 = select i1 %.not, float %1491, float %1490, !dbg !14
  %1499 = bitcast float %1498 to i32, !dbg !14
  %1500 = call i32 @llvm.amdgcn.ds.bpermute(i32 %987, i32 %1499), !dbg !14
  %1501 = bitcast i32 %1500 to float, !dbg !14
  %1502 = select i1 %.not, float %1490, float %1501, !dbg !14
  %1503 = select i1 %.not, float %1501, float %1491, !dbg !14
  %1504 = select i1 %.not1, float %1502, float %1496, !dbg !14
  %1505 = bitcast float %1504 to i32, !dbg !14
  %1506 = call i32 @llvm.amdgcn.ds.bpermute(i32 %995, i32 %1505), !dbg !14
  %1507 = bitcast i32 %1506 to float, !dbg !14
  %1508 = select i1 %.not1, float %1496, float %1507, !dbg !14
  %1509 = select i1 %.not1, float %1507, float %1502, !dbg !14
  %1510 = select i1 %.not1, float %1503, float %1497, !dbg !14
  %1511 = bitcast float %1510 to i32, !dbg !14
  %1512 = call i32 @llvm.amdgcn.ds.bpermute(i32 %995, i32 %1511), !dbg !14
  %1513 = bitcast i32 %1512 to float, !dbg !14
  %1514 = select i1 %.not1, float %1497, float %1513, !dbg !14
  %1515 = select i1 %.not1, float %1513, float %1503, !dbg !14
  %1516 = shufflevector <8 x float> %1464, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1517 = fmul <4 x float> %1516, %1066, !dbg !35
  %1518 = insertelement <4 x float> poison, float %1508, i64 0, !dbg !36
  %1519 = insertelement <4 x float> %1518, float %1514, i64 1, !dbg !36
  %1520 = insertelement <4 x float> %1519, float %1509, i64 2, !dbg !36
  %1521 = insertelement <4 x float> %1520, float %1515, i64 3, !dbg !36
  %1522 = fadd <4 x float> %1517, %1521, !dbg !36
  %1523 = shufflevector <4 x float> %1471, <4 x float> %1522, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %1524 = insertvalue [2 x <8 x float>] poison, <8 x float> %1523, 0, !dbg !12
  %1525 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1420, <4 x float> zeroinitializer) #9, !dbg !14
  %1526 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1422, <4 x float> %1525) #9, !dbg !14
  %1527 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1424, <4 x float> %1526) #9, !dbg !14
  %1528 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1426, <4 x float> %1527) #9, !dbg !14
  %1529 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %1428, <4 x float> %1528) #9, !dbg !14
  %1530 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %1430, <4 x float> %1529) #9, !dbg !14
  %1531 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %1432, <4 x float> %1530) #9, !dbg !14
  %1532 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %1434, <4 x float> %1531) #9, !dbg !14
  %1533 = extractelement <4 x float> %1532, i64 0, !dbg !14
  %1534 = extractelement <4 x float> %1532, i64 1, !dbg !14
  %1535 = extractelement <4 x float> %1532, i64 2, !dbg !14
  %1536 = extractelement <4 x float> %1532, i64 3, !dbg !14
  %1537 = select i1 %.not, float %1534, float %1533, !dbg !14
  %1538 = bitcast float %1537 to i32, !dbg !14
  %1539 = call i32 @llvm.amdgcn.ds.bpermute(i32 %987, i32 %1538), !dbg !14
  %1540 = bitcast i32 %1539 to float, !dbg !14
  %1541 = select i1 %.not, float %1533, float %1540, !dbg !14
  %1542 = select i1 %.not, float %1540, float %1534, !dbg !14
  %1543 = select i1 %.not, float %1536, float %1535, !dbg !14
  %1544 = bitcast float %1543 to i32, !dbg !14
  %1545 = call i32 @llvm.amdgcn.ds.bpermute(i32 %987, i32 %1544), !dbg !14
  %1546 = bitcast i32 %1545 to float, !dbg !14
  %1547 = select i1 %.not, float %1535, float %1546, !dbg !14
  %1548 = select i1 %.not, float %1546, float %1536, !dbg !14
  %1549 = select i1 %.not1, float %1547, float %1541, !dbg !14
  %1550 = bitcast float %1549 to i32, !dbg !14
  %1551 = call i32 @llvm.amdgcn.ds.bpermute(i32 %995, i32 %1550), !dbg !14
  %1552 = bitcast i32 %1551 to float, !dbg !14
  %1553 = select i1 %.not1, float %1541, float %1552, !dbg !14
  %1554 = select i1 %.not1, float %1552, float %1547, !dbg !14
  %1555 = select i1 %.not1, float %1548, float %1542, !dbg !14
  %1556 = bitcast float %1555 to i32, !dbg !14
  %1557 = call i32 @llvm.amdgcn.ds.bpermute(i32 %995, i32 %1556), !dbg !14
  %1558 = bitcast i32 %1557 to float, !dbg !14
  %1559 = select i1 %.not1, float %1542, float %1558, !dbg !14
  %1560 = select i1 %.not1, float %1558, float %1548, !dbg !14
  %1561 = extractvalue [2 x <8 x float>] %208, 1, !dbg !35
  %1562 = shufflevector <8 x float> %1561, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1563 = fmul <4 x float> %1562, %1074, !dbg !35
  %1564 = insertelement <4 x float> poison, float %1553, i64 0, !dbg !36
  %1565 = insertelement <4 x float> %1564, float %1559, i64 1, !dbg !36
  %1566 = insertelement <4 x float> %1565, float %1554, i64 2, !dbg !36
  %1567 = insertelement <4 x float> %1566, float %1560, i64 3, !dbg !36
  %1568 = fadd <4 x float> %1563, %1567, !dbg !36
  %1569 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1472, <4 x float> zeroinitializer) #9, !dbg !14
  %1570 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1474, <4 x float> %1569) #9, !dbg !14
  %1571 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1476, <4 x float> %1570) #9, !dbg !14
  %1572 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1478, <4 x float> %1571) #9, !dbg !14
  %1573 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %1480, <4 x float> %1572) #9, !dbg !14
  %1574 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %1482, <4 x float> %1573) #9, !dbg !14
  %1575 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %1484, <4 x float> %1574) #9, !dbg !14
  %1576 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %1486, <4 x float> %1575) #9, !dbg !14
  %1577 = extractelement <4 x float> %1576, i64 0, !dbg !14
  %1578 = extractelement <4 x float> %1576, i64 1, !dbg !14
  %1579 = extractelement <4 x float> %1576, i64 2, !dbg !14
  %1580 = extractelement <4 x float> %1576, i64 3, !dbg !14
  %1581 = select i1 %.not, float %1578, float %1577, !dbg !14
  %1582 = bitcast float %1581 to i32, !dbg !14
  %1583 = call i32 @llvm.amdgcn.ds.bpermute(i32 %987, i32 %1582), !dbg !14
  %1584 = bitcast i32 %1583 to float, !dbg !14
  %1585 = select i1 %.not, float %1577, float %1584, !dbg !14
  %1586 = select i1 %.not, float %1584, float %1578, !dbg !14
  %1587 = select i1 %.not, float %1580, float %1579, !dbg !14
  %1588 = bitcast float %1587 to i32, !dbg !14
  %1589 = call i32 @llvm.amdgcn.ds.bpermute(i32 %987, i32 %1588), !dbg !14
  %1590 = bitcast i32 %1589 to float, !dbg !14
  %1591 = select i1 %.not, float %1579, float %1590, !dbg !14
  %1592 = select i1 %.not, float %1590, float %1580, !dbg !14
  %1593 = select i1 %.not1, float %1591, float %1585, !dbg !14
  %1594 = bitcast float %1593 to i32, !dbg !14
  %1595 = call i32 @llvm.amdgcn.ds.bpermute(i32 %995, i32 %1594), !dbg !14
  %1596 = bitcast i32 %1595 to float, !dbg !14
  %1597 = select i1 %.not1, float %1585, float %1596, !dbg !14
  %1598 = select i1 %.not1, float %1596, float %1591, !dbg !14
  %1599 = select i1 %.not1, float %1592, float %1586, !dbg !14
  %1600 = bitcast float %1599 to i32, !dbg !14
  %1601 = call i32 @llvm.amdgcn.ds.bpermute(i32 %995, i32 %1600), !dbg !14
  %1602 = bitcast i32 %1601 to float, !dbg !14
  %1603 = select i1 %.not1, float %1586, float %1602, !dbg !14
  %1604 = select i1 %.not1, float %1602, float %1592, !dbg !14
  %1605 = shufflevector <8 x float> %1561, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1606 = fmul <4 x float> %1605, %1074, !dbg !35
  %1607 = insertelement <4 x float> poison, float %1597, i64 0, !dbg !36
  %1608 = insertelement <4 x float> %1607, float %1603, i64 1, !dbg !36
  %1609 = insertelement <4 x float> %1608, float %1598, i64 2, !dbg !36
  %1610 = insertelement <4 x float> %1609, float %1604, i64 3, !dbg !36
  %1611 = fadd <4 x float> %1606, %1610, !dbg !36
  %1612 = shufflevector <4 x float> %1568, <4 x float> %1611, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %1613 = insertvalue [2 x <8 x float>] %1524, <8 x float> %1612, 1, !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1614 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local871, i32 0) #0, !dbg !14
  %1615 = bitcast <4 x i32> %1614 to <8 x half>, !dbg !14
  %1616 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local884, i32 0) #0, !dbg !14
  %1617 = bitcast <4 x i32> %1616 to <8 x half>, !dbg !14
  %1618 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local888, i32 0) #0, !dbg !14
  %1619 = bitcast <4 x i32> %1618 to <8 x half>, !dbg !14
  %1620 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local892, i32 0) #0, !dbg !14
  %1621 = bitcast <4 x i32> %1620 to <8 x half>, !dbg !14
  %1622 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local894, i32 0) #0, !dbg !14
  %1623 = bitcast <4 x i32> %1622 to <8 x half>, !dbg !14
  %1624 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local898, i32 0) #0, !dbg !14
  %1625 = bitcast <4 x i32> %1624 to <8 x half>, !dbg !14
  %1626 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local902, i32 0) #0, !dbg !14
  %1627 = bitcast <4 x i32> %1626 to <8 x half>, !dbg !14
  %1628 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local906, i32 0) #0, !dbg !14
  %1629 = bitcast <4 x i32> %1628 to <8 x half>, !dbg !14
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1630 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %1389, !dbg !18
  %1631 = getelementptr i8, ptr addrspace(1) %1630, i64 128, !dbg !18
  %1632 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local911), !dbg !18
  %1633 = zext i32 %1632 to i64, !dbg !18
  %1634 = inttoptr i64 %1633 to ptr addrspace(3), !dbg !18
  %address.local930 = getelementptr [2 x i8], ptr addrspace(1) %1631, i64 %address.local916, !dbg !18
  %1635 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1636 = ptrtoint ptr addrspace(1) %address.local930 to i64, !dbg !18
  %1637 = sub i64 %1636, %1635, !dbg !18
  %dma.buffer.offset1552 = trunc i64 %1637 to i32, !dbg !18
  %1638 = trunc i64 %1635 to i32, !dbg !18
  %1639 = insertelement <4 x i32> poison, i32 %1638, i64 0, !dbg !18
  %1640 = lshr i64 %1635, 32, !dbg !18
  %1641 = trunc i64 %1640 to i32, !dbg !18
  %1642 = insertelement <4 x i32> %1639, i32 %1641, i64 1, !dbg !18
  %1643 = insertelement <4 x i32> %1642, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1553 = insertelement <4 x i32> %1643, i32 131072, i64 3, !dbg !18
  %1644 = ptrtoint ptr addrspace(3) %1634 to i32, !dbg !18
  %dma.buffer.lds1554 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1644), !dbg !18
  %1645 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1554, i32 %dma.buffer.offset1552, <4 x i32> %dma.buffer.rsrc1553) #10, !dbg !18
  %1646 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local923), !dbg !18
  %1647 = zext i32 %1646 to i64, !dbg !18
  %1648 = inttoptr i64 %1647 to ptr addrspace(3), !dbg !18
  %address.local931 = getelementptr [2 x i8], ptr addrspace(1) %1631, i64 %address.local928, !dbg !18
  %1649 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1650 = ptrtoint ptr addrspace(1) %address.local931 to i64, !dbg !18
  %1651 = sub i64 %1650, %1649, !dbg !18
  %dma.buffer.offset1549 = trunc i64 %1651 to i32, !dbg !18
  %1652 = trunc i64 %1649 to i32, !dbg !18
  %1653 = insertelement <4 x i32> poison, i32 %1652, i64 0, !dbg !18
  %1654 = lshr i64 %1649, 32, !dbg !18
  %1655 = trunc i64 %1654 to i32, !dbg !18
  %1656 = insertelement <4 x i32> %1653, i32 %1655, i64 1, !dbg !18
  %1657 = insertelement <4 x i32> %1656, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1550 = insertelement <4 x i32> %1657, i32 131072, i64 3, !dbg !18
  %1658 = ptrtoint ptr addrspace(3) %1648 to i32, !dbg !18
  %dma.buffer.lds1551 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1658), !dbg !18
  %1659 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1551, i32 %dma.buffer.offset1549, <4 x i32> %dma.buffer.rsrc1550) #10, !dbg !18
  %1660 = shufflevector <8 x half> %1615, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1661 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1660, <4 x float> zeroinitializer) #9, !dbg !14
  %1662 = shufflevector <8 x half> %1617, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1663 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %1662, <4 x float> %1661) #9, !dbg !14
  %1664 = shufflevector <8 x half> %1619, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1665 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %1664, <4 x float> %1663) #9, !dbg !14
  %1666 = shufflevector <8 x half> %1621, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1667 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %1666, <4 x float> %1665) #9, !dbg !14
  %1668 = shufflevector <8 x half> %1623, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1669 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %1668, <4 x float> %1667) #9, !dbg !14
  %1670 = shufflevector <8 x half> %1625, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1671 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %1670, <4 x float> %1669) #9, !dbg !14
  %1672 = shufflevector <8 x half> %1627, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1673 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %1672, <4 x float> %1671) #9, !dbg !14
  %1674 = shufflevector <8 x half> %1629, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1675 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %1674, <4 x float> %1673) #9, !dbg !14
  %1676 = extractelement <4 x float> %1675, i64 0, !dbg !14
  %1677 = extractelement <4 x float> %1675, i64 1, !dbg !14
  %1678 = extractelement <4 x float> %1675, i64 2, !dbg !14
  %1679 = extractelement <4 x float> %1675, i64 3, !dbg !14
  %1680 = select i1 %.not, float %1677, float %1676, !dbg !14
  %1681 = bitcast float %1680 to i32, !dbg !14
  %1682 = call i32 @llvm.amdgcn.ds.bpermute(i32 %987, i32 %1681), !dbg !14
  %1683 = bitcast i32 %1682 to float, !dbg !14
  %1684 = select i1 %.not, float %1676, float %1683, !dbg !14
  %1685 = select i1 %.not, float %1683, float %1677, !dbg !14
  %1686 = select i1 %.not, float %1679, float %1678, !dbg !14
  %1687 = bitcast float %1686 to i32, !dbg !14
  %1688 = call i32 @llvm.amdgcn.ds.bpermute(i32 %987, i32 %1687), !dbg !14
  %1689 = bitcast i32 %1688 to float, !dbg !14
  %1690 = select i1 %.not, float %1678, float %1689, !dbg !14
  %1691 = select i1 %.not, float %1689, float %1679, !dbg !14
  %1692 = select i1 %.not1, float %1690, float %1684, !dbg !14
  %1693 = bitcast float %1692 to i32, !dbg !14
  %1694 = call i32 @llvm.amdgcn.ds.bpermute(i32 %995, i32 %1693), !dbg !14
  %1695 = bitcast i32 %1694 to float, !dbg !14
  %1696 = select i1 %.not1, float %1684, float %1695, !dbg !14
  %1697 = select i1 %.not1, float %1695, float %1690, !dbg !14
  %1698 = select i1 %.not1, float %1691, float %1685, !dbg !14
  %1699 = bitcast float %1698 to i32, !dbg !14
  %1700 = call i32 @llvm.amdgcn.ds.bpermute(i32 %995, i32 %1699), !dbg !14
  %1701 = bitcast i32 %1700 to float, !dbg !14
  %1702 = select i1 %.not1, float %1685, float %1701, !dbg !14
  %1703 = select i1 %.not1, float %1701, float %1691, !dbg !14
  %1704 = extractvalue [2 x <8 x float>] %210, 0, !dbg !35
  %1705 = shufflevector <8 x float> %1704, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1706 = fmul <4 x float> %1705, %1066, !dbg !35
  %1707 = insertelement <4 x float> poison, float %1696, i64 0, !dbg !36
  %1708 = insertelement <4 x float> %1707, float %1702, i64 1, !dbg !36
  %1709 = insertelement <4 x float> %1708, float %1697, i64 2, !dbg !36
  %1710 = insertelement <4 x float> %1709, float %1703, i64 3, !dbg !36
  %1711 = fadd <4 x float> %1706, %1710, !dbg !36
  %1712 = shufflevector <8 x half> %1615, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1713 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1712, <4 x float> zeroinitializer) #9, !dbg !14
  %1714 = shufflevector <8 x half> %1617, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1715 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %1714, <4 x float> %1713) #9, !dbg !14
  %1716 = shufflevector <8 x half> %1619, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1717 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %1716, <4 x float> %1715) #9, !dbg !14
  %1718 = shufflevector <8 x half> %1621, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1719 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %1718, <4 x float> %1717) #9, !dbg !14
  %1720 = shufflevector <8 x half> %1623, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1721 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %1720, <4 x float> %1719) #9, !dbg !14
  %1722 = shufflevector <8 x half> %1625, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1723 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %1722, <4 x float> %1721) #9, !dbg !14
  %1724 = shufflevector <8 x half> %1627, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1725 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %1724, <4 x float> %1723) #9, !dbg !14
  %1726 = shufflevector <8 x half> %1629, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1727 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %1726, <4 x float> %1725) #9, !dbg !14
  %1728 = extractelement <4 x float> %1727, i64 0, !dbg !14
  %1729 = extractelement <4 x float> %1727, i64 1, !dbg !14
  %1730 = extractelement <4 x float> %1727, i64 2, !dbg !14
  %1731 = extractelement <4 x float> %1727, i64 3, !dbg !14
  %1732 = select i1 %.not, float %1729, float %1728, !dbg !14
  %1733 = bitcast float %1732 to i32, !dbg !14
  %1734 = call i32 @llvm.amdgcn.ds.bpermute(i32 %987, i32 %1733), !dbg !14
  %1735 = bitcast i32 %1734 to float, !dbg !14
  %1736 = select i1 %.not, float %1728, float %1735, !dbg !14
  %1737 = select i1 %.not, float %1735, float %1729, !dbg !14
  %1738 = select i1 %.not, float %1731, float %1730, !dbg !14
  %1739 = bitcast float %1738 to i32, !dbg !14
  %1740 = call i32 @llvm.amdgcn.ds.bpermute(i32 %987, i32 %1739), !dbg !14
  %1741 = bitcast i32 %1740 to float, !dbg !14
  %1742 = select i1 %.not, float %1730, float %1741, !dbg !14
  %1743 = select i1 %.not, float %1741, float %1731, !dbg !14
  %1744 = select i1 %.not1, float %1742, float %1736, !dbg !14
  %1745 = bitcast float %1744 to i32, !dbg !14
  %1746 = call i32 @llvm.amdgcn.ds.bpermute(i32 %995, i32 %1745), !dbg !14
  %1747 = bitcast i32 %1746 to float, !dbg !14
  %1748 = select i1 %.not1, float %1736, float %1747, !dbg !14
  %1749 = select i1 %.not1, float %1747, float %1742, !dbg !14
  %1750 = select i1 %.not1, float %1743, float %1737, !dbg !14
  %1751 = bitcast float %1750 to i32, !dbg !14
  %1752 = call i32 @llvm.amdgcn.ds.bpermute(i32 %995, i32 %1751), !dbg !14
  %1753 = bitcast i32 %1752 to float, !dbg !14
  %1754 = select i1 %.not1, float %1737, float %1753, !dbg !14
  %1755 = select i1 %.not1, float %1753, float %1743, !dbg !14
  %1756 = shufflevector <8 x float> %1704, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1757 = fmul <4 x float> %1756, %1066, !dbg !35
  %1758 = insertelement <4 x float> poison, float %1748, i64 0, !dbg !36
  %1759 = insertelement <4 x float> %1758, float %1754, i64 1, !dbg !36
  %1760 = insertelement <4 x float> %1759, float %1749, i64 2, !dbg !36
  %1761 = insertelement <4 x float> %1760, float %1755, i64 3, !dbg !36
  %1762 = fadd <4 x float> %1757, %1761, !dbg !36
  %1763 = shufflevector <4 x float> %1711, <4 x float> %1762, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %1764 = insertvalue [2 x <8 x float>] poison, <8 x float> %1763, 0, !dbg !12
  %1765 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1660, <4 x float> zeroinitializer) #9, !dbg !14
  %1766 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1662, <4 x float> %1765) #9, !dbg !14
  %1767 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1664, <4 x float> %1766) #9, !dbg !14
  %1768 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1666, <4 x float> %1767) #9, !dbg !14
  %1769 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %1668, <4 x float> %1768) #9, !dbg !14
  %1770 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %1670, <4 x float> %1769) #9, !dbg !14
  %1771 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %1672, <4 x float> %1770) #9, !dbg !14
  %1772 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %1674, <4 x float> %1771) #9, !dbg !14
  %1773 = extractelement <4 x float> %1772, i64 0, !dbg !14
  %1774 = extractelement <4 x float> %1772, i64 1, !dbg !14
  %1775 = extractelement <4 x float> %1772, i64 2, !dbg !14
  %1776 = extractelement <4 x float> %1772, i64 3, !dbg !14
  %1777 = select i1 %.not, float %1774, float %1773, !dbg !14
  %1778 = bitcast float %1777 to i32, !dbg !14
  %1779 = call i32 @llvm.amdgcn.ds.bpermute(i32 %987, i32 %1778), !dbg !14
  %1780 = bitcast i32 %1779 to float, !dbg !14
  %1781 = select i1 %.not, float %1773, float %1780, !dbg !14
  %1782 = select i1 %.not, float %1780, float %1774, !dbg !14
  %1783 = select i1 %.not, float %1776, float %1775, !dbg !14
  %1784 = bitcast float %1783 to i32, !dbg !14
  %1785 = call i32 @llvm.amdgcn.ds.bpermute(i32 %987, i32 %1784), !dbg !14
  %1786 = bitcast i32 %1785 to float, !dbg !14
  %1787 = select i1 %.not, float %1775, float %1786, !dbg !14
  %1788 = select i1 %.not, float %1786, float %1776, !dbg !14
  %1789 = select i1 %.not1, float %1787, float %1781, !dbg !14
  %1790 = bitcast float %1789 to i32, !dbg !14
  %1791 = call i32 @llvm.amdgcn.ds.bpermute(i32 %995, i32 %1790), !dbg !14
  %1792 = bitcast i32 %1791 to float, !dbg !14
  %1793 = select i1 %.not1, float %1781, float %1792, !dbg !14
  %1794 = select i1 %.not1, float %1792, float %1787, !dbg !14
  %1795 = select i1 %.not1, float %1788, float %1782, !dbg !14
  %1796 = bitcast float %1795 to i32, !dbg !14
  %1797 = call i32 @llvm.amdgcn.ds.bpermute(i32 %995, i32 %1796), !dbg !14
  %1798 = bitcast i32 %1797 to float, !dbg !14
  %1799 = select i1 %.not1, float %1782, float %1798, !dbg !14
  %1800 = select i1 %.not1, float %1798, float %1788, !dbg !14
  %1801 = extractvalue [2 x <8 x float>] %210, 1, !dbg !35
  %1802 = shufflevector <8 x float> %1801, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1803 = fmul <4 x float> %1802, %1074, !dbg !35
  %1804 = insertelement <4 x float> poison, float %1793, i64 0, !dbg !36
  %1805 = insertelement <4 x float> %1804, float %1799, i64 1, !dbg !36
  %1806 = insertelement <4 x float> %1805, float %1794, i64 2, !dbg !36
  %1807 = insertelement <4 x float> %1806, float %1800, i64 3, !dbg !36
  %1808 = fadd <4 x float> %1803, %1807, !dbg !36
  %1809 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1712, <4 x float> zeroinitializer) #9, !dbg !14
  %1810 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1714, <4 x float> %1809) #9, !dbg !14
  %1811 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1716, <4 x float> %1810) #9, !dbg !14
  %1812 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1718, <4 x float> %1811) #9, !dbg !14
  %1813 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %1720, <4 x float> %1812) #9, !dbg !14
  %1814 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %1722, <4 x float> %1813) #9, !dbg !14
  %1815 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %1724, <4 x float> %1814) #9, !dbg !14
  %1816 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %1726, <4 x float> %1815) #9, !dbg !14
  %1817 = extractelement <4 x float> %1816, i64 0, !dbg !14
  %1818 = extractelement <4 x float> %1816, i64 1, !dbg !14
  %1819 = extractelement <4 x float> %1816, i64 2, !dbg !14
  %1820 = extractelement <4 x float> %1816, i64 3, !dbg !14
  %1821 = select i1 %.not, float %1818, float %1817, !dbg !14
  %1822 = bitcast float %1821 to i32, !dbg !14
  %1823 = call i32 @llvm.amdgcn.ds.bpermute(i32 %987, i32 %1822), !dbg !14
  %1824 = bitcast i32 %1823 to float, !dbg !14
  %1825 = select i1 %.not, float %1817, float %1824, !dbg !14
  %1826 = select i1 %.not, float %1824, float %1818, !dbg !14
  %1827 = select i1 %.not, float %1820, float %1819, !dbg !14
  %1828 = bitcast float %1827 to i32, !dbg !14
  %1829 = call i32 @llvm.amdgcn.ds.bpermute(i32 %987, i32 %1828), !dbg !14
  %1830 = bitcast i32 %1829 to float, !dbg !14
  %1831 = select i1 %.not, float %1819, float %1830, !dbg !14
  %1832 = select i1 %.not, float %1830, float %1820, !dbg !14
  %1833 = select i1 %.not1, float %1831, float %1825, !dbg !14
  %1834 = bitcast float %1833 to i32, !dbg !14
  %1835 = call i32 @llvm.amdgcn.ds.bpermute(i32 %995, i32 %1834), !dbg !14
  %1836 = bitcast i32 %1835 to float, !dbg !14
  %1837 = select i1 %.not1, float %1825, float %1836, !dbg !14
  %1838 = select i1 %.not1, float %1836, float %1831, !dbg !14
  %1839 = select i1 %.not1, float %1832, float %1826, !dbg !14
  %1840 = bitcast float %1839 to i32, !dbg !14
  %1841 = call i32 @llvm.amdgcn.ds.bpermute(i32 %995, i32 %1840), !dbg !14
  %1842 = bitcast i32 %1841 to float, !dbg !14
  %1843 = select i1 %.not1, float %1826, float %1842, !dbg !14
  %1844 = select i1 %.not1, float %1842, float %1832, !dbg !14
  %1845 = shufflevector <8 x float> %1801, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1846 = fmul <4 x float> %1845, %1074, !dbg !35
  %1847 = insertelement <4 x float> poison, float %1837, i64 0, !dbg !36
  %1848 = insertelement <4 x float> %1847, float %1843, i64 1, !dbg !36
  %1849 = insertelement <4 x float> %1848, float %1838, i64 2, !dbg !36
  %1850 = insertelement <4 x float> %1849, float %1844, i64 3, !dbg !36
  %1851 = fadd <4 x float> %1846, %1850, !dbg !36
  %1852 = shufflevector <4 x float> %1808, <4 x float> %1851, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %1853 = insertvalue [2 x <8 x float>] %1764, <8 x float> %1852, 1, !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1854 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local871, i32 0) #0, !dbg !14
  %1855 = bitcast <4 x i32> %1854 to <8 x half>, !dbg !14
  %1856 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local884, i32 0) #0, !dbg !14
  %1857 = bitcast <4 x i32> %1856 to <8 x half>, !dbg !14
  %1858 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local888, i32 0) #0, !dbg !14
  %1859 = bitcast <4 x i32> %1858 to <8 x half>, !dbg !14
  %1860 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local892, i32 0) #0, !dbg !14
  %1861 = bitcast <4 x i32> %1860 to <8 x half>, !dbg !14
  %1862 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local894, i32 0) #0, !dbg !14
  %1863 = bitcast <4 x i32> %1862 to <8 x half>, !dbg !14
  %1864 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local898, i32 0) #0, !dbg !14
  %1865 = bitcast <4 x i32> %1864 to <8 x half>, !dbg !14
  %1866 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local902, i32 0) #0, !dbg !14
  %1867 = bitcast <4 x i32> %1866 to <8 x half>, !dbg !14
  %1868 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local906, i32 0) #0, !dbg !14
  %1869 = bitcast <4 x i32> %1868 to <8 x half>, !dbg !14
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1870 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %1389, !dbg !18
  %1871 = getelementptr i8, ptr addrspace(1) %1870, i64 192, !dbg !18
  %1872 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local911), !dbg !18
  %1873 = zext i32 %1872 to i64, !dbg !18
  %1874 = inttoptr i64 %1873 to ptr addrspace(3), !dbg !18
  %address.local932 = getelementptr [2 x i8], ptr addrspace(1) %1871, i64 %address.local916, !dbg !18
  %1875 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1876 = ptrtoint ptr addrspace(1) %address.local932 to i64, !dbg !18
  %1877 = sub i64 %1876, %1875, !dbg !18
  %dma.buffer.offset1546 = trunc i64 %1877 to i32, !dbg !18
  %1878 = trunc i64 %1875 to i32, !dbg !18
  %1879 = insertelement <4 x i32> poison, i32 %1878, i64 0, !dbg !18
  %1880 = lshr i64 %1875, 32, !dbg !18
  %1881 = trunc i64 %1880 to i32, !dbg !18
  %1882 = insertelement <4 x i32> %1879, i32 %1881, i64 1, !dbg !18
  %1883 = insertelement <4 x i32> %1882, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1547 = insertelement <4 x i32> %1883, i32 131072, i64 3, !dbg !18
  %1884 = ptrtoint ptr addrspace(3) %1874 to i32, !dbg !18
  %dma.buffer.lds1548 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1884), !dbg !18
  %1885 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1548, i32 %dma.buffer.offset1546, <4 x i32> %dma.buffer.rsrc1547) #10, !dbg !18
  %1886 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local923), !dbg !18
  %1887 = zext i32 %1886 to i64, !dbg !18
  %1888 = inttoptr i64 %1887 to ptr addrspace(3), !dbg !18
  %address.local933 = getelementptr [2 x i8], ptr addrspace(1) %1871, i64 %address.local928, !dbg !18
  %1889 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1890 = ptrtoint ptr addrspace(1) %address.local933 to i64, !dbg !18
  %1891 = sub i64 %1890, %1889, !dbg !18
  %dma.buffer.offset = trunc i64 %1891 to i32, !dbg !18
  %1892 = trunc i64 %1889 to i32, !dbg !18
  %1893 = insertelement <4 x i32> poison, i32 %1892, i64 0, !dbg !18
  %1894 = lshr i64 %1889, 32, !dbg !18
  %1895 = trunc i64 %1894 to i32, !dbg !18
  %1896 = insertelement <4 x i32> %1893, i32 %1895, i64 1, !dbg !18
  %1897 = insertelement <4 x i32> %1896, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc = insertelement <4 x i32> %1897, i32 131072, i64 3, !dbg !18
  %1898 = ptrtoint ptr addrspace(3) %1888 to i32, !dbg !18
  %dma.buffer.lds = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1898), !dbg !18
  %1899 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc) #10, !dbg !18
  %1900 = shufflevector <8 x half> %1855, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1901 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1900, <4 x float> zeroinitializer) #9, !dbg !14
  %1902 = shufflevector <8 x half> %1857, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1903 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %1902, <4 x float> %1901) #9, !dbg !14
  %1904 = shufflevector <8 x half> %1859, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1905 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %1904, <4 x float> %1903) #9, !dbg !14
  %1906 = shufflevector <8 x half> %1861, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1907 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %1906, <4 x float> %1905) #9, !dbg !14
  %1908 = shufflevector <8 x half> %1863, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1909 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %1908, <4 x float> %1907) #9, !dbg !14
  %1910 = shufflevector <8 x half> %1865, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1911 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %1910, <4 x float> %1909) #9, !dbg !14
  %1912 = shufflevector <8 x half> %1867, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1913 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %1912, <4 x float> %1911) #9, !dbg !14
  %1914 = shufflevector <8 x half> %1869, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1915 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %1914, <4 x float> %1913) #9, !dbg !14
  %1916 = extractelement <4 x float> %1915, i64 0, !dbg !14
  %1917 = extractelement <4 x float> %1915, i64 1, !dbg !14
  %1918 = extractelement <4 x float> %1915, i64 2, !dbg !14
  %1919 = extractelement <4 x float> %1915, i64 3, !dbg !14
  %1920 = select i1 %.not, float %1917, float %1916, !dbg !14
  %1921 = bitcast float %1920 to i32, !dbg !14
  %1922 = call i32 @llvm.amdgcn.ds.bpermute(i32 %987, i32 %1921), !dbg !14
  %1923 = bitcast i32 %1922 to float, !dbg !14
  %1924 = select i1 %.not, float %1916, float %1923, !dbg !14
  %1925 = select i1 %.not, float %1923, float %1917, !dbg !14
  %1926 = select i1 %.not, float %1919, float %1918, !dbg !14
  %1927 = bitcast float %1926 to i32, !dbg !14
  %1928 = call i32 @llvm.amdgcn.ds.bpermute(i32 %987, i32 %1927), !dbg !14
  %1929 = bitcast i32 %1928 to float, !dbg !14
  %1930 = select i1 %.not, float %1918, float %1929, !dbg !14
  %1931 = select i1 %.not, float %1929, float %1919, !dbg !14
  %1932 = select i1 %.not1, float %1930, float %1924, !dbg !14
  %1933 = bitcast float %1932 to i32, !dbg !14
  %1934 = call i32 @llvm.amdgcn.ds.bpermute(i32 %995, i32 %1933), !dbg !14
  %1935 = bitcast i32 %1934 to float, !dbg !14
  %1936 = select i1 %.not1, float %1924, float %1935, !dbg !14
  %1937 = select i1 %.not1, float %1935, float %1930, !dbg !14
  %1938 = select i1 %.not1, float %1931, float %1925, !dbg !14
  %1939 = bitcast float %1938 to i32, !dbg !14
  %1940 = call i32 @llvm.amdgcn.ds.bpermute(i32 %995, i32 %1939), !dbg !14
  %1941 = bitcast i32 %1940 to float, !dbg !14
  %1942 = select i1 %.not1, float %1925, float %1941, !dbg !14
  %1943 = select i1 %.not1, float %1941, float %1931, !dbg !14
  %1944 = extractvalue [2 x <8 x float>] %211, 0, !dbg !35
  %1945 = shufflevector <8 x float> %1944, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1946 = fmul <4 x float> %1945, %1066, !dbg !35
  %1947 = insertelement <4 x float> poison, float %1936, i64 0, !dbg !36
  %1948 = insertelement <4 x float> %1947, float %1942, i64 1, !dbg !36
  %1949 = insertelement <4 x float> %1948, float %1937, i64 2, !dbg !36
  %1950 = insertelement <4 x float> %1949, float %1943, i64 3, !dbg !36
  %1951 = fadd <4 x float> %1946, %1950, !dbg !36
  %1952 = shufflevector <8 x half> %1855, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1953 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1952, <4 x float> zeroinitializer) #9, !dbg !14
  %1954 = shufflevector <8 x half> %1857, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1955 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %1954, <4 x float> %1953) #9, !dbg !14
  %1956 = shufflevector <8 x half> %1859, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1957 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %1956, <4 x float> %1955) #9, !dbg !14
  %1958 = shufflevector <8 x half> %1861, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1959 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %1958, <4 x float> %1957) #9, !dbg !14
  %1960 = shufflevector <8 x half> %1863, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1961 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %1960, <4 x float> %1959) #9, !dbg !14
  %1962 = shufflevector <8 x half> %1865, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1963 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %1962, <4 x float> %1961) #9, !dbg !14
  %1964 = shufflevector <8 x half> %1867, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1965 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %1964, <4 x float> %1963) #9, !dbg !14
  %1966 = shufflevector <8 x half> %1869, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1967 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %1966, <4 x float> %1965) #9, !dbg !14
  %1968 = extractelement <4 x float> %1967, i64 0, !dbg !14
  %1969 = extractelement <4 x float> %1967, i64 1, !dbg !14
  %1970 = extractelement <4 x float> %1967, i64 2, !dbg !14
  %1971 = extractelement <4 x float> %1967, i64 3, !dbg !14
  %1972 = select i1 %.not, float %1969, float %1968, !dbg !14
  %1973 = bitcast float %1972 to i32, !dbg !14
  %1974 = call i32 @llvm.amdgcn.ds.bpermute(i32 %987, i32 %1973), !dbg !14
  %1975 = bitcast i32 %1974 to float, !dbg !14
  %1976 = select i1 %.not, float %1968, float %1975, !dbg !14
  %1977 = select i1 %.not, float %1975, float %1969, !dbg !14
  %1978 = select i1 %.not, float %1971, float %1970, !dbg !14
  %1979 = bitcast float %1978 to i32, !dbg !14
  %1980 = call i32 @llvm.amdgcn.ds.bpermute(i32 %987, i32 %1979), !dbg !14
  %1981 = bitcast i32 %1980 to float, !dbg !14
  %1982 = select i1 %.not, float %1970, float %1981, !dbg !14
  %1983 = select i1 %.not, float %1981, float %1971, !dbg !14
  %1984 = select i1 %.not1, float %1982, float %1976, !dbg !14
  %1985 = bitcast float %1984 to i32, !dbg !14
  %1986 = call i32 @llvm.amdgcn.ds.bpermute(i32 %995, i32 %1985), !dbg !14
  %1987 = bitcast i32 %1986 to float, !dbg !14
  %1988 = select i1 %.not1, float %1976, float %1987, !dbg !14
  %1989 = select i1 %.not1, float %1987, float %1982, !dbg !14
  %1990 = select i1 %.not1, float %1983, float %1977, !dbg !14
  %1991 = bitcast float %1990 to i32, !dbg !14
  %1992 = call i32 @llvm.amdgcn.ds.bpermute(i32 %995, i32 %1991), !dbg !14
  %1993 = bitcast i32 %1992 to float, !dbg !14
  %1994 = select i1 %.not1, float %1977, float %1993, !dbg !14
  %1995 = select i1 %.not1, float %1993, float %1983, !dbg !14
  %1996 = shufflevector <8 x float> %1944, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1997 = fmul <4 x float> %1996, %1066, !dbg !35
  %1998 = insertelement <4 x float> poison, float %1988, i64 0, !dbg !36
  %1999 = insertelement <4 x float> %1998, float %1994, i64 1, !dbg !36
  %2000 = insertelement <4 x float> %1999, float %1989, i64 2, !dbg !36
  %2001 = insertelement <4 x float> %2000, float %1995, i64 3, !dbg !36
  %2002 = fadd <4 x float> %1997, %2001, !dbg !36
  %2003 = shufflevector <4 x float> %1951, <4 x float> %2002, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %2004 = insertvalue [2 x <8 x float>] poison, <8 x float> %2003, 0, !dbg !12
  %2005 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1900, <4 x float> zeroinitializer) #9, !dbg !14
  %2006 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1902, <4 x float> %2005) #9, !dbg !14
  %2007 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1904, <4 x float> %2006) #9, !dbg !14
  %2008 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1906, <4 x float> %2007) #9, !dbg !14
  %2009 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %1908, <4 x float> %2008) #9, !dbg !14
  %2010 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %1910, <4 x float> %2009) #9, !dbg !14
  %2011 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %1912, <4 x float> %2010) #9, !dbg !14
  %2012 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %1914, <4 x float> %2011) #9, !dbg !14
  %2013 = extractelement <4 x float> %2012, i64 0, !dbg !14
  %2014 = extractelement <4 x float> %2012, i64 1, !dbg !14
  %2015 = extractelement <4 x float> %2012, i64 2, !dbg !14
  %2016 = extractelement <4 x float> %2012, i64 3, !dbg !14
  %2017 = select i1 %.not, float %2014, float %2013, !dbg !14
  %2018 = bitcast float %2017 to i32, !dbg !14
  %2019 = call i32 @llvm.amdgcn.ds.bpermute(i32 %987, i32 %2018), !dbg !14
  %2020 = bitcast i32 %2019 to float, !dbg !14
  %2021 = select i1 %.not, float %2013, float %2020, !dbg !14
  %2022 = select i1 %.not, float %2020, float %2014, !dbg !14
  %2023 = select i1 %.not, float %2016, float %2015, !dbg !14
  %2024 = bitcast float %2023 to i32, !dbg !14
  %2025 = call i32 @llvm.amdgcn.ds.bpermute(i32 %987, i32 %2024), !dbg !14
  %2026 = bitcast i32 %2025 to float, !dbg !14
  %2027 = select i1 %.not, float %2015, float %2026, !dbg !14
  %2028 = select i1 %.not, float %2026, float %2016, !dbg !14
  %2029 = select i1 %.not1, float %2027, float %2021, !dbg !14
  %2030 = bitcast float %2029 to i32, !dbg !14
  %2031 = call i32 @llvm.amdgcn.ds.bpermute(i32 %995, i32 %2030), !dbg !14
  %2032 = bitcast i32 %2031 to float, !dbg !14
  %2033 = select i1 %.not1, float %2021, float %2032, !dbg !14
  %2034 = select i1 %.not1, float %2032, float %2027, !dbg !14
  %2035 = select i1 %.not1, float %2028, float %2022, !dbg !14
  %2036 = bitcast float %2035 to i32, !dbg !14
  %2037 = call i32 @llvm.amdgcn.ds.bpermute(i32 %995, i32 %2036), !dbg !14
  %2038 = bitcast i32 %2037 to float, !dbg !14
  %2039 = select i1 %.not1, float %2022, float %2038, !dbg !14
  %2040 = select i1 %.not1, float %2038, float %2028, !dbg !14
  %2041 = extractvalue [2 x <8 x float>] %211, 1, !dbg !35
  %2042 = shufflevector <8 x float> %2041, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %2043 = fmul <4 x float> %2042, %1074, !dbg !35
  %2044 = insertelement <4 x float> poison, float %2033, i64 0, !dbg !36
  %2045 = insertelement <4 x float> %2044, float %2039, i64 1, !dbg !36
  %2046 = insertelement <4 x float> %2045, float %2034, i64 2, !dbg !36
  %2047 = insertelement <4 x float> %2046, float %2040, i64 3, !dbg !36
  %2048 = fadd <4 x float> %2043, %2047, !dbg !36
  %2049 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1952, <4 x float> zeroinitializer) #9, !dbg !14
  %2050 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1954, <4 x float> %2049) #9, !dbg !14
  %2051 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1956, <4 x float> %2050) #9, !dbg !14
  %2052 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1958, <4 x float> %2051) #9, !dbg !14
  %2053 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %1960, <4 x float> %2052) #9, !dbg !14
  %2054 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %1962, <4 x float> %2053) #9, !dbg !14
  %2055 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %1964, <4 x float> %2054) #9, !dbg !14
  %2056 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %1966, <4 x float> %2055) #9, !dbg !14
  %2057 = extractelement <4 x float> %2056, i64 0, !dbg !14
  %2058 = extractelement <4 x float> %2056, i64 1, !dbg !14
  %2059 = extractelement <4 x float> %2056, i64 2, !dbg !14
  %2060 = extractelement <4 x float> %2056, i64 3, !dbg !14
  %2061 = select i1 %.not, float %2058, float %2057, !dbg !14
  %2062 = bitcast float %2061 to i32, !dbg !14
  %2063 = call i32 @llvm.amdgcn.ds.bpermute(i32 %987, i32 %2062), !dbg !14
  %2064 = bitcast i32 %2063 to float, !dbg !14
  %2065 = select i1 %.not, float %2057, float %2064, !dbg !14
  %2066 = select i1 %.not, float %2064, float %2058, !dbg !14
  %2067 = select i1 %.not, float %2060, float %2059, !dbg !14
  %2068 = bitcast float %2067 to i32, !dbg !14
  %2069 = call i32 @llvm.amdgcn.ds.bpermute(i32 %987, i32 %2068), !dbg !14
  %2070 = bitcast i32 %2069 to float, !dbg !14
  %2071 = select i1 %.not, float %2059, float %2070, !dbg !14
  %2072 = select i1 %.not, float %2070, float %2060, !dbg !14
  %2073 = select i1 %.not1, float %2071, float %2065, !dbg !14
  %2074 = bitcast float %2073 to i32, !dbg !14
  %2075 = call i32 @llvm.amdgcn.ds.bpermute(i32 %995, i32 %2074), !dbg !14
  %2076 = bitcast i32 %2075 to float, !dbg !14
  %2077 = select i1 %.not1, float %2065, float %2076, !dbg !14
  %2078 = select i1 %.not1, float %2076, float %2071, !dbg !14
  %2079 = select i1 %.not1, float %2072, float %2066, !dbg !14
  %2080 = bitcast float %2079 to i32, !dbg !14
  %2081 = call i32 @llvm.amdgcn.ds.bpermute(i32 %995, i32 %2080), !dbg !14
  %2082 = bitcast i32 %2081 to float, !dbg !14
  %2083 = select i1 %.not1, float %2066, float %2082, !dbg !14
  %2084 = select i1 %.not1, float %2082, float %2072, !dbg !14
  %2085 = shufflevector <8 x float> %2041, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %2086 = fmul <4 x float> %2085, %1074, !dbg !35
  %2087 = insertelement <4 x float> poison, float %2077, i64 0, !dbg !36
  %2088 = insertelement <4 x float> %2087, float %2083, i64 1, !dbg !36
  %2089 = insertelement <4 x float> %2088, float %2078, i64 2, !dbg !36
  %2090 = insertelement <4 x float> %2089, float %2084, i64 3, !dbg !36
  %2091 = fadd <4 x float> %2086, %2090, !dbg !36
  %2092 = shufflevector <4 x float> %2048, <4 x float> %2091, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %2093 = insertvalue [2 x <8 x float>] %2004, <8 x float> %2092, 1, !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %2094 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local871, i32 0) #0, !dbg !14
  %2095 = bitcast <4 x i32> %2094 to <8 x half>, !dbg !14
  %2096 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local884, i32 0) #0, !dbg !14
  %2097 = bitcast <4 x i32> %2096 to <8 x half>, !dbg !14
  %2098 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local888, i32 0) #0, !dbg !14
  %2099 = bitcast <4 x i32> %2098 to <8 x half>, !dbg !14
  %2100 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local892, i32 0) #0, !dbg !14
  %2101 = bitcast <4 x i32> %2100 to <8 x half>, !dbg !14
  %2102 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local894, i32 0) #0, !dbg !14
  %2103 = bitcast <4 x i32> %2102 to <8 x half>, !dbg !14
  %2104 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local898, i32 0) #0, !dbg !14
  %2105 = bitcast <4 x i32> %2104 to <8 x half>, !dbg !14
  %2106 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local902, i32 0) #0, !dbg !14
  %2107 = bitcast <4 x i32> %2106 to <8 x half>, !dbg !14
  %2108 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local906, i32 0) #0, !dbg !14
  %2109 = bitcast <4 x i32> %2108 to <8 x half>, !dbg !14
  %2110 = shufflevector <8 x half> %2095, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %2111 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %2110, <4 x float> zeroinitializer) #9, !dbg !14
  %2112 = shufflevector <8 x half> %2097, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %2113 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %2112, <4 x float> %2111) #9, !dbg !14
  %2114 = shufflevector <8 x half> %2099, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %2115 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %2114, <4 x float> %2113) #9, !dbg !14
  %2116 = shufflevector <8 x half> %2101, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %2117 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %2116, <4 x float> %2115) #9, !dbg !14
  %2118 = shufflevector <8 x half> %2103, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %2119 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %2118, <4 x float> %2117) #9, !dbg !14
  %2120 = shufflevector <8 x half> %2105, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %2121 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %2120, <4 x float> %2119) #9, !dbg !14
  %2122 = shufflevector <8 x half> %2107, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %2123 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %2122, <4 x float> %2121) #9, !dbg !14
  %2124 = shufflevector <8 x half> %2109, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %2125 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %2124, <4 x float> %2123) #9, !dbg !14
  %2126 = extractelement <4 x float> %2125, i64 0, !dbg !14
  %2127 = extractelement <4 x float> %2125, i64 1, !dbg !14
  %2128 = extractelement <4 x float> %2125, i64 2, !dbg !14
  %2129 = extractelement <4 x float> %2125, i64 3, !dbg !14
  %2130 = select i1 %.not, float %2127, float %2126, !dbg !14
  %2131 = bitcast float %2130 to i32, !dbg !14
  %2132 = call i32 @llvm.amdgcn.ds.bpermute(i32 %987, i32 %2131), !dbg !14
  %2133 = bitcast i32 %2132 to float, !dbg !14
  %2134 = select i1 %.not, float %2126, float %2133, !dbg !14
  %2135 = select i1 %.not, float %2133, float %2127, !dbg !14
  %2136 = select i1 %.not, float %2129, float %2128, !dbg !14
  %2137 = bitcast float %2136 to i32, !dbg !14
  %2138 = call i32 @llvm.amdgcn.ds.bpermute(i32 %987, i32 %2137), !dbg !14
  %2139 = bitcast i32 %2138 to float, !dbg !14
  %2140 = select i1 %.not, float %2128, float %2139, !dbg !14
  %2141 = select i1 %.not, float %2139, float %2129, !dbg !14
  %2142 = select i1 %.not1, float %2140, float %2134, !dbg !14
  %2143 = bitcast float %2142 to i32, !dbg !14
  %2144 = call i32 @llvm.amdgcn.ds.bpermute(i32 %995, i32 %2143), !dbg !14
  %2145 = bitcast i32 %2144 to float, !dbg !14
  %2146 = select i1 %.not1, float %2134, float %2145, !dbg !14
  %2147 = select i1 %.not1, float %2145, float %2140, !dbg !14
  %2148 = select i1 %.not1, float %2141, float %2135, !dbg !14
  %2149 = bitcast float %2148 to i32, !dbg !14
  %2150 = call i32 @llvm.amdgcn.ds.bpermute(i32 %995, i32 %2149), !dbg !14
  %2151 = bitcast i32 %2150 to float, !dbg !14
  %2152 = select i1 %.not1, float %2135, float %2151, !dbg !14
  %2153 = select i1 %.not1, float %2151, float %2141, !dbg !14
  %2154 = extractvalue [2 x <8 x float>] %212, 0, !dbg !35
  %2155 = shufflevector <8 x float> %2154, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %2156 = fmul <4 x float> %2155, %1066, !dbg !35
  %2157 = insertelement <4 x float> poison, float %2146, i64 0, !dbg !36
  %2158 = insertelement <4 x float> %2157, float %2152, i64 1, !dbg !36
  %2159 = insertelement <4 x float> %2158, float %2147, i64 2, !dbg !36
  %2160 = insertelement <4 x float> %2159, float %2153, i64 3, !dbg !36
  %2161 = fadd <4 x float> %2156, %2160, !dbg !36
  %2162 = shufflevector <8 x half> %2095, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %2163 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %2162, <4 x float> zeroinitializer) #9, !dbg !14
  %2164 = shufflevector <8 x half> %2097, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %2165 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %2164, <4 x float> %2163) #9, !dbg !14
  %2166 = shufflevector <8 x half> %2099, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %2167 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %2166, <4 x float> %2165) #9, !dbg !14
  %2168 = shufflevector <8 x half> %2101, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %2169 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %2168, <4 x float> %2167) #9, !dbg !14
  %2170 = shufflevector <8 x half> %2103, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %2171 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %2170, <4 x float> %2169) #9, !dbg !14
  %2172 = shufflevector <8 x half> %2105, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %2173 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %2172, <4 x float> %2171) #9, !dbg !14
  %2174 = shufflevector <8 x half> %2107, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %2175 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %2174, <4 x float> %2173) #9, !dbg !14
  %2176 = shufflevector <8 x half> %2109, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %2177 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %2176, <4 x float> %2175) #9, !dbg !14
  %2178 = extractelement <4 x float> %2177, i64 0, !dbg !14
  %2179 = extractelement <4 x float> %2177, i64 1, !dbg !14
  %2180 = extractelement <4 x float> %2177, i64 2, !dbg !14
  %2181 = extractelement <4 x float> %2177, i64 3, !dbg !14
  %2182 = select i1 %.not, float %2179, float %2178, !dbg !14
  %2183 = bitcast float %2182 to i32, !dbg !14
  %2184 = call i32 @llvm.amdgcn.ds.bpermute(i32 %987, i32 %2183), !dbg !14
  %2185 = bitcast i32 %2184 to float, !dbg !14
  %2186 = select i1 %.not, float %2178, float %2185, !dbg !14
  %2187 = select i1 %.not, float %2185, float %2179, !dbg !14
  %2188 = select i1 %.not, float %2181, float %2180, !dbg !14
  %2189 = bitcast float %2188 to i32, !dbg !14
  %2190 = call i32 @llvm.amdgcn.ds.bpermute(i32 %987, i32 %2189), !dbg !14
  %2191 = bitcast i32 %2190 to float, !dbg !14
  %2192 = select i1 %.not, float %2180, float %2191, !dbg !14
  %2193 = select i1 %.not, float %2191, float %2181, !dbg !14
  %2194 = select i1 %.not1, float %2192, float %2186, !dbg !14
  %2195 = bitcast float %2194 to i32, !dbg !14
  %2196 = call i32 @llvm.amdgcn.ds.bpermute(i32 %995, i32 %2195), !dbg !14
  %2197 = bitcast i32 %2196 to float, !dbg !14
  %2198 = select i1 %.not1, float %2186, float %2197, !dbg !14
  %2199 = select i1 %.not1, float %2197, float %2192, !dbg !14
  %2200 = select i1 %.not1, float %2193, float %2187, !dbg !14
  %2201 = bitcast float %2200 to i32, !dbg !14
  %2202 = call i32 @llvm.amdgcn.ds.bpermute(i32 %995, i32 %2201), !dbg !14
  %2203 = bitcast i32 %2202 to float, !dbg !14
  %2204 = select i1 %.not1, float %2187, float %2203, !dbg !14
  %2205 = select i1 %.not1, float %2203, float %2193, !dbg !14
  %2206 = shufflevector <8 x float> %2154, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %2207 = fmul <4 x float> %2206, %1066, !dbg !35
  %2208 = insertelement <4 x float> poison, float %2198, i64 0, !dbg !36
  %2209 = insertelement <4 x float> %2208, float %2204, i64 1, !dbg !36
  %2210 = insertelement <4 x float> %2209, float %2199, i64 2, !dbg !36
  %2211 = insertelement <4 x float> %2210, float %2205, i64 3, !dbg !36
  %2212 = fadd <4 x float> %2207, %2211, !dbg !36
  %2213 = shufflevector <4 x float> %2161, <4 x float> %2212, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %2214 = insertvalue [2 x <8 x float>] poison, <8 x float> %2213, 0, !dbg !12
  %2215 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %2110, <4 x float> zeroinitializer) #9, !dbg !14
  %2216 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %2112, <4 x float> %2215) #9, !dbg !14
  %2217 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %2114, <4 x float> %2216) #9, !dbg !14
  %2218 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %2116, <4 x float> %2217) #9, !dbg !14
  %2219 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %2118, <4 x float> %2218) #9, !dbg !14
  %2220 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %2120, <4 x float> %2219) #9, !dbg !14
  %2221 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %2122, <4 x float> %2220) #9, !dbg !14
  %2222 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %2124, <4 x float> %2221) #9, !dbg !14
  %2223 = extractelement <4 x float> %2222, i64 0, !dbg !14
  %2224 = extractelement <4 x float> %2222, i64 1, !dbg !14
  %2225 = extractelement <4 x float> %2222, i64 2, !dbg !14
  %2226 = extractelement <4 x float> %2222, i64 3, !dbg !14
  %2227 = select i1 %.not, float %2224, float %2223, !dbg !14
  %2228 = bitcast float %2227 to i32, !dbg !14
  %2229 = call i32 @llvm.amdgcn.ds.bpermute(i32 %987, i32 %2228), !dbg !14
  %2230 = bitcast i32 %2229 to float, !dbg !14
  %2231 = select i1 %.not, float %2223, float %2230, !dbg !14
  %2232 = select i1 %.not, float %2230, float %2224, !dbg !14
  %2233 = select i1 %.not, float %2226, float %2225, !dbg !14
  %2234 = bitcast float %2233 to i32, !dbg !14
  %2235 = call i32 @llvm.amdgcn.ds.bpermute(i32 %987, i32 %2234), !dbg !14
  %2236 = bitcast i32 %2235 to float, !dbg !14
  %2237 = select i1 %.not, float %2225, float %2236, !dbg !14
  %2238 = select i1 %.not, float %2236, float %2226, !dbg !14
  %2239 = select i1 %.not1, float %2237, float %2231, !dbg !14
  %2240 = bitcast float %2239 to i32, !dbg !14
  %2241 = call i32 @llvm.amdgcn.ds.bpermute(i32 %995, i32 %2240), !dbg !14
  %2242 = bitcast i32 %2241 to float, !dbg !14
  %2243 = select i1 %.not1, float %2231, float %2242, !dbg !14
  %2244 = select i1 %.not1, float %2242, float %2237, !dbg !14
  %2245 = select i1 %.not1, float %2238, float %2232, !dbg !14
  %2246 = bitcast float %2245 to i32, !dbg !14
  %2247 = call i32 @llvm.amdgcn.ds.bpermute(i32 %995, i32 %2246), !dbg !14
  %2248 = bitcast i32 %2247 to float, !dbg !14
  %2249 = select i1 %.not1, float %2232, float %2248, !dbg !14
  %2250 = select i1 %.not1, float %2248, float %2238, !dbg !14
  %2251 = extractvalue [2 x <8 x float>] %212, 1, !dbg !35
  %2252 = shufflevector <8 x float> %2251, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %2253 = fmul <4 x float> %2252, %1074, !dbg !35
  %2254 = insertelement <4 x float> poison, float %2243, i64 0, !dbg !36
  %2255 = insertelement <4 x float> %2254, float %2249, i64 1, !dbg !36
  %2256 = insertelement <4 x float> %2255, float %2244, i64 2, !dbg !36
  %2257 = insertelement <4 x float> %2256, float %2250, i64 3, !dbg !36
  %2258 = fadd <4 x float> %2253, %2257, !dbg !36
  %2259 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %2162, <4 x float> zeroinitializer) #9, !dbg !14
  %2260 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %2164, <4 x float> %2259) #9, !dbg !14
  %2261 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %2166, <4 x float> %2260) #9, !dbg !14
  %2262 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %2168, <4 x float> %2261) #9, !dbg !14
  %2263 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %2170, <4 x float> %2262) #9, !dbg !14
  %2264 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %2172, <4 x float> %2263) #9, !dbg !14
  %2265 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %2174, <4 x float> %2264) #9, !dbg !14
  %2266 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %2176, <4 x float> %2265) #9, !dbg !14
  %2267 = extractelement <4 x float> %2266, i64 0, !dbg !14
  %2268 = extractelement <4 x float> %2266, i64 1, !dbg !14
  %2269 = extractelement <4 x float> %2266, i64 2, !dbg !14
  %2270 = extractelement <4 x float> %2266, i64 3, !dbg !14
  %2271 = select i1 %.not, float %2268, float %2267, !dbg !14
  %2272 = bitcast float %2271 to i32, !dbg !14
  %2273 = call i32 @llvm.amdgcn.ds.bpermute(i32 %987, i32 %2272), !dbg !14
  %2274 = bitcast i32 %2273 to float, !dbg !14
  %2275 = select i1 %.not, float %2267, float %2274, !dbg !14
  %2276 = select i1 %.not, float %2274, float %2268, !dbg !14
  %2277 = select i1 %.not, float %2270, float %2269, !dbg !14
  %2278 = bitcast float %2277 to i32, !dbg !14
  %2279 = call i32 @llvm.amdgcn.ds.bpermute(i32 %987, i32 %2278), !dbg !14
  %2280 = bitcast i32 %2279 to float, !dbg !14
  %2281 = select i1 %.not, float %2269, float %2280, !dbg !14
  %2282 = select i1 %.not, float %2280, float %2270, !dbg !14
  %2283 = select i1 %.not1, float %2281, float %2275, !dbg !14
  %2284 = bitcast float %2283 to i32, !dbg !14
  %2285 = call i32 @llvm.amdgcn.ds.bpermute(i32 %995, i32 %2284), !dbg !14
  %2286 = bitcast i32 %2285 to float, !dbg !14
  %2287 = select i1 %.not1, float %2275, float %2286, !dbg !14
  %2288 = select i1 %.not1, float %2286, float %2281, !dbg !14
  %2289 = select i1 %.not1, float %2282, float %2276, !dbg !14
  %2290 = bitcast float %2289 to i32, !dbg !14
  %2291 = call i32 @llvm.amdgcn.ds.bpermute(i32 %995, i32 %2290), !dbg !14
  %2292 = bitcast i32 %2291 to float, !dbg !14
  %2293 = select i1 %.not1, float %2276, float %2292, !dbg !14
  %2294 = select i1 %.not1, float %2292, float %2282, !dbg !14
  %2295 = shufflevector <8 x float> %2251, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %2296 = fmul <4 x float> %2295, %1074, !dbg !35
  %2297 = insertelement <4 x float> poison, float %2287, i64 0, !dbg !36
  %2298 = insertelement <4 x float> %2297, float %2293, i64 1, !dbg !36
  %2299 = insertelement <4 x float> %2298, float %2288, i64 2, !dbg !36
  %2300 = insertelement <4 x float> %2299, float %2294, i64 3, !dbg !36
  %2301 = fadd <4 x float> %2296, %2300, !dbg !36
  %2302 = shufflevector <4 x float> %2258, <4 x float> %2301, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %2303 = insertvalue [2 x <8 x float>] %2214, <8 x float> %2302, 1, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %2304 = add nuw nsw i64 %207, 1, !dbg !12
  br label %206, !dbg !12

2305:                                             ; preds = %206
  %2306 = extractvalue [2 x <1 x float>] %209, 0, !dbg !19
  %2307 = shufflevector <1 x float> %2306, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %2308 = extractvalue [2 x <1 x float>] %209, 1, !dbg !19
  %2309 = shufflevector <1 x float> %2308, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  br i1 true, label %2310, label %2340, !dbg !21

2310:                                             ; preds = %2305
  %2311 = extractvalue [2 x <8 x float>] %208, 1, !dbg !19
  %2312 = shufflevector <8 x float> %2311, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !19
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %2309, !dbg !19
  %normalized = fmul <4 x float> %2312, %row.reciprocal, !dbg !19
  %2313 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !20
  %2314 = extractelement <4 x half> %2313, i64 3, !dbg !20
  %2315 = extractelement <4 x half> %2313, i64 2, !dbg !20
  %2316 = extractelement <4 x half> %2313, i64 1, !dbg !20
  %2317 = extractelement <4 x half> %2313, i64 0, !dbg !20
  %2318 = shufflevector <8 x float> %2311, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !19
  %normalized104 = fmul <4 x float> %2318, %row.reciprocal, !dbg !19
  %2319 = fptrunc <4 x float> %normalized104 to <4 x half>, !dbg !20
  %2320 = extractelement <4 x half> %2319, i64 3, !dbg !20
  %2321 = extractelement <4 x half> %2319, i64 2, !dbg !20
  %2322 = extractelement <4 x half> %2319, i64 1, !dbg !20
  %2323 = extractelement <4 x half> %2319, i64 0, !dbg !20
  %2324 = extractvalue [2 x <8 x float>] %208, 0, !dbg !19
  %2325 = shufflevector <8 x float> %2324, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !19
  %row.reciprocal105 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %2307, !dbg !19
  %normalized106 = fmul <4 x float> %2325, %row.reciprocal105, !dbg !19
  %2326 = fptrunc <4 x float> %normalized106 to <4 x half>, !dbg !20
  %2327 = extractelement <4 x half> %2326, i64 3, !dbg !20
  %2328 = extractelement <4 x half> %2326, i64 2, !dbg !20
  %2329 = extractelement <4 x half> %2326, i64 1, !dbg !20
  %2330 = extractelement <4 x half> %2326, i64 0, !dbg !20
  %2331 = shufflevector <8 x float> %2324, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !19
  %normalized107 = fmul <4 x float> %2331, %row.reciprocal105, !dbg !19
  %2332 = fptrunc <4 x float> %normalized107 to <4 x half>, !dbg !20
  %2333 = extractelement <4 x half> %2332, i64 3, !dbg !20
  %2334 = extractelement <4 x half> %2332, i64 2, !dbg !20
  %2335 = extractelement <4 x half> %2332, i64 1, !dbg !20
  %2336 = extractelement <4 x half> %2332, i64 0, !dbg !20
  %2337 = add nsw i64 %16, %22, !dbg !21
  %2338 = shl i64 %2337, 19, !dbg !21
  %2339 = and i64 %2338, 16252928, !dbg !21
  %address.thread934 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local935 = zext nneg i32 %address.thread934 to i64, !dbg !7
  %address.local936 = add i64 %25, %address.local935, !dbg !21
  %address.local937 = lshr i64 %address.local935, 6, !dbg !10
  %address.local938 = shl nuw nsw i64 %address.local937, 4, !dbg !21
  %address.local939 = add i64 %address.local936, %address.local938, !dbg !21
  %address.local940 = and i64 %address.local935, 240, !dbg !21
  %address.local941 = sub i64 %address.local939, %address.local940, !dbg !21
  %address.local942 = add i64 %address.local941, %26, !dbg !21
  %address.local943 = shl nuw nsw i64 %address.local942, 7, !dbg !21
  %address.local944 = add nuw nsw i64 %2339, %address.local943, !dbg !21
  %address.local945 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local944, !dbg !21
  %address.local946 = lshr i64 %address.local935, 2, !dbg !10
  %address.local947 = and i64 %address.local946, 12, !dbg !10
  %address.local948 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local947, !dbg !21
  store half %2336, ptr addrspace(1) %address.local948, align 2, !dbg !21
  %address.local949 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local944, !dbg !21
  %address.local950 = or disjoint i64 %address.local947, 1, !dbg !21
  %address.local951 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local949, i64 %address.local950, !dbg !21
  store half %2335, ptr addrspace(1) %address.local951, align 2, !dbg !21
  %address.local952 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local944, !dbg !21
  %address.local953 = or disjoint i64 %address.local947, 2, !dbg !21
  %address.local954 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local952, i64 %address.local953, !dbg !21
  store half %2334, ptr addrspace(1) %address.local954, align 2, !dbg !21
  %address.local955 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local944, !dbg !21
  %address.local956 = or disjoint i64 %address.local947, 3, !dbg !21
  %address.local957 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local955, i64 %address.local956, !dbg !21
  store half %2333, ptr addrspace(1) %address.local957, align 2, !dbg !21
  %address.local958 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local944, !dbg !21
  %address.local959 = or disjoint i64 %address.local947, 16, !dbg !21
  %address.local960 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local958, i64 %address.local959, !dbg !21
  store half %2330, ptr addrspace(1) %address.local960, align 2, !dbg !21
  %address.local961 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local944, !dbg !21
  %address.local962 = or disjoint i64 %address.local947, 17, !dbg !21
  %address.local963 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local961, i64 %address.local962, !dbg !21
  store half %2329, ptr addrspace(1) %address.local963, align 2, !dbg !21
  %address.local964 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local944, !dbg !21
  %address.local965 = or disjoint i64 %address.local947, 18, !dbg !21
  %address.local966 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local964, i64 %address.local965, !dbg !21
  store half %2328, ptr addrspace(1) %address.local966, align 2, !dbg !21
  %address.local967 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local944, !dbg !21
  %address.local968 = or disjoint i64 %address.local947, 19, !dbg !21
  %address.local969 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local967, i64 %address.local968, !dbg !21
  store half %2327, ptr addrspace(1) %address.local969, align 2, !dbg !21
  %address.local970 = shl i64 %address.local942, 7, !dbg !21
  %address.local971 = add i64 %address.local970, 8192, !dbg !21
  %address.local972 = add nuw nsw i64 %2339, %address.local971, !dbg !21
  %address.local973 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local972, !dbg !21
  %address.local974 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local947, !dbg !21
  store half %2323, ptr addrspace(1) %address.local974, align 2, !dbg !21
  %address.local975 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local972, !dbg !21
  %address.local976 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local975, i64 %address.local950, !dbg !21
  store half %2322, ptr addrspace(1) %address.local976, align 2, !dbg !21
  %address.local977 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local972, !dbg !21
  %address.local978 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local977, i64 %address.local953, !dbg !21
  store half %2321, ptr addrspace(1) %address.local978, align 2, !dbg !21
  %address.local979 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local972, !dbg !21
  %address.local980 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local979, i64 %address.local956, !dbg !21
  store half %2320, ptr addrspace(1) %address.local980, align 2, !dbg !21
  %address.local981 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local972, !dbg !21
  %address.local982 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local981, i64 %address.local959, !dbg !21
  store half %2317, ptr addrspace(1) %address.local982, align 2, !dbg !21
  %address.local983 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local972, !dbg !21
  %address.local984 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local983, i64 %address.local962, !dbg !21
  store half %2316, ptr addrspace(1) %address.local984, align 2, !dbg !21
  %address.local985 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local972, !dbg !21
  %address.local986 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local965, !dbg !21
  store half %2315, ptr addrspace(1) %address.local986, align 2, !dbg !21
  %address.local987 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local972, !dbg !21
  %address.local988 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local987, i64 %address.local968, !dbg !21
  store half %2314, ptr addrspace(1) %address.local988, align 2, !dbg !21
  br label %2340, !dbg !21

2340:                                             ; preds = %2310, %2305
  br i1 true, label %2341, label %2369, !dbg !21

2341:                                             ; preds = %2340
  %2342 = extractvalue [2 x <8 x float>] %210, 1, !dbg !19
  %2343 = shufflevector <8 x float> %2342, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !19
  %row.reciprocal108 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %2309, !dbg !19
  %normalized109 = fmul <4 x float> %2343, %row.reciprocal108, !dbg !19
  %2344 = fptrunc <4 x float> %normalized109 to <4 x half>, !dbg !20
  %2345 = extractelement <4 x half> %2344, i64 3, !dbg !20
  %2346 = extractelement <4 x half> %2344, i64 2, !dbg !20
  %2347 = extractelement <4 x half> %2344, i64 1, !dbg !20
  %2348 = extractelement <4 x half> %2344, i64 0, !dbg !20
  %2349 = shufflevector <8 x float> %2342, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !19
  %normalized110 = fmul <4 x float> %2349, %row.reciprocal108, !dbg !19
  %2350 = fptrunc <4 x float> %normalized110 to <4 x half>, !dbg !20
  %2351 = extractelement <4 x half> %2350, i64 3, !dbg !20
  %2352 = extractelement <4 x half> %2350, i64 2, !dbg !20
  %2353 = extractelement <4 x half> %2350, i64 1, !dbg !20
  %2354 = extractelement <4 x half> %2350, i64 0, !dbg !20
  %2355 = extractvalue [2 x <8 x float>] %210, 0, !dbg !19
  %2356 = shufflevector <8 x float> %2355, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !19
  %row.reciprocal111 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %2307, !dbg !19
  %normalized112 = fmul <4 x float> %2356, %row.reciprocal111, !dbg !19
  %2357 = fptrunc <4 x float> %normalized112 to <4 x half>, !dbg !20
  %2358 = extractelement <4 x half> %2357, i64 3, !dbg !20
  %2359 = extractelement <4 x half> %2357, i64 2, !dbg !20
  %2360 = extractelement <4 x half> %2357, i64 1, !dbg !20
  %2361 = extractelement <4 x half> %2357, i64 0, !dbg !20
  %2362 = shufflevector <8 x float> %2355, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !19
  %normalized113 = fmul <4 x float> %2362, %row.reciprocal111, !dbg !19
  %2363 = fptrunc <4 x float> %normalized113 to <4 x half>, !dbg !20
  %2364 = extractelement <4 x half> %2363, i64 3, !dbg !20
  %2365 = extractelement <4 x half> %2363, i64 2, !dbg !20
  %2366 = extractelement <4 x half> %2363, i64 1, !dbg !20
  %2367 = extractelement <4 x half> %2363, i64 0, !dbg !20
  %2368 = shl nuw nsw i64 %24, 19, !dbg !21
  %address.thread989 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local990 = zext nneg i32 %address.thread989 to i64, !dbg !7
  %address.local991 = add i64 %25, %address.local990, !dbg !21
  %address.local992 = lshr i64 %address.local990, 6, !dbg !10
  %address.local993 = shl nuw nsw i64 %address.local992, 4, !dbg !21
  %address.local994 = add i64 %address.local991, %address.local993, !dbg !21
  %address.local995 = and i64 %address.local990, 240, !dbg !21
  %address.local996 = sub i64 %address.local994, %address.local995, !dbg !21
  %address.local997 = add i64 %address.local996, %26, !dbg !21
  %address.local998 = shl nuw nsw i64 %address.local997, 7, !dbg !21
  %address.local999 = add nuw nsw i64 %2368, %address.local998, !dbg !21
  %address.local1000 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local999, !dbg !21
  %address.local1001 = lshr i64 %address.local990, 2, !dbg !10
  %address.local1002 = and i64 %address.local1001, 12, !dbg !10
  %address.local1003 = or disjoint i64 %address.local1002, 32, !dbg !21
  %address.local1004 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1000, i64 %address.local1003, !dbg !21
  store half %2367, ptr addrspace(1) %address.local1004, align 2, !dbg !21
  %address.local1005 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local999, !dbg !21
  %address.local1006 = or disjoint i64 %address.local1002, 33, !dbg !21
  %address.local1007 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1005, i64 %address.local1006, !dbg !21
  store half %2366, ptr addrspace(1) %address.local1007, align 2, !dbg !21
  %address.local1008 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local999, !dbg !21
  %address.local1009 = or disjoint i64 %address.local1002, 34, !dbg !21
  %address.local1010 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1008, i64 %address.local1009, !dbg !21
  store half %2365, ptr addrspace(1) %address.local1010, align 2, !dbg !21
  %address.local1011 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local999, !dbg !21
  %address.local1012 = or disjoint i64 %address.local1002, 35, !dbg !21
  %address.local1013 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1011, i64 %address.local1012, !dbg !21
  store half %2364, ptr addrspace(1) %address.local1013, align 2, !dbg !21
  %address.local1014 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local999, !dbg !21
  %address.local1015 = or disjoint i64 %address.local1002, 48, !dbg !21
  %address.local1016 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1014, i64 %address.local1015, !dbg !21
  store half %2361, ptr addrspace(1) %address.local1016, align 2, !dbg !21
  %address.local1017 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local999, !dbg !21
  %address.local1018 = or disjoint i64 %address.local1002, 49, !dbg !21
  %address.local1019 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1017, i64 %address.local1018, !dbg !21
  store half %2360, ptr addrspace(1) %address.local1019, align 2, !dbg !21
  %address.local1020 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local999, !dbg !21
  %address.local1021 = or disjoint i64 %address.local1002, 50, !dbg !21
  %address.local1022 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1020, i64 %address.local1021, !dbg !21
  store half %2359, ptr addrspace(1) %address.local1022, align 2, !dbg !21
  %address.local1023 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local999, !dbg !21
  %address.local1024 = or i64 %address.local1001, 51, !dbg !21
  %address.local1025 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1023, i64 %address.local1024, !dbg !21
  store half %2358, ptr addrspace(1) %address.local1025, align 2, !dbg !21
  %address.local1026 = shl i64 %address.local997, 7, !dbg !21
  %address.local1027 = add i64 %address.local1026, 8192, !dbg !21
  %address.local1028 = add nuw nsw i64 %2368, %address.local1027, !dbg !21
  %address.local1029 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1028, !dbg !21
  %address.local1030 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1029, i64 %address.local1003, !dbg !21
  store half %2354, ptr addrspace(1) %address.local1030, align 2, !dbg !21
  %address.local1031 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1028, !dbg !21
  %address.local1032 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1031, i64 %address.local1006, !dbg !21
  store half %2353, ptr addrspace(1) %address.local1032, align 2, !dbg !21
  %address.local1033 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1028, !dbg !21
  %address.local1034 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1033, i64 %address.local1009, !dbg !21
  store half %2352, ptr addrspace(1) %address.local1034, align 2, !dbg !21
  %address.local1035 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1028, !dbg !21
  %address.local1036 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1035, i64 %address.local1012, !dbg !21
  store half %2351, ptr addrspace(1) %address.local1036, align 2, !dbg !21
  %address.local1037 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1028, !dbg !21
  %address.local1038 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1037, i64 %address.local1015, !dbg !21
  store half %2348, ptr addrspace(1) %address.local1038, align 2, !dbg !21
  %address.local1039 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1028, !dbg !21
  %address.local1040 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1039, i64 %address.local1018, !dbg !21
  store half %2347, ptr addrspace(1) %address.local1040, align 2, !dbg !21
  %address.local1041 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1028, !dbg !21
  %address.local1042 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1041, i64 %address.local1021, !dbg !21
  store half %2346, ptr addrspace(1) %address.local1042, align 2, !dbg !21
  %address.local1043 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1028, !dbg !21
  %address.local1044 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1043, i64 %address.local1024, !dbg !21
  store half %2345, ptr addrspace(1) %address.local1044, align 2, !dbg !21
  br label %2369, !dbg !21

2369:                                             ; preds = %2341, %2340
  br i1 true, label %2370, label %2398, !dbg !21

2370:                                             ; preds = %2369
  %2371 = extractvalue [2 x <8 x float>] %211, 1, !dbg !19
  %2372 = shufflevector <8 x float> %2371, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !19
  %row.reciprocal114 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %2309, !dbg !19
  %normalized115 = fmul <4 x float> %2372, %row.reciprocal114, !dbg !19
  %2373 = fptrunc <4 x float> %normalized115 to <4 x half>, !dbg !20
  %2374 = extractelement <4 x half> %2373, i64 3, !dbg !20
  %2375 = extractelement <4 x half> %2373, i64 2, !dbg !20
  %2376 = extractelement <4 x half> %2373, i64 1, !dbg !20
  %2377 = extractelement <4 x half> %2373, i64 0, !dbg !20
  %2378 = shufflevector <8 x float> %2371, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !19
  %normalized116 = fmul <4 x float> %2378, %row.reciprocal114, !dbg !19
  %2379 = fptrunc <4 x float> %normalized116 to <4 x half>, !dbg !20
  %2380 = extractelement <4 x half> %2379, i64 3, !dbg !20
  %2381 = extractelement <4 x half> %2379, i64 2, !dbg !20
  %2382 = extractelement <4 x half> %2379, i64 1, !dbg !20
  %2383 = extractelement <4 x half> %2379, i64 0, !dbg !20
  %2384 = extractvalue [2 x <8 x float>] %211, 0, !dbg !19
  %2385 = shufflevector <8 x float> %2384, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !19
  %row.reciprocal117 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %2307, !dbg !19
  %normalized118 = fmul <4 x float> %2385, %row.reciprocal117, !dbg !19
  %2386 = fptrunc <4 x float> %normalized118 to <4 x half>, !dbg !20
  %2387 = extractelement <4 x half> %2386, i64 3, !dbg !20
  %2388 = extractelement <4 x half> %2386, i64 2, !dbg !20
  %2389 = extractelement <4 x half> %2386, i64 1, !dbg !20
  %2390 = extractelement <4 x half> %2386, i64 0, !dbg !20
  %2391 = shufflevector <8 x float> %2384, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !19
  %normalized119 = fmul <4 x float> %2391, %row.reciprocal117, !dbg !19
  %2392 = fptrunc <4 x float> %normalized119 to <4 x half>, !dbg !20
  %2393 = extractelement <4 x half> %2392, i64 3, !dbg !20
  %2394 = extractelement <4 x half> %2392, i64 2, !dbg !20
  %2395 = extractelement <4 x half> %2392, i64 1, !dbg !20
  %2396 = extractelement <4 x half> %2392, i64 0, !dbg !20
  %2397 = shl nuw nsw i64 %24, 19, !dbg !21
  %address.thread1045 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local1046 = zext nneg i32 %address.thread1045 to i64, !dbg !7
  %address.local1047 = add i64 %25, %address.local1046, !dbg !21
  %address.local1048 = lshr i64 %address.local1046, 6, !dbg !10
  %address.local1049 = shl nuw nsw i64 %address.local1048, 4, !dbg !21
  %address.local1050 = add i64 %address.local1047, %address.local1049, !dbg !21
  %address.local1051 = and i64 %address.local1046, 240, !dbg !21
  %address.local1052 = sub i64 %address.local1050, %address.local1051, !dbg !21
  %address.local1053 = add i64 %address.local1052, %26, !dbg !21
  %address.local1054 = shl nuw nsw i64 %address.local1053, 7, !dbg !21
  %address.local1055 = add nuw nsw i64 %2397, %address.local1054, !dbg !21
  %address.local1056 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1055, !dbg !21
  %address.local1057 = lshr i64 %address.local1046, 2, !dbg !10
  %address.local1058 = and i64 %address.local1057, 12, !dbg !10
  %address.local1059 = or disjoint i64 %address.local1058, 64, !dbg !21
  %address.local1060 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1056, i64 %address.local1059, !dbg !21
  store half %2396, ptr addrspace(1) %address.local1060, align 2, !dbg !21
  %address.local1061 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1055, !dbg !21
  %address.local1062 = or disjoint i64 %address.local1058, 65, !dbg !21
  %address.local1063 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1061, i64 %address.local1062, !dbg !21
  store half %2395, ptr addrspace(1) %address.local1063, align 2, !dbg !21
  %address.local1064 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1055, !dbg !21
  %address.local1065 = or disjoint i64 %address.local1058, 66, !dbg !21
  %address.local1066 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1064, i64 %address.local1065, !dbg !21
  store half %2394, ptr addrspace(1) %address.local1066, align 2, !dbg !21
  %address.local1067 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1055, !dbg !21
  %address.local1068 = or disjoint i64 %address.local1058, 67, !dbg !21
  %address.local1069 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1067, i64 %address.local1068, !dbg !21
  store half %2393, ptr addrspace(1) %address.local1069, align 2, !dbg !21
  %address.local1070 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1055, !dbg !21
  %address.local1071 = or disjoint i64 %address.local1058, 80, !dbg !21
  %address.local1072 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1070, i64 %address.local1071, !dbg !21
  store half %2390, ptr addrspace(1) %address.local1072, align 2, !dbg !21
  %address.local1073 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1055, !dbg !21
  %address.local1074 = or disjoint i64 %address.local1058, 81, !dbg !21
  %address.local1075 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1073, i64 %address.local1074, !dbg !21
  store half %2389, ptr addrspace(1) %address.local1075, align 2, !dbg !21
  %address.local1076 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1055, !dbg !21
  %address.local1077 = or disjoint i64 %address.local1058, 82, !dbg !21
  %address.local1078 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1076, i64 %address.local1077, !dbg !21
  store half %2388, ptr addrspace(1) %address.local1078, align 2, !dbg !21
  %address.local1079 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1055, !dbg !21
  %address.local1080 = or disjoint i64 %address.local1058, 83, !dbg !21
  %address.local1081 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1079, i64 %address.local1080, !dbg !21
  store half %2387, ptr addrspace(1) %address.local1081, align 2, !dbg !21
  %address.local1082 = shl i64 %address.local1053, 7, !dbg !21
  %address.local1083 = add i64 %address.local1082, 8192, !dbg !21
  %address.local1084 = add nuw nsw i64 %2397, %address.local1083, !dbg !21
  %address.local1085 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1084, !dbg !21
  %address.local1086 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1085, i64 %address.local1059, !dbg !21
  store half %2383, ptr addrspace(1) %address.local1086, align 2, !dbg !21
  %address.local1087 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1084, !dbg !21
  %address.local1088 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1087, i64 %address.local1062, !dbg !21
  store half %2382, ptr addrspace(1) %address.local1088, align 2, !dbg !21
  %address.local1089 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1084, !dbg !21
  %address.local1090 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1089, i64 %address.local1065, !dbg !21
  store half %2381, ptr addrspace(1) %address.local1090, align 2, !dbg !21
  %address.local1091 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1084, !dbg !21
  %address.local1092 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1091, i64 %address.local1068, !dbg !21
  store half %2380, ptr addrspace(1) %address.local1092, align 2, !dbg !21
  %address.local1093 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1084, !dbg !21
  %address.local1094 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1093, i64 %address.local1071, !dbg !21
  store half %2377, ptr addrspace(1) %address.local1094, align 2, !dbg !21
  %address.local1095 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1084, !dbg !21
  %address.local1096 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1095, i64 %address.local1074, !dbg !21
  store half %2376, ptr addrspace(1) %address.local1096, align 2, !dbg !21
  %address.local1097 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1084, !dbg !21
  %address.local1098 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1097, i64 %address.local1077, !dbg !21
  store half %2375, ptr addrspace(1) %address.local1098, align 2, !dbg !21
  %address.local1099 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1084, !dbg !21
  %address.local1100 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1099, i64 %address.local1080, !dbg !21
  store half %2374, ptr addrspace(1) %address.local1100, align 2, !dbg !21
  br label %2398, !dbg !21

2398:                                             ; preds = %2370, %2369
  br i1 true, label %2399, label %2427, !dbg !21

2399:                                             ; preds = %2398
  %2400 = extractvalue [2 x <8 x float>] %212, 1, !dbg !19
  %2401 = shufflevector <8 x float> %2400, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !19
  %row.reciprocal120 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %2309, !dbg !19
  %normalized121 = fmul <4 x float> %2401, %row.reciprocal120, !dbg !19
  %2402 = fptrunc <4 x float> %normalized121 to <4 x half>, !dbg !20
  %2403 = extractelement <4 x half> %2402, i64 3, !dbg !20
  %2404 = extractelement <4 x half> %2402, i64 2, !dbg !20
  %2405 = extractelement <4 x half> %2402, i64 1, !dbg !20
  %2406 = extractelement <4 x half> %2402, i64 0, !dbg !20
  %2407 = shufflevector <8 x float> %2400, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !19
  %normalized122 = fmul <4 x float> %2407, %row.reciprocal120, !dbg !19
  %2408 = fptrunc <4 x float> %normalized122 to <4 x half>, !dbg !20
  %2409 = extractelement <4 x half> %2408, i64 3, !dbg !20
  %2410 = extractelement <4 x half> %2408, i64 2, !dbg !20
  %2411 = extractelement <4 x half> %2408, i64 1, !dbg !20
  %2412 = extractelement <4 x half> %2408, i64 0, !dbg !20
  %2413 = extractvalue [2 x <8 x float>] %212, 0, !dbg !19
  %2414 = shufflevector <8 x float> %2413, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !19
  %row.reciprocal123 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %2307, !dbg !19
  %normalized124 = fmul <4 x float> %2414, %row.reciprocal123, !dbg !19
  %2415 = fptrunc <4 x float> %normalized124 to <4 x half>, !dbg !20
  %2416 = extractelement <4 x half> %2415, i64 3, !dbg !20
  %2417 = extractelement <4 x half> %2415, i64 2, !dbg !20
  %2418 = extractelement <4 x half> %2415, i64 1, !dbg !20
  %2419 = extractelement <4 x half> %2415, i64 0, !dbg !20
  %2420 = shufflevector <8 x float> %2413, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !19
  %normalized125 = fmul <4 x float> %2420, %row.reciprocal123, !dbg !19
  %2421 = fptrunc <4 x float> %normalized125 to <4 x half>, !dbg !20
  %2422 = extractelement <4 x half> %2421, i64 3, !dbg !20
  %2423 = extractelement <4 x half> %2421, i64 2, !dbg !20
  %2424 = extractelement <4 x half> %2421, i64 1, !dbg !20
  %2425 = extractelement <4 x half> %2421, i64 0, !dbg !20
  %2426 = shl nuw nsw i64 %24, 19, !dbg !21
  %address.thread1101 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local1102 = zext nneg i32 %address.thread1101 to i64, !dbg !7
  %address.local1103 = add i64 %25, %address.local1102, !dbg !21
  %address.local1104 = lshr i64 %address.local1102, 6, !dbg !10
  %address.local1105 = shl nuw nsw i64 %address.local1104, 4, !dbg !21
  %address.local1106 = add i64 %address.local1103, %address.local1105, !dbg !21
  %address.local1107 = and i64 %address.local1102, 240, !dbg !21
  %address.local1108 = sub i64 %address.local1106, %address.local1107, !dbg !21
  %address.local1109 = add i64 %address.local1108, %26, !dbg !21
  %address.local1110 = shl nuw nsw i64 %address.local1109, 7, !dbg !21
  %address.local1111 = add nuw nsw i64 %2426, %address.local1110, !dbg !21
  %address.local1112 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1111, !dbg !21
  %address.local1113 = lshr i64 %address.local1102, 2, !dbg !10
  %address.local1114 = and i64 %address.local1113, 12, !dbg !10
  %address.local1115 = or disjoint i64 %address.local1114, 96, !dbg !21
  %address.local1116 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1112, i64 %address.local1115, !dbg !21
  store half %2425, ptr addrspace(1) %address.local1116, align 2, !dbg !21
  %address.local1117 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1111, !dbg !21
  %address.local1118 = or disjoint i64 %address.local1114, 97, !dbg !21
  %address.local1119 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1117, i64 %address.local1118, !dbg !21
  store half %2424, ptr addrspace(1) %address.local1119, align 2, !dbg !21
  %address.local1120 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1111, !dbg !21
  %address.local1121 = or disjoint i64 %address.local1114, 98, !dbg !21
  %address.local1122 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1120, i64 %address.local1121, !dbg !21
  store half %2423, ptr addrspace(1) %address.local1122, align 2, !dbg !21
  %address.local1123 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1111, !dbg !21
  %address.local1124 = or disjoint i64 %address.local1114, 99, !dbg !21
  %address.local1125 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1123, i64 %address.local1124, !dbg !21
  store half %2422, ptr addrspace(1) %address.local1125, align 2, !dbg !21
  %address.local1126 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1111, !dbg !21
  %address.local1127 = or disjoint i64 %address.local1114, 112, !dbg !21
  %address.local1128 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1126, i64 %address.local1127, !dbg !21
  store half %2419, ptr addrspace(1) %address.local1128, align 2, !dbg !21
  %address.local1129 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1111, !dbg !21
  %address.local1130 = or disjoint i64 %address.local1114, 113, !dbg !21
  %address.local1131 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1129, i64 %address.local1130, !dbg !21
  store half %2418, ptr addrspace(1) %address.local1131, align 2, !dbg !21
  %address.local1132 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1111, !dbg !21
  %address.local1133 = or disjoint i64 %address.local1114, 114, !dbg !21
  %address.local1134 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1132, i64 %address.local1133, !dbg !21
  store half %2417, ptr addrspace(1) %address.local1134, align 2, !dbg !21
  %address.local1135 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1111, !dbg !21
  %address.local1136 = or i64 %address.local1113, 115, !dbg !21
  %address.local1137 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1135, i64 %address.local1136, !dbg !21
  store half %2416, ptr addrspace(1) %address.local1137, align 2, !dbg !21
  %address.local1138 = shl i64 %address.local1109, 7, !dbg !21
  %address.local1139 = add i64 %address.local1138, 8192, !dbg !21
  %address.local1140 = add nuw nsw i64 %2426, %address.local1139, !dbg !21
  %address.local1141 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !21
  %address.local1142 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1141, i64 %address.local1115, !dbg !21
  store half %2412, ptr addrspace(1) %address.local1142, align 2, !dbg !21
  %address.local1143 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !21
  %address.local1144 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1143, i64 %address.local1118, !dbg !21
  store half %2411, ptr addrspace(1) %address.local1144, align 2, !dbg !21
  %address.local1145 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !21
  %address.local1146 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1145, i64 %address.local1121, !dbg !21
  store half %2410, ptr addrspace(1) %address.local1146, align 2, !dbg !21
  %address.local1147 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !21
  %address.local1148 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1147, i64 %address.local1124, !dbg !21
  store half %2409, ptr addrspace(1) %address.local1148, align 2, !dbg !21
  %address.local1149 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !21
  %address.local1150 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1149, i64 %address.local1127, !dbg !21
  store half %2406, ptr addrspace(1) %address.local1150, align 2, !dbg !21
  %address.local1151 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !21
  %address.local1152 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1151, i64 %address.local1130, !dbg !21
  store half %2405, ptr addrspace(1) %address.local1152, align 2, !dbg !21
  %address.local1153 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !21
  %address.local1154 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1153, i64 %address.local1133, !dbg !21
  store half %2404, ptr addrspace(1) %address.local1154, align 2, !dbg !21
  %address.local1155 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !21
  %address.local1156 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1155, i64 %address.local1136, !dbg !21
  store half %2403, ptr addrspace(1) %address.local1156, align 2, !dbg !21
  br label %2427, !dbg !21

2427:                                             ; preds = %2399, %2398
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

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maxnum.f32(float, float) #6

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32) #7

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.hi(i32, i32) #7

; Function Attrs: convergent nocallback nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.ds.bpermute(i32, i32) #8

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare <1 x float> @llvm.maxnum.v1f32(<1 x float>, <1 x float>) #6

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.amdgcn.exp2.f32(float) #6

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.sched.barrier(i32 immarg) #5

attributes #0 = { convergent willreturn memory(read) }
attributes #1 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1" "frisk.frag_prefetch"="4" "frisk.global_prefetch"="8" "frisk.register_comparisons"="1" "frisk.register_lifetimes"="1" "uniform-work-group-size" }
attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none) }
attributes #4 = { nocallback nofree nounwind willreturn }
attributes #5 = { convergent nocallback nofree nounwind willreturn }
attributes #6 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #7 = { nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }
attributes #8 = { convergent nocallback nofree nounwind willreturn memory(none) }
attributes #9 = { convergent nounwind willreturn memory(none) }
attributes #10 = { convergent nounwind }
attributes #11 = { memory(inaccessiblemem: readwrite) }

!llvm.dbg.cu = !{!3}
!llvm.module.flags = !{!38}

!0 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/online_softmax/recheck/run-20261023-172221-hqi6mws2/retained32-baseline")
!1 = !{}
!2 = !DISubroutineType(types: !1)
!3 = distinct !DICompileUnit(language: DW_LANG_C, file: !0, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!4 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !0, file: !0, line: 23, type: !2, scopeLine: 23, spFlags: DISPFlagDefinition, unit: !3)
!5 = !DILocation(line: 23, column: 3, scope: !4)
!7 = !DILocation(line: 70, column: 22, scope: !4)
!9 = !DILocation(line: 48, column: 17, scope: !4)
!10 = !DILocation(line: 72, column: 24, scope: !4)
!12 = !DILocation(line: 60, column: 24, scope: !4)
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
