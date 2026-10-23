; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/online_softmax/final/run-20261023-171724-w2j94qwt/retained32-baseline/input.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/online_softmax/final/run-20261023-171724-w2j94qwt/retained32-baseline/input.mlir"

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
  br label %206, !dbg !7

206:                                              ; preds = %850, %4
  %207 = phi i64 [ %2309, %850 ], [ 0, %4 ], !dbg !7
  %208 = phi [2 x <8 x float>] [ %1616, %850 ], [ zeroinitializer, %4 ], !dbg !7
  %209 = phi [2 x <1 x float>] [ %1372, %850 ], [ zeroinitializer, %4 ], !dbg !7
  %210 = phi [2 x <1 x float>] [ %2308, %850 ], [ [<1 x float> splat (float 0xFFF0000000000000), <1 x float> splat (float 0xFFF0000000000000)], %4 ], !dbg !7
  %211 = phi [2 x <8 x float>] [ %1856, %850 ], [ zeroinitializer, %4 ], !dbg !7
  %212 = phi [2 x <8 x float>] [ %2096, %850 ], [ zeroinitializer, %4 ], !dbg !7
  %213 = phi [2 x <8 x float>] [ %2306, %850 ], [ zeroinitializer, %4 ], !dbg !7
  %214 = icmp slt i64 %207, %193, !dbg !7
  br i1 %214, label %215, label %2310, !dbg !7

215:                                              ; preds = %206
  %216 = shl nsw i64 %207, 7, !dbg !7
  %217 = lshr i64 %207, 12, !dbg !16
  %218 = add nuw nsw i64 %217, %16, !dbg !16
  %219 = lshr i64 %207, 5, !dbg !16
  %220 = and i64 %219, 127, !dbg !16
  %221 = shl i64 %218, 19, !dbg !16
  %222 = and i64 %221, 16252928, !dbg !16
  %223 = shl i64 %219, 19, !dbg !16
  %224 = shl i64 %207, 14, !dbg !16
  %225 = sub i64 %224, %223, !dbg !16
  %226 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %222, !dbg !16
  %227 = getelementptr [2 x i8], ptr addrspace(1) %226, i64 %220, !dbg !16
  %228 = getelementptr [2 x i8], ptr addrspace(1) %227, i64 %225, !dbg !16
  %address.thread433 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !16, !range !39, !frisk.register_lifetimes !1
  %address.local434 = zext nneg i32 %address.thread433 to i64, !dbg !8
  %address.local435 = shl nuw nsw i64 %address.local434, 3, !dbg !8
  %address.local436 = and i64 %address.local435, 1536, !dbg !8
  %address.local437 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local436, !dbg !8
  %address.local438 = ptrtoint ptr addrspace(3) %address.local437 to i64, !dbg !8
  %address.local439 = trunc i64 %address.local438 to i32, !dbg !8
  %229 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local439), !dbg !16
  %230 = zext i32 %229 to i64, !dbg !16
  %231 = inttoptr i64 %230 to ptr addrspace(3), !dbg !16
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
  %address.local450 = getelementptr [2 x i8], ptr addrspace(1) %228, i64 %address.local449, !dbg !16
  %232 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %233 = ptrtoint ptr addrspace(1) %address.local450 to i64, !dbg !16
  %234 = sub i64 %233, %232, !dbg !16
  %dma.buffer.offset1588 = trunc i64 %234 to i32, !dbg !16
  %235 = trunc i64 %232 to i32, !dbg !16
  %236 = insertelement <4 x i32> poison, i32 %235, i64 0, !dbg !16
  %237 = lshr i64 %232, 32, !dbg !16
  %238 = trunc i64 %237 to i32, !dbg !16
  %239 = insertelement <4 x i32> %236, i32 %238, i64 1, !dbg !16
  %240 = insertelement <4 x i32> %239, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1589 = insertelement <4 x i32> %240, i32 131072, i64 3, !dbg !16
  %241 = ptrtoint ptr addrspace(3) %231 to i32, !dbg !16
  %dma.buffer.lds1590 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %241), !dbg !16
  %242 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1590, i32 %dma.buffer.offset1588, <4 x i32> %dma.buffer.rsrc1589) #10, !dbg !16
  %address.local451 = or disjoint i64 %address.local434, 256, !dbg !8
  %address.local452 = shl nuw nsw i64 %address.local451, 3, !dbg !8
  %address.local453 = and i64 %address.local452, 3584, !dbg !8
  %address.local454 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local453, !dbg !8
  %address.local455 = ptrtoint ptr addrspace(3) %address.local454 to i64, !dbg !8
  %address.local456 = trunc i64 %address.local455 to i32, !dbg !8
  %243 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local456), !dbg !16
  %244 = zext i32 %243 to i64, !dbg !16
  %245 = inttoptr i64 %244 to ptr addrspace(3), !dbg !16
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
  %address.local467 = getelementptr [2 x i8], ptr addrspace(1) %228, i64 %address.local466, !dbg !16
  %246 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %247 = ptrtoint ptr addrspace(1) %address.local467 to i64, !dbg !16
  %248 = sub i64 %247, %246, !dbg !16
  %dma.buffer.offset1585 = trunc i64 %248 to i32, !dbg !16
  %249 = trunc i64 %246 to i32, !dbg !16
  %250 = insertelement <4 x i32> poison, i32 %249, i64 0, !dbg !16
  %251 = lshr i64 %246, 32, !dbg !16
  %252 = trunc i64 %251 to i32, !dbg !16
  %253 = insertelement <4 x i32> %250, i32 %252, i64 1, !dbg !16
  %254 = insertelement <4 x i32> %253, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1586 = insertelement <4 x i32> %254, i32 131072, i64 3, !dbg !16
  %255 = ptrtoint ptr addrspace(3) %245 to i32, !dbg !16
  %dma.buffer.lds1587 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %255), !dbg !16
  %256 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1587, i32 %dma.buffer.offset1585, <4 x i32> %dma.buffer.rsrc1586) #10, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %257 = add nuw i64 %207, 1024, !dbg !16
  %258 = icmp slt i64 %257, 0, !dbg !16
  %259 = sub nuw i64 -1025, %207, !dbg !16
  %260 = select i1 %258, i64 %259, i64 %257, !dbg !16
  %261 = sdiv i64 %260, 4096, !dbg !16
  %.lobit37 = ashr i64 %257, 63, !dbg !16
  %262 = xor i64 %261, %.lobit37, !dbg !16
  %263 = add nsw i64 %262, %16, !dbg !16
  %264 = add nsw i64 %219, %197, !dbg !16
  %265 = add nsw i64 %264, 32, !dbg !16
  %266 = icmp slt i64 %264, -32, !dbg !16
  %267 = sub nsw i64 -33, %264, !dbg !16
  %268 = select i1 %266, i64 %267, i64 %265, !dbg !16
  %269 = sdiv i64 %268, 128, !dbg !16
  %270 = sext i1 %266 to i64, !dbg !16
  %271 = xor i64 %269, %270, !dbg !16
  %272 = shl nsw i64 %271, 7, !dbg !16
  %273 = sub i64 %197, %272, !dbg !16
  %274 = shl i64 %263, 19, !dbg !16
  %275 = and i64 %274, 16252928, !dbg !16
  %276 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %275, !dbg !16
  %277 = getelementptr [2 x i8], ptr addrspace(1) %276, i64 %219, !dbg !16
  %278 = getelementptr [2 x i8], ptr addrspace(1) %277, i64 %273, !dbg !16
  %279 = getelementptr i8, ptr addrspace(1) %278, i64 64, !dbg !16
  %280 = getelementptr [2 x i8], ptr addrspace(1) %279, i64 %225, !dbg !16
  %address.local468 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local436, !dbg !8
  %address.local469 = ptrtoint ptr addrspace(3) %address.local468 to i64, !dbg !8
  %address.local470 = trunc i64 %address.local469 to i32, !dbg !8
  %281 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local470), !dbg !16
  %282 = zext i32 %281 to i64, !dbg !16
  %283 = inttoptr i64 %282 to ptr addrspace(3), !dbg !16
  %address.local471 = getelementptr [2 x i8], ptr addrspace(1) %280, i64 %address.local449, !dbg !16
  %284 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %285 = ptrtoint ptr addrspace(1) %address.local471 to i64, !dbg !16
  %286 = sub i64 %285, %284, !dbg !16
  %dma.buffer.offset1582 = trunc i64 %286 to i32, !dbg !16
  %287 = trunc i64 %284 to i32, !dbg !16
  %288 = insertelement <4 x i32> poison, i32 %287, i64 0, !dbg !16
  %289 = lshr i64 %284, 32, !dbg !16
  %290 = trunc i64 %289 to i32, !dbg !16
  %291 = insertelement <4 x i32> %288, i32 %290, i64 1, !dbg !16
  %292 = insertelement <4 x i32> %291, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1583 = insertelement <4 x i32> %292, i32 131072, i64 3, !dbg !16
  %293 = ptrtoint ptr addrspace(3) %283 to i32, !dbg !16
  %dma.buffer.lds1584 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %293), !dbg !16
  %294 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1584, i32 %dma.buffer.offset1582, <4 x i32> %dma.buffer.rsrc1583) #10, !dbg !16
  %address.local472 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local453, !dbg !8
  %address.local473 = ptrtoint ptr addrspace(3) %address.local472 to i64, !dbg !8
  %address.local474 = trunc i64 %address.local473 to i32, !dbg !8
  %295 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local474), !dbg !16
  %296 = zext i32 %295 to i64, !dbg !16
  %297 = inttoptr i64 %296 to ptr addrspace(3), !dbg !16
  %address.local475 = getelementptr [2 x i8], ptr addrspace(1) %280, i64 %address.local466, !dbg !16
  %298 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %299 = ptrtoint ptr addrspace(1) %address.local475 to i64, !dbg !16
  %300 = sub i64 %299, %298, !dbg !16
  %dma.buffer.offset1579 = trunc i64 %300 to i32, !dbg !16
  %301 = trunc i64 %298 to i32, !dbg !16
  %302 = insertelement <4 x i32> poison, i32 %301, i64 0, !dbg !16
  %303 = lshr i64 %298, 32, !dbg !16
  %304 = trunc i64 %303 to i32, !dbg !16
  %305 = insertelement <4 x i32> %302, i32 %304, i64 1, !dbg !16
  %306 = insertelement <4 x i32> %305, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1580 = insertelement <4 x i32> %306, i32 131072, i64 3, !dbg !16
  %307 = ptrtoint ptr addrspace(3) %297 to i32, !dbg !16
  %dma.buffer.lds1581 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %307), !dbg !16
  %308 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1581, i32 %dma.buffer.offset1579, <4 x i32> %dma.buffer.rsrc1580) #10, !dbg !16
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
  %309 = load <4 x half>, ptr addrspace(3) %address.local494, align 8, !dbg !17
  %310 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %309, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1426 = getelementptr i8, ptr addrspace(3) %address.local494, i64 4096, !dbg !17
  %311 = load <4 x half>, ptr addrspace(3) %lds.const.gep1426, align 8, !dbg !17
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
  %312 = load <4 x half>, ptr addrspace(3) %address.local523, align 8, !dbg !17
  %313 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %311, <4 x float> %310) #9, !dbg !17
  %314 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %312, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1427 = getelementptr i8, ptr addrspace(3) %address.local523, i64 4096, !dbg !17
  %315 = load <4 x half>, ptr addrspace(3) %lds.const.gep1427, align 8, !dbg !17
  %lds.const.gep1428 = getelementptr i8, ptr addrspace(3) %address.local494, i64 128, !dbg !17
  %316 = load <4 x half>, ptr addrspace(3) %lds.const.gep1428, align 8, !dbg !17
  %317 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %315, <4 x float> %314) #9, !dbg !17
  %318 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %316, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1429 = getelementptr i8, ptr addrspace(3) %address.local494, i64 4224, !dbg !17
  %319 = load <4 x half>, ptr addrspace(3) %lds.const.gep1429, align 8, !dbg !17
  %lds.const.gep1430 = getelementptr i8, ptr addrspace(3) %address.local523, i64 128, !dbg !17
  %320 = load <4 x half>, ptr addrspace(3) %lds.const.gep1430, align 8, !dbg !17
  %321 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %319, <4 x float> %318) #9, !dbg !17
  %322 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %320, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1431 = getelementptr i8, ptr addrspace(3) %address.local523, i64 4224, !dbg !17
  %323 = load <4 x half>, ptr addrspace(3) %lds.const.gep1431, align 8, !dbg !17
  %lds.const.gep1432 = getelementptr i8, ptr addrspace(3) %address.local494, i64 1024, !dbg !17
  %324 = load <4 x half>, ptr addrspace(3) %lds.const.gep1432, align 8, !dbg !17
  %325 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %323, <4 x float> %322) #9, !dbg !17
  %326 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %324, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1433 = getelementptr i8, ptr addrspace(3) %address.local494, i64 5120, !dbg !17
  %327 = load <4 x half>, ptr addrspace(3) %lds.const.gep1433, align 8, !dbg !17
  %lds.const.gep1434 = getelementptr i8, ptr addrspace(3) %address.local523, i64 1024, !dbg !17
  %328 = load <4 x half>, ptr addrspace(3) %lds.const.gep1434, align 8, !dbg !17
  %329 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %327, <4 x float> %326) #9, !dbg !17
  %330 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %328, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1435 = getelementptr i8, ptr addrspace(3) %address.local523, i64 5120, !dbg !17
  %331 = load <4 x half>, ptr addrspace(3) %lds.const.gep1435, align 8, !dbg !17
  %lds.const.gep1436 = getelementptr i8, ptr addrspace(3) %address.local494, i64 1152, !dbg !17
  %332 = load <4 x half>, ptr addrspace(3) %lds.const.gep1436, align 8, !dbg !17
  %333 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %331, <4 x float> %330) #9, !dbg !17
  %334 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %332, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1437 = getelementptr i8, ptr addrspace(3) %address.local494, i64 5248, !dbg !17
  %335 = load <4 x half>, ptr addrspace(3) %lds.const.gep1437, align 8, !dbg !17
  %lds.const.gep1438 = getelementptr i8, ptr addrspace(3) %address.local523, i64 1152, !dbg !17
  %336 = load <4 x half>, ptr addrspace(3) %lds.const.gep1438, align 8, !dbg !17
  %337 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %335, <4 x float> %334) #9, !dbg !17
  %338 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %336, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1439 = getelementptr i8, ptr addrspace(3) %address.local523, i64 5248, !dbg !17
  %339 = load <4 x half>, ptr addrspace(3) %lds.const.gep1439, align 8, !dbg !17
  %340 = load <4 x half>, ptr addrspace(3) %address.local494, align 8, !dbg !17
  %341 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %339, <4 x float> %338) #9, !dbg !17
  %342 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %340, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1440 = getelementptr i8, ptr addrspace(3) %address.local494, i64 4096, !dbg !17
  %343 = load <4 x half>, ptr addrspace(3) %lds.const.gep1440, align 8, !dbg !17
  %344 = load <4 x half>, ptr addrspace(3) %address.local523, align 8, !dbg !17
  %345 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %343, <4 x float> %342) #9, !dbg !17
  %346 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %344, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1441 = getelementptr i8, ptr addrspace(3) %address.local523, i64 4096, !dbg !17
  %347 = load <4 x half>, ptr addrspace(3) %lds.const.gep1441, align 8, !dbg !17
  %lds.const.gep1442 = getelementptr i8, ptr addrspace(3) %address.local494, i64 128, !dbg !17
  %348 = load <4 x half>, ptr addrspace(3) %lds.const.gep1442, align 8, !dbg !17
  %349 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %347, <4 x float> %346) #9, !dbg !17
  %350 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %348, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1443 = getelementptr i8, ptr addrspace(3) %address.local494, i64 4224, !dbg !17
  %351 = load <4 x half>, ptr addrspace(3) %lds.const.gep1443, align 8, !dbg !17
  %lds.const.gep1444 = getelementptr i8, ptr addrspace(3) %address.local523, i64 128, !dbg !17
  %352 = load <4 x half>, ptr addrspace(3) %lds.const.gep1444, align 8, !dbg !17
  %353 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %351, <4 x float> %350) #9, !dbg !17
  %354 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %352, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1445 = getelementptr i8, ptr addrspace(3) %address.local523, i64 4224, !dbg !17
  %355 = load <4 x half>, ptr addrspace(3) %lds.const.gep1445, align 8, !dbg !17
  %lds.const.gep1446 = getelementptr i8, ptr addrspace(3) %address.local494, i64 1024, !dbg !17
  %356 = load <4 x half>, ptr addrspace(3) %lds.const.gep1446, align 8, !dbg !17
  %357 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %355, <4 x float> %354) #9, !dbg !17
  %358 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %356, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1447 = getelementptr i8, ptr addrspace(3) %address.local494, i64 5120, !dbg !17
  %359 = load <4 x half>, ptr addrspace(3) %lds.const.gep1447, align 8, !dbg !17
  %lds.const.gep1448 = getelementptr i8, ptr addrspace(3) %address.local523, i64 1024, !dbg !17
  %360 = load <4 x half>, ptr addrspace(3) %lds.const.gep1448, align 8, !dbg !17
  %361 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %359, <4 x float> %358) #9, !dbg !17
  %362 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %360, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1449 = getelementptr i8, ptr addrspace(3) %address.local523, i64 5120, !dbg !17
  %363 = load <4 x half>, ptr addrspace(3) %lds.const.gep1449, align 8, !dbg !17
  %lds.const.gep1450 = getelementptr i8, ptr addrspace(3) %address.local494, i64 1152, !dbg !17
  %364 = load <4 x half>, ptr addrspace(3) %lds.const.gep1450, align 8, !dbg !17
  %365 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %363, <4 x float> %362) #9, !dbg !17
  %366 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %364, <4 x float> zeroinitializer) #9, !dbg !17
  %lds.const.gep1451 = getelementptr i8, ptr addrspace(3) %address.local494, i64 5248, !dbg !17
  %367 = load <4 x half>, ptr addrspace(3) %lds.const.gep1451, align 8, !dbg !17
  %lds.const.gep1452 = getelementptr i8, ptr addrspace(3) %address.local523, i64 1152, !dbg !17
  %368 = load <4 x half>, ptr addrspace(3) %lds.const.gep1452, align 8, !dbg !17
  %369 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %367, <4 x float> %366) #9, !dbg !17
  %lds.const.gep1453 = getelementptr i8, ptr addrspace(3) %address.local523, i64 5248, !dbg !17
  %370 = load <4 x half>, ptr addrspace(3) %lds.const.gep1453, align 8, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %371 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %368, <4 x float> zeroinitializer) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %372 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %370, <4 x float> %371) #9, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %373 = add nuw i64 %207, 2048, !dbg !16
  %374 = icmp slt i64 %373, 0, !dbg !16
  %375 = sub nuw i64 -2049, %207, !dbg !16
  %376 = select i1 %374, i64 %375, i64 %373, !dbg !16
  %377 = sdiv i64 %376, 4096, !dbg !16
  %.lobit38 = ashr i64 %373, 63, !dbg !16
  %378 = xor i64 %377, %.lobit38, !dbg !16
  %379 = add nsw i64 %378, %16, !dbg !16
  %380 = add nsw i64 %264, 64, !dbg !16
  %381 = icmp slt i64 %264, -64, !dbg !16
  %382 = sub nsw i64 -65, %264, !dbg !16
  %383 = select i1 %381, i64 %382, i64 %380, !dbg !16
  %384 = sdiv i64 %383, 128, !dbg !16
  %385 = sext i1 %381 to i64, !dbg !16
  %386 = xor i64 %384, %385, !dbg !16
  %387 = shl nsw i64 %386, 7, !dbg !16
  %388 = sub i64 %197, %387, !dbg !16
  %389 = shl i64 %379, 19, !dbg !16
  %390 = and i64 %389, 16252928, !dbg !16
  %391 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %390, !dbg !16
  %392 = getelementptr [2 x i8], ptr addrspace(1) %391, i64 %219, !dbg !16
  %393 = getelementptr [2 x i8], ptr addrspace(1) %392, i64 %388, !dbg !16
  %394 = getelementptr i8, ptr addrspace(1) %393, i64 128, !dbg !16
  %395 = getelementptr [2 x i8], ptr addrspace(1) %394, i64 %225, !dbg !16
  %396 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local439), !dbg !16
  %397 = zext i32 %396 to i64, !dbg !16
  %398 = inttoptr i64 %397 to ptr addrspace(3), !dbg !16
  %address.local832 = getelementptr [2 x i8], ptr addrspace(1) %395, i64 %address.local449, !dbg !16
  %399 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %400 = ptrtoint ptr addrspace(1) %address.local832 to i64, !dbg !16
  %401 = sub i64 %400, %399, !dbg !16
  %dma.buffer.offset1576 = trunc i64 %401 to i32, !dbg !16
  %402 = trunc i64 %399 to i32, !dbg !16
  %403 = insertelement <4 x i32> poison, i32 %402, i64 0, !dbg !16
  %404 = lshr i64 %399, 32, !dbg !16
  %405 = trunc i64 %404 to i32, !dbg !16
  %406 = insertelement <4 x i32> %403, i32 %405, i64 1, !dbg !16
  %407 = insertelement <4 x i32> %406, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1577 = insertelement <4 x i32> %407, i32 131072, i64 3, !dbg !16
  %408 = ptrtoint ptr addrspace(3) %398 to i32, !dbg !16
  %dma.buffer.lds1578 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %408), !dbg !16
  %409 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1578, i32 %dma.buffer.offset1576, <4 x i32> %dma.buffer.rsrc1577) #10, !dbg !16
  %410 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local456), !dbg !16
  %411 = zext i32 %410 to i64, !dbg !16
  %412 = inttoptr i64 %411 to ptr addrspace(3), !dbg !16
  %address.local833 = getelementptr [2 x i8], ptr addrspace(1) %395, i64 %address.local466, !dbg !16
  %413 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %414 = ptrtoint ptr addrspace(1) %address.local833 to i64, !dbg !16
  %415 = sub i64 %414, %413, !dbg !16
  %dma.buffer.offset1573 = trunc i64 %415 to i32, !dbg !16
  %416 = trunc i64 %413 to i32, !dbg !16
  %417 = insertelement <4 x i32> poison, i32 %416, i64 0, !dbg !16
  %418 = lshr i64 %413, 32, !dbg !16
  %419 = trunc i64 %418 to i32, !dbg !16
  %420 = insertelement <4 x i32> %417, i32 %419, i64 1, !dbg !16
  %421 = insertelement <4 x i32> %420, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1574 = insertelement <4 x i32> %421, i32 131072, i64 3, !dbg !16
  %422 = ptrtoint ptr addrspace(3) %412 to i32, !dbg !16
  %dma.buffer.lds1575 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %422), !dbg !16
  %423 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1575, i32 %dma.buffer.offset1573, <4 x i32> %dma.buffer.rsrc1574) #10, !dbg !16
  %lds.const.gep1454 = getelementptr i8, ptr addrspace(3) %address.local494, i64 8192, !dbg !17
  %424 = load <4 x half>, ptr addrspace(3) %lds.const.gep1454, align 8, !dbg !17
  %425 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %424, <4 x float> %313) #9, !dbg !17
  %lds.const.gep1455 = getelementptr i8, ptr addrspace(3) %address.local494, i64 12288, !dbg !17
  %426 = load <4 x half>, ptr addrspace(3) %lds.const.gep1455, align 8, !dbg !17
  %lds.const.gep1456 = getelementptr i8, ptr addrspace(3) %address.local523, i64 8192, !dbg !17
  %427 = load <4 x half>, ptr addrspace(3) %lds.const.gep1456, align 8, !dbg !17
  %428 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %426, <4 x float> %425) #9, !dbg !17
  %429 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %427, <4 x float> %317) #9, !dbg !17
  %lds.const.gep1457 = getelementptr i8, ptr addrspace(3) %address.local523, i64 12288, !dbg !17
  %430 = load <4 x half>, ptr addrspace(3) %lds.const.gep1457, align 8, !dbg !17
  %lds.const.gep1458 = getelementptr i8, ptr addrspace(3) %address.local494, i64 8320, !dbg !17
  %431 = load <4 x half>, ptr addrspace(3) %lds.const.gep1458, align 8, !dbg !17
  %432 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %430, <4 x float> %429) #9, !dbg !17
  %433 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %431, <4 x float> %321) #9, !dbg !17
  %lds.const.gep1459 = getelementptr i8, ptr addrspace(3) %address.local494, i64 12416, !dbg !17
  %434 = load <4 x half>, ptr addrspace(3) %lds.const.gep1459, align 8, !dbg !17
  %lds.const.gep1460 = getelementptr i8, ptr addrspace(3) %address.local523, i64 8320, !dbg !17
  %435 = load <4 x half>, ptr addrspace(3) %lds.const.gep1460, align 8, !dbg !17
  %436 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %434, <4 x float> %433) #9, !dbg !17
  %437 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %435, <4 x float> %325) #9, !dbg !17
  %lds.const.gep1461 = getelementptr i8, ptr addrspace(3) %address.local523, i64 12416, !dbg !17
  %438 = load <4 x half>, ptr addrspace(3) %lds.const.gep1461, align 8, !dbg !17
  %lds.const.gep1462 = getelementptr i8, ptr addrspace(3) %address.local494, i64 9216, !dbg !17
  %439 = load <4 x half>, ptr addrspace(3) %lds.const.gep1462, align 8, !dbg !17
  %440 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %438, <4 x float> %437) #9, !dbg !17
  %441 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %439, <4 x float> %329) #9, !dbg !17
  %lds.const.gep1463 = getelementptr i8, ptr addrspace(3) %address.local494, i64 13312, !dbg !17
  %442 = load <4 x half>, ptr addrspace(3) %lds.const.gep1463, align 8, !dbg !17
  %lds.const.gep1464 = getelementptr i8, ptr addrspace(3) %address.local523, i64 9216, !dbg !17
  %443 = load <4 x half>, ptr addrspace(3) %lds.const.gep1464, align 8, !dbg !17
  %444 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %442, <4 x float> %441) #9, !dbg !17
  %445 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %443, <4 x float> %333) #9, !dbg !17
  %lds.const.gep1465 = getelementptr i8, ptr addrspace(3) %address.local523, i64 13312, !dbg !17
  %446 = load <4 x half>, ptr addrspace(3) %lds.const.gep1465, align 8, !dbg !17
  %lds.const.gep1466 = getelementptr i8, ptr addrspace(3) %address.local494, i64 9344, !dbg !17
  %447 = load <4 x half>, ptr addrspace(3) %lds.const.gep1466, align 8, !dbg !17
  %448 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %446, <4 x float> %445) #9, !dbg !17
  %449 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %447, <4 x float> %337) #9, !dbg !17
  %lds.const.gep1467 = getelementptr i8, ptr addrspace(3) %address.local494, i64 13440, !dbg !17
  %450 = load <4 x half>, ptr addrspace(3) %lds.const.gep1467, align 8, !dbg !17
  %lds.const.gep1468 = getelementptr i8, ptr addrspace(3) %address.local523, i64 9344, !dbg !17
  %451 = load <4 x half>, ptr addrspace(3) %lds.const.gep1468, align 8, !dbg !17
  %452 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %450, <4 x float> %449) #9, !dbg !17
  %453 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %451, <4 x float> %341) #9, !dbg !17
  %lds.const.gep1469 = getelementptr i8, ptr addrspace(3) %address.local523, i64 13440, !dbg !17
  %454 = load <4 x half>, ptr addrspace(3) %lds.const.gep1469, align 8, !dbg !17
  %lds.const.gep1470 = getelementptr i8, ptr addrspace(3) %address.local494, i64 8192, !dbg !17
  %455 = load <4 x half>, ptr addrspace(3) %lds.const.gep1470, align 8, !dbg !17
  %456 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %454, <4 x float> %453) #9, !dbg !17
  %457 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %455, <4 x float> %345) #9, !dbg !17
  %lds.const.gep1471 = getelementptr i8, ptr addrspace(3) %address.local494, i64 12288, !dbg !17
  %458 = load <4 x half>, ptr addrspace(3) %lds.const.gep1471, align 8, !dbg !17
  %lds.const.gep1472 = getelementptr i8, ptr addrspace(3) %address.local523, i64 8192, !dbg !17
  %459 = load <4 x half>, ptr addrspace(3) %lds.const.gep1472, align 8, !dbg !17
  %460 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %458, <4 x float> %457) #9, !dbg !17
  %461 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %459, <4 x float> %349) #9, !dbg !17
  %lds.const.gep1473 = getelementptr i8, ptr addrspace(3) %address.local523, i64 12288, !dbg !17
  %462 = load <4 x half>, ptr addrspace(3) %lds.const.gep1473, align 8, !dbg !17
  %lds.const.gep1474 = getelementptr i8, ptr addrspace(3) %address.local494, i64 8320, !dbg !17
  %463 = load <4 x half>, ptr addrspace(3) %lds.const.gep1474, align 8, !dbg !17
  %464 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %462, <4 x float> %461) #9, !dbg !17
  %465 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %463, <4 x float> %353) #9, !dbg !17
  %lds.const.gep1475 = getelementptr i8, ptr addrspace(3) %address.local494, i64 12416, !dbg !17
  %466 = load <4 x half>, ptr addrspace(3) %lds.const.gep1475, align 8, !dbg !17
  %lds.const.gep1476 = getelementptr i8, ptr addrspace(3) %address.local523, i64 8320, !dbg !17
  %467 = load <4 x half>, ptr addrspace(3) %lds.const.gep1476, align 8, !dbg !17
  %468 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %466, <4 x float> %465) #9, !dbg !17
  %469 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %467, <4 x float> %357) #9, !dbg !17
  %lds.const.gep1477 = getelementptr i8, ptr addrspace(3) %address.local523, i64 12416, !dbg !17
  %470 = load <4 x half>, ptr addrspace(3) %lds.const.gep1477, align 8, !dbg !17
  %lds.const.gep1478 = getelementptr i8, ptr addrspace(3) %address.local494, i64 9216, !dbg !17
  %471 = load <4 x half>, ptr addrspace(3) %lds.const.gep1478, align 8, !dbg !17
  %472 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %470, <4 x float> %469) #9, !dbg !17
  %473 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %471, <4 x float> %361) #9, !dbg !17
  %lds.const.gep1479 = getelementptr i8, ptr addrspace(3) %address.local494, i64 13312, !dbg !17
  %474 = load <4 x half>, ptr addrspace(3) %lds.const.gep1479, align 8, !dbg !17
  %lds.const.gep1480 = getelementptr i8, ptr addrspace(3) %address.local523, i64 9216, !dbg !17
  %475 = load <4 x half>, ptr addrspace(3) %lds.const.gep1480, align 8, !dbg !17
  %476 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %474, <4 x float> %473) #9, !dbg !17
  %477 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %475, <4 x float> %365) #9, !dbg !17
  %lds.const.gep1481 = getelementptr i8, ptr addrspace(3) %address.local523, i64 13312, !dbg !17
  %478 = load <4 x half>, ptr addrspace(3) %lds.const.gep1481, align 8, !dbg !17
  %lds.const.gep1482 = getelementptr i8, ptr addrspace(3) %address.local494, i64 9344, !dbg !17
  %479 = load <4 x half>, ptr addrspace(3) %lds.const.gep1482, align 8, !dbg !17
  %480 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %478, <4 x float> %477) #9, !dbg !17
  %481 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %479, <4 x float> %369) #9, !dbg !17
  %lds.const.gep1483 = getelementptr i8, ptr addrspace(3) %address.local494, i64 13440, !dbg !17
  %482 = load <4 x half>, ptr addrspace(3) %lds.const.gep1483, align 8, !dbg !17
  %lds.const.gep1484 = getelementptr i8, ptr addrspace(3) %address.local523, i64 9344, !dbg !17
  %483 = load <4 x half>, ptr addrspace(3) %lds.const.gep1484, align 8, !dbg !17
  %484 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %482, <4 x float> %481) #9, !dbg !17
  %lds.const.gep1485 = getelementptr i8, ptr addrspace(3) %address.local523, i64 13440, !dbg !17
  %485 = load <4 x half>, ptr addrspace(3) %lds.const.gep1485, align 8, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %486 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %483, <4 x float> %372) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %487 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %485, <4 x float> %486) #9, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %488 = add nuw i64 %207, 3072, !dbg !16
  %489 = icmp slt i64 %488, 0, !dbg !16
  %490 = sub nuw i64 -3073, %207, !dbg !16
  %491 = select i1 %489, i64 %490, i64 %488, !dbg !16
  %492 = sdiv i64 %491, 4096, !dbg !16
  %.lobit39 = ashr i64 %488, 63, !dbg !16
  %493 = xor i64 %492, %.lobit39, !dbg !16
  %494 = add nsw i64 %493, %16, !dbg !16
  %495 = add nsw i64 %264, 96, !dbg !16
  %496 = icmp slt i64 %264, -96, !dbg !16
  %497 = sub nsw i64 -97, %264, !dbg !16
  %498 = select i1 %496, i64 %497, i64 %495, !dbg !16
  %499 = sdiv i64 %498, 128, !dbg !16
  %500 = sext i1 %496 to i64, !dbg !16
  %501 = xor i64 %499, %500, !dbg !16
  %502 = shl nsw i64 %501, 7, !dbg !16
  %503 = sub i64 %197, %502, !dbg !16
  %504 = shl i64 %494, 19, !dbg !16
  %505 = and i64 %504, 16252928, !dbg !16
  %506 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %505, !dbg !16
  %507 = getelementptr [2 x i8], ptr addrspace(1) %506, i64 %219, !dbg !16
  %508 = getelementptr [2 x i8], ptr addrspace(1) %507, i64 %503, !dbg !16
  %509 = getelementptr i8, ptr addrspace(1) %508, i64 192, !dbg !16
  %510 = getelementptr [2 x i8], ptr addrspace(1) %509, i64 %225, !dbg !16
  %511 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local470), !dbg !16
  %512 = zext i32 %511 to i64, !dbg !16
  %513 = inttoptr i64 %512 to ptr addrspace(3), !dbg !16
  %address.local850 = getelementptr [2 x i8], ptr addrspace(1) %510, i64 %address.local449, !dbg !16
  %514 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %515 = ptrtoint ptr addrspace(1) %address.local850 to i64, !dbg !16
  %516 = sub i64 %515, %514, !dbg !16
  %dma.buffer.offset1570 = trunc i64 %516 to i32, !dbg !16
  %517 = trunc i64 %514 to i32, !dbg !16
  %518 = insertelement <4 x i32> poison, i32 %517, i64 0, !dbg !16
  %519 = lshr i64 %514, 32, !dbg !16
  %520 = trunc i64 %519 to i32, !dbg !16
  %521 = insertelement <4 x i32> %518, i32 %520, i64 1, !dbg !16
  %522 = insertelement <4 x i32> %521, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1571 = insertelement <4 x i32> %522, i32 131072, i64 3, !dbg !16
  %523 = ptrtoint ptr addrspace(3) %513 to i32, !dbg !16
  %dma.buffer.lds1572 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %523), !dbg !16
  %524 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1572, i32 %dma.buffer.offset1570, <4 x i32> %dma.buffer.rsrc1571) #10, !dbg !16
  %525 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local474), !dbg !16
  %526 = zext i32 %525 to i64, !dbg !16
  %527 = inttoptr i64 %526 to ptr addrspace(3), !dbg !16
  %address.local851 = getelementptr [2 x i8], ptr addrspace(1) %510, i64 %address.local466, !dbg !16
  %528 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %529 = ptrtoint ptr addrspace(1) %address.local851 to i64, !dbg !16
  %530 = sub i64 %529, %528, !dbg !16
  %dma.buffer.offset1567 = trunc i64 %530 to i32, !dbg !16
  %531 = trunc i64 %528 to i32, !dbg !16
  %532 = insertelement <4 x i32> poison, i32 %531, i64 0, !dbg !16
  %533 = lshr i64 %528, 32, !dbg !16
  %534 = trunc i64 %533 to i32, !dbg !16
  %535 = insertelement <4 x i32> %532, i32 %534, i64 1, !dbg !16
  %536 = insertelement <4 x i32> %535, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc1568 = insertelement <4 x i32> %536, i32 131072, i64 3, !dbg !16
  %537 = ptrtoint ptr addrspace(3) %527 to i32, !dbg !16
  %dma.buffer.lds1569 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %537), !dbg !16
  %538 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1569, i32 %dma.buffer.offset1567, <4 x i32> %dma.buffer.rsrc1568) #10, !dbg !16
  %539 = load <4 x half>, ptr addrspace(3) %address.local494, align 8, !dbg !17
  %540 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %175, <4 x half> %539, <4 x float> %428) #9, !dbg !17
  %lds.const.gep1486 = getelementptr i8, ptr addrspace(3) %address.local494, i64 4096, !dbg !17
  %541 = load <4 x half>, ptr addrspace(3) %lds.const.gep1486, align 8, !dbg !17
  %542 = load <4 x half>, ptr addrspace(3) %address.local523, align 8, !dbg !17
  %543 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %541, <4 x float> %540) #9, !dbg !17
  %544 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %175, <4 x half> %542, <4 x float> %432) #9, !dbg !17
  %lds.const.gep1487 = getelementptr i8, ptr addrspace(3) %address.local523, i64 4096, !dbg !17
  %545 = load <4 x half>, ptr addrspace(3) %lds.const.gep1487, align 8, !dbg !17
  %lds.const.gep1488 = getelementptr i8, ptr addrspace(3) %address.local494, i64 128, !dbg !17
  %546 = load <4 x half>, ptr addrspace(3) %lds.const.gep1488, align 8, !dbg !17
  %547 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %545, <4 x float> %544) #9, !dbg !17
  %548 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %175, <4 x half> %546, <4 x float> %436) #9, !dbg !17
  %lds.const.gep1489 = getelementptr i8, ptr addrspace(3) %address.local494, i64 4224, !dbg !17
  %549 = load <4 x half>, ptr addrspace(3) %lds.const.gep1489, align 8, !dbg !17
  %lds.const.gep1490 = getelementptr i8, ptr addrspace(3) %address.local523, i64 128, !dbg !17
  %550 = load <4 x half>, ptr addrspace(3) %lds.const.gep1490, align 8, !dbg !17
  %551 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %549, <4 x float> %548) #9, !dbg !17
  %552 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %175, <4 x half> %550, <4 x float> %440) #9, !dbg !17
  %lds.const.gep1491 = getelementptr i8, ptr addrspace(3) %address.local523, i64 4224, !dbg !17
  %553 = load <4 x half>, ptr addrspace(3) %lds.const.gep1491, align 8, !dbg !17
  %lds.const.gep1492 = getelementptr i8, ptr addrspace(3) %address.local494, i64 1024, !dbg !17
  %554 = load <4 x half>, ptr addrspace(3) %lds.const.gep1492, align 8, !dbg !17
  %555 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %553, <4 x float> %552) #9, !dbg !17
  %556 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %175, <4 x half> %554, <4 x float> %444) #9, !dbg !17
  %lds.const.gep1493 = getelementptr i8, ptr addrspace(3) %address.local494, i64 5120, !dbg !17
  %557 = load <4 x half>, ptr addrspace(3) %lds.const.gep1493, align 8, !dbg !17
  %lds.const.gep1494 = getelementptr i8, ptr addrspace(3) %address.local523, i64 1024, !dbg !17
  %558 = load <4 x half>, ptr addrspace(3) %lds.const.gep1494, align 8, !dbg !17
  %559 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %557, <4 x float> %556) #9, !dbg !17
  %560 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %175, <4 x half> %558, <4 x float> %448) #9, !dbg !17
  %lds.const.gep1495 = getelementptr i8, ptr addrspace(3) %address.local523, i64 5120, !dbg !17
  %561 = load <4 x half>, ptr addrspace(3) %lds.const.gep1495, align 8, !dbg !17
  %lds.const.gep1496 = getelementptr i8, ptr addrspace(3) %address.local494, i64 1152, !dbg !17
  %562 = load <4 x half>, ptr addrspace(3) %lds.const.gep1496, align 8, !dbg !17
  %563 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %561, <4 x float> %560) #9, !dbg !17
  %564 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %175, <4 x half> %562, <4 x float> %452) #9, !dbg !17
  %lds.const.gep1497 = getelementptr i8, ptr addrspace(3) %address.local494, i64 5248, !dbg !17
  %565 = load <4 x half>, ptr addrspace(3) %lds.const.gep1497, align 8, !dbg !17
  %lds.const.gep1498 = getelementptr i8, ptr addrspace(3) %address.local523, i64 1152, !dbg !17
  %566 = load <4 x half>, ptr addrspace(3) %lds.const.gep1498, align 8, !dbg !17
  %567 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %565, <4 x float> %564) #9, !dbg !17
  %568 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %175, <4 x half> %566, <4 x float> %456) #9, !dbg !17
  %lds.const.gep1499 = getelementptr i8, ptr addrspace(3) %address.local523, i64 5248, !dbg !17
  %569 = load <4 x half>, ptr addrspace(3) %lds.const.gep1499, align 8, !dbg !17
  %570 = load <4 x half>, ptr addrspace(3) %address.local494, align 8, !dbg !17
  %571 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %569, <4 x float> %568) #9, !dbg !17
  %572 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %570, <4 x float> %460) #9, !dbg !17
  %lds.const.gep1500 = getelementptr i8, ptr addrspace(3) %address.local494, i64 4096, !dbg !17
  %573 = load <4 x half>, ptr addrspace(3) %lds.const.gep1500, align 8, !dbg !17
  %574 = load <4 x half>, ptr addrspace(3) %address.local523, align 8, !dbg !17
  %575 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %179, <4 x half> %573, <4 x float> %572) #9, !dbg !17
  %576 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %574, <4 x float> %464) #9, !dbg !17
  %lds.const.gep1501 = getelementptr i8, ptr addrspace(3) %address.local523, i64 4096, !dbg !17
  %577 = load <4 x half>, ptr addrspace(3) %lds.const.gep1501, align 8, !dbg !17
  %lds.const.gep1502 = getelementptr i8, ptr addrspace(3) %address.local494, i64 128, !dbg !17
  %578 = load <4 x half>, ptr addrspace(3) %lds.const.gep1502, align 8, !dbg !17
  %579 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %179, <4 x half> %577, <4 x float> %576) #9, !dbg !17
  %580 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %578, <4 x float> %468) #9, !dbg !17
  %lds.const.gep1503 = getelementptr i8, ptr addrspace(3) %address.local494, i64 4224, !dbg !17
  %581 = load <4 x half>, ptr addrspace(3) %lds.const.gep1503, align 8, !dbg !17
  %lds.const.gep1504 = getelementptr i8, ptr addrspace(3) %address.local523, i64 128, !dbg !17
  %582 = load <4 x half>, ptr addrspace(3) %lds.const.gep1504, align 8, !dbg !17
  %583 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %179, <4 x half> %581, <4 x float> %580) #9, !dbg !17
  %584 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %582, <4 x float> %472) #9, !dbg !17
  %lds.const.gep1505 = getelementptr i8, ptr addrspace(3) %address.local523, i64 4224, !dbg !17
  %585 = load <4 x half>, ptr addrspace(3) %lds.const.gep1505, align 8, !dbg !17
  %lds.const.gep1506 = getelementptr i8, ptr addrspace(3) %address.local494, i64 1024, !dbg !17
  %586 = load <4 x half>, ptr addrspace(3) %lds.const.gep1506, align 8, !dbg !17
  %587 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %179, <4 x half> %585, <4 x float> %584) #9, !dbg !17
  %588 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %586, <4 x float> %476) #9, !dbg !17
  %lds.const.gep1507 = getelementptr i8, ptr addrspace(3) %address.local494, i64 5120, !dbg !17
  %589 = load <4 x half>, ptr addrspace(3) %lds.const.gep1507, align 8, !dbg !17
  %lds.const.gep1508 = getelementptr i8, ptr addrspace(3) %address.local523, i64 1024, !dbg !17
  %590 = load <4 x half>, ptr addrspace(3) %lds.const.gep1508, align 8, !dbg !17
  %591 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %179, <4 x half> %589, <4 x float> %588) #9, !dbg !17
  %592 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %590, <4 x float> %480) #9, !dbg !17
  %lds.const.gep1509 = getelementptr i8, ptr addrspace(3) %address.local523, i64 5120, !dbg !17
  %593 = load <4 x half>, ptr addrspace(3) %lds.const.gep1509, align 8, !dbg !17
  %lds.const.gep1510 = getelementptr i8, ptr addrspace(3) %address.local494, i64 1152, !dbg !17
  %594 = load <4 x half>, ptr addrspace(3) %lds.const.gep1510, align 8, !dbg !17
  %595 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %179, <4 x half> %593, <4 x float> %592) #9, !dbg !17
  %596 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %594, <4 x float> %484) #9, !dbg !17
  %lds.const.gep1511 = getelementptr i8, ptr addrspace(3) %address.local494, i64 5248, !dbg !17
  %597 = load <4 x half>, ptr addrspace(3) %lds.const.gep1511, align 8, !dbg !17
  %lds.const.gep1512 = getelementptr i8, ptr addrspace(3) %address.local523, i64 1152, !dbg !17
  %598 = load <4 x half>, ptr addrspace(3) %lds.const.gep1512, align 8, !dbg !17
  %599 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %179, <4 x half> %597, <4 x float> %596) #9, !dbg !17
  %lds.const.gep1513 = getelementptr i8, ptr addrspace(3) %address.local523, i64 5248, !dbg !17
  %600 = load <4 x half>, ptr addrspace(3) %lds.const.gep1513, align 8, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %601 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %598, <4 x float> %487) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %602 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %179, <4 x half> %600, <4 x float> %601) #9, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %603 = add nuw nsw i64 %219, %16, !dbg !18
  %604 = shl i64 %603, 19, !dbg !18
  %605 = and i64 %604, 16252928, !dbg !18
  %606 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %605, !dbg !18
  %607 = getelementptr [2 x i8], ptr addrspace(1) %606, i64 %225, !dbg !18
  %608 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local439), !dbg !18
  %609 = zext i32 %608 to i64, !dbg !18
  %610 = inttoptr i64 %609 to ptr addrspace(3), !dbg !18
  %address.local852 = shl nuw nsw i64 %address.local434, 5, !dbg !18
  %address.local853 = and i64 %address.local852, 8064, !dbg !18
  %address.local854 = shl nuw nsw i64 %address.local434, 3, !dbg !18
  %address.local855 = and i64 %address.local854, 24, !dbg !18
  %address.local856 = or disjoint i64 %address.local853, %address.local855, !dbg !18
  %address.local857 = getelementptr [2 x i8], ptr addrspace(1) %607, i64 %address.local856, !dbg !18
  %611 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %612 = ptrtoint ptr addrspace(1) %address.local857 to i64, !dbg !18
  %613 = sub i64 %612, %611, !dbg !18
  %dma.buffer.offset1564 = trunc i64 %613 to i32, !dbg !18
  %614 = trunc i64 %611 to i32, !dbg !18
  %615 = insertelement <4 x i32> poison, i32 %614, i64 0, !dbg !18
  %616 = lshr i64 %611, 32, !dbg !18
  %617 = trunc i64 %616 to i32, !dbg !18
  %618 = insertelement <4 x i32> %615, i32 %617, i64 1, !dbg !18
  %619 = insertelement <4 x i32> %618, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1565 = insertelement <4 x i32> %619, i32 131072, i64 3, !dbg !18
  %620 = ptrtoint ptr addrspace(3) %610 to i32, !dbg !18
  %dma.buffer.lds1566 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %620), !dbg !18
  %621 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1566, i32 %dma.buffer.offset1564, <4 x i32> %dma.buffer.rsrc1565) #10, !dbg !18
  %622 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local456), !dbg !18
  %623 = zext i32 %622 to i64, !dbg !18
  %624 = inttoptr i64 %623 to ptr addrspace(3), !dbg !18
  %address.local858 = shl nuw nsw i64 %address.local451, 5, !dbg !18
  %address.local859 = and i64 %address.local858, 16256, !dbg !18
  %address.local860 = shl nuw nsw i64 %address.local434, 3, !dbg !18
  %address.local861 = and i64 %address.local860, 24, !dbg !18
  %address.local862 = or disjoint i64 %address.local859, %address.local861, !dbg !18
  %address.local863 = getelementptr [2 x i8], ptr addrspace(1) %607, i64 %address.local862, !dbg !18
  %625 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %626 = ptrtoint ptr addrspace(1) %address.local863 to i64, !dbg !18
  %627 = sub i64 %626, %625, !dbg !18
  %dma.buffer.offset1561 = trunc i64 %627 to i32, !dbg !18
  %628 = trunc i64 %625 to i32, !dbg !18
  %629 = insertelement <4 x i32> poison, i32 %628, i64 0, !dbg !18
  %630 = lshr i64 %625, 32, !dbg !18
  %631 = trunc i64 %630 to i32, !dbg !18
  %632 = insertelement <4 x i32> %629, i32 %631, i64 1, !dbg !18
  %633 = insertelement <4 x i32> %632, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1562 = insertelement <4 x i32> %633, i32 131072, i64 3, !dbg !18
  %634 = ptrtoint ptr addrspace(3) %624 to i32, !dbg !18
  %dma.buffer.lds1563 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %634), !dbg !18
  %635 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1563, i32 %dma.buffer.offset1561, <4 x i32> %dma.buffer.rsrc1562) #10, !dbg !18
  %lds.const.gep1514 = getelementptr i8, ptr addrspace(3) %address.local494, i64 8192, !dbg !17
  %636 = load <4 x half>, ptr addrspace(3) %lds.const.gep1514, align 8, !dbg !17
  %637 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %636, <4 x float> %543) #9, !dbg !17
  %lds.const.gep1515 = getelementptr i8, ptr addrspace(3) %address.local494, i64 12288, !dbg !17
  %638 = load <4 x half>, ptr addrspace(3) %lds.const.gep1515, align 8, !dbg !17
  %lds.const.gep1516 = getelementptr i8, ptr addrspace(3) %address.local523, i64 8192, !dbg !17
  %639 = load <4 x half>, ptr addrspace(3) %lds.const.gep1516, align 8, !dbg !17
  %640 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %638, <4 x float> %637) #9, !dbg !17
  %641 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %639, <4 x float> %547) #9, !dbg !17
  %lds.const.gep1517 = getelementptr i8, ptr addrspace(3) %address.local523, i64 12288, !dbg !17
  %642 = load <4 x half>, ptr addrspace(3) %lds.const.gep1517, align 8, !dbg !17
  %lds.const.gep1518 = getelementptr i8, ptr addrspace(3) %address.local494, i64 8320, !dbg !17
  %643 = load <4 x half>, ptr addrspace(3) %lds.const.gep1518, align 8, !dbg !17
  %644 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %642, <4 x float> %641) #9, !dbg !17
  %645 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %643, <4 x float> %551) #9, !dbg !17
  %lds.const.gep1519 = getelementptr i8, ptr addrspace(3) %address.local494, i64 12416, !dbg !17
  %646 = load <4 x half>, ptr addrspace(3) %lds.const.gep1519, align 8, !dbg !17
  %lds.const.gep1520 = getelementptr i8, ptr addrspace(3) %address.local523, i64 8320, !dbg !17
  %647 = load <4 x half>, ptr addrspace(3) %lds.const.gep1520, align 8, !dbg !17
  %648 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %646, <4 x float> %645) #9, !dbg !17
  %649 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %647, <4 x float> %555) #9, !dbg !17
  %lds.const.gep1521 = getelementptr i8, ptr addrspace(3) %address.local523, i64 12416, !dbg !17
  %650 = load <4 x half>, ptr addrspace(3) %lds.const.gep1521, align 8, !dbg !17
  %lds.const.gep1522 = getelementptr i8, ptr addrspace(3) %address.local494, i64 9216, !dbg !17
  %651 = load <4 x half>, ptr addrspace(3) %lds.const.gep1522, align 8, !dbg !17
  %652 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %650, <4 x float> %649) #9, !dbg !17
  %653 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %651, <4 x float> %559) #9, !dbg !17
  %lds.const.gep1523 = getelementptr i8, ptr addrspace(3) %address.local494, i64 13312, !dbg !17
  %654 = load <4 x half>, ptr addrspace(3) %lds.const.gep1523, align 8, !dbg !17
  %lds.const.gep1524 = getelementptr i8, ptr addrspace(3) %address.local523, i64 9216, !dbg !17
  %655 = load <4 x half>, ptr addrspace(3) %lds.const.gep1524, align 8, !dbg !17
  %656 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %654, <4 x float> %653) #9, !dbg !17
  %657 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %655, <4 x float> %563) #9, !dbg !17
  %lds.const.gep1525 = getelementptr i8, ptr addrspace(3) %address.local523, i64 13312, !dbg !17
  %658 = load <4 x half>, ptr addrspace(3) %lds.const.gep1525, align 8, !dbg !17
  %lds.const.gep1526 = getelementptr i8, ptr addrspace(3) %address.local494, i64 9344, !dbg !17
  %659 = load <4 x half>, ptr addrspace(3) %lds.const.gep1526, align 8, !dbg !17
  %660 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %658, <4 x float> %657) #9, !dbg !17
  %661 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %659, <4 x float> %567) #9, !dbg !17
  %lds.const.gep1527 = getelementptr i8, ptr addrspace(3) %address.local494, i64 13440, !dbg !17
  %662 = load <4 x half>, ptr addrspace(3) %lds.const.gep1527, align 8, !dbg !17
  %lds.const.gep1528 = getelementptr i8, ptr addrspace(3) %address.local523, i64 9344, !dbg !17
  %663 = load <4 x half>, ptr addrspace(3) %lds.const.gep1528, align 8, !dbg !17
  %664 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %662, <4 x float> %661) #9, !dbg !17
  %665 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %663, <4 x float> %571) #9, !dbg !17
  %lds.const.gep1529 = getelementptr i8, ptr addrspace(3) %address.local523, i64 13440, !dbg !17
  %666 = load <4 x half>, ptr addrspace(3) %lds.const.gep1529, align 8, !dbg !17
  %lds.const.gep1530 = getelementptr i8, ptr addrspace(3) %address.local494, i64 8192, !dbg !17
  %667 = load <4 x half>, ptr addrspace(3) %lds.const.gep1530, align 8, !dbg !17
  %668 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %666, <4 x float> %665) #9, !dbg !17
  %669 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %667, <4 x float> %575) #9, !dbg !17
  %lds.const.gep1531 = getelementptr i8, ptr addrspace(3) %address.local494, i64 12288, !dbg !17
  %670 = load <4 x half>, ptr addrspace(3) %lds.const.gep1531, align 8, !dbg !17
  %lds.const.gep1532 = getelementptr i8, ptr addrspace(3) %address.local523, i64 8192, !dbg !17
  %671 = load <4 x half>, ptr addrspace(3) %lds.const.gep1532, align 8, !dbg !17
  %672 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %187, <4 x half> %670, <4 x float> %669) #9, !dbg !17
  %673 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %671, <4 x float> %579) #9, !dbg !17
  %lds.const.gep1533 = getelementptr i8, ptr addrspace(3) %address.local523, i64 12288, !dbg !17
  %674 = load <4 x half>, ptr addrspace(3) %lds.const.gep1533, align 8, !dbg !17
  %lds.const.gep1534 = getelementptr i8, ptr addrspace(3) %address.local494, i64 8320, !dbg !17
  %675 = load <4 x half>, ptr addrspace(3) %lds.const.gep1534, align 8, !dbg !17
  %676 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %187, <4 x half> %674, <4 x float> %673) #9, !dbg !17
  %677 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %675, <4 x float> %583) #9, !dbg !17
  %lds.const.gep1535 = getelementptr i8, ptr addrspace(3) %address.local494, i64 12416, !dbg !17
  %678 = load <4 x half>, ptr addrspace(3) %lds.const.gep1535, align 8, !dbg !17
  %lds.const.gep1536 = getelementptr i8, ptr addrspace(3) %address.local523, i64 8320, !dbg !17
  %679 = load <4 x half>, ptr addrspace(3) %lds.const.gep1536, align 8, !dbg !17
  %680 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %187, <4 x half> %678, <4 x float> %677) #9, !dbg !17
  %681 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %679, <4 x float> %587) #9, !dbg !17
  %lds.const.gep1537 = getelementptr i8, ptr addrspace(3) %address.local523, i64 12416, !dbg !17
  %682 = load <4 x half>, ptr addrspace(3) %lds.const.gep1537, align 8, !dbg !17
  %lds.const.gep1538 = getelementptr i8, ptr addrspace(3) %address.local494, i64 9216, !dbg !17
  %683 = load <4 x half>, ptr addrspace(3) %lds.const.gep1538, align 8, !dbg !17
  %684 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %187, <4 x half> %682, <4 x float> %681) #9, !dbg !17
  %685 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %683, <4 x float> %591) #9, !dbg !17
  %lds.const.gep1539 = getelementptr i8, ptr addrspace(3) %address.local494, i64 13312, !dbg !17
  %686 = load <4 x half>, ptr addrspace(3) %lds.const.gep1539, align 8, !dbg !17
  %lds.const.gep1540 = getelementptr i8, ptr addrspace(3) %address.local523, i64 9216, !dbg !17
  %687 = load <4 x half>, ptr addrspace(3) %lds.const.gep1540, align 8, !dbg !17
  %688 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %187, <4 x half> %686, <4 x float> %685) #9, !dbg !17
  %689 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %687, <4 x float> %595) #9, !dbg !17
  %lds.const.gep1541 = getelementptr i8, ptr addrspace(3) %address.local523, i64 13312, !dbg !17
  %690 = load <4 x half>, ptr addrspace(3) %lds.const.gep1541, align 8, !dbg !17
  %lds.const.gep1542 = getelementptr i8, ptr addrspace(3) %address.local494, i64 9344, !dbg !17
  %691 = load <4 x half>, ptr addrspace(3) %lds.const.gep1542, align 8, !dbg !17
  %692 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %187, <4 x half> %690, <4 x float> %689) #9, !dbg !17
  %693 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %691, <4 x float> %599) #9, !dbg !17
  %lds.const.gep1543 = getelementptr i8, ptr addrspace(3) %address.local494, i64 13440, !dbg !17
  %694 = load <4 x half>, ptr addrspace(3) %lds.const.gep1543, align 8, !dbg !17
  %lds.const.gep1544 = getelementptr i8, ptr addrspace(3) %address.local523, i64 9344, !dbg !17
  %695 = load <4 x half>, ptr addrspace(3) %lds.const.gep1544, align 8, !dbg !17
  %696 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %187, <4 x half> %694, <4 x float> %693) #9, !dbg !17
  %lds.const.gep1545 = getelementptr i8, ptr addrspace(3) %address.local523, i64 13440, !dbg !17
  %697 = load <4 x half>, ptr addrspace(3) %lds.const.gep1545, align 8, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %698 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %695, <4 x float> %602) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %699 = or disjoint i64 %216, 127, !dbg !13
  %700 = icmp uge i64 %64, %699, !dbg !13
  %701 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %187, <4 x half> %697, <4 x float> %698) #9, !dbg !17
  %702 = and i1 %190, %700, !dbg !13
  br i1 %702, label %703, label %704, !dbg !13

703:                                              ; preds = %215
  br label %848, !dbg !13

704:                                              ; preds = %215
  %705 = add i64 %64, %32, !dbg !13
  %706 = shl nuw nsw i64 %65, 4, !dbg !13
  %707 = add i64 %705, %706, !dbg !13
  %708 = and i32 %31, 240, !dbg !13
  %index.compact = zext nneg i32 %708 to i64, !dbg !13
  %709 = sub i64 %707, %index.compact, !dbg !13
  %710 = and i64 %32, 48, !dbg !13
  %711 = or disjoint i64 %216, %710, !dbg !13
  %712 = add i64 %709, 1, !dbg !23
  %713 = lshr i64 %712, 32, !dbg !24
  %compare.high = trunc i64 %713 to i32, !dbg !24
  %714 = lshr i64 %711, 32, !dbg !24
  %compare.high1157 = trunc i64 %714 to i32, !dbg !24
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high1157, !dbg !24
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high1157, !dbg !24
  %compare.low.base = trunc i64 %712 to i32, !dbg !24
  %compare.low.base1158 = trunc i64 %711 to i32, !dbg !24
  %compare.low = icmp ugt i32 %compare.low.base, %compare.low.base1158, !dbg !24
  %compare.result = select i1 %compare.high.equal, i1 %compare.low, i1 %compare.high.order, !dbg !24
  %715 = select i1 %compare.result, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 1) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local = or i32 %compare.local.base, 1, !dbg !24
  %compare.low1159 = icmp ugt i32 %compare.low.base, %compare.local, !dbg !24
  %compare.result1160 = select i1 %compare.high.equal, i1 %compare.low1159, i1 %compare.high.order, !dbg !24
  %716 = select i1 %compare.result1160, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1161 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 2) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1162 = or i32 %compare.local.base1161, 2, !dbg !24
  %compare.low1163 = icmp ugt i32 %compare.low.base, %compare.local1162, !dbg !24
  %compare.result1164 = select i1 %compare.high.equal, i1 %compare.low1163, i1 %compare.high.order, !dbg !24
  %717 = select i1 %compare.result1164, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1165 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 3) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1166 = or i32 %compare.local.base1165, 3, !dbg !24
  %compare.low1167 = icmp ugt i32 %compare.low.base, %compare.local1166, !dbg !24
  %compare.result1168 = select i1 %compare.high.equal, i1 %compare.low1167, i1 %compare.high.order, !dbg !24
  %718 = select i1 %compare.result1168, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1169 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 4) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1170 = or i32 %compare.local.base1169, 4, !dbg !24
  %compare.low1171 = icmp ugt i32 %compare.low.base, %compare.local1170, !dbg !24
  %compare.result1172 = select i1 %compare.high.equal, i1 %compare.low1171, i1 %compare.high.order, !dbg !24
  %719 = select i1 %compare.result1172, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1173 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 5) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1174 = or i32 %compare.local.base1173, 5, !dbg !24
  %compare.low1175 = icmp ugt i32 %compare.low.base, %compare.local1174, !dbg !24
  %compare.result1176 = select i1 %compare.high.equal, i1 %compare.low1175, i1 %compare.high.order, !dbg !24
  %720 = select i1 %compare.result1176, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1177 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 6) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1178 = or i32 %compare.local.base1177, 6, !dbg !24
  %compare.low1179 = icmp ugt i32 %compare.low.base, %compare.local1178, !dbg !24
  %compare.result1180 = select i1 %compare.high.equal, i1 %compare.low1179, i1 %compare.high.order, !dbg !24
  %721 = select i1 %compare.result1180, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1181 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 7) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1182 = or i32 %compare.local.base1181, 7, !dbg !24
  %compare.low1183 = icmp ugt i32 %compare.low.base, %compare.local1182, !dbg !24
  %compare.result1184 = select i1 %compare.high.equal, i1 %compare.low1183, i1 %compare.high.order, !dbg !24
  %722 = select i1 %compare.result1184, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1185 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 8) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1186 = or i32 %compare.local.base1185, 8, !dbg !24
  %compare.low1187 = icmp ugt i32 %compare.low.base, %compare.local1186, !dbg !24
  %compare.result1188 = select i1 %compare.high.equal, i1 %compare.low1187, i1 %compare.high.order, !dbg !24
  %723 = select i1 %compare.result1188, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1189 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 9) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1190 = or i32 %compare.local.base1189, 9, !dbg !24
  %compare.low1191 = icmp ugt i32 %compare.low.base, %compare.local1190, !dbg !24
  %compare.result1192 = select i1 %compare.high.equal, i1 %compare.low1191, i1 %compare.high.order, !dbg !24
  %724 = select i1 %compare.result1192, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1193 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 10) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1194 = or i32 %compare.local.base1193, 10, !dbg !24
  %compare.low1195 = icmp ugt i32 %compare.low.base, %compare.local1194, !dbg !24
  %compare.result1196 = select i1 %compare.high.equal, i1 %compare.low1195, i1 %compare.high.order, !dbg !24
  %725 = select i1 %compare.result1196, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1197 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 11) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1198 = or i32 %compare.local.base1197, 11, !dbg !24
  %compare.low1199 = icmp ugt i32 %compare.low.base, %compare.local1198, !dbg !24
  %compare.result1200 = select i1 %compare.high.equal, i1 %compare.low1199, i1 %compare.high.order, !dbg !24
  %726 = select i1 %compare.result1200, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1201 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 12) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1202 = or i32 %compare.local.base1201, 12, !dbg !24
  %compare.low1203 = icmp ugt i32 %compare.low.base, %compare.local1202, !dbg !24
  %compare.result1204 = select i1 %compare.high.equal, i1 %compare.low1203, i1 %compare.high.order, !dbg !24
  %727 = select i1 %compare.result1204, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1205 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 13) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1206 = or i32 %compare.local.base1205, 13, !dbg !24
  %compare.low1207 = icmp ugt i32 %compare.low.base, %compare.local1206, !dbg !24
  %compare.result1208 = select i1 %compare.high.equal, i1 %compare.low1207, i1 %compare.high.order, !dbg !24
  %728 = select i1 %compare.result1208, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1209 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 14) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1210 = or i32 %compare.local.base1209, 14, !dbg !24
  %compare.low1211 = icmp ugt i32 %compare.low.base, %compare.local1210, !dbg !24
  %compare.result1212 = select i1 %compare.high.equal, i1 %compare.low1211, i1 %compare.high.order, !dbg !24
  %729 = select i1 %compare.result1212, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1213 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 15) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1214 = or i32 %compare.local.base1213, 15, !dbg !24
  %compare.low1215 = icmp ugt i32 %compare.low.base, %compare.local1214, !dbg !24
  %compare.result1216 = select i1 %compare.high.equal, i1 %compare.low1215, i1 %compare.high.order, !dbg !24
  %730 = select i1 %compare.result1216, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1217 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 16) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1218 = or i32 %compare.local.base1217, 64, !dbg !24
  %compare.low1219 = icmp ugt i32 %compare.low.base, %compare.local1218, !dbg !24
  %compare.result1220 = select i1 %compare.high.equal, i1 %compare.low1219, i1 %compare.high.order, !dbg !24
  %731 = select i1 %compare.result1220, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1221 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 17) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1222 = or i32 %compare.local.base1221, 65, !dbg !24
  %compare.low1223 = icmp ugt i32 %compare.low.base, %compare.local1222, !dbg !24
  %compare.result1224 = select i1 %compare.high.equal, i1 %compare.low1223, i1 %compare.high.order, !dbg !24
  %732 = select i1 %compare.result1224, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1225 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 18) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1226 = or i32 %compare.local.base1225, 66, !dbg !24
  %compare.low1227 = icmp ugt i32 %compare.low.base, %compare.local1226, !dbg !24
  %compare.result1228 = select i1 %compare.high.equal, i1 %compare.low1227, i1 %compare.high.order, !dbg !24
  %733 = select i1 %compare.result1228, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1229 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 19) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1230 = or i32 %compare.local.base1229, 67, !dbg !24
  %compare.low1231 = icmp ugt i32 %compare.low.base, %compare.local1230, !dbg !24
  %compare.result1232 = select i1 %compare.high.equal, i1 %compare.low1231, i1 %compare.high.order, !dbg !24
  %734 = select i1 %compare.result1232, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1233 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 20) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1234 = or i32 %compare.local.base1233, 68, !dbg !24
  %compare.low1235 = icmp ugt i32 %compare.low.base, %compare.local1234, !dbg !24
  %compare.result1236 = select i1 %compare.high.equal, i1 %compare.low1235, i1 %compare.high.order, !dbg !24
  %735 = select i1 %compare.result1236, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1237 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 21) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1238 = or i32 %compare.local.base1237, 69, !dbg !24
  %compare.low1239 = icmp ugt i32 %compare.low.base, %compare.local1238, !dbg !24
  %compare.result1240 = select i1 %compare.high.equal, i1 %compare.low1239, i1 %compare.high.order, !dbg !24
  %736 = select i1 %compare.result1240, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1241 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 22) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1242 = or i32 %compare.local.base1241, 70, !dbg !24
  %compare.low1243 = icmp ugt i32 %compare.low.base, %compare.local1242, !dbg !24
  %compare.result1244 = select i1 %compare.high.equal, i1 %compare.low1243, i1 %compare.high.order, !dbg !24
  %737 = select i1 %compare.result1244, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1245 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 23) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1246 = or i32 %compare.local.base1245, 71, !dbg !24
  %compare.low1247 = icmp ugt i32 %compare.low.base, %compare.local1246, !dbg !24
  %compare.result1248 = select i1 %compare.high.equal, i1 %compare.low1247, i1 %compare.high.order, !dbg !24
  %738 = select i1 %compare.result1248, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1249 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 24) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1250 = or i32 %compare.local.base1249, 72, !dbg !24
  %compare.low1251 = icmp ugt i32 %compare.low.base, %compare.local1250, !dbg !24
  %compare.result1252 = select i1 %compare.high.equal, i1 %compare.low1251, i1 %compare.high.order, !dbg !24
  %739 = select i1 %compare.result1252, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1253 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 25) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1254 = or i32 %compare.local.base1253, 73, !dbg !24
  %compare.low1255 = icmp ugt i32 %compare.low.base, %compare.local1254, !dbg !24
  %compare.result1256 = select i1 %compare.high.equal, i1 %compare.low1255, i1 %compare.high.order, !dbg !24
  %740 = select i1 %compare.result1256, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1257 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 26) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1258 = or i32 %compare.local.base1257, 74, !dbg !24
  %compare.low1259 = icmp ugt i32 %compare.low.base, %compare.local1258, !dbg !24
  %compare.result1260 = select i1 %compare.high.equal, i1 %compare.low1259, i1 %compare.high.order, !dbg !24
  %741 = select i1 %compare.result1260, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1261 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 27) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1262 = or i32 %compare.local.base1261, 75, !dbg !24
  %compare.low1263 = icmp ugt i32 %compare.low.base, %compare.local1262, !dbg !24
  %compare.result1264 = select i1 %compare.high.equal, i1 %compare.low1263, i1 %compare.high.order, !dbg !24
  %742 = select i1 %compare.result1264, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1265 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 28) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1266 = or i32 %compare.local.base1265, 76, !dbg !24
  %compare.low1267 = icmp ugt i32 %compare.low.base, %compare.local1266, !dbg !24
  %compare.result1268 = select i1 %compare.high.equal, i1 %compare.low1267, i1 %compare.high.order, !dbg !24
  %743 = select i1 %compare.result1268, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1269 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 29) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1270 = or i32 %compare.local.base1269, 77, !dbg !24
  %compare.low1271 = icmp ugt i32 %compare.low.base, %compare.local1270, !dbg !24
  %compare.result1272 = select i1 %compare.high.equal, i1 %compare.low1271, i1 %compare.high.order, !dbg !24
  %744 = select i1 %compare.result1272, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1273 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 30) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1274 = or i32 %compare.local.base1273, 78, !dbg !24
  %compare.low1275 = icmp ugt i32 %compare.low.base, %compare.local1274, !dbg !24
  %compare.result1276 = select i1 %compare.high.equal, i1 %compare.low1275, i1 %compare.high.order, !dbg !24
  %745 = select i1 %compare.result1276, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1277 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 31) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1278 = or i32 %compare.local.base1277, 79, !dbg !24
  %compare.low1279 = icmp ugt i32 %compare.low.base, %compare.local1278, !dbg !24
  %compare.result1280 = select i1 %compare.high.equal, i1 %compare.low1279, i1 %compare.high.order, !dbg !24
  %746 = select i1 %compare.result1280, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %747 = add i64 %709, 65, !dbg !23
  %748 = lshr i64 %747, 32, !dbg !24
  %compare.high1281 = trunc i64 %748 to i32, !dbg !24
  %749 = lshr i64 %711, 32, !dbg !24
  %compare.high1282 = trunc i64 %749 to i32, !dbg !24
  %compare.high.equal1283 = icmp eq i32 %compare.high1281, %compare.high1282, !dbg !24
  %compare.high.order1284 = icmp ugt i32 %compare.high1281, %compare.high1282, !dbg !24
  %compare.low.base1285 = trunc i64 %747 to i32, !dbg !24
  %compare.low.base1286 = trunc i64 %711 to i32, !dbg !24
  %compare.low1287 = icmp ugt i32 %compare.low.base1285, %compare.low.base1286, !dbg !24
  %compare.result1288 = select i1 %compare.high.equal1283, i1 %compare.low1287, i1 %compare.high.order1284, !dbg !24
  %750 = select i1 %compare.result1288, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1289 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 32) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1290 = or i32 %compare.local.base1289, 1, !dbg !24
  %compare.low1291 = icmp ugt i32 %compare.low.base1285, %compare.local1290, !dbg !24
  %compare.result1292 = select i1 %compare.high.equal1283, i1 %compare.low1291, i1 %compare.high.order1284, !dbg !24
  %751 = select i1 %compare.result1292, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1293 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 33) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1294 = or i32 %compare.local.base1293, 2, !dbg !24
  %compare.low1295 = icmp ugt i32 %compare.low.base1285, %compare.local1294, !dbg !24
  %compare.result1296 = select i1 %compare.high.equal1283, i1 %compare.low1295, i1 %compare.high.order1284, !dbg !24
  %752 = select i1 %compare.result1296, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1297 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 34) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1298 = or i32 %compare.local.base1297, 3, !dbg !24
  %compare.low1299 = icmp ugt i32 %compare.low.base1285, %compare.local1298, !dbg !24
  %compare.result1300 = select i1 %compare.high.equal1283, i1 %compare.low1299, i1 %compare.high.order1284, !dbg !24
  %753 = select i1 %compare.result1300, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1301 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 35) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1302 = or i32 %compare.local.base1301, 4, !dbg !24
  %compare.low1303 = icmp ugt i32 %compare.low.base1285, %compare.local1302, !dbg !24
  %compare.result1304 = select i1 %compare.high.equal1283, i1 %compare.low1303, i1 %compare.high.order1284, !dbg !24
  %754 = select i1 %compare.result1304, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1305 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 36) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1306 = or i32 %compare.local.base1305, 5, !dbg !24
  %compare.low1307 = icmp ugt i32 %compare.low.base1285, %compare.local1306, !dbg !24
  %compare.result1308 = select i1 %compare.high.equal1283, i1 %compare.low1307, i1 %compare.high.order1284, !dbg !24
  %755 = select i1 %compare.result1308, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1309 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 37) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1310 = or i32 %compare.local.base1309, 6, !dbg !24
  %compare.low1311 = icmp ugt i32 %compare.low.base1285, %compare.local1310, !dbg !24
  %compare.result1312 = select i1 %compare.high.equal1283, i1 %compare.low1311, i1 %compare.high.order1284, !dbg !24
  %756 = select i1 %compare.result1312, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1313 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 38) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1314 = or i32 %compare.local.base1313, 7, !dbg !24
  %compare.low1315 = icmp ugt i32 %compare.low.base1285, %compare.local1314, !dbg !24
  %compare.result1316 = select i1 %compare.high.equal1283, i1 %compare.low1315, i1 %compare.high.order1284, !dbg !24
  %757 = select i1 %compare.result1316, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1317 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 39) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1318 = or i32 %compare.local.base1317, 8, !dbg !24
  %compare.low1319 = icmp ugt i32 %compare.low.base1285, %compare.local1318, !dbg !24
  %compare.result1320 = select i1 %compare.high.equal1283, i1 %compare.low1319, i1 %compare.high.order1284, !dbg !24
  %758 = select i1 %compare.result1320, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1321 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 40) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1322 = or i32 %compare.local.base1321, 9, !dbg !24
  %compare.low1323 = icmp ugt i32 %compare.low.base1285, %compare.local1322, !dbg !24
  %compare.result1324 = select i1 %compare.high.equal1283, i1 %compare.low1323, i1 %compare.high.order1284, !dbg !24
  %759 = select i1 %compare.result1324, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1325 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 41) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1326 = or i32 %compare.local.base1325, 10, !dbg !24
  %compare.low1327 = icmp ugt i32 %compare.low.base1285, %compare.local1326, !dbg !24
  %compare.result1328 = select i1 %compare.high.equal1283, i1 %compare.low1327, i1 %compare.high.order1284, !dbg !24
  %760 = select i1 %compare.result1328, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1329 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 42) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1330 = or i32 %compare.local.base1329, 11, !dbg !24
  %compare.low1331 = icmp ugt i32 %compare.low.base1285, %compare.local1330, !dbg !24
  %compare.result1332 = select i1 %compare.high.equal1283, i1 %compare.low1331, i1 %compare.high.order1284, !dbg !24
  %761 = select i1 %compare.result1332, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1333 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 43) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1334 = or i32 %compare.local.base1333, 12, !dbg !24
  %compare.low1335 = icmp ugt i32 %compare.low.base1285, %compare.local1334, !dbg !24
  %compare.result1336 = select i1 %compare.high.equal1283, i1 %compare.low1335, i1 %compare.high.order1284, !dbg !24
  %762 = select i1 %compare.result1336, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1337 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 44) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1338 = or i32 %compare.local.base1337, 13, !dbg !24
  %compare.low1339 = icmp ugt i32 %compare.low.base1285, %compare.local1338, !dbg !24
  %compare.result1340 = select i1 %compare.high.equal1283, i1 %compare.low1339, i1 %compare.high.order1284, !dbg !24
  %763 = select i1 %compare.result1340, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1341 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 45) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1342 = or i32 %compare.local.base1341, 14, !dbg !24
  %compare.low1343 = icmp ugt i32 %compare.low.base1285, %compare.local1342, !dbg !24
  %compare.result1344 = select i1 %compare.high.equal1283, i1 %compare.low1343, i1 %compare.high.order1284, !dbg !24
  %764 = select i1 %compare.result1344, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1345 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 46) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1346 = or i32 %compare.local.base1345, 15, !dbg !24
  %compare.low1347 = icmp ugt i32 %compare.low.base1285, %compare.local1346, !dbg !24
  %compare.result1348 = select i1 %compare.high.equal1283, i1 %compare.low1347, i1 %compare.high.order1284, !dbg !24
  %765 = select i1 %compare.result1348, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1349 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 47) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1350 = or i32 %compare.local.base1349, 64, !dbg !24
  %compare.low1351 = icmp ugt i32 %compare.low.base1285, %compare.local1350, !dbg !24
  %compare.result1352 = select i1 %compare.high.equal1283, i1 %compare.low1351, i1 %compare.high.order1284, !dbg !24
  %766 = select i1 %compare.result1352, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1353 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 48) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1354 = or i32 %compare.local.base1353, 65, !dbg !24
  %compare.low1355 = icmp ugt i32 %compare.low.base1285, %compare.local1354, !dbg !24
  %compare.result1356 = select i1 %compare.high.equal1283, i1 %compare.low1355, i1 %compare.high.order1284, !dbg !24
  %767 = select i1 %compare.result1356, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1357 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 49) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1358 = or i32 %compare.local.base1357, 66, !dbg !24
  %compare.low1359 = icmp ugt i32 %compare.low.base1285, %compare.local1358, !dbg !24
  %compare.result1360 = select i1 %compare.high.equal1283, i1 %compare.low1359, i1 %compare.high.order1284, !dbg !24
  %768 = select i1 %compare.result1360, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1361 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 50) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1362 = or i32 %compare.local.base1361, 67, !dbg !24
  %compare.low1363 = icmp ugt i32 %compare.low.base1285, %compare.local1362, !dbg !24
  %compare.result1364 = select i1 %compare.high.equal1283, i1 %compare.low1363, i1 %compare.high.order1284, !dbg !24
  %769 = select i1 %compare.result1364, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1365 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 51) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1366 = or i32 %compare.local.base1365, 68, !dbg !24
  %compare.low1367 = icmp ugt i32 %compare.low.base1285, %compare.local1366, !dbg !24
  %compare.result1368 = select i1 %compare.high.equal1283, i1 %compare.low1367, i1 %compare.high.order1284, !dbg !24
  %770 = select i1 %compare.result1368, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1369 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 52) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1370 = or i32 %compare.local.base1369, 69, !dbg !24
  %compare.low1371 = icmp ugt i32 %compare.low.base1285, %compare.local1370, !dbg !24
  %compare.result1372 = select i1 %compare.high.equal1283, i1 %compare.low1371, i1 %compare.high.order1284, !dbg !24
  %771 = select i1 %compare.result1372, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1373 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 53) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1374 = or i32 %compare.local.base1373, 70, !dbg !24
  %compare.low1375 = icmp ugt i32 %compare.low.base1285, %compare.local1374, !dbg !24
  %compare.result1376 = select i1 %compare.high.equal1283, i1 %compare.low1375, i1 %compare.high.order1284, !dbg !24
  %772 = select i1 %compare.result1376, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1377 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 54) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1378 = or i32 %compare.local.base1377, 71, !dbg !24
  %compare.low1379 = icmp ugt i32 %compare.low.base1285, %compare.local1378, !dbg !24
  %compare.result1380 = select i1 %compare.high.equal1283, i1 %compare.low1379, i1 %compare.high.order1284, !dbg !24
  %773 = select i1 %compare.result1380, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1381 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 55) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1382 = or i32 %compare.local.base1381, 72, !dbg !24
  %compare.low1383 = icmp ugt i32 %compare.low.base1285, %compare.local1382, !dbg !24
  %compare.result1384 = select i1 %compare.high.equal1283, i1 %compare.low1383, i1 %compare.high.order1284, !dbg !24
  %774 = select i1 %compare.result1384, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1385 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 56) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1386 = or i32 %compare.local.base1385, 73, !dbg !24
  %compare.low1387 = icmp ugt i32 %compare.low.base1285, %compare.local1386, !dbg !24
  %compare.result1388 = select i1 %compare.high.equal1283, i1 %compare.low1387, i1 %compare.high.order1284, !dbg !24
  %775 = select i1 %compare.result1388, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1389 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 57) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1390 = or i32 %compare.local.base1389, 74, !dbg !24
  %compare.low1391 = icmp ugt i32 %compare.low.base1285, %compare.local1390, !dbg !24
  %compare.result1392 = select i1 %compare.high.equal1283, i1 %compare.low1391, i1 %compare.high.order1284, !dbg !24
  %776 = select i1 %compare.result1392, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1393 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 58) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1394 = or i32 %compare.local.base1393, 75, !dbg !24
  %compare.low1395 = icmp ugt i32 %compare.low.base1285, %compare.local1394, !dbg !24
  %compare.result1396 = select i1 %compare.high.equal1283, i1 %compare.low1395, i1 %compare.high.order1284, !dbg !24
  %777 = select i1 %compare.result1396, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1397 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 59) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1398 = or i32 %compare.local.base1397, 76, !dbg !24
  %compare.low1399 = icmp ugt i32 %compare.low.base1285, %compare.local1398, !dbg !24
  %compare.result1400 = select i1 %compare.high.equal1283, i1 %compare.low1399, i1 %compare.high.order1284, !dbg !24
  %778 = select i1 %compare.result1400, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1401 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 60) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1402 = or i32 %compare.local.base1401, 77, !dbg !24
  %compare.low1403 = icmp ugt i32 %compare.low.base1285, %compare.local1402, !dbg !24
  %compare.result1404 = select i1 %compare.high.equal1283, i1 %compare.low1403, i1 %compare.high.order1284, !dbg !24
  %779 = select i1 %compare.result1404, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1405 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 61) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1406 = or i32 %compare.local.base1405, 78, !dbg !24
  %compare.low1407 = icmp ugt i32 %compare.low.base1285, %compare.local1406, !dbg !24
  %compare.result1408 = select i1 %compare.high.equal1283, i1 %compare.low1407, i1 %compare.high.order1284, !dbg !24
  %780 = select i1 %compare.result1408, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1409 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 62) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1410 = or i32 %compare.local.base1409, 79, !dbg !24
  %compare.low1411 = icmp ugt i32 %compare.low.base1285, %compare.local1410, !dbg !24
  %compare.result1412 = select i1 %compare.high.equal1283, i1 %compare.low1411, i1 %compare.high.order1284, !dbg !24
  %781 = select i1 %compare.result1412, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %782 = insertelement <32 x float> poison, float %715, i64 0, !dbg !13
  %783 = insertelement <32 x float> %782, float %716, i64 1, !dbg !13
  %784 = insertelement <32 x float> %783, float %717, i64 2, !dbg !13
  %785 = insertelement <32 x float> %784, float %718, i64 3, !dbg !13
  %786 = insertelement <32 x float> %785, float %719, i64 4, !dbg !13
  %787 = insertelement <32 x float> %786, float %720, i64 5, !dbg !13
  %788 = insertelement <32 x float> %787, float %721, i64 6, !dbg !13
  %789 = insertelement <32 x float> %788, float %722, i64 7, !dbg !13
  %790 = insertelement <32 x float> %789, float %723, i64 8, !dbg !13
  %791 = insertelement <32 x float> %790, float %724, i64 9, !dbg !13
  %792 = insertelement <32 x float> %791, float %725, i64 10, !dbg !13
  %793 = insertelement <32 x float> %792, float %726, i64 11, !dbg !13
  %794 = insertelement <32 x float> %793, float %727, i64 12, !dbg !13
  %795 = insertelement <32 x float> %794, float %728, i64 13, !dbg !13
  %796 = insertelement <32 x float> %795, float %729, i64 14, !dbg !13
  %797 = insertelement <32 x float> %796, float %730, i64 15, !dbg !13
  %798 = insertelement <32 x float> %797, float %731, i64 16, !dbg !13
  %799 = insertelement <32 x float> %798, float %732, i64 17, !dbg !13
  %800 = insertelement <32 x float> %799, float %733, i64 18, !dbg !13
  %801 = insertelement <32 x float> %800, float %734, i64 19, !dbg !13
  %802 = insertelement <32 x float> %801, float %735, i64 20, !dbg !13
  %803 = insertelement <32 x float> %802, float %736, i64 21, !dbg !13
  %804 = insertelement <32 x float> %803, float %737, i64 22, !dbg !13
  %805 = insertelement <32 x float> %804, float %738, i64 23, !dbg !13
  %806 = insertelement <32 x float> %805, float %739, i64 24, !dbg !13
  %807 = insertelement <32 x float> %806, float %740, i64 25, !dbg !13
  %808 = insertelement <32 x float> %807, float %741, i64 26, !dbg !13
  %809 = insertelement <32 x float> %808, float %742, i64 27, !dbg !13
  %810 = insertelement <32 x float> %809, float %743, i64 28, !dbg !13
  %811 = insertelement <32 x float> %810, float %744, i64 29, !dbg !13
  %812 = insertelement <32 x float> %811, float %745, i64 30, !dbg !13
  %813 = insertelement <32 x float> %812, float %746, i64 31, !dbg !13
  %814 = insertvalue [2 x <32 x float>] poison, <32 x float> %813, 0, !dbg !13
  %815 = insertelement <32 x float> poison, float %750, i64 0, !dbg !13
  %816 = insertelement <32 x float> %815, float %751, i64 1, !dbg !13
  %817 = insertelement <32 x float> %816, float %752, i64 2, !dbg !13
  %818 = insertelement <32 x float> %817, float %753, i64 3, !dbg !13
  %819 = insertelement <32 x float> %818, float %754, i64 4, !dbg !13
  %820 = insertelement <32 x float> %819, float %755, i64 5, !dbg !13
  %821 = insertelement <32 x float> %820, float %756, i64 6, !dbg !13
  %822 = insertelement <32 x float> %821, float %757, i64 7, !dbg !13
  %823 = insertelement <32 x float> %822, float %758, i64 8, !dbg !13
  %824 = insertelement <32 x float> %823, float %759, i64 9, !dbg !13
  %825 = insertelement <32 x float> %824, float %760, i64 10, !dbg !13
  %826 = insertelement <32 x float> %825, float %761, i64 11, !dbg !13
  %827 = insertelement <32 x float> %826, float %762, i64 12, !dbg !13
  %828 = insertelement <32 x float> %827, float %763, i64 13, !dbg !13
  %829 = insertelement <32 x float> %828, float %764, i64 14, !dbg !13
  %830 = insertelement <32 x float> %829, float %765, i64 15, !dbg !13
  %831 = insertelement <32 x float> %830, float %766, i64 16, !dbg !13
  %832 = insertelement <32 x float> %831, float %767, i64 17, !dbg !13
  %833 = insertelement <32 x float> %832, float %768, i64 18, !dbg !13
  %834 = insertelement <32 x float> %833, float %769, i64 19, !dbg !13
  %835 = insertelement <32 x float> %834, float %770, i64 20, !dbg !13
  %836 = insertelement <32 x float> %835, float %771, i64 21, !dbg !13
  %837 = insertelement <32 x float> %836, float %772, i64 22, !dbg !13
  %838 = insertelement <32 x float> %837, float %773, i64 23, !dbg !13
  %839 = insertelement <32 x float> %838, float %774, i64 24, !dbg !13
  %840 = insertelement <32 x float> %839, float %775, i64 25, !dbg !13
  %841 = insertelement <32 x float> %840, float %776, i64 26, !dbg !13
  %842 = insertelement <32 x float> %841, float %777, i64 27, !dbg !13
  %843 = insertelement <32 x float> %842, float %778, i64 28, !dbg !13
  %844 = insertelement <32 x float> %843, float %779, i64 29, !dbg !13
  %845 = insertelement <32 x float> %844, float %780, i64 30, !dbg !13
  %846 = insertelement <32 x float> %845, float %781, i64 31, !dbg !13
  %847 = insertvalue [2 x <32 x float>] %814, <32 x float> %846, 1, !dbg !13
  br label %848, !dbg !13

848:                                              ; preds = %703, %704
  %849 = phi [2 x <32 x float>] [ %847, %704 ], [ zeroinitializer, %703 ], !dbg !13
  br label %850, !dbg !13

850:                                              ; preds = %848
  %851 = extractvalue [2 x <32 x float>] %849, 0, !dbg !26
  %852 = shufflevector <32 x float> %851, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %853 = fadd <4 x float> %640, %852, !dbg !26
  %854 = extractelement <4 x float> %853, i64 0, !dbg !26
  %855 = extractelement <4 x float> %853, i64 1, !dbg !26
  %856 = extractelement <4 x float> %853, i64 2, !dbg !26
  %857 = extractelement <4 x float> %853, i64 3, !dbg !26
  %858 = shufflevector <32 x float> %851, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %859 = fadd <4 x float> %644, %858, !dbg !26
  %860 = extractelement <4 x float> %859, i64 0, !dbg !26
  %861 = extractelement <4 x float> %859, i64 1, !dbg !26
  %862 = extractelement <4 x float> %859, i64 2, !dbg !26
  %863 = extractelement <4 x float> %859, i64 3, !dbg !26
  %864 = shufflevector <32 x float> %851, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !26
  %865 = fadd <4 x float> %648, %864, !dbg !26
  %866 = extractelement <4 x float> %865, i64 0, !dbg !26
  %867 = extractelement <4 x float> %865, i64 1, !dbg !26
  %868 = extractelement <4 x float> %865, i64 2, !dbg !26
  %869 = extractelement <4 x float> %865, i64 3, !dbg !26
  %870 = shufflevector <32 x float> %851, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !26
  %871 = fadd <4 x float> %652, %870, !dbg !26
  %872 = extractelement <4 x float> %871, i64 0, !dbg !26
  %873 = extractelement <4 x float> %871, i64 1, !dbg !26
  %874 = extractelement <4 x float> %871, i64 2, !dbg !26
  %875 = extractelement <4 x float> %871, i64 3, !dbg !26
  %876 = shufflevector <32 x float> %851, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !26
  %877 = fadd <4 x float> %656, %876, !dbg !26
  %878 = extractelement <4 x float> %877, i64 0, !dbg !26
  %879 = extractelement <4 x float> %877, i64 1, !dbg !26
  %880 = extractelement <4 x float> %877, i64 2, !dbg !26
  %881 = extractelement <4 x float> %877, i64 3, !dbg !26
  %882 = shufflevector <32 x float> %851, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !26
  %883 = fadd <4 x float> %660, %882, !dbg !26
  %884 = extractelement <4 x float> %883, i64 0, !dbg !26
  %885 = extractelement <4 x float> %883, i64 1, !dbg !26
  %886 = extractelement <4 x float> %883, i64 2, !dbg !26
  %887 = extractelement <4 x float> %883, i64 3, !dbg !26
  %888 = shufflevector <32 x float> %851, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !26
  %889 = fadd <4 x float> %664, %888, !dbg !26
  %890 = extractelement <4 x float> %889, i64 0, !dbg !26
  %891 = extractelement <4 x float> %889, i64 1, !dbg !26
  %892 = extractelement <4 x float> %889, i64 2, !dbg !26
  %893 = extractelement <4 x float> %889, i64 3, !dbg !26
  %894 = shufflevector <32 x float> %851, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !26
  %895 = fadd <4 x float> %668, %894, !dbg !26
  %896 = extractelement <4 x float> %895, i64 0, !dbg !26
  %897 = extractelement <4 x float> %895, i64 1, !dbg !26
  %898 = extractelement <4 x float> %895, i64 2, !dbg !26
  %899 = extractelement <4 x float> %895, i64 3, !dbg !26
  %900 = extractvalue [2 x <32 x float>] %849, 1, !dbg !26
  %901 = shufflevector <32 x float> %900, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %902 = fadd <4 x float> %672, %901, !dbg !26
  %903 = extractelement <4 x float> %902, i64 0, !dbg !26
  %904 = extractelement <4 x float> %902, i64 1, !dbg !26
  %905 = extractelement <4 x float> %902, i64 2, !dbg !26
  %906 = extractelement <4 x float> %902, i64 3, !dbg !26
  %907 = shufflevector <32 x float> %900, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %908 = fadd <4 x float> %676, %907, !dbg !26
  %909 = extractelement <4 x float> %908, i64 0, !dbg !26
  %910 = extractelement <4 x float> %908, i64 1, !dbg !26
  %911 = extractelement <4 x float> %908, i64 2, !dbg !26
  %912 = extractelement <4 x float> %908, i64 3, !dbg !26
  %913 = shufflevector <32 x float> %900, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !26
  %914 = fadd <4 x float> %680, %913, !dbg !26
  %915 = extractelement <4 x float> %914, i64 0, !dbg !26
  %916 = extractelement <4 x float> %914, i64 1, !dbg !26
  %917 = extractelement <4 x float> %914, i64 2, !dbg !26
  %918 = extractelement <4 x float> %914, i64 3, !dbg !26
  %919 = shufflevector <32 x float> %900, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !26
  %920 = fadd <4 x float> %684, %919, !dbg !26
  %921 = extractelement <4 x float> %920, i64 0, !dbg !26
  %922 = extractelement <4 x float> %920, i64 1, !dbg !26
  %923 = extractelement <4 x float> %920, i64 2, !dbg !26
  %924 = extractelement <4 x float> %920, i64 3, !dbg !26
  %925 = shufflevector <32 x float> %900, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !26
  %926 = fadd <4 x float> %688, %925, !dbg !26
  %927 = extractelement <4 x float> %926, i64 0, !dbg !26
  %928 = extractelement <4 x float> %926, i64 1, !dbg !26
  %929 = extractelement <4 x float> %926, i64 2, !dbg !26
  %930 = extractelement <4 x float> %926, i64 3, !dbg !26
  %931 = shufflevector <32 x float> %900, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !26
  %932 = fadd <4 x float> %692, %931, !dbg !26
  %933 = extractelement <4 x float> %932, i64 0, !dbg !26
  %934 = extractelement <4 x float> %932, i64 1, !dbg !26
  %935 = extractelement <4 x float> %932, i64 2, !dbg !26
  %936 = extractelement <4 x float> %932, i64 3, !dbg !26
  %937 = shufflevector <32 x float> %900, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !26
  %938 = fadd <4 x float> %696, %937, !dbg !26
  %939 = extractelement <4 x float> %938, i64 0, !dbg !26
  %940 = extractelement <4 x float> %938, i64 1, !dbg !26
  %941 = extractelement <4 x float> %938, i64 2, !dbg !26
  %942 = extractelement <4 x float> %938, i64 3, !dbg !26
  %943 = shufflevector <32 x float> %900, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !26
  %944 = fadd <4 x float> %701, %943, !dbg !26
  %945 = extractelement <4 x float> %944, i64 0, !dbg !26
  %946 = extractelement <4 x float> %944, i64 1, !dbg !26
  %947 = extractelement <4 x float> %944, i64 2, !dbg !26
  %948 = extractelement <4 x float> %944, i64 3, !dbg !26
  %949 = call float @llvm.maxnum.f32(float %854, float 0xFFF0000000000000), !dbg !27
  %950 = call float @llvm.maxnum.f32(float %949, float %855), !dbg !27
  %951 = call float @llvm.maxnum.f32(float %950, float %856), !dbg !27
  %952 = call float @llvm.maxnum.f32(float %951, float %857), !dbg !27
  %953 = call float @llvm.maxnum.f32(float %952, float %860), !dbg !27
  %954 = call float @llvm.maxnum.f32(float %953, float %861), !dbg !27
  %955 = call float @llvm.maxnum.f32(float %954, float %862), !dbg !27
  %956 = call float @llvm.maxnum.f32(float %955, float %863), !dbg !27
  %957 = call float @llvm.maxnum.f32(float %956, float %866), !dbg !27
  %958 = call float @llvm.maxnum.f32(float %957, float %867), !dbg !27
  %959 = call float @llvm.maxnum.f32(float %958, float %868), !dbg !27
  %960 = call float @llvm.maxnum.f32(float %959, float %869), !dbg !27
  %961 = call float @llvm.maxnum.f32(float %960, float %872), !dbg !27
  %962 = call float @llvm.maxnum.f32(float %961, float %873), !dbg !27
  %963 = call float @llvm.maxnum.f32(float %962, float %874), !dbg !27
  %964 = call float @llvm.maxnum.f32(float %963, float %875), !dbg !27
  %965 = call float @llvm.maxnum.f32(float %964, float %878), !dbg !27
  %966 = call float @llvm.maxnum.f32(float %965, float %879), !dbg !27
  %967 = call float @llvm.maxnum.f32(float %966, float %880), !dbg !27
  %968 = call float @llvm.maxnum.f32(float %967, float %881), !dbg !27
  %969 = call float @llvm.maxnum.f32(float %968, float %884), !dbg !27
  %970 = call float @llvm.maxnum.f32(float %969, float %885), !dbg !27
  %971 = call float @llvm.maxnum.f32(float %970, float %886), !dbg !27
  %972 = call float @llvm.maxnum.f32(float %971, float %887), !dbg !27
  %973 = call float @llvm.maxnum.f32(float %972, float %890), !dbg !27
  %974 = call float @llvm.maxnum.f32(float %973, float %891), !dbg !27
  %975 = call float @llvm.maxnum.f32(float %974, float %892), !dbg !27
  %976 = call float @llvm.maxnum.f32(float %975, float %893), !dbg !27
  %977 = call float @llvm.maxnum.f32(float %976, float %896), !dbg !27
  %978 = call float @llvm.maxnum.f32(float %977, float %897), !dbg !27
  %979 = call float @llvm.maxnum.f32(float %978, float %898), !dbg !27
  %980 = call float @llvm.maxnum.f32(float %979, float %899), !dbg !27
  %981 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !27
  %982 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %981), !dbg !27
  %983 = add nuw nsw i32 %982, 64, !dbg !27
  %984 = xor i32 %982, 16, !dbg !27
  %985 = and i32 %983, 192, !dbg !27
  %986 = icmp samesign ult i32 %984, %985, !dbg !27
  %987 = select i1 %986, i32 %984, i32 %982, !dbg !27
  %988 = shl nuw nsw i32 %987, 2, !dbg !27
  %989 = bitcast float %980 to i32, !dbg !27
  %990 = call i32 @llvm.amdgcn.ds.bpermute(i32 %988, i32 %989), !dbg !27
  %991 = bitcast i32 %990 to float, !dbg !27
  %992 = call float @llvm.maxnum.f32(float %980, float %991), !dbg !27
  %993 = xor i32 %982, 32, !dbg !27
  %994 = icmp samesign ult i32 %993, %985, !dbg !27
  %995 = select i1 %994, i32 %993, i32 %982, !dbg !27
  %996 = shl nuw nsw i32 %995, 2, !dbg !27
  %997 = bitcast float %992 to i32, !dbg !27
  %998 = call i32 @llvm.amdgcn.ds.bpermute(i32 %996, i32 %997), !dbg !27
  %999 = bitcast i32 %998 to float, !dbg !27
  %1000 = call float @llvm.maxnum.f32(float %992, float %999), !dbg !27
  %1001 = extractvalue [2 x <1 x float>] %210, 0, !dbg !27
  %1002 = insertelement <1 x float> poison, float %1000, i64 0, !dbg !27
  %1003 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %1002, <1 x float> %1001), !dbg !27
  %1004 = call float @llvm.maxnum.f32(float %903, float 0xFFF0000000000000), !dbg !27
  %1005 = call float @llvm.maxnum.f32(float %1004, float %904), !dbg !27
  %1006 = call float @llvm.maxnum.f32(float %1005, float %905), !dbg !27
  %1007 = call float @llvm.maxnum.f32(float %1006, float %906), !dbg !27
  %1008 = call float @llvm.maxnum.f32(float %1007, float %909), !dbg !27
  %1009 = call float @llvm.maxnum.f32(float %1008, float %910), !dbg !27
  %1010 = call float @llvm.maxnum.f32(float %1009, float %911), !dbg !27
  %1011 = call float @llvm.maxnum.f32(float %1010, float %912), !dbg !27
  %1012 = call float @llvm.maxnum.f32(float %1011, float %915), !dbg !27
  %1013 = call float @llvm.maxnum.f32(float %1012, float %916), !dbg !27
  %1014 = call float @llvm.maxnum.f32(float %1013, float %917), !dbg !27
  %1015 = call float @llvm.maxnum.f32(float %1014, float %918), !dbg !27
  %1016 = call float @llvm.maxnum.f32(float %1015, float %921), !dbg !27
  %1017 = call float @llvm.maxnum.f32(float %1016, float %922), !dbg !27
  %1018 = call float @llvm.maxnum.f32(float %1017, float %923), !dbg !27
  %1019 = call float @llvm.maxnum.f32(float %1018, float %924), !dbg !27
  %1020 = call float @llvm.maxnum.f32(float %1019, float %927), !dbg !27
  %1021 = call float @llvm.maxnum.f32(float %1020, float %928), !dbg !27
  %1022 = call float @llvm.maxnum.f32(float %1021, float %929), !dbg !27
  %1023 = call float @llvm.maxnum.f32(float %1022, float %930), !dbg !27
  %1024 = call float @llvm.maxnum.f32(float %1023, float %933), !dbg !27
  %1025 = call float @llvm.maxnum.f32(float %1024, float %934), !dbg !27
  %1026 = call float @llvm.maxnum.f32(float %1025, float %935), !dbg !27
  %1027 = call float @llvm.maxnum.f32(float %1026, float %936), !dbg !27
  %1028 = call float @llvm.maxnum.f32(float %1027, float %939), !dbg !27
  %1029 = call float @llvm.maxnum.f32(float %1028, float %940), !dbg !27
  %1030 = call float @llvm.maxnum.f32(float %1029, float %941), !dbg !27
  %1031 = call float @llvm.maxnum.f32(float %1030, float %942), !dbg !27
  %1032 = call float @llvm.maxnum.f32(float %1031, float %945), !dbg !27
  %1033 = call float @llvm.maxnum.f32(float %1032, float %946), !dbg !27
  %1034 = call float @llvm.maxnum.f32(float %1033, float %947), !dbg !27
  %1035 = call float @llvm.maxnum.f32(float %1034, float %948), !dbg !27
  %1036 = bitcast float %1035 to i32, !dbg !27
  %1037 = call i32 @llvm.amdgcn.ds.bpermute(i32 %988, i32 %1036), !dbg !27
  %1038 = bitcast i32 %1037 to float, !dbg !27
  %1039 = call float @llvm.maxnum.f32(float %1035, float %1038), !dbg !27
  %1040 = bitcast float %1039 to i32, !dbg !27
  %1041 = call i32 @llvm.amdgcn.ds.bpermute(i32 %996, i32 %1040), !dbg !27
  %1042 = bitcast i32 %1041 to float, !dbg !27
  %1043 = call float @llvm.maxnum.f32(float %1039, float %1042), !dbg !27
  %1044 = extractvalue [2 x <1 x float>] %210, 1, !dbg !27
  %1045 = insertelement <1 x float> poison, float %1043, i64 0, !dbg !27
  %1046 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %1045, <1 x float> %1044), !dbg !27
  %1047 = shufflevector <1 x float> %1003, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !28
  %1048 = fsub <4 x float> %853, %1047, !dbg !28
  %1049 = fsub <4 x float> %859, %1047, !dbg !28
  %1050 = fsub <4 x float> %865, %1047, !dbg !28
  %1051 = fsub <4 x float> %871, %1047, !dbg !28
  %1052 = fsub <4 x float> %877, %1047, !dbg !28
  %1053 = fsub <4 x float> %883, %1047, !dbg !28
  %1054 = fsub <4 x float> %889, %1047, !dbg !28
  %1055 = fsub <4 x float> %895, %1047, !dbg !28
  %1056 = shufflevector <1 x float> %1046, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !28
  %1057 = fsub <4 x float> %902, %1056, !dbg !28
  %1058 = fsub <4 x float> %908, %1056, !dbg !28
  %1059 = fsub <4 x float> %914, %1056, !dbg !28
  %1060 = fsub <4 x float> %920, %1056, !dbg !28
  %1061 = fsub <4 x float> %926, %1056, !dbg !28
  %1062 = fsub <4 x float> %932, %1056, !dbg !28
  %1063 = fsub <4 x float> %938, %1056, !dbg !28
  %1064 = fsub <4 x float> %944, %1056, !dbg !28
  %1065 = fsub <1 x float> %1001, %1003, !dbg !29
  %1066 = extractelement <1 x float> %1065, i64 0, !dbg !30
  %1067 = call float @llvm.amdgcn.exp2.f32(float %1066), !dbg !30
  %1068 = insertelement <4 x float> poison, float %1067, i64 0, !dbg !35
  %1069 = shufflevector <4 x float> %1068, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !35
  %1070 = extractvalue [2 x <1 x float>] %209, 0, !dbg !31
  %1071 = extractelement <1 x float> %1070, i64 0, !dbg !31
  %1072 = fmul float %1071, %1067, !dbg !31
  %1073 = fsub <1 x float> %1044, %1046, !dbg !29
  %1074 = extractelement <1 x float> %1073, i64 0, !dbg !30
  %1075 = call float @llvm.amdgcn.exp2.f32(float %1074), !dbg !30
  %1076 = insertelement <4 x float> poison, float %1075, i64 0, !dbg !35
  %1077 = shufflevector <4 x float> %1076, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !35
  %1078 = extractvalue [2 x <1 x float>] %209, 1, !dbg !31
  %1079 = extractelement <1 x float> %1078, i64 0, !dbg !31
  %1080 = fmul float %1079, %1075, !dbg !31
  %1081 = extractelement <4 x float> %1048, i64 0, !dbg !32
  %1082 = call float @llvm.amdgcn.exp2.f32(float %1081), !dbg !32
  %1083 = insertelement <4 x float> poison, float %1082, i64 0, !dbg !34
  %1084 = extractelement <4 x float> %1048, i64 1, !dbg !32
  %1085 = call float @llvm.amdgcn.exp2.f32(float %1084), !dbg !32
  %1086 = insertelement <4 x float> %1083, float %1085, i64 1, !dbg !34
  %1087 = fadd float %1082, %1085, !dbg !33
  %1088 = extractelement <4 x float> %1048, i64 2, !dbg !32
  %1089 = call float @llvm.amdgcn.exp2.f32(float %1088), !dbg !32
  %1090 = insertelement <4 x float> %1086, float %1089, i64 2, !dbg !34
  %1091 = fadd float %1087, %1089, !dbg !33
  %1092 = extractelement <4 x float> %1048, i64 3, !dbg !32
  %1093 = call float @llvm.amdgcn.exp2.f32(float %1092), !dbg !32
  %1094 = insertelement <4 x float> %1090, float %1093, i64 3, !dbg !34
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1094) #9, !dbg !34, !frisk.register_lifetimes !1
  %1095 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !34
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1095) #9, !dbg !33, !frisk.register_lifetimes !1
  %1096 = fadd float %1091, %1093, !dbg !33
  %1097 = extractelement <4 x float> %1049, i64 0, !dbg !32
  %1098 = call float @llvm.amdgcn.exp2.f32(float %1097), !dbg !32
  %1099 = insertelement <4 x float> poison, float %1098, i64 0, !dbg !34
  %1100 = fadd float %1096, %1098, !dbg !33
  %1101 = extractelement <4 x float> %1049, i64 1, !dbg !32
  %1102 = call float @llvm.amdgcn.exp2.f32(float %1101), !dbg !32
  %1103 = insertelement <4 x float> %1099, float %1102, i64 1, !dbg !34
  %1104 = fadd float %1100, %1102, !dbg !33
  %1105 = extractelement <4 x float> %1049, i64 2, !dbg !32
  %1106 = call float @llvm.amdgcn.exp2.f32(float %1105), !dbg !32
  %1107 = insertelement <4 x float> %1103, float %1106, i64 2, !dbg !34
  %1108 = fadd float %1104, %1106, !dbg !33
  %1109 = extractelement <4 x float> %1049, i64 3, !dbg !32
  %1110 = call float @llvm.amdgcn.exp2.f32(float %1109), !dbg !32
  %1111 = insertelement <4 x float> %1107, float %1110, i64 3, !dbg !34
  %math.result309 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1111) #9, !dbg !34, !frisk.register_lifetimes !1
  %1112 = fptrunc <4 x float> %math.result309 to <4 x half>, !dbg !34
  %math.packed310 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1112) #9, !dbg !33, !frisk.register_lifetimes !1
  %1113 = fadd float %1108, %1110, !dbg !33
  %1114 = extractelement <4 x float> %1050, i64 0, !dbg !32
  %1115 = call float @llvm.amdgcn.exp2.f32(float %1114), !dbg !32
  %1116 = insertelement <4 x float> poison, float %1115, i64 0, !dbg !34
  %1117 = fadd float %1113, %1115, !dbg !33
  %1118 = extractelement <4 x float> %1050, i64 1, !dbg !32
  %1119 = call float @llvm.amdgcn.exp2.f32(float %1118), !dbg !32
  %1120 = insertelement <4 x float> %1116, float %1119, i64 1, !dbg !34
  %1121 = fadd float %1117, %1119, !dbg !33
  %1122 = extractelement <4 x float> %1050, i64 2, !dbg !32
  %1123 = call float @llvm.amdgcn.exp2.f32(float %1122), !dbg !32
  %1124 = insertelement <4 x float> %1120, float %1123, i64 2, !dbg !34
  %1125 = fadd float %1121, %1123, !dbg !33
  %1126 = extractelement <4 x float> %1050, i64 3, !dbg !32
  %1127 = call float @llvm.amdgcn.exp2.f32(float %1126), !dbg !32
  %1128 = insertelement <4 x float> %1124, float %1127, i64 3, !dbg !34
  %math.result311 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1128) #9, !dbg !34, !frisk.register_lifetimes !1
  %1129 = fptrunc <4 x float> %math.result311 to <4 x half>, !dbg !34
  %math.packed312 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1129) #9, !dbg !33, !frisk.register_lifetimes !1
  %1130 = fadd float %1125, %1127, !dbg !33
  %1131 = extractelement <4 x float> %1051, i64 0, !dbg !32
  %1132 = call float @llvm.amdgcn.exp2.f32(float %1131), !dbg !32
  %1133 = insertelement <4 x float> poison, float %1132, i64 0, !dbg !34
  %1134 = fadd float %1130, %1132, !dbg !33
  %1135 = extractelement <4 x float> %1051, i64 1, !dbg !32
  %1136 = call float @llvm.amdgcn.exp2.f32(float %1135), !dbg !32
  %1137 = insertelement <4 x float> %1133, float %1136, i64 1, !dbg !34
  %1138 = fadd float %1134, %1136, !dbg !33
  %1139 = extractelement <4 x float> %1051, i64 2, !dbg !32
  %1140 = call float @llvm.amdgcn.exp2.f32(float %1139), !dbg !32
  %1141 = insertelement <4 x float> %1137, float %1140, i64 2, !dbg !34
  %1142 = fadd float %1138, %1140, !dbg !33
  %1143 = extractelement <4 x float> %1051, i64 3, !dbg !32
  %1144 = call float @llvm.amdgcn.exp2.f32(float %1143), !dbg !32
  %1145 = insertelement <4 x float> %1141, float %1144, i64 3, !dbg !34
  %math.result313 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1145) #9, !dbg !34, !frisk.register_lifetimes !1
  %1146 = fptrunc <4 x float> %math.result313 to <4 x half>, !dbg !34
  %math.packed314 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1146) #9, !dbg !33, !frisk.register_lifetimes !1
  %1147 = fadd float %1142, %1144, !dbg !33
  %1148 = extractelement <4 x float> %1052, i64 0, !dbg !32
  %1149 = call float @llvm.amdgcn.exp2.f32(float %1148), !dbg !32
  %1150 = insertelement <4 x float> poison, float %1149, i64 0, !dbg !34
  %1151 = fadd float %1147, %1149, !dbg !33
  %1152 = extractelement <4 x float> %1052, i64 1, !dbg !32
  %1153 = call float @llvm.amdgcn.exp2.f32(float %1152), !dbg !32
  %1154 = insertelement <4 x float> %1150, float %1153, i64 1, !dbg !34
  %1155 = fadd float %1151, %1153, !dbg !33
  %1156 = extractelement <4 x float> %1052, i64 2, !dbg !32
  %1157 = call float @llvm.amdgcn.exp2.f32(float %1156), !dbg !32
  %1158 = insertelement <4 x float> %1154, float %1157, i64 2, !dbg !34
  %1159 = fadd float %1155, %1157, !dbg !33
  %1160 = extractelement <4 x float> %1052, i64 3, !dbg !32
  %1161 = call float @llvm.amdgcn.exp2.f32(float %1160), !dbg !32
  %1162 = insertelement <4 x float> %1158, float %1161, i64 3, !dbg !34
  %math.result315 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1162) #9, !dbg !34, !frisk.register_lifetimes !1
  %1163 = fptrunc <4 x float> %math.result315 to <4 x half>, !dbg !34
  %math.packed316 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1163) #9, !dbg !33, !frisk.register_lifetimes !1
  %1164 = fadd float %1159, %1161, !dbg !33
  %1165 = extractelement <4 x float> %1053, i64 0, !dbg !32
  %1166 = call float @llvm.amdgcn.exp2.f32(float %1165), !dbg !32
  %1167 = insertelement <4 x float> poison, float %1166, i64 0, !dbg !34
  %1168 = fadd float %1164, %1166, !dbg !33
  %1169 = extractelement <4 x float> %1053, i64 1, !dbg !32
  %1170 = call float @llvm.amdgcn.exp2.f32(float %1169), !dbg !32
  %1171 = insertelement <4 x float> %1167, float %1170, i64 1, !dbg !34
  %1172 = fadd float %1168, %1170, !dbg !33
  %1173 = extractelement <4 x float> %1053, i64 2, !dbg !32
  %1174 = call float @llvm.amdgcn.exp2.f32(float %1173), !dbg !32
  %1175 = insertelement <4 x float> %1171, float %1174, i64 2, !dbg !34
  %1176 = fadd float %1172, %1174, !dbg !33
  %1177 = extractelement <4 x float> %1053, i64 3, !dbg !32
  %1178 = call float @llvm.amdgcn.exp2.f32(float %1177), !dbg !32
  %1179 = insertelement <4 x float> %1175, float %1178, i64 3, !dbg !34
  %math.result317 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1179) #9, !dbg !34, !frisk.register_lifetimes !1
  %1180 = fptrunc <4 x float> %math.result317 to <4 x half>, !dbg !34
  %math.packed318 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1180) #9, !dbg !33, !frisk.register_lifetimes !1
  %1181 = fadd float %1176, %1178, !dbg !33
  %1182 = extractelement <4 x float> %1054, i64 0, !dbg !32
  %1183 = call float @llvm.amdgcn.exp2.f32(float %1182), !dbg !32
  %1184 = insertelement <4 x float> poison, float %1183, i64 0, !dbg !34
  %1185 = fadd float %1181, %1183, !dbg !33
  %1186 = extractelement <4 x float> %1054, i64 1, !dbg !32
  %1187 = call float @llvm.amdgcn.exp2.f32(float %1186), !dbg !32
  %1188 = insertelement <4 x float> %1184, float %1187, i64 1, !dbg !34
  %1189 = fadd float %1185, %1187, !dbg !33
  %1190 = extractelement <4 x float> %1054, i64 2, !dbg !32
  %1191 = call float @llvm.amdgcn.exp2.f32(float %1190), !dbg !32
  %1192 = insertelement <4 x float> %1188, float %1191, i64 2, !dbg !34
  %1193 = fadd float %1189, %1191, !dbg !33
  %1194 = extractelement <4 x float> %1054, i64 3, !dbg !32
  %1195 = call float @llvm.amdgcn.exp2.f32(float %1194), !dbg !32
  %1196 = insertelement <4 x float> %1192, float %1195, i64 3, !dbg !34
  %math.result319 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1196) #9, !dbg !34, !frisk.register_lifetimes !1
  %1197 = fptrunc <4 x float> %math.result319 to <4 x half>, !dbg !34
  %math.packed320 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1197) #9, !dbg !33, !frisk.register_lifetimes !1
  %1198 = fadd float %1193, %1195, !dbg !33
  %1199 = extractelement <4 x float> %1055, i64 0, !dbg !32
  %1200 = call float @llvm.amdgcn.exp2.f32(float %1199), !dbg !32
  %1201 = insertelement <4 x float> poison, float %1200, i64 0, !dbg !34
  %1202 = fadd float %1198, %1200, !dbg !33
  %1203 = extractelement <4 x float> %1055, i64 1, !dbg !32
  %1204 = call float @llvm.amdgcn.exp2.f32(float %1203), !dbg !32
  %1205 = insertelement <4 x float> %1201, float %1204, i64 1, !dbg !34
  %1206 = fadd float %1202, %1204, !dbg !33
  %1207 = extractelement <4 x float> %1055, i64 2, !dbg !32
  %1208 = call float @llvm.amdgcn.exp2.f32(float %1207), !dbg !32
  %1209 = insertelement <4 x float> %1205, float %1208, i64 2, !dbg !34
  %1210 = fadd float %1206, %1208, !dbg !33
  %1211 = extractelement <4 x float> %1055, i64 3, !dbg !32
  %1212 = call float @llvm.amdgcn.exp2.f32(float %1211), !dbg !32
  %1213 = insertelement <4 x float> %1209, float %1212, i64 3, !dbg !34
  %math.result321 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1213) #9, !dbg !34, !frisk.register_lifetimes !1
  %1214 = fptrunc <4 x float> %math.result321 to <4 x half>, !dbg !34
  %math.packed322 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1214) #9, !dbg !33, !frisk.register_lifetimes !1
  %1215 = fadd float %1210, %1212, !dbg !33
  %1216 = extractelement <4 x float> %1057, i64 0, !dbg !32
  %1217 = call float @llvm.amdgcn.exp2.f32(float %1216), !dbg !32
  %1218 = insertelement <4 x float> poison, float %1217, i64 0, !dbg !34
  %1219 = extractelement <4 x float> %1057, i64 1, !dbg !32
  %1220 = call float @llvm.amdgcn.exp2.f32(float %1219), !dbg !32
  %1221 = insertelement <4 x float> %1218, float %1220, i64 1, !dbg !34
  %1222 = fadd float %1217, %1220, !dbg !33
  %1223 = extractelement <4 x float> %1057, i64 2, !dbg !32
  %1224 = call float @llvm.amdgcn.exp2.f32(float %1223), !dbg !32
  %1225 = insertelement <4 x float> %1221, float %1224, i64 2, !dbg !34
  %1226 = fadd float %1222, %1224, !dbg !33
  %1227 = extractelement <4 x float> %1057, i64 3, !dbg !32
  %1228 = call float @llvm.amdgcn.exp2.f32(float %1227), !dbg !32
  %1229 = insertelement <4 x float> %1225, float %1228, i64 3, !dbg !34
  %math.result323 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1229) #9, !dbg !34, !frisk.register_lifetimes !1
  %1230 = fptrunc <4 x float> %math.result323 to <4 x half>, !dbg !34
  %math.packed324 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1230) #9, !dbg !33, !frisk.register_lifetimes !1
  %1231 = fadd float %1226, %1228, !dbg !33
  %1232 = extractelement <4 x float> %1058, i64 0, !dbg !32
  %1233 = call float @llvm.amdgcn.exp2.f32(float %1232), !dbg !32
  %1234 = insertelement <4 x float> poison, float %1233, i64 0, !dbg !34
  %1235 = fadd float %1231, %1233, !dbg !33
  %1236 = extractelement <4 x float> %1058, i64 1, !dbg !32
  %1237 = call float @llvm.amdgcn.exp2.f32(float %1236), !dbg !32
  %1238 = insertelement <4 x float> %1234, float %1237, i64 1, !dbg !34
  %1239 = fadd float %1235, %1237, !dbg !33
  %1240 = extractelement <4 x float> %1058, i64 2, !dbg !32
  %1241 = call float @llvm.amdgcn.exp2.f32(float %1240), !dbg !32
  %1242 = insertelement <4 x float> %1238, float %1241, i64 2, !dbg !34
  %1243 = fadd float %1239, %1241, !dbg !33
  %1244 = extractelement <4 x float> %1058, i64 3, !dbg !32
  %1245 = call float @llvm.amdgcn.exp2.f32(float %1244), !dbg !32
  %1246 = insertelement <4 x float> %1242, float %1245, i64 3, !dbg !34
  %math.result325 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1246) #9, !dbg !34, !frisk.register_lifetimes !1
  %1247 = fptrunc <4 x float> %math.result325 to <4 x half>, !dbg !34
  %math.packed326 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1247) #9, !dbg !33, !frisk.register_lifetimes !1
  %1248 = fadd float %1243, %1245, !dbg !33
  %1249 = extractelement <4 x float> %1059, i64 0, !dbg !32
  %1250 = call float @llvm.amdgcn.exp2.f32(float %1249), !dbg !32
  %1251 = insertelement <4 x float> poison, float %1250, i64 0, !dbg !34
  %1252 = fadd float %1248, %1250, !dbg !33
  %1253 = extractelement <4 x float> %1059, i64 1, !dbg !32
  %1254 = call float @llvm.amdgcn.exp2.f32(float %1253), !dbg !32
  %1255 = insertelement <4 x float> %1251, float %1254, i64 1, !dbg !34
  %1256 = fadd float %1252, %1254, !dbg !33
  %1257 = extractelement <4 x float> %1059, i64 2, !dbg !32
  %1258 = call float @llvm.amdgcn.exp2.f32(float %1257), !dbg !32
  %1259 = insertelement <4 x float> %1255, float %1258, i64 2, !dbg !34
  %1260 = fadd float %1256, %1258, !dbg !33
  %1261 = extractelement <4 x float> %1059, i64 3, !dbg !32
  %1262 = call float @llvm.amdgcn.exp2.f32(float %1261), !dbg !32
  %1263 = insertelement <4 x float> %1259, float %1262, i64 3, !dbg !34
  %math.result327 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1263) #9, !dbg !34, !frisk.register_lifetimes !1
  %1264 = fptrunc <4 x float> %math.result327 to <4 x half>, !dbg !34
  %math.packed328 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1264) #9, !dbg !33, !frisk.register_lifetimes !1
  %1265 = fadd float %1260, %1262, !dbg !33
  %1266 = extractelement <4 x float> %1060, i64 0, !dbg !32
  %1267 = call float @llvm.amdgcn.exp2.f32(float %1266), !dbg !32
  %1268 = insertelement <4 x float> poison, float %1267, i64 0, !dbg !34
  %1269 = fadd float %1265, %1267, !dbg !33
  %1270 = extractelement <4 x float> %1060, i64 1, !dbg !32
  %1271 = call float @llvm.amdgcn.exp2.f32(float %1270), !dbg !32
  %1272 = insertelement <4 x float> %1268, float %1271, i64 1, !dbg !34
  %1273 = fadd float %1269, %1271, !dbg !33
  %1274 = extractelement <4 x float> %1060, i64 2, !dbg !32
  %1275 = call float @llvm.amdgcn.exp2.f32(float %1274), !dbg !32
  %1276 = insertelement <4 x float> %1272, float %1275, i64 2, !dbg !34
  %1277 = fadd float %1273, %1275, !dbg !33
  %1278 = extractelement <4 x float> %1060, i64 3, !dbg !32
  %1279 = call float @llvm.amdgcn.exp2.f32(float %1278), !dbg !32
  %1280 = insertelement <4 x float> %1276, float %1279, i64 3, !dbg !34
  %math.result329 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1280) #9, !dbg !34, !frisk.register_lifetimes !1
  %1281 = fptrunc <4 x float> %math.result329 to <4 x half>, !dbg !34
  %math.packed330 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1281) #9, !dbg !33, !frisk.register_lifetimes !1
  %1282 = fadd float %1277, %1279, !dbg !33
  %1283 = extractelement <4 x float> %1061, i64 0, !dbg !32
  %1284 = call float @llvm.amdgcn.exp2.f32(float %1283), !dbg !32
  %1285 = insertelement <4 x float> poison, float %1284, i64 0, !dbg !34
  %1286 = fadd float %1282, %1284, !dbg !33
  %1287 = extractelement <4 x float> %1061, i64 1, !dbg !32
  %1288 = call float @llvm.amdgcn.exp2.f32(float %1287), !dbg !32
  %1289 = insertelement <4 x float> %1285, float %1288, i64 1, !dbg !34
  %1290 = fadd float %1286, %1288, !dbg !33
  %1291 = extractelement <4 x float> %1061, i64 2, !dbg !32
  %1292 = call float @llvm.amdgcn.exp2.f32(float %1291), !dbg !32
  %1293 = insertelement <4 x float> %1289, float %1292, i64 2, !dbg !34
  %1294 = fadd float %1290, %1292, !dbg !33
  %1295 = extractelement <4 x float> %1061, i64 3, !dbg !32
  %1296 = call float @llvm.amdgcn.exp2.f32(float %1295), !dbg !32
  %1297 = insertelement <4 x float> %1293, float %1296, i64 3, !dbg !34
  %math.result331 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1297) #9, !dbg !34, !frisk.register_lifetimes !1
  %1298 = fptrunc <4 x float> %math.result331 to <4 x half>, !dbg !34
  %math.packed332 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1298) #9, !dbg !33, !frisk.register_lifetimes !1
  %1299 = fadd float %1294, %1296, !dbg !33
  %1300 = extractelement <4 x float> %1062, i64 0, !dbg !32
  %1301 = call float @llvm.amdgcn.exp2.f32(float %1300), !dbg !32
  %1302 = insertelement <4 x float> poison, float %1301, i64 0, !dbg !34
  %1303 = fadd float %1299, %1301, !dbg !33
  %1304 = extractelement <4 x float> %1062, i64 1, !dbg !32
  %1305 = call float @llvm.amdgcn.exp2.f32(float %1304), !dbg !32
  %1306 = insertelement <4 x float> %1302, float %1305, i64 1, !dbg !34
  %1307 = fadd float %1303, %1305, !dbg !33
  %1308 = extractelement <4 x float> %1062, i64 2, !dbg !32
  %1309 = call float @llvm.amdgcn.exp2.f32(float %1308), !dbg !32
  %1310 = insertelement <4 x float> %1306, float %1309, i64 2, !dbg !34
  %1311 = fadd float %1307, %1309, !dbg !33
  %1312 = extractelement <4 x float> %1062, i64 3, !dbg !32
  %1313 = call float @llvm.amdgcn.exp2.f32(float %1312), !dbg !32
  %1314 = insertelement <4 x float> %1310, float %1313, i64 3, !dbg !34
  %math.result333 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1314) #9, !dbg !34, !frisk.register_lifetimes !1
  %1315 = fptrunc <4 x float> %math.result333 to <4 x half>, !dbg !34
  %math.packed334 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1315) #9, !dbg !33, !frisk.register_lifetimes !1
  %1316 = fadd float %1311, %1313, !dbg !33
  %1317 = extractelement <4 x float> %1063, i64 0, !dbg !32
  %1318 = call float @llvm.amdgcn.exp2.f32(float %1317), !dbg !32
  %1319 = insertelement <4 x float> poison, float %1318, i64 0, !dbg !34
  %1320 = fadd float %1316, %1318, !dbg !33
  %1321 = extractelement <4 x float> %1063, i64 1, !dbg !32
  %1322 = call float @llvm.amdgcn.exp2.f32(float %1321), !dbg !32
  %1323 = insertelement <4 x float> %1319, float %1322, i64 1, !dbg !34
  %1324 = fadd float %1320, %1322, !dbg !33
  %1325 = extractelement <4 x float> %1063, i64 2, !dbg !32
  %1326 = call float @llvm.amdgcn.exp2.f32(float %1325), !dbg !32
  %1327 = insertelement <4 x float> %1323, float %1326, i64 2, !dbg !34
  %1328 = fadd float %1324, %1326, !dbg !33
  %1329 = extractelement <4 x float> %1063, i64 3, !dbg !32
  %1330 = call float @llvm.amdgcn.exp2.f32(float %1329), !dbg !32
  %1331 = insertelement <4 x float> %1327, float %1330, i64 3, !dbg !34
  %math.result335 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1331) #9, !dbg !34, !frisk.register_lifetimes !1
  %1332 = fptrunc <4 x float> %math.result335 to <4 x half>, !dbg !34
  %math.packed336 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1332) #9, !dbg !33, !frisk.register_lifetimes !1
  %1333 = fadd float %1328, %1330, !dbg !33
  %1334 = extractelement <4 x float> %1064, i64 0, !dbg !32
  %1335 = call float @llvm.amdgcn.exp2.f32(float %1334), !dbg !32
  %1336 = insertelement <4 x float> poison, float %1335, i64 0, !dbg !34
  %1337 = fadd float %1333, %1335, !dbg !33
  %1338 = extractelement <4 x float> %1064, i64 1, !dbg !32
  %1339 = call float @llvm.amdgcn.exp2.f32(float %1338), !dbg !32
  %1340 = insertelement <4 x float> %1336, float %1339, i64 1, !dbg !34
  %1341 = fadd float %1337, %1339, !dbg !33
  %1342 = extractelement <4 x float> %1064, i64 2, !dbg !32
  %1343 = call float @llvm.amdgcn.exp2.f32(float %1342), !dbg !32
  %1344 = insertelement <4 x float> %1340, float %1343, i64 2, !dbg !34
  %1345 = fadd float %1341, %1343, !dbg !33
  %1346 = extractelement <4 x float> %1064, i64 3, !dbg !32
  %1347 = call float @llvm.amdgcn.exp2.f32(float %1346), !dbg !32
  %1348 = insertelement <4 x float> %1344, float %1347, i64 3, !dbg !34
  %math.result337 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1348) #9, !dbg !34, !frisk.register_lifetimes !1
  %1349 = fptrunc <4 x float> %math.result337 to <4 x half>, !dbg !34
  %math.packed338 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1349) #9, !dbg !33, !frisk.register_lifetimes !1
  %1350 = fadd float %1345, %1347, !dbg !33
  %1351 = bitcast float %1215 to i32, !dbg !33
  %1352 = call i32 @llvm.amdgcn.ds.bpermute(i32 %988, i32 %1351), !dbg !33
  %1353 = bitcast i32 %1352 to float, !dbg !33
  %1354 = fadd float %1215, %1353, !dbg !33
  %1355 = bitcast float %1354 to i32, !dbg !33
  %1356 = call i32 @llvm.amdgcn.ds.bpermute(i32 %996, i32 %1355), !dbg !33
  %1357 = bitcast i32 %1356 to float, !dbg !33
  %1358 = fadd float %1354, %1357, !dbg !33
  %1359 = fadd float %1358, %1072, !dbg !33
  %1360 = insertelement <1 x float> poison, float %1359, i64 0, !dbg !7
  %1361 = insertvalue [2 x <1 x float>] poison, <1 x float> %1360, 0, !dbg !7
  %1362 = bitcast float %1350 to i32, !dbg !33
  %1363 = call i32 @llvm.amdgcn.ds.bpermute(i32 %988, i32 %1362), !dbg !33
  %1364 = bitcast i32 %1363 to float, !dbg !33
  %1365 = fadd float %1350, %1364, !dbg !33
  %1366 = bitcast float %1365 to i32, !dbg !33
  %1367 = call i32 @llvm.amdgcn.ds.bpermute(i32 %996, i32 %1366), !dbg !33
  %1368 = bitcast i32 %1367 to float, !dbg !33
  %1369 = fadd float %1365, %1368, !dbg !33
  %1370 = fadd float %1369, %1080, !dbg !33
  %1371 = insertelement <1 x float> poison, float %1370, i64 0, !dbg !7
  %1372 = insertvalue [2 x <1 x float>] %1361, <1 x float> %1371, 1, !dbg !7
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
  %1373 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local871, i32 0) #0, !dbg !14
  %1374 = bitcast <4 x i32> %1373 to <8 x half>, !dbg !14
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
  %1375 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local884, i32 0) #0, !dbg !14
  %1376 = bitcast <4 x i32> %1375 to <8 x half>, !dbg !14
  %address.local885 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local875, !dbg !14
  %address.local886 = getelementptr [2 x i8], ptr addrspace(3) %address.local885, i64 %address.local879, !dbg !14
  %address.local887 = getelementptr [2 x i8], ptr addrspace(3) %address.local886, i64 %address.local882, !dbg !14
  %address.local888 = getelementptr i8, ptr addrspace(3) %address.local887, i64 512, !dbg !14
  %1377 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local888, i32 0) #0, !dbg !14
  %1378 = bitcast <4 x i32> %1377 to <8 x half>, !dbg !14
  %address.local889 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local875, !dbg !14
  %address.local890 = getelementptr [2 x i8], ptr addrspace(3) %address.local889, i64 %address.local879, !dbg !14
  %address.local891 = getelementptr [2 x i8], ptr addrspace(3) %address.local890, i64 %address.local882, !dbg !14
  %address.local892 = getelementptr i8, ptr addrspace(3) %address.local891, i64 768, !dbg !14
  %1379 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local892, i32 0) #0, !dbg !14
  %1380 = bitcast <4 x i32> %1379 to <8 x half>, !dbg !14
  %address.local893 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local870, !dbg !14
  %address.local894 = getelementptr i8, ptr addrspace(3) %address.local893, i64 4096, !dbg !14
  %1381 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local894, i32 0) #0, !dbg !14
  %1382 = bitcast <4 x i32> %1381 to <8 x half>, !dbg !14
  %address.local895 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local875, !dbg !14
  %address.local896 = getelementptr [2 x i8], ptr addrspace(3) %address.local895, i64 %address.local879, !dbg !14
  %address.local897 = getelementptr [2 x i8], ptr addrspace(3) %address.local896, i64 %address.local882, !dbg !14
  %address.local898 = getelementptr i8, ptr addrspace(3) %address.local897, i64 4352, !dbg !14
  %1383 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local898, i32 0) #0, !dbg !14
  %1384 = bitcast <4 x i32> %1383 to <8 x half>, !dbg !14
  %address.local899 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local875, !dbg !14
  %address.local900 = getelementptr [2 x i8], ptr addrspace(3) %address.local899, i64 %address.local879, !dbg !14
  %address.local901 = getelementptr [2 x i8], ptr addrspace(3) %address.local900, i64 %address.local882, !dbg !14
  %address.local902 = getelementptr i8, ptr addrspace(3) %address.local901, i64 4608, !dbg !14
  %1385 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local902, i32 0) #0, !dbg !14
  %1386 = bitcast <4 x i32> %1385 to <8 x half>, !dbg !14
  %address.local903 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local875, !dbg !14
  %address.local904 = getelementptr [2 x i8], ptr addrspace(3) %address.local903, i64 %address.local879, !dbg !14
  %address.local905 = getelementptr [2 x i8], ptr addrspace(3) %address.local904, i64 %address.local882, !dbg !14
  %address.local906 = getelementptr i8, ptr addrspace(3) %address.local905, i64 4864, !dbg !14
  %1387 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local906, i32 0) #0, !dbg !14
  %1388 = bitcast <4 x i32> %1387 to <8 x half>, !dbg !14
  %1389 = add nuw nsw i64 %219, %16, !dbg !18
  %1390 = shl i64 %1389, 19, !dbg !18
  %1391 = and i64 %1390, 16252928, !dbg !18
  %1392 = add i64 %1391, %225, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1393 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %1392, !dbg !18
  %1394 = getelementptr i8, ptr addrspace(1) %1393, i64 64, !dbg !18
  %address.local907 = shl nuw nsw i64 %address.local872, 3, !dbg !8
  %address.local908 = and i64 %address.local907, 1536, !dbg !8
  %address.local909 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local908, !dbg !8
  %address.local910 = ptrtoint ptr addrspace(3) %address.local909 to i64, !dbg !8
  %address.local911 = trunc i64 %address.local910 to i32, !dbg !8
  %1395 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local911), !dbg !18
  %1396 = zext i32 %1395 to i64, !dbg !18
  %1397 = inttoptr i64 %1396 to ptr addrspace(3), !dbg !18
  %address.local912 = shl nuw nsw i64 %address.local872, 5, !dbg !18
  %address.local913 = and i64 %address.local912, 8064, !dbg !18
  %address.local914 = shl nuw nsw i64 %address.local872, 3, !dbg !18
  %address.local915 = and i64 %address.local914, 24, !dbg !18
  %address.local916 = or disjoint i64 %address.local913, %address.local915, !dbg !18
  %address.local917 = getelementptr [2 x i8], ptr addrspace(1) %1394, i64 %address.local916, !dbg !18
  %1398 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1399 = ptrtoint ptr addrspace(1) %address.local917 to i64, !dbg !18
  %1400 = sub i64 %1399, %1398, !dbg !18
  %dma.buffer.offset1558 = trunc i64 %1400 to i32, !dbg !18
  %1401 = trunc i64 %1398 to i32, !dbg !18
  %1402 = insertelement <4 x i32> poison, i32 %1401, i64 0, !dbg !18
  %1403 = lshr i64 %1398, 32, !dbg !18
  %1404 = trunc i64 %1403 to i32, !dbg !18
  %1405 = insertelement <4 x i32> %1402, i32 %1404, i64 1, !dbg !18
  %1406 = insertelement <4 x i32> %1405, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1559 = insertelement <4 x i32> %1406, i32 131072, i64 3, !dbg !18
  %1407 = ptrtoint ptr addrspace(3) %1397 to i32, !dbg !18
  %dma.buffer.lds1560 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1407), !dbg !18
  %1408 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1560, i32 %dma.buffer.offset1558, <4 x i32> %dma.buffer.rsrc1559) #10, !dbg !18
  %address.local918 = or disjoint i64 %address.local872, 256, !dbg !8
  %address.local919 = shl nuw nsw i64 %address.local918, 3, !dbg !8
  %address.local920 = and i64 %address.local919, 3584, !dbg !8
  %address.local921 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local920, !dbg !8
  %address.local922 = ptrtoint ptr addrspace(3) %address.local921 to i64, !dbg !8
  %address.local923 = trunc i64 %address.local922 to i32, !dbg !8
  %1409 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local923), !dbg !18
  %1410 = zext i32 %1409 to i64, !dbg !18
  %1411 = inttoptr i64 %1410 to ptr addrspace(3), !dbg !18
  %address.local924 = shl nuw nsw i64 %address.local918, 5, !dbg !18
  %address.local925 = and i64 %address.local924, 16256, !dbg !18
  %address.local926 = shl nuw nsw i64 %address.local872, 3, !dbg !18
  %address.local927 = and i64 %address.local926, 24, !dbg !18
  %address.local928 = or disjoint i64 %address.local925, %address.local927, !dbg !18
  %address.local929 = getelementptr [2 x i8], ptr addrspace(1) %1394, i64 %address.local928, !dbg !18
  %1412 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1413 = ptrtoint ptr addrspace(1) %address.local929 to i64, !dbg !18
  %1414 = sub i64 %1413, %1412, !dbg !18
  %dma.buffer.offset1555 = trunc i64 %1414 to i32, !dbg !18
  %1415 = trunc i64 %1412 to i32, !dbg !18
  %1416 = insertelement <4 x i32> poison, i32 %1415, i64 0, !dbg !18
  %1417 = lshr i64 %1412, 32, !dbg !18
  %1418 = trunc i64 %1417 to i32, !dbg !18
  %1419 = insertelement <4 x i32> %1416, i32 %1418, i64 1, !dbg !18
  %1420 = insertelement <4 x i32> %1419, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1556 = insertelement <4 x i32> %1420, i32 131072, i64 3, !dbg !18
  %1421 = ptrtoint ptr addrspace(3) %1411 to i32, !dbg !18
  %dma.buffer.lds1557 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1421), !dbg !18
  %1422 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1557, i32 %dma.buffer.offset1555, <4 x i32> %dma.buffer.rsrc1556) #10, !dbg !18
  %1423 = shufflevector <8 x half> %1374, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1424 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1423, <4 x float> zeroinitializer) #9, !dbg !14
  %1425 = shufflevector <8 x half> %1376, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1426 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %1425, <4 x float> %1424) #9, !dbg !14
  %1427 = shufflevector <8 x half> %1378, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1428 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %1427, <4 x float> %1426) #9, !dbg !14
  %1429 = shufflevector <8 x half> %1380, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1430 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %1429, <4 x float> %1428) #9, !dbg !14
  %1431 = shufflevector <8 x half> %1382, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1432 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %1431, <4 x float> %1430) #9, !dbg !14
  %1433 = shufflevector <8 x half> %1384, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1434 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %1433, <4 x float> %1432) #9, !dbg !14
  %1435 = shufflevector <8 x half> %1386, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1436 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %1435, <4 x float> %1434) #9, !dbg !14
  %1437 = shufflevector <8 x half> %1388, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1438 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %1437, <4 x float> %1436) #9, !dbg !14
  %1439 = extractelement <4 x float> %1438, i64 0, !dbg !14
  %1440 = extractelement <4 x float> %1438, i64 1, !dbg !14
  %1441 = extractelement <4 x float> %1438, i64 2, !dbg !14
  %1442 = extractelement <4 x float> %1438, i64 3, !dbg !14
  %1443 = select i1 %.not, float %1440, float %1439, !dbg !14
  %1444 = bitcast float %1443 to i32, !dbg !14
  %1445 = call i32 @llvm.amdgcn.ds.bpermute(i32 %988, i32 %1444), !dbg !14
  %1446 = bitcast i32 %1445 to float, !dbg !14
  %1447 = select i1 %.not, float %1439, float %1446, !dbg !14
  %1448 = select i1 %.not, float %1446, float %1440, !dbg !14
  %1449 = select i1 %.not, float %1442, float %1441, !dbg !14
  %1450 = bitcast float %1449 to i32, !dbg !14
  %1451 = call i32 @llvm.amdgcn.ds.bpermute(i32 %988, i32 %1450), !dbg !14
  %1452 = bitcast i32 %1451 to float, !dbg !14
  %1453 = select i1 %.not, float %1441, float %1452, !dbg !14
  %1454 = select i1 %.not, float %1452, float %1442, !dbg !14
  %1455 = select i1 %.not1, float %1453, float %1447, !dbg !14
  %1456 = bitcast float %1455 to i32, !dbg !14
  %1457 = call i32 @llvm.amdgcn.ds.bpermute(i32 %996, i32 %1456), !dbg !14
  %1458 = bitcast i32 %1457 to float, !dbg !14
  %1459 = select i1 %.not1, float %1447, float %1458, !dbg !14
  %1460 = select i1 %.not1, float %1458, float %1453, !dbg !14
  %1461 = select i1 %.not1, float %1454, float %1448, !dbg !14
  %1462 = bitcast float %1461 to i32, !dbg !14
  %1463 = call i32 @llvm.amdgcn.ds.bpermute(i32 %996, i32 %1462), !dbg !14
  %1464 = bitcast i32 %1463 to float, !dbg !14
  %1465 = select i1 %.not1, float %1448, float %1464, !dbg !14
  %1466 = select i1 %.not1, float %1464, float %1454, !dbg !14
  %1467 = extractvalue [2 x <8 x float>] %208, 0, !dbg !35
  %1468 = shufflevector <8 x float> %1467, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1469 = fmul <4 x float> %1468, %1069, !dbg !35
  %1470 = insertelement <4 x float> poison, float %1459, i64 0, !dbg !36
  %1471 = insertelement <4 x float> %1470, float %1465, i64 1, !dbg !36
  %1472 = insertelement <4 x float> %1471, float %1460, i64 2, !dbg !36
  %1473 = insertelement <4 x float> %1472, float %1466, i64 3, !dbg !36
  %1474 = fadd <4 x float> %1469, %1473, !dbg !36
  %1475 = shufflevector <8 x half> %1374, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1476 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1475, <4 x float> zeroinitializer) #9, !dbg !14
  %1477 = shufflevector <8 x half> %1376, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1478 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %1477, <4 x float> %1476) #9, !dbg !14
  %1479 = shufflevector <8 x half> %1378, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1480 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %1479, <4 x float> %1478) #9, !dbg !14
  %1481 = shufflevector <8 x half> %1380, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1482 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %1481, <4 x float> %1480) #9, !dbg !14
  %1483 = shufflevector <8 x half> %1382, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1484 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %1483, <4 x float> %1482) #9, !dbg !14
  %1485 = shufflevector <8 x half> %1384, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1486 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %1485, <4 x float> %1484) #9, !dbg !14
  %1487 = shufflevector <8 x half> %1386, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1488 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %1487, <4 x float> %1486) #9, !dbg !14
  %1489 = shufflevector <8 x half> %1388, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1490 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %1489, <4 x float> %1488) #9, !dbg !14
  %1491 = extractelement <4 x float> %1490, i64 0, !dbg !14
  %1492 = extractelement <4 x float> %1490, i64 1, !dbg !14
  %1493 = extractelement <4 x float> %1490, i64 2, !dbg !14
  %1494 = extractelement <4 x float> %1490, i64 3, !dbg !14
  %1495 = select i1 %.not, float %1492, float %1491, !dbg !14
  %1496 = bitcast float %1495 to i32, !dbg !14
  %1497 = call i32 @llvm.amdgcn.ds.bpermute(i32 %988, i32 %1496), !dbg !14
  %1498 = bitcast i32 %1497 to float, !dbg !14
  %1499 = select i1 %.not, float %1491, float %1498, !dbg !14
  %1500 = select i1 %.not, float %1498, float %1492, !dbg !14
  %1501 = select i1 %.not, float %1494, float %1493, !dbg !14
  %1502 = bitcast float %1501 to i32, !dbg !14
  %1503 = call i32 @llvm.amdgcn.ds.bpermute(i32 %988, i32 %1502), !dbg !14
  %1504 = bitcast i32 %1503 to float, !dbg !14
  %1505 = select i1 %.not, float %1493, float %1504, !dbg !14
  %1506 = select i1 %.not, float %1504, float %1494, !dbg !14
  %1507 = select i1 %.not1, float %1505, float %1499, !dbg !14
  %1508 = bitcast float %1507 to i32, !dbg !14
  %1509 = call i32 @llvm.amdgcn.ds.bpermute(i32 %996, i32 %1508), !dbg !14
  %1510 = bitcast i32 %1509 to float, !dbg !14
  %1511 = select i1 %.not1, float %1499, float %1510, !dbg !14
  %1512 = select i1 %.not1, float %1510, float %1505, !dbg !14
  %1513 = select i1 %.not1, float %1506, float %1500, !dbg !14
  %1514 = bitcast float %1513 to i32, !dbg !14
  %1515 = call i32 @llvm.amdgcn.ds.bpermute(i32 %996, i32 %1514), !dbg !14
  %1516 = bitcast i32 %1515 to float, !dbg !14
  %1517 = select i1 %.not1, float %1500, float %1516, !dbg !14
  %1518 = select i1 %.not1, float %1516, float %1506, !dbg !14
  %1519 = shufflevector <8 x float> %1467, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1520 = fmul <4 x float> %1519, %1069, !dbg !35
  %1521 = insertelement <4 x float> poison, float %1511, i64 0, !dbg !36
  %1522 = insertelement <4 x float> %1521, float %1517, i64 1, !dbg !36
  %1523 = insertelement <4 x float> %1522, float %1512, i64 2, !dbg !36
  %1524 = insertelement <4 x float> %1523, float %1518, i64 3, !dbg !36
  %1525 = fadd <4 x float> %1520, %1524, !dbg !36
  %1526 = shufflevector <4 x float> %1474, <4 x float> %1525, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1527 = insertvalue [2 x <8 x float>] poison, <8 x float> %1526, 0, !dbg !7
  %1528 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1423, <4 x float> zeroinitializer) #9, !dbg !14
  %1529 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1425, <4 x float> %1528) #9, !dbg !14
  %1530 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1427, <4 x float> %1529) #9, !dbg !14
  %1531 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1429, <4 x float> %1530) #9, !dbg !14
  %1532 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %1431, <4 x float> %1531) #9, !dbg !14
  %1533 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %1433, <4 x float> %1532) #9, !dbg !14
  %1534 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %1435, <4 x float> %1533) #9, !dbg !14
  %1535 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %1437, <4 x float> %1534) #9, !dbg !14
  %1536 = extractelement <4 x float> %1535, i64 0, !dbg !14
  %1537 = extractelement <4 x float> %1535, i64 1, !dbg !14
  %1538 = extractelement <4 x float> %1535, i64 2, !dbg !14
  %1539 = extractelement <4 x float> %1535, i64 3, !dbg !14
  %1540 = select i1 %.not, float %1537, float %1536, !dbg !14
  %1541 = bitcast float %1540 to i32, !dbg !14
  %1542 = call i32 @llvm.amdgcn.ds.bpermute(i32 %988, i32 %1541), !dbg !14
  %1543 = bitcast i32 %1542 to float, !dbg !14
  %1544 = select i1 %.not, float %1536, float %1543, !dbg !14
  %1545 = select i1 %.not, float %1543, float %1537, !dbg !14
  %1546 = select i1 %.not, float %1539, float %1538, !dbg !14
  %1547 = bitcast float %1546 to i32, !dbg !14
  %1548 = call i32 @llvm.amdgcn.ds.bpermute(i32 %988, i32 %1547), !dbg !14
  %1549 = bitcast i32 %1548 to float, !dbg !14
  %1550 = select i1 %.not, float %1538, float %1549, !dbg !14
  %1551 = select i1 %.not, float %1549, float %1539, !dbg !14
  %1552 = select i1 %.not1, float %1550, float %1544, !dbg !14
  %1553 = bitcast float %1552 to i32, !dbg !14
  %1554 = call i32 @llvm.amdgcn.ds.bpermute(i32 %996, i32 %1553), !dbg !14
  %1555 = bitcast i32 %1554 to float, !dbg !14
  %1556 = select i1 %.not1, float %1544, float %1555, !dbg !14
  %1557 = select i1 %.not1, float %1555, float %1550, !dbg !14
  %1558 = select i1 %.not1, float %1551, float %1545, !dbg !14
  %1559 = bitcast float %1558 to i32, !dbg !14
  %1560 = call i32 @llvm.amdgcn.ds.bpermute(i32 %996, i32 %1559), !dbg !14
  %1561 = bitcast i32 %1560 to float, !dbg !14
  %1562 = select i1 %.not1, float %1545, float %1561, !dbg !14
  %1563 = select i1 %.not1, float %1561, float %1551, !dbg !14
  %1564 = extractvalue [2 x <8 x float>] %208, 1, !dbg !35
  %1565 = shufflevector <8 x float> %1564, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1566 = fmul <4 x float> %1565, %1077, !dbg !35
  %1567 = insertelement <4 x float> poison, float %1556, i64 0, !dbg !36
  %1568 = insertelement <4 x float> %1567, float %1562, i64 1, !dbg !36
  %1569 = insertelement <4 x float> %1568, float %1557, i64 2, !dbg !36
  %1570 = insertelement <4 x float> %1569, float %1563, i64 3, !dbg !36
  %1571 = fadd <4 x float> %1566, %1570, !dbg !36
  %1572 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1475, <4 x float> zeroinitializer) #9, !dbg !14
  %1573 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1477, <4 x float> %1572) #9, !dbg !14
  %1574 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1479, <4 x float> %1573) #9, !dbg !14
  %1575 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1481, <4 x float> %1574) #9, !dbg !14
  %1576 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %1483, <4 x float> %1575) #9, !dbg !14
  %1577 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %1485, <4 x float> %1576) #9, !dbg !14
  %1578 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %1487, <4 x float> %1577) #9, !dbg !14
  %1579 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %1489, <4 x float> %1578) #9, !dbg !14
  %1580 = extractelement <4 x float> %1579, i64 0, !dbg !14
  %1581 = extractelement <4 x float> %1579, i64 1, !dbg !14
  %1582 = extractelement <4 x float> %1579, i64 2, !dbg !14
  %1583 = extractelement <4 x float> %1579, i64 3, !dbg !14
  %1584 = select i1 %.not, float %1581, float %1580, !dbg !14
  %1585 = bitcast float %1584 to i32, !dbg !14
  %1586 = call i32 @llvm.amdgcn.ds.bpermute(i32 %988, i32 %1585), !dbg !14
  %1587 = bitcast i32 %1586 to float, !dbg !14
  %1588 = select i1 %.not, float %1580, float %1587, !dbg !14
  %1589 = select i1 %.not, float %1587, float %1581, !dbg !14
  %1590 = select i1 %.not, float %1583, float %1582, !dbg !14
  %1591 = bitcast float %1590 to i32, !dbg !14
  %1592 = call i32 @llvm.amdgcn.ds.bpermute(i32 %988, i32 %1591), !dbg !14
  %1593 = bitcast i32 %1592 to float, !dbg !14
  %1594 = select i1 %.not, float %1582, float %1593, !dbg !14
  %1595 = select i1 %.not, float %1593, float %1583, !dbg !14
  %1596 = select i1 %.not1, float %1594, float %1588, !dbg !14
  %1597 = bitcast float %1596 to i32, !dbg !14
  %1598 = call i32 @llvm.amdgcn.ds.bpermute(i32 %996, i32 %1597), !dbg !14
  %1599 = bitcast i32 %1598 to float, !dbg !14
  %1600 = select i1 %.not1, float %1588, float %1599, !dbg !14
  %1601 = select i1 %.not1, float %1599, float %1594, !dbg !14
  %1602 = select i1 %.not1, float %1595, float %1589, !dbg !14
  %1603 = bitcast float %1602 to i32, !dbg !14
  %1604 = call i32 @llvm.amdgcn.ds.bpermute(i32 %996, i32 %1603), !dbg !14
  %1605 = bitcast i32 %1604 to float, !dbg !14
  %1606 = select i1 %.not1, float %1589, float %1605, !dbg !14
  %1607 = select i1 %.not1, float %1605, float %1595, !dbg !14
  %1608 = shufflevector <8 x float> %1564, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1609 = fmul <4 x float> %1608, %1077, !dbg !35
  %1610 = insertelement <4 x float> poison, float %1600, i64 0, !dbg !36
  %1611 = insertelement <4 x float> %1610, float %1606, i64 1, !dbg !36
  %1612 = insertelement <4 x float> %1611, float %1601, i64 2, !dbg !36
  %1613 = insertelement <4 x float> %1612, float %1607, i64 3, !dbg !36
  %1614 = fadd <4 x float> %1609, %1613, !dbg !36
  %1615 = shufflevector <4 x float> %1571, <4 x float> %1614, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1616 = insertvalue [2 x <8 x float>] %1527, <8 x float> %1615, 1, !dbg !7
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1617 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local871, i32 0) #0, !dbg !14
  %1618 = bitcast <4 x i32> %1617 to <8 x half>, !dbg !14
  %1619 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local884, i32 0) #0, !dbg !14
  %1620 = bitcast <4 x i32> %1619 to <8 x half>, !dbg !14
  %1621 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local888, i32 0) #0, !dbg !14
  %1622 = bitcast <4 x i32> %1621 to <8 x half>, !dbg !14
  %1623 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local892, i32 0) #0, !dbg !14
  %1624 = bitcast <4 x i32> %1623 to <8 x half>, !dbg !14
  %1625 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local894, i32 0) #0, !dbg !14
  %1626 = bitcast <4 x i32> %1625 to <8 x half>, !dbg !14
  %1627 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local898, i32 0) #0, !dbg !14
  %1628 = bitcast <4 x i32> %1627 to <8 x half>, !dbg !14
  %1629 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local902, i32 0) #0, !dbg !14
  %1630 = bitcast <4 x i32> %1629 to <8 x half>, !dbg !14
  %1631 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local906, i32 0) #0, !dbg !14
  %1632 = bitcast <4 x i32> %1631 to <8 x half>, !dbg !14
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1633 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %1392, !dbg !18
  %1634 = getelementptr i8, ptr addrspace(1) %1633, i64 128, !dbg !18
  %1635 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local911), !dbg !18
  %1636 = zext i32 %1635 to i64, !dbg !18
  %1637 = inttoptr i64 %1636 to ptr addrspace(3), !dbg !18
  %address.local930 = getelementptr [2 x i8], ptr addrspace(1) %1634, i64 %address.local916, !dbg !18
  %1638 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1639 = ptrtoint ptr addrspace(1) %address.local930 to i64, !dbg !18
  %1640 = sub i64 %1639, %1638, !dbg !18
  %dma.buffer.offset1552 = trunc i64 %1640 to i32, !dbg !18
  %1641 = trunc i64 %1638 to i32, !dbg !18
  %1642 = insertelement <4 x i32> poison, i32 %1641, i64 0, !dbg !18
  %1643 = lshr i64 %1638, 32, !dbg !18
  %1644 = trunc i64 %1643 to i32, !dbg !18
  %1645 = insertelement <4 x i32> %1642, i32 %1644, i64 1, !dbg !18
  %1646 = insertelement <4 x i32> %1645, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1553 = insertelement <4 x i32> %1646, i32 131072, i64 3, !dbg !18
  %1647 = ptrtoint ptr addrspace(3) %1637 to i32, !dbg !18
  %dma.buffer.lds1554 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1647), !dbg !18
  %1648 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1554, i32 %dma.buffer.offset1552, <4 x i32> %dma.buffer.rsrc1553) #10, !dbg !18
  %1649 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local923), !dbg !18
  %1650 = zext i32 %1649 to i64, !dbg !18
  %1651 = inttoptr i64 %1650 to ptr addrspace(3), !dbg !18
  %address.local931 = getelementptr [2 x i8], ptr addrspace(1) %1634, i64 %address.local928, !dbg !18
  %1652 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1653 = ptrtoint ptr addrspace(1) %address.local931 to i64, !dbg !18
  %1654 = sub i64 %1653, %1652, !dbg !18
  %dma.buffer.offset1549 = trunc i64 %1654 to i32, !dbg !18
  %1655 = trunc i64 %1652 to i32, !dbg !18
  %1656 = insertelement <4 x i32> poison, i32 %1655, i64 0, !dbg !18
  %1657 = lshr i64 %1652, 32, !dbg !18
  %1658 = trunc i64 %1657 to i32, !dbg !18
  %1659 = insertelement <4 x i32> %1656, i32 %1658, i64 1, !dbg !18
  %1660 = insertelement <4 x i32> %1659, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1550 = insertelement <4 x i32> %1660, i32 131072, i64 3, !dbg !18
  %1661 = ptrtoint ptr addrspace(3) %1651 to i32, !dbg !18
  %dma.buffer.lds1551 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1661), !dbg !18
  %1662 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1551, i32 %dma.buffer.offset1549, <4 x i32> %dma.buffer.rsrc1550) #10, !dbg !18
  %1663 = shufflevector <8 x half> %1618, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1664 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1663, <4 x float> zeroinitializer) #9, !dbg !14
  %1665 = shufflevector <8 x half> %1620, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1666 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %1665, <4 x float> %1664) #9, !dbg !14
  %1667 = shufflevector <8 x half> %1622, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1668 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %1667, <4 x float> %1666) #9, !dbg !14
  %1669 = shufflevector <8 x half> %1624, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1670 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %1669, <4 x float> %1668) #9, !dbg !14
  %1671 = shufflevector <8 x half> %1626, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1672 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %1671, <4 x float> %1670) #9, !dbg !14
  %1673 = shufflevector <8 x half> %1628, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1674 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %1673, <4 x float> %1672) #9, !dbg !14
  %1675 = shufflevector <8 x half> %1630, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1676 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %1675, <4 x float> %1674) #9, !dbg !14
  %1677 = shufflevector <8 x half> %1632, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1678 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %1677, <4 x float> %1676) #9, !dbg !14
  %1679 = extractelement <4 x float> %1678, i64 0, !dbg !14
  %1680 = extractelement <4 x float> %1678, i64 1, !dbg !14
  %1681 = extractelement <4 x float> %1678, i64 2, !dbg !14
  %1682 = extractelement <4 x float> %1678, i64 3, !dbg !14
  %1683 = select i1 %.not, float %1680, float %1679, !dbg !14
  %1684 = bitcast float %1683 to i32, !dbg !14
  %1685 = call i32 @llvm.amdgcn.ds.bpermute(i32 %988, i32 %1684), !dbg !14
  %1686 = bitcast i32 %1685 to float, !dbg !14
  %1687 = select i1 %.not, float %1679, float %1686, !dbg !14
  %1688 = select i1 %.not, float %1686, float %1680, !dbg !14
  %1689 = select i1 %.not, float %1682, float %1681, !dbg !14
  %1690 = bitcast float %1689 to i32, !dbg !14
  %1691 = call i32 @llvm.amdgcn.ds.bpermute(i32 %988, i32 %1690), !dbg !14
  %1692 = bitcast i32 %1691 to float, !dbg !14
  %1693 = select i1 %.not, float %1681, float %1692, !dbg !14
  %1694 = select i1 %.not, float %1692, float %1682, !dbg !14
  %1695 = select i1 %.not1, float %1693, float %1687, !dbg !14
  %1696 = bitcast float %1695 to i32, !dbg !14
  %1697 = call i32 @llvm.amdgcn.ds.bpermute(i32 %996, i32 %1696), !dbg !14
  %1698 = bitcast i32 %1697 to float, !dbg !14
  %1699 = select i1 %.not1, float %1687, float %1698, !dbg !14
  %1700 = select i1 %.not1, float %1698, float %1693, !dbg !14
  %1701 = select i1 %.not1, float %1694, float %1688, !dbg !14
  %1702 = bitcast float %1701 to i32, !dbg !14
  %1703 = call i32 @llvm.amdgcn.ds.bpermute(i32 %996, i32 %1702), !dbg !14
  %1704 = bitcast i32 %1703 to float, !dbg !14
  %1705 = select i1 %.not1, float %1688, float %1704, !dbg !14
  %1706 = select i1 %.not1, float %1704, float %1694, !dbg !14
  %1707 = extractvalue [2 x <8 x float>] %211, 0, !dbg !35
  %1708 = shufflevector <8 x float> %1707, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1709 = fmul <4 x float> %1708, %1069, !dbg !35
  %1710 = insertelement <4 x float> poison, float %1699, i64 0, !dbg !36
  %1711 = insertelement <4 x float> %1710, float %1705, i64 1, !dbg !36
  %1712 = insertelement <4 x float> %1711, float %1700, i64 2, !dbg !36
  %1713 = insertelement <4 x float> %1712, float %1706, i64 3, !dbg !36
  %1714 = fadd <4 x float> %1709, %1713, !dbg !36
  %1715 = shufflevector <8 x half> %1618, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1716 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1715, <4 x float> zeroinitializer) #9, !dbg !14
  %1717 = shufflevector <8 x half> %1620, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1718 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %1717, <4 x float> %1716) #9, !dbg !14
  %1719 = shufflevector <8 x half> %1622, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1720 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %1719, <4 x float> %1718) #9, !dbg !14
  %1721 = shufflevector <8 x half> %1624, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1722 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %1721, <4 x float> %1720) #9, !dbg !14
  %1723 = shufflevector <8 x half> %1626, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1724 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %1723, <4 x float> %1722) #9, !dbg !14
  %1725 = shufflevector <8 x half> %1628, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1726 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %1725, <4 x float> %1724) #9, !dbg !14
  %1727 = shufflevector <8 x half> %1630, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1728 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %1727, <4 x float> %1726) #9, !dbg !14
  %1729 = shufflevector <8 x half> %1632, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1730 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %1729, <4 x float> %1728) #9, !dbg !14
  %1731 = extractelement <4 x float> %1730, i64 0, !dbg !14
  %1732 = extractelement <4 x float> %1730, i64 1, !dbg !14
  %1733 = extractelement <4 x float> %1730, i64 2, !dbg !14
  %1734 = extractelement <4 x float> %1730, i64 3, !dbg !14
  %1735 = select i1 %.not, float %1732, float %1731, !dbg !14
  %1736 = bitcast float %1735 to i32, !dbg !14
  %1737 = call i32 @llvm.amdgcn.ds.bpermute(i32 %988, i32 %1736), !dbg !14
  %1738 = bitcast i32 %1737 to float, !dbg !14
  %1739 = select i1 %.not, float %1731, float %1738, !dbg !14
  %1740 = select i1 %.not, float %1738, float %1732, !dbg !14
  %1741 = select i1 %.not, float %1734, float %1733, !dbg !14
  %1742 = bitcast float %1741 to i32, !dbg !14
  %1743 = call i32 @llvm.amdgcn.ds.bpermute(i32 %988, i32 %1742), !dbg !14
  %1744 = bitcast i32 %1743 to float, !dbg !14
  %1745 = select i1 %.not, float %1733, float %1744, !dbg !14
  %1746 = select i1 %.not, float %1744, float %1734, !dbg !14
  %1747 = select i1 %.not1, float %1745, float %1739, !dbg !14
  %1748 = bitcast float %1747 to i32, !dbg !14
  %1749 = call i32 @llvm.amdgcn.ds.bpermute(i32 %996, i32 %1748), !dbg !14
  %1750 = bitcast i32 %1749 to float, !dbg !14
  %1751 = select i1 %.not1, float %1739, float %1750, !dbg !14
  %1752 = select i1 %.not1, float %1750, float %1745, !dbg !14
  %1753 = select i1 %.not1, float %1746, float %1740, !dbg !14
  %1754 = bitcast float %1753 to i32, !dbg !14
  %1755 = call i32 @llvm.amdgcn.ds.bpermute(i32 %996, i32 %1754), !dbg !14
  %1756 = bitcast i32 %1755 to float, !dbg !14
  %1757 = select i1 %.not1, float %1740, float %1756, !dbg !14
  %1758 = select i1 %.not1, float %1756, float %1746, !dbg !14
  %1759 = shufflevector <8 x float> %1707, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1760 = fmul <4 x float> %1759, %1069, !dbg !35
  %1761 = insertelement <4 x float> poison, float %1751, i64 0, !dbg !36
  %1762 = insertelement <4 x float> %1761, float %1757, i64 1, !dbg !36
  %1763 = insertelement <4 x float> %1762, float %1752, i64 2, !dbg !36
  %1764 = insertelement <4 x float> %1763, float %1758, i64 3, !dbg !36
  %1765 = fadd <4 x float> %1760, %1764, !dbg !36
  %1766 = shufflevector <4 x float> %1714, <4 x float> %1765, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1767 = insertvalue [2 x <8 x float>] poison, <8 x float> %1766, 0, !dbg !7
  %1768 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1663, <4 x float> zeroinitializer) #9, !dbg !14
  %1769 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1665, <4 x float> %1768) #9, !dbg !14
  %1770 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1667, <4 x float> %1769) #9, !dbg !14
  %1771 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1669, <4 x float> %1770) #9, !dbg !14
  %1772 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %1671, <4 x float> %1771) #9, !dbg !14
  %1773 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %1673, <4 x float> %1772) #9, !dbg !14
  %1774 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %1675, <4 x float> %1773) #9, !dbg !14
  %1775 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %1677, <4 x float> %1774) #9, !dbg !14
  %1776 = extractelement <4 x float> %1775, i64 0, !dbg !14
  %1777 = extractelement <4 x float> %1775, i64 1, !dbg !14
  %1778 = extractelement <4 x float> %1775, i64 2, !dbg !14
  %1779 = extractelement <4 x float> %1775, i64 3, !dbg !14
  %1780 = select i1 %.not, float %1777, float %1776, !dbg !14
  %1781 = bitcast float %1780 to i32, !dbg !14
  %1782 = call i32 @llvm.amdgcn.ds.bpermute(i32 %988, i32 %1781), !dbg !14
  %1783 = bitcast i32 %1782 to float, !dbg !14
  %1784 = select i1 %.not, float %1776, float %1783, !dbg !14
  %1785 = select i1 %.not, float %1783, float %1777, !dbg !14
  %1786 = select i1 %.not, float %1779, float %1778, !dbg !14
  %1787 = bitcast float %1786 to i32, !dbg !14
  %1788 = call i32 @llvm.amdgcn.ds.bpermute(i32 %988, i32 %1787), !dbg !14
  %1789 = bitcast i32 %1788 to float, !dbg !14
  %1790 = select i1 %.not, float %1778, float %1789, !dbg !14
  %1791 = select i1 %.not, float %1789, float %1779, !dbg !14
  %1792 = select i1 %.not1, float %1790, float %1784, !dbg !14
  %1793 = bitcast float %1792 to i32, !dbg !14
  %1794 = call i32 @llvm.amdgcn.ds.bpermute(i32 %996, i32 %1793), !dbg !14
  %1795 = bitcast i32 %1794 to float, !dbg !14
  %1796 = select i1 %.not1, float %1784, float %1795, !dbg !14
  %1797 = select i1 %.not1, float %1795, float %1790, !dbg !14
  %1798 = select i1 %.not1, float %1791, float %1785, !dbg !14
  %1799 = bitcast float %1798 to i32, !dbg !14
  %1800 = call i32 @llvm.amdgcn.ds.bpermute(i32 %996, i32 %1799), !dbg !14
  %1801 = bitcast i32 %1800 to float, !dbg !14
  %1802 = select i1 %.not1, float %1785, float %1801, !dbg !14
  %1803 = select i1 %.not1, float %1801, float %1791, !dbg !14
  %1804 = extractvalue [2 x <8 x float>] %211, 1, !dbg !35
  %1805 = shufflevector <8 x float> %1804, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1806 = fmul <4 x float> %1805, %1077, !dbg !35
  %1807 = insertelement <4 x float> poison, float %1796, i64 0, !dbg !36
  %1808 = insertelement <4 x float> %1807, float %1802, i64 1, !dbg !36
  %1809 = insertelement <4 x float> %1808, float %1797, i64 2, !dbg !36
  %1810 = insertelement <4 x float> %1809, float %1803, i64 3, !dbg !36
  %1811 = fadd <4 x float> %1806, %1810, !dbg !36
  %1812 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1715, <4 x float> zeroinitializer) #9, !dbg !14
  %1813 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1717, <4 x float> %1812) #9, !dbg !14
  %1814 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1719, <4 x float> %1813) #9, !dbg !14
  %1815 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1721, <4 x float> %1814) #9, !dbg !14
  %1816 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %1723, <4 x float> %1815) #9, !dbg !14
  %1817 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %1725, <4 x float> %1816) #9, !dbg !14
  %1818 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %1727, <4 x float> %1817) #9, !dbg !14
  %1819 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %1729, <4 x float> %1818) #9, !dbg !14
  %1820 = extractelement <4 x float> %1819, i64 0, !dbg !14
  %1821 = extractelement <4 x float> %1819, i64 1, !dbg !14
  %1822 = extractelement <4 x float> %1819, i64 2, !dbg !14
  %1823 = extractelement <4 x float> %1819, i64 3, !dbg !14
  %1824 = select i1 %.not, float %1821, float %1820, !dbg !14
  %1825 = bitcast float %1824 to i32, !dbg !14
  %1826 = call i32 @llvm.amdgcn.ds.bpermute(i32 %988, i32 %1825), !dbg !14
  %1827 = bitcast i32 %1826 to float, !dbg !14
  %1828 = select i1 %.not, float %1820, float %1827, !dbg !14
  %1829 = select i1 %.not, float %1827, float %1821, !dbg !14
  %1830 = select i1 %.not, float %1823, float %1822, !dbg !14
  %1831 = bitcast float %1830 to i32, !dbg !14
  %1832 = call i32 @llvm.amdgcn.ds.bpermute(i32 %988, i32 %1831), !dbg !14
  %1833 = bitcast i32 %1832 to float, !dbg !14
  %1834 = select i1 %.not, float %1822, float %1833, !dbg !14
  %1835 = select i1 %.not, float %1833, float %1823, !dbg !14
  %1836 = select i1 %.not1, float %1834, float %1828, !dbg !14
  %1837 = bitcast float %1836 to i32, !dbg !14
  %1838 = call i32 @llvm.amdgcn.ds.bpermute(i32 %996, i32 %1837), !dbg !14
  %1839 = bitcast i32 %1838 to float, !dbg !14
  %1840 = select i1 %.not1, float %1828, float %1839, !dbg !14
  %1841 = select i1 %.not1, float %1839, float %1834, !dbg !14
  %1842 = select i1 %.not1, float %1835, float %1829, !dbg !14
  %1843 = bitcast float %1842 to i32, !dbg !14
  %1844 = call i32 @llvm.amdgcn.ds.bpermute(i32 %996, i32 %1843), !dbg !14
  %1845 = bitcast i32 %1844 to float, !dbg !14
  %1846 = select i1 %.not1, float %1829, float %1845, !dbg !14
  %1847 = select i1 %.not1, float %1845, float %1835, !dbg !14
  %1848 = shufflevector <8 x float> %1804, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1849 = fmul <4 x float> %1848, %1077, !dbg !35
  %1850 = insertelement <4 x float> poison, float %1840, i64 0, !dbg !36
  %1851 = insertelement <4 x float> %1850, float %1846, i64 1, !dbg !36
  %1852 = insertelement <4 x float> %1851, float %1841, i64 2, !dbg !36
  %1853 = insertelement <4 x float> %1852, float %1847, i64 3, !dbg !36
  %1854 = fadd <4 x float> %1849, %1853, !dbg !36
  %1855 = shufflevector <4 x float> %1811, <4 x float> %1854, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1856 = insertvalue [2 x <8 x float>] %1767, <8 x float> %1855, 1, !dbg !7
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1857 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local871, i32 0) #0, !dbg !14
  %1858 = bitcast <4 x i32> %1857 to <8 x half>, !dbg !14
  %1859 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local884, i32 0) #0, !dbg !14
  %1860 = bitcast <4 x i32> %1859 to <8 x half>, !dbg !14
  %1861 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local888, i32 0) #0, !dbg !14
  %1862 = bitcast <4 x i32> %1861 to <8 x half>, !dbg !14
  %1863 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local892, i32 0) #0, !dbg !14
  %1864 = bitcast <4 x i32> %1863 to <8 x half>, !dbg !14
  %1865 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local894, i32 0) #0, !dbg !14
  %1866 = bitcast <4 x i32> %1865 to <8 x half>, !dbg !14
  %1867 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local898, i32 0) #0, !dbg !14
  %1868 = bitcast <4 x i32> %1867 to <8 x half>, !dbg !14
  %1869 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local902, i32 0) #0, !dbg !14
  %1870 = bitcast <4 x i32> %1869 to <8 x half>, !dbg !14
  %1871 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local906, i32 0) #0, !dbg !14
  %1872 = bitcast <4 x i32> %1871 to <8 x half>, !dbg !14
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1873 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %1392, !dbg !18
  %1874 = getelementptr i8, ptr addrspace(1) %1873, i64 192, !dbg !18
  %1875 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local911), !dbg !18
  %1876 = zext i32 %1875 to i64, !dbg !18
  %1877 = inttoptr i64 %1876 to ptr addrspace(3), !dbg !18
  %address.local932 = getelementptr [2 x i8], ptr addrspace(1) %1874, i64 %address.local916, !dbg !18
  %1878 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1879 = ptrtoint ptr addrspace(1) %address.local932 to i64, !dbg !18
  %1880 = sub i64 %1879, %1878, !dbg !18
  %dma.buffer.offset1546 = trunc i64 %1880 to i32, !dbg !18
  %1881 = trunc i64 %1878 to i32, !dbg !18
  %1882 = insertelement <4 x i32> poison, i32 %1881, i64 0, !dbg !18
  %1883 = lshr i64 %1878, 32, !dbg !18
  %1884 = trunc i64 %1883 to i32, !dbg !18
  %1885 = insertelement <4 x i32> %1882, i32 %1884, i64 1, !dbg !18
  %1886 = insertelement <4 x i32> %1885, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1547 = insertelement <4 x i32> %1886, i32 131072, i64 3, !dbg !18
  %1887 = ptrtoint ptr addrspace(3) %1877 to i32, !dbg !18
  %dma.buffer.lds1548 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1887), !dbg !18
  %1888 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1548, i32 %dma.buffer.offset1546, <4 x i32> %dma.buffer.rsrc1547) #10, !dbg !18
  %1889 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local923), !dbg !18
  %1890 = zext i32 %1889 to i64, !dbg !18
  %1891 = inttoptr i64 %1890 to ptr addrspace(3), !dbg !18
  %address.local933 = getelementptr [2 x i8], ptr addrspace(1) %1874, i64 %address.local928, !dbg !18
  %1892 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1893 = ptrtoint ptr addrspace(1) %address.local933 to i64, !dbg !18
  %1894 = sub i64 %1893, %1892, !dbg !18
  %dma.buffer.offset = trunc i64 %1894 to i32, !dbg !18
  %1895 = trunc i64 %1892 to i32, !dbg !18
  %1896 = insertelement <4 x i32> poison, i32 %1895, i64 0, !dbg !18
  %1897 = lshr i64 %1892, 32, !dbg !18
  %1898 = trunc i64 %1897 to i32, !dbg !18
  %1899 = insertelement <4 x i32> %1896, i32 %1898, i64 1, !dbg !18
  %1900 = insertelement <4 x i32> %1899, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc = insertelement <4 x i32> %1900, i32 131072, i64 3, !dbg !18
  %1901 = ptrtoint ptr addrspace(3) %1891 to i32, !dbg !18
  %dma.buffer.lds = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1901), !dbg !18
  %1902 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc) #10, !dbg !18
  %1903 = shufflevector <8 x half> %1858, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1904 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1903, <4 x float> zeroinitializer) #9, !dbg !14
  %1905 = shufflevector <8 x half> %1860, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1906 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %1905, <4 x float> %1904) #9, !dbg !14
  %1907 = shufflevector <8 x half> %1862, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1908 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %1907, <4 x float> %1906) #9, !dbg !14
  %1909 = shufflevector <8 x half> %1864, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1910 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %1909, <4 x float> %1908) #9, !dbg !14
  %1911 = shufflevector <8 x half> %1866, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1912 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %1911, <4 x float> %1910) #9, !dbg !14
  %1913 = shufflevector <8 x half> %1868, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1914 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %1913, <4 x float> %1912) #9, !dbg !14
  %1915 = shufflevector <8 x half> %1870, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1916 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %1915, <4 x float> %1914) #9, !dbg !14
  %1917 = shufflevector <8 x half> %1872, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1918 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %1917, <4 x float> %1916) #9, !dbg !14
  %1919 = extractelement <4 x float> %1918, i64 0, !dbg !14
  %1920 = extractelement <4 x float> %1918, i64 1, !dbg !14
  %1921 = extractelement <4 x float> %1918, i64 2, !dbg !14
  %1922 = extractelement <4 x float> %1918, i64 3, !dbg !14
  %1923 = select i1 %.not, float %1920, float %1919, !dbg !14
  %1924 = bitcast float %1923 to i32, !dbg !14
  %1925 = call i32 @llvm.amdgcn.ds.bpermute(i32 %988, i32 %1924), !dbg !14
  %1926 = bitcast i32 %1925 to float, !dbg !14
  %1927 = select i1 %.not, float %1919, float %1926, !dbg !14
  %1928 = select i1 %.not, float %1926, float %1920, !dbg !14
  %1929 = select i1 %.not, float %1922, float %1921, !dbg !14
  %1930 = bitcast float %1929 to i32, !dbg !14
  %1931 = call i32 @llvm.amdgcn.ds.bpermute(i32 %988, i32 %1930), !dbg !14
  %1932 = bitcast i32 %1931 to float, !dbg !14
  %1933 = select i1 %.not, float %1921, float %1932, !dbg !14
  %1934 = select i1 %.not, float %1932, float %1922, !dbg !14
  %1935 = select i1 %.not1, float %1933, float %1927, !dbg !14
  %1936 = bitcast float %1935 to i32, !dbg !14
  %1937 = call i32 @llvm.amdgcn.ds.bpermute(i32 %996, i32 %1936), !dbg !14
  %1938 = bitcast i32 %1937 to float, !dbg !14
  %1939 = select i1 %.not1, float %1927, float %1938, !dbg !14
  %1940 = select i1 %.not1, float %1938, float %1933, !dbg !14
  %1941 = select i1 %.not1, float %1934, float %1928, !dbg !14
  %1942 = bitcast float %1941 to i32, !dbg !14
  %1943 = call i32 @llvm.amdgcn.ds.bpermute(i32 %996, i32 %1942), !dbg !14
  %1944 = bitcast i32 %1943 to float, !dbg !14
  %1945 = select i1 %.not1, float %1928, float %1944, !dbg !14
  %1946 = select i1 %.not1, float %1944, float %1934, !dbg !14
  %1947 = extractvalue [2 x <8 x float>] %212, 0, !dbg !35
  %1948 = shufflevector <8 x float> %1947, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1949 = fmul <4 x float> %1948, %1069, !dbg !35
  %1950 = insertelement <4 x float> poison, float %1939, i64 0, !dbg !36
  %1951 = insertelement <4 x float> %1950, float %1945, i64 1, !dbg !36
  %1952 = insertelement <4 x float> %1951, float %1940, i64 2, !dbg !36
  %1953 = insertelement <4 x float> %1952, float %1946, i64 3, !dbg !36
  %1954 = fadd <4 x float> %1949, %1953, !dbg !36
  %1955 = shufflevector <8 x half> %1858, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1956 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1955, <4 x float> zeroinitializer) #9, !dbg !14
  %1957 = shufflevector <8 x half> %1860, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1958 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %1957, <4 x float> %1956) #9, !dbg !14
  %1959 = shufflevector <8 x half> %1862, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1960 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %1959, <4 x float> %1958) #9, !dbg !14
  %1961 = shufflevector <8 x half> %1864, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1962 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %1961, <4 x float> %1960) #9, !dbg !14
  %1963 = shufflevector <8 x half> %1866, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1964 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %1963, <4 x float> %1962) #9, !dbg !14
  %1965 = shufflevector <8 x half> %1868, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1966 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %1965, <4 x float> %1964) #9, !dbg !14
  %1967 = shufflevector <8 x half> %1870, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1968 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %1967, <4 x float> %1966) #9, !dbg !14
  %1969 = shufflevector <8 x half> %1872, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1970 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %1969, <4 x float> %1968) #9, !dbg !14
  %1971 = extractelement <4 x float> %1970, i64 0, !dbg !14
  %1972 = extractelement <4 x float> %1970, i64 1, !dbg !14
  %1973 = extractelement <4 x float> %1970, i64 2, !dbg !14
  %1974 = extractelement <4 x float> %1970, i64 3, !dbg !14
  %1975 = select i1 %.not, float %1972, float %1971, !dbg !14
  %1976 = bitcast float %1975 to i32, !dbg !14
  %1977 = call i32 @llvm.amdgcn.ds.bpermute(i32 %988, i32 %1976), !dbg !14
  %1978 = bitcast i32 %1977 to float, !dbg !14
  %1979 = select i1 %.not, float %1971, float %1978, !dbg !14
  %1980 = select i1 %.not, float %1978, float %1972, !dbg !14
  %1981 = select i1 %.not, float %1974, float %1973, !dbg !14
  %1982 = bitcast float %1981 to i32, !dbg !14
  %1983 = call i32 @llvm.amdgcn.ds.bpermute(i32 %988, i32 %1982), !dbg !14
  %1984 = bitcast i32 %1983 to float, !dbg !14
  %1985 = select i1 %.not, float %1973, float %1984, !dbg !14
  %1986 = select i1 %.not, float %1984, float %1974, !dbg !14
  %1987 = select i1 %.not1, float %1985, float %1979, !dbg !14
  %1988 = bitcast float %1987 to i32, !dbg !14
  %1989 = call i32 @llvm.amdgcn.ds.bpermute(i32 %996, i32 %1988), !dbg !14
  %1990 = bitcast i32 %1989 to float, !dbg !14
  %1991 = select i1 %.not1, float %1979, float %1990, !dbg !14
  %1992 = select i1 %.not1, float %1990, float %1985, !dbg !14
  %1993 = select i1 %.not1, float %1986, float %1980, !dbg !14
  %1994 = bitcast float %1993 to i32, !dbg !14
  %1995 = call i32 @llvm.amdgcn.ds.bpermute(i32 %996, i32 %1994), !dbg !14
  %1996 = bitcast i32 %1995 to float, !dbg !14
  %1997 = select i1 %.not1, float %1980, float %1996, !dbg !14
  %1998 = select i1 %.not1, float %1996, float %1986, !dbg !14
  %1999 = shufflevector <8 x float> %1947, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %2000 = fmul <4 x float> %1999, %1069, !dbg !35
  %2001 = insertelement <4 x float> poison, float %1991, i64 0, !dbg !36
  %2002 = insertelement <4 x float> %2001, float %1997, i64 1, !dbg !36
  %2003 = insertelement <4 x float> %2002, float %1992, i64 2, !dbg !36
  %2004 = insertelement <4 x float> %2003, float %1998, i64 3, !dbg !36
  %2005 = fadd <4 x float> %2000, %2004, !dbg !36
  %2006 = shufflevector <4 x float> %1954, <4 x float> %2005, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %2007 = insertvalue [2 x <8 x float>] poison, <8 x float> %2006, 0, !dbg !7
  %2008 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1903, <4 x float> zeroinitializer) #9, !dbg !14
  %2009 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1905, <4 x float> %2008) #9, !dbg !14
  %2010 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1907, <4 x float> %2009) #9, !dbg !14
  %2011 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1909, <4 x float> %2010) #9, !dbg !14
  %2012 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %1911, <4 x float> %2011) #9, !dbg !14
  %2013 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %1913, <4 x float> %2012) #9, !dbg !14
  %2014 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %1915, <4 x float> %2013) #9, !dbg !14
  %2015 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %1917, <4 x float> %2014) #9, !dbg !14
  %2016 = extractelement <4 x float> %2015, i64 0, !dbg !14
  %2017 = extractelement <4 x float> %2015, i64 1, !dbg !14
  %2018 = extractelement <4 x float> %2015, i64 2, !dbg !14
  %2019 = extractelement <4 x float> %2015, i64 3, !dbg !14
  %2020 = select i1 %.not, float %2017, float %2016, !dbg !14
  %2021 = bitcast float %2020 to i32, !dbg !14
  %2022 = call i32 @llvm.amdgcn.ds.bpermute(i32 %988, i32 %2021), !dbg !14
  %2023 = bitcast i32 %2022 to float, !dbg !14
  %2024 = select i1 %.not, float %2016, float %2023, !dbg !14
  %2025 = select i1 %.not, float %2023, float %2017, !dbg !14
  %2026 = select i1 %.not, float %2019, float %2018, !dbg !14
  %2027 = bitcast float %2026 to i32, !dbg !14
  %2028 = call i32 @llvm.amdgcn.ds.bpermute(i32 %988, i32 %2027), !dbg !14
  %2029 = bitcast i32 %2028 to float, !dbg !14
  %2030 = select i1 %.not, float %2018, float %2029, !dbg !14
  %2031 = select i1 %.not, float %2029, float %2019, !dbg !14
  %2032 = select i1 %.not1, float %2030, float %2024, !dbg !14
  %2033 = bitcast float %2032 to i32, !dbg !14
  %2034 = call i32 @llvm.amdgcn.ds.bpermute(i32 %996, i32 %2033), !dbg !14
  %2035 = bitcast i32 %2034 to float, !dbg !14
  %2036 = select i1 %.not1, float %2024, float %2035, !dbg !14
  %2037 = select i1 %.not1, float %2035, float %2030, !dbg !14
  %2038 = select i1 %.not1, float %2031, float %2025, !dbg !14
  %2039 = bitcast float %2038 to i32, !dbg !14
  %2040 = call i32 @llvm.amdgcn.ds.bpermute(i32 %996, i32 %2039), !dbg !14
  %2041 = bitcast i32 %2040 to float, !dbg !14
  %2042 = select i1 %.not1, float %2025, float %2041, !dbg !14
  %2043 = select i1 %.not1, float %2041, float %2031, !dbg !14
  %2044 = extractvalue [2 x <8 x float>] %212, 1, !dbg !35
  %2045 = shufflevector <8 x float> %2044, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %2046 = fmul <4 x float> %2045, %1077, !dbg !35
  %2047 = insertelement <4 x float> poison, float %2036, i64 0, !dbg !36
  %2048 = insertelement <4 x float> %2047, float %2042, i64 1, !dbg !36
  %2049 = insertelement <4 x float> %2048, float %2037, i64 2, !dbg !36
  %2050 = insertelement <4 x float> %2049, float %2043, i64 3, !dbg !36
  %2051 = fadd <4 x float> %2046, %2050, !dbg !36
  %2052 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1955, <4 x float> zeroinitializer) #9, !dbg !14
  %2053 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1957, <4 x float> %2052) #9, !dbg !14
  %2054 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1959, <4 x float> %2053) #9, !dbg !14
  %2055 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1961, <4 x float> %2054) #9, !dbg !14
  %2056 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %1963, <4 x float> %2055) #9, !dbg !14
  %2057 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %1965, <4 x float> %2056) #9, !dbg !14
  %2058 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %1967, <4 x float> %2057) #9, !dbg !14
  %2059 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %1969, <4 x float> %2058) #9, !dbg !14
  %2060 = extractelement <4 x float> %2059, i64 0, !dbg !14
  %2061 = extractelement <4 x float> %2059, i64 1, !dbg !14
  %2062 = extractelement <4 x float> %2059, i64 2, !dbg !14
  %2063 = extractelement <4 x float> %2059, i64 3, !dbg !14
  %2064 = select i1 %.not, float %2061, float %2060, !dbg !14
  %2065 = bitcast float %2064 to i32, !dbg !14
  %2066 = call i32 @llvm.amdgcn.ds.bpermute(i32 %988, i32 %2065), !dbg !14
  %2067 = bitcast i32 %2066 to float, !dbg !14
  %2068 = select i1 %.not, float %2060, float %2067, !dbg !14
  %2069 = select i1 %.not, float %2067, float %2061, !dbg !14
  %2070 = select i1 %.not, float %2063, float %2062, !dbg !14
  %2071 = bitcast float %2070 to i32, !dbg !14
  %2072 = call i32 @llvm.amdgcn.ds.bpermute(i32 %988, i32 %2071), !dbg !14
  %2073 = bitcast i32 %2072 to float, !dbg !14
  %2074 = select i1 %.not, float %2062, float %2073, !dbg !14
  %2075 = select i1 %.not, float %2073, float %2063, !dbg !14
  %2076 = select i1 %.not1, float %2074, float %2068, !dbg !14
  %2077 = bitcast float %2076 to i32, !dbg !14
  %2078 = call i32 @llvm.amdgcn.ds.bpermute(i32 %996, i32 %2077), !dbg !14
  %2079 = bitcast i32 %2078 to float, !dbg !14
  %2080 = select i1 %.not1, float %2068, float %2079, !dbg !14
  %2081 = select i1 %.not1, float %2079, float %2074, !dbg !14
  %2082 = select i1 %.not1, float %2075, float %2069, !dbg !14
  %2083 = bitcast float %2082 to i32, !dbg !14
  %2084 = call i32 @llvm.amdgcn.ds.bpermute(i32 %996, i32 %2083), !dbg !14
  %2085 = bitcast i32 %2084 to float, !dbg !14
  %2086 = select i1 %.not1, float %2069, float %2085, !dbg !14
  %2087 = select i1 %.not1, float %2085, float %2075, !dbg !14
  %2088 = shufflevector <8 x float> %2044, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %2089 = fmul <4 x float> %2088, %1077, !dbg !35
  %2090 = insertelement <4 x float> poison, float %2080, i64 0, !dbg !36
  %2091 = insertelement <4 x float> %2090, float %2086, i64 1, !dbg !36
  %2092 = insertelement <4 x float> %2091, float %2081, i64 2, !dbg !36
  %2093 = insertelement <4 x float> %2092, float %2087, i64 3, !dbg !36
  %2094 = fadd <4 x float> %2089, %2093, !dbg !36
  %2095 = shufflevector <4 x float> %2051, <4 x float> %2094, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %2096 = insertvalue [2 x <8 x float>] %2007, <8 x float> %2095, 1, !dbg !7
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %2097 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local871, i32 0) #0, !dbg !14
  %2098 = bitcast <4 x i32> %2097 to <8 x half>, !dbg !14
  %2099 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local884, i32 0) #0, !dbg !14
  %2100 = bitcast <4 x i32> %2099 to <8 x half>, !dbg !14
  %2101 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local888, i32 0) #0, !dbg !14
  %2102 = bitcast <4 x i32> %2101 to <8 x half>, !dbg !14
  %2103 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local892, i32 0) #0, !dbg !14
  %2104 = bitcast <4 x i32> %2103 to <8 x half>, !dbg !14
  %2105 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local894, i32 0) #0, !dbg !14
  %2106 = bitcast <4 x i32> %2105 to <8 x half>, !dbg !14
  %2107 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local898, i32 0) #0, !dbg !14
  %2108 = bitcast <4 x i32> %2107 to <8 x half>, !dbg !14
  %2109 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local902, i32 0) #0, !dbg !14
  %2110 = bitcast <4 x i32> %2109 to <8 x half>, !dbg !14
  %2111 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local906, i32 0) #0, !dbg !14
  %2112 = bitcast <4 x i32> %2111 to <8 x half>, !dbg !14
  %2113 = shufflevector <8 x half> %2098, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %2114 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %2113, <4 x float> zeroinitializer) #9, !dbg !14
  %2115 = shufflevector <8 x half> %2100, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %2116 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %2115, <4 x float> %2114) #9, !dbg !14
  %2117 = shufflevector <8 x half> %2102, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %2118 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %2117, <4 x float> %2116) #9, !dbg !14
  %2119 = shufflevector <8 x half> %2104, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %2120 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %2119, <4 x float> %2118) #9, !dbg !14
  %2121 = shufflevector <8 x half> %2106, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %2122 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %2121, <4 x float> %2120) #9, !dbg !14
  %2123 = shufflevector <8 x half> %2108, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %2124 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %2123, <4 x float> %2122) #9, !dbg !14
  %2125 = shufflevector <8 x half> %2110, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %2126 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %2125, <4 x float> %2124) #9, !dbg !14
  %2127 = shufflevector <8 x half> %2112, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %2128 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %2127, <4 x float> %2126) #9, !dbg !14
  %2129 = extractelement <4 x float> %2128, i64 0, !dbg !14
  %2130 = extractelement <4 x float> %2128, i64 1, !dbg !14
  %2131 = extractelement <4 x float> %2128, i64 2, !dbg !14
  %2132 = extractelement <4 x float> %2128, i64 3, !dbg !14
  %2133 = select i1 %.not, float %2130, float %2129, !dbg !14
  %2134 = bitcast float %2133 to i32, !dbg !14
  %2135 = call i32 @llvm.amdgcn.ds.bpermute(i32 %988, i32 %2134), !dbg !14
  %2136 = bitcast i32 %2135 to float, !dbg !14
  %2137 = select i1 %.not, float %2129, float %2136, !dbg !14
  %2138 = select i1 %.not, float %2136, float %2130, !dbg !14
  %2139 = select i1 %.not, float %2132, float %2131, !dbg !14
  %2140 = bitcast float %2139 to i32, !dbg !14
  %2141 = call i32 @llvm.amdgcn.ds.bpermute(i32 %988, i32 %2140), !dbg !14
  %2142 = bitcast i32 %2141 to float, !dbg !14
  %2143 = select i1 %.not, float %2131, float %2142, !dbg !14
  %2144 = select i1 %.not, float %2142, float %2132, !dbg !14
  %2145 = select i1 %.not1, float %2143, float %2137, !dbg !14
  %2146 = bitcast float %2145 to i32, !dbg !14
  %2147 = call i32 @llvm.amdgcn.ds.bpermute(i32 %996, i32 %2146), !dbg !14
  %2148 = bitcast i32 %2147 to float, !dbg !14
  %2149 = select i1 %.not1, float %2137, float %2148, !dbg !14
  %2150 = select i1 %.not1, float %2148, float %2143, !dbg !14
  %2151 = select i1 %.not1, float %2144, float %2138, !dbg !14
  %2152 = bitcast float %2151 to i32, !dbg !14
  %2153 = call i32 @llvm.amdgcn.ds.bpermute(i32 %996, i32 %2152), !dbg !14
  %2154 = bitcast i32 %2153 to float, !dbg !14
  %2155 = select i1 %.not1, float %2138, float %2154, !dbg !14
  %2156 = select i1 %.not1, float %2154, float %2144, !dbg !14
  %2157 = extractvalue [2 x <8 x float>] %213, 0, !dbg !35
  %2158 = shufflevector <8 x float> %2157, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %2159 = fmul <4 x float> %2158, %1069, !dbg !35
  %2160 = insertelement <4 x float> poison, float %2149, i64 0, !dbg !36
  %2161 = insertelement <4 x float> %2160, float %2155, i64 1, !dbg !36
  %2162 = insertelement <4 x float> %2161, float %2150, i64 2, !dbg !36
  %2163 = insertelement <4 x float> %2162, float %2156, i64 3, !dbg !36
  %2164 = fadd <4 x float> %2159, %2163, !dbg !36
  %2165 = shufflevector <8 x half> %2098, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %2166 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %2165, <4 x float> zeroinitializer) #9, !dbg !14
  %2167 = shufflevector <8 x half> %2100, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %2168 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %2167, <4 x float> %2166) #9, !dbg !14
  %2169 = shufflevector <8 x half> %2102, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %2170 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %2169, <4 x float> %2168) #9, !dbg !14
  %2171 = shufflevector <8 x half> %2104, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %2172 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %2171, <4 x float> %2170) #9, !dbg !14
  %2173 = shufflevector <8 x half> %2106, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %2174 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %2173, <4 x float> %2172) #9, !dbg !14
  %2175 = shufflevector <8 x half> %2108, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %2176 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %2175, <4 x float> %2174) #9, !dbg !14
  %2177 = shufflevector <8 x half> %2110, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %2178 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %2177, <4 x float> %2176) #9, !dbg !14
  %2179 = shufflevector <8 x half> %2112, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %2180 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %2179, <4 x float> %2178) #9, !dbg !14
  %2181 = extractelement <4 x float> %2180, i64 0, !dbg !14
  %2182 = extractelement <4 x float> %2180, i64 1, !dbg !14
  %2183 = extractelement <4 x float> %2180, i64 2, !dbg !14
  %2184 = extractelement <4 x float> %2180, i64 3, !dbg !14
  %2185 = select i1 %.not, float %2182, float %2181, !dbg !14
  %2186 = bitcast float %2185 to i32, !dbg !14
  %2187 = call i32 @llvm.amdgcn.ds.bpermute(i32 %988, i32 %2186), !dbg !14
  %2188 = bitcast i32 %2187 to float, !dbg !14
  %2189 = select i1 %.not, float %2181, float %2188, !dbg !14
  %2190 = select i1 %.not, float %2188, float %2182, !dbg !14
  %2191 = select i1 %.not, float %2184, float %2183, !dbg !14
  %2192 = bitcast float %2191 to i32, !dbg !14
  %2193 = call i32 @llvm.amdgcn.ds.bpermute(i32 %988, i32 %2192), !dbg !14
  %2194 = bitcast i32 %2193 to float, !dbg !14
  %2195 = select i1 %.not, float %2183, float %2194, !dbg !14
  %2196 = select i1 %.not, float %2194, float %2184, !dbg !14
  %2197 = select i1 %.not1, float %2195, float %2189, !dbg !14
  %2198 = bitcast float %2197 to i32, !dbg !14
  %2199 = call i32 @llvm.amdgcn.ds.bpermute(i32 %996, i32 %2198), !dbg !14
  %2200 = bitcast i32 %2199 to float, !dbg !14
  %2201 = select i1 %.not1, float %2189, float %2200, !dbg !14
  %2202 = select i1 %.not1, float %2200, float %2195, !dbg !14
  %2203 = select i1 %.not1, float %2196, float %2190, !dbg !14
  %2204 = bitcast float %2203 to i32, !dbg !14
  %2205 = call i32 @llvm.amdgcn.ds.bpermute(i32 %996, i32 %2204), !dbg !14
  %2206 = bitcast i32 %2205 to float, !dbg !14
  %2207 = select i1 %.not1, float %2190, float %2206, !dbg !14
  %2208 = select i1 %.not1, float %2206, float %2196, !dbg !14
  %2209 = shufflevector <8 x float> %2157, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %2210 = fmul <4 x float> %2209, %1069, !dbg !35
  %2211 = insertelement <4 x float> poison, float %2201, i64 0, !dbg !36
  %2212 = insertelement <4 x float> %2211, float %2207, i64 1, !dbg !36
  %2213 = insertelement <4 x float> %2212, float %2202, i64 2, !dbg !36
  %2214 = insertelement <4 x float> %2213, float %2208, i64 3, !dbg !36
  %2215 = fadd <4 x float> %2210, %2214, !dbg !36
  %2216 = shufflevector <4 x float> %2164, <4 x float> %2215, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %2217 = insertvalue [2 x <8 x float>] poison, <8 x float> %2216, 0, !dbg !7
  %2218 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %2113, <4 x float> zeroinitializer) #9, !dbg !14
  %2219 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %2115, <4 x float> %2218) #9, !dbg !14
  %2220 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %2117, <4 x float> %2219) #9, !dbg !14
  %2221 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %2119, <4 x float> %2220) #9, !dbg !14
  %2222 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %2121, <4 x float> %2221) #9, !dbg !14
  %2223 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %2123, <4 x float> %2222) #9, !dbg !14
  %2224 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %2125, <4 x float> %2223) #9, !dbg !14
  %2225 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %2127, <4 x float> %2224) #9, !dbg !14
  %2226 = extractelement <4 x float> %2225, i64 0, !dbg !14
  %2227 = extractelement <4 x float> %2225, i64 1, !dbg !14
  %2228 = extractelement <4 x float> %2225, i64 2, !dbg !14
  %2229 = extractelement <4 x float> %2225, i64 3, !dbg !14
  %2230 = select i1 %.not, float %2227, float %2226, !dbg !14
  %2231 = bitcast float %2230 to i32, !dbg !14
  %2232 = call i32 @llvm.amdgcn.ds.bpermute(i32 %988, i32 %2231), !dbg !14
  %2233 = bitcast i32 %2232 to float, !dbg !14
  %2234 = select i1 %.not, float %2226, float %2233, !dbg !14
  %2235 = select i1 %.not, float %2233, float %2227, !dbg !14
  %2236 = select i1 %.not, float %2229, float %2228, !dbg !14
  %2237 = bitcast float %2236 to i32, !dbg !14
  %2238 = call i32 @llvm.amdgcn.ds.bpermute(i32 %988, i32 %2237), !dbg !14
  %2239 = bitcast i32 %2238 to float, !dbg !14
  %2240 = select i1 %.not, float %2228, float %2239, !dbg !14
  %2241 = select i1 %.not, float %2239, float %2229, !dbg !14
  %2242 = select i1 %.not1, float %2240, float %2234, !dbg !14
  %2243 = bitcast float %2242 to i32, !dbg !14
  %2244 = call i32 @llvm.amdgcn.ds.bpermute(i32 %996, i32 %2243), !dbg !14
  %2245 = bitcast i32 %2244 to float, !dbg !14
  %2246 = select i1 %.not1, float %2234, float %2245, !dbg !14
  %2247 = select i1 %.not1, float %2245, float %2240, !dbg !14
  %2248 = select i1 %.not1, float %2241, float %2235, !dbg !14
  %2249 = bitcast float %2248 to i32, !dbg !14
  %2250 = call i32 @llvm.amdgcn.ds.bpermute(i32 %996, i32 %2249), !dbg !14
  %2251 = bitcast i32 %2250 to float, !dbg !14
  %2252 = select i1 %.not1, float %2235, float %2251, !dbg !14
  %2253 = select i1 %.not1, float %2251, float %2241, !dbg !14
  %2254 = extractvalue [2 x <8 x float>] %213, 1, !dbg !35
  %2255 = shufflevector <8 x float> %2254, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %2256 = fmul <4 x float> %2255, %1077, !dbg !35
  %2257 = insertelement <4 x float> poison, float %2246, i64 0, !dbg !36
  %2258 = insertelement <4 x float> %2257, float %2252, i64 1, !dbg !36
  %2259 = insertelement <4 x float> %2258, float %2247, i64 2, !dbg !36
  %2260 = insertelement <4 x float> %2259, float %2253, i64 3, !dbg !36
  %2261 = fadd <4 x float> %2256, %2260, !dbg !36
  %2262 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %2165, <4 x float> zeroinitializer) #9, !dbg !14
  %2263 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %2167, <4 x float> %2262) #9, !dbg !14
  %2264 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %2169, <4 x float> %2263) #9, !dbg !14
  %2265 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %2171, <4 x float> %2264) #9, !dbg !14
  %2266 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %2173, <4 x float> %2265) #9, !dbg !14
  %2267 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %2175, <4 x float> %2266) #9, !dbg !14
  %2268 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %2177, <4 x float> %2267) #9, !dbg !14
  %2269 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %2179, <4 x float> %2268) #9, !dbg !14
  %2270 = extractelement <4 x float> %2269, i64 0, !dbg !14
  %2271 = extractelement <4 x float> %2269, i64 1, !dbg !14
  %2272 = extractelement <4 x float> %2269, i64 2, !dbg !14
  %2273 = extractelement <4 x float> %2269, i64 3, !dbg !14
  %2274 = select i1 %.not, float %2271, float %2270, !dbg !14
  %2275 = bitcast float %2274 to i32, !dbg !14
  %2276 = call i32 @llvm.amdgcn.ds.bpermute(i32 %988, i32 %2275), !dbg !14
  %2277 = bitcast i32 %2276 to float, !dbg !14
  %2278 = select i1 %.not, float %2270, float %2277, !dbg !14
  %2279 = select i1 %.not, float %2277, float %2271, !dbg !14
  %2280 = select i1 %.not, float %2273, float %2272, !dbg !14
  %2281 = bitcast float %2280 to i32, !dbg !14
  %2282 = call i32 @llvm.amdgcn.ds.bpermute(i32 %988, i32 %2281), !dbg !14
  %2283 = bitcast i32 %2282 to float, !dbg !14
  %2284 = select i1 %.not, float %2272, float %2283, !dbg !14
  %2285 = select i1 %.not, float %2283, float %2273, !dbg !14
  %2286 = select i1 %.not1, float %2284, float %2278, !dbg !14
  %2287 = bitcast float %2286 to i32, !dbg !14
  %2288 = call i32 @llvm.amdgcn.ds.bpermute(i32 %996, i32 %2287), !dbg !14
  %2289 = bitcast i32 %2288 to float, !dbg !14
  %2290 = select i1 %.not1, float %2278, float %2289, !dbg !14
  %2291 = select i1 %.not1, float %2289, float %2284, !dbg !14
  %2292 = select i1 %.not1, float %2285, float %2279, !dbg !14
  %2293 = bitcast float %2292 to i32, !dbg !14
  %2294 = call i32 @llvm.amdgcn.ds.bpermute(i32 %996, i32 %2293), !dbg !14
  %2295 = bitcast i32 %2294 to float, !dbg !14
  %2296 = select i1 %.not1, float %2279, float %2295, !dbg !14
  %2297 = select i1 %.not1, float %2295, float %2285, !dbg !14
  %2298 = shufflevector <8 x float> %2254, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %2299 = fmul <4 x float> %2298, %1077, !dbg !35
  %2300 = insertelement <4 x float> poison, float %2290, i64 0, !dbg !36
  %2301 = insertelement <4 x float> %2300, float %2296, i64 1, !dbg !36
  %2302 = insertelement <4 x float> %2301, float %2291, i64 2, !dbg !36
  %2303 = insertelement <4 x float> %2302, float %2297, i64 3, !dbg !36
  %2304 = fadd <4 x float> %2299, %2303, !dbg !36
  %2305 = shufflevector <4 x float> %2261, <4 x float> %2304, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %2306 = insertvalue [2 x <8 x float>] %2217, <8 x float> %2305, 1, !dbg !7
  %2307 = insertvalue [2 x <1 x float>] poison, <1 x float> %1003, 0, !dbg !7
  %2308 = insertvalue [2 x <1 x float>] %2307, <1 x float> %1046, 1, !dbg !7
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %2309 = add nuw nsw i64 %207, 1, !dbg !7
  br label %206, !dbg !7

2310:                                             ; preds = %206
  %2311 = extractvalue [2 x <1 x float>] %209, 0, !dbg !19
  %2312 = shufflevector <1 x float> %2311, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %2313 = extractvalue [2 x <1 x float>] %209, 1, !dbg !19
  %2314 = shufflevector <1 x float> %2313, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  br i1 true, label %2315, label %2345, !dbg !21

2315:                                             ; preds = %2310
  %2316 = extractvalue [2 x <8 x float>] %208, 1, !dbg !19
  %2317 = shufflevector <8 x float> %2316, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !19
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %2314, !dbg !19
  %normalized = fmul <4 x float> %2317, %row.reciprocal, !dbg !19
  %2318 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !20
  %2319 = extractelement <4 x half> %2318, i64 3, !dbg !20
  %2320 = extractelement <4 x half> %2318, i64 2, !dbg !20
  %2321 = extractelement <4 x half> %2318, i64 1, !dbg !20
  %2322 = extractelement <4 x half> %2318, i64 0, !dbg !20
  %2323 = shufflevector <8 x float> %2316, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !19
  %normalized104 = fmul <4 x float> %2323, %row.reciprocal, !dbg !19
  %2324 = fptrunc <4 x float> %normalized104 to <4 x half>, !dbg !20
  %2325 = extractelement <4 x half> %2324, i64 3, !dbg !20
  %2326 = extractelement <4 x half> %2324, i64 2, !dbg !20
  %2327 = extractelement <4 x half> %2324, i64 1, !dbg !20
  %2328 = extractelement <4 x half> %2324, i64 0, !dbg !20
  %2329 = extractvalue [2 x <8 x float>] %208, 0, !dbg !19
  %2330 = shufflevector <8 x float> %2329, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !19
  %row.reciprocal105 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %2312, !dbg !19
  %normalized106 = fmul <4 x float> %2330, %row.reciprocal105, !dbg !19
  %2331 = fptrunc <4 x float> %normalized106 to <4 x half>, !dbg !20
  %2332 = extractelement <4 x half> %2331, i64 3, !dbg !20
  %2333 = extractelement <4 x half> %2331, i64 2, !dbg !20
  %2334 = extractelement <4 x half> %2331, i64 1, !dbg !20
  %2335 = extractelement <4 x half> %2331, i64 0, !dbg !20
  %2336 = shufflevector <8 x float> %2329, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !19
  %normalized107 = fmul <4 x float> %2336, %row.reciprocal105, !dbg !19
  %2337 = fptrunc <4 x float> %normalized107 to <4 x half>, !dbg !20
  %2338 = extractelement <4 x half> %2337, i64 3, !dbg !20
  %2339 = extractelement <4 x half> %2337, i64 2, !dbg !20
  %2340 = extractelement <4 x half> %2337, i64 1, !dbg !20
  %2341 = extractelement <4 x half> %2337, i64 0, !dbg !20
  %2342 = add nsw i64 %16, %22, !dbg !21
  %2343 = shl i64 %2342, 19, !dbg !21
  %2344 = and i64 %2343, 16252928, !dbg !21
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
  %address.local944 = add nuw nsw i64 %2344, %address.local943, !dbg !21
  %address.local945 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local944, !dbg !21
  %address.local946 = lshr i64 %address.local935, 2, !dbg !11
  %address.local947 = and i64 %address.local946, 12, !dbg !11
  %address.local948 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local947, !dbg !21
  store half %2341, ptr addrspace(1) %address.local948, align 2, !dbg !21
  %address.local949 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local944, !dbg !21
  %address.local950 = or disjoint i64 %address.local947, 1, !dbg !21
  %address.local951 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local949, i64 %address.local950, !dbg !21
  store half %2340, ptr addrspace(1) %address.local951, align 2, !dbg !21
  %address.local952 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local944, !dbg !21
  %address.local953 = or disjoint i64 %address.local947, 2, !dbg !21
  %address.local954 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local952, i64 %address.local953, !dbg !21
  store half %2339, ptr addrspace(1) %address.local954, align 2, !dbg !21
  %address.local955 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local944, !dbg !21
  %address.local956 = or disjoint i64 %address.local947, 3, !dbg !21
  %address.local957 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local955, i64 %address.local956, !dbg !21
  store half %2338, ptr addrspace(1) %address.local957, align 2, !dbg !21
  %address.local958 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local944, !dbg !21
  %address.local959 = or disjoint i64 %address.local947, 16, !dbg !21
  %address.local960 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local958, i64 %address.local959, !dbg !21
  store half %2335, ptr addrspace(1) %address.local960, align 2, !dbg !21
  %address.local961 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local944, !dbg !21
  %address.local962 = or disjoint i64 %address.local947, 17, !dbg !21
  %address.local963 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local961, i64 %address.local962, !dbg !21
  store half %2334, ptr addrspace(1) %address.local963, align 2, !dbg !21
  %address.local964 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local944, !dbg !21
  %address.local965 = or disjoint i64 %address.local947, 18, !dbg !21
  %address.local966 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local964, i64 %address.local965, !dbg !21
  store half %2333, ptr addrspace(1) %address.local966, align 2, !dbg !21
  %address.local967 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local944, !dbg !21
  %address.local968 = or disjoint i64 %address.local947, 19, !dbg !21
  %address.local969 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local967, i64 %address.local968, !dbg !21
  store half %2332, ptr addrspace(1) %address.local969, align 2, !dbg !21
  %address.local970 = shl i64 %address.local942, 7, !dbg !21
  %address.local971 = add i64 %address.local970, 8192, !dbg !21
  %address.local972 = add nuw nsw i64 %2344, %address.local971, !dbg !21
  %address.local973 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local972, !dbg !21
  %address.local974 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local947, !dbg !21
  store half %2328, ptr addrspace(1) %address.local974, align 2, !dbg !21
  %address.local975 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local972, !dbg !21
  %address.local976 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local975, i64 %address.local950, !dbg !21
  store half %2327, ptr addrspace(1) %address.local976, align 2, !dbg !21
  %address.local977 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local972, !dbg !21
  %address.local978 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local977, i64 %address.local953, !dbg !21
  store half %2326, ptr addrspace(1) %address.local978, align 2, !dbg !21
  %address.local979 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local972, !dbg !21
  %address.local980 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local979, i64 %address.local956, !dbg !21
  store half %2325, ptr addrspace(1) %address.local980, align 2, !dbg !21
  %address.local981 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local972, !dbg !21
  %address.local982 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local981, i64 %address.local959, !dbg !21
  store half %2322, ptr addrspace(1) %address.local982, align 2, !dbg !21
  %address.local983 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local972, !dbg !21
  %address.local984 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local983, i64 %address.local962, !dbg !21
  store half %2321, ptr addrspace(1) %address.local984, align 2, !dbg !21
  %address.local985 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local972, !dbg !21
  %address.local986 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local965, !dbg !21
  store half %2320, ptr addrspace(1) %address.local986, align 2, !dbg !21
  %address.local987 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local972, !dbg !21
  %address.local988 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local987, i64 %address.local968, !dbg !21
  store half %2319, ptr addrspace(1) %address.local988, align 2, !dbg !21
  br label %2345, !dbg !21

2345:                                             ; preds = %2315, %2310
  br i1 true, label %2346, label %2374, !dbg !21

2346:                                             ; preds = %2345
  %2347 = extractvalue [2 x <8 x float>] %211, 1, !dbg !19
  %2348 = shufflevector <8 x float> %2347, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !19
  %row.reciprocal108 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %2314, !dbg !19
  %normalized109 = fmul <4 x float> %2348, %row.reciprocal108, !dbg !19
  %2349 = fptrunc <4 x float> %normalized109 to <4 x half>, !dbg !20
  %2350 = extractelement <4 x half> %2349, i64 3, !dbg !20
  %2351 = extractelement <4 x half> %2349, i64 2, !dbg !20
  %2352 = extractelement <4 x half> %2349, i64 1, !dbg !20
  %2353 = extractelement <4 x half> %2349, i64 0, !dbg !20
  %2354 = shufflevector <8 x float> %2347, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !19
  %normalized110 = fmul <4 x float> %2354, %row.reciprocal108, !dbg !19
  %2355 = fptrunc <4 x float> %normalized110 to <4 x half>, !dbg !20
  %2356 = extractelement <4 x half> %2355, i64 3, !dbg !20
  %2357 = extractelement <4 x half> %2355, i64 2, !dbg !20
  %2358 = extractelement <4 x half> %2355, i64 1, !dbg !20
  %2359 = extractelement <4 x half> %2355, i64 0, !dbg !20
  %2360 = extractvalue [2 x <8 x float>] %211, 0, !dbg !19
  %2361 = shufflevector <8 x float> %2360, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !19
  %row.reciprocal111 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %2312, !dbg !19
  %normalized112 = fmul <4 x float> %2361, %row.reciprocal111, !dbg !19
  %2362 = fptrunc <4 x float> %normalized112 to <4 x half>, !dbg !20
  %2363 = extractelement <4 x half> %2362, i64 3, !dbg !20
  %2364 = extractelement <4 x half> %2362, i64 2, !dbg !20
  %2365 = extractelement <4 x half> %2362, i64 1, !dbg !20
  %2366 = extractelement <4 x half> %2362, i64 0, !dbg !20
  %2367 = shufflevector <8 x float> %2360, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !19
  %normalized113 = fmul <4 x float> %2367, %row.reciprocal111, !dbg !19
  %2368 = fptrunc <4 x float> %normalized113 to <4 x half>, !dbg !20
  %2369 = extractelement <4 x half> %2368, i64 3, !dbg !20
  %2370 = extractelement <4 x half> %2368, i64 2, !dbg !20
  %2371 = extractelement <4 x half> %2368, i64 1, !dbg !20
  %2372 = extractelement <4 x half> %2368, i64 0, !dbg !20
  %2373 = shl nuw nsw i64 %24, 19, !dbg !21
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
  %address.local999 = add nuw nsw i64 %2373, %address.local998, !dbg !21
  %address.local1000 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local999, !dbg !21
  %address.local1001 = lshr i64 %address.local990, 2, !dbg !11
  %address.local1002 = and i64 %address.local1001, 12, !dbg !11
  %address.local1003 = or disjoint i64 %address.local1002, 32, !dbg !21
  %address.local1004 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1000, i64 %address.local1003, !dbg !21
  store half %2372, ptr addrspace(1) %address.local1004, align 2, !dbg !21
  %address.local1005 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local999, !dbg !21
  %address.local1006 = or disjoint i64 %address.local1002, 33, !dbg !21
  %address.local1007 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1005, i64 %address.local1006, !dbg !21
  store half %2371, ptr addrspace(1) %address.local1007, align 2, !dbg !21
  %address.local1008 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local999, !dbg !21
  %address.local1009 = or disjoint i64 %address.local1002, 34, !dbg !21
  %address.local1010 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1008, i64 %address.local1009, !dbg !21
  store half %2370, ptr addrspace(1) %address.local1010, align 2, !dbg !21
  %address.local1011 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local999, !dbg !21
  %address.local1012 = or disjoint i64 %address.local1002, 35, !dbg !21
  %address.local1013 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1011, i64 %address.local1012, !dbg !21
  store half %2369, ptr addrspace(1) %address.local1013, align 2, !dbg !21
  %address.local1014 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local999, !dbg !21
  %address.local1015 = or disjoint i64 %address.local1002, 48, !dbg !21
  %address.local1016 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1014, i64 %address.local1015, !dbg !21
  store half %2366, ptr addrspace(1) %address.local1016, align 2, !dbg !21
  %address.local1017 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local999, !dbg !21
  %address.local1018 = or disjoint i64 %address.local1002, 49, !dbg !21
  %address.local1019 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1017, i64 %address.local1018, !dbg !21
  store half %2365, ptr addrspace(1) %address.local1019, align 2, !dbg !21
  %address.local1020 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local999, !dbg !21
  %address.local1021 = or disjoint i64 %address.local1002, 50, !dbg !21
  %address.local1022 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1020, i64 %address.local1021, !dbg !21
  store half %2364, ptr addrspace(1) %address.local1022, align 2, !dbg !21
  %address.local1023 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local999, !dbg !21
  %address.local1024 = or i64 %address.local1001, 51, !dbg !21
  %address.local1025 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1023, i64 %address.local1024, !dbg !21
  store half %2363, ptr addrspace(1) %address.local1025, align 2, !dbg !21
  %address.local1026 = shl i64 %address.local997, 7, !dbg !21
  %address.local1027 = add i64 %address.local1026, 8192, !dbg !21
  %address.local1028 = add nuw nsw i64 %2373, %address.local1027, !dbg !21
  %address.local1029 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1028, !dbg !21
  %address.local1030 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1029, i64 %address.local1003, !dbg !21
  store half %2359, ptr addrspace(1) %address.local1030, align 2, !dbg !21
  %address.local1031 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1028, !dbg !21
  %address.local1032 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1031, i64 %address.local1006, !dbg !21
  store half %2358, ptr addrspace(1) %address.local1032, align 2, !dbg !21
  %address.local1033 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1028, !dbg !21
  %address.local1034 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1033, i64 %address.local1009, !dbg !21
  store half %2357, ptr addrspace(1) %address.local1034, align 2, !dbg !21
  %address.local1035 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1028, !dbg !21
  %address.local1036 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1035, i64 %address.local1012, !dbg !21
  store half %2356, ptr addrspace(1) %address.local1036, align 2, !dbg !21
  %address.local1037 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1028, !dbg !21
  %address.local1038 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1037, i64 %address.local1015, !dbg !21
  store half %2353, ptr addrspace(1) %address.local1038, align 2, !dbg !21
  %address.local1039 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1028, !dbg !21
  %address.local1040 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1039, i64 %address.local1018, !dbg !21
  store half %2352, ptr addrspace(1) %address.local1040, align 2, !dbg !21
  %address.local1041 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1028, !dbg !21
  %address.local1042 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1041, i64 %address.local1021, !dbg !21
  store half %2351, ptr addrspace(1) %address.local1042, align 2, !dbg !21
  %address.local1043 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1028, !dbg !21
  %address.local1044 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1043, i64 %address.local1024, !dbg !21
  store half %2350, ptr addrspace(1) %address.local1044, align 2, !dbg !21
  br label %2374, !dbg !21

2374:                                             ; preds = %2346, %2345
  br i1 true, label %2375, label %2403, !dbg !21

2375:                                             ; preds = %2374
  %2376 = extractvalue [2 x <8 x float>] %212, 1, !dbg !19
  %2377 = shufflevector <8 x float> %2376, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !19
  %row.reciprocal114 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %2314, !dbg !19
  %normalized115 = fmul <4 x float> %2377, %row.reciprocal114, !dbg !19
  %2378 = fptrunc <4 x float> %normalized115 to <4 x half>, !dbg !20
  %2379 = extractelement <4 x half> %2378, i64 3, !dbg !20
  %2380 = extractelement <4 x half> %2378, i64 2, !dbg !20
  %2381 = extractelement <4 x half> %2378, i64 1, !dbg !20
  %2382 = extractelement <4 x half> %2378, i64 0, !dbg !20
  %2383 = shufflevector <8 x float> %2376, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !19
  %normalized116 = fmul <4 x float> %2383, %row.reciprocal114, !dbg !19
  %2384 = fptrunc <4 x float> %normalized116 to <4 x half>, !dbg !20
  %2385 = extractelement <4 x half> %2384, i64 3, !dbg !20
  %2386 = extractelement <4 x half> %2384, i64 2, !dbg !20
  %2387 = extractelement <4 x half> %2384, i64 1, !dbg !20
  %2388 = extractelement <4 x half> %2384, i64 0, !dbg !20
  %2389 = extractvalue [2 x <8 x float>] %212, 0, !dbg !19
  %2390 = shufflevector <8 x float> %2389, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !19
  %row.reciprocal117 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %2312, !dbg !19
  %normalized118 = fmul <4 x float> %2390, %row.reciprocal117, !dbg !19
  %2391 = fptrunc <4 x float> %normalized118 to <4 x half>, !dbg !20
  %2392 = extractelement <4 x half> %2391, i64 3, !dbg !20
  %2393 = extractelement <4 x half> %2391, i64 2, !dbg !20
  %2394 = extractelement <4 x half> %2391, i64 1, !dbg !20
  %2395 = extractelement <4 x half> %2391, i64 0, !dbg !20
  %2396 = shufflevector <8 x float> %2389, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !19
  %normalized119 = fmul <4 x float> %2396, %row.reciprocal117, !dbg !19
  %2397 = fptrunc <4 x float> %normalized119 to <4 x half>, !dbg !20
  %2398 = extractelement <4 x half> %2397, i64 3, !dbg !20
  %2399 = extractelement <4 x half> %2397, i64 2, !dbg !20
  %2400 = extractelement <4 x half> %2397, i64 1, !dbg !20
  %2401 = extractelement <4 x half> %2397, i64 0, !dbg !20
  %2402 = shl nuw nsw i64 %24, 19, !dbg !21
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
  %address.local1055 = add nuw nsw i64 %2402, %address.local1054, !dbg !21
  %address.local1056 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1055, !dbg !21
  %address.local1057 = lshr i64 %address.local1046, 2, !dbg !11
  %address.local1058 = and i64 %address.local1057, 12, !dbg !11
  %address.local1059 = or disjoint i64 %address.local1058, 64, !dbg !21
  %address.local1060 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1056, i64 %address.local1059, !dbg !21
  store half %2401, ptr addrspace(1) %address.local1060, align 2, !dbg !21
  %address.local1061 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1055, !dbg !21
  %address.local1062 = or disjoint i64 %address.local1058, 65, !dbg !21
  %address.local1063 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1061, i64 %address.local1062, !dbg !21
  store half %2400, ptr addrspace(1) %address.local1063, align 2, !dbg !21
  %address.local1064 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1055, !dbg !21
  %address.local1065 = or disjoint i64 %address.local1058, 66, !dbg !21
  %address.local1066 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1064, i64 %address.local1065, !dbg !21
  store half %2399, ptr addrspace(1) %address.local1066, align 2, !dbg !21
  %address.local1067 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1055, !dbg !21
  %address.local1068 = or disjoint i64 %address.local1058, 67, !dbg !21
  %address.local1069 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1067, i64 %address.local1068, !dbg !21
  store half %2398, ptr addrspace(1) %address.local1069, align 2, !dbg !21
  %address.local1070 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1055, !dbg !21
  %address.local1071 = or disjoint i64 %address.local1058, 80, !dbg !21
  %address.local1072 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1070, i64 %address.local1071, !dbg !21
  store half %2395, ptr addrspace(1) %address.local1072, align 2, !dbg !21
  %address.local1073 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1055, !dbg !21
  %address.local1074 = or disjoint i64 %address.local1058, 81, !dbg !21
  %address.local1075 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1073, i64 %address.local1074, !dbg !21
  store half %2394, ptr addrspace(1) %address.local1075, align 2, !dbg !21
  %address.local1076 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1055, !dbg !21
  %address.local1077 = or disjoint i64 %address.local1058, 82, !dbg !21
  %address.local1078 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1076, i64 %address.local1077, !dbg !21
  store half %2393, ptr addrspace(1) %address.local1078, align 2, !dbg !21
  %address.local1079 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1055, !dbg !21
  %address.local1080 = or disjoint i64 %address.local1058, 83, !dbg !21
  %address.local1081 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1079, i64 %address.local1080, !dbg !21
  store half %2392, ptr addrspace(1) %address.local1081, align 2, !dbg !21
  %address.local1082 = shl i64 %address.local1053, 7, !dbg !21
  %address.local1083 = add i64 %address.local1082, 8192, !dbg !21
  %address.local1084 = add nuw nsw i64 %2402, %address.local1083, !dbg !21
  %address.local1085 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1084, !dbg !21
  %address.local1086 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1085, i64 %address.local1059, !dbg !21
  store half %2388, ptr addrspace(1) %address.local1086, align 2, !dbg !21
  %address.local1087 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1084, !dbg !21
  %address.local1088 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1087, i64 %address.local1062, !dbg !21
  store half %2387, ptr addrspace(1) %address.local1088, align 2, !dbg !21
  %address.local1089 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1084, !dbg !21
  %address.local1090 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1089, i64 %address.local1065, !dbg !21
  store half %2386, ptr addrspace(1) %address.local1090, align 2, !dbg !21
  %address.local1091 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1084, !dbg !21
  %address.local1092 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1091, i64 %address.local1068, !dbg !21
  store half %2385, ptr addrspace(1) %address.local1092, align 2, !dbg !21
  %address.local1093 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1084, !dbg !21
  %address.local1094 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1093, i64 %address.local1071, !dbg !21
  store half %2382, ptr addrspace(1) %address.local1094, align 2, !dbg !21
  %address.local1095 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1084, !dbg !21
  %address.local1096 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1095, i64 %address.local1074, !dbg !21
  store half %2381, ptr addrspace(1) %address.local1096, align 2, !dbg !21
  %address.local1097 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1084, !dbg !21
  %address.local1098 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1097, i64 %address.local1077, !dbg !21
  store half %2380, ptr addrspace(1) %address.local1098, align 2, !dbg !21
  %address.local1099 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1084, !dbg !21
  %address.local1100 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1099, i64 %address.local1080, !dbg !21
  store half %2379, ptr addrspace(1) %address.local1100, align 2, !dbg !21
  br label %2403, !dbg !21

2403:                                             ; preds = %2375, %2374
  br i1 true, label %2404, label %2432, !dbg !21

2404:                                             ; preds = %2403
  %2405 = extractvalue [2 x <8 x float>] %213, 1, !dbg !19
  %2406 = shufflevector <8 x float> %2405, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !19
  %row.reciprocal120 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %2314, !dbg !19
  %normalized121 = fmul <4 x float> %2406, %row.reciprocal120, !dbg !19
  %2407 = fptrunc <4 x float> %normalized121 to <4 x half>, !dbg !20
  %2408 = extractelement <4 x half> %2407, i64 3, !dbg !20
  %2409 = extractelement <4 x half> %2407, i64 2, !dbg !20
  %2410 = extractelement <4 x half> %2407, i64 1, !dbg !20
  %2411 = extractelement <4 x half> %2407, i64 0, !dbg !20
  %2412 = shufflevector <8 x float> %2405, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !19
  %normalized122 = fmul <4 x float> %2412, %row.reciprocal120, !dbg !19
  %2413 = fptrunc <4 x float> %normalized122 to <4 x half>, !dbg !20
  %2414 = extractelement <4 x half> %2413, i64 3, !dbg !20
  %2415 = extractelement <4 x half> %2413, i64 2, !dbg !20
  %2416 = extractelement <4 x half> %2413, i64 1, !dbg !20
  %2417 = extractelement <4 x half> %2413, i64 0, !dbg !20
  %2418 = extractvalue [2 x <8 x float>] %213, 0, !dbg !19
  %2419 = shufflevector <8 x float> %2418, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !19
  %row.reciprocal123 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %2312, !dbg !19
  %normalized124 = fmul <4 x float> %2419, %row.reciprocal123, !dbg !19
  %2420 = fptrunc <4 x float> %normalized124 to <4 x half>, !dbg !20
  %2421 = extractelement <4 x half> %2420, i64 3, !dbg !20
  %2422 = extractelement <4 x half> %2420, i64 2, !dbg !20
  %2423 = extractelement <4 x half> %2420, i64 1, !dbg !20
  %2424 = extractelement <4 x half> %2420, i64 0, !dbg !20
  %2425 = shufflevector <8 x float> %2418, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !19
  %normalized125 = fmul <4 x float> %2425, %row.reciprocal123, !dbg !19
  %2426 = fptrunc <4 x float> %normalized125 to <4 x half>, !dbg !20
  %2427 = extractelement <4 x half> %2426, i64 3, !dbg !20
  %2428 = extractelement <4 x half> %2426, i64 2, !dbg !20
  %2429 = extractelement <4 x half> %2426, i64 1, !dbg !20
  %2430 = extractelement <4 x half> %2426, i64 0, !dbg !20
  %2431 = shl nuw nsw i64 %24, 19, !dbg !21
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
  %address.local1111 = add nuw nsw i64 %2431, %address.local1110, !dbg !21
  %address.local1112 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1111, !dbg !21
  %address.local1113 = lshr i64 %address.local1102, 2, !dbg !11
  %address.local1114 = and i64 %address.local1113, 12, !dbg !11
  %address.local1115 = or disjoint i64 %address.local1114, 96, !dbg !21
  %address.local1116 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1112, i64 %address.local1115, !dbg !21
  store half %2430, ptr addrspace(1) %address.local1116, align 2, !dbg !21
  %address.local1117 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1111, !dbg !21
  %address.local1118 = or disjoint i64 %address.local1114, 97, !dbg !21
  %address.local1119 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1117, i64 %address.local1118, !dbg !21
  store half %2429, ptr addrspace(1) %address.local1119, align 2, !dbg !21
  %address.local1120 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1111, !dbg !21
  %address.local1121 = or disjoint i64 %address.local1114, 98, !dbg !21
  %address.local1122 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1120, i64 %address.local1121, !dbg !21
  store half %2428, ptr addrspace(1) %address.local1122, align 2, !dbg !21
  %address.local1123 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1111, !dbg !21
  %address.local1124 = or disjoint i64 %address.local1114, 99, !dbg !21
  %address.local1125 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1123, i64 %address.local1124, !dbg !21
  store half %2427, ptr addrspace(1) %address.local1125, align 2, !dbg !21
  %address.local1126 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1111, !dbg !21
  %address.local1127 = or disjoint i64 %address.local1114, 112, !dbg !21
  %address.local1128 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1126, i64 %address.local1127, !dbg !21
  store half %2424, ptr addrspace(1) %address.local1128, align 2, !dbg !21
  %address.local1129 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1111, !dbg !21
  %address.local1130 = or disjoint i64 %address.local1114, 113, !dbg !21
  %address.local1131 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1129, i64 %address.local1130, !dbg !21
  store half %2423, ptr addrspace(1) %address.local1131, align 2, !dbg !21
  %address.local1132 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1111, !dbg !21
  %address.local1133 = or disjoint i64 %address.local1114, 114, !dbg !21
  %address.local1134 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1132, i64 %address.local1133, !dbg !21
  store half %2422, ptr addrspace(1) %address.local1134, align 2, !dbg !21
  %address.local1135 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1111, !dbg !21
  %address.local1136 = or i64 %address.local1113, 115, !dbg !21
  %address.local1137 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1135, i64 %address.local1136, !dbg !21
  store half %2421, ptr addrspace(1) %address.local1137, align 2, !dbg !21
  %address.local1138 = shl i64 %address.local1109, 7, !dbg !21
  %address.local1139 = add i64 %address.local1138, 8192, !dbg !21
  %address.local1140 = add nuw nsw i64 %2431, %address.local1139, !dbg !21
  %address.local1141 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !21
  %address.local1142 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1141, i64 %address.local1115, !dbg !21
  store half %2417, ptr addrspace(1) %address.local1142, align 2, !dbg !21
  %address.local1143 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !21
  %address.local1144 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1143, i64 %address.local1118, !dbg !21
  store half %2416, ptr addrspace(1) %address.local1144, align 2, !dbg !21
  %address.local1145 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !21
  %address.local1146 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1145, i64 %address.local1121, !dbg !21
  store half %2415, ptr addrspace(1) %address.local1146, align 2, !dbg !21
  %address.local1147 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !21
  %address.local1148 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1147, i64 %address.local1124, !dbg !21
  store half %2414, ptr addrspace(1) %address.local1148, align 2, !dbg !21
  %address.local1149 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !21
  %address.local1150 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1149, i64 %address.local1127, !dbg !21
  store half %2411, ptr addrspace(1) %address.local1150, align 2, !dbg !21
  %address.local1151 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !21
  %address.local1152 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1151, i64 %address.local1130, !dbg !21
  store half %2410, ptr addrspace(1) %address.local1152, align 2, !dbg !21
  %address.local1153 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !21
  %address.local1154 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1153, i64 %address.local1133, !dbg !21
  store half %2409, ptr addrspace(1) %address.local1154, align 2, !dbg !21
  %address.local1155 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !21
  %address.local1156 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1155, i64 %address.local1136, !dbg !21
  store half %2408, ptr addrspace(1) %address.local1156, align 2, !dbg !21
  br label %2432, !dbg !21

2432:                                             ; preds = %2404, %2403
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

!0 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/online_softmax/final/run-20261023-171724-w2j94qwt/retained32-baseline")
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
