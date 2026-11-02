; ModuleID = 'generated_ir/online_softmax_native_acc/kernel.ll'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/online_softmax_native_acc/final-export/run-20261023-182556-l6_z6lbl/input-0001-test_input_1x32x4096x128-128-128-32-online/retained32-baseline/input.mlir"

@shm_0 = addrspace(3) global [16384 x i8] undef, align 16

; Function Attrs: convergent willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

declare float @__ocml_exp2_f32(float)

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3) #1 !dbg !37 !reqd_work_group_size !65 {
  %5 = call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !38
  %6 = sext i32 %5 to i64, !dbg !38
  %7 = call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !38
  %8 = sext i32 %7 to i64, !dbg !38
  %9 = shl nsw i64 %8, 5, !dbg !38
  %10 = add nsw i64 %9, %6, !dbg !38
  %11 = lshr i64 %10, 5, !dbg !38
  %12 = and i64 %11, 576460752303423484, !dbg !38
  %13 = and i64 %6, 3, !dbg !38
  %14 = or disjoint i64 %12, %13, !dbg !38
  %15 = sub nsw i64 31, %14, !dbg !38
  %16 = lshr i64 %10, 2, !dbg !38
  %17 = icmp ugt i64 %10, 1023, !dbg !40
  %18 = add nsw i64 %14, -32, !dbg !40
  %19 = select i1 %17, i64 %18, i64 %15, !dbg !40
  %20 = sdiv i64 %19, 32, !dbg !40
  %21 = sext i1 %17 to i64, !dbg !40
  %22 = xor i64 %20, %21, !dbg !40
  %23 = add nsw i64 %16, %22, !dbg !40
  %24 = and i64 %23, 31, !dbg !40
  %25 = shl nsw i64 %15, 7, !dbg !40
  %26 = mul nsw i64 %22, -4096, !dbg !40
  %27 = add i64 %25, %26, !dbg !40
  %28 = shl nuw nsw i64 %24, 19, !dbg !40
  %29 = shl nsw i64 %27, 7, !dbg !40
  %30 = add i64 %28, %29, !dbg !40
  %31 = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !40, !range !67
  %32 = zext nneg i32 %31 to i64, !dbg !40
  %33 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %30, !dbg !40
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !40, !range !67, !frisk.register_lifetimes !34
  %address.local = zext nneg i32 %address.thread to i64, !dbg !40
  %address.local339 = shl nuw nsw i64 %address.local, 3, !dbg !40
  %address.local340 = and i64 %address.local339, 1536, !dbg !40
  %address.local341 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local340, !dbg !40
  %address.local342 = ptrtoint ptr addrspace(3) %address.local341 to i64, !dbg !40
  %address.local343 = trunc i64 %address.local342 to i32, !dbg !40
  %34 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local343), !dbg !40
  %35 = zext i32 %34 to i64, !dbg !40
  %36 = inttoptr i64 %35 to ptr addrspace(3), !dbg !40
  %address.local344 = shl nuw nsw i64 %address.local, 7, !dbg !40
  %address.local345 = and i64 %address.local344, 16256, !dbg !40
  %address.local346 = lshr i64 %address.local, 4, !dbg !40
  %address.local347 = and i64 %address.local346, 8, !dbg !40
  %address.local348 = or disjoint i64 %address.local345, %address.local347, !dbg !40
  %address.local349 = getelementptr [2 x i8], ptr addrspace(1) %33, i64 %address.local348, !dbg !40
  %37 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !40
  %38 = ptrtoint ptr addrspace(1) %address.local349 to i64, !dbg !40
  %39 = sub i64 %38, %37, !dbg !40
  %dma.buffer.offset1612 = trunc i64 %39 to i32, !dbg !40
  %40 = trunc i64 %37 to i32, !dbg !40
  %41 = insertelement <4 x i32> poison, i32 %40, i64 0, !dbg !40
  %42 = lshr i64 %37, 32, !dbg !40
  %43 = trunc i64 %42 to i32, !dbg !40
  %44 = insertelement <4 x i32> %41, i32 %43, i64 1, !dbg !40
  %45 = insertelement <4 x i32> %44, i32 33554432, i64 2, !dbg !40
  %dma.buffer.rsrc1613 = insertelement <4 x i32> %45, i32 131072, i64 3, !dbg !40
  %46 = ptrtoint ptr addrspace(3) %36 to i32, !dbg !40
  %dma.buffer.lds1614 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %46), !dbg !40
  %47 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1614, i32 %dma.buffer.offset1612, <4 x i32> %dma.buffer.rsrc1613) #10, !dbg !40
  %address.local350 = or disjoint i64 %address.local, 256, !dbg !40
  %address.local351 = shl nuw nsw i64 %address.local350, 3, !dbg !40
  %address.local352 = and i64 %address.local351, 3584, !dbg !40
  %address.local353 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local352, !dbg !40
  %address.local354 = ptrtoint ptr addrspace(3) %address.local353 to i64, !dbg !40
  %address.local355 = trunc i64 %address.local354 to i32, !dbg !40
  %48 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local355), !dbg !40
  %49 = zext i32 %48 to i64, !dbg !40
  %50 = inttoptr i64 %49 to ptr addrspace(3), !dbg !40
  %address.local356 = shl nuw nsw i64 %address.local, 7, !dbg !40
  %address.local357 = and i64 %address.local356, 16256, !dbg !40
  %address.local358 = lshr i64 %address.local350, 4, !dbg !40
  %address.local359 = and i64 %address.local358, 24, !dbg !40
  %address.local360 = or disjoint i64 %address.local357, %address.local359, !dbg !40
  %address.local361 = getelementptr [2 x i8], ptr addrspace(1) %33, i64 %address.local360, !dbg !40
  %51 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !40
  %52 = ptrtoint ptr addrspace(1) %address.local361 to i64, !dbg !40
  %53 = sub i64 %52, %51, !dbg !40
  %dma.buffer.offset1609 = trunc i64 %53 to i32, !dbg !40
  %54 = trunc i64 %51 to i32, !dbg !40
  %55 = insertelement <4 x i32> poison, i32 %54, i64 0, !dbg !40
  %56 = lshr i64 %51, 32, !dbg !40
  %57 = trunc i64 %56 to i32, !dbg !40
  %58 = insertelement <4 x i32> %55, i32 %57, i64 1, !dbg !40
  %59 = insertelement <4 x i32> %58, i32 33554432, i64 2, !dbg !40
  %dma.buffer.rsrc1610 = insertelement <4 x i32> %59, i32 131072, i64 3, !dbg !40
  %60 = ptrtoint ptr addrspace(3) %50 to i32, !dbg !40
  %dma.buffer.lds1611 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %60), !dbg !40
  %61 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1611, i32 %dma.buffer.offset1609, <4 x i32> %dma.buffer.rsrc1610) #10, !dbg !40
  %62 = call i32 @llvm.amdgcn.workgroup.id.z(), !dbg !38
  %63 = sext i32 %62 to i64, !dbg !38
  %64 = shl i64 %15, 7, !dbg !41
  %65 = lshr i64 %32, 6, !dbg !42
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !40
  fence syncscope("workgroup") release, !dbg !40
  call void @llvm.amdgcn.s.barrier(), !dbg !40
  fence syncscope("workgroup") acquire, !dbg !40
  %66 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %30, !dbg !40
  %67 = getelementptr i8, ptr addrspace(1) %66, i64 64, !dbg !40
  %address.local362 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local340, !dbg !40
  %address.local363 = ptrtoint ptr addrspace(3) %address.local362 to i64, !dbg !40
  %address.local364 = trunc i64 %address.local363 to i32, !dbg !40
  %68 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local364), !dbg !40
  %69 = zext i32 %68 to i64, !dbg !40
  %70 = inttoptr i64 %69 to ptr addrspace(3), !dbg !40
  %address.local365 = getelementptr [2 x i8], ptr addrspace(1) %67, i64 %address.local348, !dbg !40
  %71 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !40
  %72 = ptrtoint ptr addrspace(1) %address.local365 to i64, !dbg !40
  %73 = sub i64 %72, %71, !dbg !40
  %dma.buffer.offset1606 = trunc i64 %73 to i32, !dbg !40
  %74 = trunc i64 %71 to i32, !dbg !40
  %75 = insertelement <4 x i32> poison, i32 %74, i64 0, !dbg !40
  %76 = lshr i64 %71, 32, !dbg !40
  %77 = trunc i64 %76 to i32, !dbg !40
  %78 = insertelement <4 x i32> %75, i32 %77, i64 1, !dbg !40
  %79 = insertelement <4 x i32> %78, i32 33554432, i64 2, !dbg !40
  %dma.buffer.rsrc1607 = insertelement <4 x i32> %79, i32 131072, i64 3, !dbg !40
  %80 = ptrtoint ptr addrspace(3) %70 to i32, !dbg !40
  %dma.buffer.lds1608 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %80), !dbg !40
  %81 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1608, i32 %dma.buffer.offset1606, <4 x i32> %dma.buffer.rsrc1607) #10, !dbg !40
  %address.local366 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local352, !dbg !40
  %address.local367 = ptrtoint ptr addrspace(3) %address.local366 to i64, !dbg !40
  %address.local368 = trunc i64 %address.local367 to i32, !dbg !40
  %82 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local368), !dbg !40
  %83 = zext i32 %82 to i64, !dbg !40
  %84 = inttoptr i64 %83 to ptr addrspace(3), !dbg !40
  %address.local369 = getelementptr [2 x i8], ptr addrspace(1) %67, i64 %address.local360, !dbg !40
  %85 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !40
  %86 = ptrtoint ptr addrspace(1) %address.local369 to i64, !dbg !40
  %87 = sub i64 %86, %85, !dbg !40
  %dma.buffer.offset1603 = trunc i64 %87 to i32, !dbg !40
  %88 = trunc i64 %85 to i32, !dbg !40
  %89 = insertelement <4 x i32> poison, i32 %88, i64 0, !dbg !40
  %90 = lshr i64 %85, 32, !dbg !40
  %91 = trunc i64 %90 to i32, !dbg !40
  %92 = insertelement <4 x i32> %89, i32 %91, i64 1, !dbg !40
  %93 = insertelement <4 x i32> %92, i32 33554432, i64 2, !dbg !40
  %dma.buffer.rsrc1604 = insertelement <4 x i32> %93, i32 131072, i64 3, !dbg !40
  %94 = ptrtoint ptr addrspace(3) %84 to i32, !dbg !40
  %dma.buffer.lds1605 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %94), !dbg !40
  %95 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1605, i32 %dma.buffer.offset1603, <4 x i32> %dma.buffer.rsrc1604) #10, !dbg !40
  %address.local370 = lshr i32 %address.thread, 2, !dbg !42
  %address.local371 = and i32 %address.local370, 4, !dbg !42
  %address.local372 = shl nuw nsw i32 %address.thread, 1, !dbg !42
  %address.local373 = and i32 %address.local372, 384, !dbg !42
  %address.local374 = or disjoint i32 %address.local371, %address.local373, !dbg !42
  %address.local375 = shl nuw nsw i32 %address.thread, 3, !dbg !42
  %address.local376 = and i32 %address.local375, 120, !dbg !42
  %address.local377 = or disjoint i32 %address.local374, %address.local376, !dbg !42
  %address.local378 = shl nuw nsw i32 %address.thread, 5, !dbg !42
  %address.local379 = and i32 %address.local378, 1024, !dbg !42
  %address.local380 = or disjoint i32 %address.local377, %address.local379, !dbg !42
  %address.local381 = zext nneg i32 %address.local380 to i64, !dbg !42
  %address.local382 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local381, !dbg !42
  %96 = load <4 x half>, ptr addrspace(3) %address.local382, align 8, !dbg !42
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local382, i64 4096, !dbg !42
  %97 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !42
  %98 = fmul <4 x half> %96, splat (half 1.275630e-01), !dbg !42
  %lds.const.gep1413 = getelementptr i8, ptr addrspace(3) %address.local382, i64 1024, !dbg !42
  %99 = load <4 x half>, ptr addrspace(3) %lds.const.gep1413, align 8, !dbg !42
  %100 = fmul <4 x half> %97, splat (half 1.275630e-01), !dbg !42
  %lds.const.gep1414 = getelementptr i8, ptr addrspace(3) %address.local382, i64 5120, !dbg !42
  %101 = load <4 x half>, ptr addrspace(3) %lds.const.gep1414, align 8, !dbg !42
  %102 = fmul <4 x half> %99, splat (half 1.275630e-01), !dbg !42
  %103 = fmul <4 x half> %101, splat (half 1.275630e-01), !dbg !42
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !40
  fence syncscope("workgroup") release, !dbg !40
  call void @llvm.amdgcn.s.barrier(), !dbg !40
  fence syncscope("workgroup") acquire, !dbg !40
  %104 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %30, !dbg !40
  %105 = getelementptr i8, ptr addrspace(1) %104, i64 128, !dbg !40
  %106 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local343), !dbg !40
  %107 = zext i32 %106 to i64, !dbg !40
  %108 = inttoptr i64 %107 to ptr addrspace(3), !dbg !40
  %address.local425 = getelementptr [2 x i8], ptr addrspace(1) %105, i64 %address.local348, !dbg !40
  %109 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !40
  %110 = ptrtoint ptr addrspace(1) %address.local425 to i64, !dbg !40
  %111 = sub i64 %110, %109, !dbg !40
  %dma.buffer.offset1600 = trunc i64 %111 to i32, !dbg !40
  %112 = trunc i64 %109 to i32, !dbg !40
  %113 = insertelement <4 x i32> poison, i32 %112, i64 0, !dbg !40
  %114 = lshr i64 %109, 32, !dbg !40
  %115 = trunc i64 %114 to i32, !dbg !40
  %116 = insertelement <4 x i32> %113, i32 %115, i64 1, !dbg !40
  %117 = insertelement <4 x i32> %116, i32 33554432, i64 2, !dbg !40
  %dma.buffer.rsrc1601 = insertelement <4 x i32> %117, i32 131072, i64 3, !dbg !40
  %118 = ptrtoint ptr addrspace(3) %108 to i32, !dbg !40
  %dma.buffer.lds1602 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %118), !dbg !40
  %119 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1602, i32 %dma.buffer.offset1600, <4 x i32> %dma.buffer.rsrc1601) #10, !dbg !40
  %120 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local355), !dbg !40
  %121 = zext i32 %120 to i64, !dbg !40
  %122 = inttoptr i64 %121 to ptr addrspace(3), !dbg !40
  %address.local426 = getelementptr [2 x i8], ptr addrspace(1) %105, i64 %address.local360, !dbg !40
  %123 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !40
  %124 = ptrtoint ptr addrspace(1) %address.local426 to i64, !dbg !40
  %125 = sub i64 %124, %123, !dbg !40
  %dma.buffer.offset1597 = trunc i64 %125 to i32, !dbg !40
  %126 = trunc i64 %123 to i32, !dbg !40
  %127 = insertelement <4 x i32> poison, i32 %126, i64 0, !dbg !40
  %128 = lshr i64 %123, 32, !dbg !40
  %129 = trunc i64 %128 to i32, !dbg !40
  %130 = insertelement <4 x i32> %127, i32 %129, i64 1, !dbg !40
  %131 = insertelement <4 x i32> %130, i32 33554432, i64 2, !dbg !40
  %dma.buffer.rsrc1598 = insertelement <4 x i32> %131, i32 131072, i64 3, !dbg !40
  %132 = ptrtoint ptr addrspace(3) %122 to i32, !dbg !40
  %dma.buffer.lds1599 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %132), !dbg !40
  %133 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1599, i32 %dma.buffer.offset1597, <4 x i32> %dma.buffer.rsrc1598) #10, !dbg !40
  %lds.const.gep1415 = getelementptr i8, ptr addrspace(3) %address.local382, i64 8192, !dbg !42
  %134 = load <4 x half>, ptr addrspace(3) %lds.const.gep1415, align 8, !dbg !42
  %lds.const.gep1416 = getelementptr i8, ptr addrspace(3) %address.local382, i64 12288, !dbg !42
  %135 = load <4 x half>, ptr addrspace(3) %lds.const.gep1416, align 8, !dbg !42
  %lds.const.gep1417 = getelementptr i8, ptr addrspace(3) %address.local382, i64 9216, !dbg !42
  %136 = load <4 x half>, ptr addrspace(3) %lds.const.gep1417, align 8, !dbg !42
  %137 = fmul <4 x half> %134, splat (half 1.275630e-01), !dbg !42
  %lds.const.gep1418 = getelementptr i8, ptr addrspace(3) %address.local382, i64 13312, !dbg !42
  %138 = load <4 x half>, ptr addrspace(3) %lds.const.gep1418, align 8, !dbg !42
  %139 = fmul <4 x half> %135, splat (half 1.275630e-01), !dbg !42
  %140 = fmul <4 x half> %136, splat (half 1.275630e-01), !dbg !42
  %141 = fmul <4 x half> %138, splat (half 1.275630e-01), !dbg !42
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !40
  fence syncscope("workgroup") release, !dbg !40
  call void @llvm.amdgcn.s.barrier(), !dbg !40
  fence syncscope("workgroup") acquire, !dbg !40
  %142 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %30, !dbg !40
  %143 = getelementptr i8, ptr addrspace(1) %142, i64 192, !dbg !40
  %144 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local364), !dbg !40
  %145 = zext i32 %144 to i64, !dbg !40
  %146 = inttoptr i64 %145 to ptr addrspace(3), !dbg !40
  %address.local431 = getelementptr [2 x i8], ptr addrspace(1) %143, i64 %address.local348, !dbg !40
  %147 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !40
  %148 = ptrtoint ptr addrspace(1) %address.local431 to i64, !dbg !40
  %149 = sub i64 %148, %147, !dbg !40
  %dma.buffer.offset1594 = trunc i64 %149 to i32, !dbg !40
  %150 = trunc i64 %147 to i32, !dbg !40
  %151 = insertelement <4 x i32> poison, i32 %150, i64 0, !dbg !40
  %152 = lshr i64 %147, 32, !dbg !40
  %153 = trunc i64 %152 to i32, !dbg !40
  %154 = insertelement <4 x i32> %151, i32 %153, i64 1, !dbg !40
  %155 = insertelement <4 x i32> %154, i32 33554432, i64 2, !dbg !40
  %dma.buffer.rsrc1595 = insertelement <4 x i32> %155, i32 131072, i64 3, !dbg !40
  %156 = ptrtoint ptr addrspace(3) %146 to i32, !dbg !40
  %dma.buffer.lds1596 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %156), !dbg !40
  %157 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1596, i32 %dma.buffer.offset1594, <4 x i32> %dma.buffer.rsrc1595) #10, !dbg !40
  %158 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local368), !dbg !40
  %159 = zext i32 %158 to i64, !dbg !40
  %160 = inttoptr i64 %159 to ptr addrspace(3), !dbg !40
  %address.local432 = getelementptr [2 x i8], ptr addrspace(1) %143, i64 %address.local360, !dbg !40
  %161 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !40
  %162 = ptrtoint ptr addrspace(1) %address.local432 to i64, !dbg !40
  %163 = sub i64 %162, %161, !dbg !40
  %dma.buffer.offset1591 = trunc i64 %163 to i32, !dbg !40
  %164 = trunc i64 %161 to i32, !dbg !40
  %165 = insertelement <4 x i32> poison, i32 %164, i64 0, !dbg !40
  %166 = lshr i64 %161, 32, !dbg !40
  %167 = trunc i64 %166 to i32, !dbg !40
  %168 = insertelement <4 x i32> %165, i32 %167, i64 1, !dbg !40
  %169 = insertelement <4 x i32> %168, i32 33554432, i64 2, !dbg !40
  %dma.buffer.rsrc1592 = insertelement <4 x i32> %169, i32 131072, i64 3, !dbg !40
  %170 = ptrtoint ptr addrspace(3) %160 to i32, !dbg !40
  %dma.buffer.lds1593 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %170), !dbg !40
  %171 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1593, i32 %dma.buffer.offset1591, <4 x i32> %dma.buffer.rsrc1592) #10, !dbg !40
  %172 = load <4 x half>, ptr addrspace(3) %address.local382, align 8, !dbg !42
  %lds.const.gep1419 = getelementptr i8, ptr addrspace(3) %address.local382, i64 4096, !dbg !42
  %173 = load <4 x half>, ptr addrspace(3) %lds.const.gep1419, align 8, !dbg !42
  %lds.const.gep1420 = getelementptr i8, ptr addrspace(3) %address.local382, i64 1024, !dbg !42
  %174 = load <4 x half>, ptr addrspace(3) %lds.const.gep1420, align 8, !dbg !42
  %175 = fmul <4 x half> %172, splat (half 1.275630e-01), !dbg !42
  %lds.const.gep1421 = getelementptr i8, ptr addrspace(3) %address.local382, i64 5120, !dbg !42
  %176 = load <4 x half>, ptr addrspace(3) %lds.const.gep1421, align 8, !dbg !42
  %177 = fmul <4 x half> %173, splat (half 1.275630e-01), !dbg !42
  %178 = fmul <4 x half> %174, splat (half 1.275630e-01), !dbg !42
  %179 = fmul <4 x half> %176, splat (half 1.275630e-01), !dbg !42
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !40
  fence syncscope("workgroup") release, !dbg !40
  call void @llvm.amdgcn.s.barrier(), !dbg !40
  fence syncscope("workgroup") acquire, !dbg !40
  %lds.const.gep1422 = getelementptr i8, ptr addrspace(3) %address.local382, i64 8192, !dbg !42
  %180 = load <4 x half>, ptr addrspace(3) %lds.const.gep1422, align 8, !dbg !42
  %lds.const.gep1423 = getelementptr i8, ptr addrspace(3) %address.local382, i64 12288, !dbg !42
  %181 = load <4 x half>, ptr addrspace(3) %lds.const.gep1423, align 8, !dbg !42
  %lds.const.gep1424 = getelementptr i8, ptr addrspace(3) %address.local382, i64 9216, !dbg !42
  %182 = load <4 x half>, ptr addrspace(3) %lds.const.gep1424, align 8, !dbg !42
  %183 = fmul <4 x half> %180, splat (half 1.275630e-01), !dbg !42
  %lds.const.gep1425 = getelementptr i8, ptr addrspace(3) %address.local382, i64 13312, !dbg !42
  %184 = load <4 x half>, ptr addrspace(3) %lds.const.gep1425, align 8, !dbg !42
  %185 = fmul <4 x half> %181, splat (half 1.275630e-01), !dbg !42
  %186 = fmul <4 x half> %182, splat (half 1.275630e-01), !dbg !42
  %187 = fmul <4 x half> %184, splat (half 1.275630e-01), !dbg !42
  fence syncscope("workgroup") release, !dbg !39
  call void @llvm.amdgcn.s.barrier(), !dbg !39
  fence syncscope("workgroup") acquire, !dbg !39
  %188 = or disjoint i64 %64, 127, !dbg !43
  %189 = add i64 %64, 128, !dbg !43
  %190 = icmp uge i64 %189, %188, !dbg !43
  %191 = sub nsw i64 32, %14, !dbg !45
  %192 = shl i64 %16, 7, !dbg !46
  %193 = and i64 %192, 3968, !dbg !46
  %194 = shl nsw i64 %63, 12, !dbg !46
  %195 = or disjoint i64 %193, %194, !dbg !46
  %196 = lshr i64 %32, 2, !dbg !47
  %197 = shl nuw nsw i64 %32, 2, !dbg !47
  %198 = mul nsw i64 %196, -16, !dbg !47
  %199 = add nsw i64 %197, %198, !dbg !47
  %200 = and i32 %31, 1, !dbg !47
  %index.compact279 = zext nneg i32 %200 to i64, !dbg !47
  %201 = or disjoint i64 %199, %index.compact279, !dbg !47
  %202 = add nuw nsw i64 %201, 1, !dbg !47
  %203 = sdiv i64 %202, 2, !dbg !47
  br label %204, !dbg !39

204:                                              ; preds = %863, %4
  %205 = phi i64 [ %1760, %863 ], [ 0, %4 ], !dbg !39
  %206 = phi [2 x <8 x float>] [ %1745, %863 ], [ zeroinitializer, %4 ], !dbg !39
  %207 = phi [2 x <1 x float>] [ %1351, %863 ], [ zeroinitializer, %4 ], !dbg !39
  %208 = phi [2 x <1 x float>] [ %1747, %863 ], [ [<1 x float> splat (float -inf), <1 x float> splat (float -inf)], %4 ], !dbg !39
  %209 = phi [2 x <8 x float>] [ %1751, %863 ], [ zeroinitializer, %4 ], !dbg !39
  %210 = phi [2 x <8 x float>] [ %1755, %863 ], [ zeroinitializer, %4 ], !dbg !39
  %211 = phi [2 x <8 x float>] [ %1759, %863 ], [ zeroinitializer, %4 ], !dbg !39
  %212 = icmp slt i64 %205, %191, !dbg !39
  br i1 %212, label %213, label %1761, !dbg !39

213:                                              ; preds = %204
  %214 = shl nsw i64 %205, 7, !dbg !39
  %215 = lshr i64 %205, 12, !dbg !46
  %216 = add nuw nsw i64 %215, %16, !dbg !46
  %217 = lshr i64 %205, 5, !dbg !46
  %218 = and i64 %217, 127, !dbg !46
  %219 = shl i64 %216, 19, !dbg !46
  %220 = and i64 %219, 16252928, !dbg !46
  %221 = shl i64 %217, 19, !dbg !46
  %222 = shl i64 %205, 14, !dbg !46
  %223 = sub i64 %222, %221, !dbg !46
  %224 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %220, !dbg !46
  %225 = getelementptr [2 x i8], ptr addrspace(1) %224, i64 %218, !dbg !46
  %226 = getelementptr [2 x i8], ptr addrspace(1) %225, i64 %223, !dbg !46
  %address.thread433 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !46, !range !67, !frisk.register_lifetimes !34
  %address.local434 = zext nneg i32 %address.thread433 to i64, !dbg !40
  %address.local435 = shl nuw nsw i64 %address.local434, 3, !dbg !40
  %address.local436 = and i64 %address.local435, 1536, !dbg !40
  %address.local437 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local436, !dbg !40
  %address.local438 = ptrtoint ptr addrspace(3) %address.local437 to i64, !dbg !40
  %address.local439 = trunc i64 %address.local438 to i32, !dbg !40
  %227 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local439), !dbg !46
  %228 = zext i32 %227 to i64, !dbg !46
  %229 = inttoptr i64 %228 to ptr addrspace(3), !dbg !46
  %address.local440 = lshr i64 %address.local434, 4, !dbg !46
  %address.local441 = and i64 %address.local440, 8, !dbg !46
  %address.local442 = and i64 %address.local434, 123, !dbg !46
  %address.local443 = lshr i64 %address.local434, 2, !dbg !46
  %address.local444 = and i64 %address.local443, 4, !dbg !46
  %address.local445 = add nuw nsw i64 %address.local444, %address.local434, !dbg !46
  %address.local446 = and i64 %address.local445, 4, !dbg !46
  %address.local447 = or disjoint i64 %address.local442, %address.local446, !dbg !46
  %address.local448 = shl nuw nsw i64 %address.local447, 7, !dbg !46
  %address.local449 = or disjoint i64 %address.local441, %address.local448, !dbg !46
  %address.local450 = getelementptr [2 x i8], ptr addrspace(1) %226, i64 %address.local449, !dbg !46
  %230 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !46
  %231 = ptrtoint ptr addrspace(1) %address.local450 to i64, !dbg !46
  %232 = sub i64 %231, %230, !dbg !46
  %dma.buffer.offset1588 = trunc i64 %232 to i32, !dbg !46
  %233 = trunc i64 %230 to i32, !dbg !46
  %234 = insertelement <4 x i32> poison, i32 %233, i64 0, !dbg !46
  %235 = lshr i64 %230, 32, !dbg !46
  %236 = trunc i64 %235 to i32, !dbg !46
  %237 = insertelement <4 x i32> %234, i32 %236, i64 1, !dbg !46
  %238 = insertelement <4 x i32> %237, i32 33554432, i64 2, !dbg !46
  %dma.buffer.rsrc1589 = insertelement <4 x i32> %238, i32 131072, i64 3, !dbg !46
  %239 = ptrtoint ptr addrspace(3) %229 to i32, !dbg !46
  %dma.buffer.lds1590 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %239), !dbg !46
  %240 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1590, i32 %dma.buffer.offset1588, <4 x i32> %dma.buffer.rsrc1589) #10, !dbg !46
  %address.local451 = or disjoint i64 %address.local434, 256, !dbg !40
  %address.local452 = shl nuw nsw i64 %address.local451, 3, !dbg !40
  %address.local453 = and i64 %address.local452, 3584, !dbg !40
  %address.local454 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local453, !dbg !40
  %address.local455 = ptrtoint ptr addrspace(3) %address.local454 to i64, !dbg !40
  %address.local456 = trunc i64 %address.local455 to i32, !dbg !40
  %241 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local456), !dbg !46
  %242 = zext i32 %241 to i64, !dbg !46
  %243 = inttoptr i64 %242 to ptr addrspace(3), !dbg !46
  %address.local457 = lshr i64 %address.local451, 4, !dbg !46
  %address.local458 = and i64 %address.local457, 24, !dbg !46
  %address.local459 = and i64 %address.local434, 123, !dbg !46
  %address.local460 = lshr i64 %address.local434, 2, !dbg !46
  %address.local461 = and i64 %address.local460, 4, !dbg !46
  %address.local462 = add nuw nsw i64 %address.local461, %address.local434, !dbg !46
  %address.local463 = and i64 %address.local462, 4, !dbg !46
  %address.local464 = or disjoint i64 %address.local459, %address.local463, !dbg !46
  %address.local465 = shl nuw nsw i64 %address.local464, 7, !dbg !46
  %address.local466 = or disjoint i64 %address.local458, %address.local465, !dbg !46
  %address.local467 = getelementptr [2 x i8], ptr addrspace(1) %226, i64 %address.local466, !dbg !46
  %244 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !46
  %245 = ptrtoint ptr addrspace(1) %address.local467 to i64, !dbg !46
  %246 = sub i64 %245, %244, !dbg !46
  %dma.buffer.offset1585 = trunc i64 %246 to i32, !dbg !46
  %247 = trunc i64 %244 to i32, !dbg !46
  %248 = insertelement <4 x i32> poison, i32 %247, i64 0, !dbg !46
  %249 = lshr i64 %244, 32, !dbg !46
  %250 = trunc i64 %249 to i32, !dbg !46
  %251 = insertelement <4 x i32> %248, i32 %250, i64 1, !dbg !46
  %252 = insertelement <4 x i32> %251, i32 33554432, i64 2, !dbg !46
  %dma.buffer.rsrc1586 = insertelement <4 x i32> %252, i32 131072, i64 3, !dbg !46
  %253 = ptrtoint ptr addrspace(3) %243 to i32, !dbg !46
  %dma.buffer.lds1587 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %253), !dbg !46
  %254 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1587, i32 %dma.buffer.offset1585, <4 x i32> %dma.buffer.rsrc1586) #10, !dbg !46
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !46
  fence syncscope("workgroup") release, !dbg !46
  call void @llvm.amdgcn.s.barrier(), !dbg !46
  fence syncscope("workgroup") acquire, !dbg !46
  %255 = add nuw i64 %205, 1024, !dbg !46
  %256 = icmp slt i64 %255, 0, !dbg !46
  %257 = sub nuw i64 -1025, %205, !dbg !46
  %258 = select i1 %256, i64 %257, i64 %255, !dbg !46
  %259 = sdiv i64 %258, 4096, !dbg !46
  %.lobit37 = ashr i64 %255, 63, !dbg !46
  %260 = xor i64 %259, %.lobit37, !dbg !46
  %261 = add nsw i64 %260, %16, !dbg !46
  %262 = add nsw i64 %217, %195, !dbg !46
  %263 = add nsw i64 %262, 32, !dbg !46
  %264 = icmp slt i64 %262, -32, !dbg !46
  %265 = sub nsw i64 -33, %262, !dbg !46
  %266 = select i1 %264, i64 %265, i64 %263, !dbg !46
  %267 = sdiv i64 %266, 128, !dbg !46
  %268 = sext i1 %264 to i64, !dbg !46
  %269 = xor i64 %267, %268, !dbg !46
  %270 = shl nsw i64 %269, 7, !dbg !46
  %271 = sub i64 %195, %270, !dbg !46
  %272 = shl i64 %261, 19, !dbg !46
  %273 = and i64 %272, 16252928, !dbg !46
  %274 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %273, !dbg !46
  %275 = getelementptr [2 x i8], ptr addrspace(1) %274, i64 %217, !dbg !46
  %276 = getelementptr [2 x i8], ptr addrspace(1) %275, i64 %271, !dbg !46
  %277 = getelementptr i8, ptr addrspace(1) %276, i64 64, !dbg !46
  %278 = getelementptr [2 x i8], ptr addrspace(1) %277, i64 %223, !dbg !46
  %address.local468 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local436, !dbg !40
  %address.local469 = ptrtoint ptr addrspace(3) %address.local468 to i64, !dbg !40
  %address.local470 = trunc i64 %address.local469 to i32, !dbg !40
  %279 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local470), !dbg !46
  %280 = zext i32 %279 to i64, !dbg !46
  %281 = inttoptr i64 %280 to ptr addrspace(3), !dbg !46
  %address.local471 = getelementptr [2 x i8], ptr addrspace(1) %278, i64 %address.local449, !dbg !46
  %282 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !46
  %283 = ptrtoint ptr addrspace(1) %address.local471 to i64, !dbg !46
  %284 = sub i64 %283, %282, !dbg !46
  %dma.buffer.offset1582 = trunc i64 %284 to i32, !dbg !46
  %285 = trunc i64 %282 to i32, !dbg !46
  %286 = insertelement <4 x i32> poison, i32 %285, i64 0, !dbg !46
  %287 = lshr i64 %282, 32, !dbg !46
  %288 = trunc i64 %287 to i32, !dbg !46
  %289 = insertelement <4 x i32> %286, i32 %288, i64 1, !dbg !46
  %290 = insertelement <4 x i32> %289, i32 33554432, i64 2, !dbg !46
  %dma.buffer.rsrc1583 = insertelement <4 x i32> %290, i32 131072, i64 3, !dbg !46
  %291 = ptrtoint ptr addrspace(3) %281 to i32, !dbg !46
  %dma.buffer.lds1584 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %291), !dbg !46
  %292 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1584, i32 %dma.buffer.offset1582, <4 x i32> %dma.buffer.rsrc1583) #10, !dbg !46
  %address.local472 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local453, !dbg !40
  %address.local473 = ptrtoint ptr addrspace(3) %address.local472 to i64, !dbg !40
  %address.local474 = trunc i64 %address.local473 to i32, !dbg !40
  %293 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local474), !dbg !46
  %294 = zext i32 %293 to i64, !dbg !46
  %295 = inttoptr i64 %294 to ptr addrspace(3), !dbg !46
  %address.local475 = getelementptr [2 x i8], ptr addrspace(1) %278, i64 %address.local466, !dbg !46
  %296 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !46
  %297 = ptrtoint ptr addrspace(1) %address.local475 to i64, !dbg !46
  %298 = sub i64 %297, %296, !dbg !46
  %dma.buffer.offset1579 = trunc i64 %298 to i32, !dbg !46
  %299 = trunc i64 %296 to i32, !dbg !46
  %300 = insertelement <4 x i32> poison, i32 %299, i64 0, !dbg !46
  %301 = lshr i64 %296, 32, !dbg !46
  %302 = trunc i64 %301 to i32, !dbg !46
  %303 = insertelement <4 x i32> %300, i32 %302, i64 1, !dbg !46
  %304 = insertelement <4 x i32> %303, i32 33554432, i64 2, !dbg !46
  %dma.buffer.rsrc1580 = insertelement <4 x i32> %304, i32 131072, i64 3, !dbg !46
  %305 = ptrtoint ptr addrspace(3) %295 to i32, !dbg !46
  %dma.buffer.lds1581 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %305), !dbg !46
  %306 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1581, i32 %dma.buffer.offset1579, <4 x i32> %dma.buffer.rsrc1580) #10, !dbg !46
  %address.local476 = mul nuw nsw i64 %address.local434, 160, !dbg !47
  %address.local477 = lshr i64 %address.local434, 2, !dbg !47
  %address.local478 = mul nsw i64 %address.local477, -640, !dbg !47
  %address.local479 = add nsw i64 %address.local476, %address.local478, !dbg !47
  %address.local480 = shl nuw nsw i64 %address.local434, 1, !dbg !47
  %address.local481 = and i64 %address.local480, 24, !dbg !47
  %address.local482 = or disjoint i64 %address.local479, %address.local481, !dbg !47
  %address.local483 = shl i64 %address.local434, 62, !dbg !47
  %address.local484 = ashr exact i64 %address.local483, 57, !dbg !47
  %address.local485 = and i64 %address.local484, -64, !dbg !47
  %address.local486 = add nsw i64 %address.local482, %address.local485, !dbg !47
  %address.local487 = lshr i64 %address.local434, 2, !dbg !42
  %address.local488 = and i64 %address.local487, 12, !dbg !42
  %address.local489 = add nsw i64 %address.local486, %address.local488, !dbg !47
  %address.local490 = and i64 %address.local434, 63, !dbg !42
  %address.local491 = lshr i64 %address.local490, 5, !dbg !42
  %address.local492 = mul nuw nsw i64 %address.local491, 1016, !dbg !42
  %address.local493 = add i64 %address.local489, %address.local492, !dbg !47
  %address.local494 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local493, !dbg !47
  %307 = load <4 x half>, ptr addrspace(3) %address.local494, align 8, !dbg !47
  %308 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %307, <4 x float> zeroinitializer) #9, !dbg !47
  %lds.const.gep1426 = getelementptr i8, ptr addrspace(3) %address.local494, i64 4096, !dbg !47
  %309 = load <4 x half>, ptr addrspace(3) %lds.const.gep1426, align 8, !dbg !47
  %address.local505 = shl nuw nsw i32 %address.thread433, 1, !dbg !47
  %address.local506 = and i32 %address.local505, 24, !dbg !47
  %address.local507 = shl nuw nsw i32 %address.thread433, 8, !dbg !47
  %address.local508 = and i32 %address.local507, 768, !dbg !47
  %address.local509 = or disjoint i32 %address.local506, %address.local508, !dbg !47
  %address.local510 = shl nuw nsw i32 %address.thread433, 5, !dbg !47
  %address.local511 = and i32 %address.local510, 96, !dbg !47
  %address.local512 = or disjoint i32 %address.local509, %address.local511, !dbg !47
  %address.local513 = zext nneg i32 %address.local512 to i64, !dbg !47
  %address.base = call i64 asm sideeffect "", "=v,0"(i64 %203) #9, !dbg !47, !frisk.register_lifetimes !34
  %address.local514 = lshr i32 %address.thread433, 1, !dbg !47
  %address.local515 = and i32 %address.local514, 1, !dbg !47
  %address.local516 = zext nneg i32 %address.local515 to i64, !dbg !47
  %address.local517 = add nsw i64 %address.base, %address.local516, !dbg !47
  %address.local518 = shl i64 %address.local517, 6, !dbg !47
  %address.local519 = sub i64 %address.local513, %address.local518, !dbg !47
  %address.local520 = add i64 %address.local519, %address.local488, !dbg !47
  %address.local521 = add i64 %address.local520, %address.local492, !dbg !47
  %address.local522 = add i64 %address.local521, 32, !dbg !47
  %address.local523 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local522, !dbg !47
  %310 = load <4 x half>, ptr addrspace(3) %address.local523, align 8, !dbg !47
  %311 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %309, <4 x float> %308) #9, !dbg !47
  %312 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %310, <4 x float> zeroinitializer) #9, !dbg !47
  %lds.const.gep1427 = getelementptr i8, ptr addrspace(3) %address.local523, i64 4096, !dbg !47
  %313 = load <4 x half>, ptr addrspace(3) %lds.const.gep1427, align 8, !dbg !47
  %lds.const.gep1428 = getelementptr i8, ptr addrspace(3) %address.local494, i64 128, !dbg !47
  %314 = load <4 x half>, ptr addrspace(3) %lds.const.gep1428, align 8, !dbg !47
  %315 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %313, <4 x float> %312) #9, !dbg !47
  %316 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %314, <4 x float> zeroinitializer) #9, !dbg !47
  %lds.const.gep1429 = getelementptr i8, ptr addrspace(3) %address.local494, i64 4224, !dbg !47
  %317 = load <4 x half>, ptr addrspace(3) %lds.const.gep1429, align 8, !dbg !47
  %lds.const.gep1430 = getelementptr i8, ptr addrspace(3) %address.local523, i64 128, !dbg !47
  %318 = load <4 x half>, ptr addrspace(3) %lds.const.gep1430, align 8, !dbg !47
  %319 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %317, <4 x float> %316) #9, !dbg !47
  %320 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %318, <4 x float> zeroinitializer) #9, !dbg !47
  %lds.const.gep1431 = getelementptr i8, ptr addrspace(3) %address.local523, i64 4224, !dbg !47
  %321 = load <4 x half>, ptr addrspace(3) %lds.const.gep1431, align 8, !dbg !47
  %lds.const.gep1432 = getelementptr i8, ptr addrspace(3) %address.local494, i64 1024, !dbg !47
  %322 = load <4 x half>, ptr addrspace(3) %lds.const.gep1432, align 8, !dbg !47
  %323 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %321, <4 x float> %320) #9, !dbg !47
  %324 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %322, <4 x float> zeroinitializer) #9, !dbg !47
  %lds.const.gep1433 = getelementptr i8, ptr addrspace(3) %address.local494, i64 5120, !dbg !47
  %325 = load <4 x half>, ptr addrspace(3) %lds.const.gep1433, align 8, !dbg !47
  %lds.const.gep1434 = getelementptr i8, ptr addrspace(3) %address.local523, i64 1024, !dbg !47
  %326 = load <4 x half>, ptr addrspace(3) %lds.const.gep1434, align 8, !dbg !47
  %327 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %325, <4 x float> %324) #9, !dbg !47
  %328 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %326, <4 x float> zeroinitializer) #9, !dbg !47
  %lds.const.gep1435 = getelementptr i8, ptr addrspace(3) %address.local523, i64 5120, !dbg !47
  %329 = load <4 x half>, ptr addrspace(3) %lds.const.gep1435, align 8, !dbg !47
  %lds.const.gep1436 = getelementptr i8, ptr addrspace(3) %address.local494, i64 1152, !dbg !47
  %330 = load <4 x half>, ptr addrspace(3) %lds.const.gep1436, align 8, !dbg !47
  %331 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %329, <4 x float> %328) #9, !dbg !47
  %332 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %330, <4 x float> zeroinitializer) #9, !dbg !47
  %lds.const.gep1437 = getelementptr i8, ptr addrspace(3) %address.local494, i64 5248, !dbg !47
  %333 = load <4 x half>, ptr addrspace(3) %lds.const.gep1437, align 8, !dbg !47
  %lds.const.gep1438 = getelementptr i8, ptr addrspace(3) %address.local523, i64 1152, !dbg !47
  %334 = load <4 x half>, ptr addrspace(3) %lds.const.gep1438, align 8, !dbg !47
  %335 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %333, <4 x float> %332) #9, !dbg !47
  %336 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %334, <4 x float> zeroinitializer) #9, !dbg !47
  %lds.const.gep1439 = getelementptr i8, ptr addrspace(3) %address.local523, i64 5248, !dbg !47
  %337 = load <4 x half>, ptr addrspace(3) %lds.const.gep1439, align 8, !dbg !47
  %338 = load <4 x half>, ptr addrspace(3) %address.local494, align 8, !dbg !47
  %339 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %100, <4 x half> %337, <4 x float> %336) #9, !dbg !47
  %340 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %338, <4 x float> zeroinitializer) #9, !dbg !47
  %lds.const.gep1440 = getelementptr i8, ptr addrspace(3) %address.local494, i64 4096, !dbg !47
  %341 = load <4 x half>, ptr addrspace(3) %lds.const.gep1440, align 8, !dbg !47
  %342 = load <4 x half>, ptr addrspace(3) %address.local523, align 8, !dbg !47
  %343 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %341, <4 x float> %340) #9, !dbg !47
  %344 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %342, <4 x float> zeroinitializer) #9, !dbg !47
  %lds.const.gep1441 = getelementptr i8, ptr addrspace(3) %address.local523, i64 4096, !dbg !47
  %345 = load <4 x half>, ptr addrspace(3) %lds.const.gep1441, align 8, !dbg !47
  %lds.const.gep1442 = getelementptr i8, ptr addrspace(3) %address.local494, i64 128, !dbg !47
  %346 = load <4 x half>, ptr addrspace(3) %lds.const.gep1442, align 8, !dbg !47
  %347 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %345, <4 x float> %344) #9, !dbg !47
  %348 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %346, <4 x float> zeroinitializer) #9, !dbg !47
  %lds.const.gep1443 = getelementptr i8, ptr addrspace(3) %address.local494, i64 4224, !dbg !47
  %349 = load <4 x half>, ptr addrspace(3) %lds.const.gep1443, align 8, !dbg !47
  %lds.const.gep1444 = getelementptr i8, ptr addrspace(3) %address.local523, i64 128, !dbg !47
  %350 = load <4 x half>, ptr addrspace(3) %lds.const.gep1444, align 8, !dbg !47
  %351 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %349, <4 x float> %348) #9, !dbg !47
  %352 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %350, <4 x float> zeroinitializer) #9, !dbg !47
  %lds.const.gep1445 = getelementptr i8, ptr addrspace(3) %address.local523, i64 4224, !dbg !47
  %353 = load <4 x half>, ptr addrspace(3) %lds.const.gep1445, align 8, !dbg !47
  %lds.const.gep1446 = getelementptr i8, ptr addrspace(3) %address.local494, i64 1024, !dbg !47
  %354 = load <4 x half>, ptr addrspace(3) %lds.const.gep1446, align 8, !dbg !47
  %355 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %353, <4 x float> %352) #9, !dbg !47
  %356 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %354, <4 x float> zeroinitializer) #9, !dbg !47
  %lds.const.gep1447 = getelementptr i8, ptr addrspace(3) %address.local494, i64 5120, !dbg !47
  %357 = load <4 x half>, ptr addrspace(3) %lds.const.gep1447, align 8, !dbg !47
  %lds.const.gep1448 = getelementptr i8, ptr addrspace(3) %address.local523, i64 1024, !dbg !47
  %358 = load <4 x half>, ptr addrspace(3) %lds.const.gep1448, align 8, !dbg !47
  %359 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %357, <4 x float> %356) #9, !dbg !47
  %360 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %358, <4 x float> zeroinitializer) #9, !dbg !47
  %lds.const.gep1449 = getelementptr i8, ptr addrspace(3) %address.local523, i64 5120, !dbg !47
  %361 = load <4 x half>, ptr addrspace(3) %lds.const.gep1449, align 8, !dbg !47
  %lds.const.gep1450 = getelementptr i8, ptr addrspace(3) %address.local494, i64 1152, !dbg !47
  %362 = load <4 x half>, ptr addrspace(3) %lds.const.gep1450, align 8, !dbg !47
  %363 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %361, <4 x float> %360) #9, !dbg !47
  %364 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %362, <4 x float> zeroinitializer) #9, !dbg !47
  %lds.const.gep1451 = getelementptr i8, ptr addrspace(3) %address.local494, i64 5248, !dbg !47
  %365 = load <4 x half>, ptr addrspace(3) %lds.const.gep1451, align 8, !dbg !47
  %lds.const.gep1452 = getelementptr i8, ptr addrspace(3) %address.local523, i64 1152, !dbg !47
  %366 = load <4 x half>, ptr addrspace(3) %lds.const.gep1452, align 8, !dbg !47
  %367 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %365, <4 x float> %364) #9, !dbg !47
  %368 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %366, <4 x float> zeroinitializer) #9, !dbg !47
  %lds.const.gep1453 = getelementptr i8, ptr addrspace(3) %address.local523, i64 5248, !dbg !47
  %369 = load <4 x half>, ptr addrspace(3) %lds.const.gep1453, align 8, !dbg !47
  %370 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %369, <4 x float> %368) #9, !dbg !47
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !46
  fence syncscope("workgroup") release, !dbg !46
  call void @llvm.amdgcn.s.barrier(), !dbg !46
  fence syncscope("workgroup") acquire, !dbg !46
  %371 = add nuw i64 %205, 2048, !dbg !46
  %372 = icmp slt i64 %371, 0, !dbg !46
  %373 = sub nuw i64 -2049, %205, !dbg !46
  %374 = select i1 %372, i64 %373, i64 %371, !dbg !46
  %375 = sdiv i64 %374, 4096, !dbg !46
  %.lobit38 = ashr i64 %371, 63, !dbg !46
  %376 = xor i64 %375, %.lobit38, !dbg !46
  %377 = add nsw i64 %376, %16, !dbg !46
  %378 = add nsw i64 %262, 64, !dbg !46
  %379 = icmp slt i64 %262, -64, !dbg !46
  %380 = sub nsw i64 -65, %262, !dbg !46
  %381 = select i1 %379, i64 %380, i64 %378, !dbg !46
  %382 = sdiv i64 %381, 128, !dbg !46
  %383 = sext i1 %379 to i64, !dbg !46
  %384 = xor i64 %382, %383, !dbg !46
  %385 = shl nsw i64 %384, 7, !dbg !46
  %386 = sub i64 %195, %385, !dbg !46
  %387 = shl i64 %377, 19, !dbg !46
  %388 = and i64 %387, 16252928, !dbg !46
  %389 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %388, !dbg !46
  %390 = getelementptr [2 x i8], ptr addrspace(1) %389, i64 %217, !dbg !46
  %391 = getelementptr [2 x i8], ptr addrspace(1) %390, i64 %386, !dbg !46
  %392 = getelementptr i8, ptr addrspace(1) %391, i64 128, !dbg !46
  %393 = getelementptr [2 x i8], ptr addrspace(1) %392, i64 %223, !dbg !46
  %394 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local439), !dbg !46
  %395 = zext i32 %394 to i64, !dbg !46
  %396 = inttoptr i64 %395 to ptr addrspace(3), !dbg !46
  %address.local832 = getelementptr [2 x i8], ptr addrspace(1) %393, i64 %address.local449, !dbg !46
  %397 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !46
  %398 = ptrtoint ptr addrspace(1) %address.local832 to i64, !dbg !46
  %399 = sub i64 %398, %397, !dbg !46
  %dma.buffer.offset1576 = trunc i64 %399 to i32, !dbg !46
  %400 = trunc i64 %397 to i32, !dbg !46
  %401 = insertelement <4 x i32> poison, i32 %400, i64 0, !dbg !46
  %402 = lshr i64 %397, 32, !dbg !46
  %403 = trunc i64 %402 to i32, !dbg !46
  %404 = insertelement <4 x i32> %401, i32 %403, i64 1, !dbg !46
  %405 = insertelement <4 x i32> %404, i32 33554432, i64 2, !dbg !46
  %dma.buffer.rsrc1577 = insertelement <4 x i32> %405, i32 131072, i64 3, !dbg !46
  %406 = ptrtoint ptr addrspace(3) %396 to i32, !dbg !46
  %dma.buffer.lds1578 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %406), !dbg !46
  %407 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1578, i32 %dma.buffer.offset1576, <4 x i32> %dma.buffer.rsrc1577) #10, !dbg !46
  %408 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local456), !dbg !46
  %409 = zext i32 %408 to i64, !dbg !46
  %410 = inttoptr i64 %409 to ptr addrspace(3), !dbg !46
  %address.local833 = getelementptr [2 x i8], ptr addrspace(1) %393, i64 %address.local466, !dbg !46
  %411 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !46
  %412 = ptrtoint ptr addrspace(1) %address.local833 to i64, !dbg !46
  %413 = sub i64 %412, %411, !dbg !46
  %dma.buffer.offset1573 = trunc i64 %413 to i32, !dbg !46
  %414 = trunc i64 %411 to i32, !dbg !46
  %415 = insertelement <4 x i32> poison, i32 %414, i64 0, !dbg !46
  %416 = lshr i64 %411, 32, !dbg !46
  %417 = trunc i64 %416 to i32, !dbg !46
  %418 = insertelement <4 x i32> %415, i32 %417, i64 1, !dbg !46
  %419 = insertelement <4 x i32> %418, i32 33554432, i64 2, !dbg !46
  %dma.buffer.rsrc1574 = insertelement <4 x i32> %419, i32 131072, i64 3, !dbg !46
  %420 = ptrtoint ptr addrspace(3) %410 to i32, !dbg !46
  %dma.buffer.lds1575 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %420), !dbg !46
  %421 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1575, i32 %dma.buffer.offset1573, <4 x i32> %dma.buffer.rsrc1574) #10, !dbg !46
  %lds.const.gep1454 = getelementptr i8, ptr addrspace(3) %address.local494, i64 8192, !dbg !47
  %422 = load <4 x half>, ptr addrspace(3) %lds.const.gep1454, align 8, !dbg !47
  %423 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %422, <4 x float> %311) #9, !dbg !47
  %lds.const.gep1455 = getelementptr i8, ptr addrspace(3) %address.local494, i64 12288, !dbg !47
  %424 = load <4 x half>, ptr addrspace(3) %lds.const.gep1455, align 8, !dbg !47
  %lds.const.gep1456 = getelementptr i8, ptr addrspace(3) %address.local523, i64 8192, !dbg !47
  %425 = load <4 x half>, ptr addrspace(3) %lds.const.gep1456, align 8, !dbg !47
  %426 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %424, <4 x float> %423) #9, !dbg !47
  %427 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %425, <4 x float> %315) #9, !dbg !47
  %lds.const.gep1457 = getelementptr i8, ptr addrspace(3) %address.local523, i64 12288, !dbg !47
  %428 = load <4 x half>, ptr addrspace(3) %lds.const.gep1457, align 8, !dbg !47
  %lds.const.gep1458 = getelementptr i8, ptr addrspace(3) %address.local494, i64 8320, !dbg !47
  %429 = load <4 x half>, ptr addrspace(3) %lds.const.gep1458, align 8, !dbg !47
  %430 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %428, <4 x float> %427) #9, !dbg !47
  %431 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %429, <4 x float> %319) #9, !dbg !47
  %lds.const.gep1459 = getelementptr i8, ptr addrspace(3) %address.local494, i64 12416, !dbg !47
  %432 = load <4 x half>, ptr addrspace(3) %lds.const.gep1459, align 8, !dbg !47
  %lds.const.gep1460 = getelementptr i8, ptr addrspace(3) %address.local523, i64 8320, !dbg !47
  %433 = load <4 x half>, ptr addrspace(3) %lds.const.gep1460, align 8, !dbg !47
  %434 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %432, <4 x float> %431) #9, !dbg !47
  %435 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %433, <4 x float> %323) #9, !dbg !47
  %lds.const.gep1461 = getelementptr i8, ptr addrspace(3) %address.local523, i64 12416, !dbg !47
  %436 = load <4 x half>, ptr addrspace(3) %lds.const.gep1461, align 8, !dbg !47
  %lds.const.gep1462 = getelementptr i8, ptr addrspace(3) %address.local494, i64 9216, !dbg !47
  %437 = load <4 x half>, ptr addrspace(3) %lds.const.gep1462, align 8, !dbg !47
  %438 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %436, <4 x float> %435) #9, !dbg !47
  %439 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %437, <4 x float> %327) #9, !dbg !47
  %lds.const.gep1463 = getelementptr i8, ptr addrspace(3) %address.local494, i64 13312, !dbg !47
  %440 = load <4 x half>, ptr addrspace(3) %lds.const.gep1463, align 8, !dbg !47
  %lds.const.gep1464 = getelementptr i8, ptr addrspace(3) %address.local523, i64 9216, !dbg !47
  %441 = load <4 x half>, ptr addrspace(3) %lds.const.gep1464, align 8, !dbg !47
  %442 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %440, <4 x float> %439) #9, !dbg !47
  %443 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %441, <4 x float> %331) #9, !dbg !47
  %lds.const.gep1465 = getelementptr i8, ptr addrspace(3) %address.local523, i64 13312, !dbg !47
  %444 = load <4 x half>, ptr addrspace(3) %lds.const.gep1465, align 8, !dbg !47
  %lds.const.gep1466 = getelementptr i8, ptr addrspace(3) %address.local494, i64 9344, !dbg !47
  %445 = load <4 x half>, ptr addrspace(3) %lds.const.gep1466, align 8, !dbg !47
  %446 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %444, <4 x float> %443) #9, !dbg !47
  %447 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %445, <4 x float> %335) #9, !dbg !47
  %lds.const.gep1467 = getelementptr i8, ptr addrspace(3) %address.local494, i64 13440, !dbg !47
  %448 = load <4 x half>, ptr addrspace(3) %lds.const.gep1467, align 8, !dbg !47
  %lds.const.gep1468 = getelementptr i8, ptr addrspace(3) %address.local523, i64 9344, !dbg !47
  %449 = load <4 x half>, ptr addrspace(3) %lds.const.gep1468, align 8, !dbg !47
  %450 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %448, <4 x float> %447) #9, !dbg !47
  %451 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %449, <4 x float> %339) #9, !dbg !47
  %lds.const.gep1469 = getelementptr i8, ptr addrspace(3) %address.local523, i64 13440, !dbg !47
  %452 = load <4 x half>, ptr addrspace(3) %lds.const.gep1469, align 8, !dbg !47
  %lds.const.gep1470 = getelementptr i8, ptr addrspace(3) %address.local494, i64 8192, !dbg !47
  %453 = load <4 x half>, ptr addrspace(3) %lds.const.gep1470, align 8, !dbg !47
  %454 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %139, <4 x half> %452, <4 x float> %451) #9, !dbg !47
  %455 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %453, <4 x float> %343) #9, !dbg !47
  %lds.const.gep1471 = getelementptr i8, ptr addrspace(3) %address.local494, i64 12288, !dbg !47
  %456 = load <4 x half>, ptr addrspace(3) %lds.const.gep1471, align 8, !dbg !47
  %lds.const.gep1472 = getelementptr i8, ptr addrspace(3) %address.local523, i64 8192, !dbg !47
  %457 = load <4 x half>, ptr addrspace(3) %lds.const.gep1472, align 8, !dbg !47
  %458 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %456, <4 x float> %455) #9, !dbg !47
  %459 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %457, <4 x float> %347) #9, !dbg !47
  %lds.const.gep1473 = getelementptr i8, ptr addrspace(3) %address.local523, i64 12288, !dbg !47
  %460 = load <4 x half>, ptr addrspace(3) %lds.const.gep1473, align 8, !dbg !47
  %lds.const.gep1474 = getelementptr i8, ptr addrspace(3) %address.local494, i64 8320, !dbg !47
  %461 = load <4 x half>, ptr addrspace(3) %lds.const.gep1474, align 8, !dbg !47
  %462 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %460, <4 x float> %459) #9, !dbg !47
  %463 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %461, <4 x float> %351) #9, !dbg !47
  %lds.const.gep1475 = getelementptr i8, ptr addrspace(3) %address.local494, i64 12416, !dbg !47
  %464 = load <4 x half>, ptr addrspace(3) %lds.const.gep1475, align 8, !dbg !47
  %lds.const.gep1476 = getelementptr i8, ptr addrspace(3) %address.local523, i64 8320, !dbg !47
  %465 = load <4 x half>, ptr addrspace(3) %lds.const.gep1476, align 8, !dbg !47
  %466 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %464, <4 x float> %463) #9, !dbg !47
  %467 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %465, <4 x float> %355) #9, !dbg !47
  %lds.const.gep1477 = getelementptr i8, ptr addrspace(3) %address.local523, i64 12416, !dbg !47
  %468 = load <4 x half>, ptr addrspace(3) %lds.const.gep1477, align 8, !dbg !47
  %lds.const.gep1478 = getelementptr i8, ptr addrspace(3) %address.local494, i64 9216, !dbg !47
  %469 = load <4 x half>, ptr addrspace(3) %lds.const.gep1478, align 8, !dbg !47
  %470 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %468, <4 x float> %467) #9, !dbg !47
  %471 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %469, <4 x float> %359) #9, !dbg !47
  %lds.const.gep1479 = getelementptr i8, ptr addrspace(3) %address.local494, i64 13312, !dbg !47
  %472 = load <4 x half>, ptr addrspace(3) %lds.const.gep1479, align 8, !dbg !47
  %lds.const.gep1480 = getelementptr i8, ptr addrspace(3) %address.local523, i64 9216, !dbg !47
  %473 = load <4 x half>, ptr addrspace(3) %lds.const.gep1480, align 8, !dbg !47
  %474 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %472, <4 x float> %471) #9, !dbg !47
  %475 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %473, <4 x float> %363) #9, !dbg !47
  %lds.const.gep1481 = getelementptr i8, ptr addrspace(3) %address.local523, i64 13312, !dbg !47
  %476 = load <4 x half>, ptr addrspace(3) %lds.const.gep1481, align 8, !dbg !47
  %lds.const.gep1482 = getelementptr i8, ptr addrspace(3) %address.local494, i64 9344, !dbg !47
  %477 = load <4 x half>, ptr addrspace(3) %lds.const.gep1482, align 8, !dbg !47
  %478 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %476, <4 x float> %475) #9, !dbg !47
  %479 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %477, <4 x float> %367) #9, !dbg !47
  %lds.const.gep1483 = getelementptr i8, ptr addrspace(3) %address.local494, i64 13440, !dbg !47
  %480 = load <4 x half>, ptr addrspace(3) %lds.const.gep1483, align 8, !dbg !47
  %lds.const.gep1484 = getelementptr i8, ptr addrspace(3) %address.local523, i64 9344, !dbg !47
  %481 = load <4 x half>, ptr addrspace(3) %lds.const.gep1484, align 8, !dbg !47
  %482 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %480, <4 x float> %479) #9, !dbg !47
  %483 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %140, <4 x half> %481, <4 x float> %370) #9, !dbg !47
  %lds.const.gep1485 = getelementptr i8, ptr addrspace(3) %address.local523, i64 13440, !dbg !47
  %484 = load <4 x half>, ptr addrspace(3) %lds.const.gep1485, align 8, !dbg !47
  %485 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %484, <4 x float> %483) #9, !dbg !47
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !46
  fence syncscope("workgroup") release, !dbg !46
  call void @llvm.amdgcn.s.barrier(), !dbg !46
  fence syncscope("workgroup") acquire, !dbg !46
  %486 = add nuw i64 %205, 3072, !dbg !46
  %487 = icmp slt i64 %486, 0, !dbg !46
  %488 = sub nuw i64 -3073, %205, !dbg !46
  %489 = select i1 %487, i64 %488, i64 %486, !dbg !46
  %490 = sdiv i64 %489, 4096, !dbg !46
  %.lobit39 = ashr i64 %486, 63, !dbg !46
  %491 = xor i64 %490, %.lobit39, !dbg !46
  %492 = add nsw i64 %491, %16, !dbg !46
  %493 = add nsw i64 %262, 96, !dbg !46
  %494 = icmp slt i64 %262, -96, !dbg !46
  %495 = sub nsw i64 -97, %262, !dbg !46
  %496 = select i1 %494, i64 %495, i64 %493, !dbg !46
  %497 = sdiv i64 %496, 128, !dbg !46
  %498 = sext i1 %494 to i64, !dbg !46
  %499 = xor i64 %497, %498, !dbg !46
  %500 = shl nsw i64 %499, 7, !dbg !46
  %501 = sub i64 %195, %500, !dbg !46
  %502 = shl i64 %492, 19, !dbg !46
  %503 = and i64 %502, 16252928, !dbg !46
  %504 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %503, !dbg !46
  %505 = getelementptr [2 x i8], ptr addrspace(1) %504, i64 %217, !dbg !46
  %506 = getelementptr [2 x i8], ptr addrspace(1) %505, i64 %501, !dbg !46
  %507 = getelementptr i8, ptr addrspace(1) %506, i64 192, !dbg !46
  %508 = getelementptr [2 x i8], ptr addrspace(1) %507, i64 %223, !dbg !46
  %509 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local470), !dbg !46
  %510 = zext i32 %509 to i64, !dbg !46
  %511 = inttoptr i64 %510 to ptr addrspace(3), !dbg !46
  %address.local850 = getelementptr [2 x i8], ptr addrspace(1) %508, i64 %address.local449, !dbg !46
  %512 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !46
  %513 = ptrtoint ptr addrspace(1) %address.local850 to i64, !dbg !46
  %514 = sub i64 %513, %512, !dbg !46
  %dma.buffer.offset1570 = trunc i64 %514 to i32, !dbg !46
  %515 = trunc i64 %512 to i32, !dbg !46
  %516 = insertelement <4 x i32> poison, i32 %515, i64 0, !dbg !46
  %517 = lshr i64 %512, 32, !dbg !46
  %518 = trunc i64 %517 to i32, !dbg !46
  %519 = insertelement <4 x i32> %516, i32 %518, i64 1, !dbg !46
  %520 = insertelement <4 x i32> %519, i32 33554432, i64 2, !dbg !46
  %dma.buffer.rsrc1571 = insertelement <4 x i32> %520, i32 131072, i64 3, !dbg !46
  %521 = ptrtoint ptr addrspace(3) %511 to i32, !dbg !46
  %dma.buffer.lds1572 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %521), !dbg !46
  %522 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1572, i32 %dma.buffer.offset1570, <4 x i32> %dma.buffer.rsrc1571) #10, !dbg !46
  %523 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local474), !dbg !46
  %524 = zext i32 %523 to i64, !dbg !46
  %525 = inttoptr i64 %524 to ptr addrspace(3), !dbg !46
  %address.local851 = getelementptr [2 x i8], ptr addrspace(1) %508, i64 %address.local466, !dbg !46
  %526 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !46
  %527 = ptrtoint ptr addrspace(1) %address.local851 to i64, !dbg !46
  %528 = sub i64 %527, %526, !dbg !46
  %dma.buffer.offset1567 = trunc i64 %528 to i32, !dbg !46
  %529 = trunc i64 %526 to i32, !dbg !46
  %530 = insertelement <4 x i32> poison, i32 %529, i64 0, !dbg !46
  %531 = lshr i64 %526, 32, !dbg !46
  %532 = trunc i64 %531 to i32, !dbg !46
  %533 = insertelement <4 x i32> %530, i32 %532, i64 1, !dbg !46
  %534 = insertelement <4 x i32> %533, i32 33554432, i64 2, !dbg !46
  %dma.buffer.rsrc1568 = insertelement <4 x i32> %534, i32 131072, i64 3, !dbg !46
  %535 = ptrtoint ptr addrspace(3) %525 to i32, !dbg !46
  %dma.buffer.lds1569 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %535), !dbg !46
  %536 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1569, i32 %dma.buffer.offset1567, <4 x i32> %dma.buffer.rsrc1568) #10, !dbg !46
  %537 = load <4 x half>, ptr addrspace(3) %address.local494, align 8, !dbg !47
  %538 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %175, <4 x half> %537, <4 x float> %426) #9, !dbg !47
  %lds.const.gep1486 = getelementptr i8, ptr addrspace(3) %address.local494, i64 4096, !dbg !47
  %539 = load <4 x half>, ptr addrspace(3) %lds.const.gep1486, align 8, !dbg !47
  %540 = load <4 x half>, ptr addrspace(3) %address.local523, align 8, !dbg !47
  %541 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %539, <4 x float> %538) #9, !dbg !47
  %542 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %175, <4 x half> %540, <4 x float> %430) #9, !dbg !47
  %lds.const.gep1487 = getelementptr i8, ptr addrspace(3) %address.local523, i64 4096, !dbg !47
  %543 = load <4 x half>, ptr addrspace(3) %lds.const.gep1487, align 8, !dbg !47
  %lds.const.gep1488 = getelementptr i8, ptr addrspace(3) %address.local494, i64 128, !dbg !47
  %544 = load <4 x half>, ptr addrspace(3) %lds.const.gep1488, align 8, !dbg !47
  %545 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %543, <4 x float> %542) #9, !dbg !47
  %546 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %175, <4 x half> %544, <4 x float> %434) #9, !dbg !47
  %lds.const.gep1489 = getelementptr i8, ptr addrspace(3) %address.local494, i64 4224, !dbg !47
  %547 = load <4 x half>, ptr addrspace(3) %lds.const.gep1489, align 8, !dbg !47
  %lds.const.gep1490 = getelementptr i8, ptr addrspace(3) %address.local523, i64 128, !dbg !47
  %548 = load <4 x half>, ptr addrspace(3) %lds.const.gep1490, align 8, !dbg !47
  %549 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %547, <4 x float> %546) #9, !dbg !47
  %550 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %175, <4 x half> %548, <4 x float> %438) #9, !dbg !47
  %lds.const.gep1491 = getelementptr i8, ptr addrspace(3) %address.local523, i64 4224, !dbg !47
  %551 = load <4 x half>, ptr addrspace(3) %lds.const.gep1491, align 8, !dbg !47
  %lds.const.gep1492 = getelementptr i8, ptr addrspace(3) %address.local494, i64 1024, !dbg !47
  %552 = load <4 x half>, ptr addrspace(3) %lds.const.gep1492, align 8, !dbg !47
  %553 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %551, <4 x float> %550) #9, !dbg !47
  %554 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %175, <4 x half> %552, <4 x float> %442) #9, !dbg !47
  %lds.const.gep1493 = getelementptr i8, ptr addrspace(3) %address.local494, i64 5120, !dbg !47
  %555 = load <4 x half>, ptr addrspace(3) %lds.const.gep1493, align 8, !dbg !47
  %lds.const.gep1494 = getelementptr i8, ptr addrspace(3) %address.local523, i64 1024, !dbg !47
  %556 = load <4 x half>, ptr addrspace(3) %lds.const.gep1494, align 8, !dbg !47
  %557 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %555, <4 x float> %554) #9, !dbg !47
  %558 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %175, <4 x half> %556, <4 x float> %446) #9, !dbg !47
  %lds.const.gep1495 = getelementptr i8, ptr addrspace(3) %address.local523, i64 5120, !dbg !47
  %559 = load <4 x half>, ptr addrspace(3) %lds.const.gep1495, align 8, !dbg !47
  %lds.const.gep1496 = getelementptr i8, ptr addrspace(3) %address.local494, i64 1152, !dbg !47
  %560 = load <4 x half>, ptr addrspace(3) %lds.const.gep1496, align 8, !dbg !47
  %561 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %559, <4 x float> %558) #9, !dbg !47
  %562 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %175, <4 x half> %560, <4 x float> %450) #9, !dbg !47
  %lds.const.gep1497 = getelementptr i8, ptr addrspace(3) %address.local494, i64 5248, !dbg !47
  %563 = load <4 x half>, ptr addrspace(3) %lds.const.gep1497, align 8, !dbg !47
  %lds.const.gep1498 = getelementptr i8, ptr addrspace(3) %address.local523, i64 1152, !dbg !47
  %564 = load <4 x half>, ptr addrspace(3) %lds.const.gep1498, align 8, !dbg !47
  %565 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %563, <4 x float> %562) #9, !dbg !47
  %566 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %175, <4 x half> %564, <4 x float> %454) #9, !dbg !47
  %lds.const.gep1499 = getelementptr i8, ptr addrspace(3) %address.local523, i64 5248, !dbg !47
  %567 = load <4 x half>, ptr addrspace(3) %lds.const.gep1499, align 8, !dbg !47
  %568 = load <4 x half>, ptr addrspace(3) %address.local494, align 8, !dbg !47
  %569 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %177, <4 x half> %567, <4 x float> %566) #9, !dbg !47
  %570 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %568, <4 x float> %458) #9, !dbg !47
  %lds.const.gep1500 = getelementptr i8, ptr addrspace(3) %address.local494, i64 4096, !dbg !47
  %571 = load <4 x half>, ptr addrspace(3) %lds.const.gep1500, align 8, !dbg !47
  %572 = load <4 x half>, ptr addrspace(3) %address.local523, align 8, !dbg !47
  %573 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %179, <4 x half> %571, <4 x float> %570) #9, !dbg !47
  %574 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %572, <4 x float> %462) #9, !dbg !47
  %lds.const.gep1501 = getelementptr i8, ptr addrspace(3) %address.local523, i64 4096, !dbg !47
  %575 = load <4 x half>, ptr addrspace(3) %lds.const.gep1501, align 8, !dbg !47
  %lds.const.gep1502 = getelementptr i8, ptr addrspace(3) %address.local494, i64 128, !dbg !47
  %576 = load <4 x half>, ptr addrspace(3) %lds.const.gep1502, align 8, !dbg !47
  %577 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %179, <4 x half> %575, <4 x float> %574) #9, !dbg !47
  %578 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %576, <4 x float> %466) #9, !dbg !47
  %lds.const.gep1503 = getelementptr i8, ptr addrspace(3) %address.local494, i64 4224, !dbg !47
  %579 = load <4 x half>, ptr addrspace(3) %lds.const.gep1503, align 8, !dbg !47
  %lds.const.gep1504 = getelementptr i8, ptr addrspace(3) %address.local523, i64 128, !dbg !47
  %580 = load <4 x half>, ptr addrspace(3) %lds.const.gep1504, align 8, !dbg !47
  %581 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %179, <4 x half> %579, <4 x float> %578) #9, !dbg !47
  %582 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %580, <4 x float> %470) #9, !dbg !47
  %lds.const.gep1505 = getelementptr i8, ptr addrspace(3) %address.local523, i64 4224, !dbg !47
  %583 = load <4 x half>, ptr addrspace(3) %lds.const.gep1505, align 8, !dbg !47
  %lds.const.gep1506 = getelementptr i8, ptr addrspace(3) %address.local494, i64 1024, !dbg !47
  %584 = load <4 x half>, ptr addrspace(3) %lds.const.gep1506, align 8, !dbg !47
  %585 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %179, <4 x half> %583, <4 x float> %582) #9, !dbg !47
  %586 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %584, <4 x float> %474) #9, !dbg !47
  %lds.const.gep1507 = getelementptr i8, ptr addrspace(3) %address.local494, i64 5120, !dbg !47
  %587 = load <4 x half>, ptr addrspace(3) %lds.const.gep1507, align 8, !dbg !47
  %lds.const.gep1508 = getelementptr i8, ptr addrspace(3) %address.local523, i64 1024, !dbg !47
  %588 = load <4 x half>, ptr addrspace(3) %lds.const.gep1508, align 8, !dbg !47
  %589 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %179, <4 x half> %587, <4 x float> %586) #9, !dbg !47
  %590 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %588, <4 x float> %478) #9, !dbg !47
  %lds.const.gep1509 = getelementptr i8, ptr addrspace(3) %address.local523, i64 5120, !dbg !47
  %591 = load <4 x half>, ptr addrspace(3) %lds.const.gep1509, align 8, !dbg !47
  %lds.const.gep1510 = getelementptr i8, ptr addrspace(3) %address.local494, i64 1152, !dbg !47
  %592 = load <4 x half>, ptr addrspace(3) %lds.const.gep1510, align 8, !dbg !47
  %593 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %179, <4 x half> %591, <4 x float> %590) #9, !dbg !47
  %594 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %592, <4 x float> %482) #9, !dbg !47
  %lds.const.gep1511 = getelementptr i8, ptr addrspace(3) %address.local494, i64 5248, !dbg !47
  %595 = load <4 x half>, ptr addrspace(3) %lds.const.gep1511, align 8, !dbg !47
  %lds.const.gep1512 = getelementptr i8, ptr addrspace(3) %address.local523, i64 1152, !dbg !47
  %596 = load <4 x half>, ptr addrspace(3) %lds.const.gep1512, align 8, !dbg !47
  %597 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %179, <4 x half> %595, <4 x float> %594) #9, !dbg !47
  %598 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %178, <4 x half> %596, <4 x float> %485) #9, !dbg !47
  %lds.const.gep1513 = getelementptr i8, ptr addrspace(3) %address.local523, i64 5248, !dbg !47
  %599 = load <4 x half>, ptr addrspace(3) %lds.const.gep1513, align 8, !dbg !47
  %600 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %179, <4 x half> %599, <4 x float> %598) #9, !dbg !47
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !46
  fence syncscope("workgroup") release, !dbg !46
  call void @llvm.amdgcn.s.barrier(), !dbg !46
  fence syncscope("workgroup") acquire, !dbg !46
  %601 = add nuw nsw i64 %217, %16, !dbg !48
  %602 = shl i64 %601, 19, !dbg !48
  %603 = and i64 %602, 16252928, !dbg !48
  %604 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %603, !dbg !48
  %605 = getelementptr [2 x i8], ptr addrspace(1) %604, i64 %223, !dbg !48
  %606 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local439), !dbg !48
  %607 = zext i32 %606 to i64, !dbg !48
  %608 = inttoptr i64 %607 to ptr addrspace(3), !dbg !48
  %address.local852 = shl nuw nsw i64 %address.local434, 5, !dbg !48
  %address.local853 = and i64 %address.local852, 8064, !dbg !48
  %address.local854 = shl nuw nsw i64 %address.local434, 3, !dbg !48
  %address.local855 = and i64 %address.local854, 24, !dbg !48
  %address.local856 = or disjoint i64 %address.local853, %address.local855, !dbg !48
  %address.local857 = getelementptr [2 x i8], ptr addrspace(1) %605, i64 %address.local856, !dbg !48
  %609 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !48
  %610 = ptrtoint ptr addrspace(1) %address.local857 to i64, !dbg !48
  %611 = sub i64 %610, %609, !dbg !48
  %dma.buffer.offset1564 = trunc i64 %611 to i32, !dbg !48
  %612 = trunc i64 %609 to i32, !dbg !48
  %613 = insertelement <4 x i32> poison, i32 %612, i64 0, !dbg !48
  %614 = lshr i64 %609, 32, !dbg !48
  %615 = trunc i64 %614 to i32, !dbg !48
  %616 = insertelement <4 x i32> %613, i32 %615, i64 1, !dbg !48
  %617 = insertelement <4 x i32> %616, i32 33554432, i64 2, !dbg !48
  %dma.buffer.rsrc1565 = insertelement <4 x i32> %617, i32 131072, i64 3, !dbg !48
  %618 = ptrtoint ptr addrspace(3) %608 to i32, !dbg !48
  %dma.buffer.lds1566 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %618), !dbg !48
  %619 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1566, i32 %dma.buffer.offset1564, <4 x i32> %dma.buffer.rsrc1565) #10, !dbg !48
  %620 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local456), !dbg !48
  %621 = zext i32 %620 to i64, !dbg !48
  %622 = inttoptr i64 %621 to ptr addrspace(3), !dbg !48
  %address.local858 = shl nuw nsw i64 %address.local451, 5, !dbg !48
  %address.local859 = and i64 %address.local858, 16256, !dbg !48
  %address.local860 = shl nuw nsw i64 %address.local434, 3, !dbg !48
  %address.local861 = and i64 %address.local860, 24, !dbg !48
  %address.local862 = or disjoint i64 %address.local859, %address.local861, !dbg !48
  %address.local863 = getelementptr [2 x i8], ptr addrspace(1) %605, i64 %address.local862, !dbg !48
  %623 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !48
  %624 = ptrtoint ptr addrspace(1) %address.local863 to i64, !dbg !48
  %625 = sub i64 %624, %623, !dbg !48
  %dma.buffer.offset1561 = trunc i64 %625 to i32, !dbg !48
  %626 = trunc i64 %623 to i32, !dbg !48
  %627 = insertelement <4 x i32> poison, i32 %626, i64 0, !dbg !48
  %628 = lshr i64 %623, 32, !dbg !48
  %629 = trunc i64 %628 to i32, !dbg !48
  %630 = insertelement <4 x i32> %627, i32 %629, i64 1, !dbg !48
  %631 = insertelement <4 x i32> %630, i32 33554432, i64 2, !dbg !48
  %dma.buffer.rsrc1562 = insertelement <4 x i32> %631, i32 131072, i64 3, !dbg !48
  %632 = ptrtoint ptr addrspace(3) %622 to i32, !dbg !48
  %dma.buffer.lds1563 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %632), !dbg !48
  %633 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1563, i32 %dma.buffer.offset1561, <4 x i32> %dma.buffer.rsrc1562) #10, !dbg !48
  %lds.const.gep1514 = getelementptr i8, ptr addrspace(3) %address.local494, i64 8192, !dbg !47
  %634 = load <4 x half>, ptr addrspace(3) %lds.const.gep1514, align 8, !dbg !47
  %635 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %634, <4 x float> %541) #9, !dbg !47
  %lds.const.gep1515 = getelementptr i8, ptr addrspace(3) %address.local494, i64 12288, !dbg !47
  %636 = load <4 x half>, ptr addrspace(3) %lds.const.gep1515, align 8, !dbg !47
  %lds.const.gep1516 = getelementptr i8, ptr addrspace(3) %address.local523, i64 8192, !dbg !47
  %637 = load <4 x half>, ptr addrspace(3) %lds.const.gep1516, align 8, !dbg !47
  %638 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %636, <4 x float> %635) #9, !dbg !47
  %639 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %637, <4 x float> %545) #9, !dbg !47
  %lds.const.gep1517 = getelementptr i8, ptr addrspace(3) %address.local523, i64 12288, !dbg !47
  %640 = load <4 x half>, ptr addrspace(3) %lds.const.gep1517, align 8, !dbg !47
  %lds.const.gep1518 = getelementptr i8, ptr addrspace(3) %address.local494, i64 8320, !dbg !47
  %641 = load <4 x half>, ptr addrspace(3) %lds.const.gep1518, align 8, !dbg !47
  %642 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %640, <4 x float> %639) #9, !dbg !47
  %643 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %641, <4 x float> %549) #9, !dbg !47
  %lds.const.gep1519 = getelementptr i8, ptr addrspace(3) %address.local494, i64 12416, !dbg !47
  %644 = load <4 x half>, ptr addrspace(3) %lds.const.gep1519, align 8, !dbg !47
  %lds.const.gep1520 = getelementptr i8, ptr addrspace(3) %address.local523, i64 8320, !dbg !47
  %645 = load <4 x half>, ptr addrspace(3) %lds.const.gep1520, align 8, !dbg !47
  %646 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %644, <4 x float> %643) #9, !dbg !47
  %647 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %645, <4 x float> %553) #9, !dbg !47
  %lds.const.gep1521 = getelementptr i8, ptr addrspace(3) %address.local523, i64 12416, !dbg !47
  %648 = load <4 x half>, ptr addrspace(3) %lds.const.gep1521, align 8, !dbg !47
  %lds.const.gep1522 = getelementptr i8, ptr addrspace(3) %address.local494, i64 9216, !dbg !47
  %649 = load <4 x half>, ptr addrspace(3) %lds.const.gep1522, align 8, !dbg !47
  %650 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %648, <4 x float> %647) #9, !dbg !47
  %651 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %649, <4 x float> %557) #9, !dbg !47
  %lds.const.gep1523 = getelementptr i8, ptr addrspace(3) %address.local494, i64 13312, !dbg !47
  %652 = load <4 x half>, ptr addrspace(3) %lds.const.gep1523, align 8, !dbg !47
  %lds.const.gep1524 = getelementptr i8, ptr addrspace(3) %address.local523, i64 9216, !dbg !47
  %653 = load <4 x half>, ptr addrspace(3) %lds.const.gep1524, align 8, !dbg !47
  %654 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %652, <4 x float> %651) #9, !dbg !47
  %655 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %653, <4 x float> %561) #9, !dbg !47
  %lds.const.gep1525 = getelementptr i8, ptr addrspace(3) %address.local523, i64 13312, !dbg !47
  %656 = load <4 x half>, ptr addrspace(3) %lds.const.gep1525, align 8, !dbg !47
  %lds.const.gep1526 = getelementptr i8, ptr addrspace(3) %address.local494, i64 9344, !dbg !47
  %657 = load <4 x half>, ptr addrspace(3) %lds.const.gep1526, align 8, !dbg !47
  %658 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %656, <4 x float> %655) #9, !dbg !47
  %659 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %657, <4 x float> %565) #9, !dbg !47
  %lds.const.gep1527 = getelementptr i8, ptr addrspace(3) %address.local494, i64 13440, !dbg !47
  %660 = load <4 x half>, ptr addrspace(3) %lds.const.gep1527, align 8, !dbg !47
  %lds.const.gep1528 = getelementptr i8, ptr addrspace(3) %address.local523, i64 9344, !dbg !47
  %661 = load <4 x half>, ptr addrspace(3) %lds.const.gep1528, align 8, !dbg !47
  %662 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %660, <4 x float> %659) #9, !dbg !47
  %663 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %183, <4 x half> %661, <4 x float> %569) #9, !dbg !47
  %lds.const.gep1529 = getelementptr i8, ptr addrspace(3) %address.local523, i64 13440, !dbg !47
  %664 = load <4 x half>, ptr addrspace(3) %lds.const.gep1529, align 8, !dbg !47
  %lds.const.gep1530 = getelementptr i8, ptr addrspace(3) %address.local494, i64 8192, !dbg !47
  %665 = load <4 x half>, ptr addrspace(3) %lds.const.gep1530, align 8, !dbg !47
  %666 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %185, <4 x half> %664, <4 x float> %663) #9, !dbg !47
  %667 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %665, <4 x float> %573) #9, !dbg !47
  %lds.const.gep1531 = getelementptr i8, ptr addrspace(3) %address.local494, i64 12288, !dbg !47
  %668 = load <4 x half>, ptr addrspace(3) %lds.const.gep1531, align 8, !dbg !47
  %lds.const.gep1532 = getelementptr i8, ptr addrspace(3) %address.local523, i64 8192, !dbg !47
  %669 = load <4 x half>, ptr addrspace(3) %lds.const.gep1532, align 8, !dbg !47
  %670 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %187, <4 x half> %668, <4 x float> %667) #9, !dbg !47
  %671 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %669, <4 x float> %577) #9, !dbg !47
  %lds.const.gep1533 = getelementptr i8, ptr addrspace(3) %address.local523, i64 12288, !dbg !47
  %672 = load <4 x half>, ptr addrspace(3) %lds.const.gep1533, align 8, !dbg !47
  %lds.const.gep1534 = getelementptr i8, ptr addrspace(3) %address.local494, i64 8320, !dbg !47
  %673 = load <4 x half>, ptr addrspace(3) %lds.const.gep1534, align 8, !dbg !47
  %674 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %187, <4 x half> %672, <4 x float> %671) #9, !dbg !47
  %675 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %673, <4 x float> %581) #9, !dbg !47
  %lds.const.gep1535 = getelementptr i8, ptr addrspace(3) %address.local494, i64 12416, !dbg !47
  %676 = load <4 x half>, ptr addrspace(3) %lds.const.gep1535, align 8, !dbg !47
  %lds.const.gep1536 = getelementptr i8, ptr addrspace(3) %address.local523, i64 8320, !dbg !47
  %677 = load <4 x half>, ptr addrspace(3) %lds.const.gep1536, align 8, !dbg !47
  %678 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %187, <4 x half> %676, <4 x float> %675) #9, !dbg !47
  %679 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %677, <4 x float> %585) #9, !dbg !47
  %lds.const.gep1537 = getelementptr i8, ptr addrspace(3) %address.local523, i64 12416, !dbg !47
  %680 = load <4 x half>, ptr addrspace(3) %lds.const.gep1537, align 8, !dbg !47
  %lds.const.gep1538 = getelementptr i8, ptr addrspace(3) %address.local494, i64 9216, !dbg !47
  %681 = load <4 x half>, ptr addrspace(3) %lds.const.gep1538, align 8, !dbg !47
  %682 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %187, <4 x half> %680, <4 x float> %679) #9, !dbg !47
  %683 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %681, <4 x float> %589) #9, !dbg !47
  %lds.const.gep1539 = getelementptr i8, ptr addrspace(3) %address.local494, i64 13312, !dbg !47
  %684 = load <4 x half>, ptr addrspace(3) %lds.const.gep1539, align 8, !dbg !47
  %lds.const.gep1540 = getelementptr i8, ptr addrspace(3) %address.local523, i64 9216, !dbg !47
  %685 = load <4 x half>, ptr addrspace(3) %lds.const.gep1540, align 8, !dbg !47
  %686 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %187, <4 x half> %684, <4 x float> %683) #9, !dbg !47
  %687 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %685, <4 x float> %593) #9, !dbg !47
  %lds.const.gep1541 = getelementptr i8, ptr addrspace(3) %address.local523, i64 13312, !dbg !47
  %688 = load <4 x half>, ptr addrspace(3) %lds.const.gep1541, align 8, !dbg !47
  %lds.const.gep1542 = getelementptr i8, ptr addrspace(3) %address.local494, i64 9344, !dbg !47
  %689 = load <4 x half>, ptr addrspace(3) %lds.const.gep1542, align 8, !dbg !47
  %690 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %187, <4 x half> %688, <4 x float> %687) #9, !dbg !47
  %691 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %689, <4 x float> %597) #9, !dbg !47
  %lds.const.gep1543 = getelementptr i8, ptr addrspace(3) %address.local494, i64 13440, !dbg !47
  %692 = load <4 x half>, ptr addrspace(3) %lds.const.gep1543, align 8, !dbg !47
  %lds.const.gep1544 = getelementptr i8, ptr addrspace(3) %address.local523, i64 9344, !dbg !47
  %693 = load <4 x half>, ptr addrspace(3) %lds.const.gep1544, align 8, !dbg !47
  %694 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %187, <4 x half> %692, <4 x float> %691) #9, !dbg !47
  %695 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %186, <4 x half> %693, <4 x float> %600) #9, !dbg !47
  %lds.const.gep1545 = getelementptr i8, ptr addrspace(3) %address.local523, i64 13440, !dbg !47
  %696 = load <4 x half>, ptr addrspace(3) %lds.const.gep1545, align 8, !dbg !47
  %697 = or disjoint i64 %214, 127, !dbg !43
  %698 = icmp uge i64 %64, %697, !dbg !43
  %699 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %187, <4 x half> %696, <4 x float> %695) #9, !dbg !47
  %700 = and i1 %190, %698, !dbg !43
  br i1 %700, label %701, label %702, !dbg !43

701:                                              ; preds = %213
  br label %846, !dbg !43

702:                                              ; preds = %213
  %703 = add i64 %64, %32, !dbg !43
  %704 = shl nuw nsw i64 %65, 4, !dbg !43
  %705 = add i64 %703, %704, !dbg !43
  %706 = and i32 %31, 240, !dbg !43
  %index.compact = zext nneg i32 %706 to i64, !dbg !43
  %707 = sub i64 %705, %index.compact, !dbg !43
  %708 = and i64 %32, 48, !dbg !43
  %709 = or disjoint i64 %214, %708, !dbg !43
  %710 = add i64 %707, 1, !dbg !52
  %711 = lshr i64 %710, 32, !dbg !53
  %compare.high = trunc i64 %711 to i32, !dbg !53
  %712 = lshr i64 %709, 32, !dbg !53
  %compare.high1157 = trunc i64 %712 to i32, !dbg !53
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high1157, !dbg !53
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high1157, !dbg !53
  %compare.low.base = trunc i64 %710 to i32, !dbg !53
  %compare.low.base1158 = trunc i64 %709 to i32, !dbg !53
  %compare.low = icmp ugt i32 %compare.low.base, %compare.low.base1158, !dbg !53
  %compare.result = select i1 %compare.high.equal, i1 %compare.low, i1 %compare.high.order, !dbg !53
  %713 = select i1 %compare.result, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 1) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local = or i32 %compare.local.base, 1, !dbg !53
  %compare.low1159 = icmp ugt i32 %compare.low.base, %compare.local, !dbg !53
  %compare.result1160 = select i1 %compare.high.equal, i1 %compare.low1159, i1 %compare.high.order, !dbg !53
  %714 = select i1 %compare.result1160, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1161 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 2) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1162 = or i32 %compare.local.base1161, 2, !dbg !53
  %compare.low1163 = icmp ugt i32 %compare.low.base, %compare.local1162, !dbg !53
  %compare.result1164 = select i1 %compare.high.equal, i1 %compare.low1163, i1 %compare.high.order, !dbg !53
  %715 = select i1 %compare.result1164, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1165 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 3) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1166 = or i32 %compare.local.base1165, 3, !dbg !53
  %compare.low1167 = icmp ugt i32 %compare.low.base, %compare.local1166, !dbg !53
  %compare.result1168 = select i1 %compare.high.equal, i1 %compare.low1167, i1 %compare.high.order, !dbg !53
  %716 = select i1 %compare.result1168, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1169 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 4) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1170 = or i32 %compare.local.base1169, 4, !dbg !53
  %compare.low1171 = icmp ugt i32 %compare.low.base, %compare.local1170, !dbg !53
  %compare.result1172 = select i1 %compare.high.equal, i1 %compare.low1171, i1 %compare.high.order, !dbg !53
  %717 = select i1 %compare.result1172, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1173 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 5) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1174 = or i32 %compare.local.base1173, 5, !dbg !53
  %compare.low1175 = icmp ugt i32 %compare.low.base, %compare.local1174, !dbg !53
  %compare.result1176 = select i1 %compare.high.equal, i1 %compare.low1175, i1 %compare.high.order, !dbg !53
  %718 = select i1 %compare.result1176, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1177 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 6) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1178 = or i32 %compare.local.base1177, 6, !dbg !53
  %compare.low1179 = icmp ugt i32 %compare.low.base, %compare.local1178, !dbg !53
  %compare.result1180 = select i1 %compare.high.equal, i1 %compare.low1179, i1 %compare.high.order, !dbg !53
  %719 = select i1 %compare.result1180, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1181 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 7) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1182 = or i32 %compare.local.base1181, 7, !dbg !53
  %compare.low1183 = icmp ugt i32 %compare.low.base, %compare.local1182, !dbg !53
  %compare.result1184 = select i1 %compare.high.equal, i1 %compare.low1183, i1 %compare.high.order, !dbg !53
  %720 = select i1 %compare.result1184, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1185 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 8) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1186 = or i32 %compare.local.base1185, 8, !dbg !53
  %compare.low1187 = icmp ugt i32 %compare.low.base, %compare.local1186, !dbg !53
  %compare.result1188 = select i1 %compare.high.equal, i1 %compare.low1187, i1 %compare.high.order, !dbg !53
  %721 = select i1 %compare.result1188, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1189 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 9) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1190 = or i32 %compare.local.base1189, 9, !dbg !53
  %compare.low1191 = icmp ugt i32 %compare.low.base, %compare.local1190, !dbg !53
  %compare.result1192 = select i1 %compare.high.equal, i1 %compare.low1191, i1 %compare.high.order, !dbg !53
  %722 = select i1 %compare.result1192, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1193 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 10) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1194 = or i32 %compare.local.base1193, 10, !dbg !53
  %compare.low1195 = icmp ugt i32 %compare.low.base, %compare.local1194, !dbg !53
  %compare.result1196 = select i1 %compare.high.equal, i1 %compare.low1195, i1 %compare.high.order, !dbg !53
  %723 = select i1 %compare.result1196, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1197 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 11) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1198 = or i32 %compare.local.base1197, 11, !dbg !53
  %compare.low1199 = icmp ugt i32 %compare.low.base, %compare.local1198, !dbg !53
  %compare.result1200 = select i1 %compare.high.equal, i1 %compare.low1199, i1 %compare.high.order, !dbg !53
  %724 = select i1 %compare.result1200, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1201 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 12) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1202 = or i32 %compare.local.base1201, 12, !dbg !53
  %compare.low1203 = icmp ugt i32 %compare.low.base, %compare.local1202, !dbg !53
  %compare.result1204 = select i1 %compare.high.equal, i1 %compare.low1203, i1 %compare.high.order, !dbg !53
  %725 = select i1 %compare.result1204, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1205 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 13) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1206 = or i32 %compare.local.base1205, 13, !dbg !53
  %compare.low1207 = icmp ugt i32 %compare.low.base, %compare.local1206, !dbg !53
  %compare.result1208 = select i1 %compare.high.equal, i1 %compare.low1207, i1 %compare.high.order, !dbg !53
  %726 = select i1 %compare.result1208, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1209 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 14) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1210 = or i32 %compare.local.base1209, 14, !dbg !53
  %compare.low1211 = icmp ugt i32 %compare.low.base, %compare.local1210, !dbg !53
  %compare.result1212 = select i1 %compare.high.equal, i1 %compare.low1211, i1 %compare.high.order, !dbg !53
  %727 = select i1 %compare.result1212, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1213 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 15) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1214 = or i32 %compare.local.base1213, 15, !dbg !53
  %compare.low1215 = icmp ugt i32 %compare.low.base, %compare.local1214, !dbg !53
  %compare.result1216 = select i1 %compare.high.equal, i1 %compare.low1215, i1 %compare.high.order, !dbg !53
  %728 = select i1 %compare.result1216, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1217 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 16) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1218 = or i32 %compare.local.base1217, 64, !dbg !53
  %compare.low1219 = icmp ugt i32 %compare.low.base, %compare.local1218, !dbg !53
  %compare.result1220 = select i1 %compare.high.equal, i1 %compare.low1219, i1 %compare.high.order, !dbg !53
  %729 = select i1 %compare.result1220, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1221 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 17) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1222 = or i32 %compare.local.base1221, 65, !dbg !53
  %compare.low1223 = icmp ugt i32 %compare.low.base, %compare.local1222, !dbg !53
  %compare.result1224 = select i1 %compare.high.equal, i1 %compare.low1223, i1 %compare.high.order, !dbg !53
  %730 = select i1 %compare.result1224, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1225 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 18) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1226 = or i32 %compare.local.base1225, 66, !dbg !53
  %compare.low1227 = icmp ugt i32 %compare.low.base, %compare.local1226, !dbg !53
  %compare.result1228 = select i1 %compare.high.equal, i1 %compare.low1227, i1 %compare.high.order, !dbg !53
  %731 = select i1 %compare.result1228, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1229 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 19) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1230 = or i32 %compare.local.base1229, 67, !dbg !53
  %compare.low1231 = icmp ugt i32 %compare.low.base, %compare.local1230, !dbg !53
  %compare.result1232 = select i1 %compare.high.equal, i1 %compare.low1231, i1 %compare.high.order, !dbg !53
  %732 = select i1 %compare.result1232, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1233 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 20) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1234 = or i32 %compare.local.base1233, 68, !dbg !53
  %compare.low1235 = icmp ugt i32 %compare.low.base, %compare.local1234, !dbg !53
  %compare.result1236 = select i1 %compare.high.equal, i1 %compare.low1235, i1 %compare.high.order, !dbg !53
  %733 = select i1 %compare.result1236, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1237 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 21) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1238 = or i32 %compare.local.base1237, 69, !dbg !53
  %compare.low1239 = icmp ugt i32 %compare.low.base, %compare.local1238, !dbg !53
  %compare.result1240 = select i1 %compare.high.equal, i1 %compare.low1239, i1 %compare.high.order, !dbg !53
  %734 = select i1 %compare.result1240, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1241 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 22) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1242 = or i32 %compare.local.base1241, 70, !dbg !53
  %compare.low1243 = icmp ugt i32 %compare.low.base, %compare.local1242, !dbg !53
  %compare.result1244 = select i1 %compare.high.equal, i1 %compare.low1243, i1 %compare.high.order, !dbg !53
  %735 = select i1 %compare.result1244, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1245 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 23) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1246 = or i32 %compare.local.base1245, 71, !dbg !53
  %compare.low1247 = icmp ugt i32 %compare.low.base, %compare.local1246, !dbg !53
  %compare.result1248 = select i1 %compare.high.equal, i1 %compare.low1247, i1 %compare.high.order, !dbg !53
  %736 = select i1 %compare.result1248, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1249 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 24) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1250 = or i32 %compare.local.base1249, 72, !dbg !53
  %compare.low1251 = icmp ugt i32 %compare.low.base, %compare.local1250, !dbg !53
  %compare.result1252 = select i1 %compare.high.equal, i1 %compare.low1251, i1 %compare.high.order, !dbg !53
  %737 = select i1 %compare.result1252, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1253 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 25) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1254 = or i32 %compare.local.base1253, 73, !dbg !53
  %compare.low1255 = icmp ugt i32 %compare.low.base, %compare.local1254, !dbg !53
  %compare.result1256 = select i1 %compare.high.equal, i1 %compare.low1255, i1 %compare.high.order, !dbg !53
  %738 = select i1 %compare.result1256, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1257 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 26) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1258 = or i32 %compare.local.base1257, 74, !dbg !53
  %compare.low1259 = icmp ugt i32 %compare.low.base, %compare.local1258, !dbg !53
  %compare.result1260 = select i1 %compare.high.equal, i1 %compare.low1259, i1 %compare.high.order, !dbg !53
  %739 = select i1 %compare.result1260, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1261 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 27) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1262 = or i32 %compare.local.base1261, 75, !dbg !53
  %compare.low1263 = icmp ugt i32 %compare.low.base, %compare.local1262, !dbg !53
  %compare.result1264 = select i1 %compare.high.equal, i1 %compare.low1263, i1 %compare.high.order, !dbg !53
  %740 = select i1 %compare.result1264, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1265 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 28) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1266 = or i32 %compare.local.base1265, 76, !dbg !53
  %compare.low1267 = icmp ugt i32 %compare.low.base, %compare.local1266, !dbg !53
  %compare.result1268 = select i1 %compare.high.equal, i1 %compare.low1267, i1 %compare.high.order, !dbg !53
  %741 = select i1 %compare.result1268, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1269 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 29) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1270 = or i32 %compare.local.base1269, 77, !dbg !53
  %compare.low1271 = icmp ugt i32 %compare.low.base, %compare.local1270, !dbg !53
  %compare.result1272 = select i1 %compare.high.equal, i1 %compare.low1271, i1 %compare.high.order, !dbg !53
  %742 = select i1 %compare.result1272, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1273 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 30) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1274 = or i32 %compare.local.base1273, 78, !dbg !53
  %compare.low1275 = icmp ugt i32 %compare.low.base, %compare.local1274, !dbg !53
  %compare.result1276 = select i1 %compare.high.equal, i1 %compare.low1275, i1 %compare.high.order, !dbg !53
  %743 = select i1 %compare.result1276, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1277 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1158, i32 31) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1278 = or i32 %compare.local.base1277, 79, !dbg !53
  %compare.low1279 = icmp ugt i32 %compare.low.base, %compare.local1278, !dbg !53
  %compare.result1280 = select i1 %compare.high.equal, i1 %compare.low1279, i1 %compare.high.order, !dbg !53
  %744 = select i1 %compare.result1280, float 0.000000e+00, float -inf, !dbg !54
  %745 = add i64 %707, 65, !dbg !52
  %746 = lshr i64 %745, 32, !dbg !53
  %compare.high1281 = trunc i64 %746 to i32, !dbg !53
  %747 = lshr i64 %709, 32, !dbg !53
  %compare.high1282 = trunc i64 %747 to i32, !dbg !53
  %compare.high.equal1283 = icmp eq i32 %compare.high1281, %compare.high1282, !dbg !53
  %compare.high.order1284 = icmp ugt i32 %compare.high1281, %compare.high1282, !dbg !53
  %compare.low.base1285 = trunc i64 %745 to i32, !dbg !53
  %compare.low.base1286 = trunc i64 %709 to i32, !dbg !53
  %compare.low1287 = icmp ugt i32 %compare.low.base1285, %compare.low.base1286, !dbg !53
  %compare.result1288 = select i1 %compare.high.equal1283, i1 %compare.low1287, i1 %compare.high.order1284, !dbg !53
  %748 = select i1 %compare.result1288, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1289 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 32) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1290 = or i32 %compare.local.base1289, 1, !dbg !53
  %compare.low1291 = icmp ugt i32 %compare.low.base1285, %compare.local1290, !dbg !53
  %compare.result1292 = select i1 %compare.high.equal1283, i1 %compare.low1291, i1 %compare.high.order1284, !dbg !53
  %749 = select i1 %compare.result1292, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1293 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 33) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1294 = or i32 %compare.local.base1293, 2, !dbg !53
  %compare.low1295 = icmp ugt i32 %compare.low.base1285, %compare.local1294, !dbg !53
  %compare.result1296 = select i1 %compare.high.equal1283, i1 %compare.low1295, i1 %compare.high.order1284, !dbg !53
  %750 = select i1 %compare.result1296, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1297 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 34) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1298 = or i32 %compare.local.base1297, 3, !dbg !53
  %compare.low1299 = icmp ugt i32 %compare.low.base1285, %compare.local1298, !dbg !53
  %compare.result1300 = select i1 %compare.high.equal1283, i1 %compare.low1299, i1 %compare.high.order1284, !dbg !53
  %751 = select i1 %compare.result1300, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1301 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 35) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1302 = or i32 %compare.local.base1301, 4, !dbg !53
  %compare.low1303 = icmp ugt i32 %compare.low.base1285, %compare.local1302, !dbg !53
  %compare.result1304 = select i1 %compare.high.equal1283, i1 %compare.low1303, i1 %compare.high.order1284, !dbg !53
  %752 = select i1 %compare.result1304, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1305 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 36) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1306 = or i32 %compare.local.base1305, 5, !dbg !53
  %compare.low1307 = icmp ugt i32 %compare.low.base1285, %compare.local1306, !dbg !53
  %compare.result1308 = select i1 %compare.high.equal1283, i1 %compare.low1307, i1 %compare.high.order1284, !dbg !53
  %753 = select i1 %compare.result1308, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1309 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 37) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1310 = or i32 %compare.local.base1309, 6, !dbg !53
  %compare.low1311 = icmp ugt i32 %compare.low.base1285, %compare.local1310, !dbg !53
  %compare.result1312 = select i1 %compare.high.equal1283, i1 %compare.low1311, i1 %compare.high.order1284, !dbg !53
  %754 = select i1 %compare.result1312, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1313 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 38) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1314 = or i32 %compare.local.base1313, 7, !dbg !53
  %compare.low1315 = icmp ugt i32 %compare.low.base1285, %compare.local1314, !dbg !53
  %compare.result1316 = select i1 %compare.high.equal1283, i1 %compare.low1315, i1 %compare.high.order1284, !dbg !53
  %755 = select i1 %compare.result1316, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1317 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 39) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1318 = or i32 %compare.local.base1317, 8, !dbg !53
  %compare.low1319 = icmp ugt i32 %compare.low.base1285, %compare.local1318, !dbg !53
  %compare.result1320 = select i1 %compare.high.equal1283, i1 %compare.low1319, i1 %compare.high.order1284, !dbg !53
  %756 = select i1 %compare.result1320, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1321 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 40) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1322 = or i32 %compare.local.base1321, 9, !dbg !53
  %compare.low1323 = icmp ugt i32 %compare.low.base1285, %compare.local1322, !dbg !53
  %compare.result1324 = select i1 %compare.high.equal1283, i1 %compare.low1323, i1 %compare.high.order1284, !dbg !53
  %757 = select i1 %compare.result1324, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1325 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 41) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1326 = or i32 %compare.local.base1325, 10, !dbg !53
  %compare.low1327 = icmp ugt i32 %compare.low.base1285, %compare.local1326, !dbg !53
  %compare.result1328 = select i1 %compare.high.equal1283, i1 %compare.low1327, i1 %compare.high.order1284, !dbg !53
  %758 = select i1 %compare.result1328, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1329 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 42) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1330 = or i32 %compare.local.base1329, 11, !dbg !53
  %compare.low1331 = icmp ugt i32 %compare.low.base1285, %compare.local1330, !dbg !53
  %compare.result1332 = select i1 %compare.high.equal1283, i1 %compare.low1331, i1 %compare.high.order1284, !dbg !53
  %759 = select i1 %compare.result1332, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1333 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 43) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1334 = or i32 %compare.local.base1333, 12, !dbg !53
  %compare.low1335 = icmp ugt i32 %compare.low.base1285, %compare.local1334, !dbg !53
  %compare.result1336 = select i1 %compare.high.equal1283, i1 %compare.low1335, i1 %compare.high.order1284, !dbg !53
  %760 = select i1 %compare.result1336, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1337 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 44) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1338 = or i32 %compare.local.base1337, 13, !dbg !53
  %compare.low1339 = icmp ugt i32 %compare.low.base1285, %compare.local1338, !dbg !53
  %compare.result1340 = select i1 %compare.high.equal1283, i1 %compare.low1339, i1 %compare.high.order1284, !dbg !53
  %761 = select i1 %compare.result1340, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1341 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 45) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1342 = or i32 %compare.local.base1341, 14, !dbg !53
  %compare.low1343 = icmp ugt i32 %compare.low.base1285, %compare.local1342, !dbg !53
  %compare.result1344 = select i1 %compare.high.equal1283, i1 %compare.low1343, i1 %compare.high.order1284, !dbg !53
  %762 = select i1 %compare.result1344, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1345 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 46) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1346 = or i32 %compare.local.base1345, 15, !dbg !53
  %compare.low1347 = icmp ugt i32 %compare.low.base1285, %compare.local1346, !dbg !53
  %compare.result1348 = select i1 %compare.high.equal1283, i1 %compare.low1347, i1 %compare.high.order1284, !dbg !53
  %763 = select i1 %compare.result1348, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1349 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 47) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1350 = or i32 %compare.local.base1349, 64, !dbg !53
  %compare.low1351 = icmp ugt i32 %compare.low.base1285, %compare.local1350, !dbg !53
  %compare.result1352 = select i1 %compare.high.equal1283, i1 %compare.low1351, i1 %compare.high.order1284, !dbg !53
  %764 = select i1 %compare.result1352, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1353 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 48) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1354 = or i32 %compare.local.base1353, 65, !dbg !53
  %compare.low1355 = icmp ugt i32 %compare.low.base1285, %compare.local1354, !dbg !53
  %compare.result1356 = select i1 %compare.high.equal1283, i1 %compare.low1355, i1 %compare.high.order1284, !dbg !53
  %765 = select i1 %compare.result1356, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1357 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 49) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1358 = or i32 %compare.local.base1357, 66, !dbg !53
  %compare.low1359 = icmp ugt i32 %compare.low.base1285, %compare.local1358, !dbg !53
  %compare.result1360 = select i1 %compare.high.equal1283, i1 %compare.low1359, i1 %compare.high.order1284, !dbg !53
  %766 = select i1 %compare.result1360, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1361 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 50) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1362 = or i32 %compare.local.base1361, 67, !dbg !53
  %compare.low1363 = icmp ugt i32 %compare.low.base1285, %compare.local1362, !dbg !53
  %compare.result1364 = select i1 %compare.high.equal1283, i1 %compare.low1363, i1 %compare.high.order1284, !dbg !53
  %767 = select i1 %compare.result1364, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1365 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 51) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1366 = or i32 %compare.local.base1365, 68, !dbg !53
  %compare.low1367 = icmp ugt i32 %compare.low.base1285, %compare.local1366, !dbg !53
  %compare.result1368 = select i1 %compare.high.equal1283, i1 %compare.low1367, i1 %compare.high.order1284, !dbg !53
  %768 = select i1 %compare.result1368, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1369 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 52) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1370 = or i32 %compare.local.base1369, 69, !dbg !53
  %compare.low1371 = icmp ugt i32 %compare.low.base1285, %compare.local1370, !dbg !53
  %compare.result1372 = select i1 %compare.high.equal1283, i1 %compare.low1371, i1 %compare.high.order1284, !dbg !53
  %769 = select i1 %compare.result1372, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1373 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 53) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1374 = or i32 %compare.local.base1373, 70, !dbg !53
  %compare.low1375 = icmp ugt i32 %compare.low.base1285, %compare.local1374, !dbg !53
  %compare.result1376 = select i1 %compare.high.equal1283, i1 %compare.low1375, i1 %compare.high.order1284, !dbg !53
  %770 = select i1 %compare.result1376, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1377 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 54) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1378 = or i32 %compare.local.base1377, 71, !dbg !53
  %compare.low1379 = icmp ugt i32 %compare.low.base1285, %compare.local1378, !dbg !53
  %compare.result1380 = select i1 %compare.high.equal1283, i1 %compare.low1379, i1 %compare.high.order1284, !dbg !53
  %771 = select i1 %compare.result1380, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1381 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 55) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1382 = or i32 %compare.local.base1381, 72, !dbg !53
  %compare.low1383 = icmp ugt i32 %compare.low.base1285, %compare.local1382, !dbg !53
  %compare.result1384 = select i1 %compare.high.equal1283, i1 %compare.low1383, i1 %compare.high.order1284, !dbg !53
  %772 = select i1 %compare.result1384, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1385 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 56) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1386 = or i32 %compare.local.base1385, 73, !dbg !53
  %compare.low1387 = icmp ugt i32 %compare.low.base1285, %compare.local1386, !dbg !53
  %compare.result1388 = select i1 %compare.high.equal1283, i1 %compare.low1387, i1 %compare.high.order1284, !dbg !53
  %773 = select i1 %compare.result1388, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1389 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 57) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1390 = or i32 %compare.local.base1389, 74, !dbg !53
  %compare.low1391 = icmp ugt i32 %compare.low.base1285, %compare.local1390, !dbg !53
  %compare.result1392 = select i1 %compare.high.equal1283, i1 %compare.low1391, i1 %compare.high.order1284, !dbg !53
  %774 = select i1 %compare.result1392, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1393 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 58) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1394 = or i32 %compare.local.base1393, 75, !dbg !53
  %compare.low1395 = icmp ugt i32 %compare.low.base1285, %compare.local1394, !dbg !53
  %compare.result1396 = select i1 %compare.high.equal1283, i1 %compare.low1395, i1 %compare.high.order1284, !dbg !53
  %775 = select i1 %compare.result1396, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1397 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 59) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1398 = or i32 %compare.local.base1397, 76, !dbg !53
  %compare.low1399 = icmp ugt i32 %compare.low.base1285, %compare.local1398, !dbg !53
  %compare.result1400 = select i1 %compare.high.equal1283, i1 %compare.low1399, i1 %compare.high.order1284, !dbg !53
  %776 = select i1 %compare.result1400, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1401 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 60) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1402 = or i32 %compare.local.base1401, 77, !dbg !53
  %compare.low1403 = icmp ugt i32 %compare.low.base1285, %compare.local1402, !dbg !53
  %compare.result1404 = select i1 %compare.high.equal1283, i1 %compare.low1403, i1 %compare.high.order1284, !dbg !53
  %777 = select i1 %compare.result1404, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1405 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 61) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1406 = or i32 %compare.local.base1405, 78, !dbg !53
  %compare.low1407 = icmp ugt i32 %compare.low.base1285, %compare.local1406, !dbg !53
  %compare.result1408 = select i1 %compare.high.equal1283, i1 %compare.low1407, i1 %compare.high.order1284, !dbg !53
  %778 = select i1 %compare.result1408, float 0.000000e+00, float -inf, !dbg !54
  %compare.local.base1409 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1286, i32 62) #9, !dbg !53, !frisk.register_lifetimes !34
  %compare.local1410 = or i32 %compare.local.base1409, 79, !dbg !53
  %compare.low1411 = icmp ugt i32 %compare.low.base1285, %compare.local1410, !dbg !53
  %compare.result1412 = select i1 %compare.high.equal1283, i1 %compare.low1411, i1 %compare.high.order1284, !dbg !53
  %779 = select i1 %compare.result1412, float 0.000000e+00, float -inf, !dbg !54
  %780 = insertelement <32 x float> poison, float %713, i64 0, !dbg !43
  %781 = insertelement <32 x float> %780, float %714, i64 1, !dbg !43
  %782 = insertelement <32 x float> %781, float %715, i64 2, !dbg !43
  %783 = insertelement <32 x float> %782, float %716, i64 3, !dbg !43
  %784 = insertelement <32 x float> %783, float %717, i64 4, !dbg !43
  %785 = insertelement <32 x float> %784, float %718, i64 5, !dbg !43
  %786 = insertelement <32 x float> %785, float %719, i64 6, !dbg !43
  %787 = insertelement <32 x float> %786, float %720, i64 7, !dbg !43
  %788 = insertelement <32 x float> %787, float %721, i64 8, !dbg !43
  %789 = insertelement <32 x float> %788, float %722, i64 9, !dbg !43
  %790 = insertelement <32 x float> %789, float %723, i64 10, !dbg !43
  %791 = insertelement <32 x float> %790, float %724, i64 11, !dbg !43
  %792 = insertelement <32 x float> %791, float %725, i64 12, !dbg !43
  %793 = insertelement <32 x float> %792, float %726, i64 13, !dbg !43
  %794 = insertelement <32 x float> %793, float %727, i64 14, !dbg !43
  %795 = insertelement <32 x float> %794, float %728, i64 15, !dbg !43
  %796 = insertelement <32 x float> %795, float %729, i64 16, !dbg !43
  %797 = insertelement <32 x float> %796, float %730, i64 17, !dbg !43
  %798 = insertelement <32 x float> %797, float %731, i64 18, !dbg !43
  %799 = insertelement <32 x float> %798, float %732, i64 19, !dbg !43
  %800 = insertelement <32 x float> %799, float %733, i64 20, !dbg !43
  %801 = insertelement <32 x float> %800, float %734, i64 21, !dbg !43
  %802 = insertelement <32 x float> %801, float %735, i64 22, !dbg !43
  %803 = insertelement <32 x float> %802, float %736, i64 23, !dbg !43
  %804 = insertelement <32 x float> %803, float %737, i64 24, !dbg !43
  %805 = insertelement <32 x float> %804, float %738, i64 25, !dbg !43
  %806 = insertelement <32 x float> %805, float %739, i64 26, !dbg !43
  %807 = insertelement <32 x float> %806, float %740, i64 27, !dbg !43
  %808 = insertelement <32 x float> %807, float %741, i64 28, !dbg !43
  %809 = insertelement <32 x float> %808, float %742, i64 29, !dbg !43
  %810 = insertelement <32 x float> %809, float %743, i64 30, !dbg !43
  %811 = insertelement <32 x float> %810, float %744, i64 31, !dbg !43
  %812 = insertvalue [2 x <32 x float>] poison, <32 x float> %811, 0, !dbg !43
  %813 = insertelement <32 x float> poison, float %748, i64 0, !dbg !43
  %814 = insertelement <32 x float> %813, float %749, i64 1, !dbg !43
  %815 = insertelement <32 x float> %814, float %750, i64 2, !dbg !43
  %816 = insertelement <32 x float> %815, float %751, i64 3, !dbg !43
  %817 = insertelement <32 x float> %816, float %752, i64 4, !dbg !43
  %818 = insertelement <32 x float> %817, float %753, i64 5, !dbg !43
  %819 = insertelement <32 x float> %818, float %754, i64 6, !dbg !43
  %820 = insertelement <32 x float> %819, float %755, i64 7, !dbg !43
  %821 = insertelement <32 x float> %820, float %756, i64 8, !dbg !43
  %822 = insertelement <32 x float> %821, float %757, i64 9, !dbg !43
  %823 = insertelement <32 x float> %822, float %758, i64 10, !dbg !43
  %824 = insertelement <32 x float> %823, float %759, i64 11, !dbg !43
  %825 = insertelement <32 x float> %824, float %760, i64 12, !dbg !43
  %826 = insertelement <32 x float> %825, float %761, i64 13, !dbg !43
  %827 = insertelement <32 x float> %826, float %762, i64 14, !dbg !43
  %828 = insertelement <32 x float> %827, float %763, i64 15, !dbg !43
  %829 = insertelement <32 x float> %828, float %764, i64 16, !dbg !43
  %830 = insertelement <32 x float> %829, float %765, i64 17, !dbg !43
  %831 = insertelement <32 x float> %830, float %766, i64 18, !dbg !43
  %832 = insertelement <32 x float> %831, float %767, i64 19, !dbg !43
  %833 = insertelement <32 x float> %832, float %768, i64 20, !dbg !43
  %834 = insertelement <32 x float> %833, float %769, i64 21, !dbg !43
  %835 = insertelement <32 x float> %834, float %770, i64 22, !dbg !43
  %836 = insertelement <32 x float> %835, float %771, i64 23, !dbg !43
  %837 = insertelement <32 x float> %836, float %772, i64 24, !dbg !43
  %838 = insertelement <32 x float> %837, float %773, i64 25, !dbg !43
  %839 = insertelement <32 x float> %838, float %774, i64 26, !dbg !43
  %840 = insertelement <32 x float> %839, float %775, i64 27, !dbg !43
  %841 = insertelement <32 x float> %840, float %776, i64 28, !dbg !43
  %842 = insertelement <32 x float> %841, float %777, i64 29, !dbg !43
  %843 = insertelement <32 x float> %842, float %778, i64 30, !dbg !43
  %844 = insertelement <32 x float> %843, float %779, i64 31, !dbg !43
  %845 = insertvalue [2 x <32 x float>] %812, <32 x float> %844, 1, !dbg !43
  %.masked = extractvalue [2 x <32 x float>] %845, 0, !dbg !55
  %.masked1 = shufflevector <32 x float> %.masked, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !55
  %.masked2 = fadd <4 x float> %638, %.masked1, !dbg !55
  %.masked3 = shufflevector <32 x float> %.masked, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !55
  %.masked4 = fadd <4 x float> %642, %.masked3, !dbg !55
  %.masked5 = shufflevector <32 x float> %.masked, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !55
  %.masked6 = fadd <4 x float> %646, %.masked5, !dbg !55
  %.masked7 = shufflevector <32 x float> %.masked, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !55
  %.masked8 = fadd <4 x float> %650, %.masked7, !dbg !55
  %.masked9 = shufflevector <32 x float> %.masked, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !55
  %.masked10 = fadd <4 x float> %654, %.masked9, !dbg !55
  %.masked11 = shufflevector <32 x float> %.masked, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !55
  %.masked12 = fadd <4 x float> %658, %.masked11, !dbg !55
  %.masked13 = shufflevector <32 x float> %.masked, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !55
  %.masked14 = fadd <4 x float> %662, %.masked13, !dbg !55
  %.masked15 = shufflevector <32 x float> %.masked, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !55
  %.masked16 = fadd <4 x float> %666, %.masked15, !dbg !55
  %.masked17 = extractvalue [2 x <32 x float>] %845, 1, !dbg !55
  %.masked18 = shufflevector <32 x float> %.masked17, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !55
  %.masked19 = fadd <4 x float> %670, %.masked18, !dbg !55
  %.masked20 = shufflevector <32 x float> %.masked17, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !55
  %.masked21 = fadd <4 x float> %674, %.masked20, !dbg !55
  %.masked22 = shufflevector <32 x float> %.masked17, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !55
  %.masked23 = fadd <4 x float> %678, %.masked22, !dbg !55
  %.masked24 = shufflevector <32 x float> %.masked17, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !55
  %.masked25 = fadd <4 x float> %682, %.masked24, !dbg !55
  %.masked26 = shufflevector <32 x float> %.masked17, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !55
  %.masked27 = fadd <4 x float> %686, %.masked26, !dbg !55
  %.masked28 = shufflevector <32 x float> %.masked17, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !55
  %.masked29 = fadd <4 x float> %690, %.masked28, !dbg !55
  %.masked30 = shufflevector <32 x float> %.masked17, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !55
  %.masked31 = fadd <4 x float> %694, %.masked30, !dbg !55
  %.masked32 = shufflevector <32 x float> %.masked17, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !55
  %.masked33 = fadd <4 x float> %699, %.masked32, !dbg !55
  br label %846, !dbg !43

846:                                              ; preds = %702, %701
  %847 = phi <4 x float> [ %.masked2, %702 ], [ %638, %701 ], !dbg !55
  %848 = phi <4 x float> [ %.masked4, %702 ], [ %642, %701 ], !dbg !55
  %849 = phi <4 x float> [ %.masked6, %702 ], [ %646, %701 ], !dbg !55
  %850 = phi <4 x float> [ %.masked8, %702 ], [ %650, %701 ], !dbg !55
  %851 = phi <4 x float> [ %.masked10, %702 ], [ %654, %701 ], !dbg !55
  %852 = phi <4 x float> [ %.masked12, %702 ], [ %658, %701 ], !dbg !55
  %853 = phi <4 x float> [ %.masked14, %702 ], [ %662, %701 ], !dbg !55
  %854 = phi <4 x float> [ %.masked16, %702 ], [ %666, %701 ], !dbg !55
  %855 = phi <4 x float> [ %.masked19, %702 ], [ %670, %701 ], !dbg !55
  %856 = phi <4 x float> [ %.masked21, %702 ], [ %674, %701 ], !dbg !55
  %857 = phi <4 x float> [ %.masked23, %702 ], [ %678, %701 ], !dbg !55
  %858 = phi <4 x float> [ %.masked25, %702 ], [ %682, %701 ], !dbg !55
  %859 = phi <4 x float> [ %.masked27, %702 ], [ %686, %701 ], !dbg !55
  %860 = phi <4 x float> [ %.masked29, %702 ], [ %690, %701 ], !dbg !55
  %861 = phi <4 x float> [ %.masked31, %702 ], [ %694, %701 ], !dbg !55
  %862 = phi <4 x float> [ %.masked33, %702 ], [ %699, %701 ], !dbg !55
  br label %863, !dbg !43

863:                                              ; preds = %846
  %864 = extractelement <4 x float> %847, i64 0, !dbg !55
  %865 = extractelement <4 x float> %847, i64 1, !dbg !55
  %866 = extractelement <4 x float> %847, i64 2, !dbg !55
  %867 = extractelement <4 x float> %847, i64 3, !dbg !55
  %868 = extractelement <4 x float> %848, i64 0, !dbg !55
  %869 = extractelement <4 x float> %848, i64 1, !dbg !55
  %870 = extractelement <4 x float> %848, i64 2, !dbg !55
  %871 = extractelement <4 x float> %848, i64 3, !dbg !55
  %872 = extractelement <4 x float> %849, i64 0, !dbg !55
  %873 = extractelement <4 x float> %849, i64 1, !dbg !55
  %874 = extractelement <4 x float> %849, i64 2, !dbg !55
  %875 = extractelement <4 x float> %849, i64 3, !dbg !55
  %876 = extractelement <4 x float> %850, i64 0, !dbg !55
  %877 = extractelement <4 x float> %850, i64 1, !dbg !55
  %878 = extractelement <4 x float> %850, i64 2, !dbg !55
  %879 = extractelement <4 x float> %850, i64 3, !dbg !55
  %880 = extractelement <4 x float> %851, i64 0, !dbg !55
  %881 = extractelement <4 x float> %851, i64 1, !dbg !55
  %882 = extractelement <4 x float> %851, i64 2, !dbg !55
  %883 = extractelement <4 x float> %851, i64 3, !dbg !55
  %884 = extractelement <4 x float> %852, i64 0, !dbg !55
  %885 = extractelement <4 x float> %852, i64 1, !dbg !55
  %886 = extractelement <4 x float> %852, i64 2, !dbg !55
  %887 = extractelement <4 x float> %852, i64 3, !dbg !55
  %888 = extractelement <4 x float> %853, i64 0, !dbg !55
  %889 = extractelement <4 x float> %853, i64 1, !dbg !55
  %890 = extractelement <4 x float> %853, i64 2, !dbg !55
  %891 = extractelement <4 x float> %853, i64 3, !dbg !55
  %892 = extractelement <4 x float> %854, i64 0, !dbg !55
  %893 = extractelement <4 x float> %854, i64 1, !dbg !55
  %894 = extractelement <4 x float> %854, i64 2, !dbg !55
  %895 = extractelement <4 x float> %854, i64 3, !dbg !55
  %896 = extractelement <4 x float> %855, i64 0, !dbg !55
  %897 = extractelement <4 x float> %855, i64 1, !dbg !55
  %898 = extractelement <4 x float> %855, i64 2, !dbg !55
  %899 = extractelement <4 x float> %855, i64 3, !dbg !55
  %900 = extractelement <4 x float> %856, i64 0, !dbg !55
  %901 = extractelement <4 x float> %856, i64 1, !dbg !55
  %902 = extractelement <4 x float> %856, i64 2, !dbg !55
  %903 = extractelement <4 x float> %856, i64 3, !dbg !55
  %904 = extractelement <4 x float> %857, i64 0, !dbg !55
  %905 = extractelement <4 x float> %857, i64 1, !dbg !55
  %906 = extractelement <4 x float> %857, i64 2, !dbg !55
  %907 = extractelement <4 x float> %857, i64 3, !dbg !55
  %908 = extractelement <4 x float> %858, i64 0, !dbg !55
  %909 = extractelement <4 x float> %858, i64 1, !dbg !55
  %910 = extractelement <4 x float> %858, i64 2, !dbg !55
  %911 = extractelement <4 x float> %858, i64 3, !dbg !55
  %912 = extractelement <4 x float> %859, i64 0, !dbg !55
  %913 = extractelement <4 x float> %859, i64 1, !dbg !55
  %914 = extractelement <4 x float> %859, i64 2, !dbg !55
  %915 = extractelement <4 x float> %859, i64 3, !dbg !55
  %916 = extractelement <4 x float> %860, i64 0, !dbg !55
  %917 = extractelement <4 x float> %860, i64 1, !dbg !55
  %918 = extractelement <4 x float> %860, i64 2, !dbg !55
  %919 = extractelement <4 x float> %860, i64 3, !dbg !55
  %920 = extractelement <4 x float> %861, i64 0, !dbg !55
  %921 = extractelement <4 x float> %861, i64 1, !dbg !55
  %922 = extractelement <4 x float> %861, i64 2, !dbg !55
  %923 = extractelement <4 x float> %861, i64 3, !dbg !55
  %924 = extractelement <4 x float> %862, i64 0, !dbg !55
  %925 = extractelement <4 x float> %862, i64 1, !dbg !55
  %926 = extractelement <4 x float> %862, i64 2, !dbg !55
  %927 = extractelement <4 x float> %862, i64 3, !dbg !55
  %928 = call float @llvm.maxnum.f32(float %864, float -inf), !dbg !56
  %929 = call float @llvm.maxnum.f32(float %928, float %865), !dbg !56
  %930 = call float @llvm.maxnum.f32(float %929, float %866), !dbg !56
  %931 = call float @llvm.maxnum.f32(float %930, float %867), !dbg !56
  %932 = call float @llvm.maxnum.f32(float %931, float %868), !dbg !56
  %933 = call float @llvm.maxnum.f32(float %932, float %869), !dbg !56
  %934 = call float @llvm.maxnum.f32(float %933, float %870), !dbg !56
  %935 = call float @llvm.maxnum.f32(float %934, float %871), !dbg !56
  %936 = call float @llvm.maxnum.f32(float %935, float %872), !dbg !56
  %937 = call float @llvm.maxnum.f32(float %936, float %873), !dbg !56
  %938 = call float @llvm.maxnum.f32(float %937, float %874), !dbg !56
  %939 = call float @llvm.maxnum.f32(float %938, float %875), !dbg !56
  %940 = call float @llvm.maxnum.f32(float %939, float %876), !dbg !56
  %941 = call float @llvm.maxnum.f32(float %940, float %877), !dbg !56
  %942 = call float @llvm.maxnum.f32(float %941, float %878), !dbg !56
  %943 = call float @llvm.maxnum.f32(float %942, float %879), !dbg !56
  %944 = call float @llvm.maxnum.f32(float %943, float %880), !dbg !56
  %945 = call float @llvm.maxnum.f32(float %944, float %881), !dbg !56
  %946 = call float @llvm.maxnum.f32(float %945, float %882), !dbg !56
  %947 = call float @llvm.maxnum.f32(float %946, float %883), !dbg !56
  %948 = call float @llvm.maxnum.f32(float %947, float %884), !dbg !56
  %949 = call float @llvm.maxnum.f32(float %948, float %885), !dbg !56
  %950 = call float @llvm.maxnum.f32(float %949, float %886), !dbg !56
  %951 = call float @llvm.maxnum.f32(float %950, float %887), !dbg !56
  %952 = call float @llvm.maxnum.f32(float %951, float %888), !dbg !56
  %953 = call float @llvm.maxnum.f32(float %952, float %889), !dbg !56
  %954 = call float @llvm.maxnum.f32(float %953, float %890), !dbg !56
  %955 = call float @llvm.maxnum.f32(float %954, float %891), !dbg !56
  %956 = call float @llvm.maxnum.f32(float %955, float %892), !dbg !56
  %957 = call float @llvm.maxnum.f32(float %956, float %893), !dbg !56
  %958 = call float @llvm.maxnum.f32(float %957, float %894), !dbg !56
  %959 = call float @llvm.maxnum.f32(float %958, float %895), !dbg !56
  %960 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !56
  %961 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %960), !dbg !56
  %962 = add nuw nsw i32 %961, 64, !dbg !56
  %963 = xor i32 %961, 16, !dbg !56
  %964 = and i32 %962, 192, !dbg !56
  %965 = icmp samesign ult i32 %963, %964, !dbg !56
  %966 = select i1 %965, i32 %963, i32 %961, !dbg !56
  %967 = shl nuw nsw i32 %966, 2, !dbg !56
  %968 = bitcast float %959 to i32, !dbg !56
  %969 = call i32 @llvm.amdgcn.ds.bpermute(i32 %967, i32 %968), !dbg !56
  %970 = bitcast i32 %969 to float, !dbg !56
  %971 = call float @llvm.maxnum.f32(float %959, float %970), !dbg !56
  %972 = xor i32 %961, 32, !dbg !56
  %973 = icmp samesign ult i32 %972, %964, !dbg !56
  %974 = select i1 %973, i32 %972, i32 %961, !dbg !56
  %975 = shl nuw nsw i32 %974, 2, !dbg !56
  %976 = bitcast float %971 to i32, !dbg !56
  %977 = call i32 @llvm.amdgcn.ds.bpermute(i32 %975, i32 %976), !dbg !56
  %978 = bitcast i32 %977 to float, !dbg !56
  %979 = call float @llvm.maxnum.f32(float %971, float %978), !dbg !56
  %980 = extractvalue [2 x <1 x float>] %208, 0, !dbg !56
  %981 = insertelement <1 x float> poison, float %979, i64 0, !dbg !56
  %982 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %981, <1 x float> %980), !dbg !56
  %983 = call float @llvm.maxnum.f32(float %896, float -inf), !dbg !56
  %984 = call float @llvm.maxnum.f32(float %983, float %897), !dbg !56
  %985 = call float @llvm.maxnum.f32(float %984, float %898), !dbg !56
  %986 = call float @llvm.maxnum.f32(float %985, float %899), !dbg !56
  %987 = call float @llvm.maxnum.f32(float %986, float %900), !dbg !56
  %988 = call float @llvm.maxnum.f32(float %987, float %901), !dbg !56
  %989 = call float @llvm.maxnum.f32(float %988, float %902), !dbg !56
  %990 = call float @llvm.maxnum.f32(float %989, float %903), !dbg !56
  %991 = call float @llvm.maxnum.f32(float %990, float %904), !dbg !56
  %992 = call float @llvm.maxnum.f32(float %991, float %905), !dbg !56
  %993 = call float @llvm.maxnum.f32(float %992, float %906), !dbg !56
  %994 = call float @llvm.maxnum.f32(float %993, float %907), !dbg !56
  %995 = call float @llvm.maxnum.f32(float %994, float %908), !dbg !56
  %996 = call float @llvm.maxnum.f32(float %995, float %909), !dbg !56
  %997 = call float @llvm.maxnum.f32(float %996, float %910), !dbg !56
  %998 = call float @llvm.maxnum.f32(float %997, float %911), !dbg !56
  %999 = call float @llvm.maxnum.f32(float %998, float %912), !dbg !56
  %1000 = call float @llvm.maxnum.f32(float %999, float %913), !dbg !56
  %1001 = call float @llvm.maxnum.f32(float %1000, float %914), !dbg !56
  %1002 = call float @llvm.maxnum.f32(float %1001, float %915), !dbg !56
  %1003 = call float @llvm.maxnum.f32(float %1002, float %916), !dbg !56
  %1004 = call float @llvm.maxnum.f32(float %1003, float %917), !dbg !56
  %1005 = call float @llvm.maxnum.f32(float %1004, float %918), !dbg !56
  %1006 = call float @llvm.maxnum.f32(float %1005, float %919), !dbg !56
  %1007 = call float @llvm.maxnum.f32(float %1006, float %920), !dbg !56
  %1008 = call float @llvm.maxnum.f32(float %1007, float %921), !dbg !56
  %1009 = call float @llvm.maxnum.f32(float %1008, float %922), !dbg !56
  %1010 = call float @llvm.maxnum.f32(float %1009, float %923), !dbg !56
  %1011 = call float @llvm.maxnum.f32(float %1010, float %924), !dbg !56
  %1012 = call float @llvm.maxnum.f32(float %1011, float %925), !dbg !56
  %1013 = call float @llvm.maxnum.f32(float %1012, float %926), !dbg !56
  %1014 = call float @llvm.maxnum.f32(float %1013, float %927), !dbg !56
  %1015 = bitcast float %1014 to i32, !dbg !56
  %1016 = call i32 @llvm.amdgcn.ds.bpermute(i32 %967, i32 %1015), !dbg !56
  %1017 = bitcast i32 %1016 to float, !dbg !56
  %1018 = call float @llvm.maxnum.f32(float %1014, float %1017), !dbg !56
  %1019 = bitcast float %1018 to i32, !dbg !56
  %1020 = call i32 @llvm.amdgcn.ds.bpermute(i32 %975, i32 %1019), !dbg !56
  %1021 = bitcast i32 %1020 to float, !dbg !56
  %1022 = call float @llvm.maxnum.f32(float %1018, float %1021), !dbg !56
  %1023 = extractvalue [2 x <1 x float>] %208, 1, !dbg !56
  %1024 = insertelement <1 x float> poison, float %1022, i64 0, !dbg !56
  %1025 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %1024, <1 x float> %1023), !dbg !56
  %1026 = shufflevector <1 x float> %982, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !57
  %1027 = fsub <4 x float> %847, %1026, !dbg !57
  %1028 = fsub <4 x float> %848, %1026, !dbg !57
  %1029 = fsub <4 x float> %849, %1026, !dbg !57
  %1030 = fsub <4 x float> %850, %1026, !dbg !57
  %1031 = fsub <4 x float> %851, %1026, !dbg !57
  %1032 = fsub <4 x float> %852, %1026, !dbg !57
  %1033 = fsub <4 x float> %853, %1026, !dbg !57
  %1034 = fsub <4 x float> %854, %1026, !dbg !57
  %1035 = shufflevector <1 x float> %1025, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !57
  %1036 = fsub <4 x float> %855, %1035, !dbg !57
  %1037 = fsub <4 x float> %856, %1035, !dbg !57
  %1038 = fsub <4 x float> %857, %1035, !dbg !57
  %1039 = fsub <4 x float> %858, %1035, !dbg !57
  %1040 = fsub <4 x float> %859, %1035, !dbg !57
  %1041 = fsub <4 x float> %860, %1035, !dbg !57
  %1042 = fsub <4 x float> %861, %1035, !dbg !57
  %1043 = fsub <4 x float> %862, %1035, !dbg !57
  %1044 = fsub <1 x float> %980, %982, !dbg !58
  %1045 = extractelement <1 x float> %1044, i64 0, !dbg !59
  %1046 = call float @llvm.amdgcn.exp2.f32(float %1045), !dbg !59
  %1047 = insertelement <4 x float> poison, float %1046, i64 0, !dbg !64
  %1048 = shufflevector <4 x float> %1047, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !64
  %1049 = extractvalue [2 x <1 x float>] %207, 0, !dbg !60
  %1050 = extractelement <1 x float> %1049, i64 0, !dbg !60
  %1051 = fmul float %1050, %1046, !dbg !60
  %1052 = fsub <1 x float> %1023, %1025, !dbg !58
  %1053 = extractelement <1 x float> %1052, i64 0, !dbg !59
  %1054 = call float @llvm.amdgcn.exp2.f32(float %1053), !dbg !59
  %1055 = insertelement <4 x float> poison, float %1054, i64 0, !dbg !64
  %1056 = shufflevector <4 x float> %1055, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !64
  %1057 = extractvalue [2 x <1 x float>] %207, 1, !dbg !60
  %1058 = extractelement <1 x float> %1057, i64 0, !dbg !60
  %1059 = fmul float %1058, %1054, !dbg !60
  %1060 = extractelement <4 x float> %1027, i64 0, !dbg !61
  %1061 = call float @llvm.amdgcn.exp2.f32(float %1060), !dbg !61
  %1062 = insertelement <4 x float> poison, float %1061, i64 0, !dbg !63
  %1063 = extractelement <4 x float> %1027, i64 1, !dbg !61
  %1064 = call float @llvm.amdgcn.exp2.f32(float %1063), !dbg !61
  %1065 = insertelement <4 x float> %1062, float %1064, i64 1, !dbg !63
  %1066 = fadd float %1061, %1064, !dbg !62
  %1067 = extractelement <4 x float> %1027, i64 2, !dbg !61
  %1068 = call float @llvm.amdgcn.exp2.f32(float %1067), !dbg !61
  %1069 = insertelement <4 x float> %1065, float %1068, i64 2, !dbg !63
  %1070 = fadd float %1066, %1068, !dbg !62
  %1071 = extractelement <4 x float> %1027, i64 3, !dbg !61
  %1072 = call float @llvm.amdgcn.exp2.f32(float %1071), !dbg !61
  %1073 = insertelement <4 x float> %1069, float %1072, i64 3, !dbg !63
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1073) #9, !dbg !63, !frisk.register_lifetimes !34
  %1074 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !63
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1074) #9, !dbg !62, !frisk.register_lifetimes !34
  %1075 = fadd float %1070, %1072, !dbg !62
  %1076 = extractelement <4 x float> %1028, i64 0, !dbg !61
  %1077 = call float @llvm.amdgcn.exp2.f32(float %1076), !dbg !61
  %1078 = insertelement <4 x float> poison, float %1077, i64 0, !dbg !63
  %1079 = fadd float %1075, %1077, !dbg !62
  %1080 = extractelement <4 x float> %1028, i64 1, !dbg !61
  %1081 = call float @llvm.amdgcn.exp2.f32(float %1080), !dbg !61
  %1082 = insertelement <4 x float> %1078, float %1081, i64 1, !dbg !63
  %1083 = fadd float %1079, %1081, !dbg !62
  %1084 = extractelement <4 x float> %1028, i64 2, !dbg !61
  %1085 = call float @llvm.amdgcn.exp2.f32(float %1084), !dbg !61
  %1086 = insertelement <4 x float> %1082, float %1085, i64 2, !dbg !63
  %1087 = fadd float %1083, %1085, !dbg !62
  %1088 = extractelement <4 x float> %1028, i64 3, !dbg !61
  %1089 = call float @llvm.amdgcn.exp2.f32(float %1088), !dbg !61
  %1090 = insertelement <4 x float> %1086, float %1089, i64 3, !dbg !63
  %math.result309 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1090) #9, !dbg !63, !frisk.register_lifetimes !34
  %1091 = fptrunc <4 x float> %math.result309 to <4 x half>, !dbg !63
  %math.packed310 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1091) #9, !dbg !62, !frisk.register_lifetimes !34
  %1092 = fadd float %1087, %1089, !dbg !62
  %1093 = extractelement <4 x float> %1029, i64 0, !dbg !61
  %1094 = call float @llvm.amdgcn.exp2.f32(float %1093), !dbg !61
  %1095 = insertelement <4 x float> poison, float %1094, i64 0, !dbg !63
  %1096 = fadd float %1092, %1094, !dbg !62
  %1097 = extractelement <4 x float> %1029, i64 1, !dbg !61
  %1098 = call float @llvm.amdgcn.exp2.f32(float %1097), !dbg !61
  %1099 = insertelement <4 x float> %1095, float %1098, i64 1, !dbg !63
  %1100 = fadd float %1096, %1098, !dbg !62
  %1101 = extractelement <4 x float> %1029, i64 2, !dbg !61
  %1102 = call float @llvm.amdgcn.exp2.f32(float %1101), !dbg !61
  %1103 = insertelement <4 x float> %1099, float %1102, i64 2, !dbg !63
  %1104 = fadd float %1100, %1102, !dbg !62
  %1105 = extractelement <4 x float> %1029, i64 3, !dbg !61
  %1106 = call float @llvm.amdgcn.exp2.f32(float %1105), !dbg !61
  %1107 = insertelement <4 x float> %1103, float %1106, i64 3, !dbg !63
  %math.result311 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1107) #9, !dbg !63, !frisk.register_lifetimes !34
  %1108 = fptrunc <4 x float> %math.result311 to <4 x half>, !dbg !63
  %math.packed312 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1108) #9, !dbg !62, !frisk.register_lifetimes !34
  %1109 = fadd float %1104, %1106, !dbg !62
  %1110 = extractelement <4 x float> %1030, i64 0, !dbg !61
  %1111 = call float @llvm.amdgcn.exp2.f32(float %1110), !dbg !61
  %1112 = insertelement <4 x float> poison, float %1111, i64 0, !dbg !63
  %1113 = fadd float %1109, %1111, !dbg !62
  %1114 = extractelement <4 x float> %1030, i64 1, !dbg !61
  %1115 = call float @llvm.amdgcn.exp2.f32(float %1114), !dbg !61
  %1116 = insertelement <4 x float> %1112, float %1115, i64 1, !dbg !63
  %1117 = fadd float %1113, %1115, !dbg !62
  %1118 = extractelement <4 x float> %1030, i64 2, !dbg !61
  %1119 = call float @llvm.amdgcn.exp2.f32(float %1118), !dbg !61
  %1120 = insertelement <4 x float> %1116, float %1119, i64 2, !dbg !63
  %1121 = fadd float %1117, %1119, !dbg !62
  %1122 = extractelement <4 x float> %1030, i64 3, !dbg !61
  %1123 = call float @llvm.amdgcn.exp2.f32(float %1122), !dbg !61
  %1124 = insertelement <4 x float> %1120, float %1123, i64 3, !dbg !63
  %math.result313 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1124) #9, !dbg !63, !frisk.register_lifetimes !34
  %1125 = fptrunc <4 x float> %math.result313 to <4 x half>, !dbg !63
  %math.packed314 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1125) #9, !dbg !62, !frisk.register_lifetimes !34
  %1126 = fadd float %1121, %1123, !dbg !62
  %1127 = extractelement <4 x float> %1031, i64 0, !dbg !61
  %1128 = call float @llvm.amdgcn.exp2.f32(float %1127), !dbg !61
  %1129 = insertelement <4 x float> poison, float %1128, i64 0, !dbg !63
  %1130 = fadd float %1126, %1128, !dbg !62
  %1131 = extractelement <4 x float> %1031, i64 1, !dbg !61
  %1132 = call float @llvm.amdgcn.exp2.f32(float %1131), !dbg !61
  %1133 = insertelement <4 x float> %1129, float %1132, i64 1, !dbg !63
  %1134 = fadd float %1130, %1132, !dbg !62
  %1135 = extractelement <4 x float> %1031, i64 2, !dbg !61
  %1136 = call float @llvm.amdgcn.exp2.f32(float %1135), !dbg !61
  %1137 = insertelement <4 x float> %1133, float %1136, i64 2, !dbg !63
  %1138 = fadd float %1134, %1136, !dbg !62
  %1139 = extractelement <4 x float> %1031, i64 3, !dbg !61
  %1140 = call float @llvm.amdgcn.exp2.f32(float %1139), !dbg !61
  %1141 = insertelement <4 x float> %1137, float %1140, i64 3, !dbg !63
  %math.result315 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1141) #9, !dbg !63, !frisk.register_lifetimes !34
  %1142 = fptrunc <4 x float> %math.result315 to <4 x half>, !dbg !63
  %math.packed316 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1142) #9, !dbg !62, !frisk.register_lifetimes !34
  %1143 = fadd float %1138, %1140, !dbg !62
  %1144 = extractelement <4 x float> %1032, i64 0, !dbg !61
  %1145 = call float @llvm.amdgcn.exp2.f32(float %1144), !dbg !61
  %1146 = insertelement <4 x float> poison, float %1145, i64 0, !dbg !63
  %1147 = fadd float %1143, %1145, !dbg !62
  %1148 = extractelement <4 x float> %1032, i64 1, !dbg !61
  %1149 = call float @llvm.amdgcn.exp2.f32(float %1148), !dbg !61
  %1150 = insertelement <4 x float> %1146, float %1149, i64 1, !dbg !63
  %1151 = fadd float %1147, %1149, !dbg !62
  %1152 = extractelement <4 x float> %1032, i64 2, !dbg !61
  %1153 = call float @llvm.amdgcn.exp2.f32(float %1152), !dbg !61
  %1154 = insertelement <4 x float> %1150, float %1153, i64 2, !dbg !63
  %1155 = fadd float %1151, %1153, !dbg !62
  %1156 = extractelement <4 x float> %1032, i64 3, !dbg !61
  %1157 = call float @llvm.amdgcn.exp2.f32(float %1156), !dbg !61
  %1158 = insertelement <4 x float> %1154, float %1157, i64 3, !dbg !63
  %math.result317 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1158) #9, !dbg !63, !frisk.register_lifetimes !34
  %1159 = fptrunc <4 x float> %math.result317 to <4 x half>, !dbg !63
  %math.packed318 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1159) #9, !dbg !62, !frisk.register_lifetimes !34
  %1160 = fadd float %1155, %1157, !dbg !62
  %1161 = extractelement <4 x float> %1033, i64 0, !dbg !61
  %1162 = call float @llvm.amdgcn.exp2.f32(float %1161), !dbg !61
  %1163 = insertelement <4 x float> poison, float %1162, i64 0, !dbg !63
  %1164 = fadd float %1160, %1162, !dbg !62
  %1165 = extractelement <4 x float> %1033, i64 1, !dbg !61
  %1166 = call float @llvm.amdgcn.exp2.f32(float %1165), !dbg !61
  %1167 = insertelement <4 x float> %1163, float %1166, i64 1, !dbg !63
  %1168 = fadd float %1164, %1166, !dbg !62
  %1169 = extractelement <4 x float> %1033, i64 2, !dbg !61
  %1170 = call float @llvm.amdgcn.exp2.f32(float %1169), !dbg !61
  %1171 = insertelement <4 x float> %1167, float %1170, i64 2, !dbg !63
  %1172 = fadd float %1168, %1170, !dbg !62
  %1173 = extractelement <4 x float> %1033, i64 3, !dbg !61
  %1174 = call float @llvm.amdgcn.exp2.f32(float %1173), !dbg !61
  %1175 = insertelement <4 x float> %1171, float %1174, i64 3, !dbg !63
  %math.result319 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1175) #9, !dbg !63, !frisk.register_lifetimes !34
  %1176 = fptrunc <4 x float> %math.result319 to <4 x half>, !dbg !63
  %math.packed320 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1176) #9, !dbg !62, !frisk.register_lifetimes !34
  %1177 = fadd float %1172, %1174, !dbg !62
  %1178 = extractelement <4 x float> %1034, i64 0, !dbg !61
  %1179 = call float @llvm.amdgcn.exp2.f32(float %1178), !dbg !61
  %1180 = insertelement <4 x float> poison, float %1179, i64 0, !dbg !63
  %1181 = fadd float %1177, %1179, !dbg !62
  %1182 = extractelement <4 x float> %1034, i64 1, !dbg !61
  %1183 = call float @llvm.amdgcn.exp2.f32(float %1182), !dbg !61
  %1184 = insertelement <4 x float> %1180, float %1183, i64 1, !dbg !63
  %1185 = fadd float %1181, %1183, !dbg !62
  %1186 = extractelement <4 x float> %1034, i64 2, !dbg !61
  %1187 = call float @llvm.amdgcn.exp2.f32(float %1186), !dbg !61
  %1188 = insertelement <4 x float> %1184, float %1187, i64 2, !dbg !63
  %1189 = fadd float %1185, %1187, !dbg !62
  %1190 = extractelement <4 x float> %1034, i64 3, !dbg !61
  %1191 = call float @llvm.amdgcn.exp2.f32(float %1190), !dbg !61
  %1192 = insertelement <4 x float> %1188, float %1191, i64 3, !dbg !63
  %math.result321 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1192) #9, !dbg !63, !frisk.register_lifetimes !34
  %1193 = fptrunc <4 x float> %math.result321 to <4 x half>, !dbg !63
  %math.packed322 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1193) #9, !dbg !62, !frisk.register_lifetimes !34
  %1194 = fadd float %1189, %1191, !dbg !62
  %1195 = extractelement <4 x float> %1036, i64 0, !dbg !61
  %1196 = call float @llvm.amdgcn.exp2.f32(float %1195), !dbg !61
  %1197 = insertelement <4 x float> poison, float %1196, i64 0, !dbg !63
  %1198 = extractelement <4 x float> %1036, i64 1, !dbg !61
  %1199 = call float @llvm.amdgcn.exp2.f32(float %1198), !dbg !61
  %1200 = insertelement <4 x float> %1197, float %1199, i64 1, !dbg !63
  %1201 = fadd float %1196, %1199, !dbg !62
  %1202 = extractelement <4 x float> %1036, i64 2, !dbg !61
  %1203 = call float @llvm.amdgcn.exp2.f32(float %1202), !dbg !61
  %1204 = insertelement <4 x float> %1200, float %1203, i64 2, !dbg !63
  %1205 = fadd float %1201, %1203, !dbg !62
  %1206 = extractelement <4 x float> %1036, i64 3, !dbg !61
  %1207 = call float @llvm.amdgcn.exp2.f32(float %1206), !dbg !61
  %1208 = insertelement <4 x float> %1204, float %1207, i64 3, !dbg !63
  %math.result323 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1208) #9, !dbg !63, !frisk.register_lifetimes !34
  %1209 = fptrunc <4 x float> %math.result323 to <4 x half>, !dbg !63
  %math.packed324 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1209) #9, !dbg !62, !frisk.register_lifetimes !34
  %1210 = fadd float %1205, %1207, !dbg !62
  %1211 = extractelement <4 x float> %1037, i64 0, !dbg !61
  %1212 = call float @llvm.amdgcn.exp2.f32(float %1211), !dbg !61
  %1213 = insertelement <4 x float> poison, float %1212, i64 0, !dbg !63
  %1214 = fadd float %1210, %1212, !dbg !62
  %1215 = extractelement <4 x float> %1037, i64 1, !dbg !61
  %1216 = call float @llvm.amdgcn.exp2.f32(float %1215), !dbg !61
  %1217 = insertelement <4 x float> %1213, float %1216, i64 1, !dbg !63
  %1218 = fadd float %1214, %1216, !dbg !62
  %1219 = extractelement <4 x float> %1037, i64 2, !dbg !61
  %1220 = call float @llvm.amdgcn.exp2.f32(float %1219), !dbg !61
  %1221 = insertelement <4 x float> %1217, float %1220, i64 2, !dbg !63
  %1222 = fadd float %1218, %1220, !dbg !62
  %1223 = extractelement <4 x float> %1037, i64 3, !dbg !61
  %1224 = call float @llvm.amdgcn.exp2.f32(float %1223), !dbg !61
  %1225 = insertelement <4 x float> %1221, float %1224, i64 3, !dbg !63
  %math.result325 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1225) #9, !dbg !63, !frisk.register_lifetimes !34
  %1226 = fptrunc <4 x float> %math.result325 to <4 x half>, !dbg !63
  %math.packed326 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1226) #9, !dbg !62, !frisk.register_lifetimes !34
  %1227 = fadd float %1222, %1224, !dbg !62
  %1228 = extractelement <4 x float> %1038, i64 0, !dbg !61
  %1229 = call float @llvm.amdgcn.exp2.f32(float %1228), !dbg !61
  %1230 = insertelement <4 x float> poison, float %1229, i64 0, !dbg !63
  %1231 = fadd float %1227, %1229, !dbg !62
  %1232 = extractelement <4 x float> %1038, i64 1, !dbg !61
  %1233 = call float @llvm.amdgcn.exp2.f32(float %1232), !dbg !61
  %1234 = insertelement <4 x float> %1230, float %1233, i64 1, !dbg !63
  %1235 = fadd float %1231, %1233, !dbg !62
  %1236 = extractelement <4 x float> %1038, i64 2, !dbg !61
  %1237 = call float @llvm.amdgcn.exp2.f32(float %1236), !dbg !61
  %1238 = insertelement <4 x float> %1234, float %1237, i64 2, !dbg !63
  %1239 = fadd float %1235, %1237, !dbg !62
  %1240 = extractelement <4 x float> %1038, i64 3, !dbg !61
  %1241 = call float @llvm.amdgcn.exp2.f32(float %1240), !dbg !61
  %1242 = insertelement <4 x float> %1238, float %1241, i64 3, !dbg !63
  %math.result327 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1242) #9, !dbg !63, !frisk.register_lifetimes !34
  %1243 = fptrunc <4 x float> %math.result327 to <4 x half>, !dbg !63
  %math.packed328 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1243) #9, !dbg !62, !frisk.register_lifetimes !34
  %1244 = fadd float %1239, %1241, !dbg !62
  %1245 = extractelement <4 x float> %1039, i64 0, !dbg !61
  %1246 = call float @llvm.amdgcn.exp2.f32(float %1245), !dbg !61
  %1247 = insertelement <4 x float> poison, float %1246, i64 0, !dbg !63
  %1248 = fadd float %1244, %1246, !dbg !62
  %1249 = extractelement <4 x float> %1039, i64 1, !dbg !61
  %1250 = call float @llvm.amdgcn.exp2.f32(float %1249), !dbg !61
  %1251 = insertelement <4 x float> %1247, float %1250, i64 1, !dbg !63
  %1252 = fadd float %1248, %1250, !dbg !62
  %1253 = extractelement <4 x float> %1039, i64 2, !dbg !61
  %1254 = call float @llvm.amdgcn.exp2.f32(float %1253), !dbg !61
  %1255 = insertelement <4 x float> %1251, float %1254, i64 2, !dbg !63
  %1256 = fadd float %1252, %1254, !dbg !62
  %1257 = extractelement <4 x float> %1039, i64 3, !dbg !61
  %1258 = call float @llvm.amdgcn.exp2.f32(float %1257), !dbg !61
  %1259 = insertelement <4 x float> %1255, float %1258, i64 3, !dbg !63
  %math.result329 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1259) #9, !dbg !63, !frisk.register_lifetimes !34
  %1260 = fptrunc <4 x float> %math.result329 to <4 x half>, !dbg !63
  %math.packed330 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1260) #9, !dbg !62, !frisk.register_lifetimes !34
  %1261 = fadd float %1256, %1258, !dbg !62
  %1262 = extractelement <4 x float> %1040, i64 0, !dbg !61
  %1263 = call float @llvm.amdgcn.exp2.f32(float %1262), !dbg !61
  %1264 = insertelement <4 x float> poison, float %1263, i64 0, !dbg !63
  %1265 = fadd float %1261, %1263, !dbg !62
  %1266 = extractelement <4 x float> %1040, i64 1, !dbg !61
  %1267 = call float @llvm.amdgcn.exp2.f32(float %1266), !dbg !61
  %1268 = insertelement <4 x float> %1264, float %1267, i64 1, !dbg !63
  %1269 = fadd float %1265, %1267, !dbg !62
  %1270 = extractelement <4 x float> %1040, i64 2, !dbg !61
  %1271 = call float @llvm.amdgcn.exp2.f32(float %1270), !dbg !61
  %1272 = insertelement <4 x float> %1268, float %1271, i64 2, !dbg !63
  %1273 = fadd float %1269, %1271, !dbg !62
  %1274 = extractelement <4 x float> %1040, i64 3, !dbg !61
  %1275 = call float @llvm.amdgcn.exp2.f32(float %1274), !dbg !61
  %1276 = insertelement <4 x float> %1272, float %1275, i64 3, !dbg !63
  %math.result331 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1276) #9, !dbg !63, !frisk.register_lifetimes !34
  %1277 = fptrunc <4 x float> %math.result331 to <4 x half>, !dbg !63
  %math.packed332 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1277) #9, !dbg !62, !frisk.register_lifetimes !34
  %1278 = fadd float %1273, %1275, !dbg !62
  %1279 = extractelement <4 x float> %1041, i64 0, !dbg !61
  %1280 = call float @llvm.amdgcn.exp2.f32(float %1279), !dbg !61
  %1281 = insertelement <4 x float> poison, float %1280, i64 0, !dbg !63
  %1282 = fadd float %1278, %1280, !dbg !62
  %1283 = extractelement <4 x float> %1041, i64 1, !dbg !61
  %1284 = call float @llvm.amdgcn.exp2.f32(float %1283), !dbg !61
  %1285 = insertelement <4 x float> %1281, float %1284, i64 1, !dbg !63
  %1286 = fadd float %1282, %1284, !dbg !62
  %1287 = extractelement <4 x float> %1041, i64 2, !dbg !61
  %1288 = call float @llvm.amdgcn.exp2.f32(float %1287), !dbg !61
  %1289 = insertelement <4 x float> %1285, float %1288, i64 2, !dbg !63
  %1290 = fadd float %1286, %1288, !dbg !62
  %1291 = extractelement <4 x float> %1041, i64 3, !dbg !61
  %1292 = call float @llvm.amdgcn.exp2.f32(float %1291), !dbg !61
  %1293 = insertelement <4 x float> %1289, float %1292, i64 3, !dbg !63
  %math.result333 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1293) #9, !dbg !63, !frisk.register_lifetimes !34
  %1294 = fptrunc <4 x float> %math.result333 to <4 x half>, !dbg !63
  %math.packed334 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1294) #9, !dbg !62, !frisk.register_lifetimes !34
  %1295 = fadd float %1290, %1292, !dbg !62
  %1296 = extractelement <4 x float> %1042, i64 0, !dbg !61
  %1297 = call float @llvm.amdgcn.exp2.f32(float %1296), !dbg !61
  %1298 = insertelement <4 x float> poison, float %1297, i64 0, !dbg !63
  %1299 = fadd float %1295, %1297, !dbg !62
  %1300 = extractelement <4 x float> %1042, i64 1, !dbg !61
  %1301 = call float @llvm.amdgcn.exp2.f32(float %1300), !dbg !61
  %1302 = insertelement <4 x float> %1298, float %1301, i64 1, !dbg !63
  %1303 = fadd float %1299, %1301, !dbg !62
  %1304 = extractelement <4 x float> %1042, i64 2, !dbg !61
  %1305 = call float @llvm.amdgcn.exp2.f32(float %1304), !dbg !61
  %1306 = insertelement <4 x float> %1302, float %1305, i64 2, !dbg !63
  %1307 = fadd float %1303, %1305, !dbg !62
  %1308 = extractelement <4 x float> %1042, i64 3, !dbg !61
  %1309 = call float @llvm.amdgcn.exp2.f32(float %1308), !dbg !61
  %1310 = insertelement <4 x float> %1306, float %1309, i64 3, !dbg !63
  %math.result335 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1310) #9, !dbg !63, !frisk.register_lifetimes !34
  %1311 = fptrunc <4 x float> %math.result335 to <4 x half>, !dbg !63
  %math.packed336 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1311) #9, !dbg !62, !frisk.register_lifetimes !34
  %1312 = fadd float %1307, %1309, !dbg !62
  %1313 = extractelement <4 x float> %1043, i64 0, !dbg !61
  %1314 = call float @llvm.amdgcn.exp2.f32(float %1313), !dbg !61
  %1315 = insertelement <4 x float> poison, float %1314, i64 0, !dbg !63
  %1316 = fadd float %1312, %1314, !dbg !62
  %1317 = extractelement <4 x float> %1043, i64 1, !dbg !61
  %1318 = call float @llvm.amdgcn.exp2.f32(float %1317), !dbg !61
  %1319 = insertelement <4 x float> %1315, float %1318, i64 1, !dbg !63
  %1320 = fadd float %1316, %1318, !dbg !62
  %1321 = extractelement <4 x float> %1043, i64 2, !dbg !61
  %1322 = call float @llvm.amdgcn.exp2.f32(float %1321), !dbg !61
  %1323 = insertelement <4 x float> %1319, float %1322, i64 2, !dbg !63
  %1324 = fadd float %1320, %1322, !dbg !62
  %1325 = extractelement <4 x float> %1043, i64 3, !dbg !61
  %1326 = call float @llvm.amdgcn.exp2.f32(float %1325), !dbg !61
  %1327 = insertelement <4 x float> %1323, float %1326, i64 3, !dbg !63
  %math.result337 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1327) #9, !dbg !63, !frisk.register_lifetimes !34
  %1328 = fptrunc <4 x float> %math.result337 to <4 x half>, !dbg !63
  %math.packed338 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1328) #9, !dbg !62, !frisk.register_lifetimes !34
  %1329 = fadd float %1324, %1326, !dbg !62
  %1330 = bitcast float %1194 to i32, !dbg !62
  %1331 = call i32 @llvm.amdgcn.ds.bpermute(i32 %967, i32 %1330), !dbg !62
  %1332 = bitcast i32 %1331 to float, !dbg !62
  %1333 = fadd float %1194, %1332, !dbg !62
  %1334 = bitcast float %1333 to i32, !dbg !62
  %1335 = call i32 @llvm.amdgcn.ds.bpermute(i32 %975, i32 %1334), !dbg !62
  %1336 = bitcast i32 %1335 to float, !dbg !62
  %1337 = fadd float %1333, %1336, !dbg !62
  %1338 = fadd float %1337, %1051, !dbg !62
  %1339 = insertelement <1 x float> poison, float %1338, i64 0, !dbg !39
  %1340 = insertvalue [2 x <1 x float>] poison, <1 x float> %1339, 0, !dbg !39
  %1341 = bitcast float %1329 to i32, !dbg !62
  %1342 = call i32 @llvm.amdgcn.ds.bpermute(i32 %967, i32 %1341), !dbg !62
  %1343 = bitcast i32 %1342 to float, !dbg !62
  %1344 = fadd float %1329, %1343, !dbg !62
  %1345 = bitcast float %1344 to i32, !dbg !62
  %1346 = call i32 @llvm.amdgcn.ds.bpermute(i32 %975, i32 %1345), !dbg !62
  %1347 = bitcast i32 %1346 to float, !dbg !62
  %1348 = fadd float %1344, %1347, !dbg !62
  %1349 = fadd float %1348, %1059, !dbg !62
  %1350 = insertelement <1 x float> poison, float %1349, i64 0, !dbg !39
  %1351 = insertvalue [2 x <1 x float>] %1340, <1 x float> %1350, 1, !dbg !39
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !48
  fence syncscope("workgroup") release, !dbg !48
  call void @llvm.amdgcn.s.barrier(), !dbg !48
  fence syncscope("workgroup") acquire, !dbg !48
  %address.thread864 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !44, !range !67, !frisk.register_lifetimes !34
  %address.local865 = shl nuw nsw i32 %address.thread864, 3, !dbg !44
  %address.local866 = and i32 %address.local865, 120, !dbg !44
  %address.local867 = shl nuw nsw i32 %address.thread864, 5, !dbg !44
  %address.local868 = and i32 %address.local867, 1536, !dbg !44
  %address.local869 = or disjoint i32 %address.local866, %address.local868, !dbg !44
  %address.local870 = zext nneg i32 %address.local869 to i64, !dbg !44
  %address.local871 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local870, !dbg !44
  %1352 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local871, i32 0) #0, !dbg !44
  %1353 = bitcast <4 x i32> %1352 to <8 x half>, !dbg !44
  %address.local872 = zext nneg i32 %address.thread864 to i64, !dbg !40
  %address.local873 = and i64 %address.local872, 60, !dbg !44
  %address.local874 = add nuw nsw i64 %address.local873, %address.local872, !dbg !44
  %address.local875 = shl nuw nsw i64 %address.local874, 3, !dbg !44
  %address.local876 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local875, !dbg !44
  %address.local877 = and i64 %address.local872, 63, !dbg !42
  %address.local878 = lshr i64 %address.local877, 4, !dbg !44
  %address.local879 = mul nuw nsw i64 %address.local878, 384, !dbg !44
  %address.local880 = getelementptr [2 x i8], ptr addrspace(3) %address.local876, i64 %address.local879, !dbg !44
  %address.local881 = lshr i64 %address.local872, 2, !dbg !47
  %address.local882 = mul nsw i64 %address.local881, -32, !dbg !44
  %address.local883 = getelementptr [2 x i8], ptr addrspace(3) %address.local880, i64 %address.local882, !dbg !44
  %address.local884 = getelementptr i8, ptr addrspace(3) %address.local883, i64 256, !dbg !44
  %1354 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local884, i32 0) #0, !dbg !44
  %1355 = bitcast <4 x i32> %1354 to <8 x half>, !dbg !44
  %address.local885 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local875, !dbg !44
  %address.local886 = getelementptr [2 x i8], ptr addrspace(3) %address.local885, i64 %address.local879, !dbg !44
  %address.local887 = getelementptr [2 x i8], ptr addrspace(3) %address.local886, i64 %address.local882, !dbg !44
  %address.local888 = getelementptr i8, ptr addrspace(3) %address.local887, i64 512, !dbg !44
  %1356 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local888, i32 0) #0, !dbg !44
  %1357 = bitcast <4 x i32> %1356 to <8 x half>, !dbg !44
  %address.local889 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local875, !dbg !44
  %address.local890 = getelementptr [2 x i8], ptr addrspace(3) %address.local889, i64 %address.local879, !dbg !44
  %address.local891 = getelementptr [2 x i8], ptr addrspace(3) %address.local890, i64 %address.local882, !dbg !44
  %address.local892 = getelementptr i8, ptr addrspace(3) %address.local891, i64 768, !dbg !44
  %1358 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local892, i32 0) #0, !dbg !44
  %1359 = bitcast <4 x i32> %1358 to <8 x half>, !dbg !44
  %address.local893 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local870, !dbg !44
  %address.local894 = getelementptr i8, ptr addrspace(3) %address.local893, i64 4096, !dbg !44
  %1360 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local894, i32 0) #0, !dbg !44
  %1361 = bitcast <4 x i32> %1360 to <8 x half>, !dbg !44
  %address.local895 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local875, !dbg !44
  %address.local896 = getelementptr [2 x i8], ptr addrspace(3) %address.local895, i64 %address.local879, !dbg !44
  %address.local897 = getelementptr [2 x i8], ptr addrspace(3) %address.local896, i64 %address.local882, !dbg !44
  %address.local898 = getelementptr i8, ptr addrspace(3) %address.local897, i64 4352, !dbg !44
  %1362 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local898, i32 0) #0, !dbg !44
  %1363 = bitcast <4 x i32> %1362 to <8 x half>, !dbg !44
  %address.local899 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local875, !dbg !44
  %address.local900 = getelementptr [2 x i8], ptr addrspace(3) %address.local899, i64 %address.local879, !dbg !44
  %address.local901 = getelementptr [2 x i8], ptr addrspace(3) %address.local900, i64 %address.local882, !dbg !44
  %address.local902 = getelementptr i8, ptr addrspace(3) %address.local901, i64 4608, !dbg !44
  %1364 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local902, i32 0) #0, !dbg !44
  %1365 = bitcast <4 x i32> %1364 to <8 x half>, !dbg !44
  %address.local903 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local875, !dbg !44
  %address.local904 = getelementptr [2 x i8], ptr addrspace(3) %address.local903, i64 %address.local879, !dbg !44
  %address.local905 = getelementptr [2 x i8], ptr addrspace(3) %address.local904, i64 %address.local882, !dbg !44
  %address.local906 = getelementptr i8, ptr addrspace(3) %address.local905, i64 4864, !dbg !44
  %1366 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local906, i32 0) #0, !dbg !44
  %1367 = bitcast <4 x i32> %1366 to <8 x half>, !dbg !44
  %1368 = add nuw nsw i64 %217, %16, !dbg !48
  %1369 = shl i64 %1368, 19, !dbg !48
  %1370 = and i64 %1369, 16252928, !dbg !48
  %1371 = add i64 %1370, %223, !dbg !48
  fence syncscope("workgroup") release, !dbg !48
  call void @llvm.amdgcn.s.barrier(), !dbg !48
  fence syncscope("workgroup") acquire, !dbg !48
  %1372 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %1371, !dbg !48
  %1373 = getelementptr i8, ptr addrspace(1) %1372, i64 64, !dbg !48
  %address.local907 = shl nuw nsw i64 %address.local872, 3, !dbg !40
  %address.local908 = and i64 %address.local907, 1536, !dbg !40
  %address.local909 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local908, !dbg !40
  %address.local910 = ptrtoint ptr addrspace(3) %address.local909 to i64, !dbg !40
  %address.local911 = trunc i64 %address.local910 to i32, !dbg !40
  %1374 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local911), !dbg !48
  %1375 = zext i32 %1374 to i64, !dbg !48
  %1376 = inttoptr i64 %1375 to ptr addrspace(3), !dbg !48
  %address.local912 = shl nuw nsw i64 %address.local872, 5, !dbg !48
  %address.local913 = and i64 %address.local912, 8064, !dbg !48
  %address.local914 = shl nuw nsw i64 %address.local872, 3, !dbg !48
  %address.local915 = and i64 %address.local914, 24, !dbg !48
  %address.local916 = or disjoint i64 %address.local913, %address.local915, !dbg !48
  %address.local917 = getelementptr [2 x i8], ptr addrspace(1) %1373, i64 %address.local916, !dbg !48
  %1377 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !48
  %1378 = ptrtoint ptr addrspace(1) %address.local917 to i64, !dbg !48
  %1379 = sub i64 %1378, %1377, !dbg !48
  %dma.buffer.offset1558 = trunc i64 %1379 to i32, !dbg !48
  %1380 = trunc i64 %1377 to i32, !dbg !48
  %1381 = insertelement <4 x i32> poison, i32 %1380, i64 0, !dbg !48
  %1382 = lshr i64 %1377, 32, !dbg !48
  %1383 = trunc i64 %1382 to i32, !dbg !48
  %1384 = insertelement <4 x i32> %1381, i32 %1383, i64 1, !dbg !48
  %1385 = insertelement <4 x i32> %1384, i32 33554432, i64 2, !dbg !48
  %dma.buffer.rsrc1559 = insertelement <4 x i32> %1385, i32 131072, i64 3, !dbg !48
  %1386 = ptrtoint ptr addrspace(3) %1376 to i32, !dbg !48
  %dma.buffer.lds1560 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1386), !dbg !48
  %1387 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1560, i32 %dma.buffer.offset1558, <4 x i32> %dma.buffer.rsrc1559) #10, !dbg !48
  %address.local918 = or disjoint i64 %address.local872, 256, !dbg !40
  %address.local919 = shl nuw nsw i64 %address.local918, 3, !dbg !40
  %address.local920 = and i64 %address.local919, 3584, !dbg !40
  %address.local921 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local920, !dbg !40
  %address.local922 = ptrtoint ptr addrspace(3) %address.local921 to i64, !dbg !40
  %address.local923 = trunc i64 %address.local922 to i32, !dbg !40
  %1388 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local923), !dbg !48
  %1389 = zext i32 %1388 to i64, !dbg !48
  %1390 = inttoptr i64 %1389 to ptr addrspace(3), !dbg !48
  %address.local924 = shl nuw nsw i64 %address.local918, 5, !dbg !48
  %address.local925 = and i64 %address.local924, 16256, !dbg !48
  %address.local926 = shl nuw nsw i64 %address.local872, 3, !dbg !48
  %address.local927 = and i64 %address.local926, 24, !dbg !48
  %address.local928 = or disjoint i64 %address.local925, %address.local927, !dbg !48
  %address.local929 = getelementptr [2 x i8], ptr addrspace(1) %1373, i64 %address.local928, !dbg !48
  %1391 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !48
  %1392 = ptrtoint ptr addrspace(1) %address.local929 to i64, !dbg !48
  %1393 = sub i64 %1392, %1391, !dbg !48
  %dma.buffer.offset1555 = trunc i64 %1393 to i32, !dbg !48
  %1394 = trunc i64 %1391 to i32, !dbg !48
  %1395 = insertelement <4 x i32> poison, i32 %1394, i64 0, !dbg !48
  %1396 = lshr i64 %1391, 32, !dbg !48
  %1397 = trunc i64 %1396 to i32, !dbg !48
  %1398 = insertelement <4 x i32> %1395, i32 %1397, i64 1, !dbg !48
  %1399 = insertelement <4 x i32> %1398, i32 33554432, i64 2, !dbg !48
  %dma.buffer.rsrc1556 = insertelement <4 x i32> %1399, i32 131072, i64 3, !dbg !48
  %1400 = ptrtoint ptr addrspace(3) %1390 to i32, !dbg !48
  %dma.buffer.lds1557 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1400), !dbg !48
  %1401 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1557, i32 %dma.buffer.offset1555, <4 x i32> %dma.buffer.rsrc1556) #10, !dbg !48
  %1402 = shufflevector <8 x half> %1353, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  %1403 = extractvalue [2 x <8 x float>] %206, 0, !dbg !64
  %1404 = shufflevector <8 x float> %1403, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !64
  %1405 = fmul <4 x float> %1404, %1048, !dbg !64
  %1406 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1402, <4 x float> %1405) #9, !dbg !44
  %1407 = shufflevector <8 x half> %1355, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  %1408 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %1407, <4 x float> %1406) #9, !dbg !44
  %1409 = shufflevector <8 x half> %1357, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  %1410 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %1409, <4 x float> %1408) #9, !dbg !44
  %1411 = shufflevector <8 x half> %1359, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  %1412 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %1411, <4 x float> %1410) #9, !dbg !44
  %1413 = shufflevector <8 x half> %1361, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  %1414 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %1413, <4 x float> %1412) #9, !dbg !44
  %1415 = shufflevector <8 x half> %1363, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  %1416 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %1415, <4 x float> %1414) #9, !dbg !44
  %1417 = shufflevector <8 x half> %1365, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  %1418 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %1417, <4 x float> %1416) #9, !dbg !44
  %1419 = shufflevector <8 x half> %1367, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  %1420 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %1419, <4 x float> %1418) #9, !dbg !44
  %1421 = shufflevector <8 x half> %1353, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  %1422 = shufflevector <8 x float> %1403, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !64
  %1423 = fmul <4 x float> %1422, %1048, !dbg !64
  %1424 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1421, <4 x float> %1423) #9, !dbg !44
  %1425 = shufflevector <8 x half> %1355, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  %1426 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %1425, <4 x float> %1424) #9, !dbg !44
  %1427 = shufflevector <8 x half> %1357, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  %1428 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %1427, <4 x float> %1426) #9, !dbg !44
  %1429 = shufflevector <8 x half> %1359, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  %1430 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %1429, <4 x float> %1428) #9, !dbg !44
  %1431 = shufflevector <8 x half> %1361, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  %1432 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %1431, <4 x float> %1430) #9, !dbg !44
  %1433 = shufflevector <8 x half> %1363, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  %1434 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %1433, <4 x float> %1432) #9, !dbg !44
  %1435 = shufflevector <8 x half> %1365, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  %1436 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %1435, <4 x float> %1434) #9, !dbg !44
  %1437 = shufflevector <8 x half> %1367, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  %1438 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %1437, <4 x float> %1436) #9, !dbg !44
  %1439 = extractvalue [2 x <8 x float>] %206, 1, !dbg !64
  %1440 = shufflevector <8 x float> %1439, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !64
  %1441 = fmul <4 x float> %1440, %1056, !dbg !64
  %1442 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1402, <4 x float> %1441) #9, !dbg !44
  %1443 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1407, <4 x float> %1442) #9, !dbg !44
  %1444 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1409, <4 x float> %1443) #9, !dbg !44
  %1445 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1411, <4 x float> %1444) #9, !dbg !44
  %1446 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %1413, <4 x float> %1445) #9, !dbg !44
  %1447 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %1415, <4 x float> %1446) #9, !dbg !44
  %1448 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %1417, <4 x float> %1447) #9, !dbg !44
  %1449 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %1419, <4 x float> %1448) #9, !dbg !44
  %1450 = shufflevector <8 x float> %1439, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !64
  %1451 = fmul <4 x float> %1450, %1056, !dbg !64
  %1452 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1421, <4 x float> %1451) #9, !dbg !44
  %1453 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1425, <4 x float> %1452) #9, !dbg !44
  %1454 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1427, <4 x float> %1453) #9, !dbg !44
  %1455 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1429, <4 x float> %1454) #9, !dbg !44
  %1456 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %1431, <4 x float> %1455) #9, !dbg !44
  %1457 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %1433, <4 x float> %1456) #9, !dbg !44
  %1458 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %1435, <4 x float> %1457) #9, !dbg !44
  %1459 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %1437, <4 x float> %1458) #9, !dbg !44
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !48
  fence syncscope("workgroup") release, !dbg !48
  call void @llvm.amdgcn.s.barrier(), !dbg !48
  fence syncscope("workgroup") acquire, !dbg !48
  %1460 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local871, i32 0) #0, !dbg !44
  %1461 = bitcast <4 x i32> %1460 to <8 x half>, !dbg !44
  %1462 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local884, i32 0) #0, !dbg !44
  %1463 = bitcast <4 x i32> %1462 to <8 x half>, !dbg !44
  %1464 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local888, i32 0) #0, !dbg !44
  %1465 = bitcast <4 x i32> %1464 to <8 x half>, !dbg !44
  %1466 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local892, i32 0) #0, !dbg !44
  %1467 = bitcast <4 x i32> %1466 to <8 x half>, !dbg !44
  %1468 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local894, i32 0) #0, !dbg !44
  %1469 = bitcast <4 x i32> %1468 to <8 x half>, !dbg !44
  %1470 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local898, i32 0) #0, !dbg !44
  %1471 = bitcast <4 x i32> %1470 to <8 x half>, !dbg !44
  %1472 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local902, i32 0) #0, !dbg !44
  %1473 = bitcast <4 x i32> %1472 to <8 x half>, !dbg !44
  %1474 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local906, i32 0) #0, !dbg !44
  %1475 = bitcast <4 x i32> %1474 to <8 x half>, !dbg !44
  fence syncscope("workgroup") release, !dbg !48
  call void @llvm.amdgcn.s.barrier(), !dbg !48
  fence syncscope("workgroup") acquire, !dbg !48
  %1476 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %1371, !dbg !48
  %1477 = getelementptr i8, ptr addrspace(1) %1476, i64 128, !dbg !48
  %1478 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local911), !dbg !48
  %1479 = zext i32 %1478 to i64, !dbg !48
  %1480 = inttoptr i64 %1479 to ptr addrspace(3), !dbg !48
  %address.local930 = getelementptr [2 x i8], ptr addrspace(1) %1477, i64 %address.local916, !dbg !48
  %1481 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !48
  %1482 = ptrtoint ptr addrspace(1) %address.local930 to i64, !dbg !48
  %1483 = sub i64 %1482, %1481, !dbg !48
  %dma.buffer.offset1552 = trunc i64 %1483 to i32, !dbg !48
  %1484 = trunc i64 %1481 to i32, !dbg !48
  %1485 = insertelement <4 x i32> poison, i32 %1484, i64 0, !dbg !48
  %1486 = lshr i64 %1481, 32, !dbg !48
  %1487 = trunc i64 %1486 to i32, !dbg !48
  %1488 = insertelement <4 x i32> %1485, i32 %1487, i64 1, !dbg !48
  %1489 = insertelement <4 x i32> %1488, i32 33554432, i64 2, !dbg !48
  %dma.buffer.rsrc1553 = insertelement <4 x i32> %1489, i32 131072, i64 3, !dbg !48
  %1490 = ptrtoint ptr addrspace(3) %1480 to i32, !dbg !48
  %dma.buffer.lds1554 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1490), !dbg !48
  %1491 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1554, i32 %dma.buffer.offset1552, <4 x i32> %dma.buffer.rsrc1553) #10, !dbg !48
  %1492 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local923), !dbg !48
  %1493 = zext i32 %1492 to i64, !dbg !48
  %1494 = inttoptr i64 %1493 to ptr addrspace(3), !dbg !48
  %address.local931 = getelementptr [2 x i8], ptr addrspace(1) %1477, i64 %address.local928, !dbg !48
  %1495 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !48
  %1496 = ptrtoint ptr addrspace(1) %address.local931 to i64, !dbg !48
  %1497 = sub i64 %1496, %1495, !dbg !48
  %dma.buffer.offset1549 = trunc i64 %1497 to i32, !dbg !48
  %1498 = trunc i64 %1495 to i32, !dbg !48
  %1499 = insertelement <4 x i32> poison, i32 %1498, i64 0, !dbg !48
  %1500 = lshr i64 %1495, 32, !dbg !48
  %1501 = trunc i64 %1500 to i32, !dbg !48
  %1502 = insertelement <4 x i32> %1499, i32 %1501, i64 1, !dbg !48
  %1503 = insertelement <4 x i32> %1502, i32 33554432, i64 2, !dbg !48
  %dma.buffer.rsrc1550 = insertelement <4 x i32> %1503, i32 131072, i64 3, !dbg !48
  %1504 = ptrtoint ptr addrspace(3) %1494 to i32, !dbg !48
  %dma.buffer.lds1551 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1504), !dbg !48
  %1505 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1551, i32 %dma.buffer.offset1549, <4 x i32> %dma.buffer.rsrc1550) #10, !dbg !48
  %1506 = shufflevector <8 x half> %1461, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  %1507 = extractvalue [2 x <8 x float>] %209, 0, !dbg !64
  %1508 = shufflevector <8 x float> %1507, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !64
  %1509 = fmul <4 x float> %1508, %1048, !dbg !64
  %1510 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1506, <4 x float> %1509) #9, !dbg !44
  %1511 = shufflevector <8 x half> %1463, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  %1512 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %1511, <4 x float> %1510) #9, !dbg !44
  %1513 = shufflevector <8 x half> %1465, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  %1514 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %1513, <4 x float> %1512) #9, !dbg !44
  %1515 = shufflevector <8 x half> %1467, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  %1516 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %1515, <4 x float> %1514) #9, !dbg !44
  %1517 = shufflevector <8 x half> %1469, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  %1518 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %1517, <4 x float> %1516) #9, !dbg !44
  %1519 = shufflevector <8 x half> %1471, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  %1520 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %1519, <4 x float> %1518) #9, !dbg !44
  %1521 = shufflevector <8 x half> %1473, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  %1522 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %1521, <4 x float> %1520) #9, !dbg !44
  %1523 = shufflevector <8 x half> %1475, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  %1524 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %1523, <4 x float> %1522) #9, !dbg !44
  %1525 = shufflevector <8 x half> %1461, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  %1526 = shufflevector <8 x float> %1507, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !64
  %1527 = fmul <4 x float> %1526, %1048, !dbg !64
  %1528 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1525, <4 x float> %1527) #9, !dbg !44
  %1529 = shufflevector <8 x half> %1463, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  %1530 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %1529, <4 x float> %1528) #9, !dbg !44
  %1531 = shufflevector <8 x half> %1465, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  %1532 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %1531, <4 x float> %1530) #9, !dbg !44
  %1533 = shufflevector <8 x half> %1467, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  %1534 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %1533, <4 x float> %1532) #9, !dbg !44
  %1535 = shufflevector <8 x half> %1469, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  %1536 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %1535, <4 x float> %1534) #9, !dbg !44
  %1537 = shufflevector <8 x half> %1471, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  %1538 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %1537, <4 x float> %1536) #9, !dbg !44
  %1539 = shufflevector <8 x half> %1473, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  %1540 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %1539, <4 x float> %1538) #9, !dbg !44
  %1541 = shufflevector <8 x half> %1475, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  %1542 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %1541, <4 x float> %1540) #9, !dbg !44
  %1543 = extractvalue [2 x <8 x float>] %209, 1, !dbg !64
  %1544 = shufflevector <8 x float> %1543, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !64
  %1545 = fmul <4 x float> %1544, %1056, !dbg !64
  %1546 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1506, <4 x float> %1545) #9, !dbg !44
  %1547 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1511, <4 x float> %1546) #9, !dbg !44
  %1548 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1513, <4 x float> %1547) #9, !dbg !44
  %1549 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1515, <4 x float> %1548) #9, !dbg !44
  %1550 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %1517, <4 x float> %1549) #9, !dbg !44
  %1551 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %1519, <4 x float> %1550) #9, !dbg !44
  %1552 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %1521, <4 x float> %1551) #9, !dbg !44
  %1553 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %1523, <4 x float> %1552) #9, !dbg !44
  %1554 = shufflevector <8 x float> %1543, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !64
  %1555 = fmul <4 x float> %1554, %1056, !dbg !64
  %1556 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1525, <4 x float> %1555) #9, !dbg !44
  %1557 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1529, <4 x float> %1556) #9, !dbg !44
  %1558 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1531, <4 x float> %1557) #9, !dbg !44
  %1559 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1533, <4 x float> %1558) #9, !dbg !44
  %1560 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %1535, <4 x float> %1559) #9, !dbg !44
  %1561 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %1537, <4 x float> %1560) #9, !dbg !44
  %1562 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %1539, <4 x float> %1561) #9, !dbg !44
  %1563 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %1541, <4 x float> %1562) #9, !dbg !44
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !48
  fence syncscope("workgroup") release, !dbg !48
  call void @llvm.amdgcn.s.barrier(), !dbg !48
  fence syncscope("workgroup") acquire, !dbg !48
  %1564 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local871, i32 0) #0, !dbg !44
  %1565 = bitcast <4 x i32> %1564 to <8 x half>, !dbg !44
  %1566 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local884, i32 0) #0, !dbg !44
  %1567 = bitcast <4 x i32> %1566 to <8 x half>, !dbg !44
  %1568 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local888, i32 0) #0, !dbg !44
  %1569 = bitcast <4 x i32> %1568 to <8 x half>, !dbg !44
  %1570 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local892, i32 0) #0, !dbg !44
  %1571 = bitcast <4 x i32> %1570 to <8 x half>, !dbg !44
  %1572 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local894, i32 0) #0, !dbg !44
  %1573 = bitcast <4 x i32> %1572 to <8 x half>, !dbg !44
  %1574 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local898, i32 0) #0, !dbg !44
  %1575 = bitcast <4 x i32> %1574 to <8 x half>, !dbg !44
  %1576 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local902, i32 0) #0, !dbg !44
  %1577 = bitcast <4 x i32> %1576 to <8 x half>, !dbg !44
  %1578 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local906, i32 0) #0, !dbg !44
  %1579 = bitcast <4 x i32> %1578 to <8 x half>, !dbg !44
  fence syncscope("workgroup") release, !dbg !48
  call void @llvm.amdgcn.s.barrier(), !dbg !48
  fence syncscope("workgroup") acquire, !dbg !48
  %1580 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %1371, !dbg !48
  %1581 = getelementptr i8, ptr addrspace(1) %1580, i64 192, !dbg !48
  %1582 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local911), !dbg !48
  %1583 = zext i32 %1582 to i64, !dbg !48
  %1584 = inttoptr i64 %1583 to ptr addrspace(3), !dbg !48
  %address.local932 = getelementptr [2 x i8], ptr addrspace(1) %1581, i64 %address.local916, !dbg !48
  %1585 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !48
  %1586 = ptrtoint ptr addrspace(1) %address.local932 to i64, !dbg !48
  %1587 = sub i64 %1586, %1585, !dbg !48
  %dma.buffer.offset1546 = trunc i64 %1587 to i32, !dbg !48
  %1588 = trunc i64 %1585 to i32, !dbg !48
  %1589 = insertelement <4 x i32> poison, i32 %1588, i64 0, !dbg !48
  %1590 = lshr i64 %1585, 32, !dbg !48
  %1591 = trunc i64 %1590 to i32, !dbg !48
  %1592 = insertelement <4 x i32> %1589, i32 %1591, i64 1, !dbg !48
  %1593 = insertelement <4 x i32> %1592, i32 33554432, i64 2, !dbg !48
  %dma.buffer.rsrc1547 = insertelement <4 x i32> %1593, i32 131072, i64 3, !dbg !48
  %1594 = ptrtoint ptr addrspace(3) %1584 to i32, !dbg !48
  %dma.buffer.lds1548 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1594), !dbg !48
  %1595 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1548, i32 %dma.buffer.offset1546, <4 x i32> %dma.buffer.rsrc1547) #10, !dbg !48
  %1596 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local923), !dbg !48
  %1597 = zext i32 %1596 to i64, !dbg !48
  %1598 = inttoptr i64 %1597 to ptr addrspace(3), !dbg !48
  %address.local933 = getelementptr [2 x i8], ptr addrspace(1) %1581, i64 %address.local928, !dbg !48
  %1599 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !48
  %1600 = ptrtoint ptr addrspace(1) %address.local933 to i64, !dbg !48
  %1601 = sub i64 %1600, %1599, !dbg !48
  %dma.buffer.offset = trunc i64 %1601 to i32, !dbg !48
  %1602 = trunc i64 %1599 to i32, !dbg !48
  %1603 = insertelement <4 x i32> poison, i32 %1602, i64 0, !dbg !48
  %1604 = lshr i64 %1599, 32, !dbg !48
  %1605 = trunc i64 %1604 to i32, !dbg !48
  %1606 = insertelement <4 x i32> %1603, i32 %1605, i64 1, !dbg !48
  %1607 = insertelement <4 x i32> %1606, i32 33554432, i64 2, !dbg !48
  %dma.buffer.rsrc = insertelement <4 x i32> %1607, i32 131072, i64 3, !dbg !48
  %1608 = ptrtoint ptr addrspace(3) %1598 to i32, !dbg !48
  %dma.buffer.lds = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1608), !dbg !48
  %1609 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc) #10, !dbg !48
  %1610 = shufflevector <8 x half> %1565, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  %1611 = extractvalue [2 x <8 x float>] %210, 0, !dbg !64
  %1612 = shufflevector <8 x float> %1611, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !64
  %1613 = fmul <4 x float> %1612, %1048, !dbg !64
  %1614 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1610, <4 x float> %1613) #9, !dbg !44
  %1615 = shufflevector <8 x half> %1567, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  %1616 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %1615, <4 x float> %1614) #9, !dbg !44
  %1617 = shufflevector <8 x half> %1569, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  %1618 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %1617, <4 x float> %1616) #9, !dbg !44
  %1619 = shufflevector <8 x half> %1571, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  %1620 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %1619, <4 x float> %1618) #9, !dbg !44
  %1621 = shufflevector <8 x half> %1573, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  %1622 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %1621, <4 x float> %1620) #9, !dbg !44
  %1623 = shufflevector <8 x half> %1575, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  %1624 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %1623, <4 x float> %1622) #9, !dbg !44
  %1625 = shufflevector <8 x half> %1577, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  %1626 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %1625, <4 x float> %1624) #9, !dbg !44
  %1627 = shufflevector <8 x half> %1579, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  %1628 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %1627, <4 x float> %1626) #9, !dbg !44
  %1629 = shufflevector <8 x half> %1565, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  %1630 = shufflevector <8 x float> %1611, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !64
  %1631 = fmul <4 x float> %1630, %1048, !dbg !64
  %1632 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1629, <4 x float> %1631) #9, !dbg !44
  %1633 = shufflevector <8 x half> %1567, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  %1634 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %1633, <4 x float> %1632) #9, !dbg !44
  %1635 = shufflevector <8 x half> %1569, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  %1636 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %1635, <4 x float> %1634) #9, !dbg !44
  %1637 = shufflevector <8 x half> %1571, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  %1638 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %1637, <4 x float> %1636) #9, !dbg !44
  %1639 = shufflevector <8 x half> %1573, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  %1640 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %1639, <4 x float> %1638) #9, !dbg !44
  %1641 = shufflevector <8 x half> %1575, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  %1642 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %1641, <4 x float> %1640) #9, !dbg !44
  %1643 = shufflevector <8 x half> %1577, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  %1644 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %1643, <4 x float> %1642) #9, !dbg !44
  %1645 = shufflevector <8 x half> %1579, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  %1646 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %1645, <4 x float> %1644) #9, !dbg !44
  %1647 = extractvalue [2 x <8 x float>] %210, 1, !dbg !64
  %1648 = shufflevector <8 x float> %1647, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !64
  %1649 = fmul <4 x float> %1648, %1056, !dbg !64
  %1650 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1610, <4 x float> %1649) #9, !dbg !44
  %1651 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1615, <4 x float> %1650) #9, !dbg !44
  %1652 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1617, <4 x float> %1651) #9, !dbg !44
  %1653 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1619, <4 x float> %1652) #9, !dbg !44
  %1654 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %1621, <4 x float> %1653) #9, !dbg !44
  %1655 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %1623, <4 x float> %1654) #9, !dbg !44
  %1656 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %1625, <4 x float> %1655) #9, !dbg !44
  %1657 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %1627, <4 x float> %1656) #9, !dbg !44
  %1658 = shufflevector <8 x float> %1647, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !64
  %1659 = fmul <4 x float> %1658, %1056, !dbg !64
  %1660 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1629, <4 x float> %1659) #9, !dbg !44
  %1661 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1633, <4 x float> %1660) #9, !dbg !44
  %1662 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1635, <4 x float> %1661) #9, !dbg !44
  %1663 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1637, <4 x float> %1662) #9, !dbg !44
  %1664 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %1639, <4 x float> %1663) #9, !dbg !44
  %1665 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %1641, <4 x float> %1664) #9, !dbg !44
  %1666 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %1643, <4 x float> %1665) #9, !dbg !44
  %1667 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %1645, <4 x float> %1666) #9, !dbg !44
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !48
  fence syncscope("workgroup") release, !dbg !48
  call void @llvm.amdgcn.s.barrier(), !dbg !48
  fence syncscope("workgroup") acquire, !dbg !48
  %1668 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local871, i32 0) #0, !dbg !44
  %1669 = bitcast <4 x i32> %1668 to <8 x half>, !dbg !44
  %1670 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local884, i32 0) #0, !dbg !44
  %1671 = bitcast <4 x i32> %1670 to <8 x half>, !dbg !44
  %1672 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local888, i32 0) #0, !dbg !44
  %1673 = bitcast <4 x i32> %1672 to <8 x half>, !dbg !44
  %1674 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local892, i32 0) #0, !dbg !44
  %1675 = bitcast <4 x i32> %1674 to <8 x half>, !dbg !44
  %1676 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local894, i32 0) #0, !dbg !44
  %1677 = bitcast <4 x i32> %1676 to <8 x half>, !dbg !44
  %1678 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local898, i32 0) #0, !dbg !44
  %1679 = bitcast <4 x i32> %1678 to <8 x half>, !dbg !44
  %1680 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local902, i32 0) #0, !dbg !44
  %1681 = bitcast <4 x i32> %1680 to <8 x half>, !dbg !44
  %1682 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local906, i32 0) #0, !dbg !44
  %1683 = bitcast <4 x i32> %1682 to <8 x half>, !dbg !44
  %1684 = shufflevector <8 x half> %1669, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  %1685 = extractvalue [2 x <8 x float>] %211, 0, !dbg !64
  %1686 = shufflevector <8 x float> %1685, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !64
  %1687 = fmul <4 x float> %1686, %1048, !dbg !64
  %1688 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1684, <4 x float> %1687) #9, !dbg !44
  %1689 = shufflevector <8 x half> %1671, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  %1690 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %1689, <4 x float> %1688) #9, !dbg !44
  %1691 = shufflevector <8 x half> %1673, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  %1692 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %1691, <4 x float> %1690) #9, !dbg !44
  %1693 = shufflevector <8 x half> %1675, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  %1694 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %1693, <4 x float> %1692) #9, !dbg !44
  %1695 = shufflevector <8 x half> %1677, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  %1696 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %1695, <4 x float> %1694) #9, !dbg !44
  %1697 = shufflevector <8 x half> %1679, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  %1698 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %1697, <4 x float> %1696) #9, !dbg !44
  %1699 = shufflevector <8 x half> %1681, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  %1700 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %1699, <4 x float> %1698) #9, !dbg !44
  %1701 = shufflevector <8 x half> %1683, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  %1702 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %1701, <4 x float> %1700) #9, !dbg !44
  %1703 = shufflevector <8 x half> %1669, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  %1704 = shufflevector <8 x float> %1685, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !64
  %1705 = fmul <4 x float> %1704, %1048, !dbg !64
  %1706 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1703, <4 x float> %1705) #9, !dbg !44
  %1707 = shufflevector <8 x half> %1671, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  %1708 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %1707, <4 x float> %1706) #9, !dbg !44
  %1709 = shufflevector <8 x half> %1673, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  %1710 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %1709, <4 x float> %1708) #9, !dbg !44
  %1711 = shufflevector <8 x half> %1675, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  %1712 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %1711, <4 x float> %1710) #9, !dbg !44
  %1713 = shufflevector <8 x half> %1677, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  %1714 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %1713, <4 x float> %1712) #9, !dbg !44
  %1715 = shufflevector <8 x half> %1679, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  %1716 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %1715, <4 x float> %1714) #9, !dbg !44
  %1717 = shufflevector <8 x half> %1681, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  %1718 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %1717, <4 x float> %1716) #9, !dbg !44
  %1719 = shufflevector <8 x half> %1683, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  %1720 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %1719, <4 x float> %1718) #9, !dbg !44
  %1721 = extractvalue [2 x <8 x float>] %211, 1, !dbg !64
  %1722 = shufflevector <8 x float> %1721, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !64
  %1723 = fmul <4 x float> %1722, %1056, !dbg !64
  %1724 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1684, <4 x float> %1723) #9, !dbg !44
  %1725 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1689, <4 x float> %1724) #9, !dbg !44
  %1726 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1691, <4 x float> %1725) #9, !dbg !44
  %1727 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1693, <4 x float> %1726) #9, !dbg !44
  %1728 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %1695, <4 x float> %1727) #9, !dbg !44
  %1729 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %1697, <4 x float> %1728) #9, !dbg !44
  %1730 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %1699, <4 x float> %1729) #9, !dbg !44
  %1731 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %1701, <4 x float> %1730) #9, !dbg !44
  %1732 = shufflevector <8 x float> %1721, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !64
  %1733 = fmul <4 x float> %1732, %1056, !dbg !64
  %1734 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1703, <4 x float> %1733) #9, !dbg !44
  %1735 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1707, <4 x float> %1734) #9, !dbg !44
  %1736 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1709, <4 x float> %1735) #9, !dbg !44
  %1737 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1711, <4 x float> %1736) #9, !dbg !44
  %1738 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %1713, <4 x float> %1737) #9, !dbg !44
  %1739 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %1715, <4 x float> %1738) #9, !dbg !44
  %1740 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %1717, <4 x float> %1739) #9, !dbg !44
  %1741 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %1719, <4 x float> %1740) #9, !dbg !44
  %1742 = shufflevector <4 x float> %1420, <4 x float> %1438, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !39
  %1743 = insertvalue [2 x <8 x float>] poison, <8 x float> %1742, 0, !dbg !39
  %1744 = shufflevector <4 x float> %1449, <4 x float> %1459, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !39
  %1745 = insertvalue [2 x <8 x float>] %1743, <8 x float> %1744, 1, !dbg !39
  %1746 = insertvalue [2 x <1 x float>] poison, <1 x float> %982, 0, !dbg !39
  %1747 = insertvalue [2 x <1 x float>] %1746, <1 x float> %1025, 1, !dbg !39
  %1748 = shufflevector <4 x float> %1524, <4 x float> %1542, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !39
  %1749 = insertvalue [2 x <8 x float>] poison, <8 x float> %1748, 0, !dbg !39
  %1750 = shufflevector <4 x float> %1553, <4 x float> %1563, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !39
  %1751 = insertvalue [2 x <8 x float>] %1749, <8 x float> %1750, 1, !dbg !39
  %1752 = shufflevector <4 x float> %1628, <4 x float> %1646, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !39
  %1753 = insertvalue [2 x <8 x float>] poison, <8 x float> %1752, 0, !dbg !39
  %1754 = shufflevector <4 x float> %1657, <4 x float> %1667, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !39
  %1755 = insertvalue [2 x <8 x float>] %1753, <8 x float> %1754, 1, !dbg !39
  %1756 = shufflevector <4 x float> %1702, <4 x float> %1720, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !39
  %1757 = insertvalue [2 x <8 x float>] poison, <8 x float> %1756, 0, !dbg !39
  %1758 = shufflevector <4 x float> %1731, <4 x float> %1741, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !39
  %1759 = insertvalue [2 x <8 x float>] %1757, <8 x float> %1758, 1, !dbg !39
  fence syncscope("workgroup") release, !dbg !39
  call void @llvm.amdgcn.s.barrier(), !dbg !39
  fence syncscope("workgroup") acquire, !dbg !39
  %1760 = add nuw nsw i64 %205, 1, !dbg !39
  br label %204, !dbg !39

1761:                                             ; preds = %204
  %1762 = and i32 %31, 16, !dbg !39
  %.not = icmp eq i32 %1762, 0, !dbg !39
  %1763 = and i32 %31, 32, !dbg !39
  %.not36 = icmp eq i32 %1763, 0, !dbg !39
  %1764 = extractvalue [2 x <8 x float>] %211, 0, !dbg !39
  %1765 = extractelement <8 x float> %1764, i64 0, !dbg !39
  %1766 = extractelement <8 x float> %1764, i64 1, !dbg !39
  %1767 = extractelement <8 x float> %1764, i64 2, !dbg !39
  %1768 = extractelement <8 x float> %1764, i64 3, !dbg !39
  %1769 = select i1 %.not, float %1766, float %1765, !dbg !39
  %1770 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !39
  %1771 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1770), !dbg !39
  %1772 = add nuw nsw i32 %1771, 64, !dbg !39
  %1773 = xor i32 %1771, 16, !dbg !39
  %1774 = and i32 %1772, 192, !dbg !39
  %1775 = icmp samesign ult i32 %1773, %1774, !dbg !39
  %1776 = select i1 %1775, i32 %1773, i32 %1771, !dbg !39
  %1777 = shl nuw nsw i32 %1776, 2, !dbg !39
  %1778 = bitcast float %1769 to i32, !dbg !39
  %1779 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1777, i32 %1778), !dbg !39
  %1780 = bitcast i32 %1779 to float, !dbg !39
  %1781 = select i1 %.not, float %1765, float %1780, !dbg !39
  %1782 = select i1 %.not, float %1780, float %1766, !dbg !39
  %1783 = select i1 %.not, float %1768, float %1767, !dbg !39
  %1784 = bitcast float %1783 to i32, !dbg !39
  %1785 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1777, i32 %1784), !dbg !39
  %1786 = bitcast i32 %1785 to float, !dbg !39
  %1787 = select i1 %.not, float %1767, float %1786, !dbg !39
  %1788 = select i1 %.not, float %1786, float %1768, !dbg !39
  %1789 = select i1 %.not36, float %1787, float %1781, !dbg !39
  %1790 = xor i32 %1771, 32, !dbg !39
  %1791 = icmp samesign ult i32 %1790, %1774, !dbg !39
  %1792 = select i1 %1791, i32 %1790, i32 %1771, !dbg !39
  %1793 = shl nuw nsw i32 %1792, 2, !dbg !39
  %1794 = bitcast float %1789 to i32, !dbg !39
  %1795 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1793, i32 %1794), !dbg !39
  %1796 = bitcast i32 %1795 to float, !dbg !39
  %1797 = select i1 %.not36, float %1781, float %1796, !dbg !39
  %1798 = select i1 %.not36, float %1796, float %1787, !dbg !39
  %1799 = select i1 %.not36, float %1788, float %1782, !dbg !39
  %1800 = bitcast float %1799 to i32, !dbg !39
  %1801 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1793, i32 %1800), !dbg !39
  %1802 = bitcast i32 %1801 to float, !dbg !39
  %1803 = select i1 %.not36, float %1782, float %1802, !dbg !39
  %1804 = select i1 %.not36, float %1802, float %1788, !dbg !39
  %1805 = extractvalue [2 x <8 x float>] %210, 0, !dbg !39
  %1806 = extractelement <8 x float> %1805, i64 0, !dbg !39
  %1807 = extractelement <8 x float> %1805, i64 1, !dbg !39
  %1808 = extractelement <8 x float> %1805, i64 2, !dbg !39
  %1809 = extractelement <8 x float> %1805, i64 3, !dbg !39
  %1810 = select i1 %.not, float %1807, float %1806, !dbg !39
  %1811 = bitcast float %1810 to i32, !dbg !39
  %1812 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1777, i32 %1811), !dbg !39
  %1813 = bitcast i32 %1812 to float, !dbg !39
  %1814 = select i1 %.not, float %1806, float %1813, !dbg !39
  %1815 = select i1 %.not, float %1813, float %1807, !dbg !39
  %1816 = select i1 %.not, float %1809, float %1808, !dbg !39
  %1817 = bitcast float %1816 to i32, !dbg !39
  %1818 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1777, i32 %1817), !dbg !39
  %1819 = bitcast i32 %1818 to float, !dbg !39
  %1820 = select i1 %.not, float %1808, float %1819, !dbg !39
  %1821 = select i1 %.not, float %1819, float %1809, !dbg !39
  %1822 = select i1 %.not36, float %1820, float %1814, !dbg !39
  %1823 = bitcast float %1822 to i32, !dbg !39
  %1824 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1793, i32 %1823), !dbg !39
  %1825 = bitcast i32 %1824 to float, !dbg !39
  %1826 = select i1 %.not36, float %1814, float %1825, !dbg !39
  %1827 = select i1 %.not36, float %1825, float %1820, !dbg !39
  %1828 = select i1 %.not36, float %1821, float %1815, !dbg !39
  %1829 = bitcast float %1828 to i32, !dbg !39
  %1830 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1793, i32 %1829), !dbg !39
  %1831 = bitcast i32 %1830 to float, !dbg !39
  %1832 = select i1 %.not36, float %1815, float %1831, !dbg !39
  %1833 = select i1 %.not36, float %1831, float %1821, !dbg !39
  %1834 = extractvalue [2 x <8 x float>] %209, 0, !dbg !39
  %1835 = extractelement <8 x float> %1834, i64 0, !dbg !39
  %1836 = extractelement <8 x float> %1834, i64 1, !dbg !39
  %1837 = extractelement <8 x float> %1834, i64 2, !dbg !39
  %1838 = extractelement <8 x float> %1834, i64 3, !dbg !39
  %1839 = select i1 %.not, float %1836, float %1835, !dbg !39
  %1840 = bitcast float %1839 to i32, !dbg !39
  %1841 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1777, i32 %1840), !dbg !39
  %1842 = bitcast i32 %1841 to float, !dbg !39
  %1843 = select i1 %.not, float %1835, float %1842, !dbg !39
  %1844 = select i1 %.not, float %1842, float %1836, !dbg !39
  %1845 = select i1 %.not, float %1838, float %1837, !dbg !39
  %1846 = bitcast float %1845 to i32, !dbg !39
  %1847 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1777, i32 %1846), !dbg !39
  %1848 = bitcast i32 %1847 to float, !dbg !39
  %1849 = select i1 %.not, float %1837, float %1848, !dbg !39
  %1850 = select i1 %.not, float %1848, float %1838, !dbg !39
  %1851 = select i1 %.not36, float %1849, float %1843, !dbg !39
  %1852 = bitcast float %1851 to i32, !dbg !39
  %1853 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1793, i32 %1852), !dbg !39
  %1854 = bitcast i32 %1853 to float, !dbg !39
  %1855 = select i1 %.not36, float %1843, float %1854, !dbg !39
  %1856 = select i1 %.not36, float %1854, float %1849, !dbg !39
  %1857 = select i1 %.not36, float %1850, float %1844, !dbg !39
  %1858 = bitcast float %1857 to i32, !dbg !39
  %1859 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1793, i32 %1858), !dbg !39
  %1860 = bitcast i32 %1859 to float, !dbg !39
  %1861 = select i1 %.not36, float %1844, float %1860, !dbg !39
  %1862 = select i1 %.not36, float %1860, float %1850, !dbg !39
  %1863 = extractvalue [2 x <8 x float>] %206, 0, !dbg !39
  %1864 = extractelement <8 x float> %1863, i64 0, !dbg !39
  %1865 = extractelement <8 x float> %1863, i64 1, !dbg !39
  %1866 = extractelement <8 x float> %1863, i64 2, !dbg !39
  %1867 = extractelement <8 x float> %1863, i64 3, !dbg !39
  %1868 = select i1 %.not, float %1865, float %1864, !dbg !39
  %1869 = bitcast float %1868 to i32, !dbg !39
  %1870 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1777, i32 %1869), !dbg !39
  %1871 = bitcast i32 %1870 to float, !dbg !39
  %1872 = select i1 %.not, float %1864, float %1871, !dbg !39
  %1873 = select i1 %.not, float %1871, float %1865, !dbg !39
  %1874 = select i1 %.not, float %1867, float %1866, !dbg !39
  %1875 = bitcast float %1874 to i32, !dbg !39
  %1876 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1777, i32 %1875), !dbg !39
  %1877 = bitcast i32 %1876 to float, !dbg !39
  %1878 = select i1 %.not, float %1866, float %1877, !dbg !39
  %1879 = select i1 %.not, float %1877, float %1867, !dbg !39
  %1880 = select i1 %.not36, float %1878, float %1872, !dbg !39
  %1881 = bitcast float %1880 to i32, !dbg !39
  %1882 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1793, i32 %1881), !dbg !39
  %1883 = select i1 %.not36, float %1879, float %1873, !dbg !39
  %1884 = bitcast float %1883 to i32, !dbg !39
  %1885 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1793, i32 %1884), !dbg !39
  %1886 = extractvalue [2 x <1 x float>] %207, 0, !dbg !49
  %1887 = shufflevector <1 x float> %1886, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !49
  %1888 = extractelement <8 x float> %1764, i64 4, !dbg !39
  %1889 = extractelement <8 x float> %1764, i64 5, !dbg !39
  %1890 = extractelement <8 x float> %1764, i64 6, !dbg !39
  %1891 = extractelement <8 x float> %1764, i64 7, !dbg !39
  %1892 = select i1 %.not, float %1889, float %1888, !dbg !39
  %1893 = bitcast float %1892 to i32, !dbg !39
  %1894 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1777, i32 %1893), !dbg !39
  %1895 = bitcast i32 %1894 to float, !dbg !39
  %1896 = select i1 %.not, float %1888, float %1895, !dbg !39
  %1897 = select i1 %.not, float %1895, float %1889, !dbg !39
  %1898 = select i1 %.not, float %1891, float %1890, !dbg !39
  %1899 = bitcast float %1898 to i32, !dbg !39
  %1900 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1777, i32 %1899), !dbg !39
  %1901 = bitcast i32 %1900 to float, !dbg !39
  %1902 = select i1 %.not, float %1890, float %1901, !dbg !39
  %1903 = select i1 %.not, float %1901, float %1891, !dbg !39
  %1904 = select i1 %.not36, float %1902, float %1896, !dbg !39
  %1905 = bitcast float %1904 to i32, !dbg !39
  %1906 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1793, i32 %1905), !dbg !39
  %1907 = bitcast i32 %1906 to float, !dbg !39
  %1908 = select i1 %.not36, float %1896, float %1907, !dbg !39
  %1909 = select i1 %.not36, float %1907, float %1902, !dbg !39
  %1910 = select i1 %.not36, float %1903, float %1897, !dbg !39
  %1911 = bitcast float %1910 to i32, !dbg !39
  %1912 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1793, i32 %1911), !dbg !39
  %1913 = bitcast i32 %1912 to float, !dbg !39
  %1914 = select i1 %.not36, float %1897, float %1913, !dbg !39
  %1915 = select i1 %.not36, float %1913, float %1903, !dbg !39
  %1916 = extractelement <8 x float> %1805, i64 4, !dbg !39
  %1917 = extractelement <8 x float> %1805, i64 5, !dbg !39
  %1918 = extractelement <8 x float> %1805, i64 6, !dbg !39
  %1919 = extractelement <8 x float> %1805, i64 7, !dbg !39
  %1920 = select i1 %.not, float %1917, float %1916, !dbg !39
  %1921 = bitcast float %1920 to i32, !dbg !39
  %1922 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1777, i32 %1921), !dbg !39
  %1923 = bitcast i32 %1922 to float, !dbg !39
  %1924 = select i1 %.not, float %1916, float %1923, !dbg !39
  %1925 = select i1 %.not, float %1923, float %1917, !dbg !39
  %1926 = select i1 %.not, float %1919, float %1918, !dbg !39
  %1927 = bitcast float %1926 to i32, !dbg !39
  %1928 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1777, i32 %1927), !dbg !39
  %1929 = bitcast i32 %1928 to float, !dbg !39
  %1930 = select i1 %.not, float %1918, float %1929, !dbg !39
  %1931 = select i1 %.not, float %1929, float %1919, !dbg !39
  %1932 = select i1 %.not36, float %1930, float %1924, !dbg !39
  %1933 = bitcast float %1932 to i32, !dbg !39
  %1934 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1793, i32 %1933), !dbg !39
  %1935 = bitcast i32 %1934 to float, !dbg !39
  %1936 = select i1 %.not36, float %1924, float %1935, !dbg !39
  %1937 = select i1 %.not36, float %1935, float %1930, !dbg !39
  %1938 = select i1 %.not36, float %1931, float %1925, !dbg !39
  %1939 = bitcast float %1938 to i32, !dbg !39
  %1940 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1793, i32 %1939), !dbg !39
  %1941 = bitcast i32 %1940 to float, !dbg !39
  %1942 = select i1 %.not36, float %1925, float %1941, !dbg !39
  %1943 = select i1 %.not36, float %1941, float %1931, !dbg !39
  %1944 = extractelement <8 x float> %1834, i64 4, !dbg !39
  %1945 = extractelement <8 x float> %1834, i64 5, !dbg !39
  %1946 = extractelement <8 x float> %1834, i64 6, !dbg !39
  %1947 = extractelement <8 x float> %1834, i64 7, !dbg !39
  %1948 = select i1 %.not, float %1945, float %1944, !dbg !39
  %1949 = bitcast float %1948 to i32, !dbg !39
  %1950 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1777, i32 %1949), !dbg !39
  %1951 = bitcast i32 %1950 to float, !dbg !39
  %1952 = select i1 %.not, float %1944, float %1951, !dbg !39
  %1953 = select i1 %.not, float %1951, float %1945, !dbg !39
  %1954 = select i1 %.not, float %1947, float %1946, !dbg !39
  %1955 = bitcast float %1954 to i32, !dbg !39
  %1956 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1777, i32 %1955), !dbg !39
  %1957 = bitcast i32 %1956 to float, !dbg !39
  %1958 = select i1 %.not, float %1946, float %1957, !dbg !39
  %1959 = select i1 %.not, float %1957, float %1947, !dbg !39
  %1960 = select i1 %.not36, float %1958, float %1952, !dbg !39
  %1961 = bitcast float %1960 to i32, !dbg !39
  %1962 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1793, i32 %1961), !dbg !39
  %1963 = bitcast i32 %1962 to float, !dbg !39
  %1964 = select i1 %.not36, float %1952, float %1963, !dbg !39
  %1965 = select i1 %.not36, float %1963, float %1958, !dbg !39
  %1966 = select i1 %.not36, float %1959, float %1953, !dbg !39
  %1967 = bitcast float %1966 to i32, !dbg !39
  %1968 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1793, i32 %1967), !dbg !39
  %1969 = bitcast i32 %1968 to float, !dbg !39
  %1970 = select i1 %.not36, float %1953, float %1969, !dbg !39
  %1971 = select i1 %.not36, float %1969, float %1959, !dbg !39
  %1972 = extractelement <8 x float> %1863, i64 4, !dbg !39
  %1973 = extractelement <8 x float> %1863, i64 5, !dbg !39
  %1974 = extractelement <8 x float> %1863, i64 6, !dbg !39
  %1975 = extractelement <8 x float> %1863, i64 7, !dbg !39
  %1976 = select i1 %.not, float %1973, float %1972, !dbg !39
  %1977 = bitcast float %1976 to i32, !dbg !39
  %1978 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1777, i32 %1977), !dbg !39
  %1979 = bitcast i32 %1978 to float, !dbg !39
  %1980 = select i1 %.not, float %1972, float %1979, !dbg !39
  %1981 = select i1 %.not, float %1979, float %1973, !dbg !39
  %1982 = select i1 %.not, float %1975, float %1974, !dbg !39
  %1983 = bitcast float %1982 to i32, !dbg !39
  %1984 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1777, i32 %1983), !dbg !39
  %1985 = bitcast i32 %1984 to float, !dbg !39
  %1986 = select i1 %.not, float %1974, float %1985, !dbg !39
  %1987 = select i1 %.not, float %1985, float %1975, !dbg !39
  %1988 = select i1 %.not36, float %1986, float %1980, !dbg !39
  %1989 = bitcast float %1988 to i32, !dbg !39
  %1990 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1793, i32 %1989), !dbg !39
  %1991 = select i1 %.not36, float %1987, float %1981, !dbg !39
  %1992 = bitcast float %1991 to i32, !dbg !39
  %1993 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1793, i32 %1992), !dbg !39
  %1994 = extractvalue [2 x <8 x float>] %211, 1, !dbg !39
  %1995 = extractelement <8 x float> %1994, i64 0, !dbg !39
  %1996 = extractelement <8 x float> %1994, i64 1, !dbg !39
  %1997 = extractelement <8 x float> %1994, i64 2, !dbg !39
  %1998 = extractelement <8 x float> %1994, i64 3, !dbg !39
  %1999 = select i1 %.not, float %1996, float %1995, !dbg !39
  %2000 = bitcast float %1999 to i32, !dbg !39
  %2001 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1777, i32 %2000), !dbg !39
  %2002 = bitcast i32 %2001 to float, !dbg !39
  %2003 = select i1 %.not, float %1995, float %2002, !dbg !39
  %2004 = select i1 %.not, float %2002, float %1996, !dbg !39
  %2005 = select i1 %.not, float %1998, float %1997, !dbg !39
  %2006 = bitcast float %2005 to i32, !dbg !39
  %2007 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1777, i32 %2006), !dbg !39
  %2008 = bitcast i32 %2007 to float, !dbg !39
  %2009 = select i1 %.not, float %1997, float %2008, !dbg !39
  %2010 = select i1 %.not, float %2008, float %1998, !dbg !39
  %2011 = select i1 %.not36, float %2009, float %2003, !dbg !39
  %2012 = bitcast float %2011 to i32, !dbg !39
  %2013 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1793, i32 %2012), !dbg !39
  %2014 = bitcast i32 %2013 to float, !dbg !39
  %2015 = select i1 %.not36, float %2003, float %2014, !dbg !39
  %2016 = select i1 %.not36, float %2014, float %2009, !dbg !39
  %2017 = select i1 %.not36, float %2010, float %2004, !dbg !39
  %2018 = bitcast float %2017 to i32, !dbg !39
  %2019 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1793, i32 %2018), !dbg !39
  %2020 = bitcast i32 %2019 to float, !dbg !39
  %2021 = select i1 %.not36, float %2004, float %2020, !dbg !39
  %2022 = select i1 %.not36, float %2020, float %2010, !dbg !39
  %2023 = extractvalue [2 x <8 x float>] %210, 1, !dbg !39
  %2024 = extractelement <8 x float> %2023, i64 0, !dbg !39
  %2025 = extractelement <8 x float> %2023, i64 1, !dbg !39
  %2026 = extractelement <8 x float> %2023, i64 2, !dbg !39
  %2027 = extractelement <8 x float> %2023, i64 3, !dbg !39
  %2028 = select i1 %.not, float %2025, float %2024, !dbg !39
  %2029 = bitcast float %2028 to i32, !dbg !39
  %2030 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1777, i32 %2029), !dbg !39
  %2031 = bitcast i32 %2030 to float, !dbg !39
  %2032 = select i1 %.not, float %2024, float %2031, !dbg !39
  %2033 = select i1 %.not, float %2031, float %2025, !dbg !39
  %2034 = select i1 %.not, float %2027, float %2026, !dbg !39
  %2035 = bitcast float %2034 to i32, !dbg !39
  %2036 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1777, i32 %2035), !dbg !39
  %2037 = bitcast i32 %2036 to float, !dbg !39
  %2038 = select i1 %.not, float %2026, float %2037, !dbg !39
  %2039 = select i1 %.not, float %2037, float %2027, !dbg !39
  %2040 = select i1 %.not36, float %2038, float %2032, !dbg !39
  %2041 = bitcast float %2040 to i32, !dbg !39
  %2042 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1793, i32 %2041), !dbg !39
  %2043 = bitcast i32 %2042 to float, !dbg !39
  %2044 = select i1 %.not36, float %2032, float %2043, !dbg !39
  %2045 = select i1 %.not36, float %2043, float %2038, !dbg !39
  %2046 = select i1 %.not36, float %2039, float %2033, !dbg !39
  %2047 = bitcast float %2046 to i32, !dbg !39
  %2048 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1793, i32 %2047), !dbg !39
  %2049 = bitcast i32 %2048 to float, !dbg !39
  %2050 = select i1 %.not36, float %2033, float %2049, !dbg !39
  %2051 = select i1 %.not36, float %2049, float %2039, !dbg !39
  %2052 = extractvalue [2 x <8 x float>] %209, 1, !dbg !39
  %2053 = extractelement <8 x float> %2052, i64 0, !dbg !39
  %2054 = extractelement <8 x float> %2052, i64 1, !dbg !39
  %2055 = extractelement <8 x float> %2052, i64 2, !dbg !39
  %2056 = extractelement <8 x float> %2052, i64 3, !dbg !39
  %2057 = select i1 %.not, float %2054, float %2053, !dbg !39
  %2058 = bitcast float %2057 to i32, !dbg !39
  %2059 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1777, i32 %2058), !dbg !39
  %2060 = bitcast i32 %2059 to float, !dbg !39
  %2061 = select i1 %.not, float %2053, float %2060, !dbg !39
  %2062 = select i1 %.not, float %2060, float %2054, !dbg !39
  %2063 = select i1 %.not, float %2056, float %2055, !dbg !39
  %2064 = bitcast float %2063 to i32, !dbg !39
  %2065 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1777, i32 %2064), !dbg !39
  %2066 = bitcast i32 %2065 to float, !dbg !39
  %2067 = select i1 %.not, float %2055, float %2066, !dbg !39
  %2068 = select i1 %.not, float %2066, float %2056, !dbg !39
  %2069 = select i1 %.not36, float %2067, float %2061, !dbg !39
  %2070 = bitcast float %2069 to i32, !dbg !39
  %2071 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1793, i32 %2070), !dbg !39
  %2072 = bitcast i32 %2071 to float, !dbg !39
  %2073 = select i1 %.not36, float %2061, float %2072, !dbg !39
  %2074 = select i1 %.not36, float %2072, float %2067, !dbg !39
  %2075 = select i1 %.not36, float %2068, float %2062, !dbg !39
  %2076 = bitcast float %2075 to i32, !dbg !39
  %2077 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1793, i32 %2076), !dbg !39
  %2078 = bitcast i32 %2077 to float, !dbg !39
  %2079 = select i1 %.not36, float %2062, float %2078, !dbg !39
  %2080 = select i1 %.not36, float %2078, float %2068, !dbg !39
  %2081 = extractvalue [2 x <8 x float>] %206, 1, !dbg !39
  %2082 = extractelement <8 x float> %2081, i64 0, !dbg !39
  %2083 = extractelement <8 x float> %2081, i64 1, !dbg !39
  %2084 = extractelement <8 x float> %2081, i64 2, !dbg !39
  %2085 = extractelement <8 x float> %2081, i64 3, !dbg !39
  %2086 = select i1 %.not, float %2083, float %2082, !dbg !39
  %2087 = bitcast float %2086 to i32, !dbg !39
  %2088 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1777, i32 %2087), !dbg !39
  %2089 = bitcast i32 %2088 to float, !dbg !39
  %2090 = select i1 %.not, float %2082, float %2089, !dbg !39
  %2091 = select i1 %.not, float %2089, float %2083, !dbg !39
  %2092 = select i1 %.not, float %2085, float %2084, !dbg !39
  %2093 = bitcast float %2092 to i32, !dbg !39
  %2094 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1777, i32 %2093), !dbg !39
  %2095 = bitcast i32 %2094 to float, !dbg !39
  %2096 = select i1 %.not, float %2084, float %2095, !dbg !39
  %2097 = select i1 %.not, float %2095, float %2085, !dbg !39
  %2098 = select i1 %.not36, float %2096, float %2090, !dbg !39
  %2099 = bitcast float %2098 to i32, !dbg !39
  %2100 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1793, i32 %2099), !dbg !39
  %2101 = select i1 %.not36, float %2097, float %2091, !dbg !39
  %2102 = bitcast float %2101 to i32, !dbg !39
  %2103 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1793, i32 %2102), !dbg !39
  %2104 = extractvalue [2 x <1 x float>] %207, 1, !dbg !49
  %2105 = shufflevector <1 x float> %2104, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !49
  %2106 = extractelement <8 x float> %1994, i64 4, !dbg !39
  %2107 = extractelement <8 x float> %1994, i64 5, !dbg !39
  %2108 = extractelement <8 x float> %1994, i64 6, !dbg !39
  %2109 = extractelement <8 x float> %1994, i64 7, !dbg !39
  %2110 = select i1 %.not, float %2107, float %2106, !dbg !39
  %2111 = bitcast float %2110 to i32, !dbg !39
  %2112 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1777, i32 %2111), !dbg !39
  %2113 = bitcast i32 %2112 to float, !dbg !39
  %2114 = select i1 %.not, float %2106, float %2113, !dbg !39
  %2115 = select i1 %.not, float %2113, float %2107, !dbg !39
  %2116 = select i1 %.not, float %2109, float %2108, !dbg !39
  %2117 = bitcast float %2116 to i32, !dbg !39
  %2118 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1777, i32 %2117), !dbg !39
  %2119 = bitcast i32 %2118 to float, !dbg !39
  %2120 = select i1 %.not, float %2108, float %2119, !dbg !39
  %2121 = select i1 %.not, float %2119, float %2109, !dbg !39
  %2122 = select i1 %.not36, float %2120, float %2114, !dbg !39
  %2123 = bitcast float %2122 to i32, !dbg !39
  %2124 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1793, i32 %2123), !dbg !39
  %2125 = bitcast i32 %2124 to float, !dbg !39
  %2126 = select i1 %.not36, float %2114, float %2125, !dbg !39
  %2127 = select i1 %.not36, float %2125, float %2120, !dbg !39
  %2128 = select i1 %.not36, float %2121, float %2115, !dbg !39
  %2129 = bitcast float %2128 to i32, !dbg !39
  %2130 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1793, i32 %2129), !dbg !39
  %2131 = bitcast i32 %2130 to float, !dbg !39
  %2132 = select i1 %.not36, float %2115, float %2131, !dbg !39
  %2133 = select i1 %.not36, float %2131, float %2121, !dbg !39
  %2134 = extractelement <8 x float> %2023, i64 4, !dbg !39
  %2135 = extractelement <8 x float> %2023, i64 5, !dbg !39
  %2136 = extractelement <8 x float> %2023, i64 6, !dbg !39
  %2137 = extractelement <8 x float> %2023, i64 7, !dbg !39
  %2138 = select i1 %.not, float %2135, float %2134, !dbg !39
  %2139 = bitcast float %2138 to i32, !dbg !39
  %2140 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1777, i32 %2139), !dbg !39
  %2141 = bitcast i32 %2140 to float, !dbg !39
  %2142 = select i1 %.not, float %2134, float %2141, !dbg !39
  %2143 = select i1 %.not, float %2141, float %2135, !dbg !39
  %2144 = select i1 %.not, float %2137, float %2136, !dbg !39
  %2145 = bitcast float %2144 to i32, !dbg !39
  %2146 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1777, i32 %2145), !dbg !39
  %2147 = bitcast i32 %2146 to float, !dbg !39
  %2148 = select i1 %.not, float %2136, float %2147, !dbg !39
  %2149 = select i1 %.not, float %2147, float %2137, !dbg !39
  %2150 = select i1 %.not36, float %2148, float %2142, !dbg !39
  %2151 = bitcast float %2150 to i32, !dbg !39
  %2152 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1793, i32 %2151), !dbg !39
  %2153 = bitcast i32 %2152 to float, !dbg !39
  %2154 = select i1 %.not36, float %2142, float %2153, !dbg !39
  %2155 = select i1 %.not36, float %2153, float %2148, !dbg !39
  %2156 = select i1 %.not36, float %2149, float %2143, !dbg !39
  %2157 = bitcast float %2156 to i32, !dbg !39
  %2158 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1793, i32 %2157), !dbg !39
  %2159 = bitcast i32 %2158 to float, !dbg !39
  %2160 = select i1 %.not36, float %2143, float %2159, !dbg !39
  %2161 = select i1 %.not36, float %2159, float %2149, !dbg !39
  %2162 = extractelement <8 x float> %2052, i64 4, !dbg !39
  %2163 = extractelement <8 x float> %2052, i64 5, !dbg !39
  %2164 = extractelement <8 x float> %2052, i64 6, !dbg !39
  %2165 = extractelement <8 x float> %2052, i64 7, !dbg !39
  %2166 = select i1 %.not, float %2163, float %2162, !dbg !39
  %2167 = bitcast float %2166 to i32, !dbg !39
  %2168 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1777, i32 %2167), !dbg !39
  %2169 = bitcast i32 %2168 to float, !dbg !39
  %2170 = select i1 %.not, float %2162, float %2169, !dbg !39
  %2171 = select i1 %.not, float %2169, float %2163, !dbg !39
  %2172 = select i1 %.not, float %2165, float %2164, !dbg !39
  %2173 = bitcast float %2172 to i32, !dbg !39
  %2174 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1777, i32 %2173), !dbg !39
  %2175 = bitcast i32 %2174 to float, !dbg !39
  %2176 = select i1 %.not, float %2164, float %2175, !dbg !39
  %2177 = select i1 %.not, float %2175, float %2165, !dbg !39
  %2178 = select i1 %.not36, float %2176, float %2170, !dbg !39
  %2179 = bitcast float %2178 to i32, !dbg !39
  %2180 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1793, i32 %2179), !dbg !39
  %2181 = bitcast i32 %2180 to float, !dbg !39
  %2182 = select i1 %.not36, float %2170, float %2181, !dbg !39
  %2183 = select i1 %.not36, float %2181, float %2176, !dbg !39
  %2184 = select i1 %.not36, float %2177, float %2171, !dbg !39
  %2185 = bitcast float %2184 to i32, !dbg !39
  %2186 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1793, i32 %2185), !dbg !39
  %2187 = bitcast i32 %2186 to float, !dbg !39
  %2188 = select i1 %.not36, float %2171, float %2187, !dbg !39
  %2189 = select i1 %.not36, float %2187, float %2177, !dbg !39
  %2190 = extractelement <8 x float> %2081, i64 4, !dbg !39
  %2191 = extractelement <8 x float> %2081, i64 5, !dbg !39
  %2192 = extractelement <8 x float> %2081, i64 6, !dbg !39
  %2193 = extractelement <8 x float> %2081, i64 7, !dbg !39
  %2194 = select i1 %.not, float %2191, float %2190, !dbg !39
  %2195 = bitcast float %2194 to i32, !dbg !39
  %2196 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1777, i32 %2195), !dbg !39
  %2197 = bitcast i32 %2196 to float, !dbg !39
  %2198 = select i1 %.not, float %2190, float %2197, !dbg !39
  %2199 = select i1 %.not, float %2197, float %2191, !dbg !39
  %2200 = select i1 %.not, float %2193, float %2192, !dbg !39
  %2201 = bitcast float %2200 to i32, !dbg !39
  %2202 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1777, i32 %2201), !dbg !39
  %2203 = bitcast i32 %2202 to float, !dbg !39
  %2204 = select i1 %.not, float %2192, float %2203, !dbg !39
  %2205 = select i1 %.not, float %2203, float %2193, !dbg !39
  %2206 = select i1 %.not36, float %2204, float %2198, !dbg !39
  %2207 = bitcast float %2206 to i32, !dbg !39
  %2208 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1793, i32 %2207), !dbg !39
  %2209 = select i1 %.not36, float %2205, float %2199, !dbg !39
  %2210 = bitcast float %2209 to i32, !dbg !39
  %2211 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1793, i32 %2210), !dbg !39
  br i1 true, label %2212, label %2276, !dbg !51

2212:                                             ; preds = %1761
  %2213 = bitcast i32 %2208 to float, !dbg !39
  %2214 = select i1 %.not36, float %2198, float %2213, !dbg !39
  %2215 = insertelement <4 x float> poison, float %2214, i64 0, !dbg !49
  %2216 = bitcast i32 %2211 to float, !dbg !39
  %2217 = select i1 %.not36, float %2199, float %2216, !dbg !39
  %2218 = insertelement <4 x float> %2215, float %2217, i64 1, !dbg !49
  %2219 = select i1 %.not36, float %2213, float %2204, !dbg !39
  %2220 = insertelement <4 x float> %2218, float %2219, i64 2, !dbg !49
  %2221 = select i1 %.not36, float %2216, float %2205, !dbg !39
  %2222 = insertelement <4 x float> %2220, float %2221, i64 3, !dbg !49
  %row.reciprocal = fdiv arcp <4 x float> splat (float 1.000000e+00), %2105, !dbg !49
  %normalized = fmul <4 x float> %2222, %row.reciprocal, !dbg !49
  %2223 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !50
  %2224 = extractelement <4 x half> %2223, i64 3, !dbg !50
  %2225 = extractelement <4 x half> %2223, i64 2, !dbg !50
  %2226 = extractelement <4 x half> %2223, i64 1, !dbg !50
  %2227 = extractelement <4 x half> %2223, i64 0, !dbg !50
  %2228 = bitcast i32 %2100 to float, !dbg !39
  %2229 = select i1 %.not36, float %2090, float %2228, !dbg !39
  %2230 = insertelement <4 x float> poison, float %2229, i64 0, !dbg !49
  %2231 = bitcast i32 %2103 to float, !dbg !39
  %2232 = select i1 %.not36, float %2091, float %2231, !dbg !39
  %2233 = insertelement <4 x float> %2230, float %2232, i64 1, !dbg !49
  %2234 = select i1 %.not36, float %2228, float %2096, !dbg !39
  %2235 = insertelement <4 x float> %2233, float %2234, i64 2, !dbg !49
  %2236 = select i1 %.not36, float %2231, float %2097, !dbg !39
  %2237 = insertelement <4 x float> %2235, float %2236, i64 3, !dbg !49
  %normalized104 = fmul <4 x float> %2237, %row.reciprocal, !dbg !49
  %2238 = fptrunc <4 x float> %normalized104 to <4 x half>, !dbg !50
  %2239 = extractelement <4 x half> %2238, i64 3, !dbg !50
  %2240 = extractelement <4 x half> %2238, i64 2, !dbg !50
  %2241 = extractelement <4 x half> %2238, i64 1, !dbg !50
  %2242 = extractelement <4 x half> %2238, i64 0, !dbg !50
  %2243 = bitcast i32 %1990 to float, !dbg !39
  %2244 = select i1 %.not36, float %1980, float %2243, !dbg !39
  %2245 = insertelement <4 x float> poison, float %2244, i64 0, !dbg !49
  %2246 = bitcast i32 %1993 to float, !dbg !39
  %2247 = select i1 %.not36, float %1981, float %2246, !dbg !39
  %2248 = insertelement <4 x float> %2245, float %2247, i64 1, !dbg !49
  %2249 = select i1 %.not36, float %2243, float %1986, !dbg !39
  %2250 = insertelement <4 x float> %2248, float %2249, i64 2, !dbg !49
  %2251 = select i1 %.not36, float %2246, float %1987, !dbg !39
  %2252 = insertelement <4 x float> %2250, float %2251, i64 3, !dbg !49
  %row.reciprocal105 = fdiv arcp <4 x float> splat (float 1.000000e+00), %1887, !dbg !49
  %normalized106 = fmul <4 x float> %2252, %row.reciprocal105, !dbg !49
  %2253 = fptrunc <4 x float> %normalized106 to <4 x half>, !dbg !50
  %2254 = extractelement <4 x half> %2253, i64 3, !dbg !50
  %2255 = extractelement <4 x half> %2253, i64 2, !dbg !50
  %2256 = extractelement <4 x half> %2253, i64 1, !dbg !50
  %2257 = extractelement <4 x half> %2253, i64 0, !dbg !50
  %2258 = bitcast i32 %1882 to float, !dbg !39
  %2259 = select i1 %.not36, float %1872, float %2258, !dbg !39
  %2260 = insertelement <4 x float> poison, float %2259, i64 0, !dbg !49
  %2261 = bitcast i32 %1885 to float, !dbg !39
  %2262 = select i1 %.not36, float %1873, float %2261, !dbg !39
  %2263 = insertelement <4 x float> %2260, float %2262, i64 1, !dbg !49
  %2264 = select i1 %.not36, float %2258, float %1878, !dbg !39
  %2265 = insertelement <4 x float> %2263, float %2264, i64 2, !dbg !49
  %2266 = select i1 %.not36, float %2261, float %1879, !dbg !39
  %2267 = insertelement <4 x float> %2265, float %2266, i64 3, !dbg !49
  %normalized107 = fmul <4 x float> %2267, %row.reciprocal105, !dbg !49
  %2268 = fptrunc <4 x float> %normalized107 to <4 x half>, !dbg !50
  %2269 = extractelement <4 x half> %2268, i64 3, !dbg !50
  %2270 = extractelement <4 x half> %2268, i64 2, !dbg !50
  %2271 = extractelement <4 x half> %2268, i64 1, !dbg !50
  %2272 = extractelement <4 x half> %2268, i64 0, !dbg !50
  %2273 = add nsw i64 %16, %22, !dbg !51
  %2274 = shl i64 %2273, 19, !dbg !51
  %2275 = and i64 %2274, 16252928, !dbg !51
  %address.thread934 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !51, !range !67, !frisk.register_lifetimes !34
  %address.local935 = zext nneg i32 %address.thread934 to i64, !dbg !40
  %address.local936 = add i64 %25, %address.local935, !dbg !51
  %address.local937 = lshr i64 %address.local935, 6, !dbg !42
  %address.local938 = shl nuw nsw i64 %address.local937, 4, !dbg !51
  %address.local939 = add i64 %address.local936, %address.local938, !dbg !51
  %address.local940 = and i64 %address.local935, 240, !dbg !51
  %address.local941 = sub i64 %address.local939, %address.local940, !dbg !51
  %address.local942 = add i64 %address.local941, %26, !dbg !51
  %address.local943 = shl nuw nsw i64 %address.local942, 7, !dbg !51
  %address.local944 = add nuw nsw i64 %2275, %address.local943, !dbg !51
  %address.local945 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local944, !dbg !51
  %address.local946 = lshr i64 %address.local935, 2, !dbg !42
  %address.local947 = and i64 %address.local946, 12, !dbg !42
  %address.local948 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local947, !dbg !51
  store half %2272, ptr addrspace(1) %address.local948, align 2, !dbg !51
  %address.local949 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local944, !dbg !51
  %address.local950 = or disjoint i64 %address.local947, 1, !dbg !51
  %address.local951 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local949, i64 %address.local950, !dbg !51
  store half %2271, ptr addrspace(1) %address.local951, align 2, !dbg !51
  %address.local952 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local944, !dbg !51
  %address.local953 = or disjoint i64 %address.local947, 2, !dbg !51
  %address.local954 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local952, i64 %address.local953, !dbg !51
  store half %2270, ptr addrspace(1) %address.local954, align 2, !dbg !51
  %address.local955 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local944, !dbg !51
  %address.local956 = or disjoint i64 %address.local947, 3, !dbg !51
  %address.local957 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local955, i64 %address.local956, !dbg !51
  store half %2269, ptr addrspace(1) %address.local957, align 2, !dbg !51
  %address.local958 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local944, !dbg !51
  %address.local959 = or disjoint i64 %address.local947, 16, !dbg !51
  %address.local960 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local958, i64 %address.local959, !dbg !51
  store half %2257, ptr addrspace(1) %address.local960, align 2, !dbg !51
  %address.local961 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local944, !dbg !51
  %address.local962 = or disjoint i64 %address.local947, 17, !dbg !51
  %address.local963 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local961, i64 %address.local962, !dbg !51
  store half %2256, ptr addrspace(1) %address.local963, align 2, !dbg !51
  %address.local964 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local944, !dbg !51
  %address.local965 = or disjoint i64 %address.local947, 18, !dbg !51
  %address.local966 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local964, i64 %address.local965, !dbg !51
  store half %2255, ptr addrspace(1) %address.local966, align 2, !dbg !51
  %address.local967 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local944, !dbg !51
  %address.local968 = or disjoint i64 %address.local947, 19, !dbg !51
  %address.local969 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local967, i64 %address.local968, !dbg !51
  store half %2254, ptr addrspace(1) %address.local969, align 2, !dbg !51
  %address.local970 = shl i64 %address.local942, 7, !dbg !51
  %address.local971 = add i64 %address.local970, 8192, !dbg !51
  %address.local972 = add nuw nsw i64 %2275, %address.local971, !dbg !51
  %address.local973 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local972, !dbg !51
  %address.local974 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local973, i64 %address.local947, !dbg !51
  store half %2242, ptr addrspace(1) %address.local974, align 2, !dbg !51
  %address.local975 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local972, !dbg !51
  %address.local976 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local975, i64 %address.local950, !dbg !51
  store half %2241, ptr addrspace(1) %address.local976, align 2, !dbg !51
  %address.local977 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local972, !dbg !51
  %address.local978 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local977, i64 %address.local953, !dbg !51
  store half %2240, ptr addrspace(1) %address.local978, align 2, !dbg !51
  %address.local979 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local972, !dbg !51
  %address.local980 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local979, i64 %address.local956, !dbg !51
  store half %2239, ptr addrspace(1) %address.local980, align 2, !dbg !51
  %address.local981 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local972, !dbg !51
  %address.local982 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local981, i64 %address.local959, !dbg !51
  store half %2227, ptr addrspace(1) %address.local982, align 2, !dbg !51
  %address.local983 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local972, !dbg !51
  %address.local984 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local983, i64 %address.local962, !dbg !51
  store half %2226, ptr addrspace(1) %address.local984, align 2, !dbg !51
  %address.local985 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local972, !dbg !51
  %address.local986 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local965, !dbg !51
  store half %2225, ptr addrspace(1) %address.local986, align 2, !dbg !51
  %address.local987 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local972, !dbg !51
  %address.local988 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local987, i64 %address.local968, !dbg !51
  store half %2224, ptr addrspace(1) %address.local988, align 2, !dbg !51
  br label %2276, !dbg !51

2276:                                             ; preds = %2212, %1761
  br i1 true, label %2277, label %2315, !dbg !51

2277:                                             ; preds = %2276
  %2278 = insertelement <4 x float> poison, float %2182, i64 0, !dbg !49
  %2279 = insertelement <4 x float> %2278, float %2188, i64 1, !dbg !49
  %2280 = insertelement <4 x float> %2279, float %2183, i64 2, !dbg !49
  %2281 = insertelement <4 x float> %2280, float %2189, i64 3, !dbg !49
  %row.reciprocal108 = fdiv arcp <4 x float> splat (float 1.000000e+00), %2105, !dbg !49
  %normalized109 = fmul <4 x float> %2281, %row.reciprocal108, !dbg !49
  %2282 = fptrunc <4 x float> %normalized109 to <4 x half>, !dbg !50
  %2283 = extractelement <4 x half> %2282, i64 3, !dbg !50
  %2284 = extractelement <4 x half> %2282, i64 2, !dbg !50
  %2285 = extractelement <4 x half> %2282, i64 1, !dbg !50
  %2286 = extractelement <4 x half> %2282, i64 0, !dbg !50
  %2287 = insertelement <4 x float> poison, float %2073, i64 0, !dbg !49
  %2288 = insertelement <4 x float> %2287, float %2079, i64 1, !dbg !49
  %2289 = insertelement <4 x float> %2288, float %2074, i64 2, !dbg !49
  %2290 = insertelement <4 x float> %2289, float %2080, i64 3, !dbg !49
  %normalized110 = fmul <4 x float> %2290, %row.reciprocal108, !dbg !49
  %2291 = fptrunc <4 x float> %normalized110 to <4 x half>, !dbg !50
  %2292 = extractelement <4 x half> %2291, i64 3, !dbg !50
  %2293 = extractelement <4 x half> %2291, i64 2, !dbg !50
  %2294 = extractelement <4 x half> %2291, i64 1, !dbg !50
  %2295 = extractelement <4 x half> %2291, i64 0, !dbg !50
  %2296 = insertelement <4 x float> poison, float %1964, i64 0, !dbg !49
  %2297 = insertelement <4 x float> %2296, float %1970, i64 1, !dbg !49
  %2298 = insertelement <4 x float> %2297, float %1965, i64 2, !dbg !49
  %2299 = insertelement <4 x float> %2298, float %1971, i64 3, !dbg !49
  %row.reciprocal111 = fdiv arcp <4 x float> splat (float 1.000000e+00), %1887, !dbg !49
  %normalized112 = fmul <4 x float> %2299, %row.reciprocal111, !dbg !49
  %2300 = fptrunc <4 x float> %normalized112 to <4 x half>, !dbg !50
  %2301 = extractelement <4 x half> %2300, i64 3, !dbg !50
  %2302 = extractelement <4 x half> %2300, i64 2, !dbg !50
  %2303 = extractelement <4 x half> %2300, i64 1, !dbg !50
  %2304 = extractelement <4 x half> %2300, i64 0, !dbg !50
  %2305 = insertelement <4 x float> poison, float %1855, i64 0, !dbg !49
  %2306 = insertelement <4 x float> %2305, float %1861, i64 1, !dbg !49
  %2307 = insertelement <4 x float> %2306, float %1856, i64 2, !dbg !49
  %2308 = insertelement <4 x float> %2307, float %1862, i64 3, !dbg !49
  %normalized113 = fmul <4 x float> %2308, %row.reciprocal111, !dbg !49
  %2309 = fptrunc <4 x float> %normalized113 to <4 x half>, !dbg !50
  %2310 = extractelement <4 x half> %2309, i64 3, !dbg !50
  %2311 = extractelement <4 x half> %2309, i64 2, !dbg !50
  %2312 = extractelement <4 x half> %2309, i64 1, !dbg !50
  %2313 = extractelement <4 x half> %2309, i64 0, !dbg !50
  %2314 = shl nuw nsw i64 %24, 19, !dbg !51
  %address.thread989 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !51, !range !67, !frisk.register_lifetimes !34
  %address.local990 = zext nneg i32 %address.thread989 to i64, !dbg !40
  %address.local991 = add i64 %25, %address.local990, !dbg !51
  %address.local992 = lshr i64 %address.local990, 6, !dbg !42
  %address.local993 = shl nuw nsw i64 %address.local992, 4, !dbg !51
  %address.local994 = add i64 %address.local991, %address.local993, !dbg !51
  %address.local995 = and i64 %address.local990, 240, !dbg !51
  %address.local996 = sub i64 %address.local994, %address.local995, !dbg !51
  %address.local997 = add i64 %address.local996, %26, !dbg !51
  %address.local998 = shl nuw nsw i64 %address.local997, 7, !dbg !51
  %address.local999 = add nuw nsw i64 %2314, %address.local998, !dbg !51
  %address.local1000 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local999, !dbg !51
  %address.local1001 = lshr i64 %address.local990, 2, !dbg !42
  %address.local1002 = and i64 %address.local1001, 12, !dbg !42
  %address.local1003 = or disjoint i64 %address.local1002, 32, !dbg !51
  %address.local1004 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1000, i64 %address.local1003, !dbg !51
  store half %2313, ptr addrspace(1) %address.local1004, align 2, !dbg !51
  %address.local1005 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local999, !dbg !51
  %address.local1006 = or disjoint i64 %address.local1002, 33, !dbg !51
  %address.local1007 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1005, i64 %address.local1006, !dbg !51
  store half %2312, ptr addrspace(1) %address.local1007, align 2, !dbg !51
  %address.local1008 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local999, !dbg !51
  %address.local1009 = or disjoint i64 %address.local1002, 34, !dbg !51
  %address.local1010 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1008, i64 %address.local1009, !dbg !51
  store half %2311, ptr addrspace(1) %address.local1010, align 2, !dbg !51
  %address.local1011 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local999, !dbg !51
  %address.local1012 = or disjoint i64 %address.local1002, 35, !dbg !51
  %address.local1013 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1011, i64 %address.local1012, !dbg !51
  store half %2310, ptr addrspace(1) %address.local1013, align 2, !dbg !51
  %address.local1014 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local999, !dbg !51
  %address.local1015 = or disjoint i64 %address.local1002, 48, !dbg !51
  %address.local1016 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1014, i64 %address.local1015, !dbg !51
  store half %2304, ptr addrspace(1) %address.local1016, align 2, !dbg !51
  %address.local1017 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local999, !dbg !51
  %address.local1018 = or disjoint i64 %address.local1002, 49, !dbg !51
  %address.local1019 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1017, i64 %address.local1018, !dbg !51
  store half %2303, ptr addrspace(1) %address.local1019, align 2, !dbg !51
  %address.local1020 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local999, !dbg !51
  %address.local1021 = or disjoint i64 %address.local1002, 50, !dbg !51
  %address.local1022 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1020, i64 %address.local1021, !dbg !51
  store half %2302, ptr addrspace(1) %address.local1022, align 2, !dbg !51
  %address.local1023 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local999, !dbg !51
  %address.local1024 = or i64 %address.local1001, 51, !dbg !51
  %address.local1025 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1023, i64 %address.local1024, !dbg !51
  store half %2301, ptr addrspace(1) %address.local1025, align 2, !dbg !51
  %address.local1026 = shl i64 %address.local997, 7, !dbg !51
  %address.local1027 = add i64 %address.local1026, 8192, !dbg !51
  %address.local1028 = add nuw nsw i64 %2314, %address.local1027, !dbg !51
  %address.local1029 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1028, !dbg !51
  %address.local1030 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1029, i64 %address.local1003, !dbg !51
  store half %2295, ptr addrspace(1) %address.local1030, align 2, !dbg !51
  %address.local1031 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1028, !dbg !51
  %address.local1032 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1031, i64 %address.local1006, !dbg !51
  store half %2294, ptr addrspace(1) %address.local1032, align 2, !dbg !51
  %address.local1033 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1028, !dbg !51
  %address.local1034 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1033, i64 %address.local1009, !dbg !51
  store half %2293, ptr addrspace(1) %address.local1034, align 2, !dbg !51
  %address.local1035 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1028, !dbg !51
  %address.local1036 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1035, i64 %address.local1012, !dbg !51
  store half %2292, ptr addrspace(1) %address.local1036, align 2, !dbg !51
  %address.local1037 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1028, !dbg !51
  %address.local1038 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1037, i64 %address.local1015, !dbg !51
  store half %2286, ptr addrspace(1) %address.local1038, align 2, !dbg !51
  %address.local1039 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1028, !dbg !51
  %address.local1040 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1039, i64 %address.local1018, !dbg !51
  store half %2285, ptr addrspace(1) %address.local1040, align 2, !dbg !51
  %address.local1041 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1028, !dbg !51
  %address.local1042 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1041, i64 %address.local1021, !dbg !51
  store half %2284, ptr addrspace(1) %address.local1042, align 2, !dbg !51
  %address.local1043 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1028, !dbg !51
  %address.local1044 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1043, i64 %address.local1024, !dbg !51
  store half %2283, ptr addrspace(1) %address.local1044, align 2, !dbg !51
  br label %2315, !dbg !51

2315:                                             ; preds = %2277, %2276
  br i1 true, label %2316, label %2354, !dbg !51

2316:                                             ; preds = %2315
  %2317 = insertelement <4 x float> poison, float %2154, i64 0, !dbg !49
  %2318 = insertelement <4 x float> %2317, float %2160, i64 1, !dbg !49
  %2319 = insertelement <4 x float> %2318, float %2155, i64 2, !dbg !49
  %2320 = insertelement <4 x float> %2319, float %2161, i64 3, !dbg !49
  %row.reciprocal114 = fdiv arcp <4 x float> splat (float 1.000000e+00), %2105, !dbg !49
  %normalized115 = fmul <4 x float> %2320, %row.reciprocal114, !dbg !49
  %2321 = fptrunc <4 x float> %normalized115 to <4 x half>, !dbg !50
  %2322 = extractelement <4 x half> %2321, i64 3, !dbg !50
  %2323 = extractelement <4 x half> %2321, i64 2, !dbg !50
  %2324 = extractelement <4 x half> %2321, i64 1, !dbg !50
  %2325 = extractelement <4 x half> %2321, i64 0, !dbg !50
  %2326 = insertelement <4 x float> poison, float %2044, i64 0, !dbg !49
  %2327 = insertelement <4 x float> %2326, float %2050, i64 1, !dbg !49
  %2328 = insertelement <4 x float> %2327, float %2045, i64 2, !dbg !49
  %2329 = insertelement <4 x float> %2328, float %2051, i64 3, !dbg !49
  %normalized116 = fmul <4 x float> %2329, %row.reciprocal114, !dbg !49
  %2330 = fptrunc <4 x float> %normalized116 to <4 x half>, !dbg !50
  %2331 = extractelement <4 x half> %2330, i64 3, !dbg !50
  %2332 = extractelement <4 x half> %2330, i64 2, !dbg !50
  %2333 = extractelement <4 x half> %2330, i64 1, !dbg !50
  %2334 = extractelement <4 x half> %2330, i64 0, !dbg !50
  %2335 = insertelement <4 x float> poison, float %1936, i64 0, !dbg !49
  %2336 = insertelement <4 x float> %2335, float %1942, i64 1, !dbg !49
  %2337 = insertelement <4 x float> %2336, float %1937, i64 2, !dbg !49
  %2338 = insertelement <4 x float> %2337, float %1943, i64 3, !dbg !49
  %row.reciprocal117 = fdiv arcp <4 x float> splat (float 1.000000e+00), %1887, !dbg !49
  %normalized118 = fmul <4 x float> %2338, %row.reciprocal117, !dbg !49
  %2339 = fptrunc <4 x float> %normalized118 to <4 x half>, !dbg !50
  %2340 = extractelement <4 x half> %2339, i64 3, !dbg !50
  %2341 = extractelement <4 x half> %2339, i64 2, !dbg !50
  %2342 = extractelement <4 x half> %2339, i64 1, !dbg !50
  %2343 = extractelement <4 x half> %2339, i64 0, !dbg !50
  %2344 = insertelement <4 x float> poison, float %1826, i64 0, !dbg !49
  %2345 = insertelement <4 x float> %2344, float %1832, i64 1, !dbg !49
  %2346 = insertelement <4 x float> %2345, float %1827, i64 2, !dbg !49
  %2347 = insertelement <4 x float> %2346, float %1833, i64 3, !dbg !49
  %normalized119 = fmul <4 x float> %2347, %row.reciprocal117, !dbg !49
  %2348 = fptrunc <4 x float> %normalized119 to <4 x half>, !dbg !50
  %2349 = extractelement <4 x half> %2348, i64 3, !dbg !50
  %2350 = extractelement <4 x half> %2348, i64 2, !dbg !50
  %2351 = extractelement <4 x half> %2348, i64 1, !dbg !50
  %2352 = extractelement <4 x half> %2348, i64 0, !dbg !50
  %2353 = shl nuw nsw i64 %24, 19, !dbg !51
  %address.thread1045 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !51, !range !67, !frisk.register_lifetimes !34
  %address.local1046 = zext nneg i32 %address.thread1045 to i64, !dbg !40
  %address.local1047 = add i64 %25, %address.local1046, !dbg !51
  %address.local1048 = lshr i64 %address.local1046, 6, !dbg !42
  %address.local1049 = shl nuw nsw i64 %address.local1048, 4, !dbg !51
  %address.local1050 = add i64 %address.local1047, %address.local1049, !dbg !51
  %address.local1051 = and i64 %address.local1046, 240, !dbg !51
  %address.local1052 = sub i64 %address.local1050, %address.local1051, !dbg !51
  %address.local1053 = add i64 %address.local1052, %26, !dbg !51
  %address.local1054 = shl nuw nsw i64 %address.local1053, 7, !dbg !51
  %address.local1055 = add nuw nsw i64 %2353, %address.local1054, !dbg !51
  %address.local1056 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1055, !dbg !51
  %address.local1057 = lshr i64 %address.local1046, 2, !dbg !42
  %address.local1058 = and i64 %address.local1057, 12, !dbg !42
  %address.local1059 = or disjoint i64 %address.local1058, 64, !dbg !51
  %address.local1060 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1056, i64 %address.local1059, !dbg !51
  store half %2352, ptr addrspace(1) %address.local1060, align 2, !dbg !51
  %address.local1061 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1055, !dbg !51
  %address.local1062 = or disjoint i64 %address.local1058, 65, !dbg !51
  %address.local1063 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1061, i64 %address.local1062, !dbg !51
  store half %2351, ptr addrspace(1) %address.local1063, align 2, !dbg !51
  %address.local1064 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1055, !dbg !51
  %address.local1065 = or disjoint i64 %address.local1058, 66, !dbg !51
  %address.local1066 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1064, i64 %address.local1065, !dbg !51
  store half %2350, ptr addrspace(1) %address.local1066, align 2, !dbg !51
  %address.local1067 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1055, !dbg !51
  %address.local1068 = or disjoint i64 %address.local1058, 67, !dbg !51
  %address.local1069 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1067, i64 %address.local1068, !dbg !51
  store half %2349, ptr addrspace(1) %address.local1069, align 2, !dbg !51
  %address.local1070 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1055, !dbg !51
  %address.local1071 = or disjoint i64 %address.local1058, 80, !dbg !51
  %address.local1072 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1070, i64 %address.local1071, !dbg !51
  store half %2343, ptr addrspace(1) %address.local1072, align 2, !dbg !51
  %address.local1073 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1055, !dbg !51
  %address.local1074 = or disjoint i64 %address.local1058, 81, !dbg !51
  %address.local1075 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1073, i64 %address.local1074, !dbg !51
  store half %2342, ptr addrspace(1) %address.local1075, align 2, !dbg !51
  %address.local1076 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1055, !dbg !51
  %address.local1077 = or disjoint i64 %address.local1058, 82, !dbg !51
  %address.local1078 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1076, i64 %address.local1077, !dbg !51
  store half %2341, ptr addrspace(1) %address.local1078, align 2, !dbg !51
  %address.local1079 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1055, !dbg !51
  %address.local1080 = or disjoint i64 %address.local1058, 83, !dbg !51
  %address.local1081 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1079, i64 %address.local1080, !dbg !51
  store half %2340, ptr addrspace(1) %address.local1081, align 2, !dbg !51
  %address.local1082 = shl i64 %address.local1053, 7, !dbg !51
  %address.local1083 = add i64 %address.local1082, 8192, !dbg !51
  %address.local1084 = add nuw nsw i64 %2353, %address.local1083, !dbg !51
  %address.local1085 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1084, !dbg !51
  %address.local1086 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1085, i64 %address.local1059, !dbg !51
  store half %2334, ptr addrspace(1) %address.local1086, align 2, !dbg !51
  %address.local1087 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1084, !dbg !51
  %address.local1088 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1087, i64 %address.local1062, !dbg !51
  store half %2333, ptr addrspace(1) %address.local1088, align 2, !dbg !51
  %address.local1089 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1084, !dbg !51
  %address.local1090 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1089, i64 %address.local1065, !dbg !51
  store half %2332, ptr addrspace(1) %address.local1090, align 2, !dbg !51
  %address.local1091 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1084, !dbg !51
  %address.local1092 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1091, i64 %address.local1068, !dbg !51
  store half %2331, ptr addrspace(1) %address.local1092, align 2, !dbg !51
  %address.local1093 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1084, !dbg !51
  %address.local1094 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1093, i64 %address.local1071, !dbg !51
  store half %2325, ptr addrspace(1) %address.local1094, align 2, !dbg !51
  %address.local1095 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1084, !dbg !51
  %address.local1096 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1095, i64 %address.local1074, !dbg !51
  store half %2324, ptr addrspace(1) %address.local1096, align 2, !dbg !51
  %address.local1097 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1084, !dbg !51
  %address.local1098 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1097, i64 %address.local1077, !dbg !51
  store half %2323, ptr addrspace(1) %address.local1098, align 2, !dbg !51
  %address.local1099 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1084, !dbg !51
  %address.local1100 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1099, i64 %address.local1080, !dbg !51
  store half %2322, ptr addrspace(1) %address.local1100, align 2, !dbg !51
  br label %2354, !dbg !51

2354:                                             ; preds = %2316, %2315
  br i1 true, label %2355, label %2393, !dbg !51

2355:                                             ; preds = %2354
  %2356 = insertelement <4 x float> poison, float %2126, i64 0, !dbg !49
  %2357 = insertelement <4 x float> %2356, float %2132, i64 1, !dbg !49
  %2358 = insertelement <4 x float> %2357, float %2127, i64 2, !dbg !49
  %2359 = insertelement <4 x float> %2358, float %2133, i64 3, !dbg !49
  %row.reciprocal120 = fdiv arcp <4 x float> splat (float 1.000000e+00), %2105, !dbg !49
  %normalized121 = fmul <4 x float> %2359, %row.reciprocal120, !dbg !49
  %2360 = fptrunc <4 x float> %normalized121 to <4 x half>, !dbg !50
  %2361 = extractelement <4 x half> %2360, i64 3, !dbg !50
  %2362 = extractelement <4 x half> %2360, i64 2, !dbg !50
  %2363 = extractelement <4 x half> %2360, i64 1, !dbg !50
  %2364 = extractelement <4 x half> %2360, i64 0, !dbg !50
  %2365 = insertelement <4 x float> poison, float %2015, i64 0, !dbg !49
  %2366 = insertelement <4 x float> %2365, float %2021, i64 1, !dbg !49
  %2367 = insertelement <4 x float> %2366, float %2016, i64 2, !dbg !49
  %2368 = insertelement <4 x float> %2367, float %2022, i64 3, !dbg !49
  %normalized122 = fmul <4 x float> %2368, %row.reciprocal120, !dbg !49
  %2369 = fptrunc <4 x float> %normalized122 to <4 x half>, !dbg !50
  %2370 = extractelement <4 x half> %2369, i64 3, !dbg !50
  %2371 = extractelement <4 x half> %2369, i64 2, !dbg !50
  %2372 = extractelement <4 x half> %2369, i64 1, !dbg !50
  %2373 = extractelement <4 x half> %2369, i64 0, !dbg !50
  %2374 = insertelement <4 x float> poison, float %1908, i64 0, !dbg !49
  %2375 = insertelement <4 x float> %2374, float %1914, i64 1, !dbg !49
  %2376 = insertelement <4 x float> %2375, float %1909, i64 2, !dbg !49
  %2377 = insertelement <4 x float> %2376, float %1915, i64 3, !dbg !49
  %row.reciprocal123 = fdiv arcp <4 x float> splat (float 1.000000e+00), %1887, !dbg !49
  %normalized124 = fmul <4 x float> %2377, %row.reciprocal123, !dbg !49
  %2378 = fptrunc <4 x float> %normalized124 to <4 x half>, !dbg !50
  %2379 = extractelement <4 x half> %2378, i64 3, !dbg !50
  %2380 = extractelement <4 x half> %2378, i64 2, !dbg !50
  %2381 = extractelement <4 x half> %2378, i64 1, !dbg !50
  %2382 = extractelement <4 x half> %2378, i64 0, !dbg !50
  %2383 = insertelement <4 x float> poison, float %1797, i64 0, !dbg !49
  %2384 = insertelement <4 x float> %2383, float %1803, i64 1, !dbg !49
  %2385 = insertelement <4 x float> %2384, float %1798, i64 2, !dbg !49
  %2386 = insertelement <4 x float> %2385, float %1804, i64 3, !dbg !49
  %normalized125 = fmul <4 x float> %2386, %row.reciprocal123, !dbg !49
  %2387 = fptrunc <4 x float> %normalized125 to <4 x half>, !dbg !50
  %2388 = extractelement <4 x half> %2387, i64 3, !dbg !50
  %2389 = extractelement <4 x half> %2387, i64 2, !dbg !50
  %2390 = extractelement <4 x half> %2387, i64 1, !dbg !50
  %2391 = extractelement <4 x half> %2387, i64 0, !dbg !50
  %2392 = shl nuw nsw i64 %24, 19, !dbg !51
  %address.thread1101 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !51, !range !67, !frisk.register_lifetimes !34
  %address.local1102 = zext nneg i32 %address.thread1101 to i64, !dbg !40
  %address.local1103 = add i64 %25, %address.local1102, !dbg !51
  %address.local1104 = lshr i64 %address.local1102, 6, !dbg !42
  %address.local1105 = shl nuw nsw i64 %address.local1104, 4, !dbg !51
  %address.local1106 = add i64 %address.local1103, %address.local1105, !dbg !51
  %address.local1107 = and i64 %address.local1102, 240, !dbg !51
  %address.local1108 = sub i64 %address.local1106, %address.local1107, !dbg !51
  %address.local1109 = add i64 %address.local1108, %26, !dbg !51
  %address.local1110 = shl nuw nsw i64 %address.local1109, 7, !dbg !51
  %address.local1111 = add nuw nsw i64 %2392, %address.local1110, !dbg !51
  %address.local1112 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1111, !dbg !51
  %address.local1113 = lshr i64 %address.local1102, 2, !dbg !42
  %address.local1114 = and i64 %address.local1113, 12, !dbg !42
  %address.local1115 = or disjoint i64 %address.local1114, 96, !dbg !51
  %address.local1116 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1112, i64 %address.local1115, !dbg !51
  store half %2391, ptr addrspace(1) %address.local1116, align 2, !dbg !51
  %address.local1117 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1111, !dbg !51
  %address.local1118 = or disjoint i64 %address.local1114, 97, !dbg !51
  %address.local1119 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1117, i64 %address.local1118, !dbg !51
  store half %2390, ptr addrspace(1) %address.local1119, align 2, !dbg !51
  %address.local1120 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1111, !dbg !51
  %address.local1121 = or disjoint i64 %address.local1114, 98, !dbg !51
  %address.local1122 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1120, i64 %address.local1121, !dbg !51
  store half %2389, ptr addrspace(1) %address.local1122, align 2, !dbg !51
  %address.local1123 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1111, !dbg !51
  %address.local1124 = or disjoint i64 %address.local1114, 99, !dbg !51
  %address.local1125 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1123, i64 %address.local1124, !dbg !51
  store half %2388, ptr addrspace(1) %address.local1125, align 2, !dbg !51
  %address.local1126 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1111, !dbg !51
  %address.local1127 = or disjoint i64 %address.local1114, 112, !dbg !51
  %address.local1128 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1126, i64 %address.local1127, !dbg !51
  store half %2382, ptr addrspace(1) %address.local1128, align 2, !dbg !51
  %address.local1129 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1111, !dbg !51
  %address.local1130 = or disjoint i64 %address.local1114, 113, !dbg !51
  %address.local1131 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1129, i64 %address.local1130, !dbg !51
  store half %2381, ptr addrspace(1) %address.local1131, align 2, !dbg !51
  %address.local1132 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1111, !dbg !51
  %address.local1133 = or disjoint i64 %address.local1114, 114, !dbg !51
  %address.local1134 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1132, i64 %address.local1133, !dbg !51
  store half %2380, ptr addrspace(1) %address.local1134, align 2, !dbg !51
  %address.local1135 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1111, !dbg !51
  %address.local1136 = or i64 %address.local1113, 115, !dbg !51
  %address.local1137 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1135, i64 %address.local1136, !dbg !51
  store half %2379, ptr addrspace(1) %address.local1137, align 2, !dbg !51
  %address.local1138 = shl i64 %address.local1109, 7, !dbg !51
  %address.local1139 = add i64 %address.local1138, 8192, !dbg !51
  %address.local1140 = add nuw nsw i64 %2392, %address.local1139, !dbg !51
  %address.local1141 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !51
  %address.local1142 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1141, i64 %address.local1115, !dbg !51
  store half %2373, ptr addrspace(1) %address.local1142, align 2, !dbg !51
  %address.local1143 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !51
  %address.local1144 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1143, i64 %address.local1118, !dbg !51
  store half %2372, ptr addrspace(1) %address.local1144, align 2, !dbg !51
  %address.local1145 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !51
  %address.local1146 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1145, i64 %address.local1121, !dbg !51
  store half %2371, ptr addrspace(1) %address.local1146, align 2, !dbg !51
  %address.local1147 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !51
  %address.local1148 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1147, i64 %address.local1124, !dbg !51
  store half %2370, ptr addrspace(1) %address.local1148, align 2, !dbg !51
  %address.local1149 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !51
  %address.local1150 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1149, i64 %address.local1127, !dbg !51
  store half %2364, ptr addrspace(1) %address.local1150, align 2, !dbg !51
  %address.local1151 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !51
  %address.local1152 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1151, i64 %address.local1130, !dbg !51
  store half %2363, ptr addrspace(1) %address.local1152, align 2, !dbg !51
  %address.local1153 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !51
  %address.local1154 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1153, i64 %address.local1133, !dbg !51
  store half %2362, ptr addrspace(1) %address.local1154, align 2, !dbg !51
  %address.local1155 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1140, !dbg !51
  %address.local1156 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1155, i64 %address.local1136, !dbg !51
  store half %2361, ptr addrspace(1) %address.local1156, align 2, !dbg !51
  br label %2393, !dbg !51

2393:                                             ; preds = %2355, %2354
  ret void, !dbg !38
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

!llvm.dbg.cu = !{!36}
!llvm.module.flags = !{!66}

!33 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/online_softmax_native_acc/final-export/run-20261023-182556-l6_z6lbl/input-0001-test_input_1x32x4096x128-128-128-32-online/retained32-baseline")
!34 = !{}
!35 = !DISubroutineType(types: !34)
!36 = distinct !DICompileUnit(language: DW_LANG_C, file: !33, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!37 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !33, file: !33, line: 23, type: !35, scopeLine: 23, spFlags: DISPFlagDefinition, unit: !36)
!38 = !DILocation(line: 23, column: 3, scope: !37)
!39 = !DILocation(line: 60, column: 24, scope: !37)
!40 = !DILocation(line: 70, column: 22, scope: !37)
!41 = !DILocation(line: 48, column: 17, scope: !37)
!42 = !DILocation(line: 72, column: 24, scope: !37)
!43 = !DILocation(line: 77, column: 19, scope: !37)
!44 = !DILocation(line: 102, column: 23, scope: !37)
!45 = !DILocation(line: 0, scope: !37)
!46 = !DILocation(line: 71, column: 22, scope: !37)
!47 = !DILocation(line: 73, column: 26, scope: !37)
!48 = !DILocation(line: 101, column: 20, scope: !37)
!49 = !DILocation(line: 107, column: 23, scope: !37)
!50 = !DILocation(line: 108, column: 19, scope: !37)
!51 = !DILocation(line: 109, column: 9, scope: !37)
!52 = !DILocation(line: 79, column: 24, scope: !37)
!53 = !DILocation(line: 80, column: 23, scope: !37)
!54 = !DILocation(line: 81, column: 26, scope: !37)
!55 = !DILocation(line: 88, column: 21, scope: !37)
!56 = !DILocation(line: 89, column: 20, scope: !37)
!57 = !DILocation(line: 90, column: 22, scope: !37)
!58 = !DILocation(line: 91, column: 23, scope: !37)
!59 = !DILocation(line: 92, column: 20, scope: !37)
!60 = !DILocation(line: 93, column: 22, scope: !37)
!61 = !DILocation(line: 94, column: 22, scope: !37)
!62 = !DILocation(line: 95, column: 20, scope: !37)
!63 = !DILocation(line: 96, column: 16, scope: !37)
!64 = !DILocation(line: 103, column: 22, scope: !37)
!65 = !{i32 256, i32 1, i32 1}
!66 = !{i32 2, !"Debug Info Version", i32 3}
!67 = !{i32 0, i32 256}
