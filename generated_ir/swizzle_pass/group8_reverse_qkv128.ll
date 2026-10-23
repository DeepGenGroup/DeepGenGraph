; ModuleID = 'generated_ir/pipeline_cleanup/nohint_unrolled.mlir'
source_filename = "generated_ir/pipeline_cleanup/nohint_unrolled.mlir"

@shm_0 = addrspace(3) global [16384 x i8] undef, align 16

; Function Attrs: convergent willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

declare float @__ocml_exp2_f32(float)

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3) #1 !dbg !4 !reqd_work_group_size !32 {
  %5 = call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !5
  %6 = sext i32 %5 to i64, !dbg !5
  %7 = call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !5
  %8 = sext i32 %7 to i64, !dbg !5
  %9 = shl nsw i64 %6, 5, !dbg !5
  %10 = add nsw i64 %9, %8, !dbg !5
  %11 = lshr i64 %10, 5, !dbg !5
  %12 = and i64 %11, 576460752303423480, !dbg !5
  %13 = and i64 %8, 7, !dbg !5
  %14 = or disjoint i64 %12, %13, !dbg !5
  %15 = lshr i64 %10, 3, !dbg !5
  %16 = sub nsw i64 31, %14, !dbg !5
  %17 = call i32 @llvm.amdgcn.workgroup.id.z(), !dbg !5
  %18 = sext i32 %17 to i64, !dbg !5
  %19 = shl i64 %16, 7, !dbg !9
  %20 = icmp ugt i64 %10, 1023, !dbg !11
  %21 = add nsw i64 %14, -32, !dbg !11
  %22 = select i1 %20, i64 %21, i64 %16, !dbg !11
  %23 = sdiv i64 %22, 32, !dbg !11
  %24 = sext i1 %20 to i64, !dbg !11
  %25 = xor i64 %23, %24, !dbg !11
  %26 = add nsw i64 %15, %25, !dbg !11
  %27 = and i64 %26, 31, !dbg !11
  %28 = shl nsw i64 %16, 7, !dbg !11
  %29 = mul nsw i64 %25, -4096, !dbg !11
  %30 = add i64 %28, %29, !dbg !11
  %31 = shl nuw nsw i64 %27, 19, !dbg !11
  %32 = shl nsw i64 %30, 7, !dbg !11
  %33 = add i64 %31, %32, !dbg !11
  %34 = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !12, !range !34
  %35 = zext nneg i32 %34 to i64, !dbg !12
  %36 = lshr i64 %35, 6, !dbg !12
  %37 = lshr i64 %35, 2, !dbg !12
  %38 = or disjoint i64 %19, 127, !dbg !13
  %39 = add i64 %19, 128, !dbg !13
  %40 = icmp uge i64 %39, %38, !dbg !13
  %41 = sub nsw i64 32, %14, !dbg !16
  %42 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %33, !dbg !11
  %43 = shl nuw nsw i64 %35, 2, !dbg !18
  %44 = mul nsw i64 %37, -16, !dbg !18
  %45 = add nsw i64 %43, %44, !dbg !18
  %46 = and i32 %34, 1, !dbg !18
  %index.compact440 = zext nneg i32 %46 to i64, !dbg !18
  %47 = or disjoint i64 %45, %index.compact440, !dbg !18
  %48 = add nuw nsw i64 %47, 1, !dbg !18
  %49 = sdiv i64 %48, 2, !dbg !18
  %50 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %33, !dbg !11
  %51 = getelementptr i8, ptr addrspace(1) %50, i64 64, !dbg !11
  %52 = shl i64 %15, 7, !dbg !17
  %53 = and i64 %52, 3968, !dbg !17
  %54 = shl nsw i64 %18, 12, !dbg !17
  %55 = or disjoint i64 %53, %54, !dbg !17
  %56 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %33, !dbg !11
  %57 = getelementptr i8, ptr addrspace(1) %56, i64 128, !dbg !11
  %58 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %33, !dbg !11
  %59 = getelementptr i8, ptr addrspace(1) %58, i64 192, !dbg !11
  br label %60, !dbg !15

60:                                               ; preds = %847, %4
  %61 = phi i64 [ %1580, %847 ], [ 0, %4 ], !dbg !15
  %62 = phi [2 x <8 x float>] [ %1567, %847 ], [ zeroinitializer, %4 ], !dbg !15
  %63 = phi [2 x <1 x float>] [ %1189, %847 ], [ zeroinitializer, %4 ], !dbg !15
  %64 = phi [2 x <8 x float>] [ %1571, %847 ], [ zeroinitializer, %4 ], !dbg !15
  %65 = phi [2 x <8 x float>] [ %1575, %847 ], [ zeroinitializer, %4 ], !dbg !15
  %66 = phi [2 x <8 x float>] [ %1579, %847 ], [ zeroinitializer, %4 ], !dbg !15
  %67 = icmp slt i64 %61, %41, !dbg !15
  br i1 %67, label %68, label %1581, !dbg !15

68:                                               ; preds = %60
  %69 = shl nsw i64 %61, 7, !dbg !15
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !11, !range !34, !frisk.register_lifetimes !1
  %address.local = zext nneg i32 %address.thread to i64, !dbg !12
  %address.local498 = shl nuw nsw i64 %address.local, 3, !dbg !11
  %address.local499 = and i64 %address.local498, 1536, !dbg !11
  %address.local500 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local499, !dbg !11
  %address.local501 = ptrtoint ptr addrspace(3) %address.local500 to i64, !dbg !11
  %address.local502 = trunc i64 %address.local501 to i32, !dbg !11
  %70 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local502), !dbg !11
  %71 = zext i32 %70 to i64, !dbg !11
  %72 = inttoptr i64 %71 to ptr addrspace(3), !dbg !11
  %address.local503 = and i64 %address.local, 251, !dbg !11
  %address.local504 = lshr i64 %address.local, 2, !dbg !11
  %address.local505 = and i64 %address.local504, 4, !dbg !11
  %address.local506 = add nuw nsw i64 %address.local505, %address.local, !dbg !11
  %address.local507 = and i64 %address.local506, 4, !dbg !11
  %address.local508 = or disjoint i64 %address.local503, %address.local507, !dbg !11
  %address.local509 = shl nuw nsw i64 %address.local508, 7, !dbg !11
  %address.local510 = and i64 %address.local509, 16256, !dbg !11
  %address.local511 = lshr i64 %address.local, 4, !dbg !11
  %address.local512 = and i64 %address.local511, 8, !dbg !11
  %address.local513 = or disjoint i64 %address.local510, %address.local512, !dbg !11
  %address.local514 = getelementptr [2 x i8], ptr addrspace(1) %42, i64 %address.local513, !dbg !11
  %73 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %74 = ptrtoint ptr addrspace(1) %address.local514 to i64, !dbg !11
  %75 = sub i64 %74, %73, !dbg !11
  %dma.buffer.offset1772 = trunc i64 %75 to i32, !dbg !11
  %76 = trunc i64 %73 to i32, !dbg !11
  %77 = insertelement <4 x i32> poison, i32 %76, i64 0, !dbg !11
  %78 = lshr i64 %73, 32, !dbg !11
  %79 = trunc i64 %78 to i32, !dbg !11
  %80 = insertelement <4 x i32> %77, i32 %79, i64 1, !dbg !11
  %81 = insertelement <4 x i32> %80, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc1773 = insertelement <4 x i32> %81, i32 131072, i64 3, !dbg !11
  %82 = ptrtoint ptr addrspace(3) %72 to i32, !dbg !11
  %dma.buffer.lds1774 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %82), !dbg !11
  %83 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1774, i32 %dma.buffer.offset1772, <4 x i32> %dma.buffer.rsrc1773) #10, !dbg !11
  %address.local515 = or disjoint i64 %address.local, 256, !dbg !11
  %address.local516 = shl nuw nsw i64 %address.local515, 3, !dbg !11
  %address.local517 = and i64 %address.local516, 3584, !dbg !11
  %address.local518 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local517, !dbg !11
  %address.local519 = ptrtoint ptr addrspace(3) %address.local518 to i64, !dbg !11
  %address.local520 = trunc i64 %address.local519 to i32, !dbg !11
  %84 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local520), !dbg !11
  %85 = zext i32 %84 to i64, !dbg !11
  %86 = inttoptr i64 %85 to ptr addrspace(3), !dbg !11
  %address.local521 = and i64 %address.local515, 507, !dbg !11
  %address.local522 = lshr i64 %address.local, 2, !dbg !11
  %address.local523 = and i64 %address.local522, 4, !dbg !11
  %address.local524 = add nuw nsw i64 %address.local523, %address.local, !dbg !11
  %address.local525 = and i64 %address.local524, 4, !dbg !11
  %address.local526 = or disjoint i64 %address.local521, %address.local525, !dbg !11
  %address.local527 = shl nuw nsw i64 %address.local526, 7, !dbg !11
  %address.local528 = and i64 %address.local527, 16256, !dbg !11
  %address.local529 = lshr i64 %address.local515, 4, !dbg !11
  %address.local530 = and i64 %address.local529, 24, !dbg !11
  %address.local531 = or disjoint i64 %address.local528, %address.local530, !dbg !11
  %address.local532 = getelementptr [2 x i8], ptr addrspace(1) %42, i64 %address.local531, !dbg !11
  %87 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %88 = ptrtoint ptr addrspace(1) %address.local532 to i64, !dbg !11
  %89 = sub i64 %88, %87, !dbg !11
  %dma.buffer.offset1769 = trunc i64 %89 to i32, !dbg !11
  %90 = trunc i64 %87 to i32, !dbg !11
  %91 = insertelement <4 x i32> poison, i32 %90, i64 0, !dbg !11
  %92 = lshr i64 %87, 32, !dbg !11
  %93 = trunc i64 %92 to i32, !dbg !11
  %94 = insertelement <4 x i32> %91, i32 %93, i64 1, !dbg !11
  %95 = insertelement <4 x i32> %94, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc1770 = insertelement <4 x i32> %95, i32 131072, i64 3, !dbg !11
  %96 = ptrtoint ptr addrspace(3) %86 to i32, !dbg !11
  %dma.buffer.lds1771 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %96), !dbg !11
  %97 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1771, i32 %dma.buffer.offset1769, <4 x i32> %dma.buffer.rsrc1770) #10, !dbg !11
  %98 = lshr i64 %61, 12, !dbg !17
  %99 = add nuw nsw i64 %98, %15, !dbg !17
  %100 = lshr i64 %61, 5, !dbg !17
  %101 = and i64 %100, 127, !dbg !17
  %102 = shl i64 %99, 19, !dbg !17
  %103 = and i64 %102, 16252928, !dbg !17
  %104 = shl i64 %100, 19, !dbg !17
  %105 = shl i64 %61, 14, !dbg !17
  %106 = sub i64 %105, %104, !dbg !17
  %107 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %103, !dbg !17
  %108 = getelementptr [2 x i8], ptr addrspace(1) %107, i64 %101, !dbg !17
  %109 = getelementptr [2 x i8], ptr addrspace(1) %108, i64 %106, !dbg !17
  %address.local533 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local499, !dbg !17
  %address.local534 = ptrtoint ptr addrspace(3) %address.local533 to i64, !dbg !17
  %address.local535 = trunc i64 %address.local534 to i32, !dbg !17
  %110 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local535), !dbg !17
  %111 = zext i32 %110 to i64, !dbg !17
  %112 = inttoptr i64 %111 to ptr addrspace(3), !dbg !17
  %address.local536 = getelementptr [2 x i8], ptr addrspace(1) %109, i64 %address.local513, !dbg !17
  %113 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %114 = ptrtoint ptr addrspace(1) %address.local536 to i64, !dbg !17
  %115 = sub i64 %114, %113, !dbg !17
  %dma.buffer.offset1766 = trunc i64 %115 to i32, !dbg !17
  %116 = trunc i64 %113 to i32, !dbg !17
  %117 = insertelement <4 x i32> poison, i32 %116, i64 0, !dbg !17
  %118 = lshr i64 %113, 32, !dbg !17
  %119 = trunc i64 %118 to i32, !dbg !17
  %120 = insertelement <4 x i32> %117, i32 %119, i64 1, !dbg !17
  %121 = insertelement <4 x i32> %120, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc1767 = insertelement <4 x i32> %121, i32 131072, i64 3, !dbg !17
  %122 = ptrtoint ptr addrspace(3) %112 to i32, !dbg !17
  %dma.buffer.lds1768 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %122), !dbg !17
  %123 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1768, i32 %dma.buffer.offset1766, <4 x i32> %dma.buffer.rsrc1767) #10, !dbg !17
  %address.local537 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local517, !dbg !17
  %address.local538 = ptrtoint ptr addrspace(3) %address.local537 to i64, !dbg !17
  %address.local539 = trunc i64 %address.local538 to i32, !dbg !17
  %124 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local539), !dbg !17
  %125 = zext i32 %124 to i64, !dbg !17
  %126 = inttoptr i64 %125 to ptr addrspace(3), !dbg !17
  %address.local540 = getelementptr [2 x i8], ptr addrspace(1) %109, i64 %address.local531, !dbg !17
  %127 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %128 = ptrtoint ptr addrspace(1) %address.local540 to i64, !dbg !17
  %129 = sub i64 %128, %127, !dbg !17
  %dma.buffer.offset1763 = trunc i64 %129 to i32, !dbg !17
  %130 = trunc i64 %127 to i32, !dbg !17
  %131 = insertelement <4 x i32> poison, i32 %130, i64 0, !dbg !17
  %132 = lshr i64 %127, 32, !dbg !17
  %133 = trunc i64 %132 to i32, !dbg !17
  %134 = insertelement <4 x i32> %131, i32 %133, i64 1, !dbg !17
  %135 = insertelement <4 x i32> %134, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc1764 = insertelement <4 x i32> %135, i32 131072, i64 3, !dbg !17
  %136 = ptrtoint ptr addrspace(3) %126 to i32, !dbg !17
  %dma.buffer.lds1765 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %136), !dbg !17
  %137 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1765, i32 %dma.buffer.offset1763, <4 x i32> %dma.buffer.rsrc1764) #10, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %address.local541 = shl nuw nsw i64 %address.local, 3, !dbg !12
  %address.local542 = lshr i64 %address.local, 6, !dbg !12
  %address.local543 = mul nuw nsw i64 %address.local542, 160, !dbg !12
  %address.local544 = add nuw nsw i64 %address.local541, %address.local543, !dbg !12
  %address.local545 = lshr i64 %address.local, 2, !dbg !12
  %address.local546 = and i64 %address.local545, 12, !dbg !12
  %address.local547 = add nuw nsw i64 %address.local544, %address.local546, !dbg !12
  %address.local548 = and i64 %address.local, 32, !dbg !12
  %address.local549 = icmp eq i64 %address.local548, 0, !dbg !12
  %address.local550 = select i1 %address.local549, i64 0, i64 1016, !dbg !12
  %address.local551 = add nuw nsw i64 %address.local547, %address.local550, !dbg !12
  %address.local552 = lshr i64 %address.local, 1, !dbg !12
  %address.local553 = and i64 %address.local552, 64, !dbg !12
  %address.local554 = shl nuw nsw i32 %address.thread, 3, !dbg !12
  %address.local555 = and i32 %address.local554, 1920, !dbg !12
  %address.local556 = zext nneg i32 %address.local555 to i64, !dbg !12
  %address.local557 = or disjoint i64 %address.local553, %address.local556, !dbg !12
  %address.local558 = lshr i64 %address.local, 2, !dbg !12
  %address.local559 = and i64 %address.local558, 1, !dbg !12
  %address.local560 = and i64 %address.local542, %address.local559, !dbg !12
  %address.local561 = shl nuw nsw i64 %address.local560, 6, !dbg !12
  %address.local562 = add nuw nsw i64 %address.local557, %address.local561, !dbg !12
  %address.local563 = sub nsw i64 %address.local551, %address.local562, !dbg !12
  %address.local564 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local563, !dbg !12
  %138 = load <4 x half>, ptr addrspace(3) %address.local564, align 8, !dbg !12
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local564, i64 4096, !dbg !12
  %139 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !12
  %lds.const.gep1583 = getelementptr i8, ptr addrspace(3) %address.local564, i64 1024, !dbg !12
  %140 = load <4 x half>, ptr addrspace(3) %lds.const.gep1583, align 8, !dbg !12
  %141 = fmul <4 x half> %138, splat (half 0xH3015), !dbg !12
  %lds.const.gep1584 = getelementptr i8, ptr addrspace(3) %address.local564, i64 5120, !dbg !12
  %142 = load <4 x half>, ptr addrspace(3) %lds.const.gep1584, align 8, !dbg !12
  %143 = fmul <4 x half> %139, splat (half 0xH3015), !dbg !12
  %144 = fmul <4 x half> %140, splat (half 0xH3015), !dbg !12
  %145 = fmul <4 x half> %142, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %address.local571 = mul nuw nsw i64 %address.local, 160, !dbg !18
  %address.local572 = mul nsw i64 %address.local558, -640, !dbg !18
  %address.local573 = add nsw i64 %address.local571, %address.local572, !dbg !18
  %address.local574 = shl nuw nsw i64 %address.local, 1, !dbg !18
  %address.local575 = and i64 %address.local574, 24, !dbg !18
  %address.local576 = or disjoint i64 %address.local573, %address.local575, !dbg !18
  %address.local577 = shl i64 %address.local, 62, !dbg !18
  %address.local578 = ashr exact i64 %address.local577, 57, !dbg !18
  %address.local579 = and i64 %address.local578, -64, !dbg !18
  %address.local580 = add nsw i64 %address.local576, %address.local579, !dbg !18
  %address.local581 = add nsw i64 %address.local580, %address.local546, !dbg !18
  %address.local582 = add i64 %address.local581, %address.local550, !dbg !18
  %address.local583 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local582, !dbg !18
  %146 = load <4 x half>, ptr addrspace(3) %address.local583, align 8, !dbg !18
  %147 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %146, <4 x float> zeroinitializer) #9, !dbg !18
  %lds.const.gep1585 = getelementptr i8, ptr addrspace(3) %address.local583, i64 4096, !dbg !18
  %148 = load <4 x half>, ptr addrspace(3) %lds.const.gep1585, align 8, !dbg !18
  %address.local594 = shl nuw nsw i32 %address.thread, 1, !dbg !18
  %address.local595 = and i32 %address.local594, 24, !dbg !18
  %address.local596 = shl nuw nsw i32 %address.thread, 8, !dbg !18
  %address.local597 = and i32 %address.local596, 768, !dbg !18
  %address.local598 = or disjoint i32 %address.local595, %address.local597, !dbg !18
  %address.local599 = shl nuw nsw i32 %address.thread, 5, !dbg !18
  %address.local600 = and i32 %address.local599, 96, !dbg !18
  %address.local601 = or disjoint i32 %address.local598, %address.local600, !dbg !18
  %address.local602 = zext nneg i32 %address.local601 to i64, !dbg !18
  %address.base = call i64 asm sideeffect "", "=v,0"(i64 %49) #9, !dbg !18, !frisk.register_lifetimes !1
  %address.local603 = lshr i32 %address.thread, 1, !dbg !18
  %address.local604 = and i32 %address.local603, 1, !dbg !18
  %address.local605 = zext nneg i32 %address.local604 to i64, !dbg !18
  %address.local606 = add nsw i64 %address.base, %address.local605, !dbg !18
  %address.local607 = shl i64 %address.local606, 6, !dbg !18
  %address.local608 = sub i64 %address.local602, %address.local607, !dbg !18
  %address.local609 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local608, !dbg !18
  %address.local610 = getelementptr [2 x i8], ptr addrspace(3) %address.local609, i64 %address.local546, !dbg !18
  %address.local611 = getelementptr [2 x i8], ptr addrspace(3) %address.local610, i64 %address.local550, !dbg !18
  %address.local612 = getelementptr i8, ptr addrspace(3) %address.local611, i64 64, !dbg !18
  %149 = load <4 x half>, ptr addrspace(3) %address.local612, align 8, !dbg !18
  %150 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %143, <4 x half> %148, <4 x float> %147) #9, !dbg !18
  %151 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %149, <4 x float> zeroinitializer) #9, !dbg !18
  %lds.const.gep1586 = getelementptr i8, ptr addrspace(3) %address.local612, i64 4096, !dbg !18
  %152 = load <4 x half>, ptr addrspace(3) %lds.const.gep1586, align 8, !dbg !18
  %lds.const.gep1587 = getelementptr i8, ptr addrspace(3) %address.local583, i64 128, !dbg !18
  %153 = load <4 x half>, ptr addrspace(3) %lds.const.gep1587, align 8, !dbg !18
  %154 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %143, <4 x half> %152, <4 x float> %151) #9, !dbg !18
  %155 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %153, <4 x float> zeroinitializer) #9, !dbg !18
  %lds.const.gep1588 = getelementptr i8, ptr addrspace(3) %address.local583, i64 4224, !dbg !18
  %156 = load <4 x half>, ptr addrspace(3) %lds.const.gep1588, align 8, !dbg !18
  %lds.const.gep1589 = getelementptr i8, ptr addrspace(3) %address.local612, i64 128, !dbg !18
  %157 = load <4 x half>, ptr addrspace(3) %lds.const.gep1589, align 8, !dbg !18
  %158 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %143, <4 x half> %156, <4 x float> %155) #9, !dbg !18
  %159 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %157, <4 x float> zeroinitializer) #9, !dbg !18
  %lds.const.gep1590 = getelementptr i8, ptr addrspace(3) %address.local612, i64 4224, !dbg !18
  %160 = load <4 x half>, ptr addrspace(3) %lds.const.gep1590, align 8, !dbg !18
  %lds.const.gep1591 = getelementptr i8, ptr addrspace(3) %address.local583, i64 1024, !dbg !18
  %161 = load <4 x half>, ptr addrspace(3) %lds.const.gep1591, align 8, !dbg !18
  %162 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %143, <4 x half> %160, <4 x float> %159) #9, !dbg !18
  %163 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %161, <4 x float> zeroinitializer) #9, !dbg !18
  %lds.const.gep1592 = getelementptr i8, ptr addrspace(3) %address.local583, i64 5120, !dbg !18
  %164 = load <4 x half>, ptr addrspace(3) %lds.const.gep1592, align 8, !dbg !18
  %lds.const.gep1593 = getelementptr i8, ptr addrspace(3) %address.local612, i64 1024, !dbg !18
  %165 = load <4 x half>, ptr addrspace(3) %lds.const.gep1593, align 8, !dbg !18
  %166 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %143, <4 x half> %164, <4 x float> %163) #9, !dbg !18
  %167 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %165, <4 x float> zeroinitializer) #9, !dbg !18
  %lds.const.gep1594 = getelementptr i8, ptr addrspace(3) %address.local612, i64 5120, !dbg !18
  %168 = load <4 x half>, ptr addrspace(3) %lds.const.gep1594, align 8, !dbg !18
  %lds.const.gep1595 = getelementptr i8, ptr addrspace(3) %address.local583, i64 1152, !dbg !18
  %169 = load <4 x half>, ptr addrspace(3) %lds.const.gep1595, align 8, !dbg !18
  %170 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %143, <4 x half> %168, <4 x float> %167) #9, !dbg !18
  %171 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %169, <4 x float> zeroinitializer) #9, !dbg !18
  %lds.const.gep1596 = getelementptr i8, ptr addrspace(3) %address.local583, i64 5248, !dbg !18
  %172 = load <4 x half>, ptr addrspace(3) %lds.const.gep1596, align 8, !dbg !18
  %lds.const.gep1597 = getelementptr i8, ptr addrspace(3) %address.local612, i64 1152, !dbg !18
  %173 = load <4 x half>, ptr addrspace(3) %lds.const.gep1597, align 8, !dbg !18
  %174 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %143, <4 x half> %172, <4 x float> %171) #9, !dbg !18
  %175 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %141, <4 x half> %173, <4 x float> zeroinitializer) #9, !dbg !18
  %lds.const.gep1598 = getelementptr i8, ptr addrspace(3) %address.local612, i64 5248, !dbg !18
  %176 = load <4 x half>, ptr addrspace(3) %lds.const.gep1598, align 8, !dbg !18
  %177 = load <4 x half>, ptr addrspace(3) %address.local583, align 8, !dbg !18
  %178 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %143, <4 x half> %176, <4 x float> %175) #9, !dbg !18
  %179 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %144, <4 x half> %177, <4 x float> zeroinitializer) #9, !dbg !18
  %lds.const.gep1599 = getelementptr i8, ptr addrspace(3) %address.local583, i64 4096, !dbg !18
  %180 = load <4 x half>, ptr addrspace(3) %lds.const.gep1599, align 8, !dbg !18
  %181 = load <4 x half>, ptr addrspace(3) %address.local612, align 8, !dbg !18
  %182 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %145, <4 x half> %180, <4 x float> %179) #9, !dbg !18
  %183 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %144, <4 x half> %181, <4 x float> zeroinitializer) #9, !dbg !18
  %lds.const.gep1600 = getelementptr i8, ptr addrspace(3) %address.local612, i64 4096, !dbg !18
  %184 = load <4 x half>, ptr addrspace(3) %lds.const.gep1600, align 8, !dbg !18
  %lds.const.gep1601 = getelementptr i8, ptr addrspace(3) %address.local583, i64 128, !dbg !18
  %185 = load <4 x half>, ptr addrspace(3) %lds.const.gep1601, align 8, !dbg !18
  %186 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %145, <4 x half> %184, <4 x float> %183) #9, !dbg !18
  %187 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %144, <4 x half> %185, <4 x float> zeroinitializer) #9, !dbg !18
  %lds.const.gep1602 = getelementptr i8, ptr addrspace(3) %address.local583, i64 4224, !dbg !18
  %188 = load <4 x half>, ptr addrspace(3) %lds.const.gep1602, align 8, !dbg !18
  %lds.const.gep1603 = getelementptr i8, ptr addrspace(3) %address.local612, i64 128, !dbg !18
  %189 = load <4 x half>, ptr addrspace(3) %lds.const.gep1603, align 8, !dbg !18
  %190 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %145, <4 x half> %188, <4 x float> %187) #9, !dbg !18
  %191 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %144, <4 x half> %189, <4 x float> zeroinitializer) #9, !dbg !18
  %lds.const.gep1604 = getelementptr i8, ptr addrspace(3) %address.local612, i64 4224, !dbg !18
  %192 = load <4 x half>, ptr addrspace(3) %lds.const.gep1604, align 8, !dbg !18
  %lds.const.gep1605 = getelementptr i8, ptr addrspace(3) %address.local583, i64 1024, !dbg !18
  %193 = load <4 x half>, ptr addrspace(3) %lds.const.gep1605, align 8, !dbg !18
  %194 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %145, <4 x half> %192, <4 x float> %191) #9, !dbg !18
  %195 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %144, <4 x half> %193, <4 x float> zeroinitializer) #9, !dbg !18
  %lds.const.gep1606 = getelementptr i8, ptr addrspace(3) %address.local583, i64 5120, !dbg !18
  %196 = load <4 x half>, ptr addrspace(3) %lds.const.gep1606, align 8, !dbg !18
  %lds.const.gep1607 = getelementptr i8, ptr addrspace(3) %address.local612, i64 1024, !dbg !18
  %197 = load <4 x half>, ptr addrspace(3) %lds.const.gep1607, align 8, !dbg !18
  %198 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %145, <4 x half> %196, <4 x float> %195) #9, !dbg !18
  %199 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %144, <4 x half> %197, <4 x float> zeroinitializer) #9, !dbg !18
  %lds.const.gep1608 = getelementptr i8, ptr addrspace(3) %address.local612, i64 5120, !dbg !18
  %200 = load <4 x half>, ptr addrspace(3) %lds.const.gep1608, align 8, !dbg !18
  %lds.const.gep1609 = getelementptr i8, ptr addrspace(3) %address.local583, i64 1152, !dbg !18
  %201 = load <4 x half>, ptr addrspace(3) %lds.const.gep1609, align 8, !dbg !18
  %202 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %145, <4 x half> %200, <4 x float> %199) #9, !dbg !18
  %203 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %144, <4 x half> %201, <4 x float> zeroinitializer) #9, !dbg !18
  %lds.const.gep1610 = getelementptr i8, ptr addrspace(3) %address.local583, i64 5248, !dbg !18
  %204 = load <4 x half>, ptr addrspace(3) %lds.const.gep1610, align 8, !dbg !18
  %lds.const.gep1611 = getelementptr i8, ptr addrspace(3) %address.local612, i64 1152, !dbg !18
  %205 = load <4 x half>, ptr addrspace(3) %lds.const.gep1611, align 8, !dbg !18
  %206 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %145, <4 x half> %204, <4 x float> %203) #9, !dbg !18
  %lds.const.gep1612 = getelementptr i8, ptr addrspace(3) %address.local612, i64 5248, !dbg !18
  %207 = load <4 x half>, ptr addrspace(3) %lds.const.gep1612, align 8, !dbg !18
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !18
  %208 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %144, <4 x half> %205, <4 x float> zeroinitializer) #9, !dbg !18
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !18
  %209 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %145, <4 x half> %207, <4 x float> %208) #9, !dbg !18
  %210 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local502), !dbg !11
  %211 = zext i32 %210 to i64, !dbg !11
  %212 = inttoptr i64 %211 to ptr addrspace(3), !dbg !11
  %address.local921 = getelementptr [2 x i8], ptr addrspace(1) %51, i64 %address.local513, !dbg !11
  %213 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %214 = ptrtoint ptr addrspace(1) %address.local921 to i64, !dbg !11
  %215 = sub i64 %214, %213, !dbg !11
  %dma.buffer.offset1760 = trunc i64 %215 to i32, !dbg !11
  %216 = trunc i64 %213 to i32, !dbg !11
  %217 = insertelement <4 x i32> poison, i32 %216, i64 0, !dbg !11
  %218 = lshr i64 %213, 32, !dbg !11
  %219 = trunc i64 %218 to i32, !dbg !11
  %220 = insertelement <4 x i32> %217, i32 %219, i64 1, !dbg !11
  %221 = insertelement <4 x i32> %220, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc1761 = insertelement <4 x i32> %221, i32 131072, i64 3, !dbg !11
  %222 = ptrtoint ptr addrspace(3) %212 to i32, !dbg !11
  %dma.buffer.lds1762 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %222), !dbg !11
  %223 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1762, i32 %dma.buffer.offset1760, <4 x i32> %dma.buffer.rsrc1761) #10, !dbg !11
  %224 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local520), !dbg !11
  %225 = zext i32 %224 to i64, !dbg !11
  %226 = inttoptr i64 %225 to ptr addrspace(3), !dbg !11
  %address.local922 = getelementptr [2 x i8], ptr addrspace(1) %51, i64 %address.local531, !dbg !11
  %227 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %228 = ptrtoint ptr addrspace(1) %address.local922 to i64, !dbg !11
  %229 = sub i64 %228, %227, !dbg !11
  %dma.buffer.offset1757 = trunc i64 %229 to i32, !dbg !11
  %230 = trunc i64 %227 to i32, !dbg !11
  %231 = insertelement <4 x i32> poison, i32 %230, i64 0, !dbg !11
  %232 = lshr i64 %227, 32, !dbg !11
  %233 = trunc i64 %232 to i32, !dbg !11
  %234 = insertelement <4 x i32> %231, i32 %233, i64 1, !dbg !11
  %235 = insertelement <4 x i32> %234, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc1758 = insertelement <4 x i32> %235, i32 131072, i64 3, !dbg !11
  %236 = ptrtoint ptr addrspace(3) %226 to i32, !dbg !11
  %dma.buffer.lds1759 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %236), !dbg !11
  %237 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1759, i32 %dma.buffer.offset1757, <4 x i32> %dma.buffer.rsrc1758) #10, !dbg !11
  %238 = add nuw i64 %61, 1024, !dbg !17
  %239 = icmp slt i64 %238, 0, !dbg !17
  %240 = sub nuw i64 -1025, %61, !dbg !17
  %241 = select i1 %239, i64 %240, i64 %238, !dbg !17
  %242 = sdiv i64 %241, 4096, !dbg !17
  %.lobit63 = ashr i64 %238, 63, !dbg !17
  %243 = xor i64 %242, %.lobit63, !dbg !17
  %244 = add nsw i64 %243, %15, !dbg !17
  %245 = add nsw i64 %100, %55, !dbg !17
  %246 = add nsw i64 %245, 32, !dbg !17
  %247 = icmp slt i64 %245, -32, !dbg !17
  %248 = sub nsw i64 -33, %245, !dbg !17
  %249 = select i1 %247, i64 %248, i64 %246, !dbg !17
  %250 = sdiv i64 %249, 128, !dbg !17
  %251 = sext i1 %247 to i64, !dbg !17
  %252 = xor i64 %250, %251, !dbg !17
  %253 = shl nsw i64 %252, 7, !dbg !17
  %254 = sub i64 %55, %253, !dbg !17
  %255 = shl i64 %244, 19, !dbg !17
  %256 = and i64 %255, 16252928, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %257 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %256, !dbg !17
  %258 = getelementptr [2 x i8], ptr addrspace(1) %257, i64 %100, !dbg !17
  %259 = getelementptr [2 x i8], ptr addrspace(1) %258, i64 %254, !dbg !17
  %260 = getelementptr i8, ptr addrspace(1) %259, i64 64, !dbg !17
  %261 = getelementptr [2 x i8], ptr addrspace(1) %260, i64 %106, !dbg !17
  %262 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local535), !dbg !17
  %263 = zext i32 %262 to i64, !dbg !17
  %264 = inttoptr i64 %263 to ptr addrspace(3), !dbg !17
  %address.local923 = getelementptr [2 x i8], ptr addrspace(1) %261, i64 %address.local513, !dbg !17
  %265 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %266 = ptrtoint ptr addrspace(1) %address.local923 to i64, !dbg !17
  %267 = sub i64 %266, %265, !dbg !17
  %dma.buffer.offset1754 = trunc i64 %267 to i32, !dbg !17
  %268 = trunc i64 %265 to i32, !dbg !17
  %269 = insertelement <4 x i32> poison, i32 %268, i64 0, !dbg !17
  %270 = lshr i64 %265, 32, !dbg !17
  %271 = trunc i64 %270 to i32, !dbg !17
  %272 = insertelement <4 x i32> %269, i32 %271, i64 1, !dbg !17
  %273 = insertelement <4 x i32> %272, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc1755 = insertelement <4 x i32> %273, i32 131072, i64 3, !dbg !17
  %274 = ptrtoint ptr addrspace(3) %264 to i32, !dbg !17
  %dma.buffer.lds1756 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %274), !dbg !17
  %275 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1756, i32 %dma.buffer.offset1754, <4 x i32> %dma.buffer.rsrc1755) #10, !dbg !17
  %276 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local539), !dbg !17
  %277 = zext i32 %276 to i64, !dbg !17
  %278 = inttoptr i64 %277 to ptr addrspace(3), !dbg !17
  %address.local924 = getelementptr [2 x i8], ptr addrspace(1) %261, i64 %address.local531, !dbg !17
  %279 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %280 = ptrtoint ptr addrspace(1) %address.local924 to i64, !dbg !17
  %281 = sub i64 %280, %279, !dbg !17
  %dma.buffer.offset1751 = trunc i64 %281 to i32, !dbg !17
  %282 = trunc i64 %279 to i32, !dbg !17
  %283 = insertelement <4 x i32> poison, i32 %282, i64 0, !dbg !17
  %284 = lshr i64 %279, 32, !dbg !17
  %285 = trunc i64 %284 to i32, !dbg !17
  %286 = insertelement <4 x i32> %283, i32 %285, i64 1, !dbg !17
  %287 = insertelement <4 x i32> %286, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc1752 = insertelement <4 x i32> %287, i32 131072, i64 3, !dbg !17
  %288 = ptrtoint ptr addrspace(3) %278 to i32, !dbg !17
  %dma.buffer.lds1753 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %288), !dbg !17
  %289 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1753, i32 %dma.buffer.offset1751, <4 x i32> %dma.buffer.rsrc1752) #10, !dbg !17
  %290 = load <4 x half>, ptr addrspace(3) %address.local564, align 8, !dbg !12
  %lds.const.gep1613 = getelementptr i8, ptr addrspace(3) %address.local564, i64 4096, !dbg !12
  %291 = load <4 x half>, ptr addrspace(3) %lds.const.gep1613, align 8, !dbg !12
  %lds.const.gep1614 = getelementptr i8, ptr addrspace(3) %address.local564, i64 1024, !dbg !12
  %292 = load <4 x half>, ptr addrspace(3) %lds.const.gep1614, align 8, !dbg !12
  %293 = fmul <4 x half> %290, splat (half 0xH3015), !dbg !12
  %lds.const.gep1615 = getelementptr i8, ptr addrspace(3) %address.local564, i64 5120, !dbg !12
  %294 = load <4 x half>, ptr addrspace(3) %lds.const.gep1615, align 8, !dbg !12
  %295 = fmul <4 x half> %291, splat (half 0xH3015), !dbg !12
  %296 = fmul <4 x half> %292, splat (half 0xH3015), !dbg !12
  %297 = fmul <4 x half> %294, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %298 = load <4 x half>, ptr addrspace(3) %address.local583, align 8, !dbg !18
  %299 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %293, <4 x half> %298, <4 x float> %150) #9, !dbg !18
  %lds.const.gep1616 = getelementptr i8, ptr addrspace(3) %address.local583, i64 4096, !dbg !18
  %300 = load <4 x half>, ptr addrspace(3) %lds.const.gep1616, align 8, !dbg !18
  %301 = load <4 x half>, ptr addrspace(3) %address.local612, align 8, !dbg !18
  %302 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %295, <4 x half> %300, <4 x float> %299) #9, !dbg !18
  %303 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %293, <4 x half> %301, <4 x float> %154) #9, !dbg !18
  %lds.const.gep1617 = getelementptr i8, ptr addrspace(3) %address.local612, i64 4096, !dbg !18
  %304 = load <4 x half>, ptr addrspace(3) %lds.const.gep1617, align 8, !dbg !18
  %lds.const.gep1618 = getelementptr i8, ptr addrspace(3) %address.local583, i64 128, !dbg !18
  %305 = load <4 x half>, ptr addrspace(3) %lds.const.gep1618, align 8, !dbg !18
  %306 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %295, <4 x half> %304, <4 x float> %303) #9, !dbg !18
  %307 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %293, <4 x half> %305, <4 x float> %158) #9, !dbg !18
  %lds.const.gep1619 = getelementptr i8, ptr addrspace(3) %address.local583, i64 4224, !dbg !18
  %308 = load <4 x half>, ptr addrspace(3) %lds.const.gep1619, align 8, !dbg !18
  %lds.const.gep1620 = getelementptr i8, ptr addrspace(3) %address.local612, i64 128, !dbg !18
  %309 = load <4 x half>, ptr addrspace(3) %lds.const.gep1620, align 8, !dbg !18
  %310 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %295, <4 x half> %308, <4 x float> %307) #9, !dbg !18
  %311 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %293, <4 x half> %309, <4 x float> %162) #9, !dbg !18
  %lds.const.gep1621 = getelementptr i8, ptr addrspace(3) %address.local612, i64 4224, !dbg !18
  %312 = load <4 x half>, ptr addrspace(3) %lds.const.gep1621, align 8, !dbg !18
  %lds.const.gep1622 = getelementptr i8, ptr addrspace(3) %address.local583, i64 1024, !dbg !18
  %313 = load <4 x half>, ptr addrspace(3) %lds.const.gep1622, align 8, !dbg !18
  %314 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %295, <4 x half> %312, <4 x float> %311) #9, !dbg !18
  %315 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %293, <4 x half> %313, <4 x float> %166) #9, !dbg !18
  %lds.const.gep1623 = getelementptr i8, ptr addrspace(3) %address.local583, i64 5120, !dbg !18
  %316 = load <4 x half>, ptr addrspace(3) %lds.const.gep1623, align 8, !dbg !18
  %lds.const.gep1624 = getelementptr i8, ptr addrspace(3) %address.local612, i64 1024, !dbg !18
  %317 = load <4 x half>, ptr addrspace(3) %lds.const.gep1624, align 8, !dbg !18
  %318 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %295, <4 x half> %316, <4 x float> %315) #9, !dbg !18
  %319 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %293, <4 x half> %317, <4 x float> %170) #9, !dbg !18
  %lds.const.gep1625 = getelementptr i8, ptr addrspace(3) %address.local612, i64 5120, !dbg !18
  %320 = load <4 x half>, ptr addrspace(3) %lds.const.gep1625, align 8, !dbg !18
  %lds.const.gep1626 = getelementptr i8, ptr addrspace(3) %address.local583, i64 1152, !dbg !18
  %321 = load <4 x half>, ptr addrspace(3) %lds.const.gep1626, align 8, !dbg !18
  %322 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %295, <4 x half> %320, <4 x float> %319) #9, !dbg !18
  %323 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %293, <4 x half> %321, <4 x float> %174) #9, !dbg !18
  %lds.const.gep1627 = getelementptr i8, ptr addrspace(3) %address.local583, i64 5248, !dbg !18
  %324 = load <4 x half>, ptr addrspace(3) %lds.const.gep1627, align 8, !dbg !18
  %lds.const.gep1628 = getelementptr i8, ptr addrspace(3) %address.local612, i64 1152, !dbg !18
  %325 = load <4 x half>, ptr addrspace(3) %lds.const.gep1628, align 8, !dbg !18
  %326 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %295, <4 x half> %324, <4 x float> %323) #9, !dbg !18
  %327 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %293, <4 x half> %325, <4 x float> %178) #9, !dbg !18
  %lds.const.gep1629 = getelementptr i8, ptr addrspace(3) %address.local612, i64 5248, !dbg !18
  %328 = load <4 x half>, ptr addrspace(3) %lds.const.gep1629, align 8, !dbg !18
  %329 = load <4 x half>, ptr addrspace(3) %address.local583, align 8, !dbg !18
  %330 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %295, <4 x half> %328, <4 x float> %327) #9, !dbg !18
  %331 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %296, <4 x half> %329, <4 x float> %182) #9, !dbg !18
  %lds.const.gep1630 = getelementptr i8, ptr addrspace(3) %address.local583, i64 4096, !dbg !18
  %332 = load <4 x half>, ptr addrspace(3) %lds.const.gep1630, align 8, !dbg !18
  %333 = load <4 x half>, ptr addrspace(3) %address.local612, align 8, !dbg !18
  %334 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %297, <4 x half> %332, <4 x float> %331) #9, !dbg !18
  %335 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %296, <4 x half> %333, <4 x float> %186) #9, !dbg !18
  %lds.const.gep1631 = getelementptr i8, ptr addrspace(3) %address.local612, i64 4096, !dbg !18
  %336 = load <4 x half>, ptr addrspace(3) %lds.const.gep1631, align 8, !dbg !18
  %lds.const.gep1632 = getelementptr i8, ptr addrspace(3) %address.local583, i64 128, !dbg !18
  %337 = load <4 x half>, ptr addrspace(3) %lds.const.gep1632, align 8, !dbg !18
  %338 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %297, <4 x half> %336, <4 x float> %335) #9, !dbg !18
  %339 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %296, <4 x half> %337, <4 x float> %190) #9, !dbg !18
  %lds.const.gep1633 = getelementptr i8, ptr addrspace(3) %address.local583, i64 4224, !dbg !18
  %340 = load <4 x half>, ptr addrspace(3) %lds.const.gep1633, align 8, !dbg !18
  %lds.const.gep1634 = getelementptr i8, ptr addrspace(3) %address.local612, i64 128, !dbg !18
  %341 = load <4 x half>, ptr addrspace(3) %lds.const.gep1634, align 8, !dbg !18
  %342 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %297, <4 x half> %340, <4 x float> %339) #9, !dbg !18
  %343 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %296, <4 x half> %341, <4 x float> %194) #9, !dbg !18
  %lds.const.gep1635 = getelementptr i8, ptr addrspace(3) %address.local612, i64 4224, !dbg !18
  %344 = load <4 x half>, ptr addrspace(3) %lds.const.gep1635, align 8, !dbg !18
  %lds.const.gep1636 = getelementptr i8, ptr addrspace(3) %address.local583, i64 1024, !dbg !18
  %345 = load <4 x half>, ptr addrspace(3) %lds.const.gep1636, align 8, !dbg !18
  %346 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %297, <4 x half> %344, <4 x float> %343) #9, !dbg !18
  %347 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %296, <4 x half> %345, <4 x float> %198) #9, !dbg !18
  %lds.const.gep1637 = getelementptr i8, ptr addrspace(3) %address.local583, i64 5120, !dbg !18
  %348 = load <4 x half>, ptr addrspace(3) %lds.const.gep1637, align 8, !dbg !18
  %lds.const.gep1638 = getelementptr i8, ptr addrspace(3) %address.local612, i64 1024, !dbg !18
  %349 = load <4 x half>, ptr addrspace(3) %lds.const.gep1638, align 8, !dbg !18
  %350 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %297, <4 x half> %348, <4 x float> %347) #9, !dbg !18
  %351 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %296, <4 x half> %349, <4 x float> %202) #9, !dbg !18
  %lds.const.gep1639 = getelementptr i8, ptr addrspace(3) %address.local612, i64 5120, !dbg !18
  %352 = load <4 x half>, ptr addrspace(3) %lds.const.gep1639, align 8, !dbg !18
  %lds.const.gep1640 = getelementptr i8, ptr addrspace(3) %address.local583, i64 1152, !dbg !18
  %353 = load <4 x half>, ptr addrspace(3) %lds.const.gep1640, align 8, !dbg !18
  %354 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %297, <4 x half> %352, <4 x float> %351) #9, !dbg !18
  %355 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %296, <4 x half> %353, <4 x float> %206) #9, !dbg !18
  %lds.const.gep1641 = getelementptr i8, ptr addrspace(3) %address.local583, i64 5248, !dbg !18
  %356 = load <4 x half>, ptr addrspace(3) %lds.const.gep1641, align 8, !dbg !18
  %lds.const.gep1642 = getelementptr i8, ptr addrspace(3) %address.local612, i64 1152, !dbg !18
  %357 = load <4 x half>, ptr addrspace(3) %lds.const.gep1642, align 8, !dbg !18
  %358 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %297, <4 x half> %356, <4 x float> %355) #9, !dbg !18
  %lds.const.gep1643 = getelementptr i8, ptr addrspace(3) %address.local612, i64 5248, !dbg !18
  %359 = load <4 x half>, ptr addrspace(3) %lds.const.gep1643, align 8, !dbg !18
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !18
  %360 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %296, <4 x half> %357, <4 x float> %209) #9, !dbg !18
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !18
  %361 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %297, <4 x half> %359, <4 x float> %360) #9, !dbg !18
  %362 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local502), !dbg !11
  %363 = zext i32 %362 to i64, !dbg !11
  %364 = inttoptr i64 %363 to ptr addrspace(3), !dbg !11
  %address.local925 = getelementptr [2 x i8], ptr addrspace(1) %57, i64 %address.local513, !dbg !11
  %365 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %366 = ptrtoint ptr addrspace(1) %address.local925 to i64, !dbg !11
  %367 = sub i64 %366, %365, !dbg !11
  %dma.buffer.offset1748 = trunc i64 %367 to i32, !dbg !11
  %368 = trunc i64 %365 to i32, !dbg !11
  %369 = insertelement <4 x i32> poison, i32 %368, i64 0, !dbg !11
  %370 = lshr i64 %365, 32, !dbg !11
  %371 = trunc i64 %370 to i32, !dbg !11
  %372 = insertelement <4 x i32> %369, i32 %371, i64 1, !dbg !11
  %373 = insertelement <4 x i32> %372, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc1749 = insertelement <4 x i32> %373, i32 131072, i64 3, !dbg !11
  %374 = ptrtoint ptr addrspace(3) %364 to i32, !dbg !11
  %dma.buffer.lds1750 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %374), !dbg !11
  %375 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1750, i32 %dma.buffer.offset1748, <4 x i32> %dma.buffer.rsrc1749) #10, !dbg !11
  %376 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local520), !dbg !11
  %377 = zext i32 %376 to i64, !dbg !11
  %378 = inttoptr i64 %377 to ptr addrspace(3), !dbg !11
  %address.local926 = getelementptr [2 x i8], ptr addrspace(1) %57, i64 %address.local531, !dbg !11
  %379 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %380 = ptrtoint ptr addrspace(1) %address.local926 to i64, !dbg !11
  %381 = sub i64 %380, %379, !dbg !11
  %dma.buffer.offset1745 = trunc i64 %381 to i32, !dbg !11
  %382 = trunc i64 %379 to i32, !dbg !11
  %383 = insertelement <4 x i32> poison, i32 %382, i64 0, !dbg !11
  %384 = lshr i64 %379, 32, !dbg !11
  %385 = trunc i64 %384 to i32, !dbg !11
  %386 = insertelement <4 x i32> %383, i32 %385, i64 1, !dbg !11
  %387 = insertelement <4 x i32> %386, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc1746 = insertelement <4 x i32> %387, i32 131072, i64 3, !dbg !11
  %388 = ptrtoint ptr addrspace(3) %378 to i32, !dbg !11
  %dma.buffer.lds1747 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %388), !dbg !11
  %389 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1747, i32 %dma.buffer.offset1745, <4 x i32> %dma.buffer.rsrc1746) #10, !dbg !11
  %390 = add nuw i64 %61, 2048, !dbg !17
  %391 = icmp slt i64 %390, 0, !dbg !17
  %392 = sub nuw i64 -2049, %61, !dbg !17
  %393 = select i1 %391, i64 %392, i64 %390, !dbg !17
  %394 = sdiv i64 %393, 4096, !dbg !17
  %.lobit64 = ashr i64 %390, 63, !dbg !17
  %395 = xor i64 %394, %.lobit64, !dbg !17
  %396 = add nsw i64 %395, %15, !dbg !17
  %397 = add nsw i64 %245, 64, !dbg !17
  %398 = icmp slt i64 %245, -64, !dbg !17
  %399 = sub nsw i64 -65, %245, !dbg !17
  %400 = select i1 %398, i64 %399, i64 %397, !dbg !17
  %401 = sdiv i64 %400, 128, !dbg !17
  %402 = sext i1 %398 to i64, !dbg !17
  %403 = xor i64 %401, %402, !dbg !17
  %404 = shl nsw i64 %403, 7, !dbg !17
  %405 = sub i64 %55, %404, !dbg !17
  %406 = shl i64 %396, 19, !dbg !17
  %407 = and i64 %406, 16252928, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %408 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %407, !dbg !17
  %409 = getelementptr [2 x i8], ptr addrspace(1) %408, i64 %100, !dbg !17
  %410 = getelementptr [2 x i8], ptr addrspace(1) %409, i64 %405, !dbg !17
  %411 = getelementptr i8, ptr addrspace(1) %410, i64 128, !dbg !17
  %412 = getelementptr [2 x i8], ptr addrspace(1) %411, i64 %106, !dbg !17
  %413 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local535), !dbg !17
  %414 = zext i32 %413 to i64, !dbg !17
  %415 = inttoptr i64 %414 to ptr addrspace(3), !dbg !17
  %address.local927 = getelementptr [2 x i8], ptr addrspace(1) %412, i64 %address.local513, !dbg !17
  %416 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %417 = ptrtoint ptr addrspace(1) %address.local927 to i64, !dbg !17
  %418 = sub i64 %417, %416, !dbg !17
  %dma.buffer.offset1742 = trunc i64 %418 to i32, !dbg !17
  %419 = trunc i64 %416 to i32, !dbg !17
  %420 = insertelement <4 x i32> poison, i32 %419, i64 0, !dbg !17
  %421 = lshr i64 %416, 32, !dbg !17
  %422 = trunc i64 %421 to i32, !dbg !17
  %423 = insertelement <4 x i32> %420, i32 %422, i64 1, !dbg !17
  %424 = insertelement <4 x i32> %423, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc1743 = insertelement <4 x i32> %424, i32 131072, i64 3, !dbg !17
  %425 = ptrtoint ptr addrspace(3) %415 to i32, !dbg !17
  %dma.buffer.lds1744 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %425), !dbg !17
  %426 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1744, i32 %dma.buffer.offset1742, <4 x i32> %dma.buffer.rsrc1743) #10, !dbg !17
  %427 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local539), !dbg !17
  %428 = zext i32 %427 to i64, !dbg !17
  %429 = inttoptr i64 %428 to ptr addrspace(3), !dbg !17
  %address.local928 = getelementptr [2 x i8], ptr addrspace(1) %412, i64 %address.local531, !dbg !17
  %430 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %431 = ptrtoint ptr addrspace(1) %address.local928 to i64, !dbg !17
  %432 = sub i64 %431, %430, !dbg !17
  %dma.buffer.offset1739 = trunc i64 %432 to i32, !dbg !17
  %433 = trunc i64 %430 to i32, !dbg !17
  %434 = insertelement <4 x i32> poison, i32 %433, i64 0, !dbg !17
  %435 = lshr i64 %430, 32, !dbg !17
  %436 = trunc i64 %435 to i32, !dbg !17
  %437 = insertelement <4 x i32> %434, i32 %436, i64 1, !dbg !17
  %438 = insertelement <4 x i32> %437, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc1740 = insertelement <4 x i32> %438, i32 131072, i64 3, !dbg !17
  %439 = ptrtoint ptr addrspace(3) %429 to i32, !dbg !17
  %dma.buffer.lds1741 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %439), !dbg !17
  %440 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1741, i32 %dma.buffer.offset1739, <4 x i32> %dma.buffer.rsrc1740) #10, !dbg !17
  %441 = load <4 x half>, ptr addrspace(3) %address.local564, align 8, !dbg !12
  %lds.const.gep1644 = getelementptr i8, ptr addrspace(3) %address.local564, i64 4096, !dbg !12
  %442 = load <4 x half>, ptr addrspace(3) %lds.const.gep1644, align 8, !dbg !12
  %lds.const.gep1645 = getelementptr i8, ptr addrspace(3) %address.local564, i64 1024, !dbg !12
  %443 = load <4 x half>, ptr addrspace(3) %lds.const.gep1645, align 8, !dbg !12
  %444 = fmul <4 x half> %441, splat (half 0xH3015), !dbg !12
  %lds.const.gep1646 = getelementptr i8, ptr addrspace(3) %address.local564, i64 5120, !dbg !12
  %445 = load <4 x half>, ptr addrspace(3) %lds.const.gep1646, align 8, !dbg !12
  %446 = fmul <4 x half> %442, splat (half 0xH3015), !dbg !12
  %447 = fmul <4 x half> %443, splat (half 0xH3015), !dbg !12
  %448 = fmul <4 x half> %445, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %449 = load <4 x half>, ptr addrspace(3) %address.local583, align 8, !dbg !18
  %450 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %444, <4 x half> %449, <4 x float> %302) #9, !dbg !18
  %lds.const.gep1647 = getelementptr i8, ptr addrspace(3) %address.local583, i64 4096, !dbg !18
  %451 = load <4 x half>, ptr addrspace(3) %lds.const.gep1647, align 8, !dbg !18
  %452 = load <4 x half>, ptr addrspace(3) %address.local612, align 8, !dbg !18
  %453 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %446, <4 x half> %451, <4 x float> %450) #9, !dbg !18
  %454 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %444, <4 x half> %452, <4 x float> %306) #9, !dbg !18
  %lds.const.gep1648 = getelementptr i8, ptr addrspace(3) %address.local612, i64 4096, !dbg !18
  %455 = load <4 x half>, ptr addrspace(3) %lds.const.gep1648, align 8, !dbg !18
  %lds.const.gep1649 = getelementptr i8, ptr addrspace(3) %address.local583, i64 128, !dbg !18
  %456 = load <4 x half>, ptr addrspace(3) %lds.const.gep1649, align 8, !dbg !18
  %457 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %446, <4 x half> %455, <4 x float> %454) #9, !dbg !18
  %458 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %444, <4 x half> %456, <4 x float> %310) #9, !dbg !18
  %lds.const.gep1650 = getelementptr i8, ptr addrspace(3) %address.local583, i64 4224, !dbg !18
  %459 = load <4 x half>, ptr addrspace(3) %lds.const.gep1650, align 8, !dbg !18
  %lds.const.gep1651 = getelementptr i8, ptr addrspace(3) %address.local612, i64 128, !dbg !18
  %460 = load <4 x half>, ptr addrspace(3) %lds.const.gep1651, align 8, !dbg !18
  %461 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %446, <4 x half> %459, <4 x float> %458) #9, !dbg !18
  %462 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %444, <4 x half> %460, <4 x float> %314) #9, !dbg !18
  %lds.const.gep1652 = getelementptr i8, ptr addrspace(3) %address.local612, i64 4224, !dbg !18
  %463 = load <4 x half>, ptr addrspace(3) %lds.const.gep1652, align 8, !dbg !18
  %lds.const.gep1653 = getelementptr i8, ptr addrspace(3) %address.local583, i64 1024, !dbg !18
  %464 = load <4 x half>, ptr addrspace(3) %lds.const.gep1653, align 8, !dbg !18
  %465 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %446, <4 x half> %463, <4 x float> %462) #9, !dbg !18
  %466 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %444, <4 x half> %464, <4 x float> %318) #9, !dbg !18
  %lds.const.gep1654 = getelementptr i8, ptr addrspace(3) %address.local583, i64 5120, !dbg !18
  %467 = load <4 x half>, ptr addrspace(3) %lds.const.gep1654, align 8, !dbg !18
  %lds.const.gep1655 = getelementptr i8, ptr addrspace(3) %address.local612, i64 1024, !dbg !18
  %468 = load <4 x half>, ptr addrspace(3) %lds.const.gep1655, align 8, !dbg !18
  %469 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %446, <4 x half> %467, <4 x float> %466) #9, !dbg !18
  %470 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %444, <4 x half> %468, <4 x float> %322) #9, !dbg !18
  %lds.const.gep1656 = getelementptr i8, ptr addrspace(3) %address.local612, i64 5120, !dbg !18
  %471 = load <4 x half>, ptr addrspace(3) %lds.const.gep1656, align 8, !dbg !18
  %lds.const.gep1657 = getelementptr i8, ptr addrspace(3) %address.local583, i64 1152, !dbg !18
  %472 = load <4 x half>, ptr addrspace(3) %lds.const.gep1657, align 8, !dbg !18
  %473 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %446, <4 x half> %471, <4 x float> %470) #9, !dbg !18
  %474 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %444, <4 x half> %472, <4 x float> %326) #9, !dbg !18
  %lds.const.gep1658 = getelementptr i8, ptr addrspace(3) %address.local583, i64 5248, !dbg !18
  %475 = load <4 x half>, ptr addrspace(3) %lds.const.gep1658, align 8, !dbg !18
  %lds.const.gep1659 = getelementptr i8, ptr addrspace(3) %address.local612, i64 1152, !dbg !18
  %476 = load <4 x half>, ptr addrspace(3) %lds.const.gep1659, align 8, !dbg !18
  %477 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %446, <4 x half> %475, <4 x float> %474) #9, !dbg !18
  %478 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %444, <4 x half> %476, <4 x float> %330) #9, !dbg !18
  %lds.const.gep1660 = getelementptr i8, ptr addrspace(3) %address.local612, i64 5248, !dbg !18
  %479 = load <4 x half>, ptr addrspace(3) %lds.const.gep1660, align 8, !dbg !18
  %480 = load <4 x half>, ptr addrspace(3) %address.local583, align 8, !dbg !18
  %481 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %446, <4 x half> %479, <4 x float> %478) #9, !dbg !18
  %482 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %447, <4 x half> %480, <4 x float> %334) #9, !dbg !18
  %lds.const.gep1661 = getelementptr i8, ptr addrspace(3) %address.local583, i64 4096, !dbg !18
  %483 = load <4 x half>, ptr addrspace(3) %lds.const.gep1661, align 8, !dbg !18
  %484 = load <4 x half>, ptr addrspace(3) %address.local612, align 8, !dbg !18
  %485 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %448, <4 x half> %483, <4 x float> %482) #9, !dbg !18
  %486 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %447, <4 x half> %484, <4 x float> %338) #9, !dbg !18
  %lds.const.gep1662 = getelementptr i8, ptr addrspace(3) %address.local612, i64 4096, !dbg !18
  %487 = load <4 x half>, ptr addrspace(3) %lds.const.gep1662, align 8, !dbg !18
  %lds.const.gep1663 = getelementptr i8, ptr addrspace(3) %address.local583, i64 128, !dbg !18
  %488 = load <4 x half>, ptr addrspace(3) %lds.const.gep1663, align 8, !dbg !18
  %489 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %448, <4 x half> %487, <4 x float> %486) #9, !dbg !18
  %490 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %447, <4 x half> %488, <4 x float> %342) #9, !dbg !18
  %lds.const.gep1664 = getelementptr i8, ptr addrspace(3) %address.local583, i64 4224, !dbg !18
  %491 = load <4 x half>, ptr addrspace(3) %lds.const.gep1664, align 8, !dbg !18
  %lds.const.gep1665 = getelementptr i8, ptr addrspace(3) %address.local612, i64 128, !dbg !18
  %492 = load <4 x half>, ptr addrspace(3) %lds.const.gep1665, align 8, !dbg !18
  %493 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %448, <4 x half> %491, <4 x float> %490) #9, !dbg !18
  %494 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %447, <4 x half> %492, <4 x float> %346) #9, !dbg !18
  %lds.const.gep1666 = getelementptr i8, ptr addrspace(3) %address.local612, i64 4224, !dbg !18
  %495 = load <4 x half>, ptr addrspace(3) %lds.const.gep1666, align 8, !dbg !18
  %lds.const.gep1667 = getelementptr i8, ptr addrspace(3) %address.local583, i64 1024, !dbg !18
  %496 = load <4 x half>, ptr addrspace(3) %lds.const.gep1667, align 8, !dbg !18
  %497 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %448, <4 x half> %495, <4 x float> %494) #9, !dbg !18
  %498 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %447, <4 x half> %496, <4 x float> %350) #9, !dbg !18
  %lds.const.gep1668 = getelementptr i8, ptr addrspace(3) %address.local583, i64 5120, !dbg !18
  %499 = load <4 x half>, ptr addrspace(3) %lds.const.gep1668, align 8, !dbg !18
  %lds.const.gep1669 = getelementptr i8, ptr addrspace(3) %address.local612, i64 1024, !dbg !18
  %500 = load <4 x half>, ptr addrspace(3) %lds.const.gep1669, align 8, !dbg !18
  %501 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %448, <4 x half> %499, <4 x float> %498) #9, !dbg !18
  %502 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %447, <4 x half> %500, <4 x float> %354) #9, !dbg !18
  %lds.const.gep1670 = getelementptr i8, ptr addrspace(3) %address.local612, i64 5120, !dbg !18
  %503 = load <4 x half>, ptr addrspace(3) %lds.const.gep1670, align 8, !dbg !18
  %lds.const.gep1671 = getelementptr i8, ptr addrspace(3) %address.local583, i64 1152, !dbg !18
  %504 = load <4 x half>, ptr addrspace(3) %lds.const.gep1671, align 8, !dbg !18
  %505 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %448, <4 x half> %503, <4 x float> %502) #9, !dbg !18
  %506 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %447, <4 x half> %504, <4 x float> %358) #9, !dbg !18
  %lds.const.gep1672 = getelementptr i8, ptr addrspace(3) %address.local583, i64 5248, !dbg !18
  %507 = load <4 x half>, ptr addrspace(3) %lds.const.gep1672, align 8, !dbg !18
  %lds.const.gep1673 = getelementptr i8, ptr addrspace(3) %address.local612, i64 1152, !dbg !18
  %508 = load <4 x half>, ptr addrspace(3) %lds.const.gep1673, align 8, !dbg !18
  %509 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %448, <4 x half> %507, <4 x float> %506) #9, !dbg !18
  %lds.const.gep1674 = getelementptr i8, ptr addrspace(3) %address.local612, i64 5248, !dbg !18
  %510 = load <4 x half>, ptr addrspace(3) %lds.const.gep1674, align 8, !dbg !18
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !18
  %511 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %447, <4 x half> %508, <4 x float> %361) #9, !dbg !18
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !18
  %512 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %448, <4 x half> %510, <4 x float> %511) #9, !dbg !18
  %513 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local502), !dbg !11
  %514 = zext i32 %513 to i64, !dbg !11
  %515 = inttoptr i64 %514 to ptr addrspace(3), !dbg !11
  %address.local929 = getelementptr [2 x i8], ptr addrspace(1) %59, i64 %address.local513, !dbg !11
  %516 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %517 = ptrtoint ptr addrspace(1) %address.local929 to i64, !dbg !11
  %518 = sub i64 %517, %516, !dbg !11
  %dma.buffer.offset1736 = trunc i64 %518 to i32, !dbg !11
  %519 = trunc i64 %516 to i32, !dbg !11
  %520 = insertelement <4 x i32> poison, i32 %519, i64 0, !dbg !11
  %521 = lshr i64 %516, 32, !dbg !11
  %522 = trunc i64 %521 to i32, !dbg !11
  %523 = insertelement <4 x i32> %520, i32 %522, i64 1, !dbg !11
  %524 = insertelement <4 x i32> %523, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc1737 = insertelement <4 x i32> %524, i32 131072, i64 3, !dbg !11
  %525 = ptrtoint ptr addrspace(3) %515 to i32, !dbg !11
  %dma.buffer.lds1738 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %525), !dbg !11
  %526 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1738, i32 %dma.buffer.offset1736, <4 x i32> %dma.buffer.rsrc1737) #10, !dbg !11
  %527 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local520), !dbg !11
  %528 = zext i32 %527 to i64, !dbg !11
  %529 = inttoptr i64 %528 to ptr addrspace(3), !dbg !11
  %address.local930 = getelementptr [2 x i8], ptr addrspace(1) %59, i64 %address.local531, !dbg !11
  %530 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %531 = ptrtoint ptr addrspace(1) %address.local930 to i64, !dbg !11
  %532 = sub i64 %531, %530, !dbg !11
  %dma.buffer.offset1733 = trunc i64 %532 to i32, !dbg !11
  %533 = trunc i64 %530 to i32, !dbg !11
  %534 = insertelement <4 x i32> poison, i32 %533, i64 0, !dbg !11
  %535 = lshr i64 %530, 32, !dbg !11
  %536 = trunc i64 %535 to i32, !dbg !11
  %537 = insertelement <4 x i32> %534, i32 %536, i64 1, !dbg !11
  %538 = insertelement <4 x i32> %537, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc1734 = insertelement <4 x i32> %538, i32 131072, i64 3, !dbg !11
  %539 = ptrtoint ptr addrspace(3) %529 to i32, !dbg !11
  %dma.buffer.lds1735 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %539), !dbg !11
  %540 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1735, i32 %dma.buffer.offset1733, <4 x i32> %dma.buffer.rsrc1734) #10, !dbg !11
  %541 = add nuw i64 %61, 3072, !dbg !17
  %542 = icmp slt i64 %541, 0, !dbg !17
  %543 = sub nuw i64 -3073, %61, !dbg !17
  %544 = select i1 %542, i64 %543, i64 %541, !dbg !17
  %545 = sdiv i64 %544, 4096, !dbg !17
  %.lobit65 = ashr i64 %541, 63, !dbg !17
  %546 = xor i64 %545, %.lobit65, !dbg !17
  %547 = add nsw i64 %546, %15, !dbg !17
  %548 = add nsw i64 %245, 96, !dbg !17
  %549 = icmp slt i64 %245, -96, !dbg !17
  %550 = sub nsw i64 -97, %245, !dbg !17
  %551 = select i1 %549, i64 %550, i64 %548, !dbg !17
  %552 = sdiv i64 %551, 128, !dbg !17
  %553 = sext i1 %549 to i64, !dbg !17
  %554 = xor i64 %552, %553, !dbg !17
  %555 = shl nsw i64 %554, 7, !dbg !17
  %556 = sub i64 %55, %555, !dbg !17
  %557 = shl i64 %547, 19, !dbg !17
  %558 = and i64 %557, 16252928, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %559 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %558, !dbg !17
  %560 = getelementptr [2 x i8], ptr addrspace(1) %559, i64 %100, !dbg !17
  %561 = getelementptr [2 x i8], ptr addrspace(1) %560, i64 %556, !dbg !17
  %562 = getelementptr i8, ptr addrspace(1) %561, i64 192, !dbg !17
  %563 = getelementptr [2 x i8], ptr addrspace(1) %562, i64 %106, !dbg !17
  %564 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local535), !dbg !17
  %565 = zext i32 %564 to i64, !dbg !17
  %566 = inttoptr i64 %565 to ptr addrspace(3), !dbg !17
  %address.local931 = getelementptr [2 x i8], ptr addrspace(1) %563, i64 %address.local513, !dbg !17
  %567 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %568 = ptrtoint ptr addrspace(1) %address.local931 to i64, !dbg !17
  %569 = sub i64 %568, %567, !dbg !17
  %dma.buffer.offset1730 = trunc i64 %569 to i32, !dbg !17
  %570 = trunc i64 %567 to i32, !dbg !17
  %571 = insertelement <4 x i32> poison, i32 %570, i64 0, !dbg !17
  %572 = lshr i64 %567, 32, !dbg !17
  %573 = trunc i64 %572 to i32, !dbg !17
  %574 = insertelement <4 x i32> %571, i32 %573, i64 1, !dbg !17
  %575 = insertelement <4 x i32> %574, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc1731 = insertelement <4 x i32> %575, i32 131072, i64 3, !dbg !17
  %576 = ptrtoint ptr addrspace(3) %566 to i32, !dbg !17
  %dma.buffer.lds1732 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %576), !dbg !17
  %577 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1732, i32 %dma.buffer.offset1730, <4 x i32> %dma.buffer.rsrc1731) #10, !dbg !17
  %578 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local539), !dbg !17
  %579 = zext i32 %578 to i64, !dbg !17
  %580 = inttoptr i64 %579 to ptr addrspace(3), !dbg !17
  %address.local932 = getelementptr [2 x i8], ptr addrspace(1) %563, i64 %address.local531, !dbg !17
  %581 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %582 = ptrtoint ptr addrspace(1) %address.local932 to i64, !dbg !17
  %583 = sub i64 %582, %581, !dbg !17
  %dma.buffer.offset1727 = trunc i64 %583 to i32, !dbg !17
  %584 = trunc i64 %581 to i32, !dbg !17
  %585 = insertelement <4 x i32> poison, i32 %584, i64 0, !dbg !17
  %586 = lshr i64 %581, 32, !dbg !17
  %587 = trunc i64 %586 to i32, !dbg !17
  %588 = insertelement <4 x i32> %585, i32 %587, i64 1, !dbg !17
  %589 = insertelement <4 x i32> %588, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc1728 = insertelement <4 x i32> %589, i32 131072, i64 3, !dbg !17
  %590 = ptrtoint ptr addrspace(3) %580 to i32, !dbg !17
  %dma.buffer.lds1729 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %590), !dbg !17
  %591 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1729, i32 %dma.buffer.offset1727, <4 x i32> %dma.buffer.rsrc1728) #10, !dbg !17
  %592 = load <4 x half>, ptr addrspace(3) %address.local564, align 8, !dbg !12
  %lds.const.gep1675 = getelementptr i8, ptr addrspace(3) %address.local564, i64 4096, !dbg !12
  %593 = load <4 x half>, ptr addrspace(3) %lds.const.gep1675, align 8, !dbg !12
  %lds.const.gep1676 = getelementptr i8, ptr addrspace(3) %address.local564, i64 1024, !dbg !12
  %594 = load <4 x half>, ptr addrspace(3) %lds.const.gep1676, align 8, !dbg !12
  %595 = fmul <4 x half> %592, splat (half 0xH3015), !dbg !12
  %lds.const.gep1677 = getelementptr i8, ptr addrspace(3) %address.local564, i64 5120, !dbg !12
  %596 = load <4 x half>, ptr addrspace(3) %lds.const.gep1677, align 8, !dbg !12
  %597 = fmul <4 x half> %593, splat (half 0xH3015), !dbg !12
  %598 = fmul <4 x half> %594, splat (half 0xH3015), !dbg !12
  %599 = fmul <4 x half> %596, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %600 = add nuw nsw i64 %100, %15, !dbg !19
  %601 = shl i64 %600, 19, !dbg !19
  %602 = and i64 %601, 16252928, !dbg !19
  %603 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %602, !dbg !19
  %604 = getelementptr [2 x i8], ptr addrspace(1) %603, i64 %106, !dbg !19
  %605 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local502), !dbg !19
  %606 = zext i32 %605 to i64, !dbg !19
  %607 = inttoptr i64 %606 to ptr addrspace(3), !dbg !19
  %address.local933 = shl nuw nsw i64 %address.local508, 5, !dbg !19
  %address.local934 = and i64 %address.local933, 8064, !dbg !19
  %address.local935 = shl nuw nsw i64 %address.local, 3, !dbg !19
  %address.local936 = and i64 %address.local935, 24, !dbg !19
  %address.local937 = or disjoint i64 %address.local934, %address.local936, !dbg !19
  %address.local938 = getelementptr [2 x i8], ptr addrspace(1) %604, i64 %address.local937, !dbg !19
  %608 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !19
  %609 = ptrtoint ptr addrspace(1) %address.local938 to i64, !dbg !19
  %610 = sub i64 %609, %608, !dbg !19
  %dma.buffer.offset1724 = trunc i64 %610 to i32, !dbg !19
  %611 = trunc i64 %608 to i32, !dbg !19
  %612 = insertelement <4 x i32> poison, i32 %611, i64 0, !dbg !19
  %613 = lshr i64 %608, 32, !dbg !19
  %614 = trunc i64 %613 to i32, !dbg !19
  %615 = insertelement <4 x i32> %612, i32 %614, i64 1, !dbg !19
  %616 = insertelement <4 x i32> %615, i32 33554432, i64 2, !dbg !19
  %dma.buffer.rsrc1725 = insertelement <4 x i32> %616, i32 131072, i64 3, !dbg !19
  %617 = ptrtoint ptr addrspace(3) %607 to i32, !dbg !19
  %dma.buffer.lds1726 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %617), !dbg !19
  %618 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1726, i32 %dma.buffer.offset1724, <4 x i32> %dma.buffer.rsrc1725) #10, !dbg !19
  %619 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local520), !dbg !19
  %620 = zext i32 %619 to i64, !dbg !19
  %621 = inttoptr i64 %620 to ptr addrspace(3), !dbg !19
  %address.local939 = shl nuw nsw i64 %address.local526, 5, !dbg !19
  %address.local940 = and i64 %address.local939, 16256, !dbg !19
  %address.local941 = shl nuw nsw i64 %address.local, 3, !dbg !19
  %address.local942 = and i64 %address.local941, 24, !dbg !19
  %address.local943 = or disjoint i64 %address.local940, %address.local942, !dbg !19
  %address.local944 = getelementptr [2 x i8], ptr addrspace(1) %604, i64 %address.local943, !dbg !19
  %622 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !19
  %623 = ptrtoint ptr addrspace(1) %address.local944 to i64, !dbg !19
  %624 = sub i64 %623, %622, !dbg !19
  %dma.buffer.offset1721 = trunc i64 %624 to i32, !dbg !19
  %625 = trunc i64 %622 to i32, !dbg !19
  %626 = insertelement <4 x i32> poison, i32 %625, i64 0, !dbg !19
  %627 = lshr i64 %622, 32, !dbg !19
  %628 = trunc i64 %627 to i32, !dbg !19
  %629 = insertelement <4 x i32> %626, i32 %628, i64 1, !dbg !19
  %630 = insertelement <4 x i32> %629, i32 33554432, i64 2, !dbg !19
  %dma.buffer.rsrc1722 = insertelement <4 x i32> %630, i32 131072, i64 3, !dbg !19
  %631 = ptrtoint ptr addrspace(3) %621 to i32, !dbg !19
  %dma.buffer.lds1723 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %631), !dbg !19
  %632 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1723, i32 %dma.buffer.offset1721, <4 x i32> %dma.buffer.rsrc1722) #10, !dbg !19
  %633 = load <4 x half>, ptr addrspace(3) %address.local583, align 8, !dbg !18
  %634 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %595, <4 x half> %633, <4 x float> %453) #9, !dbg !18
  %lds.const.gep1678 = getelementptr i8, ptr addrspace(3) %address.local583, i64 4096, !dbg !18
  %635 = load <4 x half>, ptr addrspace(3) %lds.const.gep1678, align 8, !dbg !18
  %636 = load <4 x half>, ptr addrspace(3) %address.local612, align 8, !dbg !18
  %637 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %597, <4 x half> %635, <4 x float> %634) #9, !dbg !18
  %638 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %595, <4 x half> %636, <4 x float> %457) #9, !dbg !18
  %lds.const.gep1679 = getelementptr i8, ptr addrspace(3) %address.local612, i64 4096, !dbg !18
  %639 = load <4 x half>, ptr addrspace(3) %lds.const.gep1679, align 8, !dbg !18
  %lds.const.gep1680 = getelementptr i8, ptr addrspace(3) %address.local583, i64 128, !dbg !18
  %640 = load <4 x half>, ptr addrspace(3) %lds.const.gep1680, align 8, !dbg !18
  %641 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %597, <4 x half> %639, <4 x float> %638) #9, !dbg !18
  %642 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %595, <4 x half> %640, <4 x float> %461) #9, !dbg !18
  %lds.const.gep1681 = getelementptr i8, ptr addrspace(3) %address.local583, i64 4224, !dbg !18
  %643 = load <4 x half>, ptr addrspace(3) %lds.const.gep1681, align 8, !dbg !18
  %lds.const.gep1682 = getelementptr i8, ptr addrspace(3) %address.local612, i64 128, !dbg !18
  %644 = load <4 x half>, ptr addrspace(3) %lds.const.gep1682, align 8, !dbg !18
  %645 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %597, <4 x half> %643, <4 x float> %642) #9, !dbg !18
  %646 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %595, <4 x half> %644, <4 x float> %465) #9, !dbg !18
  %lds.const.gep1683 = getelementptr i8, ptr addrspace(3) %address.local612, i64 4224, !dbg !18
  %647 = load <4 x half>, ptr addrspace(3) %lds.const.gep1683, align 8, !dbg !18
  %lds.const.gep1684 = getelementptr i8, ptr addrspace(3) %address.local583, i64 1024, !dbg !18
  %648 = load <4 x half>, ptr addrspace(3) %lds.const.gep1684, align 8, !dbg !18
  %649 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %597, <4 x half> %647, <4 x float> %646) #9, !dbg !18
  %650 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %595, <4 x half> %648, <4 x float> %469) #9, !dbg !18
  %lds.const.gep1685 = getelementptr i8, ptr addrspace(3) %address.local583, i64 5120, !dbg !18
  %651 = load <4 x half>, ptr addrspace(3) %lds.const.gep1685, align 8, !dbg !18
  %lds.const.gep1686 = getelementptr i8, ptr addrspace(3) %address.local612, i64 1024, !dbg !18
  %652 = load <4 x half>, ptr addrspace(3) %lds.const.gep1686, align 8, !dbg !18
  %653 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %597, <4 x half> %651, <4 x float> %650) #9, !dbg !18
  %654 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %595, <4 x half> %652, <4 x float> %473) #9, !dbg !18
  %lds.const.gep1687 = getelementptr i8, ptr addrspace(3) %address.local612, i64 5120, !dbg !18
  %655 = load <4 x half>, ptr addrspace(3) %lds.const.gep1687, align 8, !dbg !18
  %lds.const.gep1688 = getelementptr i8, ptr addrspace(3) %address.local583, i64 1152, !dbg !18
  %656 = load <4 x half>, ptr addrspace(3) %lds.const.gep1688, align 8, !dbg !18
  %657 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %597, <4 x half> %655, <4 x float> %654) #9, !dbg !18
  %658 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %595, <4 x half> %656, <4 x float> %477) #9, !dbg !18
  %lds.const.gep1689 = getelementptr i8, ptr addrspace(3) %address.local583, i64 5248, !dbg !18
  %659 = load <4 x half>, ptr addrspace(3) %lds.const.gep1689, align 8, !dbg !18
  %lds.const.gep1690 = getelementptr i8, ptr addrspace(3) %address.local612, i64 1152, !dbg !18
  %660 = load <4 x half>, ptr addrspace(3) %lds.const.gep1690, align 8, !dbg !18
  %661 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %597, <4 x half> %659, <4 x float> %658) #9, !dbg !18
  %662 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %595, <4 x half> %660, <4 x float> %481) #9, !dbg !18
  %lds.const.gep1691 = getelementptr i8, ptr addrspace(3) %address.local612, i64 5248, !dbg !18
  %663 = load <4 x half>, ptr addrspace(3) %lds.const.gep1691, align 8, !dbg !18
  %664 = load <4 x half>, ptr addrspace(3) %address.local583, align 8, !dbg !18
  %665 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %597, <4 x half> %663, <4 x float> %662) #9, !dbg !18
  %666 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %598, <4 x half> %664, <4 x float> %485) #9, !dbg !18
  %lds.const.gep1692 = getelementptr i8, ptr addrspace(3) %address.local583, i64 4096, !dbg !18
  %667 = load <4 x half>, ptr addrspace(3) %lds.const.gep1692, align 8, !dbg !18
  %668 = load <4 x half>, ptr addrspace(3) %address.local612, align 8, !dbg !18
  %669 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %599, <4 x half> %667, <4 x float> %666) #9, !dbg !18
  %670 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %598, <4 x half> %668, <4 x float> %489) #9, !dbg !18
  %lds.const.gep1693 = getelementptr i8, ptr addrspace(3) %address.local612, i64 4096, !dbg !18
  %671 = load <4 x half>, ptr addrspace(3) %lds.const.gep1693, align 8, !dbg !18
  %lds.const.gep1694 = getelementptr i8, ptr addrspace(3) %address.local583, i64 128, !dbg !18
  %672 = load <4 x half>, ptr addrspace(3) %lds.const.gep1694, align 8, !dbg !18
  %673 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %599, <4 x half> %671, <4 x float> %670) #9, !dbg !18
  %674 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %598, <4 x half> %672, <4 x float> %493) #9, !dbg !18
  %lds.const.gep1695 = getelementptr i8, ptr addrspace(3) %address.local583, i64 4224, !dbg !18
  %675 = load <4 x half>, ptr addrspace(3) %lds.const.gep1695, align 8, !dbg !18
  %lds.const.gep1696 = getelementptr i8, ptr addrspace(3) %address.local612, i64 128, !dbg !18
  %676 = load <4 x half>, ptr addrspace(3) %lds.const.gep1696, align 8, !dbg !18
  %677 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %599, <4 x half> %675, <4 x float> %674) #9, !dbg !18
  %678 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %598, <4 x half> %676, <4 x float> %497) #9, !dbg !18
  %lds.const.gep1697 = getelementptr i8, ptr addrspace(3) %address.local612, i64 4224, !dbg !18
  %679 = load <4 x half>, ptr addrspace(3) %lds.const.gep1697, align 8, !dbg !18
  %lds.const.gep1698 = getelementptr i8, ptr addrspace(3) %address.local583, i64 1024, !dbg !18
  %680 = load <4 x half>, ptr addrspace(3) %lds.const.gep1698, align 8, !dbg !18
  %681 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %599, <4 x half> %679, <4 x float> %678) #9, !dbg !18
  %682 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %598, <4 x half> %680, <4 x float> %501) #9, !dbg !18
  %lds.const.gep1699 = getelementptr i8, ptr addrspace(3) %address.local583, i64 5120, !dbg !18
  %683 = load <4 x half>, ptr addrspace(3) %lds.const.gep1699, align 8, !dbg !18
  %lds.const.gep1700 = getelementptr i8, ptr addrspace(3) %address.local612, i64 1024, !dbg !18
  %684 = load <4 x half>, ptr addrspace(3) %lds.const.gep1700, align 8, !dbg !18
  %685 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %599, <4 x half> %683, <4 x float> %682) #9, !dbg !18
  %686 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %598, <4 x half> %684, <4 x float> %505) #9, !dbg !18
  %lds.const.gep1701 = getelementptr i8, ptr addrspace(3) %address.local612, i64 5120, !dbg !18
  %687 = load <4 x half>, ptr addrspace(3) %lds.const.gep1701, align 8, !dbg !18
  %lds.const.gep1702 = getelementptr i8, ptr addrspace(3) %address.local583, i64 1152, !dbg !18
  %688 = load <4 x half>, ptr addrspace(3) %lds.const.gep1702, align 8, !dbg !18
  %689 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %599, <4 x half> %687, <4 x float> %686) #9, !dbg !18
  %690 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %598, <4 x half> %688, <4 x float> %509) #9, !dbg !18
  %lds.const.gep1703 = getelementptr i8, ptr addrspace(3) %address.local583, i64 5248, !dbg !18
  %691 = load <4 x half>, ptr addrspace(3) %lds.const.gep1703, align 8, !dbg !18
  %lds.const.gep1704 = getelementptr i8, ptr addrspace(3) %address.local612, i64 1152, !dbg !18
  %692 = load <4 x half>, ptr addrspace(3) %lds.const.gep1704, align 8, !dbg !18
  %693 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %599, <4 x half> %691, <4 x float> %690) #9, !dbg !18
  %lds.const.gep1705 = getelementptr i8, ptr addrspace(3) %address.local612, i64 5248, !dbg !18
  %694 = load <4 x half>, ptr addrspace(3) %lds.const.gep1705, align 8, !dbg !18
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !18
  %695 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %598, <4 x half> %692, <4 x float> %512) #9, !dbg !18
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %696 = or disjoint i64 %69, 127, !dbg !13
  %697 = icmp uge i64 %19, %696, !dbg !13
  %698 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %599, <4 x half> %694, <4 x float> %695) #9, !dbg !18
  %699 = and i1 %40, %697, !dbg !13
  br i1 %699, label %700, label %701, !dbg !13

700:                                              ; preds = %68
  br label %845, !dbg !13

701:                                              ; preds = %68
  %702 = add i64 %19, %35, !dbg !13
  %703 = shl nuw nsw i64 %36, 4, !dbg !13
  %704 = add i64 %702, %703, !dbg !13
  %705 = and i32 %34, 240, !dbg !13
  %index.compact = zext nneg i32 %705 to i64, !dbg !13
  %706 = sub i64 %704, %index.compact, !dbg !13
  %707 = and i64 %35, 48, !dbg !13
  %708 = or disjoint i64 %69, %707, !dbg !13
  %709 = add i64 %706, 1, !dbg !24
  %710 = lshr i64 %709, 32, !dbg !25
  %compare.high = trunc i64 %710 to i32, !dbg !25
  %711 = lshr i64 %708, 32, !dbg !25
  %compare.high1327 = trunc i64 %711 to i32, !dbg !25
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high1327, !dbg !25
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high1327, !dbg !25
  %compare.low.base = trunc i64 %709 to i32, !dbg !25
  %compare.low.base1328 = trunc i64 %708 to i32, !dbg !25
  %compare.low = icmp ugt i32 %compare.low.base, %compare.low.base1328, !dbg !25
  %compare.result = select i1 %compare.high.equal, i1 %compare.low, i1 %compare.high.order, !dbg !25
  %712 = select i1 %compare.result, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1328, i32 1) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local = or i32 %compare.local.base, 1, !dbg !25
  %compare.low1329 = icmp ugt i32 %compare.low.base, %compare.local, !dbg !25
  %compare.result1330 = select i1 %compare.high.equal, i1 %compare.low1329, i1 %compare.high.order, !dbg !25
  %713 = select i1 %compare.result1330, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1331 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1328, i32 2) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1332 = or i32 %compare.local.base1331, 2, !dbg !25
  %compare.low1333 = icmp ugt i32 %compare.low.base, %compare.local1332, !dbg !25
  %compare.result1334 = select i1 %compare.high.equal, i1 %compare.low1333, i1 %compare.high.order, !dbg !25
  %714 = select i1 %compare.result1334, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1335 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1328, i32 3) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1336 = or i32 %compare.local.base1335, 3, !dbg !25
  %compare.low1337 = icmp ugt i32 %compare.low.base, %compare.local1336, !dbg !25
  %compare.result1338 = select i1 %compare.high.equal, i1 %compare.low1337, i1 %compare.high.order, !dbg !25
  %715 = select i1 %compare.result1338, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1339 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1328, i32 4) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1340 = or i32 %compare.local.base1339, 4, !dbg !25
  %compare.low1341 = icmp ugt i32 %compare.low.base, %compare.local1340, !dbg !25
  %compare.result1342 = select i1 %compare.high.equal, i1 %compare.low1341, i1 %compare.high.order, !dbg !25
  %716 = select i1 %compare.result1342, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1343 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1328, i32 5) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1344 = or i32 %compare.local.base1343, 5, !dbg !25
  %compare.low1345 = icmp ugt i32 %compare.low.base, %compare.local1344, !dbg !25
  %compare.result1346 = select i1 %compare.high.equal, i1 %compare.low1345, i1 %compare.high.order, !dbg !25
  %717 = select i1 %compare.result1346, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1347 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1328, i32 6) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1348 = or i32 %compare.local.base1347, 6, !dbg !25
  %compare.low1349 = icmp ugt i32 %compare.low.base, %compare.local1348, !dbg !25
  %compare.result1350 = select i1 %compare.high.equal, i1 %compare.low1349, i1 %compare.high.order, !dbg !25
  %718 = select i1 %compare.result1350, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1351 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1328, i32 7) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1352 = or i32 %compare.local.base1351, 7, !dbg !25
  %compare.low1353 = icmp ugt i32 %compare.low.base, %compare.local1352, !dbg !25
  %compare.result1354 = select i1 %compare.high.equal, i1 %compare.low1353, i1 %compare.high.order, !dbg !25
  %719 = select i1 %compare.result1354, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1355 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1328, i32 8) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1356 = or i32 %compare.local.base1355, 8, !dbg !25
  %compare.low1357 = icmp ugt i32 %compare.low.base, %compare.local1356, !dbg !25
  %compare.result1358 = select i1 %compare.high.equal, i1 %compare.low1357, i1 %compare.high.order, !dbg !25
  %720 = select i1 %compare.result1358, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1359 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1328, i32 9) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1360 = or i32 %compare.local.base1359, 9, !dbg !25
  %compare.low1361 = icmp ugt i32 %compare.low.base, %compare.local1360, !dbg !25
  %compare.result1362 = select i1 %compare.high.equal, i1 %compare.low1361, i1 %compare.high.order, !dbg !25
  %721 = select i1 %compare.result1362, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1363 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1328, i32 10) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1364 = or i32 %compare.local.base1363, 10, !dbg !25
  %compare.low1365 = icmp ugt i32 %compare.low.base, %compare.local1364, !dbg !25
  %compare.result1366 = select i1 %compare.high.equal, i1 %compare.low1365, i1 %compare.high.order, !dbg !25
  %722 = select i1 %compare.result1366, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1367 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1328, i32 11) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1368 = or i32 %compare.local.base1367, 11, !dbg !25
  %compare.low1369 = icmp ugt i32 %compare.low.base, %compare.local1368, !dbg !25
  %compare.result1370 = select i1 %compare.high.equal, i1 %compare.low1369, i1 %compare.high.order, !dbg !25
  %723 = select i1 %compare.result1370, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1371 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1328, i32 12) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1372 = or i32 %compare.local.base1371, 12, !dbg !25
  %compare.low1373 = icmp ugt i32 %compare.low.base, %compare.local1372, !dbg !25
  %compare.result1374 = select i1 %compare.high.equal, i1 %compare.low1373, i1 %compare.high.order, !dbg !25
  %724 = select i1 %compare.result1374, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1375 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1328, i32 13) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1376 = or i32 %compare.local.base1375, 13, !dbg !25
  %compare.low1377 = icmp ugt i32 %compare.low.base, %compare.local1376, !dbg !25
  %compare.result1378 = select i1 %compare.high.equal, i1 %compare.low1377, i1 %compare.high.order, !dbg !25
  %725 = select i1 %compare.result1378, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1379 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1328, i32 14) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1380 = or i32 %compare.local.base1379, 14, !dbg !25
  %compare.low1381 = icmp ugt i32 %compare.low.base, %compare.local1380, !dbg !25
  %compare.result1382 = select i1 %compare.high.equal, i1 %compare.low1381, i1 %compare.high.order, !dbg !25
  %726 = select i1 %compare.result1382, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1383 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1328, i32 15) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1384 = or i32 %compare.local.base1383, 15, !dbg !25
  %compare.low1385 = icmp ugt i32 %compare.low.base, %compare.local1384, !dbg !25
  %compare.result1386 = select i1 %compare.high.equal, i1 %compare.low1385, i1 %compare.high.order, !dbg !25
  %727 = select i1 %compare.result1386, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1387 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1328, i32 16) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1388 = or i32 %compare.local.base1387, 64, !dbg !25
  %compare.low1389 = icmp ugt i32 %compare.low.base, %compare.local1388, !dbg !25
  %compare.result1390 = select i1 %compare.high.equal, i1 %compare.low1389, i1 %compare.high.order, !dbg !25
  %728 = select i1 %compare.result1390, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1391 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1328, i32 17) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1392 = or i32 %compare.local.base1391, 65, !dbg !25
  %compare.low1393 = icmp ugt i32 %compare.low.base, %compare.local1392, !dbg !25
  %compare.result1394 = select i1 %compare.high.equal, i1 %compare.low1393, i1 %compare.high.order, !dbg !25
  %729 = select i1 %compare.result1394, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1395 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1328, i32 18) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1396 = or i32 %compare.local.base1395, 66, !dbg !25
  %compare.low1397 = icmp ugt i32 %compare.low.base, %compare.local1396, !dbg !25
  %compare.result1398 = select i1 %compare.high.equal, i1 %compare.low1397, i1 %compare.high.order, !dbg !25
  %730 = select i1 %compare.result1398, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1399 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1328, i32 19) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1400 = or i32 %compare.local.base1399, 67, !dbg !25
  %compare.low1401 = icmp ugt i32 %compare.low.base, %compare.local1400, !dbg !25
  %compare.result1402 = select i1 %compare.high.equal, i1 %compare.low1401, i1 %compare.high.order, !dbg !25
  %731 = select i1 %compare.result1402, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1403 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1328, i32 20) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1404 = or i32 %compare.local.base1403, 68, !dbg !25
  %compare.low1405 = icmp ugt i32 %compare.low.base, %compare.local1404, !dbg !25
  %compare.result1406 = select i1 %compare.high.equal, i1 %compare.low1405, i1 %compare.high.order, !dbg !25
  %732 = select i1 %compare.result1406, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1407 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1328, i32 21) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1408 = or i32 %compare.local.base1407, 69, !dbg !25
  %compare.low1409 = icmp ugt i32 %compare.low.base, %compare.local1408, !dbg !25
  %compare.result1410 = select i1 %compare.high.equal, i1 %compare.low1409, i1 %compare.high.order, !dbg !25
  %733 = select i1 %compare.result1410, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1411 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1328, i32 22) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1412 = or i32 %compare.local.base1411, 70, !dbg !25
  %compare.low1413 = icmp ugt i32 %compare.low.base, %compare.local1412, !dbg !25
  %compare.result1414 = select i1 %compare.high.equal, i1 %compare.low1413, i1 %compare.high.order, !dbg !25
  %734 = select i1 %compare.result1414, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1415 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1328, i32 23) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1416 = or i32 %compare.local.base1415, 71, !dbg !25
  %compare.low1417 = icmp ugt i32 %compare.low.base, %compare.local1416, !dbg !25
  %compare.result1418 = select i1 %compare.high.equal, i1 %compare.low1417, i1 %compare.high.order, !dbg !25
  %735 = select i1 %compare.result1418, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1419 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1328, i32 24) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1420 = or i32 %compare.local.base1419, 72, !dbg !25
  %compare.low1421 = icmp ugt i32 %compare.low.base, %compare.local1420, !dbg !25
  %compare.result1422 = select i1 %compare.high.equal, i1 %compare.low1421, i1 %compare.high.order, !dbg !25
  %736 = select i1 %compare.result1422, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1423 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1328, i32 25) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1424 = or i32 %compare.local.base1423, 73, !dbg !25
  %compare.low1425 = icmp ugt i32 %compare.low.base, %compare.local1424, !dbg !25
  %compare.result1426 = select i1 %compare.high.equal, i1 %compare.low1425, i1 %compare.high.order, !dbg !25
  %737 = select i1 %compare.result1426, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1427 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1328, i32 26) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1428 = or i32 %compare.local.base1427, 74, !dbg !25
  %compare.low1429 = icmp ugt i32 %compare.low.base, %compare.local1428, !dbg !25
  %compare.result1430 = select i1 %compare.high.equal, i1 %compare.low1429, i1 %compare.high.order, !dbg !25
  %738 = select i1 %compare.result1430, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1431 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1328, i32 27) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1432 = or i32 %compare.local.base1431, 75, !dbg !25
  %compare.low1433 = icmp ugt i32 %compare.low.base, %compare.local1432, !dbg !25
  %compare.result1434 = select i1 %compare.high.equal, i1 %compare.low1433, i1 %compare.high.order, !dbg !25
  %739 = select i1 %compare.result1434, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1435 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1328, i32 28) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1436 = or i32 %compare.local.base1435, 76, !dbg !25
  %compare.low1437 = icmp ugt i32 %compare.low.base, %compare.local1436, !dbg !25
  %compare.result1438 = select i1 %compare.high.equal, i1 %compare.low1437, i1 %compare.high.order, !dbg !25
  %740 = select i1 %compare.result1438, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1439 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1328, i32 29) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1440 = or i32 %compare.local.base1439, 77, !dbg !25
  %compare.low1441 = icmp ugt i32 %compare.low.base, %compare.local1440, !dbg !25
  %compare.result1442 = select i1 %compare.high.equal, i1 %compare.low1441, i1 %compare.high.order, !dbg !25
  %741 = select i1 %compare.result1442, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1443 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1328, i32 30) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1444 = or i32 %compare.local.base1443, 78, !dbg !25
  %compare.low1445 = icmp ugt i32 %compare.low.base, %compare.local1444, !dbg !25
  %compare.result1446 = select i1 %compare.high.equal, i1 %compare.low1445, i1 %compare.high.order, !dbg !25
  %742 = select i1 %compare.result1446, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1447 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1328, i32 31) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1448 = or i32 %compare.local.base1447, 79, !dbg !25
  %compare.low1449 = icmp ugt i32 %compare.low.base, %compare.local1448, !dbg !25
  %compare.result1450 = select i1 %compare.high.equal, i1 %compare.low1449, i1 %compare.high.order, !dbg !25
  %743 = select i1 %compare.result1450, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %744 = add i64 %706, 65, !dbg !24
  %745 = lshr i64 %744, 32, !dbg !25
  %compare.high1451 = trunc i64 %745 to i32, !dbg !25
  %746 = lshr i64 %708, 32, !dbg !25
  %compare.high1452 = trunc i64 %746 to i32, !dbg !25
  %compare.high.equal1453 = icmp eq i32 %compare.high1451, %compare.high1452, !dbg !25
  %compare.high.order1454 = icmp ugt i32 %compare.high1451, %compare.high1452, !dbg !25
  %compare.low.base1455 = trunc i64 %744 to i32, !dbg !25
  %compare.low.base1456 = trunc i64 %708 to i32, !dbg !25
  %compare.low1457 = icmp ugt i32 %compare.low.base1455, %compare.low.base1456, !dbg !25
  %compare.result1458 = select i1 %compare.high.equal1453, i1 %compare.low1457, i1 %compare.high.order1454, !dbg !25
  %747 = select i1 %compare.result1458, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1459 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1456, i32 32) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1460 = or i32 %compare.local.base1459, 1, !dbg !25
  %compare.low1461 = icmp ugt i32 %compare.low.base1455, %compare.local1460, !dbg !25
  %compare.result1462 = select i1 %compare.high.equal1453, i1 %compare.low1461, i1 %compare.high.order1454, !dbg !25
  %748 = select i1 %compare.result1462, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1463 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1456, i32 33) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1464 = or i32 %compare.local.base1463, 2, !dbg !25
  %compare.low1465 = icmp ugt i32 %compare.low.base1455, %compare.local1464, !dbg !25
  %compare.result1466 = select i1 %compare.high.equal1453, i1 %compare.low1465, i1 %compare.high.order1454, !dbg !25
  %749 = select i1 %compare.result1466, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1467 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1456, i32 34) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1468 = or i32 %compare.local.base1467, 3, !dbg !25
  %compare.low1469 = icmp ugt i32 %compare.low.base1455, %compare.local1468, !dbg !25
  %compare.result1470 = select i1 %compare.high.equal1453, i1 %compare.low1469, i1 %compare.high.order1454, !dbg !25
  %750 = select i1 %compare.result1470, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1471 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1456, i32 35) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1472 = or i32 %compare.local.base1471, 4, !dbg !25
  %compare.low1473 = icmp ugt i32 %compare.low.base1455, %compare.local1472, !dbg !25
  %compare.result1474 = select i1 %compare.high.equal1453, i1 %compare.low1473, i1 %compare.high.order1454, !dbg !25
  %751 = select i1 %compare.result1474, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1475 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1456, i32 36) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1476 = or i32 %compare.local.base1475, 5, !dbg !25
  %compare.low1477 = icmp ugt i32 %compare.low.base1455, %compare.local1476, !dbg !25
  %compare.result1478 = select i1 %compare.high.equal1453, i1 %compare.low1477, i1 %compare.high.order1454, !dbg !25
  %752 = select i1 %compare.result1478, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1479 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1456, i32 37) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1480 = or i32 %compare.local.base1479, 6, !dbg !25
  %compare.low1481 = icmp ugt i32 %compare.low.base1455, %compare.local1480, !dbg !25
  %compare.result1482 = select i1 %compare.high.equal1453, i1 %compare.low1481, i1 %compare.high.order1454, !dbg !25
  %753 = select i1 %compare.result1482, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1483 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1456, i32 38) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1484 = or i32 %compare.local.base1483, 7, !dbg !25
  %compare.low1485 = icmp ugt i32 %compare.low.base1455, %compare.local1484, !dbg !25
  %compare.result1486 = select i1 %compare.high.equal1453, i1 %compare.low1485, i1 %compare.high.order1454, !dbg !25
  %754 = select i1 %compare.result1486, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1487 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1456, i32 39) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1488 = or i32 %compare.local.base1487, 8, !dbg !25
  %compare.low1489 = icmp ugt i32 %compare.low.base1455, %compare.local1488, !dbg !25
  %compare.result1490 = select i1 %compare.high.equal1453, i1 %compare.low1489, i1 %compare.high.order1454, !dbg !25
  %755 = select i1 %compare.result1490, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1491 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1456, i32 40) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1492 = or i32 %compare.local.base1491, 9, !dbg !25
  %compare.low1493 = icmp ugt i32 %compare.low.base1455, %compare.local1492, !dbg !25
  %compare.result1494 = select i1 %compare.high.equal1453, i1 %compare.low1493, i1 %compare.high.order1454, !dbg !25
  %756 = select i1 %compare.result1494, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1495 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1456, i32 41) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1496 = or i32 %compare.local.base1495, 10, !dbg !25
  %compare.low1497 = icmp ugt i32 %compare.low.base1455, %compare.local1496, !dbg !25
  %compare.result1498 = select i1 %compare.high.equal1453, i1 %compare.low1497, i1 %compare.high.order1454, !dbg !25
  %757 = select i1 %compare.result1498, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1499 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1456, i32 42) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1500 = or i32 %compare.local.base1499, 11, !dbg !25
  %compare.low1501 = icmp ugt i32 %compare.low.base1455, %compare.local1500, !dbg !25
  %compare.result1502 = select i1 %compare.high.equal1453, i1 %compare.low1501, i1 %compare.high.order1454, !dbg !25
  %758 = select i1 %compare.result1502, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1503 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1456, i32 43) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1504 = or i32 %compare.local.base1503, 12, !dbg !25
  %compare.low1505 = icmp ugt i32 %compare.low.base1455, %compare.local1504, !dbg !25
  %compare.result1506 = select i1 %compare.high.equal1453, i1 %compare.low1505, i1 %compare.high.order1454, !dbg !25
  %759 = select i1 %compare.result1506, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1507 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1456, i32 44) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1508 = or i32 %compare.local.base1507, 13, !dbg !25
  %compare.low1509 = icmp ugt i32 %compare.low.base1455, %compare.local1508, !dbg !25
  %compare.result1510 = select i1 %compare.high.equal1453, i1 %compare.low1509, i1 %compare.high.order1454, !dbg !25
  %760 = select i1 %compare.result1510, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1511 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1456, i32 45) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1512 = or i32 %compare.local.base1511, 14, !dbg !25
  %compare.low1513 = icmp ugt i32 %compare.low.base1455, %compare.local1512, !dbg !25
  %compare.result1514 = select i1 %compare.high.equal1453, i1 %compare.low1513, i1 %compare.high.order1454, !dbg !25
  %761 = select i1 %compare.result1514, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1515 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1456, i32 46) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1516 = or i32 %compare.local.base1515, 15, !dbg !25
  %compare.low1517 = icmp ugt i32 %compare.low.base1455, %compare.local1516, !dbg !25
  %compare.result1518 = select i1 %compare.high.equal1453, i1 %compare.low1517, i1 %compare.high.order1454, !dbg !25
  %762 = select i1 %compare.result1518, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1519 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1456, i32 47) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1520 = or i32 %compare.local.base1519, 64, !dbg !25
  %compare.low1521 = icmp ugt i32 %compare.low.base1455, %compare.local1520, !dbg !25
  %compare.result1522 = select i1 %compare.high.equal1453, i1 %compare.low1521, i1 %compare.high.order1454, !dbg !25
  %763 = select i1 %compare.result1522, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1523 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1456, i32 48) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1524 = or i32 %compare.local.base1523, 65, !dbg !25
  %compare.low1525 = icmp ugt i32 %compare.low.base1455, %compare.local1524, !dbg !25
  %compare.result1526 = select i1 %compare.high.equal1453, i1 %compare.low1525, i1 %compare.high.order1454, !dbg !25
  %764 = select i1 %compare.result1526, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1527 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1456, i32 49) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1528 = or i32 %compare.local.base1527, 66, !dbg !25
  %compare.low1529 = icmp ugt i32 %compare.low.base1455, %compare.local1528, !dbg !25
  %compare.result1530 = select i1 %compare.high.equal1453, i1 %compare.low1529, i1 %compare.high.order1454, !dbg !25
  %765 = select i1 %compare.result1530, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1531 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1456, i32 50) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1532 = or i32 %compare.local.base1531, 67, !dbg !25
  %compare.low1533 = icmp ugt i32 %compare.low.base1455, %compare.local1532, !dbg !25
  %compare.result1534 = select i1 %compare.high.equal1453, i1 %compare.low1533, i1 %compare.high.order1454, !dbg !25
  %766 = select i1 %compare.result1534, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1535 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1456, i32 51) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1536 = or i32 %compare.local.base1535, 68, !dbg !25
  %compare.low1537 = icmp ugt i32 %compare.low.base1455, %compare.local1536, !dbg !25
  %compare.result1538 = select i1 %compare.high.equal1453, i1 %compare.low1537, i1 %compare.high.order1454, !dbg !25
  %767 = select i1 %compare.result1538, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1539 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1456, i32 52) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1540 = or i32 %compare.local.base1539, 69, !dbg !25
  %compare.low1541 = icmp ugt i32 %compare.low.base1455, %compare.local1540, !dbg !25
  %compare.result1542 = select i1 %compare.high.equal1453, i1 %compare.low1541, i1 %compare.high.order1454, !dbg !25
  %768 = select i1 %compare.result1542, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1543 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1456, i32 53) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1544 = or i32 %compare.local.base1543, 70, !dbg !25
  %compare.low1545 = icmp ugt i32 %compare.low.base1455, %compare.local1544, !dbg !25
  %compare.result1546 = select i1 %compare.high.equal1453, i1 %compare.low1545, i1 %compare.high.order1454, !dbg !25
  %769 = select i1 %compare.result1546, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1547 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1456, i32 54) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1548 = or i32 %compare.local.base1547, 71, !dbg !25
  %compare.low1549 = icmp ugt i32 %compare.low.base1455, %compare.local1548, !dbg !25
  %compare.result1550 = select i1 %compare.high.equal1453, i1 %compare.low1549, i1 %compare.high.order1454, !dbg !25
  %770 = select i1 %compare.result1550, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1551 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1456, i32 55) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1552 = or i32 %compare.local.base1551, 72, !dbg !25
  %compare.low1553 = icmp ugt i32 %compare.low.base1455, %compare.local1552, !dbg !25
  %compare.result1554 = select i1 %compare.high.equal1453, i1 %compare.low1553, i1 %compare.high.order1454, !dbg !25
  %771 = select i1 %compare.result1554, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1555 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1456, i32 56) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1556 = or i32 %compare.local.base1555, 73, !dbg !25
  %compare.low1557 = icmp ugt i32 %compare.low.base1455, %compare.local1556, !dbg !25
  %compare.result1558 = select i1 %compare.high.equal1453, i1 %compare.low1557, i1 %compare.high.order1454, !dbg !25
  %772 = select i1 %compare.result1558, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1559 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1456, i32 57) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1560 = or i32 %compare.local.base1559, 74, !dbg !25
  %compare.low1561 = icmp ugt i32 %compare.low.base1455, %compare.local1560, !dbg !25
  %compare.result1562 = select i1 %compare.high.equal1453, i1 %compare.low1561, i1 %compare.high.order1454, !dbg !25
  %773 = select i1 %compare.result1562, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1563 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1456, i32 58) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1564 = or i32 %compare.local.base1563, 75, !dbg !25
  %compare.low1565 = icmp ugt i32 %compare.low.base1455, %compare.local1564, !dbg !25
  %compare.result1566 = select i1 %compare.high.equal1453, i1 %compare.low1565, i1 %compare.high.order1454, !dbg !25
  %774 = select i1 %compare.result1566, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1567 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1456, i32 59) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1568 = or i32 %compare.local.base1567, 76, !dbg !25
  %compare.low1569 = icmp ugt i32 %compare.low.base1455, %compare.local1568, !dbg !25
  %compare.result1570 = select i1 %compare.high.equal1453, i1 %compare.low1569, i1 %compare.high.order1454, !dbg !25
  %775 = select i1 %compare.result1570, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1571 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1456, i32 60) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1572 = or i32 %compare.local.base1571, 77, !dbg !25
  %compare.low1573 = icmp ugt i32 %compare.low.base1455, %compare.local1572, !dbg !25
  %compare.result1574 = select i1 %compare.high.equal1453, i1 %compare.low1573, i1 %compare.high.order1454, !dbg !25
  %776 = select i1 %compare.result1574, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1575 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1456, i32 61) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1576 = or i32 %compare.local.base1575, 78, !dbg !25
  %compare.low1577 = icmp ugt i32 %compare.low.base1455, %compare.local1576, !dbg !25
  %compare.result1578 = select i1 %compare.high.equal1453, i1 %compare.low1577, i1 %compare.high.order1454, !dbg !25
  %777 = select i1 %compare.result1578, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %compare.local.base1579 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1456, i32 62) #9, !dbg !25, !frisk.register_lifetimes !1
  %compare.local1580 = or i32 %compare.local.base1579, 79, !dbg !25
  %compare.low1581 = icmp ugt i32 %compare.low.base1455, %compare.local1580, !dbg !25
  %compare.result1582 = select i1 %compare.high.equal1453, i1 %compare.low1581, i1 %compare.high.order1454, !dbg !25
  %778 = select i1 %compare.result1582, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !26
  %779 = insertelement <32 x float> poison, float %712, i64 0, !dbg !13
  %780 = insertelement <32 x float> %779, float %713, i64 1, !dbg !13
  %781 = insertelement <32 x float> %780, float %714, i64 2, !dbg !13
  %782 = insertelement <32 x float> %781, float %715, i64 3, !dbg !13
  %783 = insertelement <32 x float> %782, float %716, i64 4, !dbg !13
  %784 = insertelement <32 x float> %783, float %717, i64 5, !dbg !13
  %785 = insertelement <32 x float> %784, float %718, i64 6, !dbg !13
  %786 = insertelement <32 x float> %785, float %719, i64 7, !dbg !13
  %787 = insertelement <32 x float> %786, float %720, i64 8, !dbg !13
  %788 = insertelement <32 x float> %787, float %721, i64 9, !dbg !13
  %789 = insertelement <32 x float> %788, float %722, i64 10, !dbg !13
  %790 = insertelement <32 x float> %789, float %723, i64 11, !dbg !13
  %791 = insertelement <32 x float> %790, float %724, i64 12, !dbg !13
  %792 = insertelement <32 x float> %791, float %725, i64 13, !dbg !13
  %793 = insertelement <32 x float> %792, float %726, i64 14, !dbg !13
  %794 = insertelement <32 x float> %793, float %727, i64 15, !dbg !13
  %795 = insertelement <32 x float> %794, float %728, i64 16, !dbg !13
  %796 = insertelement <32 x float> %795, float %729, i64 17, !dbg !13
  %797 = insertelement <32 x float> %796, float %730, i64 18, !dbg !13
  %798 = insertelement <32 x float> %797, float %731, i64 19, !dbg !13
  %799 = insertelement <32 x float> %798, float %732, i64 20, !dbg !13
  %800 = insertelement <32 x float> %799, float %733, i64 21, !dbg !13
  %801 = insertelement <32 x float> %800, float %734, i64 22, !dbg !13
  %802 = insertelement <32 x float> %801, float %735, i64 23, !dbg !13
  %803 = insertelement <32 x float> %802, float %736, i64 24, !dbg !13
  %804 = insertelement <32 x float> %803, float %737, i64 25, !dbg !13
  %805 = insertelement <32 x float> %804, float %738, i64 26, !dbg !13
  %806 = insertelement <32 x float> %805, float %739, i64 27, !dbg !13
  %807 = insertelement <32 x float> %806, float %740, i64 28, !dbg !13
  %808 = insertelement <32 x float> %807, float %741, i64 29, !dbg !13
  %809 = insertelement <32 x float> %808, float %742, i64 30, !dbg !13
  %810 = insertelement <32 x float> %809, float %743, i64 31, !dbg !13
  %811 = insertvalue [2 x <32 x float>] poison, <32 x float> %810, 0, !dbg !13
  %812 = insertelement <32 x float> poison, float %747, i64 0, !dbg !13
  %813 = insertelement <32 x float> %812, float %748, i64 1, !dbg !13
  %814 = insertelement <32 x float> %813, float %749, i64 2, !dbg !13
  %815 = insertelement <32 x float> %814, float %750, i64 3, !dbg !13
  %816 = insertelement <32 x float> %815, float %751, i64 4, !dbg !13
  %817 = insertelement <32 x float> %816, float %752, i64 5, !dbg !13
  %818 = insertelement <32 x float> %817, float %753, i64 6, !dbg !13
  %819 = insertelement <32 x float> %818, float %754, i64 7, !dbg !13
  %820 = insertelement <32 x float> %819, float %755, i64 8, !dbg !13
  %821 = insertelement <32 x float> %820, float %756, i64 9, !dbg !13
  %822 = insertelement <32 x float> %821, float %757, i64 10, !dbg !13
  %823 = insertelement <32 x float> %822, float %758, i64 11, !dbg !13
  %824 = insertelement <32 x float> %823, float %759, i64 12, !dbg !13
  %825 = insertelement <32 x float> %824, float %760, i64 13, !dbg !13
  %826 = insertelement <32 x float> %825, float %761, i64 14, !dbg !13
  %827 = insertelement <32 x float> %826, float %762, i64 15, !dbg !13
  %828 = insertelement <32 x float> %827, float %763, i64 16, !dbg !13
  %829 = insertelement <32 x float> %828, float %764, i64 17, !dbg !13
  %830 = insertelement <32 x float> %829, float %765, i64 18, !dbg !13
  %831 = insertelement <32 x float> %830, float %766, i64 19, !dbg !13
  %832 = insertelement <32 x float> %831, float %767, i64 20, !dbg !13
  %833 = insertelement <32 x float> %832, float %768, i64 21, !dbg !13
  %834 = insertelement <32 x float> %833, float %769, i64 22, !dbg !13
  %835 = insertelement <32 x float> %834, float %770, i64 23, !dbg !13
  %836 = insertelement <32 x float> %835, float %771, i64 24, !dbg !13
  %837 = insertelement <32 x float> %836, float %772, i64 25, !dbg !13
  %838 = insertelement <32 x float> %837, float %773, i64 26, !dbg !13
  %839 = insertelement <32 x float> %838, float %774, i64 27, !dbg !13
  %840 = insertelement <32 x float> %839, float %775, i64 28, !dbg !13
  %841 = insertelement <32 x float> %840, float %776, i64 29, !dbg !13
  %842 = insertelement <32 x float> %841, float %777, i64 30, !dbg !13
  %843 = insertelement <32 x float> %842, float %778, i64 31, !dbg !13
  %844 = insertvalue [2 x <32 x float>] %811, <32 x float> %843, 1, !dbg !13
  br label %845, !dbg !13

845:                                              ; preds = %700, %701
  %846 = phi [2 x <32 x float>] [ %844, %701 ], [ zeroinitializer, %700 ], !dbg !13
  br label %847, !dbg !13

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
  %math.result468 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %883) #9, !dbg !30, !frisk.register_lifetimes !1
  %884 = fptrunc <4 x float> %math.result468 to <4 x half>, !dbg !30
  %math.packed469 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %884) #9, !dbg !29, !frisk.register_lifetimes !1
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
  %math.result470 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %902) #9, !dbg !30, !frisk.register_lifetimes !1
  %903 = fptrunc <4 x float> %math.result470 to <4 x half>, !dbg !30
  %math.packed471 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %903) #9, !dbg !29, !frisk.register_lifetimes !1
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
  %math.result472 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %921) #9, !dbg !30, !frisk.register_lifetimes !1
  %922 = fptrunc <4 x float> %math.result472 to <4 x half>, !dbg !30
  %math.packed473 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %922) #9, !dbg !29, !frisk.register_lifetimes !1
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
  %math.result474 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %940) #9, !dbg !30, !frisk.register_lifetimes !1
  %941 = fptrunc <4 x float> %math.result474 to <4 x half>, !dbg !30
  %math.packed475 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %941) #9, !dbg !29, !frisk.register_lifetimes !1
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
  %math.result476 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %959) #9, !dbg !30, !frisk.register_lifetimes !1
  %960 = fptrunc <4 x float> %math.result476 to <4 x half>, !dbg !30
  %math.packed477 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %960) #9, !dbg !29, !frisk.register_lifetimes !1
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
  %math.result478 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %978) #9, !dbg !30, !frisk.register_lifetimes !1
  %979 = fptrunc <4 x float> %math.result478 to <4 x half>, !dbg !30
  %math.packed479 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %979) #9, !dbg !29, !frisk.register_lifetimes !1
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
  %math.result480 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %997) #9, !dbg !30, !frisk.register_lifetimes !1
  %998 = fptrunc <4 x float> %math.result480 to <4 x half>, !dbg !30
  %math.packed481 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %998) #9, !dbg !29, !frisk.register_lifetimes !1
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
  %math.result482 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1016) #9, !dbg !30, !frisk.register_lifetimes !1
  %1017 = fptrunc <4 x float> %math.result482 to <4 x half>, !dbg !30
  %math.packed483 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1017) #9, !dbg !29, !frisk.register_lifetimes !1
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
  %math.result484 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1035) #9, !dbg !30, !frisk.register_lifetimes !1
  %1036 = fptrunc <4 x float> %math.result484 to <4 x half>, !dbg !30
  %math.packed485 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1036) #9, !dbg !29, !frisk.register_lifetimes !1
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
  %math.result486 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1054) #9, !dbg !30, !frisk.register_lifetimes !1
  %1055 = fptrunc <4 x float> %math.result486 to <4 x half>, !dbg !30
  %math.packed487 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1055) #9, !dbg !29, !frisk.register_lifetimes !1
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
  %math.result488 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1073) #9, !dbg !30, !frisk.register_lifetimes !1
  %1074 = fptrunc <4 x float> %math.result488 to <4 x half>, !dbg !30
  %math.packed489 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1074) #9, !dbg !29, !frisk.register_lifetimes !1
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
  %math.result490 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1092) #9, !dbg !30, !frisk.register_lifetimes !1
  %1093 = fptrunc <4 x float> %math.result490 to <4 x half>, !dbg !30
  %math.packed491 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1093) #9, !dbg !29, !frisk.register_lifetimes !1
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
  %math.result492 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1111) #9, !dbg !30, !frisk.register_lifetimes !1
  %1112 = fptrunc <4 x float> %math.result492 to <4 x half>, !dbg !30
  %math.packed493 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1112) #9, !dbg !29, !frisk.register_lifetimes !1
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
  %math.result494 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1130) #9, !dbg !30, !frisk.register_lifetimes !1
  %1131 = fptrunc <4 x float> %math.result494 to <4 x half>, !dbg !30
  %math.packed495 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1131) #9, !dbg !29, !frisk.register_lifetimes !1
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
  %math.result496 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1149) #9, !dbg !30, !frisk.register_lifetimes !1
  %1150 = fptrunc <4 x float> %math.result496 to <4 x half>, !dbg !30
  %math.packed497 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1150) #9, !dbg !29, !frisk.register_lifetimes !1
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
  %1172 = extractvalue [2 x <1 x float>] %63, 0, !dbg !29
  %1173 = extractelement <1 x float> %1172, i64 0, !dbg !29
  %1174 = fadd float %1171, %1173, !dbg !29
  %1175 = insertelement <1 x float> poison, float %1174, i64 0, !dbg !15
  %1176 = insertvalue [2 x <1 x float>] poison, <1 x float> %1175, 0, !dbg !15
  %1177 = bitcast float %1151 to i32, !dbg !29
  %1178 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1159, i32 %1177), !dbg !29
  %1179 = bitcast i32 %1178 to float, !dbg !29
  %1180 = fadd float %1151, %1179, !dbg !29
  %1181 = bitcast float %1180 to i32, !dbg !29
  %1182 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1167, i32 %1181), !dbg !29
  %1183 = bitcast i32 %1182 to float, !dbg !29
  %1184 = fadd float %1180, %1183, !dbg !29
  %1185 = extractvalue [2 x <1 x float>] %63, 1, !dbg !29
  %1186 = extractelement <1 x float> %1185, i64 0, !dbg !29
  %1187 = fadd float %1184, %1186, !dbg !29
  %1188 = insertelement <1 x float> poison, float %1187, i64 0, !dbg !15
  %1189 = insertvalue [2 x <1 x float>] %1176, <1 x float> %1188, 1, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !19
  fence syncscope("workgroup") release, !dbg !19
  call void @llvm.amdgcn.s.barrier(), !dbg !19
  fence syncscope("workgroup") acquire, !dbg !19
  %address.thread945 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !14, !range !34, !frisk.register_lifetimes !1
  %address.local946 = shl nuw nsw i32 %address.thread945, 3, !dbg !14
  %address.local947 = and i32 %address.local946, 120, !dbg !14
  %address.local948 = shl nuw nsw i32 %address.thread945, 5, !dbg !14
  %address.local949 = and i32 %address.local948, 1536, !dbg !14
  %address.local950 = or disjoint i32 %address.local947, %address.local949, !dbg !14
  %address.local951 = shl nuw nsw i32 %address.thread945, 3, !dbg !14
  %address.local952 = and i32 %address.local951, 384, !dbg !14
  %address.local953 = or disjoint i32 %address.local950, %address.local952, !dbg !14
  %address.local954 = zext nneg i32 %address.local953 to i64, !dbg !14
  %address.local955 = shl nuw nsw i32 %address.thread945, 3, !dbg !14
  %address.local956 = and i32 %address.local955, 64, !dbg !14
  %address.local957 = shl nuw nsw i32 %address.thread945, 4, !dbg !14
  %address.local958 = and i32 %address.local957, 768, !dbg !14
  %address.local959 = or disjoint i32 %address.local956, %address.local958, !dbg !14
  %address.local960 = zext nneg i32 %address.local959 to i64, !dbg !14
  %address.local961 = sub nsw i64 %address.local954, %address.local960, !dbg !14
  %address.local962 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local961, !dbg !14
  %address.local963 = zext nneg i32 %address.thread945 to i64, !dbg !12
  %address.local964 = shl nuw nsw i64 %address.local963, 3, !dbg !14
  %address.local965 = and i64 %address.local964, 448, !dbg !14
  %address.local966 = getelementptr [2 x i8], ptr addrspace(3) %address.local962, i64 %address.local965, !dbg !14
  %1190 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local966, i32 0) #0, !dbg !14
  %1191 = bitcast <4 x i32> %1190 to <8 x half>, !dbg !14
  %address.local967 = shl nuw nsw i32 %address.thread945, 3, !dbg !14
  %address.local968 = and i32 %address.local967, 88, !dbg !14
  %address.local969 = shl nuw nsw i32 %address.thread945, 3, !dbg !14
  %address.local970 = and i32 %address.local969, 32, !dbg !14
  %address.local971 = sub nsw i32 %address.local968, %address.local970, !dbg !14
  %address.local972 = shl nuw nsw i32 %address.thread945, 5, !dbg !14
  %address.local973 = and i32 %address.local972, 1536, !dbg !14
  %address.local974 = add nsw i32 %address.local971, %address.local973, !dbg !14
  %address.local975 = sext i32 %address.local974 to i64, !dbg !14
  %address.local976 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local975, !dbg !14
  %address.local977 = getelementptr i8, ptr addrspace(3) %address.local976, i64 320, !dbg !14
  %1192 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local977, i32 0) #0, !dbg !14
  %1193 = bitcast <4 x i32> %1192 to <8 x half>, !dbg !14
  %address.local978 = shl nuw nsw i32 %address.thread945, 3, !dbg !14
  %address.local979 = and i32 %address.local978, 120, !dbg !14
  %address.local980 = shl nuw nsw i32 %address.thread945, 5, !dbg !14
  %address.local981 = and i32 %address.local980, 1536, !dbg !14
  %address.local982 = or disjoint i32 %address.local979, %address.local981, !dbg !14
  %address.local983 = shl nuw nsw i32 %address.thread945, 3, !dbg !14
  %address.local984 = and i32 %address.local983, 384, !dbg !14
  %address.local985 = or disjoint i32 %address.local982, %address.local984, !dbg !14
  %address.local986 = zext nneg i32 %address.local985 to i64, !dbg !14
  %address.local987 = shl nuw nsw i32 %address.thread945, 3, !dbg !14
  %address.local988 = and i32 %address.local987, 64, !dbg !14
  %address.local989 = shl nuw nsw i32 %address.thread945, 4, !dbg !14
  %address.local990 = and i32 %address.local989, 768, !dbg !14
  %address.local991 = or disjoint i32 %address.local988, %address.local990, !dbg !14
  %address.local992 = zext nneg i32 %address.local991 to i64, !dbg !14
  %address.local993 = sub nsw i64 %address.local986, %address.local992, !dbg !14
  %address.local994 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local993, !dbg !14
  %address.local995 = getelementptr [2 x i8], ptr addrspace(3) %address.local994, i64 %address.local965, !dbg !14
  %address.local996 = getelementptr i8, ptr addrspace(3) %address.local995, i64 512, !dbg !14
  %1194 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local996, i32 0) #0, !dbg !14
  %1195 = bitcast <4 x i32> %1194 to <8 x half>, !dbg !14
  %address.local997 = shl nuw nsw i32 %address.thread945, 3, !dbg !14
  %address.local998 = and i32 %address.local997, 88, !dbg !14
  %address.local999 = add nsw i32 %address.local998, -64, !dbg !14
  %address.local1000 = shl nuw nsw i32 %address.thread945, 3, !dbg !14
  %address.local1001 = and i32 %address.local1000, 32, !dbg !14
  %address.local1002 = sub nsw i32 %address.local999, %address.local1001, !dbg !14
  %address.local1003 = shl nuw nsw i32 %address.thread945, 5, !dbg !14
  %address.local1004 = and i32 %address.local1003, 1536, !dbg !14
  %address.local1005 = add nsw i32 %address.local1002, %address.local1004, !dbg !14
  %address.local1006 = sext i32 %address.local1005 to i64, !dbg !14
  %address.local1007 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1006, !dbg !14
  %address.local1008 = getelementptr i8, ptr addrspace(3) %address.local1007, i64 960, !dbg !14
  %1196 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1008, i32 0) #0, !dbg !14
  %1197 = bitcast <4 x i32> %1196 to <8 x half>, !dbg !14
  %address.local1009 = shl nuw nsw i32 %address.thread945, 3, !dbg !14
  %address.local1010 = and i32 %address.local1009, 64, !dbg !14
  %address.local1011 = shl nuw nsw i32 %address.thread945, 4, !dbg !14
  %address.local1012 = and i32 %address.local1011, 768, !dbg !14
  %address.local1013 = or disjoint i32 %address.local1010, %address.local1012, !dbg !14
  %address.local1014 = zext nneg i32 %address.local1013 to i64, !dbg !14
  %address.local1015 = sub nsw i64 %address.local954, %address.local1014, !dbg !14
  %address.local1016 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1015, !dbg !14
  %address.local1017 = getelementptr [2 x i8], ptr addrspace(3) %address.local1016, i64 %address.local965, !dbg !14
  %address.local1018 = getelementptr i8, ptr addrspace(3) %address.local1017, i64 4096, !dbg !14
  %1198 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1018, i32 0) #0, !dbg !14
  %1199 = bitcast <4 x i32> %1198 to <8 x half>, !dbg !14
  %address.local1019 = shl nuw nsw i32 %address.thread945, 3, !dbg !14
  %address.local1020 = and i32 %address.local1019, 88, !dbg !14
  %address.local1021 = or disjoint i32 %address.local1020, 2048, !dbg !14
  %address.local1022 = shl nuw nsw i32 %address.thread945, 3, !dbg !14
  %address.local1023 = and i32 %address.local1022, 32, !dbg !14
  %address.local1024 = sub nuw nsw i32 %address.local1021, %address.local1023, !dbg !14
  %address.local1025 = shl nuw nsw i32 %address.thread945, 5, !dbg !14
  %address.local1026 = and i32 %address.local1025, 1536, !dbg !14
  %address.local1027 = add nuw nsw i32 %address.local1024, %address.local1026, !dbg !14
  %address.local1028 = zext nneg i32 %address.local1027 to i64, !dbg !14
  %address.local1029 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1028, !dbg !14
  %address.local1030 = getelementptr i8, ptr addrspace(3) %address.local1029, i64 320, !dbg !14
  %1200 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1030, i32 0) #0, !dbg !14
  %1201 = bitcast <4 x i32> %1200 to <8 x half>, !dbg !14
  %address.local1031 = shl nuw nsw i32 %address.thread945, 3, !dbg !14
  %address.local1032 = and i32 %address.local1031, 120, !dbg !14
  %address.local1033 = shl nuw nsw i32 %address.thread945, 5, !dbg !14
  %address.local1034 = and i32 %address.local1033, 1536, !dbg !14
  %address.local1035 = or disjoint i32 %address.local1032, %address.local1034, !dbg !14
  %address.local1036 = or disjoint i32 %address.local1035, 2048, !dbg !14
  %address.local1037 = shl nuw nsw i32 %address.thread945, 3, !dbg !14
  %address.local1038 = and i32 %address.local1037, 384, !dbg !14
  %address.local1039 = or disjoint i32 %address.local1038, 512, !dbg !14
  %address.local1040 = add nuw nsw i32 %address.local1036, %address.local1039, !dbg !14
  %address.local1041 = zext nneg i32 %address.local1040 to i64, !dbg !14
  %address.local1042 = shl nuw nsw i32 %address.thread945, 3, !dbg !14
  %address.local1043 = and i32 %address.local1042, 64, !dbg !14
  %address.local1044 = or disjoint i32 %address.local1043, 512, !dbg !14
  %address.local1045 = shl nuw nsw i32 %address.thread945, 4, !dbg !14
  %address.local1046 = and i32 %address.local1045, 768, !dbg !14
  %address.local1047 = add nuw nsw i32 %address.local1044, %address.local1046, !dbg !14
  %address.local1048 = zext nneg i32 %address.local1047 to i64, !dbg !14
  %address.local1049 = sub nsw i64 %address.local1041, %address.local1048, !dbg !14
  %address.local1050 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1049, !dbg !14
  %address.local1051 = getelementptr [2 x i8], ptr addrspace(3) %address.local1050, i64 %address.local965, !dbg !14
  %address.local1052 = getelementptr i8, ptr addrspace(3) %address.local1051, i64 512, !dbg !14
  %1202 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1052, i32 0) #0, !dbg !14
  %1203 = bitcast <4 x i32> %1202 to <8 x half>, !dbg !14
  %address.local1053 = shl nuw nsw i32 %address.thread945, 3, !dbg !14
  %address.local1054 = and i32 %address.local1053, 88, !dbg !14
  %address.local1055 = add nuw nsw i32 %address.local1054, 1984, !dbg !14
  %address.local1056 = shl nuw nsw i32 %address.thread945, 3, !dbg !14
  %address.local1057 = and i32 %address.local1056, 32, !dbg !14
  %address.local1058 = sub nuw nsw i32 %address.local1055, %address.local1057, !dbg !14
  %address.local1059 = shl nuw nsw i32 %address.thread945, 5, !dbg !14
  %address.local1060 = and i32 %address.local1059, 1536, !dbg !14
  %address.local1061 = add nuw nsw i32 %address.local1058, %address.local1060, !dbg !14
  %address.local1062 = zext nneg i32 %address.local1061 to i64, !dbg !14
  %address.local1063 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1062, !dbg !14
  %address.local1064 = getelementptr i8, ptr addrspace(3) %address.local1063, i64 960, !dbg !14
  %1204 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1064, i32 0) #0, !dbg !14
  %1205 = bitcast <4 x i32> %1204 to <8 x half>, !dbg !14
  %1206 = add nuw nsw i64 %100, %15, !dbg !19
  %1207 = shl i64 %1206, 19, !dbg !19
  %1208 = and i64 %1207, 16252928, !dbg !19
  %1209 = add i64 %1208, %106, !dbg !19
  fence syncscope("workgroup") release, !dbg !19
  call void @llvm.amdgcn.s.barrier(), !dbg !19
  fence syncscope("workgroup") acquire, !dbg !19
  %1210 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %1209, !dbg !19
  %1211 = getelementptr i8, ptr addrspace(1) %1210, i64 64, !dbg !19
  %address.local1065 = shl nuw nsw i64 %address.local963, 3, !dbg !11
  %address.local1066 = and i64 %address.local1065, 1536, !dbg !11
  %address.local1067 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1066, !dbg !11
  %address.local1068 = ptrtoint ptr addrspace(3) %address.local1067 to i64, !dbg !11
  %address.local1069 = trunc i64 %address.local1068 to i32, !dbg !11
  %1212 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1069), !dbg !19
  %1213 = zext i32 %1212 to i64, !dbg !19
  %1214 = inttoptr i64 %1213 to ptr addrspace(3), !dbg !19
  %address.local1070 = and i64 %address.local963, 251, !dbg !11
  %address.local1071 = lshr i64 %address.local963, 2, !dbg !11
  %address.local1072 = and i64 %address.local1071, 4, !dbg !11
  %address.local1073 = add nuw nsw i64 %address.local1072, %address.local963, !dbg !11
  %address.local1074 = and i64 %address.local1073, 4, !dbg !11
  %address.local1075 = or disjoint i64 %address.local1070, %address.local1074, !dbg !11
  %address.local1076 = shl nuw nsw i64 %address.local1075, 5, !dbg !19
  %address.local1077 = and i64 %address.local1076, 8064, !dbg !19
  %address.local1078 = shl nuw nsw i64 %address.local963, 3, !dbg !19
  %address.local1079 = and i64 %address.local1078, 24, !dbg !19
  %address.local1080 = or disjoint i64 %address.local1077, %address.local1079, !dbg !19
  %address.local1081 = getelementptr [2 x i8], ptr addrspace(1) %1211, i64 %address.local1080, !dbg !19
  %1215 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !19
  %1216 = ptrtoint ptr addrspace(1) %address.local1081 to i64, !dbg !19
  %1217 = sub i64 %1216, %1215, !dbg !19
  %dma.buffer.offset1718 = trunc i64 %1217 to i32, !dbg !19
  %1218 = trunc i64 %1215 to i32, !dbg !19
  %1219 = insertelement <4 x i32> poison, i32 %1218, i64 0, !dbg !19
  %1220 = lshr i64 %1215, 32, !dbg !19
  %1221 = trunc i64 %1220 to i32, !dbg !19
  %1222 = insertelement <4 x i32> %1219, i32 %1221, i64 1, !dbg !19
  %1223 = insertelement <4 x i32> %1222, i32 33554432, i64 2, !dbg !19
  %dma.buffer.rsrc1719 = insertelement <4 x i32> %1223, i32 131072, i64 3, !dbg !19
  %1224 = ptrtoint ptr addrspace(3) %1214 to i32, !dbg !19
  %dma.buffer.lds1720 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1224), !dbg !19
  %1225 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1720, i32 %dma.buffer.offset1718, <4 x i32> %dma.buffer.rsrc1719) #10, !dbg !19
  %address.local1082 = or disjoint i64 %address.local963, 256, !dbg !11
  %address.local1083 = shl nuw nsw i64 %address.local1082, 3, !dbg !11
  %address.local1084 = and i64 %address.local1083, 3584, !dbg !11
  %address.local1085 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1084, !dbg !11
  %address.local1086 = ptrtoint ptr addrspace(3) %address.local1085 to i64, !dbg !11
  %address.local1087 = trunc i64 %address.local1086 to i32, !dbg !11
  %1226 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1087), !dbg !19
  %1227 = zext i32 %1226 to i64, !dbg !19
  %1228 = inttoptr i64 %1227 to ptr addrspace(3), !dbg !19
  %address.local1088 = and i64 %address.local1082, 507, !dbg !11
  %address.local1089 = lshr i64 %address.local963, 2, !dbg !11
  %address.local1090 = and i64 %address.local1089, 4, !dbg !11
  %address.local1091 = add nuw nsw i64 %address.local1090, %address.local963, !dbg !11
  %address.local1092 = and i64 %address.local1091, 4, !dbg !11
  %address.local1093 = or disjoint i64 %address.local1088, %address.local1092, !dbg !11
  %address.local1094 = shl nuw nsw i64 %address.local1093, 5, !dbg !19
  %address.local1095 = and i64 %address.local1094, 16256, !dbg !19
  %address.local1096 = shl nuw nsw i64 %address.local963, 3, !dbg !19
  %address.local1097 = and i64 %address.local1096, 24, !dbg !19
  %address.local1098 = or disjoint i64 %address.local1095, %address.local1097, !dbg !19
  %address.local1099 = getelementptr [2 x i8], ptr addrspace(1) %1211, i64 %address.local1098, !dbg !19
  %1229 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !19
  %1230 = ptrtoint ptr addrspace(1) %address.local1099 to i64, !dbg !19
  %1231 = sub i64 %1230, %1229, !dbg !19
  %dma.buffer.offset1715 = trunc i64 %1231 to i32, !dbg !19
  %1232 = trunc i64 %1229 to i32, !dbg !19
  %1233 = insertelement <4 x i32> poison, i32 %1232, i64 0, !dbg !19
  %1234 = lshr i64 %1229, 32, !dbg !19
  %1235 = trunc i64 %1234 to i32, !dbg !19
  %1236 = insertelement <4 x i32> %1233, i32 %1235, i64 1, !dbg !19
  %1237 = insertelement <4 x i32> %1236, i32 33554432, i64 2, !dbg !19
  %dma.buffer.rsrc1716 = insertelement <4 x i32> %1237, i32 131072, i64 3, !dbg !19
  %1238 = ptrtoint ptr addrspace(3) %1228 to i32, !dbg !19
  %dma.buffer.lds1717 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1238), !dbg !19
  %1239 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1717, i32 %dma.buffer.offset1715, <4 x i32> %dma.buffer.rsrc1716) #10, !dbg !19
  %1240 = shufflevector <8 x half> %1191, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1241 = extractvalue [2 x <8 x float>] %62, 0, !dbg !31
  %1242 = shufflevector <8 x float> %1241, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %1243 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1240, <4 x float> %1242) #9, !dbg !14
  %1244 = shufflevector <8 x half> %1193, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1245 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed469, <4 x half> %1244, <4 x float> %1243) #9, !dbg !14
  %1246 = shufflevector <8 x half> %1195, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1247 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed471, <4 x half> %1246, <4 x float> %1245) #9, !dbg !14
  %1248 = shufflevector <8 x half> %1197, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1249 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed473, <4 x half> %1248, <4 x float> %1247) #9, !dbg !14
  %1250 = shufflevector <8 x half> %1199, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1251 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed475, <4 x half> %1250, <4 x float> %1249) #9, !dbg !14
  %1252 = shufflevector <8 x half> %1201, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1253 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed477, <4 x half> %1252, <4 x float> %1251) #9, !dbg !14
  %1254 = shufflevector <8 x half> %1203, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1255 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed479, <4 x half> %1254, <4 x float> %1253) #9, !dbg !14
  %1256 = shufflevector <8 x half> %1205, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1257 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed481, <4 x half> %1256, <4 x float> %1255) #9, !dbg !14
  %1258 = shufflevector <8 x half> %1191, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1259 = shufflevector <8 x float> %1241, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %1260 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1258, <4 x float> %1259) #9, !dbg !14
  %1261 = shufflevector <8 x half> %1193, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1262 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed469, <4 x half> %1261, <4 x float> %1260) #9, !dbg !14
  %1263 = shufflevector <8 x half> %1195, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1264 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed471, <4 x half> %1263, <4 x float> %1262) #9, !dbg !14
  %1265 = shufflevector <8 x half> %1197, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1266 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed473, <4 x half> %1265, <4 x float> %1264) #9, !dbg !14
  %1267 = shufflevector <8 x half> %1199, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1268 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed475, <4 x half> %1267, <4 x float> %1266) #9, !dbg !14
  %1269 = shufflevector <8 x half> %1201, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1270 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed477, <4 x half> %1269, <4 x float> %1268) #9, !dbg !14
  %1271 = shufflevector <8 x half> %1203, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1272 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed479, <4 x half> %1271, <4 x float> %1270) #9, !dbg !14
  %1273 = shufflevector <8 x half> %1205, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1274 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed481, <4 x half> %1273, <4 x float> %1272) #9, !dbg !14
  %1275 = extractvalue [2 x <8 x float>] %62, 1, !dbg !31
  %1276 = shufflevector <8 x float> %1275, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %1277 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed483, <4 x half> %1240, <4 x float> %1276) #9, !dbg !14
  %1278 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed485, <4 x half> %1244, <4 x float> %1277) #9, !dbg !14
  %1279 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed487, <4 x half> %1246, <4 x float> %1278) #9, !dbg !14
  %1280 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed489, <4 x half> %1248, <4 x float> %1279) #9, !dbg !14
  %1281 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed491, <4 x half> %1250, <4 x float> %1280) #9, !dbg !14
  %1282 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed493, <4 x half> %1252, <4 x float> %1281) #9, !dbg !14
  %1283 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed495, <4 x half> %1254, <4 x float> %1282) #9, !dbg !14
  %1284 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed497, <4 x half> %1256, <4 x float> %1283) #9, !dbg !14
  %1285 = shufflevector <8 x float> %1275, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %1286 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed483, <4 x half> %1258, <4 x float> %1285) #9, !dbg !14
  %1287 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed485, <4 x half> %1261, <4 x float> %1286) #9, !dbg !14
  %1288 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed487, <4 x half> %1263, <4 x float> %1287) #9, !dbg !14
  %1289 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed489, <4 x half> %1265, <4 x float> %1288) #9, !dbg !14
  %1290 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed491, <4 x half> %1267, <4 x float> %1289) #9, !dbg !14
  %1291 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed493, <4 x half> %1269, <4 x float> %1290) #9, !dbg !14
  %1292 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed495, <4 x half> %1271, <4 x float> %1291) #9, !dbg !14
  %1293 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed497, <4 x half> %1273, <4 x float> %1292) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !19
  fence syncscope("workgroup") release, !dbg !19
  call void @llvm.amdgcn.s.barrier(), !dbg !19
  fence syncscope("workgroup") acquire, !dbg !19
  %1294 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local966, i32 0) #0, !dbg !14
  %1295 = bitcast <4 x i32> %1294 to <8 x half>, !dbg !14
  %1296 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local977, i32 0) #0, !dbg !14
  %1297 = bitcast <4 x i32> %1296 to <8 x half>, !dbg !14
  %1298 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local996, i32 0) #0, !dbg !14
  %1299 = bitcast <4 x i32> %1298 to <8 x half>, !dbg !14
  %1300 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1008, i32 0) #0, !dbg !14
  %1301 = bitcast <4 x i32> %1300 to <8 x half>, !dbg !14
  %1302 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1018, i32 0) #0, !dbg !14
  %1303 = bitcast <4 x i32> %1302 to <8 x half>, !dbg !14
  %1304 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1030, i32 0) #0, !dbg !14
  %1305 = bitcast <4 x i32> %1304 to <8 x half>, !dbg !14
  %1306 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1052, i32 0) #0, !dbg !14
  %1307 = bitcast <4 x i32> %1306 to <8 x half>, !dbg !14
  %1308 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1064, i32 0) #0, !dbg !14
  %1309 = bitcast <4 x i32> %1308 to <8 x half>, !dbg !14
  fence syncscope("workgroup") release, !dbg !19
  call void @llvm.amdgcn.s.barrier(), !dbg !19
  fence syncscope("workgroup") acquire, !dbg !19
  %1310 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %1209, !dbg !19
  %1311 = getelementptr i8, ptr addrspace(1) %1310, i64 128, !dbg !19
  %1312 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1069), !dbg !19
  %1313 = zext i32 %1312 to i64, !dbg !19
  %1314 = inttoptr i64 %1313 to ptr addrspace(3), !dbg !19
  %address.local1100 = getelementptr [2 x i8], ptr addrspace(1) %1311, i64 %address.local1080, !dbg !19
  %1315 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !19
  %1316 = ptrtoint ptr addrspace(1) %address.local1100 to i64, !dbg !19
  %1317 = sub i64 %1316, %1315, !dbg !19
  %dma.buffer.offset1712 = trunc i64 %1317 to i32, !dbg !19
  %1318 = trunc i64 %1315 to i32, !dbg !19
  %1319 = insertelement <4 x i32> poison, i32 %1318, i64 0, !dbg !19
  %1320 = lshr i64 %1315, 32, !dbg !19
  %1321 = trunc i64 %1320 to i32, !dbg !19
  %1322 = insertelement <4 x i32> %1319, i32 %1321, i64 1, !dbg !19
  %1323 = insertelement <4 x i32> %1322, i32 33554432, i64 2, !dbg !19
  %dma.buffer.rsrc1713 = insertelement <4 x i32> %1323, i32 131072, i64 3, !dbg !19
  %1324 = ptrtoint ptr addrspace(3) %1314 to i32, !dbg !19
  %dma.buffer.lds1714 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1324), !dbg !19
  %1325 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1714, i32 %dma.buffer.offset1712, <4 x i32> %dma.buffer.rsrc1713) #10, !dbg !19
  %1326 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1087), !dbg !19
  %1327 = zext i32 %1326 to i64, !dbg !19
  %1328 = inttoptr i64 %1327 to ptr addrspace(3), !dbg !19
  %address.local1101 = getelementptr [2 x i8], ptr addrspace(1) %1311, i64 %address.local1098, !dbg !19
  %1329 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !19
  %1330 = ptrtoint ptr addrspace(1) %address.local1101 to i64, !dbg !19
  %1331 = sub i64 %1330, %1329, !dbg !19
  %dma.buffer.offset1709 = trunc i64 %1331 to i32, !dbg !19
  %1332 = trunc i64 %1329 to i32, !dbg !19
  %1333 = insertelement <4 x i32> poison, i32 %1332, i64 0, !dbg !19
  %1334 = lshr i64 %1329, 32, !dbg !19
  %1335 = trunc i64 %1334 to i32, !dbg !19
  %1336 = insertelement <4 x i32> %1333, i32 %1335, i64 1, !dbg !19
  %1337 = insertelement <4 x i32> %1336, i32 33554432, i64 2, !dbg !19
  %dma.buffer.rsrc1710 = insertelement <4 x i32> %1337, i32 131072, i64 3, !dbg !19
  %1338 = ptrtoint ptr addrspace(3) %1328 to i32, !dbg !19
  %dma.buffer.lds1711 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1338), !dbg !19
  %1339 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1711, i32 %dma.buffer.offset1709, <4 x i32> %dma.buffer.rsrc1710) #10, !dbg !19
  %1340 = shufflevector <8 x half> %1295, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1341 = extractvalue [2 x <8 x float>] %64, 0, !dbg !31
  %1342 = shufflevector <8 x float> %1341, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %1343 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1340, <4 x float> %1342) #9, !dbg !14
  %1344 = shufflevector <8 x half> %1297, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1345 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed469, <4 x half> %1344, <4 x float> %1343) #9, !dbg !14
  %1346 = shufflevector <8 x half> %1299, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1347 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed471, <4 x half> %1346, <4 x float> %1345) #9, !dbg !14
  %1348 = shufflevector <8 x half> %1301, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1349 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed473, <4 x half> %1348, <4 x float> %1347) #9, !dbg !14
  %1350 = shufflevector <8 x half> %1303, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1351 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed475, <4 x half> %1350, <4 x float> %1349) #9, !dbg !14
  %1352 = shufflevector <8 x half> %1305, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1353 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed477, <4 x half> %1352, <4 x float> %1351) #9, !dbg !14
  %1354 = shufflevector <8 x half> %1307, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1355 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed479, <4 x half> %1354, <4 x float> %1353) #9, !dbg !14
  %1356 = shufflevector <8 x half> %1309, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1357 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed481, <4 x half> %1356, <4 x float> %1355) #9, !dbg !14
  %1358 = shufflevector <8 x half> %1295, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1359 = shufflevector <8 x float> %1341, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %1360 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1358, <4 x float> %1359) #9, !dbg !14
  %1361 = shufflevector <8 x half> %1297, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1362 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed469, <4 x half> %1361, <4 x float> %1360) #9, !dbg !14
  %1363 = shufflevector <8 x half> %1299, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1364 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed471, <4 x half> %1363, <4 x float> %1362) #9, !dbg !14
  %1365 = shufflevector <8 x half> %1301, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1366 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed473, <4 x half> %1365, <4 x float> %1364) #9, !dbg !14
  %1367 = shufflevector <8 x half> %1303, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1368 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed475, <4 x half> %1367, <4 x float> %1366) #9, !dbg !14
  %1369 = shufflevector <8 x half> %1305, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1370 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed477, <4 x half> %1369, <4 x float> %1368) #9, !dbg !14
  %1371 = shufflevector <8 x half> %1307, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1372 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed479, <4 x half> %1371, <4 x float> %1370) #9, !dbg !14
  %1373 = shufflevector <8 x half> %1309, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1374 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed481, <4 x half> %1373, <4 x float> %1372) #9, !dbg !14
  %1375 = extractvalue [2 x <8 x float>] %64, 1, !dbg !31
  %1376 = shufflevector <8 x float> %1375, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %1377 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed483, <4 x half> %1340, <4 x float> %1376) #9, !dbg !14
  %1378 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed485, <4 x half> %1344, <4 x float> %1377) #9, !dbg !14
  %1379 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed487, <4 x half> %1346, <4 x float> %1378) #9, !dbg !14
  %1380 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed489, <4 x half> %1348, <4 x float> %1379) #9, !dbg !14
  %1381 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed491, <4 x half> %1350, <4 x float> %1380) #9, !dbg !14
  %1382 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed493, <4 x half> %1352, <4 x float> %1381) #9, !dbg !14
  %1383 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed495, <4 x half> %1354, <4 x float> %1382) #9, !dbg !14
  %1384 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed497, <4 x half> %1356, <4 x float> %1383) #9, !dbg !14
  %1385 = shufflevector <8 x float> %1375, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %1386 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed483, <4 x half> %1358, <4 x float> %1385) #9, !dbg !14
  %1387 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed485, <4 x half> %1361, <4 x float> %1386) #9, !dbg !14
  %1388 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed487, <4 x half> %1363, <4 x float> %1387) #9, !dbg !14
  %1389 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed489, <4 x half> %1365, <4 x float> %1388) #9, !dbg !14
  %1390 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed491, <4 x half> %1367, <4 x float> %1389) #9, !dbg !14
  %1391 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed493, <4 x half> %1369, <4 x float> %1390) #9, !dbg !14
  %1392 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed495, <4 x half> %1371, <4 x float> %1391) #9, !dbg !14
  %1393 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed497, <4 x half> %1373, <4 x float> %1392) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !19
  fence syncscope("workgroup") release, !dbg !19
  call void @llvm.amdgcn.s.barrier(), !dbg !19
  fence syncscope("workgroup") acquire, !dbg !19
  %1394 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local966, i32 0) #0, !dbg !14
  %1395 = bitcast <4 x i32> %1394 to <8 x half>, !dbg !14
  %1396 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local977, i32 0) #0, !dbg !14
  %1397 = bitcast <4 x i32> %1396 to <8 x half>, !dbg !14
  %1398 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local996, i32 0) #0, !dbg !14
  %1399 = bitcast <4 x i32> %1398 to <8 x half>, !dbg !14
  %1400 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1008, i32 0) #0, !dbg !14
  %1401 = bitcast <4 x i32> %1400 to <8 x half>, !dbg !14
  %1402 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1018, i32 0) #0, !dbg !14
  %1403 = bitcast <4 x i32> %1402 to <8 x half>, !dbg !14
  %1404 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1030, i32 0) #0, !dbg !14
  %1405 = bitcast <4 x i32> %1404 to <8 x half>, !dbg !14
  %1406 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1052, i32 0) #0, !dbg !14
  %1407 = bitcast <4 x i32> %1406 to <8 x half>, !dbg !14
  %1408 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1064, i32 0) #0, !dbg !14
  %1409 = bitcast <4 x i32> %1408 to <8 x half>, !dbg !14
  fence syncscope("workgroup") release, !dbg !19
  call void @llvm.amdgcn.s.barrier(), !dbg !19
  fence syncscope("workgroup") acquire, !dbg !19
  %1410 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %1209, !dbg !19
  %1411 = getelementptr i8, ptr addrspace(1) %1410, i64 192, !dbg !19
  %1412 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1069), !dbg !19
  %1413 = zext i32 %1412 to i64, !dbg !19
  %1414 = inttoptr i64 %1413 to ptr addrspace(3), !dbg !19
  %address.local1102 = getelementptr [2 x i8], ptr addrspace(1) %1411, i64 %address.local1080, !dbg !19
  %1415 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !19
  %1416 = ptrtoint ptr addrspace(1) %address.local1102 to i64, !dbg !19
  %1417 = sub i64 %1416, %1415, !dbg !19
  %dma.buffer.offset1706 = trunc i64 %1417 to i32, !dbg !19
  %1418 = trunc i64 %1415 to i32, !dbg !19
  %1419 = insertelement <4 x i32> poison, i32 %1418, i64 0, !dbg !19
  %1420 = lshr i64 %1415, 32, !dbg !19
  %1421 = trunc i64 %1420 to i32, !dbg !19
  %1422 = insertelement <4 x i32> %1419, i32 %1421, i64 1, !dbg !19
  %1423 = insertelement <4 x i32> %1422, i32 33554432, i64 2, !dbg !19
  %dma.buffer.rsrc1707 = insertelement <4 x i32> %1423, i32 131072, i64 3, !dbg !19
  %1424 = ptrtoint ptr addrspace(3) %1414 to i32, !dbg !19
  %dma.buffer.lds1708 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1424), !dbg !19
  %1425 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1708, i32 %dma.buffer.offset1706, <4 x i32> %dma.buffer.rsrc1707) #10, !dbg !19
  %1426 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local1087), !dbg !19
  %1427 = zext i32 %1426 to i64, !dbg !19
  %1428 = inttoptr i64 %1427 to ptr addrspace(3), !dbg !19
  %address.local1103 = getelementptr [2 x i8], ptr addrspace(1) %1411, i64 %address.local1098, !dbg !19
  %1429 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !19
  %1430 = ptrtoint ptr addrspace(1) %address.local1103 to i64, !dbg !19
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
  %1440 = shufflevector <8 x half> %1395, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1441 = extractvalue [2 x <8 x float>] %65, 0, !dbg !31
  %1442 = shufflevector <8 x float> %1441, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %1443 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1440, <4 x float> %1442) #9, !dbg !14
  %1444 = shufflevector <8 x half> %1397, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1445 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed469, <4 x half> %1444, <4 x float> %1443) #9, !dbg !14
  %1446 = shufflevector <8 x half> %1399, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1447 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed471, <4 x half> %1446, <4 x float> %1445) #9, !dbg !14
  %1448 = shufflevector <8 x half> %1401, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1449 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed473, <4 x half> %1448, <4 x float> %1447) #9, !dbg !14
  %1450 = shufflevector <8 x half> %1403, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1451 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed475, <4 x half> %1450, <4 x float> %1449) #9, !dbg !14
  %1452 = shufflevector <8 x half> %1405, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1453 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed477, <4 x half> %1452, <4 x float> %1451) #9, !dbg !14
  %1454 = shufflevector <8 x half> %1407, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1455 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed479, <4 x half> %1454, <4 x float> %1453) #9, !dbg !14
  %1456 = shufflevector <8 x half> %1409, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1457 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed481, <4 x half> %1456, <4 x float> %1455) #9, !dbg !14
  %1458 = shufflevector <8 x half> %1395, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1459 = shufflevector <8 x float> %1441, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %1460 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1458, <4 x float> %1459) #9, !dbg !14
  %1461 = shufflevector <8 x half> %1397, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1462 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed469, <4 x half> %1461, <4 x float> %1460) #9, !dbg !14
  %1463 = shufflevector <8 x half> %1399, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1464 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed471, <4 x half> %1463, <4 x float> %1462) #9, !dbg !14
  %1465 = shufflevector <8 x half> %1401, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1466 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed473, <4 x half> %1465, <4 x float> %1464) #9, !dbg !14
  %1467 = shufflevector <8 x half> %1403, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1468 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed475, <4 x half> %1467, <4 x float> %1466) #9, !dbg !14
  %1469 = shufflevector <8 x half> %1405, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1470 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed477, <4 x half> %1469, <4 x float> %1468) #9, !dbg !14
  %1471 = shufflevector <8 x half> %1407, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1472 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed479, <4 x half> %1471, <4 x float> %1470) #9, !dbg !14
  %1473 = shufflevector <8 x half> %1409, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1474 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed481, <4 x half> %1473, <4 x float> %1472) #9, !dbg !14
  %1475 = extractvalue [2 x <8 x float>] %65, 1, !dbg !31
  %1476 = shufflevector <8 x float> %1475, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %1477 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed483, <4 x half> %1440, <4 x float> %1476) #9, !dbg !14
  %1478 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed485, <4 x half> %1444, <4 x float> %1477) #9, !dbg !14
  %1479 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed487, <4 x half> %1446, <4 x float> %1478) #9, !dbg !14
  %1480 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed489, <4 x half> %1448, <4 x float> %1479) #9, !dbg !14
  %1481 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed491, <4 x half> %1450, <4 x float> %1480) #9, !dbg !14
  %1482 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed493, <4 x half> %1452, <4 x float> %1481) #9, !dbg !14
  %1483 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed495, <4 x half> %1454, <4 x float> %1482) #9, !dbg !14
  %1484 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed497, <4 x half> %1456, <4 x float> %1483) #9, !dbg !14
  %1485 = shufflevector <8 x float> %1475, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %1486 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed483, <4 x half> %1458, <4 x float> %1485) #9, !dbg !14
  %1487 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed485, <4 x half> %1461, <4 x float> %1486) #9, !dbg !14
  %1488 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed487, <4 x half> %1463, <4 x float> %1487) #9, !dbg !14
  %1489 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed489, <4 x half> %1465, <4 x float> %1488) #9, !dbg !14
  %1490 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed491, <4 x half> %1467, <4 x float> %1489) #9, !dbg !14
  %1491 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed493, <4 x half> %1469, <4 x float> %1490) #9, !dbg !14
  %1492 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed495, <4 x half> %1471, <4 x float> %1491) #9, !dbg !14
  %1493 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed497, <4 x half> %1473, <4 x float> %1492) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !19
  fence syncscope("workgroup") release, !dbg !19
  call void @llvm.amdgcn.s.barrier(), !dbg !19
  fence syncscope("workgroup") acquire, !dbg !19
  %1494 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local966, i32 0) #0, !dbg !14
  %1495 = bitcast <4 x i32> %1494 to <8 x half>, !dbg !14
  %1496 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local977, i32 0) #0, !dbg !14
  %1497 = bitcast <4 x i32> %1496 to <8 x half>, !dbg !14
  %1498 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local996, i32 0) #0, !dbg !14
  %1499 = bitcast <4 x i32> %1498 to <8 x half>, !dbg !14
  %1500 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1008, i32 0) #0, !dbg !14
  %1501 = bitcast <4 x i32> %1500 to <8 x half>, !dbg !14
  %1502 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1018, i32 0) #0, !dbg !14
  %1503 = bitcast <4 x i32> %1502 to <8 x half>, !dbg !14
  %1504 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1030, i32 0) #0, !dbg !14
  %1505 = bitcast <4 x i32> %1504 to <8 x half>, !dbg !14
  %1506 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1052, i32 0) #0, !dbg !14
  %1507 = bitcast <4 x i32> %1506 to <8 x half>, !dbg !14
  %1508 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1064, i32 0) #0, !dbg !14
  %1509 = bitcast <4 x i32> %1508 to <8 x half>, !dbg !14
  %1510 = shufflevector <8 x half> %1495, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1511 = extractvalue [2 x <8 x float>] %66, 0, !dbg !31
  %1512 = shufflevector <8 x float> %1511, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %1513 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1510, <4 x float> %1512) #9, !dbg !14
  %1514 = shufflevector <8 x half> %1497, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1515 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed469, <4 x half> %1514, <4 x float> %1513) #9, !dbg !14
  %1516 = shufflevector <8 x half> %1499, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1517 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed471, <4 x half> %1516, <4 x float> %1515) #9, !dbg !14
  %1518 = shufflevector <8 x half> %1501, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1519 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed473, <4 x half> %1518, <4 x float> %1517) #9, !dbg !14
  %1520 = shufflevector <8 x half> %1503, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1521 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed475, <4 x half> %1520, <4 x float> %1519) #9, !dbg !14
  %1522 = shufflevector <8 x half> %1505, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1523 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed477, <4 x half> %1522, <4 x float> %1521) #9, !dbg !14
  %1524 = shufflevector <8 x half> %1507, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1525 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed479, <4 x half> %1524, <4 x float> %1523) #9, !dbg !14
  %1526 = shufflevector <8 x half> %1509, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1527 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed481, <4 x half> %1526, <4 x float> %1525) #9, !dbg !14
  %1528 = shufflevector <8 x half> %1495, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1529 = shufflevector <8 x float> %1511, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %1530 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1528, <4 x float> %1529) #9, !dbg !14
  %1531 = shufflevector <8 x half> %1497, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1532 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed469, <4 x half> %1531, <4 x float> %1530) #9, !dbg !14
  %1533 = shufflevector <8 x half> %1499, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1534 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed471, <4 x half> %1533, <4 x float> %1532) #9, !dbg !14
  %1535 = shufflevector <8 x half> %1501, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1536 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed473, <4 x half> %1535, <4 x float> %1534) #9, !dbg !14
  %1537 = shufflevector <8 x half> %1503, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1538 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed475, <4 x half> %1537, <4 x float> %1536) #9, !dbg !14
  %1539 = shufflevector <8 x half> %1505, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1540 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed477, <4 x half> %1539, <4 x float> %1538) #9, !dbg !14
  %1541 = shufflevector <8 x half> %1507, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1542 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed479, <4 x half> %1541, <4 x float> %1540) #9, !dbg !14
  %1543 = shufflevector <8 x half> %1509, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1544 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed481, <4 x half> %1543, <4 x float> %1542) #9, !dbg !14
  %1545 = extractvalue [2 x <8 x float>] %66, 1, !dbg !31
  %1546 = shufflevector <8 x float> %1545, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %1547 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed483, <4 x half> %1510, <4 x float> %1546) #9, !dbg !14
  %1548 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed485, <4 x half> %1514, <4 x float> %1547) #9, !dbg !14
  %1549 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed487, <4 x half> %1516, <4 x float> %1548) #9, !dbg !14
  %1550 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed489, <4 x half> %1518, <4 x float> %1549) #9, !dbg !14
  %1551 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed491, <4 x half> %1520, <4 x float> %1550) #9, !dbg !14
  %1552 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed493, <4 x half> %1522, <4 x float> %1551) #9, !dbg !14
  %1553 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed495, <4 x half> %1524, <4 x float> %1552) #9, !dbg !14
  %1554 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed497, <4 x half> %1526, <4 x float> %1553) #9, !dbg !14
  %1555 = shufflevector <8 x float> %1545, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %1556 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed483, <4 x half> %1528, <4 x float> %1555) #9, !dbg !14
  %1557 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed485, <4 x half> %1531, <4 x float> %1556) #9, !dbg !14
  %1558 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed487, <4 x half> %1533, <4 x float> %1557) #9, !dbg !14
  %1559 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed489, <4 x half> %1535, <4 x float> %1558) #9, !dbg !14
  %1560 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed491, <4 x half> %1537, <4 x float> %1559) #9, !dbg !14
  %1561 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed493, <4 x half> %1539, <4 x float> %1560) #9, !dbg !14
  %1562 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed495, <4 x half> %1541, <4 x float> %1561) #9, !dbg !14
  %1563 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed497, <4 x half> %1543, <4 x float> %1562) #9, !dbg !14
  %1564 = shufflevector <4 x float> %1257, <4 x float> %1274, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1565 = insertvalue [2 x <8 x float>] poison, <8 x float> %1564, 0, !dbg !15
  %1566 = shufflevector <4 x float> %1284, <4 x float> %1293, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1567 = insertvalue [2 x <8 x float>] %1565, <8 x float> %1566, 1, !dbg !15
  %1568 = shufflevector <4 x float> %1357, <4 x float> %1374, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1569 = insertvalue [2 x <8 x float>] poison, <8 x float> %1568, 0, !dbg !15
  %1570 = shufflevector <4 x float> %1384, <4 x float> %1393, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1571 = insertvalue [2 x <8 x float>] %1569, <8 x float> %1570, 1, !dbg !15
  %1572 = shufflevector <4 x float> %1457, <4 x float> %1474, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1573 = insertvalue [2 x <8 x float>] poison, <8 x float> %1572, 0, !dbg !15
  %1574 = shufflevector <4 x float> %1484, <4 x float> %1493, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1575 = insertvalue [2 x <8 x float>] %1573, <8 x float> %1574, 1, !dbg !15
  %1576 = shufflevector <4 x float> %1527, <4 x float> %1544, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1577 = insertvalue [2 x <8 x float>] poison, <8 x float> %1576, 0, !dbg !15
  %1578 = shufflevector <4 x float> %1554, <4 x float> %1563, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1579 = insertvalue [2 x <8 x float>] %1577, <8 x float> %1578, 1, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %1580 = add nuw nsw i64 %61, 1, !dbg !15
  br label %60, !dbg !15

1581:                                             ; preds = %60
  %1582 = and i32 %34, 16, !dbg !15
  %.not = icmp eq i32 %1582, 0, !dbg !15
  %1583 = and i32 %34, 32, !dbg !15
  %.not62 = icmp eq i32 %1583, 0, !dbg !15
  %1584 = extractvalue [2 x <8 x float>] %66, 0, !dbg !15
  %1585 = extractelement <8 x float> %1584, i64 0, !dbg !15
  %1586 = extractelement <8 x float> %1584, i64 1, !dbg !15
  %1587 = extractelement <8 x float> %1584, i64 2, !dbg !15
  %1588 = extractelement <8 x float> %1584, i64 3, !dbg !15
  %1589 = select i1 %.not, float %1586, float %1585, !dbg !15
  %1590 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !15
  %1591 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1590), !dbg !15
  %1592 = add nuw nsw i32 %1591, 64, !dbg !15
  %1593 = xor i32 %1591, 16, !dbg !15
  %1594 = and i32 %1592, 192, !dbg !15
  %1595 = icmp samesign ult i32 %1593, %1594, !dbg !15
  %1596 = select i1 %1595, i32 %1593, i32 %1591, !dbg !15
  %1597 = shl nuw nsw i32 %1596, 2, !dbg !15
  %1598 = bitcast float %1589 to i32, !dbg !15
  %1599 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1597, i32 %1598), !dbg !15
  %1600 = bitcast i32 %1599 to float, !dbg !15
  %1601 = select i1 %.not, float %1585, float %1600, !dbg !15
  %1602 = select i1 %.not, float %1600, float %1586, !dbg !15
  %1603 = select i1 %.not, float %1588, float %1587, !dbg !15
  %1604 = bitcast float %1603 to i32, !dbg !15
  %1605 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1597, i32 %1604), !dbg !15
  %1606 = bitcast i32 %1605 to float, !dbg !15
  %1607 = select i1 %.not, float %1587, float %1606, !dbg !15
  %1608 = select i1 %.not, float %1606, float %1588, !dbg !15
  %1609 = select i1 %.not62, float %1607, float %1601, !dbg !15
  %1610 = xor i32 %1591, 32, !dbg !15
  %1611 = icmp samesign ult i32 %1610, %1594, !dbg !15
  %1612 = select i1 %1611, i32 %1610, i32 %1591, !dbg !15
  %1613 = shl nuw nsw i32 %1612, 2, !dbg !15
  %1614 = bitcast float %1609 to i32, !dbg !15
  %1615 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1613, i32 %1614), !dbg !15
  %1616 = bitcast i32 %1615 to float, !dbg !15
  %1617 = select i1 %.not62, float %1601, float %1616, !dbg !15
  %1618 = select i1 %.not62, float %1616, float %1607, !dbg !15
  %1619 = select i1 %.not62, float %1608, float %1602, !dbg !15
  %1620 = bitcast float %1619 to i32, !dbg !15
  %1621 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1613, i32 %1620), !dbg !15
  %1622 = bitcast i32 %1621 to float, !dbg !15
  %1623 = select i1 %.not62, float %1602, float %1622, !dbg !15
  %1624 = select i1 %.not62, float %1622, float %1608, !dbg !15
  %1625 = extractvalue [2 x <8 x float>] %65, 0, !dbg !15
  %1626 = extractelement <8 x float> %1625, i64 0, !dbg !15
  %1627 = extractelement <8 x float> %1625, i64 1, !dbg !15
  %1628 = extractelement <8 x float> %1625, i64 2, !dbg !15
  %1629 = extractelement <8 x float> %1625, i64 3, !dbg !15
  %1630 = select i1 %.not, float %1627, float %1626, !dbg !15
  %1631 = bitcast float %1630 to i32, !dbg !15
  %1632 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1597, i32 %1631), !dbg !15
  %1633 = bitcast i32 %1632 to float, !dbg !15
  %1634 = select i1 %.not, float %1626, float %1633, !dbg !15
  %1635 = select i1 %.not, float %1633, float %1627, !dbg !15
  %1636 = select i1 %.not, float %1629, float %1628, !dbg !15
  %1637 = bitcast float %1636 to i32, !dbg !15
  %1638 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1597, i32 %1637), !dbg !15
  %1639 = bitcast i32 %1638 to float, !dbg !15
  %1640 = select i1 %.not, float %1628, float %1639, !dbg !15
  %1641 = select i1 %.not, float %1639, float %1629, !dbg !15
  %1642 = select i1 %.not62, float %1640, float %1634, !dbg !15
  %1643 = bitcast float %1642 to i32, !dbg !15
  %1644 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1613, i32 %1643), !dbg !15
  %1645 = bitcast i32 %1644 to float, !dbg !15
  %1646 = select i1 %.not62, float %1634, float %1645, !dbg !15
  %1647 = select i1 %.not62, float %1645, float %1640, !dbg !15
  %1648 = select i1 %.not62, float %1641, float %1635, !dbg !15
  %1649 = bitcast float %1648 to i32, !dbg !15
  %1650 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1613, i32 %1649), !dbg !15
  %1651 = bitcast i32 %1650 to float, !dbg !15
  %1652 = select i1 %.not62, float %1635, float %1651, !dbg !15
  %1653 = select i1 %.not62, float %1651, float %1641, !dbg !15
  %1654 = extractvalue [2 x <8 x float>] %64, 0, !dbg !15
  %1655 = extractelement <8 x float> %1654, i64 0, !dbg !15
  %1656 = extractelement <8 x float> %1654, i64 1, !dbg !15
  %1657 = extractelement <8 x float> %1654, i64 2, !dbg !15
  %1658 = extractelement <8 x float> %1654, i64 3, !dbg !15
  %1659 = select i1 %.not, float %1656, float %1655, !dbg !15
  %1660 = bitcast float %1659 to i32, !dbg !15
  %1661 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1597, i32 %1660), !dbg !15
  %1662 = bitcast i32 %1661 to float, !dbg !15
  %1663 = select i1 %.not, float %1655, float %1662, !dbg !15
  %1664 = select i1 %.not, float %1662, float %1656, !dbg !15
  %1665 = select i1 %.not, float %1658, float %1657, !dbg !15
  %1666 = bitcast float %1665 to i32, !dbg !15
  %1667 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1597, i32 %1666), !dbg !15
  %1668 = bitcast i32 %1667 to float, !dbg !15
  %1669 = select i1 %.not, float %1657, float %1668, !dbg !15
  %1670 = select i1 %.not, float %1668, float %1658, !dbg !15
  %1671 = select i1 %.not62, float %1669, float %1663, !dbg !15
  %1672 = bitcast float %1671 to i32, !dbg !15
  %1673 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1613, i32 %1672), !dbg !15
  %1674 = bitcast i32 %1673 to float, !dbg !15
  %1675 = select i1 %.not62, float %1663, float %1674, !dbg !15
  %1676 = select i1 %.not62, float %1674, float %1669, !dbg !15
  %1677 = select i1 %.not62, float %1670, float %1664, !dbg !15
  %1678 = bitcast float %1677 to i32, !dbg !15
  %1679 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1613, i32 %1678), !dbg !15
  %1680 = bitcast i32 %1679 to float, !dbg !15
  %1681 = select i1 %.not62, float %1664, float %1680, !dbg !15
  %1682 = select i1 %.not62, float %1680, float %1670, !dbg !15
  %1683 = extractvalue [2 x <8 x float>] %62, 0, !dbg !15
  %1684 = extractelement <8 x float> %1683, i64 0, !dbg !15
  %1685 = extractelement <8 x float> %1683, i64 1, !dbg !15
  %1686 = extractelement <8 x float> %1683, i64 2, !dbg !15
  %1687 = extractelement <8 x float> %1683, i64 3, !dbg !15
  %1688 = select i1 %.not, float %1685, float %1684, !dbg !15
  %1689 = bitcast float %1688 to i32, !dbg !15
  %1690 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1597, i32 %1689), !dbg !15
  %1691 = bitcast i32 %1690 to float, !dbg !15
  %1692 = select i1 %.not, float %1684, float %1691, !dbg !15
  %1693 = select i1 %.not, float %1691, float %1685, !dbg !15
  %1694 = select i1 %.not, float %1687, float %1686, !dbg !15
  %1695 = bitcast float %1694 to i32, !dbg !15
  %1696 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1597, i32 %1695), !dbg !15
  %1697 = bitcast i32 %1696 to float, !dbg !15
  %1698 = select i1 %.not, float %1686, float %1697, !dbg !15
  %1699 = select i1 %.not, float %1697, float %1687, !dbg !15
  %1700 = select i1 %.not62, float %1698, float %1692, !dbg !15
  %1701 = bitcast float %1700 to i32, !dbg !15
  %1702 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1613, i32 %1701), !dbg !15
  %1703 = select i1 %.not62, float %1699, float %1693, !dbg !15
  %1704 = bitcast float %1703 to i32, !dbg !15
  %1705 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1613, i32 %1704), !dbg !15
  %1706 = extractvalue [2 x <1 x float>] %63, 0, !dbg !20
  %1707 = shufflevector <1 x float> %1706, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !20
  %1708 = extractelement <8 x float> %1584, i64 4, !dbg !15
  %1709 = extractelement <8 x float> %1584, i64 5, !dbg !15
  %1710 = extractelement <8 x float> %1584, i64 6, !dbg !15
  %1711 = extractelement <8 x float> %1584, i64 7, !dbg !15
  %1712 = select i1 %.not, float %1709, float %1708, !dbg !15
  %1713 = bitcast float %1712 to i32, !dbg !15
  %1714 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1597, i32 %1713), !dbg !15
  %1715 = bitcast i32 %1714 to float, !dbg !15
  %1716 = select i1 %.not, float %1708, float %1715, !dbg !15
  %1717 = select i1 %.not, float %1715, float %1709, !dbg !15
  %1718 = select i1 %.not, float %1711, float %1710, !dbg !15
  %1719 = bitcast float %1718 to i32, !dbg !15
  %1720 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1597, i32 %1719), !dbg !15
  %1721 = bitcast i32 %1720 to float, !dbg !15
  %1722 = select i1 %.not, float %1710, float %1721, !dbg !15
  %1723 = select i1 %.not, float %1721, float %1711, !dbg !15
  %1724 = select i1 %.not62, float %1722, float %1716, !dbg !15
  %1725 = bitcast float %1724 to i32, !dbg !15
  %1726 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1613, i32 %1725), !dbg !15
  %1727 = bitcast i32 %1726 to float, !dbg !15
  %1728 = select i1 %.not62, float %1716, float %1727, !dbg !15
  %1729 = select i1 %.not62, float %1727, float %1722, !dbg !15
  %1730 = select i1 %.not62, float %1723, float %1717, !dbg !15
  %1731 = bitcast float %1730 to i32, !dbg !15
  %1732 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1613, i32 %1731), !dbg !15
  %1733 = bitcast i32 %1732 to float, !dbg !15
  %1734 = select i1 %.not62, float %1717, float %1733, !dbg !15
  %1735 = select i1 %.not62, float %1733, float %1723, !dbg !15
  %1736 = extractelement <8 x float> %1625, i64 4, !dbg !15
  %1737 = extractelement <8 x float> %1625, i64 5, !dbg !15
  %1738 = extractelement <8 x float> %1625, i64 6, !dbg !15
  %1739 = extractelement <8 x float> %1625, i64 7, !dbg !15
  %1740 = select i1 %.not, float %1737, float %1736, !dbg !15
  %1741 = bitcast float %1740 to i32, !dbg !15
  %1742 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1597, i32 %1741), !dbg !15
  %1743 = bitcast i32 %1742 to float, !dbg !15
  %1744 = select i1 %.not, float %1736, float %1743, !dbg !15
  %1745 = select i1 %.not, float %1743, float %1737, !dbg !15
  %1746 = select i1 %.not, float %1739, float %1738, !dbg !15
  %1747 = bitcast float %1746 to i32, !dbg !15
  %1748 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1597, i32 %1747), !dbg !15
  %1749 = bitcast i32 %1748 to float, !dbg !15
  %1750 = select i1 %.not, float %1738, float %1749, !dbg !15
  %1751 = select i1 %.not, float %1749, float %1739, !dbg !15
  %1752 = select i1 %.not62, float %1750, float %1744, !dbg !15
  %1753 = bitcast float %1752 to i32, !dbg !15
  %1754 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1613, i32 %1753), !dbg !15
  %1755 = bitcast i32 %1754 to float, !dbg !15
  %1756 = select i1 %.not62, float %1744, float %1755, !dbg !15
  %1757 = select i1 %.not62, float %1755, float %1750, !dbg !15
  %1758 = select i1 %.not62, float %1751, float %1745, !dbg !15
  %1759 = bitcast float %1758 to i32, !dbg !15
  %1760 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1613, i32 %1759), !dbg !15
  %1761 = bitcast i32 %1760 to float, !dbg !15
  %1762 = select i1 %.not62, float %1745, float %1761, !dbg !15
  %1763 = select i1 %.not62, float %1761, float %1751, !dbg !15
  %1764 = extractelement <8 x float> %1654, i64 4, !dbg !15
  %1765 = extractelement <8 x float> %1654, i64 5, !dbg !15
  %1766 = extractelement <8 x float> %1654, i64 6, !dbg !15
  %1767 = extractelement <8 x float> %1654, i64 7, !dbg !15
  %1768 = select i1 %.not, float %1765, float %1764, !dbg !15
  %1769 = bitcast float %1768 to i32, !dbg !15
  %1770 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1597, i32 %1769), !dbg !15
  %1771 = bitcast i32 %1770 to float, !dbg !15
  %1772 = select i1 %.not, float %1764, float %1771, !dbg !15
  %1773 = select i1 %.not, float %1771, float %1765, !dbg !15
  %1774 = select i1 %.not, float %1767, float %1766, !dbg !15
  %1775 = bitcast float %1774 to i32, !dbg !15
  %1776 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1597, i32 %1775), !dbg !15
  %1777 = bitcast i32 %1776 to float, !dbg !15
  %1778 = select i1 %.not, float %1766, float %1777, !dbg !15
  %1779 = select i1 %.not, float %1777, float %1767, !dbg !15
  %1780 = select i1 %.not62, float %1778, float %1772, !dbg !15
  %1781 = bitcast float %1780 to i32, !dbg !15
  %1782 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1613, i32 %1781), !dbg !15
  %1783 = bitcast i32 %1782 to float, !dbg !15
  %1784 = select i1 %.not62, float %1772, float %1783, !dbg !15
  %1785 = select i1 %.not62, float %1783, float %1778, !dbg !15
  %1786 = select i1 %.not62, float %1779, float %1773, !dbg !15
  %1787 = bitcast float %1786 to i32, !dbg !15
  %1788 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1613, i32 %1787), !dbg !15
  %1789 = bitcast i32 %1788 to float, !dbg !15
  %1790 = select i1 %.not62, float %1773, float %1789, !dbg !15
  %1791 = select i1 %.not62, float %1789, float %1779, !dbg !15
  %1792 = extractelement <8 x float> %1683, i64 4, !dbg !15
  %1793 = extractelement <8 x float> %1683, i64 5, !dbg !15
  %1794 = extractelement <8 x float> %1683, i64 6, !dbg !15
  %1795 = extractelement <8 x float> %1683, i64 7, !dbg !15
  %1796 = select i1 %.not, float %1793, float %1792, !dbg !15
  %1797 = bitcast float %1796 to i32, !dbg !15
  %1798 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1597, i32 %1797), !dbg !15
  %1799 = bitcast i32 %1798 to float, !dbg !15
  %1800 = select i1 %.not, float %1792, float %1799, !dbg !15
  %1801 = select i1 %.not, float %1799, float %1793, !dbg !15
  %1802 = select i1 %.not, float %1795, float %1794, !dbg !15
  %1803 = bitcast float %1802 to i32, !dbg !15
  %1804 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1597, i32 %1803), !dbg !15
  %1805 = bitcast i32 %1804 to float, !dbg !15
  %1806 = select i1 %.not, float %1794, float %1805, !dbg !15
  %1807 = select i1 %.not, float %1805, float %1795, !dbg !15
  %1808 = select i1 %.not62, float %1806, float %1800, !dbg !15
  %1809 = bitcast float %1808 to i32, !dbg !15
  %1810 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1613, i32 %1809), !dbg !15
  %1811 = select i1 %.not62, float %1807, float %1801, !dbg !15
  %1812 = bitcast float %1811 to i32, !dbg !15
  %1813 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1613, i32 %1812), !dbg !15
  %1814 = extractvalue [2 x <8 x float>] %66, 1, !dbg !15
  %1815 = extractelement <8 x float> %1814, i64 0, !dbg !15
  %1816 = extractelement <8 x float> %1814, i64 1, !dbg !15
  %1817 = extractelement <8 x float> %1814, i64 2, !dbg !15
  %1818 = extractelement <8 x float> %1814, i64 3, !dbg !15
  %1819 = select i1 %.not, float %1816, float %1815, !dbg !15
  %1820 = bitcast float %1819 to i32, !dbg !15
  %1821 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1597, i32 %1820), !dbg !15
  %1822 = bitcast i32 %1821 to float, !dbg !15
  %1823 = select i1 %.not, float %1815, float %1822, !dbg !15
  %1824 = select i1 %.not, float %1822, float %1816, !dbg !15
  %1825 = select i1 %.not, float %1818, float %1817, !dbg !15
  %1826 = bitcast float %1825 to i32, !dbg !15
  %1827 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1597, i32 %1826), !dbg !15
  %1828 = bitcast i32 %1827 to float, !dbg !15
  %1829 = select i1 %.not, float %1817, float %1828, !dbg !15
  %1830 = select i1 %.not, float %1828, float %1818, !dbg !15
  %1831 = select i1 %.not62, float %1829, float %1823, !dbg !15
  %1832 = bitcast float %1831 to i32, !dbg !15
  %1833 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1613, i32 %1832), !dbg !15
  %1834 = bitcast i32 %1833 to float, !dbg !15
  %1835 = select i1 %.not62, float %1823, float %1834, !dbg !15
  %1836 = select i1 %.not62, float %1834, float %1829, !dbg !15
  %1837 = select i1 %.not62, float %1830, float %1824, !dbg !15
  %1838 = bitcast float %1837 to i32, !dbg !15
  %1839 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1613, i32 %1838), !dbg !15
  %1840 = bitcast i32 %1839 to float, !dbg !15
  %1841 = select i1 %.not62, float %1824, float %1840, !dbg !15
  %1842 = select i1 %.not62, float %1840, float %1830, !dbg !15
  %1843 = extractvalue [2 x <8 x float>] %65, 1, !dbg !15
  %1844 = extractelement <8 x float> %1843, i64 0, !dbg !15
  %1845 = extractelement <8 x float> %1843, i64 1, !dbg !15
  %1846 = extractelement <8 x float> %1843, i64 2, !dbg !15
  %1847 = extractelement <8 x float> %1843, i64 3, !dbg !15
  %1848 = select i1 %.not, float %1845, float %1844, !dbg !15
  %1849 = bitcast float %1848 to i32, !dbg !15
  %1850 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1597, i32 %1849), !dbg !15
  %1851 = bitcast i32 %1850 to float, !dbg !15
  %1852 = select i1 %.not, float %1844, float %1851, !dbg !15
  %1853 = select i1 %.not, float %1851, float %1845, !dbg !15
  %1854 = select i1 %.not, float %1847, float %1846, !dbg !15
  %1855 = bitcast float %1854 to i32, !dbg !15
  %1856 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1597, i32 %1855), !dbg !15
  %1857 = bitcast i32 %1856 to float, !dbg !15
  %1858 = select i1 %.not, float %1846, float %1857, !dbg !15
  %1859 = select i1 %.not, float %1857, float %1847, !dbg !15
  %1860 = select i1 %.not62, float %1858, float %1852, !dbg !15
  %1861 = bitcast float %1860 to i32, !dbg !15
  %1862 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1613, i32 %1861), !dbg !15
  %1863 = bitcast i32 %1862 to float, !dbg !15
  %1864 = select i1 %.not62, float %1852, float %1863, !dbg !15
  %1865 = select i1 %.not62, float %1863, float %1858, !dbg !15
  %1866 = select i1 %.not62, float %1859, float %1853, !dbg !15
  %1867 = bitcast float %1866 to i32, !dbg !15
  %1868 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1613, i32 %1867), !dbg !15
  %1869 = bitcast i32 %1868 to float, !dbg !15
  %1870 = select i1 %.not62, float %1853, float %1869, !dbg !15
  %1871 = select i1 %.not62, float %1869, float %1859, !dbg !15
  %1872 = extractvalue [2 x <8 x float>] %64, 1, !dbg !15
  %1873 = extractelement <8 x float> %1872, i64 0, !dbg !15
  %1874 = extractelement <8 x float> %1872, i64 1, !dbg !15
  %1875 = extractelement <8 x float> %1872, i64 2, !dbg !15
  %1876 = extractelement <8 x float> %1872, i64 3, !dbg !15
  %1877 = select i1 %.not, float %1874, float %1873, !dbg !15
  %1878 = bitcast float %1877 to i32, !dbg !15
  %1879 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1597, i32 %1878), !dbg !15
  %1880 = bitcast i32 %1879 to float, !dbg !15
  %1881 = select i1 %.not, float %1873, float %1880, !dbg !15
  %1882 = select i1 %.not, float %1880, float %1874, !dbg !15
  %1883 = select i1 %.not, float %1876, float %1875, !dbg !15
  %1884 = bitcast float %1883 to i32, !dbg !15
  %1885 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1597, i32 %1884), !dbg !15
  %1886 = bitcast i32 %1885 to float, !dbg !15
  %1887 = select i1 %.not, float %1875, float %1886, !dbg !15
  %1888 = select i1 %.not, float %1886, float %1876, !dbg !15
  %1889 = select i1 %.not62, float %1887, float %1881, !dbg !15
  %1890 = bitcast float %1889 to i32, !dbg !15
  %1891 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1613, i32 %1890), !dbg !15
  %1892 = bitcast i32 %1891 to float, !dbg !15
  %1893 = select i1 %.not62, float %1881, float %1892, !dbg !15
  %1894 = select i1 %.not62, float %1892, float %1887, !dbg !15
  %1895 = select i1 %.not62, float %1888, float %1882, !dbg !15
  %1896 = bitcast float %1895 to i32, !dbg !15
  %1897 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1613, i32 %1896), !dbg !15
  %1898 = bitcast i32 %1897 to float, !dbg !15
  %1899 = select i1 %.not62, float %1882, float %1898, !dbg !15
  %1900 = select i1 %.not62, float %1898, float %1888, !dbg !15
  %1901 = extractvalue [2 x <8 x float>] %62, 1, !dbg !15
  %1902 = extractelement <8 x float> %1901, i64 0, !dbg !15
  %1903 = extractelement <8 x float> %1901, i64 1, !dbg !15
  %1904 = extractelement <8 x float> %1901, i64 2, !dbg !15
  %1905 = extractelement <8 x float> %1901, i64 3, !dbg !15
  %1906 = select i1 %.not, float %1903, float %1902, !dbg !15
  %1907 = bitcast float %1906 to i32, !dbg !15
  %1908 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1597, i32 %1907), !dbg !15
  %1909 = bitcast i32 %1908 to float, !dbg !15
  %1910 = select i1 %.not, float %1902, float %1909, !dbg !15
  %1911 = select i1 %.not, float %1909, float %1903, !dbg !15
  %1912 = select i1 %.not, float %1905, float %1904, !dbg !15
  %1913 = bitcast float %1912 to i32, !dbg !15
  %1914 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1597, i32 %1913), !dbg !15
  %1915 = bitcast i32 %1914 to float, !dbg !15
  %1916 = select i1 %.not, float %1904, float %1915, !dbg !15
  %1917 = select i1 %.not, float %1915, float %1905, !dbg !15
  %1918 = select i1 %.not62, float %1916, float %1910, !dbg !15
  %1919 = bitcast float %1918 to i32, !dbg !15
  %1920 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1613, i32 %1919), !dbg !15
  %1921 = select i1 %.not62, float %1917, float %1911, !dbg !15
  %1922 = bitcast float %1921 to i32, !dbg !15
  %1923 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1613, i32 %1922), !dbg !15
  %1924 = extractvalue [2 x <1 x float>] %63, 1, !dbg !20
  %1925 = shufflevector <1 x float> %1924, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !20
  %1926 = extractelement <8 x float> %1814, i64 4, !dbg !15
  %1927 = extractelement <8 x float> %1814, i64 5, !dbg !15
  %1928 = extractelement <8 x float> %1814, i64 6, !dbg !15
  %1929 = extractelement <8 x float> %1814, i64 7, !dbg !15
  %1930 = select i1 %.not, float %1927, float %1926, !dbg !15
  %1931 = bitcast float %1930 to i32, !dbg !15
  %1932 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1597, i32 %1931), !dbg !15
  %1933 = bitcast i32 %1932 to float, !dbg !15
  %1934 = select i1 %.not, float %1926, float %1933, !dbg !15
  %1935 = select i1 %.not, float %1933, float %1927, !dbg !15
  %1936 = select i1 %.not, float %1929, float %1928, !dbg !15
  %1937 = bitcast float %1936 to i32, !dbg !15
  %1938 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1597, i32 %1937), !dbg !15
  %1939 = bitcast i32 %1938 to float, !dbg !15
  %1940 = select i1 %.not, float %1928, float %1939, !dbg !15
  %1941 = select i1 %.not, float %1939, float %1929, !dbg !15
  %1942 = select i1 %.not62, float %1940, float %1934, !dbg !15
  %1943 = bitcast float %1942 to i32, !dbg !15
  %1944 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1613, i32 %1943), !dbg !15
  %1945 = bitcast i32 %1944 to float, !dbg !15
  %1946 = select i1 %.not62, float %1934, float %1945, !dbg !15
  %1947 = select i1 %.not62, float %1945, float %1940, !dbg !15
  %1948 = select i1 %.not62, float %1941, float %1935, !dbg !15
  %1949 = bitcast float %1948 to i32, !dbg !15
  %1950 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1613, i32 %1949), !dbg !15
  %1951 = bitcast i32 %1950 to float, !dbg !15
  %1952 = select i1 %.not62, float %1935, float %1951, !dbg !15
  %1953 = select i1 %.not62, float %1951, float %1941, !dbg !15
  %1954 = extractelement <8 x float> %1843, i64 4, !dbg !15
  %1955 = extractelement <8 x float> %1843, i64 5, !dbg !15
  %1956 = extractelement <8 x float> %1843, i64 6, !dbg !15
  %1957 = extractelement <8 x float> %1843, i64 7, !dbg !15
  %1958 = select i1 %.not, float %1955, float %1954, !dbg !15
  %1959 = bitcast float %1958 to i32, !dbg !15
  %1960 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1597, i32 %1959), !dbg !15
  %1961 = bitcast i32 %1960 to float, !dbg !15
  %1962 = select i1 %.not, float %1954, float %1961, !dbg !15
  %1963 = select i1 %.not, float %1961, float %1955, !dbg !15
  %1964 = select i1 %.not, float %1957, float %1956, !dbg !15
  %1965 = bitcast float %1964 to i32, !dbg !15
  %1966 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1597, i32 %1965), !dbg !15
  %1967 = bitcast i32 %1966 to float, !dbg !15
  %1968 = select i1 %.not, float %1956, float %1967, !dbg !15
  %1969 = select i1 %.not, float %1967, float %1957, !dbg !15
  %1970 = select i1 %.not62, float %1968, float %1962, !dbg !15
  %1971 = bitcast float %1970 to i32, !dbg !15
  %1972 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1613, i32 %1971), !dbg !15
  %1973 = bitcast i32 %1972 to float, !dbg !15
  %1974 = select i1 %.not62, float %1962, float %1973, !dbg !15
  %1975 = select i1 %.not62, float %1973, float %1968, !dbg !15
  %1976 = select i1 %.not62, float %1969, float %1963, !dbg !15
  %1977 = bitcast float %1976 to i32, !dbg !15
  %1978 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1613, i32 %1977), !dbg !15
  %1979 = bitcast i32 %1978 to float, !dbg !15
  %1980 = select i1 %.not62, float %1963, float %1979, !dbg !15
  %1981 = select i1 %.not62, float %1979, float %1969, !dbg !15
  %1982 = extractelement <8 x float> %1872, i64 4, !dbg !15
  %1983 = extractelement <8 x float> %1872, i64 5, !dbg !15
  %1984 = extractelement <8 x float> %1872, i64 6, !dbg !15
  %1985 = extractelement <8 x float> %1872, i64 7, !dbg !15
  %1986 = select i1 %.not, float %1983, float %1982, !dbg !15
  %1987 = bitcast float %1986 to i32, !dbg !15
  %1988 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1597, i32 %1987), !dbg !15
  %1989 = bitcast i32 %1988 to float, !dbg !15
  %1990 = select i1 %.not, float %1982, float %1989, !dbg !15
  %1991 = select i1 %.not, float %1989, float %1983, !dbg !15
  %1992 = select i1 %.not, float %1985, float %1984, !dbg !15
  %1993 = bitcast float %1992 to i32, !dbg !15
  %1994 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1597, i32 %1993), !dbg !15
  %1995 = bitcast i32 %1994 to float, !dbg !15
  %1996 = select i1 %.not, float %1984, float %1995, !dbg !15
  %1997 = select i1 %.not, float %1995, float %1985, !dbg !15
  %1998 = select i1 %.not62, float %1996, float %1990, !dbg !15
  %1999 = bitcast float %1998 to i32, !dbg !15
  %2000 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1613, i32 %1999), !dbg !15
  %2001 = bitcast i32 %2000 to float, !dbg !15
  %2002 = select i1 %.not62, float %1990, float %2001, !dbg !15
  %2003 = select i1 %.not62, float %2001, float %1996, !dbg !15
  %2004 = select i1 %.not62, float %1997, float %1991, !dbg !15
  %2005 = bitcast float %2004 to i32, !dbg !15
  %2006 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1613, i32 %2005), !dbg !15
  %2007 = bitcast i32 %2006 to float, !dbg !15
  %2008 = select i1 %.not62, float %1991, float %2007, !dbg !15
  %2009 = select i1 %.not62, float %2007, float %1997, !dbg !15
  %2010 = extractelement <8 x float> %1901, i64 4, !dbg !15
  %2011 = extractelement <8 x float> %1901, i64 5, !dbg !15
  %2012 = extractelement <8 x float> %1901, i64 6, !dbg !15
  %2013 = extractelement <8 x float> %1901, i64 7, !dbg !15
  %2014 = select i1 %.not, float %2011, float %2010, !dbg !15
  %2015 = bitcast float %2014 to i32, !dbg !15
  %2016 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1597, i32 %2015), !dbg !15
  %2017 = bitcast i32 %2016 to float, !dbg !15
  %2018 = select i1 %.not, float %2010, float %2017, !dbg !15
  %2019 = select i1 %.not, float %2017, float %2011, !dbg !15
  %2020 = select i1 %.not, float %2013, float %2012, !dbg !15
  %2021 = bitcast float %2020 to i32, !dbg !15
  %2022 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1597, i32 %2021), !dbg !15
  %2023 = bitcast i32 %2022 to float, !dbg !15
  %2024 = select i1 %.not, float %2012, float %2023, !dbg !15
  %2025 = select i1 %.not, float %2023, float %2013, !dbg !15
  %2026 = select i1 %.not62, float %2024, float %2018, !dbg !15
  %2027 = bitcast float %2026 to i32, !dbg !15
  %2028 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1613, i32 %2027), !dbg !15
  %2029 = select i1 %.not62, float %2025, float %2019, !dbg !15
  %2030 = bitcast float %2029 to i32, !dbg !15
  %2031 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1613, i32 %2030), !dbg !15
  br i1 true, label %2032, label %2096, !dbg !22

2032:                                             ; preds = %1581
  %2033 = bitcast i32 %2028 to float, !dbg !15
  %2034 = select i1 %.not62, float %2018, float %2033, !dbg !15
  %2035 = insertelement <4 x float> poison, float %2034, i64 0, !dbg !20
  %2036 = bitcast i32 %2031 to float, !dbg !15
  %2037 = select i1 %.not62, float %2019, float %2036, !dbg !15
  %2038 = insertelement <4 x float> %2035, float %2037, i64 1, !dbg !20
  %2039 = select i1 %.not62, float %2033, float %2024, !dbg !15
  %2040 = insertelement <4 x float> %2038, float %2039, i64 2, !dbg !20
  %2041 = select i1 %.not62, float %2036, float %2025, !dbg !15
  %2042 = insertelement <4 x float> %2040, float %2041, i64 3, !dbg !20
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1925, !dbg !20
  %normalized = fmul <4 x float> %2042, %row.reciprocal, !dbg !20
  %2043 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !21
  %2044 = extractelement <4 x half> %2043, i64 3, !dbg !21
  %2045 = extractelement <4 x half> %2043, i64 2, !dbg !21
  %2046 = extractelement <4 x half> %2043, i64 1, !dbg !21
  %2047 = extractelement <4 x half> %2043, i64 0, !dbg !21
  %2048 = bitcast i32 %1920 to float, !dbg !15
  %2049 = select i1 %.not62, float %1910, float %2048, !dbg !15
  %2050 = insertelement <4 x float> poison, float %2049, i64 0, !dbg !20
  %2051 = bitcast i32 %1923 to float, !dbg !15
  %2052 = select i1 %.not62, float %1911, float %2051, !dbg !15
  %2053 = insertelement <4 x float> %2050, float %2052, i64 1, !dbg !20
  %2054 = select i1 %.not62, float %2048, float %1916, !dbg !15
  %2055 = insertelement <4 x float> %2053, float %2054, i64 2, !dbg !20
  %2056 = select i1 %.not62, float %2051, float %1917, !dbg !15
  %2057 = insertelement <4 x float> %2055, float %2056, i64 3, !dbg !20
  %normalized130 = fmul <4 x float> %2057, %row.reciprocal, !dbg !20
  %2058 = fptrunc <4 x float> %normalized130 to <4 x half>, !dbg !21
  %2059 = extractelement <4 x half> %2058, i64 3, !dbg !21
  %2060 = extractelement <4 x half> %2058, i64 2, !dbg !21
  %2061 = extractelement <4 x half> %2058, i64 1, !dbg !21
  %2062 = extractelement <4 x half> %2058, i64 0, !dbg !21
  %2063 = bitcast i32 %1810 to float, !dbg !15
  %2064 = select i1 %.not62, float %1800, float %2063, !dbg !15
  %2065 = insertelement <4 x float> poison, float %2064, i64 0, !dbg !20
  %2066 = bitcast i32 %1813 to float, !dbg !15
  %2067 = select i1 %.not62, float %1801, float %2066, !dbg !15
  %2068 = insertelement <4 x float> %2065, float %2067, i64 1, !dbg !20
  %2069 = select i1 %.not62, float %2063, float %1806, !dbg !15
  %2070 = insertelement <4 x float> %2068, float %2069, i64 2, !dbg !20
  %2071 = select i1 %.not62, float %2066, float %1807, !dbg !15
  %2072 = insertelement <4 x float> %2070, float %2071, i64 3, !dbg !20
  %row.reciprocal131 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1707, !dbg !20
  %normalized132 = fmul <4 x float> %2072, %row.reciprocal131, !dbg !20
  %2073 = fptrunc <4 x float> %normalized132 to <4 x half>, !dbg !21
  %2074 = extractelement <4 x half> %2073, i64 3, !dbg !21
  %2075 = extractelement <4 x half> %2073, i64 2, !dbg !21
  %2076 = extractelement <4 x half> %2073, i64 1, !dbg !21
  %2077 = extractelement <4 x half> %2073, i64 0, !dbg !21
  %2078 = bitcast i32 %1702 to float, !dbg !15
  %2079 = select i1 %.not62, float %1692, float %2078, !dbg !15
  %2080 = insertelement <4 x float> poison, float %2079, i64 0, !dbg !20
  %2081 = bitcast i32 %1705 to float, !dbg !15
  %2082 = select i1 %.not62, float %1693, float %2081, !dbg !15
  %2083 = insertelement <4 x float> %2080, float %2082, i64 1, !dbg !20
  %2084 = select i1 %.not62, float %2078, float %1698, !dbg !15
  %2085 = insertelement <4 x float> %2083, float %2084, i64 2, !dbg !20
  %2086 = select i1 %.not62, float %2081, float %1699, !dbg !15
  %2087 = insertelement <4 x float> %2085, float %2086, i64 3, !dbg !20
  %normalized133 = fmul <4 x float> %2087, %row.reciprocal131, !dbg !20
  %2088 = fptrunc <4 x float> %normalized133 to <4 x half>, !dbg !21
  %2089 = extractelement <4 x half> %2088, i64 3, !dbg !21
  %2090 = extractelement <4 x half> %2088, i64 2, !dbg !21
  %2091 = extractelement <4 x half> %2088, i64 1, !dbg !21
  %2092 = extractelement <4 x half> %2088, i64 0, !dbg !21
  %2093 = add nsw i64 %15, %25, !dbg !22
  %2094 = shl i64 %2093, 19, !dbg !22
  %2095 = and i64 %2094, 16252928, !dbg !22
  %address.thread1104 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !22, !range !34, !frisk.register_lifetimes !1
  %address.local1105 = zext nneg i32 %address.thread1104 to i64, !dbg !12
  %address.local1106 = add i64 %28, %address.local1105, !dbg !22
  %address.local1107 = lshr i64 %address.local1105, 6, !dbg !12
  %address.local1108 = shl nuw nsw i64 %address.local1107, 4, !dbg !22
  %address.local1109 = add i64 %address.local1106, %address.local1108, !dbg !22
  %address.local1110 = and i64 %address.local1105, 240, !dbg !22
  %address.local1111 = sub i64 %address.local1109, %address.local1110, !dbg !22
  %address.local1112 = add i64 %address.local1111, %29, !dbg !22
  %address.local1113 = shl nuw nsw i64 %address.local1112, 7, !dbg !22
  %address.local1114 = add nuw nsw i64 %2095, %address.local1113, !dbg !22
  %address.local1115 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1114, !dbg !22
  %address.local1116 = lshr i64 %address.local1105, 2, !dbg !12
  %address.local1117 = and i64 %address.local1116, 12, !dbg !12
  %address.local1118 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1115, i64 %address.local1117, !dbg !22
  store half %2092, ptr addrspace(1) %address.local1118, align 2, !dbg !22
  %address.local1119 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1114, !dbg !22
  %address.local1120 = or disjoint i64 %address.local1117, 1, !dbg !22
  %address.local1121 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1119, i64 %address.local1120, !dbg !22
  store half %2091, ptr addrspace(1) %address.local1121, align 2, !dbg !22
  %address.local1122 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1114, !dbg !22
  %address.local1123 = or disjoint i64 %address.local1117, 2, !dbg !22
  %address.local1124 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1122, i64 %address.local1123, !dbg !22
  store half %2090, ptr addrspace(1) %address.local1124, align 2, !dbg !22
  %address.local1125 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1114, !dbg !22
  %address.local1126 = or disjoint i64 %address.local1117, 3, !dbg !22
  %address.local1127 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1125, i64 %address.local1126, !dbg !22
  store half %2089, ptr addrspace(1) %address.local1127, align 2, !dbg !22
  %address.local1128 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1114, !dbg !22
  %address.local1129 = or disjoint i64 %address.local1117, 16, !dbg !22
  %address.local1130 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1128, i64 %address.local1129, !dbg !22
  store half %2077, ptr addrspace(1) %address.local1130, align 2, !dbg !22
  %address.local1131 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1114, !dbg !22
  %address.local1132 = or disjoint i64 %address.local1117, 17, !dbg !22
  %address.local1133 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1131, i64 %address.local1132, !dbg !22
  store half %2076, ptr addrspace(1) %address.local1133, align 2, !dbg !22
  %address.local1134 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1114, !dbg !22
  %address.local1135 = or disjoint i64 %address.local1117, 18, !dbg !22
  %address.local1136 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1134, i64 %address.local1135, !dbg !22
  store half %2075, ptr addrspace(1) %address.local1136, align 2, !dbg !22
  %address.local1137 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1114, !dbg !22
  %address.local1138 = or disjoint i64 %address.local1117, 19, !dbg !22
  %address.local1139 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1137, i64 %address.local1138, !dbg !22
  store half %2074, ptr addrspace(1) %address.local1139, align 2, !dbg !22
  %address.local1140 = shl i64 %address.local1112, 7, !dbg !22
  %address.local1141 = add i64 %address.local1140, 8192, !dbg !22
  %address.local1142 = add nuw nsw i64 %2095, %address.local1141, !dbg !22
  %address.local1143 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1142, !dbg !22
  %address.local1144 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1143, i64 %address.local1117, !dbg !22
  store half %2062, ptr addrspace(1) %address.local1144, align 2, !dbg !22
  %address.local1145 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1142, !dbg !22
  %address.local1146 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1145, i64 %address.local1120, !dbg !22
  store half %2061, ptr addrspace(1) %address.local1146, align 2, !dbg !22
  %address.local1147 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1142, !dbg !22
  %address.local1148 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1147, i64 %address.local1123, !dbg !22
  store half %2060, ptr addrspace(1) %address.local1148, align 2, !dbg !22
  %address.local1149 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1142, !dbg !22
  %address.local1150 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1149, i64 %address.local1126, !dbg !22
  store half %2059, ptr addrspace(1) %address.local1150, align 2, !dbg !22
  %address.local1151 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1142, !dbg !22
  %address.local1152 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1151, i64 %address.local1129, !dbg !22
  store half %2047, ptr addrspace(1) %address.local1152, align 2, !dbg !22
  %address.local1153 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1142, !dbg !22
  %address.local1154 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1153, i64 %address.local1132, !dbg !22
  store half %2046, ptr addrspace(1) %address.local1154, align 2, !dbg !22
  %address.local1155 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1142, !dbg !22
  %address.local1156 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1155, i64 %address.local1135, !dbg !22
  store half %2045, ptr addrspace(1) %address.local1156, align 2, !dbg !22
  %address.local1157 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1142, !dbg !22
  %address.local1158 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1157, i64 %address.local1138, !dbg !22
  store half %2044, ptr addrspace(1) %address.local1158, align 2, !dbg !22
  br label %2096, !dbg !22

2096:                                             ; preds = %2032, %1581
  br i1 true, label %2097, label %2135, !dbg !22

2097:                                             ; preds = %2096
  %2098 = insertelement <4 x float> poison, float %2002, i64 0, !dbg !20
  %2099 = insertelement <4 x float> %2098, float %2008, i64 1, !dbg !20
  %2100 = insertelement <4 x float> %2099, float %2003, i64 2, !dbg !20
  %2101 = insertelement <4 x float> %2100, float %2009, i64 3, !dbg !20
  %row.reciprocal134 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1925, !dbg !20
  %normalized135 = fmul <4 x float> %2101, %row.reciprocal134, !dbg !20
  %2102 = fptrunc <4 x float> %normalized135 to <4 x half>, !dbg !21
  %2103 = extractelement <4 x half> %2102, i64 3, !dbg !21
  %2104 = extractelement <4 x half> %2102, i64 2, !dbg !21
  %2105 = extractelement <4 x half> %2102, i64 1, !dbg !21
  %2106 = extractelement <4 x half> %2102, i64 0, !dbg !21
  %2107 = insertelement <4 x float> poison, float %1893, i64 0, !dbg !20
  %2108 = insertelement <4 x float> %2107, float %1899, i64 1, !dbg !20
  %2109 = insertelement <4 x float> %2108, float %1894, i64 2, !dbg !20
  %2110 = insertelement <4 x float> %2109, float %1900, i64 3, !dbg !20
  %normalized136 = fmul <4 x float> %2110, %row.reciprocal134, !dbg !20
  %2111 = fptrunc <4 x float> %normalized136 to <4 x half>, !dbg !21
  %2112 = extractelement <4 x half> %2111, i64 3, !dbg !21
  %2113 = extractelement <4 x half> %2111, i64 2, !dbg !21
  %2114 = extractelement <4 x half> %2111, i64 1, !dbg !21
  %2115 = extractelement <4 x half> %2111, i64 0, !dbg !21
  %2116 = insertelement <4 x float> poison, float %1784, i64 0, !dbg !20
  %2117 = insertelement <4 x float> %2116, float %1790, i64 1, !dbg !20
  %2118 = insertelement <4 x float> %2117, float %1785, i64 2, !dbg !20
  %2119 = insertelement <4 x float> %2118, float %1791, i64 3, !dbg !20
  %row.reciprocal137 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1707, !dbg !20
  %normalized138 = fmul <4 x float> %2119, %row.reciprocal137, !dbg !20
  %2120 = fptrunc <4 x float> %normalized138 to <4 x half>, !dbg !21
  %2121 = extractelement <4 x half> %2120, i64 3, !dbg !21
  %2122 = extractelement <4 x half> %2120, i64 2, !dbg !21
  %2123 = extractelement <4 x half> %2120, i64 1, !dbg !21
  %2124 = extractelement <4 x half> %2120, i64 0, !dbg !21
  %2125 = insertelement <4 x float> poison, float %1675, i64 0, !dbg !20
  %2126 = insertelement <4 x float> %2125, float %1681, i64 1, !dbg !20
  %2127 = insertelement <4 x float> %2126, float %1676, i64 2, !dbg !20
  %2128 = insertelement <4 x float> %2127, float %1682, i64 3, !dbg !20
  %normalized139 = fmul <4 x float> %2128, %row.reciprocal137, !dbg !20
  %2129 = fptrunc <4 x float> %normalized139 to <4 x half>, !dbg !21
  %2130 = extractelement <4 x half> %2129, i64 3, !dbg !21
  %2131 = extractelement <4 x half> %2129, i64 2, !dbg !21
  %2132 = extractelement <4 x half> %2129, i64 1, !dbg !21
  %2133 = extractelement <4 x half> %2129, i64 0, !dbg !21
  %2134 = shl nuw nsw i64 %27, 19, !dbg !22
  %address.thread1159 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !22, !range !34, !frisk.register_lifetimes !1
  %address.local1160 = zext nneg i32 %address.thread1159 to i64, !dbg !12
  %address.local1161 = add i64 %28, %address.local1160, !dbg !22
  %address.local1162 = lshr i64 %address.local1160, 6, !dbg !12
  %address.local1163 = shl nuw nsw i64 %address.local1162, 4, !dbg !22
  %address.local1164 = add i64 %address.local1161, %address.local1163, !dbg !22
  %address.local1165 = and i64 %address.local1160, 240, !dbg !22
  %address.local1166 = sub i64 %address.local1164, %address.local1165, !dbg !22
  %address.local1167 = add i64 %address.local1166, %29, !dbg !22
  %address.local1168 = shl nuw nsw i64 %address.local1167, 7, !dbg !22
  %address.local1169 = add nuw nsw i64 %2134, %address.local1168, !dbg !22
  %address.local1170 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1169, !dbg !22
  %address.local1171 = lshr i64 %address.local1160, 2, !dbg !12
  %address.local1172 = and i64 %address.local1171, 12, !dbg !12
  %address.local1173 = or disjoint i64 %address.local1172, 32, !dbg !22
  %address.local1174 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1170, i64 %address.local1173, !dbg !22
  store half %2133, ptr addrspace(1) %address.local1174, align 2, !dbg !22
  %address.local1175 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1169, !dbg !22
  %address.local1176 = or disjoint i64 %address.local1172, 33, !dbg !22
  %address.local1177 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1175, i64 %address.local1176, !dbg !22
  store half %2132, ptr addrspace(1) %address.local1177, align 2, !dbg !22
  %address.local1178 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1169, !dbg !22
  %address.local1179 = or disjoint i64 %address.local1172, 34, !dbg !22
  %address.local1180 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1178, i64 %address.local1179, !dbg !22
  store half %2131, ptr addrspace(1) %address.local1180, align 2, !dbg !22
  %address.local1181 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1169, !dbg !22
  %address.local1182 = or disjoint i64 %address.local1172, 35, !dbg !22
  %address.local1183 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1181, i64 %address.local1182, !dbg !22
  store half %2130, ptr addrspace(1) %address.local1183, align 2, !dbg !22
  %address.local1184 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1169, !dbg !22
  %address.local1185 = or disjoint i64 %address.local1172, 48, !dbg !22
  %address.local1186 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1184, i64 %address.local1185, !dbg !22
  store half %2124, ptr addrspace(1) %address.local1186, align 2, !dbg !22
  %address.local1187 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1169, !dbg !22
  %address.local1188 = or disjoint i64 %address.local1172, 49, !dbg !22
  %address.local1189 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1187, i64 %address.local1188, !dbg !22
  store half %2123, ptr addrspace(1) %address.local1189, align 2, !dbg !22
  %address.local1190 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1169, !dbg !22
  %address.local1191 = or disjoint i64 %address.local1172, 50, !dbg !22
  %address.local1192 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1190, i64 %address.local1191, !dbg !22
  store half %2122, ptr addrspace(1) %address.local1192, align 2, !dbg !22
  %address.local1193 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1169, !dbg !22
  %address.local1194 = or i64 %address.local1171, 51, !dbg !22
  %address.local1195 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1193, i64 %address.local1194, !dbg !22
  store half %2121, ptr addrspace(1) %address.local1195, align 2, !dbg !22
  %address.local1196 = shl i64 %address.local1167, 7, !dbg !22
  %address.local1197 = add i64 %address.local1196, 8192, !dbg !22
  %address.local1198 = add nuw nsw i64 %2134, %address.local1197, !dbg !22
  %address.local1199 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1198, !dbg !22
  %address.local1200 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1199, i64 %address.local1173, !dbg !22
  store half %2115, ptr addrspace(1) %address.local1200, align 2, !dbg !22
  %address.local1201 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1198, !dbg !22
  %address.local1202 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1201, i64 %address.local1176, !dbg !22
  store half %2114, ptr addrspace(1) %address.local1202, align 2, !dbg !22
  %address.local1203 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1198, !dbg !22
  %address.local1204 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1203, i64 %address.local1179, !dbg !22
  store half %2113, ptr addrspace(1) %address.local1204, align 2, !dbg !22
  %address.local1205 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1198, !dbg !22
  %address.local1206 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1205, i64 %address.local1182, !dbg !22
  store half %2112, ptr addrspace(1) %address.local1206, align 2, !dbg !22
  %address.local1207 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1198, !dbg !22
  %address.local1208 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1207, i64 %address.local1185, !dbg !22
  store half %2106, ptr addrspace(1) %address.local1208, align 2, !dbg !22
  %address.local1209 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1198, !dbg !22
  %address.local1210 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1209, i64 %address.local1188, !dbg !22
  store half %2105, ptr addrspace(1) %address.local1210, align 2, !dbg !22
  %address.local1211 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1198, !dbg !22
  %address.local1212 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1211, i64 %address.local1191, !dbg !22
  store half %2104, ptr addrspace(1) %address.local1212, align 2, !dbg !22
  %address.local1213 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1198, !dbg !22
  %address.local1214 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1213, i64 %address.local1194, !dbg !22
  store half %2103, ptr addrspace(1) %address.local1214, align 2, !dbg !22
  br label %2135, !dbg !22

2135:                                             ; preds = %2097, %2096
  br i1 true, label %2136, label %2174, !dbg !22

2136:                                             ; preds = %2135
  %2137 = insertelement <4 x float> poison, float %1974, i64 0, !dbg !20
  %2138 = insertelement <4 x float> %2137, float %1980, i64 1, !dbg !20
  %2139 = insertelement <4 x float> %2138, float %1975, i64 2, !dbg !20
  %2140 = insertelement <4 x float> %2139, float %1981, i64 3, !dbg !20
  %row.reciprocal140 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1925, !dbg !20
  %normalized141 = fmul <4 x float> %2140, %row.reciprocal140, !dbg !20
  %2141 = fptrunc <4 x float> %normalized141 to <4 x half>, !dbg !21
  %2142 = extractelement <4 x half> %2141, i64 3, !dbg !21
  %2143 = extractelement <4 x half> %2141, i64 2, !dbg !21
  %2144 = extractelement <4 x half> %2141, i64 1, !dbg !21
  %2145 = extractelement <4 x half> %2141, i64 0, !dbg !21
  %2146 = insertelement <4 x float> poison, float %1864, i64 0, !dbg !20
  %2147 = insertelement <4 x float> %2146, float %1870, i64 1, !dbg !20
  %2148 = insertelement <4 x float> %2147, float %1865, i64 2, !dbg !20
  %2149 = insertelement <4 x float> %2148, float %1871, i64 3, !dbg !20
  %normalized142 = fmul <4 x float> %2149, %row.reciprocal140, !dbg !20
  %2150 = fptrunc <4 x float> %normalized142 to <4 x half>, !dbg !21
  %2151 = extractelement <4 x half> %2150, i64 3, !dbg !21
  %2152 = extractelement <4 x half> %2150, i64 2, !dbg !21
  %2153 = extractelement <4 x half> %2150, i64 1, !dbg !21
  %2154 = extractelement <4 x half> %2150, i64 0, !dbg !21
  %2155 = insertelement <4 x float> poison, float %1756, i64 0, !dbg !20
  %2156 = insertelement <4 x float> %2155, float %1762, i64 1, !dbg !20
  %2157 = insertelement <4 x float> %2156, float %1757, i64 2, !dbg !20
  %2158 = insertelement <4 x float> %2157, float %1763, i64 3, !dbg !20
  %row.reciprocal143 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1707, !dbg !20
  %normalized144 = fmul <4 x float> %2158, %row.reciprocal143, !dbg !20
  %2159 = fptrunc <4 x float> %normalized144 to <4 x half>, !dbg !21
  %2160 = extractelement <4 x half> %2159, i64 3, !dbg !21
  %2161 = extractelement <4 x half> %2159, i64 2, !dbg !21
  %2162 = extractelement <4 x half> %2159, i64 1, !dbg !21
  %2163 = extractelement <4 x half> %2159, i64 0, !dbg !21
  %2164 = insertelement <4 x float> poison, float %1646, i64 0, !dbg !20
  %2165 = insertelement <4 x float> %2164, float %1652, i64 1, !dbg !20
  %2166 = insertelement <4 x float> %2165, float %1647, i64 2, !dbg !20
  %2167 = insertelement <4 x float> %2166, float %1653, i64 3, !dbg !20
  %normalized145 = fmul <4 x float> %2167, %row.reciprocal143, !dbg !20
  %2168 = fptrunc <4 x float> %normalized145 to <4 x half>, !dbg !21
  %2169 = extractelement <4 x half> %2168, i64 3, !dbg !21
  %2170 = extractelement <4 x half> %2168, i64 2, !dbg !21
  %2171 = extractelement <4 x half> %2168, i64 1, !dbg !21
  %2172 = extractelement <4 x half> %2168, i64 0, !dbg !21
  %2173 = shl nuw nsw i64 %27, 19, !dbg !22
  %address.thread1215 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !22, !range !34, !frisk.register_lifetimes !1
  %address.local1216 = zext nneg i32 %address.thread1215 to i64, !dbg !12
  %address.local1217 = add i64 %28, %address.local1216, !dbg !22
  %address.local1218 = lshr i64 %address.local1216, 6, !dbg !12
  %address.local1219 = shl nuw nsw i64 %address.local1218, 4, !dbg !22
  %address.local1220 = add i64 %address.local1217, %address.local1219, !dbg !22
  %address.local1221 = and i64 %address.local1216, 240, !dbg !22
  %address.local1222 = sub i64 %address.local1220, %address.local1221, !dbg !22
  %address.local1223 = add i64 %address.local1222, %29, !dbg !22
  %address.local1224 = shl nuw nsw i64 %address.local1223, 7, !dbg !22
  %address.local1225 = add nuw nsw i64 %2173, %address.local1224, !dbg !22
  %address.local1226 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1225, !dbg !22
  %address.local1227 = lshr i64 %address.local1216, 2, !dbg !12
  %address.local1228 = and i64 %address.local1227, 12, !dbg !12
  %address.local1229 = or disjoint i64 %address.local1228, 64, !dbg !22
  %address.local1230 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1226, i64 %address.local1229, !dbg !22
  store half %2172, ptr addrspace(1) %address.local1230, align 2, !dbg !22
  %address.local1231 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1225, !dbg !22
  %address.local1232 = or disjoint i64 %address.local1228, 65, !dbg !22
  %address.local1233 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1231, i64 %address.local1232, !dbg !22
  store half %2171, ptr addrspace(1) %address.local1233, align 2, !dbg !22
  %address.local1234 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1225, !dbg !22
  %address.local1235 = or disjoint i64 %address.local1228, 66, !dbg !22
  %address.local1236 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1234, i64 %address.local1235, !dbg !22
  store half %2170, ptr addrspace(1) %address.local1236, align 2, !dbg !22
  %address.local1237 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1225, !dbg !22
  %address.local1238 = or disjoint i64 %address.local1228, 67, !dbg !22
  %address.local1239 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1237, i64 %address.local1238, !dbg !22
  store half %2169, ptr addrspace(1) %address.local1239, align 2, !dbg !22
  %address.local1240 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1225, !dbg !22
  %address.local1241 = or disjoint i64 %address.local1228, 80, !dbg !22
  %address.local1242 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1240, i64 %address.local1241, !dbg !22
  store half %2163, ptr addrspace(1) %address.local1242, align 2, !dbg !22
  %address.local1243 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1225, !dbg !22
  %address.local1244 = or disjoint i64 %address.local1228, 81, !dbg !22
  %address.local1245 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1243, i64 %address.local1244, !dbg !22
  store half %2162, ptr addrspace(1) %address.local1245, align 2, !dbg !22
  %address.local1246 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1225, !dbg !22
  %address.local1247 = or disjoint i64 %address.local1228, 82, !dbg !22
  %address.local1248 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1246, i64 %address.local1247, !dbg !22
  store half %2161, ptr addrspace(1) %address.local1248, align 2, !dbg !22
  %address.local1249 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1225, !dbg !22
  %address.local1250 = or disjoint i64 %address.local1228, 83, !dbg !22
  %address.local1251 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1249, i64 %address.local1250, !dbg !22
  store half %2160, ptr addrspace(1) %address.local1251, align 2, !dbg !22
  %address.local1252 = shl i64 %address.local1223, 7, !dbg !22
  %address.local1253 = add i64 %address.local1252, 8192, !dbg !22
  %address.local1254 = add nuw nsw i64 %2173, %address.local1253, !dbg !22
  %address.local1255 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1254, !dbg !22
  %address.local1256 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1255, i64 %address.local1229, !dbg !22
  store half %2154, ptr addrspace(1) %address.local1256, align 2, !dbg !22
  %address.local1257 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1254, !dbg !22
  %address.local1258 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1257, i64 %address.local1232, !dbg !22
  store half %2153, ptr addrspace(1) %address.local1258, align 2, !dbg !22
  %address.local1259 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1254, !dbg !22
  %address.local1260 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1259, i64 %address.local1235, !dbg !22
  store half %2152, ptr addrspace(1) %address.local1260, align 2, !dbg !22
  %address.local1261 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1254, !dbg !22
  %address.local1262 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1261, i64 %address.local1238, !dbg !22
  store half %2151, ptr addrspace(1) %address.local1262, align 2, !dbg !22
  %address.local1263 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1254, !dbg !22
  %address.local1264 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1263, i64 %address.local1241, !dbg !22
  store half %2145, ptr addrspace(1) %address.local1264, align 2, !dbg !22
  %address.local1265 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1254, !dbg !22
  %address.local1266 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1265, i64 %address.local1244, !dbg !22
  store half %2144, ptr addrspace(1) %address.local1266, align 2, !dbg !22
  %address.local1267 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1254, !dbg !22
  %address.local1268 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1267, i64 %address.local1247, !dbg !22
  store half %2143, ptr addrspace(1) %address.local1268, align 2, !dbg !22
  %address.local1269 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1254, !dbg !22
  %address.local1270 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1269, i64 %address.local1250, !dbg !22
  store half %2142, ptr addrspace(1) %address.local1270, align 2, !dbg !22
  br label %2174, !dbg !22

2174:                                             ; preds = %2136, %2135
  br i1 true, label %2175, label %2213, !dbg !22

2175:                                             ; preds = %2174
  %2176 = insertelement <4 x float> poison, float %1946, i64 0, !dbg !20
  %2177 = insertelement <4 x float> %2176, float %1952, i64 1, !dbg !20
  %2178 = insertelement <4 x float> %2177, float %1947, i64 2, !dbg !20
  %2179 = insertelement <4 x float> %2178, float %1953, i64 3, !dbg !20
  %row.reciprocal146 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1925, !dbg !20
  %normalized147 = fmul <4 x float> %2179, %row.reciprocal146, !dbg !20
  %2180 = fptrunc <4 x float> %normalized147 to <4 x half>, !dbg !21
  %2181 = extractelement <4 x half> %2180, i64 3, !dbg !21
  %2182 = extractelement <4 x half> %2180, i64 2, !dbg !21
  %2183 = extractelement <4 x half> %2180, i64 1, !dbg !21
  %2184 = extractelement <4 x half> %2180, i64 0, !dbg !21
  %2185 = insertelement <4 x float> poison, float %1835, i64 0, !dbg !20
  %2186 = insertelement <4 x float> %2185, float %1841, i64 1, !dbg !20
  %2187 = insertelement <4 x float> %2186, float %1836, i64 2, !dbg !20
  %2188 = insertelement <4 x float> %2187, float %1842, i64 3, !dbg !20
  %normalized148 = fmul <4 x float> %2188, %row.reciprocal146, !dbg !20
  %2189 = fptrunc <4 x float> %normalized148 to <4 x half>, !dbg !21
  %2190 = extractelement <4 x half> %2189, i64 3, !dbg !21
  %2191 = extractelement <4 x half> %2189, i64 2, !dbg !21
  %2192 = extractelement <4 x half> %2189, i64 1, !dbg !21
  %2193 = extractelement <4 x half> %2189, i64 0, !dbg !21
  %2194 = insertelement <4 x float> poison, float %1728, i64 0, !dbg !20
  %2195 = insertelement <4 x float> %2194, float %1734, i64 1, !dbg !20
  %2196 = insertelement <4 x float> %2195, float %1729, i64 2, !dbg !20
  %2197 = insertelement <4 x float> %2196, float %1735, i64 3, !dbg !20
  %row.reciprocal149 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1707, !dbg !20
  %normalized150 = fmul <4 x float> %2197, %row.reciprocal149, !dbg !20
  %2198 = fptrunc <4 x float> %normalized150 to <4 x half>, !dbg !21
  %2199 = extractelement <4 x half> %2198, i64 3, !dbg !21
  %2200 = extractelement <4 x half> %2198, i64 2, !dbg !21
  %2201 = extractelement <4 x half> %2198, i64 1, !dbg !21
  %2202 = extractelement <4 x half> %2198, i64 0, !dbg !21
  %2203 = insertelement <4 x float> poison, float %1617, i64 0, !dbg !20
  %2204 = insertelement <4 x float> %2203, float %1623, i64 1, !dbg !20
  %2205 = insertelement <4 x float> %2204, float %1618, i64 2, !dbg !20
  %2206 = insertelement <4 x float> %2205, float %1624, i64 3, !dbg !20
  %normalized151 = fmul <4 x float> %2206, %row.reciprocal149, !dbg !20
  %2207 = fptrunc <4 x float> %normalized151 to <4 x half>, !dbg !21
  %2208 = extractelement <4 x half> %2207, i64 3, !dbg !21
  %2209 = extractelement <4 x half> %2207, i64 2, !dbg !21
  %2210 = extractelement <4 x half> %2207, i64 1, !dbg !21
  %2211 = extractelement <4 x half> %2207, i64 0, !dbg !21
  %2212 = shl nuw nsw i64 %27, 19, !dbg !22
  %address.thread1271 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !22, !range !34, !frisk.register_lifetimes !1
  %address.local1272 = zext nneg i32 %address.thread1271 to i64, !dbg !12
  %address.local1273 = add i64 %28, %address.local1272, !dbg !22
  %address.local1274 = lshr i64 %address.local1272, 6, !dbg !12
  %address.local1275 = shl nuw nsw i64 %address.local1274, 4, !dbg !22
  %address.local1276 = add i64 %address.local1273, %address.local1275, !dbg !22
  %address.local1277 = and i64 %address.local1272, 240, !dbg !22
  %address.local1278 = sub i64 %address.local1276, %address.local1277, !dbg !22
  %address.local1279 = add i64 %address.local1278, %29, !dbg !22
  %address.local1280 = shl nuw nsw i64 %address.local1279, 7, !dbg !22
  %address.local1281 = add nuw nsw i64 %2212, %address.local1280, !dbg !22
  %address.local1282 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1281, !dbg !22
  %address.local1283 = lshr i64 %address.local1272, 2, !dbg !12
  %address.local1284 = and i64 %address.local1283, 12, !dbg !12
  %address.local1285 = or disjoint i64 %address.local1284, 96, !dbg !22
  %address.local1286 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1282, i64 %address.local1285, !dbg !22
  store half %2211, ptr addrspace(1) %address.local1286, align 2, !dbg !22
  %address.local1287 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1281, !dbg !22
  %address.local1288 = or disjoint i64 %address.local1284, 97, !dbg !22
  %address.local1289 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1287, i64 %address.local1288, !dbg !22
  store half %2210, ptr addrspace(1) %address.local1289, align 2, !dbg !22
  %address.local1290 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1281, !dbg !22
  %address.local1291 = or disjoint i64 %address.local1284, 98, !dbg !22
  %address.local1292 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1290, i64 %address.local1291, !dbg !22
  store half %2209, ptr addrspace(1) %address.local1292, align 2, !dbg !22
  %address.local1293 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1281, !dbg !22
  %address.local1294 = or disjoint i64 %address.local1284, 99, !dbg !22
  %address.local1295 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1293, i64 %address.local1294, !dbg !22
  store half %2208, ptr addrspace(1) %address.local1295, align 2, !dbg !22
  %address.local1296 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1281, !dbg !22
  %address.local1297 = or disjoint i64 %address.local1284, 112, !dbg !22
  %address.local1298 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1296, i64 %address.local1297, !dbg !22
  store half %2202, ptr addrspace(1) %address.local1298, align 2, !dbg !22
  %address.local1299 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1281, !dbg !22
  %address.local1300 = or disjoint i64 %address.local1284, 113, !dbg !22
  %address.local1301 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1299, i64 %address.local1300, !dbg !22
  store half %2201, ptr addrspace(1) %address.local1301, align 2, !dbg !22
  %address.local1302 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1281, !dbg !22
  %address.local1303 = or disjoint i64 %address.local1284, 114, !dbg !22
  %address.local1304 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1302, i64 %address.local1303, !dbg !22
  store half %2200, ptr addrspace(1) %address.local1304, align 2, !dbg !22
  %address.local1305 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1281, !dbg !22
  %address.local1306 = or i64 %address.local1283, 115, !dbg !22
  %address.local1307 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1305, i64 %address.local1306, !dbg !22
  store half %2199, ptr addrspace(1) %address.local1307, align 2, !dbg !22
  %address.local1308 = shl i64 %address.local1279, 7, !dbg !22
  %address.local1309 = add i64 %address.local1308, 8192, !dbg !22
  %address.local1310 = add nuw nsw i64 %2212, %address.local1309, !dbg !22
  %address.local1311 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1310, !dbg !22
  %address.local1312 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1311, i64 %address.local1285, !dbg !22
  store half %2193, ptr addrspace(1) %address.local1312, align 2, !dbg !22
  %address.local1313 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1310, !dbg !22
  %address.local1314 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1313, i64 %address.local1288, !dbg !22
  store half %2192, ptr addrspace(1) %address.local1314, align 2, !dbg !22
  %address.local1315 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1310, !dbg !22
  %address.local1316 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1315, i64 %address.local1291, !dbg !22
  store half %2191, ptr addrspace(1) %address.local1316, align 2, !dbg !22
  %address.local1317 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1310, !dbg !22
  %address.local1318 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1317, i64 %address.local1294, !dbg !22
  store half %2190, ptr addrspace(1) %address.local1318, align 2, !dbg !22
  %address.local1319 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1310, !dbg !22
  %address.local1320 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1319, i64 %address.local1297, !dbg !22
  store half %2184, ptr addrspace(1) %address.local1320, align 2, !dbg !22
  %address.local1321 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1310, !dbg !22
  %address.local1322 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1321, i64 %address.local1300, !dbg !22
  store half %2183, ptr addrspace(1) %address.local1322, align 2, !dbg !22
  %address.local1323 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1310, !dbg !22
  %address.local1324 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1323, i64 %address.local1303, !dbg !22
  store half %2182, ptr addrspace(1) %address.local1324, align 2, !dbg !22
  %address.local1325 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1310, !dbg !22
  %address.local1326 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1325, i64 %address.local1306, !dbg !22
  store half %2181, ptr addrspace(1) %address.local1326, align 2, !dbg !22
  br label %2213, !dbg !22

2213:                                             ; preds = %2175, %2174
  ret void, !dbg !5
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.y() #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.z() #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #2

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32) #3

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.hi(i32, i32) #3

; Function Attrs: convergent nocallback nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.ds.bpermute(i32, i32) #4

; Function Attrs: convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.readfirstlane.i32(i32) #5

; Function Attrs: nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.asyncmark() #6

; Function Attrs: nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.wait.asyncmark(i16 immarg) #6

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier() #7

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.amdgcn.exp2.f32(float) #8

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.sched.barrier(i32 immarg) #7

attributes #0 = { convergent willreturn memory(read) }
attributes #1 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1" "frisk.frag_prefetch"="4" "frisk.global_prefetch"="8" "frisk.register_comparisons"="1" "frisk.register_lifetimes"="1" "uniform-work-group-size" }
attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }
attributes #4 = { convergent nocallback nofree nounwind willreturn memory(none) }
attributes #5 = { convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none) }
attributes #6 = { nocallback nofree nounwind willreturn }
attributes #7 = { convergent nocallback nofree nounwind willreturn }
attributes #8 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #9 = { convergent nounwind willreturn memory(none) }
attributes #10 = { convergent nounwind }
attributes #11 = { memory(inaccessiblemem: readwrite) }

!llvm.dbg.cu = !{!3}
!llvm.module.flags = !{!33}

!0 = !DIFile(filename: "nohint_unrolled.mlir", directory: "generated_ir/pipeline_cleanup")
!1 = !{}
!2 = !DISubroutineType(types: !1)
!3 = distinct !DICompileUnit(language: DW_LANG_C, file: !0, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!4 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !0, file: !0, line: 22, type: !2, scopeLine: 22, spFlags: DISPFlagDefinition, unit: !3)
!5 = !DILocation(line: 22, column: 3, scope: !4)
!9 = !DILocation(line: 46, column: 17, scope: !4)
!11 = !DILocation(line: 66, column: 22, scope: !4)
!12 = !DILocation(line: 68, column: 24, scope: !4)
!13 = !DILocation(line: 73, column: 19, scope: !4)
!14 = !DILocation(line: 93, column: 23, scope: !4)
!15 = !DILocation(line: 56, column: 24, scope: !4)
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
