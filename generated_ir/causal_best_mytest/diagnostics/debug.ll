; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/causal_best_mytest/causal.input.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/causal_best_mytest/causal.input.mlir"

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
  %15 = lshr i64 %10, 2, !dbg !5
  %16 = shl i64 %14, 7, !dbg !10
  %17 = lshr i64 %10, 10, !dbg !12
  %18 = add nuw nsw i64 %15, %17, !dbg !12
  %19 = and i64 %18, 31, !dbg !12
  %20 = shl nsw i64 %14, 7, !dbg !12
  %21 = mul nsw i64 %17, -4096, !dbg !12
  %22 = add i64 %20, %21, !dbg !12
  %23 = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !13, !range !39
  %24 = zext nneg i32 %23 to i64, !dbg !13
  %25 = and i64 %24, 240, !dbg !17
  %26 = shl nuw nsw i64 %24, 2, !dbg !7
  %27 = shl nuw nsw i64 %24, 2, !dbg !7
  %28 = and i64 %27, 1008, !dbg !7
  %29 = sub nsw i64 %26, %28, !dbg !7
  %30 = and i32 %23, 1, !dbg !7
  %index.compact405 = zext nneg i32 %30 to i64, !dbg !7
  %31 = or disjoint i64 %29, %index.compact405, !dbg !7
  %32 = add nuw nsw i64 %31, 1, !dbg !7
  %33 = sdiv i64 %32, 2, !dbg !7
  %34 = add i64 %16, %24, !dbg !17
  %35 = lshr i64 %24, 2, !dbg !17
  %36 = and i64 %35, 48, !dbg !17
  %37 = add i64 %34, %36, !dbg !17
  %38 = sub i64 %37, %25, !dbg !17
  %39 = and i64 %24, 48, !dbg !17
  %40 = add i64 %38, 1, !dbg !18
  %41 = add i64 %38, 65, !dbg !18
  br label %42, !dbg !8

42:                                               ; preds = %236, %4
  %43 = phi i64 [ %1152, %236 ], [ 0, %4 ], !dbg !8
  %44 = phi [2 x <8 x float>] [ %1137, %236 ], [ zeroinitializer, %4 ], !dbg !8
  %45 = phi [2 x <1 x float>] [ %791, %236 ], [ zeroinitializer, %4 ], !dbg !8
  %46 = phi [2 x <1 x float>] [ %1139, %236 ], [ [<1 x float> splat (float 0xFFF0000000000000), <1 x float> splat (float 0xFFF0000000000000)], %4 ], !dbg !8
  %47 = phi [2 x <8 x float>] [ %1143, %236 ], [ zeroinitializer, %4 ], !dbg !8
  %48 = phi [2 x <8 x float>] [ %1147, %236 ], [ zeroinitializer, %4 ], !dbg !8
  %49 = phi [2 x <8 x float>] [ %1151, %236 ], [ zeroinitializer, %4 ], !dbg !8
  %.not = icmp samesign ugt i64 %43, %14, !dbg !8
  br i1 %.not, label %1153, label %50, !dbg !8

50:                                               ; preds = %42
  %51 = shl nsw i64 %43, 7, !dbg !15
  %52 = lshr i64 %43, 5, !dbg !15
  %53 = shl i64 %52, 12, !dbg !15
  %54 = sub i64 %51, %53, !dbg !15
  br label %55, !dbg !22

55:                                               ; preds = %59, %50
  %56 = phi i64 [ %235, %59 ], [ 0, %50 ], !dbg !22
  %57 = phi [2 x <32 x float>] [ %234, %59 ], [ zeroinitializer, %50 ], !dbg !22
  %58 = icmp samesign ult i64 %56, 4, !dbg !22
  br i1 %58, label %59, label %236, !dbg !22

59:                                               ; preds = %55
  %60 = shl nuw nsw i64 %56, 5, !dbg !22
  %.idx = shl nuw nsw i64 %19, 20, !dbg !12
  %.idx102 = shl i64 %22, 8, !dbg !12
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %23) #9, !dbg !12, !range !39, !frisk.register_lifetimes !1
  %address.local = zext nneg i32 %address.thread to i64, !dbg !13
  %address.local462 = shl nuw nsw i64 %address.local, 3, !dbg !12
  %address.local463 = and i64 %address.local462, 1536, !dbg !12
  %address.local464 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local463, !dbg !12
  %address.local465 = ptrtoint ptr addrspace(3) %address.local464 to i64, !dbg !12
  %address.local466 = trunc i64 %address.local465 to i32, !dbg !12
  %61 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local466), !dbg !12
  %address.local467 = shl nuw nsw i64 %address.local, 8, !dbg !12
  %address.local468 = and i64 %address.local467, 32512, !dbg !12
  %62 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !12
  %63 = add i64 %.idx, %.idx102, !dbg !12
  %.idx1521 = shl nuw nsw i64 %56, 6, !dbg !12
  %64 = or disjoint i64 %63, %.idx1521, !dbg !12
  %65 = lshr i64 %address.local, 3, !dbg !12
  %address.local472.idx = and i64 %65, 16, !dbg !12
  %66 = or disjoint i64 %address.local468, %address.local472.idx, !dbg !12
  %67 = or disjoint i64 %64, %66, !dbg !12
  %dma.buffer.offset1479 = trunc i64 %67 to i32, !dbg !12
  %68 = insertelement <2 x i64> poison, i64 %62, i64 0, !dbg !12
  %69 = bitcast <2 x i64> %68 to <4 x i32>, !dbg !12
  %dma.buffer.rsrc1480 = shufflevector <4 x i32> %69, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !12
  %dma.buffer.lds1481 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %61), !dbg !12
  %70 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1481, i32 %dma.buffer.offset1479, <4 x i32> %dma.buffer.rsrc1480) #10, !dbg !12
  %address.local473 = or disjoint i64 %address.local, 256, !dbg !12
  %address.local474 = shl nuw nsw i64 %address.local473, 3, !dbg !12
  %address.local475 = and i64 %address.local474, 3584, !dbg !12
  %address.local476 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local475, !dbg !12
  %address.local477 = ptrtoint ptr addrspace(3) %address.local476 to i64, !dbg !12
  %address.local478 = trunc i64 %address.local477 to i32, !dbg !12
  %71 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local478), !dbg !12
  %address.local479 = shl nuw nsw i64 %address.local, 8, !dbg !12
  %address.local480 = and i64 %address.local479, 32512, !dbg !12
  %72 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !12
  %73 = add i64 %.idx, %.idx102, !dbg !12
  %.idx1522 = shl nuw nsw i64 %56, 6, !dbg !12
  %74 = or disjoint i64 %73, %.idx1522, !dbg !12
  %75 = or disjoint i64 %74, %address.local480, !dbg !12
  %76 = lshr i64 %address.local473, 3, !dbg !12
  %address.local484.idx = and i64 %76, 48, !dbg !12
  %77 = or disjoint i64 %75, %address.local484.idx, !dbg !12
  %dma.buffer.offset1476 = trunc i64 %77 to i32, !dbg !12
  %78 = insertelement <2 x i64> poison, i64 %72, i64 0, !dbg !12
  %79 = bitcast <2 x i64> %78 to <4 x i32>, !dbg !12
  %dma.buffer.rsrc1477 = shufflevector <4 x i32> %79, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !12
  %dma.buffer.lds1478 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %71), !dbg !12
  %80 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1478, i32 %dma.buffer.offset1476, <4 x i32> %dma.buffer.rsrc1477) #10, !dbg !12
  %81 = shl nuw nsw i64 %56, 10, !dbg !15
  %82 = add nuw i64 %43, %81, !dbg !15
  %.lobit103 = ashr i64 %82, 63, !dbg !15
  %83 = xor i64 %82, %.lobit103, !dbg !15
  %84 = sdiv i64 %83, 4096, !dbg !15
  %.lobit104 = ashr i64 %82, 63, !dbg !15
  %85 = xor i64 %84, %.lobit104, !dbg !15
  %86 = add nsw i64 %85, %15, !dbg !15
  %87 = add nuw nsw i64 %60, %52, !dbg !15
  %88 = and i64 %87, 127, !dbg !15
  %89 = shl i64 %86, 19, !dbg !15
  %90 = and i64 %89, 16252928, !dbg !15
  %.idx105 = shl i64 %54, 8, !dbg !15
  %address.local485 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local463, !dbg !15
  %address.local486 = ptrtoint ptr addrspace(3) %address.local485 to i64, !dbg !15
  %address.local487 = trunc i64 %address.local486 to i32, !dbg !15
  %91 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local487), !dbg !15
  %address.local491 = and i64 %address.local, 123, !dbg !15
  %address.local492 = lshr i64 %address.local, 2, !dbg !15
  %address.local493 = and i64 %address.local492, 4, !dbg !15
  %address.local494 = add nuw nsw i64 %address.local493, %address.local, !dbg !15
  %address.local495 = and i64 %address.local494, 4, !dbg !15
  %address.local496 = or disjoint i64 %address.local491, %address.local495, !dbg !15
  %address.local497 = shl nuw nsw i64 %address.local496, 8, !dbg !15
  %92 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !15
  %.idx15231525 = or disjoint i64 %90, %88, !dbg !15
  %93 = shl nuw nsw i64 %.idx15231525, 1, !dbg !15
  %94 = add i64 %93, %.idx105, !dbg !15
  %95 = lshr i64 %address.local, 3, !dbg !15
  %address.local490.idx = and i64 %95, 16, !dbg !15
  %96 = or disjoint i64 %address.local490.idx, %address.local497, !dbg !15
  %97 = add i64 %94, %96, !dbg !15
  %dma.buffer.offset1473 = trunc i64 %97 to i32, !dbg !15
  %98 = insertelement <2 x i64> poison, i64 %92, i64 0, !dbg !15
  %99 = bitcast <2 x i64> %98 to <4 x i32>, !dbg !15
  %dma.buffer.rsrc1474 = shufflevector <4 x i32> %99, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %dma.buffer.lds1475 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %91), !dbg !15
  %100 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1475, i32 %dma.buffer.offset1473, <4 x i32> %dma.buffer.rsrc1474) #10, !dbg !15
  %address.local499 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local475, !dbg !15
  %address.local500 = ptrtoint ptr addrspace(3) %address.local499 to i64, !dbg !15
  %address.local501 = trunc i64 %address.local500 to i32, !dbg !15
  %101 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local501), !dbg !15
  %address.local505 = and i64 %address.local, 123, !dbg !15
  %address.local506 = lshr i64 %address.local, 2, !dbg !15
  %address.local507 = and i64 %address.local506, 4, !dbg !15
  %address.local508 = add nuw nsw i64 %address.local507, %address.local, !dbg !15
  %address.local509 = and i64 %address.local508, 4, !dbg !15
  %address.local510 = or disjoint i64 %address.local505, %address.local509, !dbg !15
  %address.local511 = shl nuw nsw i64 %address.local510, 8, !dbg !15
  %102 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !15
  %.idx15261528 = or disjoint i64 %90, %88, !dbg !15
  %103 = shl nuw nsw i64 %.idx15261528, 1, !dbg !15
  %104 = add i64 %103, %.idx105, !dbg !15
  %105 = lshr i64 %address.local473, 3, !dbg !15
  %address.local504.idx = and i64 %105, 48, !dbg !15
  %106 = add nuw nsw i64 %104, %address.local504.idx, !dbg !15
  %107 = add i64 %106, %address.local511, !dbg !15
  %dma.buffer.offset = trunc i64 %107 to i32, !dbg !15
  %108 = insertelement <2 x i64> poison, i64 %102, i64 0, !dbg !15
  %109 = bitcast <2 x i64> %108 to <4 x i32>, !dbg !15
  %dma.buffer.rsrc = shufflevector <4 x i32> %109, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %dma.buffer.lds = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %101), !dbg !15
  %110 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc) #10, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %address.local513 = lshr i32 %address.thread, 2, !dbg !13
  %address.local514 = and i32 %address.local513, 4, !dbg !13
  %address.local515 = shl nuw nsw i32 %address.thread, 1, !dbg !13
  %address.local516 = and i32 %address.local515, 384, !dbg !13
  %address.local517 = or disjoint i32 %address.local514, %address.local516, !dbg !13
  %address.local518 = shl nuw nsw i32 %address.thread, 3, !dbg !13
  %address.local519 = and i32 %address.local518, 120, !dbg !13
  %address.local520 = or disjoint i32 %address.local517, %address.local519, !dbg !13
  %address.local521 = shl nuw nsw i32 %address.thread, 5, !dbg !13
  %address.local522 = and i32 %address.local521, 1024, !dbg !13
  %address.local523 = or disjoint i32 %address.local520, %address.local522, !dbg !13
  %address.local524 = zext nneg i32 %address.local523 to i64, !dbg !13
  %address.local525 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local524, !dbg !13
  %111 = load <4 x half>, ptr addrspace(3) %address.local525, align 8, !dbg !13
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local525, i64 4096, !dbg !13
  %112 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !13
  %lds.const.gep1443 = getelementptr i8, ptr addrspace(3) %address.local525, i64 1024, !dbg !13
  %113 = load <4 x half>, ptr addrspace(3) %lds.const.gep1443, align 8, !dbg !13
  %114 = fmul <4 x half> %111, splat (half 0xH3015), !dbg !13
  %lds.const.gep1444 = getelementptr i8, ptr addrspace(3) %address.local525, i64 5120, !dbg !13
  %115 = load <4 x half>, ptr addrspace(3) %lds.const.gep1444, align 8, !dbg !13
  %116 = fmul <4 x half> %112, splat (half 0xH3015), !dbg !13
  %117 = fmul <4 x half> %113, splat (half 0xH3015), !dbg !13
  %118 = fmul <4 x half> %115, splat (half 0xH3015), !dbg !13
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %address.local533 = lshr i64 %address.local, 2, !dbg !7
  %address.local536 = shl nuw nsw i64 %address.local, 1, !dbg !7
  %address.local537 = and i64 %address.local536, 24, !dbg !7
  %address.local539 = shl i64 %address.local, 62, !dbg !7
  %address.local540 = ashr exact i64 %address.local539, 57, !dbg !7
  %address.local541 = and i64 %address.local540, -64, !dbg !7
  %address.local543 = lshr i64 %address.local, 2, !dbg !13
  %address.local544 = and i64 %address.local543, 12, !dbg !13
  %119 = and i64 %address.local, 32, !dbg !13
  %.not1529 = icmp eq i64 %119, 0, !dbg !13
  %address.local548 = select i1 %.not1529, i64 0, i64 1016, !dbg !13
  %.idx1530 = mul nuw nsw i64 %address.local, 320, !dbg !7
  %120 = getelementptr i8, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %.idx1530, !dbg !7
  %.idx1531 = mul nsw i64 %address.local533, -1280, !dbg !7
  %121 = getelementptr i8, ptr addrspace(3) %120, i64 %.idx1531, !dbg !7
  %122 = getelementptr [2 x i8], ptr addrspace(3) %121, i64 %address.local537, !dbg !7
  %123 = getelementptr [2 x i8], ptr addrspace(3) %122, i64 %address.local541, !dbg !7
  %124 = getelementptr [2 x i8], ptr addrspace(3) %123, i64 %address.local544, !dbg !7
  %address.local550 = getelementptr [2 x i8], ptr addrspace(3) %124, i64 %address.local548, !dbg !7
  %125 = load <4 x half>, ptr addrspace(3) %address.local550, align 8, !dbg !7
  %126 = extractvalue [2 x <32 x float>] %57, 0, !dbg !36
  %127 = shufflevector <32 x float> %126, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !36
  %128 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %114, <4 x half> %125, <4 x float> %127) #9, !dbg !7
  %lds.const.gep1445 = getelementptr i8, ptr addrspace(3) %address.local550, i64 4096, !dbg !7
  %129 = load <4 x half>, ptr addrspace(3) %lds.const.gep1445, align 8, !dbg !7
  %130 = shufflevector <32 x float> %126, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !36
  %131 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %116, <4 x half> %129, <4 x float> %128) #9, !dbg !7
  %132 = shufflevector <32 x float> %126, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !36
  %133 = shufflevector <32 x float> %126, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !36
  %134 = shufflevector <32 x float> %126, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !36
  %135 = shufflevector <32 x float> %126, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !36
  %136 = shufflevector <32 x float> %126, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !36
  %137 = shufflevector <32 x float> %126, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !36
  %138 = extractvalue [2 x <32 x float>] %57, 1, !dbg !36
  %139 = shufflevector <32 x float> %138, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !36
  %140 = shufflevector <32 x float> %138, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !36
  %141 = shufflevector <32 x float> %138, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !36
  %142 = shufflevector <32 x float> %138, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !36
  %143 = shufflevector <32 x float> %138, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !36
  %144 = shufflevector <32 x float> %138, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !36
  %145 = shufflevector <32 x float> %138, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !36
  %146 = shufflevector <32 x float> %138, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !36
  %address.local561 = shl nuw nsw i32 %address.thread, 1, !dbg !7
  %address.local562 = and i32 %address.local561, 24, !dbg !7
  %address.local563 = shl nuw nsw i32 %address.thread, 8, !dbg !7
  %address.local564 = and i32 %address.local563, 768, !dbg !7
  %address.local565 = or disjoint i32 %address.local562, %address.local564, !dbg !7
  %address.local566 = shl nuw nsw i32 %address.thread, 5, !dbg !7
  %address.local567 = and i32 %address.local566, 96, !dbg !7
  %address.local568 = or disjoint i32 %address.local565, %address.local567, !dbg !7
  %address.local569 = zext nneg i32 %address.local568 to i64, !dbg !7
  %address.base = call i64 asm sideeffect "", "=v,0"(i64 %33) #9, !dbg !7, !frisk.register_lifetimes !1
  %address.local570 = lshr i32 %address.thread, 1, !dbg !7
  %address.local571 = and i32 %address.local570, 1, !dbg !7
  %address.local572 = zext nneg i32 %address.local571 to i64, !dbg !7
  %address.local573 = add nsw i64 %address.base, %address.local572, !dbg !7
  %address.local574 = shl i64 %address.local573, 6, !dbg !7
  %address.local575 = sub i64 %address.local569, %address.local574, !dbg !7
  %address.local576 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local575, !dbg !7
  %address.local577 = getelementptr [2 x i8], ptr addrspace(3) %address.local576, i64 %address.local544, !dbg !7
  %address.local578 = getelementptr [2 x i8], ptr addrspace(3) %address.local577, i64 %address.local548, !dbg !7
  %address.local579 = getelementptr i8, ptr addrspace(3) %address.local578, i64 64, !dbg !7
  %147 = load <4 x half>, ptr addrspace(3) %address.local579, align 8, !dbg !7
  %lds.const.gep1446 = getelementptr i8, ptr addrspace(3) %address.local578, i64 4160, !dbg !7
  %148 = load <4 x half>, ptr addrspace(3) %lds.const.gep1446, align 8, !dbg !7
  %149 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %114, <4 x half> %147, <4 x float> %130) #9, !dbg !7
  %lds.const.gep1447 = getelementptr i8, ptr addrspace(3) %address.local550, i64 128, !dbg !7
  %150 = load <4 x half>, ptr addrspace(3) %lds.const.gep1447, align 8, !dbg !7
  %151 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %116, <4 x half> %148, <4 x float> %149) #9, !dbg !7
  %lds.const.gep1448 = getelementptr i8, ptr addrspace(3) %address.local550, i64 4224, !dbg !7
  %152 = load <4 x half>, ptr addrspace(3) %lds.const.gep1448, align 8, !dbg !7
  %153 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %114, <4 x half> %150, <4 x float> %132) #9, !dbg !7
  %154 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %116, <4 x half> %152, <4 x float> %153) #9, !dbg !7
  %lds.const.gep1449 = getelementptr i8, ptr addrspace(3) %address.local578, i64 192, !dbg !7
  %155 = load <4 x half>, ptr addrspace(3) %lds.const.gep1449, align 8, !dbg !7
  %lds.const.gep1450 = getelementptr i8, ptr addrspace(3) %address.local578, i64 4288, !dbg !7
  %156 = load <4 x half>, ptr addrspace(3) %lds.const.gep1450, align 8, !dbg !7
  %157 = shufflevector <4 x float> %154, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !36
  %158 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %114, <4 x half> %155, <4 x float> %133) #9, !dbg !7
  %159 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %116, <4 x half> %156, <4 x float> %158) #9, !dbg !7
  %lds.const.gep1451 = getelementptr i8, ptr addrspace(3) %address.local550, i64 1024, !dbg !7
  %160 = load <4 x half>, ptr addrspace(3) %lds.const.gep1451, align 8, !dbg !7
  %161 = shufflevector <4 x float> %159, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !36
  %162 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %114, <4 x half> %160, <4 x float> %134) #9, !dbg !7
  %lds.const.gep1452 = getelementptr i8, ptr addrspace(3) %address.local550, i64 5120, !dbg !7
  %163 = load <4 x half>, ptr addrspace(3) %lds.const.gep1452, align 8, !dbg !7
  %164 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %116, <4 x half> %163, <4 x float> %162) #9, !dbg !7
  %lds.const.gep1453 = getelementptr i8, ptr addrspace(3) %address.local578, i64 1088, !dbg !7
  %165 = load <4 x half>, ptr addrspace(3) %lds.const.gep1453, align 8, !dbg !7
  %lds.const.gep1454 = getelementptr i8, ptr addrspace(3) %address.local578, i64 5184, !dbg !7
  %166 = load <4 x half>, ptr addrspace(3) %lds.const.gep1454, align 8, !dbg !7
  %167 = shufflevector <4 x float> %164, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !36
  %168 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %114, <4 x half> %165, <4 x float> %135) #9, !dbg !7
  %169 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %116, <4 x half> %166, <4 x float> %168) #9, !dbg !7
  %lds.const.gep1455 = getelementptr i8, ptr addrspace(3) %address.local550, i64 1152, !dbg !7
  %170 = load <4 x half>, ptr addrspace(3) %lds.const.gep1455, align 8, !dbg !7
  %171 = shufflevector <4 x float> %169, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !36
  %172 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %114, <4 x half> %170, <4 x float> %136) #9, !dbg !7
  %lds.const.gep1456 = getelementptr i8, ptr addrspace(3) %address.local550, i64 5248, !dbg !7
  %173 = load <4 x half>, ptr addrspace(3) %lds.const.gep1456, align 8, !dbg !7
  %174 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %116, <4 x half> %173, <4 x float> %172) #9, !dbg !7
  %lds.const.gep1457 = getelementptr i8, ptr addrspace(3) %address.local578, i64 1216, !dbg !7
  %175 = load <4 x half>, ptr addrspace(3) %lds.const.gep1457, align 8, !dbg !7
  %lds.const.gep1458 = getelementptr i8, ptr addrspace(3) %address.local578, i64 5312, !dbg !7
  %176 = load <4 x half>, ptr addrspace(3) %lds.const.gep1458, align 8, !dbg !7
  %177 = shufflevector <4 x float> %174, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !36
  %178 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %114, <4 x half> %175, <4 x float> %137) #9, !dbg !7
  %179 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %116, <4 x half> %176, <4 x float> %178) #9, !dbg !7
  %180 = load <4 x half>, ptr addrspace(3) %address.local550, align 8, !dbg !7
  %181 = shufflevector <4 x float> %179, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !36
  %182 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %117, <4 x half> %180, <4 x float> %139) #9, !dbg !7
  %lds.const.gep1459 = getelementptr i8, ptr addrspace(3) %address.local550, i64 4096, !dbg !7
  %183 = load <4 x half>, ptr addrspace(3) %lds.const.gep1459, align 8, !dbg !7
  %184 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %118, <4 x half> %183, <4 x float> %182) #9, !dbg !7
  %185 = load <4 x half>, ptr addrspace(3) %address.local579, align 8, !dbg !7
  %lds.const.gep1460 = getelementptr i8, ptr addrspace(3) %address.local578, i64 4160, !dbg !7
  %186 = load <4 x half>, ptr addrspace(3) %lds.const.gep1460, align 8, !dbg !7
  %187 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %117, <4 x half> %185, <4 x float> %140) #9, !dbg !7
  %188 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %118, <4 x half> %186, <4 x float> %187) #9, !dbg !7
  %lds.const.gep1461 = getelementptr i8, ptr addrspace(3) %address.local550, i64 128, !dbg !7
  %189 = load <4 x half>, ptr addrspace(3) %lds.const.gep1461, align 8, !dbg !7
  %190 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %117, <4 x half> %189, <4 x float> %141) #9, !dbg !7
  %lds.const.gep1462 = getelementptr i8, ptr addrspace(3) %address.local550, i64 4224, !dbg !7
  %191 = load <4 x half>, ptr addrspace(3) %lds.const.gep1462, align 8, !dbg !7
  %192 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %118, <4 x half> %191, <4 x float> %190) #9, !dbg !7
  %lds.const.gep1463 = getelementptr i8, ptr addrspace(3) %address.local578, i64 192, !dbg !7
  %193 = load <4 x half>, ptr addrspace(3) %lds.const.gep1463, align 8, !dbg !7
  %lds.const.gep1464 = getelementptr i8, ptr addrspace(3) %address.local578, i64 4288, !dbg !7
  %194 = load <4 x half>, ptr addrspace(3) %lds.const.gep1464, align 8, !dbg !7
  %195 = shufflevector <4 x float> %192, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !36
  %196 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %117, <4 x half> %193, <4 x float> %142) #9, !dbg !7
  %197 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %118, <4 x half> %194, <4 x float> %196) #9, !dbg !7
  %lds.const.gep1465 = getelementptr i8, ptr addrspace(3) %address.local550, i64 1024, !dbg !7
  %198 = load <4 x half>, ptr addrspace(3) %lds.const.gep1465, align 8, !dbg !7
  %199 = shufflevector <4 x float> %197, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !36
  %200 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %117, <4 x half> %198, <4 x float> %143) #9, !dbg !7
  %lds.const.gep1466 = getelementptr i8, ptr addrspace(3) %address.local550, i64 5120, !dbg !7
  %201 = load <4 x half>, ptr addrspace(3) %lds.const.gep1466, align 8, !dbg !7
  %202 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %118, <4 x half> %201, <4 x float> %200) #9, !dbg !7
  %lds.const.gep1467 = getelementptr i8, ptr addrspace(3) %address.local578, i64 1088, !dbg !7
  %203 = load <4 x half>, ptr addrspace(3) %lds.const.gep1467, align 8, !dbg !7
  %lds.const.gep1468 = getelementptr i8, ptr addrspace(3) %address.local578, i64 5184, !dbg !7
  %204 = load <4 x half>, ptr addrspace(3) %lds.const.gep1468, align 8, !dbg !7
  %205 = shufflevector <4 x float> %202, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !36
  %206 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %117, <4 x half> %203, <4 x float> %144) #9, !dbg !7
  %207 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %118, <4 x half> %204, <4 x float> %206) #9, !dbg !7
  %lds.const.gep1469 = getelementptr i8, ptr addrspace(3) %address.local550, i64 1152, !dbg !7
  %208 = load <4 x half>, ptr addrspace(3) %lds.const.gep1469, align 8, !dbg !7
  %209 = shufflevector <4 x float> %207, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !36
  %210 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %117, <4 x half> %208, <4 x float> %145) #9, !dbg !7
  %lds.const.gep1470 = getelementptr i8, ptr addrspace(3) %address.local550, i64 5248, !dbg !7
  %211 = load <4 x half>, ptr addrspace(3) %lds.const.gep1470, align 8, !dbg !7
  %212 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %118, <4 x half> %211, <4 x float> %210) #9, !dbg !7
  %lds.const.gep1471 = getelementptr i8, ptr addrspace(3) %address.local578, i64 1216, !dbg !7
  %213 = load <4 x half>, ptr addrspace(3) %lds.const.gep1471, align 8, !dbg !7
  %lds.const.gep1472 = getelementptr i8, ptr addrspace(3) %address.local578, i64 5312, !dbg !7
  %214 = load <4 x half>, ptr addrspace(3) %lds.const.gep1472, align 8, !dbg !7
  %215 = shufflevector <4 x float> %212, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !36
  %216 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %117, <4 x half> %213, <4 x float> %146) #9, !dbg !7
  %217 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %118, <4 x half> %214, <4 x float> %216) #9, !dbg !7
  %218 = shufflevector <4 x float> %217, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !36
  %219 = shufflevector <4 x float> %131, <4 x float> %151, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %220 = shufflevector <32 x float> %219, <32 x float> %157, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %221 = shufflevector <32 x float> %220, <32 x float> %161, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %222 = shufflevector <32 x float> %221, <32 x float> %167, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %223 = shufflevector <32 x float> %222, <32 x float> %171, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %224 = shufflevector <32 x float> %223, <32 x float> %177, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %225 = shufflevector <32 x float> %224, <32 x float> %181, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 32, i32 33, i32 34, i32 35>, !dbg !22
  %226 = insertvalue [2 x <32 x float>] poison, <32 x float> %225, 0, !dbg !22
  %227 = shufflevector <4 x float> %184, <4 x float> %188, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %228 = shufflevector <32 x float> %227, <32 x float> %195, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %229 = shufflevector <32 x float> %228, <32 x float> %199, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %230 = shufflevector <32 x float> %229, <32 x float> %205, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %231 = shufflevector <32 x float> %230, <32 x float> %209, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %232 = shufflevector <32 x float> %231, <32 x float> %215, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %233 = shufflevector <32 x float> %232, <32 x float> %218, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 32, i32 33, i32 34, i32 35>, !dbg !22
  %234 = insertvalue [2 x <32 x float>] %226, <32 x float> %233, 1, !dbg !22
  fence syncscope("workgroup") release, !dbg !22
  call void @llvm.amdgcn.s.barrier(), !dbg !22
  fence syncscope("workgroup") acquire, !dbg !22
  %235 = add nuw nsw i64 %56, 1, !dbg !22
  br label %55, !dbg !22

236:                                              ; preds = %55
  %237 = add nuw nsw i64 %52, %15, !dbg !16
  %238 = shl i64 %237, 19, !dbg !16
  %239 = and i64 %238, 16252928, !dbg !16
  %240 = shl nsw i64 %54, 7, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %address.thread888 = call i32 asm sideeffect "", "=v,0"(i32 %23) #9, !dbg !16, !range !39, !frisk.register_lifetimes !1
  %address.local889 = zext nneg i32 %address.thread888 to i64, !dbg !13
  %address.local890 = shl nuw nsw i64 %address.local889, 3, !dbg !12
  %address.local891 = and i64 %address.local890, 1536, !dbg !12
  %address.local892 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local891, !dbg !12
  %address.local893 = ptrtoint ptr addrspace(3) %address.local892 to i64, !dbg !12
  %address.local894 = trunc i64 %address.local893 to i32, !dbg !12
  %241 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local894), !dbg !16
  %address.local895 = shl nuw nsw i64 %address.local889, 5, !dbg !16
  %address.local896 = and i64 %address.local895, 8064, !dbg !16
  %address.local897 = shl nuw nsw i64 %address.local889, 3, !dbg !16
  %address.local898 = and i64 %address.local897, 24, !dbg !16
  %address.local899 = or disjoint i64 %address.local896, %address.local898, !dbg !16
  %242 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !16
  %.idx1507 = shl i64 %54, 8, !dbg !16
  %243 = or disjoint i64 %239, %address.local899, !dbg !16
  %244 = shl nuw nsw i64 %243, 1, !dbg !16
  %245 = add i64 %244, %.idx1507, !dbg !16
  %dma.buffer.offset1503 = trunc i64 %245 to i32, !dbg !16
  %246 = insertelement <2 x i64> poison, i64 %242, i64 0, !dbg !16
  %247 = bitcast <2 x i64> %246 to <4 x i32>, !dbg !16
  %dma.buffer.rsrc1504 = shufflevector <4 x i32> %247, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !16
  %dma.buffer.lds1505 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %241), !dbg !16
  %248 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1505, i32 %dma.buffer.offset1503, <4 x i32> %dma.buffer.rsrc1504) #10, !dbg !16
  %address.local901 = or disjoint i64 %address.local889, 256, !dbg !12
  %address.local902 = shl nuw nsw i64 %address.local901, 3, !dbg !12
  %address.local903 = and i64 %address.local902, 3584, !dbg !12
  %address.local904 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local903, !dbg !12
  %address.local905 = ptrtoint ptr addrspace(3) %address.local904 to i64, !dbg !12
  %address.local906 = trunc i64 %address.local905 to i32, !dbg !12
  %249 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local906), !dbg !16
  %address.local907 = shl nuw nsw i64 %address.local901, 5, !dbg !16
  %address.local908 = and i64 %address.local907, 16256, !dbg !16
  %address.local909 = shl nuw nsw i64 %address.local889, 3, !dbg !16
  %address.local910 = and i64 %address.local909, 24, !dbg !16
  %address.local911 = or disjoint i64 %address.local908, %address.local910, !dbg !16
  %250 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !16
  %.idx1509 = shl i64 %54, 8, !dbg !16
  %251 = or disjoint i64 %239, %address.local911, !dbg !16
  %252 = shl nuw nsw i64 %251, 1, !dbg !16
  %253 = add i64 %252, %.idx1509, !dbg !16
  %dma.buffer.offset1500 = trunc i64 %253 to i32, !dbg !16
  %254 = insertelement <2 x i64> poison, i64 %250, i64 0, !dbg !16
  %255 = bitcast <2 x i64> %254 to <4 x i32>, !dbg !16
  %dma.buffer.rsrc1501 = shufflevector <4 x i32> %255, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !16
  %dma.buffer.lds1502 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %249), !dbg !16
  %256 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1502, i32 %dma.buffer.offset1500, <4 x i32> %dma.buffer.rsrc1501) #10, !dbg !16
  %257 = or disjoint i64 %51, %39, !dbg !17
  %258 = lshr i64 %40, 32, !dbg !23
  %compare.high = trunc nuw i64 %258 to i32, !dbg !23
  %259 = lshr i64 %51, 32, !dbg !23
  %compare.high1187 = trunc nuw i64 %259 to i32, !dbg !23
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high1187, !dbg !23
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high1187, !dbg !23
  %compare.low.base = trunc i64 %40 to i32, !dbg !23
  %compare.low.base1188 = trunc i64 %257 to i32, !dbg !23
  %compare.low = icmp ugt i32 %compare.low.base, %compare.low.base1188, !dbg !23
  %compare.result = select i1 %compare.high.equal, i1 %compare.low, i1 %compare.high.order, !dbg !23
  %260 = select i1 %compare.result, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1188, i32 1) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local = or i32 %compare.local.base, 1, !dbg !23
  %compare.low1189 = icmp ult i32 %compare.local, %compare.low.base, !dbg !23
  %compare.result1190 = select i1 %compare.high.equal, i1 %compare.low1189, i1 %compare.high.order, !dbg !23
  %261 = select i1 %compare.result1190, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1191 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1188, i32 2) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1192 = or i32 %compare.local.base1191, 2, !dbg !23
  %compare.low1193 = icmp ult i32 %compare.local1192, %compare.low.base, !dbg !23
  %compare.result1194 = select i1 %compare.high.equal, i1 %compare.low1193, i1 %compare.high.order, !dbg !23
  %262 = select i1 %compare.result1194, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1195 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1188, i32 3) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1196 = or i32 %compare.local.base1195, 3, !dbg !23
  %compare.low1197 = icmp ult i32 %compare.local1196, %compare.low.base, !dbg !23
  %compare.result1198 = select i1 %compare.high.equal, i1 %compare.low1197, i1 %compare.high.order, !dbg !23
  %263 = select i1 %compare.result1198, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %264 = extractvalue [2 x <32 x float>] %57, 0, !dbg !25
  %265 = shufflevector <32 x float> %264, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %266 = insertelement <4 x float> poison, float %260, i64 0, !dbg !25
  %267 = insertelement <4 x float> %266, float %261, i64 1, !dbg !25
  %268 = insertelement <4 x float> %267, float %262, i64 2, !dbg !25
  %269 = insertelement <4 x float> %268, float %263, i64 3, !dbg !25
  %270 = fadd <4 x float> %265, %269, !dbg !25
  %271 = extractelement <4 x float> %270, i64 0, !dbg !25
  %272 = extractelement <4 x float> %270, i64 1, !dbg !25
  %273 = extractelement <4 x float> %270, i64 2, !dbg !25
  %274 = extractelement <4 x float> %270, i64 3, !dbg !25
  %compare.local.base1199 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1188, i32 4) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1200 = or i32 %compare.local.base1199, 4, !dbg !23
  %compare.low1201 = icmp ult i32 %compare.local1200, %compare.low.base, !dbg !23
  %compare.result1202 = select i1 %compare.high.equal, i1 %compare.low1201, i1 %compare.high.order, !dbg !23
  %275 = select i1 %compare.result1202, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1203 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1188, i32 5) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1204 = or i32 %compare.local.base1203, 5, !dbg !23
  %compare.low1205 = icmp ult i32 %compare.local1204, %compare.low.base, !dbg !23
  %compare.result1206 = select i1 %compare.high.equal, i1 %compare.low1205, i1 %compare.high.order, !dbg !23
  %276 = select i1 %compare.result1206, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1207 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1188, i32 6) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1208 = or i32 %compare.local.base1207, 6, !dbg !23
  %compare.low1209 = icmp ult i32 %compare.local1208, %compare.low.base, !dbg !23
  %compare.result1210 = select i1 %compare.high.equal, i1 %compare.low1209, i1 %compare.high.order, !dbg !23
  %277 = select i1 %compare.result1210, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1211 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1188, i32 7) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1212 = or i32 %compare.local.base1211, 7, !dbg !23
  %compare.low1213 = icmp ult i32 %compare.local1212, %compare.low.base, !dbg !23
  %compare.result1214 = select i1 %compare.high.equal, i1 %compare.low1213, i1 %compare.high.order, !dbg !23
  %278 = select i1 %compare.result1214, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %279 = shufflevector <32 x float> %264, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %280 = insertelement <4 x float> poison, float %275, i64 0, !dbg !25
  %281 = insertelement <4 x float> %280, float %276, i64 1, !dbg !25
  %282 = insertelement <4 x float> %281, float %277, i64 2, !dbg !25
  %283 = insertelement <4 x float> %282, float %278, i64 3, !dbg !25
  %284 = fadd <4 x float> %279, %283, !dbg !25
  %285 = extractelement <4 x float> %284, i64 0, !dbg !25
  %286 = extractelement <4 x float> %284, i64 1, !dbg !25
  %287 = extractelement <4 x float> %284, i64 2, !dbg !25
  %288 = extractelement <4 x float> %284, i64 3, !dbg !25
  %compare.local.base1215 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1188, i32 8) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1216 = or i32 %compare.local.base1215, 8, !dbg !23
  %compare.low1217 = icmp ult i32 %compare.local1216, %compare.low.base, !dbg !23
  %compare.result1218 = select i1 %compare.high.equal, i1 %compare.low1217, i1 %compare.high.order, !dbg !23
  %289 = select i1 %compare.result1218, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1219 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1188, i32 9) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1220 = or i32 %compare.local.base1219, 9, !dbg !23
  %compare.low1221 = icmp ult i32 %compare.local1220, %compare.low.base, !dbg !23
  %compare.result1222 = select i1 %compare.high.equal, i1 %compare.low1221, i1 %compare.high.order, !dbg !23
  %290 = select i1 %compare.result1222, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1223 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1188, i32 10) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1224 = or i32 %compare.local.base1223, 10, !dbg !23
  %compare.low1225 = icmp ult i32 %compare.local1224, %compare.low.base, !dbg !23
  %compare.result1226 = select i1 %compare.high.equal, i1 %compare.low1225, i1 %compare.high.order, !dbg !23
  %291 = select i1 %compare.result1226, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1227 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1188, i32 11) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1228 = or i32 %compare.local.base1227, 11, !dbg !23
  %compare.low1229 = icmp ult i32 %compare.local1228, %compare.low.base, !dbg !23
  %compare.result1230 = select i1 %compare.high.equal, i1 %compare.low1229, i1 %compare.high.order, !dbg !23
  %292 = select i1 %compare.result1230, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %293 = shufflevector <32 x float> %264, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !25
  %294 = insertelement <4 x float> poison, float %289, i64 0, !dbg !25
  %295 = insertelement <4 x float> %294, float %290, i64 1, !dbg !25
  %296 = insertelement <4 x float> %295, float %291, i64 2, !dbg !25
  %297 = insertelement <4 x float> %296, float %292, i64 3, !dbg !25
  %298 = fadd <4 x float> %293, %297, !dbg !25
  %299 = extractelement <4 x float> %298, i64 0, !dbg !25
  %300 = extractelement <4 x float> %298, i64 1, !dbg !25
  %301 = extractelement <4 x float> %298, i64 2, !dbg !25
  %302 = extractelement <4 x float> %298, i64 3, !dbg !25
  %compare.local.base1231 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1188, i32 12) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1232 = or i32 %compare.local.base1231, 12, !dbg !23
  %compare.low1233 = icmp ult i32 %compare.local1232, %compare.low.base, !dbg !23
  %compare.result1234 = select i1 %compare.high.equal, i1 %compare.low1233, i1 %compare.high.order, !dbg !23
  %303 = select i1 %compare.result1234, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1235 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1188, i32 13) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1236 = or i32 %compare.local.base1235, 13, !dbg !23
  %compare.low1237 = icmp ult i32 %compare.local1236, %compare.low.base, !dbg !23
  %compare.result1238 = select i1 %compare.high.equal, i1 %compare.low1237, i1 %compare.high.order, !dbg !23
  %304 = select i1 %compare.result1238, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1239 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1188, i32 14) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1240 = or i32 %compare.local.base1239, 14, !dbg !23
  %compare.low1241 = icmp ult i32 %compare.local1240, %compare.low.base, !dbg !23
  %compare.result1242 = select i1 %compare.high.equal, i1 %compare.low1241, i1 %compare.high.order, !dbg !23
  %305 = select i1 %compare.result1242, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1243 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1188, i32 15) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1244 = or i32 %compare.local.base1243, 15, !dbg !23
  %compare.low1245 = icmp ult i32 %compare.local1244, %compare.low.base, !dbg !23
  %compare.result1246 = select i1 %compare.high.equal, i1 %compare.low1245, i1 %compare.high.order, !dbg !23
  %306 = select i1 %compare.result1246, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %307 = shufflevector <32 x float> %264, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !25
  %308 = insertelement <4 x float> poison, float %303, i64 0, !dbg !25
  %309 = insertelement <4 x float> %308, float %304, i64 1, !dbg !25
  %310 = insertelement <4 x float> %309, float %305, i64 2, !dbg !25
  %311 = insertelement <4 x float> %310, float %306, i64 3, !dbg !25
  %312 = fadd <4 x float> %307, %311, !dbg !25
  %313 = extractelement <4 x float> %312, i64 0, !dbg !25
  %314 = extractelement <4 x float> %312, i64 1, !dbg !25
  %315 = extractelement <4 x float> %312, i64 2, !dbg !25
  %316 = extractelement <4 x float> %312, i64 3, !dbg !25
  %compare.local.base1247 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1188, i32 16) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1248 = or i32 %compare.local.base1247, 64, !dbg !23
  %compare.low1249 = icmp ult i32 %compare.local1248, %compare.low.base, !dbg !23
  %compare.result1250 = select i1 %compare.high.equal, i1 %compare.low1249, i1 %compare.high.order, !dbg !23
  %317 = select i1 %compare.result1250, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1251 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1188, i32 17) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1252 = or i32 %compare.local.base1251, 65, !dbg !23
  %compare.low1253 = icmp ult i32 %compare.local1252, %compare.low.base, !dbg !23
  %compare.result1254 = select i1 %compare.high.equal, i1 %compare.low1253, i1 %compare.high.order, !dbg !23
  %318 = select i1 %compare.result1254, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1255 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1188, i32 18) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1256 = or i32 %compare.local.base1255, 66, !dbg !23
  %compare.low1257 = icmp ult i32 %compare.local1256, %compare.low.base, !dbg !23
  %compare.result1258 = select i1 %compare.high.equal, i1 %compare.low1257, i1 %compare.high.order, !dbg !23
  %319 = select i1 %compare.result1258, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1259 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1188, i32 19) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1260 = or i32 %compare.local.base1259, 67, !dbg !23
  %compare.low1261 = icmp ult i32 %compare.local1260, %compare.low.base, !dbg !23
  %compare.result1262 = select i1 %compare.high.equal, i1 %compare.low1261, i1 %compare.high.order, !dbg !23
  %320 = select i1 %compare.result1262, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %321 = shufflevector <32 x float> %264, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !25
  %322 = insertelement <4 x float> poison, float %317, i64 0, !dbg !25
  %323 = insertelement <4 x float> %322, float %318, i64 1, !dbg !25
  %324 = insertelement <4 x float> %323, float %319, i64 2, !dbg !25
  %325 = insertelement <4 x float> %324, float %320, i64 3, !dbg !25
  %326 = fadd <4 x float> %321, %325, !dbg !25
  %327 = extractelement <4 x float> %326, i64 0, !dbg !25
  %328 = extractelement <4 x float> %326, i64 1, !dbg !25
  %329 = extractelement <4 x float> %326, i64 2, !dbg !25
  %330 = extractelement <4 x float> %326, i64 3, !dbg !25
  %compare.local.base1263 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1188, i32 20) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1264 = or i32 %compare.local.base1263, 68, !dbg !23
  %compare.low1265 = icmp ult i32 %compare.local1264, %compare.low.base, !dbg !23
  %compare.result1266 = select i1 %compare.high.equal, i1 %compare.low1265, i1 %compare.high.order, !dbg !23
  %331 = select i1 %compare.result1266, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1267 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1188, i32 21) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1268 = or i32 %compare.local.base1267, 69, !dbg !23
  %compare.low1269 = icmp ult i32 %compare.local1268, %compare.low.base, !dbg !23
  %compare.result1270 = select i1 %compare.high.equal, i1 %compare.low1269, i1 %compare.high.order, !dbg !23
  %332 = select i1 %compare.result1270, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1271 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1188, i32 22) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1272 = or i32 %compare.local.base1271, 70, !dbg !23
  %compare.low1273 = icmp ult i32 %compare.local1272, %compare.low.base, !dbg !23
  %compare.result1274 = select i1 %compare.high.equal, i1 %compare.low1273, i1 %compare.high.order, !dbg !23
  %333 = select i1 %compare.result1274, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1275 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1188, i32 23) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1276 = or i32 %compare.local.base1275, 71, !dbg !23
  %compare.low1277 = icmp ult i32 %compare.local1276, %compare.low.base, !dbg !23
  %compare.result1278 = select i1 %compare.high.equal, i1 %compare.low1277, i1 %compare.high.order, !dbg !23
  %334 = select i1 %compare.result1278, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %335 = shufflevector <32 x float> %264, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !25
  %336 = insertelement <4 x float> poison, float %331, i64 0, !dbg !25
  %337 = insertelement <4 x float> %336, float %332, i64 1, !dbg !25
  %338 = insertelement <4 x float> %337, float %333, i64 2, !dbg !25
  %339 = insertelement <4 x float> %338, float %334, i64 3, !dbg !25
  %340 = fadd <4 x float> %335, %339, !dbg !25
  %341 = extractelement <4 x float> %340, i64 0, !dbg !25
  %342 = extractelement <4 x float> %340, i64 1, !dbg !25
  %343 = extractelement <4 x float> %340, i64 2, !dbg !25
  %344 = extractelement <4 x float> %340, i64 3, !dbg !25
  %compare.local.base1279 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1188, i32 24) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1280 = or i32 %compare.local.base1279, 72, !dbg !23
  %compare.low1281 = icmp ult i32 %compare.local1280, %compare.low.base, !dbg !23
  %compare.result1282 = select i1 %compare.high.equal, i1 %compare.low1281, i1 %compare.high.order, !dbg !23
  %345 = select i1 %compare.result1282, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1283 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1188, i32 25) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1284 = or i32 %compare.local.base1283, 73, !dbg !23
  %compare.low1285 = icmp ult i32 %compare.local1284, %compare.low.base, !dbg !23
  %compare.result1286 = select i1 %compare.high.equal, i1 %compare.low1285, i1 %compare.high.order, !dbg !23
  %346 = select i1 %compare.result1286, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1287 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1188, i32 26) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1288 = or i32 %compare.local.base1287, 74, !dbg !23
  %compare.low1289 = icmp ult i32 %compare.local1288, %compare.low.base, !dbg !23
  %compare.result1290 = select i1 %compare.high.equal, i1 %compare.low1289, i1 %compare.high.order, !dbg !23
  %347 = select i1 %compare.result1290, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1291 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1188, i32 27) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1292 = or i32 %compare.local.base1291, 75, !dbg !23
  %compare.low1293 = icmp ult i32 %compare.local1292, %compare.low.base, !dbg !23
  %compare.result1294 = select i1 %compare.high.equal, i1 %compare.low1293, i1 %compare.high.order, !dbg !23
  %348 = select i1 %compare.result1294, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %349 = shufflevector <32 x float> %264, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !25
  %350 = insertelement <4 x float> poison, float %345, i64 0, !dbg !25
  %351 = insertelement <4 x float> %350, float %346, i64 1, !dbg !25
  %352 = insertelement <4 x float> %351, float %347, i64 2, !dbg !25
  %353 = insertelement <4 x float> %352, float %348, i64 3, !dbg !25
  %354 = fadd <4 x float> %349, %353, !dbg !25
  %355 = extractelement <4 x float> %354, i64 0, !dbg !25
  %356 = extractelement <4 x float> %354, i64 1, !dbg !25
  %357 = extractelement <4 x float> %354, i64 2, !dbg !25
  %358 = extractelement <4 x float> %354, i64 3, !dbg !25
  %compare.local.base1295 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1188, i32 28) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1296 = or i32 %compare.local.base1295, 76, !dbg !23
  %compare.low1297 = icmp ult i32 %compare.local1296, %compare.low.base, !dbg !23
  %compare.result1298 = select i1 %compare.high.equal, i1 %compare.low1297, i1 %compare.high.order, !dbg !23
  %359 = select i1 %compare.result1298, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1299 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1188, i32 29) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1300 = or i32 %compare.local.base1299, 77, !dbg !23
  %compare.low1301 = icmp ult i32 %compare.local1300, %compare.low.base, !dbg !23
  %compare.result1302 = select i1 %compare.high.equal, i1 %compare.low1301, i1 %compare.high.order, !dbg !23
  %360 = select i1 %compare.result1302, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1303 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1188, i32 30) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1304 = or i32 %compare.local.base1303, 78, !dbg !23
  %compare.low1305 = icmp ult i32 %compare.local1304, %compare.low.base, !dbg !23
  %compare.result1306 = select i1 %compare.high.equal, i1 %compare.low1305, i1 %compare.high.order, !dbg !23
  %361 = select i1 %compare.result1306, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1307 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1188, i32 31) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1308 = or i32 %compare.local.base1307, 79, !dbg !23
  %compare.low1309 = icmp ult i32 %compare.local1308, %compare.low.base, !dbg !23
  %compare.result1310 = select i1 %compare.high.equal, i1 %compare.low1309, i1 %compare.high.order, !dbg !23
  %362 = select i1 %compare.result1310, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %363 = shufflevector <32 x float> %264, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !25
  %364 = insertelement <4 x float> poison, float %359, i64 0, !dbg !25
  %365 = insertelement <4 x float> %364, float %360, i64 1, !dbg !25
  %366 = insertelement <4 x float> %365, float %361, i64 2, !dbg !25
  %367 = insertelement <4 x float> %366, float %362, i64 3, !dbg !25
  %368 = fadd <4 x float> %363, %367, !dbg !25
  %369 = extractelement <4 x float> %368, i64 0, !dbg !25
  %370 = extractelement <4 x float> %368, i64 1, !dbg !25
  %371 = extractelement <4 x float> %368, i64 2, !dbg !25
  %372 = extractelement <4 x float> %368, i64 3, !dbg !25
  %373 = lshr i64 %41, 32, !dbg !23
  %compare.high1311 = trunc nuw i64 %373 to i32, !dbg !23
  %374 = lshr i64 %51, 32, !dbg !23
  %compare.high1312 = trunc nuw i64 %374 to i32, !dbg !23
  %compare.high.equal1313 = icmp eq i32 %compare.high1311, %compare.high1312, !dbg !23
  %compare.high.order1314 = icmp ugt i32 %compare.high1311, %compare.high1312, !dbg !23
  %compare.low.base1315 = trunc i64 %41 to i32, !dbg !23
  %compare.low.base1316 = trunc i64 %257 to i32, !dbg !23
  %compare.low1317 = icmp ugt i32 %compare.low.base1315, %compare.low.base1316, !dbg !23
  %compare.result1318 = select i1 %compare.high.equal1313, i1 %compare.low1317, i1 %compare.high.order1314, !dbg !23
  %375 = select i1 %compare.result1318, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1319 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1316, i32 32) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1320 = or i32 %compare.local.base1319, 1, !dbg !23
  %compare.low1321 = icmp ult i32 %compare.local1320, %compare.low.base1315, !dbg !23
  %compare.result1322 = select i1 %compare.high.equal1313, i1 %compare.low1321, i1 %compare.high.order1314, !dbg !23
  %376 = select i1 %compare.result1322, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1323 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1316, i32 33) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1324 = or i32 %compare.local.base1323, 2, !dbg !23
  %compare.low1325 = icmp ult i32 %compare.local1324, %compare.low.base1315, !dbg !23
  %compare.result1326 = select i1 %compare.high.equal1313, i1 %compare.low1325, i1 %compare.high.order1314, !dbg !23
  %377 = select i1 %compare.result1326, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1327 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1316, i32 34) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1328 = or i32 %compare.local.base1327, 3, !dbg !23
  %compare.low1329 = icmp ult i32 %compare.local1328, %compare.low.base1315, !dbg !23
  %compare.result1330 = select i1 %compare.high.equal1313, i1 %compare.low1329, i1 %compare.high.order1314, !dbg !23
  %378 = select i1 %compare.result1330, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %379 = extractvalue [2 x <32 x float>] %57, 1, !dbg !25
  %380 = shufflevector <32 x float> %379, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %381 = insertelement <4 x float> poison, float %375, i64 0, !dbg !25
  %382 = insertelement <4 x float> %381, float %376, i64 1, !dbg !25
  %383 = insertelement <4 x float> %382, float %377, i64 2, !dbg !25
  %384 = insertelement <4 x float> %383, float %378, i64 3, !dbg !25
  %385 = fadd <4 x float> %380, %384, !dbg !25
  %386 = extractelement <4 x float> %385, i64 0, !dbg !25
  %387 = extractelement <4 x float> %385, i64 1, !dbg !25
  %388 = extractelement <4 x float> %385, i64 2, !dbg !25
  %389 = extractelement <4 x float> %385, i64 3, !dbg !25
  %compare.local.base1331 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1316, i32 35) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1332 = or i32 %compare.local.base1331, 4, !dbg !23
  %compare.low1333 = icmp ult i32 %compare.local1332, %compare.low.base1315, !dbg !23
  %compare.result1334 = select i1 %compare.high.equal1313, i1 %compare.low1333, i1 %compare.high.order1314, !dbg !23
  %390 = select i1 %compare.result1334, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1335 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1316, i32 36) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1336 = or i32 %compare.local.base1335, 5, !dbg !23
  %compare.low1337 = icmp ult i32 %compare.local1336, %compare.low.base1315, !dbg !23
  %compare.result1338 = select i1 %compare.high.equal1313, i1 %compare.low1337, i1 %compare.high.order1314, !dbg !23
  %391 = select i1 %compare.result1338, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1339 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1316, i32 37) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1340 = or i32 %compare.local.base1339, 6, !dbg !23
  %compare.low1341 = icmp ult i32 %compare.local1340, %compare.low.base1315, !dbg !23
  %compare.result1342 = select i1 %compare.high.equal1313, i1 %compare.low1341, i1 %compare.high.order1314, !dbg !23
  %392 = select i1 %compare.result1342, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1343 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1316, i32 38) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1344 = or i32 %compare.local.base1343, 7, !dbg !23
  %compare.low1345 = icmp ult i32 %compare.local1344, %compare.low.base1315, !dbg !23
  %compare.result1346 = select i1 %compare.high.equal1313, i1 %compare.low1345, i1 %compare.high.order1314, !dbg !23
  %393 = select i1 %compare.result1346, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %394 = shufflevector <32 x float> %379, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %395 = insertelement <4 x float> poison, float %390, i64 0, !dbg !25
  %396 = insertelement <4 x float> %395, float %391, i64 1, !dbg !25
  %397 = insertelement <4 x float> %396, float %392, i64 2, !dbg !25
  %398 = insertelement <4 x float> %397, float %393, i64 3, !dbg !25
  %399 = fadd <4 x float> %394, %398, !dbg !25
  %400 = extractelement <4 x float> %399, i64 0, !dbg !25
  %401 = extractelement <4 x float> %399, i64 1, !dbg !25
  %402 = extractelement <4 x float> %399, i64 2, !dbg !25
  %403 = extractelement <4 x float> %399, i64 3, !dbg !25
  %compare.local.base1347 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1316, i32 39) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1348 = or i32 %compare.local.base1347, 8, !dbg !23
  %compare.low1349 = icmp ult i32 %compare.local1348, %compare.low.base1315, !dbg !23
  %compare.result1350 = select i1 %compare.high.equal1313, i1 %compare.low1349, i1 %compare.high.order1314, !dbg !23
  %404 = select i1 %compare.result1350, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1351 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1316, i32 40) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1352 = or i32 %compare.local.base1351, 9, !dbg !23
  %compare.low1353 = icmp ult i32 %compare.local1352, %compare.low.base1315, !dbg !23
  %compare.result1354 = select i1 %compare.high.equal1313, i1 %compare.low1353, i1 %compare.high.order1314, !dbg !23
  %405 = select i1 %compare.result1354, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1355 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1316, i32 41) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1356 = or i32 %compare.local.base1355, 10, !dbg !23
  %compare.low1357 = icmp ult i32 %compare.local1356, %compare.low.base1315, !dbg !23
  %compare.result1358 = select i1 %compare.high.equal1313, i1 %compare.low1357, i1 %compare.high.order1314, !dbg !23
  %406 = select i1 %compare.result1358, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1359 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1316, i32 42) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1360 = or i32 %compare.local.base1359, 11, !dbg !23
  %compare.low1361 = icmp ult i32 %compare.local1360, %compare.low.base1315, !dbg !23
  %compare.result1362 = select i1 %compare.high.equal1313, i1 %compare.low1361, i1 %compare.high.order1314, !dbg !23
  %407 = select i1 %compare.result1362, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %408 = shufflevector <32 x float> %379, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !25
  %409 = insertelement <4 x float> poison, float %404, i64 0, !dbg !25
  %410 = insertelement <4 x float> %409, float %405, i64 1, !dbg !25
  %411 = insertelement <4 x float> %410, float %406, i64 2, !dbg !25
  %412 = insertelement <4 x float> %411, float %407, i64 3, !dbg !25
  %413 = fadd <4 x float> %408, %412, !dbg !25
  %414 = extractelement <4 x float> %413, i64 0, !dbg !25
  %415 = extractelement <4 x float> %413, i64 1, !dbg !25
  %416 = extractelement <4 x float> %413, i64 2, !dbg !25
  %417 = extractelement <4 x float> %413, i64 3, !dbg !25
  %compare.local.base1363 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1316, i32 43) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1364 = or i32 %compare.local.base1363, 12, !dbg !23
  %compare.low1365 = icmp ult i32 %compare.local1364, %compare.low.base1315, !dbg !23
  %compare.result1366 = select i1 %compare.high.equal1313, i1 %compare.low1365, i1 %compare.high.order1314, !dbg !23
  %418 = select i1 %compare.result1366, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1367 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1316, i32 44) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1368 = or i32 %compare.local.base1367, 13, !dbg !23
  %compare.low1369 = icmp ult i32 %compare.local1368, %compare.low.base1315, !dbg !23
  %compare.result1370 = select i1 %compare.high.equal1313, i1 %compare.low1369, i1 %compare.high.order1314, !dbg !23
  %419 = select i1 %compare.result1370, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1371 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1316, i32 45) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1372 = or i32 %compare.local.base1371, 14, !dbg !23
  %compare.low1373 = icmp ult i32 %compare.local1372, %compare.low.base1315, !dbg !23
  %compare.result1374 = select i1 %compare.high.equal1313, i1 %compare.low1373, i1 %compare.high.order1314, !dbg !23
  %420 = select i1 %compare.result1374, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1375 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1316, i32 46) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1376 = or i32 %compare.local.base1375, 15, !dbg !23
  %compare.low1377 = icmp ult i32 %compare.local1376, %compare.low.base1315, !dbg !23
  %compare.result1378 = select i1 %compare.high.equal1313, i1 %compare.low1377, i1 %compare.high.order1314, !dbg !23
  %421 = select i1 %compare.result1378, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %422 = shufflevector <32 x float> %379, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !25
  %423 = insertelement <4 x float> poison, float %418, i64 0, !dbg !25
  %424 = insertelement <4 x float> %423, float %419, i64 1, !dbg !25
  %425 = insertelement <4 x float> %424, float %420, i64 2, !dbg !25
  %426 = insertelement <4 x float> %425, float %421, i64 3, !dbg !25
  %427 = fadd <4 x float> %422, %426, !dbg !25
  %428 = extractelement <4 x float> %427, i64 0, !dbg !25
  %429 = extractelement <4 x float> %427, i64 1, !dbg !25
  %430 = extractelement <4 x float> %427, i64 2, !dbg !25
  %431 = extractelement <4 x float> %427, i64 3, !dbg !25
  %compare.local.base1379 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1316, i32 47) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1380 = or i32 %compare.local.base1379, 64, !dbg !23
  %compare.low1381 = icmp ult i32 %compare.local1380, %compare.low.base1315, !dbg !23
  %compare.result1382 = select i1 %compare.high.equal1313, i1 %compare.low1381, i1 %compare.high.order1314, !dbg !23
  %432 = select i1 %compare.result1382, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1383 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1316, i32 48) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1384 = or i32 %compare.local.base1383, 65, !dbg !23
  %compare.low1385 = icmp ult i32 %compare.local1384, %compare.low.base1315, !dbg !23
  %compare.result1386 = select i1 %compare.high.equal1313, i1 %compare.low1385, i1 %compare.high.order1314, !dbg !23
  %433 = select i1 %compare.result1386, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1387 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1316, i32 49) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1388 = or i32 %compare.local.base1387, 66, !dbg !23
  %compare.low1389 = icmp ult i32 %compare.local1388, %compare.low.base1315, !dbg !23
  %compare.result1390 = select i1 %compare.high.equal1313, i1 %compare.low1389, i1 %compare.high.order1314, !dbg !23
  %434 = select i1 %compare.result1390, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1391 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1316, i32 50) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1392 = or i32 %compare.local.base1391, 67, !dbg !23
  %compare.low1393 = icmp ult i32 %compare.local1392, %compare.low.base1315, !dbg !23
  %compare.result1394 = select i1 %compare.high.equal1313, i1 %compare.low1393, i1 %compare.high.order1314, !dbg !23
  %435 = select i1 %compare.result1394, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %436 = shufflevector <32 x float> %379, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !25
  %437 = insertelement <4 x float> poison, float %432, i64 0, !dbg !25
  %438 = insertelement <4 x float> %437, float %433, i64 1, !dbg !25
  %439 = insertelement <4 x float> %438, float %434, i64 2, !dbg !25
  %440 = insertelement <4 x float> %439, float %435, i64 3, !dbg !25
  %441 = fadd <4 x float> %436, %440, !dbg !25
  %442 = extractelement <4 x float> %441, i64 0, !dbg !25
  %443 = extractelement <4 x float> %441, i64 1, !dbg !25
  %444 = extractelement <4 x float> %441, i64 2, !dbg !25
  %445 = extractelement <4 x float> %441, i64 3, !dbg !25
  %compare.local.base1395 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1316, i32 51) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1396 = or i32 %compare.local.base1395, 68, !dbg !23
  %compare.low1397 = icmp ult i32 %compare.local1396, %compare.low.base1315, !dbg !23
  %compare.result1398 = select i1 %compare.high.equal1313, i1 %compare.low1397, i1 %compare.high.order1314, !dbg !23
  %446 = select i1 %compare.result1398, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1399 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1316, i32 52) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1400 = or i32 %compare.local.base1399, 69, !dbg !23
  %compare.low1401 = icmp ult i32 %compare.local1400, %compare.low.base1315, !dbg !23
  %compare.result1402 = select i1 %compare.high.equal1313, i1 %compare.low1401, i1 %compare.high.order1314, !dbg !23
  %447 = select i1 %compare.result1402, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1403 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1316, i32 53) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1404 = or i32 %compare.local.base1403, 70, !dbg !23
  %compare.low1405 = icmp ult i32 %compare.local1404, %compare.low.base1315, !dbg !23
  %compare.result1406 = select i1 %compare.high.equal1313, i1 %compare.low1405, i1 %compare.high.order1314, !dbg !23
  %448 = select i1 %compare.result1406, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1407 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1316, i32 54) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1408 = or i32 %compare.local.base1407, 71, !dbg !23
  %compare.low1409 = icmp ult i32 %compare.local1408, %compare.low.base1315, !dbg !23
  %compare.result1410 = select i1 %compare.high.equal1313, i1 %compare.low1409, i1 %compare.high.order1314, !dbg !23
  %449 = select i1 %compare.result1410, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %450 = shufflevector <32 x float> %379, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !25
  %451 = insertelement <4 x float> poison, float %446, i64 0, !dbg !25
  %452 = insertelement <4 x float> %451, float %447, i64 1, !dbg !25
  %453 = insertelement <4 x float> %452, float %448, i64 2, !dbg !25
  %454 = insertelement <4 x float> %453, float %449, i64 3, !dbg !25
  %455 = fadd <4 x float> %450, %454, !dbg !25
  %456 = extractelement <4 x float> %455, i64 0, !dbg !25
  %457 = extractelement <4 x float> %455, i64 1, !dbg !25
  %458 = extractelement <4 x float> %455, i64 2, !dbg !25
  %459 = extractelement <4 x float> %455, i64 3, !dbg !25
  %compare.local.base1411 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1316, i32 55) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1412 = or i32 %compare.local.base1411, 72, !dbg !23
  %compare.low1413 = icmp ult i32 %compare.local1412, %compare.low.base1315, !dbg !23
  %compare.result1414 = select i1 %compare.high.equal1313, i1 %compare.low1413, i1 %compare.high.order1314, !dbg !23
  %460 = select i1 %compare.result1414, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1415 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1316, i32 56) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1416 = or i32 %compare.local.base1415, 73, !dbg !23
  %compare.low1417 = icmp ult i32 %compare.local1416, %compare.low.base1315, !dbg !23
  %compare.result1418 = select i1 %compare.high.equal1313, i1 %compare.low1417, i1 %compare.high.order1314, !dbg !23
  %461 = select i1 %compare.result1418, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1419 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1316, i32 57) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1420 = or i32 %compare.local.base1419, 74, !dbg !23
  %compare.low1421 = icmp ult i32 %compare.local1420, %compare.low.base1315, !dbg !23
  %compare.result1422 = select i1 %compare.high.equal1313, i1 %compare.low1421, i1 %compare.high.order1314, !dbg !23
  %462 = select i1 %compare.result1422, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1423 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1316, i32 58) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1424 = or i32 %compare.local.base1423, 75, !dbg !23
  %compare.low1425 = icmp ult i32 %compare.local1424, %compare.low.base1315, !dbg !23
  %compare.result1426 = select i1 %compare.high.equal1313, i1 %compare.low1425, i1 %compare.high.order1314, !dbg !23
  %463 = select i1 %compare.result1426, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %464 = shufflevector <32 x float> %379, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !25
  %465 = insertelement <4 x float> poison, float %460, i64 0, !dbg !25
  %466 = insertelement <4 x float> %465, float %461, i64 1, !dbg !25
  %467 = insertelement <4 x float> %466, float %462, i64 2, !dbg !25
  %468 = insertelement <4 x float> %467, float %463, i64 3, !dbg !25
  %469 = fadd <4 x float> %464, %468, !dbg !25
  %470 = extractelement <4 x float> %469, i64 0, !dbg !25
  %471 = extractelement <4 x float> %469, i64 1, !dbg !25
  %472 = extractelement <4 x float> %469, i64 2, !dbg !25
  %473 = extractelement <4 x float> %469, i64 3, !dbg !25
  %compare.local.base1427 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1316, i32 59) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1428 = or i32 %compare.local.base1427, 76, !dbg !23
  %compare.low1429 = icmp ult i32 %compare.local1428, %compare.low.base1315, !dbg !23
  %compare.result1430 = select i1 %compare.high.equal1313, i1 %compare.low1429, i1 %compare.high.order1314, !dbg !23
  %474 = select i1 %compare.result1430, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1431 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1316, i32 60) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1432 = or i32 %compare.local.base1431, 77, !dbg !23
  %compare.low1433 = icmp ult i32 %compare.local1432, %compare.low.base1315, !dbg !23
  %compare.result1434 = select i1 %compare.high.equal1313, i1 %compare.low1433, i1 %compare.high.order1314, !dbg !23
  %475 = select i1 %compare.result1434, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1435 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1316, i32 61) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1436 = or i32 %compare.local.base1435, 78, !dbg !23
  %compare.low1437 = icmp ult i32 %compare.local1436, %compare.low.base1315, !dbg !23
  %compare.result1438 = select i1 %compare.high.equal1313, i1 %compare.low1437, i1 %compare.high.order1314, !dbg !23
  %476 = select i1 %compare.result1438, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1439 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1316, i32 62) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1440 = or i32 %compare.local.base1439, 79, !dbg !23
  %compare.low1441 = icmp ult i32 %compare.local1440, %compare.low.base1315, !dbg !23
  %compare.result1442 = select i1 %compare.high.equal1313, i1 %compare.low1441, i1 %compare.high.order1314, !dbg !23
  %477 = select i1 %compare.result1442, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %478 = shufflevector <32 x float> %379, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !25
  %479 = insertelement <4 x float> poison, float %474, i64 0, !dbg !25
  %480 = insertelement <4 x float> %479, float %475, i64 1, !dbg !25
  %481 = insertelement <4 x float> %480, float %476, i64 2, !dbg !25
  %482 = insertelement <4 x float> %481, float %477, i64 3, !dbg !25
  %483 = fadd <4 x float> %478, %482, !dbg !25
  %484 = extractelement <4 x float> %483, i64 0, !dbg !25
  %485 = extractelement <4 x float> %483, i64 1, !dbg !25
  %486 = extractelement <4 x float> %483, i64 2, !dbg !25
  %487 = extractelement <4 x float> %483, i64 3, !dbg !25
  %488 = call float @llvm.maxnum.f32(float %271, float 0xFFF0000000000000), !dbg !26, !frisk.local_reduction !1
  %local.reduce222 = call float @llvm.maxnum.f32(float %488, float %285), !dbg !26, !frisk.local_reduction !1
  %local.reduce223 = call float @llvm.maxnum.f32(float %272, float %286), !dbg !26, !frisk.local_reduction !1
  %local.reduce224 = call float @llvm.maxnum.f32(float %273, float %287), !dbg !26, !frisk.local_reduction !1
  %local.reduce225 = call float @llvm.maxnum.f32(float %274, float %288), !dbg !26, !frisk.local_reduction !1
  %local.reduce226 = call float @llvm.maxnum.f32(float %local.reduce222, float %299), !dbg !26, !frisk.local_reduction !1
  %local.reduce227 = call float @llvm.maxnum.f32(float %local.reduce223, float %300), !dbg !26, !frisk.local_reduction !1
  %local.reduce228 = call float @llvm.maxnum.f32(float %local.reduce224, float %301), !dbg !26, !frisk.local_reduction !1
  %local.reduce229 = call float @llvm.maxnum.f32(float %local.reduce225, float %302), !dbg !26, !frisk.local_reduction !1
  %local.reduce230 = call float @llvm.maxnum.f32(float %local.reduce226, float %313), !dbg !26, !frisk.local_reduction !1
  %local.reduce231 = call float @llvm.maxnum.f32(float %local.reduce227, float %314), !dbg !26, !frisk.local_reduction !1
  %local.reduce232 = call float @llvm.maxnum.f32(float %local.reduce228, float %315), !dbg !26, !frisk.local_reduction !1
  %local.reduce233 = call float @llvm.maxnum.f32(float %local.reduce229, float %316), !dbg !26, !frisk.local_reduction !1
  %local.reduce234 = call float @llvm.maxnum.f32(float %local.reduce230, float %327), !dbg !26, !frisk.local_reduction !1
  %local.reduce235 = call float @llvm.maxnum.f32(float %local.reduce231, float %328), !dbg !26, !frisk.local_reduction !1
  %local.reduce236 = call float @llvm.maxnum.f32(float %local.reduce232, float %329), !dbg !26, !frisk.local_reduction !1
  %local.reduce237 = call float @llvm.maxnum.f32(float %local.reduce233, float %330), !dbg !26, !frisk.local_reduction !1
  %local.reduce238 = call float @llvm.maxnum.f32(float %local.reduce234, float %341), !dbg !26, !frisk.local_reduction !1
  %local.reduce239 = call float @llvm.maxnum.f32(float %local.reduce235, float %342), !dbg !26, !frisk.local_reduction !1
  %local.reduce240 = call float @llvm.maxnum.f32(float %local.reduce236, float %343), !dbg !26, !frisk.local_reduction !1
  %local.reduce241 = call float @llvm.maxnum.f32(float %local.reduce237, float %344), !dbg !26, !frisk.local_reduction !1
  %local.reduce242 = call float @llvm.maxnum.f32(float %local.reduce238, float %355), !dbg !26, !frisk.local_reduction !1
  %local.reduce243 = call float @llvm.maxnum.f32(float %local.reduce239, float %356), !dbg !26, !frisk.local_reduction !1
  %local.reduce244 = call float @llvm.maxnum.f32(float %local.reduce240, float %357), !dbg !26, !frisk.local_reduction !1
  %local.reduce245 = call float @llvm.maxnum.f32(float %local.reduce241, float %358), !dbg !26, !frisk.local_reduction !1
  %local.reduce246 = call float @llvm.maxnum.f32(float %local.reduce242, float %369), !dbg !26, !frisk.local_reduction !1
  %local.reduce247 = call float @llvm.maxnum.f32(float %local.reduce243, float %370), !dbg !26, !frisk.local_reduction !1
  %local.reduce248 = call float @llvm.maxnum.f32(float %local.reduce244, float %371), !dbg !26, !frisk.local_reduction !1
  %local.reduce249 = call float @llvm.maxnum.f32(float %local.reduce245, float %372), !dbg !26, !frisk.local_reduction !1
  %local.reduce250 = call float @llvm.maxnum.f32(float %local.reduce246, float %local.reduce247), !dbg !26, !frisk.local_reduction !1
  %local.reduce251 = call float @llvm.maxnum.f32(float %local.reduce248, float %local.reduce249), !dbg !26, !frisk.local_reduction !1
  %489 = call float @llvm.maxnum.f32(float %local.reduce250, float %local.reduce251), !dbg !26, !frisk.local_reduction !1
  %490 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !26
  %491 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %490), !dbg !26
  %492 = add nuw nsw i32 %491, 64, !dbg !26
  %493 = xor i32 %491, 16, !dbg !26
  %494 = and i32 %492, 192, !dbg !26
  %495 = icmp samesign ult i32 %493, %494, !dbg !26
  %496 = select i1 %495, i32 %493, i32 %491, !dbg !26
  %497 = shl nuw nsw i32 %496, 2, !dbg !26
  %498 = bitcast float %489 to i32, !dbg !26
  %499 = call i32 @llvm.amdgcn.ds.bpermute(i32 %497, i32 %498), !dbg !26
  %500 = bitcast i32 %499 to float, !dbg !26
  %501 = call float @llvm.maxnum.f32(float %489, float %500), !dbg !26
  %502 = xor i32 %491, 32, !dbg !26
  %503 = icmp samesign ult i32 %502, %494, !dbg !26
  %504 = select i1 %503, i32 %502, i32 %491, !dbg !26
  %505 = shl nuw nsw i32 %504, 2, !dbg !26
  %506 = bitcast float %501 to i32, !dbg !26
  %507 = call i32 @llvm.amdgcn.ds.bpermute(i32 %505, i32 %506), !dbg !26
  %508 = bitcast i32 %507 to float, !dbg !26
  %509 = call float @llvm.maxnum.f32(float %501, float %508), !dbg !26
  %510 = extractvalue [2 x <1 x float>] %46, 0, !dbg !26
  %511 = insertelement <1 x float> poison, float %509, i64 0, !dbg !26
  %512 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %511, <1 x float> %510), !dbg !26
  %513 = call float @llvm.maxnum.f32(float %386, float 0xFFF0000000000000), !dbg !26, !frisk.local_reduction !1
  %local.reduce191 = call float @llvm.maxnum.f32(float %513, float %400), !dbg !26, !frisk.local_reduction !1
  %local.reduce192 = call float @llvm.maxnum.f32(float %387, float %401), !dbg !26, !frisk.local_reduction !1
  %local.reduce193 = call float @llvm.maxnum.f32(float %388, float %402), !dbg !26, !frisk.local_reduction !1
  %local.reduce194 = call float @llvm.maxnum.f32(float %389, float %403), !dbg !26, !frisk.local_reduction !1
  %local.reduce195 = call float @llvm.maxnum.f32(float %local.reduce191, float %414), !dbg !26, !frisk.local_reduction !1
  %local.reduce196 = call float @llvm.maxnum.f32(float %local.reduce192, float %415), !dbg !26, !frisk.local_reduction !1
  %local.reduce197 = call float @llvm.maxnum.f32(float %local.reduce193, float %416), !dbg !26, !frisk.local_reduction !1
  %local.reduce198 = call float @llvm.maxnum.f32(float %local.reduce194, float %417), !dbg !26, !frisk.local_reduction !1
  %local.reduce199 = call float @llvm.maxnum.f32(float %local.reduce195, float %428), !dbg !26, !frisk.local_reduction !1
  %local.reduce200 = call float @llvm.maxnum.f32(float %local.reduce196, float %429), !dbg !26, !frisk.local_reduction !1
  %local.reduce201 = call float @llvm.maxnum.f32(float %local.reduce197, float %430), !dbg !26, !frisk.local_reduction !1
  %local.reduce202 = call float @llvm.maxnum.f32(float %local.reduce198, float %431), !dbg !26, !frisk.local_reduction !1
  %local.reduce203 = call float @llvm.maxnum.f32(float %local.reduce199, float %442), !dbg !26, !frisk.local_reduction !1
  %local.reduce204 = call float @llvm.maxnum.f32(float %local.reduce200, float %443), !dbg !26, !frisk.local_reduction !1
  %local.reduce205 = call float @llvm.maxnum.f32(float %local.reduce201, float %444), !dbg !26, !frisk.local_reduction !1
  %local.reduce206 = call float @llvm.maxnum.f32(float %local.reduce202, float %445), !dbg !26, !frisk.local_reduction !1
  %local.reduce207 = call float @llvm.maxnum.f32(float %local.reduce203, float %456), !dbg !26, !frisk.local_reduction !1
  %local.reduce208 = call float @llvm.maxnum.f32(float %local.reduce204, float %457), !dbg !26, !frisk.local_reduction !1
  %local.reduce209 = call float @llvm.maxnum.f32(float %local.reduce205, float %458), !dbg !26, !frisk.local_reduction !1
  %local.reduce210 = call float @llvm.maxnum.f32(float %local.reduce206, float %459), !dbg !26, !frisk.local_reduction !1
  %local.reduce211 = call float @llvm.maxnum.f32(float %local.reduce207, float %470), !dbg !26, !frisk.local_reduction !1
  %local.reduce212 = call float @llvm.maxnum.f32(float %local.reduce208, float %471), !dbg !26, !frisk.local_reduction !1
  %local.reduce213 = call float @llvm.maxnum.f32(float %local.reduce209, float %472), !dbg !26, !frisk.local_reduction !1
  %local.reduce214 = call float @llvm.maxnum.f32(float %local.reduce210, float %473), !dbg !26, !frisk.local_reduction !1
  %local.reduce215 = call float @llvm.maxnum.f32(float %local.reduce211, float %484), !dbg !26, !frisk.local_reduction !1
  %local.reduce216 = call float @llvm.maxnum.f32(float %local.reduce212, float %485), !dbg !26, !frisk.local_reduction !1
  %local.reduce217 = call float @llvm.maxnum.f32(float %local.reduce213, float %486), !dbg !26, !frisk.local_reduction !1
  %local.reduce218 = call float @llvm.maxnum.f32(float %local.reduce214, float %487), !dbg !26, !frisk.local_reduction !1
  %local.reduce219 = call float @llvm.maxnum.f32(float %local.reduce215, float %local.reduce216), !dbg !26, !frisk.local_reduction !1
  %local.reduce220 = call float @llvm.maxnum.f32(float %local.reduce217, float %local.reduce218), !dbg !26, !frisk.local_reduction !1
  %514 = call float @llvm.maxnum.f32(float %local.reduce219, float %local.reduce220), !dbg !26, !frisk.local_reduction !1
  %515 = bitcast float %514 to i32, !dbg !26
  %516 = call i32 @llvm.amdgcn.ds.bpermute(i32 %497, i32 %515), !dbg !26
  %517 = bitcast i32 %516 to float, !dbg !26
  %518 = call float @llvm.maxnum.f32(float %514, float %517), !dbg !26
  %519 = bitcast float %518 to i32, !dbg !26
  %520 = call i32 @llvm.amdgcn.ds.bpermute(i32 %505, i32 %519), !dbg !26
  %521 = bitcast i32 %520 to float, !dbg !26
  %522 = call float @llvm.maxnum.f32(float %518, float %521), !dbg !26
  %523 = extractvalue [2 x <1 x float>] %46, 1, !dbg !26
  %524 = insertelement <1 x float> poison, float %522, i64 0, !dbg !26
  %525 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %524, <1 x float> %523), !dbg !26
  %526 = shufflevector <1 x float> %512, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !27
  %527 = fsub <4 x float> %270, %526, !dbg !27
  %528 = fsub <4 x float> %284, %526, !dbg !27
  %529 = fsub <4 x float> %298, %526, !dbg !27
  %530 = fsub <4 x float> %312, %526, !dbg !27
  %531 = fsub <4 x float> %326, %526, !dbg !27
  %532 = fsub <4 x float> %340, %526, !dbg !27
  %533 = fsub <4 x float> %354, %526, !dbg !27
  %534 = fsub <4 x float> %368, %526, !dbg !27
  %535 = shufflevector <1 x float> %525, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !27
  %536 = fsub <4 x float> %385, %535, !dbg !27
  %537 = fsub <4 x float> %399, %535, !dbg !27
  %538 = fsub <4 x float> %413, %535, !dbg !27
  %539 = fsub <4 x float> %427, %535, !dbg !27
  %540 = fsub <4 x float> %441, %535, !dbg !27
  %541 = fsub <4 x float> %455, %535, !dbg !27
  %542 = fsub <4 x float> %469, %535, !dbg !27
  %543 = fsub <4 x float> %483, %535, !dbg !27
  %544 = fsub <1 x float> %510, %512, !dbg !28
  %545 = extractelement <1 x float> %544, i64 0, !dbg !29
  %546 = call float @llvm.amdgcn.exp2.f32(float %545), !dbg !29
  %547 = insertelement <4 x float> poison, float %546, i64 0, !dbg !34
  %548 = shufflevector <4 x float> %547, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !34
  %549 = extractvalue [2 x <1 x float>] %45, 0, !dbg !30
  %550 = extractelement <1 x float> %549, i64 0, !dbg !30
  %551 = fmul float %550, %546, !dbg !30
  %552 = fsub <1 x float> %523, %525, !dbg !28
  %553 = extractelement <1 x float> %552, i64 0, !dbg !29
  %554 = call float @llvm.amdgcn.exp2.f32(float %553), !dbg !29
  %555 = insertelement <4 x float> poison, float %554, i64 0, !dbg !34
  %556 = shufflevector <4 x float> %555, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !34
  %557 = extractvalue [2 x <1 x float>] %45, 1, !dbg !30
  %558 = extractelement <1 x float> %557, i64 0, !dbg !30
  %559 = fmul float %558, %554, !dbg !30
  %560 = extractelement <4 x float> %527, i64 0, !dbg !31
  %561 = call float @llvm.amdgcn.exp2.f32(float %560), !dbg !31
  %562 = insertelement <4 x float> poison, float %561, i64 0, !dbg !33
  %563 = extractelement <4 x float> %527, i64 1, !dbg !31
  %564 = call float @llvm.amdgcn.exp2.f32(float %563), !dbg !31
  %565 = insertelement <4 x float> %562, float %564, i64 1, !dbg !33
  %566 = extractelement <4 x float> %527, i64 2, !dbg !31
  %567 = call float @llvm.amdgcn.exp2.f32(float %566), !dbg !31
  %568 = insertelement <4 x float> %565, float %567, i64 2, !dbg !33
  %569 = extractelement <4 x float> %527, i64 3, !dbg !31
  %570 = call float @llvm.amdgcn.exp2.f32(float %569), !dbg !31
  %571 = insertelement <4 x float> %568, float %570, i64 3, !dbg !33
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %571) #9, !dbg !33, !frisk.register_lifetimes !1
  %572 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !33
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %572) #9, !dbg !31, !frisk.register_lifetimes !1
  %573 = extractelement <4 x float> %528, i64 0, !dbg !31
  %574 = call float @llvm.amdgcn.exp2.f32(float %573), !dbg !31
  %575 = insertelement <4 x float> poison, float %574, i64 0, !dbg !33
  %local.reduce160 = fadd float %561, %574, !dbg !32, !frisk.local_reduction !1
  %576 = extractelement <4 x float> %528, i64 1, !dbg !31
  %577 = call float @llvm.amdgcn.exp2.f32(float %576), !dbg !31
  %578 = insertelement <4 x float> %575, float %577, i64 1, !dbg !33
  %local.reduce161 = fadd float %564, %577, !dbg !32, !frisk.local_reduction !1
  %579 = extractelement <4 x float> %528, i64 2, !dbg !31
  %580 = call float @llvm.amdgcn.exp2.f32(float %579), !dbg !31
  %581 = insertelement <4 x float> %578, float %580, i64 2, !dbg !33
  %local.reduce162 = fadd float %567, %580, !dbg !32, !frisk.local_reduction !1
  %582 = extractelement <4 x float> %528, i64 3, !dbg !31
  %583 = call float @llvm.amdgcn.exp2.f32(float %582), !dbg !31
  %584 = insertelement <4 x float> %581, float %583, i64 3, !dbg !33
  %math.result432 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %584) #9, !dbg !33, !frisk.register_lifetimes !1
  %585 = fptrunc <4 x float> %math.result432 to <4 x half>, !dbg !33
  %math.packed433 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %585) #9, !dbg !32, !frisk.register_lifetimes !1
  %local.reduce163 = fadd float %570, %583, !dbg !32, !frisk.local_reduction !1
  %586 = extractelement <4 x float> %529, i64 0, !dbg !31
  %587 = call float @llvm.amdgcn.exp2.f32(float %586), !dbg !31
  %588 = insertelement <4 x float> poison, float %587, i64 0, !dbg !33
  %local.reduce164 = fadd float %local.reduce160, %587, !dbg !32, !frisk.local_reduction !1
  %589 = extractelement <4 x float> %529, i64 1, !dbg !31
  %590 = call float @llvm.amdgcn.exp2.f32(float %589), !dbg !31
  %591 = insertelement <4 x float> %588, float %590, i64 1, !dbg !33
  %local.reduce165 = fadd float %local.reduce161, %590, !dbg !32, !frisk.local_reduction !1
  %592 = extractelement <4 x float> %529, i64 2, !dbg !31
  %593 = call float @llvm.amdgcn.exp2.f32(float %592), !dbg !31
  %594 = insertelement <4 x float> %591, float %593, i64 2, !dbg !33
  %local.reduce166 = fadd float %local.reduce162, %593, !dbg !32, !frisk.local_reduction !1
  %595 = extractelement <4 x float> %529, i64 3, !dbg !31
  %596 = call float @llvm.amdgcn.exp2.f32(float %595), !dbg !31
  %597 = insertelement <4 x float> %594, float %596, i64 3, !dbg !33
  %math.result434 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %597) #9, !dbg !33, !frisk.register_lifetimes !1
  %598 = fptrunc <4 x float> %math.result434 to <4 x half>, !dbg !33
  %math.packed435 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %598) #9, !dbg !32, !frisk.register_lifetimes !1
  %local.reduce167 = fadd float %local.reduce163, %596, !dbg !32, !frisk.local_reduction !1
  %599 = extractelement <4 x float> %530, i64 0, !dbg !31
  %600 = call float @llvm.amdgcn.exp2.f32(float %599), !dbg !31
  %601 = insertelement <4 x float> poison, float %600, i64 0, !dbg !33
  %local.reduce168 = fadd float %local.reduce164, %600, !dbg !32, !frisk.local_reduction !1
  %602 = extractelement <4 x float> %530, i64 1, !dbg !31
  %603 = call float @llvm.amdgcn.exp2.f32(float %602), !dbg !31
  %604 = insertelement <4 x float> %601, float %603, i64 1, !dbg !33
  %local.reduce169 = fadd float %local.reduce165, %603, !dbg !32, !frisk.local_reduction !1
  %605 = extractelement <4 x float> %530, i64 2, !dbg !31
  %606 = call float @llvm.amdgcn.exp2.f32(float %605), !dbg !31
  %607 = insertelement <4 x float> %604, float %606, i64 2, !dbg !33
  %local.reduce170 = fadd float %local.reduce166, %606, !dbg !32, !frisk.local_reduction !1
  %608 = extractelement <4 x float> %530, i64 3, !dbg !31
  %609 = call float @llvm.amdgcn.exp2.f32(float %608), !dbg !31
  %610 = insertelement <4 x float> %607, float %609, i64 3, !dbg !33
  %math.result436 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %610) #9, !dbg !33, !frisk.register_lifetimes !1
  %611 = fptrunc <4 x float> %math.result436 to <4 x half>, !dbg !33
  %math.packed437 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %611) #9, !dbg !32, !frisk.register_lifetimes !1
  %local.reduce171 = fadd float %local.reduce167, %609, !dbg !32, !frisk.local_reduction !1
  %612 = extractelement <4 x float> %531, i64 0, !dbg !31
  %613 = call float @llvm.amdgcn.exp2.f32(float %612), !dbg !31
  %614 = insertelement <4 x float> poison, float %613, i64 0, !dbg !33
  %local.reduce172 = fadd float %local.reduce168, %613, !dbg !32, !frisk.local_reduction !1
  %615 = extractelement <4 x float> %531, i64 1, !dbg !31
  %616 = call float @llvm.amdgcn.exp2.f32(float %615), !dbg !31
  %617 = insertelement <4 x float> %614, float %616, i64 1, !dbg !33
  %local.reduce173 = fadd float %local.reduce169, %616, !dbg !32, !frisk.local_reduction !1
  %618 = extractelement <4 x float> %531, i64 2, !dbg !31
  %619 = call float @llvm.amdgcn.exp2.f32(float %618), !dbg !31
  %620 = insertelement <4 x float> %617, float %619, i64 2, !dbg !33
  %local.reduce174 = fadd float %local.reduce170, %619, !dbg !32, !frisk.local_reduction !1
  %621 = extractelement <4 x float> %531, i64 3, !dbg !31
  %622 = call float @llvm.amdgcn.exp2.f32(float %621), !dbg !31
  %623 = insertelement <4 x float> %620, float %622, i64 3, !dbg !33
  %math.result438 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %623) #9, !dbg !33, !frisk.register_lifetimes !1
  %624 = fptrunc <4 x float> %math.result438 to <4 x half>, !dbg !33
  %math.packed439 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %624) #9, !dbg !32, !frisk.register_lifetimes !1
  %local.reduce175 = fadd float %local.reduce171, %622, !dbg !32, !frisk.local_reduction !1
  %625 = extractelement <4 x float> %532, i64 0, !dbg !31
  %626 = call float @llvm.amdgcn.exp2.f32(float %625), !dbg !31
  %627 = insertelement <4 x float> poison, float %626, i64 0, !dbg !33
  %local.reduce176 = fadd float %local.reduce172, %626, !dbg !32, !frisk.local_reduction !1
  %628 = extractelement <4 x float> %532, i64 1, !dbg !31
  %629 = call float @llvm.amdgcn.exp2.f32(float %628), !dbg !31
  %630 = insertelement <4 x float> %627, float %629, i64 1, !dbg !33
  %local.reduce177 = fadd float %local.reduce173, %629, !dbg !32, !frisk.local_reduction !1
  %631 = extractelement <4 x float> %532, i64 2, !dbg !31
  %632 = call float @llvm.amdgcn.exp2.f32(float %631), !dbg !31
  %633 = insertelement <4 x float> %630, float %632, i64 2, !dbg !33
  %local.reduce178 = fadd float %local.reduce174, %632, !dbg !32, !frisk.local_reduction !1
  %634 = extractelement <4 x float> %532, i64 3, !dbg !31
  %635 = call float @llvm.amdgcn.exp2.f32(float %634), !dbg !31
  %636 = insertelement <4 x float> %633, float %635, i64 3, !dbg !33
  %math.result440 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %636) #9, !dbg !33, !frisk.register_lifetimes !1
  %637 = fptrunc <4 x float> %math.result440 to <4 x half>, !dbg !33
  %math.packed441 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %637) #9, !dbg !32, !frisk.register_lifetimes !1
  %local.reduce179 = fadd float %local.reduce175, %635, !dbg !32, !frisk.local_reduction !1
  %638 = extractelement <4 x float> %533, i64 0, !dbg !31
  %639 = call float @llvm.amdgcn.exp2.f32(float %638), !dbg !31
  %640 = insertelement <4 x float> poison, float %639, i64 0, !dbg !33
  %local.reduce180 = fadd float %local.reduce176, %639, !dbg !32, !frisk.local_reduction !1
  %641 = extractelement <4 x float> %533, i64 1, !dbg !31
  %642 = call float @llvm.amdgcn.exp2.f32(float %641), !dbg !31
  %643 = insertelement <4 x float> %640, float %642, i64 1, !dbg !33
  %local.reduce181 = fadd float %local.reduce177, %642, !dbg !32, !frisk.local_reduction !1
  %644 = extractelement <4 x float> %533, i64 2, !dbg !31
  %645 = call float @llvm.amdgcn.exp2.f32(float %644), !dbg !31
  %646 = insertelement <4 x float> %643, float %645, i64 2, !dbg !33
  %local.reduce182 = fadd float %local.reduce178, %645, !dbg !32, !frisk.local_reduction !1
  %647 = extractelement <4 x float> %533, i64 3, !dbg !31
  %648 = call float @llvm.amdgcn.exp2.f32(float %647), !dbg !31
  %649 = insertelement <4 x float> %646, float %648, i64 3, !dbg !33
  %math.result442 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %649) #9, !dbg !33, !frisk.register_lifetimes !1
  %650 = fptrunc <4 x float> %math.result442 to <4 x half>, !dbg !33
  %math.packed443 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %650) #9, !dbg !32, !frisk.register_lifetimes !1
  %local.reduce183 = fadd float %local.reduce179, %648, !dbg !32, !frisk.local_reduction !1
  %651 = extractelement <4 x float> %534, i64 0, !dbg !31
  %652 = call float @llvm.amdgcn.exp2.f32(float %651), !dbg !31
  %653 = insertelement <4 x float> poison, float %652, i64 0, !dbg !33
  %local.reduce184 = fadd float %local.reduce180, %652, !dbg !32, !frisk.local_reduction !1
  %654 = extractelement <4 x float> %534, i64 1, !dbg !31
  %655 = call float @llvm.amdgcn.exp2.f32(float %654), !dbg !31
  %656 = insertelement <4 x float> %653, float %655, i64 1, !dbg !33
  %local.reduce185 = fadd float %local.reduce181, %655, !dbg !32, !frisk.local_reduction !1
  %local.reduce188 = fadd float %local.reduce184, %local.reduce185, !dbg !32, !frisk.local_reduction !1
  %657 = extractelement <4 x float> %534, i64 2, !dbg !31
  %658 = call float @llvm.amdgcn.exp2.f32(float %657), !dbg !31
  %659 = insertelement <4 x float> %656, float %658, i64 2, !dbg !33
  %local.reduce186 = fadd float %local.reduce182, %658, !dbg !32, !frisk.local_reduction !1
  %660 = extractelement <4 x float> %534, i64 3, !dbg !31
  %661 = call float @llvm.amdgcn.exp2.f32(float %660), !dbg !31
  %662 = insertelement <4 x float> %659, float %661, i64 3, !dbg !33
  %math.result444 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %662) #9, !dbg !33, !frisk.register_lifetimes !1
  %663 = fptrunc <4 x float> %math.result444 to <4 x half>, !dbg !33
  %math.packed445 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %663) #9, !dbg !32, !frisk.register_lifetimes !1
  %local.reduce187 = fadd float %local.reduce183, %661, !dbg !32, !frisk.local_reduction !1
  %local.reduce189 = fadd float %local.reduce186, %local.reduce187, !dbg !32, !frisk.local_reduction !1
  %664 = fadd float %local.reduce188, %local.reduce189, !dbg !32, !frisk.local_reduction !1
  %665 = extractelement <4 x float> %536, i64 0, !dbg !31
  %666 = call float @llvm.amdgcn.exp2.f32(float %665), !dbg !31
  %667 = insertelement <4 x float> poison, float %666, i64 0, !dbg !33
  %668 = extractelement <4 x float> %536, i64 1, !dbg !31
  %669 = call float @llvm.amdgcn.exp2.f32(float %668), !dbg !31
  %670 = insertelement <4 x float> %667, float %669, i64 1, !dbg !33
  %671 = extractelement <4 x float> %536, i64 2, !dbg !31
  %672 = call float @llvm.amdgcn.exp2.f32(float %671), !dbg !31
  %673 = insertelement <4 x float> %670, float %672, i64 2, !dbg !33
  %674 = extractelement <4 x float> %536, i64 3, !dbg !31
  %675 = call float @llvm.amdgcn.exp2.f32(float %674), !dbg !31
  %676 = insertelement <4 x float> %673, float %675, i64 3, !dbg !33
  %math.result446 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %676) #9, !dbg !33, !frisk.register_lifetimes !1
  %677 = fptrunc <4 x float> %math.result446 to <4 x half>, !dbg !33
  %math.packed447 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %677) #9, !dbg !31, !frisk.register_lifetimes !1
  %678 = extractelement <4 x float> %537, i64 0, !dbg !31
  %679 = call float @llvm.amdgcn.exp2.f32(float %678), !dbg !31
  %680 = insertelement <4 x float> poison, float %679, i64 0, !dbg !33
  %local.reduce = fadd float %666, %679, !dbg !32, !frisk.local_reduction !1
  %681 = extractelement <4 x float> %537, i64 1, !dbg !31
  %682 = call float @llvm.amdgcn.exp2.f32(float %681), !dbg !31
  %683 = insertelement <4 x float> %680, float %682, i64 1, !dbg !33
  %local.reduce130 = fadd float %669, %682, !dbg !32, !frisk.local_reduction !1
  %684 = extractelement <4 x float> %537, i64 2, !dbg !31
  %685 = call float @llvm.amdgcn.exp2.f32(float %684), !dbg !31
  %686 = insertelement <4 x float> %683, float %685, i64 2, !dbg !33
  %local.reduce131 = fadd float %672, %685, !dbg !32, !frisk.local_reduction !1
  %687 = extractelement <4 x float> %537, i64 3, !dbg !31
  %688 = call float @llvm.amdgcn.exp2.f32(float %687), !dbg !31
  %689 = insertelement <4 x float> %686, float %688, i64 3, !dbg !33
  %math.result448 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %689) #9, !dbg !33, !frisk.register_lifetimes !1
  %690 = fptrunc <4 x float> %math.result448 to <4 x half>, !dbg !33
  %math.packed449 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %690) #9, !dbg !32, !frisk.register_lifetimes !1
  %local.reduce132 = fadd float %675, %688, !dbg !32, !frisk.local_reduction !1
  %691 = extractelement <4 x float> %538, i64 0, !dbg !31
  %692 = call float @llvm.amdgcn.exp2.f32(float %691), !dbg !31
  %693 = insertelement <4 x float> poison, float %692, i64 0, !dbg !33
  %local.reduce133 = fadd float %local.reduce, %692, !dbg !32, !frisk.local_reduction !1
  %694 = extractelement <4 x float> %538, i64 1, !dbg !31
  %695 = call float @llvm.amdgcn.exp2.f32(float %694), !dbg !31
  %696 = insertelement <4 x float> %693, float %695, i64 1, !dbg !33
  %local.reduce134 = fadd float %local.reduce130, %695, !dbg !32, !frisk.local_reduction !1
  %697 = extractelement <4 x float> %538, i64 2, !dbg !31
  %698 = call float @llvm.amdgcn.exp2.f32(float %697), !dbg !31
  %699 = insertelement <4 x float> %696, float %698, i64 2, !dbg !33
  %local.reduce135 = fadd float %local.reduce131, %698, !dbg !32, !frisk.local_reduction !1
  %700 = extractelement <4 x float> %538, i64 3, !dbg !31
  %701 = call float @llvm.amdgcn.exp2.f32(float %700), !dbg !31
  %702 = insertelement <4 x float> %699, float %701, i64 3, !dbg !33
  %math.result450 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %702) #9, !dbg !33, !frisk.register_lifetimes !1
  %703 = fptrunc <4 x float> %math.result450 to <4 x half>, !dbg !33
  %math.packed451 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %703) #9, !dbg !32, !frisk.register_lifetimes !1
  %local.reduce136 = fadd float %local.reduce132, %701, !dbg !32, !frisk.local_reduction !1
  %704 = extractelement <4 x float> %539, i64 0, !dbg !31
  %705 = call float @llvm.amdgcn.exp2.f32(float %704), !dbg !31
  %706 = insertelement <4 x float> poison, float %705, i64 0, !dbg !33
  %local.reduce137 = fadd float %local.reduce133, %705, !dbg !32, !frisk.local_reduction !1
  %707 = extractelement <4 x float> %539, i64 1, !dbg !31
  %708 = call float @llvm.amdgcn.exp2.f32(float %707), !dbg !31
  %709 = insertelement <4 x float> %706, float %708, i64 1, !dbg !33
  %local.reduce138 = fadd float %local.reduce134, %708, !dbg !32, !frisk.local_reduction !1
  %710 = extractelement <4 x float> %539, i64 2, !dbg !31
  %711 = call float @llvm.amdgcn.exp2.f32(float %710), !dbg !31
  %712 = insertelement <4 x float> %709, float %711, i64 2, !dbg !33
  %local.reduce139 = fadd float %local.reduce135, %711, !dbg !32, !frisk.local_reduction !1
  %713 = extractelement <4 x float> %539, i64 3, !dbg !31
  %714 = call float @llvm.amdgcn.exp2.f32(float %713), !dbg !31
  %715 = insertelement <4 x float> %712, float %714, i64 3, !dbg !33
  %math.result452 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %715) #9, !dbg !33, !frisk.register_lifetimes !1
  %716 = fptrunc <4 x float> %math.result452 to <4 x half>, !dbg !33
  %math.packed453 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %716) #9, !dbg !32, !frisk.register_lifetimes !1
  %local.reduce140 = fadd float %local.reduce136, %714, !dbg !32, !frisk.local_reduction !1
  %717 = extractelement <4 x float> %540, i64 0, !dbg !31
  %718 = call float @llvm.amdgcn.exp2.f32(float %717), !dbg !31
  %719 = insertelement <4 x float> poison, float %718, i64 0, !dbg !33
  %local.reduce141 = fadd float %local.reduce137, %718, !dbg !32, !frisk.local_reduction !1
  %720 = extractelement <4 x float> %540, i64 1, !dbg !31
  %721 = call float @llvm.amdgcn.exp2.f32(float %720), !dbg !31
  %722 = insertelement <4 x float> %719, float %721, i64 1, !dbg !33
  %local.reduce142 = fadd float %local.reduce138, %721, !dbg !32, !frisk.local_reduction !1
  %723 = extractelement <4 x float> %540, i64 2, !dbg !31
  %724 = call float @llvm.amdgcn.exp2.f32(float %723), !dbg !31
  %725 = insertelement <4 x float> %722, float %724, i64 2, !dbg !33
  %local.reduce143 = fadd float %local.reduce139, %724, !dbg !32, !frisk.local_reduction !1
  %726 = extractelement <4 x float> %540, i64 3, !dbg !31
  %727 = call float @llvm.amdgcn.exp2.f32(float %726), !dbg !31
  %728 = insertelement <4 x float> %725, float %727, i64 3, !dbg !33
  %math.result454 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %728) #9, !dbg !33, !frisk.register_lifetimes !1
  %729 = fptrunc <4 x float> %math.result454 to <4 x half>, !dbg !33
  %math.packed455 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %729) #9, !dbg !32, !frisk.register_lifetimes !1
  %local.reduce144 = fadd float %local.reduce140, %727, !dbg !32, !frisk.local_reduction !1
  %730 = extractelement <4 x float> %541, i64 0, !dbg !31
  %731 = call float @llvm.amdgcn.exp2.f32(float %730), !dbg !31
  %732 = insertelement <4 x float> poison, float %731, i64 0, !dbg !33
  %local.reduce145 = fadd float %local.reduce141, %731, !dbg !32, !frisk.local_reduction !1
  %733 = extractelement <4 x float> %541, i64 1, !dbg !31
  %734 = call float @llvm.amdgcn.exp2.f32(float %733), !dbg !31
  %735 = insertelement <4 x float> %732, float %734, i64 1, !dbg !33
  %local.reduce146 = fadd float %local.reduce142, %734, !dbg !32, !frisk.local_reduction !1
  %736 = extractelement <4 x float> %541, i64 2, !dbg !31
  %737 = call float @llvm.amdgcn.exp2.f32(float %736), !dbg !31
  %738 = insertelement <4 x float> %735, float %737, i64 2, !dbg !33
  %local.reduce147 = fadd float %local.reduce143, %737, !dbg !32, !frisk.local_reduction !1
  %739 = extractelement <4 x float> %541, i64 3, !dbg !31
  %740 = call float @llvm.amdgcn.exp2.f32(float %739), !dbg !31
  %741 = insertelement <4 x float> %738, float %740, i64 3, !dbg !33
  %math.result456 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %741) #9, !dbg !33, !frisk.register_lifetimes !1
  %742 = fptrunc <4 x float> %math.result456 to <4 x half>, !dbg !33
  %math.packed457 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %742) #9, !dbg !32, !frisk.register_lifetimes !1
  %local.reduce148 = fadd float %local.reduce144, %740, !dbg !32, !frisk.local_reduction !1
  %743 = extractelement <4 x float> %542, i64 0, !dbg !31
  %744 = call float @llvm.amdgcn.exp2.f32(float %743), !dbg !31
  %745 = insertelement <4 x float> poison, float %744, i64 0, !dbg !33
  %local.reduce149 = fadd float %local.reduce145, %744, !dbg !32, !frisk.local_reduction !1
  %746 = extractelement <4 x float> %542, i64 1, !dbg !31
  %747 = call float @llvm.amdgcn.exp2.f32(float %746), !dbg !31
  %748 = insertelement <4 x float> %745, float %747, i64 1, !dbg !33
  %local.reduce150 = fadd float %local.reduce146, %747, !dbg !32, !frisk.local_reduction !1
  %749 = extractelement <4 x float> %542, i64 2, !dbg !31
  %750 = call float @llvm.amdgcn.exp2.f32(float %749), !dbg !31
  %751 = insertelement <4 x float> %748, float %750, i64 2, !dbg !33
  %local.reduce151 = fadd float %local.reduce147, %750, !dbg !32, !frisk.local_reduction !1
  %752 = extractelement <4 x float> %542, i64 3, !dbg !31
  %753 = call float @llvm.amdgcn.exp2.f32(float %752), !dbg !31
  %754 = insertelement <4 x float> %751, float %753, i64 3, !dbg !33
  %math.result458 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %754) #9, !dbg !33, !frisk.register_lifetimes !1
  %755 = fptrunc <4 x float> %math.result458 to <4 x half>, !dbg !33
  %math.packed459 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %755) #9, !dbg !32, !frisk.register_lifetimes !1
  %local.reduce152 = fadd float %local.reduce148, %753, !dbg !32, !frisk.local_reduction !1
  %756 = extractelement <4 x float> %543, i64 0, !dbg !31
  %757 = call float @llvm.amdgcn.exp2.f32(float %756), !dbg !31
  %758 = insertelement <4 x float> poison, float %757, i64 0, !dbg !33
  %local.reduce153 = fadd float %local.reduce149, %757, !dbg !32, !frisk.local_reduction !1
  %759 = extractelement <4 x float> %543, i64 1, !dbg !31
  %760 = call float @llvm.amdgcn.exp2.f32(float %759), !dbg !31
  %761 = insertelement <4 x float> %758, float %760, i64 1, !dbg !33
  %local.reduce154 = fadd float %local.reduce150, %760, !dbg !32, !frisk.local_reduction !1
  %local.reduce157 = fadd float %local.reduce153, %local.reduce154, !dbg !32, !frisk.local_reduction !1
  %762 = extractelement <4 x float> %543, i64 2, !dbg !31
  %763 = call float @llvm.amdgcn.exp2.f32(float %762), !dbg !31
  %764 = insertelement <4 x float> %761, float %763, i64 2, !dbg !33
  %local.reduce155 = fadd float %local.reduce151, %763, !dbg !32, !frisk.local_reduction !1
  %765 = extractelement <4 x float> %543, i64 3, !dbg !31
  %766 = call float @llvm.amdgcn.exp2.f32(float %765), !dbg !31
  %767 = insertelement <4 x float> %764, float %766, i64 3, !dbg !33
  %math.result460 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %767) #9, !dbg !33, !frisk.register_lifetimes !1
  %768 = fptrunc <4 x float> %math.result460 to <4 x half>, !dbg !33
  %math.packed461 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %768) #9, !dbg !32, !frisk.register_lifetimes !1
  %local.reduce156 = fadd float %local.reduce152, %766, !dbg !32, !frisk.local_reduction !1
  %local.reduce158 = fadd float %local.reduce155, %local.reduce156, !dbg !32, !frisk.local_reduction !1
  %769 = fadd float %local.reduce157, %local.reduce158, !dbg !32, !frisk.local_reduction !1
  %770 = bitcast float %664 to i32, !dbg !32
  %771 = call i32 @llvm.amdgcn.ds.bpermute(i32 %497, i32 %770), !dbg !32
  %772 = bitcast i32 %771 to float, !dbg !32
  %773 = fadd float %664, %772, !dbg !32
  %774 = bitcast float %773 to i32, !dbg !32
  %775 = call i32 @llvm.amdgcn.ds.bpermute(i32 %505, i32 %774), !dbg !32
  %776 = bitcast i32 %775 to float, !dbg !32
  %777 = fadd float %773, %776, !dbg !32
  %778 = fadd float %777, %551, !dbg !32
  %779 = insertelement <1 x float> poison, float %778, i64 0, !dbg !8
  %780 = insertvalue [2 x <1 x float>] poison, <1 x float> %779, 0, !dbg !8
  %781 = bitcast float %769 to i32, !dbg !32
  %782 = call i32 @llvm.amdgcn.ds.bpermute(i32 %497, i32 %781), !dbg !32
  %783 = bitcast i32 %782 to float, !dbg !32
  %784 = fadd float %769, %783, !dbg !32
  %785 = bitcast float %784 to i32, !dbg !32
  %786 = call i32 @llvm.amdgcn.ds.bpermute(i32 %505, i32 %785), !dbg !32
  %787 = bitcast i32 %786 to float, !dbg !32
  %788 = fadd float %784, %787, !dbg !32
  %789 = fadd float %788, %559, !dbg !32
  %790 = insertelement <1 x float> poison, float %789, i64 0, !dbg !8
  %791 = insertvalue [2 x <1 x float>] %780, <1 x float> %790, 1, !dbg !8
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %address.local913 = shl nuw nsw i32 %address.thread888, 3, !dbg !14
  %address.local914 = and i32 %address.local913, 120, !dbg !14
  %address.local915 = shl nuw nsw i32 %address.thread888, 5, !dbg !14
  %address.local916 = and i32 %address.local915, 1536, !dbg !14
  %address.local917 = or disjoint i32 %address.local914, %address.local916, !dbg !14
  %address.local918 = zext nneg i32 %address.local917 to i64, !dbg !14
  %address.local919 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local918, !dbg !14
  %792 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local919, i32 0) #0, !dbg !14
  %793 = bitcast <4 x i32> %792 to <8 x half>, !dbg !14
  %address.local920 = and i64 %address.local889, 60, !dbg !14
  %address.local921 = add nuw nsw i64 %address.local920, %address.local889, !dbg !14
  %address.local922 = shl nuw nsw i64 %address.local921, 3, !dbg !14
  %address.local923 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local922, !dbg !14
  %address.local924 = lshr i64 %address.local889, 4, !dbg !14
  %address.local925 = and i64 %address.local924, 3, !dbg !14
  %address.local926 = mul nuw nsw i64 %address.local925, 384, !dbg !14
  %address.local927 = getelementptr [2 x i8], ptr addrspace(3) %address.local923, i64 %address.local926, !dbg !14
  %address.local928 = lshr i64 %address.local889, 2, !dbg !7
  %address.local929 = mul nsw i64 %address.local928, -32, !dbg !14
  %address.local930 = getelementptr [2 x i8], ptr addrspace(3) %address.local927, i64 %address.local929, !dbg !14
  %address.local931 = getelementptr i8, ptr addrspace(3) %address.local930, i64 256, !dbg !14
  %794 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local931, i32 0) #0, !dbg !14
  %795 = bitcast <4 x i32> %794 to <8 x half>, !dbg !14
  %address.local932 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local922, !dbg !14
  %address.local933 = getelementptr [2 x i8], ptr addrspace(3) %address.local932, i64 %address.local926, !dbg !14
  %address.local934 = getelementptr [2 x i8], ptr addrspace(3) %address.local933, i64 %address.local929, !dbg !14
  %address.local935 = getelementptr i8, ptr addrspace(3) %address.local934, i64 512, !dbg !14
  %796 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local935, i32 0) #0, !dbg !14
  %797 = bitcast <4 x i32> %796 to <8 x half>, !dbg !14
  %address.local936 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local922, !dbg !14
  %address.local937 = getelementptr [2 x i8], ptr addrspace(3) %address.local936, i64 %address.local926, !dbg !14
  %address.local938 = getelementptr [2 x i8], ptr addrspace(3) %address.local937, i64 %address.local929, !dbg !14
  %address.local939 = getelementptr i8, ptr addrspace(3) %address.local938, i64 768, !dbg !14
  %798 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local939, i32 0) #0, !dbg !14
  %799 = bitcast <4 x i32> %798 to <8 x half>, !dbg !14
  %address.local940 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local918, !dbg !14
  %address.local941 = getelementptr i8, ptr addrspace(3) %address.local940, i64 4096, !dbg !14
  %800 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local941, i32 0) #0, !dbg !14
  %801 = bitcast <4 x i32> %800 to <8 x half>, !dbg !14
  %address.local942 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local922, !dbg !14
  %address.local943 = getelementptr [2 x i8], ptr addrspace(3) %address.local942, i64 %address.local926, !dbg !14
  %address.local944 = getelementptr [2 x i8], ptr addrspace(3) %address.local943, i64 %address.local929, !dbg !14
  %address.local945 = getelementptr i8, ptr addrspace(3) %address.local944, i64 4352, !dbg !14
  %802 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local945, i32 0) #0, !dbg !14
  %803 = bitcast <4 x i32> %802 to <8 x half>, !dbg !14
  %address.local946 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local922, !dbg !14
  %address.local947 = getelementptr [2 x i8], ptr addrspace(3) %address.local946, i64 %address.local926, !dbg !14
  %address.local948 = getelementptr [2 x i8], ptr addrspace(3) %address.local947, i64 %address.local929, !dbg !14
  %address.local949 = getelementptr i8, ptr addrspace(3) %address.local948, i64 4608, !dbg !14
  %804 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local949, i32 0) #0, !dbg !14
  %805 = bitcast <4 x i32> %804 to <8 x half>, !dbg !14
  %address.local950 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local922, !dbg !14
  %address.local951 = getelementptr [2 x i8], ptr addrspace(3) %address.local950, i64 %address.local926, !dbg !14
  %address.local952 = getelementptr [2 x i8], ptr addrspace(3) %address.local951, i64 %address.local929, !dbg !14
  %address.local953 = getelementptr i8, ptr addrspace(3) %address.local952, i64 4864, !dbg !14
  %806 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local953, i32 0) #0, !dbg !14
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %807 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local894), !dbg !16
  %dma.buffer.lds1499 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %807), !dbg !16
  %808 = add nuw nsw i64 %52, %15, !dbg !16
  %809 = shl i64 %808, 19, !dbg !16
  %810 = and i64 %809, 16252928, !dbg !16
  %811 = add i64 %810, %240, !dbg !16
  %812 = or disjoint i64 %811, 32, !dbg !16
  %813 = or disjoint i64 %address.local899, %812, !dbg !16
  %.tr = trunc i64 %813 to i32, !dbg !16
  %dma.buffer.offset1497 = shl i32 %.tr, 1, !dbg !16
  %814 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !16
  %815 = insertelement <2 x i64> poison, i64 %814, i64 0, !dbg !16
  %816 = bitcast <2 x i64> %815 to <4 x i32>, !dbg !16
  %dma.buffer.rsrc1498 = shufflevector <4 x i32> %816, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !16
  %817 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1499, i32 %dma.buffer.offset1497, <4 x i32> %dma.buffer.rsrc1498) #10, !dbg !16
  %818 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local906), !dbg !16
  %dma.buffer.lds1496 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %818), !dbg !16
  %819 = or disjoint i64 %811, 32, !dbg !16
  %820 = or disjoint i64 %address.local911, %819, !dbg !16
  %.tr1512 = trunc i64 %820 to i32, !dbg !16
  %dma.buffer.offset1494 = shl i32 %.tr1512, 1, !dbg !16
  %821 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !16
  %822 = insertelement <2 x i64> poison, i64 %821, i64 0, !dbg !16
  %823 = bitcast <2 x i64> %822 to <4 x i32>, !dbg !16
  %dma.buffer.rsrc1495 = shufflevector <4 x i32> %823, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !16
  %824 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1496, i32 %dma.buffer.offset1494, <4 x i32> %dma.buffer.rsrc1495) #10, !dbg !16
  %825 = shufflevector <8 x half> %793, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %826 = extractvalue [2 x <8 x float>] %44, 0, !dbg !34
  %827 = shufflevector <8 x float> %826, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %828 = fmul <4 x float> %827, %548, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %829 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %825, <4 x float> %828) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %830 = shufflevector <8 x half> %793, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %831 = shufflevector <8 x float> %826, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %832 = fmul <4 x float> %831, %548, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %833 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %830, <4 x float> %832) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !34
  %834 = extractvalue [2 x <8 x float>] %44, 1, !dbg !34
  %835 = shufflevector <8 x float> %834, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %836 = fmul <4 x float> %835, %556, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %837 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed447, <4 x half> %825, <4 x float> %836) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !34
  %838 = shufflevector <8 x float> %834, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %839 = fmul <4 x float> %838, %556, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %840 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed447, <4 x half> %830, <4 x float> %839) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %841 = shufflevector <8 x half> %795, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %842 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed433, <4 x half> %841, <4 x float> %829) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %843 = shufflevector <8 x half> %795, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %844 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed433, <4 x half> %843, <4 x float> %833) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %845 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed449, <4 x half> %841, <4 x float> %837) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %846 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed449, <4 x half> %843, <4 x float> %840) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %847 = shufflevector <8 x half> %797, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %848 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed435, <4 x half> %847, <4 x float> %842) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %849 = shufflevector <8 x half> %797, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %850 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed435, <4 x half> %849, <4 x float> %844) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %851 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed451, <4 x half> %847, <4 x float> %845) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %852 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed451, <4 x half> %849, <4 x float> %846) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %853 = shufflevector <8 x half> %799, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %854 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed437, <4 x half> %853, <4 x float> %848) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %855 = shufflevector <8 x half> %799, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %856 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed437, <4 x half> %855, <4 x float> %850) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %857 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed453, <4 x half> %853, <4 x float> %851) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %858 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed453, <4 x half> %855, <4 x float> %852) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %859 = shufflevector <8 x half> %801, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %860 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed439, <4 x half> %859, <4 x float> %854) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %861 = shufflevector <8 x half> %801, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %862 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed439, <4 x half> %861, <4 x float> %856) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %863 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed455, <4 x half> %859, <4 x float> %857) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %864 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed455, <4 x half> %861, <4 x float> %858) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %865 = shufflevector <8 x half> %803, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %866 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed441, <4 x half> %865, <4 x float> %860) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %867 = shufflevector <8 x half> %803, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %868 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed441, <4 x half> %867, <4 x float> %862) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %869 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed457, <4 x half> %865, <4 x float> %863) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %870 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed457, <4 x half> %867, <4 x float> %864) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %871 = shufflevector <8 x half> %805, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %872 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed443, <4 x half> %871, <4 x float> %866) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %873 = shufflevector <8 x half> %805, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %874 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed443, <4 x half> %873, <4 x float> %868) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %875 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed459, <4 x half> %871, <4 x float> %869) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %876 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed459, <4 x half> %873, <4 x float> %870) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %877 = bitcast <4 x i32> %806 to <8 x half>, !dbg !14
  %878 = shufflevector <8 x half> %877, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %879 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed445, <4 x half> %878, <4 x float> %872) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %880 = shufflevector <8 x half> %877, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %881 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed445, <4 x half> %880, <4 x float> %874) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %882 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed461, <4 x half> %878, <4 x float> %875) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %883 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed461, <4 x half> %880, <4 x float> %876) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %884 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local919, i32 0) #0, !dbg !14
  %885 = bitcast <4 x i32> %884 to <8 x half>, !dbg !14
  %886 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local931, i32 0) #0, !dbg !14
  %887 = bitcast <4 x i32> %886 to <8 x half>, !dbg !14
  %888 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local935, i32 0) #0, !dbg !14
  %889 = bitcast <4 x i32> %888 to <8 x half>, !dbg !14
  %890 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local939, i32 0) #0, !dbg !14
  %891 = bitcast <4 x i32> %890 to <8 x half>, !dbg !14
  %892 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local941, i32 0) #0, !dbg !14
  %893 = bitcast <4 x i32> %892 to <8 x half>, !dbg !14
  %894 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local945, i32 0) #0, !dbg !14
  %895 = bitcast <4 x i32> %894 to <8 x half>, !dbg !14
  %896 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local949, i32 0) #0, !dbg !14
  %897 = bitcast <4 x i32> %896 to <8 x half>, !dbg !14
  %898 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local953, i32 0) #0, !dbg !14
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %899 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local894), !dbg !16
  %dma.buffer.lds1493 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %899), !dbg !16
  %900 = or disjoint i64 %811, 64, !dbg !16
  %901 = or disjoint i64 %address.local899, %900, !dbg !16
  %.tr1514 = trunc i64 %901 to i32, !dbg !16
  %dma.buffer.offset1491 = shl i32 %.tr1514, 1, !dbg !16
  %902 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !16
  %903 = insertelement <2 x i64> poison, i64 %902, i64 0, !dbg !16
  %904 = bitcast <2 x i64> %903 to <4 x i32>, !dbg !16
  %dma.buffer.rsrc1492 = shufflevector <4 x i32> %904, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !16
  %905 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1493, i32 %dma.buffer.offset1491, <4 x i32> %dma.buffer.rsrc1492) #10, !dbg !16
  %906 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local906), !dbg !16
  %dma.buffer.lds1490 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %906), !dbg !16
  %907 = or disjoint i64 %811, 64, !dbg !16
  %908 = or disjoint i64 %address.local911, %907, !dbg !16
  %.tr1516 = trunc i64 %908 to i32, !dbg !16
  %dma.buffer.offset1488 = shl i32 %.tr1516, 1, !dbg !16
  %909 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !16
  %910 = insertelement <2 x i64> poison, i64 %909, i64 0, !dbg !16
  %911 = bitcast <2 x i64> %910 to <4 x i32>, !dbg !16
  %dma.buffer.rsrc1489 = shufflevector <4 x i32> %911, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !16
  %912 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1490, i32 %dma.buffer.offset1488, <4 x i32> %dma.buffer.rsrc1489) #10, !dbg !16
  %913 = shufflevector <8 x half> %885, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %914 = extractvalue [2 x <8 x float>] %47, 0, !dbg !34
  %915 = shufflevector <8 x float> %914, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %916 = fmul <4 x float> %915, %548, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %917 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %913, <4 x float> %916) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %918 = shufflevector <8 x half> %885, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %919 = shufflevector <8 x float> %914, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %920 = fmul <4 x float> %919, %548, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %921 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %918, <4 x float> %920) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !34
  %922 = extractvalue [2 x <8 x float>] %47, 1, !dbg !34
  %923 = shufflevector <8 x float> %922, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %924 = fmul <4 x float> %923, %556, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %925 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed447, <4 x half> %913, <4 x float> %924) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !34
  %926 = shufflevector <8 x float> %922, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %927 = fmul <4 x float> %926, %556, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %928 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed447, <4 x half> %918, <4 x float> %927) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %929 = shufflevector <8 x half> %887, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %930 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed433, <4 x half> %929, <4 x float> %917) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %931 = shufflevector <8 x half> %887, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %932 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed433, <4 x half> %931, <4 x float> %921) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %933 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed449, <4 x half> %929, <4 x float> %925) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %934 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed449, <4 x half> %931, <4 x float> %928) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %935 = shufflevector <8 x half> %889, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %936 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed435, <4 x half> %935, <4 x float> %930) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %937 = shufflevector <8 x half> %889, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %938 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed435, <4 x half> %937, <4 x float> %932) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %939 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed451, <4 x half> %935, <4 x float> %933) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %940 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed451, <4 x half> %937, <4 x float> %934) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %941 = shufflevector <8 x half> %891, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %942 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed437, <4 x half> %941, <4 x float> %936) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %943 = shufflevector <8 x half> %891, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %944 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed437, <4 x half> %943, <4 x float> %938) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %945 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed453, <4 x half> %941, <4 x float> %939) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %946 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed453, <4 x half> %943, <4 x float> %940) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %947 = shufflevector <8 x half> %893, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %948 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed439, <4 x half> %947, <4 x float> %942) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %949 = shufflevector <8 x half> %893, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %950 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed439, <4 x half> %949, <4 x float> %944) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %951 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed455, <4 x half> %947, <4 x float> %945) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %952 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed455, <4 x half> %949, <4 x float> %946) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %953 = shufflevector <8 x half> %895, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %954 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed441, <4 x half> %953, <4 x float> %948) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %955 = shufflevector <8 x half> %895, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %956 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed441, <4 x half> %955, <4 x float> %950) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %957 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed457, <4 x half> %953, <4 x float> %951) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %958 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed457, <4 x half> %955, <4 x float> %952) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %959 = shufflevector <8 x half> %897, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %960 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed443, <4 x half> %959, <4 x float> %954) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %961 = shufflevector <8 x half> %897, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %962 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed443, <4 x half> %961, <4 x float> %956) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %963 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed459, <4 x half> %959, <4 x float> %957) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %964 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed459, <4 x half> %961, <4 x float> %958) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %965 = bitcast <4 x i32> %898 to <8 x half>, !dbg !14
  %966 = shufflevector <8 x half> %965, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %967 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed445, <4 x half> %966, <4 x float> %960) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %968 = shufflevector <8 x half> %965, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %969 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed445, <4 x half> %968, <4 x float> %962) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %970 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed461, <4 x half> %966, <4 x float> %963) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %971 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed461, <4 x half> %968, <4 x float> %964) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %972 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local919, i32 0) #0, !dbg !14
  %973 = bitcast <4 x i32> %972 to <8 x half>, !dbg !14
  %974 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local931, i32 0) #0, !dbg !14
  %975 = bitcast <4 x i32> %974 to <8 x half>, !dbg !14
  %976 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local935, i32 0) #0, !dbg !14
  %977 = bitcast <4 x i32> %976 to <8 x half>, !dbg !14
  %978 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local939, i32 0) #0, !dbg !14
  %979 = bitcast <4 x i32> %978 to <8 x half>, !dbg !14
  %980 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local941, i32 0) #0, !dbg !14
  %981 = bitcast <4 x i32> %980 to <8 x half>, !dbg !14
  %982 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local945, i32 0) #0, !dbg !14
  %983 = bitcast <4 x i32> %982 to <8 x half>, !dbg !14
  %984 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local949, i32 0) #0, !dbg !14
  %985 = bitcast <4 x i32> %984 to <8 x half>, !dbg !14
  %986 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local953, i32 0) #0, !dbg !14
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %987 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local894), !dbg !16
  %dma.buffer.lds1487 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %987), !dbg !16
  %988 = or disjoint i64 %811, 96, !dbg !16
  %989 = or disjoint i64 %address.local899, %988, !dbg !16
  %.tr1518 = trunc i64 %989 to i32, !dbg !16
  %dma.buffer.offset1485 = shl i32 %.tr1518, 1, !dbg !16
  %990 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !16
  %991 = insertelement <2 x i64> poison, i64 %990, i64 0, !dbg !16
  %992 = bitcast <2 x i64> %991 to <4 x i32>, !dbg !16
  %dma.buffer.rsrc1486 = shufflevector <4 x i32> %992, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !16
  %993 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1487, i32 %dma.buffer.offset1485, <4 x i32> %dma.buffer.rsrc1486) #10, !dbg !16
  %994 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local906), !dbg !16
  %dma.buffer.lds1484 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %994), !dbg !16
  %995 = or disjoint i64 %811, 96, !dbg !16
  %996 = or disjoint i64 %address.local911, %995, !dbg !16
  %.tr1520 = trunc i64 %996 to i32, !dbg !16
  %dma.buffer.offset1482 = shl i32 %.tr1520, 1, !dbg !16
  %997 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !16
  %998 = insertelement <2 x i64> poison, i64 %997, i64 0, !dbg !16
  %999 = bitcast <2 x i64> %998 to <4 x i32>, !dbg !16
  %dma.buffer.rsrc1483 = shufflevector <4 x i32> %999, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !16
  %1000 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1484, i32 %dma.buffer.offset1482, <4 x i32> %dma.buffer.rsrc1483) #10, !dbg !16
  %1001 = shufflevector <8 x half> %973, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1002 = extractvalue [2 x <8 x float>] %48, 0, !dbg !34
  %1003 = shufflevector <8 x float> %1002, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %1004 = fmul <4 x float> %1003, %548, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1005 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1001, <4 x float> %1004) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1006 = shufflevector <8 x half> %973, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1007 = shufflevector <8 x float> %1002, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %1008 = fmul <4 x float> %1007, %548, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1009 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1006, <4 x float> %1008) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !34
  %1010 = extractvalue [2 x <8 x float>] %48, 1, !dbg !34
  %1011 = shufflevector <8 x float> %1010, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %1012 = fmul <4 x float> %1011, %556, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1013 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed447, <4 x half> %1001, <4 x float> %1012) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !34
  %1014 = shufflevector <8 x float> %1010, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %1015 = fmul <4 x float> %1014, %556, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1016 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed447, <4 x half> %1006, <4 x float> %1015) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1017 = shufflevector <8 x half> %975, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1018 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed433, <4 x half> %1017, <4 x float> %1005) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1019 = shufflevector <8 x half> %975, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1020 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed433, <4 x half> %1019, <4 x float> %1009) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1021 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed449, <4 x half> %1017, <4 x float> %1013) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1022 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed449, <4 x half> %1019, <4 x float> %1016) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1023 = shufflevector <8 x half> %977, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1024 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed435, <4 x half> %1023, <4 x float> %1018) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1025 = shufflevector <8 x half> %977, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1026 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed435, <4 x half> %1025, <4 x float> %1020) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1027 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed451, <4 x half> %1023, <4 x float> %1021) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1028 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed451, <4 x half> %1025, <4 x float> %1022) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1029 = shufflevector <8 x half> %979, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1030 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed437, <4 x half> %1029, <4 x float> %1024) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1031 = shufflevector <8 x half> %979, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1032 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed437, <4 x half> %1031, <4 x float> %1026) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1033 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed453, <4 x half> %1029, <4 x float> %1027) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1034 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed453, <4 x half> %1031, <4 x float> %1028) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1035 = shufflevector <8 x half> %981, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1036 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed439, <4 x half> %1035, <4 x float> %1030) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1037 = shufflevector <8 x half> %981, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1038 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed439, <4 x half> %1037, <4 x float> %1032) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1039 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed455, <4 x half> %1035, <4 x float> %1033) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1040 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed455, <4 x half> %1037, <4 x float> %1034) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1041 = shufflevector <8 x half> %983, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1042 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed441, <4 x half> %1041, <4 x float> %1036) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1043 = shufflevector <8 x half> %983, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1044 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed441, <4 x half> %1043, <4 x float> %1038) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1045 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed457, <4 x half> %1041, <4 x float> %1039) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1046 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed457, <4 x half> %1043, <4 x float> %1040) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1047 = shufflevector <8 x half> %985, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1048 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed443, <4 x half> %1047, <4 x float> %1042) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1049 = shufflevector <8 x half> %985, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1050 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed443, <4 x half> %1049, <4 x float> %1044) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1051 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed459, <4 x half> %1047, <4 x float> %1045) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1052 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed459, <4 x half> %1049, <4 x float> %1046) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1053 = bitcast <4 x i32> %986 to <8 x half>, !dbg !14
  %1054 = shufflevector <8 x half> %1053, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1055 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed445, <4 x half> %1054, <4 x float> %1048) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1056 = shufflevector <8 x half> %1053, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1057 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed445, <4 x half> %1056, <4 x float> %1050) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1058 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed461, <4 x half> %1054, <4 x float> %1051) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1059 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed461, <4 x half> %1056, <4 x float> %1052) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %1060 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local919, i32 0) #0, !dbg !14
  %1061 = bitcast <4 x i32> %1060 to <8 x half>, !dbg !14
  %1062 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local931, i32 0) #0, !dbg !14
  %1063 = bitcast <4 x i32> %1062 to <8 x half>, !dbg !14
  %1064 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local935, i32 0) #0, !dbg !14
  %1065 = bitcast <4 x i32> %1064 to <8 x half>, !dbg !14
  %1066 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local939, i32 0) #0, !dbg !14
  %1067 = bitcast <4 x i32> %1066 to <8 x half>, !dbg !14
  %1068 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local941, i32 0) #0, !dbg !14
  %1069 = bitcast <4 x i32> %1068 to <8 x half>, !dbg !14
  %1070 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local945, i32 0) #0, !dbg !14
  %1071 = bitcast <4 x i32> %1070 to <8 x half>, !dbg !14
  %1072 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local949, i32 0) #0, !dbg !14
  %1073 = bitcast <4 x i32> %1072 to <8 x half>, !dbg !14
  %1074 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local953, i32 0) #0, !dbg !14
  %1075 = shufflevector <8 x half> %1061, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1076 = extractvalue [2 x <8 x float>] %49, 0, !dbg !34
  %1077 = shufflevector <8 x float> %1076, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %1078 = fmul <4 x float> %1077, %548, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1079 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1075, <4 x float> %1078) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1080 = shufflevector <8 x half> %1061, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1081 = shufflevector <8 x float> %1076, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %1082 = fmul <4 x float> %1081, %548, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1083 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1080, <4 x float> %1082) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !34
  %1084 = extractvalue [2 x <8 x float>] %49, 1, !dbg !34
  %1085 = shufflevector <8 x float> %1084, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %1086 = fmul <4 x float> %1085, %556, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1087 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed447, <4 x half> %1075, <4 x float> %1086) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !34
  %1088 = shufflevector <8 x float> %1084, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %1089 = fmul <4 x float> %1088, %556, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1090 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed447, <4 x half> %1080, <4 x float> %1089) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1091 = shufflevector <8 x half> %1063, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1092 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed433, <4 x half> %1091, <4 x float> %1079) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1093 = shufflevector <8 x half> %1063, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1094 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed433, <4 x half> %1093, <4 x float> %1083) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1095 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed449, <4 x half> %1091, <4 x float> %1087) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1096 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed449, <4 x half> %1093, <4 x float> %1090) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1097 = shufflevector <8 x half> %1065, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1098 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed435, <4 x half> %1097, <4 x float> %1092) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1099 = shufflevector <8 x half> %1065, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1100 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed435, <4 x half> %1099, <4 x float> %1094) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1101 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed451, <4 x half> %1097, <4 x float> %1095) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1102 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed451, <4 x half> %1099, <4 x float> %1096) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1103 = shufflevector <8 x half> %1067, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1104 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed437, <4 x half> %1103, <4 x float> %1098) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1105 = shufflevector <8 x half> %1067, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1106 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed437, <4 x half> %1105, <4 x float> %1100) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1107 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed453, <4 x half> %1103, <4 x float> %1101) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1108 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed453, <4 x half> %1105, <4 x float> %1102) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1109 = shufflevector <8 x half> %1069, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1110 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed439, <4 x half> %1109, <4 x float> %1104) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1111 = shufflevector <8 x half> %1069, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1112 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed439, <4 x half> %1111, <4 x float> %1106) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1113 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed455, <4 x half> %1109, <4 x float> %1107) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1114 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed455, <4 x half> %1111, <4 x float> %1108) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1115 = shufflevector <8 x half> %1071, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1116 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed441, <4 x half> %1115, <4 x float> %1110) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1117 = shufflevector <8 x half> %1071, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1118 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed441, <4 x half> %1117, <4 x float> %1112) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1119 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed457, <4 x half> %1115, <4 x float> %1113) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1120 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed457, <4 x half> %1117, <4 x float> %1114) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1121 = shufflevector <8 x half> %1073, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1122 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed443, <4 x half> %1121, <4 x float> %1116) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1123 = shufflevector <8 x half> %1073, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1124 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed443, <4 x half> %1123, <4 x float> %1118) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1125 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed459, <4 x half> %1121, <4 x float> %1119) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1126 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed459, <4 x half> %1123, <4 x float> %1120) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1127 = bitcast <4 x i32> %1074 to <8 x half>, !dbg !14
  %1128 = shufflevector <8 x half> %1127, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1129 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed445, <4 x half> %1128, <4 x float> %1122) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1130 = shufflevector <8 x half> %1127, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1131 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed445, <4 x half> %1130, <4 x float> %1124) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1132 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed461, <4 x half> %1128, <4 x float> %1125) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1133 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed461, <4 x half> %1130, <4 x float> %1126) #9, !dbg !14, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !8
  %1134 = shufflevector <4 x float> %879, <4 x float> %881, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %1135 = insertvalue [2 x <8 x float>] poison, <8 x float> %1134, 0, !dbg !8
  %1136 = shufflevector <4 x float> %882, <4 x float> %883, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %1137 = insertvalue [2 x <8 x float>] %1135, <8 x float> %1136, 1, !dbg !8
  %1138 = insertvalue [2 x <1 x float>] poison, <1 x float> %512, 0, !dbg !8
  %1139 = insertvalue [2 x <1 x float>] %1138, <1 x float> %525, 1, !dbg !8
  %1140 = shufflevector <4 x float> %967, <4 x float> %969, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %1141 = insertvalue [2 x <8 x float>] poison, <8 x float> %1140, 0, !dbg !8
  %1142 = shufflevector <4 x float> %970, <4 x float> %971, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %1143 = insertvalue [2 x <8 x float>] %1141, <8 x float> %1142, 1, !dbg !8
  %1144 = shufflevector <4 x float> %1055, <4 x float> %1057, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %1145 = insertvalue [2 x <8 x float>] poison, <8 x float> %1144, 0, !dbg !8
  %1146 = shufflevector <4 x float> %1058, <4 x float> %1059, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %1147 = insertvalue [2 x <8 x float>] %1145, <8 x float> %1146, 1, !dbg !8
  %1148 = shufflevector <4 x float> %1129, <4 x float> %1131, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %1149 = insertvalue [2 x <8 x float>] poison, <8 x float> %1148, 0, !dbg !8
  %1150 = shufflevector <4 x float> %1132, <4 x float> %1133, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %1151 = insertvalue [2 x <8 x float>] %1149, <8 x float> %1150, 1, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %1152 = add nuw nsw i64 %43, 1, !dbg !8
  br label %42, !dbg !8

1153:                                             ; preds = %42
  %1154 = and i32 %23, 16, !dbg !8
  %.not36 = icmp eq i32 %1154, 0, !dbg !8
  %1155 = and i32 %23, 32, !dbg !8
  %.not37 = icmp eq i32 %1155, 0, !dbg !8
  %1156 = extractvalue [2 x <8 x float>] %49, 0, !dbg !8
  %1157 = extractelement <8 x float> %1156, i64 0, !dbg !8
  %1158 = extractelement <8 x float> %1156, i64 1, !dbg !8
  %1159 = extractelement <8 x float> %1156, i64 2, !dbg !8
  %1160 = extractelement <8 x float> %1156, i64 3, !dbg !8
  %1161 = select i1 %.not36, float %1158, float %1157, !dbg !8
  %1162 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !8
  %1163 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1162), !dbg !8
  %1164 = add nuw nsw i32 %1163, 64, !dbg !8
  %1165 = xor i32 %1163, 16, !dbg !8
  %1166 = and i32 %1164, 192, !dbg !8
  %1167 = icmp samesign ult i32 %1165, %1166, !dbg !8
  %1168 = select i1 %1167, i32 %1165, i32 %1163, !dbg !8
  %1169 = shl nuw nsw i32 %1168, 2, !dbg !8
  %1170 = bitcast float %1161 to i32, !dbg !8
  %1171 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1169, i32 %1170), !dbg !8
  %1172 = bitcast i32 %1171 to float, !dbg !8
  %1173 = select i1 %.not36, float %1157, float %1172, !dbg !8
  %1174 = select i1 %.not36, float %1172, float %1158, !dbg !8
  %1175 = select i1 %.not36, float %1160, float %1159, !dbg !8
  %1176 = bitcast float %1175 to i32, !dbg !8
  %1177 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1169, i32 %1176), !dbg !8
  %1178 = bitcast i32 %1177 to float, !dbg !8
  %1179 = select i1 %.not36, float %1159, float %1178, !dbg !8
  %1180 = select i1 %.not36, float %1178, float %1160, !dbg !8
  %1181 = select i1 %.not37, float %1179, float %1173, !dbg !8
  %1182 = xor i32 %1163, 32, !dbg !8
  %1183 = icmp samesign ult i32 %1182, %1166, !dbg !8
  %1184 = select i1 %1183, i32 %1182, i32 %1163, !dbg !8
  %1185 = shl nuw nsw i32 %1184, 2, !dbg !8
  %1186 = bitcast float %1181 to i32, !dbg !8
  %1187 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1185, i32 %1186), !dbg !8
  %1188 = bitcast i32 %1187 to float, !dbg !8
  %1189 = select i1 %.not37, float %1173, float %1188, !dbg !8
  %1190 = select i1 %.not37, float %1188, float %1179, !dbg !8
  %1191 = select i1 %.not37, float %1180, float %1174, !dbg !8
  %1192 = bitcast float %1191 to i32, !dbg !8
  %1193 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1185, i32 %1192), !dbg !8
  %1194 = bitcast i32 %1193 to float, !dbg !8
  %1195 = select i1 %.not37, float %1174, float %1194, !dbg !8
  %1196 = select i1 %.not37, float %1194, float %1180, !dbg !8
  %1197 = extractvalue [2 x <8 x float>] %48, 0, !dbg !8
  %1198 = extractelement <8 x float> %1197, i64 0, !dbg !8
  %1199 = extractelement <8 x float> %1197, i64 1, !dbg !8
  %1200 = extractelement <8 x float> %1197, i64 2, !dbg !8
  %1201 = extractelement <8 x float> %1197, i64 3, !dbg !8
  %1202 = select i1 %.not36, float %1199, float %1198, !dbg !8
  %1203 = bitcast float %1202 to i32, !dbg !8
  %1204 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1169, i32 %1203), !dbg !8
  %1205 = bitcast i32 %1204 to float, !dbg !8
  %1206 = select i1 %.not36, float %1198, float %1205, !dbg !8
  %1207 = select i1 %.not36, float %1205, float %1199, !dbg !8
  %1208 = select i1 %.not36, float %1201, float %1200, !dbg !8
  %1209 = bitcast float %1208 to i32, !dbg !8
  %1210 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1169, i32 %1209), !dbg !8
  %1211 = bitcast i32 %1210 to float, !dbg !8
  %1212 = select i1 %.not36, float %1200, float %1211, !dbg !8
  %1213 = select i1 %.not36, float %1211, float %1201, !dbg !8
  %1214 = select i1 %.not37, float %1212, float %1206, !dbg !8
  %1215 = bitcast float %1214 to i32, !dbg !8
  %1216 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1185, i32 %1215), !dbg !8
  %1217 = bitcast i32 %1216 to float, !dbg !8
  %1218 = select i1 %.not37, float %1206, float %1217, !dbg !8
  %1219 = select i1 %.not37, float %1217, float %1212, !dbg !8
  %1220 = select i1 %.not37, float %1213, float %1207, !dbg !8
  %1221 = bitcast float %1220 to i32, !dbg !8
  %1222 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1185, i32 %1221), !dbg !8
  %1223 = bitcast i32 %1222 to float, !dbg !8
  %1224 = select i1 %.not37, float %1207, float %1223, !dbg !8
  %1225 = select i1 %.not37, float %1223, float %1213, !dbg !8
  %1226 = extractvalue [2 x <8 x float>] %47, 0, !dbg !8
  %1227 = extractelement <8 x float> %1226, i64 0, !dbg !8
  %1228 = extractelement <8 x float> %1226, i64 1, !dbg !8
  %1229 = extractelement <8 x float> %1226, i64 2, !dbg !8
  %1230 = extractelement <8 x float> %1226, i64 3, !dbg !8
  %1231 = select i1 %.not36, float %1228, float %1227, !dbg !8
  %1232 = bitcast float %1231 to i32, !dbg !8
  %1233 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1169, i32 %1232), !dbg !8
  %1234 = bitcast i32 %1233 to float, !dbg !8
  %1235 = select i1 %.not36, float %1227, float %1234, !dbg !8
  %1236 = select i1 %.not36, float %1234, float %1228, !dbg !8
  %1237 = select i1 %.not36, float %1230, float %1229, !dbg !8
  %1238 = bitcast float %1237 to i32, !dbg !8
  %1239 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1169, i32 %1238), !dbg !8
  %1240 = bitcast i32 %1239 to float, !dbg !8
  %1241 = select i1 %.not36, float %1229, float %1240, !dbg !8
  %1242 = select i1 %.not36, float %1240, float %1230, !dbg !8
  %1243 = select i1 %.not37, float %1241, float %1235, !dbg !8
  %1244 = bitcast float %1243 to i32, !dbg !8
  %1245 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1185, i32 %1244), !dbg !8
  %1246 = bitcast i32 %1245 to float, !dbg !8
  %1247 = select i1 %.not37, float %1235, float %1246, !dbg !8
  %1248 = select i1 %.not37, float %1246, float %1241, !dbg !8
  %1249 = select i1 %.not37, float %1242, float %1236, !dbg !8
  %1250 = bitcast float %1249 to i32, !dbg !8
  %1251 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1185, i32 %1250), !dbg !8
  %1252 = bitcast i32 %1251 to float, !dbg !8
  %1253 = select i1 %.not37, float %1236, float %1252, !dbg !8
  %1254 = select i1 %.not37, float %1252, float %1242, !dbg !8
  %1255 = extractvalue [2 x <8 x float>] %44, 0, !dbg !8
  %1256 = extractelement <8 x float> %1255, i64 0, !dbg !8
  %1257 = extractelement <8 x float> %1255, i64 1, !dbg !8
  %1258 = extractelement <8 x float> %1255, i64 2, !dbg !8
  %1259 = extractelement <8 x float> %1255, i64 3, !dbg !8
  %1260 = select i1 %.not36, float %1257, float %1256, !dbg !8
  %1261 = bitcast float %1260 to i32, !dbg !8
  %1262 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1169, i32 %1261), !dbg !8
  %1263 = bitcast i32 %1262 to float, !dbg !8
  %1264 = select i1 %.not36, float %1256, float %1263, !dbg !8
  %1265 = select i1 %.not36, float %1263, float %1257, !dbg !8
  %1266 = select i1 %.not36, float %1259, float %1258, !dbg !8
  %1267 = bitcast float %1266 to i32, !dbg !8
  %1268 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1169, i32 %1267), !dbg !8
  %1269 = bitcast i32 %1268 to float, !dbg !8
  %1270 = select i1 %.not36, float %1258, float %1269, !dbg !8
  %1271 = select i1 %.not36, float %1269, float %1259, !dbg !8
  %1272 = select i1 %.not37, float %1270, float %1264, !dbg !8
  %1273 = bitcast float %1272 to i32, !dbg !8
  %1274 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1185, i32 %1273), !dbg !8
  %1275 = select i1 %.not37, float %1271, float %1265, !dbg !8
  %1276 = bitcast float %1275 to i32, !dbg !8
  %1277 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1185, i32 %1276), !dbg !8
  %1278 = extractvalue [2 x <1 x float>] %45, 0, !dbg !19
  %1279 = shufflevector <1 x float> %1278, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %1280 = extractelement <8 x float> %1156, i64 4, !dbg !8
  %1281 = extractelement <8 x float> %1156, i64 5, !dbg !8
  %1282 = extractelement <8 x float> %1156, i64 6, !dbg !8
  %1283 = extractelement <8 x float> %1156, i64 7, !dbg !8
  %1284 = select i1 %.not36, float %1281, float %1280, !dbg !8
  %1285 = bitcast float %1284 to i32, !dbg !8
  %1286 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1169, i32 %1285), !dbg !8
  %1287 = bitcast i32 %1286 to float, !dbg !8
  %1288 = select i1 %.not36, float %1280, float %1287, !dbg !8
  %1289 = select i1 %.not36, float %1287, float %1281, !dbg !8
  %1290 = select i1 %.not36, float %1283, float %1282, !dbg !8
  %1291 = bitcast float %1290 to i32, !dbg !8
  %1292 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1169, i32 %1291), !dbg !8
  %1293 = bitcast i32 %1292 to float, !dbg !8
  %1294 = select i1 %.not36, float %1282, float %1293, !dbg !8
  %1295 = select i1 %.not36, float %1293, float %1283, !dbg !8
  %1296 = select i1 %.not37, float %1294, float %1288, !dbg !8
  %1297 = bitcast float %1296 to i32, !dbg !8
  %1298 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1185, i32 %1297), !dbg !8
  %1299 = bitcast i32 %1298 to float, !dbg !8
  %1300 = select i1 %.not37, float %1288, float %1299, !dbg !8
  %1301 = select i1 %.not37, float %1299, float %1294, !dbg !8
  %1302 = select i1 %.not37, float %1295, float %1289, !dbg !8
  %1303 = bitcast float %1302 to i32, !dbg !8
  %1304 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1185, i32 %1303), !dbg !8
  %1305 = bitcast i32 %1304 to float, !dbg !8
  %1306 = select i1 %.not37, float %1289, float %1305, !dbg !8
  %1307 = select i1 %.not37, float %1305, float %1295, !dbg !8
  %1308 = extractelement <8 x float> %1197, i64 4, !dbg !8
  %1309 = extractelement <8 x float> %1197, i64 5, !dbg !8
  %1310 = extractelement <8 x float> %1197, i64 6, !dbg !8
  %1311 = extractelement <8 x float> %1197, i64 7, !dbg !8
  %1312 = select i1 %.not36, float %1309, float %1308, !dbg !8
  %1313 = bitcast float %1312 to i32, !dbg !8
  %1314 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1169, i32 %1313), !dbg !8
  %1315 = bitcast i32 %1314 to float, !dbg !8
  %1316 = select i1 %.not36, float %1308, float %1315, !dbg !8
  %1317 = select i1 %.not36, float %1315, float %1309, !dbg !8
  %1318 = select i1 %.not36, float %1311, float %1310, !dbg !8
  %1319 = bitcast float %1318 to i32, !dbg !8
  %1320 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1169, i32 %1319), !dbg !8
  %1321 = bitcast i32 %1320 to float, !dbg !8
  %1322 = select i1 %.not36, float %1310, float %1321, !dbg !8
  %1323 = select i1 %.not36, float %1321, float %1311, !dbg !8
  %1324 = select i1 %.not37, float %1322, float %1316, !dbg !8
  %1325 = bitcast float %1324 to i32, !dbg !8
  %1326 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1185, i32 %1325), !dbg !8
  %1327 = bitcast i32 %1326 to float, !dbg !8
  %1328 = select i1 %.not37, float %1316, float %1327, !dbg !8
  %1329 = select i1 %.not37, float %1327, float %1322, !dbg !8
  %1330 = select i1 %.not37, float %1323, float %1317, !dbg !8
  %1331 = bitcast float %1330 to i32, !dbg !8
  %1332 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1185, i32 %1331), !dbg !8
  %1333 = bitcast i32 %1332 to float, !dbg !8
  %1334 = select i1 %.not37, float %1317, float %1333, !dbg !8
  %1335 = select i1 %.not37, float %1333, float %1323, !dbg !8
  %1336 = extractelement <8 x float> %1226, i64 4, !dbg !8
  %1337 = extractelement <8 x float> %1226, i64 5, !dbg !8
  %1338 = extractelement <8 x float> %1226, i64 6, !dbg !8
  %1339 = extractelement <8 x float> %1226, i64 7, !dbg !8
  %1340 = select i1 %.not36, float %1337, float %1336, !dbg !8
  %1341 = bitcast float %1340 to i32, !dbg !8
  %1342 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1169, i32 %1341), !dbg !8
  %1343 = bitcast i32 %1342 to float, !dbg !8
  %1344 = select i1 %.not36, float %1336, float %1343, !dbg !8
  %1345 = select i1 %.not36, float %1343, float %1337, !dbg !8
  %1346 = select i1 %.not36, float %1339, float %1338, !dbg !8
  %1347 = bitcast float %1346 to i32, !dbg !8
  %1348 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1169, i32 %1347), !dbg !8
  %1349 = bitcast i32 %1348 to float, !dbg !8
  %1350 = select i1 %.not36, float %1338, float %1349, !dbg !8
  %1351 = select i1 %.not36, float %1349, float %1339, !dbg !8
  %1352 = select i1 %.not37, float %1350, float %1344, !dbg !8
  %1353 = bitcast float %1352 to i32, !dbg !8
  %1354 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1185, i32 %1353), !dbg !8
  %1355 = bitcast i32 %1354 to float, !dbg !8
  %1356 = select i1 %.not37, float %1344, float %1355, !dbg !8
  %1357 = select i1 %.not37, float %1355, float %1350, !dbg !8
  %1358 = select i1 %.not37, float %1351, float %1345, !dbg !8
  %1359 = bitcast float %1358 to i32, !dbg !8
  %1360 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1185, i32 %1359), !dbg !8
  %1361 = bitcast i32 %1360 to float, !dbg !8
  %1362 = select i1 %.not37, float %1345, float %1361, !dbg !8
  %1363 = select i1 %.not37, float %1361, float %1351, !dbg !8
  %1364 = extractelement <8 x float> %1255, i64 4, !dbg !8
  %1365 = extractelement <8 x float> %1255, i64 5, !dbg !8
  %1366 = extractelement <8 x float> %1255, i64 6, !dbg !8
  %1367 = extractelement <8 x float> %1255, i64 7, !dbg !8
  %1368 = select i1 %.not36, float %1365, float %1364, !dbg !8
  %1369 = bitcast float %1368 to i32, !dbg !8
  %1370 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1169, i32 %1369), !dbg !8
  %1371 = bitcast i32 %1370 to float, !dbg !8
  %1372 = select i1 %.not36, float %1364, float %1371, !dbg !8
  %1373 = select i1 %.not36, float %1371, float %1365, !dbg !8
  %1374 = select i1 %.not36, float %1367, float %1366, !dbg !8
  %1375 = bitcast float %1374 to i32, !dbg !8
  %1376 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1169, i32 %1375), !dbg !8
  %1377 = bitcast i32 %1376 to float, !dbg !8
  %1378 = select i1 %.not36, float %1366, float %1377, !dbg !8
  %1379 = select i1 %.not36, float %1377, float %1367, !dbg !8
  %1380 = select i1 %.not37, float %1378, float %1372, !dbg !8
  %1381 = bitcast float %1380 to i32, !dbg !8
  %1382 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1185, i32 %1381), !dbg !8
  %1383 = select i1 %.not37, float %1379, float %1373, !dbg !8
  %1384 = bitcast float %1383 to i32, !dbg !8
  %1385 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1185, i32 %1384), !dbg !8
  %1386 = extractvalue [2 x <8 x float>] %49, 1, !dbg !8
  %1387 = extractelement <8 x float> %1386, i64 0, !dbg !8
  %1388 = extractelement <8 x float> %1386, i64 1, !dbg !8
  %1389 = extractelement <8 x float> %1386, i64 2, !dbg !8
  %1390 = extractelement <8 x float> %1386, i64 3, !dbg !8
  %1391 = select i1 %.not36, float %1388, float %1387, !dbg !8
  %1392 = bitcast float %1391 to i32, !dbg !8
  %1393 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1169, i32 %1392), !dbg !8
  %1394 = bitcast i32 %1393 to float, !dbg !8
  %1395 = select i1 %.not36, float %1387, float %1394, !dbg !8
  %1396 = select i1 %.not36, float %1394, float %1388, !dbg !8
  %1397 = select i1 %.not36, float %1390, float %1389, !dbg !8
  %1398 = bitcast float %1397 to i32, !dbg !8
  %1399 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1169, i32 %1398), !dbg !8
  %1400 = bitcast i32 %1399 to float, !dbg !8
  %1401 = select i1 %.not36, float %1389, float %1400, !dbg !8
  %1402 = select i1 %.not36, float %1400, float %1390, !dbg !8
  %1403 = select i1 %.not37, float %1401, float %1395, !dbg !8
  %1404 = bitcast float %1403 to i32, !dbg !8
  %1405 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1185, i32 %1404), !dbg !8
  %1406 = bitcast i32 %1405 to float, !dbg !8
  %1407 = select i1 %.not37, float %1395, float %1406, !dbg !8
  %1408 = select i1 %.not37, float %1406, float %1401, !dbg !8
  %1409 = select i1 %.not37, float %1402, float %1396, !dbg !8
  %1410 = bitcast float %1409 to i32, !dbg !8
  %1411 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1185, i32 %1410), !dbg !8
  %1412 = bitcast i32 %1411 to float, !dbg !8
  %1413 = select i1 %.not37, float %1396, float %1412, !dbg !8
  %1414 = select i1 %.not37, float %1412, float %1402, !dbg !8
  %1415 = extractvalue [2 x <8 x float>] %48, 1, !dbg !8
  %1416 = extractelement <8 x float> %1415, i64 0, !dbg !8
  %1417 = extractelement <8 x float> %1415, i64 1, !dbg !8
  %1418 = extractelement <8 x float> %1415, i64 2, !dbg !8
  %1419 = extractelement <8 x float> %1415, i64 3, !dbg !8
  %1420 = select i1 %.not36, float %1417, float %1416, !dbg !8
  %1421 = bitcast float %1420 to i32, !dbg !8
  %1422 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1169, i32 %1421), !dbg !8
  %1423 = bitcast i32 %1422 to float, !dbg !8
  %1424 = select i1 %.not36, float %1416, float %1423, !dbg !8
  %1425 = select i1 %.not36, float %1423, float %1417, !dbg !8
  %1426 = select i1 %.not36, float %1419, float %1418, !dbg !8
  %1427 = bitcast float %1426 to i32, !dbg !8
  %1428 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1169, i32 %1427), !dbg !8
  %1429 = bitcast i32 %1428 to float, !dbg !8
  %1430 = select i1 %.not36, float %1418, float %1429, !dbg !8
  %1431 = select i1 %.not36, float %1429, float %1419, !dbg !8
  %1432 = select i1 %.not37, float %1430, float %1424, !dbg !8
  %1433 = bitcast float %1432 to i32, !dbg !8
  %1434 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1185, i32 %1433), !dbg !8
  %1435 = bitcast i32 %1434 to float, !dbg !8
  %1436 = select i1 %.not37, float %1424, float %1435, !dbg !8
  %1437 = select i1 %.not37, float %1435, float %1430, !dbg !8
  %1438 = select i1 %.not37, float %1431, float %1425, !dbg !8
  %1439 = bitcast float %1438 to i32, !dbg !8
  %1440 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1185, i32 %1439), !dbg !8
  %1441 = bitcast i32 %1440 to float, !dbg !8
  %1442 = select i1 %.not37, float %1425, float %1441, !dbg !8
  %1443 = select i1 %.not37, float %1441, float %1431, !dbg !8
  %1444 = extractvalue [2 x <8 x float>] %47, 1, !dbg !8
  %1445 = extractelement <8 x float> %1444, i64 0, !dbg !8
  %1446 = extractelement <8 x float> %1444, i64 1, !dbg !8
  %1447 = extractelement <8 x float> %1444, i64 2, !dbg !8
  %1448 = extractelement <8 x float> %1444, i64 3, !dbg !8
  %1449 = select i1 %.not36, float %1446, float %1445, !dbg !8
  %1450 = bitcast float %1449 to i32, !dbg !8
  %1451 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1169, i32 %1450), !dbg !8
  %1452 = bitcast i32 %1451 to float, !dbg !8
  %1453 = select i1 %.not36, float %1445, float %1452, !dbg !8
  %1454 = select i1 %.not36, float %1452, float %1446, !dbg !8
  %1455 = select i1 %.not36, float %1448, float %1447, !dbg !8
  %1456 = bitcast float %1455 to i32, !dbg !8
  %1457 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1169, i32 %1456), !dbg !8
  %1458 = bitcast i32 %1457 to float, !dbg !8
  %1459 = select i1 %.not36, float %1447, float %1458, !dbg !8
  %1460 = select i1 %.not36, float %1458, float %1448, !dbg !8
  %1461 = select i1 %.not37, float %1459, float %1453, !dbg !8
  %1462 = bitcast float %1461 to i32, !dbg !8
  %1463 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1185, i32 %1462), !dbg !8
  %1464 = bitcast i32 %1463 to float, !dbg !8
  %1465 = select i1 %.not37, float %1453, float %1464, !dbg !8
  %1466 = select i1 %.not37, float %1464, float %1459, !dbg !8
  %1467 = select i1 %.not37, float %1460, float %1454, !dbg !8
  %1468 = bitcast float %1467 to i32, !dbg !8
  %1469 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1185, i32 %1468), !dbg !8
  %1470 = bitcast i32 %1469 to float, !dbg !8
  %1471 = select i1 %.not37, float %1454, float %1470, !dbg !8
  %1472 = select i1 %.not37, float %1470, float %1460, !dbg !8
  %1473 = extractvalue [2 x <8 x float>] %44, 1, !dbg !8
  %1474 = extractelement <8 x float> %1473, i64 0, !dbg !8
  %1475 = extractelement <8 x float> %1473, i64 1, !dbg !8
  %1476 = extractelement <8 x float> %1473, i64 2, !dbg !8
  %1477 = extractelement <8 x float> %1473, i64 3, !dbg !8
  %1478 = select i1 %.not36, float %1475, float %1474, !dbg !8
  %1479 = bitcast float %1478 to i32, !dbg !8
  %1480 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1169, i32 %1479), !dbg !8
  %1481 = bitcast i32 %1480 to float, !dbg !8
  %1482 = select i1 %.not36, float %1474, float %1481, !dbg !8
  %1483 = select i1 %.not36, float %1481, float %1475, !dbg !8
  %1484 = select i1 %.not36, float %1477, float %1476, !dbg !8
  %1485 = bitcast float %1484 to i32, !dbg !8
  %1486 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1169, i32 %1485), !dbg !8
  %1487 = bitcast i32 %1486 to float, !dbg !8
  %1488 = select i1 %.not36, float %1476, float %1487, !dbg !8
  %1489 = select i1 %.not36, float %1487, float %1477, !dbg !8
  %1490 = select i1 %.not37, float %1488, float %1482, !dbg !8
  %1491 = bitcast float %1490 to i32, !dbg !8
  %1492 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1185, i32 %1491), !dbg !8
  %1493 = select i1 %.not37, float %1489, float %1483, !dbg !8
  %1494 = bitcast float %1493 to i32, !dbg !8
  %1495 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1185, i32 %1494), !dbg !8
  %1496 = extractvalue [2 x <1 x float>] %45, 1, !dbg !19
  %1497 = shufflevector <1 x float> %1496, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %1498 = extractelement <8 x float> %1386, i64 4, !dbg !8
  %1499 = extractelement <8 x float> %1386, i64 5, !dbg !8
  %1500 = extractelement <8 x float> %1386, i64 6, !dbg !8
  %1501 = extractelement <8 x float> %1386, i64 7, !dbg !8
  %1502 = select i1 %.not36, float %1499, float %1498, !dbg !8
  %1503 = bitcast float %1502 to i32, !dbg !8
  %1504 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1169, i32 %1503), !dbg !8
  %1505 = bitcast i32 %1504 to float, !dbg !8
  %1506 = select i1 %.not36, float %1498, float %1505, !dbg !8
  %1507 = select i1 %.not36, float %1505, float %1499, !dbg !8
  %1508 = select i1 %.not36, float %1501, float %1500, !dbg !8
  %1509 = bitcast float %1508 to i32, !dbg !8
  %1510 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1169, i32 %1509), !dbg !8
  %1511 = bitcast i32 %1510 to float, !dbg !8
  %1512 = select i1 %.not36, float %1500, float %1511, !dbg !8
  %1513 = select i1 %.not36, float %1511, float %1501, !dbg !8
  %1514 = select i1 %.not37, float %1512, float %1506, !dbg !8
  %1515 = bitcast float %1514 to i32, !dbg !8
  %1516 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1185, i32 %1515), !dbg !8
  %1517 = bitcast i32 %1516 to float, !dbg !8
  %1518 = select i1 %.not37, float %1506, float %1517, !dbg !8
  %1519 = select i1 %.not37, float %1517, float %1512, !dbg !8
  %1520 = select i1 %.not37, float %1513, float %1507, !dbg !8
  %1521 = bitcast float %1520 to i32, !dbg !8
  %1522 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1185, i32 %1521), !dbg !8
  %1523 = bitcast i32 %1522 to float, !dbg !8
  %1524 = select i1 %.not37, float %1507, float %1523, !dbg !8
  %1525 = select i1 %.not37, float %1523, float %1513, !dbg !8
  %1526 = extractelement <8 x float> %1415, i64 4, !dbg !8
  %1527 = extractelement <8 x float> %1415, i64 5, !dbg !8
  %1528 = extractelement <8 x float> %1415, i64 6, !dbg !8
  %1529 = extractelement <8 x float> %1415, i64 7, !dbg !8
  %1530 = select i1 %.not36, float %1527, float %1526, !dbg !8
  %1531 = bitcast float %1530 to i32, !dbg !8
  %1532 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1169, i32 %1531), !dbg !8
  %1533 = bitcast i32 %1532 to float, !dbg !8
  %1534 = select i1 %.not36, float %1526, float %1533, !dbg !8
  %1535 = select i1 %.not36, float %1533, float %1527, !dbg !8
  %1536 = select i1 %.not36, float %1529, float %1528, !dbg !8
  %1537 = bitcast float %1536 to i32, !dbg !8
  %1538 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1169, i32 %1537), !dbg !8
  %1539 = bitcast i32 %1538 to float, !dbg !8
  %1540 = select i1 %.not36, float %1528, float %1539, !dbg !8
  %1541 = select i1 %.not36, float %1539, float %1529, !dbg !8
  %1542 = select i1 %.not37, float %1540, float %1534, !dbg !8
  %1543 = bitcast float %1542 to i32, !dbg !8
  %1544 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1185, i32 %1543), !dbg !8
  %1545 = bitcast i32 %1544 to float, !dbg !8
  %1546 = select i1 %.not37, float %1534, float %1545, !dbg !8
  %1547 = select i1 %.not37, float %1545, float %1540, !dbg !8
  %1548 = select i1 %.not37, float %1541, float %1535, !dbg !8
  %1549 = bitcast float %1548 to i32, !dbg !8
  %1550 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1185, i32 %1549), !dbg !8
  %1551 = bitcast i32 %1550 to float, !dbg !8
  %1552 = select i1 %.not37, float %1535, float %1551, !dbg !8
  %1553 = select i1 %.not37, float %1551, float %1541, !dbg !8
  %1554 = extractelement <8 x float> %1444, i64 4, !dbg !8
  %1555 = extractelement <8 x float> %1444, i64 5, !dbg !8
  %1556 = extractelement <8 x float> %1444, i64 6, !dbg !8
  %1557 = extractelement <8 x float> %1444, i64 7, !dbg !8
  %1558 = select i1 %.not36, float %1555, float %1554, !dbg !8
  %1559 = bitcast float %1558 to i32, !dbg !8
  %1560 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1169, i32 %1559), !dbg !8
  %1561 = bitcast i32 %1560 to float, !dbg !8
  %1562 = select i1 %.not36, float %1554, float %1561, !dbg !8
  %1563 = select i1 %.not36, float %1561, float %1555, !dbg !8
  %1564 = select i1 %.not36, float %1557, float %1556, !dbg !8
  %1565 = bitcast float %1564 to i32, !dbg !8
  %1566 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1169, i32 %1565), !dbg !8
  %1567 = bitcast i32 %1566 to float, !dbg !8
  %1568 = select i1 %.not36, float %1556, float %1567, !dbg !8
  %1569 = select i1 %.not36, float %1567, float %1557, !dbg !8
  %1570 = select i1 %.not37, float %1568, float %1562, !dbg !8
  %1571 = bitcast float %1570 to i32, !dbg !8
  %1572 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1185, i32 %1571), !dbg !8
  %1573 = bitcast i32 %1572 to float, !dbg !8
  %1574 = select i1 %.not37, float %1562, float %1573, !dbg !8
  %1575 = select i1 %.not37, float %1573, float %1568, !dbg !8
  %1576 = select i1 %.not37, float %1569, float %1563, !dbg !8
  %1577 = bitcast float %1576 to i32, !dbg !8
  %1578 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1185, i32 %1577), !dbg !8
  %1579 = bitcast i32 %1578 to float, !dbg !8
  %1580 = select i1 %.not37, float %1563, float %1579, !dbg !8
  %1581 = select i1 %.not37, float %1579, float %1569, !dbg !8
  %1582 = extractelement <8 x float> %1473, i64 4, !dbg !8
  %1583 = extractelement <8 x float> %1473, i64 5, !dbg !8
  %1584 = extractelement <8 x float> %1473, i64 6, !dbg !8
  %1585 = extractelement <8 x float> %1473, i64 7, !dbg !8
  %1586 = select i1 %.not36, float %1583, float %1582, !dbg !8
  %1587 = bitcast float %1586 to i32, !dbg !8
  %1588 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1169, i32 %1587), !dbg !8
  %1589 = bitcast i32 %1588 to float, !dbg !8
  %1590 = select i1 %.not36, float %1582, float %1589, !dbg !8
  %1591 = select i1 %.not36, float %1589, float %1583, !dbg !8
  %1592 = select i1 %.not36, float %1585, float %1584, !dbg !8
  %1593 = bitcast float %1592 to i32, !dbg !8
  %1594 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1169, i32 %1593), !dbg !8
  %1595 = bitcast i32 %1594 to float, !dbg !8
  %1596 = select i1 %.not36, float %1584, float %1595, !dbg !8
  %1597 = select i1 %.not36, float %1595, float %1585, !dbg !8
  %1598 = select i1 %.not37, float %1596, float %1590, !dbg !8
  %1599 = bitcast float %1598 to i32, !dbg !8
  %1600 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1185, i32 %1599), !dbg !8
  %1601 = select i1 %.not37, float %1597, float %1591, !dbg !8
  %1602 = bitcast float %1601 to i32, !dbg !8
  %1603 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1185, i32 %1602), !dbg !8
  %1604 = bitcast i32 %1600 to float, !dbg !8
  %1605 = select i1 %.not37, float %1590, float %1604, !dbg !8
  %1606 = insertelement <4 x float> poison, float %1605, i64 0, !dbg !19
  %1607 = bitcast i32 %1603 to float, !dbg !8
  %1608 = select i1 %.not37, float %1591, float %1607, !dbg !8
  %1609 = insertelement <4 x float> %1606, float %1608, i64 1, !dbg !19
  %1610 = select i1 %.not37, float %1604, float %1596, !dbg !8
  %1611 = insertelement <4 x float> %1609, float %1610, i64 2, !dbg !19
  %1612 = select i1 %.not37, float %1607, float %1597, !dbg !8
  %1613 = insertelement <4 x float> %1611, float %1612, i64 3, !dbg !19
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1497, !dbg !19
  %normalized = fmul <4 x float> %1613, %row.reciprocal, !dbg !19
  %1614 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !20
  %1615 = extractelement <4 x half> %1614, i64 3, !dbg !20
  %1616 = extractelement <4 x half> %1614, i64 2, !dbg !20
  %1617 = extractelement <4 x half> %1614, i64 1, !dbg !20
  %1618 = extractelement <4 x half> %1614, i64 0, !dbg !20
  %1619 = bitcast i32 %1492 to float, !dbg !8
  %1620 = select i1 %.not37, float %1482, float %1619, !dbg !8
  %1621 = insertelement <4 x float> poison, float %1620, i64 0, !dbg !19
  %1622 = bitcast i32 %1495 to float, !dbg !8
  %1623 = select i1 %.not37, float %1483, float %1622, !dbg !8
  %1624 = insertelement <4 x float> %1621, float %1623, i64 1, !dbg !19
  %1625 = select i1 %.not37, float %1619, float %1488, !dbg !8
  %1626 = insertelement <4 x float> %1624, float %1625, i64 2, !dbg !19
  %1627 = select i1 %.not37, float %1622, float %1489, !dbg !8
  %1628 = insertelement <4 x float> %1626, float %1627, i64 3, !dbg !19
  %normalized108 = fmul <4 x float> %1628, %row.reciprocal, !dbg !19
  %1629 = fptrunc <4 x float> %normalized108 to <4 x half>, !dbg !20
  %1630 = extractelement <4 x half> %1629, i64 3, !dbg !20
  %1631 = extractelement <4 x half> %1629, i64 2, !dbg !20
  %1632 = extractelement <4 x half> %1629, i64 1, !dbg !20
  %1633 = extractelement <4 x half> %1629, i64 0, !dbg !20
  %1634 = bitcast i32 %1382 to float, !dbg !8
  %1635 = select i1 %.not37, float %1372, float %1634, !dbg !8
  %1636 = insertelement <4 x float> poison, float %1635, i64 0, !dbg !19
  %1637 = bitcast i32 %1385 to float, !dbg !8
  %1638 = select i1 %.not37, float %1373, float %1637, !dbg !8
  %1639 = insertelement <4 x float> %1636, float %1638, i64 1, !dbg !19
  %1640 = select i1 %.not37, float %1634, float %1378, !dbg !8
  %1641 = insertelement <4 x float> %1639, float %1640, i64 2, !dbg !19
  %1642 = select i1 %.not37, float %1637, float %1379, !dbg !8
  %1643 = insertelement <4 x float> %1641, float %1642, i64 3, !dbg !19
  %row.reciprocal109 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1279, !dbg !19
  %normalized110 = fmul <4 x float> %1643, %row.reciprocal109, !dbg !19
  %1644 = fptrunc <4 x float> %normalized110 to <4 x half>, !dbg !20
  %1645 = extractelement <4 x half> %1644, i64 3, !dbg !20
  %1646 = extractelement <4 x half> %1644, i64 2, !dbg !20
  %1647 = extractelement <4 x half> %1644, i64 1, !dbg !20
  %1648 = extractelement <4 x half> %1644, i64 0, !dbg !20
  %1649 = bitcast i32 %1274 to float, !dbg !8
  %1650 = select i1 %.not37, float %1264, float %1649, !dbg !8
  %1651 = insertelement <4 x float> poison, float %1650, i64 0, !dbg !19
  %1652 = bitcast i32 %1277 to float, !dbg !8
  %1653 = select i1 %.not37, float %1265, float %1652, !dbg !8
  %1654 = insertelement <4 x float> %1651, float %1653, i64 1, !dbg !19
  %1655 = select i1 %.not37, float %1649, float %1270, !dbg !8
  %1656 = insertelement <4 x float> %1654, float %1655, i64 2, !dbg !19
  %1657 = select i1 %.not37, float %1652, float %1271, !dbg !8
  %1658 = insertelement <4 x float> %1656, float %1657, i64 3, !dbg !19
  %normalized111 = fmul <4 x float> %1658, %row.reciprocal109, !dbg !19
  %1659 = fptrunc <4 x float> %normalized111 to <4 x half>, !dbg !20
  %1660 = extractelement <4 x half> %1659, i64 3, !dbg !20
  %1661 = extractelement <4 x half> %1659, i64 2, !dbg !20
  %1662 = extractelement <4 x half> %1659, i64 1, !dbg !20
  %1663 = extractelement <4 x half> %1659, i64 0, !dbg !20
  %1664 = add nuw nsw i64 %15, %17, !dbg !21
  %1665 = shl i64 %1664, 19, !dbg !21
  %1666 = and i64 %1665, 16252928, !dbg !21
  %address.thread960 = call i32 asm sideeffect "", "=v,0"(i32 %23) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local961 = zext nneg i32 %address.thread960 to i64, !dbg !13
  %address.local962 = add nuw i64 %20, %address.local961, !dbg !21
  %address.local963 = lshr i64 %address.local961, 2, !dbg !17
  %address.local964 = and i64 %address.local963, 48, !dbg !17
  %address.local965 = add i64 %address.local962, %address.local964, !dbg !21
  %address.local966 = and i64 %address.local961, 240, !dbg !21
  %address.local968 = sub i64 %address.local965, %address.local966, !dbg !21
  %address.local969 = add i64 %address.local968, %21, !dbg !21
  %address.local970 = shl nuw nsw i64 %address.local969, 7, !dbg !21
  %address.local971 = add nuw nsw i64 %1666, %address.local970, !dbg !21
  %address.local972 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local971, !dbg !21
  %address.local973 = lshr i64 %address.local961, 2, !dbg !13
  %address.local974 = and i64 %address.local973, 12, !dbg !13
  %address.local975 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local972, i64 %address.local974, !dbg !21
  store half %1663, ptr addrspace(1) %address.local975, align 2, !dbg !21
  %address.local976 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local971, !dbg !21
  %address.local977 = or disjoint i64 %address.local974, 1, !dbg !21
  %address.local978 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local976, i64 %address.local977, !dbg !21
  store half %1662, ptr addrspace(1) %address.local978, align 2, !dbg !21
  %address.local979 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local971, !dbg !21
  %address.local980 = or disjoint i64 %address.local974, 2, !dbg !21
  %address.local981 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local979, i64 %address.local980, !dbg !21
  store half %1661, ptr addrspace(1) %address.local981, align 2, !dbg !21
  %address.local982 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local971, !dbg !21
  %address.local983 = or disjoint i64 %address.local974, 3, !dbg !21
  %address.local984 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local983, !dbg !21
  store half %1660, ptr addrspace(1) %address.local984, align 2, !dbg !21
  %address.local985 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local971, !dbg !21
  %address.local986 = or disjoint i64 %address.local974, 16, !dbg !21
  %address.local987 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local986, !dbg !21
  store half %1648, ptr addrspace(1) %address.local987, align 2, !dbg !21
  %address.local988 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local971, !dbg !21
  %address.local989 = or disjoint i64 %address.local974, 17, !dbg !21
  %address.local990 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local988, i64 %address.local989, !dbg !21
  store half %1647, ptr addrspace(1) %address.local990, align 2, !dbg !21
  %address.local991 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local971, !dbg !21
  %address.local992 = or disjoint i64 %address.local974, 18, !dbg !21
  %address.local993 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local991, i64 %address.local992, !dbg !21
  store half %1646, ptr addrspace(1) %address.local993, align 2, !dbg !21
  %address.local994 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local971, !dbg !21
  %address.local995 = or disjoint i64 %address.local974, 19, !dbg !21
  %address.local996 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local994, i64 %address.local995, !dbg !21
  store half %1645, ptr addrspace(1) %address.local996, align 2, !dbg !21
  %address.local997 = shl i64 %address.local969, 7, !dbg !21
  %address.local998 = add i64 %address.local997, 8192, !dbg !21
  %address.local999 = add nuw nsw i64 %1666, %address.local998, !dbg !21
  %address.local1000 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local999, !dbg !21
  %address.local1001 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1000, i64 %address.local974, !dbg !21
  store half %1633, ptr addrspace(1) %address.local1001, align 2, !dbg !21
  %address.local1002 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local999, !dbg !21
  %address.local1003 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1002, i64 %address.local977, !dbg !21
  store half %1632, ptr addrspace(1) %address.local1003, align 2, !dbg !21
  %address.local1004 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local999, !dbg !21
  %address.local1005 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1004, i64 %address.local980, !dbg !21
  store half %1631, ptr addrspace(1) %address.local1005, align 2, !dbg !21
  %address.local1006 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local999, !dbg !21
  %address.local1007 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1006, i64 %address.local983, !dbg !21
  store half %1630, ptr addrspace(1) %address.local1007, align 2, !dbg !21
  %address.local1008 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local999, !dbg !21
  %address.local1009 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1008, i64 %address.local986, !dbg !21
  store half %1618, ptr addrspace(1) %address.local1009, align 2, !dbg !21
  %address.local1010 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local999, !dbg !21
  %address.local1011 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local989, !dbg !21
  store half %1617, ptr addrspace(1) %address.local1011, align 2, !dbg !21
  %address.local1012 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local999, !dbg !21
  %address.local1013 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1012, i64 %address.local992, !dbg !21
  store half %1616, ptr addrspace(1) %address.local1013, align 2, !dbg !21
  %address.local1014 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local999, !dbg !21
  %address.local1015 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1014, i64 %address.local995, !dbg !21
  store half %1615, ptr addrspace(1) %address.local1015, align 2, !dbg !21
  %1667 = insertelement <4 x float> poison, float %1574, i64 0, !dbg !19
  %1668 = insertelement <4 x float> %1667, float %1580, i64 1, !dbg !19
  %1669 = insertelement <4 x float> %1668, float %1575, i64 2, !dbg !19
  %1670 = insertelement <4 x float> %1669, float %1581, i64 3, !dbg !19
  %row.reciprocal112 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1497, !dbg !19
  %normalized113 = fmul <4 x float> %1670, %row.reciprocal112, !dbg !19
  %1671 = fptrunc <4 x float> %normalized113 to <4 x half>, !dbg !20
  %1672 = extractelement <4 x half> %1671, i64 3, !dbg !20
  %1673 = extractelement <4 x half> %1671, i64 2, !dbg !20
  %1674 = extractelement <4 x half> %1671, i64 1, !dbg !20
  %1675 = extractelement <4 x half> %1671, i64 0, !dbg !20
  %1676 = insertelement <4 x float> poison, float %1465, i64 0, !dbg !19
  %1677 = insertelement <4 x float> %1676, float %1471, i64 1, !dbg !19
  %1678 = insertelement <4 x float> %1677, float %1466, i64 2, !dbg !19
  %1679 = insertelement <4 x float> %1678, float %1472, i64 3, !dbg !19
  %normalized114 = fmul <4 x float> %1679, %row.reciprocal112, !dbg !19
  %1680 = fptrunc <4 x float> %normalized114 to <4 x half>, !dbg !20
  %1681 = extractelement <4 x half> %1680, i64 3, !dbg !20
  %1682 = extractelement <4 x half> %1680, i64 2, !dbg !20
  %1683 = extractelement <4 x half> %1680, i64 1, !dbg !20
  %1684 = extractelement <4 x half> %1680, i64 0, !dbg !20
  %1685 = insertelement <4 x float> poison, float %1356, i64 0, !dbg !19
  %1686 = insertelement <4 x float> %1685, float %1362, i64 1, !dbg !19
  %1687 = insertelement <4 x float> %1686, float %1357, i64 2, !dbg !19
  %1688 = insertelement <4 x float> %1687, float %1363, i64 3, !dbg !19
  %row.reciprocal115 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1279, !dbg !19
  %normalized116 = fmul <4 x float> %1688, %row.reciprocal115, !dbg !19
  %1689 = fptrunc <4 x float> %normalized116 to <4 x half>, !dbg !20
  %1690 = extractelement <4 x half> %1689, i64 3, !dbg !20
  %1691 = extractelement <4 x half> %1689, i64 2, !dbg !20
  %1692 = extractelement <4 x half> %1689, i64 1, !dbg !20
  %1693 = extractelement <4 x half> %1689, i64 0, !dbg !20
  %1694 = insertelement <4 x float> poison, float %1247, i64 0, !dbg !19
  %1695 = insertelement <4 x float> %1694, float %1253, i64 1, !dbg !19
  %1696 = insertelement <4 x float> %1695, float %1248, i64 2, !dbg !19
  %1697 = insertelement <4 x float> %1696, float %1254, i64 3, !dbg !19
  %normalized117 = fmul <4 x float> %1697, %row.reciprocal115, !dbg !19
  %1698 = fptrunc <4 x float> %normalized117 to <4 x half>, !dbg !20
  %1699 = extractelement <4 x half> %1698, i64 3, !dbg !20
  %1700 = extractelement <4 x half> %1698, i64 2, !dbg !20
  %1701 = extractelement <4 x half> %1698, i64 1, !dbg !20
  %1702 = extractelement <4 x half> %1698, i64 0, !dbg !20
  %1703 = shl nuw nsw i64 %19, 19, !dbg !21
  %address.thread1016 = call i32 asm sideeffect "", "=v,0"(i32 %23) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local1017 = zext nneg i32 %address.thread1016 to i64, !dbg !13
  %address.local1018 = add nuw i64 %20, %address.local1017, !dbg !21
  %address.local1019 = lshr i64 %address.local1017, 2, !dbg !17
  %address.local1020 = and i64 %address.local1019, 48, !dbg !17
  %address.local1021 = add i64 %address.local1018, %address.local1020, !dbg !21
  %address.local1022 = and i64 %address.local1017, 240, !dbg !21
  %address.local1024 = sub i64 %address.local1021, %address.local1022, !dbg !21
  %address.local1025 = add i64 %address.local1024, %21, !dbg !21
  %address.local1026 = shl nuw nsw i64 %address.local1025, 7, !dbg !21
  %address.local1027 = add nuw nsw i64 %1703, %address.local1026, !dbg !21
  %address.local1028 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1027, !dbg !21
  %address.local1029 = lshr i64 %address.local1017, 2, !dbg !13
  %address.local1030 = and i64 %address.local1029, 12, !dbg !13
  %address.local1031 = or disjoint i64 %address.local1030, 32, !dbg !21
  %address.local1032 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1028, i64 %address.local1031, !dbg !21
  store half %1702, ptr addrspace(1) %address.local1032, align 2, !dbg !21
  %address.local1033 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1027, !dbg !21
  %address.local1034 = or disjoint i64 %address.local1030, 33, !dbg !21
  %address.local1035 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1033, i64 %address.local1034, !dbg !21
  store half %1701, ptr addrspace(1) %address.local1035, align 2, !dbg !21
  %address.local1036 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1027, !dbg !21
  %address.local1037 = or disjoint i64 %address.local1030, 34, !dbg !21
  %address.local1038 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1036, i64 %address.local1037, !dbg !21
  store half %1700, ptr addrspace(1) %address.local1038, align 2, !dbg !21
  %address.local1039 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1027, !dbg !21
  %address.local1040 = or disjoint i64 %address.local1030, 35, !dbg !21
  %address.local1041 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1039, i64 %address.local1040, !dbg !21
  store half %1699, ptr addrspace(1) %address.local1041, align 2, !dbg !21
  %address.local1042 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1027, !dbg !21
  %address.local1043 = or disjoint i64 %address.local1030, 48, !dbg !21
  %address.local1044 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1042, i64 %address.local1043, !dbg !21
  store half %1693, ptr addrspace(1) %address.local1044, align 2, !dbg !21
  %address.local1045 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1027, !dbg !21
  %address.local1046 = or disjoint i64 %address.local1030, 49, !dbg !21
  %address.local1047 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1045, i64 %address.local1046, !dbg !21
  store half %1692, ptr addrspace(1) %address.local1047, align 2, !dbg !21
  %address.local1048 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1027, !dbg !21
  %address.local1049 = or disjoint i64 %address.local1030, 50, !dbg !21
  %address.local1050 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1048, i64 %address.local1049, !dbg !21
  store half %1691, ptr addrspace(1) %address.local1050, align 2, !dbg !21
  %address.local1051 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1027, !dbg !21
  %address.local1052 = or i64 %address.local1029, 51, !dbg !21
  %address.local1053 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1051, i64 %address.local1052, !dbg !21
  store half %1690, ptr addrspace(1) %address.local1053, align 2, !dbg !21
  %address.local1054 = shl i64 %address.local1025, 7, !dbg !21
  %address.local1055 = add i64 %address.local1054, 8192, !dbg !21
  %address.local1056 = add nuw nsw i64 %1703, %address.local1055, !dbg !21
  %address.local1057 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1056, !dbg !21
  %address.local1058 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1057, i64 %address.local1031, !dbg !21
  store half %1684, ptr addrspace(1) %address.local1058, align 2, !dbg !21
  %address.local1059 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1056, !dbg !21
  %address.local1060 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1059, i64 %address.local1034, !dbg !21
  store half %1683, ptr addrspace(1) %address.local1060, align 2, !dbg !21
  %address.local1061 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1056, !dbg !21
  %address.local1062 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1061, i64 %address.local1037, !dbg !21
  store half %1682, ptr addrspace(1) %address.local1062, align 2, !dbg !21
  %address.local1063 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1056, !dbg !21
  %address.local1064 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1063, i64 %address.local1040, !dbg !21
  store half %1681, ptr addrspace(1) %address.local1064, align 2, !dbg !21
  %address.local1065 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1056, !dbg !21
  %address.local1066 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1065, i64 %address.local1043, !dbg !21
  store half %1675, ptr addrspace(1) %address.local1066, align 2, !dbg !21
  %address.local1067 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1056, !dbg !21
  %address.local1068 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1067, i64 %address.local1046, !dbg !21
  store half %1674, ptr addrspace(1) %address.local1068, align 2, !dbg !21
  %address.local1069 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1056, !dbg !21
  %address.local1070 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1069, i64 %address.local1049, !dbg !21
  store half %1673, ptr addrspace(1) %address.local1070, align 2, !dbg !21
  %address.local1071 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1056, !dbg !21
  %address.local1072 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1071, i64 %address.local1052, !dbg !21
  store half %1672, ptr addrspace(1) %address.local1072, align 2, !dbg !21
  %1704 = insertelement <4 x float> poison, float %1546, i64 0, !dbg !19
  %1705 = insertelement <4 x float> %1704, float %1552, i64 1, !dbg !19
  %1706 = insertelement <4 x float> %1705, float %1547, i64 2, !dbg !19
  %1707 = insertelement <4 x float> %1706, float %1553, i64 3, !dbg !19
  %row.reciprocal118 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1497, !dbg !19
  %normalized119 = fmul <4 x float> %1707, %row.reciprocal118, !dbg !19
  %1708 = fptrunc <4 x float> %normalized119 to <4 x half>, !dbg !20
  %1709 = extractelement <4 x half> %1708, i64 3, !dbg !20
  %1710 = extractelement <4 x half> %1708, i64 2, !dbg !20
  %1711 = extractelement <4 x half> %1708, i64 1, !dbg !20
  %1712 = extractelement <4 x half> %1708, i64 0, !dbg !20
  %1713 = insertelement <4 x float> poison, float %1436, i64 0, !dbg !19
  %1714 = insertelement <4 x float> %1713, float %1442, i64 1, !dbg !19
  %1715 = insertelement <4 x float> %1714, float %1437, i64 2, !dbg !19
  %1716 = insertelement <4 x float> %1715, float %1443, i64 3, !dbg !19
  %normalized120 = fmul <4 x float> %1716, %row.reciprocal118, !dbg !19
  %1717 = fptrunc <4 x float> %normalized120 to <4 x half>, !dbg !20
  %1718 = extractelement <4 x half> %1717, i64 3, !dbg !20
  %1719 = extractelement <4 x half> %1717, i64 2, !dbg !20
  %1720 = extractelement <4 x half> %1717, i64 1, !dbg !20
  %1721 = extractelement <4 x half> %1717, i64 0, !dbg !20
  %1722 = insertelement <4 x float> poison, float %1328, i64 0, !dbg !19
  %1723 = insertelement <4 x float> %1722, float %1334, i64 1, !dbg !19
  %1724 = insertelement <4 x float> %1723, float %1329, i64 2, !dbg !19
  %1725 = insertelement <4 x float> %1724, float %1335, i64 3, !dbg !19
  %row.reciprocal121 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1279, !dbg !19
  %normalized122 = fmul <4 x float> %1725, %row.reciprocal121, !dbg !19
  %1726 = fptrunc <4 x float> %normalized122 to <4 x half>, !dbg !20
  %1727 = extractelement <4 x half> %1726, i64 3, !dbg !20
  %1728 = extractelement <4 x half> %1726, i64 2, !dbg !20
  %1729 = extractelement <4 x half> %1726, i64 1, !dbg !20
  %1730 = extractelement <4 x half> %1726, i64 0, !dbg !20
  %1731 = insertelement <4 x float> poison, float %1218, i64 0, !dbg !19
  %1732 = insertelement <4 x float> %1731, float %1224, i64 1, !dbg !19
  %1733 = insertelement <4 x float> %1732, float %1219, i64 2, !dbg !19
  %1734 = insertelement <4 x float> %1733, float %1225, i64 3, !dbg !19
  %normalized123 = fmul <4 x float> %1734, %row.reciprocal121, !dbg !19
  %1735 = fptrunc <4 x float> %normalized123 to <4 x half>, !dbg !20
  %1736 = extractelement <4 x half> %1735, i64 3, !dbg !20
  %1737 = extractelement <4 x half> %1735, i64 2, !dbg !20
  %1738 = extractelement <4 x half> %1735, i64 1, !dbg !20
  %1739 = extractelement <4 x half> %1735, i64 0, !dbg !20
  %1740 = shl nuw nsw i64 %19, 19, !dbg !21
  %address.thread1073 = call i32 asm sideeffect "", "=v,0"(i32 %23) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local1074 = zext nneg i32 %address.thread1073 to i64, !dbg !13
  %address.local1075 = add nuw i64 %20, %address.local1074, !dbg !21
  %address.local1076 = lshr i64 %address.local1074, 2, !dbg !17
  %address.local1077 = and i64 %address.local1076, 48, !dbg !17
  %address.local1078 = add i64 %address.local1075, %address.local1077, !dbg !21
  %address.local1079 = and i64 %address.local1074, 240, !dbg !21
  %address.local1081 = sub i64 %address.local1078, %address.local1079, !dbg !21
  %address.local1082 = add i64 %address.local1081, %21, !dbg !21
  %address.local1083 = shl nuw nsw i64 %address.local1082, 7, !dbg !21
  %address.local1084 = add nuw nsw i64 %1740, %address.local1083, !dbg !21
  %address.local1085 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1084, !dbg !21
  %address.local1086 = lshr i64 %address.local1074, 2, !dbg !13
  %address.local1087 = and i64 %address.local1086, 12, !dbg !13
  %address.local1088 = or disjoint i64 %address.local1087, 64, !dbg !21
  %address.local1089 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1085, i64 %address.local1088, !dbg !21
  store half %1739, ptr addrspace(1) %address.local1089, align 2, !dbg !21
  %address.local1090 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1084, !dbg !21
  %address.local1091 = or disjoint i64 %address.local1087, 65, !dbg !21
  %address.local1092 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1090, i64 %address.local1091, !dbg !21
  store half %1738, ptr addrspace(1) %address.local1092, align 2, !dbg !21
  %address.local1093 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1084, !dbg !21
  %address.local1094 = or disjoint i64 %address.local1087, 66, !dbg !21
  %address.local1095 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1093, i64 %address.local1094, !dbg !21
  store half %1737, ptr addrspace(1) %address.local1095, align 2, !dbg !21
  %address.local1096 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1084, !dbg !21
  %address.local1097 = or disjoint i64 %address.local1087, 67, !dbg !21
  %address.local1098 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1096, i64 %address.local1097, !dbg !21
  store half %1736, ptr addrspace(1) %address.local1098, align 2, !dbg !21
  %address.local1099 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1084, !dbg !21
  %address.local1100 = or disjoint i64 %address.local1087, 80, !dbg !21
  %address.local1101 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1099, i64 %address.local1100, !dbg !21
  store half %1730, ptr addrspace(1) %address.local1101, align 2, !dbg !21
  %address.local1102 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1084, !dbg !21
  %address.local1103 = or disjoint i64 %address.local1087, 81, !dbg !21
  %address.local1104 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1102, i64 %address.local1103, !dbg !21
  store half %1729, ptr addrspace(1) %address.local1104, align 2, !dbg !21
  %address.local1105 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1084, !dbg !21
  %address.local1106 = or disjoint i64 %address.local1087, 82, !dbg !21
  %address.local1107 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1105, i64 %address.local1106, !dbg !21
  store half %1728, ptr addrspace(1) %address.local1107, align 2, !dbg !21
  %address.local1108 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1084, !dbg !21
  %address.local1109 = or disjoint i64 %address.local1087, 83, !dbg !21
  %address.local1110 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1108, i64 %address.local1109, !dbg !21
  store half %1727, ptr addrspace(1) %address.local1110, align 2, !dbg !21
  %address.local1111 = shl i64 %address.local1082, 7, !dbg !21
  %address.local1112 = add i64 %address.local1111, 8192, !dbg !21
  %address.local1113 = add nuw nsw i64 %1740, %address.local1112, !dbg !21
  %address.local1114 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1113, !dbg !21
  %address.local1115 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1114, i64 %address.local1088, !dbg !21
  store half %1721, ptr addrspace(1) %address.local1115, align 2, !dbg !21
  %address.local1116 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1113, !dbg !21
  %address.local1117 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1116, i64 %address.local1091, !dbg !21
  store half %1720, ptr addrspace(1) %address.local1117, align 2, !dbg !21
  %address.local1118 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1113, !dbg !21
  %address.local1119 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1118, i64 %address.local1094, !dbg !21
  store half %1719, ptr addrspace(1) %address.local1119, align 2, !dbg !21
  %address.local1120 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1113, !dbg !21
  %address.local1121 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1120, i64 %address.local1097, !dbg !21
  store half %1718, ptr addrspace(1) %address.local1121, align 2, !dbg !21
  %address.local1122 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1113, !dbg !21
  %address.local1123 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1122, i64 %address.local1100, !dbg !21
  store half %1712, ptr addrspace(1) %address.local1123, align 2, !dbg !21
  %address.local1124 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1113, !dbg !21
  %address.local1125 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1124, i64 %address.local1103, !dbg !21
  store half %1711, ptr addrspace(1) %address.local1125, align 2, !dbg !21
  %address.local1126 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1113, !dbg !21
  %address.local1127 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1126, i64 %address.local1106, !dbg !21
  store half %1710, ptr addrspace(1) %address.local1127, align 2, !dbg !21
  %address.local1128 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1113, !dbg !21
  %address.local1129 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1128, i64 %address.local1109, !dbg !21
  store half %1709, ptr addrspace(1) %address.local1129, align 2, !dbg !21
  %1741 = insertelement <4 x float> poison, float %1518, i64 0, !dbg !19
  %1742 = insertelement <4 x float> %1741, float %1524, i64 1, !dbg !19
  %1743 = insertelement <4 x float> %1742, float %1519, i64 2, !dbg !19
  %1744 = insertelement <4 x float> %1743, float %1525, i64 3, !dbg !19
  %row.reciprocal124 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1497, !dbg !19
  %normalized125 = fmul <4 x float> %1744, %row.reciprocal124, !dbg !19
  %1745 = fptrunc <4 x float> %normalized125 to <4 x half>, !dbg !20
  %1746 = extractelement <4 x half> %1745, i64 3, !dbg !20
  %1747 = extractelement <4 x half> %1745, i64 2, !dbg !20
  %1748 = extractelement <4 x half> %1745, i64 1, !dbg !20
  %1749 = extractelement <4 x half> %1745, i64 0, !dbg !20
  %1750 = insertelement <4 x float> poison, float %1407, i64 0, !dbg !19
  %1751 = insertelement <4 x float> %1750, float %1413, i64 1, !dbg !19
  %1752 = insertelement <4 x float> %1751, float %1408, i64 2, !dbg !19
  %1753 = insertelement <4 x float> %1752, float %1414, i64 3, !dbg !19
  %normalized126 = fmul <4 x float> %1753, %row.reciprocal124, !dbg !19
  %1754 = fptrunc <4 x float> %normalized126 to <4 x half>, !dbg !20
  %1755 = extractelement <4 x half> %1754, i64 3, !dbg !20
  %1756 = extractelement <4 x half> %1754, i64 2, !dbg !20
  %1757 = extractelement <4 x half> %1754, i64 1, !dbg !20
  %1758 = extractelement <4 x half> %1754, i64 0, !dbg !20
  %1759 = insertelement <4 x float> poison, float %1300, i64 0, !dbg !19
  %1760 = insertelement <4 x float> %1759, float %1306, i64 1, !dbg !19
  %1761 = insertelement <4 x float> %1760, float %1301, i64 2, !dbg !19
  %1762 = insertelement <4 x float> %1761, float %1307, i64 3, !dbg !19
  %row.reciprocal127 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1279, !dbg !19
  %normalized128 = fmul <4 x float> %1762, %row.reciprocal127, !dbg !19
  %1763 = fptrunc <4 x float> %normalized128 to <4 x half>, !dbg !20
  %1764 = extractelement <4 x half> %1763, i64 3, !dbg !20
  %1765 = extractelement <4 x half> %1763, i64 2, !dbg !20
  %1766 = extractelement <4 x half> %1763, i64 1, !dbg !20
  %1767 = extractelement <4 x half> %1763, i64 0, !dbg !20
  %1768 = insertelement <4 x float> poison, float %1189, i64 0, !dbg !19
  %1769 = insertelement <4 x float> %1768, float %1195, i64 1, !dbg !19
  %1770 = insertelement <4 x float> %1769, float %1190, i64 2, !dbg !19
  %1771 = insertelement <4 x float> %1770, float %1196, i64 3, !dbg !19
  %normalized129 = fmul <4 x float> %1771, %row.reciprocal127, !dbg !19
  %1772 = fptrunc <4 x float> %normalized129 to <4 x half>, !dbg !20
  %1773 = extractelement <4 x half> %1772, i64 3, !dbg !20
  %1774 = extractelement <4 x half> %1772, i64 2, !dbg !20
  %1775 = extractelement <4 x half> %1772, i64 1, !dbg !20
  %1776 = extractelement <4 x half> %1772, i64 0, !dbg !20
  %1777 = shl nuw nsw i64 %19, 19, !dbg !21
  %address.thread1130 = call i32 asm sideeffect "", "=v,0"(i32 %23) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local1131 = zext nneg i32 %address.thread1130 to i64, !dbg !13
  %address.local1132 = add nuw i64 %20, %address.local1131, !dbg !21
  %address.local1133 = lshr i64 %address.local1131, 2, !dbg !17
  %address.local1134 = and i64 %address.local1133, 48, !dbg !17
  %address.local1135 = add i64 %address.local1132, %address.local1134, !dbg !21
  %address.local1136 = and i64 %address.local1131, 240, !dbg !21
  %address.local1138 = sub i64 %address.local1135, %address.local1136, !dbg !21
  %address.local1139 = add i64 %address.local1138, %21, !dbg !21
  %address.local1140 = shl nuw nsw i64 %address.local1139, 7, !dbg !21
  %address.local1141 = add nuw nsw i64 %1777, %address.local1140, !dbg !21
  %address.local1142 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1141, !dbg !21
  %address.local1143 = lshr i64 %address.local1131, 2, !dbg !13
  %address.local1144 = and i64 %address.local1143, 12, !dbg !13
  %address.local1145 = or disjoint i64 %address.local1144, 96, !dbg !21
  %address.local1146 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1142, i64 %address.local1145, !dbg !21
  store half %1776, ptr addrspace(1) %address.local1146, align 2, !dbg !21
  %address.local1147 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1141, !dbg !21
  %address.local1148 = or disjoint i64 %address.local1144, 97, !dbg !21
  %address.local1149 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1147, i64 %address.local1148, !dbg !21
  store half %1775, ptr addrspace(1) %address.local1149, align 2, !dbg !21
  %address.local1150 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1141, !dbg !21
  %address.local1151 = or disjoint i64 %address.local1144, 98, !dbg !21
  %address.local1152 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1150, i64 %address.local1151, !dbg !21
  store half %1774, ptr addrspace(1) %address.local1152, align 2, !dbg !21
  %address.local1153 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1141, !dbg !21
  %address.local1154 = or disjoint i64 %address.local1144, 99, !dbg !21
  %address.local1155 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1153, i64 %address.local1154, !dbg !21
  store half %1773, ptr addrspace(1) %address.local1155, align 2, !dbg !21
  %address.local1156 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1141, !dbg !21
  %address.local1157 = or disjoint i64 %address.local1144, 112, !dbg !21
  %address.local1158 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1156, i64 %address.local1157, !dbg !21
  store half %1767, ptr addrspace(1) %address.local1158, align 2, !dbg !21
  %address.local1159 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1141, !dbg !21
  %address.local1160 = or disjoint i64 %address.local1144, 113, !dbg !21
  %address.local1161 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1159, i64 %address.local1160, !dbg !21
  store half %1766, ptr addrspace(1) %address.local1161, align 2, !dbg !21
  %address.local1162 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1141, !dbg !21
  %address.local1163 = or disjoint i64 %address.local1144, 114, !dbg !21
  %address.local1164 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1162, i64 %address.local1163, !dbg !21
  store half %1765, ptr addrspace(1) %address.local1164, align 2, !dbg !21
  %address.local1165 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1141, !dbg !21
  %address.local1166 = or i64 %address.local1143, 115, !dbg !21
  %address.local1167 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1165, i64 %address.local1166, !dbg !21
  store half %1764, ptr addrspace(1) %address.local1167, align 2, !dbg !21
  %address.local1168 = shl i64 %address.local1139, 7, !dbg !21
  %address.local1169 = add i64 %address.local1168, 8192, !dbg !21
  %address.local1170 = add nuw nsw i64 %1777, %address.local1169, !dbg !21
  %address.local1171 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1170, !dbg !21
  %address.local1172 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1171, i64 %address.local1145, !dbg !21
  store half %1758, ptr addrspace(1) %address.local1172, align 2, !dbg !21
  %address.local1173 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1170, !dbg !21
  %address.local1174 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1173, i64 %address.local1148, !dbg !21
  store half %1757, ptr addrspace(1) %address.local1174, align 2, !dbg !21
  %address.local1175 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1170, !dbg !21
  %address.local1176 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1175, i64 %address.local1151, !dbg !21
  store half %1756, ptr addrspace(1) %address.local1176, align 2, !dbg !21
  %address.local1177 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1170, !dbg !21
  %address.local1178 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1177, i64 %address.local1154, !dbg !21
  store half %1755, ptr addrspace(1) %address.local1178, align 2, !dbg !21
  %address.local1179 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1170, !dbg !21
  %address.local1180 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1179, i64 %address.local1157, !dbg !21
  store half %1749, ptr addrspace(1) %address.local1180, align 2, !dbg !21
  %address.local1181 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1170, !dbg !21
  %address.local1182 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1181, i64 %address.local1160, !dbg !21
  store half %1748, ptr addrspace(1) %address.local1182, align 2, !dbg !21
  %address.local1183 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1170, !dbg !21
  %address.local1184 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1183, i64 %address.local1163, !dbg !21
  store half %1747, ptr addrspace(1) %address.local1184, align 2, !dbg !21
  %address.local1185 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1170, !dbg !21
  %address.local1186 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1185, i64 %address.local1166, !dbg !21
  store half %1746, ptr addrspace(1) %address.local1186, align 2, !dbg !21
  ret void, !dbg !5
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.y() #2

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

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier() #5

; Function Attrs: convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.readfirstlane.i32(i32) #6

; Function Attrs: nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.asyncmark() #7

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maxnum.f32(float, float) #8

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare <1 x float> @llvm.maxnum.v1f32(<1 x float>, <1 x float>) #8

; Function Attrs: nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.wait.asyncmark(i16 immarg) #7

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.amdgcn.exp2.f32(float) #8

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.sched.barrier(i32 immarg) #5

attributes #0 = { convergent willreturn memory(read) }
attributes #1 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1" "frisk.global_prefetch"="8" "frisk.register_comparisons"="1" "frisk.register_lifetimes"="1" "uniform-work-group-size" }
attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }
attributes #4 = { convergent nocallback nofree nounwind willreturn memory(none) }
attributes #5 = { convergent nocallback nofree nounwind willreturn }
attributes #6 = { convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none) }
attributes #7 = { nocallback nofree nounwind willreturn }
attributes #8 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #9 = { convergent nounwind willreturn memory(none) }
attributes #10 = { convergent nounwind }
attributes #11 = { memory(inaccessiblemem: readwrite) }

!llvm.dbg.cu = !{!3}
!llvm.module.flags = !{!38}

!0 = !DIFile(filename: "causal.input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/causal_best_mytest")
!1 = !{}
!2 = !DISubroutineType(types: !1)
!3 = distinct !DICompileUnit(language: DW_LANG_C, file: !0, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!4 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !0, file: !0, line: 23, type: !2, scopeLine: 23, spFlags: DISPFlagDefinition, unit: !3)
!5 = !DILocation(line: 23, column: 3, scope: !4)
!7 = !DILocation(line: 73, column: 26, scope: !4)
!8 = !DILocation(line: 60, column: 24, scope: !4)
!10 = !DILocation(line: 48, column: 17, scope: !4)
!12 = !DILocation(line: 70, column: 22, scope: !4)
!13 = !DILocation(line: 72, column: 24, scope: !4)
!14 = !DILocation(line: 102, column: 23, scope: !4)
!15 = !DILocation(line: 71, column: 22, scope: !4)
!16 = !DILocation(line: 101, column: 20, scope: !4)
!17 = !DILocation(line: 77, column: 19, scope: !4)
!18 = !DILocation(line: 79, column: 24, scope: !4)
!19 = !DILocation(line: 107, column: 23, scope: !4)
!20 = !DILocation(line: 108, column: 19, scope: !4)
!21 = !DILocation(line: 109, column: 9, scope: !4)
!22 = !DILocation(line: 63, column: 17, scope: !4)
!23 = !DILocation(line: 80, column: 23, scope: !4)
!24 = !DILocation(line: 81, column: 26, scope: !4)
!25 = !DILocation(line: 88, column: 21, scope: !4)
!26 = !DILocation(line: 89, column: 20, scope: !4)
!27 = !DILocation(line: 90, column: 22, scope: !4)
!28 = !DILocation(line: 91, column: 23, scope: !4)
!29 = !DILocation(line: 92, column: 20, scope: !4)
!30 = !DILocation(line: 93, column: 22, scope: !4)
!31 = !DILocation(line: 94, column: 22, scope: !4)
!32 = !DILocation(line: 95, column: 20, scope: !4)
!33 = !DILocation(line: 96, column: 16, scope: !4)
!34 = !DILocation(line: 103, column: 22, scope: !4)
!36 = !DILocation(line: 74, column: 23, scope: !4)
!37 = !{i32 256, i32 1, i32 1}
!38 = !{i32 2, !"Debug Info Version", i32 3}
!39 = !{i32 0, i32 256}
!40 = !{i32 4}
