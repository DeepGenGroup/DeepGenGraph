; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/causal_cvt_pack_mytest/causal.input.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/causal_cvt_pack_mytest/causal.input.mlir"

@shm_0 = addrspace(3) global [16384 x i8] undef, align 16

; Function Attrs: convergent willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

declare float @__ocml_exp2_f32(float)

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3) #1 !dbg !4 !reqd_work_group_size !37 {
  %kdma.tid = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !5
  %kdma.wave = and i32 %kdma.tid, 960, !dbg !5
  %kdma.lane = lshr i32 %kdma.tid, 1, !dbg !5
  %5 = and i32 %kdma.lane, 31, !dbg !5
  %kdma.owner = or disjoint i32 %kdma.wave, %5, !dbg !5
  %kdma.owner.hi = or disjoint i32 %kdma.owner, 32, !dbg !5
  %6 = shl nuw nsw i32 %kdma.tid, 3, !dbg !5
  %kdma.intra = and i32 %6, 8, !dbg !5
  %kdma.intra.hi = xor i32 %kdma.intra, 8, !dbg !5
  %7 = call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !5
  %8 = sext i32 %7 to i64, !dbg !5
  %9 = call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !5
  %10 = sext i32 %9 to i64, !dbg !5
  %11 = shl nsw i64 %10, 5, !dbg !5
  %12 = add nsw i64 %11, %8, !dbg !5
  %13 = lshr i64 %12, 5, !dbg !5
  %14 = and i64 %13, 576460752303423484, !dbg !5
  %15 = and i64 %8, 3, !dbg !5
  %16 = or disjoint i64 %14, %15, !dbg !5
  %17 = sub nsw i64 31, %16, !dbg !5
  %18 = lshr i64 %12, 2, !dbg !5
  %19 = and i64 %18, 31, !dbg !5
  %20 = call i32 @llvm.amdgcn.workgroup.id.z(), !dbg !5
  %21 = sext i32 %20 to i64, !dbg !5
  %22 = icmp ugt i64 %12, 1023, !dbg !8
  %23 = add nsw i64 %16, -32, !dbg !8
  %24 = select i1 %22, i64 %23, i64 %17, !dbg !8
  %25 = sdiv i64 %24, 32, !dbg !8
  %26 = sext i1 %22 to i64, !dbg !8
  %27 = xor i64 %25, %26, !dbg !8
  %28 = add nsw i64 %18, %27, !dbg !8
  %29 = and i64 %28, 31, !dbg !8
  %30 = shl i64 %17, 7, !dbg !8
  %31 = mul nsw i64 %27, -4096, !dbg !8
  %32 = add i64 %30, %31, !dbg !8
  %33 = shl nsw i64 %32, 7, !dbg !8
  %34 = zext nneg i32 %kdma.tid to i64, !dbg !8
  %.idx = shl nuw nsw i64 %29, 20, !dbg !8
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #9, !dbg !8, !range !39, !frisk.register_lifetimes !1
  %address.local = zext nneg i32 %address.thread to i64, !dbg !8
  %address.local471 = shl nuw nsw i64 %address.local, 3, !dbg !8
  %address.local472 = and i64 %address.local471, 1536, !dbg !8
  %address.local473 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local472, !dbg !8
  %address.local474 = ptrtoint ptr addrspace(3) %address.local473 to i64, !dbg !8
  %address.local475 = trunc i64 %address.local474 to i32, !dbg !8
  %35 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local475), !dbg !8
  %address.local476 = shl nuw nsw i64 %address.local, 7, !dbg !8
  %address.local477 = and i64 %address.local476, 16256, !dbg !8
  %address.local478 = lshr i64 %address.local, 4, !dbg !8
  %address.local479 = and i64 %address.local478, 8, !dbg !8
  %address.local480 = or disjoint i64 %address.local477, %address.local479, !dbg !8
  %36 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %.idx1703 = shl i64 %32, 8, !dbg !8
  %37 = add i64 %.idx, %.idx1703, !dbg !8
  %address.local481.idx = shl nuw nsw i64 %address.local480, 1, !dbg !8
  %38 = or disjoint i64 %37, %address.local481.idx, !dbg !8
  %dma.buffer.offset1701 = trunc i64 %38 to i32, !dbg !8
  %39 = insertelement <2 x i64> poison, i64 %36, i64 0, !dbg !8
  %40 = bitcast <2 x i64> %39 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc1702 = shufflevector <4 x i32> %40, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
  %41 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %35, i32 %dma.buffer.offset1701, <4 x i32> %dma.buffer.rsrc1702) #10, !dbg !8
  %address.local482 = or disjoint i64 %address.local, 256, !dbg !8
  %address.local483 = shl nuw nsw i64 %address.local482, 3, !dbg !8
  %address.local484 = and i64 %address.local483, 3584, !dbg !8
  %address.local485 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local484, !dbg !8
  %address.local486 = ptrtoint ptr addrspace(3) %address.local485 to i64, !dbg !8
  %address.local487 = trunc i64 %address.local486 to i32, !dbg !8
  %42 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local487), !dbg !8
  %address.local490 = lshr i64 %address.local482, 4, !dbg !8
  %address.local491 = and i64 %address.local490, 24, !dbg !8
  %address.local492 = or disjoint i64 %address.local477, %address.local491, !dbg !8
  %address.local493.idx = shl nuw nsw i64 %address.local492, 1, !dbg !8
  %43 = or disjoint i64 %37, %address.local493.idx, !dbg !8
  %dma.buffer.offset1699 = trunc i64 %43 to i32, !dbg !8
  %44 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %42, i32 %dma.buffer.offset1699, <4 x i32> %dma.buffer.rsrc1702) #10, !dbg !8
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %45 = shl nuw nsw i64 %29, 19, !dbg !8
  %46 = add i64 %45, %33, !dbg !8
  %address.local494 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local472, !dbg !8
  %address.local495 = ptrtoint ptr addrspace(3) %address.local494 to i64, !dbg !8
  %address.local496 = trunc i64 %address.local495 to i32, !dbg !8
  %47 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local496), !dbg !8
  %48 = or disjoint i64 %46, 32, !dbg !8
  %49 = or disjoint i64 %address.local480, %48, !dbg !8
  %.tr = trunc i64 %49 to i32, !dbg !8
  %dma.buffer.offset1697 = shl i32 %.tr, 1, !dbg !8
  %50 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %47, i32 %dma.buffer.offset1697, <4 x i32> %dma.buffer.rsrc1702) #10, !dbg !8
  %address.local498 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local484, !dbg !8
  %address.local499 = ptrtoint ptr addrspace(3) %address.local498 to i64, !dbg !8
  %address.local500 = trunc i64 %address.local499 to i32, !dbg !8
  %51 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local500), !dbg !8
  %52 = or disjoint i64 %address.local492, %48, !dbg !8
  %.tr1707 = trunc i64 %52 to i32, !dbg !8
  %dma.buffer.offset1695 = shl i32 %.tr1707, 1, !dbg !8
  %53 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %51, i32 %dma.buffer.offset1695, <4 x i32> %dma.buffer.rsrc1702) #10, !dbg !8
  %54 = and i64 %34, 240, !dbg !18
  %address.local502 = lshr i32 %address.thread, 2, !dbg !12
  %address.local503 = and i32 %address.local502, 4, !dbg !12
  %address.local504 = shl nuw nsw i32 %address.thread, 1, !dbg !12
  %address.local505 = and i32 %address.local504, 384, !dbg !12
  %address.local506 = or disjoint i32 %address.local503, %address.local505, !dbg !12
  %address.local507 = shl nuw nsw i32 %address.thread, 3, !dbg !12
  %address.local508 = and i32 %address.local507, 120, !dbg !12
  %address.local509 = or disjoint i32 %address.local506, %address.local508, !dbg !12
  %address.local510 = shl nuw nsw i32 %address.thread, 5, !dbg !12
  %address.local511 = and i32 %address.local510, 1024, !dbg !12
  %address.local512 = or disjoint i32 %address.local509, %address.local511, !dbg !12
  %address.local513 = zext nneg i32 %address.local512 to i64, !dbg !12
  %address.local514 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local513, !dbg !12
  %55 = load <4 x half>, ptr addrspace(3) %address.local514, align 8, !dbg !12
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local514, i64 4096, !dbg !12
  %56 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !12
  %57 = fmul <4 x half> %55, splat (half 0xH3015), !dbg !12
  %lds.const.gep1524 = getelementptr i8, ptr addrspace(3) %address.local514, i64 1024, !dbg !12
  %58 = load <4 x half>, ptr addrspace(3) %lds.const.gep1524, align 8, !dbg !12
  %59 = fmul <4 x half> %56, splat (half 0xH3015), !dbg !12
  %lds.const.gep1525 = getelementptr i8, ptr addrspace(3) %address.local514, i64 5120, !dbg !12
  %60 = load <4 x half>, ptr addrspace(3) %lds.const.gep1525, align 8, !dbg !12
  %61 = fmul <4 x half> %58, splat (half 0xH3015), !dbg !12
  %62 = fmul <4 x half> %60, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %63 = or disjoint i64 %46, 64, !dbg !8
  %64 = or disjoint i64 %address.local480, %63, !dbg !8
  %.tr1709 = trunc i64 %64 to i32, !dbg !8
  %dma.buffer.offset1693 = shl i32 %.tr1709, 1, !dbg !8
  %65 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %35, i32 %dma.buffer.offset1693, <4 x i32> %dma.buffer.rsrc1702) #10, !dbg !8
  %66 = or disjoint i64 %address.local492, %63, !dbg !8
  %.tr1711 = trunc i64 %66 to i32, !dbg !8
  %dma.buffer.offset1691 = shl i32 %.tr1711, 1, !dbg !8
  %67 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %42, i32 %dma.buffer.offset1691, <4 x i32> %dma.buffer.rsrc1702) #10, !dbg !8
  %lds.const.gep1526 = getelementptr i8, ptr addrspace(3) %address.local514, i64 8192, !dbg !12
  %68 = load <4 x half>, ptr addrspace(3) %lds.const.gep1526, align 8, !dbg !12
  %lds.const.gep1527 = getelementptr i8, ptr addrspace(3) %address.local514, i64 12288, !dbg !12
  %69 = load <4 x half>, ptr addrspace(3) %lds.const.gep1527, align 8, !dbg !12
  %lds.const.gep1528 = getelementptr i8, ptr addrspace(3) %address.local514, i64 9216, !dbg !12
  %70 = load <4 x half>, ptr addrspace(3) %lds.const.gep1528, align 8, !dbg !12
  %71 = fmul <4 x half> %68, splat (half 0xH3015), !dbg !12
  %lds.const.gep1529 = getelementptr i8, ptr addrspace(3) %address.local514, i64 13312, !dbg !12
  %72 = load <4 x half>, ptr addrspace(3) %lds.const.gep1529, align 8, !dbg !12
  %73 = fmul <4 x half> %69, splat (half 0xH3015), !dbg !12
  %74 = fmul <4 x half> %70, splat (half 0xH3015), !dbg !12
  %75 = fmul <4 x half> %72, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %76 = or disjoint i64 %46, 96, !dbg !8
  %77 = or disjoint i64 %address.local480, %76, !dbg !8
  %.tr1713 = trunc i64 %77 to i32, !dbg !8
  %dma.buffer.offset1689 = shl i32 %.tr1713, 1, !dbg !8
  %78 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %47, i32 %dma.buffer.offset1689, <4 x i32> %dma.buffer.rsrc1702) #10, !dbg !8
  %79 = or disjoint i64 %address.local492, %76, !dbg !8
  %.tr1715 = trunc i64 %79 to i32, !dbg !8
  %dma.buffer.offset1687 = shl i32 %.tr1715, 1, !dbg !8
  %80 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %51, i32 %dma.buffer.offset1687, <4 x i32> %dma.buffer.rsrc1702) #10, !dbg !8
  %81 = load <4 x half>, ptr addrspace(3) %address.local514, align 8, !dbg !12
  %82 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !12
  %83 = load <4 x half>, ptr addrspace(3) %lds.const.gep1524, align 8, !dbg !12
  %84 = fmul <4 x half> %81, splat (half 0xH3015), !dbg !12
  %85 = load <4 x half>, ptr addrspace(3) %lds.const.gep1525, align 8, !dbg !12
  %86 = fmul <4 x half> %82, splat (half 0xH3015), !dbg !12
  %87 = fmul <4 x half> %83, splat (half 0xH3015), !dbg !12
  %88 = fmul <4 x half> %85, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %89 = load <4 x half>, ptr addrspace(3) %lds.const.gep1526, align 8, !dbg !12
  %90 = load <4 x half>, ptr addrspace(3) %lds.const.gep1527, align 8, !dbg !12
  %91 = load <4 x half>, ptr addrspace(3) %lds.const.gep1528, align 8, !dbg !12
  %92 = fmul <4 x half> %89, splat (half 0xH3015), !dbg !12
  %93 = load <4 x half>, ptr addrspace(3) %lds.const.gep1529, align 8, !dbg !12
  %94 = fmul <4 x half> %90, splat (half 0xH3015), !dbg !12
  %95 = fmul <4 x half> %91, splat (half 0xH3015), !dbg !12
  %96 = fmul <4 x half> %93, splat (half 0xH3015), !dbg !12
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %97 = sub nsw i64 32, %16, !dbg !14
  %98 = shl nuw nsw i64 %19, 12, !dbg !15
  %99 = shl nsw i64 %21, 17, !dbg !15
  %100 = shl nuw nsw i64 %19, 7, !dbg !15
  %101 = shl nsw i64 %21, 12, !dbg !15
  %102 = or disjoint i64 %100, %101, !dbg !15
  %103 = shl nuw nsw i64 %34, 2, !dbg !16
  %104 = and i64 %103, 3084, !dbg !16
  %105 = and i32 %kdma.tid, 1, !dbg !16
  %index.compact411 = zext nneg i32 %105 to i64, !dbg !16
  %106 = or disjoint i64 %104, %index.compact411, !dbg !16
  %107 = add nuw nsw i64 %106, 1, !dbg !16
  %108 = lshr i64 %107, 1, !dbg !16
  %109 = add i64 %30, %34, !dbg !18
  %110 = lshr i64 %34, 2, !dbg !18
  %111 = and i64 %110, 48, !dbg !18
  %112 = add i64 %109, %111, !dbg !18
  %113 = sub i64 %112, %54, !dbg !18
  %114 = and i64 %34, 48, !dbg !18
  %115 = add i64 %113, 1, !dbg !19
  %116 = add i64 %113, 65, !dbg !19
  br label %117, !dbg !7

117:                                              ; preds = %126, %4
  %118 = phi i64 [ %1268, %126 ], [ 0, %4 ], !dbg !7
  %119 = phi [2 x <8 x float>] [ %1253, %126 ], [ zeroinitializer, %4 ], !dbg !7
  %120 = phi [2 x <1 x float>] [ %1019, %126 ], [ zeroinitializer, %4 ], !dbg !7
  %121 = phi [2 x <1 x float>] [ %1255, %126 ], [ [<1 x float> splat (float 0xFFF0000000000000), <1 x float> splat (float 0xFFF0000000000000)], %4 ], !dbg !7
  %122 = phi [2 x <8 x float>] [ %1259, %126 ], [ zeroinitializer, %4 ], !dbg !7
  %123 = phi [2 x <8 x float>] [ %1263, %126 ], [ zeroinitializer, %4 ], !dbg !7
  %124 = phi [2 x <8 x float>] [ %1267, %126 ], [ zeroinitializer, %4 ], !dbg !7
  %125 = icmp slt i64 %118, %97, !dbg !7
  br i1 %125, label %126, label %1269, !dbg !7

126:                                              ; preds = %117
  %127 = lshr i64 %118, 12, !dbg !15
  %128 = add nuw nsw i64 %127, %18, !dbg !15
  %129 = lshr i64 %118, 5, !dbg !15
  %130 = and i64 %129, 127, !dbg !15
  %131 = shl nsw i64 %118, 7, !dbg !15
  %132 = shl i64 %128, 19, !dbg !15
  %133 = and i64 %132, 16252928, !dbg !15
  %134 = shl i64 %129, 19, !dbg !15
  %135 = shl i64 %118, 14, !dbg !15
  %136 = sub i64 %135, %134, !dbg !15
  %address.thread565 = call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #9, !dbg !15, !range !39, !frisk.register_lifetimes !1
  %address.local566 = zext nneg i32 %address.thread565 to i64, !dbg !8
  %address.local567 = shl nuw nsw i64 %address.local566, 3, !dbg !8
  %address.local568 = and i64 %address.local567, 1536, !dbg !8
  %address.local569 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local568, !dbg !8
  %address.local570 = ptrtoint ptr addrspace(3) %address.local569 to i64, !dbg !8
  %address.local571 = trunc i64 %address.local570 to i32, !dbg !8
  %137 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local571), !dbg !15
  %138 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !15
  %.idx17161719 = or disjoint i64 %133, %130, !dbg !15
  %139 = add i64 %.idx17161719, %136, !dbg !15
  %140 = insertelement <2 x i64> poison, i64 %138, i64 0, !dbg !15
  %141 = bitcast <2 x i64> %140 to <4 x i32>, !dbg !15
  %dma.buffer.rsrc1686 = shufflevector <4 x i32> %141, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %kdma.source = zext nneg i32 %kdma.owner to i64, !dbg !15
  %kdma.source1859 = lshr i64 %kdma.source, 4, !dbg !15
  %kdma.source1860 = and i64 %kdma.source1859, 8, !dbg !15
  %kdma.source1861 = and i64 %kdma.source, 91, !dbg !15
  %kdma.source1862 = lshr i64 %kdma.source, 2, !dbg !15
  %kdma.source1863 = and i64 %kdma.source1862, 4, !dbg !15
  %kdma.source1864 = add nuw nsw i64 %kdma.source1863, %kdma.source, !dbg !15
  %kdma.source1865 = and i64 %kdma.source1864, 4, !dbg !15
  %kdma.source1866 = or disjoint i64 %kdma.source1861, %kdma.source1865, !dbg !15
  %kdma.source1867 = shl nuw nsw i64 %kdma.source1866, 7, !dbg !15
  %kdma.source1868 = or disjoint i64 %kdma.source1860, %kdma.source1867, !dbg !15
  %kdma.source1869 = add nuw nsw i64 %139, %kdma.source1868, !dbg !15
  %kdma.source1870 = trunc i64 %kdma.source1869 to i32, !dbg !15
  %kdma.source1871 = shl i32 %kdma.source1870, 1, !dbg !15
  %kdma.offset = add i32 %kdma.source1871, %kdma.intra, !dbg !15
  %142 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %137, i32 %kdma.offset, <4 x i32> %dma.buffer.rsrc1686) #10, !dbg !15
  %kdma.source1872 = zext nneg i32 %kdma.owner.hi to i64, !dbg !15
  %kdma.source1873 = lshr i64 %kdma.source1872, 4, !dbg !15
  %kdma.source1874 = and i64 %kdma.source1873, 8, !dbg !15
  %kdma.source1875 = and i64 %kdma.source1872, 123, !dbg !15
  %kdma.source1876 = lshr i64 %kdma.source1872, 2, !dbg !15
  %kdma.source1877 = and i64 %kdma.source1876, 4, !dbg !15
  %kdma.source1878 = add nuw nsw i64 %kdma.source1877, %kdma.source1872, !dbg !15
  %kdma.source1879 = and i64 %kdma.source1878, 4, !dbg !15
  %kdma.source1880 = or disjoint i64 %kdma.source1875, %kdma.source1879, !dbg !15
  %kdma.source1881 = shl nuw nsw i64 %kdma.source1880, 7, !dbg !15
  %kdma.source1882 = or disjoint i64 %kdma.source1874, %kdma.source1881, !dbg !15
  %kdma.source1883 = add i64 %139, %kdma.source1882, !dbg !15
  %kdma.source1884 = trunc i64 %kdma.source1883 to i32, !dbg !15
  %kdma.source1885 = shl i32 %kdma.source1884, 1, !dbg !15
  %kdma.offset1886 = add i32 %kdma.source1885, %kdma.intra.hi, !dbg !15
  %kdma.lds1887 = add i32 %137, 512, !dbg !15
  %143 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1887, i32 %kdma.offset1886, <4 x i32> %dma.buffer.rsrc1686) #10, !dbg !15
  %address.local583 = or disjoint i64 %address.local566, 256, !dbg !8
  %address.local584 = shl nuw nsw i64 %address.local583, 3, !dbg !8
  %address.local585 = and i64 %address.local584, 3584, !dbg !8
  %address.local586 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local585, !dbg !8
  %address.local587 = ptrtoint ptr addrspace(3) %address.local586 to i64, !dbg !8
  %address.local588 = trunc i64 %address.local587 to i32, !dbg !8
  %144 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local588), !dbg !15
  %kdma.source1899 = or disjoint i64 %kdma.source1868, 16, !dbg !15
  %kdma.source1900 = add nuw nsw i64 %139, %kdma.source1899, !dbg !15
  %kdma.source1901 = trunc i64 %kdma.source1900 to i32, !dbg !15
  %kdma.source1902 = shl i32 %kdma.source1901, 1, !dbg !15
  %kdma.offset1903 = add i32 %kdma.source1902, %kdma.intra, !dbg !15
  %145 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %144, i32 %kdma.offset1903, <4 x i32> %dma.buffer.rsrc1686) #10, !dbg !15
  %kdma.source1916 = or disjoint i64 %kdma.source1882, 16, !dbg !15
  %kdma.source1917 = add i64 %139, %kdma.source1916, !dbg !15
  %kdma.source1918 = trunc i64 %kdma.source1917 to i32, !dbg !15
  %kdma.source1919 = shl i32 %kdma.source1918, 1, !dbg !15
  %kdma.offset1920 = add i32 %kdma.source1919, %kdma.intra.hi, !dbg !15
  %kdma.lds1921 = add i32 %144, 512, !dbg !15
  %146 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1921, i32 %kdma.offset1920, <4 x i32> %dma.buffer.rsrc1686) #10, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %147 = add nuw i64 %118, %98, !dbg !15
  %148 = add i64 %147, %99, !dbg !15
  %149 = add i64 %148, 1024, !dbg !15
  %150 = icmp slt i64 %149, 0, !dbg !15
  %151 = sub i64 -1025, %148, !dbg !15
  %152 = select i1 %150, i64 %151, i64 %149, !dbg !15
  %153 = sdiv i64 %152, 4096, !dbg !15
  %.lobit37 = ashr i64 %149, 63, !dbg !15
  %154 = xor i64 %153, %.lobit37, !dbg !15
  %155 = add nsw i64 %129, %102, !dbg !15
  %156 = add nsw i64 %155, 32, !dbg !15
  %157 = icmp slt i64 %155, -32, !dbg !15
  %158 = sub nsw i64 -33, %155, !dbg !15
  %159 = select i1 %157, i64 %158, i64 %156, !dbg !15
  %160 = sdiv i64 %159, 128, !dbg !15
  %161 = sext i1 %157 to i64, !dbg !15
  %162 = xor i64 %160, %161, !dbg !15
  %163 = shl nsw i64 %162, 7, !dbg !15
  %164 = sub i64 %102, %163, !dbg !15
  %165 = shl i64 %154, 19, !dbg !15
  %166 = and i64 %165, 16252928, !dbg !15
  %address.local600 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local568, !dbg !8
  %address.local601 = ptrtoint ptr addrspace(3) %address.local600 to i64, !dbg !8
  %address.local602 = trunc i64 %address.local601 to i32, !dbg !8
  %167 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local602), !dbg !15
  %.idx17261730 = add nuw nsw i64 %166, %129, !dbg !15
  %168 = add i64 %.idx17261730, %164, !dbg !15
  %169 = add i64 %168, %136, !dbg !15
  %kdma.source1933 = add i64 %169, %kdma.source1868, !dbg !15
  %kdma.source1934 = trunc i64 %kdma.source1933 to i32, !dbg !15
  %kdma.source1935 = shl i32 %kdma.source1934, 1, !dbg !15
  %kdma.source1936 = add i32 %kdma.source1935, 64, !dbg !15
  %kdma.offset1937 = add i32 %kdma.source1936, %kdma.intra, !dbg !15
  %170 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %167, i32 %kdma.offset1937, <4 x i32> %dma.buffer.rsrc1686) #10, !dbg !15
  %kdma.source1950 = add i64 %169, %kdma.source1882, !dbg !15
  %kdma.source1951 = trunc i64 %kdma.source1950 to i32, !dbg !15
  %kdma.source1952 = shl i32 %kdma.source1951, 1, !dbg !15
  %kdma.source1953 = add i32 %kdma.source1952, 64, !dbg !15
  %kdma.offset1954 = add i32 %kdma.source1953, %kdma.intra.hi, !dbg !15
  %kdma.lds1955 = add i32 %167, 512, !dbg !15
  %171 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1955, i32 %kdma.offset1954, <4 x i32> %dma.buffer.rsrc1686) #10, !dbg !15
  %address.local604 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local585, !dbg !8
  %address.local605 = ptrtoint ptr addrspace(3) %address.local604 to i64, !dbg !8
  %address.local606 = trunc i64 %address.local605 to i32, !dbg !8
  %172 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local606), !dbg !15
  %kdma.source1968 = add i64 %169, %kdma.source1899, !dbg !15
  %kdma.source1969 = trunc i64 %kdma.source1968 to i32, !dbg !15
  %kdma.source1970 = shl i32 %kdma.source1969, 1, !dbg !15
  %kdma.source1971 = add i32 %kdma.source1970, 64, !dbg !15
  %kdma.offset1972 = add i32 %kdma.source1971, %kdma.intra, !dbg !15
  %173 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %172, i32 %kdma.offset1972, <4 x i32> %dma.buffer.rsrc1686) #10, !dbg !15
  %kdma.source1986 = add i64 %169, %kdma.source1916, !dbg !15
  %kdma.source1987 = trunc i64 %kdma.source1986 to i32, !dbg !15
  %kdma.source1988 = shl i32 %kdma.source1987, 1, !dbg !15
  %kdma.source1989 = add i32 %kdma.source1988, 64, !dbg !15
  %kdma.offset1990 = add i32 %kdma.source1989, %kdma.intra.hi, !dbg !15
  %kdma.lds1991 = add i32 %172, 512, !dbg !15
  %174 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1991, i32 %kdma.offset1990, <4 x i32> %dma.buffer.rsrc1686) #10, !dbg !15
  %address.local609 = lshr i64 %address.local566, 2, !dbg !16
  %address.local612 = shl nuw nsw i64 %address.local566, 1, !dbg !16
  %address.local613 = and i64 %address.local612, 24, !dbg !16
  %address.local615 = shl i64 %address.local566, 62, !dbg !16
  %address.local616 = ashr exact i64 %address.local615, 57, !dbg !16
  %address.local617 = and i64 %address.local616, -64, !dbg !16
  %address.local620 = and i64 %address.local609, 12, !dbg !12
  %address.local622 = and i64 %address.local566, 63, !dbg !12
  %address.local623 = lshr i64 %address.local622, 5, !dbg !12
  %address.local624 = mul nuw nsw i64 %address.local623, 1016, !dbg !12
  %.idx1738 = mul nuw nsw i64 %address.local566, 320, !dbg !16
  %175 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %.idx1738, !dbg !16
  %.idx1739 = mul nsw i64 %address.local609, -1280, !dbg !16
  %176 = getelementptr i8, ptr addrspace(3) %175, i64 %.idx1739, !dbg !16
  %177 = getelementptr [2 x i8], ptr addrspace(3) %176, i64 %address.local613, !dbg !16
  %178 = getelementptr [2 x i8], ptr addrspace(3) %177, i64 %address.local617, !dbg !16
  %179 = getelementptr [2 x i8], ptr addrspace(3) %178, i64 %address.local620, !dbg !16
  %address.local626 = getelementptr [2 x i8], ptr addrspace(3) %179, i64 %address.local624, !dbg !16
  %180 = add nsw i64 %.idx1738, %.idx1739, !dbg !16
  %.idx2258 = shl nuw nsw i64 %address.local613, 1, !dbg !16
  %181 = or disjoint i64 %180, %.idx2258, !dbg !16
  %182 = or disjoint i64 %address.local617, %address.local620, !dbg !16
  %183 = shl nsw i64 %182, 1, !dbg !16
  %184 = add nsw i64 %183, %181, !dbg !16
  %address.local626.idx = mul nuw nsw i64 %address.local623, 2032, !dbg !16
  %185 = add nsw i64 %184, %address.local626.idx, !dbg !16
  %186 = lshr i64 %185, 6, !dbg !16
  %187 = and i64 %186, 8, !dbg !16
  %188 = xor i64 %185, %187, !dbg !16
  %189 = sub i64 %188, %185, !dbg !16
  %kdma.swizzled = getelementptr i8, ptr addrspace(3) %address.local626, i64 %189, !dbg !16
  %190 = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8, !dbg !16
  %191 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %190, <4 x float> zeroinitializer) #9, !dbg !16
  %kdma.gep = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4096, !dbg !16
  %192 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8, !dbg !16
  %address.local637 = shl nuw nsw i32 %address.thread565, 1, !dbg !16
  %address.local638 = and i32 %address.local637, 24, !dbg !16
  %address.local639 = shl nuw nsw i32 %address.thread565, 8, !dbg !16
  %address.local640 = and i32 %address.local639, 768, !dbg !16
  %address.local641 = or disjoint i32 %address.local638, %address.local640, !dbg !16
  %address.local642 = shl nuw nsw i32 %address.thread565, 5, !dbg !16
  %address.local643 = and i32 %address.local642, 96, !dbg !16
  %address.local644 = or disjoint i32 %address.local641, %address.local643, !dbg !16
  %address.local645 = zext nneg i32 %address.local644 to i64, !dbg !16
  %address.base = call i64 asm sideeffect "", "=v,0"(i64 %108) #9, !dbg !16, !frisk.register_lifetimes !1
  %address.local646 = lshr i32 %address.thread565, 1, !dbg !16
  %address.local647 = and i32 %address.local646, 1, !dbg !16
  %address.local648 = zext nneg i32 %address.local647 to i64, !dbg !16
  %address.local649 = add nsw i64 %address.base, %address.local648, !dbg !16
  %address.local650 = shl i64 %address.local649, 6, !dbg !16
  %address.local651 = sub i64 %address.local645, %address.local650, !dbg !16
  %193 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local651, !dbg !16
  %194 = getelementptr [2 x i8], ptr addrspace(3) %193, i64 %address.local620, !dbg !16
  %195 = getelementptr [2 x i8], ptr addrspace(3) %194, i64 %address.local624, !dbg !16
  %address.local655 = getelementptr i8, ptr addrspace(3) %195, i64 64, !dbg !16
  %.idx22612264 = add i64 %address.local651, %address.local620, !dbg !16
  %196 = shl i64 %.idx22612264, 1, !dbg !16
  %197 = add i64 %196, %address.local626.idx, !dbg !16
  %198 = add i64 %197, 64, !dbg !16
  %199 = lshr i64 %198, 6, !dbg !16
  %200 = and i64 %199, 8, !dbg !16
  %201 = xor i64 %198, %200, !dbg !16
  %202 = sub i64 %201, %198, !dbg !16
  %kdma.swizzled2132 = getelementptr i8, ptr addrspace(3) %address.local655, i64 %202, !dbg !16
  %203 = load <4 x half>, ptr addrspace(3) %kdma.swizzled2132, align 8, !dbg !16, !frisk.k_lds_pair !42
  %kdma.gep2136 = getelementptr i8, ptr addrspace(3) %kdma.swizzled2132, i64 128, !dbg !16
  %204 = load <4 x half>, ptr addrspace(3) %kdma.gep2136, align 8, !dbg !16, !frisk.k_lds_pair !42
  %kdma.gep2133 = getelementptr i8, ptr addrspace(3) %kdma.swizzled2132, i64 4096, !dbg !16
  %205 = load <4 x half>, ptr addrspace(3) %kdma.gep2133, align 8, !dbg !16, !frisk.k_lds_pair !43
  %kdma.gep2137 = getelementptr i8, ptr addrspace(3) %kdma.swizzled2132, i64 4224, !dbg !16
  %206 = load <4 x half>, ptr addrspace(3) %kdma.gep2137, align 8, !dbg !16, !frisk.k_lds_pair !43
  %kdma.gep2134 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 128, !dbg !16
  %207 = load <4 x half>, ptr addrspace(3) %kdma.gep2134, align 8, !dbg !16, !frisk.k_lds_pair !44
  %kdma.gep2138 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1024, !dbg !16
  %208 = load <4 x half>, ptr addrspace(3) %kdma.gep2138, align 8, !dbg !16, !frisk.k_lds_pair !44
  %kdma.gep2135 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4224, !dbg !16
  %209 = load <4 x half>, ptr addrspace(3) %kdma.gep2135, align 8, !dbg !16, !frisk.k_lds_pair !45
  %kdma.gep2139 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5120, !dbg !16
  %210 = load <4 x half>, ptr addrspace(3) %kdma.gep2139, align 8, !dbg !16, !frisk.k_lds_pair !45
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16, !frisk.k_lds_prefetch !1
  %211 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %192, <4 x float> %191) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16, !frisk.k_lds_prefetch !1
  %kdma.gep2140 = getelementptr i8, ptr addrspace(3) %kdma.swizzled2132, i64 1024, !dbg !16
  %212 = load <4 x half>, ptr addrspace(3) %kdma.gep2140, align 8, !dbg !16, !frisk.k_lds_pair !40
  %kdma.gep2144 = getelementptr i8, ptr addrspace(3) %kdma.swizzled2132, i64 1152, !dbg !16
  %213 = load <4 x half>, ptr addrspace(3) %kdma.gep2144, align 8, !dbg !16, !frisk.k_lds_pair !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16, !frisk.k_lds_prefetch !1
  %214 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %203, <4 x float> zeroinitializer) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16, !frisk.k_lds_prefetch !1
  %kdma.gep2141 = getelementptr i8, ptr addrspace(3) %kdma.swizzled2132, i64 5120, !dbg !16
  %215 = load <4 x half>, ptr addrspace(3) %kdma.gep2141, align 8, !dbg !16, !frisk.k_lds_pair !46
  %kdma.gep2145 = getelementptr i8, ptr addrspace(3) %kdma.swizzled2132, i64 5248, !dbg !16
  %216 = load <4 x half>, ptr addrspace(3) %kdma.gep2145, align 8, !dbg !16, !frisk.k_lds_pair !46
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16, !frisk.k_lds_prefetch !1
  %217 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %205, <4 x float> %214) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16, !frisk.k_lds_prefetch !1
  %kdma.gep2142 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1152, !dbg !16
  %218 = load <4 x half>, ptr addrspace(3) %kdma.gep2142, align 8, !dbg !16, !frisk.k_lds_pair !47
  %kdma.gep2143 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5248, !dbg !16
  %219 = load <4 x half>, ptr addrspace(3) %kdma.gep2143, align 8, !dbg !16, !frisk.k_lds_pair !47
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16, !frisk.k_lds_prefetch !1
  %220 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %207, <4 x float> zeroinitializer) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16, !frisk.k_lds_prefetch !1
  %221 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %209, <4 x float> %220) #9, !dbg !16
  %222 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %204, <4 x float> zeroinitializer) #9, !dbg !16
  %223 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %206, <4 x float> %222) #9, !dbg !16
  %224 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %208, <4 x float> zeroinitializer) #9, !dbg !16
  %225 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %210, <4 x float> %224) #9, !dbg !16
  %226 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %212, <4 x float> zeroinitializer) #9, !dbg !16
  %227 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %215, <4 x float> %226) #9, !dbg !16
  %228 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %218, <4 x float> zeroinitializer) #9, !dbg !16
  %229 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %219, <4 x float> %228) #9, !dbg !16
  %230 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %213, <4 x float> zeroinitializer) #9, !dbg !16
  %231 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %216, <4 x float> %230) #9, !dbg !16
  %232 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %190, <4 x float> zeroinitializer) #9, !dbg !16
  %233 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %192, <4 x float> %232) #9, !dbg !16
  %234 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %203, <4 x float> zeroinitializer) #9, !dbg !16
  %235 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %205, <4 x float> %234) #9, !dbg !16
  %236 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %207, <4 x float> zeroinitializer) #9, !dbg !16
  %237 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %209, <4 x float> %236) #9, !dbg !16
  %238 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %204, <4 x float> zeroinitializer) #9, !dbg !16
  %239 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %206, <4 x float> %238) #9, !dbg !16
  %240 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %208, <4 x float> zeroinitializer) #9, !dbg !16
  %241 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %210, <4 x float> %240) #9, !dbg !16
  %242 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %212, <4 x float> zeroinitializer) #9, !dbg !16
  %243 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %215, <4 x float> %242) #9, !dbg !16
  %244 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %218, <4 x float> zeroinitializer) #9, !dbg !16
  %245 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %219, <4 x float> %244) #9, !dbg !16
  %246 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %213, <4 x float> zeroinitializer) #9, !dbg !16
  %247 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %216, <4 x float> %246) #9, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %248 = add i64 %148, 2048, !dbg !15
  %249 = icmp slt i64 %248, 0, !dbg !15
  %250 = sub i64 -2049, %148, !dbg !15
  %251 = select i1 %249, i64 %250, i64 %248, !dbg !15
  %252 = sdiv i64 %251, 4096, !dbg !15
  %.lobit38 = ashr i64 %248, 63, !dbg !15
  %253 = xor i64 %252, %.lobit38, !dbg !15
  %254 = add nsw i64 %155, 64, !dbg !15
  %255 = icmp slt i64 %155, -64, !dbg !15
  %256 = sub nsw i64 -65, %155, !dbg !15
  %257 = select i1 %255, i64 %256, i64 %254, !dbg !15
  %258 = sdiv i64 %257, 128, !dbg !15
  %259 = sext i1 %255 to i64, !dbg !15
  %260 = xor i64 %258, %259, !dbg !15
  %261 = shl nsw i64 %260, 7, !dbg !15
  %262 = sub i64 %102, %261, !dbg !15
  %263 = shl i64 %253, 19, !dbg !15
  %264 = and i64 %263, 16252928, !dbg !15
  %.idx17401744 = add nuw nsw i64 %264, %129, !dbg !15
  %265 = add i64 %.idx17401744, %262, !dbg !15
  %266 = add i64 %265, %136, !dbg !15
  %kdma.source2003 = add i64 %266, %kdma.source1868, !dbg !15
  %kdma.source2004 = trunc i64 %kdma.source2003 to i32, !dbg !15
  %kdma.source2005 = shl i32 %kdma.source2004, 1, !dbg !15
  %kdma.source2006 = add i32 %kdma.source2005, 128, !dbg !15
  %kdma.offset2007 = add i32 %kdma.source2006, %kdma.intra, !dbg !15
  %267 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %137, i32 %kdma.offset2007, <4 x i32> %dma.buffer.rsrc1686) #10, !dbg !15
  %kdma.source2020 = add i64 %266, %kdma.source1882, !dbg !15
  %kdma.source2021 = trunc i64 %kdma.source2020 to i32, !dbg !15
  %kdma.source2022 = shl i32 %kdma.source2021, 1, !dbg !15
  %kdma.source2023 = add i32 %kdma.source2022, 128, !dbg !15
  %kdma.offset2024 = add i32 %kdma.source2023, %kdma.intra.hi, !dbg !15
  %268 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1887, i32 %kdma.offset2024, <4 x i32> %dma.buffer.rsrc1686) #10, !dbg !15
  %kdma.source2038 = add i64 %266, %kdma.source1899, !dbg !15
  %kdma.source2039 = trunc i64 %kdma.source2038 to i32, !dbg !15
  %kdma.source2040 = shl i32 %kdma.source2039, 1, !dbg !15
  %kdma.source2041 = add i32 %kdma.source2040, 128, !dbg !15
  %kdma.offset2042 = add i32 %kdma.source2041, %kdma.intra, !dbg !15
  %269 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %144, i32 %kdma.offset2042, <4 x i32> %dma.buffer.rsrc1686) #10, !dbg !15
  %kdma.source2056 = add i64 %266, %kdma.source1916, !dbg !15
  %kdma.source2057 = trunc i64 %kdma.source2056 to i32, !dbg !15
  %kdma.source2058 = shl i32 %kdma.source2057, 1, !dbg !15
  %kdma.source2059 = add i32 %kdma.source2058, 128, !dbg !15
  %kdma.offset2060 = add i32 %kdma.source2059, %kdma.intra.hi, !dbg !15
  %270 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1921, i32 %kdma.offset2060, <4 x i32> %dma.buffer.rsrc1686) #10, !dbg !15
  %kdma.gep2162 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8192, !dbg !16
  %271 = load <4 x half>, ptr addrspace(3) %kdma.gep2162, align 8, !dbg !16, !frisk.k_lds_pair !48
  %kdma.gep2166 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8320, !dbg !16
  %272 = load <4 x half>, ptr addrspace(3) %kdma.gep2166, align 8, !dbg !16, !frisk.k_lds_pair !48
  %kdma.gep2163 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12288, !dbg !16
  %273 = load <4 x half>, ptr addrspace(3) %kdma.gep2163, align 8, !dbg !16, !frisk.k_lds_pair !49
  %kdma.gep2167 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12416, !dbg !16
  %274 = load <4 x half>, ptr addrspace(3) %kdma.gep2167, align 8, !dbg !16, !frisk.k_lds_pair !49
  %kdma.gep2164 = getelementptr i8, ptr addrspace(3) %kdma.swizzled2132, i64 8192, !dbg !16
  %275 = load <4 x half>, ptr addrspace(3) %kdma.gep2164, align 8, !dbg !16, !frisk.k_lds_pair !50
  %kdma.gep2168 = getelementptr i8, ptr addrspace(3) %kdma.swizzled2132, i64 8320, !dbg !16
  %276 = load <4 x half>, ptr addrspace(3) %kdma.gep2168, align 8, !dbg !16, !frisk.k_lds_pair !50
  %kdma.gep2165 = getelementptr i8, ptr addrspace(3) %kdma.swizzled2132, i64 12288, !dbg !16
  %277 = load <4 x half>, ptr addrspace(3) %kdma.gep2165, align 8, !dbg !16, !frisk.k_lds_pair !51
  %kdma.gep2169 = getelementptr i8, ptr addrspace(3) %kdma.swizzled2132, i64 12416, !dbg !16
  %278 = load <4 x half>, ptr addrspace(3) %kdma.gep2169, align 8, !dbg !16, !frisk.k_lds_pair !51
  %kdma.gep2170 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9216, !dbg !16
  %279 = load <4 x half>, ptr addrspace(3) %kdma.gep2170, align 8, !dbg !16, !frisk.k_lds_pair !52
  %kdma.gep2174 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9344, !dbg !16
  %280 = load <4 x half>, ptr addrspace(3) %kdma.gep2174, align 8, !dbg !16, !frisk.k_lds_pair !52
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16, !frisk.k_lds_prefetch !1
  %281 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %271, <4 x float> %211) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16, !frisk.k_lds_prefetch !1
  %kdma.gep2171 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13312, !dbg !16
  %282 = load <4 x half>, ptr addrspace(3) %kdma.gep2171, align 8, !dbg !16, !frisk.k_lds_pair !53
  %kdma.gep2175 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13440, !dbg !16
  %283 = load <4 x half>, ptr addrspace(3) %kdma.gep2175, align 8, !dbg !16, !frisk.k_lds_pair !53
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16, !frisk.k_lds_prefetch !1
  %284 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %273, <4 x float> %281) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16, !frisk.k_lds_prefetch !1
  %kdma.gep2172 = getelementptr i8, ptr addrspace(3) %kdma.swizzled2132, i64 9216, !dbg !16
  %285 = load <4 x half>, ptr addrspace(3) %kdma.gep2172, align 8, !dbg !16, !frisk.k_lds_pair !54
  %kdma.gep2176 = getelementptr i8, ptr addrspace(3) %kdma.swizzled2132, i64 9344, !dbg !16
  %286 = load <4 x half>, ptr addrspace(3) %kdma.gep2176, align 8, !dbg !16, !frisk.k_lds_pair !54
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16, !frisk.k_lds_prefetch !1
  %287 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %275, <4 x float> %217) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16, !frisk.k_lds_prefetch !1
  %kdma.gep2173 = getelementptr i8, ptr addrspace(3) %kdma.swizzled2132, i64 13312, !dbg !16
  %288 = load <4 x half>, ptr addrspace(3) %kdma.gep2173, align 8, !dbg !16, !frisk.k_lds_pair !55
  %kdma.gep2177 = getelementptr i8, ptr addrspace(3) %kdma.swizzled2132, i64 13440, !dbg !16
  %289 = load <4 x half>, ptr addrspace(3) %kdma.gep2177, align 8, !dbg !16, !frisk.k_lds_pair !55
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16, !frisk.k_lds_prefetch !1
  %290 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %277, <4 x float> %287) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16, !frisk.k_lds_prefetch !1
  %291 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %272, <4 x float> %221) #9, !dbg !16
  %292 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %274, <4 x float> %291) #9, !dbg !16
  %293 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %276, <4 x float> %223) #9, !dbg !16
  %294 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %278, <4 x float> %293) #9, !dbg !16
  %295 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %279, <4 x float> %225) #9, !dbg !16
  %296 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %282, <4 x float> %295) #9, !dbg !16
  %297 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %285, <4 x float> %227) #9, !dbg !16
  %298 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %288, <4 x float> %297) #9, !dbg !16
  %299 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %280, <4 x float> %229) #9, !dbg !16
  %300 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %283, <4 x float> %299) #9, !dbg !16
  %301 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %286, <4 x float> %231) #9, !dbg !16
  %302 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %289, <4 x float> %301) #9, !dbg !16
  %303 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %271, <4 x float> %233) #9, !dbg !16
  %304 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %273, <4 x float> %303) #9, !dbg !16
  %305 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %275, <4 x float> %235) #9, !dbg !16
  %306 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %277, <4 x float> %305) #9, !dbg !16
  %307 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %272, <4 x float> %237) #9, !dbg !16
  %308 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %274, <4 x float> %307) #9, !dbg !16
  %309 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %276, <4 x float> %239) #9, !dbg !16
  %310 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %278, <4 x float> %309) #9, !dbg !16
  %311 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %279, <4 x float> %241) #9, !dbg !16
  %312 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %282, <4 x float> %311) #9, !dbg !16
  %313 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %285, <4 x float> %243) #9, !dbg !16
  %314 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %288, <4 x float> %313) #9, !dbg !16
  %315 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %280, <4 x float> %245) #9, !dbg !16
  %316 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %283, <4 x float> %315) #9, !dbg !16
  %317 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %286, <4 x float> %247) #9, !dbg !16
  %318 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %289, <4 x float> %317) #9, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %319 = add i64 %148, 3072, !dbg !15
  %320 = icmp slt i64 %319, 0, !dbg !15
  %321 = sub i64 -3073, %148, !dbg !15
  %322 = select i1 %320, i64 %321, i64 %319, !dbg !15
  %323 = sdiv i64 %322, 4096, !dbg !15
  %.lobit39 = ashr i64 %319, 63, !dbg !15
  %324 = xor i64 %323, %.lobit39, !dbg !15
  %325 = add nsw i64 %155, 96, !dbg !15
  %326 = icmp slt i64 %155, -96, !dbg !15
  %327 = sub nsw i64 -97, %155, !dbg !15
  %328 = select i1 %326, i64 %327, i64 %325, !dbg !15
  %329 = sdiv i64 %328, 128, !dbg !15
  %330 = sext i1 %326 to i64, !dbg !15
  %331 = xor i64 %329, %330, !dbg !15
  %332 = shl nsw i64 %331, 7, !dbg !15
  %333 = sub i64 %102, %332, !dbg !15
  %334 = shl i64 %324, 19, !dbg !15
  %335 = and i64 %334, 16252928, !dbg !15
  %.idx17521756 = add nuw nsw i64 %335, %129, !dbg !15
  %336 = add i64 %.idx17521756, %333, !dbg !15
  %337 = add i64 %336, %136, !dbg !15
  %kdma.source2073 = add i64 %337, %kdma.source1868, !dbg !15
  %kdma.source2074 = trunc i64 %kdma.source2073 to i32, !dbg !15
  %kdma.source2075 = shl i32 %kdma.source2074, 1, !dbg !15
  %kdma.source2076 = add i32 %kdma.source2075, 192, !dbg !15
  %kdma.offset2077 = add i32 %kdma.source2076, %kdma.intra, !dbg !15
  %338 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %167, i32 %kdma.offset2077, <4 x i32> %dma.buffer.rsrc1686) #10, !dbg !15
  %kdma.source2090 = add i64 %337, %kdma.source1882, !dbg !15
  %kdma.source2091 = trunc i64 %kdma.source2090 to i32, !dbg !15
  %kdma.source2092 = shl i32 %kdma.source2091, 1, !dbg !15
  %kdma.source2093 = add i32 %kdma.source2092, 192, !dbg !15
  %kdma.offset2094 = add i32 %kdma.source2093, %kdma.intra.hi, !dbg !15
  %339 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1955, i32 %kdma.offset2094, <4 x i32> %dma.buffer.rsrc1686) #10, !dbg !15
  %kdma.source2108 = add i64 %337, %kdma.source1899, !dbg !15
  %kdma.source2109 = trunc i64 %kdma.source2108 to i32, !dbg !15
  %kdma.source2110 = shl i32 %kdma.source2109, 1, !dbg !15
  %kdma.source2111 = add i32 %kdma.source2110, 192, !dbg !15
  %kdma.offset2112 = add i32 %kdma.source2111, %kdma.intra, !dbg !15
  %340 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %172, i32 %kdma.offset2112, <4 x i32> %dma.buffer.rsrc1686) #10, !dbg !15
  %kdma.source2126 = add i64 %337, %kdma.source1916, !dbg !15
  %kdma.source2127 = trunc i64 %kdma.source2126 to i32, !dbg !15
  %kdma.source2128 = shl i32 %kdma.source2127, 1, !dbg !15
  %kdma.source2129 = add i32 %kdma.source2128, 192, !dbg !15
  %kdma.offset2130 = add i32 %kdma.source2129, %kdma.intra.hi, !dbg !15
  %341 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1991, i32 %kdma.offset2130, <4 x i32> %dma.buffer.rsrc1686) #10, !dbg !15
  %342 = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8, !dbg !16, !frisk.k_lds_pair !56
  %343 = load <4 x half>, ptr addrspace(3) %kdma.gep2134, align 8, !dbg !16, !frisk.k_lds_pair !56
  %344 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8, !dbg !16, !frisk.k_lds_pair !57
  %345 = load <4 x half>, ptr addrspace(3) %kdma.gep2135, align 8, !dbg !16, !frisk.k_lds_pair !57
  %346 = load <4 x half>, ptr addrspace(3) %kdma.swizzled2132, align 8, !dbg !16, !frisk.k_lds_pair !58
  %347 = load <4 x half>, ptr addrspace(3) %kdma.gep2136, align 8, !dbg !16, !frisk.k_lds_pair !58
  %348 = load <4 x half>, ptr addrspace(3) %kdma.gep2133, align 8, !dbg !16, !frisk.k_lds_pair !59
  %349 = load <4 x half>, ptr addrspace(3) %kdma.gep2137, align 8, !dbg !16, !frisk.k_lds_pair !59
  %350 = load <4 x half>, ptr addrspace(3) %kdma.gep2138, align 8, !dbg !16, !frisk.k_lds_pair !60
  %351 = load <4 x half>, ptr addrspace(3) %kdma.gep2142, align 8, !dbg !16, !frisk.k_lds_pair !60
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16, !frisk.k_lds_prefetch !1
  %352 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %342, <4 x float> %284) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16, !frisk.k_lds_prefetch !1
  %353 = load <4 x half>, ptr addrspace(3) %kdma.gep2139, align 8, !dbg !16, !frisk.k_lds_pair !61
  %354 = load <4 x half>, ptr addrspace(3) %kdma.gep2143, align 8, !dbg !16, !frisk.k_lds_pair !61
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16, !frisk.k_lds_prefetch !1
  %355 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %344, <4 x float> %352) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16, !frisk.k_lds_prefetch !1
  %356 = load <4 x half>, ptr addrspace(3) %kdma.gep2140, align 8, !dbg !16, !frisk.k_lds_pair !62
  %357 = load <4 x half>, ptr addrspace(3) %kdma.gep2144, align 8, !dbg !16, !frisk.k_lds_pair !62
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16, !frisk.k_lds_prefetch !1
  %358 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %346, <4 x float> %290) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16, !frisk.k_lds_prefetch !1
  %359 = load <4 x half>, ptr addrspace(3) %kdma.gep2141, align 8, !dbg !16, !frisk.k_lds_pair !63
  %360 = load <4 x half>, ptr addrspace(3) %kdma.gep2145, align 8, !dbg !16, !frisk.k_lds_pair !63
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16, !frisk.k_lds_prefetch !1
  %361 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %348, <4 x float> %358) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16, !frisk.k_lds_prefetch !1
  %362 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %343, <4 x float> %292) #9, !dbg !16
  %363 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %345, <4 x float> %362) #9, !dbg !16
  %364 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %347, <4 x float> %294) #9, !dbg !16
  %365 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %349, <4 x float> %364) #9, !dbg !16
  %366 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %350, <4 x float> %296) #9, !dbg !16
  %367 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %353, <4 x float> %366) #9, !dbg !16
  %368 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %356, <4 x float> %298) #9, !dbg !16
  %369 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %359, <4 x float> %368) #9, !dbg !16
  %370 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %351, <4 x float> %300) #9, !dbg !16
  %371 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %354, <4 x float> %370) #9, !dbg !16
  %372 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %357, <4 x float> %302) #9, !dbg !16
  %373 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %360, <4 x float> %372) #9, !dbg !16
  %374 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %342, <4 x float> %304) #9, !dbg !16
  %375 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %344, <4 x float> %374) #9, !dbg !16
  %376 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %346, <4 x float> %306) #9, !dbg !16
  %377 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %348, <4 x float> %376) #9, !dbg !16
  %378 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %343, <4 x float> %308) #9, !dbg !16
  %379 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %345, <4 x float> %378) #9, !dbg !16
  %380 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %347, <4 x float> %310) #9, !dbg !16
  %381 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %349, <4 x float> %380) #9, !dbg !16
  %382 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %350, <4 x float> %312) #9, !dbg !16
  %383 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %353, <4 x float> %382) #9, !dbg !16
  %384 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %356, <4 x float> %314) #9, !dbg !16
  %385 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %359, <4 x float> %384) #9, !dbg !16
  %386 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %351, <4 x float> %316) #9, !dbg !16
  %387 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %354, <4 x float> %386) #9, !dbg !16
  %388 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %357, <4 x float> %318) #9, !dbg !16
  %389 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %360, <4 x float> %388) #9, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %390 = add nuw nsw i64 %129, %18, !dbg !17
  %391 = shl i64 %390, 19, !dbg !17
  %392 = and i64 %391, 16252928, !dbg !17
  %address.local984 = shl nuw nsw i64 %address.local566, 5, !dbg !17
  %address.local985 = and i64 %address.local984, 8064, !dbg !17
  %address.local987 = and i64 %address.local567, 24, !dbg !17
  %address.local988 = or disjoint i64 %address.local985, %address.local987, !dbg !17
  %393 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %.idx17641766 = add i64 %392, %136, !dbg !17
  %394 = or disjoint i64 %.idx17641766, %address.local988, !dbg !17
  %.tr1767 = trunc i64 %394 to i32, !dbg !17
  %dma.buffer.offset1669 = shl i32 %.tr1767, 1, !dbg !17
  %395 = insertelement <2 x i64> poison, i64 %393, i64 0, !dbg !17
  %396 = bitcast <2 x i64> %395 to <4 x i32>, !dbg !17
  %dma.buffer.rsrc1670 = shufflevector <4 x i32> %396, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !17
  %397 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %137, i32 %dma.buffer.offset1669, <4 x i32> %dma.buffer.rsrc1670) #10, !dbg !17
  %address.local990 = shl nuw nsw i64 %address.local583, 5, !dbg !17
  %address.local991 = and i64 %address.local990, 16256, !dbg !17
  %address.local994 = or disjoint i64 %address.local991, %address.local987, !dbg !17
  %398 = or disjoint i64 %.idx17641766, %address.local994, !dbg !17
  %.tr1771 = trunc i64 %398 to i32, !dbg !17
  %dma.buffer.offset1667 = shl i32 %.tr1771, 1, !dbg !17
  %399 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %144, i32 %dma.buffer.offset1667, <4 x i32> %dma.buffer.rsrc1670) #10, !dbg !17
  %400 = load <4 x half>, ptr addrspace(3) %kdma.gep2162, align 8, !dbg !16
  %401 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %400, <4 x float> %355) #9, !dbg !16
  %402 = or disjoint i64 %131, %114, !dbg !18
  %403 = lshr i64 %115, 32, !dbg !24
  %compare.high = trunc nuw i64 %403 to i32, !dbg !24
  %404 = lshr i64 %131, 32, !dbg !24
  %compare.high1268 = trunc nuw i64 %404 to i32, !dbg !24
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high1268, !dbg !24
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high1268, !dbg !24
  %compare.low.base = trunc i64 %115 to i32, !dbg !24
  %compare.low.base1269 = trunc i64 %402 to i32, !dbg !24
  %compare.low = icmp ugt i32 %compare.low.base, %compare.low.base1269, !dbg !24
  %compare.result = select i1 %compare.high.equal, i1 %compare.low, i1 %compare.high.order, !dbg !24
  %405 = select i1 %compare.result, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 1) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local = or i32 %compare.local.base, 1, !dbg !24
  %compare.low1270 = icmp ult i32 %compare.local, %compare.low.base, !dbg !24
  %compare.result1271 = select i1 %compare.high.equal, i1 %compare.low1270, i1 %compare.high.order, !dbg !24
  %406 = select i1 %compare.result1271, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1272 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 2) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1273 = or i32 %compare.local.base1272, 2, !dbg !24
  %compare.low1274 = icmp ult i32 %compare.local1273, %compare.low.base, !dbg !24
  %compare.result1275 = select i1 %compare.high.equal, i1 %compare.low1274, i1 %compare.high.order, !dbg !24
  %407 = select i1 %compare.result1275, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1276 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 3) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1277 = or i32 %compare.local.base1276, 3, !dbg !24
  %compare.low1278 = icmp ult i32 %compare.local1277, %compare.low.base, !dbg !24
  %compare.result1279 = select i1 %compare.high.equal, i1 %compare.low1278, i1 %compare.high.order, !dbg !24
  %408 = select i1 %compare.result1279, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1280 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 4) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1281 = or i32 %compare.local.base1280, 4, !dbg !24
  %compare.low1282 = icmp ult i32 %compare.local1281, %compare.low.base, !dbg !24
  %compare.result1283 = select i1 %compare.high.equal, i1 %compare.low1282, i1 %compare.high.order, !dbg !24
  %409 = select i1 %compare.result1283, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1284 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 5) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1285 = or i32 %compare.local.base1284, 5, !dbg !24
  %compare.low1286 = icmp ult i32 %compare.local1285, %compare.low.base, !dbg !24
  %compare.result1287 = select i1 %compare.high.equal, i1 %compare.low1286, i1 %compare.high.order, !dbg !24
  %410 = select i1 %compare.result1287, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1288 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 6) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1289 = or i32 %compare.local.base1288, 6, !dbg !24
  %compare.low1290 = icmp ult i32 %compare.local1289, %compare.low.base, !dbg !24
  %compare.result1291 = select i1 %compare.high.equal, i1 %compare.low1290, i1 %compare.high.order, !dbg !24
  %411 = select i1 %compare.result1291, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1292 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 7) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1293 = or i32 %compare.local.base1292, 7, !dbg !24
  %compare.low1294 = icmp ult i32 %compare.local1293, %compare.low.base, !dbg !24
  %compare.result1295 = select i1 %compare.high.equal, i1 %compare.low1294, i1 %compare.high.order, !dbg !24
  %412 = select i1 %compare.result1295, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1296 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 8) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1297 = or i32 %compare.local.base1296, 8, !dbg !24
  %compare.low1298 = icmp ult i32 %compare.local1297, %compare.low.base, !dbg !24
  %compare.result1299 = select i1 %compare.high.equal, i1 %compare.low1298, i1 %compare.high.order, !dbg !24
  %413 = select i1 %compare.result1299, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1300 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 9) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1301 = or i32 %compare.local.base1300, 9, !dbg !24
  %compare.low1302 = icmp ult i32 %compare.local1301, %compare.low.base, !dbg !24
  %compare.result1303 = select i1 %compare.high.equal, i1 %compare.low1302, i1 %compare.high.order, !dbg !24
  %414 = select i1 %compare.result1303, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1304 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 10) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1305 = or i32 %compare.local.base1304, 10, !dbg !24
  %compare.low1306 = icmp ult i32 %compare.local1305, %compare.low.base, !dbg !24
  %compare.result1307 = select i1 %compare.high.equal, i1 %compare.low1306, i1 %compare.high.order, !dbg !24
  %415 = select i1 %compare.result1307, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1308 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 11) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1309 = or i32 %compare.local.base1308, 11, !dbg !24
  %compare.low1310 = icmp ult i32 %compare.local1309, %compare.low.base, !dbg !24
  %compare.result1311 = select i1 %compare.high.equal, i1 %compare.low1310, i1 %compare.high.order, !dbg !24
  %416 = select i1 %compare.result1311, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1312 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 12) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1313 = or i32 %compare.local.base1312, 12, !dbg !24
  %compare.low1314 = icmp ult i32 %compare.local1313, %compare.low.base, !dbg !24
  %compare.result1315 = select i1 %compare.high.equal, i1 %compare.low1314, i1 %compare.high.order, !dbg !24
  %417 = select i1 %compare.result1315, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1316 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 13) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1317 = or i32 %compare.local.base1316, 13, !dbg !24
  %compare.low1318 = icmp ult i32 %compare.local1317, %compare.low.base, !dbg !24
  %compare.result1319 = select i1 %compare.high.equal, i1 %compare.low1318, i1 %compare.high.order, !dbg !24
  %418 = select i1 %compare.result1319, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1320 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 14) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1321 = or i32 %compare.local.base1320, 14, !dbg !24
  %compare.low1322 = icmp ult i32 %compare.local1321, %compare.low.base, !dbg !24
  %compare.result1323 = select i1 %compare.high.equal, i1 %compare.low1322, i1 %compare.high.order, !dbg !24
  %419 = select i1 %compare.result1323, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1324 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 15) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1325 = or i32 %compare.local.base1324, 15, !dbg !24
  %compare.low1326 = icmp ult i32 %compare.local1325, %compare.low.base, !dbg !24
  %compare.result1327 = select i1 %compare.high.equal, i1 %compare.low1326, i1 %compare.high.order, !dbg !24
  %420 = select i1 %compare.result1327, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1328 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 16) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1329 = or i32 %compare.local.base1328, 64, !dbg !24
  %compare.low1330 = icmp ult i32 %compare.local1329, %compare.low.base, !dbg !24
  %compare.result1331 = select i1 %compare.high.equal, i1 %compare.low1330, i1 %compare.high.order, !dbg !24
  %421 = select i1 %compare.result1331, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1332 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 17) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1333 = or i32 %compare.local.base1332, 65, !dbg !24
  %compare.low1334 = icmp ult i32 %compare.local1333, %compare.low.base, !dbg !24
  %compare.result1335 = select i1 %compare.high.equal, i1 %compare.low1334, i1 %compare.high.order, !dbg !24
  %422 = select i1 %compare.result1335, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1336 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 18) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1337 = or i32 %compare.local.base1336, 66, !dbg !24
  %compare.low1338 = icmp ult i32 %compare.local1337, %compare.low.base, !dbg !24
  %compare.result1339 = select i1 %compare.high.equal, i1 %compare.low1338, i1 %compare.high.order, !dbg !24
  %423 = select i1 %compare.result1339, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1340 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 19) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1341 = or i32 %compare.local.base1340, 67, !dbg !24
  %compare.low1342 = icmp ult i32 %compare.local1341, %compare.low.base, !dbg !24
  %compare.result1343 = select i1 %compare.high.equal, i1 %compare.low1342, i1 %compare.high.order, !dbg !24
  %424 = select i1 %compare.result1343, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1344 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 20) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1345 = or i32 %compare.local.base1344, 68, !dbg !24
  %compare.low1346 = icmp ult i32 %compare.local1345, %compare.low.base, !dbg !24
  %compare.result1347 = select i1 %compare.high.equal, i1 %compare.low1346, i1 %compare.high.order, !dbg !24
  %425 = select i1 %compare.result1347, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1348 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 21) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1349 = or i32 %compare.local.base1348, 69, !dbg !24
  %compare.low1350 = icmp ult i32 %compare.local1349, %compare.low.base, !dbg !24
  %compare.result1351 = select i1 %compare.high.equal, i1 %compare.low1350, i1 %compare.high.order, !dbg !24
  %426 = select i1 %compare.result1351, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1352 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 22) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1353 = or i32 %compare.local.base1352, 70, !dbg !24
  %compare.low1354 = icmp ult i32 %compare.local1353, %compare.low.base, !dbg !24
  %compare.result1355 = select i1 %compare.high.equal, i1 %compare.low1354, i1 %compare.high.order, !dbg !24
  %427 = select i1 %compare.result1355, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1356 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 23) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1357 = or i32 %compare.local.base1356, 71, !dbg !24
  %compare.low1358 = icmp ult i32 %compare.local1357, %compare.low.base, !dbg !24
  %compare.result1359 = select i1 %compare.high.equal, i1 %compare.low1358, i1 %compare.high.order, !dbg !24
  %428 = select i1 %compare.result1359, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1360 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 24) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1361 = or i32 %compare.local.base1360, 72, !dbg !24
  %compare.low1362 = icmp ult i32 %compare.local1361, %compare.low.base, !dbg !24
  %compare.result1363 = select i1 %compare.high.equal, i1 %compare.low1362, i1 %compare.high.order, !dbg !24
  %429 = select i1 %compare.result1363, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1364 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 25) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1365 = or i32 %compare.local.base1364, 73, !dbg !24
  %compare.low1366 = icmp ult i32 %compare.local1365, %compare.low.base, !dbg !24
  %compare.result1367 = select i1 %compare.high.equal, i1 %compare.low1366, i1 %compare.high.order, !dbg !24
  %430 = select i1 %compare.result1367, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1368 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 26) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1369 = or i32 %compare.local.base1368, 74, !dbg !24
  %compare.low1370 = icmp ult i32 %compare.local1369, %compare.low.base, !dbg !24
  %compare.result1371 = select i1 %compare.high.equal, i1 %compare.low1370, i1 %compare.high.order, !dbg !24
  %431 = select i1 %compare.result1371, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1372 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 27) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1373 = or i32 %compare.local.base1372, 75, !dbg !24
  %compare.low1374 = icmp ult i32 %compare.local1373, %compare.low.base, !dbg !24
  %compare.result1375 = select i1 %compare.high.equal, i1 %compare.low1374, i1 %compare.high.order, !dbg !24
  %432 = select i1 %compare.result1375, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1376 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 28) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1377 = or i32 %compare.local.base1376, 76, !dbg !24
  %compare.low1378 = icmp ult i32 %compare.local1377, %compare.low.base, !dbg !24
  %compare.result1379 = select i1 %compare.high.equal, i1 %compare.low1378, i1 %compare.high.order, !dbg !24
  %433 = select i1 %compare.result1379, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1380 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 29) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1381 = or i32 %compare.local.base1380, 77, !dbg !24
  %compare.low1382 = icmp ult i32 %compare.local1381, %compare.low.base, !dbg !24
  %compare.result1383 = select i1 %compare.high.equal, i1 %compare.low1382, i1 %compare.high.order, !dbg !24
  %434 = select i1 %compare.result1383, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1384 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 30) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1385 = or i32 %compare.local.base1384, 78, !dbg !24
  %compare.low1386 = icmp ult i32 %compare.local1385, %compare.low.base, !dbg !24
  %compare.result1387 = select i1 %compare.high.equal, i1 %compare.low1386, i1 %compare.high.order, !dbg !24
  %435 = select i1 %compare.result1387, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1388 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 31) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1389 = or i32 %compare.local.base1388, 79, !dbg !24
  %compare.low1390 = icmp ult i32 %compare.local1389, %compare.low.base, !dbg !24
  %compare.result1391 = select i1 %compare.high.equal, i1 %compare.low1390, i1 %compare.high.order, !dbg !24
  %436 = select i1 %compare.result1391, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %437 = lshr i64 %116, 32, !dbg !24
  %compare.high1392 = trunc nuw i64 %437 to i32, !dbg !24
  %compare.high.equal1394 = icmp eq i32 %compare.high1392, %compare.high1268, !dbg !24
  %compare.high.order1395 = icmp ugt i32 %compare.high1392, %compare.high1268, !dbg !24
  %compare.low.base1396 = trunc i64 %116 to i32, !dbg !24
  %compare.low1398 = icmp ugt i32 %compare.low.base1396, %compare.low.base1269, !dbg !24
  %compare.result1399 = select i1 %compare.high.equal1394, i1 %compare.low1398, i1 %compare.high.order1395, !dbg !24
  %438 = select i1 %compare.result1399, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1400 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 32) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1401 = or i32 %compare.local.base1400, 1, !dbg !24
  %compare.low1402 = icmp ult i32 %compare.local1401, %compare.low.base1396, !dbg !24
  %compare.result1403 = select i1 %compare.high.equal1394, i1 %compare.low1402, i1 %compare.high.order1395, !dbg !24
  %439 = select i1 %compare.result1403, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1404 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 33) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1405 = or i32 %compare.local.base1404, 2, !dbg !24
  %compare.low1406 = icmp ult i32 %compare.local1405, %compare.low.base1396, !dbg !24
  %compare.result1407 = select i1 %compare.high.equal1394, i1 %compare.low1406, i1 %compare.high.order1395, !dbg !24
  %440 = select i1 %compare.result1407, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1408 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 34) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1409 = or i32 %compare.local.base1408, 3, !dbg !24
  %compare.low1410 = icmp ult i32 %compare.local1409, %compare.low.base1396, !dbg !24
  %compare.result1411 = select i1 %compare.high.equal1394, i1 %compare.low1410, i1 %compare.high.order1395, !dbg !24
  %441 = select i1 %compare.result1411, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1412 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 35) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1413 = or i32 %compare.local.base1412, 4, !dbg !24
  %compare.low1414 = icmp ult i32 %compare.local1413, %compare.low.base1396, !dbg !24
  %compare.result1415 = select i1 %compare.high.equal1394, i1 %compare.low1414, i1 %compare.high.order1395, !dbg !24
  %442 = select i1 %compare.result1415, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1416 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 36) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1417 = or i32 %compare.local.base1416, 5, !dbg !24
  %compare.low1418 = icmp ult i32 %compare.local1417, %compare.low.base1396, !dbg !24
  %compare.result1419 = select i1 %compare.high.equal1394, i1 %compare.low1418, i1 %compare.high.order1395, !dbg !24
  %443 = select i1 %compare.result1419, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1420 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 37) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1421 = or i32 %compare.local.base1420, 6, !dbg !24
  %compare.low1422 = icmp ult i32 %compare.local1421, %compare.low.base1396, !dbg !24
  %compare.result1423 = select i1 %compare.high.equal1394, i1 %compare.low1422, i1 %compare.high.order1395, !dbg !24
  %444 = select i1 %compare.result1423, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1424 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 38) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1425 = or i32 %compare.local.base1424, 7, !dbg !24
  %compare.low1426 = icmp ult i32 %compare.local1425, %compare.low.base1396, !dbg !24
  %compare.result1427 = select i1 %compare.high.equal1394, i1 %compare.low1426, i1 %compare.high.order1395, !dbg !24
  %445 = select i1 %compare.result1427, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1428 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 39) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1429 = or i32 %compare.local.base1428, 8, !dbg !24
  %compare.low1430 = icmp ult i32 %compare.local1429, %compare.low.base1396, !dbg !24
  %compare.result1431 = select i1 %compare.high.equal1394, i1 %compare.low1430, i1 %compare.high.order1395, !dbg !24
  %446 = select i1 %compare.result1431, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1432 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 40) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1433 = or i32 %compare.local.base1432, 9, !dbg !24
  %compare.low1434 = icmp ult i32 %compare.local1433, %compare.low.base1396, !dbg !24
  %compare.result1435 = select i1 %compare.high.equal1394, i1 %compare.low1434, i1 %compare.high.order1395, !dbg !24
  %447 = select i1 %compare.result1435, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1436 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 41) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1437 = or i32 %compare.local.base1436, 10, !dbg !24
  %compare.low1438 = icmp ult i32 %compare.local1437, %compare.low.base1396, !dbg !24
  %compare.result1439 = select i1 %compare.high.equal1394, i1 %compare.low1438, i1 %compare.high.order1395, !dbg !24
  %448 = select i1 %compare.result1439, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1440 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 42) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1441 = or i32 %compare.local.base1440, 11, !dbg !24
  %compare.low1442 = icmp ult i32 %compare.local1441, %compare.low.base1396, !dbg !24
  %compare.result1443 = select i1 %compare.high.equal1394, i1 %compare.low1442, i1 %compare.high.order1395, !dbg !24
  %449 = select i1 %compare.result1443, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1444 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 43) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1445 = or i32 %compare.local.base1444, 12, !dbg !24
  %compare.low1446 = icmp ult i32 %compare.local1445, %compare.low.base1396, !dbg !24
  %compare.result1447 = select i1 %compare.high.equal1394, i1 %compare.low1446, i1 %compare.high.order1395, !dbg !24
  %450 = select i1 %compare.result1447, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1448 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 44) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1449 = or i32 %compare.local.base1448, 13, !dbg !24
  %compare.low1450 = icmp ult i32 %compare.local1449, %compare.low.base1396, !dbg !24
  %compare.result1451 = select i1 %compare.high.equal1394, i1 %compare.low1450, i1 %compare.high.order1395, !dbg !24
  %451 = select i1 %compare.result1451, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1452 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 45) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1453 = or i32 %compare.local.base1452, 14, !dbg !24
  %compare.low1454 = icmp ult i32 %compare.local1453, %compare.low.base1396, !dbg !24
  %compare.result1455 = select i1 %compare.high.equal1394, i1 %compare.low1454, i1 %compare.high.order1395, !dbg !24
  %452 = select i1 %compare.result1455, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1456 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 46) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1457 = or i32 %compare.local.base1456, 15, !dbg !24
  %compare.low1458 = icmp ult i32 %compare.local1457, %compare.low.base1396, !dbg !24
  %compare.result1459 = select i1 %compare.high.equal1394, i1 %compare.low1458, i1 %compare.high.order1395, !dbg !24
  %453 = select i1 %compare.result1459, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1460 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 47) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1461 = or i32 %compare.local.base1460, 64, !dbg !24
  %compare.low1462 = icmp ult i32 %compare.local1461, %compare.low.base1396, !dbg !24
  %compare.result1463 = select i1 %compare.high.equal1394, i1 %compare.low1462, i1 %compare.high.order1395, !dbg !24
  %454 = select i1 %compare.result1463, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1464 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 48) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1465 = or i32 %compare.local.base1464, 65, !dbg !24
  %compare.low1466 = icmp ult i32 %compare.local1465, %compare.low.base1396, !dbg !24
  %compare.result1467 = select i1 %compare.high.equal1394, i1 %compare.low1466, i1 %compare.high.order1395, !dbg !24
  %455 = select i1 %compare.result1467, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1468 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 49) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1469 = or i32 %compare.local.base1468, 66, !dbg !24
  %compare.low1470 = icmp ult i32 %compare.local1469, %compare.low.base1396, !dbg !24
  %compare.result1471 = select i1 %compare.high.equal1394, i1 %compare.low1470, i1 %compare.high.order1395, !dbg !24
  %456 = select i1 %compare.result1471, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1472 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 50) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1473 = or i32 %compare.local.base1472, 67, !dbg !24
  %compare.low1474 = icmp ult i32 %compare.local1473, %compare.low.base1396, !dbg !24
  %compare.result1475 = select i1 %compare.high.equal1394, i1 %compare.low1474, i1 %compare.high.order1395, !dbg !24
  %457 = select i1 %compare.result1475, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1476 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 51) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1477 = or i32 %compare.local.base1476, 68, !dbg !24
  %compare.low1478 = icmp ult i32 %compare.local1477, %compare.low.base1396, !dbg !24
  %compare.result1479 = select i1 %compare.high.equal1394, i1 %compare.low1478, i1 %compare.high.order1395, !dbg !24
  %458 = select i1 %compare.result1479, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1480 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 52) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1481 = or i32 %compare.local.base1480, 69, !dbg !24
  %compare.low1482 = icmp ult i32 %compare.local1481, %compare.low.base1396, !dbg !24
  %compare.result1483 = select i1 %compare.high.equal1394, i1 %compare.low1482, i1 %compare.high.order1395, !dbg !24
  %459 = select i1 %compare.result1483, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1484 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 53) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1485 = or i32 %compare.local.base1484, 70, !dbg !24
  %compare.low1486 = icmp ult i32 %compare.local1485, %compare.low.base1396, !dbg !24
  %compare.result1487 = select i1 %compare.high.equal1394, i1 %compare.low1486, i1 %compare.high.order1395, !dbg !24
  %460 = select i1 %compare.result1487, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1488 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 54) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1489 = or i32 %compare.local.base1488, 71, !dbg !24
  %compare.low1490 = icmp ult i32 %compare.local1489, %compare.low.base1396, !dbg !24
  %compare.result1491 = select i1 %compare.high.equal1394, i1 %compare.low1490, i1 %compare.high.order1395, !dbg !24
  %461 = select i1 %compare.result1491, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1492 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 55) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1493 = or i32 %compare.local.base1492, 72, !dbg !24
  %compare.low1494 = icmp ult i32 %compare.local1493, %compare.low.base1396, !dbg !24
  %compare.result1495 = select i1 %compare.high.equal1394, i1 %compare.low1494, i1 %compare.high.order1395, !dbg !24
  %462 = select i1 %compare.result1495, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1496 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 56) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1497 = or i32 %compare.local.base1496, 73, !dbg !24
  %compare.low1498 = icmp ult i32 %compare.local1497, %compare.low.base1396, !dbg !24
  %compare.result1499 = select i1 %compare.high.equal1394, i1 %compare.low1498, i1 %compare.high.order1395, !dbg !24
  %463 = select i1 %compare.result1499, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1500 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 57) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1501 = or i32 %compare.local.base1500, 74, !dbg !24
  %compare.low1502 = icmp ult i32 %compare.local1501, %compare.low.base1396, !dbg !24
  %compare.result1503 = select i1 %compare.high.equal1394, i1 %compare.low1502, i1 %compare.high.order1395, !dbg !24
  %464 = select i1 %compare.result1503, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1504 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 58) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1505 = or i32 %compare.local.base1504, 75, !dbg !24
  %compare.low1506 = icmp ult i32 %compare.local1505, %compare.low.base1396, !dbg !24
  %compare.result1507 = select i1 %compare.high.equal1394, i1 %compare.low1506, i1 %compare.high.order1395, !dbg !24
  %465 = select i1 %compare.result1507, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1508 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 59) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1509 = or i32 %compare.local.base1508, 76, !dbg !24
  %compare.low1510 = icmp ult i32 %compare.local1509, %compare.low.base1396, !dbg !24
  %compare.result1511 = select i1 %compare.high.equal1394, i1 %compare.low1510, i1 %compare.high.order1395, !dbg !24
  %466 = select i1 %compare.result1511, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1512 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 60) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1513 = or i32 %compare.local.base1512, 77, !dbg !24
  %compare.low1514 = icmp ult i32 %compare.local1513, %compare.low.base1396, !dbg !24
  %compare.result1515 = select i1 %compare.high.equal1394, i1 %compare.low1514, i1 %compare.high.order1395, !dbg !24
  %467 = select i1 %compare.result1515, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1516 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 61) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1517 = or i32 %compare.local.base1516, 78, !dbg !24
  %compare.low1518 = icmp ult i32 %compare.local1517, %compare.low.base1396, !dbg !24
  %compare.result1519 = select i1 %compare.high.equal1394, i1 %compare.low1518, i1 %compare.high.order1395, !dbg !24
  %468 = select i1 %compare.result1519, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1520 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1269, i32 62) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1521 = or i32 %compare.local.base1520, 79, !dbg !24
  %compare.low1522 = icmp ult i32 %compare.local1521, %compare.low.base1396, !dbg !24
  %compare.result1523 = select i1 %compare.high.equal1394, i1 %compare.low1522, i1 %compare.high.order1395, !dbg !24
  %469 = select i1 %compare.result1523, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %470 = insertelement <4 x float> poison, float %405, i64 0, !dbg !26
  %471 = insertelement <4 x float> %470, float %406, i64 1, !dbg !26
  %472 = insertelement <4 x float> %471, float %407, i64 2, !dbg !26
  %473 = insertelement <4 x float> %472, float %408, i64 3, !dbg !26
  %474 = insertelement <4 x float> poison, float %409, i64 0, !dbg !26
  %475 = insertelement <4 x float> %474, float %410, i64 1, !dbg !26
  %476 = insertelement <4 x float> %475, float %411, i64 2, !dbg !26
  %477 = insertelement <4 x float> %476, float %412, i64 3, !dbg !26
  %478 = insertelement <4 x float> poison, float %413, i64 0, !dbg !26
  %479 = insertelement <4 x float> %478, float %414, i64 1, !dbg !26
  %480 = insertelement <4 x float> %479, float %415, i64 2, !dbg !26
  %481 = insertelement <4 x float> %480, float %416, i64 3, !dbg !26
  %482 = insertelement <4 x float> poison, float %417, i64 0, !dbg !26
  %483 = insertelement <4 x float> %482, float %418, i64 1, !dbg !26
  %484 = insertelement <4 x float> %483, float %419, i64 2, !dbg !26
  %485 = insertelement <4 x float> %484, float %420, i64 3, !dbg !26
  %486 = insertelement <4 x float> poison, float %421, i64 0, !dbg !26
  %487 = insertelement <4 x float> %486, float %422, i64 1, !dbg !26
  %488 = insertelement <4 x float> %487, float %423, i64 2, !dbg !26
  %489 = insertelement <4 x float> %488, float %424, i64 3, !dbg !26
  %490 = insertelement <4 x float> poison, float %425, i64 0, !dbg !26
  %491 = insertelement <4 x float> %490, float %426, i64 1, !dbg !26
  %492 = insertelement <4 x float> %491, float %427, i64 2, !dbg !26
  %493 = insertelement <4 x float> %492, float %428, i64 3, !dbg !26
  %494 = insertelement <4 x float> poison, float %429, i64 0, !dbg !26
  %495 = insertelement <4 x float> %494, float %430, i64 1, !dbg !26
  %496 = insertelement <4 x float> %495, float %431, i64 2, !dbg !26
  %497 = insertelement <4 x float> %496, float %432, i64 3, !dbg !26
  %498 = insertelement <4 x float> poison, float %433, i64 0, !dbg !26
  %499 = insertelement <4 x float> %498, float %434, i64 1, !dbg !26
  %500 = insertelement <4 x float> %499, float %435, i64 2, !dbg !26
  %501 = insertelement <4 x float> %500, float %436, i64 3, !dbg !26
  %502 = insertelement <4 x float> poison, float %438, i64 0, !dbg !26
  %503 = insertelement <4 x float> %502, float %439, i64 1, !dbg !26
  %504 = insertelement <4 x float> %503, float %440, i64 2, !dbg !26
  %505 = insertelement <4 x float> %504, float %441, i64 3, !dbg !26
  %506 = insertelement <4 x float> poison, float %442, i64 0, !dbg !26
  %507 = insertelement <4 x float> %506, float %443, i64 1, !dbg !26
  %508 = insertelement <4 x float> %507, float %444, i64 2, !dbg !26
  %509 = insertelement <4 x float> %508, float %445, i64 3, !dbg !26
  %510 = insertelement <4 x float> poison, float %446, i64 0, !dbg !26
  %511 = insertelement <4 x float> %510, float %447, i64 1, !dbg !26
  %512 = insertelement <4 x float> %511, float %448, i64 2, !dbg !26
  %513 = insertelement <4 x float> %512, float %449, i64 3, !dbg !26
  %514 = insertelement <4 x float> poison, float %450, i64 0, !dbg !26
  %515 = insertelement <4 x float> %514, float %451, i64 1, !dbg !26
  %516 = insertelement <4 x float> %515, float %452, i64 2, !dbg !26
  %517 = insertelement <4 x float> %516, float %453, i64 3, !dbg !26
  %518 = insertelement <4 x float> poison, float %454, i64 0, !dbg !26
  %519 = insertelement <4 x float> %518, float %455, i64 1, !dbg !26
  %520 = insertelement <4 x float> %519, float %456, i64 2, !dbg !26
  %521 = insertelement <4 x float> %520, float %457, i64 3, !dbg !26
  %522 = insertelement <4 x float> poison, float %458, i64 0, !dbg !26
  %523 = insertelement <4 x float> %522, float %459, i64 1, !dbg !26
  %524 = insertelement <4 x float> %523, float %460, i64 2, !dbg !26
  %525 = insertelement <4 x float> %524, float %461, i64 3, !dbg !26
  %526 = insertelement <4 x float> poison, float %462, i64 0, !dbg !26
  %527 = insertelement <4 x float> %526, float %463, i64 1, !dbg !26
  %528 = insertelement <4 x float> %527, float %464, i64 2, !dbg !26
  %529 = insertelement <4 x float> %528, float %465, i64 3, !dbg !26
  %530 = insertelement <4 x float> poison, float %466, i64 0, !dbg !26
  %531 = insertelement <4 x float> %530, float %467, i64 1, !dbg !26
  %532 = insertelement <4 x float> %531, float %468, i64 2, !dbg !26
  %533 = insertelement <4 x float> %532, float %469, i64 3, !dbg !26
  %534 = load <4 x half>, ptr addrspace(3) %kdma.gep2163, align 8, !dbg !16, !frisk.k_tail_pair !42
  %535 = load <4 x half>, ptr addrspace(3) %kdma.gep2167, align 8, !dbg !16, !frisk.k_tail_pair !42
  %536 = load <4 x half>, ptr addrspace(3) %kdma.gep2164, align 8, !dbg !16, !frisk.k_tail_pair !43
  %537 = load <4 x half>, ptr addrspace(3) %kdma.gep2168, align 8, !dbg !16, !frisk.k_tail_pair !43
  %538 = load <4 x half>, ptr addrspace(3) %kdma.gep2165, align 8, !dbg !16, !frisk.k_tail_pair !44
  %539 = load <4 x half>, ptr addrspace(3) %kdma.gep2169, align 8, !dbg !16, !frisk.k_tail_pair !44
  %540 = load <4 x half>, ptr addrspace(3) %kdma.gep2166, align 8, !dbg !16, !frisk.k_tail_pair !45
  %541 = load <4 x half>, ptr addrspace(3) %kdma.gep2170, align 8, !dbg !16, !frisk.k_tail_pair !45
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16, !frisk.k_tail_prefetch !1
  %542 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %534, <4 x float> %401) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16, !frisk.k_tail_prefetch !1
  %543 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %536, <4 x float> %361) #9, !dbg !16
  %544 = fadd <4 x float> %542, %473, !dbg !26
  %545 = extractelement <4 x float> %544, i64 0, !dbg !26
  %546 = call float @llvm.maxnum.f32(float %545, float 0xFFF0000000000000), !dbg !27, !frisk.local_reduction !1
  %547 = extractelement <4 x float> %544, i64 1, !dbg !26
  %548 = extractelement <4 x float> %544, i64 2, !dbg !26
  %549 = extractelement <4 x float> %544, i64 3, !dbg !26
  %550 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %538, <4 x float> %543) #9, !dbg !16
  %551 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %540, <4 x float> %363) #9, !dbg !16
  %552 = fadd <4 x float> %550, %477, !dbg !26
  %553 = extractelement <4 x float> %552, i64 0, !dbg !26
  %local.reduce218 = call float @llvm.maxnum.f32(float %546, float %553), !dbg !27, !frisk.local_reduction !1
  %554 = extractelement <4 x float> %552, i64 1, !dbg !26
  %local.reduce219 = call float @llvm.maxnum.f32(float %547, float %554), !dbg !27, !frisk.local_reduction !1
  %555 = extractelement <4 x float> %552, i64 2, !dbg !26
  %local.reduce220 = call float @llvm.maxnum.f32(float %548, float %555), !dbg !27, !frisk.local_reduction !1
  %556 = extractelement <4 x float> %552, i64 3, !dbg !26
  %local.reduce221 = call float @llvm.maxnum.f32(float %549, float %556), !dbg !27, !frisk.local_reduction !1
  %557 = load <4 x half>, ptr addrspace(3) %kdma.gep2171, align 8, !dbg !16, !frisk.k_tail_pair !40
  %558 = load <4 x half>, ptr addrspace(3) %kdma.gep2175, align 8, !dbg !16, !frisk.k_tail_pair !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16, !frisk.k_tail_prefetch !1
  %559 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %535, <4 x float> %551) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16, !frisk.k_tail_prefetch !1
  %560 = load <4 x half>, ptr addrspace(3) %kdma.gep2172, align 8, !dbg !16, !frisk.k_tail_pair !46
  %561 = load <4 x half>, ptr addrspace(3) %kdma.gep2176, align 8, !dbg !16, !frisk.k_tail_pair !46
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16, !frisk.k_tail_prefetch !1
  %562 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %537, <4 x float> %365) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !26, !frisk.k_tail_prefetch !1
  %563 = fadd <4 x float> %559, %481, !dbg !26
  %564 = extractelement <4 x float> %563, i64 0, !dbg !26
  %local.reduce222 = call float @llvm.maxnum.f32(float %local.reduce218, float %564), !dbg !27, !frisk.local_reduction !1
  %565 = extractelement <4 x float> %563, i64 1, !dbg !26
  %local.reduce223 = call float @llvm.maxnum.f32(float %local.reduce219, float %565), !dbg !27, !frisk.local_reduction !1
  %566 = extractelement <4 x float> %563, i64 2, !dbg !26
  %local.reduce224 = call float @llvm.maxnum.f32(float %local.reduce220, float %566), !dbg !27, !frisk.local_reduction !1
  %567 = extractelement <4 x float> %563, i64 3, !dbg !26
  %local.reduce225 = call float @llvm.maxnum.f32(float %local.reduce221, float %567), !dbg !27, !frisk.local_reduction !1
  %568 = load <4 x half>, ptr addrspace(3) %kdma.gep2173, align 8, !dbg !16, !frisk.k_tail_pair !47
  %569 = load <4 x half>, ptr addrspace(3) %kdma.gep2177, align 8, !dbg !16, !frisk.k_tail_pair !47
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16, !frisk.k_tail_prefetch !1
  %570 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %539, <4 x float> %562) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16, !frisk.k_tail_prefetch !1
  %571 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %541, <4 x float> %367) #9, !dbg !16
  %572 = fadd <4 x float> %570, %485, !dbg !26
  %573 = extractelement <4 x float> %572, i64 0, !dbg !26
  %local.reduce226 = call float @llvm.maxnum.f32(float %local.reduce222, float %573), !dbg !27, !frisk.local_reduction !1
  %574 = extractelement <4 x float> %572, i64 1, !dbg !26
  %local.reduce227 = call float @llvm.maxnum.f32(float %local.reduce223, float %574), !dbg !27, !frisk.local_reduction !1
  %575 = extractelement <4 x float> %572, i64 2, !dbg !26
  %local.reduce228 = call float @llvm.maxnum.f32(float %local.reduce224, float %575), !dbg !27, !frisk.local_reduction !1
  %576 = extractelement <4 x float> %572, i64 3, !dbg !26
  %local.reduce229 = call float @llvm.maxnum.f32(float %local.reduce225, float %576), !dbg !27, !frisk.local_reduction !1
  %577 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %557, <4 x float> %571) #9, !dbg !16
  %578 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %560, <4 x float> %369) #9, !dbg !16
  %579 = fadd <4 x float> %577, %489, !dbg !26
  %580 = extractelement <4 x float> %579, i64 0, !dbg !26
  %local.reduce230 = call float @llvm.maxnum.f32(float %local.reduce226, float %580), !dbg !27, !frisk.local_reduction !1
  %581 = extractelement <4 x float> %579, i64 1, !dbg !26
  %local.reduce231 = call float @llvm.maxnum.f32(float %local.reduce227, float %581), !dbg !27, !frisk.local_reduction !1
  %582 = extractelement <4 x float> %579, i64 2, !dbg !26
  %local.reduce232 = call float @llvm.maxnum.f32(float %local.reduce228, float %582), !dbg !27, !frisk.local_reduction !1
  %583 = extractelement <4 x float> %579, i64 3, !dbg !26
  %local.reduce233 = call float @llvm.maxnum.f32(float %local.reduce229, float %583), !dbg !27, !frisk.local_reduction !1
  %584 = load <4 x half>, ptr addrspace(3) %kdma.gep2174, align 8, !dbg !16
  %585 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %568, <4 x float> %578) #9, !dbg !16
  %586 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %584, <4 x float> %371) #9, !dbg !16
  %587 = fadd <4 x float> %585, %493, !dbg !26
  %588 = extractelement <4 x float> %587, i64 0, !dbg !26
  %local.reduce234 = call float @llvm.maxnum.f32(float %local.reduce230, float %588), !dbg !27, !frisk.local_reduction !1
  %589 = extractelement <4 x float> %587, i64 1, !dbg !26
  %local.reduce235 = call float @llvm.maxnum.f32(float %local.reduce231, float %589), !dbg !27, !frisk.local_reduction !1
  %590 = extractelement <4 x float> %587, i64 2, !dbg !26
  %local.reduce236 = call float @llvm.maxnum.f32(float %local.reduce232, float %590), !dbg !27, !frisk.local_reduction !1
  %591 = extractelement <4 x float> %587, i64 3, !dbg !26
  %local.reduce237 = call float @llvm.maxnum.f32(float %local.reduce233, float %591), !dbg !27, !frisk.local_reduction !1
  %592 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %558, <4 x float> %586) #9, !dbg !16
  %593 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %561, <4 x float> %373) #9, !dbg !16
  %594 = fadd <4 x float> %592, %497, !dbg !26
  %595 = extractelement <4 x float> %594, i64 0, !dbg !26
  %local.reduce238 = call float @llvm.maxnum.f32(float %local.reduce234, float %595), !dbg !27, !frisk.local_reduction !1
  %596 = extractelement <4 x float> %594, i64 1, !dbg !26
  %local.reduce239 = call float @llvm.maxnum.f32(float %local.reduce235, float %596), !dbg !27, !frisk.local_reduction !1
  %597 = extractelement <4 x float> %594, i64 2, !dbg !26
  %local.reduce240 = call float @llvm.maxnum.f32(float %local.reduce236, float %597), !dbg !27, !frisk.local_reduction !1
  %598 = extractelement <4 x float> %594, i64 3, !dbg !26
  %local.reduce241 = call float @llvm.maxnum.f32(float %local.reduce237, float %598), !dbg !27, !frisk.local_reduction !1
  %599 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %569, <4 x float> %593) #9, !dbg !16
  %600 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %400, <4 x float> %375) #9, !dbg !16
  %601 = fadd <4 x float> %599, %501, !dbg !26
  %602 = extractelement <4 x float> %601, i64 0, !dbg !26
  %local.reduce242 = call float @llvm.maxnum.f32(float %local.reduce238, float %602), !dbg !27, !frisk.local_reduction !1
  %603 = extractelement <4 x float> %601, i64 1, !dbg !26
  %local.reduce243 = call float @llvm.maxnum.f32(float %local.reduce239, float %603), !dbg !27, !frisk.local_reduction !1
  %604 = extractelement <4 x float> %601, i64 2, !dbg !26
  %local.reduce244 = call float @llvm.maxnum.f32(float %local.reduce240, float %604), !dbg !27, !frisk.local_reduction !1
  %605 = extractelement <4 x float> %601, i64 3, !dbg !26
  %local.reduce245 = call float @llvm.maxnum.f32(float %local.reduce241, float %605), !dbg !27, !frisk.local_reduction !1
  %local.reduce246 = call float @llvm.maxnum.f32(float %local.reduce242, float %local.reduce243), !dbg !27, !frisk.local_reduction !1
  %local.reduce247 = call float @llvm.maxnum.f32(float %local.reduce244, float %local.reduce245), !dbg !27, !frisk.local_reduction !1
  %606 = call float @llvm.maxnum.f32(float %local.reduce246, float %local.reduce247), !dbg !27, !frisk.local_reduction !1
  %607 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %534, <4 x float> %600) #9, !dbg !16
  %608 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %536, <4 x float> %377) #9, !dbg !16
  %609 = fadd <4 x float> %607, %505, !dbg !26
  %610 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %538, <4 x float> %608) #9, !dbg !16
  %611 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %540, <4 x float> %379) #9, !dbg !16
  %612 = fadd <4 x float> %610, %509, !dbg !26
  %613 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %535, <4 x float> %611) #9, !dbg !16
  %614 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %537, <4 x float> %381) #9, !dbg !16
  %615 = fadd <4 x float> %613, %513, !dbg !26
  %616 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %539, <4 x float> %614) #9, !dbg !16
  %617 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %541, <4 x float> %383) #9, !dbg !16
  %618 = fadd <4 x float> %616, %517, !dbg !26
  %619 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %557, <4 x float> %617) #9, !dbg !16
  %620 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %560, <4 x float> %385) #9, !dbg !16
  %621 = fadd <4 x float> %619, %521, !dbg !26
  %622 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %568, <4 x float> %620) #9, !dbg !16
  %623 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %584, <4 x float> %387) #9, !dbg !16
  %624 = fadd <4 x float> %622, %525, !dbg !26
  %625 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %558, <4 x float> %623) #9, !dbg !16
  %626 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %561, <4 x float> %389) #9, !dbg !16
  %627 = fadd <4 x float> %625, %529, !dbg !26
  %628 = extractelement <4 x float> %609, i64 0, !dbg !26
  %629 = extractelement <4 x float> %609, i64 1, !dbg !26
  %630 = extractelement <4 x float> %609, i64 2, !dbg !26
  %631 = extractelement <4 x float> %609, i64 3, !dbg !26
  %632 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %569, <4 x float> %626) #9, !dbg !16
  %633 = extractelement <4 x float> %612, i64 0, !dbg !26
  %634 = extractelement <4 x float> %612, i64 1, !dbg !26
  %635 = extractelement <4 x float> %612, i64 2, !dbg !26
  %636 = extractelement <4 x float> %612, i64 3, !dbg !26
  %637 = extractelement <4 x float> %615, i64 0, !dbg !26
  %638 = extractelement <4 x float> %615, i64 1, !dbg !26
  %639 = extractelement <4 x float> %615, i64 2, !dbg !26
  %640 = extractelement <4 x float> %615, i64 3, !dbg !26
  %641 = fadd <4 x float> %632, %533, !dbg !26
  %642 = extractelement <4 x float> %618, i64 0, !dbg !26
  %643 = extractelement <4 x float> %618, i64 1, !dbg !26
  %644 = extractelement <4 x float> %618, i64 2, !dbg !26
  %645 = extractelement <4 x float> %618, i64 3, !dbg !26
  %646 = extractelement <4 x float> %621, i64 0, !dbg !26
  %647 = extractelement <4 x float> %621, i64 1, !dbg !26
  %648 = extractelement <4 x float> %621, i64 2, !dbg !26
  %649 = extractelement <4 x float> %621, i64 3, !dbg !26
  %650 = extractelement <4 x float> %624, i64 0, !dbg !26
  %651 = extractelement <4 x float> %624, i64 1, !dbg !26
  %652 = extractelement <4 x float> %624, i64 2, !dbg !26
  %653 = extractelement <4 x float> %624, i64 3, !dbg !26
  %654 = extractelement <4 x float> %627, i64 0, !dbg !26
  %655 = extractelement <4 x float> %627, i64 1, !dbg !26
  %656 = extractelement <4 x float> %627, i64 2, !dbg !26
  %657 = extractelement <4 x float> %627, i64 3, !dbg !26
  %658 = extractelement <4 x float> %641, i64 0, !dbg !26
  %659 = extractelement <4 x float> %641, i64 1, !dbg !26
  %660 = extractelement <4 x float> %641, i64 2, !dbg !26
  %661 = extractelement <4 x float> %641, i64 3, !dbg !26
  %662 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !27
  %663 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %662), !dbg !27
  %664 = add nuw nsw i32 %663, 64, !dbg !27
  %665 = xor i32 %663, 16, !dbg !27
  %666 = and i32 %664, 192, !dbg !27
  %667 = icmp samesign ult i32 %665, %666, !dbg !27
  %668 = select i1 %667, i32 %665, i32 %663, !dbg !27
  %669 = shl nuw nsw i32 %668, 2, !dbg !27
  %670 = bitcast float %606 to i32, !dbg !27
  %671 = call i32 @llvm.amdgcn.ds.bpermute(i32 %669, i32 %670), !dbg !27
  %672 = bitcast i32 %671 to float, !dbg !27
  %673 = call float @llvm.maxnum.f32(float %606, float %672), !dbg !27
  %674 = xor i32 %663, 32, !dbg !27
  %675 = icmp samesign ult i32 %674, %666, !dbg !27
  %676 = select i1 %675, i32 %674, i32 %663, !dbg !27
  %677 = shl nuw nsw i32 %676, 2, !dbg !27
  %678 = bitcast float %673 to i32, !dbg !27
  %679 = call i32 @llvm.amdgcn.ds.bpermute(i32 %677, i32 %678), !dbg !27
  %680 = bitcast i32 %679 to float, !dbg !27
  %681 = call float @llvm.maxnum.f32(float %673, float %680), !dbg !27
  %682 = call float @llvm.maxnum.f32(float %628, float 0xFFF0000000000000), !dbg !27, !frisk.local_reduction !1
  %local.reduce187 = call float @llvm.maxnum.f32(float %682, float %633), !dbg !27, !frisk.local_reduction !1
  %local.reduce188 = call float @llvm.maxnum.f32(float %629, float %634), !dbg !27, !frisk.local_reduction !1
  %local.reduce189 = call float @llvm.maxnum.f32(float %630, float %635), !dbg !27, !frisk.local_reduction !1
  %local.reduce190 = call float @llvm.maxnum.f32(float %631, float %636), !dbg !27, !frisk.local_reduction !1
  %local.reduce191 = call float @llvm.maxnum.f32(float %local.reduce187, float %637), !dbg !27, !frisk.local_reduction !1
  %local.reduce192 = call float @llvm.maxnum.f32(float %local.reduce188, float %638), !dbg !27, !frisk.local_reduction !1
  %local.reduce193 = call float @llvm.maxnum.f32(float %local.reduce189, float %639), !dbg !27, !frisk.local_reduction !1
  %local.reduce194 = call float @llvm.maxnum.f32(float %local.reduce190, float %640), !dbg !27, !frisk.local_reduction !1
  %local.reduce195 = call float @llvm.maxnum.f32(float %local.reduce191, float %642), !dbg !27, !frisk.local_reduction !1
  %local.reduce196 = call float @llvm.maxnum.f32(float %local.reduce192, float %643), !dbg !27, !frisk.local_reduction !1
  %local.reduce197 = call float @llvm.maxnum.f32(float %local.reduce193, float %644), !dbg !27, !frisk.local_reduction !1
  %local.reduce198 = call float @llvm.maxnum.f32(float %local.reduce194, float %645), !dbg !27, !frisk.local_reduction !1
  %local.reduce199 = call float @llvm.maxnum.f32(float %local.reduce195, float %646), !dbg !27, !frisk.local_reduction !1
  %local.reduce200 = call float @llvm.maxnum.f32(float %local.reduce196, float %647), !dbg !27, !frisk.local_reduction !1
  %local.reduce201 = call float @llvm.maxnum.f32(float %local.reduce197, float %648), !dbg !27, !frisk.local_reduction !1
  %local.reduce202 = call float @llvm.maxnum.f32(float %local.reduce198, float %649), !dbg !27, !frisk.local_reduction !1
  %local.reduce203 = call float @llvm.maxnum.f32(float %local.reduce199, float %650), !dbg !27, !frisk.local_reduction !1
  %local.reduce204 = call float @llvm.maxnum.f32(float %local.reduce200, float %651), !dbg !27, !frisk.local_reduction !1
  %local.reduce205 = call float @llvm.maxnum.f32(float %local.reduce201, float %652), !dbg !27, !frisk.local_reduction !1
  %local.reduce206 = call float @llvm.maxnum.f32(float %local.reduce202, float %653), !dbg !27, !frisk.local_reduction !1
  %local.reduce207 = call float @llvm.maxnum.f32(float %local.reduce203, float %654), !dbg !27, !frisk.local_reduction !1
  %local.reduce208 = call float @llvm.maxnum.f32(float %local.reduce204, float %655), !dbg !27, !frisk.local_reduction !1
  %local.reduce209 = call float @llvm.maxnum.f32(float %local.reduce205, float %656), !dbg !27, !frisk.local_reduction !1
  %local.reduce210 = call float @llvm.maxnum.f32(float %local.reduce206, float %657), !dbg !27, !frisk.local_reduction !1
  %local.reduce211 = call float @llvm.maxnum.f32(float %local.reduce207, float %658), !dbg !27, !frisk.local_reduction !1
  %local.reduce212 = call float @llvm.maxnum.f32(float %local.reduce208, float %659), !dbg !27, !frisk.local_reduction !1
  %local.reduce213 = call float @llvm.maxnum.f32(float %local.reduce209, float %660), !dbg !27, !frisk.local_reduction !1
  %local.reduce214 = call float @llvm.maxnum.f32(float %local.reduce210, float %661), !dbg !27, !frisk.local_reduction !1
  %local.reduce215 = call float @llvm.maxnum.f32(float %local.reduce211, float %local.reduce212), !dbg !27, !frisk.local_reduction !1
  %local.reduce216 = call float @llvm.maxnum.f32(float %local.reduce213, float %local.reduce214), !dbg !27, !frisk.local_reduction !1
  %683 = call float @llvm.maxnum.f32(float %local.reduce215, float %local.reduce216), !dbg !27, !frisk.local_reduction !1
  %684 = bitcast float %683 to i32, !dbg !27
  %685 = call i32 @llvm.amdgcn.ds.bpermute(i32 %669, i32 %684), !dbg !27
  %686 = bitcast i32 %685 to float, !dbg !27
  %687 = call float @llvm.maxnum.f32(float %683, float %686), !dbg !27
  %688 = bitcast float %687 to i32, !dbg !27
  %689 = call i32 @llvm.amdgcn.ds.bpermute(i32 %677, i32 %688), !dbg !27
  %690 = bitcast i32 %689 to float, !dbg !27
  %691 = call float @llvm.maxnum.f32(float %687, float %690), !dbg !27
  %692 = extractvalue [2 x <1 x float>] %121, 0, !dbg !27
  %693 = insertelement <1 x float> poison, float %681, i64 0, !dbg !27
  %694 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %693, <1 x float> %692), !dbg !27
  %695 = extractvalue [2 x <1 x float>] %121, 1, !dbg !27
  %696 = insertelement <1 x float> poison, float %691, i64 0, !dbg !27
  %697 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %696, <1 x float> %695), !dbg !27
  %698 = shufflevector <1 x float> %694, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !28
  %699 = shufflevector <1 x float> %697, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !28
  %700 = fsub <1 x float> %692, %694, !dbg !29
  %701 = extractelement <1 x float> %700, i64 0, !dbg !30
  %702 = call float @llvm.amdgcn.exp2.f32(float %701), !dbg !30
  %703 = insertelement <4 x float> poison, float %702, i64 0, !dbg !35
  %704 = shufflevector <4 x float> %703, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !35
  %705 = extractvalue [2 x <1 x float>] %120, 0, !dbg !31
  %706 = extractelement <1 x float> %705, i64 0, !dbg !31
  %707 = fmul float %706, %702, !dbg !31
  %708 = fsub <1 x float> %695, %697, !dbg !29
  %709 = extractelement <1 x float> %708, i64 0, !dbg !30
  %710 = call float @llvm.amdgcn.exp2.f32(float %709), !dbg !30
  %711 = insertelement <4 x float> poison, float %710, i64 0, !dbg !35
  %712 = shufflevector <4 x float> %711, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !35
  %713 = extractvalue [2 x <1 x float>] %120, 1, !dbg !31
  %714 = extractelement <1 x float> %713, i64 0, !dbg !31
  %715 = fmul float %714, %710, !dbg !31
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %vdb.address1788 = or disjoint i64 %.idx17641766, 32, !dbg !17
  %vdb.address1789 = or disjoint i64 %address.local988, %vdb.address1788, !dbg !17
  %vdb.address1790 = trunc i64 %vdb.address1789 to i32, !dbg !17
  %vdb.address1791 = shl i32 %vdb.address1790, 1, !dbg !17
  %vdb.address1798 = or disjoint i64 %address.local994, %vdb.address1788, !dbg !17
  %vdb.address1799 = trunc i64 %vdb.address1798 to i32, !dbg !17
  %vdb.address1800 = shl i32 %vdb.address1799, 1, !dbg !17
  %vdb.dma.base = add i32 %137, 8192, !dbg !13
  %716 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1791, <4 x i32> %dma.buffer.rsrc1670) #10, !dbg !17
  %vdb.dma.base1805 = add i32 %144, 8192, !dbg !13
  %717 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1805, i32 %vdb.address1800, <4 x i32> %dma.buffer.rsrc1670) #10, !dbg !17
  %address.local996 = shl nuw nsw i32 %address.thread565, 3, !dbg !13
  %address.local997 = and i32 %address.local996, 120, !dbg !13
  %address.local999 = and i32 %address.local642, 1536, !dbg !13
  %address.local1000 = or disjoint i32 %address.local997, %address.local999, !dbg !13
  %address.local1001 = zext nneg i32 %address.local1000 to i64, !dbg !13
  %address.local1002 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1001, !dbg !13
  %718 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1002, i32 0) #0, !dbg !13, !frisk.v_double_buffer !41
  %719 = bitcast <4 x i32> %718 to <8 x half>, !dbg !13
  %address.local1003 = and i64 %address.local566, 60, !dbg !13
  %address.local1004 = add nuw nsw i64 %address.local1003, %address.local566, !dbg !13
  %address.local1006.idx = shl nuw nsw i64 %address.local1004, 4, !dbg !13
  %address.local1006 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %address.local1006.idx, !dbg !13
  %address.local1007 = lshr i64 %address.local622, 4, !dbg !13
  %address.local1009.idx = mul nuw nsw i64 %address.local1007, 768, !dbg !13
  %address.local1009 = getelementptr i8, ptr addrspace(3) %address.local1006, i64 %address.local1009.idx, !dbg !13
  %address.local1011.idx = mul nsw i64 %address.local609, -64, !dbg !13
  %address.local1011 = getelementptr i8, ptr addrspace(3) %address.local1009, i64 %address.local1011.idx, !dbg !13
  %address.local1012 = getelementptr i8, ptr addrspace(3) %address.local1011, i64 256, !dbg !13
  %720 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1012, i32 0) #0, !dbg !13, !frisk.v_double_buffer !41
  %721 = bitcast <4 x i32> %720 to <8 x half>, !dbg !13
  %address.local1016 = getelementptr i8, ptr addrspace(3) %address.local1011, i64 512, !dbg !13
  %722 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1016, i32 0) #0, !dbg !13, !frisk.v_double_buffer !41
  %723 = bitcast <4 x i32> %722 to <8 x half>, !dbg !13
  %address.local1020 = getelementptr i8, ptr addrspace(3) %address.local1011, i64 768, !dbg !13
  %724 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1020, i32 0) #0, !dbg !13, !frisk.v_double_buffer !41
  %725 = bitcast <4 x i32> %724 to <8 x half>, !dbg !13
  %address.local1022 = getelementptr i8, ptr addrspace(3) %address.local1002, i64 4096, !dbg !13
  %726 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1022, i32 0) #0, !dbg !13, !frisk.v_double_buffer !41
  %727 = bitcast <4 x i32> %726 to <8 x half>, !dbg !13
  %address.local1026 = getelementptr i8, ptr addrspace(3) %address.local1011, i64 4352, !dbg !13
  %728 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1026, i32 0) #0, !dbg !13, !frisk.v_double_buffer !41
  %729 = bitcast <4 x i32> %728 to <8 x half>, !dbg !13
  %address.local1030 = getelementptr i8, ptr addrspace(3) %address.local1011, i64 4608, !dbg !13
  %730 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1030, i32 0) #0, !dbg !13, !frisk.v_double_buffer !41
  %731 = bitcast <4 x i32> %730 to <8 x half>, !dbg !13
  %address.local1034 = getelementptr i8, ptr addrspace(3) %address.local1011, i64 4864, !dbg !13
  %732 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1034, i32 0) #0, !dbg !13, !frisk.v_double_buffer !41
  %733 = shufflevector <8 x half> %719, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %734 = extractvalue [2 x <8 x float>] %119, 0, !dbg !35
  %735 = shufflevector <8 x float> %734, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %736 = fmul <4 x float> %735, %704, !dbg !35
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %737 = fsub <4 x float> %544, %698, !dbg !28
  %738 = extractelement <4 x float> %737, i64 0, !dbg !32
  %739 = call float @llvm.amdgcn.exp2.f32(float %738), !dbg !32
  %740 = insertelement <4 x float> poison, float %739, i64 0, !dbg !34
  %741 = extractelement <4 x float> %737, i64 1, !dbg !32
  %742 = call float @llvm.amdgcn.exp2.f32(float %741), !dbg !32
  %743 = insertelement <4 x float> %740, float %742, i64 1, !dbg !34
  %744 = extractelement <4 x float> %737, i64 2, !dbg !32
  %745 = call float @llvm.amdgcn.exp2.f32(float %744), !dbg !32
  %746 = insertelement <4 x float> %743, float %745, i64 2, !dbg !34
  %747 = extractelement <4 x float> %737, i64 3, !dbg !32
  %748 = call float @llvm.amdgcn.exp2.f32(float %747), !dbg !32
  %749 = insertelement <4 x float> %746, float %748, i64 3, !dbg !34
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %749) #9, !dbg !34, !frisk.register_lifetimes !1
  %.cvtpack.f0 = extractelement <4 x float> %math.result, i32 0, !dbg !34
  %.cvtpack.f1 = extractelement <4 x float> %math.result, i32 1, !dbg !34
  %.cvtpack.f2 = extractelement <4 x float> %math.result, i32 2, !dbg !34
  %.cvtpack.f3 = extractelement <4 x float> %math.result, i32 3, !dbg !34
  %.cvtpack.p0 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f0, float %.cvtpack.f1), !dbg !34
  %.cvtpack.p1 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f2, float %.cvtpack.f3), !dbg !34
  %.cvtpack.v0 = insertelement <2 x i32> poison, i32 %.cvtpack.p0, i32 0, !dbg !34
  %.cvtpack.v1 = insertelement <2 x i32> %.cvtpack.v0, i32 %.cvtpack.p1, i32 1, !dbg !34
  %750 = bitcast <2 x i32> %.cvtpack.v1 to <4 x half>, !dbg !34
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %750) #9, !dbg !32, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %751 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %733, <4 x float> %736) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %752 = shufflevector <8 x half> %719, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %753 = shufflevector <8 x float> %734, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %754 = fmul <4 x float> %753, %704, !dbg !35
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %755 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %752, <4 x float> %754) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !35
  %756 = extractvalue [2 x <8 x float>] %119, 1, !dbg !35
  %757 = shufflevector <8 x float> %756, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %758 = fmul <4 x float> %757, %712, !dbg !35
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %759 = fsub <4 x float> %609, %699, !dbg !28
  %760 = extractelement <4 x float> %759, i64 0, !dbg !32
  %761 = call float @llvm.amdgcn.exp2.f32(float %760), !dbg !32
  %762 = insertelement <4 x float> poison, float %761, i64 0, !dbg !34
  %763 = extractelement <4 x float> %759, i64 1, !dbg !32
  %764 = call float @llvm.amdgcn.exp2.f32(float %763), !dbg !32
  %765 = insertelement <4 x float> %762, float %764, i64 1, !dbg !34
  %766 = extractelement <4 x float> %759, i64 2, !dbg !32
  %767 = call float @llvm.amdgcn.exp2.f32(float %766), !dbg !32
  %768 = insertelement <4 x float> %765, float %767, i64 2, !dbg !34
  %769 = extractelement <4 x float> %759, i64 3, !dbg !32
  %770 = call float @llvm.amdgcn.exp2.f32(float %769), !dbg !32
  %771 = insertelement <4 x float> %768, float %770, i64 3, !dbg !34
  %math.result455 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %771) #9, !dbg !34, !frisk.register_lifetimes !1
  %.cvtpack.f02265 = extractelement <4 x float> %math.result455, i32 0, !dbg !34
  %.cvtpack.f12266 = extractelement <4 x float> %math.result455, i32 1, !dbg !34
  %.cvtpack.f22267 = extractelement <4 x float> %math.result455, i32 2, !dbg !34
  %.cvtpack.f32268 = extractelement <4 x float> %math.result455, i32 3, !dbg !34
  %.cvtpack.p02269 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02265, float %.cvtpack.f12266), !dbg !34
  %.cvtpack.p12270 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22267, float %.cvtpack.f32268), !dbg !34
  %.cvtpack.v02271 = insertelement <2 x i32> poison, i32 %.cvtpack.p02269, i32 0, !dbg !34
  %.cvtpack.v12272 = insertelement <2 x i32> %.cvtpack.v02271, i32 %.cvtpack.p12270, i32 1, !dbg !34
  %772 = bitcast <2 x i32> %.cvtpack.v12272 to <4 x half>, !dbg !34
  %math.packed456 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %772) #9, !dbg !32, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %773 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed456, <4 x half> %733, <4 x float> %758) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !35
  %774 = shufflevector <8 x float> %756, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %775 = fmul <4 x float> %774, %712, !dbg !35
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %776 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed456, <4 x half> %752, <4 x float> %775) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %777 = shufflevector <8 x half> %721, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %778 = fsub <4 x float> %552, %698, !dbg !28
  %779 = extractelement <4 x float> %778, i64 0, !dbg !32
  %780 = call float @llvm.amdgcn.exp2.f32(float %779), !dbg !32
  %local.reduce156 = fadd float %739, %780, !dbg !33, !frisk.local_reduction !1
  %781 = insertelement <4 x float> poison, float %780, i64 0, !dbg !34
  %782 = extractelement <4 x float> %778, i64 1, !dbg !32
  %783 = call float @llvm.amdgcn.exp2.f32(float %782), !dbg !32
  %local.reduce157 = fadd float %742, %783, !dbg !33, !frisk.local_reduction !1
  %784 = insertelement <4 x float> %781, float %783, i64 1, !dbg !34
  %785 = extractelement <4 x float> %778, i64 2, !dbg !32
  %786 = call float @llvm.amdgcn.exp2.f32(float %785), !dbg !32
  %local.reduce158 = fadd float %745, %786, !dbg !33, !frisk.local_reduction !1
  %787 = insertelement <4 x float> %784, float %786, i64 2, !dbg !34
  %788 = extractelement <4 x float> %778, i64 3, !dbg !32
  %789 = call float @llvm.amdgcn.exp2.f32(float %788), !dbg !32
  %local.reduce159 = fadd float %748, %789, !dbg !33, !frisk.local_reduction !1
  %790 = insertelement <4 x float> %787, float %789, i64 3, !dbg !34
  %math.result441 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %790) #9, !dbg !34, !frisk.register_lifetimes !1
  %.cvtpack.f02273 = extractelement <4 x float> %math.result441, i32 0, !dbg !34
  %.cvtpack.f12274 = extractelement <4 x float> %math.result441, i32 1, !dbg !34
  %.cvtpack.f22275 = extractelement <4 x float> %math.result441, i32 2, !dbg !34
  %.cvtpack.f32276 = extractelement <4 x float> %math.result441, i32 3, !dbg !34
  %.cvtpack.p02277 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02273, float %.cvtpack.f12274), !dbg !34
  %.cvtpack.p12278 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22275, float %.cvtpack.f32276), !dbg !34
  %.cvtpack.v02279 = insertelement <2 x i32> poison, i32 %.cvtpack.p02277, i32 0, !dbg !34
  %.cvtpack.v12280 = insertelement <2 x i32> %.cvtpack.v02279, i32 %.cvtpack.p12278, i32 1, !dbg !34
  %791 = bitcast <2 x i32> %.cvtpack.v12280 to <4 x half>, !dbg !34
  %math.packed442 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %791) #9, !dbg !33, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %792 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed442, <4 x half> %777, <4 x float> %751) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %793 = shufflevector <8 x half> %721, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %794 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed442, <4 x half> %793, <4 x float> %755) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %795 = fsub <4 x float> %612, %699, !dbg !28
  %796 = extractelement <4 x float> %795, i64 0, !dbg !32
  %797 = call float @llvm.amdgcn.exp2.f32(float %796), !dbg !32
  %local.reduce = fadd float %761, %797, !dbg !33, !frisk.local_reduction !1
  %798 = insertelement <4 x float> poison, float %797, i64 0, !dbg !34
  %799 = extractelement <4 x float> %795, i64 1, !dbg !32
  %800 = call float @llvm.amdgcn.exp2.f32(float %799), !dbg !32
  %local.reduce126 = fadd float %764, %800, !dbg !33, !frisk.local_reduction !1
  %801 = insertelement <4 x float> %798, float %800, i64 1, !dbg !34
  %802 = extractelement <4 x float> %795, i64 2, !dbg !32
  %803 = call float @llvm.amdgcn.exp2.f32(float %802), !dbg !32
  %local.reduce127 = fadd float %767, %803, !dbg !33, !frisk.local_reduction !1
  %804 = insertelement <4 x float> %801, float %803, i64 2, !dbg !34
  %805 = extractelement <4 x float> %795, i64 3, !dbg !32
  %806 = call float @llvm.amdgcn.exp2.f32(float %805), !dbg !32
  %local.reduce128 = fadd float %770, %806, !dbg !33, !frisk.local_reduction !1
  %807 = insertelement <4 x float> %804, float %806, i64 3, !dbg !34
  %math.result457 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %807) #9, !dbg !34, !frisk.register_lifetimes !1
  %.cvtpack.f02281 = extractelement <4 x float> %math.result457, i32 0, !dbg !34
  %.cvtpack.f12282 = extractelement <4 x float> %math.result457, i32 1, !dbg !34
  %.cvtpack.f22283 = extractelement <4 x float> %math.result457, i32 2, !dbg !34
  %.cvtpack.f32284 = extractelement <4 x float> %math.result457, i32 3, !dbg !34
  %.cvtpack.p02285 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02281, float %.cvtpack.f12282), !dbg !34
  %.cvtpack.p12286 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22283, float %.cvtpack.f32284), !dbg !34
  %.cvtpack.v02287 = insertelement <2 x i32> poison, i32 %.cvtpack.p02285, i32 0, !dbg !34
  %.cvtpack.v12288 = insertelement <2 x i32> %.cvtpack.v02287, i32 %.cvtpack.p12286, i32 1, !dbg !34
  %808 = bitcast <2 x i32> %.cvtpack.v12288 to <4 x half>, !dbg !34
  %math.packed458 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %808) #9, !dbg !33, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %809 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed458, <4 x half> %777, <4 x float> %773) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %810 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed458, <4 x half> %793, <4 x float> %776) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %811 = shufflevector <8 x half> %723, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %812 = fsub <4 x float> %563, %698, !dbg !28
  %813 = extractelement <4 x float> %812, i64 0, !dbg !32
  %814 = call float @llvm.amdgcn.exp2.f32(float %813), !dbg !32
  %local.reduce160 = fadd float %local.reduce156, %814, !dbg !33, !frisk.local_reduction !1
  %815 = insertelement <4 x float> poison, float %814, i64 0, !dbg !34
  %816 = extractelement <4 x float> %812, i64 1, !dbg !32
  %817 = call float @llvm.amdgcn.exp2.f32(float %816), !dbg !32
  %local.reduce161 = fadd float %local.reduce157, %817, !dbg !33, !frisk.local_reduction !1
  %818 = insertelement <4 x float> %815, float %817, i64 1, !dbg !34
  %819 = extractelement <4 x float> %812, i64 2, !dbg !32
  %820 = call float @llvm.amdgcn.exp2.f32(float %819), !dbg !32
  %local.reduce162 = fadd float %local.reduce158, %820, !dbg !33, !frisk.local_reduction !1
  %821 = insertelement <4 x float> %818, float %820, i64 2, !dbg !34
  %822 = extractelement <4 x float> %812, i64 3, !dbg !32
  %823 = call float @llvm.amdgcn.exp2.f32(float %822), !dbg !32
  %local.reduce163 = fadd float %local.reduce159, %823, !dbg !33, !frisk.local_reduction !1
  %824 = insertelement <4 x float> %821, float %823, i64 3, !dbg !34
  %math.result443 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %824) #9, !dbg !34, !frisk.register_lifetimes !1
  %.cvtpack.f02289 = extractelement <4 x float> %math.result443, i32 0, !dbg !34
  %.cvtpack.f12290 = extractelement <4 x float> %math.result443, i32 1, !dbg !34
  %.cvtpack.f22291 = extractelement <4 x float> %math.result443, i32 2, !dbg !34
  %.cvtpack.f32292 = extractelement <4 x float> %math.result443, i32 3, !dbg !34
  %.cvtpack.p02293 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02289, float %.cvtpack.f12290), !dbg !34
  %.cvtpack.p12294 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22291, float %.cvtpack.f32292), !dbg !34
  %.cvtpack.v02295 = insertelement <2 x i32> poison, i32 %.cvtpack.p02293, i32 0, !dbg !34
  %.cvtpack.v12296 = insertelement <2 x i32> %.cvtpack.v02295, i32 %.cvtpack.p12294, i32 1, !dbg !34
  %825 = bitcast <2 x i32> %.cvtpack.v12296 to <4 x half>, !dbg !34
  %math.packed444 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %825) #9, !dbg !33, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %826 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed444, <4 x half> %811, <4 x float> %792) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %827 = shufflevector <8 x half> %723, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %828 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed444, <4 x half> %827, <4 x float> %794) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %829 = fsub <4 x float> %615, %699, !dbg !28
  %830 = extractelement <4 x float> %829, i64 0, !dbg !32
  %831 = call float @llvm.amdgcn.exp2.f32(float %830), !dbg !32
  %local.reduce129 = fadd float %local.reduce, %831, !dbg !33, !frisk.local_reduction !1
  %832 = insertelement <4 x float> poison, float %831, i64 0, !dbg !34
  %833 = extractelement <4 x float> %829, i64 1, !dbg !32
  %834 = call float @llvm.amdgcn.exp2.f32(float %833), !dbg !32
  %local.reduce130 = fadd float %local.reduce126, %834, !dbg !33, !frisk.local_reduction !1
  %835 = insertelement <4 x float> %832, float %834, i64 1, !dbg !34
  %836 = extractelement <4 x float> %829, i64 2, !dbg !32
  %837 = call float @llvm.amdgcn.exp2.f32(float %836), !dbg !32
  %local.reduce131 = fadd float %local.reduce127, %837, !dbg !33, !frisk.local_reduction !1
  %838 = insertelement <4 x float> %835, float %837, i64 2, !dbg !34
  %839 = extractelement <4 x float> %829, i64 3, !dbg !32
  %840 = call float @llvm.amdgcn.exp2.f32(float %839), !dbg !32
  %local.reduce132 = fadd float %local.reduce128, %840, !dbg !33, !frisk.local_reduction !1
  %841 = insertelement <4 x float> %838, float %840, i64 3, !dbg !34
  %math.result459 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %841) #9, !dbg !34, !frisk.register_lifetimes !1
  %.cvtpack.f02297 = extractelement <4 x float> %math.result459, i32 0, !dbg !34
  %.cvtpack.f12298 = extractelement <4 x float> %math.result459, i32 1, !dbg !34
  %.cvtpack.f22299 = extractelement <4 x float> %math.result459, i32 2, !dbg !34
  %.cvtpack.f32300 = extractelement <4 x float> %math.result459, i32 3, !dbg !34
  %.cvtpack.p02301 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02297, float %.cvtpack.f12298), !dbg !34
  %.cvtpack.p12302 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22299, float %.cvtpack.f32300), !dbg !34
  %.cvtpack.v02303 = insertelement <2 x i32> poison, i32 %.cvtpack.p02301, i32 0, !dbg !34
  %.cvtpack.v12304 = insertelement <2 x i32> %.cvtpack.v02303, i32 %.cvtpack.p12302, i32 1, !dbg !34
  %842 = bitcast <2 x i32> %.cvtpack.v12304 to <4 x half>, !dbg !34
  %math.packed460 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %842) #9, !dbg !33, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %843 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed460, <4 x half> %811, <4 x float> %809) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %844 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed460, <4 x half> %827, <4 x float> %810) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %845 = shufflevector <8 x half> %725, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %846 = fsub <4 x float> %572, %698, !dbg !28
  %847 = extractelement <4 x float> %846, i64 0, !dbg !32
  %848 = call float @llvm.amdgcn.exp2.f32(float %847), !dbg !32
  %local.reduce164 = fadd float %local.reduce160, %848, !dbg !33, !frisk.local_reduction !1
  %849 = insertelement <4 x float> poison, float %848, i64 0, !dbg !34
  %850 = extractelement <4 x float> %846, i64 1, !dbg !32
  %851 = call float @llvm.amdgcn.exp2.f32(float %850), !dbg !32
  %local.reduce165 = fadd float %local.reduce161, %851, !dbg !33, !frisk.local_reduction !1
  %852 = insertelement <4 x float> %849, float %851, i64 1, !dbg !34
  %853 = extractelement <4 x float> %846, i64 2, !dbg !32
  %854 = call float @llvm.amdgcn.exp2.f32(float %853), !dbg !32
  %local.reduce166 = fadd float %local.reduce162, %854, !dbg !33, !frisk.local_reduction !1
  %855 = insertelement <4 x float> %852, float %854, i64 2, !dbg !34
  %856 = extractelement <4 x float> %846, i64 3, !dbg !32
  %857 = call float @llvm.amdgcn.exp2.f32(float %856), !dbg !32
  %local.reduce167 = fadd float %local.reduce163, %857, !dbg !33, !frisk.local_reduction !1
  %858 = insertelement <4 x float> %855, float %857, i64 3, !dbg !34
  %math.result445 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %858) #9, !dbg !34, !frisk.register_lifetimes !1
  %.cvtpack.f02305 = extractelement <4 x float> %math.result445, i32 0, !dbg !34
  %.cvtpack.f12306 = extractelement <4 x float> %math.result445, i32 1, !dbg !34
  %.cvtpack.f22307 = extractelement <4 x float> %math.result445, i32 2, !dbg !34
  %.cvtpack.f32308 = extractelement <4 x float> %math.result445, i32 3, !dbg !34
  %.cvtpack.p02309 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02305, float %.cvtpack.f12306), !dbg !34
  %.cvtpack.p12310 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22307, float %.cvtpack.f32308), !dbg !34
  %.cvtpack.v02311 = insertelement <2 x i32> poison, i32 %.cvtpack.p02309, i32 0, !dbg !34
  %.cvtpack.v12312 = insertelement <2 x i32> %.cvtpack.v02311, i32 %.cvtpack.p12310, i32 1, !dbg !34
  %859 = bitcast <2 x i32> %.cvtpack.v12312 to <4 x half>, !dbg !34
  %math.packed446 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %859) #9, !dbg !33, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %860 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed446, <4 x half> %845, <4 x float> %826) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %861 = shufflevector <8 x half> %725, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %862 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed446, <4 x half> %861, <4 x float> %828) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %863 = fsub <4 x float> %618, %699, !dbg !28
  %864 = extractelement <4 x float> %863, i64 0, !dbg !32
  %865 = call float @llvm.amdgcn.exp2.f32(float %864), !dbg !32
  %local.reduce133 = fadd float %local.reduce129, %865, !dbg !33, !frisk.local_reduction !1
  %866 = insertelement <4 x float> poison, float %865, i64 0, !dbg !34
  %867 = extractelement <4 x float> %863, i64 1, !dbg !32
  %868 = call float @llvm.amdgcn.exp2.f32(float %867), !dbg !32
  %local.reduce134 = fadd float %local.reduce130, %868, !dbg !33, !frisk.local_reduction !1
  %869 = insertelement <4 x float> %866, float %868, i64 1, !dbg !34
  %870 = extractelement <4 x float> %863, i64 2, !dbg !32
  %871 = call float @llvm.amdgcn.exp2.f32(float %870), !dbg !32
  %local.reduce135 = fadd float %local.reduce131, %871, !dbg !33, !frisk.local_reduction !1
  %872 = insertelement <4 x float> %869, float %871, i64 2, !dbg !34
  %873 = extractelement <4 x float> %863, i64 3, !dbg !32
  %874 = call float @llvm.amdgcn.exp2.f32(float %873), !dbg !32
  %local.reduce136 = fadd float %local.reduce132, %874, !dbg !33, !frisk.local_reduction !1
  %875 = insertelement <4 x float> %872, float %874, i64 3, !dbg !34
  %math.result461 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %875) #9, !dbg !34, !frisk.register_lifetimes !1
  %.cvtpack.f02313 = extractelement <4 x float> %math.result461, i32 0, !dbg !34
  %.cvtpack.f12314 = extractelement <4 x float> %math.result461, i32 1, !dbg !34
  %.cvtpack.f22315 = extractelement <4 x float> %math.result461, i32 2, !dbg !34
  %.cvtpack.f32316 = extractelement <4 x float> %math.result461, i32 3, !dbg !34
  %.cvtpack.p02317 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02313, float %.cvtpack.f12314), !dbg !34
  %.cvtpack.p12318 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22315, float %.cvtpack.f32316), !dbg !34
  %.cvtpack.v02319 = insertelement <2 x i32> poison, i32 %.cvtpack.p02317, i32 0, !dbg !34
  %.cvtpack.v12320 = insertelement <2 x i32> %.cvtpack.v02319, i32 %.cvtpack.p12318, i32 1, !dbg !34
  %876 = bitcast <2 x i32> %.cvtpack.v12320 to <4 x half>, !dbg !34
  %math.packed462 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %876) #9, !dbg !33, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %877 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed462, <4 x half> %845, <4 x float> %843) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %878 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed462, <4 x half> %861, <4 x float> %844) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %879 = shufflevector <8 x half> %727, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %880 = fsub <4 x float> %579, %698, !dbg !28
  %881 = extractelement <4 x float> %880, i64 0, !dbg !32
  %882 = call float @llvm.amdgcn.exp2.f32(float %881), !dbg !32
  %local.reduce168 = fadd float %local.reduce164, %882, !dbg !33, !frisk.local_reduction !1
  %883 = insertelement <4 x float> poison, float %882, i64 0, !dbg !34
  %884 = extractelement <4 x float> %880, i64 1, !dbg !32
  %885 = call float @llvm.amdgcn.exp2.f32(float %884), !dbg !32
  %local.reduce169 = fadd float %local.reduce165, %885, !dbg !33, !frisk.local_reduction !1
  %886 = insertelement <4 x float> %883, float %885, i64 1, !dbg !34
  %887 = extractelement <4 x float> %880, i64 2, !dbg !32
  %888 = call float @llvm.amdgcn.exp2.f32(float %887), !dbg !32
  %local.reduce170 = fadd float %local.reduce166, %888, !dbg !33, !frisk.local_reduction !1
  %889 = insertelement <4 x float> %886, float %888, i64 2, !dbg !34
  %890 = extractelement <4 x float> %880, i64 3, !dbg !32
  %891 = call float @llvm.amdgcn.exp2.f32(float %890), !dbg !32
  %local.reduce171 = fadd float %local.reduce167, %891, !dbg !33, !frisk.local_reduction !1
  %892 = insertelement <4 x float> %889, float %891, i64 3, !dbg !34
  %math.result447 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %892) #9, !dbg !34, !frisk.register_lifetimes !1
  %.cvtpack.f02321 = extractelement <4 x float> %math.result447, i32 0, !dbg !34
  %.cvtpack.f12322 = extractelement <4 x float> %math.result447, i32 1, !dbg !34
  %.cvtpack.f22323 = extractelement <4 x float> %math.result447, i32 2, !dbg !34
  %.cvtpack.f32324 = extractelement <4 x float> %math.result447, i32 3, !dbg !34
  %.cvtpack.p02325 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02321, float %.cvtpack.f12322), !dbg !34
  %.cvtpack.p12326 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22323, float %.cvtpack.f32324), !dbg !34
  %.cvtpack.v02327 = insertelement <2 x i32> poison, i32 %.cvtpack.p02325, i32 0, !dbg !34
  %.cvtpack.v12328 = insertelement <2 x i32> %.cvtpack.v02327, i32 %.cvtpack.p12326, i32 1, !dbg !34
  %893 = bitcast <2 x i32> %.cvtpack.v12328 to <4 x half>, !dbg !34
  %math.packed448 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %893) #9, !dbg !33, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %894 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed448, <4 x half> %879, <4 x float> %860) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %895 = shufflevector <8 x half> %727, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %896 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed448, <4 x half> %895, <4 x float> %862) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %897 = fsub <4 x float> %621, %699, !dbg !28
  %898 = extractelement <4 x float> %897, i64 0, !dbg !32
  %899 = call float @llvm.amdgcn.exp2.f32(float %898), !dbg !32
  %local.reduce137 = fadd float %local.reduce133, %899, !dbg !33, !frisk.local_reduction !1
  %900 = insertelement <4 x float> poison, float %899, i64 0, !dbg !34
  %901 = extractelement <4 x float> %897, i64 1, !dbg !32
  %902 = call float @llvm.amdgcn.exp2.f32(float %901), !dbg !32
  %local.reduce138 = fadd float %local.reduce134, %902, !dbg !33, !frisk.local_reduction !1
  %903 = insertelement <4 x float> %900, float %902, i64 1, !dbg !34
  %904 = extractelement <4 x float> %897, i64 2, !dbg !32
  %905 = call float @llvm.amdgcn.exp2.f32(float %904), !dbg !32
  %local.reduce139 = fadd float %local.reduce135, %905, !dbg !33, !frisk.local_reduction !1
  %906 = insertelement <4 x float> %903, float %905, i64 2, !dbg !34
  %907 = extractelement <4 x float> %897, i64 3, !dbg !32
  %908 = call float @llvm.amdgcn.exp2.f32(float %907), !dbg !32
  %local.reduce140 = fadd float %local.reduce136, %908, !dbg !33, !frisk.local_reduction !1
  %909 = insertelement <4 x float> %906, float %908, i64 3, !dbg !34
  %math.result463 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %909) #9, !dbg !34, !frisk.register_lifetimes !1
  %.cvtpack.f02329 = extractelement <4 x float> %math.result463, i32 0, !dbg !34
  %.cvtpack.f12330 = extractelement <4 x float> %math.result463, i32 1, !dbg !34
  %.cvtpack.f22331 = extractelement <4 x float> %math.result463, i32 2, !dbg !34
  %.cvtpack.f32332 = extractelement <4 x float> %math.result463, i32 3, !dbg !34
  %.cvtpack.p02333 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02329, float %.cvtpack.f12330), !dbg !34
  %.cvtpack.p12334 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22331, float %.cvtpack.f32332), !dbg !34
  %.cvtpack.v02335 = insertelement <2 x i32> poison, i32 %.cvtpack.p02333, i32 0, !dbg !34
  %.cvtpack.v12336 = insertelement <2 x i32> %.cvtpack.v02335, i32 %.cvtpack.p12334, i32 1, !dbg !34
  %910 = bitcast <2 x i32> %.cvtpack.v12336 to <4 x half>, !dbg !34
  %math.packed464 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %910) #9, !dbg !33, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %911 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed464, <4 x half> %879, <4 x float> %877) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %912 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed464, <4 x half> %895, <4 x float> %878) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %913 = shufflevector <8 x half> %729, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %914 = fsub <4 x float> %587, %698, !dbg !28
  %915 = extractelement <4 x float> %914, i64 0, !dbg !32
  %916 = call float @llvm.amdgcn.exp2.f32(float %915), !dbg !32
  %local.reduce172 = fadd float %local.reduce168, %916, !dbg !33, !frisk.local_reduction !1
  %917 = insertelement <4 x float> poison, float %916, i64 0, !dbg !34
  %918 = extractelement <4 x float> %914, i64 1, !dbg !32
  %919 = call float @llvm.amdgcn.exp2.f32(float %918), !dbg !32
  %local.reduce173 = fadd float %local.reduce169, %919, !dbg !33, !frisk.local_reduction !1
  %920 = insertelement <4 x float> %917, float %919, i64 1, !dbg !34
  %921 = extractelement <4 x float> %914, i64 2, !dbg !32
  %922 = call float @llvm.amdgcn.exp2.f32(float %921), !dbg !32
  %local.reduce174 = fadd float %local.reduce170, %922, !dbg !33, !frisk.local_reduction !1
  %923 = insertelement <4 x float> %920, float %922, i64 2, !dbg !34
  %924 = extractelement <4 x float> %914, i64 3, !dbg !32
  %925 = call float @llvm.amdgcn.exp2.f32(float %924), !dbg !32
  %local.reduce175 = fadd float %local.reduce171, %925, !dbg !33, !frisk.local_reduction !1
  %926 = insertelement <4 x float> %923, float %925, i64 3, !dbg !34
  %math.result449 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %926) #9, !dbg !34, !frisk.register_lifetimes !1
  %.cvtpack.f02337 = extractelement <4 x float> %math.result449, i32 0, !dbg !34
  %.cvtpack.f12338 = extractelement <4 x float> %math.result449, i32 1, !dbg !34
  %.cvtpack.f22339 = extractelement <4 x float> %math.result449, i32 2, !dbg !34
  %.cvtpack.f32340 = extractelement <4 x float> %math.result449, i32 3, !dbg !34
  %.cvtpack.p02341 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02337, float %.cvtpack.f12338), !dbg !34
  %.cvtpack.p12342 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22339, float %.cvtpack.f32340), !dbg !34
  %.cvtpack.v02343 = insertelement <2 x i32> poison, i32 %.cvtpack.p02341, i32 0, !dbg !34
  %.cvtpack.v12344 = insertelement <2 x i32> %.cvtpack.v02343, i32 %.cvtpack.p12342, i32 1, !dbg !34
  %927 = bitcast <2 x i32> %.cvtpack.v12344 to <4 x half>, !dbg !34
  %math.packed450 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %927) #9, !dbg !33, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %928 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed450, <4 x half> %913, <4 x float> %894) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %929 = shufflevector <8 x half> %729, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %930 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed450, <4 x half> %929, <4 x float> %896) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %931 = fsub <4 x float> %624, %699, !dbg !28
  %932 = extractelement <4 x float> %931, i64 0, !dbg !32
  %933 = call float @llvm.amdgcn.exp2.f32(float %932), !dbg !32
  %local.reduce141 = fadd float %local.reduce137, %933, !dbg !33, !frisk.local_reduction !1
  %934 = insertelement <4 x float> poison, float %933, i64 0, !dbg !34
  %935 = extractelement <4 x float> %931, i64 1, !dbg !32
  %936 = call float @llvm.amdgcn.exp2.f32(float %935), !dbg !32
  %local.reduce142 = fadd float %local.reduce138, %936, !dbg !33, !frisk.local_reduction !1
  %937 = insertelement <4 x float> %934, float %936, i64 1, !dbg !34
  %938 = extractelement <4 x float> %931, i64 2, !dbg !32
  %939 = call float @llvm.amdgcn.exp2.f32(float %938), !dbg !32
  %local.reduce143 = fadd float %local.reduce139, %939, !dbg !33, !frisk.local_reduction !1
  %940 = insertelement <4 x float> %937, float %939, i64 2, !dbg !34
  %941 = extractelement <4 x float> %931, i64 3, !dbg !32
  %942 = call float @llvm.amdgcn.exp2.f32(float %941), !dbg !32
  %local.reduce144 = fadd float %local.reduce140, %942, !dbg !33, !frisk.local_reduction !1
  %943 = insertelement <4 x float> %940, float %942, i64 3, !dbg !34
  %math.result465 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %943) #9, !dbg !34, !frisk.register_lifetimes !1
  %.cvtpack.f02345 = extractelement <4 x float> %math.result465, i32 0, !dbg !34
  %.cvtpack.f12346 = extractelement <4 x float> %math.result465, i32 1, !dbg !34
  %.cvtpack.f22347 = extractelement <4 x float> %math.result465, i32 2, !dbg !34
  %.cvtpack.f32348 = extractelement <4 x float> %math.result465, i32 3, !dbg !34
  %.cvtpack.p02349 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02345, float %.cvtpack.f12346), !dbg !34
  %.cvtpack.p12350 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22347, float %.cvtpack.f32348), !dbg !34
  %.cvtpack.v02351 = insertelement <2 x i32> poison, i32 %.cvtpack.p02349, i32 0, !dbg !34
  %.cvtpack.v12352 = insertelement <2 x i32> %.cvtpack.v02351, i32 %.cvtpack.p12350, i32 1, !dbg !34
  %944 = bitcast <2 x i32> %.cvtpack.v12352 to <4 x half>, !dbg !34
  %math.packed466 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %944) #9, !dbg !33, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %945 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed466, <4 x half> %913, <4 x float> %911) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %946 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed466, <4 x half> %929, <4 x float> %912) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %947 = shufflevector <8 x half> %731, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %948 = fsub <4 x float> %594, %698, !dbg !28
  %949 = extractelement <4 x float> %948, i64 0, !dbg !32
  %950 = call float @llvm.amdgcn.exp2.f32(float %949), !dbg !32
  %local.reduce176 = fadd float %local.reduce172, %950, !dbg !33, !frisk.local_reduction !1
  %951 = insertelement <4 x float> poison, float %950, i64 0, !dbg !34
  %952 = extractelement <4 x float> %948, i64 1, !dbg !32
  %953 = call float @llvm.amdgcn.exp2.f32(float %952), !dbg !32
  %local.reduce177 = fadd float %local.reduce173, %953, !dbg !33, !frisk.local_reduction !1
  %954 = insertelement <4 x float> %951, float %953, i64 1, !dbg !34
  %955 = extractelement <4 x float> %948, i64 2, !dbg !32
  %956 = call float @llvm.amdgcn.exp2.f32(float %955), !dbg !32
  %local.reduce178 = fadd float %local.reduce174, %956, !dbg !33, !frisk.local_reduction !1
  %957 = insertelement <4 x float> %954, float %956, i64 2, !dbg !34
  %958 = extractelement <4 x float> %948, i64 3, !dbg !32
  %959 = call float @llvm.amdgcn.exp2.f32(float %958), !dbg !32
  %local.reduce179 = fadd float %local.reduce175, %959, !dbg !33, !frisk.local_reduction !1
  %960 = insertelement <4 x float> %957, float %959, i64 3, !dbg !34
  %math.result451 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %960) #9, !dbg !34, !frisk.register_lifetimes !1
  %.cvtpack.f02353 = extractelement <4 x float> %math.result451, i32 0, !dbg !34
  %.cvtpack.f12354 = extractelement <4 x float> %math.result451, i32 1, !dbg !34
  %.cvtpack.f22355 = extractelement <4 x float> %math.result451, i32 2, !dbg !34
  %.cvtpack.f32356 = extractelement <4 x float> %math.result451, i32 3, !dbg !34
  %.cvtpack.p02357 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02353, float %.cvtpack.f12354), !dbg !34
  %.cvtpack.p12358 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22355, float %.cvtpack.f32356), !dbg !34
  %.cvtpack.v02359 = insertelement <2 x i32> poison, i32 %.cvtpack.p02357, i32 0, !dbg !34
  %.cvtpack.v12360 = insertelement <2 x i32> %.cvtpack.v02359, i32 %.cvtpack.p12358, i32 1, !dbg !34
  %961 = bitcast <2 x i32> %.cvtpack.v12360 to <4 x half>, !dbg !34
  %math.packed452 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %961) #9, !dbg !33, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %962 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed452, <4 x half> %947, <4 x float> %928) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %963 = shufflevector <8 x half> %731, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %964 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed452, <4 x half> %963, <4 x float> %930) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %965 = fsub <4 x float> %627, %699, !dbg !28
  %966 = extractelement <4 x float> %965, i64 0, !dbg !32
  %967 = call float @llvm.amdgcn.exp2.f32(float %966), !dbg !32
  %local.reduce145 = fadd float %local.reduce141, %967, !dbg !33, !frisk.local_reduction !1
  %968 = insertelement <4 x float> poison, float %967, i64 0, !dbg !34
  %969 = extractelement <4 x float> %965, i64 1, !dbg !32
  %970 = call float @llvm.amdgcn.exp2.f32(float %969), !dbg !32
  %local.reduce146 = fadd float %local.reduce142, %970, !dbg !33, !frisk.local_reduction !1
  %971 = insertelement <4 x float> %968, float %970, i64 1, !dbg !34
  %972 = extractelement <4 x float> %965, i64 2, !dbg !32
  %973 = call float @llvm.amdgcn.exp2.f32(float %972), !dbg !32
  %local.reduce147 = fadd float %local.reduce143, %973, !dbg !33, !frisk.local_reduction !1
  %974 = insertelement <4 x float> %971, float %973, i64 2, !dbg !34
  %975 = extractelement <4 x float> %965, i64 3, !dbg !32
  %976 = call float @llvm.amdgcn.exp2.f32(float %975), !dbg !32
  %local.reduce148 = fadd float %local.reduce144, %976, !dbg !33, !frisk.local_reduction !1
  %977 = insertelement <4 x float> %974, float %976, i64 3, !dbg !34
  %math.result467 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %977) #9, !dbg !34, !frisk.register_lifetimes !1
  %.cvtpack.f02361 = extractelement <4 x float> %math.result467, i32 0, !dbg !34
  %.cvtpack.f12362 = extractelement <4 x float> %math.result467, i32 1, !dbg !34
  %.cvtpack.f22363 = extractelement <4 x float> %math.result467, i32 2, !dbg !34
  %.cvtpack.f32364 = extractelement <4 x float> %math.result467, i32 3, !dbg !34
  %.cvtpack.p02365 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02361, float %.cvtpack.f12362), !dbg !34
  %.cvtpack.p12366 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22363, float %.cvtpack.f32364), !dbg !34
  %.cvtpack.v02367 = insertelement <2 x i32> poison, i32 %.cvtpack.p02365, i32 0, !dbg !34
  %.cvtpack.v12368 = insertelement <2 x i32> %.cvtpack.v02367, i32 %.cvtpack.p12366, i32 1, !dbg !34
  %978 = bitcast <2 x i32> %.cvtpack.v12368 to <4 x half>, !dbg !34
  %math.packed468 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %978) #9, !dbg !33, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %979 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed468, <4 x half> %947, <4 x float> %945) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %980 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed468, <4 x half> %963, <4 x float> %946) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %981 = bitcast <4 x i32> %732 to <8 x half>, !dbg !13
  %982 = shufflevector <8 x half> %981, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %983 = fsub <4 x float> %601, %698, !dbg !28
  %984 = extractelement <4 x float> %983, i64 0, !dbg !32
  %985 = call float @llvm.amdgcn.exp2.f32(float %984), !dbg !32
  %local.reduce180 = fadd float %local.reduce176, %985, !dbg !33, !frisk.local_reduction !1
  %986 = insertelement <4 x float> poison, float %985, i64 0, !dbg !34
  %987 = extractelement <4 x float> %983, i64 1, !dbg !32
  %988 = call float @llvm.amdgcn.exp2.f32(float %987), !dbg !32
  %local.reduce181 = fadd float %local.reduce177, %988, !dbg !33, !frisk.local_reduction !1
  %local.reduce184 = fadd float %local.reduce180, %local.reduce181, !dbg !33, !frisk.local_reduction !1
  %989 = insertelement <4 x float> %986, float %988, i64 1, !dbg !34
  %990 = extractelement <4 x float> %983, i64 2, !dbg !32
  %991 = call float @llvm.amdgcn.exp2.f32(float %990), !dbg !32
  %local.reduce182 = fadd float %local.reduce178, %991, !dbg !33, !frisk.local_reduction !1
  %992 = insertelement <4 x float> %989, float %991, i64 2, !dbg !34
  %993 = extractelement <4 x float> %983, i64 3, !dbg !32
  %994 = call float @llvm.amdgcn.exp2.f32(float %993), !dbg !32
  %local.reduce183 = fadd float %local.reduce179, %994, !dbg !33, !frisk.local_reduction !1
  %local.reduce185 = fadd float %local.reduce182, %local.reduce183, !dbg !33, !frisk.local_reduction !1
  %995 = fadd float %local.reduce184, %local.reduce185, !dbg !33, !frisk.local_reduction !1
  %996 = fadd float %995, %707, !dbg !33
  %997 = insertelement <1 x float> poison, float %996, i64 0, !dbg !7
  %998 = insertvalue [2 x <1 x float>] poison, <1 x float> %997, 0, !dbg !7
  %999 = insertelement <4 x float> %992, float %994, i64 3, !dbg !34
  %math.result453 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %999) #9, !dbg !34, !frisk.register_lifetimes !1
  %.cvtpack.f02369 = extractelement <4 x float> %math.result453, i32 0, !dbg !34
  %.cvtpack.f12370 = extractelement <4 x float> %math.result453, i32 1, !dbg !34
  %.cvtpack.f22371 = extractelement <4 x float> %math.result453, i32 2, !dbg !34
  %.cvtpack.f32372 = extractelement <4 x float> %math.result453, i32 3, !dbg !34
  %.cvtpack.p02373 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02369, float %.cvtpack.f12370), !dbg !34
  %.cvtpack.p12374 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22371, float %.cvtpack.f32372), !dbg !34
  %.cvtpack.v02375 = insertelement <2 x i32> poison, i32 %.cvtpack.p02373, i32 0, !dbg !34
  %.cvtpack.v12376 = insertelement <2 x i32> %.cvtpack.v02375, i32 %.cvtpack.p12374, i32 1, !dbg !34
  %1000 = bitcast <2 x i32> %.cvtpack.v12376 to <4 x half>, !dbg !34
  %math.packed454 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1000) #9, !dbg !33, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %1001 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed454, <4 x half> %982, <4 x float> %962) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1002 = shufflevector <8 x half> %981, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1003 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed454, <4 x half> %1002, <4 x float> %964) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1004 = fsub <4 x float> %641, %699, !dbg !28
  %1005 = extractelement <4 x float> %1004, i64 0, !dbg !32
  %1006 = call float @llvm.amdgcn.exp2.f32(float %1005), !dbg !32
  %local.reduce149 = fadd float %local.reduce145, %1006, !dbg !33, !frisk.local_reduction !1
  %1007 = insertelement <4 x float> poison, float %1006, i64 0, !dbg !34
  %1008 = extractelement <4 x float> %1004, i64 1, !dbg !32
  %1009 = call float @llvm.amdgcn.exp2.f32(float %1008), !dbg !32
  %local.reduce150 = fadd float %local.reduce146, %1009, !dbg !33, !frisk.local_reduction !1
  %local.reduce153 = fadd float %local.reduce149, %local.reduce150, !dbg !33, !frisk.local_reduction !1
  %1010 = insertelement <4 x float> %1007, float %1009, i64 1, !dbg !34
  %1011 = extractelement <4 x float> %1004, i64 2, !dbg !32
  %1012 = call float @llvm.amdgcn.exp2.f32(float %1011), !dbg !32
  %local.reduce151 = fadd float %local.reduce147, %1012, !dbg !33, !frisk.local_reduction !1
  %1013 = insertelement <4 x float> %1010, float %1012, i64 2, !dbg !34
  %1014 = extractelement <4 x float> %1004, i64 3, !dbg !32
  %1015 = call float @llvm.amdgcn.exp2.f32(float %1014), !dbg !32
  %local.reduce152 = fadd float %local.reduce148, %1015, !dbg !33, !frisk.local_reduction !1
  %local.reduce154 = fadd float %local.reduce151, %local.reduce152, !dbg !33, !frisk.local_reduction !1
  %1016 = fadd float %local.reduce153, %local.reduce154, !dbg !33, !frisk.local_reduction !1
  %1017 = fadd float %1016, %715, !dbg !33
  %1018 = insertelement <1 x float> poison, float %1017, i64 0, !dbg !7
  %1019 = insertvalue [2 x <1 x float>] %998, <1 x float> %1018, 1, !dbg !7
  %1020 = insertelement <4 x float> %1013, float %1015, i64 3, !dbg !34
  %math.result469 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1020) #9, !dbg !34, !frisk.register_lifetimes !1
  %.cvtpack.f02377 = extractelement <4 x float> %math.result469, i32 0, !dbg !34
  %.cvtpack.f12378 = extractelement <4 x float> %math.result469, i32 1, !dbg !34
  %.cvtpack.f22379 = extractelement <4 x float> %math.result469, i32 2, !dbg !34
  %.cvtpack.f32380 = extractelement <4 x float> %math.result469, i32 3, !dbg !34
  %.cvtpack.p02381 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02377, float %.cvtpack.f12378), !dbg !34
  %.cvtpack.p12382 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22379, float %.cvtpack.f32380), !dbg !34
  %.cvtpack.v02383 = insertelement <2 x i32> poison, i32 %.cvtpack.p02381, i32 0, !dbg !34
  %.cvtpack.v12384 = insertelement <2 x i32> %.cvtpack.v02383, i32 %.cvtpack.p12382, i32 1, !dbg !34
  %1021 = bitcast <2 x i32> %.cvtpack.v12384 to <4 x half>, !dbg !34
  %math.packed470 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1021) #9, !dbg !33, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %1022 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed470, <4 x half> %982, <4 x float> %979) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1023 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed470, <4 x half> %1002, <4 x float> %980) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %vdb.address1814 = or disjoint i64 %.idx17641766, 64, !dbg !17
  %vdb.address1815 = or disjoint i64 %address.local988, %vdb.address1814, !dbg !17
  %vdb.address1816 = trunc i64 %vdb.address1815 to i32, !dbg !17
  %vdb.address1817 = shl i32 %vdb.address1816, 1, !dbg !17
  %vdb.address1824 = or disjoint i64 %address.local994, %vdb.address1814, !dbg !17
  %vdb.address1825 = trunc i64 %vdb.address1824 to i32, !dbg !17
  %vdb.address1826 = shl i32 %vdb.address1825, 1, !dbg !17
  %1024 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %137, i32 %vdb.address1817, <4 x i32> %dma.buffer.rsrc1670) #10, !dbg !17
  %1025 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %144, i32 %vdb.address1826, <4 x i32> %dma.buffer.rsrc1670) #10, !dbg !17
  %vdb.read = getelementptr i8, ptr addrspace(3) %address.local1002, i64 8192, !dbg !13
  %1026 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #0, !dbg !13, !frisk.v_double_buffer !41
  %1027 = bitcast <4 x i32> %1026 to <8 x half>, !dbg !13
  %vdb.read1806 = getelementptr i8, ptr addrspace(3) %address.local1011, i64 8448, !dbg !13
  %1028 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1806, i32 0) #0, !dbg !13, !frisk.v_double_buffer !41
  %1029 = bitcast <4 x i32> %1028 to <8 x half>, !dbg !13
  %vdb.read1807 = getelementptr i8, ptr addrspace(3) %address.local1011, i64 8704, !dbg !13
  %1030 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1807, i32 0) #0, !dbg !13, !frisk.v_double_buffer !41
  %1031 = bitcast <4 x i32> %1030 to <8 x half>, !dbg !13
  %vdb.read1808 = getelementptr i8, ptr addrspace(3) %address.local1011, i64 8960, !dbg !13
  %1032 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1808, i32 0) #0, !dbg !13, !frisk.v_double_buffer !41
  %1033 = bitcast <4 x i32> %1032 to <8 x half>, !dbg !13
  %vdb.read1809 = getelementptr i8, ptr addrspace(3) %address.local1002, i64 12288, !dbg !13
  %1034 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1809, i32 0) #0, !dbg !13, !frisk.v_double_buffer !41
  %1035 = bitcast <4 x i32> %1034 to <8 x half>, !dbg !13
  %vdb.read1810 = getelementptr i8, ptr addrspace(3) %address.local1011, i64 12544, !dbg !13
  %1036 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1810, i32 0) #0, !dbg !13, !frisk.v_double_buffer !41
  %1037 = bitcast <4 x i32> %1036 to <8 x half>, !dbg !13
  %vdb.read1811 = getelementptr i8, ptr addrspace(3) %address.local1011, i64 12800, !dbg !13
  %1038 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1811, i32 0) #0, !dbg !13, !frisk.v_double_buffer !41
  %1039 = bitcast <4 x i32> %1038 to <8 x half>, !dbg !13
  %vdb.read1812 = getelementptr i8, ptr addrspace(3) %address.local1011, i64 13056, !dbg !13
  %1040 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1812, i32 0) #0, !dbg !13, !frisk.v_double_buffer !41
  %1041 = shufflevector <8 x half> %1027, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1042 = extractvalue [2 x <8 x float>] %122, 0, !dbg !35
  %1043 = shufflevector <8 x float> %1042, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1044 = fmul <4 x float> %1043, %704, !dbg !35
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1045 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1041, <4 x float> %1044) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1046 = shufflevector <8 x half> %1027, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1047 = shufflevector <8 x float> %1042, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1048 = fmul <4 x float> %1047, %704, !dbg !35
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1049 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1046, <4 x float> %1048) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !35
  %1050 = extractvalue [2 x <8 x float>] %122, 1, !dbg !35
  %1051 = shufflevector <8 x float> %1050, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1052 = fmul <4 x float> %1051, %712, !dbg !35
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1053 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed456, <4 x half> %1041, <4 x float> %1052) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !35
  %1054 = shufflevector <8 x float> %1050, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1055 = fmul <4 x float> %1054, %712, !dbg !35
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1056 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed456, <4 x half> %1046, <4 x float> %1055) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1057 = shufflevector <8 x half> %1029, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1058 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed442, <4 x half> %1057, <4 x float> %1045) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1059 = shufflevector <8 x half> %1029, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1060 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed442, <4 x half> %1059, <4 x float> %1049) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1061 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed458, <4 x half> %1057, <4 x float> %1053) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1062 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed458, <4 x half> %1059, <4 x float> %1056) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1063 = shufflevector <8 x half> %1031, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1064 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed444, <4 x half> %1063, <4 x float> %1058) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1065 = shufflevector <8 x half> %1031, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1066 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed444, <4 x half> %1065, <4 x float> %1060) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1067 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed460, <4 x half> %1063, <4 x float> %1061) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1068 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed460, <4 x half> %1065, <4 x float> %1062) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1069 = shufflevector <8 x half> %1033, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1070 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed446, <4 x half> %1069, <4 x float> %1064) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1071 = shufflevector <8 x half> %1033, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1072 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed446, <4 x half> %1071, <4 x float> %1066) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1073 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed462, <4 x half> %1069, <4 x float> %1067) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1074 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed462, <4 x half> %1071, <4 x float> %1068) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1075 = shufflevector <8 x half> %1035, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1076 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed448, <4 x half> %1075, <4 x float> %1070) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1077 = shufflevector <8 x half> %1035, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1078 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed448, <4 x half> %1077, <4 x float> %1072) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1079 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed464, <4 x half> %1075, <4 x float> %1073) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1080 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed464, <4 x half> %1077, <4 x float> %1074) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1081 = shufflevector <8 x half> %1037, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1082 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed450, <4 x half> %1081, <4 x float> %1076) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1083 = shufflevector <8 x half> %1037, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1084 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed450, <4 x half> %1083, <4 x float> %1078) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1085 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed466, <4 x half> %1081, <4 x float> %1079) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1086 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed466, <4 x half> %1083, <4 x float> %1080) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1087 = shufflevector <8 x half> %1039, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1088 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed452, <4 x half> %1087, <4 x float> %1082) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1089 = shufflevector <8 x half> %1039, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1090 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed452, <4 x half> %1089, <4 x float> %1084) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1091 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed468, <4 x half> %1087, <4 x float> %1085) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1092 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed468, <4 x half> %1089, <4 x float> %1086) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1093 = bitcast <4 x i32> %1040 to <8 x half>, !dbg !13
  %1094 = shufflevector <8 x half> %1093, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1095 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed454, <4 x half> %1094, <4 x float> %1088) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1096 = shufflevector <8 x half> %1093, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1097 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed454, <4 x half> %1096, <4 x float> %1090) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1098 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed470, <4 x half> %1094, <4 x float> %1091) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1099 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed470, <4 x half> %1096, <4 x float> %1092) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %vdb.address1832 = or disjoint i64 %.idx17641766, 96, !dbg !17
  %vdb.address1833 = or disjoint i64 %address.local988, %vdb.address1832, !dbg !17
  %vdb.address1834 = trunc i64 %vdb.address1833 to i32, !dbg !17
  %vdb.address1835 = shl i32 %vdb.address1834, 1, !dbg !17
  %vdb.address1842 = or disjoint i64 %address.local994, %vdb.address1832, !dbg !17
  %vdb.address1843 = trunc i64 %vdb.address1842 to i32, !dbg !17
  %vdb.address1844 = shl i32 %vdb.address1843, 1, !dbg !17
  %1100 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1835, <4 x i32> %dma.buffer.rsrc1670) #10, !dbg !17
  %1101 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1805, i32 %vdb.address1844, <4 x i32> %dma.buffer.rsrc1670) #10, !dbg !17
  %1102 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1002, i32 0) #0, !dbg !13, !frisk.v_double_buffer !41
  %1103 = bitcast <4 x i32> %1102 to <8 x half>, !dbg !13
  %1104 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1012, i32 0) #0, !dbg !13, !frisk.v_double_buffer !41
  %1105 = bitcast <4 x i32> %1104 to <8 x half>, !dbg !13
  %1106 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1016, i32 0) #0, !dbg !13, !frisk.v_double_buffer !41
  %1107 = bitcast <4 x i32> %1106 to <8 x half>, !dbg !13
  %1108 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1020, i32 0) #0, !dbg !13, !frisk.v_double_buffer !41
  %1109 = bitcast <4 x i32> %1108 to <8 x half>, !dbg !13
  %1110 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1022, i32 0) #0, !dbg !13, !frisk.v_double_buffer !41
  %1111 = bitcast <4 x i32> %1110 to <8 x half>, !dbg !13
  %1112 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1026, i32 0) #0, !dbg !13, !frisk.v_double_buffer !41
  %1113 = bitcast <4 x i32> %1112 to <8 x half>, !dbg !13
  %1114 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1030, i32 0) #0, !dbg !13, !frisk.v_double_buffer !41
  %1115 = bitcast <4 x i32> %1114 to <8 x half>, !dbg !13
  %1116 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1034, i32 0) #0, !dbg !13, !frisk.v_double_buffer !41
  %1117 = shufflevector <8 x half> %1103, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1118 = extractvalue [2 x <8 x float>] %123, 0, !dbg !35
  %1119 = shufflevector <8 x float> %1118, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1120 = fmul <4 x float> %1119, %704, !dbg !35
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1121 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1117, <4 x float> %1120) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1122 = shufflevector <8 x half> %1103, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1123 = shufflevector <8 x float> %1118, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1124 = fmul <4 x float> %1123, %704, !dbg !35
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1125 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1122, <4 x float> %1124) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !35
  %1126 = extractvalue [2 x <8 x float>] %123, 1, !dbg !35
  %1127 = shufflevector <8 x float> %1126, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1128 = fmul <4 x float> %1127, %712, !dbg !35
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1129 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed456, <4 x half> %1117, <4 x float> %1128) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !35
  %1130 = shufflevector <8 x float> %1126, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1131 = fmul <4 x float> %1130, %712, !dbg !35
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1132 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed456, <4 x half> %1122, <4 x float> %1131) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1133 = shufflevector <8 x half> %1105, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1134 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed442, <4 x half> %1133, <4 x float> %1121) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1135 = shufflevector <8 x half> %1105, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1136 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed442, <4 x half> %1135, <4 x float> %1125) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1137 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed458, <4 x half> %1133, <4 x float> %1129) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1138 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed458, <4 x half> %1135, <4 x float> %1132) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1139 = shufflevector <8 x half> %1107, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1140 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed444, <4 x half> %1139, <4 x float> %1134) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1141 = shufflevector <8 x half> %1107, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1142 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed444, <4 x half> %1141, <4 x float> %1136) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1143 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed460, <4 x half> %1139, <4 x float> %1137) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1144 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed460, <4 x half> %1141, <4 x float> %1138) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1145 = shufflevector <8 x half> %1109, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1146 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed446, <4 x half> %1145, <4 x float> %1140) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1147 = shufflevector <8 x half> %1109, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1148 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed446, <4 x half> %1147, <4 x float> %1142) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1149 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed462, <4 x half> %1145, <4 x float> %1143) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1150 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed462, <4 x half> %1147, <4 x float> %1144) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1151 = shufflevector <8 x half> %1111, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1152 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed448, <4 x half> %1151, <4 x float> %1146) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1153 = shufflevector <8 x half> %1111, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1154 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed448, <4 x half> %1153, <4 x float> %1148) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1155 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed464, <4 x half> %1151, <4 x float> %1149) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1156 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed464, <4 x half> %1153, <4 x float> %1150) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1157 = shufflevector <8 x half> %1113, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1158 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed450, <4 x half> %1157, <4 x float> %1152) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1159 = shufflevector <8 x half> %1113, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1160 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed450, <4 x half> %1159, <4 x float> %1154) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1161 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed466, <4 x half> %1157, <4 x float> %1155) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1162 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed466, <4 x half> %1159, <4 x float> %1156) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1163 = shufflevector <8 x half> %1115, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1164 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed452, <4 x half> %1163, <4 x float> %1158) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1165 = shufflevector <8 x half> %1115, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1166 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed452, <4 x half> %1165, <4 x float> %1160) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1167 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed468, <4 x half> %1163, <4 x float> %1161) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1168 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed468, <4 x half> %1165, <4 x float> %1162) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1169 = bitcast <4 x i32> %1116 to <8 x half>, !dbg !13
  %1170 = shufflevector <8 x half> %1169, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1171 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed454, <4 x half> %1170, <4 x float> %1164) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1172 = shufflevector <8 x half> %1169, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1173 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed454, <4 x half> %1172, <4 x float> %1166) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1174 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed470, <4 x half> %1170, <4 x float> %1167) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1175 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed470, <4 x half> %1172, <4 x float> %1168) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %1176 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #0, !dbg !13, !frisk.v_double_buffer !41
  %1177 = bitcast <4 x i32> %1176 to <8 x half>, !dbg !13
  %1178 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1806, i32 0) #0, !dbg !13, !frisk.v_double_buffer !41
  %1179 = bitcast <4 x i32> %1178 to <8 x half>, !dbg !13
  %1180 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1807, i32 0) #0, !dbg !13, !frisk.v_double_buffer !41
  %1181 = bitcast <4 x i32> %1180 to <8 x half>, !dbg !13
  %1182 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1808, i32 0) #0, !dbg !13, !frisk.v_double_buffer !41
  %1183 = bitcast <4 x i32> %1182 to <8 x half>, !dbg !13
  %1184 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1809, i32 0) #0, !dbg !13, !frisk.v_double_buffer !41
  %1185 = bitcast <4 x i32> %1184 to <8 x half>, !dbg !13
  %1186 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1810, i32 0) #0, !dbg !13, !frisk.v_double_buffer !41
  %1187 = bitcast <4 x i32> %1186 to <8 x half>, !dbg !13
  %1188 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1811, i32 0) #0, !dbg !13, !frisk.v_double_buffer !41
  %1189 = bitcast <4 x i32> %1188 to <8 x half>, !dbg !13
  %1190 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1812, i32 0) #0, !dbg !13, !frisk.v_double_buffer !41
  %1191 = shufflevector <8 x half> %1177, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1192 = extractvalue [2 x <8 x float>] %124, 0, !dbg !35
  %1193 = shufflevector <8 x float> %1192, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1194 = fmul <4 x float> %1193, %704, !dbg !35
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1195 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1191, <4 x float> %1194) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1196 = shufflevector <8 x half> %1177, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1197 = shufflevector <8 x float> %1192, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1198 = fmul <4 x float> %1197, %704, !dbg !35
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1199 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1196, <4 x float> %1198) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !35
  %1200 = extractvalue [2 x <8 x float>] %124, 1, !dbg !35
  %1201 = shufflevector <8 x float> %1200, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1202 = fmul <4 x float> %1201, %712, !dbg !35
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1203 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed456, <4 x half> %1191, <4 x float> %1202) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !35
  %1204 = shufflevector <8 x float> %1200, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1205 = fmul <4 x float> %1204, %712, !dbg !35
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1206 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed456, <4 x half> %1196, <4 x float> %1205) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1207 = shufflevector <8 x half> %1179, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1208 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed442, <4 x half> %1207, <4 x float> %1195) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1209 = shufflevector <8 x half> %1179, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1210 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed442, <4 x half> %1209, <4 x float> %1199) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1211 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed458, <4 x half> %1207, <4 x float> %1203) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1212 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed458, <4 x half> %1209, <4 x float> %1206) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1213 = shufflevector <8 x half> %1181, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1214 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed444, <4 x half> %1213, <4 x float> %1208) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1215 = shufflevector <8 x half> %1181, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1216 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed444, <4 x half> %1215, <4 x float> %1210) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1217 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed460, <4 x half> %1213, <4 x float> %1211) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1218 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed460, <4 x half> %1215, <4 x float> %1212) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1219 = shufflevector <8 x half> %1183, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1220 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed446, <4 x half> %1219, <4 x float> %1214) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1221 = shufflevector <8 x half> %1183, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1222 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed446, <4 x half> %1221, <4 x float> %1216) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1223 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed462, <4 x half> %1219, <4 x float> %1217) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1224 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed462, <4 x half> %1221, <4 x float> %1218) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1225 = shufflevector <8 x half> %1185, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1226 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed448, <4 x half> %1225, <4 x float> %1220) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1227 = shufflevector <8 x half> %1185, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1228 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed448, <4 x half> %1227, <4 x float> %1222) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1229 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed464, <4 x half> %1225, <4 x float> %1223) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1230 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed464, <4 x half> %1227, <4 x float> %1224) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1231 = shufflevector <8 x half> %1187, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1232 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed450, <4 x half> %1231, <4 x float> %1226) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1233 = shufflevector <8 x half> %1187, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1234 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed450, <4 x half> %1233, <4 x float> %1228) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1235 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed466, <4 x half> %1231, <4 x float> %1229) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1236 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed466, <4 x half> %1233, <4 x float> %1230) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1237 = shufflevector <8 x half> %1189, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1238 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed452, <4 x half> %1237, <4 x float> %1232) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1239 = shufflevector <8 x half> %1189, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1240 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed452, <4 x half> %1239, <4 x float> %1234) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1241 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed468, <4 x half> %1237, <4 x float> %1235) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1242 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed468, <4 x half> %1239, <4 x float> %1236) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1243 = bitcast <4 x i32> %1190 to <8 x half>, !dbg !13
  %1244 = shufflevector <8 x half> %1243, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1245 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed454, <4 x half> %1244, <4 x float> %1238) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1246 = shufflevector <8 x half> %1243, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1247 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed454, <4 x half> %1246, <4 x float> %1240) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1248 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed470, <4 x half> %1244, <4 x float> %1241) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1249 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed470, <4 x half> %1246, <4 x float> %1242) #9, !dbg !13, !frisk.pv_interleave !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !7
  %1250 = shufflevector <4 x float> %1001, <4 x float> %1003, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1251 = insertvalue [2 x <8 x float>] poison, <8 x float> %1250, 0, !dbg !7
  %1252 = shufflevector <4 x float> %1022, <4 x float> %1023, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1253 = insertvalue [2 x <8 x float>] %1251, <8 x float> %1252, 1, !dbg !7
  %1254 = insertvalue [2 x <1 x float>] poison, <1 x float> %694, 0, !dbg !7
  %1255 = insertvalue [2 x <1 x float>] %1254, <1 x float> %697, 1, !dbg !7
  %1256 = shufflevector <4 x float> %1095, <4 x float> %1097, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1257 = insertvalue [2 x <8 x float>] poison, <8 x float> %1256, 0, !dbg !7
  %1258 = shufflevector <4 x float> %1098, <4 x float> %1099, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1259 = insertvalue [2 x <8 x float>] %1257, <8 x float> %1258, 1, !dbg !7
  %1260 = shufflevector <4 x float> %1171, <4 x float> %1173, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1261 = insertvalue [2 x <8 x float>] poison, <8 x float> %1260, 0, !dbg !7
  %1262 = shufflevector <4 x float> %1174, <4 x float> %1175, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1263 = insertvalue [2 x <8 x float>] %1261, <8 x float> %1262, 1, !dbg !7
  %1264 = shufflevector <4 x float> %1245, <4 x float> %1247, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1265 = insertvalue [2 x <8 x float>] poison, <8 x float> %1264, 0, !dbg !7
  %1266 = shufflevector <4 x float> %1248, <4 x float> %1249, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1267 = insertvalue [2 x <8 x float>] %1265, <8 x float> %1266, 1, !dbg !7
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %1268 = add nuw nsw i64 %118, 1, !dbg !7
  br label %117, !dbg !7

1269:                                             ; preds = %117
  %lane.sum.field251 = extractvalue [2 x <1 x float>] %120, 1, !dbg !7
  %lane.sum.local252 = extractelement <1 x float> %lane.sum.field251, i64 0, !dbg !7
  %lane.sum.lane.lo253 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !7
  %lane.sum.lane254 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %lane.sum.lane.lo253), !dbg !7
  %1270 = shl nuw nsw i32 %lane.sum.lane254, 2, !dbg !7
  %1271 = xor i32 %1270, 64, !dbg !7
  %1272 = bitcast float %lane.sum.local252 to i32, !dbg !7
  %lane.sum.shuffle255 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1271, i32 %1272), !dbg !7
  %1273 = bitcast i32 %lane.sum.shuffle255 to float, !dbg !7
  %lane.sum.final256 = fadd float %lane.sum.local252, %1273, !dbg !7
  %1274 = xor i32 %1270, 128, !dbg !7
  %1275 = bitcast float %lane.sum.final256 to i32, !dbg !7
  %lane.sum.shuffle257 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1274, i32 %1275), !dbg !7
  %1276 = bitcast i32 %lane.sum.shuffle257 to float, !dbg !7
  %lane.sum.final258 = fadd float %lane.sum.final256, %1276, !dbg !7
  %1277 = insertelement <1 x float> poison, float %lane.sum.final258, i64 0, !dbg !7
  %lane.sum.field = extractvalue [2 x <1 x float>] %120, 0, !dbg !7
  %lane.sum.local = extractelement <1 x float> %lane.sum.field, i64 0, !dbg !7
  %1278 = bitcast float %lane.sum.local to i32, !dbg !7
  %lane.sum.shuffle = call i32 @llvm.amdgcn.ds.bpermute(i32 %1271, i32 %1278), !dbg !7
  %1279 = bitcast i32 %lane.sum.shuffle to float, !dbg !7
  %lane.sum.final = fadd float %lane.sum.local, %1279, !dbg !7
  %1280 = bitcast float %lane.sum.final to i32, !dbg !7
  %lane.sum.shuffle249 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1274, i32 %1280), !dbg !7
  %1281 = bitcast i32 %lane.sum.shuffle249 to float, !dbg !7
  %lane.sum.final250 = fadd float %lane.sum.final, %1281, !dbg !7
  %1282 = insertelement <1 x float> poison, float %lane.sum.final250, i64 0, !dbg !7
  %1283 = and i32 %kdma.tid, 16, !dbg !7
  %.not = icmp eq i32 %1283, 0, !dbg !7
  %1284 = and i32 %kdma.tid, 32, !dbg !7
  %.not36 = icmp eq i32 %1284, 0, !dbg !7
  %1285 = extractvalue [2 x <8 x float>] %124, 0, !dbg !7
  %1286 = extractelement <8 x float> %1285, i64 0, !dbg !7
  %1287 = extractelement <8 x float> %1285, i64 1, !dbg !7
  %1288 = extractelement <8 x float> %1285, i64 2, !dbg !7
  %1289 = extractelement <8 x float> %1285, i64 3, !dbg !7
  %1290 = select i1 %.not, float %1287, float %1286, !dbg !7
  %1291 = add nuw nsw i32 %lane.sum.lane254, 64, !dbg !7
  %1292 = xor i32 %lane.sum.lane254, 16, !dbg !7
  %1293 = and i32 %1291, 192, !dbg !7
  %1294 = icmp samesign ult i32 %1292, %1293, !dbg !7
  %1295 = select i1 %1294, i32 %1292, i32 %lane.sum.lane254, !dbg !7
  %1296 = shl nuw nsw i32 %1295, 2, !dbg !7
  %1297 = bitcast float %1290 to i32, !dbg !7
  %1298 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1296, i32 %1297), !dbg !7
  %1299 = bitcast i32 %1298 to float, !dbg !7
  %1300 = select i1 %.not, float %1286, float %1299, !dbg !7
  %1301 = select i1 %.not, float %1299, float %1287, !dbg !7
  %1302 = select i1 %.not, float %1289, float %1288, !dbg !7
  %1303 = bitcast float %1302 to i32, !dbg !7
  %1304 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1296, i32 %1303), !dbg !7
  %1305 = bitcast i32 %1304 to float, !dbg !7
  %1306 = select i1 %.not, float %1288, float %1305, !dbg !7
  %1307 = select i1 %.not, float %1305, float %1289, !dbg !7
  %1308 = select i1 %.not36, float %1306, float %1300, !dbg !7
  %1309 = xor i32 %lane.sum.lane254, 32, !dbg !7
  %1310 = icmp samesign ult i32 %1309, %1293, !dbg !7
  %1311 = select i1 %1310, i32 %1309, i32 %lane.sum.lane254, !dbg !7
  %1312 = shl nuw nsw i32 %1311, 2, !dbg !7
  %1313 = bitcast float %1308 to i32, !dbg !7
  %1314 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1312, i32 %1313), !dbg !7
  %1315 = bitcast i32 %1314 to float, !dbg !7
  %1316 = select i1 %.not36, float %1300, float %1315, !dbg !7
  %1317 = select i1 %.not36, float %1315, float %1306, !dbg !7
  %1318 = select i1 %.not36, float %1307, float %1301, !dbg !7
  %1319 = bitcast float %1318 to i32, !dbg !7
  %1320 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1312, i32 %1319), !dbg !7
  %1321 = bitcast i32 %1320 to float, !dbg !7
  %1322 = select i1 %.not36, float %1301, float %1321, !dbg !7
  %1323 = select i1 %.not36, float %1321, float %1307, !dbg !7
  %1324 = extractelement <8 x float> %1285, i64 4, !dbg !7
  %1325 = extractelement <8 x float> %1285, i64 5, !dbg !7
  %1326 = extractelement <8 x float> %1285, i64 6, !dbg !7
  %1327 = extractelement <8 x float> %1285, i64 7, !dbg !7
  %1328 = select i1 %.not, float %1325, float %1324, !dbg !7
  %1329 = bitcast float %1328 to i32, !dbg !7
  %1330 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1296, i32 %1329), !dbg !7
  %1331 = bitcast i32 %1330 to float, !dbg !7
  %1332 = select i1 %.not, float %1324, float %1331, !dbg !7
  %1333 = select i1 %.not, float %1331, float %1325, !dbg !7
  %1334 = select i1 %.not, float %1327, float %1326, !dbg !7
  %1335 = bitcast float %1334 to i32, !dbg !7
  %1336 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1296, i32 %1335), !dbg !7
  %1337 = bitcast i32 %1336 to float, !dbg !7
  %1338 = select i1 %.not, float %1326, float %1337, !dbg !7
  %1339 = select i1 %.not, float %1337, float %1327, !dbg !7
  %1340 = select i1 %.not36, float %1338, float %1332, !dbg !7
  %1341 = bitcast float %1340 to i32, !dbg !7
  %1342 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1312, i32 %1341), !dbg !7
  %1343 = bitcast i32 %1342 to float, !dbg !7
  %1344 = select i1 %.not36, float %1332, float %1343, !dbg !7
  %1345 = select i1 %.not36, float %1343, float %1338, !dbg !7
  %1346 = select i1 %.not36, float %1339, float %1333, !dbg !7
  %1347 = bitcast float %1346 to i32, !dbg !7
  %1348 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1312, i32 %1347), !dbg !7
  %1349 = bitcast i32 %1348 to float, !dbg !7
  %1350 = select i1 %.not36, float %1333, float %1349, !dbg !7
  %1351 = select i1 %.not36, float %1349, float %1339, !dbg !7
  %1352 = extractvalue [2 x <8 x float>] %124, 1, !dbg !7
  %1353 = extractelement <8 x float> %1352, i64 0, !dbg !7
  %1354 = extractelement <8 x float> %1352, i64 1, !dbg !7
  %1355 = extractelement <8 x float> %1352, i64 2, !dbg !7
  %1356 = extractelement <8 x float> %1352, i64 3, !dbg !7
  %1357 = select i1 %.not, float %1354, float %1353, !dbg !7
  %1358 = bitcast float %1357 to i32, !dbg !7
  %1359 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1296, i32 %1358), !dbg !7
  %1360 = bitcast i32 %1359 to float, !dbg !7
  %1361 = select i1 %.not, float %1353, float %1360, !dbg !7
  %1362 = select i1 %.not, float %1360, float %1354, !dbg !7
  %1363 = select i1 %.not, float %1356, float %1355, !dbg !7
  %1364 = bitcast float %1363 to i32, !dbg !7
  %1365 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1296, i32 %1364), !dbg !7
  %1366 = bitcast i32 %1365 to float, !dbg !7
  %1367 = select i1 %.not, float %1355, float %1366, !dbg !7
  %1368 = select i1 %.not, float %1366, float %1356, !dbg !7
  %1369 = select i1 %.not36, float %1367, float %1361, !dbg !7
  %1370 = bitcast float %1369 to i32, !dbg !7
  %1371 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1312, i32 %1370), !dbg !7
  %1372 = bitcast i32 %1371 to float, !dbg !7
  %1373 = select i1 %.not36, float %1361, float %1372, !dbg !7
  %1374 = select i1 %.not36, float %1372, float %1367, !dbg !7
  %1375 = select i1 %.not36, float %1368, float %1362, !dbg !7
  %1376 = bitcast float %1375 to i32, !dbg !7
  %1377 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1312, i32 %1376), !dbg !7
  %1378 = bitcast i32 %1377 to float, !dbg !7
  %1379 = select i1 %.not36, float %1362, float %1378, !dbg !7
  %1380 = select i1 %.not36, float %1378, float %1368, !dbg !7
  %1381 = extractelement <8 x float> %1352, i64 4, !dbg !7
  %1382 = extractelement <8 x float> %1352, i64 5, !dbg !7
  %1383 = extractelement <8 x float> %1352, i64 6, !dbg !7
  %1384 = extractelement <8 x float> %1352, i64 7, !dbg !7
  %1385 = select i1 %.not, float %1382, float %1381, !dbg !7
  %1386 = bitcast float %1385 to i32, !dbg !7
  %1387 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1296, i32 %1386), !dbg !7
  %1388 = bitcast i32 %1387 to float, !dbg !7
  %1389 = select i1 %.not, float %1381, float %1388, !dbg !7
  %1390 = select i1 %.not, float %1388, float %1382, !dbg !7
  %1391 = select i1 %.not, float %1384, float %1383, !dbg !7
  %1392 = bitcast float %1391 to i32, !dbg !7
  %1393 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1296, i32 %1392), !dbg !7
  %1394 = bitcast i32 %1393 to float, !dbg !7
  %1395 = select i1 %.not, float %1383, float %1394, !dbg !7
  %1396 = select i1 %.not, float %1394, float %1384, !dbg !7
  %1397 = select i1 %.not36, float %1395, float %1389, !dbg !7
  %1398 = bitcast float %1397 to i32, !dbg !7
  %1399 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1312, i32 %1398), !dbg !7
  %1400 = bitcast i32 %1399 to float, !dbg !7
  %1401 = select i1 %.not36, float %1389, float %1400, !dbg !7
  %1402 = select i1 %.not36, float %1400, float %1395, !dbg !7
  %1403 = select i1 %.not36, float %1396, float %1390, !dbg !7
  %1404 = bitcast float %1403 to i32, !dbg !7
  %1405 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1312, i32 %1404), !dbg !7
  %1406 = bitcast i32 %1405 to float, !dbg !7
  %1407 = select i1 %.not36, float %1390, float %1406, !dbg !7
  %1408 = select i1 %.not36, float %1406, float %1396, !dbg !7
  %1409 = extractvalue [2 x <8 x float>] %123, 0, !dbg !7
  %1410 = extractelement <8 x float> %1409, i64 0, !dbg !7
  %1411 = extractelement <8 x float> %1409, i64 1, !dbg !7
  %1412 = extractelement <8 x float> %1409, i64 2, !dbg !7
  %1413 = extractelement <8 x float> %1409, i64 3, !dbg !7
  %1414 = select i1 %.not, float %1411, float %1410, !dbg !7
  %1415 = bitcast float %1414 to i32, !dbg !7
  %1416 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1296, i32 %1415), !dbg !7
  %1417 = bitcast i32 %1416 to float, !dbg !7
  %1418 = select i1 %.not, float %1410, float %1417, !dbg !7
  %1419 = select i1 %.not, float %1417, float %1411, !dbg !7
  %1420 = select i1 %.not, float %1413, float %1412, !dbg !7
  %1421 = bitcast float %1420 to i32, !dbg !7
  %1422 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1296, i32 %1421), !dbg !7
  %1423 = bitcast i32 %1422 to float, !dbg !7
  %1424 = select i1 %.not, float %1412, float %1423, !dbg !7
  %1425 = select i1 %.not, float %1423, float %1413, !dbg !7
  %1426 = select i1 %.not36, float %1424, float %1418, !dbg !7
  %1427 = bitcast float %1426 to i32, !dbg !7
  %1428 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1312, i32 %1427), !dbg !7
  %1429 = bitcast i32 %1428 to float, !dbg !7
  %1430 = select i1 %.not36, float %1418, float %1429, !dbg !7
  %1431 = select i1 %.not36, float %1429, float %1424, !dbg !7
  %1432 = select i1 %.not36, float %1425, float %1419, !dbg !7
  %1433 = bitcast float %1432 to i32, !dbg !7
  %1434 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1312, i32 %1433), !dbg !7
  %1435 = bitcast i32 %1434 to float, !dbg !7
  %1436 = select i1 %.not36, float %1419, float %1435, !dbg !7
  %1437 = select i1 %.not36, float %1435, float %1425, !dbg !7
  %1438 = extractelement <8 x float> %1409, i64 4, !dbg !7
  %1439 = extractelement <8 x float> %1409, i64 5, !dbg !7
  %1440 = extractelement <8 x float> %1409, i64 6, !dbg !7
  %1441 = extractelement <8 x float> %1409, i64 7, !dbg !7
  %1442 = select i1 %.not, float %1439, float %1438, !dbg !7
  %1443 = bitcast float %1442 to i32, !dbg !7
  %1444 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1296, i32 %1443), !dbg !7
  %1445 = bitcast i32 %1444 to float, !dbg !7
  %1446 = select i1 %.not, float %1438, float %1445, !dbg !7
  %1447 = select i1 %.not, float %1445, float %1439, !dbg !7
  %1448 = select i1 %.not, float %1441, float %1440, !dbg !7
  %1449 = bitcast float %1448 to i32, !dbg !7
  %1450 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1296, i32 %1449), !dbg !7
  %1451 = bitcast i32 %1450 to float, !dbg !7
  %1452 = select i1 %.not, float %1440, float %1451, !dbg !7
  %1453 = select i1 %.not, float %1451, float %1441, !dbg !7
  %1454 = select i1 %.not36, float %1452, float %1446, !dbg !7
  %1455 = bitcast float %1454 to i32, !dbg !7
  %1456 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1312, i32 %1455), !dbg !7
  %1457 = bitcast i32 %1456 to float, !dbg !7
  %1458 = select i1 %.not36, float %1446, float %1457, !dbg !7
  %1459 = select i1 %.not36, float %1457, float %1452, !dbg !7
  %1460 = select i1 %.not36, float %1453, float %1447, !dbg !7
  %1461 = bitcast float %1460 to i32, !dbg !7
  %1462 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1312, i32 %1461), !dbg !7
  %1463 = bitcast i32 %1462 to float, !dbg !7
  %1464 = select i1 %.not36, float %1447, float %1463, !dbg !7
  %1465 = select i1 %.not36, float %1463, float %1453, !dbg !7
  %1466 = extractvalue [2 x <8 x float>] %123, 1, !dbg !7
  %1467 = extractelement <8 x float> %1466, i64 0, !dbg !7
  %1468 = extractelement <8 x float> %1466, i64 1, !dbg !7
  %1469 = extractelement <8 x float> %1466, i64 2, !dbg !7
  %1470 = extractelement <8 x float> %1466, i64 3, !dbg !7
  %1471 = select i1 %.not, float %1468, float %1467, !dbg !7
  %1472 = bitcast float %1471 to i32, !dbg !7
  %1473 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1296, i32 %1472), !dbg !7
  %1474 = bitcast i32 %1473 to float, !dbg !7
  %1475 = select i1 %.not, float %1467, float %1474, !dbg !7
  %1476 = select i1 %.not, float %1474, float %1468, !dbg !7
  %1477 = select i1 %.not, float %1470, float %1469, !dbg !7
  %1478 = bitcast float %1477 to i32, !dbg !7
  %1479 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1296, i32 %1478), !dbg !7
  %1480 = bitcast i32 %1479 to float, !dbg !7
  %1481 = select i1 %.not, float %1469, float %1480, !dbg !7
  %1482 = select i1 %.not, float %1480, float %1470, !dbg !7
  %1483 = select i1 %.not36, float %1481, float %1475, !dbg !7
  %1484 = bitcast float %1483 to i32, !dbg !7
  %1485 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1312, i32 %1484), !dbg !7
  %1486 = bitcast i32 %1485 to float, !dbg !7
  %1487 = select i1 %.not36, float %1475, float %1486, !dbg !7
  %1488 = select i1 %.not36, float %1486, float %1481, !dbg !7
  %1489 = select i1 %.not36, float %1482, float %1476, !dbg !7
  %1490 = bitcast float %1489 to i32, !dbg !7
  %1491 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1312, i32 %1490), !dbg !7
  %1492 = bitcast i32 %1491 to float, !dbg !7
  %1493 = select i1 %.not36, float %1476, float %1492, !dbg !7
  %1494 = select i1 %.not36, float %1492, float %1482, !dbg !7
  %1495 = extractelement <8 x float> %1466, i64 4, !dbg !7
  %1496 = extractelement <8 x float> %1466, i64 5, !dbg !7
  %1497 = extractelement <8 x float> %1466, i64 6, !dbg !7
  %1498 = extractelement <8 x float> %1466, i64 7, !dbg !7
  %1499 = select i1 %.not, float %1496, float %1495, !dbg !7
  %1500 = bitcast float %1499 to i32, !dbg !7
  %1501 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1296, i32 %1500), !dbg !7
  %1502 = bitcast i32 %1501 to float, !dbg !7
  %1503 = select i1 %.not, float %1495, float %1502, !dbg !7
  %1504 = select i1 %.not, float %1502, float %1496, !dbg !7
  %1505 = select i1 %.not, float %1498, float %1497, !dbg !7
  %1506 = bitcast float %1505 to i32, !dbg !7
  %1507 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1296, i32 %1506), !dbg !7
  %1508 = bitcast i32 %1507 to float, !dbg !7
  %1509 = select i1 %.not, float %1497, float %1508, !dbg !7
  %1510 = select i1 %.not, float %1508, float %1498, !dbg !7
  %1511 = select i1 %.not36, float %1509, float %1503, !dbg !7
  %1512 = bitcast float %1511 to i32, !dbg !7
  %1513 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1312, i32 %1512), !dbg !7
  %1514 = bitcast i32 %1513 to float, !dbg !7
  %1515 = select i1 %.not36, float %1503, float %1514, !dbg !7
  %1516 = select i1 %.not36, float %1514, float %1509, !dbg !7
  %1517 = select i1 %.not36, float %1510, float %1504, !dbg !7
  %1518 = bitcast float %1517 to i32, !dbg !7
  %1519 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1312, i32 %1518), !dbg !7
  %1520 = bitcast i32 %1519 to float, !dbg !7
  %1521 = select i1 %.not36, float %1504, float %1520, !dbg !7
  %1522 = select i1 %.not36, float %1520, float %1510, !dbg !7
  %1523 = extractvalue [2 x <8 x float>] %122, 0, !dbg !7
  %1524 = extractelement <8 x float> %1523, i64 0, !dbg !7
  %1525 = extractelement <8 x float> %1523, i64 1, !dbg !7
  %1526 = extractelement <8 x float> %1523, i64 2, !dbg !7
  %1527 = extractelement <8 x float> %1523, i64 3, !dbg !7
  %1528 = select i1 %.not, float %1525, float %1524, !dbg !7
  %1529 = bitcast float %1528 to i32, !dbg !7
  %1530 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1296, i32 %1529), !dbg !7
  %1531 = bitcast i32 %1530 to float, !dbg !7
  %1532 = select i1 %.not, float %1524, float %1531, !dbg !7
  %1533 = select i1 %.not, float %1531, float %1525, !dbg !7
  %1534 = select i1 %.not, float %1527, float %1526, !dbg !7
  %1535 = bitcast float %1534 to i32, !dbg !7
  %1536 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1296, i32 %1535), !dbg !7
  %1537 = bitcast i32 %1536 to float, !dbg !7
  %1538 = select i1 %.not, float %1526, float %1537, !dbg !7
  %1539 = select i1 %.not, float %1537, float %1527, !dbg !7
  %1540 = select i1 %.not36, float %1538, float %1532, !dbg !7
  %1541 = bitcast float %1540 to i32, !dbg !7
  %1542 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1312, i32 %1541), !dbg !7
  %1543 = bitcast i32 %1542 to float, !dbg !7
  %1544 = select i1 %.not36, float %1532, float %1543, !dbg !7
  %1545 = select i1 %.not36, float %1543, float %1538, !dbg !7
  %1546 = select i1 %.not36, float %1539, float %1533, !dbg !7
  %1547 = bitcast float %1546 to i32, !dbg !7
  %1548 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1312, i32 %1547), !dbg !7
  %1549 = bitcast i32 %1548 to float, !dbg !7
  %1550 = select i1 %.not36, float %1533, float %1549, !dbg !7
  %1551 = select i1 %.not36, float %1549, float %1539, !dbg !7
  %1552 = extractelement <8 x float> %1523, i64 4, !dbg !7
  %1553 = extractelement <8 x float> %1523, i64 5, !dbg !7
  %1554 = extractelement <8 x float> %1523, i64 6, !dbg !7
  %1555 = extractelement <8 x float> %1523, i64 7, !dbg !7
  %1556 = select i1 %.not, float %1553, float %1552, !dbg !7
  %1557 = bitcast float %1556 to i32, !dbg !7
  %1558 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1296, i32 %1557), !dbg !7
  %1559 = bitcast i32 %1558 to float, !dbg !7
  %1560 = select i1 %.not, float %1552, float %1559, !dbg !7
  %1561 = select i1 %.not, float %1559, float %1553, !dbg !7
  %1562 = select i1 %.not, float %1555, float %1554, !dbg !7
  %1563 = bitcast float %1562 to i32, !dbg !7
  %1564 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1296, i32 %1563), !dbg !7
  %1565 = bitcast i32 %1564 to float, !dbg !7
  %1566 = select i1 %.not, float %1554, float %1565, !dbg !7
  %1567 = select i1 %.not, float %1565, float %1555, !dbg !7
  %1568 = select i1 %.not36, float %1566, float %1560, !dbg !7
  %1569 = bitcast float %1568 to i32, !dbg !7
  %1570 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1312, i32 %1569), !dbg !7
  %1571 = bitcast i32 %1570 to float, !dbg !7
  %1572 = select i1 %.not36, float %1560, float %1571, !dbg !7
  %1573 = select i1 %.not36, float %1571, float %1566, !dbg !7
  %1574 = select i1 %.not36, float %1567, float %1561, !dbg !7
  %1575 = bitcast float %1574 to i32, !dbg !7
  %1576 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1312, i32 %1575), !dbg !7
  %1577 = bitcast i32 %1576 to float, !dbg !7
  %1578 = select i1 %.not36, float %1561, float %1577, !dbg !7
  %1579 = select i1 %.not36, float %1577, float %1567, !dbg !7
  %1580 = extractvalue [2 x <8 x float>] %122, 1, !dbg !7
  %1581 = extractelement <8 x float> %1580, i64 0, !dbg !7
  %1582 = extractelement <8 x float> %1580, i64 1, !dbg !7
  %1583 = extractelement <8 x float> %1580, i64 2, !dbg !7
  %1584 = extractelement <8 x float> %1580, i64 3, !dbg !7
  %1585 = select i1 %.not, float %1582, float %1581, !dbg !7
  %1586 = bitcast float %1585 to i32, !dbg !7
  %1587 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1296, i32 %1586), !dbg !7
  %1588 = bitcast i32 %1587 to float, !dbg !7
  %1589 = select i1 %.not, float %1581, float %1588, !dbg !7
  %1590 = select i1 %.not, float %1588, float %1582, !dbg !7
  %1591 = select i1 %.not, float %1584, float %1583, !dbg !7
  %1592 = bitcast float %1591 to i32, !dbg !7
  %1593 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1296, i32 %1592), !dbg !7
  %1594 = bitcast i32 %1593 to float, !dbg !7
  %1595 = select i1 %.not, float %1583, float %1594, !dbg !7
  %1596 = select i1 %.not, float %1594, float %1584, !dbg !7
  %1597 = select i1 %.not36, float %1595, float %1589, !dbg !7
  %1598 = bitcast float %1597 to i32, !dbg !7
  %1599 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1312, i32 %1598), !dbg !7
  %1600 = bitcast i32 %1599 to float, !dbg !7
  %1601 = select i1 %.not36, float %1589, float %1600, !dbg !7
  %1602 = select i1 %.not36, float %1600, float %1595, !dbg !7
  %1603 = select i1 %.not36, float %1596, float %1590, !dbg !7
  %1604 = bitcast float %1603 to i32, !dbg !7
  %1605 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1312, i32 %1604), !dbg !7
  %1606 = bitcast i32 %1605 to float, !dbg !7
  %1607 = select i1 %.not36, float %1590, float %1606, !dbg !7
  %1608 = select i1 %.not36, float %1606, float %1596, !dbg !7
  %1609 = extractelement <8 x float> %1580, i64 4, !dbg !7
  %1610 = extractelement <8 x float> %1580, i64 5, !dbg !7
  %1611 = extractelement <8 x float> %1580, i64 6, !dbg !7
  %1612 = extractelement <8 x float> %1580, i64 7, !dbg !7
  %1613 = select i1 %.not, float %1610, float %1609, !dbg !7
  %1614 = bitcast float %1613 to i32, !dbg !7
  %1615 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1296, i32 %1614), !dbg !7
  %1616 = bitcast i32 %1615 to float, !dbg !7
  %1617 = select i1 %.not, float %1609, float %1616, !dbg !7
  %1618 = select i1 %.not, float %1616, float %1610, !dbg !7
  %1619 = select i1 %.not, float %1612, float %1611, !dbg !7
  %1620 = bitcast float %1619 to i32, !dbg !7
  %1621 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1296, i32 %1620), !dbg !7
  %1622 = bitcast i32 %1621 to float, !dbg !7
  %1623 = select i1 %.not, float %1611, float %1622, !dbg !7
  %1624 = select i1 %.not, float %1622, float %1612, !dbg !7
  %1625 = select i1 %.not36, float %1623, float %1617, !dbg !7
  %1626 = bitcast float %1625 to i32, !dbg !7
  %1627 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1312, i32 %1626), !dbg !7
  %1628 = bitcast i32 %1627 to float, !dbg !7
  %1629 = select i1 %.not36, float %1617, float %1628, !dbg !7
  %1630 = select i1 %.not36, float %1628, float %1623, !dbg !7
  %1631 = select i1 %.not36, float %1624, float %1618, !dbg !7
  %1632 = bitcast float %1631 to i32, !dbg !7
  %1633 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1312, i32 %1632), !dbg !7
  %1634 = bitcast i32 %1633 to float, !dbg !7
  %1635 = select i1 %.not36, float %1618, float %1634, !dbg !7
  %1636 = select i1 %.not36, float %1634, float %1624, !dbg !7
  %1637 = extractvalue [2 x <8 x float>] %119, 0, !dbg !7
  %1638 = extractelement <8 x float> %1637, i64 0, !dbg !7
  %1639 = extractelement <8 x float> %1637, i64 1, !dbg !7
  %1640 = extractelement <8 x float> %1637, i64 2, !dbg !7
  %1641 = extractelement <8 x float> %1637, i64 3, !dbg !7
  %1642 = select i1 %.not, float %1639, float %1638, !dbg !7
  %1643 = bitcast float %1642 to i32, !dbg !7
  %1644 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1296, i32 %1643), !dbg !7
  %1645 = bitcast i32 %1644 to float, !dbg !7
  %1646 = select i1 %.not, float %1638, float %1645, !dbg !7
  %1647 = select i1 %.not, float %1645, float %1639, !dbg !7
  %1648 = select i1 %.not, float %1641, float %1640, !dbg !7
  %1649 = bitcast float %1648 to i32, !dbg !7
  %1650 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1296, i32 %1649), !dbg !7
  %1651 = bitcast i32 %1650 to float, !dbg !7
  %1652 = select i1 %.not, float %1640, float %1651, !dbg !7
  %1653 = select i1 %.not, float %1651, float %1641, !dbg !7
  %1654 = select i1 %.not36, float %1652, float %1646, !dbg !7
  %1655 = bitcast float %1654 to i32, !dbg !7
  %1656 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1312, i32 %1655), !dbg !7
  %1657 = select i1 %.not36, float %1653, float %1647, !dbg !7
  %1658 = bitcast float %1657 to i32, !dbg !7
  %1659 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1312, i32 %1658), !dbg !7
  %1660 = extractelement <8 x float> %1637, i64 4, !dbg !7
  %1661 = extractelement <8 x float> %1637, i64 5, !dbg !7
  %1662 = extractelement <8 x float> %1637, i64 6, !dbg !7
  %1663 = extractelement <8 x float> %1637, i64 7, !dbg !7
  %1664 = select i1 %.not, float %1661, float %1660, !dbg !7
  %1665 = bitcast float %1664 to i32, !dbg !7
  %1666 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1296, i32 %1665), !dbg !7
  %1667 = bitcast i32 %1666 to float, !dbg !7
  %1668 = select i1 %.not, float %1660, float %1667, !dbg !7
  %1669 = select i1 %.not, float %1667, float %1661, !dbg !7
  %1670 = select i1 %.not, float %1663, float %1662, !dbg !7
  %1671 = bitcast float %1670 to i32, !dbg !7
  %1672 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1296, i32 %1671), !dbg !7
  %1673 = bitcast i32 %1672 to float, !dbg !7
  %1674 = select i1 %.not, float %1662, float %1673, !dbg !7
  %1675 = select i1 %.not, float %1673, float %1663, !dbg !7
  %1676 = select i1 %.not36, float %1674, float %1668, !dbg !7
  %1677 = bitcast float %1676 to i32, !dbg !7
  %1678 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1312, i32 %1677), !dbg !7
  %1679 = select i1 %.not36, float %1675, float %1669, !dbg !7
  %1680 = bitcast float %1679 to i32, !dbg !7
  %1681 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1312, i32 %1680), !dbg !7
  %1682 = extractvalue [2 x <8 x float>] %119, 1, !dbg !7
  %1683 = extractelement <8 x float> %1682, i64 0, !dbg !7
  %1684 = extractelement <8 x float> %1682, i64 1, !dbg !7
  %1685 = extractelement <8 x float> %1682, i64 2, !dbg !7
  %1686 = extractelement <8 x float> %1682, i64 3, !dbg !7
  %1687 = select i1 %.not, float %1684, float %1683, !dbg !7
  %1688 = bitcast float %1687 to i32, !dbg !7
  %1689 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1296, i32 %1688), !dbg !7
  %1690 = bitcast i32 %1689 to float, !dbg !7
  %1691 = select i1 %.not, float %1683, float %1690, !dbg !7
  %1692 = select i1 %.not, float %1690, float %1684, !dbg !7
  %1693 = select i1 %.not, float %1686, float %1685, !dbg !7
  %1694 = bitcast float %1693 to i32, !dbg !7
  %1695 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1296, i32 %1694), !dbg !7
  %1696 = bitcast i32 %1695 to float, !dbg !7
  %1697 = select i1 %.not, float %1685, float %1696, !dbg !7
  %1698 = select i1 %.not, float %1696, float %1686, !dbg !7
  %1699 = select i1 %.not36, float %1697, float %1691, !dbg !7
  %1700 = bitcast float %1699 to i32, !dbg !7
  %1701 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1312, i32 %1700), !dbg !7
  %1702 = select i1 %.not36, float %1698, float %1692, !dbg !7
  %1703 = bitcast float %1702 to i32, !dbg !7
  %1704 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1312, i32 %1703), !dbg !7
  %1705 = extractelement <8 x float> %1682, i64 4, !dbg !7
  %1706 = extractelement <8 x float> %1682, i64 5, !dbg !7
  %1707 = extractelement <8 x float> %1682, i64 6, !dbg !7
  %1708 = extractelement <8 x float> %1682, i64 7, !dbg !7
  %1709 = select i1 %.not, float %1706, float %1705, !dbg !7
  %1710 = bitcast float %1709 to i32, !dbg !7
  %1711 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1296, i32 %1710), !dbg !7
  %1712 = bitcast i32 %1711 to float, !dbg !7
  %1713 = select i1 %.not, float %1705, float %1712, !dbg !7
  %1714 = select i1 %.not, float %1712, float %1706, !dbg !7
  %1715 = select i1 %.not, float %1708, float %1707, !dbg !7
  %1716 = bitcast float %1715 to i32, !dbg !7
  %1717 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1296, i32 %1716), !dbg !7
  %1718 = bitcast i32 %1717 to float, !dbg !7
  %1719 = select i1 %.not, float %1707, float %1718, !dbg !7
  %1720 = select i1 %.not, float %1718, float %1708, !dbg !7
  %1721 = select i1 %.not36, float %1719, float %1713, !dbg !7
  %1722 = bitcast float %1721 to i32, !dbg !7
  %1723 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1312, i32 %1722), !dbg !7
  %1724 = select i1 %.not36, float %1720, float %1714, !dbg !7
  %1725 = bitcast float %1724 to i32, !dbg !7
  %1726 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1312, i32 %1725), !dbg !7
  %1727 = shufflevector <1 x float> %1282, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !20
  %1728 = shufflevector <1 x float> %1277, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !20
  %1729 = bitcast i32 %1723 to float, !dbg !7
  %1730 = select i1 %.not36, float %1713, float %1729, !dbg !7
  %1731 = insertelement <4 x float> poison, float %1730, i64 0, !dbg !20
  %1732 = bitcast i32 %1726 to float, !dbg !7
  %1733 = select i1 %.not36, float %1714, float %1732, !dbg !7
  %1734 = insertelement <4 x float> %1731, float %1733, i64 1, !dbg !20
  %1735 = select i1 %.not36, float %1729, float %1719, !dbg !7
  %1736 = insertelement <4 x float> %1734, float %1735, i64 2, !dbg !20
  %1737 = select i1 %.not36, float %1732, float %1720, !dbg !7
  %1738 = insertelement <4 x float> %1736, float %1737, i64 3, !dbg !20
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1728, !dbg !20
  %normalized = fmul <4 x float> %1738, %row.reciprocal, !dbg !20
  %1739 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !21
  %1740 = extractelement <4 x half> %1739, i64 3, !dbg !21
  %1741 = extractelement <4 x half> %1739, i64 2, !dbg !21
  %1742 = extractelement <4 x half> %1739, i64 1, !dbg !21
  %1743 = extractelement <4 x half> %1739, i64 0, !dbg !21
  %1744 = bitcast i32 %1701 to float, !dbg !7
  %1745 = select i1 %.not36, float %1691, float %1744, !dbg !7
  %1746 = insertelement <4 x float> poison, float %1745, i64 0, !dbg !20
  %1747 = bitcast i32 %1704 to float, !dbg !7
  %1748 = select i1 %.not36, float %1692, float %1747, !dbg !7
  %1749 = insertelement <4 x float> %1746, float %1748, i64 1, !dbg !20
  %1750 = select i1 %.not36, float %1744, float %1697, !dbg !7
  %1751 = insertelement <4 x float> %1749, float %1750, i64 2, !dbg !20
  %1752 = select i1 %.not36, float %1747, float %1698, !dbg !7
  %1753 = insertelement <4 x float> %1751, float %1752, i64 3, !dbg !20
  %normalized104 = fmul <4 x float> %1753, %row.reciprocal, !dbg !20
  %1754 = fptrunc <4 x float> %normalized104 to <4 x half>, !dbg !21
  %1755 = extractelement <4 x half> %1754, i64 3, !dbg !21
  %1756 = extractelement <4 x half> %1754, i64 2, !dbg !21
  %1757 = extractelement <4 x half> %1754, i64 1, !dbg !21
  %1758 = extractelement <4 x half> %1754, i64 0, !dbg !21
  %1759 = bitcast i32 %1678 to float, !dbg !7
  %1760 = select i1 %.not36, float %1668, float %1759, !dbg !7
  %1761 = insertelement <4 x float> poison, float %1760, i64 0, !dbg !20
  %1762 = bitcast i32 %1681 to float, !dbg !7
  %1763 = select i1 %.not36, float %1669, float %1762, !dbg !7
  %1764 = insertelement <4 x float> %1761, float %1763, i64 1, !dbg !20
  %1765 = select i1 %.not36, float %1759, float %1674, !dbg !7
  %1766 = insertelement <4 x float> %1764, float %1765, i64 2, !dbg !20
  %1767 = select i1 %.not36, float %1762, float %1675, !dbg !7
  %1768 = insertelement <4 x float> %1766, float %1767, i64 3, !dbg !20
  %row.reciprocal105 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1727, !dbg !20
  %normalized106 = fmul <4 x float> %1768, %row.reciprocal105, !dbg !20
  %1769 = fptrunc <4 x float> %normalized106 to <4 x half>, !dbg !21
  %1770 = extractelement <4 x half> %1769, i64 3, !dbg !21
  %1771 = extractelement <4 x half> %1769, i64 2, !dbg !21
  %1772 = extractelement <4 x half> %1769, i64 1, !dbg !21
  %1773 = extractelement <4 x half> %1769, i64 0, !dbg !21
  %1774 = bitcast i32 %1656 to float, !dbg !7
  %1775 = select i1 %.not36, float %1646, float %1774, !dbg !7
  %1776 = insertelement <4 x float> poison, float %1775, i64 0, !dbg !20
  %1777 = bitcast i32 %1659 to float, !dbg !7
  %1778 = select i1 %.not36, float %1647, float %1777, !dbg !7
  %1779 = insertelement <4 x float> %1776, float %1778, i64 1, !dbg !20
  %1780 = select i1 %.not36, float %1774, float %1652, !dbg !7
  %1781 = insertelement <4 x float> %1779, float %1780, i64 2, !dbg !20
  %1782 = select i1 %.not36, float %1777, float %1653, !dbg !7
  %1783 = insertelement <4 x float> %1781, float %1782, i64 3, !dbg !20
  %normalized107 = fmul <4 x float> %1783, %row.reciprocal105, !dbg !20
  %1784 = fptrunc <4 x float> %normalized107 to <4 x half>, !dbg !21
  %1785 = extractelement <4 x half> %1784, i64 3, !dbg !21
  %1786 = extractelement <4 x half> %1784, i64 2, !dbg !21
  %1787 = extractelement <4 x half> %1784, i64 1, !dbg !21
  %1788 = extractelement <4 x half> %1784, i64 0, !dbg !21
  %address.thread1041 = call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #9, !dbg !22, !range !39, !frisk.register_lifetimes !1
  %address.local1042 = zext nneg i32 %address.thread1041 to i64, !dbg !8
  %address.local1043 = add i64 %30, %address.local1042, !dbg !22
  %address.local1044 = lshr i64 %address.local1042, 2, !dbg !18
  %address.local1045 = and i64 %address.local1044, 48, !dbg !18
  %address.local1046 = add i64 %address.local1043, %address.local1045, !dbg !22
  %address.local1047 = and i64 %address.local1042, 240, !dbg !22
  %address.local1049 = sub i64 %address.local1046, %address.local1047, !dbg !22
  %address.local1050 = add i64 %address.local1049, %31, !dbg !22
  %address.local1051 = shl nuw nsw i64 %address.local1050, 7, !dbg !22
  %1789 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %45, !dbg !22
  %address.local1053 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %1789, i64 %address.local1051, !dbg !22
  %address.local1055 = and i64 %address.local1044, 12, !dbg !12
  %address.local1056 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1055, !dbg !22
  store half %1788, ptr addrspace(1) %address.local1056, align 2, !dbg !22
  %address.local1058 = or disjoint i64 %address.local1055, 1, !dbg !22
  %address.local1059 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1058, !dbg !22
  store half %1787, ptr addrspace(1) %address.local1059, align 2, !dbg !22
  %address.local1061 = or disjoint i64 %address.local1055, 2, !dbg !22
  %address.local1062 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1061, !dbg !22
  store half %1786, ptr addrspace(1) %address.local1062, align 2, !dbg !22
  %address.local1064 = or disjoint i64 %address.local1055, 3, !dbg !22
  %address.local1065 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1064, !dbg !22
  store half %1785, ptr addrspace(1) %address.local1065, align 2, !dbg !22
  %address.local1067 = or disjoint i64 %address.local1055, 16, !dbg !22
  %address.local1068 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1067, !dbg !22
  store half %1773, ptr addrspace(1) %address.local1068, align 2, !dbg !22
  %address.local1070 = or disjoint i64 %address.local1055, 17, !dbg !22
  %address.local1071 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1070, !dbg !22
  store half %1772, ptr addrspace(1) %address.local1071, align 2, !dbg !22
  %address.local1073 = or disjoint i64 %address.local1055, 18, !dbg !22
  %address.local1074 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1073, !dbg !22
  store half %1771, ptr addrspace(1) %address.local1074, align 2, !dbg !22
  %address.local1076 = or disjoint i64 %address.local1055, 19, !dbg !22
  %address.local1077 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1076, !dbg !22
  store half %1770, ptr addrspace(1) %address.local1077, align 2, !dbg !22
  %address.local1081 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local1053, i64 16384, !dbg !22
  %address.local1082 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1055, !dbg !22
  store half %1758, ptr addrspace(1) %address.local1082, align 2, !dbg !22
  %address.local1084 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1058, !dbg !22
  store half %1757, ptr addrspace(1) %address.local1084, align 2, !dbg !22
  %address.local1086 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1061, !dbg !22
  store half %1756, ptr addrspace(1) %address.local1086, align 2, !dbg !22
  %address.local1088 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1064, !dbg !22
  store half %1755, ptr addrspace(1) %address.local1088, align 2, !dbg !22
  %address.local1090 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1067, !dbg !22
  store half %1743, ptr addrspace(1) %address.local1090, align 2, !dbg !22
  %address.local1092 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1070, !dbg !22
  store half %1742, ptr addrspace(1) %address.local1092, align 2, !dbg !22
  %address.local1094 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1073, !dbg !22
  store half %1741, ptr addrspace(1) %address.local1094, align 2, !dbg !22
  %address.local1096 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1076, !dbg !22
  store half %1740, ptr addrspace(1) %address.local1096, align 2, !dbg !22
  %1790 = insertelement <4 x float> poison, float %1629, i64 0, !dbg !20
  %1791 = insertelement <4 x float> %1790, float %1635, i64 1, !dbg !20
  %1792 = insertelement <4 x float> %1791, float %1630, i64 2, !dbg !20
  %1793 = insertelement <4 x float> %1792, float %1636, i64 3, !dbg !20
  %normalized109 = fmul <4 x float> %1793, %row.reciprocal, !dbg !20
  %1794 = fptrunc <4 x float> %normalized109 to <4 x half>, !dbg !21
  %1795 = extractelement <4 x half> %1794, i64 3, !dbg !21
  %1796 = extractelement <4 x half> %1794, i64 2, !dbg !21
  %1797 = extractelement <4 x half> %1794, i64 1, !dbg !21
  %1798 = extractelement <4 x half> %1794, i64 0, !dbg !21
  %1799 = insertelement <4 x float> poison, float %1601, i64 0, !dbg !20
  %1800 = insertelement <4 x float> %1799, float %1607, i64 1, !dbg !20
  %1801 = insertelement <4 x float> %1800, float %1602, i64 2, !dbg !20
  %1802 = insertelement <4 x float> %1801, float %1608, i64 3, !dbg !20
  %normalized110 = fmul <4 x float> %1802, %row.reciprocal, !dbg !20
  %1803 = fptrunc <4 x float> %normalized110 to <4 x half>, !dbg !21
  %1804 = extractelement <4 x half> %1803, i64 3, !dbg !21
  %1805 = extractelement <4 x half> %1803, i64 2, !dbg !21
  %1806 = extractelement <4 x half> %1803, i64 1, !dbg !21
  %1807 = extractelement <4 x half> %1803, i64 0, !dbg !21
  %1808 = insertelement <4 x float> poison, float %1572, i64 0, !dbg !20
  %1809 = insertelement <4 x float> %1808, float %1578, i64 1, !dbg !20
  %1810 = insertelement <4 x float> %1809, float %1573, i64 2, !dbg !20
  %1811 = insertelement <4 x float> %1810, float %1579, i64 3, !dbg !20
  %normalized112 = fmul <4 x float> %1811, %row.reciprocal105, !dbg !20
  %1812 = fptrunc <4 x float> %normalized112 to <4 x half>, !dbg !21
  %1813 = extractelement <4 x half> %1812, i64 3, !dbg !21
  %1814 = extractelement <4 x half> %1812, i64 2, !dbg !21
  %1815 = extractelement <4 x half> %1812, i64 1, !dbg !21
  %1816 = extractelement <4 x half> %1812, i64 0, !dbg !21
  %1817 = insertelement <4 x float> poison, float %1544, i64 0, !dbg !20
  %1818 = insertelement <4 x float> %1817, float %1550, i64 1, !dbg !20
  %1819 = insertelement <4 x float> %1818, float %1545, i64 2, !dbg !20
  %1820 = insertelement <4 x float> %1819, float %1551, i64 3, !dbg !20
  %normalized113 = fmul <4 x float> %1820, %row.reciprocal105, !dbg !20
  %1821 = fptrunc <4 x float> %normalized113 to <4 x half>, !dbg !21
  %1822 = extractelement <4 x half> %1821, i64 3, !dbg !21
  %1823 = extractelement <4 x half> %1821, i64 2, !dbg !21
  %1824 = extractelement <4 x half> %1821, i64 1, !dbg !21
  %1825 = extractelement <4 x half> %1821, i64 0, !dbg !21
  %address.local1112 = or disjoint i64 %address.local1055, 32, !dbg !22
  %address.local1113 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1112, !dbg !22
  store half %1825, ptr addrspace(1) %address.local1113, align 2, !dbg !22
  %address.local1115 = or disjoint i64 %address.local1055, 33, !dbg !22
  %address.local1116 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1115, !dbg !22
  store half %1824, ptr addrspace(1) %address.local1116, align 2, !dbg !22
  %address.local1118 = or disjoint i64 %address.local1055, 34, !dbg !22
  %address.local1119 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1118, !dbg !22
  store half %1823, ptr addrspace(1) %address.local1119, align 2, !dbg !22
  %address.local1121 = or disjoint i64 %address.local1055, 35, !dbg !22
  %address.local1122 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1121, !dbg !22
  store half %1822, ptr addrspace(1) %address.local1122, align 2, !dbg !22
  %address.local1124 = or disjoint i64 %address.local1055, 48, !dbg !22
  %address.local1125 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1124, !dbg !22
  store half %1816, ptr addrspace(1) %address.local1125, align 2, !dbg !22
  %address.local1127 = or disjoint i64 %address.local1055, 49, !dbg !22
  %address.local1128 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1127, !dbg !22
  store half %1815, ptr addrspace(1) %address.local1128, align 2, !dbg !22
  %address.local1130 = or disjoint i64 %address.local1055, 50, !dbg !22
  %address.local1131 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1130, !dbg !22
  store half %1814, ptr addrspace(1) %address.local1131, align 2, !dbg !22
  %address.local1133 = or i64 %address.local1044, 51, !dbg !22
  %address.local1134 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1133, !dbg !22
  store half %1813, ptr addrspace(1) %address.local1134, align 2, !dbg !22
  %address.local1139 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1112, !dbg !22
  store half %1807, ptr addrspace(1) %address.local1139, align 2, !dbg !22
  %address.local1141 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1115, !dbg !22
  store half %1806, ptr addrspace(1) %address.local1141, align 2, !dbg !22
  %address.local1143 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1118, !dbg !22
  store half %1805, ptr addrspace(1) %address.local1143, align 2, !dbg !22
  %address.local1145 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1121, !dbg !22
  store half %1804, ptr addrspace(1) %address.local1145, align 2, !dbg !22
  %address.local1147 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1124, !dbg !22
  store half %1798, ptr addrspace(1) %address.local1147, align 2, !dbg !22
  %address.local1149 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1127, !dbg !22
  store half %1797, ptr addrspace(1) %address.local1149, align 2, !dbg !22
  %address.local1151 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1130, !dbg !22
  store half %1796, ptr addrspace(1) %address.local1151, align 2, !dbg !22
  %address.local1153 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1133, !dbg !22
  store half %1795, ptr addrspace(1) %address.local1153, align 2, !dbg !22
  %1826 = insertelement <4 x float> poison, float %1515, i64 0, !dbg !20
  %1827 = insertelement <4 x float> %1826, float %1521, i64 1, !dbg !20
  %1828 = insertelement <4 x float> %1827, float %1516, i64 2, !dbg !20
  %1829 = insertelement <4 x float> %1828, float %1522, i64 3, !dbg !20
  %normalized115 = fmul <4 x float> %1829, %row.reciprocal, !dbg !20
  %1830 = fptrunc <4 x float> %normalized115 to <4 x half>, !dbg !21
  %1831 = extractelement <4 x half> %1830, i64 3, !dbg !21
  %1832 = extractelement <4 x half> %1830, i64 2, !dbg !21
  %1833 = extractelement <4 x half> %1830, i64 1, !dbg !21
  %1834 = extractelement <4 x half> %1830, i64 0, !dbg !21
  %1835 = insertelement <4 x float> poison, float %1487, i64 0, !dbg !20
  %1836 = insertelement <4 x float> %1835, float %1493, i64 1, !dbg !20
  %1837 = insertelement <4 x float> %1836, float %1488, i64 2, !dbg !20
  %1838 = insertelement <4 x float> %1837, float %1494, i64 3, !dbg !20
  %normalized116 = fmul <4 x float> %1838, %row.reciprocal, !dbg !20
  %1839 = fptrunc <4 x float> %normalized116 to <4 x half>, !dbg !21
  %1840 = extractelement <4 x half> %1839, i64 3, !dbg !21
  %1841 = extractelement <4 x half> %1839, i64 2, !dbg !21
  %1842 = extractelement <4 x half> %1839, i64 1, !dbg !21
  %1843 = extractelement <4 x half> %1839, i64 0, !dbg !21
  %1844 = insertelement <4 x float> poison, float %1458, i64 0, !dbg !20
  %1845 = insertelement <4 x float> %1844, float %1464, i64 1, !dbg !20
  %1846 = insertelement <4 x float> %1845, float %1459, i64 2, !dbg !20
  %1847 = insertelement <4 x float> %1846, float %1465, i64 3, !dbg !20
  %normalized118 = fmul <4 x float> %1847, %row.reciprocal105, !dbg !20
  %1848 = fptrunc <4 x float> %normalized118 to <4 x half>, !dbg !21
  %1849 = extractelement <4 x half> %1848, i64 3, !dbg !21
  %1850 = extractelement <4 x half> %1848, i64 2, !dbg !21
  %1851 = extractelement <4 x half> %1848, i64 1, !dbg !21
  %1852 = extractelement <4 x half> %1848, i64 0, !dbg !21
  %1853 = insertelement <4 x float> poison, float %1430, i64 0, !dbg !20
  %1854 = insertelement <4 x float> %1853, float %1436, i64 1, !dbg !20
  %1855 = insertelement <4 x float> %1854, float %1431, i64 2, !dbg !20
  %1856 = insertelement <4 x float> %1855, float %1437, i64 3, !dbg !20
  %normalized119 = fmul <4 x float> %1856, %row.reciprocal105, !dbg !20
  %1857 = fptrunc <4 x float> %normalized119 to <4 x half>, !dbg !21
  %1858 = extractelement <4 x half> %1857, i64 3, !dbg !21
  %1859 = extractelement <4 x half> %1857, i64 2, !dbg !21
  %1860 = extractelement <4 x half> %1857, i64 1, !dbg !21
  %1861 = extractelement <4 x half> %1857, i64 0, !dbg !21
  %address.local1169 = or disjoint i64 %address.local1055, 64, !dbg !22
  %address.local1170 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1169, !dbg !22
  store half %1861, ptr addrspace(1) %address.local1170, align 2, !dbg !22
  %address.local1172 = or disjoint i64 %address.local1055, 65, !dbg !22
  %address.local1173 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1172, !dbg !22
  store half %1860, ptr addrspace(1) %address.local1173, align 2, !dbg !22
  %address.local1175 = or disjoint i64 %address.local1055, 66, !dbg !22
  %address.local1176 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1175, !dbg !22
  store half %1859, ptr addrspace(1) %address.local1176, align 2, !dbg !22
  %address.local1178 = or disjoint i64 %address.local1055, 67, !dbg !22
  %address.local1179 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1178, !dbg !22
  store half %1858, ptr addrspace(1) %address.local1179, align 2, !dbg !22
  %address.local1181 = or disjoint i64 %address.local1055, 80, !dbg !22
  %address.local1182 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1181, !dbg !22
  store half %1852, ptr addrspace(1) %address.local1182, align 2, !dbg !22
  %address.local1184 = or disjoint i64 %address.local1055, 81, !dbg !22
  %address.local1185 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1184, !dbg !22
  store half %1851, ptr addrspace(1) %address.local1185, align 2, !dbg !22
  %address.local1187 = or disjoint i64 %address.local1055, 82, !dbg !22
  %address.local1188 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1187, !dbg !22
  store half %1850, ptr addrspace(1) %address.local1188, align 2, !dbg !22
  %address.local1190 = or disjoint i64 %address.local1055, 83, !dbg !22
  %address.local1191 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1190, !dbg !22
  store half %1849, ptr addrspace(1) %address.local1191, align 2, !dbg !22
  %address.local1196 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1169, !dbg !22
  store half %1843, ptr addrspace(1) %address.local1196, align 2, !dbg !22
  %address.local1198 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1172, !dbg !22
  store half %1842, ptr addrspace(1) %address.local1198, align 2, !dbg !22
  %address.local1200 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1175, !dbg !22
  store half %1841, ptr addrspace(1) %address.local1200, align 2, !dbg !22
  %address.local1202 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1178, !dbg !22
  store half %1840, ptr addrspace(1) %address.local1202, align 2, !dbg !22
  %address.local1204 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1181, !dbg !22
  store half %1834, ptr addrspace(1) %address.local1204, align 2, !dbg !22
  %address.local1206 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1184, !dbg !22
  store half %1833, ptr addrspace(1) %address.local1206, align 2, !dbg !22
  %address.local1208 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1187, !dbg !22
  store half %1832, ptr addrspace(1) %address.local1208, align 2, !dbg !22
  %address.local1210 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1190, !dbg !22
  store half %1831, ptr addrspace(1) %address.local1210, align 2, !dbg !22
  %1862 = insertelement <4 x float> poison, float %1401, i64 0, !dbg !20
  %1863 = insertelement <4 x float> %1862, float %1407, i64 1, !dbg !20
  %1864 = insertelement <4 x float> %1863, float %1402, i64 2, !dbg !20
  %1865 = insertelement <4 x float> %1864, float %1408, i64 3, !dbg !20
  %normalized121 = fmul <4 x float> %1865, %row.reciprocal, !dbg !20
  %1866 = fptrunc <4 x float> %normalized121 to <4 x half>, !dbg !21
  %1867 = extractelement <4 x half> %1866, i64 3, !dbg !21
  %1868 = extractelement <4 x half> %1866, i64 2, !dbg !21
  %1869 = extractelement <4 x half> %1866, i64 1, !dbg !21
  %1870 = extractelement <4 x half> %1866, i64 0, !dbg !21
  %1871 = insertelement <4 x float> poison, float %1373, i64 0, !dbg !20
  %1872 = insertelement <4 x float> %1871, float %1379, i64 1, !dbg !20
  %1873 = insertelement <4 x float> %1872, float %1374, i64 2, !dbg !20
  %1874 = insertelement <4 x float> %1873, float %1380, i64 3, !dbg !20
  %normalized122 = fmul <4 x float> %1874, %row.reciprocal, !dbg !20
  %1875 = fptrunc <4 x float> %normalized122 to <4 x half>, !dbg !21
  %1876 = extractelement <4 x half> %1875, i64 3, !dbg !21
  %1877 = extractelement <4 x half> %1875, i64 2, !dbg !21
  %1878 = extractelement <4 x half> %1875, i64 1, !dbg !21
  %1879 = extractelement <4 x half> %1875, i64 0, !dbg !21
  %1880 = insertelement <4 x float> poison, float %1344, i64 0, !dbg !20
  %1881 = insertelement <4 x float> %1880, float %1350, i64 1, !dbg !20
  %1882 = insertelement <4 x float> %1881, float %1345, i64 2, !dbg !20
  %1883 = insertelement <4 x float> %1882, float %1351, i64 3, !dbg !20
  %normalized124 = fmul <4 x float> %1883, %row.reciprocal105, !dbg !20
  %1884 = fptrunc <4 x float> %normalized124 to <4 x half>, !dbg !21
  %1885 = extractelement <4 x half> %1884, i64 3, !dbg !21
  %1886 = extractelement <4 x half> %1884, i64 2, !dbg !21
  %1887 = extractelement <4 x half> %1884, i64 1, !dbg !21
  %1888 = extractelement <4 x half> %1884, i64 0, !dbg !21
  %1889 = insertelement <4 x float> poison, float %1316, i64 0, !dbg !20
  %1890 = insertelement <4 x float> %1889, float %1322, i64 1, !dbg !20
  %1891 = insertelement <4 x float> %1890, float %1317, i64 2, !dbg !20
  %1892 = insertelement <4 x float> %1891, float %1323, i64 3, !dbg !20
  %normalized125 = fmul <4 x float> %1892, %row.reciprocal105, !dbg !20
  %1893 = fptrunc <4 x float> %normalized125 to <4 x half>, !dbg !21
  %1894 = extractelement <4 x half> %1893, i64 3, !dbg !21
  %1895 = extractelement <4 x half> %1893, i64 2, !dbg !21
  %1896 = extractelement <4 x half> %1893, i64 1, !dbg !21
  %1897 = extractelement <4 x half> %1893, i64 0, !dbg !21
  %address.local1226 = or disjoint i64 %address.local1055, 96, !dbg !22
  %address.local1227 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1226, !dbg !22
  store half %1897, ptr addrspace(1) %address.local1227, align 2, !dbg !22
  %address.local1229 = or disjoint i64 %address.local1055, 97, !dbg !22
  %address.local1230 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1229, !dbg !22
  store half %1896, ptr addrspace(1) %address.local1230, align 2, !dbg !22
  %address.local1232 = or disjoint i64 %address.local1055, 98, !dbg !22
  %address.local1233 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1232, !dbg !22
  store half %1895, ptr addrspace(1) %address.local1233, align 2, !dbg !22
  %address.local1235 = or disjoint i64 %address.local1055, 99, !dbg !22
  %address.local1236 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1235, !dbg !22
  store half %1894, ptr addrspace(1) %address.local1236, align 2, !dbg !22
  %address.local1238 = or disjoint i64 %address.local1055, 112, !dbg !22
  %address.local1239 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1238, !dbg !22
  store half %1888, ptr addrspace(1) %address.local1239, align 2, !dbg !22
  %address.local1241 = or disjoint i64 %address.local1055, 113, !dbg !22
  %address.local1242 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1241, !dbg !22
  store half %1887, ptr addrspace(1) %address.local1242, align 2, !dbg !22
  %address.local1244 = or disjoint i64 %address.local1055, 114, !dbg !22
  %address.local1245 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1244, !dbg !22
  store half %1886, ptr addrspace(1) %address.local1245, align 2, !dbg !22
  %address.local1247 = or i64 %address.local1044, 115, !dbg !22
  %address.local1248 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1247, !dbg !22
  store half %1885, ptr addrspace(1) %address.local1248, align 2, !dbg !22
  %address.local1253 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1226, !dbg !22
  store half %1879, ptr addrspace(1) %address.local1253, align 2, !dbg !22
  %address.local1255 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1229, !dbg !22
  store half %1878, ptr addrspace(1) %address.local1255, align 2, !dbg !22
  %address.local1257 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1232, !dbg !22
  store half %1877, ptr addrspace(1) %address.local1257, align 2, !dbg !22
  %address.local1259 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1235, !dbg !22
  store half %1876, ptr addrspace(1) %address.local1259, align 2, !dbg !22
  %address.local1261 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1238, !dbg !22
  store half %1870, ptr addrspace(1) %address.local1261, align 2, !dbg !22
  %address.local1263 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1241, !dbg !22
  store half %1869, ptr addrspace(1) %address.local1263, align 2, !dbg !22
  %address.local1265 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1244, !dbg !22
  store half %1868, ptr addrspace(1) %address.local1265, align 2, !dbg !22
  %address.local1267 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1247, !dbg !22
  store half %1867, ptr addrspace(1) %address.local1267, align 2, !dbg !22
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

; Function Attrs: convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.readfirstlane.i32(i32) #3

; Function Attrs: nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.asyncmark() #4

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
attributes #1 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1" "frisk.global_prefetch"="8" "frisk.k_dma"="64-swizzle:1" "frisk.k_lds_prefetch"="8" "frisk.k_tail_prefetch"="4" "frisk.register_comparisons"="1" "frisk.register_lifetimes"="1" "frisk.softmax_cvt_pack"="rtz" "frisk.softmax_fragments"="pv" "uniform-work-group-size" }
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
!llvm.module.flags = !{!38}

!0 = !DIFile(filename: "causal.input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/causal_cvt_pack_mytest")
!1 = !{}
!2 = !DISubroutineType(types: !1)
!3 = distinct !DICompileUnit(language: DW_LANG_C, file: !0, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!4 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !0, file: !0, line: 23, type: !2, scopeLine: 23, spFlags: DISPFlagDefinition, unit: !3)
!5 = !DILocation(line: 23, column: 3, scope: !4)
!7 = !DILocation(line: 60, column: 24, scope: !4)
!8 = !DILocation(line: 70, column: 22, scope: !4)
!12 = !DILocation(line: 72, column: 24, scope: !4)
!13 = !DILocation(line: 102, column: 23, scope: !4)
!14 = !DILocation(line: 0, scope: !4)
!15 = !DILocation(line: 71, column: 22, scope: !4)
!16 = !DILocation(line: 73, column: 26, scope: !4)
!17 = !DILocation(line: 101, column: 20, scope: !4)
!18 = !DILocation(line: 77, column: 19, scope: !4)
!19 = !DILocation(line: 79, column: 24, scope: !4)
!20 = !DILocation(line: 107, column: 23, scope: !4)
!21 = !DILocation(line: 108, column: 19, scope: !4)
!22 = !DILocation(line: 109, column: 9, scope: !4)
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
!37 = !{i32 256, i32 1, i32 1}
!38 = !{i32 2, !"Debug Info Version", i32 3}
!39 = !{i32 0, i32 256}
!40 = !{i32 4}
!41 = !{i64 8192}
!42 = !{i32 0}
!43 = !{i32 1}
!44 = !{i32 2}
!45 = !{i32 3}
!46 = !{i32 5}
!47 = !{i32 6}
!48 = !{i32 7}
!49 = !{i32 8}
!50 = !{i32 9}
!51 = !{i32 10}
!52 = !{i32 11}
!53 = !{i32 12}
!54 = !{i32 13}
!55 = !{i32 14}
!56 = !{i32 15}
!57 = !{i32 16}
!58 = !{i32 17}
!59 = !{i32 18}
!60 = !{i32 19}
!61 = !{i32 20}
!62 = !{i32 21}
!63 = !{i32 22}
