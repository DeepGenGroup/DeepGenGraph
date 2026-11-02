; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/causal_best_mytest/causal.input.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/causal_best_mytest/causal.input.mlir"

@shm_0 = addrspace(3) global [16384 x i8] undef, align 16

; Function Attrs: convergent willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

declare float @__ocml_exp2_f32(float)

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3) #1 !dbg !4 !reqd_work_group_size !36 {
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
  %17 = lshr i64 %12, 2, !dbg !5
  %18 = lshr i64 %12, 10, !dbg !8
  %19 = add nuw nsw i64 %17, %18, !dbg !8
  %20 = shl i64 %16, 7, !dbg !8
  %21 = mul nsw i64 %18, -4096, !dbg !8
  %22 = add i64 %20, %21, !dbg !8
  %23 = shl i64 %19, 19, !dbg !8
  %24 = and i64 %23, 16252928, !dbg !8
  %25 = shl nsw i64 %22, 7, !dbg !8
  %26 = add i64 %24, %25, !dbg !8
  %27 = zext nneg i32 %kdma.tid to i64, !dbg !8
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #9, !dbg !8, !range !38, !frisk.register_lifetimes !1
  %address.local = zext nneg i32 %address.thread to i64, !dbg !8
  %address.local472 = shl nuw nsw i64 %address.local, 3, !dbg !8
  %address.local473 = and i64 %address.local472, 1536, !dbg !8
  %address.local474 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local473, !dbg !8
  %address.local475 = ptrtoint ptr addrspace(3) %address.local474 to i64, !dbg !8
  %address.local476 = trunc i64 %address.local475 to i32, !dbg !8
  %28 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local476), !dbg !8
  %address.local477 = shl nuw nsw i64 %address.local, 7, !dbg !8
  %address.local478 = and i64 %address.local477, 16256, !dbg !8
  %address.local479 = lshr i64 %address.local, 4, !dbg !8
  %address.local480 = and i64 %address.local479, 8, !dbg !8
  %address.local481 = or disjoint i64 %address.local478, %address.local480, !dbg !8
  %29 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %.idx1727 = or disjoint i64 %26, %address.local481, !dbg !8
  %.idx1727.tr = trunc i64 %.idx1727 to i32, !dbg !8
  %dma.buffer.offset1724 = shl i32 %.idx1727.tr, 1, !dbg !8
  %30 = insertelement <2 x i64> poison, i64 %29, i64 0, !dbg !8
  %31 = bitcast <2 x i64> %30 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc1725 = shufflevector <4 x i32> %31, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
  %dma.buffer.lds1726 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %28), !dbg !8
  %32 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1726, i32 %dma.buffer.offset1724, <4 x i32> %dma.buffer.rsrc1725) #10, !dbg !8
  %address.local483 = or disjoint i64 %address.local, 256, !dbg !8
  %address.local484 = shl nuw nsw i64 %address.local483, 3, !dbg !8
  %address.local485 = and i64 %address.local484, 3584, !dbg !8
  %address.local486 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local485, !dbg !8
  %address.local487 = ptrtoint ptr addrspace(3) %address.local486 to i64, !dbg !8
  %address.local488 = trunc i64 %address.local487 to i32, !dbg !8
  %33 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local488), !dbg !8
  %address.local491 = lshr i64 %address.local483, 4, !dbg !8
  %address.local492 = and i64 %address.local491, 24, !dbg !8
  %address.local493 = or disjoint i64 %address.local478, %address.local492, !dbg !8
  %.idx1728 = or disjoint i64 %26, %address.local493, !dbg !8
  %.idx1728.tr = trunc i64 %.idx1728 to i32, !dbg !8
  %dma.buffer.offset1721 = shl i32 %.idx1728.tr, 1, !dbg !8
  %dma.buffer.lds1723 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %33), !dbg !8
  %34 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1723, i32 %dma.buffer.offset1721, <4 x i32> %dma.buffer.rsrc1725) #10, !dbg !8
  %35 = call i32 @llvm.amdgcn.workgroup.id.z(), !dbg !5
  %36 = sext i32 %35 to i64, !dbg !5
  %37 = and i64 %27, 240, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %address.local495 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local473, !dbg !8
  %address.local496 = ptrtoint ptr addrspace(3) %address.local495 to i64, !dbg !8
  %address.local497 = trunc i64 %address.local496 to i32, !dbg !8
  %38 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local497), !dbg !8
  %39 = or disjoint i64 %26, 32, !dbg !8
  %40 = or disjoint i64 %address.local481, %39, !dbg !8
  %.tr = trunc i64 %40 to i32, !dbg !8
  %dma.buffer.offset1718 = shl i32 %.tr, 1, !dbg !8
  %dma.buffer.lds1720 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %38), !dbg !8
  %41 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1720, i32 %dma.buffer.offset1718, <4 x i32> %dma.buffer.rsrc1725) #10, !dbg !8
  %address.local499 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local485, !dbg !8
  %address.local500 = ptrtoint ptr addrspace(3) %address.local499 to i64, !dbg !8
  %address.local501 = trunc i64 %address.local500 to i32, !dbg !8
  %42 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local501), !dbg !8
  %43 = or disjoint i64 %address.local493, %39, !dbg !8
  %.tr1729 = trunc i64 %43 to i32, !dbg !8
  %dma.buffer.offset1715 = shl i32 %.tr1729, 1, !dbg !8
  %dma.buffer.lds1717 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %42), !dbg !8
  %44 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1717, i32 %dma.buffer.offset1715, <4 x i32> %dma.buffer.rsrc1725) #10, !dbg !8
  %address.local503 = lshr i32 %address.thread, 2, !dbg !11
  %address.local504 = and i32 %address.local503, 4, !dbg !11
  %address.local505 = shl nuw nsw i32 %address.thread, 1, !dbg !11
  %address.local506 = and i32 %address.local505, 384, !dbg !11
  %address.local507 = or disjoint i32 %address.local504, %address.local506, !dbg !11
  %address.local508 = shl nuw nsw i32 %address.thread, 3, !dbg !11
  %address.local509 = and i32 %address.local508, 120, !dbg !11
  %address.local510 = or disjoint i32 %address.local507, %address.local509, !dbg !11
  %address.local511 = shl nuw nsw i32 %address.thread, 5, !dbg !11
  %address.local512 = and i32 %address.local511, 1024, !dbg !11
  %address.local513 = or disjoint i32 %address.local510, %address.local512, !dbg !11
  %address.local514 = zext nneg i32 %address.local513 to i64, !dbg !11
  %address.local515 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local514, !dbg !11
  %45 = load <4 x half>, ptr addrspace(3) %address.local515, align 8, !dbg !11
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local515, i64 4096, !dbg !11
  %46 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !11
  %47 = fmul <4 x half> %45, splat (half 0xH3015), !dbg !11
  %lds.const.gep1525 = getelementptr i8, ptr addrspace(3) %address.local515, i64 1024, !dbg !11
  %48 = load <4 x half>, ptr addrspace(3) %lds.const.gep1525, align 8, !dbg !11
  %49 = fmul <4 x half> %46, splat (half 0xH3015), !dbg !11
  %lds.const.gep1526 = getelementptr i8, ptr addrspace(3) %address.local515, i64 5120, !dbg !11
  %50 = load <4 x half>, ptr addrspace(3) %lds.const.gep1526, align 8, !dbg !11
  %51 = fmul <4 x half> %48, splat (half 0xH3015), !dbg !11
  %52 = fmul <4 x half> %50, splat (half 0xH3015), !dbg !11
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %53 = or disjoint i64 %26, 64, !dbg !8
  %54 = or disjoint i64 %address.local481, %53, !dbg !8
  %.tr1730 = trunc i64 %54 to i32, !dbg !8
  %dma.buffer.offset1712 = shl i32 %.tr1730, 1, !dbg !8
  %55 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1726, i32 %dma.buffer.offset1712, <4 x i32> %dma.buffer.rsrc1725) #10, !dbg !8
  %56 = or disjoint i64 %address.local493, %53, !dbg !8
  %.tr1731 = trunc i64 %56 to i32, !dbg !8
  %dma.buffer.offset1709 = shl i32 %.tr1731, 1, !dbg !8
  %57 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1723, i32 %dma.buffer.offset1709, <4 x i32> %dma.buffer.rsrc1725) #10, !dbg !8
  %lds.const.gep1527 = getelementptr i8, ptr addrspace(3) %address.local515, i64 8192, !dbg !11
  %58 = load <4 x half>, ptr addrspace(3) %lds.const.gep1527, align 8, !dbg !11
  %lds.const.gep1528 = getelementptr i8, ptr addrspace(3) %address.local515, i64 12288, !dbg !11
  %59 = load <4 x half>, ptr addrspace(3) %lds.const.gep1528, align 8, !dbg !11
  %lds.const.gep1529 = getelementptr i8, ptr addrspace(3) %address.local515, i64 9216, !dbg !11
  %60 = load <4 x half>, ptr addrspace(3) %lds.const.gep1529, align 8, !dbg !11
  %61 = fmul <4 x half> %58, splat (half 0xH3015), !dbg !11
  %lds.const.gep1530 = getelementptr i8, ptr addrspace(3) %address.local515, i64 13312, !dbg !11
  %62 = load <4 x half>, ptr addrspace(3) %lds.const.gep1530, align 8, !dbg !11
  %63 = fmul <4 x half> %59, splat (half 0xH3015), !dbg !11
  %64 = fmul <4 x half> %60, splat (half 0xH3015), !dbg !11
  %65 = fmul <4 x half> %62, splat (half 0xH3015), !dbg !11
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %66 = or disjoint i64 %26, 96, !dbg !8
  %67 = or disjoint i64 %address.local481, %66, !dbg !8
  %.tr1732 = trunc i64 %67 to i32, !dbg !8
  %dma.buffer.offset1706 = shl i32 %.tr1732, 1, !dbg !8
  %68 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1720, i32 %dma.buffer.offset1706, <4 x i32> %dma.buffer.rsrc1725) #10, !dbg !8
  %69 = or disjoint i64 %address.local493, %66, !dbg !8
  %.tr1733 = trunc i64 %69 to i32, !dbg !8
  %dma.buffer.offset1703 = shl i32 %.tr1733, 1, !dbg !8
  %70 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1717, i32 %dma.buffer.offset1703, <4 x i32> %dma.buffer.rsrc1725) #10, !dbg !8
  %71 = load <4 x half>, ptr addrspace(3) %address.local515, align 8, !dbg !11
  %72 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !11
  %73 = load <4 x half>, ptr addrspace(3) %lds.const.gep1525, align 8, !dbg !11
  %74 = fmul <4 x half> %71, splat (half 0xH3015), !dbg !11
  %75 = load <4 x half>, ptr addrspace(3) %lds.const.gep1526, align 8, !dbg !11
  %76 = fmul <4 x half> %72, splat (half 0xH3015), !dbg !11
  %77 = fmul <4 x half> %73, splat (half 0xH3015), !dbg !11
  %78 = fmul <4 x half> %75, splat (half 0xH3015), !dbg !11
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %79 = load <4 x half>, ptr addrspace(3) %lds.const.gep1527, align 8, !dbg !11
  %80 = load <4 x half>, ptr addrspace(3) %lds.const.gep1528, align 8, !dbg !11
  %81 = load <4 x half>, ptr addrspace(3) %lds.const.gep1529, align 8, !dbg !11
  %82 = fmul <4 x half> %79, splat (half 0xH3015), !dbg !11
  %83 = load <4 x half>, ptr addrspace(3) %lds.const.gep1530, align 8, !dbg !11
  %84 = fmul <4 x half> %80, splat (half 0xH3015), !dbg !11
  %85 = fmul <4 x half> %81, splat (half 0xH3015), !dbg !11
  %86 = fmul <4 x half> %83, splat (half 0xH3015), !dbg !11
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %87 = shl i64 %17, 7, !dbg !14
  %88 = and i64 %87, 3968, !dbg !14
  %89 = shl nsw i64 %36, 12, !dbg !14
  %90 = or disjoint i64 %88, %89, !dbg !14
  %91 = shl nuw nsw i64 %27, 2, !dbg !15
  %92 = and i64 %91, 3084, !dbg !15
  %93 = and i32 %kdma.tid, 1, !dbg !15
  %index.compact412 = zext nneg i32 %93 to i64, !dbg !15
  %94 = or disjoint i64 %92, %index.compact412, !dbg !15
  %95 = add nuw nsw i64 %94, 1, !dbg !15
  %96 = lshr i64 %95, 1, !dbg !15
  %97 = add i64 %20, %27, !dbg !17
  %98 = lshr i64 %27, 2, !dbg !17
  %99 = and i64 %98, 48, !dbg !17
  %100 = add i64 %97, %99, !dbg !17
  %101 = sub i64 %100, %37, !dbg !17
  %102 = and i64 %27, 48, !dbg !17
  %103 = add i64 %101, 1, !dbg !18
  %104 = add i64 %101, 65, !dbg !18
  br label %105, !dbg !7

105:                                              ; preds = %113, %4
  %106 = phi i64 [ %1256, %113 ], [ 0, %4 ], !dbg !7
  %107 = phi [2 x <8 x float>] [ %1241, %113 ], [ zeroinitializer, %4 ], !dbg !7
  %108 = phi [2 x <1 x float>] [ %1007, %113 ], [ zeroinitializer, %4 ], !dbg !7
  %109 = phi [2 x <1 x float>] [ %1243, %113 ], [ [<1 x float> splat (float 0xFFF0000000000000), <1 x float> splat (float 0xFFF0000000000000)], %4 ], !dbg !7
  %110 = phi [2 x <8 x float>] [ %1247, %113 ], [ zeroinitializer, %4 ], !dbg !7
  %111 = phi [2 x <8 x float>] [ %1251, %113 ], [ zeroinitializer, %4 ], !dbg !7
  %112 = phi [2 x <8 x float>] [ %1255, %113 ], [ zeroinitializer, %4 ], !dbg !7
  %.not = icmp samesign ugt i64 %106, %16, !dbg !7
  br i1 %.not, label %1257, label %113, !dbg !7

113:                                              ; preds = %105
  %114 = lshr i64 %106, 12, !dbg !14
  %115 = add nuw nsw i64 %114, %17, !dbg !14
  %116 = lshr i64 %106, 5, !dbg !14
  %117 = and i64 %116, 127, !dbg !14
  %118 = shl nsw i64 %106, 7, !dbg !14
  %119 = shl i64 %115, 19, !dbg !14
  %120 = and i64 %119, 16252928, !dbg !14
  %121 = shl i64 %116, 19, !dbg !14
  %122 = shl i64 %106, 14, !dbg !14
  %123 = sub i64 %122, %121, !dbg !14
  %address.thread566 = call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #9, !dbg !14, !range !38, !frisk.register_lifetimes !1
  %address.local567 = zext nneg i32 %address.thread566 to i64, !dbg !8
  %address.local568 = shl nuw nsw i64 %address.local567, 3, !dbg !8
  %address.local569 = and i64 %address.local568, 1536, !dbg !8
  %address.local570 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local569, !dbg !8
  %address.local571 = ptrtoint ptr addrspace(3) %address.local570 to i64, !dbg !8
  %address.local572 = trunc i64 %address.local571 to i32, !dbg !8
  %124 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local572), !dbg !14
  %125 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !14
  %.idx1736 = or disjoint i64 %120, %117, !dbg !14
  %126 = add i64 %.idx1736, %123, !dbg !14
  %127 = insertelement <2 x i64> poison, i64 %125, i64 0, !dbg !14
  %128 = bitcast <2 x i64> %127 to <4 x i32>, !dbg !14
  %dma.buffer.rsrc1701 = shufflevector <4 x i32> %128, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !14
  %dma.buffer.lds1702 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %124), !dbg !14
  %kdma.source = zext nneg i32 %kdma.owner to i64, !dbg !14
  %kdma.source1878 = lshr i64 %kdma.source, 4, !dbg !14
  %kdma.source1879 = and i64 %kdma.source1878, 8, !dbg !14
  %kdma.source1880 = and i64 %kdma.source, 91, !dbg !14
  %kdma.source1881 = lshr i64 %kdma.source, 2, !dbg !14
  %kdma.source1882 = and i64 %kdma.source1881, 4, !dbg !14
  %kdma.source1883 = add nuw nsw i64 %kdma.source1882, %kdma.source, !dbg !14
  %kdma.source1884 = and i64 %kdma.source1883, 4, !dbg !14
  %kdma.source1885 = or disjoint i64 %kdma.source1880, %kdma.source1884, !dbg !14
  %kdma.source1886 = shl nuw nsw i64 %kdma.source1885, 7, !dbg !14
  %kdma.source1887 = or disjoint i64 %kdma.source1879, %kdma.source1886, !dbg !14
  %kdma.source1888 = add nuw nsw i64 %126, %kdma.source1887, !dbg !14
  %kdma.source1889 = trunc i64 %kdma.source1888 to i32, !dbg !14
  %kdma.source1890 = shl i32 %kdma.source1889, 1, !dbg !14
  %kdma.offset = add i32 %kdma.source1890, %kdma.intra, !dbg !14
  %129 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1702, i32 %kdma.offset, <4 x i32> %dma.buffer.rsrc1701) #10, !dbg !14
  %kdma.source1891 = zext nneg i32 %kdma.owner.hi to i64, !dbg !14
  %kdma.source1892 = lshr i64 %kdma.source1891, 4, !dbg !14
  %kdma.source1893 = and i64 %kdma.source1892, 8, !dbg !14
  %kdma.source1894 = and i64 %kdma.source1891, 123, !dbg !14
  %kdma.source1895 = lshr i64 %kdma.source1891, 2, !dbg !14
  %kdma.source1896 = and i64 %kdma.source1895, 4, !dbg !14
  %kdma.source1897 = add nuw nsw i64 %kdma.source1896, %kdma.source1891, !dbg !14
  %kdma.source1898 = and i64 %kdma.source1897, 4, !dbg !14
  %kdma.source1899 = or disjoint i64 %kdma.source1894, %kdma.source1898, !dbg !14
  %kdma.source1900 = shl nuw nsw i64 %kdma.source1899, 7, !dbg !14
  %kdma.source1901 = or disjoint i64 %kdma.source1893, %kdma.source1900, !dbg !14
  %kdma.source1902 = add i64 %126, %kdma.source1901, !dbg !14
  %kdma.source1903 = trunc i64 %kdma.source1902 to i32, !dbg !14
  %kdma.source1904 = shl i32 %kdma.source1903, 1, !dbg !14
  %kdma.offset1905 = add i32 %kdma.source1904, %kdma.intra.hi, !dbg !14
  %kdma.lds1906 = add i32 %dma.buffer.lds1702, 512, !dbg !14
  %130 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1906, i32 %kdma.offset1905, <4 x i32> %dma.buffer.rsrc1701) #10, !dbg !14
  %address.local584 = or disjoint i64 %address.local567, 256, !dbg !8
  %address.local585 = shl nuw nsw i64 %address.local584, 3, !dbg !8
  %address.local586 = and i64 %address.local585, 3584, !dbg !8
  %address.local587 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local586, !dbg !8
  %address.local588 = ptrtoint ptr addrspace(3) %address.local587 to i64, !dbg !8
  %address.local589 = trunc i64 %address.local588 to i32, !dbg !8
  %131 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local589), !dbg !14
  %dma.buffer.lds1699 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %131), !dbg !14
  %kdma.source1918 = or disjoint i64 %kdma.source1887, 16, !dbg !14
  %kdma.source1919 = add nuw nsw i64 %126, %kdma.source1918, !dbg !14
  %kdma.source1920 = trunc i64 %kdma.source1919 to i32, !dbg !14
  %kdma.source1921 = shl i32 %kdma.source1920, 1, !dbg !14
  %kdma.offset1922 = add i32 %kdma.source1921, %kdma.intra, !dbg !14
  %132 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1699, i32 %kdma.offset1922, <4 x i32> %dma.buffer.rsrc1701) #10, !dbg !14
  %kdma.source1935 = or disjoint i64 %kdma.source1901, 16, !dbg !14
  %kdma.source1936 = add i64 %126, %kdma.source1935, !dbg !14
  %kdma.source1937 = trunc i64 %kdma.source1936 to i32, !dbg !14
  %kdma.source1938 = shl i32 %kdma.source1937, 1, !dbg !14
  %kdma.offset1939 = add i32 %kdma.source1938, %kdma.intra.hi, !dbg !14
  %kdma.lds1940 = add i32 %dma.buffer.lds1699, 512, !dbg !14
  %133 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1940, i32 %kdma.offset1939, <4 x i32> %dma.buffer.rsrc1701) #10, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %134 = add nuw i64 %106, 1024, !dbg !14
  %135 = icmp slt i64 %134, 0, !dbg !14
  %136 = sub nuw i64 -1025, %106, !dbg !14
  %137 = select i1 %135, i64 %136, i64 %134, !dbg !14
  %138 = sdiv i64 %137, 4096, !dbg !14
  %.lobit38 = ashr i64 %134, 63, !dbg !14
  %139 = xor i64 %138, %.lobit38, !dbg !14
  %140 = add nsw i64 %139, %17, !dbg !14
  %141 = add nsw i64 %116, %90, !dbg !14
  %142 = add nsw i64 %141, 32, !dbg !14
  %143 = icmp slt i64 %141, -32, !dbg !14
  %144 = sub nsw i64 -33, %141, !dbg !14
  %145 = select i1 %143, i64 %144, i64 %142, !dbg !14
  %146 = sdiv i64 %145, 128, !dbg !14
  %147 = sext i1 %143 to i64, !dbg !14
  %148 = xor i64 %146, %147, !dbg !14
  %149 = shl nsw i64 %148, 7, !dbg !14
  %150 = sub i64 %90, %149, !dbg !14
  %151 = shl i64 %140, 19, !dbg !14
  %152 = and i64 %151, 16252928, !dbg !14
  %address.local601 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local569, !dbg !8
  %address.local602 = ptrtoint ptr addrspace(3) %address.local601 to i64, !dbg !8
  %address.local603 = trunc i64 %address.local602 to i32, !dbg !8
  %153 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local603), !dbg !14
  %.idx1745 = add nuw nsw i64 %152, %116, !dbg !14
  %154 = add i64 %.idx1745, %150, !dbg !14
  %155 = add i64 %154, %123, !dbg !14
  %dma.buffer.lds1696 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %153), !dbg !14
  %kdma.source1952 = add i64 %155, %kdma.source1887, !dbg !14
  %kdma.source1953 = trunc i64 %kdma.source1952 to i32, !dbg !14
  %kdma.source1954 = shl i32 %kdma.source1953, 1, !dbg !14
  %kdma.source1955 = add i32 %kdma.source1954, 64, !dbg !14
  %kdma.offset1956 = add i32 %kdma.source1955, %kdma.intra, !dbg !14
  %156 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1696, i32 %kdma.offset1956, <4 x i32> %dma.buffer.rsrc1701) #10, !dbg !14
  %kdma.source1969 = add i64 %155, %kdma.source1901, !dbg !14
  %kdma.source1970 = trunc i64 %kdma.source1969 to i32, !dbg !14
  %kdma.source1971 = shl i32 %kdma.source1970, 1, !dbg !14
  %kdma.source1972 = add i32 %kdma.source1971, 64, !dbg !14
  %kdma.offset1973 = add i32 %kdma.source1972, %kdma.intra.hi, !dbg !14
  %kdma.lds1974 = add i32 %dma.buffer.lds1696, 512, !dbg !14
  %157 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1974, i32 %kdma.offset1973, <4 x i32> %dma.buffer.rsrc1701) #10, !dbg !14
  %address.local605 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local586, !dbg !8
  %address.local606 = ptrtoint ptr addrspace(3) %address.local605 to i64, !dbg !8
  %address.local607 = trunc i64 %address.local606 to i32, !dbg !8
  %158 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local607), !dbg !14
  %dma.buffer.lds1693 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %158), !dbg !14
  %kdma.source1987 = add i64 %155, %kdma.source1918, !dbg !14
  %kdma.source1988 = trunc i64 %kdma.source1987 to i32, !dbg !14
  %kdma.source1989 = shl i32 %kdma.source1988, 1, !dbg !14
  %kdma.source1990 = add i32 %kdma.source1989, 64, !dbg !14
  %kdma.offset1991 = add i32 %kdma.source1990, %kdma.intra, !dbg !14
  %159 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1693, i32 %kdma.offset1991, <4 x i32> %dma.buffer.rsrc1701) #10, !dbg !14
  %kdma.source2005 = add i64 %155, %kdma.source1935, !dbg !14
  %kdma.source2006 = trunc i64 %kdma.source2005 to i32, !dbg !14
  %kdma.source2007 = shl i32 %kdma.source2006, 1, !dbg !14
  %kdma.source2008 = add i32 %kdma.source2007, 64, !dbg !14
  %kdma.offset2009 = add i32 %kdma.source2008, %kdma.intra.hi, !dbg !14
  %kdma.lds2010 = add i32 %dma.buffer.lds1693, 512, !dbg !14
  %160 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds2010, i32 %kdma.offset2009, <4 x i32> %dma.buffer.rsrc1701) #10, !dbg !14
  %address.local610 = lshr i64 %address.local567, 2, !dbg !15
  %address.local613 = shl nuw nsw i64 %address.local567, 1, !dbg !15
  %address.local614 = and i64 %address.local613, 24, !dbg !15
  %address.local616 = shl i64 %address.local567, 62, !dbg !15
  %address.local617 = ashr exact i64 %address.local616, 57, !dbg !15
  %address.local618 = and i64 %address.local617, -64, !dbg !15
  %address.local621 = and i64 %address.local610, 12, !dbg !11
  %address.local623 = and i64 %address.local567, 63, !dbg !11
  %address.local624 = lshr i64 %address.local623, 5, !dbg !11
  %address.local625 = mul nuw nsw i64 %address.local624, 1016, !dbg !11
  %.idx = mul nuw nsw i64 %address.local567, 320, !dbg !15
  %161 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %.idx, !dbg !15
  %.idx1752 = mul nsw i64 %address.local610, -1280, !dbg !15
  %162 = getelementptr i8, ptr addrspace(3) %161, i64 %.idx1752, !dbg !15
  %163 = getelementptr [2 x i8], ptr addrspace(3) %162, i64 %address.local614, !dbg !15
  %164 = getelementptr [2 x i8], ptr addrspace(3) %163, i64 %address.local618, !dbg !15
  %165 = getelementptr [2 x i8], ptr addrspace(3) %164, i64 %address.local621, !dbg !15
  %address.local627 = getelementptr [2 x i8], ptr addrspace(3) %165, i64 %address.local625, !dbg !15
  %166 = add nsw i64 %.idx, %.idx1752, !dbg !15
  %.idx2277 = shl nuw nsw i64 %address.local614, 1, !dbg !15
  %167 = or disjoint i64 %166, %.idx2277, !dbg !15
  %168 = or disjoint i64 %address.local618, %address.local621, !dbg !15
  %169 = shl nsw i64 %168, 1, !dbg !15
  %170 = add nsw i64 %169, %167, !dbg !15
  %address.local627.idx = mul nuw nsw i64 %address.local624, 2032, !dbg !15
  %171 = add nsw i64 %170, %address.local627.idx, !dbg !15
  %172 = lshr i64 %171, 6, !dbg !15
  %173 = and i64 %172, 8, !dbg !15
  %174 = xor i64 %171, %173, !dbg !15
  %175 = sub i64 %174, %171, !dbg !15
  %kdma.swizzled = getelementptr i8, ptr addrspace(3) %address.local627, i64 %175, !dbg !15
  %176 = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8, !dbg !15
  %177 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %47, <4 x half> %176, <4 x float> zeroinitializer) #9, !dbg !15
  %kdma.gep = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4096, !dbg !15
  %178 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8, !dbg !15
  %address.local638 = shl nuw nsw i32 %address.thread566, 1, !dbg !15
  %address.local639 = and i32 %address.local638, 24, !dbg !15
  %address.local640 = shl nuw nsw i32 %address.thread566, 8, !dbg !15
  %address.local641 = and i32 %address.local640, 768, !dbg !15
  %address.local642 = or disjoint i32 %address.local639, %address.local641, !dbg !15
  %address.local643 = shl nuw nsw i32 %address.thread566, 5, !dbg !15
  %address.local644 = and i32 %address.local643, 96, !dbg !15
  %address.local645 = or disjoint i32 %address.local642, %address.local644, !dbg !15
  %address.local646 = zext nneg i32 %address.local645 to i64, !dbg !15
  %address.base = call i64 asm sideeffect "", "=v,0"(i64 %96) #9, !dbg !15, !frisk.register_lifetimes !1
  %address.local647 = lshr i32 %address.thread566, 1, !dbg !15
  %address.local648 = and i32 %address.local647, 1, !dbg !15
  %address.local649 = zext nneg i32 %address.local648 to i64, !dbg !15
  %address.local650 = add nsw i64 %address.base, %address.local649, !dbg !15
  %address.local651 = shl i64 %address.local650, 6, !dbg !15
  %address.local652 = sub i64 %address.local646, %address.local651, !dbg !15
  %179 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local652, !dbg !15
  %180 = getelementptr [2 x i8], ptr addrspace(3) %179, i64 %address.local621, !dbg !15
  %181 = getelementptr [2 x i8], ptr addrspace(3) %180, i64 %address.local625, !dbg !15
  %address.local656 = getelementptr i8, ptr addrspace(3) %181, i64 64, !dbg !15
  %.idx22802283 = add i64 %address.local652, %address.local621, !dbg !15
  %182 = shl i64 %.idx22802283, 1, !dbg !15
  %183 = add i64 %182, %address.local627.idx, !dbg !15
  %184 = add i64 %183, 64, !dbg !15
  %185 = lshr i64 %184, 6, !dbg !15
  %186 = and i64 %185, 8, !dbg !15
  %187 = xor i64 %184, %186, !dbg !15
  %188 = sub i64 %187, %184, !dbg !15
  %kdma.swizzled2151 = getelementptr i8, ptr addrspace(3) %address.local656, i64 %188, !dbg !15
  %189 = load <4 x half>, ptr addrspace(3) %kdma.swizzled2151, align 8, !dbg !15, !frisk.k_lds_pair !41
  %kdma.gep2155 = getelementptr i8, ptr addrspace(3) %kdma.swizzled2151, i64 128, !dbg !15
  %190 = load <4 x half>, ptr addrspace(3) %kdma.gep2155, align 8, !dbg !15, !frisk.k_lds_pair !41
  %kdma.gep2152 = getelementptr i8, ptr addrspace(3) %kdma.swizzled2151, i64 4096, !dbg !15
  %191 = load <4 x half>, ptr addrspace(3) %kdma.gep2152, align 8, !dbg !15, !frisk.k_lds_pair !42
  %kdma.gep2156 = getelementptr i8, ptr addrspace(3) %kdma.swizzled2151, i64 4224, !dbg !15
  %192 = load <4 x half>, ptr addrspace(3) %kdma.gep2156, align 8, !dbg !15, !frisk.k_lds_pair !42
  %kdma.gep2153 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 128, !dbg !15
  %193 = load <4 x half>, ptr addrspace(3) %kdma.gep2153, align 8, !dbg !15, !frisk.k_lds_pair !43
  %kdma.gep2157 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1024, !dbg !15
  %194 = load <4 x half>, ptr addrspace(3) %kdma.gep2157, align 8, !dbg !15, !frisk.k_lds_pair !43
  %kdma.gep2154 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4224, !dbg !15
  %195 = load <4 x half>, ptr addrspace(3) %kdma.gep2154, align 8, !dbg !15, !frisk.k_lds_pair !44
  %kdma.gep2158 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5120, !dbg !15
  %196 = load <4 x half>, ptr addrspace(3) %kdma.gep2158, align 8, !dbg !15, !frisk.k_lds_pair !44
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %197 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %49, <4 x half> %178, <4 x float> %177) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %kdma.gep2159 = getelementptr i8, ptr addrspace(3) %kdma.swizzled2151, i64 1024, !dbg !15
  %198 = load <4 x half>, ptr addrspace(3) %kdma.gep2159, align 8, !dbg !15, !frisk.k_lds_pair !39
  %kdma.gep2163 = getelementptr i8, ptr addrspace(3) %kdma.swizzled2151, i64 1152, !dbg !15
  %199 = load <4 x half>, ptr addrspace(3) %kdma.gep2163, align 8, !dbg !15, !frisk.k_lds_pair !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %200 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %47, <4 x half> %189, <4 x float> zeroinitializer) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %kdma.gep2160 = getelementptr i8, ptr addrspace(3) %kdma.swizzled2151, i64 5120, !dbg !15
  %201 = load <4 x half>, ptr addrspace(3) %kdma.gep2160, align 8, !dbg !15, !frisk.k_lds_pair !45
  %kdma.gep2164 = getelementptr i8, ptr addrspace(3) %kdma.swizzled2151, i64 5248, !dbg !15
  %202 = load <4 x half>, ptr addrspace(3) %kdma.gep2164, align 8, !dbg !15, !frisk.k_lds_pair !45
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %203 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %49, <4 x half> %191, <4 x float> %200) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %kdma.gep2161 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1152, !dbg !15
  %204 = load <4 x half>, ptr addrspace(3) %kdma.gep2161, align 8, !dbg !15, !frisk.k_lds_pair !46
  %kdma.gep2162 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5248, !dbg !15
  %205 = load <4 x half>, ptr addrspace(3) %kdma.gep2162, align 8, !dbg !15, !frisk.k_lds_pair !46
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %206 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %47, <4 x half> %193, <4 x float> zeroinitializer) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %207 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %49, <4 x half> %195, <4 x float> %206) #9, !dbg !15
  %208 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %47, <4 x half> %190, <4 x float> zeroinitializer) #9, !dbg !15
  %209 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %49, <4 x half> %192, <4 x float> %208) #9, !dbg !15
  %210 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %47, <4 x half> %194, <4 x float> zeroinitializer) #9, !dbg !15
  %211 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %49, <4 x half> %196, <4 x float> %210) #9, !dbg !15
  %212 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %47, <4 x half> %198, <4 x float> zeroinitializer) #9, !dbg !15
  %213 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %49, <4 x half> %201, <4 x float> %212) #9, !dbg !15
  %214 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %47, <4 x half> %204, <4 x float> zeroinitializer) #9, !dbg !15
  %215 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %49, <4 x half> %205, <4 x float> %214) #9, !dbg !15
  %216 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %47, <4 x half> %199, <4 x float> zeroinitializer) #9, !dbg !15
  %217 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %49, <4 x half> %202, <4 x float> %216) #9, !dbg !15
  %218 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %51, <4 x half> %176, <4 x float> zeroinitializer) #9, !dbg !15
  %219 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %52, <4 x half> %178, <4 x float> %218) #9, !dbg !15
  %220 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %51, <4 x half> %189, <4 x float> zeroinitializer) #9, !dbg !15
  %221 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %52, <4 x half> %191, <4 x float> %220) #9, !dbg !15
  %222 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %51, <4 x half> %193, <4 x float> zeroinitializer) #9, !dbg !15
  %223 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %52, <4 x half> %195, <4 x float> %222) #9, !dbg !15
  %224 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %51, <4 x half> %190, <4 x float> zeroinitializer) #9, !dbg !15
  %225 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %52, <4 x half> %192, <4 x float> %224) #9, !dbg !15
  %226 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %51, <4 x half> %194, <4 x float> zeroinitializer) #9, !dbg !15
  %227 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %52, <4 x half> %196, <4 x float> %226) #9, !dbg !15
  %228 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %51, <4 x half> %198, <4 x float> zeroinitializer) #9, !dbg !15
  %229 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %52, <4 x half> %201, <4 x float> %228) #9, !dbg !15
  %230 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %51, <4 x half> %204, <4 x float> zeroinitializer) #9, !dbg !15
  %231 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %52, <4 x half> %205, <4 x float> %230) #9, !dbg !15
  %232 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %51, <4 x half> %199, <4 x float> zeroinitializer) #9, !dbg !15
  %233 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %52, <4 x half> %202, <4 x float> %232) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %234 = add nuw i64 %106, 2048, !dbg !14
  %235 = icmp slt i64 %234, 0, !dbg !14
  %236 = sub nuw i64 -2049, %106, !dbg !14
  %237 = select i1 %235, i64 %236, i64 %234, !dbg !14
  %238 = sdiv i64 %237, 4096, !dbg !14
  %.lobit39 = ashr i64 %234, 63, !dbg !14
  %239 = xor i64 %238, %.lobit39, !dbg !14
  %240 = add nsw i64 %239, %17, !dbg !14
  %241 = add nsw i64 %141, 64, !dbg !14
  %242 = icmp slt i64 %141, -64, !dbg !14
  %243 = sub nsw i64 -65, %141, !dbg !14
  %244 = select i1 %242, i64 %243, i64 %241, !dbg !14
  %245 = sdiv i64 %244, 128, !dbg !14
  %246 = sext i1 %242 to i64, !dbg !14
  %247 = xor i64 %245, %246, !dbg !14
  %248 = shl nsw i64 %247, 7, !dbg !14
  %249 = sub i64 %90, %248, !dbg !14
  %250 = shl i64 %240, 19, !dbg !14
  %251 = and i64 %250, 16252928, !dbg !14
  %.idx17531757 = add nuw nsw i64 %251, %116, !dbg !14
  %252 = add i64 %.idx17531757, %249, !dbg !14
  %253 = add i64 %252, %123, !dbg !14
  %kdma.source2022 = add i64 %253, %kdma.source1887, !dbg !14
  %kdma.source2023 = trunc i64 %kdma.source2022 to i32, !dbg !14
  %kdma.source2024 = shl i32 %kdma.source2023, 1, !dbg !14
  %kdma.source2025 = add i32 %kdma.source2024, 128, !dbg !14
  %kdma.offset2026 = add i32 %kdma.source2025, %kdma.intra, !dbg !14
  %254 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1702, i32 %kdma.offset2026, <4 x i32> %dma.buffer.rsrc1701) #10, !dbg !14
  %kdma.source2039 = add i64 %253, %kdma.source1901, !dbg !14
  %kdma.source2040 = trunc i64 %kdma.source2039 to i32, !dbg !14
  %kdma.source2041 = shl i32 %kdma.source2040, 1, !dbg !14
  %kdma.source2042 = add i32 %kdma.source2041, 128, !dbg !14
  %kdma.offset2043 = add i32 %kdma.source2042, %kdma.intra.hi, !dbg !14
  %255 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1906, i32 %kdma.offset2043, <4 x i32> %dma.buffer.rsrc1701) #10, !dbg !14
  %kdma.source2057 = add i64 %253, %kdma.source1918, !dbg !14
  %kdma.source2058 = trunc i64 %kdma.source2057 to i32, !dbg !14
  %kdma.source2059 = shl i32 %kdma.source2058, 1, !dbg !14
  %kdma.source2060 = add i32 %kdma.source2059, 128, !dbg !14
  %kdma.offset2061 = add i32 %kdma.source2060, %kdma.intra, !dbg !14
  %256 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1699, i32 %kdma.offset2061, <4 x i32> %dma.buffer.rsrc1701) #10, !dbg !14
  %kdma.source2075 = add i64 %253, %kdma.source1935, !dbg !14
  %kdma.source2076 = trunc i64 %kdma.source2075 to i32, !dbg !14
  %kdma.source2077 = shl i32 %kdma.source2076, 1, !dbg !14
  %kdma.source2078 = add i32 %kdma.source2077, 128, !dbg !14
  %kdma.offset2079 = add i32 %kdma.source2078, %kdma.intra.hi, !dbg !14
  %257 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1940, i32 %kdma.offset2079, <4 x i32> %dma.buffer.rsrc1701) #10, !dbg !14
  %kdma.gep2181 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8192, !dbg !15
  %258 = load <4 x half>, ptr addrspace(3) %kdma.gep2181, align 8, !dbg !15, !frisk.k_lds_pair !47
  %kdma.gep2185 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8320, !dbg !15
  %259 = load <4 x half>, ptr addrspace(3) %kdma.gep2185, align 8, !dbg !15, !frisk.k_lds_pair !47
  %kdma.gep2182 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12288, !dbg !15
  %260 = load <4 x half>, ptr addrspace(3) %kdma.gep2182, align 8, !dbg !15, !frisk.k_lds_pair !48
  %kdma.gep2186 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12416, !dbg !15
  %261 = load <4 x half>, ptr addrspace(3) %kdma.gep2186, align 8, !dbg !15, !frisk.k_lds_pair !48
  %kdma.gep2183 = getelementptr i8, ptr addrspace(3) %kdma.swizzled2151, i64 8192, !dbg !15
  %262 = load <4 x half>, ptr addrspace(3) %kdma.gep2183, align 8, !dbg !15, !frisk.k_lds_pair !49
  %kdma.gep2187 = getelementptr i8, ptr addrspace(3) %kdma.swizzled2151, i64 8320, !dbg !15
  %263 = load <4 x half>, ptr addrspace(3) %kdma.gep2187, align 8, !dbg !15, !frisk.k_lds_pair !49
  %kdma.gep2184 = getelementptr i8, ptr addrspace(3) %kdma.swizzled2151, i64 12288, !dbg !15
  %264 = load <4 x half>, ptr addrspace(3) %kdma.gep2184, align 8, !dbg !15, !frisk.k_lds_pair !50
  %kdma.gep2188 = getelementptr i8, ptr addrspace(3) %kdma.swizzled2151, i64 12416, !dbg !15
  %265 = load <4 x half>, ptr addrspace(3) %kdma.gep2188, align 8, !dbg !15, !frisk.k_lds_pair !50
  %kdma.gep2189 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9216, !dbg !15
  %266 = load <4 x half>, ptr addrspace(3) %kdma.gep2189, align 8, !dbg !15, !frisk.k_lds_pair !51
  %kdma.gep2193 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9344, !dbg !15
  %267 = load <4 x half>, ptr addrspace(3) %kdma.gep2193, align 8, !dbg !15, !frisk.k_lds_pair !51
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %268 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %258, <4 x float> %197) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %kdma.gep2190 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13312, !dbg !15
  %269 = load <4 x half>, ptr addrspace(3) %kdma.gep2190, align 8, !dbg !15, !frisk.k_lds_pair !52
  %kdma.gep2194 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13440, !dbg !15
  %270 = load <4 x half>, ptr addrspace(3) %kdma.gep2194, align 8, !dbg !15, !frisk.k_lds_pair !52
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %271 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %63, <4 x half> %260, <4 x float> %268) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %kdma.gep2191 = getelementptr i8, ptr addrspace(3) %kdma.swizzled2151, i64 9216, !dbg !15
  %272 = load <4 x half>, ptr addrspace(3) %kdma.gep2191, align 8, !dbg !15, !frisk.k_lds_pair !53
  %kdma.gep2195 = getelementptr i8, ptr addrspace(3) %kdma.swizzled2151, i64 9344, !dbg !15
  %273 = load <4 x half>, ptr addrspace(3) %kdma.gep2195, align 8, !dbg !15, !frisk.k_lds_pair !53
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %274 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %262, <4 x float> %203) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %kdma.gep2192 = getelementptr i8, ptr addrspace(3) %kdma.swizzled2151, i64 13312, !dbg !15
  %275 = load <4 x half>, ptr addrspace(3) %kdma.gep2192, align 8, !dbg !15, !frisk.k_lds_pair !54
  %kdma.gep2196 = getelementptr i8, ptr addrspace(3) %kdma.swizzled2151, i64 13440, !dbg !15
  %276 = load <4 x half>, ptr addrspace(3) %kdma.gep2196, align 8, !dbg !15, !frisk.k_lds_pair !54
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %277 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %63, <4 x half> %264, <4 x float> %274) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %278 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %259, <4 x float> %207) #9, !dbg !15
  %279 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %63, <4 x half> %261, <4 x float> %278) #9, !dbg !15
  %280 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %263, <4 x float> %209) #9, !dbg !15
  %281 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %63, <4 x half> %265, <4 x float> %280) #9, !dbg !15
  %282 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %266, <4 x float> %211) #9, !dbg !15
  %283 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %63, <4 x half> %269, <4 x float> %282) #9, !dbg !15
  %284 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %272, <4 x float> %213) #9, !dbg !15
  %285 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %63, <4 x half> %275, <4 x float> %284) #9, !dbg !15
  %286 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %267, <4 x float> %215) #9, !dbg !15
  %287 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %63, <4 x half> %270, <4 x float> %286) #9, !dbg !15
  %288 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %273, <4 x float> %217) #9, !dbg !15
  %289 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %63, <4 x half> %276, <4 x float> %288) #9, !dbg !15
  %290 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %258, <4 x float> %219) #9, !dbg !15
  %291 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %260, <4 x float> %290) #9, !dbg !15
  %292 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %262, <4 x float> %221) #9, !dbg !15
  %293 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %264, <4 x float> %292) #9, !dbg !15
  %294 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %259, <4 x float> %223) #9, !dbg !15
  %295 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %261, <4 x float> %294) #9, !dbg !15
  %296 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %263, <4 x float> %225) #9, !dbg !15
  %297 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %265, <4 x float> %296) #9, !dbg !15
  %298 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %266, <4 x float> %227) #9, !dbg !15
  %299 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %269, <4 x float> %298) #9, !dbg !15
  %300 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %272, <4 x float> %229) #9, !dbg !15
  %301 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %275, <4 x float> %300) #9, !dbg !15
  %302 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %267, <4 x float> %231) #9, !dbg !15
  %303 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %270, <4 x float> %302) #9, !dbg !15
  %304 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %273, <4 x float> %233) #9, !dbg !15
  %305 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %276, <4 x float> %304) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %306 = add nuw i64 %106, 3072, !dbg !14
  %307 = icmp slt i64 %306, 0, !dbg !14
  %308 = sub nuw i64 -3073, %106, !dbg !14
  %309 = select i1 %307, i64 %308, i64 %306, !dbg !14
  %310 = sdiv i64 %309, 4096, !dbg !14
  %.lobit40 = ashr i64 %306, 63, !dbg !14
  %311 = xor i64 %310, %.lobit40, !dbg !14
  %312 = add nsw i64 %311, %17, !dbg !14
  %313 = add nsw i64 %141, 96, !dbg !14
  %314 = icmp slt i64 %141, -96, !dbg !14
  %315 = sub nsw i64 -97, %141, !dbg !14
  %316 = select i1 %314, i64 %315, i64 %313, !dbg !14
  %317 = sdiv i64 %316, 128, !dbg !14
  %318 = sext i1 %314 to i64, !dbg !14
  %319 = xor i64 %317, %318, !dbg !14
  %320 = shl nsw i64 %319, 7, !dbg !14
  %321 = sub i64 %90, %320, !dbg !14
  %322 = shl i64 %312, 19, !dbg !14
  %323 = and i64 %322, 16252928, !dbg !14
  %.idx17651769 = add nuw nsw i64 %323, %116, !dbg !14
  %324 = add i64 %.idx17651769, %321, !dbg !14
  %325 = add i64 %324, %123, !dbg !14
  %kdma.source2092 = add i64 %325, %kdma.source1887, !dbg !14
  %kdma.source2093 = trunc i64 %kdma.source2092 to i32, !dbg !14
  %kdma.source2094 = shl i32 %kdma.source2093, 1, !dbg !14
  %kdma.source2095 = add i32 %kdma.source2094, 192, !dbg !14
  %kdma.offset2096 = add i32 %kdma.source2095, %kdma.intra, !dbg !14
  %326 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1696, i32 %kdma.offset2096, <4 x i32> %dma.buffer.rsrc1701) #10, !dbg !14
  %kdma.source2109 = add i64 %325, %kdma.source1901, !dbg !14
  %kdma.source2110 = trunc i64 %kdma.source2109 to i32, !dbg !14
  %kdma.source2111 = shl i32 %kdma.source2110, 1, !dbg !14
  %kdma.source2112 = add i32 %kdma.source2111, 192, !dbg !14
  %kdma.offset2113 = add i32 %kdma.source2112, %kdma.intra.hi, !dbg !14
  %327 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1974, i32 %kdma.offset2113, <4 x i32> %dma.buffer.rsrc1701) #10, !dbg !14
  %kdma.source2127 = add i64 %325, %kdma.source1918, !dbg !14
  %kdma.source2128 = trunc i64 %kdma.source2127 to i32, !dbg !14
  %kdma.source2129 = shl i32 %kdma.source2128, 1, !dbg !14
  %kdma.source2130 = add i32 %kdma.source2129, 192, !dbg !14
  %kdma.offset2131 = add i32 %kdma.source2130, %kdma.intra, !dbg !14
  %328 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1693, i32 %kdma.offset2131, <4 x i32> %dma.buffer.rsrc1701) #10, !dbg !14
  %kdma.source2145 = add i64 %325, %kdma.source1935, !dbg !14
  %kdma.source2146 = trunc i64 %kdma.source2145 to i32, !dbg !14
  %kdma.source2147 = shl i32 %kdma.source2146, 1, !dbg !14
  %kdma.source2148 = add i32 %kdma.source2147, 192, !dbg !14
  %kdma.offset2149 = add i32 %kdma.source2148, %kdma.intra.hi, !dbg !14
  %329 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds2010, i32 %kdma.offset2149, <4 x i32> %dma.buffer.rsrc1701) #10, !dbg !14
  %330 = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8, !dbg !15, !frisk.k_lds_pair !55
  %331 = load <4 x half>, ptr addrspace(3) %kdma.gep2153, align 8, !dbg !15, !frisk.k_lds_pair !55
  %332 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8, !dbg !15, !frisk.k_lds_pair !56
  %333 = load <4 x half>, ptr addrspace(3) %kdma.gep2154, align 8, !dbg !15, !frisk.k_lds_pair !56
  %334 = load <4 x half>, ptr addrspace(3) %kdma.swizzled2151, align 8, !dbg !15, !frisk.k_lds_pair !57
  %335 = load <4 x half>, ptr addrspace(3) %kdma.gep2155, align 8, !dbg !15, !frisk.k_lds_pair !57
  %336 = load <4 x half>, ptr addrspace(3) %kdma.gep2152, align 8, !dbg !15, !frisk.k_lds_pair !58
  %337 = load <4 x half>, ptr addrspace(3) %kdma.gep2156, align 8, !dbg !15, !frisk.k_lds_pair !58
  %338 = load <4 x half>, ptr addrspace(3) %kdma.gep2157, align 8, !dbg !15, !frisk.k_lds_pair !59
  %339 = load <4 x half>, ptr addrspace(3) %kdma.gep2161, align 8, !dbg !15, !frisk.k_lds_pair !59
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %340 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %330, <4 x float> %271) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %341 = load <4 x half>, ptr addrspace(3) %kdma.gep2158, align 8, !dbg !15, !frisk.k_lds_pair !60
  %342 = load <4 x half>, ptr addrspace(3) %kdma.gep2162, align 8, !dbg !15, !frisk.k_lds_pair !60
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %343 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %76, <4 x half> %332, <4 x float> %340) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %344 = load <4 x half>, ptr addrspace(3) %kdma.gep2159, align 8, !dbg !15, !frisk.k_lds_pair !61
  %345 = load <4 x half>, ptr addrspace(3) %kdma.gep2163, align 8, !dbg !15, !frisk.k_lds_pair !61
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %346 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %334, <4 x float> %277) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %347 = load <4 x half>, ptr addrspace(3) %kdma.gep2160, align 8, !dbg !15, !frisk.k_lds_pair !62
  %348 = load <4 x half>, ptr addrspace(3) %kdma.gep2164, align 8, !dbg !15, !frisk.k_lds_pair !62
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %349 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %76, <4 x half> %336, <4 x float> %346) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %350 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %331, <4 x float> %279) #9, !dbg !15
  %351 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %76, <4 x half> %333, <4 x float> %350) #9, !dbg !15
  %352 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %335, <4 x float> %281) #9, !dbg !15
  %353 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %76, <4 x half> %337, <4 x float> %352) #9, !dbg !15
  %354 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %338, <4 x float> %283) #9, !dbg !15
  %355 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %76, <4 x half> %341, <4 x float> %354) #9, !dbg !15
  %356 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %344, <4 x float> %285) #9, !dbg !15
  %357 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %76, <4 x half> %347, <4 x float> %356) #9, !dbg !15
  %358 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %339, <4 x float> %287) #9, !dbg !15
  %359 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %76, <4 x half> %342, <4 x float> %358) #9, !dbg !15
  %360 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %345, <4 x float> %289) #9, !dbg !15
  %361 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %76, <4 x half> %348, <4 x float> %360) #9, !dbg !15
  %362 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %330, <4 x float> %291) #9, !dbg !15
  %363 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %332, <4 x float> %362) #9, !dbg !15
  %364 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %334, <4 x float> %293) #9, !dbg !15
  %365 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %336, <4 x float> %364) #9, !dbg !15
  %366 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %331, <4 x float> %295) #9, !dbg !15
  %367 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %333, <4 x float> %366) #9, !dbg !15
  %368 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %335, <4 x float> %297) #9, !dbg !15
  %369 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %337, <4 x float> %368) #9, !dbg !15
  %370 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %338, <4 x float> %299) #9, !dbg !15
  %371 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %341, <4 x float> %370) #9, !dbg !15
  %372 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %344, <4 x float> %301) #9, !dbg !15
  %373 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %347, <4 x float> %372) #9, !dbg !15
  %374 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %339, <4 x float> %303) #9, !dbg !15
  %375 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %342, <4 x float> %374) #9, !dbg !15
  %376 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %345, <4 x float> %305) #9, !dbg !15
  %377 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %348, <4 x float> %376) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %378 = add nuw nsw i64 %116, %17, !dbg !16
  %379 = shl i64 %378, 19, !dbg !16
  %380 = and i64 %379, 16252928, !dbg !16
  %address.local985 = shl nuw nsw i64 %address.local567, 5, !dbg !16
  %address.local986 = and i64 %address.local985, 8064, !dbg !16
  %address.local988 = and i64 %address.local568, 24, !dbg !16
  %address.local989 = or disjoint i64 %address.local986, %address.local988, !dbg !16
  %381 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !16
  %.idx17771779 = add i64 %380, %123, !dbg !16
  %382 = or disjoint i64 %.idx17771779, %address.local989, !dbg !16
  %.tr1780 = trunc i64 %382 to i32, !dbg !16
  %dma.buffer.offset1676 = shl i32 %.tr1780, 1, !dbg !16
  %383 = insertelement <2 x i64> poison, i64 %381, i64 0, !dbg !16
  %384 = bitcast <2 x i64> %383 to <4 x i32>, !dbg !16
  %dma.buffer.rsrc1677 = shufflevector <4 x i32> %384, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !16
  %385 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1702, i32 %dma.buffer.offset1676, <4 x i32> %dma.buffer.rsrc1677) #10, !dbg !16
  %address.local991 = shl nuw nsw i64 %address.local584, 5, !dbg !16
  %address.local992 = and i64 %address.local991, 16256, !dbg !16
  %address.local995 = or disjoint i64 %address.local992, %address.local988, !dbg !16
  %386 = or disjoint i64 %.idx17771779, %address.local995, !dbg !16
  %.tr1784 = trunc i64 %386 to i32, !dbg !16
  %dma.buffer.offset1673 = shl i32 %.tr1784, 1, !dbg !16
  %387 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1699, i32 %dma.buffer.offset1673, <4 x i32> %dma.buffer.rsrc1677) #10, !dbg !16
  %388 = load <4 x half>, ptr addrspace(3) %kdma.gep2181, align 8, !dbg !15
  %389 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %388, <4 x float> %343) #9, !dbg !15
  %390 = or disjoint i64 %118, %102, !dbg !17
  %391 = lshr i64 %103, 32, !dbg !23
  %compare.high = trunc nuw i64 %391 to i32, !dbg !23
  %392 = lshr i64 %118, 32, !dbg !23
  %compare.high1269 = trunc nuw i64 %392 to i32, !dbg !23
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high1269, !dbg !23
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high1269, !dbg !23
  %compare.low.base = trunc i64 %103 to i32, !dbg !23
  %compare.low.base1270 = trunc i64 %390 to i32, !dbg !23
  %compare.low = icmp ugt i32 %compare.low.base, %compare.low.base1270, !dbg !23
  %compare.result = select i1 %compare.high.equal, i1 %compare.low, i1 %compare.high.order, !dbg !23
  %393 = select i1 %compare.result, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 1) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local = or i32 %compare.local.base, 1, !dbg !23
  %compare.low1271 = icmp ult i32 %compare.local, %compare.low.base, !dbg !23
  %compare.result1272 = select i1 %compare.high.equal, i1 %compare.low1271, i1 %compare.high.order, !dbg !23
  %394 = select i1 %compare.result1272, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1273 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 2) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1274 = or i32 %compare.local.base1273, 2, !dbg !23
  %compare.low1275 = icmp ult i32 %compare.local1274, %compare.low.base, !dbg !23
  %compare.result1276 = select i1 %compare.high.equal, i1 %compare.low1275, i1 %compare.high.order, !dbg !23
  %395 = select i1 %compare.result1276, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1277 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 3) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1278 = or i32 %compare.local.base1277, 3, !dbg !23
  %compare.low1279 = icmp ult i32 %compare.local1278, %compare.low.base, !dbg !23
  %compare.result1280 = select i1 %compare.high.equal, i1 %compare.low1279, i1 %compare.high.order, !dbg !23
  %396 = select i1 %compare.result1280, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %397 = insertelement <4 x float> poison, float %393, i64 0, !dbg !25
  %398 = insertelement <4 x float> %397, float %394, i64 1, !dbg !25
  %399 = insertelement <4 x float> %398, float %395, i64 2, !dbg !25
  %400 = insertelement <4 x float> %399, float %396, i64 3, !dbg !25
  %compare.local.base1281 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 4) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1282 = or i32 %compare.local.base1281, 4, !dbg !23
  %compare.low1283 = icmp ult i32 %compare.local1282, %compare.low.base, !dbg !23
  %compare.result1284 = select i1 %compare.high.equal, i1 %compare.low1283, i1 %compare.high.order, !dbg !23
  %401 = select i1 %compare.result1284, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1285 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 5) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1286 = or i32 %compare.local.base1285, 5, !dbg !23
  %compare.low1287 = icmp ult i32 %compare.local1286, %compare.low.base, !dbg !23
  %compare.result1288 = select i1 %compare.high.equal, i1 %compare.low1287, i1 %compare.high.order, !dbg !23
  %402 = select i1 %compare.result1288, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1289 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 6) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1290 = or i32 %compare.local.base1289, 6, !dbg !23
  %compare.low1291 = icmp ult i32 %compare.local1290, %compare.low.base, !dbg !23
  %compare.result1292 = select i1 %compare.high.equal, i1 %compare.low1291, i1 %compare.high.order, !dbg !23
  %403 = select i1 %compare.result1292, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1293 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 7) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1294 = or i32 %compare.local.base1293, 7, !dbg !23
  %compare.low1295 = icmp ult i32 %compare.local1294, %compare.low.base, !dbg !23
  %compare.result1296 = select i1 %compare.high.equal, i1 %compare.low1295, i1 %compare.high.order, !dbg !23
  %404 = select i1 %compare.result1296, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %405 = insertelement <4 x float> poison, float %401, i64 0, !dbg !25
  %406 = insertelement <4 x float> %405, float %402, i64 1, !dbg !25
  %407 = insertelement <4 x float> %406, float %403, i64 2, !dbg !25
  %408 = insertelement <4 x float> %407, float %404, i64 3, !dbg !25
  %compare.local.base1297 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 8) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1298 = or i32 %compare.local.base1297, 8, !dbg !23
  %compare.low1299 = icmp ult i32 %compare.local1298, %compare.low.base, !dbg !23
  %compare.result1300 = select i1 %compare.high.equal, i1 %compare.low1299, i1 %compare.high.order, !dbg !23
  %409 = select i1 %compare.result1300, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1301 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 9) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1302 = or i32 %compare.local.base1301, 9, !dbg !23
  %compare.low1303 = icmp ult i32 %compare.local1302, %compare.low.base, !dbg !23
  %compare.result1304 = select i1 %compare.high.equal, i1 %compare.low1303, i1 %compare.high.order, !dbg !23
  %410 = select i1 %compare.result1304, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1305 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 10) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1306 = or i32 %compare.local.base1305, 10, !dbg !23
  %compare.low1307 = icmp ult i32 %compare.local1306, %compare.low.base, !dbg !23
  %compare.result1308 = select i1 %compare.high.equal, i1 %compare.low1307, i1 %compare.high.order, !dbg !23
  %411 = select i1 %compare.result1308, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1309 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 11) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1310 = or i32 %compare.local.base1309, 11, !dbg !23
  %compare.low1311 = icmp ult i32 %compare.local1310, %compare.low.base, !dbg !23
  %compare.result1312 = select i1 %compare.high.equal, i1 %compare.low1311, i1 %compare.high.order, !dbg !23
  %412 = select i1 %compare.result1312, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %413 = insertelement <4 x float> poison, float %409, i64 0, !dbg !25
  %414 = insertelement <4 x float> %413, float %410, i64 1, !dbg !25
  %415 = insertelement <4 x float> %414, float %411, i64 2, !dbg !25
  %416 = insertelement <4 x float> %415, float %412, i64 3, !dbg !25
  %compare.local.base1313 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 12) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1314 = or i32 %compare.local.base1313, 12, !dbg !23
  %compare.low1315 = icmp ult i32 %compare.local1314, %compare.low.base, !dbg !23
  %compare.result1316 = select i1 %compare.high.equal, i1 %compare.low1315, i1 %compare.high.order, !dbg !23
  %417 = select i1 %compare.result1316, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1317 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 13) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1318 = or i32 %compare.local.base1317, 13, !dbg !23
  %compare.low1319 = icmp ult i32 %compare.local1318, %compare.low.base, !dbg !23
  %compare.result1320 = select i1 %compare.high.equal, i1 %compare.low1319, i1 %compare.high.order, !dbg !23
  %418 = select i1 %compare.result1320, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1321 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 14) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1322 = or i32 %compare.local.base1321, 14, !dbg !23
  %compare.low1323 = icmp ult i32 %compare.local1322, %compare.low.base, !dbg !23
  %compare.result1324 = select i1 %compare.high.equal, i1 %compare.low1323, i1 %compare.high.order, !dbg !23
  %419 = select i1 %compare.result1324, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1325 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 15) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1326 = or i32 %compare.local.base1325, 15, !dbg !23
  %compare.low1327 = icmp ult i32 %compare.local1326, %compare.low.base, !dbg !23
  %compare.result1328 = select i1 %compare.high.equal, i1 %compare.low1327, i1 %compare.high.order, !dbg !23
  %420 = select i1 %compare.result1328, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %421 = insertelement <4 x float> poison, float %417, i64 0, !dbg !25
  %422 = insertelement <4 x float> %421, float %418, i64 1, !dbg !25
  %423 = insertelement <4 x float> %422, float %419, i64 2, !dbg !25
  %424 = insertelement <4 x float> %423, float %420, i64 3, !dbg !25
  %compare.local.base1329 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 16) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1330 = or i32 %compare.local.base1329, 64, !dbg !23
  %compare.low1331 = icmp ult i32 %compare.local1330, %compare.low.base, !dbg !23
  %compare.result1332 = select i1 %compare.high.equal, i1 %compare.low1331, i1 %compare.high.order, !dbg !23
  %425 = select i1 %compare.result1332, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1333 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 17) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1334 = or i32 %compare.local.base1333, 65, !dbg !23
  %compare.low1335 = icmp ult i32 %compare.local1334, %compare.low.base, !dbg !23
  %compare.result1336 = select i1 %compare.high.equal, i1 %compare.low1335, i1 %compare.high.order, !dbg !23
  %426 = select i1 %compare.result1336, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1337 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 18) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1338 = or i32 %compare.local.base1337, 66, !dbg !23
  %compare.low1339 = icmp ult i32 %compare.local1338, %compare.low.base, !dbg !23
  %compare.result1340 = select i1 %compare.high.equal, i1 %compare.low1339, i1 %compare.high.order, !dbg !23
  %427 = select i1 %compare.result1340, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1341 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 19) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1342 = or i32 %compare.local.base1341, 67, !dbg !23
  %compare.low1343 = icmp ult i32 %compare.local1342, %compare.low.base, !dbg !23
  %compare.result1344 = select i1 %compare.high.equal, i1 %compare.low1343, i1 %compare.high.order, !dbg !23
  %428 = select i1 %compare.result1344, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %429 = insertelement <4 x float> poison, float %425, i64 0, !dbg !25
  %430 = insertelement <4 x float> %429, float %426, i64 1, !dbg !25
  %431 = insertelement <4 x float> %430, float %427, i64 2, !dbg !25
  %432 = insertelement <4 x float> %431, float %428, i64 3, !dbg !25
  %compare.local.base1345 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 20) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1346 = or i32 %compare.local.base1345, 68, !dbg !23
  %compare.low1347 = icmp ult i32 %compare.local1346, %compare.low.base, !dbg !23
  %compare.result1348 = select i1 %compare.high.equal, i1 %compare.low1347, i1 %compare.high.order, !dbg !23
  %433 = select i1 %compare.result1348, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1349 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 21) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1350 = or i32 %compare.local.base1349, 69, !dbg !23
  %compare.low1351 = icmp ult i32 %compare.local1350, %compare.low.base, !dbg !23
  %compare.result1352 = select i1 %compare.high.equal, i1 %compare.low1351, i1 %compare.high.order, !dbg !23
  %434 = select i1 %compare.result1352, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1353 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 22) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1354 = or i32 %compare.local.base1353, 70, !dbg !23
  %compare.low1355 = icmp ult i32 %compare.local1354, %compare.low.base, !dbg !23
  %compare.result1356 = select i1 %compare.high.equal, i1 %compare.low1355, i1 %compare.high.order, !dbg !23
  %435 = select i1 %compare.result1356, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1357 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 23) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1358 = or i32 %compare.local.base1357, 71, !dbg !23
  %compare.low1359 = icmp ult i32 %compare.local1358, %compare.low.base, !dbg !23
  %compare.result1360 = select i1 %compare.high.equal, i1 %compare.low1359, i1 %compare.high.order, !dbg !23
  %436 = select i1 %compare.result1360, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %437 = insertelement <4 x float> poison, float %433, i64 0, !dbg !25
  %438 = insertelement <4 x float> %437, float %434, i64 1, !dbg !25
  %439 = insertelement <4 x float> %438, float %435, i64 2, !dbg !25
  %440 = insertelement <4 x float> %439, float %436, i64 3, !dbg !25
  %compare.local.base1361 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 24) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1362 = or i32 %compare.local.base1361, 72, !dbg !23
  %compare.low1363 = icmp ult i32 %compare.local1362, %compare.low.base, !dbg !23
  %compare.result1364 = select i1 %compare.high.equal, i1 %compare.low1363, i1 %compare.high.order, !dbg !23
  %441 = select i1 %compare.result1364, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1365 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 25) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1366 = or i32 %compare.local.base1365, 73, !dbg !23
  %compare.low1367 = icmp ult i32 %compare.local1366, %compare.low.base, !dbg !23
  %compare.result1368 = select i1 %compare.high.equal, i1 %compare.low1367, i1 %compare.high.order, !dbg !23
  %442 = select i1 %compare.result1368, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1369 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 26) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1370 = or i32 %compare.local.base1369, 74, !dbg !23
  %compare.low1371 = icmp ult i32 %compare.local1370, %compare.low.base, !dbg !23
  %compare.result1372 = select i1 %compare.high.equal, i1 %compare.low1371, i1 %compare.high.order, !dbg !23
  %443 = select i1 %compare.result1372, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1373 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 27) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1374 = or i32 %compare.local.base1373, 75, !dbg !23
  %compare.low1375 = icmp ult i32 %compare.local1374, %compare.low.base, !dbg !23
  %compare.result1376 = select i1 %compare.high.equal, i1 %compare.low1375, i1 %compare.high.order, !dbg !23
  %444 = select i1 %compare.result1376, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %445 = insertelement <4 x float> poison, float %441, i64 0, !dbg !25
  %446 = insertelement <4 x float> %445, float %442, i64 1, !dbg !25
  %447 = insertelement <4 x float> %446, float %443, i64 2, !dbg !25
  %448 = insertelement <4 x float> %447, float %444, i64 3, !dbg !25
  %compare.local.base1377 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 28) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1378 = or i32 %compare.local.base1377, 76, !dbg !23
  %compare.low1379 = icmp ult i32 %compare.local1378, %compare.low.base, !dbg !23
  %compare.result1380 = select i1 %compare.high.equal, i1 %compare.low1379, i1 %compare.high.order, !dbg !23
  %449 = select i1 %compare.result1380, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1381 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 29) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1382 = or i32 %compare.local.base1381, 77, !dbg !23
  %compare.low1383 = icmp ult i32 %compare.local1382, %compare.low.base, !dbg !23
  %compare.result1384 = select i1 %compare.high.equal, i1 %compare.low1383, i1 %compare.high.order, !dbg !23
  %450 = select i1 %compare.result1384, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1385 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 30) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1386 = or i32 %compare.local.base1385, 78, !dbg !23
  %compare.low1387 = icmp ult i32 %compare.local1386, %compare.low.base, !dbg !23
  %compare.result1388 = select i1 %compare.high.equal, i1 %compare.low1387, i1 %compare.high.order, !dbg !23
  %451 = select i1 %compare.result1388, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1389 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 31) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1390 = or i32 %compare.local.base1389, 79, !dbg !23
  %compare.low1391 = icmp ult i32 %compare.local1390, %compare.low.base, !dbg !23
  %compare.result1392 = select i1 %compare.high.equal, i1 %compare.low1391, i1 %compare.high.order, !dbg !23
  %452 = select i1 %compare.result1392, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %453 = insertelement <4 x float> poison, float %449, i64 0, !dbg !25
  %454 = insertelement <4 x float> %453, float %450, i64 1, !dbg !25
  %455 = insertelement <4 x float> %454, float %451, i64 2, !dbg !25
  %456 = insertelement <4 x float> %455, float %452, i64 3, !dbg !25
  %457 = lshr i64 %104, 32, !dbg !23
  %compare.high1393 = trunc nuw i64 %457 to i32, !dbg !23
  %compare.high.equal1395 = icmp eq i32 %compare.high1393, %compare.high1269, !dbg !23
  %compare.high.order1396 = icmp ugt i32 %compare.high1393, %compare.high1269, !dbg !23
  %compare.low.base1397 = trunc i64 %104 to i32, !dbg !23
  %compare.low1399 = icmp ugt i32 %compare.low.base1397, %compare.low.base1270, !dbg !23
  %compare.result1400 = select i1 %compare.high.equal1395, i1 %compare.low1399, i1 %compare.high.order1396, !dbg !23
  %458 = select i1 %compare.result1400, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1401 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 32) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1402 = or i32 %compare.local.base1401, 1, !dbg !23
  %compare.low1403 = icmp ult i32 %compare.local1402, %compare.low.base1397, !dbg !23
  %compare.result1404 = select i1 %compare.high.equal1395, i1 %compare.low1403, i1 %compare.high.order1396, !dbg !23
  %459 = select i1 %compare.result1404, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1405 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 33) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1406 = or i32 %compare.local.base1405, 2, !dbg !23
  %compare.low1407 = icmp ult i32 %compare.local1406, %compare.low.base1397, !dbg !23
  %compare.result1408 = select i1 %compare.high.equal1395, i1 %compare.low1407, i1 %compare.high.order1396, !dbg !23
  %460 = select i1 %compare.result1408, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1409 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 34) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1410 = or i32 %compare.local.base1409, 3, !dbg !23
  %compare.low1411 = icmp ult i32 %compare.local1410, %compare.low.base1397, !dbg !23
  %compare.result1412 = select i1 %compare.high.equal1395, i1 %compare.low1411, i1 %compare.high.order1396, !dbg !23
  %461 = select i1 %compare.result1412, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %462 = insertelement <4 x float> poison, float %458, i64 0, !dbg !25
  %463 = insertelement <4 x float> %462, float %459, i64 1, !dbg !25
  %464 = insertelement <4 x float> %463, float %460, i64 2, !dbg !25
  %465 = insertelement <4 x float> %464, float %461, i64 3, !dbg !25
  %compare.local.base1413 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 35) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1414 = or i32 %compare.local.base1413, 4, !dbg !23
  %compare.low1415 = icmp ult i32 %compare.local1414, %compare.low.base1397, !dbg !23
  %compare.result1416 = select i1 %compare.high.equal1395, i1 %compare.low1415, i1 %compare.high.order1396, !dbg !23
  %466 = select i1 %compare.result1416, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1417 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 36) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1418 = or i32 %compare.local.base1417, 5, !dbg !23
  %compare.low1419 = icmp ult i32 %compare.local1418, %compare.low.base1397, !dbg !23
  %compare.result1420 = select i1 %compare.high.equal1395, i1 %compare.low1419, i1 %compare.high.order1396, !dbg !23
  %467 = select i1 %compare.result1420, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1421 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 37) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1422 = or i32 %compare.local.base1421, 6, !dbg !23
  %compare.low1423 = icmp ult i32 %compare.local1422, %compare.low.base1397, !dbg !23
  %compare.result1424 = select i1 %compare.high.equal1395, i1 %compare.low1423, i1 %compare.high.order1396, !dbg !23
  %468 = select i1 %compare.result1424, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1425 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 38) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1426 = or i32 %compare.local.base1425, 7, !dbg !23
  %compare.low1427 = icmp ult i32 %compare.local1426, %compare.low.base1397, !dbg !23
  %compare.result1428 = select i1 %compare.high.equal1395, i1 %compare.low1427, i1 %compare.high.order1396, !dbg !23
  %469 = select i1 %compare.result1428, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %470 = insertelement <4 x float> poison, float %466, i64 0, !dbg !25
  %471 = insertelement <4 x float> %470, float %467, i64 1, !dbg !25
  %472 = insertelement <4 x float> %471, float %468, i64 2, !dbg !25
  %473 = insertelement <4 x float> %472, float %469, i64 3, !dbg !25
  %compare.local.base1429 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 39) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1430 = or i32 %compare.local.base1429, 8, !dbg !23
  %compare.low1431 = icmp ult i32 %compare.local1430, %compare.low.base1397, !dbg !23
  %compare.result1432 = select i1 %compare.high.equal1395, i1 %compare.low1431, i1 %compare.high.order1396, !dbg !23
  %474 = select i1 %compare.result1432, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1433 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 40) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1434 = or i32 %compare.local.base1433, 9, !dbg !23
  %compare.low1435 = icmp ult i32 %compare.local1434, %compare.low.base1397, !dbg !23
  %compare.result1436 = select i1 %compare.high.equal1395, i1 %compare.low1435, i1 %compare.high.order1396, !dbg !23
  %475 = select i1 %compare.result1436, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1437 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 41) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1438 = or i32 %compare.local.base1437, 10, !dbg !23
  %compare.low1439 = icmp ult i32 %compare.local1438, %compare.low.base1397, !dbg !23
  %compare.result1440 = select i1 %compare.high.equal1395, i1 %compare.low1439, i1 %compare.high.order1396, !dbg !23
  %476 = select i1 %compare.result1440, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1441 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 42) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1442 = or i32 %compare.local.base1441, 11, !dbg !23
  %compare.low1443 = icmp ult i32 %compare.local1442, %compare.low.base1397, !dbg !23
  %compare.result1444 = select i1 %compare.high.equal1395, i1 %compare.low1443, i1 %compare.high.order1396, !dbg !23
  %477 = select i1 %compare.result1444, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %478 = insertelement <4 x float> poison, float %474, i64 0, !dbg !25
  %479 = insertelement <4 x float> %478, float %475, i64 1, !dbg !25
  %480 = insertelement <4 x float> %479, float %476, i64 2, !dbg !25
  %481 = insertelement <4 x float> %480, float %477, i64 3, !dbg !25
  %compare.local.base1445 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 43) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1446 = or i32 %compare.local.base1445, 12, !dbg !23
  %compare.low1447 = icmp ult i32 %compare.local1446, %compare.low.base1397, !dbg !23
  %compare.result1448 = select i1 %compare.high.equal1395, i1 %compare.low1447, i1 %compare.high.order1396, !dbg !23
  %482 = select i1 %compare.result1448, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1449 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 44) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1450 = or i32 %compare.local.base1449, 13, !dbg !23
  %compare.low1451 = icmp ult i32 %compare.local1450, %compare.low.base1397, !dbg !23
  %compare.result1452 = select i1 %compare.high.equal1395, i1 %compare.low1451, i1 %compare.high.order1396, !dbg !23
  %483 = select i1 %compare.result1452, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1453 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 45) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1454 = or i32 %compare.local.base1453, 14, !dbg !23
  %compare.low1455 = icmp ult i32 %compare.local1454, %compare.low.base1397, !dbg !23
  %compare.result1456 = select i1 %compare.high.equal1395, i1 %compare.low1455, i1 %compare.high.order1396, !dbg !23
  %484 = select i1 %compare.result1456, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1457 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 46) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1458 = or i32 %compare.local.base1457, 15, !dbg !23
  %compare.low1459 = icmp ult i32 %compare.local1458, %compare.low.base1397, !dbg !23
  %compare.result1460 = select i1 %compare.high.equal1395, i1 %compare.low1459, i1 %compare.high.order1396, !dbg !23
  %485 = select i1 %compare.result1460, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %486 = insertelement <4 x float> poison, float %482, i64 0, !dbg !25
  %487 = insertelement <4 x float> %486, float %483, i64 1, !dbg !25
  %488 = insertelement <4 x float> %487, float %484, i64 2, !dbg !25
  %489 = insertelement <4 x float> %488, float %485, i64 3, !dbg !25
  %compare.local.base1461 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 47) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1462 = or i32 %compare.local.base1461, 64, !dbg !23
  %compare.low1463 = icmp ult i32 %compare.local1462, %compare.low.base1397, !dbg !23
  %compare.result1464 = select i1 %compare.high.equal1395, i1 %compare.low1463, i1 %compare.high.order1396, !dbg !23
  %490 = select i1 %compare.result1464, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1465 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 48) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1466 = or i32 %compare.local.base1465, 65, !dbg !23
  %compare.low1467 = icmp ult i32 %compare.local1466, %compare.low.base1397, !dbg !23
  %compare.result1468 = select i1 %compare.high.equal1395, i1 %compare.low1467, i1 %compare.high.order1396, !dbg !23
  %491 = select i1 %compare.result1468, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1469 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 49) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1470 = or i32 %compare.local.base1469, 66, !dbg !23
  %compare.low1471 = icmp ult i32 %compare.local1470, %compare.low.base1397, !dbg !23
  %compare.result1472 = select i1 %compare.high.equal1395, i1 %compare.low1471, i1 %compare.high.order1396, !dbg !23
  %492 = select i1 %compare.result1472, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1473 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 50) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1474 = or i32 %compare.local.base1473, 67, !dbg !23
  %compare.low1475 = icmp ult i32 %compare.local1474, %compare.low.base1397, !dbg !23
  %compare.result1476 = select i1 %compare.high.equal1395, i1 %compare.low1475, i1 %compare.high.order1396, !dbg !23
  %493 = select i1 %compare.result1476, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %494 = insertelement <4 x float> poison, float %490, i64 0, !dbg !25
  %495 = insertelement <4 x float> %494, float %491, i64 1, !dbg !25
  %496 = insertelement <4 x float> %495, float %492, i64 2, !dbg !25
  %497 = insertelement <4 x float> %496, float %493, i64 3, !dbg !25
  %compare.local.base1477 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 51) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1478 = or i32 %compare.local.base1477, 68, !dbg !23
  %compare.low1479 = icmp ult i32 %compare.local1478, %compare.low.base1397, !dbg !23
  %compare.result1480 = select i1 %compare.high.equal1395, i1 %compare.low1479, i1 %compare.high.order1396, !dbg !23
  %498 = select i1 %compare.result1480, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1481 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 52) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1482 = or i32 %compare.local.base1481, 69, !dbg !23
  %compare.low1483 = icmp ult i32 %compare.local1482, %compare.low.base1397, !dbg !23
  %compare.result1484 = select i1 %compare.high.equal1395, i1 %compare.low1483, i1 %compare.high.order1396, !dbg !23
  %499 = select i1 %compare.result1484, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1485 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 53) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1486 = or i32 %compare.local.base1485, 70, !dbg !23
  %compare.low1487 = icmp ult i32 %compare.local1486, %compare.low.base1397, !dbg !23
  %compare.result1488 = select i1 %compare.high.equal1395, i1 %compare.low1487, i1 %compare.high.order1396, !dbg !23
  %500 = select i1 %compare.result1488, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1489 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 54) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1490 = or i32 %compare.local.base1489, 71, !dbg !23
  %compare.low1491 = icmp ult i32 %compare.local1490, %compare.low.base1397, !dbg !23
  %compare.result1492 = select i1 %compare.high.equal1395, i1 %compare.low1491, i1 %compare.high.order1396, !dbg !23
  %501 = select i1 %compare.result1492, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %502 = insertelement <4 x float> poison, float %498, i64 0, !dbg !25
  %503 = insertelement <4 x float> %502, float %499, i64 1, !dbg !25
  %504 = insertelement <4 x float> %503, float %500, i64 2, !dbg !25
  %505 = insertelement <4 x float> %504, float %501, i64 3, !dbg !25
  %compare.local.base1493 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 55) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1494 = or i32 %compare.local.base1493, 72, !dbg !23
  %compare.low1495 = icmp ult i32 %compare.local1494, %compare.low.base1397, !dbg !23
  %compare.result1496 = select i1 %compare.high.equal1395, i1 %compare.low1495, i1 %compare.high.order1396, !dbg !23
  %506 = select i1 %compare.result1496, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1497 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 56) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1498 = or i32 %compare.local.base1497, 73, !dbg !23
  %compare.low1499 = icmp ult i32 %compare.local1498, %compare.low.base1397, !dbg !23
  %compare.result1500 = select i1 %compare.high.equal1395, i1 %compare.low1499, i1 %compare.high.order1396, !dbg !23
  %507 = select i1 %compare.result1500, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1501 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 57) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1502 = or i32 %compare.local.base1501, 74, !dbg !23
  %compare.low1503 = icmp ult i32 %compare.local1502, %compare.low.base1397, !dbg !23
  %compare.result1504 = select i1 %compare.high.equal1395, i1 %compare.low1503, i1 %compare.high.order1396, !dbg !23
  %508 = select i1 %compare.result1504, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1505 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 58) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1506 = or i32 %compare.local.base1505, 75, !dbg !23
  %compare.low1507 = icmp ult i32 %compare.local1506, %compare.low.base1397, !dbg !23
  %compare.result1508 = select i1 %compare.high.equal1395, i1 %compare.low1507, i1 %compare.high.order1396, !dbg !23
  %509 = select i1 %compare.result1508, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %510 = insertelement <4 x float> poison, float %506, i64 0, !dbg !25
  %511 = insertelement <4 x float> %510, float %507, i64 1, !dbg !25
  %512 = insertelement <4 x float> %511, float %508, i64 2, !dbg !25
  %513 = insertelement <4 x float> %512, float %509, i64 3, !dbg !25
  %compare.local.base1509 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 59) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1510 = or i32 %compare.local.base1509, 76, !dbg !23
  %compare.low1511 = icmp ult i32 %compare.local1510, %compare.low.base1397, !dbg !23
  %compare.result1512 = select i1 %compare.high.equal1395, i1 %compare.low1511, i1 %compare.high.order1396, !dbg !23
  %514 = select i1 %compare.result1512, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1513 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 60) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1514 = or i32 %compare.local.base1513, 77, !dbg !23
  %compare.low1515 = icmp ult i32 %compare.local1514, %compare.low.base1397, !dbg !23
  %compare.result1516 = select i1 %compare.high.equal1395, i1 %compare.low1515, i1 %compare.high.order1396, !dbg !23
  %515 = select i1 %compare.result1516, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1517 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 61) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1518 = or i32 %compare.local.base1517, 78, !dbg !23
  %compare.low1519 = icmp ult i32 %compare.local1518, %compare.low.base1397, !dbg !23
  %compare.result1520 = select i1 %compare.high.equal1395, i1 %compare.low1519, i1 %compare.high.order1396, !dbg !23
  %516 = select i1 %compare.result1520, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base1521 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 62) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local1522 = or i32 %compare.local.base1521, 79, !dbg !23
  %compare.low1523 = icmp ult i32 %compare.local1522, %compare.low.base1397, !dbg !23
  %compare.result1524 = select i1 %compare.high.equal1395, i1 %compare.low1523, i1 %compare.high.order1396, !dbg !23
  %517 = select i1 %compare.result1524, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %518 = insertelement <4 x float> poison, float %514, i64 0, !dbg !25
  %519 = insertelement <4 x float> %518, float %515, i64 1, !dbg !25
  %520 = insertelement <4 x float> %519, float %516, i64 2, !dbg !25
  %521 = insertelement <4 x float> %520, float %517, i64 3, !dbg !25
  %522 = load <4 x half>, ptr addrspace(3) %kdma.gep2182, align 8, !dbg !15, !frisk.k_tail_pair !41
  %523 = load <4 x half>, ptr addrspace(3) %kdma.gep2186, align 8, !dbg !15, !frisk.k_tail_pair !41
  %524 = load <4 x half>, ptr addrspace(3) %kdma.gep2183, align 8, !dbg !15, !frisk.k_tail_pair !42
  %525 = load <4 x half>, ptr addrspace(3) %kdma.gep2187, align 8, !dbg !15, !frisk.k_tail_pair !42
  %526 = load <4 x half>, ptr addrspace(3) %kdma.gep2184, align 8, !dbg !15, !frisk.k_tail_pair !43
  %527 = load <4 x half>, ptr addrspace(3) %kdma.gep2188, align 8, !dbg !15, !frisk.k_tail_pair !43
  %528 = load <4 x half>, ptr addrspace(3) %kdma.gep2185, align 8, !dbg !15, !frisk.k_tail_pair !44
  %529 = load <4 x half>, ptr addrspace(3) %kdma.gep2189, align 8, !dbg !15, !frisk.k_tail_pair !44
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_tail_prefetch !1
  %530 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %522, <4 x float> %389) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_tail_prefetch !1
  %531 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %524, <4 x float> %349) #9, !dbg !15
  %532 = fadd <4 x float> %530, %400, !dbg !25
  %533 = extractelement <4 x float> %532, i64 0, !dbg !25
  %534 = call float @llvm.maxnum.f32(float %533, float 0xFFF0000000000000), !dbg !26, !frisk.local_reduction !1
  %535 = extractelement <4 x float> %532, i64 1, !dbg !25
  %536 = extractelement <4 x float> %532, i64 2, !dbg !25
  %537 = extractelement <4 x float> %532, i64 3, !dbg !25
  %538 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %526, <4 x float> %531) #9, !dbg !15
  %539 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %528, <4 x float> %351) #9, !dbg !15
  %540 = fadd <4 x float> %538, %408, !dbg !25
  %541 = extractelement <4 x float> %540, i64 0, !dbg !25
  %local.reduce219 = call float @llvm.maxnum.f32(float %534, float %541), !dbg !26, !frisk.local_reduction !1
  %542 = extractelement <4 x float> %540, i64 1, !dbg !25
  %local.reduce220 = call float @llvm.maxnum.f32(float %535, float %542), !dbg !26, !frisk.local_reduction !1
  %543 = extractelement <4 x float> %540, i64 2, !dbg !25
  %local.reduce221 = call float @llvm.maxnum.f32(float %536, float %543), !dbg !26, !frisk.local_reduction !1
  %544 = extractelement <4 x float> %540, i64 3, !dbg !25
  %local.reduce222 = call float @llvm.maxnum.f32(float %537, float %544), !dbg !26, !frisk.local_reduction !1
  %545 = load <4 x half>, ptr addrspace(3) %kdma.gep2190, align 8, !dbg !15, !frisk.k_tail_pair !39
  %546 = load <4 x half>, ptr addrspace(3) %kdma.gep2194, align 8, !dbg !15, !frisk.k_tail_pair !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_tail_prefetch !1
  %547 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %523, <4 x float> %539) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_tail_prefetch !1
  %548 = load <4 x half>, ptr addrspace(3) %kdma.gep2191, align 8, !dbg !15, !frisk.k_tail_pair !45
  %549 = load <4 x half>, ptr addrspace(3) %kdma.gep2195, align 8, !dbg !15, !frisk.k_tail_pair !45
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_tail_prefetch !1
  %550 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %525, <4 x float> %353) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25, !frisk.k_tail_prefetch !1
  %551 = fadd <4 x float> %547, %416, !dbg !25
  %552 = extractelement <4 x float> %551, i64 0, !dbg !25
  %local.reduce223 = call float @llvm.maxnum.f32(float %local.reduce219, float %552), !dbg !26, !frisk.local_reduction !1
  %553 = extractelement <4 x float> %551, i64 1, !dbg !25
  %local.reduce224 = call float @llvm.maxnum.f32(float %local.reduce220, float %553), !dbg !26, !frisk.local_reduction !1
  %554 = extractelement <4 x float> %551, i64 2, !dbg !25
  %local.reduce225 = call float @llvm.maxnum.f32(float %local.reduce221, float %554), !dbg !26, !frisk.local_reduction !1
  %555 = extractelement <4 x float> %551, i64 3, !dbg !25
  %local.reduce226 = call float @llvm.maxnum.f32(float %local.reduce222, float %555), !dbg !26, !frisk.local_reduction !1
  %556 = load <4 x half>, ptr addrspace(3) %kdma.gep2192, align 8, !dbg !15, !frisk.k_tail_pair !46
  %557 = load <4 x half>, ptr addrspace(3) %kdma.gep2196, align 8, !dbg !15, !frisk.k_tail_pair !46
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_tail_prefetch !1
  %558 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %527, <4 x float> %550) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_tail_prefetch !1
  %559 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %529, <4 x float> %355) #9, !dbg !15
  %560 = fadd <4 x float> %558, %424, !dbg !25
  %561 = extractelement <4 x float> %560, i64 0, !dbg !25
  %local.reduce227 = call float @llvm.maxnum.f32(float %local.reduce223, float %561), !dbg !26, !frisk.local_reduction !1
  %562 = extractelement <4 x float> %560, i64 1, !dbg !25
  %local.reduce228 = call float @llvm.maxnum.f32(float %local.reduce224, float %562), !dbg !26, !frisk.local_reduction !1
  %563 = extractelement <4 x float> %560, i64 2, !dbg !25
  %local.reduce229 = call float @llvm.maxnum.f32(float %local.reduce225, float %563), !dbg !26, !frisk.local_reduction !1
  %564 = extractelement <4 x float> %560, i64 3, !dbg !25
  %local.reduce230 = call float @llvm.maxnum.f32(float %local.reduce226, float %564), !dbg !26, !frisk.local_reduction !1
  %565 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %545, <4 x float> %559) #9, !dbg !15
  %566 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %548, <4 x float> %357) #9, !dbg !15
  %567 = fadd <4 x float> %565, %432, !dbg !25
  %568 = extractelement <4 x float> %567, i64 0, !dbg !25
  %local.reduce231 = call float @llvm.maxnum.f32(float %local.reduce227, float %568), !dbg !26, !frisk.local_reduction !1
  %569 = extractelement <4 x float> %567, i64 1, !dbg !25
  %local.reduce232 = call float @llvm.maxnum.f32(float %local.reduce228, float %569), !dbg !26, !frisk.local_reduction !1
  %570 = extractelement <4 x float> %567, i64 2, !dbg !25
  %local.reduce233 = call float @llvm.maxnum.f32(float %local.reduce229, float %570), !dbg !26, !frisk.local_reduction !1
  %571 = extractelement <4 x float> %567, i64 3, !dbg !25
  %local.reduce234 = call float @llvm.maxnum.f32(float %local.reduce230, float %571), !dbg !26, !frisk.local_reduction !1
  %572 = load <4 x half>, ptr addrspace(3) %kdma.gep2193, align 8, !dbg !15
  %573 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %556, <4 x float> %566) #9, !dbg !15
  %574 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %572, <4 x float> %359) #9, !dbg !15
  %575 = fadd <4 x float> %573, %440, !dbg !25
  %576 = extractelement <4 x float> %575, i64 0, !dbg !25
  %local.reduce235 = call float @llvm.maxnum.f32(float %local.reduce231, float %576), !dbg !26, !frisk.local_reduction !1
  %577 = extractelement <4 x float> %575, i64 1, !dbg !25
  %local.reduce236 = call float @llvm.maxnum.f32(float %local.reduce232, float %577), !dbg !26, !frisk.local_reduction !1
  %578 = extractelement <4 x float> %575, i64 2, !dbg !25
  %local.reduce237 = call float @llvm.maxnum.f32(float %local.reduce233, float %578), !dbg !26, !frisk.local_reduction !1
  %579 = extractelement <4 x float> %575, i64 3, !dbg !25
  %local.reduce238 = call float @llvm.maxnum.f32(float %local.reduce234, float %579), !dbg !26, !frisk.local_reduction !1
  %580 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %546, <4 x float> %574) #9, !dbg !15
  %581 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %549, <4 x float> %361) #9, !dbg !15
  %582 = fadd <4 x float> %580, %448, !dbg !25
  %583 = extractelement <4 x float> %582, i64 0, !dbg !25
  %local.reduce239 = call float @llvm.maxnum.f32(float %local.reduce235, float %583), !dbg !26, !frisk.local_reduction !1
  %584 = extractelement <4 x float> %582, i64 1, !dbg !25
  %local.reduce240 = call float @llvm.maxnum.f32(float %local.reduce236, float %584), !dbg !26, !frisk.local_reduction !1
  %585 = extractelement <4 x float> %582, i64 2, !dbg !25
  %local.reduce241 = call float @llvm.maxnum.f32(float %local.reduce237, float %585), !dbg !26, !frisk.local_reduction !1
  %586 = extractelement <4 x float> %582, i64 3, !dbg !25
  %local.reduce242 = call float @llvm.maxnum.f32(float %local.reduce238, float %586), !dbg !26, !frisk.local_reduction !1
  %587 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %557, <4 x float> %581) #9, !dbg !15
  %588 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %388, <4 x float> %363) #9, !dbg !15
  %589 = fadd <4 x float> %587, %456, !dbg !25
  %590 = extractelement <4 x float> %589, i64 0, !dbg !25
  %local.reduce243 = call float @llvm.maxnum.f32(float %local.reduce239, float %590), !dbg !26, !frisk.local_reduction !1
  %591 = extractelement <4 x float> %589, i64 1, !dbg !25
  %local.reduce244 = call float @llvm.maxnum.f32(float %local.reduce240, float %591), !dbg !26, !frisk.local_reduction !1
  %592 = extractelement <4 x float> %589, i64 2, !dbg !25
  %local.reduce245 = call float @llvm.maxnum.f32(float %local.reduce241, float %592), !dbg !26, !frisk.local_reduction !1
  %593 = extractelement <4 x float> %589, i64 3, !dbg !25
  %local.reduce246 = call float @llvm.maxnum.f32(float %local.reduce242, float %593), !dbg !26, !frisk.local_reduction !1
  %local.reduce247 = call float @llvm.maxnum.f32(float %local.reduce243, float %local.reduce244), !dbg !26, !frisk.local_reduction !1
  %local.reduce248 = call float @llvm.maxnum.f32(float %local.reduce245, float %local.reduce246), !dbg !26, !frisk.local_reduction !1
  %594 = call float @llvm.maxnum.f32(float %local.reduce247, float %local.reduce248), !dbg !26, !frisk.local_reduction !1
  %595 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %522, <4 x float> %588) #9, !dbg !15
  %596 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %524, <4 x float> %365) #9, !dbg !15
  %597 = fadd <4 x float> %595, %465, !dbg !25
  %598 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %526, <4 x float> %596) #9, !dbg !15
  %599 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %528, <4 x float> %367) #9, !dbg !15
  %600 = fadd <4 x float> %598, %473, !dbg !25
  %601 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %523, <4 x float> %599) #9, !dbg !15
  %602 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %525, <4 x float> %369) #9, !dbg !15
  %603 = fadd <4 x float> %601, %481, !dbg !25
  %604 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %527, <4 x float> %602) #9, !dbg !15
  %605 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %529, <4 x float> %371) #9, !dbg !15
  %606 = fadd <4 x float> %604, %489, !dbg !25
  %607 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %545, <4 x float> %605) #9, !dbg !15
  %608 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %548, <4 x float> %373) #9, !dbg !15
  %609 = fadd <4 x float> %607, %497, !dbg !25
  %610 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %556, <4 x float> %608) #9, !dbg !15
  %611 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %572, <4 x float> %375) #9, !dbg !15
  %612 = fadd <4 x float> %610, %505, !dbg !25
  %613 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %546, <4 x float> %611) #9, !dbg !15
  %614 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %549, <4 x float> %377) #9, !dbg !15
  %615 = fadd <4 x float> %613, %513, !dbg !25
  %616 = extractelement <4 x float> %597, i64 0, !dbg !25
  %617 = extractelement <4 x float> %597, i64 1, !dbg !25
  %618 = extractelement <4 x float> %597, i64 2, !dbg !25
  %619 = extractelement <4 x float> %597, i64 3, !dbg !25
  %620 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %557, <4 x float> %614) #9, !dbg !15
  %621 = extractelement <4 x float> %600, i64 0, !dbg !25
  %622 = extractelement <4 x float> %600, i64 1, !dbg !25
  %623 = extractelement <4 x float> %600, i64 2, !dbg !25
  %624 = extractelement <4 x float> %600, i64 3, !dbg !25
  %625 = extractelement <4 x float> %603, i64 0, !dbg !25
  %626 = extractelement <4 x float> %603, i64 1, !dbg !25
  %627 = extractelement <4 x float> %603, i64 2, !dbg !25
  %628 = extractelement <4 x float> %603, i64 3, !dbg !25
  %629 = fadd <4 x float> %620, %521, !dbg !25
  %630 = extractelement <4 x float> %606, i64 0, !dbg !25
  %631 = extractelement <4 x float> %606, i64 1, !dbg !25
  %632 = extractelement <4 x float> %606, i64 2, !dbg !25
  %633 = extractelement <4 x float> %606, i64 3, !dbg !25
  %634 = extractelement <4 x float> %609, i64 0, !dbg !25
  %635 = extractelement <4 x float> %609, i64 1, !dbg !25
  %636 = extractelement <4 x float> %609, i64 2, !dbg !25
  %637 = extractelement <4 x float> %609, i64 3, !dbg !25
  %638 = extractelement <4 x float> %612, i64 0, !dbg !25
  %639 = extractelement <4 x float> %612, i64 1, !dbg !25
  %640 = extractelement <4 x float> %612, i64 2, !dbg !25
  %641 = extractelement <4 x float> %612, i64 3, !dbg !25
  %642 = extractelement <4 x float> %615, i64 0, !dbg !25
  %643 = extractelement <4 x float> %615, i64 1, !dbg !25
  %644 = extractelement <4 x float> %615, i64 2, !dbg !25
  %645 = extractelement <4 x float> %615, i64 3, !dbg !25
  %646 = extractelement <4 x float> %629, i64 0, !dbg !25
  %647 = extractelement <4 x float> %629, i64 1, !dbg !25
  %648 = extractelement <4 x float> %629, i64 2, !dbg !25
  %649 = extractelement <4 x float> %629, i64 3, !dbg !25
  %650 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !26
  %651 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %650), !dbg !26
  %652 = add nuw nsw i32 %651, 64, !dbg !26
  %653 = xor i32 %651, 16, !dbg !26
  %654 = and i32 %652, 192, !dbg !26
  %655 = icmp samesign ult i32 %653, %654, !dbg !26
  %656 = select i1 %655, i32 %653, i32 %651, !dbg !26
  %657 = shl nuw nsw i32 %656, 2, !dbg !26
  %658 = bitcast float %594 to i32, !dbg !26
  %659 = call i32 @llvm.amdgcn.ds.bpermute(i32 %657, i32 %658), !dbg !26
  %660 = bitcast i32 %659 to float, !dbg !26
  %661 = call float @llvm.maxnum.f32(float %594, float %660), !dbg !26
  %662 = xor i32 %651, 32, !dbg !26
  %663 = icmp samesign ult i32 %662, %654, !dbg !26
  %664 = select i1 %663, i32 %662, i32 %651, !dbg !26
  %665 = shl nuw nsw i32 %664, 2, !dbg !26
  %666 = bitcast float %661 to i32, !dbg !26
  %667 = call i32 @llvm.amdgcn.ds.bpermute(i32 %665, i32 %666), !dbg !26
  %668 = bitcast i32 %667 to float, !dbg !26
  %669 = call float @llvm.maxnum.f32(float %661, float %668), !dbg !26
  %670 = extractvalue [2 x <1 x float>] %109, 0, !dbg !26
  %671 = insertelement <1 x float> poison, float %669, i64 0, !dbg !26
  %672 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %671, <1 x float> %670), !dbg !26
  %673 = call float @llvm.maxnum.f32(float %616, float 0xFFF0000000000000), !dbg !26, !frisk.local_reduction !1
  %local.reduce188 = call float @llvm.maxnum.f32(float %673, float %621), !dbg !26, !frisk.local_reduction !1
  %local.reduce189 = call float @llvm.maxnum.f32(float %617, float %622), !dbg !26, !frisk.local_reduction !1
  %local.reduce190 = call float @llvm.maxnum.f32(float %618, float %623), !dbg !26, !frisk.local_reduction !1
  %local.reduce191 = call float @llvm.maxnum.f32(float %619, float %624), !dbg !26, !frisk.local_reduction !1
  %local.reduce192 = call float @llvm.maxnum.f32(float %local.reduce188, float %625), !dbg !26, !frisk.local_reduction !1
  %local.reduce193 = call float @llvm.maxnum.f32(float %local.reduce189, float %626), !dbg !26, !frisk.local_reduction !1
  %local.reduce194 = call float @llvm.maxnum.f32(float %local.reduce190, float %627), !dbg !26, !frisk.local_reduction !1
  %local.reduce195 = call float @llvm.maxnum.f32(float %local.reduce191, float %628), !dbg !26, !frisk.local_reduction !1
  %local.reduce196 = call float @llvm.maxnum.f32(float %local.reduce192, float %630), !dbg !26, !frisk.local_reduction !1
  %local.reduce197 = call float @llvm.maxnum.f32(float %local.reduce193, float %631), !dbg !26, !frisk.local_reduction !1
  %local.reduce198 = call float @llvm.maxnum.f32(float %local.reduce194, float %632), !dbg !26, !frisk.local_reduction !1
  %local.reduce199 = call float @llvm.maxnum.f32(float %local.reduce195, float %633), !dbg !26, !frisk.local_reduction !1
  %local.reduce200 = call float @llvm.maxnum.f32(float %local.reduce196, float %634), !dbg !26, !frisk.local_reduction !1
  %local.reduce201 = call float @llvm.maxnum.f32(float %local.reduce197, float %635), !dbg !26, !frisk.local_reduction !1
  %local.reduce202 = call float @llvm.maxnum.f32(float %local.reduce198, float %636), !dbg !26, !frisk.local_reduction !1
  %local.reduce203 = call float @llvm.maxnum.f32(float %local.reduce199, float %637), !dbg !26, !frisk.local_reduction !1
  %local.reduce204 = call float @llvm.maxnum.f32(float %local.reduce200, float %638), !dbg !26, !frisk.local_reduction !1
  %local.reduce205 = call float @llvm.maxnum.f32(float %local.reduce201, float %639), !dbg !26, !frisk.local_reduction !1
  %local.reduce206 = call float @llvm.maxnum.f32(float %local.reduce202, float %640), !dbg !26, !frisk.local_reduction !1
  %local.reduce207 = call float @llvm.maxnum.f32(float %local.reduce203, float %641), !dbg !26, !frisk.local_reduction !1
  %local.reduce208 = call float @llvm.maxnum.f32(float %local.reduce204, float %642), !dbg !26, !frisk.local_reduction !1
  %local.reduce209 = call float @llvm.maxnum.f32(float %local.reduce205, float %643), !dbg !26, !frisk.local_reduction !1
  %local.reduce210 = call float @llvm.maxnum.f32(float %local.reduce206, float %644), !dbg !26, !frisk.local_reduction !1
  %local.reduce211 = call float @llvm.maxnum.f32(float %local.reduce207, float %645), !dbg !26, !frisk.local_reduction !1
  %local.reduce212 = call float @llvm.maxnum.f32(float %local.reduce208, float %646), !dbg !26, !frisk.local_reduction !1
  %local.reduce213 = call float @llvm.maxnum.f32(float %local.reduce209, float %647), !dbg !26, !frisk.local_reduction !1
  %local.reduce214 = call float @llvm.maxnum.f32(float %local.reduce210, float %648), !dbg !26, !frisk.local_reduction !1
  %local.reduce215 = call float @llvm.maxnum.f32(float %local.reduce211, float %649), !dbg !26, !frisk.local_reduction !1
  %local.reduce216 = call float @llvm.maxnum.f32(float %local.reduce212, float %local.reduce213), !dbg !26, !frisk.local_reduction !1
  %local.reduce217 = call float @llvm.maxnum.f32(float %local.reduce214, float %local.reduce215), !dbg !26, !frisk.local_reduction !1
  %674 = call float @llvm.maxnum.f32(float %local.reduce216, float %local.reduce217), !dbg !26, !frisk.local_reduction !1
  %675 = bitcast float %674 to i32, !dbg !26
  %676 = call i32 @llvm.amdgcn.ds.bpermute(i32 %657, i32 %675), !dbg !26
  %677 = bitcast i32 %676 to float, !dbg !26
  %678 = call float @llvm.maxnum.f32(float %674, float %677), !dbg !26
  %679 = bitcast float %678 to i32, !dbg !26
  %680 = call i32 @llvm.amdgcn.ds.bpermute(i32 %665, i32 %679), !dbg !26
  %681 = bitcast i32 %680 to float, !dbg !26
  %682 = call float @llvm.maxnum.f32(float %678, float %681), !dbg !26
  %683 = extractvalue [2 x <1 x float>] %109, 1, !dbg !26
  %684 = insertelement <1 x float> poison, float %682, i64 0, !dbg !26
  %685 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %684, <1 x float> %683), !dbg !26
  %686 = shufflevector <1 x float> %672, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !27
  %687 = shufflevector <1 x float> %685, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !27
  %688 = fsub <1 x float> %670, %672, !dbg !28
  %689 = extractelement <1 x float> %688, i64 0, !dbg !29
  %690 = call float @llvm.amdgcn.exp2.f32(float %689), !dbg !29
  %691 = insertelement <4 x float> poison, float %690, i64 0, !dbg !34
  %692 = shufflevector <4 x float> %691, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !34
  %693 = extractvalue [2 x <1 x float>] %108, 0, !dbg !30
  %694 = extractelement <1 x float> %693, i64 0, !dbg !30
  %695 = fmul float %694, %690, !dbg !30
  %696 = fsub <1 x float> %683, %685, !dbg !28
  %697 = extractelement <1 x float> %696, i64 0, !dbg !29
  %698 = call float @llvm.amdgcn.exp2.f32(float %697), !dbg !29
  %699 = insertelement <4 x float> poison, float %698, i64 0, !dbg !34
  %700 = shufflevector <4 x float> %699, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !34
  %701 = extractvalue [2 x <1 x float>] %108, 1, !dbg !30
  %702 = extractelement <1 x float> %701, i64 0, !dbg !30
  %703 = fmul float %702, %698, !dbg !30
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %vdb.address1802 = or disjoint i64 %.idx17771779, 32, !dbg !16
  %vdb.address1803 = or disjoint i64 %address.local989, %vdb.address1802, !dbg !16
  %vdb.address1804 = trunc i64 %vdb.address1803 to i32, !dbg !16
  %vdb.address1805 = shl i32 %vdb.address1804, 1, !dbg !16
  %vdb.address1813 = or disjoint i64 %address.local995, %vdb.address1802, !dbg !16
  %vdb.address1814 = trunc i64 %vdb.address1813 to i32, !dbg !16
  %vdb.address1815 = shl i32 %vdb.address1814, 1, !dbg !16
  %vdb.dma.base = add i32 %dma.buffer.lds1702, 8192, !dbg !13
  %704 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1805, <4 x i32> %dma.buffer.rsrc1677) #10, !dbg !16
  %vdb.dma.base1820 = add i32 %dma.buffer.lds1699, 8192, !dbg !13
  %705 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1820, i32 %vdb.address1815, <4 x i32> %dma.buffer.rsrc1677) #10, !dbg !16
  %address.local997 = shl nuw nsw i32 %address.thread566, 3, !dbg !13
  %address.local998 = and i32 %address.local997, 120, !dbg !13
  %address.local1000 = and i32 %address.local643, 1536, !dbg !13
  %address.local1001 = or disjoint i32 %address.local998, %address.local1000, !dbg !13
  %address.local1002 = zext nneg i32 %address.local1001 to i64, !dbg !13
  %address.local1003 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1002, !dbg !13
  %706 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1003, i32 0) #0, !dbg !13, !frisk.v_double_buffer !40
  %707 = bitcast <4 x i32> %706 to <8 x half>, !dbg !13
  %address.local1004 = and i64 %address.local567, 60, !dbg !13
  %address.local1005 = add nuw nsw i64 %address.local1004, %address.local567, !dbg !13
  %address.local1007.idx = shl nuw nsw i64 %address.local1005, 4, !dbg !13
  %address.local1007 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %address.local1007.idx, !dbg !13
  %address.local1008 = lshr i64 %address.local623, 4, !dbg !13
  %address.local1010.idx = mul nuw nsw i64 %address.local1008, 768, !dbg !13
  %address.local1010 = getelementptr i8, ptr addrspace(3) %address.local1007, i64 %address.local1010.idx, !dbg !13
  %address.local1012.idx = mul nsw i64 %address.local610, -64, !dbg !13
  %address.local1012 = getelementptr i8, ptr addrspace(3) %address.local1010, i64 %address.local1012.idx, !dbg !13
  %address.local1013 = getelementptr i8, ptr addrspace(3) %address.local1012, i64 256, !dbg !13
  %708 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1013, i32 0) #0, !dbg !13, !frisk.v_double_buffer !40
  %709 = bitcast <4 x i32> %708 to <8 x half>, !dbg !13
  %address.local1017 = getelementptr i8, ptr addrspace(3) %address.local1012, i64 512, !dbg !13
  %710 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1017, i32 0) #0, !dbg !13, !frisk.v_double_buffer !40
  %711 = bitcast <4 x i32> %710 to <8 x half>, !dbg !13
  %address.local1021 = getelementptr i8, ptr addrspace(3) %address.local1012, i64 768, !dbg !13
  %712 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1021, i32 0) #0, !dbg !13, !frisk.v_double_buffer !40
  %713 = bitcast <4 x i32> %712 to <8 x half>, !dbg !13
  %address.local1023 = getelementptr i8, ptr addrspace(3) %address.local1003, i64 4096, !dbg !13
  %714 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1023, i32 0) #0, !dbg !13, !frisk.v_double_buffer !40
  %715 = bitcast <4 x i32> %714 to <8 x half>, !dbg !13
  %address.local1027 = getelementptr i8, ptr addrspace(3) %address.local1012, i64 4352, !dbg !13
  %716 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1027, i32 0) #0, !dbg !13, !frisk.v_double_buffer !40
  %717 = bitcast <4 x i32> %716 to <8 x half>, !dbg !13
  %address.local1031 = getelementptr i8, ptr addrspace(3) %address.local1012, i64 4608, !dbg !13
  %718 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1031, i32 0) #0, !dbg !13, !frisk.v_double_buffer !40
  %719 = bitcast <4 x i32> %718 to <8 x half>, !dbg !13
  %address.local1035 = getelementptr i8, ptr addrspace(3) %address.local1012, i64 4864, !dbg !13
  %720 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1035, i32 0) #0, !dbg !13, !frisk.v_double_buffer !40
  %721 = shufflevector <8 x half> %707, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %722 = extractvalue [2 x <8 x float>] %107, 0, !dbg !34
  %723 = shufflevector <8 x float> %722, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %724 = fmul <4 x float> %723, %692, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %725 = fsub <4 x float> %532, %686, !dbg !27
  %726 = extractelement <4 x float> %725, i64 0, !dbg !31
  %727 = call float @llvm.amdgcn.exp2.f32(float %726), !dbg !31
  %728 = insertelement <4 x float> poison, float %727, i64 0, !dbg !33
  %729 = extractelement <4 x float> %725, i64 1, !dbg !31
  %730 = call float @llvm.amdgcn.exp2.f32(float %729), !dbg !31
  %731 = insertelement <4 x float> %728, float %730, i64 1, !dbg !33
  %732 = extractelement <4 x float> %725, i64 2, !dbg !31
  %733 = call float @llvm.amdgcn.exp2.f32(float %732), !dbg !31
  %734 = insertelement <4 x float> %731, float %733, i64 2, !dbg !33
  %735 = extractelement <4 x float> %725, i64 3, !dbg !31
  %736 = call float @llvm.amdgcn.exp2.f32(float %735), !dbg !31
  %737 = insertelement <4 x float> %734, float %736, i64 3, !dbg !33
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %737) #9, !dbg !33, !frisk.register_lifetimes !1
  %738 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !33
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %738) #9, !dbg !31, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %739 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %721, <4 x float> %724) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %740 = shufflevector <8 x half> %707, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %741 = shufflevector <8 x float> %722, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %742 = fmul <4 x float> %741, %692, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %743 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %740, <4 x float> %742) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !34
  %744 = extractvalue [2 x <8 x float>] %107, 1, !dbg !34
  %745 = shufflevector <8 x float> %744, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %746 = fmul <4 x float> %745, %700, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %747 = fsub <4 x float> %597, %687, !dbg !27
  %748 = extractelement <4 x float> %747, i64 0, !dbg !31
  %749 = call float @llvm.amdgcn.exp2.f32(float %748), !dbg !31
  %750 = insertelement <4 x float> poison, float %749, i64 0, !dbg !33
  %751 = extractelement <4 x float> %747, i64 1, !dbg !31
  %752 = call float @llvm.amdgcn.exp2.f32(float %751), !dbg !31
  %753 = insertelement <4 x float> %750, float %752, i64 1, !dbg !33
  %754 = extractelement <4 x float> %747, i64 2, !dbg !31
  %755 = call float @llvm.amdgcn.exp2.f32(float %754), !dbg !31
  %756 = insertelement <4 x float> %753, float %755, i64 2, !dbg !33
  %757 = extractelement <4 x float> %747, i64 3, !dbg !31
  %758 = call float @llvm.amdgcn.exp2.f32(float %757), !dbg !31
  %759 = insertelement <4 x float> %756, float %758, i64 3, !dbg !33
  %math.result456 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %759) #9, !dbg !33, !frisk.register_lifetimes !1
  %760 = fptrunc <4 x float> %math.result456 to <4 x half>, !dbg !33
  %math.packed457 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %760) #9, !dbg !31, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %761 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed457, <4 x half> %721, <4 x float> %746) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !34
  %762 = shufflevector <8 x float> %744, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %763 = fmul <4 x float> %762, %700, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %764 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed457, <4 x half> %740, <4 x float> %763) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %765 = shufflevector <8 x half> %709, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %766 = fsub <4 x float> %540, %686, !dbg !27
  %767 = extractelement <4 x float> %766, i64 0, !dbg !31
  %768 = call float @llvm.amdgcn.exp2.f32(float %767), !dbg !31
  %local.reduce157 = fadd float %727, %768, !dbg !32, !frisk.local_reduction !1
  %769 = insertelement <4 x float> poison, float %768, i64 0, !dbg !33
  %770 = extractelement <4 x float> %766, i64 1, !dbg !31
  %771 = call float @llvm.amdgcn.exp2.f32(float %770), !dbg !31
  %local.reduce158 = fadd float %730, %771, !dbg !32, !frisk.local_reduction !1
  %772 = insertelement <4 x float> %769, float %771, i64 1, !dbg !33
  %773 = extractelement <4 x float> %766, i64 2, !dbg !31
  %774 = call float @llvm.amdgcn.exp2.f32(float %773), !dbg !31
  %local.reduce159 = fadd float %733, %774, !dbg !32, !frisk.local_reduction !1
  %775 = insertelement <4 x float> %772, float %774, i64 2, !dbg !33
  %776 = extractelement <4 x float> %766, i64 3, !dbg !31
  %777 = call float @llvm.amdgcn.exp2.f32(float %776), !dbg !31
  %local.reduce160 = fadd float %736, %777, !dbg !32, !frisk.local_reduction !1
  %778 = insertelement <4 x float> %775, float %777, i64 3, !dbg !33
  %math.result442 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %778) #9, !dbg !33, !frisk.register_lifetimes !1
  %779 = fptrunc <4 x float> %math.result442 to <4 x half>, !dbg !33
  %math.packed443 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %779) #9, !dbg !32, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %780 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed443, <4 x half> %765, <4 x float> %739) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %781 = shufflevector <8 x half> %709, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %782 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed443, <4 x half> %781, <4 x float> %743) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %783 = fsub <4 x float> %600, %687, !dbg !27
  %784 = extractelement <4 x float> %783, i64 0, !dbg !31
  %785 = call float @llvm.amdgcn.exp2.f32(float %784), !dbg !31
  %local.reduce = fadd float %749, %785, !dbg !32, !frisk.local_reduction !1
  %786 = insertelement <4 x float> poison, float %785, i64 0, !dbg !33
  %787 = extractelement <4 x float> %783, i64 1, !dbg !31
  %788 = call float @llvm.amdgcn.exp2.f32(float %787), !dbg !31
  %local.reduce127 = fadd float %752, %788, !dbg !32, !frisk.local_reduction !1
  %789 = insertelement <4 x float> %786, float %788, i64 1, !dbg !33
  %790 = extractelement <4 x float> %783, i64 2, !dbg !31
  %791 = call float @llvm.amdgcn.exp2.f32(float %790), !dbg !31
  %local.reduce128 = fadd float %755, %791, !dbg !32, !frisk.local_reduction !1
  %792 = insertelement <4 x float> %789, float %791, i64 2, !dbg !33
  %793 = extractelement <4 x float> %783, i64 3, !dbg !31
  %794 = call float @llvm.amdgcn.exp2.f32(float %793), !dbg !31
  %local.reduce129 = fadd float %758, %794, !dbg !32, !frisk.local_reduction !1
  %795 = insertelement <4 x float> %792, float %794, i64 3, !dbg !33
  %math.result458 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %795) #9, !dbg !33, !frisk.register_lifetimes !1
  %796 = fptrunc <4 x float> %math.result458 to <4 x half>, !dbg !33
  %math.packed459 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %796) #9, !dbg !32, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %797 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed459, <4 x half> %765, <4 x float> %761) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %798 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed459, <4 x half> %781, <4 x float> %764) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %799 = shufflevector <8 x half> %711, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %800 = fsub <4 x float> %551, %686, !dbg !27
  %801 = extractelement <4 x float> %800, i64 0, !dbg !31
  %802 = call float @llvm.amdgcn.exp2.f32(float %801), !dbg !31
  %local.reduce161 = fadd float %local.reduce157, %802, !dbg !32, !frisk.local_reduction !1
  %803 = insertelement <4 x float> poison, float %802, i64 0, !dbg !33
  %804 = extractelement <4 x float> %800, i64 1, !dbg !31
  %805 = call float @llvm.amdgcn.exp2.f32(float %804), !dbg !31
  %local.reduce162 = fadd float %local.reduce158, %805, !dbg !32, !frisk.local_reduction !1
  %806 = insertelement <4 x float> %803, float %805, i64 1, !dbg !33
  %807 = extractelement <4 x float> %800, i64 2, !dbg !31
  %808 = call float @llvm.amdgcn.exp2.f32(float %807), !dbg !31
  %local.reduce163 = fadd float %local.reduce159, %808, !dbg !32, !frisk.local_reduction !1
  %809 = insertelement <4 x float> %806, float %808, i64 2, !dbg !33
  %810 = extractelement <4 x float> %800, i64 3, !dbg !31
  %811 = call float @llvm.amdgcn.exp2.f32(float %810), !dbg !31
  %local.reduce164 = fadd float %local.reduce160, %811, !dbg !32, !frisk.local_reduction !1
  %812 = insertelement <4 x float> %809, float %811, i64 3, !dbg !33
  %math.result444 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %812) #9, !dbg !33, !frisk.register_lifetimes !1
  %813 = fptrunc <4 x float> %math.result444 to <4 x half>, !dbg !33
  %math.packed445 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %813) #9, !dbg !32, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %814 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed445, <4 x half> %799, <4 x float> %780) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %815 = shufflevector <8 x half> %711, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %816 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed445, <4 x half> %815, <4 x float> %782) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %817 = fsub <4 x float> %603, %687, !dbg !27
  %818 = extractelement <4 x float> %817, i64 0, !dbg !31
  %819 = call float @llvm.amdgcn.exp2.f32(float %818), !dbg !31
  %local.reduce130 = fadd float %local.reduce, %819, !dbg !32, !frisk.local_reduction !1
  %820 = insertelement <4 x float> poison, float %819, i64 0, !dbg !33
  %821 = extractelement <4 x float> %817, i64 1, !dbg !31
  %822 = call float @llvm.amdgcn.exp2.f32(float %821), !dbg !31
  %local.reduce131 = fadd float %local.reduce127, %822, !dbg !32, !frisk.local_reduction !1
  %823 = insertelement <4 x float> %820, float %822, i64 1, !dbg !33
  %824 = extractelement <4 x float> %817, i64 2, !dbg !31
  %825 = call float @llvm.amdgcn.exp2.f32(float %824), !dbg !31
  %local.reduce132 = fadd float %local.reduce128, %825, !dbg !32, !frisk.local_reduction !1
  %826 = insertelement <4 x float> %823, float %825, i64 2, !dbg !33
  %827 = extractelement <4 x float> %817, i64 3, !dbg !31
  %828 = call float @llvm.amdgcn.exp2.f32(float %827), !dbg !31
  %local.reduce133 = fadd float %local.reduce129, %828, !dbg !32, !frisk.local_reduction !1
  %829 = insertelement <4 x float> %826, float %828, i64 3, !dbg !33
  %math.result460 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %829) #9, !dbg !33, !frisk.register_lifetimes !1
  %830 = fptrunc <4 x float> %math.result460 to <4 x half>, !dbg !33
  %math.packed461 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %830) #9, !dbg !32, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %831 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed461, <4 x half> %799, <4 x float> %797) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %832 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed461, <4 x half> %815, <4 x float> %798) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %833 = shufflevector <8 x half> %713, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %834 = fsub <4 x float> %560, %686, !dbg !27
  %835 = extractelement <4 x float> %834, i64 0, !dbg !31
  %836 = call float @llvm.amdgcn.exp2.f32(float %835), !dbg !31
  %local.reduce165 = fadd float %local.reduce161, %836, !dbg !32, !frisk.local_reduction !1
  %837 = insertelement <4 x float> poison, float %836, i64 0, !dbg !33
  %838 = extractelement <4 x float> %834, i64 1, !dbg !31
  %839 = call float @llvm.amdgcn.exp2.f32(float %838), !dbg !31
  %local.reduce166 = fadd float %local.reduce162, %839, !dbg !32, !frisk.local_reduction !1
  %840 = insertelement <4 x float> %837, float %839, i64 1, !dbg !33
  %841 = extractelement <4 x float> %834, i64 2, !dbg !31
  %842 = call float @llvm.amdgcn.exp2.f32(float %841), !dbg !31
  %local.reduce167 = fadd float %local.reduce163, %842, !dbg !32, !frisk.local_reduction !1
  %843 = insertelement <4 x float> %840, float %842, i64 2, !dbg !33
  %844 = extractelement <4 x float> %834, i64 3, !dbg !31
  %845 = call float @llvm.amdgcn.exp2.f32(float %844), !dbg !31
  %local.reduce168 = fadd float %local.reduce164, %845, !dbg !32, !frisk.local_reduction !1
  %846 = insertelement <4 x float> %843, float %845, i64 3, !dbg !33
  %math.result446 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %846) #9, !dbg !33, !frisk.register_lifetimes !1
  %847 = fptrunc <4 x float> %math.result446 to <4 x half>, !dbg !33
  %math.packed447 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %847) #9, !dbg !32, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %848 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed447, <4 x half> %833, <4 x float> %814) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %849 = shufflevector <8 x half> %713, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %850 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed447, <4 x half> %849, <4 x float> %816) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %851 = fsub <4 x float> %606, %687, !dbg !27
  %852 = extractelement <4 x float> %851, i64 0, !dbg !31
  %853 = call float @llvm.amdgcn.exp2.f32(float %852), !dbg !31
  %local.reduce134 = fadd float %local.reduce130, %853, !dbg !32, !frisk.local_reduction !1
  %854 = insertelement <4 x float> poison, float %853, i64 0, !dbg !33
  %855 = extractelement <4 x float> %851, i64 1, !dbg !31
  %856 = call float @llvm.amdgcn.exp2.f32(float %855), !dbg !31
  %local.reduce135 = fadd float %local.reduce131, %856, !dbg !32, !frisk.local_reduction !1
  %857 = insertelement <4 x float> %854, float %856, i64 1, !dbg !33
  %858 = extractelement <4 x float> %851, i64 2, !dbg !31
  %859 = call float @llvm.amdgcn.exp2.f32(float %858), !dbg !31
  %local.reduce136 = fadd float %local.reduce132, %859, !dbg !32, !frisk.local_reduction !1
  %860 = insertelement <4 x float> %857, float %859, i64 2, !dbg !33
  %861 = extractelement <4 x float> %851, i64 3, !dbg !31
  %862 = call float @llvm.amdgcn.exp2.f32(float %861), !dbg !31
  %local.reduce137 = fadd float %local.reduce133, %862, !dbg !32, !frisk.local_reduction !1
  %863 = insertelement <4 x float> %860, float %862, i64 3, !dbg !33
  %math.result462 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %863) #9, !dbg !33, !frisk.register_lifetimes !1
  %864 = fptrunc <4 x float> %math.result462 to <4 x half>, !dbg !33
  %math.packed463 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %864) #9, !dbg !32, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %865 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed463, <4 x half> %833, <4 x float> %831) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %866 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed463, <4 x half> %849, <4 x float> %832) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %867 = shufflevector <8 x half> %715, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %868 = fsub <4 x float> %567, %686, !dbg !27
  %869 = extractelement <4 x float> %868, i64 0, !dbg !31
  %870 = call float @llvm.amdgcn.exp2.f32(float %869), !dbg !31
  %local.reduce169 = fadd float %local.reduce165, %870, !dbg !32, !frisk.local_reduction !1
  %871 = insertelement <4 x float> poison, float %870, i64 0, !dbg !33
  %872 = extractelement <4 x float> %868, i64 1, !dbg !31
  %873 = call float @llvm.amdgcn.exp2.f32(float %872), !dbg !31
  %local.reduce170 = fadd float %local.reduce166, %873, !dbg !32, !frisk.local_reduction !1
  %874 = insertelement <4 x float> %871, float %873, i64 1, !dbg !33
  %875 = extractelement <4 x float> %868, i64 2, !dbg !31
  %876 = call float @llvm.amdgcn.exp2.f32(float %875), !dbg !31
  %local.reduce171 = fadd float %local.reduce167, %876, !dbg !32, !frisk.local_reduction !1
  %877 = insertelement <4 x float> %874, float %876, i64 2, !dbg !33
  %878 = extractelement <4 x float> %868, i64 3, !dbg !31
  %879 = call float @llvm.amdgcn.exp2.f32(float %878), !dbg !31
  %local.reduce172 = fadd float %local.reduce168, %879, !dbg !32, !frisk.local_reduction !1
  %880 = insertelement <4 x float> %877, float %879, i64 3, !dbg !33
  %math.result448 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %880) #9, !dbg !33, !frisk.register_lifetimes !1
  %881 = fptrunc <4 x float> %math.result448 to <4 x half>, !dbg !33
  %math.packed449 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %881) #9, !dbg !32, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %882 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed449, <4 x half> %867, <4 x float> %848) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %883 = shufflevector <8 x half> %715, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %884 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed449, <4 x half> %883, <4 x float> %850) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %885 = fsub <4 x float> %609, %687, !dbg !27
  %886 = extractelement <4 x float> %885, i64 0, !dbg !31
  %887 = call float @llvm.amdgcn.exp2.f32(float %886), !dbg !31
  %local.reduce138 = fadd float %local.reduce134, %887, !dbg !32, !frisk.local_reduction !1
  %888 = insertelement <4 x float> poison, float %887, i64 0, !dbg !33
  %889 = extractelement <4 x float> %885, i64 1, !dbg !31
  %890 = call float @llvm.amdgcn.exp2.f32(float %889), !dbg !31
  %local.reduce139 = fadd float %local.reduce135, %890, !dbg !32, !frisk.local_reduction !1
  %891 = insertelement <4 x float> %888, float %890, i64 1, !dbg !33
  %892 = extractelement <4 x float> %885, i64 2, !dbg !31
  %893 = call float @llvm.amdgcn.exp2.f32(float %892), !dbg !31
  %local.reduce140 = fadd float %local.reduce136, %893, !dbg !32, !frisk.local_reduction !1
  %894 = insertelement <4 x float> %891, float %893, i64 2, !dbg !33
  %895 = extractelement <4 x float> %885, i64 3, !dbg !31
  %896 = call float @llvm.amdgcn.exp2.f32(float %895), !dbg !31
  %local.reduce141 = fadd float %local.reduce137, %896, !dbg !32, !frisk.local_reduction !1
  %897 = insertelement <4 x float> %894, float %896, i64 3, !dbg !33
  %math.result464 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %897) #9, !dbg !33, !frisk.register_lifetimes !1
  %898 = fptrunc <4 x float> %math.result464 to <4 x half>, !dbg !33
  %math.packed465 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %898) #9, !dbg !32, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %899 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed465, <4 x half> %867, <4 x float> %865) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %900 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed465, <4 x half> %883, <4 x float> %866) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %901 = shufflevector <8 x half> %717, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %902 = fsub <4 x float> %575, %686, !dbg !27
  %903 = extractelement <4 x float> %902, i64 0, !dbg !31
  %904 = call float @llvm.amdgcn.exp2.f32(float %903), !dbg !31
  %local.reduce173 = fadd float %local.reduce169, %904, !dbg !32, !frisk.local_reduction !1
  %905 = insertelement <4 x float> poison, float %904, i64 0, !dbg !33
  %906 = extractelement <4 x float> %902, i64 1, !dbg !31
  %907 = call float @llvm.amdgcn.exp2.f32(float %906), !dbg !31
  %local.reduce174 = fadd float %local.reduce170, %907, !dbg !32, !frisk.local_reduction !1
  %908 = insertelement <4 x float> %905, float %907, i64 1, !dbg !33
  %909 = extractelement <4 x float> %902, i64 2, !dbg !31
  %910 = call float @llvm.amdgcn.exp2.f32(float %909), !dbg !31
  %local.reduce175 = fadd float %local.reduce171, %910, !dbg !32, !frisk.local_reduction !1
  %911 = insertelement <4 x float> %908, float %910, i64 2, !dbg !33
  %912 = extractelement <4 x float> %902, i64 3, !dbg !31
  %913 = call float @llvm.amdgcn.exp2.f32(float %912), !dbg !31
  %local.reduce176 = fadd float %local.reduce172, %913, !dbg !32, !frisk.local_reduction !1
  %914 = insertelement <4 x float> %911, float %913, i64 3, !dbg !33
  %math.result450 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %914) #9, !dbg !33, !frisk.register_lifetimes !1
  %915 = fptrunc <4 x float> %math.result450 to <4 x half>, !dbg !33
  %math.packed451 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %915) #9, !dbg !32, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %916 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed451, <4 x half> %901, <4 x float> %882) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %917 = shufflevector <8 x half> %717, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %918 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed451, <4 x half> %917, <4 x float> %884) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %919 = fsub <4 x float> %612, %687, !dbg !27
  %920 = extractelement <4 x float> %919, i64 0, !dbg !31
  %921 = call float @llvm.amdgcn.exp2.f32(float %920), !dbg !31
  %local.reduce142 = fadd float %local.reduce138, %921, !dbg !32, !frisk.local_reduction !1
  %922 = insertelement <4 x float> poison, float %921, i64 0, !dbg !33
  %923 = extractelement <4 x float> %919, i64 1, !dbg !31
  %924 = call float @llvm.amdgcn.exp2.f32(float %923), !dbg !31
  %local.reduce143 = fadd float %local.reduce139, %924, !dbg !32, !frisk.local_reduction !1
  %925 = insertelement <4 x float> %922, float %924, i64 1, !dbg !33
  %926 = extractelement <4 x float> %919, i64 2, !dbg !31
  %927 = call float @llvm.amdgcn.exp2.f32(float %926), !dbg !31
  %local.reduce144 = fadd float %local.reduce140, %927, !dbg !32, !frisk.local_reduction !1
  %928 = insertelement <4 x float> %925, float %927, i64 2, !dbg !33
  %929 = extractelement <4 x float> %919, i64 3, !dbg !31
  %930 = call float @llvm.amdgcn.exp2.f32(float %929), !dbg !31
  %local.reduce145 = fadd float %local.reduce141, %930, !dbg !32, !frisk.local_reduction !1
  %931 = insertelement <4 x float> %928, float %930, i64 3, !dbg !33
  %math.result466 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %931) #9, !dbg !33, !frisk.register_lifetimes !1
  %932 = fptrunc <4 x float> %math.result466 to <4 x half>, !dbg !33
  %math.packed467 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %932) #9, !dbg !32, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %933 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed467, <4 x half> %901, <4 x float> %899) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %934 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed467, <4 x half> %917, <4 x float> %900) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %935 = shufflevector <8 x half> %719, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %936 = fsub <4 x float> %582, %686, !dbg !27
  %937 = extractelement <4 x float> %936, i64 0, !dbg !31
  %938 = call float @llvm.amdgcn.exp2.f32(float %937), !dbg !31
  %local.reduce177 = fadd float %local.reduce173, %938, !dbg !32, !frisk.local_reduction !1
  %939 = insertelement <4 x float> poison, float %938, i64 0, !dbg !33
  %940 = extractelement <4 x float> %936, i64 1, !dbg !31
  %941 = call float @llvm.amdgcn.exp2.f32(float %940), !dbg !31
  %local.reduce178 = fadd float %local.reduce174, %941, !dbg !32, !frisk.local_reduction !1
  %942 = insertelement <4 x float> %939, float %941, i64 1, !dbg !33
  %943 = extractelement <4 x float> %936, i64 2, !dbg !31
  %944 = call float @llvm.amdgcn.exp2.f32(float %943), !dbg !31
  %local.reduce179 = fadd float %local.reduce175, %944, !dbg !32, !frisk.local_reduction !1
  %945 = insertelement <4 x float> %942, float %944, i64 2, !dbg !33
  %946 = extractelement <4 x float> %936, i64 3, !dbg !31
  %947 = call float @llvm.amdgcn.exp2.f32(float %946), !dbg !31
  %local.reduce180 = fadd float %local.reduce176, %947, !dbg !32, !frisk.local_reduction !1
  %948 = insertelement <4 x float> %945, float %947, i64 3, !dbg !33
  %math.result452 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %948) #9, !dbg !33, !frisk.register_lifetimes !1
  %949 = fptrunc <4 x float> %math.result452 to <4 x half>, !dbg !33
  %math.packed453 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %949) #9, !dbg !32, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %950 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed453, <4 x half> %935, <4 x float> %916) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %951 = shufflevector <8 x half> %719, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %952 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed453, <4 x half> %951, <4 x float> %918) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %953 = fsub <4 x float> %615, %687, !dbg !27
  %954 = extractelement <4 x float> %953, i64 0, !dbg !31
  %955 = call float @llvm.amdgcn.exp2.f32(float %954), !dbg !31
  %local.reduce146 = fadd float %local.reduce142, %955, !dbg !32, !frisk.local_reduction !1
  %956 = insertelement <4 x float> poison, float %955, i64 0, !dbg !33
  %957 = extractelement <4 x float> %953, i64 1, !dbg !31
  %958 = call float @llvm.amdgcn.exp2.f32(float %957), !dbg !31
  %local.reduce147 = fadd float %local.reduce143, %958, !dbg !32, !frisk.local_reduction !1
  %959 = insertelement <4 x float> %956, float %958, i64 1, !dbg !33
  %960 = extractelement <4 x float> %953, i64 2, !dbg !31
  %961 = call float @llvm.amdgcn.exp2.f32(float %960), !dbg !31
  %local.reduce148 = fadd float %local.reduce144, %961, !dbg !32, !frisk.local_reduction !1
  %962 = insertelement <4 x float> %959, float %961, i64 2, !dbg !33
  %963 = extractelement <4 x float> %953, i64 3, !dbg !31
  %964 = call float @llvm.amdgcn.exp2.f32(float %963), !dbg !31
  %local.reduce149 = fadd float %local.reduce145, %964, !dbg !32, !frisk.local_reduction !1
  %965 = insertelement <4 x float> %962, float %964, i64 3, !dbg !33
  %math.result468 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %965) #9, !dbg !33, !frisk.register_lifetimes !1
  %966 = fptrunc <4 x float> %math.result468 to <4 x half>, !dbg !33
  %math.packed469 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %966) #9, !dbg !32, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %967 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed469, <4 x half> %935, <4 x float> %933) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %968 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed469, <4 x half> %951, <4 x float> %934) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %969 = bitcast <4 x i32> %720 to <8 x half>, !dbg !13
  %970 = shufflevector <8 x half> %969, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %971 = fsub <4 x float> %589, %686, !dbg !27
  %972 = extractelement <4 x float> %971, i64 0, !dbg !31
  %973 = call float @llvm.amdgcn.exp2.f32(float %972), !dbg !31
  %local.reduce181 = fadd float %local.reduce177, %973, !dbg !32, !frisk.local_reduction !1
  %974 = insertelement <4 x float> poison, float %973, i64 0, !dbg !33
  %975 = extractelement <4 x float> %971, i64 1, !dbg !31
  %976 = call float @llvm.amdgcn.exp2.f32(float %975), !dbg !31
  %local.reduce182 = fadd float %local.reduce178, %976, !dbg !32, !frisk.local_reduction !1
  %local.reduce185 = fadd float %local.reduce181, %local.reduce182, !dbg !32, !frisk.local_reduction !1
  %977 = insertelement <4 x float> %974, float %976, i64 1, !dbg !33
  %978 = extractelement <4 x float> %971, i64 2, !dbg !31
  %979 = call float @llvm.amdgcn.exp2.f32(float %978), !dbg !31
  %local.reduce183 = fadd float %local.reduce179, %979, !dbg !32, !frisk.local_reduction !1
  %980 = insertelement <4 x float> %977, float %979, i64 2, !dbg !33
  %981 = extractelement <4 x float> %971, i64 3, !dbg !31
  %982 = call float @llvm.amdgcn.exp2.f32(float %981), !dbg !31
  %local.reduce184 = fadd float %local.reduce180, %982, !dbg !32, !frisk.local_reduction !1
  %local.reduce186 = fadd float %local.reduce183, %local.reduce184, !dbg !32, !frisk.local_reduction !1
  %983 = fadd float %local.reduce185, %local.reduce186, !dbg !32, !frisk.local_reduction !1
  %984 = fadd float %983, %695, !dbg !32
  %985 = insertelement <1 x float> poison, float %984, i64 0, !dbg !7
  %986 = insertvalue [2 x <1 x float>] poison, <1 x float> %985, 0, !dbg !7
  %987 = insertelement <4 x float> %980, float %982, i64 3, !dbg !33
  %math.result454 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %987) #9, !dbg !33, !frisk.register_lifetimes !1
  %988 = fptrunc <4 x float> %math.result454 to <4 x half>, !dbg !33
  %math.packed455 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %988) #9, !dbg !32, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %989 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed455, <4 x half> %970, <4 x float> %950) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %990 = shufflevector <8 x half> %969, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %991 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed455, <4 x half> %990, <4 x float> %952) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %992 = fsub <4 x float> %629, %687, !dbg !27
  %993 = extractelement <4 x float> %992, i64 0, !dbg !31
  %994 = call float @llvm.amdgcn.exp2.f32(float %993), !dbg !31
  %local.reduce150 = fadd float %local.reduce146, %994, !dbg !32, !frisk.local_reduction !1
  %995 = insertelement <4 x float> poison, float %994, i64 0, !dbg !33
  %996 = extractelement <4 x float> %992, i64 1, !dbg !31
  %997 = call float @llvm.amdgcn.exp2.f32(float %996), !dbg !31
  %local.reduce151 = fadd float %local.reduce147, %997, !dbg !32, !frisk.local_reduction !1
  %local.reduce154 = fadd float %local.reduce150, %local.reduce151, !dbg !32, !frisk.local_reduction !1
  %998 = insertelement <4 x float> %995, float %997, i64 1, !dbg !33
  %999 = extractelement <4 x float> %992, i64 2, !dbg !31
  %1000 = call float @llvm.amdgcn.exp2.f32(float %999), !dbg !31
  %local.reduce152 = fadd float %local.reduce148, %1000, !dbg !32, !frisk.local_reduction !1
  %1001 = insertelement <4 x float> %998, float %1000, i64 2, !dbg !33
  %1002 = extractelement <4 x float> %992, i64 3, !dbg !31
  %1003 = call float @llvm.amdgcn.exp2.f32(float %1002), !dbg !31
  %local.reduce153 = fadd float %local.reduce149, %1003, !dbg !32, !frisk.local_reduction !1
  %local.reduce155 = fadd float %local.reduce152, %local.reduce153, !dbg !32, !frisk.local_reduction !1
  %1004 = fadd float %local.reduce154, %local.reduce155, !dbg !32, !frisk.local_reduction !1
  %1005 = fadd float %1004, %703, !dbg !32
  %1006 = insertelement <1 x float> poison, float %1005, i64 0, !dbg !7
  %1007 = insertvalue [2 x <1 x float>] %986, <1 x float> %1006, 1, !dbg !7
  %1008 = insertelement <4 x float> %1001, float %1003, i64 3, !dbg !33
  %math.result470 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1008) #9, !dbg !33, !frisk.register_lifetimes !1
  %1009 = fptrunc <4 x float> %math.result470 to <4 x half>, !dbg !33
  %math.packed471 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1009) #9, !dbg !32, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %1010 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed471, <4 x half> %970, <4 x float> %967) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1011 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed471, <4 x half> %990, <4 x float> %968) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %vdb.address1830 = or disjoint i64 %.idx17771779, 64, !dbg !16
  %vdb.address1831 = or disjoint i64 %address.local989, %vdb.address1830, !dbg !16
  %vdb.address1832 = trunc i64 %vdb.address1831 to i32, !dbg !16
  %vdb.address1833 = shl i32 %vdb.address1832, 1, !dbg !16
  %vdb.address1841 = or disjoint i64 %address.local995, %vdb.address1830, !dbg !16
  %vdb.address1842 = trunc i64 %vdb.address1841 to i32, !dbg !16
  %vdb.address1843 = shl i32 %vdb.address1842, 1, !dbg !16
  %1012 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1702, i32 %vdb.address1833, <4 x i32> %dma.buffer.rsrc1677) #10, !dbg !16
  %1013 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1699, i32 %vdb.address1843, <4 x i32> %dma.buffer.rsrc1677) #10, !dbg !16
  %vdb.read = getelementptr i8, ptr addrspace(3) %address.local1003, i64 8192, !dbg !13
  %1014 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #0, !dbg !13, !frisk.v_double_buffer !40
  %1015 = bitcast <4 x i32> %1014 to <8 x half>, !dbg !13
  %vdb.read1821 = getelementptr i8, ptr addrspace(3) %address.local1012, i64 8448, !dbg !13
  %1016 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1821, i32 0) #0, !dbg !13, !frisk.v_double_buffer !40
  %1017 = bitcast <4 x i32> %1016 to <8 x half>, !dbg !13
  %vdb.read1822 = getelementptr i8, ptr addrspace(3) %address.local1012, i64 8704, !dbg !13
  %1018 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1822, i32 0) #0, !dbg !13, !frisk.v_double_buffer !40
  %1019 = bitcast <4 x i32> %1018 to <8 x half>, !dbg !13
  %vdb.read1823 = getelementptr i8, ptr addrspace(3) %address.local1012, i64 8960, !dbg !13
  %1020 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1823, i32 0) #0, !dbg !13, !frisk.v_double_buffer !40
  %1021 = bitcast <4 x i32> %1020 to <8 x half>, !dbg !13
  %vdb.read1824 = getelementptr i8, ptr addrspace(3) %address.local1003, i64 12288, !dbg !13
  %1022 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1824, i32 0) #0, !dbg !13, !frisk.v_double_buffer !40
  %1023 = bitcast <4 x i32> %1022 to <8 x half>, !dbg !13
  %vdb.read1825 = getelementptr i8, ptr addrspace(3) %address.local1012, i64 12544, !dbg !13
  %1024 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1825, i32 0) #0, !dbg !13, !frisk.v_double_buffer !40
  %1025 = bitcast <4 x i32> %1024 to <8 x half>, !dbg !13
  %vdb.read1826 = getelementptr i8, ptr addrspace(3) %address.local1012, i64 12800, !dbg !13
  %1026 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1826, i32 0) #0, !dbg !13, !frisk.v_double_buffer !40
  %1027 = bitcast <4 x i32> %1026 to <8 x half>, !dbg !13
  %vdb.read1827 = getelementptr i8, ptr addrspace(3) %address.local1012, i64 13056, !dbg !13
  %1028 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1827, i32 0) #0, !dbg !13, !frisk.v_double_buffer !40
  %1029 = shufflevector <8 x half> %1015, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1030 = extractvalue [2 x <8 x float>] %110, 0, !dbg !34
  %1031 = shufflevector <8 x float> %1030, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %1032 = fmul <4 x float> %1031, %692, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1033 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1029, <4 x float> %1032) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1034 = shufflevector <8 x half> %1015, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1035 = shufflevector <8 x float> %1030, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %1036 = fmul <4 x float> %1035, %692, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1037 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1034, <4 x float> %1036) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !34
  %1038 = extractvalue [2 x <8 x float>] %110, 1, !dbg !34
  %1039 = shufflevector <8 x float> %1038, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %1040 = fmul <4 x float> %1039, %700, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1041 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed457, <4 x half> %1029, <4 x float> %1040) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !34
  %1042 = shufflevector <8 x float> %1038, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %1043 = fmul <4 x float> %1042, %700, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1044 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed457, <4 x half> %1034, <4 x float> %1043) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1045 = shufflevector <8 x half> %1017, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1046 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed443, <4 x half> %1045, <4 x float> %1033) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1047 = shufflevector <8 x half> %1017, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1048 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed443, <4 x half> %1047, <4 x float> %1037) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1049 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed459, <4 x half> %1045, <4 x float> %1041) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1050 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed459, <4 x half> %1047, <4 x float> %1044) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1051 = shufflevector <8 x half> %1019, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1052 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed445, <4 x half> %1051, <4 x float> %1046) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1053 = shufflevector <8 x half> %1019, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1054 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed445, <4 x half> %1053, <4 x float> %1048) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1055 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed461, <4 x half> %1051, <4 x float> %1049) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1056 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed461, <4 x half> %1053, <4 x float> %1050) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1057 = shufflevector <8 x half> %1021, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1058 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed447, <4 x half> %1057, <4 x float> %1052) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1059 = shufflevector <8 x half> %1021, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1060 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed447, <4 x half> %1059, <4 x float> %1054) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1061 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed463, <4 x half> %1057, <4 x float> %1055) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1062 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed463, <4 x half> %1059, <4 x float> %1056) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1063 = shufflevector <8 x half> %1023, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1064 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed449, <4 x half> %1063, <4 x float> %1058) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1065 = shufflevector <8 x half> %1023, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1066 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed449, <4 x half> %1065, <4 x float> %1060) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1067 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed465, <4 x half> %1063, <4 x float> %1061) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1068 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed465, <4 x half> %1065, <4 x float> %1062) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1069 = shufflevector <8 x half> %1025, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1070 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed451, <4 x half> %1069, <4 x float> %1064) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1071 = shufflevector <8 x half> %1025, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1072 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed451, <4 x half> %1071, <4 x float> %1066) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1073 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed467, <4 x half> %1069, <4 x float> %1067) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1074 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed467, <4 x half> %1071, <4 x float> %1068) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1075 = shufflevector <8 x half> %1027, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1076 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed453, <4 x half> %1075, <4 x float> %1070) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1077 = shufflevector <8 x half> %1027, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1078 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed453, <4 x half> %1077, <4 x float> %1072) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1079 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed469, <4 x half> %1075, <4 x float> %1073) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1080 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed469, <4 x half> %1077, <4 x float> %1074) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1081 = bitcast <4 x i32> %1028 to <8 x half>, !dbg !13
  %1082 = shufflevector <8 x half> %1081, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1083 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed455, <4 x half> %1082, <4 x float> %1076) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1084 = shufflevector <8 x half> %1081, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1085 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed455, <4 x half> %1084, <4 x float> %1078) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1086 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed471, <4 x half> %1082, <4 x float> %1079) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1087 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed471, <4 x half> %1084, <4 x float> %1080) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %vdb.address1850 = or disjoint i64 %.idx17771779, 96, !dbg !16
  %vdb.address1851 = or disjoint i64 %address.local989, %vdb.address1850, !dbg !16
  %vdb.address1852 = trunc i64 %vdb.address1851 to i32, !dbg !16
  %vdb.address1853 = shl i32 %vdb.address1852, 1, !dbg !16
  %vdb.address1861 = or disjoint i64 %address.local995, %vdb.address1850, !dbg !16
  %vdb.address1862 = trunc i64 %vdb.address1861 to i32, !dbg !16
  %vdb.address1863 = shl i32 %vdb.address1862, 1, !dbg !16
  %1088 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1853, <4 x i32> %dma.buffer.rsrc1677) #10, !dbg !16
  %1089 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1820, i32 %vdb.address1863, <4 x i32> %dma.buffer.rsrc1677) #10, !dbg !16
  %1090 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1003, i32 0) #0, !dbg !13, !frisk.v_double_buffer !40
  %1091 = bitcast <4 x i32> %1090 to <8 x half>, !dbg !13
  %1092 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1013, i32 0) #0, !dbg !13, !frisk.v_double_buffer !40
  %1093 = bitcast <4 x i32> %1092 to <8 x half>, !dbg !13
  %1094 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1017, i32 0) #0, !dbg !13, !frisk.v_double_buffer !40
  %1095 = bitcast <4 x i32> %1094 to <8 x half>, !dbg !13
  %1096 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1021, i32 0) #0, !dbg !13, !frisk.v_double_buffer !40
  %1097 = bitcast <4 x i32> %1096 to <8 x half>, !dbg !13
  %1098 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1023, i32 0) #0, !dbg !13, !frisk.v_double_buffer !40
  %1099 = bitcast <4 x i32> %1098 to <8 x half>, !dbg !13
  %1100 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1027, i32 0) #0, !dbg !13, !frisk.v_double_buffer !40
  %1101 = bitcast <4 x i32> %1100 to <8 x half>, !dbg !13
  %1102 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1031, i32 0) #0, !dbg !13, !frisk.v_double_buffer !40
  %1103 = bitcast <4 x i32> %1102 to <8 x half>, !dbg !13
  %1104 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1035, i32 0) #0, !dbg !13, !frisk.v_double_buffer !40
  %1105 = shufflevector <8 x half> %1091, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1106 = extractvalue [2 x <8 x float>] %111, 0, !dbg !34
  %1107 = shufflevector <8 x float> %1106, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %1108 = fmul <4 x float> %1107, %692, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1109 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1105, <4 x float> %1108) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1110 = shufflevector <8 x half> %1091, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1111 = shufflevector <8 x float> %1106, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %1112 = fmul <4 x float> %1111, %692, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1113 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1110, <4 x float> %1112) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !34
  %1114 = extractvalue [2 x <8 x float>] %111, 1, !dbg !34
  %1115 = shufflevector <8 x float> %1114, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %1116 = fmul <4 x float> %1115, %700, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1117 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed457, <4 x half> %1105, <4 x float> %1116) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !34
  %1118 = shufflevector <8 x float> %1114, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %1119 = fmul <4 x float> %1118, %700, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1120 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed457, <4 x half> %1110, <4 x float> %1119) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1121 = shufflevector <8 x half> %1093, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1122 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed443, <4 x half> %1121, <4 x float> %1109) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1123 = shufflevector <8 x half> %1093, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1124 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed443, <4 x half> %1123, <4 x float> %1113) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1125 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed459, <4 x half> %1121, <4 x float> %1117) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1126 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed459, <4 x half> %1123, <4 x float> %1120) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1127 = shufflevector <8 x half> %1095, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1128 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed445, <4 x half> %1127, <4 x float> %1122) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1129 = shufflevector <8 x half> %1095, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1130 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed445, <4 x half> %1129, <4 x float> %1124) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1131 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed461, <4 x half> %1127, <4 x float> %1125) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1132 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed461, <4 x half> %1129, <4 x float> %1126) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1133 = shufflevector <8 x half> %1097, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1134 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed447, <4 x half> %1133, <4 x float> %1128) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1135 = shufflevector <8 x half> %1097, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1136 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed447, <4 x half> %1135, <4 x float> %1130) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1137 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed463, <4 x half> %1133, <4 x float> %1131) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1138 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed463, <4 x half> %1135, <4 x float> %1132) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1139 = shufflevector <8 x half> %1099, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1140 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed449, <4 x half> %1139, <4 x float> %1134) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1141 = shufflevector <8 x half> %1099, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1142 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed449, <4 x half> %1141, <4 x float> %1136) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1143 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed465, <4 x half> %1139, <4 x float> %1137) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1144 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed465, <4 x half> %1141, <4 x float> %1138) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1145 = shufflevector <8 x half> %1101, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1146 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed451, <4 x half> %1145, <4 x float> %1140) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1147 = shufflevector <8 x half> %1101, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1148 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed451, <4 x half> %1147, <4 x float> %1142) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1149 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed467, <4 x half> %1145, <4 x float> %1143) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1150 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed467, <4 x half> %1147, <4 x float> %1144) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1151 = shufflevector <8 x half> %1103, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1152 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed453, <4 x half> %1151, <4 x float> %1146) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1153 = shufflevector <8 x half> %1103, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1154 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed453, <4 x half> %1153, <4 x float> %1148) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1155 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed469, <4 x half> %1151, <4 x float> %1149) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1156 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed469, <4 x half> %1153, <4 x float> %1150) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1157 = bitcast <4 x i32> %1104 to <8 x half>, !dbg !13
  %1158 = shufflevector <8 x half> %1157, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1159 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed455, <4 x half> %1158, <4 x float> %1152) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1160 = shufflevector <8 x half> %1157, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1161 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed455, <4 x half> %1160, <4 x float> %1154) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1162 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed471, <4 x half> %1158, <4 x float> %1155) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1163 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed471, <4 x half> %1160, <4 x float> %1156) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %1164 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #0, !dbg !13, !frisk.v_double_buffer !40
  %1165 = bitcast <4 x i32> %1164 to <8 x half>, !dbg !13
  %1166 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1821, i32 0) #0, !dbg !13, !frisk.v_double_buffer !40
  %1167 = bitcast <4 x i32> %1166 to <8 x half>, !dbg !13
  %1168 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1822, i32 0) #0, !dbg !13, !frisk.v_double_buffer !40
  %1169 = bitcast <4 x i32> %1168 to <8 x half>, !dbg !13
  %1170 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1823, i32 0) #0, !dbg !13, !frisk.v_double_buffer !40
  %1171 = bitcast <4 x i32> %1170 to <8 x half>, !dbg !13
  %1172 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1824, i32 0) #0, !dbg !13, !frisk.v_double_buffer !40
  %1173 = bitcast <4 x i32> %1172 to <8 x half>, !dbg !13
  %1174 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1825, i32 0) #0, !dbg !13, !frisk.v_double_buffer !40
  %1175 = bitcast <4 x i32> %1174 to <8 x half>, !dbg !13
  %1176 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1826, i32 0) #0, !dbg !13, !frisk.v_double_buffer !40
  %1177 = bitcast <4 x i32> %1176 to <8 x half>, !dbg !13
  %1178 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1827, i32 0) #0, !dbg !13, !frisk.v_double_buffer !40
  %1179 = shufflevector <8 x half> %1165, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1180 = extractvalue [2 x <8 x float>] %112, 0, !dbg !34
  %1181 = shufflevector <8 x float> %1180, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %1182 = fmul <4 x float> %1181, %692, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1183 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1179, <4 x float> %1182) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1184 = shufflevector <8 x half> %1165, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1185 = shufflevector <8 x float> %1180, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %1186 = fmul <4 x float> %1185, %692, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1187 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1184, <4 x float> %1186) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !34
  %1188 = extractvalue [2 x <8 x float>] %112, 1, !dbg !34
  %1189 = shufflevector <8 x float> %1188, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %1190 = fmul <4 x float> %1189, %700, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1191 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed457, <4 x half> %1179, <4 x float> %1190) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !34
  %1192 = shufflevector <8 x float> %1188, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %1193 = fmul <4 x float> %1192, %700, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1194 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed457, <4 x half> %1184, <4 x float> %1193) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1195 = shufflevector <8 x half> %1167, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1196 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed443, <4 x half> %1195, <4 x float> %1183) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1197 = shufflevector <8 x half> %1167, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1198 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed443, <4 x half> %1197, <4 x float> %1187) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1199 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed459, <4 x half> %1195, <4 x float> %1191) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1200 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed459, <4 x half> %1197, <4 x float> %1194) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1201 = shufflevector <8 x half> %1169, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1202 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed445, <4 x half> %1201, <4 x float> %1196) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1203 = shufflevector <8 x half> %1169, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1204 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed445, <4 x half> %1203, <4 x float> %1198) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1205 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed461, <4 x half> %1201, <4 x float> %1199) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1206 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed461, <4 x half> %1203, <4 x float> %1200) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1207 = shufflevector <8 x half> %1171, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1208 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed447, <4 x half> %1207, <4 x float> %1202) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1209 = shufflevector <8 x half> %1171, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1210 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed447, <4 x half> %1209, <4 x float> %1204) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1211 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed463, <4 x half> %1207, <4 x float> %1205) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1212 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed463, <4 x half> %1209, <4 x float> %1206) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1213 = shufflevector <8 x half> %1173, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1214 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed449, <4 x half> %1213, <4 x float> %1208) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1215 = shufflevector <8 x half> %1173, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1216 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed449, <4 x half> %1215, <4 x float> %1210) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1217 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed465, <4 x half> %1213, <4 x float> %1211) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1218 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed465, <4 x half> %1215, <4 x float> %1212) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1219 = shufflevector <8 x half> %1175, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1220 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed451, <4 x half> %1219, <4 x float> %1214) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1221 = shufflevector <8 x half> %1175, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1222 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed451, <4 x half> %1221, <4 x float> %1216) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1223 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed467, <4 x half> %1219, <4 x float> %1217) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1224 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed467, <4 x half> %1221, <4 x float> %1218) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1225 = shufflevector <8 x half> %1177, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1226 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed453, <4 x half> %1225, <4 x float> %1220) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1227 = shufflevector <8 x half> %1177, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1228 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed453, <4 x half> %1227, <4 x float> %1222) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1229 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed469, <4 x half> %1225, <4 x float> %1223) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1230 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed469, <4 x half> %1227, <4 x float> %1224) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1231 = bitcast <4 x i32> %1178 to <8 x half>, !dbg !13
  %1232 = shufflevector <8 x half> %1231, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1233 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed455, <4 x half> %1232, <4 x float> %1226) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1234 = shufflevector <8 x half> %1231, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1235 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed455, <4 x half> %1234, <4 x float> %1228) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1236 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed471, <4 x half> %1232, <4 x float> %1229) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1237 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed471, <4 x half> %1234, <4 x float> %1230) #9, !dbg !13, !frisk.pv_interleave !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !7
  %1238 = shufflevector <4 x float> %989, <4 x float> %991, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1239 = insertvalue [2 x <8 x float>] poison, <8 x float> %1238, 0, !dbg !7
  %1240 = shufflevector <4 x float> %1010, <4 x float> %1011, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1241 = insertvalue [2 x <8 x float>] %1239, <8 x float> %1240, 1, !dbg !7
  %1242 = insertvalue [2 x <1 x float>] poison, <1 x float> %672, 0, !dbg !7
  %1243 = insertvalue [2 x <1 x float>] %1242, <1 x float> %685, 1, !dbg !7
  %1244 = shufflevector <4 x float> %1083, <4 x float> %1085, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1245 = insertvalue [2 x <8 x float>] poison, <8 x float> %1244, 0, !dbg !7
  %1246 = shufflevector <4 x float> %1086, <4 x float> %1087, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1247 = insertvalue [2 x <8 x float>] %1245, <8 x float> %1246, 1, !dbg !7
  %1248 = shufflevector <4 x float> %1159, <4 x float> %1161, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1249 = insertvalue [2 x <8 x float>] poison, <8 x float> %1248, 0, !dbg !7
  %1250 = shufflevector <4 x float> %1162, <4 x float> %1163, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1251 = insertvalue [2 x <8 x float>] %1249, <8 x float> %1250, 1, !dbg !7
  %1252 = shufflevector <4 x float> %1233, <4 x float> %1235, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1253 = insertvalue [2 x <8 x float>] poison, <8 x float> %1252, 0, !dbg !7
  %1254 = shufflevector <4 x float> %1236, <4 x float> %1237, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1255 = insertvalue [2 x <8 x float>] %1253, <8 x float> %1254, 1, !dbg !7
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %1256 = add nuw nsw i64 %106, 1, !dbg !7
  br label %105, !dbg !7

1257:                                             ; preds = %105
  %lane.sum.field252 = extractvalue [2 x <1 x float>] %108, 1, !dbg !7
  %lane.sum.local253 = extractelement <1 x float> %lane.sum.field252, i64 0, !dbg !7
  %lane.sum.lane.lo254 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !7
  %lane.sum.lane255 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %lane.sum.lane.lo254), !dbg !7
  %1258 = shl nuw nsw i32 %lane.sum.lane255, 2, !dbg !7
  %1259 = xor i32 %1258, 64, !dbg !7
  %1260 = bitcast float %lane.sum.local253 to i32, !dbg !7
  %lane.sum.shuffle256 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1259, i32 %1260), !dbg !7
  %1261 = bitcast i32 %lane.sum.shuffle256 to float, !dbg !7
  %lane.sum.final257 = fadd float %lane.sum.local253, %1261, !dbg !7
  %1262 = xor i32 %1258, 128, !dbg !7
  %1263 = bitcast float %lane.sum.final257 to i32, !dbg !7
  %lane.sum.shuffle258 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1262, i32 %1263), !dbg !7
  %1264 = bitcast i32 %lane.sum.shuffle258 to float, !dbg !7
  %lane.sum.final259 = fadd float %lane.sum.final257, %1264, !dbg !7
  %1265 = insertelement <1 x float> poison, float %lane.sum.final259, i64 0, !dbg !7
  %lane.sum.field = extractvalue [2 x <1 x float>] %108, 0, !dbg !7
  %lane.sum.local = extractelement <1 x float> %lane.sum.field, i64 0, !dbg !7
  %1266 = bitcast float %lane.sum.local to i32, !dbg !7
  %lane.sum.shuffle = call i32 @llvm.amdgcn.ds.bpermute(i32 %1259, i32 %1266), !dbg !7
  %1267 = bitcast i32 %lane.sum.shuffle to float, !dbg !7
  %lane.sum.final = fadd float %lane.sum.local, %1267, !dbg !7
  %1268 = bitcast float %lane.sum.final to i32, !dbg !7
  %lane.sum.shuffle250 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1262, i32 %1268), !dbg !7
  %1269 = bitcast i32 %lane.sum.shuffle250 to float, !dbg !7
  %lane.sum.final251 = fadd float %lane.sum.final, %1269, !dbg !7
  %1270 = insertelement <1 x float> poison, float %lane.sum.final251, i64 0, !dbg !7
  %1271 = and i32 %kdma.tid, 16, !dbg !7
  %.not36 = icmp eq i32 %1271, 0, !dbg !7
  %1272 = and i32 %kdma.tid, 32, !dbg !7
  %.not37 = icmp eq i32 %1272, 0, !dbg !7
  %1273 = extractvalue [2 x <8 x float>] %112, 0, !dbg !7
  %1274 = extractelement <8 x float> %1273, i64 0, !dbg !7
  %1275 = extractelement <8 x float> %1273, i64 1, !dbg !7
  %1276 = extractelement <8 x float> %1273, i64 2, !dbg !7
  %1277 = extractelement <8 x float> %1273, i64 3, !dbg !7
  %1278 = select i1 %.not36, float %1275, float %1274, !dbg !7
  %1279 = add nuw nsw i32 %lane.sum.lane255, 64, !dbg !7
  %1280 = xor i32 %lane.sum.lane255, 16, !dbg !7
  %1281 = and i32 %1279, 192, !dbg !7
  %1282 = icmp samesign ult i32 %1280, %1281, !dbg !7
  %1283 = select i1 %1282, i32 %1280, i32 %lane.sum.lane255, !dbg !7
  %1284 = shl nuw nsw i32 %1283, 2, !dbg !7
  %1285 = bitcast float %1278 to i32, !dbg !7
  %1286 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1284, i32 %1285), !dbg !7
  %1287 = bitcast i32 %1286 to float, !dbg !7
  %1288 = select i1 %.not36, float %1274, float %1287, !dbg !7
  %1289 = select i1 %.not36, float %1287, float %1275, !dbg !7
  %1290 = select i1 %.not36, float %1277, float %1276, !dbg !7
  %1291 = bitcast float %1290 to i32, !dbg !7
  %1292 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1284, i32 %1291), !dbg !7
  %1293 = bitcast i32 %1292 to float, !dbg !7
  %1294 = select i1 %.not36, float %1276, float %1293, !dbg !7
  %1295 = select i1 %.not36, float %1293, float %1277, !dbg !7
  %1296 = select i1 %.not37, float %1294, float %1288, !dbg !7
  %1297 = xor i32 %lane.sum.lane255, 32, !dbg !7
  %1298 = icmp samesign ult i32 %1297, %1281, !dbg !7
  %1299 = select i1 %1298, i32 %1297, i32 %lane.sum.lane255, !dbg !7
  %1300 = shl nuw nsw i32 %1299, 2, !dbg !7
  %1301 = bitcast float %1296 to i32, !dbg !7
  %1302 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1300, i32 %1301), !dbg !7
  %1303 = bitcast i32 %1302 to float, !dbg !7
  %1304 = select i1 %.not37, float %1288, float %1303, !dbg !7
  %1305 = select i1 %.not37, float %1303, float %1294, !dbg !7
  %1306 = select i1 %.not37, float %1295, float %1289, !dbg !7
  %1307 = bitcast float %1306 to i32, !dbg !7
  %1308 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1300, i32 %1307), !dbg !7
  %1309 = bitcast i32 %1308 to float, !dbg !7
  %1310 = select i1 %.not37, float %1289, float %1309, !dbg !7
  %1311 = select i1 %.not37, float %1309, float %1295, !dbg !7
  %1312 = extractvalue [2 x <8 x float>] %111, 0, !dbg !7
  %1313 = extractelement <8 x float> %1312, i64 0, !dbg !7
  %1314 = extractelement <8 x float> %1312, i64 1, !dbg !7
  %1315 = extractelement <8 x float> %1312, i64 2, !dbg !7
  %1316 = extractelement <8 x float> %1312, i64 3, !dbg !7
  %1317 = select i1 %.not36, float %1314, float %1313, !dbg !7
  %1318 = bitcast float %1317 to i32, !dbg !7
  %1319 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1284, i32 %1318), !dbg !7
  %1320 = bitcast i32 %1319 to float, !dbg !7
  %1321 = select i1 %.not36, float %1313, float %1320, !dbg !7
  %1322 = select i1 %.not36, float %1320, float %1314, !dbg !7
  %1323 = select i1 %.not36, float %1316, float %1315, !dbg !7
  %1324 = bitcast float %1323 to i32, !dbg !7
  %1325 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1284, i32 %1324), !dbg !7
  %1326 = bitcast i32 %1325 to float, !dbg !7
  %1327 = select i1 %.not36, float %1315, float %1326, !dbg !7
  %1328 = select i1 %.not36, float %1326, float %1316, !dbg !7
  %1329 = select i1 %.not37, float %1327, float %1321, !dbg !7
  %1330 = bitcast float %1329 to i32, !dbg !7
  %1331 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1300, i32 %1330), !dbg !7
  %1332 = bitcast i32 %1331 to float, !dbg !7
  %1333 = select i1 %.not37, float %1321, float %1332, !dbg !7
  %1334 = select i1 %.not37, float %1332, float %1327, !dbg !7
  %1335 = select i1 %.not37, float %1328, float %1322, !dbg !7
  %1336 = bitcast float %1335 to i32, !dbg !7
  %1337 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1300, i32 %1336), !dbg !7
  %1338 = bitcast i32 %1337 to float, !dbg !7
  %1339 = select i1 %.not37, float %1322, float %1338, !dbg !7
  %1340 = select i1 %.not37, float %1338, float %1328, !dbg !7
  %1341 = extractvalue [2 x <8 x float>] %110, 0, !dbg !7
  %1342 = extractelement <8 x float> %1341, i64 0, !dbg !7
  %1343 = extractelement <8 x float> %1341, i64 1, !dbg !7
  %1344 = extractelement <8 x float> %1341, i64 2, !dbg !7
  %1345 = extractelement <8 x float> %1341, i64 3, !dbg !7
  %1346 = select i1 %.not36, float %1343, float %1342, !dbg !7
  %1347 = bitcast float %1346 to i32, !dbg !7
  %1348 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1284, i32 %1347), !dbg !7
  %1349 = bitcast i32 %1348 to float, !dbg !7
  %1350 = select i1 %.not36, float %1342, float %1349, !dbg !7
  %1351 = select i1 %.not36, float %1349, float %1343, !dbg !7
  %1352 = select i1 %.not36, float %1345, float %1344, !dbg !7
  %1353 = bitcast float %1352 to i32, !dbg !7
  %1354 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1284, i32 %1353), !dbg !7
  %1355 = bitcast i32 %1354 to float, !dbg !7
  %1356 = select i1 %.not36, float %1344, float %1355, !dbg !7
  %1357 = select i1 %.not36, float %1355, float %1345, !dbg !7
  %1358 = select i1 %.not37, float %1356, float %1350, !dbg !7
  %1359 = bitcast float %1358 to i32, !dbg !7
  %1360 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1300, i32 %1359), !dbg !7
  %1361 = bitcast i32 %1360 to float, !dbg !7
  %1362 = select i1 %.not37, float %1350, float %1361, !dbg !7
  %1363 = select i1 %.not37, float %1361, float %1356, !dbg !7
  %1364 = select i1 %.not37, float %1357, float %1351, !dbg !7
  %1365 = bitcast float %1364 to i32, !dbg !7
  %1366 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1300, i32 %1365), !dbg !7
  %1367 = bitcast i32 %1366 to float, !dbg !7
  %1368 = select i1 %.not37, float %1351, float %1367, !dbg !7
  %1369 = select i1 %.not37, float %1367, float %1357, !dbg !7
  %1370 = extractvalue [2 x <8 x float>] %107, 0, !dbg !7
  %1371 = extractelement <8 x float> %1370, i64 0, !dbg !7
  %1372 = extractelement <8 x float> %1370, i64 1, !dbg !7
  %1373 = extractelement <8 x float> %1370, i64 2, !dbg !7
  %1374 = extractelement <8 x float> %1370, i64 3, !dbg !7
  %1375 = select i1 %.not36, float %1372, float %1371, !dbg !7
  %1376 = bitcast float %1375 to i32, !dbg !7
  %1377 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1284, i32 %1376), !dbg !7
  %1378 = bitcast i32 %1377 to float, !dbg !7
  %1379 = select i1 %.not36, float %1371, float %1378, !dbg !7
  %1380 = select i1 %.not36, float %1378, float %1372, !dbg !7
  %1381 = select i1 %.not36, float %1374, float %1373, !dbg !7
  %1382 = bitcast float %1381 to i32, !dbg !7
  %1383 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1284, i32 %1382), !dbg !7
  %1384 = bitcast i32 %1383 to float, !dbg !7
  %1385 = select i1 %.not36, float %1373, float %1384, !dbg !7
  %1386 = select i1 %.not36, float %1384, float %1374, !dbg !7
  %1387 = select i1 %.not37, float %1385, float %1379, !dbg !7
  %1388 = bitcast float %1387 to i32, !dbg !7
  %1389 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1300, i32 %1388), !dbg !7
  %1390 = select i1 %.not37, float %1386, float %1380, !dbg !7
  %1391 = bitcast float %1390 to i32, !dbg !7
  %1392 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1300, i32 %1391), !dbg !7
  %1393 = shufflevector <1 x float> %1270, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %1394 = extractelement <8 x float> %1273, i64 4, !dbg !7
  %1395 = extractelement <8 x float> %1273, i64 5, !dbg !7
  %1396 = extractelement <8 x float> %1273, i64 6, !dbg !7
  %1397 = extractelement <8 x float> %1273, i64 7, !dbg !7
  %1398 = select i1 %.not36, float %1395, float %1394, !dbg !7
  %1399 = bitcast float %1398 to i32, !dbg !7
  %1400 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1284, i32 %1399), !dbg !7
  %1401 = bitcast i32 %1400 to float, !dbg !7
  %1402 = select i1 %.not36, float %1394, float %1401, !dbg !7
  %1403 = select i1 %.not36, float %1401, float %1395, !dbg !7
  %1404 = select i1 %.not36, float %1397, float %1396, !dbg !7
  %1405 = bitcast float %1404 to i32, !dbg !7
  %1406 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1284, i32 %1405), !dbg !7
  %1407 = bitcast i32 %1406 to float, !dbg !7
  %1408 = select i1 %.not36, float %1396, float %1407, !dbg !7
  %1409 = select i1 %.not36, float %1407, float %1397, !dbg !7
  %1410 = select i1 %.not37, float %1408, float %1402, !dbg !7
  %1411 = bitcast float %1410 to i32, !dbg !7
  %1412 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1300, i32 %1411), !dbg !7
  %1413 = bitcast i32 %1412 to float, !dbg !7
  %1414 = select i1 %.not37, float %1402, float %1413, !dbg !7
  %1415 = select i1 %.not37, float %1413, float %1408, !dbg !7
  %1416 = select i1 %.not37, float %1409, float %1403, !dbg !7
  %1417 = bitcast float %1416 to i32, !dbg !7
  %1418 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1300, i32 %1417), !dbg !7
  %1419 = bitcast i32 %1418 to float, !dbg !7
  %1420 = select i1 %.not37, float %1403, float %1419, !dbg !7
  %1421 = select i1 %.not37, float %1419, float %1409, !dbg !7
  %1422 = extractelement <8 x float> %1312, i64 4, !dbg !7
  %1423 = extractelement <8 x float> %1312, i64 5, !dbg !7
  %1424 = extractelement <8 x float> %1312, i64 6, !dbg !7
  %1425 = extractelement <8 x float> %1312, i64 7, !dbg !7
  %1426 = select i1 %.not36, float %1423, float %1422, !dbg !7
  %1427 = bitcast float %1426 to i32, !dbg !7
  %1428 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1284, i32 %1427), !dbg !7
  %1429 = bitcast i32 %1428 to float, !dbg !7
  %1430 = select i1 %.not36, float %1422, float %1429, !dbg !7
  %1431 = select i1 %.not36, float %1429, float %1423, !dbg !7
  %1432 = select i1 %.not36, float %1425, float %1424, !dbg !7
  %1433 = bitcast float %1432 to i32, !dbg !7
  %1434 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1284, i32 %1433), !dbg !7
  %1435 = bitcast i32 %1434 to float, !dbg !7
  %1436 = select i1 %.not36, float %1424, float %1435, !dbg !7
  %1437 = select i1 %.not36, float %1435, float %1425, !dbg !7
  %1438 = select i1 %.not37, float %1436, float %1430, !dbg !7
  %1439 = bitcast float %1438 to i32, !dbg !7
  %1440 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1300, i32 %1439), !dbg !7
  %1441 = bitcast i32 %1440 to float, !dbg !7
  %1442 = select i1 %.not37, float %1430, float %1441, !dbg !7
  %1443 = select i1 %.not37, float %1441, float %1436, !dbg !7
  %1444 = select i1 %.not37, float %1437, float %1431, !dbg !7
  %1445 = bitcast float %1444 to i32, !dbg !7
  %1446 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1300, i32 %1445), !dbg !7
  %1447 = bitcast i32 %1446 to float, !dbg !7
  %1448 = select i1 %.not37, float %1431, float %1447, !dbg !7
  %1449 = select i1 %.not37, float %1447, float %1437, !dbg !7
  %1450 = extractelement <8 x float> %1341, i64 4, !dbg !7
  %1451 = extractelement <8 x float> %1341, i64 5, !dbg !7
  %1452 = extractelement <8 x float> %1341, i64 6, !dbg !7
  %1453 = extractelement <8 x float> %1341, i64 7, !dbg !7
  %1454 = select i1 %.not36, float %1451, float %1450, !dbg !7
  %1455 = bitcast float %1454 to i32, !dbg !7
  %1456 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1284, i32 %1455), !dbg !7
  %1457 = bitcast i32 %1456 to float, !dbg !7
  %1458 = select i1 %.not36, float %1450, float %1457, !dbg !7
  %1459 = select i1 %.not36, float %1457, float %1451, !dbg !7
  %1460 = select i1 %.not36, float %1453, float %1452, !dbg !7
  %1461 = bitcast float %1460 to i32, !dbg !7
  %1462 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1284, i32 %1461), !dbg !7
  %1463 = bitcast i32 %1462 to float, !dbg !7
  %1464 = select i1 %.not36, float %1452, float %1463, !dbg !7
  %1465 = select i1 %.not36, float %1463, float %1453, !dbg !7
  %1466 = select i1 %.not37, float %1464, float %1458, !dbg !7
  %1467 = bitcast float %1466 to i32, !dbg !7
  %1468 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1300, i32 %1467), !dbg !7
  %1469 = bitcast i32 %1468 to float, !dbg !7
  %1470 = select i1 %.not37, float %1458, float %1469, !dbg !7
  %1471 = select i1 %.not37, float %1469, float %1464, !dbg !7
  %1472 = select i1 %.not37, float %1465, float %1459, !dbg !7
  %1473 = bitcast float %1472 to i32, !dbg !7
  %1474 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1300, i32 %1473), !dbg !7
  %1475 = bitcast i32 %1474 to float, !dbg !7
  %1476 = select i1 %.not37, float %1459, float %1475, !dbg !7
  %1477 = select i1 %.not37, float %1475, float %1465, !dbg !7
  %1478 = extractelement <8 x float> %1370, i64 4, !dbg !7
  %1479 = extractelement <8 x float> %1370, i64 5, !dbg !7
  %1480 = extractelement <8 x float> %1370, i64 6, !dbg !7
  %1481 = extractelement <8 x float> %1370, i64 7, !dbg !7
  %1482 = select i1 %.not36, float %1479, float %1478, !dbg !7
  %1483 = bitcast float %1482 to i32, !dbg !7
  %1484 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1284, i32 %1483), !dbg !7
  %1485 = bitcast i32 %1484 to float, !dbg !7
  %1486 = select i1 %.not36, float %1478, float %1485, !dbg !7
  %1487 = select i1 %.not36, float %1485, float %1479, !dbg !7
  %1488 = select i1 %.not36, float %1481, float %1480, !dbg !7
  %1489 = bitcast float %1488 to i32, !dbg !7
  %1490 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1284, i32 %1489), !dbg !7
  %1491 = bitcast i32 %1490 to float, !dbg !7
  %1492 = select i1 %.not36, float %1480, float %1491, !dbg !7
  %1493 = select i1 %.not36, float %1491, float %1481, !dbg !7
  %1494 = select i1 %.not37, float %1492, float %1486, !dbg !7
  %1495 = bitcast float %1494 to i32, !dbg !7
  %1496 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1300, i32 %1495), !dbg !7
  %1497 = select i1 %.not37, float %1493, float %1487, !dbg !7
  %1498 = bitcast float %1497 to i32, !dbg !7
  %1499 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1300, i32 %1498), !dbg !7
  %1500 = extractvalue [2 x <8 x float>] %112, 1, !dbg !7
  %1501 = extractelement <8 x float> %1500, i64 0, !dbg !7
  %1502 = extractelement <8 x float> %1500, i64 1, !dbg !7
  %1503 = extractelement <8 x float> %1500, i64 2, !dbg !7
  %1504 = extractelement <8 x float> %1500, i64 3, !dbg !7
  %1505 = select i1 %.not36, float %1502, float %1501, !dbg !7
  %1506 = bitcast float %1505 to i32, !dbg !7
  %1507 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1284, i32 %1506), !dbg !7
  %1508 = bitcast i32 %1507 to float, !dbg !7
  %1509 = select i1 %.not36, float %1501, float %1508, !dbg !7
  %1510 = select i1 %.not36, float %1508, float %1502, !dbg !7
  %1511 = select i1 %.not36, float %1504, float %1503, !dbg !7
  %1512 = bitcast float %1511 to i32, !dbg !7
  %1513 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1284, i32 %1512), !dbg !7
  %1514 = bitcast i32 %1513 to float, !dbg !7
  %1515 = select i1 %.not36, float %1503, float %1514, !dbg !7
  %1516 = select i1 %.not36, float %1514, float %1504, !dbg !7
  %1517 = select i1 %.not37, float %1515, float %1509, !dbg !7
  %1518 = bitcast float %1517 to i32, !dbg !7
  %1519 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1300, i32 %1518), !dbg !7
  %1520 = bitcast i32 %1519 to float, !dbg !7
  %1521 = select i1 %.not37, float %1509, float %1520, !dbg !7
  %1522 = select i1 %.not37, float %1520, float %1515, !dbg !7
  %1523 = select i1 %.not37, float %1516, float %1510, !dbg !7
  %1524 = bitcast float %1523 to i32, !dbg !7
  %1525 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1300, i32 %1524), !dbg !7
  %1526 = bitcast i32 %1525 to float, !dbg !7
  %1527 = select i1 %.not37, float %1510, float %1526, !dbg !7
  %1528 = select i1 %.not37, float %1526, float %1516, !dbg !7
  %1529 = extractvalue [2 x <8 x float>] %111, 1, !dbg !7
  %1530 = extractelement <8 x float> %1529, i64 0, !dbg !7
  %1531 = extractelement <8 x float> %1529, i64 1, !dbg !7
  %1532 = extractelement <8 x float> %1529, i64 2, !dbg !7
  %1533 = extractelement <8 x float> %1529, i64 3, !dbg !7
  %1534 = select i1 %.not36, float %1531, float %1530, !dbg !7
  %1535 = bitcast float %1534 to i32, !dbg !7
  %1536 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1284, i32 %1535), !dbg !7
  %1537 = bitcast i32 %1536 to float, !dbg !7
  %1538 = select i1 %.not36, float %1530, float %1537, !dbg !7
  %1539 = select i1 %.not36, float %1537, float %1531, !dbg !7
  %1540 = select i1 %.not36, float %1533, float %1532, !dbg !7
  %1541 = bitcast float %1540 to i32, !dbg !7
  %1542 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1284, i32 %1541), !dbg !7
  %1543 = bitcast i32 %1542 to float, !dbg !7
  %1544 = select i1 %.not36, float %1532, float %1543, !dbg !7
  %1545 = select i1 %.not36, float %1543, float %1533, !dbg !7
  %1546 = select i1 %.not37, float %1544, float %1538, !dbg !7
  %1547 = bitcast float %1546 to i32, !dbg !7
  %1548 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1300, i32 %1547), !dbg !7
  %1549 = bitcast i32 %1548 to float, !dbg !7
  %1550 = select i1 %.not37, float %1538, float %1549, !dbg !7
  %1551 = select i1 %.not37, float %1549, float %1544, !dbg !7
  %1552 = select i1 %.not37, float %1545, float %1539, !dbg !7
  %1553 = bitcast float %1552 to i32, !dbg !7
  %1554 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1300, i32 %1553), !dbg !7
  %1555 = bitcast i32 %1554 to float, !dbg !7
  %1556 = select i1 %.not37, float %1539, float %1555, !dbg !7
  %1557 = select i1 %.not37, float %1555, float %1545, !dbg !7
  %1558 = extractvalue [2 x <8 x float>] %110, 1, !dbg !7
  %1559 = extractelement <8 x float> %1558, i64 0, !dbg !7
  %1560 = extractelement <8 x float> %1558, i64 1, !dbg !7
  %1561 = extractelement <8 x float> %1558, i64 2, !dbg !7
  %1562 = extractelement <8 x float> %1558, i64 3, !dbg !7
  %1563 = select i1 %.not36, float %1560, float %1559, !dbg !7
  %1564 = bitcast float %1563 to i32, !dbg !7
  %1565 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1284, i32 %1564), !dbg !7
  %1566 = bitcast i32 %1565 to float, !dbg !7
  %1567 = select i1 %.not36, float %1559, float %1566, !dbg !7
  %1568 = select i1 %.not36, float %1566, float %1560, !dbg !7
  %1569 = select i1 %.not36, float %1562, float %1561, !dbg !7
  %1570 = bitcast float %1569 to i32, !dbg !7
  %1571 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1284, i32 %1570), !dbg !7
  %1572 = bitcast i32 %1571 to float, !dbg !7
  %1573 = select i1 %.not36, float %1561, float %1572, !dbg !7
  %1574 = select i1 %.not36, float %1572, float %1562, !dbg !7
  %1575 = select i1 %.not37, float %1573, float %1567, !dbg !7
  %1576 = bitcast float %1575 to i32, !dbg !7
  %1577 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1300, i32 %1576), !dbg !7
  %1578 = bitcast i32 %1577 to float, !dbg !7
  %1579 = select i1 %.not37, float %1567, float %1578, !dbg !7
  %1580 = select i1 %.not37, float %1578, float %1573, !dbg !7
  %1581 = select i1 %.not37, float %1574, float %1568, !dbg !7
  %1582 = bitcast float %1581 to i32, !dbg !7
  %1583 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1300, i32 %1582), !dbg !7
  %1584 = bitcast i32 %1583 to float, !dbg !7
  %1585 = select i1 %.not37, float %1568, float %1584, !dbg !7
  %1586 = select i1 %.not37, float %1584, float %1574, !dbg !7
  %1587 = extractvalue [2 x <8 x float>] %107, 1, !dbg !7
  %1588 = extractelement <8 x float> %1587, i64 0, !dbg !7
  %1589 = extractelement <8 x float> %1587, i64 1, !dbg !7
  %1590 = extractelement <8 x float> %1587, i64 2, !dbg !7
  %1591 = extractelement <8 x float> %1587, i64 3, !dbg !7
  %1592 = select i1 %.not36, float %1589, float %1588, !dbg !7
  %1593 = bitcast float %1592 to i32, !dbg !7
  %1594 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1284, i32 %1593), !dbg !7
  %1595 = bitcast i32 %1594 to float, !dbg !7
  %1596 = select i1 %.not36, float %1588, float %1595, !dbg !7
  %1597 = select i1 %.not36, float %1595, float %1589, !dbg !7
  %1598 = select i1 %.not36, float %1591, float %1590, !dbg !7
  %1599 = bitcast float %1598 to i32, !dbg !7
  %1600 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1284, i32 %1599), !dbg !7
  %1601 = bitcast i32 %1600 to float, !dbg !7
  %1602 = select i1 %.not36, float %1590, float %1601, !dbg !7
  %1603 = select i1 %.not36, float %1601, float %1591, !dbg !7
  %1604 = select i1 %.not37, float %1602, float %1596, !dbg !7
  %1605 = bitcast float %1604 to i32, !dbg !7
  %1606 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1300, i32 %1605), !dbg !7
  %1607 = select i1 %.not37, float %1603, float %1597, !dbg !7
  %1608 = bitcast float %1607 to i32, !dbg !7
  %1609 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1300, i32 %1608), !dbg !7
  %1610 = shufflevector <1 x float> %1265, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %1611 = extractelement <8 x float> %1500, i64 4, !dbg !7
  %1612 = extractelement <8 x float> %1500, i64 5, !dbg !7
  %1613 = extractelement <8 x float> %1500, i64 6, !dbg !7
  %1614 = extractelement <8 x float> %1500, i64 7, !dbg !7
  %1615 = select i1 %.not36, float %1612, float %1611, !dbg !7
  %1616 = bitcast float %1615 to i32, !dbg !7
  %1617 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1284, i32 %1616), !dbg !7
  %1618 = bitcast i32 %1617 to float, !dbg !7
  %1619 = select i1 %.not36, float %1611, float %1618, !dbg !7
  %1620 = select i1 %.not36, float %1618, float %1612, !dbg !7
  %1621 = select i1 %.not36, float %1614, float %1613, !dbg !7
  %1622 = bitcast float %1621 to i32, !dbg !7
  %1623 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1284, i32 %1622), !dbg !7
  %1624 = bitcast i32 %1623 to float, !dbg !7
  %1625 = select i1 %.not36, float %1613, float %1624, !dbg !7
  %1626 = select i1 %.not36, float %1624, float %1614, !dbg !7
  %1627 = select i1 %.not37, float %1625, float %1619, !dbg !7
  %1628 = bitcast float %1627 to i32, !dbg !7
  %1629 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1300, i32 %1628), !dbg !7
  %1630 = bitcast i32 %1629 to float, !dbg !7
  %1631 = select i1 %.not37, float %1619, float %1630, !dbg !7
  %1632 = select i1 %.not37, float %1630, float %1625, !dbg !7
  %1633 = select i1 %.not37, float %1626, float %1620, !dbg !7
  %1634 = bitcast float %1633 to i32, !dbg !7
  %1635 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1300, i32 %1634), !dbg !7
  %1636 = bitcast i32 %1635 to float, !dbg !7
  %1637 = select i1 %.not37, float %1620, float %1636, !dbg !7
  %1638 = select i1 %.not37, float %1636, float %1626, !dbg !7
  %1639 = extractelement <8 x float> %1529, i64 4, !dbg !7
  %1640 = extractelement <8 x float> %1529, i64 5, !dbg !7
  %1641 = extractelement <8 x float> %1529, i64 6, !dbg !7
  %1642 = extractelement <8 x float> %1529, i64 7, !dbg !7
  %1643 = select i1 %.not36, float %1640, float %1639, !dbg !7
  %1644 = bitcast float %1643 to i32, !dbg !7
  %1645 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1284, i32 %1644), !dbg !7
  %1646 = bitcast i32 %1645 to float, !dbg !7
  %1647 = select i1 %.not36, float %1639, float %1646, !dbg !7
  %1648 = select i1 %.not36, float %1646, float %1640, !dbg !7
  %1649 = select i1 %.not36, float %1642, float %1641, !dbg !7
  %1650 = bitcast float %1649 to i32, !dbg !7
  %1651 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1284, i32 %1650), !dbg !7
  %1652 = bitcast i32 %1651 to float, !dbg !7
  %1653 = select i1 %.not36, float %1641, float %1652, !dbg !7
  %1654 = select i1 %.not36, float %1652, float %1642, !dbg !7
  %1655 = select i1 %.not37, float %1653, float %1647, !dbg !7
  %1656 = bitcast float %1655 to i32, !dbg !7
  %1657 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1300, i32 %1656), !dbg !7
  %1658 = bitcast i32 %1657 to float, !dbg !7
  %1659 = select i1 %.not37, float %1647, float %1658, !dbg !7
  %1660 = select i1 %.not37, float %1658, float %1653, !dbg !7
  %1661 = select i1 %.not37, float %1654, float %1648, !dbg !7
  %1662 = bitcast float %1661 to i32, !dbg !7
  %1663 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1300, i32 %1662), !dbg !7
  %1664 = bitcast i32 %1663 to float, !dbg !7
  %1665 = select i1 %.not37, float %1648, float %1664, !dbg !7
  %1666 = select i1 %.not37, float %1664, float %1654, !dbg !7
  %1667 = extractelement <8 x float> %1558, i64 4, !dbg !7
  %1668 = extractelement <8 x float> %1558, i64 5, !dbg !7
  %1669 = extractelement <8 x float> %1558, i64 6, !dbg !7
  %1670 = extractelement <8 x float> %1558, i64 7, !dbg !7
  %1671 = select i1 %.not36, float %1668, float %1667, !dbg !7
  %1672 = bitcast float %1671 to i32, !dbg !7
  %1673 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1284, i32 %1672), !dbg !7
  %1674 = bitcast i32 %1673 to float, !dbg !7
  %1675 = select i1 %.not36, float %1667, float %1674, !dbg !7
  %1676 = select i1 %.not36, float %1674, float %1668, !dbg !7
  %1677 = select i1 %.not36, float %1670, float %1669, !dbg !7
  %1678 = bitcast float %1677 to i32, !dbg !7
  %1679 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1284, i32 %1678), !dbg !7
  %1680 = bitcast i32 %1679 to float, !dbg !7
  %1681 = select i1 %.not36, float %1669, float %1680, !dbg !7
  %1682 = select i1 %.not36, float %1680, float %1670, !dbg !7
  %1683 = select i1 %.not37, float %1681, float %1675, !dbg !7
  %1684 = bitcast float %1683 to i32, !dbg !7
  %1685 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1300, i32 %1684), !dbg !7
  %1686 = bitcast i32 %1685 to float, !dbg !7
  %1687 = select i1 %.not37, float %1675, float %1686, !dbg !7
  %1688 = select i1 %.not37, float %1686, float %1681, !dbg !7
  %1689 = select i1 %.not37, float %1682, float %1676, !dbg !7
  %1690 = bitcast float %1689 to i32, !dbg !7
  %1691 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1300, i32 %1690), !dbg !7
  %1692 = bitcast i32 %1691 to float, !dbg !7
  %1693 = select i1 %.not37, float %1676, float %1692, !dbg !7
  %1694 = select i1 %.not37, float %1692, float %1682, !dbg !7
  %1695 = extractelement <8 x float> %1587, i64 4, !dbg !7
  %1696 = extractelement <8 x float> %1587, i64 5, !dbg !7
  %1697 = extractelement <8 x float> %1587, i64 6, !dbg !7
  %1698 = extractelement <8 x float> %1587, i64 7, !dbg !7
  %1699 = select i1 %.not36, float %1696, float %1695, !dbg !7
  %1700 = bitcast float %1699 to i32, !dbg !7
  %1701 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1284, i32 %1700), !dbg !7
  %1702 = bitcast i32 %1701 to float, !dbg !7
  %1703 = select i1 %.not36, float %1695, float %1702, !dbg !7
  %1704 = select i1 %.not36, float %1702, float %1696, !dbg !7
  %1705 = select i1 %.not36, float %1698, float %1697, !dbg !7
  %1706 = bitcast float %1705 to i32, !dbg !7
  %1707 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1284, i32 %1706), !dbg !7
  %1708 = bitcast i32 %1707 to float, !dbg !7
  %1709 = select i1 %.not36, float %1697, float %1708, !dbg !7
  %1710 = select i1 %.not36, float %1708, float %1698, !dbg !7
  %1711 = select i1 %.not37, float %1709, float %1703, !dbg !7
  %1712 = bitcast float %1711 to i32, !dbg !7
  %1713 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1300, i32 %1712), !dbg !7
  %1714 = select i1 %.not37, float %1710, float %1704, !dbg !7
  %1715 = bitcast float %1714 to i32, !dbg !7
  %1716 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1300, i32 %1715), !dbg !7
  %1717 = bitcast i32 %1713 to float, !dbg !7
  %1718 = select i1 %.not37, float %1703, float %1717, !dbg !7
  %1719 = insertelement <4 x float> poison, float %1718, i64 0, !dbg !19
  %1720 = bitcast i32 %1716 to float, !dbg !7
  %1721 = select i1 %.not37, float %1704, float %1720, !dbg !7
  %1722 = insertelement <4 x float> %1719, float %1721, i64 1, !dbg !19
  %1723 = select i1 %.not37, float %1717, float %1709, !dbg !7
  %1724 = insertelement <4 x float> %1722, float %1723, i64 2, !dbg !19
  %1725 = select i1 %.not37, float %1720, float %1710, !dbg !7
  %1726 = insertelement <4 x float> %1724, float %1725, i64 3, !dbg !19
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1610, !dbg !19
  %normalized = fmul <4 x float> %1726, %row.reciprocal, !dbg !19
  %1727 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !20
  %1728 = extractelement <4 x half> %1727, i64 3, !dbg !20
  %1729 = extractelement <4 x half> %1727, i64 2, !dbg !20
  %1730 = extractelement <4 x half> %1727, i64 1, !dbg !20
  %1731 = extractelement <4 x half> %1727, i64 0, !dbg !20
  %1732 = bitcast i32 %1606 to float, !dbg !7
  %1733 = select i1 %.not37, float %1596, float %1732, !dbg !7
  %1734 = insertelement <4 x float> poison, float %1733, i64 0, !dbg !19
  %1735 = bitcast i32 %1609 to float, !dbg !7
  %1736 = select i1 %.not37, float %1597, float %1735, !dbg !7
  %1737 = insertelement <4 x float> %1734, float %1736, i64 1, !dbg !19
  %1738 = select i1 %.not37, float %1732, float %1602, !dbg !7
  %1739 = insertelement <4 x float> %1737, float %1738, i64 2, !dbg !19
  %1740 = select i1 %.not37, float %1735, float %1603, !dbg !7
  %1741 = insertelement <4 x float> %1739, float %1740, i64 3, !dbg !19
  %normalized105 = fmul <4 x float> %1741, %row.reciprocal, !dbg !19
  %1742 = fptrunc <4 x float> %normalized105 to <4 x half>, !dbg !20
  %1743 = extractelement <4 x half> %1742, i64 3, !dbg !20
  %1744 = extractelement <4 x half> %1742, i64 2, !dbg !20
  %1745 = extractelement <4 x half> %1742, i64 1, !dbg !20
  %1746 = extractelement <4 x half> %1742, i64 0, !dbg !20
  %1747 = bitcast i32 %1496 to float, !dbg !7
  %1748 = select i1 %.not37, float %1486, float %1747, !dbg !7
  %1749 = insertelement <4 x float> poison, float %1748, i64 0, !dbg !19
  %1750 = bitcast i32 %1499 to float, !dbg !7
  %1751 = select i1 %.not37, float %1487, float %1750, !dbg !7
  %1752 = insertelement <4 x float> %1749, float %1751, i64 1, !dbg !19
  %1753 = select i1 %.not37, float %1747, float %1492, !dbg !7
  %1754 = insertelement <4 x float> %1752, float %1753, i64 2, !dbg !19
  %1755 = select i1 %.not37, float %1750, float %1493, !dbg !7
  %1756 = insertelement <4 x float> %1754, float %1755, i64 3, !dbg !19
  %row.reciprocal106 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1393, !dbg !19
  %normalized107 = fmul <4 x float> %1756, %row.reciprocal106, !dbg !19
  %1757 = fptrunc <4 x float> %normalized107 to <4 x half>, !dbg !20
  %1758 = extractelement <4 x half> %1757, i64 3, !dbg !20
  %1759 = extractelement <4 x half> %1757, i64 2, !dbg !20
  %1760 = extractelement <4 x half> %1757, i64 1, !dbg !20
  %1761 = extractelement <4 x half> %1757, i64 0, !dbg !20
  %1762 = bitcast i32 %1389 to float, !dbg !7
  %1763 = select i1 %.not37, float %1379, float %1762, !dbg !7
  %1764 = insertelement <4 x float> poison, float %1763, i64 0, !dbg !19
  %1765 = bitcast i32 %1392 to float, !dbg !7
  %1766 = select i1 %.not37, float %1380, float %1765, !dbg !7
  %1767 = insertelement <4 x float> %1764, float %1766, i64 1, !dbg !19
  %1768 = select i1 %.not37, float %1762, float %1385, !dbg !7
  %1769 = insertelement <4 x float> %1767, float %1768, i64 2, !dbg !19
  %1770 = select i1 %.not37, float %1765, float %1386, !dbg !7
  %1771 = insertelement <4 x float> %1769, float %1770, i64 3, !dbg !19
  %normalized108 = fmul <4 x float> %1771, %row.reciprocal106, !dbg !19
  %1772 = fptrunc <4 x float> %normalized108 to <4 x half>, !dbg !20
  %1773 = extractelement <4 x half> %1772, i64 3, !dbg !20
  %1774 = extractelement <4 x half> %1772, i64 2, !dbg !20
  %1775 = extractelement <4 x half> %1772, i64 1, !dbg !20
  %1776 = extractelement <4 x half> %1772, i64 0, !dbg !20
  %address.thread1042 = call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #9, !dbg !21, !range !38, !frisk.register_lifetimes !1
  %address.local1043 = zext nneg i32 %address.thread1042 to i64, !dbg !8
  %address.local1044 = add nuw i64 %20, %address.local1043, !dbg !21
  %address.local1045 = lshr i64 %address.local1043, 2, !dbg !17
  %address.local1046 = and i64 %address.local1045, 48, !dbg !17
  %address.local1047 = add i64 %address.local1044, %address.local1046, !dbg !21
  %address.local1048 = and i64 %address.local1043, 240, !dbg !21
  %address.local1050 = sub i64 %address.local1047, %address.local1048, !dbg !21
  %address.local1051 = add i64 %address.local1050, %21, !dbg !21
  %address.local1052 = shl nuw nsw i64 %address.local1051, 7, !dbg !21
  %1777 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %24, !dbg !21
  %address.local1054 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %1777, i64 %address.local1052, !dbg !21
  %address.local1056 = and i64 %address.local1045, 12, !dbg !11
  %address.local1057 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1056, !dbg !21
  store half %1776, ptr addrspace(1) %address.local1057, align 2, !dbg !21
  %address.local1059 = or disjoint i64 %address.local1056, 1, !dbg !21
  %address.local1060 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1059, !dbg !21
  store half %1775, ptr addrspace(1) %address.local1060, align 2, !dbg !21
  %address.local1062 = or disjoint i64 %address.local1056, 2, !dbg !21
  %address.local1063 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1062, !dbg !21
  store half %1774, ptr addrspace(1) %address.local1063, align 2, !dbg !21
  %address.local1065 = or disjoint i64 %address.local1056, 3, !dbg !21
  %address.local1066 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1065, !dbg !21
  store half %1773, ptr addrspace(1) %address.local1066, align 2, !dbg !21
  %address.local1068 = or disjoint i64 %address.local1056, 16, !dbg !21
  %address.local1069 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1068, !dbg !21
  store half %1761, ptr addrspace(1) %address.local1069, align 2, !dbg !21
  %address.local1071 = or disjoint i64 %address.local1056, 17, !dbg !21
  %address.local1072 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1071, !dbg !21
  store half %1760, ptr addrspace(1) %address.local1072, align 2, !dbg !21
  %address.local1074 = or disjoint i64 %address.local1056, 18, !dbg !21
  %address.local1075 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1074, !dbg !21
  store half %1759, ptr addrspace(1) %address.local1075, align 2, !dbg !21
  %address.local1077 = or disjoint i64 %address.local1056, 19, !dbg !21
  %address.local1078 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1077, !dbg !21
  store half %1758, ptr addrspace(1) %address.local1078, align 2, !dbg !21
  %address.local1080 = add nuw i64 %address.local1052, 8192, !dbg !21
  %address.local1082 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %1777, i64 %address.local1080, !dbg !21
  %address.local1083 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1056, !dbg !21
  store half %1746, ptr addrspace(1) %address.local1083, align 2, !dbg !21
  %address.local1085 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1059, !dbg !21
  store half %1745, ptr addrspace(1) %address.local1085, align 2, !dbg !21
  %address.local1087 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1062, !dbg !21
  store half %1744, ptr addrspace(1) %address.local1087, align 2, !dbg !21
  %address.local1089 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1065, !dbg !21
  store half %1743, ptr addrspace(1) %address.local1089, align 2, !dbg !21
  %address.local1091 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1068, !dbg !21
  store half %1731, ptr addrspace(1) %address.local1091, align 2, !dbg !21
  %address.local1093 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1071, !dbg !21
  store half %1730, ptr addrspace(1) %address.local1093, align 2, !dbg !21
  %address.local1095 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1074, !dbg !21
  store half %1729, ptr addrspace(1) %address.local1095, align 2, !dbg !21
  %address.local1097 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1077, !dbg !21
  store half %1728, ptr addrspace(1) %address.local1097, align 2, !dbg !21
  %1778 = insertelement <4 x float> poison, float %1687, i64 0, !dbg !19
  %1779 = insertelement <4 x float> %1778, float %1693, i64 1, !dbg !19
  %1780 = insertelement <4 x float> %1779, float %1688, i64 2, !dbg !19
  %1781 = insertelement <4 x float> %1780, float %1694, i64 3, !dbg !19
  %normalized110 = fmul <4 x float> %1781, %row.reciprocal, !dbg !19
  %1782 = fptrunc <4 x float> %normalized110 to <4 x half>, !dbg !20
  %1783 = extractelement <4 x half> %1782, i64 3, !dbg !20
  %1784 = extractelement <4 x half> %1782, i64 2, !dbg !20
  %1785 = extractelement <4 x half> %1782, i64 1, !dbg !20
  %1786 = extractelement <4 x half> %1782, i64 0, !dbg !20
  %1787 = insertelement <4 x float> poison, float %1579, i64 0, !dbg !19
  %1788 = insertelement <4 x float> %1787, float %1585, i64 1, !dbg !19
  %1789 = insertelement <4 x float> %1788, float %1580, i64 2, !dbg !19
  %1790 = insertelement <4 x float> %1789, float %1586, i64 3, !dbg !19
  %normalized111 = fmul <4 x float> %1790, %row.reciprocal, !dbg !19
  %1791 = fptrunc <4 x float> %normalized111 to <4 x half>, !dbg !20
  %1792 = extractelement <4 x half> %1791, i64 3, !dbg !20
  %1793 = extractelement <4 x half> %1791, i64 2, !dbg !20
  %1794 = extractelement <4 x half> %1791, i64 1, !dbg !20
  %1795 = extractelement <4 x half> %1791, i64 0, !dbg !20
  %1796 = insertelement <4 x float> poison, float %1470, i64 0, !dbg !19
  %1797 = insertelement <4 x float> %1796, float %1476, i64 1, !dbg !19
  %1798 = insertelement <4 x float> %1797, float %1471, i64 2, !dbg !19
  %1799 = insertelement <4 x float> %1798, float %1477, i64 3, !dbg !19
  %normalized113 = fmul <4 x float> %1799, %row.reciprocal106, !dbg !19
  %1800 = fptrunc <4 x float> %normalized113 to <4 x half>, !dbg !20
  %1801 = extractelement <4 x half> %1800, i64 3, !dbg !20
  %1802 = extractelement <4 x half> %1800, i64 2, !dbg !20
  %1803 = extractelement <4 x half> %1800, i64 1, !dbg !20
  %1804 = extractelement <4 x half> %1800, i64 0, !dbg !20
  %1805 = insertelement <4 x float> poison, float %1362, i64 0, !dbg !19
  %1806 = insertelement <4 x float> %1805, float %1368, i64 1, !dbg !19
  %1807 = insertelement <4 x float> %1806, float %1363, i64 2, !dbg !19
  %1808 = insertelement <4 x float> %1807, float %1369, i64 3, !dbg !19
  %normalized114 = fmul <4 x float> %1808, %row.reciprocal106, !dbg !19
  %1809 = fptrunc <4 x float> %normalized114 to <4 x half>, !dbg !20
  %1810 = extractelement <4 x half> %1809, i64 3, !dbg !20
  %1811 = extractelement <4 x half> %1809, i64 2, !dbg !20
  %1812 = extractelement <4 x half> %1809, i64 1, !dbg !20
  %1813 = extractelement <4 x half> %1809, i64 0, !dbg !20
  %address.local1113 = or disjoint i64 %address.local1056, 32, !dbg !21
  %address.local1114 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1113, !dbg !21
  store half %1813, ptr addrspace(1) %address.local1114, align 2, !dbg !21
  %address.local1116 = or disjoint i64 %address.local1056, 33, !dbg !21
  %address.local1117 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1116, !dbg !21
  store half %1812, ptr addrspace(1) %address.local1117, align 2, !dbg !21
  %address.local1119 = or disjoint i64 %address.local1056, 34, !dbg !21
  %address.local1120 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1119, !dbg !21
  store half %1811, ptr addrspace(1) %address.local1120, align 2, !dbg !21
  %address.local1122 = or disjoint i64 %address.local1056, 35, !dbg !21
  %address.local1123 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1122, !dbg !21
  store half %1810, ptr addrspace(1) %address.local1123, align 2, !dbg !21
  %address.local1125 = or disjoint i64 %address.local1056, 48, !dbg !21
  %address.local1126 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1125, !dbg !21
  store half %1804, ptr addrspace(1) %address.local1126, align 2, !dbg !21
  %address.local1128 = or disjoint i64 %address.local1056, 49, !dbg !21
  %address.local1129 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1128, !dbg !21
  store half %1803, ptr addrspace(1) %address.local1129, align 2, !dbg !21
  %address.local1131 = or disjoint i64 %address.local1056, 50, !dbg !21
  %address.local1132 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1131, !dbg !21
  store half %1802, ptr addrspace(1) %address.local1132, align 2, !dbg !21
  %address.local1134 = or i64 %address.local1045, 51, !dbg !21
  %address.local1135 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1134, !dbg !21
  store half %1801, ptr addrspace(1) %address.local1135, align 2, !dbg !21
  %address.local1140 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1113, !dbg !21
  store half %1795, ptr addrspace(1) %address.local1140, align 2, !dbg !21
  %address.local1142 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1116, !dbg !21
  store half %1794, ptr addrspace(1) %address.local1142, align 2, !dbg !21
  %address.local1144 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1119, !dbg !21
  store half %1793, ptr addrspace(1) %address.local1144, align 2, !dbg !21
  %address.local1146 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1122, !dbg !21
  store half %1792, ptr addrspace(1) %address.local1146, align 2, !dbg !21
  %address.local1148 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1125, !dbg !21
  store half %1786, ptr addrspace(1) %address.local1148, align 2, !dbg !21
  %address.local1150 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1128, !dbg !21
  store half %1785, ptr addrspace(1) %address.local1150, align 2, !dbg !21
  %address.local1152 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1131, !dbg !21
  store half %1784, ptr addrspace(1) %address.local1152, align 2, !dbg !21
  %address.local1154 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1134, !dbg !21
  store half %1783, ptr addrspace(1) %address.local1154, align 2, !dbg !21
  %1814 = insertelement <4 x float> poison, float %1659, i64 0, !dbg !19
  %1815 = insertelement <4 x float> %1814, float %1665, i64 1, !dbg !19
  %1816 = insertelement <4 x float> %1815, float %1660, i64 2, !dbg !19
  %1817 = insertelement <4 x float> %1816, float %1666, i64 3, !dbg !19
  %normalized116 = fmul <4 x float> %1817, %row.reciprocal, !dbg !19
  %1818 = fptrunc <4 x float> %normalized116 to <4 x half>, !dbg !20
  %1819 = extractelement <4 x half> %1818, i64 3, !dbg !20
  %1820 = extractelement <4 x half> %1818, i64 2, !dbg !20
  %1821 = extractelement <4 x half> %1818, i64 1, !dbg !20
  %1822 = extractelement <4 x half> %1818, i64 0, !dbg !20
  %1823 = insertelement <4 x float> poison, float %1550, i64 0, !dbg !19
  %1824 = insertelement <4 x float> %1823, float %1556, i64 1, !dbg !19
  %1825 = insertelement <4 x float> %1824, float %1551, i64 2, !dbg !19
  %1826 = insertelement <4 x float> %1825, float %1557, i64 3, !dbg !19
  %normalized117 = fmul <4 x float> %1826, %row.reciprocal, !dbg !19
  %1827 = fptrunc <4 x float> %normalized117 to <4 x half>, !dbg !20
  %1828 = extractelement <4 x half> %1827, i64 3, !dbg !20
  %1829 = extractelement <4 x half> %1827, i64 2, !dbg !20
  %1830 = extractelement <4 x half> %1827, i64 1, !dbg !20
  %1831 = extractelement <4 x half> %1827, i64 0, !dbg !20
  %1832 = insertelement <4 x float> poison, float %1442, i64 0, !dbg !19
  %1833 = insertelement <4 x float> %1832, float %1448, i64 1, !dbg !19
  %1834 = insertelement <4 x float> %1833, float %1443, i64 2, !dbg !19
  %1835 = insertelement <4 x float> %1834, float %1449, i64 3, !dbg !19
  %normalized119 = fmul <4 x float> %1835, %row.reciprocal106, !dbg !19
  %1836 = fptrunc <4 x float> %normalized119 to <4 x half>, !dbg !20
  %1837 = extractelement <4 x half> %1836, i64 3, !dbg !20
  %1838 = extractelement <4 x half> %1836, i64 2, !dbg !20
  %1839 = extractelement <4 x half> %1836, i64 1, !dbg !20
  %1840 = extractelement <4 x half> %1836, i64 0, !dbg !20
  %1841 = insertelement <4 x float> poison, float %1333, i64 0, !dbg !19
  %1842 = insertelement <4 x float> %1841, float %1339, i64 1, !dbg !19
  %1843 = insertelement <4 x float> %1842, float %1334, i64 2, !dbg !19
  %1844 = insertelement <4 x float> %1843, float %1340, i64 3, !dbg !19
  %normalized120 = fmul <4 x float> %1844, %row.reciprocal106, !dbg !19
  %1845 = fptrunc <4 x float> %normalized120 to <4 x half>, !dbg !20
  %1846 = extractelement <4 x half> %1845, i64 3, !dbg !20
  %1847 = extractelement <4 x half> %1845, i64 2, !dbg !20
  %1848 = extractelement <4 x half> %1845, i64 1, !dbg !20
  %1849 = extractelement <4 x half> %1845, i64 0, !dbg !20
  %address.local1170 = or disjoint i64 %address.local1056, 64, !dbg !21
  %address.local1171 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1170, !dbg !21
  store half %1849, ptr addrspace(1) %address.local1171, align 2, !dbg !21
  %address.local1173 = or disjoint i64 %address.local1056, 65, !dbg !21
  %address.local1174 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1173, !dbg !21
  store half %1848, ptr addrspace(1) %address.local1174, align 2, !dbg !21
  %address.local1176 = or disjoint i64 %address.local1056, 66, !dbg !21
  %address.local1177 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1176, !dbg !21
  store half %1847, ptr addrspace(1) %address.local1177, align 2, !dbg !21
  %address.local1179 = or disjoint i64 %address.local1056, 67, !dbg !21
  %address.local1180 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1179, !dbg !21
  store half %1846, ptr addrspace(1) %address.local1180, align 2, !dbg !21
  %address.local1182 = or disjoint i64 %address.local1056, 80, !dbg !21
  %address.local1183 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1182, !dbg !21
  store half %1840, ptr addrspace(1) %address.local1183, align 2, !dbg !21
  %address.local1185 = or disjoint i64 %address.local1056, 81, !dbg !21
  %address.local1186 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1185, !dbg !21
  store half %1839, ptr addrspace(1) %address.local1186, align 2, !dbg !21
  %address.local1188 = or disjoint i64 %address.local1056, 82, !dbg !21
  %address.local1189 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1188, !dbg !21
  store half %1838, ptr addrspace(1) %address.local1189, align 2, !dbg !21
  %address.local1191 = or disjoint i64 %address.local1056, 83, !dbg !21
  %address.local1192 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1191, !dbg !21
  store half %1837, ptr addrspace(1) %address.local1192, align 2, !dbg !21
  %address.local1197 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1170, !dbg !21
  store half %1831, ptr addrspace(1) %address.local1197, align 2, !dbg !21
  %address.local1199 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1173, !dbg !21
  store half %1830, ptr addrspace(1) %address.local1199, align 2, !dbg !21
  %address.local1201 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1176, !dbg !21
  store half %1829, ptr addrspace(1) %address.local1201, align 2, !dbg !21
  %address.local1203 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1179, !dbg !21
  store half %1828, ptr addrspace(1) %address.local1203, align 2, !dbg !21
  %address.local1205 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1182, !dbg !21
  store half %1822, ptr addrspace(1) %address.local1205, align 2, !dbg !21
  %address.local1207 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1185, !dbg !21
  store half %1821, ptr addrspace(1) %address.local1207, align 2, !dbg !21
  %address.local1209 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1188, !dbg !21
  store half %1820, ptr addrspace(1) %address.local1209, align 2, !dbg !21
  %address.local1211 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1191, !dbg !21
  store half %1819, ptr addrspace(1) %address.local1211, align 2, !dbg !21
  %1850 = insertelement <4 x float> poison, float %1631, i64 0, !dbg !19
  %1851 = insertelement <4 x float> %1850, float %1637, i64 1, !dbg !19
  %1852 = insertelement <4 x float> %1851, float %1632, i64 2, !dbg !19
  %1853 = insertelement <4 x float> %1852, float %1638, i64 3, !dbg !19
  %normalized122 = fmul <4 x float> %1853, %row.reciprocal, !dbg !19
  %1854 = fptrunc <4 x float> %normalized122 to <4 x half>, !dbg !20
  %1855 = extractelement <4 x half> %1854, i64 3, !dbg !20
  %1856 = extractelement <4 x half> %1854, i64 2, !dbg !20
  %1857 = extractelement <4 x half> %1854, i64 1, !dbg !20
  %1858 = extractelement <4 x half> %1854, i64 0, !dbg !20
  %1859 = insertelement <4 x float> poison, float %1521, i64 0, !dbg !19
  %1860 = insertelement <4 x float> %1859, float %1527, i64 1, !dbg !19
  %1861 = insertelement <4 x float> %1860, float %1522, i64 2, !dbg !19
  %1862 = insertelement <4 x float> %1861, float %1528, i64 3, !dbg !19
  %normalized123 = fmul <4 x float> %1862, %row.reciprocal, !dbg !19
  %1863 = fptrunc <4 x float> %normalized123 to <4 x half>, !dbg !20
  %1864 = extractelement <4 x half> %1863, i64 3, !dbg !20
  %1865 = extractelement <4 x half> %1863, i64 2, !dbg !20
  %1866 = extractelement <4 x half> %1863, i64 1, !dbg !20
  %1867 = extractelement <4 x half> %1863, i64 0, !dbg !20
  %1868 = insertelement <4 x float> poison, float %1414, i64 0, !dbg !19
  %1869 = insertelement <4 x float> %1868, float %1420, i64 1, !dbg !19
  %1870 = insertelement <4 x float> %1869, float %1415, i64 2, !dbg !19
  %1871 = insertelement <4 x float> %1870, float %1421, i64 3, !dbg !19
  %normalized125 = fmul <4 x float> %1871, %row.reciprocal106, !dbg !19
  %1872 = fptrunc <4 x float> %normalized125 to <4 x half>, !dbg !20
  %1873 = extractelement <4 x half> %1872, i64 3, !dbg !20
  %1874 = extractelement <4 x half> %1872, i64 2, !dbg !20
  %1875 = extractelement <4 x half> %1872, i64 1, !dbg !20
  %1876 = extractelement <4 x half> %1872, i64 0, !dbg !20
  %1877 = insertelement <4 x float> poison, float %1304, i64 0, !dbg !19
  %1878 = insertelement <4 x float> %1877, float %1310, i64 1, !dbg !19
  %1879 = insertelement <4 x float> %1878, float %1305, i64 2, !dbg !19
  %1880 = insertelement <4 x float> %1879, float %1311, i64 3, !dbg !19
  %normalized126 = fmul <4 x float> %1880, %row.reciprocal106, !dbg !19
  %1881 = fptrunc <4 x float> %normalized126 to <4 x half>, !dbg !20
  %1882 = extractelement <4 x half> %1881, i64 3, !dbg !20
  %1883 = extractelement <4 x half> %1881, i64 2, !dbg !20
  %1884 = extractelement <4 x half> %1881, i64 1, !dbg !20
  %1885 = extractelement <4 x half> %1881, i64 0, !dbg !20
  %address.local1227 = or disjoint i64 %address.local1056, 96, !dbg !21
  %address.local1228 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1227, !dbg !21
  store half %1885, ptr addrspace(1) %address.local1228, align 2, !dbg !21
  %address.local1230 = or disjoint i64 %address.local1056, 97, !dbg !21
  %address.local1231 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1230, !dbg !21
  store half %1884, ptr addrspace(1) %address.local1231, align 2, !dbg !21
  %address.local1233 = or disjoint i64 %address.local1056, 98, !dbg !21
  %address.local1234 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1233, !dbg !21
  store half %1883, ptr addrspace(1) %address.local1234, align 2, !dbg !21
  %address.local1236 = or disjoint i64 %address.local1056, 99, !dbg !21
  %address.local1237 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1236, !dbg !21
  store half %1882, ptr addrspace(1) %address.local1237, align 2, !dbg !21
  %address.local1239 = or disjoint i64 %address.local1056, 112, !dbg !21
  %address.local1240 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1239, !dbg !21
  store half %1876, ptr addrspace(1) %address.local1240, align 2, !dbg !21
  %address.local1242 = or disjoint i64 %address.local1056, 113, !dbg !21
  %address.local1243 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1242, !dbg !21
  store half %1875, ptr addrspace(1) %address.local1243, align 2, !dbg !21
  %address.local1245 = or disjoint i64 %address.local1056, 114, !dbg !21
  %address.local1246 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1245, !dbg !21
  store half %1874, ptr addrspace(1) %address.local1246, align 2, !dbg !21
  %address.local1248 = or i64 %address.local1045, 115, !dbg !21
  %address.local1249 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1248, !dbg !21
  store half %1873, ptr addrspace(1) %address.local1249, align 2, !dbg !21
  %address.local1254 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1227, !dbg !21
  store half %1867, ptr addrspace(1) %address.local1254, align 2, !dbg !21
  %address.local1256 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1230, !dbg !21
  store half %1866, ptr addrspace(1) %address.local1256, align 2, !dbg !21
  %address.local1258 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1233, !dbg !21
  store half %1865, ptr addrspace(1) %address.local1258, align 2, !dbg !21
  %address.local1260 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1236, !dbg !21
  store half %1864, ptr addrspace(1) %address.local1260, align 2, !dbg !21
  %address.local1262 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1239, !dbg !21
  store half %1858, ptr addrspace(1) %address.local1262, align 2, !dbg !21
  %address.local1264 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1242, !dbg !21
  store half %1857, ptr addrspace(1) %address.local1264, align 2, !dbg !21
  %address.local1266 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1245, !dbg !21
  store half %1856, ptr addrspace(1) %address.local1266, align 2, !dbg !21
  %address.local1268 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1248, !dbg !21
  store half %1855, ptr addrspace(1) %address.local1268, align 2, !dbg !21
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
attributes #1 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1" "frisk.global_prefetch"="8" "frisk.k_dma"="64-swizzle:1" "frisk.k_lds_prefetch"="8" "frisk.k_tail_prefetch"="4" "frisk.register_comparisons"="1" "frisk.register_lifetimes"="1" "frisk.softmax_fragments"="pv" "uniform-work-group-size" }
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
!llvm.module.flags = !{!37}

!0 = !DIFile(filename: "causal.input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/causal_best_mytest")
!1 = !{}
!2 = !DISubroutineType(types: !1)
!3 = distinct !DICompileUnit(language: DW_LANG_C, file: !0, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!4 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !0, file: !0, line: 23, type: !2, scopeLine: 23, spFlags: DISPFlagDefinition, unit: !3)
!5 = !DILocation(line: 23, column: 3, scope: !4)
!7 = !DILocation(line: 60, column: 24, scope: !4)
!8 = !DILocation(line: 70, column: 22, scope: !4)
!11 = !DILocation(line: 72, column: 24, scope: !4)
!13 = !DILocation(line: 102, column: 23, scope: !4)
!14 = !DILocation(line: 71, column: 22, scope: !4)
!15 = !DILocation(line: 73, column: 26, scope: !4)
!16 = !DILocation(line: 101, column: 20, scope: !4)
!17 = !DILocation(line: 77, column: 19, scope: !4)
!18 = !DILocation(line: 79, column: 24, scope: !4)
!19 = !DILocation(line: 107, column: 23, scope: !4)
!20 = !DILocation(line: 108, column: 19, scope: !4)
!21 = !DILocation(line: 109, column: 9, scope: !4)
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
!36 = !{i32 256, i32 1, i32 1}
!37 = !{i32 2, !"Debug Info Version", i32 3}
!38 = !{i32 0, i32 256}
!39 = !{i32 4}
!40 = !{i64 8192}
!41 = !{i32 0}
!42 = !{i32 1}
!43 = !{i32 2}
!44 = !{i32 3}
!45 = !{i32 5}
!46 = !{i32 6}
!47 = !{i32 7}
!48 = !{i32 8}
!49 = !{i32 9}
!50 = !{i32 10}
!51 = !{i32 11}
!52 = !{i32 12}
!53 = !{i32 13}
!54 = !{i32 14}
!55 = !{i32 15}
!56 = !{i32 16}
!57 = !{i32 17}
!58 = !{i32 18}
!59 = !{i32 19}
!60 = !{i32 20}
!61 = !{i32 21}
!62 = !{i32 22}
