; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/causal_cvt_pack_mytest/causal.input.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/causal_cvt_pack_mytest/causal.input.mlir"

@shm_0 = addrspace(3) global [16384 x i8] undef, align 16

; Function Attrs: convergent willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3) #1 !dbg !4 !reqd_work_group_size !31 {
  %kdma.tid = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !8
  %kdma.wave = and i32 %kdma.tid, 960, !dbg !8
  %kdma.lane = lshr i32 %kdma.tid, 1, !dbg !8
  %5 = and i32 %kdma.lane, 31, !dbg !8
  %kdma.owner = or disjoint i32 %kdma.wave, %5, !dbg !8
  %kdma.owner.hi = or disjoint i32 %kdma.owner, 32, !dbg !8
  %6 = shl nuw nsw i32 %kdma.tid, 3, !dbg !8
  %kdma.intra = and i32 %6, 8, !dbg !8
  %kdma.intra.hi = xor i32 %kdma.intra, 8, !dbg !8
  %7 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !8
  %8 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %7), !dbg !8
  %9 = add nuw nsw i32 %8, 64, !dbg !8
  %10 = xor i32 %8, 16, !dbg !8
  %11 = and i32 %9, 192, !dbg !8
  %12 = icmp samesign ult i32 %10, %11, !dbg !8
  %13 = select i1 %12, i32 %10, i32 %8, !dbg !8
  %14 = shl nuw nsw i32 %13, 2, !dbg !8
  %15 = xor i32 %8, 32, !dbg !8
  %16 = icmp samesign ult i32 %15, %11, !dbg !8
  %17 = select i1 %16, i32 %15, i32 %8, !dbg !8
  %18 = shl nuw nsw i32 %17, 2, !dbg !8
  %19 = call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !5
  %20 = sext i32 %19 to i64, !dbg !5
  %21 = call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !5
  %22 = sext i32 %21 to i64, !dbg !5
  %23 = shl nsw i64 %22, 5, !dbg !5
  %24 = add nsw i64 %23, %20, !dbg !5
  %25 = lshr i64 %24, 5, !dbg !5
  %26 = and i64 %25, 576460752303423484, !dbg !5
  %27 = and i64 %20, 3, !dbg !5
  %28 = or disjoint i64 %26, %27, !dbg !5
  %29 = sub nsw i64 31, %28, !dbg !5
  %30 = lshr i64 %24, 2, !dbg !5
  %31 = and i64 %30, 31, !dbg !5
  %32 = call i32 @llvm.amdgcn.workgroup.id.z(), !dbg !5
  %33 = sext i32 %32 to i64, !dbg !5
  %34 = icmp ugt i64 %24, 1023, !dbg !9
  %35 = add nsw i64 %28, -32, !dbg !9
  %36 = select i1 %34, i64 %35, i64 %29, !dbg !9
  %37 = sdiv i64 %36, 32, !dbg !9
  %38 = sext i1 %34 to i64, !dbg !9
  %39 = xor i64 %37, %38, !dbg !9
  %40 = add nsw i64 %30, %39, !dbg !9
  %41 = and i64 %40, 31, !dbg !9
  %42 = shl i64 %29, 7, !dbg !9
  %43 = mul nsw i64 %39, -4096, !dbg !9
  %44 = add i64 %42, %43, !dbg !9
  %45 = shl nsw i64 %44, 7, !dbg !9
  %46 = zext nneg i32 %kdma.tid to i64, !dbg !9
  %.idx = shl nuw nsw i64 %41, 20, !dbg !9
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #9, !dbg !9, !range !33, !frisk.register_lifetimes !1
  %47 = shl nuw nsw i32 %address.thread, 3, !dbg !9
  %48 = and i32 %47, 1536, !dbg !9
  %address.local466 = zext nneg i32 %48 to i64, !dbg !9
  %address.local467 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local466, !dbg !9
  %address.local468 = ptrtoint ptr addrspace(3) %address.local467 to i64, !dbg !9
  %address.local469 = trunc i64 %address.local468 to i32, !dbg !9
  %49 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local469), !dbg !9
  %address.local470 = lshr i32 %address.thread, 4, !dbg !9
  %address.local471 = and i32 %address.local470, 8, !dbg !9
  %address.local472 = shl nuw nsw i32 %address.thread, 7, !dbg !9
  %address.local473 = and i32 %address.local472, 16256, !dbg !9
  %address.local474 = or disjoint i32 %address.local471, %address.local473, !dbg !9
  %50 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !9
  %.idx1444 = shl i64 %44, 8, !dbg !9
  %51 = add i64 %.idx, %.idx1444, !dbg !9
  %52 = shl nuw nsw i32 %address.local474, 1, !dbg !9
  %53 = trunc i64 %51 to i32, !dbg !9
  %dma.buffer.offset1442 = or disjoint i32 %52, %53, !dbg !9
  %54 = insertelement <2 x i64> poison, i64 %50, i64 0, !dbg !9
  %55 = bitcast <2 x i64> %54 to <4 x i32>, !dbg !9
  %dma.buffer.rsrc1443 = shufflevector <4 x i32> %55, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !9
  %56 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %49, i32 %dma.buffer.offset1442, <4 x i32> %dma.buffer.rsrc1443) #10, !dbg !9
  %address.local479 = or disjoint i64 %address.local466, 2048, !dbg !9
  %address.local480 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local479, !dbg !9
  %address.local481 = ptrtoint ptr addrspace(3) %address.local480 to i64, !dbg !9
  %address.local482 = trunc i64 %address.local481 to i32, !dbg !9
  %57 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local482), !dbg !9
  %address.local488 = or disjoint i32 %address.local474, 16, !dbg !9
  %58 = shl nuw nsw i32 %address.local488, 1, !dbg !9
  %dma.buffer.offset1440 = or disjoint i32 %58, %53, !dbg !9
  %59 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %57, i32 %dma.buffer.offset1440, <4 x i32> %dma.buffer.rsrc1443) #10, !dbg !9
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %60 = shl nuw nsw i64 %41, 19, !dbg !9
  %61 = add i64 %60, %45, !dbg !9
  %address.local491 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local466, !dbg !9
  %address.local492 = ptrtoint ptr addrspace(3) %address.local491 to i64, !dbg !9
  %address.local493 = trunc i64 %address.local492 to i32, !dbg !9
  %62 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local493), !dbg !9
  %63 = trunc i64 %61 to i32, !dbg !9
  %.tr = or disjoint i32 %address.local474, %63, !dbg !9
  %64 = shl i32 %.tr, 1, !dbg !9
  %dma.buffer.offset1438 = or disjoint i32 %64, 64, !dbg !9
  %65 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %62, i32 %dma.buffer.offset1438, <4 x i32> %dma.buffer.rsrc1443) #10, !dbg !9
  %address.local495 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local479, !dbg !9
  %address.local496 = ptrtoint ptr addrspace(3) %address.local495 to i64, !dbg !9
  %address.local497 = trunc i64 %address.local496 to i32, !dbg !9
  %66 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local497), !dbg !9
  %.tr1448 = or disjoint i32 %address.local488, %63, !dbg !9
  %67 = shl i32 %.tr1448, 1, !dbg !9
  %dma.buffer.offset1436 = or disjoint i32 %67, 64, !dbg !9
  %68 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %66, i32 %dma.buffer.offset1436, <4 x i32> %dma.buffer.rsrc1443) #10, !dbg !9
  %69 = and i64 %46, 240, !dbg !19
  %address.local499 = lshr i32 %address.thread, 2, !dbg !13
  %address.local500 = and i32 %address.local499, 4, !dbg !13
  %address.local501 = shl nuw nsw i32 %address.thread, 1, !dbg !13
  %address.local502 = and i32 %address.local501, 384, !dbg !13
  %address.local503 = or disjoint i32 %address.local500, %address.local502, !dbg !13
  %address.local505 = and i32 %47, 120, !dbg !13
  %address.local506 = or disjoint i32 %address.local503, %address.local505, !dbg !13
  %address.local507 = shl nuw nsw i32 %address.thread, 5, !dbg !13
  %address.local508 = and i32 %address.local507, 1024, !dbg !13
  %address.local509 = or disjoint i32 %address.local506, %address.local508, !dbg !13
  %address.local510 = zext nneg i32 %address.local509 to i64, !dbg !13
  %address.local511 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local510, !dbg !13
  %70 = load <4 x half>, ptr addrspace(3) %address.local511, align 8, !dbg !13
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local511, i64 4096, !dbg !13
  %71 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !13
  %72 = fmul <4 x half> %70, splat (half 0xH3015), !dbg !13
  %lds.const.gep1265 = getelementptr i8, ptr addrspace(3) %address.local511, i64 1024, !dbg !13
  %73 = load <4 x half>, ptr addrspace(3) %lds.const.gep1265, align 8, !dbg !13
  %74 = fmul <4 x half> %71, splat (half 0xH3015), !dbg !13
  %lds.const.gep1266 = getelementptr i8, ptr addrspace(3) %address.local511, i64 5120, !dbg !13
  %75 = load <4 x half>, ptr addrspace(3) %lds.const.gep1266, align 8, !dbg !13
  %76 = fmul <4 x half> %73, splat (half 0xH3015), !dbg !13
  %77 = fmul <4 x half> %75, splat (half 0xH3015), !dbg !13
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %dma.buffer.offset1434 = or disjoint i32 %64, 128, !dbg !9
  %78 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %49, i32 %dma.buffer.offset1434, <4 x i32> %dma.buffer.rsrc1443) #10, !dbg !9
  %dma.buffer.offset1432 = or disjoint i32 %67, 128, !dbg !9
  %79 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %57, i32 %dma.buffer.offset1432, <4 x i32> %dma.buffer.rsrc1443) #10, !dbg !9
  %lds.const.gep1267 = getelementptr i8, ptr addrspace(3) %address.local511, i64 8192, !dbg !13
  %80 = load <4 x half>, ptr addrspace(3) %lds.const.gep1267, align 8, !dbg !13
  %lds.const.gep1268 = getelementptr i8, ptr addrspace(3) %address.local511, i64 12288, !dbg !13
  %81 = load <4 x half>, ptr addrspace(3) %lds.const.gep1268, align 8, !dbg !13
  %lds.const.gep1269 = getelementptr i8, ptr addrspace(3) %address.local511, i64 9216, !dbg !13
  %82 = load <4 x half>, ptr addrspace(3) %lds.const.gep1269, align 8, !dbg !13
  %83 = fmul <4 x half> %80, splat (half 0xH3015), !dbg !13
  %lds.const.gep1270 = getelementptr i8, ptr addrspace(3) %address.local511, i64 13312, !dbg !13
  %84 = load <4 x half>, ptr addrspace(3) %lds.const.gep1270, align 8, !dbg !13
  %85 = fmul <4 x half> %81, splat (half 0xH3015), !dbg !13
  %86 = fmul <4 x half> %82, splat (half 0xH3015), !dbg !13
  %87 = fmul <4 x half> %84, splat (half 0xH3015), !dbg !13
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %dma.buffer.offset1430 = or disjoint i32 %64, 192, !dbg !9
  %88 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %62, i32 %dma.buffer.offset1430, <4 x i32> %dma.buffer.rsrc1443) #10, !dbg !9
  %dma.buffer.offset1428 = or disjoint i32 %67, 192, !dbg !9
  %89 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %66, i32 %dma.buffer.offset1428, <4 x i32> %dma.buffer.rsrc1443) #10, !dbg !9
  %90 = load <4 x half>, ptr addrspace(3) %address.local511, align 8, !dbg !13
  %91 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !13
  %92 = load <4 x half>, ptr addrspace(3) %lds.const.gep1265, align 8, !dbg !13
  %93 = fmul <4 x half> %90, splat (half 0xH3015), !dbg !13
  %94 = load <4 x half>, ptr addrspace(3) %lds.const.gep1266, align 8, !dbg !13
  %95 = fmul <4 x half> %91, splat (half 0xH3015), !dbg !13
  %96 = fmul <4 x half> %92, splat (half 0xH3015), !dbg !13
  %97 = fmul <4 x half> %94, splat (half 0xH3015), !dbg !13
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %98 = load <4 x half>, ptr addrspace(3) %lds.const.gep1267, align 8, !dbg !13
  %99 = load <4 x half>, ptr addrspace(3) %lds.const.gep1268, align 8, !dbg !13
  %100 = load <4 x half>, ptr addrspace(3) %lds.const.gep1269, align 8, !dbg !13
  %101 = fmul <4 x half> %98, splat (half 0xH3015), !dbg !13
  %102 = load <4 x half>, ptr addrspace(3) %lds.const.gep1270, align 8, !dbg !13
  %103 = fmul <4 x half> %99, splat (half 0xH3015), !dbg !13
  %104 = fmul <4 x half> %100, splat (half 0xH3015), !dbg !13
  %105 = fmul <4 x half> %102, splat (half 0xH3015), !dbg !13
  fence syncscope("workgroup") release, !dbg !6
  call void @llvm.amdgcn.s.barrier(), !dbg !6
  fence syncscope("workgroup") acquire, !dbg !6
  %106 = sub nsw i64 32, %28, !dbg !15
  %107 = shl nuw nsw i64 %31, 12, !dbg !16
  %108 = shl nsw i64 %33, 17, !dbg !16
  %109 = shl nuw nsw i64 %31, 7, !dbg !16
  %110 = shl nsw i64 %33, 12, !dbg !16
  %111 = or disjoint i64 %109, %110, !dbg !16
  %112 = shl nuw nsw i64 %46, 2, !dbg !17
  %113 = and i64 %112, 3084, !dbg !17
  %114 = and i32 %kdma.tid, 1, !dbg !17
  %index.compact395 = zext nneg i32 %114 to i64, !dbg !17
  %115 = or disjoint i64 %113, %index.compact395, !dbg !17
  %116 = add nuw nsw i64 %115, 1, !dbg !17
  %117 = lshr i64 %116, 1, !dbg !17
  %118 = add i64 %42, %46, !dbg !19
  %119 = lshr i64 %46, 2, !dbg !19
  %120 = and i64 %119, 48, !dbg !19
  %121 = add i64 %118, %120, !dbg !19
  %122 = sub i64 %121, %69, !dbg !19
  %123 = and i64 %46, 48, !dbg !19
  %124 = trunc i64 %122 to i32, !dbg !19
  %125 = add i32 %124, 1, !dbg !19
  %126 = add i32 %124, 65, !dbg !19
  br label %127, !dbg !6

127:                                              ; preds = %136, %4
  %128 = phi i64 [ %1275, %136 ], [ 0, %4 ], !dbg !6
  %129 = phi [2 x <8 x float>] [ %1260, %136 ], [ zeroinitializer, %4 ], !dbg !6
  %130 = phi [2 x <1 x float>] [ %1026, %136 ], [ zeroinitializer, %4 ], !dbg !6
  %131 = phi [2 x <1 x float>] [ %1262, %136 ], [ [<1 x float> splat (float 0xFFF0000000000000), <1 x float> splat (float 0xFFF0000000000000)], %4 ], !dbg !6
  %132 = phi [2 x <8 x float>] [ %1266, %136 ], [ zeroinitializer, %4 ], !dbg !6
  %133 = phi [2 x <8 x float>] [ %1270, %136 ], [ zeroinitializer, %4 ], !dbg !6
  %134 = phi [2 x <8 x float>] [ %1274, %136 ], [ zeroinitializer, %4 ], !dbg !6
  %135 = icmp slt i64 %128, %106, !dbg !6
  br i1 %135, label %136, label %1276, !dbg !6

136:                                              ; preds = %127
  %137 = lshr i64 %128, 12, !dbg !16
  %138 = add nuw nsw i64 %137, %30, !dbg !16
  %139 = lshr i64 %128, 5, !dbg !16
  %140 = and i64 %139, 127, !dbg !16
  %141 = shl nsw i64 %128, 7, !dbg !16
  %142 = shl i64 %138, 19, !dbg !16
  %143 = and i64 %142, 16252928, !dbg !16
  %144 = shl i64 %139, 19, !dbg !16
  %145 = shl i64 %128, 14, !dbg !16
  %146 = sub i64 %145, %144, !dbg !16
  %address.thread562 = call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #9, !dbg !16, !range !33, !frisk.register_lifetimes !1
  %address.local563 = zext nneg i32 %address.thread562 to i64, !dbg !9
  %address.local569 = lshr i64 %address.local563, 2, !dbg !16
  %147 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %.idx14571460 = or disjoint i64 %143, %140, !dbg !16
  %148 = add i64 %.idx14571460, %146, !dbg !16
  %149 = insertelement <2 x i64> poison, i64 %147, i64 0, !dbg !16
  %150 = bitcast <2 x i64> %149 to <4 x i32>, !dbg !16
  %dma.buffer.rsrc1427 = shufflevector <4 x i32> %150, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !16
  %kdma.source = zext nneg i32 %kdma.owner to i64, !dbg !16
  %kdma.source1594 = lshr i64 %kdma.source, 4, !dbg !16
  %kdma.source1595 = shl nuw nsw i64 %kdma.source1594, 9, !dbg !16
  %kdma.source1596 = shl nuw nsw i64 %kdma.source, 7, !dbg !16
  %kdma.source1597 = add nuw nsw i64 %kdma.source1595, %kdma.source1596, !dbg !16
  %kdma.source1598 = lshr i64 %kdma.source, 5, !dbg !16
  %kdma.source1599 = lshr i64 %kdma.source, 2, !dbg !16
  %kdma.source1600 = and i64 %kdma.source1599, 1, !dbg !16
  %kdma.source1601 = and i64 %kdma.source1594, %kdma.source1600, !dbg !16
  %kdma.source1602 = or disjoint i64 %kdma.source1598, %kdma.source1601, !dbg !16
  %kdma.source1603 = shl nuw nsw i64 %kdma.source1602, 10, !dbg !16
  %kdma.source1604 = sub nsw i64 %kdma.source1597, %kdma.source1603, !dbg !16
  %kdma.source1605 = and i64 %kdma.source1604, 16256, !dbg !16
  %kdma.source1606 = lshr i32 %kdma.tid, 4, !dbg !16
  %kdma.source1607 = and i32 %kdma.source1606, 8, !dbg !16
  %kdma.source1608 = zext nneg i32 %kdma.source1607 to i64, !dbg !16
  %kdma.source1609 = or disjoint i64 %kdma.source1605, %kdma.source1608, !dbg !16
  %kdma.source1610 = add i64 %148, %kdma.source1609, !dbg !16
  %kdma.source1611 = trunc i64 %kdma.source1610 to i32, !dbg !16
  %kdma.source1612 = shl i32 %kdma.source1611, 1, !dbg !16
  %kdma.offset = add i32 %kdma.source1612, %kdma.intra, !dbg !16
  %151 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %49, i32 %kdma.offset, <4 x i32> %dma.buffer.rsrc1427) #10, !dbg !16
  %kdma.source1613 = zext nneg i32 %kdma.owner.hi to i64, !dbg !16
  %kdma.source1614 = lshr i64 %kdma.source1613, 4, !dbg !16
  %kdma.source1615 = shl nuw nsw i64 %kdma.source1614, 9, !dbg !16
  %kdma.source1616 = shl nuw nsw i64 %kdma.source1613, 7, !dbg !16
  %kdma.source1617 = add nuw nsw i64 %kdma.source1615, %kdma.source1616, !dbg !16
  %kdma.source1618 = lshr i64 %kdma.source1613, 5, !dbg !16
  %kdma.source1619 = lshr i64 %kdma.source1613, 2, !dbg !16
  %kdma.source1620 = and i64 %kdma.source1619, 1, !dbg !16
  %kdma.source1621 = and i64 %kdma.source1614, %kdma.source1620, !dbg !16
  %kdma.source1622 = add nuw nsw i64 %kdma.source1618, %kdma.source1621, !dbg !16
  %kdma.source1623 = shl nuw nsw i64 %kdma.source1622, 10, !dbg !16
  %kdma.source1624 = sub nsw i64 %kdma.source1617, %kdma.source1623, !dbg !16
  %kdma.source1625 = and i64 %kdma.source1624, 16256, !dbg !16
  %kdma.source1629 = or disjoint i64 %kdma.source1625, %kdma.source1608, !dbg !16
  %kdma.source1630 = add i64 %148, %kdma.source1629, !dbg !16
  %kdma.source1631 = trunc i64 %kdma.source1630 to i32, !dbg !16
  %kdma.source1632 = shl i32 %kdma.source1631, 1, !dbg !16
  %kdma.offset1633 = add i32 %kdma.source1632, %kdma.intra.hi, !dbg !16
  %kdma.lds1634 = add i32 %49, 512, !dbg !16
  %152 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1634, i32 %kdma.offset1633, <4 x i32> %dma.buffer.rsrc1427) #10, !dbg !16
  %kdma.source1636 = or disjoint i64 %kdma.source, 256, !dbg !16
  %kdma.source1637 = lshr i64 %kdma.source1636, 4, !dbg !16
  %kdma.source1638 = shl nuw nsw i64 %kdma.source1637, 9, !dbg !16
  %kdma.source1640 = add nuw nsw i64 %kdma.source1638, %kdma.source1596, !dbg !16
  %kdma.source1641 = lshr i64 %kdma.source1636, 5, !dbg !16
  %kdma.source1644 = and i64 %kdma.source1637, %kdma.source1600, !dbg !16
  %kdma.source1645 = or disjoint i64 %kdma.source1641, %kdma.source1644, !dbg !16
  %kdma.source1646 = shl nuw nsw i64 %kdma.source1645, 10, !dbg !16
  %kdma.source1647 = sub nsw i64 %kdma.source1640, %kdma.source1646, !dbg !16
  %kdma.source1648 = and i64 %kdma.source1647, 16256, !dbg !16
  %kdma.source1651 = or disjoint i32 %kdma.source1607, 16, !dbg !16
  %kdma.source1652 = zext nneg i32 %kdma.source1651 to i64, !dbg !16
  %kdma.source1653 = or disjoint i64 %kdma.source1648, %kdma.source1652, !dbg !16
  %kdma.source1654 = add i64 %148, %kdma.source1653, !dbg !16
  %kdma.source1655 = trunc i64 %kdma.source1654 to i32, !dbg !16
  %kdma.source1656 = shl i32 %kdma.source1655, 1, !dbg !16
  %kdma.offset1657 = add i32 %kdma.source1656, %kdma.intra, !dbg !16
  %153 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %57, i32 %kdma.offset1657, <4 x i32> %dma.buffer.rsrc1427) #10, !dbg !16
  %kdma.source1660 = or disjoint i64 %kdma.source1613, 256, !dbg !16
  %kdma.source1661 = lshr i64 %kdma.source1660, 4, !dbg !16
  %kdma.source1662 = shl nuw nsw i64 %kdma.source1661, 9, !dbg !16
  %kdma.source1664 = add nuw nsw i64 %kdma.source1662, %kdma.source1616, !dbg !16
  %kdma.source1665 = lshr i64 %kdma.source1660, 5, !dbg !16
  %kdma.source1668 = and i64 %kdma.source1661, %kdma.source1620, !dbg !16
  %kdma.source1669 = add nuw nsw i64 %kdma.source1665, %kdma.source1668, !dbg !16
  %kdma.source1670 = shl nuw nsw i64 %kdma.source1669, 10, !dbg !16
  %kdma.source1671 = sub nsw i64 %kdma.source1664, %kdma.source1670, !dbg !16
  %kdma.source1672 = and i64 %kdma.source1671, 16256, !dbg !16
  %kdma.source1677 = or disjoint i64 %kdma.source1672, %kdma.source1652, !dbg !16
  %kdma.source1678 = add i64 %148, %kdma.source1677, !dbg !16
  %kdma.source1679 = trunc i64 %kdma.source1678 to i32, !dbg !16
  %kdma.source1680 = shl i32 %kdma.source1679, 1, !dbg !16
  %kdma.offset1681 = add i32 %kdma.source1680, %kdma.intra.hi, !dbg !16
  %kdma.lds1682 = add i32 %57, 512, !dbg !16
  %154 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1682, i32 %kdma.offset1681, <4 x i32> %dma.buffer.rsrc1427) #10, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %155 = add nuw i64 %128, %107, !dbg !16
  %156 = add i64 %155, %108, !dbg !16
  %157 = add i64 %156, 1024, !dbg !16
  %158 = icmp slt i64 %157, 0, !dbg !16
  %159 = sub i64 -1025, %156, !dbg !16
  %160 = select i1 %158, i64 %159, i64 %157, !dbg !16
  %161 = sdiv i64 %160, 4096, !dbg !16
  %.lobit41 = ashr i64 %157, 63, !dbg !16
  %162 = xor i64 %161, %.lobit41, !dbg !16
  %163 = add nsw i64 %139, %111, !dbg !16
  %164 = add nsw i64 %163, 32, !dbg !16
  %165 = icmp slt i64 %163, -32, !dbg !16
  %166 = sub nsw i64 -33, %163, !dbg !16
  %167 = select i1 %165, i64 %166, i64 %164, !dbg !16
  %168 = sdiv i64 %167, 128, !dbg !16
  %169 = sext i1 %165 to i64, !dbg !16
  %170 = xor i64 %168, %169, !dbg !16
  %171 = shl nsw i64 %170, 7, !dbg !16
  %172 = sub i64 %111, %171, !dbg !16
  %173 = shl i64 %162, 19, !dbg !16
  %174 = and i64 %173, 16252928, !dbg !16
  %.idx14671471 = add nuw nsw i64 %174, %139, !dbg !16
  %175 = add i64 %.idx14671471, %172, !dbg !16
  %176 = add i64 %175, %146, !dbg !16
  %kdma.source1700 = add i64 %176, %kdma.source1609, !dbg !16
  %kdma.source1701 = trunc i64 %kdma.source1700 to i32, !dbg !16
  %kdma.source1702 = shl i32 %kdma.source1701, 1, !dbg !16
  %kdma.source1703 = add i32 %kdma.source1702, 64, !dbg !16
  %kdma.offset1704 = add i32 %kdma.source1703, %kdma.intra, !dbg !16
  %177 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %62, i32 %kdma.offset1704, <4 x i32> %dma.buffer.rsrc1427) #10, !dbg !16
  %kdma.source1723 = add i64 %176, %kdma.source1629, !dbg !16
  %kdma.source1724 = trunc i64 %kdma.source1723 to i32, !dbg !16
  %kdma.source1725 = shl i32 %kdma.source1724, 1, !dbg !16
  %kdma.source1726 = add i32 %kdma.source1725, 64, !dbg !16
  %kdma.offset1727 = add i32 %kdma.source1726, %kdma.intra.hi, !dbg !16
  %kdma.lds1728 = add i32 %62, 512, !dbg !16
  %178 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1728, i32 %kdma.offset1727, <4 x i32> %dma.buffer.rsrc1427) #10, !dbg !16
  %kdma.source1748 = add i64 %176, %kdma.source1653, !dbg !16
  %kdma.source1749 = trunc i64 %kdma.source1748 to i32, !dbg !16
  %kdma.source1750 = shl i32 %kdma.source1749, 1, !dbg !16
  %kdma.source1751 = add i32 %kdma.source1750, 64, !dbg !16
  %kdma.offset1752 = add i32 %kdma.source1751, %kdma.intra, !dbg !16
  %179 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %66, i32 %kdma.offset1752, <4 x i32> %dma.buffer.rsrc1427) #10, !dbg !16
  %kdma.source1773 = add i64 %176, %kdma.source1677, !dbg !16
  %kdma.source1774 = trunc i64 %kdma.source1773 to i32, !dbg !16
  %kdma.source1775 = shl i32 %kdma.source1774, 1, !dbg !16
  %kdma.source1776 = add i32 %kdma.source1775, 64, !dbg !16
  %kdma.offset1777 = add i32 %kdma.source1776, %kdma.intra.hi, !dbg !16
  %kdma.lds1778 = add i32 %66, 512, !dbg !16
  %180 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1778, i32 %kdma.offset1777, <4 x i32> %dma.buffer.rsrc1427) #10, !dbg !16
  %address.local606 = shl nuw nsw i64 %address.local563, 1, !dbg !17
  %address.local607 = and i64 %address.local606, 24, !dbg !17
  %address.local609 = shl i64 %address.local563, 62, !dbg !17
  %address.local610 = ashr exact i64 %address.local609, 57, !dbg !17
  %address.local611 = and i64 %address.local610, -64, !dbg !17
  %address.local614 = and i64 %address.local569, 12, !dbg !13
  %address.local616 = and i64 %address.local563, 63, !dbg !13
  %address.local617 = lshr i64 %address.local616, 5, !dbg !13
  %address.local618 = mul nuw nsw i64 %address.local617, 1016, !dbg !13
  %.idx1479 = mul nuw nsw i64 %address.local563, 320, !dbg !17
  %181 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %.idx1479, !dbg !17
  %.idx1480 = mul nsw i64 %address.local569, -1280, !dbg !17
  %182 = getelementptr i8, ptr addrspace(3) %181, i64 %.idx1480, !dbg !17
  %183 = getelementptr [2 x i8], ptr addrspace(3) %182, i64 %address.local607, !dbg !17
  %184 = getelementptr [2 x i8], ptr addrspace(3) %183, i64 %address.local611, !dbg !17
  %185 = getelementptr [2 x i8], ptr addrspace(3) %184, i64 %address.local614, !dbg !17
  %address.local620 = getelementptr [2 x i8], ptr addrspace(3) %185, i64 %address.local618, !dbg !17
  %186 = add nsw i64 %.idx1479, %.idx1480, !dbg !17
  %.idx2097 = shl nuw nsw i64 %address.local607, 1, !dbg !17
  %187 = or disjoint i64 %186, %.idx2097, !dbg !17
  %188 = or disjoint i64 %address.local611, %address.local614, !dbg !17
  %189 = shl nsw i64 %188, 1, !dbg !17
  %190 = add nsw i64 %189, %187, !dbg !17
  %address.local620.idx = mul nuw nsw i64 %address.local617, 2032, !dbg !17
  %191 = add nsw i64 %190, %address.local620.idx, !dbg !17
  %192 = lshr i64 %191, 6, !dbg !17
  %193 = and i64 %192, 8, !dbg !17
  %194 = xor i64 %191, %193, !dbg !17
  %195 = sub i64 %194, %191, !dbg !17
  %kdma.swizzled = getelementptr i8, ptr addrspace(3) %address.local620, i64 %195, !dbg !17
  %196 = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8, !dbg !17
  %197 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %196, <4 x float> zeroinitializer) #9, !dbg !17
  %kdma.gep = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4096, !dbg !17
  %198 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8, !dbg !17
  %address.local631 = shl nuw nsw i32 %address.thread562, 1, !dbg !17
  %address.local632 = and i32 %address.local631, 24, !dbg !17
  %address.local633 = shl nuw nsw i32 %address.thread562, 8, !dbg !17
  %address.local634 = and i32 %address.local633, 768, !dbg !17
  %address.local635 = or disjoint i32 %address.local632, %address.local634, !dbg !17
  %address.local636 = shl nuw nsw i32 %address.thread562, 5, !dbg !17
  %address.local637 = and i32 %address.local636, 96, !dbg !17
  %address.local638 = or disjoint i32 %address.local635, %address.local637, !dbg !17
  %address.local639 = zext nneg i32 %address.local638 to i64, !dbg !17
  %address.base = call i64 asm sideeffect "", "=v,0"(i64 %117) #9, !dbg !17, !frisk.register_lifetimes !1
  %address.local640 = lshr i32 %address.thread562, 1, !dbg !17
  %address.local641 = and i32 %address.local640, 1, !dbg !17
  %address.local642 = zext nneg i32 %address.local641 to i64, !dbg !17
  %address.local643 = add nsw i64 %address.base, %address.local642, !dbg !17
  %address.local644 = shl i64 %address.local643, 6, !dbg !17
  %address.local645 = sub i64 %address.local639, %address.local644, !dbg !17
  %199 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local645, !dbg !17
  %200 = getelementptr [2 x i8], ptr addrspace(3) %199, i64 %address.local614, !dbg !17
  %201 = getelementptr [2 x i8], ptr addrspace(3) %200, i64 %address.local618, !dbg !17
  %address.local649 = getelementptr i8, ptr addrspace(3) %201, i64 64, !dbg !17
  %.idx21002103 = add i64 %address.local645, %address.local614, !dbg !17
  %202 = shl i64 %.idx21002103, 1, !dbg !17
  %203 = add i64 %202, %address.local620.idx, !dbg !17
  %204 = add i64 %203, 64, !dbg !17
  %205 = lshr i64 %204, 6, !dbg !17
  %206 = and i64 %205, 8, !dbg !17
  %207 = xor i64 %204, %206, !dbg !17
  %208 = sub i64 %207, %204, !dbg !17
  %kdma.swizzled1971 = getelementptr i8, ptr addrspace(3) %address.local649, i64 %208, !dbg !17
  %209 = load <4 x half>, ptr addrspace(3) %kdma.swizzled1971, align 8, !dbg !17, !frisk.k_lds_pair !36
  %kdma.gep1975 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1971, i64 128, !dbg !17
  %210 = load <4 x half>, ptr addrspace(3) %kdma.gep1975, align 8, !dbg !17, !frisk.k_lds_pair !36
  %kdma.gep1972 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1971, i64 4096, !dbg !17
  %211 = load <4 x half>, ptr addrspace(3) %kdma.gep1972, align 8, !dbg !17, !frisk.k_lds_pair !37
  %kdma.gep1976 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1971, i64 4224, !dbg !17
  %212 = load <4 x half>, ptr addrspace(3) %kdma.gep1976, align 8, !dbg !17, !frisk.k_lds_pair !37
  %kdma.gep1973 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 128, !dbg !17
  %213 = load <4 x half>, ptr addrspace(3) %kdma.gep1973, align 8, !dbg !17, !frisk.k_lds_pair !38
  %kdma.gep1977 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1024, !dbg !17
  %214 = load <4 x half>, ptr addrspace(3) %kdma.gep1977, align 8, !dbg !17, !frisk.k_lds_pair !38
  %kdma.gep1974 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4224, !dbg !17
  %215 = load <4 x half>, ptr addrspace(3) %kdma.gep1974, align 8, !dbg !17, !frisk.k_lds_pair !39
  %kdma.gep1978 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5120, !dbg !17
  %216 = load <4 x half>, ptr addrspace(3) %kdma.gep1978, align 8, !dbg !17, !frisk.k_lds_pair !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_lds_prefetch !1
  %217 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %198, <4 x float> %197) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_lds_prefetch !1
  %kdma.gep1979 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1971, i64 1024, !dbg !17
  %218 = load <4 x half>, ptr addrspace(3) %kdma.gep1979, align 8, !dbg !17, !frisk.k_lds_pair !34
  %kdma.gep1983 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1971, i64 1152, !dbg !17
  %219 = load <4 x half>, ptr addrspace(3) %kdma.gep1983, align 8, !dbg !17, !frisk.k_lds_pair !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_lds_prefetch !1
  %220 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %209, <4 x float> zeroinitializer) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_lds_prefetch !1
  %kdma.gep1980 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1971, i64 5120, !dbg !17
  %221 = load <4 x half>, ptr addrspace(3) %kdma.gep1980, align 8, !dbg !17, !frisk.k_lds_pair !40
  %kdma.gep1984 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1971, i64 5248, !dbg !17
  %222 = load <4 x half>, ptr addrspace(3) %kdma.gep1984, align 8, !dbg !17, !frisk.k_lds_pair !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_lds_prefetch !1
  %223 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %211, <4 x float> %220) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_lds_prefetch !1
  %kdma.gep1981 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1152, !dbg !17
  %224 = load <4 x half>, ptr addrspace(3) %kdma.gep1981, align 8, !dbg !17, !frisk.k_lds_pair !41
  %kdma.gep1982 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5248, !dbg !17
  %225 = load <4 x half>, ptr addrspace(3) %kdma.gep1982, align 8, !dbg !17, !frisk.k_lds_pair !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_lds_prefetch !1
  %226 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %213, <4 x float> zeroinitializer) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_lds_prefetch !1
  %227 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %215, <4 x float> %226) #9, !dbg !17
  %228 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %210, <4 x float> zeroinitializer) #9, !dbg !17
  %229 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %212, <4 x float> %228) #9, !dbg !17
  %230 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %214, <4 x float> zeroinitializer) #9, !dbg !17
  %231 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %216, <4 x float> %230) #9, !dbg !17
  %232 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %218, <4 x float> zeroinitializer) #9, !dbg !17
  %233 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %221, <4 x float> %232) #9, !dbg !17
  %234 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %224, <4 x float> zeroinitializer) #9, !dbg !17
  %235 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %225, <4 x float> %234) #9, !dbg !17
  %236 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %219, <4 x float> zeroinitializer) #9, !dbg !17
  %237 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %222, <4 x float> %236) #9, !dbg !17
  %238 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %76, <4 x half> %196, <4 x float> zeroinitializer) #9, !dbg !17
  %239 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %198, <4 x float> %238) #9, !dbg !17
  %240 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %76, <4 x half> %209, <4 x float> zeroinitializer) #9, !dbg !17
  %241 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %211, <4 x float> %240) #9, !dbg !17
  %242 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %76, <4 x half> %213, <4 x float> zeroinitializer) #9, !dbg !17
  %243 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %215, <4 x float> %242) #9, !dbg !17
  %244 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %76, <4 x half> %210, <4 x float> zeroinitializer) #9, !dbg !17
  %245 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %212, <4 x float> %244) #9, !dbg !17
  %246 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %76, <4 x half> %214, <4 x float> zeroinitializer) #9, !dbg !17
  %247 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %216, <4 x float> %246) #9, !dbg !17
  %248 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %76, <4 x half> %218, <4 x float> zeroinitializer) #9, !dbg !17
  %249 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %221, <4 x float> %248) #9, !dbg !17
  %250 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %76, <4 x half> %224, <4 x float> zeroinitializer) #9, !dbg !17
  %251 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %225, <4 x float> %250) #9, !dbg !17
  %252 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %76, <4 x half> %219, <4 x float> zeroinitializer) #9, !dbg !17
  %253 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %222, <4 x float> %252) #9, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %254 = add i64 %156, 2048, !dbg !16
  %255 = icmp slt i64 %254, 0, !dbg !16
  %256 = sub i64 -2049, %156, !dbg !16
  %257 = select i1 %255, i64 %256, i64 %254, !dbg !16
  %258 = sdiv i64 %257, 4096, !dbg !16
  %.lobit42 = ashr i64 %254, 63, !dbg !16
  %259 = xor i64 %258, %.lobit42, !dbg !16
  %260 = add nsw i64 %163, 64, !dbg !16
  %261 = icmp slt i64 %163, -64, !dbg !16
  %262 = sub nsw i64 -65, %163, !dbg !16
  %263 = select i1 %261, i64 %262, i64 %260, !dbg !16
  %264 = sdiv i64 %263, 128, !dbg !16
  %265 = sext i1 %261 to i64, !dbg !16
  %266 = xor i64 %264, %265, !dbg !16
  %267 = shl nsw i64 %266, 7, !dbg !16
  %268 = sub i64 %111, %267, !dbg !16
  %269 = shl i64 %259, 19, !dbg !16
  %270 = and i64 %269, 16252928, !dbg !16
  %.idx14811485 = add nuw nsw i64 %270, %139, !dbg !16
  %271 = add i64 %.idx14811485, %268, !dbg !16
  %272 = add i64 %271, %146, !dbg !16
  %kdma.source1796 = add i64 %272, %kdma.source1609, !dbg !16
  %kdma.source1797 = trunc i64 %kdma.source1796 to i32, !dbg !16
  %kdma.source1798 = shl i32 %kdma.source1797, 1, !dbg !16
  %kdma.source1799 = add i32 %kdma.source1798, 128, !dbg !16
  %kdma.offset1800 = add i32 %kdma.source1799, %kdma.intra, !dbg !16
  %273 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %49, i32 %kdma.offset1800, <4 x i32> %dma.buffer.rsrc1427) #10, !dbg !16
  %kdma.source1819 = add i64 %272, %kdma.source1629, !dbg !16
  %kdma.source1820 = trunc i64 %kdma.source1819 to i32, !dbg !16
  %kdma.source1821 = shl i32 %kdma.source1820, 1, !dbg !16
  %kdma.source1822 = add i32 %kdma.source1821, 128, !dbg !16
  %kdma.offset1823 = add i32 %kdma.source1822, %kdma.intra.hi, !dbg !16
  %274 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1634, i32 %kdma.offset1823, <4 x i32> %dma.buffer.rsrc1427) #10, !dbg !16
  %kdma.source1844 = add i64 %272, %kdma.source1653, !dbg !16
  %kdma.source1845 = trunc i64 %kdma.source1844 to i32, !dbg !16
  %kdma.source1846 = shl i32 %kdma.source1845, 1, !dbg !16
  %kdma.source1847 = add i32 %kdma.source1846, 128, !dbg !16
  %kdma.offset1848 = add i32 %kdma.source1847, %kdma.intra, !dbg !16
  %275 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %57, i32 %kdma.offset1848, <4 x i32> %dma.buffer.rsrc1427) #10, !dbg !16
  %kdma.source1869 = add i64 %272, %kdma.source1677, !dbg !16
  %kdma.source1870 = trunc i64 %kdma.source1869 to i32, !dbg !16
  %kdma.source1871 = shl i32 %kdma.source1870, 1, !dbg !16
  %kdma.source1872 = add i32 %kdma.source1871, 128, !dbg !16
  %kdma.offset1873 = add i32 %kdma.source1872, %kdma.intra.hi, !dbg !16
  %276 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1682, i32 %kdma.offset1873, <4 x i32> %dma.buffer.rsrc1427) #10, !dbg !16
  %kdma.gep2001 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8192, !dbg !17
  %277 = load <4 x half>, ptr addrspace(3) %kdma.gep2001, align 8, !dbg !17, !frisk.k_lds_pair !42
  %kdma.gep2005 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8320, !dbg !17
  %278 = load <4 x half>, ptr addrspace(3) %kdma.gep2005, align 8, !dbg !17, !frisk.k_lds_pair !42
  %kdma.gep2002 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12288, !dbg !17
  %279 = load <4 x half>, ptr addrspace(3) %kdma.gep2002, align 8, !dbg !17, !frisk.k_lds_pair !43
  %kdma.gep2006 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12416, !dbg !17
  %280 = load <4 x half>, ptr addrspace(3) %kdma.gep2006, align 8, !dbg !17, !frisk.k_lds_pair !43
  %kdma.gep2003 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1971, i64 8192, !dbg !17
  %281 = load <4 x half>, ptr addrspace(3) %kdma.gep2003, align 8, !dbg !17, !frisk.k_lds_pair !44
  %kdma.gep2007 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1971, i64 8320, !dbg !17
  %282 = load <4 x half>, ptr addrspace(3) %kdma.gep2007, align 8, !dbg !17, !frisk.k_lds_pair !44
  %kdma.gep2004 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1971, i64 12288, !dbg !17
  %283 = load <4 x half>, ptr addrspace(3) %kdma.gep2004, align 8, !dbg !17, !frisk.k_lds_pair !45
  %kdma.gep2008 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1971, i64 12416, !dbg !17
  %284 = load <4 x half>, ptr addrspace(3) %kdma.gep2008, align 8, !dbg !17, !frisk.k_lds_pair !45
  %kdma.gep2009 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9216, !dbg !17
  %285 = load <4 x half>, ptr addrspace(3) %kdma.gep2009, align 8, !dbg !17, !frisk.k_lds_pair !46
  %kdma.gep2013 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9344, !dbg !17
  %286 = load <4 x half>, ptr addrspace(3) %kdma.gep2013, align 8, !dbg !17, !frisk.k_lds_pair !46
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_lds_prefetch !1
  %287 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %277, <4 x float> %217) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_lds_prefetch !1
  %kdma.gep2010 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13312, !dbg !17
  %288 = load <4 x half>, ptr addrspace(3) %kdma.gep2010, align 8, !dbg !17, !frisk.k_lds_pair !47
  %kdma.gep2014 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13440, !dbg !17
  %289 = load <4 x half>, ptr addrspace(3) %kdma.gep2014, align 8, !dbg !17, !frisk.k_lds_pair !47
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_lds_prefetch !1
  %290 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %279, <4 x float> %287) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_lds_prefetch !1
  %kdma.gep2011 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1971, i64 9216, !dbg !17
  %291 = load <4 x half>, ptr addrspace(3) %kdma.gep2011, align 8, !dbg !17, !frisk.k_lds_pair !48
  %kdma.gep2015 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1971, i64 9344, !dbg !17
  %292 = load <4 x half>, ptr addrspace(3) %kdma.gep2015, align 8, !dbg !17, !frisk.k_lds_pair !48
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_lds_prefetch !1
  %293 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %281, <4 x float> %223) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_lds_prefetch !1
  %kdma.gep2012 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1971, i64 13312, !dbg !17
  %294 = load <4 x half>, ptr addrspace(3) %kdma.gep2012, align 8, !dbg !17, !frisk.k_lds_pair !49
  %kdma.gep2016 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1971, i64 13440, !dbg !17
  %295 = load <4 x half>, ptr addrspace(3) %kdma.gep2016, align 8, !dbg !17, !frisk.k_lds_pair !49
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_lds_prefetch !1
  %296 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %283, <4 x float> %293) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_lds_prefetch !1
  %297 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %278, <4 x float> %227) #9, !dbg !17
  %298 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %280, <4 x float> %297) #9, !dbg !17
  %299 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %282, <4 x float> %229) #9, !dbg !17
  %300 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %284, <4 x float> %299) #9, !dbg !17
  %301 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %285, <4 x float> %231) #9, !dbg !17
  %302 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %288, <4 x float> %301) #9, !dbg !17
  %303 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %291, <4 x float> %233) #9, !dbg !17
  %304 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %294, <4 x float> %303) #9, !dbg !17
  %305 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %286, <4 x float> %235) #9, !dbg !17
  %306 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %289, <4 x float> %305) #9, !dbg !17
  %307 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %292, <4 x float> %237) #9, !dbg !17
  %308 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %295, <4 x float> %307) #9, !dbg !17
  %309 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %277, <4 x float> %239) #9, !dbg !17
  %310 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %279, <4 x float> %309) #9, !dbg !17
  %311 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %281, <4 x float> %241) #9, !dbg !17
  %312 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %283, <4 x float> %311) #9, !dbg !17
  %313 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %278, <4 x float> %243) #9, !dbg !17
  %314 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %280, <4 x float> %313) #9, !dbg !17
  %315 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %282, <4 x float> %245) #9, !dbg !17
  %316 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %284, <4 x float> %315) #9, !dbg !17
  %317 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %285, <4 x float> %247) #9, !dbg !17
  %318 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %288, <4 x float> %317) #9, !dbg !17
  %319 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %291, <4 x float> %249) #9, !dbg !17
  %320 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %294, <4 x float> %319) #9, !dbg !17
  %321 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %286, <4 x float> %251) #9, !dbg !17
  %322 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %289, <4 x float> %321) #9, !dbg !17
  %323 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %292, <4 x float> %253) #9, !dbg !17
  %324 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %295, <4 x float> %323) #9, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %325 = add i64 %156, 3072, !dbg !16
  %326 = icmp slt i64 %325, 0, !dbg !16
  %327 = sub i64 -3073, %156, !dbg !16
  %328 = select i1 %326, i64 %327, i64 %325, !dbg !16
  %329 = sdiv i64 %328, 4096, !dbg !16
  %.lobit43 = ashr i64 %325, 63, !dbg !16
  %330 = xor i64 %329, %.lobit43, !dbg !16
  %331 = add nsw i64 %163, 96, !dbg !16
  %332 = icmp slt i64 %163, -96, !dbg !16
  %333 = sub nsw i64 -97, %163, !dbg !16
  %334 = select i1 %332, i64 %333, i64 %331, !dbg !16
  %335 = sdiv i64 %334, 128, !dbg !16
  %336 = sext i1 %332 to i64, !dbg !16
  %337 = xor i64 %335, %336, !dbg !16
  %338 = shl nsw i64 %337, 7, !dbg !16
  %339 = sub i64 %111, %338, !dbg !16
  %340 = shl i64 %330, 19, !dbg !16
  %341 = and i64 %340, 16252928, !dbg !16
  %.idx14931497 = add nuw nsw i64 %341, %139, !dbg !16
  %342 = add i64 %.idx14931497, %339, !dbg !16
  %343 = add i64 %342, %146, !dbg !16
  %kdma.source1892 = add i64 %343, %kdma.source1609, !dbg !16
  %kdma.source1893 = trunc i64 %kdma.source1892 to i32, !dbg !16
  %kdma.source1894 = shl i32 %kdma.source1893, 1, !dbg !16
  %kdma.source1895 = add i32 %kdma.source1894, 192, !dbg !16
  %kdma.offset1896 = add i32 %kdma.source1895, %kdma.intra, !dbg !16
  %344 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %62, i32 %kdma.offset1896, <4 x i32> %dma.buffer.rsrc1427) #10, !dbg !16
  %kdma.source1915 = add i64 %343, %kdma.source1629, !dbg !16
  %kdma.source1916 = trunc i64 %kdma.source1915 to i32, !dbg !16
  %kdma.source1917 = shl i32 %kdma.source1916, 1, !dbg !16
  %kdma.source1918 = add i32 %kdma.source1917, 192, !dbg !16
  %kdma.offset1919 = add i32 %kdma.source1918, %kdma.intra.hi, !dbg !16
  %345 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1728, i32 %kdma.offset1919, <4 x i32> %dma.buffer.rsrc1427) #10, !dbg !16
  %kdma.source1940 = add i64 %343, %kdma.source1653, !dbg !16
  %kdma.source1941 = trunc i64 %kdma.source1940 to i32, !dbg !16
  %kdma.source1942 = shl i32 %kdma.source1941, 1, !dbg !16
  %kdma.source1943 = add i32 %kdma.source1942, 192, !dbg !16
  %kdma.offset1944 = add i32 %kdma.source1943, %kdma.intra, !dbg !16
  %346 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %66, i32 %kdma.offset1944, <4 x i32> %dma.buffer.rsrc1427) #10, !dbg !16
  %kdma.source1965 = add i64 %343, %kdma.source1677, !dbg !16
  %kdma.source1966 = trunc i64 %kdma.source1965 to i32, !dbg !16
  %kdma.source1967 = shl i32 %kdma.source1966, 1, !dbg !16
  %kdma.source1968 = add i32 %kdma.source1967, 192, !dbg !16
  %kdma.offset1969 = add i32 %kdma.source1968, %kdma.intra.hi, !dbg !16
  %347 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1778, i32 %kdma.offset1969, <4 x i32> %dma.buffer.rsrc1427) #10, !dbg !16
  %348 = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8, !dbg !17, !frisk.k_lds_pair !50
  %349 = load <4 x half>, ptr addrspace(3) %kdma.gep1973, align 8, !dbg !17, !frisk.k_lds_pair !50
  %350 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8, !dbg !17, !frisk.k_lds_pair !51
  %351 = load <4 x half>, ptr addrspace(3) %kdma.gep1974, align 8, !dbg !17, !frisk.k_lds_pair !51
  %352 = load <4 x half>, ptr addrspace(3) %kdma.swizzled1971, align 8, !dbg !17, !frisk.k_lds_pair !52
  %353 = load <4 x half>, ptr addrspace(3) %kdma.gep1975, align 8, !dbg !17, !frisk.k_lds_pair !52
  %354 = load <4 x half>, ptr addrspace(3) %kdma.gep1972, align 8, !dbg !17, !frisk.k_lds_pair !53
  %355 = load <4 x half>, ptr addrspace(3) %kdma.gep1976, align 8, !dbg !17, !frisk.k_lds_pair !53
  %356 = load <4 x half>, ptr addrspace(3) %kdma.gep1977, align 8, !dbg !17, !frisk.k_lds_pair !54
  %357 = load <4 x half>, ptr addrspace(3) %kdma.gep1981, align 8, !dbg !17, !frisk.k_lds_pair !54
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_lds_prefetch !1
  %358 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %348, <4 x float> %290) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_lds_prefetch !1
  %359 = load <4 x half>, ptr addrspace(3) %kdma.gep1978, align 8, !dbg !17, !frisk.k_lds_pair !55
  %360 = load <4 x half>, ptr addrspace(3) %kdma.gep1982, align 8, !dbg !17, !frisk.k_lds_pair !55
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_lds_prefetch !1
  %361 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %350, <4 x float> %358) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_lds_prefetch !1
  %362 = load <4 x half>, ptr addrspace(3) %kdma.gep1979, align 8, !dbg !17, !frisk.k_lds_pair !56
  %363 = load <4 x half>, ptr addrspace(3) %kdma.gep1983, align 8, !dbg !17, !frisk.k_lds_pair !56
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_lds_prefetch !1
  %364 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %352, <4 x float> %296) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_lds_prefetch !1
  %365 = load <4 x half>, ptr addrspace(3) %kdma.gep1980, align 8, !dbg !17, !frisk.k_lds_pair !57
  %366 = load <4 x half>, ptr addrspace(3) %kdma.gep1984, align 8, !dbg !17, !frisk.k_lds_pair !57
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_lds_prefetch !1
  %367 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %354, <4 x float> %364) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_lds_prefetch !1
  %368 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %349, <4 x float> %298) #9, !dbg !17
  %369 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %351, <4 x float> %368) #9, !dbg !17
  %370 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %353, <4 x float> %300) #9, !dbg !17
  %371 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %355, <4 x float> %370) #9, !dbg !17
  %372 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %356, <4 x float> %302) #9, !dbg !17
  %373 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %359, <4 x float> %372) #9, !dbg !17
  %374 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %362, <4 x float> %304) #9, !dbg !17
  %375 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %365, <4 x float> %374) #9, !dbg !17
  %376 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %357, <4 x float> %306) #9, !dbg !17
  %377 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %360, <4 x float> %376) #9, !dbg !17
  %378 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %363, <4 x float> %308) #9, !dbg !17
  %379 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %366, <4 x float> %378) #9, !dbg !17
  %380 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %348, <4 x float> %310) #9, !dbg !17
  %381 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %350, <4 x float> %380) #9, !dbg !17
  %382 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %352, <4 x float> %312) #9, !dbg !17
  %383 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %354, <4 x float> %382) #9, !dbg !17
  %384 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %349, <4 x float> %314) #9, !dbg !17
  %385 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %351, <4 x float> %384) #9, !dbg !17
  %386 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %353, <4 x float> %316) #9, !dbg !17
  %387 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %355, <4 x float> %386) #9, !dbg !17
  %388 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %356, <4 x float> %318) #9, !dbg !17
  %389 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %359, <4 x float> %388) #9, !dbg !17
  %390 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %362, <4 x float> %320) #9, !dbg !17
  %391 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %365, <4 x float> %390) #9, !dbg !17
  %392 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %357, <4 x float> %322) #9, !dbg !17
  %393 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %360, <4 x float> %392) #9, !dbg !17
  %394 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %363, <4 x float> %324) #9, !dbg !17
  %395 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %366, <4 x float> %394) #9, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %396 = add nuw nsw i64 %139, %30, !dbg !18
  %397 = shl i64 %396, 19, !dbg !18
  %398 = and i64 %397, 16252928, !dbg !18
  %address.local978 = shl nuw nsw i64 %address.local563, 5, !dbg !18
  %address.local979 = and i64 %address.local978, 8064, !dbg !18
  %address.local980 = shl nuw nsw i64 %address.local563, 3, !dbg !9
  %address.local981 = shl nuw nsw i64 %address.local569, 5, !dbg !18
  %address.local982 = sub nsw i64 %address.local980, %address.local981, !dbg !18
  %address.local983 = add nsw i64 %address.local979, %address.local982, !dbg !18
  %399 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %.idx15051507 = add i64 %398, %146, !dbg !18
  %400 = add i64 %.idx15051507, %address.local983, !dbg !18
  %.tr1508 = trunc i64 %400 to i32, !dbg !18
  %dma.buffer.offset1410 = shl i32 %.tr1508, 1, !dbg !18
  %401 = insertelement <2 x i64> poison, i64 %399, i64 0, !dbg !18
  %402 = bitcast <2 x i64> %401 to <4 x i32>, !dbg !18
  %dma.buffer.rsrc1411 = shufflevector <4 x i32> %402, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !18
  %403 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %49, i32 %dma.buffer.offset1410, <4 x i32> %dma.buffer.rsrc1411) #10, !dbg !18
  %address.local985 = shl nuw nsw i32 %address.thread562, 3, !dbg !18
  %address.local986 = and i32 %address.local985, 24, !dbg !18
  %address.local988 = and i32 %address.local636, 8064, !dbg !18
  %address.local989 = or disjoint i32 %address.local986, %address.local988, !dbg !18
  %404 = trunc i64 %.idx15051507 to i32, !dbg !18
  %address.local990 = or disjoint i32 %address.local989, %404, !dbg !18
  %.tr1512 = shl i32 %address.local990, 1, !dbg !18
  %dma.buffer.offset1408 = or i32 %.tr1512, 16384, !dbg !18
  %405 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %57, i32 %dma.buffer.offset1408, <4 x i32> %dma.buffer.rsrc1411) #10, !dbg !18
  %406 = load <4 x half>, ptr addrspace(3) %kdma.gep2001, align 8, !dbg !17, !frisk.k_tail_pair !36
  %407 = load <4 x half>, ptr addrspace(3) %kdma.gep2005, align 8, !dbg !17, !frisk.k_tail_pair !36
  %408 = load <4 x half>, ptr addrspace(3) %kdma.gep2002, align 8, !dbg !17, !frisk.k_tail_pair !37
  %409 = load <4 x half>, ptr addrspace(3) %kdma.gep2006, align 8, !dbg !17, !frisk.k_tail_pair !37
  %410 = load <4 x half>, ptr addrspace(3) %kdma.gep2003, align 8, !dbg !17, !frisk.k_tail_pair !38
  %411 = load <4 x half>, ptr addrspace(3) %kdma.gep2007, align 8, !dbg !17, !frisk.k_tail_pair !38
  %412 = load <4 x half>, ptr addrspace(3) %kdma.gep2004, align 8, !dbg !17, !frisk.k_tail_pair !39
  %413 = load <4 x half>, ptr addrspace(3) %kdma.gep2008, align 8, !dbg !17, !frisk.k_tail_pair !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_tail_prefetch !1
  %414 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %406, <4 x float> %361) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !19, !frisk.k_tail_prefetch !1
  %415 = or disjoint i64 %141, %123, !dbg !19
  %416 = trunc i64 %415 to i32, !dbg !19
  %.not44 = icmp ugt i32 %125, %416, !dbg !19
  %417 = or disjoint i32 %416, 1, !dbg !19
  %.not45 = icmp ugt i32 %125, %417, !dbg !19
  %418 = or disjoint i32 %416, 2, !dbg !19
  %.not46 = icmp ugt i32 %125, %418, !dbg !19
  %419 = or disjoint i32 %416, 3, !dbg !19
  %.not47 = icmp ugt i32 %125, %419, !dbg !19
  %420 = or disjoint i32 %416, 4, !dbg !19
  %.not48 = icmp ugt i32 %125, %420, !dbg !19
  %421 = or disjoint i32 %416, 5, !dbg !19
  %.not49 = icmp ugt i32 %125, %421, !dbg !19
  %422 = or disjoint i32 %416, 6, !dbg !19
  %.not50 = icmp ugt i32 %125, %422, !dbg !19
  %423 = or disjoint i32 %416, 7, !dbg !19
  %.not51 = icmp ugt i32 %125, %423, !dbg !19
  %424 = or disjoint i32 %416, 8, !dbg !19
  %.not52 = icmp ugt i32 %125, %424, !dbg !19
  %425 = or disjoint i32 %416, 9, !dbg !19
  %.not53 = icmp ugt i32 %125, %425, !dbg !19
  %426 = or disjoint i32 %416, 10, !dbg !19
  %.not54 = icmp ugt i32 %125, %426, !dbg !19
  %427 = or disjoint i32 %416, 11, !dbg !19
  %.not55 = icmp ugt i32 %125, %427, !dbg !19
  %428 = or disjoint i32 %416, 12, !dbg !19
  %.not56 = icmp ugt i32 %125, %428, !dbg !19
  %429 = or disjoint i32 %416, 13, !dbg !19
  %.not57 = icmp ugt i32 %125, %429, !dbg !19
  %430 = or disjoint i32 %416, 14, !dbg !19
  %.not58 = icmp ugt i32 %125, %430, !dbg !19
  %431 = or disjoint i32 %416, 15, !dbg !19
  %.not59 = icmp ugt i32 %125, %431, !dbg !19
  %432 = or disjoint i32 %416, 64, !dbg !19
  %.not60 = icmp ugt i32 %125, %432, !dbg !19
  %433 = or disjoint i32 %416, 65, !dbg !19
  %.not61 = icmp ugt i32 %125, %433, !dbg !19
  %434 = or disjoint i32 %416, 66, !dbg !19
  %.not62 = icmp ugt i32 %125, %434, !dbg !19
  %435 = or disjoint i32 %416, 67, !dbg !19
  %.not63 = icmp ugt i32 %125, %435, !dbg !19
  %436 = or disjoint i32 %416, 68, !dbg !19
  %.not64 = icmp ugt i32 %125, %436, !dbg !19
  %437 = or disjoint i32 %416, 69, !dbg !19
  %.not65 = icmp ugt i32 %125, %437, !dbg !19
  %438 = or disjoint i32 %416, 70, !dbg !19
  %.not66 = icmp ugt i32 %125, %438, !dbg !19
  %439 = or disjoint i32 %416, 71, !dbg !19
  %.not67 = icmp ugt i32 %125, %439, !dbg !19
  %440 = or disjoint i32 %416, 72, !dbg !19
  %.not68 = icmp ugt i32 %125, %440, !dbg !19
  %441 = or disjoint i32 %416, 73, !dbg !19
  %.not69 = icmp ugt i32 %125, %441, !dbg !19
  %442 = or disjoint i32 %416, 74, !dbg !19
  %.not70 = icmp ugt i32 %125, %442, !dbg !19
  %443 = or disjoint i32 %416, 75, !dbg !19
  %.not71 = icmp ugt i32 %125, %443, !dbg !19
  %444 = or disjoint i32 %416, 76, !dbg !19
  %.not72 = icmp ugt i32 %125, %444, !dbg !19
  %445 = or disjoint i32 %416, 77, !dbg !19
  %.not73 = icmp ugt i32 %125, %445, !dbg !19
  %446 = or disjoint i32 %416, 78, !dbg !19
  %.not74 = icmp ugt i32 %125, %446, !dbg !19
  %447 = or disjoint i32 %416, 79, !dbg !19
  %.not75 = icmp ugt i32 %125, %447, !dbg !19
  %.not76 = icmp ugt i32 %126, %416, !dbg !19
  %.not77 = icmp ugt i32 %126, %417, !dbg !19
  %.not78 = icmp ugt i32 %126, %418, !dbg !19
  %.not79 = icmp ugt i32 %126, %419, !dbg !19
  %.not80 = icmp ugt i32 %126, %420, !dbg !19
  %.not81 = icmp ugt i32 %126, %421, !dbg !19
  %.not82 = icmp ugt i32 %126, %422, !dbg !19
  %.not83 = icmp ugt i32 %126, %423, !dbg !19
  %.not84 = icmp ugt i32 %126, %424, !dbg !19
  %.not85 = icmp ugt i32 %126, %425, !dbg !19
  %.not86 = icmp ugt i32 %126, %426, !dbg !19
  %.not87 = icmp ugt i32 %126, %427, !dbg !19
  %.not88 = icmp ugt i32 %126, %428, !dbg !19
  %.not89 = icmp ugt i32 %126, %429, !dbg !19
  %.not90 = icmp ugt i32 %126, %430, !dbg !19
  %.not91 = icmp ugt i32 %126, %431, !dbg !19
  %.not92 = icmp ugt i32 %126, %432, !dbg !19
  %.not93 = icmp ugt i32 %126, %433, !dbg !19
  %.not94 = icmp ugt i32 %126, %434, !dbg !19
  %.not95 = icmp ugt i32 %126, %435, !dbg !19
  %.not96 = icmp ugt i32 %126, %436, !dbg !19
  %.not97 = icmp ugt i32 %126, %437, !dbg !19
  %.not98 = icmp ugt i32 %126, %438, !dbg !19
  %.not99 = icmp ugt i32 %126, %439, !dbg !19
  %.not100 = icmp ugt i32 %126, %440, !dbg !19
  %.not101 = icmp ugt i32 %126, %441, !dbg !19
  %.not102 = icmp ugt i32 %126, %442, !dbg !19
  %.not103 = icmp ugt i32 %126, %443, !dbg !19
  %.not104 = icmp ugt i32 %126, %444, !dbg !19
  %.not105 = icmp ugt i32 %126, %445, !dbg !19
  %.not106 = icmp ugt i32 %126, %446, !dbg !19
  %.not107 = icmp ugt i32 %126, %447, !dbg !19
  %448 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %408, <4 x float> %414) #9, !dbg !17
  %449 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %410, <4 x float> %367) #9, !dbg !17
  %450 = extractelement <4 x float> %448, i64 0, !dbg !17
  %451 = select i1 %.not44, float %450, float 0xFFF0000000000000, !dbg !19
  %452 = extractelement <4 x float> %448, i64 1, !dbg !17
  %453 = select i1 %.not45, float %452, float 0xFFF0000000000000, !dbg !19
  %454 = call float @llvm.maxnum.f32(float %451, float 0xFFF0000000000000), !dbg !8, !frisk.local_reduction !1
  %455 = extractelement <4 x float> %448, i64 2, !dbg !17
  %456 = select i1 %.not46, float %455, float 0xFFF0000000000000, !dbg !19
  %457 = extractelement <4 x float> %448, i64 3, !dbg !17
  %458 = select i1 %.not47, float %457, float 0xFFF0000000000000, !dbg !19
  %459 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %412, <4 x float> %449) #9, !dbg !17
  %460 = load <4 x half>, ptr addrspace(3) %kdma.gep2009, align 8, !dbg !17, !frisk.k_tail_pair !34
  %461 = load <4 x half>, ptr addrspace(3) %kdma.gep2013, align 8, !dbg !17, !frisk.k_tail_pair !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_tail_prefetch !1
  %462 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %407, <4 x float> %369) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_tail_prefetch !1
  %463 = extractelement <4 x float> %459, i64 0, !dbg !17
  %464 = select i1 %.not48, float %463, float 0xFFF0000000000000, !dbg !19
  %465 = extractelement <4 x float> %459, i64 1, !dbg !17
  %466 = select i1 %.not49, float %465, float 0xFFF0000000000000, !dbg !19
  %local.reduce200 = call float @llvm.maxnum.f32(float %454, float %464), !dbg !8, !frisk.local_reduction !1
  %467 = extractelement <4 x float> %459, i64 2, !dbg !17
  %468 = select i1 %.not50, float %467, float 0xFFF0000000000000, !dbg !19
  %local.reduce201 = call float @llvm.maxnum.f32(float %453, float %466), !dbg !8, !frisk.local_reduction !1
  %469 = extractelement <4 x float> %459, i64 3, !dbg !17
  %470 = select i1 %.not51, float %469, float 0xFFF0000000000000, !dbg !19
  %local.reduce202 = call float @llvm.maxnum.f32(float %456, float %468), !dbg !8, !frisk.local_reduction !1
  %local.reduce203 = call float @llvm.maxnum.f32(float %458, float %470), !dbg !8, !frisk.local_reduction !1
  %471 = load <4 x half>, ptr addrspace(3) %kdma.gep2010, align 8, !dbg !17, !frisk.k_tail_pair !40
  %472 = load <4 x half>, ptr addrspace(3) %kdma.gep2014, align 8, !dbg !17, !frisk.k_tail_pair !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_tail_prefetch !1
  %473 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %409, <4 x float> %462) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_tail_prefetch !1
  %474 = load <4 x half>, ptr addrspace(3) %kdma.gep2011, align 8, !dbg !17, !frisk.k_tail_pair !41
  %475 = load <4 x half>, ptr addrspace(3) %kdma.gep2015, align 8, !dbg !17, !frisk.k_tail_pair !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_tail_prefetch !1
  %476 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %411, <4 x float> %371) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_tail_prefetch !1
  %477 = extractelement <4 x float> %473, i64 0, !dbg !17
  %478 = select i1 %.not52, float %477, float 0xFFF0000000000000, !dbg !19
  %479 = extractelement <4 x float> %473, i64 1, !dbg !17
  %480 = select i1 %.not53, float %479, float 0xFFF0000000000000, !dbg !19
  %local.reduce204 = call float @llvm.maxnum.f32(float %local.reduce200, float %478), !dbg !8, !frisk.local_reduction !1
  %481 = extractelement <4 x float> %473, i64 2, !dbg !17
  %482 = select i1 %.not54, float %481, float 0xFFF0000000000000, !dbg !19
  %local.reduce205 = call float @llvm.maxnum.f32(float %local.reduce201, float %480), !dbg !8, !frisk.local_reduction !1
  %483 = extractelement <4 x float> %473, i64 3, !dbg !17
  %484 = select i1 %.not55, float %483, float 0xFFF0000000000000, !dbg !19
  %local.reduce206 = call float @llvm.maxnum.f32(float %local.reduce202, float %482), !dbg !8, !frisk.local_reduction !1
  %local.reduce207 = call float @llvm.maxnum.f32(float %local.reduce203, float %484), !dbg !8, !frisk.local_reduction !1
  %485 = load <4 x half>, ptr addrspace(3) %kdma.gep2012, align 8, !dbg !17, !frisk.k_tail_pair !42
  %486 = load <4 x half>, ptr addrspace(3) %kdma.gep2016, align 8, !dbg !17, !frisk.k_tail_pair !42
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_tail_prefetch !1
  %487 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %413, <4 x float> %476) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_tail_prefetch !1
  %488 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %460, <4 x float> %373) #9, !dbg !17
  %489 = extractelement <4 x float> %487, i64 0, !dbg !17
  %490 = select i1 %.not56, float %489, float 0xFFF0000000000000, !dbg !19
  %491 = extractelement <4 x float> %487, i64 1, !dbg !17
  %492 = select i1 %.not57, float %491, float 0xFFF0000000000000, !dbg !19
  %local.reduce208 = call float @llvm.maxnum.f32(float %local.reduce204, float %490), !dbg !8, !frisk.local_reduction !1
  %493 = extractelement <4 x float> %487, i64 2, !dbg !17
  %494 = select i1 %.not58, float %493, float 0xFFF0000000000000, !dbg !19
  %local.reduce209 = call float @llvm.maxnum.f32(float %local.reduce205, float %492), !dbg !8, !frisk.local_reduction !1
  %495 = extractelement <4 x float> %487, i64 3, !dbg !17
  %496 = select i1 %.not59, float %495, float 0xFFF0000000000000, !dbg !19
  %local.reduce210 = call float @llvm.maxnum.f32(float %local.reduce206, float %494), !dbg !8, !frisk.local_reduction !1
  %local.reduce211 = call float @llvm.maxnum.f32(float %local.reduce207, float %496), !dbg !8, !frisk.local_reduction !1
  %497 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %471, <4 x float> %488) #9, !dbg !17
  %498 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %474, <4 x float> %375) #9, !dbg !17
  %499 = extractelement <4 x float> %497, i64 0, !dbg !17
  %500 = select i1 %.not60, float %499, float 0xFFF0000000000000, !dbg !19
  %501 = extractelement <4 x float> %497, i64 1, !dbg !17
  %502 = select i1 %.not61, float %501, float 0xFFF0000000000000, !dbg !19
  %local.reduce212 = call float @llvm.maxnum.f32(float %local.reduce208, float %500), !dbg !8, !frisk.local_reduction !1
  %503 = extractelement <4 x float> %497, i64 2, !dbg !17
  %504 = select i1 %.not62, float %503, float 0xFFF0000000000000, !dbg !19
  %local.reduce213 = call float @llvm.maxnum.f32(float %local.reduce209, float %502), !dbg !8, !frisk.local_reduction !1
  %505 = extractelement <4 x float> %497, i64 3, !dbg !17
  %506 = select i1 %.not63, float %505, float 0xFFF0000000000000, !dbg !19
  %local.reduce214 = call float @llvm.maxnum.f32(float %local.reduce210, float %504), !dbg !8, !frisk.local_reduction !1
  %local.reduce215 = call float @llvm.maxnum.f32(float %local.reduce211, float %506), !dbg !8, !frisk.local_reduction !1
  %507 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %485, <4 x float> %498) #9, !dbg !17
  %508 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %461, <4 x float> %377) #9, !dbg !17
  %509 = extractelement <4 x float> %507, i64 0, !dbg !17
  %510 = select i1 %.not64, float %509, float 0xFFF0000000000000, !dbg !19
  %511 = extractelement <4 x float> %507, i64 1, !dbg !17
  %512 = select i1 %.not65, float %511, float 0xFFF0000000000000, !dbg !19
  %local.reduce216 = call float @llvm.maxnum.f32(float %local.reduce212, float %510), !dbg !8, !frisk.local_reduction !1
  %513 = extractelement <4 x float> %507, i64 2, !dbg !17
  %514 = select i1 %.not66, float %513, float 0xFFF0000000000000, !dbg !19
  %local.reduce217 = call float @llvm.maxnum.f32(float %local.reduce213, float %512), !dbg !8, !frisk.local_reduction !1
  %515 = extractelement <4 x float> %507, i64 3, !dbg !17
  %516 = select i1 %.not67, float %515, float 0xFFF0000000000000, !dbg !19
  %local.reduce218 = call float @llvm.maxnum.f32(float %local.reduce214, float %514), !dbg !8, !frisk.local_reduction !1
  %local.reduce219 = call float @llvm.maxnum.f32(float %local.reduce215, float %516), !dbg !8, !frisk.local_reduction !1
  %517 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %472, <4 x float> %508) #9, !dbg !17
  %518 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %475, <4 x float> %379) #9, !dbg !17
  %519 = extractelement <4 x float> %517, i64 0, !dbg !17
  %520 = select i1 %.not68, float %519, float 0xFFF0000000000000, !dbg !19
  %521 = extractelement <4 x float> %517, i64 1, !dbg !17
  %522 = select i1 %.not69, float %521, float 0xFFF0000000000000, !dbg !19
  %local.reduce220 = call float @llvm.maxnum.f32(float %local.reduce216, float %520), !dbg !8, !frisk.local_reduction !1
  %523 = extractelement <4 x float> %517, i64 2, !dbg !17
  %524 = select i1 %.not70, float %523, float 0xFFF0000000000000, !dbg !19
  %local.reduce221 = call float @llvm.maxnum.f32(float %local.reduce217, float %522), !dbg !8, !frisk.local_reduction !1
  %525 = extractelement <4 x float> %517, i64 3, !dbg !17
  %526 = select i1 %.not71, float %525, float 0xFFF0000000000000, !dbg !19
  %local.reduce222 = call float @llvm.maxnum.f32(float %local.reduce218, float %524), !dbg !8, !frisk.local_reduction !1
  %local.reduce223 = call float @llvm.maxnum.f32(float %local.reduce219, float %526), !dbg !8, !frisk.local_reduction !1
  %527 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %486, <4 x float> %518) #9, !dbg !17
  %528 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %406, <4 x float> %381) #9, !dbg !17
  %529 = extractelement <4 x float> %527, i64 0, !dbg !17
  %530 = select i1 %.not72, float %529, float 0xFFF0000000000000, !dbg !19
  %531 = extractelement <4 x float> %527, i64 1, !dbg !17
  %532 = select i1 %.not73, float %531, float 0xFFF0000000000000, !dbg !19
  %local.reduce224 = call float @llvm.maxnum.f32(float %local.reduce220, float %530), !dbg !8, !frisk.local_reduction !1
  %533 = extractelement <4 x float> %527, i64 2, !dbg !17
  %534 = select i1 %.not74, float %533, float 0xFFF0000000000000, !dbg !19
  %local.reduce225 = call float @llvm.maxnum.f32(float %local.reduce221, float %532), !dbg !8, !frisk.local_reduction !1
  %535 = extractelement <4 x float> %527, i64 3, !dbg !17
  %536 = select i1 %.not75, float %535, float 0xFFF0000000000000, !dbg !19
  %local.reduce226 = call float @llvm.maxnum.f32(float %local.reduce222, float %534), !dbg !8, !frisk.local_reduction !1
  %local.reduce227 = call float @llvm.maxnum.f32(float %local.reduce223, float %536), !dbg !8, !frisk.local_reduction !1
  %local.reduce228 = call float @llvm.maxnum.f32(float %local.reduce224, float %local.reduce225), !dbg !8, !frisk.local_reduction !1
  %local.reduce229 = call float @llvm.maxnum.f32(float %local.reduce226, float %local.reduce227), !dbg !8, !frisk.local_reduction !1
  %537 = call float @llvm.maxnum.f32(float %local.reduce228, float %local.reduce229), !dbg !8, !frisk.local_reduction !1
  %538 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %105, <4 x half> %408, <4 x float> %528) #9, !dbg !17
  %539 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %410, <4 x float> %383) #9, !dbg !17
  %540 = extractelement <4 x float> %538, i64 0, !dbg !17
  %541 = select i1 %.not76, float %540, float 0xFFF0000000000000, !dbg !19
  %542 = extractelement <4 x float> %538, i64 1, !dbg !17
  %543 = select i1 %.not77, float %542, float 0xFFF0000000000000, !dbg !19
  %544 = extractelement <4 x float> %538, i64 2, !dbg !17
  %545 = select i1 %.not78, float %544, float 0xFFF0000000000000, !dbg !19
  %546 = extractelement <4 x float> %538, i64 3, !dbg !17
  %547 = select i1 %.not79, float %546, float 0xFFF0000000000000, !dbg !19
  %548 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %105, <4 x half> %412, <4 x float> %539) #9, !dbg !17
  %549 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %407, <4 x float> %385) #9, !dbg !17
  %550 = extractelement <4 x float> %548, i64 0, !dbg !17
  %551 = select i1 %.not80, float %550, float 0xFFF0000000000000, !dbg !19
  %552 = extractelement <4 x float> %548, i64 1, !dbg !17
  %553 = select i1 %.not81, float %552, float 0xFFF0000000000000, !dbg !19
  %554 = extractelement <4 x float> %548, i64 2, !dbg !17
  %555 = select i1 %.not82, float %554, float 0xFFF0000000000000, !dbg !19
  %556 = extractelement <4 x float> %548, i64 3, !dbg !17
  %557 = select i1 %.not83, float %556, float 0xFFF0000000000000, !dbg !19
  %558 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %105, <4 x half> %409, <4 x float> %549) #9, !dbg !17
  %559 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %411, <4 x float> %387) #9, !dbg !17
  %560 = extractelement <4 x float> %558, i64 0, !dbg !17
  %561 = select i1 %.not84, float %560, float 0xFFF0000000000000, !dbg !19
  %562 = extractelement <4 x float> %558, i64 1, !dbg !17
  %563 = select i1 %.not85, float %562, float 0xFFF0000000000000, !dbg !19
  %564 = extractelement <4 x float> %558, i64 2, !dbg !17
  %565 = select i1 %.not86, float %564, float 0xFFF0000000000000, !dbg !19
  %566 = extractelement <4 x float> %558, i64 3, !dbg !17
  %567 = select i1 %.not87, float %566, float 0xFFF0000000000000, !dbg !19
  %568 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %105, <4 x half> %413, <4 x float> %559) #9, !dbg !17
  %569 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %460, <4 x float> %389) #9, !dbg !17
  %570 = extractelement <4 x float> %568, i64 0, !dbg !17
  %571 = select i1 %.not88, float %570, float 0xFFF0000000000000, !dbg !19
  %572 = extractelement <4 x float> %568, i64 1, !dbg !17
  %573 = select i1 %.not89, float %572, float 0xFFF0000000000000, !dbg !19
  %574 = extractelement <4 x float> %568, i64 2, !dbg !17
  %575 = select i1 %.not90, float %574, float 0xFFF0000000000000, !dbg !19
  %576 = extractelement <4 x float> %568, i64 3, !dbg !17
  %577 = select i1 %.not91, float %576, float 0xFFF0000000000000, !dbg !19
  %578 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %105, <4 x half> %471, <4 x float> %569) #9, !dbg !17
  %579 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %474, <4 x float> %391) #9, !dbg !17
  %580 = extractelement <4 x float> %578, i64 0, !dbg !17
  %581 = select i1 %.not92, float %580, float 0xFFF0000000000000, !dbg !19
  %582 = extractelement <4 x float> %578, i64 1, !dbg !17
  %583 = select i1 %.not93, float %582, float 0xFFF0000000000000, !dbg !19
  %584 = extractelement <4 x float> %578, i64 2, !dbg !17
  %585 = select i1 %.not94, float %584, float 0xFFF0000000000000, !dbg !19
  %586 = extractelement <4 x float> %578, i64 3, !dbg !17
  %587 = select i1 %.not95, float %586, float 0xFFF0000000000000, !dbg !19
  %588 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %105, <4 x half> %485, <4 x float> %579) #9, !dbg !17
  %589 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %461, <4 x float> %393) #9, !dbg !17
  %590 = extractelement <4 x float> %588, i64 0, !dbg !17
  %591 = select i1 %.not96, float %590, float 0xFFF0000000000000, !dbg !19
  %592 = extractelement <4 x float> %588, i64 1, !dbg !17
  %593 = select i1 %.not97, float %592, float 0xFFF0000000000000, !dbg !19
  %594 = extractelement <4 x float> %588, i64 2, !dbg !17
  %595 = select i1 %.not98, float %594, float 0xFFF0000000000000, !dbg !19
  %596 = extractelement <4 x float> %588, i64 3, !dbg !17
  %597 = select i1 %.not99, float %596, float 0xFFF0000000000000, !dbg !19
  %598 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %105, <4 x half> %472, <4 x float> %589) #9, !dbg !17
  %599 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %475, <4 x float> %395) #9, !dbg !17
  %600 = extractelement <4 x float> %598, i64 0, !dbg !17
  %601 = select i1 %.not100, float %600, float 0xFFF0000000000000, !dbg !19
  %602 = extractelement <4 x float> %598, i64 1, !dbg !17
  %603 = select i1 %.not101, float %602, float 0xFFF0000000000000, !dbg !19
  %604 = extractelement <4 x float> %598, i64 2, !dbg !17
  %605 = select i1 %.not102, float %604, float 0xFFF0000000000000, !dbg !19
  %606 = extractelement <4 x float> %598, i64 3, !dbg !17
  %607 = select i1 %.not103, float %606, float 0xFFF0000000000000, !dbg !19
  %608 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %105, <4 x half> %486, <4 x float> %599) #9, !dbg !17
  %609 = extractelement <4 x float> %608, i64 0, !dbg !17
  %610 = select i1 %.not104, float %609, float 0xFFF0000000000000, !dbg !19
  %611 = extractelement <4 x float> %608, i64 1, !dbg !17
  %612 = select i1 %.not105, float %611, float 0xFFF0000000000000, !dbg !19
  %613 = extractelement <4 x float> %608, i64 2, !dbg !17
  %614 = select i1 %.not106, float %613, float 0xFFF0000000000000, !dbg !19
  %615 = extractelement <4 x float> %608, i64 3, !dbg !17
  %616 = select i1 %.not107, float %615, float 0xFFF0000000000000, !dbg !19
  %617 = bitcast float %537 to i32, !dbg !8
  %618 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %617), !dbg !8
  %619 = bitcast i32 %618 to float, !dbg !8
  %620 = call float @llvm.maxnum.f32(float %537, float %619), !dbg !8
  %621 = bitcast float %620 to i32, !dbg !8
  %622 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %621), !dbg !8
  %623 = bitcast i32 %622 to float, !dbg !8
  %624 = call float @llvm.maxnum.f32(float %620, float %623), !dbg !8
  %625 = call float @llvm.maxnum.f32(float %541, float 0xFFF0000000000000), !dbg !8, !frisk.local_reduction !1
  %local.reduce169 = call float @llvm.maxnum.f32(float %625, float %551), !dbg !8, !frisk.local_reduction !1
  %local.reduce170 = call float @llvm.maxnum.f32(float %543, float %553), !dbg !8, !frisk.local_reduction !1
  %local.reduce171 = call float @llvm.maxnum.f32(float %545, float %555), !dbg !8, !frisk.local_reduction !1
  %local.reduce172 = call float @llvm.maxnum.f32(float %547, float %557), !dbg !8, !frisk.local_reduction !1
  %local.reduce173 = call float @llvm.maxnum.f32(float %local.reduce169, float %561), !dbg !8, !frisk.local_reduction !1
  %local.reduce174 = call float @llvm.maxnum.f32(float %local.reduce170, float %563), !dbg !8, !frisk.local_reduction !1
  %local.reduce175 = call float @llvm.maxnum.f32(float %local.reduce171, float %565), !dbg !8, !frisk.local_reduction !1
  %local.reduce176 = call float @llvm.maxnum.f32(float %local.reduce172, float %567), !dbg !8, !frisk.local_reduction !1
  %local.reduce177 = call float @llvm.maxnum.f32(float %local.reduce173, float %571), !dbg !8, !frisk.local_reduction !1
  %local.reduce178 = call float @llvm.maxnum.f32(float %local.reduce174, float %573), !dbg !8, !frisk.local_reduction !1
  %local.reduce179 = call float @llvm.maxnum.f32(float %local.reduce175, float %575), !dbg !8, !frisk.local_reduction !1
  %local.reduce180 = call float @llvm.maxnum.f32(float %local.reduce176, float %577), !dbg !8, !frisk.local_reduction !1
  %local.reduce181 = call float @llvm.maxnum.f32(float %local.reduce177, float %581), !dbg !8, !frisk.local_reduction !1
  %local.reduce182 = call float @llvm.maxnum.f32(float %local.reduce178, float %583), !dbg !8, !frisk.local_reduction !1
  %local.reduce183 = call float @llvm.maxnum.f32(float %local.reduce179, float %585), !dbg !8, !frisk.local_reduction !1
  %local.reduce184 = call float @llvm.maxnum.f32(float %local.reduce180, float %587), !dbg !8, !frisk.local_reduction !1
  %local.reduce185 = call float @llvm.maxnum.f32(float %local.reduce181, float %591), !dbg !8, !frisk.local_reduction !1
  %local.reduce186 = call float @llvm.maxnum.f32(float %local.reduce182, float %593), !dbg !8, !frisk.local_reduction !1
  %local.reduce187 = call float @llvm.maxnum.f32(float %local.reduce183, float %595), !dbg !8, !frisk.local_reduction !1
  %local.reduce188 = call float @llvm.maxnum.f32(float %local.reduce184, float %597), !dbg !8, !frisk.local_reduction !1
  %local.reduce189 = call float @llvm.maxnum.f32(float %local.reduce185, float %601), !dbg !8, !frisk.local_reduction !1
  %local.reduce190 = call float @llvm.maxnum.f32(float %local.reduce186, float %603), !dbg !8, !frisk.local_reduction !1
  %local.reduce191 = call float @llvm.maxnum.f32(float %local.reduce187, float %605), !dbg !8, !frisk.local_reduction !1
  %local.reduce192 = call float @llvm.maxnum.f32(float %local.reduce188, float %607), !dbg !8, !frisk.local_reduction !1
  %local.reduce193 = call float @llvm.maxnum.f32(float %local.reduce189, float %610), !dbg !8, !frisk.local_reduction !1
  %local.reduce194 = call float @llvm.maxnum.f32(float %local.reduce190, float %612), !dbg !8, !frisk.local_reduction !1
  %local.reduce195 = call float @llvm.maxnum.f32(float %local.reduce191, float %614), !dbg !8, !frisk.local_reduction !1
  %local.reduce196 = call float @llvm.maxnum.f32(float %local.reduce192, float %616), !dbg !8, !frisk.local_reduction !1
  %local.reduce197 = call float @llvm.maxnum.f32(float %local.reduce193, float %local.reduce194), !dbg !8, !frisk.local_reduction !1
  %local.reduce198 = call float @llvm.maxnum.f32(float %local.reduce195, float %local.reduce196), !dbg !8, !frisk.local_reduction !1
  %626 = call float @llvm.maxnum.f32(float %local.reduce197, float %local.reduce198), !dbg !8, !frisk.local_reduction !1
  %627 = bitcast float %626 to i32, !dbg !8
  %628 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %627), !dbg !8
  %629 = bitcast i32 %628 to float, !dbg !8
  %630 = call float @llvm.maxnum.f32(float %626, float %629), !dbg !8
  %631 = bitcast float %630 to i32, !dbg !8
  %632 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %631), !dbg !8
  %633 = bitcast i32 %632 to float, !dbg !8
  %634 = call float @llvm.maxnum.f32(float %630, float %633), !dbg !8
  %635 = extractvalue [2 x <1 x float>] %131, 0, !dbg !8
  %636 = insertelement <1 x float> poison, float %624, i64 0, !dbg !8
  %637 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %636, <1 x float> %635), !dbg !8
  %638 = extractvalue [2 x <1 x float>] %131, 1, !dbg !8
  %639 = insertelement <1 x float> poison, float %634, i64 0, !dbg !8
  %640 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %639, <1 x float> %638), !dbg !8
  %641 = insertelement <4 x float> poison, float %451, i64 0, !dbg !23
  %642 = insertelement <4 x float> %641, float %453, i64 1, !dbg !23
  %643 = insertelement <4 x float> %642, float %456, i64 2, !dbg !23
  %644 = insertelement <4 x float> %643, float %458, i64 3, !dbg !23
  %645 = shufflevector <1 x float> %637, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %646 = insertelement <4 x float> poison, float %464, i64 0, !dbg !23
  %647 = insertelement <4 x float> %646, float %466, i64 1, !dbg !23
  %648 = insertelement <4 x float> %647, float %468, i64 2, !dbg !23
  %649 = insertelement <4 x float> %648, float %470, i64 3, !dbg !23
  %650 = insertelement <4 x float> poison, float %478, i64 0, !dbg !23
  %651 = insertelement <4 x float> %650, float %480, i64 1, !dbg !23
  %652 = insertelement <4 x float> %651, float %482, i64 2, !dbg !23
  %653 = insertelement <4 x float> %652, float %484, i64 3, !dbg !23
  %654 = insertelement <4 x float> poison, float %490, i64 0, !dbg !23
  %655 = insertelement <4 x float> %654, float %492, i64 1, !dbg !23
  %656 = insertelement <4 x float> %655, float %494, i64 2, !dbg !23
  %657 = insertelement <4 x float> %656, float %496, i64 3, !dbg !23
  %658 = insertelement <4 x float> poison, float %500, i64 0, !dbg !23
  %659 = insertelement <4 x float> %658, float %502, i64 1, !dbg !23
  %660 = insertelement <4 x float> %659, float %504, i64 2, !dbg !23
  %661 = insertelement <4 x float> %660, float %506, i64 3, !dbg !23
  %662 = insertelement <4 x float> poison, float %510, i64 0, !dbg !23
  %663 = insertelement <4 x float> %662, float %512, i64 1, !dbg !23
  %664 = insertelement <4 x float> %663, float %514, i64 2, !dbg !23
  %665 = insertelement <4 x float> %664, float %516, i64 3, !dbg !23
  %666 = insertelement <4 x float> poison, float %520, i64 0, !dbg !23
  %667 = insertelement <4 x float> %666, float %522, i64 1, !dbg !23
  %668 = insertelement <4 x float> %667, float %524, i64 2, !dbg !23
  %669 = insertelement <4 x float> %668, float %526, i64 3, !dbg !23
  %670 = insertelement <4 x float> poison, float %530, i64 0, !dbg !23
  %671 = insertelement <4 x float> %670, float %532, i64 1, !dbg !23
  %672 = insertelement <4 x float> %671, float %534, i64 2, !dbg !23
  %673 = insertelement <4 x float> %672, float %536, i64 3, !dbg !23
  %674 = insertelement <4 x float> poison, float %541, i64 0, !dbg !23
  %675 = insertelement <4 x float> %674, float %543, i64 1, !dbg !23
  %676 = insertelement <4 x float> %675, float %545, i64 2, !dbg !23
  %677 = insertelement <4 x float> %676, float %547, i64 3, !dbg !23
  %678 = shufflevector <1 x float> %640, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %679 = insertelement <4 x float> poison, float %551, i64 0, !dbg !23
  %680 = insertelement <4 x float> %679, float %553, i64 1, !dbg !23
  %681 = insertelement <4 x float> %680, float %555, i64 2, !dbg !23
  %682 = insertelement <4 x float> %681, float %557, i64 3, !dbg !23
  %683 = insertelement <4 x float> poison, float %561, i64 0, !dbg !23
  %684 = insertelement <4 x float> %683, float %563, i64 1, !dbg !23
  %685 = insertelement <4 x float> %684, float %565, i64 2, !dbg !23
  %686 = insertelement <4 x float> %685, float %567, i64 3, !dbg !23
  %687 = insertelement <4 x float> poison, float %571, i64 0, !dbg !23
  %688 = insertelement <4 x float> %687, float %573, i64 1, !dbg !23
  %689 = insertelement <4 x float> %688, float %575, i64 2, !dbg !23
  %690 = insertelement <4 x float> %689, float %577, i64 3, !dbg !23
  %691 = insertelement <4 x float> poison, float %581, i64 0, !dbg !23
  %692 = insertelement <4 x float> %691, float %583, i64 1, !dbg !23
  %693 = insertelement <4 x float> %692, float %585, i64 2, !dbg !23
  %694 = insertelement <4 x float> %693, float %587, i64 3, !dbg !23
  %695 = insertelement <4 x float> poison, float %591, i64 0, !dbg !23
  %696 = insertelement <4 x float> %695, float %593, i64 1, !dbg !23
  %697 = insertelement <4 x float> %696, float %595, i64 2, !dbg !23
  %698 = insertelement <4 x float> %697, float %597, i64 3, !dbg !23
  %699 = insertelement <4 x float> poison, float %601, i64 0, !dbg !23
  %700 = insertelement <4 x float> %699, float %603, i64 1, !dbg !23
  %701 = insertelement <4 x float> %700, float %605, i64 2, !dbg !23
  %702 = insertelement <4 x float> %701, float %607, i64 3, !dbg !23
  %703 = insertelement <4 x float> poison, float %610, i64 0, !dbg !23
  %704 = insertelement <4 x float> %703, float %612, i64 1, !dbg !23
  %705 = insertelement <4 x float> %704, float %614, i64 2, !dbg !23
  %706 = insertelement <4 x float> %705, float %616, i64 3, !dbg !23
  %707 = fsub <1 x float> %635, %637, !dbg !24
  %708 = extractelement <1 x float> %707, i64 0, !dbg !25
  %709 = call float @llvm.amdgcn.exp2.f32(float %708), !dbg !25
  %710 = insertelement <4 x float> poison, float %709, i64 0, !dbg !30
  %711 = shufflevector <4 x float> %710, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !30
  %712 = extractvalue [2 x <1 x float>] %130, 0, !dbg !26
  %713 = extractelement <1 x float> %712, i64 0, !dbg !26
  %714 = fmul float %713, %709, !dbg !26
  %715 = fsub <1 x float> %638, %640, !dbg !24
  %716 = extractelement <1 x float> %715, i64 0, !dbg !25
  %717 = call float @llvm.amdgcn.exp2.f32(float %716), !dbg !25
  %718 = insertelement <4 x float> poison, float %717, i64 0, !dbg !30
  %719 = shufflevector <4 x float> %718, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !30
  %720 = extractvalue [2 x <1 x float>] %130, 1, !dbg !26
  %721 = extractelement <1 x float> %720, i64 0, !dbg !26
  %722 = fmul float %721, %717, !dbg !26
  %723 = extractvalue [2 x <8 x float>] %129, 0, !dbg !30
  %724 = shufflevector <8 x float> %723, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %725 = fmul <4 x float> %724, %711, !dbg !30
  %726 = shufflevector <8 x float> %723, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %727 = fmul <4 x float> %726, %711, !dbg !30
  %728 = extractvalue [2 x <8 x float>] %129, 1, !dbg !30
  %729 = shufflevector <8 x float> %728, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %730 = fmul <4 x float> %729, %719, !dbg !30
  %731 = shufflevector <8 x float> %728, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %732 = fmul <4 x float> %731, %719, !dbg !30
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %vdb.address1528 = or disjoint i64 %.idx15051507, 32, !dbg !18
  %vdb.address1529 = add i64 %address.local983, %vdb.address1528, !dbg !18
  %vdb.address1530 = trunc i64 %vdb.address1529 to i32, !dbg !18
  %vdb.address1531 = shl i32 %vdb.address1530, 1, !dbg !18
  %vdb.address1539 = or i32 %.tr1512, 16448, !dbg !18
  %vdb.dma.base = add i32 %49, 8192, !dbg !14
  %733 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1531, <4 x i32> %dma.buffer.rsrc1411) #10, !dbg !18
  %vdb.dma.base1544 = add i32 %57, 8192, !dbg !14
  %734 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1544, i32 %vdb.address1539, <4 x i32> %dma.buffer.rsrc1411) #10, !dbg !18
  %address.local994 = and i32 %address.local985, 120, !dbg !14
  %address.local996 = and i32 %address.local636, 1536, !dbg !14
  %address.local997 = or disjoint i32 %address.local994, %address.local996, !dbg !14
  %address.local998 = zext nneg i32 %address.local997 to i64, !dbg !14
  %address.local999 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local998, !dbg !14
  %735 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local999, i32 0) #0, !dbg !14, !frisk.v_double_buffer !35
  %736 = bitcast <4 x i32> %735 to <8 x half>, !dbg !14
  %address.local1000 = and i64 %address.local563, 60, !dbg !14
  %address.local1001 = add nuw nsw i64 %address.local1000, %address.local563, !dbg !14
  %address.local1003.idx = shl nuw nsw i64 %address.local1001, 4, !dbg !14
  %address.local1003 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %address.local1003.idx, !dbg !14
  %address.local1004 = lshr i64 %address.local616, 4, !dbg !14
  %address.local1006.idx = mul nuw nsw i64 %address.local1004, 768, !dbg !14
  %address.local1006 = getelementptr i8, ptr addrspace(3) %address.local1003, i64 %address.local1006.idx, !dbg !14
  %address.local1008.idx = mul nsw i64 %address.local569, -64, !dbg !14
  %address.local1008 = getelementptr i8, ptr addrspace(3) %address.local1006, i64 %address.local1008.idx, !dbg !14
  %address.local1009 = getelementptr i8, ptr addrspace(3) %address.local1008, i64 256, !dbg !14
  %737 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1009, i32 0) #0, !dbg !14, !frisk.v_double_buffer !35
  %738 = bitcast <4 x i32> %737 to <8 x half>, !dbg !14
  %address.local1013 = getelementptr i8, ptr addrspace(3) %address.local1008, i64 512, !dbg !14
  %739 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1013, i32 0) #0, !dbg !14, !frisk.v_double_buffer !35
  %740 = bitcast <4 x i32> %739 to <8 x half>, !dbg !14
  %address.local1017 = getelementptr i8, ptr addrspace(3) %address.local1008, i64 768, !dbg !14
  %741 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1017, i32 0) #0, !dbg !14, !frisk.v_double_buffer !35
  %742 = bitcast <4 x i32> %741 to <8 x half>, !dbg !14
  %address.local1019 = getelementptr i8, ptr addrspace(3) %address.local999, i64 4096, !dbg !14
  %743 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1019, i32 0) #0, !dbg !14, !frisk.v_double_buffer !35
  %744 = bitcast <4 x i32> %743 to <8 x half>, !dbg !14
  %address.local1023 = getelementptr i8, ptr addrspace(3) %address.local1008, i64 4352, !dbg !14
  %745 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1023, i32 0) #0, !dbg !14, !frisk.v_double_buffer !35
  %746 = bitcast <4 x i32> %745 to <8 x half>, !dbg !14
  %address.local1027 = getelementptr i8, ptr addrspace(3) %address.local1008, i64 4608, !dbg !14
  %747 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1027, i32 0) #0, !dbg !14, !frisk.v_double_buffer !35
  %748 = bitcast <4 x i32> %747 to <8 x half>, !dbg !14
  %address.local1031 = getelementptr i8, ptr addrspace(3) %address.local1008, i64 4864, !dbg !14
  %749 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1031, i32 0) #0, !dbg !14, !frisk.v_double_buffer !35
  %750 = shufflevector <8 x half> %736, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %751 = fsub <4 x float> %644, %645, !dbg !23
  %752 = extractelement <4 x float> %751, i64 0, !dbg !27
  %753 = call float @llvm.amdgcn.exp2.f32(float %752), !dbg !27
  %754 = insertelement <4 x float> poison, float %753, i64 0, !dbg !29
  %755 = extractelement <4 x float> %751, i64 1, !dbg !27
  %756 = call float @llvm.amdgcn.exp2.f32(float %755), !dbg !27
  %757 = insertelement <4 x float> %754, float %756, i64 1, !dbg !29
  %758 = extractelement <4 x float> %751, i64 2, !dbg !27
  %759 = call float @llvm.amdgcn.exp2.f32(float %758), !dbg !27
  %760 = insertelement <4 x float> %757, float %759, i64 2, !dbg !29
  %761 = extractelement <4 x float> %751, i64 3, !dbg !27
  %762 = call float @llvm.amdgcn.exp2.f32(float %761), !dbg !27
  %763 = insertelement <4 x float> %760, float %762, i64 3, !dbg !29
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %763) #9, !dbg !29, !frisk.register_lifetimes !1
  %.cvtpack.f0 = extractelement <4 x float> %math.result, i32 0, !dbg !29
  %.cvtpack.f1 = extractelement <4 x float> %math.result, i32 1, !dbg !29
  %.cvtpack.f2 = extractelement <4 x float> %math.result, i32 2, !dbg !29
  %.cvtpack.f3 = extractelement <4 x float> %math.result, i32 3, !dbg !29
  %.cvtpack.p0 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f0, float %.cvtpack.f1), !dbg !29
  %.cvtpack.p1 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f2, float %.cvtpack.f3), !dbg !29
  %.cvtpack.v0 = insertelement <2 x i32> poison, i32 %.cvtpack.p0, i32 0, !dbg !29
  %.cvtpack.v1 = insertelement <2 x i32> %.cvtpack.v0, i32 %.cvtpack.p1, i32 1, !dbg !29
  %764 = bitcast <2 x i32> %.cvtpack.v1 to <4 x half>, !dbg !29
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %764) #9, !dbg !27, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %765 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %750, <4 x float> %725) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %766 = shufflevector <8 x half> %736, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %767 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %766, <4 x float> %727) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %768 = fsub <4 x float> %677, %678, !dbg !23
  %769 = extractelement <4 x float> %768, i64 0, !dbg !27
  %770 = call float @llvm.amdgcn.exp2.f32(float %769), !dbg !27
  %771 = insertelement <4 x float> poison, float %770, i64 0, !dbg !29
  %772 = extractelement <4 x float> %768, i64 1, !dbg !27
  %773 = call float @llvm.amdgcn.exp2.f32(float %772), !dbg !27
  %774 = insertelement <4 x float> %771, float %773, i64 1, !dbg !29
  %775 = extractelement <4 x float> %768, i64 2, !dbg !27
  %776 = call float @llvm.amdgcn.exp2.f32(float %775), !dbg !27
  %777 = insertelement <4 x float> %774, float %776, i64 2, !dbg !29
  %778 = extractelement <4 x float> %768, i64 3, !dbg !27
  %779 = call float @llvm.amdgcn.exp2.f32(float %778), !dbg !27
  %780 = insertelement <4 x float> %777, float %779, i64 3, !dbg !29
  %math.result449 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %780) #9, !dbg !29, !frisk.register_lifetimes !1
  %.cvtpack.f02104 = extractelement <4 x float> %math.result449, i32 0, !dbg !29
  %.cvtpack.f12105 = extractelement <4 x float> %math.result449, i32 1, !dbg !29
  %.cvtpack.f22106 = extractelement <4 x float> %math.result449, i32 2, !dbg !29
  %.cvtpack.f32107 = extractelement <4 x float> %math.result449, i32 3, !dbg !29
  %.cvtpack.p02108 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02104, float %.cvtpack.f12105), !dbg !29
  %.cvtpack.p12109 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22106, float %.cvtpack.f32107), !dbg !29
  %.cvtpack.v02110 = insertelement <2 x i32> poison, i32 %.cvtpack.p02108, i32 0, !dbg !29
  %.cvtpack.v12111 = insertelement <2 x i32> %.cvtpack.v02110, i32 %.cvtpack.p12109, i32 1, !dbg !29
  %781 = bitcast <2 x i32> %.cvtpack.v12111 to <4 x half>, !dbg !29
  %math.packed450 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %781) #9, !dbg !27, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %782 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed450, <4 x half> %750, <4 x float> %730) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %783 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed450, <4 x half> %766, <4 x float> %732) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %784 = shufflevector <8 x half> %738, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %785 = fsub <4 x float> %649, %645, !dbg !23
  %786 = extractelement <4 x float> %785, i64 0, !dbg !27
  %787 = call float @llvm.amdgcn.exp2.f32(float %786), !dbg !27
  %local.reduce138 = fadd float %753, %787, !dbg !28, !frisk.local_reduction !1
  %788 = insertelement <4 x float> poison, float %787, i64 0, !dbg !29
  %789 = extractelement <4 x float> %785, i64 1, !dbg !27
  %790 = call float @llvm.amdgcn.exp2.f32(float %789), !dbg !27
  %local.reduce139 = fadd float %756, %790, !dbg !28, !frisk.local_reduction !1
  %791 = insertelement <4 x float> %788, float %790, i64 1, !dbg !29
  %792 = extractelement <4 x float> %785, i64 2, !dbg !27
  %793 = call float @llvm.amdgcn.exp2.f32(float %792), !dbg !27
  %local.reduce140 = fadd float %759, %793, !dbg !28, !frisk.local_reduction !1
  %794 = insertelement <4 x float> %791, float %793, i64 2, !dbg !29
  %795 = extractelement <4 x float> %785, i64 3, !dbg !27
  %796 = call float @llvm.amdgcn.exp2.f32(float %795), !dbg !27
  %local.reduce141 = fadd float %762, %796, !dbg !28, !frisk.local_reduction !1
  %797 = insertelement <4 x float> %794, float %796, i64 3, !dbg !29
  %math.result435 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %797) #9, !dbg !29, !frisk.register_lifetimes !1
  %.cvtpack.f02112 = extractelement <4 x float> %math.result435, i32 0, !dbg !29
  %.cvtpack.f12113 = extractelement <4 x float> %math.result435, i32 1, !dbg !29
  %.cvtpack.f22114 = extractelement <4 x float> %math.result435, i32 2, !dbg !29
  %.cvtpack.f32115 = extractelement <4 x float> %math.result435, i32 3, !dbg !29
  %.cvtpack.p02116 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02112, float %.cvtpack.f12113), !dbg !29
  %.cvtpack.p12117 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22114, float %.cvtpack.f32115), !dbg !29
  %.cvtpack.v02118 = insertelement <2 x i32> poison, i32 %.cvtpack.p02116, i32 0, !dbg !29
  %.cvtpack.v12119 = insertelement <2 x i32> %.cvtpack.v02118, i32 %.cvtpack.p12117, i32 1, !dbg !29
  %798 = bitcast <2 x i32> %.cvtpack.v12119 to <4 x half>, !dbg !29
  %math.packed436 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %798) #9, !dbg !28, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %799 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed436, <4 x half> %784, <4 x float> %765) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %800 = shufflevector <8 x half> %738, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %801 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed436, <4 x half> %800, <4 x float> %767) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %802 = fsub <4 x float> %682, %678, !dbg !23
  %803 = extractelement <4 x float> %802, i64 0, !dbg !27
  %804 = call float @llvm.amdgcn.exp2.f32(float %803), !dbg !27
  %local.reduce = fadd float %770, %804, !dbg !28, !frisk.local_reduction !1
  %805 = insertelement <4 x float> poison, float %804, i64 0, !dbg !29
  %806 = extractelement <4 x float> %802, i64 1, !dbg !27
  %807 = call float @llvm.amdgcn.exp2.f32(float %806), !dbg !27
  %local.reduce108 = fadd float %773, %807, !dbg !28, !frisk.local_reduction !1
  %808 = insertelement <4 x float> %805, float %807, i64 1, !dbg !29
  %809 = extractelement <4 x float> %802, i64 2, !dbg !27
  %810 = call float @llvm.amdgcn.exp2.f32(float %809), !dbg !27
  %local.reduce109 = fadd float %776, %810, !dbg !28, !frisk.local_reduction !1
  %811 = insertelement <4 x float> %808, float %810, i64 2, !dbg !29
  %812 = extractelement <4 x float> %802, i64 3, !dbg !27
  %813 = call float @llvm.amdgcn.exp2.f32(float %812), !dbg !27
  %local.reduce110 = fadd float %779, %813, !dbg !28, !frisk.local_reduction !1
  %814 = insertelement <4 x float> %811, float %813, i64 3, !dbg !29
  %math.result451 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %814) #9, !dbg !29, !frisk.register_lifetimes !1
  %.cvtpack.f02120 = extractelement <4 x float> %math.result451, i32 0, !dbg !29
  %.cvtpack.f12121 = extractelement <4 x float> %math.result451, i32 1, !dbg !29
  %.cvtpack.f22122 = extractelement <4 x float> %math.result451, i32 2, !dbg !29
  %.cvtpack.f32123 = extractelement <4 x float> %math.result451, i32 3, !dbg !29
  %.cvtpack.p02124 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02120, float %.cvtpack.f12121), !dbg !29
  %.cvtpack.p12125 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22122, float %.cvtpack.f32123), !dbg !29
  %.cvtpack.v02126 = insertelement <2 x i32> poison, i32 %.cvtpack.p02124, i32 0, !dbg !29
  %.cvtpack.v12127 = insertelement <2 x i32> %.cvtpack.v02126, i32 %.cvtpack.p12125, i32 1, !dbg !29
  %815 = bitcast <2 x i32> %.cvtpack.v12127 to <4 x half>, !dbg !29
  %math.packed452 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %815) #9, !dbg !28, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %816 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed452, <4 x half> %784, <4 x float> %782) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %817 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed452, <4 x half> %800, <4 x float> %783) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %818 = shufflevector <8 x half> %740, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %819 = fsub <4 x float> %653, %645, !dbg !23
  %820 = extractelement <4 x float> %819, i64 0, !dbg !27
  %821 = call float @llvm.amdgcn.exp2.f32(float %820), !dbg !27
  %local.reduce142 = fadd float %local.reduce138, %821, !dbg !28, !frisk.local_reduction !1
  %822 = insertelement <4 x float> poison, float %821, i64 0, !dbg !29
  %823 = extractelement <4 x float> %819, i64 1, !dbg !27
  %824 = call float @llvm.amdgcn.exp2.f32(float %823), !dbg !27
  %local.reduce143 = fadd float %local.reduce139, %824, !dbg !28, !frisk.local_reduction !1
  %825 = insertelement <4 x float> %822, float %824, i64 1, !dbg !29
  %826 = extractelement <4 x float> %819, i64 2, !dbg !27
  %827 = call float @llvm.amdgcn.exp2.f32(float %826), !dbg !27
  %local.reduce144 = fadd float %local.reduce140, %827, !dbg !28, !frisk.local_reduction !1
  %828 = insertelement <4 x float> %825, float %827, i64 2, !dbg !29
  %829 = extractelement <4 x float> %819, i64 3, !dbg !27
  %830 = call float @llvm.amdgcn.exp2.f32(float %829), !dbg !27
  %local.reduce145 = fadd float %local.reduce141, %830, !dbg !28, !frisk.local_reduction !1
  %831 = insertelement <4 x float> %828, float %830, i64 3, !dbg !29
  %math.result437 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %831) #9, !dbg !29, !frisk.register_lifetimes !1
  %.cvtpack.f02128 = extractelement <4 x float> %math.result437, i32 0, !dbg !29
  %.cvtpack.f12129 = extractelement <4 x float> %math.result437, i32 1, !dbg !29
  %.cvtpack.f22130 = extractelement <4 x float> %math.result437, i32 2, !dbg !29
  %.cvtpack.f32131 = extractelement <4 x float> %math.result437, i32 3, !dbg !29
  %.cvtpack.p02132 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02128, float %.cvtpack.f12129), !dbg !29
  %.cvtpack.p12133 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22130, float %.cvtpack.f32131), !dbg !29
  %.cvtpack.v02134 = insertelement <2 x i32> poison, i32 %.cvtpack.p02132, i32 0, !dbg !29
  %.cvtpack.v12135 = insertelement <2 x i32> %.cvtpack.v02134, i32 %.cvtpack.p12133, i32 1, !dbg !29
  %832 = bitcast <2 x i32> %.cvtpack.v12135 to <4 x half>, !dbg !29
  %math.packed438 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %832) #9, !dbg !28, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %833 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed438, <4 x half> %818, <4 x float> %799) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %834 = shufflevector <8 x half> %740, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %835 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed438, <4 x half> %834, <4 x float> %801) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %836 = fsub <4 x float> %686, %678, !dbg !23
  %837 = extractelement <4 x float> %836, i64 0, !dbg !27
  %838 = call float @llvm.amdgcn.exp2.f32(float %837), !dbg !27
  %local.reduce111 = fadd float %local.reduce, %838, !dbg !28, !frisk.local_reduction !1
  %839 = insertelement <4 x float> poison, float %838, i64 0, !dbg !29
  %840 = extractelement <4 x float> %836, i64 1, !dbg !27
  %841 = call float @llvm.amdgcn.exp2.f32(float %840), !dbg !27
  %local.reduce112 = fadd float %local.reduce108, %841, !dbg !28, !frisk.local_reduction !1
  %842 = insertelement <4 x float> %839, float %841, i64 1, !dbg !29
  %843 = extractelement <4 x float> %836, i64 2, !dbg !27
  %844 = call float @llvm.amdgcn.exp2.f32(float %843), !dbg !27
  %local.reduce113 = fadd float %local.reduce109, %844, !dbg !28, !frisk.local_reduction !1
  %845 = insertelement <4 x float> %842, float %844, i64 2, !dbg !29
  %846 = extractelement <4 x float> %836, i64 3, !dbg !27
  %847 = call float @llvm.amdgcn.exp2.f32(float %846), !dbg !27
  %local.reduce114 = fadd float %local.reduce110, %847, !dbg !28, !frisk.local_reduction !1
  %848 = insertelement <4 x float> %845, float %847, i64 3, !dbg !29
  %math.result453 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %848) #9, !dbg !29, !frisk.register_lifetimes !1
  %.cvtpack.f02136 = extractelement <4 x float> %math.result453, i32 0, !dbg !29
  %.cvtpack.f12137 = extractelement <4 x float> %math.result453, i32 1, !dbg !29
  %.cvtpack.f22138 = extractelement <4 x float> %math.result453, i32 2, !dbg !29
  %.cvtpack.f32139 = extractelement <4 x float> %math.result453, i32 3, !dbg !29
  %.cvtpack.p02140 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02136, float %.cvtpack.f12137), !dbg !29
  %.cvtpack.p12141 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22138, float %.cvtpack.f32139), !dbg !29
  %.cvtpack.v02142 = insertelement <2 x i32> poison, i32 %.cvtpack.p02140, i32 0, !dbg !29
  %.cvtpack.v12143 = insertelement <2 x i32> %.cvtpack.v02142, i32 %.cvtpack.p12141, i32 1, !dbg !29
  %849 = bitcast <2 x i32> %.cvtpack.v12143 to <4 x half>, !dbg !29
  %math.packed454 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %849) #9, !dbg !28, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %850 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed454, <4 x half> %818, <4 x float> %816) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %851 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed454, <4 x half> %834, <4 x float> %817) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %852 = shufflevector <8 x half> %742, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %853 = fsub <4 x float> %657, %645, !dbg !23
  %854 = extractelement <4 x float> %853, i64 0, !dbg !27
  %855 = call float @llvm.amdgcn.exp2.f32(float %854), !dbg !27
  %local.reduce146 = fadd float %local.reduce142, %855, !dbg !28, !frisk.local_reduction !1
  %856 = insertelement <4 x float> poison, float %855, i64 0, !dbg !29
  %857 = extractelement <4 x float> %853, i64 1, !dbg !27
  %858 = call float @llvm.amdgcn.exp2.f32(float %857), !dbg !27
  %local.reduce147 = fadd float %local.reduce143, %858, !dbg !28, !frisk.local_reduction !1
  %859 = insertelement <4 x float> %856, float %858, i64 1, !dbg !29
  %860 = extractelement <4 x float> %853, i64 2, !dbg !27
  %861 = call float @llvm.amdgcn.exp2.f32(float %860), !dbg !27
  %local.reduce148 = fadd float %local.reduce144, %861, !dbg !28, !frisk.local_reduction !1
  %862 = insertelement <4 x float> %859, float %861, i64 2, !dbg !29
  %863 = extractelement <4 x float> %853, i64 3, !dbg !27
  %864 = call float @llvm.amdgcn.exp2.f32(float %863), !dbg !27
  %local.reduce149 = fadd float %local.reduce145, %864, !dbg !28, !frisk.local_reduction !1
  %865 = insertelement <4 x float> %862, float %864, i64 3, !dbg !29
  %math.result439 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %865) #9, !dbg !29, !frisk.register_lifetimes !1
  %.cvtpack.f02144 = extractelement <4 x float> %math.result439, i32 0, !dbg !29
  %.cvtpack.f12145 = extractelement <4 x float> %math.result439, i32 1, !dbg !29
  %.cvtpack.f22146 = extractelement <4 x float> %math.result439, i32 2, !dbg !29
  %.cvtpack.f32147 = extractelement <4 x float> %math.result439, i32 3, !dbg !29
  %.cvtpack.p02148 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02144, float %.cvtpack.f12145), !dbg !29
  %.cvtpack.p12149 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22146, float %.cvtpack.f32147), !dbg !29
  %.cvtpack.v02150 = insertelement <2 x i32> poison, i32 %.cvtpack.p02148, i32 0, !dbg !29
  %.cvtpack.v12151 = insertelement <2 x i32> %.cvtpack.v02150, i32 %.cvtpack.p12149, i32 1, !dbg !29
  %866 = bitcast <2 x i32> %.cvtpack.v12151 to <4 x half>, !dbg !29
  %math.packed440 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %866) #9, !dbg !28, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %867 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed440, <4 x half> %852, <4 x float> %833) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %868 = shufflevector <8 x half> %742, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %869 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed440, <4 x half> %868, <4 x float> %835) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %870 = fsub <4 x float> %690, %678, !dbg !23
  %871 = extractelement <4 x float> %870, i64 0, !dbg !27
  %872 = call float @llvm.amdgcn.exp2.f32(float %871), !dbg !27
  %local.reduce115 = fadd float %local.reduce111, %872, !dbg !28, !frisk.local_reduction !1
  %873 = insertelement <4 x float> poison, float %872, i64 0, !dbg !29
  %874 = extractelement <4 x float> %870, i64 1, !dbg !27
  %875 = call float @llvm.amdgcn.exp2.f32(float %874), !dbg !27
  %local.reduce116 = fadd float %local.reduce112, %875, !dbg !28, !frisk.local_reduction !1
  %876 = insertelement <4 x float> %873, float %875, i64 1, !dbg !29
  %877 = extractelement <4 x float> %870, i64 2, !dbg !27
  %878 = call float @llvm.amdgcn.exp2.f32(float %877), !dbg !27
  %local.reduce117 = fadd float %local.reduce113, %878, !dbg !28, !frisk.local_reduction !1
  %879 = insertelement <4 x float> %876, float %878, i64 2, !dbg !29
  %880 = extractelement <4 x float> %870, i64 3, !dbg !27
  %881 = call float @llvm.amdgcn.exp2.f32(float %880), !dbg !27
  %local.reduce118 = fadd float %local.reduce114, %881, !dbg !28, !frisk.local_reduction !1
  %882 = insertelement <4 x float> %879, float %881, i64 3, !dbg !29
  %math.result455 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %882) #9, !dbg !29, !frisk.register_lifetimes !1
  %.cvtpack.f02152 = extractelement <4 x float> %math.result455, i32 0, !dbg !29
  %.cvtpack.f12153 = extractelement <4 x float> %math.result455, i32 1, !dbg !29
  %.cvtpack.f22154 = extractelement <4 x float> %math.result455, i32 2, !dbg !29
  %.cvtpack.f32155 = extractelement <4 x float> %math.result455, i32 3, !dbg !29
  %.cvtpack.p02156 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02152, float %.cvtpack.f12153), !dbg !29
  %.cvtpack.p12157 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22154, float %.cvtpack.f32155), !dbg !29
  %.cvtpack.v02158 = insertelement <2 x i32> poison, i32 %.cvtpack.p02156, i32 0, !dbg !29
  %.cvtpack.v12159 = insertelement <2 x i32> %.cvtpack.v02158, i32 %.cvtpack.p12157, i32 1, !dbg !29
  %883 = bitcast <2 x i32> %.cvtpack.v12159 to <4 x half>, !dbg !29
  %math.packed456 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %883) #9, !dbg !28, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %884 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed456, <4 x half> %852, <4 x float> %850) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %885 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed456, <4 x half> %868, <4 x float> %851) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %886 = shufflevector <8 x half> %744, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %887 = fsub <4 x float> %661, %645, !dbg !23
  %888 = extractelement <4 x float> %887, i64 0, !dbg !27
  %889 = call float @llvm.amdgcn.exp2.f32(float %888), !dbg !27
  %local.reduce150 = fadd float %local.reduce146, %889, !dbg !28, !frisk.local_reduction !1
  %890 = insertelement <4 x float> poison, float %889, i64 0, !dbg !29
  %891 = extractelement <4 x float> %887, i64 1, !dbg !27
  %892 = call float @llvm.amdgcn.exp2.f32(float %891), !dbg !27
  %local.reduce151 = fadd float %local.reduce147, %892, !dbg !28, !frisk.local_reduction !1
  %893 = insertelement <4 x float> %890, float %892, i64 1, !dbg !29
  %894 = extractelement <4 x float> %887, i64 2, !dbg !27
  %895 = call float @llvm.amdgcn.exp2.f32(float %894), !dbg !27
  %local.reduce152 = fadd float %local.reduce148, %895, !dbg !28, !frisk.local_reduction !1
  %896 = insertelement <4 x float> %893, float %895, i64 2, !dbg !29
  %897 = extractelement <4 x float> %887, i64 3, !dbg !27
  %898 = call float @llvm.amdgcn.exp2.f32(float %897), !dbg !27
  %local.reduce153 = fadd float %local.reduce149, %898, !dbg !28, !frisk.local_reduction !1
  %899 = insertelement <4 x float> %896, float %898, i64 3, !dbg !29
  %math.result441 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %899) #9, !dbg !29, !frisk.register_lifetimes !1
  %.cvtpack.f02160 = extractelement <4 x float> %math.result441, i32 0, !dbg !29
  %.cvtpack.f12161 = extractelement <4 x float> %math.result441, i32 1, !dbg !29
  %.cvtpack.f22162 = extractelement <4 x float> %math.result441, i32 2, !dbg !29
  %.cvtpack.f32163 = extractelement <4 x float> %math.result441, i32 3, !dbg !29
  %.cvtpack.p02164 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02160, float %.cvtpack.f12161), !dbg !29
  %.cvtpack.p12165 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22162, float %.cvtpack.f32163), !dbg !29
  %.cvtpack.v02166 = insertelement <2 x i32> poison, i32 %.cvtpack.p02164, i32 0, !dbg !29
  %.cvtpack.v12167 = insertelement <2 x i32> %.cvtpack.v02166, i32 %.cvtpack.p12165, i32 1, !dbg !29
  %900 = bitcast <2 x i32> %.cvtpack.v12167 to <4 x half>, !dbg !29
  %math.packed442 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %900) #9, !dbg !28, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %901 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed442, <4 x half> %886, <4 x float> %867) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %902 = shufflevector <8 x half> %744, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %903 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed442, <4 x half> %902, <4 x float> %869) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %904 = fsub <4 x float> %694, %678, !dbg !23
  %905 = extractelement <4 x float> %904, i64 0, !dbg !27
  %906 = call float @llvm.amdgcn.exp2.f32(float %905), !dbg !27
  %local.reduce119 = fadd float %local.reduce115, %906, !dbg !28, !frisk.local_reduction !1
  %907 = insertelement <4 x float> poison, float %906, i64 0, !dbg !29
  %908 = extractelement <4 x float> %904, i64 1, !dbg !27
  %909 = call float @llvm.amdgcn.exp2.f32(float %908), !dbg !27
  %local.reduce120 = fadd float %local.reduce116, %909, !dbg !28, !frisk.local_reduction !1
  %910 = insertelement <4 x float> %907, float %909, i64 1, !dbg !29
  %911 = extractelement <4 x float> %904, i64 2, !dbg !27
  %912 = call float @llvm.amdgcn.exp2.f32(float %911), !dbg !27
  %local.reduce121 = fadd float %local.reduce117, %912, !dbg !28, !frisk.local_reduction !1
  %913 = insertelement <4 x float> %910, float %912, i64 2, !dbg !29
  %914 = extractelement <4 x float> %904, i64 3, !dbg !27
  %915 = call float @llvm.amdgcn.exp2.f32(float %914), !dbg !27
  %local.reduce122 = fadd float %local.reduce118, %915, !dbg !28, !frisk.local_reduction !1
  %916 = insertelement <4 x float> %913, float %915, i64 3, !dbg !29
  %math.result457 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %916) #9, !dbg !29, !frisk.register_lifetimes !1
  %.cvtpack.f02168 = extractelement <4 x float> %math.result457, i32 0, !dbg !29
  %.cvtpack.f12169 = extractelement <4 x float> %math.result457, i32 1, !dbg !29
  %.cvtpack.f22170 = extractelement <4 x float> %math.result457, i32 2, !dbg !29
  %.cvtpack.f32171 = extractelement <4 x float> %math.result457, i32 3, !dbg !29
  %.cvtpack.p02172 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02168, float %.cvtpack.f12169), !dbg !29
  %.cvtpack.p12173 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22170, float %.cvtpack.f32171), !dbg !29
  %.cvtpack.v02174 = insertelement <2 x i32> poison, i32 %.cvtpack.p02172, i32 0, !dbg !29
  %.cvtpack.v12175 = insertelement <2 x i32> %.cvtpack.v02174, i32 %.cvtpack.p12173, i32 1, !dbg !29
  %917 = bitcast <2 x i32> %.cvtpack.v12175 to <4 x half>, !dbg !29
  %math.packed458 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %917) #9, !dbg !28, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %918 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed458, <4 x half> %886, <4 x float> %884) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %919 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed458, <4 x half> %902, <4 x float> %885) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %920 = shufflevector <8 x half> %746, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %921 = fsub <4 x float> %665, %645, !dbg !23
  %922 = extractelement <4 x float> %921, i64 0, !dbg !27
  %923 = call float @llvm.amdgcn.exp2.f32(float %922), !dbg !27
  %local.reduce154 = fadd float %local.reduce150, %923, !dbg !28, !frisk.local_reduction !1
  %924 = insertelement <4 x float> poison, float %923, i64 0, !dbg !29
  %925 = extractelement <4 x float> %921, i64 1, !dbg !27
  %926 = call float @llvm.amdgcn.exp2.f32(float %925), !dbg !27
  %local.reduce155 = fadd float %local.reduce151, %926, !dbg !28, !frisk.local_reduction !1
  %927 = insertelement <4 x float> %924, float %926, i64 1, !dbg !29
  %928 = extractelement <4 x float> %921, i64 2, !dbg !27
  %929 = call float @llvm.amdgcn.exp2.f32(float %928), !dbg !27
  %local.reduce156 = fadd float %local.reduce152, %929, !dbg !28, !frisk.local_reduction !1
  %930 = insertelement <4 x float> %927, float %929, i64 2, !dbg !29
  %931 = extractelement <4 x float> %921, i64 3, !dbg !27
  %932 = call float @llvm.amdgcn.exp2.f32(float %931), !dbg !27
  %local.reduce157 = fadd float %local.reduce153, %932, !dbg !28, !frisk.local_reduction !1
  %933 = insertelement <4 x float> %930, float %932, i64 3, !dbg !29
  %math.result443 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %933) #9, !dbg !29, !frisk.register_lifetimes !1
  %.cvtpack.f02176 = extractelement <4 x float> %math.result443, i32 0, !dbg !29
  %.cvtpack.f12177 = extractelement <4 x float> %math.result443, i32 1, !dbg !29
  %.cvtpack.f22178 = extractelement <4 x float> %math.result443, i32 2, !dbg !29
  %.cvtpack.f32179 = extractelement <4 x float> %math.result443, i32 3, !dbg !29
  %.cvtpack.p02180 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02176, float %.cvtpack.f12177), !dbg !29
  %.cvtpack.p12181 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22178, float %.cvtpack.f32179), !dbg !29
  %.cvtpack.v02182 = insertelement <2 x i32> poison, i32 %.cvtpack.p02180, i32 0, !dbg !29
  %.cvtpack.v12183 = insertelement <2 x i32> %.cvtpack.v02182, i32 %.cvtpack.p12181, i32 1, !dbg !29
  %934 = bitcast <2 x i32> %.cvtpack.v12183 to <4 x half>, !dbg !29
  %math.packed444 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %934) #9, !dbg !28, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %935 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed444, <4 x half> %920, <4 x float> %901) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %936 = shufflevector <8 x half> %746, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %937 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed444, <4 x half> %936, <4 x float> %903) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %938 = fsub <4 x float> %698, %678, !dbg !23
  %939 = extractelement <4 x float> %938, i64 0, !dbg !27
  %940 = call float @llvm.amdgcn.exp2.f32(float %939), !dbg !27
  %local.reduce123 = fadd float %local.reduce119, %940, !dbg !28, !frisk.local_reduction !1
  %941 = insertelement <4 x float> poison, float %940, i64 0, !dbg !29
  %942 = extractelement <4 x float> %938, i64 1, !dbg !27
  %943 = call float @llvm.amdgcn.exp2.f32(float %942), !dbg !27
  %local.reduce124 = fadd float %local.reduce120, %943, !dbg !28, !frisk.local_reduction !1
  %944 = insertelement <4 x float> %941, float %943, i64 1, !dbg !29
  %945 = extractelement <4 x float> %938, i64 2, !dbg !27
  %946 = call float @llvm.amdgcn.exp2.f32(float %945), !dbg !27
  %local.reduce125 = fadd float %local.reduce121, %946, !dbg !28, !frisk.local_reduction !1
  %947 = insertelement <4 x float> %944, float %946, i64 2, !dbg !29
  %948 = extractelement <4 x float> %938, i64 3, !dbg !27
  %949 = call float @llvm.amdgcn.exp2.f32(float %948), !dbg !27
  %local.reduce126 = fadd float %local.reduce122, %949, !dbg !28, !frisk.local_reduction !1
  %950 = insertelement <4 x float> %947, float %949, i64 3, !dbg !29
  %math.result459 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %950) #9, !dbg !29, !frisk.register_lifetimes !1
  %.cvtpack.f02184 = extractelement <4 x float> %math.result459, i32 0, !dbg !29
  %.cvtpack.f12185 = extractelement <4 x float> %math.result459, i32 1, !dbg !29
  %.cvtpack.f22186 = extractelement <4 x float> %math.result459, i32 2, !dbg !29
  %.cvtpack.f32187 = extractelement <4 x float> %math.result459, i32 3, !dbg !29
  %.cvtpack.p02188 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02184, float %.cvtpack.f12185), !dbg !29
  %.cvtpack.p12189 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22186, float %.cvtpack.f32187), !dbg !29
  %.cvtpack.v02190 = insertelement <2 x i32> poison, i32 %.cvtpack.p02188, i32 0, !dbg !29
  %.cvtpack.v12191 = insertelement <2 x i32> %.cvtpack.v02190, i32 %.cvtpack.p12189, i32 1, !dbg !29
  %951 = bitcast <2 x i32> %.cvtpack.v12191 to <4 x half>, !dbg !29
  %math.packed460 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %951) #9, !dbg !28, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %952 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed460, <4 x half> %920, <4 x float> %918) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %953 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed460, <4 x half> %936, <4 x float> %919) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %954 = shufflevector <8 x half> %748, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %955 = fsub <4 x float> %669, %645, !dbg !23
  %956 = extractelement <4 x float> %955, i64 0, !dbg !27
  %957 = call float @llvm.amdgcn.exp2.f32(float %956), !dbg !27
  %local.reduce158 = fadd float %local.reduce154, %957, !dbg !28, !frisk.local_reduction !1
  %958 = insertelement <4 x float> poison, float %957, i64 0, !dbg !29
  %959 = extractelement <4 x float> %955, i64 1, !dbg !27
  %960 = call float @llvm.amdgcn.exp2.f32(float %959), !dbg !27
  %local.reduce159 = fadd float %local.reduce155, %960, !dbg !28, !frisk.local_reduction !1
  %961 = insertelement <4 x float> %958, float %960, i64 1, !dbg !29
  %962 = extractelement <4 x float> %955, i64 2, !dbg !27
  %963 = call float @llvm.amdgcn.exp2.f32(float %962), !dbg !27
  %local.reduce160 = fadd float %local.reduce156, %963, !dbg !28, !frisk.local_reduction !1
  %964 = insertelement <4 x float> %961, float %963, i64 2, !dbg !29
  %965 = extractelement <4 x float> %955, i64 3, !dbg !27
  %966 = call float @llvm.amdgcn.exp2.f32(float %965), !dbg !27
  %local.reduce161 = fadd float %local.reduce157, %966, !dbg !28, !frisk.local_reduction !1
  %967 = insertelement <4 x float> %964, float %966, i64 3, !dbg !29
  %math.result445 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %967) #9, !dbg !29, !frisk.register_lifetimes !1
  %.cvtpack.f02192 = extractelement <4 x float> %math.result445, i32 0, !dbg !29
  %.cvtpack.f12193 = extractelement <4 x float> %math.result445, i32 1, !dbg !29
  %.cvtpack.f22194 = extractelement <4 x float> %math.result445, i32 2, !dbg !29
  %.cvtpack.f32195 = extractelement <4 x float> %math.result445, i32 3, !dbg !29
  %.cvtpack.p02196 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02192, float %.cvtpack.f12193), !dbg !29
  %.cvtpack.p12197 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22194, float %.cvtpack.f32195), !dbg !29
  %.cvtpack.v02198 = insertelement <2 x i32> poison, i32 %.cvtpack.p02196, i32 0, !dbg !29
  %.cvtpack.v12199 = insertelement <2 x i32> %.cvtpack.v02198, i32 %.cvtpack.p12197, i32 1, !dbg !29
  %968 = bitcast <2 x i32> %.cvtpack.v12199 to <4 x half>, !dbg !29
  %math.packed446 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %968) #9, !dbg !28, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %969 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed446, <4 x half> %954, <4 x float> %935) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %970 = shufflevector <8 x half> %748, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %971 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed446, <4 x half> %970, <4 x float> %937) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %972 = fsub <4 x float> %702, %678, !dbg !23
  %973 = extractelement <4 x float> %972, i64 0, !dbg !27
  %974 = call float @llvm.amdgcn.exp2.f32(float %973), !dbg !27
  %local.reduce127 = fadd float %local.reduce123, %974, !dbg !28, !frisk.local_reduction !1
  %975 = insertelement <4 x float> poison, float %974, i64 0, !dbg !29
  %976 = extractelement <4 x float> %972, i64 1, !dbg !27
  %977 = call float @llvm.amdgcn.exp2.f32(float %976), !dbg !27
  %local.reduce128 = fadd float %local.reduce124, %977, !dbg !28, !frisk.local_reduction !1
  %978 = insertelement <4 x float> %975, float %977, i64 1, !dbg !29
  %979 = extractelement <4 x float> %972, i64 2, !dbg !27
  %980 = call float @llvm.amdgcn.exp2.f32(float %979), !dbg !27
  %local.reduce129 = fadd float %local.reduce125, %980, !dbg !28, !frisk.local_reduction !1
  %981 = insertelement <4 x float> %978, float %980, i64 2, !dbg !29
  %982 = extractelement <4 x float> %972, i64 3, !dbg !27
  %983 = call float @llvm.amdgcn.exp2.f32(float %982), !dbg !27
  %local.reduce130 = fadd float %local.reduce126, %983, !dbg !28, !frisk.local_reduction !1
  %984 = insertelement <4 x float> %981, float %983, i64 3, !dbg !29
  %math.result461 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %984) #9, !dbg !29, !frisk.register_lifetimes !1
  %.cvtpack.f02200 = extractelement <4 x float> %math.result461, i32 0, !dbg !29
  %.cvtpack.f12201 = extractelement <4 x float> %math.result461, i32 1, !dbg !29
  %.cvtpack.f22202 = extractelement <4 x float> %math.result461, i32 2, !dbg !29
  %.cvtpack.f32203 = extractelement <4 x float> %math.result461, i32 3, !dbg !29
  %.cvtpack.p02204 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02200, float %.cvtpack.f12201), !dbg !29
  %.cvtpack.p12205 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22202, float %.cvtpack.f32203), !dbg !29
  %.cvtpack.v02206 = insertelement <2 x i32> poison, i32 %.cvtpack.p02204, i32 0, !dbg !29
  %.cvtpack.v12207 = insertelement <2 x i32> %.cvtpack.v02206, i32 %.cvtpack.p12205, i32 1, !dbg !29
  %985 = bitcast <2 x i32> %.cvtpack.v12207 to <4 x half>, !dbg !29
  %math.packed462 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %985) #9, !dbg !28, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %986 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed462, <4 x half> %954, <4 x float> %952) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %987 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed462, <4 x half> %970, <4 x float> %953) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %988 = bitcast <4 x i32> %749 to <8 x half>, !dbg !14
  %989 = shufflevector <8 x half> %988, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %990 = fsub <4 x float> %673, %645, !dbg !23
  %991 = extractelement <4 x float> %990, i64 0, !dbg !27
  %992 = call float @llvm.amdgcn.exp2.f32(float %991), !dbg !27
  %local.reduce162 = fadd float %local.reduce158, %992, !dbg !28, !frisk.local_reduction !1
  %993 = insertelement <4 x float> poison, float %992, i64 0, !dbg !29
  %994 = extractelement <4 x float> %990, i64 1, !dbg !27
  %995 = call float @llvm.amdgcn.exp2.f32(float %994), !dbg !27
  %local.reduce163 = fadd float %local.reduce159, %995, !dbg !28, !frisk.local_reduction !1
  %local.reduce166 = fadd float %local.reduce162, %local.reduce163, !dbg !28, !frisk.local_reduction !1
  %996 = insertelement <4 x float> %993, float %995, i64 1, !dbg !29
  %997 = extractelement <4 x float> %990, i64 2, !dbg !27
  %998 = call float @llvm.amdgcn.exp2.f32(float %997), !dbg !27
  %local.reduce164 = fadd float %local.reduce160, %998, !dbg !28, !frisk.local_reduction !1
  %999 = insertelement <4 x float> %996, float %998, i64 2, !dbg !29
  %1000 = extractelement <4 x float> %990, i64 3, !dbg !27
  %1001 = call float @llvm.amdgcn.exp2.f32(float %1000), !dbg !27
  %local.reduce165 = fadd float %local.reduce161, %1001, !dbg !28, !frisk.local_reduction !1
  %local.reduce167 = fadd float %local.reduce164, %local.reduce165, !dbg !28, !frisk.local_reduction !1
  %1002 = fadd float %local.reduce166, %local.reduce167, !dbg !28, !frisk.local_reduction !1
  %1003 = fadd float %1002, %714, !dbg !28
  %1004 = insertelement <1 x float> poison, float %1003, i64 0, !dbg !6
  %1005 = insertvalue [2 x <1 x float>] poison, <1 x float> %1004, 0, !dbg !6
  %1006 = insertelement <4 x float> %999, float %1001, i64 3, !dbg !29
  %math.result447 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1006) #9, !dbg !29, !frisk.register_lifetimes !1
  %.cvtpack.f02208 = extractelement <4 x float> %math.result447, i32 0, !dbg !29
  %.cvtpack.f12209 = extractelement <4 x float> %math.result447, i32 1, !dbg !29
  %.cvtpack.f22210 = extractelement <4 x float> %math.result447, i32 2, !dbg !29
  %.cvtpack.f32211 = extractelement <4 x float> %math.result447, i32 3, !dbg !29
  %.cvtpack.p02212 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02208, float %.cvtpack.f12209), !dbg !29
  %.cvtpack.p12213 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22210, float %.cvtpack.f32211), !dbg !29
  %.cvtpack.v02214 = insertelement <2 x i32> poison, i32 %.cvtpack.p02212, i32 0, !dbg !29
  %.cvtpack.v12215 = insertelement <2 x i32> %.cvtpack.v02214, i32 %.cvtpack.p12213, i32 1, !dbg !29
  %1007 = bitcast <2 x i32> %.cvtpack.v12215 to <4 x half>, !dbg !29
  %math.packed448 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1007) #9, !dbg !28, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %1008 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed448, <4 x half> %989, <4 x float> %969) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1009 = shufflevector <8 x half> %988, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1010 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed448, <4 x half> %1009, <4 x float> %971) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1011 = fsub <4 x float> %706, %678, !dbg !23
  %1012 = extractelement <4 x float> %1011, i64 0, !dbg !27
  %1013 = call float @llvm.amdgcn.exp2.f32(float %1012), !dbg !27
  %local.reduce131 = fadd float %local.reduce127, %1013, !dbg !28, !frisk.local_reduction !1
  %1014 = insertelement <4 x float> poison, float %1013, i64 0, !dbg !29
  %1015 = extractelement <4 x float> %1011, i64 1, !dbg !27
  %1016 = call float @llvm.amdgcn.exp2.f32(float %1015), !dbg !27
  %local.reduce132 = fadd float %local.reduce128, %1016, !dbg !28, !frisk.local_reduction !1
  %local.reduce135 = fadd float %local.reduce131, %local.reduce132, !dbg !28, !frisk.local_reduction !1
  %1017 = insertelement <4 x float> %1014, float %1016, i64 1, !dbg !29
  %1018 = extractelement <4 x float> %1011, i64 2, !dbg !27
  %1019 = call float @llvm.amdgcn.exp2.f32(float %1018), !dbg !27
  %local.reduce133 = fadd float %local.reduce129, %1019, !dbg !28, !frisk.local_reduction !1
  %1020 = insertelement <4 x float> %1017, float %1019, i64 2, !dbg !29
  %1021 = extractelement <4 x float> %1011, i64 3, !dbg !27
  %1022 = call float @llvm.amdgcn.exp2.f32(float %1021), !dbg !27
  %local.reduce134 = fadd float %local.reduce130, %1022, !dbg !28, !frisk.local_reduction !1
  %local.reduce136 = fadd float %local.reduce133, %local.reduce134, !dbg !28, !frisk.local_reduction !1
  %1023 = fadd float %local.reduce135, %local.reduce136, !dbg !28, !frisk.local_reduction !1
  %1024 = fadd float %1023, %722, !dbg !28
  %1025 = insertelement <1 x float> poison, float %1024, i64 0, !dbg !6
  %1026 = insertvalue [2 x <1 x float>] %1005, <1 x float> %1025, 1, !dbg !6
  %1027 = insertelement <4 x float> %1020, float %1022, i64 3, !dbg !29
  %math.result463 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1027) #9, !dbg !29, !frisk.register_lifetimes !1
  %.cvtpack.f02216 = extractelement <4 x float> %math.result463, i32 0, !dbg !29
  %.cvtpack.f12217 = extractelement <4 x float> %math.result463, i32 1, !dbg !29
  %.cvtpack.f22218 = extractelement <4 x float> %math.result463, i32 2, !dbg !29
  %.cvtpack.f32219 = extractelement <4 x float> %math.result463, i32 3, !dbg !29
  %.cvtpack.p02220 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02216, float %.cvtpack.f12217), !dbg !29
  %.cvtpack.p12221 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22218, float %.cvtpack.f32219), !dbg !29
  %.cvtpack.v02222 = insertelement <2 x i32> poison, i32 %.cvtpack.p02220, i32 0, !dbg !29
  %.cvtpack.v12223 = insertelement <2 x i32> %.cvtpack.v02222, i32 %.cvtpack.p12221, i32 1, !dbg !29
  %1028 = bitcast <2 x i32> %.cvtpack.v12223 to <4 x half>, !dbg !29
  %math.packed464 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1028) #9, !dbg !28, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %1029 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed464, <4 x half> %989, <4 x float> %986) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1030 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed464, <4 x half> %1009, <4 x float> %987) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !30
  %1031 = extractvalue [2 x <8 x float>] %132, 0, !dbg !30
  %1032 = shufflevector <8 x float> %1031, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %1033 = fmul <4 x float> %1032, %711, !dbg !30
  %1034 = shufflevector <8 x float> %1031, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %1035 = fmul <4 x float> %1034, %711, !dbg !30
  %1036 = extractvalue [2 x <8 x float>] %132, 1, !dbg !30
  %1037 = shufflevector <8 x float> %1036, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %1038 = fmul <4 x float> %1037, %719, !dbg !30
  %1039 = shufflevector <8 x float> %1036, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %1040 = fmul <4 x float> %1039, %719, !dbg !30
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %vdb.address1552 = or disjoint i64 %.idx15051507, 64, !dbg !18
  %vdb.address1553 = add i64 %address.local983, %vdb.address1552, !dbg !18
  %vdb.address1554 = trunc i64 %vdb.address1553 to i32, !dbg !18
  %vdb.address1555 = shl i32 %vdb.address1554, 1, !dbg !18
  %vdb.address1563 = or i32 %.tr1512, 16512, !dbg !18
  %1041 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %49, i32 %vdb.address1555, <4 x i32> %dma.buffer.rsrc1411) #10, !dbg !18
  %1042 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %57, i32 %vdb.address1563, <4 x i32> %dma.buffer.rsrc1411) #10, !dbg !18
  %vdb.read = getelementptr i8, ptr addrspace(3) %address.local999, i64 8192, !dbg !14
  %1043 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #0, !dbg !14, !frisk.v_double_buffer !35
  %1044 = bitcast <4 x i32> %1043 to <8 x half>, !dbg !14
  %vdb.read1545 = getelementptr i8, ptr addrspace(3) %address.local1008, i64 8448, !dbg !14
  %1045 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1545, i32 0) #0, !dbg !14, !frisk.v_double_buffer !35
  %1046 = bitcast <4 x i32> %1045 to <8 x half>, !dbg !14
  %vdb.read1546 = getelementptr i8, ptr addrspace(3) %address.local1008, i64 8704, !dbg !14
  %1047 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1546, i32 0) #0, !dbg !14, !frisk.v_double_buffer !35
  %1048 = bitcast <4 x i32> %1047 to <8 x half>, !dbg !14
  %vdb.read1547 = getelementptr i8, ptr addrspace(3) %address.local1008, i64 8960, !dbg !14
  %1049 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1547, i32 0) #0, !dbg !14, !frisk.v_double_buffer !35
  %1050 = bitcast <4 x i32> %1049 to <8 x half>, !dbg !14
  %vdb.read1548 = getelementptr i8, ptr addrspace(3) %address.local999, i64 12288, !dbg !14
  %1051 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1548, i32 0) #0, !dbg !14, !frisk.v_double_buffer !35
  %1052 = bitcast <4 x i32> %1051 to <8 x half>, !dbg !14
  %vdb.read1549 = getelementptr i8, ptr addrspace(3) %address.local1008, i64 12544, !dbg !14
  %1053 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1549, i32 0) #0, !dbg !14, !frisk.v_double_buffer !35
  %1054 = bitcast <4 x i32> %1053 to <8 x half>, !dbg !14
  %vdb.read1550 = getelementptr i8, ptr addrspace(3) %address.local1008, i64 12800, !dbg !14
  %1055 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1550, i32 0) #0, !dbg !14, !frisk.v_double_buffer !35
  %1056 = bitcast <4 x i32> %1055 to <8 x half>, !dbg !14
  %vdb.read1551 = getelementptr i8, ptr addrspace(3) %address.local1008, i64 13056, !dbg !14
  %1057 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1551, i32 0) #0, !dbg !14, !frisk.v_double_buffer !35
  %1058 = shufflevector <8 x half> %1044, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1059 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1058, <4 x float> %1033) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1060 = shufflevector <8 x half> %1044, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1061 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1060, <4 x float> %1035) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1062 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed450, <4 x half> %1058, <4 x float> %1038) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1063 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed450, <4 x half> %1060, <4 x float> %1040) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1064 = shufflevector <8 x half> %1046, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1065 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed436, <4 x half> %1064, <4 x float> %1059) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1066 = shufflevector <8 x half> %1046, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1067 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed436, <4 x half> %1066, <4 x float> %1061) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1068 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed452, <4 x half> %1064, <4 x float> %1062) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1069 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed452, <4 x half> %1066, <4 x float> %1063) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1070 = shufflevector <8 x half> %1048, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1071 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed438, <4 x half> %1070, <4 x float> %1065) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1072 = shufflevector <8 x half> %1048, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1073 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed438, <4 x half> %1072, <4 x float> %1067) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1074 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed454, <4 x half> %1070, <4 x float> %1068) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1075 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed454, <4 x half> %1072, <4 x float> %1069) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1076 = shufflevector <8 x half> %1050, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1077 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed440, <4 x half> %1076, <4 x float> %1071) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1078 = shufflevector <8 x half> %1050, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1079 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed440, <4 x half> %1078, <4 x float> %1073) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1080 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed456, <4 x half> %1076, <4 x float> %1074) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1081 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed456, <4 x half> %1078, <4 x float> %1075) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1082 = shufflevector <8 x half> %1052, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1083 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed442, <4 x half> %1082, <4 x float> %1077) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1084 = shufflevector <8 x half> %1052, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1085 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed442, <4 x half> %1084, <4 x float> %1079) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1086 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed458, <4 x half> %1082, <4 x float> %1080) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1087 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed458, <4 x half> %1084, <4 x float> %1081) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1088 = shufflevector <8 x half> %1054, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1089 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed444, <4 x half> %1088, <4 x float> %1083) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1090 = shufflevector <8 x half> %1054, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1091 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed444, <4 x half> %1090, <4 x float> %1085) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1092 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed460, <4 x half> %1088, <4 x float> %1086) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1093 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed460, <4 x half> %1090, <4 x float> %1087) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1094 = shufflevector <8 x half> %1056, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1095 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed446, <4 x half> %1094, <4 x float> %1089) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1096 = shufflevector <8 x half> %1056, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1097 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed446, <4 x half> %1096, <4 x float> %1091) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1098 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed462, <4 x half> %1094, <4 x float> %1092) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1099 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed462, <4 x half> %1096, <4 x float> %1093) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1100 = bitcast <4 x i32> %1057 to <8 x half>, !dbg !14
  %1101 = shufflevector <8 x half> %1100, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1102 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed448, <4 x half> %1101, <4 x float> %1095) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1103 = shufflevector <8 x half> %1100, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1104 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed448, <4 x half> %1103, <4 x float> %1097) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1105 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed464, <4 x half> %1101, <4 x float> %1098) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1106 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed464, <4 x half> %1103, <4 x float> %1099) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !30
  %1107 = extractvalue [2 x <8 x float>] %133, 0, !dbg !30
  %1108 = shufflevector <8 x float> %1107, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %1109 = fmul <4 x float> %1108, %711, !dbg !30
  %1110 = shufflevector <8 x float> %1107, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %1111 = fmul <4 x float> %1110, %711, !dbg !30
  %1112 = extractvalue [2 x <8 x float>] %133, 1, !dbg !30
  %1113 = shufflevector <8 x float> %1112, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %1114 = fmul <4 x float> %1113, %719, !dbg !30
  %1115 = shufflevector <8 x float> %1112, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %1116 = fmul <4 x float> %1115, %719, !dbg !30
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %vdb.address1568 = or disjoint i64 %.idx15051507, 96, !dbg !18
  %vdb.address1569 = add i64 %address.local983, %vdb.address1568, !dbg !18
  %vdb.address1570 = trunc i64 %vdb.address1569 to i32, !dbg !18
  %vdb.address1571 = shl i32 %vdb.address1570, 1, !dbg !18
  %vdb.address1579 = or i32 %.tr1512, 16576, !dbg !18
  %1117 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1571, <4 x i32> %dma.buffer.rsrc1411) #10, !dbg !18
  %1118 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1544, i32 %vdb.address1579, <4 x i32> %dma.buffer.rsrc1411) #10, !dbg !18
  %1119 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local999, i32 0) #0, !dbg !14, !frisk.v_double_buffer !35
  %1120 = bitcast <4 x i32> %1119 to <8 x half>, !dbg !14
  %1121 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1009, i32 0) #0, !dbg !14, !frisk.v_double_buffer !35
  %1122 = bitcast <4 x i32> %1121 to <8 x half>, !dbg !14
  %1123 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1013, i32 0) #0, !dbg !14, !frisk.v_double_buffer !35
  %1124 = bitcast <4 x i32> %1123 to <8 x half>, !dbg !14
  %1125 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1017, i32 0) #0, !dbg !14, !frisk.v_double_buffer !35
  %1126 = bitcast <4 x i32> %1125 to <8 x half>, !dbg !14
  %1127 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1019, i32 0) #0, !dbg !14, !frisk.v_double_buffer !35
  %1128 = bitcast <4 x i32> %1127 to <8 x half>, !dbg !14
  %1129 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1023, i32 0) #0, !dbg !14, !frisk.v_double_buffer !35
  %1130 = bitcast <4 x i32> %1129 to <8 x half>, !dbg !14
  %1131 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1027, i32 0) #0, !dbg !14, !frisk.v_double_buffer !35
  %1132 = bitcast <4 x i32> %1131 to <8 x half>, !dbg !14
  %1133 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1031, i32 0) #0, !dbg !14, !frisk.v_double_buffer !35
  %1134 = shufflevector <8 x half> %1120, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1135 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1134, <4 x float> %1109) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1136 = shufflevector <8 x half> %1120, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1137 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1136, <4 x float> %1111) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1138 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed450, <4 x half> %1134, <4 x float> %1114) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1139 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed450, <4 x half> %1136, <4 x float> %1116) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1140 = shufflevector <8 x half> %1122, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1141 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed436, <4 x half> %1140, <4 x float> %1135) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1142 = shufflevector <8 x half> %1122, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1143 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed436, <4 x half> %1142, <4 x float> %1137) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1144 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed452, <4 x half> %1140, <4 x float> %1138) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1145 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed452, <4 x half> %1142, <4 x float> %1139) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1146 = shufflevector <8 x half> %1124, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1147 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed438, <4 x half> %1146, <4 x float> %1141) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1148 = shufflevector <8 x half> %1124, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1149 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed438, <4 x half> %1148, <4 x float> %1143) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1150 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed454, <4 x half> %1146, <4 x float> %1144) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1151 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed454, <4 x half> %1148, <4 x float> %1145) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1152 = shufflevector <8 x half> %1126, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1153 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed440, <4 x half> %1152, <4 x float> %1147) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1154 = shufflevector <8 x half> %1126, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1155 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed440, <4 x half> %1154, <4 x float> %1149) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1156 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed456, <4 x half> %1152, <4 x float> %1150) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1157 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed456, <4 x half> %1154, <4 x float> %1151) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1158 = shufflevector <8 x half> %1128, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1159 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed442, <4 x half> %1158, <4 x float> %1153) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1160 = shufflevector <8 x half> %1128, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1161 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed442, <4 x half> %1160, <4 x float> %1155) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1162 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed458, <4 x half> %1158, <4 x float> %1156) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1163 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed458, <4 x half> %1160, <4 x float> %1157) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1164 = shufflevector <8 x half> %1130, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1165 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed444, <4 x half> %1164, <4 x float> %1159) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1166 = shufflevector <8 x half> %1130, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1167 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed444, <4 x half> %1166, <4 x float> %1161) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1168 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed460, <4 x half> %1164, <4 x float> %1162) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1169 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed460, <4 x half> %1166, <4 x float> %1163) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1170 = shufflevector <8 x half> %1132, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1171 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed446, <4 x half> %1170, <4 x float> %1165) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1172 = shufflevector <8 x half> %1132, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1173 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed446, <4 x half> %1172, <4 x float> %1167) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1174 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed462, <4 x half> %1170, <4 x float> %1168) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1175 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed462, <4 x half> %1172, <4 x float> %1169) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1176 = bitcast <4 x i32> %1133 to <8 x half>, !dbg !14
  %1177 = shufflevector <8 x half> %1176, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1178 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed448, <4 x half> %1177, <4 x float> %1171) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1179 = shufflevector <8 x half> %1176, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1180 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed448, <4 x half> %1179, <4 x float> %1173) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1181 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed464, <4 x half> %1177, <4 x float> %1174) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1182 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed464, <4 x half> %1179, <4 x float> %1175) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !30
  %1183 = extractvalue [2 x <8 x float>] %134, 0, !dbg !30
  %1184 = shufflevector <8 x float> %1183, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %1185 = fmul <4 x float> %1184, %711, !dbg !30
  %1186 = shufflevector <8 x float> %1183, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %1187 = fmul <4 x float> %1186, %711, !dbg !30
  %1188 = extractvalue [2 x <8 x float>] %134, 1, !dbg !30
  %1189 = shufflevector <8 x float> %1188, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %1190 = fmul <4 x float> %1189, %719, !dbg !30
  %1191 = shufflevector <8 x float> %1188, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %1192 = fmul <4 x float> %1191, %719, !dbg !30
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1193 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #0, !dbg !14, !frisk.v_double_buffer !35
  %1194 = bitcast <4 x i32> %1193 to <8 x half>, !dbg !14
  %1195 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1545, i32 0) #0, !dbg !14, !frisk.v_double_buffer !35
  %1196 = bitcast <4 x i32> %1195 to <8 x half>, !dbg !14
  %1197 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1546, i32 0) #0, !dbg !14, !frisk.v_double_buffer !35
  %1198 = bitcast <4 x i32> %1197 to <8 x half>, !dbg !14
  %1199 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1547, i32 0) #0, !dbg !14, !frisk.v_double_buffer !35
  %1200 = bitcast <4 x i32> %1199 to <8 x half>, !dbg !14
  %1201 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1548, i32 0) #0, !dbg !14, !frisk.v_double_buffer !35
  %1202 = bitcast <4 x i32> %1201 to <8 x half>, !dbg !14
  %1203 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1549, i32 0) #0, !dbg !14, !frisk.v_double_buffer !35
  %1204 = bitcast <4 x i32> %1203 to <8 x half>, !dbg !14
  %1205 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1550, i32 0) #0, !dbg !14, !frisk.v_double_buffer !35
  %1206 = bitcast <4 x i32> %1205 to <8 x half>, !dbg !14
  %1207 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1551, i32 0) #0, !dbg !14, !frisk.v_double_buffer !35
  %1208 = shufflevector <8 x half> %1194, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1209 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1208, <4 x float> %1185) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1210 = shufflevector <8 x half> %1194, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1211 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1210, <4 x float> %1187) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1212 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed450, <4 x half> %1208, <4 x float> %1190) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1213 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed450, <4 x half> %1210, <4 x float> %1192) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1214 = shufflevector <8 x half> %1196, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1215 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed436, <4 x half> %1214, <4 x float> %1209) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1216 = shufflevector <8 x half> %1196, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1217 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed436, <4 x half> %1216, <4 x float> %1211) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1218 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed452, <4 x half> %1214, <4 x float> %1212) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1219 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed452, <4 x half> %1216, <4 x float> %1213) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1220 = shufflevector <8 x half> %1198, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1221 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed438, <4 x half> %1220, <4 x float> %1215) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1222 = shufflevector <8 x half> %1198, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1223 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed438, <4 x half> %1222, <4 x float> %1217) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1224 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed454, <4 x half> %1220, <4 x float> %1218) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1225 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed454, <4 x half> %1222, <4 x float> %1219) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1226 = shufflevector <8 x half> %1200, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1227 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed440, <4 x half> %1226, <4 x float> %1221) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1228 = shufflevector <8 x half> %1200, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1229 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed440, <4 x half> %1228, <4 x float> %1223) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1230 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed456, <4 x half> %1226, <4 x float> %1224) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1231 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed456, <4 x half> %1228, <4 x float> %1225) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1232 = shufflevector <8 x half> %1202, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1233 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed442, <4 x half> %1232, <4 x float> %1227) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1234 = shufflevector <8 x half> %1202, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1235 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed442, <4 x half> %1234, <4 x float> %1229) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1236 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed458, <4 x half> %1232, <4 x float> %1230) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1237 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed458, <4 x half> %1234, <4 x float> %1231) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1238 = shufflevector <8 x half> %1204, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1239 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed444, <4 x half> %1238, <4 x float> %1233) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1240 = shufflevector <8 x half> %1204, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1241 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed444, <4 x half> %1240, <4 x float> %1235) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1242 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed460, <4 x half> %1238, <4 x float> %1236) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1243 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed460, <4 x half> %1240, <4 x float> %1237) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1244 = shufflevector <8 x half> %1206, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1245 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed446, <4 x half> %1244, <4 x float> %1239) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1246 = shufflevector <8 x half> %1206, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1247 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed446, <4 x half> %1246, <4 x float> %1241) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1248 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed462, <4 x half> %1244, <4 x float> %1242) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1249 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed462, <4 x half> %1246, <4 x float> %1243) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1250 = bitcast <4 x i32> %1207 to <8 x half>, !dbg !14
  %1251 = shufflevector <8 x half> %1250, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1252 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed448, <4 x half> %1251, <4 x float> %1245) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1253 = shufflevector <8 x half> %1250, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1254 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed448, <4 x half> %1253, <4 x float> %1247) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1255 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed464, <4 x half> %1251, <4 x float> %1248) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1256 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed464, <4 x half> %1253, <4 x float> %1249) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !6
  %1257 = shufflevector <4 x float> %1008, <4 x float> %1010, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !6
  %1258 = insertvalue [2 x <8 x float>] poison, <8 x float> %1257, 0, !dbg !6
  %1259 = shufflevector <4 x float> %1029, <4 x float> %1030, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !6
  %1260 = insertvalue [2 x <8 x float>] %1258, <8 x float> %1259, 1, !dbg !6
  %1261 = insertvalue [2 x <1 x float>] poison, <1 x float> %637, 0, !dbg !6
  %1262 = insertvalue [2 x <1 x float>] %1261, <1 x float> %640, 1, !dbg !6
  %1263 = shufflevector <4 x float> %1102, <4 x float> %1104, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !6
  %1264 = insertvalue [2 x <8 x float>] poison, <8 x float> %1263, 0, !dbg !6
  %1265 = shufflevector <4 x float> %1105, <4 x float> %1106, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !6
  %1266 = insertvalue [2 x <8 x float>] %1264, <8 x float> %1265, 1, !dbg !6
  %1267 = shufflevector <4 x float> %1178, <4 x float> %1180, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !6
  %1268 = insertvalue [2 x <8 x float>] poison, <8 x float> %1267, 0, !dbg !6
  %1269 = shufflevector <4 x float> %1181, <4 x float> %1182, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !6
  %1270 = insertvalue [2 x <8 x float>] %1268, <8 x float> %1269, 1, !dbg !6
  %1271 = shufflevector <4 x float> %1252, <4 x float> %1254, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !6
  %1272 = insertvalue [2 x <8 x float>] poison, <8 x float> %1271, 0, !dbg !6
  %1273 = shufflevector <4 x float> %1255, <4 x float> %1256, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !6
  %1274 = insertvalue [2 x <8 x float>] %1272, <8 x float> %1273, 1, !dbg !6
  fence syncscope("workgroup") release, !dbg !6
  call void @llvm.amdgcn.s.barrier(), !dbg !6
  fence syncscope("workgroup") acquire, !dbg !6
  %1275 = add nuw nsw i64 %128, 1, !dbg !6
  br label %127, !dbg !6

1276:                                             ; preds = %127
  %lane.sum.field233 = extractvalue [2 x <1 x float>] %130, 1, !dbg !6
  %lane.sum.local234 = extractelement <1 x float> %lane.sum.field233, i64 0, !dbg !6
  %1277 = shl nuw nsw i32 %8, 2, !dbg !6
  %1278 = xor i32 %1277, 64, !dbg !6
  %1279 = bitcast float %lane.sum.local234 to i32, !dbg !6
  %lane.sum.shuffle237 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1278, i32 %1279), !dbg !6
  %1280 = bitcast i32 %lane.sum.shuffle237 to float, !dbg !6
  %lane.sum.final238 = fadd float %lane.sum.local234, %1280, !dbg !6
  %1281 = xor i32 %1277, 128, !dbg !6
  %1282 = bitcast float %lane.sum.final238 to i32, !dbg !6
  %lane.sum.shuffle239 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1281, i32 %1282), !dbg !6
  %1283 = bitcast i32 %lane.sum.shuffle239 to float, !dbg !6
  %lane.sum.final240 = fadd float %lane.sum.final238, %1283, !dbg !6
  %1284 = insertelement <1 x float> poison, float %lane.sum.final240, i64 0, !dbg !6
  %lane.sum.field = extractvalue [2 x <1 x float>] %130, 0, !dbg !6
  %lane.sum.local = extractelement <1 x float> %lane.sum.field, i64 0, !dbg !6
  %1285 = bitcast float %lane.sum.local to i32, !dbg !6
  %lane.sum.shuffle = call i32 @llvm.amdgcn.ds.bpermute(i32 %1278, i32 %1285), !dbg !6
  %1286 = bitcast i32 %lane.sum.shuffle to float, !dbg !6
  %lane.sum.final = fadd float %lane.sum.local, %1286, !dbg !6
  %1287 = bitcast float %lane.sum.final to i32, !dbg !6
  %lane.sum.shuffle231 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1281, i32 %1287), !dbg !6
  %1288 = bitcast i32 %lane.sum.shuffle231 to float, !dbg !6
  %lane.sum.final232 = fadd float %lane.sum.final, %1288, !dbg !6
  %1289 = insertelement <1 x float> poison, float %lane.sum.final232, i64 0, !dbg !6
  %1290 = and i32 %kdma.tid, 16, !dbg !6
  %.not = icmp eq i32 %1290, 0, !dbg !6
  %1291 = and i32 %kdma.tid, 32, !dbg !6
  %.not40 = icmp eq i32 %1291, 0, !dbg !6
  %1292 = extractvalue [2 x <8 x float>] %134, 0, !dbg !6
  %1293 = extractelement <8 x float> %1292, i64 0, !dbg !6
  %1294 = extractelement <8 x float> %1292, i64 1, !dbg !6
  %1295 = extractelement <8 x float> %1292, i64 2, !dbg !6
  %1296 = extractelement <8 x float> %1292, i64 3, !dbg !6
  %1297 = select i1 %.not, float %1294, float %1293, !dbg !6
  %1298 = bitcast float %1297 to i32, !dbg !6
  %1299 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1298), !dbg !6
  %1300 = bitcast i32 %1299 to float, !dbg !6
  %1301 = select i1 %.not, float %1293, float %1300, !dbg !6
  %1302 = select i1 %.not, float %1300, float %1294, !dbg !6
  %1303 = select i1 %.not, float %1296, float %1295, !dbg !6
  %1304 = bitcast float %1303 to i32, !dbg !6
  %1305 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1304), !dbg !6
  %1306 = bitcast i32 %1305 to float, !dbg !6
  %1307 = select i1 %.not, float %1295, float %1306, !dbg !6
  %1308 = select i1 %.not, float %1306, float %1296, !dbg !6
  %1309 = select i1 %.not40, float %1307, float %1301, !dbg !6
  %1310 = bitcast float %1309 to i32, !dbg !6
  %1311 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1310), !dbg !6
  %1312 = bitcast i32 %1311 to float, !dbg !6
  %1313 = select i1 %.not40, float %1301, float %1312, !dbg !6
  %1314 = select i1 %.not40, float %1312, float %1307, !dbg !6
  %1315 = select i1 %.not40, float %1308, float %1302, !dbg !6
  %1316 = bitcast float %1315 to i32, !dbg !6
  %1317 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1316), !dbg !6
  %1318 = bitcast i32 %1317 to float, !dbg !6
  %1319 = select i1 %.not40, float %1302, float %1318, !dbg !6
  %1320 = select i1 %.not40, float %1318, float %1308, !dbg !6
  %1321 = extractelement <8 x float> %1292, i64 4, !dbg !6
  %1322 = extractelement <8 x float> %1292, i64 5, !dbg !6
  %1323 = extractelement <8 x float> %1292, i64 6, !dbg !6
  %1324 = extractelement <8 x float> %1292, i64 7, !dbg !6
  %1325 = select i1 %.not, float %1322, float %1321, !dbg !6
  %1326 = bitcast float %1325 to i32, !dbg !6
  %1327 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1326), !dbg !6
  %1328 = bitcast i32 %1327 to float, !dbg !6
  %1329 = select i1 %.not, float %1321, float %1328, !dbg !6
  %1330 = select i1 %.not, float %1328, float %1322, !dbg !6
  %1331 = select i1 %.not, float %1324, float %1323, !dbg !6
  %1332 = bitcast float %1331 to i32, !dbg !6
  %1333 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1332), !dbg !6
  %1334 = bitcast i32 %1333 to float, !dbg !6
  %1335 = select i1 %.not, float %1323, float %1334, !dbg !6
  %1336 = select i1 %.not, float %1334, float %1324, !dbg !6
  %1337 = select i1 %.not40, float %1335, float %1329, !dbg !6
  %1338 = bitcast float %1337 to i32, !dbg !6
  %1339 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1338), !dbg !6
  %1340 = bitcast i32 %1339 to float, !dbg !6
  %1341 = select i1 %.not40, float %1329, float %1340, !dbg !6
  %1342 = select i1 %.not40, float %1340, float %1335, !dbg !6
  %1343 = select i1 %.not40, float %1336, float %1330, !dbg !6
  %1344 = bitcast float %1343 to i32, !dbg !6
  %1345 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1344), !dbg !6
  %1346 = bitcast i32 %1345 to float, !dbg !6
  %1347 = select i1 %.not40, float %1330, float %1346, !dbg !6
  %1348 = select i1 %.not40, float %1346, float %1336, !dbg !6
  %1349 = extractvalue [2 x <8 x float>] %134, 1, !dbg !6
  %1350 = extractelement <8 x float> %1349, i64 0, !dbg !6
  %1351 = extractelement <8 x float> %1349, i64 1, !dbg !6
  %1352 = extractelement <8 x float> %1349, i64 2, !dbg !6
  %1353 = extractelement <8 x float> %1349, i64 3, !dbg !6
  %1354 = select i1 %.not, float %1351, float %1350, !dbg !6
  %1355 = bitcast float %1354 to i32, !dbg !6
  %1356 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1355), !dbg !6
  %1357 = bitcast i32 %1356 to float, !dbg !6
  %1358 = select i1 %.not, float %1350, float %1357, !dbg !6
  %1359 = select i1 %.not, float %1357, float %1351, !dbg !6
  %1360 = select i1 %.not, float %1353, float %1352, !dbg !6
  %1361 = bitcast float %1360 to i32, !dbg !6
  %1362 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1361), !dbg !6
  %1363 = bitcast i32 %1362 to float, !dbg !6
  %1364 = select i1 %.not, float %1352, float %1363, !dbg !6
  %1365 = select i1 %.not, float %1363, float %1353, !dbg !6
  %1366 = select i1 %.not40, float %1364, float %1358, !dbg !6
  %1367 = bitcast float %1366 to i32, !dbg !6
  %1368 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1367), !dbg !6
  %1369 = bitcast i32 %1368 to float, !dbg !6
  %1370 = select i1 %.not40, float %1358, float %1369, !dbg !6
  %1371 = select i1 %.not40, float %1369, float %1364, !dbg !6
  %1372 = select i1 %.not40, float %1365, float %1359, !dbg !6
  %1373 = bitcast float %1372 to i32, !dbg !6
  %1374 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1373), !dbg !6
  %1375 = bitcast i32 %1374 to float, !dbg !6
  %1376 = select i1 %.not40, float %1359, float %1375, !dbg !6
  %1377 = select i1 %.not40, float %1375, float %1365, !dbg !6
  %1378 = extractelement <8 x float> %1349, i64 4, !dbg !6
  %1379 = extractelement <8 x float> %1349, i64 5, !dbg !6
  %1380 = extractelement <8 x float> %1349, i64 6, !dbg !6
  %1381 = extractelement <8 x float> %1349, i64 7, !dbg !6
  %1382 = select i1 %.not, float %1379, float %1378, !dbg !6
  %1383 = bitcast float %1382 to i32, !dbg !6
  %1384 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1383), !dbg !6
  %1385 = bitcast i32 %1384 to float, !dbg !6
  %1386 = select i1 %.not, float %1378, float %1385, !dbg !6
  %1387 = select i1 %.not, float %1385, float %1379, !dbg !6
  %1388 = select i1 %.not, float %1381, float %1380, !dbg !6
  %1389 = bitcast float %1388 to i32, !dbg !6
  %1390 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1389), !dbg !6
  %1391 = bitcast i32 %1390 to float, !dbg !6
  %1392 = select i1 %.not, float %1380, float %1391, !dbg !6
  %1393 = select i1 %.not, float %1391, float %1381, !dbg !6
  %1394 = select i1 %.not40, float %1392, float %1386, !dbg !6
  %1395 = bitcast float %1394 to i32, !dbg !6
  %1396 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1395), !dbg !6
  %1397 = bitcast i32 %1396 to float, !dbg !6
  %1398 = select i1 %.not40, float %1386, float %1397, !dbg !6
  %1399 = select i1 %.not40, float %1397, float %1392, !dbg !6
  %1400 = select i1 %.not40, float %1393, float %1387, !dbg !6
  %1401 = bitcast float %1400 to i32, !dbg !6
  %1402 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1401), !dbg !6
  %1403 = bitcast i32 %1402 to float, !dbg !6
  %1404 = select i1 %.not40, float %1387, float %1403, !dbg !6
  %1405 = select i1 %.not40, float %1403, float %1393, !dbg !6
  %1406 = extractvalue [2 x <8 x float>] %133, 0, !dbg !6
  %1407 = extractelement <8 x float> %1406, i64 0, !dbg !6
  %1408 = extractelement <8 x float> %1406, i64 1, !dbg !6
  %1409 = extractelement <8 x float> %1406, i64 2, !dbg !6
  %1410 = extractelement <8 x float> %1406, i64 3, !dbg !6
  %1411 = select i1 %.not, float %1408, float %1407, !dbg !6
  %1412 = bitcast float %1411 to i32, !dbg !6
  %1413 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1412), !dbg !6
  %1414 = bitcast i32 %1413 to float, !dbg !6
  %1415 = select i1 %.not, float %1407, float %1414, !dbg !6
  %1416 = select i1 %.not, float %1414, float %1408, !dbg !6
  %1417 = select i1 %.not, float %1410, float %1409, !dbg !6
  %1418 = bitcast float %1417 to i32, !dbg !6
  %1419 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1418), !dbg !6
  %1420 = bitcast i32 %1419 to float, !dbg !6
  %1421 = select i1 %.not, float %1409, float %1420, !dbg !6
  %1422 = select i1 %.not, float %1420, float %1410, !dbg !6
  %1423 = select i1 %.not40, float %1421, float %1415, !dbg !6
  %1424 = bitcast float %1423 to i32, !dbg !6
  %1425 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1424), !dbg !6
  %1426 = bitcast i32 %1425 to float, !dbg !6
  %1427 = select i1 %.not40, float %1415, float %1426, !dbg !6
  %1428 = select i1 %.not40, float %1426, float %1421, !dbg !6
  %1429 = select i1 %.not40, float %1422, float %1416, !dbg !6
  %1430 = bitcast float %1429 to i32, !dbg !6
  %1431 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1430), !dbg !6
  %1432 = bitcast i32 %1431 to float, !dbg !6
  %1433 = select i1 %.not40, float %1416, float %1432, !dbg !6
  %1434 = select i1 %.not40, float %1432, float %1422, !dbg !6
  %1435 = extractelement <8 x float> %1406, i64 4, !dbg !6
  %1436 = extractelement <8 x float> %1406, i64 5, !dbg !6
  %1437 = extractelement <8 x float> %1406, i64 6, !dbg !6
  %1438 = extractelement <8 x float> %1406, i64 7, !dbg !6
  %1439 = select i1 %.not, float %1436, float %1435, !dbg !6
  %1440 = bitcast float %1439 to i32, !dbg !6
  %1441 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1440), !dbg !6
  %1442 = bitcast i32 %1441 to float, !dbg !6
  %1443 = select i1 %.not, float %1435, float %1442, !dbg !6
  %1444 = select i1 %.not, float %1442, float %1436, !dbg !6
  %1445 = select i1 %.not, float %1438, float %1437, !dbg !6
  %1446 = bitcast float %1445 to i32, !dbg !6
  %1447 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1446), !dbg !6
  %1448 = bitcast i32 %1447 to float, !dbg !6
  %1449 = select i1 %.not, float %1437, float %1448, !dbg !6
  %1450 = select i1 %.not, float %1448, float %1438, !dbg !6
  %1451 = select i1 %.not40, float %1449, float %1443, !dbg !6
  %1452 = bitcast float %1451 to i32, !dbg !6
  %1453 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1452), !dbg !6
  %1454 = bitcast i32 %1453 to float, !dbg !6
  %1455 = select i1 %.not40, float %1443, float %1454, !dbg !6
  %1456 = select i1 %.not40, float %1454, float %1449, !dbg !6
  %1457 = select i1 %.not40, float %1450, float %1444, !dbg !6
  %1458 = bitcast float %1457 to i32, !dbg !6
  %1459 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1458), !dbg !6
  %1460 = bitcast i32 %1459 to float, !dbg !6
  %1461 = select i1 %.not40, float %1444, float %1460, !dbg !6
  %1462 = select i1 %.not40, float %1460, float %1450, !dbg !6
  %1463 = extractvalue [2 x <8 x float>] %133, 1, !dbg !6
  %1464 = extractelement <8 x float> %1463, i64 0, !dbg !6
  %1465 = extractelement <8 x float> %1463, i64 1, !dbg !6
  %1466 = extractelement <8 x float> %1463, i64 2, !dbg !6
  %1467 = extractelement <8 x float> %1463, i64 3, !dbg !6
  %1468 = select i1 %.not, float %1465, float %1464, !dbg !6
  %1469 = bitcast float %1468 to i32, !dbg !6
  %1470 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1469), !dbg !6
  %1471 = bitcast i32 %1470 to float, !dbg !6
  %1472 = select i1 %.not, float %1464, float %1471, !dbg !6
  %1473 = select i1 %.not, float %1471, float %1465, !dbg !6
  %1474 = select i1 %.not, float %1467, float %1466, !dbg !6
  %1475 = bitcast float %1474 to i32, !dbg !6
  %1476 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1475), !dbg !6
  %1477 = bitcast i32 %1476 to float, !dbg !6
  %1478 = select i1 %.not, float %1466, float %1477, !dbg !6
  %1479 = select i1 %.not, float %1477, float %1467, !dbg !6
  %1480 = select i1 %.not40, float %1478, float %1472, !dbg !6
  %1481 = bitcast float %1480 to i32, !dbg !6
  %1482 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1481), !dbg !6
  %1483 = bitcast i32 %1482 to float, !dbg !6
  %1484 = select i1 %.not40, float %1472, float %1483, !dbg !6
  %1485 = select i1 %.not40, float %1483, float %1478, !dbg !6
  %1486 = select i1 %.not40, float %1479, float %1473, !dbg !6
  %1487 = bitcast float %1486 to i32, !dbg !6
  %1488 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1487), !dbg !6
  %1489 = bitcast i32 %1488 to float, !dbg !6
  %1490 = select i1 %.not40, float %1473, float %1489, !dbg !6
  %1491 = select i1 %.not40, float %1489, float %1479, !dbg !6
  %1492 = extractelement <8 x float> %1463, i64 4, !dbg !6
  %1493 = extractelement <8 x float> %1463, i64 5, !dbg !6
  %1494 = extractelement <8 x float> %1463, i64 6, !dbg !6
  %1495 = extractelement <8 x float> %1463, i64 7, !dbg !6
  %1496 = select i1 %.not, float %1493, float %1492, !dbg !6
  %1497 = bitcast float %1496 to i32, !dbg !6
  %1498 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1497), !dbg !6
  %1499 = bitcast i32 %1498 to float, !dbg !6
  %1500 = select i1 %.not, float %1492, float %1499, !dbg !6
  %1501 = select i1 %.not, float %1499, float %1493, !dbg !6
  %1502 = select i1 %.not, float %1495, float %1494, !dbg !6
  %1503 = bitcast float %1502 to i32, !dbg !6
  %1504 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1503), !dbg !6
  %1505 = bitcast i32 %1504 to float, !dbg !6
  %1506 = select i1 %.not, float %1494, float %1505, !dbg !6
  %1507 = select i1 %.not, float %1505, float %1495, !dbg !6
  %1508 = select i1 %.not40, float %1506, float %1500, !dbg !6
  %1509 = bitcast float %1508 to i32, !dbg !6
  %1510 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1509), !dbg !6
  %1511 = bitcast i32 %1510 to float, !dbg !6
  %1512 = select i1 %.not40, float %1500, float %1511, !dbg !6
  %1513 = select i1 %.not40, float %1511, float %1506, !dbg !6
  %1514 = select i1 %.not40, float %1507, float %1501, !dbg !6
  %1515 = bitcast float %1514 to i32, !dbg !6
  %1516 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1515), !dbg !6
  %1517 = bitcast i32 %1516 to float, !dbg !6
  %1518 = select i1 %.not40, float %1501, float %1517, !dbg !6
  %1519 = select i1 %.not40, float %1517, float %1507, !dbg !6
  %1520 = extractvalue [2 x <8 x float>] %132, 0, !dbg !6
  %1521 = extractelement <8 x float> %1520, i64 0, !dbg !6
  %1522 = extractelement <8 x float> %1520, i64 1, !dbg !6
  %1523 = extractelement <8 x float> %1520, i64 2, !dbg !6
  %1524 = extractelement <8 x float> %1520, i64 3, !dbg !6
  %1525 = select i1 %.not, float %1522, float %1521, !dbg !6
  %1526 = bitcast float %1525 to i32, !dbg !6
  %1527 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1526), !dbg !6
  %1528 = bitcast i32 %1527 to float, !dbg !6
  %1529 = select i1 %.not, float %1521, float %1528, !dbg !6
  %1530 = select i1 %.not, float %1528, float %1522, !dbg !6
  %1531 = select i1 %.not, float %1524, float %1523, !dbg !6
  %1532 = bitcast float %1531 to i32, !dbg !6
  %1533 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1532), !dbg !6
  %1534 = bitcast i32 %1533 to float, !dbg !6
  %1535 = select i1 %.not, float %1523, float %1534, !dbg !6
  %1536 = select i1 %.not, float %1534, float %1524, !dbg !6
  %1537 = select i1 %.not40, float %1535, float %1529, !dbg !6
  %1538 = bitcast float %1537 to i32, !dbg !6
  %1539 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1538), !dbg !6
  %1540 = bitcast i32 %1539 to float, !dbg !6
  %1541 = select i1 %.not40, float %1529, float %1540, !dbg !6
  %1542 = select i1 %.not40, float %1540, float %1535, !dbg !6
  %1543 = select i1 %.not40, float %1536, float %1530, !dbg !6
  %1544 = bitcast float %1543 to i32, !dbg !6
  %1545 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1544), !dbg !6
  %1546 = bitcast i32 %1545 to float, !dbg !6
  %1547 = select i1 %.not40, float %1530, float %1546, !dbg !6
  %1548 = select i1 %.not40, float %1546, float %1536, !dbg !6
  %1549 = extractelement <8 x float> %1520, i64 4, !dbg !6
  %1550 = extractelement <8 x float> %1520, i64 5, !dbg !6
  %1551 = extractelement <8 x float> %1520, i64 6, !dbg !6
  %1552 = extractelement <8 x float> %1520, i64 7, !dbg !6
  %1553 = select i1 %.not, float %1550, float %1549, !dbg !6
  %1554 = bitcast float %1553 to i32, !dbg !6
  %1555 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1554), !dbg !6
  %1556 = bitcast i32 %1555 to float, !dbg !6
  %1557 = select i1 %.not, float %1549, float %1556, !dbg !6
  %1558 = select i1 %.not, float %1556, float %1550, !dbg !6
  %1559 = select i1 %.not, float %1552, float %1551, !dbg !6
  %1560 = bitcast float %1559 to i32, !dbg !6
  %1561 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1560), !dbg !6
  %1562 = bitcast i32 %1561 to float, !dbg !6
  %1563 = select i1 %.not, float %1551, float %1562, !dbg !6
  %1564 = select i1 %.not, float %1562, float %1552, !dbg !6
  %1565 = select i1 %.not40, float %1563, float %1557, !dbg !6
  %1566 = bitcast float %1565 to i32, !dbg !6
  %1567 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1566), !dbg !6
  %1568 = bitcast i32 %1567 to float, !dbg !6
  %1569 = select i1 %.not40, float %1557, float %1568, !dbg !6
  %1570 = select i1 %.not40, float %1568, float %1563, !dbg !6
  %1571 = select i1 %.not40, float %1564, float %1558, !dbg !6
  %1572 = bitcast float %1571 to i32, !dbg !6
  %1573 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1572), !dbg !6
  %1574 = bitcast i32 %1573 to float, !dbg !6
  %1575 = select i1 %.not40, float %1558, float %1574, !dbg !6
  %1576 = select i1 %.not40, float %1574, float %1564, !dbg !6
  %1577 = extractvalue [2 x <8 x float>] %132, 1, !dbg !6
  %1578 = extractelement <8 x float> %1577, i64 0, !dbg !6
  %1579 = extractelement <8 x float> %1577, i64 1, !dbg !6
  %1580 = extractelement <8 x float> %1577, i64 2, !dbg !6
  %1581 = extractelement <8 x float> %1577, i64 3, !dbg !6
  %1582 = select i1 %.not, float %1579, float %1578, !dbg !6
  %1583 = bitcast float %1582 to i32, !dbg !6
  %1584 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1583), !dbg !6
  %1585 = bitcast i32 %1584 to float, !dbg !6
  %1586 = select i1 %.not, float %1578, float %1585, !dbg !6
  %1587 = select i1 %.not, float %1585, float %1579, !dbg !6
  %1588 = select i1 %.not, float %1581, float %1580, !dbg !6
  %1589 = bitcast float %1588 to i32, !dbg !6
  %1590 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1589), !dbg !6
  %1591 = bitcast i32 %1590 to float, !dbg !6
  %1592 = select i1 %.not, float %1580, float %1591, !dbg !6
  %1593 = select i1 %.not, float %1591, float %1581, !dbg !6
  %1594 = select i1 %.not40, float %1592, float %1586, !dbg !6
  %1595 = bitcast float %1594 to i32, !dbg !6
  %1596 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1595), !dbg !6
  %1597 = bitcast i32 %1596 to float, !dbg !6
  %1598 = select i1 %.not40, float %1586, float %1597, !dbg !6
  %1599 = select i1 %.not40, float %1597, float %1592, !dbg !6
  %1600 = select i1 %.not40, float %1593, float %1587, !dbg !6
  %1601 = bitcast float %1600 to i32, !dbg !6
  %1602 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1601), !dbg !6
  %1603 = bitcast i32 %1602 to float, !dbg !6
  %1604 = select i1 %.not40, float %1587, float %1603, !dbg !6
  %1605 = select i1 %.not40, float %1603, float %1593, !dbg !6
  %1606 = extractelement <8 x float> %1577, i64 4, !dbg !6
  %1607 = extractelement <8 x float> %1577, i64 5, !dbg !6
  %1608 = extractelement <8 x float> %1577, i64 6, !dbg !6
  %1609 = extractelement <8 x float> %1577, i64 7, !dbg !6
  %1610 = select i1 %.not, float %1607, float %1606, !dbg !6
  %1611 = bitcast float %1610 to i32, !dbg !6
  %1612 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1611), !dbg !6
  %1613 = bitcast i32 %1612 to float, !dbg !6
  %1614 = select i1 %.not, float %1606, float %1613, !dbg !6
  %1615 = select i1 %.not, float %1613, float %1607, !dbg !6
  %1616 = select i1 %.not, float %1609, float %1608, !dbg !6
  %1617 = bitcast float %1616 to i32, !dbg !6
  %1618 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1617), !dbg !6
  %1619 = bitcast i32 %1618 to float, !dbg !6
  %1620 = select i1 %.not, float %1608, float %1619, !dbg !6
  %1621 = select i1 %.not, float %1619, float %1609, !dbg !6
  %1622 = select i1 %.not40, float %1620, float %1614, !dbg !6
  %1623 = bitcast float %1622 to i32, !dbg !6
  %1624 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1623), !dbg !6
  %1625 = bitcast i32 %1624 to float, !dbg !6
  %1626 = select i1 %.not40, float %1614, float %1625, !dbg !6
  %1627 = select i1 %.not40, float %1625, float %1620, !dbg !6
  %1628 = select i1 %.not40, float %1621, float %1615, !dbg !6
  %1629 = bitcast float %1628 to i32, !dbg !6
  %1630 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1629), !dbg !6
  %1631 = bitcast i32 %1630 to float, !dbg !6
  %1632 = select i1 %.not40, float %1615, float %1631, !dbg !6
  %1633 = select i1 %.not40, float %1631, float %1621, !dbg !6
  %1634 = extractvalue [2 x <8 x float>] %129, 0, !dbg !6
  %1635 = extractelement <8 x float> %1634, i64 0, !dbg !6
  %1636 = extractelement <8 x float> %1634, i64 1, !dbg !6
  %1637 = extractelement <8 x float> %1634, i64 2, !dbg !6
  %1638 = extractelement <8 x float> %1634, i64 3, !dbg !6
  %1639 = select i1 %.not, float %1636, float %1635, !dbg !6
  %1640 = bitcast float %1639 to i32, !dbg !6
  %1641 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1640), !dbg !6
  %1642 = bitcast i32 %1641 to float, !dbg !6
  %1643 = select i1 %.not, float %1635, float %1642, !dbg !6
  %1644 = select i1 %.not, float %1642, float %1636, !dbg !6
  %1645 = select i1 %.not, float %1638, float %1637, !dbg !6
  %1646 = bitcast float %1645 to i32, !dbg !6
  %1647 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1646), !dbg !6
  %1648 = bitcast i32 %1647 to float, !dbg !6
  %1649 = select i1 %.not, float %1637, float %1648, !dbg !6
  %1650 = select i1 %.not, float %1648, float %1638, !dbg !6
  %1651 = select i1 %.not40, float %1649, float %1643, !dbg !6
  %1652 = bitcast float %1651 to i32, !dbg !6
  %1653 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1652), !dbg !6
  %1654 = select i1 %.not40, float %1650, float %1644, !dbg !6
  %1655 = bitcast float %1654 to i32, !dbg !6
  %1656 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1655), !dbg !6
  %1657 = extractelement <8 x float> %1634, i64 4, !dbg !6
  %1658 = extractelement <8 x float> %1634, i64 5, !dbg !6
  %1659 = extractelement <8 x float> %1634, i64 6, !dbg !6
  %1660 = extractelement <8 x float> %1634, i64 7, !dbg !6
  %1661 = select i1 %.not, float %1658, float %1657, !dbg !6
  %1662 = bitcast float %1661 to i32, !dbg !6
  %1663 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1662), !dbg !6
  %1664 = bitcast i32 %1663 to float, !dbg !6
  %1665 = select i1 %.not, float %1657, float %1664, !dbg !6
  %1666 = select i1 %.not, float %1664, float %1658, !dbg !6
  %1667 = select i1 %.not, float %1660, float %1659, !dbg !6
  %1668 = bitcast float %1667 to i32, !dbg !6
  %1669 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1668), !dbg !6
  %1670 = bitcast i32 %1669 to float, !dbg !6
  %1671 = select i1 %.not, float %1659, float %1670, !dbg !6
  %1672 = select i1 %.not, float %1670, float %1660, !dbg !6
  %1673 = select i1 %.not40, float %1671, float %1665, !dbg !6
  %1674 = bitcast float %1673 to i32, !dbg !6
  %1675 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1674), !dbg !6
  %1676 = select i1 %.not40, float %1672, float %1666, !dbg !6
  %1677 = bitcast float %1676 to i32, !dbg !6
  %1678 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1677), !dbg !6
  %1679 = extractvalue [2 x <8 x float>] %129, 1, !dbg !6
  %1680 = extractelement <8 x float> %1679, i64 0, !dbg !6
  %1681 = extractelement <8 x float> %1679, i64 1, !dbg !6
  %1682 = extractelement <8 x float> %1679, i64 2, !dbg !6
  %1683 = extractelement <8 x float> %1679, i64 3, !dbg !6
  %1684 = select i1 %.not, float %1681, float %1680, !dbg !6
  %1685 = bitcast float %1684 to i32, !dbg !6
  %1686 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1685), !dbg !6
  %1687 = bitcast i32 %1686 to float, !dbg !6
  %1688 = select i1 %.not, float %1680, float %1687, !dbg !6
  %1689 = select i1 %.not, float %1687, float %1681, !dbg !6
  %1690 = select i1 %.not, float %1683, float %1682, !dbg !6
  %1691 = bitcast float %1690 to i32, !dbg !6
  %1692 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1691), !dbg !6
  %1693 = bitcast i32 %1692 to float, !dbg !6
  %1694 = select i1 %.not, float %1682, float %1693, !dbg !6
  %1695 = select i1 %.not, float %1693, float %1683, !dbg !6
  %1696 = select i1 %.not40, float %1694, float %1688, !dbg !6
  %1697 = bitcast float %1696 to i32, !dbg !6
  %1698 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1697), !dbg !6
  %1699 = select i1 %.not40, float %1695, float %1689, !dbg !6
  %1700 = bitcast float %1699 to i32, !dbg !6
  %1701 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1700), !dbg !6
  %1702 = extractelement <8 x float> %1679, i64 4, !dbg !6
  %1703 = extractelement <8 x float> %1679, i64 5, !dbg !6
  %1704 = extractelement <8 x float> %1679, i64 6, !dbg !6
  %1705 = extractelement <8 x float> %1679, i64 7, !dbg !6
  %1706 = select i1 %.not, float %1703, float %1702, !dbg !6
  %1707 = bitcast float %1706 to i32, !dbg !6
  %1708 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1707), !dbg !6
  %1709 = bitcast i32 %1708 to float, !dbg !6
  %1710 = select i1 %.not, float %1702, float %1709, !dbg !6
  %1711 = select i1 %.not, float %1709, float %1703, !dbg !6
  %1712 = select i1 %.not, float %1705, float %1704, !dbg !6
  %1713 = bitcast float %1712 to i32, !dbg !6
  %1714 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1713), !dbg !6
  %1715 = bitcast i32 %1714 to float, !dbg !6
  %1716 = select i1 %.not, float %1704, float %1715, !dbg !6
  %1717 = select i1 %.not, float %1715, float %1705, !dbg !6
  %1718 = select i1 %.not40, float %1716, float %1710, !dbg !6
  %1719 = bitcast float %1718 to i32, !dbg !6
  %1720 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1719), !dbg !6
  %1721 = select i1 %.not40, float %1717, float %1711, !dbg !6
  %1722 = bitcast float %1721 to i32, !dbg !6
  %1723 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1722), !dbg !6
  %1724 = shufflevector <1 x float> %1289, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !20
  %1725 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1724, !dbg !20
  %1726 = shufflevector <1 x float> %1284, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !20
  %1727 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1726, !dbg !20
  %1728 = bitcast i32 %1720 to float, !dbg !6
  %1729 = select i1 %.not40, float %1710, float %1728, !dbg !6
  %1730 = insertelement <4 x float> poison, float %1729, i64 0, !dbg !20
  %1731 = bitcast i32 %1723 to float, !dbg !6
  %1732 = select i1 %.not40, float %1711, float %1731, !dbg !6
  %1733 = insertelement <4 x float> %1730, float %1732, i64 1, !dbg !20
  %1734 = select i1 %.not40, float %1728, float %1716, !dbg !6
  %1735 = insertelement <4 x float> %1733, float %1734, i64 2, !dbg !20
  %1736 = select i1 %.not40, float %1731, float %1717, !dbg !6
  %1737 = insertelement <4 x float> %1735, float %1736, i64 3, !dbg !20
  %1738 = fmul <4 x float> %1737, %1727, !dbg !20
  %1739 = fptrunc <4 x float> %1738 to <4 x half>, !dbg !21
  %1740 = extractelement <4 x half> %1739, i64 3, !dbg !21
  %1741 = extractelement <4 x half> %1739, i64 2, !dbg !21
  %1742 = extractelement <4 x half> %1739, i64 1, !dbg !21
  %1743 = extractelement <4 x half> %1739, i64 0, !dbg !21
  %1744 = bitcast i32 %1698 to float, !dbg !6
  %1745 = select i1 %.not40, float %1688, float %1744, !dbg !6
  %1746 = insertelement <4 x float> poison, float %1745, i64 0, !dbg !20
  %1747 = bitcast i32 %1701 to float, !dbg !6
  %1748 = select i1 %.not40, float %1689, float %1747, !dbg !6
  %1749 = insertelement <4 x float> %1746, float %1748, i64 1, !dbg !20
  %1750 = select i1 %.not40, float %1744, float %1694, !dbg !6
  %1751 = insertelement <4 x float> %1749, float %1750, i64 2, !dbg !20
  %1752 = select i1 %.not40, float %1747, float %1695, !dbg !6
  %1753 = insertelement <4 x float> %1751, float %1752, i64 3, !dbg !20
  %1754 = fmul <4 x float> %1753, %1727, !dbg !20
  %1755 = fptrunc <4 x float> %1754 to <4 x half>, !dbg !21
  %1756 = extractelement <4 x half> %1755, i64 3, !dbg !21
  %1757 = extractelement <4 x half> %1755, i64 2, !dbg !21
  %1758 = extractelement <4 x half> %1755, i64 1, !dbg !21
  %1759 = extractelement <4 x half> %1755, i64 0, !dbg !21
  %1760 = bitcast i32 %1675 to float, !dbg !6
  %1761 = select i1 %.not40, float %1665, float %1760, !dbg !6
  %1762 = insertelement <4 x float> poison, float %1761, i64 0, !dbg !20
  %1763 = bitcast i32 %1678 to float, !dbg !6
  %1764 = select i1 %.not40, float %1666, float %1763, !dbg !6
  %1765 = insertelement <4 x float> %1762, float %1764, i64 1, !dbg !20
  %1766 = select i1 %.not40, float %1760, float %1671, !dbg !6
  %1767 = insertelement <4 x float> %1765, float %1766, i64 2, !dbg !20
  %1768 = select i1 %.not40, float %1763, float %1672, !dbg !6
  %1769 = insertelement <4 x float> %1767, float %1768, i64 3, !dbg !20
  %1770 = fmul <4 x float> %1769, %1725, !dbg !20
  %1771 = fptrunc <4 x float> %1770 to <4 x half>, !dbg !21
  %1772 = extractelement <4 x half> %1771, i64 3, !dbg !21
  %1773 = extractelement <4 x half> %1771, i64 2, !dbg !21
  %1774 = extractelement <4 x half> %1771, i64 1, !dbg !21
  %1775 = extractelement <4 x half> %1771, i64 0, !dbg !21
  %1776 = bitcast i32 %1653 to float, !dbg !6
  %1777 = select i1 %.not40, float %1643, float %1776, !dbg !6
  %1778 = insertelement <4 x float> poison, float %1777, i64 0, !dbg !20
  %1779 = bitcast i32 %1656 to float, !dbg !6
  %1780 = select i1 %.not40, float %1644, float %1779, !dbg !6
  %1781 = insertelement <4 x float> %1778, float %1780, i64 1, !dbg !20
  %1782 = select i1 %.not40, float %1776, float %1649, !dbg !6
  %1783 = insertelement <4 x float> %1781, float %1782, i64 2, !dbg !20
  %1784 = select i1 %.not40, float %1779, float %1650, !dbg !6
  %1785 = insertelement <4 x float> %1783, float %1784, i64 3, !dbg !20
  %1786 = fmul <4 x float> %1785, %1725, !dbg !20
  %1787 = fptrunc <4 x float> %1786 to <4 x half>, !dbg !21
  %1788 = extractelement <4 x half> %1787, i64 3, !dbg !21
  %1789 = extractelement <4 x half> %1787, i64 2, !dbg !21
  %1790 = extractelement <4 x half> %1787, i64 1, !dbg !21
  %1791 = extractelement <4 x half> %1787, i64 0, !dbg !21
  %address.thread1038 = call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #9, !dbg !22, !range !33, !frisk.register_lifetimes !1
  %address.local1039 = zext nneg i32 %address.thread1038 to i64, !dbg !9
  %address.local1040 = add i64 %42, %address.local1039, !dbg !22
  %address.local1041 = lshr i64 %address.local1039, 2, !dbg !19
  %address.local1042 = and i64 %address.local1041, 48, !dbg !19
  %address.local1043 = add i64 %address.local1040, %address.local1042, !dbg !22
  %address.local1044 = and i64 %address.local1039, 240, !dbg !22
  %address.local1046 = sub i64 %address.local1043, %address.local1044, !dbg !22
  %address.local1047 = add i64 %address.local1046, %43, !dbg !22
  %address.local1048 = shl nuw nsw i64 %address.local1047, 7, !dbg !22
  %1792 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %60, !dbg !22
  %address.local1050 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %1792, i64 %address.local1048, !dbg !22
  %address.local1052 = and i64 %address.local1041, 12, !dbg !13
  %address.local1053 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1050, i64 %address.local1052, !dbg !22
  store half %1791, ptr addrspace(1) %address.local1053, align 2, !dbg !22
  %address.local1055 = or disjoint i64 %address.local1052, 1, !dbg !22
  %address.local1056 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1050, i64 %address.local1055, !dbg !22
  store half %1790, ptr addrspace(1) %address.local1056, align 2, !dbg !22
  %address.local1058 = or disjoint i64 %address.local1052, 2, !dbg !22
  %address.local1059 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1050, i64 %address.local1058, !dbg !22
  store half %1789, ptr addrspace(1) %address.local1059, align 2, !dbg !22
  %address.local1061 = or disjoint i64 %address.local1052, 3, !dbg !22
  %address.local1062 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1050, i64 %address.local1061, !dbg !22
  store half %1788, ptr addrspace(1) %address.local1062, align 2, !dbg !22
  %address.local1064 = or disjoint i64 %address.local1052, 16, !dbg !22
  %address.local1065 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1050, i64 %address.local1064, !dbg !22
  store half %1775, ptr addrspace(1) %address.local1065, align 2, !dbg !22
  %address.local1067 = or disjoint i64 %address.local1052, 17, !dbg !22
  %address.local1068 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1050, i64 %address.local1067, !dbg !22
  store half %1774, ptr addrspace(1) %address.local1068, align 2, !dbg !22
  %address.local1070 = or disjoint i64 %address.local1052, 18, !dbg !22
  %address.local1071 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1050, i64 %address.local1070, !dbg !22
  store half %1773, ptr addrspace(1) %address.local1071, align 2, !dbg !22
  %address.local1073 = or disjoint i64 %address.local1052, 19, !dbg !22
  %address.local1074 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1050, i64 %address.local1073, !dbg !22
  store half %1772, ptr addrspace(1) %address.local1074, align 2, !dbg !22
  %address.local1078 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local1050, i64 16384, !dbg !22
  %address.local1079 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1078, i64 %address.local1052, !dbg !22
  store half %1759, ptr addrspace(1) %address.local1079, align 2, !dbg !22
  %address.local1081 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1078, i64 %address.local1055, !dbg !22
  store half %1758, ptr addrspace(1) %address.local1081, align 2, !dbg !22
  %address.local1083 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1078, i64 %address.local1058, !dbg !22
  store half %1757, ptr addrspace(1) %address.local1083, align 2, !dbg !22
  %address.local1085 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1078, i64 %address.local1061, !dbg !22
  store half %1756, ptr addrspace(1) %address.local1085, align 2, !dbg !22
  %address.local1087 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1078, i64 %address.local1064, !dbg !22
  store half %1743, ptr addrspace(1) %address.local1087, align 2, !dbg !22
  %address.local1089 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1078, i64 %address.local1067, !dbg !22
  store half %1742, ptr addrspace(1) %address.local1089, align 2, !dbg !22
  %address.local1091 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1078, i64 %address.local1070, !dbg !22
  store half %1741, ptr addrspace(1) %address.local1091, align 2, !dbg !22
  %address.local1093 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1078, i64 %address.local1073, !dbg !22
  store half %1740, ptr addrspace(1) %address.local1093, align 2, !dbg !22
  %1793 = insertelement <4 x float> poison, float %1626, i64 0, !dbg !20
  %1794 = insertelement <4 x float> %1793, float %1632, i64 1, !dbg !20
  %1795 = insertelement <4 x float> %1794, float %1627, i64 2, !dbg !20
  %1796 = insertelement <4 x float> %1795, float %1633, i64 3, !dbg !20
  %1797 = fmul <4 x float> %1796, %1727, !dbg !20
  %1798 = fptrunc <4 x float> %1797 to <4 x half>, !dbg !21
  %1799 = extractelement <4 x half> %1798, i64 3, !dbg !21
  %1800 = extractelement <4 x half> %1798, i64 2, !dbg !21
  %1801 = extractelement <4 x half> %1798, i64 1, !dbg !21
  %1802 = extractelement <4 x half> %1798, i64 0, !dbg !21
  %1803 = insertelement <4 x float> poison, float %1598, i64 0, !dbg !20
  %1804 = insertelement <4 x float> %1803, float %1604, i64 1, !dbg !20
  %1805 = insertelement <4 x float> %1804, float %1599, i64 2, !dbg !20
  %1806 = insertelement <4 x float> %1805, float %1605, i64 3, !dbg !20
  %1807 = fmul <4 x float> %1806, %1727, !dbg !20
  %1808 = fptrunc <4 x float> %1807 to <4 x half>, !dbg !21
  %1809 = extractelement <4 x half> %1808, i64 3, !dbg !21
  %1810 = extractelement <4 x half> %1808, i64 2, !dbg !21
  %1811 = extractelement <4 x half> %1808, i64 1, !dbg !21
  %1812 = extractelement <4 x half> %1808, i64 0, !dbg !21
  %1813 = insertelement <4 x float> poison, float %1569, i64 0, !dbg !20
  %1814 = insertelement <4 x float> %1813, float %1575, i64 1, !dbg !20
  %1815 = insertelement <4 x float> %1814, float %1570, i64 2, !dbg !20
  %1816 = insertelement <4 x float> %1815, float %1576, i64 3, !dbg !20
  %1817 = fmul <4 x float> %1816, %1725, !dbg !20
  %1818 = fptrunc <4 x float> %1817 to <4 x half>, !dbg !21
  %1819 = extractelement <4 x half> %1818, i64 3, !dbg !21
  %1820 = extractelement <4 x half> %1818, i64 2, !dbg !21
  %1821 = extractelement <4 x half> %1818, i64 1, !dbg !21
  %1822 = extractelement <4 x half> %1818, i64 0, !dbg !21
  %1823 = insertelement <4 x float> poison, float %1541, i64 0, !dbg !20
  %1824 = insertelement <4 x float> %1823, float %1547, i64 1, !dbg !20
  %1825 = insertelement <4 x float> %1824, float %1542, i64 2, !dbg !20
  %1826 = insertelement <4 x float> %1825, float %1548, i64 3, !dbg !20
  %1827 = fmul <4 x float> %1826, %1725, !dbg !20
  %1828 = fptrunc <4 x float> %1827 to <4 x half>, !dbg !21
  %1829 = extractelement <4 x half> %1828, i64 3, !dbg !21
  %1830 = extractelement <4 x half> %1828, i64 2, !dbg !21
  %1831 = extractelement <4 x half> %1828, i64 1, !dbg !21
  %1832 = extractelement <4 x half> %1828, i64 0, !dbg !21
  %address.local1109 = or disjoint i64 %address.local1052, 32, !dbg !22
  %address.local1110 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1050, i64 %address.local1109, !dbg !22
  store half %1832, ptr addrspace(1) %address.local1110, align 2, !dbg !22
  %address.local1112 = or disjoint i64 %address.local1052, 33, !dbg !22
  %address.local1113 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1050, i64 %address.local1112, !dbg !22
  store half %1831, ptr addrspace(1) %address.local1113, align 2, !dbg !22
  %address.local1115 = or disjoint i64 %address.local1052, 34, !dbg !22
  %address.local1116 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1050, i64 %address.local1115, !dbg !22
  store half %1830, ptr addrspace(1) %address.local1116, align 2, !dbg !22
  %address.local1118 = or disjoint i64 %address.local1052, 35, !dbg !22
  %address.local1119 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1050, i64 %address.local1118, !dbg !22
  store half %1829, ptr addrspace(1) %address.local1119, align 2, !dbg !22
  %address.local1121 = or disjoint i64 %address.local1052, 48, !dbg !22
  %address.local1122 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1050, i64 %address.local1121, !dbg !22
  store half %1822, ptr addrspace(1) %address.local1122, align 2, !dbg !22
  %address.local1124 = or disjoint i64 %address.local1052, 49, !dbg !22
  %address.local1125 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1050, i64 %address.local1124, !dbg !22
  store half %1821, ptr addrspace(1) %address.local1125, align 2, !dbg !22
  %address.local1127 = or disjoint i64 %address.local1052, 50, !dbg !22
  %address.local1128 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1050, i64 %address.local1127, !dbg !22
  store half %1820, ptr addrspace(1) %address.local1128, align 2, !dbg !22
  %address.local1130 = or i64 %address.local1041, 51, !dbg !22
  %address.local1131 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1050, i64 %address.local1130, !dbg !22
  store half %1819, ptr addrspace(1) %address.local1131, align 2, !dbg !22
  %address.local1136 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1078, i64 %address.local1109, !dbg !22
  store half %1812, ptr addrspace(1) %address.local1136, align 2, !dbg !22
  %address.local1138 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1078, i64 %address.local1112, !dbg !22
  store half %1811, ptr addrspace(1) %address.local1138, align 2, !dbg !22
  %address.local1140 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1078, i64 %address.local1115, !dbg !22
  store half %1810, ptr addrspace(1) %address.local1140, align 2, !dbg !22
  %address.local1142 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1078, i64 %address.local1118, !dbg !22
  store half %1809, ptr addrspace(1) %address.local1142, align 2, !dbg !22
  %address.local1144 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1078, i64 %address.local1121, !dbg !22
  store half %1802, ptr addrspace(1) %address.local1144, align 2, !dbg !22
  %address.local1146 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1078, i64 %address.local1124, !dbg !22
  store half %1801, ptr addrspace(1) %address.local1146, align 2, !dbg !22
  %address.local1148 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1078, i64 %address.local1127, !dbg !22
  store half %1800, ptr addrspace(1) %address.local1148, align 2, !dbg !22
  %address.local1150 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1078, i64 %address.local1130, !dbg !22
  store half %1799, ptr addrspace(1) %address.local1150, align 2, !dbg !22
  %1833 = insertelement <4 x float> poison, float %1512, i64 0, !dbg !20
  %1834 = insertelement <4 x float> %1833, float %1518, i64 1, !dbg !20
  %1835 = insertelement <4 x float> %1834, float %1513, i64 2, !dbg !20
  %1836 = insertelement <4 x float> %1835, float %1519, i64 3, !dbg !20
  %1837 = fmul <4 x float> %1836, %1727, !dbg !20
  %1838 = fptrunc <4 x float> %1837 to <4 x half>, !dbg !21
  %1839 = extractelement <4 x half> %1838, i64 3, !dbg !21
  %1840 = extractelement <4 x half> %1838, i64 2, !dbg !21
  %1841 = extractelement <4 x half> %1838, i64 1, !dbg !21
  %1842 = extractelement <4 x half> %1838, i64 0, !dbg !21
  %1843 = insertelement <4 x float> poison, float %1484, i64 0, !dbg !20
  %1844 = insertelement <4 x float> %1843, float %1490, i64 1, !dbg !20
  %1845 = insertelement <4 x float> %1844, float %1485, i64 2, !dbg !20
  %1846 = insertelement <4 x float> %1845, float %1491, i64 3, !dbg !20
  %1847 = fmul <4 x float> %1846, %1727, !dbg !20
  %1848 = fptrunc <4 x float> %1847 to <4 x half>, !dbg !21
  %1849 = extractelement <4 x half> %1848, i64 3, !dbg !21
  %1850 = extractelement <4 x half> %1848, i64 2, !dbg !21
  %1851 = extractelement <4 x half> %1848, i64 1, !dbg !21
  %1852 = extractelement <4 x half> %1848, i64 0, !dbg !21
  %1853 = insertelement <4 x float> poison, float %1455, i64 0, !dbg !20
  %1854 = insertelement <4 x float> %1853, float %1461, i64 1, !dbg !20
  %1855 = insertelement <4 x float> %1854, float %1456, i64 2, !dbg !20
  %1856 = insertelement <4 x float> %1855, float %1462, i64 3, !dbg !20
  %1857 = fmul <4 x float> %1856, %1725, !dbg !20
  %1858 = fptrunc <4 x float> %1857 to <4 x half>, !dbg !21
  %1859 = extractelement <4 x half> %1858, i64 3, !dbg !21
  %1860 = extractelement <4 x half> %1858, i64 2, !dbg !21
  %1861 = extractelement <4 x half> %1858, i64 1, !dbg !21
  %1862 = extractelement <4 x half> %1858, i64 0, !dbg !21
  %1863 = insertelement <4 x float> poison, float %1427, i64 0, !dbg !20
  %1864 = insertelement <4 x float> %1863, float %1433, i64 1, !dbg !20
  %1865 = insertelement <4 x float> %1864, float %1428, i64 2, !dbg !20
  %1866 = insertelement <4 x float> %1865, float %1434, i64 3, !dbg !20
  %1867 = fmul <4 x float> %1866, %1725, !dbg !20
  %1868 = fptrunc <4 x float> %1867 to <4 x half>, !dbg !21
  %1869 = extractelement <4 x half> %1868, i64 3, !dbg !21
  %1870 = extractelement <4 x half> %1868, i64 2, !dbg !21
  %1871 = extractelement <4 x half> %1868, i64 1, !dbg !21
  %1872 = extractelement <4 x half> %1868, i64 0, !dbg !21
  %address.local1166 = or disjoint i64 %address.local1052, 64, !dbg !22
  %address.local1167 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1050, i64 %address.local1166, !dbg !22
  store half %1872, ptr addrspace(1) %address.local1167, align 2, !dbg !22
  %address.local1169 = or disjoint i64 %address.local1052, 65, !dbg !22
  %address.local1170 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1050, i64 %address.local1169, !dbg !22
  store half %1871, ptr addrspace(1) %address.local1170, align 2, !dbg !22
  %address.local1172 = or disjoint i64 %address.local1052, 66, !dbg !22
  %address.local1173 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1050, i64 %address.local1172, !dbg !22
  store half %1870, ptr addrspace(1) %address.local1173, align 2, !dbg !22
  %address.local1175 = or disjoint i64 %address.local1052, 67, !dbg !22
  %address.local1176 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1050, i64 %address.local1175, !dbg !22
  store half %1869, ptr addrspace(1) %address.local1176, align 2, !dbg !22
  %address.local1178 = or disjoint i64 %address.local1052, 80, !dbg !22
  %address.local1179 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1050, i64 %address.local1178, !dbg !22
  store half %1862, ptr addrspace(1) %address.local1179, align 2, !dbg !22
  %address.local1181 = or disjoint i64 %address.local1052, 81, !dbg !22
  %address.local1182 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1050, i64 %address.local1181, !dbg !22
  store half %1861, ptr addrspace(1) %address.local1182, align 2, !dbg !22
  %address.local1184 = or disjoint i64 %address.local1052, 82, !dbg !22
  %address.local1185 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1050, i64 %address.local1184, !dbg !22
  store half %1860, ptr addrspace(1) %address.local1185, align 2, !dbg !22
  %address.local1187 = or disjoint i64 %address.local1052, 83, !dbg !22
  %address.local1188 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1050, i64 %address.local1187, !dbg !22
  store half %1859, ptr addrspace(1) %address.local1188, align 2, !dbg !22
  %address.local1193 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1078, i64 %address.local1166, !dbg !22
  store half %1852, ptr addrspace(1) %address.local1193, align 2, !dbg !22
  %address.local1195 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1078, i64 %address.local1169, !dbg !22
  store half %1851, ptr addrspace(1) %address.local1195, align 2, !dbg !22
  %address.local1197 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1078, i64 %address.local1172, !dbg !22
  store half %1850, ptr addrspace(1) %address.local1197, align 2, !dbg !22
  %address.local1199 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1078, i64 %address.local1175, !dbg !22
  store half %1849, ptr addrspace(1) %address.local1199, align 2, !dbg !22
  %address.local1201 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1078, i64 %address.local1178, !dbg !22
  store half %1842, ptr addrspace(1) %address.local1201, align 2, !dbg !22
  %address.local1203 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1078, i64 %address.local1181, !dbg !22
  store half %1841, ptr addrspace(1) %address.local1203, align 2, !dbg !22
  %address.local1205 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1078, i64 %address.local1184, !dbg !22
  store half %1840, ptr addrspace(1) %address.local1205, align 2, !dbg !22
  %address.local1207 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1078, i64 %address.local1187, !dbg !22
  store half %1839, ptr addrspace(1) %address.local1207, align 2, !dbg !22
  %1873 = insertelement <4 x float> poison, float %1398, i64 0, !dbg !20
  %1874 = insertelement <4 x float> %1873, float %1404, i64 1, !dbg !20
  %1875 = insertelement <4 x float> %1874, float %1399, i64 2, !dbg !20
  %1876 = insertelement <4 x float> %1875, float %1405, i64 3, !dbg !20
  %1877 = fmul <4 x float> %1876, %1727, !dbg !20
  %1878 = fptrunc <4 x float> %1877 to <4 x half>, !dbg !21
  %1879 = extractelement <4 x half> %1878, i64 3, !dbg !21
  %1880 = extractelement <4 x half> %1878, i64 2, !dbg !21
  %1881 = extractelement <4 x half> %1878, i64 1, !dbg !21
  %1882 = extractelement <4 x half> %1878, i64 0, !dbg !21
  %1883 = insertelement <4 x float> poison, float %1370, i64 0, !dbg !20
  %1884 = insertelement <4 x float> %1883, float %1376, i64 1, !dbg !20
  %1885 = insertelement <4 x float> %1884, float %1371, i64 2, !dbg !20
  %1886 = insertelement <4 x float> %1885, float %1377, i64 3, !dbg !20
  %1887 = fmul <4 x float> %1886, %1727, !dbg !20
  %1888 = fptrunc <4 x float> %1887 to <4 x half>, !dbg !21
  %1889 = extractelement <4 x half> %1888, i64 3, !dbg !21
  %1890 = extractelement <4 x half> %1888, i64 2, !dbg !21
  %1891 = extractelement <4 x half> %1888, i64 1, !dbg !21
  %1892 = extractelement <4 x half> %1888, i64 0, !dbg !21
  %1893 = insertelement <4 x float> poison, float %1341, i64 0, !dbg !20
  %1894 = insertelement <4 x float> %1893, float %1347, i64 1, !dbg !20
  %1895 = insertelement <4 x float> %1894, float %1342, i64 2, !dbg !20
  %1896 = insertelement <4 x float> %1895, float %1348, i64 3, !dbg !20
  %1897 = fmul <4 x float> %1896, %1725, !dbg !20
  %1898 = fptrunc <4 x float> %1897 to <4 x half>, !dbg !21
  %1899 = extractelement <4 x half> %1898, i64 3, !dbg !21
  %1900 = extractelement <4 x half> %1898, i64 2, !dbg !21
  %1901 = extractelement <4 x half> %1898, i64 1, !dbg !21
  %1902 = extractelement <4 x half> %1898, i64 0, !dbg !21
  %1903 = insertelement <4 x float> poison, float %1313, i64 0, !dbg !20
  %1904 = insertelement <4 x float> %1903, float %1319, i64 1, !dbg !20
  %1905 = insertelement <4 x float> %1904, float %1314, i64 2, !dbg !20
  %1906 = insertelement <4 x float> %1905, float %1320, i64 3, !dbg !20
  %1907 = fmul <4 x float> %1906, %1725, !dbg !20
  %1908 = fptrunc <4 x float> %1907 to <4 x half>, !dbg !21
  %1909 = extractelement <4 x half> %1908, i64 3, !dbg !21
  %1910 = extractelement <4 x half> %1908, i64 2, !dbg !21
  %1911 = extractelement <4 x half> %1908, i64 1, !dbg !21
  %1912 = extractelement <4 x half> %1908, i64 0, !dbg !21
  %address.local1223 = or disjoint i64 %address.local1052, 96, !dbg !22
  %address.local1224 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1050, i64 %address.local1223, !dbg !22
  store half %1912, ptr addrspace(1) %address.local1224, align 2, !dbg !22
  %address.local1226 = or disjoint i64 %address.local1052, 97, !dbg !22
  %address.local1227 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1050, i64 %address.local1226, !dbg !22
  store half %1911, ptr addrspace(1) %address.local1227, align 2, !dbg !22
  %address.local1229 = or disjoint i64 %address.local1052, 98, !dbg !22
  %address.local1230 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1050, i64 %address.local1229, !dbg !22
  store half %1910, ptr addrspace(1) %address.local1230, align 2, !dbg !22
  %address.local1232 = or disjoint i64 %address.local1052, 99, !dbg !22
  %address.local1233 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1050, i64 %address.local1232, !dbg !22
  store half %1909, ptr addrspace(1) %address.local1233, align 2, !dbg !22
  %address.local1235 = or disjoint i64 %address.local1052, 112, !dbg !22
  %address.local1236 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1050, i64 %address.local1235, !dbg !22
  store half %1902, ptr addrspace(1) %address.local1236, align 2, !dbg !22
  %address.local1238 = or disjoint i64 %address.local1052, 113, !dbg !22
  %address.local1239 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1050, i64 %address.local1238, !dbg !22
  store half %1901, ptr addrspace(1) %address.local1239, align 2, !dbg !22
  %address.local1241 = or disjoint i64 %address.local1052, 114, !dbg !22
  %address.local1242 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1050, i64 %address.local1241, !dbg !22
  store half %1900, ptr addrspace(1) %address.local1242, align 2, !dbg !22
  %address.local1244 = or i64 %address.local1041, 115, !dbg !22
  %address.local1245 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1050, i64 %address.local1244, !dbg !22
  store half %1899, ptr addrspace(1) %address.local1245, align 2, !dbg !22
  %address.local1250 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1078, i64 %address.local1223, !dbg !22
  store half %1892, ptr addrspace(1) %address.local1250, align 2, !dbg !22
  %address.local1252 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1078, i64 %address.local1226, !dbg !22
  store half %1891, ptr addrspace(1) %address.local1252, align 2, !dbg !22
  %address.local1254 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1078, i64 %address.local1229, !dbg !22
  store half %1890, ptr addrspace(1) %address.local1254, align 2, !dbg !22
  %address.local1256 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1078, i64 %address.local1232, !dbg !22
  store half %1889, ptr addrspace(1) %address.local1256, align 2, !dbg !22
  %address.local1258 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1078, i64 %address.local1235, !dbg !22
  store half %1882, ptr addrspace(1) %address.local1258, align 2, !dbg !22
  %address.local1260 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1078, i64 %address.local1238, !dbg !22
  store half %1881, ptr addrspace(1) %address.local1260, align 2, !dbg !22
  %address.local1262 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1078, i64 %address.local1241, !dbg !22
  store half %1880, ptr addrspace(1) %address.local1262, align 2, !dbg !22
  %address.local1264 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1078, i64 %address.local1244, !dbg !22
  store half %1879, ptr addrspace(1) %address.local1264, align 2, !dbg !22
  ret void, !dbg !5
}

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32) #2

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.hi(i32, i32) #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #3

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.y() #3

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.z() #3

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #3

; Function Attrs: convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.readfirstlane.i32(i32) #4

; Function Attrs: nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.asyncmark() #5

; Function Attrs: nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.wait.asyncmark(i16 immarg) #5

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier() #6

; Function Attrs: convergent nocallback nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.ds.bpermute(i32, i32) #7

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maxnum.f32(float, float) #8

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare <1 x float> @llvm.maxnum.v1f32(<1 x float>, <1 x float>) #8

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.amdgcn.exp2.f32(float) #8

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.sched.barrier(i32 immarg) #6

attributes #0 = { convergent willreturn memory(read) }
attributes #1 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1" "frisk.global_prefetch"="8" "frisk.k_dma"="64-swizzle:1" "frisk.k_lds_prefetch"="8" "frisk.k_tail_prefetch"="4" "frisk.register_comparisons"="1" "frisk.register_lifetimes"="1" "frisk.softmax_cvt_pack"="rtz" "frisk.softmax_fragments"="pv" "uniform-work-group-size" }
attributes #2 = { nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }
attributes #3 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #4 = { convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none) }
attributes #5 = { nocallback nofree nounwind willreturn }
attributes #6 = { convergent nocallback nofree nounwind willreturn }
attributes #7 = { convergent nocallback nofree nounwind willreturn memory(none) }
attributes #8 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #9 = { convergent nounwind willreturn memory(none) }
attributes #10 = { convergent nounwind }
attributes #11 = { memory(inaccessiblemem: readwrite) }

!llvm.dbg.cu = !{!3}
!llvm.module.flags = !{!32}

!0 = !DIFile(filename: "causal.input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/causal_cvt_pack_mytest")
!1 = !{}
!2 = !DISubroutineType(types: !1)
!3 = distinct !DICompileUnit(language: DW_LANG_C, file: !0, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!4 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !0, file: !0, line: 23, type: !2, scopeLine: 23, spFlags: DISPFlagDefinition, unit: !3)
!5 = !DILocation(line: 23, column: 3, scope: !4)
!6 = !DILocation(line: 60, column: 24, scope: !4)
!8 = !DILocation(line: 89, column: 20, scope: !4)
!9 = !DILocation(line: 70, column: 22, scope: !4)
!13 = !DILocation(line: 72, column: 24, scope: !4)
!14 = !DILocation(line: 104, column: 20, scope: !4)
!15 = !DILocation(line: 0, scope: !4)
!16 = !DILocation(line: 71, column: 22, scope: !4)
!17 = !DILocation(line: 74, column: 23, scope: !4)
!18 = !DILocation(line: 101, column: 20, scope: !4)
!19 = !DILocation(line: 77, column: 19, scope: !4)
!20 = !DILocation(line: 107, column: 23, scope: !4)
!21 = !DILocation(line: 108, column: 19, scope: !4)
!22 = !DILocation(line: 109, column: 9, scope: !4)
!23 = !DILocation(line: 90, column: 22, scope: !4)
!24 = !DILocation(line: 91, column: 23, scope: !4)
!25 = !DILocation(line: 92, column: 20, scope: !4)
!26 = !DILocation(line: 93, column: 22, scope: !4)
!27 = !DILocation(line: 94, column: 22, scope: !4)
!28 = !DILocation(line: 95, column: 20, scope: !4)
!29 = !DILocation(line: 96, column: 16, scope: !4)
!30 = !DILocation(line: 103, column: 22, scope: !4)
!31 = !{i32 256, i32 1, i32 1}
!32 = !{i32 2, !"Debug Info Version", i32 3}
!33 = !{i32 0, i32 256}
!34 = !{i32 4}
!35 = !{i64 8192}
!36 = !{i32 0}
!37 = !{i32 1}
!38 = !{i32 2}
!39 = !{i32 3}
!40 = !{i32 5}
!41 = !{i32 6}
!42 = !{i32 7}
!43 = !{i32 8}
!44 = !{i32 9}
!45 = !{i32 10}
!46 = !{i32 11}
!47 = !{i32 12}
!48 = !{i32 13}
!49 = !{i32 14}
!50 = !{i32 15}
!51 = !{i32 16}
!52 = !{i32 17}
!53 = !{i32 18}
!54 = !{i32 19}
!55 = !{i32 20}
!56 = !{i32 21}
!57 = !{i32 22}
