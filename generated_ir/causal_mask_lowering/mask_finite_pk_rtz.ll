; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/causal_cvt_pack_mytest/causal.input.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/causal_cvt_pack_mytest/causal.input.mlir"

@shm_0 = addrspace(3) global [16384 x i8] undef, align 16

; Function Attrs: convergent willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

declare float @__ocml_exp2_f32(float)

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3) #1 !dbg !4 !reqd_work_group_size !33 {
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
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #9, !dbg !8, !range !35, !frisk.register_lifetimes !1
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
  %.idx1447 = shl i64 %32, 8, !dbg !8
  %37 = add i64 %.idx, %.idx1447, !dbg !8
  %address.local481.idx = shl nuw nsw i64 %address.local480, 1, !dbg !8
  %38 = or disjoint i64 %37, %address.local481.idx, !dbg !8
  %dma.buffer.offset1445 = trunc i64 %38 to i32, !dbg !8
  %39 = insertelement <2 x i64> poison, i64 %36, i64 0, !dbg !8
  %40 = bitcast <2 x i64> %39 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc1446 = shufflevector <4 x i32> %40, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
  %41 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %35, i32 %dma.buffer.offset1445, <4 x i32> %dma.buffer.rsrc1446) #10, !dbg !8
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
  %dma.buffer.offset1443 = trunc i64 %43 to i32, !dbg !8
  %44 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %42, i32 %dma.buffer.offset1443, <4 x i32> %dma.buffer.rsrc1446) #10, !dbg !8
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
  %dma.buffer.offset1441 = shl i32 %.tr, 1, !dbg !8
  %50 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %47, i32 %dma.buffer.offset1441, <4 x i32> %dma.buffer.rsrc1446) #10, !dbg !8
  %address.local498 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local484, !dbg !8
  %address.local499 = ptrtoint ptr addrspace(3) %address.local498 to i64, !dbg !8
  %address.local500 = trunc i64 %address.local499 to i32, !dbg !8
  %51 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local500), !dbg !8
  %52 = or disjoint i64 %address.local492, %48, !dbg !8
  %.tr1451 = trunc i64 %52 to i32, !dbg !8
  %dma.buffer.offset1439 = shl i32 %.tr1451, 1, !dbg !8
  %53 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %51, i32 %dma.buffer.offset1439, <4 x i32> %dma.buffer.rsrc1446) #10, !dbg !8
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
  %lds.const.gep1268 = getelementptr i8, ptr addrspace(3) %address.local514, i64 1024, !dbg !12
  %58 = load <4 x half>, ptr addrspace(3) %lds.const.gep1268, align 8, !dbg !12
  %59 = fmul <4 x half> %56, splat (half 0xH3015), !dbg !12
  %lds.const.gep1269 = getelementptr i8, ptr addrspace(3) %address.local514, i64 5120, !dbg !12
  %60 = load <4 x half>, ptr addrspace(3) %lds.const.gep1269, align 8, !dbg !12
  %61 = fmul <4 x half> %58, splat (half 0xH3015), !dbg !12
  %62 = fmul <4 x half> %60, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %63 = or disjoint i64 %46, 64, !dbg !8
  %64 = or disjoint i64 %address.local480, %63, !dbg !8
  %.tr1453 = trunc i64 %64 to i32, !dbg !8
  %dma.buffer.offset1437 = shl i32 %.tr1453, 1, !dbg !8
  %65 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %35, i32 %dma.buffer.offset1437, <4 x i32> %dma.buffer.rsrc1446) #10, !dbg !8
  %66 = or disjoint i64 %address.local492, %63, !dbg !8
  %.tr1455 = trunc i64 %66 to i32, !dbg !8
  %dma.buffer.offset1435 = shl i32 %.tr1455, 1, !dbg !8
  %67 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %42, i32 %dma.buffer.offset1435, <4 x i32> %dma.buffer.rsrc1446) #10, !dbg !8
  %lds.const.gep1270 = getelementptr i8, ptr addrspace(3) %address.local514, i64 8192, !dbg !12
  %68 = load <4 x half>, ptr addrspace(3) %lds.const.gep1270, align 8, !dbg !12
  %lds.const.gep1271 = getelementptr i8, ptr addrspace(3) %address.local514, i64 12288, !dbg !12
  %69 = load <4 x half>, ptr addrspace(3) %lds.const.gep1271, align 8, !dbg !12
  %lds.const.gep1272 = getelementptr i8, ptr addrspace(3) %address.local514, i64 9216, !dbg !12
  %70 = load <4 x half>, ptr addrspace(3) %lds.const.gep1272, align 8, !dbg !12
  %71 = fmul <4 x half> %68, splat (half 0xH3015), !dbg !12
  %lds.const.gep1273 = getelementptr i8, ptr addrspace(3) %address.local514, i64 13312, !dbg !12
  %72 = load <4 x half>, ptr addrspace(3) %lds.const.gep1273, align 8, !dbg !12
  %73 = fmul <4 x half> %69, splat (half 0xH3015), !dbg !12
  %74 = fmul <4 x half> %70, splat (half 0xH3015), !dbg !12
  %75 = fmul <4 x half> %72, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %76 = or disjoint i64 %46, 96, !dbg !8
  %77 = or disjoint i64 %address.local480, %76, !dbg !8
  %.tr1457 = trunc i64 %77 to i32, !dbg !8
  %dma.buffer.offset1433 = shl i32 %.tr1457, 1, !dbg !8
  %78 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %47, i32 %dma.buffer.offset1433, <4 x i32> %dma.buffer.rsrc1446) #10, !dbg !8
  %79 = or disjoint i64 %address.local492, %76, !dbg !8
  %.tr1459 = trunc i64 %79 to i32, !dbg !8
  %dma.buffer.offset1431 = shl i32 %.tr1459, 1, !dbg !8
  %80 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %51, i32 %dma.buffer.offset1431, <4 x i32> %dma.buffer.rsrc1446) #10, !dbg !8
  %81 = load <4 x half>, ptr addrspace(3) %address.local514, align 8, !dbg !12
  %82 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !12
  %83 = load <4 x half>, ptr addrspace(3) %lds.const.gep1268, align 8, !dbg !12
  %84 = fmul <4 x half> %81, splat (half 0xH3015), !dbg !12
  %85 = load <4 x half>, ptr addrspace(3) %lds.const.gep1269, align 8, !dbg !12
  %86 = fmul <4 x half> %82, splat (half 0xH3015), !dbg !12
  %87 = fmul <4 x half> %83, splat (half 0xH3015), !dbg !12
  %88 = fmul <4 x half> %85, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %89 = load <4 x half>, ptr addrspace(3) %lds.const.gep1270, align 8, !dbg !12
  %90 = load <4 x half>, ptr addrspace(3) %lds.const.gep1271, align 8, !dbg !12
  %91 = load <4 x half>, ptr addrspace(3) %lds.const.gep1272, align 8, !dbg !12
  %92 = fmul <4 x half> %89, splat (half 0xH3015), !dbg !12
  %93 = load <4 x half>, ptr addrspace(3) %lds.const.gep1273, align 8, !dbg !12
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
  %115 = trunc i64 %113 to i32, !dbg !18
  %116 = add i32 %115, 1, !dbg !18
  %117 = add i32 %115, 65, !dbg !18
  br label %118, !dbg !7

118:                                              ; preds = %127, %4
  %119 = phi i64 [ %1282, %127 ], [ 0, %4 ], !dbg !7
  %120 = phi [2 x <8 x float>] [ %1267, %127 ], [ zeroinitializer, %4 ], !dbg !7
  %121 = phi [2 x <1 x float>] [ %1033, %127 ], [ zeroinitializer, %4 ], !dbg !7
  %122 = phi [2 x <1 x float>] [ %1269, %127 ], [ [<1 x float> splat (float 0xFFF0000000000000), <1 x float> splat (float 0xFFF0000000000000)], %4 ], !dbg !7
  %123 = phi [2 x <8 x float>] [ %1273, %127 ], [ zeroinitializer, %4 ], !dbg !7
  %124 = phi [2 x <8 x float>] [ %1277, %127 ], [ zeroinitializer, %4 ], !dbg !7
  %125 = phi [2 x <8 x float>] [ %1281, %127 ], [ zeroinitializer, %4 ], !dbg !7
  %126 = icmp slt i64 %119, %97, !dbg !7
  br i1 %126, label %127, label %1283, !dbg !7

127:                                              ; preds = %118
  %128 = lshr i64 %119, 12, !dbg !15
  %129 = add nuw nsw i64 %128, %18, !dbg !15
  %130 = lshr i64 %119, 5, !dbg !15
  %131 = and i64 %130, 127, !dbg !15
  %132 = shl nsw i64 %119, 7, !dbg !15
  %133 = shl i64 %129, 19, !dbg !15
  %134 = and i64 %133, 16252928, !dbg !15
  %135 = shl i64 %130, 19, !dbg !15
  %136 = shl i64 %119, 14, !dbg !15
  %137 = sub i64 %136, %135, !dbg !15
  %address.thread565 = call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #9, !dbg !15, !range !35, !frisk.register_lifetimes !1
  %address.local566 = zext nneg i32 %address.thread565 to i64, !dbg !8
  %address.local567 = shl nuw nsw i64 %address.local566, 3, !dbg !8
  %address.local568 = and i64 %address.local567, 1536, !dbg !8
  %address.local569 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local568, !dbg !8
  %address.local570 = ptrtoint ptr addrspace(3) %address.local569 to i64, !dbg !8
  %address.local571 = trunc i64 %address.local570 to i32, !dbg !8
  %138 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local571), !dbg !15
  %139 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !15
  %.idx14601463 = or disjoint i64 %134, %131, !dbg !15
  %140 = add i64 %.idx14601463, %137, !dbg !15
  %141 = insertelement <2 x i64> poison, i64 %139, i64 0, !dbg !15
  %142 = bitcast <2 x i64> %141 to <4 x i32>, !dbg !15
  %dma.buffer.rsrc1430 = shufflevector <4 x i32> %142, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %kdma.source = zext nneg i32 %kdma.owner to i64, !dbg !15
  %kdma.source1603 = lshr i64 %kdma.source, 4, !dbg !15
  %kdma.source1604 = and i64 %kdma.source1603, 8, !dbg !15
  %kdma.source1605 = and i64 %kdma.source, 91, !dbg !15
  %kdma.source1606 = lshr i64 %kdma.source, 2, !dbg !15
  %kdma.source1607 = and i64 %kdma.source1606, 4, !dbg !15
  %kdma.source1608 = add nuw nsw i64 %kdma.source1607, %kdma.source, !dbg !15
  %kdma.source1609 = and i64 %kdma.source1608, 4, !dbg !15
  %kdma.source1610 = or disjoint i64 %kdma.source1605, %kdma.source1609, !dbg !15
  %kdma.source1611 = shl nuw nsw i64 %kdma.source1610, 7, !dbg !15
  %kdma.source1612 = or disjoint i64 %kdma.source1604, %kdma.source1611, !dbg !15
  %kdma.source1613 = add nuw nsw i64 %140, %kdma.source1612, !dbg !15
  %kdma.source1614 = trunc i64 %kdma.source1613 to i32, !dbg !15
  %kdma.source1615 = shl i32 %kdma.source1614, 1, !dbg !15
  %kdma.offset = add i32 %kdma.source1615, %kdma.intra, !dbg !15
  %143 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %138, i32 %kdma.offset, <4 x i32> %dma.buffer.rsrc1430) #10, !dbg !15
  %kdma.source1616 = zext nneg i32 %kdma.owner.hi to i64, !dbg !15
  %kdma.source1617 = lshr i64 %kdma.source1616, 4, !dbg !15
  %kdma.source1618 = and i64 %kdma.source1617, 8, !dbg !15
  %kdma.source1619 = and i64 %kdma.source1616, 123, !dbg !15
  %kdma.source1620 = lshr i64 %kdma.source1616, 2, !dbg !15
  %kdma.source1621 = and i64 %kdma.source1620, 4, !dbg !15
  %kdma.source1622 = add nuw nsw i64 %kdma.source1621, %kdma.source1616, !dbg !15
  %kdma.source1623 = and i64 %kdma.source1622, 4, !dbg !15
  %kdma.source1624 = or disjoint i64 %kdma.source1619, %kdma.source1623, !dbg !15
  %kdma.source1625 = shl nuw nsw i64 %kdma.source1624, 7, !dbg !15
  %kdma.source1626 = or disjoint i64 %kdma.source1618, %kdma.source1625, !dbg !15
  %kdma.source1627 = add i64 %140, %kdma.source1626, !dbg !15
  %kdma.source1628 = trunc i64 %kdma.source1627 to i32, !dbg !15
  %kdma.source1629 = shl i32 %kdma.source1628, 1, !dbg !15
  %kdma.offset1630 = add i32 %kdma.source1629, %kdma.intra.hi, !dbg !15
  %kdma.lds1631 = add i32 %138, 512, !dbg !15
  %144 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1631, i32 %kdma.offset1630, <4 x i32> %dma.buffer.rsrc1430) #10, !dbg !15
  %address.local583 = or disjoint i64 %address.local566, 256, !dbg !8
  %address.local584 = shl nuw nsw i64 %address.local583, 3, !dbg !8
  %address.local585 = and i64 %address.local584, 3584, !dbg !8
  %address.local586 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local585, !dbg !8
  %address.local587 = ptrtoint ptr addrspace(3) %address.local586 to i64, !dbg !8
  %address.local588 = trunc i64 %address.local587 to i32, !dbg !8
  %145 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local588), !dbg !15
  %kdma.source1643 = or disjoint i64 %kdma.source1612, 16, !dbg !15
  %kdma.source1644 = add nuw nsw i64 %140, %kdma.source1643, !dbg !15
  %kdma.source1645 = trunc i64 %kdma.source1644 to i32, !dbg !15
  %kdma.source1646 = shl i32 %kdma.source1645, 1, !dbg !15
  %kdma.offset1647 = add i32 %kdma.source1646, %kdma.intra, !dbg !15
  %146 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %145, i32 %kdma.offset1647, <4 x i32> %dma.buffer.rsrc1430) #10, !dbg !15
  %kdma.source1660 = or disjoint i64 %kdma.source1626, 16, !dbg !15
  %kdma.source1661 = add i64 %140, %kdma.source1660, !dbg !15
  %kdma.source1662 = trunc i64 %kdma.source1661 to i32, !dbg !15
  %kdma.source1663 = shl i32 %kdma.source1662, 1, !dbg !15
  %kdma.offset1664 = add i32 %kdma.source1663, %kdma.intra.hi, !dbg !15
  %kdma.lds1665 = add i32 %145, 512, !dbg !15
  %147 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1665, i32 %kdma.offset1664, <4 x i32> %dma.buffer.rsrc1430) #10, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %148 = add nuw i64 %119, %98, !dbg !15
  %149 = add i64 %148, %99, !dbg !15
  %150 = add i64 %149, 1024, !dbg !15
  %151 = icmp slt i64 %150, 0, !dbg !15
  %152 = sub i64 -1025, %149, !dbg !15
  %153 = select i1 %151, i64 %152, i64 %150, !dbg !15
  %154 = sdiv i64 %153, 4096, !dbg !15
  %.lobit37 = ashr i64 %150, 63, !dbg !15
  %155 = xor i64 %154, %.lobit37, !dbg !15
  %156 = add nsw i64 %130, %102, !dbg !15
  %157 = add nsw i64 %156, 32, !dbg !15
  %158 = icmp slt i64 %156, -32, !dbg !15
  %159 = sub nsw i64 -33, %156, !dbg !15
  %160 = select i1 %158, i64 %159, i64 %157, !dbg !15
  %161 = sdiv i64 %160, 128, !dbg !15
  %162 = sext i1 %158 to i64, !dbg !15
  %163 = xor i64 %161, %162, !dbg !15
  %164 = shl nsw i64 %163, 7, !dbg !15
  %165 = sub i64 %102, %164, !dbg !15
  %166 = shl i64 %155, 19, !dbg !15
  %167 = and i64 %166, 16252928, !dbg !15
  %address.local600 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local568, !dbg !8
  %address.local601 = ptrtoint ptr addrspace(3) %address.local600 to i64, !dbg !8
  %address.local602 = trunc i64 %address.local601 to i32, !dbg !8
  %168 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local602), !dbg !15
  %.idx14701474 = add nuw nsw i64 %167, %130, !dbg !15
  %169 = add i64 %.idx14701474, %165, !dbg !15
  %170 = add i64 %169, %137, !dbg !15
  %kdma.source1677 = add i64 %170, %kdma.source1612, !dbg !15
  %kdma.source1678 = trunc i64 %kdma.source1677 to i32, !dbg !15
  %kdma.source1679 = shl i32 %kdma.source1678, 1, !dbg !15
  %kdma.source1680 = add i32 %kdma.source1679, 64, !dbg !15
  %kdma.offset1681 = add i32 %kdma.source1680, %kdma.intra, !dbg !15
  %171 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %168, i32 %kdma.offset1681, <4 x i32> %dma.buffer.rsrc1430) #10, !dbg !15
  %kdma.source1694 = add i64 %170, %kdma.source1626, !dbg !15
  %kdma.source1695 = trunc i64 %kdma.source1694 to i32, !dbg !15
  %kdma.source1696 = shl i32 %kdma.source1695, 1, !dbg !15
  %kdma.source1697 = add i32 %kdma.source1696, 64, !dbg !15
  %kdma.offset1698 = add i32 %kdma.source1697, %kdma.intra.hi, !dbg !15
  %kdma.lds1699 = add i32 %168, 512, !dbg !15
  %172 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1699, i32 %kdma.offset1698, <4 x i32> %dma.buffer.rsrc1430) #10, !dbg !15
  %address.local604 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local585, !dbg !8
  %address.local605 = ptrtoint ptr addrspace(3) %address.local604 to i64, !dbg !8
  %address.local606 = trunc i64 %address.local605 to i32, !dbg !8
  %173 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local606), !dbg !15
  %kdma.source1712 = add i64 %170, %kdma.source1643, !dbg !15
  %kdma.source1713 = trunc i64 %kdma.source1712 to i32, !dbg !15
  %kdma.source1714 = shl i32 %kdma.source1713, 1, !dbg !15
  %kdma.source1715 = add i32 %kdma.source1714, 64, !dbg !15
  %kdma.offset1716 = add i32 %kdma.source1715, %kdma.intra, !dbg !15
  %174 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %173, i32 %kdma.offset1716, <4 x i32> %dma.buffer.rsrc1430) #10, !dbg !15
  %kdma.source1730 = add i64 %170, %kdma.source1660, !dbg !15
  %kdma.source1731 = trunc i64 %kdma.source1730 to i32, !dbg !15
  %kdma.source1732 = shl i32 %kdma.source1731, 1, !dbg !15
  %kdma.source1733 = add i32 %kdma.source1732, 64, !dbg !15
  %kdma.offset1734 = add i32 %kdma.source1733, %kdma.intra.hi, !dbg !15
  %kdma.lds1735 = add i32 %173, 512, !dbg !15
  %175 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1735, i32 %kdma.offset1734, <4 x i32> %dma.buffer.rsrc1430) #10, !dbg !15
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
  %.idx1482 = mul nuw nsw i64 %address.local566, 320, !dbg !16
  %176 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %.idx1482, !dbg !16
  %.idx1483 = mul nsw i64 %address.local609, -1280, !dbg !16
  %177 = getelementptr i8, ptr addrspace(3) %176, i64 %.idx1483, !dbg !16
  %178 = getelementptr [2 x i8], ptr addrspace(3) %177, i64 %address.local613, !dbg !16
  %179 = getelementptr [2 x i8], ptr addrspace(3) %178, i64 %address.local617, !dbg !16
  %180 = getelementptr [2 x i8], ptr addrspace(3) %179, i64 %address.local620, !dbg !16
  %address.local626 = getelementptr [2 x i8], ptr addrspace(3) %180, i64 %address.local624, !dbg !16
  %181 = add nsw i64 %.idx1482, %.idx1483, !dbg !16
  %.idx2002 = shl nuw nsw i64 %address.local613, 1, !dbg !16
  %182 = or disjoint i64 %181, %.idx2002, !dbg !16
  %183 = or disjoint i64 %address.local617, %address.local620, !dbg !16
  %184 = shl nsw i64 %183, 1, !dbg !16
  %185 = add nsw i64 %184, %182, !dbg !16
  %address.local626.idx = mul nuw nsw i64 %address.local623, 2032, !dbg !16
  %186 = add nsw i64 %185, %address.local626.idx, !dbg !16
  %187 = lshr i64 %186, 6, !dbg !16
  %188 = and i64 %187, 8, !dbg !16
  %189 = xor i64 %186, %188, !dbg !16
  %190 = sub i64 %189, %186, !dbg !16
  %kdma.swizzled = getelementptr i8, ptr addrspace(3) %address.local626, i64 %190, !dbg !16
  %191 = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8, !dbg !16
  %192 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %191, <4 x float> zeroinitializer) #9, !dbg !16
  %kdma.gep = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4096, !dbg !16
  %193 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8, !dbg !16
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
  %194 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local651, !dbg !16
  %195 = getelementptr [2 x i8], ptr addrspace(3) %194, i64 %address.local620, !dbg !16
  %196 = getelementptr [2 x i8], ptr addrspace(3) %195, i64 %address.local624, !dbg !16
  %address.local655 = getelementptr i8, ptr addrspace(3) %196, i64 64, !dbg !16
  %.idx20052008 = add i64 %address.local651, %address.local620, !dbg !16
  %197 = shl i64 %.idx20052008, 1, !dbg !16
  %198 = add i64 %197, %address.local626.idx, !dbg !16
  %199 = add i64 %198, 64, !dbg !16
  %200 = lshr i64 %199, 6, !dbg !16
  %201 = and i64 %200, 8, !dbg !16
  %202 = xor i64 %199, %201, !dbg !16
  %203 = sub i64 %202, %199, !dbg !16
  %kdma.swizzled1876 = getelementptr i8, ptr addrspace(3) %address.local655, i64 %203, !dbg !16
  %204 = load <4 x half>, ptr addrspace(3) %kdma.swizzled1876, align 8, !dbg !16, !frisk.k_lds_pair !38
  %kdma.gep1880 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1876, i64 128, !dbg !16
  %205 = load <4 x half>, ptr addrspace(3) %kdma.gep1880, align 8, !dbg !16, !frisk.k_lds_pair !38
  %kdma.gep1877 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1876, i64 4096, !dbg !16
  %206 = load <4 x half>, ptr addrspace(3) %kdma.gep1877, align 8, !dbg !16, !frisk.k_lds_pair !39
  %kdma.gep1881 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1876, i64 4224, !dbg !16
  %207 = load <4 x half>, ptr addrspace(3) %kdma.gep1881, align 8, !dbg !16, !frisk.k_lds_pair !39
  %kdma.gep1878 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 128, !dbg !16
  %208 = load <4 x half>, ptr addrspace(3) %kdma.gep1878, align 8, !dbg !16, !frisk.k_lds_pair !40
  %kdma.gep1882 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1024, !dbg !16
  %209 = load <4 x half>, ptr addrspace(3) %kdma.gep1882, align 8, !dbg !16, !frisk.k_lds_pair !40
  %kdma.gep1879 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4224, !dbg !16
  %210 = load <4 x half>, ptr addrspace(3) %kdma.gep1879, align 8, !dbg !16, !frisk.k_lds_pair !41
  %kdma.gep1883 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5120, !dbg !16
  %211 = load <4 x half>, ptr addrspace(3) %kdma.gep1883, align 8, !dbg !16, !frisk.k_lds_pair !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16, !frisk.k_lds_prefetch !1
  %212 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %193, <4 x float> %192) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16, !frisk.k_lds_prefetch !1
  %kdma.gep1884 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1876, i64 1024, !dbg !16
  %213 = load <4 x half>, ptr addrspace(3) %kdma.gep1884, align 8, !dbg !16, !frisk.k_lds_pair !36
  %kdma.gep1888 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1876, i64 1152, !dbg !16
  %214 = load <4 x half>, ptr addrspace(3) %kdma.gep1888, align 8, !dbg !16, !frisk.k_lds_pair !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16, !frisk.k_lds_prefetch !1
  %215 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %204, <4 x float> zeroinitializer) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16, !frisk.k_lds_prefetch !1
  %kdma.gep1885 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1876, i64 5120, !dbg !16
  %216 = load <4 x half>, ptr addrspace(3) %kdma.gep1885, align 8, !dbg !16, !frisk.k_lds_pair !42
  %kdma.gep1889 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1876, i64 5248, !dbg !16
  %217 = load <4 x half>, ptr addrspace(3) %kdma.gep1889, align 8, !dbg !16, !frisk.k_lds_pair !42
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16, !frisk.k_lds_prefetch !1
  %218 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %206, <4 x float> %215) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16, !frisk.k_lds_prefetch !1
  %kdma.gep1886 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1152, !dbg !16
  %219 = load <4 x half>, ptr addrspace(3) %kdma.gep1886, align 8, !dbg !16, !frisk.k_lds_pair !43
  %kdma.gep1887 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5248, !dbg !16
  %220 = load <4 x half>, ptr addrspace(3) %kdma.gep1887, align 8, !dbg !16, !frisk.k_lds_pair !43
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16, !frisk.k_lds_prefetch !1
  %221 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %208, <4 x float> zeroinitializer) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16, !frisk.k_lds_prefetch !1
  %222 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %210, <4 x float> %221) #9, !dbg !16
  %223 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %205, <4 x float> zeroinitializer) #9, !dbg !16
  %224 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %207, <4 x float> %223) #9, !dbg !16
  %225 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %209, <4 x float> zeroinitializer) #9, !dbg !16
  %226 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %211, <4 x float> %225) #9, !dbg !16
  %227 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %213, <4 x float> zeroinitializer) #9, !dbg !16
  %228 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %216, <4 x float> %227) #9, !dbg !16
  %229 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %219, <4 x float> zeroinitializer) #9, !dbg !16
  %230 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %220, <4 x float> %229) #9, !dbg !16
  %231 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %214, <4 x float> zeroinitializer) #9, !dbg !16
  %232 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %217, <4 x float> %231) #9, !dbg !16
  %233 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %191, <4 x float> zeroinitializer) #9, !dbg !16
  %234 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %193, <4 x float> %233) #9, !dbg !16
  %235 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %204, <4 x float> zeroinitializer) #9, !dbg !16
  %236 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %206, <4 x float> %235) #9, !dbg !16
  %237 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %208, <4 x float> zeroinitializer) #9, !dbg !16
  %238 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %210, <4 x float> %237) #9, !dbg !16
  %239 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %205, <4 x float> zeroinitializer) #9, !dbg !16
  %240 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %207, <4 x float> %239) #9, !dbg !16
  %241 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %209, <4 x float> zeroinitializer) #9, !dbg !16
  %242 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %211, <4 x float> %241) #9, !dbg !16
  %243 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %213, <4 x float> zeroinitializer) #9, !dbg !16
  %244 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %216, <4 x float> %243) #9, !dbg !16
  %245 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %219, <4 x float> zeroinitializer) #9, !dbg !16
  %246 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %220, <4 x float> %245) #9, !dbg !16
  %247 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %214, <4 x float> zeroinitializer) #9, !dbg !16
  %248 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %217, <4 x float> %247) #9, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %249 = add i64 %149, 2048, !dbg !15
  %250 = icmp slt i64 %249, 0, !dbg !15
  %251 = sub i64 -2049, %149, !dbg !15
  %252 = select i1 %250, i64 %251, i64 %249, !dbg !15
  %253 = sdiv i64 %252, 4096, !dbg !15
  %.lobit38 = ashr i64 %249, 63, !dbg !15
  %254 = xor i64 %253, %.lobit38, !dbg !15
  %255 = add nsw i64 %156, 64, !dbg !15
  %256 = icmp slt i64 %156, -64, !dbg !15
  %257 = sub nsw i64 -65, %156, !dbg !15
  %258 = select i1 %256, i64 %257, i64 %255, !dbg !15
  %259 = sdiv i64 %258, 128, !dbg !15
  %260 = sext i1 %256 to i64, !dbg !15
  %261 = xor i64 %259, %260, !dbg !15
  %262 = shl nsw i64 %261, 7, !dbg !15
  %263 = sub i64 %102, %262, !dbg !15
  %264 = shl i64 %254, 19, !dbg !15
  %265 = and i64 %264, 16252928, !dbg !15
  %.idx14841488 = add nuw nsw i64 %265, %130, !dbg !15
  %266 = add i64 %.idx14841488, %263, !dbg !15
  %267 = add i64 %266, %137, !dbg !15
  %kdma.source1747 = add i64 %267, %kdma.source1612, !dbg !15
  %kdma.source1748 = trunc i64 %kdma.source1747 to i32, !dbg !15
  %kdma.source1749 = shl i32 %kdma.source1748, 1, !dbg !15
  %kdma.source1750 = add i32 %kdma.source1749, 128, !dbg !15
  %kdma.offset1751 = add i32 %kdma.source1750, %kdma.intra, !dbg !15
  %268 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %138, i32 %kdma.offset1751, <4 x i32> %dma.buffer.rsrc1430) #10, !dbg !15
  %kdma.source1764 = add i64 %267, %kdma.source1626, !dbg !15
  %kdma.source1765 = trunc i64 %kdma.source1764 to i32, !dbg !15
  %kdma.source1766 = shl i32 %kdma.source1765, 1, !dbg !15
  %kdma.source1767 = add i32 %kdma.source1766, 128, !dbg !15
  %kdma.offset1768 = add i32 %kdma.source1767, %kdma.intra.hi, !dbg !15
  %269 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1631, i32 %kdma.offset1768, <4 x i32> %dma.buffer.rsrc1430) #10, !dbg !15
  %kdma.source1782 = add i64 %267, %kdma.source1643, !dbg !15
  %kdma.source1783 = trunc i64 %kdma.source1782 to i32, !dbg !15
  %kdma.source1784 = shl i32 %kdma.source1783, 1, !dbg !15
  %kdma.source1785 = add i32 %kdma.source1784, 128, !dbg !15
  %kdma.offset1786 = add i32 %kdma.source1785, %kdma.intra, !dbg !15
  %270 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %145, i32 %kdma.offset1786, <4 x i32> %dma.buffer.rsrc1430) #10, !dbg !15
  %kdma.source1800 = add i64 %267, %kdma.source1660, !dbg !15
  %kdma.source1801 = trunc i64 %kdma.source1800 to i32, !dbg !15
  %kdma.source1802 = shl i32 %kdma.source1801, 1, !dbg !15
  %kdma.source1803 = add i32 %kdma.source1802, 128, !dbg !15
  %kdma.offset1804 = add i32 %kdma.source1803, %kdma.intra.hi, !dbg !15
  %271 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1665, i32 %kdma.offset1804, <4 x i32> %dma.buffer.rsrc1430) #10, !dbg !15
  %kdma.gep1906 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8192, !dbg !16
  %272 = load <4 x half>, ptr addrspace(3) %kdma.gep1906, align 8, !dbg !16, !frisk.k_lds_pair !44
  %kdma.gep1910 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8320, !dbg !16
  %273 = load <4 x half>, ptr addrspace(3) %kdma.gep1910, align 8, !dbg !16, !frisk.k_lds_pair !44
  %kdma.gep1907 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12288, !dbg !16
  %274 = load <4 x half>, ptr addrspace(3) %kdma.gep1907, align 8, !dbg !16, !frisk.k_lds_pair !45
  %kdma.gep1911 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12416, !dbg !16
  %275 = load <4 x half>, ptr addrspace(3) %kdma.gep1911, align 8, !dbg !16, !frisk.k_lds_pair !45
  %kdma.gep1908 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1876, i64 8192, !dbg !16
  %276 = load <4 x half>, ptr addrspace(3) %kdma.gep1908, align 8, !dbg !16, !frisk.k_lds_pair !46
  %kdma.gep1912 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1876, i64 8320, !dbg !16
  %277 = load <4 x half>, ptr addrspace(3) %kdma.gep1912, align 8, !dbg !16, !frisk.k_lds_pair !46
  %kdma.gep1909 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1876, i64 12288, !dbg !16
  %278 = load <4 x half>, ptr addrspace(3) %kdma.gep1909, align 8, !dbg !16, !frisk.k_lds_pair !47
  %kdma.gep1913 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1876, i64 12416, !dbg !16
  %279 = load <4 x half>, ptr addrspace(3) %kdma.gep1913, align 8, !dbg !16, !frisk.k_lds_pair !47
  %kdma.gep1914 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9216, !dbg !16
  %280 = load <4 x half>, ptr addrspace(3) %kdma.gep1914, align 8, !dbg !16, !frisk.k_lds_pair !48
  %kdma.gep1918 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9344, !dbg !16
  %281 = load <4 x half>, ptr addrspace(3) %kdma.gep1918, align 8, !dbg !16, !frisk.k_lds_pair !48
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16, !frisk.k_lds_prefetch !1
  %282 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %272, <4 x float> %212) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16, !frisk.k_lds_prefetch !1
  %kdma.gep1915 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13312, !dbg !16
  %283 = load <4 x half>, ptr addrspace(3) %kdma.gep1915, align 8, !dbg !16, !frisk.k_lds_pair !49
  %kdma.gep1919 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13440, !dbg !16
  %284 = load <4 x half>, ptr addrspace(3) %kdma.gep1919, align 8, !dbg !16, !frisk.k_lds_pair !49
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16, !frisk.k_lds_prefetch !1
  %285 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %274, <4 x float> %282) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16, !frisk.k_lds_prefetch !1
  %kdma.gep1916 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1876, i64 9216, !dbg !16
  %286 = load <4 x half>, ptr addrspace(3) %kdma.gep1916, align 8, !dbg !16, !frisk.k_lds_pair !50
  %kdma.gep1920 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1876, i64 9344, !dbg !16
  %287 = load <4 x half>, ptr addrspace(3) %kdma.gep1920, align 8, !dbg !16, !frisk.k_lds_pair !50
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16, !frisk.k_lds_prefetch !1
  %288 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %276, <4 x float> %218) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16, !frisk.k_lds_prefetch !1
  %kdma.gep1917 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1876, i64 13312, !dbg !16
  %289 = load <4 x half>, ptr addrspace(3) %kdma.gep1917, align 8, !dbg !16, !frisk.k_lds_pair !51
  %kdma.gep1921 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1876, i64 13440, !dbg !16
  %290 = load <4 x half>, ptr addrspace(3) %kdma.gep1921, align 8, !dbg !16, !frisk.k_lds_pair !51
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16, !frisk.k_lds_prefetch !1
  %291 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %278, <4 x float> %288) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16, !frisk.k_lds_prefetch !1
  %292 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %273, <4 x float> %222) #9, !dbg !16
  %293 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %275, <4 x float> %292) #9, !dbg !16
  %294 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %277, <4 x float> %224) #9, !dbg !16
  %295 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %279, <4 x float> %294) #9, !dbg !16
  %296 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %280, <4 x float> %226) #9, !dbg !16
  %297 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %283, <4 x float> %296) #9, !dbg !16
  %298 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %286, <4 x float> %228) #9, !dbg !16
  %299 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %289, <4 x float> %298) #9, !dbg !16
  %300 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %281, <4 x float> %230) #9, !dbg !16
  %301 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %284, <4 x float> %300) #9, !dbg !16
  %302 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %287, <4 x float> %232) #9, !dbg !16
  %303 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %290, <4 x float> %302) #9, !dbg !16
  %304 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %272, <4 x float> %234) #9, !dbg !16
  %305 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %274, <4 x float> %304) #9, !dbg !16
  %306 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %276, <4 x float> %236) #9, !dbg !16
  %307 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %278, <4 x float> %306) #9, !dbg !16
  %308 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %273, <4 x float> %238) #9, !dbg !16
  %309 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %275, <4 x float> %308) #9, !dbg !16
  %310 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %277, <4 x float> %240) #9, !dbg !16
  %311 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %279, <4 x float> %310) #9, !dbg !16
  %312 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %280, <4 x float> %242) #9, !dbg !16
  %313 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %283, <4 x float> %312) #9, !dbg !16
  %314 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %286, <4 x float> %244) #9, !dbg !16
  %315 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %289, <4 x float> %314) #9, !dbg !16
  %316 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %281, <4 x float> %246) #9, !dbg !16
  %317 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %284, <4 x float> %316) #9, !dbg !16
  %318 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %287, <4 x float> %248) #9, !dbg !16
  %319 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %290, <4 x float> %318) #9, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %320 = add i64 %149, 3072, !dbg !15
  %321 = icmp slt i64 %320, 0, !dbg !15
  %322 = sub i64 -3073, %149, !dbg !15
  %323 = select i1 %321, i64 %322, i64 %320, !dbg !15
  %324 = sdiv i64 %323, 4096, !dbg !15
  %.lobit39 = ashr i64 %320, 63, !dbg !15
  %325 = xor i64 %324, %.lobit39, !dbg !15
  %326 = add nsw i64 %156, 96, !dbg !15
  %327 = icmp slt i64 %156, -96, !dbg !15
  %328 = sub nsw i64 -97, %156, !dbg !15
  %329 = select i1 %327, i64 %328, i64 %326, !dbg !15
  %330 = sdiv i64 %329, 128, !dbg !15
  %331 = sext i1 %327 to i64, !dbg !15
  %332 = xor i64 %330, %331, !dbg !15
  %333 = shl nsw i64 %332, 7, !dbg !15
  %334 = sub i64 %102, %333, !dbg !15
  %335 = shl i64 %325, 19, !dbg !15
  %336 = and i64 %335, 16252928, !dbg !15
  %.idx14961500 = add nuw nsw i64 %336, %130, !dbg !15
  %337 = add i64 %.idx14961500, %334, !dbg !15
  %338 = add i64 %337, %137, !dbg !15
  %kdma.source1817 = add i64 %338, %kdma.source1612, !dbg !15
  %kdma.source1818 = trunc i64 %kdma.source1817 to i32, !dbg !15
  %kdma.source1819 = shl i32 %kdma.source1818, 1, !dbg !15
  %kdma.source1820 = add i32 %kdma.source1819, 192, !dbg !15
  %kdma.offset1821 = add i32 %kdma.source1820, %kdma.intra, !dbg !15
  %339 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %168, i32 %kdma.offset1821, <4 x i32> %dma.buffer.rsrc1430) #10, !dbg !15
  %kdma.source1834 = add i64 %338, %kdma.source1626, !dbg !15
  %kdma.source1835 = trunc i64 %kdma.source1834 to i32, !dbg !15
  %kdma.source1836 = shl i32 %kdma.source1835, 1, !dbg !15
  %kdma.source1837 = add i32 %kdma.source1836, 192, !dbg !15
  %kdma.offset1838 = add i32 %kdma.source1837, %kdma.intra.hi, !dbg !15
  %340 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1699, i32 %kdma.offset1838, <4 x i32> %dma.buffer.rsrc1430) #10, !dbg !15
  %kdma.source1852 = add i64 %338, %kdma.source1643, !dbg !15
  %kdma.source1853 = trunc i64 %kdma.source1852 to i32, !dbg !15
  %kdma.source1854 = shl i32 %kdma.source1853, 1, !dbg !15
  %kdma.source1855 = add i32 %kdma.source1854, 192, !dbg !15
  %kdma.offset1856 = add i32 %kdma.source1855, %kdma.intra, !dbg !15
  %341 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %173, i32 %kdma.offset1856, <4 x i32> %dma.buffer.rsrc1430) #10, !dbg !15
  %kdma.source1870 = add i64 %338, %kdma.source1660, !dbg !15
  %kdma.source1871 = trunc i64 %kdma.source1870 to i32, !dbg !15
  %kdma.source1872 = shl i32 %kdma.source1871, 1, !dbg !15
  %kdma.source1873 = add i32 %kdma.source1872, 192, !dbg !15
  %kdma.offset1874 = add i32 %kdma.source1873, %kdma.intra.hi, !dbg !15
  %342 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1735, i32 %kdma.offset1874, <4 x i32> %dma.buffer.rsrc1430) #10, !dbg !15
  %343 = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8, !dbg !16, !frisk.k_lds_pair !52
  %344 = load <4 x half>, ptr addrspace(3) %kdma.gep1878, align 8, !dbg !16, !frisk.k_lds_pair !52
  %345 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8, !dbg !16, !frisk.k_lds_pair !53
  %346 = load <4 x half>, ptr addrspace(3) %kdma.gep1879, align 8, !dbg !16, !frisk.k_lds_pair !53
  %347 = load <4 x half>, ptr addrspace(3) %kdma.swizzled1876, align 8, !dbg !16, !frisk.k_lds_pair !54
  %348 = load <4 x half>, ptr addrspace(3) %kdma.gep1880, align 8, !dbg !16, !frisk.k_lds_pair !54
  %349 = load <4 x half>, ptr addrspace(3) %kdma.gep1877, align 8, !dbg !16, !frisk.k_lds_pair !55
  %350 = load <4 x half>, ptr addrspace(3) %kdma.gep1881, align 8, !dbg !16, !frisk.k_lds_pair !55
  %351 = load <4 x half>, ptr addrspace(3) %kdma.gep1882, align 8, !dbg !16, !frisk.k_lds_pair !56
  %352 = load <4 x half>, ptr addrspace(3) %kdma.gep1886, align 8, !dbg !16, !frisk.k_lds_pair !56
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16, !frisk.k_lds_prefetch !1
  %353 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %343, <4 x float> %285) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16, !frisk.k_lds_prefetch !1
  %354 = load <4 x half>, ptr addrspace(3) %kdma.gep1883, align 8, !dbg !16, !frisk.k_lds_pair !57
  %355 = load <4 x half>, ptr addrspace(3) %kdma.gep1887, align 8, !dbg !16, !frisk.k_lds_pair !57
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16, !frisk.k_lds_prefetch !1
  %356 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %345, <4 x float> %353) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16, !frisk.k_lds_prefetch !1
  %357 = load <4 x half>, ptr addrspace(3) %kdma.gep1884, align 8, !dbg !16, !frisk.k_lds_pair !58
  %358 = load <4 x half>, ptr addrspace(3) %kdma.gep1888, align 8, !dbg !16, !frisk.k_lds_pair !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16, !frisk.k_lds_prefetch !1
  %359 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %347, <4 x float> %291) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16, !frisk.k_lds_prefetch !1
  %360 = load <4 x half>, ptr addrspace(3) %kdma.gep1885, align 8, !dbg !16, !frisk.k_lds_pair !59
  %361 = load <4 x half>, ptr addrspace(3) %kdma.gep1889, align 8, !dbg !16, !frisk.k_lds_pair !59
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16, !frisk.k_lds_prefetch !1
  %362 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %349, <4 x float> %359) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16, !frisk.k_lds_prefetch !1
  %363 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %344, <4 x float> %293) #9, !dbg !16
  %364 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %346, <4 x float> %363) #9, !dbg !16
  %365 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %348, <4 x float> %295) #9, !dbg !16
  %366 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %350, <4 x float> %365) #9, !dbg !16
  %367 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %351, <4 x float> %297) #9, !dbg !16
  %368 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %354, <4 x float> %367) #9, !dbg !16
  %369 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %357, <4 x float> %299) #9, !dbg !16
  %370 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %360, <4 x float> %369) #9, !dbg !16
  %371 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %352, <4 x float> %301) #9, !dbg !16
  %372 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %355, <4 x float> %371) #9, !dbg !16
  %373 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %358, <4 x float> %303) #9, !dbg !16
  %374 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %361, <4 x float> %373) #9, !dbg !16
  %375 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %343, <4 x float> %305) #9, !dbg !16
  %376 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %345, <4 x float> %375) #9, !dbg !16
  %377 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %347, <4 x float> %307) #9, !dbg !16
  %378 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %349, <4 x float> %377) #9, !dbg !16
  %379 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %344, <4 x float> %309) #9, !dbg !16
  %380 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %346, <4 x float> %379) #9, !dbg !16
  %381 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %348, <4 x float> %311) #9, !dbg !16
  %382 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %350, <4 x float> %381) #9, !dbg !16
  %383 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %351, <4 x float> %313) #9, !dbg !16
  %384 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %354, <4 x float> %383) #9, !dbg !16
  %385 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %357, <4 x float> %315) #9, !dbg !16
  %386 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %360, <4 x float> %385) #9, !dbg !16
  %387 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %352, <4 x float> %317) #9, !dbg !16
  %388 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %355, <4 x float> %387) #9, !dbg !16
  %389 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %358, <4 x float> %319) #9, !dbg !16
  %390 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %361, <4 x float> %389) #9, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %391 = add nuw nsw i64 %130, %18, !dbg !17
  %392 = shl i64 %391, 19, !dbg !17
  %393 = and i64 %392, 16252928, !dbg !17
  %address.local984 = shl nuw nsw i64 %address.local566, 5, !dbg !17
  %address.local985 = and i64 %address.local984, 8064, !dbg !17
  %address.local987 = and i64 %address.local567, 24, !dbg !17
  %address.local988 = or disjoint i64 %address.local985, %address.local987, !dbg !17
  %394 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %.idx15081510 = add i64 %393, %137, !dbg !17
  %395 = or disjoint i64 %.idx15081510, %address.local988, !dbg !17
  %.tr1511 = trunc i64 %395 to i32, !dbg !17
  %dma.buffer.offset1413 = shl i32 %.tr1511, 1, !dbg !17
  %396 = insertelement <2 x i64> poison, i64 %394, i64 0, !dbg !17
  %397 = bitcast <2 x i64> %396 to <4 x i32>, !dbg !17
  %dma.buffer.rsrc1414 = shufflevector <4 x i32> %397, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !17
  %398 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %138, i32 %dma.buffer.offset1413, <4 x i32> %dma.buffer.rsrc1414) #10, !dbg !17
  %address.local990 = shl nuw nsw i64 %address.local583, 5, !dbg !17
  %address.local991 = and i64 %address.local990, 16256, !dbg !17
  %address.local994 = or disjoint i64 %address.local991, %address.local987, !dbg !17
  %399 = or disjoint i64 %.idx15081510, %address.local994, !dbg !17
  %.tr1515 = trunc i64 %399 to i32, !dbg !17
  %dma.buffer.offset1411 = shl i32 %.tr1515, 1, !dbg !17
  %400 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %145, i32 %dma.buffer.offset1411, <4 x i32> %dma.buffer.rsrc1414) #10, !dbg !17
  %401 = load <4 x half>, ptr addrspace(3) %kdma.gep1906, align 8, !dbg !16, !frisk.k_tail_pair !38
  %402 = load <4 x half>, ptr addrspace(3) %kdma.gep1910, align 8, !dbg !16, !frisk.k_tail_pair !38
  %403 = load <4 x half>, ptr addrspace(3) %kdma.gep1907, align 8, !dbg !16, !frisk.k_tail_pair !39
  %404 = load <4 x half>, ptr addrspace(3) %kdma.gep1911, align 8, !dbg !16, !frisk.k_tail_pair !39
  %405 = load <4 x half>, ptr addrspace(3) %kdma.gep1908, align 8, !dbg !16, !frisk.k_tail_pair !40
  %406 = load <4 x half>, ptr addrspace(3) %kdma.gep1912, align 8, !dbg !16, !frisk.k_tail_pair !40
  %407 = load <4 x half>, ptr addrspace(3) %kdma.gep1909, align 8, !dbg !16, !frisk.k_tail_pair !41
  %408 = load <4 x half>, ptr addrspace(3) %kdma.gep1913, align 8, !dbg !16, !frisk.k_tail_pair !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16, !frisk.k_tail_prefetch !1
  %409 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %401, <4 x float> %356) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !18, !frisk.k_tail_prefetch !1
  %410 = or disjoint i64 %132, %114, !dbg !18
  %411 = trunc i64 %410 to i32, !dbg !18
  %.not40 = icmp ugt i32 %116, %411, !dbg !18
  %412 = or disjoint i32 %411, 1, !dbg !18
  %.not41 = icmp ugt i32 %116, %412, !dbg !18
  %413 = or disjoint i32 %411, 2, !dbg !18
  %.not42 = icmp ugt i32 %116, %413, !dbg !18
  %414 = or disjoint i32 %411, 3, !dbg !18
  %.not43 = icmp ugt i32 %116, %414, !dbg !18
  %415 = or disjoint i32 %411, 4, !dbg !18
  %.not44 = icmp ugt i32 %116, %415, !dbg !18
  %416 = or disjoint i32 %411, 5, !dbg !18
  %.not45 = icmp ugt i32 %116, %416, !dbg !18
  %417 = or disjoint i32 %411, 6, !dbg !18
  %.not46 = icmp ugt i32 %116, %417, !dbg !18
  %418 = or disjoint i32 %411, 7, !dbg !18
  %.not47 = icmp ugt i32 %116, %418, !dbg !18
  %419 = or disjoint i32 %411, 8, !dbg !18
  %.not48 = icmp ugt i32 %116, %419, !dbg !18
  %420 = or disjoint i32 %411, 9, !dbg !18
  %.not49 = icmp ugt i32 %116, %420, !dbg !18
  %421 = or disjoint i32 %411, 10, !dbg !18
  %.not50 = icmp ugt i32 %116, %421, !dbg !18
  %422 = or disjoint i32 %411, 11, !dbg !18
  %.not51 = icmp ugt i32 %116, %422, !dbg !18
  %423 = or disjoint i32 %411, 12, !dbg !18
  %.not52 = icmp ugt i32 %116, %423, !dbg !18
  %424 = or disjoint i32 %411, 13, !dbg !18
  %.not53 = icmp ugt i32 %116, %424, !dbg !18
  %425 = or disjoint i32 %411, 14, !dbg !18
  %.not54 = icmp ugt i32 %116, %425, !dbg !18
  %426 = or disjoint i32 %411, 15, !dbg !18
  %.not55 = icmp ugt i32 %116, %426, !dbg !18
  %427 = or disjoint i32 %411, 64, !dbg !18
  %.not56 = icmp ugt i32 %116, %427, !dbg !18
  %428 = or disjoint i32 %411, 65, !dbg !18
  %.not57 = icmp ugt i32 %116, %428, !dbg !18
  %429 = or disjoint i32 %411, 66, !dbg !18
  %.not58 = icmp ugt i32 %116, %429, !dbg !18
  %430 = or disjoint i32 %411, 67, !dbg !18
  %.not59 = icmp ugt i32 %116, %430, !dbg !18
  %431 = or disjoint i32 %411, 68, !dbg !18
  %.not60 = icmp ugt i32 %116, %431, !dbg !18
  %432 = or disjoint i32 %411, 69, !dbg !18
  %.not61 = icmp ugt i32 %116, %432, !dbg !18
  %433 = or disjoint i32 %411, 70, !dbg !18
  %.not62 = icmp ugt i32 %116, %433, !dbg !18
  %434 = or disjoint i32 %411, 71, !dbg !18
  %.not63 = icmp ugt i32 %116, %434, !dbg !18
  %435 = or disjoint i32 %411, 72, !dbg !18
  %.not64 = icmp ugt i32 %116, %435, !dbg !18
  %436 = or disjoint i32 %411, 73, !dbg !18
  %.not65 = icmp ugt i32 %116, %436, !dbg !18
  %437 = or disjoint i32 %411, 74, !dbg !18
  %.not66 = icmp ugt i32 %116, %437, !dbg !18
  %438 = or disjoint i32 %411, 75, !dbg !18
  %.not67 = icmp ugt i32 %116, %438, !dbg !18
  %439 = or disjoint i32 %411, 76, !dbg !18
  %.not68 = icmp ugt i32 %116, %439, !dbg !18
  %440 = or disjoint i32 %411, 77, !dbg !18
  %.not69 = icmp ugt i32 %116, %440, !dbg !18
  %441 = or disjoint i32 %411, 78, !dbg !18
  %.not70 = icmp ugt i32 %116, %441, !dbg !18
  %442 = or disjoint i32 %411, 79, !dbg !18
  %.not71 = icmp ugt i32 %116, %442, !dbg !18
  %.not72 = icmp ugt i32 %117, %411, !dbg !18
  %.not73 = icmp ugt i32 %117, %412, !dbg !18
  %.not74 = icmp ugt i32 %117, %413, !dbg !18
  %.not75 = icmp ugt i32 %117, %414, !dbg !18
  %.not76 = icmp ugt i32 %117, %415, !dbg !18
  %.not77 = icmp ugt i32 %117, %416, !dbg !18
  %.not78 = icmp ugt i32 %117, %417, !dbg !18
  %.not79 = icmp ugt i32 %117, %418, !dbg !18
  %.not80 = icmp ugt i32 %117, %419, !dbg !18
  %.not81 = icmp ugt i32 %117, %420, !dbg !18
  %.not82 = icmp ugt i32 %117, %421, !dbg !18
  %.not83 = icmp ugt i32 %117, %422, !dbg !18
  %.not84 = icmp ugt i32 %117, %423, !dbg !18
  %.not85 = icmp ugt i32 %117, %424, !dbg !18
  %.not86 = icmp ugt i32 %117, %425, !dbg !18
  %.not87 = icmp ugt i32 %117, %426, !dbg !18
  %.not88 = icmp ugt i32 %117, %427, !dbg !18
  %.not89 = icmp ugt i32 %117, %428, !dbg !18
  %.not90 = icmp ugt i32 %117, %429, !dbg !18
  %.not91 = icmp ugt i32 %117, %430, !dbg !18
  %.not92 = icmp ugt i32 %117, %431, !dbg !18
  %.not93 = icmp ugt i32 %117, %432, !dbg !18
  %.not94 = icmp ugt i32 %117, %433, !dbg !18
  %.not95 = icmp ugt i32 %117, %434, !dbg !18
  %.not96 = icmp ugt i32 %117, %435, !dbg !18
  %.not97 = icmp ugt i32 %117, %436, !dbg !18
  %.not98 = icmp ugt i32 %117, %437, !dbg !18
  %.not99 = icmp ugt i32 %117, %438, !dbg !18
  %.not100 = icmp ugt i32 %117, %439, !dbg !18
  %.not101 = icmp ugt i32 %117, %440, !dbg !18
  %.not102 = icmp ugt i32 %117, %441, !dbg !18
  %.not103 = icmp ugt i32 %117, %442, !dbg !18
  %443 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %403, <4 x float> %409) #9, !dbg !16
  %444 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %405, <4 x float> %362) #9, !dbg !16
  %445 = extractelement <4 x float> %443, i64 0, !dbg !22
  %446 = select i1 %.not40, float %445, float 0xFFF0000000000000, !dbg !18
  %447 = extractelement <4 x float> %443, i64 1, !dbg !22
  %448 = select i1 %.not41, float %447, float 0xFFF0000000000000, !dbg !18
  %449 = call float @llvm.maxnum.f32(float %446, float 0xFFF0000000000000), !dbg !23, !frisk.local_reduction !1
  %450 = extractelement <4 x float> %443, i64 2, !dbg !22
  %451 = select i1 %.not42, float %450, float 0xFFF0000000000000, !dbg !18
  %452 = extractelement <4 x float> %443, i64 3, !dbg !22
  %453 = select i1 %.not43, float %452, float 0xFFF0000000000000, !dbg !18
  %454 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %407, <4 x float> %444) #9, !dbg !16
  %455 = load <4 x half>, ptr addrspace(3) %kdma.gep1914, align 8, !dbg !16, !frisk.k_tail_pair !36
  %456 = load <4 x half>, ptr addrspace(3) %kdma.gep1918, align 8, !dbg !16, !frisk.k_tail_pair !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16, !frisk.k_tail_prefetch !1
  %457 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %402, <4 x float> %364) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !22, !frisk.k_tail_prefetch !1
  %458 = extractelement <4 x float> %454, i64 0, !dbg !22
  %459 = select i1 %.not44, float %458, float 0xFFF0000000000000, !dbg !18
  %460 = extractelement <4 x float> %454, i64 1, !dbg !22
  %461 = select i1 %.not45, float %460, float 0xFFF0000000000000, !dbg !18
  %local.reduce218 = call float @llvm.maxnum.f32(float %449, float %459), !dbg !23, !frisk.local_reduction !1
  %462 = extractelement <4 x float> %454, i64 2, !dbg !22
  %463 = select i1 %.not46, float %462, float 0xFFF0000000000000, !dbg !18
  %local.reduce219 = call float @llvm.maxnum.f32(float %448, float %461), !dbg !23, !frisk.local_reduction !1
  %464 = extractelement <4 x float> %454, i64 3, !dbg !22
  %465 = select i1 %.not47, float %464, float 0xFFF0000000000000, !dbg !18
  %local.reduce220 = call float @llvm.maxnum.f32(float %451, float %463), !dbg !23, !frisk.local_reduction !1
  %local.reduce221 = call float @llvm.maxnum.f32(float %453, float %465), !dbg !23, !frisk.local_reduction !1
  %466 = load <4 x half>, ptr addrspace(3) %kdma.gep1915, align 8, !dbg !16, !frisk.k_tail_pair !42
  %467 = load <4 x half>, ptr addrspace(3) %kdma.gep1919, align 8, !dbg !16, !frisk.k_tail_pair !42
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16, !frisk.k_tail_prefetch !1
  %468 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %404, <4 x float> %457) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16, !frisk.k_tail_prefetch !1
  %469 = load <4 x half>, ptr addrspace(3) %kdma.gep1916, align 8, !dbg !16, !frisk.k_tail_pair !43
  %470 = load <4 x half>, ptr addrspace(3) %kdma.gep1920, align 8, !dbg !16, !frisk.k_tail_pair !43
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16, !frisk.k_tail_prefetch !1
  %471 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %406, <4 x float> %366) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !22, !frisk.k_tail_prefetch !1
  %472 = extractelement <4 x float> %468, i64 0, !dbg !22
  %473 = select i1 %.not48, float %472, float 0xFFF0000000000000, !dbg !18
  %474 = extractelement <4 x float> %468, i64 1, !dbg !22
  %475 = select i1 %.not49, float %474, float 0xFFF0000000000000, !dbg !18
  %local.reduce222 = call float @llvm.maxnum.f32(float %local.reduce218, float %473), !dbg !23, !frisk.local_reduction !1
  %476 = extractelement <4 x float> %468, i64 2, !dbg !22
  %477 = select i1 %.not50, float %476, float 0xFFF0000000000000, !dbg !18
  %local.reduce223 = call float @llvm.maxnum.f32(float %local.reduce219, float %475), !dbg !23, !frisk.local_reduction !1
  %478 = extractelement <4 x float> %468, i64 3, !dbg !22
  %479 = select i1 %.not51, float %478, float 0xFFF0000000000000, !dbg !18
  %local.reduce224 = call float @llvm.maxnum.f32(float %local.reduce220, float %477), !dbg !23, !frisk.local_reduction !1
  %local.reduce225 = call float @llvm.maxnum.f32(float %local.reduce221, float %479), !dbg !23, !frisk.local_reduction !1
  %480 = load <4 x half>, ptr addrspace(3) %kdma.gep1917, align 8, !dbg !16, !frisk.k_tail_pair !44
  %481 = load <4 x half>, ptr addrspace(3) %kdma.gep1921, align 8, !dbg !16, !frisk.k_tail_pair !44
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16, !frisk.k_tail_prefetch !1
  %482 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %408, <4 x float> %471) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16, !frisk.k_tail_prefetch !1
  %483 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %455, <4 x float> %368) #9, !dbg !16
  %484 = extractelement <4 x float> %482, i64 0, !dbg !22
  %485 = select i1 %.not52, float %484, float 0xFFF0000000000000, !dbg !18
  %486 = extractelement <4 x float> %482, i64 1, !dbg !22
  %487 = select i1 %.not53, float %486, float 0xFFF0000000000000, !dbg !18
  %local.reduce226 = call float @llvm.maxnum.f32(float %local.reduce222, float %485), !dbg !23, !frisk.local_reduction !1
  %488 = extractelement <4 x float> %482, i64 2, !dbg !22
  %489 = select i1 %.not54, float %488, float 0xFFF0000000000000, !dbg !18
  %local.reduce227 = call float @llvm.maxnum.f32(float %local.reduce223, float %487), !dbg !23, !frisk.local_reduction !1
  %490 = extractelement <4 x float> %482, i64 3, !dbg !22
  %491 = select i1 %.not55, float %490, float 0xFFF0000000000000, !dbg !18
  %local.reduce228 = call float @llvm.maxnum.f32(float %local.reduce224, float %489), !dbg !23, !frisk.local_reduction !1
  %local.reduce229 = call float @llvm.maxnum.f32(float %local.reduce225, float %491), !dbg !23, !frisk.local_reduction !1
  %492 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %466, <4 x float> %483) #9, !dbg !16
  %493 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %469, <4 x float> %370) #9, !dbg !16
  %494 = extractelement <4 x float> %492, i64 0, !dbg !22
  %495 = select i1 %.not56, float %494, float 0xFFF0000000000000, !dbg !18
  %496 = extractelement <4 x float> %492, i64 1, !dbg !22
  %497 = select i1 %.not57, float %496, float 0xFFF0000000000000, !dbg !18
  %local.reduce230 = call float @llvm.maxnum.f32(float %local.reduce226, float %495), !dbg !23, !frisk.local_reduction !1
  %498 = extractelement <4 x float> %492, i64 2, !dbg !22
  %499 = select i1 %.not58, float %498, float 0xFFF0000000000000, !dbg !18
  %local.reduce231 = call float @llvm.maxnum.f32(float %local.reduce227, float %497), !dbg !23, !frisk.local_reduction !1
  %500 = extractelement <4 x float> %492, i64 3, !dbg !22
  %501 = select i1 %.not59, float %500, float 0xFFF0000000000000, !dbg !18
  %local.reduce232 = call float @llvm.maxnum.f32(float %local.reduce228, float %499), !dbg !23, !frisk.local_reduction !1
  %local.reduce233 = call float @llvm.maxnum.f32(float %local.reduce229, float %501), !dbg !23, !frisk.local_reduction !1
  %502 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %480, <4 x float> %493) #9, !dbg !16
  %503 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %456, <4 x float> %372) #9, !dbg !16
  %504 = extractelement <4 x float> %502, i64 0, !dbg !22
  %505 = select i1 %.not60, float %504, float 0xFFF0000000000000, !dbg !18
  %506 = extractelement <4 x float> %502, i64 1, !dbg !22
  %507 = select i1 %.not61, float %506, float 0xFFF0000000000000, !dbg !18
  %local.reduce234 = call float @llvm.maxnum.f32(float %local.reduce230, float %505), !dbg !23, !frisk.local_reduction !1
  %508 = extractelement <4 x float> %502, i64 2, !dbg !22
  %509 = select i1 %.not62, float %508, float 0xFFF0000000000000, !dbg !18
  %local.reduce235 = call float @llvm.maxnum.f32(float %local.reduce231, float %507), !dbg !23, !frisk.local_reduction !1
  %510 = extractelement <4 x float> %502, i64 3, !dbg !22
  %511 = select i1 %.not63, float %510, float 0xFFF0000000000000, !dbg !18
  %local.reduce236 = call float @llvm.maxnum.f32(float %local.reduce232, float %509), !dbg !23, !frisk.local_reduction !1
  %local.reduce237 = call float @llvm.maxnum.f32(float %local.reduce233, float %511), !dbg !23, !frisk.local_reduction !1
  %512 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %467, <4 x float> %503) #9, !dbg !16
  %513 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %470, <4 x float> %374) #9, !dbg !16
  %514 = extractelement <4 x float> %512, i64 0, !dbg !22
  %515 = select i1 %.not64, float %514, float 0xFFF0000000000000, !dbg !18
  %516 = extractelement <4 x float> %512, i64 1, !dbg !22
  %517 = select i1 %.not65, float %516, float 0xFFF0000000000000, !dbg !18
  %local.reduce238 = call float @llvm.maxnum.f32(float %local.reduce234, float %515), !dbg !23, !frisk.local_reduction !1
  %518 = extractelement <4 x float> %512, i64 2, !dbg !22
  %519 = select i1 %.not66, float %518, float 0xFFF0000000000000, !dbg !18
  %local.reduce239 = call float @llvm.maxnum.f32(float %local.reduce235, float %517), !dbg !23, !frisk.local_reduction !1
  %520 = extractelement <4 x float> %512, i64 3, !dbg !22
  %521 = select i1 %.not67, float %520, float 0xFFF0000000000000, !dbg !18
  %local.reduce240 = call float @llvm.maxnum.f32(float %local.reduce236, float %519), !dbg !23, !frisk.local_reduction !1
  %local.reduce241 = call float @llvm.maxnum.f32(float %local.reduce237, float %521), !dbg !23, !frisk.local_reduction !1
  %522 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %481, <4 x float> %513) #9, !dbg !16
  %523 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %401, <4 x float> %376) #9, !dbg !16
  %524 = extractelement <4 x float> %522, i64 0, !dbg !22
  %525 = select i1 %.not68, float %524, float 0xFFF0000000000000, !dbg !18
  %526 = extractelement <4 x float> %522, i64 1, !dbg !22
  %527 = select i1 %.not69, float %526, float 0xFFF0000000000000, !dbg !18
  %local.reduce242 = call float @llvm.maxnum.f32(float %local.reduce238, float %525), !dbg !23, !frisk.local_reduction !1
  %528 = extractelement <4 x float> %522, i64 2, !dbg !22
  %529 = select i1 %.not70, float %528, float 0xFFF0000000000000, !dbg !18
  %local.reduce243 = call float @llvm.maxnum.f32(float %local.reduce239, float %527), !dbg !23, !frisk.local_reduction !1
  %530 = extractelement <4 x float> %522, i64 3, !dbg !22
  %531 = select i1 %.not71, float %530, float 0xFFF0000000000000, !dbg !18
  %local.reduce244 = call float @llvm.maxnum.f32(float %local.reduce240, float %529), !dbg !23, !frisk.local_reduction !1
  %local.reduce245 = call float @llvm.maxnum.f32(float %local.reduce241, float %531), !dbg !23, !frisk.local_reduction !1
  %local.reduce246 = call float @llvm.maxnum.f32(float %local.reduce242, float %local.reduce243), !dbg !23, !frisk.local_reduction !1
  %local.reduce247 = call float @llvm.maxnum.f32(float %local.reduce244, float %local.reduce245), !dbg !23, !frisk.local_reduction !1
  %532 = call float @llvm.maxnum.f32(float %local.reduce246, float %local.reduce247), !dbg !23, !frisk.local_reduction !1
  %533 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %403, <4 x float> %523) #9, !dbg !16
  %534 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %405, <4 x float> %378) #9, !dbg !16
  %535 = extractelement <4 x float> %533, i64 0, !dbg !22
  %536 = select i1 %.not72, float %535, float 0xFFF0000000000000, !dbg !18
  %537 = extractelement <4 x float> %533, i64 1, !dbg !22
  %538 = select i1 %.not73, float %537, float 0xFFF0000000000000, !dbg !18
  %539 = extractelement <4 x float> %533, i64 2, !dbg !22
  %540 = select i1 %.not74, float %539, float 0xFFF0000000000000, !dbg !18
  %541 = extractelement <4 x float> %533, i64 3, !dbg !22
  %542 = select i1 %.not75, float %541, float 0xFFF0000000000000, !dbg !18
  %543 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %407, <4 x float> %534) #9, !dbg !16
  %544 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %402, <4 x float> %380) #9, !dbg !16
  %545 = extractelement <4 x float> %543, i64 0, !dbg !22
  %546 = select i1 %.not76, float %545, float 0xFFF0000000000000, !dbg !18
  %547 = extractelement <4 x float> %543, i64 1, !dbg !22
  %548 = select i1 %.not77, float %547, float 0xFFF0000000000000, !dbg !18
  %549 = extractelement <4 x float> %543, i64 2, !dbg !22
  %550 = select i1 %.not78, float %549, float 0xFFF0000000000000, !dbg !18
  %551 = extractelement <4 x float> %543, i64 3, !dbg !22
  %552 = select i1 %.not79, float %551, float 0xFFF0000000000000, !dbg !18
  %553 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %404, <4 x float> %544) #9, !dbg !16
  %554 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %406, <4 x float> %382) #9, !dbg !16
  %555 = extractelement <4 x float> %553, i64 0, !dbg !22
  %556 = select i1 %.not80, float %555, float 0xFFF0000000000000, !dbg !18
  %557 = extractelement <4 x float> %553, i64 1, !dbg !22
  %558 = select i1 %.not81, float %557, float 0xFFF0000000000000, !dbg !18
  %559 = extractelement <4 x float> %553, i64 2, !dbg !22
  %560 = select i1 %.not82, float %559, float 0xFFF0000000000000, !dbg !18
  %561 = extractelement <4 x float> %553, i64 3, !dbg !22
  %562 = select i1 %.not83, float %561, float 0xFFF0000000000000, !dbg !18
  %563 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %408, <4 x float> %554) #9, !dbg !16
  %564 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %455, <4 x float> %384) #9, !dbg !16
  %565 = extractelement <4 x float> %563, i64 0, !dbg !22
  %566 = select i1 %.not84, float %565, float 0xFFF0000000000000, !dbg !18
  %567 = extractelement <4 x float> %563, i64 1, !dbg !22
  %568 = select i1 %.not85, float %567, float 0xFFF0000000000000, !dbg !18
  %569 = extractelement <4 x float> %563, i64 2, !dbg !22
  %570 = select i1 %.not86, float %569, float 0xFFF0000000000000, !dbg !18
  %571 = extractelement <4 x float> %563, i64 3, !dbg !22
  %572 = select i1 %.not87, float %571, float 0xFFF0000000000000, !dbg !18
  %573 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %466, <4 x float> %564) #9, !dbg !16
  %574 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %469, <4 x float> %386) #9, !dbg !16
  %575 = extractelement <4 x float> %573, i64 0, !dbg !22
  %576 = select i1 %.not88, float %575, float 0xFFF0000000000000, !dbg !18
  %577 = extractelement <4 x float> %573, i64 1, !dbg !22
  %578 = select i1 %.not89, float %577, float 0xFFF0000000000000, !dbg !18
  %579 = extractelement <4 x float> %573, i64 2, !dbg !22
  %580 = select i1 %.not90, float %579, float 0xFFF0000000000000, !dbg !18
  %581 = extractelement <4 x float> %573, i64 3, !dbg !22
  %582 = select i1 %.not91, float %581, float 0xFFF0000000000000, !dbg !18
  %583 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %480, <4 x float> %574) #9, !dbg !16
  %584 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %456, <4 x float> %388) #9, !dbg !16
  %585 = extractelement <4 x float> %583, i64 0, !dbg !22
  %586 = select i1 %.not92, float %585, float 0xFFF0000000000000, !dbg !18
  %587 = extractelement <4 x float> %583, i64 1, !dbg !22
  %588 = select i1 %.not93, float %587, float 0xFFF0000000000000, !dbg !18
  %589 = extractelement <4 x float> %583, i64 2, !dbg !22
  %590 = select i1 %.not94, float %589, float 0xFFF0000000000000, !dbg !18
  %591 = extractelement <4 x float> %583, i64 3, !dbg !22
  %592 = select i1 %.not95, float %591, float 0xFFF0000000000000, !dbg !18
  %593 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %467, <4 x float> %584) #9, !dbg !16
  %594 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %470, <4 x float> %390) #9, !dbg !16
  %595 = extractelement <4 x float> %593, i64 0, !dbg !22
  %596 = select i1 %.not96, float %595, float 0xFFF0000000000000, !dbg !18
  %597 = extractelement <4 x float> %593, i64 1, !dbg !22
  %598 = select i1 %.not97, float %597, float 0xFFF0000000000000, !dbg !18
  %599 = extractelement <4 x float> %593, i64 2, !dbg !22
  %600 = select i1 %.not98, float %599, float 0xFFF0000000000000, !dbg !18
  %601 = extractelement <4 x float> %593, i64 3, !dbg !22
  %602 = select i1 %.not99, float %601, float 0xFFF0000000000000, !dbg !18
  %603 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %481, <4 x float> %594) #9, !dbg !16
  %604 = extractelement <4 x float> %603, i64 0, !dbg !22
  %605 = select i1 %.not100, float %604, float 0xFFF0000000000000, !dbg !18
  %606 = extractelement <4 x float> %603, i64 1, !dbg !22
  %607 = select i1 %.not101, float %606, float 0xFFF0000000000000, !dbg !18
  %608 = extractelement <4 x float> %603, i64 2, !dbg !22
  %609 = select i1 %.not102, float %608, float 0xFFF0000000000000, !dbg !18
  %610 = extractelement <4 x float> %603, i64 3, !dbg !22
  %611 = select i1 %.not103, float %610, float 0xFFF0000000000000, !dbg !18
  %612 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !23
  %613 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %612), !dbg !23
  %614 = add nuw nsw i32 %613, 64, !dbg !23
  %615 = xor i32 %613, 16, !dbg !23
  %616 = and i32 %614, 192, !dbg !23
  %617 = icmp samesign ult i32 %615, %616, !dbg !23
  %618 = select i1 %617, i32 %615, i32 %613, !dbg !23
  %619 = shl nuw nsw i32 %618, 2, !dbg !23
  %620 = bitcast float %532 to i32, !dbg !23
  %621 = call i32 @llvm.amdgcn.ds.bpermute(i32 %619, i32 %620), !dbg !23
  %622 = bitcast i32 %621 to float, !dbg !23
  %623 = call float @llvm.maxnum.f32(float %532, float %622), !dbg !23
  %624 = xor i32 %613, 32, !dbg !23
  %625 = icmp samesign ult i32 %624, %616, !dbg !23
  %626 = select i1 %625, i32 %624, i32 %613, !dbg !23
  %627 = shl nuw nsw i32 %626, 2, !dbg !23
  %628 = bitcast float %623 to i32, !dbg !23
  %629 = call i32 @llvm.amdgcn.ds.bpermute(i32 %627, i32 %628), !dbg !23
  %630 = bitcast i32 %629 to float, !dbg !23
  %631 = call float @llvm.maxnum.f32(float %623, float %630), !dbg !23
  %632 = call float @llvm.maxnum.f32(float %536, float 0xFFF0000000000000), !dbg !23, !frisk.local_reduction !1
  %local.reduce187 = call float @llvm.maxnum.f32(float %632, float %546), !dbg !23, !frisk.local_reduction !1
  %local.reduce188 = call float @llvm.maxnum.f32(float %538, float %548), !dbg !23, !frisk.local_reduction !1
  %local.reduce189 = call float @llvm.maxnum.f32(float %540, float %550), !dbg !23, !frisk.local_reduction !1
  %local.reduce190 = call float @llvm.maxnum.f32(float %542, float %552), !dbg !23, !frisk.local_reduction !1
  %local.reduce191 = call float @llvm.maxnum.f32(float %local.reduce187, float %556), !dbg !23, !frisk.local_reduction !1
  %local.reduce192 = call float @llvm.maxnum.f32(float %local.reduce188, float %558), !dbg !23, !frisk.local_reduction !1
  %local.reduce193 = call float @llvm.maxnum.f32(float %local.reduce189, float %560), !dbg !23, !frisk.local_reduction !1
  %local.reduce194 = call float @llvm.maxnum.f32(float %local.reduce190, float %562), !dbg !23, !frisk.local_reduction !1
  %local.reduce195 = call float @llvm.maxnum.f32(float %local.reduce191, float %566), !dbg !23, !frisk.local_reduction !1
  %local.reduce196 = call float @llvm.maxnum.f32(float %local.reduce192, float %568), !dbg !23, !frisk.local_reduction !1
  %local.reduce197 = call float @llvm.maxnum.f32(float %local.reduce193, float %570), !dbg !23, !frisk.local_reduction !1
  %local.reduce198 = call float @llvm.maxnum.f32(float %local.reduce194, float %572), !dbg !23, !frisk.local_reduction !1
  %local.reduce199 = call float @llvm.maxnum.f32(float %local.reduce195, float %576), !dbg !23, !frisk.local_reduction !1
  %local.reduce200 = call float @llvm.maxnum.f32(float %local.reduce196, float %578), !dbg !23, !frisk.local_reduction !1
  %local.reduce201 = call float @llvm.maxnum.f32(float %local.reduce197, float %580), !dbg !23, !frisk.local_reduction !1
  %local.reduce202 = call float @llvm.maxnum.f32(float %local.reduce198, float %582), !dbg !23, !frisk.local_reduction !1
  %local.reduce203 = call float @llvm.maxnum.f32(float %local.reduce199, float %586), !dbg !23, !frisk.local_reduction !1
  %local.reduce204 = call float @llvm.maxnum.f32(float %local.reduce200, float %588), !dbg !23, !frisk.local_reduction !1
  %local.reduce205 = call float @llvm.maxnum.f32(float %local.reduce201, float %590), !dbg !23, !frisk.local_reduction !1
  %local.reduce206 = call float @llvm.maxnum.f32(float %local.reduce202, float %592), !dbg !23, !frisk.local_reduction !1
  %local.reduce207 = call float @llvm.maxnum.f32(float %local.reduce203, float %596), !dbg !23, !frisk.local_reduction !1
  %local.reduce208 = call float @llvm.maxnum.f32(float %local.reduce204, float %598), !dbg !23, !frisk.local_reduction !1
  %local.reduce209 = call float @llvm.maxnum.f32(float %local.reduce205, float %600), !dbg !23, !frisk.local_reduction !1
  %local.reduce210 = call float @llvm.maxnum.f32(float %local.reduce206, float %602), !dbg !23, !frisk.local_reduction !1
  %local.reduce211 = call float @llvm.maxnum.f32(float %local.reduce207, float %605), !dbg !23, !frisk.local_reduction !1
  %local.reduce212 = call float @llvm.maxnum.f32(float %local.reduce208, float %607), !dbg !23, !frisk.local_reduction !1
  %local.reduce213 = call float @llvm.maxnum.f32(float %local.reduce209, float %609), !dbg !23, !frisk.local_reduction !1
  %local.reduce214 = call float @llvm.maxnum.f32(float %local.reduce210, float %611), !dbg !23, !frisk.local_reduction !1
  %local.reduce215 = call float @llvm.maxnum.f32(float %local.reduce211, float %local.reduce212), !dbg !23, !frisk.local_reduction !1
  %local.reduce216 = call float @llvm.maxnum.f32(float %local.reduce213, float %local.reduce214), !dbg !23, !frisk.local_reduction !1
  %633 = call float @llvm.maxnum.f32(float %local.reduce215, float %local.reduce216), !dbg !23, !frisk.local_reduction !1
  %634 = bitcast float %633 to i32, !dbg !23
  %635 = call i32 @llvm.amdgcn.ds.bpermute(i32 %619, i32 %634), !dbg !23
  %636 = bitcast i32 %635 to float, !dbg !23
  %637 = call float @llvm.maxnum.f32(float %633, float %636), !dbg !23
  %638 = bitcast float %637 to i32, !dbg !23
  %639 = call i32 @llvm.amdgcn.ds.bpermute(i32 %627, i32 %638), !dbg !23
  %640 = bitcast i32 %639 to float, !dbg !23
  %641 = call float @llvm.maxnum.f32(float %637, float %640), !dbg !23
  %642 = extractvalue [2 x <1 x float>] %122, 0, !dbg !23
  %643 = insertelement <1 x float> poison, float %631, i64 0, !dbg !23
  %644 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %643, <1 x float> %642), !dbg !23
  %645 = extractvalue [2 x <1 x float>] %122, 1, !dbg !23
  %646 = insertelement <1 x float> poison, float %641, i64 0, !dbg !23
  %647 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %646, <1 x float> %645), !dbg !23
  %648 = insertelement <4 x float> poison, float %446, i64 0, !dbg !24
  %649 = insertelement <4 x float> %648, float %448, i64 1, !dbg !24
  %650 = insertelement <4 x float> %649, float %451, i64 2, !dbg !24
  %651 = insertelement <4 x float> %650, float %453, i64 3, !dbg !24
  %652 = shufflevector <1 x float> %644, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !24
  %653 = insertelement <4 x float> poison, float %459, i64 0, !dbg !24
  %654 = insertelement <4 x float> %653, float %461, i64 1, !dbg !24
  %655 = insertelement <4 x float> %654, float %463, i64 2, !dbg !24
  %656 = insertelement <4 x float> %655, float %465, i64 3, !dbg !24
  %657 = insertelement <4 x float> poison, float %473, i64 0, !dbg !24
  %658 = insertelement <4 x float> %657, float %475, i64 1, !dbg !24
  %659 = insertelement <4 x float> %658, float %477, i64 2, !dbg !24
  %660 = insertelement <4 x float> %659, float %479, i64 3, !dbg !24
  %661 = insertelement <4 x float> poison, float %485, i64 0, !dbg !24
  %662 = insertelement <4 x float> %661, float %487, i64 1, !dbg !24
  %663 = insertelement <4 x float> %662, float %489, i64 2, !dbg !24
  %664 = insertelement <4 x float> %663, float %491, i64 3, !dbg !24
  %665 = insertelement <4 x float> poison, float %495, i64 0, !dbg !24
  %666 = insertelement <4 x float> %665, float %497, i64 1, !dbg !24
  %667 = insertelement <4 x float> %666, float %499, i64 2, !dbg !24
  %668 = insertelement <4 x float> %667, float %501, i64 3, !dbg !24
  %669 = insertelement <4 x float> poison, float %505, i64 0, !dbg !24
  %670 = insertelement <4 x float> %669, float %507, i64 1, !dbg !24
  %671 = insertelement <4 x float> %670, float %509, i64 2, !dbg !24
  %672 = insertelement <4 x float> %671, float %511, i64 3, !dbg !24
  %673 = insertelement <4 x float> poison, float %515, i64 0, !dbg !24
  %674 = insertelement <4 x float> %673, float %517, i64 1, !dbg !24
  %675 = insertelement <4 x float> %674, float %519, i64 2, !dbg !24
  %676 = insertelement <4 x float> %675, float %521, i64 3, !dbg !24
  %677 = insertelement <4 x float> poison, float %525, i64 0, !dbg !24
  %678 = insertelement <4 x float> %677, float %527, i64 1, !dbg !24
  %679 = insertelement <4 x float> %678, float %529, i64 2, !dbg !24
  %680 = insertelement <4 x float> %679, float %531, i64 3, !dbg !24
  %681 = insertelement <4 x float> poison, float %536, i64 0, !dbg !24
  %682 = insertelement <4 x float> %681, float %538, i64 1, !dbg !24
  %683 = insertelement <4 x float> %682, float %540, i64 2, !dbg !24
  %684 = insertelement <4 x float> %683, float %542, i64 3, !dbg !24
  %685 = shufflevector <1 x float> %647, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !24
  %686 = insertelement <4 x float> poison, float %546, i64 0, !dbg !24
  %687 = insertelement <4 x float> %686, float %548, i64 1, !dbg !24
  %688 = insertelement <4 x float> %687, float %550, i64 2, !dbg !24
  %689 = insertelement <4 x float> %688, float %552, i64 3, !dbg !24
  %690 = insertelement <4 x float> poison, float %556, i64 0, !dbg !24
  %691 = insertelement <4 x float> %690, float %558, i64 1, !dbg !24
  %692 = insertelement <4 x float> %691, float %560, i64 2, !dbg !24
  %693 = insertelement <4 x float> %692, float %562, i64 3, !dbg !24
  %694 = insertelement <4 x float> poison, float %566, i64 0, !dbg !24
  %695 = insertelement <4 x float> %694, float %568, i64 1, !dbg !24
  %696 = insertelement <4 x float> %695, float %570, i64 2, !dbg !24
  %697 = insertelement <4 x float> %696, float %572, i64 3, !dbg !24
  %698 = insertelement <4 x float> poison, float %576, i64 0, !dbg !24
  %699 = insertelement <4 x float> %698, float %578, i64 1, !dbg !24
  %700 = insertelement <4 x float> %699, float %580, i64 2, !dbg !24
  %701 = insertelement <4 x float> %700, float %582, i64 3, !dbg !24
  %702 = insertelement <4 x float> poison, float %586, i64 0, !dbg !24
  %703 = insertelement <4 x float> %702, float %588, i64 1, !dbg !24
  %704 = insertelement <4 x float> %703, float %590, i64 2, !dbg !24
  %705 = insertelement <4 x float> %704, float %592, i64 3, !dbg !24
  %706 = insertelement <4 x float> poison, float %596, i64 0, !dbg !24
  %707 = insertelement <4 x float> %706, float %598, i64 1, !dbg !24
  %708 = insertelement <4 x float> %707, float %600, i64 2, !dbg !24
  %709 = insertelement <4 x float> %708, float %602, i64 3, !dbg !24
  %710 = insertelement <4 x float> poison, float %605, i64 0, !dbg !24
  %711 = insertelement <4 x float> %710, float %607, i64 1, !dbg !24
  %712 = insertelement <4 x float> %711, float %609, i64 2, !dbg !24
  %713 = insertelement <4 x float> %712, float %611, i64 3, !dbg !24
  %714 = fsub <1 x float> %642, %644, !dbg !25
  %715 = extractelement <1 x float> %714, i64 0, !dbg !26
  %716 = call float @llvm.amdgcn.exp2.f32(float %715), !dbg !26
  %717 = insertelement <4 x float> poison, float %716, i64 0, !dbg !31
  %718 = shufflevector <4 x float> %717, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !31
  %719 = extractvalue [2 x <1 x float>] %121, 0, !dbg !27
  %720 = extractelement <1 x float> %719, i64 0, !dbg !27
  %721 = fmul float %720, %716, !dbg !27
  %722 = fsub <1 x float> %645, %647, !dbg !25
  %723 = extractelement <1 x float> %722, i64 0, !dbg !26
  %724 = call float @llvm.amdgcn.exp2.f32(float %723), !dbg !26
  %725 = insertelement <4 x float> poison, float %724, i64 0, !dbg !31
  %726 = shufflevector <4 x float> %725, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !31
  %727 = extractvalue [2 x <1 x float>] %121, 1, !dbg !27
  %728 = extractelement <1 x float> %727, i64 0, !dbg !27
  %729 = fmul float %728, %724, !dbg !27
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %vdb.address1532 = or disjoint i64 %.idx15081510, 32, !dbg !17
  %vdb.address1533 = or disjoint i64 %address.local988, %vdb.address1532, !dbg !17
  %vdb.address1534 = trunc i64 %vdb.address1533 to i32, !dbg !17
  %vdb.address1535 = shl i32 %vdb.address1534, 1, !dbg !17
  %vdb.address1542 = or disjoint i64 %address.local994, %vdb.address1532, !dbg !17
  %vdb.address1543 = trunc i64 %vdb.address1542 to i32, !dbg !17
  %vdb.address1544 = shl i32 %vdb.address1543, 1, !dbg !17
  %vdb.dma.base = add i32 %138, 8192, !dbg !13
  %730 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1535, <4 x i32> %dma.buffer.rsrc1414) #10, !dbg !17
  %vdb.dma.base1549 = add i32 %145, 8192, !dbg !13
  %731 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1549, i32 %vdb.address1544, <4 x i32> %dma.buffer.rsrc1414) #10, !dbg !17
  %address.local996 = shl nuw nsw i32 %address.thread565, 3, !dbg !13
  %address.local997 = and i32 %address.local996, 120, !dbg !13
  %address.local999 = and i32 %address.local642, 1536, !dbg !13
  %address.local1000 = or disjoint i32 %address.local997, %address.local999, !dbg !13
  %address.local1001 = zext nneg i32 %address.local1000 to i64, !dbg !13
  %address.local1002 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1001, !dbg !13
  %732 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1002, i32 0) #0, !dbg !13, !frisk.v_double_buffer !37
  %733 = bitcast <4 x i32> %732 to <8 x half>, !dbg !13
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
  %734 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1012, i32 0) #0, !dbg !13, !frisk.v_double_buffer !37
  %735 = bitcast <4 x i32> %734 to <8 x half>, !dbg !13
  %address.local1016 = getelementptr i8, ptr addrspace(3) %address.local1011, i64 512, !dbg !13
  %736 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1016, i32 0) #0, !dbg !13, !frisk.v_double_buffer !37
  %737 = bitcast <4 x i32> %736 to <8 x half>, !dbg !13
  %address.local1020 = getelementptr i8, ptr addrspace(3) %address.local1011, i64 768, !dbg !13
  %738 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1020, i32 0) #0, !dbg !13, !frisk.v_double_buffer !37
  %739 = bitcast <4 x i32> %738 to <8 x half>, !dbg !13
  %address.local1022 = getelementptr i8, ptr addrspace(3) %address.local1002, i64 4096, !dbg !13
  %740 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1022, i32 0) #0, !dbg !13, !frisk.v_double_buffer !37
  %741 = bitcast <4 x i32> %740 to <8 x half>, !dbg !13
  %address.local1026 = getelementptr i8, ptr addrspace(3) %address.local1011, i64 4352, !dbg !13
  %742 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1026, i32 0) #0, !dbg !13, !frisk.v_double_buffer !37
  %743 = bitcast <4 x i32> %742 to <8 x half>, !dbg !13
  %address.local1030 = getelementptr i8, ptr addrspace(3) %address.local1011, i64 4608, !dbg !13
  %744 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1030, i32 0) #0, !dbg !13, !frisk.v_double_buffer !37
  %745 = bitcast <4 x i32> %744 to <8 x half>, !dbg !13
  %address.local1034 = getelementptr i8, ptr addrspace(3) %address.local1011, i64 4864, !dbg !13
  %746 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1034, i32 0) #0, !dbg !13, !frisk.v_double_buffer !37
  %747 = shufflevector <8 x half> %733, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %748 = extractvalue [2 x <8 x float>] %120, 0, !dbg !31
  %749 = shufflevector <8 x float> %748, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %750 = fmul <4 x float> %749, %718, !dbg !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %751 = fsub <4 x float> %651, %652, !dbg !24
  %752 = extractelement <4 x float> %751, i64 0, !dbg !28
  %753 = call float @llvm.amdgcn.exp2.f32(float %752), !dbg !28
  %754 = insertelement <4 x float> poison, float %753, i64 0, !dbg !30
  %755 = extractelement <4 x float> %751, i64 1, !dbg !28
  %756 = call float @llvm.amdgcn.exp2.f32(float %755), !dbg !28
  %757 = insertelement <4 x float> %754, float %756, i64 1, !dbg !30
  %758 = extractelement <4 x float> %751, i64 2, !dbg !28
  %759 = call float @llvm.amdgcn.exp2.f32(float %758), !dbg !28
  %760 = insertelement <4 x float> %757, float %759, i64 2, !dbg !30
  %761 = extractelement <4 x float> %751, i64 3, !dbg !28
  %762 = call float @llvm.amdgcn.exp2.f32(float %761), !dbg !28
  %763 = insertelement <4 x float> %760, float %762, i64 3, !dbg !30
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %763) #9, !dbg !30, !frisk.register_lifetimes !1
  %.cvtpack.f0 = extractelement <4 x float> %math.result, i32 0, !dbg !30
  %.cvtpack.f1 = extractelement <4 x float> %math.result, i32 1, !dbg !30
  %.cvtpack.f2 = extractelement <4 x float> %math.result, i32 2, !dbg !30
  %.cvtpack.f3 = extractelement <4 x float> %math.result, i32 3, !dbg !30
  %.cvtpack.p0 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f0, float %.cvtpack.f1), !dbg !30
  %.cvtpack.p1 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f2, float %.cvtpack.f3), !dbg !30
  %.cvtpack.v0 = insertelement <2 x i32> poison, i32 %.cvtpack.p0, i32 0, !dbg !30
  %.cvtpack.v1 = insertelement <2 x i32> %.cvtpack.v0, i32 %.cvtpack.p1, i32 1, !dbg !30
  %764 = bitcast <2 x i32> %.cvtpack.v1 to <4 x half>, !dbg !30
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %764) #9, !dbg !28, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %765 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %747, <4 x float> %750) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %766 = shufflevector <8 x half> %733, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %767 = shufflevector <8 x float> %748, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %768 = fmul <4 x float> %767, %718, !dbg !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %769 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %766, <4 x float> %768) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !31
  %770 = extractvalue [2 x <8 x float>] %120, 1, !dbg !31
  %771 = shufflevector <8 x float> %770, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %772 = fmul <4 x float> %771, %726, !dbg !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %773 = fsub <4 x float> %684, %685, !dbg !24
  %774 = extractelement <4 x float> %773, i64 0, !dbg !28
  %775 = call float @llvm.amdgcn.exp2.f32(float %774), !dbg !28
  %776 = insertelement <4 x float> poison, float %775, i64 0, !dbg !30
  %777 = extractelement <4 x float> %773, i64 1, !dbg !28
  %778 = call float @llvm.amdgcn.exp2.f32(float %777), !dbg !28
  %779 = insertelement <4 x float> %776, float %778, i64 1, !dbg !30
  %780 = extractelement <4 x float> %773, i64 2, !dbg !28
  %781 = call float @llvm.amdgcn.exp2.f32(float %780), !dbg !28
  %782 = insertelement <4 x float> %779, float %781, i64 2, !dbg !30
  %783 = extractelement <4 x float> %773, i64 3, !dbg !28
  %784 = call float @llvm.amdgcn.exp2.f32(float %783), !dbg !28
  %785 = insertelement <4 x float> %782, float %784, i64 3, !dbg !30
  %math.result455 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %785) #9, !dbg !30, !frisk.register_lifetimes !1
  %.cvtpack.f02009 = extractelement <4 x float> %math.result455, i32 0, !dbg !30
  %.cvtpack.f12010 = extractelement <4 x float> %math.result455, i32 1, !dbg !30
  %.cvtpack.f22011 = extractelement <4 x float> %math.result455, i32 2, !dbg !30
  %.cvtpack.f32012 = extractelement <4 x float> %math.result455, i32 3, !dbg !30
  %.cvtpack.p02013 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02009, float %.cvtpack.f12010), !dbg !30
  %.cvtpack.p12014 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22011, float %.cvtpack.f32012), !dbg !30
  %.cvtpack.v02015 = insertelement <2 x i32> poison, i32 %.cvtpack.p02013, i32 0, !dbg !30
  %.cvtpack.v12016 = insertelement <2 x i32> %.cvtpack.v02015, i32 %.cvtpack.p12014, i32 1, !dbg !30
  %786 = bitcast <2 x i32> %.cvtpack.v12016 to <4 x half>, !dbg !30
  %math.packed456 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %786) #9, !dbg !28, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %787 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed456, <4 x half> %747, <4 x float> %772) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !31
  %788 = shufflevector <8 x float> %770, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %789 = fmul <4 x float> %788, %726, !dbg !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %790 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed456, <4 x half> %766, <4 x float> %789) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %791 = shufflevector <8 x half> %735, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %792 = fsub <4 x float> %656, %652, !dbg !24
  %793 = extractelement <4 x float> %792, i64 0, !dbg !28
  %794 = call float @llvm.amdgcn.exp2.f32(float %793), !dbg !28
  %local.reduce156 = fadd float %753, %794, !dbg !29, !frisk.local_reduction !1
  %795 = insertelement <4 x float> poison, float %794, i64 0, !dbg !30
  %796 = extractelement <4 x float> %792, i64 1, !dbg !28
  %797 = call float @llvm.amdgcn.exp2.f32(float %796), !dbg !28
  %local.reduce157 = fadd float %756, %797, !dbg !29, !frisk.local_reduction !1
  %798 = insertelement <4 x float> %795, float %797, i64 1, !dbg !30
  %799 = extractelement <4 x float> %792, i64 2, !dbg !28
  %800 = call float @llvm.amdgcn.exp2.f32(float %799), !dbg !28
  %local.reduce158 = fadd float %759, %800, !dbg !29, !frisk.local_reduction !1
  %801 = insertelement <4 x float> %798, float %800, i64 2, !dbg !30
  %802 = extractelement <4 x float> %792, i64 3, !dbg !28
  %803 = call float @llvm.amdgcn.exp2.f32(float %802), !dbg !28
  %local.reduce159 = fadd float %762, %803, !dbg !29, !frisk.local_reduction !1
  %804 = insertelement <4 x float> %801, float %803, i64 3, !dbg !30
  %math.result441 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %804) #9, !dbg !30, !frisk.register_lifetimes !1
  %.cvtpack.f02017 = extractelement <4 x float> %math.result441, i32 0, !dbg !30
  %.cvtpack.f12018 = extractelement <4 x float> %math.result441, i32 1, !dbg !30
  %.cvtpack.f22019 = extractelement <4 x float> %math.result441, i32 2, !dbg !30
  %.cvtpack.f32020 = extractelement <4 x float> %math.result441, i32 3, !dbg !30
  %.cvtpack.p02021 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02017, float %.cvtpack.f12018), !dbg !30
  %.cvtpack.p12022 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22019, float %.cvtpack.f32020), !dbg !30
  %.cvtpack.v02023 = insertelement <2 x i32> poison, i32 %.cvtpack.p02021, i32 0, !dbg !30
  %.cvtpack.v12024 = insertelement <2 x i32> %.cvtpack.v02023, i32 %.cvtpack.p12022, i32 1, !dbg !30
  %805 = bitcast <2 x i32> %.cvtpack.v12024 to <4 x half>, !dbg !30
  %math.packed442 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %805) #9, !dbg !29, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %806 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed442, <4 x half> %791, <4 x float> %765) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %807 = shufflevector <8 x half> %735, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %808 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed442, <4 x half> %807, <4 x float> %769) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %809 = fsub <4 x float> %689, %685, !dbg !24
  %810 = extractelement <4 x float> %809, i64 0, !dbg !28
  %811 = call float @llvm.amdgcn.exp2.f32(float %810), !dbg !28
  %local.reduce = fadd float %775, %811, !dbg !29, !frisk.local_reduction !1
  %812 = insertelement <4 x float> poison, float %811, i64 0, !dbg !30
  %813 = extractelement <4 x float> %809, i64 1, !dbg !28
  %814 = call float @llvm.amdgcn.exp2.f32(float %813), !dbg !28
  %local.reduce126 = fadd float %778, %814, !dbg !29, !frisk.local_reduction !1
  %815 = insertelement <4 x float> %812, float %814, i64 1, !dbg !30
  %816 = extractelement <4 x float> %809, i64 2, !dbg !28
  %817 = call float @llvm.amdgcn.exp2.f32(float %816), !dbg !28
  %local.reduce127 = fadd float %781, %817, !dbg !29, !frisk.local_reduction !1
  %818 = insertelement <4 x float> %815, float %817, i64 2, !dbg !30
  %819 = extractelement <4 x float> %809, i64 3, !dbg !28
  %820 = call float @llvm.amdgcn.exp2.f32(float %819), !dbg !28
  %local.reduce128 = fadd float %784, %820, !dbg !29, !frisk.local_reduction !1
  %821 = insertelement <4 x float> %818, float %820, i64 3, !dbg !30
  %math.result457 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %821) #9, !dbg !30, !frisk.register_lifetimes !1
  %.cvtpack.f02025 = extractelement <4 x float> %math.result457, i32 0, !dbg !30
  %.cvtpack.f12026 = extractelement <4 x float> %math.result457, i32 1, !dbg !30
  %.cvtpack.f22027 = extractelement <4 x float> %math.result457, i32 2, !dbg !30
  %.cvtpack.f32028 = extractelement <4 x float> %math.result457, i32 3, !dbg !30
  %.cvtpack.p02029 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02025, float %.cvtpack.f12026), !dbg !30
  %.cvtpack.p12030 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22027, float %.cvtpack.f32028), !dbg !30
  %.cvtpack.v02031 = insertelement <2 x i32> poison, i32 %.cvtpack.p02029, i32 0, !dbg !30
  %.cvtpack.v12032 = insertelement <2 x i32> %.cvtpack.v02031, i32 %.cvtpack.p12030, i32 1, !dbg !30
  %822 = bitcast <2 x i32> %.cvtpack.v12032 to <4 x half>, !dbg !30
  %math.packed458 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %822) #9, !dbg !29, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %823 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed458, <4 x half> %791, <4 x float> %787) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %824 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed458, <4 x half> %807, <4 x float> %790) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %825 = shufflevector <8 x half> %737, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %826 = fsub <4 x float> %660, %652, !dbg !24
  %827 = extractelement <4 x float> %826, i64 0, !dbg !28
  %828 = call float @llvm.amdgcn.exp2.f32(float %827), !dbg !28
  %local.reduce160 = fadd float %local.reduce156, %828, !dbg !29, !frisk.local_reduction !1
  %829 = insertelement <4 x float> poison, float %828, i64 0, !dbg !30
  %830 = extractelement <4 x float> %826, i64 1, !dbg !28
  %831 = call float @llvm.amdgcn.exp2.f32(float %830), !dbg !28
  %local.reduce161 = fadd float %local.reduce157, %831, !dbg !29, !frisk.local_reduction !1
  %832 = insertelement <4 x float> %829, float %831, i64 1, !dbg !30
  %833 = extractelement <4 x float> %826, i64 2, !dbg !28
  %834 = call float @llvm.amdgcn.exp2.f32(float %833), !dbg !28
  %local.reduce162 = fadd float %local.reduce158, %834, !dbg !29, !frisk.local_reduction !1
  %835 = insertelement <4 x float> %832, float %834, i64 2, !dbg !30
  %836 = extractelement <4 x float> %826, i64 3, !dbg !28
  %837 = call float @llvm.amdgcn.exp2.f32(float %836), !dbg !28
  %local.reduce163 = fadd float %local.reduce159, %837, !dbg !29, !frisk.local_reduction !1
  %838 = insertelement <4 x float> %835, float %837, i64 3, !dbg !30
  %math.result443 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %838) #9, !dbg !30, !frisk.register_lifetimes !1
  %.cvtpack.f02033 = extractelement <4 x float> %math.result443, i32 0, !dbg !30
  %.cvtpack.f12034 = extractelement <4 x float> %math.result443, i32 1, !dbg !30
  %.cvtpack.f22035 = extractelement <4 x float> %math.result443, i32 2, !dbg !30
  %.cvtpack.f32036 = extractelement <4 x float> %math.result443, i32 3, !dbg !30
  %.cvtpack.p02037 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02033, float %.cvtpack.f12034), !dbg !30
  %.cvtpack.p12038 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22035, float %.cvtpack.f32036), !dbg !30
  %.cvtpack.v02039 = insertelement <2 x i32> poison, i32 %.cvtpack.p02037, i32 0, !dbg !30
  %.cvtpack.v12040 = insertelement <2 x i32> %.cvtpack.v02039, i32 %.cvtpack.p12038, i32 1, !dbg !30
  %839 = bitcast <2 x i32> %.cvtpack.v12040 to <4 x half>, !dbg !30
  %math.packed444 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %839) #9, !dbg !29, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %840 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed444, <4 x half> %825, <4 x float> %806) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %841 = shufflevector <8 x half> %737, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %842 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed444, <4 x half> %841, <4 x float> %808) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %843 = fsub <4 x float> %693, %685, !dbg !24
  %844 = extractelement <4 x float> %843, i64 0, !dbg !28
  %845 = call float @llvm.amdgcn.exp2.f32(float %844), !dbg !28
  %local.reduce129 = fadd float %local.reduce, %845, !dbg !29, !frisk.local_reduction !1
  %846 = insertelement <4 x float> poison, float %845, i64 0, !dbg !30
  %847 = extractelement <4 x float> %843, i64 1, !dbg !28
  %848 = call float @llvm.amdgcn.exp2.f32(float %847), !dbg !28
  %local.reduce130 = fadd float %local.reduce126, %848, !dbg !29, !frisk.local_reduction !1
  %849 = insertelement <4 x float> %846, float %848, i64 1, !dbg !30
  %850 = extractelement <4 x float> %843, i64 2, !dbg !28
  %851 = call float @llvm.amdgcn.exp2.f32(float %850), !dbg !28
  %local.reduce131 = fadd float %local.reduce127, %851, !dbg !29, !frisk.local_reduction !1
  %852 = insertelement <4 x float> %849, float %851, i64 2, !dbg !30
  %853 = extractelement <4 x float> %843, i64 3, !dbg !28
  %854 = call float @llvm.amdgcn.exp2.f32(float %853), !dbg !28
  %local.reduce132 = fadd float %local.reduce128, %854, !dbg !29, !frisk.local_reduction !1
  %855 = insertelement <4 x float> %852, float %854, i64 3, !dbg !30
  %math.result459 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %855) #9, !dbg !30, !frisk.register_lifetimes !1
  %.cvtpack.f02041 = extractelement <4 x float> %math.result459, i32 0, !dbg !30
  %.cvtpack.f12042 = extractelement <4 x float> %math.result459, i32 1, !dbg !30
  %.cvtpack.f22043 = extractelement <4 x float> %math.result459, i32 2, !dbg !30
  %.cvtpack.f32044 = extractelement <4 x float> %math.result459, i32 3, !dbg !30
  %.cvtpack.p02045 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02041, float %.cvtpack.f12042), !dbg !30
  %.cvtpack.p12046 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22043, float %.cvtpack.f32044), !dbg !30
  %.cvtpack.v02047 = insertelement <2 x i32> poison, i32 %.cvtpack.p02045, i32 0, !dbg !30
  %.cvtpack.v12048 = insertelement <2 x i32> %.cvtpack.v02047, i32 %.cvtpack.p12046, i32 1, !dbg !30
  %856 = bitcast <2 x i32> %.cvtpack.v12048 to <4 x half>, !dbg !30
  %math.packed460 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %856) #9, !dbg !29, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %857 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed460, <4 x half> %825, <4 x float> %823) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %858 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed460, <4 x half> %841, <4 x float> %824) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %859 = shufflevector <8 x half> %739, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %860 = fsub <4 x float> %664, %652, !dbg !24
  %861 = extractelement <4 x float> %860, i64 0, !dbg !28
  %862 = call float @llvm.amdgcn.exp2.f32(float %861), !dbg !28
  %local.reduce164 = fadd float %local.reduce160, %862, !dbg !29, !frisk.local_reduction !1
  %863 = insertelement <4 x float> poison, float %862, i64 0, !dbg !30
  %864 = extractelement <4 x float> %860, i64 1, !dbg !28
  %865 = call float @llvm.amdgcn.exp2.f32(float %864), !dbg !28
  %local.reduce165 = fadd float %local.reduce161, %865, !dbg !29, !frisk.local_reduction !1
  %866 = insertelement <4 x float> %863, float %865, i64 1, !dbg !30
  %867 = extractelement <4 x float> %860, i64 2, !dbg !28
  %868 = call float @llvm.amdgcn.exp2.f32(float %867), !dbg !28
  %local.reduce166 = fadd float %local.reduce162, %868, !dbg !29, !frisk.local_reduction !1
  %869 = insertelement <4 x float> %866, float %868, i64 2, !dbg !30
  %870 = extractelement <4 x float> %860, i64 3, !dbg !28
  %871 = call float @llvm.amdgcn.exp2.f32(float %870), !dbg !28
  %local.reduce167 = fadd float %local.reduce163, %871, !dbg !29, !frisk.local_reduction !1
  %872 = insertelement <4 x float> %869, float %871, i64 3, !dbg !30
  %math.result445 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %872) #9, !dbg !30, !frisk.register_lifetimes !1
  %.cvtpack.f02049 = extractelement <4 x float> %math.result445, i32 0, !dbg !30
  %.cvtpack.f12050 = extractelement <4 x float> %math.result445, i32 1, !dbg !30
  %.cvtpack.f22051 = extractelement <4 x float> %math.result445, i32 2, !dbg !30
  %.cvtpack.f32052 = extractelement <4 x float> %math.result445, i32 3, !dbg !30
  %.cvtpack.p02053 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02049, float %.cvtpack.f12050), !dbg !30
  %.cvtpack.p12054 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22051, float %.cvtpack.f32052), !dbg !30
  %.cvtpack.v02055 = insertelement <2 x i32> poison, i32 %.cvtpack.p02053, i32 0, !dbg !30
  %.cvtpack.v12056 = insertelement <2 x i32> %.cvtpack.v02055, i32 %.cvtpack.p12054, i32 1, !dbg !30
  %873 = bitcast <2 x i32> %.cvtpack.v12056 to <4 x half>, !dbg !30
  %math.packed446 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %873) #9, !dbg !29, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %874 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed446, <4 x half> %859, <4 x float> %840) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %875 = shufflevector <8 x half> %739, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %876 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed446, <4 x half> %875, <4 x float> %842) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %877 = fsub <4 x float> %697, %685, !dbg !24
  %878 = extractelement <4 x float> %877, i64 0, !dbg !28
  %879 = call float @llvm.amdgcn.exp2.f32(float %878), !dbg !28
  %local.reduce133 = fadd float %local.reduce129, %879, !dbg !29, !frisk.local_reduction !1
  %880 = insertelement <4 x float> poison, float %879, i64 0, !dbg !30
  %881 = extractelement <4 x float> %877, i64 1, !dbg !28
  %882 = call float @llvm.amdgcn.exp2.f32(float %881), !dbg !28
  %local.reduce134 = fadd float %local.reduce130, %882, !dbg !29, !frisk.local_reduction !1
  %883 = insertelement <4 x float> %880, float %882, i64 1, !dbg !30
  %884 = extractelement <4 x float> %877, i64 2, !dbg !28
  %885 = call float @llvm.amdgcn.exp2.f32(float %884), !dbg !28
  %local.reduce135 = fadd float %local.reduce131, %885, !dbg !29, !frisk.local_reduction !1
  %886 = insertelement <4 x float> %883, float %885, i64 2, !dbg !30
  %887 = extractelement <4 x float> %877, i64 3, !dbg !28
  %888 = call float @llvm.amdgcn.exp2.f32(float %887), !dbg !28
  %local.reduce136 = fadd float %local.reduce132, %888, !dbg !29, !frisk.local_reduction !1
  %889 = insertelement <4 x float> %886, float %888, i64 3, !dbg !30
  %math.result461 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %889) #9, !dbg !30, !frisk.register_lifetimes !1
  %.cvtpack.f02057 = extractelement <4 x float> %math.result461, i32 0, !dbg !30
  %.cvtpack.f12058 = extractelement <4 x float> %math.result461, i32 1, !dbg !30
  %.cvtpack.f22059 = extractelement <4 x float> %math.result461, i32 2, !dbg !30
  %.cvtpack.f32060 = extractelement <4 x float> %math.result461, i32 3, !dbg !30
  %.cvtpack.p02061 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02057, float %.cvtpack.f12058), !dbg !30
  %.cvtpack.p12062 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22059, float %.cvtpack.f32060), !dbg !30
  %.cvtpack.v02063 = insertelement <2 x i32> poison, i32 %.cvtpack.p02061, i32 0, !dbg !30
  %.cvtpack.v12064 = insertelement <2 x i32> %.cvtpack.v02063, i32 %.cvtpack.p12062, i32 1, !dbg !30
  %890 = bitcast <2 x i32> %.cvtpack.v12064 to <4 x half>, !dbg !30
  %math.packed462 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %890) #9, !dbg !29, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %891 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed462, <4 x half> %859, <4 x float> %857) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %892 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed462, <4 x half> %875, <4 x float> %858) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %893 = shufflevector <8 x half> %741, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %894 = fsub <4 x float> %668, %652, !dbg !24
  %895 = extractelement <4 x float> %894, i64 0, !dbg !28
  %896 = call float @llvm.amdgcn.exp2.f32(float %895), !dbg !28
  %local.reduce168 = fadd float %local.reduce164, %896, !dbg !29, !frisk.local_reduction !1
  %897 = insertelement <4 x float> poison, float %896, i64 0, !dbg !30
  %898 = extractelement <4 x float> %894, i64 1, !dbg !28
  %899 = call float @llvm.amdgcn.exp2.f32(float %898), !dbg !28
  %local.reduce169 = fadd float %local.reduce165, %899, !dbg !29, !frisk.local_reduction !1
  %900 = insertelement <4 x float> %897, float %899, i64 1, !dbg !30
  %901 = extractelement <4 x float> %894, i64 2, !dbg !28
  %902 = call float @llvm.amdgcn.exp2.f32(float %901), !dbg !28
  %local.reduce170 = fadd float %local.reduce166, %902, !dbg !29, !frisk.local_reduction !1
  %903 = insertelement <4 x float> %900, float %902, i64 2, !dbg !30
  %904 = extractelement <4 x float> %894, i64 3, !dbg !28
  %905 = call float @llvm.amdgcn.exp2.f32(float %904), !dbg !28
  %local.reduce171 = fadd float %local.reduce167, %905, !dbg !29, !frisk.local_reduction !1
  %906 = insertelement <4 x float> %903, float %905, i64 3, !dbg !30
  %math.result447 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %906) #9, !dbg !30, !frisk.register_lifetimes !1
  %.cvtpack.f02065 = extractelement <4 x float> %math.result447, i32 0, !dbg !30
  %.cvtpack.f12066 = extractelement <4 x float> %math.result447, i32 1, !dbg !30
  %.cvtpack.f22067 = extractelement <4 x float> %math.result447, i32 2, !dbg !30
  %.cvtpack.f32068 = extractelement <4 x float> %math.result447, i32 3, !dbg !30
  %.cvtpack.p02069 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02065, float %.cvtpack.f12066), !dbg !30
  %.cvtpack.p12070 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22067, float %.cvtpack.f32068), !dbg !30
  %.cvtpack.v02071 = insertelement <2 x i32> poison, i32 %.cvtpack.p02069, i32 0, !dbg !30
  %.cvtpack.v12072 = insertelement <2 x i32> %.cvtpack.v02071, i32 %.cvtpack.p12070, i32 1, !dbg !30
  %907 = bitcast <2 x i32> %.cvtpack.v12072 to <4 x half>, !dbg !30
  %math.packed448 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %907) #9, !dbg !29, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %908 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed448, <4 x half> %893, <4 x float> %874) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %909 = shufflevector <8 x half> %741, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %910 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed448, <4 x half> %909, <4 x float> %876) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %911 = fsub <4 x float> %701, %685, !dbg !24
  %912 = extractelement <4 x float> %911, i64 0, !dbg !28
  %913 = call float @llvm.amdgcn.exp2.f32(float %912), !dbg !28
  %local.reduce137 = fadd float %local.reduce133, %913, !dbg !29, !frisk.local_reduction !1
  %914 = insertelement <4 x float> poison, float %913, i64 0, !dbg !30
  %915 = extractelement <4 x float> %911, i64 1, !dbg !28
  %916 = call float @llvm.amdgcn.exp2.f32(float %915), !dbg !28
  %local.reduce138 = fadd float %local.reduce134, %916, !dbg !29, !frisk.local_reduction !1
  %917 = insertelement <4 x float> %914, float %916, i64 1, !dbg !30
  %918 = extractelement <4 x float> %911, i64 2, !dbg !28
  %919 = call float @llvm.amdgcn.exp2.f32(float %918), !dbg !28
  %local.reduce139 = fadd float %local.reduce135, %919, !dbg !29, !frisk.local_reduction !1
  %920 = insertelement <4 x float> %917, float %919, i64 2, !dbg !30
  %921 = extractelement <4 x float> %911, i64 3, !dbg !28
  %922 = call float @llvm.amdgcn.exp2.f32(float %921), !dbg !28
  %local.reduce140 = fadd float %local.reduce136, %922, !dbg !29, !frisk.local_reduction !1
  %923 = insertelement <4 x float> %920, float %922, i64 3, !dbg !30
  %math.result463 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %923) #9, !dbg !30, !frisk.register_lifetimes !1
  %.cvtpack.f02073 = extractelement <4 x float> %math.result463, i32 0, !dbg !30
  %.cvtpack.f12074 = extractelement <4 x float> %math.result463, i32 1, !dbg !30
  %.cvtpack.f22075 = extractelement <4 x float> %math.result463, i32 2, !dbg !30
  %.cvtpack.f32076 = extractelement <4 x float> %math.result463, i32 3, !dbg !30
  %.cvtpack.p02077 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02073, float %.cvtpack.f12074), !dbg !30
  %.cvtpack.p12078 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22075, float %.cvtpack.f32076), !dbg !30
  %.cvtpack.v02079 = insertelement <2 x i32> poison, i32 %.cvtpack.p02077, i32 0, !dbg !30
  %.cvtpack.v12080 = insertelement <2 x i32> %.cvtpack.v02079, i32 %.cvtpack.p12078, i32 1, !dbg !30
  %924 = bitcast <2 x i32> %.cvtpack.v12080 to <4 x half>, !dbg !30
  %math.packed464 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %924) #9, !dbg !29, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %925 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed464, <4 x half> %893, <4 x float> %891) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %926 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed464, <4 x half> %909, <4 x float> %892) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %927 = shufflevector <8 x half> %743, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %928 = fsub <4 x float> %672, %652, !dbg !24
  %929 = extractelement <4 x float> %928, i64 0, !dbg !28
  %930 = call float @llvm.amdgcn.exp2.f32(float %929), !dbg !28
  %local.reduce172 = fadd float %local.reduce168, %930, !dbg !29, !frisk.local_reduction !1
  %931 = insertelement <4 x float> poison, float %930, i64 0, !dbg !30
  %932 = extractelement <4 x float> %928, i64 1, !dbg !28
  %933 = call float @llvm.amdgcn.exp2.f32(float %932), !dbg !28
  %local.reduce173 = fadd float %local.reduce169, %933, !dbg !29, !frisk.local_reduction !1
  %934 = insertelement <4 x float> %931, float %933, i64 1, !dbg !30
  %935 = extractelement <4 x float> %928, i64 2, !dbg !28
  %936 = call float @llvm.amdgcn.exp2.f32(float %935), !dbg !28
  %local.reduce174 = fadd float %local.reduce170, %936, !dbg !29, !frisk.local_reduction !1
  %937 = insertelement <4 x float> %934, float %936, i64 2, !dbg !30
  %938 = extractelement <4 x float> %928, i64 3, !dbg !28
  %939 = call float @llvm.amdgcn.exp2.f32(float %938), !dbg !28
  %local.reduce175 = fadd float %local.reduce171, %939, !dbg !29, !frisk.local_reduction !1
  %940 = insertelement <4 x float> %937, float %939, i64 3, !dbg !30
  %math.result449 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %940) #9, !dbg !30, !frisk.register_lifetimes !1
  %.cvtpack.f02081 = extractelement <4 x float> %math.result449, i32 0, !dbg !30
  %.cvtpack.f12082 = extractelement <4 x float> %math.result449, i32 1, !dbg !30
  %.cvtpack.f22083 = extractelement <4 x float> %math.result449, i32 2, !dbg !30
  %.cvtpack.f32084 = extractelement <4 x float> %math.result449, i32 3, !dbg !30
  %.cvtpack.p02085 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02081, float %.cvtpack.f12082), !dbg !30
  %.cvtpack.p12086 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22083, float %.cvtpack.f32084), !dbg !30
  %.cvtpack.v02087 = insertelement <2 x i32> poison, i32 %.cvtpack.p02085, i32 0, !dbg !30
  %.cvtpack.v12088 = insertelement <2 x i32> %.cvtpack.v02087, i32 %.cvtpack.p12086, i32 1, !dbg !30
  %941 = bitcast <2 x i32> %.cvtpack.v12088 to <4 x half>, !dbg !30
  %math.packed450 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %941) #9, !dbg !29, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %942 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed450, <4 x half> %927, <4 x float> %908) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %943 = shufflevector <8 x half> %743, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %944 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed450, <4 x half> %943, <4 x float> %910) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %945 = fsub <4 x float> %705, %685, !dbg !24
  %946 = extractelement <4 x float> %945, i64 0, !dbg !28
  %947 = call float @llvm.amdgcn.exp2.f32(float %946), !dbg !28
  %local.reduce141 = fadd float %local.reduce137, %947, !dbg !29, !frisk.local_reduction !1
  %948 = insertelement <4 x float> poison, float %947, i64 0, !dbg !30
  %949 = extractelement <4 x float> %945, i64 1, !dbg !28
  %950 = call float @llvm.amdgcn.exp2.f32(float %949), !dbg !28
  %local.reduce142 = fadd float %local.reduce138, %950, !dbg !29, !frisk.local_reduction !1
  %951 = insertelement <4 x float> %948, float %950, i64 1, !dbg !30
  %952 = extractelement <4 x float> %945, i64 2, !dbg !28
  %953 = call float @llvm.amdgcn.exp2.f32(float %952), !dbg !28
  %local.reduce143 = fadd float %local.reduce139, %953, !dbg !29, !frisk.local_reduction !1
  %954 = insertelement <4 x float> %951, float %953, i64 2, !dbg !30
  %955 = extractelement <4 x float> %945, i64 3, !dbg !28
  %956 = call float @llvm.amdgcn.exp2.f32(float %955), !dbg !28
  %local.reduce144 = fadd float %local.reduce140, %956, !dbg !29, !frisk.local_reduction !1
  %957 = insertelement <4 x float> %954, float %956, i64 3, !dbg !30
  %math.result465 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %957) #9, !dbg !30, !frisk.register_lifetimes !1
  %.cvtpack.f02089 = extractelement <4 x float> %math.result465, i32 0, !dbg !30
  %.cvtpack.f12090 = extractelement <4 x float> %math.result465, i32 1, !dbg !30
  %.cvtpack.f22091 = extractelement <4 x float> %math.result465, i32 2, !dbg !30
  %.cvtpack.f32092 = extractelement <4 x float> %math.result465, i32 3, !dbg !30
  %.cvtpack.p02093 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02089, float %.cvtpack.f12090), !dbg !30
  %.cvtpack.p12094 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22091, float %.cvtpack.f32092), !dbg !30
  %.cvtpack.v02095 = insertelement <2 x i32> poison, i32 %.cvtpack.p02093, i32 0, !dbg !30
  %.cvtpack.v12096 = insertelement <2 x i32> %.cvtpack.v02095, i32 %.cvtpack.p12094, i32 1, !dbg !30
  %958 = bitcast <2 x i32> %.cvtpack.v12096 to <4 x half>, !dbg !30
  %math.packed466 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %958) #9, !dbg !29, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %959 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed466, <4 x half> %927, <4 x float> %925) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %960 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed466, <4 x half> %943, <4 x float> %926) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %961 = shufflevector <8 x half> %745, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %962 = fsub <4 x float> %676, %652, !dbg !24
  %963 = extractelement <4 x float> %962, i64 0, !dbg !28
  %964 = call float @llvm.amdgcn.exp2.f32(float %963), !dbg !28
  %local.reduce176 = fadd float %local.reduce172, %964, !dbg !29, !frisk.local_reduction !1
  %965 = insertelement <4 x float> poison, float %964, i64 0, !dbg !30
  %966 = extractelement <4 x float> %962, i64 1, !dbg !28
  %967 = call float @llvm.amdgcn.exp2.f32(float %966), !dbg !28
  %local.reduce177 = fadd float %local.reduce173, %967, !dbg !29, !frisk.local_reduction !1
  %968 = insertelement <4 x float> %965, float %967, i64 1, !dbg !30
  %969 = extractelement <4 x float> %962, i64 2, !dbg !28
  %970 = call float @llvm.amdgcn.exp2.f32(float %969), !dbg !28
  %local.reduce178 = fadd float %local.reduce174, %970, !dbg !29, !frisk.local_reduction !1
  %971 = insertelement <4 x float> %968, float %970, i64 2, !dbg !30
  %972 = extractelement <4 x float> %962, i64 3, !dbg !28
  %973 = call float @llvm.amdgcn.exp2.f32(float %972), !dbg !28
  %local.reduce179 = fadd float %local.reduce175, %973, !dbg !29, !frisk.local_reduction !1
  %974 = insertelement <4 x float> %971, float %973, i64 3, !dbg !30
  %math.result451 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %974) #9, !dbg !30, !frisk.register_lifetimes !1
  %.cvtpack.f02097 = extractelement <4 x float> %math.result451, i32 0, !dbg !30
  %.cvtpack.f12098 = extractelement <4 x float> %math.result451, i32 1, !dbg !30
  %.cvtpack.f22099 = extractelement <4 x float> %math.result451, i32 2, !dbg !30
  %.cvtpack.f32100 = extractelement <4 x float> %math.result451, i32 3, !dbg !30
  %.cvtpack.p02101 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02097, float %.cvtpack.f12098), !dbg !30
  %.cvtpack.p12102 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22099, float %.cvtpack.f32100), !dbg !30
  %.cvtpack.v02103 = insertelement <2 x i32> poison, i32 %.cvtpack.p02101, i32 0, !dbg !30
  %.cvtpack.v12104 = insertelement <2 x i32> %.cvtpack.v02103, i32 %.cvtpack.p12102, i32 1, !dbg !30
  %975 = bitcast <2 x i32> %.cvtpack.v12104 to <4 x half>, !dbg !30
  %math.packed452 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %975) #9, !dbg !29, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %976 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed452, <4 x half> %961, <4 x float> %942) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %977 = shufflevector <8 x half> %745, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %978 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed452, <4 x half> %977, <4 x float> %944) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %979 = fsub <4 x float> %709, %685, !dbg !24
  %980 = extractelement <4 x float> %979, i64 0, !dbg !28
  %981 = call float @llvm.amdgcn.exp2.f32(float %980), !dbg !28
  %local.reduce145 = fadd float %local.reduce141, %981, !dbg !29, !frisk.local_reduction !1
  %982 = insertelement <4 x float> poison, float %981, i64 0, !dbg !30
  %983 = extractelement <4 x float> %979, i64 1, !dbg !28
  %984 = call float @llvm.amdgcn.exp2.f32(float %983), !dbg !28
  %local.reduce146 = fadd float %local.reduce142, %984, !dbg !29, !frisk.local_reduction !1
  %985 = insertelement <4 x float> %982, float %984, i64 1, !dbg !30
  %986 = extractelement <4 x float> %979, i64 2, !dbg !28
  %987 = call float @llvm.amdgcn.exp2.f32(float %986), !dbg !28
  %local.reduce147 = fadd float %local.reduce143, %987, !dbg !29, !frisk.local_reduction !1
  %988 = insertelement <4 x float> %985, float %987, i64 2, !dbg !30
  %989 = extractelement <4 x float> %979, i64 3, !dbg !28
  %990 = call float @llvm.amdgcn.exp2.f32(float %989), !dbg !28
  %local.reduce148 = fadd float %local.reduce144, %990, !dbg !29, !frisk.local_reduction !1
  %991 = insertelement <4 x float> %988, float %990, i64 3, !dbg !30
  %math.result467 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %991) #9, !dbg !30, !frisk.register_lifetimes !1
  %.cvtpack.f02105 = extractelement <4 x float> %math.result467, i32 0, !dbg !30
  %.cvtpack.f12106 = extractelement <4 x float> %math.result467, i32 1, !dbg !30
  %.cvtpack.f22107 = extractelement <4 x float> %math.result467, i32 2, !dbg !30
  %.cvtpack.f32108 = extractelement <4 x float> %math.result467, i32 3, !dbg !30
  %.cvtpack.p02109 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02105, float %.cvtpack.f12106), !dbg !30
  %.cvtpack.p12110 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22107, float %.cvtpack.f32108), !dbg !30
  %.cvtpack.v02111 = insertelement <2 x i32> poison, i32 %.cvtpack.p02109, i32 0, !dbg !30
  %.cvtpack.v12112 = insertelement <2 x i32> %.cvtpack.v02111, i32 %.cvtpack.p12110, i32 1, !dbg !30
  %992 = bitcast <2 x i32> %.cvtpack.v12112 to <4 x half>, !dbg !30
  %math.packed468 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %992) #9, !dbg !29, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %993 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed468, <4 x half> %961, <4 x float> %959) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %994 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed468, <4 x half> %977, <4 x float> %960) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %995 = bitcast <4 x i32> %746 to <8 x half>, !dbg !13
  %996 = shufflevector <8 x half> %995, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %997 = fsub <4 x float> %680, %652, !dbg !24
  %998 = extractelement <4 x float> %997, i64 0, !dbg !28
  %999 = call float @llvm.amdgcn.exp2.f32(float %998), !dbg !28
  %local.reduce180 = fadd float %local.reduce176, %999, !dbg !29, !frisk.local_reduction !1
  %1000 = insertelement <4 x float> poison, float %999, i64 0, !dbg !30
  %1001 = extractelement <4 x float> %997, i64 1, !dbg !28
  %1002 = call float @llvm.amdgcn.exp2.f32(float %1001), !dbg !28
  %local.reduce181 = fadd float %local.reduce177, %1002, !dbg !29, !frisk.local_reduction !1
  %local.reduce184 = fadd float %local.reduce180, %local.reduce181, !dbg !29, !frisk.local_reduction !1
  %1003 = insertelement <4 x float> %1000, float %1002, i64 1, !dbg !30
  %1004 = extractelement <4 x float> %997, i64 2, !dbg !28
  %1005 = call float @llvm.amdgcn.exp2.f32(float %1004), !dbg !28
  %local.reduce182 = fadd float %local.reduce178, %1005, !dbg !29, !frisk.local_reduction !1
  %1006 = insertelement <4 x float> %1003, float %1005, i64 2, !dbg !30
  %1007 = extractelement <4 x float> %997, i64 3, !dbg !28
  %1008 = call float @llvm.amdgcn.exp2.f32(float %1007), !dbg !28
  %local.reduce183 = fadd float %local.reduce179, %1008, !dbg !29, !frisk.local_reduction !1
  %local.reduce185 = fadd float %local.reduce182, %local.reduce183, !dbg !29, !frisk.local_reduction !1
  %1009 = fadd float %local.reduce184, %local.reduce185, !dbg !29, !frisk.local_reduction !1
  %1010 = fadd float %1009, %721, !dbg !29
  %1011 = insertelement <1 x float> poison, float %1010, i64 0, !dbg !7
  %1012 = insertvalue [2 x <1 x float>] poison, <1 x float> %1011, 0, !dbg !7
  %1013 = insertelement <4 x float> %1006, float %1008, i64 3, !dbg !30
  %math.result453 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1013) #9, !dbg !30, !frisk.register_lifetimes !1
  %.cvtpack.f02113 = extractelement <4 x float> %math.result453, i32 0, !dbg !30
  %.cvtpack.f12114 = extractelement <4 x float> %math.result453, i32 1, !dbg !30
  %.cvtpack.f22115 = extractelement <4 x float> %math.result453, i32 2, !dbg !30
  %.cvtpack.f32116 = extractelement <4 x float> %math.result453, i32 3, !dbg !30
  %.cvtpack.p02117 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02113, float %.cvtpack.f12114), !dbg !30
  %.cvtpack.p12118 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22115, float %.cvtpack.f32116), !dbg !30
  %.cvtpack.v02119 = insertelement <2 x i32> poison, i32 %.cvtpack.p02117, i32 0, !dbg !30
  %.cvtpack.v12120 = insertelement <2 x i32> %.cvtpack.v02119, i32 %.cvtpack.p12118, i32 1, !dbg !30
  %1014 = bitcast <2 x i32> %.cvtpack.v12120 to <4 x half>, !dbg !30
  %math.packed454 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1014) #9, !dbg !29, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %1015 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed454, <4 x half> %996, <4 x float> %976) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1016 = shufflevector <8 x half> %995, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1017 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed454, <4 x half> %1016, <4 x float> %978) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1018 = fsub <4 x float> %713, %685, !dbg !24
  %1019 = extractelement <4 x float> %1018, i64 0, !dbg !28
  %1020 = call float @llvm.amdgcn.exp2.f32(float %1019), !dbg !28
  %local.reduce149 = fadd float %local.reduce145, %1020, !dbg !29, !frisk.local_reduction !1
  %1021 = insertelement <4 x float> poison, float %1020, i64 0, !dbg !30
  %1022 = extractelement <4 x float> %1018, i64 1, !dbg !28
  %1023 = call float @llvm.amdgcn.exp2.f32(float %1022), !dbg !28
  %local.reduce150 = fadd float %local.reduce146, %1023, !dbg !29, !frisk.local_reduction !1
  %local.reduce153 = fadd float %local.reduce149, %local.reduce150, !dbg !29, !frisk.local_reduction !1
  %1024 = insertelement <4 x float> %1021, float %1023, i64 1, !dbg !30
  %1025 = extractelement <4 x float> %1018, i64 2, !dbg !28
  %1026 = call float @llvm.amdgcn.exp2.f32(float %1025), !dbg !28
  %local.reduce151 = fadd float %local.reduce147, %1026, !dbg !29, !frisk.local_reduction !1
  %1027 = insertelement <4 x float> %1024, float %1026, i64 2, !dbg !30
  %1028 = extractelement <4 x float> %1018, i64 3, !dbg !28
  %1029 = call float @llvm.amdgcn.exp2.f32(float %1028), !dbg !28
  %local.reduce152 = fadd float %local.reduce148, %1029, !dbg !29, !frisk.local_reduction !1
  %local.reduce154 = fadd float %local.reduce151, %local.reduce152, !dbg !29, !frisk.local_reduction !1
  %1030 = fadd float %local.reduce153, %local.reduce154, !dbg !29, !frisk.local_reduction !1
  %1031 = fadd float %1030, %729, !dbg !29
  %1032 = insertelement <1 x float> poison, float %1031, i64 0, !dbg !7
  %1033 = insertvalue [2 x <1 x float>] %1012, <1 x float> %1032, 1, !dbg !7
  %1034 = insertelement <4 x float> %1027, float %1029, i64 3, !dbg !30
  %math.result469 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1034) #9, !dbg !30, !frisk.register_lifetimes !1
  %.cvtpack.f02121 = extractelement <4 x float> %math.result469, i32 0, !dbg !30
  %.cvtpack.f12122 = extractelement <4 x float> %math.result469, i32 1, !dbg !30
  %.cvtpack.f22123 = extractelement <4 x float> %math.result469, i32 2, !dbg !30
  %.cvtpack.f32124 = extractelement <4 x float> %math.result469, i32 3, !dbg !30
  %.cvtpack.p02125 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02121, float %.cvtpack.f12122), !dbg !30
  %.cvtpack.p12126 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22123, float %.cvtpack.f32124), !dbg !30
  %.cvtpack.v02127 = insertelement <2 x i32> poison, i32 %.cvtpack.p02125, i32 0, !dbg !30
  %.cvtpack.v12128 = insertelement <2 x i32> %.cvtpack.v02127, i32 %.cvtpack.p12126, i32 1, !dbg !30
  %1035 = bitcast <2 x i32> %.cvtpack.v12128 to <4 x half>, !dbg !30
  %math.packed470 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1035) #9, !dbg !29, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %1036 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed470, <4 x half> %996, <4 x float> %993) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1037 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed470, <4 x half> %1016, <4 x float> %994) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %vdb.address1558 = or disjoint i64 %.idx15081510, 64, !dbg !17
  %vdb.address1559 = or disjoint i64 %address.local988, %vdb.address1558, !dbg !17
  %vdb.address1560 = trunc i64 %vdb.address1559 to i32, !dbg !17
  %vdb.address1561 = shl i32 %vdb.address1560, 1, !dbg !17
  %vdb.address1568 = or disjoint i64 %address.local994, %vdb.address1558, !dbg !17
  %vdb.address1569 = trunc i64 %vdb.address1568 to i32, !dbg !17
  %vdb.address1570 = shl i32 %vdb.address1569, 1, !dbg !17
  %1038 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %138, i32 %vdb.address1561, <4 x i32> %dma.buffer.rsrc1414) #10, !dbg !17
  %1039 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %145, i32 %vdb.address1570, <4 x i32> %dma.buffer.rsrc1414) #10, !dbg !17
  %vdb.read = getelementptr i8, ptr addrspace(3) %address.local1002, i64 8192, !dbg !13
  %1040 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #0, !dbg !13, !frisk.v_double_buffer !37
  %1041 = bitcast <4 x i32> %1040 to <8 x half>, !dbg !13
  %vdb.read1550 = getelementptr i8, ptr addrspace(3) %address.local1011, i64 8448, !dbg !13
  %1042 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1550, i32 0) #0, !dbg !13, !frisk.v_double_buffer !37
  %1043 = bitcast <4 x i32> %1042 to <8 x half>, !dbg !13
  %vdb.read1551 = getelementptr i8, ptr addrspace(3) %address.local1011, i64 8704, !dbg !13
  %1044 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1551, i32 0) #0, !dbg !13, !frisk.v_double_buffer !37
  %1045 = bitcast <4 x i32> %1044 to <8 x half>, !dbg !13
  %vdb.read1552 = getelementptr i8, ptr addrspace(3) %address.local1011, i64 8960, !dbg !13
  %1046 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1552, i32 0) #0, !dbg !13, !frisk.v_double_buffer !37
  %1047 = bitcast <4 x i32> %1046 to <8 x half>, !dbg !13
  %vdb.read1553 = getelementptr i8, ptr addrspace(3) %address.local1002, i64 12288, !dbg !13
  %1048 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1553, i32 0) #0, !dbg !13, !frisk.v_double_buffer !37
  %1049 = bitcast <4 x i32> %1048 to <8 x half>, !dbg !13
  %vdb.read1554 = getelementptr i8, ptr addrspace(3) %address.local1011, i64 12544, !dbg !13
  %1050 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1554, i32 0) #0, !dbg !13, !frisk.v_double_buffer !37
  %1051 = bitcast <4 x i32> %1050 to <8 x half>, !dbg !13
  %vdb.read1555 = getelementptr i8, ptr addrspace(3) %address.local1011, i64 12800, !dbg !13
  %1052 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1555, i32 0) #0, !dbg !13, !frisk.v_double_buffer !37
  %1053 = bitcast <4 x i32> %1052 to <8 x half>, !dbg !13
  %vdb.read1556 = getelementptr i8, ptr addrspace(3) %address.local1011, i64 13056, !dbg !13
  %1054 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1556, i32 0) #0, !dbg !13, !frisk.v_double_buffer !37
  %1055 = shufflevector <8 x half> %1041, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1056 = extractvalue [2 x <8 x float>] %123, 0, !dbg !31
  %1057 = shufflevector <8 x float> %1056, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %1058 = fmul <4 x float> %1057, %718, !dbg !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1059 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1055, <4 x float> %1058) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1060 = shufflevector <8 x half> %1041, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1061 = shufflevector <8 x float> %1056, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %1062 = fmul <4 x float> %1061, %718, !dbg !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1063 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1060, <4 x float> %1062) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !31
  %1064 = extractvalue [2 x <8 x float>] %123, 1, !dbg !31
  %1065 = shufflevector <8 x float> %1064, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %1066 = fmul <4 x float> %1065, %726, !dbg !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1067 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed456, <4 x half> %1055, <4 x float> %1066) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !31
  %1068 = shufflevector <8 x float> %1064, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %1069 = fmul <4 x float> %1068, %726, !dbg !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1070 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed456, <4 x half> %1060, <4 x float> %1069) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1071 = shufflevector <8 x half> %1043, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1072 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed442, <4 x half> %1071, <4 x float> %1059) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1073 = shufflevector <8 x half> %1043, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1074 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed442, <4 x half> %1073, <4 x float> %1063) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1075 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed458, <4 x half> %1071, <4 x float> %1067) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1076 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed458, <4 x half> %1073, <4 x float> %1070) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1077 = shufflevector <8 x half> %1045, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1078 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed444, <4 x half> %1077, <4 x float> %1072) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1079 = shufflevector <8 x half> %1045, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1080 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed444, <4 x half> %1079, <4 x float> %1074) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1081 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed460, <4 x half> %1077, <4 x float> %1075) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1082 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed460, <4 x half> %1079, <4 x float> %1076) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1083 = shufflevector <8 x half> %1047, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1084 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed446, <4 x half> %1083, <4 x float> %1078) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1085 = shufflevector <8 x half> %1047, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1086 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed446, <4 x half> %1085, <4 x float> %1080) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1087 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed462, <4 x half> %1083, <4 x float> %1081) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1088 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed462, <4 x half> %1085, <4 x float> %1082) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1089 = shufflevector <8 x half> %1049, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1090 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed448, <4 x half> %1089, <4 x float> %1084) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1091 = shufflevector <8 x half> %1049, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1092 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed448, <4 x half> %1091, <4 x float> %1086) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1093 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed464, <4 x half> %1089, <4 x float> %1087) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1094 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed464, <4 x half> %1091, <4 x float> %1088) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1095 = shufflevector <8 x half> %1051, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1096 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed450, <4 x half> %1095, <4 x float> %1090) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1097 = shufflevector <8 x half> %1051, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1098 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed450, <4 x half> %1097, <4 x float> %1092) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1099 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed466, <4 x half> %1095, <4 x float> %1093) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1100 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed466, <4 x half> %1097, <4 x float> %1094) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1101 = shufflevector <8 x half> %1053, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1102 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed452, <4 x half> %1101, <4 x float> %1096) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1103 = shufflevector <8 x half> %1053, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1104 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed452, <4 x half> %1103, <4 x float> %1098) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1105 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed468, <4 x half> %1101, <4 x float> %1099) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1106 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed468, <4 x half> %1103, <4 x float> %1100) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1107 = bitcast <4 x i32> %1054 to <8 x half>, !dbg !13
  %1108 = shufflevector <8 x half> %1107, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1109 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed454, <4 x half> %1108, <4 x float> %1102) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1110 = shufflevector <8 x half> %1107, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1111 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed454, <4 x half> %1110, <4 x float> %1104) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1112 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed470, <4 x half> %1108, <4 x float> %1105) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1113 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed470, <4 x half> %1110, <4 x float> %1106) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %vdb.address1576 = or disjoint i64 %.idx15081510, 96, !dbg !17
  %vdb.address1577 = or disjoint i64 %address.local988, %vdb.address1576, !dbg !17
  %vdb.address1578 = trunc i64 %vdb.address1577 to i32, !dbg !17
  %vdb.address1579 = shl i32 %vdb.address1578, 1, !dbg !17
  %vdb.address1586 = or disjoint i64 %address.local994, %vdb.address1576, !dbg !17
  %vdb.address1587 = trunc i64 %vdb.address1586 to i32, !dbg !17
  %vdb.address1588 = shl i32 %vdb.address1587, 1, !dbg !17
  %1114 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1579, <4 x i32> %dma.buffer.rsrc1414) #10, !dbg !17
  %1115 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1549, i32 %vdb.address1588, <4 x i32> %dma.buffer.rsrc1414) #10, !dbg !17
  %1116 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1002, i32 0) #0, !dbg !13, !frisk.v_double_buffer !37
  %1117 = bitcast <4 x i32> %1116 to <8 x half>, !dbg !13
  %1118 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1012, i32 0) #0, !dbg !13, !frisk.v_double_buffer !37
  %1119 = bitcast <4 x i32> %1118 to <8 x half>, !dbg !13
  %1120 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1016, i32 0) #0, !dbg !13, !frisk.v_double_buffer !37
  %1121 = bitcast <4 x i32> %1120 to <8 x half>, !dbg !13
  %1122 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1020, i32 0) #0, !dbg !13, !frisk.v_double_buffer !37
  %1123 = bitcast <4 x i32> %1122 to <8 x half>, !dbg !13
  %1124 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1022, i32 0) #0, !dbg !13, !frisk.v_double_buffer !37
  %1125 = bitcast <4 x i32> %1124 to <8 x half>, !dbg !13
  %1126 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1026, i32 0) #0, !dbg !13, !frisk.v_double_buffer !37
  %1127 = bitcast <4 x i32> %1126 to <8 x half>, !dbg !13
  %1128 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1030, i32 0) #0, !dbg !13, !frisk.v_double_buffer !37
  %1129 = bitcast <4 x i32> %1128 to <8 x half>, !dbg !13
  %1130 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1034, i32 0) #0, !dbg !13, !frisk.v_double_buffer !37
  %1131 = shufflevector <8 x half> %1117, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1132 = extractvalue [2 x <8 x float>] %124, 0, !dbg !31
  %1133 = shufflevector <8 x float> %1132, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %1134 = fmul <4 x float> %1133, %718, !dbg !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1135 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1131, <4 x float> %1134) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1136 = shufflevector <8 x half> %1117, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1137 = shufflevector <8 x float> %1132, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %1138 = fmul <4 x float> %1137, %718, !dbg !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1139 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1136, <4 x float> %1138) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !31
  %1140 = extractvalue [2 x <8 x float>] %124, 1, !dbg !31
  %1141 = shufflevector <8 x float> %1140, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %1142 = fmul <4 x float> %1141, %726, !dbg !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1143 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed456, <4 x half> %1131, <4 x float> %1142) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !31
  %1144 = shufflevector <8 x float> %1140, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %1145 = fmul <4 x float> %1144, %726, !dbg !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1146 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed456, <4 x half> %1136, <4 x float> %1145) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1147 = shufflevector <8 x half> %1119, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1148 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed442, <4 x half> %1147, <4 x float> %1135) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1149 = shufflevector <8 x half> %1119, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1150 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed442, <4 x half> %1149, <4 x float> %1139) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1151 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed458, <4 x half> %1147, <4 x float> %1143) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1152 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed458, <4 x half> %1149, <4 x float> %1146) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1153 = shufflevector <8 x half> %1121, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1154 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed444, <4 x half> %1153, <4 x float> %1148) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1155 = shufflevector <8 x half> %1121, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1156 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed444, <4 x half> %1155, <4 x float> %1150) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1157 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed460, <4 x half> %1153, <4 x float> %1151) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1158 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed460, <4 x half> %1155, <4 x float> %1152) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1159 = shufflevector <8 x half> %1123, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1160 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed446, <4 x half> %1159, <4 x float> %1154) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1161 = shufflevector <8 x half> %1123, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1162 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed446, <4 x half> %1161, <4 x float> %1156) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1163 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed462, <4 x half> %1159, <4 x float> %1157) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1164 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed462, <4 x half> %1161, <4 x float> %1158) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1165 = shufflevector <8 x half> %1125, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1166 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed448, <4 x half> %1165, <4 x float> %1160) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1167 = shufflevector <8 x half> %1125, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1168 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed448, <4 x half> %1167, <4 x float> %1162) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1169 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed464, <4 x half> %1165, <4 x float> %1163) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1170 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed464, <4 x half> %1167, <4 x float> %1164) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1171 = shufflevector <8 x half> %1127, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1172 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed450, <4 x half> %1171, <4 x float> %1166) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1173 = shufflevector <8 x half> %1127, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1174 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed450, <4 x half> %1173, <4 x float> %1168) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1175 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed466, <4 x half> %1171, <4 x float> %1169) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1176 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed466, <4 x half> %1173, <4 x float> %1170) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1177 = shufflevector <8 x half> %1129, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1178 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed452, <4 x half> %1177, <4 x float> %1172) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1179 = shufflevector <8 x half> %1129, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1180 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed452, <4 x half> %1179, <4 x float> %1174) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1181 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed468, <4 x half> %1177, <4 x float> %1175) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1182 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed468, <4 x half> %1179, <4 x float> %1176) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1183 = bitcast <4 x i32> %1130 to <8 x half>, !dbg !13
  %1184 = shufflevector <8 x half> %1183, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1185 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed454, <4 x half> %1184, <4 x float> %1178) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1186 = shufflevector <8 x half> %1183, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1187 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed454, <4 x half> %1186, <4 x float> %1180) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1188 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed470, <4 x half> %1184, <4 x float> %1181) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1189 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed470, <4 x half> %1186, <4 x float> %1182) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %1190 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #0, !dbg !13, !frisk.v_double_buffer !37
  %1191 = bitcast <4 x i32> %1190 to <8 x half>, !dbg !13
  %1192 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1550, i32 0) #0, !dbg !13, !frisk.v_double_buffer !37
  %1193 = bitcast <4 x i32> %1192 to <8 x half>, !dbg !13
  %1194 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1551, i32 0) #0, !dbg !13, !frisk.v_double_buffer !37
  %1195 = bitcast <4 x i32> %1194 to <8 x half>, !dbg !13
  %1196 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1552, i32 0) #0, !dbg !13, !frisk.v_double_buffer !37
  %1197 = bitcast <4 x i32> %1196 to <8 x half>, !dbg !13
  %1198 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1553, i32 0) #0, !dbg !13, !frisk.v_double_buffer !37
  %1199 = bitcast <4 x i32> %1198 to <8 x half>, !dbg !13
  %1200 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1554, i32 0) #0, !dbg !13, !frisk.v_double_buffer !37
  %1201 = bitcast <4 x i32> %1200 to <8 x half>, !dbg !13
  %1202 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1555, i32 0) #0, !dbg !13, !frisk.v_double_buffer !37
  %1203 = bitcast <4 x i32> %1202 to <8 x half>, !dbg !13
  %1204 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1556, i32 0) #0, !dbg !13, !frisk.v_double_buffer !37
  %1205 = shufflevector <8 x half> %1191, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1206 = extractvalue [2 x <8 x float>] %125, 0, !dbg !31
  %1207 = shufflevector <8 x float> %1206, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %1208 = fmul <4 x float> %1207, %718, !dbg !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1209 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1205, <4 x float> %1208) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1210 = shufflevector <8 x half> %1191, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1211 = shufflevector <8 x float> %1206, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %1212 = fmul <4 x float> %1211, %718, !dbg !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1213 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1210, <4 x float> %1212) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !31
  %1214 = extractvalue [2 x <8 x float>] %125, 1, !dbg !31
  %1215 = shufflevector <8 x float> %1214, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %1216 = fmul <4 x float> %1215, %726, !dbg !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1217 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed456, <4 x half> %1205, <4 x float> %1216) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !31
  %1218 = shufflevector <8 x float> %1214, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %1219 = fmul <4 x float> %1218, %726, !dbg !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1220 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed456, <4 x half> %1210, <4 x float> %1219) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1221 = shufflevector <8 x half> %1193, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1222 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed442, <4 x half> %1221, <4 x float> %1209) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1223 = shufflevector <8 x half> %1193, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1224 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed442, <4 x half> %1223, <4 x float> %1213) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1225 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed458, <4 x half> %1221, <4 x float> %1217) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1226 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed458, <4 x half> %1223, <4 x float> %1220) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1227 = shufflevector <8 x half> %1195, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1228 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed444, <4 x half> %1227, <4 x float> %1222) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1229 = shufflevector <8 x half> %1195, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1230 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed444, <4 x half> %1229, <4 x float> %1224) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1231 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed460, <4 x half> %1227, <4 x float> %1225) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1232 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed460, <4 x half> %1229, <4 x float> %1226) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1233 = shufflevector <8 x half> %1197, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1234 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed446, <4 x half> %1233, <4 x float> %1228) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1235 = shufflevector <8 x half> %1197, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1236 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed446, <4 x half> %1235, <4 x float> %1230) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1237 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed462, <4 x half> %1233, <4 x float> %1231) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1238 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed462, <4 x half> %1235, <4 x float> %1232) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1239 = shufflevector <8 x half> %1199, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1240 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed448, <4 x half> %1239, <4 x float> %1234) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1241 = shufflevector <8 x half> %1199, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1242 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed448, <4 x half> %1241, <4 x float> %1236) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1243 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed464, <4 x half> %1239, <4 x float> %1237) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1244 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed464, <4 x half> %1241, <4 x float> %1238) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1245 = shufflevector <8 x half> %1201, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1246 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed450, <4 x half> %1245, <4 x float> %1240) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1247 = shufflevector <8 x half> %1201, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1248 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed450, <4 x half> %1247, <4 x float> %1242) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1249 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed466, <4 x half> %1245, <4 x float> %1243) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1250 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed466, <4 x half> %1247, <4 x float> %1244) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1251 = shufflevector <8 x half> %1203, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1252 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed452, <4 x half> %1251, <4 x float> %1246) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1253 = shufflevector <8 x half> %1203, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1254 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed452, <4 x half> %1253, <4 x float> %1248) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1255 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed468, <4 x half> %1251, <4 x float> %1249) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1256 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed468, <4 x half> %1253, <4 x float> %1250) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1257 = bitcast <4 x i32> %1204 to <8 x half>, !dbg !13
  %1258 = shufflevector <8 x half> %1257, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1259 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed454, <4 x half> %1258, <4 x float> %1252) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1260 = shufflevector <8 x half> %1257, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1261 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed454, <4 x half> %1260, <4 x float> %1254) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1262 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed470, <4 x half> %1258, <4 x float> %1255) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1263 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed470, <4 x half> %1260, <4 x float> %1256) #9, !dbg !13, !frisk.pv_interleave !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !7
  %1264 = shufflevector <4 x float> %1015, <4 x float> %1017, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1265 = insertvalue [2 x <8 x float>] poison, <8 x float> %1264, 0, !dbg !7
  %1266 = shufflevector <4 x float> %1036, <4 x float> %1037, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1267 = insertvalue [2 x <8 x float>] %1265, <8 x float> %1266, 1, !dbg !7
  %1268 = insertvalue [2 x <1 x float>] poison, <1 x float> %644, 0, !dbg !7
  %1269 = insertvalue [2 x <1 x float>] %1268, <1 x float> %647, 1, !dbg !7
  %1270 = shufflevector <4 x float> %1109, <4 x float> %1111, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1271 = insertvalue [2 x <8 x float>] poison, <8 x float> %1270, 0, !dbg !7
  %1272 = shufflevector <4 x float> %1112, <4 x float> %1113, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1273 = insertvalue [2 x <8 x float>] %1271, <8 x float> %1272, 1, !dbg !7
  %1274 = shufflevector <4 x float> %1185, <4 x float> %1187, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1275 = insertvalue [2 x <8 x float>] poison, <8 x float> %1274, 0, !dbg !7
  %1276 = shufflevector <4 x float> %1188, <4 x float> %1189, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1277 = insertvalue [2 x <8 x float>] %1275, <8 x float> %1276, 1, !dbg !7
  %1278 = shufflevector <4 x float> %1259, <4 x float> %1261, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1279 = insertvalue [2 x <8 x float>] poison, <8 x float> %1278, 0, !dbg !7
  %1280 = shufflevector <4 x float> %1262, <4 x float> %1263, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1281 = insertvalue [2 x <8 x float>] %1279, <8 x float> %1280, 1, !dbg !7
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %1282 = add nuw nsw i64 %119, 1, !dbg !7
  br label %118, !dbg !7

1283:                                             ; preds = %118
  %lane.sum.field251 = extractvalue [2 x <1 x float>] %121, 1, !dbg !7
  %lane.sum.local252 = extractelement <1 x float> %lane.sum.field251, i64 0, !dbg !7
  %lane.sum.lane.lo253 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !7
  %lane.sum.lane254 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %lane.sum.lane.lo253), !dbg !7
  %1284 = shl nuw nsw i32 %lane.sum.lane254, 2, !dbg !7
  %1285 = xor i32 %1284, 64, !dbg !7
  %1286 = bitcast float %lane.sum.local252 to i32, !dbg !7
  %lane.sum.shuffle255 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1285, i32 %1286), !dbg !7
  %1287 = bitcast i32 %lane.sum.shuffle255 to float, !dbg !7
  %lane.sum.final256 = fadd float %lane.sum.local252, %1287, !dbg !7
  %1288 = xor i32 %1284, 128, !dbg !7
  %1289 = bitcast float %lane.sum.final256 to i32, !dbg !7
  %lane.sum.shuffle257 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1288, i32 %1289), !dbg !7
  %1290 = bitcast i32 %lane.sum.shuffle257 to float, !dbg !7
  %lane.sum.final258 = fadd float %lane.sum.final256, %1290, !dbg !7
  %1291 = insertelement <1 x float> poison, float %lane.sum.final258, i64 0, !dbg !7
  %lane.sum.field = extractvalue [2 x <1 x float>] %121, 0, !dbg !7
  %lane.sum.local = extractelement <1 x float> %lane.sum.field, i64 0, !dbg !7
  %1292 = bitcast float %lane.sum.local to i32, !dbg !7
  %lane.sum.shuffle = call i32 @llvm.amdgcn.ds.bpermute(i32 %1285, i32 %1292), !dbg !7
  %1293 = bitcast i32 %lane.sum.shuffle to float, !dbg !7
  %lane.sum.final = fadd float %lane.sum.local, %1293, !dbg !7
  %1294 = bitcast float %lane.sum.final to i32, !dbg !7
  %lane.sum.shuffle249 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1288, i32 %1294), !dbg !7
  %1295 = bitcast i32 %lane.sum.shuffle249 to float, !dbg !7
  %lane.sum.final250 = fadd float %lane.sum.final, %1295, !dbg !7
  %1296 = insertelement <1 x float> poison, float %lane.sum.final250, i64 0, !dbg !7
  %1297 = and i32 %kdma.tid, 16, !dbg !7
  %.not = icmp eq i32 %1297, 0, !dbg !7
  %1298 = and i32 %kdma.tid, 32, !dbg !7
  %.not36 = icmp eq i32 %1298, 0, !dbg !7
  %1299 = extractvalue [2 x <8 x float>] %125, 0, !dbg !7
  %1300 = extractelement <8 x float> %1299, i64 0, !dbg !7
  %1301 = extractelement <8 x float> %1299, i64 1, !dbg !7
  %1302 = extractelement <8 x float> %1299, i64 2, !dbg !7
  %1303 = extractelement <8 x float> %1299, i64 3, !dbg !7
  %1304 = select i1 %.not, float %1301, float %1300, !dbg !7
  %1305 = add nuw nsw i32 %lane.sum.lane254, 64, !dbg !7
  %1306 = xor i32 %lane.sum.lane254, 16, !dbg !7
  %1307 = and i32 %1305, 192, !dbg !7
  %1308 = icmp samesign ult i32 %1306, %1307, !dbg !7
  %1309 = select i1 %1308, i32 %1306, i32 %lane.sum.lane254, !dbg !7
  %1310 = shl nuw nsw i32 %1309, 2, !dbg !7
  %1311 = bitcast float %1304 to i32, !dbg !7
  %1312 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1310, i32 %1311), !dbg !7
  %1313 = bitcast i32 %1312 to float, !dbg !7
  %1314 = select i1 %.not, float %1300, float %1313, !dbg !7
  %1315 = select i1 %.not, float %1313, float %1301, !dbg !7
  %1316 = select i1 %.not, float %1303, float %1302, !dbg !7
  %1317 = bitcast float %1316 to i32, !dbg !7
  %1318 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1310, i32 %1317), !dbg !7
  %1319 = bitcast i32 %1318 to float, !dbg !7
  %1320 = select i1 %.not, float %1302, float %1319, !dbg !7
  %1321 = select i1 %.not, float %1319, float %1303, !dbg !7
  %1322 = select i1 %.not36, float %1320, float %1314, !dbg !7
  %1323 = xor i32 %lane.sum.lane254, 32, !dbg !7
  %1324 = icmp samesign ult i32 %1323, %1307, !dbg !7
  %1325 = select i1 %1324, i32 %1323, i32 %lane.sum.lane254, !dbg !7
  %1326 = shl nuw nsw i32 %1325, 2, !dbg !7
  %1327 = bitcast float %1322 to i32, !dbg !7
  %1328 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1326, i32 %1327), !dbg !7
  %1329 = bitcast i32 %1328 to float, !dbg !7
  %1330 = select i1 %.not36, float %1314, float %1329, !dbg !7
  %1331 = select i1 %.not36, float %1329, float %1320, !dbg !7
  %1332 = select i1 %.not36, float %1321, float %1315, !dbg !7
  %1333 = bitcast float %1332 to i32, !dbg !7
  %1334 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1326, i32 %1333), !dbg !7
  %1335 = bitcast i32 %1334 to float, !dbg !7
  %1336 = select i1 %.not36, float %1315, float %1335, !dbg !7
  %1337 = select i1 %.not36, float %1335, float %1321, !dbg !7
  %1338 = extractelement <8 x float> %1299, i64 4, !dbg !7
  %1339 = extractelement <8 x float> %1299, i64 5, !dbg !7
  %1340 = extractelement <8 x float> %1299, i64 6, !dbg !7
  %1341 = extractelement <8 x float> %1299, i64 7, !dbg !7
  %1342 = select i1 %.not, float %1339, float %1338, !dbg !7
  %1343 = bitcast float %1342 to i32, !dbg !7
  %1344 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1310, i32 %1343), !dbg !7
  %1345 = bitcast i32 %1344 to float, !dbg !7
  %1346 = select i1 %.not, float %1338, float %1345, !dbg !7
  %1347 = select i1 %.not, float %1345, float %1339, !dbg !7
  %1348 = select i1 %.not, float %1341, float %1340, !dbg !7
  %1349 = bitcast float %1348 to i32, !dbg !7
  %1350 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1310, i32 %1349), !dbg !7
  %1351 = bitcast i32 %1350 to float, !dbg !7
  %1352 = select i1 %.not, float %1340, float %1351, !dbg !7
  %1353 = select i1 %.not, float %1351, float %1341, !dbg !7
  %1354 = select i1 %.not36, float %1352, float %1346, !dbg !7
  %1355 = bitcast float %1354 to i32, !dbg !7
  %1356 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1326, i32 %1355), !dbg !7
  %1357 = bitcast i32 %1356 to float, !dbg !7
  %1358 = select i1 %.not36, float %1346, float %1357, !dbg !7
  %1359 = select i1 %.not36, float %1357, float %1352, !dbg !7
  %1360 = select i1 %.not36, float %1353, float %1347, !dbg !7
  %1361 = bitcast float %1360 to i32, !dbg !7
  %1362 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1326, i32 %1361), !dbg !7
  %1363 = bitcast i32 %1362 to float, !dbg !7
  %1364 = select i1 %.not36, float %1347, float %1363, !dbg !7
  %1365 = select i1 %.not36, float %1363, float %1353, !dbg !7
  %1366 = extractvalue [2 x <8 x float>] %125, 1, !dbg !7
  %1367 = extractelement <8 x float> %1366, i64 0, !dbg !7
  %1368 = extractelement <8 x float> %1366, i64 1, !dbg !7
  %1369 = extractelement <8 x float> %1366, i64 2, !dbg !7
  %1370 = extractelement <8 x float> %1366, i64 3, !dbg !7
  %1371 = select i1 %.not, float %1368, float %1367, !dbg !7
  %1372 = bitcast float %1371 to i32, !dbg !7
  %1373 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1310, i32 %1372), !dbg !7
  %1374 = bitcast i32 %1373 to float, !dbg !7
  %1375 = select i1 %.not, float %1367, float %1374, !dbg !7
  %1376 = select i1 %.not, float %1374, float %1368, !dbg !7
  %1377 = select i1 %.not, float %1370, float %1369, !dbg !7
  %1378 = bitcast float %1377 to i32, !dbg !7
  %1379 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1310, i32 %1378), !dbg !7
  %1380 = bitcast i32 %1379 to float, !dbg !7
  %1381 = select i1 %.not, float %1369, float %1380, !dbg !7
  %1382 = select i1 %.not, float %1380, float %1370, !dbg !7
  %1383 = select i1 %.not36, float %1381, float %1375, !dbg !7
  %1384 = bitcast float %1383 to i32, !dbg !7
  %1385 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1326, i32 %1384), !dbg !7
  %1386 = bitcast i32 %1385 to float, !dbg !7
  %1387 = select i1 %.not36, float %1375, float %1386, !dbg !7
  %1388 = select i1 %.not36, float %1386, float %1381, !dbg !7
  %1389 = select i1 %.not36, float %1382, float %1376, !dbg !7
  %1390 = bitcast float %1389 to i32, !dbg !7
  %1391 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1326, i32 %1390), !dbg !7
  %1392 = bitcast i32 %1391 to float, !dbg !7
  %1393 = select i1 %.not36, float %1376, float %1392, !dbg !7
  %1394 = select i1 %.not36, float %1392, float %1382, !dbg !7
  %1395 = extractelement <8 x float> %1366, i64 4, !dbg !7
  %1396 = extractelement <8 x float> %1366, i64 5, !dbg !7
  %1397 = extractelement <8 x float> %1366, i64 6, !dbg !7
  %1398 = extractelement <8 x float> %1366, i64 7, !dbg !7
  %1399 = select i1 %.not, float %1396, float %1395, !dbg !7
  %1400 = bitcast float %1399 to i32, !dbg !7
  %1401 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1310, i32 %1400), !dbg !7
  %1402 = bitcast i32 %1401 to float, !dbg !7
  %1403 = select i1 %.not, float %1395, float %1402, !dbg !7
  %1404 = select i1 %.not, float %1402, float %1396, !dbg !7
  %1405 = select i1 %.not, float %1398, float %1397, !dbg !7
  %1406 = bitcast float %1405 to i32, !dbg !7
  %1407 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1310, i32 %1406), !dbg !7
  %1408 = bitcast i32 %1407 to float, !dbg !7
  %1409 = select i1 %.not, float %1397, float %1408, !dbg !7
  %1410 = select i1 %.not, float %1408, float %1398, !dbg !7
  %1411 = select i1 %.not36, float %1409, float %1403, !dbg !7
  %1412 = bitcast float %1411 to i32, !dbg !7
  %1413 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1326, i32 %1412), !dbg !7
  %1414 = bitcast i32 %1413 to float, !dbg !7
  %1415 = select i1 %.not36, float %1403, float %1414, !dbg !7
  %1416 = select i1 %.not36, float %1414, float %1409, !dbg !7
  %1417 = select i1 %.not36, float %1410, float %1404, !dbg !7
  %1418 = bitcast float %1417 to i32, !dbg !7
  %1419 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1326, i32 %1418), !dbg !7
  %1420 = bitcast i32 %1419 to float, !dbg !7
  %1421 = select i1 %.not36, float %1404, float %1420, !dbg !7
  %1422 = select i1 %.not36, float %1420, float %1410, !dbg !7
  %1423 = extractvalue [2 x <8 x float>] %124, 0, !dbg !7
  %1424 = extractelement <8 x float> %1423, i64 0, !dbg !7
  %1425 = extractelement <8 x float> %1423, i64 1, !dbg !7
  %1426 = extractelement <8 x float> %1423, i64 2, !dbg !7
  %1427 = extractelement <8 x float> %1423, i64 3, !dbg !7
  %1428 = select i1 %.not, float %1425, float %1424, !dbg !7
  %1429 = bitcast float %1428 to i32, !dbg !7
  %1430 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1310, i32 %1429), !dbg !7
  %1431 = bitcast i32 %1430 to float, !dbg !7
  %1432 = select i1 %.not, float %1424, float %1431, !dbg !7
  %1433 = select i1 %.not, float %1431, float %1425, !dbg !7
  %1434 = select i1 %.not, float %1427, float %1426, !dbg !7
  %1435 = bitcast float %1434 to i32, !dbg !7
  %1436 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1310, i32 %1435), !dbg !7
  %1437 = bitcast i32 %1436 to float, !dbg !7
  %1438 = select i1 %.not, float %1426, float %1437, !dbg !7
  %1439 = select i1 %.not, float %1437, float %1427, !dbg !7
  %1440 = select i1 %.not36, float %1438, float %1432, !dbg !7
  %1441 = bitcast float %1440 to i32, !dbg !7
  %1442 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1326, i32 %1441), !dbg !7
  %1443 = bitcast i32 %1442 to float, !dbg !7
  %1444 = select i1 %.not36, float %1432, float %1443, !dbg !7
  %1445 = select i1 %.not36, float %1443, float %1438, !dbg !7
  %1446 = select i1 %.not36, float %1439, float %1433, !dbg !7
  %1447 = bitcast float %1446 to i32, !dbg !7
  %1448 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1326, i32 %1447), !dbg !7
  %1449 = bitcast i32 %1448 to float, !dbg !7
  %1450 = select i1 %.not36, float %1433, float %1449, !dbg !7
  %1451 = select i1 %.not36, float %1449, float %1439, !dbg !7
  %1452 = extractelement <8 x float> %1423, i64 4, !dbg !7
  %1453 = extractelement <8 x float> %1423, i64 5, !dbg !7
  %1454 = extractelement <8 x float> %1423, i64 6, !dbg !7
  %1455 = extractelement <8 x float> %1423, i64 7, !dbg !7
  %1456 = select i1 %.not, float %1453, float %1452, !dbg !7
  %1457 = bitcast float %1456 to i32, !dbg !7
  %1458 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1310, i32 %1457), !dbg !7
  %1459 = bitcast i32 %1458 to float, !dbg !7
  %1460 = select i1 %.not, float %1452, float %1459, !dbg !7
  %1461 = select i1 %.not, float %1459, float %1453, !dbg !7
  %1462 = select i1 %.not, float %1455, float %1454, !dbg !7
  %1463 = bitcast float %1462 to i32, !dbg !7
  %1464 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1310, i32 %1463), !dbg !7
  %1465 = bitcast i32 %1464 to float, !dbg !7
  %1466 = select i1 %.not, float %1454, float %1465, !dbg !7
  %1467 = select i1 %.not, float %1465, float %1455, !dbg !7
  %1468 = select i1 %.not36, float %1466, float %1460, !dbg !7
  %1469 = bitcast float %1468 to i32, !dbg !7
  %1470 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1326, i32 %1469), !dbg !7
  %1471 = bitcast i32 %1470 to float, !dbg !7
  %1472 = select i1 %.not36, float %1460, float %1471, !dbg !7
  %1473 = select i1 %.not36, float %1471, float %1466, !dbg !7
  %1474 = select i1 %.not36, float %1467, float %1461, !dbg !7
  %1475 = bitcast float %1474 to i32, !dbg !7
  %1476 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1326, i32 %1475), !dbg !7
  %1477 = bitcast i32 %1476 to float, !dbg !7
  %1478 = select i1 %.not36, float %1461, float %1477, !dbg !7
  %1479 = select i1 %.not36, float %1477, float %1467, !dbg !7
  %1480 = extractvalue [2 x <8 x float>] %124, 1, !dbg !7
  %1481 = extractelement <8 x float> %1480, i64 0, !dbg !7
  %1482 = extractelement <8 x float> %1480, i64 1, !dbg !7
  %1483 = extractelement <8 x float> %1480, i64 2, !dbg !7
  %1484 = extractelement <8 x float> %1480, i64 3, !dbg !7
  %1485 = select i1 %.not, float %1482, float %1481, !dbg !7
  %1486 = bitcast float %1485 to i32, !dbg !7
  %1487 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1310, i32 %1486), !dbg !7
  %1488 = bitcast i32 %1487 to float, !dbg !7
  %1489 = select i1 %.not, float %1481, float %1488, !dbg !7
  %1490 = select i1 %.not, float %1488, float %1482, !dbg !7
  %1491 = select i1 %.not, float %1484, float %1483, !dbg !7
  %1492 = bitcast float %1491 to i32, !dbg !7
  %1493 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1310, i32 %1492), !dbg !7
  %1494 = bitcast i32 %1493 to float, !dbg !7
  %1495 = select i1 %.not, float %1483, float %1494, !dbg !7
  %1496 = select i1 %.not, float %1494, float %1484, !dbg !7
  %1497 = select i1 %.not36, float %1495, float %1489, !dbg !7
  %1498 = bitcast float %1497 to i32, !dbg !7
  %1499 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1326, i32 %1498), !dbg !7
  %1500 = bitcast i32 %1499 to float, !dbg !7
  %1501 = select i1 %.not36, float %1489, float %1500, !dbg !7
  %1502 = select i1 %.not36, float %1500, float %1495, !dbg !7
  %1503 = select i1 %.not36, float %1496, float %1490, !dbg !7
  %1504 = bitcast float %1503 to i32, !dbg !7
  %1505 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1326, i32 %1504), !dbg !7
  %1506 = bitcast i32 %1505 to float, !dbg !7
  %1507 = select i1 %.not36, float %1490, float %1506, !dbg !7
  %1508 = select i1 %.not36, float %1506, float %1496, !dbg !7
  %1509 = extractelement <8 x float> %1480, i64 4, !dbg !7
  %1510 = extractelement <8 x float> %1480, i64 5, !dbg !7
  %1511 = extractelement <8 x float> %1480, i64 6, !dbg !7
  %1512 = extractelement <8 x float> %1480, i64 7, !dbg !7
  %1513 = select i1 %.not, float %1510, float %1509, !dbg !7
  %1514 = bitcast float %1513 to i32, !dbg !7
  %1515 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1310, i32 %1514), !dbg !7
  %1516 = bitcast i32 %1515 to float, !dbg !7
  %1517 = select i1 %.not, float %1509, float %1516, !dbg !7
  %1518 = select i1 %.not, float %1516, float %1510, !dbg !7
  %1519 = select i1 %.not, float %1512, float %1511, !dbg !7
  %1520 = bitcast float %1519 to i32, !dbg !7
  %1521 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1310, i32 %1520), !dbg !7
  %1522 = bitcast i32 %1521 to float, !dbg !7
  %1523 = select i1 %.not, float %1511, float %1522, !dbg !7
  %1524 = select i1 %.not, float %1522, float %1512, !dbg !7
  %1525 = select i1 %.not36, float %1523, float %1517, !dbg !7
  %1526 = bitcast float %1525 to i32, !dbg !7
  %1527 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1326, i32 %1526), !dbg !7
  %1528 = bitcast i32 %1527 to float, !dbg !7
  %1529 = select i1 %.not36, float %1517, float %1528, !dbg !7
  %1530 = select i1 %.not36, float %1528, float %1523, !dbg !7
  %1531 = select i1 %.not36, float %1524, float %1518, !dbg !7
  %1532 = bitcast float %1531 to i32, !dbg !7
  %1533 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1326, i32 %1532), !dbg !7
  %1534 = bitcast i32 %1533 to float, !dbg !7
  %1535 = select i1 %.not36, float %1518, float %1534, !dbg !7
  %1536 = select i1 %.not36, float %1534, float %1524, !dbg !7
  %1537 = extractvalue [2 x <8 x float>] %123, 0, !dbg !7
  %1538 = extractelement <8 x float> %1537, i64 0, !dbg !7
  %1539 = extractelement <8 x float> %1537, i64 1, !dbg !7
  %1540 = extractelement <8 x float> %1537, i64 2, !dbg !7
  %1541 = extractelement <8 x float> %1537, i64 3, !dbg !7
  %1542 = select i1 %.not, float %1539, float %1538, !dbg !7
  %1543 = bitcast float %1542 to i32, !dbg !7
  %1544 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1310, i32 %1543), !dbg !7
  %1545 = bitcast i32 %1544 to float, !dbg !7
  %1546 = select i1 %.not, float %1538, float %1545, !dbg !7
  %1547 = select i1 %.not, float %1545, float %1539, !dbg !7
  %1548 = select i1 %.not, float %1541, float %1540, !dbg !7
  %1549 = bitcast float %1548 to i32, !dbg !7
  %1550 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1310, i32 %1549), !dbg !7
  %1551 = bitcast i32 %1550 to float, !dbg !7
  %1552 = select i1 %.not, float %1540, float %1551, !dbg !7
  %1553 = select i1 %.not, float %1551, float %1541, !dbg !7
  %1554 = select i1 %.not36, float %1552, float %1546, !dbg !7
  %1555 = bitcast float %1554 to i32, !dbg !7
  %1556 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1326, i32 %1555), !dbg !7
  %1557 = bitcast i32 %1556 to float, !dbg !7
  %1558 = select i1 %.not36, float %1546, float %1557, !dbg !7
  %1559 = select i1 %.not36, float %1557, float %1552, !dbg !7
  %1560 = select i1 %.not36, float %1553, float %1547, !dbg !7
  %1561 = bitcast float %1560 to i32, !dbg !7
  %1562 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1326, i32 %1561), !dbg !7
  %1563 = bitcast i32 %1562 to float, !dbg !7
  %1564 = select i1 %.not36, float %1547, float %1563, !dbg !7
  %1565 = select i1 %.not36, float %1563, float %1553, !dbg !7
  %1566 = extractelement <8 x float> %1537, i64 4, !dbg !7
  %1567 = extractelement <8 x float> %1537, i64 5, !dbg !7
  %1568 = extractelement <8 x float> %1537, i64 6, !dbg !7
  %1569 = extractelement <8 x float> %1537, i64 7, !dbg !7
  %1570 = select i1 %.not, float %1567, float %1566, !dbg !7
  %1571 = bitcast float %1570 to i32, !dbg !7
  %1572 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1310, i32 %1571), !dbg !7
  %1573 = bitcast i32 %1572 to float, !dbg !7
  %1574 = select i1 %.not, float %1566, float %1573, !dbg !7
  %1575 = select i1 %.not, float %1573, float %1567, !dbg !7
  %1576 = select i1 %.not, float %1569, float %1568, !dbg !7
  %1577 = bitcast float %1576 to i32, !dbg !7
  %1578 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1310, i32 %1577), !dbg !7
  %1579 = bitcast i32 %1578 to float, !dbg !7
  %1580 = select i1 %.not, float %1568, float %1579, !dbg !7
  %1581 = select i1 %.not, float %1579, float %1569, !dbg !7
  %1582 = select i1 %.not36, float %1580, float %1574, !dbg !7
  %1583 = bitcast float %1582 to i32, !dbg !7
  %1584 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1326, i32 %1583), !dbg !7
  %1585 = bitcast i32 %1584 to float, !dbg !7
  %1586 = select i1 %.not36, float %1574, float %1585, !dbg !7
  %1587 = select i1 %.not36, float %1585, float %1580, !dbg !7
  %1588 = select i1 %.not36, float %1581, float %1575, !dbg !7
  %1589 = bitcast float %1588 to i32, !dbg !7
  %1590 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1326, i32 %1589), !dbg !7
  %1591 = bitcast i32 %1590 to float, !dbg !7
  %1592 = select i1 %.not36, float %1575, float %1591, !dbg !7
  %1593 = select i1 %.not36, float %1591, float %1581, !dbg !7
  %1594 = extractvalue [2 x <8 x float>] %123, 1, !dbg !7
  %1595 = extractelement <8 x float> %1594, i64 0, !dbg !7
  %1596 = extractelement <8 x float> %1594, i64 1, !dbg !7
  %1597 = extractelement <8 x float> %1594, i64 2, !dbg !7
  %1598 = extractelement <8 x float> %1594, i64 3, !dbg !7
  %1599 = select i1 %.not, float %1596, float %1595, !dbg !7
  %1600 = bitcast float %1599 to i32, !dbg !7
  %1601 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1310, i32 %1600), !dbg !7
  %1602 = bitcast i32 %1601 to float, !dbg !7
  %1603 = select i1 %.not, float %1595, float %1602, !dbg !7
  %1604 = select i1 %.not, float %1602, float %1596, !dbg !7
  %1605 = select i1 %.not, float %1598, float %1597, !dbg !7
  %1606 = bitcast float %1605 to i32, !dbg !7
  %1607 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1310, i32 %1606), !dbg !7
  %1608 = bitcast i32 %1607 to float, !dbg !7
  %1609 = select i1 %.not, float %1597, float %1608, !dbg !7
  %1610 = select i1 %.not, float %1608, float %1598, !dbg !7
  %1611 = select i1 %.not36, float %1609, float %1603, !dbg !7
  %1612 = bitcast float %1611 to i32, !dbg !7
  %1613 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1326, i32 %1612), !dbg !7
  %1614 = bitcast i32 %1613 to float, !dbg !7
  %1615 = select i1 %.not36, float %1603, float %1614, !dbg !7
  %1616 = select i1 %.not36, float %1614, float %1609, !dbg !7
  %1617 = select i1 %.not36, float %1610, float %1604, !dbg !7
  %1618 = bitcast float %1617 to i32, !dbg !7
  %1619 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1326, i32 %1618), !dbg !7
  %1620 = bitcast i32 %1619 to float, !dbg !7
  %1621 = select i1 %.not36, float %1604, float %1620, !dbg !7
  %1622 = select i1 %.not36, float %1620, float %1610, !dbg !7
  %1623 = extractelement <8 x float> %1594, i64 4, !dbg !7
  %1624 = extractelement <8 x float> %1594, i64 5, !dbg !7
  %1625 = extractelement <8 x float> %1594, i64 6, !dbg !7
  %1626 = extractelement <8 x float> %1594, i64 7, !dbg !7
  %1627 = select i1 %.not, float %1624, float %1623, !dbg !7
  %1628 = bitcast float %1627 to i32, !dbg !7
  %1629 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1310, i32 %1628), !dbg !7
  %1630 = bitcast i32 %1629 to float, !dbg !7
  %1631 = select i1 %.not, float %1623, float %1630, !dbg !7
  %1632 = select i1 %.not, float %1630, float %1624, !dbg !7
  %1633 = select i1 %.not, float %1626, float %1625, !dbg !7
  %1634 = bitcast float %1633 to i32, !dbg !7
  %1635 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1310, i32 %1634), !dbg !7
  %1636 = bitcast i32 %1635 to float, !dbg !7
  %1637 = select i1 %.not, float %1625, float %1636, !dbg !7
  %1638 = select i1 %.not, float %1636, float %1626, !dbg !7
  %1639 = select i1 %.not36, float %1637, float %1631, !dbg !7
  %1640 = bitcast float %1639 to i32, !dbg !7
  %1641 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1326, i32 %1640), !dbg !7
  %1642 = bitcast i32 %1641 to float, !dbg !7
  %1643 = select i1 %.not36, float %1631, float %1642, !dbg !7
  %1644 = select i1 %.not36, float %1642, float %1637, !dbg !7
  %1645 = select i1 %.not36, float %1638, float %1632, !dbg !7
  %1646 = bitcast float %1645 to i32, !dbg !7
  %1647 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1326, i32 %1646), !dbg !7
  %1648 = bitcast i32 %1647 to float, !dbg !7
  %1649 = select i1 %.not36, float %1632, float %1648, !dbg !7
  %1650 = select i1 %.not36, float %1648, float %1638, !dbg !7
  %1651 = extractvalue [2 x <8 x float>] %120, 0, !dbg !7
  %1652 = extractelement <8 x float> %1651, i64 0, !dbg !7
  %1653 = extractelement <8 x float> %1651, i64 1, !dbg !7
  %1654 = extractelement <8 x float> %1651, i64 2, !dbg !7
  %1655 = extractelement <8 x float> %1651, i64 3, !dbg !7
  %1656 = select i1 %.not, float %1653, float %1652, !dbg !7
  %1657 = bitcast float %1656 to i32, !dbg !7
  %1658 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1310, i32 %1657), !dbg !7
  %1659 = bitcast i32 %1658 to float, !dbg !7
  %1660 = select i1 %.not, float %1652, float %1659, !dbg !7
  %1661 = select i1 %.not, float %1659, float %1653, !dbg !7
  %1662 = select i1 %.not, float %1655, float %1654, !dbg !7
  %1663 = bitcast float %1662 to i32, !dbg !7
  %1664 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1310, i32 %1663), !dbg !7
  %1665 = bitcast i32 %1664 to float, !dbg !7
  %1666 = select i1 %.not, float %1654, float %1665, !dbg !7
  %1667 = select i1 %.not, float %1665, float %1655, !dbg !7
  %1668 = select i1 %.not36, float %1666, float %1660, !dbg !7
  %1669 = bitcast float %1668 to i32, !dbg !7
  %1670 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1326, i32 %1669), !dbg !7
  %1671 = select i1 %.not36, float %1667, float %1661, !dbg !7
  %1672 = bitcast float %1671 to i32, !dbg !7
  %1673 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1326, i32 %1672), !dbg !7
  %1674 = extractelement <8 x float> %1651, i64 4, !dbg !7
  %1675 = extractelement <8 x float> %1651, i64 5, !dbg !7
  %1676 = extractelement <8 x float> %1651, i64 6, !dbg !7
  %1677 = extractelement <8 x float> %1651, i64 7, !dbg !7
  %1678 = select i1 %.not, float %1675, float %1674, !dbg !7
  %1679 = bitcast float %1678 to i32, !dbg !7
  %1680 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1310, i32 %1679), !dbg !7
  %1681 = bitcast i32 %1680 to float, !dbg !7
  %1682 = select i1 %.not, float %1674, float %1681, !dbg !7
  %1683 = select i1 %.not, float %1681, float %1675, !dbg !7
  %1684 = select i1 %.not, float %1677, float %1676, !dbg !7
  %1685 = bitcast float %1684 to i32, !dbg !7
  %1686 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1310, i32 %1685), !dbg !7
  %1687 = bitcast i32 %1686 to float, !dbg !7
  %1688 = select i1 %.not, float %1676, float %1687, !dbg !7
  %1689 = select i1 %.not, float %1687, float %1677, !dbg !7
  %1690 = select i1 %.not36, float %1688, float %1682, !dbg !7
  %1691 = bitcast float %1690 to i32, !dbg !7
  %1692 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1326, i32 %1691), !dbg !7
  %1693 = select i1 %.not36, float %1689, float %1683, !dbg !7
  %1694 = bitcast float %1693 to i32, !dbg !7
  %1695 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1326, i32 %1694), !dbg !7
  %1696 = extractvalue [2 x <8 x float>] %120, 1, !dbg !7
  %1697 = extractelement <8 x float> %1696, i64 0, !dbg !7
  %1698 = extractelement <8 x float> %1696, i64 1, !dbg !7
  %1699 = extractelement <8 x float> %1696, i64 2, !dbg !7
  %1700 = extractelement <8 x float> %1696, i64 3, !dbg !7
  %1701 = select i1 %.not, float %1698, float %1697, !dbg !7
  %1702 = bitcast float %1701 to i32, !dbg !7
  %1703 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1310, i32 %1702), !dbg !7
  %1704 = bitcast i32 %1703 to float, !dbg !7
  %1705 = select i1 %.not, float %1697, float %1704, !dbg !7
  %1706 = select i1 %.not, float %1704, float %1698, !dbg !7
  %1707 = select i1 %.not, float %1700, float %1699, !dbg !7
  %1708 = bitcast float %1707 to i32, !dbg !7
  %1709 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1310, i32 %1708), !dbg !7
  %1710 = bitcast i32 %1709 to float, !dbg !7
  %1711 = select i1 %.not, float %1699, float %1710, !dbg !7
  %1712 = select i1 %.not, float %1710, float %1700, !dbg !7
  %1713 = select i1 %.not36, float %1711, float %1705, !dbg !7
  %1714 = bitcast float %1713 to i32, !dbg !7
  %1715 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1326, i32 %1714), !dbg !7
  %1716 = select i1 %.not36, float %1712, float %1706, !dbg !7
  %1717 = bitcast float %1716 to i32, !dbg !7
  %1718 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1326, i32 %1717), !dbg !7
  %1719 = extractelement <8 x float> %1696, i64 4, !dbg !7
  %1720 = extractelement <8 x float> %1696, i64 5, !dbg !7
  %1721 = extractelement <8 x float> %1696, i64 6, !dbg !7
  %1722 = extractelement <8 x float> %1696, i64 7, !dbg !7
  %1723 = select i1 %.not, float %1720, float %1719, !dbg !7
  %1724 = bitcast float %1723 to i32, !dbg !7
  %1725 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1310, i32 %1724), !dbg !7
  %1726 = bitcast i32 %1725 to float, !dbg !7
  %1727 = select i1 %.not, float %1719, float %1726, !dbg !7
  %1728 = select i1 %.not, float %1726, float %1720, !dbg !7
  %1729 = select i1 %.not, float %1722, float %1721, !dbg !7
  %1730 = bitcast float %1729 to i32, !dbg !7
  %1731 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1310, i32 %1730), !dbg !7
  %1732 = bitcast i32 %1731 to float, !dbg !7
  %1733 = select i1 %.not, float %1721, float %1732, !dbg !7
  %1734 = select i1 %.not, float %1732, float %1722, !dbg !7
  %1735 = select i1 %.not36, float %1733, float %1727, !dbg !7
  %1736 = bitcast float %1735 to i32, !dbg !7
  %1737 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1326, i32 %1736), !dbg !7
  %1738 = select i1 %.not36, float %1734, float %1728, !dbg !7
  %1739 = bitcast float %1738 to i32, !dbg !7
  %1740 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1326, i32 %1739), !dbg !7
  %1741 = shufflevector <1 x float> %1296, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %1742 = shufflevector <1 x float> %1291, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %1743 = bitcast i32 %1737 to float, !dbg !7
  %1744 = select i1 %.not36, float %1727, float %1743, !dbg !7
  %1745 = insertelement <4 x float> poison, float %1744, i64 0, !dbg !19
  %1746 = bitcast i32 %1740 to float, !dbg !7
  %1747 = select i1 %.not36, float %1728, float %1746, !dbg !7
  %1748 = insertelement <4 x float> %1745, float %1747, i64 1, !dbg !19
  %1749 = select i1 %.not36, float %1743, float %1733, !dbg !7
  %1750 = insertelement <4 x float> %1748, float %1749, i64 2, !dbg !19
  %1751 = select i1 %.not36, float %1746, float %1734, !dbg !7
  %1752 = insertelement <4 x float> %1750, float %1751, i64 3, !dbg !19
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1742, !dbg !19
  %normalized = fmul <4 x float> %1752, %row.reciprocal, !dbg !19
  %1753 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !20
  %1754 = extractelement <4 x half> %1753, i64 3, !dbg !20
  %1755 = extractelement <4 x half> %1753, i64 2, !dbg !20
  %1756 = extractelement <4 x half> %1753, i64 1, !dbg !20
  %1757 = extractelement <4 x half> %1753, i64 0, !dbg !20
  %1758 = bitcast i32 %1715 to float, !dbg !7
  %1759 = select i1 %.not36, float %1705, float %1758, !dbg !7
  %1760 = insertelement <4 x float> poison, float %1759, i64 0, !dbg !19
  %1761 = bitcast i32 %1718 to float, !dbg !7
  %1762 = select i1 %.not36, float %1706, float %1761, !dbg !7
  %1763 = insertelement <4 x float> %1760, float %1762, i64 1, !dbg !19
  %1764 = select i1 %.not36, float %1758, float %1711, !dbg !7
  %1765 = insertelement <4 x float> %1763, float %1764, i64 2, !dbg !19
  %1766 = select i1 %.not36, float %1761, float %1712, !dbg !7
  %1767 = insertelement <4 x float> %1765, float %1766, i64 3, !dbg !19
  %normalized104 = fmul <4 x float> %1767, %row.reciprocal, !dbg !19
  %1768 = fptrunc <4 x float> %normalized104 to <4 x half>, !dbg !20
  %1769 = extractelement <4 x half> %1768, i64 3, !dbg !20
  %1770 = extractelement <4 x half> %1768, i64 2, !dbg !20
  %1771 = extractelement <4 x half> %1768, i64 1, !dbg !20
  %1772 = extractelement <4 x half> %1768, i64 0, !dbg !20
  %1773 = bitcast i32 %1692 to float, !dbg !7
  %1774 = select i1 %.not36, float %1682, float %1773, !dbg !7
  %1775 = insertelement <4 x float> poison, float %1774, i64 0, !dbg !19
  %1776 = bitcast i32 %1695 to float, !dbg !7
  %1777 = select i1 %.not36, float %1683, float %1776, !dbg !7
  %1778 = insertelement <4 x float> %1775, float %1777, i64 1, !dbg !19
  %1779 = select i1 %.not36, float %1773, float %1688, !dbg !7
  %1780 = insertelement <4 x float> %1778, float %1779, i64 2, !dbg !19
  %1781 = select i1 %.not36, float %1776, float %1689, !dbg !7
  %1782 = insertelement <4 x float> %1780, float %1781, i64 3, !dbg !19
  %row.reciprocal105 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1741, !dbg !19
  %normalized106 = fmul <4 x float> %1782, %row.reciprocal105, !dbg !19
  %1783 = fptrunc <4 x float> %normalized106 to <4 x half>, !dbg !20
  %1784 = extractelement <4 x half> %1783, i64 3, !dbg !20
  %1785 = extractelement <4 x half> %1783, i64 2, !dbg !20
  %1786 = extractelement <4 x half> %1783, i64 1, !dbg !20
  %1787 = extractelement <4 x half> %1783, i64 0, !dbg !20
  %1788 = bitcast i32 %1670 to float, !dbg !7
  %1789 = select i1 %.not36, float %1660, float %1788, !dbg !7
  %1790 = insertelement <4 x float> poison, float %1789, i64 0, !dbg !19
  %1791 = bitcast i32 %1673 to float, !dbg !7
  %1792 = select i1 %.not36, float %1661, float %1791, !dbg !7
  %1793 = insertelement <4 x float> %1790, float %1792, i64 1, !dbg !19
  %1794 = select i1 %.not36, float %1788, float %1666, !dbg !7
  %1795 = insertelement <4 x float> %1793, float %1794, i64 2, !dbg !19
  %1796 = select i1 %.not36, float %1791, float %1667, !dbg !7
  %1797 = insertelement <4 x float> %1795, float %1796, i64 3, !dbg !19
  %normalized107 = fmul <4 x float> %1797, %row.reciprocal105, !dbg !19
  %1798 = fptrunc <4 x float> %normalized107 to <4 x half>, !dbg !20
  %1799 = extractelement <4 x half> %1798, i64 3, !dbg !20
  %1800 = extractelement <4 x half> %1798, i64 2, !dbg !20
  %1801 = extractelement <4 x half> %1798, i64 1, !dbg !20
  %1802 = extractelement <4 x half> %1798, i64 0, !dbg !20
  %address.thread1041 = call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #9, !dbg !21, !range !35, !frisk.register_lifetimes !1
  %address.local1042 = zext nneg i32 %address.thread1041 to i64, !dbg !8
  %address.local1043 = add i64 %30, %address.local1042, !dbg !21
  %address.local1044 = lshr i64 %address.local1042, 2, !dbg !18
  %address.local1045 = and i64 %address.local1044, 48, !dbg !18
  %address.local1046 = add i64 %address.local1043, %address.local1045, !dbg !21
  %address.local1047 = and i64 %address.local1042, 240, !dbg !21
  %address.local1049 = sub i64 %address.local1046, %address.local1047, !dbg !21
  %address.local1050 = add i64 %address.local1049, %31, !dbg !21
  %address.local1051 = shl nuw nsw i64 %address.local1050, 7, !dbg !21
  %1803 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %45, !dbg !21
  %address.local1053 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %1803, i64 %address.local1051, !dbg !21
  %address.local1055 = and i64 %address.local1044, 12, !dbg !12
  %address.local1056 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1055, !dbg !21
  store half %1802, ptr addrspace(1) %address.local1056, align 2, !dbg !21
  %address.local1058 = or disjoint i64 %address.local1055, 1, !dbg !21
  %address.local1059 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1058, !dbg !21
  store half %1801, ptr addrspace(1) %address.local1059, align 2, !dbg !21
  %address.local1061 = or disjoint i64 %address.local1055, 2, !dbg !21
  %address.local1062 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1061, !dbg !21
  store half %1800, ptr addrspace(1) %address.local1062, align 2, !dbg !21
  %address.local1064 = or disjoint i64 %address.local1055, 3, !dbg !21
  %address.local1065 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1064, !dbg !21
  store half %1799, ptr addrspace(1) %address.local1065, align 2, !dbg !21
  %address.local1067 = or disjoint i64 %address.local1055, 16, !dbg !21
  %address.local1068 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1067, !dbg !21
  store half %1787, ptr addrspace(1) %address.local1068, align 2, !dbg !21
  %address.local1070 = or disjoint i64 %address.local1055, 17, !dbg !21
  %address.local1071 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1070, !dbg !21
  store half %1786, ptr addrspace(1) %address.local1071, align 2, !dbg !21
  %address.local1073 = or disjoint i64 %address.local1055, 18, !dbg !21
  %address.local1074 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1073, !dbg !21
  store half %1785, ptr addrspace(1) %address.local1074, align 2, !dbg !21
  %address.local1076 = or disjoint i64 %address.local1055, 19, !dbg !21
  %address.local1077 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1076, !dbg !21
  store half %1784, ptr addrspace(1) %address.local1077, align 2, !dbg !21
  %address.local1081 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local1053, i64 16384, !dbg !21
  %address.local1082 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1055, !dbg !21
  store half %1772, ptr addrspace(1) %address.local1082, align 2, !dbg !21
  %address.local1084 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1058, !dbg !21
  store half %1771, ptr addrspace(1) %address.local1084, align 2, !dbg !21
  %address.local1086 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1061, !dbg !21
  store half %1770, ptr addrspace(1) %address.local1086, align 2, !dbg !21
  %address.local1088 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1064, !dbg !21
  store half %1769, ptr addrspace(1) %address.local1088, align 2, !dbg !21
  %address.local1090 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1067, !dbg !21
  store half %1757, ptr addrspace(1) %address.local1090, align 2, !dbg !21
  %address.local1092 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1070, !dbg !21
  store half %1756, ptr addrspace(1) %address.local1092, align 2, !dbg !21
  %address.local1094 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1073, !dbg !21
  store half %1755, ptr addrspace(1) %address.local1094, align 2, !dbg !21
  %address.local1096 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1076, !dbg !21
  store half %1754, ptr addrspace(1) %address.local1096, align 2, !dbg !21
  %1804 = insertelement <4 x float> poison, float %1643, i64 0, !dbg !19
  %1805 = insertelement <4 x float> %1804, float %1649, i64 1, !dbg !19
  %1806 = insertelement <4 x float> %1805, float %1644, i64 2, !dbg !19
  %1807 = insertelement <4 x float> %1806, float %1650, i64 3, !dbg !19
  %normalized109 = fmul <4 x float> %1807, %row.reciprocal, !dbg !19
  %1808 = fptrunc <4 x float> %normalized109 to <4 x half>, !dbg !20
  %1809 = extractelement <4 x half> %1808, i64 3, !dbg !20
  %1810 = extractelement <4 x half> %1808, i64 2, !dbg !20
  %1811 = extractelement <4 x half> %1808, i64 1, !dbg !20
  %1812 = extractelement <4 x half> %1808, i64 0, !dbg !20
  %1813 = insertelement <4 x float> poison, float %1615, i64 0, !dbg !19
  %1814 = insertelement <4 x float> %1813, float %1621, i64 1, !dbg !19
  %1815 = insertelement <4 x float> %1814, float %1616, i64 2, !dbg !19
  %1816 = insertelement <4 x float> %1815, float %1622, i64 3, !dbg !19
  %normalized110 = fmul <4 x float> %1816, %row.reciprocal, !dbg !19
  %1817 = fptrunc <4 x float> %normalized110 to <4 x half>, !dbg !20
  %1818 = extractelement <4 x half> %1817, i64 3, !dbg !20
  %1819 = extractelement <4 x half> %1817, i64 2, !dbg !20
  %1820 = extractelement <4 x half> %1817, i64 1, !dbg !20
  %1821 = extractelement <4 x half> %1817, i64 0, !dbg !20
  %1822 = insertelement <4 x float> poison, float %1586, i64 0, !dbg !19
  %1823 = insertelement <4 x float> %1822, float %1592, i64 1, !dbg !19
  %1824 = insertelement <4 x float> %1823, float %1587, i64 2, !dbg !19
  %1825 = insertelement <4 x float> %1824, float %1593, i64 3, !dbg !19
  %normalized112 = fmul <4 x float> %1825, %row.reciprocal105, !dbg !19
  %1826 = fptrunc <4 x float> %normalized112 to <4 x half>, !dbg !20
  %1827 = extractelement <4 x half> %1826, i64 3, !dbg !20
  %1828 = extractelement <4 x half> %1826, i64 2, !dbg !20
  %1829 = extractelement <4 x half> %1826, i64 1, !dbg !20
  %1830 = extractelement <4 x half> %1826, i64 0, !dbg !20
  %1831 = insertelement <4 x float> poison, float %1558, i64 0, !dbg !19
  %1832 = insertelement <4 x float> %1831, float %1564, i64 1, !dbg !19
  %1833 = insertelement <4 x float> %1832, float %1559, i64 2, !dbg !19
  %1834 = insertelement <4 x float> %1833, float %1565, i64 3, !dbg !19
  %normalized113 = fmul <4 x float> %1834, %row.reciprocal105, !dbg !19
  %1835 = fptrunc <4 x float> %normalized113 to <4 x half>, !dbg !20
  %1836 = extractelement <4 x half> %1835, i64 3, !dbg !20
  %1837 = extractelement <4 x half> %1835, i64 2, !dbg !20
  %1838 = extractelement <4 x half> %1835, i64 1, !dbg !20
  %1839 = extractelement <4 x half> %1835, i64 0, !dbg !20
  %address.local1112 = or disjoint i64 %address.local1055, 32, !dbg !21
  %address.local1113 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1112, !dbg !21
  store half %1839, ptr addrspace(1) %address.local1113, align 2, !dbg !21
  %address.local1115 = or disjoint i64 %address.local1055, 33, !dbg !21
  %address.local1116 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1115, !dbg !21
  store half %1838, ptr addrspace(1) %address.local1116, align 2, !dbg !21
  %address.local1118 = or disjoint i64 %address.local1055, 34, !dbg !21
  %address.local1119 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1118, !dbg !21
  store half %1837, ptr addrspace(1) %address.local1119, align 2, !dbg !21
  %address.local1121 = or disjoint i64 %address.local1055, 35, !dbg !21
  %address.local1122 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1121, !dbg !21
  store half %1836, ptr addrspace(1) %address.local1122, align 2, !dbg !21
  %address.local1124 = or disjoint i64 %address.local1055, 48, !dbg !21
  %address.local1125 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1124, !dbg !21
  store half %1830, ptr addrspace(1) %address.local1125, align 2, !dbg !21
  %address.local1127 = or disjoint i64 %address.local1055, 49, !dbg !21
  %address.local1128 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1127, !dbg !21
  store half %1829, ptr addrspace(1) %address.local1128, align 2, !dbg !21
  %address.local1130 = or disjoint i64 %address.local1055, 50, !dbg !21
  %address.local1131 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1130, !dbg !21
  store half %1828, ptr addrspace(1) %address.local1131, align 2, !dbg !21
  %address.local1133 = or i64 %address.local1044, 51, !dbg !21
  %address.local1134 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1133, !dbg !21
  store half %1827, ptr addrspace(1) %address.local1134, align 2, !dbg !21
  %address.local1139 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1112, !dbg !21
  store half %1821, ptr addrspace(1) %address.local1139, align 2, !dbg !21
  %address.local1141 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1115, !dbg !21
  store half %1820, ptr addrspace(1) %address.local1141, align 2, !dbg !21
  %address.local1143 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1118, !dbg !21
  store half %1819, ptr addrspace(1) %address.local1143, align 2, !dbg !21
  %address.local1145 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1121, !dbg !21
  store half %1818, ptr addrspace(1) %address.local1145, align 2, !dbg !21
  %address.local1147 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1124, !dbg !21
  store half %1812, ptr addrspace(1) %address.local1147, align 2, !dbg !21
  %address.local1149 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1127, !dbg !21
  store half %1811, ptr addrspace(1) %address.local1149, align 2, !dbg !21
  %address.local1151 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1130, !dbg !21
  store half %1810, ptr addrspace(1) %address.local1151, align 2, !dbg !21
  %address.local1153 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1133, !dbg !21
  store half %1809, ptr addrspace(1) %address.local1153, align 2, !dbg !21
  %1840 = insertelement <4 x float> poison, float %1529, i64 0, !dbg !19
  %1841 = insertelement <4 x float> %1840, float %1535, i64 1, !dbg !19
  %1842 = insertelement <4 x float> %1841, float %1530, i64 2, !dbg !19
  %1843 = insertelement <4 x float> %1842, float %1536, i64 3, !dbg !19
  %normalized115 = fmul <4 x float> %1843, %row.reciprocal, !dbg !19
  %1844 = fptrunc <4 x float> %normalized115 to <4 x half>, !dbg !20
  %1845 = extractelement <4 x half> %1844, i64 3, !dbg !20
  %1846 = extractelement <4 x half> %1844, i64 2, !dbg !20
  %1847 = extractelement <4 x half> %1844, i64 1, !dbg !20
  %1848 = extractelement <4 x half> %1844, i64 0, !dbg !20
  %1849 = insertelement <4 x float> poison, float %1501, i64 0, !dbg !19
  %1850 = insertelement <4 x float> %1849, float %1507, i64 1, !dbg !19
  %1851 = insertelement <4 x float> %1850, float %1502, i64 2, !dbg !19
  %1852 = insertelement <4 x float> %1851, float %1508, i64 3, !dbg !19
  %normalized116 = fmul <4 x float> %1852, %row.reciprocal, !dbg !19
  %1853 = fptrunc <4 x float> %normalized116 to <4 x half>, !dbg !20
  %1854 = extractelement <4 x half> %1853, i64 3, !dbg !20
  %1855 = extractelement <4 x half> %1853, i64 2, !dbg !20
  %1856 = extractelement <4 x half> %1853, i64 1, !dbg !20
  %1857 = extractelement <4 x half> %1853, i64 0, !dbg !20
  %1858 = insertelement <4 x float> poison, float %1472, i64 0, !dbg !19
  %1859 = insertelement <4 x float> %1858, float %1478, i64 1, !dbg !19
  %1860 = insertelement <4 x float> %1859, float %1473, i64 2, !dbg !19
  %1861 = insertelement <4 x float> %1860, float %1479, i64 3, !dbg !19
  %normalized118 = fmul <4 x float> %1861, %row.reciprocal105, !dbg !19
  %1862 = fptrunc <4 x float> %normalized118 to <4 x half>, !dbg !20
  %1863 = extractelement <4 x half> %1862, i64 3, !dbg !20
  %1864 = extractelement <4 x half> %1862, i64 2, !dbg !20
  %1865 = extractelement <4 x half> %1862, i64 1, !dbg !20
  %1866 = extractelement <4 x half> %1862, i64 0, !dbg !20
  %1867 = insertelement <4 x float> poison, float %1444, i64 0, !dbg !19
  %1868 = insertelement <4 x float> %1867, float %1450, i64 1, !dbg !19
  %1869 = insertelement <4 x float> %1868, float %1445, i64 2, !dbg !19
  %1870 = insertelement <4 x float> %1869, float %1451, i64 3, !dbg !19
  %normalized119 = fmul <4 x float> %1870, %row.reciprocal105, !dbg !19
  %1871 = fptrunc <4 x float> %normalized119 to <4 x half>, !dbg !20
  %1872 = extractelement <4 x half> %1871, i64 3, !dbg !20
  %1873 = extractelement <4 x half> %1871, i64 2, !dbg !20
  %1874 = extractelement <4 x half> %1871, i64 1, !dbg !20
  %1875 = extractelement <4 x half> %1871, i64 0, !dbg !20
  %address.local1169 = or disjoint i64 %address.local1055, 64, !dbg !21
  %address.local1170 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1169, !dbg !21
  store half %1875, ptr addrspace(1) %address.local1170, align 2, !dbg !21
  %address.local1172 = or disjoint i64 %address.local1055, 65, !dbg !21
  %address.local1173 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1172, !dbg !21
  store half %1874, ptr addrspace(1) %address.local1173, align 2, !dbg !21
  %address.local1175 = or disjoint i64 %address.local1055, 66, !dbg !21
  %address.local1176 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1175, !dbg !21
  store half %1873, ptr addrspace(1) %address.local1176, align 2, !dbg !21
  %address.local1178 = or disjoint i64 %address.local1055, 67, !dbg !21
  %address.local1179 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1178, !dbg !21
  store half %1872, ptr addrspace(1) %address.local1179, align 2, !dbg !21
  %address.local1181 = or disjoint i64 %address.local1055, 80, !dbg !21
  %address.local1182 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1181, !dbg !21
  store half %1866, ptr addrspace(1) %address.local1182, align 2, !dbg !21
  %address.local1184 = or disjoint i64 %address.local1055, 81, !dbg !21
  %address.local1185 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1184, !dbg !21
  store half %1865, ptr addrspace(1) %address.local1185, align 2, !dbg !21
  %address.local1187 = or disjoint i64 %address.local1055, 82, !dbg !21
  %address.local1188 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1187, !dbg !21
  store half %1864, ptr addrspace(1) %address.local1188, align 2, !dbg !21
  %address.local1190 = or disjoint i64 %address.local1055, 83, !dbg !21
  %address.local1191 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1190, !dbg !21
  store half %1863, ptr addrspace(1) %address.local1191, align 2, !dbg !21
  %address.local1196 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1169, !dbg !21
  store half %1857, ptr addrspace(1) %address.local1196, align 2, !dbg !21
  %address.local1198 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1172, !dbg !21
  store half %1856, ptr addrspace(1) %address.local1198, align 2, !dbg !21
  %address.local1200 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1175, !dbg !21
  store half %1855, ptr addrspace(1) %address.local1200, align 2, !dbg !21
  %address.local1202 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1178, !dbg !21
  store half %1854, ptr addrspace(1) %address.local1202, align 2, !dbg !21
  %address.local1204 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1181, !dbg !21
  store half %1848, ptr addrspace(1) %address.local1204, align 2, !dbg !21
  %address.local1206 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1184, !dbg !21
  store half %1847, ptr addrspace(1) %address.local1206, align 2, !dbg !21
  %address.local1208 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1187, !dbg !21
  store half %1846, ptr addrspace(1) %address.local1208, align 2, !dbg !21
  %address.local1210 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1190, !dbg !21
  store half %1845, ptr addrspace(1) %address.local1210, align 2, !dbg !21
  %1876 = insertelement <4 x float> poison, float %1415, i64 0, !dbg !19
  %1877 = insertelement <4 x float> %1876, float %1421, i64 1, !dbg !19
  %1878 = insertelement <4 x float> %1877, float %1416, i64 2, !dbg !19
  %1879 = insertelement <4 x float> %1878, float %1422, i64 3, !dbg !19
  %normalized121 = fmul <4 x float> %1879, %row.reciprocal, !dbg !19
  %1880 = fptrunc <4 x float> %normalized121 to <4 x half>, !dbg !20
  %1881 = extractelement <4 x half> %1880, i64 3, !dbg !20
  %1882 = extractelement <4 x half> %1880, i64 2, !dbg !20
  %1883 = extractelement <4 x half> %1880, i64 1, !dbg !20
  %1884 = extractelement <4 x half> %1880, i64 0, !dbg !20
  %1885 = insertelement <4 x float> poison, float %1387, i64 0, !dbg !19
  %1886 = insertelement <4 x float> %1885, float %1393, i64 1, !dbg !19
  %1887 = insertelement <4 x float> %1886, float %1388, i64 2, !dbg !19
  %1888 = insertelement <4 x float> %1887, float %1394, i64 3, !dbg !19
  %normalized122 = fmul <4 x float> %1888, %row.reciprocal, !dbg !19
  %1889 = fptrunc <4 x float> %normalized122 to <4 x half>, !dbg !20
  %1890 = extractelement <4 x half> %1889, i64 3, !dbg !20
  %1891 = extractelement <4 x half> %1889, i64 2, !dbg !20
  %1892 = extractelement <4 x half> %1889, i64 1, !dbg !20
  %1893 = extractelement <4 x half> %1889, i64 0, !dbg !20
  %1894 = insertelement <4 x float> poison, float %1358, i64 0, !dbg !19
  %1895 = insertelement <4 x float> %1894, float %1364, i64 1, !dbg !19
  %1896 = insertelement <4 x float> %1895, float %1359, i64 2, !dbg !19
  %1897 = insertelement <4 x float> %1896, float %1365, i64 3, !dbg !19
  %normalized124 = fmul <4 x float> %1897, %row.reciprocal105, !dbg !19
  %1898 = fptrunc <4 x float> %normalized124 to <4 x half>, !dbg !20
  %1899 = extractelement <4 x half> %1898, i64 3, !dbg !20
  %1900 = extractelement <4 x half> %1898, i64 2, !dbg !20
  %1901 = extractelement <4 x half> %1898, i64 1, !dbg !20
  %1902 = extractelement <4 x half> %1898, i64 0, !dbg !20
  %1903 = insertelement <4 x float> poison, float %1330, i64 0, !dbg !19
  %1904 = insertelement <4 x float> %1903, float %1336, i64 1, !dbg !19
  %1905 = insertelement <4 x float> %1904, float %1331, i64 2, !dbg !19
  %1906 = insertelement <4 x float> %1905, float %1337, i64 3, !dbg !19
  %normalized125 = fmul <4 x float> %1906, %row.reciprocal105, !dbg !19
  %1907 = fptrunc <4 x float> %normalized125 to <4 x half>, !dbg !20
  %1908 = extractelement <4 x half> %1907, i64 3, !dbg !20
  %1909 = extractelement <4 x half> %1907, i64 2, !dbg !20
  %1910 = extractelement <4 x half> %1907, i64 1, !dbg !20
  %1911 = extractelement <4 x half> %1907, i64 0, !dbg !20
  %address.local1226 = or disjoint i64 %address.local1055, 96, !dbg !21
  %address.local1227 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1226, !dbg !21
  store half %1911, ptr addrspace(1) %address.local1227, align 2, !dbg !21
  %address.local1229 = or disjoint i64 %address.local1055, 97, !dbg !21
  %address.local1230 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1229, !dbg !21
  store half %1910, ptr addrspace(1) %address.local1230, align 2, !dbg !21
  %address.local1232 = or disjoint i64 %address.local1055, 98, !dbg !21
  %address.local1233 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1232, !dbg !21
  store half %1909, ptr addrspace(1) %address.local1233, align 2, !dbg !21
  %address.local1235 = or disjoint i64 %address.local1055, 99, !dbg !21
  %address.local1236 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1235, !dbg !21
  store half %1908, ptr addrspace(1) %address.local1236, align 2, !dbg !21
  %address.local1238 = or disjoint i64 %address.local1055, 112, !dbg !21
  %address.local1239 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1238, !dbg !21
  store half %1902, ptr addrspace(1) %address.local1239, align 2, !dbg !21
  %address.local1241 = or disjoint i64 %address.local1055, 113, !dbg !21
  %address.local1242 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1241, !dbg !21
  store half %1901, ptr addrspace(1) %address.local1242, align 2, !dbg !21
  %address.local1244 = or disjoint i64 %address.local1055, 114, !dbg !21
  %address.local1245 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1244, !dbg !21
  store half %1900, ptr addrspace(1) %address.local1245, align 2, !dbg !21
  %address.local1247 = or i64 %address.local1044, 115, !dbg !21
  %address.local1248 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1247, !dbg !21
  store half %1899, ptr addrspace(1) %address.local1248, align 2, !dbg !21
  %address.local1253 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1226, !dbg !21
  store half %1893, ptr addrspace(1) %address.local1253, align 2, !dbg !21
  %address.local1255 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1229, !dbg !21
  store half %1892, ptr addrspace(1) %address.local1255, align 2, !dbg !21
  %address.local1257 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1232, !dbg !21
  store half %1891, ptr addrspace(1) %address.local1257, align 2, !dbg !21
  %address.local1259 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1235, !dbg !21
  store half %1890, ptr addrspace(1) %address.local1259, align 2, !dbg !21
  %address.local1261 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1238, !dbg !21
  store half %1884, ptr addrspace(1) %address.local1261, align 2, !dbg !21
  %address.local1263 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1241, !dbg !21
  store half %1883, ptr addrspace(1) %address.local1263, align 2, !dbg !21
  %address.local1265 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1244, !dbg !21
  store half %1882, ptr addrspace(1) %address.local1265, align 2, !dbg !21
  %address.local1267 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1247, !dbg !21
  store half %1881, ptr addrspace(1) %address.local1267, align 2, !dbg !21
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
!llvm.module.flags = !{!34}

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
!19 = !DILocation(line: 107, column: 23, scope: !4)
!20 = !DILocation(line: 108, column: 19, scope: !4)
!21 = !DILocation(line: 109, column: 9, scope: !4)
!22 = !DILocation(line: 74, column: 23, scope: !4)
!23 = !DILocation(line: 89, column: 20, scope: !4)
!24 = !DILocation(line: 90, column: 22, scope: !4)
!25 = !DILocation(line: 91, column: 23, scope: !4)
!26 = !DILocation(line: 92, column: 20, scope: !4)
!27 = !DILocation(line: 93, column: 22, scope: !4)
!28 = !DILocation(line: 94, column: 22, scope: !4)
!29 = !DILocation(line: 95, column: 20, scope: !4)
!30 = !DILocation(line: 96, column: 16, scope: !4)
!31 = !DILocation(line: 103, column: 22, scope: !4)
!33 = !{i32 256, i32 1, i32 1}
!34 = !{i32 2, !"Debug Info Version", i32 3}
!35 = !{i32 0, i32 256}
!36 = !{i32 4}
!37 = !{i64 8192}
!38 = !{i32 0}
!39 = !{i32 1}
!40 = !{i32 2}
!41 = !{i32 3}
!42 = !{i32 5}
!43 = !{i32 6}
!44 = !{i32 7}
!45 = !{i32 8}
!46 = !{i32 9}
!47 = !{i32 10}
!48 = !{i32 11}
!49 = !{i32 12}
!50 = !{i32 13}
!51 = !{i32 14}
!52 = !{i32 15}
!53 = !{i32 16}
!54 = !{i32 17}
!55 = !{i32 18}
!56 = !{i32 19}
!57 = !{i32 20}
!58 = !{i32 21}
!59 = !{i32 22}
