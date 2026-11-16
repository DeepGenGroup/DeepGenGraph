; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/causal_cvt_pack_mytest/causal.input.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/causal_cvt_pack_mytest/causal.input.mlir"

@shm_0 = addrspace(3) global [16384 x i8] undef, align 16

; Function Attrs: convergent willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

declare float @__ocml_exp2_f32(float)

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3) #1 !dbg !4 !reqd_work_group_size !35 {
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
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #9, !dbg !8, !range !37, !frisk.register_lifetimes !1
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
  %119 = phi i64 [ %1298, %127 ], [ 0, %4 ], !dbg !7
  %120 = phi [2 x <8 x float>] [ %1283, %127 ], [ zeroinitializer, %4 ], !dbg !7
  %121 = phi [2 x <1 x float>] [ %1049, %127 ], [ zeroinitializer, %4 ], !dbg !7
  %122 = phi [2 x <1 x float>] [ %1285, %127 ], [ [<1 x float> splat (float 0xFFF0000000000000), <1 x float> splat (float 0xFFF0000000000000)], %4 ], !dbg !7
  %123 = phi [2 x <8 x float>] [ %1289, %127 ], [ zeroinitializer, %4 ], !dbg !7
  %124 = phi [2 x <8 x float>] [ %1293, %127 ], [ zeroinitializer, %4 ], !dbg !7
  %125 = phi [2 x <8 x float>] [ %1297, %127 ], [ zeroinitializer, %4 ], !dbg !7
  %126 = icmp slt i64 %119, %97, !dbg !7
  br i1 %126, label %127, label %1299, !dbg !7

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
  %address.thread565 = call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #9, !dbg !15, !range !37, !frisk.register_lifetimes !1
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
  %204 = load <4 x half>, ptr addrspace(3) %kdma.swizzled1876, align 8, !dbg !16, !frisk.k_lds_pair !40
  %kdma.gep1880 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1876, i64 128, !dbg !16
  %205 = load <4 x half>, ptr addrspace(3) %kdma.gep1880, align 8, !dbg !16, !frisk.k_lds_pair !40
  %kdma.gep1877 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1876, i64 4096, !dbg !16
  %206 = load <4 x half>, ptr addrspace(3) %kdma.gep1877, align 8, !dbg !16, !frisk.k_lds_pair !41
  %kdma.gep1881 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1876, i64 4224, !dbg !16
  %207 = load <4 x half>, ptr addrspace(3) %kdma.gep1881, align 8, !dbg !16, !frisk.k_lds_pair !41
  %kdma.gep1878 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 128, !dbg !16
  %208 = load <4 x half>, ptr addrspace(3) %kdma.gep1878, align 8, !dbg !16, !frisk.k_lds_pair !42
  %kdma.gep1882 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1024, !dbg !16
  %209 = load <4 x half>, ptr addrspace(3) %kdma.gep1882, align 8, !dbg !16, !frisk.k_lds_pair !42
  %kdma.gep1879 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4224, !dbg !16
  %210 = load <4 x half>, ptr addrspace(3) %kdma.gep1879, align 8, !dbg !16, !frisk.k_lds_pair !43
  %kdma.gep1883 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5120, !dbg !16
  %211 = load <4 x half>, ptr addrspace(3) %kdma.gep1883, align 8, !dbg !16, !frisk.k_lds_pair !43
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16, !frisk.k_lds_prefetch !1
  %212 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %193, <4 x float> %192) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16, !frisk.k_lds_prefetch !1
  %kdma.gep1884 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1876, i64 1024, !dbg !16
  %213 = load <4 x half>, ptr addrspace(3) %kdma.gep1884, align 8, !dbg !16, !frisk.k_lds_pair !38
  %kdma.gep1888 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1876, i64 1152, !dbg !16
  %214 = load <4 x half>, ptr addrspace(3) %kdma.gep1888, align 8, !dbg !16, !frisk.k_lds_pair !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16, !frisk.k_lds_prefetch !1
  %215 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %204, <4 x float> zeroinitializer) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16, !frisk.k_lds_prefetch !1
  %kdma.gep1885 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1876, i64 5120, !dbg !16
  %216 = load <4 x half>, ptr addrspace(3) %kdma.gep1885, align 8, !dbg !16, !frisk.k_lds_pair !44
  %kdma.gep1889 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1876, i64 5248, !dbg !16
  %217 = load <4 x half>, ptr addrspace(3) %kdma.gep1889, align 8, !dbg !16, !frisk.k_lds_pair !44
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16, !frisk.k_lds_prefetch !1
  %218 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %206, <4 x float> %215) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16, !frisk.k_lds_prefetch !1
  %kdma.gep1886 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1152, !dbg !16
  %219 = load <4 x half>, ptr addrspace(3) %kdma.gep1886, align 8, !dbg !16, !frisk.k_lds_pair !45
  %kdma.gep1887 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5248, !dbg !16
  %220 = load <4 x half>, ptr addrspace(3) %kdma.gep1887, align 8, !dbg !16, !frisk.k_lds_pair !45
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
  %272 = load <4 x half>, ptr addrspace(3) %kdma.gep1906, align 8, !dbg !16, !frisk.k_lds_pair !46
  %kdma.gep1910 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8320, !dbg !16
  %273 = load <4 x half>, ptr addrspace(3) %kdma.gep1910, align 8, !dbg !16, !frisk.k_lds_pair !46
  %kdma.gep1907 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12288, !dbg !16
  %274 = load <4 x half>, ptr addrspace(3) %kdma.gep1907, align 8, !dbg !16, !frisk.k_lds_pair !47
  %kdma.gep1911 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12416, !dbg !16
  %275 = load <4 x half>, ptr addrspace(3) %kdma.gep1911, align 8, !dbg !16, !frisk.k_lds_pair !47
  %kdma.gep1908 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1876, i64 8192, !dbg !16
  %276 = load <4 x half>, ptr addrspace(3) %kdma.gep1908, align 8, !dbg !16, !frisk.k_lds_pair !48
  %kdma.gep1912 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1876, i64 8320, !dbg !16
  %277 = load <4 x half>, ptr addrspace(3) %kdma.gep1912, align 8, !dbg !16, !frisk.k_lds_pair !48
  %kdma.gep1909 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1876, i64 12288, !dbg !16
  %278 = load <4 x half>, ptr addrspace(3) %kdma.gep1909, align 8, !dbg !16, !frisk.k_lds_pair !49
  %kdma.gep1913 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1876, i64 12416, !dbg !16
  %279 = load <4 x half>, ptr addrspace(3) %kdma.gep1913, align 8, !dbg !16, !frisk.k_lds_pair !49
  %kdma.gep1914 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9216, !dbg !16
  %280 = load <4 x half>, ptr addrspace(3) %kdma.gep1914, align 8, !dbg !16, !frisk.k_lds_pair !50
  %kdma.gep1918 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9344, !dbg !16
  %281 = load <4 x half>, ptr addrspace(3) %kdma.gep1918, align 8, !dbg !16, !frisk.k_lds_pair !50
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16, !frisk.k_lds_prefetch !1
  %282 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %272, <4 x float> %212) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16, !frisk.k_lds_prefetch !1
  %kdma.gep1915 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13312, !dbg !16
  %283 = load <4 x half>, ptr addrspace(3) %kdma.gep1915, align 8, !dbg !16, !frisk.k_lds_pair !51
  %kdma.gep1919 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13440, !dbg !16
  %284 = load <4 x half>, ptr addrspace(3) %kdma.gep1919, align 8, !dbg !16, !frisk.k_lds_pair !51
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16, !frisk.k_lds_prefetch !1
  %285 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %274, <4 x float> %282) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16, !frisk.k_lds_prefetch !1
  %kdma.gep1916 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1876, i64 9216, !dbg !16
  %286 = load <4 x half>, ptr addrspace(3) %kdma.gep1916, align 8, !dbg !16, !frisk.k_lds_pair !52
  %kdma.gep1920 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1876, i64 9344, !dbg !16
  %287 = load <4 x half>, ptr addrspace(3) %kdma.gep1920, align 8, !dbg !16, !frisk.k_lds_pair !52
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16, !frisk.k_lds_prefetch !1
  %288 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %276, <4 x float> %218) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16, !frisk.k_lds_prefetch !1
  %kdma.gep1917 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1876, i64 13312, !dbg !16
  %289 = load <4 x half>, ptr addrspace(3) %kdma.gep1917, align 8, !dbg !16, !frisk.k_lds_pair !53
  %kdma.gep1921 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1876, i64 13440, !dbg !16
  %290 = load <4 x half>, ptr addrspace(3) %kdma.gep1921, align 8, !dbg !16, !frisk.k_lds_pair !53
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
  %343 = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8, !dbg !16, !frisk.k_lds_pair !54
  %344 = load <4 x half>, ptr addrspace(3) %kdma.gep1878, align 8, !dbg !16, !frisk.k_lds_pair !54
  %345 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8, !dbg !16, !frisk.k_lds_pair !55
  %346 = load <4 x half>, ptr addrspace(3) %kdma.gep1879, align 8, !dbg !16, !frisk.k_lds_pair !55
  %347 = load <4 x half>, ptr addrspace(3) %kdma.swizzled1876, align 8, !dbg !16, !frisk.k_lds_pair !56
  %348 = load <4 x half>, ptr addrspace(3) %kdma.gep1880, align 8, !dbg !16, !frisk.k_lds_pair !56
  %349 = load <4 x half>, ptr addrspace(3) %kdma.gep1877, align 8, !dbg !16, !frisk.k_lds_pair !57
  %350 = load <4 x half>, ptr addrspace(3) %kdma.gep1881, align 8, !dbg !16, !frisk.k_lds_pair !57
  %351 = load <4 x half>, ptr addrspace(3) %kdma.gep1882, align 8, !dbg !16, !frisk.k_lds_pair !58
  %352 = load <4 x half>, ptr addrspace(3) %kdma.gep1886, align 8, !dbg !16, !frisk.k_lds_pair !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16, !frisk.k_lds_prefetch !1
  %353 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %343, <4 x float> %285) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16, !frisk.k_lds_prefetch !1
  %354 = load <4 x half>, ptr addrspace(3) %kdma.gep1883, align 8, !dbg !16, !frisk.k_lds_pair !59
  %355 = load <4 x half>, ptr addrspace(3) %kdma.gep1887, align 8, !dbg !16, !frisk.k_lds_pair !59
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16, !frisk.k_lds_prefetch !1
  %356 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %345, <4 x float> %353) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16, !frisk.k_lds_prefetch !1
  %357 = load <4 x half>, ptr addrspace(3) %kdma.gep1884, align 8, !dbg !16, !frisk.k_lds_pair !60
  %358 = load <4 x half>, ptr addrspace(3) %kdma.gep1888, align 8, !dbg !16, !frisk.k_lds_pair !60
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16, !frisk.k_lds_prefetch !1
  %359 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %347, <4 x float> %291) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16, !frisk.k_lds_prefetch !1
  %360 = load <4 x half>, ptr addrspace(3) %kdma.gep1885, align 8, !dbg !16, !frisk.k_lds_pair !61
  %361 = load <4 x half>, ptr addrspace(3) %kdma.gep1889, align 8, !dbg !16, !frisk.k_lds_pair !61
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
  %401 = load <4 x half>, ptr addrspace(3) %kdma.gep1906, align 8, !dbg !16, !frisk.k_tail_pair !40
  %402 = load <4 x half>, ptr addrspace(3) %kdma.gep1910, align 8, !dbg !16, !frisk.k_tail_pair !40
  %403 = load <4 x half>, ptr addrspace(3) %kdma.gep1907, align 8, !dbg !16, !frisk.k_tail_pair !41
  %404 = load <4 x half>, ptr addrspace(3) %kdma.gep1911, align 8, !dbg !16, !frisk.k_tail_pair !41
  %405 = load <4 x half>, ptr addrspace(3) %kdma.gep1908, align 8, !dbg !16, !frisk.k_tail_pair !42
  %406 = load <4 x half>, ptr addrspace(3) %kdma.gep1912, align 8, !dbg !16, !frisk.k_tail_pair !42
  %407 = load <4 x half>, ptr addrspace(3) %kdma.gep1909, align 8, !dbg !16, !frisk.k_tail_pair !43
  %408 = load <4 x half>, ptr addrspace(3) %kdma.gep1913, align 8, !dbg !16, !frisk.k_tail_pair !43
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16, !frisk.k_tail_prefetch !1
  %409 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %401, <4 x float> %356) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !18, !frisk.k_tail_prefetch !1
  %410 = or disjoint i64 %132, %114, !dbg !18
  %411 = trunc i64 %410 to i32, !dbg !18
  %.not40 = icmp ugt i32 %116, %411, !dbg !18
  %412 = select i1 %.not40, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !23
  %413 = or disjoint i32 %411, 1, !dbg !18
  %.not41 = icmp ugt i32 %116, %413, !dbg !18
  %414 = select i1 %.not41, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !23
  %415 = or disjoint i32 %411, 2, !dbg !18
  %.not42 = icmp ugt i32 %116, %415, !dbg !18
  %416 = select i1 %.not42, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !23
  %417 = or disjoint i32 %411, 3, !dbg !18
  %.not43 = icmp ugt i32 %116, %417, !dbg !18
  %418 = select i1 %.not43, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !23
  %419 = or disjoint i32 %411, 4, !dbg !18
  %.not44 = icmp ugt i32 %116, %419, !dbg !18
  %420 = select i1 %.not44, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !23
  %421 = or disjoint i32 %411, 5, !dbg !18
  %.not45 = icmp ugt i32 %116, %421, !dbg !18
  %422 = select i1 %.not45, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !23
  %423 = or disjoint i32 %411, 6, !dbg !18
  %.not46 = icmp ugt i32 %116, %423, !dbg !18
  %424 = select i1 %.not46, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !23
  %425 = or disjoint i32 %411, 7, !dbg !18
  %.not47 = icmp ugt i32 %116, %425, !dbg !18
  %426 = select i1 %.not47, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !23
  %427 = or disjoint i32 %411, 8, !dbg !18
  %.not48 = icmp ugt i32 %116, %427, !dbg !18
  %428 = select i1 %.not48, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !23
  %429 = or disjoint i32 %411, 9, !dbg !18
  %.not49 = icmp ugt i32 %116, %429, !dbg !18
  %430 = select i1 %.not49, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !23
  %431 = or disjoint i32 %411, 10, !dbg !18
  %.not50 = icmp ugt i32 %116, %431, !dbg !18
  %432 = select i1 %.not50, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !23
  %433 = or disjoint i32 %411, 11, !dbg !18
  %.not51 = icmp ugt i32 %116, %433, !dbg !18
  %434 = select i1 %.not51, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !23
  %435 = or disjoint i32 %411, 12, !dbg !18
  %.not52 = icmp ugt i32 %116, %435, !dbg !18
  %436 = select i1 %.not52, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !23
  %437 = or disjoint i32 %411, 13, !dbg !18
  %.not53 = icmp ugt i32 %116, %437, !dbg !18
  %438 = select i1 %.not53, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !23
  %439 = or disjoint i32 %411, 14, !dbg !18
  %.not54 = icmp ugt i32 %116, %439, !dbg !18
  %440 = select i1 %.not54, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !23
  %441 = or disjoint i32 %411, 15, !dbg !18
  %.not55 = icmp ugt i32 %116, %441, !dbg !18
  %442 = select i1 %.not55, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !23
  %443 = or disjoint i32 %411, 64, !dbg !18
  %.not56 = icmp ugt i32 %116, %443, !dbg !18
  %444 = select i1 %.not56, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !23
  %445 = or disjoint i32 %411, 65, !dbg !18
  %.not57 = icmp ugt i32 %116, %445, !dbg !18
  %446 = select i1 %.not57, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !23
  %447 = or disjoint i32 %411, 66, !dbg !18
  %.not58 = icmp ugt i32 %116, %447, !dbg !18
  %448 = select i1 %.not58, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !23
  %449 = or disjoint i32 %411, 67, !dbg !18
  %.not59 = icmp ugt i32 %116, %449, !dbg !18
  %450 = select i1 %.not59, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !23
  %451 = or disjoint i32 %411, 68, !dbg !18
  %.not60 = icmp ugt i32 %116, %451, !dbg !18
  %452 = select i1 %.not60, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !23
  %453 = or disjoint i32 %411, 69, !dbg !18
  %.not61 = icmp ugt i32 %116, %453, !dbg !18
  %454 = select i1 %.not61, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !23
  %455 = or disjoint i32 %411, 70, !dbg !18
  %.not62 = icmp ugt i32 %116, %455, !dbg !18
  %456 = select i1 %.not62, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !23
  %457 = or disjoint i32 %411, 71, !dbg !18
  %.not63 = icmp ugt i32 %116, %457, !dbg !18
  %458 = select i1 %.not63, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !23
  %459 = or disjoint i32 %411, 72, !dbg !18
  %.not64 = icmp ugt i32 %116, %459, !dbg !18
  %460 = select i1 %.not64, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !23
  %461 = or disjoint i32 %411, 73, !dbg !18
  %.not65 = icmp ugt i32 %116, %461, !dbg !18
  %462 = select i1 %.not65, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !23
  %463 = or disjoint i32 %411, 74, !dbg !18
  %.not66 = icmp ugt i32 %116, %463, !dbg !18
  %464 = select i1 %.not66, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !23
  %465 = or disjoint i32 %411, 75, !dbg !18
  %.not67 = icmp ugt i32 %116, %465, !dbg !18
  %466 = select i1 %.not67, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !23
  %467 = or disjoint i32 %411, 76, !dbg !18
  %.not68 = icmp ugt i32 %116, %467, !dbg !18
  %468 = select i1 %.not68, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !23
  %469 = or disjoint i32 %411, 77, !dbg !18
  %.not69 = icmp ugt i32 %116, %469, !dbg !18
  %470 = select i1 %.not69, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !23
  %471 = or disjoint i32 %411, 78, !dbg !18
  %.not70 = icmp ugt i32 %116, %471, !dbg !18
  %472 = select i1 %.not70, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !23
  %473 = or disjoint i32 %411, 79, !dbg !18
  %.not71 = icmp ugt i32 %116, %473, !dbg !18
  %474 = select i1 %.not71, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !23
  %.not72 = icmp ugt i32 %117, %411, !dbg !18
  %475 = select i1 %.not72, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !23
  %.not73 = icmp ugt i32 %117, %413, !dbg !18
  %476 = select i1 %.not73, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !23
  %.not74 = icmp ugt i32 %117, %415, !dbg !18
  %477 = select i1 %.not74, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !23
  %.not75 = icmp ugt i32 %117, %417, !dbg !18
  %478 = select i1 %.not75, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !23
  %.not76 = icmp ugt i32 %117, %419, !dbg !18
  %479 = select i1 %.not76, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !23
  %.not77 = icmp ugt i32 %117, %421, !dbg !18
  %480 = select i1 %.not77, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !23
  %.not78 = icmp ugt i32 %117, %423, !dbg !18
  %481 = select i1 %.not78, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !23
  %.not79 = icmp ugt i32 %117, %425, !dbg !18
  %482 = select i1 %.not79, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !23
  %.not80 = icmp ugt i32 %117, %427, !dbg !18
  %483 = select i1 %.not80, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !23
  %.not81 = icmp ugt i32 %117, %429, !dbg !18
  %484 = select i1 %.not81, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !23
  %.not82 = icmp ugt i32 %117, %431, !dbg !18
  %485 = select i1 %.not82, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !23
  %.not83 = icmp ugt i32 %117, %433, !dbg !18
  %486 = select i1 %.not83, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !23
  %.not84 = icmp ugt i32 %117, %435, !dbg !18
  %487 = select i1 %.not84, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !23
  %.not85 = icmp ugt i32 %117, %437, !dbg !18
  %488 = select i1 %.not85, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !23
  %.not86 = icmp ugt i32 %117, %439, !dbg !18
  %489 = select i1 %.not86, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !23
  %.not87 = icmp ugt i32 %117, %441, !dbg !18
  %490 = select i1 %.not87, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !23
  %.not88 = icmp ugt i32 %117, %443, !dbg !18
  %491 = select i1 %.not88, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !23
  %.not89 = icmp ugt i32 %117, %445, !dbg !18
  %492 = select i1 %.not89, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !23
  %.not90 = icmp ugt i32 %117, %447, !dbg !18
  %493 = select i1 %.not90, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !23
  %.not91 = icmp ugt i32 %117, %449, !dbg !18
  %494 = select i1 %.not91, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !23
  %.not92 = icmp ugt i32 %117, %451, !dbg !18
  %495 = select i1 %.not92, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !23
  %.not93 = icmp ugt i32 %117, %453, !dbg !18
  %496 = select i1 %.not93, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !23
  %.not94 = icmp ugt i32 %117, %455, !dbg !18
  %497 = select i1 %.not94, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !23
  %.not95 = icmp ugt i32 %117, %457, !dbg !18
  %498 = select i1 %.not95, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !23
  %.not96 = icmp ugt i32 %117, %459, !dbg !18
  %499 = select i1 %.not96, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !23
  %.not97 = icmp ugt i32 %117, %461, !dbg !18
  %500 = select i1 %.not97, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !23
  %.not98 = icmp ugt i32 %117, %463, !dbg !18
  %501 = select i1 %.not98, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !23
  %.not99 = icmp ugt i32 %117, %465, !dbg !18
  %502 = select i1 %.not99, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !23
  %.not100 = icmp ugt i32 %117, %467, !dbg !18
  %503 = select i1 %.not100, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !23
  %.not101 = icmp ugt i32 %117, %469, !dbg !18
  %504 = select i1 %.not101, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !23
  %.not102 = icmp ugt i32 %117, %471, !dbg !18
  %505 = select i1 %.not102, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !23
  %.not103 = icmp ugt i32 %117, %473, !dbg !18
  %506 = select i1 %.not103, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !23
  %507 = insertelement <4 x float> poison, float %412, i64 0, !dbg !24
  %508 = insertelement <4 x float> %507, float %414, i64 1, !dbg !24
  %509 = insertelement <4 x float> %508, float %416, i64 2, !dbg !24
  %510 = insertelement <4 x float> %509, float %418, i64 3, !dbg !24
  %511 = insertelement <4 x float> poison, float %420, i64 0, !dbg !24
  %512 = insertelement <4 x float> %511, float %422, i64 1, !dbg !24
  %513 = insertelement <4 x float> %512, float %424, i64 2, !dbg !24
  %514 = insertelement <4 x float> %513, float %426, i64 3, !dbg !24
  %515 = insertelement <4 x float> poison, float %428, i64 0, !dbg !24
  %516 = insertelement <4 x float> %515, float %430, i64 1, !dbg !24
  %517 = insertelement <4 x float> %516, float %432, i64 2, !dbg !24
  %518 = insertelement <4 x float> %517, float %434, i64 3, !dbg !24
  %519 = insertelement <4 x float> poison, float %436, i64 0, !dbg !24
  %520 = insertelement <4 x float> %519, float %438, i64 1, !dbg !24
  %521 = insertelement <4 x float> %520, float %440, i64 2, !dbg !24
  %522 = insertelement <4 x float> %521, float %442, i64 3, !dbg !24
  %523 = insertelement <4 x float> poison, float %444, i64 0, !dbg !24
  %524 = insertelement <4 x float> %523, float %446, i64 1, !dbg !24
  %525 = insertelement <4 x float> %524, float %448, i64 2, !dbg !24
  %526 = insertelement <4 x float> %525, float %450, i64 3, !dbg !24
  %527 = insertelement <4 x float> poison, float %452, i64 0, !dbg !24
  %528 = insertelement <4 x float> %527, float %454, i64 1, !dbg !24
  %529 = insertelement <4 x float> %528, float %456, i64 2, !dbg !24
  %530 = insertelement <4 x float> %529, float %458, i64 3, !dbg !24
  %531 = insertelement <4 x float> poison, float %460, i64 0, !dbg !24
  %532 = insertelement <4 x float> %531, float %462, i64 1, !dbg !24
  %533 = insertelement <4 x float> %532, float %464, i64 2, !dbg !24
  %534 = insertelement <4 x float> %533, float %466, i64 3, !dbg !24
  %535 = insertelement <4 x float> poison, float %468, i64 0, !dbg !24
  %536 = insertelement <4 x float> %535, float %470, i64 1, !dbg !24
  %537 = insertelement <4 x float> %536, float %472, i64 2, !dbg !24
  %538 = insertelement <4 x float> %537, float %474, i64 3, !dbg !24
  %539 = insertelement <4 x float> poison, float %475, i64 0, !dbg !24
  %540 = insertelement <4 x float> %539, float %476, i64 1, !dbg !24
  %541 = insertelement <4 x float> %540, float %477, i64 2, !dbg !24
  %542 = insertelement <4 x float> %541, float %478, i64 3, !dbg !24
  %543 = insertelement <4 x float> poison, float %479, i64 0, !dbg !24
  %544 = insertelement <4 x float> %543, float %480, i64 1, !dbg !24
  %545 = insertelement <4 x float> %544, float %481, i64 2, !dbg !24
  %546 = insertelement <4 x float> %545, float %482, i64 3, !dbg !24
  %547 = insertelement <4 x float> poison, float %483, i64 0, !dbg !24
  %548 = insertelement <4 x float> %547, float %484, i64 1, !dbg !24
  %549 = insertelement <4 x float> %548, float %485, i64 2, !dbg !24
  %550 = insertelement <4 x float> %549, float %486, i64 3, !dbg !24
  %551 = insertelement <4 x float> poison, float %487, i64 0, !dbg !24
  %552 = insertelement <4 x float> %551, float %488, i64 1, !dbg !24
  %553 = insertelement <4 x float> %552, float %489, i64 2, !dbg !24
  %554 = insertelement <4 x float> %553, float %490, i64 3, !dbg !24
  %555 = insertelement <4 x float> poison, float %491, i64 0, !dbg !24
  %556 = insertelement <4 x float> %555, float %492, i64 1, !dbg !24
  %557 = insertelement <4 x float> %556, float %493, i64 2, !dbg !24
  %558 = insertelement <4 x float> %557, float %494, i64 3, !dbg !24
  %559 = insertelement <4 x float> poison, float %495, i64 0, !dbg !24
  %560 = insertelement <4 x float> %559, float %496, i64 1, !dbg !24
  %561 = insertelement <4 x float> %560, float %497, i64 2, !dbg !24
  %562 = insertelement <4 x float> %561, float %498, i64 3, !dbg !24
  %563 = insertelement <4 x float> poison, float %499, i64 0, !dbg !24
  %564 = insertelement <4 x float> %563, float %500, i64 1, !dbg !24
  %565 = insertelement <4 x float> %564, float %501, i64 2, !dbg !24
  %566 = insertelement <4 x float> %565, float %502, i64 3, !dbg !24
  %567 = insertelement <4 x float> poison, float %503, i64 0, !dbg !24
  %568 = insertelement <4 x float> %567, float %504, i64 1, !dbg !24
  %569 = insertelement <4 x float> %568, float %505, i64 2, !dbg !24
  %570 = insertelement <4 x float> %569, float %506, i64 3, !dbg !24
  %571 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %403, <4 x float> %409) #9, !dbg !16
  %572 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %405, <4 x float> %362) #9, !dbg !16
  %573 = fadd <4 x float> %571, %510, !dbg !24
  %574 = extractelement <4 x float> %573, i64 0, !dbg !24
  %575 = call float @llvm.maxnum.f32(float %574, float 0xFFF0000000000000), !dbg !25, !frisk.local_reduction !1
  %576 = extractelement <4 x float> %573, i64 1, !dbg !24
  %577 = extractelement <4 x float> %573, i64 2, !dbg !24
  %578 = extractelement <4 x float> %573, i64 3, !dbg !24
  %579 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %407, <4 x float> %572) #9, !dbg !16
  %580 = load <4 x half>, ptr addrspace(3) %kdma.gep1914, align 8, !dbg !16, !frisk.k_tail_pair !38
  %581 = load <4 x half>, ptr addrspace(3) %kdma.gep1918, align 8, !dbg !16, !frisk.k_tail_pair !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16, !frisk.k_tail_prefetch !1
  %582 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %402, <4 x float> %364) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !24, !frisk.k_tail_prefetch !1
  %583 = fadd <4 x float> %579, %514, !dbg !24
  %584 = extractelement <4 x float> %583, i64 0, !dbg !24
  %local.reduce218 = call float @llvm.maxnum.f32(float %575, float %584), !dbg !25, !frisk.local_reduction !1
  %585 = extractelement <4 x float> %583, i64 1, !dbg !24
  %local.reduce219 = call float @llvm.maxnum.f32(float %576, float %585), !dbg !25, !frisk.local_reduction !1
  %586 = extractelement <4 x float> %583, i64 2, !dbg !24
  %local.reduce220 = call float @llvm.maxnum.f32(float %577, float %586), !dbg !25, !frisk.local_reduction !1
  %587 = extractelement <4 x float> %583, i64 3, !dbg !24
  %local.reduce221 = call float @llvm.maxnum.f32(float %578, float %587), !dbg !25, !frisk.local_reduction !1
  %588 = load <4 x half>, ptr addrspace(3) %kdma.gep1915, align 8, !dbg !16, !frisk.k_tail_pair !44
  %589 = load <4 x half>, ptr addrspace(3) %kdma.gep1919, align 8, !dbg !16, !frisk.k_tail_pair !44
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16, !frisk.k_tail_prefetch !1
  %590 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %404, <4 x float> %582) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16, !frisk.k_tail_prefetch !1
  %591 = load <4 x half>, ptr addrspace(3) %kdma.gep1916, align 8, !dbg !16, !frisk.k_tail_pair !45
  %592 = load <4 x half>, ptr addrspace(3) %kdma.gep1920, align 8, !dbg !16, !frisk.k_tail_pair !45
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16, !frisk.k_tail_prefetch !1
  %593 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %406, <4 x float> %366) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !24, !frisk.k_tail_prefetch !1
  %594 = fadd <4 x float> %590, %518, !dbg !24
  %595 = extractelement <4 x float> %594, i64 0, !dbg !24
  %local.reduce222 = call float @llvm.maxnum.f32(float %local.reduce218, float %595), !dbg !25, !frisk.local_reduction !1
  %596 = extractelement <4 x float> %594, i64 1, !dbg !24
  %local.reduce223 = call float @llvm.maxnum.f32(float %local.reduce219, float %596), !dbg !25, !frisk.local_reduction !1
  %597 = extractelement <4 x float> %594, i64 2, !dbg !24
  %local.reduce224 = call float @llvm.maxnum.f32(float %local.reduce220, float %597), !dbg !25, !frisk.local_reduction !1
  %598 = extractelement <4 x float> %594, i64 3, !dbg !24
  %local.reduce225 = call float @llvm.maxnum.f32(float %local.reduce221, float %598), !dbg !25, !frisk.local_reduction !1
  %599 = load <4 x half>, ptr addrspace(3) %kdma.gep1917, align 8, !dbg !16, !frisk.k_tail_pair !46
  %600 = load <4 x half>, ptr addrspace(3) %kdma.gep1921, align 8, !dbg !16, !frisk.k_tail_pair !46
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16, !frisk.k_tail_prefetch !1
  %601 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %408, <4 x float> %593) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16, !frisk.k_tail_prefetch !1
  %602 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %580, <4 x float> %368) #9, !dbg !16
  %603 = fadd <4 x float> %601, %522, !dbg !24
  %604 = extractelement <4 x float> %603, i64 0, !dbg !24
  %local.reduce226 = call float @llvm.maxnum.f32(float %local.reduce222, float %604), !dbg !25, !frisk.local_reduction !1
  %605 = extractelement <4 x float> %603, i64 1, !dbg !24
  %local.reduce227 = call float @llvm.maxnum.f32(float %local.reduce223, float %605), !dbg !25, !frisk.local_reduction !1
  %606 = extractelement <4 x float> %603, i64 2, !dbg !24
  %local.reduce228 = call float @llvm.maxnum.f32(float %local.reduce224, float %606), !dbg !25, !frisk.local_reduction !1
  %607 = extractelement <4 x float> %603, i64 3, !dbg !24
  %local.reduce229 = call float @llvm.maxnum.f32(float %local.reduce225, float %607), !dbg !25, !frisk.local_reduction !1
  %608 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %588, <4 x float> %602) #9, !dbg !16
  %609 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %591, <4 x float> %370) #9, !dbg !16
  %610 = fadd <4 x float> %608, %526, !dbg !24
  %611 = extractelement <4 x float> %610, i64 0, !dbg !24
  %local.reduce230 = call float @llvm.maxnum.f32(float %local.reduce226, float %611), !dbg !25, !frisk.local_reduction !1
  %612 = extractelement <4 x float> %610, i64 1, !dbg !24
  %local.reduce231 = call float @llvm.maxnum.f32(float %local.reduce227, float %612), !dbg !25, !frisk.local_reduction !1
  %613 = extractelement <4 x float> %610, i64 2, !dbg !24
  %local.reduce232 = call float @llvm.maxnum.f32(float %local.reduce228, float %613), !dbg !25, !frisk.local_reduction !1
  %614 = extractelement <4 x float> %610, i64 3, !dbg !24
  %local.reduce233 = call float @llvm.maxnum.f32(float %local.reduce229, float %614), !dbg !25, !frisk.local_reduction !1
  %615 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %599, <4 x float> %609) #9, !dbg !16
  %616 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %581, <4 x float> %372) #9, !dbg !16
  %617 = fadd <4 x float> %615, %530, !dbg !24
  %618 = extractelement <4 x float> %617, i64 0, !dbg !24
  %local.reduce234 = call float @llvm.maxnum.f32(float %local.reduce230, float %618), !dbg !25, !frisk.local_reduction !1
  %619 = extractelement <4 x float> %617, i64 1, !dbg !24
  %local.reduce235 = call float @llvm.maxnum.f32(float %local.reduce231, float %619), !dbg !25, !frisk.local_reduction !1
  %620 = extractelement <4 x float> %617, i64 2, !dbg !24
  %local.reduce236 = call float @llvm.maxnum.f32(float %local.reduce232, float %620), !dbg !25, !frisk.local_reduction !1
  %621 = extractelement <4 x float> %617, i64 3, !dbg !24
  %local.reduce237 = call float @llvm.maxnum.f32(float %local.reduce233, float %621), !dbg !25, !frisk.local_reduction !1
  %622 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %589, <4 x float> %616) #9, !dbg !16
  %623 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %592, <4 x float> %374) #9, !dbg !16
  %624 = fadd <4 x float> %622, %534, !dbg !24
  %625 = extractelement <4 x float> %624, i64 0, !dbg !24
  %local.reduce238 = call float @llvm.maxnum.f32(float %local.reduce234, float %625), !dbg !25, !frisk.local_reduction !1
  %626 = extractelement <4 x float> %624, i64 1, !dbg !24
  %local.reduce239 = call float @llvm.maxnum.f32(float %local.reduce235, float %626), !dbg !25, !frisk.local_reduction !1
  %627 = extractelement <4 x float> %624, i64 2, !dbg !24
  %local.reduce240 = call float @llvm.maxnum.f32(float %local.reduce236, float %627), !dbg !25, !frisk.local_reduction !1
  %628 = extractelement <4 x float> %624, i64 3, !dbg !24
  %local.reduce241 = call float @llvm.maxnum.f32(float %local.reduce237, float %628), !dbg !25, !frisk.local_reduction !1
  %629 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %600, <4 x float> %623) #9, !dbg !16
  %630 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %401, <4 x float> %376) #9, !dbg !16
  %631 = fadd <4 x float> %629, %538, !dbg !24
  %632 = extractelement <4 x float> %631, i64 0, !dbg !24
  %local.reduce242 = call float @llvm.maxnum.f32(float %local.reduce238, float %632), !dbg !25, !frisk.local_reduction !1
  %633 = extractelement <4 x float> %631, i64 1, !dbg !24
  %local.reduce243 = call float @llvm.maxnum.f32(float %local.reduce239, float %633), !dbg !25, !frisk.local_reduction !1
  %634 = extractelement <4 x float> %631, i64 2, !dbg !24
  %local.reduce244 = call float @llvm.maxnum.f32(float %local.reduce240, float %634), !dbg !25, !frisk.local_reduction !1
  %635 = extractelement <4 x float> %631, i64 3, !dbg !24
  %local.reduce245 = call float @llvm.maxnum.f32(float %local.reduce241, float %635), !dbg !25, !frisk.local_reduction !1
  %local.reduce246 = call float @llvm.maxnum.f32(float %local.reduce242, float %local.reduce243), !dbg !25, !frisk.local_reduction !1
  %local.reduce247 = call float @llvm.maxnum.f32(float %local.reduce244, float %local.reduce245), !dbg !25, !frisk.local_reduction !1
  %636 = call float @llvm.maxnum.f32(float %local.reduce246, float %local.reduce247), !dbg !25, !frisk.local_reduction !1
  %637 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %403, <4 x float> %630) #9, !dbg !16
  %638 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %405, <4 x float> %378) #9, !dbg !16
  %639 = fadd <4 x float> %637, %542, !dbg !24
  %640 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %407, <4 x float> %638) #9, !dbg !16
  %641 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %402, <4 x float> %380) #9, !dbg !16
  %642 = fadd <4 x float> %640, %546, !dbg !24
  %643 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %404, <4 x float> %641) #9, !dbg !16
  %644 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %406, <4 x float> %382) #9, !dbg !16
  %645 = fadd <4 x float> %643, %550, !dbg !24
  %646 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %408, <4 x float> %644) #9, !dbg !16
  %647 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %580, <4 x float> %384) #9, !dbg !16
  %648 = fadd <4 x float> %646, %554, !dbg !24
  %649 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %588, <4 x float> %647) #9, !dbg !16
  %650 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %591, <4 x float> %386) #9, !dbg !16
  %651 = fadd <4 x float> %649, %558, !dbg !24
  %652 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %599, <4 x float> %650) #9, !dbg !16
  %653 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %581, <4 x float> %388) #9, !dbg !16
  %654 = fadd <4 x float> %652, %562, !dbg !24
  %655 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %589, <4 x float> %653) #9, !dbg !16
  %656 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %592, <4 x float> %390) #9, !dbg !16
  %657 = fadd <4 x float> %655, %566, !dbg !24
  %658 = extractelement <4 x float> %639, i64 0, !dbg !24
  %659 = extractelement <4 x float> %639, i64 1, !dbg !24
  %660 = extractelement <4 x float> %639, i64 2, !dbg !24
  %661 = extractelement <4 x float> %639, i64 3, !dbg !24
  %662 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %600, <4 x float> %656) #9, !dbg !16
  %663 = extractelement <4 x float> %642, i64 0, !dbg !24
  %664 = extractelement <4 x float> %642, i64 1, !dbg !24
  %665 = extractelement <4 x float> %642, i64 2, !dbg !24
  %666 = extractelement <4 x float> %642, i64 3, !dbg !24
  %667 = extractelement <4 x float> %645, i64 0, !dbg !24
  %668 = extractelement <4 x float> %645, i64 1, !dbg !24
  %669 = extractelement <4 x float> %645, i64 2, !dbg !24
  %670 = extractelement <4 x float> %645, i64 3, !dbg !24
  %671 = fadd <4 x float> %662, %570, !dbg !24
  %672 = extractelement <4 x float> %648, i64 0, !dbg !24
  %673 = extractelement <4 x float> %648, i64 1, !dbg !24
  %674 = extractelement <4 x float> %648, i64 2, !dbg !24
  %675 = extractelement <4 x float> %648, i64 3, !dbg !24
  %676 = extractelement <4 x float> %651, i64 0, !dbg !24
  %677 = extractelement <4 x float> %651, i64 1, !dbg !24
  %678 = extractelement <4 x float> %651, i64 2, !dbg !24
  %679 = extractelement <4 x float> %651, i64 3, !dbg !24
  %680 = extractelement <4 x float> %654, i64 0, !dbg !24
  %681 = extractelement <4 x float> %654, i64 1, !dbg !24
  %682 = extractelement <4 x float> %654, i64 2, !dbg !24
  %683 = extractelement <4 x float> %654, i64 3, !dbg !24
  %684 = extractelement <4 x float> %657, i64 0, !dbg !24
  %685 = extractelement <4 x float> %657, i64 1, !dbg !24
  %686 = extractelement <4 x float> %657, i64 2, !dbg !24
  %687 = extractelement <4 x float> %657, i64 3, !dbg !24
  %688 = extractelement <4 x float> %671, i64 0, !dbg !24
  %689 = extractelement <4 x float> %671, i64 1, !dbg !24
  %690 = extractelement <4 x float> %671, i64 2, !dbg !24
  %691 = extractelement <4 x float> %671, i64 3, !dbg !24
  %692 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !25
  %693 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %692), !dbg !25
  %694 = add nuw nsw i32 %693, 64, !dbg !25
  %695 = xor i32 %693, 16, !dbg !25
  %696 = and i32 %694, 192, !dbg !25
  %697 = icmp samesign ult i32 %695, %696, !dbg !25
  %698 = select i1 %697, i32 %695, i32 %693, !dbg !25
  %699 = shl nuw nsw i32 %698, 2, !dbg !25
  %700 = bitcast float %636 to i32, !dbg !25
  %701 = call i32 @llvm.amdgcn.ds.bpermute(i32 %699, i32 %700), !dbg !25
  %702 = bitcast i32 %701 to float, !dbg !25
  %703 = call float @llvm.maxnum.f32(float %636, float %702), !dbg !25
  %704 = xor i32 %693, 32, !dbg !25
  %705 = icmp samesign ult i32 %704, %696, !dbg !25
  %706 = select i1 %705, i32 %704, i32 %693, !dbg !25
  %707 = shl nuw nsw i32 %706, 2, !dbg !25
  %708 = bitcast float %703 to i32, !dbg !25
  %709 = call i32 @llvm.amdgcn.ds.bpermute(i32 %707, i32 %708), !dbg !25
  %710 = bitcast i32 %709 to float, !dbg !25
  %711 = call float @llvm.maxnum.f32(float %703, float %710), !dbg !25
  %712 = call float @llvm.maxnum.f32(float %658, float 0xFFF0000000000000), !dbg !25, !frisk.local_reduction !1
  %local.reduce187 = call float @llvm.maxnum.f32(float %712, float %663), !dbg !25, !frisk.local_reduction !1
  %local.reduce188 = call float @llvm.maxnum.f32(float %659, float %664), !dbg !25, !frisk.local_reduction !1
  %local.reduce189 = call float @llvm.maxnum.f32(float %660, float %665), !dbg !25, !frisk.local_reduction !1
  %local.reduce190 = call float @llvm.maxnum.f32(float %661, float %666), !dbg !25, !frisk.local_reduction !1
  %local.reduce191 = call float @llvm.maxnum.f32(float %local.reduce187, float %667), !dbg !25, !frisk.local_reduction !1
  %local.reduce192 = call float @llvm.maxnum.f32(float %local.reduce188, float %668), !dbg !25, !frisk.local_reduction !1
  %local.reduce193 = call float @llvm.maxnum.f32(float %local.reduce189, float %669), !dbg !25, !frisk.local_reduction !1
  %local.reduce194 = call float @llvm.maxnum.f32(float %local.reduce190, float %670), !dbg !25, !frisk.local_reduction !1
  %local.reduce195 = call float @llvm.maxnum.f32(float %local.reduce191, float %672), !dbg !25, !frisk.local_reduction !1
  %local.reduce196 = call float @llvm.maxnum.f32(float %local.reduce192, float %673), !dbg !25, !frisk.local_reduction !1
  %local.reduce197 = call float @llvm.maxnum.f32(float %local.reduce193, float %674), !dbg !25, !frisk.local_reduction !1
  %local.reduce198 = call float @llvm.maxnum.f32(float %local.reduce194, float %675), !dbg !25, !frisk.local_reduction !1
  %local.reduce199 = call float @llvm.maxnum.f32(float %local.reduce195, float %676), !dbg !25, !frisk.local_reduction !1
  %local.reduce200 = call float @llvm.maxnum.f32(float %local.reduce196, float %677), !dbg !25, !frisk.local_reduction !1
  %local.reduce201 = call float @llvm.maxnum.f32(float %local.reduce197, float %678), !dbg !25, !frisk.local_reduction !1
  %local.reduce202 = call float @llvm.maxnum.f32(float %local.reduce198, float %679), !dbg !25, !frisk.local_reduction !1
  %local.reduce203 = call float @llvm.maxnum.f32(float %local.reduce199, float %680), !dbg !25, !frisk.local_reduction !1
  %local.reduce204 = call float @llvm.maxnum.f32(float %local.reduce200, float %681), !dbg !25, !frisk.local_reduction !1
  %local.reduce205 = call float @llvm.maxnum.f32(float %local.reduce201, float %682), !dbg !25, !frisk.local_reduction !1
  %local.reduce206 = call float @llvm.maxnum.f32(float %local.reduce202, float %683), !dbg !25, !frisk.local_reduction !1
  %local.reduce207 = call float @llvm.maxnum.f32(float %local.reduce203, float %684), !dbg !25, !frisk.local_reduction !1
  %local.reduce208 = call float @llvm.maxnum.f32(float %local.reduce204, float %685), !dbg !25, !frisk.local_reduction !1
  %local.reduce209 = call float @llvm.maxnum.f32(float %local.reduce205, float %686), !dbg !25, !frisk.local_reduction !1
  %local.reduce210 = call float @llvm.maxnum.f32(float %local.reduce206, float %687), !dbg !25, !frisk.local_reduction !1
  %local.reduce211 = call float @llvm.maxnum.f32(float %local.reduce207, float %688), !dbg !25, !frisk.local_reduction !1
  %local.reduce212 = call float @llvm.maxnum.f32(float %local.reduce208, float %689), !dbg !25, !frisk.local_reduction !1
  %local.reduce213 = call float @llvm.maxnum.f32(float %local.reduce209, float %690), !dbg !25, !frisk.local_reduction !1
  %local.reduce214 = call float @llvm.maxnum.f32(float %local.reduce210, float %691), !dbg !25, !frisk.local_reduction !1
  %local.reduce215 = call float @llvm.maxnum.f32(float %local.reduce211, float %local.reduce212), !dbg !25, !frisk.local_reduction !1
  %local.reduce216 = call float @llvm.maxnum.f32(float %local.reduce213, float %local.reduce214), !dbg !25, !frisk.local_reduction !1
  %713 = call float @llvm.maxnum.f32(float %local.reduce215, float %local.reduce216), !dbg !25, !frisk.local_reduction !1
  %714 = bitcast float %713 to i32, !dbg !25
  %715 = call i32 @llvm.amdgcn.ds.bpermute(i32 %699, i32 %714), !dbg !25
  %716 = bitcast i32 %715 to float, !dbg !25
  %717 = call float @llvm.maxnum.f32(float %713, float %716), !dbg !25
  %718 = bitcast float %717 to i32, !dbg !25
  %719 = call i32 @llvm.amdgcn.ds.bpermute(i32 %707, i32 %718), !dbg !25
  %720 = bitcast i32 %719 to float, !dbg !25
  %721 = call float @llvm.maxnum.f32(float %717, float %720), !dbg !25
  %722 = extractvalue [2 x <1 x float>] %122, 0, !dbg !25
  %723 = insertelement <1 x float> poison, float %711, i64 0, !dbg !25
  %724 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %723, <1 x float> %722), !dbg !25
  %725 = extractvalue [2 x <1 x float>] %122, 1, !dbg !25
  %726 = insertelement <1 x float> poison, float %721, i64 0, !dbg !25
  %727 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %726, <1 x float> %725), !dbg !25
  %728 = shufflevector <1 x float> %724, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !26
  %729 = shufflevector <1 x float> %727, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !26
  %730 = fsub <1 x float> %722, %724, !dbg !27
  %731 = extractelement <1 x float> %730, i64 0, !dbg !28
  %732 = call float @llvm.amdgcn.exp2.f32(float %731), !dbg !28
  %733 = insertelement <4 x float> poison, float %732, i64 0, !dbg !33
  %734 = shufflevector <4 x float> %733, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !33
  %735 = extractvalue [2 x <1 x float>] %121, 0, !dbg !29
  %736 = extractelement <1 x float> %735, i64 0, !dbg !29
  %737 = fmul float %736, %732, !dbg !29
  %738 = fsub <1 x float> %725, %727, !dbg !27
  %739 = extractelement <1 x float> %738, i64 0, !dbg !28
  %740 = call float @llvm.amdgcn.exp2.f32(float %739), !dbg !28
  %741 = insertelement <4 x float> poison, float %740, i64 0, !dbg !33
  %742 = shufflevector <4 x float> %741, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !33
  %743 = extractvalue [2 x <1 x float>] %121, 1, !dbg !29
  %744 = extractelement <1 x float> %743, i64 0, !dbg !29
  %745 = fmul float %744, %740, !dbg !29
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
  %746 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1535, <4 x i32> %dma.buffer.rsrc1414) #10, !dbg !17
  %vdb.dma.base1549 = add i32 %145, 8192, !dbg !13
  %747 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1549, i32 %vdb.address1544, <4 x i32> %dma.buffer.rsrc1414) #10, !dbg !17
  %address.local996 = shl nuw nsw i32 %address.thread565, 3, !dbg !13
  %address.local997 = and i32 %address.local996, 120, !dbg !13
  %address.local999 = and i32 %address.local642, 1536, !dbg !13
  %address.local1000 = or disjoint i32 %address.local997, %address.local999, !dbg !13
  %address.local1001 = zext nneg i32 %address.local1000 to i64, !dbg !13
  %address.local1002 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1001, !dbg !13
  %748 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1002, i32 0) #0, !dbg !13, !frisk.v_double_buffer !39
  %749 = bitcast <4 x i32> %748 to <8 x half>, !dbg !13
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
  %750 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1012, i32 0) #0, !dbg !13, !frisk.v_double_buffer !39
  %751 = bitcast <4 x i32> %750 to <8 x half>, !dbg !13
  %address.local1016 = getelementptr i8, ptr addrspace(3) %address.local1011, i64 512, !dbg !13
  %752 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1016, i32 0) #0, !dbg !13, !frisk.v_double_buffer !39
  %753 = bitcast <4 x i32> %752 to <8 x half>, !dbg !13
  %address.local1020 = getelementptr i8, ptr addrspace(3) %address.local1011, i64 768, !dbg !13
  %754 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1020, i32 0) #0, !dbg !13, !frisk.v_double_buffer !39
  %755 = bitcast <4 x i32> %754 to <8 x half>, !dbg !13
  %address.local1022 = getelementptr i8, ptr addrspace(3) %address.local1002, i64 4096, !dbg !13
  %756 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1022, i32 0) #0, !dbg !13, !frisk.v_double_buffer !39
  %757 = bitcast <4 x i32> %756 to <8 x half>, !dbg !13
  %address.local1026 = getelementptr i8, ptr addrspace(3) %address.local1011, i64 4352, !dbg !13
  %758 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1026, i32 0) #0, !dbg !13, !frisk.v_double_buffer !39
  %759 = bitcast <4 x i32> %758 to <8 x half>, !dbg !13
  %address.local1030 = getelementptr i8, ptr addrspace(3) %address.local1011, i64 4608, !dbg !13
  %760 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1030, i32 0) #0, !dbg !13, !frisk.v_double_buffer !39
  %761 = bitcast <4 x i32> %760 to <8 x half>, !dbg !13
  %address.local1034 = getelementptr i8, ptr addrspace(3) %address.local1011, i64 4864, !dbg !13
  %762 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1034, i32 0) #0, !dbg !13, !frisk.v_double_buffer !39
  %763 = shufflevector <8 x half> %749, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %764 = extractvalue [2 x <8 x float>] %120, 0, !dbg !33
  %765 = shufflevector <8 x float> %764, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !33
  %766 = fmul <4 x float> %765, %734, !dbg !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %767 = fsub <4 x float> %573, %728, !dbg !26
  %768 = extractelement <4 x float> %767, i64 0, !dbg !30
  %769 = call float @llvm.amdgcn.exp2.f32(float %768), !dbg !30
  %770 = insertelement <4 x float> poison, float %769, i64 0, !dbg !32
  %771 = extractelement <4 x float> %767, i64 1, !dbg !30
  %772 = call float @llvm.amdgcn.exp2.f32(float %771), !dbg !30
  %773 = insertelement <4 x float> %770, float %772, i64 1, !dbg !32
  %774 = extractelement <4 x float> %767, i64 2, !dbg !30
  %775 = call float @llvm.amdgcn.exp2.f32(float %774), !dbg !30
  %776 = insertelement <4 x float> %773, float %775, i64 2, !dbg !32
  %777 = extractelement <4 x float> %767, i64 3, !dbg !30
  %778 = call float @llvm.amdgcn.exp2.f32(float %777), !dbg !30
  %779 = insertelement <4 x float> %776, float %778, i64 3, !dbg !32
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %779) #9, !dbg !32, !frisk.register_lifetimes !1
  %.cvtpack.f0 = extractelement <4 x float> %math.result, i32 0, !dbg !32
  %.cvtpack.f1 = extractelement <4 x float> %math.result, i32 1, !dbg !32
  %.cvtpack.f2 = extractelement <4 x float> %math.result, i32 2, !dbg !32
  %.cvtpack.f3 = extractelement <4 x float> %math.result, i32 3, !dbg !32
  %.cvtpack.p0 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f0, float %.cvtpack.f1), !dbg !32
  %.cvtpack.p1 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f2, float %.cvtpack.f3), !dbg !32
  %.cvtpack.v0 = insertelement <2 x i32> poison, i32 %.cvtpack.p0, i32 0, !dbg !32
  %.cvtpack.v1 = insertelement <2 x i32> %.cvtpack.v0, i32 %.cvtpack.p1, i32 1, !dbg !32
  %780 = bitcast <2 x i32> %.cvtpack.v1 to <4 x half>, !dbg !32
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %780) #9, !dbg !30, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %781 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %763, <4 x float> %766) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %782 = shufflevector <8 x half> %749, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %783 = shufflevector <8 x float> %764, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !33
  %784 = fmul <4 x float> %783, %734, !dbg !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %785 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %782, <4 x float> %784) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !33
  %786 = extractvalue [2 x <8 x float>] %120, 1, !dbg !33
  %787 = shufflevector <8 x float> %786, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !33
  %788 = fmul <4 x float> %787, %742, !dbg !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %789 = fsub <4 x float> %639, %729, !dbg !26
  %790 = extractelement <4 x float> %789, i64 0, !dbg !30
  %791 = call float @llvm.amdgcn.exp2.f32(float %790), !dbg !30
  %792 = insertelement <4 x float> poison, float %791, i64 0, !dbg !32
  %793 = extractelement <4 x float> %789, i64 1, !dbg !30
  %794 = call float @llvm.amdgcn.exp2.f32(float %793), !dbg !30
  %795 = insertelement <4 x float> %792, float %794, i64 1, !dbg !32
  %796 = extractelement <4 x float> %789, i64 2, !dbg !30
  %797 = call float @llvm.amdgcn.exp2.f32(float %796), !dbg !30
  %798 = insertelement <4 x float> %795, float %797, i64 2, !dbg !32
  %799 = extractelement <4 x float> %789, i64 3, !dbg !30
  %800 = call float @llvm.amdgcn.exp2.f32(float %799), !dbg !30
  %801 = insertelement <4 x float> %798, float %800, i64 3, !dbg !32
  %math.result455 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %801) #9, !dbg !32, !frisk.register_lifetimes !1
  %.cvtpack.f02009 = extractelement <4 x float> %math.result455, i32 0, !dbg !32
  %.cvtpack.f12010 = extractelement <4 x float> %math.result455, i32 1, !dbg !32
  %.cvtpack.f22011 = extractelement <4 x float> %math.result455, i32 2, !dbg !32
  %.cvtpack.f32012 = extractelement <4 x float> %math.result455, i32 3, !dbg !32
  %.cvtpack.p02013 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02009, float %.cvtpack.f12010), !dbg !32
  %.cvtpack.p12014 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22011, float %.cvtpack.f32012), !dbg !32
  %.cvtpack.v02015 = insertelement <2 x i32> poison, i32 %.cvtpack.p02013, i32 0, !dbg !32
  %.cvtpack.v12016 = insertelement <2 x i32> %.cvtpack.v02015, i32 %.cvtpack.p12014, i32 1, !dbg !32
  %802 = bitcast <2 x i32> %.cvtpack.v12016 to <4 x half>, !dbg !32
  %math.packed456 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %802) #9, !dbg !30, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %803 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed456, <4 x half> %763, <4 x float> %788) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !33
  %804 = shufflevector <8 x float> %786, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !33
  %805 = fmul <4 x float> %804, %742, !dbg !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %806 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed456, <4 x half> %782, <4 x float> %805) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %807 = shufflevector <8 x half> %751, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %808 = fsub <4 x float> %583, %728, !dbg !26
  %809 = extractelement <4 x float> %808, i64 0, !dbg !30
  %810 = call float @llvm.amdgcn.exp2.f32(float %809), !dbg !30
  %local.reduce156 = fadd float %769, %810, !dbg !31, !frisk.local_reduction !1
  %811 = insertelement <4 x float> poison, float %810, i64 0, !dbg !32
  %812 = extractelement <4 x float> %808, i64 1, !dbg !30
  %813 = call float @llvm.amdgcn.exp2.f32(float %812), !dbg !30
  %local.reduce157 = fadd float %772, %813, !dbg !31, !frisk.local_reduction !1
  %814 = insertelement <4 x float> %811, float %813, i64 1, !dbg !32
  %815 = extractelement <4 x float> %808, i64 2, !dbg !30
  %816 = call float @llvm.amdgcn.exp2.f32(float %815), !dbg !30
  %local.reduce158 = fadd float %775, %816, !dbg !31, !frisk.local_reduction !1
  %817 = insertelement <4 x float> %814, float %816, i64 2, !dbg !32
  %818 = extractelement <4 x float> %808, i64 3, !dbg !30
  %819 = call float @llvm.amdgcn.exp2.f32(float %818), !dbg !30
  %local.reduce159 = fadd float %778, %819, !dbg !31, !frisk.local_reduction !1
  %820 = insertelement <4 x float> %817, float %819, i64 3, !dbg !32
  %math.result441 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %820) #9, !dbg !32, !frisk.register_lifetimes !1
  %.cvtpack.f02017 = extractelement <4 x float> %math.result441, i32 0, !dbg !32
  %.cvtpack.f12018 = extractelement <4 x float> %math.result441, i32 1, !dbg !32
  %.cvtpack.f22019 = extractelement <4 x float> %math.result441, i32 2, !dbg !32
  %.cvtpack.f32020 = extractelement <4 x float> %math.result441, i32 3, !dbg !32
  %.cvtpack.p02021 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02017, float %.cvtpack.f12018), !dbg !32
  %.cvtpack.p12022 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22019, float %.cvtpack.f32020), !dbg !32
  %.cvtpack.v02023 = insertelement <2 x i32> poison, i32 %.cvtpack.p02021, i32 0, !dbg !32
  %.cvtpack.v12024 = insertelement <2 x i32> %.cvtpack.v02023, i32 %.cvtpack.p12022, i32 1, !dbg !32
  %821 = bitcast <2 x i32> %.cvtpack.v12024 to <4 x half>, !dbg !32
  %math.packed442 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %821) #9, !dbg !31, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %822 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed442, <4 x half> %807, <4 x float> %781) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %823 = shufflevector <8 x half> %751, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %824 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed442, <4 x half> %823, <4 x float> %785) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %825 = fsub <4 x float> %642, %729, !dbg !26
  %826 = extractelement <4 x float> %825, i64 0, !dbg !30
  %827 = call float @llvm.amdgcn.exp2.f32(float %826), !dbg !30
  %local.reduce = fadd float %791, %827, !dbg !31, !frisk.local_reduction !1
  %828 = insertelement <4 x float> poison, float %827, i64 0, !dbg !32
  %829 = extractelement <4 x float> %825, i64 1, !dbg !30
  %830 = call float @llvm.amdgcn.exp2.f32(float %829), !dbg !30
  %local.reduce126 = fadd float %794, %830, !dbg !31, !frisk.local_reduction !1
  %831 = insertelement <4 x float> %828, float %830, i64 1, !dbg !32
  %832 = extractelement <4 x float> %825, i64 2, !dbg !30
  %833 = call float @llvm.amdgcn.exp2.f32(float %832), !dbg !30
  %local.reduce127 = fadd float %797, %833, !dbg !31, !frisk.local_reduction !1
  %834 = insertelement <4 x float> %831, float %833, i64 2, !dbg !32
  %835 = extractelement <4 x float> %825, i64 3, !dbg !30
  %836 = call float @llvm.amdgcn.exp2.f32(float %835), !dbg !30
  %local.reduce128 = fadd float %800, %836, !dbg !31, !frisk.local_reduction !1
  %837 = insertelement <4 x float> %834, float %836, i64 3, !dbg !32
  %math.result457 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %837) #9, !dbg !32, !frisk.register_lifetimes !1
  %.cvtpack.f02025 = extractelement <4 x float> %math.result457, i32 0, !dbg !32
  %.cvtpack.f12026 = extractelement <4 x float> %math.result457, i32 1, !dbg !32
  %.cvtpack.f22027 = extractelement <4 x float> %math.result457, i32 2, !dbg !32
  %.cvtpack.f32028 = extractelement <4 x float> %math.result457, i32 3, !dbg !32
  %.cvtpack.p02029 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02025, float %.cvtpack.f12026), !dbg !32
  %.cvtpack.p12030 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22027, float %.cvtpack.f32028), !dbg !32
  %.cvtpack.v02031 = insertelement <2 x i32> poison, i32 %.cvtpack.p02029, i32 0, !dbg !32
  %.cvtpack.v12032 = insertelement <2 x i32> %.cvtpack.v02031, i32 %.cvtpack.p12030, i32 1, !dbg !32
  %838 = bitcast <2 x i32> %.cvtpack.v12032 to <4 x half>, !dbg !32
  %math.packed458 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %838) #9, !dbg !31, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %839 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed458, <4 x half> %807, <4 x float> %803) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %840 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed458, <4 x half> %823, <4 x float> %806) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %841 = shufflevector <8 x half> %753, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %842 = fsub <4 x float> %594, %728, !dbg !26
  %843 = extractelement <4 x float> %842, i64 0, !dbg !30
  %844 = call float @llvm.amdgcn.exp2.f32(float %843), !dbg !30
  %local.reduce160 = fadd float %local.reduce156, %844, !dbg !31, !frisk.local_reduction !1
  %845 = insertelement <4 x float> poison, float %844, i64 0, !dbg !32
  %846 = extractelement <4 x float> %842, i64 1, !dbg !30
  %847 = call float @llvm.amdgcn.exp2.f32(float %846), !dbg !30
  %local.reduce161 = fadd float %local.reduce157, %847, !dbg !31, !frisk.local_reduction !1
  %848 = insertelement <4 x float> %845, float %847, i64 1, !dbg !32
  %849 = extractelement <4 x float> %842, i64 2, !dbg !30
  %850 = call float @llvm.amdgcn.exp2.f32(float %849), !dbg !30
  %local.reduce162 = fadd float %local.reduce158, %850, !dbg !31, !frisk.local_reduction !1
  %851 = insertelement <4 x float> %848, float %850, i64 2, !dbg !32
  %852 = extractelement <4 x float> %842, i64 3, !dbg !30
  %853 = call float @llvm.amdgcn.exp2.f32(float %852), !dbg !30
  %local.reduce163 = fadd float %local.reduce159, %853, !dbg !31, !frisk.local_reduction !1
  %854 = insertelement <4 x float> %851, float %853, i64 3, !dbg !32
  %math.result443 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %854) #9, !dbg !32, !frisk.register_lifetimes !1
  %.cvtpack.f02033 = extractelement <4 x float> %math.result443, i32 0, !dbg !32
  %.cvtpack.f12034 = extractelement <4 x float> %math.result443, i32 1, !dbg !32
  %.cvtpack.f22035 = extractelement <4 x float> %math.result443, i32 2, !dbg !32
  %.cvtpack.f32036 = extractelement <4 x float> %math.result443, i32 3, !dbg !32
  %.cvtpack.p02037 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02033, float %.cvtpack.f12034), !dbg !32
  %.cvtpack.p12038 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22035, float %.cvtpack.f32036), !dbg !32
  %.cvtpack.v02039 = insertelement <2 x i32> poison, i32 %.cvtpack.p02037, i32 0, !dbg !32
  %.cvtpack.v12040 = insertelement <2 x i32> %.cvtpack.v02039, i32 %.cvtpack.p12038, i32 1, !dbg !32
  %855 = bitcast <2 x i32> %.cvtpack.v12040 to <4 x half>, !dbg !32
  %math.packed444 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %855) #9, !dbg !31, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %856 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed444, <4 x half> %841, <4 x float> %822) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %857 = shufflevector <8 x half> %753, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %858 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed444, <4 x half> %857, <4 x float> %824) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %859 = fsub <4 x float> %645, %729, !dbg !26
  %860 = extractelement <4 x float> %859, i64 0, !dbg !30
  %861 = call float @llvm.amdgcn.exp2.f32(float %860), !dbg !30
  %local.reduce129 = fadd float %local.reduce, %861, !dbg !31, !frisk.local_reduction !1
  %862 = insertelement <4 x float> poison, float %861, i64 0, !dbg !32
  %863 = extractelement <4 x float> %859, i64 1, !dbg !30
  %864 = call float @llvm.amdgcn.exp2.f32(float %863), !dbg !30
  %local.reduce130 = fadd float %local.reduce126, %864, !dbg !31, !frisk.local_reduction !1
  %865 = insertelement <4 x float> %862, float %864, i64 1, !dbg !32
  %866 = extractelement <4 x float> %859, i64 2, !dbg !30
  %867 = call float @llvm.amdgcn.exp2.f32(float %866), !dbg !30
  %local.reduce131 = fadd float %local.reduce127, %867, !dbg !31, !frisk.local_reduction !1
  %868 = insertelement <4 x float> %865, float %867, i64 2, !dbg !32
  %869 = extractelement <4 x float> %859, i64 3, !dbg !30
  %870 = call float @llvm.amdgcn.exp2.f32(float %869), !dbg !30
  %local.reduce132 = fadd float %local.reduce128, %870, !dbg !31, !frisk.local_reduction !1
  %871 = insertelement <4 x float> %868, float %870, i64 3, !dbg !32
  %math.result459 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %871) #9, !dbg !32, !frisk.register_lifetimes !1
  %.cvtpack.f02041 = extractelement <4 x float> %math.result459, i32 0, !dbg !32
  %.cvtpack.f12042 = extractelement <4 x float> %math.result459, i32 1, !dbg !32
  %.cvtpack.f22043 = extractelement <4 x float> %math.result459, i32 2, !dbg !32
  %.cvtpack.f32044 = extractelement <4 x float> %math.result459, i32 3, !dbg !32
  %.cvtpack.p02045 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02041, float %.cvtpack.f12042), !dbg !32
  %.cvtpack.p12046 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22043, float %.cvtpack.f32044), !dbg !32
  %.cvtpack.v02047 = insertelement <2 x i32> poison, i32 %.cvtpack.p02045, i32 0, !dbg !32
  %.cvtpack.v12048 = insertelement <2 x i32> %.cvtpack.v02047, i32 %.cvtpack.p12046, i32 1, !dbg !32
  %872 = bitcast <2 x i32> %.cvtpack.v12048 to <4 x half>, !dbg !32
  %math.packed460 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %872) #9, !dbg !31, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %873 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed460, <4 x half> %841, <4 x float> %839) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %874 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed460, <4 x half> %857, <4 x float> %840) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %875 = shufflevector <8 x half> %755, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %876 = fsub <4 x float> %603, %728, !dbg !26
  %877 = extractelement <4 x float> %876, i64 0, !dbg !30
  %878 = call float @llvm.amdgcn.exp2.f32(float %877), !dbg !30
  %local.reduce164 = fadd float %local.reduce160, %878, !dbg !31, !frisk.local_reduction !1
  %879 = insertelement <4 x float> poison, float %878, i64 0, !dbg !32
  %880 = extractelement <4 x float> %876, i64 1, !dbg !30
  %881 = call float @llvm.amdgcn.exp2.f32(float %880), !dbg !30
  %local.reduce165 = fadd float %local.reduce161, %881, !dbg !31, !frisk.local_reduction !1
  %882 = insertelement <4 x float> %879, float %881, i64 1, !dbg !32
  %883 = extractelement <4 x float> %876, i64 2, !dbg !30
  %884 = call float @llvm.amdgcn.exp2.f32(float %883), !dbg !30
  %local.reduce166 = fadd float %local.reduce162, %884, !dbg !31, !frisk.local_reduction !1
  %885 = insertelement <4 x float> %882, float %884, i64 2, !dbg !32
  %886 = extractelement <4 x float> %876, i64 3, !dbg !30
  %887 = call float @llvm.amdgcn.exp2.f32(float %886), !dbg !30
  %local.reduce167 = fadd float %local.reduce163, %887, !dbg !31, !frisk.local_reduction !1
  %888 = insertelement <4 x float> %885, float %887, i64 3, !dbg !32
  %math.result445 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %888) #9, !dbg !32, !frisk.register_lifetimes !1
  %.cvtpack.f02049 = extractelement <4 x float> %math.result445, i32 0, !dbg !32
  %.cvtpack.f12050 = extractelement <4 x float> %math.result445, i32 1, !dbg !32
  %.cvtpack.f22051 = extractelement <4 x float> %math.result445, i32 2, !dbg !32
  %.cvtpack.f32052 = extractelement <4 x float> %math.result445, i32 3, !dbg !32
  %.cvtpack.p02053 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02049, float %.cvtpack.f12050), !dbg !32
  %.cvtpack.p12054 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22051, float %.cvtpack.f32052), !dbg !32
  %.cvtpack.v02055 = insertelement <2 x i32> poison, i32 %.cvtpack.p02053, i32 0, !dbg !32
  %.cvtpack.v12056 = insertelement <2 x i32> %.cvtpack.v02055, i32 %.cvtpack.p12054, i32 1, !dbg !32
  %889 = bitcast <2 x i32> %.cvtpack.v12056 to <4 x half>, !dbg !32
  %math.packed446 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %889) #9, !dbg !31, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %890 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed446, <4 x half> %875, <4 x float> %856) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %891 = shufflevector <8 x half> %755, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %892 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed446, <4 x half> %891, <4 x float> %858) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %893 = fsub <4 x float> %648, %729, !dbg !26
  %894 = extractelement <4 x float> %893, i64 0, !dbg !30
  %895 = call float @llvm.amdgcn.exp2.f32(float %894), !dbg !30
  %local.reduce133 = fadd float %local.reduce129, %895, !dbg !31, !frisk.local_reduction !1
  %896 = insertelement <4 x float> poison, float %895, i64 0, !dbg !32
  %897 = extractelement <4 x float> %893, i64 1, !dbg !30
  %898 = call float @llvm.amdgcn.exp2.f32(float %897), !dbg !30
  %local.reduce134 = fadd float %local.reduce130, %898, !dbg !31, !frisk.local_reduction !1
  %899 = insertelement <4 x float> %896, float %898, i64 1, !dbg !32
  %900 = extractelement <4 x float> %893, i64 2, !dbg !30
  %901 = call float @llvm.amdgcn.exp2.f32(float %900), !dbg !30
  %local.reduce135 = fadd float %local.reduce131, %901, !dbg !31, !frisk.local_reduction !1
  %902 = insertelement <4 x float> %899, float %901, i64 2, !dbg !32
  %903 = extractelement <4 x float> %893, i64 3, !dbg !30
  %904 = call float @llvm.amdgcn.exp2.f32(float %903), !dbg !30
  %local.reduce136 = fadd float %local.reduce132, %904, !dbg !31, !frisk.local_reduction !1
  %905 = insertelement <4 x float> %902, float %904, i64 3, !dbg !32
  %math.result461 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %905) #9, !dbg !32, !frisk.register_lifetimes !1
  %.cvtpack.f02057 = extractelement <4 x float> %math.result461, i32 0, !dbg !32
  %.cvtpack.f12058 = extractelement <4 x float> %math.result461, i32 1, !dbg !32
  %.cvtpack.f22059 = extractelement <4 x float> %math.result461, i32 2, !dbg !32
  %.cvtpack.f32060 = extractelement <4 x float> %math.result461, i32 3, !dbg !32
  %.cvtpack.p02061 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02057, float %.cvtpack.f12058), !dbg !32
  %.cvtpack.p12062 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22059, float %.cvtpack.f32060), !dbg !32
  %.cvtpack.v02063 = insertelement <2 x i32> poison, i32 %.cvtpack.p02061, i32 0, !dbg !32
  %.cvtpack.v12064 = insertelement <2 x i32> %.cvtpack.v02063, i32 %.cvtpack.p12062, i32 1, !dbg !32
  %906 = bitcast <2 x i32> %.cvtpack.v12064 to <4 x half>, !dbg !32
  %math.packed462 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %906) #9, !dbg !31, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %907 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed462, <4 x half> %875, <4 x float> %873) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %908 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed462, <4 x half> %891, <4 x float> %874) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %909 = shufflevector <8 x half> %757, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %910 = fsub <4 x float> %610, %728, !dbg !26
  %911 = extractelement <4 x float> %910, i64 0, !dbg !30
  %912 = call float @llvm.amdgcn.exp2.f32(float %911), !dbg !30
  %local.reduce168 = fadd float %local.reduce164, %912, !dbg !31, !frisk.local_reduction !1
  %913 = insertelement <4 x float> poison, float %912, i64 0, !dbg !32
  %914 = extractelement <4 x float> %910, i64 1, !dbg !30
  %915 = call float @llvm.amdgcn.exp2.f32(float %914), !dbg !30
  %local.reduce169 = fadd float %local.reduce165, %915, !dbg !31, !frisk.local_reduction !1
  %916 = insertelement <4 x float> %913, float %915, i64 1, !dbg !32
  %917 = extractelement <4 x float> %910, i64 2, !dbg !30
  %918 = call float @llvm.amdgcn.exp2.f32(float %917), !dbg !30
  %local.reduce170 = fadd float %local.reduce166, %918, !dbg !31, !frisk.local_reduction !1
  %919 = insertelement <4 x float> %916, float %918, i64 2, !dbg !32
  %920 = extractelement <4 x float> %910, i64 3, !dbg !30
  %921 = call float @llvm.amdgcn.exp2.f32(float %920), !dbg !30
  %local.reduce171 = fadd float %local.reduce167, %921, !dbg !31, !frisk.local_reduction !1
  %922 = insertelement <4 x float> %919, float %921, i64 3, !dbg !32
  %math.result447 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %922) #9, !dbg !32, !frisk.register_lifetimes !1
  %.cvtpack.f02065 = extractelement <4 x float> %math.result447, i32 0, !dbg !32
  %.cvtpack.f12066 = extractelement <4 x float> %math.result447, i32 1, !dbg !32
  %.cvtpack.f22067 = extractelement <4 x float> %math.result447, i32 2, !dbg !32
  %.cvtpack.f32068 = extractelement <4 x float> %math.result447, i32 3, !dbg !32
  %.cvtpack.p02069 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02065, float %.cvtpack.f12066), !dbg !32
  %.cvtpack.p12070 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22067, float %.cvtpack.f32068), !dbg !32
  %.cvtpack.v02071 = insertelement <2 x i32> poison, i32 %.cvtpack.p02069, i32 0, !dbg !32
  %.cvtpack.v12072 = insertelement <2 x i32> %.cvtpack.v02071, i32 %.cvtpack.p12070, i32 1, !dbg !32
  %923 = bitcast <2 x i32> %.cvtpack.v12072 to <4 x half>, !dbg !32
  %math.packed448 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %923) #9, !dbg !31, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %924 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed448, <4 x half> %909, <4 x float> %890) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %925 = shufflevector <8 x half> %757, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %926 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed448, <4 x half> %925, <4 x float> %892) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %927 = fsub <4 x float> %651, %729, !dbg !26
  %928 = extractelement <4 x float> %927, i64 0, !dbg !30
  %929 = call float @llvm.amdgcn.exp2.f32(float %928), !dbg !30
  %local.reduce137 = fadd float %local.reduce133, %929, !dbg !31, !frisk.local_reduction !1
  %930 = insertelement <4 x float> poison, float %929, i64 0, !dbg !32
  %931 = extractelement <4 x float> %927, i64 1, !dbg !30
  %932 = call float @llvm.amdgcn.exp2.f32(float %931), !dbg !30
  %local.reduce138 = fadd float %local.reduce134, %932, !dbg !31, !frisk.local_reduction !1
  %933 = insertelement <4 x float> %930, float %932, i64 1, !dbg !32
  %934 = extractelement <4 x float> %927, i64 2, !dbg !30
  %935 = call float @llvm.amdgcn.exp2.f32(float %934), !dbg !30
  %local.reduce139 = fadd float %local.reduce135, %935, !dbg !31, !frisk.local_reduction !1
  %936 = insertelement <4 x float> %933, float %935, i64 2, !dbg !32
  %937 = extractelement <4 x float> %927, i64 3, !dbg !30
  %938 = call float @llvm.amdgcn.exp2.f32(float %937), !dbg !30
  %local.reduce140 = fadd float %local.reduce136, %938, !dbg !31, !frisk.local_reduction !1
  %939 = insertelement <4 x float> %936, float %938, i64 3, !dbg !32
  %math.result463 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %939) #9, !dbg !32, !frisk.register_lifetimes !1
  %.cvtpack.f02073 = extractelement <4 x float> %math.result463, i32 0, !dbg !32
  %.cvtpack.f12074 = extractelement <4 x float> %math.result463, i32 1, !dbg !32
  %.cvtpack.f22075 = extractelement <4 x float> %math.result463, i32 2, !dbg !32
  %.cvtpack.f32076 = extractelement <4 x float> %math.result463, i32 3, !dbg !32
  %.cvtpack.p02077 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02073, float %.cvtpack.f12074), !dbg !32
  %.cvtpack.p12078 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22075, float %.cvtpack.f32076), !dbg !32
  %.cvtpack.v02079 = insertelement <2 x i32> poison, i32 %.cvtpack.p02077, i32 0, !dbg !32
  %.cvtpack.v12080 = insertelement <2 x i32> %.cvtpack.v02079, i32 %.cvtpack.p12078, i32 1, !dbg !32
  %940 = bitcast <2 x i32> %.cvtpack.v12080 to <4 x half>, !dbg !32
  %math.packed464 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %940) #9, !dbg !31, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %941 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed464, <4 x half> %909, <4 x float> %907) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %942 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed464, <4 x half> %925, <4 x float> %908) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %943 = shufflevector <8 x half> %759, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %944 = fsub <4 x float> %617, %728, !dbg !26
  %945 = extractelement <4 x float> %944, i64 0, !dbg !30
  %946 = call float @llvm.amdgcn.exp2.f32(float %945), !dbg !30
  %local.reduce172 = fadd float %local.reduce168, %946, !dbg !31, !frisk.local_reduction !1
  %947 = insertelement <4 x float> poison, float %946, i64 0, !dbg !32
  %948 = extractelement <4 x float> %944, i64 1, !dbg !30
  %949 = call float @llvm.amdgcn.exp2.f32(float %948), !dbg !30
  %local.reduce173 = fadd float %local.reduce169, %949, !dbg !31, !frisk.local_reduction !1
  %950 = insertelement <4 x float> %947, float %949, i64 1, !dbg !32
  %951 = extractelement <4 x float> %944, i64 2, !dbg !30
  %952 = call float @llvm.amdgcn.exp2.f32(float %951), !dbg !30
  %local.reduce174 = fadd float %local.reduce170, %952, !dbg !31, !frisk.local_reduction !1
  %953 = insertelement <4 x float> %950, float %952, i64 2, !dbg !32
  %954 = extractelement <4 x float> %944, i64 3, !dbg !30
  %955 = call float @llvm.amdgcn.exp2.f32(float %954), !dbg !30
  %local.reduce175 = fadd float %local.reduce171, %955, !dbg !31, !frisk.local_reduction !1
  %956 = insertelement <4 x float> %953, float %955, i64 3, !dbg !32
  %math.result449 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %956) #9, !dbg !32, !frisk.register_lifetimes !1
  %.cvtpack.f02081 = extractelement <4 x float> %math.result449, i32 0, !dbg !32
  %.cvtpack.f12082 = extractelement <4 x float> %math.result449, i32 1, !dbg !32
  %.cvtpack.f22083 = extractelement <4 x float> %math.result449, i32 2, !dbg !32
  %.cvtpack.f32084 = extractelement <4 x float> %math.result449, i32 3, !dbg !32
  %.cvtpack.p02085 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02081, float %.cvtpack.f12082), !dbg !32
  %.cvtpack.p12086 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22083, float %.cvtpack.f32084), !dbg !32
  %.cvtpack.v02087 = insertelement <2 x i32> poison, i32 %.cvtpack.p02085, i32 0, !dbg !32
  %.cvtpack.v12088 = insertelement <2 x i32> %.cvtpack.v02087, i32 %.cvtpack.p12086, i32 1, !dbg !32
  %957 = bitcast <2 x i32> %.cvtpack.v12088 to <4 x half>, !dbg !32
  %math.packed450 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %957) #9, !dbg !31, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %958 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed450, <4 x half> %943, <4 x float> %924) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %959 = shufflevector <8 x half> %759, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %960 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed450, <4 x half> %959, <4 x float> %926) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %961 = fsub <4 x float> %654, %729, !dbg !26
  %962 = extractelement <4 x float> %961, i64 0, !dbg !30
  %963 = call float @llvm.amdgcn.exp2.f32(float %962), !dbg !30
  %local.reduce141 = fadd float %local.reduce137, %963, !dbg !31, !frisk.local_reduction !1
  %964 = insertelement <4 x float> poison, float %963, i64 0, !dbg !32
  %965 = extractelement <4 x float> %961, i64 1, !dbg !30
  %966 = call float @llvm.amdgcn.exp2.f32(float %965), !dbg !30
  %local.reduce142 = fadd float %local.reduce138, %966, !dbg !31, !frisk.local_reduction !1
  %967 = insertelement <4 x float> %964, float %966, i64 1, !dbg !32
  %968 = extractelement <4 x float> %961, i64 2, !dbg !30
  %969 = call float @llvm.amdgcn.exp2.f32(float %968), !dbg !30
  %local.reduce143 = fadd float %local.reduce139, %969, !dbg !31, !frisk.local_reduction !1
  %970 = insertelement <4 x float> %967, float %969, i64 2, !dbg !32
  %971 = extractelement <4 x float> %961, i64 3, !dbg !30
  %972 = call float @llvm.amdgcn.exp2.f32(float %971), !dbg !30
  %local.reduce144 = fadd float %local.reduce140, %972, !dbg !31, !frisk.local_reduction !1
  %973 = insertelement <4 x float> %970, float %972, i64 3, !dbg !32
  %math.result465 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %973) #9, !dbg !32, !frisk.register_lifetimes !1
  %.cvtpack.f02089 = extractelement <4 x float> %math.result465, i32 0, !dbg !32
  %.cvtpack.f12090 = extractelement <4 x float> %math.result465, i32 1, !dbg !32
  %.cvtpack.f22091 = extractelement <4 x float> %math.result465, i32 2, !dbg !32
  %.cvtpack.f32092 = extractelement <4 x float> %math.result465, i32 3, !dbg !32
  %.cvtpack.p02093 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02089, float %.cvtpack.f12090), !dbg !32
  %.cvtpack.p12094 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22091, float %.cvtpack.f32092), !dbg !32
  %.cvtpack.v02095 = insertelement <2 x i32> poison, i32 %.cvtpack.p02093, i32 0, !dbg !32
  %.cvtpack.v12096 = insertelement <2 x i32> %.cvtpack.v02095, i32 %.cvtpack.p12094, i32 1, !dbg !32
  %974 = bitcast <2 x i32> %.cvtpack.v12096 to <4 x half>, !dbg !32
  %math.packed466 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %974) #9, !dbg !31, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %975 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed466, <4 x half> %943, <4 x float> %941) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %976 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed466, <4 x half> %959, <4 x float> %942) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %977 = shufflevector <8 x half> %761, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %978 = fsub <4 x float> %624, %728, !dbg !26
  %979 = extractelement <4 x float> %978, i64 0, !dbg !30
  %980 = call float @llvm.amdgcn.exp2.f32(float %979), !dbg !30
  %local.reduce176 = fadd float %local.reduce172, %980, !dbg !31, !frisk.local_reduction !1
  %981 = insertelement <4 x float> poison, float %980, i64 0, !dbg !32
  %982 = extractelement <4 x float> %978, i64 1, !dbg !30
  %983 = call float @llvm.amdgcn.exp2.f32(float %982), !dbg !30
  %local.reduce177 = fadd float %local.reduce173, %983, !dbg !31, !frisk.local_reduction !1
  %984 = insertelement <4 x float> %981, float %983, i64 1, !dbg !32
  %985 = extractelement <4 x float> %978, i64 2, !dbg !30
  %986 = call float @llvm.amdgcn.exp2.f32(float %985), !dbg !30
  %local.reduce178 = fadd float %local.reduce174, %986, !dbg !31, !frisk.local_reduction !1
  %987 = insertelement <4 x float> %984, float %986, i64 2, !dbg !32
  %988 = extractelement <4 x float> %978, i64 3, !dbg !30
  %989 = call float @llvm.amdgcn.exp2.f32(float %988), !dbg !30
  %local.reduce179 = fadd float %local.reduce175, %989, !dbg !31, !frisk.local_reduction !1
  %990 = insertelement <4 x float> %987, float %989, i64 3, !dbg !32
  %math.result451 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %990) #9, !dbg !32, !frisk.register_lifetimes !1
  %.cvtpack.f02097 = extractelement <4 x float> %math.result451, i32 0, !dbg !32
  %.cvtpack.f12098 = extractelement <4 x float> %math.result451, i32 1, !dbg !32
  %.cvtpack.f22099 = extractelement <4 x float> %math.result451, i32 2, !dbg !32
  %.cvtpack.f32100 = extractelement <4 x float> %math.result451, i32 3, !dbg !32
  %.cvtpack.p02101 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02097, float %.cvtpack.f12098), !dbg !32
  %.cvtpack.p12102 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22099, float %.cvtpack.f32100), !dbg !32
  %.cvtpack.v02103 = insertelement <2 x i32> poison, i32 %.cvtpack.p02101, i32 0, !dbg !32
  %.cvtpack.v12104 = insertelement <2 x i32> %.cvtpack.v02103, i32 %.cvtpack.p12102, i32 1, !dbg !32
  %991 = bitcast <2 x i32> %.cvtpack.v12104 to <4 x half>, !dbg !32
  %math.packed452 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %991) #9, !dbg !31, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %992 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed452, <4 x half> %977, <4 x float> %958) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %993 = shufflevector <8 x half> %761, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %994 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed452, <4 x half> %993, <4 x float> %960) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %995 = fsub <4 x float> %657, %729, !dbg !26
  %996 = extractelement <4 x float> %995, i64 0, !dbg !30
  %997 = call float @llvm.amdgcn.exp2.f32(float %996), !dbg !30
  %local.reduce145 = fadd float %local.reduce141, %997, !dbg !31, !frisk.local_reduction !1
  %998 = insertelement <4 x float> poison, float %997, i64 0, !dbg !32
  %999 = extractelement <4 x float> %995, i64 1, !dbg !30
  %1000 = call float @llvm.amdgcn.exp2.f32(float %999), !dbg !30
  %local.reduce146 = fadd float %local.reduce142, %1000, !dbg !31, !frisk.local_reduction !1
  %1001 = insertelement <4 x float> %998, float %1000, i64 1, !dbg !32
  %1002 = extractelement <4 x float> %995, i64 2, !dbg !30
  %1003 = call float @llvm.amdgcn.exp2.f32(float %1002), !dbg !30
  %local.reduce147 = fadd float %local.reduce143, %1003, !dbg !31, !frisk.local_reduction !1
  %1004 = insertelement <4 x float> %1001, float %1003, i64 2, !dbg !32
  %1005 = extractelement <4 x float> %995, i64 3, !dbg !30
  %1006 = call float @llvm.amdgcn.exp2.f32(float %1005), !dbg !30
  %local.reduce148 = fadd float %local.reduce144, %1006, !dbg !31, !frisk.local_reduction !1
  %1007 = insertelement <4 x float> %1004, float %1006, i64 3, !dbg !32
  %math.result467 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1007) #9, !dbg !32, !frisk.register_lifetimes !1
  %.cvtpack.f02105 = extractelement <4 x float> %math.result467, i32 0, !dbg !32
  %.cvtpack.f12106 = extractelement <4 x float> %math.result467, i32 1, !dbg !32
  %.cvtpack.f22107 = extractelement <4 x float> %math.result467, i32 2, !dbg !32
  %.cvtpack.f32108 = extractelement <4 x float> %math.result467, i32 3, !dbg !32
  %.cvtpack.p02109 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02105, float %.cvtpack.f12106), !dbg !32
  %.cvtpack.p12110 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22107, float %.cvtpack.f32108), !dbg !32
  %.cvtpack.v02111 = insertelement <2 x i32> poison, i32 %.cvtpack.p02109, i32 0, !dbg !32
  %.cvtpack.v12112 = insertelement <2 x i32> %.cvtpack.v02111, i32 %.cvtpack.p12110, i32 1, !dbg !32
  %1008 = bitcast <2 x i32> %.cvtpack.v12112 to <4 x half>, !dbg !32
  %math.packed468 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1008) #9, !dbg !31, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %1009 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed468, <4 x half> %977, <4 x float> %975) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1010 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed468, <4 x half> %993, <4 x float> %976) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1011 = bitcast <4 x i32> %762 to <8 x half>, !dbg !13
  %1012 = shufflevector <8 x half> %1011, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1013 = fsub <4 x float> %631, %728, !dbg !26
  %1014 = extractelement <4 x float> %1013, i64 0, !dbg !30
  %1015 = call float @llvm.amdgcn.exp2.f32(float %1014), !dbg !30
  %local.reduce180 = fadd float %local.reduce176, %1015, !dbg !31, !frisk.local_reduction !1
  %1016 = insertelement <4 x float> poison, float %1015, i64 0, !dbg !32
  %1017 = extractelement <4 x float> %1013, i64 1, !dbg !30
  %1018 = call float @llvm.amdgcn.exp2.f32(float %1017), !dbg !30
  %local.reduce181 = fadd float %local.reduce177, %1018, !dbg !31, !frisk.local_reduction !1
  %local.reduce184 = fadd float %local.reduce180, %local.reduce181, !dbg !31, !frisk.local_reduction !1
  %1019 = insertelement <4 x float> %1016, float %1018, i64 1, !dbg !32
  %1020 = extractelement <4 x float> %1013, i64 2, !dbg !30
  %1021 = call float @llvm.amdgcn.exp2.f32(float %1020), !dbg !30
  %local.reduce182 = fadd float %local.reduce178, %1021, !dbg !31, !frisk.local_reduction !1
  %1022 = insertelement <4 x float> %1019, float %1021, i64 2, !dbg !32
  %1023 = extractelement <4 x float> %1013, i64 3, !dbg !30
  %1024 = call float @llvm.amdgcn.exp2.f32(float %1023), !dbg !30
  %local.reduce183 = fadd float %local.reduce179, %1024, !dbg !31, !frisk.local_reduction !1
  %local.reduce185 = fadd float %local.reduce182, %local.reduce183, !dbg !31, !frisk.local_reduction !1
  %1025 = fadd float %local.reduce184, %local.reduce185, !dbg !31, !frisk.local_reduction !1
  %1026 = fadd float %1025, %737, !dbg !31
  %1027 = insertelement <1 x float> poison, float %1026, i64 0, !dbg !7
  %1028 = insertvalue [2 x <1 x float>] poison, <1 x float> %1027, 0, !dbg !7
  %1029 = insertelement <4 x float> %1022, float %1024, i64 3, !dbg !32
  %math.result453 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1029) #9, !dbg !32, !frisk.register_lifetimes !1
  %.cvtpack.f02113 = extractelement <4 x float> %math.result453, i32 0, !dbg !32
  %.cvtpack.f12114 = extractelement <4 x float> %math.result453, i32 1, !dbg !32
  %.cvtpack.f22115 = extractelement <4 x float> %math.result453, i32 2, !dbg !32
  %.cvtpack.f32116 = extractelement <4 x float> %math.result453, i32 3, !dbg !32
  %.cvtpack.p02117 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02113, float %.cvtpack.f12114), !dbg !32
  %.cvtpack.p12118 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22115, float %.cvtpack.f32116), !dbg !32
  %.cvtpack.v02119 = insertelement <2 x i32> poison, i32 %.cvtpack.p02117, i32 0, !dbg !32
  %.cvtpack.v12120 = insertelement <2 x i32> %.cvtpack.v02119, i32 %.cvtpack.p12118, i32 1, !dbg !32
  %1030 = bitcast <2 x i32> %.cvtpack.v12120 to <4 x half>, !dbg !32
  %math.packed454 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1030) #9, !dbg !31, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %1031 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed454, <4 x half> %1012, <4 x float> %992) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1032 = shufflevector <8 x half> %1011, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1033 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed454, <4 x half> %1032, <4 x float> %994) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1034 = fsub <4 x float> %671, %729, !dbg !26
  %1035 = extractelement <4 x float> %1034, i64 0, !dbg !30
  %1036 = call float @llvm.amdgcn.exp2.f32(float %1035), !dbg !30
  %local.reduce149 = fadd float %local.reduce145, %1036, !dbg !31, !frisk.local_reduction !1
  %1037 = insertelement <4 x float> poison, float %1036, i64 0, !dbg !32
  %1038 = extractelement <4 x float> %1034, i64 1, !dbg !30
  %1039 = call float @llvm.amdgcn.exp2.f32(float %1038), !dbg !30
  %local.reduce150 = fadd float %local.reduce146, %1039, !dbg !31, !frisk.local_reduction !1
  %local.reduce153 = fadd float %local.reduce149, %local.reduce150, !dbg !31, !frisk.local_reduction !1
  %1040 = insertelement <4 x float> %1037, float %1039, i64 1, !dbg !32
  %1041 = extractelement <4 x float> %1034, i64 2, !dbg !30
  %1042 = call float @llvm.amdgcn.exp2.f32(float %1041), !dbg !30
  %local.reduce151 = fadd float %local.reduce147, %1042, !dbg !31, !frisk.local_reduction !1
  %1043 = insertelement <4 x float> %1040, float %1042, i64 2, !dbg !32
  %1044 = extractelement <4 x float> %1034, i64 3, !dbg !30
  %1045 = call float @llvm.amdgcn.exp2.f32(float %1044), !dbg !30
  %local.reduce152 = fadd float %local.reduce148, %1045, !dbg !31, !frisk.local_reduction !1
  %local.reduce154 = fadd float %local.reduce151, %local.reduce152, !dbg !31, !frisk.local_reduction !1
  %1046 = fadd float %local.reduce153, %local.reduce154, !dbg !31, !frisk.local_reduction !1
  %1047 = fadd float %1046, %745, !dbg !31
  %1048 = insertelement <1 x float> poison, float %1047, i64 0, !dbg !7
  %1049 = insertvalue [2 x <1 x float>] %1028, <1 x float> %1048, 1, !dbg !7
  %1050 = insertelement <4 x float> %1043, float %1045, i64 3, !dbg !32
  %math.result469 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1050) #9, !dbg !32, !frisk.register_lifetimes !1
  %.cvtpack.f02121 = extractelement <4 x float> %math.result469, i32 0, !dbg !32
  %.cvtpack.f12122 = extractelement <4 x float> %math.result469, i32 1, !dbg !32
  %.cvtpack.f22123 = extractelement <4 x float> %math.result469, i32 2, !dbg !32
  %.cvtpack.f32124 = extractelement <4 x float> %math.result469, i32 3, !dbg !32
  %.cvtpack.p02125 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02121, float %.cvtpack.f12122), !dbg !32
  %.cvtpack.p12126 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22123, float %.cvtpack.f32124), !dbg !32
  %.cvtpack.v02127 = insertelement <2 x i32> poison, i32 %.cvtpack.p02125, i32 0, !dbg !32
  %.cvtpack.v12128 = insertelement <2 x i32> %.cvtpack.v02127, i32 %.cvtpack.p12126, i32 1, !dbg !32
  %1051 = bitcast <2 x i32> %.cvtpack.v12128 to <4 x half>, !dbg !32
  %math.packed470 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1051) #9, !dbg !31, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %1052 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed470, <4 x half> %1012, <4 x float> %1009) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1053 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed470, <4 x half> %1032, <4 x float> %1010) #9, !dbg !13, !frisk.pv_interleave !38
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
  %1054 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %138, i32 %vdb.address1561, <4 x i32> %dma.buffer.rsrc1414) #10, !dbg !17
  %1055 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %145, i32 %vdb.address1570, <4 x i32> %dma.buffer.rsrc1414) #10, !dbg !17
  %vdb.read = getelementptr i8, ptr addrspace(3) %address.local1002, i64 8192, !dbg !13
  %1056 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #0, !dbg !13, !frisk.v_double_buffer !39
  %1057 = bitcast <4 x i32> %1056 to <8 x half>, !dbg !13
  %vdb.read1550 = getelementptr i8, ptr addrspace(3) %address.local1011, i64 8448, !dbg !13
  %1058 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1550, i32 0) #0, !dbg !13, !frisk.v_double_buffer !39
  %1059 = bitcast <4 x i32> %1058 to <8 x half>, !dbg !13
  %vdb.read1551 = getelementptr i8, ptr addrspace(3) %address.local1011, i64 8704, !dbg !13
  %1060 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1551, i32 0) #0, !dbg !13, !frisk.v_double_buffer !39
  %1061 = bitcast <4 x i32> %1060 to <8 x half>, !dbg !13
  %vdb.read1552 = getelementptr i8, ptr addrspace(3) %address.local1011, i64 8960, !dbg !13
  %1062 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1552, i32 0) #0, !dbg !13, !frisk.v_double_buffer !39
  %1063 = bitcast <4 x i32> %1062 to <8 x half>, !dbg !13
  %vdb.read1553 = getelementptr i8, ptr addrspace(3) %address.local1002, i64 12288, !dbg !13
  %1064 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1553, i32 0) #0, !dbg !13, !frisk.v_double_buffer !39
  %1065 = bitcast <4 x i32> %1064 to <8 x half>, !dbg !13
  %vdb.read1554 = getelementptr i8, ptr addrspace(3) %address.local1011, i64 12544, !dbg !13
  %1066 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1554, i32 0) #0, !dbg !13, !frisk.v_double_buffer !39
  %1067 = bitcast <4 x i32> %1066 to <8 x half>, !dbg !13
  %vdb.read1555 = getelementptr i8, ptr addrspace(3) %address.local1011, i64 12800, !dbg !13
  %1068 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1555, i32 0) #0, !dbg !13, !frisk.v_double_buffer !39
  %1069 = bitcast <4 x i32> %1068 to <8 x half>, !dbg !13
  %vdb.read1556 = getelementptr i8, ptr addrspace(3) %address.local1011, i64 13056, !dbg !13
  %1070 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1556, i32 0) #0, !dbg !13, !frisk.v_double_buffer !39
  %1071 = shufflevector <8 x half> %1057, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1072 = extractvalue [2 x <8 x float>] %123, 0, !dbg !33
  %1073 = shufflevector <8 x float> %1072, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !33
  %1074 = fmul <4 x float> %1073, %734, !dbg !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1075 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1071, <4 x float> %1074) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1076 = shufflevector <8 x half> %1057, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1077 = shufflevector <8 x float> %1072, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !33
  %1078 = fmul <4 x float> %1077, %734, !dbg !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1079 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1076, <4 x float> %1078) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !33
  %1080 = extractvalue [2 x <8 x float>] %123, 1, !dbg !33
  %1081 = shufflevector <8 x float> %1080, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !33
  %1082 = fmul <4 x float> %1081, %742, !dbg !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1083 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed456, <4 x half> %1071, <4 x float> %1082) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !33
  %1084 = shufflevector <8 x float> %1080, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !33
  %1085 = fmul <4 x float> %1084, %742, !dbg !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1086 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed456, <4 x half> %1076, <4 x float> %1085) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1087 = shufflevector <8 x half> %1059, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1088 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed442, <4 x half> %1087, <4 x float> %1075) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1089 = shufflevector <8 x half> %1059, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1090 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed442, <4 x half> %1089, <4 x float> %1079) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1091 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed458, <4 x half> %1087, <4 x float> %1083) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1092 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed458, <4 x half> %1089, <4 x float> %1086) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1093 = shufflevector <8 x half> %1061, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1094 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed444, <4 x half> %1093, <4 x float> %1088) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1095 = shufflevector <8 x half> %1061, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1096 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed444, <4 x half> %1095, <4 x float> %1090) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1097 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed460, <4 x half> %1093, <4 x float> %1091) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1098 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed460, <4 x half> %1095, <4 x float> %1092) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1099 = shufflevector <8 x half> %1063, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1100 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed446, <4 x half> %1099, <4 x float> %1094) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1101 = shufflevector <8 x half> %1063, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1102 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed446, <4 x half> %1101, <4 x float> %1096) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1103 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed462, <4 x half> %1099, <4 x float> %1097) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1104 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed462, <4 x half> %1101, <4 x float> %1098) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1105 = shufflevector <8 x half> %1065, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1106 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed448, <4 x half> %1105, <4 x float> %1100) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1107 = shufflevector <8 x half> %1065, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1108 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed448, <4 x half> %1107, <4 x float> %1102) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1109 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed464, <4 x half> %1105, <4 x float> %1103) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1110 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed464, <4 x half> %1107, <4 x float> %1104) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1111 = shufflevector <8 x half> %1067, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1112 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed450, <4 x half> %1111, <4 x float> %1106) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1113 = shufflevector <8 x half> %1067, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1114 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed450, <4 x half> %1113, <4 x float> %1108) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1115 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed466, <4 x half> %1111, <4 x float> %1109) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1116 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed466, <4 x half> %1113, <4 x float> %1110) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1117 = shufflevector <8 x half> %1069, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1118 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed452, <4 x half> %1117, <4 x float> %1112) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1119 = shufflevector <8 x half> %1069, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1120 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed452, <4 x half> %1119, <4 x float> %1114) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1121 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed468, <4 x half> %1117, <4 x float> %1115) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1122 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed468, <4 x half> %1119, <4 x float> %1116) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1123 = bitcast <4 x i32> %1070 to <8 x half>, !dbg !13
  %1124 = shufflevector <8 x half> %1123, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1125 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed454, <4 x half> %1124, <4 x float> %1118) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1126 = shufflevector <8 x half> %1123, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1127 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed454, <4 x half> %1126, <4 x float> %1120) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1128 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed470, <4 x half> %1124, <4 x float> %1121) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1129 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed470, <4 x half> %1126, <4 x float> %1122) #9, !dbg !13, !frisk.pv_interleave !38
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
  %1130 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1579, <4 x i32> %dma.buffer.rsrc1414) #10, !dbg !17
  %1131 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1549, i32 %vdb.address1588, <4 x i32> %dma.buffer.rsrc1414) #10, !dbg !17
  %1132 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1002, i32 0) #0, !dbg !13, !frisk.v_double_buffer !39
  %1133 = bitcast <4 x i32> %1132 to <8 x half>, !dbg !13
  %1134 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1012, i32 0) #0, !dbg !13, !frisk.v_double_buffer !39
  %1135 = bitcast <4 x i32> %1134 to <8 x half>, !dbg !13
  %1136 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1016, i32 0) #0, !dbg !13, !frisk.v_double_buffer !39
  %1137 = bitcast <4 x i32> %1136 to <8 x half>, !dbg !13
  %1138 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1020, i32 0) #0, !dbg !13, !frisk.v_double_buffer !39
  %1139 = bitcast <4 x i32> %1138 to <8 x half>, !dbg !13
  %1140 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1022, i32 0) #0, !dbg !13, !frisk.v_double_buffer !39
  %1141 = bitcast <4 x i32> %1140 to <8 x half>, !dbg !13
  %1142 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1026, i32 0) #0, !dbg !13, !frisk.v_double_buffer !39
  %1143 = bitcast <4 x i32> %1142 to <8 x half>, !dbg !13
  %1144 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1030, i32 0) #0, !dbg !13, !frisk.v_double_buffer !39
  %1145 = bitcast <4 x i32> %1144 to <8 x half>, !dbg !13
  %1146 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1034, i32 0) #0, !dbg !13, !frisk.v_double_buffer !39
  %1147 = shufflevector <8 x half> %1133, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1148 = extractvalue [2 x <8 x float>] %124, 0, !dbg !33
  %1149 = shufflevector <8 x float> %1148, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !33
  %1150 = fmul <4 x float> %1149, %734, !dbg !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1151 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1147, <4 x float> %1150) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1152 = shufflevector <8 x half> %1133, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1153 = shufflevector <8 x float> %1148, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !33
  %1154 = fmul <4 x float> %1153, %734, !dbg !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1155 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1152, <4 x float> %1154) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !33
  %1156 = extractvalue [2 x <8 x float>] %124, 1, !dbg !33
  %1157 = shufflevector <8 x float> %1156, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !33
  %1158 = fmul <4 x float> %1157, %742, !dbg !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1159 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed456, <4 x half> %1147, <4 x float> %1158) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !33
  %1160 = shufflevector <8 x float> %1156, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !33
  %1161 = fmul <4 x float> %1160, %742, !dbg !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1162 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed456, <4 x half> %1152, <4 x float> %1161) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1163 = shufflevector <8 x half> %1135, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1164 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed442, <4 x half> %1163, <4 x float> %1151) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1165 = shufflevector <8 x half> %1135, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1166 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed442, <4 x half> %1165, <4 x float> %1155) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1167 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed458, <4 x half> %1163, <4 x float> %1159) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1168 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed458, <4 x half> %1165, <4 x float> %1162) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1169 = shufflevector <8 x half> %1137, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1170 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed444, <4 x half> %1169, <4 x float> %1164) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1171 = shufflevector <8 x half> %1137, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1172 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed444, <4 x half> %1171, <4 x float> %1166) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1173 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed460, <4 x half> %1169, <4 x float> %1167) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1174 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed460, <4 x half> %1171, <4 x float> %1168) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1175 = shufflevector <8 x half> %1139, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1176 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed446, <4 x half> %1175, <4 x float> %1170) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1177 = shufflevector <8 x half> %1139, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1178 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed446, <4 x half> %1177, <4 x float> %1172) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1179 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed462, <4 x half> %1175, <4 x float> %1173) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1180 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed462, <4 x half> %1177, <4 x float> %1174) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1181 = shufflevector <8 x half> %1141, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1182 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed448, <4 x half> %1181, <4 x float> %1176) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1183 = shufflevector <8 x half> %1141, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1184 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed448, <4 x half> %1183, <4 x float> %1178) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1185 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed464, <4 x half> %1181, <4 x float> %1179) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1186 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed464, <4 x half> %1183, <4 x float> %1180) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1187 = shufflevector <8 x half> %1143, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1188 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed450, <4 x half> %1187, <4 x float> %1182) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1189 = shufflevector <8 x half> %1143, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1190 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed450, <4 x half> %1189, <4 x float> %1184) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1191 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed466, <4 x half> %1187, <4 x float> %1185) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1192 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed466, <4 x half> %1189, <4 x float> %1186) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1193 = shufflevector <8 x half> %1145, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1194 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed452, <4 x half> %1193, <4 x float> %1188) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1195 = shufflevector <8 x half> %1145, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1196 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed452, <4 x half> %1195, <4 x float> %1190) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1197 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed468, <4 x half> %1193, <4 x float> %1191) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1198 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed468, <4 x half> %1195, <4 x float> %1192) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1199 = bitcast <4 x i32> %1146 to <8 x half>, !dbg !13
  %1200 = shufflevector <8 x half> %1199, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1201 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed454, <4 x half> %1200, <4 x float> %1194) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1202 = shufflevector <8 x half> %1199, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1203 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed454, <4 x half> %1202, <4 x float> %1196) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1204 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed470, <4 x half> %1200, <4 x float> %1197) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1205 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed470, <4 x half> %1202, <4 x float> %1198) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %1206 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #0, !dbg !13, !frisk.v_double_buffer !39
  %1207 = bitcast <4 x i32> %1206 to <8 x half>, !dbg !13
  %1208 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1550, i32 0) #0, !dbg !13, !frisk.v_double_buffer !39
  %1209 = bitcast <4 x i32> %1208 to <8 x half>, !dbg !13
  %1210 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1551, i32 0) #0, !dbg !13, !frisk.v_double_buffer !39
  %1211 = bitcast <4 x i32> %1210 to <8 x half>, !dbg !13
  %1212 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1552, i32 0) #0, !dbg !13, !frisk.v_double_buffer !39
  %1213 = bitcast <4 x i32> %1212 to <8 x half>, !dbg !13
  %1214 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1553, i32 0) #0, !dbg !13, !frisk.v_double_buffer !39
  %1215 = bitcast <4 x i32> %1214 to <8 x half>, !dbg !13
  %1216 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1554, i32 0) #0, !dbg !13, !frisk.v_double_buffer !39
  %1217 = bitcast <4 x i32> %1216 to <8 x half>, !dbg !13
  %1218 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1555, i32 0) #0, !dbg !13, !frisk.v_double_buffer !39
  %1219 = bitcast <4 x i32> %1218 to <8 x half>, !dbg !13
  %1220 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1556, i32 0) #0, !dbg !13, !frisk.v_double_buffer !39
  %1221 = shufflevector <8 x half> %1207, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %1222 = extractvalue [2 x <8 x float>] %125, 0, !dbg !33
  %1223 = shufflevector <8 x float> %1222, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !33
  %1224 = fmul <4 x float> %1223, %734, !dbg !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1225 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1221, <4 x float> %1224) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1226 = shufflevector <8 x half> %1207, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1227 = shufflevector <8 x float> %1222, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !33
  %1228 = fmul <4 x float> %1227, %734, !dbg !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1229 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1226, <4 x float> %1228) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !33
  %1230 = extractvalue [2 x <8 x float>] %125, 1, !dbg !33
  %1231 = shufflevector <8 x float> %1230, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !33
  %1232 = fmul <4 x float> %1231, %742, !dbg !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1233 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed456, <4 x half> %1221, <4 x float> %1232) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !33
  %1234 = shufflevector <8 x float> %1230, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !33
  %1235 = fmul <4 x float> %1234, %742, !dbg !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1236 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed456, <4 x half> %1226, <4 x float> %1235) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1237 = shufflevector <8 x half> %1209, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1238 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed442, <4 x half> %1237, <4 x float> %1225) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1239 = shufflevector <8 x half> %1209, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1240 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed442, <4 x half> %1239, <4 x float> %1229) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1241 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed458, <4 x half> %1237, <4 x float> %1233) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1242 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed458, <4 x half> %1239, <4 x float> %1236) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1243 = shufflevector <8 x half> %1211, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1244 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed444, <4 x half> %1243, <4 x float> %1238) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1245 = shufflevector <8 x half> %1211, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1246 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed444, <4 x half> %1245, <4 x float> %1240) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1247 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed460, <4 x half> %1243, <4 x float> %1241) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1248 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed460, <4 x half> %1245, <4 x float> %1242) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1249 = shufflevector <8 x half> %1213, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1250 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed446, <4 x half> %1249, <4 x float> %1244) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1251 = shufflevector <8 x half> %1213, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1252 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed446, <4 x half> %1251, <4 x float> %1246) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1253 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed462, <4 x half> %1249, <4 x float> %1247) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1254 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed462, <4 x half> %1251, <4 x float> %1248) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1255 = shufflevector <8 x half> %1215, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1256 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed448, <4 x half> %1255, <4 x float> %1250) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1257 = shufflevector <8 x half> %1215, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1258 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed448, <4 x half> %1257, <4 x float> %1252) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1259 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed464, <4 x half> %1255, <4 x float> %1253) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1260 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed464, <4 x half> %1257, <4 x float> %1254) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1261 = shufflevector <8 x half> %1217, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1262 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed450, <4 x half> %1261, <4 x float> %1256) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1263 = shufflevector <8 x half> %1217, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1264 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed450, <4 x half> %1263, <4 x float> %1258) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1265 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed466, <4 x half> %1261, <4 x float> %1259) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1266 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed466, <4 x half> %1263, <4 x float> %1260) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1267 = shufflevector <8 x half> %1219, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1268 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed452, <4 x half> %1267, <4 x float> %1262) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1269 = shufflevector <8 x half> %1219, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1270 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed452, <4 x half> %1269, <4 x float> %1264) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1271 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed468, <4 x half> %1267, <4 x float> %1265) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1272 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed468, <4 x half> %1269, <4 x float> %1266) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1273 = bitcast <4 x i32> %1220 to <8 x half>, !dbg !13
  %1274 = shufflevector <8 x half> %1273, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1275 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed454, <4 x half> %1274, <4 x float> %1268) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1276 = shufflevector <8 x half> %1273, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1277 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed454, <4 x half> %1276, <4 x float> %1270) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1278 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed470, <4 x half> %1274, <4 x float> %1271) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1279 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed470, <4 x half> %1276, <4 x float> %1272) #9, !dbg !13, !frisk.pv_interleave !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !7
  %1280 = shufflevector <4 x float> %1031, <4 x float> %1033, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1281 = insertvalue [2 x <8 x float>] poison, <8 x float> %1280, 0, !dbg !7
  %1282 = shufflevector <4 x float> %1052, <4 x float> %1053, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1283 = insertvalue [2 x <8 x float>] %1281, <8 x float> %1282, 1, !dbg !7
  %1284 = insertvalue [2 x <1 x float>] poison, <1 x float> %724, 0, !dbg !7
  %1285 = insertvalue [2 x <1 x float>] %1284, <1 x float> %727, 1, !dbg !7
  %1286 = shufflevector <4 x float> %1125, <4 x float> %1127, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1287 = insertvalue [2 x <8 x float>] poison, <8 x float> %1286, 0, !dbg !7
  %1288 = shufflevector <4 x float> %1128, <4 x float> %1129, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1289 = insertvalue [2 x <8 x float>] %1287, <8 x float> %1288, 1, !dbg !7
  %1290 = shufflevector <4 x float> %1201, <4 x float> %1203, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1291 = insertvalue [2 x <8 x float>] poison, <8 x float> %1290, 0, !dbg !7
  %1292 = shufflevector <4 x float> %1204, <4 x float> %1205, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1293 = insertvalue [2 x <8 x float>] %1291, <8 x float> %1292, 1, !dbg !7
  %1294 = shufflevector <4 x float> %1275, <4 x float> %1277, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1295 = insertvalue [2 x <8 x float>] poison, <8 x float> %1294, 0, !dbg !7
  %1296 = shufflevector <4 x float> %1278, <4 x float> %1279, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1297 = insertvalue [2 x <8 x float>] %1295, <8 x float> %1296, 1, !dbg !7
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %1298 = add nuw nsw i64 %119, 1, !dbg !7
  br label %118, !dbg !7

1299:                                             ; preds = %118
  %lane.sum.field251 = extractvalue [2 x <1 x float>] %121, 1, !dbg !7
  %lane.sum.local252 = extractelement <1 x float> %lane.sum.field251, i64 0, !dbg !7
  %lane.sum.lane.lo253 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !7
  %lane.sum.lane254 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %lane.sum.lane.lo253), !dbg !7
  %1300 = shl nuw nsw i32 %lane.sum.lane254, 2, !dbg !7
  %1301 = xor i32 %1300, 64, !dbg !7
  %1302 = bitcast float %lane.sum.local252 to i32, !dbg !7
  %lane.sum.shuffle255 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1301, i32 %1302), !dbg !7
  %1303 = bitcast i32 %lane.sum.shuffle255 to float, !dbg !7
  %lane.sum.final256 = fadd float %lane.sum.local252, %1303, !dbg !7
  %1304 = xor i32 %1300, 128, !dbg !7
  %1305 = bitcast float %lane.sum.final256 to i32, !dbg !7
  %lane.sum.shuffle257 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1304, i32 %1305), !dbg !7
  %1306 = bitcast i32 %lane.sum.shuffle257 to float, !dbg !7
  %lane.sum.final258 = fadd float %lane.sum.final256, %1306, !dbg !7
  %1307 = insertelement <1 x float> poison, float %lane.sum.final258, i64 0, !dbg !7
  %lane.sum.field = extractvalue [2 x <1 x float>] %121, 0, !dbg !7
  %lane.sum.local = extractelement <1 x float> %lane.sum.field, i64 0, !dbg !7
  %1308 = bitcast float %lane.sum.local to i32, !dbg !7
  %lane.sum.shuffle = call i32 @llvm.amdgcn.ds.bpermute(i32 %1301, i32 %1308), !dbg !7
  %1309 = bitcast i32 %lane.sum.shuffle to float, !dbg !7
  %lane.sum.final = fadd float %lane.sum.local, %1309, !dbg !7
  %1310 = bitcast float %lane.sum.final to i32, !dbg !7
  %lane.sum.shuffle249 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1304, i32 %1310), !dbg !7
  %1311 = bitcast i32 %lane.sum.shuffle249 to float, !dbg !7
  %lane.sum.final250 = fadd float %lane.sum.final, %1311, !dbg !7
  %1312 = insertelement <1 x float> poison, float %lane.sum.final250, i64 0, !dbg !7
  %1313 = and i32 %kdma.tid, 16, !dbg !7
  %.not = icmp eq i32 %1313, 0, !dbg !7
  %1314 = and i32 %kdma.tid, 32, !dbg !7
  %.not36 = icmp eq i32 %1314, 0, !dbg !7
  %1315 = extractvalue [2 x <8 x float>] %125, 0, !dbg !7
  %1316 = extractelement <8 x float> %1315, i64 0, !dbg !7
  %1317 = extractelement <8 x float> %1315, i64 1, !dbg !7
  %1318 = extractelement <8 x float> %1315, i64 2, !dbg !7
  %1319 = extractelement <8 x float> %1315, i64 3, !dbg !7
  %1320 = select i1 %.not, float %1317, float %1316, !dbg !7
  %1321 = add nuw nsw i32 %lane.sum.lane254, 64, !dbg !7
  %1322 = xor i32 %lane.sum.lane254, 16, !dbg !7
  %1323 = and i32 %1321, 192, !dbg !7
  %1324 = icmp samesign ult i32 %1322, %1323, !dbg !7
  %1325 = select i1 %1324, i32 %1322, i32 %lane.sum.lane254, !dbg !7
  %1326 = shl nuw nsw i32 %1325, 2, !dbg !7
  %1327 = bitcast float %1320 to i32, !dbg !7
  %1328 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1326, i32 %1327), !dbg !7
  %1329 = bitcast i32 %1328 to float, !dbg !7
  %1330 = select i1 %.not, float %1316, float %1329, !dbg !7
  %1331 = select i1 %.not, float %1329, float %1317, !dbg !7
  %1332 = select i1 %.not, float %1319, float %1318, !dbg !7
  %1333 = bitcast float %1332 to i32, !dbg !7
  %1334 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1326, i32 %1333), !dbg !7
  %1335 = bitcast i32 %1334 to float, !dbg !7
  %1336 = select i1 %.not, float %1318, float %1335, !dbg !7
  %1337 = select i1 %.not, float %1335, float %1319, !dbg !7
  %1338 = select i1 %.not36, float %1336, float %1330, !dbg !7
  %1339 = xor i32 %lane.sum.lane254, 32, !dbg !7
  %1340 = icmp samesign ult i32 %1339, %1323, !dbg !7
  %1341 = select i1 %1340, i32 %1339, i32 %lane.sum.lane254, !dbg !7
  %1342 = shl nuw nsw i32 %1341, 2, !dbg !7
  %1343 = bitcast float %1338 to i32, !dbg !7
  %1344 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1342, i32 %1343), !dbg !7
  %1345 = bitcast i32 %1344 to float, !dbg !7
  %1346 = select i1 %.not36, float %1330, float %1345, !dbg !7
  %1347 = select i1 %.not36, float %1345, float %1336, !dbg !7
  %1348 = select i1 %.not36, float %1337, float %1331, !dbg !7
  %1349 = bitcast float %1348 to i32, !dbg !7
  %1350 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1342, i32 %1349), !dbg !7
  %1351 = bitcast i32 %1350 to float, !dbg !7
  %1352 = select i1 %.not36, float %1331, float %1351, !dbg !7
  %1353 = select i1 %.not36, float %1351, float %1337, !dbg !7
  %1354 = extractelement <8 x float> %1315, i64 4, !dbg !7
  %1355 = extractelement <8 x float> %1315, i64 5, !dbg !7
  %1356 = extractelement <8 x float> %1315, i64 6, !dbg !7
  %1357 = extractelement <8 x float> %1315, i64 7, !dbg !7
  %1358 = select i1 %.not, float %1355, float %1354, !dbg !7
  %1359 = bitcast float %1358 to i32, !dbg !7
  %1360 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1326, i32 %1359), !dbg !7
  %1361 = bitcast i32 %1360 to float, !dbg !7
  %1362 = select i1 %.not, float %1354, float %1361, !dbg !7
  %1363 = select i1 %.not, float %1361, float %1355, !dbg !7
  %1364 = select i1 %.not, float %1357, float %1356, !dbg !7
  %1365 = bitcast float %1364 to i32, !dbg !7
  %1366 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1326, i32 %1365), !dbg !7
  %1367 = bitcast i32 %1366 to float, !dbg !7
  %1368 = select i1 %.not, float %1356, float %1367, !dbg !7
  %1369 = select i1 %.not, float %1367, float %1357, !dbg !7
  %1370 = select i1 %.not36, float %1368, float %1362, !dbg !7
  %1371 = bitcast float %1370 to i32, !dbg !7
  %1372 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1342, i32 %1371), !dbg !7
  %1373 = bitcast i32 %1372 to float, !dbg !7
  %1374 = select i1 %.not36, float %1362, float %1373, !dbg !7
  %1375 = select i1 %.not36, float %1373, float %1368, !dbg !7
  %1376 = select i1 %.not36, float %1369, float %1363, !dbg !7
  %1377 = bitcast float %1376 to i32, !dbg !7
  %1378 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1342, i32 %1377), !dbg !7
  %1379 = bitcast i32 %1378 to float, !dbg !7
  %1380 = select i1 %.not36, float %1363, float %1379, !dbg !7
  %1381 = select i1 %.not36, float %1379, float %1369, !dbg !7
  %1382 = extractvalue [2 x <8 x float>] %125, 1, !dbg !7
  %1383 = extractelement <8 x float> %1382, i64 0, !dbg !7
  %1384 = extractelement <8 x float> %1382, i64 1, !dbg !7
  %1385 = extractelement <8 x float> %1382, i64 2, !dbg !7
  %1386 = extractelement <8 x float> %1382, i64 3, !dbg !7
  %1387 = select i1 %.not, float %1384, float %1383, !dbg !7
  %1388 = bitcast float %1387 to i32, !dbg !7
  %1389 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1326, i32 %1388), !dbg !7
  %1390 = bitcast i32 %1389 to float, !dbg !7
  %1391 = select i1 %.not, float %1383, float %1390, !dbg !7
  %1392 = select i1 %.not, float %1390, float %1384, !dbg !7
  %1393 = select i1 %.not, float %1386, float %1385, !dbg !7
  %1394 = bitcast float %1393 to i32, !dbg !7
  %1395 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1326, i32 %1394), !dbg !7
  %1396 = bitcast i32 %1395 to float, !dbg !7
  %1397 = select i1 %.not, float %1385, float %1396, !dbg !7
  %1398 = select i1 %.not, float %1396, float %1386, !dbg !7
  %1399 = select i1 %.not36, float %1397, float %1391, !dbg !7
  %1400 = bitcast float %1399 to i32, !dbg !7
  %1401 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1342, i32 %1400), !dbg !7
  %1402 = bitcast i32 %1401 to float, !dbg !7
  %1403 = select i1 %.not36, float %1391, float %1402, !dbg !7
  %1404 = select i1 %.not36, float %1402, float %1397, !dbg !7
  %1405 = select i1 %.not36, float %1398, float %1392, !dbg !7
  %1406 = bitcast float %1405 to i32, !dbg !7
  %1407 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1342, i32 %1406), !dbg !7
  %1408 = bitcast i32 %1407 to float, !dbg !7
  %1409 = select i1 %.not36, float %1392, float %1408, !dbg !7
  %1410 = select i1 %.not36, float %1408, float %1398, !dbg !7
  %1411 = extractelement <8 x float> %1382, i64 4, !dbg !7
  %1412 = extractelement <8 x float> %1382, i64 5, !dbg !7
  %1413 = extractelement <8 x float> %1382, i64 6, !dbg !7
  %1414 = extractelement <8 x float> %1382, i64 7, !dbg !7
  %1415 = select i1 %.not, float %1412, float %1411, !dbg !7
  %1416 = bitcast float %1415 to i32, !dbg !7
  %1417 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1326, i32 %1416), !dbg !7
  %1418 = bitcast i32 %1417 to float, !dbg !7
  %1419 = select i1 %.not, float %1411, float %1418, !dbg !7
  %1420 = select i1 %.not, float %1418, float %1412, !dbg !7
  %1421 = select i1 %.not, float %1414, float %1413, !dbg !7
  %1422 = bitcast float %1421 to i32, !dbg !7
  %1423 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1326, i32 %1422), !dbg !7
  %1424 = bitcast i32 %1423 to float, !dbg !7
  %1425 = select i1 %.not, float %1413, float %1424, !dbg !7
  %1426 = select i1 %.not, float %1424, float %1414, !dbg !7
  %1427 = select i1 %.not36, float %1425, float %1419, !dbg !7
  %1428 = bitcast float %1427 to i32, !dbg !7
  %1429 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1342, i32 %1428), !dbg !7
  %1430 = bitcast i32 %1429 to float, !dbg !7
  %1431 = select i1 %.not36, float %1419, float %1430, !dbg !7
  %1432 = select i1 %.not36, float %1430, float %1425, !dbg !7
  %1433 = select i1 %.not36, float %1426, float %1420, !dbg !7
  %1434 = bitcast float %1433 to i32, !dbg !7
  %1435 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1342, i32 %1434), !dbg !7
  %1436 = bitcast i32 %1435 to float, !dbg !7
  %1437 = select i1 %.not36, float %1420, float %1436, !dbg !7
  %1438 = select i1 %.not36, float %1436, float %1426, !dbg !7
  %1439 = extractvalue [2 x <8 x float>] %124, 0, !dbg !7
  %1440 = extractelement <8 x float> %1439, i64 0, !dbg !7
  %1441 = extractelement <8 x float> %1439, i64 1, !dbg !7
  %1442 = extractelement <8 x float> %1439, i64 2, !dbg !7
  %1443 = extractelement <8 x float> %1439, i64 3, !dbg !7
  %1444 = select i1 %.not, float %1441, float %1440, !dbg !7
  %1445 = bitcast float %1444 to i32, !dbg !7
  %1446 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1326, i32 %1445), !dbg !7
  %1447 = bitcast i32 %1446 to float, !dbg !7
  %1448 = select i1 %.not, float %1440, float %1447, !dbg !7
  %1449 = select i1 %.not, float %1447, float %1441, !dbg !7
  %1450 = select i1 %.not, float %1443, float %1442, !dbg !7
  %1451 = bitcast float %1450 to i32, !dbg !7
  %1452 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1326, i32 %1451), !dbg !7
  %1453 = bitcast i32 %1452 to float, !dbg !7
  %1454 = select i1 %.not, float %1442, float %1453, !dbg !7
  %1455 = select i1 %.not, float %1453, float %1443, !dbg !7
  %1456 = select i1 %.not36, float %1454, float %1448, !dbg !7
  %1457 = bitcast float %1456 to i32, !dbg !7
  %1458 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1342, i32 %1457), !dbg !7
  %1459 = bitcast i32 %1458 to float, !dbg !7
  %1460 = select i1 %.not36, float %1448, float %1459, !dbg !7
  %1461 = select i1 %.not36, float %1459, float %1454, !dbg !7
  %1462 = select i1 %.not36, float %1455, float %1449, !dbg !7
  %1463 = bitcast float %1462 to i32, !dbg !7
  %1464 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1342, i32 %1463), !dbg !7
  %1465 = bitcast i32 %1464 to float, !dbg !7
  %1466 = select i1 %.not36, float %1449, float %1465, !dbg !7
  %1467 = select i1 %.not36, float %1465, float %1455, !dbg !7
  %1468 = extractelement <8 x float> %1439, i64 4, !dbg !7
  %1469 = extractelement <8 x float> %1439, i64 5, !dbg !7
  %1470 = extractelement <8 x float> %1439, i64 6, !dbg !7
  %1471 = extractelement <8 x float> %1439, i64 7, !dbg !7
  %1472 = select i1 %.not, float %1469, float %1468, !dbg !7
  %1473 = bitcast float %1472 to i32, !dbg !7
  %1474 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1326, i32 %1473), !dbg !7
  %1475 = bitcast i32 %1474 to float, !dbg !7
  %1476 = select i1 %.not, float %1468, float %1475, !dbg !7
  %1477 = select i1 %.not, float %1475, float %1469, !dbg !7
  %1478 = select i1 %.not, float %1471, float %1470, !dbg !7
  %1479 = bitcast float %1478 to i32, !dbg !7
  %1480 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1326, i32 %1479), !dbg !7
  %1481 = bitcast i32 %1480 to float, !dbg !7
  %1482 = select i1 %.not, float %1470, float %1481, !dbg !7
  %1483 = select i1 %.not, float %1481, float %1471, !dbg !7
  %1484 = select i1 %.not36, float %1482, float %1476, !dbg !7
  %1485 = bitcast float %1484 to i32, !dbg !7
  %1486 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1342, i32 %1485), !dbg !7
  %1487 = bitcast i32 %1486 to float, !dbg !7
  %1488 = select i1 %.not36, float %1476, float %1487, !dbg !7
  %1489 = select i1 %.not36, float %1487, float %1482, !dbg !7
  %1490 = select i1 %.not36, float %1483, float %1477, !dbg !7
  %1491 = bitcast float %1490 to i32, !dbg !7
  %1492 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1342, i32 %1491), !dbg !7
  %1493 = bitcast i32 %1492 to float, !dbg !7
  %1494 = select i1 %.not36, float %1477, float %1493, !dbg !7
  %1495 = select i1 %.not36, float %1493, float %1483, !dbg !7
  %1496 = extractvalue [2 x <8 x float>] %124, 1, !dbg !7
  %1497 = extractelement <8 x float> %1496, i64 0, !dbg !7
  %1498 = extractelement <8 x float> %1496, i64 1, !dbg !7
  %1499 = extractelement <8 x float> %1496, i64 2, !dbg !7
  %1500 = extractelement <8 x float> %1496, i64 3, !dbg !7
  %1501 = select i1 %.not, float %1498, float %1497, !dbg !7
  %1502 = bitcast float %1501 to i32, !dbg !7
  %1503 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1326, i32 %1502), !dbg !7
  %1504 = bitcast i32 %1503 to float, !dbg !7
  %1505 = select i1 %.not, float %1497, float %1504, !dbg !7
  %1506 = select i1 %.not, float %1504, float %1498, !dbg !7
  %1507 = select i1 %.not, float %1500, float %1499, !dbg !7
  %1508 = bitcast float %1507 to i32, !dbg !7
  %1509 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1326, i32 %1508), !dbg !7
  %1510 = bitcast i32 %1509 to float, !dbg !7
  %1511 = select i1 %.not, float %1499, float %1510, !dbg !7
  %1512 = select i1 %.not, float %1510, float %1500, !dbg !7
  %1513 = select i1 %.not36, float %1511, float %1505, !dbg !7
  %1514 = bitcast float %1513 to i32, !dbg !7
  %1515 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1342, i32 %1514), !dbg !7
  %1516 = bitcast i32 %1515 to float, !dbg !7
  %1517 = select i1 %.not36, float %1505, float %1516, !dbg !7
  %1518 = select i1 %.not36, float %1516, float %1511, !dbg !7
  %1519 = select i1 %.not36, float %1512, float %1506, !dbg !7
  %1520 = bitcast float %1519 to i32, !dbg !7
  %1521 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1342, i32 %1520), !dbg !7
  %1522 = bitcast i32 %1521 to float, !dbg !7
  %1523 = select i1 %.not36, float %1506, float %1522, !dbg !7
  %1524 = select i1 %.not36, float %1522, float %1512, !dbg !7
  %1525 = extractelement <8 x float> %1496, i64 4, !dbg !7
  %1526 = extractelement <8 x float> %1496, i64 5, !dbg !7
  %1527 = extractelement <8 x float> %1496, i64 6, !dbg !7
  %1528 = extractelement <8 x float> %1496, i64 7, !dbg !7
  %1529 = select i1 %.not, float %1526, float %1525, !dbg !7
  %1530 = bitcast float %1529 to i32, !dbg !7
  %1531 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1326, i32 %1530), !dbg !7
  %1532 = bitcast i32 %1531 to float, !dbg !7
  %1533 = select i1 %.not, float %1525, float %1532, !dbg !7
  %1534 = select i1 %.not, float %1532, float %1526, !dbg !7
  %1535 = select i1 %.not, float %1528, float %1527, !dbg !7
  %1536 = bitcast float %1535 to i32, !dbg !7
  %1537 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1326, i32 %1536), !dbg !7
  %1538 = bitcast i32 %1537 to float, !dbg !7
  %1539 = select i1 %.not, float %1527, float %1538, !dbg !7
  %1540 = select i1 %.not, float %1538, float %1528, !dbg !7
  %1541 = select i1 %.not36, float %1539, float %1533, !dbg !7
  %1542 = bitcast float %1541 to i32, !dbg !7
  %1543 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1342, i32 %1542), !dbg !7
  %1544 = bitcast i32 %1543 to float, !dbg !7
  %1545 = select i1 %.not36, float %1533, float %1544, !dbg !7
  %1546 = select i1 %.not36, float %1544, float %1539, !dbg !7
  %1547 = select i1 %.not36, float %1540, float %1534, !dbg !7
  %1548 = bitcast float %1547 to i32, !dbg !7
  %1549 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1342, i32 %1548), !dbg !7
  %1550 = bitcast i32 %1549 to float, !dbg !7
  %1551 = select i1 %.not36, float %1534, float %1550, !dbg !7
  %1552 = select i1 %.not36, float %1550, float %1540, !dbg !7
  %1553 = extractvalue [2 x <8 x float>] %123, 0, !dbg !7
  %1554 = extractelement <8 x float> %1553, i64 0, !dbg !7
  %1555 = extractelement <8 x float> %1553, i64 1, !dbg !7
  %1556 = extractelement <8 x float> %1553, i64 2, !dbg !7
  %1557 = extractelement <8 x float> %1553, i64 3, !dbg !7
  %1558 = select i1 %.not, float %1555, float %1554, !dbg !7
  %1559 = bitcast float %1558 to i32, !dbg !7
  %1560 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1326, i32 %1559), !dbg !7
  %1561 = bitcast i32 %1560 to float, !dbg !7
  %1562 = select i1 %.not, float %1554, float %1561, !dbg !7
  %1563 = select i1 %.not, float %1561, float %1555, !dbg !7
  %1564 = select i1 %.not, float %1557, float %1556, !dbg !7
  %1565 = bitcast float %1564 to i32, !dbg !7
  %1566 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1326, i32 %1565), !dbg !7
  %1567 = bitcast i32 %1566 to float, !dbg !7
  %1568 = select i1 %.not, float %1556, float %1567, !dbg !7
  %1569 = select i1 %.not, float %1567, float %1557, !dbg !7
  %1570 = select i1 %.not36, float %1568, float %1562, !dbg !7
  %1571 = bitcast float %1570 to i32, !dbg !7
  %1572 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1342, i32 %1571), !dbg !7
  %1573 = bitcast i32 %1572 to float, !dbg !7
  %1574 = select i1 %.not36, float %1562, float %1573, !dbg !7
  %1575 = select i1 %.not36, float %1573, float %1568, !dbg !7
  %1576 = select i1 %.not36, float %1569, float %1563, !dbg !7
  %1577 = bitcast float %1576 to i32, !dbg !7
  %1578 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1342, i32 %1577), !dbg !7
  %1579 = bitcast i32 %1578 to float, !dbg !7
  %1580 = select i1 %.not36, float %1563, float %1579, !dbg !7
  %1581 = select i1 %.not36, float %1579, float %1569, !dbg !7
  %1582 = extractelement <8 x float> %1553, i64 4, !dbg !7
  %1583 = extractelement <8 x float> %1553, i64 5, !dbg !7
  %1584 = extractelement <8 x float> %1553, i64 6, !dbg !7
  %1585 = extractelement <8 x float> %1553, i64 7, !dbg !7
  %1586 = select i1 %.not, float %1583, float %1582, !dbg !7
  %1587 = bitcast float %1586 to i32, !dbg !7
  %1588 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1326, i32 %1587), !dbg !7
  %1589 = bitcast i32 %1588 to float, !dbg !7
  %1590 = select i1 %.not, float %1582, float %1589, !dbg !7
  %1591 = select i1 %.not, float %1589, float %1583, !dbg !7
  %1592 = select i1 %.not, float %1585, float %1584, !dbg !7
  %1593 = bitcast float %1592 to i32, !dbg !7
  %1594 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1326, i32 %1593), !dbg !7
  %1595 = bitcast i32 %1594 to float, !dbg !7
  %1596 = select i1 %.not, float %1584, float %1595, !dbg !7
  %1597 = select i1 %.not, float %1595, float %1585, !dbg !7
  %1598 = select i1 %.not36, float %1596, float %1590, !dbg !7
  %1599 = bitcast float %1598 to i32, !dbg !7
  %1600 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1342, i32 %1599), !dbg !7
  %1601 = bitcast i32 %1600 to float, !dbg !7
  %1602 = select i1 %.not36, float %1590, float %1601, !dbg !7
  %1603 = select i1 %.not36, float %1601, float %1596, !dbg !7
  %1604 = select i1 %.not36, float %1597, float %1591, !dbg !7
  %1605 = bitcast float %1604 to i32, !dbg !7
  %1606 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1342, i32 %1605), !dbg !7
  %1607 = bitcast i32 %1606 to float, !dbg !7
  %1608 = select i1 %.not36, float %1591, float %1607, !dbg !7
  %1609 = select i1 %.not36, float %1607, float %1597, !dbg !7
  %1610 = extractvalue [2 x <8 x float>] %123, 1, !dbg !7
  %1611 = extractelement <8 x float> %1610, i64 0, !dbg !7
  %1612 = extractelement <8 x float> %1610, i64 1, !dbg !7
  %1613 = extractelement <8 x float> %1610, i64 2, !dbg !7
  %1614 = extractelement <8 x float> %1610, i64 3, !dbg !7
  %1615 = select i1 %.not, float %1612, float %1611, !dbg !7
  %1616 = bitcast float %1615 to i32, !dbg !7
  %1617 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1326, i32 %1616), !dbg !7
  %1618 = bitcast i32 %1617 to float, !dbg !7
  %1619 = select i1 %.not, float %1611, float %1618, !dbg !7
  %1620 = select i1 %.not, float %1618, float %1612, !dbg !7
  %1621 = select i1 %.not, float %1614, float %1613, !dbg !7
  %1622 = bitcast float %1621 to i32, !dbg !7
  %1623 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1326, i32 %1622), !dbg !7
  %1624 = bitcast i32 %1623 to float, !dbg !7
  %1625 = select i1 %.not, float %1613, float %1624, !dbg !7
  %1626 = select i1 %.not, float %1624, float %1614, !dbg !7
  %1627 = select i1 %.not36, float %1625, float %1619, !dbg !7
  %1628 = bitcast float %1627 to i32, !dbg !7
  %1629 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1342, i32 %1628), !dbg !7
  %1630 = bitcast i32 %1629 to float, !dbg !7
  %1631 = select i1 %.not36, float %1619, float %1630, !dbg !7
  %1632 = select i1 %.not36, float %1630, float %1625, !dbg !7
  %1633 = select i1 %.not36, float %1626, float %1620, !dbg !7
  %1634 = bitcast float %1633 to i32, !dbg !7
  %1635 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1342, i32 %1634), !dbg !7
  %1636 = bitcast i32 %1635 to float, !dbg !7
  %1637 = select i1 %.not36, float %1620, float %1636, !dbg !7
  %1638 = select i1 %.not36, float %1636, float %1626, !dbg !7
  %1639 = extractelement <8 x float> %1610, i64 4, !dbg !7
  %1640 = extractelement <8 x float> %1610, i64 5, !dbg !7
  %1641 = extractelement <8 x float> %1610, i64 6, !dbg !7
  %1642 = extractelement <8 x float> %1610, i64 7, !dbg !7
  %1643 = select i1 %.not, float %1640, float %1639, !dbg !7
  %1644 = bitcast float %1643 to i32, !dbg !7
  %1645 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1326, i32 %1644), !dbg !7
  %1646 = bitcast i32 %1645 to float, !dbg !7
  %1647 = select i1 %.not, float %1639, float %1646, !dbg !7
  %1648 = select i1 %.not, float %1646, float %1640, !dbg !7
  %1649 = select i1 %.not, float %1642, float %1641, !dbg !7
  %1650 = bitcast float %1649 to i32, !dbg !7
  %1651 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1326, i32 %1650), !dbg !7
  %1652 = bitcast i32 %1651 to float, !dbg !7
  %1653 = select i1 %.not, float %1641, float %1652, !dbg !7
  %1654 = select i1 %.not, float %1652, float %1642, !dbg !7
  %1655 = select i1 %.not36, float %1653, float %1647, !dbg !7
  %1656 = bitcast float %1655 to i32, !dbg !7
  %1657 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1342, i32 %1656), !dbg !7
  %1658 = bitcast i32 %1657 to float, !dbg !7
  %1659 = select i1 %.not36, float %1647, float %1658, !dbg !7
  %1660 = select i1 %.not36, float %1658, float %1653, !dbg !7
  %1661 = select i1 %.not36, float %1654, float %1648, !dbg !7
  %1662 = bitcast float %1661 to i32, !dbg !7
  %1663 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1342, i32 %1662), !dbg !7
  %1664 = bitcast i32 %1663 to float, !dbg !7
  %1665 = select i1 %.not36, float %1648, float %1664, !dbg !7
  %1666 = select i1 %.not36, float %1664, float %1654, !dbg !7
  %1667 = extractvalue [2 x <8 x float>] %120, 0, !dbg !7
  %1668 = extractelement <8 x float> %1667, i64 0, !dbg !7
  %1669 = extractelement <8 x float> %1667, i64 1, !dbg !7
  %1670 = extractelement <8 x float> %1667, i64 2, !dbg !7
  %1671 = extractelement <8 x float> %1667, i64 3, !dbg !7
  %1672 = select i1 %.not, float %1669, float %1668, !dbg !7
  %1673 = bitcast float %1672 to i32, !dbg !7
  %1674 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1326, i32 %1673), !dbg !7
  %1675 = bitcast i32 %1674 to float, !dbg !7
  %1676 = select i1 %.not, float %1668, float %1675, !dbg !7
  %1677 = select i1 %.not, float %1675, float %1669, !dbg !7
  %1678 = select i1 %.not, float %1671, float %1670, !dbg !7
  %1679 = bitcast float %1678 to i32, !dbg !7
  %1680 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1326, i32 %1679), !dbg !7
  %1681 = bitcast i32 %1680 to float, !dbg !7
  %1682 = select i1 %.not, float %1670, float %1681, !dbg !7
  %1683 = select i1 %.not, float %1681, float %1671, !dbg !7
  %1684 = select i1 %.not36, float %1682, float %1676, !dbg !7
  %1685 = bitcast float %1684 to i32, !dbg !7
  %1686 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1342, i32 %1685), !dbg !7
  %1687 = select i1 %.not36, float %1683, float %1677, !dbg !7
  %1688 = bitcast float %1687 to i32, !dbg !7
  %1689 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1342, i32 %1688), !dbg !7
  %1690 = extractelement <8 x float> %1667, i64 4, !dbg !7
  %1691 = extractelement <8 x float> %1667, i64 5, !dbg !7
  %1692 = extractelement <8 x float> %1667, i64 6, !dbg !7
  %1693 = extractelement <8 x float> %1667, i64 7, !dbg !7
  %1694 = select i1 %.not, float %1691, float %1690, !dbg !7
  %1695 = bitcast float %1694 to i32, !dbg !7
  %1696 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1326, i32 %1695), !dbg !7
  %1697 = bitcast i32 %1696 to float, !dbg !7
  %1698 = select i1 %.not, float %1690, float %1697, !dbg !7
  %1699 = select i1 %.not, float %1697, float %1691, !dbg !7
  %1700 = select i1 %.not, float %1693, float %1692, !dbg !7
  %1701 = bitcast float %1700 to i32, !dbg !7
  %1702 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1326, i32 %1701), !dbg !7
  %1703 = bitcast i32 %1702 to float, !dbg !7
  %1704 = select i1 %.not, float %1692, float %1703, !dbg !7
  %1705 = select i1 %.not, float %1703, float %1693, !dbg !7
  %1706 = select i1 %.not36, float %1704, float %1698, !dbg !7
  %1707 = bitcast float %1706 to i32, !dbg !7
  %1708 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1342, i32 %1707), !dbg !7
  %1709 = select i1 %.not36, float %1705, float %1699, !dbg !7
  %1710 = bitcast float %1709 to i32, !dbg !7
  %1711 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1342, i32 %1710), !dbg !7
  %1712 = extractvalue [2 x <8 x float>] %120, 1, !dbg !7
  %1713 = extractelement <8 x float> %1712, i64 0, !dbg !7
  %1714 = extractelement <8 x float> %1712, i64 1, !dbg !7
  %1715 = extractelement <8 x float> %1712, i64 2, !dbg !7
  %1716 = extractelement <8 x float> %1712, i64 3, !dbg !7
  %1717 = select i1 %.not, float %1714, float %1713, !dbg !7
  %1718 = bitcast float %1717 to i32, !dbg !7
  %1719 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1326, i32 %1718), !dbg !7
  %1720 = bitcast i32 %1719 to float, !dbg !7
  %1721 = select i1 %.not, float %1713, float %1720, !dbg !7
  %1722 = select i1 %.not, float %1720, float %1714, !dbg !7
  %1723 = select i1 %.not, float %1716, float %1715, !dbg !7
  %1724 = bitcast float %1723 to i32, !dbg !7
  %1725 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1326, i32 %1724), !dbg !7
  %1726 = bitcast i32 %1725 to float, !dbg !7
  %1727 = select i1 %.not, float %1715, float %1726, !dbg !7
  %1728 = select i1 %.not, float %1726, float %1716, !dbg !7
  %1729 = select i1 %.not36, float %1727, float %1721, !dbg !7
  %1730 = bitcast float %1729 to i32, !dbg !7
  %1731 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1342, i32 %1730), !dbg !7
  %1732 = select i1 %.not36, float %1728, float %1722, !dbg !7
  %1733 = bitcast float %1732 to i32, !dbg !7
  %1734 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1342, i32 %1733), !dbg !7
  %1735 = extractelement <8 x float> %1712, i64 4, !dbg !7
  %1736 = extractelement <8 x float> %1712, i64 5, !dbg !7
  %1737 = extractelement <8 x float> %1712, i64 6, !dbg !7
  %1738 = extractelement <8 x float> %1712, i64 7, !dbg !7
  %1739 = select i1 %.not, float %1736, float %1735, !dbg !7
  %1740 = bitcast float %1739 to i32, !dbg !7
  %1741 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1326, i32 %1740), !dbg !7
  %1742 = bitcast i32 %1741 to float, !dbg !7
  %1743 = select i1 %.not, float %1735, float %1742, !dbg !7
  %1744 = select i1 %.not, float %1742, float %1736, !dbg !7
  %1745 = select i1 %.not, float %1738, float %1737, !dbg !7
  %1746 = bitcast float %1745 to i32, !dbg !7
  %1747 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1326, i32 %1746), !dbg !7
  %1748 = bitcast i32 %1747 to float, !dbg !7
  %1749 = select i1 %.not, float %1737, float %1748, !dbg !7
  %1750 = select i1 %.not, float %1748, float %1738, !dbg !7
  %1751 = select i1 %.not36, float %1749, float %1743, !dbg !7
  %1752 = bitcast float %1751 to i32, !dbg !7
  %1753 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1342, i32 %1752), !dbg !7
  %1754 = select i1 %.not36, float %1750, float %1744, !dbg !7
  %1755 = bitcast float %1754 to i32, !dbg !7
  %1756 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1342, i32 %1755), !dbg !7
  %1757 = shufflevector <1 x float> %1312, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %1758 = shufflevector <1 x float> %1307, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %1759 = bitcast i32 %1753 to float, !dbg !7
  %1760 = select i1 %.not36, float %1743, float %1759, !dbg !7
  %1761 = insertelement <4 x float> poison, float %1760, i64 0, !dbg !19
  %1762 = bitcast i32 %1756 to float, !dbg !7
  %1763 = select i1 %.not36, float %1744, float %1762, !dbg !7
  %1764 = insertelement <4 x float> %1761, float %1763, i64 1, !dbg !19
  %1765 = select i1 %.not36, float %1759, float %1749, !dbg !7
  %1766 = insertelement <4 x float> %1764, float %1765, i64 2, !dbg !19
  %1767 = select i1 %.not36, float %1762, float %1750, !dbg !7
  %1768 = insertelement <4 x float> %1766, float %1767, i64 3, !dbg !19
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1758, !dbg !19
  %normalized = fmul <4 x float> %1768, %row.reciprocal, !dbg !19
  %1769 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !20
  %1770 = extractelement <4 x half> %1769, i64 3, !dbg !20
  %1771 = extractelement <4 x half> %1769, i64 2, !dbg !20
  %1772 = extractelement <4 x half> %1769, i64 1, !dbg !20
  %1773 = extractelement <4 x half> %1769, i64 0, !dbg !20
  %1774 = bitcast i32 %1731 to float, !dbg !7
  %1775 = select i1 %.not36, float %1721, float %1774, !dbg !7
  %1776 = insertelement <4 x float> poison, float %1775, i64 0, !dbg !19
  %1777 = bitcast i32 %1734 to float, !dbg !7
  %1778 = select i1 %.not36, float %1722, float %1777, !dbg !7
  %1779 = insertelement <4 x float> %1776, float %1778, i64 1, !dbg !19
  %1780 = select i1 %.not36, float %1774, float %1727, !dbg !7
  %1781 = insertelement <4 x float> %1779, float %1780, i64 2, !dbg !19
  %1782 = select i1 %.not36, float %1777, float %1728, !dbg !7
  %1783 = insertelement <4 x float> %1781, float %1782, i64 3, !dbg !19
  %normalized104 = fmul <4 x float> %1783, %row.reciprocal, !dbg !19
  %1784 = fptrunc <4 x float> %normalized104 to <4 x half>, !dbg !20
  %1785 = extractelement <4 x half> %1784, i64 3, !dbg !20
  %1786 = extractelement <4 x half> %1784, i64 2, !dbg !20
  %1787 = extractelement <4 x half> %1784, i64 1, !dbg !20
  %1788 = extractelement <4 x half> %1784, i64 0, !dbg !20
  %1789 = bitcast i32 %1708 to float, !dbg !7
  %1790 = select i1 %.not36, float %1698, float %1789, !dbg !7
  %1791 = insertelement <4 x float> poison, float %1790, i64 0, !dbg !19
  %1792 = bitcast i32 %1711 to float, !dbg !7
  %1793 = select i1 %.not36, float %1699, float %1792, !dbg !7
  %1794 = insertelement <4 x float> %1791, float %1793, i64 1, !dbg !19
  %1795 = select i1 %.not36, float %1789, float %1704, !dbg !7
  %1796 = insertelement <4 x float> %1794, float %1795, i64 2, !dbg !19
  %1797 = select i1 %.not36, float %1792, float %1705, !dbg !7
  %1798 = insertelement <4 x float> %1796, float %1797, i64 3, !dbg !19
  %row.reciprocal105 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1757, !dbg !19
  %normalized106 = fmul <4 x float> %1798, %row.reciprocal105, !dbg !19
  %1799 = fptrunc <4 x float> %normalized106 to <4 x half>, !dbg !20
  %1800 = extractelement <4 x half> %1799, i64 3, !dbg !20
  %1801 = extractelement <4 x half> %1799, i64 2, !dbg !20
  %1802 = extractelement <4 x half> %1799, i64 1, !dbg !20
  %1803 = extractelement <4 x half> %1799, i64 0, !dbg !20
  %1804 = bitcast i32 %1686 to float, !dbg !7
  %1805 = select i1 %.not36, float %1676, float %1804, !dbg !7
  %1806 = insertelement <4 x float> poison, float %1805, i64 0, !dbg !19
  %1807 = bitcast i32 %1689 to float, !dbg !7
  %1808 = select i1 %.not36, float %1677, float %1807, !dbg !7
  %1809 = insertelement <4 x float> %1806, float %1808, i64 1, !dbg !19
  %1810 = select i1 %.not36, float %1804, float %1682, !dbg !7
  %1811 = insertelement <4 x float> %1809, float %1810, i64 2, !dbg !19
  %1812 = select i1 %.not36, float %1807, float %1683, !dbg !7
  %1813 = insertelement <4 x float> %1811, float %1812, i64 3, !dbg !19
  %normalized107 = fmul <4 x float> %1813, %row.reciprocal105, !dbg !19
  %1814 = fptrunc <4 x float> %normalized107 to <4 x half>, !dbg !20
  %1815 = extractelement <4 x half> %1814, i64 3, !dbg !20
  %1816 = extractelement <4 x half> %1814, i64 2, !dbg !20
  %1817 = extractelement <4 x half> %1814, i64 1, !dbg !20
  %1818 = extractelement <4 x half> %1814, i64 0, !dbg !20
  %address.thread1041 = call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #9, !dbg !21, !range !37, !frisk.register_lifetimes !1
  %address.local1042 = zext nneg i32 %address.thread1041 to i64, !dbg !8
  %address.local1043 = add i64 %30, %address.local1042, !dbg !21
  %address.local1044 = lshr i64 %address.local1042, 2, !dbg !18
  %address.local1045 = and i64 %address.local1044, 48, !dbg !18
  %address.local1046 = add i64 %address.local1043, %address.local1045, !dbg !21
  %address.local1047 = and i64 %address.local1042, 240, !dbg !21
  %address.local1049 = sub i64 %address.local1046, %address.local1047, !dbg !21
  %address.local1050 = add i64 %address.local1049, %31, !dbg !21
  %address.local1051 = shl nuw nsw i64 %address.local1050, 7, !dbg !21
  %1819 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %45, !dbg !21
  %address.local1053 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %1819, i64 %address.local1051, !dbg !21
  %address.local1055 = and i64 %address.local1044, 12, !dbg !12
  %address.local1056 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1055, !dbg !21
  store half %1818, ptr addrspace(1) %address.local1056, align 2, !dbg !21
  %address.local1058 = or disjoint i64 %address.local1055, 1, !dbg !21
  %address.local1059 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1058, !dbg !21
  store half %1817, ptr addrspace(1) %address.local1059, align 2, !dbg !21
  %address.local1061 = or disjoint i64 %address.local1055, 2, !dbg !21
  %address.local1062 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1061, !dbg !21
  store half %1816, ptr addrspace(1) %address.local1062, align 2, !dbg !21
  %address.local1064 = or disjoint i64 %address.local1055, 3, !dbg !21
  %address.local1065 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1064, !dbg !21
  store half %1815, ptr addrspace(1) %address.local1065, align 2, !dbg !21
  %address.local1067 = or disjoint i64 %address.local1055, 16, !dbg !21
  %address.local1068 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1067, !dbg !21
  store half %1803, ptr addrspace(1) %address.local1068, align 2, !dbg !21
  %address.local1070 = or disjoint i64 %address.local1055, 17, !dbg !21
  %address.local1071 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1070, !dbg !21
  store half %1802, ptr addrspace(1) %address.local1071, align 2, !dbg !21
  %address.local1073 = or disjoint i64 %address.local1055, 18, !dbg !21
  %address.local1074 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1073, !dbg !21
  store half %1801, ptr addrspace(1) %address.local1074, align 2, !dbg !21
  %address.local1076 = or disjoint i64 %address.local1055, 19, !dbg !21
  %address.local1077 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1076, !dbg !21
  store half %1800, ptr addrspace(1) %address.local1077, align 2, !dbg !21
  %address.local1081 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local1053, i64 16384, !dbg !21
  %address.local1082 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1055, !dbg !21
  store half %1788, ptr addrspace(1) %address.local1082, align 2, !dbg !21
  %address.local1084 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1058, !dbg !21
  store half %1787, ptr addrspace(1) %address.local1084, align 2, !dbg !21
  %address.local1086 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1061, !dbg !21
  store half %1786, ptr addrspace(1) %address.local1086, align 2, !dbg !21
  %address.local1088 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1064, !dbg !21
  store half %1785, ptr addrspace(1) %address.local1088, align 2, !dbg !21
  %address.local1090 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1067, !dbg !21
  store half %1773, ptr addrspace(1) %address.local1090, align 2, !dbg !21
  %address.local1092 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1070, !dbg !21
  store half %1772, ptr addrspace(1) %address.local1092, align 2, !dbg !21
  %address.local1094 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1073, !dbg !21
  store half %1771, ptr addrspace(1) %address.local1094, align 2, !dbg !21
  %address.local1096 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1076, !dbg !21
  store half %1770, ptr addrspace(1) %address.local1096, align 2, !dbg !21
  %1820 = insertelement <4 x float> poison, float %1659, i64 0, !dbg !19
  %1821 = insertelement <4 x float> %1820, float %1665, i64 1, !dbg !19
  %1822 = insertelement <4 x float> %1821, float %1660, i64 2, !dbg !19
  %1823 = insertelement <4 x float> %1822, float %1666, i64 3, !dbg !19
  %normalized109 = fmul <4 x float> %1823, %row.reciprocal, !dbg !19
  %1824 = fptrunc <4 x float> %normalized109 to <4 x half>, !dbg !20
  %1825 = extractelement <4 x half> %1824, i64 3, !dbg !20
  %1826 = extractelement <4 x half> %1824, i64 2, !dbg !20
  %1827 = extractelement <4 x half> %1824, i64 1, !dbg !20
  %1828 = extractelement <4 x half> %1824, i64 0, !dbg !20
  %1829 = insertelement <4 x float> poison, float %1631, i64 0, !dbg !19
  %1830 = insertelement <4 x float> %1829, float %1637, i64 1, !dbg !19
  %1831 = insertelement <4 x float> %1830, float %1632, i64 2, !dbg !19
  %1832 = insertelement <4 x float> %1831, float %1638, i64 3, !dbg !19
  %normalized110 = fmul <4 x float> %1832, %row.reciprocal, !dbg !19
  %1833 = fptrunc <4 x float> %normalized110 to <4 x half>, !dbg !20
  %1834 = extractelement <4 x half> %1833, i64 3, !dbg !20
  %1835 = extractelement <4 x half> %1833, i64 2, !dbg !20
  %1836 = extractelement <4 x half> %1833, i64 1, !dbg !20
  %1837 = extractelement <4 x half> %1833, i64 0, !dbg !20
  %1838 = insertelement <4 x float> poison, float %1602, i64 0, !dbg !19
  %1839 = insertelement <4 x float> %1838, float %1608, i64 1, !dbg !19
  %1840 = insertelement <4 x float> %1839, float %1603, i64 2, !dbg !19
  %1841 = insertelement <4 x float> %1840, float %1609, i64 3, !dbg !19
  %normalized112 = fmul <4 x float> %1841, %row.reciprocal105, !dbg !19
  %1842 = fptrunc <4 x float> %normalized112 to <4 x half>, !dbg !20
  %1843 = extractelement <4 x half> %1842, i64 3, !dbg !20
  %1844 = extractelement <4 x half> %1842, i64 2, !dbg !20
  %1845 = extractelement <4 x half> %1842, i64 1, !dbg !20
  %1846 = extractelement <4 x half> %1842, i64 0, !dbg !20
  %1847 = insertelement <4 x float> poison, float %1574, i64 0, !dbg !19
  %1848 = insertelement <4 x float> %1847, float %1580, i64 1, !dbg !19
  %1849 = insertelement <4 x float> %1848, float %1575, i64 2, !dbg !19
  %1850 = insertelement <4 x float> %1849, float %1581, i64 3, !dbg !19
  %normalized113 = fmul <4 x float> %1850, %row.reciprocal105, !dbg !19
  %1851 = fptrunc <4 x float> %normalized113 to <4 x half>, !dbg !20
  %1852 = extractelement <4 x half> %1851, i64 3, !dbg !20
  %1853 = extractelement <4 x half> %1851, i64 2, !dbg !20
  %1854 = extractelement <4 x half> %1851, i64 1, !dbg !20
  %1855 = extractelement <4 x half> %1851, i64 0, !dbg !20
  %address.local1112 = or disjoint i64 %address.local1055, 32, !dbg !21
  %address.local1113 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1112, !dbg !21
  store half %1855, ptr addrspace(1) %address.local1113, align 2, !dbg !21
  %address.local1115 = or disjoint i64 %address.local1055, 33, !dbg !21
  %address.local1116 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1115, !dbg !21
  store half %1854, ptr addrspace(1) %address.local1116, align 2, !dbg !21
  %address.local1118 = or disjoint i64 %address.local1055, 34, !dbg !21
  %address.local1119 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1118, !dbg !21
  store half %1853, ptr addrspace(1) %address.local1119, align 2, !dbg !21
  %address.local1121 = or disjoint i64 %address.local1055, 35, !dbg !21
  %address.local1122 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1121, !dbg !21
  store half %1852, ptr addrspace(1) %address.local1122, align 2, !dbg !21
  %address.local1124 = or disjoint i64 %address.local1055, 48, !dbg !21
  %address.local1125 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1124, !dbg !21
  store half %1846, ptr addrspace(1) %address.local1125, align 2, !dbg !21
  %address.local1127 = or disjoint i64 %address.local1055, 49, !dbg !21
  %address.local1128 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1127, !dbg !21
  store half %1845, ptr addrspace(1) %address.local1128, align 2, !dbg !21
  %address.local1130 = or disjoint i64 %address.local1055, 50, !dbg !21
  %address.local1131 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1130, !dbg !21
  store half %1844, ptr addrspace(1) %address.local1131, align 2, !dbg !21
  %address.local1133 = or i64 %address.local1044, 51, !dbg !21
  %address.local1134 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1133, !dbg !21
  store half %1843, ptr addrspace(1) %address.local1134, align 2, !dbg !21
  %address.local1139 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1112, !dbg !21
  store half %1837, ptr addrspace(1) %address.local1139, align 2, !dbg !21
  %address.local1141 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1115, !dbg !21
  store half %1836, ptr addrspace(1) %address.local1141, align 2, !dbg !21
  %address.local1143 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1118, !dbg !21
  store half %1835, ptr addrspace(1) %address.local1143, align 2, !dbg !21
  %address.local1145 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1121, !dbg !21
  store half %1834, ptr addrspace(1) %address.local1145, align 2, !dbg !21
  %address.local1147 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1124, !dbg !21
  store half %1828, ptr addrspace(1) %address.local1147, align 2, !dbg !21
  %address.local1149 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1127, !dbg !21
  store half %1827, ptr addrspace(1) %address.local1149, align 2, !dbg !21
  %address.local1151 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1130, !dbg !21
  store half %1826, ptr addrspace(1) %address.local1151, align 2, !dbg !21
  %address.local1153 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1133, !dbg !21
  store half %1825, ptr addrspace(1) %address.local1153, align 2, !dbg !21
  %1856 = insertelement <4 x float> poison, float %1545, i64 0, !dbg !19
  %1857 = insertelement <4 x float> %1856, float %1551, i64 1, !dbg !19
  %1858 = insertelement <4 x float> %1857, float %1546, i64 2, !dbg !19
  %1859 = insertelement <4 x float> %1858, float %1552, i64 3, !dbg !19
  %normalized115 = fmul <4 x float> %1859, %row.reciprocal, !dbg !19
  %1860 = fptrunc <4 x float> %normalized115 to <4 x half>, !dbg !20
  %1861 = extractelement <4 x half> %1860, i64 3, !dbg !20
  %1862 = extractelement <4 x half> %1860, i64 2, !dbg !20
  %1863 = extractelement <4 x half> %1860, i64 1, !dbg !20
  %1864 = extractelement <4 x half> %1860, i64 0, !dbg !20
  %1865 = insertelement <4 x float> poison, float %1517, i64 0, !dbg !19
  %1866 = insertelement <4 x float> %1865, float %1523, i64 1, !dbg !19
  %1867 = insertelement <4 x float> %1866, float %1518, i64 2, !dbg !19
  %1868 = insertelement <4 x float> %1867, float %1524, i64 3, !dbg !19
  %normalized116 = fmul <4 x float> %1868, %row.reciprocal, !dbg !19
  %1869 = fptrunc <4 x float> %normalized116 to <4 x half>, !dbg !20
  %1870 = extractelement <4 x half> %1869, i64 3, !dbg !20
  %1871 = extractelement <4 x half> %1869, i64 2, !dbg !20
  %1872 = extractelement <4 x half> %1869, i64 1, !dbg !20
  %1873 = extractelement <4 x half> %1869, i64 0, !dbg !20
  %1874 = insertelement <4 x float> poison, float %1488, i64 0, !dbg !19
  %1875 = insertelement <4 x float> %1874, float %1494, i64 1, !dbg !19
  %1876 = insertelement <4 x float> %1875, float %1489, i64 2, !dbg !19
  %1877 = insertelement <4 x float> %1876, float %1495, i64 3, !dbg !19
  %normalized118 = fmul <4 x float> %1877, %row.reciprocal105, !dbg !19
  %1878 = fptrunc <4 x float> %normalized118 to <4 x half>, !dbg !20
  %1879 = extractelement <4 x half> %1878, i64 3, !dbg !20
  %1880 = extractelement <4 x half> %1878, i64 2, !dbg !20
  %1881 = extractelement <4 x half> %1878, i64 1, !dbg !20
  %1882 = extractelement <4 x half> %1878, i64 0, !dbg !20
  %1883 = insertelement <4 x float> poison, float %1460, i64 0, !dbg !19
  %1884 = insertelement <4 x float> %1883, float %1466, i64 1, !dbg !19
  %1885 = insertelement <4 x float> %1884, float %1461, i64 2, !dbg !19
  %1886 = insertelement <4 x float> %1885, float %1467, i64 3, !dbg !19
  %normalized119 = fmul <4 x float> %1886, %row.reciprocal105, !dbg !19
  %1887 = fptrunc <4 x float> %normalized119 to <4 x half>, !dbg !20
  %1888 = extractelement <4 x half> %1887, i64 3, !dbg !20
  %1889 = extractelement <4 x half> %1887, i64 2, !dbg !20
  %1890 = extractelement <4 x half> %1887, i64 1, !dbg !20
  %1891 = extractelement <4 x half> %1887, i64 0, !dbg !20
  %address.local1169 = or disjoint i64 %address.local1055, 64, !dbg !21
  %address.local1170 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1169, !dbg !21
  store half %1891, ptr addrspace(1) %address.local1170, align 2, !dbg !21
  %address.local1172 = or disjoint i64 %address.local1055, 65, !dbg !21
  %address.local1173 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1172, !dbg !21
  store half %1890, ptr addrspace(1) %address.local1173, align 2, !dbg !21
  %address.local1175 = or disjoint i64 %address.local1055, 66, !dbg !21
  %address.local1176 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1175, !dbg !21
  store half %1889, ptr addrspace(1) %address.local1176, align 2, !dbg !21
  %address.local1178 = or disjoint i64 %address.local1055, 67, !dbg !21
  %address.local1179 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1178, !dbg !21
  store half %1888, ptr addrspace(1) %address.local1179, align 2, !dbg !21
  %address.local1181 = or disjoint i64 %address.local1055, 80, !dbg !21
  %address.local1182 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1181, !dbg !21
  store half %1882, ptr addrspace(1) %address.local1182, align 2, !dbg !21
  %address.local1184 = or disjoint i64 %address.local1055, 81, !dbg !21
  %address.local1185 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1184, !dbg !21
  store half %1881, ptr addrspace(1) %address.local1185, align 2, !dbg !21
  %address.local1187 = or disjoint i64 %address.local1055, 82, !dbg !21
  %address.local1188 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1187, !dbg !21
  store half %1880, ptr addrspace(1) %address.local1188, align 2, !dbg !21
  %address.local1190 = or disjoint i64 %address.local1055, 83, !dbg !21
  %address.local1191 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1190, !dbg !21
  store half %1879, ptr addrspace(1) %address.local1191, align 2, !dbg !21
  %address.local1196 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1169, !dbg !21
  store half %1873, ptr addrspace(1) %address.local1196, align 2, !dbg !21
  %address.local1198 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1172, !dbg !21
  store half %1872, ptr addrspace(1) %address.local1198, align 2, !dbg !21
  %address.local1200 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1175, !dbg !21
  store half %1871, ptr addrspace(1) %address.local1200, align 2, !dbg !21
  %address.local1202 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1178, !dbg !21
  store half %1870, ptr addrspace(1) %address.local1202, align 2, !dbg !21
  %address.local1204 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1181, !dbg !21
  store half %1864, ptr addrspace(1) %address.local1204, align 2, !dbg !21
  %address.local1206 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1184, !dbg !21
  store half %1863, ptr addrspace(1) %address.local1206, align 2, !dbg !21
  %address.local1208 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1187, !dbg !21
  store half %1862, ptr addrspace(1) %address.local1208, align 2, !dbg !21
  %address.local1210 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1190, !dbg !21
  store half %1861, ptr addrspace(1) %address.local1210, align 2, !dbg !21
  %1892 = insertelement <4 x float> poison, float %1431, i64 0, !dbg !19
  %1893 = insertelement <4 x float> %1892, float %1437, i64 1, !dbg !19
  %1894 = insertelement <4 x float> %1893, float %1432, i64 2, !dbg !19
  %1895 = insertelement <4 x float> %1894, float %1438, i64 3, !dbg !19
  %normalized121 = fmul <4 x float> %1895, %row.reciprocal, !dbg !19
  %1896 = fptrunc <4 x float> %normalized121 to <4 x half>, !dbg !20
  %1897 = extractelement <4 x half> %1896, i64 3, !dbg !20
  %1898 = extractelement <4 x half> %1896, i64 2, !dbg !20
  %1899 = extractelement <4 x half> %1896, i64 1, !dbg !20
  %1900 = extractelement <4 x half> %1896, i64 0, !dbg !20
  %1901 = insertelement <4 x float> poison, float %1403, i64 0, !dbg !19
  %1902 = insertelement <4 x float> %1901, float %1409, i64 1, !dbg !19
  %1903 = insertelement <4 x float> %1902, float %1404, i64 2, !dbg !19
  %1904 = insertelement <4 x float> %1903, float %1410, i64 3, !dbg !19
  %normalized122 = fmul <4 x float> %1904, %row.reciprocal, !dbg !19
  %1905 = fptrunc <4 x float> %normalized122 to <4 x half>, !dbg !20
  %1906 = extractelement <4 x half> %1905, i64 3, !dbg !20
  %1907 = extractelement <4 x half> %1905, i64 2, !dbg !20
  %1908 = extractelement <4 x half> %1905, i64 1, !dbg !20
  %1909 = extractelement <4 x half> %1905, i64 0, !dbg !20
  %1910 = insertelement <4 x float> poison, float %1374, i64 0, !dbg !19
  %1911 = insertelement <4 x float> %1910, float %1380, i64 1, !dbg !19
  %1912 = insertelement <4 x float> %1911, float %1375, i64 2, !dbg !19
  %1913 = insertelement <4 x float> %1912, float %1381, i64 3, !dbg !19
  %normalized124 = fmul <4 x float> %1913, %row.reciprocal105, !dbg !19
  %1914 = fptrunc <4 x float> %normalized124 to <4 x half>, !dbg !20
  %1915 = extractelement <4 x half> %1914, i64 3, !dbg !20
  %1916 = extractelement <4 x half> %1914, i64 2, !dbg !20
  %1917 = extractelement <4 x half> %1914, i64 1, !dbg !20
  %1918 = extractelement <4 x half> %1914, i64 0, !dbg !20
  %1919 = insertelement <4 x float> poison, float %1346, i64 0, !dbg !19
  %1920 = insertelement <4 x float> %1919, float %1352, i64 1, !dbg !19
  %1921 = insertelement <4 x float> %1920, float %1347, i64 2, !dbg !19
  %1922 = insertelement <4 x float> %1921, float %1353, i64 3, !dbg !19
  %normalized125 = fmul <4 x float> %1922, %row.reciprocal105, !dbg !19
  %1923 = fptrunc <4 x float> %normalized125 to <4 x half>, !dbg !20
  %1924 = extractelement <4 x half> %1923, i64 3, !dbg !20
  %1925 = extractelement <4 x half> %1923, i64 2, !dbg !20
  %1926 = extractelement <4 x half> %1923, i64 1, !dbg !20
  %1927 = extractelement <4 x half> %1923, i64 0, !dbg !20
  %address.local1226 = or disjoint i64 %address.local1055, 96, !dbg !21
  %address.local1227 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1226, !dbg !21
  store half %1927, ptr addrspace(1) %address.local1227, align 2, !dbg !21
  %address.local1229 = or disjoint i64 %address.local1055, 97, !dbg !21
  %address.local1230 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1229, !dbg !21
  store half %1926, ptr addrspace(1) %address.local1230, align 2, !dbg !21
  %address.local1232 = or disjoint i64 %address.local1055, 98, !dbg !21
  %address.local1233 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1232, !dbg !21
  store half %1925, ptr addrspace(1) %address.local1233, align 2, !dbg !21
  %address.local1235 = or disjoint i64 %address.local1055, 99, !dbg !21
  %address.local1236 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1235, !dbg !21
  store half %1924, ptr addrspace(1) %address.local1236, align 2, !dbg !21
  %address.local1238 = or disjoint i64 %address.local1055, 112, !dbg !21
  %address.local1239 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1238, !dbg !21
  store half %1918, ptr addrspace(1) %address.local1239, align 2, !dbg !21
  %address.local1241 = or disjoint i64 %address.local1055, 113, !dbg !21
  %address.local1242 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1241, !dbg !21
  store half %1917, ptr addrspace(1) %address.local1242, align 2, !dbg !21
  %address.local1244 = or disjoint i64 %address.local1055, 114, !dbg !21
  %address.local1245 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1244, !dbg !21
  store half %1916, ptr addrspace(1) %address.local1245, align 2, !dbg !21
  %address.local1247 = or i64 %address.local1044, 115, !dbg !21
  %address.local1248 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1247, !dbg !21
  store half %1915, ptr addrspace(1) %address.local1248, align 2, !dbg !21
  %address.local1253 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1226, !dbg !21
  store half %1909, ptr addrspace(1) %address.local1253, align 2, !dbg !21
  %address.local1255 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1229, !dbg !21
  store half %1908, ptr addrspace(1) %address.local1255, align 2, !dbg !21
  %address.local1257 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1232, !dbg !21
  store half %1907, ptr addrspace(1) %address.local1257, align 2, !dbg !21
  %address.local1259 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1235, !dbg !21
  store half %1906, ptr addrspace(1) %address.local1259, align 2, !dbg !21
  %address.local1261 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1238, !dbg !21
  store half %1900, ptr addrspace(1) %address.local1261, align 2, !dbg !21
  %address.local1263 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1241, !dbg !21
  store half %1899, ptr addrspace(1) %address.local1263, align 2, !dbg !21
  %address.local1265 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1244, !dbg !21
  store half %1898, ptr addrspace(1) %address.local1265, align 2, !dbg !21
  %address.local1267 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1081, i64 %address.local1247, !dbg !21
  store half %1897, ptr addrspace(1) %address.local1267, align 2, !dbg !21
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
!llvm.module.flags = !{!36}

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
!23 = !DILocation(line: 81, column: 26, scope: !4)
!24 = !DILocation(line: 88, column: 21, scope: !4)
!25 = !DILocation(line: 89, column: 20, scope: !4)
!26 = !DILocation(line: 90, column: 22, scope: !4)
!27 = !DILocation(line: 91, column: 23, scope: !4)
!28 = !DILocation(line: 92, column: 20, scope: !4)
!29 = !DILocation(line: 93, column: 22, scope: !4)
!30 = !DILocation(line: 94, column: 22, scope: !4)
!31 = !DILocation(line: 95, column: 20, scope: !4)
!32 = !DILocation(line: 96, column: 16, scope: !4)
!33 = !DILocation(line: 103, column: 22, scope: !4)
!35 = !{i32 256, i32 1, i32 1}
!36 = !{i32 2, !"Debug Info Version", i32 3}
!37 = !{i32 0, i32 256}
!38 = !{i32 4}
!39 = !{i64 8192}
!40 = !{i32 0}
!41 = !{i32 1}
!42 = !{i32 2}
!43 = !{i32 3}
!44 = !{i32 5}
!45 = !{i32 6}
!46 = !{i32 7}
!47 = !{i32 8}
!48 = !{i32 9}
!49 = !{i32 10}
!50 = !{i32 11}
!51 = !{i32 12}
!52 = !{i32 13}
!53 = !{i32 14}
!54 = !{i32 15}
!55 = !{i32 16}
!56 = !{i32 17}
!57 = !{i32 18}
!58 = !{i32 19}
!59 = !{i32 20}
!60 = !{i32 21}
!61 = !{i32 22}
