; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/thread_v_double_buffer/causal.input.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/thread_v_double_buffer/causal.input.mlir"

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
  %address.local346 = zext nneg i32 %48 to i64, !dbg !9
  %address.local347 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local346, !dbg !9
  %address.local348 = ptrtoint ptr addrspace(3) %address.local347 to i64, !dbg !9
  %address.local349 = trunc i64 %address.local348 to i32, !dbg !9
  %49 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local349), !dbg !9
  %address.local350 = lshr i32 %address.thread, 4, !dbg !9
  %address.local351 = and i32 %address.local350, 8, !dbg !9
  %address.local352 = shl nuw nsw i32 %address.thread, 7, !dbg !9
  %address.local353 = and i32 %address.local352, 16256, !dbg !9
  %address.local354 = or disjoint i32 %address.local351, %address.local353, !dbg !9
  %50 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !9
  %.idx1356 = shl i64 %44, 8, !dbg !9
  %51 = add i64 %.idx, %.idx1356, !dbg !9
  %52 = shl nuw nsw i32 %address.local354, 1, !dbg !9
  %53 = trunc i64 %51 to i32, !dbg !9
  %dma.buffer.offset1354 = or disjoint i32 %52, %53, !dbg !9
  %54 = insertelement <2 x i64> poison, i64 %50, i64 0, !dbg !9
  %55 = bitcast <2 x i64> %54 to <4 x i32>, !dbg !9
  %dma.buffer.rsrc1355 = shufflevector <4 x i32> %55, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !9
  %56 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %49, i32 %dma.buffer.offset1354, <4 x i32> %dma.buffer.rsrc1355) #10, !dbg !9
  %address.local359 = or disjoint i64 %address.local346, 2048, !dbg !9
  %address.local360 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local359, !dbg !9
  %address.local361 = ptrtoint ptr addrspace(3) %address.local360 to i64, !dbg !9
  %address.local362 = trunc i64 %address.local361 to i32, !dbg !9
  %57 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local362), !dbg !9
  %address.local368 = or disjoint i32 %address.local354, 16, !dbg !9
  %58 = shl nuw nsw i32 %address.local368, 1, !dbg !9
  %dma.buffer.offset1352 = or disjoint i32 %58, %53, !dbg !9
  %59 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %57, i32 %dma.buffer.offset1352, <4 x i32> %dma.buffer.rsrc1355) #10, !dbg !9
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %60 = shl nuw nsw i64 %41, 19, !dbg !9
  %61 = add i64 %60, %45, !dbg !9
  %address.local371 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local346, !dbg !9
  %address.local372 = ptrtoint ptr addrspace(3) %address.local371 to i64, !dbg !9
  %address.local373 = trunc i64 %address.local372 to i32, !dbg !9
  %62 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local373), !dbg !9
  %63 = trunc i64 %61 to i32, !dbg !9
  %.tr = or disjoint i32 %address.local354, %63, !dbg !9
  %64 = shl i32 %.tr, 1, !dbg !9
  %dma.buffer.offset1350 = or disjoint i32 %64, 64, !dbg !9
  %65 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %62, i32 %dma.buffer.offset1350, <4 x i32> %dma.buffer.rsrc1355) #10, !dbg !9
  %address.local375 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local359, !dbg !9
  %address.local376 = ptrtoint ptr addrspace(3) %address.local375 to i64, !dbg !9
  %address.local377 = trunc i64 %address.local376 to i32, !dbg !9
  %66 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local377), !dbg !9
  %.tr1360 = or disjoint i32 %address.local368, %63, !dbg !9
  %67 = shl i32 %.tr1360, 1, !dbg !9
  %dma.buffer.offset1348 = or disjoint i32 %67, 64, !dbg !9
  %68 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %66, i32 %dma.buffer.offset1348, <4 x i32> %dma.buffer.rsrc1355) #10, !dbg !9
  %69 = and i64 %46, 240, !dbg !19
  %address.local379 = lshr i32 %address.thread, 2, !dbg !13
  %address.local380 = and i32 %address.local379, 4, !dbg !13
  %address.local381 = shl nuw nsw i32 %address.thread, 1, !dbg !13
  %address.local382 = and i32 %address.local381, 384, !dbg !13
  %address.local383 = or disjoint i32 %address.local380, %address.local382, !dbg !13
  %address.local385 = and i32 %47, 120, !dbg !13
  %address.local386 = or disjoint i32 %address.local383, %address.local385, !dbg !13
  %address.local387 = shl nuw nsw i32 %address.thread, 5, !dbg !13
  %address.local388 = and i32 %address.local387, 1024, !dbg !13
  %address.local389 = or disjoint i32 %address.local386, %address.local388, !dbg !13
  %address.local390 = zext nneg i32 %address.local389 to i64, !dbg !13
  %address.local391 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local390, !dbg !13
  %70 = load <4 x half>, ptr addrspace(3) %address.local391, align 8, !dbg !13
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local391, i64 4096, !dbg !13
  %71 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !13
  %72 = fmul <4 x half> %70, splat (half 0xH3015), !dbg !13
  %lds.const.gep1177 = getelementptr i8, ptr addrspace(3) %address.local391, i64 1024, !dbg !13
  %73 = load <4 x half>, ptr addrspace(3) %lds.const.gep1177, align 8, !dbg !13
  %74 = fmul <4 x half> %71, splat (half 0xH3015), !dbg !13
  %lds.const.gep1178 = getelementptr i8, ptr addrspace(3) %address.local391, i64 5120, !dbg !13
  %75 = load <4 x half>, ptr addrspace(3) %lds.const.gep1178, align 8, !dbg !13
  %76 = fmul <4 x half> %73, splat (half 0xH3015), !dbg !13
  %77 = fmul <4 x half> %75, splat (half 0xH3015), !dbg !13
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %dma.buffer.offset1346 = or disjoint i32 %64, 128, !dbg !9
  %78 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %49, i32 %dma.buffer.offset1346, <4 x i32> %dma.buffer.rsrc1355) #10, !dbg !9
  %dma.buffer.offset1344 = or disjoint i32 %67, 128, !dbg !9
  %79 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %57, i32 %dma.buffer.offset1344, <4 x i32> %dma.buffer.rsrc1355) #10, !dbg !9
  %lds.const.gep1179 = getelementptr i8, ptr addrspace(3) %address.local391, i64 8192, !dbg !13
  %80 = load <4 x half>, ptr addrspace(3) %lds.const.gep1179, align 8, !dbg !13
  %lds.const.gep1180 = getelementptr i8, ptr addrspace(3) %address.local391, i64 12288, !dbg !13
  %81 = load <4 x half>, ptr addrspace(3) %lds.const.gep1180, align 8, !dbg !13
  %lds.const.gep1181 = getelementptr i8, ptr addrspace(3) %address.local391, i64 9216, !dbg !13
  %82 = load <4 x half>, ptr addrspace(3) %lds.const.gep1181, align 8, !dbg !13
  %83 = fmul <4 x half> %80, splat (half 0xH3015), !dbg !13
  %lds.const.gep1182 = getelementptr i8, ptr addrspace(3) %address.local391, i64 13312, !dbg !13
  %84 = load <4 x half>, ptr addrspace(3) %lds.const.gep1182, align 8, !dbg !13
  %85 = fmul <4 x half> %81, splat (half 0xH3015), !dbg !13
  %86 = fmul <4 x half> %82, splat (half 0xH3015), !dbg !13
  %87 = fmul <4 x half> %84, splat (half 0xH3015), !dbg !13
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %dma.buffer.offset1342 = or disjoint i32 %64, 192, !dbg !9
  %88 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %62, i32 %dma.buffer.offset1342, <4 x i32> %dma.buffer.rsrc1355) #10, !dbg !9
  %dma.buffer.offset1340 = or disjoint i32 %67, 192, !dbg !9
  %89 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %66, i32 %dma.buffer.offset1340, <4 x i32> %dma.buffer.rsrc1355) #10, !dbg !9
  %90 = load <4 x half>, ptr addrspace(3) %address.local391, align 8, !dbg !13
  %91 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !13
  %92 = load <4 x half>, ptr addrspace(3) %lds.const.gep1177, align 8, !dbg !13
  %93 = fmul <4 x half> %90, splat (half 0xH3015), !dbg !13
  %94 = load <4 x half>, ptr addrspace(3) %lds.const.gep1178, align 8, !dbg !13
  %95 = fmul <4 x half> %91, splat (half 0xH3015), !dbg !13
  %96 = fmul <4 x half> %92, splat (half 0xH3015), !dbg !13
  %97 = fmul <4 x half> %94, splat (half 0xH3015), !dbg !13
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %98 = load <4 x half>, ptr addrspace(3) %lds.const.gep1179, align 8, !dbg !13
  %99 = load <4 x half>, ptr addrspace(3) %lds.const.gep1180, align 8, !dbg !13
  %100 = load <4 x half>, ptr addrspace(3) %lds.const.gep1181, align 8, !dbg !13
  %101 = fmul <4 x half> %98, splat (half 0xH3015), !dbg !13
  %102 = load <4 x half>, ptr addrspace(3) %lds.const.gep1182, align 8, !dbg !13
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
  %index.compact275 = zext nneg i32 %114 to i64, !dbg !17
  %115 = or disjoint i64 %113, %index.compact275, !dbg !17
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
  %128 = phi i64 [ %1401, %136 ], [ 0, %4 ], !dbg !6
  %129 = phi [2 x <8 x float>] [ %1386, %136 ], [ zeroinitializer, %4 ], !dbg !6
  %130 = phi [2 x <1 x float>] [ %1148, %136 ], [ zeroinitializer, %4 ], !dbg !6
  %131 = phi [2 x <1 x float>] [ %1388, %136 ], [ [<1 x float> splat (float 0xFFF0000000000000), <1 x float> splat (float 0xFFF0000000000000)], %4 ], !dbg !6
  %132 = phi [2 x <8 x float>] [ %1392, %136 ], [ zeroinitializer, %4 ], !dbg !6
  %133 = phi [2 x <8 x float>] [ %1396, %136 ], [ zeroinitializer, %4 ], !dbg !6
  %134 = phi [2 x <8 x float>] [ %1400, %136 ], [ zeroinitializer, %4 ], !dbg !6
  %135 = icmp slt i64 %128, %106, !dbg !6
  br i1 %135, label %136, label %1402, !dbg !6

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
  %address.thread442 = call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #9, !dbg !16, !range !33, !frisk.register_lifetimes !1
  %address.local443 = zext nneg i32 %address.thread442 to i64, !dbg !9
  %address.local449 = lshr i64 %address.local443, 2, !dbg !16
  %147 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %.idx13691372 = or disjoint i64 %143, %140, !dbg !16
  %148 = add i64 %.idx13691372, %146, !dbg !16
  %149 = insertelement <2 x i64> poison, i64 %147, i64 0, !dbg !16
  %150 = bitcast <2 x i64> %149 to <4 x i32>, !dbg !16
  %dma.buffer.rsrc1339 = shufflevector <4 x i32> %150, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !16
  %kdma.source = zext nneg i32 %kdma.owner to i64, !dbg !16
  %kdma.source1437 = lshr i64 %kdma.source, 4, !dbg !16
  %kdma.source1438 = shl nuw nsw i64 %kdma.source1437, 9, !dbg !16
  %kdma.source1439 = shl nuw nsw i64 %kdma.source, 7, !dbg !16
  %kdma.source1440 = add nuw nsw i64 %kdma.source1438, %kdma.source1439, !dbg !16
  %kdma.source1441 = lshr i64 %kdma.source, 5, !dbg !16
  %kdma.source1442 = lshr i64 %kdma.source, 2, !dbg !16
  %kdma.source1443 = and i64 %kdma.source1442, 1, !dbg !16
  %kdma.source1444 = and i64 %kdma.source1437, %kdma.source1443, !dbg !16
  %kdma.source1445 = or disjoint i64 %kdma.source1441, %kdma.source1444, !dbg !16
  %kdma.source1446 = shl nuw nsw i64 %kdma.source1445, 10, !dbg !16
  %kdma.source1447 = sub nsw i64 %kdma.source1440, %kdma.source1446, !dbg !16
  %kdma.source1448 = and i64 %kdma.source1447, 16256, !dbg !16
  %kdma.source1449 = lshr i32 %kdma.tid, 4, !dbg !16
  %kdma.source1450 = and i32 %kdma.source1449, 8, !dbg !16
  %kdma.source1451 = zext nneg i32 %kdma.source1450 to i64, !dbg !16
  %kdma.source1452 = or disjoint i64 %kdma.source1448, %kdma.source1451, !dbg !16
  %kdma.source1453 = add i64 %148, %kdma.source1452, !dbg !16
  %kdma.source1454 = trunc i64 %kdma.source1453 to i32, !dbg !16
  %kdma.source1455 = shl i32 %kdma.source1454, 1, !dbg !16
  %kdma.offset = add i32 %kdma.source1455, %kdma.intra, !dbg !16
  %151 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %49, i32 %kdma.offset, <4 x i32> %dma.buffer.rsrc1339) #10, !dbg !16
  %kdma.source1456 = zext nneg i32 %kdma.owner.hi to i64, !dbg !16
  %kdma.source1457 = lshr i64 %kdma.source1456, 4, !dbg !16
  %kdma.source1458 = shl nuw nsw i64 %kdma.source1457, 9, !dbg !16
  %kdma.source1459 = shl nuw nsw i64 %kdma.source1456, 7, !dbg !16
  %kdma.source1460 = add nuw nsw i64 %kdma.source1458, %kdma.source1459, !dbg !16
  %kdma.source1461 = lshr i64 %kdma.source1456, 5, !dbg !16
  %kdma.source1462 = lshr i64 %kdma.source1456, 2, !dbg !16
  %kdma.source1463 = and i64 %kdma.source1462, 1, !dbg !16
  %kdma.source1464 = and i64 %kdma.source1457, %kdma.source1463, !dbg !16
  %kdma.source1465 = add nuw nsw i64 %kdma.source1461, %kdma.source1464, !dbg !16
  %kdma.source1466 = shl nuw nsw i64 %kdma.source1465, 10, !dbg !16
  %kdma.source1467 = sub nsw i64 %kdma.source1460, %kdma.source1466, !dbg !16
  %kdma.source1468 = and i64 %kdma.source1467, 16256, !dbg !16
  %kdma.source1472 = or disjoint i64 %kdma.source1468, %kdma.source1451, !dbg !16
  %kdma.source1473 = add i64 %148, %kdma.source1472, !dbg !16
  %kdma.source1474 = trunc i64 %kdma.source1473 to i32, !dbg !16
  %kdma.source1475 = shl i32 %kdma.source1474, 1, !dbg !16
  %kdma.offset1476 = add i32 %kdma.source1475, %kdma.intra.hi, !dbg !16
  %kdma.lds1477 = add i32 %49, 512, !dbg !16
  %152 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1477, i32 %kdma.offset1476, <4 x i32> %dma.buffer.rsrc1339) #10, !dbg !16
  %kdma.source1479 = or disjoint i64 %kdma.source, 256, !dbg !16
  %kdma.source1480 = lshr i64 %kdma.source1479, 4, !dbg !16
  %kdma.source1481 = shl nuw nsw i64 %kdma.source1480, 9, !dbg !16
  %kdma.source1483 = add nuw nsw i64 %kdma.source1481, %kdma.source1439, !dbg !16
  %kdma.source1484 = lshr i64 %kdma.source1479, 5, !dbg !16
  %kdma.source1487 = and i64 %kdma.source1480, %kdma.source1443, !dbg !16
  %kdma.source1488 = or disjoint i64 %kdma.source1484, %kdma.source1487, !dbg !16
  %kdma.source1489 = shl nuw nsw i64 %kdma.source1488, 10, !dbg !16
  %kdma.source1490 = sub nsw i64 %kdma.source1483, %kdma.source1489, !dbg !16
  %kdma.source1491 = and i64 %kdma.source1490, 16256, !dbg !16
  %kdma.source1494 = or disjoint i32 %kdma.source1450, 16, !dbg !16
  %kdma.source1495 = zext nneg i32 %kdma.source1494 to i64, !dbg !16
  %kdma.source1496 = or disjoint i64 %kdma.source1491, %kdma.source1495, !dbg !16
  %kdma.source1497 = add i64 %148, %kdma.source1496, !dbg !16
  %kdma.source1498 = trunc i64 %kdma.source1497 to i32, !dbg !16
  %kdma.source1499 = shl i32 %kdma.source1498, 1, !dbg !16
  %kdma.offset1500 = add i32 %kdma.source1499, %kdma.intra, !dbg !16
  %153 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %57, i32 %kdma.offset1500, <4 x i32> %dma.buffer.rsrc1339) #10, !dbg !16
  %kdma.source1503 = or disjoint i64 %kdma.source1456, 256, !dbg !16
  %kdma.source1504 = lshr i64 %kdma.source1503, 4, !dbg !16
  %kdma.source1505 = shl nuw nsw i64 %kdma.source1504, 9, !dbg !16
  %kdma.source1507 = add nuw nsw i64 %kdma.source1505, %kdma.source1459, !dbg !16
  %kdma.source1508 = lshr i64 %kdma.source1503, 5, !dbg !16
  %kdma.source1511 = and i64 %kdma.source1504, %kdma.source1463, !dbg !16
  %kdma.source1512 = add nuw nsw i64 %kdma.source1508, %kdma.source1511, !dbg !16
  %kdma.source1513 = shl nuw nsw i64 %kdma.source1512, 10, !dbg !16
  %kdma.source1514 = sub nsw i64 %kdma.source1507, %kdma.source1513, !dbg !16
  %kdma.source1515 = and i64 %kdma.source1514, 16256, !dbg !16
  %kdma.source1520 = or disjoint i64 %kdma.source1515, %kdma.source1495, !dbg !16
  %kdma.source1521 = add i64 %148, %kdma.source1520, !dbg !16
  %kdma.source1522 = trunc i64 %kdma.source1521 to i32, !dbg !16
  %kdma.source1523 = shl i32 %kdma.source1522, 1, !dbg !16
  %kdma.offset1524 = add i32 %kdma.source1523, %kdma.intra.hi, !dbg !16
  %kdma.lds1525 = add i32 %57, 512, !dbg !16
  %154 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1525, i32 %kdma.offset1524, <4 x i32> %dma.buffer.rsrc1339) #10, !dbg !16
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
  %.idx13791383 = add nuw nsw i64 %174, %139, !dbg !16
  %175 = add i64 %.idx13791383, %172, !dbg !16
  %176 = add i64 %175, %146, !dbg !16
  %kdma.source1543 = add i64 %176, %kdma.source1452, !dbg !16
  %kdma.source1544 = trunc i64 %kdma.source1543 to i32, !dbg !16
  %kdma.source1545 = shl i32 %kdma.source1544, 1, !dbg !16
  %kdma.source1546 = add i32 %kdma.source1545, 64, !dbg !16
  %kdma.offset1547 = add i32 %kdma.source1546, %kdma.intra, !dbg !16
  %177 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %62, i32 %kdma.offset1547, <4 x i32> %dma.buffer.rsrc1339) #10, !dbg !16
  %kdma.source1566 = add i64 %176, %kdma.source1472, !dbg !16
  %kdma.source1567 = trunc i64 %kdma.source1566 to i32, !dbg !16
  %kdma.source1568 = shl i32 %kdma.source1567, 1, !dbg !16
  %kdma.source1569 = add i32 %kdma.source1568, 64, !dbg !16
  %kdma.offset1570 = add i32 %kdma.source1569, %kdma.intra.hi, !dbg !16
  %kdma.lds1571 = add i32 %62, 512, !dbg !16
  %178 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1571, i32 %kdma.offset1570, <4 x i32> %dma.buffer.rsrc1339) #10, !dbg !16
  %kdma.source1591 = add i64 %176, %kdma.source1496, !dbg !16
  %kdma.source1592 = trunc i64 %kdma.source1591 to i32, !dbg !16
  %kdma.source1593 = shl i32 %kdma.source1592, 1, !dbg !16
  %kdma.source1594 = add i32 %kdma.source1593, 64, !dbg !16
  %kdma.offset1595 = add i32 %kdma.source1594, %kdma.intra, !dbg !16
  %179 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %66, i32 %kdma.offset1595, <4 x i32> %dma.buffer.rsrc1339) #10, !dbg !16
  %kdma.source1616 = add i64 %176, %kdma.source1520, !dbg !16
  %kdma.source1617 = trunc i64 %kdma.source1616 to i32, !dbg !16
  %kdma.source1618 = shl i32 %kdma.source1617, 1, !dbg !16
  %kdma.source1619 = add i32 %kdma.source1618, 64, !dbg !16
  %kdma.offset1620 = add i32 %kdma.source1619, %kdma.intra.hi, !dbg !16
  %kdma.lds1621 = add i32 %66, 512, !dbg !16
  %180 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1621, i32 %kdma.offset1620, <4 x i32> %dma.buffer.rsrc1339) #10, !dbg !16
  %address.local486 = shl nuw nsw i64 %address.local443, 1, !dbg !17
  %address.local487 = and i64 %address.local486, 24, !dbg !17
  %address.local489 = shl i64 %address.local443, 62, !dbg !17
  %address.local490 = ashr exact i64 %address.local489, 57, !dbg !17
  %address.local491 = and i64 %address.local490, -64, !dbg !17
  %address.local494 = and i64 %address.local449, 12, !dbg !13
  %address.local496 = and i64 %address.local443, 32, !dbg !13
  %address.local497 = icmp eq i64 %address.local496, 0, !dbg !13
  %address.local498 = select i1 %address.local497, i64 0, i64 1016, !dbg !13
  %.idx1391 = mul nuw nsw i64 %address.local443, 320, !dbg !17
  %181 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %.idx1391, !dbg !17
  %.idx1392 = mul nsw i64 %address.local449, -1280, !dbg !17
  %182 = getelementptr i8, ptr addrspace(3) %181, i64 %.idx1392, !dbg !17
  %183 = getelementptr [2 x i8], ptr addrspace(3) %182, i64 %address.local487, !dbg !17
  %184 = getelementptr [2 x i8], ptr addrspace(3) %183, i64 %address.local491, !dbg !17
  %185 = getelementptr [2 x i8], ptr addrspace(3) %184, i64 %address.local494, !dbg !17
  %address.local500 = getelementptr [2 x i8], ptr addrspace(3) %185, i64 %address.local498, !dbg !17
  %186 = add nsw i64 %.idx1391, %.idx1392, !dbg !17
  %.idx1940 = shl nuw nsw i64 %address.local487, 1, !dbg !17
  %187 = or disjoint i64 %186, %.idx1940, !dbg !17
  %188 = or disjoint i64 %address.local491, %address.local494, !dbg !17
  %189 = add nsw i64 %188, %address.local498, !dbg !17
  %190 = shl nsw i64 %189, 1, !dbg !17
  %191 = add nsw i64 %190, %187, !dbg !17
  %192 = lshr i64 %191, 6, !dbg !17
  %193 = and i64 %192, 8, !dbg !17
  %194 = xor i64 %191, %193, !dbg !17
  %195 = sub i64 %194, %191, !dbg !17
  %kdma.swizzled = getelementptr i8, ptr addrspace(3) %address.local500, i64 %195, !dbg !17
  %196 = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8, !dbg !17
  %197 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %196, <4 x float> zeroinitializer) #9, !dbg !17
  %kdma.gep = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4096, !dbg !17
  %198 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8, !dbg !17
  %address.local511 = shl nuw nsw i32 %address.thread442, 1, !dbg !17
  %address.local512 = and i32 %address.local511, 24, !dbg !17
  %address.local513 = shl nuw nsw i32 %address.thread442, 8, !dbg !17
  %address.local514 = and i32 %address.local513, 768, !dbg !17
  %address.local515 = or disjoint i32 %address.local512, %address.local514, !dbg !17
  %address.local516 = shl nuw nsw i32 %address.thread442, 5, !dbg !17
  %address.local517 = and i32 %address.local516, 96, !dbg !17
  %address.local518 = or disjoint i32 %address.local515, %address.local517, !dbg !17
  %address.local519 = zext nneg i32 %address.local518 to i64, !dbg !17
  %address.base = call i64 asm sideeffect "", "=v,0"(i64 %117) #9, !dbg !17, !frisk.register_lifetimes !1
  %address.local520 = lshr i32 %address.thread442, 1, !dbg !17
  %address.local521 = and i32 %address.local520, 1, !dbg !17
  %address.local522 = zext nneg i32 %address.local521 to i64, !dbg !17
  %address.local523 = add nsw i64 %address.base, %address.local522, !dbg !17
  %address.local524 = shl i64 %address.local523, 6, !dbg !17
  %address.local525 = sub i64 %address.local519, %address.local524, !dbg !17
  %199 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local525, !dbg !17
  %200 = getelementptr [2 x i8], ptr addrspace(3) %199, i64 %address.local494, !dbg !17
  %201 = getelementptr [2 x i8], ptr addrspace(3) %200, i64 %address.local498, !dbg !17
  %address.local529 = getelementptr i8, ptr addrspace(3) %201, i64 64, !dbg !17
  %.idx19431946 = add i64 %address.local525, %address.local494, !dbg !17
  %202 = add i64 %.idx19431946, %address.local498, !dbg !17
  %203 = shl i64 %202, 1, !dbg !17
  %204 = add i64 %203, 64, !dbg !17
  %205 = lshr i64 %204, 6, !dbg !17
  %206 = and i64 %205, 8, !dbg !17
  %207 = xor i64 %204, %206, !dbg !17
  %208 = sub i64 %207, %204, !dbg !17
  %kdma.swizzled1814 = getelementptr i8, ptr addrspace(3) %address.local529, i64 %208, !dbg !17
  %209 = load <4 x half>, ptr addrspace(3) %kdma.swizzled1814, align 8, !dbg !17, !frisk.k_lds_pair !35
  %kdma.gep1818 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1814, i64 128, !dbg !17
  %210 = load <4 x half>, ptr addrspace(3) %kdma.gep1818, align 8, !dbg !17, !frisk.k_lds_pair !35
  %kdma.gep1815 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1814, i64 4096, !dbg !17
  %211 = load <4 x half>, ptr addrspace(3) %kdma.gep1815, align 8, !dbg !17, !frisk.k_lds_pair !36
  %kdma.gep1819 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1814, i64 4224, !dbg !17
  %212 = load <4 x half>, ptr addrspace(3) %kdma.gep1819, align 8, !dbg !17, !frisk.k_lds_pair !36
  %kdma.gep1816 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 128, !dbg !17
  %213 = load <4 x half>, ptr addrspace(3) %kdma.gep1816, align 8, !dbg !17, !frisk.k_lds_pair !37
  %kdma.gep1820 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1024, !dbg !17
  %214 = load <4 x half>, ptr addrspace(3) %kdma.gep1820, align 8, !dbg !17, !frisk.k_lds_pair !37
  %kdma.gep1817 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4224, !dbg !17
  %215 = load <4 x half>, ptr addrspace(3) %kdma.gep1817, align 8, !dbg !17, !frisk.k_lds_pair !38
  %kdma.gep1821 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5120, !dbg !17
  %216 = load <4 x half>, ptr addrspace(3) %kdma.gep1821, align 8, !dbg !17, !frisk.k_lds_pair !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_lds_prefetch !1
  %217 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %198, <4 x float> %197) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_lds_prefetch !1
  %kdma.gep1822 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1814, i64 1024, !dbg !17
  %218 = load <4 x half>, ptr addrspace(3) %kdma.gep1822, align 8, !dbg !17, !frisk.k_lds_pair !34
  %kdma.gep1826 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1814, i64 1152, !dbg !17
  %219 = load <4 x half>, ptr addrspace(3) %kdma.gep1826, align 8, !dbg !17, !frisk.k_lds_pair !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_lds_prefetch !1
  %220 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %209, <4 x float> zeroinitializer) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_lds_prefetch !1
  %kdma.gep1823 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1814, i64 5120, !dbg !17
  %221 = load <4 x half>, ptr addrspace(3) %kdma.gep1823, align 8, !dbg !17, !frisk.k_lds_pair !39
  %kdma.gep1827 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1814, i64 5248, !dbg !17
  %222 = load <4 x half>, ptr addrspace(3) %kdma.gep1827, align 8, !dbg !17, !frisk.k_lds_pair !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_lds_prefetch !1
  %223 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %211, <4 x float> %220) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_lds_prefetch !1
  %kdma.gep1824 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1152, !dbg !17
  %224 = load <4 x half>, ptr addrspace(3) %kdma.gep1824, align 8, !dbg !17, !frisk.k_lds_pair !40
  %kdma.gep1825 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5248, !dbg !17
  %225 = load <4 x half>, ptr addrspace(3) %kdma.gep1825, align 8, !dbg !17, !frisk.k_lds_pair !40
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
  %.idx13931397 = add nuw nsw i64 %270, %139, !dbg !16
  %271 = add i64 %.idx13931397, %268, !dbg !16
  %272 = add i64 %271, %146, !dbg !16
  %kdma.source1639 = add i64 %272, %kdma.source1452, !dbg !16
  %kdma.source1640 = trunc i64 %kdma.source1639 to i32, !dbg !16
  %kdma.source1641 = shl i32 %kdma.source1640, 1, !dbg !16
  %kdma.source1642 = add i32 %kdma.source1641, 128, !dbg !16
  %kdma.offset1643 = add i32 %kdma.source1642, %kdma.intra, !dbg !16
  %273 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %49, i32 %kdma.offset1643, <4 x i32> %dma.buffer.rsrc1339) #10, !dbg !16
  %kdma.source1662 = add i64 %272, %kdma.source1472, !dbg !16
  %kdma.source1663 = trunc i64 %kdma.source1662 to i32, !dbg !16
  %kdma.source1664 = shl i32 %kdma.source1663, 1, !dbg !16
  %kdma.source1665 = add i32 %kdma.source1664, 128, !dbg !16
  %kdma.offset1666 = add i32 %kdma.source1665, %kdma.intra.hi, !dbg !16
  %274 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1477, i32 %kdma.offset1666, <4 x i32> %dma.buffer.rsrc1339) #10, !dbg !16
  %kdma.source1687 = add i64 %272, %kdma.source1496, !dbg !16
  %kdma.source1688 = trunc i64 %kdma.source1687 to i32, !dbg !16
  %kdma.source1689 = shl i32 %kdma.source1688, 1, !dbg !16
  %kdma.source1690 = add i32 %kdma.source1689, 128, !dbg !16
  %kdma.offset1691 = add i32 %kdma.source1690, %kdma.intra, !dbg !16
  %275 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %57, i32 %kdma.offset1691, <4 x i32> %dma.buffer.rsrc1339) #10, !dbg !16
  %kdma.source1712 = add i64 %272, %kdma.source1520, !dbg !16
  %kdma.source1713 = trunc i64 %kdma.source1712 to i32, !dbg !16
  %kdma.source1714 = shl i32 %kdma.source1713, 1, !dbg !16
  %kdma.source1715 = add i32 %kdma.source1714, 128, !dbg !16
  %kdma.offset1716 = add i32 %kdma.source1715, %kdma.intra.hi, !dbg !16
  %276 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1525, i32 %kdma.offset1716, <4 x i32> %dma.buffer.rsrc1339) #10, !dbg !16
  %kdma.gep1844 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8192, !dbg !17
  %277 = load <4 x half>, ptr addrspace(3) %kdma.gep1844, align 8, !dbg !17, !frisk.k_lds_pair !41
  %kdma.gep1848 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8320, !dbg !17
  %278 = load <4 x half>, ptr addrspace(3) %kdma.gep1848, align 8, !dbg !17, !frisk.k_lds_pair !41
  %kdma.gep1845 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12288, !dbg !17
  %279 = load <4 x half>, ptr addrspace(3) %kdma.gep1845, align 8, !dbg !17, !frisk.k_lds_pair !42
  %kdma.gep1849 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12416, !dbg !17
  %280 = load <4 x half>, ptr addrspace(3) %kdma.gep1849, align 8, !dbg !17, !frisk.k_lds_pair !42
  %kdma.gep1846 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1814, i64 8192, !dbg !17
  %281 = load <4 x half>, ptr addrspace(3) %kdma.gep1846, align 8, !dbg !17, !frisk.k_lds_pair !43
  %kdma.gep1850 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1814, i64 8320, !dbg !17
  %282 = load <4 x half>, ptr addrspace(3) %kdma.gep1850, align 8, !dbg !17, !frisk.k_lds_pair !43
  %kdma.gep1847 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1814, i64 12288, !dbg !17
  %283 = load <4 x half>, ptr addrspace(3) %kdma.gep1847, align 8, !dbg !17, !frisk.k_lds_pair !44
  %kdma.gep1851 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1814, i64 12416, !dbg !17
  %284 = load <4 x half>, ptr addrspace(3) %kdma.gep1851, align 8, !dbg !17, !frisk.k_lds_pair !44
  %kdma.gep1852 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9216, !dbg !17
  %285 = load <4 x half>, ptr addrspace(3) %kdma.gep1852, align 8, !dbg !17, !frisk.k_lds_pair !45
  %kdma.gep1856 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9344, !dbg !17
  %286 = load <4 x half>, ptr addrspace(3) %kdma.gep1856, align 8, !dbg !17, !frisk.k_lds_pair !45
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_lds_prefetch !1
  %287 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %277, <4 x float> %217) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_lds_prefetch !1
  %kdma.gep1853 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13312, !dbg !17
  %288 = load <4 x half>, ptr addrspace(3) %kdma.gep1853, align 8, !dbg !17, !frisk.k_lds_pair !46
  %kdma.gep1857 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13440, !dbg !17
  %289 = load <4 x half>, ptr addrspace(3) %kdma.gep1857, align 8, !dbg !17, !frisk.k_lds_pair !46
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_lds_prefetch !1
  %290 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %279, <4 x float> %287) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_lds_prefetch !1
  %kdma.gep1854 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1814, i64 9216, !dbg !17
  %291 = load <4 x half>, ptr addrspace(3) %kdma.gep1854, align 8, !dbg !17, !frisk.k_lds_pair !47
  %kdma.gep1858 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1814, i64 9344, !dbg !17
  %292 = load <4 x half>, ptr addrspace(3) %kdma.gep1858, align 8, !dbg !17, !frisk.k_lds_pair !47
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_lds_prefetch !1
  %293 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %281, <4 x float> %223) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_lds_prefetch !1
  %kdma.gep1855 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1814, i64 13312, !dbg !17
  %294 = load <4 x half>, ptr addrspace(3) %kdma.gep1855, align 8, !dbg !17, !frisk.k_lds_pair !48
  %kdma.gep1859 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1814, i64 13440, !dbg !17
  %295 = load <4 x half>, ptr addrspace(3) %kdma.gep1859, align 8, !dbg !17, !frisk.k_lds_pair !48
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
  %.idx14051409 = add nuw nsw i64 %341, %139, !dbg !16
  %342 = add i64 %.idx14051409, %339, !dbg !16
  %343 = add i64 %342, %146, !dbg !16
  %kdma.source1735 = add i64 %343, %kdma.source1452, !dbg !16
  %kdma.source1736 = trunc i64 %kdma.source1735 to i32, !dbg !16
  %kdma.source1737 = shl i32 %kdma.source1736, 1, !dbg !16
  %kdma.source1738 = add i32 %kdma.source1737, 192, !dbg !16
  %kdma.offset1739 = add i32 %kdma.source1738, %kdma.intra, !dbg !16
  %344 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %62, i32 %kdma.offset1739, <4 x i32> %dma.buffer.rsrc1339) #10, !dbg !16
  %kdma.source1758 = add i64 %343, %kdma.source1472, !dbg !16
  %kdma.source1759 = trunc i64 %kdma.source1758 to i32, !dbg !16
  %kdma.source1760 = shl i32 %kdma.source1759, 1, !dbg !16
  %kdma.source1761 = add i32 %kdma.source1760, 192, !dbg !16
  %kdma.offset1762 = add i32 %kdma.source1761, %kdma.intra.hi, !dbg !16
  %345 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1571, i32 %kdma.offset1762, <4 x i32> %dma.buffer.rsrc1339) #10, !dbg !16
  %kdma.source1783 = add i64 %343, %kdma.source1496, !dbg !16
  %kdma.source1784 = trunc i64 %kdma.source1783 to i32, !dbg !16
  %kdma.source1785 = shl i32 %kdma.source1784, 1, !dbg !16
  %kdma.source1786 = add i32 %kdma.source1785, 192, !dbg !16
  %kdma.offset1787 = add i32 %kdma.source1786, %kdma.intra, !dbg !16
  %346 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %66, i32 %kdma.offset1787, <4 x i32> %dma.buffer.rsrc1339) #10, !dbg !16
  %kdma.source1808 = add i64 %343, %kdma.source1520, !dbg !16
  %kdma.source1809 = trunc i64 %kdma.source1808 to i32, !dbg !16
  %kdma.source1810 = shl i32 %kdma.source1809, 1, !dbg !16
  %kdma.source1811 = add i32 %kdma.source1810, 192, !dbg !16
  %kdma.offset1812 = add i32 %kdma.source1811, %kdma.intra.hi, !dbg !16
  %347 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1621, i32 %kdma.offset1812, <4 x i32> %dma.buffer.rsrc1339) #10, !dbg !16
  %348 = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8, !dbg !17, !frisk.k_lds_pair !49
  %349 = load <4 x half>, ptr addrspace(3) %kdma.gep1816, align 8, !dbg !17, !frisk.k_lds_pair !49
  %350 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8, !dbg !17, !frisk.k_lds_pair !50
  %351 = load <4 x half>, ptr addrspace(3) %kdma.gep1817, align 8, !dbg !17, !frisk.k_lds_pair !50
  %352 = load <4 x half>, ptr addrspace(3) %kdma.swizzled1814, align 8, !dbg !17, !frisk.k_lds_pair !51
  %353 = load <4 x half>, ptr addrspace(3) %kdma.gep1818, align 8, !dbg !17, !frisk.k_lds_pair !51
  %354 = load <4 x half>, ptr addrspace(3) %kdma.gep1815, align 8, !dbg !17, !frisk.k_lds_pair !52
  %355 = load <4 x half>, ptr addrspace(3) %kdma.gep1819, align 8, !dbg !17, !frisk.k_lds_pair !52
  %356 = load <4 x half>, ptr addrspace(3) %kdma.gep1820, align 8, !dbg !17, !frisk.k_lds_pair !53
  %357 = load <4 x half>, ptr addrspace(3) %kdma.gep1824, align 8, !dbg !17, !frisk.k_lds_pair !53
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_lds_prefetch !1
  %358 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %348, <4 x float> %290) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_lds_prefetch !1
  %359 = load <4 x half>, ptr addrspace(3) %kdma.gep1821, align 8, !dbg !17, !frisk.k_lds_pair !54
  %360 = load <4 x half>, ptr addrspace(3) %kdma.gep1825, align 8, !dbg !17, !frisk.k_lds_pair !54
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_lds_prefetch !1
  %361 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %350, <4 x float> %358) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_lds_prefetch !1
  %362 = load <4 x half>, ptr addrspace(3) %kdma.gep1822, align 8, !dbg !17, !frisk.k_lds_pair !55
  %363 = load <4 x half>, ptr addrspace(3) %kdma.gep1826, align 8, !dbg !17, !frisk.k_lds_pair !55
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_lds_prefetch !1
  %364 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %352, <4 x float> %296) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_lds_prefetch !1
  %365 = load <4 x half>, ptr addrspace(3) %kdma.gep1823, align 8, !dbg !17, !frisk.k_lds_pair !56
  %366 = load <4 x half>, ptr addrspace(3) %kdma.gep1827, align 8, !dbg !17, !frisk.k_lds_pair !56
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
  %address.local858 = shl nuw nsw i64 %address.local443, 5, !dbg !18
  %address.local859 = and i64 %address.local858, 8064, !dbg !18
  %address.local860 = shl nuw nsw i64 %address.local443, 3, !dbg !9
  %address.local861 = shl nuw nsw i64 %address.local449, 5, !dbg !18
  %address.local862 = sub nsw i64 %address.local860, %address.local861, !dbg !18
  %address.local863 = add nsw i64 %address.local859, %address.local862, !dbg !18
  %399 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %.idx14171419 = add i64 %398, %146, !dbg !18
  %400 = add i64 %.idx14171419, %address.local863, !dbg !18
  %.tr1420 = trunc i64 %400 to i32, !dbg !18
  %dma.buffer.offset1322 = shl i32 %.tr1420, 1, !dbg !18
  %401 = insertelement <2 x i64> poison, i64 %399, i64 0, !dbg !18
  %402 = bitcast <2 x i64> %401 to <4 x i32>, !dbg !18
  %dma.buffer.rsrc1323 = shufflevector <4 x i32> %402, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !18
  %403 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %49, i32 %dma.buffer.offset1322, <4 x i32> %dma.buffer.rsrc1323) #10, !dbg !18
  %address.local865 = shl nuw nsw i32 %address.thread442, 3, !dbg !18
  %address.local866 = and i32 %address.local865, 24, !dbg !18
  %address.local868 = and i32 %address.local516, 8064, !dbg !18
  %address.local869 = or disjoint i32 %address.local866, %address.local868, !dbg !18
  %404 = trunc i64 %.idx14171419 to i32, !dbg !18
  %address.local870 = or disjoint i32 %address.local869, %404, !dbg !18
  %.tr1424 = shl i32 %address.local870, 1, !dbg !18
  %dma.buffer.offset1320 = or i32 %.tr1424, 16384, !dbg !18
  %405 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %57, i32 %dma.buffer.offset1320, <4 x i32> %dma.buffer.rsrc1323) #10, !dbg !18
  %406 = load <4 x half>, ptr addrspace(3) %kdma.gep1844, align 8, !dbg !17, !frisk.k_tail_pair !35
  %407 = load <4 x half>, ptr addrspace(3) %kdma.gep1848, align 8, !dbg !17, !frisk.k_tail_pair !35
  %408 = load <4 x half>, ptr addrspace(3) %kdma.gep1845, align 8, !dbg !17, !frisk.k_tail_pair !36
  %409 = load <4 x half>, ptr addrspace(3) %kdma.gep1849, align 8, !dbg !17, !frisk.k_tail_pair !36
  %410 = load <4 x half>, ptr addrspace(3) %kdma.gep1846, align 8, !dbg !17, !frisk.k_tail_pair !37
  %411 = load <4 x half>, ptr addrspace(3) %kdma.gep1850, align 8, !dbg !17, !frisk.k_tail_pair !37
  %412 = load <4 x half>, ptr addrspace(3) %kdma.gep1847, align 8, !dbg !17, !frisk.k_tail_pair !38
  %413 = load <4 x half>, ptr addrspace(3) %kdma.gep1851, align 8, !dbg !17, !frisk.k_tail_pair !38
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
  %454 = call float @llvm.maxnum.f32(float %451, float 0xFFF0000000000000), !dbg !8
  %455 = extractelement <4 x float> %448, i64 2, !dbg !17
  %456 = select i1 %.not46, float %455, float 0xFFF0000000000000, !dbg !19
  %457 = extractelement <4 x float> %448, i64 3, !dbg !17
  %458 = select i1 %.not47, float %457, float 0xFFF0000000000000, !dbg !19
  %459 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %412, <4 x float> %449) #9, !dbg !17
  %460 = load <4 x half>, ptr addrspace(3) %kdma.gep1852, align 8, !dbg !17, !frisk.k_tail_pair !34
  %461 = load <4 x half>, ptr addrspace(3) %kdma.gep1856, align 8, !dbg !17, !frisk.k_tail_pair !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_tail_prefetch !1
  %462 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %407, <4 x float> %369) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_tail_prefetch !1
  %463 = extractelement <4 x float> %459, i64 0, !dbg !17
  %464 = select i1 %.not48, float %463, float 0xFFF0000000000000, !dbg !19
  %465 = extractelement <4 x float> %459, i64 1, !dbg !17
  %466 = select i1 %.not49, float %465, float 0xFFF0000000000000, !dbg !19
  %467 = call float @llvm.maxnum.f32(float %454, float %464), !dbg !8
  %468 = extractelement <4 x float> %459, i64 2, !dbg !17
  %469 = select i1 %.not50, float %468, float 0xFFF0000000000000, !dbg !19
  %470 = call float @llvm.maxnum.f32(float %453, float %466), !dbg !8
  %471 = extractelement <4 x float> %459, i64 3, !dbg !17
  %472 = select i1 %.not51, float %471, float 0xFFF0000000000000, !dbg !19
  %473 = call float @llvm.maxnum.f32(float %456, float %469), !dbg !8
  %474 = call float @llvm.maxnum.f32(float %458, float %472), !dbg !8
  %475 = load <4 x half>, ptr addrspace(3) %kdma.gep1853, align 8, !dbg !17, !frisk.k_tail_pair !39
  %476 = load <4 x half>, ptr addrspace(3) %kdma.gep1857, align 8, !dbg !17, !frisk.k_tail_pair !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_tail_prefetch !1
  %477 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %409, <4 x float> %462) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_tail_prefetch !1
  %478 = load <4 x half>, ptr addrspace(3) %kdma.gep1854, align 8, !dbg !17, !frisk.k_tail_pair !40
  %479 = load <4 x half>, ptr addrspace(3) %kdma.gep1858, align 8, !dbg !17, !frisk.k_tail_pair !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_tail_prefetch !1
  %480 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %411, <4 x float> %371) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_tail_prefetch !1
  %481 = extractelement <4 x float> %477, i64 0, !dbg !17
  %482 = select i1 %.not52, float %481, float 0xFFF0000000000000, !dbg !19
  %483 = extractelement <4 x float> %477, i64 1, !dbg !17
  %484 = select i1 %.not53, float %483, float 0xFFF0000000000000, !dbg !19
  %485 = call float @llvm.maxnum.f32(float %467, float %482), !dbg !8
  %486 = extractelement <4 x float> %477, i64 2, !dbg !17
  %487 = select i1 %.not54, float %486, float 0xFFF0000000000000, !dbg !19
  %488 = call float @llvm.maxnum.f32(float %470, float %484), !dbg !8
  %489 = extractelement <4 x float> %477, i64 3, !dbg !17
  %490 = select i1 %.not55, float %489, float 0xFFF0000000000000, !dbg !19
  %491 = call float @llvm.maxnum.f32(float %473, float %487), !dbg !8
  %492 = call float @llvm.maxnum.f32(float %474, float %490), !dbg !8
  %493 = load <4 x half>, ptr addrspace(3) %kdma.gep1855, align 8, !dbg !17, !frisk.k_tail_pair !41
  %494 = load <4 x half>, ptr addrspace(3) %kdma.gep1859, align 8, !dbg !17, !frisk.k_tail_pair !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_tail_prefetch !1
  %495 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %413, <4 x float> %480) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_tail_prefetch !1
  %496 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %460, <4 x float> %373) #9, !dbg !17
  %497 = extractelement <4 x float> %495, i64 0, !dbg !17
  %498 = select i1 %.not56, float %497, float 0xFFF0000000000000, !dbg !19
  %499 = extractelement <4 x float> %495, i64 1, !dbg !17
  %500 = select i1 %.not57, float %499, float 0xFFF0000000000000, !dbg !19
  %501 = call float @llvm.maxnum.f32(float %485, float %498), !dbg !8
  %502 = extractelement <4 x float> %495, i64 2, !dbg !17
  %503 = select i1 %.not58, float %502, float 0xFFF0000000000000, !dbg !19
  %504 = call float @llvm.maxnum.f32(float %488, float %500), !dbg !8
  %505 = extractelement <4 x float> %495, i64 3, !dbg !17
  %506 = select i1 %.not59, float %505, float 0xFFF0000000000000, !dbg !19
  %507 = call float @llvm.maxnum.f32(float %491, float %503), !dbg !8
  %508 = call float @llvm.maxnum.f32(float %492, float %506), !dbg !8
  %509 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %475, <4 x float> %496) #9, !dbg !17
  %510 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %478, <4 x float> %375) #9, !dbg !17
  %511 = extractelement <4 x float> %509, i64 0, !dbg !17
  %512 = select i1 %.not60, float %511, float 0xFFF0000000000000, !dbg !19
  %513 = extractelement <4 x float> %509, i64 1, !dbg !17
  %514 = select i1 %.not61, float %513, float 0xFFF0000000000000, !dbg !19
  %515 = call float @llvm.maxnum.f32(float %501, float %512), !dbg !8
  %516 = extractelement <4 x float> %509, i64 2, !dbg !17
  %517 = select i1 %.not62, float %516, float 0xFFF0000000000000, !dbg !19
  %518 = call float @llvm.maxnum.f32(float %504, float %514), !dbg !8
  %519 = extractelement <4 x float> %509, i64 3, !dbg !17
  %520 = select i1 %.not63, float %519, float 0xFFF0000000000000, !dbg !19
  %521 = call float @llvm.maxnum.f32(float %507, float %517), !dbg !8
  %522 = call float @llvm.maxnum.f32(float %508, float %520), !dbg !8
  %523 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %493, <4 x float> %510) #9, !dbg !17
  %524 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %461, <4 x float> %377) #9, !dbg !17
  %525 = extractelement <4 x float> %523, i64 0, !dbg !17
  %526 = select i1 %.not64, float %525, float 0xFFF0000000000000, !dbg !19
  %527 = extractelement <4 x float> %523, i64 1, !dbg !17
  %528 = select i1 %.not65, float %527, float 0xFFF0000000000000, !dbg !19
  %529 = call float @llvm.maxnum.f32(float %515, float %526), !dbg !8
  %530 = extractelement <4 x float> %523, i64 2, !dbg !17
  %531 = select i1 %.not66, float %530, float 0xFFF0000000000000, !dbg !19
  %532 = call float @llvm.maxnum.f32(float %518, float %528), !dbg !8
  %533 = extractelement <4 x float> %523, i64 3, !dbg !17
  %534 = select i1 %.not67, float %533, float 0xFFF0000000000000, !dbg !19
  %535 = call float @llvm.maxnum.f32(float %521, float %531), !dbg !8
  %536 = call float @llvm.maxnum.f32(float %522, float %534), !dbg !8
  %537 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %476, <4 x float> %524) #9, !dbg !17
  %538 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %479, <4 x float> %379) #9, !dbg !17
  %539 = extractelement <4 x float> %537, i64 0, !dbg !17
  %540 = select i1 %.not68, float %539, float 0xFFF0000000000000, !dbg !19
  %541 = extractelement <4 x float> %537, i64 1, !dbg !17
  %542 = select i1 %.not69, float %541, float 0xFFF0000000000000, !dbg !19
  %543 = call float @llvm.maxnum.f32(float %529, float %540), !dbg !8
  %544 = extractelement <4 x float> %537, i64 2, !dbg !17
  %545 = select i1 %.not70, float %544, float 0xFFF0000000000000, !dbg !19
  %546 = call float @llvm.maxnum.f32(float %532, float %542), !dbg !8
  %547 = extractelement <4 x float> %537, i64 3, !dbg !17
  %548 = select i1 %.not71, float %547, float 0xFFF0000000000000, !dbg !19
  %549 = call float @llvm.maxnum.f32(float %535, float %545), !dbg !8
  %550 = call float @llvm.maxnum.f32(float %536, float %548), !dbg !8
  %551 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %494, <4 x float> %538) #9, !dbg !17
  %552 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %406, <4 x float> %381) #9, !dbg !17
  %553 = extractelement <4 x float> %551, i64 0, !dbg !17
  %554 = select i1 %.not72, float %553, float 0xFFF0000000000000, !dbg !19
  %555 = extractelement <4 x float> %551, i64 1, !dbg !17
  %556 = select i1 %.not73, float %555, float 0xFFF0000000000000, !dbg !19
  %557 = call float @llvm.maxnum.f32(float %543, float %554), !dbg !8
  %558 = extractelement <4 x float> %551, i64 2, !dbg !17
  %559 = select i1 %.not74, float %558, float 0xFFF0000000000000, !dbg !19
  %560 = call float @llvm.maxnum.f32(float %546, float %556), !dbg !8
  %561 = extractelement <4 x float> %551, i64 3, !dbg !17
  %562 = select i1 %.not75, float %561, float 0xFFF0000000000000, !dbg !19
  %563 = call float @llvm.maxnum.f32(float %549, float %559), !dbg !8
  %564 = call float @llvm.maxnum.f32(float %550, float %562), !dbg !8
  %565 = call float @llvm.maxnum.f32(float %557, float %560), !dbg !8
  %566 = call float @llvm.maxnum.f32(float %563, float %564), !dbg !8
  %567 = call float @llvm.maxnum.f32(float %565, float %566), !dbg !8
  %568 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %105, <4 x half> %408, <4 x float> %552) #9, !dbg !17
  %569 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %410, <4 x float> %383) #9, !dbg !17
  %570 = extractelement <4 x float> %568, i64 0, !dbg !17
  %571 = select i1 %.not76, float %570, float 0xFFF0000000000000, !dbg !19
  %572 = extractelement <4 x float> %568, i64 1, !dbg !17
  %573 = select i1 %.not77, float %572, float 0xFFF0000000000000, !dbg !19
  %574 = extractelement <4 x float> %568, i64 2, !dbg !17
  %575 = select i1 %.not78, float %574, float 0xFFF0000000000000, !dbg !19
  %576 = extractelement <4 x float> %568, i64 3, !dbg !17
  %577 = select i1 %.not79, float %576, float 0xFFF0000000000000, !dbg !19
  %578 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %105, <4 x half> %412, <4 x float> %569) #9, !dbg !17
  %579 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %407, <4 x float> %385) #9, !dbg !17
  %580 = extractelement <4 x float> %578, i64 0, !dbg !17
  %581 = select i1 %.not80, float %580, float 0xFFF0000000000000, !dbg !19
  %582 = extractelement <4 x float> %578, i64 1, !dbg !17
  %583 = select i1 %.not81, float %582, float 0xFFF0000000000000, !dbg !19
  %584 = extractelement <4 x float> %578, i64 2, !dbg !17
  %585 = select i1 %.not82, float %584, float 0xFFF0000000000000, !dbg !19
  %586 = extractelement <4 x float> %578, i64 3, !dbg !17
  %587 = select i1 %.not83, float %586, float 0xFFF0000000000000, !dbg !19
  %588 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %105, <4 x half> %409, <4 x float> %579) #9, !dbg !17
  %589 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %411, <4 x float> %387) #9, !dbg !17
  %590 = extractelement <4 x float> %588, i64 0, !dbg !17
  %591 = select i1 %.not84, float %590, float 0xFFF0000000000000, !dbg !19
  %592 = extractelement <4 x float> %588, i64 1, !dbg !17
  %593 = select i1 %.not85, float %592, float 0xFFF0000000000000, !dbg !19
  %594 = extractelement <4 x float> %588, i64 2, !dbg !17
  %595 = select i1 %.not86, float %594, float 0xFFF0000000000000, !dbg !19
  %596 = extractelement <4 x float> %588, i64 3, !dbg !17
  %597 = select i1 %.not87, float %596, float 0xFFF0000000000000, !dbg !19
  %598 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %105, <4 x half> %413, <4 x float> %589) #9, !dbg !17
  %599 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %460, <4 x float> %389) #9, !dbg !17
  %600 = extractelement <4 x float> %598, i64 0, !dbg !17
  %601 = select i1 %.not88, float %600, float 0xFFF0000000000000, !dbg !19
  %602 = extractelement <4 x float> %598, i64 1, !dbg !17
  %603 = select i1 %.not89, float %602, float 0xFFF0000000000000, !dbg !19
  %604 = extractelement <4 x float> %598, i64 2, !dbg !17
  %605 = select i1 %.not90, float %604, float 0xFFF0000000000000, !dbg !19
  %606 = extractelement <4 x float> %598, i64 3, !dbg !17
  %607 = select i1 %.not91, float %606, float 0xFFF0000000000000, !dbg !19
  %608 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %105, <4 x half> %475, <4 x float> %599) #9, !dbg !17
  %609 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %478, <4 x float> %391) #9, !dbg !17
  %610 = extractelement <4 x float> %608, i64 0, !dbg !17
  %611 = select i1 %.not92, float %610, float 0xFFF0000000000000, !dbg !19
  %612 = extractelement <4 x float> %608, i64 1, !dbg !17
  %613 = select i1 %.not93, float %612, float 0xFFF0000000000000, !dbg !19
  %614 = extractelement <4 x float> %608, i64 2, !dbg !17
  %615 = select i1 %.not94, float %614, float 0xFFF0000000000000, !dbg !19
  %616 = extractelement <4 x float> %608, i64 3, !dbg !17
  %617 = select i1 %.not95, float %616, float 0xFFF0000000000000, !dbg !19
  %618 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %105, <4 x half> %493, <4 x float> %609) #9, !dbg !17
  %619 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %461, <4 x float> %393) #9, !dbg !17
  %620 = extractelement <4 x float> %618, i64 0, !dbg !17
  %621 = select i1 %.not96, float %620, float 0xFFF0000000000000, !dbg !19
  %622 = extractelement <4 x float> %618, i64 1, !dbg !17
  %623 = select i1 %.not97, float %622, float 0xFFF0000000000000, !dbg !19
  %624 = extractelement <4 x float> %618, i64 2, !dbg !17
  %625 = select i1 %.not98, float %624, float 0xFFF0000000000000, !dbg !19
  %626 = extractelement <4 x float> %618, i64 3, !dbg !17
  %627 = select i1 %.not99, float %626, float 0xFFF0000000000000, !dbg !19
  %628 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %105, <4 x half> %476, <4 x float> %619) #9, !dbg !17
  %629 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %479, <4 x float> %395) #9, !dbg !17
  %630 = extractelement <4 x float> %628, i64 0, !dbg !17
  %631 = select i1 %.not100, float %630, float 0xFFF0000000000000, !dbg !19
  %632 = extractelement <4 x float> %628, i64 1, !dbg !17
  %633 = select i1 %.not101, float %632, float 0xFFF0000000000000, !dbg !19
  %634 = extractelement <4 x float> %628, i64 2, !dbg !17
  %635 = select i1 %.not102, float %634, float 0xFFF0000000000000, !dbg !19
  %636 = extractelement <4 x float> %628, i64 3, !dbg !17
  %637 = select i1 %.not103, float %636, float 0xFFF0000000000000, !dbg !19
  %638 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %105, <4 x half> %494, <4 x float> %629) #9, !dbg !17
  %639 = extractelement <4 x float> %638, i64 0, !dbg !17
  %640 = select i1 %.not104, float %639, float 0xFFF0000000000000, !dbg !19
  %641 = extractelement <4 x float> %638, i64 1, !dbg !17
  %642 = select i1 %.not105, float %641, float 0xFFF0000000000000, !dbg !19
  %643 = extractelement <4 x float> %638, i64 2, !dbg !17
  %644 = select i1 %.not106, float %643, float 0xFFF0000000000000, !dbg !19
  %645 = extractelement <4 x float> %638, i64 3, !dbg !17
  %646 = select i1 %.not107, float %645, float 0xFFF0000000000000, !dbg !19
  %647 = bitcast float %567 to i32, !dbg !8
  %648 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %647), !dbg !8
  %649 = bitcast i32 %648 to float, !dbg !8
  %650 = call float @llvm.maxnum.f32(float %567, float %649), !dbg !8
  %651 = bitcast float %650 to i32, !dbg !8
  %652 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %651), !dbg !8
  %653 = bitcast i32 %652 to float, !dbg !8
  %654 = call float @llvm.maxnum.f32(float %650, float %653), !dbg !8
  %655 = call float @llvm.maxnum.f32(float %571, float 0xFFF0000000000000), !dbg !8
  %656 = call float @llvm.maxnum.f32(float %655, float %581), !dbg !8
  %657 = call float @llvm.maxnum.f32(float %573, float %583), !dbg !8
  %658 = call float @llvm.maxnum.f32(float %575, float %585), !dbg !8
  %659 = call float @llvm.maxnum.f32(float %577, float %587), !dbg !8
  %660 = call float @llvm.maxnum.f32(float %656, float %591), !dbg !8
  %661 = call float @llvm.maxnum.f32(float %657, float %593), !dbg !8
  %662 = call float @llvm.maxnum.f32(float %658, float %595), !dbg !8
  %663 = call float @llvm.maxnum.f32(float %659, float %597), !dbg !8
  %664 = call float @llvm.maxnum.f32(float %660, float %601), !dbg !8
  %665 = call float @llvm.maxnum.f32(float %661, float %603), !dbg !8
  %666 = call float @llvm.maxnum.f32(float %662, float %605), !dbg !8
  %667 = call float @llvm.maxnum.f32(float %663, float %607), !dbg !8
  %668 = call float @llvm.maxnum.f32(float %664, float %611), !dbg !8
  %669 = call float @llvm.maxnum.f32(float %665, float %613), !dbg !8
  %670 = call float @llvm.maxnum.f32(float %666, float %615), !dbg !8
  %671 = call float @llvm.maxnum.f32(float %667, float %617), !dbg !8
  %672 = call float @llvm.maxnum.f32(float %668, float %621), !dbg !8
  %673 = call float @llvm.maxnum.f32(float %669, float %623), !dbg !8
  %674 = call float @llvm.maxnum.f32(float %670, float %625), !dbg !8
  %675 = call float @llvm.maxnum.f32(float %671, float %627), !dbg !8
  %676 = call float @llvm.maxnum.f32(float %672, float %631), !dbg !8
  %677 = call float @llvm.maxnum.f32(float %673, float %633), !dbg !8
  %678 = call float @llvm.maxnum.f32(float %674, float %635), !dbg !8
  %679 = call float @llvm.maxnum.f32(float %675, float %637), !dbg !8
  %680 = call float @llvm.maxnum.f32(float %676, float %640), !dbg !8
  %681 = call float @llvm.maxnum.f32(float %677, float %642), !dbg !8
  %682 = call float @llvm.maxnum.f32(float %678, float %644), !dbg !8
  %683 = call float @llvm.maxnum.f32(float %679, float %646), !dbg !8
  %684 = call float @llvm.maxnum.f32(float %680, float %681), !dbg !8
  %685 = call float @llvm.maxnum.f32(float %682, float %683), !dbg !8
  %686 = call float @llvm.maxnum.f32(float %684, float %685), !dbg !8
  %687 = bitcast float %686 to i32, !dbg !8
  %688 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %687), !dbg !8
  %689 = bitcast i32 %688 to float, !dbg !8
  %690 = call float @llvm.maxnum.f32(float %686, float %689), !dbg !8
  %691 = bitcast float %690 to i32, !dbg !8
  %692 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %691), !dbg !8
  %693 = bitcast i32 %692 to float, !dbg !8
  %694 = call float @llvm.maxnum.f32(float %690, float %693), !dbg !8
  %695 = extractvalue [2 x <1 x float>] %131, 0, !dbg !8
  %696 = insertelement <1 x float> poison, float %654, i64 0, !dbg !8
  %697 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %696, <1 x float> %695), !dbg !8
  %698 = extractvalue [2 x <1 x float>] %131, 1, !dbg !8
  %699 = insertelement <1 x float> poison, float %694, i64 0, !dbg !8
  %700 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %699, <1 x float> %698), !dbg !8
  %701 = insertelement <4 x float> poison, float %451, i64 0, !dbg !23
  %702 = insertelement <4 x float> %701, float %453, i64 1, !dbg !23
  %703 = insertelement <4 x float> %702, float %456, i64 2, !dbg !23
  %704 = insertelement <4 x float> %703, float %458, i64 3, !dbg !23
  %705 = shufflevector <1 x float> %697, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %706 = insertelement <4 x float> poison, float %464, i64 0, !dbg !23
  %707 = insertelement <4 x float> %706, float %466, i64 1, !dbg !23
  %708 = insertelement <4 x float> %707, float %469, i64 2, !dbg !23
  %709 = insertelement <4 x float> %708, float %472, i64 3, !dbg !23
  %710 = insertelement <4 x float> poison, float %482, i64 0, !dbg !23
  %711 = insertelement <4 x float> %710, float %484, i64 1, !dbg !23
  %712 = insertelement <4 x float> %711, float %487, i64 2, !dbg !23
  %713 = insertelement <4 x float> %712, float %490, i64 3, !dbg !23
  %714 = insertelement <4 x float> poison, float %498, i64 0, !dbg !23
  %715 = insertelement <4 x float> %714, float %500, i64 1, !dbg !23
  %716 = insertelement <4 x float> %715, float %503, i64 2, !dbg !23
  %717 = insertelement <4 x float> %716, float %506, i64 3, !dbg !23
  %718 = insertelement <4 x float> poison, float %512, i64 0, !dbg !23
  %719 = insertelement <4 x float> %718, float %514, i64 1, !dbg !23
  %720 = insertelement <4 x float> %719, float %517, i64 2, !dbg !23
  %721 = insertelement <4 x float> %720, float %520, i64 3, !dbg !23
  %722 = insertelement <4 x float> poison, float %526, i64 0, !dbg !23
  %723 = insertelement <4 x float> %722, float %528, i64 1, !dbg !23
  %724 = insertelement <4 x float> %723, float %531, i64 2, !dbg !23
  %725 = insertelement <4 x float> %724, float %534, i64 3, !dbg !23
  %726 = insertelement <4 x float> poison, float %540, i64 0, !dbg !23
  %727 = insertelement <4 x float> %726, float %542, i64 1, !dbg !23
  %728 = insertelement <4 x float> %727, float %545, i64 2, !dbg !23
  %729 = insertelement <4 x float> %728, float %548, i64 3, !dbg !23
  %730 = insertelement <4 x float> poison, float %554, i64 0, !dbg !23
  %731 = insertelement <4 x float> %730, float %556, i64 1, !dbg !23
  %732 = insertelement <4 x float> %731, float %559, i64 2, !dbg !23
  %733 = insertelement <4 x float> %732, float %562, i64 3, !dbg !23
  %734 = insertelement <4 x float> poison, float %571, i64 0, !dbg !23
  %735 = insertelement <4 x float> %734, float %573, i64 1, !dbg !23
  %736 = insertelement <4 x float> %735, float %575, i64 2, !dbg !23
  %737 = insertelement <4 x float> %736, float %577, i64 3, !dbg !23
  %738 = shufflevector <1 x float> %700, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %739 = insertelement <4 x float> poison, float %581, i64 0, !dbg !23
  %740 = insertelement <4 x float> %739, float %583, i64 1, !dbg !23
  %741 = insertelement <4 x float> %740, float %585, i64 2, !dbg !23
  %742 = insertelement <4 x float> %741, float %587, i64 3, !dbg !23
  %743 = insertelement <4 x float> poison, float %591, i64 0, !dbg !23
  %744 = insertelement <4 x float> %743, float %593, i64 1, !dbg !23
  %745 = insertelement <4 x float> %744, float %595, i64 2, !dbg !23
  %746 = insertelement <4 x float> %745, float %597, i64 3, !dbg !23
  %747 = insertelement <4 x float> poison, float %601, i64 0, !dbg !23
  %748 = insertelement <4 x float> %747, float %603, i64 1, !dbg !23
  %749 = insertelement <4 x float> %748, float %605, i64 2, !dbg !23
  %750 = insertelement <4 x float> %749, float %607, i64 3, !dbg !23
  %751 = insertelement <4 x float> poison, float %611, i64 0, !dbg !23
  %752 = insertelement <4 x float> %751, float %613, i64 1, !dbg !23
  %753 = insertelement <4 x float> %752, float %615, i64 2, !dbg !23
  %754 = insertelement <4 x float> %753, float %617, i64 3, !dbg !23
  %755 = insertelement <4 x float> poison, float %621, i64 0, !dbg !23
  %756 = insertelement <4 x float> %755, float %623, i64 1, !dbg !23
  %757 = insertelement <4 x float> %756, float %625, i64 2, !dbg !23
  %758 = insertelement <4 x float> %757, float %627, i64 3, !dbg !23
  %759 = insertelement <4 x float> poison, float %631, i64 0, !dbg !23
  %760 = insertelement <4 x float> %759, float %633, i64 1, !dbg !23
  %761 = insertelement <4 x float> %760, float %635, i64 2, !dbg !23
  %762 = insertelement <4 x float> %761, float %637, i64 3, !dbg !23
  %763 = insertelement <4 x float> poison, float %640, i64 0, !dbg !23
  %764 = insertelement <4 x float> %763, float %642, i64 1, !dbg !23
  %765 = insertelement <4 x float> %764, float %644, i64 2, !dbg !23
  %766 = insertelement <4 x float> %765, float %646, i64 3, !dbg !23
  %767 = fsub <1 x float> %695, %697, !dbg !24
  %768 = extractelement <1 x float> %767, i64 0, !dbg !25
  %769 = call float @llvm.amdgcn.exp2.f32(float %768), !dbg !25
  %770 = insertelement <4 x float> poison, float %769, i64 0, !dbg !30
  %771 = shufflevector <4 x float> %770, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !30
  %772 = extractvalue [2 x <1 x float>] %130, 0, !dbg !26
  %773 = extractelement <1 x float> %772, i64 0, !dbg !26
  %774 = fmul float %773, %769, !dbg !26
  %775 = fsub <1 x float> %698, %700, !dbg !24
  %776 = extractelement <1 x float> %775, i64 0, !dbg !25
  %777 = call float @llvm.amdgcn.exp2.f32(float %776), !dbg !25
  %778 = insertelement <4 x float> poison, float %777, i64 0, !dbg !30
  %779 = shufflevector <4 x float> %778, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !30
  %780 = extractvalue [2 x <1 x float>] %130, 1, !dbg !26
  %781 = extractelement <1 x float> %780, i64 0, !dbg !26
  %782 = fmul float %781, %777, !dbg !26
  %783 = extractvalue [2 x <8 x float>] %129, 0, !dbg !30
  %784 = shufflevector <8 x float> %783, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %785 = fmul <4 x float> %784, %771, !dbg !30
  %786 = shufflevector <8 x float> %783, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %787 = fmul <4 x float> %786, %771, !dbg !30
  %788 = extractvalue [2 x <8 x float>] %129, 1, !dbg !30
  %789 = shufflevector <8 x float> %788, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %790 = fmul <4 x float> %789, %779, !dbg !30
  %791 = shufflevector <8 x float> %788, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %792 = fmul <4 x float> %791, %779, !dbg !30
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %793 = or disjoint i64 %.idx14171419, 32, !dbg !18
  %794 = add i64 %address.local863, %793, !dbg !18
  %.tr1426 = trunc i64 %794 to i32, !dbg !18
  %dma.buffer.offset1318 = shl i32 %.tr1426, 1, !dbg !18
  %795 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %62, i32 %dma.buffer.offset1318, <4 x i32> %dma.buffer.rsrc1323) #10, !dbg !18
  %dma.buffer.offset1316 = or i32 %.tr1424, 16448, !dbg !18
  %796 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %66, i32 %dma.buffer.offset1316, <4 x i32> %dma.buffer.rsrc1323) #10, !dbg !18
  %address.local876 = and i32 %address.local865, 120, !dbg !14
  %address.local878 = and i32 %address.local516, 1536, !dbg !14
  %address.local879 = or disjoint i32 %address.local876, %address.local878, !dbg !14
  %address.local880 = zext nneg i32 %address.local879 to i64, !dbg !14
  %address.local881 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local880, !dbg !14
  %797 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local881, i32 0) #0, !dbg !14
  %798 = bitcast <4 x i32> %797 to <8 x half>, !dbg !14
  %address.local887 = or disjoint i32 %address.local879, 128, !dbg !14
  %address.local888 = zext nneg i32 %address.local887 to i64, !dbg !14
  %address.local889 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local888, !dbg !14
  %799 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local889, i32 0) #0, !dbg !14
  %800 = bitcast <4 x i32> %799 to <8 x half>, !dbg !14
  %address.local895 = or disjoint i32 %address.local879, 256, !dbg !14
  %address.local896 = zext nneg i32 %address.local895 to i64, !dbg !14
  %address.local897 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local896, !dbg !14
  %801 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local897, i32 0) #0, !dbg !14
  %802 = bitcast <4 x i32> %801 to <8 x half>, !dbg !14
  %address.local903 = or disjoint i32 %address.local879, 384, !dbg !14
  %address.local904 = zext nneg i32 %address.local903 to i64, !dbg !14
  %address.local905 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local904, !dbg !14
  %803 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local905, i32 0) #0, !dbg !14
  %804 = bitcast <4 x i32> %803 to <8 x half>, !dbg !14
  %address.local911 = or disjoint i32 %address.local879, 2048, !dbg !14
  %address.local912 = zext nneg i32 %address.local911 to i64, !dbg !14
  %address.local913 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local912, !dbg !14
  %805 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local913, i32 0) #0, !dbg !14
  %806 = bitcast <4 x i32> %805 to <8 x half>, !dbg !14
  %address.local919 = or disjoint i32 %address.local879, 2176, !dbg !14
  %address.local920 = zext nneg i32 %address.local919 to i64, !dbg !14
  %address.local921 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local920, !dbg !14
  %807 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local921, i32 0) #0, !dbg !14
  %808 = bitcast <4 x i32> %807 to <8 x half>, !dbg !14
  %address.local927 = or disjoint i32 %address.local879, 2304, !dbg !14
  %address.local928 = zext nneg i32 %address.local927 to i64, !dbg !14
  %address.local929 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local928, !dbg !14
  %809 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local929, i32 0) #0, !dbg !14
  %810 = bitcast <4 x i32> %809 to <8 x half>, !dbg !14
  %address.local935 = or disjoint i32 %address.local879, 2432, !dbg !14
  %address.local936 = zext nneg i32 %address.local935 to i64, !dbg !14
  %address.local937 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local936, !dbg !14
  %811 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local937, i32 0) #0, !dbg !14
  %812 = shufflevector <8 x half> %798, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %813 = fsub <4 x float> %704, %705, !dbg !23
  %814 = extractelement <4 x float> %813, i64 0, !dbg !27
  %815 = call float @llvm.amdgcn.exp2.f32(float %814), !dbg !27
  %816 = insertelement <4 x float> poison, float %815, i64 0, !dbg !29
  %817 = extractelement <4 x float> %813, i64 1, !dbg !27
  %818 = call float @llvm.amdgcn.exp2.f32(float %817), !dbg !27
  %819 = insertelement <4 x float> %816, float %818, i64 1, !dbg !29
  %820 = extractelement <4 x float> %813, i64 2, !dbg !27
  %821 = call float @llvm.amdgcn.exp2.f32(float %820), !dbg !27
  %822 = insertelement <4 x float> %819, float %821, i64 2, !dbg !29
  %823 = extractelement <4 x float> %813, i64 3, !dbg !27
  %824 = call float @llvm.amdgcn.exp2.f32(float %823), !dbg !27
  %825 = insertelement <4 x float> %822, float %824, i64 3, !dbg !29
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %825) #9, !dbg !29, !frisk.register_lifetimes !1
  %.cvtpack.f0 = extractelement <4 x float> %math.result, i32 0, !dbg !29
  %.cvtpack.f1 = extractelement <4 x float> %math.result, i32 1, !dbg !29
  %.cvtpack.f2 = extractelement <4 x float> %math.result, i32 2, !dbg !29
  %.cvtpack.f3 = extractelement <4 x float> %math.result, i32 3, !dbg !29
  %.cvtpack.p0 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f0, float %.cvtpack.f1), !dbg !29
  %.cvtpack.p1 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f2, float %.cvtpack.f3), !dbg !29
  %.cvtpack.v0 = insertelement <2 x i32> poison, i32 %.cvtpack.p0, i32 0, !dbg !29
  %.cvtpack.v1 = insertelement <2 x i32> %.cvtpack.v0, i32 %.cvtpack.p1, i32 1, !dbg !29
  %826 = bitcast <2 x i32> %.cvtpack.v1 to <4 x half>, !dbg !29
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %826) #9, !dbg !27, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %827 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %812, <4 x float> %785) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %828 = shufflevector <8 x half> %798, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %829 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %828, <4 x float> %787) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %830 = fsub <4 x float> %737, %738, !dbg !23
  %831 = extractelement <4 x float> %830, i64 0, !dbg !27
  %832 = call float @llvm.amdgcn.exp2.f32(float %831), !dbg !27
  %833 = insertelement <4 x float> poison, float %832, i64 0, !dbg !29
  %834 = extractelement <4 x float> %830, i64 1, !dbg !27
  %835 = call float @llvm.amdgcn.exp2.f32(float %834), !dbg !27
  %836 = insertelement <4 x float> %833, float %835, i64 1, !dbg !29
  %837 = extractelement <4 x float> %830, i64 2, !dbg !27
  %838 = call float @llvm.amdgcn.exp2.f32(float %837), !dbg !27
  %839 = insertelement <4 x float> %836, float %838, i64 2, !dbg !29
  %840 = extractelement <4 x float> %830, i64 3, !dbg !27
  %841 = call float @llvm.amdgcn.exp2.f32(float %840), !dbg !27
  %842 = insertelement <4 x float> %839, float %841, i64 3, !dbg !29
  %math.result329 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %842) #9, !dbg !29, !frisk.register_lifetimes !1
  %.cvtpack.f01947 = extractelement <4 x float> %math.result329, i32 0, !dbg !29
  %.cvtpack.f11948 = extractelement <4 x float> %math.result329, i32 1, !dbg !29
  %.cvtpack.f21949 = extractelement <4 x float> %math.result329, i32 2, !dbg !29
  %.cvtpack.f31950 = extractelement <4 x float> %math.result329, i32 3, !dbg !29
  %.cvtpack.p01951 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f01947, float %.cvtpack.f11948), !dbg !29
  %.cvtpack.p11952 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f21949, float %.cvtpack.f31950), !dbg !29
  %.cvtpack.v01953 = insertelement <2 x i32> poison, i32 %.cvtpack.p01951, i32 0, !dbg !29
  %.cvtpack.v11954 = insertelement <2 x i32> %.cvtpack.v01953, i32 %.cvtpack.p11952, i32 1, !dbg !29
  %843 = bitcast <2 x i32> %.cvtpack.v11954 to <4 x half>, !dbg !29
  %math.packed330 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %843) #9, !dbg !27, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %844 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %812, <4 x float> %790) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %845 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %828, <4 x float> %792) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %846 = shufflevector <8 x half> %800, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %847 = fsub <4 x float> %709, %705, !dbg !23
  %848 = extractelement <4 x float> %847, i64 0, !dbg !27
  %849 = call float @llvm.amdgcn.exp2.f32(float %848), !dbg !27
  %850 = fadd float %815, %849, !dbg !28
  %851 = insertelement <4 x float> poison, float %849, i64 0, !dbg !29
  %852 = extractelement <4 x float> %847, i64 1, !dbg !27
  %853 = call float @llvm.amdgcn.exp2.f32(float %852), !dbg !27
  %854 = fadd float %818, %853, !dbg !28
  %855 = insertelement <4 x float> %851, float %853, i64 1, !dbg !29
  %856 = extractelement <4 x float> %847, i64 2, !dbg !27
  %857 = call float @llvm.amdgcn.exp2.f32(float %856), !dbg !27
  %858 = fadd float %821, %857, !dbg !28
  %859 = insertelement <4 x float> %855, float %857, i64 2, !dbg !29
  %860 = extractelement <4 x float> %847, i64 3, !dbg !27
  %861 = call float @llvm.amdgcn.exp2.f32(float %860), !dbg !27
  %862 = fadd float %824, %861, !dbg !28
  %863 = insertelement <4 x float> %859, float %861, i64 3, !dbg !29
  %math.result315 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %863) #9, !dbg !29, !frisk.register_lifetimes !1
  %.cvtpack.f01955 = extractelement <4 x float> %math.result315, i32 0, !dbg !29
  %.cvtpack.f11956 = extractelement <4 x float> %math.result315, i32 1, !dbg !29
  %.cvtpack.f21957 = extractelement <4 x float> %math.result315, i32 2, !dbg !29
  %.cvtpack.f31958 = extractelement <4 x float> %math.result315, i32 3, !dbg !29
  %.cvtpack.p01959 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f01955, float %.cvtpack.f11956), !dbg !29
  %.cvtpack.p11960 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f21957, float %.cvtpack.f31958), !dbg !29
  %.cvtpack.v01961 = insertelement <2 x i32> poison, i32 %.cvtpack.p01959, i32 0, !dbg !29
  %.cvtpack.v11962 = insertelement <2 x i32> %.cvtpack.v01961, i32 %.cvtpack.p11960, i32 1, !dbg !29
  %864 = bitcast <2 x i32> %.cvtpack.v11962 to <4 x half>, !dbg !29
  %math.packed316 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %864) #9, !dbg !28, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %865 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %846, <4 x float> %827) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %866 = shufflevector <8 x half> %800, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %867 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %866, <4 x float> %829) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %868 = fsub <4 x float> %742, %738, !dbg !23
  %869 = extractelement <4 x float> %868, i64 0, !dbg !27
  %870 = call float @llvm.amdgcn.exp2.f32(float %869), !dbg !27
  %871 = fadd float %832, %870, !dbg !28
  %872 = insertelement <4 x float> poison, float %870, i64 0, !dbg !29
  %873 = extractelement <4 x float> %868, i64 1, !dbg !27
  %874 = call float @llvm.amdgcn.exp2.f32(float %873), !dbg !27
  %875 = fadd float %835, %874, !dbg !28
  %876 = insertelement <4 x float> %872, float %874, i64 1, !dbg !29
  %877 = extractelement <4 x float> %868, i64 2, !dbg !27
  %878 = call float @llvm.amdgcn.exp2.f32(float %877), !dbg !27
  %879 = fadd float %838, %878, !dbg !28
  %880 = insertelement <4 x float> %876, float %878, i64 2, !dbg !29
  %881 = extractelement <4 x float> %868, i64 3, !dbg !27
  %882 = call float @llvm.amdgcn.exp2.f32(float %881), !dbg !27
  %883 = fadd float %841, %882, !dbg !28
  %884 = insertelement <4 x float> %880, float %882, i64 3, !dbg !29
  %math.result331 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %884) #9, !dbg !29, !frisk.register_lifetimes !1
  %.cvtpack.f01963 = extractelement <4 x float> %math.result331, i32 0, !dbg !29
  %.cvtpack.f11964 = extractelement <4 x float> %math.result331, i32 1, !dbg !29
  %.cvtpack.f21965 = extractelement <4 x float> %math.result331, i32 2, !dbg !29
  %.cvtpack.f31966 = extractelement <4 x float> %math.result331, i32 3, !dbg !29
  %.cvtpack.p01967 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f01963, float %.cvtpack.f11964), !dbg !29
  %.cvtpack.p11968 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f21965, float %.cvtpack.f31966), !dbg !29
  %.cvtpack.v01969 = insertelement <2 x i32> poison, i32 %.cvtpack.p01967, i32 0, !dbg !29
  %.cvtpack.v11970 = insertelement <2 x i32> %.cvtpack.v01969, i32 %.cvtpack.p11968, i32 1, !dbg !29
  %885 = bitcast <2 x i32> %.cvtpack.v11970 to <4 x half>, !dbg !29
  %math.packed332 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %885) #9, !dbg !28, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %886 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %846, <4 x float> %844) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %887 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %866, <4 x float> %845) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %888 = shufflevector <8 x half> %802, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %889 = fsub <4 x float> %713, %705, !dbg !23
  %890 = extractelement <4 x float> %889, i64 0, !dbg !27
  %891 = call float @llvm.amdgcn.exp2.f32(float %890), !dbg !27
  %892 = fadd float %850, %891, !dbg !28
  %893 = insertelement <4 x float> poison, float %891, i64 0, !dbg !29
  %894 = extractelement <4 x float> %889, i64 1, !dbg !27
  %895 = call float @llvm.amdgcn.exp2.f32(float %894), !dbg !27
  %896 = fadd float %854, %895, !dbg !28
  %897 = insertelement <4 x float> %893, float %895, i64 1, !dbg !29
  %898 = extractelement <4 x float> %889, i64 2, !dbg !27
  %899 = call float @llvm.amdgcn.exp2.f32(float %898), !dbg !27
  %900 = fadd float %858, %899, !dbg !28
  %901 = insertelement <4 x float> %897, float %899, i64 2, !dbg !29
  %902 = extractelement <4 x float> %889, i64 3, !dbg !27
  %903 = call float @llvm.amdgcn.exp2.f32(float %902), !dbg !27
  %904 = fadd float %862, %903, !dbg !28
  %905 = insertelement <4 x float> %901, float %903, i64 3, !dbg !29
  %math.result317 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %905) #9, !dbg !29, !frisk.register_lifetimes !1
  %.cvtpack.f01971 = extractelement <4 x float> %math.result317, i32 0, !dbg !29
  %.cvtpack.f11972 = extractelement <4 x float> %math.result317, i32 1, !dbg !29
  %.cvtpack.f21973 = extractelement <4 x float> %math.result317, i32 2, !dbg !29
  %.cvtpack.f31974 = extractelement <4 x float> %math.result317, i32 3, !dbg !29
  %.cvtpack.p01975 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f01971, float %.cvtpack.f11972), !dbg !29
  %.cvtpack.p11976 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f21973, float %.cvtpack.f31974), !dbg !29
  %.cvtpack.v01977 = insertelement <2 x i32> poison, i32 %.cvtpack.p01975, i32 0, !dbg !29
  %.cvtpack.v11978 = insertelement <2 x i32> %.cvtpack.v01977, i32 %.cvtpack.p11976, i32 1, !dbg !29
  %906 = bitcast <2 x i32> %.cvtpack.v11978 to <4 x half>, !dbg !29
  %math.packed318 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %906) #9, !dbg !28, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %907 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %888, <4 x float> %865) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %908 = shufflevector <8 x half> %802, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %909 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %908, <4 x float> %867) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %910 = fsub <4 x float> %746, %738, !dbg !23
  %911 = extractelement <4 x float> %910, i64 0, !dbg !27
  %912 = call float @llvm.amdgcn.exp2.f32(float %911), !dbg !27
  %913 = fadd float %871, %912, !dbg !28
  %914 = insertelement <4 x float> poison, float %912, i64 0, !dbg !29
  %915 = extractelement <4 x float> %910, i64 1, !dbg !27
  %916 = call float @llvm.amdgcn.exp2.f32(float %915), !dbg !27
  %917 = fadd float %875, %916, !dbg !28
  %918 = insertelement <4 x float> %914, float %916, i64 1, !dbg !29
  %919 = extractelement <4 x float> %910, i64 2, !dbg !27
  %920 = call float @llvm.amdgcn.exp2.f32(float %919), !dbg !27
  %921 = fadd float %879, %920, !dbg !28
  %922 = insertelement <4 x float> %918, float %920, i64 2, !dbg !29
  %923 = extractelement <4 x float> %910, i64 3, !dbg !27
  %924 = call float @llvm.amdgcn.exp2.f32(float %923), !dbg !27
  %925 = fadd float %883, %924, !dbg !28
  %926 = insertelement <4 x float> %922, float %924, i64 3, !dbg !29
  %math.result333 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %926) #9, !dbg !29, !frisk.register_lifetimes !1
  %.cvtpack.f01979 = extractelement <4 x float> %math.result333, i32 0, !dbg !29
  %.cvtpack.f11980 = extractelement <4 x float> %math.result333, i32 1, !dbg !29
  %.cvtpack.f21981 = extractelement <4 x float> %math.result333, i32 2, !dbg !29
  %.cvtpack.f31982 = extractelement <4 x float> %math.result333, i32 3, !dbg !29
  %.cvtpack.p01983 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f01979, float %.cvtpack.f11980), !dbg !29
  %.cvtpack.p11984 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f21981, float %.cvtpack.f31982), !dbg !29
  %.cvtpack.v01985 = insertelement <2 x i32> poison, i32 %.cvtpack.p01983, i32 0, !dbg !29
  %.cvtpack.v11986 = insertelement <2 x i32> %.cvtpack.v01985, i32 %.cvtpack.p11984, i32 1, !dbg !29
  %927 = bitcast <2 x i32> %.cvtpack.v11986 to <4 x half>, !dbg !29
  %math.packed334 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %927) #9, !dbg !28, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %928 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %888, <4 x float> %886) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %929 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %908, <4 x float> %887) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %930 = shufflevector <8 x half> %804, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %931 = fsub <4 x float> %717, %705, !dbg !23
  %932 = extractelement <4 x float> %931, i64 0, !dbg !27
  %933 = call float @llvm.amdgcn.exp2.f32(float %932), !dbg !27
  %934 = fadd float %892, %933, !dbg !28
  %935 = insertelement <4 x float> poison, float %933, i64 0, !dbg !29
  %936 = extractelement <4 x float> %931, i64 1, !dbg !27
  %937 = call float @llvm.amdgcn.exp2.f32(float %936), !dbg !27
  %938 = fadd float %896, %937, !dbg !28
  %939 = insertelement <4 x float> %935, float %937, i64 1, !dbg !29
  %940 = extractelement <4 x float> %931, i64 2, !dbg !27
  %941 = call float @llvm.amdgcn.exp2.f32(float %940), !dbg !27
  %942 = fadd float %900, %941, !dbg !28
  %943 = insertelement <4 x float> %939, float %941, i64 2, !dbg !29
  %944 = extractelement <4 x float> %931, i64 3, !dbg !27
  %945 = call float @llvm.amdgcn.exp2.f32(float %944), !dbg !27
  %946 = fadd float %904, %945, !dbg !28
  %947 = insertelement <4 x float> %943, float %945, i64 3, !dbg !29
  %math.result319 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %947) #9, !dbg !29, !frisk.register_lifetimes !1
  %.cvtpack.f01987 = extractelement <4 x float> %math.result319, i32 0, !dbg !29
  %.cvtpack.f11988 = extractelement <4 x float> %math.result319, i32 1, !dbg !29
  %.cvtpack.f21989 = extractelement <4 x float> %math.result319, i32 2, !dbg !29
  %.cvtpack.f31990 = extractelement <4 x float> %math.result319, i32 3, !dbg !29
  %.cvtpack.p01991 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f01987, float %.cvtpack.f11988), !dbg !29
  %.cvtpack.p11992 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f21989, float %.cvtpack.f31990), !dbg !29
  %.cvtpack.v01993 = insertelement <2 x i32> poison, i32 %.cvtpack.p01991, i32 0, !dbg !29
  %.cvtpack.v11994 = insertelement <2 x i32> %.cvtpack.v01993, i32 %.cvtpack.p11992, i32 1, !dbg !29
  %948 = bitcast <2 x i32> %.cvtpack.v11994 to <4 x half>, !dbg !29
  %math.packed320 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %948) #9, !dbg !28, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %949 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %930, <4 x float> %907) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %950 = shufflevector <8 x half> %804, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %951 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %950, <4 x float> %909) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %952 = fsub <4 x float> %750, %738, !dbg !23
  %953 = extractelement <4 x float> %952, i64 0, !dbg !27
  %954 = call float @llvm.amdgcn.exp2.f32(float %953), !dbg !27
  %955 = fadd float %913, %954, !dbg !28
  %956 = insertelement <4 x float> poison, float %954, i64 0, !dbg !29
  %957 = extractelement <4 x float> %952, i64 1, !dbg !27
  %958 = call float @llvm.amdgcn.exp2.f32(float %957), !dbg !27
  %959 = fadd float %917, %958, !dbg !28
  %960 = insertelement <4 x float> %956, float %958, i64 1, !dbg !29
  %961 = extractelement <4 x float> %952, i64 2, !dbg !27
  %962 = call float @llvm.amdgcn.exp2.f32(float %961), !dbg !27
  %963 = fadd float %921, %962, !dbg !28
  %964 = insertelement <4 x float> %960, float %962, i64 2, !dbg !29
  %965 = extractelement <4 x float> %952, i64 3, !dbg !27
  %966 = call float @llvm.amdgcn.exp2.f32(float %965), !dbg !27
  %967 = fadd float %925, %966, !dbg !28
  %968 = insertelement <4 x float> %964, float %966, i64 3, !dbg !29
  %math.result335 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %968) #9, !dbg !29, !frisk.register_lifetimes !1
  %.cvtpack.f01995 = extractelement <4 x float> %math.result335, i32 0, !dbg !29
  %.cvtpack.f11996 = extractelement <4 x float> %math.result335, i32 1, !dbg !29
  %.cvtpack.f21997 = extractelement <4 x float> %math.result335, i32 2, !dbg !29
  %.cvtpack.f31998 = extractelement <4 x float> %math.result335, i32 3, !dbg !29
  %.cvtpack.p01999 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f01995, float %.cvtpack.f11996), !dbg !29
  %.cvtpack.p12000 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f21997, float %.cvtpack.f31998), !dbg !29
  %.cvtpack.v02001 = insertelement <2 x i32> poison, i32 %.cvtpack.p01999, i32 0, !dbg !29
  %.cvtpack.v12002 = insertelement <2 x i32> %.cvtpack.v02001, i32 %.cvtpack.p12000, i32 1, !dbg !29
  %969 = bitcast <2 x i32> %.cvtpack.v12002 to <4 x half>, !dbg !29
  %math.packed336 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %969) #9, !dbg !28, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %970 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %930, <4 x float> %928) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %971 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %950, <4 x float> %929) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %972 = shufflevector <8 x half> %806, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %973 = fsub <4 x float> %721, %705, !dbg !23
  %974 = extractelement <4 x float> %973, i64 0, !dbg !27
  %975 = call float @llvm.amdgcn.exp2.f32(float %974), !dbg !27
  %976 = fadd float %934, %975, !dbg !28
  %977 = insertelement <4 x float> poison, float %975, i64 0, !dbg !29
  %978 = extractelement <4 x float> %973, i64 1, !dbg !27
  %979 = call float @llvm.amdgcn.exp2.f32(float %978), !dbg !27
  %980 = fadd float %938, %979, !dbg !28
  %981 = insertelement <4 x float> %977, float %979, i64 1, !dbg !29
  %982 = extractelement <4 x float> %973, i64 2, !dbg !27
  %983 = call float @llvm.amdgcn.exp2.f32(float %982), !dbg !27
  %984 = fadd float %942, %983, !dbg !28
  %985 = insertelement <4 x float> %981, float %983, i64 2, !dbg !29
  %986 = extractelement <4 x float> %973, i64 3, !dbg !27
  %987 = call float @llvm.amdgcn.exp2.f32(float %986), !dbg !27
  %988 = fadd float %946, %987, !dbg !28
  %989 = insertelement <4 x float> %985, float %987, i64 3, !dbg !29
  %math.result321 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %989) #9, !dbg !29, !frisk.register_lifetimes !1
  %.cvtpack.f02003 = extractelement <4 x float> %math.result321, i32 0, !dbg !29
  %.cvtpack.f12004 = extractelement <4 x float> %math.result321, i32 1, !dbg !29
  %.cvtpack.f22005 = extractelement <4 x float> %math.result321, i32 2, !dbg !29
  %.cvtpack.f32006 = extractelement <4 x float> %math.result321, i32 3, !dbg !29
  %.cvtpack.p02007 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02003, float %.cvtpack.f12004), !dbg !29
  %.cvtpack.p12008 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22005, float %.cvtpack.f32006), !dbg !29
  %.cvtpack.v02009 = insertelement <2 x i32> poison, i32 %.cvtpack.p02007, i32 0, !dbg !29
  %.cvtpack.v12010 = insertelement <2 x i32> %.cvtpack.v02009, i32 %.cvtpack.p12008, i32 1, !dbg !29
  %990 = bitcast <2 x i32> %.cvtpack.v12010 to <4 x half>, !dbg !29
  %math.packed322 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %990) #9, !dbg !28, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %991 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %972, <4 x float> %949) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %992 = shufflevector <8 x half> %806, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %993 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %992, <4 x float> %951) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %994 = fsub <4 x float> %754, %738, !dbg !23
  %995 = extractelement <4 x float> %994, i64 0, !dbg !27
  %996 = call float @llvm.amdgcn.exp2.f32(float %995), !dbg !27
  %997 = fadd float %955, %996, !dbg !28
  %998 = insertelement <4 x float> poison, float %996, i64 0, !dbg !29
  %999 = extractelement <4 x float> %994, i64 1, !dbg !27
  %1000 = call float @llvm.amdgcn.exp2.f32(float %999), !dbg !27
  %1001 = fadd float %959, %1000, !dbg !28
  %1002 = insertelement <4 x float> %998, float %1000, i64 1, !dbg !29
  %1003 = extractelement <4 x float> %994, i64 2, !dbg !27
  %1004 = call float @llvm.amdgcn.exp2.f32(float %1003), !dbg !27
  %1005 = fadd float %963, %1004, !dbg !28
  %1006 = insertelement <4 x float> %1002, float %1004, i64 2, !dbg !29
  %1007 = extractelement <4 x float> %994, i64 3, !dbg !27
  %1008 = call float @llvm.amdgcn.exp2.f32(float %1007), !dbg !27
  %1009 = fadd float %967, %1008, !dbg !28
  %1010 = insertelement <4 x float> %1006, float %1008, i64 3, !dbg !29
  %math.result337 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1010) #9, !dbg !29, !frisk.register_lifetimes !1
  %.cvtpack.f02011 = extractelement <4 x float> %math.result337, i32 0, !dbg !29
  %.cvtpack.f12012 = extractelement <4 x float> %math.result337, i32 1, !dbg !29
  %.cvtpack.f22013 = extractelement <4 x float> %math.result337, i32 2, !dbg !29
  %.cvtpack.f32014 = extractelement <4 x float> %math.result337, i32 3, !dbg !29
  %.cvtpack.p02015 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02011, float %.cvtpack.f12012), !dbg !29
  %.cvtpack.p12016 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22013, float %.cvtpack.f32014), !dbg !29
  %.cvtpack.v02017 = insertelement <2 x i32> poison, i32 %.cvtpack.p02015, i32 0, !dbg !29
  %.cvtpack.v12018 = insertelement <2 x i32> %.cvtpack.v02017, i32 %.cvtpack.p12016, i32 1, !dbg !29
  %1011 = bitcast <2 x i32> %.cvtpack.v12018 to <4 x half>, !dbg !29
  %math.packed338 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1011) #9, !dbg !28, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %1012 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %972, <4 x float> %970) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1013 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %992, <4 x float> %971) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1014 = shufflevector <8 x half> %808, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1015 = fsub <4 x float> %725, %705, !dbg !23
  %1016 = extractelement <4 x float> %1015, i64 0, !dbg !27
  %1017 = call float @llvm.amdgcn.exp2.f32(float %1016), !dbg !27
  %1018 = fadd float %976, %1017, !dbg !28
  %1019 = insertelement <4 x float> poison, float %1017, i64 0, !dbg !29
  %1020 = extractelement <4 x float> %1015, i64 1, !dbg !27
  %1021 = call float @llvm.amdgcn.exp2.f32(float %1020), !dbg !27
  %1022 = fadd float %980, %1021, !dbg !28
  %1023 = insertelement <4 x float> %1019, float %1021, i64 1, !dbg !29
  %1024 = extractelement <4 x float> %1015, i64 2, !dbg !27
  %1025 = call float @llvm.amdgcn.exp2.f32(float %1024), !dbg !27
  %1026 = fadd float %984, %1025, !dbg !28
  %1027 = insertelement <4 x float> %1023, float %1025, i64 2, !dbg !29
  %1028 = extractelement <4 x float> %1015, i64 3, !dbg !27
  %1029 = call float @llvm.amdgcn.exp2.f32(float %1028), !dbg !27
  %1030 = fadd float %988, %1029, !dbg !28
  %1031 = insertelement <4 x float> %1027, float %1029, i64 3, !dbg !29
  %math.result323 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1031) #9, !dbg !29, !frisk.register_lifetimes !1
  %.cvtpack.f02019 = extractelement <4 x float> %math.result323, i32 0, !dbg !29
  %.cvtpack.f12020 = extractelement <4 x float> %math.result323, i32 1, !dbg !29
  %.cvtpack.f22021 = extractelement <4 x float> %math.result323, i32 2, !dbg !29
  %.cvtpack.f32022 = extractelement <4 x float> %math.result323, i32 3, !dbg !29
  %.cvtpack.p02023 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02019, float %.cvtpack.f12020), !dbg !29
  %.cvtpack.p12024 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22021, float %.cvtpack.f32022), !dbg !29
  %.cvtpack.v02025 = insertelement <2 x i32> poison, i32 %.cvtpack.p02023, i32 0, !dbg !29
  %.cvtpack.v12026 = insertelement <2 x i32> %.cvtpack.v02025, i32 %.cvtpack.p12024, i32 1, !dbg !29
  %1032 = bitcast <2 x i32> %.cvtpack.v12026 to <4 x half>, !dbg !29
  %math.packed324 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1032) #9, !dbg !28, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %1033 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1014, <4 x float> %991) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1034 = shufflevector <8 x half> %808, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1035 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1034, <4 x float> %993) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1036 = fsub <4 x float> %758, %738, !dbg !23
  %1037 = extractelement <4 x float> %1036, i64 0, !dbg !27
  %1038 = call float @llvm.amdgcn.exp2.f32(float %1037), !dbg !27
  %1039 = fadd float %997, %1038, !dbg !28
  %1040 = insertelement <4 x float> poison, float %1038, i64 0, !dbg !29
  %1041 = extractelement <4 x float> %1036, i64 1, !dbg !27
  %1042 = call float @llvm.amdgcn.exp2.f32(float %1041), !dbg !27
  %1043 = fadd float %1001, %1042, !dbg !28
  %1044 = insertelement <4 x float> %1040, float %1042, i64 1, !dbg !29
  %1045 = extractelement <4 x float> %1036, i64 2, !dbg !27
  %1046 = call float @llvm.amdgcn.exp2.f32(float %1045), !dbg !27
  %1047 = fadd float %1005, %1046, !dbg !28
  %1048 = insertelement <4 x float> %1044, float %1046, i64 2, !dbg !29
  %1049 = extractelement <4 x float> %1036, i64 3, !dbg !27
  %1050 = call float @llvm.amdgcn.exp2.f32(float %1049), !dbg !27
  %1051 = fadd float %1009, %1050, !dbg !28
  %1052 = insertelement <4 x float> %1048, float %1050, i64 3, !dbg !29
  %math.result339 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1052) #9, !dbg !29, !frisk.register_lifetimes !1
  %.cvtpack.f02027 = extractelement <4 x float> %math.result339, i32 0, !dbg !29
  %.cvtpack.f12028 = extractelement <4 x float> %math.result339, i32 1, !dbg !29
  %.cvtpack.f22029 = extractelement <4 x float> %math.result339, i32 2, !dbg !29
  %.cvtpack.f32030 = extractelement <4 x float> %math.result339, i32 3, !dbg !29
  %.cvtpack.p02031 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02027, float %.cvtpack.f12028), !dbg !29
  %.cvtpack.p12032 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22029, float %.cvtpack.f32030), !dbg !29
  %.cvtpack.v02033 = insertelement <2 x i32> poison, i32 %.cvtpack.p02031, i32 0, !dbg !29
  %.cvtpack.v12034 = insertelement <2 x i32> %.cvtpack.v02033, i32 %.cvtpack.p12032, i32 1, !dbg !29
  %1053 = bitcast <2 x i32> %.cvtpack.v12034 to <4 x half>, !dbg !29
  %math.packed340 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1053) #9, !dbg !28, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %1054 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed340, <4 x half> %1014, <4 x float> %1012) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1055 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed340, <4 x half> %1034, <4 x float> %1013) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1056 = shufflevector <8 x half> %810, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1057 = fsub <4 x float> %729, %705, !dbg !23
  %1058 = extractelement <4 x float> %1057, i64 0, !dbg !27
  %1059 = call float @llvm.amdgcn.exp2.f32(float %1058), !dbg !27
  %1060 = fadd float %1018, %1059, !dbg !28
  %1061 = insertelement <4 x float> poison, float %1059, i64 0, !dbg !29
  %1062 = extractelement <4 x float> %1057, i64 1, !dbg !27
  %1063 = call float @llvm.amdgcn.exp2.f32(float %1062), !dbg !27
  %1064 = fadd float %1022, %1063, !dbg !28
  %1065 = insertelement <4 x float> %1061, float %1063, i64 1, !dbg !29
  %1066 = extractelement <4 x float> %1057, i64 2, !dbg !27
  %1067 = call float @llvm.amdgcn.exp2.f32(float %1066), !dbg !27
  %1068 = fadd float %1026, %1067, !dbg !28
  %1069 = insertelement <4 x float> %1065, float %1067, i64 2, !dbg !29
  %1070 = extractelement <4 x float> %1057, i64 3, !dbg !27
  %1071 = call float @llvm.amdgcn.exp2.f32(float %1070), !dbg !27
  %1072 = fadd float %1030, %1071, !dbg !28
  %1073 = insertelement <4 x float> %1069, float %1071, i64 3, !dbg !29
  %math.result325 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1073) #9, !dbg !29, !frisk.register_lifetimes !1
  %.cvtpack.f02035 = extractelement <4 x float> %math.result325, i32 0, !dbg !29
  %.cvtpack.f12036 = extractelement <4 x float> %math.result325, i32 1, !dbg !29
  %.cvtpack.f22037 = extractelement <4 x float> %math.result325, i32 2, !dbg !29
  %.cvtpack.f32038 = extractelement <4 x float> %math.result325, i32 3, !dbg !29
  %.cvtpack.p02039 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02035, float %.cvtpack.f12036), !dbg !29
  %.cvtpack.p12040 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22037, float %.cvtpack.f32038), !dbg !29
  %.cvtpack.v02041 = insertelement <2 x i32> poison, i32 %.cvtpack.p02039, i32 0, !dbg !29
  %.cvtpack.v12042 = insertelement <2 x i32> %.cvtpack.v02041, i32 %.cvtpack.p12040, i32 1, !dbg !29
  %1074 = bitcast <2 x i32> %.cvtpack.v12042 to <4 x half>, !dbg !29
  %math.packed326 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1074) #9, !dbg !28, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %1075 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1056, <4 x float> %1033) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1076 = shufflevector <8 x half> %810, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1077 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1076, <4 x float> %1035) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1078 = fsub <4 x float> %762, %738, !dbg !23
  %1079 = extractelement <4 x float> %1078, i64 0, !dbg !27
  %1080 = call float @llvm.amdgcn.exp2.f32(float %1079), !dbg !27
  %1081 = fadd float %1039, %1080, !dbg !28
  %1082 = insertelement <4 x float> poison, float %1080, i64 0, !dbg !29
  %1083 = extractelement <4 x float> %1078, i64 1, !dbg !27
  %1084 = call float @llvm.amdgcn.exp2.f32(float %1083), !dbg !27
  %1085 = fadd float %1043, %1084, !dbg !28
  %1086 = insertelement <4 x float> %1082, float %1084, i64 1, !dbg !29
  %1087 = extractelement <4 x float> %1078, i64 2, !dbg !27
  %1088 = call float @llvm.amdgcn.exp2.f32(float %1087), !dbg !27
  %1089 = fadd float %1047, %1088, !dbg !28
  %1090 = insertelement <4 x float> %1086, float %1088, i64 2, !dbg !29
  %1091 = extractelement <4 x float> %1078, i64 3, !dbg !27
  %1092 = call float @llvm.amdgcn.exp2.f32(float %1091), !dbg !27
  %1093 = fadd float %1051, %1092, !dbg !28
  %1094 = insertelement <4 x float> %1090, float %1092, i64 3, !dbg !29
  %math.result341 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1094) #9, !dbg !29, !frisk.register_lifetimes !1
  %.cvtpack.f02043 = extractelement <4 x float> %math.result341, i32 0, !dbg !29
  %.cvtpack.f12044 = extractelement <4 x float> %math.result341, i32 1, !dbg !29
  %.cvtpack.f22045 = extractelement <4 x float> %math.result341, i32 2, !dbg !29
  %.cvtpack.f32046 = extractelement <4 x float> %math.result341, i32 3, !dbg !29
  %.cvtpack.p02047 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02043, float %.cvtpack.f12044), !dbg !29
  %.cvtpack.p12048 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22045, float %.cvtpack.f32046), !dbg !29
  %.cvtpack.v02049 = insertelement <2 x i32> poison, i32 %.cvtpack.p02047, i32 0, !dbg !29
  %.cvtpack.v12050 = insertelement <2 x i32> %.cvtpack.v02049, i32 %.cvtpack.p12048, i32 1, !dbg !29
  %1095 = bitcast <2 x i32> %.cvtpack.v12050 to <4 x half>, !dbg !29
  %math.packed342 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1095) #9, !dbg !28, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %1096 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed342, <4 x half> %1056, <4 x float> %1054) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1097 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed342, <4 x half> %1076, <4 x float> %1055) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1098 = bitcast <4 x i32> %811 to <8 x half>, !dbg !14
  %1099 = shufflevector <8 x half> %1098, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1100 = fsub <4 x float> %733, %705, !dbg !23
  %1101 = extractelement <4 x float> %1100, i64 0, !dbg !27
  %1102 = call float @llvm.amdgcn.exp2.f32(float %1101), !dbg !27
  %1103 = fadd float %1060, %1102, !dbg !28
  %1104 = insertelement <4 x float> poison, float %1102, i64 0, !dbg !29
  %1105 = extractelement <4 x float> %1100, i64 1, !dbg !27
  %1106 = call float @llvm.amdgcn.exp2.f32(float %1105), !dbg !27
  %1107 = fadd float %1064, %1106, !dbg !28
  %1108 = fadd float %1103, %1107, !dbg !28
  %1109 = insertelement <4 x float> %1104, float %1106, i64 1, !dbg !29
  %1110 = extractelement <4 x float> %1100, i64 2, !dbg !27
  %1111 = call float @llvm.amdgcn.exp2.f32(float %1110), !dbg !27
  %1112 = fadd float %1068, %1111, !dbg !28
  %1113 = insertelement <4 x float> %1109, float %1111, i64 2, !dbg !29
  %1114 = extractelement <4 x float> %1100, i64 3, !dbg !27
  %1115 = call float @llvm.amdgcn.exp2.f32(float %1114), !dbg !27
  %1116 = fadd float %1072, %1115, !dbg !28
  %1117 = fadd float %1112, %1116, !dbg !28
  %1118 = fadd float %1108, %1117, !dbg !28
  %1119 = fadd float %1118, %774, !dbg !28
  %1120 = insertelement <1 x float> poison, float %1119, i64 0, !dbg !6
  %1121 = insertvalue [2 x <1 x float>] poison, <1 x float> %1120, 0, !dbg !6
  %1122 = insertelement <4 x float> %1113, float %1115, i64 3, !dbg !29
  %math.result327 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1122) #9, !dbg !29, !frisk.register_lifetimes !1
  %.cvtpack.f02051 = extractelement <4 x float> %math.result327, i32 0, !dbg !29
  %.cvtpack.f12052 = extractelement <4 x float> %math.result327, i32 1, !dbg !29
  %.cvtpack.f22053 = extractelement <4 x float> %math.result327, i32 2, !dbg !29
  %.cvtpack.f32054 = extractelement <4 x float> %math.result327, i32 3, !dbg !29
  %.cvtpack.p02055 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02051, float %.cvtpack.f12052), !dbg !29
  %.cvtpack.p12056 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22053, float %.cvtpack.f32054), !dbg !29
  %.cvtpack.v02057 = insertelement <2 x i32> poison, i32 %.cvtpack.p02055, i32 0, !dbg !29
  %.cvtpack.v12058 = insertelement <2 x i32> %.cvtpack.v02057, i32 %.cvtpack.p12056, i32 1, !dbg !29
  %1123 = bitcast <2 x i32> %.cvtpack.v12058 to <4 x half>, !dbg !29
  %math.packed328 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1123) #9, !dbg !28, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %1124 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1099, <4 x float> %1075) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1125 = shufflevector <8 x half> %1098, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1126 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1125, <4 x float> %1077) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1127 = fsub <4 x float> %766, %738, !dbg !23
  %1128 = extractelement <4 x float> %1127, i64 0, !dbg !27
  %1129 = call float @llvm.amdgcn.exp2.f32(float %1128), !dbg !27
  %1130 = fadd float %1081, %1129, !dbg !28
  %1131 = insertelement <4 x float> poison, float %1129, i64 0, !dbg !29
  %1132 = extractelement <4 x float> %1127, i64 1, !dbg !27
  %1133 = call float @llvm.amdgcn.exp2.f32(float %1132), !dbg !27
  %1134 = fadd float %1085, %1133, !dbg !28
  %1135 = fadd float %1130, %1134, !dbg !28
  %1136 = insertelement <4 x float> %1131, float %1133, i64 1, !dbg !29
  %1137 = extractelement <4 x float> %1127, i64 2, !dbg !27
  %1138 = call float @llvm.amdgcn.exp2.f32(float %1137), !dbg !27
  %1139 = fadd float %1089, %1138, !dbg !28
  %1140 = insertelement <4 x float> %1136, float %1138, i64 2, !dbg !29
  %1141 = extractelement <4 x float> %1127, i64 3, !dbg !27
  %1142 = call float @llvm.amdgcn.exp2.f32(float %1141), !dbg !27
  %1143 = fadd float %1093, %1142, !dbg !28
  %1144 = fadd float %1139, %1143, !dbg !28
  %1145 = fadd float %1135, %1144, !dbg !28
  %1146 = fadd float %1145, %782, !dbg !28
  %1147 = insertelement <1 x float> poison, float %1146, i64 0, !dbg !6
  %1148 = insertvalue [2 x <1 x float>] %1121, <1 x float> %1147, 1, !dbg !6
  %1149 = insertelement <4 x float> %1140, float %1142, i64 3, !dbg !29
  %math.result343 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1149) #9, !dbg !29, !frisk.register_lifetimes !1
  %.cvtpack.f02059 = extractelement <4 x float> %math.result343, i32 0, !dbg !29
  %.cvtpack.f12060 = extractelement <4 x float> %math.result343, i32 1, !dbg !29
  %.cvtpack.f22061 = extractelement <4 x float> %math.result343, i32 2, !dbg !29
  %.cvtpack.f32062 = extractelement <4 x float> %math.result343, i32 3, !dbg !29
  %.cvtpack.p02063 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02059, float %.cvtpack.f12060), !dbg !29
  %.cvtpack.p12064 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22061, float %.cvtpack.f32062), !dbg !29
  %.cvtpack.v02065 = insertelement <2 x i32> poison, i32 %.cvtpack.p02063, i32 0, !dbg !29
  %.cvtpack.v12066 = insertelement <2 x i32> %.cvtpack.v02065, i32 %.cvtpack.p12064, i32 1, !dbg !29
  %1150 = bitcast <2 x i32> %.cvtpack.v12066 to <4 x half>, !dbg !29
  %math.packed344 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1150) #9, !dbg !28, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %1151 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed344, <4 x half> %1099, <4 x float> %1096) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1152 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed344, <4 x half> %1125, <4 x float> %1097) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !30
  %1153 = extractvalue [2 x <8 x float>] %132, 0, !dbg !30
  %1154 = shufflevector <8 x float> %1153, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %1155 = fmul <4 x float> %1154, %771, !dbg !30
  %1156 = shufflevector <8 x float> %1153, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %1157 = fmul <4 x float> %1156, %771, !dbg !30
  %1158 = extractvalue [2 x <8 x float>] %132, 1, !dbg !30
  %1159 = shufflevector <8 x float> %1158, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %1160 = fmul <4 x float> %1159, %779, !dbg !30
  %1161 = shufflevector <8 x float> %1158, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %1162 = fmul <4 x float> %1161, %779, !dbg !30
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1163 = or disjoint i64 %.idx14171419, 64, !dbg !18
  %1164 = add i64 %address.local863, %1163, !dbg !18
  %.tr1430 = trunc i64 %1164 to i32, !dbg !18
  %dma.buffer.offset1314 = shl i32 %.tr1430, 1, !dbg !18
  %1165 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %49, i32 %dma.buffer.offset1314, <4 x i32> %dma.buffer.rsrc1323) #10, !dbg !18
  %dma.buffer.offset1312 = or i32 %.tr1424, 16512, !dbg !18
  %1166 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %57, i32 %dma.buffer.offset1312, <4 x i32> %dma.buffer.rsrc1323) #10, !dbg !18
  %address.local940 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local880, !dbg !14
  %1167 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local940, i32 0) #0, !dbg !14
  %1168 = bitcast <4 x i32> %1167 to <8 x half>, !dbg !14
  %address.local941 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local888, !dbg !14
  %1169 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local941, i32 0) #0, !dbg !14
  %1170 = bitcast <4 x i32> %1169 to <8 x half>, !dbg !14
  %address.local942 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local896, !dbg !14
  %1171 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local942, i32 0) #0, !dbg !14
  %1172 = bitcast <4 x i32> %1171 to <8 x half>, !dbg !14
  %address.local943 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local904, !dbg !14
  %1173 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local943, i32 0) #0, !dbg !14
  %1174 = bitcast <4 x i32> %1173 to <8 x half>, !dbg !14
  %address.local944 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local912, !dbg !14
  %1175 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local944, i32 0) #0, !dbg !14
  %1176 = bitcast <4 x i32> %1175 to <8 x half>, !dbg !14
  %address.local945 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local920, !dbg !14
  %1177 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local945, i32 0) #0, !dbg !14
  %1178 = bitcast <4 x i32> %1177 to <8 x half>, !dbg !14
  %address.local946 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local928, !dbg !14
  %1179 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local946, i32 0) #0, !dbg !14
  %1180 = bitcast <4 x i32> %1179 to <8 x half>, !dbg !14
  %address.local947 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local936, !dbg !14
  %1181 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local947, i32 0) #0, !dbg !14
  %1182 = shufflevector <8 x half> %1168, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1183 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1182, <4 x float> %1155) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1184 = shufflevector <8 x half> %1168, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1185 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1184, <4 x float> %1157) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1186 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1182, <4 x float> %1160) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1187 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1184, <4 x float> %1162) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1188 = shufflevector <8 x half> %1170, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1189 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %1188, <4 x float> %1183) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1190 = shufflevector <8 x half> %1170, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1191 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %1190, <4 x float> %1185) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1192 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %1188, <4 x float> %1186) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1193 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %1190, <4 x float> %1187) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1194 = shufflevector <8 x half> %1172, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1195 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %1194, <4 x float> %1189) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1196 = shufflevector <8 x half> %1172, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1197 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %1196, <4 x float> %1191) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1198 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %1194, <4 x float> %1192) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1199 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %1196, <4 x float> %1193) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1200 = shufflevector <8 x half> %1174, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1201 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %1200, <4 x float> %1195) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1202 = shufflevector <8 x half> %1174, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1203 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %1202, <4 x float> %1197) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1204 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %1200, <4 x float> %1198) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1205 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %1202, <4 x float> %1199) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1206 = shufflevector <8 x half> %1176, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1207 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %1206, <4 x float> %1201) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1208 = shufflevector <8 x half> %1176, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1209 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %1208, <4 x float> %1203) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1210 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %1206, <4 x float> %1204) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1211 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %1208, <4 x float> %1205) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1212 = shufflevector <8 x half> %1178, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1213 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1212, <4 x float> %1207) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1214 = shufflevector <8 x half> %1178, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1215 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1214, <4 x float> %1209) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1216 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed340, <4 x half> %1212, <4 x float> %1210) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1217 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed340, <4 x half> %1214, <4 x float> %1211) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1218 = shufflevector <8 x half> %1180, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1219 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1218, <4 x float> %1213) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1220 = shufflevector <8 x half> %1180, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1221 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1220, <4 x float> %1215) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1222 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed342, <4 x half> %1218, <4 x float> %1216) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1223 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed342, <4 x half> %1220, <4 x float> %1217) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1224 = bitcast <4 x i32> %1181 to <8 x half>, !dbg !14
  %1225 = shufflevector <8 x half> %1224, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1226 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1225, <4 x float> %1219) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1227 = shufflevector <8 x half> %1224, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1228 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1227, <4 x float> %1221) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1229 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed344, <4 x half> %1225, <4 x float> %1222) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1230 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed344, <4 x half> %1227, <4 x float> %1223) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !30
  %1231 = extractvalue [2 x <8 x float>] %133, 0, !dbg !30
  %1232 = shufflevector <8 x float> %1231, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %1233 = fmul <4 x float> %1232, %771, !dbg !30
  %1234 = shufflevector <8 x float> %1231, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %1235 = fmul <4 x float> %1234, %771, !dbg !30
  %1236 = extractvalue [2 x <8 x float>] %133, 1, !dbg !30
  %1237 = shufflevector <8 x float> %1236, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %1238 = fmul <4 x float> %1237, %779, !dbg !30
  %1239 = shufflevector <8 x float> %1236, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %1240 = fmul <4 x float> %1239, %779, !dbg !30
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1241 = or disjoint i64 %.idx14171419, 96, !dbg !18
  %1242 = add i64 %address.local863, %1241, !dbg !18
  %.tr1434 = trunc i64 %1242 to i32, !dbg !18
  %dma.buffer.offset1310 = shl i32 %.tr1434, 1, !dbg !18
  %1243 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %62, i32 %dma.buffer.offset1310, <4 x i32> %dma.buffer.rsrc1323) #10, !dbg !18
  %dma.buffer.offset = or i32 %.tr1424, 16576, !dbg !18
  %1244 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %66, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc1323) #10, !dbg !18
  %1245 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local881, i32 0) #0, !dbg !14
  %1246 = bitcast <4 x i32> %1245 to <8 x half>, !dbg !14
  %1247 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local889, i32 0) #0, !dbg !14
  %1248 = bitcast <4 x i32> %1247 to <8 x half>, !dbg !14
  %1249 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local897, i32 0) #0, !dbg !14
  %1250 = bitcast <4 x i32> %1249 to <8 x half>, !dbg !14
  %1251 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local905, i32 0) #0, !dbg !14
  %1252 = bitcast <4 x i32> %1251 to <8 x half>, !dbg !14
  %1253 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local913, i32 0) #0, !dbg !14
  %1254 = bitcast <4 x i32> %1253 to <8 x half>, !dbg !14
  %1255 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local921, i32 0) #0, !dbg !14
  %1256 = bitcast <4 x i32> %1255 to <8 x half>, !dbg !14
  %1257 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local929, i32 0) #0, !dbg !14
  %1258 = bitcast <4 x i32> %1257 to <8 x half>, !dbg !14
  %1259 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local937, i32 0) #0, !dbg !14
  %1260 = shufflevector <8 x half> %1246, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1261 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1260, <4 x float> %1233) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1262 = shufflevector <8 x half> %1246, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1263 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1262, <4 x float> %1235) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1264 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1260, <4 x float> %1238) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1265 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1262, <4 x float> %1240) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1266 = shufflevector <8 x half> %1248, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1267 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %1266, <4 x float> %1261) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1268 = shufflevector <8 x half> %1248, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1269 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %1268, <4 x float> %1263) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1270 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %1266, <4 x float> %1264) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1271 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %1268, <4 x float> %1265) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1272 = shufflevector <8 x half> %1250, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1273 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %1272, <4 x float> %1267) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1274 = shufflevector <8 x half> %1250, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1275 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %1274, <4 x float> %1269) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1276 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %1272, <4 x float> %1270) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1277 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %1274, <4 x float> %1271) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1278 = shufflevector <8 x half> %1252, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1279 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %1278, <4 x float> %1273) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1280 = shufflevector <8 x half> %1252, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1281 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %1280, <4 x float> %1275) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1282 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %1278, <4 x float> %1276) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1283 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %1280, <4 x float> %1277) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1284 = shufflevector <8 x half> %1254, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1285 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %1284, <4 x float> %1279) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1286 = shufflevector <8 x half> %1254, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1287 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %1286, <4 x float> %1281) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1288 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %1284, <4 x float> %1282) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1289 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %1286, <4 x float> %1283) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1290 = shufflevector <8 x half> %1256, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1291 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1290, <4 x float> %1285) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1292 = shufflevector <8 x half> %1256, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1293 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1292, <4 x float> %1287) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1294 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed340, <4 x half> %1290, <4 x float> %1288) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1295 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed340, <4 x half> %1292, <4 x float> %1289) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1296 = shufflevector <8 x half> %1258, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1297 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1296, <4 x float> %1291) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1298 = shufflevector <8 x half> %1258, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1299 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1298, <4 x float> %1293) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1300 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed342, <4 x half> %1296, <4 x float> %1294) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1301 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed342, <4 x half> %1298, <4 x float> %1295) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1302 = bitcast <4 x i32> %1259 to <8 x half>, !dbg !14
  %1303 = shufflevector <8 x half> %1302, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1304 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1303, <4 x float> %1297) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1305 = shufflevector <8 x half> %1302, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1306 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1305, <4 x float> %1299) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1307 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed344, <4 x half> %1303, <4 x float> %1300) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1308 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed344, <4 x half> %1305, <4 x float> %1301) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !30
  %1309 = extractvalue [2 x <8 x float>] %134, 0, !dbg !30
  %1310 = shufflevector <8 x float> %1309, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %1311 = fmul <4 x float> %1310, %771, !dbg !30
  %1312 = shufflevector <8 x float> %1309, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %1313 = fmul <4 x float> %1312, %771, !dbg !30
  %1314 = extractvalue [2 x <8 x float>] %134, 1, !dbg !30
  %1315 = shufflevector <8 x float> %1314, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %1316 = fmul <4 x float> %1315, %779, !dbg !30
  %1317 = shufflevector <8 x float> %1314, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %1318 = fmul <4 x float> %1317, %779, !dbg !30
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1319 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local940, i32 0) #0, !dbg !14
  %1320 = bitcast <4 x i32> %1319 to <8 x half>, !dbg !14
  %1321 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local941, i32 0) #0, !dbg !14
  %1322 = bitcast <4 x i32> %1321 to <8 x half>, !dbg !14
  %1323 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local942, i32 0) #0, !dbg !14
  %1324 = bitcast <4 x i32> %1323 to <8 x half>, !dbg !14
  %1325 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local943, i32 0) #0, !dbg !14
  %1326 = bitcast <4 x i32> %1325 to <8 x half>, !dbg !14
  %1327 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local944, i32 0) #0, !dbg !14
  %1328 = bitcast <4 x i32> %1327 to <8 x half>, !dbg !14
  %1329 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local945, i32 0) #0, !dbg !14
  %1330 = bitcast <4 x i32> %1329 to <8 x half>, !dbg !14
  %1331 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local946, i32 0) #0, !dbg !14
  %1332 = bitcast <4 x i32> %1331 to <8 x half>, !dbg !14
  %1333 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local947, i32 0) #0, !dbg !14
  %1334 = shufflevector <8 x half> %1320, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1335 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1334, <4 x float> %1311) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1336 = shufflevector <8 x half> %1320, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1337 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1336, <4 x float> %1313) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1338 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1334, <4 x float> %1316) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1339 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1336, <4 x float> %1318) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1340 = shufflevector <8 x half> %1322, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1341 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %1340, <4 x float> %1335) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1342 = shufflevector <8 x half> %1322, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1343 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %1342, <4 x float> %1337) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1344 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %1340, <4 x float> %1338) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1345 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %1342, <4 x float> %1339) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1346 = shufflevector <8 x half> %1324, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1347 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %1346, <4 x float> %1341) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1348 = shufflevector <8 x half> %1324, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1349 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %1348, <4 x float> %1343) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1350 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %1346, <4 x float> %1344) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1351 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %1348, <4 x float> %1345) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1352 = shufflevector <8 x half> %1326, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1353 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %1352, <4 x float> %1347) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1354 = shufflevector <8 x half> %1326, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1355 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %1354, <4 x float> %1349) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1356 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %1352, <4 x float> %1350) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1357 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %1354, <4 x float> %1351) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1358 = shufflevector <8 x half> %1328, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1359 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %1358, <4 x float> %1353) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1360 = shufflevector <8 x half> %1328, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1361 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %1360, <4 x float> %1355) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1362 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %1358, <4 x float> %1356) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1363 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %1360, <4 x float> %1357) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1364 = shufflevector <8 x half> %1330, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1365 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1364, <4 x float> %1359) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1366 = shufflevector <8 x half> %1330, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1367 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1366, <4 x float> %1361) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1368 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed340, <4 x half> %1364, <4 x float> %1362) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1369 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed340, <4 x half> %1366, <4 x float> %1363) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1370 = shufflevector <8 x half> %1332, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1371 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1370, <4 x float> %1365) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1372 = shufflevector <8 x half> %1332, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1373 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1372, <4 x float> %1367) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1374 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed342, <4 x half> %1370, <4 x float> %1368) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1375 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed342, <4 x half> %1372, <4 x float> %1369) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1376 = bitcast <4 x i32> %1333 to <8 x half>, !dbg !14
  %1377 = shufflevector <8 x half> %1376, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1378 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1377, <4 x float> %1371) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1379 = shufflevector <8 x half> %1376, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1380 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1379, <4 x float> %1373) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1381 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed344, <4 x half> %1377, <4 x float> %1374) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1382 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed344, <4 x half> %1379, <4 x float> %1375) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !6
  %1383 = shufflevector <4 x float> %1124, <4 x float> %1126, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !6
  %1384 = insertvalue [2 x <8 x float>] poison, <8 x float> %1383, 0, !dbg !6
  %1385 = shufflevector <4 x float> %1151, <4 x float> %1152, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !6
  %1386 = insertvalue [2 x <8 x float>] %1384, <8 x float> %1385, 1, !dbg !6
  %1387 = insertvalue [2 x <1 x float>] poison, <1 x float> %697, 0, !dbg !6
  %1388 = insertvalue [2 x <1 x float>] %1387, <1 x float> %700, 1, !dbg !6
  %1389 = shufflevector <4 x float> %1226, <4 x float> %1228, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !6
  %1390 = insertvalue [2 x <8 x float>] poison, <8 x float> %1389, 0, !dbg !6
  %1391 = shufflevector <4 x float> %1229, <4 x float> %1230, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !6
  %1392 = insertvalue [2 x <8 x float>] %1390, <8 x float> %1391, 1, !dbg !6
  %1393 = shufflevector <4 x float> %1304, <4 x float> %1306, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !6
  %1394 = insertvalue [2 x <8 x float>] poison, <8 x float> %1393, 0, !dbg !6
  %1395 = shufflevector <4 x float> %1307, <4 x float> %1308, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !6
  %1396 = insertvalue [2 x <8 x float>] %1394, <8 x float> %1395, 1, !dbg !6
  %1397 = shufflevector <4 x float> %1378, <4 x float> %1380, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !6
  %1398 = insertvalue [2 x <8 x float>] poison, <8 x float> %1397, 0, !dbg !6
  %1399 = shufflevector <4 x float> %1381, <4 x float> %1382, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !6
  %1400 = insertvalue [2 x <8 x float>] %1398, <8 x float> %1399, 1, !dbg !6
  fence syncscope("workgroup") release, !dbg !6
  call void @llvm.amdgcn.s.barrier(), !dbg !6
  fence syncscope("workgroup") acquire, !dbg !6
  %1401 = add nuw nsw i64 %128, 1, !dbg !6
  br label %127, !dbg !6

1402:                                             ; preds = %127
  %1403 = extractvalue [2 x <1 x float>] %130, 1, !dbg !6
  %1404 = extractelement <1 x float> %1403, i64 0, !dbg !6
  %1405 = bitcast float %1404 to i32, !dbg !6
  %1406 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1405), !dbg !6
  %1407 = bitcast i32 %1406 to float, !dbg !6
  %1408 = fadd float %1404, %1407, !dbg !6
  %1409 = bitcast float %1408 to i32, !dbg !6
  %1410 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1409), !dbg !6
  %1411 = bitcast i32 %1410 to float, !dbg !6
  %1412 = fadd float %1408, %1411, !dbg !6
  %1413 = extractvalue [2 x <1 x float>] %130, 0, !dbg !6
  %1414 = extractelement <1 x float> %1413, i64 0, !dbg !6
  %1415 = bitcast float %1414 to i32, !dbg !6
  %1416 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1415), !dbg !6
  %1417 = bitcast i32 %1416 to float, !dbg !6
  %1418 = fadd float %1414, %1417, !dbg !6
  %1419 = bitcast float %1418 to i32, !dbg !6
  %1420 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1419), !dbg !6
  %1421 = bitcast i32 %1420 to float, !dbg !6
  %1422 = fadd float %1418, %1421, !dbg !6
  %1423 = and i32 %kdma.tid, 16, !dbg !6
  %.not = icmp eq i32 %1423, 0, !dbg !6
  %1424 = and i32 %kdma.tid, 32, !dbg !6
  %.not40 = icmp eq i32 %1424, 0, !dbg !6
  %1425 = extractvalue [2 x <8 x float>] %134, 0, !dbg !6
  %1426 = extractelement <8 x float> %1425, i64 0, !dbg !6
  %1427 = extractelement <8 x float> %1425, i64 1, !dbg !6
  %1428 = extractelement <8 x float> %1425, i64 2, !dbg !6
  %1429 = extractelement <8 x float> %1425, i64 3, !dbg !6
  %1430 = select i1 %.not, float %1427, float %1426, !dbg !6
  %1431 = bitcast float %1430 to i32, !dbg !6
  %1432 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1431), !dbg !6
  %1433 = bitcast i32 %1432 to float, !dbg !6
  %1434 = select i1 %.not, float %1426, float %1433, !dbg !6
  %1435 = select i1 %.not, float %1433, float %1427, !dbg !6
  %1436 = select i1 %.not, float %1429, float %1428, !dbg !6
  %1437 = bitcast float %1436 to i32, !dbg !6
  %1438 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1437), !dbg !6
  %1439 = bitcast i32 %1438 to float, !dbg !6
  %1440 = select i1 %.not, float %1428, float %1439, !dbg !6
  %1441 = select i1 %.not, float %1439, float %1429, !dbg !6
  %1442 = select i1 %.not40, float %1440, float %1434, !dbg !6
  %1443 = bitcast float %1442 to i32, !dbg !6
  %1444 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1443), !dbg !6
  %1445 = bitcast i32 %1444 to float, !dbg !6
  %1446 = select i1 %.not40, float %1434, float %1445, !dbg !6
  %1447 = select i1 %.not40, float %1445, float %1440, !dbg !6
  %1448 = select i1 %.not40, float %1441, float %1435, !dbg !6
  %1449 = bitcast float %1448 to i32, !dbg !6
  %1450 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1449), !dbg !6
  %1451 = bitcast i32 %1450 to float, !dbg !6
  %1452 = select i1 %.not40, float %1435, float %1451, !dbg !6
  %1453 = select i1 %.not40, float %1451, float %1441, !dbg !6
  %1454 = extractelement <8 x float> %1425, i64 4, !dbg !6
  %1455 = extractelement <8 x float> %1425, i64 5, !dbg !6
  %1456 = extractelement <8 x float> %1425, i64 6, !dbg !6
  %1457 = extractelement <8 x float> %1425, i64 7, !dbg !6
  %1458 = select i1 %.not, float %1455, float %1454, !dbg !6
  %1459 = bitcast float %1458 to i32, !dbg !6
  %1460 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1459), !dbg !6
  %1461 = bitcast i32 %1460 to float, !dbg !6
  %1462 = select i1 %.not, float %1454, float %1461, !dbg !6
  %1463 = select i1 %.not, float %1461, float %1455, !dbg !6
  %1464 = select i1 %.not, float %1457, float %1456, !dbg !6
  %1465 = bitcast float %1464 to i32, !dbg !6
  %1466 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1465), !dbg !6
  %1467 = bitcast i32 %1466 to float, !dbg !6
  %1468 = select i1 %.not, float %1456, float %1467, !dbg !6
  %1469 = select i1 %.not, float %1467, float %1457, !dbg !6
  %1470 = select i1 %.not40, float %1468, float %1462, !dbg !6
  %1471 = bitcast float %1470 to i32, !dbg !6
  %1472 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1471), !dbg !6
  %1473 = bitcast i32 %1472 to float, !dbg !6
  %1474 = select i1 %.not40, float %1462, float %1473, !dbg !6
  %1475 = select i1 %.not40, float %1473, float %1468, !dbg !6
  %1476 = select i1 %.not40, float %1469, float %1463, !dbg !6
  %1477 = bitcast float %1476 to i32, !dbg !6
  %1478 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1477), !dbg !6
  %1479 = bitcast i32 %1478 to float, !dbg !6
  %1480 = select i1 %.not40, float %1463, float %1479, !dbg !6
  %1481 = select i1 %.not40, float %1479, float %1469, !dbg !6
  %1482 = extractvalue [2 x <8 x float>] %134, 1, !dbg !6
  %1483 = extractelement <8 x float> %1482, i64 0, !dbg !6
  %1484 = extractelement <8 x float> %1482, i64 1, !dbg !6
  %1485 = extractelement <8 x float> %1482, i64 2, !dbg !6
  %1486 = extractelement <8 x float> %1482, i64 3, !dbg !6
  %1487 = select i1 %.not, float %1484, float %1483, !dbg !6
  %1488 = bitcast float %1487 to i32, !dbg !6
  %1489 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1488), !dbg !6
  %1490 = bitcast i32 %1489 to float, !dbg !6
  %1491 = select i1 %.not, float %1483, float %1490, !dbg !6
  %1492 = select i1 %.not, float %1490, float %1484, !dbg !6
  %1493 = select i1 %.not, float %1486, float %1485, !dbg !6
  %1494 = bitcast float %1493 to i32, !dbg !6
  %1495 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1494), !dbg !6
  %1496 = bitcast i32 %1495 to float, !dbg !6
  %1497 = select i1 %.not, float %1485, float %1496, !dbg !6
  %1498 = select i1 %.not, float %1496, float %1486, !dbg !6
  %1499 = select i1 %.not40, float %1497, float %1491, !dbg !6
  %1500 = bitcast float %1499 to i32, !dbg !6
  %1501 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1500), !dbg !6
  %1502 = bitcast i32 %1501 to float, !dbg !6
  %1503 = select i1 %.not40, float %1491, float %1502, !dbg !6
  %1504 = select i1 %.not40, float %1502, float %1497, !dbg !6
  %1505 = select i1 %.not40, float %1498, float %1492, !dbg !6
  %1506 = bitcast float %1505 to i32, !dbg !6
  %1507 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1506), !dbg !6
  %1508 = bitcast i32 %1507 to float, !dbg !6
  %1509 = select i1 %.not40, float %1492, float %1508, !dbg !6
  %1510 = select i1 %.not40, float %1508, float %1498, !dbg !6
  %1511 = extractelement <8 x float> %1482, i64 4, !dbg !6
  %1512 = extractelement <8 x float> %1482, i64 5, !dbg !6
  %1513 = extractelement <8 x float> %1482, i64 6, !dbg !6
  %1514 = extractelement <8 x float> %1482, i64 7, !dbg !6
  %1515 = select i1 %.not, float %1512, float %1511, !dbg !6
  %1516 = bitcast float %1515 to i32, !dbg !6
  %1517 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1516), !dbg !6
  %1518 = bitcast i32 %1517 to float, !dbg !6
  %1519 = select i1 %.not, float %1511, float %1518, !dbg !6
  %1520 = select i1 %.not, float %1518, float %1512, !dbg !6
  %1521 = select i1 %.not, float %1514, float %1513, !dbg !6
  %1522 = bitcast float %1521 to i32, !dbg !6
  %1523 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1522), !dbg !6
  %1524 = bitcast i32 %1523 to float, !dbg !6
  %1525 = select i1 %.not, float %1513, float %1524, !dbg !6
  %1526 = select i1 %.not, float %1524, float %1514, !dbg !6
  %1527 = select i1 %.not40, float %1525, float %1519, !dbg !6
  %1528 = bitcast float %1527 to i32, !dbg !6
  %1529 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1528), !dbg !6
  %1530 = bitcast i32 %1529 to float, !dbg !6
  %1531 = select i1 %.not40, float %1519, float %1530, !dbg !6
  %1532 = select i1 %.not40, float %1530, float %1525, !dbg !6
  %1533 = select i1 %.not40, float %1526, float %1520, !dbg !6
  %1534 = bitcast float %1533 to i32, !dbg !6
  %1535 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1534), !dbg !6
  %1536 = bitcast i32 %1535 to float, !dbg !6
  %1537 = select i1 %.not40, float %1520, float %1536, !dbg !6
  %1538 = select i1 %.not40, float %1536, float %1526, !dbg !6
  %1539 = extractvalue [2 x <8 x float>] %133, 0, !dbg !6
  %1540 = extractelement <8 x float> %1539, i64 0, !dbg !6
  %1541 = extractelement <8 x float> %1539, i64 1, !dbg !6
  %1542 = extractelement <8 x float> %1539, i64 2, !dbg !6
  %1543 = extractelement <8 x float> %1539, i64 3, !dbg !6
  %1544 = select i1 %.not, float %1541, float %1540, !dbg !6
  %1545 = bitcast float %1544 to i32, !dbg !6
  %1546 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1545), !dbg !6
  %1547 = bitcast i32 %1546 to float, !dbg !6
  %1548 = select i1 %.not, float %1540, float %1547, !dbg !6
  %1549 = select i1 %.not, float %1547, float %1541, !dbg !6
  %1550 = select i1 %.not, float %1543, float %1542, !dbg !6
  %1551 = bitcast float %1550 to i32, !dbg !6
  %1552 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1551), !dbg !6
  %1553 = bitcast i32 %1552 to float, !dbg !6
  %1554 = select i1 %.not, float %1542, float %1553, !dbg !6
  %1555 = select i1 %.not, float %1553, float %1543, !dbg !6
  %1556 = select i1 %.not40, float %1554, float %1548, !dbg !6
  %1557 = bitcast float %1556 to i32, !dbg !6
  %1558 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1557), !dbg !6
  %1559 = bitcast i32 %1558 to float, !dbg !6
  %1560 = select i1 %.not40, float %1548, float %1559, !dbg !6
  %1561 = select i1 %.not40, float %1559, float %1554, !dbg !6
  %1562 = select i1 %.not40, float %1555, float %1549, !dbg !6
  %1563 = bitcast float %1562 to i32, !dbg !6
  %1564 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1563), !dbg !6
  %1565 = bitcast i32 %1564 to float, !dbg !6
  %1566 = select i1 %.not40, float %1549, float %1565, !dbg !6
  %1567 = select i1 %.not40, float %1565, float %1555, !dbg !6
  %1568 = extractelement <8 x float> %1539, i64 4, !dbg !6
  %1569 = extractelement <8 x float> %1539, i64 5, !dbg !6
  %1570 = extractelement <8 x float> %1539, i64 6, !dbg !6
  %1571 = extractelement <8 x float> %1539, i64 7, !dbg !6
  %1572 = select i1 %.not, float %1569, float %1568, !dbg !6
  %1573 = bitcast float %1572 to i32, !dbg !6
  %1574 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1573), !dbg !6
  %1575 = bitcast i32 %1574 to float, !dbg !6
  %1576 = select i1 %.not, float %1568, float %1575, !dbg !6
  %1577 = select i1 %.not, float %1575, float %1569, !dbg !6
  %1578 = select i1 %.not, float %1571, float %1570, !dbg !6
  %1579 = bitcast float %1578 to i32, !dbg !6
  %1580 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1579), !dbg !6
  %1581 = bitcast i32 %1580 to float, !dbg !6
  %1582 = select i1 %.not, float %1570, float %1581, !dbg !6
  %1583 = select i1 %.not, float %1581, float %1571, !dbg !6
  %1584 = select i1 %.not40, float %1582, float %1576, !dbg !6
  %1585 = bitcast float %1584 to i32, !dbg !6
  %1586 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1585), !dbg !6
  %1587 = bitcast i32 %1586 to float, !dbg !6
  %1588 = select i1 %.not40, float %1576, float %1587, !dbg !6
  %1589 = select i1 %.not40, float %1587, float %1582, !dbg !6
  %1590 = select i1 %.not40, float %1583, float %1577, !dbg !6
  %1591 = bitcast float %1590 to i32, !dbg !6
  %1592 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1591), !dbg !6
  %1593 = bitcast i32 %1592 to float, !dbg !6
  %1594 = select i1 %.not40, float %1577, float %1593, !dbg !6
  %1595 = select i1 %.not40, float %1593, float %1583, !dbg !6
  %1596 = extractvalue [2 x <8 x float>] %133, 1, !dbg !6
  %1597 = extractelement <8 x float> %1596, i64 0, !dbg !6
  %1598 = extractelement <8 x float> %1596, i64 1, !dbg !6
  %1599 = extractelement <8 x float> %1596, i64 2, !dbg !6
  %1600 = extractelement <8 x float> %1596, i64 3, !dbg !6
  %1601 = select i1 %.not, float %1598, float %1597, !dbg !6
  %1602 = bitcast float %1601 to i32, !dbg !6
  %1603 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1602), !dbg !6
  %1604 = bitcast i32 %1603 to float, !dbg !6
  %1605 = select i1 %.not, float %1597, float %1604, !dbg !6
  %1606 = select i1 %.not, float %1604, float %1598, !dbg !6
  %1607 = select i1 %.not, float %1600, float %1599, !dbg !6
  %1608 = bitcast float %1607 to i32, !dbg !6
  %1609 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1608), !dbg !6
  %1610 = bitcast i32 %1609 to float, !dbg !6
  %1611 = select i1 %.not, float %1599, float %1610, !dbg !6
  %1612 = select i1 %.not, float %1610, float %1600, !dbg !6
  %1613 = select i1 %.not40, float %1611, float %1605, !dbg !6
  %1614 = bitcast float %1613 to i32, !dbg !6
  %1615 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1614), !dbg !6
  %1616 = bitcast i32 %1615 to float, !dbg !6
  %1617 = select i1 %.not40, float %1605, float %1616, !dbg !6
  %1618 = select i1 %.not40, float %1616, float %1611, !dbg !6
  %1619 = select i1 %.not40, float %1612, float %1606, !dbg !6
  %1620 = bitcast float %1619 to i32, !dbg !6
  %1621 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1620), !dbg !6
  %1622 = bitcast i32 %1621 to float, !dbg !6
  %1623 = select i1 %.not40, float %1606, float %1622, !dbg !6
  %1624 = select i1 %.not40, float %1622, float %1612, !dbg !6
  %1625 = extractelement <8 x float> %1596, i64 4, !dbg !6
  %1626 = extractelement <8 x float> %1596, i64 5, !dbg !6
  %1627 = extractelement <8 x float> %1596, i64 6, !dbg !6
  %1628 = extractelement <8 x float> %1596, i64 7, !dbg !6
  %1629 = select i1 %.not, float %1626, float %1625, !dbg !6
  %1630 = bitcast float %1629 to i32, !dbg !6
  %1631 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1630), !dbg !6
  %1632 = bitcast i32 %1631 to float, !dbg !6
  %1633 = select i1 %.not, float %1625, float %1632, !dbg !6
  %1634 = select i1 %.not, float %1632, float %1626, !dbg !6
  %1635 = select i1 %.not, float %1628, float %1627, !dbg !6
  %1636 = bitcast float %1635 to i32, !dbg !6
  %1637 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1636), !dbg !6
  %1638 = bitcast i32 %1637 to float, !dbg !6
  %1639 = select i1 %.not, float %1627, float %1638, !dbg !6
  %1640 = select i1 %.not, float %1638, float %1628, !dbg !6
  %1641 = select i1 %.not40, float %1639, float %1633, !dbg !6
  %1642 = bitcast float %1641 to i32, !dbg !6
  %1643 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1642), !dbg !6
  %1644 = bitcast i32 %1643 to float, !dbg !6
  %1645 = select i1 %.not40, float %1633, float %1644, !dbg !6
  %1646 = select i1 %.not40, float %1644, float %1639, !dbg !6
  %1647 = select i1 %.not40, float %1640, float %1634, !dbg !6
  %1648 = bitcast float %1647 to i32, !dbg !6
  %1649 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1648), !dbg !6
  %1650 = bitcast i32 %1649 to float, !dbg !6
  %1651 = select i1 %.not40, float %1634, float %1650, !dbg !6
  %1652 = select i1 %.not40, float %1650, float %1640, !dbg !6
  %1653 = extractvalue [2 x <8 x float>] %132, 0, !dbg !6
  %1654 = extractelement <8 x float> %1653, i64 0, !dbg !6
  %1655 = extractelement <8 x float> %1653, i64 1, !dbg !6
  %1656 = extractelement <8 x float> %1653, i64 2, !dbg !6
  %1657 = extractelement <8 x float> %1653, i64 3, !dbg !6
  %1658 = select i1 %.not, float %1655, float %1654, !dbg !6
  %1659 = bitcast float %1658 to i32, !dbg !6
  %1660 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1659), !dbg !6
  %1661 = bitcast i32 %1660 to float, !dbg !6
  %1662 = select i1 %.not, float %1654, float %1661, !dbg !6
  %1663 = select i1 %.not, float %1661, float %1655, !dbg !6
  %1664 = select i1 %.not, float %1657, float %1656, !dbg !6
  %1665 = bitcast float %1664 to i32, !dbg !6
  %1666 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1665), !dbg !6
  %1667 = bitcast i32 %1666 to float, !dbg !6
  %1668 = select i1 %.not, float %1656, float %1667, !dbg !6
  %1669 = select i1 %.not, float %1667, float %1657, !dbg !6
  %1670 = select i1 %.not40, float %1668, float %1662, !dbg !6
  %1671 = bitcast float %1670 to i32, !dbg !6
  %1672 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1671), !dbg !6
  %1673 = bitcast i32 %1672 to float, !dbg !6
  %1674 = select i1 %.not40, float %1662, float %1673, !dbg !6
  %1675 = select i1 %.not40, float %1673, float %1668, !dbg !6
  %1676 = select i1 %.not40, float %1669, float %1663, !dbg !6
  %1677 = bitcast float %1676 to i32, !dbg !6
  %1678 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1677), !dbg !6
  %1679 = bitcast i32 %1678 to float, !dbg !6
  %1680 = select i1 %.not40, float %1663, float %1679, !dbg !6
  %1681 = select i1 %.not40, float %1679, float %1669, !dbg !6
  %1682 = extractelement <8 x float> %1653, i64 4, !dbg !6
  %1683 = extractelement <8 x float> %1653, i64 5, !dbg !6
  %1684 = extractelement <8 x float> %1653, i64 6, !dbg !6
  %1685 = extractelement <8 x float> %1653, i64 7, !dbg !6
  %1686 = select i1 %.not, float %1683, float %1682, !dbg !6
  %1687 = bitcast float %1686 to i32, !dbg !6
  %1688 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1687), !dbg !6
  %1689 = bitcast i32 %1688 to float, !dbg !6
  %1690 = select i1 %.not, float %1682, float %1689, !dbg !6
  %1691 = select i1 %.not, float %1689, float %1683, !dbg !6
  %1692 = select i1 %.not, float %1685, float %1684, !dbg !6
  %1693 = bitcast float %1692 to i32, !dbg !6
  %1694 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1693), !dbg !6
  %1695 = bitcast i32 %1694 to float, !dbg !6
  %1696 = select i1 %.not, float %1684, float %1695, !dbg !6
  %1697 = select i1 %.not, float %1695, float %1685, !dbg !6
  %1698 = select i1 %.not40, float %1696, float %1690, !dbg !6
  %1699 = bitcast float %1698 to i32, !dbg !6
  %1700 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1699), !dbg !6
  %1701 = bitcast i32 %1700 to float, !dbg !6
  %1702 = select i1 %.not40, float %1690, float %1701, !dbg !6
  %1703 = select i1 %.not40, float %1701, float %1696, !dbg !6
  %1704 = select i1 %.not40, float %1697, float %1691, !dbg !6
  %1705 = bitcast float %1704 to i32, !dbg !6
  %1706 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1705), !dbg !6
  %1707 = bitcast i32 %1706 to float, !dbg !6
  %1708 = select i1 %.not40, float %1691, float %1707, !dbg !6
  %1709 = select i1 %.not40, float %1707, float %1697, !dbg !6
  %1710 = extractvalue [2 x <8 x float>] %132, 1, !dbg !6
  %1711 = extractelement <8 x float> %1710, i64 0, !dbg !6
  %1712 = extractelement <8 x float> %1710, i64 1, !dbg !6
  %1713 = extractelement <8 x float> %1710, i64 2, !dbg !6
  %1714 = extractelement <8 x float> %1710, i64 3, !dbg !6
  %1715 = select i1 %.not, float %1712, float %1711, !dbg !6
  %1716 = bitcast float %1715 to i32, !dbg !6
  %1717 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1716), !dbg !6
  %1718 = bitcast i32 %1717 to float, !dbg !6
  %1719 = select i1 %.not, float %1711, float %1718, !dbg !6
  %1720 = select i1 %.not, float %1718, float %1712, !dbg !6
  %1721 = select i1 %.not, float %1714, float %1713, !dbg !6
  %1722 = bitcast float %1721 to i32, !dbg !6
  %1723 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1722), !dbg !6
  %1724 = bitcast i32 %1723 to float, !dbg !6
  %1725 = select i1 %.not, float %1713, float %1724, !dbg !6
  %1726 = select i1 %.not, float %1724, float %1714, !dbg !6
  %1727 = select i1 %.not40, float %1725, float %1719, !dbg !6
  %1728 = bitcast float %1727 to i32, !dbg !6
  %1729 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1728), !dbg !6
  %1730 = bitcast i32 %1729 to float, !dbg !6
  %1731 = select i1 %.not40, float %1719, float %1730, !dbg !6
  %1732 = select i1 %.not40, float %1730, float %1725, !dbg !6
  %1733 = select i1 %.not40, float %1726, float %1720, !dbg !6
  %1734 = bitcast float %1733 to i32, !dbg !6
  %1735 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1734), !dbg !6
  %1736 = bitcast i32 %1735 to float, !dbg !6
  %1737 = select i1 %.not40, float %1720, float %1736, !dbg !6
  %1738 = select i1 %.not40, float %1736, float %1726, !dbg !6
  %1739 = extractelement <8 x float> %1710, i64 4, !dbg !6
  %1740 = extractelement <8 x float> %1710, i64 5, !dbg !6
  %1741 = extractelement <8 x float> %1710, i64 6, !dbg !6
  %1742 = extractelement <8 x float> %1710, i64 7, !dbg !6
  %1743 = select i1 %.not, float %1740, float %1739, !dbg !6
  %1744 = bitcast float %1743 to i32, !dbg !6
  %1745 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1744), !dbg !6
  %1746 = bitcast i32 %1745 to float, !dbg !6
  %1747 = select i1 %.not, float %1739, float %1746, !dbg !6
  %1748 = select i1 %.not, float %1746, float %1740, !dbg !6
  %1749 = select i1 %.not, float %1742, float %1741, !dbg !6
  %1750 = bitcast float %1749 to i32, !dbg !6
  %1751 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1750), !dbg !6
  %1752 = bitcast i32 %1751 to float, !dbg !6
  %1753 = select i1 %.not, float %1741, float %1752, !dbg !6
  %1754 = select i1 %.not, float %1752, float %1742, !dbg !6
  %1755 = select i1 %.not40, float %1753, float %1747, !dbg !6
  %1756 = bitcast float %1755 to i32, !dbg !6
  %1757 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1756), !dbg !6
  %1758 = bitcast i32 %1757 to float, !dbg !6
  %1759 = select i1 %.not40, float %1747, float %1758, !dbg !6
  %1760 = select i1 %.not40, float %1758, float %1753, !dbg !6
  %1761 = select i1 %.not40, float %1754, float %1748, !dbg !6
  %1762 = bitcast float %1761 to i32, !dbg !6
  %1763 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1762), !dbg !6
  %1764 = bitcast i32 %1763 to float, !dbg !6
  %1765 = select i1 %.not40, float %1748, float %1764, !dbg !6
  %1766 = select i1 %.not40, float %1764, float %1754, !dbg !6
  %1767 = extractvalue [2 x <8 x float>] %129, 0, !dbg !6
  %1768 = extractelement <8 x float> %1767, i64 0, !dbg !6
  %1769 = extractelement <8 x float> %1767, i64 1, !dbg !6
  %1770 = extractelement <8 x float> %1767, i64 2, !dbg !6
  %1771 = extractelement <8 x float> %1767, i64 3, !dbg !6
  %1772 = select i1 %.not, float %1769, float %1768, !dbg !6
  %1773 = bitcast float %1772 to i32, !dbg !6
  %1774 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1773), !dbg !6
  %1775 = bitcast i32 %1774 to float, !dbg !6
  %1776 = select i1 %.not, float %1768, float %1775, !dbg !6
  %1777 = select i1 %.not, float %1775, float %1769, !dbg !6
  %1778 = select i1 %.not, float %1771, float %1770, !dbg !6
  %1779 = bitcast float %1778 to i32, !dbg !6
  %1780 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1779), !dbg !6
  %1781 = bitcast i32 %1780 to float, !dbg !6
  %1782 = select i1 %.not, float %1770, float %1781, !dbg !6
  %1783 = select i1 %.not, float %1781, float %1771, !dbg !6
  %1784 = select i1 %.not40, float %1782, float %1776, !dbg !6
  %1785 = bitcast float %1784 to i32, !dbg !6
  %1786 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1785), !dbg !6
  %1787 = select i1 %.not40, float %1783, float %1777, !dbg !6
  %1788 = bitcast float %1787 to i32, !dbg !6
  %1789 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1788), !dbg !6
  %1790 = extractelement <8 x float> %1767, i64 4, !dbg !6
  %1791 = extractelement <8 x float> %1767, i64 5, !dbg !6
  %1792 = extractelement <8 x float> %1767, i64 6, !dbg !6
  %1793 = extractelement <8 x float> %1767, i64 7, !dbg !6
  %1794 = select i1 %.not, float %1791, float %1790, !dbg !6
  %1795 = bitcast float %1794 to i32, !dbg !6
  %1796 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1795), !dbg !6
  %1797 = bitcast i32 %1796 to float, !dbg !6
  %1798 = select i1 %.not, float %1790, float %1797, !dbg !6
  %1799 = select i1 %.not, float %1797, float %1791, !dbg !6
  %1800 = select i1 %.not, float %1793, float %1792, !dbg !6
  %1801 = bitcast float %1800 to i32, !dbg !6
  %1802 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1801), !dbg !6
  %1803 = bitcast i32 %1802 to float, !dbg !6
  %1804 = select i1 %.not, float %1792, float %1803, !dbg !6
  %1805 = select i1 %.not, float %1803, float %1793, !dbg !6
  %1806 = select i1 %.not40, float %1804, float %1798, !dbg !6
  %1807 = bitcast float %1806 to i32, !dbg !6
  %1808 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1807), !dbg !6
  %1809 = select i1 %.not40, float %1805, float %1799, !dbg !6
  %1810 = bitcast float %1809 to i32, !dbg !6
  %1811 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1810), !dbg !6
  %1812 = extractvalue [2 x <8 x float>] %129, 1, !dbg !6
  %1813 = extractelement <8 x float> %1812, i64 0, !dbg !6
  %1814 = extractelement <8 x float> %1812, i64 1, !dbg !6
  %1815 = extractelement <8 x float> %1812, i64 2, !dbg !6
  %1816 = extractelement <8 x float> %1812, i64 3, !dbg !6
  %1817 = select i1 %.not, float %1814, float %1813, !dbg !6
  %1818 = bitcast float %1817 to i32, !dbg !6
  %1819 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1818), !dbg !6
  %1820 = bitcast i32 %1819 to float, !dbg !6
  %1821 = select i1 %.not, float %1813, float %1820, !dbg !6
  %1822 = select i1 %.not, float %1820, float %1814, !dbg !6
  %1823 = select i1 %.not, float %1816, float %1815, !dbg !6
  %1824 = bitcast float %1823 to i32, !dbg !6
  %1825 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1824), !dbg !6
  %1826 = bitcast i32 %1825 to float, !dbg !6
  %1827 = select i1 %.not, float %1815, float %1826, !dbg !6
  %1828 = select i1 %.not, float %1826, float %1816, !dbg !6
  %1829 = select i1 %.not40, float %1827, float %1821, !dbg !6
  %1830 = bitcast float %1829 to i32, !dbg !6
  %1831 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1830), !dbg !6
  %1832 = select i1 %.not40, float %1828, float %1822, !dbg !6
  %1833 = bitcast float %1832 to i32, !dbg !6
  %1834 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1833), !dbg !6
  %1835 = extractelement <8 x float> %1812, i64 4, !dbg !6
  %1836 = extractelement <8 x float> %1812, i64 5, !dbg !6
  %1837 = extractelement <8 x float> %1812, i64 6, !dbg !6
  %1838 = extractelement <8 x float> %1812, i64 7, !dbg !6
  %1839 = select i1 %.not, float %1836, float %1835, !dbg !6
  %1840 = bitcast float %1839 to i32, !dbg !6
  %1841 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1840), !dbg !6
  %1842 = bitcast i32 %1841 to float, !dbg !6
  %1843 = select i1 %.not, float %1835, float %1842, !dbg !6
  %1844 = select i1 %.not, float %1842, float %1836, !dbg !6
  %1845 = select i1 %.not, float %1838, float %1837, !dbg !6
  %1846 = bitcast float %1845 to i32, !dbg !6
  %1847 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1846), !dbg !6
  %1848 = bitcast i32 %1847 to float, !dbg !6
  %1849 = select i1 %.not, float %1837, float %1848, !dbg !6
  %1850 = select i1 %.not, float %1848, float %1838, !dbg !6
  %1851 = select i1 %.not40, float %1849, float %1843, !dbg !6
  %1852 = bitcast float %1851 to i32, !dbg !6
  %1853 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1852), !dbg !6
  %1854 = select i1 %.not40, float %1850, float %1844, !dbg !6
  %1855 = bitcast float %1854 to i32, !dbg !6
  %1856 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1855), !dbg !6
  %1857 = insertelement <4 x float> poison, float %1422, i64 0, !dbg !20
  %1858 = fdiv arcp <4 x float> <float 0x3FF0000000000000, float poison, float poison, float poison>, %1857, !dbg !20
  %1859 = shufflevector <4 x float> %1858, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !20
  %1860 = insertelement <4 x float> poison, float %1412, i64 0, !dbg !20
  %1861 = fdiv arcp <4 x float> <float 0x3FF0000000000000, float poison, float poison, float poison>, %1860, !dbg !20
  %1862 = shufflevector <4 x float> %1861, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !20
  %1863 = bitcast i32 %1853 to float, !dbg !6
  %1864 = select i1 %.not40, float %1843, float %1863, !dbg !6
  %1865 = insertelement <4 x float> poison, float %1864, i64 0, !dbg !20
  %1866 = bitcast i32 %1856 to float, !dbg !6
  %1867 = select i1 %.not40, float %1844, float %1866, !dbg !6
  %1868 = insertelement <4 x float> %1865, float %1867, i64 1, !dbg !20
  %1869 = select i1 %.not40, float %1863, float %1849, !dbg !6
  %1870 = insertelement <4 x float> %1868, float %1869, i64 2, !dbg !20
  %1871 = select i1 %.not40, float %1866, float %1850, !dbg !6
  %1872 = insertelement <4 x float> %1870, float %1871, i64 3, !dbg !20
  %1873 = fmul <4 x float> %1872, %1862, !dbg !20
  %1874 = fptrunc <4 x float> %1873 to <4 x half>, !dbg !21
  %1875 = extractelement <4 x half> %1874, i64 3, !dbg !21
  %1876 = extractelement <4 x half> %1874, i64 2, !dbg !21
  %1877 = extractelement <4 x half> %1874, i64 1, !dbg !21
  %1878 = extractelement <4 x half> %1874, i64 0, !dbg !21
  %1879 = bitcast i32 %1831 to float, !dbg !6
  %1880 = select i1 %.not40, float %1821, float %1879, !dbg !6
  %1881 = insertelement <4 x float> poison, float %1880, i64 0, !dbg !20
  %1882 = bitcast i32 %1834 to float, !dbg !6
  %1883 = select i1 %.not40, float %1822, float %1882, !dbg !6
  %1884 = insertelement <4 x float> %1881, float %1883, i64 1, !dbg !20
  %1885 = select i1 %.not40, float %1879, float %1827, !dbg !6
  %1886 = insertelement <4 x float> %1884, float %1885, i64 2, !dbg !20
  %1887 = select i1 %.not40, float %1882, float %1828, !dbg !6
  %1888 = insertelement <4 x float> %1886, float %1887, i64 3, !dbg !20
  %1889 = fmul <4 x float> %1888, %1862, !dbg !20
  %1890 = fptrunc <4 x float> %1889 to <4 x half>, !dbg !21
  %1891 = extractelement <4 x half> %1890, i64 3, !dbg !21
  %1892 = extractelement <4 x half> %1890, i64 2, !dbg !21
  %1893 = extractelement <4 x half> %1890, i64 1, !dbg !21
  %1894 = extractelement <4 x half> %1890, i64 0, !dbg !21
  %1895 = bitcast i32 %1808 to float, !dbg !6
  %1896 = select i1 %.not40, float %1798, float %1895, !dbg !6
  %1897 = insertelement <4 x float> poison, float %1896, i64 0, !dbg !20
  %1898 = bitcast i32 %1811 to float, !dbg !6
  %1899 = select i1 %.not40, float %1799, float %1898, !dbg !6
  %1900 = insertelement <4 x float> %1897, float %1899, i64 1, !dbg !20
  %1901 = select i1 %.not40, float %1895, float %1804, !dbg !6
  %1902 = insertelement <4 x float> %1900, float %1901, i64 2, !dbg !20
  %1903 = select i1 %.not40, float %1898, float %1805, !dbg !6
  %1904 = insertelement <4 x float> %1902, float %1903, i64 3, !dbg !20
  %1905 = fmul <4 x float> %1904, %1859, !dbg !20
  %1906 = fptrunc <4 x float> %1905 to <4 x half>, !dbg !21
  %1907 = extractelement <4 x half> %1906, i64 3, !dbg !21
  %1908 = extractelement <4 x half> %1906, i64 2, !dbg !21
  %1909 = extractelement <4 x half> %1906, i64 1, !dbg !21
  %1910 = extractelement <4 x half> %1906, i64 0, !dbg !21
  %1911 = bitcast i32 %1786 to float, !dbg !6
  %1912 = select i1 %.not40, float %1776, float %1911, !dbg !6
  %1913 = insertelement <4 x float> poison, float %1912, i64 0, !dbg !20
  %1914 = bitcast i32 %1789 to float, !dbg !6
  %1915 = select i1 %.not40, float %1777, float %1914, !dbg !6
  %1916 = insertelement <4 x float> %1913, float %1915, i64 1, !dbg !20
  %1917 = select i1 %.not40, float %1911, float %1782, !dbg !6
  %1918 = insertelement <4 x float> %1916, float %1917, i64 2, !dbg !20
  %1919 = select i1 %.not40, float %1914, float %1783, !dbg !6
  %1920 = insertelement <4 x float> %1918, float %1919, i64 3, !dbg !20
  %1921 = fmul <4 x float> %1920, %1859, !dbg !20
  %1922 = fptrunc <4 x float> %1921 to <4 x half>, !dbg !21
  %1923 = extractelement <4 x half> %1922, i64 3, !dbg !21
  %1924 = extractelement <4 x half> %1922, i64 2, !dbg !21
  %1925 = extractelement <4 x half> %1922, i64 1, !dbg !21
  %1926 = extractelement <4 x half> %1922, i64 0, !dbg !21
  %address.thread950 = call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #9, !dbg !22, !range !33, !frisk.register_lifetimes !1
  %address.local951 = zext nneg i32 %address.thread950 to i64, !dbg !9
  %address.local952 = add i64 %42, %address.local951, !dbg !22
  %address.local953 = lshr i64 %address.local951, 2, !dbg !19
  %address.local954 = and i64 %address.local953, 48, !dbg !19
  %address.local955 = add i64 %address.local952, %address.local954, !dbg !22
  %address.local956 = and i64 %address.local951, 240, !dbg !22
  %address.local958 = sub i64 %address.local955, %address.local956, !dbg !22
  %address.local959 = add i64 %address.local958, %43, !dbg !22
  %address.local960 = shl nuw nsw i64 %address.local959, 7, !dbg !22
  %1927 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %60, !dbg !22
  %address.local962 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %1927, i64 %address.local960, !dbg !22
  %address.local964 = and i64 %address.local953, 12, !dbg !13
  %address.local965 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local962, i64 %address.local964, !dbg !22
  store half %1926, ptr addrspace(1) %address.local965, align 2, !dbg !22
  %address.local967 = or disjoint i64 %address.local964, 1, !dbg !22
  %address.local968 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local962, i64 %address.local967, !dbg !22
  store half %1925, ptr addrspace(1) %address.local968, align 2, !dbg !22
  %address.local970 = or disjoint i64 %address.local964, 2, !dbg !22
  %address.local971 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local962, i64 %address.local970, !dbg !22
  store half %1924, ptr addrspace(1) %address.local971, align 2, !dbg !22
  %address.local973 = or disjoint i64 %address.local964, 3, !dbg !22
  %address.local974 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local962, i64 %address.local973, !dbg !22
  store half %1923, ptr addrspace(1) %address.local974, align 2, !dbg !22
  %address.local976 = or disjoint i64 %address.local964, 16, !dbg !22
  %address.local977 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local962, i64 %address.local976, !dbg !22
  store half %1910, ptr addrspace(1) %address.local977, align 2, !dbg !22
  %address.local979 = or disjoint i64 %address.local964, 17, !dbg !22
  %address.local980 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local962, i64 %address.local979, !dbg !22
  store half %1909, ptr addrspace(1) %address.local980, align 2, !dbg !22
  %address.local982 = or disjoint i64 %address.local964, 18, !dbg !22
  %address.local983 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local962, i64 %address.local982, !dbg !22
  store half %1908, ptr addrspace(1) %address.local983, align 2, !dbg !22
  %address.local985 = or disjoint i64 %address.local964, 19, !dbg !22
  %address.local986 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local962, i64 %address.local985, !dbg !22
  store half %1907, ptr addrspace(1) %address.local986, align 2, !dbg !22
  %address.local990 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local962, i64 16384, !dbg !22
  %address.local991 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local990, i64 %address.local964, !dbg !22
  store half %1894, ptr addrspace(1) %address.local991, align 2, !dbg !22
  %address.local993 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local990, i64 %address.local967, !dbg !22
  store half %1893, ptr addrspace(1) %address.local993, align 2, !dbg !22
  %address.local995 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local990, i64 %address.local970, !dbg !22
  store half %1892, ptr addrspace(1) %address.local995, align 2, !dbg !22
  %address.local997 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local990, i64 %address.local973, !dbg !22
  store half %1891, ptr addrspace(1) %address.local997, align 2, !dbg !22
  %address.local999 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local990, i64 %address.local976, !dbg !22
  store half %1878, ptr addrspace(1) %address.local999, align 2, !dbg !22
  %address.local1001 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local990, i64 %address.local979, !dbg !22
  store half %1877, ptr addrspace(1) %address.local1001, align 2, !dbg !22
  %address.local1003 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local990, i64 %address.local982, !dbg !22
  store half %1876, ptr addrspace(1) %address.local1003, align 2, !dbg !22
  %address.local1005 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local990, i64 %address.local985, !dbg !22
  store half %1875, ptr addrspace(1) %address.local1005, align 2, !dbg !22
  %1928 = insertelement <4 x float> poison, float %1759, i64 0, !dbg !20
  %1929 = insertelement <4 x float> %1928, float %1765, i64 1, !dbg !20
  %1930 = insertelement <4 x float> %1929, float %1760, i64 2, !dbg !20
  %1931 = insertelement <4 x float> %1930, float %1766, i64 3, !dbg !20
  %1932 = fmul <4 x float> %1931, %1862, !dbg !20
  %1933 = fptrunc <4 x float> %1932 to <4 x half>, !dbg !21
  %1934 = extractelement <4 x half> %1933, i64 3, !dbg !21
  %1935 = extractelement <4 x half> %1933, i64 2, !dbg !21
  %1936 = extractelement <4 x half> %1933, i64 1, !dbg !21
  %1937 = extractelement <4 x half> %1933, i64 0, !dbg !21
  %1938 = insertelement <4 x float> poison, float %1731, i64 0, !dbg !20
  %1939 = insertelement <4 x float> %1938, float %1737, i64 1, !dbg !20
  %1940 = insertelement <4 x float> %1939, float %1732, i64 2, !dbg !20
  %1941 = insertelement <4 x float> %1940, float %1738, i64 3, !dbg !20
  %1942 = fmul <4 x float> %1941, %1862, !dbg !20
  %1943 = fptrunc <4 x float> %1942 to <4 x half>, !dbg !21
  %1944 = extractelement <4 x half> %1943, i64 3, !dbg !21
  %1945 = extractelement <4 x half> %1943, i64 2, !dbg !21
  %1946 = extractelement <4 x half> %1943, i64 1, !dbg !21
  %1947 = extractelement <4 x half> %1943, i64 0, !dbg !21
  %1948 = insertelement <4 x float> poison, float %1702, i64 0, !dbg !20
  %1949 = insertelement <4 x float> %1948, float %1708, i64 1, !dbg !20
  %1950 = insertelement <4 x float> %1949, float %1703, i64 2, !dbg !20
  %1951 = insertelement <4 x float> %1950, float %1709, i64 3, !dbg !20
  %1952 = fmul <4 x float> %1951, %1859, !dbg !20
  %1953 = fptrunc <4 x float> %1952 to <4 x half>, !dbg !21
  %1954 = extractelement <4 x half> %1953, i64 3, !dbg !21
  %1955 = extractelement <4 x half> %1953, i64 2, !dbg !21
  %1956 = extractelement <4 x half> %1953, i64 1, !dbg !21
  %1957 = extractelement <4 x half> %1953, i64 0, !dbg !21
  %1958 = insertelement <4 x float> poison, float %1674, i64 0, !dbg !20
  %1959 = insertelement <4 x float> %1958, float %1680, i64 1, !dbg !20
  %1960 = insertelement <4 x float> %1959, float %1675, i64 2, !dbg !20
  %1961 = insertelement <4 x float> %1960, float %1681, i64 3, !dbg !20
  %1962 = fmul <4 x float> %1961, %1859, !dbg !20
  %1963 = fptrunc <4 x float> %1962 to <4 x half>, !dbg !21
  %1964 = extractelement <4 x half> %1963, i64 3, !dbg !21
  %1965 = extractelement <4 x half> %1963, i64 2, !dbg !21
  %1966 = extractelement <4 x half> %1963, i64 1, !dbg !21
  %1967 = extractelement <4 x half> %1963, i64 0, !dbg !21
  %address.local1021 = or disjoint i64 %address.local964, 32, !dbg !22
  %address.local1022 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local962, i64 %address.local1021, !dbg !22
  store half %1967, ptr addrspace(1) %address.local1022, align 2, !dbg !22
  %address.local1024 = or disjoint i64 %address.local964, 33, !dbg !22
  %address.local1025 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local962, i64 %address.local1024, !dbg !22
  store half %1966, ptr addrspace(1) %address.local1025, align 2, !dbg !22
  %address.local1027 = or disjoint i64 %address.local964, 34, !dbg !22
  %address.local1028 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local962, i64 %address.local1027, !dbg !22
  store half %1965, ptr addrspace(1) %address.local1028, align 2, !dbg !22
  %address.local1030 = or disjoint i64 %address.local964, 35, !dbg !22
  %address.local1031 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local962, i64 %address.local1030, !dbg !22
  store half %1964, ptr addrspace(1) %address.local1031, align 2, !dbg !22
  %address.local1033 = or disjoint i64 %address.local964, 48, !dbg !22
  %address.local1034 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local962, i64 %address.local1033, !dbg !22
  store half %1957, ptr addrspace(1) %address.local1034, align 2, !dbg !22
  %address.local1036 = or disjoint i64 %address.local964, 49, !dbg !22
  %address.local1037 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local962, i64 %address.local1036, !dbg !22
  store half %1956, ptr addrspace(1) %address.local1037, align 2, !dbg !22
  %address.local1039 = or disjoint i64 %address.local964, 50, !dbg !22
  %address.local1040 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local962, i64 %address.local1039, !dbg !22
  store half %1955, ptr addrspace(1) %address.local1040, align 2, !dbg !22
  %address.local1042 = or i64 %address.local953, 51, !dbg !22
  %address.local1043 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local962, i64 %address.local1042, !dbg !22
  store half %1954, ptr addrspace(1) %address.local1043, align 2, !dbg !22
  %address.local1048 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local990, i64 %address.local1021, !dbg !22
  store half %1947, ptr addrspace(1) %address.local1048, align 2, !dbg !22
  %address.local1050 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local990, i64 %address.local1024, !dbg !22
  store half %1946, ptr addrspace(1) %address.local1050, align 2, !dbg !22
  %address.local1052 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local990, i64 %address.local1027, !dbg !22
  store half %1945, ptr addrspace(1) %address.local1052, align 2, !dbg !22
  %address.local1054 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local990, i64 %address.local1030, !dbg !22
  store half %1944, ptr addrspace(1) %address.local1054, align 2, !dbg !22
  %address.local1056 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local990, i64 %address.local1033, !dbg !22
  store half %1937, ptr addrspace(1) %address.local1056, align 2, !dbg !22
  %address.local1058 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local990, i64 %address.local1036, !dbg !22
  store half %1936, ptr addrspace(1) %address.local1058, align 2, !dbg !22
  %address.local1060 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local990, i64 %address.local1039, !dbg !22
  store half %1935, ptr addrspace(1) %address.local1060, align 2, !dbg !22
  %address.local1062 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local990, i64 %address.local1042, !dbg !22
  store half %1934, ptr addrspace(1) %address.local1062, align 2, !dbg !22
  %1968 = insertelement <4 x float> poison, float %1645, i64 0, !dbg !20
  %1969 = insertelement <4 x float> %1968, float %1651, i64 1, !dbg !20
  %1970 = insertelement <4 x float> %1969, float %1646, i64 2, !dbg !20
  %1971 = insertelement <4 x float> %1970, float %1652, i64 3, !dbg !20
  %1972 = fmul <4 x float> %1971, %1862, !dbg !20
  %1973 = fptrunc <4 x float> %1972 to <4 x half>, !dbg !21
  %1974 = extractelement <4 x half> %1973, i64 3, !dbg !21
  %1975 = extractelement <4 x half> %1973, i64 2, !dbg !21
  %1976 = extractelement <4 x half> %1973, i64 1, !dbg !21
  %1977 = extractelement <4 x half> %1973, i64 0, !dbg !21
  %1978 = insertelement <4 x float> poison, float %1617, i64 0, !dbg !20
  %1979 = insertelement <4 x float> %1978, float %1623, i64 1, !dbg !20
  %1980 = insertelement <4 x float> %1979, float %1618, i64 2, !dbg !20
  %1981 = insertelement <4 x float> %1980, float %1624, i64 3, !dbg !20
  %1982 = fmul <4 x float> %1981, %1862, !dbg !20
  %1983 = fptrunc <4 x float> %1982 to <4 x half>, !dbg !21
  %1984 = extractelement <4 x half> %1983, i64 3, !dbg !21
  %1985 = extractelement <4 x half> %1983, i64 2, !dbg !21
  %1986 = extractelement <4 x half> %1983, i64 1, !dbg !21
  %1987 = extractelement <4 x half> %1983, i64 0, !dbg !21
  %1988 = insertelement <4 x float> poison, float %1588, i64 0, !dbg !20
  %1989 = insertelement <4 x float> %1988, float %1594, i64 1, !dbg !20
  %1990 = insertelement <4 x float> %1989, float %1589, i64 2, !dbg !20
  %1991 = insertelement <4 x float> %1990, float %1595, i64 3, !dbg !20
  %1992 = fmul <4 x float> %1991, %1859, !dbg !20
  %1993 = fptrunc <4 x float> %1992 to <4 x half>, !dbg !21
  %1994 = extractelement <4 x half> %1993, i64 3, !dbg !21
  %1995 = extractelement <4 x half> %1993, i64 2, !dbg !21
  %1996 = extractelement <4 x half> %1993, i64 1, !dbg !21
  %1997 = extractelement <4 x half> %1993, i64 0, !dbg !21
  %1998 = insertelement <4 x float> poison, float %1560, i64 0, !dbg !20
  %1999 = insertelement <4 x float> %1998, float %1566, i64 1, !dbg !20
  %2000 = insertelement <4 x float> %1999, float %1561, i64 2, !dbg !20
  %2001 = insertelement <4 x float> %2000, float %1567, i64 3, !dbg !20
  %2002 = fmul <4 x float> %2001, %1859, !dbg !20
  %2003 = fptrunc <4 x float> %2002 to <4 x half>, !dbg !21
  %2004 = extractelement <4 x half> %2003, i64 3, !dbg !21
  %2005 = extractelement <4 x half> %2003, i64 2, !dbg !21
  %2006 = extractelement <4 x half> %2003, i64 1, !dbg !21
  %2007 = extractelement <4 x half> %2003, i64 0, !dbg !21
  %address.local1078 = or disjoint i64 %address.local964, 64, !dbg !22
  %address.local1079 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local962, i64 %address.local1078, !dbg !22
  store half %2007, ptr addrspace(1) %address.local1079, align 2, !dbg !22
  %address.local1081 = or disjoint i64 %address.local964, 65, !dbg !22
  %address.local1082 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local962, i64 %address.local1081, !dbg !22
  store half %2006, ptr addrspace(1) %address.local1082, align 2, !dbg !22
  %address.local1084 = or disjoint i64 %address.local964, 66, !dbg !22
  %address.local1085 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local962, i64 %address.local1084, !dbg !22
  store half %2005, ptr addrspace(1) %address.local1085, align 2, !dbg !22
  %address.local1087 = or disjoint i64 %address.local964, 67, !dbg !22
  %address.local1088 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local962, i64 %address.local1087, !dbg !22
  store half %2004, ptr addrspace(1) %address.local1088, align 2, !dbg !22
  %address.local1090 = or disjoint i64 %address.local964, 80, !dbg !22
  %address.local1091 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local962, i64 %address.local1090, !dbg !22
  store half %1997, ptr addrspace(1) %address.local1091, align 2, !dbg !22
  %address.local1093 = or disjoint i64 %address.local964, 81, !dbg !22
  %address.local1094 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local962, i64 %address.local1093, !dbg !22
  store half %1996, ptr addrspace(1) %address.local1094, align 2, !dbg !22
  %address.local1096 = or disjoint i64 %address.local964, 82, !dbg !22
  %address.local1097 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local962, i64 %address.local1096, !dbg !22
  store half %1995, ptr addrspace(1) %address.local1097, align 2, !dbg !22
  %address.local1099 = or disjoint i64 %address.local964, 83, !dbg !22
  %address.local1100 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local962, i64 %address.local1099, !dbg !22
  store half %1994, ptr addrspace(1) %address.local1100, align 2, !dbg !22
  %address.local1105 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local990, i64 %address.local1078, !dbg !22
  store half %1987, ptr addrspace(1) %address.local1105, align 2, !dbg !22
  %address.local1107 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local990, i64 %address.local1081, !dbg !22
  store half %1986, ptr addrspace(1) %address.local1107, align 2, !dbg !22
  %address.local1109 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local990, i64 %address.local1084, !dbg !22
  store half %1985, ptr addrspace(1) %address.local1109, align 2, !dbg !22
  %address.local1111 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local990, i64 %address.local1087, !dbg !22
  store half %1984, ptr addrspace(1) %address.local1111, align 2, !dbg !22
  %address.local1113 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local990, i64 %address.local1090, !dbg !22
  store half %1977, ptr addrspace(1) %address.local1113, align 2, !dbg !22
  %address.local1115 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local990, i64 %address.local1093, !dbg !22
  store half %1976, ptr addrspace(1) %address.local1115, align 2, !dbg !22
  %address.local1117 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local990, i64 %address.local1096, !dbg !22
  store half %1975, ptr addrspace(1) %address.local1117, align 2, !dbg !22
  %address.local1119 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local990, i64 %address.local1099, !dbg !22
  store half %1974, ptr addrspace(1) %address.local1119, align 2, !dbg !22
  %2008 = insertelement <4 x float> poison, float %1531, i64 0, !dbg !20
  %2009 = insertelement <4 x float> %2008, float %1537, i64 1, !dbg !20
  %2010 = insertelement <4 x float> %2009, float %1532, i64 2, !dbg !20
  %2011 = insertelement <4 x float> %2010, float %1538, i64 3, !dbg !20
  %2012 = fmul <4 x float> %2011, %1862, !dbg !20
  %2013 = fptrunc <4 x float> %2012 to <4 x half>, !dbg !21
  %2014 = extractelement <4 x half> %2013, i64 3, !dbg !21
  %2015 = extractelement <4 x half> %2013, i64 2, !dbg !21
  %2016 = extractelement <4 x half> %2013, i64 1, !dbg !21
  %2017 = extractelement <4 x half> %2013, i64 0, !dbg !21
  %2018 = insertelement <4 x float> poison, float %1503, i64 0, !dbg !20
  %2019 = insertelement <4 x float> %2018, float %1509, i64 1, !dbg !20
  %2020 = insertelement <4 x float> %2019, float %1504, i64 2, !dbg !20
  %2021 = insertelement <4 x float> %2020, float %1510, i64 3, !dbg !20
  %2022 = fmul <4 x float> %2021, %1862, !dbg !20
  %2023 = fptrunc <4 x float> %2022 to <4 x half>, !dbg !21
  %2024 = extractelement <4 x half> %2023, i64 3, !dbg !21
  %2025 = extractelement <4 x half> %2023, i64 2, !dbg !21
  %2026 = extractelement <4 x half> %2023, i64 1, !dbg !21
  %2027 = extractelement <4 x half> %2023, i64 0, !dbg !21
  %2028 = insertelement <4 x float> poison, float %1474, i64 0, !dbg !20
  %2029 = insertelement <4 x float> %2028, float %1480, i64 1, !dbg !20
  %2030 = insertelement <4 x float> %2029, float %1475, i64 2, !dbg !20
  %2031 = insertelement <4 x float> %2030, float %1481, i64 3, !dbg !20
  %2032 = fmul <4 x float> %2031, %1859, !dbg !20
  %2033 = fptrunc <4 x float> %2032 to <4 x half>, !dbg !21
  %2034 = extractelement <4 x half> %2033, i64 3, !dbg !21
  %2035 = extractelement <4 x half> %2033, i64 2, !dbg !21
  %2036 = extractelement <4 x half> %2033, i64 1, !dbg !21
  %2037 = extractelement <4 x half> %2033, i64 0, !dbg !21
  %2038 = insertelement <4 x float> poison, float %1446, i64 0, !dbg !20
  %2039 = insertelement <4 x float> %2038, float %1452, i64 1, !dbg !20
  %2040 = insertelement <4 x float> %2039, float %1447, i64 2, !dbg !20
  %2041 = insertelement <4 x float> %2040, float %1453, i64 3, !dbg !20
  %2042 = fmul <4 x float> %2041, %1859, !dbg !20
  %2043 = fptrunc <4 x float> %2042 to <4 x half>, !dbg !21
  %2044 = extractelement <4 x half> %2043, i64 3, !dbg !21
  %2045 = extractelement <4 x half> %2043, i64 2, !dbg !21
  %2046 = extractelement <4 x half> %2043, i64 1, !dbg !21
  %2047 = extractelement <4 x half> %2043, i64 0, !dbg !21
  %address.local1135 = or disjoint i64 %address.local964, 96, !dbg !22
  %address.local1136 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local962, i64 %address.local1135, !dbg !22
  store half %2047, ptr addrspace(1) %address.local1136, align 2, !dbg !22
  %address.local1138 = or disjoint i64 %address.local964, 97, !dbg !22
  %address.local1139 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local962, i64 %address.local1138, !dbg !22
  store half %2046, ptr addrspace(1) %address.local1139, align 2, !dbg !22
  %address.local1141 = or disjoint i64 %address.local964, 98, !dbg !22
  %address.local1142 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local962, i64 %address.local1141, !dbg !22
  store half %2045, ptr addrspace(1) %address.local1142, align 2, !dbg !22
  %address.local1144 = or disjoint i64 %address.local964, 99, !dbg !22
  %address.local1145 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local962, i64 %address.local1144, !dbg !22
  store half %2044, ptr addrspace(1) %address.local1145, align 2, !dbg !22
  %address.local1147 = or disjoint i64 %address.local964, 112, !dbg !22
  %address.local1148 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local962, i64 %address.local1147, !dbg !22
  store half %2037, ptr addrspace(1) %address.local1148, align 2, !dbg !22
  %address.local1150 = or disjoint i64 %address.local964, 113, !dbg !22
  %address.local1151 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local962, i64 %address.local1150, !dbg !22
  store half %2036, ptr addrspace(1) %address.local1151, align 2, !dbg !22
  %address.local1153 = or disjoint i64 %address.local964, 114, !dbg !22
  %address.local1154 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local962, i64 %address.local1153, !dbg !22
  store half %2035, ptr addrspace(1) %address.local1154, align 2, !dbg !22
  %address.local1156 = or i64 %address.local953, 115, !dbg !22
  %address.local1157 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local962, i64 %address.local1156, !dbg !22
  store half %2034, ptr addrspace(1) %address.local1157, align 2, !dbg !22
  %address.local1162 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local990, i64 %address.local1135, !dbg !22
  store half %2027, ptr addrspace(1) %address.local1162, align 2, !dbg !22
  %address.local1164 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local990, i64 %address.local1138, !dbg !22
  store half %2026, ptr addrspace(1) %address.local1164, align 2, !dbg !22
  %address.local1166 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local990, i64 %address.local1141, !dbg !22
  store half %2025, ptr addrspace(1) %address.local1166, align 2, !dbg !22
  %address.local1168 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local990, i64 %address.local1144, !dbg !22
  store half %2024, ptr addrspace(1) %address.local1168, align 2, !dbg !22
  %address.local1170 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local990, i64 %address.local1147, !dbg !22
  store half %2017, ptr addrspace(1) %address.local1170, align 2, !dbg !22
  %address.local1172 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local990, i64 %address.local1150, !dbg !22
  store half %2016, ptr addrspace(1) %address.local1172, align 2, !dbg !22
  %address.local1174 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local990, i64 %address.local1153, !dbg !22
  store half %2015, ptr addrspace(1) %address.local1174, align 2, !dbg !22
  %address.local1176 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local990, i64 %address.local1156, !dbg !22
  store half %2014, ptr addrspace(1) %address.local1176, align 2, !dbg !22
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

!0 = !DIFile(filename: "causal.input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/thread_v_double_buffer")
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
!35 = !{i32 0}
!36 = !{i32 1}
!37 = !{i32 2}
!38 = !{i32 3}
!39 = !{i32 5}
!40 = !{i32 6}
!41 = !{i32 7}
!42 = !{i32 8}
!43 = !{i32 9}
!44 = !{i32 10}
!45 = !{i32 11}
!46 = !{i32 12}
!47 = !{i32 13}
!48 = !{i32 14}
!49 = !{i32 15}
!50 = !{i32 16}
!51 = !{i32 17}
!52 = !{i32 18}
!53 = !{i32 19}
!54 = !{i32 20}
!55 = !{i32 21}
!56 = !{i32 22}
