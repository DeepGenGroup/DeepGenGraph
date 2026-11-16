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
  %address.local332 = zext nneg i32 %48 to i64, !dbg !9
  %address.local333 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local332, !dbg !9
  %address.local334 = ptrtoint ptr addrspace(3) %address.local333 to i64, !dbg !9
  %address.local335 = trunc i64 %address.local334 to i32, !dbg !9
  %49 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local335), !dbg !9
  %address.local336 = lshr i32 %address.thread, 4, !dbg !9
  %address.local337 = and i32 %address.local336, 8, !dbg !9
  %address.local338 = shl nuw nsw i32 %address.thread, 7, !dbg !9
  %address.local339 = and i32 %address.local338, 16256, !dbg !9
  %address.local340 = or disjoint i32 %address.local337, %address.local339, !dbg !9
  %50 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !9
  %.idx1310 = shl i64 %44, 8, !dbg !9
  %51 = add i64 %.idx, %.idx1310, !dbg !9
  %52 = shl nuw nsw i32 %address.local340, 1, !dbg !9
  %53 = trunc i64 %51 to i32, !dbg !9
  %dma.buffer.offset1308 = or disjoint i32 %52, %53, !dbg !9
  %54 = insertelement <2 x i64> poison, i64 %50, i64 0, !dbg !9
  %55 = bitcast <2 x i64> %54 to <4 x i32>, !dbg !9
  %dma.buffer.rsrc1309 = shufflevector <4 x i32> %55, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !9
  %56 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %49, i32 %dma.buffer.offset1308, <4 x i32> %dma.buffer.rsrc1309) #10, !dbg !9
  %address.local345 = or disjoint i64 %address.local332, 2048, !dbg !9
  %address.local346 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local345, !dbg !9
  %address.local347 = ptrtoint ptr addrspace(3) %address.local346 to i64, !dbg !9
  %address.local348 = trunc i64 %address.local347 to i32, !dbg !9
  %57 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local348), !dbg !9
  %address.local354 = or disjoint i32 %address.local340, 16, !dbg !9
  %58 = shl nuw nsw i32 %address.local354, 1, !dbg !9
  %dma.buffer.offset1306 = or disjoint i32 %58, %53, !dbg !9
  %59 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %57, i32 %dma.buffer.offset1306, <4 x i32> %dma.buffer.rsrc1309) #10, !dbg !9
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %60 = shl nuw nsw i64 %41, 19, !dbg !9
  %61 = add i64 %60, %45, !dbg !9
  %address.local357 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local332, !dbg !9
  %address.local358 = ptrtoint ptr addrspace(3) %address.local357 to i64, !dbg !9
  %address.local359 = trunc i64 %address.local358 to i32, !dbg !9
  %62 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local359), !dbg !9
  %63 = trunc i64 %61 to i32, !dbg !9
  %.tr = or disjoint i32 %address.local340, %63, !dbg !9
  %64 = shl i32 %.tr, 1, !dbg !9
  %dma.buffer.offset1304 = or disjoint i32 %64, 64, !dbg !9
  %65 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %62, i32 %dma.buffer.offset1304, <4 x i32> %dma.buffer.rsrc1309) #10, !dbg !9
  %address.local361 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local345, !dbg !9
  %address.local362 = ptrtoint ptr addrspace(3) %address.local361 to i64, !dbg !9
  %address.local363 = trunc i64 %address.local362 to i32, !dbg !9
  %66 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local363), !dbg !9
  %.tr1314 = or disjoint i32 %address.local354, %63, !dbg !9
  %67 = shl i32 %.tr1314, 1, !dbg !9
  %dma.buffer.offset1302 = or disjoint i32 %67, 64, !dbg !9
  %68 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %66, i32 %dma.buffer.offset1302, <4 x i32> %dma.buffer.rsrc1309) #10, !dbg !9
  %69 = and i64 %46, 240, !dbg !19
  %address.local365 = lshr i32 %address.thread, 2, !dbg !13
  %address.local366 = and i32 %address.local365, 4, !dbg !13
  %address.local367 = shl nuw nsw i32 %address.thread, 1, !dbg !13
  %address.local368 = and i32 %address.local367, 384, !dbg !13
  %address.local369 = or disjoint i32 %address.local366, %address.local368, !dbg !13
  %address.local371 = and i32 %47, 120, !dbg !13
  %address.local372 = or disjoint i32 %address.local369, %address.local371, !dbg !13
  %address.local373 = shl nuw nsw i32 %address.thread, 5, !dbg !13
  %address.local374 = and i32 %address.local373, 1024, !dbg !13
  %address.local375 = or disjoint i32 %address.local372, %address.local374, !dbg !13
  %address.local376 = zext nneg i32 %address.local375 to i64, !dbg !13
  %address.local377 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local376, !dbg !13
  %70 = load <4 x half>, ptr addrspace(3) %address.local377, align 8, !dbg !13
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local377, i64 4096, !dbg !13
  %71 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !13
  %72 = fmul <4 x half> %70, splat (half 0xH3015), !dbg !13
  %lds.const.gep1131 = getelementptr i8, ptr addrspace(3) %address.local377, i64 1024, !dbg !13
  %73 = load <4 x half>, ptr addrspace(3) %lds.const.gep1131, align 8, !dbg !13
  %74 = fmul <4 x half> %71, splat (half 0xH3015), !dbg !13
  %lds.const.gep1132 = getelementptr i8, ptr addrspace(3) %address.local377, i64 5120, !dbg !13
  %75 = load <4 x half>, ptr addrspace(3) %lds.const.gep1132, align 8, !dbg !13
  %76 = fmul <4 x half> %73, splat (half 0xH3015), !dbg !13
  %77 = fmul <4 x half> %75, splat (half 0xH3015), !dbg !13
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %dma.buffer.offset1300 = or disjoint i32 %64, 128, !dbg !9
  %78 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %49, i32 %dma.buffer.offset1300, <4 x i32> %dma.buffer.rsrc1309) #10, !dbg !9
  %dma.buffer.offset1298 = or disjoint i32 %67, 128, !dbg !9
  %79 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %57, i32 %dma.buffer.offset1298, <4 x i32> %dma.buffer.rsrc1309) #10, !dbg !9
  %lds.const.gep1133 = getelementptr i8, ptr addrspace(3) %address.local377, i64 8192, !dbg !13
  %80 = load <4 x half>, ptr addrspace(3) %lds.const.gep1133, align 8, !dbg !13
  %lds.const.gep1134 = getelementptr i8, ptr addrspace(3) %address.local377, i64 12288, !dbg !13
  %81 = load <4 x half>, ptr addrspace(3) %lds.const.gep1134, align 8, !dbg !13
  %lds.const.gep1135 = getelementptr i8, ptr addrspace(3) %address.local377, i64 9216, !dbg !13
  %82 = load <4 x half>, ptr addrspace(3) %lds.const.gep1135, align 8, !dbg !13
  %83 = fmul <4 x half> %80, splat (half 0xH3015), !dbg !13
  %lds.const.gep1136 = getelementptr i8, ptr addrspace(3) %address.local377, i64 13312, !dbg !13
  %84 = load <4 x half>, ptr addrspace(3) %lds.const.gep1136, align 8, !dbg !13
  %85 = fmul <4 x half> %81, splat (half 0xH3015), !dbg !13
  %86 = fmul <4 x half> %82, splat (half 0xH3015), !dbg !13
  %87 = fmul <4 x half> %84, splat (half 0xH3015), !dbg !13
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %dma.buffer.offset1296 = or disjoint i32 %64, 192, !dbg !9
  %88 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %62, i32 %dma.buffer.offset1296, <4 x i32> %dma.buffer.rsrc1309) #10, !dbg !9
  %dma.buffer.offset1294 = or disjoint i32 %67, 192, !dbg !9
  %89 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %66, i32 %dma.buffer.offset1294, <4 x i32> %dma.buffer.rsrc1309) #10, !dbg !9
  %90 = load <4 x half>, ptr addrspace(3) %address.local377, align 8, !dbg !13
  %91 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !13
  %92 = load <4 x half>, ptr addrspace(3) %lds.const.gep1131, align 8, !dbg !13
  %93 = fmul <4 x half> %90, splat (half 0xH3015), !dbg !13
  %94 = load <4 x half>, ptr addrspace(3) %lds.const.gep1132, align 8, !dbg !13
  %95 = fmul <4 x half> %91, splat (half 0xH3015), !dbg !13
  %96 = fmul <4 x half> %92, splat (half 0xH3015), !dbg !13
  %97 = fmul <4 x half> %94, splat (half 0xH3015), !dbg !13
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %98 = load <4 x half>, ptr addrspace(3) %lds.const.gep1133, align 8, !dbg !13
  %99 = load <4 x half>, ptr addrspace(3) %lds.const.gep1134, align 8, !dbg !13
  %100 = load <4 x half>, ptr addrspace(3) %lds.const.gep1135, align 8, !dbg !13
  %101 = fmul <4 x half> %98, splat (half 0xH3015), !dbg !13
  %102 = load <4 x half>, ptr addrspace(3) %lds.const.gep1136, align 8, !dbg !13
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
  %index.compact262 = zext nneg i32 %114 to i64, !dbg !17
  %115 = or disjoint i64 %113, %index.compact262, !dbg !17
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
  %128 = phi i64 [ %1395, %136 ], [ 0, %4 ], !dbg !6
  %129 = phi [2 x <8 x float>] [ %1380, %136 ], [ zeroinitializer, %4 ], !dbg !6
  %130 = phi [2 x <1 x float>] [ %1146, %136 ], [ zeroinitializer, %4 ], !dbg !6
  %131 = phi [2 x <1 x float>] [ %1382, %136 ], [ [<1 x float> splat (float 0xFFF0000000000000), <1 x float> splat (float 0xFFF0000000000000)], %4 ], !dbg !6
  %132 = phi [2 x <8 x float>] [ %1386, %136 ], [ zeroinitializer, %4 ], !dbg !6
  %133 = phi [2 x <8 x float>] [ %1390, %136 ], [ zeroinitializer, %4 ], !dbg !6
  %134 = phi [2 x <8 x float>] [ %1394, %136 ], [ zeroinitializer, %4 ], !dbg !6
  %135 = icmp slt i64 %128, %106, !dbg !6
  br i1 %135, label %136, label %1396, !dbg !6

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
  %address.thread428 = call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #9, !dbg !16, !range !33, !frisk.register_lifetimes !1
  %address.local429 = zext nneg i32 %address.thread428 to i64, !dbg !9
  %address.local435 = lshr i64 %address.local429, 2, !dbg !16
  %147 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %.idx13231326 = or disjoint i64 %143, %140, !dbg !16
  %148 = add i64 %.idx13231326, %146, !dbg !16
  %149 = insertelement <2 x i64> poison, i64 %147, i64 0, !dbg !16
  %150 = bitcast <2 x i64> %149 to <4 x i32>, !dbg !16
  %dma.buffer.rsrc1293 = shufflevector <4 x i32> %150, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !16
  %kdma.source = zext nneg i32 %kdma.owner to i64, !dbg !16
  %kdma.source1460 = lshr i64 %kdma.source, 4, !dbg !16
  %kdma.source1461 = shl nuw nsw i64 %kdma.source1460, 9, !dbg !16
  %kdma.source1462 = shl nuw nsw i64 %kdma.source, 7, !dbg !16
  %kdma.source1463 = add nuw nsw i64 %kdma.source1461, %kdma.source1462, !dbg !16
  %kdma.source1464 = lshr i64 %kdma.source, 5, !dbg !16
  %kdma.source1465 = lshr i64 %kdma.source, 2, !dbg !16
  %kdma.source1466 = and i64 %kdma.source1465, 1, !dbg !16
  %kdma.source1467 = and i64 %kdma.source1460, %kdma.source1466, !dbg !16
  %kdma.source1468 = or disjoint i64 %kdma.source1464, %kdma.source1467, !dbg !16
  %kdma.source1469 = shl nuw nsw i64 %kdma.source1468, 10, !dbg !16
  %kdma.source1470 = sub nsw i64 %kdma.source1463, %kdma.source1469, !dbg !16
  %kdma.source1471 = and i64 %kdma.source1470, 16256, !dbg !16
  %kdma.source1472 = lshr i32 %kdma.tid, 4, !dbg !16
  %kdma.source1473 = and i32 %kdma.source1472, 8, !dbg !16
  %kdma.source1474 = zext nneg i32 %kdma.source1473 to i64, !dbg !16
  %kdma.source1475 = or disjoint i64 %kdma.source1471, %kdma.source1474, !dbg !16
  %kdma.source1476 = add i64 %148, %kdma.source1475, !dbg !16
  %kdma.source1477 = trunc i64 %kdma.source1476 to i32, !dbg !16
  %kdma.source1478 = shl i32 %kdma.source1477, 1, !dbg !16
  %kdma.offset = add i32 %kdma.source1478, %kdma.intra, !dbg !16
  %151 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %49, i32 %kdma.offset, <4 x i32> %dma.buffer.rsrc1293) #10, !dbg !16
  %kdma.source1479 = zext nneg i32 %kdma.owner.hi to i64, !dbg !16
  %kdma.source1480 = lshr i64 %kdma.source1479, 4, !dbg !16
  %kdma.source1481 = shl nuw nsw i64 %kdma.source1480, 9, !dbg !16
  %kdma.source1482 = shl nuw nsw i64 %kdma.source1479, 7, !dbg !16
  %kdma.source1483 = add nuw nsw i64 %kdma.source1481, %kdma.source1482, !dbg !16
  %kdma.source1484 = lshr i64 %kdma.source1479, 5, !dbg !16
  %kdma.source1485 = lshr i64 %kdma.source1479, 2, !dbg !16
  %kdma.source1486 = and i64 %kdma.source1485, 1, !dbg !16
  %kdma.source1487 = and i64 %kdma.source1480, %kdma.source1486, !dbg !16
  %kdma.source1488 = add nuw nsw i64 %kdma.source1484, %kdma.source1487, !dbg !16
  %kdma.source1489 = shl nuw nsw i64 %kdma.source1488, 10, !dbg !16
  %kdma.source1490 = sub nsw i64 %kdma.source1483, %kdma.source1489, !dbg !16
  %kdma.source1491 = and i64 %kdma.source1490, 16256, !dbg !16
  %kdma.source1495 = or disjoint i64 %kdma.source1491, %kdma.source1474, !dbg !16
  %kdma.source1496 = add i64 %148, %kdma.source1495, !dbg !16
  %kdma.source1497 = trunc i64 %kdma.source1496 to i32, !dbg !16
  %kdma.source1498 = shl i32 %kdma.source1497, 1, !dbg !16
  %kdma.offset1499 = add i32 %kdma.source1498, %kdma.intra.hi, !dbg !16
  %kdma.lds1500 = add i32 %49, 512, !dbg !16
  %152 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1500, i32 %kdma.offset1499, <4 x i32> %dma.buffer.rsrc1293) #10, !dbg !16
  %kdma.source1502 = or disjoint i64 %kdma.source, 256, !dbg !16
  %kdma.source1503 = lshr i64 %kdma.source1502, 4, !dbg !16
  %kdma.source1504 = shl nuw nsw i64 %kdma.source1503, 9, !dbg !16
  %kdma.source1506 = add nuw nsw i64 %kdma.source1504, %kdma.source1462, !dbg !16
  %kdma.source1507 = lshr i64 %kdma.source1502, 5, !dbg !16
  %kdma.source1510 = and i64 %kdma.source1503, %kdma.source1466, !dbg !16
  %kdma.source1511 = or disjoint i64 %kdma.source1507, %kdma.source1510, !dbg !16
  %kdma.source1512 = shl nuw nsw i64 %kdma.source1511, 10, !dbg !16
  %kdma.source1513 = sub nsw i64 %kdma.source1506, %kdma.source1512, !dbg !16
  %kdma.source1514 = and i64 %kdma.source1513, 16256, !dbg !16
  %kdma.source1517 = or disjoint i32 %kdma.source1473, 16, !dbg !16
  %kdma.source1518 = zext nneg i32 %kdma.source1517 to i64, !dbg !16
  %kdma.source1519 = or disjoint i64 %kdma.source1514, %kdma.source1518, !dbg !16
  %kdma.source1520 = add i64 %148, %kdma.source1519, !dbg !16
  %kdma.source1521 = trunc i64 %kdma.source1520 to i32, !dbg !16
  %kdma.source1522 = shl i32 %kdma.source1521, 1, !dbg !16
  %kdma.offset1523 = add i32 %kdma.source1522, %kdma.intra, !dbg !16
  %153 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %57, i32 %kdma.offset1523, <4 x i32> %dma.buffer.rsrc1293) #10, !dbg !16
  %kdma.source1526 = or disjoint i64 %kdma.source1479, 256, !dbg !16
  %kdma.source1527 = lshr i64 %kdma.source1526, 4, !dbg !16
  %kdma.source1528 = shl nuw nsw i64 %kdma.source1527, 9, !dbg !16
  %kdma.source1530 = add nuw nsw i64 %kdma.source1528, %kdma.source1482, !dbg !16
  %kdma.source1531 = lshr i64 %kdma.source1526, 5, !dbg !16
  %kdma.source1534 = and i64 %kdma.source1527, %kdma.source1486, !dbg !16
  %kdma.source1535 = add nuw nsw i64 %kdma.source1531, %kdma.source1534, !dbg !16
  %kdma.source1536 = shl nuw nsw i64 %kdma.source1535, 10, !dbg !16
  %kdma.source1537 = sub nsw i64 %kdma.source1530, %kdma.source1536, !dbg !16
  %kdma.source1538 = and i64 %kdma.source1537, 16256, !dbg !16
  %kdma.source1543 = or disjoint i64 %kdma.source1538, %kdma.source1518, !dbg !16
  %kdma.source1544 = add i64 %148, %kdma.source1543, !dbg !16
  %kdma.source1545 = trunc i64 %kdma.source1544 to i32, !dbg !16
  %kdma.source1546 = shl i32 %kdma.source1545, 1, !dbg !16
  %kdma.offset1547 = add i32 %kdma.source1546, %kdma.intra.hi, !dbg !16
  %kdma.lds1548 = add i32 %57, 512, !dbg !16
  %154 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1548, i32 %kdma.offset1547, <4 x i32> %dma.buffer.rsrc1293) #10, !dbg !16
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
  %.idx13331337 = add nuw nsw i64 %174, %139, !dbg !16
  %175 = add i64 %.idx13331337, %172, !dbg !16
  %176 = add i64 %175, %146, !dbg !16
  %kdma.source1566 = add i64 %176, %kdma.source1475, !dbg !16
  %kdma.source1567 = trunc i64 %kdma.source1566 to i32, !dbg !16
  %kdma.source1568 = shl i32 %kdma.source1567, 1, !dbg !16
  %kdma.source1569 = add i32 %kdma.source1568, 64, !dbg !16
  %kdma.offset1570 = add i32 %kdma.source1569, %kdma.intra, !dbg !16
  %177 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %62, i32 %kdma.offset1570, <4 x i32> %dma.buffer.rsrc1293) #10, !dbg !16
  %kdma.source1589 = add i64 %176, %kdma.source1495, !dbg !16
  %kdma.source1590 = trunc i64 %kdma.source1589 to i32, !dbg !16
  %kdma.source1591 = shl i32 %kdma.source1590, 1, !dbg !16
  %kdma.source1592 = add i32 %kdma.source1591, 64, !dbg !16
  %kdma.offset1593 = add i32 %kdma.source1592, %kdma.intra.hi, !dbg !16
  %kdma.lds1594 = add i32 %62, 512, !dbg !16
  %178 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1594, i32 %kdma.offset1593, <4 x i32> %dma.buffer.rsrc1293) #10, !dbg !16
  %kdma.source1614 = add i64 %176, %kdma.source1519, !dbg !16
  %kdma.source1615 = trunc i64 %kdma.source1614 to i32, !dbg !16
  %kdma.source1616 = shl i32 %kdma.source1615, 1, !dbg !16
  %kdma.source1617 = add i32 %kdma.source1616, 64, !dbg !16
  %kdma.offset1618 = add i32 %kdma.source1617, %kdma.intra, !dbg !16
  %179 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %66, i32 %kdma.offset1618, <4 x i32> %dma.buffer.rsrc1293) #10, !dbg !16
  %kdma.source1639 = add i64 %176, %kdma.source1543, !dbg !16
  %kdma.source1640 = trunc i64 %kdma.source1639 to i32, !dbg !16
  %kdma.source1641 = shl i32 %kdma.source1640, 1, !dbg !16
  %kdma.source1642 = add i32 %kdma.source1641, 64, !dbg !16
  %kdma.offset1643 = add i32 %kdma.source1642, %kdma.intra.hi, !dbg !16
  %kdma.lds1644 = add i32 %66, 512, !dbg !16
  %180 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1644, i32 %kdma.offset1643, <4 x i32> %dma.buffer.rsrc1293) #10, !dbg !16
  %address.local472 = shl nuw nsw i64 %address.local429, 1, !dbg !17
  %address.local473 = and i64 %address.local472, 24, !dbg !17
  %address.local475 = shl i64 %address.local429, 62, !dbg !17
  %address.local476 = ashr exact i64 %address.local475, 57, !dbg !17
  %address.local477 = and i64 %address.local476, -64, !dbg !17
  %address.local480 = and i64 %address.local435, 12, !dbg !13
  %address.local482 = and i64 %address.local429, 63, !dbg !13
  %address.local483 = lshr i64 %address.local482, 5, !dbg !13
  %address.local484 = mul nuw nsw i64 %address.local483, 1016, !dbg !13
  %.idx1345 = mul nuw nsw i64 %address.local429, 320, !dbg !17
  %181 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %.idx1345, !dbg !17
  %.idx1346 = mul nsw i64 %address.local435, -1280, !dbg !17
  %182 = getelementptr i8, ptr addrspace(3) %181, i64 %.idx1346, !dbg !17
  %183 = getelementptr [2 x i8], ptr addrspace(3) %182, i64 %address.local473, !dbg !17
  %184 = getelementptr [2 x i8], ptr addrspace(3) %183, i64 %address.local477, !dbg !17
  %185 = getelementptr [2 x i8], ptr addrspace(3) %184, i64 %address.local480, !dbg !17
  %address.local486 = getelementptr [2 x i8], ptr addrspace(3) %185, i64 %address.local484, !dbg !17
  %186 = add nsw i64 %.idx1345, %.idx1346, !dbg !17
  %.idx1963 = shl nuw nsw i64 %address.local473, 1, !dbg !17
  %187 = or disjoint i64 %186, %.idx1963, !dbg !17
  %188 = or disjoint i64 %address.local477, %address.local480, !dbg !17
  %189 = shl nsw i64 %188, 1, !dbg !17
  %190 = add nsw i64 %189, %187, !dbg !17
  %address.local486.idx = mul nuw nsw i64 %address.local483, 2032, !dbg !17
  %191 = add nsw i64 %190, %address.local486.idx, !dbg !17
  %192 = lshr i64 %191, 6, !dbg !17
  %193 = and i64 %192, 8, !dbg !17
  %194 = xor i64 %191, %193, !dbg !17
  %195 = sub i64 %194, %191, !dbg !17
  %kdma.swizzled = getelementptr i8, ptr addrspace(3) %address.local486, i64 %195, !dbg !17
  %196 = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8, !dbg !17
  %197 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %196, <4 x float> zeroinitializer) #9, !dbg !17
  %kdma.gep = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4096, !dbg !17
  %198 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8, !dbg !17
  %address.local497 = shl nuw nsw i32 %address.thread428, 1, !dbg !17
  %address.local498 = and i32 %address.local497, 24, !dbg !17
  %address.local499 = shl nuw nsw i32 %address.thread428, 8, !dbg !17
  %address.local500 = and i32 %address.local499, 768, !dbg !17
  %address.local501 = or disjoint i32 %address.local498, %address.local500, !dbg !17
  %address.local502 = shl nuw nsw i32 %address.thread428, 5, !dbg !17
  %address.local503 = and i32 %address.local502, 96, !dbg !17
  %address.local504 = or disjoint i32 %address.local501, %address.local503, !dbg !17
  %address.local505 = zext nneg i32 %address.local504 to i64, !dbg !17
  %address.base = call i64 asm sideeffect "", "=v,0"(i64 %117) #9, !dbg !17, !frisk.register_lifetimes !1
  %address.local506 = lshr i32 %address.thread428, 1, !dbg !17
  %address.local507 = and i32 %address.local506, 1, !dbg !17
  %address.local508 = zext nneg i32 %address.local507 to i64, !dbg !17
  %address.local509 = add nsw i64 %address.base, %address.local508, !dbg !17
  %address.local510 = shl i64 %address.local509, 6, !dbg !17
  %address.local511 = sub i64 %address.local505, %address.local510, !dbg !17
  %199 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local511, !dbg !17
  %200 = getelementptr [2 x i8], ptr addrspace(3) %199, i64 %address.local480, !dbg !17
  %201 = getelementptr [2 x i8], ptr addrspace(3) %200, i64 %address.local484, !dbg !17
  %address.local515 = getelementptr i8, ptr addrspace(3) %201, i64 64, !dbg !17
  %.idx19661969 = add i64 %address.local511, %address.local480, !dbg !17
  %202 = shl i64 %.idx19661969, 1, !dbg !17
  %203 = add i64 %202, %address.local486.idx, !dbg !17
  %204 = add i64 %203, 64, !dbg !17
  %205 = lshr i64 %204, 6, !dbg !17
  %206 = and i64 %205, 8, !dbg !17
  %207 = xor i64 %204, %206, !dbg !17
  %208 = sub i64 %207, %204, !dbg !17
  %kdma.swizzled1837 = getelementptr i8, ptr addrspace(3) %address.local515, i64 %208, !dbg !17
  %209 = load <4 x half>, ptr addrspace(3) %kdma.swizzled1837, align 8, !dbg !17, !frisk.k_lds_pair !36
  %kdma.gep1841 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1837, i64 128, !dbg !17
  %210 = load <4 x half>, ptr addrspace(3) %kdma.gep1841, align 8, !dbg !17, !frisk.k_lds_pair !36
  %kdma.gep1838 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1837, i64 4096, !dbg !17
  %211 = load <4 x half>, ptr addrspace(3) %kdma.gep1838, align 8, !dbg !17, !frisk.k_lds_pair !37
  %kdma.gep1842 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1837, i64 4224, !dbg !17
  %212 = load <4 x half>, ptr addrspace(3) %kdma.gep1842, align 8, !dbg !17, !frisk.k_lds_pair !37
  %kdma.gep1839 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 128, !dbg !17
  %213 = load <4 x half>, ptr addrspace(3) %kdma.gep1839, align 8, !dbg !17, !frisk.k_lds_pair !38
  %kdma.gep1843 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1024, !dbg !17
  %214 = load <4 x half>, ptr addrspace(3) %kdma.gep1843, align 8, !dbg !17, !frisk.k_lds_pair !38
  %kdma.gep1840 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4224, !dbg !17
  %215 = load <4 x half>, ptr addrspace(3) %kdma.gep1840, align 8, !dbg !17, !frisk.k_lds_pair !39
  %kdma.gep1844 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5120, !dbg !17
  %216 = load <4 x half>, ptr addrspace(3) %kdma.gep1844, align 8, !dbg !17, !frisk.k_lds_pair !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_lds_prefetch !1
  %217 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %198, <4 x float> %197) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_lds_prefetch !1
  %kdma.gep1845 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1837, i64 1024, !dbg !17
  %218 = load <4 x half>, ptr addrspace(3) %kdma.gep1845, align 8, !dbg !17, !frisk.k_lds_pair !34
  %kdma.gep1849 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1837, i64 1152, !dbg !17
  %219 = load <4 x half>, ptr addrspace(3) %kdma.gep1849, align 8, !dbg !17, !frisk.k_lds_pair !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_lds_prefetch !1
  %220 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %209, <4 x float> zeroinitializer) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_lds_prefetch !1
  %kdma.gep1846 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1837, i64 5120, !dbg !17
  %221 = load <4 x half>, ptr addrspace(3) %kdma.gep1846, align 8, !dbg !17, !frisk.k_lds_pair !40
  %kdma.gep1850 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1837, i64 5248, !dbg !17
  %222 = load <4 x half>, ptr addrspace(3) %kdma.gep1850, align 8, !dbg !17, !frisk.k_lds_pair !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_lds_prefetch !1
  %223 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %211, <4 x float> %220) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_lds_prefetch !1
  %kdma.gep1847 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1152, !dbg !17
  %224 = load <4 x half>, ptr addrspace(3) %kdma.gep1847, align 8, !dbg !17, !frisk.k_lds_pair !41
  %kdma.gep1848 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5248, !dbg !17
  %225 = load <4 x half>, ptr addrspace(3) %kdma.gep1848, align 8, !dbg !17, !frisk.k_lds_pair !41
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
  %.idx13471351 = add nuw nsw i64 %270, %139, !dbg !16
  %271 = add i64 %.idx13471351, %268, !dbg !16
  %272 = add i64 %271, %146, !dbg !16
  %kdma.source1662 = add i64 %272, %kdma.source1475, !dbg !16
  %kdma.source1663 = trunc i64 %kdma.source1662 to i32, !dbg !16
  %kdma.source1664 = shl i32 %kdma.source1663, 1, !dbg !16
  %kdma.source1665 = add i32 %kdma.source1664, 128, !dbg !16
  %kdma.offset1666 = add i32 %kdma.source1665, %kdma.intra, !dbg !16
  %273 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %49, i32 %kdma.offset1666, <4 x i32> %dma.buffer.rsrc1293) #10, !dbg !16
  %kdma.source1685 = add i64 %272, %kdma.source1495, !dbg !16
  %kdma.source1686 = trunc i64 %kdma.source1685 to i32, !dbg !16
  %kdma.source1687 = shl i32 %kdma.source1686, 1, !dbg !16
  %kdma.source1688 = add i32 %kdma.source1687, 128, !dbg !16
  %kdma.offset1689 = add i32 %kdma.source1688, %kdma.intra.hi, !dbg !16
  %274 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1500, i32 %kdma.offset1689, <4 x i32> %dma.buffer.rsrc1293) #10, !dbg !16
  %kdma.source1710 = add i64 %272, %kdma.source1519, !dbg !16
  %kdma.source1711 = trunc i64 %kdma.source1710 to i32, !dbg !16
  %kdma.source1712 = shl i32 %kdma.source1711, 1, !dbg !16
  %kdma.source1713 = add i32 %kdma.source1712, 128, !dbg !16
  %kdma.offset1714 = add i32 %kdma.source1713, %kdma.intra, !dbg !16
  %275 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %57, i32 %kdma.offset1714, <4 x i32> %dma.buffer.rsrc1293) #10, !dbg !16
  %kdma.source1735 = add i64 %272, %kdma.source1543, !dbg !16
  %kdma.source1736 = trunc i64 %kdma.source1735 to i32, !dbg !16
  %kdma.source1737 = shl i32 %kdma.source1736, 1, !dbg !16
  %kdma.source1738 = add i32 %kdma.source1737, 128, !dbg !16
  %kdma.offset1739 = add i32 %kdma.source1738, %kdma.intra.hi, !dbg !16
  %276 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1548, i32 %kdma.offset1739, <4 x i32> %dma.buffer.rsrc1293) #10, !dbg !16
  %kdma.gep1867 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8192, !dbg !17
  %277 = load <4 x half>, ptr addrspace(3) %kdma.gep1867, align 8, !dbg !17, !frisk.k_lds_pair !42
  %kdma.gep1871 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8320, !dbg !17
  %278 = load <4 x half>, ptr addrspace(3) %kdma.gep1871, align 8, !dbg !17, !frisk.k_lds_pair !42
  %kdma.gep1868 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12288, !dbg !17
  %279 = load <4 x half>, ptr addrspace(3) %kdma.gep1868, align 8, !dbg !17, !frisk.k_lds_pair !43
  %kdma.gep1872 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12416, !dbg !17
  %280 = load <4 x half>, ptr addrspace(3) %kdma.gep1872, align 8, !dbg !17, !frisk.k_lds_pair !43
  %kdma.gep1869 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1837, i64 8192, !dbg !17
  %281 = load <4 x half>, ptr addrspace(3) %kdma.gep1869, align 8, !dbg !17, !frisk.k_lds_pair !44
  %kdma.gep1873 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1837, i64 8320, !dbg !17
  %282 = load <4 x half>, ptr addrspace(3) %kdma.gep1873, align 8, !dbg !17, !frisk.k_lds_pair !44
  %kdma.gep1870 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1837, i64 12288, !dbg !17
  %283 = load <4 x half>, ptr addrspace(3) %kdma.gep1870, align 8, !dbg !17, !frisk.k_lds_pair !45
  %kdma.gep1874 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1837, i64 12416, !dbg !17
  %284 = load <4 x half>, ptr addrspace(3) %kdma.gep1874, align 8, !dbg !17, !frisk.k_lds_pair !45
  %kdma.gep1875 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9216, !dbg !17
  %285 = load <4 x half>, ptr addrspace(3) %kdma.gep1875, align 8, !dbg !17, !frisk.k_lds_pair !46
  %kdma.gep1879 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9344, !dbg !17
  %286 = load <4 x half>, ptr addrspace(3) %kdma.gep1879, align 8, !dbg !17, !frisk.k_lds_pair !46
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_lds_prefetch !1
  %287 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %277, <4 x float> %217) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_lds_prefetch !1
  %kdma.gep1876 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13312, !dbg !17
  %288 = load <4 x half>, ptr addrspace(3) %kdma.gep1876, align 8, !dbg !17, !frisk.k_lds_pair !47
  %kdma.gep1880 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13440, !dbg !17
  %289 = load <4 x half>, ptr addrspace(3) %kdma.gep1880, align 8, !dbg !17, !frisk.k_lds_pair !47
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_lds_prefetch !1
  %290 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %279, <4 x float> %287) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_lds_prefetch !1
  %kdma.gep1877 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1837, i64 9216, !dbg !17
  %291 = load <4 x half>, ptr addrspace(3) %kdma.gep1877, align 8, !dbg !17, !frisk.k_lds_pair !48
  %kdma.gep1881 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1837, i64 9344, !dbg !17
  %292 = load <4 x half>, ptr addrspace(3) %kdma.gep1881, align 8, !dbg !17, !frisk.k_lds_pair !48
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_lds_prefetch !1
  %293 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %281, <4 x float> %223) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_lds_prefetch !1
  %kdma.gep1878 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1837, i64 13312, !dbg !17
  %294 = load <4 x half>, ptr addrspace(3) %kdma.gep1878, align 8, !dbg !17, !frisk.k_lds_pair !49
  %kdma.gep1882 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1837, i64 13440, !dbg !17
  %295 = load <4 x half>, ptr addrspace(3) %kdma.gep1882, align 8, !dbg !17, !frisk.k_lds_pair !49
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
  %.idx13591363 = add nuw nsw i64 %341, %139, !dbg !16
  %342 = add i64 %.idx13591363, %339, !dbg !16
  %343 = add i64 %342, %146, !dbg !16
  %kdma.source1758 = add i64 %343, %kdma.source1475, !dbg !16
  %kdma.source1759 = trunc i64 %kdma.source1758 to i32, !dbg !16
  %kdma.source1760 = shl i32 %kdma.source1759, 1, !dbg !16
  %kdma.source1761 = add i32 %kdma.source1760, 192, !dbg !16
  %kdma.offset1762 = add i32 %kdma.source1761, %kdma.intra, !dbg !16
  %344 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %62, i32 %kdma.offset1762, <4 x i32> %dma.buffer.rsrc1293) #10, !dbg !16
  %kdma.source1781 = add i64 %343, %kdma.source1495, !dbg !16
  %kdma.source1782 = trunc i64 %kdma.source1781 to i32, !dbg !16
  %kdma.source1783 = shl i32 %kdma.source1782, 1, !dbg !16
  %kdma.source1784 = add i32 %kdma.source1783, 192, !dbg !16
  %kdma.offset1785 = add i32 %kdma.source1784, %kdma.intra.hi, !dbg !16
  %345 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1594, i32 %kdma.offset1785, <4 x i32> %dma.buffer.rsrc1293) #10, !dbg !16
  %kdma.source1806 = add i64 %343, %kdma.source1519, !dbg !16
  %kdma.source1807 = trunc i64 %kdma.source1806 to i32, !dbg !16
  %kdma.source1808 = shl i32 %kdma.source1807, 1, !dbg !16
  %kdma.source1809 = add i32 %kdma.source1808, 192, !dbg !16
  %kdma.offset1810 = add i32 %kdma.source1809, %kdma.intra, !dbg !16
  %346 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %66, i32 %kdma.offset1810, <4 x i32> %dma.buffer.rsrc1293) #10, !dbg !16
  %kdma.source1831 = add i64 %343, %kdma.source1543, !dbg !16
  %kdma.source1832 = trunc i64 %kdma.source1831 to i32, !dbg !16
  %kdma.source1833 = shl i32 %kdma.source1832, 1, !dbg !16
  %kdma.source1834 = add i32 %kdma.source1833, 192, !dbg !16
  %kdma.offset1835 = add i32 %kdma.source1834, %kdma.intra.hi, !dbg !16
  %347 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1644, i32 %kdma.offset1835, <4 x i32> %dma.buffer.rsrc1293) #10, !dbg !16
  %348 = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8, !dbg !17, !frisk.k_lds_pair !50
  %349 = load <4 x half>, ptr addrspace(3) %kdma.gep1839, align 8, !dbg !17, !frisk.k_lds_pair !50
  %350 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8, !dbg !17, !frisk.k_lds_pair !51
  %351 = load <4 x half>, ptr addrspace(3) %kdma.gep1840, align 8, !dbg !17, !frisk.k_lds_pair !51
  %352 = load <4 x half>, ptr addrspace(3) %kdma.swizzled1837, align 8, !dbg !17, !frisk.k_lds_pair !52
  %353 = load <4 x half>, ptr addrspace(3) %kdma.gep1841, align 8, !dbg !17, !frisk.k_lds_pair !52
  %354 = load <4 x half>, ptr addrspace(3) %kdma.gep1838, align 8, !dbg !17, !frisk.k_lds_pair !53
  %355 = load <4 x half>, ptr addrspace(3) %kdma.gep1842, align 8, !dbg !17, !frisk.k_lds_pair !53
  %356 = load <4 x half>, ptr addrspace(3) %kdma.gep1843, align 8, !dbg !17, !frisk.k_lds_pair !54
  %357 = load <4 x half>, ptr addrspace(3) %kdma.gep1847, align 8, !dbg !17, !frisk.k_lds_pair !54
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_lds_prefetch !1
  %358 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %348, <4 x float> %290) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_lds_prefetch !1
  %359 = load <4 x half>, ptr addrspace(3) %kdma.gep1844, align 8, !dbg !17, !frisk.k_lds_pair !55
  %360 = load <4 x half>, ptr addrspace(3) %kdma.gep1848, align 8, !dbg !17, !frisk.k_lds_pair !55
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_lds_prefetch !1
  %361 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %350, <4 x float> %358) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_lds_prefetch !1
  %362 = load <4 x half>, ptr addrspace(3) %kdma.gep1845, align 8, !dbg !17, !frisk.k_lds_pair !56
  %363 = load <4 x half>, ptr addrspace(3) %kdma.gep1849, align 8, !dbg !17, !frisk.k_lds_pair !56
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_lds_prefetch !1
  %364 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %352, <4 x float> %296) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_lds_prefetch !1
  %365 = load <4 x half>, ptr addrspace(3) %kdma.gep1846, align 8, !dbg !17, !frisk.k_lds_pair !57
  %366 = load <4 x half>, ptr addrspace(3) %kdma.gep1850, align 8, !dbg !17, !frisk.k_lds_pair !57
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
  %address.local844 = shl nuw nsw i64 %address.local429, 5, !dbg !18
  %address.local845 = and i64 %address.local844, 8064, !dbg !18
  %address.local846 = shl nuw nsw i64 %address.local429, 3, !dbg !9
  %address.local847 = shl nuw nsw i64 %address.local435, 5, !dbg !18
  %address.local848 = sub nsw i64 %address.local846, %address.local847, !dbg !18
  %address.local849 = add nsw i64 %address.local845, %address.local848, !dbg !18
  %399 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %.idx13711373 = add i64 %398, %146, !dbg !18
  %400 = add i64 %.idx13711373, %address.local849, !dbg !18
  %.tr1374 = trunc i64 %400 to i32, !dbg !18
  %dma.buffer.offset1276 = shl i32 %.tr1374, 1, !dbg !18
  %401 = insertelement <2 x i64> poison, i64 %399, i64 0, !dbg !18
  %402 = bitcast <2 x i64> %401 to <4 x i32>, !dbg !18
  %dma.buffer.rsrc1277 = shufflevector <4 x i32> %402, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !18
  %403 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %49, i32 %dma.buffer.offset1276, <4 x i32> %dma.buffer.rsrc1277) #10, !dbg !18
  %address.local851 = shl nuw nsw i32 %address.thread428, 3, !dbg !18
  %address.local852 = and i32 %address.local851, 24, !dbg !18
  %address.local854 = and i32 %address.local502, 8064, !dbg !18
  %address.local855 = or disjoint i32 %address.local852, %address.local854, !dbg !18
  %404 = trunc i64 %.idx13711373 to i32, !dbg !18
  %address.local856 = or disjoint i32 %address.local855, %404, !dbg !18
  %.tr1378 = shl i32 %address.local856, 1, !dbg !18
  %dma.buffer.offset1274 = or i32 %.tr1378, 16384, !dbg !18
  %405 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %57, i32 %dma.buffer.offset1274, <4 x i32> %dma.buffer.rsrc1277) #10, !dbg !18
  %406 = load <4 x half>, ptr addrspace(3) %kdma.gep1867, align 8, !dbg !17, !frisk.k_tail_pair !36
  %407 = load <4 x half>, ptr addrspace(3) %kdma.gep1871, align 8, !dbg !17, !frisk.k_tail_pair !36
  %408 = load <4 x half>, ptr addrspace(3) %kdma.gep1868, align 8, !dbg !17, !frisk.k_tail_pair !37
  %409 = load <4 x half>, ptr addrspace(3) %kdma.gep1872, align 8, !dbg !17, !frisk.k_tail_pair !37
  %410 = load <4 x half>, ptr addrspace(3) %kdma.gep1869, align 8, !dbg !17, !frisk.k_tail_pair !38
  %411 = load <4 x half>, ptr addrspace(3) %kdma.gep1873, align 8, !dbg !17, !frisk.k_tail_pair !38
  %412 = load <4 x half>, ptr addrspace(3) %kdma.gep1870, align 8, !dbg !17, !frisk.k_tail_pair !39
  %413 = load <4 x half>, ptr addrspace(3) %kdma.gep1874, align 8, !dbg !17, !frisk.k_tail_pair !39
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
  %460 = load <4 x half>, ptr addrspace(3) %kdma.gep1875, align 8, !dbg !17, !frisk.k_tail_pair !34
  %461 = load <4 x half>, ptr addrspace(3) %kdma.gep1879, align 8, !dbg !17, !frisk.k_tail_pair !34
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
  %475 = load <4 x half>, ptr addrspace(3) %kdma.gep1876, align 8, !dbg !17, !frisk.k_tail_pair !40
  %476 = load <4 x half>, ptr addrspace(3) %kdma.gep1880, align 8, !dbg !17, !frisk.k_tail_pair !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_tail_prefetch !1
  %477 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %409, <4 x float> %462) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17, !frisk.k_tail_prefetch !1
  %478 = load <4 x half>, ptr addrspace(3) %kdma.gep1877, align 8, !dbg !17, !frisk.k_tail_pair !41
  %479 = load <4 x half>, ptr addrspace(3) %kdma.gep1881, align 8, !dbg !17, !frisk.k_tail_pair !41
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
  %493 = load <4 x half>, ptr addrspace(3) %kdma.gep1878, align 8, !dbg !17, !frisk.k_tail_pair !42
  %494 = load <4 x half>, ptr addrspace(3) %kdma.gep1882, align 8, !dbg !17, !frisk.k_tail_pair !42
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
  %vdb.address1394 = or disjoint i64 %.idx13711373, 32, !dbg !18
  %vdb.address1395 = add i64 %address.local849, %vdb.address1394, !dbg !18
  %vdb.address1396 = trunc i64 %vdb.address1395 to i32, !dbg !18
  %vdb.address1397 = shl i32 %vdb.address1396, 1, !dbg !18
  %vdb.address1405 = or i32 %.tr1378, 16448, !dbg !18
  %vdb.dma.base = add i32 %49, 8192, !dbg !14
  %793 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1397, <4 x i32> %dma.buffer.rsrc1277) #10, !dbg !18
  %vdb.dma.base1410 = add i32 %57, 8192, !dbg !14
  %794 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1410, i32 %vdb.address1405, <4 x i32> %dma.buffer.rsrc1277) #10, !dbg !18
  %address.local860 = and i32 %address.local851, 120, !dbg !14
  %address.local862 = and i32 %address.local502, 1536, !dbg !14
  %address.local863 = or disjoint i32 %address.local860, %address.local862, !dbg !14
  %address.local864 = zext nneg i32 %address.local863 to i64, !dbg !14
  %address.local865 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local864, !dbg !14
  %795 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local865, i32 0) #0, !dbg !14, !frisk.v_double_buffer !35
  %796 = bitcast <4 x i32> %795 to <8 x half>, !dbg !14
  %address.local866 = and i64 %address.local429, 60, !dbg !14
  %address.local867 = add nuw nsw i64 %address.local866, %address.local429, !dbg !14
  %address.local869.idx = shl nuw nsw i64 %address.local867, 4, !dbg !14
  %address.local869 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %address.local869.idx, !dbg !14
  %address.local870 = lshr i64 %address.local482, 4, !dbg !14
  %address.local872.idx = mul nuw nsw i64 %address.local870, 768, !dbg !14
  %address.local872 = getelementptr i8, ptr addrspace(3) %address.local869, i64 %address.local872.idx, !dbg !14
  %address.local874.idx = mul nsw i64 %address.local435, -64, !dbg !14
  %address.local874 = getelementptr i8, ptr addrspace(3) %address.local872, i64 %address.local874.idx, !dbg !14
  %address.local875 = getelementptr i8, ptr addrspace(3) %address.local874, i64 256, !dbg !14
  %797 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local875, i32 0) #0, !dbg !14, !frisk.v_double_buffer !35
  %798 = bitcast <4 x i32> %797 to <8 x half>, !dbg !14
  %address.local879 = getelementptr i8, ptr addrspace(3) %address.local874, i64 512, !dbg !14
  %799 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local879, i32 0) #0, !dbg !14, !frisk.v_double_buffer !35
  %800 = bitcast <4 x i32> %799 to <8 x half>, !dbg !14
  %address.local883 = getelementptr i8, ptr addrspace(3) %address.local874, i64 768, !dbg !14
  %801 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local883, i32 0) #0, !dbg !14, !frisk.v_double_buffer !35
  %802 = bitcast <4 x i32> %801 to <8 x half>, !dbg !14
  %address.local885 = getelementptr i8, ptr addrspace(3) %address.local865, i64 4096, !dbg !14
  %803 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local885, i32 0) #0, !dbg !14, !frisk.v_double_buffer !35
  %804 = bitcast <4 x i32> %803 to <8 x half>, !dbg !14
  %address.local889 = getelementptr i8, ptr addrspace(3) %address.local874, i64 4352, !dbg !14
  %805 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local889, i32 0) #0, !dbg !14, !frisk.v_double_buffer !35
  %806 = bitcast <4 x i32> %805 to <8 x half>, !dbg !14
  %address.local893 = getelementptr i8, ptr addrspace(3) %address.local874, i64 4608, !dbg !14
  %807 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local893, i32 0) #0, !dbg !14, !frisk.v_double_buffer !35
  %808 = bitcast <4 x i32> %807 to <8 x half>, !dbg !14
  %address.local897 = getelementptr i8, ptr addrspace(3) %address.local874, i64 4864, !dbg !14
  %809 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local897, i32 0) #0, !dbg !14, !frisk.v_double_buffer !35
  %810 = shufflevector <8 x half> %796, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %811 = fsub <4 x float> %704, %705, !dbg !23
  %812 = extractelement <4 x float> %811, i64 0, !dbg !27
  %813 = call float @llvm.amdgcn.exp2.f32(float %812), !dbg !27
  %814 = insertelement <4 x float> poison, float %813, i64 0, !dbg !29
  %815 = extractelement <4 x float> %811, i64 1, !dbg !27
  %816 = call float @llvm.amdgcn.exp2.f32(float %815), !dbg !27
  %817 = insertelement <4 x float> %814, float %816, i64 1, !dbg !29
  %818 = extractelement <4 x float> %811, i64 2, !dbg !27
  %819 = call float @llvm.amdgcn.exp2.f32(float %818), !dbg !27
  %820 = insertelement <4 x float> %817, float %819, i64 2, !dbg !29
  %821 = extractelement <4 x float> %811, i64 3, !dbg !27
  %822 = call float @llvm.amdgcn.exp2.f32(float %821), !dbg !27
  %823 = insertelement <4 x float> %820, float %822, i64 3, !dbg !29
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %823) #9, !dbg !29, !frisk.register_lifetimes !1
  %.cvtpack.f0 = extractelement <4 x float> %math.result, i32 0, !dbg !29
  %.cvtpack.f1 = extractelement <4 x float> %math.result, i32 1, !dbg !29
  %.cvtpack.f2 = extractelement <4 x float> %math.result, i32 2, !dbg !29
  %.cvtpack.f3 = extractelement <4 x float> %math.result, i32 3, !dbg !29
  %.cvtpack.p0 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f0, float %.cvtpack.f1), !dbg !29
  %.cvtpack.p1 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f2, float %.cvtpack.f3), !dbg !29
  %.cvtpack.v0 = insertelement <2 x i32> poison, i32 %.cvtpack.p0, i32 0, !dbg !29
  %.cvtpack.v1 = insertelement <2 x i32> %.cvtpack.v0, i32 %.cvtpack.p1, i32 1, !dbg !29
  %824 = bitcast <2 x i32> %.cvtpack.v1 to <4 x half>, !dbg !29
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %824) #9, !dbg !27, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %825 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %810, <4 x float> %785) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %826 = shufflevector <8 x half> %796, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %827 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %826, <4 x float> %787) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %828 = fsub <4 x float> %737, %738, !dbg !23
  %829 = extractelement <4 x float> %828, i64 0, !dbg !27
  %830 = call float @llvm.amdgcn.exp2.f32(float %829), !dbg !27
  %831 = insertelement <4 x float> poison, float %830, i64 0, !dbg !29
  %832 = extractelement <4 x float> %828, i64 1, !dbg !27
  %833 = call float @llvm.amdgcn.exp2.f32(float %832), !dbg !27
  %834 = insertelement <4 x float> %831, float %833, i64 1, !dbg !29
  %835 = extractelement <4 x float> %828, i64 2, !dbg !27
  %836 = call float @llvm.amdgcn.exp2.f32(float %835), !dbg !27
  %837 = insertelement <4 x float> %834, float %836, i64 2, !dbg !29
  %838 = extractelement <4 x float> %828, i64 3, !dbg !27
  %839 = call float @llvm.amdgcn.exp2.f32(float %838), !dbg !27
  %840 = insertelement <4 x float> %837, float %839, i64 3, !dbg !29
  %math.result315 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %840) #9, !dbg !29, !frisk.register_lifetimes !1
  %.cvtpack.f01970 = extractelement <4 x float> %math.result315, i32 0, !dbg !29
  %.cvtpack.f11971 = extractelement <4 x float> %math.result315, i32 1, !dbg !29
  %.cvtpack.f21972 = extractelement <4 x float> %math.result315, i32 2, !dbg !29
  %.cvtpack.f31973 = extractelement <4 x float> %math.result315, i32 3, !dbg !29
  %.cvtpack.p01974 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f01970, float %.cvtpack.f11971), !dbg !29
  %.cvtpack.p11975 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f21972, float %.cvtpack.f31973), !dbg !29
  %.cvtpack.v01976 = insertelement <2 x i32> poison, i32 %.cvtpack.p01974, i32 0, !dbg !29
  %.cvtpack.v11977 = insertelement <2 x i32> %.cvtpack.v01976, i32 %.cvtpack.p11975, i32 1, !dbg !29
  %841 = bitcast <2 x i32> %.cvtpack.v11977 to <4 x half>, !dbg !29
  %math.packed316 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %841) #9, !dbg !27, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %842 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %810, <4 x float> %790) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %843 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %826, <4 x float> %792) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %844 = shufflevector <8 x half> %798, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %845 = fsub <4 x float> %709, %705, !dbg !23
  %846 = extractelement <4 x float> %845, i64 0, !dbg !27
  %847 = call float @llvm.amdgcn.exp2.f32(float %846), !dbg !27
  %848 = fadd float %813, %847, !dbg !28
  %849 = insertelement <4 x float> poison, float %847, i64 0, !dbg !29
  %850 = extractelement <4 x float> %845, i64 1, !dbg !27
  %851 = call float @llvm.amdgcn.exp2.f32(float %850), !dbg !27
  %852 = fadd float %816, %851, !dbg !28
  %853 = insertelement <4 x float> %849, float %851, i64 1, !dbg !29
  %854 = extractelement <4 x float> %845, i64 2, !dbg !27
  %855 = call float @llvm.amdgcn.exp2.f32(float %854), !dbg !27
  %856 = fadd float %819, %855, !dbg !28
  %857 = insertelement <4 x float> %853, float %855, i64 2, !dbg !29
  %858 = extractelement <4 x float> %845, i64 3, !dbg !27
  %859 = call float @llvm.amdgcn.exp2.f32(float %858), !dbg !27
  %860 = fadd float %822, %859, !dbg !28
  %861 = insertelement <4 x float> %857, float %859, i64 3, !dbg !29
  %math.result301 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %861) #9, !dbg !29, !frisk.register_lifetimes !1
  %.cvtpack.f01978 = extractelement <4 x float> %math.result301, i32 0, !dbg !29
  %.cvtpack.f11979 = extractelement <4 x float> %math.result301, i32 1, !dbg !29
  %.cvtpack.f21980 = extractelement <4 x float> %math.result301, i32 2, !dbg !29
  %.cvtpack.f31981 = extractelement <4 x float> %math.result301, i32 3, !dbg !29
  %.cvtpack.p01982 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f01978, float %.cvtpack.f11979), !dbg !29
  %.cvtpack.p11983 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f21980, float %.cvtpack.f31981), !dbg !29
  %.cvtpack.v01984 = insertelement <2 x i32> poison, i32 %.cvtpack.p01982, i32 0, !dbg !29
  %.cvtpack.v11985 = insertelement <2 x i32> %.cvtpack.v01984, i32 %.cvtpack.p11983, i32 1, !dbg !29
  %862 = bitcast <2 x i32> %.cvtpack.v11985 to <4 x half>, !dbg !29
  %math.packed302 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %862) #9, !dbg !28, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %863 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed302, <4 x half> %844, <4 x float> %825) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %864 = shufflevector <8 x half> %798, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %865 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed302, <4 x half> %864, <4 x float> %827) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %866 = fsub <4 x float> %742, %738, !dbg !23
  %867 = extractelement <4 x float> %866, i64 0, !dbg !27
  %868 = call float @llvm.amdgcn.exp2.f32(float %867), !dbg !27
  %869 = fadd float %830, %868, !dbg !28
  %870 = insertelement <4 x float> poison, float %868, i64 0, !dbg !29
  %871 = extractelement <4 x float> %866, i64 1, !dbg !27
  %872 = call float @llvm.amdgcn.exp2.f32(float %871), !dbg !27
  %873 = fadd float %833, %872, !dbg !28
  %874 = insertelement <4 x float> %870, float %872, i64 1, !dbg !29
  %875 = extractelement <4 x float> %866, i64 2, !dbg !27
  %876 = call float @llvm.amdgcn.exp2.f32(float %875), !dbg !27
  %877 = fadd float %836, %876, !dbg !28
  %878 = insertelement <4 x float> %874, float %876, i64 2, !dbg !29
  %879 = extractelement <4 x float> %866, i64 3, !dbg !27
  %880 = call float @llvm.amdgcn.exp2.f32(float %879), !dbg !27
  %881 = fadd float %839, %880, !dbg !28
  %882 = insertelement <4 x float> %878, float %880, i64 3, !dbg !29
  %math.result317 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %882) #9, !dbg !29, !frisk.register_lifetimes !1
  %.cvtpack.f01986 = extractelement <4 x float> %math.result317, i32 0, !dbg !29
  %.cvtpack.f11987 = extractelement <4 x float> %math.result317, i32 1, !dbg !29
  %.cvtpack.f21988 = extractelement <4 x float> %math.result317, i32 2, !dbg !29
  %.cvtpack.f31989 = extractelement <4 x float> %math.result317, i32 3, !dbg !29
  %.cvtpack.p01990 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f01986, float %.cvtpack.f11987), !dbg !29
  %.cvtpack.p11991 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f21988, float %.cvtpack.f31989), !dbg !29
  %.cvtpack.v01992 = insertelement <2 x i32> poison, i32 %.cvtpack.p01990, i32 0, !dbg !29
  %.cvtpack.v11993 = insertelement <2 x i32> %.cvtpack.v01992, i32 %.cvtpack.p11991, i32 1, !dbg !29
  %883 = bitcast <2 x i32> %.cvtpack.v11993 to <4 x half>, !dbg !29
  %math.packed318 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %883) #9, !dbg !28, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %884 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %844, <4 x float> %842) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %885 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %864, <4 x float> %843) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %886 = shufflevector <8 x half> %800, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %887 = fsub <4 x float> %713, %705, !dbg !23
  %888 = extractelement <4 x float> %887, i64 0, !dbg !27
  %889 = call float @llvm.amdgcn.exp2.f32(float %888), !dbg !27
  %890 = fadd float %848, %889, !dbg !28
  %891 = insertelement <4 x float> poison, float %889, i64 0, !dbg !29
  %892 = extractelement <4 x float> %887, i64 1, !dbg !27
  %893 = call float @llvm.amdgcn.exp2.f32(float %892), !dbg !27
  %894 = fadd float %852, %893, !dbg !28
  %895 = insertelement <4 x float> %891, float %893, i64 1, !dbg !29
  %896 = extractelement <4 x float> %887, i64 2, !dbg !27
  %897 = call float @llvm.amdgcn.exp2.f32(float %896), !dbg !27
  %898 = fadd float %856, %897, !dbg !28
  %899 = insertelement <4 x float> %895, float %897, i64 2, !dbg !29
  %900 = extractelement <4 x float> %887, i64 3, !dbg !27
  %901 = call float @llvm.amdgcn.exp2.f32(float %900), !dbg !27
  %902 = fadd float %860, %901, !dbg !28
  %903 = insertelement <4 x float> %899, float %901, i64 3, !dbg !29
  %math.result303 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %903) #9, !dbg !29, !frisk.register_lifetimes !1
  %.cvtpack.f01994 = extractelement <4 x float> %math.result303, i32 0, !dbg !29
  %.cvtpack.f11995 = extractelement <4 x float> %math.result303, i32 1, !dbg !29
  %.cvtpack.f21996 = extractelement <4 x float> %math.result303, i32 2, !dbg !29
  %.cvtpack.f31997 = extractelement <4 x float> %math.result303, i32 3, !dbg !29
  %.cvtpack.p01998 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f01994, float %.cvtpack.f11995), !dbg !29
  %.cvtpack.p11999 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f21996, float %.cvtpack.f31997), !dbg !29
  %.cvtpack.v02000 = insertelement <2 x i32> poison, i32 %.cvtpack.p01998, i32 0, !dbg !29
  %.cvtpack.v12001 = insertelement <2 x i32> %.cvtpack.v02000, i32 %.cvtpack.p11999, i32 1, !dbg !29
  %904 = bitcast <2 x i32> %.cvtpack.v12001 to <4 x half>, !dbg !29
  %math.packed304 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %904) #9, !dbg !28, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %905 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed304, <4 x half> %886, <4 x float> %863) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %906 = shufflevector <8 x half> %800, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %907 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed304, <4 x half> %906, <4 x float> %865) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %908 = fsub <4 x float> %746, %738, !dbg !23
  %909 = extractelement <4 x float> %908, i64 0, !dbg !27
  %910 = call float @llvm.amdgcn.exp2.f32(float %909), !dbg !27
  %911 = fadd float %869, %910, !dbg !28
  %912 = insertelement <4 x float> poison, float %910, i64 0, !dbg !29
  %913 = extractelement <4 x float> %908, i64 1, !dbg !27
  %914 = call float @llvm.amdgcn.exp2.f32(float %913), !dbg !27
  %915 = fadd float %873, %914, !dbg !28
  %916 = insertelement <4 x float> %912, float %914, i64 1, !dbg !29
  %917 = extractelement <4 x float> %908, i64 2, !dbg !27
  %918 = call float @llvm.amdgcn.exp2.f32(float %917), !dbg !27
  %919 = fadd float %877, %918, !dbg !28
  %920 = insertelement <4 x float> %916, float %918, i64 2, !dbg !29
  %921 = extractelement <4 x float> %908, i64 3, !dbg !27
  %922 = call float @llvm.amdgcn.exp2.f32(float %921), !dbg !27
  %923 = fadd float %881, %922, !dbg !28
  %924 = insertelement <4 x float> %920, float %922, i64 3, !dbg !29
  %math.result319 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %924) #9, !dbg !29, !frisk.register_lifetimes !1
  %.cvtpack.f02002 = extractelement <4 x float> %math.result319, i32 0, !dbg !29
  %.cvtpack.f12003 = extractelement <4 x float> %math.result319, i32 1, !dbg !29
  %.cvtpack.f22004 = extractelement <4 x float> %math.result319, i32 2, !dbg !29
  %.cvtpack.f32005 = extractelement <4 x float> %math.result319, i32 3, !dbg !29
  %.cvtpack.p02006 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02002, float %.cvtpack.f12003), !dbg !29
  %.cvtpack.p12007 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22004, float %.cvtpack.f32005), !dbg !29
  %.cvtpack.v02008 = insertelement <2 x i32> poison, i32 %.cvtpack.p02006, i32 0, !dbg !29
  %.cvtpack.v12009 = insertelement <2 x i32> %.cvtpack.v02008, i32 %.cvtpack.p12007, i32 1, !dbg !29
  %925 = bitcast <2 x i32> %.cvtpack.v12009 to <4 x half>, !dbg !29
  %math.packed320 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %925) #9, !dbg !28, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %926 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %886, <4 x float> %884) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %927 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %906, <4 x float> %885) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %928 = shufflevector <8 x half> %802, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %929 = fsub <4 x float> %717, %705, !dbg !23
  %930 = extractelement <4 x float> %929, i64 0, !dbg !27
  %931 = call float @llvm.amdgcn.exp2.f32(float %930), !dbg !27
  %932 = fadd float %890, %931, !dbg !28
  %933 = insertelement <4 x float> poison, float %931, i64 0, !dbg !29
  %934 = extractelement <4 x float> %929, i64 1, !dbg !27
  %935 = call float @llvm.amdgcn.exp2.f32(float %934), !dbg !27
  %936 = fadd float %894, %935, !dbg !28
  %937 = insertelement <4 x float> %933, float %935, i64 1, !dbg !29
  %938 = extractelement <4 x float> %929, i64 2, !dbg !27
  %939 = call float @llvm.amdgcn.exp2.f32(float %938), !dbg !27
  %940 = fadd float %898, %939, !dbg !28
  %941 = insertelement <4 x float> %937, float %939, i64 2, !dbg !29
  %942 = extractelement <4 x float> %929, i64 3, !dbg !27
  %943 = call float @llvm.amdgcn.exp2.f32(float %942), !dbg !27
  %944 = fadd float %902, %943, !dbg !28
  %945 = insertelement <4 x float> %941, float %943, i64 3, !dbg !29
  %math.result305 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %945) #9, !dbg !29, !frisk.register_lifetimes !1
  %.cvtpack.f02010 = extractelement <4 x float> %math.result305, i32 0, !dbg !29
  %.cvtpack.f12011 = extractelement <4 x float> %math.result305, i32 1, !dbg !29
  %.cvtpack.f22012 = extractelement <4 x float> %math.result305, i32 2, !dbg !29
  %.cvtpack.f32013 = extractelement <4 x float> %math.result305, i32 3, !dbg !29
  %.cvtpack.p02014 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02010, float %.cvtpack.f12011), !dbg !29
  %.cvtpack.p12015 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22012, float %.cvtpack.f32013), !dbg !29
  %.cvtpack.v02016 = insertelement <2 x i32> poison, i32 %.cvtpack.p02014, i32 0, !dbg !29
  %.cvtpack.v12017 = insertelement <2 x i32> %.cvtpack.v02016, i32 %.cvtpack.p12015, i32 1, !dbg !29
  %946 = bitcast <2 x i32> %.cvtpack.v12017 to <4 x half>, !dbg !29
  %math.packed306 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %946) #9, !dbg !28, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %947 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed306, <4 x half> %928, <4 x float> %905) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %948 = shufflevector <8 x half> %802, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %949 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed306, <4 x half> %948, <4 x float> %907) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %950 = fsub <4 x float> %750, %738, !dbg !23
  %951 = extractelement <4 x float> %950, i64 0, !dbg !27
  %952 = call float @llvm.amdgcn.exp2.f32(float %951), !dbg !27
  %953 = fadd float %911, %952, !dbg !28
  %954 = insertelement <4 x float> poison, float %952, i64 0, !dbg !29
  %955 = extractelement <4 x float> %950, i64 1, !dbg !27
  %956 = call float @llvm.amdgcn.exp2.f32(float %955), !dbg !27
  %957 = fadd float %915, %956, !dbg !28
  %958 = insertelement <4 x float> %954, float %956, i64 1, !dbg !29
  %959 = extractelement <4 x float> %950, i64 2, !dbg !27
  %960 = call float @llvm.amdgcn.exp2.f32(float %959), !dbg !27
  %961 = fadd float %919, %960, !dbg !28
  %962 = insertelement <4 x float> %958, float %960, i64 2, !dbg !29
  %963 = extractelement <4 x float> %950, i64 3, !dbg !27
  %964 = call float @llvm.amdgcn.exp2.f32(float %963), !dbg !27
  %965 = fadd float %923, %964, !dbg !28
  %966 = insertelement <4 x float> %962, float %964, i64 3, !dbg !29
  %math.result321 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %966) #9, !dbg !29, !frisk.register_lifetimes !1
  %.cvtpack.f02018 = extractelement <4 x float> %math.result321, i32 0, !dbg !29
  %.cvtpack.f12019 = extractelement <4 x float> %math.result321, i32 1, !dbg !29
  %.cvtpack.f22020 = extractelement <4 x float> %math.result321, i32 2, !dbg !29
  %.cvtpack.f32021 = extractelement <4 x float> %math.result321, i32 3, !dbg !29
  %.cvtpack.p02022 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02018, float %.cvtpack.f12019), !dbg !29
  %.cvtpack.p12023 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22020, float %.cvtpack.f32021), !dbg !29
  %.cvtpack.v02024 = insertelement <2 x i32> poison, i32 %.cvtpack.p02022, i32 0, !dbg !29
  %.cvtpack.v12025 = insertelement <2 x i32> %.cvtpack.v02024, i32 %.cvtpack.p12023, i32 1, !dbg !29
  %967 = bitcast <2 x i32> %.cvtpack.v12025 to <4 x half>, !dbg !29
  %math.packed322 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %967) #9, !dbg !28, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %968 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %928, <4 x float> %926) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %969 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %948, <4 x float> %927) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %970 = shufflevector <8 x half> %804, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %971 = fsub <4 x float> %721, %705, !dbg !23
  %972 = extractelement <4 x float> %971, i64 0, !dbg !27
  %973 = call float @llvm.amdgcn.exp2.f32(float %972), !dbg !27
  %974 = fadd float %932, %973, !dbg !28
  %975 = insertelement <4 x float> poison, float %973, i64 0, !dbg !29
  %976 = extractelement <4 x float> %971, i64 1, !dbg !27
  %977 = call float @llvm.amdgcn.exp2.f32(float %976), !dbg !27
  %978 = fadd float %936, %977, !dbg !28
  %979 = insertelement <4 x float> %975, float %977, i64 1, !dbg !29
  %980 = extractelement <4 x float> %971, i64 2, !dbg !27
  %981 = call float @llvm.amdgcn.exp2.f32(float %980), !dbg !27
  %982 = fadd float %940, %981, !dbg !28
  %983 = insertelement <4 x float> %979, float %981, i64 2, !dbg !29
  %984 = extractelement <4 x float> %971, i64 3, !dbg !27
  %985 = call float @llvm.amdgcn.exp2.f32(float %984), !dbg !27
  %986 = fadd float %944, %985, !dbg !28
  %987 = insertelement <4 x float> %983, float %985, i64 3, !dbg !29
  %math.result307 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %987) #9, !dbg !29, !frisk.register_lifetimes !1
  %.cvtpack.f02026 = extractelement <4 x float> %math.result307, i32 0, !dbg !29
  %.cvtpack.f12027 = extractelement <4 x float> %math.result307, i32 1, !dbg !29
  %.cvtpack.f22028 = extractelement <4 x float> %math.result307, i32 2, !dbg !29
  %.cvtpack.f32029 = extractelement <4 x float> %math.result307, i32 3, !dbg !29
  %.cvtpack.p02030 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02026, float %.cvtpack.f12027), !dbg !29
  %.cvtpack.p12031 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22028, float %.cvtpack.f32029), !dbg !29
  %.cvtpack.v02032 = insertelement <2 x i32> poison, i32 %.cvtpack.p02030, i32 0, !dbg !29
  %.cvtpack.v12033 = insertelement <2 x i32> %.cvtpack.v02032, i32 %.cvtpack.p12031, i32 1, !dbg !29
  %988 = bitcast <2 x i32> %.cvtpack.v12033 to <4 x half>, !dbg !29
  %math.packed308 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %988) #9, !dbg !28, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %989 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed308, <4 x half> %970, <4 x float> %947) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %990 = shufflevector <8 x half> %804, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %991 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed308, <4 x half> %990, <4 x float> %949) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %992 = fsub <4 x float> %754, %738, !dbg !23
  %993 = extractelement <4 x float> %992, i64 0, !dbg !27
  %994 = call float @llvm.amdgcn.exp2.f32(float %993), !dbg !27
  %995 = fadd float %953, %994, !dbg !28
  %996 = insertelement <4 x float> poison, float %994, i64 0, !dbg !29
  %997 = extractelement <4 x float> %992, i64 1, !dbg !27
  %998 = call float @llvm.amdgcn.exp2.f32(float %997), !dbg !27
  %999 = fadd float %957, %998, !dbg !28
  %1000 = insertelement <4 x float> %996, float %998, i64 1, !dbg !29
  %1001 = extractelement <4 x float> %992, i64 2, !dbg !27
  %1002 = call float @llvm.amdgcn.exp2.f32(float %1001), !dbg !27
  %1003 = fadd float %961, %1002, !dbg !28
  %1004 = insertelement <4 x float> %1000, float %1002, i64 2, !dbg !29
  %1005 = extractelement <4 x float> %992, i64 3, !dbg !27
  %1006 = call float @llvm.amdgcn.exp2.f32(float %1005), !dbg !27
  %1007 = fadd float %965, %1006, !dbg !28
  %1008 = insertelement <4 x float> %1004, float %1006, i64 3, !dbg !29
  %math.result323 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1008) #9, !dbg !29, !frisk.register_lifetimes !1
  %.cvtpack.f02034 = extractelement <4 x float> %math.result323, i32 0, !dbg !29
  %.cvtpack.f12035 = extractelement <4 x float> %math.result323, i32 1, !dbg !29
  %.cvtpack.f22036 = extractelement <4 x float> %math.result323, i32 2, !dbg !29
  %.cvtpack.f32037 = extractelement <4 x float> %math.result323, i32 3, !dbg !29
  %.cvtpack.p02038 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02034, float %.cvtpack.f12035), !dbg !29
  %.cvtpack.p12039 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22036, float %.cvtpack.f32037), !dbg !29
  %.cvtpack.v02040 = insertelement <2 x i32> poison, i32 %.cvtpack.p02038, i32 0, !dbg !29
  %.cvtpack.v12041 = insertelement <2 x i32> %.cvtpack.v02040, i32 %.cvtpack.p12039, i32 1, !dbg !29
  %1009 = bitcast <2 x i32> %.cvtpack.v12041 to <4 x half>, !dbg !29
  %math.packed324 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1009) #9, !dbg !28, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %1010 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %970, <4 x float> %968) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1011 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %990, <4 x float> %969) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1012 = shufflevector <8 x half> %806, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1013 = fsub <4 x float> %725, %705, !dbg !23
  %1014 = extractelement <4 x float> %1013, i64 0, !dbg !27
  %1015 = call float @llvm.amdgcn.exp2.f32(float %1014), !dbg !27
  %1016 = fadd float %974, %1015, !dbg !28
  %1017 = insertelement <4 x float> poison, float %1015, i64 0, !dbg !29
  %1018 = extractelement <4 x float> %1013, i64 1, !dbg !27
  %1019 = call float @llvm.amdgcn.exp2.f32(float %1018), !dbg !27
  %1020 = fadd float %978, %1019, !dbg !28
  %1021 = insertelement <4 x float> %1017, float %1019, i64 1, !dbg !29
  %1022 = extractelement <4 x float> %1013, i64 2, !dbg !27
  %1023 = call float @llvm.amdgcn.exp2.f32(float %1022), !dbg !27
  %1024 = fadd float %982, %1023, !dbg !28
  %1025 = insertelement <4 x float> %1021, float %1023, i64 2, !dbg !29
  %1026 = extractelement <4 x float> %1013, i64 3, !dbg !27
  %1027 = call float @llvm.amdgcn.exp2.f32(float %1026), !dbg !27
  %1028 = fadd float %986, %1027, !dbg !28
  %1029 = insertelement <4 x float> %1025, float %1027, i64 3, !dbg !29
  %math.result309 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1029) #9, !dbg !29, !frisk.register_lifetimes !1
  %.cvtpack.f02042 = extractelement <4 x float> %math.result309, i32 0, !dbg !29
  %.cvtpack.f12043 = extractelement <4 x float> %math.result309, i32 1, !dbg !29
  %.cvtpack.f22044 = extractelement <4 x float> %math.result309, i32 2, !dbg !29
  %.cvtpack.f32045 = extractelement <4 x float> %math.result309, i32 3, !dbg !29
  %.cvtpack.p02046 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02042, float %.cvtpack.f12043), !dbg !29
  %.cvtpack.p12047 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22044, float %.cvtpack.f32045), !dbg !29
  %.cvtpack.v02048 = insertelement <2 x i32> poison, i32 %.cvtpack.p02046, i32 0, !dbg !29
  %.cvtpack.v12049 = insertelement <2 x i32> %.cvtpack.v02048, i32 %.cvtpack.p12047, i32 1, !dbg !29
  %1030 = bitcast <2 x i32> %.cvtpack.v12049 to <4 x half>, !dbg !29
  %math.packed310 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1030) #9, !dbg !28, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %1031 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %1012, <4 x float> %989) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1032 = shufflevector <8 x half> %806, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1033 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %1032, <4 x float> %991) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1034 = fsub <4 x float> %758, %738, !dbg !23
  %1035 = extractelement <4 x float> %1034, i64 0, !dbg !27
  %1036 = call float @llvm.amdgcn.exp2.f32(float %1035), !dbg !27
  %1037 = fadd float %995, %1036, !dbg !28
  %1038 = insertelement <4 x float> poison, float %1036, i64 0, !dbg !29
  %1039 = extractelement <4 x float> %1034, i64 1, !dbg !27
  %1040 = call float @llvm.amdgcn.exp2.f32(float %1039), !dbg !27
  %1041 = fadd float %999, %1040, !dbg !28
  %1042 = insertelement <4 x float> %1038, float %1040, i64 1, !dbg !29
  %1043 = extractelement <4 x float> %1034, i64 2, !dbg !27
  %1044 = call float @llvm.amdgcn.exp2.f32(float %1043), !dbg !27
  %1045 = fadd float %1003, %1044, !dbg !28
  %1046 = insertelement <4 x float> %1042, float %1044, i64 2, !dbg !29
  %1047 = extractelement <4 x float> %1034, i64 3, !dbg !27
  %1048 = call float @llvm.amdgcn.exp2.f32(float %1047), !dbg !27
  %1049 = fadd float %1007, %1048, !dbg !28
  %1050 = insertelement <4 x float> %1046, float %1048, i64 3, !dbg !29
  %math.result325 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1050) #9, !dbg !29, !frisk.register_lifetimes !1
  %.cvtpack.f02050 = extractelement <4 x float> %math.result325, i32 0, !dbg !29
  %.cvtpack.f12051 = extractelement <4 x float> %math.result325, i32 1, !dbg !29
  %.cvtpack.f22052 = extractelement <4 x float> %math.result325, i32 2, !dbg !29
  %.cvtpack.f32053 = extractelement <4 x float> %math.result325, i32 3, !dbg !29
  %.cvtpack.p02054 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02050, float %.cvtpack.f12051), !dbg !29
  %.cvtpack.p12055 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22052, float %.cvtpack.f32053), !dbg !29
  %.cvtpack.v02056 = insertelement <2 x i32> poison, i32 %.cvtpack.p02054, i32 0, !dbg !29
  %.cvtpack.v12057 = insertelement <2 x i32> %.cvtpack.v02056, i32 %.cvtpack.p12055, i32 1, !dbg !29
  %1051 = bitcast <2 x i32> %.cvtpack.v12057 to <4 x half>, !dbg !29
  %math.packed326 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1051) #9, !dbg !28, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %1052 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1012, <4 x float> %1010) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1053 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1032, <4 x float> %1011) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1054 = shufflevector <8 x half> %808, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1055 = fsub <4 x float> %729, %705, !dbg !23
  %1056 = extractelement <4 x float> %1055, i64 0, !dbg !27
  %1057 = call float @llvm.amdgcn.exp2.f32(float %1056), !dbg !27
  %1058 = fadd float %1016, %1057, !dbg !28
  %1059 = insertelement <4 x float> poison, float %1057, i64 0, !dbg !29
  %1060 = extractelement <4 x float> %1055, i64 1, !dbg !27
  %1061 = call float @llvm.amdgcn.exp2.f32(float %1060), !dbg !27
  %1062 = fadd float %1020, %1061, !dbg !28
  %1063 = insertelement <4 x float> %1059, float %1061, i64 1, !dbg !29
  %1064 = extractelement <4 x float> %1055, i64 2, !dbg !27
  %1065 = call float @llvm.amdgcn.exp2.f32(float %1064), !dbg !27
  %1066 = fadd float %1024, %1065, !dbg !28
  %1067 = insertelement <4 x float> %1063, float %1065, i64 2, !dbg !29
  %1068 = extractelement <4 x float> %1055, i64 3, !dbg !27
  %1069 = call float @llvm.amdgcn.exp2.f32(float %1068), !dbg !27
  %1070 = fadd float %1028, %1069, !dbg !28
  %1071 = insertelement <4 x float> %1067, float %1069, i64 3, !dbg !29
  %math.result311 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1071) #9, !dbg !29, !frisk.register_lifetimes !1
  %.cvtpack.f02058 = extractelement <4 x float> %math.result311, i32 0, !dbg !29
  %.cvtpack.f12059 = extractelement <4 x float> %math.result311, i32 1, !dbg !29
  %.cvtpack.f22060 = extractelement <4 x float> %math.result311, i32 2, !dbg !29
  %.cvtpack.f32061 = extractelement <4 x float> %math.result311, i32 3, !dbg !29
  %.cvtpack.p02062 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02058, float %.cvtpack.f12059), !dbg !29
  %.cvtpack.p12063 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22060, float %.cvtpack.f32061), !dbg !29
  %.cvtpack.v02064 = insertelement <2 x i32> poison, i32 %.cvtpack.p02062, i32 0, !dbg !29
  %.cvtpack.v12065 = insertelement <2 x i32> %.cvtpack.v02064, i32 %.cvtpack.p12063, i32 1, !dbg !29
  %1072 = bitcast <2 x i32> %.cvtpack.v12065 to <4 x half>, !dbg !29
  %math.packed312 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1072) #9, !dbg !28, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %1073 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %1054, <4 x float> %1031) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1074 = shufflevector <8 x half> %808, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1075 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %1074, <4 x float> %1033) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1076 = fsub <4 x float> %762, %738, !dbg !23
  %1077 = extractelement <4 x float> %1076, i64 0, !dbg !27
  %1078 = call float @llvm.amdgcn.exp2.f32(float %1077), !dbg !27
  %1079 = fadd float %1037, %1078, !dbg !28
  %1080 = insertelement <4 x float> poison, float %1078, i64 0, !dbg !29
  %1081 = extractelement <4 x float> %1076, i64 1, !dbg !27
  %1082 = call float @llvm.amdgcn.exp2.f32(float %1081), !dbg !27
  %1083 = fadd float %1041, %1082, !dbg !28
  %1084 = insertelement <4 x float> %1080, float %1082, i64 1, !dbg !29
  %1085 = extractelement <4 x float> %1076, i64 2, !dbg !27
  %1086 = call float @llvm.amdgcn.exp2.f32(float %1085), !dbg !27
  %1087 = fadd float %1045, %1086, !dbg !28
  %1088 = insertelement <4 x float> %1084, float %1086, i64 2, !dbg !29
  %1089 = extractelement <4 x float> %1076, i64 3, !dbg !27
  %1090 = call float @llvm.amdgcn.exp2.f32(float %1089), !dbg !27
  %1091 = fadd float %1049, %1090, !dbg !28
  %1092 = insertelement <4 x float> %1088, float %1090, i64 3, !dbg !29
  %math.result327 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1092) #9, !dbg !29, !frisk.register_lifetimes !1
  %.cvtpack.f02066 = extractelement <4 x float> %math.result327, i32 0, !dbg !29
  %.cvtpack.f12067 = extractelement <4 x float> %math.result327, i32 1, !dbg !29
  %.cvtpack.f22068 = extractelement <4 x float> %math.result327, i32 2, !dbg !29
  %.cvtpack.f32069 = extractelement <4 x float> %math.result327, i32 3, !dbg !29
  %.cvtpack.p02070 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02066, float %.cvtpack.f12067), !dbg !29
  %.cvtpack.p12071 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22068, float %.cvtpack.f32069), !dbg !29
  %.cvtpack.v02072 = insertelement <2 x i32> poison, i32 %.cvtpack.p02070, i32 0, !dbg !29
  %.cvtpack.v12073 = insertelement <2 x i32> %.cvtpack.v02072, i32 %.cvtpack.p12071, i32 1, !dbg !29
  %1093 = bitcast <2 x i32> %.cvtpack.v12073 to <4 x half>, !dbg !29
  %math.packed328 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1093) #9, !dbg !28, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %1094 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1054, <4 x float> %1052) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1095 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1074, <4 x float> %1053) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1096 = bitcast <4 x i32> %809 to <8 x half>, !dbg !14
  %1097 = shufflevector <8 x half> %1096, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1098 = fsub <4 x float> %733, %705, !dbg !23
  %1099 = extractelement <4 x float> %1098, i64 0, !dbg !27
  %1100 = call float @llvm.amdgcn.exp2.f32(float %1099), !dbg !27
  %1101 = fadd float %1058, %1100, !dbg !28
  %1102 = insertelement <4 x float> poison, float %1100, i64 0, !dbg !29
  %1103 = extractelement <4 x float> %1098, i64 1, !dbg !27
  %1104 = call float @llvm.amdgcn.exp2.f32(float %1103), !dbg !27
  %1105 = fadd float %1062, %1104, !dbg !28
  %1106 = fadd float %1101, %1105, !dbg !28
  %1107 = insertelement <4 x float> %1102, float %1104, i64 1, !dbg !29
  %1108 = extractelement <4 x float> %1098, i64 2, !dbg !27
  %1109 = call float @llvm.amdgcn.exp2.f32(float %1108), !dbg !27
  %1110 = fadd float %1066, %1109, !dbg !28
  %1111 = insertelement <4 x float> %1107, float %1109, i64 2, !dbg !29
  %1112 = extractelement <4 x float> %1098, i64 3, !dbg !27
  %1113 = call float @llvm.amdgcn.exp2.f32(float %1112), !dbg !27
  %1114 = fadd float %1070, %1113, !dbg !28
  %1115 = fadd float %1110, %1114, !dbg !28
  %1116 = fadd float %1106, %1115, !dbg !28
  %1117 = fadd float %1116, %774, !dbg !28
  %1118 = insertelement <1 x float> poison, float %1117, i64 0, !dbg !6
  %1119 = insertvalue [2 x <1 x float>] poison, <1 x float> %1118, 0, !dbg !6
  %1120 = insertelement <4 x float> %1111, float %1113, i64 3, !dbg !29
  %math.result313 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1120) #9, !dbg !29, !frisk.register_lifetimes !1
  %.cvtpack.f02074 = extractelement <4 x float> %math.result313, i32 0, !dbg !29
  %.cvtpack.f12075 = extractelement <4 x float> %math.result313, i32 1, !dbg !29
  %.cvtpack.f22076 = extractelement <4 x float> %math.result313, i32 2, !dbg !29
  %.cvtpack.f32077 = extractelement <4 x float> %math.result313, i32 3, !dbg !29
  %.cvtpack.p02078 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02074, float %.cvtpack.f12075), !dbg !29
  %.cvtpack.p12079 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22076, float %.cvtpack.f32077), !dbg !29
  %.cvtpack.v02080 = insertelement <2 x i32> poison, i32 %.cvtpack.p02078, i32 0, !dbg !29
  %.cvtpack.v12081 = insertelement <2 x i32> %.cvtpack.v02080, i32 %.cvtpack.p12079, i32 1, !dbg !29
  %1121 = bitcast <2 x i32> %.cvtpack.v12081 to <4 x half>, !dbg !29
  %math.packed314 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1121) #9, !dbg !28, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %1122 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %1097, <4 x float> %1073) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1123 = shufflevector <8 x half> %1096, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1124 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %1123, <4 x float> %1075) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1125 = fsub <4 x float> %766, %738, !dbg !23
  %1126 = extractelement <4 x float> %1125, i64 0, !dbg !27
  %1127 = call float @llvm.amdgcn.exp2.f32(float %1126), !dbg !27
  %1128 = fadd float %1079, %1127, !dbg !28
  %1129 = insertelement <4 x float> poison, float %1127, i64 0, !dbg !29
  %1130 = extractelement <4 x float> %1125, i64 1, !dbg !27
  %1131 = call float @llvm.amdgcn.exp2.f32(float %1130), !dbg !27
  %1132 = fadd float %1083, %1131, !dbg !28
  %1133 = fadd float %1128, %1132, !dbg !28
  %1134 = insertelement <4 x float> %1129, float %1131, i64 1, !dbg !29
  %1135 = extractelement <4 x float> %1125, i64 2, !dbg !27
  %1136 = call float @llvm.amdgcn.exp2.f32(float %1135), !dbg !27
  %1137 = fadd float %1087, %1136, !dbg !28
  %1138 = insertelement <4 x float> %1134, float %1136, i64 2, !dbg !29
  %1139 = extractelement <4 x float> %1125, i64 3, !dbg !27
  %1140 = call float @llvm.amdgcn.exp2.f32(float %1139), !dbg !27
  %1141 = fadd float %1091, %1140, !dbg !28
  %1142 = fadd float %1137, %1141, !dbg !28
  %1143 = fadd float %1133, %1142, !dbg !28
  %1144 = fadd float %1143, %782, !dbg !28
  %1145 = insertelement <1 x float> poison, float %1144, i64 0, !dbg !6
  %1146 = insertvalue [2 x <1 x float>] %1119, <1 x float> %1145, 1, !dbg !6
  %1147 = insertelement <4 x float> %1138, float %1140, i64 3, !dbg !29
  %math.result329 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1147) #9, !dbg !29, !frisk.register_lifetimes !1
  %.cvtpack.f02082 = extractelement <4 x float> %math.result329, i32 0, !dbg !29
  %.cvtpack.f12083 = extractelement <4 x float> %math.result329, i32 1, !dbg !29
  %.cvtpack.f22084 = extractelement <4 x float> %math.result329, i32 2, !dbg !29
  %.cvtpack.f32085 = extractelement <4 x float> %math.result329, i32 3, !dbg !29
  %.cvtpack.p02086 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02082, float %.cvtpack.f12083), !dbg !29
  %.cvtpack.p12087 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22084, float %.cvtpack.f32085), !dbg !29
  %.cvtpack.v02088 = insertelement <2 x i32> poison, i32 %.cvtpack.p02086, i32 0, !dbg !29
  %.cvtpack.v12089 = insertelement <2 x i32> %.cvtpack.v02088, i32 %.cvtpack.p12087, i32 1, !dbg !29
  %1148 = bitcast <2 x i32> %.cvtpack.v12089 to <4 x half>, !dbg !29
  %math.packed330 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1148) #9, !dbg !28, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %1149 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1097, <4 x float> %1094) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1150 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1123, <4 x float> %1095) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !30
  %1151 = extractvalue [2 x <8 x float>] %132, 0, !dbg !30
  %1152 = shufflevector <8 x float> %1151, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %1153 = fmul <4 x float> %1152, %771, !dbg !30
  %1154 = shufflevector <8 x float> %1151, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %1155 = fmul <4 x float> %1154, %771, !dbg !30
  %1156 = extractvalue [2 x <8 x float>] %132, 1, !dbg !30
  %1157 = shufflevector <8 x float> %1156, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %1158 = fmul <4 x float> %1157, %779, !dbg !30
  %1159 = shufflevector <8 x float> %1156, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %1160 = fmul <4 x float> %1159, %779, !dbg !30
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %vdb.address1418 = or disjoint i64 %.idx13711373, 64, !dbg !18
  %vdb.address1419 = add i64 %address.local849, %vdb.address1418, !dbg !18
  %vdb.address1420 = trunc i64 %vdb.address1419 to i32, !dbg !18
  %vdb.address1421 = shl i32 %vdb.address1420, 1, !dbg !18
  %vdb.address1429 = or i32 %.tr1378, 16512, !dbg !18
  %1161 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %49, i32 %vdb.address1421, <4 x i32> %dma.buffer.rsrc1277) #10, !dbg !18
  %1162 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %57, i32 %vdb.address1429, <4 x i32> %dma.buffer.rsrc1277) #10, !dbg !18
  %vdb.read = getelementptr i8, ptr addrspace(3) %address.local865, i64 8192, !dbg !14
  %1163 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #0, !dbg !14, !frisk.v_double_buffer !35
  %1164 = bitcast <4 x i32> %1163 to <8 x half>, !dbg !14
  %vdb.read1411 = getelementptr i8, ptr addrspace(3) %address.local874, i64 8448, !dbg !14
  %1165 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1411, i32 0) #0, !dbg !14, !frisk.v_double_buffer !35
  %1166 = bitcast <4 x i32> %1165 to <8 x half>, !dbg !14
  %vdb.read1412 = getelementptr i8, ptr addrspace(3) %address.local874, i64 8704, !dbg !14
  %1167 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1412, i32 0) #0, !dbg !14, !frisk.v_double_buffer !35
  %1168 = bitcast <4 x i32> %1167 to <8 x half>, !dbg !14
  %vdb.read1413 = getelementptr i8, ptr addrspace(3) %address.local874, i64 8960, !dbg !14
  %1169 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1413, i32 0) #0, !dbg !14, !frisk.v_double_buffer !35
  %1170 = bitcast <4 x i32> %1169 to <8 x half>, !dbg !14
  %vdb.read1414 = getelementptr i8, ptr addrspace(3) %address.local865, i64 12288, !dbg !14
  %1171 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1414, i32 0) #0, !dbg !14, !frisk.v_double_buffer !35
  %1172 = bitcast <4 x i32> %1171 to <8 x half>, !dbg !14
  %vdb.read1415 = getelementptr i8, ptr addrspace(3) %address.local874, i64 12544, !dbg !14
  %1173 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1415, i32 0) #0, !dbg !14, !frisk.v_double_buffer !35
  %1174 = bitcast <4 x i32> %1173 to <8 x half>, !dbg !14
  %vdb.read1416 = getelementptr i8, ptr addrspace(3) %address.local874, i64 12800, !dbg !14
  %1175 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1416, i32 0) #0, !dbg !14, !frisk.v_double_buffer !35
  %1176 = bitcast <4 x i32> %1175 to <8 x half>, !dbg !14
  %vdb.read1417 = getelementptr i8, ptr addrspace(3) %address.local874, i64 13056, !dbg !14
  %1177 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1417, i32 0) #0, !dbg !14, !frisk.v_double_buffer !35
  %1178 = shufflevector <8 x half> %1164, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1179 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1178, <4 x float> %1153) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1180 = shufflevector <8 x half> %1164, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1181 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1180, <4 x float> %1155) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1182 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %1178, <4 x float> %1158) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1183 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %1180, <4 x float> %1160) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1184 = shufflevector <8 x half> %1166, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1185 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed302, <4 x half> %1184, <4 x float> %1179) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1186 = shufflevector <8 x half> %1166, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1187 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed302, <4 x half> %1186, <4 x float> %1181) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1188 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %1184, <4 x float> %1182) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1189 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %1186, <4 x float> %1183) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1190 = shufflevector <8 x half> %1168, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1191 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed304, <4 x half> %1190, <4 x float> %1185) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1192 = shufflevector <8 x half> %1168, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1193 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed304, <4 x half> %1192, <4 x float> %1187) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1194 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %1190, <4 x float> %1188) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1195 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %1192, <4 x float> %1189) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1196 = shufflevector <8 x half> %1170, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1197 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed306, <4 x half> %1196, <4 x float> %1191) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1198 = shufflevector <8 x half> %1170, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1199 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed306, <4 x half> %1198, <4 x float> %1193) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1200 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %1196, <4 x float> %1194) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1201 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %1198, <4 x float> %1195) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1202 = shufflevector <8 x half> %1172, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1203 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed308, <4 x half> %1202, <4 x float> %1197) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1204 = shufflevector <8 x half> %1172, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1205 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed308, <4 x half> %1204, <4 x float> %1199) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1206 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1202, <4 x float> %1200) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1207 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1204, <4 x float> %1201) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1208 = shufflevector <8 x half> %1174, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1209 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %1208, <4 x float> %1203) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1210 = shufflevector <8 x half> %1174, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1211 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %1210, <4 x float> %1205) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1212 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1208, <4 x float> %1206) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1213 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1210, <4 x float> %1207) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1214 = shufflevector <8 x half> %1176, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1215 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %1214, <4 x float> %1209) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1216 = shufflevector <8 x half> %1176, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1217 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %1216, <4 x float> %1211) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1218 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1214, <4 x float> %1212) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1219 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1216, <4 x float> %1213) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1220 = bitcast <4 x i32> %1177 to <8 x half>, !dbg !14
  %1221 = shufflevector <8 x half> %1220, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1222 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %1221, <4 x float> %1215) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1223 = shufflevector <8 x half> %1220, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1224 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %1223, <4 x float> %1217) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1225 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1221, <4 x float> %1218) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1226 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1223, <4 x float> %1219) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !30
  %1227 = extractvalue [2 x <8 x float>] %133, 0, !dbg !30
  %1228 = shufflevector <8 x float> %1227, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %1229 = fmul <4 x float> %1228, %771, !dbg !30
  %1230 = shufflevector <8 x float> %1227, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %1231 = fmul <4 x float> %1230, %771, !dbg !30
  %1232 = extractvalue [2 x <8 x float>] %133, 1, !dbg !30
  %1233 = shufflevector <8 x float> %1232, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %1234 = fmul <4 x float> %1233, %779, !dbg !30
  %1235 = shufflevector <8 x float> %1232, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %1236 = fmul <4 x float> %1235, %779, !dbg !30
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %vdb.address1434 = or disjoint i64 %.idx13711373, 96, !dbg !18
  %vdb.address1435 = add i64 %address.local849, %vdb.address1434, !dbg !18
  %vdb.address1436 = trunc i64 %vdb.address1435 to i32, !dbg !18
  %vdb.address1437 = shl i32 %vdb.address1436, 1, !dbg !18
  %vdb.address1445 = or i32 %.tr1378, 16576, !dbg !18
  %1237 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1437, <4 x i32> %dma.buffer.rsrc1277) #10, !dbg !18
  %1238 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1410, i32 %vdb.address1445, <4 x i32> %dma.buffer.rsrc1277) #10, !dbg !18
  %1239 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local865, i32 0) #0, !dbg !14, !frisk.v_double_buffer !35
  %1240 = bitcast <4 x i32> %1239 to <8 x half>, !dbg !14
  %1241 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local875, i32 0) #0, !dbg !14, !frisk.v_double_buffer !35
  %1242 = bitcast <4 x i32> %1241 to <8 x half>, !dbg !14
  %1243 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local879, i32 0) #0, !dbg !14, !frisk.v_double_buffer !35
  %1244 = bitcast <4 x i32> %1243 to <8 x half>, !dbg !14
  %1245 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local883, i32 0) #0, !dbg !14, !frisk.v_double_buffer !35
  %1246 = bitcast <4 x i32> %1245 to <8 x half>, !dbg !14
  %1247 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local885, i32 0) #0, !dbg !14, !frisk.v_double_buffer !35
  %1248 = bitcast <4 x i32> %1247 to <8 x half>, !dbg !14
  %1249 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local889, i32 0) #0, !dbg !14, !frisk.v_double_buffer !35
  %1250 = bitcast <4 x i32> %1249 to <8 x half>, !dbg !14
  %1251 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local893, i32 0) #0, !dbg !14, !frisk.v_double_buffer !35
  %1252 = bitcast <4 x i32> %1251 to <8 x half>, !dbg !14
  %1253 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local897, i32 0) #0, !dbg !14, !frisk.v_double_buffer !35
  %1254 = shufflevector <8 x half> %1240, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1255 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1254, <4 x float> %1229) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1256 = shufflevector <8 x half> %1240, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1257 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1256, <4 x float> %1231) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1258 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %1254, <4 x float> %1234) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1259 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %1256, <4 x float> %1236) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1260 = shufflevector <8 x half> %1242, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1261 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed302, <4 x half> %1260, <4 x float> %1255) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1262 = shufflevector <8 x half> %1242, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1263 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed302, <4 x half> %1262, <4 x float> %1257) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1264 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %1260, <4 x float> %1258) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1265 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %1262, <4 x float> %1259) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1266 = shufflevector <8 x half> %1244, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1267 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed304, <4 x half> %1266, <4 x float> %1261) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1268 = shufflevector <8 x half> %1244, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1269 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed304, <4 x half> %1268, <4 x float> %1263) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1270 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %1266, <4 x float> %1264) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1271 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %1268, <4 x float> %1265) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1272 = shufflevector <8 x half> %1246, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1273 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed306, <4 x half> %1272, <4 x float> %1267) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1274 = shufflevector <8 x half> %1246, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1275 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed306, <4 x half> %1274, <4 x float> %1269) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1276 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %1272, <4 x float> %1270) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1277 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %1274, <4 x float> %1271) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1278 = shufflevector <8 x half> %1248, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1279 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed308, <4 x half> %1278, <4 x float> %1273) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1280 = shufflevector <8 x half> %1248, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1281 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed308, <4 x half> %1280, <4 x float> %1275) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1282 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1278, <4 x float> %1276) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1283 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1280, <4 x float> %1277) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1284 = shufflevector <8 x half> %1250, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1285 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %1284, <4 x float> %1279) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1286 = shufflevector <8 x half> %1250, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1287 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %1286, <4 x float> %1281) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1288 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1284, <4 x float> %1282) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1289 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1286, <4 x float> %1283) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1290 = shufflevector <8 x half> %1252, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1291 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %1290, <4 x float> %1285) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1292 = shufflevector <8 x half> %1252, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1293 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %1292, <4 x float> %1287) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1294 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1290, <4 x float> %1288) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1295 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1292, <4 x float> %1289) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1296 = bitcast <4 x i32> %1253 to <8 x half>, !dbg !14
  %1297 = shufflevector <8 x half> %1296, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1298 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %1297, <4 x float> %1291) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1299 = shufflevector <8 x half> %1296, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1300 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %1299, <4 x float> %1293) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1301 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1297, <4 x float> %1294) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1302 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1299, <4 x float> %1295) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !30
  %1303 = extractvalue [2 x <8 x float>] %134, 0, !dbg !30
  %1304 = shufflevector <8 x float> %1303, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %1305 = fmul <4 x float> %1304, %771, !dbg !30
  %1306 = shufflevector <8 x float> %1303, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %1307 = fmul <4 x float> %1306, %771, !dbg !30
  %1308 = extractvalue [2 x <8 x float>] %134, 1, !dbg !30
  %1309 = shufflevector <8 x float> %1308, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %1310 = fmul <4 x float> %1309, %779, !dbg !30
  %1311 = shufflevector <8 x float> %1308, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %1312 = fmul <4 x float> %1311, %779, !dbg !30
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1313 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #0, !dbg !14, !frisk.v_double_buffer !35
  %1314 = bitcast <4 x i32> %1313 to <8 x half>, !dbg !14
  %1315 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1411, i32 0) #0, !dbg !14, !frisk.v_double_buffer !35
  %1316 = bitcast <4 x i32> %1315 to <8 x half>, !dbg !14
  %1317 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1412, i32 0) #0, !dbg !14, !frisk.v_double_buffer !35
  %1318 = bitcast <4 x i32> %1317 to <8 x half>, !dbg !14
  %1319 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1413, i32 0) #0, !dbg !14, !frisk.v_double_buffer !35
  %1320 = bitcast <4 x i32> %1319 to <8 x half>, !dbg !14
  %1321 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1414, i32 0) #0, !dbg !14, !frisk.v_double_buffer !35
  %1322 = bitcast <4 x i32> %1321 to <8 x half>, !dbg !14
  %1323 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1415, i32 0) #0, !dbg !14, !frisk.v_double_buffer !35
  %1324 = bitcast <4 x i32> %1323 to <8 x half>, !dbg !14
  %1325 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1416, i32 0) #0, !dbg !14, !frisk.v_double_buffer !35
  %1326 = bitcast <4 x i32> %1325 to <8 x half>, !dbg !14
  %1327 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1417, i32 0) #0, !dbg !14, !frisk.v_double_buffer !35
  %1328 = shufflevector <8 x half> %1314, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1329 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1328, <4 x float> %1305) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1330 = shufflevector <8 x half> %1314, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1331 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1330, <4 x float> %1307) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1332 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %1328, <4 x float> %1310) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1333 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed316, <4 x half> %1330, <4 x float> %1312) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1334 = shufflevector <8 x half> %1316, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1335 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed302, <4 x half> %1334, <4 x float> %1329) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1336 = shufflevector <8 x half> %1316, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1337 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed302, <4 x half> %1336, <4 x float> %1331) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1338 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %1334, <4 x float> %1332) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1339 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed318, <4 x half> %1336, <4 x float> %1333) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1340 = shufflevector <8 x half> %1318, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1341 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed304, <4 x half> %1340, <4 x float> %1335) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1342 = shufflevector <8 x half> %1318, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1343 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed304, <4 x half> %1342, <4 x float> %1337) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1344 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %1340, <4 x float> %1338) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1345 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed320, <4 x half> %1342, <4 x float> %1339) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1346 = shufflevector <8 x half> %1320, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1347 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed306, <4 x half> %1346, <4 x float> %1341) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1348 = shufflevector <8 x half> %1320, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1349 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed306, <4 x half> %1348, <4 x float> %1343) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1350 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %1346, <4 x float> %1344) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1351 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %1348, <4 x float> %1345) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1352 = shufflevector <8 x half> %1322, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1353 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed308, <4 x half> %1352, <4 x float> %1347) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1354 = shufflevector <8 x half> %1322, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1355 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed308, <4 x half> %1354, <4 x float> %1349) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1356 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1352, <4 x float> %1350) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1357 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %1354, <4 x float> %1351) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1358 = shufflevector <8 x half> %1324, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1359 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %1358, <4 x float> %1353) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1360 = shufflevector <8 x half> %1324, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1361 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed310, <4 x half> %1360, <4 x float> %1355) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1362 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1358, <4 x float> %1356) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1363 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %1360, <4 x float> %1357) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1364 = shufflevector <8 x half> %1326, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1365 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %1364, <4 x float> %1359) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1366 = shufflevector <8 x half> %1326, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1367 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed312, <4 x half> %1366, <4 x float> %1361) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1368 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1364, <4 x float> %1362) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1369 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %1366, <4 x float> %1363) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1370 = bitcast <4 x i32> %1327 to <8 x half>, !dbg !14
  %1371 = shufflevector <8 x half> %1370, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1372 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %1371, <4 x float> %1365) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1373 = shufflevector <8 x half> %1370, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1374 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed314, <4 x half> %1373, <4 x float> %1367) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1375 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1371, <4 x float> %1368) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1376 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %1373, <4 x float> %1369) #9, !dbg !14, !frisk.pv_interleave !34
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !6
  %1377 = shufflevector <4 x float> %1122, <4 x float> %1124, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !6
  %1378 = insertvalue [2 x <8 x float>] poison, <8 x float> %1377, 0, !dbg !6
  %1379 = shufflevector <4 x float> %1149, <4 x float> %1150, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !6
  %1380 = insertvalue [2 x <8 x float>] %1378, <8 x float> %1379, 1, !dbg !6
  %1381 = insertvalue [2 x <1 x float>] poison, <1 x float> %697, 0, !dbg !6
  %1382 = insertvalue [2 x <1 x float>] %1381, <1 x float> %700, 1, !dbg !6
  %1383 = shufflevector <4 x float> %1222, <4 x float> %1224, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !6
  %1384 = insertvalue [2 x <8 x float>] poison, <8 x float> %1383, 0, !dbg !6
  %1385 = shufflevector <4 x float> %1225, <4 x float> %1226, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !6
  %1386 = insertvalue [2 x <8 x float>] %1384, <8 x float> %1385, 1, !dbg !6
  %1387 = shufflevector <4 x float> %1298, <4 x float> %1300, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !6
  %1388 = insertvalue [2 x <8 x float>] poison, <8 x float> %1387, 0, !dbg !6
  %1389 = shufflevector <4 x float> %1301, <4 x float> %1302, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !6
  %1390 = insertvalue [2 x <8 x float>] %1388, <8 x float> %1389, 1, !dbg !6
  %1391 = shufflevector <4 x float> %1372, <4 x float> %1374, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !6
  %1392 = insertvalue [2 x <8 x float>] poison, <8 x float> %1391, 0, !dbg !6
  %1393 = shufflevector <4 x float> %1375, <4 x float> %1376, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !6
  %1394 = insertvalue [2 x <8 x float>] %1392, <8 x float> %1393, 1, !dbg !6
  fence syncscope("workgroup") release, !dbg !6
  call void @llvm.amdgcn.s.barrier(), !dbg !6
  fence syncscope("workgroup") acquire, !dbg !6
  %1395 = add nuw nsw i64 %128, 1, !dbg !6
  br label %127, !dbg !6

1396:                                             ; preds = %127
  %1397 = extractvalue [2 x <1 x float>] %130, 1, !dbg !6
  %1398 = extractelement <1 x float> %1397, i64 0, !dbg !6
  %1399 = bitcast float %1398 to i32, !dbg !6
  %1400 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1399), !dbg !6
  %1401 = bitcast i32 %1400 to float, !dbg !6
  %1402 = fadd float %1398, %1401, !dbg !6
  %1403 = bitcast float %1402 to i32, !dbg !6
  %1404 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1403), !dbg !6
  %1405 = bitcast i32 %1404 to float, !dbg !6
  %1406 = fadd float %1402, %1405, !dbg !6
  %1407 = extractvalue [2 x <1 x float>] %130, 0, !dbg !6
  %1408 = extractelement <1 x float> %1407, i64 0, !dbg !6
  %1409 = bitcast float %1408 to i32, !dbg !6
  %1410 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1409), !dbg !6
  %1411 = bitcast i32 %1410 to float, !dbg !6
  %1412 = fadd float %1408, %1411, !dbg !6
  %1413 = bitcast float %1412 to i32, !dbg !6
  %1414 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1413), !dbg !6
  %1415 = bitcast i32 %1414 to float, !dbg !6
  %1416 = fadd float %1412, %1415, !dbg !6
  %1417 = and i32 %kdma.tid, 16, !dbg !6
  %.not = icmp eq i32 %1417, 0, !dbg !6
  %1418 = and i32 %kdma.tid, 32, !dbg !6
  %.not40 = icmp eq i32 %1418, 0, !dbg !6
  %1419 = extractvalue [2 x <8 x float>] %134, 0, !dbg !6
  %1420 = extractelement <8 x float> %1419, i64 0, !dbg !6
  %1421 = extractelement <8 x float> %1419, i64 1, !dbg !6
  %1422 = extractelement <8 x float> %1419, i64 2, !dbg !6
  %1423 = extractelement <8 x float> %1419, i64 3, !dbg !6
  %1424 = select i1 %.not, float %1421, float %1420, !dbg !6
  %1425 = bitcast float %1424 to i32, !dbg !6
  %1426 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1425), !dbg !6
  %1427 = bitcast i32 %1426 to float, !dbg !6
  %1428 = select i1 %.not, float %1420, float %1427, !dbg !6
  %1429 = select i1 %.not, float %1427, float %1421, !dbg !6
  %1430 = select i1 %.not, float %1423, float %1422, !dbg !6
  %1431 = bitcast float %1430 to i32, !dbg !6
  %1432 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1431), !dbg !6
  %1433 = bitcast i32 %1432 to float, !dbg !6
  %1434 = select i1 %.not, float %1422, float %1433, !dbg !6
  %1435 = select i1 %.not, float %1433, float %1423, !dbg !6
  %1436 = select i1 %.not40, float %1434, float %1428, !dbg !6
  %1437 = bitcast float %1436 to i32, !dbg !6
  %1438 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1437), !dbg !6
  %1439 = bitcast i32 %1438 to float, !dbg !6
  %1440 = select i1 %.not40, float %1428, float %1439, !dbg !6
  %1441 = select i1 %.not40, float %1439, float %1434, !dbg !6
  %1442 = select i1 %.not40, float %1435, float %1429, !dbg !6
  %1443 = bitcast float %1442 to i32, !dbg !6
  %1444 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1443), !dbg !6
  %1445 = bitcast i32 %1444 to float, !dbg !6
  %1446 = select i1 %.not40, float %1429, float %1445, !dbg !6
  %1447 = select i1 %.not40, float %1445, float %1435, !dbg !6
  %1448 = extractelement <8 x float> %1419, i64 4, !dbg !6
  %1449 = extractelement <8 x float> %1419, i64 5, !dbg !6
  %1450 = extractelement <8 x float> %1419, i64 6, !dbg !6
  %1451 = extractelement <8 x float> %1419, i64 7, !dbg !6
  %1452 = select i1 %.not, float %1449, float %1448, !dbg !6
  %1453 = bitcast float %1452 to i32, !dbg !6
  %1454 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1453), !dbg !6
  %1455 = bitcast i32 %1454 to float, !dbg !6
  %1456 = select i1 %.not, float %1448, float %1455, !dbg !6
  %1457 = select i1 %.not, float %1455, float %1449, !dbg !6
  %1458 = select i1 %.not, float %1451, float %1450, !dbg !6
  %1459 = bitcast float %1458 to i32, !dbg !6
  %1460 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1459), !dbg !6
  %1461 = bitcast i32 %1460 to float, !dbg !6
  %1462 = select i1 %.not, float %1450, float %1461, !dbg !6
  %1463 = select i1 %.not, float %1461, float %1451, !dbg !6
  %1464 = select i1 %.not40, float %1462, float %1456, !dbg !6
  %1465 = bitcast float %1464 to i32, !dbg !6
  %1466 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1465), !dbg !6
  %1467 = bitcast i32 %1466 to float, !dbg !6
  %1468 = select i1 %.not40, float %1456, float %1467, !dbg !6
  %1469 = select i1 %.not40, float %1467, float %1462, !dbg !6
  %1470 = select i1 %.not40, float %1463, float %1457, !dbg !6
  %1471 = bitcast float %1470 to i32, !dbg !6
  %1472 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1471), !dbg !6
  %1473 = bitcast i32 %1472 to float, !dbg !6
  %1474 = select i1 %.not40, float %1457, float %1473, !dbg !6
  %1475 = select i1 %.not40, float %1473, float %1463, !dbg !6
  %1476 = extractvalue [2 x <8 x float>] %134, 1, !dbg !6
  %1477 = extractelement <8 x float> %1476, i64 0, !dbg !6
  %1478 = extractelement <8 x float> %1476, i64 1, !dbg !6
  %1479 = extractelement <8 x float> %1476, i64 2, !dbg !6
  %1480 = extractelement <8 x float> %1476, i64 3, !dbg !6
  %1481 = select i1 %.not, float %1478, float %1477, !dbg !6
  %1482 = bitcast float %1481 to i32, !dbg !6
  %1483 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1482), !dbg !6
  %1484 = bitcast i32 %1483 to float, !dbg !6
  %1485 = select i1 %.not, float %1477, float %1484, !dbg !6
  %1486 = select i1 %.not, float %1484, float %1478, !dbg !6
  %1487 = select i1 %.not, float %1480, float %1479, !dbg !6
  %1488 = bitcast float %1487 to i32, !dbg !6
  %1489 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1488), !dbg !6
  %1490 = bitcast i32 %1489 to float, !dbg !6
  %1491 = select i1 %.not, float %1479, float %1490, !dbg !6
  %1492 = select i1 %.not, float %1490, float %1480, !dbg !6
  %1493 = select i1 %.not40, float %1491, float %1485, !dbg !6
  %1494 = bitcast float %1493 to i32, !dbg !6
  %1495 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1494), !dbg !6
  %1496 = bitcast i32 %1495 to float, !dbg !6
  %1497 = select i1 %.not40, float %1485, float %1496, !dbg !6
  %1498 = select i1 %.not40, float %1496, float %1491, !dbg !6
  %1499 = select i1 %.not40, float %1492, float %1486, !dbg !6
  %1500 = bitcast float %1499 to i32, !dbg !6
  %1501 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1500), !dbg !6
  %1502 = bitcast i32 %1501 to float, !dbg !6
  %1503 = select i1 %.not40, float %1486, float %1502, !dbg !6
  %1504 = select i1 %.not40, float %1502, float %1492, !dbg !6
  %1505 = extractelement <8 x float> %1476, i64 4, !dbg !6
  %1506 = extractelement <8 x float> %1476, i64 5, !dbg !6
  %1507 = extractelement <8 x float> %1476, i64 6, !dbg !6
  %1508 = extractelement <8 x float> %1476, i64 7, !dbg !6
  %1509 = select i1 %.not, float %1506, float %1505, !dbg !6
  %1510 = bitcast float %1509 to i32, !dbg !6
  %1511 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1510), !dbg !6
  %1512 = bitcast i32 %1511 to float, !dbg !6
  %1513 = select i1 %.not, float %1505, float %1512, !dbg !6
  %1514 = select i1 %.not, float %1512, float %1506, !dbg !6
  %1515 = select i1 %.not, float %1508, float %1507, !dbg !6
  %1516 = bitcast float %1515 to i32, !dbg !6
  %1517 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1516), !dbg !6
  %1518 = bitcast i32 %1517 to float, !dbg !6
  %1519 = select i1 %.not, float %1507, float %1518, !dbg !6
  %1520 = select i1 %.not, float %1518, float %1508, !dbg !6
  %1521 = select i1 %.not40, float %1519, float %1513, !dbg !6
  %1522 = bitcast float %1521 to i32, !dbg !6
  %1523 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1522), !dbg !6
  %1524 = bitcast i32 %1523 to float, !dbg !6
  %1525 = select i1 %.not40, float %1513, float %1524, !dbg !6
  %1526 = select i1 %.not40, float %1524, float %1519, !dbg !6
  %1527 = select i1 %.not40, float %1520, float %1514, !dbg !6
  %1528 = bitcast float %1527 to i32, !dbg !6
  %1529 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1528), !dbg !6
  %1530 = bitcast i32 %1529 to float, !dbg !6
  %1531 = select i1 %.not40, float %1514, float %1530, !dbg !6
  %1532 = select i1 %.not40, float %1530, float %1520, !dbg !6
  %1533 = extractvalue [2 x <8 x float>] %133, 0, !dbg !6
  %1534 = extractelement <8 x float> %1533, i64 0, !dbg !6
  %1535 = extractelement <8 x float> %1533, i64 1, !dbg !6
  %1536 = extractelement <8 x float> %1533, i64 2, !dbg !6
  %1537 = extractelement <8 x float> %1533, i64 3, !dbg !6
  %1538 = select i1 %.not, float %1535, float %1534, !dbg !6
  %1539 = bitcast float %1538 to i32, !dbg !6
  %1540 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1539), !dbg !6
  %1541 = bitcast i32 %1540 to float, !dbg !6
  %1542 = select i1 %.not, float %1534, float %1541, !dbg !6
  %1543 = select i1 %.not, float %1541, float %1535, !dbg !6
  %1544 = select i1 %.not, float %1537, float %1536, !dbg !6
  %1545 = bitcast float %1544 to i32, !dbg !6
  %1546 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1545), !dbg !6
  %1547 = bitcast i32 %1546 to float, !dbg !6
  %1548 = select i1 %.not, float %1536, float %1547, !dbg !6
  %1549 = select i1 %.not, float %1547, float %1537, !dbg !6
  %1550 = select i1 %.not40, float %1548, float %1542, !dbg !6
  %1551 = bitcast float %1550 to i32, !dbg !6
  %1552 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1551), !dbg !6
  %1553 = bitcast i32 %1552 to float, !dbg !6
  %1554 = select i1 %.not40, float %1542, float %1553, !dbg !6
  %1555 = select i1 %.not40, float %1553, float %1548, !dbg !6
  %1556 = select i1 %.not40, float %1549, float %1543, !dbg !6
  %1557 = bitcast float %1556 to i32, !dbg !6
  %1558 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1557), !dbg !6
  %1559 = bitcast i32 %1558 to float, !dbg !6
  %1560 = select i1 %.not40, float %1543, float %1559, !dbg !6
  %1561 = select i1 %.not40, float %1559, float %1549, !dbg !6
  %1562 = extractelement <8 x float> %1533, i64 4, !dbg !6
  %1563 = extractelement <8 x float> %1533, i64 5, !dbg !6
  %1564 = extractelement <8 x float> %1533, i64 6, !dbg !6
  %1565 = extractelement <8 x float> %1533, i64 7, !dbg !6
  %1566 = select i1 %.not, float %1563, float %1562, !dbg !6
  %1567 = bitcast float %1566 to i32, !dbg !6
  %1568 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1567), !dbg !6
  %1569 = bitcast i32 %1568 to float, !dbg !6
  %1570 = select i1 %.not, float %1562, float %1569, !dbg !6
  %1571 = select i1 %.not, float %1569, float %1563, !dbg !6
  %1572 = select i1 %.not, float %1565, float %1564, !dbg !6
  %1573 = bitcast float %1572 to i32, !dbg !6
  %1574 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1573), !dbg !6
  %1575 = bitcast i32 %1574 to float, !dbg !6
  %1576 = select i1 %.not, float %1564, float %1575, !dbg !6
  %1577 = select i1 %.not, float %1575, float %1565, !dbg !6
  %1578 = select i1 %.not40, float %1576, float %1570, !dbg !6
  %1579 = bitcast float %1578 to i32, !dbg !6
  %1580 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1579), !dbg !6
  %1581 = bitcast i32 %1580 to float, !dbg !6
  %1582 = select i1 %.not40, float %1570, float %1581, !dbg !6
  %1583 = select i1 %.not40, float %1581, float %1576, !dbg !6
  %1584 = select i1 %.not40, float %1577, float %1571, !dbg !6
  %1585 = bitcast float %1584 to i32, !dbg !6
  %1586 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1585), !dbg !6
  %1587 = bitcast i32 %1586 to float, !dbg !6
  %1588 = select i1 %.not40, float %1571, float %1587, !dbg !6
  %1589 = select i1 %.not40, float %1587, float %1577, !dbg !6
  %1590 = extractvalue [2 x <8 x float>] %133, 1, !dbg !6
  %1591 = extractelement <8 x float> %1590, i64 0, !dbg !6
  %1592 = extractelement <8 x float> %1590, i64 1, !dbg !6
  %1593 = extractelement <8 x float> %1590, i64 2, !dbg !6
  %1594 = extractelement <8 x float> %1590, i64 3, !dbg !6
  %1595 = select i1 %.not, float %1592, float %1591, !dbg !6
  %1596 = bitcast float %1595 to i32, !dbg !6
  %1597 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1596), !dbg !6
  %1598 = bitcast i32 %1597 to float, !dbg !6
  %1599 = select i1 %.not, float %1591, float %1598, !dbg !6
  %1600 = select i1 %.not, float %1598, float %1592, !dbg !6
  %1601 = select i1 %.not, float %1594, float %1593, !dbg !6
  %1602 = bitcast float %1601 to i32, !dbg !6
  %1603 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1602), !dbg !6
  %1604 = bitcast i32 %1603 to float, !dbg !6
  %1605 = select i1 %.not, float %1593, float %1604, !dbg !6
  %1606 = select i1 %.not, float %1604, float %1594, !dbg !6
  %1607 = select i1 %.not40, float %1605, float %1599, !dbg !6
  %1608 = bitcast float %1607 to i32, !dbg !6
  %1609 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1608), !dbg !6
  %1610 = bitcast i32 %1609 to float, !dbg !6
  %1611 = select i1 %.not40, float %1599, float %1610, !dbg !6
  %1612 = select i1 %.not40, float %1610, float %1605, !dbg !6
  %1613 = select i1 %.not40, float %1606, float %1600, !dbg !6
  %1614 = bitcast float %1613 to i32, !dbg !6
  %1615 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1614), !dbg !6
  %1616 = bitcast i32 %1615 to float, !dbg !6
  %1617 = select i1 %.not40, float %1600, float %1616, !dbg !6
  %1618 = select i1 %.not40, float %1616, float %1606, !dbg !6
  %1619 = extractelement <8 x float> %1590, i64 4, !dbg !6
  %1620 = extractelement <8 x float> %1590, i64 5, !dbg !6
  %1621 = extractelement <8 x float> %1590, i64 6, !dbg !6
  %1622 = extractelement <8 x float> %1590, i64 7, !dbg !6
  %1623 = select i1 %.not, float %1620, float %1619, !dbg !6
  %1624 = bitcast float %1623 to i32, !dbg !6
  %1625 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1624), !dbg !6
  %1626 = bitcast i32 %1625 to float, !dbg !6
  %1627 = select i1 %.not, float %1619, float %1626, !dbg !6
  %1628 = select i1 %.not, float %1626, float %1620, !dbg !6
  %1629 = select i1 %.not, float %1622, float %1621, !dbg !6
  %1630 = bitcast float %1629 to i32, !dbg !6
  %1631 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1630), !dbg !6
  %1632 = bitcast i32 %1631 to float, !dbg !6
  %1633 = select i1 %.not, float %1621, float %1632, !dbg !6
  %1634 = select i1 %.not, float %1632, float %1622, !dbg !6
  %1635 = select i1 %.not40, float %1633, float %1627, !dbg !6
  %1636 = bitcast float %1635 to i32, !dbg !6
  %1637 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1636), !dbg !6
  %1638 = bitcast i32 %1637 to float, !dbg !6
  %1639 = select i1 %.not40, float %1627, float %1638, !dbg !6
  %1640 = select i1 %.not40, float %1638, float %1633, !dbg !6
  %1641 = select i1 %.not40, float %1634, float %1628, !dbg !6
  %1642 = bitcast float %1641 to i32, !dbg !6
  %1643 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1642), !dbg !6
  %1644 = bitcast i32 %1643 to float, !dbg !6
  %1645 = select i1 %.not40, float %1628, float %1644, !dbg !6
  %1646 = select i1 %.not40, float %1644, float %1634, !dbg !6
  %1647 = extractvalue [2 x <8 x float>] %132, 0, !dbg !6
  %1648 = extractelement <8 x float> %1647, i64 0, !dbg !6
  %1649 = extractelement <8 x float> %1647, i64 1, !dbg !6
  %1650 = extractelement <8 x float> %1647, i64 2, !dbg !6
  %1651 = extractelement <8 x float> %1647, i64 3, !dbg !6
  %1652 = select i1 %.not, float %1649, float %1648, !dbg !6
  %1653 = bitcast float %1652 to i32, !dbg !6
  %1654 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1653), !dbg !6
  %1655 = bitcast i32 %1654 to float, !dbg !6
  %1656 = select i1 %.not, float %1648, float %1655, !dbg !6
  %1657 = select i1 %.not, float %1655, float %1649, !dbg !6
  %1658 = select i1 %.not, float %1651, float %1650, !dbg !6
  %1659 = bitcast float %1658 to i32, !dbg !6
  %1660 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1659), !dbg !6
  %1661 = bitcast i32 %1660 to float, !dbg !6
  %1662 = select i1 %.not, float %1650, float %1661, !dbg !6
  %1663 = select i1 %.not, float %1661, float %1651, !dbg !6
  %1664 = select i1 %.not40, float %1662, float %1656, !dbg !6
  %1665 = bitcast float %1664 to i32, !dbg !6
  %1666 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1665), !dbg !6
  %1667 = bitcast i32 %1666 to float, !dbg !6
  %1668 = select i1 %.not40, float %1656, float %1667, !dbg !6
  %1669 = select i1 %.not40, float %1667, float %1662, !dbg !6
  %1670 = select i1 %.not40, float %1663, float %1657, !dbg !6
  %1671 = bitcast float %1670 to i32, !dbg !6
  %1672 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1671), !dbg !6
  %1673 = bitcast i32 %1672 to float, !dbg !6
  %1674 = select i1 %.not40, float %1657, float %1673, !dbg !6
  %1675 = select i1 %.not40, float %1673, float %1663, !dbg !6
  %1676 = extractelement <8 x float> %1647, i64 4, !dbg !6
  %1677 = extractelement <8 x float> %1647, i64 5, !dbg !6
  %1678 = extractelement <8 x float> %1647, i64 6, !dbg !6
  %1679 = extractelement <8 x float> %1647, i64 7, !dbg !6
  %1680 = select i1 %.not, float %1677, float %1676, !dbg !6
  %1681 = bitcast float %1680 to i32, !dbg !6
  %1682 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1681), !dbg !6
  %1683 = bitcast i32 %1682 to float, !dbg !6
  %1684 = select i1 %.not, float %1676, float %1683, !dbg !6
  %1685 = select i1 %.not, float %1683, float %1677, !dbg !6
  %1686 = select i1 %.not, float %1679, float %1678, !dbg !6
  %1687 = bitcast float %1686 to i32, !dbg !6
  %1688 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1687), !dbg !6
  %1689 = bitcast i32 %1688 to float, !dbg !6
  %1690 = select i1 %.not, float %1678, float %1689, !dbg !6
  %1691 = select i1 %.not, float %1689, float %1679, !dbg !6
  %1692 = select i1 %.not40, float %1690, float %1684, !dbg !6
  %1693 = bitcast float %1692 to i32, !dbg !6
  %1694 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1693), !dbg !6
  %1695 = bitcast i32 %1694 to float, !dbg !6
  %1696 = select i1 %.not40, float %1684, float %1695, !dbg !6
  %1697 = select i1 %.not40, float %1695, float %1690, !dbg !6
  %1698 = select i1 %.not40, float %1691, float %1685, !dbg !6
  %1699 = bitcast float %1698 to i32, !dbg !6
  %1700 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1699), !dbg !6
  %1701 = bitcast i32 %1700 to float, !dbg !6
  %1702 = select i1 %.not40, float %1685, float %1701, !dbg !6
  %1703 = select i1 %.not40, float %1701, float %1691, !dbg !6
  %1704 = extractvalue [2 x <8 x float>] %132, 1, !dbg !6
  %1705 = extractelement <8 x float> %1704, i64 0, !dbg !6
  %1706 = extractelement <8 x float> %1704, i64 1, !dbg !6
  %1707 = extractelement <8 x float> %1704, i64 2, !dbg !6
  %1708 = extractelement <8 x float> %1704, i64 3, !dbg !6
  %1709 = select i1 %.not, float %1706, float %1705, !dbg !6
  %1710 = bitcast float %1709 to i32, !dbg !6
  %1711 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1710), !dbg !6
  %1712 = bitcast i32 %1711 to float, !dbg !6
  %1713 = select i1 %.not, float %1705, float %1712, !dbg !6
  %1714 = select i1 %.not, float %1712, float %1706, !dbg !6
  %1715 = select i1 %.not, float %1708, float %1707, !dbg !6
  %1716 = bitcast float %1715 to i32, !dbg !6
  %1717 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1716), !dbg !6
  %1718 = bitcast i32 %1717 to float, !dbg !6
  %1719 = select i1 %.not, float %1707, float %1718, !dbg !6
  %1720 = select i1 %.not, float %1718, float %1708, !dbg !6
  %1721 = select i1 %.not40, float %1719, float %1713, !dbg !6
  %1722 = bitcast float %1721 to i32, !dbg !6
  %1723 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1722), !dbg !6
  %1724 = bitcast i32 %1723 to float, !dbg !6
  %1725 = select i1 %.not40, float %1713, float %1724, !dbg !6
  %1726 = select i1 %.not40, float %1724, float %1719, !dbg !6
  %1727 = select i1 %.not40, float %1720, float %1714, !dbg !6
  %1728 = bitcast float %1727 to i32, !dbg !6
  %1729 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1728), !dbg !6
  %1730 = bitcast i32 %1729 to float, !dbg !6
  %1731 = select i1 %.not40, float %1714, float %1730, !dbg !6
  %1732 = select i1 %.not40, float %1730, float %1720, !dbg !6
  %1733 = extractelement <8 x float> %1704, i64 4, !dbg !6
  %1734 = extractelement <8 x float> %1704, i64 5, !dbg !6
  %1735 = extractelement <8 x float> %1704, i64 6, !dbg !6
  %1736 = extractelement <8 x float> %1704, i64 7, !dbg !6
  %1737 = select i1 %.not, float %1734, float %1733, !dbg !6
  %1738 = bitcast float %1737 to i32, !dbg !6
  %1739 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1738), !dbg !6
  %1740 = bitcast i32 %1739 to float, !dbg !6
  %1741 = select i1 %.not, float %1733, float %1740, !dbg !6
  %1742 = select i1 %.not, float %1740, float %1734, !dbg !6
  %1743 = select i1 %.not, float %1736, float %1735, !dbg !6
  %1744 = bitcast float %1743 to i32, !dbg !6
  %1745 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1744), !dbg !6
  %1746 = bitcast i32 %1745 to float, !dbg !6
  %1747 = select i1 %.not, float %1735, float %1746, !dbg !6
  %1748 = select i1 %.not, float %1746, float %1736, !dbg !6
  %1749 = select i1 %.not40, float %1747, float %1741, !dbg !6
  %1750 = bitcast float %1749 to i32, !dbg !6
  %1751 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1750), !dbg !6
  %1752 = bitcast i32 %1751 to float, !dbg !6
  %1753 = select i1 %.not40, float %1741, float %1752, !dbg !6
  %1754 = select i1 %.not40, float %1752, float %1747, !dbg !6
  %1755 = select i1 %.not40, float %1748, float %1742, !dbg !6
  %1756 = bitcast float %1755 to i32, !dbg !6
  %1757 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1756), !dbg !6
  %1758 = bitcast i32 %1757 to float, !dbg !6
  %1759 = select i1 %.not40, float %1742, float %1758, !dbg !6
  %1760 = select i1 %.not40, float %1758, float %1748, !dbg !6
  %1761 = extractvalue [2 x <8 x float>] %129, 0, !dbg !6
  %1762 = extractelement <8 x float> %1761, i64 0, !dbg !6
  %1763 = extractelement <8 x float> %1761, i64 1, !dbg !6
  %1764 = extractelement <8 x float> %1761, i64 2, !dbg !6
  %1765 = extractelement <8 x float> %1761, i64 3, !dbg !6
  %1766 = select i1 %.not, float %1763, float %1762, !dbg !6
  %1767 = bitcast float %1766 to i32, !dbg !6
  %1768 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1767), !dbg !6
  %1769 = bitcast i32 %1768 to float, !dbg !6
  %1770 = select i1 %.not, float %1762, float %1769, !dbg !6
  %1771 = select i1 %.not, float %1769, float %1763, !dbg !6
  %1772 = select i1 %.not, float %1765, float %1764, !dbg !6
  %1773 = bitcast float %1772 to i32, !dbg !6
  %1774 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1773), !dbg !6
  %1775 = bitcast i32 %1774 to float, !dbg !6
  %1776 = select i1 %.not, float %1764, float %1775, !dbg !6
  %1777 = select i1 %.not, float %1775, float %1765, !dbg !6
  %1778 = select i1 %.not40, float %1776, float %1770, !dbg !6
  %1779 = bitcast float %1778 to i32, !dbg !6
  %1780 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1779), !dbg !6
  %1781 = select i1 %.not40, float %1777, float %1771, !dbg !6
  %1782 = bitcast float %1781 to i32, !dbg !6
  %1783 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1782), !dbg !6
  %1784 = extractelement <8 x float> %1761, i64 4, !dbg !6
  %1785 = extractelement <8 x float> %1761, i64 5, !dbg !6
  %1786 = extractelement <8 x float> %1761, i64 6, !dbg !6
  %1787 = extractelement <8 x float> %1761, i64 7, !dbg !6
  %1788 = select i1 %.not, float %1785, float %1784, !dbg !6
  %1789 = bitcast float %1788 to i32, !dbg !6
  %1790 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1789), !dbg !6
  %1791 = bitcast i32 %1790 to float, !dbg !6
  %1792 = select i1 %.not, float %1784, float %1791, !dbg !6
  %1793 = select i1 %.not, float %1791, float %1785, !dbg !6
  %1794 = select i1 %.not, float %1787, float %1786, !dbg !6
  %1795 = bitcast float %1794 to i32, !dbg !6
  %1796 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1795), !dbg !6
  %1797 = bitcast i32 %1796 to float, !dbg !6
  %1798 = select i1 %.not, float %1786, float %1797, !dbg !6
  %1799 = select i1 %.not, float %1797, float %1787, !dbg !6
  %1800 = select i1 %.not40, float %1798, float %1792, !dbg !6
  %1801 = bitcast float %1800 to i32, !dbg !6
  %1802 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1801), !dbg !6
  %1803 = select i1 %.not40, float %1799, float %1793, !dbg !6
  %1804 = bitcast float %1803 to i32, !dbg !6
  %1805 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1804), !dbg !6
  %1806 = extractvalue [2 x <8 x float>] %129, 1, !dbg !6
  %1807 = extractelement <8 x float> %1806, i64 0, !dbg !6
  %1808 = extractelement <8 x float> %1806, i64 1, !dbg !6
  %1809 = extractelement <8 x float> %1806, i64 2, !dbg !6
  %1810 = extractelement <8 x float> %1806, i64 3, !dbg !6
  %1811 = select i1 %.not, float %1808, float %1807, !dbg !6
  %1812 = bitcast float %1811 to i32, !dbg !6
  %1813 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1812), !dbg !6
  %1814 = bitcast i32 %1813 to float, !dbg !6
  %1815 = select i1 %.not, float %1807, float %1814, !dbg !6
  %1816 = select i1 %.not, float %1814, float %1808, !dbg !6
  %1817 = select i1 %.not, float %1810, float %1809, !dbg !6
  %1818 = bitcast float %1817 to i32, !dbg !6
  %1819 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1818), !dbg !6
  %1820 = bitcast i32 %1819 to float, !dbg !6
  %1821 = select i1 %.not, float %1809, float %1820, !dbg !6
  %1822 = select i1 %.not, float %1820, float %1810, !dbg !6
  %1823 = select i1 %.not40, float %1821, float %1815, !dbg !6
  %1824 = bitcast float %1823 to i32, !dbg !6
  %1825 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1824), !dbg !6
  %1826 = select i1 %.not40, float %1822, float %1816, !dbg !6
  %1827 = bitcast float %1826 to i32, !dbg !6
  %1828 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1827), !dbg !6
  %1829 = extractelement <8 x float> %1806, i64 4, !dbg !6
  %1830 = extractelement <8 x float> %1806, i64 5, !dbg !6
  %1831 = extractelement <8 x float> %1806, i64 6, !dbg !6
  %1832 = extractelement <8 x float> %1806, i64 7, !dbg !6
  %1833 = select i1 %.not, float %1830, float %1829, !dbg !6
  %1834 = bitcast float %1833 to i32, !dbg !6
  %1835 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1834), !dbg !6
  %1836 = bitcast i32 %1835 to float, !dbg !6
  %1837 = select i1 %.not, float %1829, float %1836, !dbg !6
  %1838 = select i1 %.not, float %1836, float %1830, !dbg !6
  %1839 = select i1 %.not, float %1832, float %1831, !dbg !6
  %1840 = bitcast float %1839 to i32, !dbg !6
  %1841 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1840), !dbg !6
  %1842 = bitcast i32 %1841 to float, !dbg !6
  %1843 = select i1 %.not, float %1831, float %1842, !dbg !6
  %1844 = select i1 %.not, float %1842, float %1832, !dbg !6
  %1845 = select i1 %.not40, float %1843, float %1837, !dbg !6
  %1846 = bitcast float %1845 to i32, !dbg !6
  %1847 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1846), !dbg !6
  %1848 = select i1 %.not40, float %1844, float %1838, !dbg !6
  %1849 = bitcast float %1848 to i32, !dbg !6
  %1850 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1849), !dbg !6
  %1851 = insertelement <4 x float> poison, float %1416, i64 0, !dbg !20
  %1852 = fdiv arcp <4 x float> <float 0x3FF0000000000000, float poison, float poison, float poison>, %1851, !dbg !20
  %1853 = shufflevector <4 x float> %1852, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !20
  %1854 = insertelement <4 x float> poison, float %1406, i64 0, !dbg !20
  %1855 = fdiv arcp <4 x float> <float 0x3FF0000000000000, float poison, float poison, float poison>, %1854, !dbg !20
  %1856 = shufflevector <4 x float> %1855, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !20
  %1857 = bitcast i32 %1847 to float, !dbg !6
  %1858 = select i1 %.not40, float %1837, float %1857, !dbg !6
  %1859 = insertelement <4 x float> poison, float %1858, i64 0, !dbg !20
  %1860 = bitcast i32 %1850 to float, !dbg !6
  %1861 = select i1 %.not40, float %1838, float %1860, !dbg !6
  %1862 = insertelement <4 x float> %1859, float %1861, i64 1, !dbg !20
  %1863 = select i1 %.not40, float %1857, float %1843, !dbg !6
  %1864 = insertelement <4 x float> %1862, float %1863, i64 2, !dbg !20
  %1865 = select i1 %.not40, float %1860, float %1844, !dbg !6
  %1866 = insertelement <4 x float> %1864, float %1865, i64 3, !dbg !20
  %1867 = fmul <4 x float> %1866, %1856, !dbg !20
  %1868 = fptrunc <4 x float> %1867 to <4 x half>, !dbg !21
  %1869 = extractelement <4 x half> %1868, i64 3, !dbg !21
  %1870 = extractelement <4 x half> %1868, i64 2, !dbg !21
  %1871 = extractelement <4 x half> %1868, i64 1, !dbg !21
  %1872 = extractelement <4 x half> %1868, i64 0, !dbg !21
  %1873 = bitcast i32 %1825 to float, !dbg !6
  %1874 = select i1 %.not40, float %1815, float %1873, !dbg !6
  %1875 = insertelement <4 x float> poison, float %1874, i64 0, !dbg !20
  %1876 = bitcast i32 %1828 to float, !dbg !6
  %1877 = select i1 %.not40, float %1816, float %1876, !dbg !6
  %1878 = insertelement <4 x float> %1875, float %1877, i64 1, !dbg !20
  %1879 = select i1 %.not40, float %1873, float %1821, !dbg !6
  %1880 = insertelement <4 x float> %1878, float %1879, i64 2, !dbg !20
  %1881 = select i1 %.not40, float %1876, float %1822, !dbg !6
  %1882 = insertelement <4 x float> %1880, float %1881, i64 3, !dbg !20
  %1883 = fmul <4 x float> %1882, %1856, !dbg !20
  %1884 = fptrunc <4 x float> %1883 to <4 x half>, !dbg !21
  %1885 = extractelement <4 x half> %1884, i64 3, !dbg !21
  %1886 = extractelement <4 x half> %1884, i64 2, !dbg !21
  %1887 = extractelement <4 x half> %1884, i64 1, !dbg !21
  %1888 = extractelement <4 x half> %1884, i64 0, !dbg !21
  %1889 = bitcast i32 %1802 to float, !dbg !6
  %1890 = select i1 %.not40, float %1792, float %1889, !dbg !6
  %1891 = insertelement <4 x float> poison, float %1890, i64 0, !dbg !20
  %1892 = bitcast i32 %1805 to float, !dbg !6
  %1893 = select i1 %.not40, float %1793, float %1892, !dbg !6
  %1894 = insertelement <4 x float> %1891, float %1893, i64 1, !dbg !20
  %1895 = select i1 %.not40, float %1889, float %1798, !dbg !6
  %1896 = insertelement <4 x float> %1894, float %1895, i64 2, !dbg !20
  %1897 = select i1 %.not40, float %1892, float %1799, !dbg !6
  %1898 = insertelement <4 x float> %1896, float %1897, i64 3, !dbg !20
  %1899 = fmul <4 x float> %1898, %1853, !dbg !20
  %1900 = fptrunc <4 x float> %1899 to <4 x half>, !dbg !21
  %1901 = extractelement <4 x half> %1900, i64 3, !dbg !21
  %1902 = extractelement <4 x half> %1900, i64 2, !dbg !21
  %1903 = extractelement <4 x half> %1900, i64 1, !dbg !21
  %1904 = extractelement <4 x half> %1900, i64 0, !dbg !21
  %1905 = bitcast i32 %1780 to float, !dbg !6
  %1906 = select i1 %.not40, float %1770, float %1905, !dbg !6
  %1907 = insertelement <4 x float> poison, float %1906, i64 0, !dbg !20
  %1908 = bitcast i32 %1783 to float, !dbg !6
  %1909 = select i1 %.not40, float %1771, float %1908, !dbg !6
  %1910 = insertelement <4 x float> %1907, float %1909, i64 1, !dbg !20
  %1911 = select i1 %.not40, float %1905, float %1776, !dbg !6
  %1912 = insertelement <4 x float> %1910, float %1911, i64 2, !dbg !20
  %1913 = select i1 %.not40, float %1908, float %1777, !dbg !6
  %1914 = insertelement <4 x float> %1912, float %1913, i64 3, !dbg !20
  %1915 = fmul <4 x float> %1914, %1853, !dbg !20
  %1916 = fptrunc <4 x float> %1915 to <4 x half>, !dbg !21
  %1917 = extractelement <4 x half> %1916, i64 3, !dbg !21
  %1918 = extractelement <4 x half> %1916, i64 2, !dbg !21
  %1919 = extractelement <4 x half> %1916, i64 1, !dbg !21
  %1920 = extractelement <4 x half> %1916, i64 0, !dbg !21
  %address.thread904 = call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #9, !dbg !22, !range !33, !frisk.register_lifetimes !1
  %address.local905 = zext nneg i32 %address.thread904 to i64, !dbg !9
  %address.local906 = add i64 %42, %address.local905, !dbg !22
  %address.local907 = lshr i64 %address.local905, 2, !dbg !19
  %address.local908 = and i64 %address.local907, 48, !dbg !19
  %address.local909 = add i64 %address.local906, %address.local908, !dbg !22
  %address.local910 = and i64 %address.local905, 240, !dbg !22
  %address.local912 = sub i64 %address.local909, %address.local910, !dbg !22
  %address.local913 = add i64 %address.local912, %43, !dbg !22
  %address.local914 = shl nuw nsw i64 %address.local913, 7, !dbg !22
  %1921 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %60, !dbg !22
  %address.local916 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %1921, i64 %address.local914, !dbg !22
  %address.local918 = and i64 %address.local907, 12, !dbg !13
  %address.local919 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local916, i64 %address.local918, !dbg !22
  store half %1920, ptr addrspace(1) %address.local919, align 2, !dbg !22
  %address.local921 = or disjoint i64 %address.local918, 1, !dbg !22
  %address.local922 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local916, i64 %address.local921, !dbg !22
  store half %1919, ptr addrspace(1) %address.local922, align 2, !dbg !22
  %address.local924 = or disjoint i64 %address.local918, 2, !dbg !22
  %address.local925 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local916, i64 %address.local924, !dbg !22
  store half %1918, ptr addrspace(1) %address.local925, align 2, !dbg !22
  %address.local927 = or disjoint i64 %address.local918, 3, !dbg !22
  %address.local928 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local916, i64 %address.local927, !dbg !22
  store half %1917, ptr addrspace(1) %address.local928, align 2, !dbg !22
  %address.local930 = or disjoint i64 %address.local918, 16, !dbg !22
  %address.local931 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local916, i64 %address.local930, !dbg !22
  store half %1904, ptr addrspace(1) %address.local931, align 2, !dbg !22
  %address.local933 = or disjoint i64 %address.local918, 17, !dbg !22
  %address.local934 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local916, i64 %address.local933, !dbg !22
  store half %1903, ptr addrspace(1) %address.local934, align 2, !dbg !22
  %address.local936 = or disjoint i64 %address.local918, 18, !dbg !22
  %address.local937 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local916, i64 %address.local936, !dbg !22
  store half %1902, ptr addrspace(1) %address.local937, align 2, !dbg !22
  %address.local939 = or disjoint i64 %address.local918, 19, !dbg !22
  %address.local940 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local916, i64 %address.local939, !dbg !22
  store half %1901, ptr addrspace(1) %address.local940, align 2, !dbg !22
  %address.local944 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local916, i64 16384, !dbg !22
  %address.local945 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local944, i64 %address.local918, !dbg !22
  store half %1888, ptr addrspace(1) %address.local945, align 2, !dbg !22
  %address.local947 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local944, i64 %address.local921, !dbg !22
  store half %1887, ptr addrspace(1) %address.local947, align 2, !dbg !22
  %address.local949 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local944, i64 %address.local924, !dbg !22
  store half %1886, ptr addrspace(1) %address.local949, align 2, !dbg !22
  %address.local951 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local944, i64 %address.local927, !dbg !22
  store half %1885, ptr addrspace(1) %address.local951, align 2, !dbg !22
  %address.local953 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local944, i64 %address.local930, !dbg !22
  store half %1872, ptr addrspace(1) %address.local953, align 2, !dbg !22
  %address.local955 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local944, i64 %address.local933, !dbg !22
  store half %1871, ptr addrspace(1) %address.local955, align 2, !dbg !22
  %address.local957 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local944, i64 %address.local936, !dbg !22
  store half %1870, ptr addrspace(1) %address.local957, align 2, !dbg !22
  %address.local959 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local944, i64 %address.local939, !dbg !22
  store half %1869, ptr addrspace(1) %address.local959, align 2, !dbg !22
  %1922 = insertelement <4 x float> poison, float %1753, i64 0, !dbg !20
  %1923 = insertelement <4 x float> %1922, float %1759, i64 1, !dbg !20
  %1924 = insertelement <4 x float> %1923, float %1754, i64 2, !dbg !20
  %1925 = insertelement <4 x float> %1924, float %1760, i64 3, !dbg !20
  %1926 = fmul <4 x float> %1925, %1856, !dbg !20
  %1927 = fptrunc <4 x float> %1926 to <4 x half>, !dbg !21
  %1928 = extractelement <4 x half> %1927, i64 3, !dbg !21
  %1929 = extractelement <4 x half> %1927, i64 2, !dbg !21
  %1930 = extractelement <4 x half> %1927, i64 1, !dbg !21
  %1931 = extractelement <4 x half> %1927, i64 0, !dbg !21
  %1932 = insertelement <4 x float> poison, float %1725, i64 0, !dbg !20
  %1933 = insertelement <4 x float> %1932, float %1731, i64 1, !dbg !20
  %1934 = insertelement <4 x float> %1933, float %1726, i64 2, !dbg !20
  %1935 = insertelement <4 x float> %1934, float %1732, i64 3, !dbg !20
  %1936 = fmul <4 x float> %1935, %1856, !dbg !20
  %1937 = fptrunc <4 x float> %1936 to <4 x half>, !dbg !21
  %1938 = extractelement <4 x half> %1937, i64 3, !dbg !21
  %1939 = extractelement <4 x half> %1937, i64 2, !dbg !21
  %1940 = extractelement <4 x half> %1937, i64 1, !dbg !21
  %1941 = extractelement <4 x half> %1937, i64 0, !dbg !21
  %1942 = insertelement <4 x float> poison, float %1696, i64 0, !dbg !20
  %1943 = insertelement <4 x float> %1942, float %1702, i64 1, !dbg !20
  %1944 = insertelement <4 x float> %1943, float %1697, i64 2, !dbg !20
  %1945 = insertelement <4 x float> %1944, float %1703, i64 3, !dbg !20
  %1946 = fmul <4 x float> %1945, %1853, !dbg !20
  %1947 = fptrunc <4 x float> %1946 to <4 x half>, !dbg !21
  %1948 = extractelement <4 x half> %1947, i64 3, !dbg !21
  %1949 = extractelement <4 x half> %1947, i64 2, !dbg !21
  %1950 = extractelement <4 x half> %1947, i64 1, !dbg !21
  %1951 = extractelement <4 x half> %1947, i64 0, !dbg !21
  %1952 = insertelement <4 x float> poison, float %1668, i64 0, !dbg !20
  %1953 = insertelement <4 x float> %1952, float %1674, i64 1, !dbg !20
  %1954 = insertelement <4 x float> %1953, float %1669, i64 2, !dbg !20
  %1955 = insertelement <4 x float> %1954, float %1675, i64 3, !dbg !20
  %1956 = fmul <4 x float> %1955, %1853, !dbg !20
  %1957 = fptrunc <4 x float> %1956 to <4 x half>, !dbg !21
  %1958 = extractelement <4 x half> %1957, i64 3, !dbg !21
  %1959 = extractelement <4 x half> %1957, i64 2, !dbg !21
  %1960 = extractelement <4 x half> %1957, i64 1, !dbg !21
  %1961 = extractelement <4 x half> %1957, i64 0, !dbg !21
  %address.local975 = or disjoint i64 %address.local918, 32, !dbg !22
  %address.local976 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local916, i64 %address.local975, !dbg !22
  store half %1961, ptr addrspace(1) %address.local976, align 2, !dbg !22
  %address.local978 = or disjoint i64 %address.local918, 33, !dbg !22
  %address.local979 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local916, i64 %address.local978, !dbg !22
  store half %1960, ptr addrspace(1) %address.local979, align 2, !dbg !22
  %address.local981 = or disjoint i64 %address.local918, 34, !dbg !22
  %address.local982 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local916, i64 %address.local981, !dbg !22
  store half %1959, ptr addrspace(1) %address.local982, align 2, !dbg !22
  %address.local984 = or disjoint i64 %address.local918, 35, !dbg !22
  %address.local985 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local916, i64 %address.local984, !dbg !22
  store half %1958, ptr addrspace(1) %address.local985, align 2, !dbg !22
  %address.local987 = or disjoint i64 %address.local918, 48, !dbg !22
  %address.local988 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local916, i64 %address.local987, !dbg !22
  store half %1951, ptr addrspace(1) %address.local988, align 2, !dbg !22
  %address.local990 = or disjoint i64 %address.local918, 49, !dbg !22
  %address.local991 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local916, i64 %address.local990, !dbg !22
  store half %1950, ptr addrspace(1) %address.local991, align 2, !dbg !22
  %address.local993 = or disjoint i64 %address.local918, 50, !dbg !22
  %address.local994 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local916, i64 %address.local993, !dbg !22
  store half %1949, ptr addrspace(1) %address.local994, align 2, !dbg !22
  %address.local996 = or i64 %address.local907, 51, !dbg !22
  %address.local997 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local916, i64 %address.local996, !dbg !22
  store half %1948, ptr addrspace(1) %address.local997, align 2, !dbg !22
  %address.local1002 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local944, i64 %address.local975, !dbg !22
  store half %1941, ptr addrspace(1) %address.local1002, align 2, !dbg !22
  %address.local1004 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local944, i64 %address.local978, !dbg !22
  store half %1940, ptr addrspace(1) %address.local1004, align 2, !dbg !22
  %address.local1006 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local944, i64 %address.local981, !dbg !22
  store half %1939, ptr addrspace(1) %address.local1006, align 2, !dbg !22
  %address.local1008 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local944, i64 %address.local984, !dbg !22
  store half %1938, ptr addrspace(1) %address.local1008, align 2, !dbg !22
  %address.local1010 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local944, i64 %address.local987, !dbg !22
  store half %1931, ptr addrspace(1) %address.local1010, align 2, !dbg !22
  %address.local1012 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local944, i64 %address.local990, !dbg !22
  store half %1930, ptr addrspace(1) %address.local1012, align 2, !dbg !22
  %address.local1014 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local944, i64 %address.local993, !dbg !22
  store half %1929, ptr addrspace(1) %address.local1014, align 2, !dbg !22
  %address.local1016 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local944, i64 %address.local996, !dbg !22
  store half %1928, ptr addrspace(1) %address.local1016, align 2, !dbg !22
  %1962 = insertelement <4 x float> poison, float %1639, i64 0, !dbg !20
  %1963 = insertelement <4 x float> %1962, float %1645, i64 1, !dbg !20
  %1964 = insertelement <4 x float> %1963, float %1640, i64 2, !dbg !20
  %1965 = insertelement <4 x float> %1964, float %1646, i64 3, !dbg !20
  %1966 = fmul <4 x float> %1965, %1856, !dbg !20
  %1967 = fptrunc <4 x float> %1966 to <4 x half>, !dbg !21
  %1968 = extractelement <4 x half> %1967, i64 3, !dbg !21
  %1969 = extractelement <4 x half> %1967, i64 2, !dbg !21
  %1970 = extractelement <4 x half> %1967, i64 1, !dbg !21
  %1971 = extractelement <4 x half> %1967, i64 0, !dbg !21
  %1972 = insertelement <4 x float> poison, float %1611, i64 0, !dbg !20
  %1973 = insertelement <4 x float> %1972, float %1617, i64 1, !dbg !20
  %1974 = insertelement <4 x float> %1973, float %1612, i64 2, !dbg !20
  %1975 = insertelement <4 x float> %1974, float %1618, i64 3, !dbg !20
  %1976 = fmul <4 x float> %1975, %1856, !dbg !20
  %1977 = fptrunc <4 x float> %1976 to <4 x half>, !dbg !21
  %1978 = extractelement <4 x half> %1977, i64 3, !dbg !21
  %1979 = extractelement <4 x half> %1977, i64 2, !dbg !21
  %1980 = extractelement <4 x half> %1977, i64 1, !dbg !21
  %1981 = extractelement <4 x half> %1977, i64 0, !dbg !21
  %1982 = insertelement <4 x float> poison, float %1582, i64 0, !dbg !20
  %1983 = insertelement <4 x float> %1982, float %1588, i64 1, !dbg !20
  %1984 = insertelement <4 x float> %1983, float %1583, i64 2, !dbg !20
  %1985 = insertelement <4 x float> %1984, float %1589, i64 3, !dbg !20
  %1986 = fmul <4 x float> %1985, %1853, !dbg !20
  %1987 = fptrunc <4 x float> %1986 to <4 x half>, !dbg !21
  %1988 = extractelement <4 x half> %1987, i64 3, !dbg !21
  %1989 = extractelement <4 x half> %1987, i64 2, !dbg !21
  %1990 = extractelement <4 x half> %1987, i64 1, !dbg !21
  %1991 = extractelement <4 x half> %1987, i64 0, !dbg !21
  %1992 = insertelement <4 x float> poison, float %1554, i64 0, !dbg !20
  %1993 = insertelement <4 x float> %1992, float %1560, i64 1, !dbg !20
  %1994 = insertelement <4 x float> %1993, float %1555, i64 2, !dbg !20
  %1995 = insertelement <4 x float> %1994, float %1561, i64 3, !dbg !20
  %1996 = fmul <4 x float> %1995, %1853, !dbg !20
  %1997 = fptrunc <4 x float> %1996 to <4 x half>, !dbg !21
  %1998 = extractelement <4 x half> %1997, i64 3, !dbg !21
  %1999 = extractelement <4 x half> %1997, i64 2, !dbg !21
  %2000 = extractelement <4 x half> %1997, i64 1, !dbg !21
  %2001 = extractelement <4 x half> %1997, i64 0, !dbg !21
  %address.local1032 = or disjoint i64 %address.local918, 64, !dbg !22
  %address.local1033 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local916, i64 %address.local1032, !dbg !22
  store half %2001, ptr addrspace(1) %address.local1033, align 2, !dbg !22
  %address.local1035 = or disjoint i64 %address.local918, 65, !dbg !22
  %address.local1036 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local916, i64 %address.local1035, !dbg !22
  store half %2000, ptr addrspace(1) %address.local1036, align 2, !dbg !22
  %address.local1038 = or disjoint i64 %address.local918, 66, !dbg !22
  %address.local1039 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local916, i64 %address.local1038, !dbg !22
  store half %1999, ptr addrspace(1) %address.local1039, align 2, !dbg !22
  %address.local1041 = or disjoint i64 %address.local918, 67, !dbg !22
  %address.local1042 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local916, i64 %address.local1041, !dbg !22
  store half %1998, ptr addrspace(1) %address.local1042, align 2, !dbg !22
  %address.local1044 = or disjoint i64 %address.local918, 80, !dbg !22
  %address.local1045 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local916, i64 %address.local1044, !dbg !22
  store half %1991, ptr addrspace(1) %address.local1045, align 2, !dbg !22
  %address.local1047 = or disjoint i64 %address.local918, 81, !dbg !22
  %address.local1048 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local916, i64 %address.local1047, !dbg !22
  store half %1990, ptr addrspace(1) %address.local1048, align 2, !dbg !22
  %address.local1050 = or disjoint i64 %address.local918, 82, !dbg !22
  %address.local1051 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local916, i64 %address.local1050, !dbg !22
  store half %1989, ptr addrspace(1) %address.local1051, align 2, !dbg !22
  %address.local1053 = or disjoint i64 %address.local918, 83, !dbg !22
  %address.local1054 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local916, i64 %address.local1053, !dbg !22
  store half %1988, ptr addrspace(1) %address.local1054, align 2, !dbg !22
  %address.local1059 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local944, i64 %address.local1032, !dbg !22
  store half %1981, ptr addrspace(1) %address.local1059, align 2, !dbg !22
  %address.local1061 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local944, i64 %address.local1035, !dbg !22
  store half %1980, ptr addrspace(1) %address.local1061, align 2, !dbg !22
  %address.local1063 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local944, i64 %address.local1038, !dbg !22
  store half %1979, ptr addrspace(1) %address.local1063, align 2, !dbg !22
  %address.local1065 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local944, i64 %address.local1041, !dbg !22
  store half %1978, ptr addrspace(1) %address.local1065, align 2, !dbg !22
  %address.local1067 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local944, i64 %address.local1044, !dbg !22
  store half %1971, ptr addrspace(1) %address.local1067, align 2, !dbg !22
  %address.local1069 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local944, i64 %address.local1047, !dbg !22
  store half %1970, ptr addrspace(1) %address.local1069, align 2, !dbg !22
  %address.local1071 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local944, i64 %address.local1050, !dbg !22
  store half %1969, ptr addrspace(1) %address.local1071, align 2, !dbg !22
  %address.local1073 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local944, i64 %address.local1053, !dbg !22
  store half %1968, ptr addrspace(1) %address.local1073, align 2, !dbg !22
  %2002 = insertelement <4 x float> poison, float %1525, i64 0, !dbg !20
  %2003 = insertelement <4 x float> %2002, float %1531, i64 1, !dbg !20
  %2004 = insertelement <4 x float> %2003, float %1526, i64 2, !dbg !20
  %2005 = insertelement <4 x float> %2004, float %1532, i64 3, !dbg !20
  %2006 = fmul <4 x float> %2005, %1856, !dbg !20
  %2007 = fptrunc <4 x float> %2006 to <4 x half>, !dbg !21
  %2008 = extractelement <4 x half> %2007, i64 3, !dbg !21
  %2009 = extractelement <4 x half> %2007, i64 2, !dbg !21
  %2010 = extractelement <4 x half> %2007, i64 1, !dbg !21
  %2011 = extractelement <4 x half> %2007, i64 0, !dbg !21
  %2012 = insertelement <4 x float> poison, float %1497, i64 0, !dbg !20
  %2013 = insertelement <4 x float> %2012, float %1503, i64 1, !dbg !20
  %2014 = insertelement <4 x float> %2013, float %1498, i64 2, !dbg !20
  %2015 = insertelement <4 x float> %2014, float %1504, i64 3, !dbg !20
  %2016 = fmul <4 x float> %2015, %1856, !dbg !20
  %2017 = fptrunc <4 x float> %2016 to <4 x half>, !dbg !21
  %2018 = extractelement <4 x half> %2017, i64 3, !dbg !21
  %2019 = extractelement <4 x half> %2017, i64 2, !dbg !21
  %2020 = extractelement <4 x half> %2017, i64 1, !dbg !21
  %2021 = extractelement <4 x half> %2017, i64 0, !dbg !21
  %2022 = insertelement <4 x float> poison, float %1468, i64 0, !dbg !20
  %2023 = insertelement <4 x float> %2022, float %1474, i64 1, !dbg !20
  %2024 = insertelement <4 x float> %2023, float %1469, i64 2, !dbg !20
  %2025 = insertelement <4 x float> %2024, float %1475, i64 3, !dbg !20
  %2026 = fmul <4 x float> %2025, %1853, !dbg !20
  %2027 = fptrunc <4 x float> %2026 to <4 x half>, !dbg !21
  %2028 = extractelement <4 x half> %2027, i64 3, !dbg !21
  %2029 = extractelement <4 x half> %2027, i64 2, !dbg !21
  %2030 = extractelement <4 x half> %2027, i64 1, !dbg !21
  %2031 = extractelement <4 x half> %2027, i64 0, !dbg !21
  %2032 = insertelement <4 x float> poison, float %1440, i64 0, !dbg !20
  %2033 = insertelement <4 x float> %2032, float %1446, i64 1, !dbg !20
  %2034 = insertelement <4 x float> %2033, float %1441, i64 2, !dbg !20
  %2035 = insertelement <4 x float> %2034, float %1447, i64 3, !dbg !20
  %2036 = fmul <4 x float> %2035, %1853, !dbg !20
  %2037 = fptrunc <4 x float> %2036 to <4 x half>, !dbg !21
  %2038 = extractelement <4 x half> %2037, i64 3, !dbg !21
  %2039 = extractelement <4 x half> %2037, i64 2, !dbg !21
  %2040 = extractelement <4 x half> %2037, i64 1, !dbg !21
  %2041 = extractelement <4 x half> %2037, i64 0, !dbg !21
  %address.local1089 = or disjoint i64 %address.local918, 96, !dbg !22
  %address.local1090 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local916, i64 %address.local1089, !dbg !22
  store half %2041, ptr addrspace(1) %address.local1090, align 2, !dbg !22
  %address.local1092 = or disjoint i64 %address.local918, 97, !dbg !22
  %address.local1093 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local916, i64 %address.local1092, !dbg !22
  store half %2040, ptr addrspace(1) %address.local1093, align 2, !dbg !22
  %address.local1095 = or disjoint i64 %address.local918, 98, !dbg !22
  %address.local1096 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local916, i64 %address.local1095, !dbg !22
  store half %2039, ptr addrspace(1) %address.local1096, align 2, !dbg !22
  %address.local1098 = or disjoint i64 %address.local918, 99, !dbg !22
  %address.local1099 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local916, i64 %address.local1098, !dbg !22
  store half %2038, ptr addrspace(1) %address.local1099, align 2, !dbg !22
  %address.local1101 = or disjoint i64 %address.local918, 112, !dbg !22
  %address.local1102 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local916, i64 %address.local1101, !dbg !22
  store half %2031, ptr addrspace(1) %address.local1102, align 2, !dbg !22
  %address.local1104 = or disjoint i64 %address.local918, 113, !dbg !22
  %address.local1105 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local916, i64 %address.local1104, !dbg !22
  store half %2030, ptr addrspace(1) %address.local1105, align 2, !dbg !22
  %address.local1107 = or disjoint i64 %address.local918, 114, !dbg !22
  %address.local1108 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local916, i64 %address.local1107, !dbg !22
  store half %2029, ptr addrspace(1) %address.local1108, align 2, !dbg !22
  %address.local1110 = or i64 %address.local907, 115, !dbg !22
  %address.local1111 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local916, i64 %address.local1110, !dbg !22
  store half %2028, ptr addrspace(1) %address.local1111, align 2, !dbg !22
  %address.local1116 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local944, i64 %address.local1089, !dbg !22
  store half %2021, ptr addrspace(1) %address.local1116, align 2, !dbg !22
  %address.local1118 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local944, i64 %address.local1092, !dbg !22
  store half %2020, ptr addrspace(1) %address.local1118, align 2, !dbg !22
  %address.local1120 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local944, i64 %address.local1095, !dbg !22
  store half %2019, ptr addrspace(1) %address.local1120, align 2, !dbg !22
  %address.local1122 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local944, i64 %address.local1098, !dbg !22
  store half %2018, ptr addrspace(1) %address.local1122, align 2, !dbg !22
  %address.local1124 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local944, i64 %address.local1101, !dbg !22
  store half %2011, ptr addrspace(1) %address.local1124, align 2, !dbg !22
  %address.local1126 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local944, i64 %address.local1104, !dbg !22
  store half %2010, ptr addrspace(1) %address.local1126, align 2, !dbg !22
  %address.local1128 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local944, i64 %address.local1107, !dbg !22
  store half %2009, ptr addrspace(1) %address.local1128, align 2, !dbg !22
  %address.local1130 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local944, i64 %address.local1110, !dbg !22
  store half %2008, ptr addrspace(1) %address.local1130, align 2, !dbg !22
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
