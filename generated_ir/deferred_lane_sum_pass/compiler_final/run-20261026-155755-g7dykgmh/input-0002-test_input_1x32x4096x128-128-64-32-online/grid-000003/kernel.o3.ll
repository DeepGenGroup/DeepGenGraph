; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/deferred_lane_sum_pass/compiler_final/run-20261026-155755-g7dykgmh/input-0002-test_input_1x32x4096x128-128-64-32-online/grid-000003/kernel.ll'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/deferred_lane_sum_pass/compiler_final/run-20261026-155755-g7dykgmh/input-0002-test_input_1x32x4096x128-128-64-32-online/grid-000003/input.mlir"

@shm_0 = addrspace(3) global [16384 x i8] undef, align 16

; Function Attrs: convergent mustprogress nofree willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) nofree writeonly captures(none) %3) local_unnamed_addr #1 !dbg !3 !reqd_work_group_size !6 {
  %5 = tail call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !7
  %6 = sext i32 %5 to i64, !dbg !7
  %7 = tail call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !7
  %8 = sext i32 %7 to i64, !dbg !7
  %9 = shl nsw i64 %8, 5, !dbg !7
  %10 = add nsw i64 %9, %6, !dbg !7
  %11 = lshr i64 %10, 5, !dbg !7
  %12 = and i64 %11, 576460752303423484, !dbg !7
  %13 = and i64 %6, 3, !dbg !7
  %14 = or disjoint i64 %12, %13, !dbg !7
  %15 = sub nsw i64 31, %14, !dbg !7
  %16 = lshr i64 %10, 2, !dbg !7
  %17 = icmp ugt i64 %10, 1023, !dbg !8
  %18 = add nsw i64 %14, -32, !dbg !8
  %19 = select i1 %17, i64 %18, i64 %15, !dbg !8
  %20 = sdiv i64 %19, 32, !dbg !8
  %21 = sext i1 %17 to i64, !dbg !8
  %22 = xor i64 %20, %21, !dbg !8
  %23 = add nsw i64 %22, %16, !dbg !8
  %24 = shl i64 %15, 7, !dbg !8
  %25 = shl i64 %22, 12, !dbg !8
  %26 = sub i64 %24, %25, !dbg !8
  %27 = shl i64 %23, 19, !dbg !8
  %28 = and i64 %27, 16252928, !dbg !8
  %29 = tail call i32 @llvm.amdgcn.workitem.id.x(), !dbg !8, !range !9
  %.idx = shl i64 %26, 8, !dbg !8
  %address.thread = tail call i32 asm sideeffect "", "=v,0"(i32 %29) #8, !dbg !8, !range !9, !frisk.register_lifetimes !5
  %address.local = zext nneg i32 %address.thread to i64, !dbg !8
  %address.local262 = shl nuw nsw i64 %address.local, 3, !dbg !8
  %address.local263 = and i64 %address.local262, 1536, !dbg !8
  %address.local264 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local263, !dbg !8
  %address.local267 = shl nuw nsw i64 %address.local, 7, !dbg !8
  %address.local268 = and i64 %address.local267, 16256, !dbg !8
  %address.local269 = lshr i64 %address.local, 4, !dbg !8
  %address.local270 = and i64 %address.local269, 8, !dbg !8
  %address.local271 = or disjoint i64 %address.local268, %address.local270, !dbg !8
  %30 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %31 = or disjoint i64 %address.local271, %28, !dbg !8
  %32 = shl nuw nsw i64 %31, 1, !dbg !8
  %33 = add i64 %32, %.idx, !dbg !8
  %dma.buffer.offset1082 = trunc i64 %33 to i32, !dbg !8
  %34 = insertelement <2 x i64> poison, i64 %30, i64 0, !dbg !8
  %35 = bitcast <2 x i64> %34 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc1083 = shufflevector <4 x i32> %35, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
  %address.local273 = or disjoint i64 %address.local, 256, !dbg !8
  %address.local274 = shl nuw nsw i64 %address.local273, 3, !dbg !8
  %address.local275 = and i64 %address.local274, 3584, !dbg !8
  %address.local276 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local275, !dbg !8
  %address.local281 = lshr i64 %address.local273, 4, !dbg !8
  %address.local282 = and i64 %address.local281, 24, !dbg !8
  %address.local283 = or disjoint i64 %address.local282, %address.local268, !dbg !8
  %36 = or disjoint i64 %address.local283, %28, !dbg !8
  %37 = shl nuw nsw i64 %36, 1, !dbg !8
  %38 = add i64 %37, %.idx, !dbg !8
  %dma.buffer.offset1079 = trunc i64 %38 to i32, !dbg !8
  %39 = tail call i32 @llvm.amdgcn.workgroup.id.z(), !dbg !7
  %40 = sext i32 %39 to i64, !dbg !7
  %address.local285 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local263, !dbg !8
  %.idx3 = shl nuw nsw i64 %28, 1, !dbg !8
  %41 = add i64 %.idx3, %.idx, !dbg !8
  %address.local288.idx = shl nuw nsw i64 %address.local271, 1, !dbg !8
  %42 = or disjoint i64 %address.local288.idx, %41, !dbg !8
  %43 = trunc i64 %42 to i32, !dbg !8
  %dma.buffer.offset1076 = or disjoint i32 %43, 64, !dbg !8
  %address.local289 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local275, !dbg !8
  %44 = insertelement <4 x ptr addrspace(3)> poison, ptr addrspace(3) %address.local264, i64 0, !dbg !8
  %45 = insertelement <4 x ptr addrspace(3)> %44, ptr addrspace(3) %address.local276, i64 1, !dbg !8
  %46 = insertelement <4 x ptr addrspace(3)> %45, ptr addrspace(3) %address.local285, i64 2, !dbg !8
  %47 = insertelement <4 x ptr addrspace(3)> %46, ptr addrspace(3) %address.local289, i64 3, !dbg !8
  %48 = ptrtoint <4 x ptr addrspace(3)> %47 to <4 x i64>, !dbg !8
  %49 = trunc <4 x i64> %48 to <4 x i32>, !dbg !8
  %50 = extractelement <4 x i32> %49, i64 0, !dbg !8
  %51 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %50), !dbg !8
  %dma.buffer.lds1084 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %51), !dbg !8
  %52 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1084, i32 %dma.buffer.offset1082, <4 x i32> %dma.buffer.rsrc1083) #9, !dbg !8
  %53 = extractelement <4 x i32> %49, i64 1, !dbg !8
  %54 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %53), !dbg !8
  %dma.buffer.lds1081 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %54), !dbg !8
  %55 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1081, i32 %dma.buffer.offset1079, <4 x i32> %dma.buffer.rsrc1083) #9, !dbg !8
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %56 = extractelement <4 x i32> %49, i64 2, !dbg !8
  %57 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %56), !dbg !8
  %dma.buffer.lds1078 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %57), !dbg !8
  %58 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1078, i32 %dma.buffer.offset1076, <4 x i32> %dma.buffer.rsrc1083) #9, !dbg !8
  %59 = extractelement <4 x i32> %49, i64 3, !dbg !8
  %60 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %59), !dbg !8
  %address.local292.idx = shl nuw nsw i64 %address.local283, 1, !dbg !8
  %61 = or disjoint i64 %address.local292.idx, %41, !dbg !8
  %62 = trunc i64 %61 to i32, !dbg !8
  %dma.buffer.offset1073 = or disjoint i32 %62, 64, !dbg !8
  %dma.buffer.lds1075 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %60), !dbg !8
  %63 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1075, i32 %dma.buffer.offset1073, <4 x i32> %dma.buffer.rsrc1083) #9, !dbg !8
  %address.local293 = lshr i32 %address.thread, 2, !dbg !10
  %address.local294 = and i32 %address.local293, 4, !dbg !10
  %address.local295 = shl nuw nsw i32 %address.thread, 1, !dbg !10
  %address.local296 = and i32 %address.local295, 384, !dbg !10
  %address.local297 = or disjoint i32 %address.local294, %address.local296, !dbg !10
  %address.local298 = shl nuw nsw i32 %address.thread, 3, !dbg !10
  %address.local299 = and i32 %address.local298, 120, !dbg !10
  %address.local300 = or disjoint i32 %address.local297, %address.local299, !dbg !10
  %address.local301 = shl nuw nsw i32 %address.thread, 5, !dbg !10
  %address.local302 = and i32 %address.local301, 1024, !dbg !10
  %address.local303 = or disjoint i32 %address.local300, %address.local302, !dbg !10
  %address.local304 = zext nneg i32 %address.local303 to i64, !dbg !10
  %address.local305 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local304, !dbg !10
  %64 = load <4 x half>, ptr addrspace(3) %address.local305, align 8, !dbg !10
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local305, i64 4096, !dbg !10
  %65 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !10
  %66 = fmul <4 x half> %64, splat (half 1.275630e-01), !dbg !10
  %lds.const.gep971 = getelementptr i8, ptr addrspace(3) %address.local305, i64 1024, !dbg !10
  %67 = load <4 x half>, ptr addrspace(3) %lds.const.gep971, align 8, !dbg !10
  %68 = fmul <4 x half> %65, splat (half 1.275630e-01), !dbg !10
  %lds.const.gep972 = getelementptr i8, ptr addrspace(3) %address.local305, i64 5120, !dbg !10
  %69 = load <4 x half>, ptr addrspace(3) %lds.const.gep972, align 8, !dbg !10
  %70 = fmul <4 x half> %67, splat (half 1.275630e-01), !dbg !10
  %71 = fmul <4 x half> %69, splat (half 1.275630e-01), !dbg !10
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %dma.buffer.offset1070 = or disjoint i32 %43, 128, !dbg !8
  %72 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1084, i32 %dma.buffer.offset1070, <4 x i32> %dma.buffer.rsrc1083) #9, !dbg !8
  %dma.buffer.offset1067 = or disjoint i32 %62, 128, !dbg !8
  %73 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1081, i32 %dma.buffer.offset1067, <4 x i32> %dma.buffer.rsrc1083) #9, !dbg !8
  %lds.const.gep973 = getelementptr i8, ptr addrspace(3) %address.local305, i64 8192, !dbg !10
  %74 = load <4 x half>, ptr addrspace(3) %lds.const.gep973, align 8, !dbg !10
  %lds.const.gep974 = getelementptr i8, ptr addrspace(3) %address.local305, i64 12288, !dbg !10
  %75 = load <4 x half>, ptr addrspace(3) %lds.const.gep974, align 8, !dbg !10
  %lds.const.gep975 = getelementptr i8, ptr addrspace(3) %address.local305, i64 9216, !dbg !10
  %76 = load <4 x half>, ptr addrspace(3) %lds.const.gep975, align 8, !dbg !10
  %77 = fmul <4 x half> %74, splat (half 1.275630e-01), !dbg !10
  %lds.const.gep976 = getelementptr i8, ptr addrspace(3) %address.local305, i64 13312, !dbg !10
  %78 = load <4 x half>, ptr addrspace(3) %lds.const.gep976, align 8, !dbg !10
  %79 = fmul <4 x half> %75, splat (half 1.275630e-01), !dbg !10
  %80 = fmul <4 x half> %76, splat (half 1.275630e-01), !dbg !10
  %81 = fmul <4 x half> %78, splat (half 1.275630e-01), !dbg !10
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %dma.buffer.offset1064 = or disjoint i32 %43, 192, !dbg !8
  %82 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1078, i32 %dma.buffer.offset1064, <4 x i32> %dma.buffer.rsrc1083) #9, !dbg !8
  %dma.buffer.offset1061 = or disjoint i32 %62, 192, !dbg !8
  %83 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1075, i32 %dma.buffer.offset1061, <4 x i32> %dma.buffer.rsrc1083) #9, !dbg !8
  %84 = load <4 x half>, ptr addrspace(3) %address.local305, align 8, !dbg !10
  %85 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !10
  %86 = load <4 x half>, ptr addrspace(3) %lds.const.gep971, align 8, !dbg !10
  %87 = fmul <4 x half> %84, splat (half 1.275630e-01), !dbg !10
  %88 = load <4 x half>, ptr addrspace(3) %lds.const.gep972, align 8, !dbg !10
  %89 = fmul <4 x half> %85, splat (half 1.275630e-01), !dbg !10
  %90 = fmul <4 x half> %86, splat (half 1.275630e-01), !dbg !10
  %91 = fmul <4 x half> %88, splat (half 1.275630e-01), !dbg !10
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %92 = load <4 x half>, ptr addrspace(3) %lds.const.gep973, align 8, !dbg !10
  %93 = load <4 x half>, ptr addrspace(3) %lds.const.gep974, align 8, !dbg !10
  %94 = load <4 x half>, ptr addrspace(3) %lds.const.gep975, align 8, !dbg !10
  %95 = fmul <4 x half> %92, splat (half 1.275630e-01), !dbg !10
  %96 = load <4 x half>, ptr addrspace(3) %lds.const.gep976, align 8, !dbg !10
  %97 = fmul <4 x half> %93, splat (half 1.275630e-01), !dbg !10
  %98 = fmul <4 x half> %94, splat (half 1.275630e-01), !dbg !10
  %99 = fmul <4 x half> %96, splat (half 1.275630e-01), !dbg !10
  %100 = or disjoint i64 %24, 127, !dbg !11
  %101 = add i64 %24, 128, !dbg !11
  %102 = icmp uge i64 %101, %100, !dbg !11
  %103 = shl nsw i64 %15, 1, !dbg !12
  %104 = shl i64 %16, 7, !dbg !13
  %105 = and i64 %104, 3968, !dbg !13
  %106 = shl nsw i64 %40, 12, !dbg !13
  %107 = or disjoint i64 %105, %106, !dbg !13
  %108 = shl nuw nsw i32 %29, 2, !dbg !14
  %109 = and i32 %108, 12, !dbg !14
  %110 = and i32 %29, 1, !dbg !14
  %111 = or disjoint i32 %109, %110, !dbg !14
  %112 = trunc nuw nsw i32 %111 to i16, !dbg !14
  %.lhs.trunc = add nuw nsw i16 %112, 1, !dbg !14
  %113 = lshr i16 %.lhs.trunc, 1, !dbg !14
  %.sext = zext nneg i16 %113 to i64, !dbg !14
  %114 = icmp ult i64 %10, 1024, !dbg !12
  br i1 %114, label %.lr.ph, label %.._crit_edge_crit_edge, !dbg !12

.._crit_edge_crit_edge:                           ; preds = %4
  %.pre = tail call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !12
  %.pre63 = tail call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %.pre), !dbg !12
  %.pre64 = add nuw nsw i32 %.pre63, 64, !dbg !12
  %.pre65 = xor i32 %.pre63, 16, !dbg !12
  %.pre67 = and i32 %.pre64, 192, !dbg !12
  %.pre69 = xor i32 %.pre63, 32, !dbg !12
  br label %._crit_edge, !dbg !12

.lr.ph:                                           ; preds = %4
  %115 = zext nneg i32 %29 to i64, !dbg !8
  %116 = ptrtoint ptr addrspace(1) %1 to i64
  %117 = insertelement <2 x i64> poison, i64 %116, i64 0
  %118 = bitcast <2 x i64> %117 to <4 x i32>
  %dma.buffer.rsrc1059 = shufflevector <4 x i32> %118, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  %119 = shl nsw i64 %107, 1
  %invariant.op = or disjoint i64 %119, 64, !dbg !12
  %invariant.op52 = or disjoint i64 %119, 128, !dbg !12
  %invariant.op54 = or disjoint i64 %119, 192, !dbg !12
  %120 = ptrtoint ptr addrspace(1) %2 to i64
  %121 = insertelement <2 x i64> poison, i64 %120, i64 0
  %122 = bitcast <2 x i64> %121 to <4 x i32>
  %dma.buffer.rsrc1047 = shufflevector <4 x i32> %122, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  %123 = lshr i64 %115, 2
  %124 = and i64 %123, 48
  %125 = and i32 %29, 240
  %index.compact = zext nneg i32 %125 to i64
  %126 = sub nsw i64 %115, %index.compact
  %127 = add nsw i64 %126, %124
  %128 = add nsw i64 %127, %24
  %129 = and i64 %115, 48
  %130 = add i64 %128, 1
  %131 = lshr i64 %130, 32
  %compare.high = trunc nuw i64 %131 to i32
  %compare.low.base = trunc i64 %130 to i32
  %132 = add i64 %128, 65
  %133 = lshr i64 %132, 32
  %compare.high903 = trunc nuw i64 %133 to i32
  %compare.low.base907 = trunc i64 %132 to i32
  %134 = insertelement <4 x i32> poison, i32 %compare.low.base907, i64 0, !dbg !15
  %135 = shufflevector <4 x i32> %134, <4 x i32> poison, <4 x i32> zeroinitializer, !dbg !15
  %136 = insertelement <4 x i32> poison, i32 %compare.low.base, i64 0, !dbg !15
  %137 = shufflevector <4 x i32> %136, <4 x i32> poison, <4 x i32> zeroinitializer, !dbg !15
  br label %138, !dbg !12

138:                                              ; preds = %.lr.ph, %402
  %139 = phi [2 x <8 x float>] [ zeroinitializer, %.lr.ph ], [ %810, %402 ]
  %140 = phi [2 x <8 x float>] [ zeroinitializer, %.lr.ph ], [ %806, %402 ]
  %141 = phi [2 x <8 x float>] [ zeroinitializer, %.lr.ph ], [ %802, %402 ]
  %142 = phi [2 x <1 x float>] [ [<1 x float> splat (float -inf), <1 x float> splat (float -inf)], %.lr.ph ], [ %798, %402 ]
  %143 = phi [2 x <1 x float>] [ zeroinitializer, %.lr.ph ], [ %618, %402 ]
  %144 = phi [2 x <8 x float>] [ zeroinitializer, %.lr.ph ], [ %796, %402 ]
  %145 = phi i64 [ 0, %.lr.ph ], [ %811, %402 ]
  %146 = shl nuw nsw i64 %145, 6, !dbg !12
  %147 = lshr i64 %145, 13, !dbg !13
  %148 = add nuw nsw i64 %147, %16, !dbg !13
  %149 = lshr i64 %145, 6, !dbg !13
  %150 = shl i64 %148, 20, !dbg !13
  %151 = and i64 %150, 32505856, !dbg !13
  %152 = shl i64 %149, 19, !dbg !13
  %153 = shl nuw nsw i64 %145, 13, !dbg !13
  %154 = sub i64 %153, %152, !dbg !13
  %address.thread356 = tail call i32 asm sideeffect "", "=v,0"(i32 %29) #8, !dbg !13, !range !9, !frisk.register_lifetimes !5
  %address.local357 = zext nneg i32 %address.thread356 to i64, !dbg !8
  %address.local358 = shl nuw nsw i64 %address.local357, 3, !dbg !8
  %address.local359 = and i64 %address.local358, 1536, !dbg !13
  %address.local360 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local359, !dbg !13
  %address.local361 = ptrtoint ptr addrspace(3) %address.local360 to i64, !dbg !13
  %address.local362 = trunc i64 %address.local361 to i32, !dbg !13
  %155 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local362), !dbg !13
  %address.local363 = lshr i64 %address.local357, 3, !dbg !13
  %address.local364 = and i64 %address.local363, 24, !dbg !13
  %address.local365 = and i64 %address.local357, 59, !dbg !13
  %address.local366 = lshr i64 %address.local357, 2, !dbg !13
  %address.local367 = and i64 %address.local366, 4, !dbg !13
  %address.local368 = add nuw nsw i64 %address.local367, %address.local357, !dbg !13
  %address.local369 = and i64 %address.local368, 4, !dbg !13
  %address.local370 = or disjoint i64 %address.local369, %address.local365, !dbg !13
  %address.local371 = shl nuw nsw i64 %address.local370, 7, !dbg !13
  %address.local372 = or disjoint i64 %address.local371, %address.local364, !dbg !13
  %156 = shl nuw nsw i64 %149, 1, !dbg !13
  %.idx10 = and i64 %156, 254, !dbg !13
  %157 = or disjoint i64 %151, %.idx10, !dbg !13
  %158 = or disjoint i64 %address.local372, %154, !dbg !13
  %159 = shl i64 %158, 1, !dbg !13
  %160 = add i64 %157, %159, !dbg !13
  %dma.buffer.offset1058 = trunc i64 %160 to i32, !dbg !13
  %dma.buffer.lds1060 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %155), !dbg !13
  %161 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1060, i32 %dma.buffer.offset1058, <4 x i32> %dma.buffer.rsrc1059) #9, !dbg !13
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  tail call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %162 = add nuw nsw i64 %145, 2048, !dbg !13
  %163 = lshr i64 %162, 13, !dbg !13
  %164 = add nuw nsw i64 %163, %16, !dbg !13
  %165 = add nsw i64 %149, %107, !dbg !13
  %166 = shl i64 %164, 20, !dbg !13
  %167 = and i64 %166, 32505856, !dbg !13
  %address.local374 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local359, !dbg !13
  %address.local375 = ptrtoint ptr addrspace(3) %address.local374 to i64, !dbg !13
  %address.local376 = trunc i64 %address.local375 to i32, !dbg !13
  %168 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local376), !dbg !13
  %.idx14 = add i64 %159, %156, !dbg !13
  %.reass = add i64 %.idx14, %invariant.op
  %169 = add i64 %.reass, %167, !dbg !13
  %dma.buffer.lds1057 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %168), !dbg !13
  %address.local382 = shl nuw nsw i64 %address.local357, 1, !dbg !14
  %address.local383 = and i64 %address.local382, 24, !dbg !14
  %address.local385 = shl i64 %address.local357, 62, !dbg !14
  %address.local386 = ashr exact i64 %address.local385, 57, !dbg !14
  %address.local387 = and i64 %address.local386, -64, !dbg !14
  %address.local390 = and i64 %address.local366, 12, !dbg !10
  %170 = and i64 %address.local357, 32, !dbg !14
  %.not16 = icmp eq i64 %170, 0, !dbg !14
  %address.local394 = select i1 %.not16, i64 0, i64 504, !dbg !14
  %.idx17 = mul nuw nsw i64 %address.local357, 320, !dbg !14
  %171 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %.idx17, !dbg !14
  %.idx18 = mul nsw i64 %address.local366, -1280, !dbg !14
  %172 = getelementptr i8, ptr addrspace(3) %171, i64 %.idx18, !dbg !14
  %173 = getelementptr [2 x i8], ptr addrspace(3) %172, i64 %address.local383, !dbg !14
  %174 = getelementptr [2 x i8], ptr addrspace(3) %173, i64 %address.local387, !dbg !14
  %175 = getelementptr [2 x i8], ptr addrspace(3) %174, i64 %address.local390, !dbg !14
  %address.local396 = getelementptr [2 x i8], ptr addrspace(3) %175, i64 %address.local394, !dbg !14
  %lds.const.gep984 = getelementptr i8, ptr addrspace(3) %address.local396, i64 2048, !dbg !14
  %address.local407 = shl nuw nsw i32 %address.thread356, 1, !dbg !14
  %address.local408 = and i32 %address.local407, 24, !dbg !14
  %address.local409 = shl nuw nsw i32 %address.thread356, 8, !dbg !14
  %address.local410 = and i32 %address.local409, 768, !dbg !14
  %address.local411 = or disjoint i32 %address.local408, %address.local410, !dbg !14
  %address.local412 = shl nuw nsw i32 %address.thread356, 5, !dbg !14
  %address.local413 = and i32 %address.local412, 96, !dbg !14
  %address.local414 = or disjoint i32 %address.local411, %address.local413, !dbg !14
  %address.local415 = zext nneg i32 %address.local414 to i64, !dbg !14
  %address.local416 = lshr i32 %address.thread356, 1, !dbg !14
  %address.local417 = and i32 %address.local416, 1, !dbg !14
  %address.local418 = zext nneg i32 %address.local417 to i64, !dbg !14
  %lds.const.gep986 = getelementptr i8, ptr addrspace(3) %address.local396, i64 128, !dbg !14
  %lds.const.gep987 = getelementptr i8, ptr addrspace(3) %address.local396, i64 2176, !dbg !14
  %176 = add nuw nsw i64 %145, 4096, !dbg !13
  %177 = lshr i64 %176, 13, !dbg !13
  %178 = add nuw nsw i64 %177, %16, !dbg !13
  %179 = insertelement <2 x i64> poison, i64 %165, i64 0, !dbg !13
  %180 = shufflevector <2 x i64> %179, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !13
  %181 = icmp slt <2 x i64> %180, <i64 -32, i64 -64>, !dbg !13
  %182 = extractelement <2 x i1> %181, i64 0, !dbg !13
  %183 = sext i1 %182 to i64, !dbg !13
  %184 = add nsw <2 x i64> %180, <i64 32, i64 64>, !dbg !13
  %185 = sub nsw <2 x i64> <i64 -33, i64 -65>, %180, !dbg !13
  %186 = select <2 x i1> %181, <2 x i64> %185, <2 x i64> %184, !dbg !13
  %187 = sdiv <2 x i64> %186, splat (i64 128), !dbg !13
  %188 = extractelement <2 x i64> %187, i64 0, !dbg !13
  %189 = xor i64 %188, %183, !dbg !13
  %190 = shl i64 %189, 8, !dbg !13
  %191 = sub i64 %169, %190, !dbg !13
  %dma.buffer.offset1055 = trunc i64 %191 to i32, !dbg !13
  %192 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1057, i32 %dma.buffer.offset1055, <4 x i32> %dma.buffer.rsrc1059) #9, !dbg !13
  %193 = load <4 x half>, ptr addrspace(3) %address.local396, align 8, !dbg !14
  %194 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %193, <4 x float> zeroinitializer) #8, !dbg !14
  %195 = load <4 x half>, ptr addrspace(3) %lds.const.gep984, align 8, !dbg !14
  %address.base = tail call i64 asm sideeffect "", "=v,0"(i64 %.sext) #8, !dbg !14, !frisk.register_lifetimes !5
  %address.local419 = add nsw i64 %address.base, %address.local418, !dbg !14
  %address.local420 = shl i64 %address.local419, 6, !dbg !14
  %address.local421 = sub i64 %address.local415, %address.local420, !dbg !14
  %196 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local421, !dbg !14
  %197 = getelementptr [2 x i8], ptr addrspace(3) %196, i64 %address.local390, !dbg !14
  %198 = getelementptr [2 x i8], ptr addrspace(3) %197, i64 %address.local394, !dbg !14
  %address.local425 = getelementptr i8, ptr addrspace(3) %198, i64 64, !dbg !14
  %199 = load <4 x half>, ptr addrspace(3) %address.local425, align 8, !dbg !14
  %200 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %68, <4 x half> %195, <4 x float> %194) #8, !dbg !14
  %201 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %199, <4 x float> zeroinitializer) #8, !dbg !14
  %lds.const.gep985 = getelementptr i8, ptr addrspace(3) %198, i64 2112, !dbg !14
  %202 = load <4 x half>, ptr addrspace(3) %lds.const.gep985, align 8, !dbg !14
  %203 = load <4 x half>, ptr addrspace(3) %lds.const.gep986, align 8, !dbg !14
  %204 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %68, <4 x half> %202, <4 x float> %201) #8, !dbg !14
  %205 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %203, <4 x float> zeroinitializer) #8, !dbg !14
  %206 = load <4 x half>, ptr addrspace(3) %lds.const.gep987, align 8, !dbg !14
  %lds.const.gep988 = getelementptr i8, ptr addrspace(3) %198, i64 192, !dbg !14
  %207 = load <4 x half>, ptr addrspace(3) %lds.const.gep988, align 8, !dbg !14
  %208 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %68, <4 x half> %206, <4 x float> %205) #8, !dbg !14
  %209 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %207, <4 x float> zeroinitializer) #8, !dbg !14
  %lds.const.gep989 = getelementptr i8, ptr addrspace(3) %198, i64 2240, !dbg !14
  %210 = load <4 x half>, ptr addrspace(3) %lds.const.gep989, align 8, !dbg !14
  %211 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %68, <4 x half> %210, <4 x float> %209) #8, !dbg !14
  %212 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %193, <4 x float> zeroinitializer) #8, !dbg !14
  %213 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %195, <4 x float> %212) #8, !dbg !14
  %214 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %199, <4 x float> zeroinitializer) #8, !dbg !14
  %215 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %202, <4 x float> %214) #8, !dbg !14
  %216 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %203, <4 x float> zeroinitializer) #8, !dbg !14
  %217 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %206, <4 x float> %216) #8, !dbg !14
  %218 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %207, <4 x float> zeroinitializer) #8, !dbg !14
  %219 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %210, <4 x float> %218) #8, !dbg !14
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  tail call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %220 = extractelement <2 x i1> %181, i64 1, !dbg !13
  %221 = sext i1 %220 to i64, !dbg !13
  %222 = extractelement <2 x i64> %187, i64 1, !dbg !13
  %223 = xor i64 %222, %221, !dbg !13
  %224 = shl i64 %178, 20, !dbg !13
  %225 = and i64 %224, 32505856, !dbg !13
  %.reass53 = add i64 %.idx14, %invariant.op52
  %226 = add i64 %.reass53, %225, !dbg !13
  %227 = shl i64 %223, 8, !dbg !13
  %228 = sub i64 %226, %227, !dbg !13
  %dma.buffer.offset1052 = trunc i64 %228 to i32, !dbg !13
  %229 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1060, i32 %dma.buffer.offset1052, <4 x i32> %dma.buffer.rsrc1059) #9, !dbg !13
  %lds.const.gep996 = getelementptr i8, ptr addrspace(3) %address.local396, i64 4096, !dbg !14
  %230 = load <4 x half>, ptr addrspace(3) %lds.const.gep996, align 8, !dbg !14
  %231 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %230, <4 x float> %200) #8, !dbg !14
  %lds.const.gep997 = getelementptr i8, ptr addrspace(3) %address.local396, i64 6144, !dbg !14
  %232 = load <4 x half>, ptr addrspace(3) %lds.const.gep997, align 8, !dbg !14
  %lds.const.gep998 = getelementptr i8, ptr addrspace(3) %198, i64 4160, !dbg !14
  %233 = load <4 x half>, ptr addrspace(3) %lds.const.gep998, align 8, !dbg !14
  %234 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %232, <4 x float> %231) #8, !dbg !14
  %235 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %233, <4 x float> %204) #8, !dbg !14
  %lds.const.gep999 = getelementptr i8, ptr addrspace(3) %198, i64 6208, !dbg !14
  %236 = load <4 x half>, ptr addrspace(3) %lds.const.gep999, align 8, !dbg !14
  %lds.const.gep1000 = getelementptr i8, ptr addrspace(3) %address.local396, i64 4224, !dbg !14
  %237 = load <4 x half>, ptr addrspace(3) %lds.const.gep1000, align 8, !dbg !14
  %238 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %236, <4 x float> %235) #8, !dbg !14
  %239 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %237, <4 x float> %208) #8, !dbg !14
  %lds.const.gep1001 = getelementptr i8, ptr addrspace(3) %address.local396, i64 6272, !dbg !14
  %240 = load <4 x half>, ptr addrspace(3) %lds.const.gep1001, align 8, !dbg !14
  %lds.const.gep1002 = getelementptr i8, ptr addrspace(3) %198, i64 4288, !dbg !14
  %241 = load <4 x half>, ptr addrspace(3) %lds.const.gep1002, align 8, !dbg !14
  %242 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %240, <4 x float> %239) #8, !dbg !14
  %243 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %241, <4 x float> %211) #8, !dbg !14
  %lds.const.gep1003 = getelementptr i8, ptr addrspace(3) %198, i64 6336, !dbg !14
  %244 = load <4 x half>, ptr addrspace(3) %lds.const.gep1003, align 8, !dbg !14
  %245 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %244, <4 x float> %243) #8, !dbg !14
  %246 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %230, <4 x float> %213) #8, !dbg !14
  %247 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %232, <4 x float> %246) #8, !dbg !14
  %248 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %233, <4 x float> %215) #8, !dbg !14
  %249 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %236, <4 x float> %248) #8, !dbg !14
  %250 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %237, <4 x float> %217) #8, !dbg !14
  %251 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %240, <4 x float> %250) #8, !dbg !14
  %252 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %241, <4 x float> %219) #8, !dbg !14
  %253 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %244, <4 x float> %252) #8, !dbg !14
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  tail call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %254 = add nuw nsw i64 %145, 6144, !dbg !13
  %255 = lshr i64 %254, 13, !dbg !13
  %256 = add nuw nsw i64 %255, %16, !dbg !13
  %257 = add nsw i64 %165, 96, !dbg !13
  %258 = icmp slt i64 %165, -96, !dbg !13
  %259 = sub nsw i64 -97, %165, !dbg !13
  %260 = select i1 %258, i64 %259, i64 %257, !dbg !13
  %261 = sdiv i64 %260, 128, !dbg !13
  %262 = sext i1 %258 to i64, !dbg !13
  %263 = xor i64 %261, %262, !dbg !13
  %264 = shl i64 %256, 20, !dbg !13
  %265 = and i64 %264, 32505856, !dbg !13
  %.reass55 = add i64 %.idx14, %invariant.op54
  %266 = add i64 %.reass55, %265, !dbg !13
  %267 = shl i64 %263, 8, !dbg !13
  %268 = sub i64 %266, %267, !dbg !13
  %dma.buffer.offset1049 = trunc i64 %268 to i32, !dbg !13
  %269 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1057, i32 %dma.buffer.offset1049, <4 x i32> %dma.buffer.rsrc1059) #9, !dbg !13
  %270 = load <4 x half>, ptr addrspace(3) %address.local396, align 8, !dbg !14
  %271 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %270, <4 x float> %234) #8, !dbg !14
  %272 = load <4 x half>, ptr addrspace(3) %lds.const.gep984, align 8, !dbg !14
  %273 = load <4 x half>, ptr addrspace(3) %address.local425, align 8, !dbg !14
  %274 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %272, <4 x float> %271) #8, !dbg !14
  %275 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %273, <4 x float> %238) #8, !dbg !14
  %276 = load <4 x half>, ptr addrspace(3) %lds.const.gep985, align 8, !dbg !14
  %277 = load <4 x half>, ptr addrspace(3) %lds.const.gep986, align 8, !dbg !14
  %278 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %276, <4 x float> %275) #8, !dbg !14
  %279 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %277, <4 x float> %242) #8, !dbg !14
  %280 = load <4 x half>, ptr addrspace(3) %lds.const.gep987, align 8, !dbg !14
  %281 = load <4 x half>, ptr addrspace(3) %lds.const.gep988, align 8, !dbg !14
  %282 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %280, <4 x float> %279) #8, !dbg !14
  %283 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %281, <4 x float> %245) #8, !dbg !14
  %284 = load <4 x half>, ptr addrspace(3) %lds.const.gep989, align 8, !dbg !14
  %285 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %284, <4 x float> %283) #8, !dbg !14
  %286 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %270, <4 x float> %247) #8, !dbg !14
  %287 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %272, <4 x float> %286) #8, !dbg !14
  %288 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %273, <4 x float> %249) #8, !dbg !14
  %289 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %276, <4 x float> %288) #8, !dbg !14
  %290 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %277, <4 x float> %251) #8, !dbg !14
  %291 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %280, <4 x float> %290) #8, !dbg !14
  %292 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %281, <4 x float> %253) #8, !dbg !14
  %293 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %284, <4 x float> %292) #8, !dbg !14
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  tail call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %294 = add nuw nsw i64 %149, %16, !dbg !16
  %295 = shl i64 %294, 19, !dbg !16
  %296 = and i64 %295, 16252928, !dbg !16
  %address.local576 = shl nuw nsw i64 %address.local357, 6, !dbg !16
  %address.local577 = and i64 %address.local576, 16128, !dbg !16
  %.idx2730 = add i64 %296, %154, !dbg !16
  %297 = shl i64 %.idx2730, 1, !dbg !16
  %298 = or disjoint i64 %address.local577, %297, !dbg !16
  %address.local579 = shl nuw nsw i64 %address.local357, 4, !dbg !16
  %address.local581.idx = and i64 %address.local579, 48, !dbg !16
  %299 = or disjoint i64 %298, %address.local581.idx, !dbg !16
  %dma.buffer.offset1046 = trunc i64 %299 to i32, !dbg !16
  %300 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1060, i32 %dma.buffer.offset1046, <4 x i32> %dma.buffer.rsrc1047) #9, !dbg !16
  %301 = load <4 x half>, ptr addrspace(3) %lds.const.gep996, align 8, !dbg !14
  %302 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %301, <4 x float> %274) #8, !dbg !14
  %303 = load <4 x half>, ptr addrspace(3) %lds.const.gep997, align 8, !dbg !14
  %304 = load <4 x half>, ptr addrspace(3) %lds.const.gep998, align 8, !dbg !14
  %305 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %303, <4 x float> %302) #8, !dbg !14
  %306 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %304, <4 x float> %278) #8, !dbg !14
  %307 = load <4 x half>, ptr addrspace(3) %lds.const.gep999, align 8, !dbg !14
  %308 = load <4 x half>, ptr addrspace(3) %lds.const.gep1000, align 8, !dbg !14
  %309 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %307, <4 x float> %306) #8, !dbg !14
  %310 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %308, <4 x float> %282) #8, !dbg !14
  %311 = load <4 x half>, ptr addrspace(3) %lds.const.gep1001, align 8, !dbg !14
  %312 = load <4 x half>, ptr addrspace(3) %lds.const.gep1002, align 8, !dbg !14
  %313 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %311, <4 x float> %310) #8, !dbg !14
  %314 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %312, <4 x float> %285) #8, !dbg !14
  %315 = load <4 x half>, ptr addrspace(3) %lds.const.gep1003, align 8, !dbg !14
  %316 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %315, <4 x float> %314) #8, !dbg !14
  %317 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %301, <4 x float> %287) #8, !dbg !14
  %318 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %303, <4 x float> %317) #8, !dbg !14
  %319 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %304, <4 x float> %289) #8, !dbg !14
  %320 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %307, <4 x float> %319) #8, !dbg !14
  %321 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %308, <4 x float> %291) #8, !dbg !14
  %322 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %311, <4 x float> %321) #8, !dbg !14
  %323 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %312, <4 x float> %293) #8, !dbg !14
  %324 = or disjoint i64 %146, 63, !dbg !11
  %325 = icmp uge i64 %24, %324, !dbg !11
  %326 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %315, <4 x float> %323) #8, !dbg !14
  %327 = and i1 %102, %325, !dbg !11
  br i1 %327, label %402, label %328, !dbg !11

328:                                              ; preds = %138
  %329 = or disjoint i64 %146, %129, !dbg !11
  %330 = lshr i64 %145, 26, !dbg !15
  %compare.high843 = trunc nuw i64 %330 to i32, !dbg !15
  %compare.low.base844 = trunc i64 %329 to i32, !dbg !15
  %compare.local.base = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base844, i32 1) #8, !dbg !15, !frisk.register_lifetimes !5
  %compare.local.base847 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base844, i32 2) #8, !dbg !15, !frisk.register_lifetimes !5
  %compare.local.base851 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base844, i32 3) #8, !dbg !15, !frisk.register_lifetimes !5
  %331 = insertelement <4 x i32> poison, i32 %compare.low.base844, i64 0, !dbg !15
  %332 = insertelement <4 x i32> %331, i32 %compare.local.base, i64 1, !dbg !15
  %333 = insertelement <4 x i32> %332, i32 %compare.local.base847, i64 2, !dbg !15
  %334 = insertelement <4 x i32> %333, i32 %compare.local.base851, i64 3, !dbg !15
  %335 = or <4 x i32> %334, <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %336 = icmp ugt <4 x i32> %137, %335, !dbg !15
  %compare.local.base855 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base844, i32 4) #8, !dbg !15, !frisk.register_lifetimes !5
  %compare.local.base859 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base844, i32 5) #8, !dbg !15, !frisk.register_lifetimes !5
  %compare.local.base863 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base844, i32 6) #8, !dbg !15, !frisk.register_lifetimes !5
  %compare.local.base867 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base844, i32 7) #8, !dbg !15, !frisk.register_lifetimes !5
  %compare.local.base871 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base844, i32 8) #8, !dbg !15, !frisk.register_lifetimes !5
  %compare.local.base875 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base844, i32 9) #8, !dbg !15, !frisk.register_lifetimes !5
  %compare.local.base879 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base844, i32 10) #8, !dbg !15, !frisk.register_lifetimes !5
  %compare.local.base883 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base844, i32 11) #8, !dbg !15, !frisk.register_lifetimes !5
  %compare.local.base887 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base844, i32 12) #8, !dbg !15, !frisk.register_lifetimes !5
  %compare.local.base891 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base844, i32 13) #8, !dbg !15, !frisk.register_lifetimes !5
  %compare.local.base895 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base844, i32 14) #8, !dbg !15, !frisk.register_lifetimes !5
  %compare.local.base899 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base844, i32 15) #8, !dbg !15, !frisk.register_lifetimes !5
  %337 = insertelement <4 x i32> poison, i32 %compare.local.base871, i64 0, !dbg !15
  %338 = insertelement <4 x i32> %337, i32 %compare.local.base875, i64 1, !dbg !15
  %339 = insertelement <4 x i32> %338, i32 %compare.local.base879, i64 2, !dbg !15
  %340 = insertelement <4 x i32> %339, i32 %compare.local.base883, i64 3, !dbg !15
  %341 = or <4 x i32> %340, <i32 8, i32 9, i32 10, i32 11>, !dbg !15
  %342 = icmp ult <4 x i32> %341, %137, !dbg !15
  %343 = insertelement <4 x i32> poison, i32 %compare.local.base855, i64 0, !dbg !15
  %344 = insertelement <4 x i32> %343, i32 %compare.local.base859, i64 1, !dbg !15
  %345 = insertelement <4 x i32> %344, i32 %compare.local.base863, i64 2, !dbg !15
  %346 = insertelement <4 x i32> %345, i32 %compare.local.base867, i64 3, !dbg !15
  %347 = or <4 x i32> %346, <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %348 = icmp ult <4 x i32> %347, %137, !dbg !15
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high843, !dbg !15
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high843, !dbg !15
  %349 = insertelement <4 x i1> poison, i1 %compare.high.equal, i64 0, !dbg !15
  %350 = shufflevector <4 x i1> %349, <4 x i1> poison, <4 x i32> zeroinitializer, !dbg !15
  %351 = insertelement <4 x i1> poison, i1 %compare.high.order, i64 0, !dbg !15
  %352 = shufflevector <4 x i1> %351, <4 x i1> poison, <4 x i32> zeroinitializer, !dbg !15
  %353 = select <4 x i1> %350, <4 x i1> %336, <4 x i1> %352, !dbg !15
  %354 = select <4 x i1> %353, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !17
  %355 = select <4 x i1> %350, <4 x i1> %348, <4 x i1> %352, !dbg !15
  %356 = select <4 x i1> %355, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !17
  %357 = select <4 x i1> %350, <4 x i1> %342, <4 x i1> %352, !dbg !15
  %358 = select <4 x i1> %357, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !17
  %359 = insertelement <4 x i32> poison, i32 %compare.local.base887, i64 0, !dbg !15
  %360 = insertelement <4 x i32> %359, i32 %compare.local.base891, i64 1, !dbg !15
  %361 = insertelement <4 x i32> %360, i32 %compare.local.base895, i64 2, !dbg !15
  %362 = insertelement <4 x i32> %361, i32 %compare.local.base899, i64 3, !dbg !15
  %363 = or <4 x i32> %362, <i32 12, i32 13, i32 14, i32 15>, !dbg !15
  %364 = icmp ult <4 x i32> %363, %137, !dbg !15
  %365 = select <4 x i1> %350, <4 x i1> %364, <4 x i1> %352, !dbg !15
  %366 = select <4 x i1> %365, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !17
  %compare.local.base911 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base844, i32 16) #8, !dbg !15, !frisk.register_lifetimes !5
  %compare.local.base915 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base844, i32 17) #8, !dbg !15, !frisk.register_lifetimes !5
  %compare.local.base919 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base844, i32 18) #8, !dbg !15, !frisk.register_lifetimes !5
  %367 = insertelement <4 x i32> %331, i32 %compare.local.base911, i64 1, !dbg !15
  %368 = insertelement <4 x i32> %367, i32 %compare.local.base915, i64 2, !dbg !15
  %369 = insertelement <4 x i32> %368, i32 %compare.local.base919, i64 3, !dbg !15
  %370 = or <4 x i32> %369, <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %371 = icmp ugt <4 x i32> %135, %370, !dbg !15
  %compare.local.base923 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base844, i32 19) #8, !dbg !15, !frisk.register_lifetimes !5
  %compare.local.base927 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base844, i32 20) #8, !dbg !15, !frisk.register_lifetimes !5
  %compare.local.base931 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base844, i32 21) #8, !dbg !15, !frisk.register_lifetimes !5
  %compare.local.base935 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base844, i32 22) #8, !dbg !15, !frisk.register_lifetimes !5
  %compare.local.base939 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base844, i32 23) #8, !dbg !15, !frisk.register_lifetimes !5
  %compare.local.base943 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base844, i32 24) #8, !dbg !15, !frisk.register_lifetimes !5
  %compare.local.base947 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base844, i32 25) #8, !dbg !15, !frisk.register_lifetimes !5
  %compare.local.base951 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base844, i32 26) #8, !dbg !15, !frisk.register_lifetimes !5
  %compare.local.base955 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base844, i32 27) #8, !dbg !15, !frisk.register_lifetimes !5
  %compare.local.base959 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base844, i32 28) #8, !dbg !15, !frisk.register_lifetimes !5
  %compare.local.base963 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base844, i32 29) #8, !dbg !15, !frisk.register_lifetimes !5
  %compare.local.base967 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base844, i32 30) #8, !dbg !15, !frisk.register_lifetimes !5
  %372 = insertelement <4 x i32> poison, i32 %compare.local.base939, i64 0, !dbg !15
  %373 = insertelement <4 x i32> %372, i32 %compare.local.base943, i64 1, !dbg !15
  %374 = insertelement <4 x i32> %373, i32 %compare.local.base947, i64 2, !dbg !15
  %375 = insertelement <4 x i32> %374, i32 %compare.local.base951, i64 3, !dbg !15
  %376 = or <4 x i32> %375, <i32 8, i32 9, i32 10, i32 11>, !dbg !15
  %377 = icmp ult <4 x i32> %376, %135, !dbg !15
  %378 = insertelement <4 x i32> poison, i32 %compare.local.base923, i64 0, !dbg !15
  %379 = insertelement <4 x i32> %378, i32 %compare.local.base927, i64 1, !dbg !15
  %380 = insertelement <4 x i32> %379, i32 %compare.local.base931, i64 2, !dbg !15
  %381 = insertelement <4 x i32> %380, i32 %compare.local.base935, i64 3, !dbg !15
  %382 = or <4 x i32> %381, <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %383 = icmp ult <4 x i32> %382, %135, !dbg !15
  %compare.high.equal905 = icmp eq i32 %compare.high903, %compare.high843, !dbg !15
  %compare.high.order906 = icmp ugt i32 %compare.high903, %compare.high843, !dbg !15
  %384 = insertelement <4 x i1> poison, i1 %compare.high.equal905, i64 0, !dbg !15
  %385 = shufflevector <4 x i1> %384, <4 x i1> poison, <4 x i32> zeroinitializer, !dbg !15
  %386 = insertelement <4 x i1> poison, i1 %compare.high.order906, i64 0, !dbg !15
  %387 = shufflevector <4 x i1> %386, <4 x i1> poison, <4 x i32> zeroinitializer, !dbg !15
  %388 = select <4 x i1> %385, <4 x i1> %371, <4 x i1> %387, !dbg !15
  %389 = select <4 x i1> %388, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !17
  %390 = select <4 x i1> %385, <4 x i1> %383, <4 x i1> %387, !dbg !15
  %391 = select <4 x i1> %390, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !17
  %392 = select <4 x i1> %385, <4 x i1> %377, <4 x i1> %387, !dbg !15
  %393 = select <4 x i1> %392, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !17
  %394 = insertelement <4 x i32> poison, i32 %compare.local.base955, i64 0, !dbg !15
  %395 = insertelement <4 x i32> %394, i32 %compare.local.base959, i64 1, !dbg !15
  %396 = insertelement <4 x i32> %395, i32 %compare.local.base963, i64 2, !dbg !15
  %397 = insertelement <4 x i32> %396, i32 %compare.local.base967, i64 3, !dbg !15
  %398 = or <4 x i32> %397, <i32 12, i32 13, i32 14, i32 15>, !dbg !15
  %399 = icmp ult <4 x i32> %398, %135, !dbg !15
  %400 = select <4 x i1> %385, <4 x i1> %399, <4 x i1> %387, !dbg !15
  %401 = select <4 x i1> %400, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !17
  %.masked77 = fadd <4 x float> %305, %354, !dbg !18
  %.masked79 = fadd <4 x float> %309, %356, !dbg !18
  %.masked81 = fadd <4 x float> %313, %358, !dbg !18
  %.masked83 = fadd <4 x float> %316, %366, !dbg !18
  %.masked86 = fadd <4 x float> %318, %389, !dbg !18
  %.masked88 = fadd <4 x float> %320, %391, !dbg !18
  %.masked90 = fadd <4 x float> %322, %393, !dbg !18
  %.masked92 = fadd <4 x float> %326, %401, !dbg !18
  br label %402, !dbg !11

402:                                              ; preds = %328, %138
  %403 = phi <4 x float> [ %.masked77, %328 ], [ %305, %138 ], !dbg !18
  %404 = phi <4 x float> [ %.masked79, %328 ], [ %309, %138 ], !dbg !18
  %405 = phi <4 x float> [ %.masked81, %328 ], [ %313, %138 ], !dbg !18
  %406 = phi <4 x float> [ %.masked83, %328 ], [ %316, %138 ], !dbg !18
  %407 = phi <4 x float> [ %.masked86, %328 ], [ %318, %138 ], !dbg !18
  %408 = phi <4 x float> [ %.masked88, %328 ], [ %320, %138 ], !dbg !18
  %409 = phi <4 x float> [ %.masked90, %328 ], [ %322, %138 ], !dbg !18
  %410 = phi <4 x float> [ %.masked92, %328 ], [ %326, %138 ], !dbg !18
  %411 = extractelement <4 x float> %403, i64 0, !dbg !18
  %412 = extractelement <4 x float> %403, i64 1, !dbg !18
  %413 = extractelement <4 x float> %403, i64 2, !dbg !18
  %414 = extractelement <4 x float> %403, i64 3, !dbg !18
  %415 = extractelement <4 x float> %404, i64 0, !dbg !18
  %416 = extractelement <4 x float> %404, i64 1, !dbg !18
  %417 = extractelement <4 x float> %404, i64 2, !dbg !18
  %418 = extractelement <4 x float> %404, i64 3, !dbg !18
  %419 = extractelement <4 x float> %405, i64 0, !dbg !18
  %420 = extractelement <4 x float> %405, i64 1, !dbg !18
  %421 = extractelement <4 x float> %405, i64 2, !dbg !18
  %422 = extractelement <4 x float> %405, i64 3, !dbg !18
  %423 = extractelement <4 x float> %406, i64 0, !dbg !18
  %424 = extractelement <4 x float> %406, i64 1, !dbg !18
  %425 = extractelement <4 x float> %406, i64 2, !dbg !18
  %426 = extractelement <4 x float> %406, i64 3, !dbg !18
  %427 = extractelement <4 x float> %407, i64 0, !dbg !18
  %428 = extractelement <4 x float> %407, i64 1, !dbg !18
  %429 = extractelement <4 x float> %407, i64 2, !dbg !18
  %430 = extractelement <4 x float> %407, i64 3, !dbg !18
  %431 = extractelement <4 x float> %408, i64 0, !dbg !18
  %432 = extractelement <4 x float> %408, i64 1, !dbg !18
  %433 = extractelement <4 x float> %408, i64 2, !dbg !18
  %434 = extractelement <4 x float> %408, i64 3, !dbg !18
  %435 = extractelement <4 x float> %409, i64 0, !dbg !18
  %436 = extractelement <4 x float> %409, i64 1, !dbg !18
  %437 = extractelement <4 x float> %409, i64 2, !dbg !18
  %438 = extractelement <4 x float> %409, i64 3, !dbg !18
  %439 = extractelement <4 x float> %410, i64 0, !dbg !18
  %440 = extractelement <4 x float> %410, i64 1, !dbg !18
  %441 = extractelement <4 x float> %410, i64 2, !dbg !18
  %442 = extractelement <4 x float> %410, i64 3, !dbg !18
  %443 = tail call float @llvm.maxnum.f32(float %411, float -inf), !dbg !19, !frisk.local_reduction !5
  %local.reduce137 = tail call float @llvm.maxnum.f32(float %443, float %415), !dbg !19, !frisk.local_reduction !5
  %local.reduce138 = tail call float @llvm.maxnum.f32(float %412, float %416), !dbg !19, !frisk.local_reduction !5
  %local.reduce139 = tail call float @llvm.maxnum.f32(float %413, float %417), !dbg !19, !frisk.local_reduction !5
  %local.reduce140 = tail call float @llvm.maxnum.f32(float %414, float %418), !dbg !19, !frisk.local_reduction !5
  %local.reduce141 = tail call float @llvm.maxnum.f32(float %local.reduce137, float %419), !dbg !19, !frisk.local_reduction !5
  %local.reduce142 = tail call float @llvm.maxnum.f32(float %local.reduce138, float %420), !dbg !19, !frisk.local_reduction !5
  %local.reduce143 = tail call float @llvm.maxnum.f32(float %local.reduce139, float %421), !dbg !19, !frisk.local_reduction !5
  %local.reduce144 = tail call float @llvm.maxnum.f32(float %local.reduce140, float %422), !dbg !19, !frisk.local_reduction !5
  %local.reduce145 = tail call float @llvm.maxnum.f32(float %local.reduce141, float %423), !dbg !19, !frisk.local_reduction !5
  %local.reduce146 = tail call float @llvm.maxnum.f32(float %local.reduce142, float %424), !dbg !19, !frisk.local_reduction !5
  %local.reduce147 = tail call float @llvm.maxnum.f32(float %local.reduce143, float %425), !dbg !19, !frisk.local_reduction !5
  %local.reduce148 = tail call float @llvm.maxnum.f32(float %local.reduce144, float %426), !dbg !19, !frisk.local_reduction !5
  %local.reduce149 = tail call float @llvm.maxnum.f32(float %local.reduce145, float %local.reduce146), !dbg !19, !frisk.local_reduction !5
  %local.reduce150 = tail call float @llvm.maxnum.f32(float %local.reduce147, float %local.reduce148), !dbg !19, !frisk.local_reduction !5
  %444 = tail call float @llvm.maxnum.f32(float %local.reduce149, float %local.reduce150), !dbg !19, !frisk.local_reduction !5
  %445 = tail call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !19
  %446 = tail call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %445), !dbg !19
  %447 = add nuw nsw i32 %446, 64, !dbg !19
  %448 = xor i32 %446, 16, !dbg !19
  %449 = and i32 %447, 192, !dbg !19
  %450 = icmp samesign ult i32 %448, %449, !dbg !19
  %451 = select i1 %450, i32 %448, i32 %446, !dbg !19
  %452 = shl nuw nsw i32 %451, 2, !dbg !19
  %453 = bitcast float %444 to i32, !dbg !19
  %454 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %452, i32 %453), !dbg !19
  %455 = bitcast i32 %454 to float, !dbg !19
  %456 = tail call float @llvm.maxnum.f32(float %444, float %455), !dbg !19
  %457 = xor i32 %446, 32, !dbg !19
  %458 = icmp samesign ult i32 %457, %449, !dbg !19
  %459 = select i1 %458, i32 %457, i32 %446, !dbg !19
  %460 = shl nuw nsw i32 %459, 2, !dbg !19
  %461 = bitcast float %456 to i32, !dbg !19
  %462 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %460, i32 %461), !dbg !19
  %463 = bitcast i32 %462 to float, !dbg !19
  %464 = tail call float @llvm.maxnum.f32(float %456, float %463), !dbg !19
  %465 = extractvalue [2 x <1 x float>] %142, 0, !dbg !19
  %466 = insertelement <1 x float> poison, float %464, i64 0, !dbg !19
  %467 = tail call <1 x float> @llvm.maxnum.v1f32(<1 x float> %466, <1 x float> %465), !dbg !19
  %468 = tail call float @llvm.maxnum.f32(float %427, float -inf), !dbg !19, !frisk.local_reduction !5
  %local.reduce122 = tail call float @llvm.maxnum.f32(float %468, float %431), !dbg !19, !frisk.local_reduction !5
  %local.reduce123 = tail call float @llvm.maxnum.f32(float %428, float %432), !dbg !19, !frisk.local_reduction !5
  %local.reduce124 = tail call float @llvm.maxnum.f32(float %429, float %433), !dbg !19, !frisk.local_reduction !5
  %local.reduce125 = tail call float @llvm.maxnum.f32(float %430, float %434), !dbg !19, !frisk.local_reduction !5
  %local.reduce126 = tail call float @llvm.maxnum.f32(float %local.reduce122, float %435), !dbg !19, !frisk.local_reduction !5
  %local.reduce127 = tail call float @llvm.maxnum.f32(float %local.reduce123, float %436), !dbg !19, !frisk.local_reduction !5
  %local.reduce128 = tail call float @llvm.maxnum.f32(float %local.reduce124, float %437), !dbg !19, !frisk.local_reduction !5
  %local.reduce129 = tail call float @llvm.maxnum.f32(float %local.reduce125, float %438), !dbg !19, !frisk.local_reduction !5
  %local.reduce130 = tail call float @llvm.maxnum.f32(float %local.reduce126, float %439), !dbg !19, !frisk.local_reduction !5
  %local.reduce131 = tail call float @llvm.maxnum.f32(float %local.reduce127, float %440), !dbg !19, !frisk.local_reduction !5
  %local.reduce132 = tail call float @llvm.maxnum.f32(float %local.reduce128, float %441), !dbg !19, !frisk.local_reduction !5
  %local.reduce133 = tail call float @llvm.maxnum.f32(float %local.reduce129, float %442), !dbg !19, !frisk.local_reduction !5
  %local.reduce134 = tail call float @llvm.maxnum.f32(float %local.reduce130, float %local.reduce131), !dbg !19, !frisk.local_reduction !5
  %local.reduce135 = tail call float @llvm.maxnum.f32(float %local.reduce132, float %local.reduce133), !dbg !19, !frisk.local_reduction !5
  %469 = tail call float @llvm.maxnum.f32(float %local.reduce134, float %local.reduce135), !dbg !19, !frisk.local_reduction !5
  %470 = bitcast float %469 to i32, !dbg !19
  %471 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %452, i32 %470), !dbg !19
  %472 = bitcast i32 %471 to float, !dbg !19
  %473 = tail call float @llvm.maxnum.f32(float %469, float %472), !dbg !19
  %474 = bitcast float %473 to i32, !dbg !19
  %475 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %460, i32 %474), !dbg !19
  %476 = bitcast i32 %475 to float, !dbg !19
  %477 = tail call float @llvm.maxnum.f32(float %473, float %476), !dbg !19
  %478 = extractvalue [2 x <1 x float>] %142, 1, !dbg !19
  %479 = insertelement <1 x float> poison, float %477, i64 0, !dbg !19
  %480 = tail call <1 x float> @llvm.maxnum.v1f32(<1 x float> %479, <1 x float> %478), !dbg !19
  %481 = shufflevector <1 x float> %467, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !20
  %482 = fsub <4 x float> %403, %481, !dbg !20
  %483 = fsub <4 x float> %404, %481, !dbg !20
  %484 = fsub <4 x float> %405, %481, !dbg !20
  %485 = fsub <4 x float> %406, %481, !dbg !20
  %486 = shufflevector <1 x float> %480, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !20
  %487 = fsub <4 x float> %407, %486, !dbg !20
  %488 = fsub <4 x float> %408, %486, !dbg !20
  %489 = fsub <4 x float> %409, %486, !dbg !20
  %490 = fsub <4 x float> %410, %486, !dbg !20
  %491 = fsub <1 x float> %465, %467, !dbg !21
  %492 = extractelement <1 x float> %491, i64 0, !dbg !22
  %493 = tail call float @llvm.amdgcn.exp2.f32(float %492), !dbg !22
  %494 = insertelement <4 x float> poison, float %493, i64 0, !dbg !23
  %495 = shufflevector <4 x float> %494, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %496 = extractvalue [2 x <1 x float>] %143, 0, !dbg !24
  %497 = extractelement <1 x float> %496, i64 0, !dbg !24
  %498 = fmul float %497, %493, !dbg !24
  %499 = fsub <1 x float> %478, %480, !dbg !21
  %500 = extractelement <1 x float> %499, i64 0, !dbg !22
  %501 = tail call float @llvm.amdgcn.exp2.f32(float %500), !dbg !22
  %502 = insertelement <4 x float> poison, float %501, i64 0, !dbg !23
  %503 = shufflevector <4 x float> %502, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %504 = extractvalue [2 x <1 x float>] %143, 1, !dbg !24
  %505 = extractelement <1 x float> %504, i64 0, !dbg !24
  %506 = fmul float %505, %501, !dbg !24
  %507 = extractelement <4 x float> %482, i64 0, !dbg !25
  %508 = tail call float @llvm.amdgcn.exp2.f32(float %507), !dbg !25
  %509 = insertelement <4 x float> poison, float %508, i64 0, !dbg !26
  %510 = extractelement <4 x float> %482, i64 1, !dbg !25
  %511 = tail call float @llvm.amdgcn.exp2.f32(float %510), !dbg !25
  %512 = insertelement <4 x float> %509, float %511, i64 1, !dbg !26
  %513 = extractelement <4 x float> %482, i64 2, !dbg !25
  %514 = tail call float @llvm.amdgcn.exp2.f32(float %513), !dbg !25
  %515 = insertelement <4 x float> %512, float %514, i64 2, !dbg !26
  %516 = extractelement <4 x float> %482, i64 3, !dbg !25
  %517 = tail call float @llvm.amdgcn.exp2.f32(float %516), !dbg !25
  %518 = insertelement <4 x float> %515, float %517, i64 3, !dbg !26
  %math.result = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %518) #8, !dbg !26, !frisk.register_lifetimes !5
  %519 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !26
  %math.packed = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %519) #8, !dbg !25, !frisk.register_lifetimes !5
  %520 = extractelement <4 x float> %483, i64 0, !dbg !25
  %521 = tail call float @llvm.amdgcn.exp2.f32(float %520), !dbg !25
  %522 = insertelement <4 x float> poison, float %521, i64 0, !dbg !26
  %local.reduce107 = fadd float %508, %521, !dbg !27, !frisk.local_reduction !5
  %523 = extractelement <4 x float> %483, i64 1, !dbg !25
  %524 = tail call float @llvm.amdgcn.exp2.f32(float %523), !dbg !25
  %525 = insertelement <4 x float> %522, float %524, i64 1, !dbg !26
  %local.reduce108 = fadd float %511, %524, !dbg !27, !frisk.local_reduction !5
  %526 = extractelement <4 x float> %483, i64 2, !dbg !25
  %527 = tail call float @llvm.amdgcn.exp2.f32(float %526), !dbg !25
  %528 = insertelement <4 x float> %525, float %527, i64 2, !dbg !26
  %local.reduce109 = fadd float %514, %527, !dbg !27, !frisk.local_reduction !5
  %529 = extractelement <4 x float> %483, i64 3, !dbg !25
  %530 = tail call float @llvm.amdgcn.exp2.f32(float %529), !dbg !25
  %531 = insertelement <4 x float> %528, float %530, i64 3, !dbg !26
  %math.result248 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %531) #8, !dbg !26, !frisk.register_lifetimes !5
  %532 = fptrunc <4 x float> %math.result248 to <4 x half>, !dbg !26
  %math.packed249 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %532) #8, !dbg !27, !frisk.register_lifetimes !5
  %local.reduce110 = fadd float %517, %530, !dbg !27, !frisk.local_reduction !5
  %533 = extractelement <4 x float> %484, i64 0, !dbg !25
  %534 = tail call float @llvm.amdgcn.exp2.f32(float %533), !dbg !25
  %535 = insertelement <4 x float> poison, float %534, i64 0, !dbg !26
  %local.reduce111 = fadd float %534, %local.reduce107, !dbg !27, !frisk.local_reduction !5
  %536 = extractelement <4 x float> %484, i64 1, !dbg !25
  %537 = tail call float @llvm.amdgcn.exp2.f32(float %536), !dbg !25
  %538 = insertelement <4 x float> %535, float %537, i64 1, !dbg !26
  %local.reduce112 = fadd float %537, %local.reduce108, !dbg !27, !frisk.local_reduction !5
  %539 = extractelement <4 x float> %484, i64 2, !dbg !25
  %540 = tail call float @llvm.amdgcn.exp2.f32(float %539), !dbg !25
  %541 = insertelement <4 x float> %538, float %540, i64 2, !dbg !26
  %local.reduce113 = fadd float %540, %local.reduce109, !dbg !27, !frisk.local_reduction !5
  %542 = extractelement <4 x float> %484, i64 3, !dbg !25
  %543 = tail call float @llvm.amdgcn.exp2.f32(float %542), !dbg !25
  %544 = insertelement <4 x float> %541, float %543, i64 3, !dbg !26
  %math.result250 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %544) #8, !dbg !26, !frisk.register_lifetimes !5
  %545 = fptrunc <4 x float> %math.result250 to <4 x half>, !dbg !26
  %math.packed251 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %545) #8, !dbg !27, !frisk.register_lifetimes !5
  %local.reduce114 = fadd float %543, %local.reduce110, !dbg !27, !frisk.local_reduction !5
  %546 = extractelement <4 x float> %485, i64 0, !dbg !25
  %547 = tail call float @llvm.amdgcn.exp2.f32(float %546), !dbg !25
  %548 = insertelement <4 x float> poison, float %547, i64 0, !dbg !26
  %local.reduce115 = fadd float %547, %local.reduce111, !dbg !27, !frisk.local_reduction !5
  %549 = extractelement <4 x float> %485, i64 1, !dbg !25
  %550 = tail call float @llvm.amdgcn.exp2.f32(float %549), !dbg !25
  %551 = insertelement <4 x float> %548, float %550, i64 1, !dbg !26
  %local.reduce116 = fadd float %550, %local.reduce112, !dbg !27, !frisk.local_reduction !5
  %local.reduce119 = fadd float %local.reduce115, %local.reduce116, !dbg !27, !frisk.local_reduction !5
  %552 = extractelement <4 x float> %485, i64 2, !dbg !25
  %553 = tail call float @llvm.amdgcn.exp2.f32(float %552), !dbg !25
  %554 = insertelement <4 x float> %551, float %553, i64 2, !dbg !26
  %local.reduce117 = fadd float %553, %local.reduce113, !dbg !27, !frisk.local_reduction !5
  %555 = extractelement <4 x float> %485, i64 3, !dbg !25
  %556 = tail call float @llvm.amdgcn.exp2.f32(float %555), !dbg !25
  %557 = insertelement <4 x float> %554, float %556, i64 3, !dbg !26
  %math.result252 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %557) #8, !dbg !26, !frisk.register_lifetimes !5
  %558 = fptrunc <4 x float> %math.result252 to <4 x half>, !dbg !26
  %math.packed253 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %558) #8, !dbg !27, !frisk.register_lifetimes !5
  %local.reduce118 = fadd float %556, %local.reduce114, !dbg !27, !frisk.local_reduction !5
  %local.reduce120 = fadd float %local.reduce117, %local.reduce118, !dbg !27, !frisk.local_reduction !5
  %559 = fadd float %local.reduce119, %local.reduce120, !dbg !27, !frisk.local_reduction !5
  %560 = fadd float %498, %559, !dbg !27
  %561 = insertelement <1 x float> poison, float %560, i64 0, !dbg !12
  %562 = insertvalue [2 x <1 x float>] poison, <1 x float> %561, 0, !dbg !12
  %563 = extractelement <4 x float> %487, i64 0, !dbg !25
  %564 = tail call float @llvm.amdgcn.exp2.f32(float %563), !dbg !25
  %565 = insertelement <4 x float> poison, float %564, i64 0, !dbg !26
  %566 = extractelement <4 x float> %487, i64 1, !dbg !25
  %567 = tail call float @llvm.amdgcn.exp2.f32(float %566), !dbg !25
  %568 = insertelement <4 x float> %565, float %567, i64 1, !dbg !26
  %569 = extractelement <4 x float> %487, i64 2, !dbg !25
  %570 = tail call float @llvm.amdgcn.exp2.f32(float %569), !dbg !25
  %571 = insertelement <4 x float> %568, float %570, i64 2, !dbg !26
  %572 = extractelement <4 x float> %487, i64 3, !dbg !25
  %573 = tail call float @llvm.amdgcn.exp2.f32(float %572), !dbg !25
  %574 = insertelement <4 x float> %571, float %573, i64 3, !dbg !26
  %math.result254 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %574) #8, !dbg !26, !frisk.register_lifetimes !5
  %575 = fptrunc <4 x float> %math.result254 to <4 x half>, !dbg !26
  %math.packed255 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %575) #8, !dbg !25, !frisk.register_lifetimes !5
  %576 = extractelement <4 x float> %488, i64 0, !dbg !25
  %577 = tail call float @llvm.amdgcn.exp2.f32(float %576), !dbg !25
  %578 = insertelement <4 x float> poison, float %577, i64 0, !dbg !26
  %local.reduce = fadd float %564, %577, !dbg !27, !frisk.local_reduction !5
  %579 = extractelement <4 x float> %488, i64 1, !dbg !25
  %580 = tail call float @llvm.amdgcn.exp2.f32(float %579), !dbg !25
  %581 = insertelement <4 x float> %578, float %580, i64 1, !dbg !26
  %local.reduce93 = fadd float %567, %580, !dbg !27, !frisk.local_reduction !5
  %582 = extractelement <4 x float> %488, i64 2, !dbg !25
  %583 = tail call float @llvm.amdgcn.exp2.f32(float %582), !dbg !25
  %584 = insertelement <4 x float> %581, float %583, i64 2, !dbg !26
  %local.reduce94 = fadd float %570, %583, !dbg !27, !frisk.local_reduction !5
  %585 = extractelement <4 x float> %488, i64 3, !dbg !25
  %586 = tail call float @llvm.amdgcn.exp2.f32(float %585), !dbg !25
  %587 = insertelement <4 x float> %584, float %586, i64 3, !dbg !26
  %math.result256 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %587) #8, !dbg !26, !frisk.register_lifetimes !5
  %588 = fptrunc <4 x float> %math.result256 to <4 x half>, !dbg !26
  %math.packed257 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %588) #8, !dbg !27, !frisk.register_lifetimes !5
  %local.reduce95 = fadd float %573, %586, !dbg !27, !frisk.local_reduction !5
  %589 = extractelement <4 x float> %489, i64 0, !dbg !25
  %590 = tail call float @llvm.amdgcn.exp2.f32(float %589), !dbg !25
  %591 = insertelement <4 x float> poison, float %590, i64 0, !dbg !26
  %local.reduce96 = fadd float %590, %local.reduce, !dbg !27, !frisk.local_reduction !5
  %592 = extractelement <4 x float> %489, i64 1, !dbg !25
  %593 = tail call float @llvm.amdgcn.exp2.f32(float %592), !dbg !25
  %594 = insertelement <4 x float> %591, float %593, i64 1, !dbg !26
  %local.reduce97 = fadd float %593, %local.reduce93, !dbg !27, !frisk.local_reduction !5
  %595 = extractelement <4 x float> %489, i64 2, !dbg !25
  %596 = tail call float @llvm.amdgcn.exp2.f32(float %595), !dbg !25
  %597 = insertelement <4 x float> %594, float %596, i64 2, !dbg !26
  %local.reduce98 = fadd float %596, %local.reduce94, !dbg !27, !frisk.local_reduction !5
  %598 = extractelement <4 x float> %489, i64 3, !dbg !25
  %599 = tail call float @llvm.amdgcn.exp2.f32(float %598), !dbg !25
  %600 = insertelement <4 x float> %597, float %599, i64 3, !dbg !26
  %math.result258 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %600) #8, !dbg !26, !frisk.register_lifetimes !5
  %601 = fptrunc <4 x float> %math.result258 to <4 x half>, !dbg !26
  %math.packed259 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %601) #8, !dbg !27, !frisk.register_lifetimes !5
  %local.reduce99 = fadd float %599, %local.reduce95, !dbg !27, !frisk.local_reduction !5
  %602 = extractelement <4 x float> %490, i64 0, !dbg !25
  %603 = tail call float @llvm.amdgcn.exp2.f32(float %602), !dbg !25
  %604 = insertelement <4 x float> poison, float %603, i64 0, !dbg !26
  %local.reduce100 = fadd float %603, %local.reduce96, !dbg !27, !frisk.local_reduction !5
  %605 = extractelement <4 x float> %490, i64 1, !dbg !25
  %606 = tail call float @llvm.amdgcn.exp2.f32(float %605), !dbg !25
  %607 = insertelement <4 x float> %604, float %606, i64 1, !dbg !26
  %local.reduce101 = fadd float %606, %local.reduce97, !dbg !27, !frisk.local_reduction !5
  %local.reduce104 = fadd float %local.reduce100, %local.reduce101, !dbg !27, !frisk.local_reduction !5
  %608 = extractelement <4 x float> %490, i64 2, !dbg !25
  %609 = tail call float @llvm.amdgcn.exp2.f32(float %608), !dbg !25
  %610 = insertelement <4 x float> %607, float %609, i64 2, !dbg !26
  %local.reduce102 = fadd float %609, %local.reduce98, !dbg !27, !frisk.local_reduction !5
  %611 = extractelement <4 x float> %490, i64 3, !dbg !25
  %612 = tail call float @llvm.amdgcn.exp2.f32(float %611), !dbg !25
  %613 = insertelement <4 x float> %610, float %612, i64 3, !dbg !26
  %math.result260 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %613) #8, !dbg !26, !frisk.register_lifetimes !5
  %614 = fptrunc <4 x float> %math.result260 to <4 x half>, !dbg !26
  %math.packed261 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %614) #8, !dbg !27, !frisk.register_lifetimes !5
  %local.reduce103 = fadd float %612, %local.reduce99, !dbg !27, !frisk.local_reduction !5
  %local.reduce105 = fadd float %local.reduce102, %local.reduce103, !dbg !27, !frisk.local_reduction !5
  %615 = fadd float %local.reduce104, %local.reduce105, !dbg !27, !frisk.local_reduction !5
  %616 = fadd float %506, %615, !dbg !27
  %617 = insertelement <1 x float> poison, float %616, i64 0, !dbg !12
  %618 = insertvalue [2 x <1 x float>] %562, <1 x float> %617, 1, !dbg !12
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  tail call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %address.thread582 = tail call i32 asm sideeffect "", "=v,0"(i32 %29) #8, !dbg !28, !range !9, !frisk.register_lifetimes !5
  %address.local583 = zext nneg i32 %address.thread582 to i64, !dbg !8
  %address.local584 = and i64 %address.local583, 60, !dbg !28
  %address.local585 = add nuw nsw i64 %address.local584, %address.local583, !dbg !28
  %address.local587.idx = shl nuw nsw i64 %address.local585, 4, !dbg !28
  %address.local587 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %address.local587.idx, !dbg !28
  %address.local588 = lshr i64 %address.local583, 4, !dbg !28
  %address.local589 = and i64 %address.local588, 3, !dbg !28
  %address.local591.idx = mul nuw nsw i64 %address.local589, 768, !dbg !28
  %address.local591 = getelementptr i8, ptr addrspace(3) %address.local587, i64 %address.local591.idx, !dbg !28
  %address.local592 = lshr i64 %address.local583, 2, !dbg !14
  %address.local594.idx = mul nsw i64 %address.local592, -64, !dbg !28
  %address.local594 = getelementptr i8, ptr addrspace(3) %address.local591, i64 %address.local594.idx, !dbg !28
  %619 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local594, i32 0) #10, !dbg !28
  %620 = bitcast <4 x i32> %619 to <8 x half>, !dbg !28
  %address.local598 = getelementptr i8, ptr addrspace(3) %address.local594, i64 256, !dbg !28
  %621 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local598, i32 0) #10, !dbg !28
  %622 = bitcast <4 x i32> %621 to <8 x half>, !dbg !28
  %address.local602 = getelementptr i8, ptr addrspace(3) %address.local594, i64 512, !dbg !28
  %623 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local602, i32 0) #10, !dbg !28
  %624 = bitcast <4 x i32> %623 to <8 x half>, !dbg !28
  %address.local606 = getelementptr i8, ptr addrspace(3) %address.local594, i64 768, !dbg !28
  %625 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local606, i32 0) #10, !dbg !28
  %626 = bitcast <4 x i32> %625 to <8 x half>, !dbg !28
  fence syncscope("workgroup") release, !dbg !16
  tail call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %address.local607 = shl nuw nsw i64 %address.local583, 3, !dbg !8
  %address.local608 = and i64 %address.local607, 1536, !dbg !13
  %address.local609 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local608, !dbg !13
  %address.local610 = ptrtoint ptr addrspace(3) %address.local609 to i64, !dbg !13
  %address.local611 = trunc i64 %address.local610 to i32, !dbg !13
  %627 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local611), !dbg !16
  %address.local612 = shl nuw nsw i64 %address.local583, 5, !dbg !16
  %address.local613 = and i64 %address.local612, 8064, !dbg !16
  %address.local615 = and i64 %address.local607, 24, !dbg !16
  %address.local616 = or disjoint i64 %address.local613, %address.local615, !dbg !16
  %628 = or disjoint i64 %.idx2730, %address.local616, !dbg !16
  %629 = trunc i64 %628 to i32, !dbg !16
  %.tr = shl i32 %629, 1, !dbg !16
  %dma.buffer.offset1043 = or i32 %.tr, 64, !dbg !16
  %dma.buffer.lds1045 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %627), !dbg !16
  %630 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1045, i32 %dma.buffer.offset1043, <4 x i32> %dma.buffer.rsrc1047) #9, !dbg !16
  %631 = shufflevector <8 x half> %620, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %632 = extractvalue [2 x <8 x float>] %144, 0, !dbg !23
  %633 = shufflevector <8 x float> %632, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %634 = fmul <4 x float> %633, %495, !dbg !23
  %635 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %631, <4 x float> %634) #8, !dbg !28
  %636 = shufflevector <8 x half> %622, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %637 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %636, <4 x float> %635) #8, !dbg !28
  %638 = shufflevector <8 x half> %624, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %639 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %638, <4 x float> %637) #8, !dbg !28
  %640 = shufflevector <8 x half> %626, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %641 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %640, <4 x float> %639) #8, !dbg !28
  %642 = shufflevector <8 x half> %620, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %643 = shufflevector <8 x float> %632, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %644 = fmul <4 x float> %643, %495, !dbg !23
  %645 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %642, <4 x float> %644) #8, !dbg !28
  %646 = shufflevector <8 x half> %622, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %647 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %646, <4 x float> %645) #8, !dbg !28
  %648 = shufflevector <8 x half> %624, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %649 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %648, <4 x float> %647) #8, !dbg !28
  %650 = shufflevector <8 x half> %626, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %651 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %650, <4 x float> %649) #8, !dbg !28
  %652 = extractvalue [2 x <8 x float>] %144, 1, !dbg !23
  %653 = shufflevector <8 x float> %652, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %654 = fmul <4 x float> %653, %503, !dbg !23
  %655 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %631, <4 x float> %654) #8, !dbg !28
  %656 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %636, <4 x float> %655) #8, !dbg !28
  %657 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed259, <4 x half> %638, <4 x float> %656) #8, !dbg !28
  %658 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed261, <4 x half> %640, <4 x float> %657) #8, !dbg !28
  %659 = shufflevector <8 x float> %652, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %660 = fmul <4 x float> %659, %503, !dbg !23
  %661 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %642, <4 x float> %660) #8, !dbg !28
  %662 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %646, <4 x float> %661) #8, !dbg !28
  %663 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed259, <4 x half> %648, <4 x float> %662) #8, !dbg !28
  %664 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed261, <4 x half> %650, <4 x float> %663) #8, !dbg !28
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  tail call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %665 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local594, i32 0) #10, !dbg !28
  %666 = bitcast <4 x i32> %665 to <8 x half>, !dbg !28
  %667 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local598, i32 0) #10, !dbg !28
  %668 = bitcast <4 x i32> %667 to <8 x half>, !dbg !28
  %669 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local602, i32 0) #10, !dbg !28
  %670 = bitcast <4 x i32> %669 to <8 x half>, !dbg !28
  %671 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local606, i32 0) #10, !dbg !28
  %672 = bitcast <4 x i32> %671 to <8 x half>, !dbg !28
  fence syncscope("workgroup") release, !dbg !16
  tail call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %dma.buffer.offset1040 = or i32 %.tr, 128, !dbg !16
  %673 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1045, i32 %dma.buffer.offset1040, <4 x i32> %dma.buffer.rsrc1047) #9, !dbg !16
  %674 = shufflevector <8 x half> %666, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %675 = extractvalue [2 x <8 x float>] %141, 0, !dbg !23
  %676 = shufflevector <8 x float> %675, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %677 = fmul <4 x float> %676, %495, !dbg !23
  %678 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %674, <4 x float> %677) #8, !dbg !28
  %679 = shufflevector <8 x half> %668, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %680 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %679, <4 x float> %678) #8, !dbg !28
  %681 = shufflevector <8 x half> %670, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %682 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %681, <4 x float> %680) #8, !dbg !28
  %683 = shufflevector <8 x half> %672, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %684 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %683, <4 x float> %682) #8, !dbg !28
  %685 = shufflevector <8 x half> %666, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %686 = shufflevector <8 x float> %675, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %687 = fmul <4 x float> %686, %495, !dbg !23
  %688 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %685, <4 x float> %687) #8, !dbg !28
  %689 = shufflevector <8 x half> %668, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %690 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %689, <4 x float> %688) #8, !dbg !28
  %691 = shufflevector <8 x half> %670, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %692 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %691, <4 x float> %690) #8, !dbg !28
  %693 = shufflevector <8 x half> %672, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %694 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %693, <4 x float> %692) #8, !dbg !28
  %695 = extractvalue [2 x <8 x float>] %141, 1, !dbg !23
  %696 = shufflevector <8 x float> %695, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %697 = fmul <4 x float> %696, %503, !dbg !23
  %698 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %674, <4 x float> %697) #8, !dbg !28
  %699 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %679, <4 x float> %698) #8, !dbg !28
  %700 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed259, <4 x half> %681, <4 x float> %699) #8, !dbg !28
  %701 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed261, <4 x half> %683, <4 x float> %700) #8, !dbg !28
  %702 = shufflevector <8 x float> %695, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %703 = fmul <4 x float> %702, %503, !dbg !23
  %704 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %685, <4 x float> %703) #8, !dbg !28
  %705 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %689, <4 x float> %704) #8, !dbg !28
  %706 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed259, <4 x half> %691, <4 x float> %705) #8, !dbg !28
  %707 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed261, <4 x half> %693, <4 x float> %706) #8, !dbg !28
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  tail call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %708 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local594, i32 0) #10, !dbg !28
  %709 = bitcast <4 x i32> %708 to <8 x half>, !dbg !28
  %710 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local598, i32 0) #10, !dbg !28
  %711 = bitcast <4 x i32> %710 to <8 x half>, !dbg !28
  %712 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local602, i32 0) #10, !dbg !28
  %713 = bitcast <4 x i32> %712 to <8 x half>, !dbg !28
  %714 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local606, i32 0) #10, !dbg !28
  %715 = bitcast <4 x i32> %714 to <8 x half>, !dbg !28
  fence syncscope("workgroup") release, !dbg !16
  tail call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %dma.buffer.offset = or i32 %.tr, 192, !dbg !16
  %716 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1045, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc1047) #9, !dbg !16
  %717 = shufflevector <8 x half> %709, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %718 = extractvalue [2 x <8 x float>] %140, 0, !dbg !23
  %719 = shufflevector <8 x float> %718, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %720 = fmul <4 x float> %719, %495, !dbg !23
  %721 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %717, <4 x float> %720) #8, !dbg !28
  %722 = shufflevector <8 x half> %711, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %723 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %722, <4 x float> %721) #8, !dbg !28
  %724 = shufflevector <8 x half> %713, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %725 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %724, <4 x float> %723) #8, !dbg !28
  %726 = shufflevector <8 x half> %715, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %727 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %726, <4 x float> %725) #8, !dbg !28
  %728 = shufflevector <8 x half> %709, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %729 = shufflevector <8 x float> %718, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %730 = fmul <4 x float> %729, %495, !dbg !23
  %731 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %728, <4 x float> %730) #8, !dbg !28
  %732 = shufflevector <8 x half> %711, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %733 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %732, <4 x float> %731) #8, !dbg !28
  %734 = shufflevector <8 x half> %713, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %735 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %734, <4 x float> %733) #8, !dbg !28
  %736 = shufflevector <8 x half> %715, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %737 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %736, <4 x float> %735) #8, !dbg !28
  %738 = extractvalue [2 x <8 x float>] %140, 1, !dbg !23
  %739 = shufflevector <8 x float> %738, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %740 = fmul <4 x float> %739, %503, !dbg !23
  %741 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %717, <4 x float> %740) #8, !dbg !28
  %742 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %722, <4 x float> %741) #8, !dbg !28
  %743 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed259, <4 x half> %724, <4 x float> %742) #8, !dbg !28
  %744 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed261, <4 x half> %726, <4 x float> %743) #8, !dbg !28
  %745 = shufflevector <8 x float> %738, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %746 = fmul <4 x float> %745, %503, !dbg !23
  %747 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %728, <4 x float> %746) #8, !dbg !28
  %748 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %732, <4 x float> %747) #8, !dbg !28
  %749 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed259, <4 x half> %734, <4 x float> %748) #8, !dbg !28
  %750 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed261, <4 x half> %736, <4 x float> %749) #8, !dbg !28
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  tail call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %751 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local594, i32 0) #10, !dbg !28
  %752 = bitcast <4 x i32> %751 to <8 x half>, !dbg !28
  %753 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local598, i32 0) #10, !dbg !28
  %754 = bitcast <4 x i32> %753 to <8 x half>, !dbg !28
  %755 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local602, i32 0) #10, !dbg !28
  %756 = bitcast <4 x i32> %755 to <8 x half>, !dbg !28
  %757 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local606, i32 0) #10, !dbg !28
  %758 = bitcast <4 x i32> %757 to <8 x half>, !dbg !28
  %759 = shufflevector <8 x half> %752, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %760 = extractvalue [2 x <8 x float>] %139, 0, !dbg !23
  %761 = shufflevector <8 x float> %760, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %762 = fmul <4 x float> %761, %495, !dbg !23
  %763 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %759, <4 x float> %762) #8, !dbg !28
  %764 = shufflevector <8 x half> %754, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %765 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %764, <4 x float> %763) #8, !dbg !28
  %766 = shufflevector <8 x half> %756, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %767 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %766, <4 x float> %765) #8, !dbg !28
  %768 = shufflevector <8 x half> %758, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %769 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %768, <4 x float> %767) #8, !dbg !28
  %770 = shufflevector <8 x half> %752, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %771 = shufflevector <8 x float> %760, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %772 = fmul <4 x float> %771, %495, !dbg !23
  %773 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %770, <4 x float> %772) #8, !dbg !28
  %774 = shufflevector <8 x half> %754, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %775 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %774, <4 x float> %773) #8, !dbg !28
  %776 = shufflevector <8 x half> %756, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %777 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %776, <4 x float> %775) #8, !dbg !28
  %778 = shufflevector <8 x half> %758, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %779 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %778, <4 x float> %777) #8, !dbg !28
  %780 = extractvalue [2 x <8 x float>] %139, 1, !dbg !23
  %781 = shufflevector <8 x float> %780, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !23
  %782 = fmul <4 x float> %781, %503, !dbg !23
  %783 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %759, <4 x float> %782) #8, !dbg !28
  %784 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %764, <4 x float> %783) #8, !dbg !28
  %785 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed259, <4 x half> %766, <4 x float> %784) #8, !dbg !28
  %786 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed261, <4 x half> %768, <4 x float> %785) #8, !dbg !28
  %787 = shufflevector <8 x float> %780, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !23
  %788 = fmul <4 x float> %787, %503, !dbg !23
  %789 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %770, <4 x float> %788) #8, !dbg !28
  %790 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %774, <4 x float> %789) #8, !dbg !28
  %791 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed259, <4 x half> %776, <4 x float> %790) #8, !dbg !28
  %792 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed261, <4 x half> %778, <4 x float> %791) #8, !dbg !28
  %793 = shufflevector <4 x float> %641, <4 x float> %651, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %794 = insertvalue [2 x <8 x float>] poison, <8 x float> %793, 0, !dbg !12
  %795 = shufflevector <4 x float> %658, <4 x float> %664, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %796 = insertvalue [2 x <8 x float>] %794, <8 x float> %795, 1, !dbg !12
  %797 = insertvalue [2 x <1 x float>] poison, <1 x float> %467, 0, !dbg !12
  %798 = insertvalue [2 x <1 x float>] %797, <1 x float> %480, 1, !dbg !12
  %799 = shufflevector <4 x float> %684, <4 x float> %694, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %800 = insertvalue [2 x <8 x float>] poison, <8 x float> %799, 0, !dbg !12
  %801 = shufflevector <4 x float> %701, <4 x float> %707, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %802 = insertvalue [2 x <8 x float>] %800, <8 x float> %801, 1, !dbg !12
  %803 = shufflevector <4 x float> %727, <4 x float> %737, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %804 = insertvalue [2 x <8 x float>] poison, <8 x float> %803, 0, !dbg !12
  %805 = shufflevector <4 x float> %744, <4 x float> %750, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %806 = insertvalue [2 x <8 x float>] %804, <8 x float> %805, 1, !dbg !12
  %807 = shufflevector <4 x float> %769, <4 x float> %779, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %808 = insertvalue [2 x <8 x float>] poison, <8 x float> %807, 0, !dbg !12
  %809 = shufflevector <4 x float> %786, <4 x float> %792, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %810 = insertvalue [2 x <8 x float>] %808, <8 x float> %809, 1, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  tail call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %811 = add nuw nsw i64 %145, 1, !dbg !12
  %.not57 = icmp sgt i64 %145, %103, !dbg !12
  br i1 %.not57, label %._crit_edge, label %138, !dbg !12

._crit_edge:                                      ; preds = %402, %.._crit_edge_crit_edge
  %.pre-phi70 = phi i32 [ %.pre69, %.._crit_edge_crit_edge ], [ %457, %402 ], !dbg !12
  %.pre-phi68 = phi i32 [ %.pre67, %.._crit_edge_crit_edge ], [ %449, %402 ], !dbg !12
  %.pre-phi66 = phi i32 [ %.pre65, %.._crit_edge_crit_edge ], [ %448, %402 ], !dbg !12
  %lane.sum.lane157.pre-phi = phi i32 [ %.pre63, %.._crit_edge_crit_edge ], [ %446, %402 ], !dbg !12
  %.lcssa47 = phi [2 x <8 x float>] [ zeroinitializer, %.._crit_edge_crit_edge ], [ %796, %402 ], !dbg !12
  %.lcssa46 = phi [2 x <1 x float>] [ zeroinitializer, %.._crit_edge_crit_edge ], [ %618, %402 ], !dbg !12
  %.lcssa45 = phi [2 x <8 x float>] [ zeroinitializer, %.._crit_edge_crit_edge ], [ %802, %402 ], !dbg !12
  %.lcssa44 = phi [2 x <8 x float>] [ zeroinitializer, %.._crit_edge_crit_edge ], [ %806, %402 ], !dbg !12
  %.lcssa = phi [2 x <8 x float>] [ zeroinitializer, %.._crit_edge_crit_edge ], [ %810, %402 ], !dbg !12
  %lane.sum.field154 = extractvalue [2 x <1 x float>] %.lcssa46, 1, !dbg !12
  %lane.sum.local155 = extractelement <1 x float> %lane.sum.field154, i64 0, !dbg !12
  %812 = shl nuw nsw i32 %lane.sum.lane157.pre-phi, 2, !dbg !12
  %813 = xor i32 %812, 64, !dbg !12
  %814 = bitcast float %lane.sum.local155 to i32, !dbg !12
  %lane.sum.shuffle158 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %813, i32 %814), !dbg !12
  %815 = bitcast i32 %lane.sum.shuffle158 to float, !dbg !12
  %lane.sum.final159 = fadd float %lane.sum.local155, %815, !dbg !12
  %816 = xor i32 %812, 128, !dbg !12
  %817 = bitcast float %lane.sum.final159 to i32, !dbg !12
  %lane.sum.shuffle160 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %816, i32 %817), !dbg !12
  %818 = bitcast i32 %lane.sum.shuffle160 to float, !dbg !12
  %lane.sum.final161 = fadd float %lane.sum.final159, %818, !dbg !12
  %819 = insertelement <1 x float> poison, float %lane.sum.final161, i64 0, !dbg !12
  %lane.sum.field = extractvalue [2 x <1 x float>] %.lcssa46, 0, !dbg !12
  %lane.sum.local = extractelement <1 x float> %lane.sum.field, i64 0, !dbg !12
  %820 = bitcast float %lane.sum.local to i32, !dbg !12
  %lane.sum.shuffle = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %813, i32 %820), !dbg !12
  %821 = bitcast i32 %lane.sum.shuffle to float, !dbg !12
  %lane.sum.final = fadd float %lane.sum.local, %821, !dbg !12
  %822 = bitcast float %lane.sum.final to i32, !dbg !12
  %lane.sum.shuffle152 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %816, i32 %822), !dbg !12
  %823 = bitcast i32 %lane.sum.shuffle152 to float, !dbg !12
  %lane.sum.final153 = fadd float %lane.sum.final, %823, !dbg !12
  %824 = insertelement <1 x float> poison, float %lane.sum.final153, i64 0, !dbg !12
  %825 = and i32 %29, 16, !dbg !12
  %.not = icmp eq i32 %825, 0, !dbg !12
  %826 = and i32 %29, 32, !dbg !12
  %.not18 = icmp eq i32 %826, 0, !dbg !12
  %827 = extractvalue [2 x <8 x float>] %.lcssa, 0, !dbg !12
  %828 = extractelement <8 x float> %827, i64 0, !dbg !12
  %829 = extractelement <8 x float> %827, i64 1, !dbg !12
  %830 = extractelement <8 x float> %827, i64 2, !dbg !12
  %831 = extractelement <8 x float> %827, i64 3, !dbg !12
  %832 = select i1 %.not, float %829, float %828, !dbg !12
  %833 = icmp samesign ult i32 %.pre-phi66, %.pre-phi68, !dbg !12
  %834 = select i1 %833, i32 %.pre-phi66, i32 %lane.sum.lane157.pre-phi, !dbg !12
  %835 = shl nuw nsw i32 %834, 2, !dbg !12
  %836 = bitcast float %832 to i32, !dbg !12
  %837 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %835, i32 %836), !dbg !12
  %838 = bitcast i32 %837 to float, !dbg !12
  %839 = select i1 %.not, float %828, float %838, !dbg !12
  %840 = select i1 %.not, float %838, float %829, !dbg !12
  %841 = select i1 %.not, float %831, float %830, !dbg !12
  %842 = bitcast float %841 to i32, !dbg !12
  %843 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %835, i32 %842), !dbg !12
  %844 = bitcast i32 %843 to float, !dbg !12
  %845 = select i1 %.not, float %830, float %844, !dbg !12
  %846 = select i1 %.not, float %844, float %831, !dbg !12
  %847 = select i1 %.not18, float %845, float %839, !dbg !12
  %848 = icmp samesign ult i32 %.pre-phi70, %.pre-phi68, !dbg !12
  %849 = select i1 %848, i32 %.pre-phi70, i32 %lane.sum.lane157.pre-phi, !dbg !12
  %850 = shl nuw nsw i32 %849, 2, !dbg !12
  %851 = bitcast float %847 to i32, !dbg !12
  %852 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %850, i32 %851), !dbg !12
  %853 = bitcast i32 %852 to float, !dbg !12
  %854 = select i1 %.not18, float %846, float %840, !dbg !12
  %855 = bitcast float %854 to i32, !dbg !12
  %856 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %850, i32 %855), !dbg !12
  %857 = bitcast i32 %856 to float, !dbg !12
  %858 = insertelement <4 x i1> poison, i1 %.not18, i64 0, !dbg !12
  %859 = shufflevector <4 x i1> %858, <4 x i1> poison, <4 x i32> zeroinitializer, !dbg !12
  %860 = insertelement <4 x float> poison, float %839, i64 0, !dbg !12
  %861 = insertelement <4 x float> %860, float %840, i64 1, !dbg !12
  %862 = insertelement <4 x float> %861, float %853, i64 2, !dbg !12
  %863 = insertelement <4 x float> %862, float %857, i64 3, !dbg !12
  %864 = insertelement <4 x float> poison, float %853, i64 0, !dbg !12
  %865 = insertelement <4 x float> %864, float %857, i64 1, !dbg !12
  %866 = insertelement <4 x float> %865, float %845, i64 2, !dbg !12
  %867 = insertelement <4 x float> %866, float %846, i64 3, !dbg !12
  %868 = select <4 x i1> %859, <4 x float> %863, <4 x float> %867, !dbg !12
  %869 = extractvalue [2 x <8 x float>] %.lcssa44, 0, !dbg !12
  %870 = extractelement <8 x float> %869, i64 0, !dbg !12
  %871 = extractelement <8 x float> %869, i64 1, !dbg !12
  %872 = extractelement <8 x float> %869, i64 2, !dbg !12
  %873 = extractelement <8 x float> %869, i64 3, !dbg !12
  %874 = select i1 %.not, float %871, float %870, !dbg !12
  %875 = bitcast float %874 to i32, !dbg !12
  %876 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %835, i32 %875), !dbg !12
  %877 = bitcast i32 %876 to float, !dbg !12
  %878 = select i1 %.not, float %870, float %877, !dbg !12
  %879 = select i1 %.not, float %877, float %871, !dbg !12
  %880 = select i1 %.not, float %873, float %872, !dbg !12
  %881 = bitcast float %880 to i32, !dbg !12
  %882 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %835, i32 %881), !dbg !12
  %883 = bitcast i32 %882 to float, !dbg !12
  %884 = select i1 %.not, float %872, float %883, !dbg !12
  %885 = select i1 %.not, float %883, float %873, !dbg !12
  %886 = select i1 %.not18, float %884, float %878, !dbg !12
  %887 = bitcast float %886 to i32, !dbg !12
  %888 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %850, i32 %887), !dbg !12
  %889 = bitcast i32 %888 to float, !dbg !12
  %890 = select i1 %.not18, float %885, float %879, !dbg !12
  %891 = bitcast float %890 to i32, !dbg !12
  %892 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %850, i32 %891), !dbg !12
  %893 = bitcast i32 %892 to float, !dbg !12
  %894 = insertelement <4 x float> poison, float %878, i64 0, !dbg !12
  %895 = insertelement <4 x float> %894, float %879, i64 1, !dbg !12
  %896 = insertelement <4 x float> %895, float %889, i64 2, !dbg !12
  %897 = insertelement <4 x float> %896, float %893, i64 3, !dbg !12
  %898 = insertelement <4 x float> poison, float %889, i64 0, !dbg !12
  %899 = insertelement <4 x float> %898, float %893, i64 1, !dbg !12
  %900 = insertelement <4 x float> %899, float %884, i64 2, !dbg !12
  %901 = insertelement <4 x float> %900, float %885, i64 3, !dbg !12
  %902 = select <4 x i1> %859, <4 x float> %897, <4 x float> %901, !dbg !12
  %903 = extractvalue [2 x <8 x float>] %.lcssa45, 0, !dbg !12
  %904 = extractelement <8 x float> %903, i64 0, !dbg !12
  %905 = extractelement <8 x float> %903, i64 1, !dbg !12
  %906 = extractelement <8 x float> %903, i64 2, !dbg !12
  %907 = extractelement <8 x float> %903, i64 3, !dbg !12
  %908 = select i1 %.not, float %905, float %904, !dbg !12
  %909 = bitcast float %908 to i32, !dbg !12
  %910 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %835, i32 %909), !dbg !12
  %911 = bitcast i32 %910 to float, !dbg !12
  %912 = select i1 %.not, float %904, float %911, !dbg !12
  %913 = select i1 %.not, float %911, float %905, !dbg !12
  %914 = select i1 %.not, float %907, float %906, !dbg !12
  %915 = bitcast float %914 to i32, !dbg !12
  %916 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %835, i32 %915), !dbg !12
  %917 = bitcast i32 %916 to float, !dbg !12
  %918 = select i1 %.not, float %906, float %917, !dbg !12
  %919 = select i1 %.not, float %917, float %907, !dbg !12
  %920 = select i1 %.not18, float %918, float %912, !dbg !12
  %921 = bitcast float %920 to i32, !dbg !12
  %922 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %850, i32 %921), !dbg !12
  %923 = bitcast i32 %922 to float, !dbg !12
  %924 = select i1 %.not18, float %919, float %913, !dbg !12
  %925 = bitcast float %924 to i32, !dbg !12
  %926 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %850, i32 %925), !dbg !12
  %927 = bitcast i32 %926 to float, !dbg !12
  %928 = insertelement <4 x float> poison, float %912, i64 0, !dbg !12
  %929 = insertelement <4 x float> %928, float %913, i64 1, !dbg !12
  %930 = insertelement <4 x float> %929, float %923, i64 2, !dbg !12
  %931 = insertelement <4 x float> %930, float %927, i64 3, !dbg !12
  %932 = insertelement <4 x float> poison, float %923, i64 0, !dbg !12
  %933 = insertelement <4 x float> %932, float %927, i64 1, !dbg !12
  %934 = insertelement <4 x float> %933, float %918, i64 2, !dbg !12
  %935 = insertelement <4 x float> %934, float %919, i64 3, !dbg !12
  %936 = select <4 x i1> %859, <4 x float> %931, <4 x float> %935, !dbg !12
  %937 = extractvalue [2 x <8 x float>] %.lcssa47, 0, !dbg !12
  %938 = extractelement <8 x float> %937, i64 0, !dbg !12
  %939 = extractelement <8 x float> %937, i64 1, !dbg !12
  %940 = extractelement <8 x float> %937, i64 2, !dbg !12
  %941 = extractelement <8 x float> %937, i64 3, !dbg !12
  %942 = select i1 %.not, float %939, float %938, !dbg !12
  %943 = bitcast float %942 to i32, !dbg !12
  %944 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %835, i32 %943), !dbg !12
  %945 = bitcast i32 %944 to float, !dbg !12
  %946 = select i1 %.not, float %938, float %945, !dbg !12
  %947 = select i1 %.not, float %945, float %939, !dbg !12
  %948 = select i1 %.not, float %941, float %940, !dbg !12
  %949 = bitcast float %948 to i32, !dbg !12
  %950 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %835, i32 %949), !dbg !12
  %951 = bitcast i32 %950 to float, !dbg !12
  %952 = select i1 %.not, float %940, float %951, !dbg !12
  %953 = select i1 %.not, float %951, float %941, !dbg !12
  %954 = select i1 %.not18, float %952, float %946, !dbg !12
  %955 = bitcast float %954 to i32, !dbg !12
  %956 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %850, i32 %955), !dbg !12
  %957 = select i1 %.not18, float %953, float %947, !dbg !12
  %958 = bitcast float %957 to i32, !dbg !12
  %959 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %850, i32 %958), !dbg !12
  %960 = shufflevector <1 x float> %824, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !29
  %961 = extractelement <8 x float> %827, i64 4, !dbg !12
  %962 = extractelement <8 x float> %827, i64 5, !dbg !12
  %963 = extractelement <8 x float> %827, i64 6, !dbg !12
  %964 = extractelement <8 x float> %827, i64 7, !dbg !12
  %965 = select i1 %.not, float %962, float %961, !dbg !12
  %966 = bitcast float %965 to i32, !dbg !12
  %967 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %835, i32 %966), !dbg !12
  %968 = bitcast i32 %967 to float, !dbg !12
  %969 = select i1 %.not, float %961, float %968, !dbg !12
  %970 = select i1 %.not, float %968, float %962, !dbg !12
  %971 = select i1 %.not, float %964, float %963, !dbg !12
  %972 = bitcast float %971 to i32, !dbg !12
  %973 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %835, i32 %972), !dbg !12
  %974 = bitcast i32 %973 to float, !dbg !12
  %975 = select i1 %.not, float %963, float %974, !dbg !12
  %976 = select i1 %.not, float %974, float %964, !dbg !12
  %977 = select i1 %.not18, float %975, float %969, !dbg !12
  %978 = bitcast float %977 to i32, !dbg !12
  %979 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %850, i32 %978), !dbg !12
  %980 = bitcast i32 %979 to float, !dbg !12
  %981 = select i1 %.not18, float %976, float %970, !dbg !12
  %982 = bitcast float %981 to i32, !dbg !12
  %983 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %850, i32 %982), !dbg !12
  %984 = bitcast i32 %983 to float, !dbg !12
  %985 = insertelement <4 x float> poison, float %969, i64 0, !dbg !12
  %986 = insertelement <4 x float> %985, float %970, i64 1, !dbg !12
  %987 = insertelement <4 x float> %986, float %980, i64 2, !dbg !12
  %988 = insertelement <4 x float> %987, float %984, i64 3, !dbg !12
  %989 = insertelement <4 x float> poison, float %980, i64 0, !dbg !12
  %990 = insertelement <4 x float> %989, float %984, i64 1, !dbg !12
  %991 = insertelement <4 x float> %990, float %975, i64 2, !dbg !12
  %992 = insertelement <4 x float> %991, float %976, i64 3, !dbg !12
  %993 = select <4 x i1> %859, <4 x float> %988, <4 x float> %992, !dbg !12
  %994 = extractelement <8 x float> %869, i64 4, !dbg !12
  %995 = extractelement <8 x float> %869, i64 5, !dbg !12
  %996 = extractelement <8 x float> %869, i64 6, !dbg !12
  %997 = extractelement <8 x float> %869, i64 7, !dbg !12
  %998 = select i1 %.not, float %995, float %994, !dbg !12
  %999 = bitcast float %998 to i32, !dbg !12
  %1000 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %835, i32 %999), !dbg !12
  %1001 = bitcast i32 %1000 to float, !dbg !12
  %1002 = select i1 %.not, float %994, float %1001, !dbg !12
  %1003 = select i1 %.not, float %1001, float %995, !dbg !12
  %1004 = select i1 %.not, float %997, float %996, !dbg !12
  %1005 = bitcast float %1004 to i32, !dbg !12
  %1006 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %835, i32 %1005), !dbg !12
  %1007 = bitcast i32 %1006 to float, !dbg !12
  %1008 = select i1 %.not, float %996, float %1007, !dbg !12
  %1009 = select i1 %.not, float %1007, float %997, !dbg !12
  %1010 = select i1 %.not18, float %1008, float %1002, !dbg !12
  %1011 = bitcast float %1010 to i32, !dbg !12
  %1012 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %850, i32 %1011), !dbg !12
  %1013 = bitcast i32 %1012 to float, !dbg !12
  %1014 = select i1 %.not18, float %1009, float %1003, !dbg !12
  %1015 = bitcast float %1014 to i32, !dbg !12
  %1016 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %850, i32 %1015), !dbg !12
  %1017 = bitcast i32 %1016 to float, !dbg !12
  %1018 = insertelement <4 x float> poison, float %1002, i64 0, !dbg !12
  %1019 = insertelement <4 x float> %1018, float %1003, i64 1, !dbg !12
  %1020 = insertelement <4 x float> %1019, float %1013, i64 2, !dbg !12
  %1021 = insertelement <4 x float> %1020, float %1017, i64 3, !dbg !12
  %1022 = insertelement <4 x float> poison, float %1013, i64 0, !dbg !12
  %1023 = insertelement <4 x float> %1022, float %1017, i64 1, !dbg !12
  %1024 = insertelement <4 x float> %1023, float %1008, i64 2, !dbg !12
  %1025 = insertelement <4 x float> %1024, float %1009, i64 3, !dbg !12
  %1026 = select <4 x i1> %859, <4 x float> %1021, <4 x float> %1025, !dbg !12
  %1027 = extractelement <8 x float> %903, i64 4, !dbg !12
  %1028 = extractelement <8 x float> %903, i64 5, !dbg !12
  %1029 = extractelement <8 x float> %903, i64 6, !dbg !12
  %1030 = extractelement <8 x float> %903, i64 7, !dbg !12
  %1031 = select i1 %.not, float %1028, float %1027, !dbg !12
  %1032 = bitcast float %1031 to i32, !dbg !12
  %1033 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %835, i32 %1032), !dbg !12
  %1034 = bitcast i32 %1033 to float, !dbg !12
  %1035 = select i1 %.not, float %1027, float %1034, !dbg !12
  %1036 = select i1 %.not, float %1034, float %1028, !dbg !12
  %1037 = select i1 %.not, float %1030, float %1029, !dbg !12
  %1038 = bitcast float %1037 to i32, !dbg !12
  %1039 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %835, i32 %1038), !dbg !12
  %1040 = bitcast i32 %1039 to float, !dbg !12
  %1041 = select i1 %.not, float %1029, float %1040, !dbg !12
  %1042 = select i1 %.not, float %1040, float %1030, !dbg !12
  %1043 = select i1 %.not18, float %1041, float %1035, !dbg !12
  %1044 = bitcast float %1043 to i32, !dbg !12
  %1045 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %850, i32 %1044), !dbg !12
  %1046 = bitcast i32 %1045 to float, !dbg !12
  %1047 = select i1 %.not18, float %1042, float %1036, !dbg !12
  %1048 = bitcast float %1047 to i32, !dbg !12
  %1049 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %850, i32 %1048), !dbg !12
  %1050 = bitcast i32 %1049 to float, !dbg !12
  %1051 = insertelement <4 x float> poison, float %1035, i64 0, !dbg !12
  %1052 = insertelement <4 x float> %1051, float %1036, i64 1, !dbg !12
  %1053 = insertelement <4 x float> %1052, float %1046, i64 2, !dbg !12
  %1054 = insertelement <4 x float> %1053, float %1050, i64 3, !dbg !12
  %1055 = insertelement <4 x float> poison, float %1046, i64 0, !dbg !12
  %1056 = insertelement <4 x float> %1055, float %1050, i64 1, !dbg !12
  %1057 = insertelement <4 x float> %1056, float %1041, i64 2, !dbg !12
  %1058 = insertelement <4 x float> %1057, float %1042, i64 3, !dbg !12
  %1059 = select <4 x i1> %859, <4 x float> %1054, <4 x float> %1058, !dbg !12
  %1060 = extractelement <8 x float> %937, i64 4, !dbg !12
  %1061 = extractelement <8 x float> %937, i64 5, !dbg !12
  %1062 = extractelement <8 x float> %937, i64 6, !dbg !12
  %1063 = extractelement <8 x float> %937, i64 7, !dbg !12
  %1064 = select i1 %.not, float %1061, float %1060, !dbg !12
  %1065 = bitcast float %1064 to i32, !dbg !12
  %1066 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %835, i32 %1065), !dbg !12
  %1067 = bitcast i32 %1066 to float, !dbg !12
  %1068 = select i1 %.not, float %1060, float %1067, !dbg !12
  %1069 = select i1 %.not, float %1067, float %1061, !dbg !12
  %1070 = select i1 %.not, float %1063, float %1062, !dbg !12
  %1071 = bitcast float %1070 to i32, !dbg !12
  %1072 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %835, i32 %1071), !dbg !12
  %1073 = bitcast i32 %1072 to float, !dbg !12
  %1074 = select i1 %.not, float %1062, float %1073, !dbg !12
  %1075 = select i1 %.not, float %1073, float %1063, !dbg !12
  %1076 = select i1 %.not18, float %1074, float %1068, !dbg !12
  %1077 = bitcast float %1076 to i32, !dbg !12
  %1078 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %850, i32 %1077), !dbg !12
  %1079 = select i1 %.not18, float %1075, float %1069, !dbg !12
  %1080 = bitcast float %1079 to i32, !dbg !12
  %1081 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %850, i32 %1080), !dbg !12
  %1082 = extractvalue [2 x <8 x float>] %.lcssa, 1, !dbg !12
  %1083 = extractelement <8 x float> %1082, i64 0, !dbg !12
  %1084 = extractelement <8 x float> %1082, i64 1, !dbg !12
  %1085 = extractelement <8 x float> %1082, i64 2, !dbg !12
  %1086 = extractelement <8 x float> %1082, i64 3, !dbg !12
  %1087 = select i1 %.not, float %1084, float %1083, !dbg !12
  %1088 = bitcast float %1087 to i32, !dbg !12
  %1089 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %835, i32 %1088), !dbg !12
  %1090 = bitcast i32 %1089 to float, !dbg !12
  %1091 = select i1 %.not, float %1083, float %1090, !dbg !12
  %1092 = select i1 %.not, float %1090, float %1084, !dbg !12
  %1093 = select i1 %.not, float %1086, float %1085, !dbg !12
  %1094 = bitcast float %1093 to i32, !dbg !12
  %1095 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %835, i32 %1094), !dbg !12
  %1096 = bitcast i32 %1095 to float, !dbg !12
  %1097 = select i1 %.not, float %1085, float %1096, !dbg !12
  %1098 = select i1 %.not, float %1096, float %1086, !dbg !12
  %1099 = select i1 %.not18, float %1097, float %1091, !dbg !12
  %1100 = bitcast float %1099 to i32, !dbg !12
  %1101 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %850, i32 %1100), !dbg !12
  %1102 = bitcast i32 %1101 to float, !dbg !12
  %1103 = select i1 %.not18, float %1098, float %1092, !dbg !12
  %1104 = bitcast float %1103 to i32, !dbg !12
  %1105 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %850, i32 %1104), !dbg !12
  %1106 = bitcast i32 %1105 to float, !dbg !12
  %1107 = insertelement <4 x float> poison, float %1091, i64 0, !dbg !12
  %1108 = insertelement <4 x float> %1107, float %1092, i64 1, !dbg !12
  %1109 = insertelement <4 x float> %1108, float %1102, i64 2, !dbg !12
  %1110 = insertelement <4 x float> %1109, float %1106, i64 3, !dbg !12
  %1111 = insertelement <4 x float> poison, float %1102, i64 0, !dbg !12
  %1112 = insertelement <4 x float> %1111, float %1106, i64 1, !dbg !12
  %1113 = insertelement <4 x float> %1112, float %1097, i64 2, !dbg !12
  %1114 = insertelement <4 x float> %1113, float %1098, i64 3, !dbg !12
  %1115 = select <4 x i1> %859, <4 x float> %1110, <4 x float> %1114, !dbg !12
  %1116 = extractvalue [2 x <8 x float>] %.lcssa44, 1, !dbg !12
  %1117 = extractelement <8 x float> %1116, i64 0, !dbg !12
  %1118 = extractelement <8 x float> %1116, i64 1, !dbg !12
  %1119 = extractelement <8 x float> %1116, i64 2, !dbg !12
  %1120 = extractelement <8 x float> %1116, i64 3, !dbg !12
  %1121 = select i1 %.not, float %1118, float %1117, !dbg !12
  %1122 = bitcast float %1121 to i32, !dbg !12
  %1123 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %835, i32 %1122), !dbg !12
  %1124 = bitcast i32 %1123 to float, !dbg !12
  %1125 = select i1 %.not, float %1117, float %1124, !dbg !12
  %1126 = select i1 %.not, float %1124, float %1118, !dbg !12
  %1127 = select i1 %.not, float %1120, float %1119, !dbg !12
  %1128 = bitcast float %1127 to i32, !dbg !12
  %1129 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %835, i32 %1128), !dbg !12
  %1130 = bitcast i32 %1129 to float, !dbg !12
  %1131 = select i1 %.not, float %1119, float %1130, !dbg !12
  %1132 = select i1 %.not, float %1130, float %1120, !dbg !12
  %1133 = select i1 %.not18, float %1131, float %1125, !dbg !12
  %1134 = bitcast float %1133 to i32, !dbg !12
  %1135 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %850, i32 %1134), !dbg !12
  %1136 = bitcast i32 %1135 to float, !dbg !12
  %1137 = select i1 %.not18, float %1132, float %1126, !dbg !12
  %1138 = bitcast float %1137 to i32, !dbg !12
  %1139 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %850, i32 %1138), !dbg !12
  %1140 = bitcast i32 %1139 to float, !dbg !12
  %1141 = insertelement <4 x float> poison, float %1125, i64 0, !dbg !12
  %1142 = insertelement <4 x float> %1141, float %1126, i64 1, !dbg !12
  %1143 = insertelement <4 x float> %1142, float %1136, i64 2, !dbg !12
  %1144 = insertelement <4 x float> %1143, float %1140, i64 3, !dbg !12
  %1145 = insertelement <4 x float> poison, float %1136, i64 0, !dbg !12
  %1146 = insertelement <4 x float> %1145, float %1140, i64 1, !dbg !12
  %1147 = insertelement <4 x float> %1146, float %1131, i64 2, !dbg !12
  %1148 = insertelement <4 x float> %1147, float %1132, i64 3, !dbg !12
  %1149 = select <4 x i1> %859, <4 x float> %1144, <4 x float> %1148, !dbg !12
  %1150 = extractvalue [2 x <8 x float>] %.lcssa45, 1, !dbg !12
  %1151 = extractelement <8 x float> %1150, i64 0, !dbg !12
  %1152 = extractelement <8 x float> %1150, i64 1, !dbg !12
  %1153 = extractelement <8 x float> %1150, i64 2, !dbg !12
  %1154 = extractelement <8 x float> %1150, i64 3, !dbg !12
  %1155 = select i1 %.not, float %1152, float %1151, !dbg !12
  %1156 = bitcast float %1155 to i32, !dbg !12
  %1157 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %835, i32 %1156), !dbg !12
  %1158 = bitcast i32 %1157 to float, !dbg !12
  %1159 = select i1 %.not, float %1151, float %1158, !dbg !12
  %1160 = select i1 %.not, float %1158, float %1152, !dbg !12
  %1161 = select i1 %.not, float %1154, float %1153, !dbg !12
  %1162 = bitcast float %1161 to i32, !dbg !12
  %1163 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %835, i32 %1162), !dbg !12
  %1164 = bitcast i32 %1163 to float, !dbg !12
  %1165 = select i1 %.not, float %1153, float %1164, !dbg !12
  %1166 = select i1 %.not, float %1164, float %1154, !dbg !12
  %1167 = select i1 %.not18, float %1165, float %1159, !dbg !12
  %1168 = bitcast float %1167 to i32, !dbg !12
  %1169 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %850, i32 %1168), !dbg !12
  %1170 = bitcast i32 %1169 to float, !dbg !12
  %1171 = select i1 %.not18, float %1166, float %1160, !dbg !12
  %1172 = bitcast float %1171 to i32, !dbg !12
  %1173 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %850, i32 %1172), !dbg !12
  %1174 = bitcast i32 %1173 to float, !dbg !12
  %1175 = insertelement <4 x float> poison, float %1159, i64 0, !dbg !12
  %1176 = insertelement <4 x float> %1175, float %1160, i64 1, !dbg !12
  %1177 = insertelement <4 x float> %1176, float %1170, i64 2, !dbg !12
  %1178 = insertelement <4 x float> %1177, float %1174, i64 3, !dbg !12
  %1179 = insertelement <4 x float> poison, float %1170, i64 0, !dbg !12
  %1180 = insertelement <4 x float> %1179, float %1174, i64 1, !dbg !12
  %1181 = insertelement <4 x float> %1180, float %1165, i64 2, !dbg !12
  %1182 = insertelement <4 x float> %1181, float %1166, i64 3, !dbg !12
  %1183 = select <4 x i1> %859, <4 x float> %1178, <4 x float> %1182, !dbg !12
  %1184 = extractvalue [2 x <8 x float>] %.lcssa47, 1, !dbg !12
  %1185 = extractelement <8 x float> %1184, i64 0, !dbg !12
  %1186 = extractelement <8 x float> %1184, i64 1, !dbg !12
  %1187 = extractelement <8 x float> %1184, i64 2, !dbg !12
  %1188 = extractelement <8 x float> %1184, i64 3, !dbg !12
  %1189 = select i1 %.not, float %1186, float %1185, !dbg !12
  %1190 = bitcast float %1189 to i32, !dbg !12
  %1191 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %835, i32 %1190), !dbg !12
  %1192 = bitcast i32 %1191 to float, !dbg !12
  %1193 = select i1 %.not, float %1185, float %1192, !dbg !12
  %1194 = select i1 %.not, float %1192, float %1186, !dbg !12
  %1195 = select i1 %.not, float %1188, float %1187, !dbg !12
  %1196 = bitcast float %1195 to i32, !dbg !12
  %1197 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %835, i32 %1196), !dbg !12
  %1198 = bitcast i32 %1197 to float, !dbg !12
  %1199 = select i1 %.not, float %1187, float %1198, !dbg !12
  %1200 = select i1 %.not, float %1198, float %1188, !dbg !12
  %1201 = select i1 %.not18, float %1199, float %1193, !dbg !12
  %1202 = bitcast float %1201 to i32, !dbg !12
  %1203 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %850, i32 %1202), !dbg !12
  %1204 = select i1 %.not18, float %1200, float %1194, !dbg !12
  %1205 = bitcast float %1204 to i32, !dbg !12
  %1206 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %850, i32 %1205), !dbg !12
  %1207 = shufflevector <1 x float> %819, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !29
  %1208 = extractelement <8 x float> %1082, i64 4, !dbg !12
  %1209 = extractelement <8 x float> %1082, i64 5, !dbg !12
  %1210 = extractelement <8 x float> %1082, i64 6, !dbg !12
  %1211 = extractelement <8 x float> %1082, i64 7, !dbg !12
  %1212 = select i1 %.not, float %1209, float %1208, !dbg !12
  %1213 = bitcast float %1212 to i32, !dbg !12
  %1214 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %835, i32 %1213), !dbg !12
  %1215 = bitcast i32 %1214 to float, !dbg !12
  %1216 = select i1 %.not, float %1208, float %1215, !dbg !12
  %1217 = select i1 %.not, float %1215, float %1209, !dbg !12
  %1218 = select i1 %.not, float %1211, float %1210, !dbg !12
  %1219 = bitcast float %1218 to i32, !dbg !12
  %1220 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %835, i32 %1219), !dbg !12
  %1221 = bitcast i32 %1220 to float, !dbg !12
  %1222 = select i1 %.not, float %1210, float %1221, !dbg !12
  %1223 = select i1 %.not, float %1221, float %1211, !dbg !12
  %1224 = select i1 %.not18, float %1222, float %1216, !dbg !12
  %1225 = bitcast float %1224 to i32, !dbg !12
  %1226 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %850, i32 %1225), !dbg !12
  %1227 = bitcast i32 %1226 to float, !dbg !12
  %1228 = select i1 %.not18, float %1223, float %1217, !dbg !12
  %1229 = bitcast float %1228 to i32, !dbg !12
  %1230 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %850, i32 %1229), !dbg !12
  %1231 = bitcast i32 %1230 to float, !dbg !12
  %1232 = insertelement <4 x float> poison, float %1216, i64 0, !dbg !12
  %1233 = insertelement <4 x float> %1232, float %1217, i64 1, !dbg !12
  %1234 = insertelement <4 x float> %1233, float %1227, i64 2, !dbg !12
  %1235 = insertelement <4 x float> %1234, float %1231, i64 3, !dbg !12
  %1236 = insertelement <4 x float> poison, float %1227, i64 0, !dbg !12
  %1237 = insertelement <4 x float> %1236, float %1231, i64 1, !dbg !12
  %1238 = insertelement <4 x float> %1237, float %1222, i64 2, !dbg !12
  %1239 = insertelement <4 x float> %1238, float %1223, i64 3, !dbg !12
  %1240 = select <4 x i1> %859, <4 x float> %1235, <4 x float> %1239, !dbg !12
  %1241 = extractelement <8 x float> %1116, i64 4, !dbg !12
  %1242 = extractelement <8 x float> %1116, i64 5, !dbg !12
  %1243 = extractelement <8 x float> %1116, i64 6, !dbg !12
  %1244 = extractelement <8 x float> %1116, i64 7, !dbg !12
  %1245 = select i1 %.not, float %1242, float %1241, !dbg !12
  %1246 = bitcast float %1245 to i32, !dbg !12
  %1247 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %835, i32 %1246), !dbg !12
  %1248 = bitcast i32 %1247 to float, !dbg !12
  %1249 = select i1 %.not, float %1241, float %1248, !dbg !12
  %1250 = select i1 %.not, float %1248, float %1242, !dbg !12
  %1251 = select i1 %.not, float %1244, float %1243, !dbg !12
  %1252 = bitcast float %1251 to i32, !dbg !12
  %1253 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %835, i32 %1252), !dbg !12
  %1254 = bitcast i32 %1253 to float, !dbg !12
  %1255 = select i1 %.not, float %1243, float %1254, !dbg !12
  %1256 = select i1 %.not, float %1254, float %1244, !dbg !12
  %1257 = select i1 %.not18, float %1255, float %1249, !dbg !12
  %1258 = bitcast float %1257 to i32, !dbg !12
  %1259 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %850, i32 %1258), !dbg !12
  %1260 = bitcast i32 %1259 to float, !dbg !12
  %1261 = select i1 %.not18, float %1256, float %1250, !dbg !12
  %1262 = bitcast float %1261 to i32, !dbg !12
  %1263 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %850, i32 %1262), !dbg !12
  %1264 = bitcast i32 %1263 to float, !dbg !12
  %1265 = insertelement <4 x float> poison, float %1249, i64 0, !dbg !12
  %1266 = insertelement <4 x float> %1265, float %1250, i64 1, !dbg !12
  %1267 = insertelement <4 x float> %1266, float %1260, i64 2, !dbg !12
  %1268 = insertelement <4 x float> %1267, float %1264, i64 3, !dbg !12
  %1269 = insertelement <4 x float> poison, float %1260, i64 0, !dbg !12
  %1270 = insertelement <4 x float> %1269, float %1264, i64 1, !dbg !12
  %1271 = insertelement <4 x float> %1270, float %1255, i64 2, !dbg !12
  %1272 = insertelement <4 x float> %1271, float %1256, i64 3, !dbg !12
  %1273 = select <4 x i1> %859, <4 x float> %1268, <4 x float> %1272, !dbg !12
  %1274 = extractelement <8 x float> %1150, i64 4, !dbg !12
  %1275 = extractelement <8 x float> %1150, i64 5, !dbg !12
  %1276 = extractelement <8 x float> %1150, i64 6, !dbg !12
  %1277 = extractelement <8 x float> %1150, i64 7, !dbg !12
  %1278 = select i1 %.not, float %1275, float %1274, !dbg !12
  %1279 = bitcast float %1278 to i32, !dbg !12
  %1280 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %835, i32 %1279), !dbg !12
  %1281 = bitcast i32 %1280 to float, !dbg !12
  %1282 = select i1 %.not, float %1274, float %1281, !dbg !12
  %1283 = select i1 %.not, float %1281, float %1275, !dbg !12
  %1284 = select i1 %.not, float %1277, float %1276, !dbg !12
  %1285 = bitcast float %1284 to i32, !dbg !12
  %1286 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %835, i32 %1285), !dbg !12
  %1287 = bitcast i32 %1286 to float, !dbg !12
  %1288 = select i1 %.not, float %1276, float %1287, !dbg !12
  %1289 = select i1 %.not, float %1287, float %1277, !dbg !12
  %1290 = select i1 %.not18, float %1288, float %1282, !dbg !12
  %1291 = bitcast float %1290 to i32, !dbg !12
  %1292 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %850, i32 %1291), !dbg !12
  %1293 = bitcast i32 %1292 to float, !dbg !12
  %1294 = select i1 %.not18, float %1289, float %1283, !dbg !12
  %1295 = bitcast float %1294 to i32, !dbg !12
  %1296 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %850, i32 %1295), !dbg !12
  %1297 = bitcast i32 %1296 to float, !dbg !12
  %1298 = insertelement <4 x float> poison, float %1282, i64 0, !dbg !12
  %1299 = insertelement <4 x float> %1298, float %1283, i64 1, !dbg !12
  %1300 = insertelement <4 x float> %1299, float %1293, i64 2, !dbg !12
  %1301 = insertelement <4 x float> %1300, float %1297, i64 3, !dbg !12
  %1302 = insertelement <4 x float> poison, float %1293, i64 0, !dbg !12
  %1303 = insertelement <4 x float> %1302, float %1297, i64 1, !dbg !12
  %1304 = insertelement <4 x float> %1303, float %1288, i64 2, !dbg !12
  %1305 = insertelement <4 x float> %1304, float %1289, i64 3, !dbg !12
  %1306 = select <4 x i1> %859, <4 x float> %1301, <4 x float> %1305, !dbg !12
  %1307 = extractelement <8 x float> %1184, i64 4, !dbg !12
  %1308 = extractelement <8 x float> %1184, i64 5, !dbg !12
  %1309 = extractelement <8 x float> %1184, i64 6, !dbg !12
  %1310 = extractelement <8 x float> %1184, i64 7, !dbg !12
  %1311 = select i1 %.not, float %1308, float %1307, !dbg !12
  %1312 = bitcast float %1311 to i32, !dbg !12
  %1313 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %835, i32 %1312), !dbg !12
  %1314 = bitcast i32 %1313 to float, !dbg !12
  %1315 = select i1 %.not, float %1307, float %1314, !dbg !12
  %1316 = select i1 %.not, float %1314, float %1308, !dbg !12
  %1317 = select i1 %.not, float %1310, float %1309, !dbg !12
  %1318 = bitcast float %1317 to i32, !dbg !12
  %1319 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %835, i32 %1318), !dbg !12
  %1320 = bitcast i32 %1319 to float, !dbg !12
  %1321 = select i1 %.not, float %1309, float %1320, !dbg !12
  %1322 = select i1 %.not, float %1320, float %1310, !dbg !12
  %1323 = select i1 %.not18, float %1321, float %1315, !dbg !12
  %1324 = bitcast float %1323 to i32, !dbg !12
  %1325 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %850, i32 %1324), !dbg !12
  %1326 = select i1 %.not18, float %1322, float %1316, !dbg !12
  %1327 = bitcast float %1326 to i32, !dbg !12
  %1328 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %850, i32 %1327), !dbg !12
  %1329 = bitcast i32 %1325 to float, !dbg !12
  %1330 = bitcast i32 %1328 to float, !dbg !12
  %1331 = insertelement <4 x float> poison, float %1315, i64 0, !dbg !12
  %1332 = insertelement <4 x float> %1331, float %1316, i64 1, !dbg !12
  %1333 = insertelement <4 x float> %1332, float %1329, i64 2, !dbg !12
  %1334 = insertelement <4 x float> %1333, float %1330, i64 3, !dbg !12
  %1335 = insertelement <4 x float> poison, float %1329, i64 0, !dbg !12
  %1336 = insertelement <4 x float> %1335, float %1330, i64 1, !dbg !12
  %1337 = insertelement <4 x float> %1336, float %1321, i64 2, !dbg !12
  %1338 = insertelement <4 x float> %1337, float %1322, i64 3, !dbg !12
  %1339 = select <4 x i1> %859, <4 x float> %1334, <4 x float> %1338, !dbg !12
  %row.reciprocal = fdiv arcp <4 x float> splat (float 1.000000e+00), %1207, !dbg !29
  %normalized = fmul <4 x float> %row.reciprocal, %1339, !dbg !29
  %1340 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !30
  %1341 = extractelement <4 x half> %1340, i64 3, !dbg !30
  %1342 = extractelement <4 x half> %1340, i64 2, !dbg !30
  %1343 = extractelement <4 x half> %1340, i64 1, !dbg !30
  %1344 = extractelement <4 x half> %1340, i64 0, !dbg !30
  %1345 = bitcast i32 %1203 to float, !dbg !12
  %1346 = bitcast i32 %1206 to float, !dbg !12
  %1347 = insertelement <4 x float> poison, float %1193, i64 0, !dbg !12
  %1348 = insertelement <4 x float> %1347, float %1194, i64 1, !dbg !12
  %1349 = insertelement <4 x float> %1348, float %1345, i64 2, !dbg !12
  %1350 = insertelement <4 x float> %1349, float %1346, i64 3, !dbg !12
  %1351 = insertelement <4 x float> poison, float %1345, i64 0, !dbg !12
  %1352 = insertelement <4 x float> %1351, float %1346, i64 1, !dbg !12
  %1353 = insertelement <4 x float> %1352, float %1199, i64 2, !dbg !12
  %1354 = insertelement <4 x float> %1353, float %1200, i64 3, !dbg !12
  %1355 = select <4 x i1> %859, <4 x float> %1350, <4 x float> %1354, !dbg !12
  %normalized54 = fmul <4 x float> %row.reciprocal, %1355, !dbg !29
  %1356 = fptrunc <4 x float> %normalized54 to <4 x half>, !dbg !30
  %1357 = extractelement <4 x half> %1356, i64 3, !dbg !30
  %1358 = extractelement <4 x half> %1356, i64 2, !dbg !30
  %1359 = extractelement <4 x half> %1356, i64 1, !dbg !30
  %1360 = extractelement <4 x half> %1356, i64 0, !dbg !30
  %1361 = bitcast i32 %1078 to float, !dbg !12
  %1362 = bitcast i32 %1081 to float, !dbg !12
  %1363 = insertelement <4 x float> poison, float %1068, i64 0, !dbg !12
  %1364 = insertelement <4 x float> %1363, float %1069, i64 1, !dbg !12
  %1365 = insertelement <4 x float> %1364, float %1361, i64 2, !dbg !12
  %1366 = insertelement <4 x float> %1365, float %1362, i64 3, !dbg !12
  %1367 = insertelement <4 x float> poison, float %1361, i64 0, !dbg !12
  %1368 = insertelement <4 x float> %1367, float %1362, i64 1, !dbg !12
  %1369 = insertelement <4 x float> %1368, float %1074, i64 2, !dbg !12
  %1370 = insertelement <4 x float> %1369, float %1075, i64 3, !dbg !12
  %1371 = select <4 x i1> %859, <4 x float> %1366, <4 x float> %1370, !dbg !12
  %row.reciprocal55 = fdiv arcp <4 x float> splat (float 1.000000e+00), %960, !dbg !29
  %normalized56 = fmul <4 x float> %row.reciprocal55, %1371, !dbg !29
  %1372 = fptrunc <4 x float> %normalized56 to <4 x half>, !dbg !30
  %1373 = extractelement <4 x half> %1372, i64 3, !dbg !30
  %1374 = extractelement <4 x half> %1372, i64 2, !dbg !30
  %1375 = extractelement <4 x half> %1372, i64 1, !dbg !30
  %1376 = extractelement <4 x half> %1372, i64 0, !dbg !30
  %1377 = bitcast i32 %956 to float, !dbg !12
  %1378 = bitcast i32 %959 to float, !dbg !12
  %1379 = insertelement <4 x float> poison, float %946, i64 0, !dbg !12
  %1380 = insertelement <4 x float> %1379, float %947, i64 1, !dbg !12
  %1381 = insertelement <4 x float> %1380, float %1377, i64 2, !dbg !12
  %1382 = insertelement <4 x float> %1381, float %1378, i64 3, !dbg !12
  %1383 = insertelement <4 x float> poison, float %1377, i64 0, !dbg !12
  %1384 = insertelement <4 x float> %1383, float %1378, i64 1, !dbg !12
  %1385 = insertelement <4 x float> %1384, float %952, i64 2, !dbg !12
  %1386 = insertelement <4 x float> %1385, float %953, i64 3, !dbg !12
  %1387 = select <4 x i1> %859, <4 x float> %1382, <4 x float> %1386, !dbg !12
  %normalized57 = fmul <4 x float> %row.reciprocal55, %1387, !dbg !29
  %1388 = fptrunc <4 x float> %normalized57 to <4 x half>, !dbg !30
  %1389 = extractelement <4 x half> %1388, i64 3, !dbg !30
  %1390 = extractelement <4 x half> %1388, i64 2, !dbg !30
  %1391 = extractelement <4 x half> %1388, i64 1, !dbg !30
  %1392 = extractelement <4 x half> %1388, i64 0, !dbg !30
  %address.thread620 = tail call i32 asm sideeffect "", "=v,0"(i32 %29) #8, !dbg !31, !range !9, !frisk.register_lifetimes !5
  %address.local621 = zext nneg i32 %address.thread620 to i64, !dbg !8
  %1393 = lshr i64 %address.local621, 2, !dbg !31
  %address.local624 = and i64 %1393, 48, !dbg !31
  %address.local626 = and i64 %address.local621, 240, !dbg !31
  %address.local625 = add i64 %26, %address.local621, !dbg !31
  %address.local627 = sub i64 %address.local625, %address.local626, !dbg !31
  %address.local628 = add i64 %address.local627, %address.local624, !dbg !31
  %1394 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %28, !dbg !31
  %address.local631.idx = shl nuw nsw i64 %address.local628, 8, !dbg !31
  %address.local631 = getelementptr inbounds nuw i8, ptr addrspace(1) %1394, i64 %address.local631.idx, !dbg !31
  %address.local633 = and i64 %1393, 12, !dbg !10
  %address.local634 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local631, i64 %address.local633, !dbg !31
  store half %1392, ptr addrspace(1) %address.local634, align 2, !dbg !31
  %address.local636 = or disjoint i64 %address.local633, 1, !dbg !31
  %address.local637 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local631, i64 %address.local636, !dbg !31
  store half %1391, ptr addrspace(1) %address.local637, align 2, !dbg !31
  %address.local639 = or disjoint i64 %address.local633, 2, !dbg !31
  %address.local640 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local631, i64 %address.local639, !dbg !31
  store half %1390, ptr addrspace(1) %address.local640, align 2, !dbg !31
  %address.local642 = or disjoint i64 %address.local633, 3, !dbg !31
  %address.local643 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local631, i64 %address.local642, !dbg !31
  store half %1389, ptr addrspace(1) %address.local643, align 2, !dbg !31
  %address.local645 = or disjoint i64 %address.local633, 16, !dbg !31
  %address.local646 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local631, i64 %address.local645, !dbg !31
  store half %1376, ptr addrspace(1) %address.local646, align 2, !dbg !31
  %address.local648 = or disjoint i64 %address.local633, 17, !dbg !31
  %address.local649 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local631, i64 %address.local648, !dbg !31
  store half %1375, ptr addrspace(1) %address.local649, align 2, !dbg !31
  %address.local651 = or disjoint i64 %address.local633, 18, !dbg !31
  %address.local652 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local631, i64 %address.local651, !dbg !31
  store half %1374, ptr addrspace(1) %address.local652, align 2, !dbg !31
  %address.local654 = or disjoint i64 %address.local633, 19, !dbg !31
  %address.local655 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local631, i64 %address.local654, !dbg !31
  store half %1373, ptr addrspace(1) %address.local655, align 2, !dbg !31
  %address.local659 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local631, i64 16384, !dbg !31
  %address.local660 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local659, i64 %address.local633, !dbg !31
  store half %1360, ptr addrspace(1) %address.local660, align 2, !dbg !31
  %address.local662 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local659, i64 %address.local636, !dbg !31
  store half %1359, ptr addrspace(1) %address.local662, align 2, !dbg !31
  %address.local664 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local659, i64 %address.local639, !dbg !31
  store half %1358, ptr addrspace(1) %address.local664, align 2, !dbg !31
  %address.local666 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local659, i64 %address.local642, !dbg !31
  store half %1357, ptr addrspace(1) %address.local666, align 2, !dbg !31
  %address.local668 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local659, i64 %address.local645, !dbg !31
  store half %1344, ptr addrspace(1) %address.local668, align 2, !dbg !31
  %address.local670 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local659, i64 %address.local648, !dbg !31
  store half %1343, ptr addrspace(1) %address.local670, align 2, !dbg !31
  %address.local672 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local659, i64 %address.local651, !dbg !31
  store half %1342, ptr addrspace(1) %address.local672, align 2, !dbg !31
  %address.local674 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local659, i64 %address.local654, !dbg !31
  store half %1341, ptr addrspace(1) %address.local674, align 2, !dbg !31
  %normalized59 = fmul <4 x float> %row.reciprocal, %1306, !dbg !29
  %1395 = fptrunc <4 x float> %normalized59 to <4 x half>, !dbg !30
  %1396 = extractelement <4 x half> %1395, i64 3, !dbg !30
  %1397 = extractelement <4 x half> %1395, i64 2, !dbg !30
  %1398 = extractelement <4 x half> %1395, i64 1, !dbg !30
  %1399 = extractelement <4 x half> %1395, i64 0, !dbg !30
  %normalized60 = fmul <4 x float> %row.reciprocal, %1183, !dbg !29
  %1400 = fptrunc <4 x float> %normalized60 to <4 x half>, !dbg !30
  %1401 = extractelement <4 x half> %1400, i64 3, !dbg !30
  %1402 = extractelement <4 x half> %1400, i64 2, !dbg !30
  %1403 = extractelement <4 x half> %1400, i64 1, !dbg !30
  %1404 = extractelement <4 x half> %1400, i64 0, !dbg !30
  %normalized62 = fmul <4 x float> %row.reciprocal55, %1059, !dbg !29
  %1405 = fptrunc <4 x float> %normalized62 to <4 x half>, !dbg !30
  %1406 = extractelement <4 x half> %1405, i64 3, !dbg !30
  %1407 = extractelement <4 x half> %1405, i64 2, !dbg !30
  %1408 = extractelement <4 x half> %1405, i64 1, !dbg !30
  %1409 = extractelement <4 x half> %1405, i64 0, !dbg !30
  %normalized63 = fmul <4 x float> %row.reciprocal55, %936, !dbg !29
  %1410 = fptrunc <4 x float> %normalized63 to <4 x half>, !dbg !30
  %1411 = extractelement <4 x half> %1410, i64 3, !dbg !30
  %1412 = extractelement <4 x half> %1410, i64 2, !dbg !30
  %1413 = extractelement <4 x half> %1410, i64 1, !dbg !30
  %1414 = extractelement <4 x half> %1410, i64 0, !dbg !30
  %address.local689 = or disjoint i64 %address.local633, 32, !dbg !31
  %address.local690 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local631, i64 %address.local689, !dbg !31
  store half %1414, ptr addrspace(1) %address.local690, align 2, !dbg !31
  %address.local692 = or disjoint i64 %address.local633, 33, !dbg !31
  %address.local693 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local631, i64 %address.local692, !dbg !31
  store half %1413, ptr addrspace(1) %address.local693, align 2, !dbg !31
  %address.local695 = or disjoint i64 %address.local633, 34, !dbg !31
  %address.local696 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local631, i64 %address.local695, !dbg !31
  store half %1412, ptr addrspace(1) %address.local696, align 2, !dbg !31
  %address.local698 = or disjoint i64 %address.local633, 35, !dbg !31
  %address.local699 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local631, i64 %address.local698, !dbg !31
  store half %1411, ptr addrspace(1) %address.local699, align 2, !dbg !31
  %address.local701 = or disjoint i64 %address.local633, 48, !dbg !31
  %address.local702 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local631, i64 %address.local701, !dbg !31
  store half %1409, ptr addrspace(1) %address.local702, align 2, !dbg !31
  %address.local704 = or disjoint i64 %address.local633, 49, !dbg !31
  %address.local705 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local631, i64 %address.local704, !dbg !31
  store half %1408, ptr addrspace(1) %address.local705, align 2, !dbg !31
  %address.local707 = or disjoint i64 %address.local633, 50, !dbg !31
  %address.local708 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local631, i64 %address.local707, !dbg !31
  store half %1407, ptr addrspace(1) %address.local708, align 2, !dbg !31
  %address.local710 = or i64 %1393, 51, !dbg !31
  %address.local711 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local631, i64 %address.local710, !dbg !31
  store half %1406, ptr addrspace(1) %address.local711, align 2, !dbg !31
  %address.local716 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local659, i64 %address.local689, !dbg !31
  store half %1404, ptr addrspace(1) %address.local716, align 2, !dbg !31
  %address.local718 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local659, i64 %address.local692, !dbg !31
  store half %1403, ptr addrspace(1) %address.local718, align 2, !dbg !31
  %address.local720 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local659, i64 %address.local695, !dbg !31
  store half %1402, ptr addrspace(1) %address.local720, align 2, !dbg !31
  %address.local722 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local659, i64 %address.local698, !dbg !31
  store half %1401, ptr addrspace(1) %address.local722, align 2, !dbg !31
  %address.local724 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local659, i64 %address.local701, !dbg !31
  store half %1399, ptr addrspace(1) %address.local724, align 2, !dbg !31
  %address.local726 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local659, i64 %address.local704, !dbg !31
  store half %1398, ptr addrspace(1) %address.local726, align 2, !dbg !31
  %address.local728 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local659, i64 %address.local707, !dbg !31
  store half %1397, ptr addrspace(1) %address.local728, align 2, !dbg !31
  %address.local730 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local659, i64 %address.local710, !dbg !31
  store half %1396, ptr addrspace(1) %address.local730, align 2, !dbg !31
  %normalized65 = fmul <4 x float> %row.reciprocal, %1273, !dbg !29
  %1415 = fptrunc <4 x float> %normalized65 to <4 x half>, !dbg !30
  %1416 = extractelement <4 x half> %1415, i64 3, !dbg !30
  %1417 = extractelement <4 x half> %1415, i64 2, !dbg !30
  %1418 = extractelement <4 x half> %1415, i64 1, !dbg !30
  %1419 = extractelement <4 x half> %1415, i64 0, !dbg !30
  %normalized66 = fmul <4 x float> %row.reciprocal, %1149, !dbg !29
  %1420 = fptrunc <4 x float> %normalized66 to <4 x half>, !dbg !30
  %1421 = extractelement <4 x half> %1420, i64 3, !dbg !30
  %1422 = extractelement <4 x half> %1420, i64 2, !dbg !30
  %1423 = extractelement <4 x half> %1420, i64 1, !dbg !30
  %1424 = extractelement <4 x half> %1420, i64 0, !dbg !30
  %normalized68 = fmul <4 x float> %row.reciprocal55, %1026, !dbg !29
  %1425 = fptrunc <4 x float> %normalized68 to <4 x half>, !dbg !30
  %1426 = extractelement <4 x half> %1425, i64 3, !dbg !30
  %1427 = extractelement <4 x half> %1425, i64 2, !dbg !30
  %1428 = extractelement <4 x half> %1425, i64 1, !dbg !30
  %1429 = extractelement <4 x half> %1425, i64 0, !dbg !30
  %normalized69 = fmul <4 x float> %row.reciprocal55, %902, !dbg !29
  %1430 = fptrunc <4 x float> %normalized69 to <4 x half>, !dbg !30
  %1431 = extractelement <4 x half> %1430, i64 3, !dbg !30
  %1432 = extractelement <4 x half> %1430, i64 2, !dbg !30
  %1433 = extractelement <4 x half> %1430, i64 1, !dbg !30
  %1434 = extractelement <4 x half> %1430, i64 0, !dbg !30
  %address.local745 = or disjoint i64 %address.local633, 64, !dbg !31
  %address.local746 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local631, i64 %address.local745, !dbg !31
  store half %1434, ptr addrspace(1) %address.local746, align 2, !dbg !31
  %address.local748 = or disjoint i64 %address.local633, 65, !dbg !31
  %address.local749 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local631, i64 %address.local748, !dbg !31
  store half %1433, ptr addrspace(1) %address.local749, align 2, !dbg !31
  %address.local751 = or disjoint i64 %address.local633, 66, !dbg !31
  %address.local752 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local631, i64 %address.local751, !dbg !31
  store half %1432, ptr addrspace(1) %address.local752, align 2, !dbg !31
  %address.local754 = or disjoint i64 %address.local633, 67, !dbg !31
  %address.local755 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local631, i64 %address.local754, !dbg !31
  store half %1431, ptr addrspace(1) %address.local755, align 2, !dbg !31
  %address.local757 = or disjoint i64 %address.local633, 80, !dbg !31
  %address.local758 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local631, i64 %address.local757, !dbg !31
  store half %1429, ptr addrspace(1) %address.local758, align 2, !dbg !31
  %address.local760 = or disjoint i64 %address.local633, 81, !dbg !31
  %address.local761 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local631, i64 %address.local760, !dbg !31
  store half %1428, ptr addrspace(1) %address.local761, align 2, !dbg !31
  %address.local763 = or disjoint i64 %address.local633, 82, !dbg !31
  %address.local764 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local631, i64 %address.local763, !dbg !31
  store half %1427, ptr addrspace(1) %address.local764, align 2, !dbg !31
  %address.local766 = or disjoint i64 %address.local633, 83, !dbg !31
  %address.local767 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local631, i64 %address.local766, !dbg !31
  store half %1426, ptr addrspace(1) %address.local767, align 2, !dbg !31
  %address.local772 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local659, i64 %address.local745, !dbg !31
  store half %1424, ptr addrspace(1) %address.local772, align 2, !dbg !31
  %address.local774 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local659, i64 %address.local748, !dbg !31
  store half %1423, ptr addrspace(1) %address.local774, align 2, !dbg !31
  %address.local776 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local659, i64 %address.local751, !dbg !31
  store half %1422, ptr addrspace(1) %address.local776, align 2, !dbg !31
  %address.local778 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local659, i64 %address.local754, !dbg !31
  store half %1421, ptr addrspace(1) %address.local778, align 2, !dbg !31
  %address.local780 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local659, i64 %address.local757, !dbg !31
  store half %1419, ptr addrspace(1) %address.local780, align 2, !dbg !31
  %address.local782 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local659, i64 %address.local760, !dbg !31
  store half %1418, ptr addrspace(1) %address.local782, align 2, !dbg !31
  %address.local784 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local659, i64 %address.local763, !dbg !31
  store half %1417, ptr addrspace(1) %address.local784, align 2, !dbg !31
  %address.local786 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local659, i64 %address.local766, !dbg !31
  store half %1416, ptr addrspace(1) %address.local786, align 2, !dbg !31
  %normalized71 = fmul <4 x float> %row.reciprocal, %1240, !dbg !29
  %1435 = fptrunc <4 x float> %normalized71 to <4 x half>, !dbg !30
  %1436 = extractelement <4 x half> %1435, i64 3, !dbg !30
  %1437 = extractelement <4 x half> %1435, i64 2, !dbg !30
  %1438 = extractelement <4 x half> %1435, i64 1, !dbg !30
  %1439 = extractelement <4 x half> %1435, i64 0, !dbg !30
  %normalized72 = fmul <4 x float> %row.reciprocal, %1115, !dbg !29
  %1440 = fptrunc <4 x float> %normalized72 to <4 x half>, !dbg !30
  %1441 = extractelement <4 x half> %1440, i64 3, !dbg !30
  %1442 = extractelement <4 x half> %1440, i64 2, !dbg !30
  %1443 = extractelement <4 x half> %1440, i64 1, !dbg !30
  %1444 = extractelement <4 x half> %1440, i64 0, !dbg !30
  %normalized74 = fmul <4 x float> %row.reciprocal55, %993, !dbg !29
  %1445 = fptrunc <4 x float> %normalized74 to <4 x half>, !dbg !30
  %1446 = extractelement <4 x half> %1445, i64 3, !dbg !30
  %1447 = extractelement <4 x half> %1445, i64 2, !dbg !30
  %1448 = extractelement <4 x half> %1445, i64 1, !dbg !30
  %1449 = extractelement <4 x half> %1445, i64 0, !dbg !30
  %normalized75 = fmul <4 x float> %row.reciprocal55, %868, !dbg !29
  %1450 = fptrunc <4 x float> %normalized75 to <4 x half>, !dbg !30
  %1451 = extractelement <4 x half> %1450, i64 3, !dbg !30
  %1452 = extractelement <4 x half> %1450, i64 2, !dbg !30
  %1453 = extractelement <4 x half> %1450, i64 1, !dbg !30
  %1454 = extractelement <4 x half> %1450, i64 0, !dbg !30
  %address.local801 = or disjoint i64 %address.local633, 96, !dbg !31
  %address.local802 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local631, i64 %address.local801, !dbg !31
  store half %1454, ptr addrspace(1) %address.local802, align 2, !dbg !31
  %address.local804 = or disjoint i64 %address.local633, 97, !dbg !31
  %address.local805 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local631, i64 %address.local804, !dbg !31
  store half %1453, ptr addrspace(1) %address.local805, align 2, !dbg !31
  %address.local807 = or disjoint i64 %address.local633, 98, !dbg !31
  %address.local808 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local631, i64 %address.local807, !dbg !31
  store half %1452, ptr addrspace(1) %address.local808, align 2, !dbg !31
  %address.local810 = or disjoint i64 %address.local633, 99, !dbg !31
  %address.local811 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local631, i64 %address.local810, !dbg !31
  store half %1451, ptr addrspace(1) %address.local811, align 2, !dbg !31
  %address.local813 = or disjoint i64 %address.local633, 112, !dbg !31
  %address.local814 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local631, i64 %address.local813, !dbg !31
  store half %1449, ptr addrspace(1) %address.local814, align 2, !dbg !31
  %address.local816 = or disjoint i64 %address.local633, 113, !dbg !31
  %address.local817 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local631, i64 %address.local816, !dbg !31
  store half %1448, ptr addrspace(1) %address.local817, align 2, !dbg !31
  %address.local819 = or disjoint i64 %address.local633, 114, !dbg !31
  %address.local820 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local631, i64 %address.local819, !dbg !31
  store half %1447, ptr addrspace(1) %address.local820, align 2, !dbg !31
  %address.local822 = or i64 %1393, 115, !dbg !31
  %address.local823 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local631, i64 %address.local822, !dbg !31
  store half %1446, ptr addrspace(1) %address.local823, align 2, !dbg !31
  %address.local828 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local659, i64 %address.local801, !dbg !31
  store half %1444, ptr addrspace(1) %address.local828, align 2, !dbg !31
  %address.local830 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local659, i64 %address.local804, !dbg !31
  store half %1443, ptr addrspace(1) %address.local830, align 2, !dbg !31
  %address.local832 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local659, i64 %address.local807, !dbg !31
  store half %1442, ptr addrspace(1) %address.local832, align 2, !dbg !31
  %address.local834 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local659, i64 %address.local810, !dbg !31
  store half %1441, ptr addrspace(1) %address.local834, align 2, !dbg !31
  %address.local836 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local659, i64 %address.local813, !dbg !31
  store half %1439, ptr addrspace(1) %address.local836, align 2, !dbg !31
  %address.local838 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local659, i64 %address.local816, !dbg !31
  store half %1438, ptr addrspace(1) %address.local838, align 2, !dbg !31
  %address.local840 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local659, i64 %address.local819, !dbg !31
  store half %1437, ptr addrspace(1) %address.local840, align 2, !dbg !31
  %address.local842 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local659, i64 %address.local822, !dbg !31
  store half %1436, ptr addrspace(1) %address.local842, align 2, !dbg !31
  ret void, !dbg !7
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.y() #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #2

; Function Attrs: convergent mustprogress nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.readfirstlane.i32(i32) #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.z() #2

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier() #4

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32) #5

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.hi(i32, i32) #5

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.ds.bpermute(i32, i32) #6

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maxnum.f32(float, float) #7

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare <1 x float> @llvm.maxnum.v1f32(<1 x float>, <1 x float>) #7

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.amdgcn.exp2.f32(float) #7

attributes #0 = { convergent mustprogress nofree willreturn memory(read) }
attributes #1 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1" "frisk.global_prefetch"="8" "frisk.register_comparisons"="1" "frisk.register_lifetimes"="1" "uniform-work-group-size" }
attributes #2 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { convergent mustprogress nocallback nocreateundeforpoison nofree nounwind willreturn memory(none) }
attributes #4 = { convergent mustprogress nocallback nofree nounwind willreturn }
attributes #5 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }
attributes #6 = { convergent mustprogress nocallback nofree nounwind willreturn memory(none) }
attributes #7 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #8 = { convergent nounwind willreturn memory(none) }
attributes #9 = { convergent nounwind }
attributes #10 = { convergent willreturn memory(read) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!1 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/deferred_lane_sum_pass/compiler_final/run-20261026-155755-g7dykgmh/input-0002-test_input_1x32x4096x128-128-64-32-online/grid-000003")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !1, file: !1, line: 23, type: !4, scopeLine: 23, spFlags: DISPFlagDefinition, unit: !0)
!4 = !DISubroutineType(types: !5)
!5 = !{}
!6 = !{i32 256, i32 1, i32 1}
!7 = !DILocation(line: 23, column: 3, scope: !3)
!8 = !DILocation(line: 70, column: 22, scope: !3)
!9 = !{i32 0, i32 256}
!10 = !DILocation(line: 72, column: 24, scope: !3)
!11 = !DILocation(line: 77, column: 19, scope: !3)
!12 = !DILocation(line: 60, column: 24, scope: !3)
!13 = !DILocation(line: 71, column: 22, scope: !3)
!14 = !DILocation(line: 73, column: 26, scope: !3)
!15 = !DILocation(line: 80, column: 23, scope: !3)
!16 = !DILocation(line: 101, column: 20, scope: !3)
!17 = !DILocation(line: 81, column: 26, scope: !3)
!18 = !DILocation(line: 88, column: 21, scope: !3)
!19 = !DILocation(line: 89, column: 20, scope: !3)
!20 = !DILocation(line: 90, column: 22, scope: !3)
!21 = !DILocation(line: 91, column: 23, scope: !3)
!22 = !DILocation(line: 92, column: 20, scope: !3)
!23 = !DILocation(line: 103, column: 22, scope: !3)
!24 = !DILocation(line: 93, column: 22, scope: !3)
!25 = !DILocation(line: 94, column: 22, scope: !3)
!26 = !DILocation(line: 96, column: 16, scope: !3)
!27 = !DILocation(line: 95, column: 20, scope: !3)
!28 = !DILocation(line: 102, column: 23, scope: !3)
!29 = !DILocation(line: 107, column: 23, scope: !3)
!30 = !DILocation(line: 108, column: 19, scope: !3)
!31 = !DILocation(line: 109, column: 9, scope: !3)
