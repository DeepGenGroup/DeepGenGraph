; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/noncausal_k_dma_pass/mytest/run-20261030-162611-z5os6mty/grid-000003/kernel.ll'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/noncausal_k_dma_pass/mytest/run-20261030-162611-z5os6mty/grid-000003/input.mlir"

@shm_0 = addrspace(3) global [16384 x i8] undef, align 16

; Function Attrs: convergent mustprogress nofree willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) nofree writeonly captures(none) %3) local_unnamed_addr #1 !dbg !3 !reqd_work_group_size !6 {
  %kdma.tid = tail call i32 @llvm.amdgcn.workitem.id.x(), !dbg !7
  %kdma.wave = and i32 %kdma.tid, 960, !dbg !7
  %kdma.lane = lshr i32 %kdma.tid, 1, !dbg !7
  %5 = and i32 %kdma.lane, 31, !dbg !7
  %kdma.owner = or disjoint i32 %5, %kdma.wave, !dbg !7
  %kdma.owner.hi = or disjoint i32 %kdma.owner, 32, !dbg !7
  %6 = shl nuw nsw i32 %kdma.tid, 3, !dbg !7
  %kdma.intra = and i32 %6, 8, !dbg !7
  %kdma.intra.hi = xor i32 %kdma.intra, 8, !dbg !7
  %7 = tail call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !7
  %8 = sext i32 %7 to i64, !dbg !7
  %9 = tail call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !7
  %10 = sext i32 %9 to i64, !dbg !7
  %11 = shl nsw i64 %10, 5, !dbg !7
  %12 = add nsw i64 %11, %8, !dbg !7
  %13 = lshr i64 %12, 5, !dbg !7
  %14 = and i64 %13, 576460752303423456, !dbg !7
  %15 = and i64 %8, 31, !dbg !7
  %16 = or disjoint i64 %14, %15, !dbg !7
  %17 = sub nsw i64 31, %16, !dbg !7
  %18 = icmp ugt i64 %12, 1023, !dbg !8
  %19 = add nsw i64 %16, -32, !dbg !8
  %20 = select i1 %18, i64 %19, i64 %17, !dbg !8
  %21 = sdiv i64 %20, 32, !dbg !8
  %22 = sext i1 %18 to i64, !dbg !8
  %23 = xor i64 %21, %22, !dbg !8
  %24 = add nsw i64 %23, %13, !dbg !8
  %25 = shl nsw i64 %17, 7, !dbg !8
  %26 = shl i64 %23, 12, !dbg !8
  %27 = sub i64 %25, %26, !dbg !8
  %28 = shl i64 %24, 19, !dbg !8
  %29 = and i64 %28, 16252928, !dbg !8
  %30 = shl nsw i64 %27, 7, !dbg !8
  %31 = add i64 %29, %30, !dbg !8
  %address.thread = tail call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #8, !dbg !8, !range !9, !frisk.register_lifetimes !5
  %address.local = zext nneg i32 %address.thread to i64, !dbg !8
  %address.local271 = shl nuw nsw i64 %address.local, 3, !dbg !8
  %address.local272 = and i64 %address.local271, 1536, !dbg !8
  %address.local273 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local272, !dbg !8
  %address.local276 = shl nuw nsw i64 %address.local, 7, !dbg !8
  %address.local277 = and i64 %address.local276, 16256, !dbg !8
  %address.local278 = lshr i64 %address.local, 4, !dbg !8
  %address.local279 = and i64 %address.local278, 8, !dbg !8
  %address.local280 = or disjoint i64 %address.local277, %address.local279, !dbg !8
  %32 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %.idx1266 = or disjoint i64 %address.local280, %31, !dbg !8
  %.idx1266.tr = trunc i64 %.idx1266 to i32, !dbg !8
  %dma.buffer.offset1263 = shl i32 %.idx1266.tr, 1, !dbg !8
  %33 = insertelement <2 x i64> poison, i64 %32, i64 0, !dbg !8
  %34 = bitcast <2 x i64> %33 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc1264 = shufflevector <4 x i32> %34, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
  %address.local282 = or disjoint i64 %address.local, 256, !dbg !8
  %address.local283 = shl nuw nsw i64 %address.local282, 3, !dbg !8
  %address.local284 = and i64 %address.local283, 3584, !dbg !8
  %address.local285 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local284, !dbg !8
  %address.local290 = lshr i64 %address.local282, 4, !dbg !8
  %address.local291 = and i64 %address.local290, 24, !dbg !8
  %address.local292 = or disjoint i64 %address.local291, %address.local277, !dbg !8
  %.idx1267 = or disjoint i64 %address.local292, %31, !dbg !8
  %.idx1267.tr = trunc i64 %.idx1267 to i32, !dbg !8
  %dma.buffer.offset1260 = shl i32 %.idx1267.tr, 1, !dbg !8
  %address.local294 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local272, !dbg !8
  %35 = or disjoint i64 %31, 32, !dbg !8
  %36 = or disjoint i64 %address.local280, %35, !dbg !8
  %.tr = trunc i64 %36 to i32, !dbg !8
  %dma.buffer.offset1257 = shl i32 %.tr, 1, !dbg !8
  %address.local298 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local284, !dbg !8
  %37 = insertelement <4 x ptr addrspace(3)> poison, ptr addrspace(3) %address.local273, i64 0, !dbg !8
  %38 = insertelement <4 x ptr addrspace(3)> %37, ptr addrspace(3) %address.local285, i64 1, !dbg !8
  %39 = insertelement <4 x ptr addrspace(3)> %38, ptr addrspace(3) %address.local294, i64 2, !dbg !8
  %40 = insertelement <4 x ptr addrspace(3)> %39, ptr addrspace(3) %address.local298, i64 3, !dbg !8
  %41 = ptrtoint <4 x ptr addrspace(3)> %40 to <4 x i64>, !dbg !8
  %42 = trunc <4 x i64> %41 to <4 x i32>, !dbg !8
  %43 = extractelement <4 x i32> %42, i64 0, !dbg !8
  %44 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %43), !dbg !8
  %dma.buffer.lds1265 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %44), !dbg !8
  %45 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1265, i32 %dma.buffer.offset1263, <4 x i32> %dma.buffer.rsrc1264) #9, !dbg !8
  %46 = extractelement <4 x i32> %42, i64 1, !dbg !8
  %47 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %46), !dbg !8
  %dma.buffer.lds1262 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %47), !dbg !8
  %48 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1262, i32 %dma.buffer.offset1260, <4 x i32> %dma.buffer.rsrc1264) #9, !dbg !8
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %49 = extractelement <4 x i32> %42, i64 2, !dbg !8
  %50 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %49), !dbg !8
  %dma.buffer.lds1259 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %50), !dbg !8
  %51 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1259, i32 %dma.buffer.offset1257, <4 x i32> %dma.buffer.rsrc1264) #9, !dbg !8
  %52 = extractelement <4 x i32> %42, i64 3, !dbg !8
  %53 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %52), !dbg !8
  %54 = or disjoint i64 %address.local292, %35, !dbg !8
  %.tr1268 = trunc i64 %54 to i32, !dbg !8
  %dma.buffer.offset1254 = shl i32 %.tr1268, 1, !dbg !8
  %dma.buffer.lds1256 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %53), !dbg !8
  %55 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1256, i32 %dma.buffer.offset1254, <4 x i32> %dma.buffer.rsrc1264) #9, !dbg !8
  %address.local302 = lshr i32 %address.thread, 2, !dbg !10
  %address.local303 = and i32 %address.local302, 4, !dbg !10
  %address.local304 = shl nuw nsw i32 %address.thread, 1, !dbg !10
  %address.local305 = and i32 %address.local304, 384, !dbg !10
  %address.local306 = or disjoint i32 %address.local303, %address.local305, !dbg !10
  %address.local307 = shl nuw nsw i32 %address.thread, 3, !dbg !10
  %address.local308 = and i32 %address.local307, 120, !dbg !10
  %address.local309 = or disjoint i32 %address.local306, %address.local308, !dbg !10
  %address.local310 = shl nuw nsw i32 %address.thread, 5, !dbg !10
  %address.local311 = and i32 %address.local310, 1024, !dbg !10
  %address.local312 = or disjoint i32 %address.local309, %address.local311, !dbg !10
  %address.local313 = zext nneg i32 %address.local312 to i64, !dbg !10
  %address.local314 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local313, !dbg !10
  %56 = load <4 x half>, ptr addrspace(3) %address.local314, align 8, !dbg !10
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local314, i64 4096, !dbg !10
  %57 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !10
  %58 = fmul <4 x half> %56, splat (half 1.275630e-01), !dbg !10
  %lds.const.gep1064 = getelementptr i8, ptr addrspace(3) %address.local314, i64 1024, !dbg !10
  %59 = load <4 x half>, ptr addrspace(3) %lds.const.gep1064, align 8, !dbg !10
  %60 = fmul <4 x half> %57, splat (half 1.275630e-01), !dbg !10
  %lds.const.gep1065 = getelementptr i8, ptr addrspace(3) %address.local314, i64 5120, !dbg !10
  %61 = load <4 x half>, ptr addrspace(3) %lds.const.gep1065, align 8, !dbg !10
  %62 = fmul <4 x half> %59, splat (half 1.275630e-01), !dbg !10
  %63 = fmul <4 x half> %61, splat (half 1.275630e-01), !dbg !10
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %64 = or disjoint i64 %31, 64, !dbg !8
  %65 = or disjoint i64 %address.local280, %64, !dbg !8
  %.tr1269 = trunc i64 %65 to i32, !dbg !8
  %dma.buffer.offset1251 = shl i32 %.tr1269, 1, !dbg !8
  %66 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1265, i32 %dma.buffer.offset1251, <4 x i32> %dma.buffer.rsrc1264) #9, !dbg !8
  %67 = or disjoint i64 %address.local292, %64, !dbg !8
  %.tr1270 = trunc i64 %67 to i32, !dbg !8
  %dma.buffer.offset1248 = shl i32 %.tr1270, 1, !dbg !8
  %68 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1262, i32 %dma.buffer.offset1248, <4 x i32> %dma.buffer.rsrc1264) #9, !dbg !8
  %lds.const.gep1066 = getelementptr i8, ptr addrspace(3) %address.local314, i64 8192, !dbg !10
  %69 = load <4 x half>, ptr addrspace(3) %lds.const.gep1066, align 8, !dbg !10
  %lds.const.gep1067 = getelementptr i8, ptr addrspace(3) %address.local314, i64 12288, !dbg !10
  %70 = load <4 x half>, ptr addrspace(3) %lds.const.gep1067, align 8, !dbg !10
  %lds.const.gep1068 = getelementptr i8, ptr addrspace(3) %address.local314, i64 9216, !dbg !10
  %71 = load <4 x half>, ptr addrspace(3) %lds.const.gep1068, align 8, !dbg !10
  %72 = fmul <4 x half> %69, splat (half 1.275630e-01), !dbg !10
  %lds.const.gep1069 = getelementptr i8, ptr addrspace(3) %address.local314, i64 13312, !dbg !10
  %73 = load <4 x half>, ptr addrspace(3) %lds.const.gep1069, align 8, !dbg !10
  %74 = fmul <4 x half> %70, splat (half 1.275630e-01), !dbg !10
  %75 = fmul <4 x half> %71, splat (half 1.275630e-01), !dbg !10
  %76 = fmul <4 x half> %73, splat (half 1.275630e-01), !dbg !10
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %77 = or disjoint i64 %31, 96, !dbg !8
  %78 = or disjoint i64 %address.local280, %77, !dbg !8
  %.tr1271 = trunc i64 %78 to i32, !dbg !8
  %dma.buffer.offset1245 = shl i32 %.tr1271, 1, !dbg !8
  %79 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1259, i32 %dma.buffer.offset1245, <4 x i32> %dma.buffer.rsrc1264) #9, !dbg !8
  %80 = or disjoint i64 %address.local292, %77, !dbg !8
  %.tr1272 = trunc i64 %80 to i32, !dbg !8
  %dma.buffer.offset1242 = shl i32 %.tr1272, 1, !dbg !8
  %81 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1256, i32 %dma.buffer.offset1242, <4 x i32> %dma.buffer.rsrc1264) #9, !dbg !8
  %82 = load <4 x half>, ptr addrspace(3) %address.local314, align 8, !dbg !10
  %83 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !10
  %84 = load <4 x half>, ptr addrspace(3) %lds.const.gep1064, align 8, !dbg !10
  %85 = fmul <4 x half> %82, splat (half 1.275630e-01), !dbg !10
  %86 = load <4 x half>, ptr addrspace(3) %lds.const.gep1065, align 8, !dbg !10
  %87 = fmul <4 x half> %83, splat (half 1.275630e-01), !dbg !10
  %88 = fmul <4 x half> %84, splat (half 1.275630e-01), !dbg !10
  %89 = fmul <4 x half> %86, splat (half 1.275630e-01), !dbg !10
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %90 = load <4 x half>, ptr addrspace(3) %lds.const.gep1066, align 8, !dbg !10
  %91 = load <4 x half>, ptr addrspace(3) %lds.const.gep1067, align 8, !dbg !10
  %92 = load <4 x half>, ptr addrspace(3) %lds.const.gep1068, align 8, !dbg !10
  %93 = fmul <4 x half> %90, splat (half 1.275630e-01), !dbg !10
  %94 = load <4 x half>, ptr addrspace(3) %lds.const.gep1069, align 8, !dbg !10
  %95 = fmul <4 x half> %91, splat (half 1.275630e-01), !dbg !10
  %96 = fmul <4 x half> %92, splat (half 1.275630e-01), !dbg !10
  %97 = fmul <4 x half> %94, splat (half 1.275630e-01), !dbg !10
  fence syncscope("workgroup") release, !dbg !11
  tail call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %98 = shl nuw nsw i32 %kdma.tid, 2, !dbg !12
  %99 = and i32 %98, 3084, !dbg !12
  %100 = and i32 %kdma.tid, 1, !dbg !12
  %101 = add nuw nsw i32 %100, 1, !dbg !12
  %narrow = or disjoint i32 %101, %99, !dbg !12
  %102 = lshr i32 %narrow, 1, !dbg !12
  %103 = zext nneg i32 %102 to i64, !dbg !12
  %104 = shl i64 %13, 19
  %105 = and i64 %104, 16252928
  %106 = ptrtoint ptr addrspace(1) %1 to i64
  %107 = insertelement <2 x i64> poison, i64 %106, i64 0
  %108 = bitcast <2 x i64> %107 to <4 x i32>
  %dma.buffer.rsrc1240 = shufflevector <4 x i32> %108, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  %kdma.source = zext nneg i32 %kdma.owner to i64
  %kdma.source1373 = lshr i64 %kdma.source, 4
  %kdma.source1374 = and i64 %kdma.source1373, 8
  %kdma.source1375 = and i64 %kdma.source, 91
  %kdma.source1376 = lshr i64 %kdma.source, 2
  %kdma.source1377 = and i64 %kdma.source1376, 4
  %kdma.source1378 = add nuw nsw i64 %kdma.source1377, %kdma.source
  %kdma.source1379 = and i64 %kdma.source1378, 4
  %kdma.source1380 = or disjoint i64 %kdma.source1379, %kdma.source1375
  %kdma.source1381 = shl nuw nsw i64 %kdma.source1380, 7
  %kdma.source1382 = or disjoint i64 %kdma.source1381, %kdma.source1374
  %kdma.source1386 = zext nneg i32 %kdma.owner.hi to i64
  %kdma.source1387 = lshr i64 %kdma.source1386, 4
  %kdma.source1388 = and i64 %kdma.source1387, 8
  %kdma.source1389 = and i64 %kdma.source1386, 123
  %kdma.source1390 = lshr i64 %kdma.source1386, 2
  %kdma.source1391 = and i64 %kdma.source1390, 4
  %kdma.source1392 = add nuw nsw i64 %kdma.source1391, %kdma.source1386
  %kdma.source1393 = and i64 %kdma.source1392, 4
  %kdma.source1394 = or disjoint i64 %kdma.source1393, %kdma.source1389
  %kdma.source1395 = shl nuw nsw i64 %kdma.source1394, 7
  %kdma.source1396 = or disjoint i64 %kdma.source1395, %kdma.source1388
  %kdma.source1413 = or disjoint i64 %kdma.source1382, 16
  %kdma.source1430 = or disjoint i64 %kdma.source1396, 16
  %109 = ptrtoint ptr addrspace(1) %2 to i64
  %110 = insertelement <2 x i64> poison, i64 %109, i64 0
  %111 = bitcast <2 x i64> %110 to <4 x i32>
  %dma.buffer.rsrc1216 = shufflevector <4 x i32> %111, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  br label %112, !dbg !11

112:                                              ; preds = %4, %112
  %113 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %1181, %112 ]
  %114 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %1177, %112 ]
  %115 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %1173, %112 ]
  %116 = phi [2 x <1 x float>] [ [<1 x float> splat (float -inf), <1 x float> splat (float -inf)], %4 ], [ %1169, %112 ]
  %117 = phi [2 x <1 x float>] [ zeroinitializer, %4 ], [ %861, %112 ]
  %118 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %1167, %112 ]
  %119 = phi i64 [ 0, %4 ], [ %1182, %112 ]
  %120 = shl nuw nsw i64 %119, 14, !dbg !13
  %121 = add nuw nsw i64 %120, %105, !dbg !13
  %address.thread365 = tail call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #8, !dbg !13, !range !9, !frisk.register_lifetimes !5
  %address.local366 = zext nneg i32 %address.thread365 to i64, !dbg !8
  %address.local367 = shl nuw nsw i64 %address.local366, 3, !dbg !8
  %address.local368 = and i64 %address.local367, 1536, !dbg !8
  %address.local369 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local368, !dbg !8
  %kdma.source1383 = or disjoint i64 %121, %kdma.source1382, !dbg !13
  %kdma.source1384 = trunc nuw nsw i64 %kdma.source1383 to i32, !dbg !13
  %kdma.source1385 = shl nuw nsw i32 %kdma.source1384, 1, !dbg !13
  %kdma.offset = or disjoint i32 %kdma.source1385, %kdma.intra, !dbg !13
  %kdma.source1397 = or disjoint i64 %121, %kdma.source1396, !dbg !13
  %kdma.source1398 = trunc nuw nsw i64 %kdma.source1397 to i32, !dbg !13
  %kdma.source1399 = shl nuw nsw i32 %kdma.source1398, 1, !dbg !13
  %kdma.offset1400 = or disjoint i32 %kdma.source1399, %kdma.intra.hi, !dbg !13
  %address.local383 = or disjoint i64 %address.local366, 256, !dbg !8
  %address.local384 = shl nuw nsw i64 %address.local383, 3, !dbg !8
  %address.local385 = and i64 %address.local384, 3584, !dbg !8
  %address.local386 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local385, !dbg !8
  %kdma.source1414 = or disjoint i64 %121, %kdma.source1413, !dbg !13
  %kdma.source1415 = trunc nuw nsw i64 %kdma.source1414 to i32, !dbg !13
  %kdma.source1416 = shl nuw nsw i32 %kdma.source1415, 1, !dbg !13
  %kdma.offset1417 = or disjoint i32 %kdma.source1416, %kdma.intra, !dbg !13
  %kdma.source1431 = or disjoint i64 %121, %kdma.source1430, !dbg !13
  %kdma.source1432 = trunc nuw nsw i64 %kdma.source1431 to i32, !dbg !13
  %kdma.source1433 = shl nuw nsw i32 %kdma.source1432, 1, !dbg !13
  %kdma.offset1434 = or disjoint i32 %kdma.source1433, %kdma.intra.hi, !dbg !13
  %122 = or disjoint i64 %121, 32, !dbg !13
  %address.local400 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local368, !dbg !8
  %kdma.source1447 = or disjoint i64 %122, %kdma.source1382, !dbg !13
  %kdma.source1448 = trunc nuw nsw i64 %kdma.source1447 to i32, !dbg !13
  %kdma.source1449 = shl nuw nsw i32 %kdma.source1448, 1, !dbg !13
  %kdma.offset1450 = or disjoint i32 %kdma.source1449, %kdma.intra, !dbg !13
  %kdma.source1463 = or disjoint i64 %122, %kdma.source1396, !dbg !13
  %kdma.source1464 = trunc nuw nsw i64 %kdma.source1463 to i32, !dbg !13
  %kdma.source1465 = shl nuw nsw i32 %kdma.source1464, 1, !dbg !13
  %kdma.offset1466 = or disjoint i32 %kdma.source1465, %kdma.intra.hi, !dbg !13
  %address.local404 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local385, !dbg !8
  %123 = insertelement <4 x ptr addrspace(3)> poison, ptr addrspace(3) %address.local369, i64 0, !dbg !8
  %124 = insertelement <4 x ptr addrspace(3)> %123, ptr addrspace(3) %address.local386, i64 1, !dbg !8
  %125 = insertelement <4 x ptr addrspace(3)> %124, ptr addrspace(3) %address.local400, i64 2, !dbg !8
  %126 = insertelement <4 x ptr addrspace(3)> %125, ptr addrspace(3) %address.local404, i64 3, !dbg !8
  %127 = ptrtoint <4 x ptr addrspace(3)> %126 to <4 x i64>, !dbg !8
  %128 = trunc <4 x i64> %127 to <4 x i32>, !dbg !8
  %129 = extractelement <4 x i32> %128, i64 0, !dbg !13
  %130 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %129), !dbg !13
  %dma.buffer.lds1241 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %130), !dbg !13
  %131 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1241, i32 %kdma.offset, <4 x i32> %dma.buffer.rsrc1240) #9, !dbg !13
  %kdma.lds1401 = add i32 %dma.buffer.lds1241, 512, !dbg !13
  %132 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1401, i32 %kdma.offset1400, <4 x i32> %dma.buffer.rsrc1240) #9, !dbg !13
  %133 = extractelement <4 x i32> %128, i64 1, !dbg !13
  %134 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %133), !dbg !13
  %dma.buffer.lds1238 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %134), !dbg !13
  %135 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1238, i32 %kdma.offset1417, <4 x i32> %dma.buffer.rsrc1240) #9, !dbg !13
  %kdma.lds1435 = add i32 %dma.buffer.lds1238, 512, !dbg !13
  %136 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1435, i32 %kdma.offset1434, <4 x i32> %dma.buffer.rsrc1240) #9, !dbg !13
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  tail call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %137 = extractelement <4 x i32> %128, i64 2, !dbg !13
  %138 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %137), !dbg !13
  %dma.buffer.lds1235 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %138), !dbg !13
  %139 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1235, i32 %kdma.offset1450, <4 x i32> %dma.buffer.rsrc1240) #9, !dbg !13
  %kdma.lds1467 = add i32 %dma.buffer.lds1235, 512, !dbg !13
  %140 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1467, i32 %kdma.offset1466, <4 x i32> %dma.buffer.rsrc1240) #9, !dbg !13
  %141 = extractelement <4 x i32> %128, i64 3, !dbg !13
  %142 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %141), !dbg !13
  %dma.buffer.lds1232 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %142), !dbg !13
  %kdma.source1480 = or disjoint i64 %122, %kdma.source1413, !dbg !13
  %kdma.source1481 = trunc nuw nsw i64 %kdma.source1480 to i32, !dbg !13
  %kdma.source1482 = shl nuw nsw i32 %kdma.source1481, 1, !dbg !13
  %kdma.offset1483 = or disjoint i32 %kdma.source1482, %kdma.intra, !dbg !13
  %143 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1232, i32 %kdma.offset1483, <4 x i32> %dma.buffer.rsrc1240) #9, !dbg !13
  %kdma.source1497 = or disjoint i64 %122, %kdma.source1430, !dbg !13
  %kdma.source1498 = trunc nuw nsw i64 %kdma.source1497 to i32, !dbg !13
  %kdma.source1499 = shl nuw nsw i32 %kdma.source1498, 1, !dbg !13
  %kdma.offset1500 = or disjoint i32 %kdma.source1499, %kdma.intra.hi, !dbg !13
  %kdma.lds1501 = add i32 %dma.buffer.lds1232, 512, !dbg !13
  %144 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1501, i32 %kdma.offset1500, <4 x i32> %dma.buffer.rsrc1240) #9, !dbg !13
  %address.local409 = lshr i64 %address.local366, 2, !dbg !12
  %address.local412 = shl nuw nsw i64 %address.local366, 1, !dbg !12
  %address.local413 = and i64 %address.local412, 24, !dbg !12
  %address.local415 = shl i64 %address.local366, 62, !dbg !12
  %address.local416 = ashr exact i64 %address.local415, 57, !dbg !12
  %address.local417 = and i64 %address.local416, -64, !dbg !12
  %address.local420 = and i64 %address.local409, 12, !dbg !10
  %address.local422 = and i64 %address.local366, 63, !dbg !10
  %address.local423 = lshr i64 %address.local422, 5, !dbg !10
  %address.local424 = mul nuw nsw i64 %address.local423, 1016, !dbg !10
  %.idx = mul nuw nsw i64 %address.local366, 320, !dbg !12
  %145 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %.idx, !dbg !12
  %.idx1277 = mul nsw i64 %address.local409, -1280, !dbg !12
  %146 = getelementptr i8, ptr addrspace(3) %145, i64 %.idx1277, !dbg !12
  %147 = getelementptr [2 x i8], ptr addrspace(3) %146, i64 %address.local413, !dbg !12
  %148 = getelementptr [2 x i8], ptr addrspace(3) %147, i64 %address.local417, !dbg !12
  %149 = getelementptr [2 x i8], ptr addrspace(3) %148, i64 %address.local420, !dbg !12
  %address.local426 = getelementptr [2 x i8], ptr addrspace(3) %149, i64 %address.local424, !dbg !12
  %150 = add nsw i64 %.idx1277, %.idx, !dbg !12
  %.idx1 = shl nuw nsw i64 %address.local413, 1, !dbg !12
  %151 = or disjoint i64 %150, %.idx1, !dbg !12
  %152 = or disjoint i64 %address.local417, %address.local420, !dbg !12
  %153 = shl nsw i64 %152, 1, !dbg !12
  %address.local426.idx = mul nuw nsw i64 %address.local423, 2032, !dbg !12
  %154 = add nsw i64 %151, %address.local426.idx, !dbg !12
  %155 = add nsw i64 %154, %153, !dbg !12
  %156 = lshr i64 %155, 6, !dbg !12
  %157 = and i64 %156, 8, !dbg !12
  %158 = xor i64 %157, %155, !dbg !12
  %159 = sub i64 %158, %155, !dbg !12
  %kdma.swizzled = getelementptr i8, ptr addrspace(3) %address.local426, i64 %159, !dbg !12
  %160 = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8, !dbg !12
  %161 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %58, <4 x half> %160, <4 x float> zeroinitializer) #8, !dbg !12
  %kdma.gep = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4096, !dbg !12
  %162 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8, !dbg !12
  %address.local437 = shl nuw nsw i32 %address.thread365, 1, !dbg !12
  %address.local438 = and i32 %address.local437, 24, !dbg !12
  %address.local439 = shl nuw nsw i32 %address.thread365, 8, !dbg !12
  %address.local440 = and i32 %address.local439, 768, !dbg !12
  %address.local441 = or disjoint i32 %address.local438, %address.local440, !dbg !12
  %address.local442 = shl nuw nsw i32 %address.thread365, 5, !dbg !12
  %address.local443 = and i32 %address.local442, 96, !dbg !12
  %address.local444 = or disjoint i32 %address.local441, %address.local443, !dbg !12
  %address.local445 = zext nneg i32 %address.local444 to i64, !dbg !12
  %address.base = tail call i64 asm sideeffect "", "=v,0"(i64 %103) #8, !dbg !12, !frisk.register_lifetimes !5
  %address.local446 = lshr i32 %address.thread365, 1, !dbg !12
  %address.local447 = and i32 %address.local446, 1, !dbg !12
  %address.local448 = zext nneg i32 %address.local447 to i64, !dbg !12
  %address.local449 = add nsw i64 %address.base, %address.local448, !dbg !12
  %address.local450 = shl i64 %address.local449, 6, !dbg !12
  %address.local451 = sub i64 %address.local445, %address.local450, !dbg !12
  %163 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local451, !dbg !12
  %164 = getelementptr [2 x i8], ptr addrspace(3) %163, i64 %address.local420, !dbg !12
  %165 = getelementptr [2 x i8], ptr addrspace(3) %164, i64 %address.local424, !dbg !12
  %address.local455 = getelementptr i8, ptr addrspace(3) %165, i64 64, !dbg !12
  %.idx47 = add i64 %address.local451, %address.local420, !dbg !12
  %166 = shl i64 %.idx47, 1, !dbg !12
  %167 = add nuw nsw i64 %address.local426.idx, 64, !dbg !12
  %168 = add i64 %167, %166, !dbg !12
  %169 = lshr i64 %168, 6, !dbg !12
  %170 = and i64 %169, 8, !dbg !12
  %171 = xor i64 %170, %168, !dbg !12
  %172 = sub i64 %171, %168, !dbg !12
  %kdma.swizzled1634 = getelementptr i8, ptr addrspace(3) %address.local455, i64 %172, !dbg !12
  %173 = load <4 x half>, ptr addrspace(3) %kdma.swizzled1634, align 8, !dbg !12
  %174 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %162, <4 x float> %161) #8, !dbg !12
  %175 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %58, <4 x half> %173, <4 x float> zeroinitializer) #8, !dbg !12
  %kdma.gep1635 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1634, i64 4096, !dbg !12
  %176 = load <4 x half>, ptr addrspace(3) %kdma.gep1635, align 8, !dbg !12
  %kdma.gep1636 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 128, !dbg !12
  %177 = load <4 x half>, ptr addrspace(3) %kdma.gep1636, align 8, !dbg !12
  %178 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %176, <4 x float> %175) #8, !dbg !12
  %179 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %58, <4 x half> %177, <4 x float> zeroinitializer) #8, !dbg !12
  %kdma.gep1637 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4224, !dbg !12
  %180 = load <4 x half>, ptr addrspace(3) %kdma.gep1637, align 8, !dbg !12
  %kdma.gep1638 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1634, i64 128, !dbg !12
  %181 = load <4 x half>, ptr addrspace(3) %kdma.gep1638, align 8, !dbg !12
  %182 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %180, <4 x float> %179) #8, !dbg !12
  %183 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %58, <4 x half> %181, <4 x float> zeroinitializer) #8, !dbg !12
  %kdma.gep1639 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1634, i64 4224, !dbg !12
  %184 = load <4 x half>, ptr addrspace(3) %kdma.gep1639, align 8, !dbg !12
  %kdma.gep1640 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1024, !dbg !12
  %185 = load <4 x half>, ptr addrspace(3) %kdma.gep1640, align 8, !dbg !12
  %186 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %184, <4 x float> %183) #8, !dbg !12
  %187 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %58, <4 x half> %185, <4 x float> zeroinitializer) #8, !dbg !12
  %kdma.gep1641 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5120, !dbg !12
  %188 = load <4 x half>, ptr addrspace(3) %kdma.gep1641, align 8, !dbg !12
  %kdma.gep1642 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1634, i64 1024, !dbg !12
  %189 = load <4 x half>, ptr addrspace(3) %kdma.gep1642, align 8, !dbg !12
  %190 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %188, <4 x float> %187) #8, !dbg !12
  %191 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %58, <4 x half> %189, <4 x float> zeroinitializer) #8, !dbg !12
  %kdma.gep1643 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1634, i64 5120, !dbg !12
  %192 = load <4 x half>, ptr addrspace(3) %kdma.gep1643, align 8, !dbg !12
  %kdma.gep1644 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1152, !dbg !12
  %193 = load <4 x half>, ptr addrspace(3) %kdma.gep1644, align 8, !dbg !12
  %194 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %192, <4 x float> %191) #8, !dbg !12
  %195 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %58, <4 x half> %193, <4 x float> zeroinitializer) #8, !dbg !12
  %kdma.gep1645 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5248, !dbg !12
  %196 = load <4 x half>, ptr addrspace(3) %kdma.gep1645, align 8, !dbg !12
  %kdma.gep1646 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1634, i64 1152, !dbg !12
  %197 = load <4 x half>, ptr addrspace(3) %kdma.gep1646, align 8, !dbg !12
  %198 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %196, <4 x float> %195) #8, !dbg !12
  %199 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %58, <4 x half> %197, <4 x float> zeroinitializer) #8, !dbg !12
  %kdma.gep1647 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1634, i64 5248, !dbg !12
  %200 = load <4 x half>, ptr addrspace(3) %kdma.gep1647, align 8, !dbg !12
  %201 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %200, <4 x float> %199) #8, !dbg !12
  %202 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %160, <4 x float> zeroinitializer) #8, !dbg !12
  %203 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %63, <4 x half> %162, <4 x float> %202) #8, !dbg !12
  %204 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %173, <4 x float> zeroinitializer) #8, !dbg !12
  %205 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %63, <4 x half> %176, <4 x float> %204) #8, !dbg !12
  %206 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %177, <4 x float> zeroinitializer) #8, !dbg !12
  %207 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %63, <4 x half> %180, <4 x float> %206) #8, !dbg !12
  %208 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %181, <4 x float> zeroinitializer) #8, !dbg !12
  %209 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %63, <4 x half> %184, <4 x float> %208) #8, !dbg !12
  %210 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %185, <4 x float> zeroinitializer) #8, !dbg !12
  %211 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %63, <4 x half> %188, <4 x float> %210) #8, !dbg !12
  %212 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %189, <4 x float> zeroinitializer) #8, !dbg !12
  %213 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %63, <4 x half> %192, <4 x float> %212) #8, !dbg !12
  %214 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %193, <4 x float> zeroinitializer) #8, !dbg !12
  %215 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %63, <4 x half> %196, <4 x float> %214) #8, !dbg !12
  %216 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %197, <4 x float> zeroinitializer) #8, !dbg !12
  %217 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %63, <4 x half> %200, <4 x float> %216) #8, !dbg !12
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  tail call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %218 = or disjoint i64 %121, 64, !dbg !13
  %kdma.source1513 = or disjoint i64 %218, %kdma.source1382, !dbg !13
  %kdma.source1514 = trunc nuw nsw i64 %kdma.source1513 to i32, !dbg !13
  %kdma.source1515 = shl nuw nsw i32 %kdma.source1514, 1, !dbg !13
  %kdma.offset1516 = or disjoint i32 %kdma.source1515, %kdma.intra, !dbg !13
  %219 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1241, i32 %kdma.offset1516, <4 x i32> %dma.buffer.rsrc1240) #9, !dbg !13
  %kdma.source1529 = or disjoint i64 %218, %kdma.source1396, !dbg !13
  %kdma.source1530 = trunc nuw nsw i64 %kdma.source1529 to i32, !dbg !13
  %kdma.source1531 = shl nuw nsw i32 %kdma.source1530, 1, !dbg !13
  %kdma.offset1532 = or disjoint i32 %kdma.source1531, %kdma.intra.hi, !dbg !13
  %220 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1401, i32 %kdma.offset1532, <4 x i32> %dma.buffer.rsrc1240) #9, !dbg !13
  %kdma.source1546 = or disjoint i64 %218, %kdma.source1413, !dbg !13
  %kdma.source1547 = trunc nuw nsw i64 %kdma.source1546 to i32, !dbg !13
  %kdma.source1548 = shl nuw nsw i32 %kdma.source1547, 1, !dbg !13
  %kdma.offset1549 = or disjoint i32 %kdma.source1548, %kdma.intra, !dbg !13
  %221 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1238, i32 %kdma.offset1549, <4 x i32> %dma.buffer.rsrc1240) #9, !dbg !13
  %kdma.source1563 = or disjoint i64 %218, %kdma.source1430, !dbg !13
  %kdma.source1564 = trunc nuw nsw i64 %kdma.source1563 to i32, !dbg !13
  %kdma.source1565 = shl nuw nsw i32 %kdma.source1564, 1, !dbg !13
  %kdma.offset1566 = or disjoint i32 %kdma.source1565, %kdma.intra.hi, !dbg !13
  %222 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1435, i32 %kdma.offset1566, <4 x i32> %dma.buffer.rsrc1240) #9, !dbg !13
  %kdma.gep1664 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8192, !dbg !12
  %223 = load <4 x half>, ptr addrspace(3) %kdma.gep1664, align 8, !dbg !12
  %224 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %223, <4 x float> %174) #8, !dbg !12
  %kdma.gep1665 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12288, !dbg !12
  %225 = load <4 x half>, ptr addrspace(3) %kdma.gep1665, align 8, !dbg !12
  %kdma.gep1666 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1634, i64 8192, !dbg !12
  %226 = load <4 x half>, ptr addrspace(3) %kdma.gep1666, align 8, !dbg !12
  %227 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %225, <4 x float> %224) #8, !dbg !12
  %228 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %226, <4 x float> %178) #8, !dbg !12
  %kdma.gep1667 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1634, i64 12288, !dbg !12
  %229 = load <4 x half>, ptr addrspace(3) %kdma.gep1667, align 8, !dbg !12
  %kdma.gep1668 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8320, !dbg !12
  %230 = load <4 x half>, ptr addrspace(3) %kdma.gep1668, align 8, !dbg !12
  %231 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %229, <4 x float> %228) #8, !dbg !12
  %232 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %230, <4 x float> %182) #8, !dbg !12
  %kdma.gep1669 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12416, !dbg !12
  %233 = load <4 x half>, ptr addrspace(3) %kdma.gep1669, align 8, !dbg !12
  %kdma.gep1670 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1634, i64 8320, !dbg !12
  %234 = load <4 x half>, ptr addrspace(3) %kdma.gep1670, align 8, !dbg !12
  %235 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %233, <4 x float> %232) #8, !dbg !12
  %236 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %234, <4 x float> %186) #8, !dbg !12
  %kdma.gep1671 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1634, i64 12416, !dbg !12
  %237 = load <4 x half>, ptr addrspace(3) %kdma.gep1671, align 8, !dbg !12
  %kdma.gep1672 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9216, !dbg !12
  %238 = load <4 x half>, ptr addrspace(3) %kdma.gep1672, align 8, !dbg !12
  %239 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %237, <4 x float> %236) #8, !dbg !12
  %240 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %238, <4 x float> %190) #8, !dbg !12
  %kdma.gep1673 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13312, !dbg !12
  %241 = load <4 x half>, ptr addrspace(3) %kdma.gep1673, align 8, !dbg !12
  %kdma.gep1674 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1634, i64 9216, !dbg !12
  %242 = load <4 x half>, ptr addrspace(3) %kdma.gep1674, align 8, !dbg !12
  %243 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %241, <4 x float> %240) #8, !dbg !12
  %244 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %242, <4 x float> %194) #8, !dbg !12
  %kdma.gep1675 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1634, i64 13312, !dbg !12
  %245 = load <4 x half>, ptr addrspace(3) %kdma.gep1675, align 8, !dbg !12
  %kdma.gep1676 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9344, !dbg !12
  %246 = load <4 x half>, ptr addrspace(3) %kdma.gep1676, align 8, !dbg !12
  %247 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %245, <4 x float> %244) #8, !dbg !12
  %248 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %246, <4 x float> %198) #8, !dbg !12
  %kdma.gep1677 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13440, !dbg !12
  %249 = load <4 x half>, ptr addrspace(3) %kdma.gep1677, align 8, !dbg !12
  %kdma.gep1678 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1634, i64 9344, !dbg !12
  %250 = load <4 x half>, ptr addrspace(3) %kdma.gep1678, align 8, !dbg !12
  %251 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %249, <4 x float> %248) #8, !dbg !12
  %252 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %250, <4 x float> %201) #8, !dbg !12
  %kdma.gep1679 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1634, i64 13440, !dbg !12
  %253 = load <4 x half>, ptr addrspace(3) %kdma.gep1679, align 8, !dbg !12
  %254 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %253, <4 x float> %252) #8, !dbg !12
  %255 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %223, <4 x float> %203) #8, !dbg !12
  %256 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %76, <4 x half> %225, <4 x float> %255) #8, !dbg !12
  %257 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %226, <4 x float> %205) #8, !dbg !12
  %258 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %76, <4 x half> %229, <4 x float> %257) #8, !dbg !12
  %259 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %230, <4 x float> %207) #8, !dbg !12
  %260 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %76, <4 x half> %233, <4 x float> %259) #8, !dbg !12
  %261 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %234, <4 x float> %209) #8, !dbg !12
  %262 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %76, <4 x half> %237, <4 x float> %261) #8, !dbg !12
  %263 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %238, <4 x float> %211) #8, !dbg !12
  %264 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %76, <4 x half> %241, <4 x float> %263) #8, !dbg !12
  %265 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %242, <4 x float> %213) #8, !dbg !12
  %266 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %76, <4 x half> %245, <4 x float> %265) #8, !dbg !12
  %267 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %246, <4 x float> %215) #8, !dbg !12
  %268 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %76, <4 x half> %249, <4 x float> %267) #8, !dbg !12
  %269 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %250, <4 x float> %217) #8, !dbg !12
  %270 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %76, <4 x half> %253, <4 x float> %269) #8, !dbg !12
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  tail call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %271 = or disjoint i64 %121, 96, !dbg !13
  %kdma.source1579 = or disjoint i64 %271, %kdma.source1382, !dbg !13
  %kdma.source1580 = trunc nuw nsw i64 %kdma.source1579 to i32, !dbg !13
  %kdma.source1581 = shl nuw nsw i32 %kdma.source1580, 1, !dbg !13
  %kdma.offset1582 = or disjoint i32 %kdma.source1581, %kdma.intra, !dbg !13
  %272 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1235, i32 %kdma.offset1582, <4 x i32> %dma.buffer.rsrc1240) #9, !dbg !13
  %kdma.source1595 = or disjoint i64 %271, %kdma.source1396, !dbg !13
  %kdma.source1596 = trunc nuw nsw i64 %kdma.source1595 to i32, !dbg !13
  %kdma.source1597 = shl nuw nsw i32 %kdma.source1596, 1, !dbg !13
  %kdma.offset1598 = or disjoint i32 %kdma.source1597, %kdma.intra.hi, !dbg !13
  %273 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1467, i32 %kdma.offset1598, <4 x i32> %dma.buffer.rsrc1240) #9, !dbg !13
  %kdma.source1612 = or disjoint i64 %271, %kdma.source1413, !dbg !13
  %kdma.source1613 = trunc nuw nsw i64 %kdma.source1612 to i32, !dbg !13
  %kdma.source1614 = shl nuw nsw i32 %kdma.source1613, 1, !dbg !13
  %kdma.offset1615 = or disjoint i32 %kdma.source1614, %kdma.intra, !dbg !13
  %274 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1232, i32 %kdma.offset1615, <4 x i32> %dma.buffer.rsrc1240) #9, !dbg !13
  %kdma.source1629 = or disjoint i64 %271, %kdma.source1430, !dbg !13
  %kdma.source1630 = trunc nuw nsw i64 %kdma.source1629 to i32, !dbg !13
  %kdma.source1631 = shl nuw nsw i32 %kdma.source1630, 1, !dbg !13
  %kdma.offset1632 = or disjoint i32 %kdma.source1631, %kdma.intra.hi, !dbg !13
  %275 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1501, i32 %kdma.offset1632, <4 x i32> %dma.buffer.rsrc1240) #9, !dbg !13
  %276 = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8, !dbg !12
  %277 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %276, <4 x float> %227) #8, !dbg !12
  %278 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8, !dbg !12
  %279 = load <4 x half>, ptr addrspace(3) %kdma.swizzled1634, align 8, !dbg !12
  %280 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %278, <4 x float> %277) #8, !dbg !12
  %281 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %279, <4 x float> %231) #8, !dbg !12
  %282 = load <4 x half>, ptr addrspace(3) %kdma.gep1635, align 8, !dbg !12
  %283 = load <4 x half>, ptr addrspace(3) %kdma.gep1636, align 8, !dbg !12
  %284 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %282, <4 x float> %281) #8, !dbg !12
  %285 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %283, <4 x float> %235) #8, !dbg !12
  %286 = load <4 x half>, ptr addrspace(3) %kdma.gep1637, align 8, !dbg !12
  %287 = load <4 x half>, ptr addrspace(3) %kdma.gep1638, align 8, !dbg !12
  %288 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %286, <4 x float> %285) #8, !dbg !12
  %289 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %287, <4 x float> %239) #8, !dbg !12
  %290 = load <4 x half>, ptr addrspace(3) %kdma.gep1639, align 8, !dbg !12
  %291 = load <4 x half>, ptr addrspace(3) %kdma.gep1640, align 8, !dbg !12
  %292 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %290, <4 x float> %289) #8, !dbg !12
  %293 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %291, <4 x float> %243) #8, !dbg !12
  %294 = load <4 x half>, ptr addrspace(3) %kdma.gep1641, align 8, !dbg !12
  %295 = load <4 x half>, ptr addrspace(3) %kdma.gep1642, align 8, !dbg !12
  %296 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %294, <4 x float> %293) #8, !dbg !12
  %297 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %295, <4 x float> %247) #8, !dbg !12
  %298 = load <4 x half>, ptr addrspace(3) %kdma.gep1643, align 8, !dbg !12
  %299 = load <4 x half>, ptr addrspace(3) %kdma.gep1644, align 8, !dbg !12
  %300 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %298, <4 x float> %297) #8, !dbg !12
  %301 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %299, <4 x float> %251) #8, !dbg !12
  %302 = load <4 x half>, ptr addrspace(3) %kdma.gep1645, align 8, !dbg !12
  %303 = load <4 x half>, ptr addrspace(3) %kdma.gep1646, align 8, !dbg !12
  %304 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %302, <4 x float> %301) #8, !dbg !12
  %305 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %303, <4 x float> %254) #8, !dbg !12
  %306 = load <4 x half>, ptr addrspace(3) %kdma.gep1647, align 8, !dbg !12
  %307 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %306, <4 x float> %305) #8, !dbg !12
  %308 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %276, <4 x float> %256) #8, !dbg !12
  %309 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %278, <4 x float> %308) #8, !dbg !12
  %310 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %279, <4 x float> %258) #8, !dbg !12
  %311 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %282, <4 x float> %310) #8, !dbg !12
  %312 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %283, <4 x float> %260) #8, !dbg !12
  %313 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %286, <4 x float> %312) #8, !dbg !12
  %314 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %287, <4 x float> %262) #8, !dbg !12
  %315 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %290, <4 x float> %314) #8, !dbg !12
  %316 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %291, <4 x float> %264) #8, !dbg !12
  %317 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %294, <4 x float> %316) #8, !dbg !12
  %318 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %295, <4 x float> %266) #8, !dbg !12
  %319 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %298, <4 x float> %318) #8, !dbg !12
  %320 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %299, <4 x float> %268) #8, !dbg !12
  %321 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %302, <4 x float> %320) #8, !dbg !12
  %322 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %303, <4 x float> %270) #8, !dbg !12
  %323 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %306, <4 x float> %322) #8, !dbg !12
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  tail call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %address.local784 = shl nuw nsw i64 %address.local366, 5, !dbg !14
  %address.local785 = and i64 %address.local784, 8064, !dbg !14
  %address.local787 = and i64 %address.local367, 24, !dbg !14
  %address.local788 = or disjoint i64 %address.local785, %address.local787, !dbg !14
  %.idx12861287 = or disjoint i64 %address.local788, %121, !dbg !14
  %.idx12861287.tr = trunc nuw nsw i64 %.idx12861287 to i32, !dbg !14
  %dma.buffer.offset1215 = shl nuw nsw i32 %.idx12861287.tr, 1, !dbg !14
  %324 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1241, i32 %dma.buffer.offset1215, <4 x i32> %dma.buffer.rsrc1216) #9, !dbg !14
  %address.local790 = shl nuw nsw i64 %address.local383, 5, !dbg !14
  %address.local791 = and i64 %address.local790, 16256, !dbg !14
  %address.local794 = or disjoint i64 %address.local791, %address.local787, !dbg !14
  %.idx12881289 = or disjoint i64 %address.local794, %121, !dbg !14
  %.idx12881289.tr = trunc nuw nsw i64 %.idx12881289 to i32, !dbg !14
  %dma.buffer.offset1212 = shl nuw nsw i32 %.idx12881289.tr, 1, !dbg !14
  %325 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1238, i32 %dma.buffer.offset1212, <4 x i32> %dma.buffer.rsrc1216) #9, !dbg !14
  %326 = load <4 x half>, ptr addrspace(3) %kdma.gep1664, align 8, !dbg !12
  %327 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %326, <4 x float> %280) #8, !dbg !12
  %328 = load <4 x half>, ptr addrspace(3) %kdma.gep1665, align 8, !dbg !12
  %329 = load <4 x half>, ptr addrspace(3) %kdma.gep1666, align 8, !dbg !12
  %330 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %328, <4 x float> %327) #8, !dbg !12
  %331 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %329, <4 x float> %284) #8, !dbg !12
  %332 = extractelement <4 x float> %330, i64 0, !dbg !15
  %333 = tail call float @llvm.maxnum.f32(float %332, float -inf), !dbg !16
  %334 = extractelement <4 x float> %330, i64 1, !dbg !15
  %335 = tail call float @llvm.maxnum.f32(float %333, float %334), !dbg !16
  %336 = extractelement <4 x float> %330, i64 2, !dbg !15
  %337 = tail call float @llvm.maxnum.f32(float %335, float %336), !dbg !16
  %338 = extractelement <4 x float> %330, i64 3, !dbg !15
  %339 = tail call float @llvm.maxnum.f32(float %337, float %338), !dbg !16
  %340 = load <4 x half>, ptr addrspace(3) %kdma.gep1667, align 8, !dbg !12
  %341 = load <4 x half>, ptr addrspace(3) %kdma.gep1668, align 8, !dbg !12
  %342 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %340, <4 x float> %331) #8, !dbg !12
  %343 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %341, <4 x float> %288) #8, !dbg !12
  %344 = extractelement <4 x float> %342, i64 0, !dbg !15
  %345 = tail call float @llvm.maxnum.f32(float %339, float %344), !dbg !16
  %346 = extractelement <4 x float> %342, i64 1, !dbg !15
  %347 = tail call float @llvm.maxnum.f32(float %345, float %346), !dbg !16
  %348 = extractelement <4 x float> %342, i64 2, !dbg !15
  %349 = tail call float @llvm.maxnum.f32(float %347, float %348), !dbg !16
  %350 = extractelement <4 x float> %342, i64 3, !dbg !15
  %351 = tail call float @llvm.maxnum.f32(float %349, float %350), !dbg !16
  %352 = load <4 x half>, ptr addrspace(3) %kdma.gep1669, align 8, !dbg !12
  %353 = load <4 x half>, ptr addrspace(3) %kdma.gep1670, align 8, !dbg !12
  %354 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %352, <4 x float> %343) #8, !dbg !12
  %355 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %353, <4 x float> %292) #8, !dbg !12
  %356 = extractelement <4 x float> %354, i64 0, !dbg !15
  %357 = tail call float @llvm.maxnum.f32(float %351, float %356), !dbg !16
  %358 = extractelement <4 x float> %354, i64 1, !dbg !15
  %359 = tail call float @llvm.maxnum.f32(float %357, float %358), !dbg !16
  %360 = extractelement <4 x float> %354, i64 2, !dbg !15
  %361 = tail call float @llvm.maxnum.f32(float %359, float %360), !dbg !16
  %362 = extractelement <4 x float> %354, i64 3, !dbg !15
  %363 = tail call float @llvm.maxnum.f32(float %361, float %362), !dbg !16
  %364 = load <4 x half>, ptr addrspace(3) %kdma.gep1671, align 8, !dbg !12
  %365 = load <4 x half>, ptr addrspace(3) %kdma.gep1672, align 8, !dbg !12
  %366 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %364, <4 x float> %355) #8, !dbg !12
  %367 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %365, <4 x float> %296) #8, !dbg !12
  %368 = extractelement <4 x float> %366, i64 0, !dbg !15
  %369 = tail call float @llvm.maxnum.f32(float %363, float %368), !dbg !16
  %370 = extractelement <4 x float> %366, i64 1, !dbg !15
  %371 = tail call float @llvm.maxnum.f32(float %369, float %370), !dbg !16
  %372 = extractelement <4 x float> %366, i64 2, !dbg !15
  %373 = tail call float @llvm.maxnum.f32(float %371, float %372), !dbg !16
  %374 = extractelement <4 x float> %366, i64 3, !dbg !15
  %375 = tail call float @llvm.maxnum.f32(float %373, float %374), !dbg !16
  %376 = load <4 x half>, ptr addrspace(3) %kdma.gep1673, align 8, !dbg !12
  %377 = load <4 x half>, ptr addrspace(3) %kdma.gep1674, align 8, !dbg !12
  %378 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %376, <4 x float> %367) #8, !dbg !12
  %379 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %377, <4 x float> %300) #8, !dbg !12
  %380 = extractelement <4 x float> %378, i64 0, !dbg !15
  %381 = tail call float @llvm.maxnum.f32(float %375, float %380), !dbg !16
  %382 = extractelement <4 x float> %378, i64 1, !dbg !15
  %383 = tail call float @llvm.maxnum.f32(float %381, float %382), !dbg !16
  %384 = extractelement <4 x float> %378, i64 2, !dbg !15
  %385 = tail call float @llvm.maxnum.f32(float %383, float %384), !dbg !16
  %386 = extractelement <4 x float> %378, i64 3, !dbg !15
  %387 = tail call float @llvm.maxnum.f32(float %385, float %386), !dbg !16
  %388 = load <4 x half>, ptr addrspace(3) %kdma.gep1675, align 8, !dbg !12
  %389 = load <4 x half>, ptr addrspace(3) %kdma.gep1676, align 8, !dbg !12
  %390 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %388, <4 x float> %379) #8, !dbg !12
  %391 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %389, <4 x float> %304) #8, !dbg !12
  %392 = extractelement <4 x float> %390, i64 0, !dbg !15
  %393 = tail call float @llvm.maxnum.f32(float %387, float %392), !dbg !16
  %394 = extractelement <4 x float> %390, i64 1, !dbg !15
  %395 = tail call float @llvm.maxnum.f32(float %393, float %394), !dbg !16
  %396 = extractelement <4 x float> %390, i64 2, !dbg !15
  %397 = tail call float @llvm.maxnum.f32(float %395, float %396), !dbg !16
  %398 = extractelement <4 x float> %390, i64 3, !dbg !15
  %399 = tail call float @llvm.maxnum.f32(float %397, float %398), !dbg !16
  %400 = load <4 x half>, ptr addrspace(3) %kdma.gep1677, align 8, !dbg !12
  %401 = load <4 x half>, ptr addrspace(3) %kdma.gep1678, align 8, !dbg !12
  %402 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %400, <4 x float> %391) #8, !dbg !12
  %403 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %401, <4 x float> %307) #8, !dbg !12
  %404 = extractelement <4 x float> %402, i64 0, !dbg !15
  %405 = tail call float @llvm.maxnum.f32(float %399, float %404), !dbg !16
  %406 = extractelement <4 x float> %402, i64 1, !dbg !15
  %407 = tail call float @llvm.maxnum.f32(float %405, float %406), !dbg !16
  %408 = extractelement <4 x float> %402, i64 2, !dbg !15
  %409 = tail call float @llvm.maxnum.f32(float %407, float %408), !dbg !16
  %410 = extractelement <4 x float> %402, i64 3, !dbg !15
  %411 = tail call float @llvm.maxnum.f32(float %409, float %410), !dbg !16
  %412 = load <4 x half>, ptr addrspace(3) %kdma.gep1679, align 8, !dbg !12
  %413 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %412, <4 x float> %403) #8, !dbg !12
  %414 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %326, <4 x float> %309) #8, !dbg !12
  %415 = extractelement <4 x float> %413, i64 0, !dbg !15
  %416 = tail call float @llvm.maxnum.f32(float %411, float %415), !dbg !16
  %417 = extractelement <4 x float> %413, i64 1, !dbg !15
  %418 = tail call float @llvm.maxnum.f32(float %416, float %417), !dbg !16
  %419 = extractelement <4 x float> %413, i64 2, !dbg !15
  %420 = tail call float @llvm.maxnum.f32(float %418, float %419), !dbg !16
  %421 = extractelement <4 x float> %413, i64 3, !dbg !15
  %422 = tail call float @llvm.maxnum.f32(float %420, float %421), !dbg !16
  %423 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %328, <4 x float> %414) #8, !dbg !12
  %424 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %329, <4 x float> %311) #8, !dbg !12
  %425 = extractelement <4 x float> %423, i64 0, !dbg !15
  %426 = extractelement <4 x float> %423, i64 1, !dbg !15
  %427 = extractelement <4 x float> %423, i64 2, !dbg !15
  %428 = extractelement <4 x float> %423, i64 3, !dbg !15
  %429 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %340, <4 x float> %424) #8, !dbg !12
  %430 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %341, <4 x float> %313) #8, !dbg !12
  %431 = extractelement <4 x float> %429, i64 0, !dbg !15
  %432 = extractelement <4 x float> %429, i64 1, !dbg !15
  %433 = extractelement <4 x float> %429, i64 2, !dbg !15
  %434 = extractelement <4 x float> %429, i64 3, !dbg !15
  %435 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %352, <4 x float> %430) #8, !dbg !12
  %436 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %353, <4 x float> %315) #8, !dbg !12
  %437 = extractelement <4 x float> %435, i64 0, !dbg !15
  %438 = extractelement <4 x float> %435, i64 1, !dbg !15
  %439 = extractelement <4 x float> %435, i64 2, !dbg !15
  %440 = extractelement <4 x float> %435, i64 3, !dbg !15
  %441 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %364, <4 x float> %436) #8, !dbg !12
  %442 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %365, <4 x float> %317) #8, !dbg !12
  %443 = extractelement <4 x float> %441, i64 0, !dbg !15
  %444 = extractelement <4 x float> %441, i64 1, !dbg !15
  %445 = extractelement <4 x float> %441, i64 2, !dbg !15
  %446 = extractelement <4 x float> %441, i64 3, !dbg !15
  %447 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %376, <4 x float> %442) #8, !dbg !12
  %448 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %377, <4 x float> %319) #8, !dbg !12
  %449 = extractelement <4 x float> %447, i64 0, !dbg !15
  %450 = extractelement <4 x float> %447, i64 1, !dbg !15
  %451 = extractelement <4 x float> %447, i64 2, !dbg !15
  %452 = extractelement <4 x float> %447, i64 3, !dbg !15
  %453 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %388, <4 x float> %448) #8, !dbg !12
  %454 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %389, <4 x float> %321) #8, !dbg !12
  %455 = extractelement <4 x float> %453, i64 0, !dbg !15
  %456 = extractelement <4 x float> %453, i64 1, !dbg !15
  %457 = extractelement <4 x float> %453, i64 2, !dbg !15
  %458 = extractelement <4 x float> %453, i64 3, !dbg !15
  %459 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %400, <4 x float> %454) #8, !dbg !12
  %460 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %401, <4 x float> %323) #8, !dbg !12
  %461 = extractelement <4 x float> %459, i64 0, !dbg !15
  %462 = extractelement <4 x float> %459, i64 1, !dbg !15
  %463 = extractelement <4 x float> %459, i64 2, !dbg !15
  %464 = extractelement <4 x float> %459, i64 3, !dbg !15
  %465 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %412, <4 x float> %460) #8, !dbg !12
  %466 = extractelement <4 x float> %465, i64 0, !dbg !15
  %467 = extractelement <4 x float> %465, i64 1, !dbg !15
  %468 = extractelement <4 x float> %465, i64 2, !dbg !15
  %469 = extractelement <4 x float> %465, i64 3, !dbg !15
  %470 = tail call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !16
  %471 = tail call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %470), !dbg !16
  %472 = add nuw nsw i32 %471, 64, !dbg !16
  %473 = xor i32 %471, 16, !dbg !16
  %474 = and i32 %472, 192, !dbg !16
  %475 = icmp samesign ult i32 %473, %474, !dbg !16
  %476 = select i1 %475, i32 %473, i32 %471, !dbg !16
  %477 = shl nuw nsw i32 %476, 2, !dbg !16
  %478 = bitcast float %422 to i32, !dbg !16
  %479 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %477, i32 %478), !dbg !16
  %480 = bitcast i32 %479 to float, !dbg !16
  %481 = tail call float @llvm.maxnum.f32(float %422, float %480), !dbg !16
  %482 = xor i32 %471, 32, !dbg !16
  %483 = icmp samesign ult i32 %482, %474, !dbg !16
  %484 = select i1 %483, i32 %482, i32 %471, !dbg !16
  %485 = shl nuw nsw i32 %484, 2, !dbg !16
  %486 = bitcast float %481 to i32, !dbg !16
  %487 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %485, i32 %486), !dbg !16
  %488 = bitcast i32 %487 to float, !dbg !16
  %489 = tail call float @llvm.maxnum.f32(float %481, float %488), !dbg !16
  %490 = extractvalue [2 x <1 x float>] %116, 0, !dbg !16
  %491 = insertelement <1 x float> poison, float %489, i64 0, !dbg !16
  %492 = tail call <1 x float> @llvm.maxnum.v1f32(<1 x float> %491, <1 x float> %490), !dbg !16
  %493 = tail call float @llvm.maxnum.f32(float %425, float -inf), !dbg !16
  %494 = tail call float @llvm.maxnum.f32(float %493, float %426), !dbg !16
  %495 = tail call float @llvm.maxnum.f32(float %494, float %427), !dbg !16
  %496 = tail call float @llvm.maxnum.f32(float %495, float %428), !dbg !16
  %497 = tail call float @llvm.maxnum.f32(float %496, float %431), !dbg !16
  %498 = tail call float @llvm.maxnum.f32(float %497, float %432), !dbg !16
  %499 = tail call float @llvm.maxnum.f32(float %498, float %433), !dbg !16
  %500 = tail call float @llvm.maxnum.f32(float %499, float %434), !dbg !16
  %501 = tail call float @llvm.maxnum.f32(float %500, float %437), !dbg !16
  %502 = tail call float @llvm.maxnum.f32(float %501, float %438), !dbg !16
  %503 = tail call float @llvm.maxnum.f32(float %502, float %439), !dbg !16
  %504 = tail call float @llvm.maxnum.f32(float %503, float %440), !dbg !16
  %505 = tail call float @llvm.maxnum.f32(float %504, float %443), !dbg !16
  %506 = tail call float @llvm.maxnum.f32(float %505, float %444), !dbg !16
  %507 = tail call float @llvm.maxnum.f32(float %506, float %445), !dbg !16
  %508 = tail call float @llvm.maxnum.f32(float %507, float %446), !dbg !16
  %509 = tail call float @llvm.maxnum.f32(float %508, float %449), !dbg !16
  %510 = tail call float @llvm.maxnum.f32(float %509, float %450), !dbg !16
  %511 = tail call float @llvm.maxnum.f32(float %510, float %451), !dbg !16
  %512 = tail call float @llvm.maxnum.f32(float %511, float %452), !dbg !16
  %513 = tail call float @llvm.maxnum.f32(float %512, float %455), !dbg !16
  %514 = tail call float @llvm.maxnum.f32(float %513, float %456), !dbg !16
  %515 = tail call float @llvm.maxnum.f32(float %514, float %457), !dbg !16
  %516 = tail call float @llvm.maxnum.f32(float %515, float %458), !dbg !16
  %517 = tail call float @llvm.maxnum.f32(float %516, float %461), !dbg !16
  %518 = tail call float @llvm.maxnum.f32(float %517, float %462), !dbg !16
  %519 = tail call float @llvm.maxnum.f32(float %518, float %463), !dbg !16
  %520 = tail call float @llvm.maxnum.f32(float %519, float %464), !dbg !16
  %521 = tail call float @llvm.maxnum.f32(float %520, float %466), !dbg !16
  %522 = tail call float @llvm.maxnum.f32(float %521, float %467), !dbg !16
  %523 = tail call float @llvm.maxnum.f32(float %522, float %468), !dbg !16
  %524 = tail call float @llvm.maxnum.f32(float %523, float %469), !dbg !16
  %525 = bitcast float %524 to i32, !dbg !16
  %526 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %477, i32 %525), !dbg !16
  %527 = bitcast i32 %526 to float, !dbg !16
  %528 = tail call float @llvm.maxnum.f32(float %524, float %527), !dbg !16
  %529 = bitcast float %528 to i32, !dbg !16
  %530 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %485, i32 %529), !dbg !16
  %531 = bitcast i32 %530 to float, !dbg !16
  %532 = tail call float @llvm.maxnum.f32(float %528, float %531), !dbg !16
  %533 = extractvalue [2 x <1 x float>] %116, 1, !dbg !16
  %534 = insertelement <1 x float> poison, float %532, i64 0, !dbg !16
  %535 = tail call <1 x float> @llvm.maxnum.v1f32(<1 x float> %534, <1 x float> %533), !dbg !16
  %536 = shufflevector <1 x float> %492, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !17
  %537 = fsub <4 x float> %330, %536, !dbg !17
  %538 = fsub <4 x float> %342, %536, !dbg !17
  %539 = fsub <4 x float> %354, %536, !dbg !17
  %540 = fsub <4 x float> %366, %536, !dbg !17
  %541 = fsub <4 x float> %378, %536, !dbg !17
  %542 = fsub <4 x float> %390, %536, !dbg !17
  %543 = fsub <4 x float> %402, %536, !dbg !17
  %544 = fsub <4 x float> %413, %536, !dbg !17
  %545 = shufflevector <1 x float> %535, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !17
  %546 = fsub <4 x float> %423, %545, !dbg !17
  %547 = fsub <4 x float> %429, %545, !dbg !17
  %548 = fsub <4 x float> %435, %545, !dbg !17
  %549 = fsub <4 x float> %441, %545, !dbg !17
  %550 = fsub <4 x float> %447, %545, !dbg !17
  %551 = fsub <4 x float> %453, %545, !dbg !17
  %552 = fsub <4 x float> %459, %545, !dbg !17
  %553 = fsub <4 x float> %465, %545, !dbg !17
  %554 = fsub <1 x float> %490, %492, !dbg !18
  %555 = extractelement <1 x float> %554, i64 0, !dbg !19
  %556 = tail call float @llvm.amdgcn.exp2.f32(float %555), !dbg !19
  %557 = insertelement <4 x float> poison, float %556, i64 0, !dbg !20
  %558 = shufflevector <4 x float> %557, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !20
  %559 = extractvalue [2 x <1 x float>] %117, 0, !dbg !21
  %560 = extractelement <1 x float> %559, i64 0, !dbg !21
  %561 = fmul float %560, %556, !dbg !21
  %562 = fsub <1 x float> %533, %535, !dbg !18
  %563 = extractelement <1 x float> %562, i64 0, !dbg !19
  %564 = tail call float @llvm.amdgcn.exp2.f32(float %563), !dbg !19
  %565 = insertelement <4 x float> poison, float %564, i64 0, !dbg !20
  %566 = shufflevector <4 x float> %565, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !20
  %567 = extractvalue [2 x <1 x float>] %117, 1, !dbg !21
  %568 = extractelement <1 x float> %567, i64 0, !dbg !21
  %569 = fmul float %568, %564, !dbg !21
  %570 = extractelement <4 x float> %537, i64 0, !dbg !22
  %571 = tail call float @llvm.amdgcn.exp2.f32(float %570), !dbg !22
  %572 = insertelement <4 x float> poison, float %571, i64 0, !dbg !23
  %573 = extractelement <4 x float> %537, i64 1, !dbg !22
  %574 = tail call float @llvm.amdgcn.exp2.f32(float %573), !dbg !22
  %575 = insertelement <4 x float> %572, float %574, i64 1, !dbg !23
  %576 = fadd float %571, %574, !dbg !24
  %577 = extractelement <4 x float> %537, i64 2, !dbg !22
  %578 = tail call float @llvm.amdgcn.exp2.f32(float %577), !dbg !22
  %579 = insertelement <4 x float> %575, float %578, i64 2, !dbg !23
  %580 = fadd float %578, %576, !dbg !24
  %581 = extractelement <4 x float> %537, i64 3, !dbg !22
  %582 = tail call float @llvm.amdgcn.exp2.f32(float %581), !dbg !22
  %583 = insertelement <4 x float> %579, float %582, i64 3, !dbg !23
  %math.result = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %583) #8, !dbg !23, !frisk.register_lifetimes !5
  %584 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !23
  %math.packed = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %584) #8, !dbg !24, !frisk.register_lifetimes !5
  %585 = fadd float %582, %580, !dbg !24
  %586 = extractelement <4 x float> %538, i64 0, !dbg !22
  %587 = tail call float @llvm.amdgcn.exp2.f32(float %586), !dbg !22
  %588 = insertelement <4 x float> poison, float %587, i64 0, !dbg !23
  %589 = fadd float %587, %585, !dbg !24
  %590 = extractelement <4 x float> %538, i64 1, !dbg !22
  %591 = tail call float @llvm.amdgcn.exp2.f32(float %590), !dbg !22
  %592 = insertelement <4 x float> %588, float %591, i64 1, !dbg !23
  %593 = fadd float %591, %589, !dbg !24
  %594 = extractelement <4 x float> %538, i64 2, !dbg !22
  %595 = tail call float @llvm.amdgcn.exp2.f32(float %594), !dbg !22
  %596 = insertelement <4 x float> %592, float %595, i64 2, !dbg !23
  %597 = fadd float %595, %593, !dbg !24
  %598 = extractelement <4 x float> %538, i64 3, !dbg !22
  %599 = tail call float @llvm.amdgcn.exp2.f32(float %598), !dbg !22
  %600 = insertelement <4 x float> %596, float %599, i64 3, !dbg !23
  %math.result241 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %600) #8, !dbg !23, !frisk.register_lifetimes !5
  %601 = fptrunc <4 x float> %math.result241 to <4 x half>, !dbg !23
  %math.packed242 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %601) #8, !dbg !24, !frisk.register_lifetimes !5
  %602 = fadd float %599, %597, !dbg !24
  %603 = extractelement <4 x float> %539, i64 0, !dbg !22
  %604 = tail call float @llvm.amdgcn.exp2.f32(float %603), !dbg !22
  %605 = insertelement <4 x float> poison, float %604, i64 0, !dbg !23
  %606 = fadd float %604, %602, !dbg !24
  %607 = extractelement <4 x float> %539, i64 1, !dbg !22
  %608 = tail call float @llvm.amdgcn.exp2.f32(float %607), !dbg !22
  %609 = insertelement <4 x float> %605, float %608, i64 1, !dbg !23
  %610 = fadd float %608, %606, !dbg !24
  %611 = extractelement <4 x float> %539, i64 2, !dbg !22
  %612 = tail call float @llvm.amdgcn.exp2.f32(float %611), !dbg !22
  %613 = insertelement <4 x float> %609, float %612, i64 2, !dbg !23
  %614 = fadd float %612, %610, !dbg !24
  %615 = extractelement <4 x float> %539, i64 3, !dbg !22
  %616 = tail call float @llvm.amdgcn.exp2.f32(float %615), !dbg !22
  %617 = insertelement <4 x float> %613, float %616, i64 3, !dbg !23
  %math.result243 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %617) #8, !dbg !23, !frisk.register_lifetimes !5
  %618 = fptrunc <4 x float> %math.result243 to <4 x half>, !dbg !23
  %math.packed244 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %618) #8, !dbg !24, !frisk.register_lifetimes !5
  %619 = fadd float %616, %614, !dbg !24
  %620 = extractelement <4 x float> %540, i64 0, !dbg !22
  %621 = tail call float @llvm.amdgcn.exp2.f32(float %620), !dbg !22
  %622 = insertelement <4 x float> poison, float %621, i64 0, !dbg !23
  %623 = fadd float %621, %619, !dbg !24
  %624 = extractelement <4 x float> %540, i64 1, !dbg !22
  %625 = tail call float @llvm.amdgcn.exp2.f32(float %624), !dbg !22
  %626 = insertelement <4 x float> %622, float %625, i64 1, !dbg !23
  %627 = fadd float %625, %623, !dbg !24
  %628 = extractelement <4 x float> %540, i64 2, !dbg !22
  %629 = tail call float @llvm.amdgcn.exp2.f32(float %628), !dbg !22
  %630 = insertelement <4 x float> %626, float %629, i64 2, !dbg !23
  %631 = fadd float %629, %627, !dbg !24
  %632 = extractelement <4 x float> %540, i64 3, !dbg !22
  %633 = tail call float @llvm.amdgcn.exp2.f32(float %632), !dbg !22
  %634 = insertelement <4 x float> %630, float %633, i64 3, !dbg !23
  %math.result245 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %634) #8, !dbg !23, !frisk.register_lifetimes !5
  %635 = fptrunc <4 x float> %math.result245 to <4 x half>, !dbg !23
  %math.packed246 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %635) #8, !dbg !24, !frisk.register_lifetimes !5
  %636 = fadd float %633, %631, !dbg !24
  %637 = extractelement <4 x float> %541, i64 0, !dbg !22
  %638 = tail call float @llvm.amdgcn.exp2.f32(float %637), !dbg !22
  %639 = insertelement <4 x float> poison, float %638, i64 0, !dbg !23
  %640 = fadd float %638, %636, !dbg !24
  %641 = extractelement <4 x float> %541, i64 1, !dbg !22
  %642 = tail call float @llvm.amdgcn.exp2.f32(float %641), !dbg !22
  %643 = insertelement <4 x float> %639, float %642, i64 1, !dbg !23
  %644 = fadd float %642, %640, !dbg !24
  %645 = extractelement <4 x float> %541, i64 2, !dbg !22
  %646 = tail call float @llvm.amdgcn.exp2.f32(float %645), !dbg !22
  %647 = insertelement <4 x float> %643, float %646, i64 2, !dbg !23
  %648 = fadd float %646, %644, !dbg !24
  %649 = extractelement <4 x float> %541, i64 3, !dbg !22
  %650 = tail call float @llvm.amdgcn.exp2.f32(float %649), !dbg !22
  %651 = insertelement <4 x float> %647, float %650, i64 3, !dbg !23
  %math.result247 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %651) #8, !dbg !23, !frisk.register_lifetimes !5
  %652 = fptrunc <4 x float> %math.result247 to <4 x half>, !dbg !23
  %math.packed248 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %652) #8, !dbg !24, !frisk.register_lifetimes !5
  %653 = fadd float %650, %648, !dbg !24
  %654 = extractelement <4 x float> %542, i64 0, !dbg !22
  %655 = tail call float @llvm.amdgcn.exp2.f32(float %654), !dbg !22
  %656 = insertelement <4 x float> poison, float %655, i64 0, !dbg !23
  %657 = fadd float %655, %653, !dbg !24
  %658 = extractelement <4 x float> %542, i64 1, !dbg !22
  %659 = tail call float @llvm.amdgcn.exp2.f32(float %658), !dbg !22
  %660 = insertelement <4 x float> %656, float %659, i64 1, !dbg !23
  %661 = fadd float %659, %657, !dbg !24
  %662 = extractelement <4 x float> %542, i64 2, !dbg !22
  %663 = tail call float @llvm.amdgcn.exp2.f32(float %662), !dbg !22
  %664 = insertelement <4 x float> %660, float %663, i64 2, !dbg !23
  %665 = fadd float %663, %661, !dbg !24
  %666 = extractelement <4 x float> %542, i64 3, !dbg !22
  %667 = tail call float @llvm.amdgcn.exp2.f32(float %666), !dbg !22
  %668 = insertelement <4 x float> %664, float %667, i64 3, !dbg !23
  %math.result249 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %668) #8, !dbg !23, !frisk.register_lifetimes !5
  %669 = fptrunc <4 x float> %math.result249 to <4 x half>, !dbg !23
  %math.packed250 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %669) #8, !dbg !24, !frisk.register_lifetimes !5
  %670 = fadd float %667, %665, !dbg !24
  %671 = extractelement <4 x float> %543, i64 0, !dbg !22
  %672 = tail call float @llvm.amdgcn.exp2.f32(float %671), !dbg !22
  %673 = insertelement <4 x float> poison, float %672, i64 0, !dbg !23
  %674 = fadd float %672, %670, !dbg !24
  %675 = extractelement <4 x float> %543, i64 1, !dbg !22
  %676 = tail call float @llvm.amdgcn.exp2.f32(float %675), !dbg !22
  %677 = insertelement <4 x float> %673, float %676, i64 1, !dbg !23
  %678 = fadd float %676, %674, !dbg !24
  %679 = extractelement <4 x float> %543, i64 2, !dbg !22
  %680 = tail call float @llvm.amdgcn.exp2.f32(float %679), !dbg !22
  %681 = insertelement <4 x float> %677, float %680, i64 2, !dbg !23
  %682 = fadd float %680, %678, !dbg !24
  %683 = extractelement <4 x float> %543, i64 3, !dbg !22
  %684 = tail call float @llvm.amdgcn.exp2.f32(float %683), !dbg !22
  %685 = insertelement <4 x float> %681, float %684, i64 3, !dbg !23
  %math.result251 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %685) #8, !dbg !23, !frisk.register_lifetimes !5
  %686 = fptrunc <4 x float> %math.result251 to <4 x half>, !dbg !23
  %math.packed252 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %686) #8, !dbg !24, !frisk.register_lifetimes !5
  %687 = fadd float %684, %682, !dbg !24
  %688 = extractelement <4 x float> %544, i64 0, !dbg !22
  %689 = tail call float @llvm.amdgcn.exp2.f32(float %688), !dbg !22
  %690 = insertelement <4 x float> poison, float %689, i64 0, !dbg !23
  %691 = fadd float %689, %687, !dbg !24
  %692 = extractelement <4 x float> %544, i64 1, !dbg !22
  %693 = tail call float @llvm.amdgcn.exp2.f32(float %692), !dbg !22
  %694 = insertelement <4 x float> %690, float %693, i64 1, !dbg !23
  %695 = fadd float %693, %691, !dbg !24
  %696 = extractelement <4 x float> %544, i64 2, !dbg !22
  %697 = tail call float @llvm.amdgcn.exp2.f32(float %696), !dbg !22
  %698 = insertelement <4 x float> %694, float %697, i64 2, !dbg !23
  %699 = fadd float %697, %695, !dbg !24
  %700 = extractelement <4 x float> %544, i64 3, !dbg !22
  %701 = tail call float @llvm.amdgcn.exp2.f32(float %700), !dbg !22
  %702 = insertelement <4 x float> %698, float %701, i64 3, !dbg !23
  %math.result253 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %702) #8, !dbg !23, !frisk.register_lifetimes !5
  %703 = fptrunc <4 x float> %math.result253 to <4 x half>, !dbg !23
  %math.packed254 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %703) #8, !dbg !24, !frisk.register_lifetimes !5
  %704 = fadd float %701, %699, !dbg !24
  %705 = extractelement <4 x float> %546, i64 0, !dbg !22
  %706 = tail call float @llvm.amdgcn.exp2.f32(float %705), !dbg !22
  %707 = insertelement <4 x float> poison, float %706, i64 0, !dbg !23
  %708 = extractelement <4 x float> %546, i64 1, !dbg !22
  %709 = tail call float @llvm.amdgcn.exp2.f32(float %708), !dbg !22
  %710 = insertelement <4 x float> %707, float %709, i64 1, !dbg !23
  %711 = fadd float %706, %709, !dbg !24
  %712 = extractelement <4 x float> %546, i64 2, !dbg !22
  %713 = tail call float @llvm.amdgcn.exp2.f32(float %712), !dbg !22
  %714 = insertelement <4 x float> %710, float %713, i64 2, !dbg !23
  %715 = fadd float %713, %711, !dbg !24
  %716 = extractelement <4 x float> %546, i64 3, !dbg !22
  %717 = tail call float @llvm.amdgcn.exp2.f32(float %716), !dbg !22
  %718 = insertelement <4 x float> %714, float %717, i64 3, !dbg !23
  %math.result255 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %718) #8, !dbg !23, !frisk.register_lifetimes !5
  %719 = fptrunc <4 x float> %math.result255 to <4 x half>, !dbg !23
  %math.packed256 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %719) #8, !dbg !24, !frisk.register_lifetimes !5
  %720 = fadd float %717, %715, !dbg !24
  %721 = extractelement <4 x float> %547, i64 0, !dbg !22
  %722 = tail call float @llvm.amdgcn.exp2.f32(float %721), !dbg !22
  %723 = insertelement <4 x float> poison, float %722, i64 0, !dbg !23
  %724 = fadd float %722, %720, !dbg !24
  %725 = extractelement <4 x float> %547, i64 1, !dbg !22
  %726 = tail call float @llvm.amdgcn.exp2.f32(float %725), !dbg !22
  %727 = insertelement <4 x float> %723, float %726, i64 1, !dbg !23
  %728 = fadd float %726, %724, !dbg !24
  %729 = extractelement <4 x float> %547, i64 2, !dbg !22
  %730 = tail call float @llvm.amdgcn.exp2.f32(float %729), !dbg !22
  %731 = insertelement <4 x float> %727, float %730, i64 2, !dbg !23
  %732 = fadd float %730, %728, !dbg !24
  %733 = extractelement <4 x float> %547, i64 3, !dbg !22
  %734 = tail call float @llvm.amdgcn.exp2.f32(float %733), !dbg !22
  %735 = insertelement <4 x float> %731, float %734, i64 3, !dbg !23
  %math.result257 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %735) #8, !dbg !23, !frisk.register_lifetimes !5
  %736 = fptrunc <4 x float> %math.result257 to <4 x half>, !dbg !23
  %math.packed258 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %736) #8, !dbg !24, !frisk.register_lifetimes !5
  %737 = fadd float %734, %732, !dbg !24
  %738 = extractelement <4 x float> %548, i64 0, !dbg !22
  %739 = tail call float @llvm.amdgcn.exp2.f32(float %738), !dbg !22
  %740 = insertelement <4 x float> poison, float %739, i64 0, !dbg !23
  %741 = fadd float %739, %737, !dbg !24
  %742 = extractelement <4 x float> %548, i64 1, !dbg !22
  %743 = tail call float @llvm.amdgcn.exp2.f32(float %742), !dbg !22
  %744 = insertelement <4 x float> %740, float %743, i64 1, !dbg !23
  %745 = fadd float %743, %741, !dbg !24
  %746 = extractelement <4 x float> %548, i64 2, !dbg !22
  %747 = tail call float @llvm.amdgcn.exp2.f32(float %746), !dbg !22
  %748 = insertelement <4 x float> %744, float %747, i64 2, !dbg !23
  %749 = fadd float %747, %745, !dbg !24
  %750 = extractelement <4 x float> %548, i64 3, !dbg !22
  %751 = tail call float @llvm.amdgcn.exp2.f32(float %750), !dbg !22
  %752 = insertelement <4 x float> %748, float %751, i64 3, !dbg !23
  %math.result259 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %752) #8, !dbg !23, !frisk.register_lifetimes !5
  %753 = fptrunc <4 x float> %math.result259 to <4 x half>, !dbg !23
  %math.packed260 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %753) #8, !dbg !24, !frisk.register_lifetimes !5
  %754 = fadd float %751, %749, !dbg !24
  %755 = extractelement <4 x float> %549, i64 0, !dbg !22
  %756 = tail call float @llvm.amdgcn.exp2.f32(float %755), !dbg !22
  %757 = insertelement <4 x float> poison, float %756, i64 0, !dbg !23
  %758 = fadd float %756, %754, !dbg !24
  %759 = extractelement <4 x float> %549, i64 1, !dbg !22
  %760 = tail call float @llvm.amdgcn.exp2.f32(float %759), !dbg !22
  %761 = insertelement <4 x float> %757, float %760, i64 1, !dbg !23
  %762 = fadd float %760, %758, !dbg !24
  %763 = extractelement <4 x float> %549, i64 2, !dbg !22
  %764 = tail call float @llvm.amdgcn.exp2.f32(float %763), !dbg !22
  %765 = insertelement <4 x float> %761, float %764, i64 2, !dbg !23
  %766 = fadd float %764, %762, !dbg !24
  %767 = extractelement <4 x float> %549, i64 3, !dbg !22
  %768 = tail call float @llvm.amdgcn.exp2.f32(float %767), !dbg !22
  %769 = insertelement <4 x float> %765, float %768, i64 3, !dbg !23
  %math.result261 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %769) #8, !dbg !23, !frisk.register_lifetimes !5
  %770 = fptrunc <4 x float> %math.result261 to <4 x half>, !dbg !23
  %math.packed262 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %770) #8, !dbg !24, !frisk.register_lifetimes !5
  %771 = fadd float %768, %766, !dbg !24
  %772 = extractelement <4 x float> %550, i64 0, !dbg !22
  %773 = tail call float @llvm.amdgcn.exp2.f32(float %772), !dbg !22
  %774 = insertelement <4 x float> poison, float %773, i64 0, !dbg !23
  %775 = fadd float %773, %771, !dbg !24
  %776 = extractelement <4 x float> %550, i64 1, !dbg !22
  %777 = tail call float @llvm.amdgcn.exp2.f32(float %776), !dbg !22
  %778 = insertelement <4 x float> %774, float %777, i64 1, !dbg !23
  %779 = fadd float %777, %775, !dbg !24
  %780 = extractelement <4 x float> %550, i64 2, !dbg !22
  %781 = tail call float @llvm.amdgcn.exp2.f32(float %780), !dbg !22
  %782 = insertelement <4 x float> %778, float %781, i64 2, !dbg !23
  %783 = fadd float %781, %779, !dbg !24
  %784 = extractelement <4 x float> %550, i64 3, !dbg !22
  %785 = tail call float @llvm.amdgcn.exp2.f32(float %784), !dbg !22
  %786 = insertelement <4 x float> %782, float %785, i64 3, !dbg !23
  %math.result263 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %786) #8, !dbg !23, !frisk.register_lifetimes !5
  %787 = fptrunc <4 x float> %math.result263 to <4 x half>, !dbg !23
  %math.packed264 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %787) #8, !dbg !24, !frisk.register_lifetimes !5
  %788 = fadd float %785, %783, !dbg !24
  %789 = extractelement <4 x float> %551, i64 0, !dbg !22
  %790 = tail call float @llvm.amdgcn.exp2.f32(float %789), !dbg !22
  %791 = insertelement <4 x float> poison, float %790, i64 0, !dbg !23
  %792 = fadd float %790, %788, !dbg !24
  %793 = extractelement <4 x float> %551, i64 1, !dbg !22
  %794 = tail call float @llvm.amdgcn.exp2.f32(float %793), !dbg !22
  %795 = insertelement <4 x float> %791, float %794, i64 1, !dbg !23
  %796 = fadd float %794, %792, !dbg !24
  %797 = extractelement <4 x float> %551, i64 2, !dbg !22
  %798 = tail call float @llvm.amdgcn.exp2.f32(float %797), !dbg !22
  %799 = insertelement <4 x float> %795, float %798, i64 2, !dbg !23
  %800 = fadd float %798, %796, !dbg !24
  %801 = extractelement <4 x float> %551, i64 3, !dbg !22
  %802 = tail call float @llvm.amdgcn.exp2.f32(float %801), !dbg !22
  %803 = insertelement <4 x float> %799, float %802, i64 3, !dbg !23
  %math.result265 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %803) #8, !dbg !23, !frisk.register_lifetimes !5
  %804 = fptrunc <4 x float> %math.result265 to <4 x half>, !dbg !23
  %math.packed266 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %804) #8, !dbg !24, !frisk.register_lifetimes !5
  %805 = fadd float %802, %800, !dbg !24
  %806 = extractelement <4 x float> %552, i64 0, !dbg !22
  %807 = tail call float @llvm.amdgcn.exp2.f32(float %806), !dbg !22
  %808 = insertelement <4 x float> poison, float %807, i64 0, !dbg !23
  %809 = fadd float %807, %805, !dbg !24
  %810 = extractelement <4 x float> %552, i64 1, !dbg !22
  %811 = tail call float @llvm.amdgcn.exp2.f32(float %810), !dbg !22
  %812 = insertelement <4 x float> %808, float %811, i64 1, !dbg !23
  %813 = fadd float %811, %809, !dbg !24
  %814 = extractelement <4 x float> %552, i64 2, !dbg !22
  %815 = tail call float @llvm.amdgcn.exp2.f32(float %814), !dbg !22
  %816 = insertelement <4 x float> %812, float %815, i64 2, !dbg !23
  %817 = fadd float %815, %813, !dbg !24
  %818 = extractelement <4 x float> %552, i64 3, !dbg !22
  %819 = tail call float @llvm.amdgcn.exp2.f32(float %818), !dbg !22
  %820 = insertelement <4 x float> %816, float %819, i64 3, !dbg !23
  %math.result267 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %820) #8, !dbg !23, !frisk.register_lifetimes !5
  %821 = fptrunc <4 x float> %math.result267 to <4 x half>, !dbg !23
  %math.packed268 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %821) #8, !dbg !24, !frisk.register_lifetimes !5
  %822 = fadd float %819, %817, !dbg !24
  %823 = extractelement <4 x float> %553, i64 0, !dbg !22
  %824 = tail call float @llvm.amdgcn.exp2.f32(float %823), !dbg !22
  %825 = insertelement <4 x float> poison, float %824, i64 0, !dbg !23
  %826 = fadd float %824, %822, !dbg !24
  %827 = extractelement <4 x float> %553, i64 1, !dbg !22
  %828 = tail call float @llvm.amdgcn.exp2.f32(float %827), !dbg !22
  %829 = insertelement <4 x float> %825, float %828, i64 1, !dbg !23
  %830 = fadd float %828, %826, !dbg !24
  %831 = extractelement <4 x float> %553, i64 2, !dbg !22
  %832 = tail call float @llvm.amdgcn.exp2.f32(float %831), !dbg !22
  %833 = insertelement <4 x float> %829, float %832, i64 2, !dbg !23
  %834 = fadd float %832, %830, !dbg !24
  %835 = extractelement <4 x float> %553, i64 3, !dbg !22
  %836 = tail call float @llvm.amdgcn.exp2.f32(float %835), !dbg !22
  %837 = insertelement <4 x float> %833, float %836, i64 3, !dbg !23
  %math.result269 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %837) #8, !dbg !23, !frisk.register_lifetimes !5
  %838 = fptrunc <4 x float> %math.result269 to <4 x half>, !dbg !23
  %math.packed270 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %838) #8, !dbg !24, !frisk.register_lifetimes !5
  %839 = fadd float %836, %834, !dbg !24
  %840 = bitcast float %704 to i32, !dbg !24
  %841 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %477, i32 %840), !dbg !24
  %842 = bitcast i32 %841 to float, !dbg !24
  %843 = fadd float %704, %842, !dbg !24
  %844 = bitcast float %843 to i32, !dbg !24
  %845 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %485, i32 %844), !dbg !24
  %846 = bitcast i32 %845 to float, !dbg !24
  %847 = fadd float %843, %846, !dbg !24
  %848 = fadd float %561, %847, !dbg !24
  %849 = insertelement <1 x float> poison, float %848, i64 0, !dbg !11
  %850 = insertvalue [2 x <1 x float>] poison, <1 x float> %849, 0, !dbg !11
  %851 = bitcast float %839 to i32, !dbg !24
  %852 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %477, i32 %851), !dbg !24
  %853 = bitcast i32 %852 to float, !dbg !24
  %854 = fadd float %839, %853, !dbg !24
  %855 = bitcast float %854 to i32, !dbg !24
  %856 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %485, i32 %855), !dbg !24
  %857 = bitcast i32 %856 to float, !dbg !24
  %858 = fadd float %854, %857, !dbg !24
  %859 = fadd float %569, %858, !dbg !24
  %860 = insertelement <1 x float> poison, float %859, i64 0, !dbg !11
  %861 = insertvalue [2 x <1 x float>] %850, <1 x float> %860, 1, !dbg !11
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %vdb.address1303 = or disjoint i64 %address.local788, %122, !dbg !14
  %vdb.address1304 = trunc nuw nsw i64 %vdb.address1303 to i32, !dbg !14
  %vdb.address1305 = shl nuw nsw i32 %vdb.address1304, 1, !dbg !14
  %vdb.address1312 = or disjoint i64 %address.local794, %122, !dbg !14
  %vdb.address1313 = trunc nuw nsw i64 %vdb.address1312 to i32, !dbg !14
  %vdb.address1314 = shl nuw nsw i32 %vdb.address1313, 1, !dbg !14
  %vdb.dma.base = add i32 %dma.buffer.lds1241, 8192, !dbg !25
  %862 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1305, <4 x i32> %dma.buffer.rsrc1216) #9, !dbg !14
  %vdb.dma.base1319 = add i32 %dma.buffer.lds1238, 8192, !dbg !25
  %863 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1319, i32 %vdb.address1314, <4 x i32> %dma.buffer.rsrc1216) #9, !dbg !14
  %address.local796 = shl nuw nsw i32 %address.thread365, 3, !dbg !25
  %address.local797 = and i32 %address.local796, 120, !dbg !25
  %address.local799 = and i32 %address.local442, 1536, !dbg !25
  %address.local800 = or disjoint i32 %address.local797, %address.local799, !dbg !25
  %address.local801 = zext nneg i32 %address.local800 to i64, !dbg !25
  %address.local802 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local801, !dbg !25
  %864 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local802, i32 0) #10, !dbg !25, !frisk.v_double_buffer !26
  %865 = bitcast <4 x i32> %864 to <8 x half>, !dbg !25
  %address.local803 = and i64 %address.local366, 60, !dbg !25
  %address.local804 = add nuw nsw i64 %address.local803, %address.local366, !dbg !25
  %address.local806.idx = shl nuw nsw i64 %address.local804, 4, !dbg !25
  %address.local806 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %address.local806.idx, !dbg !25
  %address.local807 = lshr i64 %address.local422, 4, !dbg !25
  %address.local809.idx = mul nuw nsw i64 %address.local807, 768, !dbg !25
  %address.local809 = getelementptr i8, ptr addrspace(3) %address.local806, i64 %address.local809.idx, !dbg !25
  %address.local811.idx = mul nsw i64 %address.local409, -64, !dbg !25
  %address.local811 = getelementptr i8, ptr addrspace(3) %address.local809, i64 %address.local811.idx, !dbg !25
  %address.local812 = getelementptr i8, ptr addrspace(3) %address.local811, i64 256, !dbg !25
  %866 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local812, i32 0) #10, !dbg !25, !frisk.v_double_buffer !26
  %867 = bitcast <4 x i32> %866 to <8 x half>, !dbg !25
  %address.local816 = getelementptr i8, ptr addrspace(3) %address.local811, i64 512, !dbg !25
  %868 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local816, i32 0) #10, !dbg !25, !frisk.v_double_buffer !26
  %869 = bitcast <4 x i32> %868 to <8 x half>, !dbg !25
  %address.local820 = getelementptr i8, ptr addrspace(3) %address.local811, i64 768, !dbg !25
  %870 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local820, i32 0) #10, !dbg !25, !frisk.v_double_buffer !26
  %871 = bitcast <4 x i32> %870 to <8 x half>, !dbg !25
  %address.local822 = getelementptr i8, ptr addrspace(3) %address.local802, i64 4096, !dbg !25
  %872 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local822, i32 0) #10, !dbg !25, !frisk.v_double_buffer !26
  %873 = bitcast <4 x i32> %872 to <8 x half>, !dbg !25
  %address.local826 = getelementptr i8, ptr addrspace(3) %address.local811, i64 4352, !dbg !25
  %874 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local826, i32 0) #10, !dbg !25, !frisk.v_double_buffer !26
  %875 = bitcast <4 x i32> %874 to <8 x half>, !dbg !25
  %address.local830 = getelementptr i8, ptr addrspace(3) %address.local811, i64 4608, !dbg !25
  %876 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local830, i32 0) #10, !dbg !25, !frisk.v_double_buffer !26
  %877 = bitcast <4 x i32> %876 to <8 x half>, !dbg !25
  %address.local834 = getelementptr i8, ptr addrspace(3) %address.local811, i64 4864, !dbg !25
  %878 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local834, i32 0) #10, !dbg !25, !frisk.v_double_buffer !26
  %879 = shufflevector <8 x half> %865, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %880 = extractvalue [2 x <8 x float>] %118, 0, !dbg !20
  %881 = shufflevector <8 x float> %880, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !20
  %882 = fmul <4 x float> %881, %558, !dbg !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %883 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %879, <4 x float> %882) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %884 = shufflevector <8 x half> %865, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %885 = shufflevector <8 x float> %880, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !20
  %886 = fmul <4 x float> %885, %558, !dbg !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %887 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %884, <4 x float> %886) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !20
  %888 = extractvalue [2 x <8 x float>] %118, 1, !dbg !20
  %889 = shufflevector <8 x float> %888, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !20
  %890 = fmul <4 x float> %889, %566, !dbg !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %891 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed256, <4 x half> %879, <4 x float> %890) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !20
  %892 = shufflevector <8 x float> %888, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !20
  %893 = fmul <4 x float> %892, %566, !dbg !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %894 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed256, <4 x half> %884, <4 x float> %893) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %895 = shufflevector <8 x half> %867, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %896 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed242, <4 x half> %895, <4 x float> %883) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %897 = shufflevector <8 x half> %867, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %898 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed242, <4 x half> %897, <4 x float> %887) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %899 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed258, <4 x half> %895, <4 x float> %891) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %900 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed258, <4 x half> %897, <4 x float> %894) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %901 = shufflevector <8 x half> %869, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %902 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed244, <4 x half> %901, <4 x float> %896) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %903 = shufflevector <8 x half> %869, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %904 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed244, <4 x half> %903, <4 x float> %898) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %905 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed260, <4 x half> %901, <4 x float> %899) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %906 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed260, <4 x half> %903, <4 x float> %900) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %907 = shufflevector <8 x half> %871, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %908 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed246, <4 x half> %907, <4 x float> %902) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %909 = shufflevector <8 x half> %871, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %910 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed246, <4 x half> %909, <4 x float> %904) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %911 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed262, <4 x half> %907, <4 x float> %905) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %912 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed262, <4 x half> %909, <4 x float> %906) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %913 = shufflevector <8 x half> %873, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %914 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed248, <4 x half> %913, <4 x float> %908) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %915 = shufflevector <8 x half> %873, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %916 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed248, <4 x half> %915, <4 x float> %910) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %917 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed264, <4 x half> %913, <4 x float> %911) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %918 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed264, <4 x half> %915, <4 x float> %912) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %919 = shufflevector <8 x half> %875, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %920 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed250, <4 x half> %919, <4 x float> %914) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %921 = shufflevector <8 x half> %875, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %922 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed250, <4 x half> %921, <4 x float> %916) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %923 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed266, <4 x half> %919, <4 x float> %917) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %924 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed266, <4 x half> %921, <4 x float> %918) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %925 = shufflevector <8 x half> %877, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %926 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed252, <4 x half> %925, <4 x float> %920) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %927 = shufflevector <8 x half> %877, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %928 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed252, <4 x half> %927, <4 x float> %922) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %929 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed268, <4 x half> %925, <4 x float> %923) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %930 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed268, <4 x half> %927, <4 x float> %924) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %931 = bitcast <4 x i32> %878 to <8 x half>, !dbg !25
  %932 = shufflevector <8 x half> %931, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %933 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed254, <4 x half> %932, <4 x float> %926) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %934 = shufflevector <8 x half> %931, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %935 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed254, <4 x half> %934, <4 x float> %928) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %936 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed270, <4 x half> %932, <4 x float> %929) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %937 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed270, <4 x half> %934, <4 x float> %930) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %vdb.address1329 = or disjoint i64 %address.local788, %218, !dbg !14
  %vdb.address1330 = trunc nuw nsw i64 %vdb.address1329 to i32, !dbg !14
  %vdb.address1331 = shl nuw nsw i32 %vdb.address1330, 1, !dbg !14
  %vdb.address1338 = or disjoint i64 %address.local794, %218, !dbg !14
  %vdb.address1339 = trunc nuw nsw i64 %vdb.address1338 to i32, !dbg !14
  %vdb.address1340 = shl nuw nsw i32 %vdb.address1339, 1, !dbg !14
  %938 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1241, i32 %vdb.address1331, <4 x i32> %dma.buffer.rsrc1216) #9, !dbg !14
  %939 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1238, i32 %vdb.address1340, <4 x i32> %dma.buffer.rsrc1216) #9, !dbg !14
  %vdb.read = getelementptr i8, ptr addrspace(3) %address.local802, i64 8192, !dbg !25
  %940 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #10, !dbg !25, !frisk.v_double_buffer !26
  %941 = bitcast <4 x i32> %940 to <8 x half>, !dbg !25
  %vdb.read1320 = getelementptr i8, ptr addrspace(3) %address.local811, i64 8448, !dbg !25
  %942 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1320, i32 0) #10, !dbg !25, !frisk.v_double_buffer !26
  %943 = bitcast <4 x i32> %942 to <8 x half>, !dbg !25
  %vdb.read1321 = getelementptr i8, ptr addrspace(3) %address.local811, i64 8704, !dbg !25
  %944 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1321, i32 0) #10, !dbg !25, !frisk.v_double_buffer !26
  %945 = bitcast <4 x i32> %944 to <8 x half>, !dbg !25
  %vdb.read1322 = getelementptr i8, ptr addrspace(3) %address.local811, i64 8960, !dbg !25
  %946 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1322, i32 0) #10, !dbg !25, !frisk.v_double_buffer !26
  %947 = bitcast <4 x i32> %946 to <8 x half>, !dbg !25
  %vdb.read1323 = getelementptr i8, ptr addrspace(3) %address.local802, i64 12288, !dbg !25
  %948 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1323, i32 0) #10, !dbg !25, !frisk.v_double_buffer !26
  %949 = bitcast <4 x i32> %948 to <8 x half>, !dbg !25
  %vdb.read1324 = getelementptr i8, ptr addrspace(3) %address.local811, i64 12544, !dbg !25
  %950 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1324, i32 0) #10, !dbg !25, !frisk.v_double_buffer !26
  %951 = bitcast <4 x i32> %950 to <8 x half>, !dbg !25
  %vdb.read1325 = getelementptr i8, ptr addrspace(3) %address.local811, i64 12800, !dbg !25
  %952 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1325, i32 0) #10, !dbg !25, !frisk.v_double_buffer !26
  %953 = bitcast <4 x i32> %952 to <8 x half>, !dbg !25
  %vdb.read1326 = getelementptr i8, ptr addrspace(3) %address.local811, i64 13056, !dbg !25
  %954 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1326, i32 0) #10, !dbg !25, !frisk.v_double_buffer !26
  %955 = shufflevector <8 x half> %941, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %956 = extractvalue [2 x <8 x float>] %115, 0, !dbg !20
  %957 = shufflevector <8 x float> %956, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !20
  %958 = fmul <4 x float> %957, %558, !dbg !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %959 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %955, <4 x float> %958) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %960 = shufflevector <8 x half> %941, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %961 = shufflevector <8 x float> %956, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !20
  %962 = fmul <4 x float> %961, %558, !dbg !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %963 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %960, <4 x float> %962) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !20
  %964 = extractvalue [2 x <8 x float>] %115, 1, !dbg !20
  %965 = shufflevector <8 x float> %964, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !20
  %966 = fmul <4 x float> %965, %566, !dbg !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %967 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed256, <4 x half> %955, <4 x float> %966) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !20
  %968 = shufflevector <8 x float> %964, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !20
  %969 = fmul <4 x float> %968, %566, !dbg !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %970 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed256, <4 x half> %960, <4 x float> %969) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %971 = shufflevector <8 x half> %943, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %972 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed242, <4 x half> %971, <4 x float> %959) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %973 = shufflevector <8 x half> %943, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %974 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed242, <4 x half> %973, <4 x float> %963) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %975 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed258, <4 x half> %971, <4 x float> %967) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %976 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed258, <4 x half> %973, <4 x float> %970) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %977 = shufflevector <8 x half> %945, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %978 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed244, <4 x half> %977, <4 x float> %972) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %979 = shufflevector <8 x half> %945, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %980 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed244, <4 x half> %979, <4 x float> %974) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %981 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed260, <4 x half> %977, <4 x float> %975) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %982 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed260, <4 x half> %979, <4 x float> %976) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %983 = shufflevector <8 x half> %947, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %984 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed246, <4 x half> %983, <4 x float> %978) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %985 = shufflevector <8 x half> %947, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %986 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed246, <4 x half> %985, <4 x float> %980) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %987 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed262, <4 x half> %983, <4 x float> %981) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %988 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed262, <4 x half> %985, <4 x float> %982) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %989 = shufflevector <8 x half> %949, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %990 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed248, <4 x half> %989, <4 x float> %984) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %991 = shufflevector <8 x half> %949, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %992 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed248, <4 x half> %991, <4 x float> %986) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %993 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed264, <4 x half> %989, <4 x float> %987) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %994 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed264, <4 x half> %991, <4 x float> %988) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %995 = shufflevector <8 x half> %951, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %996 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed250, <4 x half> %995, <4 x float> %990) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %997 = shufflevector <8 x half> %951, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %998 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed250, <4 x half> %997, <4 x float> %992) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %999 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed266, <4 x half> %995, <4 x float> %993) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1000 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed266, <4 x half> %997, <4 x float> %994) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1001 = shufflevector <8 x half> %953, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1002 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed252, <4 x half> %1001, <4 x float> %996) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1003 = shufflevector <8 x half> %953, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1004 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed252, <4 x half> %1003, <4 x float> %998) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1005 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed268, <4 x half> %1001, <4 x float> %999) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1006 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed268, <4 x half> %1003, <4 x float> %1000) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1007 = bitcast <4 x i32> %954 to <8 x half>, !dbg !25
  %1008 = shufflevector <8 x half> %1007, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1009 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed254, <4 x half> %1008, <4 x float> %1002) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1010 = shufflevector <8 x half> %1007, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1011 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed254, <4 x half> %1010, <4 x float> %1004) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1012 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed270, <4 x half> %1008, <4 x float> %1005) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1013 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed270, <4 x half> %1010, <4 x float> %1006) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %vdb.address1347 = or disjoint i64 %address.local788, %271, !dbg !14
  %vdb.address1348 = trunc nuw nsw i64 %vdb.address1347 to i32, !dbg !14
  %vdb.address1349 = shl nuw nsw i32 %vdb.address1348, 1, !dbg !14
  %vdb.address1356 = or disjoint i64 %address.local794, %271, !dbg !14
  %vdb.address1357 = trunc nuw nsw i64 %vdb.address1356 to i32, !dbg !14
  %vdb.address1358 = shl nuw nsw i32 %vdb.address1357, 1, !dbg !14
  %1014 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1349, <4 x i32> %dma.buffer.rsrc1216) #9, !dbg !14
  %1015 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1319, i32 %vdb.address1358, <4 x i32> %dma.buffer.rsrc1216) #9, !dbg !14
  %1016 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local802, i32 0) #10, !dbg !25, !frisk.v_double_buffer !26
  %1017 = bitcast <4 x i32> %1016 to <8 x half>, !dbg !25
  %1018 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local812, i32 0) #10, !dbg !25, !frisk.v_double_buffer !26
  %1019 = bitcast <4 x i32> %1018 to <8 x half>, !dbg !25
  %1020 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local816, i32 0) #10, !dbg !25, !frisk.v_double_buffer !26
  %1021 = bitcast <4 x i32> %1020 to <8 x half>, !dbg !25
  %1022 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local820, i32 0) #10, !dbg !25, !frisk.v_double_buffer !26
  %1023 = bitcast <4 x i32> %1022 to <8 x half>, !dbg !25
  %1024 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local822, i32 0) #10, !dbg !25, !frisk.v_double_buffer !26
  %1025 = bitcast <4 x i32> %1024 to <8 x half>, !dbg !25
  %1026 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local826, i32 0) #10, !dbg !25, !frisk.v_double_buffer !26
  %1027 = bitcast <4 x i32> %1026 to <8 x half>, !dbg !25
  %1028 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local830, i32 0) #10, !dbg !25, !frisk.v_double_buffer !26
  %1029 = bitcast <4 x i32> %1028 to <8 x half>, !dbg !25
  %1030 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local834, i32 0) #10, !dbg !25, !frisk.v_double_buffer !26
  %1031 = shufflevector <8 x half> %1017, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %1032 = extractvalue [2 x <8 x float>] %114, 0, !dbg !20
  %1033 = shufflevector <8 x float> %1032, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !20
  %1034 = fmul <4 x float> %1033, %558, !dbg !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1035 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1031, <4 x float> %1034) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1036 = shufflevector <8 x half> %1017, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %1037 = shufflevector <8 x float> %1032, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !20
  %1038 = fmul <4 x float> %1037, %558, !dbg !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1039 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1036, <4 x float> %1038) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !20
  %1040 = extractvalue [2 x <8 x float>] %114, 1, !dbg !20
  %1041 = shufflevector <8 x float> %1040, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !20
  %1042 = fmul <4 x float> %1041, %566, !dbg !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1043 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed256, <4 x half> %1031, <4 x float> %1042) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !20
  %1044 = shufflevector <8 x float> %1040, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !20
  %1045 = fmul <4 x float> %1044, %566, !dbg !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1046 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed256, <4 x half> %1036, <4 x float> %1045) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1047 = shufflevector <8 x half> %1019, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1048 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed242, <4 x half> %1047, <4 x float> %1035) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1049 = shufflevector <8 x half> %1019, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1050 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed242, <4 x half> %1049, <4 x float> %1039) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1051 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed258, <4 x half> %1047, <4 x float> %1043) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1052 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed258, <4 x half> %1049, <4 x float> %1046) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1053 = shufflevector <8 x half> %1021, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1054 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed244, <4 x half> %1053, <4 x float> %1048) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1055 = shufflevector <8 x half> %1021, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1056 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed244, <4 x half> %1055, <4 x float> %1050) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1057 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed260, <4 x half> %1053, <4 x float> %1051) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1058 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed260, <4 x half> %1055, <4 x float> %1052) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1059 = shufflevector <8 x half> %1023, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1060 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed246, <4 x half> %1059, <4 x float> %1054) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1061 = shufflevector <8 x half> %1023, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1062 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed246, <4 x half> %1061, <4 x float> %1056) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1063 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed262, <4 x half> %1059, <4 x float> %1057) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1064 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed262, <4 x half> %1061, <4 x float> %1058) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1065 = shufflevector <8 x half> %1025, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1066 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed248, <4 x half> %1065, <4 x float> %1060) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1067 = shufflevector <8 x half> %1025, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1068 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed248, <4 x half> %1067, <4 x float> %1062) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1069 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed264, <4 x half> %1065, <4 x float> %1063) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1070 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed264, <4 x half> %1067, <4 x float> %1064) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1071 = shufflevector <8 x half> %1027, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1072 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed250, <4 x half> %1071, <4 x float> %1066) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1073 = shufflevector <8 x half> %1027, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1074 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed250, <4 x half> %1073, <4 x float> %1068) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1075 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed266, <4 x half> %1071, <4 x float> %1069) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1076 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed266, <4 x half> %1073, <4 x float> %1070) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1077 = shufflevector <8 x half> %1029, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1078 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed252, <4 x half> %1077, <4 x float> %1072) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1079 = shufflevector <8 x half> %1029, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1080 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed252, <4 x half> %1079, <4 x float> %1074) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1081 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed268, <4 x half> %1077, <4 x float> %1075) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1082 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed268, <4 x half> %1079, <4 x float> %1076) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1083 = bitcast <4 x i32> %1030 to <8 x half>, !dbg !25
  %1084 = shufflevector <8 x half> %1083, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1085 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed254, <4 x half> %1084, <4 x float> %1078) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1086 = shufflevector <8 x half> %1083, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1087 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed254, <4 x half> %1086, <4 x float> %1080) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1088 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed270, <4 x half> %1084, <4 x float> %1081) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1089 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed270, <4 x half> %1086, <4 x float> %1082) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %1090 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #10, !dbg !25, !frisk.v_double_buffer !26
  %1091 = bitcast <4 x i32> %1090 to <8 x half>, !dbg !25
  %1092 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1320, i32 0) #10, !dbg !25, !frisk.v_double_buffer !26
  %1093 = bitcast <4 x i32> %1092 to <8 x half>, !dbg !25
  %1094 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1321, i32 0) #10, !dbg !25, !frisk.v_double_buffer !26
  %1095 = bitcast <4 x i32> %1094 to <8 x half>, !dbg !25
  %1096 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1322, i32 0) #10, !dbg !25, !frisk.v_double_buffer !26
  %1097 = bitcast <4 x i32> %1096 to <8 x half>, !dbg !25
  %1098 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1323, i32 0) #10, !dbg !25, !frisk.v_double_buffer !26
  %1099 = bitcast <4 x i32> %1098 to <8 x half>, !dbg !25
  %1100 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1324, i32 0) #10, !dbg !25, !frisk.v_double_buffer !26
  %1101 = bitcast <4 x i32> %1100 to <8 x half>, !dbg !25
  %1102 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1325, i32 0) #10, !dbg !25, !frisk.v_double_buffer !26
  %1103 = bitcast <4 x i32> %1102 to <8 x half>, !dbg !25
  %1104 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1326, i32 0) #10, !dbg !25, !frisk.v_double_buffer !26
  %1105 = shufflevector <8 x half> %1091, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %1106 = extractvalue [2 x <8 x float>] %113, 0, !dbg !20
  %1107 = shufflevector <8 x float> %1106, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !20
  %1108 = fmul <4 x float> %1107, %558, !dbg !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1109 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1105, <4 x float> %1108) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1110 = shufflevector <8 x half> %1091, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %1111 = shufflevector <8 x float> %1106, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !20
  %1112 = fmul <4 x float> %1111, %558, !dbg !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1113 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1110, <4 x float> %1112) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !20
  %1114 = extractvalue [2 x <8 x float>] %113, 1, !dbg !20
  %1115 = shufflevector <8 x float> %1114, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !20
  %1116 = fmul <4 x float> %1115, %566, !dbg !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1117 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed256, <4 x half> %1105, <4 x float> %1116) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !20
  %1118 = shufflevector <8 x float> %1114, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !20
  %1119 = fmul <4 x float> %1118, %566, !dbg !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1120 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed256, <4 x half> %1110, <4 x float> %1119) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1121 = shufflevector <8 x half> %1093, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1122 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed242, <4 x half> %1121, <4 x float> %1109) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1123 = shufflevector <8 x half> %1093, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1124 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed242, <4 x half> %1123, <4 x float> %1113) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1125 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed258, <4 x half> %1121, <4 x float> %1117) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1126 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed258, <4 x half> %1123, <4 x float> %1120) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1127 = shufflevector <8 x half> %1095, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1128 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed244, <4 x half> %1127, <4 x float> %1122) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1129 = shufflevector <8 x half> %1095, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1130 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed244, <4 x half> %1129, <4 x float> %1124) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1131 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed260, <4 x half> %1127, <4 x float> %1125) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1132 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed260, <4 x half> %1129, <4 x float> %1126) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1133 = shufflevector <8 x half> %1097, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1134 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed246, <4 x half> %1133, <4 x float> %1128) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1135 = shufflevector <8 x half> %1097, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1136 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed246, <4 x half> %1135, <4 x float> %1130) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1137 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed262, <4 x half> %1133, <4 x float> %1131) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1138 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed262, <4 x half> %1135, <4 x float> %1132) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1139 = shufflevector <8 x half> %1099, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1140 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed248, <4 x half> %1139, <4 x float> %1134) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1141 = shufflevector <8 x half> %1099, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1142 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed248, <4 x half> %1141, <4 x float> %1136) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1143 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed264, <4 x half> %1139, <4 x float> %1137) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1144 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed264, <4 x half> %1141, <4 x float> %1138) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1145 = shufflevector <8 x half> %1101, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1146 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed250, <4 x half> %1145, <4 x float> %1140) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1147 = shufflevector <8 x half> %1101, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1148 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed250, <4 x half> %1147, <4 x float> %1142) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1149 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed266, <4 x half> %1145, <4 x float> %1143) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1150 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed266, <4 x half> %1147, <4 x float> %1144) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1151 = shufflevector <8 x half> %1103, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1152 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed252, <4 x half> %1151, <4 x float> %1146) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1153 = shufflevector <8 x half> %1103, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1154 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed252, <4 x half> %1153, <4 x float> %1148) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1155 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed268, <4 x half> %1151, <4 x float> %1149) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1156 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed268, <4 x half> %1153, <4 x float> %1150) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1157 = bitcast <4 x i32> %1104 to <8 x half>, !dbg !25
  %1158 = shufflevector <8 x half> %1157, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1159 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed254, <4 x half> %1158, <4 x float> %1152) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1160 = shufflevector <8 x half> %1157, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1161 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed254, <4 x half> %1160, <4 x float> %1154) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1162 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed270, <4 x half> %1158, <4 x float> %1155) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %1163 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed270, <4 x half> %1160, <4 x float> %1156) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !11
  %1164 = shufflevector <4 x float> %933, <4 x float> %935, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1165 = insertvalue [2 x <8 x float>] poison, <8 x float> %1164, 0, !dbg !11
  %1166 = shufflevector <4 x float> %936, <4 x float> %937, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1167 = insertvalue [2 x <8 x float>] %1165, <8 x float> %1166, 1, !dbg !11
  %1168 = insertvalue [2 x <1 x float>] poison, <1 x float> %492, 0, !dbg !11
  %1169 = insertvalue [2 x <1 x float>] %1168, <1 x float> %535, 1, !dbg !11
  %1170 = shufflevector <4 x float> %1009, <4 x float> %1011, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1171 = insertvalue [2 x <8 x float>] poison, <8 x float> %1170, 0, !dbg !11
  %1172 = shufflevector <4 x float> %1012, <4 x float> %1013, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1173 = insertvalue [2 x <8 x float>] %1171, <8 x float> %1172, 1, !dbg !11
  %1174 = shufflevector <4 x float> %1085, <4 x float> %1087, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1175 = insertvalue [2 x <8 x float>] poison, <8 x float> %1174, 0, !dbg !11
  %1176 = shufflevector <4 x float> %1088, <4 x float> %1089, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1177 = insertvalue [2 x <8 x float>] %1175, <8 x float> %1176, 1, !dbg !11
  %1178 = shufflevector <4 x float> %1159, <4 x float> %1161, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1179 = insertvalue [2 x <8 x float>] poison, <8 x float> %1178, 0, !dbg !11
  %1180 = shufflevector <4 x float> %1162, <4 x float> %1163, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1181 = insertvalue [2 x <8 x float>] %1179, <8 x float> %1180, 1, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  tail call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %1182 = add nuw nsw i64 %119, 1, !dbg !11
  %1183 = icmp samesign ult i64 %119, 31, !dbg !11
  br i1 %1183, label %112, label %1184, !dbg !11

1184:                                             ; preds = %112
  %1185 = and i32 %kdma.tid, 16, !dbg !11
  %.not = icmp eq i32 %1185, 0, !dbg !11
  %1186 = and i32 %kdma.tid, 32, !dbg !11
  %.not36 = icmp eq i32 %1186, 0, !dbg !11
  %1187 = extractelement <4 x float> %1159, i64 0, !dbg !11
  %1188 = extractelement <4 x float> %1159, i64 1, !dbg !11
  %1189 = extractelement <4 x float> %1159, i64 2, !dbg !11
  %1190 = extractelement <4 x float> %1159, i64 3, !dbg !11
  %1191 = select i1 %.not, float %1188, float %1187, !dbg !11
  %1192 = bitcast float %1191 to i32, !dbg !11
  %1193 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %477, i32 %1192), !dbg !11
  %1194 = bitcast i32 %1193 to float, !dbg !11
  %1195 = select i1 %.not, float %1187, float %1194, !dbg !11
  %1196 = select i1 %.not, float %1194, float %1188, !dbg !11
  %1197 = select i1 %.not, float %1190, float %1189, !dbg !11
  %1198 = bitcast float %1197 to i32, !dbg !11
  %1199 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %477, i32 %1198), !dbg !11
  %1200 = bitcast i32 %1199 to float, !dbg !11
  %1201 = select i1 %.not, float %1189, float %1200, !dbg !11
  %1202 = select i1 %.not, float %1200, float %1190, !dbg !11
  %1203 = select i1 %.not36, float %1201, float %1195, !dbg !11
  %1204 = bitcast float %1203 to i32, !dbg !11
  %1205 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %485, i32 %1204), !dbg !11
  %1206 = bitcast i32 %1205 to float, !dbg !11
  %1207 = select i1 %.not36, float %1202, float %1196, !dbg !11
  %1208 = bitcast float %1207 to i32, !dbg !11
  %1209 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %485, i32 %1208), !dbg !11
  %1210 = bitcast i32 %1209 to float, !dbg !11
  %1211 = insertelement <4 x i1> poison, i1 %.not36, i64 0, !dbg !11
  %1212 = shufflevector <4 x i1> %1211, <4 x i1> poison, <4 x i32> zeroinitializer, !dbg !11
  %1213 = insertelement <4 x float> poison, float %1195, i64 0, !dbg !11
  %1214 = insertelement <4 x float> %1213, float %1196, i64 1, !dbg !11
  %1215 = insertelement <4 x float> %1214, float %1206, i64 2, !dbg !11
  %1216 = insertelement <4 x float> %1215, float %1210, i64 3, !dbg !11
  %1217 = insertelement <4 x float> poison, float %1206, i64 0, !dbg !11
  %1218 = insertelement <4 x float> %1217, float %1210, i64 1, !dbg !11
  %1219 = insertelement <4 x float> %1218, float %1201, i64 2, !dbg !11
  %1220 = insertelement <4 x float> %1219, float %1202, i64 3, !dbg !11
  %1221 = select <4 x i1> %1212, <4 x float> %1216, <4 x float> %1220, !dbg !11
  %1222 = extractelement <4 x float> %1085, i64 0, !dbg !11
  %1223 = extractelement <4 x float> %1085, i64 1, !dbg !11
  %1224 = extractelement <4 x float> %1085, i64 2, !dbg !11
  %1225 = extractelement <4 x float> %1085, i64 3, !dbg !11
  %1226 = select i1 %.not, float %1223, float %1222, !dbg !11
  %1227 = bitcast float %1226 to i32, !dbg !11
  %1228 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %477, i32 %1227), !dbg !11
  %1229 = bitcast i32 %1228 to float, !dbg !11
  %1230 = select i1 %.not, float %1222, float %1229, !dbg !11
  %1231 = select i1 %.not, float %1229, float %1223, !dbg !11
  %1232 = select i1 %.not, float %1225, float %1224, !dbg !11
  %1233 = bitcast float %1232 to i32, !dbg !11
  %1234 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %477, i32 %1233), !dbg !11
  %1235 = bitcast i32 %1234 to float, !dbg !11
  %1236 = select i1 %.not, float %1224, float %1235, !dbg !11
  %1237 = select i1 %.not, float %1235, float %1225, !dbg !11
  %1238 = select i1 %.not36, float %1236, float %1230, !dbg !11
  %1239 = bitcast float %1238 to i32, !dbg !11
  %1240 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %485, i32 %1239), !dbg !11
  %1241 = bitcast i32 %1240 to float, !dbg !11
  %1242 = select i1 %.not36, float %1237, float %1231, !dbg !11
  %1243 = bitcast float %1242 to i32, !dbg !11
  %1244 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %485, i32 %1243), !dbg !11
  %1245 = bitcast i32 %1244 to float, !dbg !11
  %1246 = insertelement <4 x float> poison, float %1230, i64 0, !dbg !11
  %1247 = insertelement <4 x float> %1246, float %1231, i64 1, !dbg !11
  %1248 = insertelement <4 x float> %1247, float %1241, i64 2, !dbg !11
  %1249 = insertelement <4 x float> %1248, float %1245, i64 3, !dbg !11
  %1250 = insertelement <4 x float> poison, float %1241, i64 0, !dbg !11
  %1251 = insertelement <4 x float> %1250, float %1245, i64 1, !dbg !11
  %1252 = insertelement <4 x float> %1251, float %1236, i64 2, !dbg !11
  %1253 = insertelement <4 x float> %1252, float %1237, i64 3, !dbg !11
  %1254 = select <4 x i1> %1212, <4 x float> %1249, <4 x float> %1253, !dbg !11
  %1255 = extractelement <4 x float> %1009, i64 0, !dbg !11
  %1256 = extractelement <4 x float> %1009, i64 1, !dbg !11
  %1257 = extractelement <4 x float> %1009, i64 2, !dbg !11
  %1258 = extractelement <4 x float> %1009, i64 3, !dbg !11
  %1259 = select i1 %.not, float %1256, float %1255, !dbg !11
  %1260 = bitcast float %1259 to i32, !dbg !11
  %1261 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %477, i32 %1260), !dbg !11
  %1262 = bitcast i32 %1261 to float, !dbg !11
  %1263 = select i1 %.not, float %1255, float %1262, !dbg !11
  %1264 = select i1 %.not, float %1262, float %1256, !dbg !11
  %1265 = select i1 %.not, float %1258, float %1257, !dbg !11
  %1266 = bitcast float %1265 to i32, !dbg !11
  %1267 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %477, i32 %1266), !dbg !11
  %1268 = bitcast i32 %1267 to float, !dbg !11
  %1269 = select i1 %.not, float %1257, float %1268, !dbg !11
  %1270 = select i1 %.not, float %1268, float %1258, !dbg !11
  %1271 = select i1 %.not36, float %1269, float %1263, !dbg !11
  %1272 = bitcast float %1271 to i32, !dbg !11
  %1273 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %485, i32 %1272), !dbg !11
  %1274 = bitcast i32 %1273 to float, !dbg !11
  %1275 = select i1 %.not36, float %1270, float %1264, !dbg !11
  %1276 = bitcast float %1275 to i32, !dbg !11
  %1277 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %485, i32 %1276), !dbg !11
  %1278 = bitcast i32 %1277 to float, !dbg !11
  %1279 = insertelement <4 x float> poison, float %1263, i64 0, !dbg !11
  %1280 = insertelement <4 x float> %1279, float %1264, i64 1, !dbg !11
  %1281 = insertelement <4 x float> %1280, float %1274, i64 2, !dbg !11
  %1282 = insertelement <4 x float> %1281, float %1278, i64 3, !dbg !11
  %1283 = insertelement <4 x float> poison, float %1274, i64 0, !dbg !11
  %1284 = insertelement <4 x float> %1283, float %1278, i64 1, !dbg !11
  %1285 = insertelement <4 x float> %1284, float %1269, i64 2, !dbg !11
  %1286 = insertelement <4 x float> %1285, float %1270, i64 3, !dbg !11
  %1287 = select <4 x i1> %1212, <4 x float> %1282, <4 x float> %1286, !dbg !11
  %1288 = extractelement <4 x float> %933, i64 0, !dbg !11
  %1289 = extractelement <4 x float> %933, i64 1, !dbg !11
  %1290 = extractelement <4 x float> %933, i64 2, !dbg !11
  %1291 = extractelement <4 x float> %933, i64 3, !dbg !11
  %1292 = select i1 %.not, float %1289, float %1288, !dbg !11
  %1293 = bitcast float %1292 to i32, !dbg !11
  %1294 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %477, i32 %1293), !dbg !11
  %1295 = bitcast i32 %1294 to float, !dbg !11
  %1296 = select i1 %.not, float %1288, float %1295, !dbg !11
  %1297 = select i1 %.not, float %1295, float %1289, !dbg !11
  %1298 = select i1 %.not, float %1291, float %1290, !dbg !11
  %1299 = bitcast float %1298 to i32, !dbg !11
  %1300 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %477, i32 %1299), !dbg !11
  %1301 = bitcast i32 %1300 to float, !dbg !11
  %1302 = select i1 %.not, float %1290, float %1301, !dbg !11
  %1303 = select i1 %.not, float %1301, float %1291, !dbg !11
  %1304 = select i1 %.not36, float %1302, float %1296, !dbg !11
  %1305 = bitcast float %1304 to i32, !dbg !11
  %1306 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %485, i32 %1305), !dbg !11
  %1307 = select i1 %.not36, float %1303, float %1297, !dbg !11
  %1308 = bitcast float %1307 to i32, !dbg !11
  %1309 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %485, i32 %1308), !dbg !11
  %1310 = shufflevector <1 x float> %849, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !28
  %1311 = extractelement <4 x float> %1161, i64 0, !dbg !11
  %1312 = extractelement <4 x float> %1161, i64 1, !dbg !11
  %1313 = extractelement <4 x float> %1161, i64 2, !dbg !11
  %1314 = extractelement <4 x float> %1161, i64 3, !dbg !11
  %1315 = select i1 %.not, float %1312, float %1311, !dbg !11
  %1316 = bitcast float %1315 to i32, !dbg !11
  %1317 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %477, i32 %1316), !dbg !11
  %1318 = bitcast i32 %1317 to float, !dbg !11
  %1319 = select i1 %.not, float %1311, float %1318, !dbg !11
  %1320 = select i1 %.not, float %1318, float %1312, !dbg !11
  %1321 = select i1 %.not, float %1314, float %1313, !dbg !11
  %1322 = bitcast float %1321 to i32, !dbg !11
  %1323 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %477, i32 %1322), !dbg !11
  %1324 = bitcast i32 %1323 to float, !dbg !11
  %1325 = select i1 %.not, float %1313, float %1324, !dbg !11
  %1326 = select i1 %.not, float %1324, float %1314, !dbg !11
  %1327 = select i1 %.not36, float %1325, float %1319, !dbg !11
  %1328 = bitcast float %1327 to i32, !dbg !11
  %1329 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %485, i32 %1328), !dbg !11
  %1330 = bitcast i32 %1329 to float, !dbg !11
  %1331 = select i1 %.not36, float %1326, float %1320, !dbg !11
  %1332 = bitcast float %1331 to i32, !dbg !11
  %1333 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %485, i32 %1332), !dbg !11
  %1334 = bitcast i32 %1333 to float, !dbg !11
  %1335 = insertelement <4 x float> poison, float %1319, i64 0, !dbg !11
  %1336 = insertelement <4 x float> %1335, float %1320, i64 1, !dbg !11
  %1337 = insertelement <4 x float> %1336, float %1330, i64 2, !dbg !11
  %1338 = insertelement <4 x float> %1337, float %1334, i64 3, !dbg !11
  %1339 = insertelement <4 x float> poison, float %1330, i64 0, !dbg !11
  %1340 = insertelement <4 x float> %1339, float %1334, i64 1, !dbg !11
  %1341 = insertelement <4 x float> %1340, float %1325, i64 2, !dbg !11
  %1342 = insertelement <4 x float> %1341, float %1326, i64 3, !dbg !11
  %1343 = select <4 x i1> %1212, <4 x float> %1338, <4 x float> %1342, !dbg !11
  %1344 = extractelement <4 x float> %1087, i64 0, !dbg !11
  %1345 = extractelement <4 x float> %1087, i64 1, !dbg !11
  %1346 = extractelement <4 x float> %1087, i64 2, !dbg !11
  %1347 = extractelement <4 x float> %1087, i64 3, !dbg !11
  %1348 = select i1 %.not, float %1345, float %1344, !dbg !11
  %1349 = bitcast float %1348 to i32, !dbg !11
  %1350 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %477, i32 %1349), !dbg !11
  %1351 = bitcast i32 %1350 to float, !dbg !11
  %1352 = select i1 %.not, float %1344, float %1351, !dbg !11
  %1353 = select i1 %.not, float %1351, float %1345, !dbg !11
  %1354 = select i1 %.not, float %1347, float %1346, !dbg !11
  %1355 = bitcast float %1354 to i32, !dbg !11
  %1356 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %477, i32 %1355), !dbg !11
  %1357 = bitcast i32 %1356 to float, !dbg !11
  %1358 = select i1 %.not, float %1346, float %1357, !dbg !11
  %1359 = select i1 %.not, float %1357, float %1347, !dbg !11
  %1360 = select i1 %.not36, float %1358, float %1352, !dbg !11
  %1361 = bitcast float %1360 to i32, !dbg !11
  %1362 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %485, i32 %1361), !dbg !11
  %1363 = bitcast i32 %1362 to float, !dbg !11
  %1364 = select i1 %.not36, float %1359, float %1353, !dbg !11
  %1365 = bitcast float %1364 to i32, !dbg !11
  %1366 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %485, i32 %1365), !dbg !11
  %1367 = bitcast i32 %1366 to float, !dbg !11
  %1368 = insertelement <4 x float> poison, float %1352, i64 0, !dbg !11
  %1369 = insertelement <4 x float> %1368, float %1353, i64 1, !dbg !11
  %1370 = insertelement <4 x float> %1369, float %1363, i64 2, !dbg !11
  %1371 = insertelement <4 x float> %1370, float %1367, i64 3, !dbg !11
  %1372 = insertelement <4 x float> poison, float %1363, i64 0, !dbg !11
  %1373 = insertelement <4 x float> %1372, float %1367, i64 1, !dbg !11
  %1374 = insertelement <4 x float> %1373, float %1358, i64 2, !dbg !11
  %1375 = insertelement <4 x float> %1374, float %1359, i64 3, !dbg !11
  %1376 = select <4 x i1> %1212, <4 x float> %1371, <4 x float> %1375, !dbg !11
  %1377 = extractelement <4 x float> %1011, i64 0, !dbg !11
  %1378 = extractelement <4 x float> %1011, i64 1, !dbg !11
  %1379 = extractelement <4 x float> %1011, i64 2, !dbg !11
  %1380 = extractelement <4 x float> %1011, i64 3, !dbg !11
  %1381 = select i1 %.not, float %1378, float %1377, !dbg !11
  %1382 = bitcast float %1381 to i32, !dbg !11
  %1383 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %477, i32 %1382), !dbg !11
  %1384 = bitcast i32 %1383 to float, !dbg !11
  %1385 = select i1 %.not, float %1377, float %1384, !dbg !11
  %1386 = select i1 %.not, float %1384, float %1378, !dbg !11
  %1387 = select i1 %.not, float %1380, float %1379, !dbg !11
  %1388 = bitcast float %1387 to i32, !dbg !11
  %1389 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %477, i32 %1388), !dbg !11
  %1390 = bitcast i32 %1389 to float, !dbg !11
  %1391 = select i1 %.not, float %1379, float %1390, !dbg !11
  %1392 = select i1 %.not, float %1390, float %1380, !dbg !11
  %1393 = select i1 %.not36, float %1391, float %1385, !dbg !11
  %1394 = bitcast float %1393 to i32, !dbg !11
  %1395 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %485, i32 %1394), !dbg !11
  %1396 = bitcast i32 %1395 to float, !dbg !11
  %1397 = select i1 %.not36, float %1392, float %1386, !dbg !11
  %1398 = bitcast float %1397 to i32, !dbg !11
  %1399 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %485, i32 %1398), !dbg !11
  %1400 = bitcast i32 %1399 to float, !dbg !11
  %1401 = insertelement <4 x float> poison, float %1385, i64 0, !dbg !11
  %1402 = insertelement <4 x float> %1401, float %1386, i64 1, !dbg !11
  %1403 = insertelement <4 x float> %1402, float %1396, i64 2, !dbg !11
  %1404 = insertelement <4 x float> %1403, float %1400, i64 3, !dbg !11
  %1405 = insertelement <4 x float> poison, float %1396, i64 0, !dbg !11
  %1406 = insertelement <4 x float> %1405, float %1400, i64 1, !dbg !11
  %1407 = insertelement <4 x float> %1406, float %1391, i64 2, !dbg !11
  %1408 = insertelement <4 x float> %1407, float %1392, i64 3, !dbg !11
  %1409 = select <4 x i1> %1212, <4 x float> %1404, <4 x float> %1408, !dbg !11
  %1410 = extractelement <4 x float> %935, i64 0, !dbg !11
  %1411 = extractelement <4 x float> %935, i64 1, !dbg !11
  %1412 = extractelement <4 x float> %935, i64 2, !dbg !11
  %1413 = extractelement <4 x float> %935, i64 3, !dbg !11
  %1414 = select i1 %.not, float %1411, float %1410, !dbg !11
  %1415 = bitcast float %1414 to i32, !dbg !11
  %1416 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %477, i32 %1415), !dbg !11
  %1417 = bitcast i32 %1416 to float, !dbg !11
  %1418 = select i1 %.not, float %1410, float %1417, !dbg !11
  %1419 = select i1 %.not, float %1417, float %1411, !dbg !11
  %1420 = select i1 %.not, float %1413, float %1412, !dbg !11
  %1421 = bitcast float %1420 to i32, !dbg !11
  %1422 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %477, i32 %1421), !dbg !11
  %1423 = bitcast i32 %1422 to float, !dbg !11
  %1424 = select i1 %.not, float %1412, float %1423, !dbg !11
  %1425 = select i1 %.not, float %1423, float %1413, !dbg !11
  %1426 = select i1 %.not36, float %1424, float %1418, !dbg !11
  %1427 = bitcast float %1426 to i32, !dbg !11
  %1428 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %485, i32 %1427), !dbg !11
  %1429 = select i1 %.not36, float %1425, float %1419, !dbg !11
  %1430 = bitcast float %1429 to i32, !dbg !11
  %1431 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %485, i32 %1430), !dbg !11
  %1432 = extractelement <4 x float> %1162, i64 0, !dbg !11
  %1433 = extractelement <4 x float> %1162, i64 1, !dbg !11
  %1434 = extractelement <4 x float> %1162, i64 2, !dbg !11
  %1435 = extractelement <4 x float> %1162, i64 3, !dbg !11
  %1436 = select i1 %.not, float %1433, float %1432, !dbg !11
  %1437 = bitcast float %1436 to i32, !dbg !11
  %1438 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %477, i32 %1437), !dbg !11
  %1439 = bitcast i32 %1438 to float, !dbg !11
  %1440 = select i1 %.not, float %1432, float %1439, !dbg !11
  %1441 = select i1 %.not, float %1439, float %1433, !dbg !11
  %1442 = select i1 %.not, float %1435, float %1434, !dbg !11
  %1443 = bitcast float %1442 to i32, !dbg !11
  %1444 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %477, i32 %1443), !dbg !11
  %1445 = bitcast i32 %1444 to float, !dbg !11
  %1446 = select i1 %.not, float %1434, float %1445, !dbg !11
  %1447 = select i1 %.not, float %1445, float %1435, !dbg !11
  %1448 = select i1 %.not36, float %1446, float %1440, !dbg !11
  %1449 = bitcast float %1448 to i32, !dbg !11
  %1450 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %485, i32 %1449), !dbg !11
  %1451 = bitcast i32 %1450 to float, !dbg !11
  %1452 = select i1 %.not36, float %1447, float %1441, !dbg !11
  %1453 = bitcast float %1452 to i32, !dbg !11
  %1454 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %485, i32 %1453), !dbg !11
  %1455 = bitcast i32 %1454 to float, !dbg !11
  %1456 = insertelement <4 x float> poison, float %1440, i64 0, !dbg !11
  %1457 = insertelement <4 x float> %1456, float %1441, i64 1, !dbg !11
  %1458 = insertelement <4 x float> %1457, float %1451, i64 2, !dbg !11
  %1459 = insertelement <4 x float> %1458, float %1455, i64 3, !dbg !11
  %1460 = insertelement <4 x float> poison, float %1451, i64 0, !dbg !11
  %1461 = insertelement <4 x float> %1460, float %1455, i64 1, !dbg !11
  %1462 = insertelement <4 x float> %1461, float %1446, i64 2, !dbg !11
  %1463 = insertelement <4 x float> %1462, float %1447, i64 3, !dbg !11
  %1464 = select <4 x i1> %1212, <4 x float> %1459, <4 x float> %1463, !dbg !11
  %1465 = extractelement <4 x float> %1088, i64 0, !dbg !11
  %1466 = extractelement <4 x float> %1088, i64 1, !dbg !11
  %1467 = extractelement <4 x float> %1088, i64 2, !dbg !11
  %1468 = extractelement <4 x float> %1088, i64 3, !dbg !11
  %1469 = select i1 %.not, float %1466, float %1465, !dbg !11
  %1470 = bitcast float %1469 to i32, !dbg !11
  %1471 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %477, i32 %1470), !dbg !11
  %1472 = bitcast i32 %1471 to float, !dbg !11
  %1473 = select i1 %.not, float %1465, float %1472, !dbg !11
  %1474 = select i1 %.not, float %1472, float %1466, !dbg !11
  %1475 = select i1 %.not, float %1468, float %1467, !dbg !11
  %1476 = bitcast float %1475 to i32, !dbg !11
  %1477 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %477, i32 %1476), !dbg !11
  %1478 = bitcast i32 %1477 to float, !dbg !11
  %1479 = select i1 %.not, float %1467, float %1478, !dbg !11
  %1480 = select i1 %.not, float %1478, float %1468, !dbg !11
  %1481 = select i1 %.not36, float %1479, float %1473, !dbg !11
  %1482 = bitcast float %1481 to i32, !dbg !11
  %1483 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %485, i32 %1482), !dbg !11
  %1484 = bitcast i32 %1483 to float, !dbg !11
  %1485 = select i1 %.not36, float %1480, float %1474, !dbg !11
  %1486 = bitcast float %1485 to i32, !dbg !11
  %1487 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %485, i32 %1486), !dbg !11
  %1488 = bitcast i32 %1487 to float, !dbg !11
  %1489 = insertelement <4 x float> poison, float %1473, i64 0, !dbg !11
  %1490 = insertelement <4 x float> %1489, float %1474, i64 1, !dbg !11
  %1491 = insertelement <4 x float> %1490, float %1484, i64 2, !dbg !11
  %1492 = insertelement <4 x float> %1491, float %1488, i64 3, !dbg !11
  %1493 = insertelement <4 x float> poison, float %1484, i64 0, !dbg !11
  %1494 = insertelement <4 x float> %1493, float %1488, i64 1, !dbg !11
  %1495 = insertelement <4 x float> %1494, float %1479, i64 2, !dbg !11
  %1496 = insertelement <4 x float> %1495, float %1480, i64 3, !dbg !11
  %1497 = select <4 x i1> %1212, <4 x float> %1492, <4 x float> %1496, !dbg !11
  %1498 = extractelement <4 x float> %1012, i64 0, !dbg !11
  %1499 = extractelement <4 x float> %1012, i64 1, !dbg !11
  %1500 = extractelement <4 x float> %1012, i64 2, !dbg !11
  %1501 = extractelement <4 x float> %1012, i64 3, !dbg !11
  %1502 = select i1 %.not, float %1499, float %1498, !dbg !11
  %1503 = bitcast float %1502 to i32, !dbg !11
  %1504 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %477, i32 %1503), !dbg !11
  %1505 = bitcast i32 %1504 to float, !dbg !11
  %1506 = select i1 %.not, float %1498, float %1505, !dbg !11
  %1507 = select i1 %.not, float %1505, float %1499, !dbg !11
  %1508 = select i1 %.not, float %1501, float %1500, !dbg !11
  %1509 = bitcast float %1508 to i32, !dbg !11
  %1510 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %477, i32 %1509), !dbg !11
  %1511 = bitcast i32 %1510 to float, !dbg !11
  %1512 = select i1 %.not, float %1500, float %1511, !dbg !11
  %1513 = select i1 %.not, float %1511, float %1501, !dbg !11
  %1514 = select i1 %.not36, float %1512, float %1506, !dbg !11
  %1515 = bitcast float %1514 to i32, !dbg !11
  %1516 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %485, i32 %1515), !dbg !11
  %1517 = bitcast i32 %1516 to float, !dbg !11
  %1518 = select i1 %.not36, float %1513, float %1507, !dbg !11
  %1519 = bitcast float %1518 to i32, !dbg !11
  %1520 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %485, i32 %1519), !dbg !11
  %1521 = bitcast i32 %1520 to float, !dbg !11
  %1522 = insertelement <4 x float> poison, float %1506, i64 0, !dbg !11
  %1523 = insertelement <4 x float> %1522, float %1507, i64 1, !dbg !11
  %1524 = insertelement <4 x float> %1523, float %1517, i64 2, !dbg !11
  %1525 = insertelement <4 x float> %1524, float %1521, i64 3, !dbg !11
  %1526 = insertelement <4 x float> poison, float %1517, i64 0, !dbg !11
  %1527 = insertelement <4 x float> %1526, float %1521, i64 1, !dbg !11
  %1528 = insertelement <4 x float> %1527, float %1512, i64 2, !dbg !11
  %1529 = insertelement <4 x float> %1528, float %1513, i64 3, !dbg !11
  %1530 = select <4 x i1> %1212, <4 x float> %1525, <4 x float> %1529, !dbg !11
  %1531 = extractelement <4 x float> %936, i64 0, !dbg !11
  %1532 = extractelement <4 x float> %936, i64 1, !dbg !11
  %1533 = extractelement <4 x float> %936, i64 2, !dbg !11
  %1534 = extractelement <4 x float> %936, i64 3, !dbg !11
  %1535 = select i1 %.not, float %1532, float %1531, !dbg !11
  %1536 = bitcast float %1535 to i32, !dbg !11
  %1537 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %477, i32 %1536), !dbg !11
  %1538 = bitcast i32 %1537 to float, !dbg !11
  %1539 = select i1 %.not, float %1531, float %1538, !dbg !11
  %1540 = select i1 %.not, float %1538, float %1532, !dbg !11
  %1541 = select i1 %.not, float %1534, float %1533, !dbg !11
  %1542 = bitcast float %1541 to i32, !dbg !11
  %1543 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %477, i32 %1542), !dbg !11
  %1544 = bitcast i32 %1543 to float, !dbg !11
  %1545 = select i1 %.not, float %1533, float %1544, !dbg !11
  %1546 = select i1 %.not, float %1544, float %1534, !dbg !11
  %1547 = select i1 %.not36, float %1545, float %1539, !dbg !11
  %1548 = bitcast float %1547 to i32, !dbg !11
  %1549 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %485, i32 %1548), !dbg !11
  %1550 = select i1 %.not36, float %1546, float %1540, !dbg !11
  %1551 = bitcast float %1550 to i32, !dbg !11
  %1552 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %485, i32 %1551), !dbg !11
  %1553 = shufflevector <1 x float> %860, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !28
  %1554 = extractelement <4 x float> %1163, i64 0, !dbg !11
  %1555 = extractelement <4 x float> %1163, i64 1, !dbg !11
  %1556 = extractelement <4 x float> %1163, i64 2, !dbg !11
  %1557 = extractelement <4 x float> %1163, i64 3, !dbg !11
  %1558 = select i1 %.not, float %1555, float %1554, !dbg !11
  %1559 = bitcast float %1558 to i32, !dbg !11
  %1560 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %477, i32 %1559), !dbg !11
  %1561 = bitcast i32 %1560 to float, !dbg !11
  %1562 = select i1 %.not, float %1554, float %1561, !dbg !11
  %1563 = select i1 %.not, float %1561, float %1555, !dbg !11
  %1564 = select i1 %.not, float %1557, float %1556, !dbg !11
  %1565 = bitcast float %1564 to i32, !dbg !11
  %1566 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %477, i32 %1565), !dbg !11
  %1567 = bitcast i32 %1566 to float, !dbg !11
  %1568 = select i1 %.not, float %1556, float %1567, !dbg !11
  %1569 = select i1 %.not, float %1567, float %1557, !dbg !11
  %1570 = select i1 %.not36, float %1568, float %1562, !dbg !11
  %1571 = bitcast float %1570 to i32, !dbg !11
  %1572 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %485, i32 %1571), !dbg !11
  %1573 = bitcast i32 %1572 to float, !dbg !11
  %1574 = select i1 %.not36, float %1569, float %1563, !dbg !11
  %1575 = bitcast float %1574 to i32, !dbg !11
  %1576 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %485, i32 %1575), !dbg !11
  %1577 = bitcast i32 %1576 to float, !dbg !11
  %1578 = insertelement <4 x float> poison, float %1562, i64 0, !dbg !11
  %1579 = insertelement <4 x float> %1578, float %1563, i64 1, !dbg !11
  %1580 = insertelement <4 x float> %1579, float %1573, i64 2, !dbg !11
  %1581 = insertelement <4 x float> %1580, float %1577, i64 3, !dbg !11
  %1582 = insertelement <4 x float> poison, float %1573, i64 0, !dbg !11
  %1583 = insertelement <4 x float> %1582, float %1577, i64 1, !dbg !11
  %1584 = insertelement <4 x float> %1583, float %1568, i64 2, !dbg !11
  %1585 = insertelement <4 x float> %1584, float %1569, i64 3, !dbg !11
  %1586 = select <4 x i1> %1212, <4 x float> %1581, <4 x float> %1585, !dbg !11
  %1587 = extractelement <4 x float> %1089, i64 0, !dbg !11
  %1588 = extractelement <4 x float> %1089, i64 1, !dbg !11
  %1589 = extractelement <4 x float> %1089, i64 2, !dbg !11
  %1590 = extractelement <4 x float> %1089, i64 3, !dbg !11
  %1591 = select i1 %.not, float %1588, float %1587, !dbg !11
  %1592 = bitcast float %1591 to i32, !dbg !11
  %1593 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %477, i32 %1592), !dbg !11
  %1594 = bitcast i32 %1593 to float, !dbg !11
  %1595 = select i1 %.not, float %1587, float %1594, !dbg !11
  %1596 = select i1 %.not, float %1594, float %1588, !dbg !11
  %1597 = select i1 %.not, float %1590, float %1589, !dbg !11
  %1598 = bitcast float %1597 to i32, !dbg !11
  %1599 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %477, i32 %1598), !dbg !11
  %1600 = bitcast i32 %1599 to float, !dbg !11
  %1601 = select i1 %.not, float %1589, float %1600, !dbg !11
  %1602 = select i1 %.not, float %1600, float %1590, !dbg !11
  %1603 = select i1 %.not36, float %1601, float %1595, !dbg !11
  %1604 = bitcast float %1603 to i32, !dbg !11
  %1605 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %485, i32 %1604), !dbg !11
  %1606 = bitcast i32 %1605 to float, !dbg !11
  %1607 = select i1 %.not36, float %1602, float %1596, !dbg !11
  %1608 = bitcast float %1607 to i32, !dbg !11
  %1609 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %485, i32 %1608), !dbg !11
  %1610 = bitcast i32 %1609 to float, !dbg !11
  %1611 = insertelement <4 x float> poison, float %1595, i64 0, !dbg !11
  %1612 = insertelement <4 x float> %1611, float %1596, i64 1, !dbg !11
  %1613 = insertelement <4 x float> %1612, float %1606, i64 2, !dbg !11
  %1614 = insertelement <4 x float> %1613, float %1610, i64 3, !dbg !11
  %1615 = insertelement <4 x float> poison, float %1606, i64 0, !dbg !11
  %1616 = insertelement <4 x float> %1615, float %1610, i64 1, !dbg !11
  %1617 = insertelement <4 x float> %1616, float %1601, i64 2, !dbg !11
  %1618 = insertelement <4 x float> %1617, float %1602, i64 3, !dbg !11
  %1619 = select <4 x i1> %1212, <4 x float> %1614, <4 x float> %1618, !dbg !11
  %1620 = extractelement <4 x float> %1013, i64 0, !dbg !11
  %1621 = extractelement <4 x float> %1013, i64 1, !dbg !11
  %1622 = extractelement <4 x float> %1013, i64 2, !dbg !11
  %1623 = extractelement <4 x float> %1013, i64 3, !dbg !11
  %1624 = select i1 %.not, float %1621, float %1620, !dbg !11
  %1625 = bitcast float %1624 to i32, !dbg !11
  %1626 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %477, i32 %1625), !dbg !11
  %1627 = bitcast i32 %1626 to float, !dbg !11
  %1628 = select i1 %.not, float %1620, float %1627, !dbg !11
  %1629 = select i1 %.not, float %1627, float %1621, !dbg !11
  %1630 = select i1 %.not, float %1623, float %1622, !dbg !11
  %1631 = bitcast float %1630 to i32, !dbg !11
  %1632 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %477, i32 %1631), !dbg !11
  %1633 = bitcast i32 %1632 to float, !dbg !11
  %1634 = select i1 %.not, float %1622, float %1633, !dbg !11
  %1635 = select i1 %.not, float %1633, float %1623, !dbg !11
  %1636 = select i1 %.not36, float %1634, float %1628, !dbg !11
  %1637 = bitcast float %1636 to i32, !dbg !11
  %1638 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %485, i32 %1637), !dbg !11
  %1639 = bitcast i32 %1638 to float, !dbg !11
  %1640 = select i1 %.not36, float %1635, float %1629, !dbg !11
  %1641 = bitcast float %1640 to i32, !dbg !11
  %1642 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %485, i32 %1641), !dbg !11
  %1643 = bitcast i32 %1642 to float, !dbg !11
  %1644 = insertelement <4 x float> poison, float %1628, i64 0, !dbg !11
  %1645 = insertelement <4 x float> %1644, float %1629, i64 1, !dbg !11
  %1646 = insertelement <4 x float> %1645, float %1639, i64 2, !dbg !11
  %1647 = insertelement <4 x float> %1646, float %1643, i64 3, !dbg !11
  %1648 = insertelement <4 x float> poison, float %1639, i64 0, !dbg !11
  %1649 = insertelement <4 x float> %1648, float %1643, i64 1, !dbg !11
  %1650 = insertelement <4 x float> %1649, float %1634, i64 2, !dbg !11
  %1651 = insertelement <4 x float> %1650, float %1635, i64 3, !dbg !11
  %1652 = select <4 x i1> %1212, <4 x float> %1647, <4 x float> %1651, !dbg !11
  %1653 = extractelement <4 x float> %937, i64 0, !dbg !11
  %1654 = extractelement <4 x float> %937, i64 1, !dbg !11
  %1655 = extractelement <4 x float> %937, i64 2, !dbg !11
  %1656 = extractelement <4 x float> %937, i64 3, !dbg !11
  %1657 = select i1 %.not, float %1654, float %1653, !dbg !11
  %1658 = bitcast float %1657 to i32, !dbg !11
  %1659 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %477, i32 %1658), !dbg !11
  %1660 = bitcast i32 %1659 to float, !dbg !11
  %1661 = select i1 %.not, float %1653, float %1660, !dbg !11
  %1662 = select i1 %.not, float %1660, float %1654, !dbg !11
  %1663 = select i1 %.not, float %1656, float %1655, !dbg !11
  %1664 = bitcast float %1663 to i32, !dbg !11
  %1665 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %477, i32 %1664), !dbg !11
  %1666 = bitcast i32 %1665 to float, !dbg !11
  %1667 = select i1 %.not, float %1655, float %1666, !dbg !11
  %1668 = select i1 %.not, float %1666, float %1656, !dbg !11
  %1669 = select i1 %.not36, float %1667, float %1661, !dbg !11
  %1670 = bitcast float %1669 to i32, !dbg !11
  %1671 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %485, i32 %1670), !dbg !11
  %1672 = select i1 %.not36, float %1668, float %1662, !dbg !11
  %1673 = bitcast float %1672 to i32, !dbg !11
  %1674 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %485, i32 %1673), !dbg !11
  %1675 = bitcast i32 %1671 to float, !dbg !11
  %1676 = bitcast i32 %1674 to float, !dbg !11
  %1677 = insertelement <4 x float> poison, float %1661, i64 0, !dbg !11
  %1678 = insertelement <4 x float> %1677, float %1662, i64 1, !dbg !11
  %1679 = insertelement <4 x float> %1678, float %1675, i64 2, !dbg !11
  %1680 = insertelement <4 x float> %1679, float %1676, i64 3, !dbg !11
  %1681 = insertelement <4 x float> poison, float %1675, i64 0, !dbg !11
  %1682 = insertelement <4 x float> %1681, float %1676, i64 1, !dbg !11
  %1683 = insertelement <4 x float> %1682, float %1667, i64 2, !dbg !11
  %1684 = insertelement <4 x float> %1683, float %1668, i64 3, !dbg !11
  %1685 = select <4 x i1> %1212, <4 x float> %1680, <4 x float> %1684, !dbg !11
  %row.reciprocal = fdiv arcp <4 x float> splat (float 1.000000e+00), %1553, !dbg !28
  %normalized = fmul <4 x float> %row.reciprocal, %1685, !dbg !28
  %1686 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !29
  %1687 = extractelement <4 x half> %1686, i64 3, !dbg !29
  %1688 = extractelement <4 x half> %1686, i64 2, !dbg !29
  %1689 = extractelement <4 x half> %1686, i64 1, !dbg !29
  %1690 = extractelement <4 x half> %1686, i64 0, !dbg !29
  %1691 = bitcast i32 %1549 to float, !dbg !11
  %1692 = bitcast i32 %1552 to float, !dbg !11
  %1693 = insertelement <4 x float> poison, float %1539, i64 0, !dbg !11
  %1694 = insertelement <4 x float> %1693, float %1540, i64 1, !dbg !11
  %1695 = insertelement <4 x float> %1694, float %1691, i64 2, !dbg !11
  %1696 = insertelement <4 x float> %1695, float %1692, i64 3, !dbg !11
  %1697 = insertelement <4 x float> poison, float %1691, i64 0, !dbg !11
  %1698 = insertelement <4 x float> %1697, float %1692, i64 1, !dbg !11
  %1699 = insertelement <4 x float> %1698, float %1545, i64 2, !dbg !11
  %1700 = insertelement <4 x float> %1699, float %1546, i64 3, !dbg !11
  %1701 = select <4 x i1> %1212, <4 x float> %1696, <4 x float> %1700, !dbg !11
  %normalized37 = fmul <4 x float> %row.reciprocal, %1701, !dbg !28
  %1702 = fptrunc <4 x float> %normalized37 to <4 x half>, !dbg !29
  %1703 = extractelement <4 x half> %1702, i64 3, !dbg !29
  %1704 = extractelement <4 x half> %1702, i64 2, !dbg !29
  %1705 = extractelement <4 x half> %1702, i64 1, !dbg !29
  %1706 = extractelement <4 x half> %1702, i64 0, !dbg !29
  %1707 = bitcast i32 %1428 to float, !dbg !11
  %1708 = bitcast i32 %1431 to float, !dbg !11
  %1709 = insertelement <4 x float> poison, float %1418, i64 0, !dbg !11
  %1710 = insertelement <4 x float> %1709, float %1419, i64 1, !dbg !11
  %1711 = insertelement <4 x float> %1710, float %1707, i64 2, !dbg !11
  %1712 = insertelement <4 x float> %1711, float %1708, i64 3, !dbg !11
  %1713 = insertelement <4 x float> poison, float %1707, i64 0, !dbg !11
  %1714 = insertelement <4 x float> %1713, float %1708, i64 1, !dbg !11
  %1715 = insertelement <4 x float> %1714, float %1424, i64 2, !dbg !11
  %1716 = insertelement <4 x float> %1715, float %1425, i64 3, !dbg !11
  %1717 = select <4 x i1> %1212, <4 x float> %1712, <4 x float> %1716, !dbg !11
  %row.reciprocal38 = fdiv arcp <4 x float> splat (float 1.000000e+00), %1310, !dbg !28
  %normalized39 = fmul <4 x float> %row.reciprocal38, %1717, !dbg !28
  %1718 = fptrunc <4 x float> %normalized39 to <4 x half>, !dbg !29
  %1719 = extractelement <4 x half> %1718, i64 3, !dbg !29
  %1720 = extractelement <4 x half> %1718, i64 2, !dbg !29
  %1721 = extractelement <4 x half> %1718, i64 1, !dbg !29
  %1722 = extractelement <4 x half> %1718, i64 0, !dbg !29
  %1723 = bitcast i32 %1306 to float, !dbg !11
  %1724 = bitcast i32 %1309 to float, !dbg !11
  %1725 = insertelement <4 x float> poison, float %1296, i64 0, !dbg !11
  %1726 = insertelement <4 x float> %1725, float %1297, i64 1, !dbg !11
  %1727 = insertelement <4 x float> %1726, float %1723, i64 2, !dbg !11
  %1728 = insertelement <4 x float> %1727, float %1724, i64 3, !dbg !11
  %1729 = insertelement <4 x float> poison, float %1723, i64 0, !dbg !11
  %1730 = insertelement <4 x float> %1729, float %1724, i64 1, !dbg !11
  %1731 = insertelement <4 x float> %1730, float %1302, i64 2, !dbg !11
  %1732 = insertelement <4 x float> %1731, float %1303, i64 3, !dbg !11
  %1733 = select <4 x i1> %1212, <4 x float> %1728, <4 x float> %1732, !dbg !11
  %normalized40 = fmul <4 x float> %row.reciprocal38, %1733, !dbg !28
  %1734 = fptrunc <4 x float> %normalized40 to <4 x half>, !dbg !29
  %1735 = extractelement <4 x half> %1734, i64 3, !dbg !29
  %1736 = extractelement <4 x half> %1734, i64 2, !dbg !29
  %1737 = extractelement <4 x half> %1734, i64 1, !dbg !29
  %1738 = extractelement <4 x half> %1734, i64 0, !dbg !29
  %address.thread841 = tail call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #8, !dbg !30, !range !9, !frisk.register_lifetimes !5
  %address.local842 = zext nneg i32 %address.thread841 to i64, !dbg !8
  %1739 = lshr i64 %address.local842, 2, !dbg !30
  %address.local845 = and i64 %1739, 48, !dbg !30
  %address.local847 = and i64 %address.local842, 240, !dbg !30
  %address.local846 = add i64 %27, %address.local842, !dbg !30
  %address.local848 = sub i64 %address.local846, %address.local847, !dbg !30
  %address.local849 = add i64 %address.local848, %address.local845, !dbg !30
  %1740 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %29, !dbg !30
  %address.local852.idx = shl nuw nsw i64 %address.local849, 8, !dbg !30
  %address.local852 = getelementptr inbounds nuw i8, ptr addrspace(1) %1740, i64 %address.local852.idx, !dbg !30
  %address.local854 = and i64 %1739, 12, !dbg !10
  %address.local855 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local852, i64 %address.local854, !dbg !30
  store half %1738, ptr addrspace(1) %address.local855, align 2, !dbg !30
  %address.local857 = or disjoint i64 %address.local854, 1, !dbg !30
  %address.local858 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local852, i64 %address.local857, !dbg !30
  store half %1737, ptr addrspace(1) %address.local858, align 2, !dbg !30
  %address.local860 = or disjoint i64 %address.local854, 2, !dbg !30
  %address.local861 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local852, i64 %address.local860, !dbg !30
  store half %1736, ptr addrspace(1) %address.local861, align 2, !dbg !30
  %address.local863 = or disjoint i64 %address.local854, 3, !dbg !30
  %address.local864 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local852, i64 %address.local863, !dbg !30
  store half %1735, ptr addrspace(1) %address.local864, align 2, !dbg !30
  %address.local866 = or disjoint i64 %address.local854, 16, !dbg !30
  %address.local867 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local852, i64 %address.local866, !dbg !30
  store half %1722, ptr addrspace(1) %address.local867, align 2, !dbg !30
  %address.local869 = or disjoint i64 %address.local854, 17, !dbg !30
  %address.local870 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local852, i64 %address.local869, !dbg !30
  store half %1721, ptr addrspace(1) %address.local870, align 2, !dbg !30
  %address.local872 = or disjoint i64 %address.local854, 18, !dbg !30
  %address.local873 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local852, i64 %address.local872, !dbg !30
  store half %1720, ptr addrspace(1) %address.local873, align 2, !dbg !30
  %address.local875 = or disjoint i64 %address.local854, 19, !dbg !30
  %address.local876 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local852, i64 %address.local875, !dbg !30
  store half %1719, ptr addrspace(1) %address.local876, align 2, !dbg !30
  %address.local880 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local852, i64 16384, !dbg !30
  %address.local881 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local880, i64 %address.local854, !dbg !30
  store half %1706, ptr addrspace(1) %address.local881, align 2, !dbg !30
  %address.local883 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local880, i64 %address.local857, !dbg !30
  store half %1705, ptr addrspace(1) %address.local883, align 2, !dbg !30
  %address.local885 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local880, i64 %address.local860, !dbg !30
  store half %1704, ptr addrspace(1) %address.local885, align 2, !dbg !30
  %address.local887 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local880, i64 %address.local863, !dbg !30
  store half %1703, ptr addrspace(1) %address.local887, align 2, !dbg !30
  %address.local889 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local880, i64 %address.local866, !dbg !30
  store half %1690, ptr addrspace(1) %address.local889, align 2, !dbg !30
  %address.local891 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local880, i64 %address.local869, !dbg !30
  store half %1689, ptr addrspace(1) %address.local891, align 2, !dbg !30
  %address.local893 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local880, i64 %address.local872, !dbg !30
  store half %1688, ptr addrspace(1) %address.local893, align 2, !dbg !30
  %address.local895 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local880, i64 %address.local875, !dbg !30
  store half %1687, ptr addrspace(1) %address.local895, align 2, !dbg !30
  %normalized42 = fmul <4 x float> %row.reciprocal, %1652, !dbg !28
  %1741 = fptrunc <4 x float> %normalized42 to <4 x half>, !dbg !29
  %1742 = extractelement <4 x half> %1741, i64 3, !dbg !29
  %1743 = extractelement <4 x half> %1741, i64 2, !dbg !29
  %1744 = extractelement <4 x half> %1741, i64 1, !dbg !29
  %1745 = extractelement <4 x half> %1741, i64 0, !dbg !29
  %normalized43 = fmul <4 x float> %row.reciprocal, %1530, !dbg !28
  %1746 = fptrunc <4 x float> %normalized43 to <4 x half>, !dbg !29
  %1747 = extractelement <4 x half> %1746, i64 3, !dbg !29
  %1748 = extractelement <4 x half> %1746, i64 2, !dbg !29
  %1749 = extractelement <4 x half> %1746, i64 1, !dbg !29
  %1750 = extractelement <4 x half> %1746, i64 0, !dbg !29
  %normalized45 = fmul <4 x float> %row.reciprocal38, %1409, !dbg !28
  %1751 = fptrunc <4 x float> %normalized45 to <4 x half>, !dbg !29
  %1752 = extractelement <4 x half> %1751, i64 3, !dbg !29
  %1753 = extractelement <4 x half> %1751, i64 2, !dbg !29
  %1754 = extractelement <4 x half> %1751, i64 1, !dbg !29
  %1755 = extractelement <4 x half> %1751, i64 0, !dbg !29
  %normalized46 = fmul <4 x float> %row.reciprocal38, %1287, !dbg !28
  %1756 = fptrunc <4 x float> %normalized46 to <4 x half>, !dbg !29
  %1757 = extractelement <4 x half> %1756, i64 3, !dbg !29
  %1758 = extractelement <4 x half> %1756, i64 2, !dbg !29
  %1759 = extractelement <4 x half> %1756, i64 1, !dbg !29
  %1760 = extractelement <4 x half> %1756, i64 0, !dbg !29
  %address.local910 = or disjoint i64 %address.local854, 32, !dbg !30
  %address.local911 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local852, i64 %address.local910, !dbg !30
  store half %1760, ptr addrspace(1) %address.local911, align 2, !dbg !30
  %address.local913 = or disjoint i64 %address.local854, 33, !dbg !30
  %address.local914 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local852, i64 %address.local913, !dbg !30
  store half %1759, ptr addrspace(1) %address.local914, align 2, !dbg !30
  %address.local916 = or disjoint i64 %address.local854, 34, !dbg !30
  %address.local917 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local852, i64 %address.local916, !dbg !30
  store half %1758, ptr addrspace(1) %address.local917, align 2, !dbg !30
  %address.local919 = or disjoint i64 %address.local854, 35, !dbg !30
  %address.local920 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local852, i64 %address.local919, !dbg !30
  store half %1757, ptr addrspace(1) %address.local920, align 2, !dbg !30
  %address.local922 = or disjoint i64 %address.local854, 48, !dbg !30
  %address.local923 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local852, i64 %address.local922, !dbg !30
  store half %1755, ptr addrspace(1) %address.local923, align 2, !dbg !30
  %address.local925 = or disjoint i64 %address.local854, 49, !dbg !30
  %address.local926 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local852, i64 %address.local925, !dbg !30
  store half %1754, ptr addrspace(1) %address.local926, align 2, !dbg !30
  %address.local928 = or disjoint i64 %address.local854, 50, !dbg !30
  %address.local929 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local852, i64 %address.local928, !dbg !30
  store half %1753, ptr addrspace(1) %address.local929, align 2, !dbg !30
  %address.local931 = or i64 %1739, 51, !dbg !30
  %address.local932 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local852, i64 %address.local931, !dbg !30
  store half %1752, ptr addrspace(1) %address.local932, align 2, !dbg !30
  %address.local937 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local880, i64 %address.local910, !dbg !30
  store half %1750, ptr addrspace(1) %address.local937, align 2, !dbg !30
  %address.local939 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local880, i64 %address.local913, !dbg !30
  store half %1749, ptr addrspace(1) %address.local939, align 2, !dbg !30
  %address.local941 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local880, i64 %address.local916, !dbg !30
  store half %1748, ptr addrspace(1) %address.local941, align 2, !dbg !30
  %address.local943 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local880, i64 %address.local919, !dbg !30
  store half %1747, ptr addrspace(1) %address.local943, align 2, !dbg !30
  %address.local945 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local880, i64 %address.local922, !dbg !30
  store half %1745, ptr addrspace(1) %address.local945, align 2, !dbg !30
  %address.local947 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local880, i64 %address.local925, !dbg !30
  store half %1744, ptr addrspace(1) %address.local947, align 2, !dbg !30
  %address.local949 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local880, i64 %address.local928, !dbg !30
  store half %1743, ptr addrspace(1) %address.local949, align 2, !dbg !30
  %address.local951 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local880, i64 %address.local931, !dbg !30
  store half %1742, ptr addrspace(1) %address.local951, align 2, !dbg !30
  %normalized48 = fmul <4 x float> %row.reciprocal, %1619, !dbg !28
  %1761 = fptrunc <4 x float> %normalized48 to <4 x half>, !dbg !29
  %1762 = extractelement <4 x half> %1761, i64 3, !dbg !29
  %1763 = extractelement <4 x half> %1761, i64 2, !dbg !29
  %1764 = extractelement <4 x half> %1761, i64 1, !dbg !29
  %1765 = extractelement <4 x half> %1761, i64 0, !dbg !29
  %normalized49 = fmul <4 x float> %row.reciprocal, %1497, !dbg !28
  %1766 = fptrunc <4 x float> %normalized49 to <4 x half>, !dbg !29
  %1767 = extractelement <4 x half> %1766, i64 3, !dbg !29
  %1768 = extractelement <4 x half> %1766, i64 2, !dbg !29
  %1769 = extractelement <4 x half> %1766, i64 1, !dbg !29
  %1770 = extractelement <4 x half> %1766, i64 0, !dbg !29
  %normalized51 = fmul <4 x float> %row.reciprocal38, %1376, !dbg !28
  %1771 = fptrunc <4 x float> %normalized51 to <4 x half>, !dbg !29
  %1772 = extractelement <4 x half> %1771, i64 3, !dbg !29
  %1773 = extractelement <4 x half> %1771, i64 2, !dbg !29
  %1774 = extractelement <4 x half> %1771, i64 1, !dbg !29
  %1775 = extractelement <4 x half> %1771, i64 0, !dbg !29
  %normalized52 = fmul <4 x float> %row.reciprocal38, %1254, !dbg !28
  %1776 = fptrunc <4 x float> %normalized52 to <4 x half>, !dbg !29
  %1777 = extractelement <4 x half> %1776, i64 3, !dbg !29
  %1778 = extractelement <4 x half> %1776, i64 2, !dbg !29
  %1779 = extractelement <4 x half> %1776, i64 1, !dbg !29
  %1780 = extractelement <4 x half> %1776, i64 0, !dbg !29
  %address.local966 = or disjoint i64 %address.local854, 64, !dbg !30
  %address.local967 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local852, i64 %address.local966, !dbg !30
  store half %1780, ptr addrspace(1) %address.local967, align 2, !dbg !30
  %address.local969 = or disjoint i64 %address.local854, 65, !dbg !30
  %address.local970 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local852, i64 %address.local969, !dbg !30
  store half %1779, ptr addrspace(1) %address.local970, align 2, !dbg !30
  %address.local972 = or disjoint i64 %address.local854, 66, !dbg !30
  %address.local973 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local852, i64 %address.local972, !dbg !30
  store half %1778, ptr addrspace(1) %address.local973, align 2, !dbg !30
  %address.local975 = or disjoint i64 %address.local854, 67, !dbg !30
  %address.local976 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local852, i64 %address.local975, !dbg !30
  store half %1777, ptr addrspace(1) %address.local976, align 2, !dbg !30
  %address.local978 = or disjoint i64 %address.local854, 80, !dbg !30
  %address.local979 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local852, i64 %address.local978, !dbg !30
  store half %1775, ptr addrspace(1) %address.local979, align 2, !dbg !30
  %address.local981 = or disjoint i64 %address.local854, 81, !dbg !30
  %address.local982 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local852, i64 %address.local981, !dbg !30
  store half %1774, ptr addrspace(1) %address.local982, align 2, !dbg !30
  %address.local984 = or disjoint i64 %address.local854, 82, !dbg !30
  %address.local985 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local852, i64 %address.local984, !dbg !30
  store half %1773, ptr addrspace(1) %address.local985, align 2, !dbg !30
  %address.local987 = or disjoint i64 %address.local854, 83, !dbg !30
  %address.local988 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local852, i64 %address.local987, !dbg !30
  store half %1772, ptr addrspace(1) %address.local988, align 2, !dbg !30
  %address.local993 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local880, i64 %address.local966, !dbg !30
  store half %1770, ptr addrspace(1) %address.local993, align 2, !dbg !30
  %address.local995 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local880, i64 %address.local969, !dbg !30
  store half %1769, ptr addrspace(1) %address.local995, align 2, !dbg !30
  %address.local997 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local880, i64 %address.local972, !dbg !30
  store half %1768, ptr addrspace(1) %address.local997, align 2, !dbg !30
  %address.local999 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local880, i64 %address.local975, !dbg !30
  store half %1767, ptr addrspace(1) %address.local999, align 2, !dbg !30
  %address.local1001 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local880, i64 %address.local978, !dbg !30
  store half %1765, ptr addrspace(1) %address.local1001, align 2, !dbg !30
  %address.local1003 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local880, i64 %address.local981, !dbg !30
  store half %1764, ptr addrspace(1) %address.local1003, align 2, !dbg !30
  %address.local1005 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local880, i64 %address.local984, !dbg !30
  store half %1763, ptr addrspace(1) %address.local1005, align 2, !dbg !30
  %address.local1007 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local880, i64 %address.local987, !dbg !30
  store half %1762, ptr addrspace(1) %address.local1007, align 2, !dbg !30
  %normalized54 = fmul <4 x float> %row.reciprocal, %1586, !dbg !28
  %1781 = fptrunc <4 x float> %normalized54 to <4 x half>, !dbg !29
  %1782 = extractelement <4 x half> %1781, i64 3, !dbg !29
  %1783 = extractelement <4 x half> %1781, i64 2, !dbg !29
  %1784 = extractelement <4 x half> %1781, i64 1, !dbg !29
  %1785 = extractelement <4 x half> %1781, i64 0, !dbg !29
  %normalized55 = fmul <4 x float> %row.reciprocal, %1464, !dbg !28
  %1786 = fptrunc <4 x float> %normalized55 to <4 x half>, !dbg !29
  %1787 = extractelement <4 x half> %1786, i64 3, !dbg !29
  %1788 = extractelement <4 x half> %1786, i64 2, !dbg !29
  %1789 = extractelement <4 x half> %1786, i64 1, !dbg !29
  %1790 = extractelement <4 x half> %1786, i64 0, !dbg !29
  %normalized57 = fmul <4 x float> %row.reciprocal38, %1343, !dbg !28
  %1791 = fptrunc <4 x float> %normalized57 to <4 x half>, !dbg !29
  %1792 = extractelement <4 x half> %1791, i64 3, !dbg !29
  %1793 = extractelement <4 x half> %1791, i64 2, !dbg !29
  %1794 = extractelement <4 x half> %1791, i64 1, !dbg !29
  %1795 = extractelement <4 x half> %1791, i64 0, !dbg !29
  %normalized58 = fmul <4 x float> %row.reciprocal38, %1221, !dbg !28
  %1796 = fptrunc <4 x float> %normalized58 to <4 x half>, !dbg !29
  %1797 = extractelement <4 x half> %1796, i64 3, !dbg !29
  %1798 = extractelement <4 x half> %1796, i64 2, !dbg !29
  %1799 = extractelement <4 x half> %1796, i64 1, !dbg !29
  %1800 = extractelement <4 x half> %1796, i64 0, !dbg !29
  %address.local1022 = or disjoint i64 %address.local854, 96, !dbg !30
  %address.local1023 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local852, i64 %address.local1022, !dbg !30
  store half %1800, ptr addrspace(1) %address.local1023, align 2, !dbg !30
  %address.local1025 = or disjoint i64 %address.local854, 97, !dbg !30
  %address.local1026 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local852, i64 %address.local1025, !dbg !30
  store half %1799, ptr addrspace(1) %address.local1026, align 2, !dbg !30
  %address.local1028 = or disjoint i64 %address.local854, 98, !dbg !30
  %address.local1029 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local852, i64 %address.local1028, !dbg !30
  store half %1798, ptr addrspace(1) %address.local1029, align 2, !dbg !30
  %address.local1031 = or disjoint i64 %address.local854, 99, !dbg !30
  %address.local1032 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local852, i64 %address.local1031, !dbg !30
  store half %1797, ptr addrspace(1) %address.local1032, align 2, !dbg !30
  %address.local1034 = or disjoint i64 %address.local854, 112, !dbg !30
  %address.local1035 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local852, i64 %address.local1034, !dbg !30
  store half %1795, ptr addrspace(1) %address.local1035, align 2, !dbg !30
  %address.local1037 = or disjoint i64 %address.local854, 113, !dbg !30
  %address.local1038 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local852, i64 %address.local1037, !dbg !30
  store half %1794, ptr addrspace(1) %address.local1038, align 2, !dbg !30
  %address.local1040 = or disjoint i64 %address.local854, 114, !dbg !30
  %address.local1041 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local852, i64 %address.local1040, !dbg !30
  store half %1793, ptr addrspace(1) %address.local1041, align 2, !dbg !30
  %address.local1043 = or i64 %1739, 115, !dbg !30
  %address.local1044 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local852, i64 %address.local1043, !dbg !30
  store half %1792, ptr addrspace(1) %address.local1044, align 2, !dbg !30
  %address.local1049 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local880, i64 %address.local1022, !dbg !30
  store half %1790, ptr addrspace(1) %address.local1049, align 2, !dbg !30
  %address.local1051 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local880, i64 %address.local1025, !dbg !30
  store half %1789, ptr addrspace(1) %address.local1051, align 2, !dbg !30
  %address.local1053 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local880, i64 %address.local1028, !dbg !30
  store half %1788, ptr addrspace(1) %address.local1053, align 2, !dbg !30
  %address.local1055 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local880, i64 %address.local1031, !dbg !30
  store half %1787, ptr addrspace(1) %address.local1055, align 2, !dbg !30
  %address.local1057 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local880, i64 %address.local1034, !dbg !30
  store half %1785, ptr addrspace(1) %address.local1057, align 2, !dbg !30
  %address.local1059 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local880, i64 %address.local1037, !dbg !30
  store half %1784, ptr addrspace(1) %address.local1059, align 2, !dbg !30
  %address.local1061 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local880, i64 %address.local1040, !dbg !30
  store half %1783, ptr addrspace(1) %address.local1061, align 2, !dbg !30
  %address.local1063 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local880, i64 %address.local1043, !dbg !30
  store half %1782, ptr addrspace(1) %address.local1063, align 2, !dbg !30
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

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.sched.barrier(i32 immarg) #4

attributes #0 = { convergent mustprogress nofree willreturn memory(read) }
attributes #1 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1" "frisk.global_prefetch"="8" "frisk.k_dma"="64-swizzle:1" "frisk.register_comparisons"="1" "frisk.register_lifetimes"="1" "uniform-work-group-size" }
attributes #2 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { convergent mustprogress nocallback nocreateundeforpoison nofree nounwind willreturn memory(none) }
attributes #4 = { convergent mustprogress nocallback nofree nounwind willreturn }
attributes #5 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }
attributes #6 = { convergent mustprogress nocallback nofree nounwind willreturn memory(none) }
attributes #7 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #8 = { convergent nounwind willreturn memory(none) }
attributes #9 = { convergent nounwind }
attributes #10 = { convergent willreturn memory(read) }
attributes #11 = { memory(inaccessiblemem: readwrite) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!1 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/noncausal_k_dma_pass/mytest/run-20261030-162611-z5os6mty/grid-000003")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !1, file: !1, line: 23, type: !4, scopeLine: 23, spFlags: DISPFlagDefinition, unit: !0)
!4 = !DISubroutineType(types: !5)
!5 = !{}
!6 = !{i32 256, i32 1, i32 1}
!7 = !DILocation(line: 23, column: 3, scope: !3)
!8 = !DILocation(line: 67, column: 22, scope: !3)
!9 = !{i32 0, i32 256}
!10 = !DILocation(line: 69, column: 24, scope: !3)
!11 = !DILocation(line: 57, column: 24, scope: !3)
!12 = !DILocation(line: 70, column: 26, scope: !3)
!13 = !DILocation(line: 68, column: 22, scope: !3)
!14 = !DILocation(line: 86, column: 20, scope: !3)
!15 = !DILocation(line: 71, column: 23, scope: !3)
!16 = !DILocation(line: 74, column: 20, scope: !3)
!17 = !DILocation(line: 75, column: 22, scope: !3)
!18 = !DILocation(line: 76, column: 23, scope: !3)
!19 = !DILocation(line: 77, column: 20, scope: !3)
!20 = !DILocation(line: 88, column: 22, scope: !3)
!21 = !DILocation(line: 78, column: 22, scope: !3)
!22 = !DILocation(line: 79, column: 22, scope: !3)
!23 = !DILocation(line: 81, column: 16, scope: !3)
!24 = !DILocation(line: 80, column: 20, scope: !3)
!25 = !DILocation(line: 87, column: 23, scope: !3)
!26 = !{i64 8192}
!27 = !{i32 4}
!28 = !DILocation(line: 92, column: 23, scope: !3)
!29 = !DILocation(line: 93, column: 19, scope: !3)
!30 = !DILocation(line: 94, column: 9, scope: !3)
