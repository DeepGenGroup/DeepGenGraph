; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/causal_best_mytest/group4.ll'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/causal_best_mytest/causal.input.mlir"

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
  %14 = and i64 %13, 576460752303423484, !dbg !7
  %15 = and i64 %8, 3, !dbg !7
  %16 = or disjoint i64 %14, %15, !dbg !7
  %17 = lshr i64 %12, 2, !dbg !7
  %18 = lshr i64 %12, 10, !dbg !8
  %19 = add nuw nsw i64 %17, %18, !dbg !8
  %20 = shl i64 %16, 7, !dbg !8
  %21 = shl i64 %18, 12, !dbg !8
  %22 = sub i64 %20, %21, !dbg !8
  %23 = shl i64 %19, 19, !dbg !8
  %24 = and i64 %23, 16252928, !dbg !8
  %25 = shl nsw i64 %22, 7, !dbg !8
  %26 = add i64 %25, %24, !dbg !8
  %27 = zext nneg i32 %kdma.tid to i64, !dbg !8
  %address.thread = tail call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #8, !dbg !8, !range !9, !frisk.register_lifetimes !5
  %address.local = zext nneg i32 %address.thread to i64, !dbg !8
  %address.local472 = shl nuw nsw i64 %address.local, 3, !dbg !8
  %address.local473 = and i64 %address.local472, 1536, !dbg !8
  %address.local474 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local473, !dbg !8
  %address.local477 = shl nuw nsw i64 %address.local, 7, !dbg !8
  %address.local478 = and i64 %address.local477, 16256, !dbg !8
  %address.local479 = lshr i64 %address.local, 4, !dbg !8
  %address.local480 = and i64 %address.local479, 8, !dbg !8
  %address.local481 = or disjoint i64 %address.local478, %address.local480, !dbg !8
  %28 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %.idx1727 = or disjoint i64 %address.local481, %26, !dbg !8
  %.idx1727.tr = trunc i64 %.idx1727 to i32, !dbg !8
  %dma.buffer.offset1724 = shl i32 %.idx1727.tr, 1, !dbg !8
  %29 = insertelement <2 x i64> poison, i64 %28, i64 0, !dbg !8
  %30 = bitcast <2 x i64> %29 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc1725 = shufflevector <4 x i32> %30, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
  %address.local483 = or disjoint i64 %address.local, 256, !dbg !8
  %address.local484 = shl nuw nsw i64 %address.local483, 3, !dbg !8
  %address.local485 = and i64 %address.local484, 3584, !dbg !8
  %address.local486 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local485, !dbg !8
  %address.local491 = lshr i64 %address.local483, 4, !dbg !8
  %address.local492 = and i64 %address.local491, 24, !dbg !8
  %address.local493 = or disjoint i64 %address.local492, %address.local478, !dbg !8
  %.idx1728 = or disjoint i64 %address.local493, %26, !dbg !8
  %.idx1728.tr = trunc i64 %.idx1728 to i32, !dbg !8
  %dma.buffer.offset1721 = shl i32 %.idx1728.tr, 1, !dbg !8
  %31 = tail call i32 @llvm.amdgcn.workgroup.id.z(), !dbg !7
  %32 = sext i32 %31 to i64, !dbg !7
  %address.local495 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local473, !dbg !8
  %33 = or disjoint i64 %26, 32, !dbg !8
  %34 = or disjoint i64 %address.local481, %33, !dbg !8
  %.tr = trunc i64 %34 to i32, !dbg !8
  %dma.buffer.offset1718 = shl i32 %.tr, 1, !dbg !8
  %address.local499 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local485, !dbg !8
  %35 = insertelement <4 x ptr addrspace(3)> poison, ptr addrspace(3) %address.local474, i64 0, !dbg !8
  %36 = insertelement <4 x ptr addrspace(3)> %35, ptr addrspace(3) %address.local486, i64 1, !dbg !8
  %37 = insertelement <4 x ptr addrspace(3)> %36, ptr addrspace(3) %address.local495, i64 2, !dbg !8
  %38 = insertelement <4 x ptr addrspace(3)> %37, ptr addrspace(3) %address.local499, i64 3, !dbg !8
  %39 = ptrtoint <4 x ptr addrspace(3)> %38 to <4 x i64>, !dbg !8
  %40 = trunc <4 x i64> %39 to <4 x i32>, !dbg !8
  %41 = extractelement <4 x i32> %40, i64 0, !dbg !8
  %42 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %41), !dbg !8
  %dma.buffer.lds1726 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %42), !dbg !8
  %43 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1726, i32 %dma.buffer.offset1724, <4 x i32> %dma.buffer.rsrc1725) #9, !dbg !8
  %44 = extractelement <4 x i32> %40, i64 1, !dbg !8
  %45 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %44), !dbg !8
  %dma.buffer.lds1723 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %45), !dbg !8
  %46 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1723, i32 %dma.buffer.offset1721, <4 x i32> %dma.buffer.rsrc1725) #9, !dbg !8
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %47 = extractelement <4 x i32> %40, i64 2, !dbg !8
  %48 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %47), !dbg !8
  %dma.buffer.lds1720 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %48), !dbg !8
  %49 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1720, i32 %dma.buffer.offset1718, <4 x i32> %dma.buffer.rsrc1725) #9, !dbg !8
  %50 = extractelement <4 x i32> %40, i64 3, !dbg !8
  %51 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %50), !dbg !8
  %52 = or disjoint i64 %address.local493, %33, !dbg !8
  %.tr1729 = trunc i64 %52 to i32, !dbg !8
  %dma.buffer.offset1715 = shl i32 %.tr1729, 1, !dbg !8
  %dma.buffer.lds1717 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %51), !dbg !8
  %53 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1717, i32 %dma.buffer.offset1715, <4 x i32> %dma.buffer.rsrc1725) #9, !dbg !8
  %address.local503 = lshr i32 %address.thread, 2, !dbg !10
  %address.local504 = and i32 %address.local503, 4, !dbg !10
  %address.local505 = shl nuw nsw i32 %address.thread, 1, !dbg !10
  %address.local506 = and i32 %address.local505, 384, !dbg !10
  %address.local507 = or disjoint i32 %address.local504, %address.local506, !dbg !10
  %address.local508 = shl nuw nsw i32 %address.thread, 3, !dbg !10
  %address.local509 = and i32 %address.local508, 120, !dbg !10
  %address.local510 = or disjoint i32 %address.local507, %address.local509, !dbg !10
  %address.local511 = shl nuw nsw i32 %address.thread, 5, !dbg !10
  %address.local512 = and i32 %address.local511, 1024, !dbg !10
  %address.local513 = or disjoint i32 %address.local510, %address.local512, !dbg !10
  %address.local514 = zext nneg i32 %address.local513 to i64, !dbg !10
  %address.local515 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local514, !dbg !10
  %54 = load <4 x half>, ptr addrspace(3) %address.local515, align 8, !dbg !10
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local515, i64 4096, !dbg !10
  %55 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !10
  %56 = fmul <4 x half> %54, splat (half 1.275630e-01), !dbg !10
  %lds.const.gep1525 = getelementptr i8, ptr addrspace(3) %address.local515, i64 1024, !dbg !10
  %57 = load <4 x half>, ptr addrspace(3) %lds.const.gep1525, align 8, !dbg !10
  %58 = fmul <4 x half> %55, splat (half 1.275630e-01), !dbg !10
  %lds.const.gep1526 = getelementptr i8, ptr addrspace(3) %address.local515, i64 5120, !dbg !10
  %59 = load <4 x half>, ptr addrspace(3) %lds.const.gep1526, align 8, !dbg !10
  %60 = fmul <4 x half> %57, splat (half 1.275630e-01), !dbg !10
  %61 = fmul <4 x half> %59, splat (half 1.275630e-01), !dbg !10
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %62 = or disjoint i64 %26, 64, !dbg !8
  %63 = or disjoint i64 %address.local481, %62, !dbg !8
  %.tr1730 = trunc i64 %63 to i32, !dbg !8
  %dma.buffer.offset1712 = shl i32 %.tr1730, 1, !dbg !8
  %64 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1726, i32 %dma.buffer.offset1712, <4 x i32> %dma.buffer.rsrc1725) #9, !dbg !8
  %65 = or disjoint i64 %address.local493, %62, !dbg !8
  %.tr1731 = trunc i64 %65 to i32, !dbg !8
  %dma.buffer.offset1709 = shl i32 %.tr1731, 1, !dbg !8
  %66 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1723, i32 %dma.buffer.offset1709, <4 x i32> %dma.buffer.rsrc1725) #9, !dbg !8
  %lds.const.gep1527 = getelementptr i8, ptr addrspace(3) %address.local515, i64 8192, !dbg !10
  %67 = load <4 x half>, ptr addrspace(3) %lds.const.gep1527, align 8, !dbg !10
  %lds.const.gep1528 = getelementptr i8, ptr addrspace(3) %address.local515, i64 12288, !dbg !10
  %68 = load <4 x half>, ptr addrspace(3) %lds.const.gep1528, align 8, !dbg !10
  %lds.const.gep1529 = getelementptr i8, ptr addrspace(3) %address.local515, i64 9216, !dbg !10
  %69 = load <4 x half>, ptr addrspace(3) %lds.const.gep1529, align 8, !dbg !10
  %70 = fmul <4 x half> %67, splat (half 1.275630e-01), !dbg !10
  %lds.const.gep1530 = getelementptr i8, ptr addrspace(3) %address.local515, i64 13312, !dbg !10
  %71 = load <4 x half>, ptr addrspace(3) %lds.const.gep1530, align 8, !dbg !10
  %72 = fmul <4 x half> %68, splat (half 1.275630e-01), !dbg !10
  %73 = fmul <4 x half> %69, splat (half 1.275630e-01), !dbg !10
  %74 = fmul <4 x half> %71, splat (half 1.275630e-01), !dbg !10
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %75 = or disjoint i64 %26, 96, !dbg !8
  %76 = or disjoint i64 %address.local481, %75, !dbg !8
  %.tr1732 = trunc i64 %76 to i32, !dbg !8
  %dma.buffer.offset1706 = shl i32 %.tr1732, 1, !dbg !8
  %77 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1720, i32 %dma.buffer.offset1706, <4 x i32> %dma.buffer.rsrc1725) #9, !dbg !8
  %78 = or disjoint i64 %address.local493, %75, !dbg !8
  %.tr1733 = trunc i64 %78 to i32, !dbg !8
  %dma.buffer.offset1703 = shl i32 %.tr1733, 1, !dbg !8
  %79 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1717, i32 %dma.buffer.offset1703, <4 x i32> %dma.buffer.rsrc1725) #9, !dbg !8
  %80 = load <4 x half>, ptr addrspace(3) %address.local515, align 8, !dbg !10
  %81 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !10
  %82 = load <4 x half>, ptr addrspace(3) %lds.const.gep1525, align 8, !dbg !10
  %83 = fmul <4 x half> %80, splat (half 1.275630e-01), !dbg !10
  %84 = load <4 x half>, ptr addrspace(3) %lds.const.gep1526, align 8, !dbg !10
  %85 = fmul <4 x half> %81, splat (half 1.275630e-01), !dbg !10
  %86 = fmul <4 x half> %82, splat (half 1.275630e-01), !dbg !10
  %87 = fmul <4 x half> %84, splat (half 1.275630e-01), !dbg !10
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %88 = load <4 x half>, ptr addrspace(3) %lds.const.gep1527, align 8, !dbg !10
  %89 = load <4 x half>, ptr addrspace(3) %lds.const.gep1528, align 8, !dbg !10
  %90 = load <4 x half>, ptr addrspace(3) %lds.const.gep1529, align 8, !dbg !10
  %91 = fmul <4 x half> %88, splat (half 1.275630e-01), !dbg !10
  %92 = load <4 x half>, ptr addrspace(3) %lds.const.gep1530, align 8, !dbg !10
  %93 = fmul <4 x half> %89, splat (half 1.275630e-01), !dbg !10
  %94 = fmul <4 x half> %90, splat (half 1.275630e-01), !dbg !10
  %95 = fmul <4 x half> %92, splat (half 1.275630e-01), !dbg !10
  fence syncscope("workgroup") release, !dbg !11
  tail call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %96 = shl i64 %17, 7, !dbg !12
  %97 = and i64 %96, 3968, !dbg !12
  %98 = shl nsw i64 %32, 12, !dbg !12
  %99 = or disjoint i64 %97, %98, !dbg !12
  %100 = shl nuw nsw i64 %27, 2, !dbg !13
  %101 = and i64 %100, 3084, !dbg !13
  %102 = and i32 %kdma.tid, 1, !dbg !13
  %narrow = add nuw nsw i32 %102, 1, !dbg !13
  %103 = zext nneg i32 %narrow to i64, !dbg !13
  %104 = or disjoint i64 %101, %103, !dbg !13
  %105 = lshr i64 %104, 1, !dbg !13
  %106 = lshr i64 %27, 2, !dbg !14
  %107 = and i64 %106, 48, !dbg !14
  %108 = and i64 %27, 783, !dbg !14
  %109 = or disjoint i64 %108, %107, !dbg !14
  %110 = add i64 %109, %20, !dbg !14
  %111 = and i64 %27, 48, !dbg !14
  %112 = add nuw nsw i64 %110, 1, !dbg !15
  %113 = add i64 %110, 65, !dbg !15
  %k0.prime.2349 = shl i64 %17, 19, !dbg !12
  %k0.prime.2350 = and i64 %k0.prime.2349, 16252928, !dbg !12
  %k0.prime.kdma.source = zext nneg i32 %kdma.owner to i64, !dbg !12
  %k0.prime.kdma.source1878 = lshr i64 %k0.prime.kdma.source, 4, !dbg !12
  %k0.prime.kdma.source1879 = and i64 %k0.prime.kdma.source1878, 8, !dbg !12
  %k0.prime.kdma.source1880 = and i64 %k0.prime.kdma.source, 91, !dbg !12
  %k0.prime.kdma.source1881 = lshr i64 %k0.prime.kdma.source, 2, !dbg !12
  %k0.prime.kdma.source1882 = and i64 %k0.prime.kdma.source1881, 4, !dbg !12
  %k0.prime.kdma.source1883 = add nuw nsw i64 %k0.prime.kdma.source1882, %k0.prime.kdma.source, !dbg !12
  %k0.prime.kdma.source1884 = and i64 %k0.prime.kdma.source1883, 4, !dbg !12
  %k0.prime.kdma.source1885 = or disjoint i64 %k0.prime.kdma.source1884, %k0.prime.kdma.source1880, !dbg !12
  %k0.prime.kdma.source1886 = shl nuw nsw i64 %k0.prime.kdma.source1885, 7, !dbg !12
  %k0.prime.kdma.source1887 = or disjoint i64 %k0.prime.kdma.source1886, %k0.prime.kdma.source1879, !dbg !12
  %k0.prime.kdma.source1888 = or disjoint i64 %k0.prime.kdma.source1887, %k0.prime.2350, !dbg !12
  %k0.prime.kdma.source1889 = trunc nuw nsw i64 %k0.prime.kdma.source1888 to i32, !dbg !12
  %k0.prime.kdma.source1890 = shl nuw nsw i32 %k0.prime.kdma.source1889, 1, !dbg !12
  %k0.prime.kdma.offset = or disjoint i32 %k0.prime.kdma.source1890, %kdma.intra, !dbg !12
  %k0.prime.2357 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !12
  %k0.prime.2358 = insertelement <2 x i64> poison, i64 %k0.prime.2357, i64 0, !dbg !12
  %k0.prime.2359 = bitcast <2 x i64> %k0.prime.2358 to <4 x i32>, !dbg !12
  %k0.prime.dma.buffer.rsrc1701 = shufflevector <4 x i32> %k0.prime.2359, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !12
  %k0.prime.2360 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1726, i32 %k0.prime.kdma.offset, <4 x i32> %k0.prime.dma.buffer.rsrc1701) #9, !dbg !12, !frisk.cross_bn_k0 !5
  %k0.prime.kdma.lds1906 = add i32 %dma.buffer.lds1726, 512, !dbg !12
  %k0.prime.kdma.source1891 = zext nneg i32 %kdma.owner.hi to i64, !dbg !12
  %k0.prime.kdma.source1892 = lshr i64 %k0.prime.kdma.source1891, 4, !dbg !12
  %k0.prime.kdma.source1893 = and i64 %k0.prime.kdma.source1892, 8, !dbg !12
  %k0.prime.kdma.source1894 = and i64 %k0.prime.kdma.source1891, 123, !dbg !12
  %k0.prime.kdma.source1895 = lshr i64 %k0.prime.kdma.source1891, 2, !dbg !12
  %k0.prime.kdma.source1896 = and i64 %k0.prime.kdma.source1895, 4, !dbg !12
  %k0.prime.kdma.source1897 = add nuw nsw i64 %k0.prime.kdma.source1896, %k0.prime.kdma.source1891, !dbg !12
  %k0.prime.kdma.source1898 = and i64 %k0.prime.kdma.source1897, 4, !dbg !12
  %k0.prime.kdma.source1899 = or disjoint i64 %k0.prime.kdma.source1898, %k0.prime.kdma.source1894, !dbg !12
  %k0.prime.kdma.source1900 = shl nuw nsw i64 %k0.prime.kdma.source1899, 7, !dbg !12
  %k0.prime.kdma.source1901 = or disjoint i64 %k0.prime.kdma.source1900, %k0.prime.kdma.source1893, !dbg !12
  %k0.prime.kdma.source1902 = or disjoint i64 %k0.prime.kdma.source1901, %k0.prime.2350, !dbg !12
  %k0.prime.kdma.source1903 = trunc nuw nsw i64 %k0.prime.kdma.source1902 to i32, !dbg !12
  %k0.prime.kdma.source1904 = shl nuw nsw i32 %k0.prime.kdma.source1903, 1, !dbg !12
  %k0.prime.kdma.offset1905 = or disjoint i32 %k0.prime.kdma.source1904, %kdma.intra.hi, !dbg !12
  %k0.prime.2361 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.kdma.lds1906, i32 %k0.prime.kdma.offset1905, <4 x i32> %k0.prime.dma.buffer.rsrc1701) #9, !dbg !12, !frisk.cross_bn_k0 !5
  %k0.prime.kdma.source1918 = or disjoint i64 %k0.prime.kdma.source1887, 16, !dbg !12
  %k0.prime.kdma.source1919 = or disjoint i64 %k0.prime.kdma.source1918, %k0.prime.2350, !dbg !12
  %k0.prime.kdma.source1920 = trunc nuw nsw i64 %k0.prime.kdma.source1919 to i32, !dbg !12
  %k0.prime.kdma.source1921 = shl nuw nsw i32 %k0.prime.kdma.source1920, 1, !dbg !12
  %k0.prime.kdma.offset1922 = or disjoint i32 %k0.prime.kdma.source1921, %kdma.intra, !dbg !12
  %k0.prime.2363 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1723, i32 %k0.prime.kdma.offset1922, <4 x i32> %k0.prime.dma.buffer.rsrc1701) #9, !dbg !12, !frisk.cross_bn_k0 !5
  %k0.prime.kdma.lds1940 = add i32 %dma.buffer.lds1723, 512, !dbg !12
  %k0.prime.kdma.source1935 = or disjoint i64 %k0.prime.kdma.source1901, 16, !dbg !12
  %k0.prime.kdma.source1936 = or disjoint i64 %k0.prime.kdma.source1935, %k0.prime.2350, !dbg !12
  %k0.prime.kdma.source1937 = trunc nuw nsw i64 %k0.prime.kdma.source1936 to i32, !dbg !12
  %k0.prime.kdma.source1938 = shl nuw nsw i32 %k0.prime.kdma.source1937, 1, !dbg !12
  %k0.prime.kdma.offset1939 = or disjoint i32 %k0.prime.kdma.source1938, %kdma.intra.hi, !dbg !12
  %k0.prime.2364 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.kdma.lds1940, i32 %k0.prime.kdma.offset1939, <4 x i32> %k0.prime.dma.buffer.rsrc1701) #9, !dbg !12, !frisk.cross_bn_k0 !5
  %kdma.source1955 = or disjoint i32 %kdma.intra, 64
  %kdma.source2025 = or disjoint i32 %kdma.intra, 128
  %kdma.source2095 = or disjoint i32 %kdma.intra, 192
  %114 = ptrtoint ptr addrspace(1) %2 to i64
  %115 = insertelement <2 x i64> poison, i64 %114, i64 0
  %116 = bitcast <2 x i64> %115 to <4 x i32>
  %dma.buffer.rsrc1677 = shufflevector <4 x i32> %116, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  %117 = lshr i64 %112, 32
  %compare.high = trunc nuw i64 %117 to i32
  %compare.low.base = trunc i64 %112 to i32
  %118 = lshr i64 %113, 32
  %compare.high1393 = trunc nuw i64 %118 to i32
  %compare.low.base1397 = trunc i64 %113 to i32
  %119 = insertelement <4 x i32> poison, i32 %compare.low.base1397, i64 0, !dbg !16
  %120 = shufflevector <4 x i32> %119, <4 x i32> poison, <4 x i32> zeroinitializer, !dbg !16
  %121 = insertelement <4 x i32> poison, i32 %compare.low.base, i64 0, !dbg !16
  %122 = shufflevector <4 x i32> %121, <4 x i32> poison, <4 x i32> zeroinitializer, !dbg !16
  %invariant.op = sub i32 72, %kdma.intra, !dbg !11
  %invariant.op112 = sub i32 72, %kdma.intra, !dbg !11
  %invariant.op113 = sub i32 136, %kdma.intra, !dbg !11
  %invariant.op114 = sub i32 136, %kdma.intra, !dbg !11
  %invariant.op115 = sub i32 200, %kdma.intra, !dbg !11
  %invariant.op116 = sub i32 200, %kdma.intra, !dbg !11
  br label %123, !dbg !11

123:                                              ; preds = %4, %k0.join
  %124 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %1259, %k0.join ]
  %125 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %1255, %k0.join ]
  %126 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %1251, %k0.join ]
  %127 = phi [2 x <1 x float>] [ [<1 x float> splat (float -inf), <1 x float> splat (float -inf)], %4 ], [ %1247, %k0.join ]
  %128 = phi [2 x <1 x float>] [ zeroinitializer, %4 ], [ %1010, %k0.join ]
  %129 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %1245, %k0.join ]
  %130 = phi i64 [ 0, %4 ], [ %1167, %k0.join ]
  %131 = lshr i64 %130, 5, !dbg !12
  %132 = shl nuw nsw i64 %130, 7, !dbg !12
  %133 = shl i64 %131, 19, !dbg !12
  %134 = shl i64 %130, 14, !dbg !12
  %135 = sub i64 %134, %133, !dbg !12
  %address.thread566 = tail call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #8, !dbg !12, !range !9, !frisk.register_lifetimes !5
  %address.local567 = zext nneg i32 %address.thread566 to i64, !dbg !8
  %address.local568 = shl nuw nsw i64 %address.local567, 3, !dbg !8
  %address.local569 = and i64 %address.local568, 1536, !dbg !8
  %address.local570 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local569, !dbg !8
  %address.local584 = or disjoint i64 %address.local567, 256, !dbg !8
  %address.local585 = shl nuw nsw i64 %address.local584, 3, !dbg !8
  %address.local586 = and i64 %address.local585, 3584, !dbg !8
  %address.local587 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local586, !dbg !8
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  tail call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %136 = add nuw nsw i64 %130, 1024, !dbg !12
  %137 = lshr i64 %136, 12, !dbg !12
  %138 = add nuw nsw i64 %137, %17, !dbg !12
  %139 = add nsw i64 %131, %99, !dbg !12
  %140 = shl i64 %138, 19, !dbg !12
  %141 = and i64 %140, 16252928, !dbg !12
  %address.local601 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local569, !dbg !8
  %142 = add i64 %135, %131, !dbg !12
  %.idx1745 = add i64 %142, %99, !dbg !12
  %143 = add i64 %.idx1745, %141, !dbg !12
  %address.local605 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local586, !dbg !8
  %144 = insertelement <4 x ptr addrspace(3)> poison, ptr addrspace(3) %address.local570, i64 0, !dbg !8
  %145 = insertelement <4 x ptr addrspace(3)> %144, ptr addrspace(3) %address.local587, i64 1, !dbg !8
  %146 = insertelement <4 x ptr addrspace(3)> %145, ptr addrspace(3) %address.local601, i64 2, !dbg !8
  %147 = insertelement <4 x ptr addrspace(3)> %146, ptr addrspace(3) %address.local605, i64 3, !dbg !8
  %148 = ptrtoint <4 x ptr addrspace(3)> %147 to <4 x i64>, !dbg !8
  %149 = trunc <4 x i64> %148 to <4 x i32>, !dbg !8
  %150 = extractelement <4 x i32> %149, i64 0, !dbg !12
  %151 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %150), !dbg !12
  %dma.buffer.lds1702 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %151), !dbg !12
  %kdma.lds1906 = add i32 %dma.buffer.lds1702, 512, !dbg !12
  %152 = extractelement <4 x i32> %149, i64 1, !dbg !12
  %153 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %152), !dbg !12
  %dma.buffer.lds1699 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %153), !dbg !12
  %kdma.lds1940 = add i32 %dma.buffer.lds1699, 512, !dbg !12
  %154 = extractelement <4 x i32> %149, i64 2, !dbg !12
  %155 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %154), !dbg !12
  %dma.buffer.lds1696 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %155), !dbg !12
  %kdma.lds1974 = add i32 %dma.buffer.lds1696, 512, !dbg !12
  %156 = extractelement <4 x i32> %149, i64 3, !dbg !12
  %address.local610 = lshr i64 %address.local567, 2, !dbg !13
  %address.local613 = shl nuw nsw i64 %address.local567, 1, !dbg !13
  %address.local614 = and i64 %address.local613, 24, !dbg !13
  %address.local616 = shl i64 %address.local567, 62, !dbg !13
  %address.local617 = ashr exact i64 %address.local616, 57, !dbg !13
  %address.local618 = and i64 %address.local617, -64, !dbg !13
  %address.local621 = and i64 %address.local610, 12, !dbg !10
  %address.local623 = and i64 %address.local567, 63, !dbg !10
  %address.local624 = lshr i64 %address.local623, 5, !dbg !10
  %address.local625 = mul nuw nsw i64 %address.local624, 1016, !dbg !10
  %.idx = mul nuw nsw i64 %address.local567, 320, !dbg !13
  %157 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %.idx, !dbg !13
  %.idx1752 = mul nsw i64 %address.local610, -1280, !dbg !13
  %158 = getelementptr i8, ptr addrspace(3) %157, i64 %.idx1752, !dbg !13
  %159 = getelementptr [2 x i8], ptr addrspace(3) %158, i64 %address.local614, !dbg !13
  %160 = getelementptr [2 x i8], ptr addrspace(3) %159, i64 %address.local618, !dbg !13
  %161 = getelementptr [2 x i8], ptr addrspace(3) %160, i64 %address.local621, !dbg !13
  %address.local627 = getelementptr [2 x i8], ptr addrspace(3) %161, i64 %address.local625, !dbg !13
  %162 = add nsw i64 %.idx1752, %.idx, !dbg !13
  %.idx2277 = shl nuw nsw i64 %address.local614, 1, !dbg !13
  %163 = or disjoint i64 %162, %.idx2277, !dbg !13
  %164 = or disjoint i64 %address.local618, %address.local621, !dbg !13
  %165 = shl nsw i64 %164, 1, !dbg !13
  %address.local627.idx = mul nuw nsw i64 %address.local624, 2032, !dbg !13
  %166 = add nsw i64 %163, %address.local627.idx, !dbg !13
  %167 = add nsw i64 %166, %165, !dbg !13
  %168 = lshr i64 %167, 6, !dbg !13
  %169 = and i64 %168, 8, !dbg !13
  %170 = xor i64 %169, %167, !dbg !13
  %171 = sub i64 %170, %167, !dbg !13
  %kdma.swizzled = getelementptr i8, ptr addrspace(3) %address.local627, i64 %171, !dbg !13
  %kdma.gep = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4096, !dbg !13
  %address.local638 = shl nuw nsw i32 %address.thread566, 1, !dbg !13
  %address.local639 = and i32 %address.local638, 24, !dbg !13
  %address.local640 = shl nuw nsw i32 %address.thread566, 8, !dbg !13
  %address.local641 = and i32 %address.local640, 768, !dbg !13
  %address.local642 = or disjoint i32 %address.local639, %address.local641, !dbg !13
  %address.local643 = shl nuw nsw i32 %address.thread566, 5, !dbg !13
  %address.local644 = and i32 %address.local643, 96, !dbg !13
  %address.local645 = or disjoint i32 %address.local642, %address.local644, !dbg !13
  %address.local646 = zext nneg i32 %address.local645 to i64, !dbg !13
  %address.local647 = lshr i32 %address.thread566, 1, !dbg !13
  %address.local648 = and i32 %address.local647, 1, !dbg !13
  %address.local649 = zext nneg i32 %address.local648 to i64, !dbg !13
  %172 = add nuw nsw i64 %address.local627.idx, 64, !dbg !13
  %kdma.gep2153 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 128, !dbg !13
  %kdma.gep2157 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1024, !dbg !13
  %kdma.gep2154 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4224, !dbg !13
  %kdma.gep2158 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5120, !dbg !13
  %kdma.gep2161 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1152, !dbg !13
  %kdma.gep2162 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5248, !dbg !13
  %173 = add nuw nsw i64 %130, 2048, !dbg !12
  %174 = lshr i64 %173, 12, !dbg !12
  %175 = add nuw nsw i64 %174, %17, !dbg !12
  %176 = insertelement <2 x i64> poison, i64 %139, i64 0, !dbg !12
  %177 = shufflevector <2 x i64> %176, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !12
  %178 = icmp slt <2 x i64> %177, <i64 -32, i64 -64>, !dbg !12
  %179 = extractelement <2 x i1> %178, i64 0, !dbg !12
  %180 = sext i1 %179 to i64, !dbg !12
  %181 = add nsw <2 x i64> %177, <i64 32, i64 64>, !dbg !12
  %182 = sub nsw <2 x i64> <i64 -33, i64 -65>, %177, !dbg !12
  %183 = select <2 x i1> %178, <2 x i64> %182, <2 x i64> %181, !dbg !12
  %184 = sdiv <2 x i64> %183, splat (i64 128), !dbg !12
  %185 = extractelement <2 x i64> %184, i64 0, !dbg !12
  %186 = xor i64 %185, %180, !dbg !12
  %187 = shl nsw i64 %186, 7, !dbg !12
  %188 = sub i64 %143, %187, !dbg !12
  %kdma.source1952 = add i64 %188, %k0.prime.kdma.source1887, !dbg !12
  %kdma.source1953 = trunc i64 %kdma.source1952 to i32, !dbg !12
  %kdma.source1954 = shl i32 %kdma.source1953, 1, !dbg !12
  %kdma.offset1956 = add i32 %kdma.source1955, %kdma.source1954, !dbg !12
  %kdma.source1969 = add i64 %188, %k0.prime.kdma.source1901, !dbg !12
  %kdma.source1970 = trunc i64 %kdma.source1969 to i32, !dbg !12
  %kdma.source1971 = shl i32 %kdma.source1970, 1, !dbg !12
  %kdma.offset1973.reass.reass = add i32 %kdma.source1971, %invariant.op
  %189 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1696, i32 %kdma.offset1956, <4 x i32> %k0.prime.dma.buffer.rsrc1701) #9, !dbg !12
  %190 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1974, i32 %kdma.offset1973.reass.reass, <4 x i32> %k0.prime.dma.buffer.rsrc1701) #9, !dbg !12
  %191 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %156), !dbg !12
  %dma.buffer.lds1693 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %191), !dbg !12
  %kdma.source1987 = add i64 %188, %k0.prime.kdma.source1918, !dbg !12
  %kdma.source1988 = trunc i64 %kdma.source1987 to i32, !dbg !12
  %kdma.source1989 = shl i32 %kdma.source1988, 1, !dbg !12
  %kdma.offset1991 = add i32 %kdma.source1955, %kdma.source1989, !dbg !12
  %192 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1693, i32 %kdma.offset1991, <4 x i32> %k0.prime.dma.buffer.rsrc1701) #9, !dbg !12
  %kdma.source2005 = add i64 %188, %k0.prime.kdma.source1935, !dbg !12
  %kdma.source2006 = trunc i64 %kdma.source2005 to i32, !dbg !12
  %kdma.source2007 = shl i32 %kdma.source2006, 1, !dbg !12
  %kdma.offset2009.reass.reass = add i32 %kdma.source2007, %invariant.op112
  %kdma.lds2010 = add i32 %dma.buffer.lds1693, 512, !dbg !12
  %193 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds2010, i32 %kdma.offset2009.reass.reass, <4 x i32> %k0.prime.dma.buffer.rsrc1701) #9, !dbg !12
  %194 = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8, !dbg !13
  %195 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %56, <4 x half> %194, <4 x float> zeroinitializer) #8, !dbg !13
  %196 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8, !dbg !13
  %address.base = tail call i64 asm sideeffect "", "=v,0"(i64 %105) #8, !dbg !13, !frisk.register_lifetimes !5
  %address.local650 = add nsw i64 %address.base, %address.local649, !dbg !13
  %address.local651 = shl i64 %address.local650, 6, !dbg !13
  %address.local652 = sub i64 %address.local646, %address.local651, !dbg !13
  %197 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local652, !dbg !13
  %198 = getelementptr [2 x i8], ptr addrspace(3) %197, i64 %address.local621, !dbg !13
  %199 = getelementptr [2 x i8], ptr addrspace(3) %198, i64 %address.local625, !dbg !13
  %address.local656 = getelementptr i8, ptr addrspace(3) %199, i64 64, !dbg !13
  %.idx22802283 = add i64 %address.local652, %address.local621, !dbg !13
  %200 = shl i64 %.idx22802283, 1, !dbg !13
  %201 = add i64 %172, %200, !dbg !13
  %202 = lshr i64 %201, 6, !dbg !13
  %203 = and i64 %202, 8, !dbg !13
  %204 = xor i64 %203, %201, !dbg !13
  %205 = sub i64 %204, %201, !dbg !13
  %kdma.swizzled2151 = getelementptr i8, ptr addrspace(3) %address.local656, i64 %205, !dbg !13
  %206 = load <4 x half>, ptr addrspace(3) %kdma.swizzled2151, align 8, !dbg !13, !frisk.k_lds_pair !17
  %kdma.gep2155 = getelementptr i8, ptr addrspace(3) %kdma.swizzled2151, i64 128, !dbg !13
  %207 = load <4 x half>, ptr addrspace(3) %kdma.gep2155, align 8, !dbg !13, !frisk.k_lds_pair !17
  %kdma.gep2152 = getelementptr i8, ptr addrspace(3) %kdma.swizzled2151, i64 4096, !dbg !13
  %208 = load <4 x half>, ptr addrspace(3) %kdma.gep2152, align 8, !dbg !13, !frisk.k_lds_pair !18
  %kdma.gep2156 = getelementptr i8, ptr addrspace(3) %kdma.swizzled2151, i64 4224, !dbg !13
  %209 = load <4 x half>, ptr addrspace(3) %kdma.gep2156, align 8, !dbg !13, !frisk.k_lds_pair !18
  %210 = load <4 x half>, ptr addrspace(3) %kdma.gep2153, align 8, !dbg !13, !frisk.k_lds_pair !19
  %211 = load <4 x half>, ptr addrspace(3) %kdma.gep2157, align 8, !dbg !13, !frisk.k_lds_pair !19
  %212 = load <4 x half>, ptr addrspace(3) %kdma.gep2154, align 8, !dbg !13, !frisk.k_lds_pair !20
  %213 = load <4 x half>, ptr addrspace(3) %kdma.gep2158, align 8, !dbg !13, !frisk.k_lds_pair !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %214 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %58, <4 x half> %196, <4 x float> %195) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %kdma.gep2159 = getelementptr i8, ptr addrspace(3) %kdma.swizzled2151, i64 1024, !dbg !13
  %215 = load <4 x half>, ptr addrspace(3) %kdma.gep2159, align 8, !dbg !13, !frisk.k_lds_pair !21
  %kdma.gep2163 = getelementptr i8, ptr addrspace(3) %kdma.swizzled2151, i64 1152, !dbg !13
  %216 = load <4 x half>, ptr addrspace(3) %kdma.gep2163, align 8, !dbg !13, !frisk.k_lds_pair !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %217 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %56, <4 x half> %206, <4 x float> zeroinitializer) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %kdma.gep2160 = getelementptr i8, ptr addrspace(3) %kdma.swizzled2151, i64 5120, !dbg !13
  %218 = load <4 x half>, ptr addrspace(3) %kdma.gep2160, align 8, !dbg !13, !frisk.k_lds_pair !22
  %kdma.gep2164 = getelementptr i8, ptr addrspace(3) %kdma.swizzled2151, i64 5248, !dbg !13
  %219 = load <4 x half>, ptr addrspace(3) %kdma.gep2164, align 8, !dbg !13, !frisk.k_lds_pair !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %220 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %58, <4 x half> %208, <4 x float> %217) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %221 = load <4 x half>, ptr addrspace(3) %kdma.gep2161, align 8, !dbg !13, !frisk.k_lds_pair !23
  %222 = load <4 x half>, ptr addrspace(3) %kdma.gep2162, align 8, !dbg !13, !frisk.k_lds_pair !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %223 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %56, <4 x half> %210, <4 x float> zeroinitializer) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %224 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %58, <4 x half> %212, <4 x float> %223) #8, !dbg !13
  %225 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %56, <4 x half> %207, <4 x float> zeroinitializer) #8, !dbg !13
  %226 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %58, <4 x half> %209, <4 x float> %225) #8, !dbg !13
  %227 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %56, <4 x half> %211, <4 x float> zeroinitializer) #8, !dbg !13
  %228 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %58, <4 x half> %213, <4 x float> %227) #8, !dbg !13
  %229 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %56, <4 x half> %215, <4 x float> zeroinitializer) #8, !dbg !13
  %230 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %58, <4 x half> %218, <4 x float> %229) #8, !dbg !13
  %231 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %56, <4 x half> %221, <4 x float> zeroinitializer) #8, !dbg !13
  %232 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %58, <4 x half> %222, <4 x float> %231) #8, !dbg !13
  %233 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %56, <4 x half> %216, <4 x float> zeroinitializer) #8, !dbg !13
  %234 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %58, <4 x half> %219, <4 x float> %233) #8, !dbg !13
  %235 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %194, <4 x float> zeroinitializer) #8, !dbg !13
  %236 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %196, <4 x float> %235) #8, !dbg !13
  %237 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %206, <4 x float> zeroinitializer) #8, !dbg !13
  %238 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %208, <4 x float> %237) #8, !dbg !13
  %239 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %210, <4 x float> zeroinitializer) #8, !dbg !13
  %240 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %212, <4 x float> %239) #8, !dbg !13
  %241 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %207, <4 x float> zeroinitializer) #8, !dbg !13
  %242 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %209, <4 x float> %241) #8, !dbg !13
  %243 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %211, <4 x float> zeroinitializer) #8, !dbg !13
  %244 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %213, <4 x float> %243) #8, !dbg !13
  %245 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %215, <4 x float> zeroinitializer) #8, !dbg !13
  %246 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %218, <4 x float> %245) #8, !dbg !13
  %247 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %221, <4 x float> zeroinitializer) #8, !dbg !13
  %248 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %222, <4 x float> %247) #8, !dbg !13
  %249 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %216, <4 x float> zeroinitializer) #8, !dbg !13
  %250 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %219, <4 x float> %249) #8, !dbg !13
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  tail call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %251 = extractelement <2 x i1> %178, i64 1, !dbg !12
  %252 = sext i1 %251 to i64, !dbg !12
  %253 = extractelement <2 x i64> %184, i64 1, !dbg !12
  %254 = xor i64 %253, %252, !dbg !12
  %255 = shl i64 %175, 19, !dbg !12
  %256 = and i64 %255, 16252928, !dbg !12
  %257 = add i64 %.idx1745, %256, !dbg !12
  %258 = shl nsw i64 %254, 7, !dbg !12
  %259 = sub i64 %257, %258, !dbg !12
  %kdma.source2022 = add i64 %259, %k0.prime.kdma.source1887, !dbg !12
  %kdma.source2023 = trunc i64 %kdma.source2022 to i32, !dbg !12
  %kdma.source2024 = shl i32 %kdma.source2023, 1, !dbg !12
  %kdma.offset2026 = add i32 %kdma.source2025, %kdma.source2024, !dbg !12
  %260 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1702, i32 %kdma.offset2026, <4 x i32> %k0.prime.dma.buffer.rsrc1701) #9, !dbg !12
  %kdma.source2039 = add i64 %259, %k0.prime.kdma.source1901, !dbg !12
  %kdma.source2040 = trunc i64 %kdma.source2039 to i32, !dbg !12
  %kdma.source2041 = shl i32 %kdma.source2040, 1, !dbg !12
  %kdma.offset2043.reass.reass = add i32 %kdma.source2041, %invariant.op113
  %261 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1906, i32 %kdma.offset2043.reass.reass, <4 x i32> %k0.prime.dma.buffer.rsrc1701) #9, !dbg !12
  %kdma.source2057 = add i64 %259, %k0.prime.kdma.source1918, !dbg !12
  %kdma.source2058 = trunc i64 %kdma.source2057 to i32, !dbg !12
  %kdma.source2059 = shl i32 %kdma.source2058, 1, !dbg !12
  %kdma.offset2061 = add i32 %kdma.source2025, %kdma.source2059, !dbg !12
  %262 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1699, i32 %kdma.offset2061, <4 x i32> %k0.prime.dma.buffer.rsrc1701) #9, !dbg !12
  %kdma.source2075 = add i64 %259, %k0.prime.kdma.source1935, !dbg !12
  %kdma.source2076 = trunc i64 %kdma.source2075 to i32, !dbg !12
  %kdma.source2077 = shl i32 %kdma.source2076, 1, !dbg !12
  %kdma.offset2079.reass.reass = add i32 %kdma.source2077, %invariant.op114
  %263 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1940, i32 %kdma.offset2079.reass.reass, <4 x i32> %k0.prime.dma.buffer.rsrc1701) #9, !dbg !12
  %kdma.gep2181 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8192, !dbg !13
  %264 = load <4 x half>, ptr addrspace(3) %kdma.gep2181, align 8, !dbg !13, !frisk.k_lds_pair !24
  %kdma.gep2185 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8320, !dbg !13
  %265 = load <4 x half>, ptr addrspace(3) %kdma.gep2185, align 8, !dbg !13, !frisk.k_lds_pair !24
  %kdma.gep2182 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12288, !dbg !13
  %266 = load <4 x half>, ptr addrspace(3) %kdma.gep2182, align 8, !dbg !13, !frisk.k_lds_pair !25
  %kdma.gep2186 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12416, !dbg !13
  %267 = load <4 x half>, ptr addrspace(3) %kdma.gep2186, align 8, !dbg !13, !frisk.k_lds_pair !25
  %kdma.gep2183 = getelementptr i8, ptr addrspace(3) %kdma.swizzled2151, i64 8192, !dbg !13
  %268 = load <4 x half>, ptr addrspace(3) %kdma.gep2183, align 8, !dbg !13, !frisk.k_lds_pair !26
  %kdma.gep2187 = getelementptr i8, ptr addrspace(3) %kdma.swizzled2151, i64 8320, !dbg !13
  %269 = load <4 x half>, ptr addrspace(3) %kdma.gep2187, align 8, !dbg !13, !frisk.k_lds_pair !26
  %kdma.gep2184 = getelementptr i8, ptr addrspace(3) %kdma.swizzled2151, i64 12288, !dbg !13
  %270 = load <4 x half>, ptr addrspace(3) %kdma.gep2184, align 8, !dbg !13, !frisk.k_lds_pair !27
  %kdma.gep2188 = getelementptr i8, ptr addrspace(3) %kdma.swizzled2151, i64 12416, !dbg !13
  %271 = load <4 x half>, ptr addrspace(3) %kdma.gep2188, align 8, !dbg !13, !frisk.k_lds_pair !27
  %kdma.gep2189 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9216, !dbg !13
  %272 = load <4 x half>, ptr addrspace(3) %kdma.gep2189, align 8, !dbg !13, !frisk.k_lds_pair !28
  %kdma.gep2193 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9344, !dbg !13
  %273 = load <4 x half>, ptr addrspace(3) %kdma.gep2193, align 8, !dbg !13, !frisk.k_lds_pair !28
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %274 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %264, <4 x float> %214) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %kdma.gep2190 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13312, !dbg !13
  %275 = load <4 x half>, ptr addrspace(3) %kdma.gep2190, align 8, !dbg !13, !frisk.k_lds_pair !29
  %kdma.gep2194 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13440, !dbg !13
  %276 = load <4 x half>, ptr addrspace(3) %kdma.gep2194, align 8, !dbg !13, !frisk.k_lds_pair !29
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %277 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %266, <4 x float> %274) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %kdma.gep2191 = getelementptr i8, ptr addrspace(3) %kdma.swizzled2151, i64 9216, !dbg !13
  %278 = load <4 x half>, ptr addrspace(3) %kdma.gep2191, align 8, !dbg !13, !frisk.k_lds_pair !30
  %kdma.gep2195 = getelementptr i8, ptr addrspace(3) %kdma.swizzled2151, i64 9344, !dbg !13
  %279 = load <4 x half>, ptr addrspace(3) %kdma.gep2195, align 8, !dbg !13, !frisk.k_lds_pair !30
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %280 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %268, <4 x float> %220) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %kdma.gep2192 = getelementptr i8, ptr addrspace(3) %kdma.swizzled2151, i64 13312, !dbg !13
  %281 = load <4 x half>, ptr addrspace(3) %kdma.gep2192, align 8, !dbg !13, !frisk.k_lds_pair !31
  %kdma.gep2196 = getelementptr i8, ptr addrspace(3) %kdma.swizzled2151, i64 13440, !dbg !13
  %282 = load <4 x half>, ptr addrspace(3) %kdma.gep2196, align 8, !dbg !13, !frisk.k_lds_pair !31
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %283 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %270, <4 x float> %280) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %284 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %265, <4 x float> %224) #8, !dbg !13
  %285 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %267, <4 x float> %284) #8, !dbg !13
  %286 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %269, <4 x float> %226) #8, !dbg !13
  %287 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %271, <4 x float> %286) #8, !dbg !13
  %288 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %272, <4 x float> %228) #8, !dbg !13
  %289 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %275, <4 x float> %288) #8, !dbg !13
  %290 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %278, <4 x float> %230) #8, !dbg !13
  %291 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %281, <4 x float> %290) #8, !dbg !13
  %292 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %273, <4 x float> %232) #8, !dbg !13
  %293 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %276, <4 x float> %292) #8, !dbg !13
  %294 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %279, <4 x float> %234) #8, !dbg !13
  %295 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %282, <4 x float> %294) #8, !dbg !13
  %296 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %264, <4 x float> %236) #8, !dbg !13
  %297 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %266, <4 x float> %296) #8, !dbg !13
  %298 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %268, <4 x float> %238) #8, !dbg !13
  %299 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %270, <4 x float> %298) #8, !dbg !13
  %300 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %265, <4 x float> %240) #8, !dbg !13
  %301 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %267, <4 x float> %300) #8, !dbg !13
  %302 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %269, <4 x float> %242) #8, !dbg !13
  %303 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %271, <4 x float> %302) #8, !dbg !13
  %304 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %272, <4 x float> %244) #8, !dbg !13
  %305 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %275, <4 x float> %304) #8, !dbg !13
  %306 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %278, <4 x float> %246) #8, !dbg !13
  %307 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %281, <4 x float> %306) #8, !dbg !13
  %308 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %273, <4 x float> %248) #8, !dbg !13
  %309 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %276, <4 x float> %308) #8, !dbg !13
  %310 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %279, <4 x float> %250) #8, !dbg !13
  %311 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %282, <4 x float> %310) #8, !dbg !13
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  tail call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %312 = add nuw nsw i64 %130, 3072, !dbg !12
  %313 = lshr i64 %312, 12, !dbg !12
  %314 = add nuw nsw i64 %313, %17, !dbg !12
  %315 = add nsw i64 %139, 96, !dbg !12
  %316 = icmp slt i64 %139, -96, !dbg !12
  %317 = sub nsw i64 -97, %139, !dbg !12
  %318 = select i1 %316, i64 %317, i64 %315, !dbg !12
  %319 = sdiv i64 %318, 128, !dbg !12
  %320 = sext i1 %316 to i64, !dbg !12
  %321 = xor i64 %319, %320, !dbg !12
  %322 = shl i64 %314, 19, !dbg !12
  %323 = and i64 %322, 16252928, !dbg !12
  %324 = add i64 %.idx1745, %323, !dbg !12
  %325 = shl nsw i64 %321, 7, !dbg !12
  %326 = sub i64 %324, %325, !dbg !12
  %kdma.source2092 = add i64 %326, %k0.prime.kdma.source1887, !dbg !12
  %kdma.source2093 = trunc i64 %kdma.source2092 to i32, !dbg !12
  %kdma.source2094 = shl i32 %kdma.source2093, 1, !dbg !12
  %kdma.offset2096 = add i32 %kdma.source2095, %kdma.source2094, !dbg !12
  %327 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1696, i32 %kdma.offset2096, <4 x i32> %k0.prime.dma.buffer.rsrc1701) #9, !dbg !12
  %kdma.source2109 = add i64 %326, %k0.prime.kdma.source1901, !dbg !12
  %kdma.source2110 = trunc i64 %kdma.source2109 to i32, !dbg !12
  %kdma.source2111 = shl i32 %kdma.source2110, 1, !dbg !12
  %kdma.offset2113.reass.reass = add i32 %kdma.source2111, %invariant.op115
  %328 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1974, i32 %kdma.offset2113.reass.reass, <4 x i32> %k0.prime.dma.buffer.rsrc1701) #9, !dbg !12
  %kdma.source2127 = add i64 %326, %k0.prime.kdma.source1918, !dbg !12
  %kdma.source2128 = trunc i64 %kdma.source2127 to i32, !dbg !12
  %kdma.source2129 = shl i32 %kdma.source2128, 1, !dbg !12
  %kdma.offset2131 = add i32 %kdma.source2095, %kdma.source2129, !dbg !12
  %329 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1693, i32 %kdma.offset2131, <4 x i32> %k0.prime.dma.buffer.rsrc1701) #9, !dbg !12
  %kdma.source2145 = add i64 %326, %k0.prime.kdma.source1935, !dbg !12
  %kdma.source2146 = trunc i64 %kdma.source2145 to i32, !dbg !12
  %kdma.source2147 = shl i32 %kdma.source2146, 1, !dbg !12
  %kdma.offset2149.reass.reass = add i32 %kdma.source2147, %invariant.op116
  %330 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds2010, i32 %kdma.offset2149.reass.reass, <4 x i32> %k0.prime.dma.buffer.rsrc1701) #9, !dbg !12
  %331 = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8, !dbg !13, !frisk.k_lds_pair !32
  %332 = load <4 x half>, ptr addrspace(3) %kdma.gep2153, align 8, !dbg !13, !frisk.k_lds_pair !32
  %333 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8, !dbg !13, !frisk.k_lds_pair !33
  %334 = load <4 x half>, ptr addrspace(3) %kdma.gep2154, align 8, !dbg !13, !frisk.k_lds_pair !33
  %335 = load <4 x half>, ptr addrspace(3) %kdma.swizzled2151, align 8, !dbg !13, !frisk.k_lds_pair !34
  %336 = load <4 x half>, ptr addrspace(3) %kdma.gep2155, align 8, !dbg !13, !frisk.k_lds_pair !34
  %337 = load <4 x half>, ptr addrspace(3) %kdma.gep2152, align 8, !dbg !13, !frisk.k_lds_pair !35
  %338 = load <4 x half>, ptr addrspace(3) %kdma.gep2156, align 8, !dbg !13, !frisk.k_lds_pair !35
  %339 = load <4 x half>, ptr addrspace(3) %kdma.gep2157, align 8, !dbg !13, !frisk.k_lds_pair !36
  %340 = load <4 x half>, ptr addrspace(3) %kdma.gep2161, align 8, !dbg !13, !frisk.k_lds_pair !36
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %341 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %331, <4 x float> %277) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %342 = load <4 x half>, ptr addrspace(3) %kdma.gep2158, align 8, !dbg !13, !frisk.k_lds_pair !37
  %343 = load <4 x half>, ptr addrspace(3) %kdma.gep2162, align 8, !dbg !13, !frisk.k_lds_pair !37
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %344 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %333, <4 x float> %341) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %345 = load <4 x half>, ptr addrspace(3) %kdma.gep2159, align 8, !dbg !13, !frisk.k_lds_pair !38
  %346 = load <4 x half>, ptr addrspace(3) %kdma.gep2163, align 8, !dbg !13, !frisk.k_lds_pair !38
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %347 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %335, <4 x float> %283) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %348 = load <4 x half>, ptr addrspace(3) %kdma.gep2160, align 8, !dbg !13, !frisk.k_lds_pair !39
  %349 = load <4 x half>, ptr addrspace(3) %kdma.gep2164, align 8, !dbg !13, !frisk.k_lds_pair !39
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %350 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %337, <4 x float> %347) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %351 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %332, <4 x float> %285) #8, !dbg !13
  %352 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %334, <4 x float> %351) #8, !dbg !13
  %353 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %336, <4 x float> %287) #8, !dbg !13
  %354 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %338, <4 x float> %353) #8, !dbg !13
  %355 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %339, <4 x float> %289) #8, !dbg !13
  %356 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %342, <4 x float> %355) #8, !dbg !13
  %357 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %345, <4 x float> %291) #8, !dbg !13
  %358 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %348, <4 x float> %357) #8, !dbg !13
  %359 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %340, <4 x float> %293) #8, !dbg !13
  %360 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %343, <4 x float> %359) #8, !dbg !13
  %361 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %346, <4 x float> %295) #8, !dbg !13
  %362 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %349, <4 x float> %361) #8, !dbg !13
  %363 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %331, <4 x float> %297) #8, !dbg !13
  %364 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %333, <4 x float> %363) #8, !dbg !13
  %365 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %335, <4 x float> %299) #8, !dbg !13
  %366 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %337, <4 x float> %365) #8, !dbg !13
  %367 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %332, <4 x float> %301) #8, !dbg !13
  %368 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %334, <4 x float> %367) #8, !dbg !13
  %369 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %336, <4 x float> %303) #8, !dbg !13
  %370 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %338, <4 x float> %369) #8, !dbg !13
  %371 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %339, <4 x float> %305) #8, !dbg !13
  %372 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %342, <4 x float> %371) #8, !dbg !13
  %373 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %345, <4 x float> %307) #8, !dbg !13
  %374 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %348, <4 x float> %373) #8, !dbg !13
  %375 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %340, <4 x float> %309) #8, !dbg !13
  %376 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %343, <4 x float> %375) #8, !dbg !13
  %377 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %346, <4 x float> %311) #8, !dbg !13
  %378 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %349, <4 x float> %377) #8, !dbg !13
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  tail call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %379 = add nuw nsw i64 %131, %17, !dbg !40
  %380 = shl i64 %379, 19, !dbg !40
  %381 = and i64 %380, 16252928, !dbg !40
  %address.local985 = shl nuw nsw i64 %address.local567, 5, !dbg !40
  %address.local986 = and i64 %address.local985, 8064, !dbg !40
  %address.local988 = and i64 %address.local568, 24, !dbg !40
  %address.local989 = or disjoint i64 %address.local986, %address.local988, !dbg !40
  %.idx17771779 = add i64 %381, %135, !dbg !40
  %382 = or disjoint i64 %address.local989, %.idx17771779, !dbg !40
  %.tr1780 = trunc i64 %382 to i32, !dbg !40
  %dma.buffer.offset1676 = shl i32 %.tr1780, 1, !dbg !40
  %383 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1702, i32 %dma.buffer.offset1676, <4 x i32> %dma.buffer.rsrc1677) #9, !dbg !40
  %address.local991 = shl nuw nsw i64 %address.local584, 5, !dbg !40
  %address.local992 = and i64 %address.local991, 16256, !dbg !40
  %address.local995 = or disjoint i64 %address.local992, %address.local988, !dbg !40
  %384 = or disjoint i64 %address.local995, %.idx17771779, !dbg !40
  %.tr1784 = trunc i64 %384 to i32, !dbg !40
  %dma.buffer.offset1673 = shl i32 %.tr1784, 1, !dbg !40
  %385 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1699, i32 %dma.buffer.offset1673, <4 x i32> %dma.buffer.rsrc1677) #9, !dbg !40
  %386 = load <4 x half>, ptr addrspace(3) %kdma.gep2181, align 8, !dbg !13
  %387 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %386, <4 x float> %344) #8, !dbg !13
  %388 = or disjoint i64 %132, %111, !dbg !14
  %389 = lshr i64 %130, 25, !dbg !16
  %compare.high1269 = trunc nuw nsw i64 %389 to i32, !dbg !16
  %compare.low.base1270 = trunc i64 %388 to i32, !dbg !16
  %compare.local.base = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 1) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1273 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 2) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1277 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 3) #8, !dbg !16, !frisk.register_lifetimes !5
  %390 = insertelement <4 x i32> poison, i32 %compare.low.base1270, i64 0, !dbg !16
  %391 = insertelement <4 x i32> %390, i32 %compare.local.base, i64 1, !dbg !16
  %392 = insertelement <4 x i32> %391, i32 %compare.local.base1273, i64 2, !dbg !16
  %393 = insertelement <4 x i32> %392, i32 %compare.local.base1277, i64 3, !dbg !16
  %394 = or <4 x i32> %393, <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %395 = icmp ugt <4 x i32> %122, %394, !dbg !16
  %compare.local.base1281 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 4) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1285 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 5) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1289 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 6) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1293 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 7) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1297 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 8) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1301 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 9) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1305 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 10) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1309 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 11) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1313 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 12) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1317 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 13) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1321 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 14) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1325 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 15) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1329 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 16) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1333 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 17) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1337 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 18) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1341 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 19) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1345 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 20) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1349 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 21) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1353 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 22) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1357 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 23) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1361 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 24) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1365 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 25) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1369 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 26) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1373 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 27) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1377 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 28) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1381 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 29) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1385 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 30) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1389 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 31) #8, !dbg !16, !frisk.register_lifetimes !5
  %396 = insertelement <4 x i32> poison, i32 %compare.local.base1361, i64 0, !dbg !16
  %397 = insertelement <4 x i32> %396, i32 %compare.local.base1365, i64 1, !dbg !16
  %398 = insertelement <4 x i32> %397, i32 %compare.local.base1369, i64 2, !dbg !16
  %399 = insertelement <4 x i32> %398, i32 %compare.local.base1373, i64 3, !dbg !16
  %400 = or <4 x i32> %399, <i32 72, i32 73, i32 74, i32 75>, !dbg !16
  %401 = icmp ult <4 x i32> %400, %122, !dbg !16
  %402 = insertelement <4 x i32> poison, i32 %compare.local.base1345, i64 0, !dbg !16
  %403 = insertelement <4 x i32> %402, i32 %compare.local.base1349, i64 1, !dbg !16
  %404 = insertelement <4 x i32> %403, i32 %compare.local.base1353, i64 2, !dbg !16
  %405 = insertelement <4 x i32> %404, i32 %compare.local.base1357, i64 3, !dbg !16
  %406 = or <4 x i32> %405, <i32 68, i32 69, i32 70, i32 71>, !dbg !16
  %407 = icmp ult <4 x i32> %406, %122, !dbg !16
  %408 = insertelement <4 x i32> poison, i32 %compare.local.base1329, i64 0, !dbg !16
  %409 = insertelement <4 x i32> %408, i32 %compare.local.base1333, i64 1, !dbg !16
  %410 = insertelement <4 x i32> %409, i32 %compare.local.base1337, i64 2, !dbg !16
  %411 = insertelement <4 x i32> %410, i32 %compare.local.base1341, i64 3, !dbg !16
  %412 = or <4 x i32> %411, <i32 64, i32 65, i32 66, i32 67>, !dbg !16
  %413 = icmp ult <4 x i32> %412, %122, !dbg !16
  %414 = insertelement <4 x i32> poison, i32 %compare.local.base1313, i64 0, !dbg !16
  %415 = insertelement <4 x i32> %414, i32 %compare.local.base1317, i64 1, !dbg !16
  %416 = insertelement <4 x i32> %415, i32 %compare.local.base1321, i64 2, !dbg !16
  %417 = insertelement <4 x i32> %416, i32 %compare.local.base1325, i64 3, !dbg !16
  %418 = or <4 x i32> %417, <i32 12, i32 13, i32 14, i32 15>, !dbg !16
  %419 = icmp ult <4 x i32> %418, %122, !dbg !16
  %420 = insertelement <4 x i32> poison, i32 %compare.local.base1297, i64 0, !dbg !16
  %421 = insertelement <4 x i32> %420, i32 %compare.local.base1301, i64 1, !dbg !16
  %422 = insertelement <4 x i32> %421, i32 %compare.local.base1305, i64 2, !dbg !16
  %423 = insertelement <4 x i32> %422, i32 %compare.local.base1309, i64 3, !dbg !16
  %424 = or <4 x i32> %423, <i32 8, i32 9, i32 10, i32 11>, !dbg !16
  %425 = icmp ult <4 x i32> %424, %122, !dbg !16
  %426 = insertelement <4 x i32> poison, i32 %compare.local.base1281, i64 0, !dbg !16
  %427 = insertelement <4 x i32> %426, i32 %compare.local.base1285, i64 1, !dbg !16
  %428 = insertelement <4 x i32> %427, i32 %compare.local.base1289, i64 2, !dbg !16
  %429 = insertelement <4 x i32> %428, i32 %compare.local.base1293, i64 3, !dbg !16
  %430 = or <4 x i32> %429, <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %431 = icmp ult <4 x i32> %430, %122, !dbg !16
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high1269, !dbg !16
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high1269, !dbg !16
  %432 = insertelement <4 x i1> poison, i1 %compare.high.equal, i64 0, !dbg !16
  %433 = shufflevector <4 x i1> %432, <4 x i1> poison, <4 x i32> zeroinitializer, !dbg !16
  %434 = insertelement <4 x i1> poison, i1 %compare.high.order, i64 0, !dbg !16
  %435 = shufflevector <4 x i1> %434, <4 x i1> poison, <4 x i32> zeroinitializer, !dbg !16
  %436 = select <4 x i1> %433, <4 x i1> %395, <4 x i1> %435, !dbg !16
  %437 = select <4 x i1> %436, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !41
  %438 = select <4 x i1> %433, <4 x i1> %431, <4 x i1> %435, !dbg !16
  %439 = select <4 x i1> %438, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !41
  %440 = select <4 x i1> %433, <4 x i1> %425, <4 x i1> %435, !dbg !16
  %441 = select <4 x i1> %440, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !41
  %442 = select <4 x i1> %433, <4 x i1> %419, <4 x i1> %435, !dbg !16
  %443 = select <4 x i1> %442, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !41
  %444 = select <4 x i1> %433, <4 x i1> %413, <4 x i1> %435, !dbg !16
  %445 = select <4 x i1> %444, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !41
  %446 = select <4 x i1> %433, <4 x i1> %407, <4 x i1> %435, !dbg !16
  %447 = select <4 x i1> %446, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !41
  %448 = select <4 x i1> %433, <4 x i1> %401, <4 x i1> %435, !dbg !16
  %449 = select <4 x i1> %448, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !41
  %450 = insertelement <4 x i32> poison, i32 %compare.local.base1377, i64 0, !dbg !16
  %451 = insertelement <4 x i32> %450, i32 %compare.local.base1381, i64 1, !dbg !16
  %452 = insertelement <4 x i32> %451, i32 %compare.local.base1385, i64 2, !dbg !16
  %453 = insertelement <4 x i32> %452, i32 %compare.local.base1389, i64 3, !dbg !16
  %454 = or <4 x i32> %453, <i32 76, i32 77, i32 78, i32 79>, !dbg !16
  %455 = icmp ult <4 x i32> %454, %122, !dbg !16
  %456 = select <4 x i1> %433, <4 x i1> %455, <4 x i1> %435, !dbg !16
  %457 = select <4 x i1> %456, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !41
  %compare.local.base1401 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 32) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1405 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 33) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1409 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 34) #8, !dbg !16, !frisk.register_lifetimes !5
  %458 = insertelement <4 x i32> %390, i32 %compare.local.base1401, i64 1, !dbg !16
  %459 = insertelement <4 x i32> %458, i32 %compare.local.base1405, i64 2, !dbg !16
  %460 = insertelement <4 x i32> %459, i32 %compare.local.base1409, i64 3, !dbg !16
  %461 = or <4 x i32> %460, <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %462 = icmp ugt <4 x i32> %120, %461, !dbg !16
  %compare.local.base1413 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 35) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1417 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 36) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1421 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 37) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1425 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 38) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1429 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 39) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1433 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 40) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1437 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 41) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1441 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 42) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1445 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 43) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1449 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 44) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1453 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 45) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1457 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 46) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1461 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 47) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1465 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 48) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1469 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 49) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1473 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 50) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1477 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 51) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1481 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 52) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1485 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 53) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1489 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 54) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1493 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 55) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1497 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 56) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1501 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 57) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1505 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 58) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1509 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 59) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1513 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 60) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1517 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 61) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1521 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 62) #8, !dbg !16, !frisk.register_lifetimes !5
  %463 = insertelement <4 x i32> poison, i32 %compare.local.base1493, i64 0, !dbg !16
  %464 = insertelement <4 x i32> %463, i32 %compare.local.base1497, i64 1, !dbg !16
  %465 = insertelement <4 x i32> %464, i32 %compare.local.base1501, i64 2, !dbg !16
  %466 = insertelement <4 x i32> %465, i32 %compare.local.base1505, i64 3, !dbg !16
  %467 = or <4 x i32> %466, <i32 72, i32 73, i32 74, i32 75>, !dbg !16
  %468 = icmp ult <4 x i32> %467, %120, !dbg !16
  %469 = insertelement <4 x i32> poison, i32 %compare.local.base1477, i64 0, !dbg !16
  %470 = insertelement <4 x i32> %469, i32 %compare.local.base1481, i64 1, !dbg !16
  %471 = insertelement <4 x i32> %470, i32 %compare.local.base1485, i64 2, !dbg !16
  %472 = insertelement <4 x i32> %471, i32 %compare.local.base1489, i64 3, !dbg !16
  %473 = or <4 x i32> %472, <i32 68, i32 69, i32 70, i32 71>, !dbg !16
  %474 = icmp ult <4 x i32> %473, %120, !dbg !16
  %475 = insertelement <4 x i32> poison, i32 %compare.local.base1461, i64 0, !dbg !16
  %476 = insertelement <4 x i32> %475, i32 %compare.local.base1465, i64 1, !dbg !16
  %477 = insertelement <4 x i32> %476, i32 %compare.local.base1469, i64 2, !dbg !16
  %478 = insertelement <4 x i32> %477, i32 %compare.local.base1473, i64 3, !dbg !16
  %479 = or <4 x i32> %478, <i32 64, i32 65, i32 66, i32 67>, !dbg !16
  %480 = icmp ult <4 x i32> %479, %120, !dbg !16
  %481 = insertelement <4 x i32> poison, i32 %compare.local.base1445, i64 0, !dbg !16
  %482 = insertelement <4 x i32> %481, i32 %compare.local.base1449, i64 1, !dbg !16
  %483 = insertelement <4 x i32> %482, i32 %compare.local.base1453, i64 2, !dbg !16
  %484 = insertelement <4 x i32> %483, i32 %compare.local.base1457, i64 3, !dbg !16
  %485 = or <4 x i32> %484, <i32 12, i32 13, i32 14, i32 15>, !dbg !16
  %486 = icmp ult <4 x i32> %485, %120, !dbg !16
  %487 = insertelement <4 x i32> poison, i32 %compare.local.base1429, i64 0, !dbg !16
  %488 = insertelement <4 x i32> %487, i32 %compare.local.base1433, i64 1, !dbg !16
  %489 = insertelement <4 x i32> %488, i32 %compare.local.base1437, i64 2, !dbg !16
  %490 = insertelement <4 x i32> %489, i32 %compare.local.base1441, i64 3, !dbg !16
  %491 = or <4 x i32> %490, <i32 8, i32 9, i32 10, i32 11>, !dbg !16
  %492 = icmp ult <4 x i32> %491, %120, !dbg !16
  %493 = insertelement <4 x i32> poison, i32 %compare.local.base1413, i64 0, !dbg !16
  %494 = insertelement <4 x i32> %493, i32 %compare.local.base1417, i64 1, !dbg !16
  %495 = insertelement <4 x i32> %494, i32 %compare.local.base1421, i64 2, !dbg !16
  %496 = insertelement <4 x i32> %495, i32 %compare.local.base1425, i64 3, !dbg !16
  %497 = or <4 x i32> %496, <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %498 = icmp ult <4 x i32> %497, %120, !dbg !16
  %compare.high.equal1395 = icmp eq i32 %compare.high1393, %compare.high1269, !dbg !16
  %compare.high.order1396 = icmp ugt i32 %compare.high1393, %compare.high1269, !dbg !16
  %499 = insertelement <4 x i1> poison, i1 %compare.high.equal1395, i64 0, !dbg !16
  %500 = shufflevector <4 x i1> %499, <4 x i1> poison, <4 x i32> zeroinitializer, !dbg !16
  %501 = insertelement <4 x i1> poison, i1 %compare.high.order1396, i64 0, !dbg !16
  %502 = shufflevector <4 x i1> %501, <4 x i1> poison, <4 x i32> zeroinitializer, !dbg !16
  %503 = select <4 x i1> %500, <4 x i1> %462, <4 x i1> %502, !dbg !16
  %504 = select <4 x i1> %503, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !41
  %505 = select <4 x i1> %500, <4 x i1> %498, <4 x i1> %502, !dbg !16
  %506 = select <4 x i1> %505, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !41
  %507 = select <4 x i1> %500, <4 x i1> %492, <4 x i1> %502, !dbg !16
  %508 = select <4 x i1> %507, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !41
  %509 = select <4 x i1> %500, <4 x i1> %486, <4 x i1> %502, !dbg !16
  %510 = select <4 x i1> %509, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !41
  %511 = select <4 x i1> %500, <4 x i1> %480, <4 x i1> %502, !dbg !16
  %512 = select <4 x i1> %511, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !41
  %513 = select <4 x i1> %500, <4 x i1> %474, <4 x i1> %502, !dbg !16
  %514 = select <4 x i1> %513, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !41
  %515 = select <4 x i1> %500, <4 x i1> %468, <4 x i1> %502, !dbg !16
  %516 = select <4 x i1> %515, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !41
  %517 = insertelement <4 x i32> poison, i32 %compare.local.base1509, i64 0, !dbg !16
  %518 = insertelement <4 x i32> %517, i32 %compare.local.base1513, i64 1, !dbg !16
  %519 = insertelement <4 x i32> %518, i32 %compare.local.base1517, i64 2, !dbg !16
  %520 = insertelement <4 x i32> %519, i32 %compare.local.base1521, i64 3, !dbg !16
  %521 = or <4 x i32> %520, <i32 76, i32 77, i32 78, i32 79>, !dbg !16
  %522 = icmp ult <4 x i32> %521, %120, !dbg !16
  %523 = select <4 x i1> %500, <4 x i1> %522, <4 x i1> %502, !dbg !16
  %524 = select <4 x i1> %523, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !41
  %525 = load <4 x half>, ptr addrspace(3) %kdma.gep2182, align 8, !dbg !13, !frisk.k_tail_pair !17
  %526 = load <4 x half>, ptr addrspace(3) %kdma.gep2186, align 8, !dbg !13, !frisk.k_tail_pair !17
  %527 = load <4 x half>, ptr addrspace(3) %kdma.gep2183, align 8, !dbg !13, !frisk.k_tail_pair !18
  %528 = load <4 x half>, ptr addrspace(3) %kdma.gep2187, align 8, !dbg !13, !frisk.k_tail_pair !18
  %529 = load <4 x half>, ptr addrspace(3) %kdma.gep2184, align 8, !dbg !13, !frisk.k_tail_pair !19
  %530 = load <4 x half>, ptr addrspace(3) %kdma.gep2188, align 8, !dbg !13, !frisk.k_tail_pair !19
  %531 = load <4 x half>, ptr addrspace(3) %kdma.gep2185, align 8, !dbg !13, !frisk.k_tail_pair !20
  %532 = load <4 x half>, ptr addrspace(3) %kdma.gep2189, align 8, !dbg !13, !frisk.k_tail_pair !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_tail_prefetch !5
  %533 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %525, <4 x float> %387) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_tail_prefetch !5
  %534 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %527, <4 x float> %350) #8, !dbg !13
  %535 = fadd <4 x float> %437, %533, !dbg !42
  %536 = extractelement <4 x float> %535, i64 0, !dbg !42
  %537 = tail call float @llvm.maxnum.f32(float %536, float -inf), !dbg !43, !frisk.local_reduction !5
  %538 = extractelement <4 x float> %535, i64 1, !dbg !42
  %539 = extractelement <4 x float> %535, i64 2, !dbg !42
  %540 = extractelement <4 x float> %535, i64 3, !dbg !42
  %541 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %529, <4 x float> %534) #8, !dbg !13
  %542 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %531, <4 x float> %352) #8, !dbg !13
  %543 = fadd <4 x float> %439, %541, !dbg !42
  %544 = extractelement <4 x float> %543, i64 0, !dbg !42
  %local.reduce219 = tail call float @llvm.maxnum.f32(float %537, float %544), !dbg !43, !frisk.local_reduction !5
  %545 = extractelement <4 x float> %543, i64 1, !dbg !42
  %local.reduce220 = tail call float @llvm.maxnum.f32(float %538, float %545), !dbg !43, !frisk.local_reduction !5
  %546 = extractelement <4 x float> %543, i64 2, !dbg !42
  %local.reduce221 = tail call float @llvm.maxnum.f32(float %539, float %546), !dbg !43, !frisk.local_reduction !5
  %547 = extractelement <4 x float> %543, i64 3, !dbg !42
  %local.reduce222 = tail call float @llvm.maxnum.f32(float %540, float %547), !dbg !43, !frisk.local_reduction !5
  %548 = load <4 x half>, ptr addrspace(3) %kdma.gep2190, align 8, !dbg !13, !frisk.k_tail_pair !21
  %549 = load <4 x half>, ptr addrspace(3) %kdma.gep2194, align 8, !dbg !13, !frisk.k_tail_pair !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_tail_prefetch !5
  %550 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %526, <4 x float> %542) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_tail_prefetch !5
  %551 = load <4 x half>, ptr addrspace(3) %kdma.gep2191, align 8, !dbg !13, !frisk.k_tail_pair !22
  %552 = load <4 x half>, ptr addrspace(3) %kdma.gep2195, align 8, !dbg !13, !frisk.k_tail_pair !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_tail_prefetch !5
  %553 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %528, <4 x float> %354) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.k_tail_prefetch !5
  %554 = fadd <4 x float> %441, %550, !dbg !42
  %555 = extractelement <4 x float> %554, i64 0, !dbg !42
  %local.reduce223 = tail call float @llvm.maxnum.f32(float %local.reduce219, float %555), !dbg !43, !frisk.local_reduction !5
  %556 = extractelement <4 x float> %554, i64 1, !dbg !42
  %local.reduce224 = tail call float @llvm.maxnum.f32(float %local.reduce220, float %556), !dbg !43, !frisk.local_reduction !5
  %557 = extractelement <4 x float> %554, i64 2, !dbg !42
  %local.reduce225 = tail call float @llvm.maxnum.f32(float %local.reduce221, float %557), !dbg !43, !frisk.local_reduction !5
  %558 = extractelement <4 x float> %554, i64 3, !dbg !42
  %local.reduce226 = tail call float @llvm.maxnum.f32(float %local.reduce222, float %558), !dbg !43, !frisk.local_reduction !5
  %559 = load <4 x half>, ptr addrspace(3) %kdma.gep2192, align 8, !dbg !13, !frisk.k_tail_pair !23
  %560 = load <4 x half>, ptr addrspace(3) %kdma.gep2196, align 8, !dbg !13, !frisk.k_tail_pair !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_tail_prefetch !5
  %561 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %530, <4 x float> %553) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_tail_prefetch !5
  %562 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %532, <4 x float> %356) #8, !dbg !13
  %563 = fadd <4 x float> %443, %561, !dbg !42
  %564 = extractelement <4 x float> %563, i64 0, !dbg !42
  %local.reduce227 = tail call float @llvm.maxnum.f32(float %local.reduce223, float %564), !dbg !43, !frisk.local_reduction !5
  %565 = extractelement <4 x float> %563, i64 1, !dbg !42
  %local.reduce228 = tail call float @llvm.maxnum.f32(float %local.reduce224, float %565), !dbg !43, !frisk.local_reduction !5
  %566 = extractelement <4 x float> %563, i64 2, !dbg !42
  %local.reduce229 = tail call float @llvm.maxnum.f32(float %local.reduce225, float %566), !dbg !43, !frisk.local_reduction !5
  %567 = extractelement <4 x float> %563, i64 3, !dbg !42
  %local.reduce230 = tail call float @llvm.maxnum.f32(float %local.reduce226, float %567), !dbg !43, !frisk.local_reduction !5
  %568 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %548, <4 x float> %562) #8, !dbg !13
  %569 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %551, <4 x float> %358) #8, !dbg !13
  %570 = fadd <4 x float> %445, %568, !dbg !42
  %571 = extractelement <4 x float> %570, i64 0, !dbg !42
  %local.reduce231 = tail call float @llvm.maxnum.f32(float %local.reduce227, float %571), !dbg !43, !frisk.local_reduction !5
  %572 = extractelement <4 x float> %570, i64 1, !dbg !42
  %local.reduce232 = tail call float @llvm.maxnum.f32(float %local.reduce228, float %572), !dbg !43, !frisk.local_reduction !5
  %573 = extractelement <4 x float> %570, i64 2, !dbg !42
  %local.reduce233 = tail call float @llvm.maxnum.f32(float %local.reduce229, float %573), !dbg !43, !frisk.local_reduction !5
  %574 = extractelement <4 x float> %570, i64 3, !dbg !42
  %local.reduce234 = tail call float @llvm.maxnum.f32(float %local.reduce230, float %574), !dbg !43, !frisk.local_reduction !5
  %575 = load <4 x half>, ptr addrspace(3) %kdma.gep2193, align 8, !dbg !13
  %576 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %559, <4 x float> %569) #8, !dbg !13
  %577 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %575, <4 x float> %360) #8, !dbg !13
  %578 = fadd <4 x float> %447, %576, !dbg !42
  %579 = extractelement <4 x float> %578, i64 0, !dbg !42
  %local.reduce235 = tail call float @llvm.maxnum.f32(float %local.reduce231, float %579), !dbg !43, !frisk.local_reduction !5
  %580 = extractelement <4 x float> %578, i64 1, !dbg !42
  %local.reduce236 = tail call float @llvm.maxnum.f32(float %local.reduce232, float %580), !dbg !43, !frisk.local_reduction !5
  %581 = extractelement <4 x float> %578, i64 2, !dbg !42
  %local.reduce237 = tail call float @llvm.maxnum.f32(float %local.reduce233, float %581), !dbg !43, !frisk.local_reduction !5
  %582 = extractelement <4 x float> %578, i64 3, !dbg !42
  %local.reduce238 = tail call float @llvm.maxnum.f32(float %local.reduce234, float %582), !dbg !43, !frisk.local_reduction !5
  %583 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %549, <4 x float> %577) #8, !dbg !13
  %584 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %552, <4 x float> %362) #8, !dbg !13
  %585 = fadd <4 x float> %449, %583, !dbg !42
  %586 = extractelement <4 x float> %585, i64 0, !dbg !42
  %local.reduce239 = tail call float @llvm.maxnum.f32(float %local.reduce235, float %586), !dbg !43, !frisk.local_reduction !5
  %587 = extractelement <4 x float> %585, i64 1, !dbg !42
  %local.reduce240 = tail call float @llvm.maxnum.f32(float %local.reduce236, float %587), !dbg !43, !frisk.local_reduction !5
  %588 = extractelement <4 x float> %585, i64 2, !dbg !42
  %local.reduce241 = tail call float @llvm.maxnum.f32(float %local.reduce237, float %588), !dbg !43, !frisk.local_reduction !5
  %589 = extractelement <4 x float> %585, i64 3, !dbg !42
  %local.reduce242 = tail call float @llvm.maxnum.f32(float %local.reduce238, float %589), !dbg !43, !frisk.local_reduction !5
  %590 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %560, <4 x float> %584) #8, !dbg !13
  %591 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %386, <4 x float> %364) #8, !dbg !13
  %592 = fadd <4 x float> %457, %590, !dbg !42
  %593 = extractelement <4 x float> %592, i64 0, !dbg !42
  %local.reduce243 = tail call float @llvm.maxnum.f32(float %local.reduce239, float %593), !dbg !43, !frisk.local_reduction !5
  %594 = extractelement <4 x float> %592, i64 1, !dbg !42
  %local.reduce244 = tail call float @llvm.maxnum.f32(float %local.reduce240, float %594), !dbg !43, !frisk.local_reduction !5
  %595 = extractelement <4 x float> %592, i64 2, !dbg !42
  %local.reduce245 = tail call float @llvm.maxnum.f32(float %local.reduce241, float %595), !dbg !43, !frisk.local_reduction !5
  %596 = extractelement <4 x float> %592, i64 3, !dbg !42
  %local.reduce246 = tail call float @llvm.maxnum.f32(float %local.reduce242, float %596), !dbg !43, !frisk.local_reduction !5
  %local.reduce247 = tail call float @llvm.maxnum.f32(float %local.reduce243, float %local.reduce244), !dbg !43, !frisk.local_reduction !5
  %local.reduce248 = tail call float @llvm.maxnum.f32(float %local.reduce245, float %local.reduce246), !dbg !43, !frisk.local_reduction !5
  %597 = tail call float @llvm.maxnum.f32(float %local.reduce247, float %local.reduce248), !dbg !43, !frisk.local_reduction !5
  %598 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %525, <4 x float> %591) #8, !dbg !13
  %599 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %527, <4 x float> %366) #8, !dbg !13
  %600 = fadd <4 x float> %504, %598, !dbg !42
  %601 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %529, <4 x float> %599) #8, !dbg !13
  %602 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %531, <4 x float> %368) #8, !dbg !13
  %603 = fadd <4 x float> %506, %601, !dbg !42
  %604 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %526, <4 x float> %602) #8, !dbg !13
  %605 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %528, <4 x float> %370) #8, !dbg !13
  %606 = fadd <4 x float> %508, %604, !dbg !42
  %607 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %530, <4 x float> %605) #8, !dbg !13
  %608 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %532, <4 x float> %372) #8, !dbg !13
  %609 = fadd <4 x float> %510, %607, !dbg !42
  %610 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %548, <4 x float> %608) #8, !dbg !13
  %611 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %551, <4 x float> %374) #8, !dbg !13
  %612 = fadd <4 x float> %512, %610, !dbg !42
  %613 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %559, <4 x float> %611) #8, !dbg !13
  %614 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %575, <4 x float> %376) #8, !dbg !13
  %615 = fadd <4 x float> %514, %613, !dbg !42
  %616 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %549, <4 x float> %614) #8, !dbg !13
  %617 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %552, <4 x float> %378) #8, !dbg !13
  %618 = fadd <4 x float> %516, %616, !dbg !42
  %619 = extractelement <4 x float> %600, i64 0, !dbg !42
  %620 = extractelement <4 x float> %600, i64 1, !dbg !42
  %621 = extractelement <4 x float> %600, i64 2, !dbg !42
  %622 = extractelement <4 x float> %600, i64 3, !dbg !42
  %623 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %560, <4 x float> %617) #8, !dbg !13
  %624 = extractelement <4 x float> %603, i64 0, !dbg !42
  %625 = extractelement <4 x float> %603, i64 1, !dbg !42
  %626 = extractelement <4 x float> %603, i64 2, !dbg !42
  %627 = extractelement <4 x float> %603, i64 3, !dbg !42
  %628 = extractelement <4 x float> %606, i64 0, !dbg !42
  %629 = extractelement <4 x float> %606, i64 1, !dbg !42
  %630 = extractelement <4 x float> %606, i64 2, !dbg !42
  %631 = extractelement <4 x float> %606, i64 3, !dbg !42
  %632 = fadd <4 x float> %524, %623, !dbg !42
  %633 = extractelement <4 x float> %609, i64 0, !dbg !42
  %634 = extractelement <4 x float> %609, i64 1, !dbg !42
  %635 = extractelement <4 x float> %609, i64 2, !dbg !42
  %636 = extractelement <4 x float> %609, i64 3, !dbg !42
  %637 = extractelement <4 x float> %612, i64 0, !dbg !42
  %638 = extractelement <4 x float> %612, i64 1, !dbg !42
  %639 = extractelement <4 x float> %612, i64 2, !dbg !42
  %640 = extractelement <4 x float> %612, i64 3, !dbg !42
  %641 = extractelement <4 x float> %615, i64 0, !dbg !42
  %642 = extractelement <4 x float> %615, i64 1, !dbg !42
  %643 = extractelement <4 x float> %615, i64 2, !dbg !42
  %644 = extractelement <4 x float> %615, i64 3, !dbg !42
  %645 = extractelement <4 x float> %618, i64 0, !dbg !42
  %646 = extractelement <4 x float> %618, i64 1, !dbg !42
  %647 = extractelement <4 x float> %618, i64 2, !dbg !42
  %648 = extractelement <4 x float> %618, i64 3, !dbg !42
  %649 = extractelement <4 x float> %632, i64 0, !dbg !42
  %650 = extractelement <4 x float> %632, i64 1, !dbg !42
  %651 = extractelement <4 x float> %632, i64 2, !dbg !42
  %652 = extractelement <4 x float> %632, i64 3, !dbg !42
  %653 = tail call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !43
  %654 = tail call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %653), !dbg !43
  %655 = add nuw nsw i32 %654, 64, !dbg !43
  %656 = xor i32 %654, 16, !dbg !43
  %657 = and i32 %655, 192, !dbg !43
  %658 = icmp samesign ult i32 %656, %657, !dbg !43
  %659 = select i1 %658, i32 %656, i32 %654, !dbg !43
  %660 = shl nuw nsw i32 %659, 2, !dbg !43
  %661 = bitcast float %597 to i32, !dbg !43
  %662 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %660, i32 %661), !dbg !43
  %663 = bitcast i32 %662 to float, !dbg !43
  %664 = tail call float @llvm.maxnum.f32(float %597, float %663), !dbg !43
  %665 = xor i32 %654, 32, !dbg !43
  %666 = icmp samesign ult i32 %665, %657, !dbg !43
  %667 = select i1 %666, i32 %665, i32 %654, !dbg !43
  %668 = shl nuw nsw i32 %667, 2, !dbg !43
  %669 = bitcast float %664 to i32, !dbg !43
  %670 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %668, i32 %669), !dbg !43
  %671 = bitcast i32 %670 to float, !dbg !43
  %672 = tail call float @llvm.maxnum.f32(float %664, float %671), !dbg !43
  %673 = extractvalue [2 x <1 x float>] %127, 0, !dbg !43
  %674 = insertelement <1 x float> poison, float %672, i64 0, !dbg !43
  %675 = tail call <1 x float> @llvm.maxnum.v1f32(<1 x float> %674, <1 x float> %673), !dbg !43
  %676 = tail call float @llvm.maxnum.f32(float %619, float -inf), !dbg !43, !frisk.local_reduction !5
  %local.reduce188 = tail call float @llvm.maxnum.f32(float %676, float %624), !dbg !43, !frisk.local_reduction !5
  %local.reduce189 = tail call float @llvm.maxnum.f32(float %620, float %625), !dbg !43, !frisk.local_reduction !5
  %local.reduce190 = tail call float @llvm.maxnum.f32(float %621, float %626), !dbg !43, !frisk.local_reduction !5
  %local.reduce191 = tail call float @llvm.maxnum.f32(float %622, float %627), !dbg !43, !frisk.local_reduction !5
  %local.reduce192 = tail call float @llvm.maxnum.f32(float %local.reduce188, float %628), !dbg !43, !frisk.local_reduction !5
  %local.reduce193 = tail call float @llvm.maxnum.f32(float %local.reduce189, float %629), !dbg !43, !frisk.local_reduction !5
  %local.reduce194 = tail call float @llvm.maxnum.f32(float %local.reduce190, float %630), !dbg !43, !frisk.local_reduction !5
  %local.reduce195 = tail call float @llvm.maxnum.f32(float %local.reduce191, float %631), !dbg !43, !frisk.local_reduction !5
  %local.reduce196 = tail call float @llvm.maxnum.f32(float %local.reduce192, float %633), !dbg !43, !frisk.local_reduction !5
  %local.reduce197 = tail call float @llvm.maxnum.f32(float %local.reduce193, float %634), !dbg !43, !frisk.local_reduction !5
  %local.reduce198 = tail call float @llvm.maxnum.f32(float %local.reduce194, float %635), !dbg !43, !frisk.local_reduction !5
  %local.reduce199 = tail call float @llvm.maxnum.f32(float %local.reduce195, float %636), !dbg !43, !frisk.local_reduction !5
  %local.reduce200 = tail call float @llvm.maxnum.f32(float %local.reduce196, float %637), !dbg !43, !frisk.local_reduction !5
  %local.reduce201 = tail call float @llvm.maxnum.f32(float %local.reduce197, float %638), !dbg !43, !frisk.local_reduction !5
  %local.reduce202 = tail call float @llvm.maxnum.f32(float %local.reduce198, float %639), !dbg !43, !frisk.local_reduction !5
  %local.reduce203 = tail call float @llvm.maxnum.f32(float %local.reduce199, float %640), !dbg !43, !frisk.local_reduction !5
  %local.reduce204 = tail call float @llvm.maxnum.f32(float %local.reduce200, float %641), !dbg !43, !frisk.local_reduction !5
  %local.reduce205 = tail call float @llvm.maxnum.f32(float %local.reduce201, float %642), !dbg !43, !frisk.local_reduction !5
  %local.reduce206 = tail call float @llvm.maxnum.f32(float %local.reduce202, float %643), !dbg !43, !frisk.local_reduction !5
  %local.reduce207 = tail call float @llvm.maxnum.f32(float %local.reduce203, float %644), !dbg !43, !frisk.local_reduction !5
  %local.reduce208 = tail call float @llvm.maxnum.f32(float %local.reduce204, float %645), !dbg !43, !frisk.local_reduction !5
  %local.reduce209 = tail call float @llvm.maxnum.f32(float %local.reduce205, float %646), !dbg !43, !frisk.local_reduction !5
  %local.reduce210 = tail call float @llvm.maxnum.f32(float %local.reduce206, float %647), !dbg !43, !frisk.local_reduction !5
  %local.reduce211 = tail call float @llvm.maxnum.f32(float %local.reduce207, float %648), !dbg !43, !frisk.local_reduction !5
  %local.reduce212 = tail call float @llvm.maxnum.f32(float %local.reduce208, float %649), !dbg !43, !frisk.local_reduction !5
  %local.reduce213 = tail call float @llvm.maxnum.f32(float %local.reduce209, float %650), !dbg !43, !frisk.local_reduction !5
  %local.reduce214 = tail call float @llvm.maxnum.f32(float %local.reduce210, float %651), !dbg !43, !frisk.local_reduction !5
  %local.reduce215 = tail call float @llvm.maxnum.f32(float %local.reduce211, float %652), !dbg !43, !frisk.local_reduction !5
  %local.reduce216 = tail call float @llvm.maxnum.f32(float %local.reduce212, float %local.reduce213), !dbg !43, !frisk.local_reduction !5
  %local.reduce217 = tail call float @llvm.maxnum.f32(float %local.reduce214, float %local.reduce215), !dbg !43, !frisk.local_reduction !5
  %677 = tail call float @llvm.maxnum.f32(float %local.reduce216, float %local.reduce217), !dbg !43, !frisk.local_reduction !5
  %678 = bitcast float %677 to i32, !dbg !43
  %679 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %660, i32 %678), !dbg !43
  %680 = bitcast i32 %679 to float, !dbg !43
  %681 = tail call float @llvm.maxnum.f32(float %677, float %680), !dbg !43
  %682 = bitcast float %681 to i32, !dbg !43
  %683 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %668, i32 %682), !dbg !43
  %684 = bitcast i32 %683 to float, !dbg !43
  %685 = tail call float @llvm.maxnum.f32(float %681, float %684), !dbg !43
  %686 = extractvalue [2 x <1 x float>] %127, 1, !dbg !43
  %687 = insertelement <1 x float> poison, float %685, i64 0, !dbg !43
  %688 = tail call <1 x float> @llvm.maxnum.v1f32(<1 x float> %687, <1 x float> %686), !dbg !43
  %689 = shufflevector <1 x float> %675, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !44
  %690 = shufflevector <1 x float> %688, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !44
  %691 = fsub <1 x float> %673, %675, !dbg !45
  %692 = extractelement <1 x float> %691, i64 0, !dbg !46
  %693 = tail call float @llvm.amdgcn.exp2.f32(float %692), !dbg !46
  %694 = insertelement <4 x float> poison, float %693, i64 0, !dbg !47
  %695 = shufflevector <4 x float> %694, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !47
  %696 = extractvalue [2 x <1 x float>] %128, 0, !dbg !48
  %697 = extractelement <1 x float> %696, i64 0, !dbg !48
  %698 = fmul float %697, %693, !dbg !48
  %699 = fsub <1 x float> %686, %688, !dbg !45
  %700 = extractelement <1 x float> %699, i64 0, !dbg !46
  %701 = tail call float @llvm.amdgcn.exp2.f32(float %700), !dbg !46
  %702 = insertelement <4 x float> poison, float %701, i64 0, !dbg !47
  %703 = shufflevector <4 x float> %702, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !47
  %704 = extractvalue [2 x <1 x float>] %128, 1, !dbg !48
  %705 = extractelement <1 x float> %704, i64 0, !dbg !48
  %706 = fmul float %705, %701, !dbg !48
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !40
  fence syncscope("workgroup") release, !dbg !40
  tail call void @llvm.amdgcn.s.barrier(), !dbg !40
  fence syncscope("workgroup") acquire, !dbg !40
  %vdb.address1802 = or disjoint i64 %.idx17771779, 32, !dbg !40
  %vdb.address1803 = or disjoint i64 %address.local989, %vdb.address1802, !dbg !40
  %vdb.address1804 = trunc i64 %vdb.address1803 to i32, !dbg !40
  %vdb.address1805 = shl i32 %vdb.address1804, 1, !dbg !40
  %vdb.address1813 = or disjoint i64 %address.local995, %vdb.address1802, !dbg !40
  %vdb.address1814 = trunc i64 %vdb.address1813 to i32, !dbg !40
  %vdb.address1815 = shl i32 %vdb.address1814, 1, !dbg !40
  %vdb.dma.base = add i32 %dma.buffer.lds1702, 8192, !dbg !49
  %707 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1805, <4 x i32> %dma.buffer.rsrc1677) #9, !dbg !40
  %vdb.dma.base1820 = add i32 %dma.buffer.lds1699, 8192, !dbg !49
  %708 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1820, i32 %vdb.address1815, <4 x i32> %dma.buffer.rsrc1677) #9, !dbg !40
  %address.local997 = shl nuw nsw i32 %address.thread566, 3, !dbg !49
  %address.local998 = and i32 %address.local997, 120, !dbg !49
  %address.local1000 = and i32 %address.local643, 1536, !dbg !49
  %address.local1001 = or disjoint i32 %address.local998, %address.local1000, !dbg !49
  %address.local1002 = zext nneg i32 %address.local1001 to i64, !dbg !49
  %address.local1003 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1002, !dbg !49
  %709 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1003, i32 0) #11, !dbg !49, !frisk.v_double_buffer !50
  %710 = bitcast <4 x i32> %709 to <8 x half>, !dbg !49
  %address.local1004 = and i64 %address.local567, 60, !dbg !49
  %address.local1005 = add nuw nsw i64 %address.local1004, %address.local567, !dbg !49
  %address.local1007.idx = shl nuw nsw i64 %address.local1005, 4, !dbg !49
  %address.local1007 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %address.local1007.idx, !dbg !49
  %address.local1008 = lshr i64 %address.local623, 4, !dbg !49
  %address.local1010.idx = mul nuw nsw i64 %address.local1008, 768, !dbg !49
  %address.local1010 = getelementptr i8, ptr addrspace(3) %address.local1007, i64 %address.local1010.idx, !dbg !49
  %address.local1012.idx = mul nsw i64 %address.local610, -64, !dbg !49
  %address.local1012 = getelementptr i8, ptr addrspace(3) %address.local1010, i64 %address.local1012.idx, !dbg !49
  %address.local1013 = getelementptr i8, ptr addrspace(3) %address.local1012, i64 256, !dbg !49
  %711 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1013, i32 0) #11, !dbg !49, !frisk.v_double_buffer !50
  %712 = bitcast <4 x i32> %711 to <8 x half>, !dbg !49
  %address.local1017 = getelementptr i8, ptr addrspace(3) %address.local1012, i64 512, !dbg !49
  %713 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1017, i32 0) #11, !dbg !49, !frisk.v_double_buffer !50
  %714 = bitcast <4 x i32> %713 to <8 x half>, !dbg !49
  %address.local1021 = getelementptr i8, ptr addrspace(3) %address.local1012, i64 768, !dbg !49
  %715 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1021, i32 0) #11, !dbg !49, !frisk.v_double_buffer !50
  %716 = bitcast <4 x i32> %715 to <8 x half>, !dbg !49
  %address.local1023 = getelementptr i8, ptr addrspace(3) %address.local1003, i64 4096, !dbg !49
  %717 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1023, i32 0) #11, !dbg !49, !frisk.v_double_buffer !50
  %718 = bitcast <4 x i32> %717 to <8 x half>, !dbg !49
  %address.local1027 = getelementptr i8, ptr addrspace(3) %address.local1012, i64 4352, !dbg !49
  %719 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1027, i32 0) #11, !dbg !49, !frisk.v_double_buffer !50
  %720 = bitcast <4 x i32> %719 to <8 x half>, !dbg !49
  %address.local1031 = getelementptr i8, ptr addrspace(3) %address.local1012, i64 4608, !dbg !49
  %721 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1031, i32 0) #11, !dbg !49, !frisk.v_double_buffer !50
  %722 = bitcast <4 x i32> %721 to <8 x half>, !dbg !49
  %address.local1035 = getelementptr i8, ptr addrspace(3) %address.local1012, i64 4864, !dbg !49
  %723 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1035, i32 0) #11, !dbg !49, !frisk.v_double_buffer !50
  %724 = shufflevector <8 x half> %710, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  %725 = extractvalue [2 x <8 x float>] %129, 0, !dbg !47
  %726 = shufflevector <8 x float> %725, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !47
  %727 = fmul <4 x float> %726, %695, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %728 = fsub <4 x float> %535, %689, !dbg !44
  %729 = extractelement <4 x float> %728, i64 0, !dbg !51
  %730 = tail call float @llvm.amdgcn.exp2.f32(float %729), !dbg !51
  %731 = insertelement <4 x float> poison, float %730, i64 0, !dbg !52
  %732 = extractelement <4 x float> %728, i64 1, !dbg !51
  %733 = tail call float @llvm.amdgcn.exp2.f32(float %732), !dbg !51
  %734 = insertelement <4 x float> %731, float %733, i64 1, !dbg !52
  %735 = extractelement <4 x float> %728, i64 2, !dbg !51
  %736 = tail call float @llvm.amdgcn.exp2.f32(float %735), !dbg !51
  %737 = insertelement <4 x float> %734, float %736, i64 2, !dbg !52
  %738 = extractelement <4 x float> %728, i64 3, !dbg !51
  %739 = tail call float @llvm.amdgcn.exp2.f32(float %738), !dbg !51
  %740 = insertelement <4 x float> %737, float %739, i64 3, !dbg !52
  %math.result = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %740) #8, !dbg !52, !frisk.register_lifetimes !5
  %741 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !52
  %math.packed = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %741) #8, !dbg !51, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49, !frisk.softmax_fragments !5
  %742 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %724, <4 x float> %727) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %743 = shufflevector <8 x half> %710, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  %744 = shufflevector <8 x float> %725, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !47
  %745 = fmul <4 x float> %744, %695, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %746 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %743, <4 x float> %745) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %747 = extractvalue [2 x <8 x float>] %129, 1, !dbg !47
  %748 = shufflevector <8 x float> %747, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !47
  %749 = fmul <4 x float> %748, %703, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %750 = fsub <4 x float> %600, %690, !dbg !44
  %751 = extractelement <4 x float> %750, i64 0, !dbg !51
  %752 = tail call float @llvm.amdgcn.exp2.f32(float %751), !dbg !51
  %753 = insertelement <4 x float> poison, float %752, i64 0, !dbg !52
  %754 = extractelement <4 x float> %750, i64 1, !dbg !51
  %755 = tail call float @llvm.amdgcn.exp2.f32(float %754), !dbg !51
  %756 = insertelement <4 x float> %753, float %755, i64 1, !dbg !52
  %757 = extractelement <4 x float> %750, i64 2, !dbg !51
  %758 = tail call float @llvm.amdgcn.exp2.f32(float %757), !dbg !51
  %759 = insertelement <4 x float> %756, float %758, i64 2, !dbg !52
  %760 = extractelement <4 x float> %750, i64 3, !dbg !51
  %761 = tail call float @llvm.amdgcn.exp2.f32(float %760), !dbg !51
  %762 = insertelement <4 x float> %759, float %761, i64 3, !dbg !52
  %math.result456 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %762) #8, !dbg !52, !frisk.register_lifetimes !5
  %763 = fptrunc <4 x float> %math.result456 to <4 x half>, !dbg !52
  %math.packed457 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %763) #8, !dbg !51, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49, !frisk.softmax_fragments !5
  %764 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed457, <4 x half> %724, <4 x float> %749) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %765 = shufflevector <8 x float> %747, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !47
  %766 = fmul <4 x float> %765, %703, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %767 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed457, <4 x half> %743, <4 x float> %766) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %768 = shufflevector <8 x half> %712, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %769 = fsub <4 x float> %543, %689, !dbg !44
  %770 = extractelement <4 x float> %769, i64 0, !dbg !51
  %771 = tail call float @llvm.amdgcn.exp2.f32(float %770), !dbg !51
  %local.reduce157 = fadd float %730, %771, !dbg !53, !frisk.local_reduction !5
  %k0.sum.anchor = tail call float asm sideeffect "", "=v,0"(float %local.reduce157) #8, !dbg !53, !frisk.cross_bn_sum_anchor !5
  %772 = insertelement <4 x float> poison, float %771, i64 0, !dbg !52
  %773 = extractelement <4 x float> %769, i64 1, !dbg !51
  %774 = tail call float @llvm.amdgcn.exp2.f32(float %773), !dbg !51
  %local.reduce158 = fadd float %733, %774, !dbg !53, !frisk.local_reduction !5
  %k0.sum.anchor2284 = tail call float asm sideeffect "", "=v,0"(float %local.reduce158) #8, !dbg !53, !frisk.cross_bn_sum_anchor !5
  %775 = insertelement <4 x float> %772, float %774, i64 1, !dbg !52
  %776 = extractelement <4 x float> %769, i64 2, !dbg !51
  %777 = tail call float @llvm.amdgcn.exp2.f32(float %776), !dbg !51
  %local.reduce159 = fadd float %736, %777, !dbg !53, !frisk.local_reduction !5
  %k0.sum.anchor2285 = tail call float asm sideeffect "", "=v,0"(float %local.reduce159) #8, !dbg !53, !frisk.cross_bn_sum_anchor !5
  %778 = insertelement <4 x float> %775, float %777, i64 2, !dbg !52
  %779 = extractelement <4 x float> %769, i64 3, !dbg !51
  %780 = tail call float @llvm.amdgcn.exp2.f32(float %779), !dbg !51
  %local.reduce160 = fadd float %739, %780, !dbg !53, !frisk.local_reduction !5
  %k0.sum.anchor2286 = tail call float asm sideeffect "", "=v,0"(float %local.reduce160) #8, !dbg !53, !frisk.cross_bn_sum_anchor !5
  %781 = insertelement <4 x float> %778, float %780, i64 3, !dbg !52
  %math.result442 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %781) #8, !dbg !52, !frisk.register_lifetimes !5
  %782 = fptrunc <4 x float> %math.result442 to <4 x half>, !dbg !52
  %math.packed443 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %782) #8, !dbg !53, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49, !frisk.softmax_fragments !5
  %783 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed443, <4 x half> %768, <4 x float> %742) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %784 = shufflevector <8 x half> %712, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %785 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed443, <4 x half> %784, <4 x float> %746) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %786 = fsub <4 x float> %603, %690, !dbg !44
  %787 = extractelement <4 x float> %786, i64 0, !dbg !51
  %788 = tail call float @llvm.amdgcn.exp2.f32(float %787), !dbg !51
  %local.reduce = fadd float %752, %788, !dbg !53, !frisk.local_reduction !5
  %k0.sum.anchor2287 = tail call float asm sideeffect "", "=v,0"(float %local.reduce) #8, !dbg !53, !frisk.cross_bn_sum_anchor !5
  %789 = insertelement <4 x float> poison, float %788, i64 0, !dbg !52
  %790 = extractelement <4 x float> %786, i64 1, !dbg !51
  %791 = tail call float @llvm.amdgcn.exp2.f32(float %790), !dbg !51
  %local.reduce127 = fadd float %755, %791, !dbg !53, !frisk.local_reduction !5
  %k0.sum.anchor2288 = tail call float asm sideeffect "", "=v,0"(float %local.reduce127) #8, !dbg !53, !frisk.cross_bn_sum_anchor !5
  %792 = insertelement <4 x float> %789, float %791, i64 1, !dbg !52
  %793 = extractelement <4 x float> %786, i64 2, !dbg !51
  %794 = tail call float @llvm.amdgcn.exp2.f32(float %793), !dbg !51
  %local.reduce128 = fadd float %758, %794, !dbg !53, !frisk.local_reduction !5
  %k0.sum.anchor2289 = tail call float asm sideeffect "", "=v,0"(float %local.reduce128) #8, !dbg !53, !frisk.cross_bn_sum_anchor !5
  %795 = insertelement <4 x float> %792, float %794, i64 2, !dbg !52
  %796 = extractelement <4 x float> %786, i64 3, !dbg !51
  %797 = tail call float @llvm.amdgcn.exp2.f32(float %796), !dbg !51
  %local.reduce129 = fadd float %761, %797, !dbg !53, !frisk.local_reduction !5
  %k0.sum.anchor2290 = tail call float asm sideeffect "", "=v,0"(float %local.reduce129) #8, !dbg !53, !frisk.cross_bn_sum_anchor !5
  %798 = insertelement <4 x float> %795, float %797, i64 3, !dbg !52
  %math.result458 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %798) #8, !dbg !52, !frisk.register_lifetimes !5
  %799 = fptrunc <4 x float> %math.result458 to <4 x half>, !dbg !52
  %math.packed459 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %799) #8, !dbg !53, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49, !frisk.softmax_fragments !5
  %800 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed459, <4 x half> %768, <4 x float> %764) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %801 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed459, <4 x half> %784, <4 x float> %767) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %802 = shufflevector <8 x half> %714, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %803 = fsub <4 x float> %554, %689, !dbg !44
  %804 = extractelement <4 x float> %803, i64 0, !dbg !51
  %805 = tail call float @llvm.amdgcn.exp2.f32(float %804), !dbg !51
  %local.reduce161 = fadd float %805, %k0.sum.anchor, !dbg !53, !frisk.local_reduction !5
  %k0.sum.anchor2291 = tail call float asm sideeffect "", "=v,0"(float %local.reduce161) #8, !dbg !53, !frisk.cross_bn_sum_anchor !5
  %806 = insertelement <4 x float> poison, float %805, i64 0, !dbg !52
  %807 = extractelement <4 x float> %803, i64 1, !dbg !51
  %808 = tail call float @llvm.amdgcn.exp2.f32(float %807), !dbg !51
  %local.reduce162 = fadd float %808, %k0.sum.anchor2284, !dbg !53, !frisk.local_reduction !5
  %k0.sum.anchor2292 = tail call float asm sideeffect "", "=v,0"(float %local.reduce162) #8, !dbg !53, !frisk.cross_bn_sum_anchor !5
  %809 = insertelement <4 x float> %806, float %808, i64 1, !dbg !52
  %810 = extractelement <4 x float> %803, i64 2, !dbg !51
  %811 = tail call float @llvm.amdgcn.exp2.f32(float %810), !dbg !51
  %local.reduce163 = fadd float %811, %k0.sum.anchor2285, !dbg !53, !frisk.local_reduction !5
  %k0.sum.anchor2293 = tail call float asm sideeffect "", "=v,0"(float %local.reduce163) #8, !dbg !53, !frisk.cross_bn_sum_anchor !5
  %812 = insertelement <4 x float> %809, float %811, i64 2, !dbg !52
  %813 = extractelement <4 x float> %803, i64 3, !dbg !51
  %814 = tail call float @llvm.amdgcn.exp2.f32(float %813), !dbg !51
  %local.reduce164 = fadd float %814, %k0.sum.anchor2286, !dbg !53, !frisk.local_reduction !5
  %k0.sum.anchor2294 = tail call float asm sideeffect "", "=v,0"(float %local.reduce164) #8, !dbg !53, !frisk.cross_bn_sum_anchor !5
  %815 = insertelement <4 x float> %812, float %814, i64 3, !dbg !52
  %math.result444 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %815) #8, !dbg !52, !frisk.register_lifetimes !5
  %816 = fptrunc <4 x float> %math.result444 to <4 x half>, !dbg !52
  %math.packed445 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %816) #8, !dbg !53, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49, !frisk.softmax_fragments !5
  %817 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed445, <4 x half> %802, <4 x float> %783) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %818 = shufflevector <8 x half> %714, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %819 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed445, <4 x half> %818, <4 x float> %785) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %820 = fsub <4 x float> %606, %690, !dbg !44
  %821 = extractelement <4 x float> %820, i64 0, !dbg !51
  %822 = tail call float @llvm.amdgcn.exp2.f32(float %821), !dbg !51
  %local.reduce130 = fadd float %822, %k0.sum.anchor2287, !dbg !53, !frisk.local_reduction !5
  %k0.sum.anchor2295 = tail call float asm sideeffect "", "=v,0"(float %local.reduce130) #8, !dbg !53, !frisk.cross_bn_sum_anchor !5
  %823 = insertelement <4 x float> poison, float %822, i64 0, !dbg !52
  %824 = extractelement <4 x float> %820, i64 1, !dbg !51
  %825 = tail call float @llvm.amdgcn.exp2.f32(float %824), !dbg !51
  %local.reduce131 = fadd float %825, %k0.sum.anchor2288, !dbg !53, !frisk.local_reduction !5
  %k0.sum.anchor2296 = tail call float asm sideeffect "", "=v,0"(float %local.reduce131) #8, !dbg !53, !frisk.cross_bn_sum_anchor !5
  %826 = insertelement <4 x float> %823, float %825, i64 1, !dbg !52
  %827 = extractelement <4 x float> %820, i64 2, !dbg !51
  %828 = tail call float @llvm.amdgcn.exp2.f32(float %827), !dbg !51
  %local.reduce132 = fadd float %828, %k0.sum.anchor2289, !dbg !53, !frisk.local_reduction !5
  %k0.sum.anchor2297 = tail call float asm sideeffect "", "=v,0"(float %local.reduce132) #8, !dbg !53, !frisk.cross_bn_sum_anchor !5
  %829 = insertelement <4 x float> %826, float %828, i64 2, !dbg !52
  %830 = extractelement <4 x float> %820, i64 3, !dbg !51
  %831 = tail call float @llvm.amdgcn.exp2.f32(float %830), !dbg !51
  %local.reduce133 = fadd float %831, %k0.sum.anchor2290, !dbg !53, !frisk.local_reduction !5
  %k0.sum.anchor2298 = tail call float asm sideeffect "", "=v,0"(float %local.reduce133) #8, !dbg !53, !frisk.cross_bn_sum_anchor !5
  %832 = insertelement <4 x float> %829, float %831, i64 3, !dbg !52
  %math.result460 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %832) #8, !dbg !52, !frisk.register_lifetimes !5
  %833 = fptrunc <4 x float> %math.result460 to <4 x half>, !dbg !52
  %math.packed461 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %833) #8, !dbg !53, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49, !frisk.softmax_fragments !5
  %834 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed461, <4 x half> %802, <4 x float> %800) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %835 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed461, <4 x half> %818, <4 x float> %801) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %836 = shufflevector <8 x half> %716, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %837 = fsub <4 x float> %563, %689, !dbg !44
  %838 = extractelement <4 x float> %837, i64 0, !dbg !51
  %839 = tail call float @llvm.amdgcn.exp2.f32(float %838), !dbg !51
  %local.reduce165 = fadd float %839, %k0.sum.anchor2291, !dbg !53, !frisk.local_reduction !5
  %k0.sum.anchor2299 = tail call float asm sideeffect "", "=v,0"(float %local.reduce165) #8, !dbg !53, !frisk.cross_bn_sum_anchor !5
  %840 = insertelement <4 x float> poison, float %839, i64 0, !dbg !52
  %841 = extractelement <4 x float> %837, i64 1, !dbg !51
  %842 = tail call float @llvm.amdgcn.exp2.f32(float %841), !dbg !51
  %local.reduce166 = fadd float %842, %k0.sum.anchor2292, !dbg !53, !frisk.local_reduction !5
  %k0.sum.anchor2300 = tail call float asm sideeffect "", "=v,0"(float %local.reduce166) #8, !dbg !53, !frisk.cross_bn_sum_anchor !5
  %843 = insertelement <4 x float> %840, float %842, i64 1, !dbg !52
  %844 = extractelement <4 x float> %837, i64 2, !dbg !51
  %845 = tail call float @llvm.amdgcn.exp2.f32(float %844), !dbg !51
  %local.reduce167 = fadd float %845, %k0.sum.anchor2293, !dbg !53, !frisk.local_reduction !5
  %k0.sum.anchor2301 = tail call float asm sideeffect "", "=v,0"(float %local.reduce167) #8, !dbg !53, !frisk.cross_bn_sum_anchor !5
  %846 = insertelement <4 x float> %843, float %845, i64 2, !dbg !52
  %847 = extractelement <4 x float> %837, i64 3, !dbg !51
  %848 = tail call float @llvm.amdgcn.exp2.f32(float %847), !dbg !51
  %local.reduce168 = fadd float %848, %k0.sum.anchor2294, !dbg !53, !frisk.local_reduction !5
  %k0.sum.anchor2302 = tail call float asm sideeffect "", "=v,0"(float %local.reduce168) #8, !dbg !53, !frisk.cross_bn_sum_anchor !5
  %849 = insertelement <4 x float> %846, float %848, i64 3, !dbg !52
  %math.result446 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %849) #8, !dbg !52, !frisk.register_lifetimes !5
  %850 = fptrunc <4 x float> %math.result446 to <4 x half>, !dbg !52
  %math.packed447 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %850) #8, !dbg !53, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49, !frisk.softmax_fragments !5
  %851 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed447, <4 x half> %836, <4 x float> %817) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %852 = shufflevector <8 x half> %716, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %853 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed447, <4 x half> %852, <4 x float> %819) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %854 = fsub <4 x float> %609, %690, !dbg !44
  %855 = extractelement <4 x float> %854, i64 0, !dbg !51
  %856 = tail call float @llvm.amdgcn.exp2.f32(float %855), !dbg !51
  %local.reduce134 = fadd float %856, %k0.sum.anchor2295, !dbg !53, !frisk.local_reduction !5
  %k0.sum.anchor2303 = tail call float asm sideeffect "", "=v,0"(float %local.reduce134) #8, !dbg !53, !frisk.cross_bn_sum_anchor !5
  %857 = insertelement <4 x float> poison, float %856, i64 0, !dbg !52
  %858 = extractelement <4 x float> %854, i64 1, !dbg !51
  %859 = tail call float @llvm.amdgcn.exp2.f32(float %858), !dbg !51
  %local.reduce135 = fadd float %859, %k0.sum.anchor2296, !dbg !53, !frisk.local_reduction !5
  %k0.sum.anchor2304 = tail call float asm sideeffect "", "=v,0"(float %local.reduce135) #8, !dbg !53, !frisk.cross_bn_sum_anchor !5
  %860 = insertelement <4 x float> %857, float %859, i64 1, !dbg !52
  %861 = extractelement <4 x float> %854, i64 2, !dbg !51
  %862 = tail call float @llvm.amdgcn.exp2.f32(float %861), !dbg !51
  %local.reduce136 = fadd float %862, %k0.sum.anchor2297, !dbg !53, !frisk.local_reduction !5
  %k0.sum.anchor2305 = tail call float asm sideeffect "", "=v,0"(float %local.reduce136) #8, !dbg !53, !frisk.cross_bn_sum_anchor !5
  %863 = insertelement <4 x float> %860, float %862, i64 2, !dbg !52
  %864 = extractelement <4 x float> %854, i64 3, !dbg !51
  %865 = tail call float @llvm.amdgcn.exp2.f32(float %864), !dbg !51
  %local.reduce137 = fadd float %865, %k0.sum.anchor2298, !dbg !53, !frisk.local_reduction !5
  %k0.sum.anchor2306 = tail call float asm sideeffect "", "=v,0"(float %local.reduce137) #8, !dbg !53, !frisk.cross_bn_sum_anchor !5
  %866 = insertelement <4 x float> %863, float %865, i64 3, !dbg !52
  %math.result462 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %866) #8, !dbg !52, !frisk.register_lifetimes !5
  %867 = fptrunc <4 x float> %math.result462 to <4 x half>, !dbg !52
  %math.packed463 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %867) #8, !dbg !53, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49, !frisk.softmax_fragments !5
  %868 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed463, <4 x half> %836, <4 x float> %834) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %869 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed463, <4 x half> %852, <4 x float> %835) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %870 = shufflevector <8 x half> %718, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %871 = fsub <4 x float> %570, %689, !dbg !44
  %872 = extractelement <4 x float> %871, i64 0, !dbg !51
  %873 = tail call float @llvm.amdgcn.exp2.f32(float %872), !dbg !51
  %local.reduce169 = fadd float %873, %k0.sum.anchor2299, !dbg !53, !frisk.local_reduction !5
  %k0.sum.anchor2307 = tail call float asm sideeffect "", "=v,0"(float %local.reduce169) #8, !dbg !53, !frisk.cross_bn_sum_anchor !5
  %874 = insertelement <4 x float> poison, float %873, i64 0, !dbg !52
  %875 = extractelement <4 x float> %871, i64 1, !dbg !51
  %876 = tail call float @llvm.amdgcn.exp2.f32(float %875), !dbg !51
  %local.reduce170 = fadd float %876, %k0.sum.anchor2300, !dbg !53, !frisk.local_reduction !5
  %k0.sum.anchor2308 = tail call float asm sideeffect "", "=v,0"(float %local.reduce170) #8, !dbg !53, !frisk.cross_bn_sum_anchor !5
  %877 = insertelement <4 x float> %874, float %876, i64 1, !dbg !52
  %878 = extractelement <4 x float> %871, i64 2, !dbg !51
  %879 = tail call float @llvm.amdgcn.exp2.f32(float %878), !dbg !51
  %local.reduce171 = fadd float %879, %k0.sum.anchor2301, !dbg !53, !frisk.local_reduction !5
  %k0.sum.anchor2309 = tail call float asm sideeffect "", "=v,0"(float %local.reduce171) #8, !dbg !53, !frisk.cross_bn_sum_anchor !5
  %880 = insertelement <4 x float> %877, float %879, i64 2, !dbg !52
  %881 = extractelement <4 x float> %871, i64 3, !dbg !51
  %882 = tail call float @llvm.amdgcn.exp2.f32(float %881), !dbg !51
  %local.reduce172 = fadd float %882, %k0.sum.anchor2302, !dbg !53, !frisk.local_reduction !5
  %k0.sum.anchor2310 = tail call float asm sideeffect "", "=v,0"(float %local.reduce172) #8, !dbg !53, !frisk.cross_bn_sum_anchor !5
  %883 = insertelement <4 x float> %880, float %882, i64 3, !dbg !52
  %math.result448 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %883) #8, !dbg !52, !frisk.register_lifetimes !5
  %884 = fptrunc <4 x float> %math.result448 to <4 x half>, !dbg !52
  %math.packed449 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %884) #8, !dbg !53, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49, !frisk.softmax_fragments !5
  %885 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed449, <4 x half> %870, <4 x float> %851) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %886 = shufflevector <8 x half> %718, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %887 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed449, <4 x half> %886, <4 x float> %853) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %888 = fsub <4 x float> %612, %690, !dbg !44
  %889 = extractelement <4 x float> %888, i64 0, !dbg !51
  %890 = tail call float @llvm.amdgcn.exp2.f32(float %889), !dbg !51
  %local.reduce138 = fadd float %890, %k0.sum.anchor2303, !dbg !53, !frisk.local_reduction !5
  %k0.sum.anchor2311 = tail call float asm sideeffect "", "=v,0"(float %local.reduce138) #8, !dbg !53, !frisk.cross_bn_sum_anchor !5
  %891 = insertelement <4 x float> poison, float %890, i64 0, !dbg !52
  %892 = extractelement <4 x float> %888, i64 1, !dbg !51
  %893 = tail call float @llvm.amdgcn.exp2.f32(float %892), !dbg !51
  %local.reduce139 = fadd float %893, %k0.sum.anchor2304, !dbg !53, !frisk.local_reduction !5
  %k0.sum.anchor2312 = tail call float asm sideeffect "", "=v,0"(float %local.reduce139) #8, !dbg !53, !frisk.cross_bn_sum_anchor !5
  %894 = insertelement <4 x float> %891, float %893, i64 1, !dbg !52
  %895 = extractelement <4 x float> %888, i64 2, !dbg !51
  %896 = tail call float @llvm.amdgcn.exp2.f32(float %895), !dbg !51
  %local.reduce140 = fadd float %896, %k0.sum.anchor2305, !dbg !53, !frisk.local_reduction !5
  %k0.sum.anchor2313 = tail call float asm sideeffect "", "=v,0"(float %local.reduce140) #8, !dbg !53, !frisk.cross_bn_sum_anchor !5
  %897 = insertelement <4 x float> %894, float %896, i64 2, !dbg !52
  %898 = extractelement <4 x float> %888, i64 3, !dbg !51
  %899 = tail call float @llvm.amdgcn.exp2.f32(float %898), !dbg !51
  %local.reduce141 = fadd float %899, %k0.sum.anchor2306, !dbg !53, !frisk.local_reduction !5
  %k0.sum.anchor2314 = tail call float asm sideeffect "", "=v,0"(float %local.reduce141) #8, !dbg !53, !frisk.cross_bn_sum_anchor !5
  %900 = insertelement <4 x float> %897, float %899, i64 3, !dbg !52
  %math.result464 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %900) #8, !dbg !52, !frisk.register_lifetimes !5
  %901 = fptrunc <4 x float> %math.result464 to <4 x half>, !dbg !52
  %math.packed465 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %901) #8, !dbg !53, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49, !frisk.softmax_fragments !5
  %902 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed465, <4 x half> %870, <4 x float> %868) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %903 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed465, <4 x half> %886, <4 x float> %869) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %904 = shufflevector <8 x half> %720, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %905 = fsub <4 x float> %578, %689, !dbg !44
  %906 = extractelement <4 x float> %905, i64 0, !dbg !51
  %907 = tail call float @llvm.amdgcn.exp2.f32(float %906), !dbg !51
  %local.reduce173 = fadd float %907, %k0.sum.anchor2307, !dbg !53, !frisk.local_reduction !5
  %k0.sum.anchor2315 = tail call float asm sideeffect "", "=v,0"(float %local.reduce173) #8, !dbg !53, !frisk.cross_bn_sum_anchor !5
  %908 = insertelement <4 x float> poison, float %907, i64 0, !dbg !52
  %909 = extractelement <4 x float> %905, i64 1, !dbg !51
  %910 = tail call float @llvm.amdgcn.exp2.f32(float %909), !dbg !51
  %local.reduce174 = fadd float %910, %k0.sum.anchor2308, !dbg !53, !frisk.local_reduction !5
  %k0.sum.anchor2316 = tail call float asm sideeffect "", "=v,0"(float %local.reduce174) #8, !dbg !53, !frisk.cross_bn_sum_anchor !5
  %911 = insertelement <4 x float> %908, float %910, i64 1, !dbg !52
  %912 = extractelement <4 x float> %905, i64 2, !dbg !51
  %913 = tail call float @llvm.amdgcn.exp2.f32(float %912), !dbg !51
  %local.reduce175 = fadd float %913, %k0.sum.anchor2309, !dbg !53, !frisk.local_reduction !5
  %k0.sum.anchor2317 = tail call float asm sideeffect "", "=v,0"(float %local.reduce175) #8, !dbg !53, !frisk.cross_bn_sum_anchor !5
  %914 = insertelement <4 x float> %911, float %913, i64 2, !dbg !52
  %915 = extractelement <4 x float> %905, i64 3, !dbg !51
  %916 = tail call float @llvm.amdgcn.exp2.f32(float %915), !dbg !51
  %local.reduce176 = fadd float %916, %k0.sum.anchor2310, !dbg !53, !frisk.local_reduction !5
  %k0.sum.anchor2318 = tail call float asm sideeffect "", "=v,0"(float %local.reduce176) #8, !dbg !53, !frisk.cross_bn_sum_anchor !5
  %917 = insertelement <4 x float> %914, float %916, i64 3, !dbg !52
  %math.result450 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %917) #8, !dbg !52, !frisk.register_lifetimes !5
  %918 = fptrunc <4 x float> %math.result450 to <4 x half>, !dbg !52
  %math.packed451 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %918) #8, !dbg !53, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49, !frisk.softmax_fragments !5
  %919 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed451, <4 x half> %904, <4 x float> %885) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %920 = shufflevector <8 x half> %720, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %921 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed451, <4 x half> %920, <4 x float> %887) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %922 = fsub <4 x float> %615, %690, !dbg !44
  %923 = extractelement <4 x float> %922, i64 0, !dbg !51
  %924 = tail call float @llvm.amdgcn.exp2.f32(float %923), !dbg !51
  %local.reduce142 = fadd float %924, %k0.sum.anchor2311, !dbg !53, !frisk.local_reduction !5
  %k0.sum.anchor2319 = tail call float asm sideeffect "", "=v,0"(float %local.reduce142) #8, !dbg !53, !frisk.cross_bn_sum_anchor !5
  %925 = insertelement <4 x float> poison, float %924, i64 0, !dbg !52
  %926 = extractelement <4 x float> %922, i64 1, !dbg !51
  %927 = tail call float @llvm.amdgcn.exp2.f32(float %926), !dbg !51
  %local.reduce143 = fadd float %927, %k0.sum.anchor2312, !dbg !53, !frisk.local_reduction !5
  %k0.sum.anchor2320 = tail call float asm sideeffect "", "=v,0"(float %local.reduce143) #8, !dbg !53, !frisk.cross_bn_sum_anchor !5
  %928 = insertelement <4 x float> %925, float %927, i64 1, !dbg !52
  %929 = extractelement <4 x float> %922, i64 2, !dbg !51
  %930 = tail call float @llvm.amdgcn.exp2.f32(float %929), !dbg !51
  %local.reduce144 = fadd float %930, %k0.sum.anchor2313, !dbg !53, !frisk.local_reduction !5
  %k0.sum.anchor2321 = tail call float asm sideeffect "", "=v,0"(float %local.reduce144) #8, !dbg !53, !frisk.cross_bn_sum_anchor !5
  %931 = insertelement <4 x float> %928, float %930, i64 2, !dbg !52
  %932 = extractelement <4 x float> %922, i64 3, !dbg !51
  %933 = tail call float @llvm.amdgcn.exp2.f32(float %932), !dbg !51
  %local.reduce145 = fadd float %933, %k0.sum.anchor2314, !dbg !53, !frisk.local_reduction !5
  %k0.sum.anchor2322 = tail call float asm sideeffect "", "=v,0"(float %local.reduce145) #8, !dbg !53, !frisk.cross_bn_sum_anchor !5
  %934 = insertelement <4 x float> %931, float %933, i64 3, !dbg !52
  %math.result466 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %934) #8, !dbg !52, !frisk.register_lifetimes !5
  %935 = fptrunc <4 x float> %math.result466 to <4 x half>, !dbg !52
  %math.packed467 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %935) #8, !dbg !53, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49, !frisk.softmax_fragments !5
  %936 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed467, <4 x half> %904, <4 x float> %902) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %937 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed467, <4 x half> %920, <4 x float> %903) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %938 = shufflevector <8 x half> %722, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %939 = fsub <4 x float> %585, %689, !dbg !44
  %940 = extractelement <4 x float> %939, i64 0, !dbg !51
  %941 = tail call float @llvm.amdgcn.exp2.f32(float %940), !dbg !51
  %local.reduce177 = fadd float %941, %k0.sum.anchor2315, !dbg !53, !frisk.local_reduction !5
  %k0.sum.anchor2323 = tail call float asm sideeffect "", "=v,0"(float %local.reduce177) #8, !dbg !53, !frisk.cross_bn_sum_anchor !5
  %942 = insertelement <4 x float> poison, float %941, i64 0, !dbg !52
  %943 = extractelement <4 x float> %939, i64 1, !dbg !51
  %944 = tail call float @llvm.amdgcn.exp2.f32(float %943), !dbg !51
  %local.reduce178 = fadd float %944, %k0.sum.anchor2316, !dbg !53, !frisk.local_reduction !5
  %k0.sum.anchor2324 = tail call float asm sideeffect "", "=v,0"(float %local.reduce178) #8, !dbg !53, !frisk.cross_bn_sum_anchor !5
  %945 = insertelement <4 x float> %942, float %944, i64 1, !dbg !52
  %946 = extractelement <4 x float> %939, i64 2, !dbg !51
  %947 = tail call float @llvm.amdgcn.exp2.f32(float %946), !dbg !51
  %local.reduce179 = fadd float %947, %k0.sum.anchor2317, !dbg !53, !frisk.local_reduction !5
  %k0.sum.anchor2325 = tail call float asm sideeffect "", "=v,0"(float %local.reduce179) #8, !dbg !53, !frisk.cross_bn_sum_anchor !5
  %948 = insertelement <4 x float> %945, float %947, i64 2, !dbg !52
  %949 = extractelement <4 x float> %939, i64 3, !dbg !51
  %950 = tail call float @llvm.amdgcn.exp2.f32(float %949), !dbg !51
  %local.reduce180 = fadd float %950, %k0.sum.anchor2318, !dbg !53, !frisk.local_reduction !5
  %k0.sum.anchor2326 = tail call float asm sideeffect "", "=v,0"(float %local.reduce180) #8, !dbg !53, !frisk.cross_bn_sum_anchor !5
  %951 = insertelement <4 x float> %948, float %950, i64 3, !dbg !52
  %math.result452 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %951) #8, !dbg !52, !frisk.register_lifetimes !5
  %952 = fptrunc <4 x float> %math.result452 to <4 x half>, !dbg !52
  %math.packed453 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %952) #8, !dbg !53, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49, !frisk.softmax_fragments !5
  %953 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed453, <4 x half> %938, <4 x float> %919) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %954 = shufflevector <8 x half> %722, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %955 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed453, <4 x half> %954, <4 x float> %921) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %956 = fsub <4 x float> %618, %690, !dbg !44
  %957 = extractelement <4 x float> %956, i64 0, !dbg !51
  %958 = tail call float @llvm.amdgcn.exp2.f32(float %957), !dbg !51
  %local.reduce146 = fadd float %958, %k0.sum.anchor2319, !dbg !53, !frisk.local_reduction !5
  %k0.sum.anchor2327 = tail call float asm sideeffect "", "=v,0"(float %local.reduce146) #8, !dbg !53, !frisk.cross_bn_sum_anchor !5
  %959 = insertelement <4 x float> poison, float %958, i64 0, !dbg !52
  %960 = extractelement <4 x float> %956, i64 1, !dbg !51
  %961 = tail call float @llvm.amdgcn.exp2.f32(float %960), !dbg !51
  %local.reduce147 = fadd float %961, %k0.sum.anchor2320, !dbg !53, !frisk.local_reduction !5
  %k0.sum.anchor2328 = tail call float asm sideeffect "", "=v,0"(float %local.reduce147) #8, !dbg !53, !frisk.cross_bn_sum_anchor !5
  %962 = insertelement <4 x float> %959, float %961, i64 1, !dbg !52
  %963 = extractelement <4 x float> %956, i64 2, !dbg !51
  %964 = tail call float @llvm.amdgcn.exp2.f32(float %963), !dbg !51
  %local.reduce148 = fadd float %964, %k0.sum.anchor2321, !dbg !53, !frisk.local_reduction !5
  %k0.sum.anchor2329 = tail call float asm sideeffect "", "=v,0"(float %local.reduce148) #8, !dbg !53, !frisk.cross_bn_sum_anchor !5
  %965 = insertelement <4 x float> %962, float %964, i64 2, !dbg !52
  %966 = extractelement <4 x float> %956, i64 3, !dbg !51
  %967 = tail call float @llvm.amdgcn.exp2.f32(float %966), !dbg !51
  %local.reduce149 = fadd float %967, %k0.sum.anchor2322, !dbg !53, !frisk.local_reduction !5
  %k0.sum.anchor2330 = tail call float asm sideeffect "", "=v,0"(float %local.reduce149) #8, !dbg !53, !frisk.cross_bn_sum_anchor !5
  %968 = insertelement <4 x float> %965, float %967, i64 3, !dbg !52
  %math.result468 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %968) #8, !dbg !52, !frisk.register_lifetimes !5
  %969 = fptrunc <4 x float> %math.result468 to <4 x half>, !dbg !52
  %math.packed469 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %969) #8, !dbg !53, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49, !frisk.softmax_fragments !5
  %970 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed469, <4 x half> %938, <4 x float> %936) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %971 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed469, <4 x half> %954, <4 x float> %937) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %972 = bitcast <4 x i32> %723 to <8 x half>, !dbg !49
  %973 = shufflevector <8 x half> %972, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %974 = fsub <4 x float> %592, %689, !dbg !44
  %975 = extractelement <4 x float> %974, i64 0, !dbg !51
  %976 = tail call float @llvm.amdgcn.exp2.f32(float %975), !dbg !51
  %local.reduce181 = fadd float %976, %k0.sum.anchor2323, !dbg !53, !frisk.local_reduction !5
  %k0.sum.anchor2331 = tail call float asm sideeffect "", "=v,0"(float %local.reduce181) #8, !dbg !53, !frisk.cross_bn_sum_anchor !5
  %977 = insertelement <4 x float> poison, float %976, i64 0, !dbg !52
  %978 = extractelement <4 x float> %974, i64 1, !dbg !51
  %979 = tail call float @llvm.amdgcn.exp2.f32(float %978), !dbg !51
  %local.reduce182 = fadd float %979, %k0.sum.anchor2324, !dbg !53, !frisk.local_reduction !5
  %k0.sum.anchor2332 = tail call float asm sideeffect "", "=v,0"(float %local.reduce182) #8, !dbg !53, !frisk.cross_bn_sum_anchor !5
  %local.reduce185 = fadd float %k0.sum.anchor2331, %k0.sum.anchor2332, !dbg !53, !frisk.local_reduction !5
  %k0.sum.anchor2333 = tail call float asm sideeffect "", "=v,0"(float %local.reduce185) #8, !dbg !53, !frisk.cross_bn_sum_anchor !5
  %980 = insertelement <4 x float> %977, float %979, i64 1, !dbg !52
  %981 = extractelement <4 x float> %974, i64 2, !dbg !51
  %982 = tail call float @llvm.amdgcn.exp2.f32(float %981), !dbg !51
  %local.reduce183 = fadd float %982, %k0.sum.anchor2325, !dbg !53, !frisk.local_reduction !5
  %k0.sum.anchor2334 = tail call float asm sideeffect "", "=v,0"(float %local.reduce183) #8, !dbg !53, !frisk.cross_bn_sum_anchor !5
  %983 = insertelement <4 x float> %980, float %982, i64 2, !dbg !52
  %984 = extractelement <4 x float> %974, i64 3, !dbg !51
  %985 = tail call float @llvm.amdgcn.exp2.f32(float %984), !dbg !51
  %local.reduce184 = fadd float %985, %k0.sum.anchor2326, !dbg !53, !frisk.local_reduction !5
  %k0.sum.anchor2335 = tail call float asm sideeffect "", "=v,0"(float %local.reduce184) #8, !dbg !53, !frisk.cross_bn_sum_anchor !5
  %local.reduce186 = fadd float %k0.sum.anchor2334, %k0.sum.anchor2335, !dbg !53, !frisk.local_reduction !5
  %k0.sum.anchor2336 = tail call float asm sideeffect "", "=v,0"(float %local.reduce186) #8, !dbg !53, !frisk.cross_bn_sum_anchor !5
  %986 = fadd float %k0.sum.anchor2333, %k0.sum.anchor2336, !dbg !53, !frisk.local_reduction !5
  %k0.sum.anchor2337 = tail call float asm sideeffect "", "=v,0"(float %986) #8, !dbg !53, !frisk.cross_bn_sum_anchor !5
  %987 = fadd float %698, %k0.sum.anchor2337, !dbg !53
  %k0.sum.anchor2338 = tail call float asm sideeffect "", "=v,0"(float %987) #8, !dbg !53, !frisk.cross_bn_sum_anchor !5
  %988 = insertelement <1 x float> poison, float %k0.sum.anchor2338, i64 0, !dbg !11
  %989 = insertvalue [2 x <1 x float>] poison, <1 x float> %988, 0, !dbg !11
  %990 = insertelement <4 x float> %983, float %985, i64 3, !dbg !52
  %math.result454 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %990) #8, !dbg !52, !frisk.register_lifetimes !5
  %991 = fptrunc <4 x float> %math.result454 to <4 x half>, !dbg !52
  %math.packed455 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %991) #8, !dbg !53, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49, !frisk.softmax_fragments !5
  %992 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed455, <4 x half> %973, <4 x float> %953) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %993 = shufflevector <8 x half> %972, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %994 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed455, <4 x half> %993, <4 x float> %955) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %995 = fsub <4 x float> %632, %690, !dbg !44
  %996 = extractelement <4 x float> %995, i64 0, !dbg !51
  %997 = tail call float @llvm.amdgcn.exp2.f32(float %996), !dbg !51
  %local.reduce150 = fadd float %997, %k0.sum.anchor2327, !dbg !53, !frisk.local_reduction !5
  %k0.sum.anchor2339 = tail call float asm sideeffect "", "=v,0"(float %local.reduce150) #8, !dbg !53, !frisk.cross_bn_sum_anchor !5
  %998 = insertelement <4 x float> poison, float %997, i64 0, !dbg !52
  %999 = extractelement <4 x float> %995, i64 1, !dbg !51
  %1000 = tail call float @llvm.amdgcn.exp2.f32(float %999), !dbg !51
  %local.reduce151 = fadd float %1000, %k0.sum.anchor2328, !dbg !53, !frisk.local_reduction !5
  %k0.sum.anchor2340 = tail call float asm sideeffect "", "=v,0"(float %local.reduce151) #8, !dbg !53, !frisk.cross_bn_sum_anchor !5
  %local.reduce154 = fadd float %k0.sum.anchor2339, %k0.sum.anchor2340, !dbg !53, !frisk.local_reduction !5
  %k0.sum.anchor2341 = tail call float asm sideeffect "", "=v,0"(float %local.reduce154) #8, !dbg !53, !frisk.cross_bn_sum_anchor !5
  %1001 = insertelement <4 x float> %998, float %1000, i64 1, !dbg !52
  %1002 = extractelement <4 x float> %995, i64 2, !dbg !51
  %1003 = tail call float @llvm.amdgcn.exp2.f32(float %1002), !dbg !51
  %local.reduce152 = fadd float %1003, %k0.sum.anchor2329, !dbg !53, !frisk.local_reduction !5
  %k0.sum.anchor2342 = tail call float asm sideeffect "", "=v,0"(float %local.reduce152) #8, !dbg !53, !frisk.cross_bn_sum_anchor !5
  %1004 = insertelement <4 x float> %1001, float %1003, i64 2, !dbg !52
  %1005 = extractelement <4 x float> %995, i64 3, !dbg !51
  %1006 = tail call float @llvm.amdgcn.exp2.f32(float %1005), !dbg !51
  %local.reduce153 = fadd float %1006, %k0.sum.anchor2330, !dbg !53, !frisk.local_reduction !5
  %k0.sum.anchor2343 = tail call float asm sideeffect "", "=v,0"(float %local.reduce153) #8, !dbg !53, !frisk.cross_bn_sum_anchor !5
  %local.reduce155 = fadd float %k0.sum.anchor2342, %k0.sum.anchor2343, !dbg !53, !frisk.local_reduction !5
  %k0.sum.anchor2344 = tail call float asm sideeffect "", "=v,0"(float %local.reduce155) #8, !dbg !53, !frisk.cross_bn_sum_anchor !5
  %1007 = fadd float %k0.sum.anchor2341, %k0.sum.anchor2344, !dbg !53, !frisk.local_reduction !5
  %k0.sum.anchor2345 = tail call float asm sideeffect "", "=v,0"(float %1007) #8, !dbg !53, !frisk.cross_bn_sum_anchor !5
  %1008 = fadd float %706, %k0.sum.anchor2345, !dbg !53
  %k0.sum.anchor2346 = tail call float asm sideeffect "", "=v,0"(float %1008) #8, !dbg !53, !frisk.cross_bn_sum_anchor !5
  %1009 = insertelement <1 x float> poison, float %k0.sum.anchor2346, i64 0, !dbg !11
  %1010 = insertvalue [2 x <1 x float>] %989, <1 x float> %1009, 1, !dbg !11
  %1011 = insertelement <4 x float> %1004, float %1006, i64 3, !dbg !52
  %math.result470 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1011) #8, !dbg !52, !frisk.register_lifetimes !5
  %1012 = fptrunc <4 x float> %math.result470 to <4 x half>, !dbg !52
  %math.packed471 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1012) #8, !dbg !53, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49, !frisk.softmax_fragments !5
  %1013 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed471, <4 x half> %973, <4 x float> %970) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1014 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed471, <4 x half> %993, <4 x float> %971) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !40
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !40
  fence syncscope("workgroup") release, !dbg !40
  tail call void @llvm.amdgcn.s.barrier(), !dbg !40
  fence syncscope("workgroup") acquire, !dbg !40
  %vdb.address1830 = or disjoint i64 %.idx17771779, 64, !dbg !40
  %vdb.address1831 = or disjoint i64 %address.local989, %vdb.address1830, !dbg !40
  %vdb.address1832 = trunc i64 %vdb.address1831 to i32, !dbg !40
  %vdb.address1833 = shl i32 %vdb.address1832, 1, !dbg !40
  %vdb.address1841 = or disjoint i64 %address.local995, %vdb.address1830, !dbg !40
  %vdb.address1842 = trunc i64 %vdb.address1841 to i32, !dbg !40
  %vdb.address1843 = shl i32 %vdb.address1842, 1, !dbg !40
  %1015 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1702, i32 %vdb.address1833, <4 x i32> %dma.buffer.rsrc1677) #9, !dbg !40
  %1016 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1699, i32 %vdb.address1843, <4 x i32> %dma.buffer.rsrc1677) #9, !dbg !40
  %vdb.read = getelementptr i8, ptr addrspace(3) %address.local1003, i64 8192, !dbg !49
  %1017 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #11, !dbg !49, !frisk.v_double_buffer !50
  %1018 = bitcast <4 x i32> %1017 to <8 x half>, !dbg !49
  %vdb.read1821 = getelementptr i8, ptr addrspace(3) %address.local1012, i64 8448, !dbg !49
  %1019 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1821, i32 0) #11, !dbg !49, !frisk.v_double_buffer !50
  %1020 = bitcast <4 x i32> %1019 to <8 x half>, !dbg !49
  %vdb.read1822 = getelementptr i8, ptr addrspace(3) %address.local1012, i64 8704, !dbg !49
  %1021 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1822, i32 0) #11, !dbg !49, !frisk.v_double_buffer !50
  %1022 = bitcast <4 x i32> %1021 to <8 x half>, !dbg !49
  %vdb.read1823 = getelementptr i8, ptr addrspace(3) %address.local1012, i64 8960, !dbg !49
  %1023 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1823, i32 0) #11, !dbg !49, !frisk.v_double_buffer !50
  %1024 = bitcast <4 x i32> %1023 to <8 x half>, !dbg !49
  %vdb.read1824 = getelementptr i8, ptr addrspace(3) %address.local1003, i64 12288, !dbg !49
  %1025 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1824, i32 0) #11, !dbg !49, !frisk.v_double_buffer !50
  %1026 = bitcast <4 x i32> %1025 to <8 x half>, !dbg !49
  %vdb.read1825 = getelementptr i8, ptr addrspace(3) %address.local1012, i64 12544, !dbg !49
  %1027 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1825, i32 0) #11, !dbg !49, !frisk.v_double_buffer !50
  %1028 = bitcast <4 x i32> %1027 to <8 x half>, !dbg !49
  %vdb.read1826 = getelementptr i8, ptr addrspace(3) %address.local1012, i64 12800, !dbg !49
  %1029 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1826, i32 0) #11, !dbg !49, !frisk.v_double_buffer !50
  %1030 = bitcast <4 x i32> %1029 to <8 x half>, !dbg !49
  %vdb.read1827 = getelementptr i8, ptr addrspace(3) %address.local1012, i64 13056, !dbg !49
  %1031 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1827, i32 0) #11, !dbg !49, !frisk.v_double_buffer !50
  %1032 = shufflevector <8 x half> %1018, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  %1033 = extractvalue [2 x <8 x float>] %126, 0, !dbg !47
  %1034 = shufflevector <8 x float> %1033, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !47
  %1035 = fmul <4 x float> %1034, %695, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1036 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1032, <4 x float> %1035) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1037 = shufflevector <8 x half> %1018, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  %1038 = shufflevector <8 x float> %1033, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !47
  %1039 = fmul <4 x float> %1038, %695, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1040 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1037, <4 x float> %1039) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1041 = extractvalue [2 x <8 x float>] %126, 1, !dbg !47
  %1042 = shufflevector <8 x float> %1041, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !47
  %1043 = fmul <4 x float> %1042, %703, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1044 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed457, <4 x half> %1032, <4 x float> %1043) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1045 = shufflevector <8 x float> %1041, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !47
  %1046 = fmul <4 x float> %1045, %703, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1047 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed457, <4 x half> %1037, <4 x float> %1046) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1048 = shufflevector <8 x half> %1020, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1049 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed443, <4 x half> %1048, <4 x float> %1036) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1050 = shufflevector <8 x half> %1020, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1051 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed443, <4 x half> %1050, <4 x float> %1040) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1052 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed459, <4 x half> %1048, <4 x float> %1044) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1053 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed459, <4 x half> %1050, <4 x float> %1047) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1054 = shufflevector <8 x half> %1022, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1055 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed445, <4 x half> %1054, <4 x float> %1049) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1056 = shufflevector <8 x half> %1022, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1057 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed445, <4 x half> %1056, <4 x float> %1051) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1058 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed461, <4 x half> %1054, <4 x float> %1052) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1059 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed461, <4 x half> %1056, <4 x float> %1053) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1060 = shufflevector <8 x half> %1024, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1061 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed447, <4 x half> %1060, <4 x float> %1055) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1062 = shufflevector <8 x half> %1024, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1063 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed447, <4 x half> %1062, <4 x float> %1057) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1064 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed463, <4 x half> %1060, <4 x float> %1058) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1065 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed463, <4 x half> %1062, <4 x float> %1059) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1066 = shufflevector <8 x half> %1026, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1067 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed449, <4 x half> %1066, <4 x float> %1061) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1068 = shufflevector <8 x half> %1026, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1069 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed449, <4 x half> %1068, <4 x float> %1063) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1070 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed465, <4 x half> %1066, <4 x float> %1064) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1071 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed465, <4 x half> %1068, <4 x float> %1065) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1072 = shufflevector <8 x half> %1028, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1073 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed451, <4 x half> %1072, <4 x float> %1067) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1074 = shufflevector <8 x half> %1028, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1075 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed451, <4 x half> %1074, <4 x float> %1069) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1076 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed467, <4 x half> %1072, <4 x float> %1070) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1077 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed467, <4 x half> %1074, <4 x float> %1071) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1078 = shufflevector <8 x half> %1030, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1079 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed453, <4 x half> %1078, <4 x float> %1073) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1080 = shufflevector <8 x half> %1030, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1081 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed453, <4 x half> %1080, <4 x float> %1075) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1082 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed469, <4 x half> %1078, <4 x float> %1076) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1083 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed469, <4 x half> %1080, <4 x float> %1077) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1084 = bitcast <4 x i32> %1031 to <8 x half>, !dbg !49
  %1085 = shufflevector <8 x half> %1084, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1086 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed455, <4 x half> %1085, <4 x float> %1079) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1087 = shufflevector <8 x half> %1084, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1088 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed455, <4 x half> %1087, <4 x float> %1081) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1089 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed471, <4 x half> %1085, <4 x float> %1082) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1090 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed471, <4 x half> %1087, <4 x float> %1083) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !40
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !40
  fence syncscope("workgroup") release, !dbg !40
  tail call void @llvm.amdgcn.s.barrier(), !dbg !40
  fence syncscope("workgroup") acquire, !dbg !40
  %vdb.address1850 = or disjoint i64 %.idx17771779, 96, !dbg !40
  %vdb.address1851 = or disjoint i64 %address.local989, %vdb.address1850, !dbg !40
  %vdb.address1852 = trunc i64 %vdb.address1851 to i32, !dbg !40
  %vdb.address1853 = shl i32 %vdb.address1852, 1, !dbg !40
  %vdb.address1861 = or disjoint i64 %address.local995, %vdb.address1850, !dbg !40
  %vdb.address1862 = trunc i64 %vdb.address1861 to i32, !dbg !40
  %vdb.address1863 = shl i32 %vdb.address1862, 1, !dbg !40
  %1091 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1853, <4 x i32> %dma.buffer.rsrc1677) #9, !dbg !40
  %1092 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1820, i32 %vdb.address1863, <4 x i32> %dma.buffer.rsrc1677) #9, !dbg !40
  %1093 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1003, i32 0) #11, !dbg !49, !frisk.v_double_buffer !50
  %1094 = bitcast <4 x i32> %1093 to <8 x half>, !dbg !49
  %1095 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1013, i32 0) #11, !dbg !49, !frisk.v_double_buffer !50
  %1096 = bitcast <4 x i32> %1095 to <8 x half>, !dbg !49
  %1097 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1017, i32 0) #11, !dbg !49, !frisk.v_double_buffer !50
  %1098 = bitcast <4 x i32> %1097 to <8 x half>, !dbg !49
  %1099 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1021, i32 0) #11, !dbg !49, !frisk.v_double_buffer !50
  %1100 = bitcast <4 x i32> %1099 to <8 x half>, !dbg !49
  %1101 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1023, i32 0) #11, !dbg !49, !frisk.v_double_buffer !50
  %1102 = bitcast <4 x i32> %1101 to <8 x half>, !dbg !49
  %1103 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1027, i32 0) #11, !dbg !49, !frisk.v_double_buffer !50
  %1104 = bitcast <4 x i32> %1103 to <8 x half>, !dbg !49
  %1105 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1031, i32 0) #11, !dbg !49, !frisk.v_double_buffer !50
  %1106 = bitcast <4 x i32> %1105 to <8 x half>, !dbg !49
  %1107 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1035, i32 0) #11, !dbg !49, !frisk.v_double_buffer !50
  %1108 = shufflevector <8 x half> %1094, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  %1109 = extractvalue [2 x <8 x float>] %125, 0, !dbg !47
  %1110 = shufflevector <8 x float> %1109, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !47
  %1111 = fmul <4 x float> %1110, %695, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1112 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1108, <4 x float> %1111) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1113 = shufflevector <8 x half> %1094, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  %1114 = shufflevector <8 x float> %1109, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !47
  %1115 = fmul <4 x float> %1114, %695, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1116 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1113, <4 x float> %1115) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1117 = extractvalue [2 x <8 x float>] %125, 1, !dbg !47
  %1118 = shufflevector <8 x float> %1117, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !47
  %1119 = fmul <4 x float> %1118, %703, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1120 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed457, <4 x half> %1108, <4 x float> %1119) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1121 = shufflevector <8 x float> %1117, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !47
  %1122 = fmul <4 x float> %1121, %703, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1123 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed457, <4 x half> %1113, <4 x float> %1122) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1124 = shufflevector <8 x half> %1096, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1125 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed443, <4 x half> %1124, <4 x float> %1112) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1126 = shufflevector <8 x half> %1096, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1127 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed443, <4 x half> %1126, <4 x float> %1116) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1128 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed459, <4 x half> %1124, <4 x float> %1120) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1129 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed459, <4 x half> %1126, <4 x float> %1123) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1130 = shufflevector <8 x half> %1098, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1131 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed445, <4 x half> %1130, <4 x float> %1125) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1132 = shufflevector <8 x half> %1098, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1133 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed445, <4 x half> %1132, <4 x float> %1127) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1134 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed461, <4 x half> %1130, <4 x float> %1128) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1135 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed461, <4 x half> %1132, <4 x float> %1129) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1136 = shufflevector <8 x half> %1100, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1137 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed447, <4 x half> %1136, <4 x float> %1131) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1138 = shufflevector <8 x half> %1100, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1139 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed447, <4 x half> %1138, <4 x float> %1133) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1140 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed463, <4 x half> %1136, <4 x float> %1134) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1141 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed463, <4 x half> %1138, <4 x float> %1135) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1142 = shufflevector <8 x half> %1102, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1143 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed449, <4 x half> %1142, <4 x float> %1137) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1144 = shufflevector <8 x half> %1102, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1145 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed449, <4 x half> %1144, <4 x float> %1139) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1146 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed465, <4 x half> %1142, <4 x float> %1140) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1147 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed465, <4 x half> %1144, <4 x float> %1141) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1148 = shufflevector <8 x half> %1104, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1149 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed451, <4 x half> %1148, <4 x float> %1143) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1150 = shufflevector <8 x half> %1104, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1151 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed451, <4 x half> %1150, <4 x float> %1145) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1152 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed467, <4 x half> %1148, <4 x float> %1146) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1153 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed467, <4 x half> %1150, <4 x float> %1147) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1154 = shufflevector <8 x half> %1106, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1155 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed453, <4 x half> %1154, <4 x float> %1149) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1156 = shufflevector <8 x half> %1106, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1157 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed453, <4 x half> %1156, <4 x float> %1151) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1158 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed469, <4 x half> %1154, <4 x float> %1152) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1159 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed469, <4 x half> %1156, <4 x float> %1153) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1160 = bitcast <4 x i32> %1107 to <8 x half>, !dbg !49
  %1161 = shufflevector <8 x half> %1160, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1162 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed455, <4 x half> %1161, <4 x float> %1155) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1163 = shufflevector <8 x half> %1160, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1164 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed455, <4 x half> %1163, <4 x float> %1157) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1165 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed471, <4 x half> %1161, <4 x float> %1158) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1166 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed471, <4 x half> %1163, <4 x float> %1159) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !40
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !40
  fence syncscope("workgroup") release, !dbg !40
  tail call void @llvm.amdgcn.s.barrier(), !dbg !40
  fence syncscope("workgroup") acquire, !dbg !40
  %1167 = add nuw nsw i64 %130, 1, !dbg !11
  %k0.has.next.not.not = icmp samesign ult i64 %130, %16, !dbg !49
  br i1 %k0.has.next.not.not, label %k0.issue, label %k0.join, !dbg !49

k0.issue:                                         ; preds = %123
  %k0.next. = lshr i64 %1167, 12, !dbg !12
  %k0.next.2365 = add nuw nsw i64 %k0.next., %17, !dbg !12
  %k0.next.2366 = shl i64 %k0.next.2365, 19, !dbg !12
  %k0.next.2367 = and i64 %k0.next.2366, 16252928, !dbg !12
  %k0.next.2368 = lshr i64 %1167, 5, !dbg !12
  %k0.next.2369 = and i64 %k0.next.2368, 127, !dbg !12
  %k0.next..idx1736 = or disjoint i64 %k0.next.2367, %k0.next.2369, !dbg !12
  %k0.next.2370 = shl i64 %1167, 14, !dbg !12
  %k0.next.2371 = shl i64 %k0.next.2368, 19, !dbg !12
  %k0.next.2372 = sub i64 %k0.next.2370, %k0.next.2371, !dbg !12
  %k0.next.2373 = add i64 %k0.next..idx1736, %k0.next.2372, !dbg !12
  %k0.next.kdma.source1888 = add nuw nsw i64 %k0.next.2373, %k0.prime.kdma.source1887, !dbg !12
  %k0.next.kdma.source1889 = trunc i64 %k0.next.kdma.source1888 to i32, !dbg !12
  %k0.next.kdma.source1890 = shl i32 %k0.next.kdma.source1889, 1, !dbg !12
  %k0.next.kdma.offset = add i32 %k0.next.kdma.source1890, %kdma.intra, !dbg !12
  %k0.next.2374 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1702, i32 %k0.next.kdma.offset, <4 x i32> %k0.prime.dma.buffer.rsrc1701) #9, !dbg !12, !frisk.cross_bn_k0 !5
  %k0.next.kdma.source1902 = add i64 %k0.next.2373, %k0.prime.kdma.source1901, !dbg !12
  %k0.next.kdma.source1903 = trunc i64 %k0.next.kdma.source1902 to i32, !dbg !12
  %k0.next.kdma.source1904 = shl i32 %k0.next.kdma.source1903, 1, !dbg !12
  %k0.next.kdma.offset1905 = add i32 %k0.next.kdma.source1904, %kdma.intra.hi, !dbg !12
  %k0.next.2375 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1906, i32 %k0.next.kdma.offset1905, <4 x i32> %k0.prime.dma.buffer.rsrc1701) #9, !dbg !12, !frisk.cross_bn_k0 !5
  %k0.next.kdma.source1919 = add nuw nsw i64 %k0.next.2373, %k0.prime.kdma.source1918, !dbg !12
  %k0.next.kdma.source1920 = trunc i64 %k0.next.kdma.source1919 to i32, !dbg !12
  %k0.next.kdma.source1921 = shl i32 %k0.next.kdma.source1920, 1, !dbg !12
  %k0.next.kdma.offset1922 = add i32 %k0.next.kdma.source1921, %kdma.intra, !dbg !12
  %k0.next.2376 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1699, i32 %k0.next.kdma.offset1922, <4 x i32> %k0.prime.dma.buffer.rsrc1701) #9, !dbg !12, !frisk.cross_bn_k0 !5
  %k0.next.kdma.source1936 = add i64 %k0.next.2373, %k0.prime.kdma.source1935, !dbg !12
  %k0.next.kdma.source1937 = trunc i64 %k0.next.kdma.source1936 to i32, !dbg !12
  %k0.next.kdma.source1938 = shl i32 %k0.next.kdma.source1937, 1, !dbg !12
  %k0.next.kdma.offset1939 = add i32 %k0.next.kdma.source1938, %kdma.intra.hi, !dbg !12
  %k0.next.2377 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1940, i32 %k0.next.kdma.offset1939, <4 x i32> %k0.prime.dma.buffer.rsrc1701) #9, !dbg !12, !frisk.cross_bn_k0 !5
  br label %k0.join

k0.join:                                          ; preds = %k0.issue, %123
  %1168 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #11, !dbg !49, !frisk.v_double_buffer !50
  %1169 = bitcast <4 x i32> %1168 to <8 x half>, !dbg !49
  %1170 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1821, i32 0) #11, !dbg !49, !frisk.v_double_buffer !50
  %1171 = bitcast <4 x i32> %1170 to <8 x half>, !dbg !49
  %1172 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1822, i32 0) #11, !dbg !49, !frisk.v_double_buffer !50
  %1173 = bitcast <4 x i32> %1172 to <8 x half>, !dbg !49
  %1174 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1823, i32 0) #11, !dbg !49, !frisk.v_double_buffer !50
  %1175 = bitcast <4 x i32> %1174 to <8 x half>, !dbg !49
  %1176 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1824, i32 0) #11, !dbg !49, !frisk.v_double_buffer !50
  %1177 = bitcast <4 x i32> %1176 to <8 x half>, !dbg !49
  %1178 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1825, i32 0) #11, !dbg !49, !frisk.v_double_buffer !50
  %1179 = bitcast <4 x i32> %1178 to <8 x half>, !dbg !49
  %1180 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1826, i32 0) #11, !dbg !49, !frisk.v_double_buffer !50
  %1181 = bitcast <4 x i32> %1180 to <8 x half>, !dbg !49
  %1182 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1827, i32 0) #11, !dbg !49, !frisk.v_double_buffer !50
  %1183 = shufflevector <8 x half> %1169, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  %1184 = extractvalue [2 x <8 x float>] %124, 0, !dbg !47
  %1185 = shufflevector <8 x float> %1184, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !47
  %1186 = fmul <4 x float> %1185, %695, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1187 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1183, <4 x float> %1186) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1188 = shufflevector <8 x half> %1169, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  %1189 = shufflevector <8 x float> %1184, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !47
  %1190 = fmul <4 x float> %1189, %695, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1191 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1188, <4 x float> %1190) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1192 = extractvalue [2 x <8 x float>] %124, 1, !dbg !47
  %1193 = shufflevector <8 x float> %1192, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !47
  %1194 = fmul <4 x float> %1193, %703, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1195 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed457, <4 x half> %1183, <4 x float> %1194) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1196 = shufflevector <8 x float> %1192, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !47
  %1197 = fmul <4 x float> %1196, %703, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1198 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed457, <4 x half> %1188, <4 x float> %1197) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1199 = shufflevector <8 x half> %1171, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1200 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed443, <4 x half> %1199, <4 x float> %1187) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1201 = shufflevector <8 x half> %1171, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1202 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed443, <4 x half> %1201, <4 x float> %1191) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1203 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed459, <4 x half> %1199, <4 x float> %1195) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1204 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed459, <4 x half> %1201, <4 x float> %1198) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1205 = shufflevector <8 x half> %1173, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1206 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed445, <4 x half> %1205, <4 x float> %1200) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1207 = shufflevector <8 x half> %1173, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1208 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed445, <4 x half> %1207, <4 x float> %1202) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1209 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed461, <4 x half> %1205, <4 x float> %1203) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1210 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed461, <4 x half> %1207, <4 x float> %1204) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1211 = shufflevector <8 x half> %1175, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1212 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed447, <4 x half> %1211, <4 x float> %1206) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1213 = shufflevector <8 x half> %1175, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1214 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed447, <4 x half> %1213, <4 x float> %1208) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1215 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed463, <4 x half> %1211, <4 x float> %1209) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1216 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed463, <4 x half> %1213, <4 x float> %1210) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1217 = shufflevector <8 x half> %1177, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1218 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed449, <4 x half> %1217, <4 x float> %1212) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1219 = shufflevector <8 x half> %1177, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1220 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed449, <4 x half> %1219, <4 x float> %1214) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1221 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed465, <4 x half> %1217, <4 x float> %1215) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1222 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed465, <4 x half> %1219, <4 x float> %1216) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1223 = shufflevector <8 x half> %1179, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1224 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed451, <4 x half> %1223, <4 x float> %1218) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1225 = shufflevector <8 x half> %1179, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1226 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed451, <4 x half> %1225, <4 x float> %1220) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1227 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed467, <4 x half> %1223, <4 x float> %1221) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1228 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed467, <4 x half> %1225, <4 x float> %1222) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1229 = shufflevector <8 x half> %1181, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1230 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed453, <4 x half> %1229, <4 x float> %1224) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1231 = shufflevector <8 x half> %1181, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1232 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed453, <4 x half> %1231, <4 x float> %1226) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1233 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed469, <4 x half> %1229, <4 x float> %1227) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1234 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed469, <4 x half> %1231, <4 x float> %1228) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1235 = bitcast <4 x i32> %1182 to <8 x half>, !dbg !49
  %1236 = shufflevector <8 x half> %1235, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1237 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed455, <4 x half> %1236, <4 x float> %1230) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1238 = shufflevector <8 x half> %1235, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1239 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed455, <4 x half> %1238, <4 x float> %1232) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1240 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed471, <4 x half> %1236, <4 x float> %1233) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1241 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed471, <4 x half> %1238, <4 x float> %1234) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !11
  %1242 = shufflevector <4 x float> %992, <4 x float> %994, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1243 = insertvalue [2 x <8 x float>] poison, <8 x float> %1242, 0, !dbg !11
  %1244 = shufflevector <4 x float> %1013, <4 x float> %1014, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1245 = insertvalue [2 x <8 x float>] %1243, <8 x float> %1244, 1, !dbg !11
  %1246 = insertvalue [2 x <1 x float>] poison, <1 x float> %675, 0, !dbg !11
  %1247 = insertvalue [2 x <1 x float>] %1246, <1 x float> %688, 1, !dbg !11
  %1248 = shufflevector <4 x float> %1086, <4 x float> %1088, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1249 = insertvalue [2 x <8 x float>] poison, <8 x float> %1248, 0, !dbg !11
  %1250 = shufflevector <4 x float> %1089, <4 x float> %1090, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1251 = insertvalue [2 x <8 x float>] %1249, <8 x float> %1250, 1, !dbg !11
  %1252 = shufflevector <4 x float> %1162, <4 x float> %1164, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1253 = insertvalue [2 x <8 x float>] poison, <8 x float> %1252, 0, !dbg !11
  %1254 = shufflevector <4 x float> %1165, <4 x float> %1166, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1255 = insertvalue [2 x <8 x float>] %1253, <8 x float> %1254, 1, !dbg !11
  %1256 = shufflevector <4 x float> %1237, <4 x float> %1239, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1257 = insertvalue [2 x <8 x float>] poison, <8 x float> %1256, 0, !dbg !11
  %1258 = shufflevector <4 x float> %1240, <4 x float> %1241, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1259 = insertvalue [2 x <8 x float>] %1257, <8 x float> %1258, 1, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  tail call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  br i1 %k0.has.next.not.not, label %123, label %1260, !dbg !11

1260:                                             ; preds = %k0.join
  %1261 = shl nuw nsw i32 %654, 2, !dbg !11
  %1262 = xor i32 %1261, 64, !dbg !11
  %1263 = bitcast float %k0.sum.anchor2346 to i32, !dbg !11
  %lane.sum.shuffle256 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1262, i32 %1263), !dbg !11
  %1264 = bitcast i32 %lane.sum.shuffle256 to float, !dbg !11
  %lane.sum.final257 = fadd float %k0.sum.anchor2346, %1264, !dbg !11
  %1265 = xor i32 %1261, 128, !dbg !11
  %1266 = bitcast float %lane.sum.final257 to i32, !dbg !11
  %lane.sum.shuffle258 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1265, i32 %1266), !dbg !11
  %1267 = bitcast i32 %lane.sum.shuffle258 to float, !dbg !11
  %lane.sum.final259 = fadd float %lane.sum.final257, %1267, !dbg !11
  %1268 = insertelement <1 x float> poison, float %lane.sum.final259, i64 0, !dbg !11
  %1269 = bitcast float %k0.sum.anchor2338 to i32, !dbg !11
  %lane.sum.shuffle = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1262, i32 %1269), !dbg !11
  %1270 = bitcast i32 %lane.sum.shuffle to float, !dbg !11
  %lane.sum.final = fadd float %k0.sum.anchor2338, %1270, !dbg !11
  %1271 = bitcast float %lane.sum.final to i32, !dbg !11
  %lane.sum.shuffle250 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1265, i32 %1271), !dbg !11
  %1272 = bitcast i32 %lane.sum.shuffle250 to float, !dbg !11
  %lane.sum.final251 = fadd float %lane.sum.final, %1272, !dbg !11
  %1273 = insertelement <1 x float> poison, float %lane.sum.final251, i64 0, !dbg !11
  %1274 = and i32 %kdma.tid, 16, !dbg !11
  %.not36 = icmp eq i32 %1274, 0, !dbg !11
  %1275 = and i32 %kdma.tid, 32, !dbg !11
  %.not37 = icmp eq i32 %1275, 0, !dbg !11
  %1276 = extractelement <4 x float> %1237, i64 0, !dbg !11
  %1277 = extractelement <4 x float> %1237, i64 1, !dbg !11
  %1278 = extractelement <4 x float> %1237, i64 2, !dbg !11
  %1279 = extractelement <4 x float> %1237, i64 3, !dbg !11
  %1280 = select i1 %.not36, float %1277, float %1276, !dbg !11
  %1281 = bitcast float %1280 to i32, !dbg !11
  %1282 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %660, i32 %1281), !dbg !11
  %1283 = bitcast i32 %1282 to float, !dbg !11
  %1284 = select i1 %.not36, float %1276, float %1283, !dbg !11
  %1285 = select i1 %.not36, float %1283, float %1277, !dbg !11
  %1286 = select i1 %.not36, float %1279, float %1278, !dbg !11
  %1287 = bitcast float %1286 to i32, !dbg !11
  %1288 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %660, i32 %1287), !dbg !11
  %1289 = bitcast i32 %1288 to float, !dbg !11
  %1290 = select i1 %.not36, float %1278, float %1289, !dbg !11
  %1291 = select i1 %.not36, float %1289, float %1279, !dbg !11
  %1292 = select i1 %.not37, float %1290, float %1284, !dbg !11
  %1293 = bitcast float %1292 to i32, !dbg !11
  %1294 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %668, i32 %1293), !dbg !11
  %1295 = bitcast i32 %1294 to float, !dbg !11
  %1296 = select i1 %.not37, float %1291, float %1285, !dbg !11
  %1297 = bitcast float %1296 to i32, !dbg !11
  %1298 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %668, i32 %1297), !dbg !11
  %1299 = bitcast i32 %1298 to float, !dbg !11
  %1300 = insertelement <4 x i1> poison, i1 %.not37, i64 0, !dbg !11
  %1301 = shufflevector <4 x i1> %1300, <4 x i1> poison, <4 x i32> zeroinitializer, !dbg !11
  %1302 = insertelement <4 x float> poison, float %1284, i64 0, !dbg !11
  %1303 = insertelement <4 x float> %1302, float %1285, i64 1, !dbg !11
  %1304 = insertelement <4 x float> %1303, float %1295, i64 2, !dbg !11
  %1305 = insertelement <4 x float> %1304, float %1299, i64 3, !dbg !11
  %1306 = insertelement <4 x float> poison, float %1295, i64 0, !dbg !11
  %1307 = insertelement <4 x float> %1306, float %1299, i64 1, !dbg !11
  %1308 = insertelement <4 x float> %1307, float %1290, i64 2, !dbg !11
  %1309 = insertelement <4 x float> %1308, float %1291, i64 3, !dbg !11
  %1310 = select <4 x i1> %1301, <4 x float> %1305, <4 x float> %1309, !dbg !11
  %1311 = extractelement <4 x float> %1162, i64 0, !dbg !11
  %1312 = extractelement <4 x float> %1162, i64 1, !dbg !11
  %1313 = extractelement <4 x float> %1162, i64 2, !dbg !11
  %1314 = extractelement <4 x float> %1162, i64 3, !dbg !11
  %1315 = select i1 %.not36, float %1312, float %1311, !dbg !11
  %1316 = bitcast float %1315 to i32, !dbg !11
  %1317 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %660, i32 %1316), !dbg !11
  %1318 = bitcast i32 %1317 to float, !dbg !11
  %1319 = select i1 %.not36, float %1311, float %1318, !dbg !11
  %1320 = select i1 %.not36, float %1318, float %1312, !dbg !11
  %1321 = select i1 %.not36, float %1314, float %1313, !dbg !11
  %1322 = bitcast float %1321 to i32, !dbg !11
  %1323 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %660, i32 %1322), !dbg !11
  %1324 = bitcast i32 %1323 to float, !dbg !11
  %1325 = select i1 %.not36, float %1313, float %1324, !dbg !11
  %1326 = select i1 %.not36, float %1324, float %1314, !dbg !11
  %1327 = select i1 %.not37, float %1325, float %1319, !dbg !11
  %1328 = bitcast float %1327 to i32, !dbg !11
  %1329 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %668, i32 %1328), !dbg !11
  %1330 = bitcast i32 %1329 to float, !dbg !11
  %1331 = select i1 %.not37, float %1326, float %1320, !dbg !11
  %1332 = bitcast float %1331 to i32, !dbg !11
  %1333 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %668, i32 %1332), !dbg !11
  %1334 = bitcast i32 %1333 to float, !dbg !11
  %1335 = insertelement <4 x float> poison, float %1319, i64 0, !dbg !11
  %1336 = insertelement <4 x float> %1335, float %1320, i64 1, !dbg !11
  %1337 = insertelement <4 x float> %1336, float %1330, i64 2, !dbg !11
  %1338 = insertelement <4 x float> %1337, float %1334, i64 3, !dbg !11
  %1339 = insertelement <4 x float> poison, float %1330, i64 0, !dbg !11
  %1340 = insertelement <4 x float> %1339, float %1334, i64 1, !dbg !11
  %1341 = insertelement <4 x float> %1340, float %1325, i64 2, !dbg !11
  %1342 = insertelement <4 x float> %1341, float %1326, i64 3, !dbg !11
  %1343 = select <4 x i1> %1301, <4 x float> %1338, <4 x float> %1342, !dbg !11
  %1344 = extractelement <4 x float> %1086, i64 0, !dbg !11
  %1345 = extractelement <4 x float> %1086, i64 1, !dbg !11
  %1346 = extractelement <4 x float> %1086, i64 2, !dbg !11
  %1347 = extractelement <4 x float> %1086, i64 3, !dbg !11
  %1348 = select i1 %.not36, float %1345, float %1344, !dbg !11
  %1349 = bitcast float %1348 to i32, !dbg !11
  %1350 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %660, i32 %1349), !dbg !11
  %1351 = bitcast i32 %1350 to float, !dbg !11
  %1352 = select i1 %.not36, float %1344, float %1351, !dbg !11
  %1353 = select i1 %.not36, float %1351, float %1345, !dbg !11
  %1354 = select i1 %.not36, float %1347, float %1346, !dbg !11
  %1355 = bitcast float %1354 to i32, !dbg !11
  %1356 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %660, i32 %1355), !dbg !11
  %1357 = bitcast i32 %1356 to float, !dbg !11
  %1358 = select i1 %.not36, float %1346, float %1357, !dbg !11
  %1359 = select i1 %.not36, float %1357, float %1347, !dbg !11
  %1360 = select i1 %.not37, float %1358, float %1352, !dbg !11
  %1361 = bitcast float %1360 to i32, !dbg !11
  %1362 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %668, i32 %1361), !dbg !11
  %1363 = bitcast i32 %1362 to float, !dbg !11
  %1364 = select i1 %.not37, float %1359, float %1353, !dbg !11
  %1365 = bitcast float %1364 to i32, !dbg !11
  %1366 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %668, i32 %1365), !dbg !11
  %1367 = bitcast i32 %1366 to float, !dbg !11
  %1368 = insertelement <4 x float> poison, float %1352, i64 0, !dbg !11
  %1369 = insertelement <4 x float> %1368, float %1353, i64 1, !dbg !11
  %1370 = insertelement <4 x float> %1369, float %1363, i64 2, !dbg !11
  %1371 = insertelement <4 x float> %1370, float %1367, i64 3, !dbg !11
  %1372 = insertelement <4 x float> poison, float %1363, i64 0, !dbg !11
  %1373 = insertelement <4 x float> %1372, float %1367, i64 1, !dbg !11
  %1374 = insertelement <4 x float> %1373, float %1358, i64 2, !dbg !11
  %1375 = insertelement <4 x float> %1374, float %1359, i64 3, !dbg !11
  %1376 = select <4 x i1> %1301, <4 x float> %1371, <4 x float> %1375, !dbg !11
  %1377 = extractelement <4 x float> %992, i64 0, !dbg !11
  %1378 = extractelement <4 x float> %992, i64 1, !dbg !11
  %1379 = extractelement <4 x float> %992, i64 2, !dbg !11
  %1380 = extractelement <4 x float> %992, i64 3, !dbg !11
  %1381 = select i1 %.not36, float %1378, float %1377, !dbg !11
  %1382 = bitcast float %1381 to i32, !dbg !11
  %1383 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %660, i32 %1382), !dbg !11
  %1384 = bitcast i32 %1383 to float, !dbg !11
  %1385 = select i1 %.not36, float %1377, float %1384, !dbg !11
  %1386 = select i1 %.not36, float %1384, float %1378, !dbg !11
  %1387 = select i1 %.not36, float %1380, float %1379, !dbg !11
  %1388 = bitcast float %1387 to i32, !dbg !11
  %1389 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %660, i32 %1388), !dbg !11
  %1390 = bitcast i32 %1389 to float, !dbg !11
  %1391 = select i1 %.not36, float %1379, float %1390, !dbg !11
  %1392 = select i1 %.not36, float %1390, float %1380, !dbg !11
  %1393 = select i1 %.not37, float %1391, float %1385, !dbg !11
  %1394 = bitcast float %1393 to i32, !dbg !11
  %1395 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %668, i32 %1394), !dbg !11
  %1396 = select i1 %.not37, float %1392, float %1386, !dbg !11
  %1397 = bitcast float %1396 to i32, !dbg !11
  %1398 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %668, i32 %1397), !dbg !11
  %1399 = shufflevector <1 x float> %1273, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !54
  %1400 = extractelement <4 x float> %1239, i64 0, !dbg !11
  %1401 = extractelement <4 x float> %1239, i64 1, !dbg !11
  %1402 = extractelement <4 x float> %1239, i64 2, !dbg !11
  %1403 = extractelement <4 x float> %1239, i64 3, !dbg !11
  %1404 = select i1 %.not36, float %1401, float %1400, !dbg !11
  %1405 = bitcast float %1404 to i32, !dbg !11
  %1406 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %660, i32 %1405), !dbg !11
  %1407 = bitcast i32 %1406 to float, !dbg !11
  %1408 = select i1 %.not36, float %1400, float %1407, !dbg !11
  %1409 = select i1 %.not36, float %1407, float %1401, !dbg !11
  %1410 = select i1 %.not36, float %1403, float %1402, !dbg !11
  %1411 = bitcast float %1410 to i32, !dbg !11
  %1412 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %660, i32 %1411), !dbg !11
  %1413 = bitcast i32 %1412 to float, !dbg !11
  %1414 = select i1 %.not36, float %1402, float %1413, !dbg !11
  %1415 = select i1 %.not36, float %1413, float %1403, !dbg !11
  %1416 = select i1 %.not37, float %1414, float %1408, !dbg !11
  %1417 = bitcast float %1416 to i32, !dbg !11
  %1418 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %668, i32 %1417), !dbg !11
  %1419 = bitcast i32 %1418 to float, !dbg !11
  %1420 = select i1 %.not37, float %1415, float %1409, !dbg !11
  %1421 = bitcast float %1420 to i32, !dbg !11
  %1422 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %668, i32 %1421), !dbg !11
  %1423 = bitcast i32 %1422 to float, !dbg !11
  %1424 = insertelement <4 x float> poison, float %1408, i64 0, !dbg !11
  %1425 = insertelement <4 x float> %1424, float %1409, i64 1, !dbg !11
  %1426 = insertelement <4 x float> %1425, float %1419, i64 2, !dbg !11
  %1427 = insertelement <4 x float> %1426, float %1423, i64 3, !dbg !11
  %1428 = insertelement <4 x float> poison, float %1419, i64 0, !dbg !11
  %1429 = insertelement <4 x float> %1428, float %1423, i64 1, !dbg !11
  %1430 = insertelement <4 x float> %1429, float %1414, i64 2, !dbg !11
  %1431 = insertelement <4 x float> %1430, float %1415, i64 3, !dbg !11
  %1432 = select <4 x i1> %1301, <4 x float> %1427, <4 x float> %1431, !dbg !11
  %1433 = extractelement <4 x float> %1164, i64 0, !dbg !11
  %1434 = extractelement <4 x float> %1164, i64 1, !dbg !11
  %1435 = extractelement <4 x float> %1164, i64 2, !dbg !11
  %1436 = extractelement <4 x float> %1164, i64 3, !dbg !11
  %1437 = select i1 %.not36, float %1434, float %1433, !dbg !11
  %1438 = bitcast float %1437 to i32, !dbg !11
  %1439 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %660, i32 %1438), !dbg !11
  %1440 = bitcast i32 %1439 to float, !dbg !11
  %1441 = select i1 %.not36, float %1433, float %1440, !dbg !11
  %1442 = select i1 %.not36, float %1440, float %1434, !dbg !11
  %1443 = select i1 %.not36, float %1436, float %1435, !dbg !11
  %1444 = bitcast float %1443 to i32, !dbg !11
  %1445 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %660, i32 %1444), !dbg !11
  %1446 = bitcast i32 %1445 to float, !dbg !11
  %1447 = select i1 %.not36, float %1435, float %1446, !dbg !11
  %1448 = select i1 %.not36, float %1446, float %1436, !dbg !11
  %1449 = select i1 %.not37, float %1447, float %1441, !dbg !11
  %1450 = bitcast float %1449 to i32, !dbg !11
  %1451 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %668, i32 %1450), !dbg !11
  %1452 = bitcast i32 %1451 to float, !dbg !11
  %1453 = select i1 %.not37, float %1448, float %1442, !dbg !11
  %1454 = bitcast float %1453 to i32, !dbg !11
  %1455 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %668, i32 %1454), !dbg !11
  %1456 = bitcast i32 %1455 to float, !dbg !11
  %1457 = insertelement <4 x float> poison, float %1441, i64 0, !dbg !11
  %1458 = insertelement <4 x float> %1457, float %1442, i64 1, !dbg !11
  %1459 = insertelement <4 x float> %1458, float %1452, i64 2, !dbg !11
  %1460 = insertelement <4 x float> %1459, float %1456, i64 3, !dbg !11
  %1461 = insertelement <4 x float> poison, float %1452, i64 0, !dbg !11
  %1462 = insertelement <4 x float> %1461, float %1456, i64 1, !dbg !11
  %1463 = insertelement <4 x float> %1462, float %1447, i64 2, !dbg !11
  %1464 = insertelement <4 x float> %1463, float %1448, i64 3, !dbg !11
  %1465 = select <4 x i1> %1301, <4 x float> %1460, <4 x float> %1464, !dbg !11
  %1466 = extractelement <4 x float> %1088, i64 0, !dbg !11
  %1467 = extractelement <4 x float> %1088, i64 1, !dbg !11
  %1468 = extractelement <4 x float> %1088, i64 2, !dbg !11
  %1469 = extractelement <4 x float> %1088, i64 3, !dbg !11
  %1470 = select i1 %.not36, float %1467, float %1466, !dbg !11
  %1471 = bitcast float %1470 to i32, !dbg !11
  %1472 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %660, i32 %1471), !dbg !11
  %1473 = bitcast i32 %1472 to float, !dbg !11
  %1474 = select i1 %.not36, float %1466, float %1473, !dbg !11
  %1475 = select i1 %.not36, float %1473, float %1467, !dbg !11
  %1476 = select i1 %.not36, float %1469, float %1468, !dbg !11
  %1477 = bitcast float %1476 to i32, !dbg !11
  %1478 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %660, i32 %1477), !dbg !11
  %1479 = bitcast i32 %1478 to float, !dbg !11
  %1480 = select i1 %.not36, float %1468, float %1479, !dbg !11
  %1481 = select i1 %.not36, float %1479, float %1469, !dbg !11
  %1482 = select i1 %.not37, float %1480, float %1474, !dbg !11
  %1483 = bitcast float %1482 to i32, !dbg !11
  %1484 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %668, i32 %1483), !dbg !11
  %1485 = bitcast i32 %1484 to float, !dbg !11
  %1486 = select i1 %.not37, float %1481, float %1475, !dbg !11
  %1487 = bitcast float %1486 to i32, !dbg !11
  %1488 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %668, i32 %1487), !dbg !11
  %1489 = bitcast i32 %1488 to float, !dbg !11
  %1490 = insertelement <4 x float> poison, float %1474, i64 0, !dbg !11
  %1491 = insertelement <4 x float> %1490, float %1475, i64 1, !dbg !11
  %1492 = insertelement <4 x float> %1491, float %1485, i64 2, !dbg !11
  %1493 = insertelement <4 x float> %1492, float %1489, i64 3, !dbg !11
  %1494 = insertelement <4 x float> poison, float %1485, i64 0, !dbg !11
  %1495 = insertelement <4 x float> %1494, float %1489, i64 1, !dbg !11
  %1496 = insertelement <4 x float> %1495, float %1480, i64 2, !dbg !11
  %1497 = insertelement <4 x float> %1496, float %1481, i64 3, !dbg !11
  %1498 = select <4 x i1> %1301, <4 x float> %1493, <4 x float> %1497, !dbg !11
  %1499 = extractelement <4 x float> %994, i64 0, !dbg !11
  %1500 = extractelement <4 x float> %994, i64 1, !dbg !11
  %1501 = extractelement <4 x float> %994, i64 2, !dbg !11
  %1502 = extractelement <4 x float> %994, i64 3, !dbg !11
  %1503 = select i1 %.not36, float %1500, float %1499, !dbg !11
  %1504 = bitcast float %1503 to i32, !dbg !11
  %1505 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %660, i32 %1504), !dbg !11
  %1506 = bitcast i32 %1505 to float, !dbg !11
  %1507 = select i1 %.not36, float %1499, float %1506, !dbg !11
  %1508 = select i1 %.not36, float %1506, float %1500, !dbg !11
  %1509 = select i1 %.not36, float %1502, float %1501, !dbg !11
  %1510 = bitcast float %1509 to i32, !dbg !11
  %1511 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %660, i32 %1510), !dbg !11
  %1512 = bitcast i32 %1511 to float, !dbg !11
  %1513 = select i1 %.not36, float %1501, float %1512, !dbg !11
  %1514 = select i1 %.not36, float %1512, float %1502, !dbg !11
  %1515 = select i1 %.not37, float %1513, float %1507, !dbg !11
  %1516 = bitcast float %1515 to i32, !dbg !11
  %1517 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %668, i32 %1516), !dbg !11
  %1518 = select i1 %.not37, float %1514, float %1508, !dbg !11
  %1519 = bitcast float %1518 to i32, !dbg !11
  %1520 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %668, i32 %1519), !dbg !11
  %1521 = extractelement <4 x float> %1240, i64 0, !dbg !11
  %1522 = extractelement <4 x float> %1240, i64 1, !dbg !11
  %1523 = extractelement <4 x float> %1240, i64 2, !dbg !11
  %1524 = extractelement <4 x float> %1240, i64 3, !dbg !11
  %1525 = select i1 %.not36, float %1522, float %1521, !dbg !11
  %1526 = bitcast float %1525 to i32, !dbg !11
  %1527 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %660, i32 %1526), !dbg !11
  %1528 = bitcast i32 %1527 to float, !dbg !11
  %1529 = select i1 %.not36, float %1521, float %1528, !dbg !11
  %1530 = select i1 %.not36, float %1528, float %1522, !dbg !11
  %1531 = select i1 %.not36, float %1524, float %1523, !dbg !11
  %1532 = bitcast float %1531 to i32, !dbg !11
  %1533 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %660, i32 %1532), !dbg !11
  %1534 = bitcast i32 %1533 to float, !dbg !11
  %1535 = select i1 %.not36, float %1523, float %1534, !dbg !11
  %1536 = select i1 %.not36, float %1534, float %1524, !dbg !11
  %1537 = select i1 %.not37, float %1535, float %1529, !dbg !11
  %1538 = bitcast float %1537 to i32, !dbg !11
  %1539 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %668, i32 %1538), !dbg !11
  %1540 = bitcast i32 %1539 to float, !dbg !11
  %1541 = select i1 %.not37, float %1536, float %1530, !dbg !11
  %1542 = bitcast float %1541 to i32, !dbg !11
  %1543 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %668, i32 %1542), !dbg !11
  %1544 = bitcast i32 %1543 to float, !dbg !11
  %1545 = insertelement <4 x float> poison, float %1529, i64 0, !dbg !11
  %1546 = insertelement <4 x float> %1545, float %1530, i64 1, !dbg !11
  %1547 = insertelement <4 x float> %1546, float %1540, i64 2, !dbg !11
  %1548 = insertelement <4 x float> %1547, float %1544, i64 3, !dbg !11
  %1549 = insertelement <4 x float> poison, float %1540, i64 0, !dbg !11
  %1550 = insertelement <4 x float> %1549, float %1544, i64 1, !dbg !11
  %1551 = insertelement <4 x float> %1550, float %1535, i64 2, !dbg !11
  %1552 = insertelement <4 x float> %1551, float %1536, i64 3, !dbg !11
  %1553 = select <4 x i1> %1301, <4 x float> %1548, <4 x float> %1552, !dbg !11
  %1554 = extractelement <4 x float> %1165, i64 0, !dbg !11
  %1555 = extractelement <4 x float> %1165, i64 1, !dbg !11
  %1556 = extractelement <4 x float> %1165, i64 2, !dbg !11
  %1557 = extractelement <4 x float> %1165, i64 3, !dbg !11
  %1558 = select i1 %.not36, float %1555, float %1554, !dbg !11
  %1559 = bitcast float %1558 to i32, !dbg !11
  %1560 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %660, i32 %1559), !dbg !11
  %1561 = bitcast i32 %1560 to float, !dbg !11
  %1562 = select i1 %.not36, float %1554, float %1561, !dbg !11
  %1563 = select i1 %.not36, float %1561, float %1555, !dbg !11
  %1564 = select i1 %.not36, float %1557, float %1556, !dbg !11
  %1565 = bitcast float %1564 to i32, !dbg !11
  %1566 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %660, i32 %1565), !dbg !11
  %1567 = bitcast i32 %1566 to float, !dbg !11
  %1568 = select i1 %.not36, float %1556, float %1567, !dbg !11
  %1569 = select i1 %.not36, float %1567, float %1557, !dbg !11
  %1570 = select i1 %.not37, float %1568, float %1562, !dbg !11
  %1571 = bitcast float %1570 to i32, !dbg !11
  %1572 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %668, i32 %1571), !dbg !11
  %1573 = bitcast i32 %1572 to float, !dbg !11
  %1574 = select i1 %.not37, float %1569, float %1563, !dbg !11
  %1575 = bitcast float %1574 to i32, !dbg !11
  %1576 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %668, i32 %1575), !dbg !11
  %1577 = bitcast i32 %1576 to float, !dbg !11
  %1578 = insertelement <4 x float> poison, float %1562, i64 0, !dbg !11
  %1579 = insertelement <4 x float> %1578, float %1563, i64 1, !dbg !11
  %1580 = insertelement <4 x float> %1579, float %1573, i64 2, !dbg !11
  %1581 = insertelement <4 x float> %1580, float %1577, i64 3, !dbg !11
  %1582 = insertelement <4 x float> poison, float %1573, i64 0, !dbg !11
  %1583 = insertelement <4 x float> %1582, float %1577, i64 1, !dbg !11
  %1584 = insertelement <4 x float> %1583, float %1568, i64 2, !dbg !11
  %1585 = insertelement <4 x float> %1584, float %1569, i64 3, !dbg !11
  %1586 = select <4 x i1> %1301, <4 x float> %1581, <4 x float> %1585, !dbg !11
  %1587 = extractelement <4 x float> %1089, i64 0, !dbg !11
  %1588 = extractelement <4 x float> %1089, i64 1, !dbg !11
  %1589 = extractelement <4 x float> %1089, i64 2, !dbg !11
  %1590 = extractelement <4 x float> %1089, i64 3, !dbg !11
  %1591 = select i1 %.not36, float %1588, float %1587, !dbg !11
  %1592 = bitcast float %1591 to i32, !dbg !11
  %1593 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %660, i32 %1592), !dbg !11
  %1594 = bitcast i32 %1593 to float, !dbg !11
  %1595 = select i1 %.not36, float %1587, float %1594, !dbg !11
  %1596 = select i1 %.not36, float %1594, float %1588, !dbg !11
  %1597 = select i1 %.not36, float %1590, float %1589, !dbg !11
  %1598 = bitcast float %1597 to i32, !dbg !11
  %1599 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %660, i32 %1598), !dbg !11
  %1600 = bitcast i32 %1599 to float, !dbg !11
  %1601 = select i1 %.not36, float %1589, float %1600, !dbg !11
  %1602 = select i1 %.not36, float %1600, float %1590, !dbg !11
  %1603 = select i1 %.not37, float %1601, float %1595, !dbg !11
  %1604 = bitcast float %1603 to i32, !dbg !11
  %1605 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %668, i32 %1604), !dbg !11
  %1606 = bitcast i32 %1605 to float, !dbg !11
  %1607 = select i1 %.not37, float %1602, float %1596, !dbg !11
  %1608 = bitcast float %1607 to i32, !dbg !11
  %1609 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %668, i32 %1608), !dbg !11
  %1610 = bitcast i32 %1609 to float, !dbg !11
  %1611 = insertelement <4 x float> poison, float %1595, i64 0, !dbg !11
  %1612 = insertelement <4 x float> %1611, float %1596, i64 1, !dbg !11
  %1613 = insertelement <4 x float> %1612, float %1606, i64 2, !dbg !11
  %1614 = insertelement <4 x float> %1613, float %1610, i64 3, !dbg !11
  %1615 = insertelement <4 x float> poison, float %1606, i64 0, !dbg !11
  %1616 = insertelement <4 x float> %1615, float %1610, i64 1, !dbg !11
  %1617 = insertelement <4 x float> %1616, float %1601, i64 2, !dbg !11
  %1618 = insertelement <4 x float> %1617, float %1602, i64 3, !dbg !11
  %1619 = select <4 x i1> %1301, <4 x float> %1614, <4 x float> %1618, !dbg !11
  %1620 = extractelement <4 x float> %1013, i64 0, !dbg !11
  %1621 = extractelement <4 x float> %1013, i64 1, !dbg !11
  %1622 = extractelement <4 x float> %1013, i64 2, !dbg !11
  %1623 = extractelement <4 x float> %1013, i64 3, !dbg !11
  %1624 = select i1 %.not36, float %1621, float %1620, !dbg !11
  %1625 = bitcast float %1624 to i32, !dbg !11
  %1626 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %660, i32 %1625), !dbg !11
  %1627 = bitcast i32 %1626 to float, !dbg !11
  %1628 = select i1 %.not36, float %1620, float %1627, !dbg !11
  %1629 = select i1 %.not36, float %1627, float %1621, !dbg !11
  %1630 = select i1 %.not36, float %1623, float %1622, !dbg !11
  %1631 = bitcast float %1630 to i32, !dbg !11
  %1632 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %660, i32 %1631), !dbg !11
  %1633 = bitcast i32 %1632 to float, !dbg !11
  %1634 = select i1 %.not36, float %1622, float %1633, !dbg !11
  %1635 = select i1 %.not36, float %1633, float %1623, !dbg !11
  %1636 = select i1 %.not37, float %1634, float %1628, !dbg !11
  %1637 = bitcast float %1636 to i32, !dbg !11
  %1638 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %668, i32 %1637), !dbg !11
  %1639 = select i1 %.not37, float %1635, float %1629, !dbg !11
  %1640 = bitcast float %1639 to i32, !dbg !11
  %1641 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %668, i32 %1640), !dbg !11
  %1642 = shufflevector <1 x float> %1268, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !54
  %1643 = extractelement <4 x float> %1241, i64 0, !dbg !11
  %1644 = extractelement <4 x float> %1241, i64 1, !dbg !11
  %1645 = extractelement <4 x float> %1241, i64 2, !dbg !11
  %1646 = extractelement <4 x float> %1241, i64 3, !dbg !11
  %1647 = select i1 %.not36, float %1644, float %1643, !dbg !11
  %1648 = bitcast float %1647 to i32, !dbg !11
  %1649 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %660, i32 %1648), !dbg !11
  %1650 = bitcast i32 %1649 to float, !dbg !11
  %1651 = select i1 %.not36, float %1643, float %1650, !dbg !11
  %1652 = select i1 %.not36, float %1650, float %1644, !dbg !11
  %1653 = select i1 %.not36, float %1646, float %1645, !dbg !11
  %1654 = bitcast float %1653 to i32, !dbg !11
  %1655 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %660, i32 %1654), !dbg !11
  %1656 = bitcast i32 %1655 to float, !dbg !11
  %1657 = select i1 %.not36, float %1645, float %1656, !dbg !11
  %1658 = select i1 %.not36, float %1656, float %1646, !dbg !11
  %1659 = select i1 %.not37, float %1657, float %1651, !dbg !11
  %1660 = bitcast float %1659 to i32, !dbg !11
  %1661 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %668, i32 %1660), !dbg !11
  %1662 = bitcast i32 %1661 to float, !dbg !11
  %1663 = select i1 %.not37, float %1658, float %1652, !dbg !11
  %1664 = bitcast float %1663 to i32, !dbg !11
  %1665 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %668, i32 %1664), !dbg !11
  %1666 = bitcast i32 %1665 to float, !dbg !11
  %1667 = insertelement <4 x float> poison, float %1651, i64 0, !dbg !11
  %1668 = insertelement <4 x float> %1667, float %1652, i64 1, !dbg !11
  %1669 = insertelement <4 x float> %1668, float %1662, i64 2, !dbg !11
  %1670 = insertelement <4 x float> %1669, float %1666, i64 3, !dbg !11
  %1671 = insertelement <4 x float> poison, float %1662, i64 0, !dbg !11
  %1672 = insertelement <4 x float> %1671, float %1666, i64 1, !dbg !11
  %1673 = insertelement <4 x float> %1672, float %1657, i64 2, !dbg !11
  %1674 = insertelement <4 x float> %1673, float %1658, i64 3, !dbg !11
  %1675 = select <4 x i1> %1301, <4 x float> %1670, <4 x float> %1674, !dbg !11
  %1676 = extractelement <4 x float> %1166, i64 0, !dbg !11
  %1677 = extractelement <4 x float> %1166, i64 1, !dbg !11
  %1678 = extractelement <4 x float> %1166, i64 2, !dbg !11
  %1679 = extractelement <4 x float> %1166, i64 3, !dbg !11
  %1680 = select i1 %.not36, float %1677, float %1676, !dbg !11
  %1681 = bitcast float %1680 to i32, !dbg !11
  %1682 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %660, i32 %1681), !dbg !11
  %1683 = bitcast i32 %1682 to float, !dbg !11
  %1684 = select i1 %.not36, float %1676, float %1683, !dbg !11
  %1685 = select i1 %.not36, float %1683, float %1677, !dbg !11
  %1686 = select i1 %.not36, float %1679, float %1678, !dbg !11
  %1687 = bitcast float %1686 to i32, !dbg !11
  %1688 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %660, i32 %1687), !dbg !11
  %1689 = bitcast i32 %1688 to float, !dbg !11
  %1690 = select i1 %.not36, float %1678, float %1689, !dbg !11
  %1691 = select i1 %.not36, float %1689, float %1679, !dbg !11
  %1692 = select i1 %.not37, float %1690, float %1684, !dbg !11
  %1693 = bitcast float %1692 to i32, !dbg !11
  %1694 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %668, i32 %1693), !dbg !11
  %1695 = bitcast i32 %1694 to float, !dbg !11
  %1696 = select i1 %.not37, float %1691, float %1685, !dbg !11
  %1697 = bitcast float %1696 to i32, !dbg !11
  %1698 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %668, i32 %1697), !dbg !11
  %1699 = bitcast i32 %1698 to float, !dbg !11
  %1700 = insertelement <4 x float> poison, float %1684, i64 0, !dbg !11
  %1701 = insertelement <4 x float> %1700, float %1685, i64 1, !dbg !11
  %1702 = insertelement <4 x float> %1701, float %1695, i64 2, !dbg !11
  %1703 = insertelement <4 x float> %1702, float %1699, i64 3, !dbg !11
  %1704 = insertelement <4 x float> poison, float %1695, i64 0, !dbg !11
  %1705 = insertelement <4 x float> %1704, float %1699, i64 1, !dbg !11
  %1706 = insertelement <4 x float> %1705, float %1690, i64 2, !dbg !11
  %1707 = insertelement <4 x float> %1706, float %1691, i64 3, !dbg !11
  %1708 = select <4 x i1> %1301, <4 x float> %1703, <4 x float> %1707, !dbg !11
  %1709 = extractelement <4 x float> %1090, i64 0, !dbg !11
  %1710 = extractelement <4 x float> %1090, i64 1, !dbg !11
  %1711 = extractelement <4 x float> %1090, i64 2, !dbg !11
  %1712 = extractelement <4 x float> %1090, i64 3, !dbg !11
  %1713 = select i1 %.not36, float %1710, float %1709, !dbg !11
  %1714 = bitcast float %1713 to i32, !dbg !11
  %1715 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %660, i32 %1714), !dbg !11
  %1716 = bitcast i32 %1715 to float, !dbg !11
  %1717 = select i1 %.not36, float %1709, float %1716, !dbg !11
  %1718 = select i1 %.not36, float %1716, float %1710, !dbg !11
  %1719 = select i1 %.not36, float %1712, float %1711, !dbg !11
  %1720 = bitcast float %1719 to i32, !dbg !11
  %1721 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %660, i32 %1720), !dbg !11
  %1722 = bitcast i32 %1721 to float, !dbg !11
  %1723 = select i1 %.not36, float %1711, float %1722, !dbg !11
  %1724 = select i1 %.not36, float %1722, float %1712, !dbg !11
  %1725 = select i1 %.not37, float %1723, float %1717, !dbg !11
  %1726 = bitcast float %1725 to i32, !dbg !11
  %1727 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %668, i32 %1726), !dbg !11
  %1728 = bitcast i32 %1727 to float, !dbg !11
  %1729 = select i1 %.not37, float %1724, float %1718, !dbg !11
  %1730 = bitcast float %1729 to i32, !dbg !11
  %1731 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %668, i32 %1730), !dbg !11
  %1732 = bitcast i32 %1731 to float, !dbg !11
  %1733 = insertelement <4 x float> poison, float %1717, i64 0, !dbg !11
  %1734 = insertelement <4 x float> %1733, float %1718, i64 1, !dbg !11
  %1735 = insertelement <4 x float> %1734, float %1728, i64 2, !dbg !11
  %1736 = insertelement <4 x float> %1735, float %1732, i64 3, !dbg !11
  %1737 = insertelement <4 x float> poison, float %1728, i64 0, !dbg !11
  %1738 = insertelement <4 x float> %1737, float %1732, i64 1, !dbg !11
  %1739 = insertelement <4 x float> %1738, float %1723, i64 2, !dbg !11
  %1740 = insertelement <4 x float> %1739, float %1724, i64 3, !dbg !11
  %1741 = select <4 x i1> %1301, <4 x float> %1736, <4 x float> %1740, !dbg !11
  %1742 = extractelement <4 x float> %1014, i64 0, !dbg !11
  %1743 = extractelement <4 x float> %1014, i64 1, !dbg !11
  %1744 = extractelement <4 x float> %1014, i64 2, !dbg !11
  %1745 = extractelement <4 x float> %1014, i64 3, !dbg !11
  %1746 = select i1 %.not36, float %1743, float %1742, !dbg !11
  %1747 = bitcast float %1746 to i32, !dbg !11
  %1748 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %660, i32 %1747), !dbg !11
  %1749 = bitcast i32 %1748 to float, !dbg !11
  %1750 = select i1 %.not36, float %1742, float %1749, !dbg !11
  %1751 = select i1 %.not36, float %1749, float %1743, !dbg !11
  %1752 = select i1 %.not36, float %1745, float %1744, !dbg !11
  %1753 = bitcast float %1752 to i32, !dbg !11
  %1754 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %660, i32 %1753), !dbg !11
  %1755 = bitcast i32 %1754 to float, !dbg !11
  %1756 = select i1 %.not36, float %1744, float %1755, !dbg !11
  %1757 = select i1 %.not36, float %1755, float %1745, !dbg !11
  %1758 = select i1 %.not37, float %1756, float %1750, !dbg !11
  %1759 = bitcast float %1758 to i32, !dbg !11
  %1760 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %668, i32 %1759), !dbg !11
  %1761 = select i1 %.not37, float %1757, float %1751, !dbg !11
  %1762 = bitcast float %1761 to i32, !dbg !11
  %1763 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %668, i32 %1762), !dbg !11
  %1764 = bitcast i32 %1760 to float, !dbg !11
  %1765 = bitcast i32 %1763 to float, !dbg !11
  %1766 = insertelement <4 x float> poison, float %1750, i64 0, !dbg !11
  %1767 = insertelement <4 x float> %1766, float %1751, i64 1, !dbg !11
  %1768 = insertelement <4 x float> %1767, float %1764, i64 2, !dbg !11
  %1769 = insertelement <4 x float> %1768, float %1765, i64 3, !dbg !11
  %1770 = insertelement <4 x float> poison, float %1764, i64 0, !dbg !11
  %1771 = insertelement <4 x float> %1770, float %1765, i64 1, !dbg !11
  %1772 = insertelement <4 x float> %1771, float %1756, i64 2, !dbg !11
  %1773 = insertelement <4 x float> %1772, float %1757, i64 3, !dbg !11
  %1774 = select <4 x i1> %1301, <4 x float> %1769, <4 x float> %1773, !dbg !11
  %row.reciprocal = fdiv arcp <4 x float> splat (float 1.000000e+00), %1642, !dbg !54
  %normalized = fmul <4 x float> %row.reciprocal, %1774, !dbg !54
  %1775 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !55
  %1776 = extractelement <4 x half> %1775, i64 3, !dbg !55
  %1777 = extractelement <4 x half> %1775, i64 2, !dbg !55
  %1778 = extractelement <4 x half> %1775, i64 1, !dbg !55
  %1779 = extractelement <4 x half> %1775, i64 0, !dbg !55
  %1780 = bitcast i32 %1638 to float, !dbg !11
  %1781 = bitcast i32 %1641 to float, !dbg !11
  %1782 = insertelement <4 x float> poison, float %1628, i64 0, !dbg !11
  %1783 = insertelement <4 x float> %1782, float %1629, i64 1, !dbg !11
  %1784 = insertelement <4 x float> %1783, float %1780, i64 2, !dbg !11
  %1785 = insertelement <4 x float> %1784, float %1781, i64 3, !dbg !11
  %1786 = insertelement <4 x float> poison, float %1780, i64 0, !dbg !11
  %1787 = insertelement <4 x float> %1786, float %1781, i64 1, !dbg !11
  %1788 = insertelement <4 x float> %1787, float %1634, i64 2, !dbg !11
  %1789 = insertelement <4 x float> %1788, float %1635, i64 3, !dbg !11
  %1790 = select <4 x i1> %1301, <4 x float> %1785, <4 x float> %1789, !dbg !11
  %normalized105 = fmul <4 x float> %row.reciprocal, %1790, !dbg !54
  %1791 = fptrunc <4 x float> %normalized105 to <4 x half>, !dbg !55
  %1792 = extractelement <4 x half> %1791, i64 3, !dbg !55
  %1793 = extractelement <4 x half> %1791, i64 2, !dbg !55
  %1794 = extractelement <4 x half> %1791, i64 1, !dbg !55
  %1795 = extractelement <4 x half> %1791, i64 0, !dbg !55
  %1796 = bitcast i32 %1517 to float, !dbg !11
  %1797 = bitcast i32 %1520 to float, !dbg !11
  %1798 = insertelement <4 x float> poison, float %1507, i64 0, !dbg !11
  %1799 = insertelement <4 x float> %1798, float %1508, i64 1, !dbg !11
  %1800 = insertelement <4 x float> %1799, float %1796, i64 2, !dbg !11
  %1801 = insertelement <4 x float> %1800, float %1797, i64 3, !dbg !11
  %1802 = insertelement <4 x float> poison, float %1796, i64 0, !dbg !11
  %1803 = insertelement <4 x float> %1802, float %1797, i64 1, !dbg !11
  %1804 = insertelement <4 x float> %1803, float %1513, i64 2, !dbg !11
  %1805 = insertelement <4 x float> %1804, float %1514, i64 3, !dbg !11
  %1806 = select <4 x i1> %1301, <4 x float> %1801, <4 x float> %1805, !dbg !11
  %row.reciprocal106 = fdiv arcp <4 x float> splat (float 1.000000e+00), %1399, !dbg !54
  %normalized107 = fmul <4 x float> %row.reciprocal106, %1806, !dbg !54
  %1807 = fptrunc <4 x float> %normalized107 to <4 x half>, !dbg !55
  %1808 = extractelement <4 x half> %1807, i64 3, !dbg !55
  %1809 = extractelement <4 x half> %1807, i64 2, !dbg !55
  %1810 = extractelement <4 x half> %1807, i64 1, !dbg !55
  %1811 = extractelement <4 x half> %1807, i64 0, !dbg !55
  %1812 = bitcast i32 %1395 to float, !dbg !11
  %1813 = bitcast i32 %1398 to float, !dbg !11
  %1814 = insertelement <4 x float> poison, float %1385, i64 0, !dbg !11
  %1815 = insertelement <4 x float> %1814, float %1386, i64 1, !dbg !11
  %1816 = insertelement <4 x float> %1815, float %1812, i64 2, !dbg !11
  %1817 = insertelement <4 x float> %1816, float %1813, i64 3, !dbg !11
  %1818 = insertelement <4 x float> poison, float %1812, i64 0, !dbg !11
  %1819 = insertelement <4 x float> %1818, float %1813, i64 1, !dbg !11
  %1820 = insertelement <4 x float> %1819, float %1391, i64 2, !dbg !11
  %1821 = insertelement <4 x float> %1820, float %1392, i64 3, !dbg !11
  %1822 = select <4 x i1> %1301, <4 x float> %1817, <4 x float> %1821, !dbg !11
  %normalized108 = fmul <4 x float> %row.reciprocal106, %1822, !dbg !54
  %1823 = fptrunc <4 x float> %normalized108 to <4 x half>, !dbg !55
  %1824 = extractelement <4 x half> %1823, i64 3, !dbg !55
  %1825 = extractelement <4 x half> %1823, i64 2, !dbg !55
  %1826 = extractelement <4 x half> %1823, i64 1, !dbg !55
  %1827 = extractelement <4 x half> %1823, i64 0, !dbg !55
  %address.thread1042 = tail call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #8, !dbg !56, !range !9, !frisk.register_lifetimes !5
  %address.local1043 = zext nneg i32 %address.thread1042 to i64, !dbg !8
  %address.local1045 = lshr i64 %address.local1043, 2, !dbg !14
  %address.local1046 = and i64 %address.local1045, 48, !dbg !14
  %address.local1048 = and i64 %address.local1043, 240, !dbg !56
  %address.local1047 = add i64 %22, %address.local1043, !dbg !56
  %address.local1050 = sub i64 %address.local1047, %address.local1048, !dbg !56
  %address.local1051 = add i64 %address.local1050, %address.local1046, !dbg !56
  %1828 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %24, !dbg !56
  %address.local1054.idx = shl nuw nsw i64 %address.local1051, 8, !dbg !56
  %address.local1054 = getelementptr inbounds nuw i8, ptr addrspace(1) %1828, i64 %address.local1054.idx, !dbg !56
  %address.local1056 = and i64 %address.local1045, 12, !dbg !10
  %address.local1057 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1056, !dbg !56
  store half %1827, ptr addrspace(1) %address.local1057, align 2, !dbg !56
  %address.local1059 = or disjoint i64 %address.local1056, 1, !dbg !56
  %address.local1060 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1059, !dbg !56
  store half %1826, ptr addrspace(1) %address.local1060, align 2, !dbg !56
  %address.local1062 = or disjoint i64 %address.local1056, 2, !dbg !56
  %address.local1063 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1062, !dbg !56
  store half %1825, ptr addrspace(1) %address.local1063, align 2, !dbg !56
  %address.local1065 = or disjoint i64 %address.local1056, 3, !dbg !56
  %address.local1066 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1065, !dbg !56
  store half %1824, ptr addrspace(1) %address.local1066, align 2, !dbg !56
  %address.local1068 = or disjoint i64 %address.local1056, 16, !dbg !56
  %address.local1069 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1068, !dbg !56
  store half %1811, ptr addrspace(1) %address.local1069, align 2, !dbg !56
  %address.local1071 = or disjoint i64 %address.local1056, 17, !dbg !56
  %address.local1072 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1071, !dbg !56
  store half %1810, ptr addrspace(1) %address.local1072, align 2, !dbg !56
  %address.local1074 = or disjoint i64 %address.local1056, 18, !dbg !56
  %address.local1075 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1074, !dbg !56
  store half %1809, ptr addrspace(1) %address.local1075, align 2, !dbg !56
  %address.local1077 = or disjoint i64 %address.local1056, 19, !dbg !56
  %address.local1078 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1077, !dbg !56
  store half %1808, ptr addrspace(1) %address.local1078, align 2, !dbg !56
  %address.local1082 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local1054, i64 16384, !dbg !56
  %address.local1083 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1056, !dbg !56
  store half %1795, ptr addrspace(1) %address.local1083, align 2, !dbg !56
  %address.local1085 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1059, !dbg !56
  store half %1794, ptr addrspace(1) %address.local1085, align 2, !dbg !56
  %address.local1087 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1062, !dbg !56
  store half %1793, ptr addrspace(1) %address.local1087, align 2, !dbg !56
  %address.local1089 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1065, !dbg !56
  store half %1792, ptr addrspace(1) %address.local1089, align 2, !dbg !56
  %address.local1091 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1068, !dbg !56
  store half %1779, ptr addrspace(1) %address.local1091, align 2, !dbg !56
  %address.local1093 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1071, !dbg !56
  store half %1778, ptr addrspace(1) %address.local1093, align 2, !dbg !56
  %address.local1095 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1074, !dbg !56
  store half %1777, ptr addrspace(1) %address.local1095, align 2, !dbg !56
  %address.local1097 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1077, !dbg !56
  store half %1776, ptr addrspace(1) %address.local1097, align 2, !dbg !56
  %normalized110 = fmul <4 x float> %row.reciprocal, %1741, !dbg !54
  %1829 = fptrunc <4 x float> %normalized110 to <4 x half>, !dbg !55
  %1830 = extractelement <4 x half> %1829, i64 3, !dbg !55
  %1831 = extractelement <4 x half> %1829, i64 2, !dbg !55
  %1832 = extractelement <4 x half> %1829, i64 1, !dbg !55
  %1833 = extractelement <4 x half> %1829, i64 0, !dbg !55
  %normalized111 = fmul <4 x float> %row.reciprocal, %1619, !dbg !54
  %1834 = fptrunc <4 x float> %normalized111 to <4 x half>, !dbg !55
  %1835 = extractelement <4 x half> %1834, i64 3, !dbg !55
  %1836 = extractelement <4 x half> %1834, i64 2, !dbg !55
  %1837 = extractelement <4 x half> %1834, i64 1, !dbg !55
  %1838 = extractelement <4 x half> %1834, i64 0, !dbg !55
  %normalized113 = fmul <4 x float> %row.reciprocal106, %1498, !dbg !54
  %1839 = fptrunc <4 x float> %normalized113 to <4 x half>, !dbg !55
  %1840 = extractelement <4 x half> %1839, i64 3, !dbg !55
  %1841 = extractelement <4 x half> %1839, i64 2, !dbg !55
  %1842 = extractelement <4 x half> %1839, i64 1, !dbg !55
  %1843 = extractelement <4 x half> %1839, i64 0, !dbg !55
  %normalized114 = fmul <4 x float> %row.reciprocal106, %1376, !dbg !54
  %1844 = fptrunc <4 x float> %normalized114 to <4 x half>, !dbg !55
  %1845 = extractelement <4 x half> %1844, i64 3, !dbg !55
  %1846 = extractelement <4 x half> %1844, i64 2, !dbg !55
  %1847 = extractelement <4 x half> %1844, i64 1, !dbg !55
  %1848 = extractelement <4 x half> %1844, i64 0, !dbg !55
  %address.local1113 = or disjoint i64 %address.local1056, 32, !dbg !56
  %address.local1114 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1113, !dbg !56
  store half %1848, ptr addrspace(1) %address.local1114, align 2, !dbg !56
  %address.local1116 = or disjoint i64 %address.local1056, 33, !dbg !56
  %address.local1117 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1116, !dbg !56
  store half %1847, ptr addrspace(1) %address.local1117, align 2, !dbg !56
  %address.local1119 = or disjoint i64 %address.local1056, 34, !dbg !56
  %address.local1120 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1119, !dbg !56
  store half %1846, ptr addrspace(1) %address.local1120, align 2, !dbg !56
  %address.local1122 = or disjoint i64 %address.local1056, 35, !dbg !56
  %address.local1123 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1122, !dbg !56
  store half %1845, ptr addrspace(1) %address.local1123, align 2, !dbg !56
  %address.local1125 = or disjoint i64 %address.local1056, 48, !dbg !56
  %address.local1126 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1125, !dbg !56
  store half %1843, ptr addrspace(1) %address.local1126, align 2, !dbg !56
  %address.local1128 = or disjoint i64 %address.local1056, 49, !dbg !56
  %address.local1129 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1128, !dbg !56
  store half %1842, ptr addrspace(1) %address.local1129, align 2, !dbg !56
  %address.local1131 = or disjoint i64 %address.local1056, 50, !dbg !56
  %address.local1132 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1131, !dbg !56
  store half %1841, ptr addrspace(1) %address.local1132, align 2, !dbg !56
  %address.local1134 = or i64 %address.local1045, 51, !dbg !56
  %address.local1135 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1134, !dbg !56
  store half %1840, ptr addrspace(1) %address.local1135, align 2, !dbg !56
  %address.local1140 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1113, !dbg !56
  store half %1838, ptr addrspace(1) %address.local1140, align 2, !dbg !56
  %address.local1142 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1116, !dbg !56
  store half %1837, ptr addrspace(1) %address.local1142, align 2, !dbg !56
  %address.local1144 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1119, !dbg !56
  store half %1836, ptr addrspace(1) %address.local1144, align 2, !dbg !56
  %address.local1146 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1122, !dbg !56
  store half %1835, ptr addrspace(1) %address.local1146, align 2, !dbg !56
  %address.local1148 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1125, !dbg !56
  store half %1833, ptr addrspace(1) %address.local1148, align 2, !dbg !56
  %address.local1150 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1128, !dbg !56
  store half %1832, ptr addrspace(1) %address.local1150, align 2, !dbg !56
  %address.local1152 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1131, !dbg !56
  store half %1831, ptr addrspace(1) %address.local1152, align 2, !dbg !56
  %address.local1154 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1134, !dbg !56
  store half %1830, ptr addrspace(1) %address.local1154, align 2, !dbg !56
  %normalized116 = fmul <4 x float> %row.reciprocal, %1708, !dbg !54
  %1849 = fptrunc <4 x float> %normalized116 to <4 x half>, !dbg !55
  %1850 = extractelement <4 x half> %1849, i64 3, !dbg !55
  %1851 = extractelement <4 x half> %1849, i64 2, !dbg !55
  %1852 = extractelement <4 x half> %1849, i64 1, !dbg !55
  %1853 = extractelement <4 x half> %1849, i64 0, !dbg !55
  %normalized117 = fmul <4 x float> %row.reciprocal, %1586, !dbg !54
  %1854 = fptrunc <4 x float> %normalized117 to <4 x half>, !dbg !55
  %1855 = extractelement <4 x half> %1854, i64 3, !dbg !55
  %1856 = extractelement <4 x half> %1854, i64 2, !dbg !55
  %1857 = extractelement <4 x half> %1854, i64 1, !dbg !55
  %1858 = extractelement <4 x half> %1854, i64 0, !dbg !55
  %normalized119 = fmul <4 x float> %row.reciprocal106, %1465, !dbg !54
  %1859 = fptrunc <4 x float> %normalized119 to <4 x half>, !dbg !55
  %1860 = extractelement <4 x half> %1859, i64 3, !dbg !55
  %1861 = extractelement <4 x half> %1859, i64 2, !dbg !55
  %1862 = extractelement <4 x half> %1859, i64 1, !dbg !55
  %1863 = extractelement <4 x half> %1859, i64 0, !dbg !55
  %normalized120 = fmul <4 x float> %row.reciprocal106, %1343, !dbg !54
  %1864 = fptrunc <4 x float> %normalized120 to <4 x half>, !dbg !55
  %1865 = extractelement <4 x half> %1864, i64 3, !dbg !55
  %1866 = extractelement <4 x half> %1864, i64 2, !dbg !55
  %1867 = extractelement <4 x half> %1864, i64 1, !dbg !55
  %1868 = extractelement <4 x half> %1864, i64 0, !dbg !55
  %address.local1170 = or disjoint i64 %address.local1056, 64, !dbg !56
  %address.local1171 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1170, !dbg !56
  store half %1868, ptr addrspace(1) %address.local1171, align 2, !dbg !56
  %address.local1173 = or disjoint i64 %address.local1056, 65, !dbg !56
  %address.local1174 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1173, !dbg !56
  store half %1867, ptr addrspace(1) %address.local1174, align 2, !dbg !56
  %address.local1176 = or disjoint i64 %address.local1056, 66, !dbg !56
  %address.local1177 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1176, !dbg !56
  store half %1866, ptr addrspace(1) %address.local1177, align 2, !dbg !56
  %address.local1179 = or disjoint i64 %address.local1056, 67, !dbg !56
  %address.local1180 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1179, !dbg !56
  store half %1865, ptr addrspace(1) %address.local1180, align 2, !dbg !56
  %address.local1182 = or disjoint i64 %address.local1056, 80, !dbg !56
  %address.local1183 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1182, !dbg !56
  store half %1863, ptr addrspace(1) %address.local1183, align 2, !dbg !56
  %address.local1185 = or disjoint i64 %address.local1056, 81, !dbg !56
  %address.local1186 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1185, !dbg !56
  store half %1862, ptr addrspace(1) %address.local1186, align 2, !dbg !56
  %address.local1188 = or disjoint i64 %address.local1056, 82, !dbg !56
  %address.local1189 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1188, !dbg !56
  store half %1861, ptr addrspace(1) %address.local1189, align 2, !dbg !56
  %address.local1191 = or disjoint i64 %address.local1056, 83, !dbg !56
  %address.local1192 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1191, !dbg !56
  store half %1860, ptr addrspace(1) %address.local1192, align 2, !dbg !56
  %address.local1197 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1170, !dbg !56
  store half %1858, ptr addrspace(1) %address.local1197, align 2, !dbg !56
  %address.local1199 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1173, !dbg !56
  store half %1857, ptr addrspace(1) %address.local1199, align 2, !dbg !56
  %address.local1201 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1176, !dbg !56
  store half %1856, ptr addrspace(1) %address.local1201, align 2, !dbg !56
  %address.local1203 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1179, !dbg !56
  store half %1855, ptr addrspace(1) %address.local1203, align 2, !dbg !56
  %address.local1205 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1182, !dbg !56
  store half %1853, ptr addrspace(1) %address.local1205, align 2, !dbg !56
  %address.local1207 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1185, !dbg !56
  store half %1852, ptr addrspace(1) %address.local1207, align 2, !dbg !56
  %address.local1209 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1188, !dbg !56
  store half %1851, ptr addrspace(1) %address.local1209, align 2, !dbg !56
  %address.local1211 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1191, !dbg !56
  store half %1850, ptr addrspace(1) %address.local1211, align 2, !dbg !56
  %normalized122 = fmul <4 x float> %row.reciprocal, %1675, !dbg !54
  %1869 = fptrunc <4 x float> %normalized122 to <4 x half>, !dbg !55
  %1870 = extractelement <4 x half> %1869, i64 3, !dbg !55
  %1871 = extractelement <4 x half> %1869, i64 2, !dbg !55
  %1872 = extractelement <4 x half> %1869, i64 1, !dbg !55
  %1873 = extractelement <4 x half> %1869, i64 0, !dbg !55
  %normalized123 = fmul <4 x float> %row.reciprocal, %1553, !dbg !54
  %1874 = fptrunc <4 x float> %normalized123 to <4 x half>, !dbg !55
  %1875 = extractelement <4 x half> %1874, i64 3, !dbg !55
  %1876 = extractelement <4 x half> %1874, i64 2, !dbg !55
  %1877 = extractelement <4 x half> %1874, i64 1, !dbg !55
  %1878 = extractelement <4 x half> %1874, i64 0, !dbg !55
  %normalized125 = fmul <4 x float> %row.reciprocal106, %1432, !dbg !54
  %1879 = fptrunc <4 x float> %normalized125 to <4 x half>, !dbg !55
  %1880 = extractelement <4 x half> %1879, i64 3, !dbg !55
  %1881 = extractelement <4 x half> %1879, i64 2, !dbg !55
  %1882 = extractelement <4 x half> %1879, i64 1, !dbg !55
  %1883 = extractelement <4 x half> %1879, i64 0, !dbg !55
  %normalized126 = fmul <4 x float> %row.reciprocal106, %1310, !dbg !54
  %1884 = fptrunc <4 x float> %normalized126 to <4 x half>, !dbg !55
  %1885 = extractelement <4 x half> %1884, i64 3, !dbg !55
  %1886 = extractelement <4 x half> %1884, i64 2, !dbg !55
  %1887 = extractelement <4 x half> %1884, i64 1, !dbg !55
  %1888 = extractelement <4 x half> %1884, i64 0, !dbg !55
  %address.local1227 = or disjoint i64 %address.local1056, 96, !dbg !56
  %address.local1228 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1227, !dbg !56
  store half %1888, ptr addrspace(1) %address.local1228, align 2, !dbg !56
  %address.local1230 = or disjoint i64 %address.local1056, 97, !dbg !56
  %address.local1231 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1230, !dbg !56
  store half %1887, ptr addrspace(1) %address.local1231, align 2, !dbg !56
  %address.local1233 = or disjoint i64 %address.local1056, 98, !dbg !56
  %address.local1234 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1233, !dbg !56
  store half %1886, ptr addrspace(1) %address.local1234, align 2, !dbg !56
  %address.local1236 = or disjoint i64 %address.local1056, 99, !dbg !56
  %address.local1237 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1236, !dbg !56
  store half %1885, ptr addrspace(1) %address.local1237, align 2, !dbg !56
  %address.local1239 = or disjoint i64 %address.local1056, 112, !dbg !56
  %address.local1240 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1239, !dbg !56
  store half %1883, ptr addrspace(1) %address.local1240, align 2, !dbg !56
  %address.local1242 = or disjoint i64 %address.local1056, 113, !dbg !56
  %address.local1243 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1242, !dbg !56
  store half %1882, ptr addrspace(1) %address.local1243, align 2, !dbg !56
  %address.local1245 = or disjoint i64 %address.local1056, 114, !dbg !56
  %address.local1246 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1245, !dbg !56
  store half %1881, ptr addrspace(1) %address.local1246, align 2, !dbg !56
  %address.local1248 = or i64 %address.local1045, 115, !dbg !56
  %address.local1249 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1248, !dbg !56
  store half %1880, ptr addrspace(1) %address.local1249, align 2, !dbg !56
  %address.local1254 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1227, !dbg !56
  store half %1878, ptr addrspace(1) %address.local1254, align 2, !dbg !56
  %address.local1256 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1230, !dbg !56
  store half %1877, ptr addrspace(1) %address.local1256, align 2, !dbg !56
  %address.local1258 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1233, !dbg !56
  store half %1876, ptr addrspace(1) %address.local1258, align 2, !dbg !56
  %address.local1260 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1236, !dbg !56
  store half %1875, ptr addrspace(1) %address.local1260, align 2, !dbg !56
  %address.local1262 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1239, !dbg !56
  store half %1873, ptr addrspace(1) %address.local1262, align 2, !dbg !56
  %address.local1264 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1242, !dbg !56
  store half %1872, ptr addrspace(1) %address.local1264, align 2, !dbg !56
  %address.local1266 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1245, !dbg !56
  store half %1871, ptr addrspace(1) %address.local1266, align 2, !dbg !56
  %address.local1268 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1248, !dbg !56
  store half %1870, ptr addrspace(1) %address.local1268, align 2, !dbg !56
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

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.sched.barrier(i32 immarg) #4

attributes #0 = { convergent mustprogress nofree willreturn memory(read) }
attributes #1 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1" "frisk.cross_bn_k0"="sum-anchor-v1" "frisk.global_prefetch"="8" "frisk.k_dma"="64-swizzle:1" "frisk.k_lds_prefetch"="8" "frisk.k_tail_prefetch"="4" "frisk.register_comparisons"="1" "frisk.register_lifetimes"="1" "frisk.softmax_fragments"="pv" "uniform-work-group-size" }
attributes #2 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { convergent mustprogress nocallback nocreateundeforpoison nofree nounwind willreturn memory(none) }
attributes #4 = { convergent mustprogress nocallback nofree nounwind willreturn }
attributes #5 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }
attributes #6 = { convergent mustprogress nocallback nofree nounwind willreturn memory(none) }
attributes #7 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #8 = { convergent nounwind willreturn memory(none) }
attributes #9 = { convergent nounwind }
attributes #10 = { memory(inaccessiblemem: readwrite) }
attributes #11 = { convergent willreturn memory(read) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!1 = !DIFile(filename: "causal.input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/causal_best_mytest")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !1, file: !1, line: 23, type: !4, scopeLine: 23, spFlags: DISPFlagDefinition, unit: !0)
!4 = !DISubroutineType(types: !5)
!5 = !{}
!6 = !{i32 256, i32 1, i32 1}
!7 = !DILocation(line: 23, column: 3, scope: !3)
!8 = !DILocation(line: 70, column: 22, scope: !3)
!9 = !{i32 0, i32 256}
!10 = !DILocation(line: 72, column: 24, scope: !3)
!11 = !DILocation(line: 60, column: 24, scope: !3)
!12 = !DILocation(line: 71, column: 22, scope: !3)
!13 = !DILocation(line: 73, column: 26, scope: !3)
!14 = !DILocation(line: 77, column: 19, scope: !3)
!15 = !DILocation(line: 79, column: 24, scope: !3)
!16 = !DILocation(line: 80, column: 23, scope: !3)
!17 = !{i32 0}
!18 = !{i32 1}
!19 = !{i32 2}
!20 = !{i32 3}
!21 = !{i32 4}
!22 = !{i32 5}
!23 = !{i32 6}
!24 = !{i32 7}
!25 = !{i32 8}
!26 = !{i32 9}
!27 = !{i32 10}
!28 = !{i32 11}
!29 = !{i32 12}
!30 = !{i32 13}
!31 = !{i32 14}
!32 = !{i32 15}
!33 = !{i32 16}
!34 = !{i32 17}
!35 = !{i32 18}
!36 = !{i32 19}
!37 = !{i32 20}
!38 = !{i32 21}
!39 = !{i32 22}
!40 = !DILocation(line: 101, column: 20, scope: !3)
!41 = !DILocation(line: 81, column: 26, scope: !3)
!42 = !DILocation(line: 88, column: 21, scope: !3)
!43 = !DILocation(line: 89, column: 20, scope: !3)
!44 = !DILocation(line: 90, column: 22, scope: !3)
!45 = !DILocation(line: 91, column: 23, scope: !3)
!46 = !DILocation(line: 92, column: 20, scope: !3)
!47 = !DILocation(line: 103, column: 22, scope: !3)
!48 = !DILocation(line: 93, column: 22, scope: !3)
!49 = !DILocation(line: 102, column: 23, scope: !3)
!50 = !{i64 8192}
!51 = !DILocation(line: 94, column: 22, scope: !3)
!52 = !DILocation(line: 96, column: 16, scope: !3)
!53 = !DILocation(line: 95, column: 20, scope: !3)
!54 = !DILocation(line: 107, column: 23, scope: !3)
!55 = !DILocation(line: 108, column: 19, scope: !3)
!56 = !DILocation(line: 109, column: 9, scope: !3)
