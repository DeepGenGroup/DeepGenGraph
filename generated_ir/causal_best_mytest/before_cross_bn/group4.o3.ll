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
  %114 = ptrtoint ptr addrspace(1) %1 to i64
  %115 = insertelement <2 x i64> poison, i64 %114, i64 0
  %116 = bitcast <2 x i64> %115 to <4 x i32>
  %dma.buffer.rsrc1701 = shufflevector <4 x i32> %116, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  %kdma.source = zext nneg i32 %kdma.owner to i64
  %kdma.source1878 = lshr i64 %kdma.source, 4
  %kdma.source1879 = and i64 %kdma.source1878, 8
  %kdma.source1880 = and i64 %kdma.source, 91
  %kdma.source1881 = lshr i64 %kdma.source, 2
  %kdma.source1882 = and i64 %kdma.source1881, 4
  %kdma.source1883 = add nuw nsw i64 %kdma.source1882, %kdma.source
  %kdma.source1884 = and i64 %kdma.source1883, 4
  %kdma.source1885 = or disjoint i64 %kdma.source1884, %kdma.source1880
  %kdma.source1886 = shl nuw nsw i64 %kdma.source1885, 7
  %kdma.source1887 = or disjoint i64 %kdma.source1886, %kdma.source1879
  %kdma.source1891 = zext nneg i32 %kdma.owner.hi to i64
  %kdma.source1892 = lshr i64 %kdma.source1891, 4
  %kdma.source1893 = and i64 %kdma.source1892, 8
  %kdma.source1894 = and i64 %kdma.source1891, 123
  %kdma.source1895 = lshr i64 %kdma.source1891, 2
  %kdma.source1896 = and i64 %kdma.source1895, 4
  %kdma.source1897 = add nuw nsw i64 %kdma.source1896, %kdma.source1891
  %kdma.source1898 = and i64 %kdma.source1897, 4
  %kdma.source1899 = or disjoint i64 %kdma.source1898, %kdma.source1894
  %kdma.source1900 = shl nuw nsw i64 %kdma.source1899, 7
  %kdma.source1901 = or disjoint i64 %kdma.source1900, %kdma.source1893
  %kdma.source1918 = or disjoint i64 %kdma.source1887, 16
  %kdma.source1935 = or disjoint i64 %kdma.source1901, 16
  %kdma.source1955 = or disjoint i32 %kdma.intra, 64
  %kdma.source2025 = or disjoint i32 %kdma.intra, 128
  %kdma.source2095 = or disjoint i32 %kdma.intra, 192
  %117 = ptrtoint ptr addrspace(1) %2 to i64
  %118 = insertelement <2 x i64> poison, i64 %117, i64 0
  %119 = bitcast <2 x i64> %118 to <4 x i32>
  %dma.buffer.rsrc1677 = shufflevector <4 x i32> %119, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  %120 = lshr i64 %112, 32
  %compare.high = trunc nuw i64 %120 to i32
  %compare.low.base = trunc i64 %112 to i32
  %121 = lshr i64 %113, 32
  %compare.high1393 = trunc nuw i64 %121 to i32
  %compare.low.base1397 = trunc i64 %113 to i32
  %122 = insertelement <4 x i32> poison, i32 %compare.low.base1397, i64 0, !dbg !16
  %123 = shufflevector <4 x i32> %122, <4 x i32> poison, <4 x i32> zeroinitializer, !dbg !16
  %124 = insertelement <4 x i32> poison, i32 %compare.low.base, i64 0, !dbg !16
  %125 = shufflevector <4 x i32> %124, <4 x i32> poison, <4 x i32> zeroinitializer, !dbg !16
  %invariant.op = sub i32 72, %kdma.intra, !dbg !11
  %invariant.op120 = sub i32 72, %kdma.intra, !dbg !11
  %invariant.op121 = sub i32 136, %kdma.intra, !dbg !11
  %invariant.op122 = sub i32 136, %kdma.intra, !dbg !11
  %invariant.op123 = sub i32 200, %kdma.intra, !dbg !11
  %invariant.op124 = sub i32 200, %kdma.intra, !dbg !11
  br label %126, !dbg !11

126:                                              ; preds = %4, %126
  %127 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %1271, %126 ]
  %128 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %1267, %126 ]
  %129 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %1263, %126 ]
  %130 = phi [2 x <1 x float>] [ [<1 x float> splat (float -inf), <1 x float> splat (float -inf)], %4 ], [ %1259, %126 ]
  %131 = phi [2 x <1 x float>] [ zeroinitializer, %4 ], [ %1023, %126 ]
  %132 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %1257, %126 ]
  %133 = phi i64 [ 0, %4 ], [ %1272, %126 ]
  %134 = lshr i64 %133, 12, !dbg !12
  %135 = add nuw nsw i64 %134, %17, !dbg !12
  %136 = lshr i64 %133, 5, !dbg !12
  %137 = and i64 %136, 127, !dbg !12
  %138 = shl nuw nsw i64 %133, 7, !dbg !12
  %139 = shl i64 %135, 19, !dbg !12
  %140 = and i64 %139, 16252928, !dbg !12
  %141 = shl i64 %136, 19, !dbg !12
  %142 = shl i64 %133, 14, !dbg !12
  %143 = sub i64 %142, %141, !dbg !12
  %address.thread566 = tail call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #8, !dbg !12, !range !9, !frisk.register_lifetimes !5
  %address.local567 = zext nneg i32 %address.thread566 to i64, !dbg !8
  %address.local568 = shl nuw nsw i64 %address.local567, 3, !dbg !8
  %address.local569 = and i64 %address.local568, 1536, !dbg !8
  %address.local570 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local569, !dbg !8
  %.idx1736 = or disjoint i64 %140, %137, !dbg !12
  %144 = add i64 %.idx1736, %143, !dbg !12
  %kdma.source1888 = add nuw nsw i64 %144, %kdma.source1887, !dbg !12
  %kdma.source1889 = trunc i64 %kdma.source1888 to i32, !dbg !12
  %kdma.source1890 = shl i32 %kdma.source1889, 1, !dbg !12
  %kdma.offset = add i32 %kdma.source1890, %kdma.intra, !dbg !12
  %kdma.source1902 = add i64 %144, %kdma.source1901, !dbg !12
  %kdma.source1903 = trunc i64 %kdma.source1902 to i32, !dbg !12
  %kdma.source1904 = shl i32 %kdma.source1903, 1, !dbg !12
  %kdma.offset1905 = add i32 %kdma.source1904, %kdma.intra.hi, !dbg !12
  %address.local584 = or disjoint i64 %address.local567, 256, !dbg !8
  %address.local585 = shl nuw nsw i64 %address.local584, 3, !dbg !8
  %address.local586 = and i64 %address.local585, 3584, !dbg !8
  %address.local587 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local586, !dbg !8
  %kdma.source1919 = add nuw nsw i64 %144, %kdma.source1918, !dbg !12
  %kdma.source1920 = trunc i64 %kdma.source1919 to i32, !dbg !12
  %kdma.source1921 = shl i32 %kdma.source1920, 1, !dbg !12
  %kdma.offset1922 = add i32 %kdma.source1921, %kdma.intra, !dbg !12
  %kdma.source1936 = add i64 %144, %kdma.source1935, !dbg !12
  %kdma.source1937 = trunc i64 %kdma.source1936 to i32, !dbg !12
  %kdma.source1938 = shl i32 %kdma.source1937, 1, !dbg !12
  %kdma.offset1939 = add i32 %kdma.source1938, %kdma.intra.hi, !dbg !12
  %145 = add nuw nsw i64 %133, 1024, !dbg !12
  %146 = lshr i64 %145, 12, !dbg !12
  %147 = add nuw nsw i64 %146, %17, !dbg !12
  %148 = add nsw i64 %136, %99, !dbg !12
  %149 = shl i64 %147, 19, !dbg !12
  %150 = and i64 %149, 16252928, !dbg !12
  %address.local601 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local569, !dbg !8
  %151 = add i64 %143, %136, !dbg !12
  %.idx1745 = add i64 %151, %99, !dbg !12
  %152 = add i64 %.idx1745, %150, !dbg !12
  %address.local605 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local586, !dbg !8
  %153 = insertelement <4 x ptr addrspace(3)> poison, ptr addrspace(3) %address.local570, i64 0, !dbg !8
  %154 = insertelement <4 x ptr addrspace(3)> %153, ptr addrspace(3) %address.local587, i64 1, !dbg !8
  %155 = insertelement <4 x ptr addrspace(3)> %154, ptr addrspace(3) %address.local601, i64 2, !dbg !8
  %156 = insertelement <4 x ptr addrspace(3)> %155, ptr addrspace(3) %address.local605, i64 3, !dbg !8
  %157 = ptrtoint <4 x ptr addrspace(3)> %156 to <4 x i64>, !dbg !8
  %158 = trunc <4 x i64> %157 to <4 x i32>, !dbg !8
  %159 = extractelement <4 x i32> %158, i64 0, !dbg !12
  %160 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %159), !dbg !12
  %dma.buffer.lds1702 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %160), !dbg !12
  %161 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1702, i32 %kdma.offset, <4 x i32> %dma.buffer.rsrc1701) #9, !dbg !12
  %kdma.lds1906 = add i32 %dma.buffer.lds1702, 512, !dbg !12
  %162 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1906, i32 %kdma.offset1905, <4 x i32> %dma.buffer.rsrc1701) #9, !dbg !12
  %163 = extractelement <4 x i32> %158, i64 1, !dbg !12
  %164 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %163), !dbg !12
  %dma.buffer.lds1699 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %164), !dbg !12
  %165 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1699, i32 %kdma.offset1922, <4 x i32> %dma.buffer.rsrc1701) #9, !dbg !12
  %kdma.lds1940 = add i32 %dma.buffer.lds1699, 512, !dbg !12
  %166 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1940, i32 %kdma.offset1939, <4 x i32> %dma.buffer.rsrc1701) #9, !dbg !12
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  tail call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %167 = extractelement <4 x i32> %158, i64 2, !dbg !12
  %168 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %167), !dbg !12
  %dma.buffer.lds1696 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %168), !dbg !12
  %kdma.lds1974 = add i32 %dma.buffer.lds1696, 512, !dbg !12
  %169 = extractelement <4 x i32> %158, i64 3, !dbg !12
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
  %170 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %.idx, !dbg !13
  %.idx1752 = mul nsw i64 %address.local610, -1280, !dbg !13
  %171 = getelementptr i8, ptr addrspace(3) %170, i64 %.idx1752, !dbg !13
  %172 = getelementptr [2 x i8], ptr addrspace(3) %171, i64 %address.local614, !dbg !13
  %173 = getelementptr [2 x i8], ptr addrspace(3) %172, i64 %address.local618, !dbg !13
  %174 = getelementptr [2 x i8], ptr addrspace(3) %173, i64 %address.local621, !dbg !13
  %address.local627 = getelementptr [2 x i8], ptr addrspace(3) %174, i64 %address.local625, !dbg !13
  %175 = add nsw i64 %.idx1752, %.idx, !dbg !13
  %.idx2277 = shl nuw nsw i64 %address.local614, 1, !dbg !13
  %176 = or disjoint i64 %175, %.idx2277, !dbg !13
  %177 = or disjoint i64 %address.local618, %address.local621, !dbg !13
  %178 = shl nsw i64 %177, 1, !dbg !13
  %address.local627.idx = mul nuw nsw i64 %address.local624, 2032, !dbg !13
  %179 = add nsw i64 %176, %address.local627.idx, !dbg !13
  %180 = add nsw i64 %179, %178, !dbg !13
  %181 = lshr i64 %180, 6, !dbg !13
  %182 = and i64 %181, 8, !dbg !13
  %183 = xor i64 %182, %180, !dbg !13
  %184 = sub i64 %183, %180, !dbg !13
  %kdma.swizzled = getelementptr i8, ptr addrspace(3) %address.local627, i64 %184, !dbg !13
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
  %185 = add nuw nsw i64 %address.local627.idx, 64, !dbg !13
  %kdma.gep2153 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 128, !dbg !13
  %kdma.gep2157 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1024, !dbg !13
  %kdma.gep2154 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4224, !dbg !13
  %kdma.gep2158 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5120, !dbg !13
  %kdma.gep2161 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1152, !dbg !13
  %kdma.gep2162 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5248, !dbg !13
  %186 = add nuw nsw i64 %133, 2048, !dbg !12
  %187 = lshr i64 %186, 12, !dbg !12
  %188 = add nuw nsw i64 %187, %17, !dbg !12
  %189 = insertelement <2 x i64> poison, i64 %148, i64 0, !dbg !12
  %190 = shufflevector <2 x i64> %189, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !12
  %191 = icmp slt <2 x i64> %190, <i64 -32, i64 -64>, !dbg !12
  %192 = extractelement <2 x i1> %191, i64 0, !dbg !12
  %193 = sext i1 %192 to i64, !dbg !12
  %194 = add nsw <2 x i64> %190, <i64 32, i64 64>, !dbg !12
  %195 = sub nsw <2 x i64> <i64 -33, i64 -65>, %190, !dbg !12
  %196 = select <2 x i1> %191, <2 x i64> %195, <2 x i64> %194, !dbg !12
  %197 = sdiv <2 x i64> %196, splat (i64 128), !dbg !12
  %198 = extractelement <2 x i64> %197, i64 0, !dbg !12
  %199 = xor i64 %198, %193, !dbg !12
  %200 = shl nsw i64 %199, 7, !dbg !12
  %201 = sub i64 %152, %200, !dbg !12
  %kdma.source1952 = add i64 %201, %kdma.source1887, !dbg !12
  %kdma.source1953 = trunc i64 %kdma.source1952 to i32, !dbg !12
  %kdma.source1954 = shl i32 %kdma.source1953, 1, !dbg !12
  %kdma.offset1956 = add i32 %kdma.source1955, %kdma.source1954, !dbg !12
  %kdma.source1969 = add i64 %201, %kdma.source1901, !dbg !12
  %kdma.source1970 = trunc i64 %kdma.source1969 to i32, !dbg !12
  %kdma.source1971 = shl i32 %kdma.source1970, 1, !dbg !12
  %kdma.offset1973.reass.reass = add i32 %kdma.source1971, %invariant.op
  %202 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1696, i32 %kdma.offset1956, <4 x i32> %dma.buffer.rsrc1701) #9, !dbg !12
  %203 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1974, i32 %kdma.offset1973.reass.reass, <4 x i32> %dma.buffer.rsrc1701) #9, !dbg !12
  %204 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %169), !dbg !12
  %dma.buffer.lds1693 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %204), !dbg !12
  %kdma.source1987 = add i64 %201, %kdma.source1918, !dbg !12
  %kdma.source1988 = trunc i64 %kdma.source1987 to i32, !dbg !12
  %kdma.source1989 = shl i32 %kdma.source1988, 1, !dbg !12
  %kdma.offset1991 = add i32 %kdma.source1955, %kdma.source1989, !dbg !12
  %205 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1693, i32 %kdma.offset1991, <4 x i32> %dma.buffer.rsrc1701) #9, !dbg !12
  %kdma.source2005 = add i64 %201, %kdma.source1935, !dbg !12
  %kdma.source2006 = trunc i64 %kdma.source2005 to i32, !dbg !12
  %kdma.source2007 = shl i32 %kdma.source2006, 1, !dbg !12
  %kdma.offset2009.reass.reass = add i32 %kdma.source2007, %invariant.op120
  %kdma.lds2010 = add i32 %dma.buffer.lds1693, 512, !dbg !12
  %206 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds2010, i32 %kdma.offset2009.reass.reass, <4 x i32> %dma.buffer.rsrc1701) #9, !dbg !12
  %207 = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8, !dbg !13
  %208 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %56, <4 x half> %207, <4 x float> zeroinitializer) #8, !dbg !13
  %209 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8, !dbg !13
  %address.base = tail call i64 asm sideeffect "", "=v,0"(i64 %105) #8, !dbg !13, !frisk.register_lifetimes !5
  %address.local650 = add nsw i64 %address.base, %address.local649, !dbg !13
  %address.local651 = shl i64 %address.local650, 6, !dbg !13
  %address.local652 = sub i64 %address.local646, %address.local651, !dbg !13
  %210 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local652, !dbg !13
  %211 = getelementptr [2 x i8], ptr addrspace(3) %210, i64 %address.local621, !dbg !13
  %212 = getelementptr [2 x i8], ptr addrspace(3) %211, i64 %address.local625, !dbg !13
  %address.local656 = getelementptr i8, ptr addrspace(3) %212, i64 64, !dbg !13
  %.idx22802283 = add i64 %address.local652, %address.local621, !dbg !13
  %213 = shl i64 %.idx22802283, 1, !dbg !13
  %214 = add i64 %185, %213, !dbg !13
  %215 = lshr i64 %214, 6, !dbg !13
  %216 = and i64 %215, 8, !dbg !13
  %217 = xor i64 %216, %214, !dbg !13
  %218 = sub i64 %217, %214, !dbg !13
  %kdma.swizzled2151 = getelementptr i8, ptr addrspace(3) %address.local656, i64 %218, !dbg !13
  %219 = load <4 x half>, ptr addrspace(3) %kdma.swizzled2151, align 8, !dbg !13, !frisk.k_lds_pair !17
  %kdma.gep2155 = getelementptr i8, ptr addrspace(3) %kdma.swizzled2151, i64 128, !dbg !13
  %220 = load <4 x half>, ptr addrspace(3) %kdma.gep2155, align 8, !dbg !13, !frisk.k_lds_pair !17
  %kdma.gep2152 = getelementptr i8, ptr addrspace(3) %kdma.swizzled2151, i64 4096, !dbg !13
  %221 = load <4 x half>, ptr addrspace(3) %kdma.gep2152, align 8, !dbg !13, !frisk.k_lds_pair !18
  %kdma.gep2156 = getelementptr i8, ptr addrspace(3) %kdma.swizzled2151, i64 4224, !dbg !13
  %222 = load <4 x half>, ptr addrspace(3) %kdma.gep2156, align 8, !dbg !13, !frisk.k_lds_pair !18
  %223 = load <4 x half>, ptr addrspace(3) %kdma.gep2153, align 8, !dbg !13, !frisk.k_lds_pair !19
  %224 = load <4 x half>, ptr addrspace(3) %kdma.gep2157, align 8, !dbg !13, !frisk.k_lds_pair !19
  %225 = load <4 x half>, ptr addrspace(3) %kdma.gep2154, align 8, !dbg !13, !frisk.k_lds_pair !20
  %226 = load <4 x half>, ptr addrspace(3) %kdma.gep2158, align 8, !dbg !13, !frisk.k_lds_pair !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %227 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %58, <4 x half> %209, <4 x float> %208) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %kdma.gep2159 = getelementptr i8, ptr addrspace(3) %kdma.swizzled2151, i64 1024, !dbg !13
  %228 = load <4 x half>, ptr addrspace(3) %kdma.gep2159, align 8, !dbg !13, !frisk.k_lds_pair !21
  %kdma.gep2163 = getelementptr i8, ptr addrspace(3) %kdma.swizzled2151, i64 1152, !dbg !13
  %229 = load <4 x half>, ptr addrspace(3) %kdma.gep2163, align 8, !dbg !13, !frisk.k_lds_pair !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %230 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %56, <4 x half> %219, <4 x float> zeroinitializer) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %kdma.gep2160 = getelementptr i8, ptr addrspace(3) %kdma.swizzled2151, i64 5120, !dbg !13
  %231 = load <4 x half>, ptr addrspace(3) %kdma.gep2160, align 8, !dbg !13, !frisk.k_lds_pair !22
  %kdma.gep2164 = getelementptr i8, ptr addrspace(3) %kdma.swizzled2151, i64 5248, !dbg !13
  %232 = load <4 x half>, ptr addrspace(3) %kdma.gep2164, align 8, !dbg !13, !frisk.k_lds_pair !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %233 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %58, <4 x half> %221, <4 x float> %230) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %234 = load <4 x half>, ptr addrspace(3) %kdma.gep2161, align 8, !dbg !13, !frisk.k_lds_pair !23
  %235 = load <4 x half>, ptr addrspace(3) %kdma.gep2162, align 8, !dbg !13, !frisk.k_lds_pair !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %236 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %56, <4 x half> %223, <4 x float> zeroinitializer) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %237 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %58, <4 x half> %225, <4 x float> %236) #8, !dbg !13
  %238 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %56, <4 x half> %220, <4 x float> zeroinitializer) #8, !dbg !13
  %239 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %58, <4 x half> %222, <4 x float> %238) #8, !dbg !13
  %240 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %56, <4 x half> %224, <4 x float> zeroinitializer) #8, !dbg !13
  %241 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %58, <4 x half> %226, <4 x float> %240) #8, !dbg !13
  %242 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %56, <4 x half> %228, <4 x float> zeroinitializer) #8, !dbg !13
  %243 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %58, <4 x half> %231, <4 x float> %242) #8, !dbg !13
  %244 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %56, <4 x half> %234, <4 x float> zeroinitializer) #8, !dbg !13
  %245 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %58, <4 x half> %235, <4 x float> %244) #8, !dbg !13
  %246 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %56, <4 x half> %229, <4 x float> zeroinitializer) #8, !dbg !13
  %247 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %58, <4 x half> %232, <4 x float> %246) #8, !dbg !13
  %248 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %207, <4 x float> zeroinitializer) #8, !dbg !13
  %249 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %209, <4 x float> %248) #8, !dbg !13
  %250 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %219, <4 x float> zeroinitializer) #8, !dbg !13
  %251 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %221, <4 x float> %250) #8, !dbg !13
  %252 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %223, <4 x float> zeroinitializer) #8, !dbg !13
  %253 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %225, <4 x float> %252) #8, !dbg !13
  %254 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %220, <4 x float> zeroinitializer) #8, !dbg !13
  %255 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %222, <4 x float> %254) #8, !dbg !13
  %256 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %224, <4 x float> zeroinitializer) #8, !dbg !13
  %257 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %226, <4 x float> %256) #8, !dbg !13
  %258 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %228, <4 x float> zeroinitializer) #8, !dbg !13
  %259 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %231, <4 x float> %258) #8, !dbg !13
  %260 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %234, <4 x float> zeroinitializer) #8, !dbg !13
  %261 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %235, <4 x float> %260) #8, !dbg !13
  %262 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %229, <4 x float> zeroinitializer) #8, !dbg !13
  %263 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %61, <4 x half> %232, <4 x float> %262) #8, !dbg !13
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  tail call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %264 = extractelement <2 x i1> %191, i64 1, !dbg !12
  %265 = sext i1 %264 to i64, !dbg !12
  %266 = extractelement <2 x i64> %197, i64 1, !dbg !12
  %267 = xor i64 %266, %265, !dbg !12
  %268 = shl i64 %188, 19, !dbg !12
  %269 = and i64 %268, 16252928, !dbg !12
  %270 = add i64 %.idx1745, %269, !dbg !12
  %271 = shl nsw i64 %267, 7, !dbg !12
  %272 = sub i64 %270, %271, !dbg !12
  %kdma.source2022 = add i64 %272, %kdma.source1887, !dbg !12
  %kdma.source2023 = trunc i64 %kdma.source2022 to i32, !dbg !12
  %kdma.source2024 = shl i32 %kdma.source2023, 1, !dbg !12
  %kdma.offset2026 = add i32 %kdma.source2025, %kdma.source2024, !dbg !12
  %273 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1702, i32 %kdma.offset2026, <4 x i32> %dma.buffer.rsrc1701) #9, !dbg !12
  %kdma.source2039 = add i64 %272, %kdma.source1901, !dbg !12
  %kdma.source2040 = trunc i64 %kdma.source2039 to i32, !dbg !12
  %kdma.source2041 = shl i32 %kdma.source2040, 1, !dbg !12
  %kdma.offset2043.reass.reass = add i32 %kdma.source2041, %invariant.op121
  %274 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1906, i32 %kdma.offset2043.reass.reass, <4 x i32> %dma.buffer.rsrc1701) #9, !dbg !12
  %kdma.source2057 = add i64 %272, %kdma.source1918, !dbg !12
  %kdma.source2058 = trunc i64 %kdma.source2057 to i32, !dbg !12
  %kdma.source2059 = shl i32 %kdma.source2058, 1, !dbg !12
  %kdma.offset2061 = add i32 %kdma.source2025, %kdma.source2059, !dbg !12
  %275 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1699, i32 %kdma.offset2061, <4 x i32> %dma.buffer.rsrc1701) #9, !dbg !12
  %kdma.source2075 = add i64 %272, %kdma.source1935, !dbg !12
  %kdma.source2076 = trunc i64 %kdma.source2075 to i32, !dbg !12
  %kdma.source2077 = shl i32 %kdma.source2076, 1, !dbg !12
  %kdma.offset2079.reass.reass = add i32 %kdma.source2077, %invariant.op122
  %276 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1940, i32 %kdma.offset2079.reass.reass, <4 x i32> %dma.buffer.rsrc1701) #9, !dbg !12
  %kdma.gep2181 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8192, !dbg !13
  %277 = load <4 x half>, ptr addrspace(3) %kdma.gep2181, align 8, !dbg !13, !frisk.k_lds_pair !24
  %kdma.gep2185 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8320, !dbg !13
  %278 = load <4 x half>, ptr addrspace(3) %kdma.gep2185, align 8, !dbg !13, !frisk.k_lds_pair !24
  %kdma.gep2182 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12288, !dbg !13
  %279 = load <4 x half>, ptr addrspace(3) %kdma.gep2182, align 8, !dbg !13, !frisk.k_lds_pair !25
  %kdma.gep2186 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12416, !dbg !13
  %280 = load <4 x half>, ptr addrspace(3) %kdma.gep2186, align 8, !dbg !13, !frisk.k_lds_pair !25
  %kdma.gep2183 = getelementptr i8, ptr addrspace(3) %kdma.swizzled2151, i64 8192, !dbg !13
  %281 = load <4 x half>, ptr addrspace(3) %kdma.gep2183, align 8, !dbg !13, !frisk.k_lds_pair !26
  %kdma.gep2187 = getelementptr i8, ptr addrspace(3) %kdma.swizzled2151, i64 8320, !dbg !13
  %282 = load <4 x half>, ptr addrspace(3) %kdma.gep2187, align 8, !dbg !13, !frisk.k_lds_pair !26
  %kdma.gep2184 = getelementptr i8, ptr addrspace(3) %kdma.swizzled2151, i64 12288, !dbg !13
  %283 = load <4 x half>, ptr addrspace(3) %kdma.gep2184, align 8, !dbg !13, !frisk.k_lds_pair !27
  %kdma.gep2188 = getelementptr i8, ptr addrspace(3) %kdma.swizzled2151, i64 12416, !dbg !13
  %284 = load <4 x half>, ptr addrspace(3) %kdma.gep2188, align 8, !dbg !13, !frisk.k_lds_pair !27
  %kdma.gep2189 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9216, !dbg !13
  %285 = load <4 x half>, ptr addrspace(3) %kdma.gep2189, align 8, !dbg !13, !frisk.k_lds_pair !28
  %kdma.gep2193 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9344, !dbg !13
  %286 = load <4 x half>, ptr addrspace(3) %kdma.gep2193, align 8, !dbg !13, !frisk.k_lds_pair !28
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %287 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %277, <4 x float> %227) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %kdma.gep2190 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13312, !dbg !13
  %288 = load <4 x half>, ptr addrspace(3) %kdma.gep2190, align 8, !dbg !13, !frisk.k_lds_pair !29
  %kdma.gep2194 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13440, !dbg !13
  %289 = load <4 x half>, ptr addrspace(3) %kdma.gep2194, align 8, !dbg !13, !frisk.k_lds_pair !29
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %290 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %279, <4 x float> %287) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %kdma.gep2191 = getelementptr i8, ptr addrspace(3) %kdma.swizzled2151, i64 9216, !dbg !13
  %291 = load <4 x half>, ptr addrspace(3) %kdma.gep2191, align 8, !dbg !13, !frisk.k_lds_pair !30
  %kdma.gep2195 = getelementptr i8, ptr addrspace(3) %kdma.swizzled2151, i64 9344, !dbg !13
  %292 = load <4 x half>, ptr addrspace(3) %kdma.gep2195, align 8, !dbg !13, !frisk.k_lds_pair !30
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %293 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %281, <4 x float> %233) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %kdma.gep2192 = getelementptr i8, ptr addrspace(3) %kdma.swizzled2151, i64 13312, !dbg !13
  %294 = load <4 x half>, ptr addrspace(3) %kdma.gep2192, align 8, !dbg !13, !frisk.k_lds_pair !31
  %kdma.gep2196 = getelementptr i8, ptr addrspace(3) %kdma.swizzled2151, i64 13440, !dbg !13
  %295 = load <4 x half>, ptr addrspace(3) %kdma.gep2196, align 8, !dbg !13, !frisk.k_lds_pair !31
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %296 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %283, <4 x float> %293) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %297 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %278, <4 x float> %237) #8, !dbg !13
  %298 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %280, <4 x float> %297) #8, !dbg !13
  %299 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %282, <4 x float> %239) #8, !dbg !13
  %300 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %284, <4 x float> %299) #8, !dbg !13
  %301 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %285, <4 x float> %241) #8, !dbg !13
  %302 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %288, <4 x float> %301) #8, !dbg !13
  %303 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %291, <4 x float> %243) #8, !dbg !13
  %304 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %294, <4 x float> %303) #8, !dbg !13
  %305 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %286, <4 x float> %245) #8, !dbg !13
  %306 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %289, <4 x float> %305) #8, !dbg !13
  %307 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %292, <4 x float> %247) #8, !dbg !13
  %308 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %295, <4 x float> %307) #8, !dbg !13
  %309 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %277, <4 x float> %249) #8, !dbg !13
  %310 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %279, <4 x float> %309) #8, !dbg !13
  %311 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %281, <4 x float> %251) #8, !dbg !13
  %312 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %283, <4 x float> %311) #8, !dbg !13
  %313 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %278, <4 x float> %253) #8, !dbg !13
  %314 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %280, <4 x float> %313) #8, !dbg !13
  %315 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %282, <4 x float> %255) #8, !dbg !13
  %316 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %284, <4 x float> %315) #8, !dbg !13
  %317 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %285, <4 x float> %257) #8, !dbg !13
  %318 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %288, <4 x float> %317) #8, !dbg !13
  %319 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %291, <4 x float> %259) #8, !dbg !13
  %320 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %294, <4 x float> %319) #8, !dbg !13
  %321 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %286, <4 x float> %261) #8, !dbg !13
  %322 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %289, <4 x float> %321) #8, !dbg !13
  %323 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %292, <4 x float> %263) #8, !dbg !13
  %324 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %295, <4 x float> %323) #8, !dbg !13
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  tail call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %325 = add nuw nsw i64 %133, 3072, !dbg !12
  %326 = lshr i64 %325, 12, !dbg !12
  %327 = add nuw nsw i64 %326, %17, !dbg !12
  %328 = add nsw i64 %148, 96, !dbg !12
  %329 = icmp slt i64 %148, -96, !dbg !12
  %330 = sub nsw i64 -97, %148, !dbg !12
  %331 = select i1 %329, i64 %330, i64 %328, !dbg !12
  %332 = sdiv i64 %331, 128, !dbg !12
  %333 = sext i1 %329 to i64, !dbg !12
  %334 = xor i64 %332, %333, !dbg !12
  %335 = shl i64 %327, 19, !dbg !12
  %336 = and i64 %335, 16252928, !dbg !12
  %337 = add i64 %.idx1745, %336, !dbg !12
  %338 = shl nsw i64 %334, 7, !dbg !12
  %339 = sub i64 %337, %338, !dbg !12
  %kdma.source2092 = add i64 %339, %kdma.source1887, !dbg !12
  %kdma.source2093 = trunc i64 %kdma.source2092 to i32, !dbg !12
  %kdma.source2094 = shl i32 %kdma.source2093, 1, !dbg !12
  %kdma.offset2096 = add i32 %kdma.source2095, %kdma.source2094, !dbg !12
  %340 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1696, i32 %kdma.offset2096, <4 x i32> %dma.buffer.rsrc1701) #9, !dbg !12
  %kdma.source2109 = add i64 %339, %kdma.source1901, !dbg !12
  %kdma.source2110 = trunc i64 %kdma.source2109 to i32, !dbg !12
  %kdma.source2111 = shl i32 %kdma.source2110, 1, !dbg !12
  %kdma.offset2113.reass.reass = add i32 %kdma.source2111, %invariant.op123
  %341 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1974, i32 %kdma.offset2113.reass.reass, <4 x i32> %dma.buffer.rsrc1701) #9, !dbg !12
  %kdma.source2127 = add i64 %339, %kdma.source1918, !dbg !12
  %kdma.source2128 = trunc i64 %kdma.source2127 to i32, !dbg !12
  %kdma.source2129 = shl i32 %kdma.source2128, 1, !dbg !12
  %kdma.offset2131 = add i32 %kdma.source2095, %kdma.source2129, !dbg !12
  %342 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1693, i32 %kdma.offset2131, <4 x i32> %dma.buffer.rsrc1701) #9, !dbg !12
  %kdma.source2145 = add i64 %339, %kdma.source1935, !dbg !12
  %kdma.source2146 = trunc i64 %kdma.source2145 to i32, !dbg !12
  %kdma.source2147 = shl i32 %kdma.source2146, 1, !dbg !12
  %kdma.offset2149.reass.reass = add i32 %kdma.source2147, %invariant.op124
  %343 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds2010, i32 %kdma.offset2149.reass.reass, <4 x i32> %dma.buffer.rsrc1701) #9, !dbg !12
  %344 = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8, !dbg !13, !frisk.k_lds_pair !32
  %345 = load <4 x half>, ptr addrspace(3) %kdma.gep2153, align 8, !dbg !13, !frisk.k_lds_pair !32
  %346 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8, !dbg !13, !frisk.k_lds_pair !33
  %347 = load <4 x half>, ptr addrspace(3) %kdma.gep2154, align 8, !dbg !13, !frisk.k_lds_pair !33
  %348 = load <4 x half>, ptr addrspace(3) %kdma.swizzled2151, align 8, !dbg !13, !frisk.k_lds_pair !34
  %349 = load <4 x half>, ptr addrspace(3) %kdma.gep2155, align 8, !dbg !13, !frisk.k_lds_pair !34
  %350 = load <4 x half>, ptr addrspace(3) %kdma.gep2152, align 8, !dbg !13, !frisk.k_lds_pair !35
  %351 = load <4 x half>, ptr addrspace(3) %kdma.gep2156, align 8, !dbg !13, !frisk.k_lds_pair !35
  %352 = load <4 x half>, ptr addrspace(3) %kdma.gep2157, align 8, !dbg !13, !frisk.k_lds_pair !36
  %353 = load <4 x half>, ptr addrspace(3) %kdma.gep2161, align 8, !dbg !13, !frisk.k_lds_pair !36
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %354 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %344, <4 x float> %290) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %355 = load <4 x half>, ptr addrspace(3) %kdma.gep2158, align 8, !dbg !13, !frisk.k_lds_pair !37
  %356 = load <4 x half>, ptr addrspace(3) %kdma.gep2162, align 8, !dbg !13, !frisk.k_lds_pair !37
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %357 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %346, <4 x float> %354) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %358 = load <4 x half>, ptr addrspace(3) %kdma.gep2159, align 8, !dbg !13, !frisk.k_lds_pair !38
  %359 = load <4 x half>, ptr addrspace(3) %kdma.gep2163, align 8, !dbg !13, !frisk.k_lds_pair !38
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %360 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %348, <4 x float> %296) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %361 = load <4 x half>, ptr addrspace(3) %kdma.gep2160, align 8, !dbg !13, !frisk.k_lds_pair !39
  %362 = load <4 x half>, ptr addrspace(3) %kdma.gep2164, align 8, !dbg !13, !frisk.k_lds_pair !39
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %363 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %350, <4 x float> %360) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %364 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %345, <4 x float> %298) #8, !dbg !13
  %365 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %347, <4 x float> %364) #8, !dbg !13
  %366 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %349, <4 x float> %300) #8, !dbg !13
  %367 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %351, <4 x float> %366) #8, !dbg !13
  %368 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %352, <4 x float> %302) #8, !dbg !13
  %369 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %355, <4 x float> %368) #8, !dbg !13
  %370 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %358, <4 x float> %304) #8, !dbg !13
  %371 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %361, <4 x float> %370) #8, !dbg !13
  %372 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %353, <4 x float> %306) #8, !dbg !13
  %373 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %356, <4 x float> %372) #8, !dbg !13
  %374 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %359, <4 x float> %308) #8, !dbg !13
  %375 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %362, <4 x float> %374) #8, !dbg !13
  %376 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %344, <4 x float> %310) #8, !dbg !13
  %377 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %346, <4 x float> %376) #8, !dbg !13
  %378 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %348, <4 x float> %312) #8, !dbg !13
  %379 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %350, <4 x float> %378) #8, !dbg !13
  %380 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %345, <4 x float> %314) #8, !dbg !13
  %381 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %347, <4 x float> %380) #8, !dbg !13
  %382 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %349, <4 x float> %316) #8, !dbg !13
  %383 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %351, <4 x float> %382) #8, !dbg !13
  %384 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %352, <4 x float> %318) #8, !dbg !13
  %385 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %355, <4 x float> %384) #8, !dbg !13
  %386 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %358, <4 x float> %320) #8, !dbg !13
  %387 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %361, <4 x float> %386) #8, !dbg !13
  %388 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %353, <4 x float> %322) #8, !dbg !13
  %389 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %356, <4 x float> %388) #8, !dbg !13
  %390 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %359, <4 x float> %324) #8, !dbg !13
  %391 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %362, <4 x float> %390) #8, !dbg !13
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  tail call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %392 = add nuw nsw i64 %136, %17, !dbg !40
  %393 = shl i64 %392, 19, !dbg !40
  %394 = and i64 %393, 16252928, !dbg !40
  %address.local985 = shl nuw nsw i64 %address.local567, 5, !dbg !40
  %address.local986 = and i64 %address.local985, 8064, !dbg !40
  %address.local988 = and i64 %address.local568, 24, !dbg !40
  %address.local989 = or disjoint i64 %address.local986, %address.local988, !dbg !40
  %.idx17771779 = add i64 %394, %143, !dbg !40
  %395 = or disjoint i64 %address.local989, %.idx17771779, !dbg !40
  %.tr1780 = trunc i64 %395 to i32, !dbg !40
  %dma.buffer.offset1676 = shl i32 %.tr1780, 1, !dbg !40
  %396 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1702, i32 %dma.buffer.offset1676, <4 x i32> %dma.buffer.rsrc1677) #9, !dbg !40
  %address.local991 = shl nuw nsw i64 %address.local584, 5, !dbg !40
  %address.local992 = and i64 %address.local991, 16256, !dbg !40
  %address.local995 = or disjoint i64 %address.local992, %address.local988, !dbg !40
  %397 = or disjoint i64 %address.local995, %.idx17771779, !dbg !40
  %.tr1784 = trunc i64 %397 to i32, !dbg !40
  %dma.buffer.offset1673 = shl i32 %.tr1784, 1, !dbg !40
  %398 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1699, i32 %dma.buffer.offset1673, <4 x i32> %dma.buffer.rsrc1677) #9, !dbg !40
  %399 = load <4 x half>, ptr addrspace(3) %kdma.gep2181, align 8, !dbg !13
  %400 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %399, <4 x float> %357) #8, !dbg !13
  %401 = or disjoint i64 %138, %111, !dbg !14
  %402 = lshr i64 %133, 25, !dbg !16
  %compare.high1269 = trunc nuw nsw i64 %402 to i32, !dbg !16
  %compare.low.base1270 = trunc i64 %401 to i32, !dbg !16
  %compare.local.base = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 1) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1273 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 2) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1277 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 3) #8, !dbg !16, !frisk.register_lifetimes !5
  %403 = insertelement <4 x i32> poison, i32 %compare.low.base1270, i64 0, !dbg !16
  %404 = insertelement <4 x i32> %403, i32 %compare.local.base, i64 1, !dbg !16
  %405 = insertelement <4 x i32> %404, i32 %compare.local.base1273, i64 2, !dbg !16
  %406 = insertelement <4 x i32> %405, i32 %compare.local.base1277, i64 3, !dbg !16
  %407 = or <4 x i32> %406, <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %408 = icmp ugt <4 x i32> %125, %407, !dbg !16
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
  %409 = insertelement <4 x i32> poison, i32 %compare.local.base1361, i64 0, !dbg !16
  %410 = insertelement <4 x i32> %409, i32 %compare.local.base1365, i64 1, !dbg !16
  %411 = insertelement <4 x i32> %410, i32 %compare.local.base1369, i64 2, !dbg !16
  %412 = insertelement <4 x i32> %411, i32 %compare.local.base1373, i64 3, !dbg !16
  %413 = or <4 x i32> %412, <i32 72, i32 73, i32 74, i32 75>, !dbg !16
  %414 = icmp ult <4 x i32> %413, %125, !dbg !16
  %415 = insertelement <4 x i32> poison, i32 %compare.local.base1345, i64 0, !dbg !16
  %416 = insertelement <4 x i32> %415, i32 %compare.local.base1349, i64 1, !dbg !16
  %417 = insertelement <4 x i32> %416, i32 %compare.local.base1353, i64 2, !dbg !16
  %418 = insertelement <4 x i32> %417, i32 %compare.local.base1357, i64 3, !dbg !16
  %419 = or <4 x i32> %418, <i32 68, i32 69, i32 70, i32 71>, !dbg !16
  %420 = icmp ult <4 x i32> %419, %125, !dbg !16
  %421 = insertelement <4 x i32> poison, i32 %compare.local.base1329, i64 0, !dbg !16
  %422 = insertelement <4 x i32> %421, i32 %compare.local.base1333, i64 1, !dbg !16
  %423 = insertelement <4 x i32> %422, i32 %compare.local.base1337, i64 2, !dbg !16
  %424 = insertelement <4 x i32> %423, i32 %compare.local.base1341, i64 3, !dbg !16
  %425 = or <4 x i32> %424, <i32 64, i32 65, i32 66, i32 67>, !dbg !16
  %426 = icmp ult <4 x i32> %425, %125, !dbg !16
  %427 = insertelement <4 x i32> poison, i32 %compare.local.base1313, i64 0, !dbg !16
  %428 = insertelement <4 x i32> %427, i32 %compare.local.base1317, i64 1, !dbg !16
  %429 = insertelement <4 x i32> %428, i32 %compare.local.base1321, i64 2, !dbg !16
  %430 = insertelement <4 x i32> %429, i32 %compare.local.base1325, i64 3, !dbg !16
  %431 = or <4 x i32> %430, <i32 12, i32 13, i32 14, i32 15>, !dbg !16
  %432 = icmp ult <4 x i32> %431, %125, !dbg !16
  %433 = insertelement <4 x i32> poison, i32 %compare.local.base1297, i64 0, !dbg !16
  %434 = insertelement <4 x i32> %433, i32 %compare.local.base1301, i64 1, !dbg !16
  %435 = insertelement <4 x i32> %434, i32 %compare.local.base1305, i64 2, !dbg !16
  %436 = insertelement <4 x i32> %435, i32 %compare.local.base1309, i64 3, !dbg !16
  %437 = or <4 x i32> %436, <i32 8, i32 9, i32 10, i32 11>, !dbg !16
  %438 = icmp ult <4 x i32> %437, %125, !dbg !16
  %439 = insertelement <4 x i32> poison, i32 %compare.local.base1281, i64 0, !dbg !16
  %440 = insertelement <4 x i32> %439, i32 %compare.local.base1285, i64 1, !dbg !16
  %441 = insertelement <4 x i32> %440, i32 %compare.local.base1289, i64 2, !dbg !16
  %442 = insertelement <4 x i32> %441, i32 %compare.local.base1293, i64 3, !dbg !16
  %443 = or <4 x i32> %442, <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %444 = icmp ult <4 x i32> %443, %125, !dbg !16
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high1269, !dbg !16
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high1269, !dbg !16
  %445 = insertelement <4 x i1> poison, i1 %compare.high.equal, i64 0, !dbg !16
  %446 = shufflevector <4 x i1> %445, <4 x i1> poison, <4 x i32> zeroinitializer, !dbg !16
  %447 = insertelement <4 x i1> poison, i1 %compare.high.order, i64 0, !dbg !16
  %448 = shufflevector <4 x i1> %447, <4 x i1> poison, <4 x i32> zeroinitializer, !dbg !16
  %449 = select <4 x i1> %446, <4 x i1> %408, <4 x i1> %448, !dbg !16
  %450 = select <4 x i1> %449, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !41
  %451 = select <4 x i1> %446, <4 x i1> %444, <4 x i1> %448, !dbg !16
  %452 = select <4 x i1> %451, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !41
  %453 = select <4 x i1> %446, <4 x i1> %438, <4 x i1> %448, !dbg !16
  %454 = select <4 x i1> %453, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !41
  %455 = select <4 x i1> %446, <4 x i1> %432, <4 x i1> %448, !dbg !16
  %456 = select <4 x i1> %455, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !41
  %457 = select <4 x i1> %446, <4 x i1> %426, <4 x i1> %448, !dbg !16
  %458 = select <4 x i1> %457, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !41
  %459 = select <4 x i1> %446, <4 x i1> %420, <4 x i1> %448, !dbg !16
  %460 = select <4 x i1> %459, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !41
  %461 = select <4 x i1> %446, <4 x i1> %414, <4 x i1> %448, !dbg !16
  %462 = select <4 x i1> %461, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !41
  %463 = insertelement <4 x i32> poison, i32 %compare.local.base1377, i64 0, !dbg !16
  %464 = insertelement <4 x i32> %463, i32 %compare.local.base1381, i64 1, !dbg !16
  %465 = insertelement <4 x i32> %464, i32 %compare.local.base1385, i64 2, !dbg !16
  %466 = insertelement <4 x i32> %465, i32 %compare.local.base1389, i64 3, !dbg !16
  %467 = or <4 x i32> %466, <i32 76, i32 77, i32 78, i32 79>, !dbg !16
  %468 = icmp ult <4 x i32> %467, %125, !dbg !16
  %469 = select <4 x i1> %446, <4 x i1> %468, <4 x i1> %448, !dbg !16
  %470 = select <4 x i1> %469, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !41
  %compare.local.base1401 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 32) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1405 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 33) #8, !dbg !16, !frisk.register_lifetimes !5
  %compare.local.base1409 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base1270, i32 34) #8, !dbg !16, !frisk.register_lifetimes !5
  %471 = insertelement <4 x i32> %403, i32 %compare.local.base1401, i64 1, !dbg !16
  %472 = insertelement <4 x i32> %471, i32 %compare.local.base1405, i64 2, !dbg !16
  %473 = insertelement <4 x i32> %472, i32 %compare.local.base1409, i64 3, !dbg !16
  %474 = or <4 x i32> %473, <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %475 = icmp ugt <4 x i32> %123, %474, !dbg !16
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
  %476 = insertelement <4 x i32> poison, i32 %compare.local.base1493, i64 0, !dbg !16
  %477 = insertelement <4 x i32> %476, i32 %compare.local.base1497, i64 1, !dbg !16
  %478 = insertelement <4 x i32> %477, i32 %compare.local.base1501, i64 2, !dbg !16
  %479 = insertelement <4 x i32> %478, i32 %compare.local.base1505, i64 3, !dbg !16
  %480 = or <4 x i32> %479, <i32 72, i32 73, i32 74, i32 75>, !dbg !16
  %481 = icmp ult <4 x i32> %480, %123, !dbg !16
  %482 = insertelement <4 x i32> poison, i32 %compare.local.base1477, i64 0, !dbg !16
  %483 = insertelement <4 x i32> %482, i32 %compare.local.base1481, i64 1, !dbg !16
  %484 = insertelement <4 x i32> %483, i32 %compare.local.base1485, i64 2, !dbg !16
  %485 = insertelement <4 x i32> %484, i32 %compare.local.base1489, i64 3, !dbg !16
  %486 = or <4 x i32> %485, <i32 68, i32 69, i32 70, i32 71>, !dbg !16
  %487 = icmp ult <4 x i32> %486, %123, !dbg !16
  %488 = insertelement <4 x i32> poison, i32 %compare.local.base1461, i64 0, !dbg !16
  %489 = insertelement <4 x i32> %488, i32 %compare.local.base1465, i64 1, !dbg !16
  %490 = insertelement <4 x i32> %489, i32 %compare.local.base1469, i64 2, !dbg !16
  %491 = insertelement <4 x i32> %490, i32 %compare.local.base1473, i64 3, !dbg !16
  %492 = or <4 x i32> %491, <i32 64, i32 65, i32 66, i32 67>, !dbg !16
  %493 = icmp ult <4 x i32> %492, %123, !dbg !16
  %494 = insertelement <4 x i32> poison, i32 %compare.local.base1445, i64 0, !dbg !16
  %495 = insertelement <4 x i32> %494, i32 %compare.local.base1449, i64 1, !dbg !16
  %496 = insertelement <4 x i32> %495, i32 %compare.local.base1453, i64 2, !dbg !16
  %497 = insertelement <4 x i32> %496, i32 %compare.local.base1457, i64 3, !dbg !16
  %498 = or <4 x i32> %497, <i32 12, i32 13, i32 14, i32 15>, !dbg !16
  %499 = icmp ult <4 x i32> %498, %123, !dbg !16
  %500 = insertelement <4 x i32> poison, i32 %compare.local.base1429, i64 0, !dbg !16
  %501 = insertelement <4 x i32> %500, i32 %compare.local.base1433, i64 1, !dbg !16
  %502 = insertelement <4 x i32> %501, i32 %compare.local.base1437, i64 2, !dbg !16
  %503 = insertelement <4 x i32> %502, i32 %compare.local.base1441, i64 3, !dbg !16
  %504 = or <4 x i32> %503, <i32 8, i32 9, i32 10, i32 11>, !dbg !16
  %505 = icmp ult <4 x i32> %504, %123, !dbg !16
  %506 = insertelement <4 x i32> poison, i32 %compare.local.base1413, i64 0, !dbg !16
  %507 = insertelement <4 x i32> %506, i32 %compare.local.base1417, i64 1, !dbg !16
  %508 = insertelement <4 x i32> %507, i32 %compare.local.base1421, i64 2, !dbg !16
  %509 = insertelement <4 x i32> %508, i32 %compare.local.base1425, i64 3, !dbg !16
  %510 = or <4 x i32> %509, <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %511 = icmp ult <4 x i32> %510, %123, !dbg !16
  %compare.high.equal1395 = icmp eq i32 %compare.high1393, %compare.high1269, !dbg !16
  %compare.high.order1396 = icmp ugt i32 %compare.high1393, %compare.high1269, !dbg !16
  %512 = insertelement <4 x i1> poison, i1 %compare.high.equal1395, i64 0, !dbg !16
  %513 = shufflevector <4 x i1> %512, <4 x i1> poison, <4 x i32> zeroinitializer, !dbg !16
  %514 = insertelement <4 x i1> poison, i1 %compare.high.order1396, i64 0, !dbg !16
  %515 = shufflevector <4 x i1> %514, <4 x i1> poison, <4 x i32> zeroinitializer, !dbg !16
  %516 = select <4 x i1> %513, <4 x i1> %475, <4 x i1> %515, !dbg !16
  %517 = select <4 x i1> %516, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !41
  %518 = select <4 x i1> %513, <4 x i1> %511, <4 x i1> %515, !dbg !16
  %519 = select <4 x i1> %518, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !41
  %520 = select <4 x i1> %513, <4 x i1> %505, <4 x i1> %515, !dbg !16
  %521 = select <4 x i1> %520, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !41
  %522 = select <4 x i1> %513, <4 x i1> %499, <4 x i1> %515, !dbg !16
  %523 = select <4 x i1> %522, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !41
  %524 = select <4 x i1> %513, <4 x i1> %493, <4 x i1> %515, !dbg !16
  %525 = select <4 x i1> %524, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !41
  %526 = select <4 x i1> %513, <4 x i1> %487, <4 x i1> %515, !dbg !16
  %527 = select <4 x i1> %526, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !41
  %528 = select <4 x i1> %513, <4 x i1> %481, <4 x i1> %515, !dbg !16
  %529 = select <4 x i1> %528, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !41
  %530 = insertelement <4 x i32> poison, i32 %compare.local.base1509, i64 0, !dbg !16
  %531 = insertelement <4 x i32> %530, i32 %compare.local.base1513, i64 1, !dbg !16
  %532 = insertelement <4 x i32> %531, i32 %compare.local.base1517, i64 2, !dbg !16
  %533 = insertelement <4 x i32> %532, i32 %compare.local.base1521, i64 3, !dbg !16
  %534 = or <4 x i32> %533, <i32 76, i32 77, i32 78, i32 79>, !dbg !16
  %535 = icmp ult <4 x i32> %534, %123, !dbg !16
  %536 = select <4 x i1> %513, <4 x i1> %535, <4 x i1> %515, !dbg !16
  %537 = select <4 x i1> %536, <4 x float> zeroinitializer, <4 x float> splat (float -inf), !dbg !41
  %538 = load <4 x half>, ptr addrspace(3) %kdma.gep2182, align 8, !dbg !13, !frisk.k_tail_pair !17
  %539 = load <4 x half>, ptr addrspace(3) %kdma.gep2186, align 8, !dbg !13, !frisk.k_tail_pair !17
  %540 = load <4 x half>, ptr addrspace(3) %kdma.gep2183, align 8, !dbg !13, !frisk.k_tail_pair !18
  %541 = load <4 x half>, ptr addrspace(3) %kdma.gep2187, align 8, !dbg !13, !frisk.k_tail_pair !18
  %542 = load <4 x half>, ptr addrspace(3) %kdma.gep2184, align 8, !dbg !13, !frisk.k_tail_pair !19
  %543 = load <4 x half>, ptr addrspace(3) %kdma.gep2188, align 8, !dbg !13, !frisk.k_tail_pair !19
  %544 = load <4 x half>, ptr addrspace(3) %kdma.gep2185, align 8, !dbg !13, !frisk.k_tail_pair !20
  %545 = load <4 x half>, ptr addrspace(3) %kdma.gep2189, align 8, !dbg !13, !frisk.k_tail_pair !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_tail_prefetch !5
  %546 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %538, <4 x float> %400) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_tail_prefetch !5
  %547 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %540, <4 x float> %363) #8, !dbg !13
  %548 = fadd <4 x float> %450, %546, !dbg !42
  %549 = extractelement <4 x float> %548, i64 0, !dbg !42
  %550 = tail call float @llvm.maxnum.f32(float %549, float -inf), !dbg !43, !frisk.local_reduction !5
  %551 = extractelement <4 x float> %548, i64 1, !dbg !42
  %552 = extractelement <4 x float> %548, i64 2, !dbg !42
  %553 = extractelement <4 x float> %548, i64 3, !dbg !42
  %554 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %542, <4 x float> %547) #8, !dbg !13
  %555 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %544, <4 x float> %365) #8, !dbg !13
  %556 = fadd <4 x float> %452, %554, !dbg !42
  %557 = extractelement <4 x float> %556, i64 0, !dbg !42
  %local.reduce219 = tail call float @llvm.maxnum.f32(float %550, float %557), !dbg !43, !frisk.local_reduction !5
  %558 = extractelement <4 x float> %556, i64 1, !dbg !42
  %local.reduce220 = tail call float @llvm.maxnum.f32(float %551, float %558), !dbg !43, !frisk.local_reduction !5
  %559 = extractelement <4 x float> %556, i64 2, !dbg !42
  %local.reduce221 = tail call float @llvm.maxnum.f32(float %552, float %559), !dbg !43, !frisk.local_reduction !5
  %560 = extractelement <4 x float> %556, i64 3, !dbg !42
  %local.reduce222 = tail call float @llvm.maxnum.f32(float %553, float %560), !dbg !43, !frisk.local_reduction !5
  %561 = load <4 x half>, ptr addrspace(3) %kdma.gep2190, align 8, !dbg !13, !frisk.k_tail_pair !21
  %562 = load <4 x half>, ptr addrspace(3) %kdma.gep2194, align 8, !dbg !13, !frisk.k_tail_pair !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_tail_prefetch !5
  %563 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %539, <4 x float> %555) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_tail_prefetch !5
  %564 = load <4 x half>, ptr addrspace(3) %kdma.gep2191, align 8, !dbg !13, !frisk.k_tail_pair !22
  %565 = load <4 x half>, ptr addrspace(3) %kdma.gep2195, align 8, !dbg !13, !frisk.k_tail_pair !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_tail_prefetch !5
  %566 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %541, <4 x float> %367) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.k_tail_prefetch !5
  %567 = fadd <4 x float> %454, %563, !dbg !42
  %568 = extractelement <4 x float> %567, i64 0, !dbg !42
  %local.reduce223 = tail call float @llvm.maxnum.f32(float %local.reduce219, float %568), !dbg !43, !frisk.local_reduction !5
  %569 = extractelement <4 x float> %567, i64 1, !dbg !42
  %local.reduce224 = tail call float @llvm.maxnum.f32(float %local.reduce220, float %569), !dbg !43, !frisk.local_reduction !5
  %570 = extractelement <4 x float> %567, i64 2, !dbg !42
  %local.reduce225 = tail call float @llvm.maxnum.f32(float %local.reduce221, float %570), !dbg !43, !frisk.local_reduction !5
  %571 = extractelement <4 x float> %567, i64 3, !dbg !42
  %local.reduce226 = tail call float @llvm.maxnum.f32(float %local.reduce222, float %571), !dbg !43, !frisk.local_reduction !5
  %572 = load <4 x half>, ptr addrspace(3) %kdma.gep2192, align 8, !dbg !13, !frisk.k_tail_pair !23
  %573 = load <4 x half>, ptr addrspace(3) %kdma.gep2196, align 8, !dbg !13, !frisk.k_tail_pair !23
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_tail_prefetch !5
  %574 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %543, <4 x float> %566) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_tail_prefetch !5
  %575 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %545, <4 x float> %369) #8, !dbg !13
  %576 = fadd <4 x float> %456, %574, !dbg !42
  %577 = extractelement <4 x float> %576, i64 0, !dbg !42
  %local.reduce227 = tail call float @llvm.maxnum.f32(float %local.reduce223, float %577), !dbg !43, !frisk.local_reduction !5
  %578 = extractelement <4 x float> %576, i64 1, !dbg !42
  %local.reduce228 = tail call float @llvm.maxnum.f32(float %local.reduce224, float %578), !dbg !43, !frisk.local_reduction !5
  %579 = extractelement <4 x float> %576, i64 2, !dbg !42
  %local.reduce229 = tail call float @llvm.maxnum.f32(float %local.reduce225, float %579), !dbg !43, !frisk.local_reduction !5
  %580 = extractelement <4 x float> %576, i64 3, !dbg !42
  %local.reduce230 = tail call float @llvm.maxnum.f32(float %local.reduce226, float %580), !dbg !43, !frisk.local_reduction !5
  %581 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %561, <4 x float> %575) #8, !dbg !13
  %582 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %564, <4 x float> %371) #8, !dbg !13
  %583 = fadd <4 x float> %458, %581, !dbg !42
  %584 = extractelement <4 x float> %583, i64 0, !dbg !42
  %local.reduce231 = tail call float @llvm.maxnum.f32(float %local.reduce227, float %584), !dbg !43, !frisk.local_reduction !5
  %585 = extractelement <4 x float> %583, i64 1, !dbg !42
  %local.reduce232 = tail call float @llvm.maxnum.f32(float %local.reduce228, float %585), !dbg !43, !frisk.local_reduction !5
  %586 = extractelement <4 x float> %583, i64 2, !dbg !42
  %local.reduce233 = tail call float @llvm.maxnum.f32(float %local.reduce229, float %586), !dbg !43, !frisk.local_reduction !5
  %587 = extractelement <4 x float> %583, i64 3, !dbg !42
  %local.reduce234 = tail call float @llvm.maxnum.f32(float %local.reduce230, float %587), !dbg !43, !frisk.local_reduction !5
  %588 = load <4 x half>, ptr addrspace(3) %kdma.gep2193, align 8, !dbg !13
  %589 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %572, <4 x float> %582) #8, !dbg !13
  %590 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %588, <4 x float> %373) #8, !dbg !13
  %591 = fadd <4 x float> %460, %589, !dbg !42
  %592 = extractelement <4 x float> %591, i64 0, !dbg !42
  %local.reduce235 = tail call float @llvm.maxnum.f32(float %local.reduce231, float %592), !dbg !43, !frisk.local_reduction !5
  %593 = extractelement <4 x float> %591, i64 1, !dbg !42
  %local.reduce236 = tail call float @llvm.maxnum.f32(float %local.reduce232, float %593), !dbg !43, !frisk.local_reduction !5
  %594 = extractelement <4 x float> %591, i64 2, !dbg !42
  %local.reduce237 = tail call float @llvm.maxnum.f32(float %local.reduce233, float %594), !dbg !43, !frisk.local_reduction !5
  %595 = extractelement <4 x float> %591, i64 3, !dbg !42
  %local.reduce238 = tail call float @llvm.maxnum.f32(float %local.reduce234, float %595), !dbg !43, !frisk.local_reduction !5
  %596 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %562, <4 x float> %590) #8, !dbg !13
  %597 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %565, <4 x float> %375) #8, !dbg !13
  %598 = fadd <4 x float> %462, %596, !dbg !42
  %599 = extractelement <4 x float> %598, i64 0, !dbg !42
  %local.reduce239 = tail call float @llvm.maxnum.f32(float %local.reduce235, float %599), !dbg !43, !frisk.local_reduction !5
  %600 = extractelement <4 x float> %598, i64 1, !dbg !42
  %local.reduce240 = tail call float @llvm.maxnum.f32(float %local.reduce236, float %600), !dbg !43, !frisk.local_reduction !5
  %601 = extractelement <4 x float> %598, i64 2, !dbg !42
  %local.reduce241 = tail call float @llvm.maxnum.f32(float %local.reduce237, float %601), !dbg !43, !frisk.local_reduction !5
  %602 = extractelement <4 x float> %598, i64 3, !dbg !42
  %local.reduce242 = tail call float @llvm.maxnum.f32(float %local.reduce238, float %602), !dbg !43, !frisk.local_reduction !5
  %603 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %573, <4 x float> %597) #8, !dbg !13
  %604 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %399, <4 x float> %377) #8, !dbg !13
  %605 = fadd <4 x float> %470, %603, !dbg !42
  %606 = extractelement <4 x float> %605, i64 0, !dbg !42
  %local.reduce243 = tail call float @llvm.maxnum.f32(float %local.reduce239, float %606), !dbg !43, !frisk.local_reduction !5
  %607 = extractelement <4 x float> %605, i64 1, !dbg !42
  %local.reduce244 = tail call float @llvm.maxnum.f32(float %local.reduce240, float %607), !dbg !43, !frisk.local_reduction !5
  %608 = extractelement <4 x float> %605, i64 2, !dbg !42
  %local.reduce245 = tail call float @llvm.maxnum.f32(float %local.reduce241, float %608), !dbg !43, !frisk.local_reduction !5
  %609 = extractelement <4 x float> %605, i64 3, !dbg !42
  %local.reduce246 = tail call float @llvm.maxnum.f32(float %local.reduce242, float %609), !dbg !43, !frisk.local_reduction !5
  %local.reduce247 = tail call float @llvm.maxnum.f32(float %local.reduce243, float %local.reduce244), !dbg !43, !frisk.local_reduction !5
  %local.reduce248 = tail call float @llvm.maxnum.f32(float %local.reduce245, float %local.reduce246), !dbg !43, !frisk.local_reduction !5
  %610 = tail call float @llvm.maxnum.f32(float %local.reduce247, float %local.reduce248), !dbg !43, !frisk.local_reduction !5
  %611 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %538, <4 x float> %604) #8, !dbg !13
  %612 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %540, <4 x float> %379) #8, !dbg !13
  %613 = fadd <4 x float> %517, %611, !dbg !42
  %614 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %542, <4 x float> %612) #8, !dbg !13
  %615 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %544, <4 x float> %381) #8, !dbg !13
  %616 = fadd <4 x float> %519, %614, !dbg !42
  %617 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %539, <4 x float> %615) #8, !dbg !13
  %618 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %541, <4 x float> %383) #8, !dbg !13
  %619 = fadd <4 x float> %521, %617, !dbg !42
  %620 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %543, <4 x float> %618) #8, !dbg !13
  %621 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %545, <4 x float> %385) #8, !dbg !13
  %622 = fadd <4 x float> %523, %620, !dbg !42
  %623 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %561, <4 x float> %621) #8, !dbg !13
  %624 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %564, <4 x float> %387) #8, !dbg !13
  %625 = fadd <4 x float> %525, %623, !dbg !42
  %626 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %572, <4 x float> %624) #8, !dbg !13
  %627 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %588, <4 x float> %389) #8, !dbg !13
  %628 = fadd <4 x float> %527, %626, !dbg !42
  %629 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %562, <4 x float> %627) #8, !dbg !13
  %630 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %565, <4 x float> %391) #8, !dbg !13
  %631 = fadd <4 x float> %529, %629, !dbg !42
  %632 = extractelement <4 x float> %613, i64 0, !dbg !42
  %633 = extractelement <4 x float> %613, i64 1, !dbg !42
  %634 = extractelement <4 x float> %613, i64 2, !dbg !42
  %635 = extractelement <4 x float> %613, i64 3, !dbg !42
  %636 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %573, <4 x float> %630) #8, !dbg !13
  %637 = extractelement <4 x float> %616, i64 0, !dbg !42
  %638 = extractelement <4 x float> %616, i64 1, !dbg !42
  %639 = extractelement <4 x float> %616, i64 2, !dbg !42
  %640 = extractelement <4 x float> %616, i64 3, !dbg !42
  %641 = extractelement <4 x float> %619, i64 0, !dbg !42
  %642 = extractelement <4 x float> %619, i64 1, !dbg !42
  %643 = extractelement <4 x float> %619, i64 2, !dbg !42
  %644 = extractelement <4 x float> %619, i64 3, !dbg !42
  %645 = fadd <4 x float> %537, %636, !dbg !42
  %646 = extractelement <4 x float> %622, i64 0, !dbg !42
  %647 = extractelement <4 x float> %622, i64 1, !dbg !42
  %648 = extractelement <4 x float> %622, i64 2, !dbg !42
  %649 = extractelement <4 x float> %622, i64 3, !dbg !42
  %650 = extractelement <4 x float> %625, i64 0, !dbg !42
  %651 = extractelement <4 x float> %625, i64 1, !dbg !42
  %652 = extractelement <4 x float> %625, i64 2, !dbg !42
  %653 = extractelement <4 x float> %625, i64 3, !dbg !42
  %654 = extractelement <4 x float> %628, i64 0, !dbg !42
  %655 = extractelement <4 x float> %628, i64 1, !dbg !42
  %656 = extractelement <4 x float> %628, i64 2, !dbg !42
  %657 = extractelement <4 x float> %628, i64 3, !dbg !42
  %658 = extractelement <4 x float> %631, i64 0, !dbg !42
  %659 = extractelement <4 x float> %631, i64 1, !dbg !42
  %660 = extractelement <4 x float> %631, i64 2, !dbg !42
  %661 = extractelement <4 x float> %631, i64 3, !dbg !42
  %662 = extractelement <4 x float> %645, i64 0, !dbg !42
  %663 = extractelement <4 x float> %645, i64 1, !dbg !42
  %664 = extractelement <4 x float> %645, i64 2, !dbg !42
  %665 = extractelement <4 x float> %645, i64 3, !dbg !42
  %666 = tail call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !43
  %667 = tail call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %666), !dbg !43
  %668 = add nuw nsw i32 %667, 64, !dbg !43
  %669 = xor i32 %667, 16, !dbg !43
  %670 = and i32 %668, 192, !dbg !43
  %671 = icmp samesign ult i32 %669, %670, !dbg !43
  %672 = select i1 %671, i32 %669, i32 %667, !dbg !43
  %673 = shl nuw nsw i32 %672, 2, !dbg !43
  %674 = bitcast float %610 to i32, !dbg !43
  %675 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %673, i32 %674), !dbg !43
  %676 = bitcast i32 %675 to float, !dbg !43
  %677 = tail call float @llvm.maxnum.f32(float %610, float %676), !dbg !43
  %678 = xor i32 %667, 32, !dbg !43
  %679 = icmp samesign ult i32 %678, %670, !dbg !43
  %680 = select i1 %679, i32 %678, i32 %667, !dbg !43
  %681 = shl nuw nsw i32 %680, 2, !dbg !43
  %682 = bitcast float %677 to i32, !dbg !43
  %683 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %681, i32 %682), !dbg !43
  %684 = bitcast i32 %683 to float, !dbg !43
  %685 = tail call float @llvm.maxnum.f32(float %677, float %684), !dbg !43
  %686 = extractvalue [2 x <1 x float>] %130, 0, !dbg !43
  %687 = insertelement <1 x float> poison, float %685, i64 0, !dbg !43
  %688 = tail call <1 x float> @llvm.maxnum.v1f32(<1 x float> %687, <1 x float> %686), !dbg !43
  %689 = tail call float @llvm.maxnum.f32(float %632, float -inf), !dbg !43, !frisk.local_reduction !5
  %local.reduce188 = tail call float @llvm.maxnum.f32(float %689, float %637), !dbg !43, !frisk.local_reduction !5
  %local.reduce189 = tail call float @llvm.maxnum.f32(float %633, float %638), !dbg !43, !frisk.local_reduction !5
  %local.reduce190 = tail call float @llvm.maxnum.f32(float %634, float %639), !dbg !43, !frisk.local_reduction !5
  %local.reduce191 = tail call float @llvm.maxnum.f32(float %635, float %640), !dbg !43, !frisk.local_reduction !5
  %local.reduce192 = tail call float @llvm.maxnum.f32(float %local.reduce188, float %641), !dbg !43, !frisk.local_reduction !5
  %local.reduce193 = tail call float @llvm.maxnum.f32(float %local.reduce189, float %642), !dbg !43, !frisk.local_reduction !5
  %local.reduce194 = tail call float @llvm.maxnum.f32(float %local.reduce190, float %643), !dbg !43, !frisk.local_reduction !5
  %local.reduce195 = tail call float @llvm.maxnum.f32(float %local.reduce191, float %644), !dbg !43, !frisk.local_reduction !5
  %local.reduce196 = tail call float @llvm.maxnum.f32(float %local.reduce192, float %646), !dbg !43, !frisk.local_reduction !5
  %local.reduce197 = tail call float @llvm.maxnum.f32(float %local.reduce193, float %647), !dbg !43, !frisk.local_reduction !5
  %local.reduce198 = tail call float @llvm.maxnum.f32(float %local.reduce194, float %648), !dbg !43, !frisk.local_reduction !5
  %local.reduce199 = tail call float @llvm.maxnum.f32(float %local.reduce195, float %649), !dbg !43, !frisk.local_reduction !5
  %local.reduce200 = tail call float @llvm.maxnum.f32(float %local.reduce196, float %650), !dbg !43, !frisk.local_reduction !5
  %local.reduce201 = tail call float @llvm.maxnum.f32(float %local.reduce197, float %651), !dbg !43, !frisk.local_reduction !5
  %local.reduce202 = tail call float @llvm.maxnum.f32(float %local.reduce198, float %652), !dbg !43, !frisk.local_reduction !5
  %local.reduce203 = tail call float @llvm.maxnum.f32(float %local.reduce199, float %653), !dbg !43, !frisk.local_reduction !5
  %local.reduce204 = tail call float @llvm.maxnum.f32(float %local.reduce200, float %654), !dbg !43, !frisk.local_reduction !5
  %local.reduce205 = tail call float @llvm.maxnum.f32(float %local.reduce201, float %655), !dbg !43, !frisk.local_reduction !5
  %local.reduce206 = tail call float @llvm.maxnum.f32(float %local.reduce202, float %656), !dbg !43, !frisk.local_reduction !5
  %local.reduce207 = tail call float @llvm.maxnum.f32(float %local.reduce203, float %657), !dbg !43, !frisk.local_reduction !5
  %local.reduce208 = tail call float @llvm.maxnum.f32(float %local.reduce204, float %658), !dbg !43, !frisk.local_reduction !5
  %local.reduce209 = tail call float @llvm.maxnum.f32(float %local.reduce205, float %659), !dbg !43, !frisk.local_reduction !5
  %local.reduce210 = tail call float @llvm.maxnum.f32(float %local.reduce206, float %660), !dbg !43, !frisk.local_reduction !5
  %local.reduce211 = tail call float @llvm.maxnum.f32(float %local.reduce207, float %661), !dbg !43, !frisk.local_reduction !5
  %local.reduce212 = tail call float @llvm.maxnum.f32(float %local.reduce208, float %662), !dbg !43, !frisk.local_reduction !5
  %local.reduce213 = tail call float @llvm.maxnum.f32(float %local.reduce209, float %663), !dbg !43, !frisk.local_reduction !5
  %local.reduce214 = tail call float @llvm.maxnum.f32(float %local.reduce210, float %664), !dbg !43, !frisk.local_reduction !5
  %local.reduce215 = tail call float @llvm.maxnum.f32(float %local.reduce211, float %665), !dbg !43, !frisk.local_reduction !5
  %local.reduce216 = tail call float @llvm.maxnum.f32(float %local.reduce212, float %local.reduce213), !dbg !43, !frisk.local_reduction !5
  %local.reduce217 = tail call float @llvm.maxnum.f32(float %local.reduce214, float %local.reduce215), !dbg !43, !frisk.local_reduction !5
  %690 = tail call float @llvm.maxnum.f32(float %local.reduce216, float %local.reduce217), !dbg !43, !frisk.local_reduction !5
  %691 = bitcast float %690 to i32, !dbg !43
  %692 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %673, i32 %691), !dbg !43
  %693 = bitcast i32 %692 to float, !dbg !43
  %694 = tail call float @llvm.maxnum.f32(float %690, float %693), !dbg !43
  %695 = bitcast float %694 to i32, !dbg !43
  %696 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %681, i32 %695), !dbg !43
  %697 = bitcast i32 %696 to float, !dbg !43
  %698 = tail call float @llvm.maxnum.f32(float %694, float %697), !dbg !43
  %699 = extractvalue [2 x <1 x float>] %130, 1, !dbg !43
  %700 = insertelement <1 x float> poison, float %698, i64 0, !dbg !43
  %701 = tail call <1 x float> @llvm.maxnum.v1f32(<1 x float> %700, <1 x float> %699), !dbg !43
  %702 = shufflevector <1 x float> %688, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !44
  %703 = shufflevector <1 x float> %701, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !44
  %704 = fsub <1 x float> %686, %688, !dbg !45
  %705 = extractelement <1 x float> %704, i64 0, !dbg !46
  %706 = tail call float @llvm.amdgcn.exp2.f32(float %705), !dbg !46
  %707 = insertelement <4 x float> poison, float %706, i64 0, !dbg !47
  %708 = shufflevector <4 x float> %707, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !47
  %709 = extractvalue [2 x <1 x float>] %131, 0, !dbg !48
  %710 = extractelement <1 x float> %709, i64 0, !dbg !48
  %711 = fmul float %710, %706, !dbg !48
  %712 = fsub <1 x float> %699, %701, !dbg !45
  %713 = extractelement <1 x float> %712, i64 0, !dbg !46
  %714 = tail call float @llvm.amdgcn.exp2.f32(float %713), !dbg !46
  %715 = insertelement <4 x float> poison, float %714, i64 0, !dbg !47
  %716 = shufflevector <4 x float> %715, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !47
  %717 = extractvalue [2 x <1 x float>] %131, 1, !dbg !48
  %718 = extractelement <1 x float> %717, i64 0, !dbg !48
  %719 = fmul float %718, %714, !dbg !48
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
  %720 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1805, <4 x i32> %dma.buffer.rsrc1677) #9, !dbg !40
  %vdb.dma.base1820 = add i32 %dma.buffer.lds1699, 8192, !dbg !49
  %721 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1820, i32 %vdb.address1815, <4 x i32> %dma.buffer.rsrc1677) #9, !dbg !40
  %address.local997 = shl nuw nsw i32 %address.thread566, 3, !dbg !49
  %address.local998 = and i32 %address.local997, 120, !dbg !49
  %address.local1000 = and i32 %address.local643, 1536, !dbg !49
  %address.local1001 = or disjoint i32 %address.local998, %address.local1000, !dbg !49
  %address.local1002 = zext nneg i32 %address.local1001 to i64, !dbg !49
  %address.local1003 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1002, !dbg !49
  %722 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1003, i32 0) #11, !dbg !49, !frisk.v_double_buffer !50
  %723 = bitcast <4 x i32> %722 to <8 x half>, !dbg !49
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
  %724 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1013, i32 0) #11, !dbg !49, !frisk.v_double_buffer !50
  %725 = bitcast <4 x i32> %724 to <8 x half>, !dbg !49
  %address.local1017 = getelementptr i8, ptr addrspace(3) %address.local1012, i64 512, !dbg !49
  %726 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1017, i32 0) #11, !dbg !49, !frisk.v_double_buffer !50
  %727 = bitcast <4 x i32> %726 to <8 x half>, !dbg !49
  %address.local1021 = getelementptr i8, ptr addrspace(3) %address.local1012, i64 768, !dbg !49
  %728 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1021, i32 0) #11, !dbg !49, !frisk.v_double_buffer !50
  %729 = bitcast <4 x i32> %728 to <8 x half>, !dbg !49
  %address.local1023 = getelementptr i8, ptr addrspace(3) %address.local1003, i64 4096, !dbg !49
  %730 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1023, i32 0) #11, !dbg !49, !frisk.v_double_buffer !50
  %731 = bitcast <4 x i32> %730 to <8 x half>, !dbg !49
  %address.local1027 = getelementptr i8, ptr addrspace(3) %address.local1012, i64 4352, !dbg !49
  %732 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1027, i32 0) #11, !dbg !49, !frisk.v_double_buffer !50
  %733 = bitcast <4 x i32> %732 to <8 x half>, !dbg !49
  %address.local1031 = getelementptr i8, ptr addrspace(3) %address.local1012, i64 4608, !dbg !49
  %734 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1031, i32 0) #11, !dbg !49, !frisk.v_double_buffer !50
  %735 = bitcast <4 x i32> %734 to <8 x half>, !dbg !49
  %address.local1035 = getelementptr i8, ptr addrspace(3) %address.local1012, i64 4864, !dbg !49
  %736 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1035, i32 0) #11, !dbg !49, !frisk.v_double_buffer !50
  %737 = shufflevector <8 x half> %723, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  %738 = extractvalue [2 x <8 x float>] %132, 0, !dbg !47
  %739 = shufflevector <8 x float> %738, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !47
  %740 = fmul <4 x float> %739, %708, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %741 = fsub <4 x float> %548, %702, !dbg !44
  %742 = extractelement <4 x float> %741, i64 0, !dbg !51
  %743 = tail call float @llvm.amdgcn.exp2.f32(float %742), !dbg !51
  %744 = insertelement <4 x float> poison, float %743, i64 0, !dbg !52
  %745 = extractelement <4 x float> %741, i64 1, !dbg !51
  %746 = tail call float @llvm.amdgcn.exp2.f32(float %745), !dbg !51
  %747 = insertelement <4 x float> %744, float %746, i64 1, !dbg !52
  %748 = extractelement <4 x float> %741, i64 2, !dbg !51
  %749 = tail call float @llvm.amdgcn.exp2.f32(float %748), !dbg !51
  %750 = insertelement <4 x float> %747, float %749, i64 2, !dbg !52
  %751 = extractelement <4 x float> %741, i64 3, !dbg !51
  %752 = tail call float @llvm.amdgcn.exp2.f32(float %751), !dbg !51
  %753 = insertelement <4 x float> %750, float %752, i64 3, !dbg !52
  %math.result = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %753) #8, !dbg !52, !frisk.register_lifetimes !5
  %754 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !52
  %math.packed = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %754) #8, !dbg !51, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49, !frisk.softmax_fragments !5
  %755 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %737, <4 x float> %740) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %756 = shufflevector <8 x half> %723, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  %757 = shufflevector <8 x float> %738, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !47
  %758 = fmul <4 x float> %757, %708, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %759 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %756, <4 x float> %758) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %760 = extractvalue [2 x <8 x float>] %132, 1, !dbg !47
  %761 = shufflevector <8 x float> %760, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !47
  %762 = fmul <4 x float> %761, %716, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %763 = fsub <4 x float> %613, %703, !dbg !44
  %764 = extractelement <4 x float> %763, i64 0, !dbg !51
  %765 = tail call float @llvm.amdgcn.exp2.f32(float %764), !dbg !51
  %766 = insertelement <4 x float> poison, float %765, i64 0, !dbg !52
  %767 = extractelement <4 x float> %763, i64 1, !dbg !51
  %768 = tail call float @llvm.amdgcn.exp2.f32(float %767), !dbg !51
  %769 = insertelement <4 x float> %766, float %768, i64 1, !dbg !52
  %770 = extractelement <4 x float> %763, i64 2, !dbg !51
  %771 = tail call float @llvm.amdgcn.exp2.f32(float %770), !dbg !51
  %772 = insertelement <4 x float> %769, float %771, i64 2, !dbg !52
  %773 = extractelement <4 x float> %763, i64 3, !dbg !51
  %774 = tail call float @llvm.amdgcn.exp2.f32(float %773), !dbg !51
  %775 = insertelement <4 x float> %772, float %774, i64 3, !dbg !52
  %math.result456 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %775) #8, !dbg !52, !frisk.register_lifetimes !5
  %776 = fptrunc <4 x float> %math.result456 to <4 x half>, !dbg !52
  %math.packed457 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %776) #8, !dbg !51, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49, !frisk.softmax_fragments !5
  %777 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed457, <4 x half> %737, <4 x float> %762) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %778 = shufflevector <8 x float> %760, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !47
  %779 = fmul <4 x float> %778, %716, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %780 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed457, <4 x half> %756, <4 x float> %779) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %781 = shufflevector <8 x half> %725, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %782 = fsub <4 x float> %556, %702, !dbg !44
  %783 = extractelement <4 x float> %782, i64 0, !dbg !51
  %784 = tail call float @llvm.amdgcn.exp2.f32(float %783), !dbg !51
  %local.reduce157 = fadd float %743, %784, !dbg !53, !frisk.local_reduction !5
  %785 = insertelement <4 x float> poison, float %784, i64 0, !dbg !52
  %786 = extractelement <4 x float> %782, i64 1, !dbg !51
  %787 = tail call float @llvm.amdgcn.exp2.f32(float %786), !dbg !51
  %local.reduce158 = fadd float %746, %787, !dbg !53, !frisk.local_reduction !5
  %788 = insertelement <4 x float> %785, float %787, i64 1, !dbg !52
  %789 = extractelement <4 x float> %782, i64 2, !dbg !51
  %790 = tail call float @llvm.amdgcn.exp2.f32(float %789), !dbg !51
  %local.reduce159 = fadd float %749, %790, !dbg !53, !frisk.local_reduction !5
  %791 = insertelement <4 x float> %788, float %790, i64 2, !dbg !52
  %792 = extractelement <4 x float> %782, i64 3, !dbg !51
  %793 = tail call float @llvm.amdgcn.exp2.f32(float %792), !dbg !51
  %local.reduce160 = fadd float %752, %793, !dbg !53, !frisk.local_reduction !5
  %794 = insertelement <4 x float> %791, float %793, i64 3, !dbg !52
  %math.result442 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %794) #8, !dbg !52, !frisk.register_lifetimes !5
  %795 = fptrunc <4 x float> %math.result442 to <4 x half>, !dbg !52
  %math.packed443 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %795) #8, !dbg !53, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49, !frisk.softmax_fragments !5
  %796 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed443, <4 x half> %781, <4 x float> %755) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %797 = shufflevector <8 x half> %725, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %798 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed443, <4 x half> %797, <4 x float> %759) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %799 = fsub <4 x float> %616, %703, !dbg !44
  %800 = extractelement <4 x float> %799, i64 0, !dbg !51
  %801 = tail call float @llvm.amdgcn.exp2.f32(float %800), !dbg !51
  %local.reduce = fadd float %765, %801, !dbg !53, !frisk.local_reduction !5
  %802 = insertelement <4 x float> poison, float %801, i64 0, !dbg !52
  %803 = extractelement <4 x float> %799, i64 1, !dbg !51
  %804 = tail call float @llvm.amdgcn.exp2.f32(float %803), !dbg !51
  %local.reduce127 = fadd float %768, %804, !dbg !53, !frisk.local_reduction !5
  %805 = insertelement <4 x float> %802, float %804, i64 1, !dbg !52
  %806 = extractelement <4 x float> %799, i64 2, !dbg !51
  %807 = tail call float @llvm.amdgcn.exp2.f32(float %806), !dbg !51
  %local.reduce128 = fadd float %771, %807, !dbg !53, !frisk.local_reduction !5
  %808 = insertelement <4 x float> %805, float %807, i64 2, !dbg !52
  %809 = extractelement <4 x float> %799, i64 3, !dbg !51
  %810 = tail call float @llvm.amdgcn.exp2.f32(float %809), !dbg !51
  %local.reduce129 = fadd float %774, %810, !dbg !53, !frisk.local_reduction !5
  %811 = insertelement <4 x float> %808, float %810, i64 3, !dbg !52
  %math.result458 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %811) #8, !dbg !52, !frisk.register_lifetimes !5
  %812 = fptrunc <4 x float> %math.result458 to <4 x half>, !dbg !52
  %math.packed459 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %812) #8, !dbg !53, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49, !frisk.softmax_fragments !5
  %813 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed459, <4 x half> %781, <4 x float> %777) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %814 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed459, <4 x half> %797, <4 x float> %780) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %815 = shufflevector <8 x half> %727, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %816 = fsub <4 x float> %567, %702, !dbg !44
  %817 = extractelement <4 x float> %816, i64 0, !dbg !51
  %818 = tail call float @llvm.amdgcn.exp2.f32(float %817), !dbg !51
  %local.reduce161 = fadd float %818, %local.reduce157, !dbg !53, !frisk.local_reduction !5
  %819 = insertelement <4 x float> poison, float %818, i64 0, !dbg !52
  %820 = extractelement <4 x float> %816, i64 1, !dbg !51
  %821 = tail call float @llvm.amdgcn.exp2.f32(float %820), !dbg !51
  %local.reduce162 = fadd float %821, %local.reduce158, !dbg !53, !frisk.local_reduction !5
  %822 = insertelement <4 x float> %819, float %821, i64 1, !dbg !52
  %823 = extractelement <4 x float> %816, i64 2, !dbg !51
  %824 = tail call float @llvm.amdgcn.exp2.f32(float %823), !dbg !51
  %local.reduce163 = fadd float %824, %local.reduce159, !dbg !53, !frisk.local_reduction !5
  %825 = insertelement <4 x float> %822, float %824, i64 2, !dbg !52
  %826 = extractelement <4 x float> %816, i64 3, !dbg !51
  %827 = tail call float @llvm.amdgcn.exp2.f32(float %826), !dbg !51
  %local.reduce164 = fadd float %827, %local.reduce160, !dbg !53, !frisk.local_reduction !5
  %828 = insertelement <4 x float> %825, float %827, i64 3, !dbg !52
  %math.result444 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %828) #8, !dbg !52, !frisk.register_lifetimes !5
  %829 = fptrunc <4 x float> %math.result444 to <4 x half>, !dbg !52
  %math.packed445 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %829) #8, !dbg !53, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49, !frisk.softmax_fragments !5
  %830 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed445, <4 x half> %815, <4 x float> %796) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %831 = shufflevector <8 x half> %727, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %832 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed445, <4 x half> %831, <4 x float> %798) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %833 = fsub <4 x float> %619, %703, !dbg !44
  %834 = extractelement <4 x float> %833, i64 0, !dbg !51
  %835 = tail call float @llvm.amdgcn.exp2.f32(float %834), !dbg !51
  %local.reduce130 = fadd float %835, %local.reduce, !dbg !53, !frisk.local_reduction !5
  %836 = insertelement <4 x float> poison, float %835, i64 0, !dbg !52
  %837 = extractelement <4 x float> %833, i64 1, !dbg !51
  %838 = tail call float @llvm.amdgcn.exp2.f32(float %837), !dbg !51
  %local.reduce131 = fadd float %838, %local.reduce127, !dbg !53, !frisk.local_reduction !5
  %839 = insertelement <4 x float> %836, float %838, i64 1, !dbg !52
  %840 = extractelement <4 x float> %833, i64 2, !dbg !51
  %841 = tail call float @llvm.amdgcn.exp2.f32(float %840), !dbg !51
  %local.reduce132 = fadd float %841, %local.reduce128, !dbg !53, !frisk.local_reduction !5
  %842 = insertelement <4 x float> %839, float %841, i64 2, !dbg !52
  %843 = extractelement <4 x float> %833, i64 3, !dbg !51
  %844 = tail call float @llvm.amdgcn.exp2.f32(float %843), !dbg !51
  %local.reduce133 = fadd float %844, %local.reduce129, !dbg !53, !frisk.local_reduction !5
  %845 = insertelement <4 x float> %842, float %844, i64 3, !dbg !52
  %math.result460 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %845) #8, !dbg !52, !frisk.register_lifetimes !5
  %846 = fptrunc <4 x float> %math.result460 to <4 x half>, !dbg !52
  %math.packed461 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %846) #8, !dbg !53, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49, !frisk.softmax_fragments !5
  %847 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed461, <4 x half> %815, <4 x float> %813) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %848 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed461, <4 x half> %831, <4 x float> %814) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %849 = shufflevector <8 x half> %729, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %850 = fsub <4 x float> %576, %702, !dbg !44
  %851 = extractelement <4 x float> %850, i64 0, !dbg !51
  %852 = tail call float @llvm.amdgcn.exp2.f32(float %851), !dbg !51
  %local.reduce165 = fadd float %852, %local.reduce161, !dbg !53, !frisk.local_reduction !5
  %853 = insertelement <4 x float> poison, float %852, i64 0, !dbg !52
  %854 = extractelement <4 x float> %850, i64 1, !dbg !51
  %855 = tail call float @llvm.amdgcn.exp2.f32(float %854), !dbg !51
  %local.reduce166 = fadd float %855, %local.reduce162, !dbg !53, !frisk.local_reduction !5
  %856 = insertelement <4 x float> %853, float %855, i64 1, !dbg !52
  %857 = extractelement <4 x float> %850, i64 2, !dbg !51
  %858 = tail call float @llvm.amdgcn.exp2.f32(float %857), !dbg !51
  %local.reduce167 = fadd float %858, %local.reduce163, !dbg !53, !frisk.local_reduction !5
  %859 = insertelement <4 x float> %856, float %858, i64 2, !dbg !52
  %860 = extractelement <4 x float> %850, i64 3, !dbg !51
  %861 = tail call float @llvm.amdgcn.exp2.f32(float %860), !dbg !51
  %local.reduce168 = fadd float %861, %local.reduce164, !dbg !53, !frisk.local_reduction !5
  %862 = insertelement <4 x float> %859, float %861, i64 3, !dbg !52
  %math.result446 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %862) #8, !dbg !52, !frisk.register_lifetimes !5
  %863 = fptrunc <4 x float> %math.result446 to <4 x half>, !dbg !52
  %math.packed447 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %863) #8, !dbg !53, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49, !frisk.softmax_fragments !5
  %864 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed447, <4 x half> %849, <4 x float> %830) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %865 = shufflevector <8 x half> %729, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %866 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed447, <4 x half> %865, <4 x float> %832) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %867 = fsub <4 x float> %622, %703, !dbg !44
  %868 = extractelement <4 x float> %867, i64 0, !dbg !51
  %869 = tail call float @llvm.amdgcn.exp2.f32(float %868), !dbg !51
  %local.reduce134 = fadd float %869, %local.reduce130, !dbg !53, !frisk.local_reduction !5
  %870 = insertelement <4 x float> poison, float %869, i64 0, !dbg !52
  %871 = extractelement <4 x float> %867, i64 1, !dbg !51
  %872 = tail call float @llvm.amdgcn.exp2.f32(float %871), !dbg !51
  %local.reduce135 = fadd float %872, %local.reduce131, !dbg !53, !frisk.local_reduction !5
  %873 = insertelement <4 x float> %870, float %872, i64 1, !dbg !52
  %874 = extractelement <4 x float> %867, i64 2, !dbg !51
  %875 = tail call float @llvm.amdgcn.exp2.f32(float %874), !dbg !51
  %local.reduce136 = fadd float %875, %local.reduce132, !dbg !53, !frisk.local_reduction !5
  %876 = insertelement <4 x float> %873, float %875, i64 2, !dbg !52
  %877 = extractelement <4 x float> %867, i64 3, !dbg !51
  %878 = tail call float @llvm.amdgcn.exp2.f32(float %877), !dbg !51
  %local.reduce137 = fadd float %878, %local.reduce133, !dbg !53, !frisk.local_reduction !5
  %879 = insertelement <4 x float> %876, float %878, i64 3, !dbg !52
  %math.result462 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %879) #8, !dbg !52, !frisk.register_lifetimes !5
  %880 = fptrunc <4 x float> %math.result462 to <4 x half>, !dbg !52
  %math.packed463 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %880) #8, !dbg !53, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49, !frisk.softmax_fragments !5
  %881 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed463, <4 x half> %849, <4 x float> %847) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %882 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed463, <4 x half> %865, <4 x float> %848) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %883 = shufflevector <8 x half> %731, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %884 = fsub <4 x float> %583, %702, !dbg !44
  %885 = extractelement <4 x float> %884, i64 0, !dbg !51
  %886 = tail call float @llvm.amdgcn.exp2.f32(float %885), !dbg !51
  %local.reduce169 = fadd float %886, %local.reduce165, !dbg !53, !frisk.local_reduction !5
  %887 = insertelement <4 x float> poison, float %886, i64 0, !dbg !52
  %888 = extractelement <4 x float> %884, i64 1, !dbg !51
  %889 = tail call float @llvm.amdgcn.exp2.f32(float %888), !dbg !51
  %local.reduce170 = fadd float %889, %local.reduce166, !dbg !53, !frisk.local_reduction !5
  %890 = insertelement <4 x float> %887, float %889, i64 1, !dbg !52
  %891 = extractelement <4 x float> %884, i64 2, !dbg !51
  %892 = tail call float @llvm.amdgcn.exp2.f32(float %891), !dbg !51
  %local.reduce171 = fadd float %892, %local.reduce167, !dbg !53, !frisk.local_reduction !5
  %893 = insertelement <4 x float> %890, float %892, i64 2, !dbg !52
  %894 = extractelement <4 x float> %884, i64 3, !dbg !51
  %895 = tail call float @llvm.amdgcn.exp2.f32(float %894), !dbg !51
  %local.reduce172 = fadd float %895, %local.reduce168, !dbg !53, !frisk.local_reduction !5
  %896 = insertelement <4 x float> %893, float %895, i64 3, !dbg !52
  %math.result448 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %896) #8, !dbg !52, !frisk.register_lifetimes !5
  %897 = fptrunc <4 x float> %math.result448 to <4 x half>, !dbg !52
  %math.packed449 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %897) #8, !dbg !53, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49, !frisk.softmax_fragments !5
  %898 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed449, <4 x half> %883, <4 x float> %864) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %899 = shufflevector <8 x half> %731, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %900 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed449, <4 x half> %899, <4 x float> %866) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %901 = fsub <4 x float> %625, %703, !dbg !44
  %902 = extractelement <4 x float> %901, i64 0, !dbg !51
  %903 = tail call float @llvm.amdgcn.exp2.f32(float %902), !dbg !51
  %local.reduce138 = fadd float %903, %local.reduce134, !dbg !53, !frisk.local_reduction !5
  %904 = insertelement <4 x float> poison, float %903, i64 0, !dbg !52
  %905 = extractelement <4 x float> %901, i64 1, !dbg !51
  %906 = tail call float @llvm.amdgcn.exp2.f32(float %905), !dbg !51
  %local.reduce139 = fadd float %906, %local.reduce135, !dbg !53, !frisk.local_reduction !5
  %907 = insertelement <4 x float> %904, float %906, i64 1, !dbg !52
  %908 = extractelement <4 x float> %901, i64 2, !dbg !51
  %909 = tail call float @llvm.amdgcn.exp2.f32(float %908), !dbg !51
  %local.reduce140 = fadd float %909, %local.reduce136, !dbg !53, !frisk.local_reduction !5
  %910 = insertelement <4 x float> %907, float %909, i64 2, !dbg !52
  %911 = extractelement <4 x float> %901, i64 3, !dbg !51
  %912 = tail call float @llvm.amdgcn.exp2.f32(float %911), !dbg !51
  %local.reduce141 = fadd float %912, %local.reduce137, !dbg !53, !frisk.local_reduction !5
  %913 = insertelement <4 x float> %910, float %912, i64 3, !dbg !52
  %math.result464 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %913) #8, !dbg !52, !frisk.register_lifetimes !5
  %914 = fptrunc <4 x float> %math.result464 to <4 x half>, !dbg !52
  %math.packed465 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %914) #8, !dbg !53, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49, !frisk.softmax_fragments !5
  %915 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed465, <4 x half> %883, <4 x float> %881) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %916 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed465, <4 x half> %899, <4 x float> %882) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %917 = shufflevector <8 x half> %733, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %918 = fsub <4 x float> %591, %702, !dbg !44
  %919 = extractelement <4 x float> %918, i64 0, !dbg !51
  %920 = tail call float @llvm.amdgcn.exp2.f32(float %919), !dbg !51
  %local.reduce173 = fadd float %920, %local.reduce169, !dbg !53, !frisk.local_reduction !5
  %921 = insertelement <4 x float> poison, float %920, i64 0, !dbg !52
  %922 = extractelement <4 x float> %918, i64 1, !dbg !51
  %923 = tail call float @llvm.amdgcn.exp2.f32(float %922), !dbg !51
  %local.reduce174 = fadd float %923, %local.reduce170, !dbg !53, !frisk.local_reduction !5
  %924 = insertelement <4 x float> %921, float %923, i64 1, !dbg !52
  %925 = extractelement <4 x float> %918, i64 2, !dbg !51
  %926 = tail call float @llvm.amdgcn.exp2.f32(float %925), !dbg !51
  %local.reduce175 = fadd float %926, %local.reduce171, !dbg !53, !frisk.local_reduction !5
  %927 = insertelement <4 x float> %924, float %926, i64 2, !dbg !52
  %928 = extractelement <4 x float> %918, i64 3, !dbg !51
  %929 = tail call float @llvm.amdgcn.exp2.f32(float %928), !dbg !51
  %local.reduce176 = fadd float %929, %local.reduce172, !dbg !53, !frisk.local_reduction !5
  %930 = insertelement <4 x float> %927, float %929, i64 3, !dbg !52
  %math.result450 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %930) #8, !dbg !52, !frisk.register_lifetimes !5
  %931 = fptrunc <4 x float> %math.result450 to <4 x half>, !dbg !52
  %math.packed451 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %931) #8, !dbg !53, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49, !frisk.softmax_fragments !5
  %932 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed451, <4 x half> %917, <4 x float> %898) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %933 = shufflevector <8 x half> %733, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %934 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed451, <4 x half> %933, <4 x float> %900) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %935 = fsub <4 x float> %628, %703, !dbg !44
  %936 = extractelement <4 x float> %935, i64 0, !dbg !51
  %937 = tail call float @llvm.amdgcn.exp2.f32(float %936), !dbg !51
  %local.reduce142 = fadd float %937, %local.reduce138, !dbg !53, !frisk.local_reduction !5
  %938 = insertelement <4 x float> poison, float %937, i64 0, !dbg !52
  %939 = extractelement <4 x float> %935, i64 1, !dbg !51
  %940 = tail call float @llvm.amdgcn.exp2.f32(float %939), !dbg !51
  %local.reduce143 = fadd float %940, %local.reduce139, !dbg !53, !frisk.local_reduction !5
  %941 = insertelement <4 x float> %938, float %940, i64 1, !dbg !52
  %942 = extractelement <4 x float> %935, i64 2, !dbg !51
  %943 = tail call float @llvm.amdgcn.exp2.f32(float %942), !dbg !51
  %local.reduce144 = fadd float %943, %local.reduce140, !dbg !53, !frisk.local_reduction !5
  %944 = insertelement <4 x float> %941, float %943, i64 2, !dbg !52
  %945 = extractelement <4 x float> %935, i64 3, !dbg !51
  %946 = tail call float @llvm.amdgcn.exp2.f32(float %945), !dbg !51
  %local.reduce145 = fadd float %946, %local.reduce141, !dbg !53, !frisk.local_reduction !5
  %947 = insertelement <4 x float> %944, float %946, i64 3, !dbg !52
  %math.result466 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %947) #8, !dbg !52, !frisk.register_lifetimes !5
  %948 = fptrunc <4 x float> %math.result466 to <4 x half>, !dbg !52
  %math.packed467 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %948) #8, !dbg !53, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49, !frisk.softmax_fragments !5
  %949 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed467, <4 x half> %917, <4 x float> %915) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %950 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed467, <4 x half> %933, <4 x float> %916) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %951 = shufflevector <8 x half> %735, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %952 = fsub <4 x float> %598, %702, !dbg !44
  %953 = extractelement <4 x float> %952, i64 0, !dbg !51
  %954 = tail call float @llvm.amdgcn.exp2.f32(float %953), !dbg !51
  %local.reduce177 = fadd float %954, %local.reduce173, !dbg !53, !frisk.local_reduction !5
  %955 = insertelement <4 x float> poison, float %954, i64 0, !dbg !52
  %956 = extractelement <4 x float> %952, i64 1, !dbg !51
  %957 = tail call float @llvm.amdgcn.exp2.f32(float %956), !dbg !51
  %local.reduce178 = fadd float %957, %local.reduce174, !dbg !53, !frisk.local_reduction !5
  %958 = insertelement <4 x float> %955, float %957, i64 1, !dbg !52
  %959 = extractelement <4 x float> %952, i64 2, !dbg !51
  %960 = tail call float @llvm.amdgcn.exp2.f32(float %959), !dbg !51
  %local.reduce179 = fadd float %960, %local.reduce175, !dbg !53, !frisk.local_reduction !5
  %961 = insertelement <4 x float> %958, float %960, i64 2, !dbg !52
  %962 = extractelement <4 x float> %952, i64 3, !dbg !51
  %963 = tail call float @llvm.amdgcn.exp2.f32(float %962), !dbg !51
  %local.reduce180 = fadd float %963, %local.reduce176, !dbg !53, !frisk.local_reduction !5
  %964 = insertelement <4 x float> %961, float %963, i64 3, !dbg !52
  %math.result452 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %964) #8, !dbg !52, !frisk.register_lifetimes !5
  %965 = fptrunc <4 x float> %math.result452 to <4 x half>, !dbg !52
  %math.packed453 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %965) #8, !dbg !53, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49, !frisk.softmax_fragments !5
  %966 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed453, <4 x half> %951, <4 x float> %932) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %967 = shufflevector <8 x half> %735, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %968 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed453, <4 x half> %967, <4 x float> %934) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %969 = fsub <4 x float> %631, %703, !dbg !44
  %970 = extractelement <4 x float> %969, i64 0, !dbg !51
  %971 = tail call float @llvm.amdgcn.exp2.f32(float %970), !dbg !51
  %local.reduce146 = fadd float %971, %local.reduce142, !dbg !53, !frisk.local_reduction !5
  %972 = insertelement <4 x float> poison, float %971, i64 0, !dbg !52
  %973 = extractelement <4 x float> %969, i64 1, !dbg !51
  %974 = tail call float @llvm.amdgcn.exp2.f32(float %973), !dbg !51
  %local.reduce147 = fadd float %974, %local.reduce143, !dbg !53, !frisk.local_reduction !5
  %975 = insertelement <4 x float> %972, float %974, i64 1, !dbg !52
  %976 = extractelement <4 x float> %969, i64 2, !dbg !51
  %977 = tail call float @llvm.amdgcn.exp2.f32(float %976), !dbg !51
  %local.reduce148 = fadd float %977, %local.reduce144, !dbg !53, !frisk.local_reduction !5
  %978 = insertelement <4 x float> %975, float %977, i64 2, !dbg !52
  %979 = extractelement <4 x float> %969, i64 3, !dbg !51
  %980 = tail call float @llvm.amdgcn.exp2.f32(float %979), !dbg !51
  %local.reduce149 = fadd float %980, %local.reduce145, !dbg !53, !frisk.local_reduction !5
  %981 = insertelement <4 x float> %978, float %980, i64 3, !dbg !52
  %math.result468 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %981) #8, !dbg !52, !frisk.register_lifetimes !5
  %982 = fptrunc <4 x float> %math.result468 to <4 x half>, !dbg !52
  %math.packed469 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %982) #8, !dbg !53, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49, !frisk.softmax_fragments !5
  %983 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed469, <4 x half> %951, <4 x float> %949) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %984 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed469, <4 x half> %967, <4 x float> %950) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %985 = bitcast <4 x i32> %736 to <8 x half>, !dbg !49
  %986 = shufflevector <8 x half> %985, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %987 = fsub <4 x float> %605, %702, !dbg !44
  %988 = extractelement <4 x float> %987, i64 0, !dbg !51
  %989 = tail call float @llvm.amdgcn.exp2.f32(float %988), !dbg !51
  %local.reduce181 = fadd float %989, %local.reduce177, !dbg !53, !frisk.local_reduction !5
  %990 = insertelement <4 x float> poison, float %989, i64 0, !dbg !52
  %991 = extractelement <4 x float> %987, i64 1, !dbg !51
  %992 = tail call float @llvm.amdgcn.exp2.f32(float %991), !dbg !51
  %local.reduce182 = fadd float %992, %local.reduce178, !dbg !53, !frisk.local_reduction !5
  %local.reduce185 = fadd float %local.reduce181, %local.reduce182, !dbg !53, !frisk.local_reduction !5
  %993 = insertelement <4 x float> %990, float %992, i64 1, !dbg !52
  %994 = extractelement <4 x float> %987, i64 2, !dbg !51
  %995 = tail call float @llvm.amdgcn.exp2.f32(float %994), !dbg !51
  %local.reduce183 = fadd float %995, %local.reduce179, !dbg !53, !frisk.local_reduction !5
  %996 = insertelement <4 x float> %993, float %995, i64 2, !dbg !52
  %997 = extractelement <4 x float> %987, i64 3, !dbg !51
  %998 = tail call float @llvm.amdgcn.exp2.f32(float %997), !dbg !51
  %local.reduce184 = fadd float %998, %local.reduce180, !dbg !53, !frisk.local_reduction !5
  %local.reduce186 = fadd float %local.reduce183, %local.reduce184, !dbg !53, !frisk.local_reduction !5
  %999 = fadd float %local.reduce185, %local.reduce186, !dbg !53, !frisk.local_reduction !5
  %1000 = fadd float %711, %999, !dbg !53
  %1001 = insertelement <1 x float> poison, float %1000, i64 0, !dbg !11
  %1002 = insertvalue [2 x <1 x float>] poison, <1 x float> %1001, 0, !dbg !11
  %1003 = insertelement <4 x float> %996, float %998, i64 3, !dbg !52
  %math.result454 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1003) #8, !dbg !52, !frisk.register_lifetimes !5
  %1004 = fptrunc <4 x float> %math.result454 to <4 x half>, !dbg !52
  %math.packed455 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1004) #8, !dbg !53, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49, !frisk.softmax_fragments !5
  %1005 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed455, <4 x half> %986, <4 x float> %966) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1006 = shufflevector <8 x half> %985, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1007 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed455, <4 x half> %1006, <4 x float> %968) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1008 = fsub <4 x float> %645, %703, !dbg !44
  %1009 = extractelement <4 x float> %1008, i64 0, !dbg !51
  %1010 = tail call float @llvm.amdgcn.exp2.f32(float %1009), !dbg !51
  %local.reduce150 = fadd float %1010, %local.reduce146, !dbg !53, !frisk.local_reduction !5
  %1011 = insertelement <4 x float> poison, float %1010, i64 0, !dbg !52
  %1012 = extractelement <4 x float> %1008, i64 1, !dbg !51
  %1013 = tail call float @llvm.amdgcn.exp2.f32(float %1012), !dbg !51
  %local.reduce151 = fadd float %1013, %local.reduce147, !dbg !53, !frisk.local_reduction !5
  %local.reduce154 = fadd float %local.reduce150, %local.reduce151, !dbg !53, !frisk.local_reduction !5
  %1014 = insertelement <4 x float> %1011, float %1013, i64 1, !dbg !52
  %1015 = extractelement <4 x float> %1008, i64 2, !dbg !51
  %1016 = tail call float @llvm.amdgcn.exp2.f32(float %1015), !dbg !51
  %local.reduce152 = fadd float %1016, %local.reduce148, !dbg !53, !frisk.local_reduction !5
  %1017 = insertelement <4 x float> %1014, float %1016, i64 2, !dbg !52
  %1018 = extractelement <4 x float> %1008, i64 3, !dbg !51
  %1019 = tail call float @llvm.amdgcn.exp2.f32(float %1018), !dbg !51
  %local.reduce153 = fadd float %1019, %local.reduce149, !dbg !53, !frisk.local_reduction !5
  %local.reduce155 = fadd float %local.reduce152, %local.reduce153, !dbg !53, !frisk.local_reduction !5
  %1020 = fadd float %local.reduce154, %local.reduce155, !dbg !53, !frisk.local_reduction !5
  %1021 = fadd float %719, %1020, !dbg !53
  %1022 = insertelement <1 x float> poison, float %1021, i64 0, !dbg !11
  %1023 = insertvalue [2 x <1 x float>] %1002, <1 x float> %1022, 1, !dbg !11
  %1024 = insertelement <4 x float> %1017, float %1019, i64 3, !dbg !52
  %math.result470 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1024) #8, !dbg !52, !frisk.register_lifetimes !5
  %1025 = fptrunc <4 x float> %math.result470 to <4 x half>, !dbg !52
  %math.packed471 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1025) #8, !dbg !53, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49, !frisk.softmax_fragments !5
  %1026 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed471, <4 x half> %986, <4 x float> %983) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1027 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed471, <4 x half> %1006, <4 x float> %984) #8, !dbg !49, !frisk.pv_interleave !21
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
  %1028 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1702, i32 %vdb.address1833, <4 x i32> %dma.buffer.rsrc1677) #9, !dbg !40
  %1029 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1699, i32 %vdb.address1843, <4 x i32> %dma.buffer.rsrc1677) #9, !dbg !40
  %vdb.read = getelementptr i8, ptr addrspace(3) %address.local1003, i64 8192, !dbg !49
  %1030 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #11, !dbg !49, !frisk.v_double_buffer !50
  %1031 = bitcast <4 x i32> %1030 to <8 x half>, !dbg !49
  %vdb.read1821 = getelementptr i8, ptr addrspace(3) %address.local1012, i64 8448, !dbg !49
  %1032 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1821, i32 0) #11, !dbg !49, !frisk.v_double_buffer !50
  %1033 = bitcast <4 x i32> %1032 to <8 x half>, !dbg !49
  %vdb.read1822 = getelementptr i8, ptr addrspace(3) %address.local1012, i64 8704, !dbg !49
  %1034 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1822, i32 0) #11, !dbg !49, !frisk.v_double_buffer !50
  %1035 = bitcast <4 x i32> %1034 to <8 x half>, !dbg !49
  %vdb.read1823 = getelementptr i8, ptr addrspace(3) %address.local1012, i64 8960, !dbg !49
  %1036 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1823, i32 0) #11, !dbg !49, !frisk.v_double_buffer !50
  %1037 = bitcast <4 x i32> %1036 to <8 x half>, !dbg !49
  %vdb.read1824 = getelementptr i8, ptr addrspace(3) %address.local1003, i64 12288, !dbg !49
  %1038 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1824, i32 0) #11, !dbg !49, !frisk.v_double_buffer !50
  %1039 = bitcast <4 x i32> %1038 to <8 x half>, !dbg !49
  %vdb.read1825 = getelementptr i8, ptr addrspace(3) %address.local1012, i64 12544, !dbg !49
  %1040 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1825, i32 0) #11, !dbg !49, !frisk.v_double_buffer !50
  %1041 = bitcast <4 x i32> %1040 to <8 x half>, !dbg !49
  %vdb.read1826 = getelementptr i8, ptr addrspace(3) %address.local1012, i64 12800, !dbg !49
  %1042 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1826, i32 0) #11, !dbg !49, !frisk.v_double_buffer !50
  %1043 = bitcast <4 x i32> %1042 to <8 x half>, !dbg !49
  %vdb.read1827 = getelementptr i8, ptr addrspace(3) %address.local1012, i64 13056, !dbg !49
  %1044 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1827, i32 0) #11, !dbg !49, !frisk.v_double_buffer !50
  %1045 = shufflevector <8 x half> %1031, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  %1046 = extractvalue [2 x <8 x float>] %129, 0, !dbg !47
  %1047 = shufflevector <8 x float> %1046, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !47
  %1048 = fmul <4 x float> %1047, %708, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1049 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1045, <4 x float> %1048) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1050 = shufflevector <8 x half> %1031, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  %1051 = shufflevector <8 x float> %1046, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !47
  %1052 = fmul <4 x float> %1051, %708, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1053 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1050, <4 x float> %1052) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1054 = extractvalue [2 x <8 x float>] %129, 1, !dbg !47
  %1055 = shufflevector <8 x float> %1054, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !47
  %1056 = fmul <4 x float> %1055, %716, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1057 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed457, <4 x half> %1045, <4 x float> %1056) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1058 = shufflevector <8 x float> %1054, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !47
  %1059 = fmul <4 x float> %1058, %716, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1060 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed457, <4 x half> %1050, <4 x float> %1059) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1061 = shufflevector <8 x half> %1033, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1062 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed443, <4 x half> %1061, <4 x float> %1049) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1063 = shufflevector <8 x half> %1033, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1064 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed443, <4 x half> %1063, <4 x float> %1053) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1065 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed459, <4 x half> %1061, <4 x float> %1057) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1066 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed459, <4 x half> %1063, <4 x float> %1060) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1067 = shufflevector <8 x half> %1035, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1068 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed445, <4 x half> %1067, <4 x float> %1062) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1069 = shufflevector <8 x half> %1035, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1070 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed445, <4 x half> %1069, <4 x float> %1064) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1071 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed461, <4 x half> %1067, <4 x float> %1065) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1072 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed461, <4 x half> %1069, <4 x float> %1066) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1073 = shufflevector <8 x half> %1037, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1074 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed447, <4 x half> %1073, <4 x float> %1068) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1075 = shufflevector <8 x half> %1037, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1076 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed447, <4 x half> %1075, <4 x float> %1070) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1077 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed463, <4 x half> %1073, <4 x float> %1071) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1078 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed463, <4 x half> %1075, <4 x float> %1072) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1079 = shufflevector <8 x half> %1039, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1080 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed449, <4 x half> %1079, <4 x float> %1074) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1081 = shufflevector <8 x half> %1039, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1082 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed449, <4 x half> %1081, <4 x float> %1076) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1083 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed465, <4 x half> %1079, <4 x float> %1077) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1084 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed465, <4 x half> %1081, <4 x float> %1078) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1085 = shufflevector <8 x half> %1041, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1086 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed451, <4 x half> %1085, <4 x float> %1080) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1087 = shufflevector <8 x half> %1041, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1088 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed451, <4 x half> %1087, <4 x float> %1082) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1089 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed467, <4 x half> %1085, <4 x float> %1083) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1090 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed467, <4 x half> %1087, <4 x float> %1084) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1091 = shufflevector <8 x half> %1043, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1092 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed453, <4 x half> %1091, <4 x float> %1086) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1093 = shufflevector <8 x half> %1043, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1094 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed453, <4 x half> %1093, <4 x float> %1088) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1095 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed469, <4 x half> %1091, <4 x float> %1089) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1096 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed469, <4 x half> %1093, <4 x float> %1090) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1097 = bitcast <4 x i32> %1044 to <8 x half>, !dbg !49
  %1098 = shufflevector <8 x half> %1097, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1099 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed455, <4 x half> %1098, <4 x float> %1092) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1100 = shufflevector <8 x half> %1097, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1101 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed455, <4 x half> %1100, <4 x float> %1094) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1102 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed471, <4 x half> %1098, <4 x float> %1095) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1103 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed471, <4 x half> %1100, <4 x float> %1096) #8, !dbg !49, !frisk.pv_interleave !21
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
  %1104 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1853, <4 x i32> %dma.buffer.rsrc1677) #9, !dbg !40
  %1105 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1820, i32 %vdb.address1863, <4 x i32> %dma.buffer.rsrc1677) #9, !dbg !40
  %1106 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1003, i32 0) #11, !dbg !49, !frisk.v_double_buffer !50
  %1107 = bitcast <4 x i32> %1106 to <8 x half>, !dbg !49
  %1108 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1013, i32 0) #11, !dbg !49, !frisk.v_double_buffer !50
  %1109 = bitcast <4 x i32> %1108 to <8 x half>, !dbg !49
  %1110 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1017, i32 0) #11, !dbg !49, !frisk.v_double_buffer !50
  %1111 = bitcast <4 x i32> %1110 to <8 x half>, !dbg !49
  %1112 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1021, i32 0) #11, !dbg !49, !frisk.v_double_buffer !50
  %1113 = bitcast <4 x i32> %1112 to <8 x half>, !dbg !49
  %1114 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1023, i32 0) #11, !dbg !49, !frisk.v_double_buffer !50
  %1115 = bitcast <4 x i32> %1114 to <8 x half>, !dbg !49
  %1116 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1027, i32 0) #11, !dbg !49, !frisk.v_double_buffer !50
  %1117 = bitcast <4 x i32> %1116 to <8 x half>, !dbg !49
  %1118 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1031, i32 0) #11, !dbg !49, !frisk.v_double_buffer !50
  %1119 = bitcast <4 x i32> %1118 to <8 x half>, !dbg !49
  %1120 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1035, i32 0) #11, !dbg !49, !frisk.v_double_buffer !50
  %1121 = shufflevector <8 x half> %1107, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  %1122 = extractvalue [2 x <8 x float>] %128, 0, !dbg !47
  %1123 = shufflevector <8 x float> %1122, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !47
  %1124 = fmul <4 x float> %1123, %708, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1125 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1121, <4 x float> %1124) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1126 = shufflevector <8 x half> %1107, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  %1127 = shufflevector <8 x float> %1122, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !47
  %1128 = fmul <4 x float> %1127, %708, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1129 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1126, <4 x float> %1128) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1130 = extractvalue [2 x <8 x float>] %128, 1, !dbg !47
  %1131 = shufflevector <8 x float> %1130, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !47
  %1132 = fmul <4 x float> %1131, %716, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1133 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed457, <4 x half> %1121, <4 x float> %1132) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1134 = shufflevector <8 x float> %1130, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !47
  %1135 = fmul <4 x float> %1134, %716, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1136 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed457, <4 x half> %1126, <4 x float> %1135) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1137 = shufflevector <8 x half> %1109, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1138 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed443, <4 x half> %1137, <4 x float> %1125) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1139 = shufflevector <8 x half> %1109, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1140 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed443, <4 x half> %1139, <4 x float> %1129) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1141 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed459, <4 x half> %1137, <4 x float> %1133) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1142 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed459, <4 x half> %1139, <4 x float> %1136) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1143 = shufflevector <8 x half> %1111, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1144 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed445, <4 x half> %1143, <4 x float> %1138) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1145 = shufflevector <8 x half> %1111, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1146 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed445, <4 x half> %1145, <4 x float> %1140) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1147 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed461, <4 x half> %1143, <4 x float> %1141) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1148 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed461, <4 x half> %1145, <4 x float> %1142) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1149 = shufflevector <8 x half> %1113, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1150 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed447, <4 x half> %1149, <4 x float> %1144) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1151 = shufflevector <8 x half> %1113, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1152 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed447, <4 x half> %1151, <4 x float> %1146) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1153 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed463, <4 x half> %1149, <4 x float> %1147) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1154 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed463, <4 x half> %1151, <4 x float> %1148) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1155 = shufflevector <8 x half> %1115, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1156 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed449, <4 x half> %1155, <4 x float> %1150) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1157 = shufflevector <8 x half> %1115, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1158 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed449, <4 x half> %1157, <4 x float> %1152) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1159 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed465, <4 x half> %1155, <4 x float> %1153) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1160 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed465, <4 x half> %1157, <4 x float> %1154) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1161 = shufflevector <8 x half> %1117, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1162 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed451, <4 x half> %1161, <4 x float> %1156) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1163 = shufflevector <8 x half> %1117, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1164 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed451, <4 x half> %1163, <4 x float> %1158) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1165 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed467, <4 x half> %1161, <4 x float> %1159) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1166 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed467, <4 x half> %1163, <4 x float> %1160) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1167 = shufflevector <8 x half> %1119, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1168 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed453, <4 x half> %1167, <4 x float> %1162) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1169 = shufflevector <8 x half> %1119, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1170 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed453, <4 x half> %1169, <4 x float> %1164) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1171 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed469, <4 x half> %1167, <4 x float> %1165) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1172 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed469, <4 x half> %1169, <4 x float> %1166) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1173 = bitcast <4 x i32> %1120 to <8 x half>, !dbg !49
  %1174 = shufflevector <8 x half> %1173, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1175 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed455, <4 x half> %1174, <4 x float> %1168) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1176 = shufflevector <8 x half> %1173, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1177 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed455, <4 x half> %1176, <4 x float> %1170) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1178 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed471, <4 x half> %1174, <4 x float> %1171) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1179 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed471, <4 x half> %1176, <4 x float> %1172) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !40
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !40
  fence syncscope("workgroup") release, !dbg !40
  tail call void @llvm.amdgcn.s.barrier(), !dbg !40
  fence syncscope("workgroup") acquire, !dbg !40
  %1180 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #11, !dbg !49, !frisk.v_double_buffer !50
  %1181 = bitcast <4 x i32> %1180 to <8 x half>, !dbg !49
  %1182 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1821, i32 0) #11, !dbg !49, !frisk.v_double_buffer !50
  %1183 = bitcast <4 x i32> %1182 to <8 x half>, !dbg !49
  %1184 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1822, i32 0) #11, !dbg !49, !frisk.v_double_buffer !50
  %1185 = bitcast <4 x i32> %1184 to <8 x half>, !dbg !49
  %1186 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1823, i32 0) #11, !dbg !49, !frisk.v_double_buffer !50
  %1187 = bitcast <4 x i32> %1186 to <8 x half>, !dbg !49
  %1188 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1824, i32 0) #11, !dbg !49, !frisk.v_double_buffer !50
  %1189 = bitcast <4 x i32> %1188 to <8 x half>, !dbg !49
  %1190 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1825, i32 0) #11, !dbg !49, !frisk.v_double_buffer !50
  %1191 = bitcast <4 x i32> %1190 to <8 x half>, !dbg !49
  %1192 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1826, i32 0) #11, !dbg !49, !frisk.v_double_buffer !50
  %1193 = bitcast <4 x i32> %1192 to <8 x half>, !dbg !49
  %1194 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1827, i32 0) #11, !dbg !49, !frisk.v_double_buffer !50
  %1195 = shufflevector <8 x half> %1181, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  %1196 = extractvalue [2 x <8 x float>] %127, 0, !dbg !47
  %1197 = shufflevector <8 x float> %1196, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !47
  %1198 = fmul <4 x float> %1197, %708, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1199 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1195, <4 x float> %1198) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1200 = shufflevector <8 x half> %1181, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  %1201 = shufflevector <8 x float> %1196, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !47
  %1202 = fmul <4 x float> %1201, %708, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1203 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1200, <4 x float> %1202) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1204 = extractvalue [2 x <8 x float>] %127, 1, !dbg !47
  %1205 = shufflevector <8 x float> %1204, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !47
  %1206 = fmul <4 x float> %1205, %716, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1207 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed457, <4 x half> %1195, <4 x float> %1206) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !47
  %1208 = shufflevector <8 x float> %1204, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !47
  %1209 = fmul <4 x float> %1208, %716, !dbg !47
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1210 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed457, <4 x half> %1200, <4 x float> %1209) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1211 = shufflevector <8 x half> %1183, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1212 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed443, <4 x half> %1211, <4 x float> %1199) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1213 = shufflevector <8 x half> %1183, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1214 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed443, <4 x half> %1213, <4 x float> %1203) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1215 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed459, <4 x half> %1211, <4 x float> %1207) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1216 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed459, <4 x half> %1213, <4 x float> %1210) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1217 = shufflevector <8 x half> %1185, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1218 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed445, <4 x half> %1217, <4 x float> %1212) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1219 = shufflevector <8 x half> %1185, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1220 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed445, <4 x half> %1219, <4 x float> %1214) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1221 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed461, <4 x half> %1217, <4 x float> %1215) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1222 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed461, <4 x half> %1219, <4 x float> %1216) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1223 = shufflevector <8 x half> %1187, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1224 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed447, <4 x half> %1223, <4 x float> %1218) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1225 = shufflevector <8 x half> %1187, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1226 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed447, <4 x half> %1225, <4 x float> %1220) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1227 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed463, <4 x half> %1223, <4 x float> %1221) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1228 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed463, <4 x half> %1225, <4 x float> %1222) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1229 = shufflevector <8 x half> %1189, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1230 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed449, <4 x half> %1229, <4 x float> %1224) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1231 = shufflevector <8 x half> %1189, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1232 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed449, <4 x half> %1231, <4 x float> %1226) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1233 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed465, <4 x half> %1229, <4 x float> %1227) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1234 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed465, <4 x half> %1231, <4 x float> %1228) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1235 = shufflevector <8 x half> %1191, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1236 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed451, <4 x half> %1235, <4 x float> %1230) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1237 = shufflevector <8 x half> %1191, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1238 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed451, <4 x half> %1237, <4 x float> %1232) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1239 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed467, <4 x half> %1235, <4 x float> %1233) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1240 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed467, <4 x half> %1237, <4 x float> %1234) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1241 = shufflevector <8 x half> %1193, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1242 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed453, <4 x half> %1241, <4 x float> %1236) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1243 = shufflevector <8 x half> %1193, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1244 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed453, <4 x half> %1243, <4 x float> %1238) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1245 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed469, <4 x half> %1241, <4 x float> %1239) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1246 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed469, <4 x half> %1243, <4 x float> %1240) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1247 = bitcast <4 x i32> %1194 to <8 x half>, !dbg !49
  %1248 = shufflevector <8 x half> %1247, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1249 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed455, <4 x half> %1248, <4 x float> %1242) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1250 = shufflevector <8 x half> %1247, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1251 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed455, <4 x half> %1250, <4 x float> %1244) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1252 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed471, <4 x half> %1248, <4 x float> %1245) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !49
  %1253 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed471, <4 x half> %1250, <4 x float> %1246) #8, !dbg !49, !frisk.pv_interleave !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !11
  %1254 = shufflevector <4 x float> %1005, <4 x float> %1007, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1255 = insertvalue [2 x <8 x float>] poison, <8 x float> %1254, 0, !dbg !11
  %1256 = shufflevector <4 x float> %1026, <4 x float> %1027, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1257 = insertvalue [2 x <8 x float>] %1255, <8 x float> %1256, 1, !dbg !11
  %1258 = insertvalue [2 x <1 x float>] poison, <1 x float> %688, 0, !dbg !11
  %1259 = insertvalue [2 x <1 x float>] %1258, <1 x float> %701, 1, !dbg !11
  %1260 = shufflevector <4 x float> %1099, <4 x float> %1101, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1261 = insertvalue [2 x <8 x float>] poison, <8 x float> %1260, 0, !dbg !11
  %1262 = shufflevector <4 x float> %1102, <4 x float> %1103, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1263 = insertvalue [2 x <8 x float>] %1261, <8 x float> %1262, 1, !dbg !11
  %1264 = shufflevector <4 x float> %1175, <4 x float> %1177, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1265 = insertvalue [2 x <8 x float>] poison, <8 x float> %1264, 0, !dbg !11
  %1266 = shufflevector <4 x float> %1178, <4 x float> %1179, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1267 = insertvalue [2 x <8 x float>] %1265, <8 x float> %1266, 1, !dbg !11
  %1268 = shufflevector <4 x float> %1249, <4 x float> %1251, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1269 = insertvalue [2 x <8 x float>] poison, <8 x float> %1268, 0, !dbg !11
  %1270 = shufflevector <4 x float> %1252, <4 x float> %1253, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1271 = insertvalue [2 x <8 x float>] %1269, <8 x float> %1270, 1, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  tail call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %1272 = add nuw nsw i64 %133, 1, !dbg !11
  %.not.not = icmp samesign ult i64 %133, %16, !dbg !11
  br i1 %.not.not, label %126, label %1273, !dbg !11

1273:                                             ; preds = %126
  %1274 = shl nuw nsw i32 %667, 2, !dbg !11
  %1275 = xor i32 %1274, 64, !dbg !11
  %1276 = bitcast float %1021 to i32, !dbg !11
  %lane.sum.shuffle256 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1275, i32 %1276), !dbg !11
  %1277 = bitcast i32 %lane.sum.shuffle256 to float, !dbg !11
  %lane.sum.final257 = fadd float %1021, %1277, !dbg !11
  %1278 = xor i32 %1274, 128, !dbg !11
  %1279 = bitcast float %lane.sum.final257 to i32, !dbg !11
  %lane.sum.shuffle258 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1278, i32 %1279), !dbg !11
  %1280 = bitcast i32 %lane.sum.shuffle258 to float, !dbg !11
  %lane.sum.final259 = fadd float %lane.sum.final257, %1280, !dbg !11
  %1281 = insertelement <1 x float> poison, float %lane.sum.final259, i64 0, !dbg !11
  %1282 = bitcast float %1000 to i32, !dbg !11
  %lane.sum.shuffle = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1275, i32 %1282), !dbg !11
  %1283 = bitcast i32 %lane.sum.shuffle to float, !dbg !11
  %lane.sum.final = fadd float %1000, %1283, !dbg !11
  %1284 = bitcast float %lane.sum.final to i32, !dbg !11
  %lane.sum.shuffle250 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1278, i32 %1284), !dbg !11
  %1285 = bitcast i32 %lane.sum.shuffle250 to float, !dbg !11
  %lane.sum.final251 = fadd float %lane.sum.final, %1285, !dbg !11
  %1286 = insertelement <1 x float> poison, float %lane.sum.final251, i64 0, !dbg !11
  %1287 = and i32 %kdma.tid, 16, !dbg !11
  %.not36 = icmp eq i32 %1287, 0, !dbg !11
  %1288 = and i32 %kdma.tid, 32, !dbg !11
  %.not37 = icmp eq i32 %1288, 0, !dbg !11
  %1289 = extractelement <4 x float> %1249, i64 0, !dbg !11
  %1290 = extractelement <4 x float> %1249, i64 1, !dbg !11
  %1291 = extractelement <4 x float> %1249, i64 2, !dbg !11
  %1292 = extractelement <4 x float> %1249, i64 3, !dbg !11
  %1293 = select i1 %.not36, float %1290, float %1289, !dbg !11
  %1294 = bitcast float %1293 to i32, !dbg !11
  %1295 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %673, i32 %1294), !dbg !11
  %1296 = bitcast i32 %1295 to float, !dbg !11
  %1297 = select i1 %.not36, float %1289, float %1296, !dbg !11
  %1298 = select i1 %.not36, float %1296, float %1290, !dbg !11
  %1299 = select i1 %.not36, float %1292, float %1291, !dbg !11
  %1300 = bitcast float %1299 to i32, !dbg !11
  %1301 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %673, i32 %1300), !dbg !11
  %1302 = bitcast i32 %1301 to float, !dbg !11
  %1303 = select i1 %.not36, float %1291, float %1302, !dbg !11
  %1304 = select i1 %.not36, float %1302, float %1292, !dbg !11
  %1305 = select i1 %.not37, float %1303, float %1297, !dbg !11
  %1306 = bitcast float %1305 to i32, !dbg !11
  %1307 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %681, i32 %1306), !dbg !11
  %1308 = bitcast i32 %1307 to float, !dbg !11
  %1309 = select i1 %.not37, float %1304, float %1298, !dbg !11
  %1310 = bitcast float %1309 to i32, !dbg !11
  %1311 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %681, i32 %1310), !dbg !11
  %1312 = bitcast i32 %1311 to float, !dbg !11
  %1313 = insertelement <4 x i1> poison, i1 %.not37, i64 0, !dbg !11
  %1314 = shufflevector <4 x i1> %1313, <4 x i1> poison, <4 x i32> zeroinitializer, !dbg !11
  %1315 = insertelement <4 x float> poison, float %1297, i64 0, !dbg !11
  %1316 = insertelement <4 x float> %1315, float %1298, i64 1, !dbg !11
  %1317 = insertelement <4 x float> %1316, float %1308, i64 2, !dbg !11
  %1318 = insertelement <4 x float> %1317, float %1312, i64 3, !dbg !11
  %1319 = insertelement <4 x float> poison, float %1308, i64 0, !dbg !11
  %1320 = insertelement <4 x float> %1319, float %1312, i64 1, !dbg !11
  %1321 = insertelement <4 x float> %1320, float %1303, i64 2, !dbg !11
  %1322 = insertelement <4 x float> %1321, float %1304, i64 3, !dbg !11
  %1323 = select <4 x i1> %1314, <4 x float> %1318, <4 x float> %1322, !dbg !11
  %1324 = extractelement <4 x float> %1175, i64 0, !dbg !11
  %1325 = extractelement <4 x float> %1175, i64 1, !dbg !11
  %1326 = extractelement <4 x float> %1175, i64 2, !dbg !11
  %1327 = extractelement <4 x float> %1175, i64 3, !dbg !11
  %1328 = select i1 %.not36, float %1325, float %1324, !dbg !11
  %1329 = bitcast float %1328 to i32, !dbg !11
  %1330 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %673, i32 %1329), !dbg !11
  %1331 = bitcast i32 %1330 to float, !dbg !11
  %1332 = select i1 %.not36, float %1324, float %1331, !dbg !11
  %1333 = select i1 %.not36, float %1331, float %1325, !dbg !11
  %1334 = select i1 %.not36, float %1327, float %1326, !dbg !11
  %1335 = bitcast float %1334 to i32, !dbg !11
  %1336 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %673, i32 %1335), !dbg !11
  %1337 = bitcast i32 %1336 to float, !dbg !11
  %1338 = select i1 %.not36, float %1326, float %1337, !dbg !11
  %1339 = select i1 %.not36, float %1337, float %1327, !dbg !11
  %1340 = select i1 %.not37, float %1338, float %1332, !dbg !11
  %1341 = bitcast float %1340 to i32, !dbg !11
  %1342 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %681, i32 %1341), !dbg !11
  %1343 = bitcast i32 %1342 to float, !dbg !11
  %1344 = select i1 %.not37, float %1339, float %1333, !dbg !11
  %1345 = bitcast float %1344 to i32, !dbg !11
  %1346 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %681, i32 %1345), !dbg !11
  %1347 = bitcast i32 %1346 to float, !dbg !11
  %1348 = insertelement <4 x float> poison, float %1332, i64 0, !dbg !11
  %1349 = insertelement <4 x float> %1348, float %1333, i64 1, !dbg !11
  %1350 = insertelement <4 x float> %1349, float %1343, i64 2, !dbg !11
  %1351 = insertelement <4 x float> %1350, float %1347, i64 3, !dbg !11
  %1352 = insertelement <4 x float> poison, float %1343, i64 0, !dbg !11
  %1353 = insertelement <4 x float> %1352, float %1347, i64 1, !dbg !11
  %1354 = insertelement <4 x float> %1353, float %1338, i64 2, !dbg !11
  %1355 = insertelement <4 x float> %1354, float %1339, i64 3, !dbg !11
  %1356 = select <4 x i1> %1314, <4 x float> %1351, <4 x float> %1355, !dbg !11
  %1357 = extractelement <4 x float> %1099, i64 0, !dbg !11
  %1358 = extractelement <4 x float> %1099, i64 1, !dbg !11
  %1359 = extractelement <4 x float> %1099, i64 2, !dbg !11
  %1360 = extractelement <4 x float> %1099, i64 3, !dbg !11
  %1361 = select i1 %.not36, float %1358, float %1357, !dbg !11
  %1362 = bitcast float %1361 to i32, !dbg !11
  %1363 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %673, i32 %1362), !dbg !11
  %1364 = bitcast i32 %1363 to float, !dbg !11
  %1365 = select i1 %.not36, float %1357, float %1364, !dbg !11
  %1366 = select i1 %.not36, float %1364, float %1358, !dbg !11
  %1367 = select i1 %.not36, float %1360, float %1359, !dbg !11
  %1368 = bitcast float %1367 to i32, !dbg !11
  %1369 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %673, i32 %1368), !dbg !11
  %1370 = bitcast i32 %1369 to float, !dbg !11
  %1371 = select i1 %.not36, float %1359, float %1370, !dbg !11
  %1372 = select i1 %.not36, float %1370, float %1360, !dbg !11
  %1373 = select i1 %.not37, float %1371, float %1365, !dbg !11
  %1374 = bitcast float %1373 to i32, !dbg !11
  %1375 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %681, i32 %1374), !dbg !11
  %1376 = bitcast i32 %1375 to float, !dbg !11
  %1377 = select i1 %.not37, float %1372, float %1366, !dbg !11
  %1378 = bitcast float %1377 to i32, !dbg !11
  %1379 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %681, i32 %1378), !dbg !11
  %1380 = bitcast i32 %1379 to float, !dbg !11
  %1381 = insertelement <4 x float> poison, float %1365, i64 0, !dbg !11
  %1382 = insertelement <4 x float> %1381, float %1366, i64 1, !dbg !11
  %1383 = insertelement <4 x float> %1382, float %1376, i64 2, !dbg !11
  %1384 = insertelement <4 x float> %1383, float %1380, i64 3, !dbg !11
  %1385 = insertelement <4 x float> poison, float %1376, i64 0, !dbg !11
  %1386 = insertelement <4 x float> %1385, float %1380, i64 1, !dbg !11
  %1387 = insertelement <4 x float> %1386, float %1371, i64 2, !dbg !11
  %1388 = insertelement <4 x float> %1387, float %1372, i64 3, !dbg !11
  %1389 = select <4 x i1> %1314, <4 x float> %1384, <4 x float> %1388, !dbg !11
  %1390 = extractelement <4 x float> %1005, i64 0, !dbg !11
  %1391 = extractelement <4 x float> %1005, i64 1, !dbg !11
  %1392 = extractelement <4 x float> %1005, i64 2, !dbg !11
  %1393 = extractelement <4 x float> %1005, i64 3, !dbg !11
  %1394 = select i1 %.not36, float %1391, float %1390, !dbg !11
  %1395 = bitcast float %1394 to i32, !dbg !11
  %1396 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %673, i32 %1395), !dbg !11
  %1397 = bitcast i32 %1396 to float, !dbg !11
  %1398 = select i1 %.not36, float %1390, float %1397, !dbg !11
  %1399 = select i1 %.not36, float %1397, float %1391, !dbg !11
  %1400 = select i1 %.not36, float %1393, float %1392, !dbg !11
  %1401 = bitcast float %1400 to i32, !dbg !11
  %1402 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %673, i32 %1401), !dbg !11
  %1403 = bitcast i32 %1402 to float, !dbg !11
  %1404 = select i1 %.not36, float %1392, float %1403, !dbg !11
  %1405 = select i1 %.not36, float %1403, float %1393, !dbg !11
  %1406 = select i1 %.not37, float %1404, float %1398, !dbg !11
  %1407 = bitcast float %1406 to i32, !dbg !11
  %1408 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %681, i32 %1407), !dbg !11
  %1409 = select i1 %.not37, float %1405, float %1399, !dbg !11
  %1410 = bitcast float %1409 to i32, !dbg !11
  %1411 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %681, i32 %1410), !dbg !11
  %1412 = shufflevector <1 x float> %1286, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !54
  %1413 = extractelement <4 x float> %1251, i64 0, !dbg !11
  %1414 = extractelement <4 x float> %1251, i64 1, !dbg !11
  %1415 = extractelement <4 x float> %1251, i64 2, !dbg !11
  %1416 = extractelement <4 x float> %1251, i64 3, !dbg !11
  %1417 = select i1 %.not36, float %1414, float %1413, !dbg !11
  %1418 = bitcast float %1417 to i32, !dbg !11
  %1419 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %673, i32 %1418), !dbg !11
  %1420 = bitcast i32 %1419 to float, !dbg !11
  %1421 = select i1 %.not36, float %1413, float %1420, !dbg !11
  %1422 = select i1 %.not36, float %1420, float %1414, !dbg !11
  %1423 = select i1 %.not36, float %1416, float %1415, !dbg !11
  %1424 = bitcast float %1423 to i32, !dbg !11
  %1425 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %673, i32 %1424), !dbg !11
  %1426 = bitcast i32 %1425 to float, !dbg !11
  %1427 = select i1 %.not36, float %1415, float %1426, !dbg !11
  %1428 = select i1 %.not36, float %1426, float %1416, !dbg !11
  %1429 = select i1 %.not37, float %1427, float %1421, !dbg !11
  %1430 = bitcast float %1429 to i32, !dbg !11
  %1431 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %681, i32 %1430), !dbg !11
  %1432 = bitcast i32 %1431 to float, !dbg !11
  %1433 = select i1 %.not37, float %1428, float %1422, !dbg !11
  %1434 = bitcast float %1433 to i32, !dbg !11
  %1435 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %681, i32 %1434), !dbg !11
  %1436 = bitcast i32 %1435 to float, !dbg !11
  %1437 = insertelement <4 x float> poison, float %1421, i64 0, !dbg !11
  %1438 = insertelement <4 x float> %1437, float %1422, i64 1, !dbg !11
  %1439 = insertelement <4 x float> %1438, float %1432, i64 2, !dbg !11
  %1440 = insertelement <4 x float> %1439, float %1436, i64 3, !dbg !11
  %1441 = insertelement <4 x float> poison, float %1432, i64 0, !dbg !11
  %1442 = insertelement <4 x float> %1441, float %1436, i64 1, !dbg !11
  %1443 = insertelement <4 x float> %1442, float %1427, i64 2, !dbg !11
  %1444 = insertelement <4 x float> %1443, float %1428, i64 3, !dbg !11
  %1445 = select <4 x i1> %1314, <4 x float> %1440, <4 x float> %1444, !dbg !11
  %1446 = extractelement <4 x float> %1177, i64 0, !dbg !11
  %1447 = extractelement <4 x float> %1177, i64 1, !dbg !11
  %1448 = extractelement <4 x float> %1177, i64 2, !dbg !11
  %1449 = extractelement <4 x float> %1177, i64 3, !dbg !11
  %1450 = select i1 %.not36, float %1447, float %1446, !dbg !11
  %1451 = bitcast float %1450 to i32, !dbg !11
  %1452 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %673, i32 %1451), !dbg !11
  %1453 = bitcast i32 %1452 to float, !dbg !11
  %1454 = select i1 %.not36, float %1446, float %1453, !dbg !11
  %1455 = select i1 %.not36, float %1453, float %1447, !dbg !11
  %1456 = select i1 %.not36, float %1449, float %1448, !dbg !11
  %1457 = bitcast float %1456 to i32, !dbg !11
  %1458 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %673, i32 %1457), !dbg !11
  %1459 = bitcast i32 %1458 to float, !dbg !11
  %1460 = select i1 %.not36, float %1448, float %1459, !dbg !11
  %1461 = select i1 %.not36, float %1459, float %1449, !dbg !11
  %1462 = select i1 %.not37, float %1460, float %1454, !dbg !11
  %1463 = bitcast float %1462 to i32, !dbg !11
  %1464 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %681, i32 %1463), !dbg !11
  %1465 = bitcast i32 %1464 to float, !dbg !11
  %1466 = select i1 %.not37, float %1461, float %1455, !dbg !11
  %1467 = bitcast float %1466 to i32, !dbg !11
  %1468 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %681, i32 %1467), !dbg !11
  %1469 = bitcast i32 %1468 to float, !dbg !11
  %1470 = insertelement <4 x float> poison, float %1454, i64 0, !dbg !11
  %1471 = insertelement <4 x float> %1470, float %1455, i64 1, !dbg !11
  %1472 = insertelement <4 x float> %1471, float %1465, i64 2, !dbg !11
  %1473 = insertelement <4 x float> %1472, float %1469, i64 3, !dbg !11
  %1474 = insertelement <4 x float> poison, float %1465, i64 0, !dbg !11
  %1475 = insertelement <4 x float> %1474, float %1469, i64 1, !dbg !11
  %1476 = insertelement <4 x float> %1475, float %1460, i64 2, !dbg !11
  %1477 = insertelement <4 x float> %1476, float %1461, i64 3, !dbg !11
  %1478 = select <4 x i1> %1314, <4 x float> %1473, <4 x float> %1477, !dbg !11
  %1479 = extractelement <4 x float> %1101, i64 0, !dbg !11
  %1480 = extractelement <4 x float> %1101, i64 1, !dbg !11
  %1481 = extractelement <4 x float> %1101, i64 2, !dbg !11
  %1482 = extractelement <4 x float> %1101, i64 3, !dbg !11
  %1483 = select i1 %.not36, float %1480, float %1479, !dbg !11
  %1484 = bitcast float %1483 to i32, !dbg !11
  %1485 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %673, i32 %1484), !dbg !11
  %1486 = bitcast i32 %1485 to float, !dbg !11
  %1487 = select i1 %.not36, float %1479, float %1486, !dbg !11
  %1488 = select i1 %.not36, float %1486, float %1480, !dbg !11
  %1489 = select i1 %.not36, float %1482, float %1481, !dbg !11
  %1490 = bitcast float %1489 to i32, !dbg !11
  %1491 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %673, i32 %1490), !dbg !11
  %1492 = bitcast i32 %1491 to float, !dbg !11
  %1493 = select i1 %.not36, float %1481, float %1492, !dbg !11
  %1494 = select i1 %.not36, float %1492, float %1482, !dbg !11
  %1495 = select i1 %.not37, float %1493, float %1487, !dbg !11
  %1496 = bitcast float %1495 to i32, !dbg !11
  %1497 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %681, i32 %1496), !dbg !11
  %1498 = bitcast i32 %1497 to float, !dbg !11
  %1499 = select i1 %.not37, float %1494, float %1488, !dbg !11
  %1500 = bitcast float %1499 to i32, !dbg !11
  %1501 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %681, i32 %1500), !dbg !11
  %1502 = bitcast i32 %1501 to float, !dbg !11
  %1503 = insertelement <4 x float> poison, float %1487, i64 0, !dbg !11
  %1504 = insertelement <4 x float> %1503, float %1488, i64 1, !dbg !11
  %1505 = insertelement <4 x float> %1504, float %1498, i64 2, !dbg !11
  %1506 = insertelement <4 x float> %1505, float %1502, i64 3, !dbg !11
  %1507 = insertelement <4 x float> poison, float %1498, i64 0, !dbg !11
  %1508 = insertelement <4 x float> %1507, float %1502, i64 1, !dbg !11
  %1509 = insertelement <4 x float> %1508, float %1493, i64 2, !dbg !11
  %1510 = insertelement <4 x float> %1509, float %1494, i64 3, !dbg !11
  %1511 = select <4 x i1> %1314, <4 x float> %1506, <4 x float> %1510, !dbg !11
  %1512 = extractelement <4 x float> %1007, i64 0, !dbg !11
  %1513 = extractelement <4 x float> %1007, i64 1, !dbg !11
  %1514 = extractelement <4 x float> %1007, i64 2, !dbg !11
  %1515 = extractelement <4 x float> %1007, i64 3, !dbg !11
  %1516 = select i1 %.not36, float %1513, float %1512, !dbg !11
  %1517 = bitcast float %1516 to i32, !dbg !11
  %1518 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %673, i32 %1517), !dbg !11
  %1519 = bitcast i32 %1518 to float, !dbg !11
  %1520 = select i1 %.not36, float %1512, float %1519, !dbg !11
  %1521 = select i1 %.not36, float %1519, float %1513, !dbg !11
  %1522 = select i1 %.not36, float %1515, float %1514, !dbg !11
  %1523 = bitcast float %1522 to i32, !dbg !11
  %1524 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %673, i32 %1523), !dbg !11
  %1525 = bitcast i32 %1524 to float, !dbg !11
  %1526 = select i1 %.not36, float %1514, float %1525, !dbg !11
  %1527 = select i1 %.not36, float %1525, float %1515, !dbg !11
  %1528 = select i1 %.not37, float %1526, float %1520, !dbg !11
  %1529 = bitcast float %1528 to i32, !dbg !11
  %1530 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %681, i32 %1529), !dbg !11
  %1531 = select i1 %.not37, float %1527, float %1521, !dbg !11
  %1532 = bitcast float %1531 to i32, !dbg !11
  %1533 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %681, i32 %1532), !dbg !11
  %1534 = extractelement <4 x float> %1252, i64 0, !dbg !11
  %1535 = extractelement <4 x float> %1252, i64 1, !dbg !11
  %1536 = extractelement <4 x float> %1252, i64 2, !dbg !11
  %1537 = extractelement <4 x float> %1252, i64 3, !dbg !11
  %1538 = select i1 %.not36, float %1535, float %1534, !dbg !11
  %1539 = bitcast float %1538 to i32, !dbg !11
  %1540 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %673, i32 %1539), !dbg !11
  %1541 = bitcast i32 %1540 to float, !dbg !11
  %1542 = select i1 %.not36, float %1534, float %1541, !dbg !11
  %1543 = select i1 %.not36, float %1541, float %1535, !dbg !11
  %1544 = select i1 %.not36, float %1537, float %1536, !dbg !11
  %1545 = bitcast float %1544 to i32, !dbg !11
  %1546 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %673, i32 %1545), !dbg !11
  %1547 = bitcast i32 %1546 to float, !dbg !11
  %1548 = select i1 %.not36, float %1536, float %1547, !dbg !11
  %1549 = select i1 %.not36, float %1547, float %1537, !dbg !11
  %1550 = select i1 %.not37, float %1548, float %1542, !dbg !11
  %1551 = bitcast float %1550 to i32, !dbg !11
  %1552 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %681, i32 %1551), !dbg !11
  %1553 = bitcast i32 %1552 to float, !dbg !11
  %1554 = select i1 %.not37, float %1549, float %1543, !dbg !11
  %1555 = bitcast float %1554 to i32, !dbg !11
  %1556 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %681, i32 %1555), !dbg !11
  %1557 = bitcast i32 %1556 to float, !dbg !11
  %1558 = insertelement <4 x float> poison, float %1542, i64 0, !dbg !11
  %1559 = insertelement <4 x float> %1558, float %1543, i64 1, !dbg !11
  %1560 = insertelement <4 x float> %1559, float %1553, i64 2, !dbg !11
  %1561 = insertelement <4 x float> %1560, float %1557, i64 3, !dbg !11
  %1562 = insertelement <4 x float> poison, float %1553, i64 0, !dbg !11
  %1563 = insertelement <4 x float> %1562, float %1557, i64 1, !dbg !11
  %1564 = insertelement <4 x float> %1563, float %1548, i64 2, !dbg !11
  %1565 = insertelement <4 x float> %1564, float %1549, i64 3, !dbg !11
  %1566 = select <4 x i1> %1314, <4 x float> %1561, <4 x float> %1565, !dbg !11
  %1567 = extractelement <4 x float> %1178, i64 0, !dbg !11
  %1568 = extractelement <4 x float> %1178, i64 1, !dbg !11
  %1569 = extractelement <4 x float> %1178, i64 2, !dbg !11
  %1570 = extractelement <4 x float> %1178, i64 3, !dbg !11
  %1571 = select i1 %.not36, float %1568, float %1567, !dbg !11
  %1572 = bitcast float %1571 to i32, !dbg !11
  %1573 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %673, i32 %1572), !dbg !11
  %1574 = bitcast i32 %1573 to float, !dbg !11
  %1575 = select i1 %.not36, float %1567, float %1574, !dbg !11
  %1576 = select i1 %.not36, float %1574, float %1568, !dbg !11
  %1577 = select i1 %.not36, float %1570, float %1569, !dbg !11
  %1578 = bitcast float %1577 to i32, !dbg !11
  %1579 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %673, i32 %1578), !dbg !11
  %1580 = bitcast i32 %1579 to float, !dbg !11
  %1581 = select i1 %.not36, float %1569, float %1580, !dbg !11
  %1582 = select i1 %.not36, float %1580, float %1570, !dbg !11
  %1583 = select i1 %.not37, float %1581, float %1575, !dbg !11
  %1584 = bitcast float %1583 to i32, !dbg !11
  %1585 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %681, i32 %1584), !dbg !11
  %1586 = bitcast i32 %1585 to float, !dbg !11
  %1587 = select i1 %.not37, float %1582, float %1576, !dbg !11
  %1588 = bitcast float %1587 to i32, !dbg !11
  %1589 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %681, i32 %1588), !dbg !11
  %1590 = bitcast i32 %1589 to float, !dbg !11
  %1591 = insertelement <4 x float> poison, float %1575, i64 0, !dbg !11
  %1592 = insertelement <4 x float> %1591, float %1576, i64 1, !dbg !11
  %1593 = insertelement <4 x float> %1592, float %1586, i64 2, !dbg !11
  %1594 = insertelement <4 x float> %1593, float %1590, i64 3, !dbg !11
  %1595 = insertelement <4 x float> poison, float %1586, i64 0, !dbg !11
  %1596 = insertelement <4 x float> %1595, float %1590, i64 1, !dbg !11
  %1597 = insertelement <4 x float> %1596, float %1581, i64 2, !dbg !11
  %1598 = insertelement <4 x float> %1597, float %1582, i64 3, !dbg !11
  %1599 = select <4 x i1> %1314, <4 x float> %1594, <4 x float> %1598, !dbg !11
  %1600 = extractelement <4 x float> %1102, i64 0, !dbg !11
  %1601 = extractelement <4 x float> %1102, i64 1, !dbg !11
  %1602 = extractelement <4 x float> %1102, i64 2, !dbg !11
  %1603 = extractelement <4 x float> %1102, i64 3, !dbg !11
  %1604 = select i1 %.not36, float %1601, float %1600, !dbg !11
  %1605 = bitcast float %1604 to i32, !dbg !11
  %1606 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %673, i32 %1605), !dbg !11
  %1607 = bitcast i32 %1606 to float, !dbg !11
  %1608 = select i1 %.not36, float %1600, float %1607, !dbg !11
  %1609 = select i1 %.not36, float %1607, float %1601, !dbg !11
  %1610 = select i1 %.not36, float %1603, float %1602, !dbg !11
  %1611 = bitcast float %1610 to i32, !dbg !11
  %1612 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %673, i32 %1611), !dbg !11
  %1613 = bitcast i32 %1612 to float, !dbg !11
  %1614 = select i1 %.not36, float %1602, float %1613, !dbg !11
  %1615 = select i1 %.not36, float %1613, float %1603, !dbg !11
  %1616 = select i1 %.not37, float %1614, float %1608, !dbg !11
  %1617 = bitcast float %1616 to i32, !dbg !11
  %1618 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %681, i32 %1617), !dbg !11
  %1619 = bitcast i32 %1618 to float, !dbg !11
  %1620 = select i1 %.not37, float %1615, float %1609, !dbg !11
  %1621 = bitcast float %1620 to i32, !dbg !11
  %1622 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %681, i32 %1621), !dbg !11
  %1623 = bitcast i32 %1622 to float, !dbg !11
  %1624 = insertelement <4 x float> poison, float %1608, i64 0, !dbg !11
  %1625 = insertelement <4 x float> %1624, float %1609, i64 1, !dbg !11
  %1626 = insertelement <4 x float> %1625, float %1619, i64 2, !dbg !11
  %1627 = insertelement <4 x float> %1626, float %1623, i64 3, !dbg !11
  %1628 = insertelement <4 x float> poison, float %1619, i64 0, !dbg !11
  %1629 = insertelement <4 x float> %1628, float %1623, i64 1, !dbg !11
  %1630 = insertelement <4 x float> %1629, float %1614, i64 2, !dbg !11
  %1631 = insertelement <4 x float> %1630, float %1615, i64 3, !dbg !11
  %1632 = select <4 x i1> %1314, <4 x float> %1627, <4 x float> %1631, !dbg !11
  %1633 = extractelement <4 x float> %1026, i64 0, !dbg !11
  %1634 = extractelement <4 x float> %1026, i64 1, !dbg !11
  %1635 = extractelement <4 x float> %1026, i64 2, !dbg !11
  %1636 = extractelement <4 x float> %1026, i64 3, !dbg !11
  %1637 = select i1 %.not36, float %1634, float %1633, !dbg !11
  %1638 = bitcast float %1637 to i32, !dbg !11
  %1639 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %673, i32 %1638), !dbg !11
  %1640 = bitcast i32 %1639 to float, !dbg !11
  %1641 = select i1 %.not36, float %1633, float %1640, !dbg !11
  %1642 = select i1 %.not36, float %1640, float %1634, !dbg !11
  %1643 = select i1 %.not36, float %1636, float %1635, !dbg !11
  %1644 = bitcast float %1643 to i32, !dbg !11
  %1645 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %673, i32 %1644), !dbg !11
  %1646 = bitcast i32 %1645 to float, !dbg !11
  %1647 = select i1 %.not36, float %1635, float %1646, !dbg !11
  %1648 = select i1 %.not36, float %1646, float %1636, !dbg !11
  %1649 = select i1 %.not37, float %1647, float %1641, !dbg !11
  %1650 = bitcast float %1649 to i32, !dbg !11
  %1651 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %681, i32 %1650), !dbg !11
  %1652 = select i1 %.not37, float %1648, float %1642, !dbg !11
  %1653 = bitcast float %1652 to i32, !dbg !11
  %1654 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %681, i32 %1653), !dbg !11
  %1655 = shufflevector <1 x float> %1281, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !54
  %1656 = extractelement <4 x float> %1253, i64 0, !dbg !11
  %1657 = extractelement <4 x float> %1253, i64 1, !dbg !11
  %1658 = extractelement <4 x float> %1253, i64 2, !dbg !11
  %1659 = extractelement <4 x float> %1253, i64 3, !dbg !11
  %1660 = select i1 %.not36, float %1657, float %1656, !dbg !11
  %1661 = bitcast float %1660 to i32, !dbg !11
  %1662 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %673, i32 %1661), !dbg !11
  %1663 = bitcast i32 %1662 to float, !dbg !11
  %1664 = select i1 %.not36, float %1656, float %1663, !dbg !11
  %1665 = select i1 %.not36, float %1663, float %1657, !dbg !11
  %1666 = select i1 %.not36, float %1659, float %1658, !dbg !11
  %1667 = bitcast float %1666 to i32, !dbg !11
  %1668 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %673, i32 %1667), !dbg !11
  %1669 = bitcast i32 %1668 to float, !dbg !11
  %1670 = select i1 %.not36, float %1658, float %1669, !dbg !11
  %1671 = select i1 %.not36, float %1669, float %1659, !dbg !11
  %1672 = select i1 %.not37, float %1670, float %1664, !dbg !11
  %1673 = bitcast float %1672 to i32, !dbg !11
  %1674 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %681, i32 %1673), !dbg !11
  %1675 = bitcast i32 %1674 to float, !dbg !11
  %1676 = select i1 %.not37, float %1671, float %1665, !dbg !11
  %1677 = bitcast float %1676 to i32, !dbg !11
  %1678 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %681, i32 %1677), !dbg !11
  %1679 = bitcast i32 %1678 to float, !dbg !11
  %1680 = insertelement <4 x float> poison, float %1664, i64 0, !dbg !11
  %1681 = insertelement <4 x float> %1680, float %1665, i64 1, !dbg !11
  %1682 = insertelement <4 x float> %1681, float %1675, i64 2, !dbg !11
  %1683 = insertelement <4 x float> %1682, float %1679, i64 3, !dbg !11
  %1684 = insertelement <4 x float> poison, float %1675, i64 0, !dbg !11
  %1685 = insertelement <4 x float> %1684, float %1679, i64 1, !dbg !11
  %1686 = insertelement <4 x float> %1685, float %1670, i64 2, !dbg !11
  %1687 = insertelement <4 x float> %1686, float %1671, i64 3, !dbg !11
  %1688 = select <4 x i1> %1314, <4 x float> %1683, <4 x float> %1687, !dbg !11
  %1689 = extractelement <4 x float> %1179, i64 0, !dbg !11
  %1690 = extractelement <4 x float> %1179, i64 1, !dbg !11
  %1691 = extractelement <4 x float> %1179, i64 2, !dbg !11
  %1692 = extractelement <4 x float> %1179, i64 3, !dbg !11
  %1693 = select i1 %.not36, float %1690, float %1689, !dbg !11
  %1694 = bitcast float %1693 to i32, !dbg !11
  %1695 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %673, i32 %1694), !dbg !11
  %1696 = bitcast i32 %1695 to float, !dbg !11
  %1697 = select i1 %.not36, float %1689, float %1696, !dbg !11
  %1698 = select i1 %.not36, float %1696, float %1690, !dbg !11
  %1699 = select i1 %.not36, float %1692, float %1691, !dbg !11
  %1700 = bitcast float %1699 to i32, !dbg !11
  %1701 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %673, i32 %1700), !dbg !11
  %1702 = bitcast i32 %1701 to float, !dbg !11
  %1703 = select i1 %.not36, float %1691, float %1702, !dbg !11
  %1704 = select i1 %.not36, float %1702, float %1692, !dbg !11
  %1705 = select i1 %.not37, float %1703, float %1697, !dbg !11
  %1706 = bitcast float %1705 to i32, !dbg !11
  %1707 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %681, i32 %1706), !dbg !11
  %1708 = bitcast i32 %1707 to float, !dbg !11
  %1709 = select i1 %.not37, float %1704, float %1698, !dbg !11
  %1710 = bitcast float %1709 to i32, !dbg !11
  %1711 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %681, i32 %1710), !dbg !11
  %1712 = bitcast i32 %1711 to float, !dbg !11
  %1713 = insertelement <4 x float> poison, float %1697, i64 0, !dbg !11
  %1714 = insertelement <4 x float> %1713, float %1698, i64 1, !dbg !11
  %1715 = insertelement <4 x float> %1714, float %1708, i64 2, !dbg !11
  %1716 = insertelement <4 x float> %1715, float %1712, i64 3, !dbg !11
  %1717 = insertelement <4 x float> poison, float %1708, i64 0, !dbg !11
  %1718 = insertelement <4 x float> %1717, float %1712, i64 1, !dbg !11
  %1719 = insertelement <4 x float> %1718, float %1703, i64 2, !dbg !11
  %1720 = insertelement <4 x float> %1719, float %1704, i64 3, !dbg !11
  %1721 = select <4 x i1> %1314, <4 x float> %1716, <4 x float> %1720, !dbg !11
  %1722 = extractelement <4 x float> %1103, i64 0, !dbg !11
  %1723 = extractelement <4 x float> %1103, i64 1, !dbg !11
  %1724 = extractelement <4 x float> %1103, i64 2, !dbg !11
  %1725 = extractelement <4 x float> %1103, i64 3, !dbg !11
  %1726 = select i1 %.not36, float %1723, float %1722, !dbg !11
  %1727 = bitcast float %1726 to i32, !dbg !11
  %1728 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %673, i32 %1727), !dbg !11
  %1729 = bitcast i32 %1728 to float, !dbg !11
  %1730 = select i1 %.not36, float %1722, float %1729, !dbg !11
  %1731 = select i1 %.not36, float %1729, float %1723, !dbg !11
  %1732 = select i1 %.not36, float %1725, float %1724, !dbg !11
  %1733 = bitcast float %1732 to i32, !dbg !11
  %1734 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %673, i32 %1733), !dbg !11
  %1735 = bitcast i32 %1734 to float, !dbg !11
  %1736 = select i1 %.not36, float %1724, float %1735, !dbg !11
  %1737 = select i1 %.not36, float %1735, float %1725, !dbg !11
  %1738 = select i1 %.not37, float %1736, float %1730, !dbg !11
  %1739 = bitcast float %1738 to i32, !dbg !11
  %1740 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %681, i32 %1739), !dbg !11
  %1741 = bitcast i32 %1740 to float, !dbg !11
  %1742 = select i1 %.not37, float %1737, float %1731, !dbg !11
  %1743 = bitcast float %1742 to i32, !dbg !11
  %1744 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %681, i32 %1743), !dbg !11
  %1745 = bitcast i32 %1744 to float, !dbg !11
  %1746 = insertelement <4 x float> poison, float %1730, i64 0, !dbg !11
  %1747 = insertelement <4 x float> %1746, float %1731, i64 1, !dbg !11
  %1748 = insertelement <4 x float> %1747, float %1741, i64 2, !dbg !11
  %1749 = insertelement <4 x float> %1748, float %1745, i64 3, !dbg !11
  %1750 = insertelement <4 x float> poison, float %1741, i64 0, !dbg !11
  %1751 = insertelement <4 x float> %1750, float %1745, i64 1, !dbg !11
  %1752 = insertelement <4 x float> %1751, float %1736, i64 2, !dbg !11
  %1753 = insertelement <4 x float> %1752, float %1737, i64 3, !dbg !11
  %1754 = select <4 x i1> %1314, <4 x float> %1749, <4 x float> %1753, !dbg !11
  %1755 = extractelement <4 x float> %1027, i64 0, !dbg !11
  %1756 = extractelement <4 x float> %1027, i64 1, !dbg !11
  %1757 = extractelement <4 x float> %1027, i64 2, !dbg !11
  %1758 = extractelement <4 x float> %1027, i64 3, !dbg !11
  %1759 = select i1 %.not36, float %1756, float %1755, !dbg !11
  %1760 = bitcast float %1759 to i32, !dbg !11
  %1761 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %673, i32 %1760), !dbg !11
  %1762 = bitcast i32 %1761 to float, !dbg !11
  %1763 = select i1 %.not36, float %1755, float %1762, !dbg !11
  %1764 = select i1 %.not36, float %1762, float %1756, !dbg !11
  %1765 = select i1 %.not36, float %1758, float %1757, !dbg !11
  %1766 = bitcast float %1765 to i32, !dbg !11
  %1767 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %673, i32 %1766), !dbg !11
  %1768 = bitcast i32 %1767 to float, !dbg !11
  %1769 = select i1 %.not36, float %1757, float %1768, !dbg !11
  %1770 = select i1 %.not36, float %1768, float %1758, !dbg !11
  %1771 = select i1 %.not37, float %1769, float %1763, !dbg !11
  %1772 = bitcast float %1771 to i32, !dbg !11
  %1773 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %681, i32 %1772), !dbg !11
  %1774 = select i1 %.not37, float %1770, float %1764, !dbg !11
  %1775 = bitcast float %1774 to i32, !dbg !11
  %1776 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %681, i32 %1775), !dbg !11
  %1777 = bitcast i32 %1773 to float, !dbg !11
  %1778 = bitcast i32 %1776 to float, !dbg !11
  %1779 = insertelement <4 x float> poison, float %1763, i64 0, !dbg !11
  %1780 = insertelement <4 x float> %1779, float %1764, i64 1, !dbg !11
  %1781 = insertelement <4 x float> %1780, float %1777, i64 2, !dbg !11
  %1782 = insertelement <4 x float> %1781, float %1778, i64 3, !dbg !11
  %1783 = insertelement <4 x float> poison, float %1777, i64 0, !dbg !11
  %1784 = insertelement <4 x float> %1783, float %1778, i64 1, !dbg !11
  %1785 = insertelement <4 x float> %1784, float %1769, i64 2, !dbg !11
  %1786 = insertelement <4 x float> %1785, float %1770, i64 3, !dbg !11
  %1787 = select <4 x i1> %1314, <4 x float> %1782, <4 x float> %1786, !dbg !11
  %row.reciprocal = fdiv arcp <4 x float> splat (float 1.000000e+00), %1655, !dbg !54
  %normalized = fmul <4 x float> %row.reciprocal, %1787, !dbg !54
  %1788 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !55
  %1789 = extractelement <4 x half> %1788, i64 3, !dbg !55
  %1790 = extractelement <4 x half> %1788, i64 2, !dbg !55
  %1791 = extractelement <4 x half> %1788, i64 1, !dbg !55
  %1792 = extractelement <4 x half> %1788, i64 0, !dbg !55
  %1793 = bitcast i32 %1651 to float, !dbg !11
  %1794 = bitcast i32 %1654 to float, !dbg !11
  %1795 = insertelement <4 x float> poison, float %1641, i64 0, !dbg !11
  %1796 = insertelement <4 x float> %1795, float %1642, i64 1, !dbg !11
  %1797 = insertelement <4 x float> %1796, float %1793, i64 2, !dbg !11
  %1798 = insertelement <4 x float> %1797, float %1794, i64 3, !dbg !11
  %1799 = insertelement <4 x float> poison, float %1793, i64 0, !dbg !11
  %1800 = insertelement <4 x float> %1799, float %1794, i64 1, !dbg !11
  %1801 = insertelement <4 x float> %1800, float %1647, i64 2, !dbg !11
  %1802 = insertelement <4 x float> %1801, float %1648, i64 3, !dbg !11
  %1803 = select <4 x i1> %1314, <4 x float> %1798, <4 x float> %1802, !dbg !11
  %normalized105 = fmul <4 x float> %row.reciprocal, %1803, !dbg !54
  %1804 = fptrunc <4 x float> %normalized105 to <4 x half>, !dbg !55
  %1805 = extractelement <4 x half> %1804, i64 3, !dbg !55
  %1806 = extractelement <4 x half> %1804, i64 2, !dbg !55
  %1807 = extractelement <4 x half> %1804, i64 1, !dbg !55
  %1808 = extractelement <4 x half> %1804, i64 0, !dbg !55
  %1809 = bitcast i32 %1530 to float, !dbg !11
  %1810 = bitcast i32 %1533 to float, !dbg !11
  %1811 = insertelement <4 x float> poison, float %1520, i64 0, !dbg !11
  %1812 = insertelement <4 x float> %1811, float %1521, i64 1, !dbg !11
  %1813 = insertelement <4 x float> %1812, float %1809, i64 2, !dbg !11
  %1814 = insertelement <4 x float> %1813, float %1810, i64 3, !dbg !11
  %1815 = insertelement <4 x float> poison, float %1809, i64 0, !dbg !11
  %1816 = insertelement <4 x float> %1815, float %1810, i64 1, !dbg !11
  %1817 = insertelement <4 x float> %1816, float %1526, i64 2, !dbg !11
  %1818 = insertelement <4 x float> %1817, float %1527, i64 3, !dbg !11
  %1819 = select <4 x i1> %1314, <4 x float> %1814, <4 x float> %1818, !dbg !11
  %row.reciprocal106 = fdiv arcp <4 x float> splat (float 1.000000e+00), %1412, !dbg !54
  %normalized107 = fmul <4 x float> %row.reciprocal106, %1819, !dbg !54
  %1820 = fptrunc <4 x float> %normalized107 to <4 x half>, !dbg !55
  %1821 = extractelement <4 x half> %1820, i64 3, !dbg !55
  %1822 = extractelement <4 x half> %1820, i64 2, !dbg !55
  %1823 = extractelement <4 x half> %1820, i64 1, !dbg !55
  %1824 = extractelement <4 x half> %1820, i64 0, !dbg !55
  %1825 = bitcast i32 %1408 to float, !dbg !11
  %1826 = bitcast i32 %1411 to float, !dbg !11
  %1827 = insertelement <4 x float> poison, float %1398, i64 0, !dbg !11
  %1828 = insertelement <4 x float> %1827, float %1399, i64 1, !dbg !11
  %1829 = insertelement <4 x float> %1828, float %1825, i64 2, !dbg !11
  %1830 = insertelement <4 x float> %1829, float %1826, i64 3, !dbg !11
  %1831 = insertelement <4 x float> poison, float %1825, i64 0, !dbg !11
  %1832 = insertelement <4 x float> %1831, float %1826, i64 1, !dbg !11
  %1833 = insertelement <4 x float> %1832, float %1404, i64 2, !dbg !11
  %1834 = insertelement <4 x float> %1833, float %1405, i64 3, !dbg !11
  %1835 = select <4 x i1> %1314, <4 x float> %1830, <4 x float> %1834, !dbg !11
  %normalized108 = fmul <4 x float> %row.reciprocal106, %1835, !dbg !54
  %1836 = fptrunc <4 x float> %normalized108 to <4 x half>, !dbg !55
  %1837 = extractelement <4 x half> %1836, i64 3, !dbg !55
  %1838 = extractelement <4 x half> %1836, i64 2, !dbg !55
  %1839 = extractelement <4 x half> %1836, i64 1, !dbg !55
  %1840 = extractelement <4 x half> %1836, i64 0, !dbg !55
  %address.thread1042 = tail call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #8, !dbg !56, !range !9, !frisk.register_lifetimes !5
  %address.local1043 = zext nneg i32 %address.thread1042 to i64, !dbg !8
  %address.local1045 = lshr i64 %address.local1043, 2, !dbg !14
  %address.local1046 = and i64 %address.local1045, 48, !dbg !14
  %address.local1048 = and i64 %address.local1043, 240, !dbg !56
  %address.local1047 = add i64 %22, %address.local1043, !dbg !56
  %address.local1050 = sub i64 %address.local1047, %address.local1048, !dbg !56
  %address.local1051 = add i64 %address.local1050, %address.local1046, !dbg !56
  %1841 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %24, !dbg !56
  %address.local1054.idx = shl nuw nsw i64 %address.local1051, 8, !dbg !56
  %address.local1054 = getelementptr inbounds nuw i8, ptr addrspace(1) %1841, i64 %address.local1054.idx, !dbg !56
  %address.local1056 = and i64 %address.local1045, 12, !dbg !10
  %address.local1057 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1056, !dbg !56
  store half %1840, ptr addrspace(1) %address.local1057, align 2, !dbg !56
  %address.local1059 = or disjoint i64 %address.local1056, 1, !dbg !56
  %address.local1060 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1059, !dbg !56
  store half %1839, ptr addrspace(1) %address.local1060, align 2, !dbg !56
  %address.local1062 = or disjoint i64 %address.local1056, 2, !dbg !56
  %address.local1063 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1062, !dbg !56
  store half %1838, ptr addrspace(1) %address.local1063, align 2, !dbg !56
  %address.local1065 = or disjoint i64 %address.local1056, 3, !dbg !56
  %address.local1066 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1065, !dbg !56
  store half %1837, ptr addrspace(1) %address.local1066, align 2, !dbg !56
  %address.local1068 = or disjoint i64 %address.local1056, 16, !dbg !56
  %address.local1069 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1068, !dbg !56
  store half %1824, ptr addrspace(1) %address.local1069, align 2, !dbg !56
  %address.local1071 = or disjoint i64 %address.local1056, 17, !dbg !56
  %address.local1072 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1071, !dbg !56
  store half %1823, ptr addrspace(1) %address.local1072, align 2, !dbg !56
  %address.local1074 = or disjoint i64 %address.local1056, 18, !dbg !56
  %address.local1075 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1074, !dbg !56
  store half %1822, ptr addrspace(1) %address.local1075, align 2, !dbg !56
  %address.local1077 = or disjoint i64 %address.local1056, 19, !dbg !56
  %address.local1078 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1077, !dbg !56
  store half %1821, ptr addrspace(1) %address.local1078, align 2, !dbg !56
  %address.local1082 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local1054, i64 16384, !dbg !56
  %address.local1083 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1056, !dbg !56
  store half %1808, ptr addrspace(1) %address.local1083, align 2, !dbg !56
  %address.local1085 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1059, !dbg !56
  store half %1807, ptr addrspace(1) %address.local1085, align 2, !dbg !56
  %address.local1087 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1062, !dbg !56
  store half %1806, ptr addrspace(1) %address.local1087, align 2, !dbg !56
  %address.local1089 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1065, !dbg !56
  store half %1805, ptr addrspace(1) %address.local1089, align 2, !dbg !56
  %address.local1091 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1068, !dbg !56
  store half %1792, ptr addrspace(1) %address.local1091, align 2, !dbg !56
  %address.local1093 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1071, !dbg !56
  store half %1791, ptr addrspace(1) %address.local1093, align 2, !dbg !56
  %address.local1095 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1074, !dbg !56
  store half %1790, ptr addrspace(1) %address.local1095, align 2, !dbg !56
  %address.local1097 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1077, !dbg !56
  store half %1789, ptr addrspace(1) %address.local1097, align 2, !dbg !56
  %normalized110 = fmul <4 x float> %row.reciprocal, %1754, !dbg !54
  %1842 = fptrunc <4 x float> %normalized110 to <4 x half>, !dbg !55
  %1843 = extractelement <4 x half> %1842, i64 3, !dbg !55
  %1844 = extractelement <4 x half> %1842, i64 2, !dbg !55
  %1845 = extractelement <4 x half> %1842, i64 1, !dbg !55
  %1846 = extractelement <4 x half> %1842, i64 0, !dbg !55
  %normalized111 = fmul <4 x float> %row.reciprocal, %1632, !dbg !54
  %1847 = fptrunc <4 x float> %normalized111 to <4 x half>, !dbg !55
  %1848 = extractelement <4 x half> %1847, i64 3, !dbg !55
  %1849 = extractelement <4 x half> %1847, i64 2, !dbg !55
  %1850 = extractelement <4 x half> %1847, i64 1, !dbg !55
  %1851 = extractelement <4 x half> %1847, i64 0, !dbg !55
  %normalized113 = fmul <4 x float> %row.reciprocal106, %1511, !dbg !54
  %1852 = fptrunc <4 x float> %normalized113 to <4 x half>, !dbg !55
  %1853 = extractelement <4 x half> %1852, i64 3, !dbg !55
  %1854 = extractelement <4 x half> %1852, i64 2, !dbg !55
  %1855 = extractelement <4 x half> %1852, i64 1, !dbg !55
  %1856 = extractelement <4 x half> %1852, i64 0, !dbg !55
  %normalized114 = fmul <4 x float> %row.reciprocal106, %1389, !dbg !54
  %1857 = fptrunc <4 x float> %normalized114 to <4 x half>, !dbg !55
  %1858 = extractelement <4 x half> %1857, i64 3, !dbg !55
  %1859 = extractelement <4 x half> %1857, i64 2, !dbg !55
  %1860 = extractelement <4 x half> %1857, i64 1, !dbg !55
  %1861 = extractelement <4 x half> %1857, i64 0, !dbg !55
  %address.local1113 = or disjoint i64 %address.local1056, 32, !dbg !56
  %address.local1114 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1113, !dbg !56
  store half %1861, ptr addrspace(1) %address.local1114, align 2, !dbg !56
  %address.local1116 = or disjoint i64 %address.local1056, 33, !dbg !56
  %address.local1117 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1116, !dbg !56
  store half %1860, ptr addrspace(1) %address.local1117, align 2, !dbg !56
  %address.local1119 = or disjoint i64 %address.local1056, 34, !dbg !56
  %address.local1120 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1119, !dbg !56
  store half %1859, ptr addrspace(1) %address.local1120, align 2, !dbg !56
  %address.local1122 = or disjoint i64 %address.local1056, 35, !dbg !56
  %address.local1123 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1122, !dbg !56
  store half %1858, ptr addrspace(1) %address.local1123, align 2, !dbg !56
  %address.local1125 = or disjoint i64 %address.local1056, 48, !dbg !56
  %address.local1126 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1125, !dbg !56
  store half %1856, ptr addrspace(1) %address.local1126, align 2, !dbg !56
  %address.local1128 = or disjoint i64 %address.local1056, 49, !dbg !56
  %address.local1129 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1128, !dbg !56
  store half %1855, ptr addrspace(1) %address.local1129, align 2, !dbg !56
  %address.local1131 = or disjoint i64 %address.local1056, 50, !dbg !56
  %address.local1132 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1131, !dbg !56
  store half %1854, ptr addrspace(1) %address.local1132, align 2, !dbg !56
  %address.local1134 = or i64 %address.local1045, 51, !dbg !56
  %address.local1135 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1134, !dbg !56
  store half %1853, ptr addrspace(1) %address.local1135, align 2, !dbg !56
  %address.local1140 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1113, !dbg !56
  store half %1851, ptr addrspace(1) %address.local1140, align 2, !dbg !56
  %address.local1142 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1116, !dbg !56
  store half %1850, ptr addrspace(1) %address.local1142, align 2, !dbg !56
  %address.local1144 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1119, !dbg !56
  store half %1849, ptr addrspace(1) %address.local1144, align 2, !dbg !56
  %address.local1146 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1122, !dbg !56
  store half %1848, ptr addrspace(1) %address.local1146, align 2, !dbg !56
  %address.local1148 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1125, !dbg !56
  store half %1846, ptr addrspace(1) %address.local1148, align 2, !dbg !56
  %address.local1150 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1128, !dbg !56
  store half %1845, ptr addrspace(1) %address.local1150, align 2, !dbg !56
  %address.local1152 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1131, !dbg !56
  store half %1844, ptr addrspace(1) %address.local1152, align 2, !dbg !56
  %address.local1154 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1134, !dbg !56
  store half %1843, ptr addrspace(1) %address.local1154, align 2, !dbg !56
  %normalized116 = fmul <4 x float> %row.reciprocal, %1721, !dbg !54
  %1862 = fptrunc <4 x float> %normalized116 to <4 x half>, !dbg !55
  %1863 = extractelement <4 x half> %1862, i64 3, !dbg !55
  %1864 = extractelement <4 x half> %1862, i64 2, !dbg !55
  %1865 = extractelement <4 x half> %1862, i64 1, !dbg !55
  %1866 = extractelement <4 x half> %1862, i64 0, !dbg !55
  %normalized117 = fmul <4 x float> %row.reciprocal, %1599, !dbg !54
  %1867 = fptrunc <4 x float> %normalized117 to <4 x half>, !dbg !55
  %1868 = extractelement <4 x half> %1867, i64 3, !dbg !55
  %1869 = extractelement <4 x half> %1867, i64 2, !dbg !55
  %1870 = extractelement <4 x half> %1867, i64 1, !dbg !55
  %1871 = extractelement <4 x half> %1867, i64 0, !dbg !55
  %normalized119 = fmul <4 x float> %row.reciprocal106, %1478, !dbg !54
  %1872 = fptrunc <4 x float> %normalized119 to <4 x half>, !dbg !55
  %1873 = extractelement <4 x half> %1872, i64 3, !dbg !55
  %1874 = extractelement <4 x half> %1872, i64 2, !dbg !55
  %1875 = extractelement <4 x half> %1872, i64 1, !dbg !55
  %1876 = extractelement <4 x half> %1872, i64 0, !dbg !55
  %normalized120 = fmul <4 x float> %row.reciprocal106, %1356, !dbg !54
  %1877 = fptrunc <4 x float> %normalized120 to <4 x half>, !dbg !55
  %1878 = extractelement <4 x half> %1877, i64 3, !dbg !55
  %1879 = extractelement <4 x half> %1877, i64 2, !dbg !55
  %1880 = extractelement <4 x half> %1877, i64 1, !dbg !55
  %1881 = extractelement <4 x half> %1877, i64 0, !dbg !55
  %address.local1170 = or disjoint i64 %address.local1056, 64, !dbg !56
  %address.local1171 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1170, !dbg !56
  store half %1881, ptr addrspace(1) %address.local1171, align 2, !dbg !56
  %address.local1173 = or disjoint i64 %address.local1056, 65, !dbg !56
  %address.local1174 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1173, !dbg !56
  store half %1880, ptr addrspace(1) %address.local1174, align 2, !dbg !56
  %address.local1176 = or disjoint i64 %address.local1056, 66, !dbg !56
  %address.local1177 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1176, !dbg !56
  store half %1879, ptr addrspace(1) %address.local1177, align 2, !dbg !56
  %address.local1179 = or disjoint i64 %address.local1056, 67, !dbg !56
  %address.local1180 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1179, !dbg !56
  store half %1878, ptr addrspace(1) %address.local1180, align 2, !dbg !56
  %address.local1182 = or disjoint i64 %address.local1056, 80, !dbg !56
  %address.local1183 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1182, !dbg !56
  store half %1876, ptr addrspace(1) %address.local1183, align 2, !dbg !56
  %address.local1185 = or disjoint i64 %address.local1056, 81, !dbg !56
  %address.local1186 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1185, !dbg !56
  store half %1875, ptr addrspace(1) %address.local1186, align 2, !dbg !56
  %address.local1188 = or disjoint i64 %address.local1056, 82, !dbg !56
  %address.local1189 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1188, !dbg !56
  store half %1874, ptr addrspace(1) %address.local1189, align 2, !dbg !56
  %address.local1191 = or disjoint i64 %address.local1056, 83, !dbg !56
  %address.local1192 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1191, !dbg !56
  store half %1873, ptr addrspace(1) %address.local1192, align 2, !dbg !56
  %address.local1197 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1170, !dbg !56
  store half %1871, ptr addrspace(1) %address.local1197, align 2, !dbg !56
  %address.local1199 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1173, !dbg !56
  store half %1870, ptr addrspace(1) %address.local1199, align 2, !dbg !56
  %address.local1201 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1176, !dbg !56
  store half %1869, ptr addrspace(1) %address.local1201, align 2, !dbg !56
  %address.local1203 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1179, !dbg !56
  store half %1868, ptr addrspace(1) %address.local1203, align 2, !dbg !56
  %address.local1205 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1182, !dbg !56
  store half %1866, ptr addrspace(1) %address.local1205, align 2, !dbg !56
  %address.local1207 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1185, !dbg !56
  store half %1865, ptr addrspace(1) %address.local1207, align 2, !dbg !56
  %address.local1209 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1188, !dbg !56
  store half %1864, ptr addrspace(1) %address.local1209, align 2, !dbg !56
  %address.local1211 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1191, !dbg !56
  store half %1863, ptr addrspace(1) %address.local1211, align 2, !dbg !56
  %normalized122 = fmul <4 x float> %row.reciprocal, %1688, !dbg !54
  %1882 = fptrunc <4 x float> %normalized122 to <4 x half>, !dbg !55
  %1883 = extractelement <4 x half> %1882, i64 3, !dbg !55
  %1884 = extractelement <4 x half> %1882, i64 2, !dbg !55
  %1885 = extractelement <4 x half> %1882, i64 1, !dbg !55
  %1886 = extractelement <4 x half> %1882, i64 0, !dbg !55
  %normalized123 = fmul <4 x float> %row.reciprocal, %1566, !dbg !54
  %1887 = fptrunc <4 x float> %normalized123 to <4 x half>, !dbg !55
  %1888 = extractelement <4 x half> %1887, i64 3, !dbg !55
  %1889 = extractelement <4 x half> %1887, i64 2, !dbg !55
  %1890 = extractelement <4 x half> %1887, i64 1, !dbg !55
  %1891 = extractelement <4 x half> %1887, i64 0, !dbg !55
  %normalized125 = fmul <4 x float> %row.reciprocal106, %1445, !dbg !54
  %1892 = fptrunc <4 x float> %normalized125 to <4 x half>, !dbg !55
  %1893 = extractelement <4 x half> %1892, i64 3, !dbg !55
  %1894 = extractelement <4 x half> %1892, i64 2, !dbg !55
  %1895 = extractelement <4 x half> %1892, i64 1, !dbg !55
  %1896 = extractelement <4 x half> %1892, i64 0, !dbg !55
  %normalized126 = fmul <4 x float> %row.reciprocal106, %1323, !dbg !54
  %1897 = fptrunc <4 x float> %normalized126 to <4 x half>, !dbg !55
  %1898 = extractelement <4 x half> %1897, i64 3, !dbg !55
  %1899 = extractelement <4 x half> %1897, i64 2, !dbg !55
  %1900 = extractelement <4 x half> %1897, i64 1, !dbg !55
  %1901 = extractelement <4 x half> %1897, i64 0, !dbg !55
  %address.local1227 = or disjoint i64 %address.local1056, 96, !dbg !56
  %address.local1228 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1227, !dbg !56
  store half %1901, ptr addrspace(1) %address.local1228, align 2, !dbg !56
  %address.local1230 = or disjoint i64 %address.local1056, 97, !dbg !56
  %address.local1231 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1230, !dbg !56
  store half %1900, ptr addrspace(1) %address.local1231, align 2, !dbg !56
  %address.local1233 = or disjoint i64 %address.local1056, 98, !dbg !56
  %address.local1234 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1233, !dbg !56
  store half %1899, ptr addrspace(1) %address.local1234, align 2, !dbg !56
  %address.local1236 = or disjoint i64 %address.local1056, 99, !dbg !56
  %address.local1237 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1236, !dbg !56
  store half %1898, ptr addrspace(1) %address.local1237, align 2, !dbg !56
  %address.local1239 = or disjoint i64 %address.local1056, 112, !dbg !56
  %address.local1240 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1239, !dbg !56
  store half %1896, ptr addrspace(1) %address.local1240, align 2, !dbg !56
  %address.local1242 = or disjoint i64 %address.local1056, 113, !dbg !56
  %address.local1243 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1242, !dbg !56
  store half %1895, ptr addrspace(1) %address.local1243, align 2, !dbg !56
  %address.local1245 = or disjoint i64 %address.local1056, 114, !dbg !56
  %address.local1246 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1245, !dbg !56
  store half %1894, ptr addrspace(1) %address.local1246, align 2, !dbg !56
  %address.local1248 = or i64 %address.local1045, 115, !dbg !56
  %address.local1249 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1054, i64 %address.local1248, !dbg !56
  store half %1893, ptr addrspace(1) %address.local1249, align 2, !dbg !56
  %address.local1254 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1227, !dbg !56
  store half %1891, ptr addrspace(1) %address.local1254, align 2, !dbg !56
  %address.local1256 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1230, !dbg !56
  store half %1890, ptr addrspace(1) %address.local1256, align 2, !dbg !56
  %address.local1258 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1233, !dbg !56
  store half %1889, ptr addrspace(1) %address.local1258, align 2, !dbg !56
  %address.local1260 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1236, !dbg !56
  store half %1888, ptr addrspace(1) %address.local1260, align 2, !dbg !56
  %address.local1262 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1239, !dbg !56
  store half %1886, ptr addrspace(1) %address.local1262, align 2, !dbg !56
  %address.local1264 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1242, !dbg !56
  store half %1885, ptr addrspace(1) %address.local1264, align 2, !dbg !56
  %address.local1266 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1245, !dbg !56
  store half %1884, ptr addrspace(1) %address.local1266, align 2, !dbg !56
  %address.local1268 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1248, !dbg !56
  store half %1883, ptr addrspace(1) %address.local1268, align 2, !dbg !56
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
attributes #1 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1" "frisk.global_prefetch"="8" "frisk.k_dma"="64-swizzle:1" "frisk.k_lds_prefetch"="8" "frisk.k_tail_prefetch"="4" "frisk.register_comparisons"="1" "frisk.register_lifetimes"="1" "frisk.softmax_fragments"="pv" "uniform-work-group-size" }
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
